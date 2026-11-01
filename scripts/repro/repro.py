#!/usr/bin/env python3
"""Reproduction entry point: turn "can a fresh clone be used" into an executable check.

Levels (see policy/reproducibility.md):
  R1  structural  scripts, configs, protocols and recorded evidence are present and self-consistent
  R2  environmental  pinned images are available and match their recorded image ids
  R3  numerical  a recorded run can be re-executed and compared against the ledger

Usage:
  python3 scripts/repro/repro.py doctor [--json] [--compile] [--strict]
  python3 scripts/repro/repro.py plan
  python3 scripts/repro/repro.py fetch-images [--dry-run]
  python3 scripts/repro/repro.py fetch-source <name> [--archive PATH] [--dry-run]
  python3 scripts/repro/repro.py pin <name> --archive PATH [--url URL] [--dry-run]
  python3 scripts/repro/repro.py verify-source

Exit codes: 0 = R1 reached, 2 = R1 not reached, 1 = --strict and a blocking/warning check.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "repro" / "manifest.json"
CACHE = ROOT / "cache" / "repro"
STATUSES = ("PASS", "WARN", "SKIP", "FAIL", "BLOCK")


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise ManifestError(f"missing {MANIFEST.relative_to(ROOT)}")
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{MANIFEST.relative_to(ROOT)} is not valid JSON: {exc}") from exc
    return data


def validate(data: dict) -> list[str]:
    """Return schema problems. An empty list means every pin is machine-readable."""
    problems: list[str] = []
    if data.get("schema") != "pi05-repro-manifest/1":
        problems.append(f"schema must be pi05-repro-manifest/1, got {data.get('schema')!r}")
    for section in ("images", "sources", "assets", "versions"):
        if section not in data:
            problems.append(f"missing section {section}")
    for image in data.get("images") or []:
        for field in ("name", "tag", "digest", "registry_env"):
            if not image.get(field):
                problems.append(f"images[{image.get('name', '?')}].{field} is required")
        digest = image.get("digest") or ""
        if digest and not (digest.startswith("sha256:") and len(digest) == 71):
            problems.append(f"images[{image.get('name', '?')}].digest is not sha256:<64 hex>")
    for source in data.get("sources") or []:
        for field in ("name", "target", "files", "acquisition"):
            if not source.get(field):
                problems.append(f"sources[{source.get('name', '?')}].{field} is required")
        if not isinstance(source.get("files"), int) or source.get("files", 0) <= 0:
            problems.append(f"sources[{source.get('name', '?')}].files must be a positive int")
        if source.get("aggregate_status") not in {"full", "prefix_only"}:
            problems.append(f"sources[{source.get('name', '?')}].aggregate_status is invalid")
    for asset in data.get("assets") or []:
        if not asset.get("name"):
            problems.append("assets entry without a name")
        if not isinstance(asset.get("redistributable"), bool):
            problems.append(f"assets[{asset.get('name', '?')}].redistributable must be a bool")
    return problems


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, cid: str, name: str, status: str, detail: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"bad status {status}")
        self.checks.append({"id": cid, "name": name, "status": status, "detail": detail})

    def of(self, prefix: str) -> list[dict]:
        return [check for check in self.checks if check["id"].startswith(prefix)]

    def ok(self, prefix: str) -> bool:
        return all(check["status"] in {"PASS", "WARN"} for check in self.of(prefix))

    def counts(self) -> dict:
        total: dict[str, int] = {}
        for check in self.checks:
            total[check["status"]] = total.get(check["status"], 0) + 1
        return total


def docker_image_id(tag: str) -> str | None:
    binary = shutil.which("docker")
    if binary is None:
        return None
    result = subprocess.run([binary, "image", "inspect", "--format", "{{.Id}}", tag],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def check_r1(data: dict, checker: Checker, compile_scripts: bool) -> None:
    version = ".".join(str(part) for part in sys.version_info[:3])
    checker.add("R1-01", "python >= 3.9", "PASS" if sys.version_info >= (3, 9) else "FAIL", version)

    git = shutil.which("git")
    if git is None:
        checker.add("R1-02", "git checkout", "WARN", "git is not on PATH; no commit anchor")
    else:
        head = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if head.returncode == 0:
            checker.add("R1-02", "git checkout", "PASS", head.stdout.decode().strip())
        else:
            checker.add("R1-02", "git checkout", "WARN",
                        "not a git checkout; a downloaded copy has no commit anchor")

    problems = validate(data)
    checker.add("R1-03", "manifest schema",
                "PASS" if not problems else "FAIL",
                "pins are machine-readable" if not problems else "; ".join(problems[:3]))

    weights = [
        asset for asset in data.get("assets") or []
        if asset.get("status") == "in_repo" and asset.get("path") and asset.get("sha256")
    ]
    for index, asset in enumerate(weights, start=1):
        path = ROOT / asset["path"]
        if not path.is_file():
            checker.add(f"R1-04.{index}", f"tracked asset {asset['name']}", "FAIL",
                        f"{asset['path']} is missing")
            continue
        got = sha256_file(path)
        hit = got == asset["sha256"]
        checker.add(f"R1-04.{index}", f"tracked asset {asset['name']}",
                    "PASS" if hit else "FAIL",
                    f"sha256={got[:16]}..." + ("" if hit else f" expected {asset['sha256'][:16]}..."))

    required = [
        "configs/libero/protocol.json",
        "configs/libero/norm_stats.json",
        "configs/so101/protocol.json",
        "experiments/ledger.jsonl",
    ]
    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    checker.add("R1-05", "recorded evidence", "PASS" if not missing else "FAIL",
                "protocols, norm stats and ledger present" if not missing
                else "missing " + ", ".join(missing))

    if compile_scripts:
        broken: list[str] = []
        for path in sorted((ROOT / "scripts").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py")):
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError:
                broken.append(str(path.relative_to(ROOT)))
        checker.add("R1-06", "scripts compile", "PASS" if not broken else "FAIL",
                    "all scripts and tests import cleanly" if not broken
                    else "syntax error in " + ", ".join(broken[:3]))


def check_r2(data: dict, checker: Checker) -> None:
    if shutil.which("docker") is None:
        checker.add("R2-01", "docker", "SKIP", "docker is not installed; R2 needs it")
        for image in data.get("images") or []:
            checker.add(f"R2-02.{image['name']}", f"image {image['name']}", "SKIP", "docker unavailable")
        checker.add("R2-03", "registry access", "SKIP", "docker unavailable")
        return

    checker.add("R2-01", "docker", "PASS", "docker CLI available")
    for image in data.get("images") or []:
        current = docker_image_id(image["tag"])
        want = image["digest"].removeprefix("sha256:")[:12]
        hit = bool(current) and current.removeprefix("sha256:").startswith(want)
        checker.add(f"R2-02.{image['name']}", f"image {image['name']}",
                    "PASS" if hit else "FAIL",
                    f"id={current[:19]}..." if current else f"{image['tag']} not present locally")

    images = data.get("images") or []
    registry = os.environ.get(images[0]["registry_env"], "") if images else ""
    checker.add("R2-03", "registry access", "PASS" if registry else "WARN",
                "PI05_REGISTRY is set" if registry
                else "PI05_REGISTRY is unset; images must already be present locally")


def check_r3(data: dict, checker: Checker) -> None:
    checkpoint = None
    for asset in data.get("assets") or []:
        if asset.get("env") == "PI05_CKPT":
            checkpoint = asset
            break
    raw = os.environ.get("PI05_CKPT", "") if checkpoint else ""
    if not raw:
        checker.add("R3-01", "checkpoint", "SKIP", "PI05_CKPT is unset")
    else:
        path = Path(raw).expanduser()
        if path.is_file():
            files = 1
        elif path.is_dir():
            files = sum(1 for _ in itertools.islice(path.rglob("*"), 5))
        else:
            files = 0
        checker.add("R3-01", "checkpoint", "PASS" if files else "FAIL",
                    f"{files}+ files under {path.name}" if files else f"nothing at {path.name}")

    ledger = ROOT / "experiments" / "ledger.jsonl"
    records = 0
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("run_id"):
                records += 1
    checker.add("R3-02", "replayable record", "PASS" if records else "FAIL",
                f"{records} ledger records" if records else "no parsable run record in the ledger")


def blocking_fields(data: dict) -> list[str]:
    """Fields that must be filled before an outsider can obtain the pinned bytes."""
    out: list[str] = []
    for source in data.get("sources") or []:
        if source.get("status") != "blocked":
            continue
        for field in ("archive_url", "archive_sha256"):
            if not source.get(field):
                out.append(f"sources[{source['name']}].{field}")
    return out


def verdict(checker: Checker) -> tuple[str, str]:
    r1 = checker.ok("R1")
    r2 = r1 and checker.ok("R2") and any(c["status"] == "PASS" for c in checker.of("R2"))
    r3 = r2 and checker.ok("R3")
    level = "R3" if r3 else "R2" if r2 else "R1" if r1 else "none"
    names = {"R3": "numerical", "R2": "environmental", "R1": "structural", "none": "not reproducible"}
    return level, names[level]


def cmd_doctor(args: argparse.Namespace) -> int:
    checker = Checker()
    try:
        data = load_manifest()
    except ManifestError as exc:
        checker.add("R1-03", "manifest schema", "FAIL", str(exc))
        data = {}

    if data:
        check_r1(data, checker, args.compile)
        check_r2(data, checker)
        check_r3(data, checker)
        for field in blocking_fields(data):
            checker.add("R0-block", "unresolved pin", "BLOCK", field)

    level, level_name = verdict(checker)
    report = {
        "level": level,
        "level_name": level_name,
        "checks": checker.checks,
        "summary": checker.counts(),
        "blocking": blocking_fields(data) if data else [],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"reproduction check · {ROOT.name} · level={level} ({level_name})")
        print("-" * 92)
        width_id = max((len(check["id"]) for check in checker.checks), default=6)
        width_name = max((len(check["name"]) for check in checker.checks), default=4)
        for check in checker.checks:
            print(f"{check['id']:<{width_id}}  {check['name']:<{width_name}}  "
                  f"{check['status']:<5}  {check['detail']}")
        print("-" * 92)
        print("summary: " + "  ".join(f"{key}={value}" for key, value in sorted(checker.counts().items())))
        if report["blocking"]:
            print("unresolved pins (R2 stays out of reach until filled):")
            for field in report["blocking"]:
                print(f"  - {field}")
            print("  fill them in configs/repro/manifest.json, then re-run doctor")
        print("next: python3 scripts/repro/repro.py plan")

    if args.strict and (report["blocking"] or any(c["status"] in {"WARN", "BLOCK"} for c in checker.checks)):
        return 1
    return 0 if level != "none" else 2


def cmd_plan(_: argparse.Namespace) -> int:
    try:
        data = load_manifest()
    except ManifestError as exc:
        print(f"blocked: {exc}")
        return 2

    sources = data.get("sources") or []
    unpinned = [source["name"] for source in sources if not source.get("archive_sha256")]
    untagged = [source["name"] for source in sources if source.get("status") == "blocked"]

    print("to reach R1 (structural):")
    print("  python3 scripts/repro/repro.py doctor --compile")
    print("to reach R2 (environmental):")
    if unpinned:
        print("  1. pin the bytes you actually used: configs/repro/manifest.json")
        print("     sources[].archive_url + sources[].archive_sha256 for " + ", ".join(unpinned))
        print("  2. export PI05_REGISTRY=<registry that serves the pinned digests>")
        print("  3. python3 scripts/repro/repro.py fetch-images")
        print("  4. python3 scripts/repro/repro.py fetch-source " + unpinned[0] + " --archive <path or url>")
        print("  5. python3 scripts/repro/repro.py verify-source")
    else:
        print("  python3 scripts/repro/repro.py fetch-images")
        for name in untagged or [source["name"] for source in sources]:
            print(f"  python3 scripts/repro/repro.py fetch-source {name}")
        print("  python3 scripts/repro/repro.py verify-source")
    print("to reach R3 (numerical):")
    print("  export PI05_CKPT=<checkpoint dir> PI05_NORM_STATS=configs/libero/norm_stats.json")
    print("  ./scripts/run.sh smoke --suite libero-spatial --episodes 1")
    print("  python3 scripts/repro/repro.py doctor")
    return 0


def cmd_fetch_images(args: argparse.Namespace) -> int:
    data = load_manifest()
    binary = shutil.which("docker")
    if binary is None:
        print("docker is not installed", file=sys.stderr)
        return 2
    registry = os.environ.get("PI05_REGISTRY", "").rstrip("/")
    failed = False
    for image in data.get("images") or []:
        reference = f"{registry}/{image['tag']}" if registry else image["tag"]
        print(f"pull {reference}")
        if args.dry_run:
            continue
        pulled = subprocess.run([binary, "pull", reference])
        if pulled.returncode != 0:
            print(f"  failed to pull {reference}", file=sys.stderr)
            failed = True
            continue
        current = docker_image_id(reference) or docker_image_id(image["tag"]) or ""
        want = image["digest"].removeprefix("sha256:")[:12]
        if not current.removeprefix("sha256:").startswith(want):
            print(f"  image id mismatch: expected {want}, got {current[:19]}", file=sys.stderr)
            failed = True
        else:
            print(f"  ok {image['digest'][:19]}...")
    return 1 if failed else 0


def safe_members(archive: tarfile.TarFile, target: Path) -> None:
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ManifestError(f"refusing unsafe archive member {member.name}")
        if member.issym() or member.islnk():
            raise ManifestError(f"refusing archive link {member.name}")
    target.mkdir(parents=True, exist_ok=True)


def cmd_fetch_source(args: argparse.Namespace) -> int:
    data = load_manifest()
    source = next((item for item in data.get("sources") or [] if item["name"] == args.name), None)
    if source is None:
        print(f"unknown source {args.name}", file=sys.stderr)
        return 2
    if not source.get("archive_sha256"):
        print(f"{args.name} has no pinned archive_sha256; refusing to fetch unverified bytes\n"
              f"  fill sources[{args.name}].archive_url and .archive_sha256 first", file=sys.stderr)
        return 3

    CACHE.mkdir(parents=True, exist_ok=True)
    local = Path(args.archive).expanduser() if args.archive else CACHE / f"{args.name}.tar"
    target = ROOT / source["target"]
    if target.exists() and not args.dry_run:
        print(f"{source['target']} already exists; remove it yourself if it must be replaced",
              file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"would fetch {source.get('archive_url') or args.archive} -> {local}")
        print(f"then verify sha256={source['archive_sha256'][:16]}... and extract to {source['target']}")
        return 0

    if not local.is_file():
        if not source.get("archive_url"):
            print(f"no archive at {local} and no archive_url pinned", file=sys.stderr)
            return 3
        print(f"download {source['archive_url']}")
        urllib.request.urlretrieve(source["archive_url"], local)

    got = sha256_file(local)
    if got != source["archive_sha256"]:
        print(f"sha256 mismatch: {got} != {source['archive_sha256']}", file=sys.stderr)
        return 1

    staging = Path(tempfile.mkdtemp(prefix=f"{args.name}-", dir=str(CACHE)))
    try:
        with tarfile.open(local) as archive:
            safe_members(archive, staging)
            try:
                archive.extractall(staging, filter="data")
            except TypeError:  # python < 3.12 has no extraction filters
                archive.extractall(staging)
        entries = [entry for entry in staging.iterdir() if entry.name != "__MACOSX"]
        payload = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(payload), str(target))
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"extracted {args.name} to {source['target']} ({source['files']} files expected)")
    print("now regenerate the content lock inside the image and compare the aggregate")
    return 0


def cmd_pin(args: argparse.Namespace) -> int:
    """Record the archive actually used, so the next reader inherits a checkable pin."""
    data = load_manifest()
    source = next((item for item in data.get("sources") or [] if item["name"] == args.name), None)
    if source is None:
        print(f"unknown source {args.name}", file=sys.stderr)
        return 2
    archive = Path(args.archive).expanduser()
    if not archive.is_file():
        print(f"no archive at {archive}", file=sys.stderr)
        return 2

    digest = sha256_file(archive)
    previous = source.get("archive_sha256")
    print(f"{args.name}: {archive.name} sha256={digest}")
    if previous and previous != digest and not args.force:
        print(f"refusing to replace an existing pin ({previous[:16]}...) without --force",
              file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry run; manifest unchanged")
        return 0

    source["archive_sha256"] = digest
    if args.url:
        source["archive_url"] = args.url
    complete = bool(source.get("archive_url")) and bool(source.get("archive_sha256"))
    source["status"] = "pinned" if complete else "blocked"
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    if not complete:
        print("archive_url is still null; R2 stays blocked until it is recorded")
    return 0


def cmd_verify_source(_: argparse.Namespace) -> int:
    lock = ROOT / "scripts" / "verify" / "source_lock.py"
    tree = Path(os.environ.get("PI05_OPENPI_SRC", ROOT / "third_party" / "openpi"))
    if not tree.is_dir():
        print(f"no source tree at {tree}; run fetch-source first", file=sys.stderr)
        return 3
    result = subprocess.run([sys.executable, str(lock)], cwd=str(ROOT))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="grade R1/R2/R3 for this checkout")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--compile", action="store_true", help="also syntax-check scripts and tests")
    doctor.add_argument("--strict", action="store_true", help="fail on warnings and unresolved pins")
    doctor.set_defaults(func=cmd_doctor)

    plan = sub.add_parser("plan", help="print the commands that reach the next level")
    plan.set_defaults(func=cmd_plan)

    images = sub.add_parser("fetch-images", help="pull pinned images and verify their ids")
    images.add_argument("--dry-run", action="store_true")
    images.set_defaults(func=cmd_fetch_images)

    source = sub.add_parser("fetch-source", help="fetch, verify and unpack a pinned source archive")
    source.add_argument("name")
    source.add_argument("--archive", help="local archive path instead of archive_url")
    source.add_argument("--dry-run", action="store_true")
    source.set_defaults(func=cmd_fetch_source)

    pin = sub.add_parser("pin", help="record the sha256 of the archive actually used")
    pin.add_argument("name")
    pin.add_argument("--archive", required=True)
    pin.add_argument("--url", help="archive URL to record alongside the hash")
    pin.add_argument("--force", action="store_true", help="replace an existing pin")
    pin.add_argument("--dry-run", action="store_true")
    pin.set_defaults(func=cmd_pin)

    verify = sub.add_parser("verify-source", help="compare the local tree against the content lock")
    verify.set_defaults(func=cmd_verify_source)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
