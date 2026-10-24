#!/usr/bin/env python3
"""Redacted security scan for files that could be published from this checkout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "security-reports"
ALLOW_IPV4 = {
    ipaddress.ip_address(value)
    for value in ("127.0.0.1", "0.0.0.0", "255.255.255.255", "10.0.2.2")
}
DOC_IPV4 = tuple(ipaddress.ip_network(value) for value in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))
DOC_IPV6 = ipaddress.ip_network("2001:db8::/32")

IPV4 = re.compile(rb"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
IPV6 = re.compile(
    rb"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{0,4}){1,7}|"
    rb"::(?:[0-9A-Fa-f]{0,4}:?){1,7})(?![0-9A-Fa-f:])"
)
TEXT_PATTERNS = {
    "private-key": re.compile(rb"BEGIN [A-Z ]*PRIVATE KEY"),
    "absolute-private-path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9._-]+"),
    "private-key-option": re.compile(rb"(?:ssh|scp)[ \t]+-i[ \t]+/", re.I),
    "private-host-alias": re.compile(rb"\blelab-libero-policy\b", re.I),
    "private-host-marker": re.compile(rb"\bremoteServer\b|\bid_ed25519_cec_sim\b"),
    "user-at-ip": re.compile(rb"[A-Za-z0-9._-]+@(?:[0-9]{1,3}\.){3}[0-9]{1,3}"),
    "ssh-port-option": re.compile(rb"ssh[ \t]+-p[ \t]+[0-9]{2,5}\b"),
    "private-source-path": re.compile(rb"/data/openpi\b"),
    "github-token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential-assignment": re.compile(
        rb"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token)"
        rb"[ \t]*[:=][ \t]*[\"']?[A-Za-z0-9/+_=.-]{16,}", re.I
    ),
}
SKIP_MEDIA = {".mp4", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def allowed_ip(value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if value.is_loopback or value.is_unspecified or value.is_multicast:
        return True
    if isinstance(value, ipaddress.IPv4Address):
        return value in ALLOW_IPV4 or any(value in network for network in DOC_IPV4)
    return value in DOC_IPV6


def inspect(data: bytes, where: str, findings: set[tuple[str, str, int]]) -> None:
    for number, line in enumerate(data.splitlines(), 1):
        for pattern, family in ((IPV4, 4), (IPV6, 6)):
            for match in pattern.finditer(line):
                try:
                    address = ipaddress.ip_address(match.group().decode("ascii"))
                except ValueError:
                    continue
                if not allowed_ip(address):
                    findings.add((f"IPv{family}", where, number))
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(line):
                findings.add((label, where, number))


def blob_ids(commit: str) -> list[tuple[str, str]]:
    entries = git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0")
    out = []
    for entry in entries:
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        fields = meta.split()
        if len(fields) == 3 and fields[1] == b"blob":
            out.append((fields[2].decode("ascii"), path.decode("utf-8", errors="replace")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also include ignored local files")
    args = parser.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = REPORT_DIR / f"scan-{stamp}.txt"
    findings: set[tuple[str, str, int]] = set()
    scanned_blobs: set[str] = set()
    try:
        in_git = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if in_git:
            suffix = ["--all"] if args.all else ["--exclude-standard"]
            path_output = git("ls-files", "-z", "--cached", "--others", *suffix)
            for raw_path in path_output.split(b"\0"):
                if not raw_path:
                    continue
                rel = raw_path.decode("utf-8", errors="replace")
                if rel in {"scripts/security/scan.sh", "scripts/security/scan_content.py"} \
                        or Path(rel).suffix.lower() in SKIP_MEDIA:
                    continue
                path = ROOT / rel
                if path.is_file():
                    inspect(path.read_bytes(), rel, findings)

            # Scan staged blobs as well as the working copy.
            for entry in git("ls-files", "--stage", "-z").split(b"\0"):
                if not entry:
                    continue
                meta, raw_path = entry.split(b"\t", 1)
                mode, oid, stage = meta.split()
                if stage != b"0" or mode == b"160000":
                    continue
                rel = raw_path.decode("utf-8", errors="replace")
                if rel in {"scripts/security/scan.sh", "scripts/security/scan_content.py"} \
                        or Path(rel).suffix.lower() in SKIP_MEDIA or oid in scanned_blobs:
                    continue
                scanned_blobs.add(oid)
                inspect(git("cat-file", "blob", oid.decode("ascii")), f"index:{rel}", findings)

            # A clean current tree does not make secrets in commits being pushed safe.
            for commit_bytes in git("rev-list", "HEAD").splitlines():
                commit = commit_bytes.decode("ascii")
                message = git("show", "-s", "--format=%B", commit)
                inspect(message, f"commit:{commit[:12]}:message", findings)
                for oid, rel in blob_ids(commit):
                    if oid in scanned_blobs or rel in {"scripts/security/scan.sh", "scripts/security/scan_content.py"} \
                            or Path(rel).suffix.lower() in SKIP_MEDIA:
                        continue
                    scanned_blobs.add(oid)
                    inspect(git("cat-file", "blob", oid), f"commit:{commit[:12]}:{rel}", findings)
        else:
            for path in ROOT.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    inspect(path.read_bytes(), path.relative_to(ROOT).as_posix(), findings)

        gitleaks = subprocess.run(["gitleaks", "dir", "--redact", "--no-banner", "--exit-code", "7", "."],
                                  cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) \
            if shutil_which("gitleaks") else None
        if gitleaks is not None and gitleaks.returncode == 7:
            findings.add(("gitleaks", "redacted scan", 0))
        elif gitleaks is not None and gitleaks.returncode != 0:
            findings.add(("gitleaks-error", "scanner execution", 0))

        lines = [f"== public repository security scan · {datetime.now(timezone.utc).isoformat()} =="]
        if gitleaks is None:
            lines.append("-- gitleaks not installed, skipped --")
        elif gitleaks.returncode == 7:
            lines.append("FAIL [gitleaks] redacted scan reported a finding")
        elif gitleaks.returncode != 0:
            lines.append("FAIL [gitleaks] scanner could not complete")
        if findings:
            lines.extend(f"FAIL [{kind}] {where}:{line}" for kind, where, line in sorted(findings))
            lines.append(f"RESULT: FAIL ({len(findings)} findings) — fix before committing or pushing")
        else:
            lines.append("RESULT: PASS (0 findings)")

        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1 if findings else 0
    except (OSError, RuntimeError) as exc:
        # Do not include exception output from a credential-bearing command.
        report.write_text(f"RESULT: FAIL (scanner error: {type(exc).__name__})\n", encoding="utf-8")
        print(f"RESULT: FAIL (scanner error: {type(exc).__name__})")
        return 2


def shutil_which(name: str) -> str | None:
    from shutil import which
    return which(name)


if __name__ == "__main__":
    sys.exit(main())
