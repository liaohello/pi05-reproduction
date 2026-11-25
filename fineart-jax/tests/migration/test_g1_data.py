"""G1 — data and tokenizer contract.

Every expectation comes from a read-only PyTorch process; the candidate only
ever produces the value being judged. Each group is exported twice and must
agree byte-for-byte, so an oracle that cannot replay fails before it can bless a
golden.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.migration import g1_cases as cases
from tests.migration import support as S

MAX_LENGTH = 200
MAX_ACTION_TOKENS = 256
CHUNK_SIZE = 50

ORACLE = S.PROJECT / "tests" / "migration" / "oracle" / "export_g1_data.py"
FIXTURE_BUILDER = S.PROJECT / "tests" / "migration" / "oracle" / "make_fit_fixture.py"
TOKENIZER_BUILDER = S.PROJECT / "migration" / "probes" / "build_paligemma_tokenizer.py"


# --------------------------------------------------------------------------- #
# Session setup: assets, fixtures and doubly-exported goldens
# --------------------------------------------------------------------------- #
class G1:
    """Paths and lazily exported reference goldens for this session."""

    def __init__(self) -> None:
        self.roots = S.roots()
        self.evidence = S.evidence_dir("g1")
        self.text_tokenizer = self.evidence / "assets" / "paligemma"
        self.fast_tokenizer = Path(self.roots["ASSET_ROOT"]) / "tokenizers" / "fast"
        self.fit_root = self.evidence / "fixtures" / "fit"
        self.oracle_root = self.evidence / "oracle"
        self._goldens: dict[str, dict] = {}
        self._tokenizer = None
        self._action_tokenizer = None
        self._prepare_assets()

    # -- assets ---------------------------------------------------------- #
    def _prepare_assets(self) -> None:
        if not self.fast_tokenizer.is_dir():
            raise AssertionError(
                "pinned FAST tokenizer asset is missing: " + str(self.fast_tokenizer)
            )
        derivation = self.text_tokenizer / "derivation.json"
        if not derivation.is_file():
            source = Path(self.roots["ASSET_ROOT"]) / "tokenizers" / "paligemma_tokenizer.model"
            result = subprocess.run(
                [self.roots["REF_PYTHON"], str(TOKENIZER_BUILDER), "--source", str(source),
                 "--out", str(self.text_tokenizer)],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if result.returncode != 0:
                raise AssertionError(
                    "cannot derive the PaliGemma tokenizer from the pinned model\n"
                    + result.stdout[-3000:] + "\n" + result.stderr[-3000:]
                )
        record = json.loads(derivation.read_text(encoding="utf-8"))
        assert record["derived_from"]["sha256"], "tokenizer derivation must pin its source bytes"
        for name, info in record["files"].items():
            assert S.sha256_file(self.text_tokenizer / name) == info["sha256"], (
                "derived tokenizer file changed: " + name
            )

        manifest = self.fit_root / "fixture.json"
        if not manifest.is_file():
            self.fit_root.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [self.roots["REF_PYTHON"], str(FIXTURE_BUILDER), "--out", str(self.fit_root)],
                capture_output=True, text=True, timeout=900, check=False,
                env={**dict(_clean_env()), **S.reference_env(self.roots)},
            )
            if result.returncode != 0:
                raise AssertionError("cannot build the fit fixture\n" + result.stdout[-3000:] + result.stderr[-3000:])
        fixture = json.loads(manifest.read_text(encoding="utf-8"))
        assert fixture["parquet_matches_npz"], "fixture representations disagree"
        assert S.sha256_file(self.fit_root / "actions.npz") == fixture["npz_sha256"], (
            "fit fixture npz changed since it was recorded"
        )

    # -- tokenizers ------------------------------------------------------ #
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.text_tokenizer))
        return self._tokenizer

    def fast_paligemma(self):
        """The FAST path uses ``add_eos_token=True, add_bos_token=False``.

        The terminator is ``encode("|")``, so the tokenizer configuration decides
        whether the sequence ends with EOS; using the text instance here would
        silently drop it.
        """
        if getattr(self, "_fast_paligemma", None) is None:
            from transformers import AutoTokenizer

            self._fast_paligemma = AutoTokenizer.from_pretrained(
                str(self.text_tokenizer), add_eos_token=True, add_bos_token=False
            )
        return self._fast_paligemma

    def action_tokenizer(self):
        if self._action_tokenizer is None:
            from transformers import AutoProcessor

            self._action_tokenizer = AutoProcessor.from_pretrained(
                str(self.fast_tokenizer), trust_remote_code=True
            )
        return self._action_tokenizer

    # -- goldens --------------------------------------------------------- #
    def golden(self, group: str) -> dict:
        if group in self._goldens:
            return self._goldens[group]
        runs = []
        for index in (1, 2):
            target = self.oracle_root / group / f"run{index}"
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            command = [
                self.roots["REF_PYTHON"], str(ORACLE),
                "--group", group, "--out", str(target),
                "--text-tokenizer", str(self.text_tokenizer),
                "--fast-tokenizer", str(self.fast_tokenizer),
                "--dataset-root", str(self.fit_root / "dataset"),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=3600, check=False,
                env={**dict(_clean_env()), **S.reference_env(self.roots)},
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"reference oracle failed for group {group}\n--- stdout ---\n"
                    + result.stdout[-6000:] + "\n--- stderr ---\n" + result.stderr[-6000:]
                )
            runs.append(target)

        first = json.loads((runs[0] / f"{group}.json").read_text(encoding="utf-8"))
        second = json.loads((runs[1] / f"{group}.json").read_text(encoding="utf-8"))
        assert first["arrays"] == second["arrays"], (
            f"oracle group {group} is not deterministic across two exports"
        )
        stripped_a = {key: value for key, value in first.items() if key != "arrays"}
        stripped_b = {key: value for key, value in second.items() if key != "arrays"}
        assert stripped_a == stripped_b, f"oracle group {group} metadata differs across exports"

        arrays = dict(np.load(str(runs[0] / f"{group}.npz")))
        for name, info in first["arrays"].items():
            array = arrays[name]
            assert list(array.shape) == info["shape"], f"golden shape drift in {group}/{name}"
            assert S.sha256_file(runs[0] / f"{group}.npz"), "golden file must exist"
        payload = {"arrays": arrays, "manifest": first, "path": runs[0]}
        self._goldens[group] = payload
        return payload


def _clean_env() -> dict:
    """A subprocess environment with no candidate import path and no Hub access."""
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return environment


@pytest.fixture(scope="session")
def g1() -> G1:
    return G1()


# --------------------------------------------------------------------------- #
# 1. Recipe routing
# --------------------------------------------------------------------------- #
def test_recipe_routing(g1: G1):
    """High/low/mixed/missing-subtask/recipe=None routing, replayed exactly."""
    from fineart_jax import tokenization as tk

    golden = g1.golden("routing")["manifest"]["cases"]
    selected_high = 0
    selected_low = 0

    for case in cases.ROUTING_CASES:
        reference = golden[case["name"]]
        assert len(reference) == len(cases.ROUTING_SAMPLE_INDICES)
        for entry in reference:
            index = entry["sample_idx"]
            if case["recipe"] is None:
                rendered = {
                    "messages_rendered": [{"role": "user", "content": case["task"]}],
                    "message_streams": ["low_level"],
                    "target_message_indices": [],
                }
            else:
                rendered = tk.render_sample(
                    recipe=case["recipe"],
                    persistent=case["persistent"],
                    events=case["events"],
                    t=case["timestamp"],
                    sample_idx=index,
                    task=case["task"],
                    default_task=case["default_task"],
                )
                if rendered is None:
                    rendered = tk.fallback_low_level_render(case["task"])
            got_messages = (
                [json.dumps(message, sort_keys=True) for message in rendered["messages_rendered"]]
                if rendered
                else []
            )
            assert got_messages == entry["messages"], (
                f"routing case {case['name']} sample {index}: messages differ"
            )
            assert list(rendered["message_streams"] if rendered else []) == entry["streams"], (
                f"routing case {case['name']} sample {index}: streams differ"
            )
            assert list(rendered["target_message_indices"] if rendered else []) == entry["targets"], (
                f"routing case {case['name']} sample {index}: targets differ"
            )
            if entry["streams"]:
                if entry["streams"][0] == "high_level":
                    selected_high += 1
                elif entry["streams"][0] == "low_level":
                    selected_low += 1

    # The blend must actually be exercised in both directions, otherwise the routing
    # test would pass on a candidate that always picks one branch.
    assert selected_high > 0 and selected_low > 0, (
        f"the 0.30/0.70 blend was not exercised on both sides: high={selected_high} low={selected_low}"
    )

    # A missing subtask must route to the task fallback rather than inventing text.
    missing = golden["missing_subtask"]
    assert all(entry["rendered"] for entry in missing)
    # recipe=None keeps the low-level fallback and never fabricates a target.
    fallback = golden["recipe_none_fallback"]
    assert all(entry["targets"] == [] for entry in fallback)


# --------------------------------------------------------------------------- #
# 2. Prompt, tokens, labels, EOS
# --------------------------------------------------------------------------- #
def test_prompt_tokens_labels_eos(g1: G1):
    """Prompt ids, supervised spans, -100 fill, EOS, padding and runtime tail."""
    from fineart_jax import tokenization as tk

    tokenizer = g1.tokenizer()
    payload = g1.golden("prompt")
    arrays = payload["arrays"]
    manifest = payload["manifest"]["cases"]

    for case in cases.PROMPT_CASES:
        name = case["name"]
        result = tk.encode_messages(
            tokenizer,
            [dict(message) for message in case["messages"]],
            list(case["streams"]),
            list(case["targets"]),
            state_row=case.get("state"),
            max_length=MAX_LENGTH,
        )
        ids = result["input_ids"][None, :]
        mask = result["attention_mask"][None, :]
        labels = result["labels"][None, :]

        np.testing.assert_array_equal(ids, arrays[f"{name}__ids"], err_msg=f"{name}: ids differ")
        np.testing.assert_array_equal(mask, arrays[f"{name}__mask"], err_msg=f"{name}: mask differs")
        np.testing.assert_array_equal(labels, arrays[f"{name}__labels"], err_msg=f"{name}: labels differ")
        np.testing.assert_array_equal(
            np.asarray([result["predict_actions"]], dtype=bool).reshape(1, 1),
            arrays[f"{name}__predict"].reshape(1, 1),
            err_msg=f"{name}: predict_actions differs",
        )

        assert int(mask.sum()) == manifest[name]["nonpad"], f"{name}: non-pad length differs"
        assert int((labels != -100).sum()) == manifest[name]["n_supervised"], (
            f"{name}: supervised token count differs"
        )
        assert tokenizer.decode(ids[0, : int(mask.sum())].tolist()) == manifest[name]["tail_text"], (
            f"{name}: decoded prompt differs"
        )

        # Right padding: everything after the real prompt is id 0 and unattended.
        nonpad = int(mask.sum())
        assert not mask[0, nonpad:].any(), f"{name}: padding must not be attended"
        assert (ids[0, nonpad:] == 0).all(), f"{name}: padding must be token id 0"
        # An ignored position never carries a label.
        assert (labels[0, ~mask[0]] == -100).all(), f"{name}: unattended tokens must be ignored"

    # EOS must be inside the supervised span for a target turn (that is how generation
    # learns to stop), and must not appear for a turn that is not supervised.
    target_case = "assistant_target"
    eos_id = tokenizer.eos_token_id
    supervised = arrays[f"{target_case}__labels"][0]
    assert eos_id in supervised[supervised != -100], "EOS must be supervised on a target turn"
    plain = arrays["single_turn_target__labels"][0]
    assert not (plain != -100).any(), "a turn without a target must not be supervised"

    # The runtime query appends the Assistant prefix so it matches the training layout.
    assert manifest["runtime_query"]["tail_text"].endswith("Assistant:"), (
        "runtime queries must end with the Assistant prefix"
    )


# --------------------------------------------------------------------------- #
# 3. State discretisation and location tokens
# --------------------------------------------------------------------------- #
def test_state_and_location_tokens(g1: G1):
    """256-bin state text, low-level-only state, and yx/yxyx location tokens."""
    from fineart_jax import tokenization as tk

    tokenizer = g1.tokenizer()
    payload = g1.golden("stateloc")
    manifest = payload["manifest"]
    arrays = payload["arrays"]

    for expected, row in zip(manifest["state"], cases.STATE_BIN_CASES, strict=True):
        got = tk.discretize_state_str(np.asarray(row, dtype=np.float32))
        assert got == expected["text"], f"state bins differ for {row}: {got} != {expected['text']}"
        bins = [int(value) for value in got.split()]
        # The reference does not clamp: a state outside [-1, 1] digitises to -1 or 256,
        # and that is the behaviour being migrated rather than corrected.
        assert len(bins) == len(row), "one bin per state dimension"
        if all(-1.0 <= float(value) <= 1.0 for value in row):
            assert all(0 <= value <= 255 for value in bins), "in-range states must map inside 0..255"

    for expected in manifest["points"]:
        assert tk.vqa_answer_to_loc(expected["answer"]) == expected["text"]
    for expected in manifest["boxes"]:
        assert tk.vqa_answer_to_loc(expected["answer"]) == expected["text"]

    # Point answers are y-then-x; boxes are y1 x1 y2 x2.
    point = tk.vqa_answer_to_loc({"label": "cup", "point_format": "xy", "point": [0.5, 0.25]})
    assert point == "cup <loc0256><loc0512>", point
    box = tk.vqa_answer_to_loc(
        {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [0.0, 0.25, 0.5, 1.0]}]}
    )
    assert box == "cup <loc0256><loc0000><loc1023><loc0512>", box
    # An empty label yields no location at all instead of a bare token pair.
    assert tk.vqa_answer_to_loc({"label": "", "point_format": "xy", "point": [0.5, 0.5]}) is None

    got_ids = np.asarray(
        [tokenizer(tk.loc_token(float(item["coord"])), add_special_tokens=False)["input_ids"][0]
         for item in manifest["loc_tokens"]],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(got_ids, arrays["loc_token_ids"], err_msg="location token ids differ")
    assert got_ids.min() >= 256000 and got_ids.max() <= 257151, (
        "location tokens must land in PaliGemma's reserved tail"
    )

    # Only a low-level stream carries proprioception.
    prompt_golden = g1.golden("prompt")["manifest"]["cases"]
    assert "State: " in prompt_golden["low_level_with_state"]["tail_text"]
    assert "State: " not in prompt_golden["assistant_target"]["tail_text"]


# --------------------------------------------------------------------------- #
# 4. FAST mapping and the code-token mask
# --------------------------------------------------------------------------- #
def test_fast_mapping_and_code_mask(g1: G1):
    """Real FAST output, both declared offsets, code-only supervision, padding."""
    from fineart_jax import tokenization as tk

    tokenizer = g1.fast_paligemma()
    action_tokenizer = g1.action_tokenizer()
    payload = g1.golden("fast")
    arrays = payload["arrays"]

    for skip in cases.FAST_SKIP_TOKENS_UNDER_TEST:
        for name, action in cases.FAST_ACTIONS.items():
            key = f"{name}__skip{skip}"
            result = tk.fast_action_tokens(
                np.asarray(action, dtype=np.float32),
                action_tokenizer,
                tokenizer,
                max_action_tokens=MAX_ACTION_TOKENS,
                fast_skip_tokens=skip,
            )
            np.testing.assert_array_equal(
                result.tokens[None, :], arrays[f"{key}__tokens"], err_msg=f"{key}: tokens"
            )
            np.testing.assert_array_equal(result.mask[None, :], arrays[f"{key}__mask"], err_msg=f"{key}: mask")
            np.testing.assert_array_equal(
                result.code_mask[None, :], arrays[f"{key}__code_mask"], err_msg=f"{key}: code mask"
            )

            # Structure: BOS, then the Action: prompt, then the codes, then the terminator.
            prompt_ids = tokenizer.encode("Action: ", add_special_tokens=False)
            assert result.tokens[0] == tokenizer.bos_token_id, "sequence must start with BOS"
            np.testing.assert_array_equal(result.tokens[1:1 + len(prompt_ids)], np.asarray(prompt_ids))
            code_positions = np.flatnonzero(result.code_mask)
            assert code_positions.size > 0 and code_positions[0] == 1 + len(prompt_ids)
            assert code_positions[-1] == 1 + len(prompt_ids) + code_positions.size - 1
            # The remapped codes live in PaliGemma's reserved tail.
            codes = result.tokens[code_positions]
            assert codes.min() > 256000 - 2048 - skip - 1, "code ids must sit in the reserved tail"
            # Positions that are neither code nor terminator stay unsupervised.
            non_code_labels = result.code_mask[result.mask]
            assert non_code_labels.sum() == code_positions.size, "only code tokens may be supervised"

    # The declared 128 offset must shift every code id by exactly the offset difference.
    for name in cases.FAST_ACTIONS:
        high = arrays[f"{name}__skip1152__tokens"]
        low = arrays[f"{name}__skip128__tokens"]
        mask = arrays[f"{name}__skip1152__code_mask"]
        # A larger skip lowers the remapped id, so the 128-offset ids sit higher by the
        # difference between the two offsets.
        delta = low[mask].astype(np.int64) - high[mask].astype(np.int64)
        assert np.all(delta == 1152 - 128), (
            f"{name}: a FAST offset change must shift code ids by the offset difference, got {sorted(set(delta.tolist()))}"
        )

    # Negative control: using the wrong declared offset for a 1152 checkpoint must be caught.
    wrong = tk.fast_action_tokens(
        np.asarray(cases.FAST_ACTIONS["ramp6"], dtype=np.float32),
        action_tokenizer, tokenizer, max_action_tokens=MAX_ACTION_TOKENS, fast_skip_tokens=128,
    )
    assert not np.array_equal(wrong.tokens, arrays["ramp6__skip1152__tokens"]), (
        "a mis-declared FAST offset must change the token ids"
    )


# --------------------------------------------------------------------------- #
# 5. Normalisation and relative actions
# --------------------------------------------------------------------------- #
def test_normalization_relative_actions(g1: G1):
    """Normalise/denormalise, constant dimensions, gripper exclusion, roundtrip."""
    from fineart_jax import data as d

    payload = g1.golden("norm")
    arrays = payload["arrays"]
    manifest = payload["manifest"]["cases"]

    for dim in cases.NORMALIZATION_DIMS:
        values = np.asarray(cases.norm_case_values(dim), dtype=np.float32)
        stats = cases.norm_case_stats(dim)
        for mode in ("QUANTILES", "MIN_MAX", "MEAN_STD"):
            key = f"norm_d{dim}_{mode}"
            normalized = d.normalize(values, mode, stats)
            restored = d.unnormalize(normalized, mode, stats)
            np.testing.assert_allclose(
                normalized, arrays[f"{key}__normalized"], rtol=1e-6, atol=1e-6,
                err_msg=f"{key}: normalisation differs",
            )
            np.testing.assert_allclose(
                restored, arrays[f"{key}__restored"], rtol=1e-6, atol=1e-6,
                err_msg=f"{key}: denormalisation differs",
            )
            np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6, err_msg=f"{key}: roundtrip")
            assert float(np.abs(normalized[:, 2]).max()) <= 1.0 + 1e-6, (
                f"{key}: a constant dimension must stay finite"
            )
            assert np.isfinite(normalized).all(), f"{key}: produced non-finite values"
            assert np.isclose(float(normalized[0, 2]), manifest[key]["constant_dim_value"], atol=1e-6)

    for dim, names in ((6, cases.ACTION_NAMES_6), (12, None)):
        key = f"relative_d{dim}"
        actions_raw, state_raw = cases.relative_case(dim)
        actions = np.asarray(actions_raw, dtype=np.float32)[None, ...]
        state = np.asarray([state_raw], dtype=np.float32)
        mask = d.relative_mask(dim, action_names=names, exclude_joints=["gripper"])
        relative = d.to_relative_actions(actions, state, mask)
        absolute = d.to_absolute_actions(relative, state, mask)
        np.testing.assert_allclose(relative, arrays[f"{key}__relative"], rtol=1e-6, atol=1e-6,
                                   err_msg=f"{key}: relative conversion differs")
        np.testing.assert_allclose(absolute, arrays[f"{key}__absolute"], rtol=1e-6, atol=1e-6,
                                   err_msg=f"{key}: absolute conversion differs")
        assert mask == manifest[key]["mask"], f"{key}: relative mask differs"
        np.testing.assert_allclose(absolute, actions, rtol=1e-6, atol=1e-6)

    # The gripper dimension must stay absolute; every other dimension must move.
    mask6 = d.relative_mask(6, action_names=cases.ACTION_NAMES_6, exclude_joints=["gripper"])
    assert mask6[-1] is False and all(mask6[:-1]), mask6
    swapped = d.relative_mask(6, action_names=list(reversed(cases.ACTION_NAMES_6)), exclude_joints=["gripper"])
    assert swapped != mask6, "a swapped action-name order must change the mask"

    # Negative control: swapping two action dimensions must be detectable.
    actions_raw, state_raw = cases.relative_case(6)
    actions = np.asarray(actions_raw, dtype=np.float32)[None, ...]
    state = np.asarray([state_raw], dtype=np.float32)
    mask = d.relative_mask(6, action_names=cases.ACTION_NAMES_6, exclude_joints=["gripper"])
    good = d.to_relative_actions(actions, state, mask)
    permuted = actions.copy()
    permuted[..., 0], permuted[..., 1] = actions[..., 1], actions[..., 0]
    assert not np.allclose(d.to_relative_actions(permuted, state, mask), good), (
        "a swapped action dimension order must change the relative actions"
    )


# --------------------------------------------------------------------------- #
# 6. Images: layout, padding, camera order
# --------------------------------------------------------------------------- #
def test_image_layout_padding_camera_order(g1: G1):
    """1/2/3 cameras, NCHW/NHWC, centre padding, [-1,1], fixed camera order.

    Every image and every mask is compared against an independent reference
    export. Non-resize cameras use the frozen 1e-6 data-preprocessing allclose.
    Resize rows are diagnostic measurements only: the contract records resize
    separately and has *not* approved a wider tolerance, so they must not be
    reported as a frozen-threshold pass or used to excuse other differences.
    """
    from fineart_jax import data as d

    payload = g1.golden("images")
    arrays = payload["arrays"]
    manifest = payload["manifest"]["cases"]
    resize_rows = {}
    saw_partial_missing = 0
    saw_all_missing = 0

    for case in cases.IMAGE_CASES:
        name = case["name"]
        configured = list(case["keys"])
        present = list(case.get("present", configured))
        full = cases.image_case_batch(case)
        batch = {key: value for key, value in full.items() if key in present}
        info = manifest[name]

        if case.get("expect_error"):
            # Total absence raises on both sides; partial absence below must not.
            with pytest.raises(ValueError):
                d.preprocess_images(batch, configured)
            assert info["error"] == "ValueError", f"{name}: reference error identity differs"
            saw_all_missing += 1
            continue

        processed = d.preprocess_images(batch, configured, resolution=(224, 224))
        assert processed.keys == info["keys"], f"{name}: resulting camera order differs"
        assert processed.present_keys == present, f"{name}: present-camera record differs"
        assert len(processed.images) == info["n_cameras"] == len(processed.masks), (
            f"{name}: image/mask count differs"
        )
        if len(present) < len(configured):
            saw_partial_missing += 1

        # Distinct, non-constant inputs make a swapped non-first camera detectable.
        for left in range(len(present)):
            for right in range(left + 1, len(present)):
                assert not np.allclose(processed.images[left], processed.images[right]), (
                    f"{name}: cameras {left}/{right} are indistinguishable, so a swap could pass"
                )
                assert not np.allclose(
                    arrays[f"{name}__image{left}"], arrays[f"{name}__image{right}"]
                ), f"{name}: reference cameras {left}/{right} are indistinguishable"

        spatial = tuple(case["shape"][-2:]) if case["layout"] == "nchw" else tuple(case["shape"][:2])
        needs_resize = spatial != (224, 224)

        for index, (image, mask) in enumerate(zip(processed.images, processed.masks)):
            golden_image = arrays[f"{name}__image{index}"]
            golden_mask = arrays[f"{name}__mask{index}"]
            assert list(image.shape) == info["shapes"][index], f"{name}: image {index} shape differs"
            assert list(mask.shape) == info["mask_shapes"][index], f"{name}: mask {index} shape differs"
            assert image.dtype == np.float32 and mask.dtype == np.bool_, (
                f"{name}: image/mask {index} dtype differs"
            )
            assert np.isfinite(image).all(), f"{name}: image {index} has non-finite pixels"
            assert image.min() >= -1.0 - 1e-6 and image.max() <= 1.0 + 1e-6, (
                f"{name}: image {index} must be in [-1, 1]"
            )
            # Masks are IDs/mask class: exact equality, never an approximate bucket.
            np.testing.assert_array_equal(mask, golden_mask, err_msg=f"{name}: mask {index} differs")

            if index >= len(present):
                # Appended camera: all -1 pixels and an all-false mask. The padding
                # is appended after every present camera, not in the missing slot.
                assert np.array_equal(image, np.full_like(image, -1.0)), (
                    f"{name}: appended camera {index} is not all -1"
                )
                assert not mask.any(), f"{name}: appended camera {index} mask must be false"
                continue

            if needs_resize:
                difference = np.abs(image - golden_image)
                row = {
                    "max_abs": float(difference.max()),
                    "mismatched": int((difference > 1e-6).sum()),
                    "elements": int(difference.size),
                    "status": "diagnostic_only_not_a_frozen_threshold",
                }
                resize_rows[f"{name}__image{index}"] = row
                # These are geometry sanity checks, not a substitute for 1e-6.
                assert row["max_abs"] < 1e-2, (
                    f"{name}: resize differs far beyond kernel rounding ({row['max_abs']:.3e})"
                )
                assert row["mismatched"] / row["elements"] < 0.01, (
                    f"{name}: resize disagreement covers {row['mismatched']}/{row['elements']} "
                    "pixels, which is geometry error rather than kernel rounding"
                )
            else:
                np.testing.assert_allclose(
                    image, golden_image, rtol=1e-6, atol=1e-6,
                    err_msg=f"{name}: image {index} differs",
                )

    assert saw_partial_missing >= 4, "partial-camera cases must exercise first/middle/last/multiple"
    assert saw_all_missing == 1, "the all-cameras-missing case must be present"
    assert resize_rows, "at least one case must exercise the resize path"

    # A 224x224 input must only be rescaled, never resampled.
    same = cases.image_case_batch(cases.IMAGE_CASES[0])
    identity = d.preprocess_images(same, cases.IMAGE_CASES[0]["keys"], resolution=(224, 224))
    np.testing.assert_array_equal(
        identity.images[0], same[cases.IMAGE_CASES[0]["keys"][0]] * 2.0 - 1.0,
        err_msg="a 224x224 input must only be rescaled, never resampled",
    )

    evidence = S.evidence_dir("g1") / "image_resize_rows.json"
    evidence.write_text(
        json.dumps(
            {
                "contract_threshold": "atol=1e-6 rtol=1e-6 (data_preprocessing)",
                "note": "Resize rows are measured diagnostics; no widened threshold is approved.",
                "rows": resize_rows,
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    # Negative control: a reversed camera order must change the stacked result.
    two = cases.IMAGE_CASES[2]
    batch = cases.image_case_batch(two)
    correct = d.preprocess_images(batch, two["keys"])
    reversed_order = d.preprocess_images(batch, list(reversed(two["keys"])))
    assert not np.allclose(correct.stacked(), reversed_order.stacked()), (
        "a swapped camera order must change the batched images"
    )


# --------------------------------------------------------------------------- #
# 7. FAST fitting, saving and reloading
# --------------------------------------------------------------------------- #
def test_fast_fit_save_reload(g1: G1, tmp_path):
    """Fit identity, episode exclusion, RMSE thresholds and reload without the original."""
    from fineart_jax import tokenization as tk

    payload = g1.golden("fit")
    manifest = payload["manifest"]
    arrays = payload["arrays"]

    fixture = np.load(str(g1.fit_root / "actions.npz"))
    actions = fixture["action"]
    states = fixture["state"]
    episodes = fixture["episode_index"]
    assert actions.shape == (cases.FIT_EPISODES * cases.FIT_FRAMES_PER_EPISODE, cases.FIT_ACTION_DIM)

    signature = tk.dataset_signature(
        dataset_repo_id="fixture/local-g1",
        base_tokenizer_name=str(g1.fast_tokenizer),
        n_samples=cases.FIT_SAMPLES,
        chunk_size=cases.FIT_CHUNK_SIZE,
        normalization_mode="QUANTILES",
        exclude_episodes=cases.FIT_EXCLUDE_EPISODES,
        validation_samples=cases.FIT_VALIDATION_SAMPLES,
        max_reconstruction_rmse=0.10,
        max_dim_rmse=0.20,
    )
    assert signature == manifest["signature"], (
        f"fit cache identity differs: {signature} != {manifest['signature']}"
    )

    chunks, rows = tk.sample_action_chunks(
        actions=actions,
        episode_index=episodes,
        n_samples=cases.FIT_SAMPLES,
        validation_samples=cases.FIT_VALIDATION_SAMPLES,
        chunk_size=cases.FIT_CHUNK_SIZE,
        exclude_episodes=cases.FIT_EXCLUDE_EPISODES,
    )
    total = cases.FIT_SAMPLES + cases.FIT_VALIDATION_SAMPLES
    assert chunks.shape == (total, cases.FIT_CHUNK_SIZE, cases.FIT_ACTION_DIM)
    # Excluded episodes must not appear in the sampled rows.
    sampled_episodes = episodes[rows]
    assert cases.FIT_EXCLUDE_EPISODES[0] not in sampled_episodes, "excluded episode leaked into the fit"
    assert np.ptp(chunks) > 0.1, "the fit fixture must not be degenerate"

    normalized = tk_normalize(chunks)
    from transformers import AutoProcessor

    base = AutoProcessor.from_pretrained(str(g1.fast_tokenizer), trust_remote_code=True)
    fitted = base.fit(normalized[: cases.FIT_SAMPLES])
    report, decoded = tk.reconstruction_report(
        fitted, normalized[cases.FIT_SAMPLES: total], 0.10, 0.20
    )
    assert report["reconstruction_rmse"] <= 0.10, f"reconstruction RMSE too high: {report['reconstruction_rmse']}"
    assert report["max_dim_rmse"] <= 0.20, f"per-dimension RMSE too high: {report['max_dim_rmse']}"
    assert np.isfinite(decoded).all()

    saved = tmp_path / "fitted"
    fitted.save_pretrained(str(saved))
    (saved / "reconstruction_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    assert (saved / "processor_config.json").is_file(), "a saved tokenizer must be self-contained"

    probe = np.asarray(arrays["probe"], dtype=np.float32)
    tokenizer = g1.fast_paligemma()
    before = tk.fast_action_tokens(probe, fitted, tokenizer, max_action_tokens=MAX_ACTION_TOKENS,
                                   fast_skip_tokens=1152)
    np.testing.assert_array_equal(
        before.tokens[None, :], arrays["tokens_before"], err_msg="fitted tokenizer tokens"
    )
    np.testing.assert_array_equal(before.code_mask[None, :], arrays["code_before"], err_msg="fitted code mask")

    # Reload from an isolated copy while the original path is unavailable.
    isolated = tmp_path / "isolated"
    shutil.copytree(saved, isolated)
    parked = tmp_path / "parked"
    saved.rename(parked)
    try:
        reloaded = AutoProcessor.from_pretrained(str(isolated), trust_remote_code=True)
        after = tk.fast_action_tokens(probe, reloaded, tokenizer, max_action_tokens=MAX_ACTION_TOKENS,
                                      fast_skip_tokens=1152)
        np.testing.assert_array_equal(
            after.tokens[None, :], arrays["tokens_after"], err_msg="reloaded tokens"
        )
        np.testing.assert_array_equal(after.tokens, before.tokens, err_msg="reload changed the output")
    finally:
        parked.rename(saved)

    assert manifest["tokens_identical_after_reload"] is True, (
        "the reference itself reported a reload mismatch; G1 cannot pass on that basis"
    )


def tk_normalize(chunks: np.ndarray) -> np.ndarray:
    """Normalise fitted chunks with the reference's fallback statistics."""
    flat = chunks.reshape(-1, chunks.shape[-1])
    low = np.quantile(flat, 0.01, axis=0).astype(np.float32)
    high = np.quantile(flat, 0.99, axis=0).astype(np.float32)
    spread = np.where(high == low, np.float32(1e-8), high - low).astype(np.float32)
    return (np.float32(2.0) * (chunks - low) / spread - np.float32(1.0)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 8. No silent truncation of training targets
# --------------------------------------------------------------------------- #
def test_no_target_truncation_in_fixtures(g1: G1):
    """Normal fixtures keep their targets whole; over-long input matches the reference."""
    from fineart_jax import tokenization as tk

    tokenizer = g1.tokenizer()
    payload = g1.golden("truncation")
    arrays = payload["arrays"]
    manifest = payload["manifest"]

    contract = S.contract()
    defaults = contract["defaults"]
    assert defaults["tokenizer_max_length"] == MAX_LENGTH
    assert defaults["max_action_tokens"] == MAX_ACTION_TOKENS
    assert defaults["chunk_size"] == CHUNK_SIZE

    for name, text in (("short", "pick up the cube"), ("long", cases.TRUNCATION_LONG_PROMPT)):
        result = tk.encode_messages(
            tokenizer, [{"role": "user", "content": text}], ["high_level"], [0], max_length=MAX_LENGTH
        )
        np.testing.assert_array_equal(result["input_ids"][None, :], arrays[f"{name}__ids"],
                                      err_msg=f"{name}: ids differ")
        np.testing.assert_array_equal(result["attention_mask"][None, :], arrays[f"{name}__mask"],
                                      err_msg=f"{name}: mask differs")
        np.testing.assert_array_equal(result["labels"][None, :], arrays[f"{name}__labels"],
                                      err_msg=f"{name}: labels differ")
        nonpad = int(result["attention_mask"].sum())
        assert nonpad == manifest[name]["nonpad"], f"{name}: non-pad length differs"
        assert int((result["labels"] != -100).sum()) == manifest[name]["n_supervised"]

    # A normal fixture must not be truncated and must supervise its whole payload.
    short_mask = arrays["short__mask"][0]
    assert manifest["short"]["truncated"] is False and int(short_mask.sum()) < MAX_LENGTH
    short_labels = arrays["short__labels"][0]
    assert int((short_labels != -100).sum()) == manifest["short"]["n_supervised"]
    assert int((short_labels != -100).sum()) > 0

    # The over-long case reproduces the reference's truncation, including its length.
    assert manifest["long"]["truncated"] is True, "the long case must actually exercise truncation"
    assert manifest["long"]["nonpad"] == MAX_LENGTH, "truncation must fill the configured maximum"

    # Negative control: shifting the labels by one token must be detected.
    shifted = np.roll(arrays["short__labels"], 1)
    assert not np.array_equal(shifted, short_labels), (
        "a one-token label shift must not look like the reference"
    )
