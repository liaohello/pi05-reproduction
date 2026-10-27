#!/usr/bin/env python3
"""E3.1 / G5: aggregate the prechecks, paired diagnostics and scope evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "so101" / "e31"
CHECKS: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append({"name": name, "passed": bool(condition), "detail": detail})


def load(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(mode: str) -> dict:
    run_dir = RUN / f"state_{mode}"
    out = run_dir / "summary_initial.json"
    if not out.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/so101/summarize_training.py"),
                        "--run-dir", str(run_dir), "--mode", "initial"],
                       capture_output=True, text=True, timeout=300)
    return load(out, {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-contact-analysis", type=Path,
                        default=RUN / "e3-contact-analysis.json")
    parser.add_argument("--output", type=Path, default=RUN / "validation.json")
    parser.add_argument("--public-output", type=Path, default=None)
    args = parser.parse_args()

    # Static contract/config fixes
    contract = (ROOT / "inference/SO101_CONTRACT.md").read_text(encoding="utf-8")
    check("contract_ten_physics_substeps", "ten physics substeps" in contract,
          "contract states 500 Hz / 50 Hz = ten substeps; contact stops may be partial")
    eval_cfg = load(ROOT / "configs/so101/eval.json", {})
    check("eval_checkpoint_step_binding", eval_cfg.get("checkpoint", {}).get("train_state_step") == 100,
          "E3 evaluated checkpoint 99 / TrainState.step 100, not 101")

    # G1 oracle
    oracle = load(RUN / "oracle_8100/oracle_report.json", {})
    check("oracle_present", bool(oracle), "oracle report")
    check("oracle_comparable", oracle.get("comparable") is True)
    check("oracle_passed", oracle.get("oracle_passed") is True)
    initial = oracle.get("initial_state", {})
    check("oracle_initial_state_exact",
          initial.get("file_matches_trace") is True
          and initial.get("restored_vs_recorded_max_abs") == 0.0
          and initial.get("frame0_state_error") == 0.0,
          "restored state/file/frame0 exact")
    check("oracle_action_prefix_exact", oracle.get("actions", {}).get("max_abs_error") == 0.0,
          "recorded applied actions match through the compared prefix")
    check("oracle_no_extra_clipping", oracle.get("actions", {}).get("extra_clipping") == 0)
    check("oracle_success_four_objects",
          oracle.get("result", {}).get("success") is True
          and oracle.get("result", {}).get("sorted_count") == 4
          and oracle.get("result", {}).get("safety_stops") == 0)
    check("oracle_substeps_ten",
          oracle.get("control", {}).get("physics_substeps_per_full_frame") == 10
          and oracle.get("time", {}).get("substeps_unique_expert") == [10]
          and oracle.get("time", {}).get("substeps_unique_runner") == [10])

    # G2 state
    state = load(RUN / "state_check/state_check.json", {})
    check("state_check_present", bool(state), "fixed-noise state check")
    check("state_all_checks", bool(state.get("checks")) and all(state["checks"].values()))
    off = state.get("modes", {}).get("off", {})
    on = state.get("modes", {}).get("on", {})
    check("state_off_negative_control",
          off.get("action_max_abs_difference") == 0.0
          and off.get("token_changed_positions") == 0
          and off.get("mask_is_valid_prefix") is True)
    check("state_on_tokens_live",
          (on.get("token_changed_positions", 0) > 0)
          and (on.get("quantized_state_changed", 0) > 0)
          and on.get("mask_is_valid_prefix") is True
          and on.get("truncated") is False)
    check("state_on_fixed_noise_output_live",
          (on.get("action_max_abs_difference") or 0.0) > 1e-6,
          "fixed noise, same images/task; only state changed")

    # Common initial and branch bindings
    b_init = load(RUN / "bindings/initial.json", {})
    b_off = load(RUN / "bindings/off_999.json", {})
    b_on = load(RUN / "bindings/on_999.json", {})
    check("initial_checkpoint_step0",
          b_init.get("train_state_step") == 0 and b_init.get("orbax_step_directory") == 0)
    check("branch_steps_actual",
          b_off.get("orbax_step_directory") == 999 and b_off.get("train_state_step") == 1000
          and b_on.get("orbax_step_directory") == 999 and b_on.get("train_state_step") == 1000,
          "Orbax directory 999 = TrainState.step 1000")
    init_off = load(RUN / "bindings/init_off_check.json", {})
    init_on = load(RUN / "bindings/init_on_check.json", {})
    check("common_initial_params",
          b_init.get("parameters_sha256") == init_off.get("parameters_sha256")
          == init_on.get("parameters_sha256")
          and b_init.get("parameters_sha256") is not None,
          "the two state modes initialize to the same parameter tree")
    check("branch_params_distinct",
          b_off.get("parameters_sha256") not in (None, b_on.get("parameters_sha256")),
          "state-off and state-on trained different final params")

    # Training branches
    for mode in ("off", "on"):
        rc = load(RUN / f"state_{mode}/run_config.json", {})
        cfg = load(RUN / f"state_{mode}/config_initial.json", {})
        run = load(RUN / f"state_{mode}/run_initial.json", {})
        model = (cfg.get("config") or {}).get("model") or {}
        check(f"training_{mode}_saved_state_mode", rc.get("state_mode") == mode)
        check(f"training_{mode}_model_flag",
              model.get("discrete_state_input") is (mode == "on"))
        check(f"training_{mode}_single_episode", str(rc.get("train_episodes")) == "0")
        check(f"training_{mode}_budget",
              rc.get("steps") == 1000 and rc.get("save_interval") == 250)
        check(f"training_{mode}_exit", run.get("exit_code") == 0)
        check(f"training_{mode}_wall_under_60min",
              (run.get("wall_time_s") or 1e9) <= 3600.0,
              f"{run.get('wall_time_s')} s")
        summary = summarize(mode)
        check(f"training_{mode}_1000_updates", summary.get("updates_completed") == 1000)
        losses = [summary.get("loss_first"), summary.get("loss_last"),
                  summary.get("loss_min"), summary.get("loss_max")]
        check(f"training_{mode}_finite_loss",
              all(value is not None and abs(float(value)) < 1e6 for value in losses))
        check(f"training_{mode}_stats_and_view",
              rc.get("norm_stats_canonical_sha256")
              == "5fd6e6d804b1299f8115846aff543589bff869c8c8a04c2868b625df70cdbae5"
              and rc.get("view_sha256")
              == "fbb8b6edda727a3a5b341865b4e856cd8b5eaced2e0e032b4e1d9b37f26b7dee")

    # Offline diagnostics
    labels = ["initial_off", "initial_on", "off_250", "off_500", "off_750", "off_999",
              "on_250", "on_500", "on_750", "on_999"]
    offline = {}
    for label in labels:
        row = load(RUN / f"offline/{label}/offline.json", {})
        offline[label] = row
        agg = row.get("aggregate", {})
        check(f"offline_{label}_present", bool(row))
        check(f"offline_{label}_32_windows",
              agg.get("windows") == 32 and len(row.get("per_window", {}).get("flow", [])) == 32)
        check(f"offline_{label}_model_loss_sanity",
              agg.get("model_loss_matches_all32") is True)
        check(f"offline_{label}_finite",
              all(isinstance(agg.get(key), (int, float))
                  for key in ("flow_true12_mse_mean", "flow_padding20_mse_mean",
                              "mae_joint_deg_mean", "mae_gripper_pct_mean")))
        check(f"offline_{label}_fixed_seeds",
              row.get("flow_seed_base") == 1000 and row.get("infer_seed_base") == 2000)
    off0 = offline.get("initial_off", {}).get("aggregate", {})
    offF = offline.get("off_999", {}).get("aggregate", {})
    on0 = offline.get("initial_on", {}).get("aggregate", {})
    onF = offline.get("on_999", {}).get("aggregate", {})
    check("offline_off_improves_true12",
          (offF.get("flow_true12_mse_mean") or 1e9) < (off0.get("flow_true12_mse_mean") or -1),
          "same 32 windows, flow rng and noise")
    check("offline_on_improves_true12",
          (onF.get("flow_true12_mse_mean") or 1e9) < (on0.get("flow_true12_mse_mean") or -1))
    check("offline_on_improves_action_mae",
          (onF.get("mae_joint_deg_mean") or 1e9) < (on0.get("mae_joint_deg_mean") or -1)
          and (onF.get("mae_gripper_pct_mean") or 1e9) < (on0.get("mae_gripper_pct_mean") or -1))
    off0_report = offline.get("initial_off", {}).get("aggregate", {})
    check("offline_padding_target_reported",
          "noise - actions" in str(off0_report.get("flow_u_target", ""))
          and "dimension ratio" in str(off0_report.get("flow_dimension_note", "")))

    # Paired rollouts
    rollouts = {}
    for mode in ("off", "on"):
        row = load(RUN / f"rollout_{mode}/rollout.json", {})
        rollouts[mode] = row
        check(f"rollout_{mode}_present", bool(row))
        check(f"rollout_{mode}_server_binding",
              bool(row.get("server_checks")) and all(row["server_checks"].values()))
        check(f"rollout_{mode}_checkpoint_step",
              row.get("server_metadata", {}).get("checkpoint_name") == "999"
              and row.get("train_state_step") == 1000)
        check(f"rollout_{mode}_training_scene", row.get("seed") == 8100)
        check(f"rollout_{mode}_trace", row.get("comparison", {}).get("trace_available") is True)
        check(f"rollout_{mode}_action_errors",
              row.get("comparison", {}).get("action_vs_expert_mae_joint_deg") is not None
              and row.get("comparison", {}).get("command_tracking_mae_joint_deg") is not None)
    off_roll = rollouts.get("off", {}).get("result", {})
    on_roll = rollouts.get("on", {}).get("result", {})
    check("rollout_guard_used",
          all(rollouts.get(mode, {}).get("server_checks", {}).get("action_contract")
              for mode in ("off", "on")))

    # Original E2/E3 preserved; contact attribution evidence
    e2 = load(ROOT / "experiments/so101/e2-training.json", {})
    e3 = load(ROOT / "experiments/so101/e3-diagnostics.json", {})
    check("original_e2_preserved",
          e2.get("initial", {}).get("updates_completed") == 100
          and e2.get("initial", {}).get("train_state_step") == 100
          and e2.get("resume", {}).get("saved_train_state_step") == 101)
    check("original_e3_preserved",
          e3.get("successes") == 0 and e3.get("valid_episodes") == 20
          and e3.get("manifest_sha256") == "b7d7daaa95532b1691141238bf270b0df6f859010c753e48ea5b3b97f1a20e38")
    contact = load(args.e3_contact_analysis, {})
    check("existing_e3_contact_manifest",
          contact.get("manifest_sha256") == e3.get("manifest_sha256"))
    check("existing_e3_contact_categories",
          contact.get("contact_groups_by_episode") == {"red_tray": 11, "table": 3, "blue_tray": 6},
          "same E3 manifest; gripper/environment contacts")
    check("existing_e3_contact_partial_frames",
          all(0 < int(row.get("stop_substeps", 0)) < 10
              for row in contact.get("episodes_detail", [])),
          "contact stops can end a control frame before ten substeps")

    # Ledger/scope and gate integrity
    rows = [json.loads(line) for line in (ROOT / "experiments/ledger.jsonl").read_text().splitlines()
            if line.strip()]
    e3_rows = [row for row in rows if row.get("scope") == "so101" and row.get("phase") == "E3"]
    check("e31_ledger_scope", all(row.get("scope") == "so101" for row in e3_rows),
          f"{len(e3_rows)} SO101 E3 records")
    status_run = subprocess.run([sys.executable, str(ROOT / "scripts/track/status.py"), "--quiet"],
                                capture_output=True, text=True, timeout=120)
    status_md = (ROOT / "reports/status.md")
    status_text = status_md.read_text(encoding="utf-8") if status_md.is_file() else ""
    check("libero_d5_untouched", "D5-2" in status_text and "0 条训练记录" in status_text)
    check("libero_d4_untouched", "libero_spatial=0/500" in status_text)

    # Entry decision
    oracle_ok = next((c["passed"] for c in CHECKS if c["name"] == "oracle_passed"), False)
    state_ok = next((c["passed"] for c in CHECKS if c["name"] == "state_all_checks"), False)
    improve_ok = next((c["passed"] for c in CHECKS if c["name"] == "offline_on_improves_true12"), False)
    placement = int(on_roll.get("sorted_count") or 0) >= 1
    e4 = bool(oracle_ok and state_ok and improve_ok and placement)
    reasons = {
        "oracle_passed": oracle_ok, "state_checks_passed": state_ok,
        "fixed_true12_improved": improve_ok,
        "training_scene_at_least_one_object": placement,
        "on_sorted_count": on_roll.get("sorted_count"),
        "off_sorted_count": off_roll.get("sorted_count"),
        "on_success": on_roll.get("success"), "off_success": off_roll.get("success"),
    }
    result = {"passed": all(c["passed"] for c in CHECKS), "checks": CHECKS,
              "e4_entry_gate": e4, "e4_reasons": reasons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.public_output:
        args.public_output.write_text(json.dumps({
            "schema_version": 1, "phase": "E3.1",
            "passed": result["passed"],
            "checks": [{"name": c["name"], "passed": c["passed"], "detail": c["detail"]}
                       for c in CHECKS],
            "e4_entry_gate": e4, "e4_reasons": reasons,
            "claim": "E3.1 debug gate; not E4 completion and not a benchmark",
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "e4_entry_gate": e4,
                      "failed": [c["name"] for c in CHECKS if not c["passed"]],
                      "e4_reasons": reasons}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
