# Process / restart audit

**Status: `AUTO_RESTART_UNRESOLVED`.**

This label is not a formality. It is the only defensible state: the host cannot fork, so
the auditor could not enumerate a single process, and no artifact in the repository proves
that whatever launched the resume cycle has stopped. Per the task's own rule, training must
therefore be treated as **possibly live**.

---

## 1. What could not be checked (BLOCKED)

| Check | Command that would settle it | Result |
|---|---|---|
| Live trainer / container | `ps -eo pid,etime,cmd \| grep -E 'train_so101\|openpi'` ; `docker ps -a --format '{{.Names}}\t{{.Status}}'` | **BLOCKED** — `fork: EAGAIN` |
| systemd transient units | `systemctl list-units --all 'resume-*' 'so101-*'` ; `systemctl list-timers --all` | **BLOCKED** |
| Unit provenance | `systemctl show -p FragmentPath,ExecStart,InvocationID <unit>` | **BLOCKED** |
| Supervisor / cron | `supervisorctl status` ; `crontab -l` ; `ls /etc/cron.*` | **BLOCKED** |
| Other agent controllers | `ps -eo cmd \| grep -Ei 'codex\|agent\|scheduler'` | **BLOCKED** |
| Docker restart policy | `docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' <c>` | **BLOCKED** |
| GPU occupancy now | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | **BLOCKED** |

Nothing in this table may be inferred from the fact that the resolver ran once. A transient
unit that fired twice leaves no on-disk trace that distinguishes it from a script.

## 2. What the artifacts do prove (VERIFIED)

These are hash-bound files written by the run itself. They need no process table.

**(a) At least two resume attempts existed, and the audited one is the second.**

- `formal/config_resume.json:4` → `run_id = "...-resume-2000b"`.
  The `b` suffix implies a prior `resume-2000` that is not the file under audit.
- The task statement records `resume-2000c/d/e` as well. Those suffixes are consistent with
  a loop that re-fires, not with a human restarting by hand.

**(b) A trainer was alive and advancing after the resume.**

`delivery/` holds four snapshots of the *same* checkpoint directory at four different steps:

| Snapshot | Step observed |
|---|---|
| `live_snapshot_after_resume_2002/` | 2002 |
| `live_snapshot_after_resume_2021/` | 2021 |
| `live_snapshot_resumed_step_2034/` | 2034 |
| `live_snapshot_after_pipeline_audit_step_2050/` | 2050 |

Monotonic growth 2002 → 2050 across four snapshots is a live optimizer loop, not a replay.

**(c) The resume never completed.**

There is **no** `formal/done_resume.json`. `train_so101.py:548-551` writes that file on the
normal return path of `train_module.main(config)`. Its absence means the process ended by
signal, container kill, or is still running — never by reaching `num_train_steps`.

**(d) Training was interrupted deliberately at least once for evaluation.**

`formal/evaluation_2000/pause/checkpoint_steps_2000.json` exists. Something stopped the
trainer to take a checkpoint read, then `evaluation_2000/state_read/...` ran a *separate*
container (`4de610d1…`, exit 0, 2026-10-30T00:00:09Z).

**(e) The host is co-tenanted.**

`evaluation_2000/state_read/probe_..._summary.json:17-37` records GPU memory *at admission*:

| GPU | used at admission | total |
|---|---|---|
| 0 | 797 MiB | 24,564 MiB |
| 1 | 6,073 MiB | 24,564 MiB |
| 2 | 6,597 MiB | 24,564 MiB |
| 3 | 6,214 MiB | 24,564 MiB |

The evaluator needed ~10 GB/GPU and had to cap the JAX pool at 71 % to fit. Three of four
GPUs already carried ~6 GB of someone else's work. This does **not** prove the Scene1
trainer was that occupant (a 4-way FSDP job would load all four symmetrically, and GPU 0
was empty), but it does prove the box is shared — which is exactly the condition under
which a "wrong container got killed / wrong container got restarted" story becomes credible.

## 3. Why "I stopped it" is not yet a defensible statement

Three separate mechanisms can recreate a trainer, and they are distinguished by different
evidence:

| Mechanism | Signature | Discriminator |
|---|---|---|
| Docker `restart=always/unless-stopped` | container is recreated with the **same** container ID after kill | `docker inspect .HostConfig.RestartPolicy` + `.State.StartedAt` |
| systemd transient unit (`systemd-run`) | **new** unit instance, new `InvocationID`, `ExecMainStartTimestamp` | `systemctl show` |
| External script/cron/agent | new **container**, new ID, same command line | `docker events --since` + shell history |

The `resume-2000b/c/d/e` naming pattern implies the *third* mechanism: something recomputes
a run-id suffix each time. Docker's restart policy cannot rename a run. That points at a
launcher script or an agent, not at Docker — but "points at" is not proof, and this audit
will not upgrade it to one.

## 4. Minimum evidence required to close this

`tools/restart_audit.sh` performs exactly one read-only pass and prints a verdict token:

```
AUTO_RESTART_RESOLVED      # no trainer, no enabled unit/timer, no restart policy, no cron
AUTO_RESTART_UNRESOLVED    # anything found, or any probe failed
```

Required before any checkpoint is trusted:

1. `AUTO_RESTART_RESOLVED` from the script, **twice, 10 minutes apart**.
2. Zero containers whose command line contains `train_so101` or `openpi`.
3. No enabled `resume-*` unit and no `RestartPolicy` other than `no`.
4. The mtime of `formal/checkpoints/2000` unchanged across the two probes — this is the
   only check that proves a live trainer is not silently re-saving over the artifact the
   whole audit depends on.

Point 4 matters more than the other three combined. `save_interval: 1` with
`keep_period: 1` (see `effective_config_audit.json`) means a running trainer can rewrite
step 2000 at any moment unless the milestone gate is what actually governs saves.
