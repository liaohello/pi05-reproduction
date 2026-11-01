#!/usr/bin/env bash
# Read-only enumeration of every mechanism that could restart the Scene1 trainer.
# Writes nothing. Kills nothing. Prints a single verdict token on the last line.
#
#   bash scene1_root_cause_audit/tools/restart_audit.sh
#
# Verdict:
#   AUTO_RESTART_RESOLVED    nothing found that can recreate a trainer
#   AUTO_RESTART_UNRESOLVED  something found, or a probe could not be completed
set -u
findings=0
probe_failed=0
say() { printf '%s\n' "$*"; }
hit() { findings=$((findings + 1)); say "  !! $*"; }

say "== 1. live trainer processes =="
if command -v ps >/dev/null 2>&1; then
  ps -eo pid,ppid,etime,cmd 2>/dev/null \
    | grep -Ei 'train_so101|openpi/scripts/train|so101.*train' \
    | grep -v "$(basename "$0")" | grep -v grep | sed 's/^/  /' | tee /tmp/.rt_ps.$$
  [ -s /tmp/.rt_ps.$$ ] && hit "trainer process present (see above)"
  rm -f /tmp/.rt_ps.$$
else
  say "  ps unavailable"; probe_failed=1
fi

say "== 2. containers =="
if command -v docker >/dev/null 2>&1; then
  docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null | sed 's/^/  /' \
    | tee /tmp/.rt_dk.$$
  grep -Ei 'scene1|so101|resume|openpi' /tmp/.rt_dk.$$ >/dev/null && hit "candidate container listed above"
  for c in $(docker ps -q 2>/dev/null); do
    pol=$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$c" 2>/dev/null)
    nam=$(docker inspect -f '{{.Name}}' "$c" 2>/dev/null)
    [ "${pol:-no}" != "no" ] && hit "container $nam has RestartPolicy=$pol"
  done
  rm -f /tmp/.rt_dk.$$
else
  say "  docker unavailable (expected on the control host, not on the GPU host)"; probe_failed=1
fi

say "== 3. systemd transient units and timers =="
if command -v systemctl >/dev/null 2>&1; then
  systemctl list-units --all --no-pager 2>/dev/null \
    | grep -Ei 'resume-|so101|scene1|openpi' | sed 's/^/  /' \
    | tee /tmp/.rt_sd.$$
  [ -s /tmp/.rt_sd.$$ ] && hit "matching unit(s) present"
  systemctl list-timers --all --no-pager 2>/dev/null | sed 's/^/  /' | tee /tmp/.rt_tm.$$
  grep -Ei 'resume-|so101|scene1' /tmp/.rt_tm.$$ >/dev/null && hit "matching timer present"
  rm -f /tmp/.rt_sd.$$ /tmp/.rt_tm.$$
else
  say "  systemctl unavailable"; probe_failed=1
fi

say "== 4. cron / supervisor / agent controllers =="
crontab -l 2>/dev/null | grep -Ei 'so101|scene1|openpi|train' | sed 's/^/  /' \
  && hit "crontab entry matches" || say "  no matching crontab entry"
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl status 2>/dev/null | grep -Ei 'so101|train' | sed 's/^/  /' \
    && hit "supervisor program matches" || say "  no matching supervisor program"
fi
ps -eo cmd 2>/dev/null | grep -Ei 'codex|scheduler|autoresume' | grep -v grep | sed 's/^/  /' \
  | tee /tmp/.rt_ag.$$
grep -Eiq 'resume|so101' /tmp/.rt_ag.$$ && hit "controller process mentions resume/so101"
rm -f /tmp/.rt_ag.$$

say "== 5. checkpoint mtime (run twice, 10 min apart) =="
CKPT="${1:-runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/formal/checkpoints}"
if [ -d "$CKPT" ]; then
  find "$CKPT" -maxdepth 2 -name '*.orbax-checkpoint-tmp' -newermt '-15 minutes' 2>/dev/null \
    | sed 's/^/  in-flight tmp: /' | tee /tmp/.rt_tmp.$$
  [ -s /tmp/.rt_tmp.$$ ] && hit "a checkpoint is being written right now"
  stat -c '  %y  %n' "$CKPT"/* 2>/dev/null | sort | tail -5
  rm -f /tmp/.rt_tmp.$$
else
  say "  checkpoint path not found locally; run this on the GPU host"; probe_failed=1
fi

say ""
if [ "$findings" -eq 0 ] && [ "$probe_failed" -eq 0 ]; then
  say "AUTO_RESTART_RESOLVED"
else
  say "AUTO_RESTART_UNRESOLVED  (findings=$findings probe_failed=$probe_failed)"
fi
