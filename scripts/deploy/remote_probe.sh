#!/usr/bin/env bash
# 远程体检探针：只读。不拉镜像、不加载模型、不启动容器、不申请 GPU 计算。
# 用法：bash remote_probe.sh <root> <checkpoint> <norm_stats> <lerobot_data>
set -u

ROOT_DIR="${1:-}"
CKPT_DIR="${2:-}"
NORM_STATS="${3:-}"
LEROBOT="${4:-}"

emit() { printf '%s=%s\n' "$1" "${2-}"; }

emit hostname "$(hostname 2>/dev/null)"
emit user "$(whoami 2>/dev/null)"
emit kernel "$(uname -r 2>/dev/null)"
emit nproc "$(nproc 2>/dev/null)"
emit mem_total_gb "$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
emit mem_avail_gb "$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')"
emit python3 "$(python3 -V 2>&1 | tr -d '\n')"

# 远程工作目录
if [ -n "$ROOT_DIR" ]; then
  if mkdir -p "$ROOT_DIR" 2>/dev/null && [ -w "$ROOT_DIR" ]; then
    emit root_writable "yes"
  else
    emit root_writable "no"
  fi
  emit root_exists "$([ -d "$ROOT_DIR" ] && echo yes || echo no)"
  emit root_free_gb "$(df -BG --output=avail "$ROOT_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
  emit root_files "$(find "$ROOT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
else
  emit root_writable "no"
  emit root_exists "no"
  emit root_free_gb ""
  emit root_files "0"
fi

# Docker 镜像（无权限时留空，不报错）
emit docker_ok "$(docker info >/dev/null 2>&1 && echo yes || echo no)"
emit docker_images "$(docker images --format '{{.Repository}}:{{.Tag}}={{.ID}}' 2>/dev/null | tr '\n' ';')"
emit docker_ps "$(docker ps --format '{{.Names}}={{.Image}}' 2>/dev/null | tr '\n' ';')"

# 模型与数据资产
if [ -n "$CKPT_DIR" ]; then
  emit ckpt_exists "$([ -d "$CKPT_DIR" ] && echo yes || echo no)"
  emit ckpt_bytes "$(du -sb "$CKPT_DIR" 2>/dev/null | awk '{print $1}')"
  emit ckpt_files "$(find "$CKPT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
else
  emit ckpt_exists "no"
  emit ckpt_bytes ""
  emit ckpt_files "0"
fi

if [ -n "$NORM_STATS" ] && [ -f "$NORM_STATS" ]; then
  emit norm_stats_sha256 "$(sha256sum "$NORM_STATS" 2>/dev/null | awk '{print $1}')"
else
  emit norm_stats_sha256 ""
fi

if [ -n "$LEROBOT" ]; then
  emit lerobot_exists "$([ -d "$LEROBOT" ] && echo yes || echo no)"
  emit lerobot_bytes "$(du -sb "$LEROBOT" 2>/dev/null | awk '{print $1}')"
else
  emit lerobot_exists "no"
  emit lerobot_bytes ""
fi

# GPU
emit gpu_count "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
emit gpu_list "$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
exit 0
