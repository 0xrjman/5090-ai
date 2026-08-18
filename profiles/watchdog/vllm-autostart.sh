#!/usr/bin/env bash
# Watchdog: auto-start vllm-qwen38 only after the GPU is idle (used < IDLE_MIB)
# for IDLE_NEEDED consecutive minutes AND the container is not running.
# Invoked every minute by cron. A state file accumulates consecutive idle minutes
# so the ~30s gap when training boots (vLLM stopped before CUDA allocates) is not
# misread as idle.
# cron: * * * * * /home/rjman/Servers/local-ai/profiles/watchdog/vllm-autostart.sh
set -euo pipefail

PROFILES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SH="$PROFILES_DIR/qwen38-27b/vllm.sh"

LOG=/home/rjman/vllm-autostart.log
NAME=vllm-qwen38
STATE=/home/rjman/.vllm-autostart.idle
IDLE_MIB=500
IDLE_NEEDED=5

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo 0 > "$STATE"
  exit 0
fi

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
used=${used:-999999}

if [ "$used" -lt "$IDLE_MIB" ]; then
  n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$STATE"
  if [ "$n" -ge "$IDLE_NEEDED" ]; then
    log "GPU idle ${used}MiB for ${n}min (>=${IDLE_NEEDED}), starting $NAME"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      docker start "$NAME" >> "$LOG" 2>&1
    else
      bash "$VLLM_SH" start >> "$LOG" 2>&1
    fi
    echo 0 > "$STATE"
  else
    log "GPU idle ${used}MiB (${n}/${IDLE_NEEDED} min), waiting"
  fi
else
  prev=$(cat "$STATE" 2>/dev/null || echo 0)
  [ "$prev" != "0" ] && log "GPU busy (${used}MiB used), reset idle counter (was ${prev})"
  echo 0 > "$STATE"
fi