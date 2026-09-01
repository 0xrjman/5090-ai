#!/usr/bin/env bash
# Watchdog: auto-restart whichever engine (vllm/sglang/ninfer) was last
# started, once the GPU has been idle (used < IDLE_MIB) for IDLE_NEEDED
# consecutive minutes AND none of the known serving containers is running.
# Invoked every minute by cron. A state file accumulates consecutive idle
# minutes so the ~30s gap when training boots (engine stopped before CUDA
# allocates) is not misread as idle.
#
# "Last started" is recorded by each profile script's start() into
# watchdog/.last-engine (values: vllm | ninfer | sglang:main | sglang:longctx).
# Missing/unrecognized state falls back to vllm, matching the old hardcoded
# behavior.
#
# cron: * * * * * $HOME/Servers/local-ai/profiles/watchdog/engine-autostart.sh
set -euo pipefail

PROFILES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAST_ENGINE_FILE="$PROFILES_DIR/watchdog/.last-engine"

LOG=$HOME/engine-autostart.log
STATE=$HOME/.engine-autostart.idle
IDLE_MIB=500
IDLE_NEEDED=5

# engine key -> container name (must match the NAME/CONTAINER each script uses)
KNOWN_CONTAINERS=(vllm-qwen38 sglang-qwen38 ninfer-qwen38-27b)

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

container_for_engine() {
  case "$1" in
    vllm) echo "vllm-qwen38" ;;
    sglang:*) echo "sglang-qwen38" ;;
    ninfer) echo "ninfer-qwen38-27b" ;;
    *) echo "vllm-qwen38" ;;
  esac
}

start_engine() {
  local engine="$1"
  case "$engine" in
    vllm) bash "$PROFILES_DIR/qwen38-27b/vllm.sh" start ;;
    ninfer) bash "$PROFILES_DIR/qwen38-27b/ninfer.sh" start ;;
    sglang:main) bash "$PROFILES_DIR/qwen38-27b/sglang.sh" main start ;;
    sglang:longctx) bash "$PROFILES_DIR/qwen38-27b/sglang.sh" longctx start ;;
    *) bash "$PROFILES_DIR/qwen38-27b/vllm.sh" start ;;
  esac
}

# Any known serving container already up? Nothing to do.
for name in "${KNOWN_CONTAINERS[@]}"; do
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo 0 > "$STATE"
    exit 0
  fi
done

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
used=${used:-999999}

if [ "$used" -lt "$IDLE_MIB" ]; then
  n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$STATE"
  if [ "$n" -ge "$IDLE_NEEDED" ]; then
    engine="$(cat "$LAST_ENGINE_FILE" 2>/dev/null || echo vllm)"
    [ -n "$engine" ] || engine="vllm"
    name="$(container_for_engine "$engine")"
    log "GPU idle ${used}MiB for ${n}min (>=${IDLE_NEEDED}), starting $engine ($name)"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      docker start "$name" >> "$LOG" 2>&1
    else
      start_engine "$engine" >> "$LOG" 2>&1
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
