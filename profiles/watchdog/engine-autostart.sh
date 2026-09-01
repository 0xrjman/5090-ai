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
# It also guards the ninfer wedge while ninfer is up: engine_core.h worker_loop()
# catches any exception, calls fail_all_locked() and RETURNS, so the worker thread
# is gone but the process lives on. Nothing external notices -- the port listens,
# docker says Up, --restart never fires, and /v1/models still returns 200 because
# it never touches the engine. Only a real generation request sees it, as 503.
#
# cron: * * * * * $HOME/Servers/local-ai/profiles/watchdog/engine-autostart.sh
set -euo pipefail

PROFILES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAST_ENGINE_FILE="$PROFILES_DIR/watchdog/.last-engine"

LOG=$HOME/engine-autostart.log
STATE=$HOME/.engine-autostart.idle
WEDGE_STATE=$HOME/.engine-autostart.wedge
IDLE_MIB=500
IDLE_NEEDED=5
WEDGE_NEEDED=2
NINFER_PORT=8020

# repo-root .env supplies API_KEY for the wedge probe
if [ -f "$PROFILES_DIR/../.env" ]; then set -a; . "$PROFILES_DIR/../.env"; set +a; fi

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

# ninfer wedge probe. /v1/models is useless here (it never reaches the engine), so
# this has to be a real generation request. 503 service_unavailable is unambiguous:
# a saturated queue returns 429, not 503.
#
# The probe must stop on its own (thinking off, prompt with a two-token answer).
# Capping it with max_tokens:1 instead ends every probe in output_limit, which
# leaves a catalogued continuation checkpoint behind once a minute -- extra churn
# on the exact continuation/materialization path that wedges the engine.
check_ninfer_wedge() {
  local body
  body=$(curl -s --max-time 30 \
    -H 'content-type: application/json' -H "x-api-key: ${API_KEY:-}" \
    -d '{"model":"local","max_tokens":64,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' \
    "http://127.0.0.1:$NINFER_PORT/v1/messages" 2>/dev/null || true)
  case "$body" in
    *service_unavailable*|*"engine is unavailable"*) ;;
    *) echo 0 > "$WEDGE_STATE"; return ;;
  esac
  local n=$(( $(cat "$WEDGE_STATE" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$WEDGE_STATE"
  if [ "$n" -lt "$WEDGE_NEEDED" ]; then
    log "ninfer wedge probe failed (${n}/${WEDGE_NEEDED})"
    return
  fi
  log "ninfer wedged -- saving log tail and restarting"
  mkdir -p "$HOME/ninfer-crash"
  docker logs --tail 400 ninfer-qwen38-27b \
    > "$HOME/ninfer-crash/wedge-$(date +%Y%m%dT%H%M%S).log" 2>&1 || true
  docker restart ninfer-qwen38-27b >> "$LOG" 2>&1 || true
  echo 0 > "$WEDGE_STATE"
  log "ninfer restarted"
}

# Any known serving container already up? Then autostart has nothing to do -- but
# ninfer being "up" is not the same as ninfer working, so probe it.
for name in "${KNOWN_CONTAINERS[@]}"; do
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo 0 > "$STATE"
    [ "$name" = ninfer-qwen38-27b ] && check_ninfer_wedge
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
