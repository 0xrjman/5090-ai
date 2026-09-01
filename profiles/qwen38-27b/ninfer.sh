#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 - NInfer build (.ninfer artifact) - NInfer
# ckpt (HF): neroued/Qwen3.8-27B-nvfp4-NInfer (base Qwen/Qwen3.8-27B via ModelScope)
# refetch:   hf download neroued/Qwen3.8-27B-nvfp4-NInfer --local-dir $HOME/models/ninfer/Qwen3.8-27B-nvfp4-NInfer
set -euo pipefail
# load repo-root .env (gitignored) — real API key etc.
_sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_sdir/../../.env" ]; then set -a; . "$_sdir/../../.env"; set +a; fi

CONTAINER=ninfer-qwen38-27b
IMAGE=ninfer:latest
MODEL_DIR=$HOME/models/ninfer/Qwen3.8-27B-nvfp4-NInfer
MODEL_FILE=/models/qwen3_8_27b_nvfp4.ninfer
PORT=8020
# The stdout log drops the exception text of an engine failure (append_failure_fields
# in serve/operational_log.cpp never emits machine_message), so a wedge is otherwise
# unattributable. The JSONL sink is the only place it survives.
JSONL_DIR="${JSONL_DIR:-$HOME/ninfer-logs}"
API_KEY="${API_KEY:-}"
API_ARGS=()
if [ -n "$API_KEY" ]; then API_ARGS+=(--api-key "$API_KEY"); fi
MODEL_ID=local
# nvfp4 (4-bit group-16) | k8v4 (K fp8 + V nvfp4) | fp8 (E4M3 row-256) | int8 (group-64) | bf16
# Bytes per token per head (K+V): nvfp4 288, k8v4 402, fp8 516, int8 528, bf16 1024.
# nvfp4 is the default for pool capacity, not speed: decode is only ~5% faster
# and prefill ~7% slower than fp8, but the 1.79x pool keeps a multi-session
# working set resident, and an evicted session costs a full re-prefill (~45 s at
# 150K) against ~1.5 s on a prefix-cache hit. See profiles/README.md for the A/B.
KV_DTYPE="${KV_DTYPE:-nvfp4}"
HOST_KV_MIB="${HOST_KV_MIB:-24576}"   # pinned-CPU KV arena for long-conv spill; 24 GiB holds ~500K tokens of fp8 KV, ~900K of nvfp4
DEVICE_STATE_SLOTS="${DEVICE_STATE_SLOTS:-4}"   # device checkpoint slots (pinned; engine default is +C)
HOST_STATE_SLOTS="${HOST_STATE_SLOTS:-8}"       # host checkpoint slots (engine default)

# ---- (VISION, KV_DTYPE) -> MAX_CONTEXT -----------------------------------
# Override per-run without editing. The login shell here is fish, which has no
# `VAR=value cmd` prefix syntax, so use env:
#   env VISION=0 bash ninfer.sh start
#   env VISION=0 KV_DTYPE=fp8 bash ninfer.sh start
#
# The device KV pool is VRAM-capped and depends on BOTH knobs, so --max-context
# has to be chosen per pair and stay <= that pair's own pool (the engine reserves
# a full max-length request's KV). Pools measured with --kv-capacity auto on a
# 32GB RTX 5090, ninfer 21a0e85f, MTP3 on:
#   vision on  + fp8     pool 229376  ->  MAX_CONTEXT=188224  (82%)
#   vision on  + nvfp4   pool 410944  ->  MAX_CONTEXT=262144  (64%)
#   vision off + fp8     pool 257920  ->  MAX_CONTEXT=251392  (97%)
#   vision off + nvfp4   pool 462144  ->  MAX_CONTEXT=400000  (87%)
# Pool tokens scale exactly with the KV byte width, so a new dtype's pool is
# predictable to <0.01% -- but measure it before adding a row here anyway.
# Vision on additionally caps images/video at 32768 merged tokens; turning it
# off buys ~12% more pool (available-after-weights 10.82 -> 11.10 GiB), not the
# whole media reservation. All pairs keep MTP3 speculative decoding.
VISION="${VISION:-1}"
case "$VISION:$KV_DTYPE" in
  1:fp8)   VISION_FLAG=(--vision); MAX_CONTEXT=188224 ;;
  1:nvfp4) VISION_FLAG=(--vision); MAX_CONTEXT=262144 ;;
  0:fp8)   VISION_FLAG=();         MAX_CONTEXT=251392 ;;
  0:nvfp4) VISION_FLAG=();         MAX_CONTEXT=400000 ;;
  *) echo "no measured KV pool for VISION=$VISION KV_DTYPE=$KV_DTYPE;" \
          "read kv_capacity_tokens from a trial start and add a row" >&2; exit 1 ;;
esac
# Re-inject prior-turn reasoning into later prompts (keeps agent long-conversations
# coherent across turns). No effect on the API response shape (reasoning is always
# a separate reasoning_content field); 0 to disable.
PRESERVE_THINKING="${PRESERVE_THINKING:-1}"
PRESERVE_FLAG=()
if [ "$PRESERVE_THINKING" = 1 ]; then PRESERVE_FLAG=(--preserve-thinking); fi

action="${1:-status}"

start() {
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "already running"
    return
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  echo "stopping sglang-qwen38 / vllm-qwen38 (VRAM+port conflict)..."
  docker stop sglang-qwen38 >/dev/null 2>&1 || true
  docker stop vllm-qwen38 >/dev/null 2>&1 || true
  # Caveat at the max-context ceiling: one max-length request fills the whole KV
  # pool, so no room for a second concurrent long request (short requests still
  # multiplex). MTP draft tokens share the pool; near the tail drafting may fall
  # back to plain decode (harmless, just loses spec speedup there).
  echo "vision=$VISION  max-context=$MAX_CONTEXT"
  docker run -d --name "$CONTAINER" --restart unless-stopped \
    --runtime=nvidia --gpus all \
    -p ${PORT}:${PORT} \
    -v ${MODEL_DIR}:/models:ro,z \
    -v ${JSONL_DIR}:/reqlog:z \
    "$IMAGE" ninfer-serve "$MODEL_FILE" \
    --host 0.0.0.0 --port ${PORT} --cors \
    --request-log-jsonl /reqlog/requests.jsonl \
    "${API_ARGS[@]}" --model-id ${MODEL_ID} \
    --max-context ${MAX_CONTEXT} --kv-capacity auto --kv-dtype ${KV_DTYPE} \
    --max-concurrency 4 --pending-timeout-ms 90000 --host-kv-mib ${HOST_KV_MIB} \
    --device-state-slots ${DEVICE_STATE_SLOTS} --host-state-slots ${HOST_STATE_SLOTS} \
    "${VISION_FLAG[@]}" \
    "${PRESERVE_FLAG[@]}" \
    --spec mtp --draft-tokens 3 --lm-head-draft
  echo "started, tail logs with: $0 logs"
  # resolve symlink first: when invoked via a symlink (e.g. ~/.local/bin/start-ninfer.sh),
  # BASH_SOURCE is the link itself, so dirname/.. would mis-resolve away from profiles/
  _self="$(readlink -f "${BASH_SOURCE[0]}")"
  _profiles_dir="$(cd "$(dirname "$_self")/.." && pwd)"
  echo "ninfer" > "$_profiles_dir/watchdog/.last-engine" 2>/dev/null || true
  _dash="$_profiles_dir/dashboard/dashboard.sh"
  [ -f "$_dash" ] && bash "$_dash" start || true
}

stop() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  echo "stopped. sglang-qwen38/vllm-qwen38 left stopped -- restart yourself if needed: docker start sglang-qwen38 | docker start vllm-qwen38"
}

status() {
  docker ps -a --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo "---"
  nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
}

logs() {
  docker logs -f --tail 100 "$CONTAINER"
}

case "$action" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  *) echo "usage: $0 {start|stop|status|logs}"; exit 1 ;;
esac