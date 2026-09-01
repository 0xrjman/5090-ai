#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 - NInfer build (.ninfer artifact) - NInfer
# ckpt (HF): neroued/Qwen3.8-27B-nvfp4-NInfer (base Qwen/Qwen3.8-27B via ModelScope)
# refetch:   hf download neroued/Qwen3.8-27B-nvfp4-NInfer --local-dir /home/rjman/models/ninfer/Qwen3.8-27B-nvfp4-NInfer
set -euo pipefail

CONTAINER=ninfer-qwen38-27b
IMAGE=ninfer:latest
MODEL_DIR=/home/rjman/models/ninfer/Qwen3.8-27B-nvfp4-NInfer
MODEL_FILE=/models/qwen3_8_27b_nvfp4.ninfer
PORT=8020
API_KEY=rjman
MODEL_ID=local

# ---- Vision toggle (default ON) -----------------------------------------
# Override per-run without editing:  VISION=0 ./ninfer.sh start
#
# On the 32GB RTX 5090 the KV pool is VRAM-capped, and --max-context must be
# <= that pool (engine reserves full KV for one max-length request). Measured:
#   VISION=1  --vision on   KV pool 188224  -> MAX_CONTEXT=188224 (tested; 200000 OOMs)
#             images/video separately capped at 32768 merged tokens.
#   VISION=0  text-only     KV pool 251392  -> MAX_CONTEXT=251392 (~245K)
#             (vision buffers freed grows the pool; nvfp4 card tops at 262144 w/ MTP off)
# Both modes keep MTP3 speculative decoding.
VISION="${VISION:-1}"
if [ "$VISION" = 1 ]; then
  VISION_FLAG=(--vision); MAX_CONTEXT=188224
else
  VISION_FLAG=();         MAX_CONTEXT=251392
fi

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
    "$IMAGE" ninfer-serve "$MODEL_FILE" \
    --host 0.0.0.0 --port ${PORT} --cors \
    --api-key ${API_KEY} --model-id ${MODEL_ID} \
    --max-context ${MAX_CONTEXT} --kv-capacity auto --kv-dtype int8 \
    --max-concurrency 4 --pending-timeout-ms 90000 \
    "${VISION_FLAG[@]}" \
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