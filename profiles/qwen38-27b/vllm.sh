#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 - unsloth build - vLLM
# ckpt (HF): unsloth/Qwen3.8-27B-NVFP4 (base: Qwen/Qwen3.8-27B)
# refetch:   hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir /home/rjman/data/models/qwen3.8-27b-nvfp4-unsloth
#
# draft (HF): z-lab/Qwen3.8-27B-DFlash2 (block-diffusion drafter, for SPEC_METHOD=dflash)
# refetch:    HF_ENDPOINT=https://hf-mirror.com hf download z-lab/Qwen3.8-27B-DFlash2 \
#               --local-dir /home/rjman/data/models/qwen3.8-27b-dflash2
set -euo pipefail
NAME=vllm-qwen38
IMG=vllm/vllm-openai:v0.27.1
MODEL=/home/rjman/data/models/qwen3.8-27b-nvfp4-unsloth
PORT=8020

SPEC_METHOD="${SPEC_METHOD:-dflash}"   # dflash (default) | mtp
DFLASH_DIR="${DFLASH_DIR:-/home/rjman/data/models/qwen3.8-27b-dflash2}"
DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-8}"

if [ "$SPEC_METHOD" = "dflash" ] && [ ! -d "$DFLASH_DIR" ]; then
  echo "WARN: SPEC_METHOD=dflash but $DFLASH_DIR is missing -- falling back to mtp." >&2
  echo "      download first: HF_ENDPOINT=https://hf-mirror.com hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir $DFLASH_DIR" >&2
  SPEC_METHOD=mtp
fi

DFLASH_MOUNT=()
case "$SPEC_METHOD" in
  dflash)
    SPEC_CONFIG="{\"method\": \"dflash\", \"model\": \"/draft\", \"num_speculative_tokens\": ${DFLASH_NUM_SPEC}}"
    DFLASH_MOUNT=(-v "$DFLASH_DIR":/draft:ro)
    ;;
  mtp)
    SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": 2}'
    ;;
  *)
    echo "unknown SPEC_METHOD: $SPEC_METHOD (expected dflash|mtp)" >&2
    exit 1
    ;;
esac

start() {
  docker ps -a --format '{{.Names}}' | grep -qx "$NAME" && docker rm -f "$NAME" >/dev/null
  docker run -d --name "$NAME" --runtime=nvidia --gpus all --restart=always \
    -p "$PORT":8000 --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_NO_USAGE_STATS=1 \
    -v "$MODEL":/models \
    "${DFLASH_MOUNT[@]}" \
    "$IMG" \
    /models \
    --served-model-name local \
    --api-key rjman \
    --max-model-len 160000 \
    --gpu-memory-utilization 0.98 \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --performance-mode interactivity \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config "$SPEC_CONFIG"
  echo "started: $NAME (port $PORT, api-key: rjman, restart=always, spec=$SPEC_METHOD)"
  echo "log:   bash $0 logs"
  echo "stop:  bash $0 stop"
  _profiles_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  echo "vllm" > "$_profiles_dir/watchdog/.last-engine" 2>/dev/null || true
  _dash="$_profiles_dir/dashboard/dashboard.sh"
  [ -f "$_dash" ] && bash "$_dash" start || true
}

stop()   { docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped: $NAME" || echo "not running"; }
status() { docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}} {{.Ports}}'; }
logs()   { docker logs -f "$NAME"; }

case "${1:-start}" in
  start|stop|status|logs) "${1:-start}" ;;
  *) echo "usage: $0 [start|stop|status|logs]"; exit 1 ;;
esac