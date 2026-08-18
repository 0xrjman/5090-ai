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
  docker run -d --name "$CONTAINER" --restart unless-stopped \
    --runtime=nvidia --gpus all \
    -p ${PORT}:${PORT} \
    -v ${MODEL_DIR}:/models:ro,z \
    "$IMAGE" ninfer-serve "$MODEL_FILE" \
    --host 0.0.0.0 --port ${PORT} --cors \
    --api-key ${API_KEY} --model-id ${MODEL_ID} \
    --max-context 200000 --kv-capacity auto --kv-dtype int8 \
    --max-concurrency 4 \
    --spec mtp --draft-tokens 3 --lm-head-draft
  echo "started, tail logs with: $0 logs"
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