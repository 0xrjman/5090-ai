#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 - RadixArk build - SGLang
# ckpt (HF): RadixArk (base: Qwen/Qwen3.8-27B) [confirm exact repo id on HF]
# refetch:   hf download <RadixArk/Qwen3.8-27B-NVFP4> --local-dir /home/rjman/models/qwen3.8-27b-nvfp4-radixark
#
# scenarios (mutually exclusive, one 27B fits the 32GB):
#   main    concurrency=2  --max-mamba-cache-size 8
#   longctx concurrency=1  --max-mamba-cache-size 4  (single session, max context)
# usage: sglang.sh [main|longctx] [start|stop|status|logs]   (default: main)
set -euo pipefail
NAME=sglang-qwen38
IMG=lmsysorg/sglang:qwen38-27b
MODEL=/home/rjman/models/qwen3.8-27b-nvfp4-radixark
PORT=8020

SCENARIO=""
case "${1:-}" in
  main|longctx) SCENARIO="$1"; shift ;;
esac
[ -n "$SCENARIO" ] || SCENARIO=main

if [ "$SCENARIO" = "longctx" ]; then
  MAMBA_CACHE=4; CONCURRENT=1; SCENARIO_NOTE="long-context"
else
  MAMBA_CACHE=8; CONCURRENT=2; SCENARIO_NOTE="main coding"
fi

start() {
  docker ps -a --format '{{.Names}}' | grep -qx "$NAME" && docker rm -f "$NAME" >/dev/null
  docker run -d --name "$NAME" --gpus all --restart=always \
    --shm-size 32g --ipc=host \
    -p 0.0.0.0:${PORT}:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$MODEL":/models:ro \
    "$IMG" \
    sglang serve \
      --trust-remote-code \
      --model-path /models \
      --mem-fraction-static 0.95 \
      --attention-backend flashinfer \
      --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --max-mamba-cache-size "$MAMBA_CACHE" \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --api-key rjman \
      --served-model-name local \
      --host 0.0.0.0 --port 30000
  echo "started: $NAME (port $PORT, api-key=rjman, model=local, $SCENARIO_NOTE, concurrency=$CONCURRENT)"
  _profiles_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  echo "sglang:$SCENARIO" > "$_profiles_dir/watchdog/.last-engine" 2>/dev/null || true
  _dash="$_profiles_dir/dashboard/dashboard.sh"
  [ -f "$_dash" ] && bash "$_dash" start || true
}

stop()   { docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped: $NAME" || echo "not running"; }
status() { docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}} {{.Ports}}'; }
logs()   { docker logs -f "$NAME"; }

case "${1:-start}" in
  start|stop|status|logs) "${1:-start}" ;;
  *) echo "usage: $0 [main|longctx] [start|stop|status|logs]"; exit 1 ;;
esac