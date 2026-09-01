#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 - RadixArk build - SGLang
# ckpt (HF): RadixArk (base: Qwen/Qwen3.8-27B) [confirm exact repo id on HF]
# refetch:   hf download <RadixArk/Qwen3.8-27B-NVFP4> --local-dir $HOME/models/qwen3.8-27b-nvfp4-radixark
#
# draft (HF): z-lab/Qwen3.8-27B-DFlash2 (block-diffusion drafter, for SPEC_METHOD=dflash -- BROKEN, see profiles/README.md)
# refetch:    HF_ENDPOINT=https://hf-mirror.com hf download z-lab/Qwen3.8-27B-DFlash2 \
#               --local-dir $HOME/models/qwen3.8-27b-dflash2
#
# MTP (SPEC_METHOD=mtp): uses the in-checkpoint MTP head, no separate draft
# download -- this is sglang's official recipe for this checkpoint per
# https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
#
# scenarios (mutually exclusive, one 27B fits the 32GB):
#   main    concurrency=2  --max-mamba-cache-size 8
#   longctx concurrency=1  --max-mamba-cache-size 4  (single session, max context)
# usage: sglang.sh [main|longctx] [start|stop|status|logs]   (default: main)
set -euo pipefail
# load repo-root .env (gitignored) — real API key etc.
_sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_sdir/../../.env" ]; then set -a; . "$_sdir/../../.env"; set +a; fi
NAME=sglang-qwen38
IMG=lmsysorg/sglang:qwen38-27b
MODEL=$HOME/models/qwen3.8-27b-nvfp4-radixark
PORT=8020
API_KEY="${API_KEY:-}"
API_ARGS=()
if [ -n "$API_KEY" ]; then API_ARGS+=(--api-key "$API_KEY"); fi

SPEC_METHOD="${SPEC_METHOD:-mtp}"   # mtp (default) | none | dflash (broken, see profiles/README.md)
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/qwen3.8-27b-dflash2}"
DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-8}"

if [ "$SPEC_METHOD" = "dflash" ] && [ ! -d "$DFLASH_DIR" ]; then
  echo "WARN: SPEC_METHOD=dflash but $DFLASH_DIR is missing -- falling back to no speculative decoding." >&2
  echo "      download first: HF_ENDPOINT=https://hf-mirror.com hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir $DFLASH_DIR" >&2
  SPEC_METHOD=none
fi

DFLASH_MOUNT=()
SPEC_ARGS=()
MAMBA_CACHE_STRATEGY=extra_buffer_lazy
case "$SPEC_METHOD" in
  dflash)
    DFLASH_MOUNT=(-v "$DFLASH_DIR":/draft:ro)
    SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-num-draft-tokens "$DFLASH_NUM_SPEC")
    MAMBA_CACHE_STRATEGY=extra_buffer   # DFLASH doesn't support extra_buffer_lazy
    ;;
  mtp)
    SPEC_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec)
    ;;
  none) ;;
  *)
    echo "unknown SPEC_METHOD: $SPEC_METHOD (expected mtp|none|dflash)" >&2
    exit 1
    ;;
esac

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
    "${DFLASH_MOUNT[@]}" \
    "$IMG" \
    sglang serve \
      --trust-remote-code \
      --model-path /models \
      --mem-fraction-static 0.95 \
      --attention-backend flashinfer \
      --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy "$MAMBA_CACHE_STRATEGY" \
      --max-mamba-cache-size "$MAMBA_CACHE" \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      "${API_ARGS[@]}" \
      --served-model-name local \
      --host 0.0.0.0 --port 30000 \
      "${SPEC_ARGS[@]}"
  echo "started: $NAME (port $PORT, auth=$([ -n "$API_KEY" ] && echo on || echo off), model=local, $SCENARIO_NOTE, concurrency=$CONCURRENT, spec=$SPEC_METHOD)"
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