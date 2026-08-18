# local-ai

Local LLM serving for a single RTX 5090. One model — **Qwen3.8-27B-NVFP4** —
served by three interchangeable inference frameworks, all on one port (8020).

## Matrix

Each profile pairs a checkpoint build with an inference framework. They all run
on the same GPU and port, so they are **mutually exclusive** (starting one stops
the siblings).

| ckpt build (source)              | vLLM      | SGLang                                        | NInfer    |
|----------------------------------|-----------|-----------------------------------------------|-----------|
| Qwen3.8-27B-NVFP4 · **unsloth**  | `vllm.sh` | —                                             | —         |
| Qwen3.8-27B-NVFP4 · **radixark** | —         | `sglang.sh main` / `sglang.sh longctx`        | —         |
| Qwen3.8-27B-NVFP4 · **NInfer**   | —         | —                                             | `ninfer.sh`|

## Usage

```bash
cd profiles/qwen38-27b

./vllm.sh   [start|stop|status|logs]                 # vLLM (unsloth build)
./sglang.sh [main|longctx] [start|stop|status|logs]  # SGLang (radixark build)
./ninfer.sh [start|stop|status|logs]                 # NInfer (.ninfer build)
```

- `sglang.sh` defaults to scenario `main` (concurrency 2). Use `longctx` for a
  single long-context session (concurrency 1).
- OpenAI-compatible API: `http://localhost:8020/v1`, model `local`, api-key `rjman`.

## Watchdog

A per-minute cron job restarts the vLLM server once the GPU has been idle for 5
consecutive minutes, and backs off while anything else is using the GPU
(`profiles/watchdog/vllm-autostart.sh`).

## Checkpoints

Weights are large (~20 GB each) and are **not** in this repo. Each profile script
documents its source (Hugging Face / ModelScope) and the exact `hf download`
command in its header.

## Layout

```
local-ai/
  profiles/          active serving matrix (this box, single 5090)
    qwen38-27b/      vllm.sh · sglang.sh · ninfer.sh
    watchdog/        vllm-autostart.sh
  _archive/          the previous B200 / 8×GPU declarative serving system, intact
```