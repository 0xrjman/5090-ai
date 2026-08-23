# profiles

One directory per model; one script per inference framework. Each script is
self-contained (`start`/`stop`/`status`/`logs`) and documents its checkpoint
source in the header.

## Convention

```
profiles/<model>/<framework>.sh
```

All profiles on a machine share one GPU and (by default) one port, so they are
mutually exclusive — a profile's `start` should stop the siblings that would
conflict.

## Dashboard (shared across profiles)

`profiles/dashboard/` is a background monitor that tails the serving container's
**real docker logs** and renders live metrics at **http://localhost:8021/**:
decode/prefill TPS (now/avg/peak), per-request **concurrent streams** with a
color-coded pressure bar (elapsed time), a concurrency-over-time chart, request
aggregates, the live queue + errors, and a raw log tail. NInfer log parsing is
implemented; vLLM/SGLang show their raw log and can be added via a parser in
`dashboard.py::PARSERS`.

Every profile's `start` auto-launches (or overwrite-restarts) it, so it is always
running current code while a server is up:

```bash
profiles/dashboard/dashboard.sh [start|stop|status|logs]
```

## Adding a new cell

1. Add a framework script next to the siblings, following the same
   `start/stop/status/logs` shape.
2. Record the checkpoint provenance in the header: the source (HF / ModelScope
   repo), the local path, and the `hf download` command.
3. End `start` with the dashboard auto-launch (so it starts/refreshes on every
   boot), exactly as the siblings do:
   ```bash
   _dash="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dashboard/dashboard.sh"
   [ -f "$_dash" ] && bash "$_dash" start || true
   ```
4. Add the row/column to the matrix in the root `README.md`.

## Notes (measured)

- SGLang + DSpark speculative decoding on this 5090 deployment gave **no** decode
  throughput gain (77 vs 78 tok/s, within noise) and shrank the KV pool, so DSpark
  is off in both scenarios. Measured pure-decode throughput is ~77-78 tok/s
  (2026-08-15, RadixArk Qwen3.8-27B-NVFP4). Re-check only after
  sglang-project/sglang PR #34742 lands and the image is repulled.
- **`vllm.sh` DFlash2 is broken, do not default to it.** Measured 2026-08-19 on
  `vllm/vllm-openai:v0.27.1` with the real `z-lab/Qwen3.8-27B-DFlash2` weights
  downloaded and mounted: vLLM rejects the checkpoint at startup with
  `pydantic_core.ValidationError: Model architectures ['DFlash2DraftModel'] are
  not supported for now` and crash-loops under `--restart=always`. This is not
  a missing-weights problem — the draft architecture just isn't in this vLLM
  build's model registry. `vllm.sh` now defaults to `SPEC_METHOD=mtp`;
  `SPEC_METHOD=dflash` is left switchable only in case a future vLLM image adds
  support — don't flip the default back without re-testing on the actual image.
- **`sglang.sh` DFlash2 is also broken, do not default to it.** Measured
  2026-08-19 on `lmsysorg/sglang:qwen38-27b` (digest
  `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`)
  with the real `z-lab/Qwen3.8-27B-DFlash2` weights mounted: after fixing an
  unrelated startup assertion (`--mamba-radix-cache-strategy extra_buffer_lazy`
  is incompatible with `--speculative-algorithm DFLASH`; must be
  `extra_buffer` — `sglang.sh` now sets this conditionally), sglang crashes
  while loading the draft model with
  `ValueError: Cannot find model module. 'DFlash2DraftModel' is not a
  registered model in the Transformers library ... and 'AutoModel' is not
  present in the model config's 'auto_map'`, then crash-loops under
  `--restart=always`. Same root cause category as the vLLM failure above: the
  draft checkpoint's `config.json` declares `architectures:
  ["DFlash2DraftModel"]` with no `auto_map` and no `modeling_*.py` in the repo
  (`/home/rjman/models/qwen3.8-27b-dflash2` only has `config.json` +
  `model.safetensors`), so it relies entirely on the inference engine having
  that exact class built into its model registry — and this sglang build
  doesn't, despite accepting the `--speculative-algorithm DFLASH` flag itself.
  **Conclusion: DFlash2 for this checkpoint is deprecated on this box — do not
  use it.** Two independent reasons:
  1. *Startup (measured above):* it doesn't even load on this vLLM or sglang
     build because the draft's `DFlash2DraftModel` isn't in the engine's model
     registry.
  2. *VRAM (the primary deprecation reason):* even where the draft model does
     load, the DFlash2 draft occupies VRAM that would otherwise go to the KV
     cache, so usable context shrinks too much to be practical. That context
     loss makes it a weak fit for this single-27B-on-32GB box — low
     usability, so it's abandoned for now.
  `sglang.sh` now defaults to `SPEC_METHOD=mtp` (the in-checkpoint MTP head, no
  separate draft download, so no draft-model VRAM cost); `SPEC_METHOD=dflash`
  is left switchable only in case a future sglang image registers this
  architecture — don't flip the default back without re-testing on the actual
  image. Weights live at
  `/home/rjman/data/models/qwen3.8-27b-dflash2` (real files) with
  `/home/rjman/models/qwen3.8-27b-dflash2` symlinked to the same dir.