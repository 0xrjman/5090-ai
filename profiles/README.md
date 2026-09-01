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

- Measured decode throughput (2026-09-01, ninfer with MTP speculative decoding on):
  per-request up to ~221 tok/s (avg ~150); ~406 tok/s system-wide with 3 concurrent
  streams. Supersedes the 2026-08-15 SGLang figure (~77-78 tok/s).
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
  (`$HOME/models/qwen3.8-27b-dflash2` only has `config.json` +
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
  `$HOME/data/models/qwen3.8-27b-dflash2` (real files) with
  `$HOME/models/qwen3.8-27b-dflash2` symlinked to the same dir.
## KV cache: `fp8` vs `nvfp4`

Qwen3.8-27B-NVFP4 on a single RTX 5090 32 GB, ninfer build `21a0e85f`
(`feat(kv-cache): use fp16 V storage and PV compute`), which is the commit that
first ships the `nvfp4` and `k8v4` KV modes.

### Storage layout

Bytes per token per head, read from `src/core/paged_kv_storage.h`
(`paged_kv_storage_layout`, `head_dim == 256`). `fp8` and `nvfp4` are
`symmetric` — the same layout for K and V; `k8v4` is the only split mode.

| `--kv-dtype` | K | V | K+V | vs `fp8` | Device KV pool, `--kv-capacity auto --vision` |
| --- | --- | --- | --- | --- | --- |
| `bf16` | 512 B | 512 B | 1024 B | 0.50x | ~115K (derived) |
| `int8` | 264 B | 264 B | 528 B | 0.98x | ~224K (derived) |
| `fp8` | 258 B | 258 B | 516 B | 1.00x | **229376 (measured)** |
| `k8v4` | 258 B | 144 B | 402 B | 1.28x | ~294K (derived) |
| `nvfp4` | 144 B | 144 B | 288 B | 1.79x | **410944 (measured)** |

The derived pool figures scale the measured `fp8` pool by the byte ratio. That
model predicted `nvfp4` at 410965 against a measured 410944 — a 0.005% error, so
the `k8v4` row is trustworthy without a separate run.

### Throughput A/B

Identical flags except `--kv-dtype`: `--max-context 188224 --kv-capacity auto
--max-concurrency 4 --host-kv-mib 24576 --device-state-slots 4
--host-state-slots 8 --vision --preserve-thinking --spec mtp --draft-tokens 3
--lm-head-draft`. Single request at a time (no concurrency), cold prefix
(`prefix_reuse_path=root`, `prefix_cache_hit_tokens=0`), 1400-1900 completion
tokens per request, thinking disabled.

TTFT is Time To First Token (`ttft_ms`); prefill and decode are
`prefill_tokens_per_second` / `decode_tokens_per_second`.

| Prompt tokens | TTFT `fp8` | TTFT `nvfp4` | Prefill `fp8` | Prefill `nvfp4` | Decode `fp8` | Decode `nvfp4` | Decode delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ~28.8K | 3.77 s | 3.82 s | 7746 tok/s | 7568 tok/s | 119.8 tok/s | 128.9 tok/s | +7.6% |
| ~62.3K | 10.35 s | 10.84 s | 6031 tok/s | 5759 tok/s | 118.8 tok/s | 119.4 tok/s | +0.5% |
| ~124.4K | 29.62 s | 31.72 s | 4206 tok/s | 3929 tok/s | 110.8 tok/s | 119.2 tok/s | +7.6% |
| ~172.2K | 50.57 s | 54.51 s | 3415 tok/s | 3163 tok/s | 103.0 tok/s | 108.6 tok/s | +5.4% |
| **29K -> 172K** | | | **-56%** | **-58%** | **-14.0%** | **-15.8%** | |

Peak VRAM is identical in both modes (30.8 GiB of 31.8 GiB): `--kv-capacity
auto` spends the same byte budget either way and simply fits more tokens into
it.

### What the numbers mean

- **`nvfp4` does not move the decode degradation knee outward.** It lifts the
  whole curve by roughly 5%, but the slope is the same (`fp8` -14.0%, `nvfp4`
  -15.8% from 29K to 172K). Expecting a 1.79x shift in where decode falls off is
  wrong.
- **Decode is dominated by weight traffic, not KV traffic.** Every decode round
  reads the full 20 GiB of weights regardless of context length. KV at 172K is
  ~7.3 GiB (`fp8`) / ~4.1 GiB (`nvfp4`), so KV is only ~27% of the bytes moved;
  halving it can move the total by ~13% at best, and dequant overhead eats part
  of that.
- **`nvfp4` costs prefill.** It is consistently slower to prefill and the gap
  widens with context (-2.3% at 29K, -7.4% at 172K). At 172K that is 50.6 s ->
  54.5 s of TTFT.
- **Pick `nvfp4` for capacity, not for speed.** The 1.79x pool (410944 vs 229376
  tokens) is what lets a multi-session working set stay resident. An evicted
  session pays a full re-prefill on its next turn (`reuse=root`, ~45 s at 150K)
  against ~1.5 s for a prefix-cache hit — three orders of magnitude larger than
  the +-7% micro-differences above. Sizing the pool so sessions are never
  evicted is worth far more than either the decode gain or the prefill loss.
- **Host KV scales the same way.** `--host-kv-mib` holds page replicas in the
  same layout, so `nvfp4` needs 1.79x fewer host bytes for the same working set.
  A 400K-token working set needs ~19 GiB under `fp8` but only ~11 GiB under
  `nvfp4`; the 24576 default has ample headroom in both.

### Observed on real agent traffic

The A/B above is a synthetic single-request benchmark. Live Claude Code traffic
behaves differently in ways worth recording. Sample: `nvfp4`, `--vision`,
`--max-context 262144`, 24 requests (mostly `finish_reason=tool_calls`), prompts
from 15 to 82565 tokens, 57 throughput samples, **zero error lines**.

**Decode does not degrade with context length in real traffic.**

| Prompt tokens | n | Median | Mean | Range |
| --- | --- | --- | --- | --- |
| 0-20K | 6 | 140.1 | 141.0 | 124.6 - 158.6 |
| 20-40K | 7 | 137.6 | 140.9 | 113.3 - 165.3 |
| 40-80K | 10 | 150.9 | 144.8 | 108.2 - 163.6 |
| >80K | 1 | 156.1 | 156.1 | (single sample) |

All requests: median 144.2, mean 143.2, range 108.2 - 165.3 tok/s. The 40-80K
bucket is the *fastest* one. The A/B's -14% slope from 29K to 172K is real — it
was measured with content held constant — but in production it is swamped by
per-request content variance, which plausibly acts through MTP draft acceptance
(structured tool-call output is far more predictable than free-form prose). That
mechanism can no longer be confirmed from the server log: the `21a0e85f` format
dropped the per-request MTP acceptance field. **Treat the A/B table as a valid
relative `fp8`-vs-`nvfp4` comparison, not as absolute throughput, and do not
expect its context slope to be visible in day-to-day use.**

**Prefix reuse carries most of the prompt.**

| Path | Requests | Share |
| --- | --- | --- |
| `private_response_replay` | 16 | 66% |
| `root` (cold) | 5 | 20% |
| `shared_stable_prefix` | 3 | 12% |

79.4% of all prompt tokens (731576 of 921525) were served from cache, so only a
fifth was actually prefilled.

| Metric | Cold (`root`) | Cache hit |
| --- | --- | --- |
| TTFT, median | 4329 ms | **863 ms** |
| Prefill, median | 8262 tok/s | 4086 tok/s |
| Decode, mean | 148.1 tok/s | 141.9 tok/s |

Prefill tokens/second being lower on a cache hit is not a regression: only the
uncached delta is prefilled (e.g. 1579 new tokens on a 39761-token prompt), so
fixed per-launch overhead is amortised over a much smaller batch. TTFT is the
number that matters there, and it drops ~5x. Decode is unaffected by the cache,
as expected.

**System throughput by concurrency.**

| Concurrent streams | n | Median | Peak |
| --- | --- | --- | --- |
| 1 | 45 | 152.2 | 328.0 |
| 2 | 3 | 190.8 | 234.2 |
| 3 | 5 | 308.0 | **382.6** |
| 4 | 2 | 87.4 | 119.4 |

The `running=4` row is two samples that happened to land while prefill held the
GPU; it is not a steady-state decode figure. Same caution for the `>80K` decode
bucket above.

**No materialization pressure at all.** Across the whole window `running` peaked
at 4 (concurrency saturated) and `waiting` at 2, but `materializing` stayed at
**0** — no session was ever evicted and pulled back.

That is the concrete payoff of sizing for capacity.

### The materialization wedge (fixed 2026-09-02)

A `std::logic_error` ("materialization source has no resident state") out of
`prepare_materialization` trips `fail_all`, after which `worker_loop` returns and
**every** request is rejected with 503 `service_unavailable` — permanently. It hit
7 times in 27 minutes of live agent traffic before it was fixed.

**It was not capacity pressure.** An earlier draft of this section blamed the
`fp8` eviction path; per-instance JSONL disproved that. Four of the five wedged
server instances had `checkpoints_dropped = 0` and `private_owners_evicted = 0`,
`host_kv_bytes` peaked at **0** across every instance, and two failures happened
with `running = 0`. Instrumenting the throw site showed the source's rewrite
checkpoint resident on device and merely *shared* (`refs 2 / owned 1`):
`resident_resources()` counts only states exclusive to the sequence, so the guard
read a shared-but-present StateImage as absent. The local build now guards the
handle `selected_state()` actually returns. See `~/ninfer-repro/` for the
reproducer and the upstream Issue draft.

**Nothing external notices this class of failure**, so it is worth knowing even
after the fix: the process stays alive, the port keeps listening, `docker ps`
reports the container `Up`, and `--restart unless-stopped` never fires.
`/v1/models` also keeps returning 200, since it never reaches the engine; only a
real generation request detects it. `profiles/watchdog/engine-autostart.sh` probes
with a real generation request every minute and restarts after two strikes.

### Measurement caveat

Runs whose completion length is only 20-40 tokens produce unusable decode
figures — CUDA-graph warmup, MTP spin-up and first-token jitter dominate, giving
non-monotonic results that vary by ~30% (an early run had `fp8` faster at 180K
than at 130K). Use at least ~1000 completion tokens per point.

Note also that the structured log introduced in `21a0e85f` dropped the
per-request MTP acceptance rate and `wait=.../round` fields that the previous
log format carried, so the decode figure can no longer be split into bandwidth
and speculative-acceptance components from the server log alone.

### Corrections to earlier `ninfer.sh` comments

- The header claimed `KV pool 188224` for vision mode. That conflated the pool
  with `--max-context`: the measured `fp8` + `--vision` pool is 229376, and
  188224 was only the context ceiling set on top of it.
- `200000 OOMs` was an `fp8`-era result. Under `nvfp4` the pool is 410944, and
  `--max-context 262144` starts cleanly.
- `nvfp4 card tops at 262144 w/ MTP off` is wrong: 262144 was verified running
  with MTP3 enabled (`--spec mtp --draft-tokens 3 --lm-head-draft`), which also
  raises `kv_max_page_groups` from 11764 to 16384.
