# Compose Files

## vLLM — Unified (`vllm.yml`)

All vLLM configs use a single compose file driven by env vars.
Set via `local-ai.sh` menu (`./local-ai.sh`), which calls `export_vllm_vars()`.

| Mode | Model | KV Cache | Context | Genesis | Notes |
|---|---|---|---|---|---|
| `text-mtp` | AEON-XS Text | fp8_e4m3 | 228K | — | Text-only, fastest prefill |
| `vision-mtp` | AEON-XS Vision | fp8_e4m3 | 208K | — | Baseline vision, MTP n=3 |
| `vision-tq-mtp` | AEON-XS Vision | turboquant | 324K | P5B, P67, PN34, PREALLOC_V2 | Max context |
| `huihui-vision-mtp` | Huihui Vision | fp8_e4m3 | 208K | — | [deprecated] |
| `huihui-vision-tq-mtp` | Huihui Vision | turboquant | 312K | all 10 | [deprecated] |

**Add a new base model**: just add a `case` in `local-ai.sh:export_vllm_vars()` — no new compose files needed.

### Legacy compose files

Previous per-mode compose files are archived in `_archive/` for reference:
`text-mtp.yml`, `vision-mtp.yml`, `vision-tq-mtp.yml`, `huihui-vision-mtp.yml`, `huihui-vision-tq-mtp.yml`

## Beellama Configs

Separate compose files (separate engine, not part of unified vllm.yml).

| File | Draft | Context | Vision |
|---|---|---|---|
| `beellama/dflash-vision.yml` | DFlash IQ4_XS | 262K | yes |
| `beellama/qwopus-mtp-vision.yml` | MTP draft=2 | 262K | yes |

## Usage

```bash
# Interactive menu
./local-ai.sh

# Direct mode switch
ENGINE=vision-tq-mtp ./local-ai.sh up

# Or manually with docker compose (requires env vars set)
docker compose -f compose/vllm.yml up -d
```
