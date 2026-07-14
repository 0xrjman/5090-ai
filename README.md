# local-ai 🚀

**Multi-engine LLM serving for NVIDIA GPUs — managed by CLI, driven by AI Agents.**

```bash
git clone https://github.com/0xrjman/local-ai && cd local-ai
./local-ai.sh
```

---

## Quick Start

```bash
# Interactive TUI menu
./local-ai.sh

# Non-interactive: start with a specific engine
./local-ai.sh up                        # uses default (glm-5.2-vllm)
ENGINE=vision-mtp ./local-ai.sh up      # specific engine
```

---

## Agent Workflow

The core usage pattern: **start the server → let AI Agents use it via OpenAI-compatible API**.

```bash
# 1. Start a server
./local-ai.sh up

# 2. Agent connects via standard OpenAI API
curl http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

**For AI Agents (structured output):**

```bash
# Discover available configs (JSON)
local-ai config list --json

# Switch config and restart
local-ai config switch vision-mtp

# Check service status (JSON)
local-ai status --json

# Health check (exit code 0 = ready)
local-ai health

# Get connection info for Agent configuration
local-ai info --json
# → {"url":"http://localhost:8020/v1","model":"local","config":"vision-mtp","port":8020}
```

---

## Engine Configurations

Declarative configs in `configs/*.yaml`. Add a new model by adding one file.

| # | Engine | GPU | VRAM | Context | Runtime |
|---|--------|-----|------|---------|---------|
| 1 | AEON-XS MTP (Vision) 🟢 | 1× 5090/B200 | ~170 GB | 208K | vLLM |
| 2 | AEON-XS MTP+TQ (Vision) | 1× B200 | ~170 GB | 324K | vLLM + Genesis |
| 3 | AEON-XS MTP (Text) 🟢 | 1× 5090/B200 | ~170 GB | 228K | vLLM |
| 4 | Huihui NVFP4+MTP (Vision) | 1× B200 | ~170 GB | 208K | vLLM [deprecated] |
| 5 | Huihui NVFP4+MTP+TQ (Vision) | 1× B200 | ~170 GB | 312K | vLLM + Genesis [deprecated] |
| 6 | Beellama DFlash Vision | 1× 5090/B200 | ~24 GB | 262K | Beellama |
| 7 | Beellama Qwopus MTP Vision | 1× 5090/B200 | ~24 GB | 262K | Beellama |
| 8 | GLM-5.2 NVFP4 · vLLM 🆕 | **8× B200** | ~116 GB | **1M** | vLLM |
| 9 | GLM-5.2 NVFP4 · SGLang 🆕 | **8× B200** | ~116 GB | **1M** | SGLang [WIP] |

> 🟢 = production-ready · 🆕 = newly added

---

## CLI Reference

### Server Commands

| Command | Description |
|---------|-------------|
| `local-ai up` | Start server |
| `local-ai down` | Stop server |
| `local-ai status` | Show status |
| `local-ai status --json` | Status as JSON |
| `local-ai logs` | Tail logs |
| `local-ai health` | Health check (exit 0 = ready) |
| `local-ai info --json` | Connection info for Agents |
| `local-ai bench` | Run benchmark |

### Config Commands

| Command | Description |
|---------|-------------|
| `local-ai config list` | List configs (JSON) |
| `local-ai config list --table` | List configs (TUI table) |
| `local-ai config show <name>` | Show config detail (JSON) |
| `local-ai config switch <name>` | Switch config + auto-restart |
| `local-ai config add ...` | Create new config |
| `local-ai config delete <name>` | Delete config |

### Env Variables

```bash
ENGINE=vision-mtp ./local-ai.sh up    # Select engine
MODEL_DIR=/path/to/models ./local-ai.sh up  # Override model path
```

### Adding a New Engine

```bash
# Add a new config file
local-ai config add \
  --name "My Custom Model" \
  --internal my-model \
  --runtime vllm \
  --weights my-model-dir \
  --hf-repo owner/model-name

# Or create configs/my-model.yaml directly
```

---

## Architecture

```
┌─────────────────────────────┐
│   CLI / TUI (local-ai.sh)   │
│  up/down/status/bench      │
├─────────────────────────────┤
│  Config Layer               │
│  configs/*.yaml            │  ← declarative, Agent-readable
│  lib/config_loader.py      │  ← JSON CLI, schema validation
├─────────────────────────────┤
│  Runtime (Docker Compose)  │
│  vllm.yml / glm-*.yml      │  ← env-var-driven
├─────────────────────────────┤
│  Engines                     │
│  vLLM · SGLang · Beellama    │
└─────────────────────────────┘
```

## Project Layout

```
local-ai/
├── local-ai.sh              # Main TUI + CLI entry point
├── configs/                  # Declarative engine configs (YAML)
│   ├── aeon-vision-mtp.yaml
│   ├── glm-5.2-vllm.yaml
│   └── ...
├── lib/
│   └── config_loader.py      # Config loader + Agent CLI
├── compose/
│   ├── vllm.yml            # Unified vLLM compose
│   ├── glm-vllm.yml        # GLM-5.2 vLLM (TP=8)
│   ├── glm-sglang.yml       # GLM-5.2 SGLang (TP=8)
│   └── beellama/           # Beellama compose files
├── genesis/vllm/           # Genesis performance patches (123 patches)
├── chat-templates/         # Jinja2 chat templates
├── cache/                  # Persistent JIT caches
└── scripts/                # Benchmark scripts
```

---

<p align="center"><sub>Built for RTX 5090 · B200 · Blackwell — managed by CLI, driven by Agents.</sub></p>