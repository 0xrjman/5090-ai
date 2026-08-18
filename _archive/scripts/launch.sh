#!/usr/bin/env bash
#
# Launch the Qwen3.6-27B NVFP4+MTP vLLM server.
#
# Usage:
#   bash scripts/launch.sh                          # use defaults (./models/)
#   MODEL_DIR=/mnt/models bash scripts/launch.sh    # custom weights path
#   bash scripts/launch.sh --force                  # skip GPU preflight
#
# Env vars (all optional, see .env.example):
#   MODEL_DIR     path to model weights             (default: ./models)
#   PORT          host port                          (default: 8020)
#   GPU_DEVICE    GPU index                          (default: 0)
#   FORCE         set 1 to skip GPU preflight        (default: 0)
#   READY_TIMEOUT seconds to wait for ready          (default: 600)

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/compose/mtp.yml"
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
FORCE="${FORCE:-0}"

# Parse args
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --down)  cd "$ROOT_DIR" && $COMPOSE_BIN -f "$COMPOSE_FILE" down; exit 0 ;;
    --help|-h)
      echo "Usage: bash scripts/launch.sh [--force] [--down]"
      echo "  --force   skip GPU preflight checks"
      echo "  --down    bring down the server"
      exit 0
      ;;
  esac
done

# Load .env if present
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a; source "${ROOT_DIR}/.env"; set +a
fi

PORT="${PORT:-8020}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"

echo "[launch] MODEL_DIR=${MODEL_DIR}"
echo "[launch] PORT=${PORT}"

# ── Preflight checks ────────────────────────────────────────────────────────
if [[ "$FORCE" != "1" ]]; then
  # Check docker
  if ! command -v docker &>/dev/null; then
    echo "[preflight] ERROR: docker not found. Install Docker first." >&2
    exit 1
  fi

  # Check nvidia-smi
  if command -v nvidia-smi &>/dev/null; then
    echo "[preflight] GPU:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || true
  else
    echo "[preflight] WARNING: nvidia-smi not found. Skipping GPU check." >&2
  fi

  # Check model weights exist
  if [[ ! -d "${MODEL_DIR}/qwen3.6-27b-nvfp4-mtp" ]]; then
    echo "[preflight] ERROR: Weights not found at ${MODEL_DIR}/qwen3.6-27b-nvfp4-mtp/" >&2
    echo "  Download the sakamakismile NVFP4+MTP weights and place them there." >&2
    echo "  Or set MODEL_DIR=/path/to/your/models" >&2
    exit 1
  fi
else
  echo "[preflight] FORCE=1 — skipping checks"
fi

# Ensure cache dirs exist
mkdir -p "${ROOT_DIR}"/cache/{triton,torch_compile,flashinfer}

# ── Bring up ─────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "[launch] bringing up vLLM NVFP4+MTP..."
$COMPOSE_BIN -f "$COMPOSE_FILE" up -d

# ── Wait for ready ───────────────────────────────────────────────────────────
URL="http://localhost:${PORT}/v1/models"
echo "[launch] waiting for ${URL} (timeout ${READY_TIMEOUT}s)..."

elapsed=0
while (( elapsed < READY_TIMEOUT )); do
  if curl -sf "$URL" >/dev/null 2>&1; then
    echo ""
    echo "[launch] ✓ Server is ready!"
    echo "[launch] API: http://localhost:${PORT}/v1"
    echo "[launch] Model: $(curl -sf "$URL" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || echo 'qwen3.6')"
    echo ""
    echo "  Test: curl http://localhost:${PORT}/v1/chat/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"qwen3.6\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'"
    exit 0
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  printf "\r[launch] waiting... %ds" "$elapsed"
done

echo ""
echo "[launch] ERROR: Server did not become ready within ${READY_TIMEOUT}s" >&2
echo "  Check logs: docker logs vllm-qwen36-nvfp4-mtp" >&2
exit 1
