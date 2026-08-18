#!/usr/bin/env bash
#
# Concurrent throughput benchmark for vLLM server.
#   - Tests multiple prompt lengths (20K, 50K, 100K tokens)
#   - Tests multiple concurrency levels per length
#   - Measures aggregate TPS, per-request TPS, TTFT
#   - Finds throughput ceiling per prompt length
#
# Env vars:
#   URL            endpoint            (default: http://localhost:8020)
#   MODEL          served model name   (default: qwen3.6)
#   LEVELS         concurrency levels  (default: "1 2 4 6")
#   RUNS           requests per level  (default: 4)
#   WARMUPS        warmup requests     (default: 1)
#   MAX_TOKENS     tokens to generate  (default: 500)
#   PROMPT_SIZES   prompt sizes in tokens (default: "20000 50000 100000")
#
# Usage:
#   bash scripts/bench-concurrent.sh
#   LEVELS="1 2 4 8" RUNS=6 bash scripts/bench-concurrent.sh

set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-local}"
LEVELS="${LEVELS:-1 2 4 6 8 10}"
RUNS="${RUNS:-4}"
WARMUPS="${WARMUPS:-1}"
MAX_TOKENS="${MAX_TOKENS:-500}"
PROMPT_SIZES="${PROMPT_SIZES:-20000 50000 100000}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  exit 1
fi

echo "=== Concurrent Throughput Benchmark ==="
echo "  URL:         ${URL}"
echo "  Model:       ${MODEL}"
echo "  Levels:      ${LEVELS}"
echo "  Requests:    ${RUNS} per level"
echo "  Max tokens:  ${MAX_TOKENS}"
echo "  Prompt sizes: ${PROMPT_SIZES} tokens"
echo ""

python3 - "$URL" "$MODEL" "$RUNS" "$WARMUPS" "$MAX_TOKENS" $PROMPT_SIZES $LEVELS << 'PYEOF'
import json, sys, time, urllib.request, statistics as s
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = sys.argv[1]
MODEL = sys.argv[2]
RUNS = int(sys.argv[3])
WARMUPS = int(sys.argv[4])
MAX_TOKENS = int(sys.argv[5])

# Parse prompt sizes and levels from remaining args
# First N args that are large numbers = prompt sizes, rest = levels
remaining = sys.argv[6:]
PROMPT_SIZES = []
LEVELS = []
for v in remaining:
    n = int(v)
    if n > 100:
        PROMPT_SIZES.append(n)
    else:
        LEVELS.append(n)

# Generate a prompt of approximately target_tokens tokens
# ~1.3 chars per token for English text
def make_prompt(target_tokens):
    base = "Explain in detail how transformer attention mechanisms work, including self-attention, multi-head attention, positional encoding, and the computational complexity trade-offs. "
    base_tokens = max(1, len(base) // 4)  # rough estimate
    repeats = max(1, target_tokens // base_tokens)
    return (base * repeats)[:target_tokens * 4]  # ~4 chars per token estimate

def send_request(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content and ttft is None:
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    return wall, ttft, completion_tokens, prompt_tokens

def run_concurrent(n_requests, prompt, max_tokens):
    results = []
    with ThreadPoolExecutor(max_workers=n_requests) as pool:
        futures = [pool.submit(send_request, prompt, max_tokens) for _ in range(n_requests)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(None)
    return results

def analyze(results):
    ok = [r for r in results if r is not None]
    failed = len(results) - len(ok)
    if not ok:
        return {"n": 0, "failed": failed}

    walls = [r[0] for r in ok]
    ttfts = [r[1] for r in ok]
    toks = [r[2] for r in ok]

    total_tokens = sum(toks)
    total_time = max(walls)
    aggregate_tps = total_tokens / total_time if total_time > 0 else 0
    per_request_tps = [t / w for t, w in zip(toks, walls)]

    return {
        "n": len(ok),
        "failed": failed,
        "aggregate_tps": aggregate_tps,
        "per_req_tps_mean": s.mean(per_request_tps),
        "per_req_tps_min": min(per_request_tps),
        "per_req_tps_max": max(per_request_tps),
        "ttft_mean": s.mean(ttfts),
        "ttft_p50": sorted(ttfts)[len(ttfts) // 2],
        "ttft_p99": sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 1 else ttfts[0],
        "toks_mean": s.mean(toks),
    }

# ── Warmup ──────────────────────────────────────────────────────────────────
print("Warming up server...")
warmup_prompt = make_prompt(500)
for i in range(WARMUPS):
    try:
        send_request(warmup_prompt, 100)
        print(f"  warm-{i+1}  OK")
    except Exception as e:
        print(f"  warm-{i+1}  FAIL: {e}")
print()

# ── Benchmark per prompt size ───────────────────────────────────────────────
for psize in PROMPT_SIZES:
    prompt = make_prompt(psize)
    # Measure actual prompt tokens via a single request
    try:
        _, _, _, actual_ptok = send_request(prompt, 1)
    except Exception:
        actual_ptok = psize

    print(f"{'='*70}")
    print(f"  PROMPT SIZE: ~{psize:,} tokens (actual: {actual_ptok:,} prompt_tokens)")
    print(f"  Output tokens: {MAX_TOKENS}")
    print(f"{'='*70}")

    header = f"{'Conc':>4s}  {'Reqs':>4s}  {'AggTPS':>8s}  {'ReqTPS':>8s}  {'TTFT avg':>9s}  {'TTFT p99':>9s}  {'Out/req':>7s}  {'Fail':>4s}"
    print(header)
    print("-" * len(header))

    best_agg = 0
    best_level = 0

    for level in LEVELS:
        raw = run_concurrent(level, prompt, MAX_TOKENS)
        m = analyze(raw)

        if m["n"] == 0:
            print(f"{level:>4d}  {'ALL FAILED':>4s}")
            continue

        print(f"{level:>4d}  {m['n']:>4d}  {m['aggregate_tps']:>8.1f}  "
              f"{m['per_req_tps_mean']:>8.1f}  {m['ttft_mean']*1000:>8.0f}ms  "
              f"{m['ttft_p99']*1000:>8.0f}ms  {m['toks_mean']:>7.0f}  "
              f"{m['failed']:>4d}")

        if m["aggregate_tps"] > best_agg:
            best_agg = m["aggregate_tps"]
            best_level = level

    print(f"\n  Peak: {best_agg:.1f} agg_TPS at concurrency={best_level}")
    print()

PYEOF

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi
