#!/usr/bin/env bash
#
# Scheduling latency benchmark — concurrent decode + new long prefill.
#
# Tests a classic scenario:
#   1. A long-context session (120K tokens) is actively decoding.
#   2. A new request (40K prompt) arrives simultaneously.
#   3. Measure: how long until the new request produces its first token?
#
# This validates chunked-prefill behavior under active decode load:
#   - Does the scheduler interleave prefill chunks with decode steps?
#   - Does MTP spec-decode affect the new request's latency?
#   - What's the effective TTFT when the engine is busy?
#
# Env vars:
#   URL                 endpoint                     (default: http://localhost:8020)
#   MODEL               served model name            (default: local)
#   CONTAINER             container name              (default: vllm-text-mtp)
#   RUNS                  measured runs               (default: 3)
#   WARMUPS               warmup runs                 (default: 1)
#   BACKGROUNDS_PROMPT    background session prompt   (default: 120000)
#   BACKGROUND_OUTPUT     background decode tokens    (default: 200)
#   NEW_REQUEST_PROMPT    new request prompt size      (default: 40000)
#   NEW_REQUEST_OUTPUT    new request output tokens     (default: 512)
#   TEMPERATURE           sampling temp                (default: 0.6)
#   TOP_P                 top-p                       (default: 0.95)
#   SAVE                  0=no save 1=save             (default: 1)
#   SAVE_DIR              where to save results        (default: bench/)
#
# Usage:
#   bash scripts/bench-scheduling.sh
#   RUNS=5 BACKGROUND_PROMPT=200000 bash scripts/bench-scheduling.sh
#   SAVE=0 bash scripts/bench-scheduling.sh              # stdout only

set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-local}"
CONTAINER="${CONTAINER:-vllm-text-mtp}"
RUNS="${RUNS:-3}"
WARMUPS="${WARMUPS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SAVE="${SAVE:-1}"
SAVE_DIR="${SAVE_DIR:-bench}"
BACKGROUND_PROMPT="${BACKGROUND_PROMPT:-120000}"
BACKGROUND_OUTPUT="${BACKGROUND_OUTPUT:-200}"
NEW_REQUEST_PROMPT="${NEW_REQUEST_PROMPT:-40000}"
NEW_REQUEST_OUTPUT="${NEW_REQUEST_OUTPUT:-512}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: ./local-ai.sh up" >&2
  exit 1
fi

echo "========================================================================"
echo "  Scheduling Latency Benchmark — Concurrent Decode + New Request"
echo "========================================================================"
echo "  URL:            ${URL}"
echo "  Model:          ${MODEL}"
echo "  Container:      ${CONTAINER}"
echo "  Warmups:        ${WARMUPS}  ×  Measured: ${RUNS}"
echo "  Temperature:    ${TEMPERATURE}  Top-p: ${TOP_P}"
echo ""
echo "  Scenario:"
echo "    Background session:  ${BACKGROUND_PROMPT} prompt → ${BACKGROUND_OUTPUT} output (decoding)"
echo "    New request:          ${NEW_REQUEST_PROMPT} prompt → ${NEW_REQUEST_OUTPUT} output"
echo "    Measure:              Time-to-first-token (TTFT) of new request"
echo ""
echo "  Key metrics:"
echo "    - TTFT (cold vs under load)"
echo "    - Prefill chunk interleaving effectiveness"
echo "    - Decode step disruption from concurrent prefill"
echo "    - KV cache pressure during the overlap window"
echo "========================================================================"
echo ""

export TIMESTAMP SAVE SAVE_DIR
python3 - "$URL" "$MODEL" "$CONTAINER" "$WARMUPS" "$RUNS" \
        "$TEMPERATURE" "$TOP_P" \
        "$BACKGROUND_PROMPT" "$BACKGROUND_OUTPUT" \
        "$NEW_REQUEST_PROMPT" "$NEW_REQUEST_OUTPUT" << 'PYEOF'
import json, os, re, shutil, subprocess, sys, time, threading, urllib.request, statistics as s
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Parse args ──────────────────────────────────────────────────────────
URL = sys.argv[1]
MODEL = sys.argv[2]
CONTAINER = sys.argv[3]
WARMUPS = int(sys.argv[4])
RUNS = int(sys.argv[5])
TEMPERATURE = float(sys.argv[6])
TOP_P = float(sys.argv[7])
BACKGROUND_PROMPT = int(sys.argv[8])
BACKGROUND_OUTPUT = int(sys.argv[9])
NEW_REQUEST_PROMPT = int(sys.argv[10])
NEW_REQUEST_OUTPUT = int(sys.argv[11])
TIMESTAMP = os.environ.get("TIMESTAMP", datetime.now().strftime("%Y%m%d-%H%M%S"))
SAVE = os.environ.get("SAVE", "1") == "1"
SAVE_DIR = os.environ.get("SAVE_DIR", "bench")

# ── Prompt generation ─────────────────────────────────────────────────
def make_prompt(target_tokens):
    """Generate text that yields approximately target_tokens tokens."""
    base = (
        "Transformer attention mechanisms have fundamentally transformed natural language processing "
        "by introducing a self-attention architecture that processes all tokens in parallel. "
        "The core innovation is the scaled dot-product attention operation: each token generates "
        "a query, key, and value vector, then computes attention scores as the dot product between "
        "its query and all keys, scaled by the square root of the head dimension. "
        "This design enables the model to directly capture long-range dependencies in a sequence, "
        "regardless of distance, which was a fundamental limitation of recurrent neural networks. "
        "The multi-head attention mechanism further enhances this by running multiple attention "
        "operations in parallel, each learning different relationship patterns. "
        "Positional encodings are added to the input embeddings to give the model information "
        "about token positions, since the self-attention operation itself is permutation-invariant. "
    )
    chars_needed = int(target_tokens * 3.8)
    repeats = max(1, chars_needed // len(base)) + 1
    text = (base * repeats)[:chars_needed]
    return text

BG_PROMPT = make_prompt(BACKGROUND_PROMPT)
NEW_PROMPT = make_prompt(NEW_REQUEST_PROMPT)

# ── Helpers ───────────────────────────────────────────────────────────
def fmt_time(sec):
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    return f"{sec:.2f}s"

def run_streaming(prompt, max_tokens, label=""):
    """Send a streaming request and return timing metrics."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    first_token_text = None
    completion_tokens = 0
    prompt_tokens = 0
    token_times = []  # timestamps of each token arrival

    try:
        with urllib.request.urlopen(req, timeout=600) as r:
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
                    if content:
                        if ttft is None:
                            ttft = time.time() - t_send
                            first_token_text = content[:20]
                        token_times.append(time.time() - t_send)
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:
        return {"error": str(e)}

    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    decode_time = max(wall - ttft, 1e-6)
    ct = max(completion_tokens, 1)

    return {
        "wall": wall,
        "ttft": ttft,
        "prefill_time": ttft,
        "decode_time": decode_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tps": prompt_tokens / ttft if prompt_tokens > 0 and ttft > 0 else 0,
        "decode_tps": ct / decode_time,
        "wall_tps": ct / wall,
        "first_token_text": first_token_text or "",
    }

def run_background_session():
    """Start a background session that decodes for BACKGROUND_OUTPUT tokens.
    Returns (future, send_time) so we can track progress."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": BG_PROMPT}],
        "max_tokens": BACKGROUND_OUTPUT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    token_count = 0
    ttft = None

    # Read the response in a generator — we'll consume it lazily
    def response_generator():
        nonlocal token_count, ttft
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    line = line.decode("utf-8", errors="ignore").rstrip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if ttft is None:
                                ttft = time.time() - t_send
                            token_count += 1
                            yield content
        except Exception:
            pass
        return

    return response_generator(), t_send, ttft

# ── Test: Cold TTFT (baseline, no concurrent load) ────────────────────
def test_cold_ttft():
    print(f"\n{'='*70}")
    print(f"  Phase 1: Cold TTFT (baseline — no concurrent load)")
    print(f"{'='*70}")

    results = []
    for i in range(RUNS):
        r = run_streaming(NEW_PROMPT, NEW_REQUEST_OUTPUT, f"cold-{i+1}")
        if r.get("error"):
            print(f"  cold-{i+1}  ✗ {r['error']}")
            continue
        results.append(r)
        print(f"  cold-{i+1}  ✓  TTFT={fmt_time(r['ttft'])}  "
              f"prefill={r['prefill_tps']:.0f} t/s  "
              f"prompt={r['prompt_tokens']:,}")
    return results

# ── Test: Loaded TTFT (background session decoding) ───────────────────
def test_loaded_ttft():
    print(f"\n{'='*70}")
    print(f"  Phase 2: Loaded TTFT (background session decoding)")
    print(f"{'='*70}")
    print(f"  Background: {BACKGROUND_PROMPT:,} prompt → {BACKGROUND_OUTPUT:,} output")
    print(f"  New request: {NEW_REQUEST_PROMPT:,} prompt → {NEW_REQUEST_OUTPUT:,} output")
    print("")

    results = []
    for i in range(RUNS):
        # Start background session
        bg_gen = run_background_session()
        t_bg_send = time.time()

        # Consume the background generator in a separate thread
        import threading
        bg_result = {"tokens": 0, "ttft": None, "done": False}

        def consume_bg():
            for token in bg_gen:
                bg_result["tokens"] += 1
            bg_result["done"] = True

        bg_thread = threading.Thread(target=consume_bg, daemon=True)
        bg_thread.start()

        # Wait for background session to enter decode phase (TTFT elapsed)
        # Give it ~1.5s to complete prefill and start decoding
        time.sleep(1.5)

        # Now send the new request
        t_new_send = time.time()
        r = run_streaming(NEW_PROMPT, NEW_REQUEST_OUTPUT, f"loaded-{i+1}")

        # Check background status at the moment new request was sent
        bg_tokens_at_send = bg_result["tokens"]

        if r.get("error"):
            print(f"  loaded-{i+1}  ✗ {r['error']}")
            continue
        results.append(r)
        print(f"  loaded-{i+1}  ✓  TTFT={fmt_time(r['ttft'])}  "
              f"prefill={r['prefill_tps']:.0f} t/s  "
              f"decode={r['decode_tps']:.1f} TPS  "
              f"[bg: ~{bg_tokens_at_send} tokens decoded]")

    # Clean up: wait for background to finish
    bg_thread.join(timeout=30)
    return results

# ── KV cache pressure test ────────────────────────────────────────────
def test_kv_pressure():
    """Measure KV cache usage during the overlap window."""
    print(f"\n{'='*70}")
    print(f"  Phase 3: KV Cache Pressure during overlap")
    print(f"{'='*70}")

    # Run a loaded test and capture KV metrics
    if not CONTAINER or CONTAINER == "none" or shutil.which("docker") is None:
        print("  (Docker not available, skipping KV metrics)")
        return None

    # Start background
    bg_gen = run_background_session()

    # Sample KV cache metrics during overlap
    kv_samples = []
    sampling = threading.Event()
    sampling.set()

    def sample_kv():
        for _ in range(10):
            if sampling.is_set():
                try:
                    resp = urllib.request.urlopen(f"{URL}/metrics", timeout=5)
                    text = resp.read().decode()
                    for line in text.split("\n"):
                        if line.startswith("vllm:kv_cache_usage_perc"):
                            val = line.split()[-1]
                            kv_samples.append(float(val))
                except Exception:
                    pass
                time.sleep(0.5)

    kv_thread = threading.Thread(target=sample_kv, daemon=True)
    kv_thread.start()

    # Wait for background to be in decode
    time.sleep(1.5)

    # Send new request during sampling
    r = run_streaming(NEW_PROMPT, NEW_REQUEST_OUTPUT)

    sampling.clear()
    kv_thread.join(timeout=5)

    if kv_samples:
        kv_mean = s.mean(kv_samples)
        kv_max = max(kv_samples)
        kv_min = min(kv_samples)
        print(f"  KV cache usage during overlap:")
        print(f"    Mean:  {kv_mean*100:.1f}%")
        print(f"    Min:   {kv_min*100:.1f}%")
        print(f"    Max:   {kv_max*100:.1f}%")
        print(f"    Samples: {len(kv_samples)}")
        return {"mean": kv_mean, "max": kv_max, "min": kv_min, "samples": len(kv_samples)}
    return None

# ── Main ────────────────────────────────────────────────────────────────
print("")

# Phase 1: Cold TTFT
cold_results = test_cold_ttft()
if cold_results:
    cold_ttft_mean = s.mean([r["ttft"] for r in cold_results])
    cold_ttft_std = s.stdev([r["ttft"] for r in cold_results]) if len(cold_results) > 1 else 0
    print(f"\n  → Cold TTFT: {fmt_time(cold_ttft_mean)} ± {fmt_time(cold_ttft_std)}")
else:
    cold_ttft_mean = cold_ttft_std = None

# Phase 2: Loaded TTFT
loaded_results = test_loaded_ttft()
if loaded_results:
    loaded_ttft_mean = s.mean([r["ttft"] for r in loaded_results])
    loaded_ttft_std = s.stdev([r["ttft"] for r in loaded_results]) if len(loaded_results) > 1 else 0
    print(f"\n  → Loaded TTFT: {fmt_time(loaded_ttft_mean)} ± {fmt_time(loaded_ttft_std)}")

    # Penalty
    if cold_ttft_mean and loaded_ttft_mean > 0:
        penalty_pct = ((loaded_ttft_mean - cold_ttft_mean) / cold_ttft_mean) * 100
        print(f"  → TTFT penalty under load: +{penalty_pct:.1f}%")
else:
    loaded_ttft_mean = loaded_ttft_std = None

# Phase 3: KV pressure
kv_data = test_kv_pressure()

# ── Summary table ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SCHEDULING LATENCY SUMMARY")
print(f"{'='*70}")
print("")

if cold_results:
    print(f"  {'Metric':<35s}  {'Cold':>15s}  {'Loaded':>15s}  {'Penalty':>10s}")
    print(f"  {'─'*35}  {'─'*15}  {'─'*15}  {'─'*10}")

    ttft_c = f"{cold_ttft_mean*1000:.0f}ms" if cold_ttft_mean else "-"
    ttft_l = f"{loaded_ttft_mean*1000:.0f}ms" if loaded_ttft_mean else "-"
    penalty = f"+{(((loaded_ttft_mean-cold_ttft_mean)/cold_ttft_mean)*100):.1f}%" if cold_ttft_mean and loaded_ttft_mean else "-"
    print(f"  {'TTFT (mean)':<35s}  {ttft_c:>15s}  {ttft_l:>15s}  {penalty:>10s}")

    prefill_c = f"{s.mean([r['prefill_tps'] for r in cold_results]):.0f} t/s"
    prefill_l = f"{s.mean([r['prefill_tps'] for r in loaded_results]):.0f} t/s" if loaded_results else "-"
    print(f"  {'Prefill throughput':<35s}  {prefill_c:>15s}  {prefill_l:>15s}  {'':>10s}")

    decode_l = f"{s.mean([r['decode_tps'] for r in loaded_results]):.1f} TPS" if loaded_results else "-"
    print(f"  {'Decode throughput (loaded)':<35s}  {'-':>15s}  {decode_l:>15s}  {'':>10s}")

if kv_data:
    print(f"")
    print(f"  KV Cache during overlap:")
    print(f"    Mean:  {kv_data['mean']*100:.1f}%")
    print(f"    Range: {kv_data['min']*100:.1f}% — {kv_data['max']*100:.1f}%")

# ── Save results ─────────────────────────────────────────────────────────
if SAVE and (cold_results or loaded_results):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"scheduling-{TIMESTAMP}.md")
    with open(save_path, "w") as f:
        f.write(f"# Scheduling Latency Bench — {TIMESTAMP}\n\n")
        f.write(f"- URL: `{URL}`\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write(f"- Background: {BACKGROUND_PROMPT:,} prompt → {BACKGROUND_OUTPUT:,} output\n")
        f.write(f"- New request: {NEW_REQUEST_PROMPT:,} prompt → {NEW_REQUEST_OUTPUT:,} output\n")
        f.write(f"- Runs: {RUNS}  Warmups: {WARMUPS}\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | Cold | Loaded | Penalty |\n")
        f.write("|--------|------|--------|---------|\n")
        if cold_ttft_mean and loaded_ttft_mean:
            f.write(f"| TTFT (mean) | {cold_ttft_mean*1000:.0f}ms | {loaded_ttft_mean*1000:.0f}ms | +{(((loaded_ttft_mean-cold_ttft_mean)/cold_ttft_mean)*100):.1f}% |\n")
            f.write(f"| Prefill t/s | {s.mean([r['prefill_tps'] for r in cold_results]):.0f} | {s.mean([r['prefill_tps'] for r in loaded_results]):.0f} | - |\n")
            f.write(f"| Decode TPS | - | {s.mean([r['decode_tps'] for r in loaded_results]):.1f} | - |\n")
        if kv_data:
            f.write(f"\n| KV Cache (mean) | - | {kv_data['mean']*100:.1f}% | - |\n")
            f.write(f"| KV Cache (range) | - | {kv_data['min']*100:.1f}%-{kv_data['max']*100:.1f}% | - |\n")
        f.write("\n## Detailed Results\n\n")
        f.write("| Run | TTFT | Prefill t/s | Decode TPS | n |\n")
        f.write("|-----|------|-------------|------------|---|\n")
        for r in loaded_results:
            f.write(f"| loaded | {r['ttft']*1000:.0f}ms | {r['prefill_tps']:.0f} | {r['decode_tps']:.1f} | {r['completion_tokens']} |\n")
    print(f"\n  Results saved: {save_path}")

print(f"\n{'='*70}")
print(f"  Done. Cold: {len(cold_results)} runs, Loaded: {len(loaded_results) if loaded_results else 0} runs.")
print(f"{'='*70}")
print("")
PYEOF

# ── GPU state ──────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi
