#!/usr/bin/env bash
#
# Sequential benchmark — staged context length testing.
#
# Tests server latency and throughput at graduated prompt sizes:
#   - Short  (1K), Medium (10K), Long (50K), XLong (100K), Max (180K+)
#   - Each tier runs at multiple output token lengths
#   - Reports TTFT, prefill TP, decode TPS, wall TPS
#   - Saves structured results to bench/ for cross-config comparison
#
# Env vars:
#   URL               endpoint                   (default: http://localhost:8020)
#   MODEL             served model name          (default: local)
#   CONTAINER         container name             (default: vllm-qwen36-nvfp4-mtp)
#   RUNS              measured runs per point    (default: 3)
#   WARMUPS           warmup runs per point      (default: 1)
#   TIERS             space-separated prompt:out  (default: see below)
#   SAVE_DIR          where to save results       (default: bench/)
#   SAVE              0=no save  1=save (default: 1)
#   TEMPERATURE       sampling temp              (default: 0.6)
#   TOP_P             top-p                      (default: 0.95)
#
# Default tiers (prompt_tokens:output_tokens):
#   1024:128   1024:512   1024:2048       # short context
#   10240:128  10240:512  10240:2048      # medium context
#   51200:128  51200:512                  # long context
#   102400:128 102400:512                 # xlong context
#   184320:128 184320:512                 # max context (fits 200K budget)
#
# Usage:
#   bash scripts/bench.sh
#   TIERS="1024:128 102400:512" RUNS=5 bash scripts/bench.sh
#   SAVE=0 bash scripts/bench.sh          # stdout only

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-local}"
CONTAINER="${CONTAINER:-vllm-qwen36-nvfp4-mtp}"
RUNS="${RUNS:-1}"
WARMUPS="${WARMUPS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SAVE="${SAVE:-1}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/bench}"

# Default tiers: prompt_tokens:output_tokens
if [[ -z "${TIERS:-}" ]]; then
  TIERS="1024:500 10240:500 51200:500 102400:500 184320:500"
fi

# ── Helpers ─────────────────────────────────────────────────────────────────
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }; }
need curl
need python3

# Timestamp for result file
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# ── Preflight ───────────────────────────────────────────────────────────────
if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: bash local-ai.sh" >&2
  exit 1
fi

# Print test plan
echo "========================================================================"
echo "  Sequential Benchmark — Staged Context Testing"
echo "========================================================================"
echo "  URL:      ${URL}"
echo "  Model:    ${MODEL}"
echo "  Container: ${CONTAINER}"
echo "  Warmups:  ${WARMUPS}  ×  Measured: ${RUNS}"
echo "  Temperature: ${TEMPERATURE}  Top-p: ${TOP_P}"
echo ""
echo "  Tiers:"
for tier in $TIERS; do
  p=${tier%%:*}; o=${tier#*:}
  fp=$(echo "$p" | sed ':l;s/\B[0-9]\{3\}\>/,&/;tl')
  printf "    %7s prompt → %4s output\n" "$fp" "$o"
done
echo ""
echo "  Estimated total requests: $(echo "$TIERS" | wc -w) × ($WARMUPS warmup + $RUNS measured) = $(( $(echo "$TIERS" | wc -w) * (WARMUPS + RUNS) ))"
echo "========================================================================"
echo ""

# ── Run benchmark in Python ────────────────────────────────────────────────
export TIMESTAMP SAVE SAVE_DIR
python3 - "$URL" "$MODEL" "$CONTAINER" "$WARMUPS" "$RUNS" "$TEMPERATURE" "$TOP_P" $TIERS << 'PYEOF'
import json, os, re, shutil, subprocess, sys, time, urllib.request, statistics as s
from datetime import datetime

# ── Parse args ──────────────────────────────────────────────────────────────
URL, MODEL, CONTAINER = sys.argv[1:4]
WARMUPS = int(sys.argv[4])
RUNS = int(sys.argv[5])
TEMPERATURE = float(sys.argv[6])
TOP_P = float(sys.argv[7])
TIERS = [(int(p), int(o)) for pair in sys.argv[8:] for p, o in [pair.split(':')]]
TIMESTAMP = os.environ.get("TIMESTAMP", datetime.now().strftime("%Y%m%d-%H%M%S"))
SAVE = os.environ.get("SAVE", "1") == "1"
SAVE_DIR = os.environ.get("SAVE_DIR", "bench")

results = []  # accumulate for summary table

# ── Prompt generation ───────────────────────────────────────────────────────
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
        "Transformer architectures typically stack multiple layers of multi-head attention followed "
        "by feed-forward networks, with residual connections and layer normalization around each "
        "sub-layer. This design has proven exceptionally scalable: larger models with more parameters "
        "and more training data consistently yield better performance across NLP benchmarks. "
        "The transformer's ability to leverage parallel computation on GPUs and TPUs has been a key "
        "factor in its adoption, enabling training runs that would be impractical with sequential "
        "architectures. Modern variants like GPT, BERT, T5, and LLaMA have pushed the boundaries "
        "of what language models can achieve, with context lengths growing from 512 tokens to "
        "over 200,000 tokens in recent models. "
    )
    # ~3.5 chars per token for this text (English prose)
    chars_needed = int(target_tokens * 3.8)
    repeats = max(1, chars_needed // len(base)) + 1
    text = (base * repeats)[:chars_needed]
    return text

# ── Single request ──────────────────────────────────────────────────────────
def run_once(prompt, max_tokens, run_label=""):
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
    completion_tokens = 0
    prompt_tokens = 0

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
                    if content and ttft is None:
                        ttft = time.time() - t_send
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:
        return {"error": str(e)}

    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall  # degenerate case: empty response
    prefill_time = ttft
    # If ttft was never detected (prefill_time == wall), use wall for decode too
    decode_time = max(wall - ttft, 1e-6)
    if ttft >= wall * 0.99:  # ttft not properly detected, fall back to wall
        decode_time = wall
    # Only compute TPS if we actually got tokens
    ct = max(completion_tokens, 1)
    return {
        "wall": wall,
        "ttft": ttft,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tps": prompt_tokens / prefill_time if prompt_tokens > 0 and prefill_time > 0 else 0,
        "decode_tps": ct / decode_time,
        "wall_tps": ct / wall,
    }

# ── Format helpers ───────────────────────────────────────────────────────────
def fmt_time(sec):
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    return f"{sec:.1f}s"

# ── Scrape docker logs for throughput metrics ────────────────────────────────
def scrape_metrics(tag):
    """Pull MTP / SpecDecode metrics and prompt throughput from docker logs."""
    if not CONTAINER or CONTAINER == "none" or shutil.which("docker") is None:
        return {}
    try:
        proc = subprocess.run(
            ["docker", "logs", CONTAINER],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", timeout=10, check=False,
        )
    except Exception:
        return {}
    out = proc.stdout
    metrics = {}
    # Avg prompt throughput:
    pp = re.findall(r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", out)
    if pp:
        metrics["avg_prompt_tps"] = float(pp[-1])
    # SpecDecode acceptance rate:
    sa = re.findall(r"SpecDecoding metrics.*?acceptance_rate[=:]\s*([0-9.]+)", out, re.DOTALL)
    if not sa:
        sa = re.findall(r"acceptance_rate[=:]\s*([0-9.]+)", out)
    if sa:
        metrics["acceptance_rate"] = float(sa[-1])
    # Draft efficiency:
    de = re.findall(r"SpecDecoding metrics.*?efficiency[=:]\s*([0-9.]+)", out, re.DOTALL)
    if not de:
        de = re.findall(r"efficiency[=:]\s*([0-9.]+)", out)
    if de:
        metrics["draft_efficiency"] = float(de[-1])
    # MTP target TPS:
    tps = re.findall(r"SpecDecoding metrics.*?target_tps[=:]\s*([0-9.]+)", out, re.DOTALL)
    if not tps:
        tps = re.findall(r"target.*?tps[=:]\s*([0-9.]+)", out, re.DOTALL)
    if tps:
        metrics["target_tps"] = float(tps[-1])
    return metrics

# ── Per-tier test ────────────────────────────────────────────────────────────
def test_tier(prompt_tok, output_tok):
    print(f"\n{'─'*70}")
    prompt = make_prompt(prompt_tok)

    raw = []
    for i in range(WARMUPS):
        r = run_once(prompt, output_tok)
    for i in range(RUNS):
        r = run_once(prompt, output_tok)
        if not r.get("error"):
            raw.append(r)

    if not raw:
        print(f"  {prompt_tok:>6,} prompt → {output_tok:>4,} output  ✗ ALL FAILED")
        return None

    # Aggregate
    m = {}
    for k in ["wall", "ttft", "prefill_time", "decode_time",
              "prefill_tps", "decode_tps", "wall_tps"]:
        vals = [r[k] for r in raw]
        m[k] = {"mean": s.mean(vals), "stdev": s.stdev(vals) if len(vals) > 1 else 0}
    m["prompt_tokens"] = raw[0]["prompt_tokens"]
    m["completion_tokens"] = round(s.mean([r["completion_tokens"] for r in raw]))
    m["n"] = len(raw)

    print(f"  {m['prompt_tokens']:>7,} tok → {output_tok:>4,} out  "
          f"TTFT={fmt_time(m['ttft']['mean']):>7s}  "
          f"prefill={m['prefill_tps']['mean']:>6,.0f} t/s  "
          f"decode={m['decode_tps']['mean']:>6.1f} TPS  "
          f"wall={m['wall_tps']['mean']:>6.1f} TPS")
    return m

# ── Main ─────────────────────────────────────────────────────────────────────
print("", flush=True)

for prompt_tok, output_tok in TIERS:
    m = test_tier(prompt_tok, output_tok)
    if m:
        results.append({**m, "prompt_tok": prompt_tok, "output_tok": output_tok})

# ── Summary table ────────────────────────────────────────────────────────────
if results:
    print(f"\n{'='*70}")
    print(f"  SEQUENTIAL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Prompt':>8s}  {'Output':>7s}  {'Wall TPS':>9s}  {'Decode TPS':>11s}  {'TTFT':>9s}  {'Prefill t/s':>12s}  {'n':>3s}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*12}  {'─'*3}")
    for m in results:
        wall_s = f"{m['wall_tps']['mean']:.1f}"
        if m['wall_tps']['stdev'] > 0.5:
            wall_s += f"±{m['wall_tps']['stdev']:.1f}"
        dec_s = f"{m['decode_tps']['mean']:.1f}"
        if m['decode_tps']['stdev'] > 0.5:
            dec_s += f"±{m['decode_tps']['stdev']:.1f}"
        print(f"  {m['prompt_tok']:>8,}→{m['prompt_tokens']:>8,}  {m['output_tok']:>7,}  {wall_s:>9s}  {dec_s:>11s}  "
              f"{fmt_time(m['ttft']['mean']):>9s}  {m['prefill_tps']['mean']:>10,.0f} t/s  {m['n']:>3d}")

    # Docker metrics (one line)
    docker_m = scrape_metrics("bench")
    if docker_m:
        parts = []
        if "avg_prompt_tps" in docker_m:
            parts.append(f"prompt={docker_m['avg_prompt_tps']:.0f} t/s")
        if "acceptance_rate" in docker_m:
            parts.append(f"accept={docker_m['acceptance_rate']:.2f}")
        if "draft_efficiency" in docker_m:
            parts.append(f"draft_eff={docker_m['draft_efficiency']:.2f}")
        if "target_tps" in docker_m:
            parts.append(f"target_TPS={docker_m['target_tps']:.0f}")
        print(f"")
        print(f"  Docker ({CONTAINER}):  {'  '.join(parts)}")

    # Save to file
    if SAVE:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f"sequential-{TIMESTAMP}.md")
        with open(save_path, "w") as f:
            f.write(f"# Sequential Bench Results — {TIMESTAMP}\n\n")
            f.write(f"- URL: `{URL}`\n")
            f.write(f"- Model: `{MODEL}`\n")
            f.write(f"- Container: `{CONTAINER}`\n")
            f.write(f"- Warmups: {WARMUPS}  Measured: {RUNS}\n")
            f.write(f"- Temperature: {TEMPERATURE}  Top-p: {TOP_P}\n\n")
            f.write("| Prompt | Output | Wall TPS | Decode TPS | TTFT | Prefill t/s | n |\n")
            f.write("|-------|-------|---------|-----------|------|------------|---|\n")
            for m in results:
                f.write(f"| {m['prompt_tok']:,} | {m['output_tok']:,} "
                        f"| {m['wall_tps']['mean']:.1f}±{m['wall_tps']['stdev']:.1f}"
                        f"| {m['decode_tps']['mean']:.1f}±{m['decode_tps']['stdev']:.1f}"
                        f"| {m['ttft']['mean']*1000:.0f}±{m['ttft']['stdev']*1000:.0f}ms"
                        f"| {m['prefill_tps']['mean']:,.0f}±{m['prefill_tps']['stdev']:,.0f}"
                        f"| {m['n']} |\n")
            if docker_m:
                f.write("\n## Docker Metrics\n\n")
                for k, v in docker_m.items():
                    f.write(f"- {k}: {v}\n")
        print(f"\n  Results saved: {save_path}")

print(f"\n{'='*70}")
print(f"  Done. {len(results)} tiers tested.")
print(f"{'='*70}")
print("")
PYEOF

# ── GPU state ───────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi
