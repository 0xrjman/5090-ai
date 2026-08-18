#!/usr/bin/env bash
#
# Context level stress test — concurrent pairs at increasing context sizes
#
# Divides max_model_len into 10 levels, sends pairs of unique prompts
# concurrently at each level. Simulates mixed-context workloads at
# different scales.
#
# Env vars:
#   URL        endpoint                     (default: http://localhost:8020)
#   MODEL      served model name            (default: local)
#   OUTPUT     output tokens per request    (default: 200)
#   SAVE       0=no save 1=save             (default: 1)
#
set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-local}"
OUTPUT="${OUTPUT:-200}"
SAVE="${SAVE:-1}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found" >&2; exit 1; }; }
need curl
need python3
SAVE_DIR="${SAVE_DIR:-bench}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="${ROOT_DIR}/${SAVE_DIR}"

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

export TIMESTAMP SAVE SAVE_DIR
python3 - "$URL" "$MODEL" "$OUTPUT" "$TIMESTAMP" << 'PYEOF'
import json, os, random, sys, time, statistics as s, urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = sys.argv[1]
MODEL = sys.argv[2]
OUTPUT = int(sys.argv[3])
TIMESTAMP = sys.argv[4]
SAVE = os.environ.get("SAVE", "1") == "1"
SAVE_DIR = os.environ.get("SAVE_DIR", "bench")

# ── Step 1: Detect max_model_len ────────────────────────────
def get_max_model_len():
    """Query vLLM model config to get max_model_len."""
    try:
        req = urllib.request.Request(f"{URL}/v1/models")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            model_data = data["data"][0]
            # vLLM exposes max_model_len in model metadata
            mmax = model_data.get("max_model_len") or \
                   model_data.get("metadata", {}).get("max_model_len")
            if mmax:
                return int(mmax)
    except Exception:
        pass
    # Fallback: try to read from env or use default
    return 262144

MAX_CTX = get_max_model_len()
print(f"  Detected max_model_len: {MAX_CTX:,} tokens")
print()

# ── Step 2: Create 10 context levels ─────────────────────────
def round_10k(n):
    """Round to nearest 10K."""
    return max(1000, round(n / 10000) * 10000)

levels = []
for i in range(1, 11):
    raw = int(MAX_CTX * i / 10)
    rounded = round_10k(raw)
    levels.append(rounded)

# Deduplicate and ensure strictly increasing
deduped = []
for v in levels:
    if not deduped or v != deduped[-1]:
        deduped.append(v)
# If dedup reduces count, adjust levels slightly
if len(deduped) < 10:
    deduped = []
    for i in range(1, 11):
        raw = int(MAX_CTX * i / 10)
        # Round to different granularities per level to avoid collisions
        granularity = 1000 if i <= 3 else (5000 if i <= 6 else 10000)
        v = max(1000, round(raw / granularity) * granularity)
        if deduped and v <= deduped[-1]:
            v = deduped[-1] + granularity
        deduped.append(v)
levels = deduped

print(f"  {'Level':>6s}  {'Tokens':>10s}  {'% of max':>8s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
for i, ctx in enumerate(levels, 1):
    pct = ctx / MAX_CTX * 100
    print(f"  {i:>6d}  {ctx:>10,}  {pct:>7.1f}%")
print()

# ── Step 3: Generate unique prompts per level ────────────────
# Different topics to produce different KV content (avoid prefix caching)
TOPICS = [
    "quantum computing and superposition states in multi-qubit systems",
    "Renaissance art techniques in fresco painting and chiaroscuro",
    "marine biology of deep-sea hydrothermal vent ecosystems",
    "TensorRT-LLM inference optimization for transformer models",
    "Baroque music ornamentation in Bach's Brandenburg concertos",
    "CRISPR-Cas9 gene editing mechanisms and ethical considerations",
    "distributed consensus algorithms in blockchain networks",
    "medieval siege warfare engineering and castle fortifications",
    "protein folding prediction with AlphaFold and RoseTTAFold",
    "electoral college history and reform proposals",
]

def make_unique_prompt(target_tokens, seed):
    """Generate a topic-specific prompt of roughly target_tokens length."""
    rng = random.Random(seed)
    topic = TOPICS[(seed - 1) % len(TOPICS)]
    # Build a paragraph of ~200 tokens from the topic
    sentences = [
        f"The study of {topic} reveals fascinating complexities that researchers continue to explore.",
        f"Recent advances in {topic} have opened new avenues for investigation and practical applications.",
        f"A comprehensive understanding of {topic} requires examining multiple perspectives and methodologies.",
        f"The historical development of {topic} shows how scientific understanding evolves over time.",
        f"Key findings in {topic} have significant implications for both theory and practice.",
        f"Experts in {topic} debate several unresolved questions that drive current research agendas.",
        f"The interdisciplinary nature of {topic} connects insights from diverse fields of study.",
        f"Methodological approaches to studying {topic} have improved substantially in recent years.",
        f"Future research directions in {topic} promise to address current limitations and gaps.",
        f"Practical applications emerging from {topic} demonstrate its real-world relevance.",
    ]
    base = " ".join(sentences)
    chars_needed = int(target_tokens * 3.8)
    repeats = max(1, chars_needed // len(base)) + 2
    text = (base * repeats)[:chars_needed]
    return text

# ── Step 4: Send request and measure ─────────────────────────
def send_request(ctx_tokens, seed, output_tokens):
    """Send a single request via curl for true streaming, return (ctx, ttft, decode_tps, prompt_tokens, completion_tokens, elapsed, ok, err)."""
    import subprocess as sp
    import shlex
    
    prompt = make_unique_prompt(ctx_tokens, seed)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    
    start_time = time.monotonic()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    
    try:
        proc = sp.Popen(
            ["curl", "-sN", "--max-time", "600",
             "-H", "Content-Type: application/json",
             "-d", "@-", f"{URL}/v1/chat/completions"],
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=0)
        proc.stdin.write(body.encode())
        proc.stdin.close()
        
        buf = b""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            buf += ch
            if ch == b"\n":
                raw_line = buf.decode("utf-8", errors="ignore").rstrip()
                buf = b""
                if not raw_line.startswith("data: "):
                    continue
                payload = raw_line[6:]
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
                        ttft = time.monotonic() - start_time
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
                    completion_tokens = usage.get("completion_tokens", completion_tokens) or completion_tokens
        
        proc.stdout.close()
        proc.wait(timeout=5)
        end_time = time.monotonic()
    except Exception as e:
        return (ctx_tokens, None, None, 0, 0, None, False, str(e))
    
    if ttft is None:
        ttft = end_time - start_time
    
    elapsed = end_time - start_time
    decode_time = elapsed - ttft
    decode_tps = completion_tokens / decode_time if decode_time > 0.001 else completion_tokens / elapsed if elapsed > 0.001 else 0
    
    # Debug TPS
    if completion_tokens > 0:
        sys.stderr.write(f"  [dbg] ctx={ctx_tokens} ttft={ttft:.3f}s elapsed={elapsed:.3f}s decode_t={decode_time:.3f}s ct={completion_tokens} tps={decode_tps:.1f}\n")
        sys.stderr.flush()
    
    return (ctx_tokens, ttft, decode_tps, prompt_tokens, completion_tokens, elapsed, True, None)

# ── Step 5: Run concurrent pairs ─────────────────────────────
print("╔══════════════════════════════════════════════════════════════╗")
print("║  Context Level Stress Test — Concurrent Pairs              ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"  Model:        {MODEL}")
print(f"  Max ctx:      {MAX_CTX:,}")
print(f"  Output:       {OUTPUT} tokens")
print()

all_results = []
pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

header = f"{'Pair':>6s}  {'Ctx1':>8s}  {'Ctx2':>8s}  {'TTFT1':>8s}  {'TTFT2':>8s}  {'TPS1':>8s}  {'TPS2':>8s}  {'Tkn1':>6s}  {'Tkn2':>6s}  {'Status':>8s}"
sep = "-" * len(header)

print(f"  {sep}")
print(f"  {header}")
print(f"  {sep}")

for pair_idx, (l1, l2) in enumerate(pairs, 1):
    ctx1 = levels[l1 - 1]
    ctx2 = levels[l2 - 1]
    print(f"  Pair {pair_idx}: levels {l1}({ctx1:,}) + {l2}({ctx2:,}) — submitting...", end="")
    
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(send_request, ctx1, l1 * 100, OUTPUT)
        f2 = ex.submit(send_request, ctx2, l2 * 100, OUTPUT)
        r1 = f1.result()
        r2 = f2.result()
    
    all_results.extend([r1, r2])
    
    def fmt_t(v):
        if v is None:
            return "  OOM  "
        return f"{v*1000:>6.0f}ms" if v < 1 else f"{v:>6.1f}s"
    def fmt_tps(v):
        if v is None:
            return "  FAIL "
        return f"{v:>6.0f}" if v < 10 else f"{v:>6.1f}"
    def fmt_prompt(v):
        return f"{v:>6,}" if v else "     -"
    def fmt_status(r):
        if r[6]:
            return "   OK  "
        return f"FAIL:{r[7][:12]}" if r[7] else "  ???  "
    
    print(f"\r  Pair {pair_idx}: levels {l1}({ctx1:>7,}) + {l2}({ctx2:>7,})  "
          f"{fmt_t(r1[1])}  {fmt_t(r2[1])}  {fmt_tps(r1[2])}  {fmt_tps(r2[2])}  "
          f"{fmt_prompt(r1[3])}  {fmt_prompt(r2[3])}  "
          f"{fmt_status(r1)}/{fmt_status(r2)}")

print(f"  {sep}")
print()

# ── Summary ──────────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════════════╗")
print("║  Summary                                                    ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()
print(f"  {'Level':>5s}  {'Ctx':>8s}  {'TTFT':>8s}  {'DecTPS':>8s}  {'pTkn':>6s}  {'cTkn':>6s}  {'Result':>8s}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

success_count = 0
fail_count = 0
ttft_values = []
tps_values = []

for i, r in enumerate(all_results, 1):
    ctx, ttft, dec_tps, pt, ct, et, ok, err = r
    if ok:
        status = "OK"
        success_count += 1
        if ttft is not None:
            ttft_values.append(ttft)
        if dec_tps is not None:
            tps_values.append(dec_tps)
    else:
        status = f"FAIL"
        fail_count += 1
    
    print(f"  {i:>5d}  {ctx:>8,}  {ttft*1000:>6.0f}ms" if ttft and ok else
          f"  {i:>5d}  {ctx:>8,}  {'  N/A  '}",
          end="")
    if ok:
        print(f"  {dec_tps:>6.0f}  {pt:>6,}  {ct:>6,}  {status:>8s}")
    else:
        print(f"  {'  N/A  '}  {'    -'}  {'    -'}  FAIL:{err[:15]}")

print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

print()
print(f"  Total requests: {len(all_results)}  |  OK: {success_count}  |  FAIL: {fail_count}")
if ttft_values:
    print(f"  TTFT avg:  {s.mean(ttft_values)*1000:.0f}ms  |  min: {min(ttft_values)*1000:.0f}ms  |  max: {max(ttft_values)*1000:.0f}ms")
if tps_values:
    print(f"  DecTPS avg: {s.mean(tps_values):.1f}  |  min: {min(tps_values):.1f}  |  max: {max(tps_values):.1f}")
print()

# ── Save ─────────────────────────────────────────────────────
if SAVE:
    md_file = f"{SAVE_DIR}/ctx-levels-{TIMESTAMP}.md"
    with open(md_file, "w") as f:
        f.write(f"# Context Level Stress Test — {TIMESTAMP}\n\n")
        f.write(f"| Param | Value |\n|---|---|\n")
        f.write(f"| Max ctx | {MAX_CTX:,} |\n")
        f.write(f"| Output | {OUTPUT} |\n")
        f.write(f"| URL | {URL} |\n\n")
        f.write("## Results\n\n")
        f.write("| # | Ctx | TTFT | DecTPS | pTkn | cTkn | Result |\n")
        f.write("|---|-----|------|--------|------|------|--------|\n")
        for i, r in enumerate(all_results, 1):
            ctx, ttft, dec_tps, pt, ct, et, ok, err = r
            if ok:
                f.write(f"| {i} | {ctx:,} | {ttft*1000:.0f}ms | {dec_tps:.0f} | {pt:,} | {ct:,} | OK |\n")
            else:
                f.write(f"| {i} | {ctx:,} | N/A | N/A | - | - | FAIL: {err[:30]} |\n")
        f.write("\n## Summary\n\n")
        if ttft_values:
            f.write(f"- TTFT avg: {s.mean(ttft_values)*1000:.0f}ms\n")
            f.write(f"- TTFT min: {min(ttft_values)*1000:.0f}ms\n")
            f.write(f"- TTFT max: {max(ttft_values)*1000:.0f}ms\n")
        if tps_values:
            f.write(f"- DecTPS avg: {s.mean(tps_values):.1f}\n")
            f.write(f"- DecTPS min: {min(tps_values):.1f}\n")
            f.write(f"- DecTPS max: {max(tps_values):.1f}\n")
        f.write(f"- OK: {success_count}/{len(all_results)}\n")
        f.write(f"- FAIL: {fail_count}/{len(all_results)}\n")
    print(f"  Saved: {md_file}")

print()
print("Done.")
PYEOF
