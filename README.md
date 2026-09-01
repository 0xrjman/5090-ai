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

## Using Claude Code

When driving this repo through Claude Code against the local serving
endpoint (non-first-party `ANTHROPIC_BASE_URL`), add to `~/.claude/settings.json`:

    {
      "env": {
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "ENABLE_TOOL_SEARCH": "false"
      }
    }

`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` strips `anthropic-beta` headers the
proxy rejects (hard 400s otherwise) and also forces tool search off.
If the endpoint 400s on `thinking: adaptive`, add
`"CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"`.

## Notes (measured)

- SGLang + DSpark speculative decoding on this 5090 deployment gave **no** decode
  throughput gain (77 vs 78 tok/s, within noise) and shrank the KV pool, so DSpark
  is off in both scenarios. Measured pure-decode throughput is ~77-78 tok/s
  (2026-08-15, RadixArk Qwen3.8-27B-NVFP4). Re-check only after
  sglang-project/sglang PR #34742 lands and the image is repulled.