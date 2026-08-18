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

## Adding a new cell

1. Add a framework script next to the siblings, following the same
   `start/stop/status/logs` shape.
2. Record the checkpoint provenance in the header: the source (HF / ModelScope
   repo), the local path, and the `hf download` command.
3. Add the row/column to the matrix in the root `README.md`.

## Notes (measured)

- SGLang + DSpark speculative decoding on this 5090 deployment gave **no** decode
  throughput gain (77 vs 78 tok/s, within noise) and shrank the KV pool, so DSpark
  is off in both scenarios. Measured pure-decode throughput is ~77-78 tok/s
  (2026-08-15, RadixArk Qwen3.8-27B-NVFP4). Re-check only after
  sglang-project/sglang PR #34742 lands and the image is repulled.