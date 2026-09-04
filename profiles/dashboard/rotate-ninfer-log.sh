#!/usr/bin/env bash
# Rollover the ninfer structured request log once it exceeds a size cap.
#
# The engine holds requests.jsonl open with a fixed offset (open flags 0x1, no
# O_APPEND), so it cannot be safely truncated in place -- a real size reduction
# needs the writer to release it, i.e. a brief engine stop. Old data is preserved
# as a .gz archive. Runs from a root systemd timer every 6h; under the cap it is
# a quiet no-op.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$(cd "$DIR/.." && pwd)"
NINFER_SH="$PROF/qwen38-27b/ninfer.sh"
CONTAINER="ninfer-qwen38-27b"

F="${NINFER_LOG:-/home/rjman/ninfer-logs/requests.jsonl}"
LOG="${ROTATE_LOG:-/home/rjman/.ninfer-log-rotate.log}"
CAP="${CAP:-1073741824}"   # 1 GiB
KEEP="${KEEP:-7}"          # newest archives to retain
# Engine lifecycle hooks, overridable for testing. start suppresses the dashboard
# restart that ninfer.sh start would otherwise trigger (see NINFER_NO_DASH).
STOP_CMD="${STOP_CMD:-bash $NINFER_SH stop}"
START_CMD="${START_CMD:-NINFER_NO_DASH=1 bash $NINFER_SH start}"

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG" 2>/dev/null || true; }

[ -f "$F" ] || { say "skip: $F not found"; exit 0; }

# Only act while the engine (the live writer) is up -- never restart it just to
# roll a stale historical log.
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  exit 0
fi

size=$(stat -c%s "$F")
if [ "$size" -lt "$CAP" ]; then
  exit 0                     # under cap: quiet no-op
fi

say "rollover: $F is $size B (cap $CAP) -> stop / gzip / reset / start"
eval "$STOP_CMD"
ts="$(date +%Y%m%d-%H%M%S)"
gzip -c "$F" > "$F.$ts.gz"
: > "$F"                     # reset live to empty (safe: writer released by stop)
# Prune old archives, keep the newest $KEEP.
{ ls -1t "$F".*.gz 2>/dev/null || true; } | tail -n +$((KEEP + 1)) | xargs -r rm -f
eval "$START_CMD"
say "rollover done: archived $F.$ts.gz, live reset, engine restarted"
