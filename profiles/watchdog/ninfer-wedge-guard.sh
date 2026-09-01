#!/usr/bin/env bash
# Detect and recover the ninfer engine wedge.
#
# engine_core.h worker_loop() catches any exception, calls fail_all_locked() and
# then RETURNS: the worker thread is gone for good, but the process stays alive
# and the HTTP layer keeps answering. So nothing external notices -- the port
# listens, `docker ps` says Up, `--restart unless-stopped` never fires, and
# /v1/models still returns 200 because it never touches the engine. Only a real
# generation request sees it, as 503 service_unavailable.
#
# cron: * * * * * $HOME/Servers/local-ai/profiles/watchdog/ninfer-wedge-guard.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/../../.env" ] && { set -a; . "$DIR/../../.env"; set +a; }

CONTAINER=ninfer-qwen38-27b
PORT=8020
LOG=$HOME/ninfer-wedge-guard.log
STATE=$HOME/.ninfer-wedge-guard.strikes
STRIKES_NEEDED=2

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || { echo 0 > "$STATE"; exit 0; }

body=$(curl -s --max-time 30 \
  -H 'content-type: application/json' -H "x-api-key: ${API_KEY:-}" \
  -d '{"model":"local","max_tokens":1,"messages":[{"role":"user","content":"ping"}]}' \
  "http://127.0.0.1:$PORT/v1/messages" 2>/dev/null || true)

case "$body" in
  *service_unavailable*|*"engine is unavailable"*) ;;
  *) echo 0 > "$STATE"; exit 0 ;;
esac

strikes=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$strikes" > "$STATE"
log "wedge probe failed ($strikes/$STRIKES_NEEDED)"
[ "$strikes" -ge "$STRIKES_NEEDED" ] || exit 0

log "wedged -- saving log tail and restarting $CONTAINER"
mkdir -p "$HOME/ninfer-crash"
docker logs --tail 400 "$CONTAINER" > "$HOME/ninfer-crash/wedge-$(date +%Y%m%dT%H%M%S).log" 2>&1 || true
docker restart "$CONTAINER" >/dev/null 2>&1 || true
echo 0 > "$STATE"
log "restarted"
