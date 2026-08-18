#!/usr/bin/env bash
# Start/stop the live serving dashboard (profiles/dashboard/dashboard.py).
# `start` is idempotent: a running dashboard is killed and replaced (overwrite
# restart) so a fresh process always reads current code. Detached in the
# background; PID + log live under $HOME (runtime state, not repo content).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${DASH_PORT:-8021}"
BIND="${DASH_BIND:-0.0.0.0}"
PY="$(command -v python3)"
RUN_DIR="${DASH_RUN_DIR:-$HOME}"
PIDFILE="$RUN_DIR/.local-ai-dash.pid"
LOGFILE="$RUN_DIR/.local-ai-dash.log"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

running_pid() {
  local p=""
  [ -f "$PIDFILE" ] && p="$(cat "$PIDFILE" 2>/dev/null || true)"
  if alive "$p"; then echo "$p"; return; fi
  p="$(ss -ltnp 2>/dev/null | awk -v pt=":$PORT" '$4==pt{match($0,/pid=[0-9]+/);print substr($0,RSTART+4,RLENGTH-4)}' | head -1 || true)"
  alive "$p" && echo "$p"
  return 0
}

stop() {
  local p; p="$(running_pid)"
  if [ -n "$p" ]; then
    kill "$p" 2>/dev/null || true
    sleep 0.5
    kill -9 "$p" 2>/dev/null || true
    echo "stopped dashboard (pid $p)"
  else
    echo "dashboard not running"
  fi
  rm -f "$PIDFILE"
}

start() {
  local old; old="$(running_pid)"
  if [ -n "$old" ]; then
    kill "$old" 2>/dev/null || true
    sleep 0.5
    kill -9 "$old" 2>/dev/null || true
    echo "overwrote running dashboard (pid $old)"
  fi
  rm -f "$PIDFILE"
  nohup "$PY" "$DIR/dashboard.py" --port "$PORT" --bind "$BIND" >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  disown 2>/dev/null || true
  sleep 0.8
  local np; np="$(cat "$PIDFILE" 2>/dev/null || true)"
  if alive "$np"; then
    echo "dashboard started: http://$BIND:$PORT/ (pid $np, log $LOGFILE)"
  else
    echo "dashboard failed to start; last log lines:" >&2
    tail -n 20 "$LOGFILE" >&2 || true
    exit 1
  fi
}

status() {
  local p; p="$(running_pid)"
  if [ -n "$p" ]; then
    echo "dashboard running (pid $p): http://localhost:$PORT/"
  else
    echo "dashboard not running"
  fi
}

logs() { tail -n "${1:-100}" "$LOGFILE"; }

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs "${2:-100}" ;;
  *) echo "usage: $0 [start|stop|status|logs [N]]"; exit 1 ;;
esac