#!/usr/bin/env python3
"""Live LLM serving dashboard.

Tails the running LLM container's docker logs (vLLM / SGLang / NInfer),
parses throughput + per-request lines, and serves a self-updating HTML status
page plus a JSON API. Stdlib only, zero GPU.

The NInfer log format is fully supported. For any other framework the dashboard
still shows the container state + live log tail; add a line-parser to PARSERS
to get metrics for it.

Log source is `docker logs` (the engine API), not the on-disk json file, which
is root-only. A background thread polls every --poll seconds; metrics are
computed over a selectable window (default 5m).

Run:  python3 dashboard.py [--port 8021] [--bind 0.0.0.0] [--poll 1.5] [--tail 200]
"""
import argparse
import datetime
import json
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# container name -> framework
SERVED = [("vllm-qwen38", "vllm"), ("sglang-qwen38", "sglang"), ("ninfer-qwen38-27b", "ninfer")]
SERVED_NAMES = {n for n, _ in SERVED}
SERVED_FW = dict(SERVED)
SERVE_API_PORT = 8020   # OpenAI endpoint all profiles expose

MAX_TP = 6000           # ~5h of 5s throughput samples
MAX_REQ = 2000
MAX_ERR = 500
MAX_RAW = 400           # recent raw lines kept for the live-log panel
MAX_SEEN = 1600         # LRU bound for log-line dedupe

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\]")
KV_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
REQ_RE = re.compile(r"\[req (\d+)\]")
MTP_RE = re.compile(r"(\d+(?:\.\d+)?)tok/round \((\d+(?:\.\d+)?)%\)")


def _num(s):
    if s is None:
        return None
    m = re.match(r"[-+]?\d+(?:\.\d+)?", str(s).strip())
    return float(m.group(0)) if m else None


def _f(s):
    return _num(s)


def _i(s):
    v = _num(s)
    return int(v) if v is not None else None


def _kv(line):
    return {k: v for k, v in KV_RE.findall(line)}


def _line_ts(line):
    m = TS_RE.match(line)
    if not m:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(m.group(1), fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_ninfer(line):
    """Return an event dict for a metric-bearing NInfer line, else None."""
    t = _line_ts(line)
    if "ninfer-serve: throughput" in line and "interval=" in line:
        kv = _kv(line)
        return {"kind": "tp", "t": t,
                "prefill": _f(kv.get("prefill")), "decode": _f(kv.get("decode")),
                "running": _i(kv.get("running")), "prefilling": _i(kv.get("prefilling")),
                "decode_ready": _i(kv.get("decode_ready")), "waiting": _i(kv.get("waiting")),
                "avg_decode_batch": _f(kv.get("avg_decode_batch"))}
    req = REQ_RE.search(line)
    if req:
        rid = req.group(1)
        if "] done" in line:
            kv = _kv(line)
            m = MTP_RE.search(line)
            return {"kind": "done", "t": t, "req": rid,
                    "finish": kv.get("finish"), "tool_calls": _i(kv.get("tool_calls")),
                    "prompt": _i(kv.get("prompt")), "gen": _i(kv.get("gen")),
                    "cache": _i(kv.get("cache")), "reuse": kv.get("reuse"),
                    "ttft_ms": _f(kv.get("ttft")), "prefill": _f(kv.get("prefill")),
                    "decode": _f(kv.get("decode")), "wall": _f(kv.get("wall")),
                    "mtp_round": float(m.group(1)) if m else None,
                    "mtp_pct": float(m.group(2)) if m else None}
        if "] error" in line:
            return {"kind": "err", "t": t, "req": rid, "msg": line.split("error", 1)[1].strip()}
        if "] rejected" in line:
            kv = _kv(line)
            msg = line.split("message=", 1)[1].strip() if "message=" in line else ""
            return {"kind": "rej", "t": t, "req": rid, "status": _i(kv.get("status")),
                    "code": kv.get("code"), "msg": msg}
        if "submitted" in line:
            kv = _kv(line)
            proto = ("anthropic" if "anthropic_messages" in line
                     else "openai" if "openai_chat_completions" in line else "?")
            return {"kind": "sub", "t": t, "req": rid, "proto": proto,
                    "stream": "non-stream" not in line,
                    "msgs": _i(kv.get("msgs")), "max_tokens": _i(kv.get("max_tokens"))}
    return None


# framework -> line parser. Add a parser here to light up metrics for a framework.
PARSERS = {"ninfer": parse_ninfer}


def docker(*args):
    # Merge stderr: docker logs replays the container's stderr on the CLI stderr,
    # and ninfer (and most servers) log to stderr -- reading only stdout loses
    # the whole recent window.
    try:
        return subprocess.run(["docker", *args], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=15).stdout or ""
    except Exception:
        return ""


def pull_lines(st, name, tail):
    """Return this poll's fresh (not-yet-seen) log lines, appended to st.raw."""
    fresh = []
    for ln in docker("logs", "--tail", str(tail), name).splitlines():
        if not ln or ln in st._seen:
            continue
        st.seen_add(ln)
        st.raw.append(ln)
        fresh.append(ln)
    return fresh


def restamp(evs, now):
    """Re-anchor sample times to real wall clock.

    Container log clocks are not reliably the host clock (ninfer logs UTC, this
    host is UTC+8), so absolute log timestamps cannot be trusted. Timestamps
    only matter as deltas, so anchor the newest line of the batch at `now` and
    back-fill the rest by their log-relative age. TZ-invariant either way.
    """
    anchor = max((ev["t"] for ev in evs if ev.get("t") is not None), default=None)
    for ev in evs:
        if ev.get("t") is not None and anchor is not None:
            ev["t"] = now - (anchor - ev["t"])
        else:
            ev["t"] = now


def discover():
    """Return (name, running, status, image) for the served container: prefer a
    running one, else the first known container present."""
    rows = []
    for line in docker("ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}").splitlines():
        p = line.split("\t")
        if p and p[0] in SERVED_NAMES:
            rows.append((p[0], (p[1] if len(p) > 1 else "").startswith("Up"),
                         p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else ""))
    if not rows:
        return None
    for r in rows:
        if r[1]:
            return r
    return rows[0]


class Store:
    def __init__(self):
        self.target = None
        self.framework = None
        self.running = False
        self.state_str = ""
        self.image = ""
        self.tp = deque(maxlen=MAX_TP)
        self.req = deque(maxlen=MAX_REQ)
        self.sub = deque(maxlen=MAX_REQ)
        self.err = deque(maxlen=MAX_ERR)
        self.raw = deque(maxlen=MAX_RAW)
        self.inflight = {}
        self._seen_q = deque()
        self._seen = set()
        self.last_line_ts = 0
        self.last_poll = time.time()
        self.docker_ok = True
        self.boot = time.time()
        self.lock = threading.Lock()

    def reset_target(self, name):
        self.tp.clear()
        self.req.clear()
        self.sub.clear()
        self.err.clear()
        self.raw.clear()
        self.inflight.clear()
        self._seen.clear()
        self._seen_q.clear()
        self.last_line_ts = 0
        self.target = name
        self.framework = SERVED_FW.get(name, name)

    def seen_add(self, ln):
        self._seen.add(ln)
        self._seen_q.append(ln)
        if len(self._seen_q) > MAX_SEEN:
            self._seen.discard(self._seen_q.popleft())

    def ingest(self, ev, now):
        if ev.get("t") is None:
            ev["t"] = now
        k = ev["kind"]
        rid = ev.get("req")
        if k == "tp":
            self.tp.append(ev)
        elif k == "done":
            self.req.append(ev)
            if rid is not None:
                self.inflight.pop(rid, None)
        elif k == "sub":
            self.sub.append(ev)
            if rid is not None:
                self.inflight[rid] = {"req": rid, "t": ev["t"], "proto": ev.get("proto"),
                                      "stream": ev.get("stream"), "msgs": ev.get("msgs"),
                                      "max_tokens": ev.get("max_tokens")}
        elif k == "err":
            self.err.append(ev)
            if rid is not None:
                self.inflight.pop(rid, None)
        elif k == "rej":
            self.err.append({"t": ev["t"], "req": rid,
                             "msg": f"{ev.get('status', '?')} {ev.get('code', '?')}: {ev.get('msg', '')}"})
            if rid is not None:
                self.inflight.pop(rid, None)
        if ev.get("t"):
            self.last_line_ts = max(self.last_line_ts, ev["t"])


def poll_loop(st, tail):
    while True:
        d = discover()
        now = time.time()
        with st.lock:
            st.last_poll = now
            if d is None:
                st.docker_ok = True
                st.running = False
                if st.target is None:
                    st.state_str = "no served container found"
            else:
                name, running, status, image = d
                st.docker_ok = True
                if name != st.target:
                    st.reset_target(name)
                st.running = running
                st.state_str = status
                st.image = image
                if running:
                    fresh = pull_lines(st, name, tail)
                    parser = PARSERS.get(st.framework)
                    if parser is not None and fresh:
                        evs = [parser(ln) for ln in fresh]
                        evs = [ev for ev in evs if ev is not None]
                        if evs:
                            restamp(evs, now)
                            for ev in evs:
                                st.ingest(ev, now)
        time.sleep(POLL)


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = int(k) + 1
    if c >= len(s):
        return round(s[-1], 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)


def _downsample(pts, buckets=140):
    """Keep the last point per time bucket -> a light polyline for the chart."""
    if not pts:
        return []
    t0, t1 = pts[0][0], pts[-1][0]
    if t1 <= t0:
        return [[round(t0, 3), pts[0][1], pts[0][2]]]
    step = (t1 - t0) / buckets
    out = []
    for i in range(buckets):
        lo, hi = t0 + i * step, t0 + (i + 1) * step
        for p in pts:
            if lo <= p[0] < hi:
                out.append([round(p[0], 3), p[1], p[2]])
    if not out or out[-1][0] != round(t1, 3):
        out.append([round(t1, 3), pts[-1][1], pts[-1][2]])
    return out


def _downsample_series(pts, buckets=140):
    """Per-bucket last point, arbitrary tuple width -> light polyline set."""
    if not pts:
        return []
    t0, t1 = pts[0][0], pts[-1][0]
    if t1 <= t0:
        return [[round(t0, 3)] + list(pts[0][1:])]
    step = (t1 - t0) / buckets
    out = []
    for i in range(buckets):
        lo, hi = t0 + i * step, t0 + (i + 1) * step
        chosen = None
        for p in pts:
            if lo <= p[0] < hi:
                chosen = p
        if chosen is not None:
            out.append([round(chosen[0], 3)] + list(chosen[1:]))
    if not out or out[-1][0] != round(t1, 3):
        out.append([round(t1, 3)] + list(pts[-1][1:]))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def build_state(st, window):
    now = time.time()
    c0 = now - window

    with st.lock:
        tp_win = [s for s in st.tp if (s["t"] or 0) >= c0]
        latest = st.tp[-1] if st.tp else None
        series = _downsample([(s["t"], s["decode"] or 0, s["prefill"] or 0) for s in tp_win])

        dec = [s["decode"] for s in tp_win]
        pre = [s["prefill"] for s in tp_win]
        active = [s for s in tp_win if (s["running"] or 0) > 0]

        tp_metrics = {
            "avg_decode": _mean(dec), "peak_decode": round(max(dec), 1) if dec else None,
            "avg_prefill": _mean(pre), "peak_prefill": round(max(pre), 1) if pre else None,
            "active_pct": round(100.0 * len(active) / len(tp_win), 1) if tp_win else None,
            "samples": len(tp_win),
        }

        if latest:
            live = {
                "decode_tps": latest["decode"], "prefill_tps": latest["prefill"],
                "running": latest["running"], "prefilling": latest["prefilling"],
                "decode_ready": latest["decode_ready"], "waiting": latest["waiting"],
                "avg_decode_batch": latest["avg_decode_batch"],
                "idle": ((latest["decode"] or 0) == 0 and (latest["prefill"] or 0) == 0
                         and (latest["running"] or 0) == 0),
            }
        else:
            live = {k: None for k in ("decode_tps", "prefill_tps", "running", "prefilling",
                                      "decode_ready", "waiting", "avg_decode_batch")}
            live["idle"] = None

        done = [r for r in st.req if (r["t"] or 0) >= c0]
        subs = [s for s in st.sub if (s["t"] or 0) >= c0]
        errs = [e for e in st.err if (e["t"] or 0) >= c0]
        if done:
            gen = [r["gen"] for r in done]
            prompt = [r["prompt"] for r in done]
            wall = [r["wall"] for r in done]
            sum_gen = sum(g for g in gen if g is not None)
            sum_wall = sum(w for w in wall if w is not None)
            reuse = {}
            for r in done:
                if r.get("reuse"):
                    reuse[r["reuse"]] = reuse.get(r["reuse"], 0) + 1
            req_agg = {
                "total_gen": int(sum_gen), "total_prompt": int(sum(g for g in prompt if g is not None)),
                "agg_tok_s": round(sum_gen / sum_wall, 1) if sum_wall > 0 else None,
                "avg_ttft_ms": round(sum(t for t in [r["ttft_ms"] for r in done] if t is not None)
                                     / max(1, len([t for t in [r["ttft_ms"] for r in done] if t is not None])), 0),
                "p50_ttft_ms": _pct([r["ttft_ms"] for r in done], 50),
                "p95_ttft_ms": _pct([r["ttft_ms"] for r in done], 95),
                "avg_wall_s": round(sum(w for w in wall if w is not None)
                                    / max(1, len([w for w in wall if w is not None])), 2),
                "max_wall_s": round(max(w for w in wall if w is not None), 2),
                "avg_decode_tps": _mean([r["decode"] for r in done]),
                "avg_mtp_pct": _mean([r["mtp_pct"] for r in done]),
                "avg_mtp_round": _mean([r["mtp_round"] for r in done]),
                "reuse": reuse,
            }
        else:
            req_agg = {}
        req_counts = {"done": len(done), "submitted": len(subs), "errors": len(errs)}

        recent_req = [{"t": r["t"], "req": r["req"], "finish": r["finish"],
                       "prompt": r["prompt"], "gen": r["gen"], "ttft_ms": r["ttft_ms"],
                       "decode": r["decode"], "wall": r["wall"], "mtp_pct": r["mtp_pct"]}
                      for r in list(st.req)[-30:]][::-1]
        recent_err = [{"t": e["t"], "req": e["req"], "msg": e["msg"]}
                      for e in list(st.err)[-20:]][::-1]
        raw = list(st.raw)[-140:]

        # In-flight streams: submitted but not yet terminal (done/error/reject).
        # No per-stream heartbeat exists in the log, so the honest "pressure" a
        # stream is applying right now = how long it has held the decoder (age_s,
        # grows each poll) plus its declared token ceiling. Sorted heaviest first.
        streams = []
        for rid, info in st.inflight.items():
            age = (now - info["t"]) if info.get("t") else None
            streams.append({"req": rid, "proto": info.get("proto"),
                            "stream": info.get("stream"), "msgs": info.get("msgs"),
                            "max_tokens": info.get("max_tokens"),
                            "age_s": round(age, 1) if age is not None else None})
        streams.sort(key=lambda s: (s["age_s"] if s["age_s"] is not None else -1.0),
                     reverse=True)
        # Concurrency over time (running/prefilling/waiting) from the throughput samples.
        conc_series = _downsample_series(
            [(s["t"], s["running"] or 0, s["prefilling"] or 0, s["waiting"] or 0)
             for s in tp_win])

        target = st.target
        framework = st.framework
        running = st.running
        state_str = st.state_str
        image = st.image
        last_seen = st.last_line_ts
        docker_ok = st.docker_ok
        boot = st.boot

    dec_vals = [p[1] for p in series]
    pre_vals = [p[2] for p in series]
    peak = max([max(dec_vals) if dec_vals else 0, max(pre_vals) if pre_vals else 0], default=0)

    return {
        "server": {"now": round(now, 3), "last_poll": round(st.last_poll, 3),
                   "boot": round(boot, 3), "poll": POLL, "docker_ok": docker_ok},
        "container": {"name": target, "framework": framework, "running": running,
                      "state": state_str, "image": image, "api_port": SERVE_API_PORT,
                      "last_seen": round(last_seen, 3),
                      "has_parser": framework in PARSERS},
        "window": window,
        "live": live,
        "tp": tp_metrics,
        "peak": round(peak, 1) or 100,
        "series": series,
        "streams": streams,
        "stream_counts": {"active": len(streams),
                          "max_age_s": streams[0]["age_s"] if streams else None},
        "conc_series": conc_series,
        "requests": {**req_counts, **req_agg},
        "recent_requests": recent_req,
        "recent_errors": recent_err,
        "log": raw,
    }


class Handler(BaseHTTPRequestHandler):
    store = None
    window_default = 300

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        elif u.path == "/api/state":
            try:
                window = int(q.get("window", [self.window_default])[0])
            except (ValueError, TypeError):
                window = self.window_default
            window = max(10, min(86400, window))
            self._send(200, json.dumps(build_state(self.store, window)), "application/json")
        elif u.path == "/api/log":
            try:
                tail = max(1, min(1000, int(q.get("tail", [150])[0])))
            except (ValueError, TypeError):
                tail = 150
            with self.store.lock:
                lines = list(self.store.raw)[-tail:]
                name = self.store.target
            self._send(200, json.dumps({"container": name, "lines": lines}), "application/json")
        else:
            self._send(404, "not found", "text/plain")


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>local-ai serving</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2129;--border:#21262d;--txt:#c9d1d9;
--muted:#8b949e;--dim:#6e7681;--green:#3fb950;--blue:#58a6ff;--amber:#d29922;
--red:#f85149;--cyan:#39c5cf;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);font-size:14px}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1240px;margin:0 auto;padding:18px 18px 60px}
.top{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:14px 16px;
background:var(--panel);border:1px solid var(--border);border-radius:12px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted);flex:0 0 auto}
.dot.up{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.err{background:var(--red);box-shadow:0 0 8px var(--red)}
h1{font-size:16px;margin:0;font-weight:600}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--panel2);
border:1px solid var(--border);border-radius:999px;padding:5px 12px;font-size:13px}
.chip b{font-weight:600;color:var(--txt)}
.spacer{flex:1}
.wins{display:flex;gap:4px}
.wins button{background:var(--panel2);border:1px solid var(--border);color:var(--muted);
border-radius:7px;padding:5px 11px;font-size:12px;cursor:pointer;font-family:var(--mono)}
.wins button.on{background:#243044;border-color:#3d5aa0;color:var(--blue)}
.meta{font-size:12px;color:var(--dim);font-family:var(--mono)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:13px 14px}
.card .k{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.card .v{font-size:26px;font-weight:650;margin-top:5px;font-family:var(--mono);line-height:1}
.card .u{font-size:12px;color:var(--dim);font-family:var(--mono);margin-top:3px}
.card.dec .v{color:var(--blue)}
.card.pre .v{color:var(--amber)}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;
padding:14px 16px;margin-top:12px}
.panel h2{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
margin:0 0 10px;font-weight:600}
.panel h2 .note{color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:860px){.row{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 16px}
.kv .i .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.kv .i .n{font-size:17px;font-family:var(--mono);font-weight:600;margin-top:2px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
th{color:var(--muted);text-align:right;font-weight:500;padding:5px 8px;border-bottom:1px solid var(--border)}
td{padding:5px 8px;text-align:right;border-bottom:1px solid #1a2029}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:var(--panel2)}
.err{color:var(--red)}
.muted{color:var(--muted)}
.tag{display:inline-block;background:var(--panel2);border:1px solid var(--border);
border-radius:5px;padding:1px 7px;font-family:var(--mono);font-size:11.5px;margin:2px 3px 2px 0}
.log{background:#0a0d12;border:1px solid var(--border);border-radius:10px;padding:10px 12px;
font-family:var(--mono);font-size:11.8px;line-height:1.55;max-height:340px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.log .e{color:var(--red)}
.log .t{color:var(--dim)}
canvas{width:100%;height:210px;display:block}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:6px;font-family:var(--mono)}
.legend i{display:inline-block;width:11px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:11px;font-family:var(--mono);
border:1px solid var(--border);background:var(--panel2)}
.badge.idle{color:var(--dim)}
.badge.busy{color:var(--green);border-color:#234f32}
.badge.pending{color:var(--amber);border-color:#4a3a12}
.streams{display:flex;flex-direction:column;gap:9px;min-height:58px}
.srow{background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:9px 12px;
  display:grid;grid-template-columns:1fr auto;gap:5px 10px;align-items:center}
.srow .id{font-family:var(--mono);font-weight:650;font-size:13px}
.srow .meta{font-size:11.5px;color:var(--muted);font-family:var(--mono)}
.srow .age{font-family:var(--mono);font-size:19px;font-weight:700;text-align:right;line-height:1}
.srow .bar{grid-column:1 / -1;height:6px;background:#0a0d12;border-radius:3px;overflow:hidden}
.srow .bar i{display:block;height:100%;border-radius:3px;transition:width .4s}
.p0{background:var(--green)}
.p1{background:var(--amber)}
.p2{background:var(--red)}
.empty{color:var(--dim);font-family:var(--mono);font-size:12.5px;padding:16px 4px}
canvas.conc{height:150px}
footer{margin-top:18px;color:var(--dim);font-size:12px;font-family:var(--mono);line-height:1.7}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="dot" id="dot"></div>
    <h1>local-ai serving</h1>
    <span class="chip" id="chip"><b>&mdash;</b></span>
    <span class="badge" id="badge">&mdash;</span>
    <div class="spacer"></div>
    <div class="wins" id="wins"></div>
    <span class="meta" id="meta"></span>
  </div>

  <div class="grid">
    <div class="card dec"><div class="k">Decode TPS &middot; now</div><div class="v" id="c_dec">-</div><div class="u">tokens/s</div></div>
    <div class="card dec"><div class="k">Decode TPS &middot; avg</div><div class="v" id="c_dec_avg">-</div><div class="u" id="u_dec_avg">window</div></div>
    <div class="card dec"><div class="k">Decode TPS &middot; peak</div><div class="v" id="c_dec_peak">-</div><div class="u" id="u_dec_peak">window</div></div>
    <div class="card"><div class="k">Requests running / waiting</div><div class="v" id="c_run">-</div><div class="u" id="u_run">live</div></div>
    <div class="card pre"><div class="k">Prefill TPS &middot; now</div><div class="v" id="c_pre">-</div><div class="u">tokens/s</div></div>
    <div class="card pre"><div class="k">Prefill TPS &middot; avg</div><div class="v" id="c_pre_avg">-</div><div class="u" id="u_pre_avg">window</div></div>
    <div class="card pre"><div class="k">Prefill TPS &middot; peak</div><div class="v" id="c_pre_peak">-</div><div class="u" id="u_pre_peak">window</div></div>
    <div class="card"><div class="k">GPU active</div><div class="v" id="c_active">-</div><div class="u">of window</div></div>
  </div>

  <div class="panel">
    <h2>Throughput (tokens/s) <span class="note" id="chart_note"></span></h2>
    <canvas id="chart"></canvas>
    <div class="legend"><span><i style="background:var(--blue)"></i>decode (generation)</span>
    <span><i style="background:var(--amber)"></i>prefill</span></div>
  </div>

  <div class="row">
    <div class="panel">
      <h2>Concurrent streams <span class="note" id="stream_note"></span></h2>
      <div class="streams" id="streams"></div>
    </div>
    <div class="panel">
      <h2>Concurrency over time <span class="note" id="conc_note"></span></h2>
      <canvas id="conc_chart" class="conc"></canvas>
      <div class="legend"><span><i style="background:var(--green)"></i>running</span>
      <span><i style="background:var(--amber)"></i>prefilling</span>
      <span><i style="background:var(--cyan)"></i>waiting</span></div>
    </div>
  </div>

  <div class="row">
    <div class="panel">
      <h2>Requests <span class="note" id="req_note"></span></h2>
      <div class="kv" id="req_kv"></div>
      <div style="margin-top:12px"><h2 style="margin-bottom:6px">Cache reuse</h2><div id="reuse" class="muted">-</div></div>
    </div>
    <div class="panel">
      <h2>Live queue &middot; errors</h2>
      <div class="kv" id="live_kv"></div>
      <div style="margin-top:12px"><h2 style="margin-bottom:6px">Recent errors</h2><div id="errs" class="muted">none</div></div>
    </div>
  </div>

  <div class="panel">
    <h2>Recent requests <span class="note">last 30, newest first</span></h2>
    <div style="overflow-x:auto">
    <table id="req_table">
      <thead><tr><th>time</th><th>req</th><th>finish</th><th>prompt</th><th>gen</th>
      <th>ttft</th><th>dec/s</th><th>wall</th><th>mtp</th></tr></thead>
      <tbody id="req_rows"></tbody>
    </table>
    </div>
  </div>

  <div class="panel">
    <h2>Raw log <span class="note">newest first &middot; live tail</span></h2>
    <div class="log" id="log"></div>
  </div>

  <footer>
    source: <span id="f_src">docker logs</span> &middot; engine API &middot; poll <span id="f_poll">-</span>s
    &middot; window metrics recompute live &middot; NInfer format supported
    (vLLM / SGLang: add a parser in <code>dashboard.py::PARSERS</code> to light up metrics;
    their live logs still show below)
  </footer>
</div>

<script>
const WINS=[[60,"1m"],[300,"5m"],[900,"15m"],[3600,"1h"]];
let win=300;
const $=id=>document.getElementById(id);
function fmt(v,d=1){return v==null?"-":(+v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});}
function fmt0(v){return v==null?"-":Math.round(v).toLocaleString();}
function tstr(t){return t==null?"-":new Date(t*1000).toLocaleTimeString([],{hour12:false});}
function ms(v){if(v==null)return "-";return v>=1000?fmt(v/1000,2)+"s":Math.round(v)+"ms";}

function buildWins(){
  const el=$("wins");el.innerHTML="";
  WINS.forEach(([s,label])=>{
    const b=document.createElement("button");b.textContent=label;b.className=(s===win)?"on":"";
    b.onclick=()=>{win=s;buildWins();fetch_();};el.appendChild(b);
  });
}

function kv(rows){
  return rows.map(([l,n])=>`<div class="i"><div class="l">${l}</div><div class="n">${n}</div></div>`).join("");
}

function drawChart(series,peak,window,now){
  const c=$("chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=210;c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=44,padR=10,padT=10,padB=20;
  const cw=W-padL-padR,ch=H-padT-padB;
  const yv=Math.max(10,peak);
  const t1=now,t0=now-window;
  x.strokeStyle="#21262d";x.fillStyle="#8b949e";x.font="10px monospace";x.lineWidth=1;
  // y grid
  for(let g=0;g<=4;g++){
    const val=yv*g/4;const y=padT+ch-ch*g/4;
    x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();
    x.fillText(Math.round(val),4,y+3);
  }
  // x labels
  for(let g=0;g<=4;g++){
    const tt=t0+window*g/4;const px=padL+cw*g/4;
    x.fillText(tstr(tt).slice(0,5),px-12,H-6);
  }
  if(!series||series.length<2){x.fillStyle="#6e7681";x.fillText("waiting for samples...",padL+10,padT+20);return;}
  const X=t=>padL+cw*((t-t0)/window);
  const Y=v=>padT+ch-ch*Math.min(v,yv)/yv;
  function line(idx,color){
    x.beginPath();x.strokeStyle=color;x.lineWidth=2;x.lineJoin="round";
    let started=false;
    for(const p of series){
      if(p[0]<t0||p[0]>t1)continue;
      const px=X(p[0]),py=Y(p[idx]);
      if(!started){x.moveTo(px,py);started=true;}else x.lineTo(px,py);
    }
    x.stroke();
  }
  line(2,"#d29922");line(1,"#58a6ff");
}

function drawConc(series,window,now){
  const c=$("conc_chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=150;c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=30,padR=8,padT=8,padB=16;
  const cw=W-padL-padR,ch=H-padT-padB;
  const t1=now,t0=now-window;
  x.strokeStyle="#21262d";x.fillStyle="#8b949e";x.font="10px monospace";x.lineWidth=1;
  let yv=4;
  for(const p of series)for(let i=1;i<p.length;i++)yv=Math.max(yv,p[i]);
  yv=Math.max(4,Math.ceil(yv));
  for(let g=0;g<=2;g++){
    const y=padT+ch-ch*g/2;
    x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();
    x.fillText(String(yv*g/2),4,y+3);
  }
  if(!series||series.length<2){x.fillStyle="#6e7681";x.fillText("waiting for samples...",padL+10,padT+16);return;}
  const X=t=>padL+cw*((t-t0)/window);
  const Y=v=>padT+ch-ch*Math.min(v,yv)/yv;
  function line(idx,color){
    x.beginPath();x.strokeStyle=color;x.lineWidth=1.5;x.lineJoin="round";
    let started=false;
    for(const p of series){
      if(p[0]<t0||p[0]>t1)continue;
      const px=X(p[0]),py=Y(p[idx]);
      if(!started){x.moveTo(px,py);started=true;}else x.lineTo(px,py);
    }
    x.stroke();
  }
  line(1,"#3fb950");line(2,"#d29922");line(3,"#39c5cf");
}

function render(s){
  const c=s.container,l=s.live,tp=s.tp,R=s.requests;
  $("dot").className="dot "+(c.running?"up":"err");
  $("chip").innerHTML="<b>"+(c.name||"no container")+"</b>&nbsp;·&nbsp;"+(c.framework||"?")
    + "&nbsp;·&nbsp;"+(c.running?(c.state||"up"):(c.state||"exited"));
  const badge=$("badge");
  if(!c.running){badge.textContent="not running";badge.className="badge";badge.style.color="var(--dim)";}
  else if(!c.has_parser){badge.textContent="parser pending";badge.className="badge pending";}
  else if(l.idle){badge.textContent="idle";badge.className="badge idle";}
  else{badge.textContent="busy";badge.className="badge busy";}
  $("meta").textContent="updated "+Math.max(0,Math.round(s.server.now-s.server.last_poll))+"s ago · auto 2s";

  const wlabel={60:"1m",300:"5m",900:"15m",3600:"1h"}[win]||win+"s";
  $("c_dec").textContent=fmt(l.decode_tps);
  $("c_dec_avg").textContent=fmt(tp.avg_decode);$("u_dec_avg").textContent=wlabel+" · "+(tp.samples||0)+" samples";
  $("c_dec_peak").textContent=fmt(tp.peak_decode);$("u_dec_peak").textContent=wlabel;
  $("c_pre").textContent=fmt(l.prefill_tps);
  $("c_pre_avg").textContent=fmt(tp.avg_prefill);$("u_pre_avg").textContent=wlabel;
  $("c_pre_peak").textContent=fmt(tp.peak_prefill);$("u_pre_peak").textContent=wlabel;
  $("c_run").textContent=(l.running==null?"-":l.running)+" / "+(l.waiting==null?"-":l.waiting);
  $("u_run").textContent="prefilling "+(l.prefilling==null?"-":l.prefilling);
  $("c_active").textContent=tp.active_pct==null?"-":fmt(tp.active_pct,1)+"%";
  $("chart_note").textContent="last "+wlabel+" · peak "+s.peak+" tok/s";

  drawChart(s.series,s.peak,win,s.server.now);

  const streams=s.streams||[];
  const maxAge=streams.length?Math.max(1,streams[0].age_s||0):1;
  $("stream_note").textContent=streams.length
    ?(streams.length+" active · heaviest "+(streams[0].age_s!=null?streams[0].age_s.toFixed(1)+"s":"-"))
    :"none in flight";
  $("streams").innerHTML=streams.length?streams.map(sr=>{
    const a=sr.age_s;
    const col=a==null?"var(--dim)":(a<30?"var(--green)":a<120?"var(--amber)":"var(--red)");
    const cls=a==null?"p0":(a<30?"p0":a<120?"p1":"p2");
    const w=a==null?0:Math.max(2,Math.min(100,(a/maxAge)*100));
    return `<div class="srow">
      <div><span class="id">${esc(sr.req)}</span>
        <span class="tag">${esc(sr.proto||"?")}</span>
        <span class="tag">${sr.stream?"stream":"oneshot"}</span></div>
      <div class="age" style="color:${col}">${a==null?"-":a.toFixed(1)+"s"}</div>
      <div class="meta">${sr.msgs!=null?sr.msgs+" msgs":"?"} &middot; max ${sr.max_tokens==null?"?":sr.max_tokens} tok</div>
      <div></div>
      <div class="bar"><i class="${cls}" style="width:${w}%"></i></div>
    </div>`;
  }).join(""):'<div class="empty">no active streams — waiting for requests&hellip;</div>';
  drawConc(s.conc_series,win,s.server.now);
  $("conc_note").textContent="last "+wlabel+" · now running "+(l.running==null?"-":l.running)+" / waiting "+(l.waiting==null?"-":l.waiting);

  $("req_note").textContent=wlabel;
  $("req_kv").innerHTML=kv([
    ["done",fmt0(R.done)],["submitted",fmt0(R.submitted)],["errors",fmt0(R.errors)],
    ["total gen",R.total_gen==null?"-":R.total_gen.toLocaleString()+" tok"],
    ["total prompt",R.total_prompt==null?"-":R.total_prompt.toLocaleString()+" tok"],
    ["output",R.agg_tok_s==null?"-":fmt(R.agg_tok_s)+" tok/s"],
    ["ttft avg",ms(R.avg_ttft_ms)],["ttft p50",ms(R.p50_ttft_ms)],["ttft p95",ms(R.p95_ttft_ms)],
    ["wall avg",R.avg_wall_s==null?"-":fmt(R.avg_wall_s,1)+"s"],["wall max",R.max_wall_s==null?"-":fmt(R.max_wall_s,1)+"s"],
    ["mtp accept",R.avg_mtp_pct==null?"-":fmt(R.avg_mtp_pct,1)+"%"],
  ]);
  const reuse=R.reuse||{};
  const rk=Object.keys(reuse);
  $("reuse").innerHTML=rk.length?rk.map(k=>`<span class="tag">${k} <b>${reuse[k]}</b></span>`).join(""):'<span class="muted">-</span>';

  $("live_kv").innerHTML=kv([
    ["running",l.running==null?"-":l.running],["waiting",l.waiting==null?"-":l.waiting],
    ["decode_ready",l.decode_ready==null?"-":l.decode_ready],
    ["avg batch",l.avg_decode_batch==null?"-":fmt(l.avg_decode_batch,2)],
  ]);
  $("errs").innerHTML=(s.recent_errors&&s.recent_errors.length)?
    s.recent_errors.slice(0,6).map(e=>`<div class="err" style="font-family:var(--mono);font-size:12px;margin:3px 0">[${tstr(e.t)}] <span class="muted">req ${e.req}</span> ${esc(e.msg)}</div>`).join("")
    :'<span class="muted">none in window</span>';

  $("req_rows").innerHTML=(s.recent_requests&&s.recent_requests.length)?
    s.recent_requests.map(r=>`<tr><td>${tstr(r.t)}</td><td>${r.req}</td><td>${esc(r.finish||"")}</td>
    <td>${fmt0(r.prompt)}</td><td>${fmt0(r.gen)}</td><td>${ms(r.ttft_ms)}</td>
    <td>${fmt(r.decode)}</td><td>${r.wall==null?"-":fmt(r.wall,1)+"s"}</td><td>${r.mtp_pct==null?"-":fmt(r.mtp_pct,1)+"%"}</td></tr>`).join("")
    :`<tr><td colspan="9" style="text-align:left" class="muted">none in window</td></tr>`;

  $("log").innerHTML=(s.log&&s.log.length)?
    s.log.slice().reverse().map(ln=>{
      const cls=/error/i.test(ln)?"e":(/throughput/.test(ln)?"t":"");
      return `<span class="${cls}">${esc(ln)}</span>`;
    }).join("\n")
    :'<span class="muted">no lines yet</span>';
}
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

async function fetch_(){
  try{
    const r=await fetch("/api/state?window="+win);
    const s=await r.json();
    render(s);
  }catch(e){
    $("dot").className="dot err";
    $("meta").textContent="unreachable: "+e;
  }
}
buildWins();fetch_();setInterval(fetch_,2000);
</script>
</body>
</html>
"""

POLL = 1.5


def main():
    global POLL
    ap = argparse.ArgumentParser(description="Live LLM serving dashboard")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--poll", type=float, default=1.5)
    ap.add_argument("--tail", type=int, default=200)
    args = ap.parse_args()
    POLL = max(0.5, args.poll)

    st = Store()
    Handler.store = st
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)

    t = threading.Thread(target=poll_loop, args=(st, args.tail), daemon=True)
    t.start()
    print(f"dashboard on http://{args.bind}:{args.port}/ (poll {args.poll}s, tail {args.tail})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()