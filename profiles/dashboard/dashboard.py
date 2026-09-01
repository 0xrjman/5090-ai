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
import hashlib
import json
import os
import re
import sqlite3
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
# Fallback device KV pool (tokens) for the estimated KV-occupancy gauge, used only
# when the engine's boot log can't be read; the live value is captured per container
# boot from `kv_capacity_tokens=` into st.kv_total (see kv_capacity_tokens()).
KV_TOTAL_TOKENS = 410944      # = 6421 pages x 64 tok/page (nvfp4 default)

MAX_TP = 6000           # ~5h of 5s throughput samples
MAX_REQ = 2000
MAX_ERR = 500
MAX_RAW = 400           # recent raw lines kept for the live-log panel
MAX_SEEN = 1600         # LRU bound for log-line dedupe
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db")

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\]")
VLLM_TS_RE = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})\b")
KV_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
REQ_RE = re.compile(r"request id=(\d+)")
MTP_RE = re.compile(r"(\d+(?:\.\d+)?)tok/round \((\d+(?:\.\d+)?)%\)")
KV_CAP_RE = re.compile(r"kv_capacity_tokens=(\d+)")


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


def _after(line, label):
    """First token after `label` (vLLM logs use `label: value,` pairs)."""
    i = line.find(label)
    if i < 0:
        return None
    return line[i + len(label):].split(",")[0].split()[0] if line[i + len(label):].split(",")[0].split() else None


def _line_ts(line):
    m = TS_RE.match(line)
    if m:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.datetime.strptime(m.group(1), fmt).timestamp()
            except ValueError:
                continue
        return None
    # vLLM: "09-01 07:49:13" (no year, container TZ may differ from host). Only
    # relative deltas are reliable; restamp() anchors the newest line to wall
    # clock so the year/TZ guess cancels out in the back-fill.
    m = VLLM_TS_RE.search(line)
    if not m:
        return None
    mo, da = int(m.group(1)), int(m.group(2))
    hh, mm, ss = (int(x) for x in m.group(3).split(":"))
    try:
        return datetime.datetime(datetime.datetime.now().year, mo, da, hh, mm, ss).timestamp()
    except ValueError:
        return None


def parse_ninfer(line):
    """Return an event dict for a metric-bearing NInfer line, else None."""
    t = _line_ts(line)
    if "[ninfer-serve] throughput" in line and "interval_ms=" in line:
        kv = _kv(line)
        return {"kind": "tp", "t": t,
                "prefill": _f(kv.get("prefill_tokens_per_second")),
                "decode": _f(kv.get("decode_tokens_per_second")),
                "running": _i(kv.get("running")), "prefilling": _i(kv.get("prefilling")),
                "decode_ready": _i(kv.get("decode_ready")), "waiting": _i(kv.get("waiting")),
                "avg_decode_batch": _f(kv.get("average_decode_batch"))}
    req = REQ_RE.search(line)
    if req:
        rid = req.group(1)
        if "status=done" in line:
            kv = _kv(line)
            m = MTP_RE.search(line)
            _dur = _f(kv.get("duration_ms"))
            return {"kind": "done", "t": t, "req": rid,
                    "finish": kv.get("finish_reason"), "tool_calls": _i(kv.get("tool_calls")),
                    "prompt": _i(kv.get("prompt_tokens")), "gen": _i(kv.get("completion_tokens")),
                    "cache": _i(kv.get("prefix_cache_hit_tokens")),
                    "reuse": kv.get("prefix_reuse_path"),
                    "ttft_ms": _f(kv.get("ttft_ms")),
                    "prefill": _f(kv.get("prefill_tokens_per_second")),
                    "decode": _f(kv.get("decode_tokens_per_second")),
                    "wall": (_dur / 1000.0) if _dur is not None else None,
                    "mtp_round": float(m.group(1)) if m else None,
                    "mtp_pct": float(m.group(2)) if m else None}
        if "status=error" in line:
            msg = line.split("message=", 1)[1].strip() if "message=" in line else ""
            return {"kind": "err", "t": t, "req": rid, "msg": msg}
        if "status=rejected" in line:
            kv = _kv(line)
            msg = line.split("message=", 1)[1].strip() if "message=" in line else ""
            return {"kind": "rej", "t": t, "req": rid, "status": _i(kv.get("status")),
                    "code": kv.get("code"), "msg": msg}
        if "status=submitted" in line:
            kv = _kv(line)
            proto = ("anthropic" if "anthropic_messages" in line
                     else "openai" if "openai_chat_completions" in line else "?")
            return {"kind": "sub", "t": t, "req": rid, "proto": proto,
                    "stream": kv.get("stream") == "true",
                    "msgs": _i(kv.get("messages")), "max_tokens": _i(kv.get("requested_output_tokens"))}
        if "status=cancelled" in line:
            return {"kind": "cancel", "t": t, "req": rid}
    return None


def parse_vllm(line):
    """Return an event dict for a metric-bearing vLLM line, else None.

    vLLM's default logs carry no per-request lines, so only the periodic
    throughput line (loggers.py) and spec-decoding line (metrics.py) are parsed.
    Timestamps are unparseable here (MM-DD, no year / wrong TZ) -> t=None and the
    poll-time fallback in restamp()/ingest() anchors them.
    """
    t = _line_ts(line)
    if "Avg prompt throughput" in line:
        return {"kind": "tp", "t": t,
                "prefill": _f(_after(line, "Avg prompt throughput:")),
                "decode": _f(_after(line, "Avg generation throughput:")),
                "running": _i(_after(line, "Running:")),
                "waiting": _i(_after(line, "Waiting:"))}
    if "SpecDecoding metrics" in line:
        return {"kind": "sd", "t": t,
                "acc_len": _f(_after(line, "Mean acceptance length:")),
                "acc_rate": _f(_after(line, "Avg Draft acceptance rate:")),
                "acc_thr": _f(_after(line, "Accepted throughput:")),
                "draft_thr": _f(_after(line, "Drafted throughput:"))}
    return None


# framework -> line parser. Add a parser here to light up metrics for a framework.
PARSERS = {"ninfer": parse_ninfer, "vllm": parse_vllm}

# Frameworks whose logs carry per-request lifecycle lines (submitted/done),
# enabling the per-stream "Concurrent streams" panel. vLLM's default logs do
# not, so that panel is N/A for it (aggregate running/waiting still shown).
PER_REQUEST = {"ninfer"}


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


def gpu_stat():
    """Real GPU utilization/memory/power via nvidia-smi (single-GPU box). Returns a
    dict or None if nvidia-smi is unavailable. ~20ms, called every poll."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu,fan.speed",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, timeout=3).stdout.strip()
        if not out:
            return None
        u, mu, mt, pw, pl, tc, fan = (x.strip() for x in out.split(",")[:7])
        return {"util": _i(u), "mem_used": _i(mu), "mem_total": _i(mt),
                "power_w": _f(pw), "power_limit_w": _f(pl),
                "temp_c": _f(tc), "fan_pct": _i(fan), "ts": time.time()}
    except Exception:
        return None


def cpu_temp():
    """CPU package temp (°C) from the x86_pkg_temp thermal zone; None if absent."""
    for i in range(32):
        try:
            z = f"/sys/class/thermal/thermal_zone{i}"
            if open(z + "/type").read().strip() == "x86_pkg_temp":
                return round(int(open(z + "/temp").read().strip()) / 1000, 1)
        except OSError:
            pass
    return None


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


def kv_capacity_tokens(name):
    m = KV_CAP_RE.findall(docker("logs", name))
    return int(m[-1]) if m else None


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
        self.sd = deque(maxlen=MAX_TP)
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
        self.gpu = deque(maxlen=120)   # (ts, util_pct) recent samples for a trend sparkline
        self.gpu_now = None            # latest full nvidia-smi snapshot
        self.therm = deque(maxlen=2400)   # (ts, gpu_c, cpu_c, gpu_w, fan_pct) ~1h at 1.5s
        self.therm_db_ts = 0.0
        self.therm_prune_ts = 0.0
        self.kv_total = None      # live KV pool (tokens) from the engine's boot log
        self.kv_started = None    # container StartedAt that kv_total was captured for
        self.boot = time.time()
        self.lock = threading.Lock()

    def reset_target(self, name):
        self.tp.clear()
        self.sd.clear()
        self.req.clear()
        self.sub.clear()
        self.err.clear()
        self.raw.clear()
        self.inflight.clear()
        self._seen.clear()
        self._seen_q.clear()
        self.last_line_ts = 0
        self.kv_total = None
        self.kv_started = None
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
        elif k == "sd":
            self.sd.append(ev)
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
        elif k == "cancel":
            if rid is not None:
                self.inflight.pop(rid, None)
        if ev.get("t"):
            self.last_line_ts = max(self.last_line_ts, ev["t"])


def init_db(path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS throughput(
      fp TEXT, ts REAL, container TEXT, framework TEXT,
      prefill REAL, decode REAL,
      running INTEGER, prefilling INTEGER, decode_ready INTEGER, waiting INTEGER,
      avg_decode_batch REAL);
    CREATE TABLE IF NOT EXISTS requests(
      fp TEXT, ts REAL, container TEXT, framework TEXT, req TEXT, kind TEXT,
      proto TEXT, stream INTEGER, finish TEXT, status INTEGER, code TEXT, msg TEXT,
      prompt INTEGER, gen INTEGER, cache INTEGER, reuse TEXT,
      ttft_ms REAL, prefill REAL, decode REAL, wall REAL,
      mtp_round REAL, mtp_pct REAL);
    CREATE INDEX IF NOT EXISTS idx_tp_ts ON throughput(ts);
    CREATE INDEX IF NOT EXISTS idx_req_ts ON requests(ts);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_tp_fp ON throughput(fp);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_req_fp ON requests(fp);
    CREATE TABLE IF NOT EXISTS thermal(
      ts REAL PRIMARY KEY, gpu_c REAL, cpu_c REAL, gpu_w REAL, gpu_fan INTEGER);
    """)
    for t in ("throughput", "requests"):
        if "fp" not in [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]:
            conn.execute(f"ALTER TABLE {t} ADD COLUMN fp TEXT")
    conn.commit()
    return conn


def _fp(ln):
    """Stable content identity for a raw log line -> idempotent inserts across
    restarts (the 200-line tail re-read on each start would otherwise re-log the
    same samples/requests, which restamp() would re-time as fresh)."""
    return hashlib.sha1(ln.encode("utf-8", "replace")).hexdigest()[:16]


def record(conn, container, framework, evs):
    """Persist this poll's (line, event) pairs. Called only from the poll thread;
    batched so a poll writes at most two executemany + one commit (negligible vs
    the docker call). INSERT OR IGNORE on the line fingerprint keeps restart
    tail-backfills from double-counting."""
    tps = [(ln, e) for ln, e in evs if e["kind"] == "tp"]
    rqs = [(ln, e) for ln, e in evs if e["kind"] in ("sub", "done", "err", "rej")]
    try:
        if tps:
            conn.executemany(
                "INSERT OR IGNORE INTO throughput(fp,ts,container,framework,prefill,decode,running,prefilling,decode_ready,waiting,avg_decode_batch) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(_fp(ln), e["t"], container, framework, e.get("prefill"), e.get("decode"),
                  e.get("running"), e.get("prefilling"), e.get("decode_ready"),
                  e.get("waiting"), e.get("avg_decode_batch")) for ln, e in tps])
        if rqs:
            cols = "fp,ts,container,framework,req,kind,proto,stream,finish,status,code,msg,prompt,gen,cache,reuse,ttft_ms,prefill,decode,wall,mtp_round,mtp_pct"
            ph = ",".join("?" for _ in cols.split(","))
            conn.executemany(
                f"INSERT OR IGNORE INTO requests({cols}) VALUES({ph})",
                [(_fp(ln), e["t"], container, framework, e.get("req"), e["kind"], e.get("proto"),
                  (1 if e.get("stream") else 0) if e.get("stream") is not None else None,
                  e.get("finish"), e.get("status"), e.get("code"), e.get("msg"),
                  e.get("prompt"), e.get("gen"), e.get("cache"), e.get("reuse"),
                  e.get("ttft_ms"), e.get("prefill"), e.get("decode"), e.get("wall"),
                  e.get("mtp_round"), e.get("mtp_pct")) for ln, e in rqs])
        conn.commit()
    except sqlite3.Error:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass


RETENTION = 86400  # seconds of throughput/requests history to keep


def poll_loop(st, tail, conn):
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
                    started = docker("inspect", "--format", "{{.State.StartedAt}}", name).strip()
                    if started and started != st.kv_started:
                        st.kv_total = kv_capacity_tokens(name)
                        st.kv_started = started
                    fresh = pull_lines(st, name, tail)
                    parser = PARSERS.get(st.framework)
                    if parser is not None and fresh:
                        evs = [(ln, ev) for ln in fresh if (ev := parser(ln)) is not None]
                        if evs:
                            restamp([ev for _, ev in evs], now)
                            for _, ev in evs:
                                st.ingest(ev, now)
                            record(conn, name, st.framework, evs)
        g = gpu_stat()
        ct = cpu_temp()
        therm = None
        prune_therm = None
        if g is not None:
            with st.lock:
                st.gpu_now = g
                st.gpu.append((now, g["util"]))
                if g.get("temp_c") is not None or ct is not None:
                    st.therm.append((now, g.get("temp_c"), ct, g.get("power_w"), g.get("fan_pct")))
                    if now - st.therm_db_ts >= 5:
                        st.therm_db_ts = now
                        therm = (now, g.get("temp_c"), ct, g.get("power_w"), g.get("fan_pct"))
                if now - st.therm_prune_ts >= 300:
                    st.therm_prune_ts = now
                    prune_therm = now - 3600
        if therm is not None:
            try:
                conn.execute("INSERT OR REPLACE INTO thermal(ts,gpu_c,cpu_c,gpu_w,gpu_fan) VALUES(?,?,?,?,?)", therm)
                conn.commit()
            except sqlite3.Error:
                pass
        if prune_therm is not None:
            try:
                cutoff = now - RETENTION
                conn.execute("DELETE FROM thermal WHERE ts < ?", (prune_therm,))
                conn.execute("DELETE FROM throughput WHERE ts < ?", (cutoff,))
                conn.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))
                conn.commit()
            except sqlite3.Error:
                pass
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
            "avg_decode": _mean([v for v in dec if (v or 0) > 0]),
            "peak_decode": round(max(dec), 1) if dec else None,
            "avg_prefill": _mean([v for v in pre if (v or 0) > 0]),
            "peak_prefill": round(max(pre), 1) if pre else None,
            "active_pct": round(100.0 * len(active) / len(tp_win), 1) if tp_win else None,
            "samples": len(tp_win),
        }

        if latest:
            live = {
                "decode_tps": latest["decode"], "prefill_tps": latest["prefill"],
                "running": latest.get("running"), "prefilling": latest.get("prefilling"),
                "decode_ready": latest.get("decode_ready"), "waiting": latest.get("waiting"),
                "avg_decode_batch": latest.get("avg_decode_batch"),
                "idle": ((latest.get("decode") or 0) == 0 and (latest.get("prefill") or 0) == 0
                         and (latest.get("running") or 0) == 0),
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
                       "decode": r["decode"], "wall": r["wall"], "mtp_pct": r["mtp_pct"],
                       "cache": r.get("cache"), "reuse": r.get("reuse")}
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
            [(s["t"], s.get("running") or 0, s.get("prefilling") or 0, s.get("waiting") or 0)
             for s in tp_win])

        # Load analysis: context-length distribution, context×decode scatter, and an
        # ESTIMATED KV-occupancy series. The KV total (denominator) is the engine's
        # resolved device pool; the numerator is an estimate (the log exposes live
        # concurrency but not live per-stream context), so it is labeled as such.
        ap = _mean(prompt) if done else None
        _buckets = [(0, 30000, "0-30K"), (30000, 60000, "30-60K"), (60000, 100000, "60-100K"),
                    (100000, 150000, "100-150K"), (150000, 10**12, "160K+")]
        ctx_hist = []
        for _lo, _hi, _lab in _buckets:
            _b = [r for r in done if r["prompt"] is not None and _lo <= r["prompt"] < _hi]
            if _b or done:
                ctx_hist.append({"label": _lab, "n": len(_b),
                                 "avg_ttft_ms": round(_mean([r["ttft_ms"] for r in _b]), 0) if _b else None,
                                 "avg_wall_s": round(_mean([r["wall"] for r in _b]), 1) if _b else None,
                                 "avg_decode": _mean([r["decode"] for r in _b]) if _b else None})
        scatter = [{"x": r["prompt"], "y": r["decode"], "ttft": r["ttft_ms"]}
                   for r in done if r["prompt"] is not None and r["decode"] is not None][-80:]
        _ap = ap or 0
        kv_total = st.kv_total or KV_TOTAL_TOKENS
        kv_series = _downsample_series(
            [(s["t"], (s.get("running") or 0) + (s.get("prefilling") or 0),
              round(((s.get("running") or 0) + (s.get("prefilling") or 0)) * _ap),
              round(min(100.0, 100.0 * ((s.get("running") or 0) + (s.get("prefilling") or 0)) * _ap
                        / kv_total), 1) if _ap else None)
             for s in tp_win])
        if latest and _ap:
            _inf = (latest.get("running") or 0) + (latest.get("prefilling") or 0)
            kv_now = {"in_flight": _inf, "tokens": round(_inf * _ap),
                      "pct": round(min(100.0, 100.0 * _inf * _ap / kv_total), 1)}
        else:
            kv_now = {"in_flight": None, "tokens": None, "pct": None}
        analysis = {"kv_total": kv_total, "avg_prompt": ap,
                    "ctx_hist": ctx_hist, "scatter": scatter,
                    "kv_series": kv_series, "kv_now": kv_now}

        sd_win = [s for s in st.sd if (s["t"] or 0) >= c0]
        specdec = ({"acc_len": _mean([s["acc_len"] for s in sd_win]),
                    "acc_rate": _mean([s["acc_rate"] for s in sd_win]),
                    "acc_thr": _mean([s["acc_thr"] for s in sd_win]),
                    "draft_thr": _mean([s["draft_thr"] for s in sd_win]),
                    "samples": len(sd_win)} if sd_win else {})

        target = st.target
        framework = st.framework
        running = st.running
        state_str = st.state_str
        image = st.image
        last_seen = st.last_line_ts
        docker_ok = st.docker_ok
        boot = st.boot
        gpu_now = st.gpu_now
        gpu_series = [[round(t, 3), u] for t, u in st.gpu][-160:]

    dec_vals = [p[1] for p in series]
    pre_vals = [p[2] for p in series]
    peak = max([max(dec_vals) if dec_vals else 0, max(pre_vals) if pre_vals else 0], default=0)

    therm_win = [p for p in st.therm if p[0] >= now - window]
    t_now = st.therm[-1] if st.therm else None
    thermal = {
        "now": ({"gpu_c": t_now[1], "cpu_c": t_now[2], "gpu_w": t_now[3],
                 "fan_pct": t_now[4]} if t_now else {}),
        "series": _downsample(therm_win),
        "max_gpu": round(max((p[1] for p in therm_win if p[1] is not None), default=0) or None, 1),
        "max_cpu": round(max((p[2] for p in therm_win if p[2] is not None), default=0) or None, 1),
    }

    return {
        "server": {"now": round(now, 3), "last_poll": round(st.last_poll, 3),
                   "boot": round(boot, 3), "poll": POLL, "docker_ok": docker_ok},
        "container": {"name": target, "framework": framework, "running": running,
                      "state": state_str, "image": image, "api_port": SERVE_API_PORT,
                      "last_seen": round(last_seen, 3),
                      "has_parser": framework in PARSERS,
                      "per_request": framework in PER_REQUEST},
        "gpu": {"now": gpu_now, "series": gpu_series},
        "thermal": thermal,
        "window": window,
        "live": live,
        "tp": tp_metrics,
        "peak": round(peak, 1) or 100,
        "series": series,
        "streams": streams,
        "stream_counts": {"active": len(streams),
                          "max_age_s": streams[0]["age_s"] if streams else None},
        "conc_series": conc_series,
        "analysis": analysis,
        "specdec": specdec,
        "requests": {**req_counts, **req_agg},
        "recent_requests": recent_req,
        "recent_errors": recent_err,
        "log": raw,
    }


class Handler(BaseHTTPRequestHandler):
    store = None
    conn = None
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
        elif u.path == "/api/history":
            try:
                limit = max(1, min(5000, int(q.get("limit", [200])[0])))
            except (ValueError, TypeError):
                limit = 200
            conn = self.conn
            rows = []
            if conn is not None:
                try:
                    rows = conn.execute(
                        "SELECT ts,container,req,kind,finish,prompt,gen,ttft_ms,wall,mtp_pct "
                        "FROM requests ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
                except sqlite3.Error:
                    rows = []
            out = [{"ts": r[0], "container": r[1], "req": r[2], "kind": r[3], "finish": r[4],
                    "prompt": r[5], "gen": r[6], "ttft_ms": r[7], "wall": r[8], "mtp_pct": r[9]}
                   for r in rows]
            self._send(200, json.dumps({"count": len(out), "requests": out}), "application/json")
        else:
            self._send(404, "not found", "text/plain")


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>local-ai serving</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#08080f;
  --panel:#0e1019;
  --panel-2:#14172a;
  --panel-3:#1c2038;
  --border:rgba(150,160,190,.10);
  --border-2:rgba(150,160,190,.20);
  --txt:#e8ebf4;
  --muted:#9aa3b8;
  --dim:#5c6478;
  --accent:#7c7bff;
  --accent-2:#a46bff;
  --grad:linear-gradient(135deg,#7c7bff 0%,#a46bff 100%);
  --dec:#5b93ff;
  --pre:#f5a524;
  --run:#2fd6a0;
  --wait:#37c6f0;
  --err:#ff6b7a;
  --ok:#37d39f;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%}
body{background:
  radial-gradient(1100px 560px at 12% -8%,rgba(124,123,255,.10),transparent 60%),
  radial-gradient(900px 480px at 105% -5%,rgba(164,107,255,.08),transparent 55%),
  var(--bg);
  color:var(--txt);font-family:var(--sans);font-size:14px;-webkit-font-smoothing:antialiased;line-height:1.45}
a{color:var(--accent);text-decoration:none}
code{font-family:var(--mono);font-size:.92em}
.wrap{max-width:1280px;margin:0 auto;padding:20px 20px 64px}
.top{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:16px 18px;
  background:linear-gradient(180deg,var(--panel-2),var(--panel));
  border:1px solid var(--border);border-radius:16px;position:relative;overflow:hidden}
.top::before{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:var(--grad);opacity:.9}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:36px;height:36px;border-radius:11px;background:var(--grad);display:grid;place-items:center;color:#fff;
  box-shadow:0 6px 18px rgba(124,123,255,.42),inset 0 1px 0 rgba(255,255,255,.25)}
.logo svg{width:20px;height:20px}
.brand h1{font-size:15.5px;font-weight:700;margin:0;letter-spacing:-.01em}
.brand .sub{font-size:10.5px;color:var(--muted);margin-top:2px;font-family:var(--mono);letter-spacing:.02em}
.dot{width:9px;height:9px;border-radius:50%;background:var(--dim);position:relative;flex:0 0 auto}
.dot.up{background:var(--ok)}
.dot.up::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:1.5px solid var(--ok);animation:pulse 2.2s ease-out infinite}
.dot.err{background:var(--err)}
@keyframes pulse{0%{transform:scale(.55);opacity:.75}70%{transform:scale(1.7);opacity:0}100%{opacity:0}}
.chip{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.03);
  border:1px solid var(--border);border-radius:999px;padding:6px 13px;font-size:12.5px;color:var(--muted)}
.chip b{font-weight:600;color:var(--txt);font-family:var(--mono)}
.spacer{flex:1}
.wins{display:flex;gap:2px;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:3px}
.wins button{background:transparent;border:none;color:var(--muted);border-radius:7px;padding:5px 12px;font-size:12px;
  cursor:pointer;font-family:var(--mono);font-weight:500;transition:color .15s,background .15s}
.wins button:hover{color:var(--txt)}
.wins button.on{background:var(--grad);color:#fff;box-shadow:0 2px 10px rgba(124,123,255,.4)}
.meta{font-size:11.5px;color:var(--dim);font-family:var(--mono)}
.therm{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11.5px;
  padding:5px 12px;border-radius:999px;border:1px solid var(--border-2);background:var(--panel);
  color:var(--muted);white-space:nowrap}
.therm .tdot{width:7px;height:7px;border-radius:50%;background:var(--run);flex:none}
.therm.warn{color:#f5c064;border-color:rgba(245,165,36,.4)}
.therm.warn .tdot{background:var(--pre)}
.therm.hot{color:#ff97a2;border-color:rgba(255,107,122,.5);animation:thermpulse 2s ease-in-out infinite}
.therm.hot .tdot{background:var(--err)}
@keyframes thermpulse{50%{box-shadow:0 0 14px rgba(255,107,122,.45)}}
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:8px;padding:5px 11px;font-size:11.5px;
  font-family:var(--mono);font-weight:500;border:1px solid var(--border);background:var(--panel-2);color:var(--muted);
  text-transform:uppercase;letter-spacing:.03em}
.badge .bdot{width:6px;height:6px;border-radius:50%;background:currentColor}
.badge.busy{color:var(--run);border-color:rgba(47,214,160,.32);background:rgba(47,214,160,.09)}
.badge.idle{color:var(--muted);background:rgba(255,255,255,.02)}
.badge.pending{color:var(--pre);border-color:rgba(245,165,36,.32);background:rgba(245,165,36,.09)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:0}
.card{background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--border);border-radius:14px;
  padding:15px 16px;position:relative;overflow:hidden;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}
.card:hover{transform:translateY(-3px);border-color:var(--border-2);box-shadow:0 14px 34px rgba(0,0,0,.45)}
.card .ct{display:flex;align-items:center;gap:9px}
.card .ct .ic{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:rgba(255,255,255,.04);
  border:1px solid var(--border);color:var(--muted);flex:0 0 auto}
.card .ct .ic svg{width:16px;height:16px}
.card .ct .k{font-size:12px;font-weight:600;color:var(--muted)}
.card .ct .ks{margin-left:auto;font-size:10px;color:var(--dim);font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em}
.card.dec .ct .ic{color:var(--dec);background:rgba(91,147,255,.12);border-color:rgba(91,147,255,.24)}
.card.pre .ct .ic{color:var(--pre);background:rgba(245,165,36,.12);border-color:rgba(245,165,36,.24)}
.card .cv{font-size:28px;font-weight:700;margin-top:12px;font-family:var(--mono);line-height:1;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.card.dec .cv{color:var(--dec);text-shadow:0 0 22px rgba(91,147,255,.35)}
.card.pre .cv{color:var(--pre);text-shadow:0 0 22px rgba(245,165,36,.30)}
.card .cu{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--dim);font-family:var(--mono);margin-top:7px}
.card .cu canvas{margin-left:auto;width:66px;height:22px}
.panel{background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--border);
  border-radius:16px;padding:16px 18px;margin-top:14px}
.panel .ph{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.panel .ph .ic{width:27px;height:27px;border-radius:8px;display:grid;place-items:center;
  background:rgba(124,123,255,.12);border:1px solid rgba(124,123,255,.24);color:var(--accent);flex:0 0 auto}
.panel .ph .ic svg{width:15px;height:15px}
.panel .ph h2{font-size:13px;font-weight:600;margin:0;color:var(--txt)}
.panel .ph .note{margin-left:auto;color:var(--dim);font-size:11.5px;font-family:var(--mono);font-weight:400}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:900px){.row{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}.wrap{padding:14px}}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:15px 16px}
.kv .i .l{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.kv .i .n{font-size:18px;font-family:var(--mono);font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.subhead{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;font-weight:600;margin:18px 0 8px}
.subhead svg{width:13px;height:13px;color:var(--dim)}
.tag{display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:7px;padding:2px 9px;font-family:var(--mono);font-size:11.5px;margin:3px 4px 3px 0;color:var(--muted)}
.tag b{color:var(--txt);font-weight:600}
.log{background:#070810;border:1px solid var(--border);border-radius:12px;padding:12px 14px;font-family:var(--mono);
  font-size:11.8px;line-height:1.6;max-height:340px;overflow:auto;white-space:pre-wrap;word-break:break-word;
  box-shadow:inset 0 3px 14px rgba(0,0,0,.45)}
.log .e{color:var(--err)}
.log .t{color:var(--dim)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
th{color:var(--dim);text-align:right;font-weight:600;padding:9px 10px;border-bottom:1px solid var(--border);
  font-size:10.5px;text-transform:uppercase;letter-spacing:.04em}
td{padding:8px 10px;text-align:right;border-bottom:1px solid rgba(255,255,255,.045);color:var(--muted)}
th:first-child,td:first-child{text-align:left}
td:first-child{color:var(--dim)}
tbody tr{transition:background .12s}
tbody tr:hover td{background:rgba(124,123,255,.055)}
.f-finish{font-weight:600}
.err{color:var(--err)}
.muted{color:var(--muted)}
canvas{width:100%;height:220px;display:block}
canvas.conc{height:150px}
.legend{display:flex;gap:18px;font-size:12px;color:var(--muted);margin-top:10px;font-family:var(--mono);flex-wrap:wrap}
.legend i{display:inline-block;width:16px;height:3px;border-radius:2px;margin-right:7px;vertical-align:middle}
.streams{display:flex;flex-direction:column;gap:10px;min-height:60px}
.srow{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:12px;padding:11px 13px;
  display:grid;grid-template-columns:1fr auto;gap:6px 10px;align-items:center;transition:border-color .15s}
.srow:hover{border-color:var(--border-2)}
.srow .id{font-family:var(--mono);font-weight:650;font-size:13px;color:var(--txt)}
.srow .meta{font-size:11.5px;color:var(--muted);font-family:var(--mono)}
.srow .age{font-family:var(--mono);font-size:20px;font-weight:700;text-align:right;line-height:1;font-variant-numeric:tabular-nums}
.srow .bar{grid-column:1/-1;height:6px;background:#070810;border-radius:4px;overflow:hidden}
.srow .bar i{display:block;height:100%;border-radius:4px;transition:width .45s ease}
.sgrp{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:12px;padding:9px 12px;transition:border-color .15s}
.sgrp:hover{border-color:var(--border-2)}
.ghead{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:7px}
.ghead b{color:var(--txt);font-weight:650;font-size:12.5px}
.gcnt{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--txt);background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:20px;padding:0 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.reqchip{font-family:var(--mono);font-size:11px;font-variant-numeric:tabular-nums;background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:6px;padding:2px 7px}
.reqchip.stale{opacity:.5;text-decoration:line-through}
.reqchip .rid{opacity:.6}
.reqchip .rage{margin-left:6px;padding-left:6px;border-left:1px solid var(--border);font-weight:600}
.cbadge{font-family:var(--mono);font-size:10.5px;font-weight:600;border-radius:6px;padding:1px 7px;white-space:nowrap}
.cbadge.warm{color:var(--ok);background:rgba(55,211,159,.1);border:1px solid rgba(55,211,159,.3)}
.cbadge.cold{color:var(--muted);background:rgba(255,255,255,.03);border:1px solid var(--border)}
.p0{background:var(--run)}.p1{background:var(--pre)}.p2{background:var(--err)}
.sdone{display:flex;flex-direction:column;gap:6px;margin-top:2px}
.sdone .d{display:grid;grid-template-columns:auto 1fr auto;gap:2px 12px;align-items:baseline;
  font-family:var(--mono);font-size:11.8px;padding:7px 11px;background:rgba(255,255,255,.02);
  border:1px solid var(--border);border-radius:9px;transition:border-color .15s}
.sdone .d:hover{border-color:var(--border-2)}
.sdone .d .rid{color:var(--txt);font-weight:650}
.sdone .d .m{color:var(--muted)}
.sdone .d .wall{font-weight:700;font-variant-numeric:tabular-nums}
.loadrow{display:grid;grid-template-columns:230px 1fr 1fr;gap:20px;align-items:stretch}
@media(max-width:1000px){.loadrow{grid-template-columns:1fr}}
.loadcol{display:flex;flex-direction:column;gap:16px}
.loadchart{display:flex;flex-direction:column}
.loadcol .kvbox{flex:1}
.spark{display:block;width:100%;height:30px;margin-top:9px}
.kvbox{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:12px;padding:14px 15px}
.kvl{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);font-weight:600}
.kvtag{margin-left:auto;font-size:9.5px;font-family:var(--mono);color:var(--dim);text-transform:uppercase;
  letter-spacing:.05em;border:1px solid var(--border);border-radius:5px;padding:1px 6px}
.kvbig{margin-top:10px;font-family:var(--mono);display:flex;align-items:baseline;gap:4px}
.kvbig b{font-size:40px;font-weight:800;letter-spacing:-.02em;color:var(--accent);line-height:1;
  font-variant-numeric:tabular-nums}
.kvbig span{font-size:13px;color:var(--dim)}
.kvbar{height:9px;background:#070810;border-radius:5px;overflow:hidden;margin-top:10px}
.kvbar i{display:block;height:100%;border-radius:5px;background:var(--grad);transition:width .5s ease}
.kvmeta{font-size:11px;color:var(--dim);font-family:var(--mono);margin-top:9px;line-height:1.5}
.kvmini{display:flex;justify-content:space-between;align-items:baseline;margin-top:9px;padding-top:9px;
  border-top:1px solid var(--border)}
.kvmini .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.kvmini .n{font-family:var(--mono);font-weight:600;font-size:13px}
.subhead2{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600;margin-bottom:8px}
.subhead2 .dim{color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400}
canvas.hist,canvas.scatter{height:auto;min-height:170px;flex:1}
.hleg{display:flex;flex-wrap:wrap;gap:3px 14px;margin-top:8px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.empty{color:var(--dim);font-family:var(--mono);font-size:12.5px;padding:16px 4px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--panel-3);border-radius:6px}
::-webkit-scrollbar-thumb:hover{background:#2a2f4d}
.shell{display:flex;gap:14px;align-items:stretch;margin-top:14px}
.sidebar{flex:0 0 auto;width:192px;background:linear-gradient(180deg,var(--panel-2),var(--panel));
  border:1px solid var(--border);border-radius:16px;padding:11px;display:flex;flex-direction:column;gap:4px;
  position:sticky;top:16px;transition:width .22s ease}
.sidebar.collapsed{width:64px}
.sb-h{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:2px 4px 6px}
.sb-title{font-size:10.5px;color:var(--dim);font-family:var(--mono);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.sidebar.collapsed .sb-title{display:none}
.sb-toggle{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:7px;width:24px;height:24px;
  cursor:pointer;display:grid;place-items:center;font-size:10px;flex:0 0 auto;transition:color .15s,border-color .15s;padding:0}
.sb-toggle:hover{color:var(--txt);border-color:var(--border-2)}
.sb-nav{display:flex;flex-direction:column;gap:5px}
.tab{display:flex;align-items:center;gap:10px;background:transparent;border:1px solid transparent;color:var(--muted);
  border-radius:10px;padding:9px 11px;cursor:pointer;font-size:13px;font-weight:500;font-family:var(--sans);
  transition:background .15s,color .15s;text-align:left;width:100%;white-space:nowrap}
.tab:hover{background:rgba(255,255,255,.03);color:var(--txt)}
.tab.on{background:var(--grad);color:#fff;box-shadow:0 2px 12px rgba(124,123,255,.35);border-color:transparent}
.tic{font-size:16px;width:20px;text-align:center;flex:0 0 auto}
.sidebar.collapsed .tab{justify-content:center;padding:10px}
.sidebar.collapsed .tablabel{display:none}
.content{flex:1 1 auto;min-width:0}
.tabpane{display:none}
.tabpane.active{display:block}
.tabpane.active.row{display:grid}
@media(max-width:900px){
  .shell{flex-direction:column}
  .sidebar{width:100%!important;position:static;flex-direction:row;align-items:center;gap:8px;overflow-x:auto}
  .sidebar.collapsed{width:100%}
  .sb-h{padding:0}
  .sb-nav{flex-direction:row;flex:1}
  .tab{width:auto}
  .sidebar.collapsed .tab{padding:9px 11px}
}
.ctip{position:fixed;z-index:60;pointer-events:none;display:none;background:var(--panel-3);
  border:1px solid var(--border-2);border-radius:9px;padding:7px 10px;font-family:var(--mono);
  font-size:11.5px;color:var(--txt);box-shadow:0 10px 28px rgba(0,0,0,.55);line-height:1.55;white-space:nowrap}
.ctip b{font-weight:600}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg></div>
      <div><h1>local-ai serving</h1><div class="sub">gpu inference monitor</div></div>
    </div>
    <div class="dot" id="dot"></div>
    <span class="chip" id="chip"><b>&mdash;</b></span>
    <span class="badge" id="badge">&mdash;</span>
    <div class="spacer"></div>
    <span class="therm" id="therm" title="GPU ≥78° warn / ≥88° hot · CPU ≥85° warn / ≥95° hot"><i class="tdot"></i><span id="therm_txt">–</span></span>
    <div class="wins" id="wins"></div>
    <span class="meta" id="meta"></span>
  </div>

  <div class="shell">
    <aside class="sidebar" id="sidebar">
      <div class="sb-h">
        <span class="sb-title">Sections</span>
        <button class="sb-toggle" id="sb_toggle" title="Toggle sidebar">&#9664;</button>
      </div>
      <nav class="sb-nav" id="sb_nav"></nav>
    </aside>
    <div class="content">
    <div class="grid">
    <div class="card dec"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></span><span class="k">Decode TPS</span><span class="ks">now</span></div><div class="cv" id="c_dec">-</div><div class="cu">tokens/s<canvas id="spark_dec"></canvas></div></div>
    <div class="card dec"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></span><span class="k">Decode TPS</span><span class="ks">avg</span></div><div class="cv" id="c_dec_avg">-</div><div class="cu" id="u_dec_avg">window</div></div>
    <div class="card dec"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></span><span class="k">Decode TPS</span><span class="ks">peak</span></div><div class="cv" id="c_dec_peak">-</div><div class="cu" id="u_dec_peak">window</div></div>
    <div class="card"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></span><span class="k">Requests</span><span class="ks">live</span></div><div class="cv" id="c_run">-</div><div class="cu" id="u_run">running / waiting</div></div>
    <div class="card pre"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></span><span class="k">Prefill TPS</span><span class="ks">now</span></div><div class="cv" id="c_pre">-</div><div class="cu">tokens/s<canvas id="spark_pre"></canvas></div></div>
    <div class="card pre"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></span><span class="k">Prefill TPS</span><span class="ks">avg</span></div><div class="cv" id="c_pre_avg">-</div><div class="cu" id="u_pre_avg">window</div></div>
    <div class="card pre"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></span><span class="k">Prefill TPS</span><span class="ks">peak</span></div><div class="cv" id="c_pre_peak">-</div><div class="cu" id="u_pre_peak">window</div></div>
    <div class="card"><div class="ct"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 14 8 8"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg></span><span class="k">GPU active</span><span class="ks">window</span></div><div class="cv" id="c_active">-</div><div class="cu">of window</div></div>
  </div>

  <div class="panel tabpane active" id="tp-throughput" data-tab="throughput">
    <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg></span><h2>Throughput</h2><span class="note" id="chart_note">tokens/s</span></div>
    <canvas id="chart"></canvas>
    <div class="legend"><span><i style="background:var(--dec)"></i>decode (generation)</span><span><i style="background:var(--pre)"></i>prefill</span><span class="dim">left=decode · right=prefill, auto-scaled</span></div>
  </div>

  <div class="panel tabpane" id="tp-therm" data-tab="therm">
    <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg></span><h2>Thermal</h2><span class="note" id="therm_note">°C</span></div>
    <canvas id="therm_chart"></canvas>
    <div class="legend"><span><i style="background:var(--dec)"></i>GPU core °C</span><span><i style="background:var(--pre)"></i>CPU package °C</span><span class="dim" id="therm_fan"></span></div>
  </div>

  <div class="row tabpane" id="tp-streams" data-tab="streams">
    <div class="panel">
      <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg></span><h2>Concurrent streams</h2><span class="note" id="stream_note"></span></div>
      <div class="streams" id="streams"></div>
      <div class="subhead"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>recently completed &middot; short-term</div>
      <div class="sdone" id="streams_done"></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M8 17v-5M13 17V6M18 17v-8"/></svg></span><h2>Concurrency over time</h2><span class="note" id="conc_note"></span></div>
      <canvas id="conc_chart" class="conc"></canvas>
      <div class="legend"><span><i style="background:var(--run)"></i>running</span><span><i style="background:var(--pre)"></i>prefilling</span><span><i style="background:var(--wait)"></i>waiting</span></div>
    </div>
  </div>

  <div class="panel tabpane" id="tp-load" data-tab="load">
    <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="7" rx="1"/><rect x="12" y="6" width="3" height="11" rx="1"/><rect x="17" y="13" width="3" height="4" rx="1"/></svg></span><h2>Load &middot; KV &middot; context</h2><span class="note" id="load_note"></span></div>
    <div class="loadrow">
      <div class="loadcol">
        <div class="kvbox">
          <div class="kvl"><span>KV occupancy</span><span class="kvtag">estimate</span></div>
          <div class="kvbig"><b id="kv_pct">-</b><span>/ 100%</span></div>
          <div class="kvbar"><i id="kv_fill"></i></div>
          <div class="kvmeta" id="kv_meta"></div>
          <div class="kvmini"><span class="l">avg context</span><span class="n" id="kv_avgctx">-</span></div>
          <div class="kvmini"><span class="l">KV pool</span><span class="n" id="kv_pool">-</span></div>
        </div>
        <div class="kvbox">
          <div class="kvl"><span>GPU power</span><span class="kvtag">nvidia-smi</span></div>
          <div class="kvbig"><b id="gpu_power">-</b><span>W now</span></div>
          <div class="kvbar"><i id="gpu_fill"></i></div>
          <div class="kvmini"><span class="l">VRAM</span><span class="n" id="gpu_mem">-</span></div>
          <div class="kvmini"><span class="l">util now</span><span class="n" id="gpu_util">-</span></div>
          <div class="kvmini"><span class="l">util trend</span><span class="n dim" id="gpu_avg">-</span></div>
          <canvas id="gpu_spark" class="spark"></canvas>
        </div>
      </div>
      <div class="loadchart"><div class="subhead2">context length distribution</div><canvas id="hist_chart"></canvas><div class="hleg" id="hist_legend"></div></div>
      <div class="loadchart"><div class="subhead2">context &times; decode tok/s <span class="dim">(color = ttft)</span></div><canvas id="scatter_chart"></canvas></div>
    </div>
  </div>

  <div class="row tabpane" id="tp-req" data-tab="req">
    <div class="panel">
      <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg></span><h2>Requests</h2><span class="note" id="req_note"></span></div>
      <div class="kv" id="req_kv"></div>
      <div class="subhead"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/></svg>Cache reuse</div>
      <div id="reuse" class="muted">-</div>
      <div class="subhead"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/></svg>Spec decoding</div>
      <div id="specdec" class="muted">-</div>
    </div>
    <div class="panel">
      <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg></span><h2>Live queue &middot; errors</h2></div>
      <div class="kv" id="live_kv"></div>
      <div class="subhead"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><path d="M12 9v4M12 17h.01"/></svg>Recent errors</div>
      <div id="errs" class="muted">none</div>
    </div>
  </div>

  <div class="panel tabpane" id="tp-recent" data-tab="recent">
    <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h18v18H3z"/><path d="M3 9h18M9 3v18"/></svg></span><h2>Recent requests</h2><span class="note">last 30 · newest first</span></div>
    <div style="overflow-x:auto"><table><thead><tr><th>time</th><th>req</th><th>finish</th><th>prompt</th><th>gen</th><th>ttft</th><th>dec/s</th><th>wall</th><th>mtp</th></tr></thead><tbody id="req_rows"></tbody></table></div>
  </div>

  <div class="panel tabpane" id="tp-log" data-tab="log">
    <div class="ph"><span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m4 17 6-6-6-6"/><path d="M12 19h8"/></svg></span><h2>Raw log</h2><span class="note">newest first · live tail</span></div>
    <div class="log" id="log"></div>
  </div>
    </div>
  </div>
</div>

<script>
const WINS=[[60,"1m"],[300,"5m"],[900,"15m"],[3600,"1h"]];
let win=300;
const $=id=>document.getElementById(id);
let LAST=null;
function fmt(v,d=1){return v==null?"-":(+v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});}
function fmt0(v){return v==null?"-":Math.round(v).toLocaleString();}
function tstr(t){return t==null?"-":new Date(t*1000).toLocaleTimeString([],{hour12:false});}
function ms(v){if(v==null)return "-";return v>=1000?fmt(v/1000,2)+"s":Math.round(v)+"ms";}

const C={dec:'#5b93ff',pre:'#f5a524',run:'#2fd6a0',wait:'#37c6f0',err:'#ff6b7a',
  grid:'rgba(150,160,190,.09)',label:'#5c6478',dim:'#5c6478',muted:'#9aa3b8'};
const CTXC=['#37d39f','#37c6f0','#5b93ff','#f5a524','#ff6b7a'];
const CTXL=['0-30K','30-60K','60-100K','100-150K','160K+'];
function ctxBand(n){return n==null?-1:n<30000?0:n<60000?1:n<100000?2:n<150000?3:4;}
function ctxColor(n){const i=ctxBand(n);return i<0?'#9aa3b8':CTXC[i];}
function hexA(hex,a){const h=hex.replace('#','');const s=h.length===3?h.split('').map(c=>c+c).join(''):h;
  const n=parseInt(s,16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}
function smoothPath(x,pts){
  if(pts.length<2)return;
  x.moveTo(pts[0][0],pts[0][1]);
  if(pts.length===2){x.lineTo(pts[1][0],pts[1][1]);return;}
  for(let i=0;i<pts.length-1;i++){
    const p0=pts[i-1]||pts[i],p1=pts[i],p2=pts[i+1],p3=pts[i+2]||p2;
    const c1x=p1[0]+(p2[0]-p0[0])/6,c1y=p1[1]+(p2[1]-p0[1])/6;
    const c2x=p2[0]-(p3[0]-p1[0])/6,c2y=p2[1]-(p3[1]-p1[1])/6;
    x.bezierCurveTo(c1x,c1y,c2x,c2y,p2[0],p2[1]);
  }
}
function areaAndLine(x,pts,color,padT,ch,glow){
  if(pts.length<2)return;
  const bottom=padT+ch;
  x.save();
  const g=x.createLinearGradient(0,padT,0,bottom);
  g.addColorStop(0,hexA(color,.26));g.addColorStop(1,hexA(color,0));
  x.beginPath();smoothPath(x,pts);x.lineTo(pts[pts.length-1][0],bottom);x.lineTo(pts[0][0],bottom);x.closePath();
  x.fillStyle=g;x.fill();
  x.beginPath();smoothPath(x,pts);x.lineJoin='round';x.lineCap='round';
  if(glow){x.shadowColor=color;x.shadowBlur=9;}
  x.strokeStyle=color;x.lineWidth=2;x.stroke();x.shadowBlur=0;
  const L=pts[pts.length-1];
  x.beginPath();x.arc(L[0],L[1],3.2,0,7);x.fillStyle=color;x.shadowColor=color;x.shadowBlur=9;x.fill();x.shadowBlur=0;
  x.restore();
}
function drawSpark(id,vals,color){
  const c=$(id);if(!c||!c.clientWidth)return;
  const dpr=window.devicePixelRatio||1;const W=c.clientWidth,H=c.clientHeight;
  if(!W||!H)return;
  if(c.width!==Math.round(W*dpr)){c.width=W*dpr;c.height=H*dpr;}
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const v=vals.filter(n=>n!=null);if(v.length<2)return;
  const mn=Math.min(...v),mx=Math.max(...v),rng=(mx-mn)||1;
  const P=vals.map((n,i)=>[i/(vals.length-1)*(W-2)+1,H-2-(((n==null?mn:n)-mn)/rng)*(H-4)]);
  const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,hexA(color,.42));g.addColorStop(1,hexA(color,0));
  x.beginPath();smoothPath(x,P);x.lineTo(P[P.length-1][0],H);x.lineTo(P[0][0],H);x.closePath();x.fillStyle=g;x.fill();
  x.beginPath();smoothPath(x,P);x.strokeStyle=color;x.lineWidth=1.5;x.lineJoin='round';x.lineCap='round';x.stroke();
  c._pts=P.map((q,i)=>({x:q[0],y:q[1],html:fmt(vals[i])}));bindTip(c,"dist");
}
function drawSpark100(id,vals,color){
  const c=$(id);if(!c||!c.clientWidth)return;
  const dpr=window.devicePixelRatio||1;const W=c.clientWidth,H=c.clientHeight;
  if(!W||!H)return;
  if(c.width!==Math.round(W*dpr)){c.width=W*dpr;c.height=H*dpr;}
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const v=vals.filter(n=>n!=null);if(v.length<2)return;
  const P=vals.map((n,i)=>[i/(vals.length-1)*(W-2)+1,H-2-((n==null?0:n)/100)*(H-4)]);
  x.strokeStyle=C.grid;x.beginPath();x.moveTo(0,H-2);x.lineTo(W,H-2);x.stroke();
  const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,hexA(color,.42));g.addColorStop(1,hexA(color,0));
  x.beginPath();smoothPath(x,P);x.lineTo(P[P.length-1][0],H);x.lineTo(P[0][0],H);x.closePath();x.fillStyle=g;x.fill();
  x.beginPath();smoothPath(x,P);x.strokeStyle=color;x.lineWidth=1.5;x.lineJoin='round';x.lineCap='round';x.stroke();
  c._pts=P.map((q,i)=>({x:q[0],y:q[1],html:Math.round(vals[i]==null?0:vals[i])+"%"}));bindTip(c,"dist");
}

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
function finishCol(f){if(!f)return C.muted;if(/cancel|error/i.test(f))return C.err;if(/limit|length/i.test(f))return C.pre;return C.dec;}
function mtpCell(m){if(m==null)return'-';const col=m>=70?C.run:m>=40?C.pre:C.dim;return `<span style="color:${col}">${fmt(m,1)}%</span>`;}

let tipEl=null;
function ensureTip(){if(!tipEl){tipEl=document.createElement("div");tipEl.className="ctip";document.body.appendChild(tipEl);}return tipEl;}
function showTip(cx,cy,html){const t=ensureTip();t.innerHTML=html;t.style.display="block";
  const tw=t.offsetWidth,th=t.offsetHeight;let px=cx+14,py=cy+14;
  if(px+tw>window.innerWidth-8)px=cx-tw-14;if(py+th>window.innerHeight-8)py=cy-th-14;
  t.style.left=px+"px";t.style.top=py+"px";}
function hideTip(){if(tipEl)tipEl.style.display="none";}
function bindTip(c,mode){if(c._tip)return;c._tip=1;
  c.addEventListener("mousemove",e=>{
    const r=c.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
    const pts=c._pts||[];if(!pts.length){hideTip();return;}
    let best=null,bd=mode==="x"?45:18*18;
    if(mode==="x"){const lim=45;for(const p of pts){const d=Math.abs(mx-p.x);if(d<lim&&d<bd){bd=d;best=p;}}}
    else{for(const p of pts){const dx=mx-p.x,dy=my-p.y,d=dx*dx+dy*dy;if(d<bd){bd=d;best=p;}}}
    if(best)showTip(e.clientX,e.clientY,best.html);else hideTip();
  });
  c.addEventListener("mouseleave",hideTip);}

function drawChart(series,peak,w,now){
  const c=$("chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=220;c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=46,padR=46,padT=12,padB=22;
  const cw=W-padL-padR,ch=H-padT-padB;
  const t1=now,t0=now-w;
  const inw=(series||[]).filter(p=>p[0]>=t0&&p[0]<=t1);
  let decPeak=0,prePeak=0;
  for(const p of inw){if((p[1]||0)>decPeak)decPeak=p[1];if((p[2]||0)>prePeak)prePeak=p[2];}
  const decYv=Math.max(10,decPeak),preYv=Math.max(10,prePeak);
  x.font="10px ui-monospace, monospace";x.lineWidth=1;
  for(let g=0;g<=4;g++){const y=padT+ch-ch*g/4;
    x.strokeStyle=C.grid;x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();
    x.fillStyle=C.dec;x.fillText(String(Math.round(decYv*g/4)),4,y+3);
    x.fillStyle=C.pre;x.fillText(String(Math.round(preYv*g/4)),W-padR+5,y+3);}
  x.fillStyle=C.label;
  for(let g=0;g<=4;g++){const tt=t0+w*g/4,px=padL+cw*g/4;x.fillText(tstr(tt).slice(0,5),px-13,H-6);}
  if(inw.length<2){x.fillStyle=C.dim;x.fillText("waiting for samples…",padL+10,padT+22);return;}
  const X=t=>padL+cw*((t-t0)/w);
  const Ydec=v=>padT+ch-ch*Math.min(v,decYv)/decYv;
  const Ypre=v=>padT+ch-ch*Math.min(v,preYv)/preYv;
  areaAndLine(x,inw.map(p=>[X(p[0]),Ypre(p[2])]),C.pre,padT,ch,false);
  areaAndLine(x,inw.map(p=>[X(p[0]),Ydec(p[1])]),C.dec,padT,ch,true);
  c._pts=inw.map(p=>({x:X(p[0]),html:`<b>${tstr(p[0])}</b><br>`+
    `<span style="color:${C.dec}">decode ${fmt(p[1])} (left axis)</span><br>`+
    `<span style="color:${C.pre}">prefill ${fmt(p[2])} (right axis)</span>`}));
  bindTip(c,"x");
}

function drawTherm(series,w,now){
  const c=$("therm_chart");if(!c)return;const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=220;c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=42,padR=12,padT=12,padB=22;
  const cw=W-padL-padR,ch=H-padT-padB;
  const t1=now,t0=now-w;
  const inw=(series||[]).filter(p=>p[0]>=t0&&p[0]<=t1);
  let hi=0;
  for(const p of inw)for(const i of [1,2])if(p[i]!=null&&p[i]>hi)hi=p[i];
  const lo=40,hi2=Math.max(45,Math.ceil((hi+3)/5)*5);
  x.font="10px ui-monospace, monospace";x.lineWidth=1;
  for(let g=0;g<=4;g++){const val=lo+(hi2-lo)*g/4,y=padT+ch-ch*g/4;
    x.strokeStyle=C.grid;x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();
    x.fillStyle=C.label;x.fillText(Math.round(val)+"°",4,y+3);}
  for(let g=0;g<=4;g++){const tt=t0+w*g/4,px=padL+cw*g/4;x.fillStyle=C.label;x.fillText(tstr(tt).slice(0,5),px-13,H-6);}
  if(inw.length<2){x.fillStyle=C.dim;x.fillText("collecting…",padL+10,padT+22);return;}
  const X=t=>padL+cw*((t-t0)/w);
  const Y=v=>padT+ch-ch*(Math.min(Math.max(v,lo),hi2)-lo)/(hi2-lo);
  const gpts=inw.filter(p=>p[1]!=null).map(p=>[X(p[0]),Y(p[1])]);
  const cpts=inw.filter(p=>p[2]!=null).map(p=>[X(p[0]),Y(p[2])]);
  if(cpts.length>1)areaAndLine(x,cpts,C.pre,padT,ch,false);
  if(gpts.length>1)areaAndLine(x,gpts,C.dec,padT,ch,true);
  c._pts=inw.map(p=>({x:X(p[0]),html:`<b>${tstr(p[0])}</b><br>`+
    `<span style="color:${C.dec}">GPU ${p[1]==null?"-":Math.round(p[1])+"°"}</span><br>`+
    `<span style="color:${C.pre}">CPU ${p[2]==null?"-":Math.round(p[2])+"°"}</span>`}));
  bindTip(c,"x");
}

function drawConc(series,w,now){
  const c=$("conc_chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=150;c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=30,padR=10,padT=10,padB=18;
  const cw=W-padL-padR,ch=H-padT-padB;
  const t1=now,t0=now-w;
  x.font="10px ui-monospace, monospace";x.lineWidth=1;
  let yv=4;for(const p of series)for(let i=1;i<p.length;i++)yv=Math.max(yv,p[i]);
  yv=Math.max(4,Math.ceil(yv));
  x.strokeStyle=C.grid;x.fillStyle=C.label;
  for(let g=0;g<=2;g++){const y=padT+ch-ch*g/2;x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();x.fillText(String(yv*g/2),4,y+3);}
  if(!series||series.length<2){x.fillStyle=C.dim;x.fillText("waiting for samples…",padL+10,padT+18);return;}
  const X=t=>padL+cw*((t-t0)/w);
  const Y=v=>padT+ch-ch*Math.min(v,yv)/yv;
  const inw=series.filter(p=>p[0]>=t0&&p[0]<=t1);
  const mk=idx=>inw.map(p=>[X(p[0]),Y(p[idx])]);
  areaAndLine(x,mk(1),C.run,padT,ch,true);
  function thin(pts,color){x.beginPath();smoothPath(x,pts);x.lineJoin='round';x.lineCap='round';x.strokeStyle=color;x.lineWidth=1.5;x.stroke();}
  thin(mk(3),C.wait);thin(mk(2),C.pre);
  c._pts=inw.map(p=>({x:X(p[0]),html:`<b>${tstr(p[0])}</b><br>`+
    `<span style="color:${C.run}">running ${p[1]==null?0:p[1]}</span><br>`+
    `<span style="color:${C.pre}">prefilling ${p[2]==null?0:p[2]}</span><br>`+
    `<span style="color:${C.wait}">waiting ${p[3]==null?0:p[3]}</span>`}));
  bindTip(c,"x");
}

function drawHistogram(buckets){
  const c=$("hist_chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=Math.max(c.clientHeight||170,170);c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padB=20,padT=14,n=buckets.length;if(!n){x.fillStyle=C.dim;x.fillText("no completed requests",10,30);return;}
  const maxN=Math.max(1,...buckets.map(b=>b.n));
  const bw=(W-16)/n,barW=bw*0.56,base=H-padB,chartH=base-padT;
  x.strokeStyle=C.grid;x.lineWidth=1;x.beginPath();x.moveTo(6,base+.5);x.lineTo(W-6,base+.5);x.stroke();
  x.font="10px ui-monospace, monospace";x.textAlign="center";
  const pts=[];
  buckets.forEach((b,i)=>{
    const cx=8+i*bw+bw/2, bh=chartH*(b.n/maxN), top=base-bh;
    const bc=CTXC[i]||C.dec;
    x.fillStyle=b.n?hexA(bc,.32):"rgba(150,160,190,.05)";
    x.fillRect(cx-barW/2, top, barW, bh);
    x.fillStyle=b.n?bc:"rgba(150,160,190,.16)";
    x.fillRect(cx-barW/2, top, barW, 2);
    x.fillStyle=b.n?C.muted:C.dim;
    x.fillText(String(b.n), cx, top-5);
    x.fillStyle=C.dim;
    x.fillText(b.label, cx, H-7);
    pts.push({x:cx,html:`<b>${b.label}</b><br>requests ${b.n}`+
      (b.avg_ttft_ms!=null?`<br>ttft ${ms(b.avg_ttft_ms)}`:"")+
      (b.avg_decode!=null?`<br>${b.avg_decode.toFixed(0)} t/s`:"")});
  });
  c._pts=pts;bindTip(c,"x");
  x.textAlign="left";
}
function drawScatter(pts){
  const c=$("scatter_chart");const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=Math.max(c.clientHeight||170,170);c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const padL=34,padR=10,padT=10,padB=20;
  const cw=W-padL-padR,ch=H-padT-padB;
  if(!pts||pts.length<2){x.fillStyle=C.dim;x.fillText("need ≥2 completed requests",padL+6,padT+24);return;}
  const mx=Math.max(...pts.map(p=>p.x))||1,my=Math.max(...pts.map(p=>p.y))||1;
  x.font="10px ui-monospace, monospace";x.lineWidth=1;x.strokeStyle=C.grid;x.fillStyle=C.dim;
  for(let g=0;g<=3;g++){
    const y=padT+ch-ch*g/3;
    x.beginPath();x.moveTo(padL,y);x.lineTo(W-padR,y);x.stroke();
    x.fillText(String(Math.round(my*g/3)),4,y+3);
  }
  for(let g=0;g<=3;g++){x.fillText(Math.round(mx*g/3/1000)+"k",padL+cw*g/3-9,H-6);}
  const tt=pts.map(p=>p.ttft).filter(v=>v!=null);
  const tmin=tt.length?Math.min(...tt):0,tmax=tt.length?Math.max(...tt):0;
  const X=v=>padL+cw*(v/mx), Y=v=>padT+ch-ch*(v/my);
  pts.forEach(p=>{
    const t=p.ttft==null?.5:((p.ttft-tmin)/((tmax-tmin)||1));
    const col=t<0.33?C.run:t<0.66?C.pre:C.err;
    x.beginPath();x.arc(X(p.x),Y(p.y),3.2,0,7);
    x.fillStyle=hexA(col,.72);x.fill();
  });
  c._pts=pts.map(p=>({x:X(p.x),y:Y(p.y),html:`<b>${Math.round(p.x)} ctx</b><br>decode ${fmt(p.y)} t/s<br>ttft ${ms(p.ttft)}`}));
  bindTip(c,"dist");
}

function render(s){
  const c=s.container,l=s.live,tp=s.tp,R=s.requests;
  LAST=s;
  $("dot").className="dot "+(c.running?"up":"err");
  $("chip").innerHTML="<b>"+(c.name||"no container")+"</b>&nbsp;·&nbsp;"+(c.framework||"?")
    + "&nbsp;·&nbsp;"+(c.running?(c.state||"up"):(c.state||"exited"));
  const badge=$("badge");
  let bcls="badge",btxt;
  if(!c.running){btxt="not running";}
  else if(!c.has_parser){btxt="parser pending";bcls="badge pending";}
  else if(l.idle){btxt="idle";bcls="badge idle";}
  else{btxt="busy";bcls="badge busy";}
  badge.className=bcls;badge.innerHTML='<span class="bdot"></span>'+btxt;
  $("meta").textContent="updated "+Math.max(0,Math.round(s.server.now-s.server.last_poll))+"s ago · auto 2s";

  const wlabel={60:"1m",300:"5m",900:"15m",3600:"1h"}[win]||win+"s";
  $("c_dec").textContent=l.idle?"idle":fmt(l.decode_tps);
  $("c_dec_avg").textContent=fmt(tp.avg_decode);$("u_dec_avg").textContent=wlabel+" · active · "+(tp.samples||0)+" samples";
  $("c_dec_peak").textContent=fmt(tp.peak_decode);$("u_dec_peak").textContent=wlabel;
  $("c_pre").textContent=l.idle?"idle":fmt(l.prefill_tps);
  $("c_pre_avg").textContent=fmt(tp.avg_prefill);$("u_pre_avg").textContent=wlabel+" · active";
  $("c_pre_peak").textContent=fmt(tp.peak_prefill);$("u_pre_peak").textContent=wlabel;
  $("c_run").textContent=(l.running==null?"-":l.running)+" / "+(l.waiting==null?"-":l.waiting);
  $("u_run").textContent="prefilling "+(l.prefilling==null?"-":l.prefilling);
  $("c_active").textContent=tp.active_pct==null?"-":fmt(tp.active_pct,1)+"%";
  $("chart_note").textContent="last "+wlabel+" · peak "+s.peak+" tok/s";

  const th=s.thermal||{},tn=th.now||{};
  if(tn.gpu_c!=null||tn.cpu_c!=null){
    $("therm_txt").textContent=(tn.gpu_c!=null?"GPU "+Math.round(tn.gpu_c)+"°":"-")
      +(tn.cpu_c!=null?" · CPU "+Math.round(tn.cpu_c)+"°":"")
      +(tn.gpu_w!=null?" · "+Math.round(tn.gpu_w)+"W":"");
    const hot=(tn.gpu_c!=null&&tn.gpu_c>=88)||(tn.cpu_c!=null&&tn.cpu_c>=95);
    const warn=(tn.gpu_c!=null&&tn.gpu_c>=78)||(tn.cpu_c!=null&&tn.cpu_c>=85);
    $("therm").className="therm"+(hot?" hot":warn?" warn":"");
  }
  $("therm_note").textContent="last "+wlabel+" · GPU max "+(th.max_gpu==null?"-":Math.round(th.max_gpu)+"°")+" · CPU max "+(th.max_cpu==null?"-":Math.round(th.max_cpu)+"°");
  $("therm_fan").textContent=tn.fan_pct!=null?"GPU fan "+tn.fan_pct+"%":"";

  drawChart(s.series,s.peak,win,s.server.now);
  if(activeTab==="therm")drawTherm(th.series,win,s.server.now);
  drawSpark("spark_dec",s.series.map(p=>p[1]).slice(-36),C.dec);
  drawSpark("spark_pre",s.series.map(p=>p[2]).slice(-36),C.pre);

  const streams=s.streams||[];
  $("stream_note").textContent=streams.length
    ?(streams.length+" active · heaviest "+(streams[0].age_s!=null?streams[0].age_s.toFixed(1)+"s":"-"))
    :(c.per_request?"none in flight":"not reported by "+c.framework);
  let streamsHTML;
  if(streams.length){
    const groups=new Map();
    for(const sr of streams){
      const k=(sr.proto||"?")+"|"+(sr.max_tokens==null?"?":sr.max_tokens);
      if(!groups.has(k))groups.set(k,{proto:sr.proto||"?",max_tokens:sr.max_tokens,items:[]});
      groups.get(k).items.push(sr);
    }
    const glist=[...groups.values()].sort((A,B)=>
      B.items.reduce((x,s)=>x+(s.age_s||0),0)-A.items.reduce((x,s)=>x+(s.age_s||0),0));
    streamsHTML=glist.map(g=>{
      const n=g.items.length;
      const sumMsgs=g.items.reduce((x,s)=>x+(s.msgs||0),0);
      const oldest=Math.max(0,...g.items.map(s=>s.age_s||0));
      const lim=g.max_tokens==null?"?":(g.max_tokens>=1000?(+(g.max_tokens/1000).toFixed(1))+"k":g.max_tokens)+" tok";
      const chips=g.items.map(sr=>{
        const a=sr.age_s;
        const col=a==null?C.dim:(a<30?C.run:a<120?C.pre:C.err);
        const stale=a!=null&&a>300;
        return `<span class="reqchip${stale?" stale":""}" style="color:${col}"><span class="rid">#${esc(sr.req)}</span>${a!=null?`<span class="rage">${a.toFixed(0)}s</span>`:""}</span>`;
      }).join("");
      return `<div class="sgrp">
        <div class="ghead"><b>${esc(g.proto)}</b><span class="gcnt">${n}</span>
          <span class="meta">${lim} max · Σ ${sumMsgs} msgs · oldest ${oldest.toFixed(0)}s</span></div>
        <div class="chips">${chips}</div>
      </div>`;
    }).join("");
  }else{
    streamsHTML=c.per_request
      ?'<div class="empty">no active streams — waiting for requests…</div>'
      :'<div class="empty">'+c.framework+' does not log per-request streams — live concurrency is in the "Requests" card and the concurrency-over-time chart…</div>';
  }
  $("streams").innerHTML=streamsHTML;
  const cbadge=r=>{const h=r.cache,u=r.reuse;
    return h>0?`<span class="cbadge warm" title="${esc(u||"cache hit")}">⚡ +${h>=1000?(+(h/1000).toFixed(1))+"k":h} cached</span>`
              :`<span class="cbadge cold" title="${esc(u||"root")}">○ cold</span>`;};
  const sd=(s.recent_requests||[]).slice(0,8);
  $("streams_done").innerHTML=sd.length?sd.map(r=>{
    const w=r.wall;const wcol=w==null?C.dim:w<30?C.run:w<120?C.pre:C.err;
    const ctx=r.prompt==null?"?":(r.prompt>=1000?(+(r.prompt/1000).toFixed(1))+"k":r.prompt);
    return `<div class="d"><span class="rid">#${r.req}</span>
      <span class="m">${ctx} ctx · ${r.gen==null?"?":fmt0(r.gen)} gen · ttft ${ms(r.ttft_ms)} · mtp ${mtpCell(r.mtp_pct)} ${cbadge(r)}</span>
      <span class="wall" style="color:${wcol}">${w==null?"-":fmt(w,1)+"s"}</span></div>`;
  }).join(""):'<div class="empty">no completed requests yet</div>';
  drawConc(s.conc_series,win,s.server.now);
  $("conc_note").textContent="last "+wlabel+" · now running "+(l.running==null?"-":l.running)+" / waiting "+(l.waiting==null?"-":l.waiting);

  const A=s.analysis||{};
  const KN=A.kv_now||{};
  $("load_note").textContent=A.kv_total?("pool "+A.kv_total.toLocaleString()+" tok · est = in-flight × avg ctx"):"";
  const kp=KN.pct;
  $("kv_pct").textContent=kp==null?"-":kp.toFixed(1)+"%";
  $("kv_fill").style.width=(kp==null?0:kp)+"%";
  $("kv_fill").style.background=kp==null?"":(kp<50?"var(--grad)":kp<80?"var(--pre)":"var(--err)");
  $("kv_meta").textContent=(KN.in_flight!=null)
    ?(KN.in_flight+" in-flight × ~"+((A.avg_prompt||0)/1000).toFixed(0)+"k ctx ≈ "+(KN.tokens||0).toLocaleString()+" tok")
    :"no in-flight requests right now";
  $("kv_avgctx").textContent=A.avg_prompt==null?"-":(A.avg_prompt/1000).toFixed(0)+"k tok";
  $("kv_pool").textContent=A.kv_total==null?"-":(A.kv_total/1000).toFixed(0)+"k tok";
  drawHistogram(A.ctx_hist||[]);
  drawScatter(A.scatter||[]);
  $("hist_legend").innerHTML=(A.ctx_hist||[]).map((b,idx)=>b.n?
    `<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${CTXC[idx]||C.dec};margin-right:5px;vertical-align:middle"></i>${b.label} · ${b.n}${b.avg_ttft_ms!=null?" · ttft "+ms(b.avg_ttft_ms):""}${b.avg_decode!=null?" · "+b.avg_decode.toFixed(0)+" t/s":""}</span>`:'').join("")
    ||'<span class="muted">no completed requests in window</span>';

  const G=s.gpu||{},gn=G.now||{};
  if(gn.power_w!=null){
    const lim=gn.power_limit_w||600, pp=Math.min(100,Math.max(0,gn.power_w/lim*100));
    $("gpu_power").textContent=Math.round(gn.power_w);
    $("gpu_fill").style.width=pp+"%";
    $("gpu_util").textContent=gn.util!=null?Math.round(gn.util)+"%":"-";
    $("gpu_mem").textContent=(gn.mem_used!=null&&gn.mem_total?(gn.mem_used/1024).toFixed(0)+"G / "+(gn.mem_total/1024).toFixed(0)+"G":"-");
  }else{
    $("gpu_power").textContent="-";$("gpu_fill").style.width="0%";$("gpu_util").textContent="-";$("gpu_mem").textContent="-";
  }
  const gser=(G.series||[]).map(p=>p[1]).filter(v=>v!=null);
  if(gser.length){
    $("gpu_avg").textContent="avg "+Math.round(gser.reduce((a,b)=>a+b,0)/gser.length)+"% · "+gser.length+"s";
    drawSpark100("gpu_spark",gser.slice(-120),C.dec);
  }else{
    $("gpu_avg").textContent="-";$("gpu_spark")&&$("gpu_spark").getContext&&$("gpu_spark").getContext("2d").clearRect(0,0,500,500);
  }

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
  const SD=s.specdec||{};
  $("specdec").innerHTML=(SD.acc_rate!=null)?
    `<span class="tag">accept rate <b>${fmt(SD.acc_rate,1)}%</b></span>
     <span class="tag">accept len <b>${fmt(SD.acc_len,2)}</b></span>
     <span class="tag">accepted <b>${fmt(SD.acc_thr,1)} t/s</b></span>
     <span class="tag">drafted <b>${fmt(SD.draft_thr,1)} t/s</b></span>`
    :'<span class="muted">n/a (spec decoding off)</span>';

  $("live_kv").innerHTML=kv([
    ["running",l.running==null?"-":l.running],["waiting",l.waiting==null?"-":l.waiting],
    ["decode_ready",l.decode_ready==null?"-":l.decode_ready],
    ["avg batch",l.avg_decode_batch==null?"-":fmt(l.avg_decode_batch,2)],
  ]);
  $("errs").innerHTML=(s.recent_errors&&s.recent_errors.length)?
    s.recent_errors.slice(0,6).map(e=>`<div class="err" style="font-family:var(--mono);font-size:12px;margin:3px 0">[${tstr(e.t)}] <span class="muted">req ${e.req}</span> ${esc(e.msg)}</div>`).join("")
    :'<span class="muted">none in window</span>';

  $("req_rows").innerHTML=(s.recent_requests&&s.recent_requests.length)?
    s.recent_requests.map(r=>`<tr><td>${tstr(r.t)}</td><td>${r.req}</td><td class="f-finish" style="color:${finishCol(r.finish)}">${esc(r.finish||"")}</td>
    <td style="color:${ctxColor(r.prompt)};font-weight:600" title="${r.prompt!=null?CTXL[ctxBand(r.prompt)]:""}">${fmt0(r.prompt)}</td><td>${fmt0(r.gen)}</td><td>${ms(r.ttft_ms)}</td>
    <td>${fmt(r.decode)}</td><td>${r.wall==null?"-":fmt(r.wall,1)+"s"}</td><td>${mtpCell(r.mtp_pct)}</td></tr>`).join("")
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
const TABS=[["throughput","&#128200;","Throughput"],["therm","&#127777;","Thermal"],["streams","&#128256;","Streams"],["load","&#128202;","Load · KV · Ctx"],["req","&#128230;","Req · Queue"],["recent","&#128344;","Recent"],["log","&#128220;","Log"]];
let activeTab=(()=>{const t=new URLSearchParams(location.search).get("tab");return TABS.some(x=>x[0]===t)?t:"throughput";})();
function buildSidebar(){
  const nav=$("sb_nav");nav.innerHTML="";
  for(const [key,ico,label] of TABS){
    const b=document.createElement("button");
    b.className="tab"+(key===activeTab?" on":"");
    b.dataset.tab=key;b.title=label;
    b.innerHTML='<span class="tic">'+ico+'</span><span class="tablabel">'+label+'</span>';
    b.onclick=()=>setTab(key);
    nav.appendChild(b);
  }
}
function setTab(key){
  activeTab=key;
  document.querySelectorAll(".tabpane").forEach(p=>p.classList.toggle("active",p.dataset.tab===key));
  document.querySelectorAll(".sb-nav .tab").forEach(b=>b.classList.toggle("on",b.dataset.tab===key));
  drawTab(key);
}
function drawTab(key){
  if(!LAST)return;const s=LAST;
  if(key==="throughput")drawChart(s.series,s.peak,win,s.server.now);
  else if(key==="therm")drawTherm((s.thermal||{}).series,win,s.server.now);
  else if(key==="streams")drawConc(s.conc_series,win,s.server.now);
  else if(key==="load"){
    const A=s.analysis||{};drawHistogram(A.ctx_hist||[]);drawScatter(A.scatter||[]);
    const gser=((s.gpu||{}).series||[]).map(p=>p[1]).filter(v=>v!=null);
    if(gser.length)drawSpark100("gpu_spark",gser.slice(-120),C.dec);
  }
}
function initSidebar(){
  const t=$("sb_toggle"),sb=$("sidebar");
  const apply=()=>{sb.classList.toggle("collapsed",t.dataset.c==="1");t.innerHTML=t.dataset.c==="1"?"&#9654;":"&#9664;";};
  t.onclick=()=>{t.dataset.c=t.dataset.c==="1"?"0":"1";localStorage.setItem("sb_c",t.dataset.c);apply();};
  const saved=localStorage.getItem("sb_c");
  t.dataset.c=saved!=null?saved:(window.innerWidth<760?"1":"0");
  apply();
  buildSidebar();
  setTab(activeTab);
}
buildWins();initSidebar();fetch_();setInterval(fetch_,2000);
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
    conn = init_db()
    Handler.conn = conn
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)

    t = threading.Thread(target=poll_loop, args=(st, args.tail, conn), daemon=True)
    t.start()
    print(f"dashboard on http://{args.bind}:{args.port}/ (poll {args.poll}s, tail {args.tail})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()