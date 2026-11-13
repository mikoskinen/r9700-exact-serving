#!/usr/bin/env python3
"""z0pf.py (11.11.2026, queue-ar dsmain, ds-main-port Z0 agent; driver of 57-p4zyzzzzz-ds-hy-z0prof2's prefill windows) --
a PREFILL window: K fresh prompts of mb.py's c4 / c8 size (--ctx tokens of filler + a task, a fresh nonce each, so no prefix
hit), max_tokens 1, --conc at a time, so the window's steps are prefill chunks (no decode rows but the first token);
/stats + GET /admin/fn4 (draft / lend / ctxskip objects) at the window's ends with their CLOCK_MONOTONIC ns, and --mark-cmd's
output there (the followed engine log's line count under --profile-ops, for d0attrprof.py ops / z0prof.py ar).
Prints "PFW {json}" (and appends it to --out): per request TTFT / prompt / cached, the /stats deltas (steps by class, ms a
step, prefill tokens), the counters' deltas. Standard library only; reads nothing of the user's.

  z0pf.py --url URL --label L [--k 8] [--ctx 1000] [--conc 1] [--set JSON] [--mark-cmd CMD] [--out FILE.jsonl]
  z0pf.py selftest"""
import argparse, json, subprocess, sys, threading, time, urllib.request

WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega vector buffer kernel stream engine queue "
         "packet record tensor matrix router expert layer token prompt window cache shard rank block slab arena").split()


def filler(nchars, seed):
    out, i, n = [], seed * 7919 + 13, 0
    while n < nchars:
        w = WORDS[i % len(WORDS)]
        i = (i * 1103515245 + 12345) & 0x7fffffff
        out.append(w); n += len(w) + 1
        if len(out) % 16 == 0: out.append("\n")
    return " ".join(out)[:nchars]


def http(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def mono():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def snap(a):
    t = mono()
    try: st = http(a.url + "/stats")
    except Exception: st = {}
    try: g = http(a.url + "/admin/fn4")
    except Exception: g = {}
    m = None
    if a.mark_cmd:
        try: m = subprocess.run(a.mark_cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e: m = "error %s" % str(e)[:80]
    return {"t": t, "stats": st, "fn4": g, "mark": m}


def one(a, i):
    q = "# pf %d %d\n" % (time.time_ns(), i) + filler(int(a.ctx * 3.9), 1000 + i) + "\n\nSummarise the text above in one sentence."
    p = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": q}], "max_tokens": 1, "stream": True,
         "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; u = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): u = ev["usage"]
            if tf is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                                  for c in ev.get("choices", [])):
                tf = time.time()
    t1 = time.time()
    return {"i": i, "ttft": round((tf or t1) - t0, 3), "wall": round(t1 - t0, 3), "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def delta(s0, s1):
    a, b = s0["stats"], s1["stats"]
    if not a or not b: return {"error": "no /stats"}
    ds = b.get("steps", 0) - a.get("steps", 0)
    dd = b.get("steps_decode", 0) - a.get("steps_decode", 0)
    dp = b.get("steps_prefill", 0) - a.get("steps_prefill", 0)
    dm = b.get("steps_mixed", 0) - a.get("steps_mixed", 0)
    ns = b.get("step_ns", 0) - a.get("step_ns", 0); nd = b.get("step_ns_decode", 0) - a.get("step_ns_decode", 0)
    o = {"steps": ds, "decode_steps": dd, "prefill_steps": dp, "mixed_steps": dm, "secs": round((s1["t"] - s0["t"]) / 1e9, 2),
         "ms_step": round(ns / max(1, ds) / 1e6, 2) if ds else None,
         "ms_nondecode_step": round((ns - nd) / max(1, ds - dd) / 1e6, 2) if ds > dd else None,
         "busy_s": round(ns / 1e9, 3)}
    def sub(k):
        x, y = (s0["fn4"].get(k) or {}), (s1["fn4"].get(k) or {})
        return {kk: (y.get(kk) - x.get(kk, 0)) if isinstance(y.get(kk), (int, float)) and not isinstance(y.get(kk), bool) else y.get(kk)
                for kk in y}
    o["draft"] = sub("draft"); o["lend"] = sub("lend"); o["ctxskip"] = sub("ctxskip")
    return o


def run(a):
    posted = None
    if a.set:
        try: http(a.url + "/admin/fn4", json.loads(a.set)); posted = True
        except Exception as e: posted = str(e)[:200]
        try: http(a.url + "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0}, timeout=120)
        except Exception: pass
    s0 = snap(a)
    recs, lock, nxt = [], threading.Lock(), [0]
    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= a.k: return
            try: r = one(a, i)
            except Exception as e: r = {"i": i, "error": str(e)[:200]}
            with lock: recs.append(r)
    th = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, a.conc))]
    for t in th: t.start()
    for t in th: t.join(timeout=1800)
    s1 = snap(a)
    ok = [r for r in recs if "error" not in r]
    sw = s1["fn4"].get("switches") or {}
    want = {k: v for k, v in (json.loads(a.set) if a.set else {}).items() if k != "epoch_bump"}
    row = {"label": a.label, "k": a.k, "ctx": a.ctx, "conc": a.conc, "posted": posted,
           "ok": posted in (None, True) and len(ok) == a.k and all(sw.get(k) == v for k, v in want.items())
                 and all((r.get("cached") or 0) == 0 for r in ok),
           "t0": s0["t"], "t1": s1["t"], "mark0": s0["mark"], "mark1": s1["mark"], "applied": sw,
           "ttft_mean": round(sum(r["ttft"] for r in ok) / len(ok), 3) if ok else None,
           "ttft_sorted": sorted(r["ttft"] for r in ok), "prompt_mean": round(sum(r.get("prompt") or 0 for r in ok) / len(ok), 1) if ok else None,
           "cached": [r.get("cached") for r in ok], "errors": [r for r in recs if "error" in r], "win": delta(s0, s1)}
    if a.out:
        with open(a.out, "a") as fh: fh.write(json.dumps(row) + "\n")
    print("PFW " + json.dumps(row), flush=True)
    return 0 if row["ok"] else 1


def selftest():
    a, b = filler(4000, 1), filler(4000, 2)
    assert len(a) == 4000 and a != b and filler(4000, 1) == a
    s0 = {"t": 0, "stats": {"steps": 10, "steps_decode": 4, "steps_prefill": 5, "steps_mixed": 1, "step_ns": 10 * 10**6, "step_ns_decode": 4 * 10**6},
          "fn4": {"lend": {"lent_units": 100, "state": 2}}}
    s1 = {"t": 2 * 10**9, "stats": {"steps": 20, "steps_decode": 6, "steps_prefill": 12, "steps_mixed": 2, "step_ns": 40 * 10**6, "step_ns_decode": 6 * 10**6},
          "fn4": {"lend": {"lent_units": 120, "state": 2}}}
    d = delta(s0, s1)
    assert d["steps"] == 10 and d["prefill_steps"] == 7 and d["mixed_steps"] == 1 and d["ms_nondecode_step"] == 3.5, d
    assert d["lend"]["lent_units"] == 20
    print("Z0PF selftest OK (filler deterministic and distinct; /stats deltas by step class; counter deltas)")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]: return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--set", default="")
    ap.add_argument("--mark-cmd", default=None)
    ap.add_argument("--out")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
