#!/usr/bin/env python3
"""ncmb.py (12.11.2026, queue-ar/dsmain/nc, ds-main-port NC-profile builder) -- mb.py's c<N> at a fixed temperature WITH
per-request records and /stats at the window's edges, for dsm-ncprof.sh. = stack4/mbt.py (md5 of the copy in the item
header) + three read-only taps; mb.py (~/gpu/ds41/mb.py) is loaded READ ONLY (importlib, no bytecode) and nothing of it
changes.

  phase  (START U: "exactly as stack4") mb.phase_c(N, a) itself, as mbt.py runs it (stream() given {"temperature": T}),
         and, like stack4's s4blocks.py phase(): GET /stats + /admin/fn4 before and after, and a 2 s sampler thread
         (GET /stats, then GET /admin/fn4, every 2.0 s) while it runs. Taps: every request's send / end (t0, ttft, t_end,
         prompt, cached, n, the closed-loop worker from the nonce line); mb.metrics() also saves the full /stats right
         after each of its two calls (the window's edges, marks w0 / w1); the sampler's /stats go to --samples (compact).
         Prints mb.py's "MB {json}" line (+ temperature), as mbt.py does.
  win    (START P, profiled) mb.closed_loop(N, ..., --gen G) with the same request builder, --warm s, then the window
         (--dur - --warm s) between two marks (mb.metrics() + full /stats at each), then a hard exit: the open streams
         close, the server cancels them (radiance:client_disconnect_total) -- no drain of long requests. Prints an
         "MB {json}" line of the window (mb.mdelta over the two metrics, requests started in the window, TTFT p50).

  ncmb.py --mb ~/gpu/ds41/mb.py --mode phase|win --temperature 0 --n 8 --tag T --dur 60 --warm 15 [--gen 512] [--ctx 1000]
          --reqs REQS.jsonl --stats-prefix PFX [--samples SAMP.jsonl]
  ncmb.py --mb ~/gpu/ds41/mb.py --selftest     offline: mb.py loads, has what this wrapper calls, stream() takes extra.

REQS.jsonl lines: {"ev": "send"|"done"|"mark", ...}; marks {"ev": "mark", "k": 0|1, "t"} (/stats files PFX-w0.json,
PFX-w1.json). Standard library only.
"""
import argparse, importlib.util, inspect, json, os, statistics, sys, threading, time, urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def http(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


class Taps:
    def __init__(self, mb, a):
        self.mb, self.a, self.lock, self.rows, self.marks = mb, a, threading.Lock(), [], 0
        self.compact = load(os.path.join(HERE, "ncone.py"), "ncone").compact

    def add(self, d):
        with self.lock: self.rows.append(d)

    def stream(self, orig, t):
        def stream(msgs, max_tokens, effort="medium", stamps=None, extra=None):
            e = dict(extra or {})
            if t is not None: e["temperature"] = t
            q = (msgs[-1].get("content") or "") if msgs else ""
            w = None
            if q.startswith("# req "):
                try: w = int(q.split("\n", 1)[0].split()[3])
                except Exception: w = None
            ts = time.time()
            self.add({"ev": "send", "t": ts, "w": w, "max_tokens": max_tokens})
            try:
                r = orig(msgs, max_tokens, effort=effort, stamps=stamps, extra=e)
            except Exception as x:
                self.add({"ev": "done", "t_send": ts, "w": w, "err": str(x)[:200], "t_end": time.time()})
                raise
            self.add({"ev": "done", "t_send": ts, "w": w, "t0": r.get("t0"), "ttft": r.get("ttft"), "t_end": time.time(),
                      "prompt": r.get("prompt"), "cached": r.get("cached"), "n": r.get("n"), "err": None})
            return r
        return stream

    def metrics(self, orig):
        def metrics():
            t = time.time()
            m = orig()
            k = self.marks
            self.marks += 1
            try:
                st = http(self.mb.URL + "/stats", 30)
                with open("%s-w%d.json" % (self.a.stats_prefix, k), "w") as f: json.dump(st, f)
            except Exception as e:
                st = {"error": str(e)[:200]}
            self.add({"ev": "mark", "k": k, "t": t, "t_stats": time.time()})
            return m
        return metrics

    def dump(self):
        with self.lock, open(self.a.reqs, "a") as f:
            for d in self.rows: f.write(json.dumps(dict(d, tag=self.a.tag)) + "\n")


class Sampler:
    """s4blocks.py's 2 s sampler (GET /stats, then GET /admin/fn4), its /stats kept (compact) with the time."""
    def __init__(self, url, path, compact, every=2.0):
        self.url, self.path, self.compact, self.every, self.out = url, path, compact, every, []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(self.every):
            t = time.time()
            try: self.out.append({"t": t, "s": self.compact(http(self.url + "/stats", 10))})
            except Exception: pass
            try: http(self.url + "/admin/fn4", 10)
            except Exception: pass

    def __enter__(self):
        self.t.start(); return self

    def __exit__(self, *e):
        self.stop.set(); self.t.join(timeout=15)
        if self.path:
            with open(self.path, "a") as f:
                for d in self.out: f.write(json.dumps(d) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True)
    ap.add_argument("--mode", choices=("phase", "win"), default="phase")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--warm", type=float, default=15)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--reqs")
    ap.add_argument("--stats-prefix")
    ap.add_argument("--samples")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb, "mb")
    need = ("stream", "phase_c", "closed_loop", "metrics", "mdelta", "URL", "text", "TASKS")
    miss = [n for n in need if not hasattr(mb, n)]
    if a.selftest:
        ok = not miss and "extra" in inspect.signature(mb.stream).parameters \
            and list(inspect.signature(mb.closed_loop).parameters)[:5] == ["n", "dur", "ctx", "gen", "stop_evt"]
        tp = Taps(mb, a)
        got = []
        st = tp.stream(lambda msgs, mt, effort="medium", stamps=None, extra=None: got.append(extra) or
                       {"t0": 1.0, "ttft": 0.5, "prompt": 1099, "cached": 0, "n": mt}, 0.0)
        r = st([{"role": "user", "content": "# req 123 5\nabc"}], 7)
        ok = ok and got == [{"temperature": 0.0}] and r["n"] == 7 and tp.rows[0]["w"] == 5 and tp.rows[1]["prompt"] == 1099
        print("SELFTEST " + ("OK: mb.py %s loads, %s present, stream() takes extra, closed_loop(n, dur, ctx, gen, stop_evt), taps record" % (
            a.mb, ", ".join(need)) if ok else "FAILED: missing %s or a tap / signature differs" % miss))
        return 0 if ok else 1
    if miss or a.temperature is None or not a.reqs or not a.stats_prefix:
        print("MB " + json.dumps({"error": "mb.py lacks %s or no --temperature / --reqs / --stats-prefix" % miss}), flush=True)
        return 1
    mb.MODEL = http(mb.URL + "/v1/models", 60)["data"][0]["id"]
    tp = Taps(mb, a)
    mb.stream = tp.stream(mb.stream, a.temperature)
    orig_metrics = mb.metrics
    if a.mode == "phase":
        mb.metrics = tp.metrics(orig_metrics)
        try: http(mb.URL + "/stats", 30); http(mb.URL + "/admin/fn4", 10)          # s4blocks.py phase(): snapx + draftc + samplec
        except Exception: pass
        with Sampler(mb.URL, a.samples, tp.compact):
            res = mb.phase_c(a.n, a)
        try: http(mb.URL + "/admin/fn4", 10); http(mb.URL + "/stats", 30)
        except Exception: pass
        tp.dump()
        print("MB " + json.dumps(dict(tag=a.tag, temperature=a.temperature, mode="phase", **res)), flush=True)
        return 0
    # win: the closed loop, two marks, a hard exit (the server cancels the open streams)
    stop, recs = threading.Event(), []
    mb.closed_loop(a.n, a.dur, a.ctx, a.gen, stop, recs)
    time.sleep(a.warm)
    mark = tp.metrics(orig_metrics)
    b = mark(); tw0 = time.time()
    time.sleep(a.dur - a.warm)
    m = mark(); tw1 = time.time()
    stop.set()
    d = mb.mdelta(b, m)
    win = [r for r in recs if not r.get("err") and tw0 <= r["t0"] < tw1]
    tt = [r["ttft"] for r in win if r.get("ttft")]
    sends = sum(1 for r in tp.rows if r["ev"] == "send" and tw0 <= r["t"] < tw1)
    tp.dump()
    print("MB " + json.dumps(dict(tag=a.tag, temperature=a.temperature, mode="win", phase="c%d" % a.n, gen=a.gen,
                                  window_s=round(tw1 - tw0, 1), agg_dec_tps=round(d["dec_tok"] / (tw1 - tw0), 1),
                                  sends_in_window=sends, done_in_window=len(win),
                                  ttft_p50=round(statistics.median(tt), 2) if tt else None, **d)), flush=True)
    sys.stdout.flush()
    os._exit(0)   # the daemon client threads die with their sockets: the server cancels their streams


if __name__ == "__main__":
    sys.exit(main())
