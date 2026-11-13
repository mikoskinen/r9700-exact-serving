#!/usr/bin/env python3
"""ncmb3.py (12.11.2026, queue-ar/dsmain/harness, the int11 builder; = ncmb2.py e2e82adb around mbt3.py instead of mbt2.py: the FIXED
PROMPT SET reset per block -- --prompt-set fixed|nonce (default fixed), --reset-per-block 1|0 (default 1), passed to mbt3.phase_fixed;
TESTING.md 12.11. "c4 METHOD FIX"; ncmb2.py is not changed). The rest is ncmb2.py's:
ncmb2.py (12.11.2026, queue-ar/dsmain/harness, ds-main-port harness builder) -- nc/ncmb.py's c<N> block (per-request records,
/stats at the window's edges, the 2 s /stats + /admin/fn4 sampler) on mbt2.py's FIXED, SEEDED ARRIVAL SCHEDULE (c8 block-mix
note 12.11.: the old closed loop's newcomer co-batch mix depends on block order, not on the arm). nc/ncmb.py is not changed;
this is a copy of its taps and sampler around mbt2.py's loop (loaded from this directory) and mb.py (loaded READ ONLY).

  phase  (= ncmb.py --mode phase) GET /stats + /admin/fn4 before and after, the 2 s sampler while it runs, mbt2.phase_fixed
         (or mbt2.phase_mb with --sched mb = ncmb.py's behaviour), drain at the end; prints "MB {json}" (mode "phase").
  win    (= ncmb.py --mode win) the same loop and window, then a hard exit right after the second mark (the open streams close,
         the server cancels them); "MB {json}" (mode "win").
  /stats at the two marks -> PFX-w0.json / PFX-w1.json (ncsteps.py c8 --sa / --sb); REQS rows = mbt2.py's (send / done /
  mark, + k, j = group, t_first, max_tokens), which ncsteps.py c8 and c8mix.py read.

  ncmb2.py --mb ~/gpu/ds41/mb.py --mode phase|win --temperature 0 --n 8 --tag T --reqs R.jsonl --stats-prefix PFX
           [--samples S.jsonl] [--sched fixed|mb] [--pattern 1x8] [--seed 1] [--window gens|time] [--gens 2] [--edge 0.5]
           [--dur 60 --warm 15] [--gen 512] [--ctx 1000] [--ignore-eos 1] [--release-delay 0.1] [--url URL]
  ncmb2.py --mb ~/gpu/ds41/mb.py --selftest
Standard library only.
"""
import argparse, importlib.util, inspect, json, os, sys, threading, time, urllib.request

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


def compact_loader(path):
    try:
        return load(path, "ncone").compact
    except Exception:
        return lambda d: d


class Sampler:
    """ncmb.py's / s4blocks.py's 2 s sampler (GET /stats, then GET /admin/fn4), its /stats kept (compact) with the time."""
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
    ap.add_argument("--mbt2", default=os.path.join(HERE, "mbt3.py"))   # mbt3.py (the name kept: the same loader)
    ap.add_argument("--ncone", default=os.path.join(HERE, "..", "nc", "ncone.py"))
    ap.add_argument("--url", default=None)
    ap.add_argument("--mode", choices=("phase", "win"), default="phase")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sched", choices=("fixed", "mb"), default="fixed")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spacing", type=int, default=None)
    ap.add_argument("--len-jitter", type=int, default=0)
    ap.add_argument("--ignore-eos", type=int, default=1)
    ap.add_argument("--release-delay", type=float, default=0.1)
    ap.add_argument("--window", choices=("gens", "time"), default="gens")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--edge", type=float, default=0.5)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--warm", type=float, default=15)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--reqs")
    ap.add_argument("--stats-prefix")
    ap.add_argument("--samples")
    ap.add_argument("--prompt-set", choices=("fixed", "nonce"), default="fixed")
    ap.add_argument("--reset-per-block", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb, "mb")
    m2 = load(a.mbt2, "mbt2")
    miss = [n for n in m2.NEED if not hasattr(mb, n)]
    if a.selftest:
        ok = not miss and "stamps" in inspect.signature(mb.stream).parameters and "extra" in inspect.signature(mb.stream).parameters
        ok = ok and all(hasattr(m2, x) for x in ("phase_fixed", "phase_mb", "Recorder", "make_plan", "cobatch"))
        ok = ok and "drain" in inspect.signature(m2.phase_fixed).parameters and all(hasattr(m2, x) for x in ("prompt_of", "prompt_seq", "reset_cache"))
        c = compact_loader(a.ncone)
        print("SELFTEST " + ("OK: mb.py %s + mbt3.py %s load, phase_fixed(drain) / phase_mb / Recorder / prompt_of / prompt_seq / reset_cache present, --prompt-set %s --reset-per-block %s, ncone.compact %s" % (
            a.mb, a.mbt2, a.prompt_set, a.reset_per_block, "loaded" if c.__name__ == "compact" else "NOT found (identity)") if ok else "FAILED: missing %s" % miss))
        return 0 if ok else 1
    if miss or a.temperature is None or not a.reqs or not a.stats_prefix:
        print("MB " + json.dumps({"error": "mb.py lacks %s or no --temperature / --reqs / --stats-prefix" % miss}), flush=True)
        return 1
    if a.url: mb.URL = a.url.rstrip("/")
    mb.MODEL = http(mb.URL + "/v1/models", 60)["data"][0]["id"]
    compact = compact_loader(a.ncone)
    orig_stream, orig_metrics, t = mb.stream, mb.metrics, a.temperature
    rec = m2.Recorder()

    def stream(msgs, max_tokens, effort="medium", stamps=None, extra=None):
        e = dict(extra or {}); e["temperature"] = t
        if a.sched == "mb" and stamps is None:
            st, ts = m2.Stamps(), time.time()
            r = orig_stream(msgs, max_tokens, effort=effort, stamps=st, extra=e)
            rec.add({"ev": "done", "t_send": ts, "t_first": st.first, "n": r.get("n"), "max_tokens": max_tokens, "prompt": r.get("prompt"),
                     "cached": r.get("cached"), "t0": r.get("t0"), "ttft": r.get("ttft"), "t_end": time.time(), "err": None,
                     "w": int(msgs[-1]["content"].split("\n", 1)[0].split()[3]) if msgs[-1]["content"].startswith("# req ") else None})
            return r
        return orig_stream(msgs, max_tokens, effort=effort, stamps=stamps, extra=e)

    marks = [0]

    def metrics():                     # ncmb.py's tap: the full /stats right after each window-edge /metrics read
        m = orig_metrics()
        k = marks[0]; marks[0] += 1
        try:
            with open("%s-w%d.json" % (a.stats_prefix, k), "w") as f: json.dump(http(mb.URL + "/stats", 30), f)
        except Exception:
            pass
        return m

    mb.stream, mb.metrics = stream, metrics
    if a.mode == "phase":
        try: http(mb.URL + "/stats", 30); http(mb.URL + "/admin/fn4", 10)
        except Exception: pass
        with Sampler(mb.URL, a.samples, compact):
            res = m2.phase_fixed(mb, a.n, a, rec) if a.sched == "fixed" else m2.phase_mb(mb, a.n, a, rec)
        try: http(mb.URL + "/admin/fn4", 10); http(mb.URL + "/stats", 30)
        except Exception: pass
        rec.dump(a.reqs, a.tag)
        print("MB " + json.dumps(dict(tag=a.tag, temperature=t, mode="phase", **res)), flush=True)
        return 0
    if a.sched != "fixed":
        print("MB " + json.dumps({"error": "--mode win runs the fixed schedule only (ncmb.py --mode win is the old loop)"}), flush=True)
        return 1
    with Sampler(mb.URL, a.samples, compact):
        res = m2.phase_fixed(mb, a.n, a, rec, drain=False)
    rec.dump(a.reqs, a.tag)
    print("MB " + json.dumps(dict(tag=a.tag, temperature=t, mode="win", gen=a.gen, **res)), flush=True)
    sys.stdout.flush()
    os._exit(0)   # the daemon client threads die with their sockets: the server cancels their streams


if __name__ == "__main__":
    sys.exit(main())
