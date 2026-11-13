#!/usr/bin/env python3
"""dlblocks.py (10.11.2026, queue-ar dsmain, ds-main-port VRAM agent; item 57-p4zyzzzzz-ds-hczzzz-dlend, body dsm-dlend.sh) --
vsblocks.py (ggblocksdz2.py + the /stats counters a phase; both loaded read only from beside this file) plus lever DL:
  * READY before every block's first phase: a 4-token request applies the block's POST (switches change at a step release);
    when the block's draft_lend is 0 (arms A / C) it then polls GET /admin/fn4 (a 4-token request between polls, at most
    --ready-s seconds) until lend.state == 0 and lend.draft_resident == lend.draft_units -- the drafter's slab taken back
    after a lent block. The wait and its outcome go into res["c1"]["xready"] and a "VDLR" line; a block that never got
    ready is marked there (the summary leaves it out), it is not hidden.
  * per phase: GET /admin/fn4 "lend" before / after (res[phase]["xlend"]: every counter's delta, the gauges' end values)
    and a sampler every --sample-s seconds during the phase (lent share = samples at state 2, transit share = state 1,
    lent_units max / mean, draft_resident min). One "VDL {json}" line a phase.
Arguments: ggblocksdz2.py's, plus --ready-s (90) and --sample-s (2). Standard library only."""
import importlib.util, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vsblocks", os.path.join(HERE, "vsblocks.py"))
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)
gg = vs.gg

GAUGES = ("state", "lent_units", "draft_resident", "draft_units", "slab_mib", "backoff_k", "probe_interval")
OPT = {"ready_s": 90.0, "sample_s": 2.0}
for flag, key in (("--ready-s", "ready_s"), ("--sample-s", "sample_s")):   # ours; ggblocksdz2's argparse must not see them
    if flag in sys.argv:
        i = sys.argv.index(flag)
        OPT[key] = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]


def fn4(url):
    try: return gg.http(url + "/admin/fn4", timeout=10)
    except Exception: return {}


def lend(s): return (s or {}).get("lend") or {}


def tiny(url):
    try: gg.http(url + "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0}, timeout=120)
    except Exception: pass


def ready(a):
    t0 = time.time()
    tiny(a.url)
    s = fn4(a.url)
    want_back = (s.get("switches") or {}).get("draft_lend") == 0
    polls, ok = 0, True
    while True:
        s = fn4(a.url); L = lend(s)
        if not s.get("pending") and (not want_back or (L.get("state") == 0 and L.get("draft_resident") == L.get("draft_units"))):
            break
        if time.time() - t0 > OPT["ready_s"]:
            ok = False
            break
        polls += 1
        tiny(a.url)
        time.sleep(0.5)
    L = lend(s)
    r = {"ok": ok, "wait_s": round(time.time() - t0, 2), "polls": polls, "reclaim_wanted": want_back,
         "draft_lend": (s.get("switches") or {}).get("draft_lend"), "state": L.get("state"),
         "draft_resident": L.get("draft_resident"), "draft_units": L.get("draft_units")}
    print("VDLR " + json.dumps(dict({"label": a.label, "block": vs._block["i"] + 1}, **r)), flush=True)
    return r


class Sampler(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.stop, self.rows = url, threading.Event(), []

    def run(self):
        while not self.stop.wait(OPT["sample_s"]):
            L = lend(fn4(self.url))
            if L: self.rows.append(L)

    def result(self):
        n = len(self.rows)
        if not n: return {"samples": 0}
        lu = [x.get("lent_units") or 0 for x in self.rows]
        return {"samples": n, "lent_share": round(sum(1 for x in self.rows if x.get("state") == 2) / n, 3),
                "transit_share": round(sum(1 for x in self.rows if x.get("state") == 1) / n, 3),
                "lent_units_max": max(lu), "lent_units_mean": round(sum(lu) / n, 1),
                "draft_resident_min": min(x.get("draft_resident") or 0 for x in self.rows)}


_vsphase = vs.phase


def phase(a, dph, name, f):
    rd = ready(a) if name == "c1" else None
    l0 = lend(fn4(a.url))
    smp = Sampler(a.url); smp.start()
    try:
        r = _vsphase(a, dph, name, f)
    finally:
        smp.stop.set(); smp.join(timeout=10)
    l1 = lend(fn4(a.url))
    x = {k: (l1.get(k) if k in GAUGES else (l1.get(k) or 0) - (l0.get(k) or 0)) for k in l1 if k != "capable"}
    x.update({"capable": l1.get("capable")}, **{"s_" + k: v for k, v in smp.result().items()})
    if isinstance(r, dict):
        r["xlend"] = x
        if rd is not None: r["xready"] = rd
    print("VDL " + json.dumps(dict({"label": a.label, "block": vs._block["i"], "phase": name}, **x)), flush=True)
    return r


gg.phase = phase

if __name__ == "__main__":
    sys.exit(gg.main())
