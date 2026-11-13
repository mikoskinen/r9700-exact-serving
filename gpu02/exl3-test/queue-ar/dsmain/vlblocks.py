#!/usr/bin/env python3
"""vlblocks.py (10.11.2026, queue-ar dsmain, ds-main-port VRAM agent; item 57-p4zyzzzzz-ds-hxz-vramstack, body
dsm-vramstack.sh) -- in-process blocks with the lessons of xdl (runs/xdsm/dlend): ggblocksdz2.py's phases (its stepprobe /
essaypos / editprobe / mb helpers, loaded read only through vsblocks.py) in a loop of our own:

  * --rearm: every block starts with draft_depth0 0 applied at a step (POST + a 4-token request, as d0attr did), then the
    block's set (--fixed with draft_depth0 1 + the arm + epoch_bump) -- the depth-0 state does not carry from block to block
    (xdl: the first c4 after start held depth 0 for 1416 cycles, later ones 139-308);
  * --settle ARM: one settle block of ARM first (--settle-phases, --settle-dur), marked "settle": the cost classes learn
    there, not in a measured block; summaries leave it out;
  * READY: after the block's POST a 4-token request applies it; with draft_lend 0 (a GET with a "lend" object) a poll
    (requests between polls, <= --ready-s) until lend.state 0 and draft_resident == draft_units (the slab back after a
    lending block); the outcome in res["c1"]["xready"] (a block never ready is marked, the summary leaves it out);
  * per phase: /stats deltas (vsblocks.delta: VRAM units, rank-0 hit, missed a step, stream MB a step), the draft counters
    (zero / probe cycles, zero moves; ggblocksdz2's "draftph"), the lend counters (deltas, gauges at the end) and a
    --sample-s sampler: lent / transit share, lent_units max / mean, draft_resident min, the time to the first lent sample
    and to 95 % of the phase's lent maximum (the fill rate), and /stats VRAM units max every second sample.
  --phases (default c1,essaypos,edit,c4,c8) sets the phase order. Rows: ggblocksdz2's schema (+ settle, rearm, xstats,
  xlend, xready). One "BLOCK" line a block; "VSX" / "VDL" lines a phase. Standard library only."""
import argparse, importlib.util, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vsblocks", os.path.join(HERE, "vsblocks.py"))
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)
gg = vs.gg
GAUGES = ("state", "lent_units", "draft_resident", "draft_units", "slab_mib", "backoff_k", "probe_interval")


def fn4(url):
    try: return gg.http(url + "/admin/fn4", timeout=10)
    except Exception: return {}


def post(url, body):
    try:
        gg.http(url + "/admin/fn4", body); return True
    except Exception as e: return str(e)[:200]


def tiny(url):
    try: gg.http(url + "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0}, timeout=120); return True
    except Exception: return False


def rearm(url):
    p = post(url, {"draft_depth0": 0}); tiny(url); s = fn4(url)
    return p is True and (s.get("switches") or {}).get("draft_depth0") == 0 and not s.get("pending")


def ready(url, ready_s):
    t0 = time.time()
    tiny(url)
    s = fn4(url)
    back = "lend" in s and (s.get("switches") or {}).get("draft_lend") == 0
    polls, ok = 0, True
    while True:
        s = fn4(url); L = s.get("lend") or {}
        if not s.get("pending") and (not back or (L.get("state") == 0 and L.get("draft_resident") == L.get("draft_units"))):
            break
        if time.time() - t0 > ready_s:
            ok = False; break
        polls += 1; tiny(url); time.sleep(0.5)
    L = s.get("lend") or {}
    return {"ok": ok, "wait_s": round(time.time() - t0, 2), "polls": polls, "reclaim_wanted": back,
            "draft_lend": (s.get("switches") or {}).get("draft_lend"), "state": L.get("state"),
            "draft_resident": L.get("draft_resident"), "draft_units": L.get("draft_units")}


class Sampler(threading.Thread):
    def __init__(self, url, every):
        super().__init__(daemon=True)
        self.url, self.every, self.stop, self.rows, self.t0 = url, every, threading.Event(), [], time.time()

    def run(self):
        k = 0
        while not self.stop.wait(self.every):
            L = fn4(self.url).get("lend") or {}
            u = vs.units(vs.snap(self.url)) if k % 2 == 0 else None
            self.rows.append({"t": round(time.time() - self.t0, 1), "state": L.get("state"), "lent": L.get("lent_units"),
                              "res": L.get("draft_resident"), "u": u})
            k += 1

    def result(self):
        rs = self.rows
        o = {"samples": len(rs)}
        us = [r["u"] for r in rs if r["u"] is not None]
        if us: o.update(vram_units_max=max(us), vram_units_min=min(us))
        L = [r for r in rs if r["state"] is not None]
        if not L: return o
        lu = [r["lent"] or 0 for r in L]; mx = max(lu)
        o.update(lent_share=round(sum(1 for r in L if r["state"] == 2) / len(L), 3),
                 transit_share=round(sum(1 for r in L if r["state"] == 1) / len(L), 3),
                 lent_units_max=mx, lent_units_mean=round(sum(lu) / len(L), 1),
                 draft_resident_min=min(r["res"] or 0 for r in L),
                 t_first_lent=next((r["t"] for r in L if r["state"] == 2), None),
                 t_to_95pct_max=next((r["t"] for r in L if mx and (r["lent"] or 0) >= 0.95 * mx), None),
                 series=[[r["t"], r["state"], r["lent"]] for r in L][:120])
        return o


def run_phase(a, dph, name, f, label, blk):
    s0 = vs.snap(a.url); l0 = fn4(a.url).get("lend") or {}; d0 = gg.draftc(a)
    smp = Sampler(a.url, a.sample_s); smp.start()
    try: r = f()
    finally:
        smp.stop.set(); smp.join(timeout=10)
    s1 = vs.snap(a.url); l1 = fn4(a.url).get("lend") or {}; d1 = gg.draftc(a)
    dph[name] = {k: (d1.get(k) or 0) - (d0.get(k) or 0) for k in d1}
    x = vs.delta(s0, s1)
    xl = {k: (l1.get(k) if k in GAUGES else (l1.get(k) or 0) - (l0.get(k) or 0)) for k in l1 if k != "capable"}
    if l1: xl["capable"] = l1.get("capable")
    xl.update({"s_" + k: v for k, v in smp.result().items()})
    if isinstance(r, dict):
        r["xstats"] = x; r["xlend"] = xl
    print("VSX " + json.dumps(dict({"label": label, "block": blk, "phase": name}, **x)), flush=True)
    print("VDL " + json.dumps(dict({"label": label, "block": blk, "phase": name}, **{k: v for k, v in xl.items() if k != "s_series"})), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    for k, d in (("--url", "http://127.0.0.1:8090"), ("--pair", None), ("--out", None), ("--settle", None)):
        ap.add_argument(k, default=d)
    ap.add_argument("--mb-dir", required=True); ap.add_argument("--edit", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True); ap.add_argument("--order", required=True)
    ap.add_argument("--dur", type=int, default=75); ap.add_argument("--warm", type=int, default=10); ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}"); ap.add_argument("--phases", default="c1,essaypos,edit,c4,c8")
    ap.add_argument("--rearm", action="store_true"); ap.add_argument("--settle-phases", default="c1,c4,c8")
    ap.add_argument("--settle-dur", type=int, default=30); ap.add_argument("--ready-s", type=float, default=90.0)
    ap.add_argument("--sample-s", type=float, default=2.0)
    a = ap.parse_args()
    fixed = json.loads(a.fixed); arms = {}
    for x in a.arm:
        k, v = x.split("=", 1); arms[k] = json.loads(v)
    plan = ([(a.settle, True)] if a.settle else []) + [(c, False) for c in a.order]
    if any(c not in arms for c, _ in plan):
        print("GGB " + json.dumps({"error": "order / settle names an arm not given (%s)" % list(arms)})); return 1
    t_start, rows = time.time(), []
    for i, (arm, settle) in enumerate(plan):
        want = dict(fixed, **arms[arm])
        rd_ok = rearm(a.url) if a.rearm else None
        e0 = fn4(a.url).get("epoch")
        posted = post(a.url, dict(want, epoch_bump=1))
        rd = ready(a.url, a.ready_s)
        dur, phases = (a.settle_dur, a.settle_phases) if settle else (a.dur, a.phases)
        a2 = argparse.Namespace(**dict(vars(a), dur=dur))
        tag = "%s-%s%d%s" % (a.label, "s" if settle else "b", i, arm)
        fs = {"c1": lambda: gg.stepprobe(a2, "essay,code"), "essaypos": lambda: gg.essaypos(a2), "edit": lambda: gg.editprobe(a2),
              "c4": lambda: gg.mb(a2, "c4", tag + "-c4"), "c8": lambda: gg.mb(a2, "c8", tag + "-c8")}
        dph, res = {}, {}
        for ph in phases.split(","):
            res[ph] = run_phase(a2, dph, ph, fs[ph], a.label, i)
        res["draftph"] = dph
        if isinstance(res.get("c1"), dict): res["c1"]["xready"] = rd
        elif res: res[next(iter(res))]["xready"] = rd
        st = fn4(a.url); sw = st.get("switches", {})
        ok = posted is True and rd["ok"] and (rd_ok is not False) and all(sw.get(k) == v for k, v in want.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in ("c4", "c8"))
        row = {"label": a.label, "block": i, "arm": arm, "settle": settle, "rearm": rd_ok, "values": arms[arm], "fixed": fixed,
               "posted": posted, "applied": sw, "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok,
               "res": res, "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        c1 = res.get("c1") or {}; ed = res.get("edit") or {}; c4 = res.get("c4") or {}; c8 = res.get("c8") or {}
        lx = lambda p: (res.get(p) or {}).get("xlend") or {}
        print("BLOCK %d %s%s %s ok=%s rearm=%s ready %s s | draft %s | essay %s tok/s acc %s | code %s tok/s | edit %s tok/s | c4 %s tok/s p50 %s %s ms, lent %s max %s | c8 %s tok/s p50 %s %s ms, lent %s max %s (first lent %s s, 95 %% at %s s)" % (
            i, arm, " (settle)" if settle else "", json.dumps(arms[arm]), ok, rd_ok, rd["wait_s"],
            " / ".join("%s %s" % (k, gg.dfmt(v)) for k, v in dph.items()) or "-",
            (c1.get("essay") or {}).get("decode_tps"), (c1.get("essay") or {}).get("acceptance"), (c1.get("code") or {}).get("decode_tps"),
            ed.get("decode_tps"), c4.get("agg_dec_tps"), c4.get("per_stream_dec_p50"), c4.get("ms_decode_step"),
            lx("c4").get("s_lent_share"), lx("c4").get("s_lent_units_max"), c8.get("agg_dec_tps"), c8.get("per_stream_dec_p50"),
            c8.get("ms_decode_step"), lx("c8").get("s_lent_share"), lx("c8").get("s_lent_units_max"),
            lx("c8").get("s_t_first_lent"), lx("c8").get("s_t_to_95pct_max")), flush=True)
    bad = sum(1 for r in rows if not r["ok"])
    print("GGB " + json.dumps({"label": a.label, "blocks": len(rows), "blocks_bad": bad, "wall_s": round(time.time() - t_start, 1)}))
    return 0 if bad < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
