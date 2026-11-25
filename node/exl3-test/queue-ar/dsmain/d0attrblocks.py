#!/usr/bin/env python3
"""d0attrblocks.py (10.11.2026, queue-ar dsmain, ds-main-port; item 57-p4zyzzzzz-ds-hczz-d0attr's driver) -- in-process
blocks over the depth-0 probe interval (fn4 draft_probe) on an NST 5 DSpark server under the adaptive window with depth 0
on (draft_depth0 1), and single instrumented windows. Standard library only; reuses stepprobe.py / mb.py (read-only).

  d0attrblocks.py blocks --url URL --mb-dir DIR --label L --a 16 --b 4096 --order ABBA [--dur 75] [--warm 10] [--gen 512]
                         [--fixed '{"gg_gemv": 2, "draft_adaptive": 1, "draft_hybrid": 0}'] [--stop FILE] [--out FILE.jsonl]
      A block: draft_depth0 0 (POST, then a 4-token request so it applies at a step: depth 0 off ends any probe and puts
      D back at the drafting depth, the measurements kept), then POST {fixed, draft_depth0 1, draft_probe A|B,
      epoch_bump 1} (depth 0 on again with its hold spent, sched/draft.h set_depth0: the first measured cycle decides at
      once, so every block starts from the same state); then the phases
        essay  stepprobe.py essay GEN (1 user, greedy) between two /stats snapshots: acceptance by draft position
        c4     mb.py --phases c4 --dur DUR --warm WARM, /stats and the draft counters snapshotted at the window's ends
        c8     the same at 8 streams
      and per phase: the draft counters (GET /admin/fn4 "draft": cycles at depth 0, probe cycles, moves) and /stats deltas
      (decode steps, ms a decode step, routed-expert units in VRAM, rank-0 hit %, missed units and host-stream MB a step),
      the share of decode steps at depth 0 and in probes. A block whose switches are not as set when it ends, with a change
      pending, whose epoch did not move, whose POST was refused or whose mb phase failed is flagged (ok false).
      Prints one "BLOCK" line a block and "D0AB {json}" (per metric A / B means, pooled B/A, pairs, t95).
  d0attrblocks.py window --url URL --mb-dir DIR --label L --phase c8 [--dur 40] [--warm 10] [--set JSON] [--reset]
                         [--mark-cmd CMD] [--out FILE.jsonl]
      One mb phase with /stats + draft counters at the window's ends and their CLOCK_MONOTONIC ns (the stamps' clock), and
      CMD's output there (e.g. the followed engine log's line count, for --profile-ops windows). --set: POST it first
      (--reset: draft_depth0 0 + a 4-token request before it). Prints "WIN {json}"."""
import argparse, json, math, os, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def mono():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def fn4(a):
    try: return http(a.url + "/admin/fn4")
    except Exception: return {}


def post(a, body):
    try:
        http(a.url + "/admin/fn4", body); return True
    except Exception as e:   # a refused body (400) queues nothing
        return str(e)[:200]


def nudge(a):
    try: http(a.url + "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0}, timeout=120); return True
    except Exception: return False


def reset_depth0(a):
    """draft_depth0 0 applied at a step (a 4-token request), then True if GET reports 0 and nothing pending."""
    p = post(a, {"draft_depth0": 0})
    nudge(a)
    s = fn4(a)
    return p is True and (s.get("switches") or {}).get("draft_depth0") == 0 and not s.get("pending")


def snap(a, mark=None):
    t = mono()
    try: st = http(a.url + "/stats")
    except Exception: st = {}
    s = fn4(a)
    m = None
    if mark:
        try: m = subprocess.run(mark, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e: m = "error %s" % str(e)[:80]
    return {"t": t, "stats": st, "draft": s.get("draft") or {}, "switches": s.get("switches") or {}, "mark": m}


def tiers(st):
    return {x.get("name"): x.get("units") for x in (st.get("experts") or {}).get("tiers", [])}


def sdelta(s0, s1):
    a, b = s0["stats"], s1["stats"]
    if not a or not b: return {"error": "no /stats"}
    ea, eb = a.get("experts", {}), b.get("experts", {}); la, lb = a.get("link", {}), b.get("link", {})
    ds = b.get("steps", 0) - a.get("steps", 0)
    dd = b.get("steps_decode", 0) - a.get("steps_decode", 0)
    dpm = (b.get("steps_prefill", 0) + b.get("steps_mixed", 0)) - (a.get("steps_prefill", 0) + a.get("steps_mixed", 0))
    dr = eb.get("routed", 0) - ea.get("routed", 0); drr = eb.get("routed_res", 0) - ea.get("routed_res", 0)
    dz = {k: (s1["draft"].get(k) or 0) - (s0["draft"].get(k) or 0) for k in s1["draft"]}
    n = max(1, ds)
    o = {"steps": ds, "decode_steps": dd, "prefill_mixed_steps": dpm, "secs": round((s1["t"] - s0["t"]) / 1e9, 2),
         "ms_step": round((b.get("step_ns", 0) - a.get("step_ns", 0)) / n / 1e6, 2) if ds > 0 else None,
         "ms_decode_step": round((b.get("step_ns_decode", 0) - a.get("step_ns_decode", 0)) / max(1, dd) / 1e6, 2) if dd > 0 else None,
         "hit_pct_rank0": round(100.0 * drr / dr, 2) if dr else None,
         "routed_step_rank0": round(dr / n, 1), "missed_step_rank0": round((dr - drr) / n, 2),
         "promotions_step": round((eb.get("promotions", 0) - ea.get("promotions", 0)) / n, 2),
         "stream_MB_step": round((lb.get("stream_bytes", 0) - la.get("stream_bytes", 0)) / n / 1e6, 1),
         "h2d_MB_step": round((lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)) / n / 1e6, 1),
         "vram_units_end": tiers(b).get("vram"), "pinned_units_end": tiers(b).get("pinned"),
         "passes_step": {k: round(((b.get("passes") or {}).get(k, 0) - (a.get("passes") or {}).get(k, 0)) / n, 2)
                         for k in ("played", "recorded", "issued_unkeyed", "issued_first", "dispatched")},
         "draft": dz,
         "zero_share": round(dz.get("zero_cycles", 0) / dd, 3) if dd > 0 else None,
         "probe_share": round(dz.get("probe_cycles", 0) / dd, 3) if dd > 0 else None}
    return o


def jlines(out, prefix="J "):
    rows = []
    for l in out.splitlines():
        if l.startswith(prefix):
            try: rows.append(json.loads(l[len(prefix):]))
            except ValueError: pass
    return rows


def essay(a):
    s0 = snap(a)
    p = subprocess.run([sys.executable, "stepprobe.py", "essay", str(a.gen)], cwd=a.mb_dir, capture_output=True, text=True,
                       timeout=1500)
    s1 = snap(a)
    r = next(iter(jlines(p.stdout)), {"error": (p.stdout[-200:] + p.stderr[-200:]).strip()})
    pa, pb = s0["stats"].get("draft_positions") or [], s1["stats"].get("draft_positions") or []
    pos = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        pos.append({"p": k + 1, "reached": re_, "accepted": ac, "rate": (ac / re_) if re_ else None})
    return {"probe": r, "positions": pos, "win": sdelta(s0, s1)}


def mbphase(a, phase, tag, mark=None):
    """mb.py PHASE with /stats + counters at its window's ends ([warm, dur] after its start) and around it."""
    pre = snap(a)
    t_start = time.time()
    p = subprocess.Popen([sys.executable, "mb.py", "--tag", tag, "--phases", phase, "--dur", str(a.dur), "--warm", str(a.warm)],
                         cwd=a.mb_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(max(0.0, t_start + a.warm + 0.5 - time.time()))
    s0 = snap(a, mark)
    time.sleep(max(0.0, t_start + a.dur - 0.5 - time.time()))
    s1 = snap(a, mark)
    try:
        out, err = p.communicate(timeout=a.dur + 900)
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate()
    post_ = snap(a)
    rows = jlines(out, "MB ")
    return {"mb": rows[0] if rows else {"error": (out[-300:] + err[-300:]).strip()}, "win": sdelta(s0, s1),
            "phase": sdelta(pre, post_), "t0": s0["t"], "t1": s1["t"], "mark0": s0["mark"], "mark1": s1["mark"]}


def g(path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f


METRICS = [("c1 essay tok/s", g(("essay", "probe", "decode_tps"))), ("c1 essay ms/step", g(("essay", "probe", "ms_per_decode_step"))),
           ("c1 essay tok/step", g(("essay", "probe", "tokens_per_step"))), ("c1 essay acc", g(("essay", "probe", "acceptance"))),
           ("c1 essay depth-0 share", g(("essay", "win", "zero_share")))]
for ph in ("c4", "c8"):
    for nm, path in (("aggregate tok/s", ("mb", "agg_dec_tps")), ("per-stream p50", ("mb", "per_stream_dec_p50")),
                     ("ms/step", ("mb", "ms_decode_step")), ("tok/step", ("mb", "tok_per_decode_step")), ("acc", ("mb", "acceptance")),
                     ("depth-0 share", ("win", "zero_share")), ("probe share", ("win", "probe_share")),
                     ("hit %", ("win", "hit_pct_rank0")), ("missed/step", ("win", "missed_step_rank0")),
                     ("stream MB/step", ("win", "stream_MB_step"))):
        METRICS.append(("%s %s" % (ph, nm), g((ph,) + path)))


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


def dfmt(w):
    d = (w or {}).get("draft") or {}
    return "zero %s probe %s moves %s of %s dec steps" % (d.get("zero_cycles"), d.get("probe_cycles"), d.get("zero_moves"),
                                                      (w or {}).get("decode_steps"))


def blocks(a):
    fixed = json.loads(a.fixed)
    val = {"A": a.a, "B": a.b}
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        if a.stop and os.path.exists(a.stop): break
        s0 = fn4(a)
        e0 = s0.get("epoch")
        rst = reset_depth0(a)
        posted = post(a, dict(fixed, **{"draft_depth0": 1, "draft_probe": val[arm], "epoch_bump": 1}))
        res = {"essay": essay(a)}
        for ph in ("c4", "c8"):
            if a.stop and os.path.exists(a.stop): break
            res[ph] = mbphase(a, ph, f"{a.label}-b{i}{arm}-{ph}")
        st = fn4(a)
        sw = st.get("switches", {})
        ok = posted is True and rst and sw.get("draft_probe") == val[arm] and sw.get("draft_depth0") == 1 \
            and all(sw.get(k) == v for k, v in fixed.items()) and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and all(isinstance(res.get(p), dict) and "error" not in res[p]["mb"] for p in ("c4", "c8"))
        row = {"label": a.label, "block": i, "arm": arm, "value": val[arm], "fixed": fixed, "reset": rst, "posted": posted,
               "applied": sw, "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as fh: fh.write(json.dumps(row) + "\n")
        e = res["essay"]; pr = e.get("probe") or {}
        pos = " ".join("p%d %s" % (p["p"], "-" if p["rate"] is None else "%.2f" % p["rate"]) for p in e.get("positions") or [])
        def ph(p):
            x = res.get(p) or {}; m = x.get("mb") or {}; w = x.get("win") or {}
            ps = w.get("passes_step") or {}
            return "%s %s tok/s p50 %s %s ms %s tok/step acc %s | %s | depth-0 %s probe %s | vram %s hit %s %% missed %s/st stream %s MB/st | passes a step played %s recorded %s unkeyed %s" % (
                p, m.get("agg_dec_tps"), m.get("per_stream_dec_p50"), m.get("ms_decode_step"), m.get("tok_per_decode_step"),
                m.get("acceptance"), dfmt(w), w.get("zero_share"), w.get("probe_share"), w.get("vram_units_end"),
                w.get("hit_pct_rank0"), w.get("missed_step_rank0"), w.get("stream_MB_step"), ps.get("played"), ps.get("recorded"),
                ps.get("issued_unkeyed"))
        print("BLOCK %d %s draft_probe=%d ok=%s reset=%s | essay %s tok/s %s ms %s tok/step acc %s | %s | by position %s | %s | %s" % (
            i, arm, val[arm], ok, rst, pr.get("decode_tps"), pr.get("ms_per_decode_step"), pr.get("tokens_per_step"),
            pr.get("acceptance"), dfmt(e.get("win")), pos or "-", ph("c4"), ph("c8")), flush=True)
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if rows[j]["arm"] != rows[j + 1]["arm"]]
    out = {"label": a.label, "switch": "draft_probe", "A": a.a, "B": a.b, "fixed": fixed, "order": a.order, "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1), "metrics": {}}
    for nm, get in METRICS:
        va = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == "A" and get(r["res"]) is not None]
        vb = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == "B" and get(r["res"]) is not None]
        pr = []
        for x, y in pairs:
            if not (x["ok"] and y["ok"]): continue
            ra, rb = (x, y) if x["arm"] == "A" else (y, x)
            ga, gb = get(ra["res"]), get(rb["res"])
            if ga and gb is not None: pr.append(gb / ga)
        m, h = ci(pr)
        out["metrics"][nm] = {"A": statistics.mean(va) if va else None, "B": statistics.mean(vb) if vb else None,
                              "pooled": (statistics.mean(vb) / statistics.mean(va)) if va and vb and statistics.mean(va) else None,
                              "pairs": pr, "pair_mean": m, "ci95": h}
    print("D0AB " + json.dumps(out), flush=True)
    return 0 if rows and out["blocks_bad"] < len(rows) else 1


def window(a):
    s = fn4(a)
    e0 = s.get("epoch")
    rst = reset_depth0(a) if a.reset else None
    posted = post(a, json.loads(a.set)) if a.set else None
    r = mbphase(a, a.phase, a.label, a.mark_cmd)
    st = fn4(a)
    want = json.loads(a.set) if a.set else {}
    sw = st.get("switches", {})
    ok = (posted in (None, True)) and rst in (None, True) and all(sw.get(k) == v for k, v in want.items() if k != "epoch_bump") \
        and not st.get("pending") and "error" not in r["mb"]
    row = {"label": a.label, "phase": a.phase, "set": want, "reset": rst, "posted": posted, "applied": sw, "epoch0": e0,
           "epoch": st.get("epoch"), "ok": ok, **r}
    if a.out:
        with open(a.out, "a") as fh: fh.write(json.dumps(row) + "\n")
    print("WIN " + json.dumps(row), flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["blocks", "window"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--a", type=int, default=16)
    ap.add_argument("--b", type=int, default=4096)
    ap.add_argument("--order", default="ABBA")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--phase", default="c8")
    ap.add_argument("--set", default="")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--mark-cmd", default=None)
    ap.add_argument("--stop", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    return blocks(a) if a.mode == "blocks" else window(a)


if __name__ == "__main__":
    sys.exit(main())
