#!/usr/bin/env python3
"""wkblocks.py (10.11.2026, queue-ar dsmain, lever W1) -- the decode blocks of 57-p4zyzzzzz-ds-*-walk (dsm-walk.sh): in-process
A / B blocks of runtime switches (POST /admin/fn4) as ggblocksdz2.py runs them, whose helpers it imports read only (http,
stepprobe, editprobe, mb, draftc), with the draft-position counters (/stats "draft_positions": reached / accepted a position)
around EVERY phase instead of one extra essay run:

  c1 essay (stepprobe essay 512), c1 code (stepprobe code 512), edit (editprobe 9000 1500 7), mb c4, mb c8 (--dur / --warm),
  and (--fixed5) a diagnostic: essay + code again at the FIXED window of 5 (draft_adaptive 0, draft_depth0 0 posted for the
  phase, the block's switches posted back after it), so every position is reached and the walk's conditional acceptance at
  positions 2-5 shows (positions only; its speed is not a stack number).

  wkblocks.py --url URL --mb-dir DIR --edit EDITPROBE --label L --arm 'A={"dspark_walk": 0}' --arm 'B={"dspark_walk": 1}'
              --order ABBA [--dur 75] [--warm 10] [--gen 512] [--fixed '{}'] [--fixed5] [--out FILE.jsonl]

A block = POST {fixed..., arm..., "epoch_bump": 1}, the phases, GET /admin/fn4: a block whose switches are not as set, with a
change pending, whose epoch did not move, whose POST was refused or whose mb phase failed is flagged (ok false) and left out.
One "BLOCK" line a block, one "WKB {json}" line at the end (per metric and arm: values, mean, t95; consecutive A/B pairs'
B/A, mean and t95; positions summed over the good blocks of each arm). Standard library only."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggblocksdz2 as gg   # read only: http, jlines, stepprobe, editprobe, mb, draftc, ci, g


def snap(a):
    try: return gg.http(a.url + "/stats")
    except Exception: return {}


def positions(s0, s1):
    pa, pb = s0.get("draft_positions") or [], s1.get("draft_positions") or []
    out = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        out.append({"p": k + 1, "reached": re_, "accepted": ac})
    return out


def phase(a, dph, pos, name, f):
    d0, s0 = gg.draftc(a), snap(a)
    r = f()
    s1, d1 = snap(a), gg.draftc(a)
    dph[name] = {k: (d1.get(k) or 0) - (d0.get(k) or 0) for k in d1}
    pos[name] = positions(s0, s1)
    return r


def pfmt(ps):
    return " ".join("p%d %d/%d" % (p["p"], p["accepted"], p["reached"]) for p in ps) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--pair")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--fixed5", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    arms, names = {}, []
    for x in a.arm:
        k, v = x.split("=", 1)
        arms[k] = json.loads(v)
        names.append(k)
    if any(c not in arms for c in a.order):
        print("WKB " + json.dumps({"error": "order %s names an arm not given (%s)" % (a.order, names)}))
        return 1
    px, py = (a.pair.split(",") if a.pair else (names + names)[:2])
    rows, t0 = [], time.time()
    for i, arm in enumerate(a.order):
        want = dict(fixed, **arms[arm])
        e0 = gg.http(a.url + "/admin/fn4").get("epoch")
        try:
            gg.http(a.url + "/admin/fn4", dict(want, epoch_bump=1))
            posted = True
        except Exception as e:
            posted = str(e)[:200]
        dph, pos = {}, {}
        res = {"c1": {}}
        res["c1"]["essay"] = phase(a, dph, pos, "essay", lambda: gg.stepprobe(a, "essay").get("essay", {}))
        res["c1"]["code"] = phase(a, dph, pos, "code", lambda: gg.stepprobe(a, "code").get("code", {}))
        res["edit"] = phase(a, dph, pos, "edit", lambda: gg.editprobe(a))
        for ph in ("c4", "c8"):
            res[ph] = phase(a, dph, pos, ph, lambda ph=ph: gg.mb(a, ph, f"{a.label}-b{i}{arm}-{ph}"))
        st_main = gg.http(a.url + "/admin/fn4")   # switches as the main phases ran them (read before the diagnostic)
        if a.fixed5:
            f5 = dict(want, draft_adaptive=0, draft_depth0=0)
            try:
                gg.http(a.url + "/admin/fn4", dict(f5, epoch_bump=1))
                f5ok = True
            except Exception as e:
                f5ok = str(e)[:200]
            sw5 = (gg.http(a.url + "/admin/fn4").get("switches") or {}) if f5ok is True else {}
            res["fixed5"] = {"posted": f5ok, "applied_before": sw5}
            res["fixed5"]["essay"] = phase(a, dph, pos, "essay5", lambda: gg.stepprobe(a, "essay").get("essay", {}))
            res["fixed5"]["code"] = phase(a, dph, pos, "code5", lambda: gg.stepprobe(a, "code").get("code", {}))
            sw5b = gg.http(a.url + "/admin/fn4").get("switches") or {}
            res["fixed5"]["applied"] = {k: sw5b.get(k) for k in ("draft_adaptive", "draft_depth0", "dspark_walk")}
            try: gg.http(a.url + "/admin/fn4", dict(want, epoch_bump=1))   # the block's switches back
            except Exception as e: res["fixed5"]["restore"] = str(e)[:200]
        res["draftph"], res["pos"] = dph, pos
        st = st_main   # 11.11.: judge the block on the main phases' switches; after the diagnostic the restore is still pending
        sw = st.get("switches", {})
        ok = posted is True and all(sw.get(k) == v for k, v in want.items()) and not st.get("pending") \
            and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in ("c4", "c8"))
        row = {"label": a.label, "block": i, "arm": arm, "values": arms[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res, "t": round(time.time() - t0, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        c1, ed = res["c1"], res.get("edit") or {}
        def pr(d, k1, k2, k3, k4): return "%s tok/s %s tok/step acc %s %s ms" % (d.get(k1), d.get(k2), d.get(k3), d.get(k4))
        line = "BLOCK %d %s %s ok=%s | essay %s [%s] | code %s [%s] | edit %s [%s] | c4 %s p50 %s [%s] | c8 %s p50 %s [%s]" % (
            i, arm, json.dumps(arms[arm]), ok,
            pr(c1.get("essay", {}), "decode_tps", "tokens_per_step", "acceptance", "ms_per_decode_step"), pfmt(pos.get("essay", [])),
            pr(c1.get("code", {}), "decode_tps", "tokens_per_step", "acceptance", "ms_per_decode_step"), pfmt(pos.get("code", [])),
            pr(ed, "decode_tps", "tokens_per_step", "acceptance", "ms_per_decode_step"), pfmt(pos.get("edit", [])),
            pr(res["c4"], "agg_dec_tps", "tok_per_decode_step", "acceptance", "ms_decode_step"), res["c4"].get("per_stream_dec_p50"), pfmt(pos.get("c4", [])),
            pr(res["c8"], "agg_dec_tps", "tok_per_decode_step", "acceptance", "ms_decode_step"), res["c8"].get("per_stream_dec_p50"), pfmt(pos.get("c8", [])))
        if a.fixed5:
            f5 = res["fixed5"]
            line += " | fixed window 5 (diagnostic; applied %s): essay %s tok/step [%s], code %s tok/step [%s]" % (
                json.dumps(f5.get("applied")), f5.get("essay", {}).get("tokens_per_step"), pfmt(pos.get("essay5", [])),
                f5.get("code", {}).get("tokens_per_step"), pfmt(pos.get("code5", [])))
        print(line, flush=True)
    # per metric and arm, pairs
    G = gg.g
    METRICS = []
    for nm, key in (("c1 essay", ("c1", "essay")), ("c1 code", ("c1", "code")), ("edit", ("edit",))):
        for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc"), ("ms_per_decode_step", "ms/step")):
            METRICS.append(("%s %s" % (nm, ml), G(key + (mk,))))
    for ph in ("c4", "c8"):
        for mk, ml in (("agg_dec_tps", "tok/s"), ("per_stream_dec_p50", "p50"), ("tok_per_decode_step", "tok/step"),
                       ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
            METRICS.append(("%s %s" % (ph, ml), G((ph, mk))))
    if a.fixed5:
        for nm in ("essay", "code"):
            METRICS.append(("fixed5 %s tok/step" % nm, G(("fixed5", nm, "tokens_per_step"))))
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if {rows[j]["arm"], rows[j + 1]["arm"]} == {px, py}]
    out = {"label": a.label, "arms": arms, "fixed": fixed, "order": a.order, "pair": [px, py], "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t0, 1), "metrics": {}, "positions": {}}
    for nm, get in METRICS:
        per = {}
        for k in names:
            v = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == k and get(r["res"]) is not None]
            m, h = gg.ci(v)
            per[k] = {"values": v, "mean": m, "t95": h}
        pr_ = []
        for x, y in pairs:
            if not (x["ok"] and y["ok"]): continue
            rx, ry = (x, y) if x["arm"] == px else (y, x)
            gx, gy = get(rx["res"]), get(ry["res"])
            if gx and gy is not None: pr_.append(gy / gx)
        m, h = gg.ci(pr_)
        out["metrics"][nm] = {"arms": per, "pairs": pr_, "pair_mean": m, "ci95": h}
    for k in names:
        acc = {}
        for r in rows:
            if not (r["ok"] and r["arm"] == k): continue
            for ph, ps in (r["res"].get("pos") or {}).items():
                t = acc.setdefault(ph, {})
                for p in ps:
                    q = t.setdefault(p["p"], [0, 0])
                    q[0] += p["accepted"]; q[1] += p["reached"]
        out["positions"][k] = {ph: [{"p": p, "accepted": v[0], "reached": v[1]} for p, v in sorted(t.items())] for ph, t in acc.items()}
    print("WKB " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
