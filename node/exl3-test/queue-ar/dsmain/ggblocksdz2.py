#!/usr/bin/env python3
"""ggblocksdz2.py (10.11.2026, queue-ar dsmain, ds41-gg agent) -- ggblocksdz.py (queued items run it, left as it is) with
ARMS instead of one switch: each arm is a set of runtime switches (POST /admin/fn4), so an arm can move several at once
(the combined drafting policy: draft_hybrid 1 + draft_depth0 1). Every block the phases a drafter needs: stepprobe essay,code 512 (c1),
an essay-only stepprobe between two /stats snapshots (acceptance by draft position), editprobe (an edit re-emitting a
~2.5k-token region), mb c4 and mb c8 (--dur / --warm), the scheduler's draft counters (GET /admin/fn4 "draft") around every
phase.

  ggblocksdz2.py --url URL --mb-dir DIR --edit EDITPROBE --label L --arm 'A={"draft_hybrid": 0}' --arm 'B={...}'
                 --order ABBAC [--pair A,B] [--dur 75] [--warm 10] [--gen 512] [--fixed '{}'] [--out FILE.jsonl]

A block = POST {fixed..., arm..., "epoch_bump": 1}, the phases, then GET /admin/fn4: a block whose switches (fixed or
the arm's) are not as set (GET reports the applied value), with a change pending, whose epoch did not move or whose POST
was refused is flagged and left out. Per metric and arm: the good blocks' values, mean and t 95 % half-width over blocks;
for --pair X,Y (default the first two arms): consecutive (X, Y) / (Y, X) blocks are pairs, each pair's Y/X, their mean
and t 95 %, and the pooled mean(Y) / mean(X). One "BLOCK" line a block, one "GGB {json}" line. Standard library only."""
import argparse, json, math, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def jlines(out, prefix="J "):
    rows = []
    for l in out.splitlines():
        if l.startswith(prefix):
            try: rows.append(json.loads(l[len(prefix):]))
            except ValueError: pass
    return rows


def stepprobe(a, prompts):
    p = subprocess.run([sys.executable, "stepprobe.py", prompts, str(a.gen)], cwd=a.mb_dir, capture_output=True,
                       text=True, timeout=1500)
    return {r.get("prompt"): r for r in jlines(p.stdout)}


def essaypos(a):
    try: s0 = http(a.url + "/stats")
    except Exception: s0 = {}
    r = stepprobe(a, "essay").get("essay", {})
    try: s1 = http(a.url + "/stats")
    except Exception: s1 = {}
    pa, pb = s0.get("draft_positions") or [], s1.get("draft_positions") or []
    pos = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        pos.append({"p": k + 1, "reached": re_, "accepted": ac, "rate": (ac / re_) if re_ else None})
    return {"probe": r, "positions": pos}


def editprobe(a):
    p = subprocess.run([sys.executable, "-I", a.edit, "9000", "1500", "7"], cwd=a.mb_dir, capture_output=True, text=True,
                       timeout=1500)
    rows = jlines(p.stdout)
    return rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def mb(a, phase, tag):
    p = subprocess.run([sys.executable, "mb.py", "--tag", tag, "--phases", phase, "--dur", str(a.dur), "--warm", str(a.warm)],
                       cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 900)
    rows = jlines(p.stdout, "MB ")
    return rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def draftc(a):
    try: return http(a.url + "/admin/fn4").get("draft") or {}
    except Exception: return {}


def phase(a, dph, name, f):
    """f() between two reads of the draft counters; their delta into dph[name]."""
    d0 = draftc(a)
    r = f()
    d1 = draftc(a)
    dph[name] = {k: (d1.get(k) or 0) - (d0.get(k) or 0) for k in d1}
    return r


def dfmt(d):
    if not d: return "-"
    return "zero %s probe %s moves %s hyb %s/%s/%s" % (d.get("zero_cycles"), d.get("probe_cycles"), d.get("zero_moves"),
                                                    d.get("hybrid_lookup"), d.get("hybrid_head"), d.get("hybrid_unahead"))


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


# (label, getter over a block's res)
def g(path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f


METRICS = []
for nm, key in (("essay", ("c1", "essay")), ("code", ("c1", "code")), ("edit", ("edit",))):
    for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc"), ("ms_per_decode_step", "ms/step")):
        METRICS.append(("c1 %s %s" % (nm, ml), g(key + (mk,))))
for ph in ("c4", "c8"):
    for mk, ml in (("agg_dec_tps", "aggregate tok/s"), ("per_stream_dec_p50", "per-stream p50"), ("tok_per_decode_step", "tok/step"),
                   ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
        METRICS.append(("%s %s" % (ph, ml), g((ph, mk))))


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
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    arms, names = {}, []
    for x in a.arm:
        k, v = x.split("=", 1)
        arms[k] = json.loads(v)
        names.append(k)
    if any(c not in arms for c in a.order):
        print("GGB " + json.dumps({"error": "order %s names an arm not given (%s)" % (a.order, names)}))
        return 1
    px, py = (a.pair.split(",") if a.pair else (names + names)[:2])
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        want = dict(fixed, **arms[arm])
        s0 = http(a.url + "/admin/fn4")
        e0 = s0.get("epoch")
        try:
            http(a.url + "/admin/fn4", dict(want, epoch_bump=1))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        dph = {}
        res = {"c1": phase(a, dph, "c1", lambda: stepprobe(a, "essay,code")),
               "essaypos": phase(a, dph, "essaypos", lambda: essaypos(a)),
               "edit": phase(a, dph, "edit", lambda: editprobe(a))}
        for ph in ("c4", "c8"):
            res[ph] = phase(a, dph, ph, lambda ph=ph: mb(a, ph, f"{a.label}-b{i}{arm}-{ph}"))
        res["draftph"] = dph
        st = http(a.url + "/admin/fn4")
        sw = st.get("switches", {})
        ok = posted is True and all(sw.get(k) == v for k, v in want.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in ("c4", "c8"))
        row = {"label": a.label, "block": i, "arm": arm, "values": arms[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        c1 = res["c1"]; ed = res.get("edit") or {}
        pos = " ".join("p%d %s" % (p["p"], "-" if p["rate"] is None else "%.2f" % p["rate"]) for p in res["essaypos"]["positions"])
        print("BLOCK %d %s %s ok=%s | draft %s | essay %s tok/s %s tok/step acc %s | code %s tok/s %s tok/step acc %s | essay by position %s | edit %s tok/s %s tok/step acc %s echo %s | c4 %s tok/s p50 %s %s tok/step acc %s %s ms | c8 %s tok/s p50 %s %s tok/step acc %s %s ms" % (
            i, arm, json.dumps(arms[arm]), ok, " / ".join("%s %s" % (k, dfmt(v)) for k, v in dph.items()) or "-",
            c1.get("essay", {}).get("decode_tps"), c1.get("essay", {}).get("tokens_per_step"), c1.get("essay", {}).get("acceptance"),
            c1.get("code", {}).get("decode_tps"), c1.get("code", {}).get("tokens_per_step"), c1.get("code", {}).get("acceptance"),
            pos or "-", ed.get("decode_tps"), ed.get("tokens_per_step"), ed.get("acceptance"), ed.get("echo_ratio"),
            res["c4"].get("agg_dec_tps"), res["c4"].get("per_stream_dec_p50"), res["c4"].get("tok_per_decode_step"),
            res["c4"].get("acceptance"), res["c4"].get("ms_decode_step"),
            res["c8"].get("agg_dec_tps"), res["c8"].get("per_stream_dec_p50"), res["c8"].get("tok_per_decode_step"),
            res["c8"].get("acceptance"), res["c8"].get("ms_decode_step")), flush=True)
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if {rows[j]["arm"], rows[j + 1]["arm"]} == {px, py}]
    out = {"label": a.label, "arms": arms, "fixed": fixed, "order": a.order, "pair": [px, py], "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1), "metrics": {}}
    for nm, get in METRICS:
        per = {}
        for k in names:
            v = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == k and get(r["res"]) is not None]
            m, h = ci(v)
            per[k] = {"values": v, "mean": m, "t95": h}
        pr = []
        for x, y in pairs:
            if not (x["ok"] and y["ok"]): continue
            rx, ry = (x, y) if x["arm"] == px else (y, x)
            gx, gy = get(rx["res"]), get(ry["res"])
            if gx and gy is not None: pr.append(gy / gx)
        m, h = ci(pr)
        mx, my = per.get(px, {}).get("mean"), per.get(py, {}).get("mean")
        out["metrics"][nm] = {"arms": per, "pairs": pr, "pair_mean": m, "ci95": h, "pooled": (my / mx) if mx and my is not None else None}
    print("GGB " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
