#!/usr/bin/env python3
"""treeblocks.py (10.11.2026, queue-ar dsmain, ds-main-port T1 agent) -- in-process blocks over SETS of runtime switches
(POST /admin/fn4) on a drafting server, for lever T1 (the DSpark tree verify at position 1: dspark_tree, dspark_tree_depth,
dspark_tree_force). ggblocksdz2.py's arms (a JSON switch set each) with accblocks.py's per-phase windows:
  - every phase alone between two /stats + /metrics snapshots and two GET /admin/fn4 reads: essay and code (stepprobe.py, one
    prompt a call), edit (editprobe.py, 1500 tok), mb c4 / c8 (--dur / --warm); --phases picks which (the screen runs
    essay,code only);
  - per phase: the verified drafts (tree nodes) a decode step = radiance:draft_tokens_total / decode steps, verify rows a step
    at c1 = 1 + that (the root row and the verified nodes; a depth-0 step is one row and no draft), the D7 counters (GET
    /admin/fn4 "draft": zero / probe cycles, moves, hybrid lookup / head / unahead), essay acceptance by draft position (/stats
    draft_positions), and every /stats key whose name holds "tree" (the engine's tree counters where it reports any);
  - --rowcost: per c1 prompt, the least-squares line ms a decode step = a + b x verify rows a step over every good block's
    (rows, ms) point of every arm -- b is the verify-row cost in ms a row at one user (an estimate: a mixed run of depth-0
    and drafting steps sits on the line only if both are linear in rows).

  treeblocks.py --url URL --mb-dir DIR --edit EDITPROBE --label L --arm 'T0={"dspark_tree": 0}' --arm 'F3={...}'
                --order 'T0 F3 F3 T0' [--pair T0,F3] [--phases essay,code,edit,c4,c8] [--dur 75] [--warm 10] [--gen 512]
                [--fixed '{}'] [--rowcost] [--out FILE.jsonl]

A block = POST {fixed..., arm..., "epoch_bump": 1}, a 4-token nudge (so the change is applied before the first measured
step), the phases, then GET /admin/fn4: a block whose switches (fixed or the arm's) are not as set (GET reports the applied
value), with a change pending, whose epoch did not move, whose POST was refused or whose phase failed is flagged and left
out. Per metric and arm: the good blocks' mean and t 95 % half-width; for --pair X,Y (default the first two arms): adjacent
(X, Y) / (Y, X) blocks are pairs, each pair's Y/X, their mean and t 95 %, and the pooled mean(Y) / mean(X). One "BLOCK"
line a block, one "TREE {json}" line at the end. Standard library only."""
import argparse, json, math, re, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
MKEYS = ["radiance:engine_steps_decode_total", "radiance:draft_tokens_total", "radiance:draft_accepted_total",
         "radiance:decode_tokens_total"]


def http(url, body=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(url):
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=15).read().decode()
    except Exception:
        return {}
    out = {}
    for k in MKEYS:
        mm = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)
        out[k.split(":")[1]] = float(mm.group(2)) if mm else None
    return out


def jlines(out, prefix="J "):
    rows = []
    for l in out.splitlines():
        if l.startswith(prefix):
            try: rows.append(json.loads(l[len(prefix):]))
            except ValueError: pass
    return rows


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def sub(a, b):
    """b - a, element-wise through lists and dicts of numbers (non-numbers: b's)."""
    if isinstance(b, dict):
        return {k: sub((a or {}).get(k) if isinstance(a, dict) else None, v) for k, v in b.items()}
    if isinstance(b, list):
        a = a if isinstance(a, list) else []
        return [sub(a[i] if i < len(a) else None, v) for i, v in enumerate(b)]
    if num(b):
        return b - (a if num(a) else 0)
    return b


def snap(a):
    try: s = http(a.url + "/stats")
    except Exception: s = {}
    try: f = http(a.url + "/admin/fn4").get("draft") or {}
    except Exception: f = {}
    return {"draft_positions": s.get("draft_positions") or [], "tree": {k: v for k, v in s.items() if "tree" in k.lower()},
            "draft": f, "m": metrics(a.url)}


def window(a, b):
    pa, pb = a["draft_positions"], b["draft_positions"]
    pos = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        pos.append({"p": k + 1, "reached": re_, "accepted": ac, "rate": (ac / re_) if re_ else None})
    m = {k: (b["m"].get(k) - a["m"].get(k)) if b["m"].get(k) is not None and a["m"].get(k) is not None else None for k in b["m"]}
    steps = m.get("engine_steps_decode_total") or 0
    drafts = m.get("draft_tokens_total") or 0
    return {"positions": pos, "decode_steps": steps, "decode_tokens": m.get("decode_tokens_total"),
            "drafts": drafts, "accepted": m.get("draft_accepted_total"),
            "drafts_per_step": round(drafts / steps, 3) if steps else None,
            "draft": {k: (b["draft"].get(k) or 0) - (a["draft"].get(k) or 0) for k in b["draft"] if num(b["draft"].get(k))},
            "tree": sub(a["tree"], b["tree"])}


def stepprobe(a, name):
    p = subprocess.run([sys.executable, "stepprobe.py", name, str(a.gen)], cwd=a.mb_dir, capture_output=True, text=True,
                       timeout=1500)
    rows = jlines(p.stdout)
    return rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


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


def phase(a, fn, c1=True):
    s0 = snap(a)
    r = fn()
    s1 = snap(a)
    if isinstance(r, dict):
        r = dict(r)
        r["win"] = window(s0, s1)
        dps = r["win"].get("drafts_per_step")
        if c1 and dps is not None: r["verify_rows_per_step"] = round(1.0 + dps, 3)   # c1 phases (one sequence) only
    return r


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


def g(path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if num(x) else None
    return f


C1 = ("essay", "code", "edit")
MB = ("c4", "c8")


def metric_list(phases):
    out = []
    for nm in C1:
        if nm not in phases: continue
        for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc"),
                       ("ms_per_decode_step", "ms/step"), ("verify_rows_per_step", "rows/step")):
            out.append(("c1 %s %s" % (nm, ml), g((nm, mk))))
    for ph in MB:
        if ph not in phases: continue
        for mk, ml in (("agg_dec_tps", "aggregate tok/s"), ("per_stream_dec_p50", "per-stream p50"),
                       ("tok_per_decode_step", "tok/step"), ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
            out.append(("%s %s" % (ph, ml), g((ph, mk))))
        out.append(("%s drafts/step" % ph, g((ph, "win", "drafts_per_step"))))
    return out


def lsq(pts):
    """ms = a + b rows over (rows, ms) points; b, a, r^2, n. None when the rows do not vary."""
    pts = [(x, y) for x, y in pts if num(x) and num(y)]
    n = len(pts)
    if n < 2: return None
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 1e-9: return None
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx
    a = my - b * mx
    syy = sum((y - my) ** 2 for _, y in pts)
    r2 = (sxy * sxy / (sxx * syy)) if syy > 0 else None
    return {"ms_per_row": round(b, 3), "intercept_ms": round(a, 2), "r2": None if r2 is None else round(r2, 3), "n": n,
            "rows_min": round(min(x for x, _ in pts), 2), "rows_max": round(max(x for x, _ in pts), 2)}


def dfmt(d):
    if not d: return "-"
    return "zero %s probe %s moves %s hyb %s/%s/%s" % (d.get("zero_cycles"), d.get("probe_cycles"), d.get("zero_moves"),
                                                    d.get("hybrid_lookup"), d.get("hybrid_head"), d.get("hybrid_unahead"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--pair")
    ap.add_argument("--phases", default="essay,code,edit,c4,c8")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--rowcost", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    arms, names = {}, []
    for x in a.arm:
        k, v = x.split("=", 1)
        arms[k] = json.loads(v)
        names.append(k)
    order = [o for o in re.split(r"[\s,]+", a.order.strip()) if o]
    phases = [p for p in a.phases.split(",") if p]
    if not order or any(o not in arms for o in order) or any(p not in C1 + MB for p in phases):
        print("TREE " + json.dumps({"error": "order %s / phases %s: an arm not given (%s) or a phase not known" % (order, phases, names)}))
        return 2
    px, py = (a.pair.split(",") if a.pair else (names + names)[:2])
    rows, t_start = [], time.time()
    for i, arm in enumerate(order):
        want = dict(fixed, **arms[arm])
        try: e0 = http(a.url + "/admin/fn4").get("epoch")
        except Exception: e0 = None
        try:
            http(a.url + "/admin/fn4", dict(want, epoch_bump=1))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        try:   # a 4-token request applies the change at a step release before the first measured step
            urllib.request.urlopen(urllib.request.Request(a.url + "/completion", data=json.dumps(
                {"prompt": "Hello", "n_predict": 4, "temperature": 0}).encode(), headers={"Content-Type": "application/json"}),
                timeout=120).read()
        except Exception:
            pass
        res = {}
        for ph in phases:
            if ph in ("essay", "code"): res[ph] = phase(a, lambda ph=ph: stepprobe(a, ph))
            elif ph == "edit": res[ph] = phase(a, lambda: editprobe(a))
            else: res[ph] = phase(a, lambda ph=ph: mb(a, ph, "%s-b%d%s-%s" % (a.label, i, arm, ph)), c1=False)
        try: st = http(a.url + "/admin/fn4")
        except Exception: st = {}
        sw = st.get("switches", {})
        ok = posted is True and all(sw.get(k) == v for k, v in want.items()) and not st.get("pending") \
            and (st.get("epoch") or 0) > (e0 or 0) and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in phases)
        row = {"label": a.label, "block": i, "arm": arm, "values": arms[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        parts = ["BLOCK %d %s %s ok=%s" % (i, arm, json.dumps(arms[arm]), ok)]
        for nm in C1:
            if nm not in res: continue
            r, w = res[nm], res[nm].get("win") or {}
            parts.append("%s %s tok/s %s tok/step acc %s %s ms rows/step %s draft %s%s" % (
                nm, r.get("decode_tps"), r.get("tokens_per_step"), r.get("acceptance"), r.get("ms_per_decode_step"),
                r.get("verify_rows_per_step"), dfmt(w.get("draft")), (" tree %s" % json.dumps(w["tree"])[:300]) if w.get("tree") else ""))
        if "essay" in res:
            pos = " ".join("p%d %s/%s" % (p["p"], p["accepted"], p["reached"]) for p in (res["essay"].get("win") or {}).get("positions", [])[:20] if p["reached"])
            parts.append("essay by position %s" % (pos or "-"))
        for ph in MB:
            if ph not in res: continue
            r, w = res[ph], res[ph].get("win") or {}
            parts.append("%s %s tok/s p50 %s %s tok/step acc %s %s ms drafts/step %s draft %s" % (
                ph, r.get("agg_dec_tps"), r.get("per_stream_dec_p50"), r.get("tok_per_decode_step"), r.get("acceptance"),
                r.get("ms_decode_step"), w.get("drafts_per_step"), dfmt(w.get("draft"))))
        print(" | ".join(parts), flush=True)
    out = {"label": a.label, "arms": arms, "fixed": fixed, "order": order, "pair": [px, py], "phases": phases,
           "blocks": len(rows), "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1),
           "metrics": {}}
    for nm, get in metric_list(phases):
        per = {}
        for k in names:
            v = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == k and get(r["res"]) is not None]
            m, h = ci(v)
            per[k] = {"values": v, "mean": m, "t95": h}
        pr = []
        for x, y in zip(rows, rows[1:]):
            if not (x["ok"] and y["ok"]) or {x["arm"], y["arm"]} != {px, py}: continue
            rx, ry = (x, y) if x["arm"] == px else (y, x)
            gx, gy = get(rx["res"]), get(ry["res"])
            if gx and gy is not None: pr.append(gy / gx)
        m, h = ci(pr)
        mx, my = per.get(px, {}).get("mean"), per.get(py, {}).get("mean")
        out["metrics"][nm] = {"arms": per, "pairs": pr, "pair_mean": m, "ci95": h,
                              "pooled": (my / mx) if mx and my is not None else None}
    if a.rowcost:
        rc = {}
        for nm in ("essay", "code"):
            if nm not in phases: continue
            pts = [((r["res"].get(nm) or {}).get("verify_rows_per_step"), (r["res"].get(nm) or {}).get("ms_per_decode_step"))
                   for r in rows if r["ok"]]
            rc[nm] = {"fit": lsq(pts), "points": [[r["arm"], (r["res"].get(nm) or {}).get("verify_rows_per_step"),
                                                    (r["res"].get(nm) or {}).get("ms_per_decode_step")] for r in rows if r["ok"]]}
        out["rowcost"] = rc
    print("TREE " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
