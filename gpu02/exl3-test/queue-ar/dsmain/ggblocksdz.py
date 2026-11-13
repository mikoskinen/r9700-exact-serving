#!/usr/bin/env python3
"""ggblocksdz.py (10.11.2026, queue-ar dsmain, ds41-gg agent) -- ggblocksdr.py (a queued item runs it, left as it is) plus the
scheduler's draft counters (GET /admin/fn4 "draft": cycles at depth 0, probe cycles, moves, hybrid proposals taken from the
lookup / left to the head, steps not issued ahead for it) read around every phase, their deltas in the row (res.draftph,
and the block's total in "draft") and on its BLOCK line. In-process A/B blocks over one runtime switch (POST /admin/fn4) on a DRAFTING server, every block the 1-user and load phases a drafter needs: stepprobe essay,code 512
(c1), an essay-only stepprobe between two /stats snapshots (acceptance by draft position, /stats draft_positions),
editprobe (an edit that re-emits a ~2.5k-token region: the case prompt lookup drafts for), mb c4 and mb c8 (closed
loop, --dur / --warm). ggblocks2.py (a queued item runs it) is left as it is; this keeps its block / pairing / t95
rules and its --fixed pins.

  ggblocksdz.py --url URL --mb-dir DIR --edit EDITPROBE --label L --switch gg_gemv --a 0 --b 2 --order ABBA
                [--dur 75] [--warm 10] [--gen 512] [--fixed '{}'] [--no-edit-a] [--out FILE.jsonl]

A block = POST {switch: value, fixed..., "epoch_bump": 1}, the phases, then GET /admin/fn4: a block whose switch (or a
fixed one) is not as set (GET reports the applied value), with a change pending, whose epoch did not move or whose POST
was refused is flagged and left out. Pairs are the order's consecutive (A, B) / (B, A) blocks. Per metric: A and B means
over the good blocks, the pooled B/A, each pair's B/A and a t 95 % interval. --no-edit-a skips the edit probe in A
blocks (a runtime cut; its metric then has no pairs). Prints one "BLOCK" line a block and one "GGB {json}" line.
Standard library only."""
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
    ap.add_argument("--switch", required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--order", default="ABBA")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--no-edit-a", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    val = {"A": a.a, "B": a.b}
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        s0 = http(a.url + "/admin/fn4")
        e0, d0 = s0.get("epoch"), s0.get("draft") or {}
        try:
            http(a.url + "/admin/fn4", dict(fixed, **{a.switch: val[arm], "epoch_bump": 1}))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        dph = {}
        res = {"c1": phase(a, dph, "c1", lambda: stepprobe(a, "essay,code")),
               "essaypos": phase(a, dph, "essaypos", lambda: essaypos(a))}
        if not (a.no_edit_a and arm == "A"):
            res["edit"] = phase(a, dph, "edit", lambda: editprobe(a))
        for ph in ("c4", "c8"):
            res[ph] = phase(a, dph, ph, lambda ph=ph: mb(a, ph, f"{a.label}-b{i}{arm}-{ph}"))
        res["draftph"] = dph
        st = http(a.url + "/admin/fn4")
        sw = st.get("switches", {})
        ok = posted is True and sw.get(a.switch) == val[arm] and all(sw.get(k) == v for k, v in fixed.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in ("c4", "c8"))
        d1 = st.get("draft") or {}
        dd = {k: (d1.get(k) or 0) - (d0.get(k) or 0) for k in d1}
        row = {"label": a.label, "block": i, "arm": arm, "value": val[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "draft": dd,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        c1 = res["c1"]; ed = res.get("edit") or {}
        pos = " ".join("p%d %s" % (p["p"], "-" if p["rate"] is None else "%.2f" % p["rate"]) for p in res["essaypos"]["positions"])
        print("BLOCK %d %s %s=%d ok=%s | draft %s | essay %s tok/s %s tok/step acc %s | code %s tok/s %s tok/step acc %s | essay by position %s | edit %s tok/s %s tok/step acc %s echo %s | c4 %s tok/s p50 %s %s tok/step acc %s %s ms | c8 %s tok/s p50 %s %s tok/step acc %s %s ms" % (
            i, arm, a.switch, val[arm], ok, " / ".join("%s %s" % (k, dfmt(v)) for k, v in dph.items()) or "-",
            c1.get("essay", {}).get("decode_tps"), c1.get("essay", {}).get("tokens_per_step"), c1.get("essay", {}).get("acceptance"),
            c1.get("code", {}).get("decode_tps"), c1.get("code", {}).get("tokens_per_step"), c1.get("code", {}).get("acceptance"),
            pos or "-", ed.get("decode_tps"), ed.get("tokens_per_step"), ed.get("acceptance"), ed.get("echo_ratio"),
            res["c4"].get("agg_dec_tps"), res["c4"].get("per_stream_dec_p50"), res["c4"].get("tok_per_decode_step"),
            res["c4"].get("acceptance"), res["c4"].get("ms_decode_step"),
            res["c8"].get("agg_dec_tps"), res["c8"].get("per_stream_dec_p50"), res["c8"].get("tok_per_decode_step"),
            res["c8"].get("acceptance"), res["c8"].get("ms_decode_step")), flush=True)
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if rows[j]["arm"] != rows[j + 1]["arm"]]
    out = {"label": a.label, "switch": a.switch, "A": a.a, "B": a.b, "fixed": fixed, "order": a.order, "blocks": len(rows),
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
    print("GGB " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
