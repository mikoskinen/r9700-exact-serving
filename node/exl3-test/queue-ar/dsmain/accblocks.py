#!/usr/bin/env python3
"""accblocks.py (10.11.2026, queue-ar dsmain, ds41-acc agent) -- in-process blocks over one runtime switch (POST /admin/fn4)
on a DRAFTING server, for the DSpark acceptance levers C1 (dspark_conf, the confidence gate) and T1 (dspark_alt, the shadow
tree at position 1). ggblocksdr.py's block / pairing / t95 rules (kept as it is), with:
  - any number of arms: --arms 'A=0,B=1,C=150' and an order of their letters (ABCCBA: each arm's mean centred on the same
    time, so a linear drift cancels in the pooled means; pairs are the order's adjacent blocks of two arms);
  - each 1-user phase alone between two /stats snapshots: essay (stepprobe 512, its acceptance BY DRAFT POSITION from
    /stats draft_positions), code (stepprobe 512), edit (editprobe.py, 1500 tok); then mb c4 and mb c8 (--dur / --warm);
  - per phase the /stats draft_conf delta: the gate's cuts and cut histogram (verify counts it set), its calibration
    (reached / accepted by draft position and sigmoid decile) and the shadow tree (verifies with a draft, chain accepted,
    first-draft misses, branch hits, gain[branches][depth]) -> tree tokens a step = (tokens + gain) / decode steps, exact for
    greedy on each verify (core/sched/draft_conf.h);
  - per phase the drafts verified a step (draft_n / decode steps: the verify rows the gate saves).

  accblocks.py --url URL --mb-dir DIR --edit EDITPROBE --label L --switch dspark_conf --arms 'A=0,B=1,C=150' --order ABCCBA
               [--dur 75] [--warm 10] [--gen 512] [--fixed '{}'] [--no-mb] [--out FILE.jsonl]

A block = POST {switch: value, fixed..., "epoch_bump": 1}, the phases, GET /admin/fn4: a block whose switch (or a fixed one)
is not as set, with a change pending, whose epoch did not move or whose POST was refused is flagged and left out. Prints one
"BLOCK" line a block and one "ACC {json}" line. Standard library only."""
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


def sub(a, b):
    """b - a, element-wise through lists and dicts of numbers."""
    if isinstance(b, dict):
        return {k: sub((a or {}).get(k) if isinstance(a, dict) else None, v) for k, v in b.items()}
    if isinstance(b, list):
        a = a if isinstance(a, list) else []
        return [sub(a[i] if i < len(a) else None, v) for i, v in enumerate(b)]
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return b - (a if isinstance(a, (int, float)) and not isinstance(a, bool) else 0)
    return b


def snap(a):
    try: s = http(a.url + "/stats")
    except Exception: s = {}
    return {"draft_positions": s.get("draft_positions") or [], "draft_conf": s.get("draft_conf") or {},
            "m": metrics(a.url)}


def window(a, b):
    """What happened between two snapshots: positions, the draft_conf delta, the metric deltas."""
    pa, pb = a["draft_positions"], b["draft_positions"]
    pos = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        pos.append({"p": k + 1, "reached": re_, "accepted": ac, "rate": (ac / re_) if re_ else None})
    m = {k: (b["m"].get(k) - a["m"].get(k)) if b["m"].get(k) is not None and a["m"].get(k) is not None else None
         for k in b["m"]}
    dc = sub(a["draft_conf"], b["draft_conf"])
    for k in ("n_alt", "len"):   # sizes, not counters
        if isinstance((dc or {}).get("shadow"), dict): dc["shadow"][k] = (b["draft_conf"].get("shadow") or {}).get(k)
    steps = m.get("engine_steps_decode_total") or 0
    toks = m.get("decode_tokens_total") or 0
    d = {"positions": pos, "draft_conf": dc, "decode_steps": steps, "decode_tokens": toks,
         "drafts_per_step": round((m.get("draft_tokens_total") or 0) / steps, 3) if steps else None}
    sh = (dc or {}).get("shadow") or {}
    gain = sh.get("gain") or []
    if steps and gain:
        d["tree_tok_step"] = [[round((toks + g) / steps, 3) for g in row] for row in gain]
        d["chain_tok_step"] = round(toks / steps, 3)
    return d


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


def phase(a, fn):
    s0 = snap(a)
    r = fn()
    s1 = snap(a)
    if isinstance(r, dict):
        r = dict(r)
        r["win"] = window(s0, s1)
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
            if isinstance(x, list):
                if not isinstance(k, int) or k >= len(x): return None
            elif not isinstance(x, dict):
                return None
            x = x[k] if isinstance(x, list) else x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f


METRICS = []
for nm in ("essay", "code", "edit"):
    for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc"),
                   ("ms_per_decode_step", "ms/step")):
        METRICS.append(("c1 %s %s" % (nm, ml), g((nm, mk))))
    METRICS.append(("c1 %s drafts/step" % nm, g((nm, "win", "drafts_per_step"))))
for ph in ("c4", "c8"):
    for mk, ml in (("agg_dec_tps", "aggregate tok/s"), ("per_stream_dec_p50", "per-stream p50"),
                   ("tok_per_decode_step", "tok/step"), ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
        METRICS.append(("%s %s" % (ph, ml), g((ph, mk))))
    METRICS.append(("%s drafts/step" % ph, g((ph, "win", "drafts_per_step"))))


def shadow_line(w):
    sh = ((w or {}).get("draft_conf") or {}).get("shadow") or {}
    if not sh.get("verifies"): return "-"
    tt = w.get("tree_tok_step") or []
    s = "verifies %s p1-miss %s hits %s chain %s tok/step" % (sh.get("verifies"), sh.get("p1_miss"), sh.get("alt_hit"),
                                                               w.get("chain_tok_step"))
    for i, row in enumerate(tt):
        s += " | %d branch(es) depth 1..%d: %s" % (i + 1, len(row), " ".join("%.3f" % x for x in row))
    return s


def conf_line(w):
    c = ((w or {}).get("draft_conf") or {}).get("conf") or {}
    if not c.get("cuts"): return "-"
    return "cuts %s hist %s drafts %s" % (c.get("cuts"), c.get("cut_hist"), c.get("cut_drafts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--switch", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--order", default="ABBA")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--no-mb", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    val = {kv.split("=")[0].strip(): int(kv.split("=")[1]) for kv in a.arms.split(",")}
    if not val or "A" not in val or any(c not in val for c in a.order):
        print("ACC " + json.dumps({"error": "arms %s do not cover order %s (A is the reference)" % (val, a.order)}))
        return 2
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        try: e0 = http(a.url + "/admin/fn4").get("epoch")
        except Exception: e0 = None
        try:
            http(a.url + "/admin/fn4", dict(fixed, **{a.switch: val[arm], "epoch_bump": 1}))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        res = {"essay": phase(a, lambda: stepprobe(a, "essay")), "code": phase(a, lambda: stepprobe(a, "code")),
               "edit": phase(a, lambda: editprobe(a))}
        if not a.no_mb:
            for ph in ("c4", "c8"):
                res[ph] = phase(a, lambda ph=ph: mb(a, ph, "%s-b%d%s-%s" % (a.label, i, arm, ph)))
        try: st = http(a.url + "/admin/fn4")
        except Exception: st = {}
        sw = st.get("switches", {})
        ok = posted is True and sw.get(a.switch) == val[arm] and all(sw.get(k) == v for k, v in fixed.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and "error" in res[p] for p in ("essay", "code", "c4", "c8"))
        row = {"label": a.label, "block": i, "arm": arm, "value": val[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        es = res["essay"]
        pos = " ".join("p%d %s/%s" % (p["p"], p["accepted"], p["reached"]) for p in (es.get("win") or {}).get("positions", []))
        parts = ["BLOCK %d %s %s=%d ok=%s" % (i, arm, a.switch, val[arm], ok)]
        for nm in ("essay", "code", "edit"):
            r = res[nm]
            parts.append("%s %s tok/s %s tok/step acc %s %s ms drafts/step %s" % (
                nm, r.get("decode_tps"), r.get("tokens_per_step"), r.get("acceptance"), r.get("ms_per_decode_step"),
                (r.get("win") or {}).get("drafts_per_step")))
        parts.append("essay by position %s" % (pos or "-"))
        for ph in ("c4", "c8"):
            r = res.get(ph) or {}
            if r:
                parts.append("%s %s tok/s p50 %s %s tok/step acc %s %s ms drafts/step %s" % (
                    ph, r.get("agg_dec_tps"), r.get("per_stream_dec_p50"), r.get("tok_per_decode_step"), r.get("acceptance"),
                    r.get("ms_decode_step"), (r.get("win") or {}).get("drafts_per_step")))
        for nm in ("essay", "code", "edit", "c4", "c8"):
            w = (res.get(nm) or {}).get("win")
            if conf_line(w) != "-": parts.append("gate %s: %s" % (nm, conf_line(w)))
            if shadow_line(w) != "-": parts.append("shadow %s: %s" % (nm, shadow_line(w)))
        print(" | ".join(parts), flush=True)
    out = {"label": a.label, "switch": a.switch, "arms": val, "fixed": fixed, "order": a.order, "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1), "metrics": {}}
    for nm, get in METRICS:
        means = {}
        for arm in val:
            v = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == arm and get(r["res"]) is not None]
            means[arm] = statistics.mean(v) if v else None
        ent = {"mean": means}
        for arm in val:
            if arm == "A": continue
            pr = []
            for x, y in zip(rows, rows[1:]):
                if not (x["ok"] and y["ok"]) or {x["arm"], y["arm"]} != {"A", arm}: continue
                ra, rb = (x, y) if x["arm"] == "A" else (y, x)
                ga, gb = get(ra["res"]), get(rb["res"])
                if ga and gb is not None: pr.append(gb / ga)
            m, h = ci(pr)
            ent[arm] = {"pooled": (means[arm] / means["A"]) if means.get(arm) is not None and means.get("A") else None,
                        "pairs": pr, "pair_mean": m, "ci95": h}
        out["metrics"][nm] = ent
    # the gate and the shadow tree, summed over each arm's good blocks, per phase
    summ = {}
    for arm in val:
        for nm in ("essay", "code", "edit", "c4", "c8"):
            acc = None
            for r in rows:
                if not (r["ok"] and r["arm"] == arm): continue
                w = (r["res"].get(nm) or {}).get("win")
                if not w: continue
                cur = {"draft_conf": w.get("draft_conf"), "decode_steps": w.get("decode_steps"),
                       "decode_tokens": w.get("decode_tokens")}
                acc = cur if acc is None else add(acc, cur)
                sh = ((acc.get("draft_conf") or {}).get("shadow"))
                if isinstance(sh, dict):
                    for k in ("n_alt", "len"): sh[k] = (((cur.get("draft_conf") or {}).get("shadow")) or {}).get(k)
            if acc:
                summ.setdefault(arm, {})[nm] = acc
                st, tk = acc.get("decode_steps") or 0, acc.get("decode_tokens") or 0
                gain = ((acc.get("draft_conf") or {}).get("shadow") or {}).get("gain") or []
                if st and gain:
                    acc["chain_tok_step"] = round(tk / st, 3)
                    acc["tree_tok_step"] = [[round((tk + x) / st, 3) for x in row] for row in gain]
    out["sums"] = summ
    print("ACC " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


def add(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return {k: add(x.get(k), y.get(k)) for k in set(x) | set(y)}
    if isinstance(x, list) or isinstance(y, list):
        x = x if isinstance(x, list) else []
        y = y if isinstance(y, list) else []
        return [add(x[i] if i < len(x) else None, y[i] if i < len(y) else None) for i in range(max(len(x), len(y)))]
    nx = isinstance(x, (int, float)) and not isinstance(x, bool)
    ny = isinstance(y, (int, float)) and not isinstance(y, bool)
    if nx or ny: return (x if nx else 0) + (y if ny else 0)
    return y if y is not None else x


if __name__ == "__main__":
    sys.exit(main())
