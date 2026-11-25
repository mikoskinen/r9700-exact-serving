#!/usr/bin/env python3
"""d0attrprof.py (10.11.2026, queue-ar dsmain, ds-main-port; item 57-p4zyzzzzz-ds-hczz-d0attr's analyser) -- where a depth-0
step of an NST 5 (DSpark, adaptive, draft_depth0 1) server spends the time an NST 0 step does not. Three modes:

  d0attrprof.py stamps STAMPS.csv[.gz] --win NAME:T0:T1 [--win ...] [--ref NAME:CSV[.gz]:T0:T1] [--json OUT] [--label L]
      RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; card 0, host
      CLOCK_MONOTONIC ns; every Nth played pass on card 0, all its dispatches). Per window, every pass whose first dispatch
      starts in it, grouped by KIND (its tape's record count): passes, median wall (first start .. last end), busy (union
      of its dispatches), dispatches, and the kernel families' device us a pass (family = the mangled name's last
      component). The TRUNK kind is the kind with the most records among the kinds with >= 25 % of the most frequent
      kind's passes. STEPS: from one trunk pass's start to the next one's; per step its period, the trunk pass's wall, each
      other kind's wall (and how many a step), and card 0's idle time (period - the union of every pass's dispatches in
      it). Median and trimmed mean (steps over 3x the median period left out: prefill / mixed chunks, stalls).
      --ref: the same reading of another CSV's window (an NST 0 run), then per trunk kernel family the device us a pass
      against it (the largest differences first) and the families only one side has.
  d0attrprof.py ops ENGINE_LOG --win NAME:L0:L1 [--win ...] [--ref-json DSOPPROF.json] [--json OUT] [--label L]
      --profile-ops tables (ds_opprof_draft.py's parser, imported read-only from beside this file), each window as
      engine-log line numbers: per rank device ms a step by class, the TRUNK's (ops up to the trunk's head, the first
      logits_gemm op; the head counted in dense_gemm as ds_opprof.py counts it) apart from the DRAFTER's (every op after the
      head that is not a sampling op; DSpark's fc, its blocks' attention / MoE / dense ops, its own head), lane waits a
      step; rank 0's drafter ops by name with calls a step (about 1 a step = the context pass, which runs at depth 0; about
      0 = the query pass, which does not) and its trunk ops by name. --ref-json: a ds_opprof.py JSON of an NST 0 run
      (summary by class per rank, rank0_by_op): per class the trunk's ms a step against it (mean over ranks and rank 0)
      and rank 0's ops by name against it, ops only one side has. PROFILED: synchronised steps, not a speed.
  d0attrprof.py summary --blocks BLOCKS.jsonl --nst0 GG2_BLOCKS.jsonl [--nst0-stats STATS_PHASES.txt]
                        [--stamps STAMPS.json] [--ops OPS.json] [--json OUT]
      The item's summary: c8 / c4 / c1 ms a decode step (P4096 vs NST 0 gg_gemv 2, P16 vs P4096; value, reference, %),
      the share of each phase's decode steps at depth 0, aggregate tok/s, the expert counters against NST 0, then the c8
      attribution: probe share (P16 vs P4096), the context pass (stamps: its wall a step on card 0, the line's gg_gemv),
      the trunk pass against NST 0 at the same plugin forms (expert kernels vs the rest), card 0's idle time between
      passes, and the remainder of the clean P4096 - NST 0 difference.
Prints "D0AP {json}" (the numbers) and readable "D0AP <label> ..." lines. Standard library only."""
import csv, gzip, json, math, os, re, statistics, sys
from collections import defaultdict

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def opn(p):
    return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")


_fam_cache = {}


def family(k):
    """The mangled kernel name's last name component (template arguments dropped)."""
    f = _fam_cache.get(k)
    if f is not None: return f
    s = k
    i = 2 if s.startswith("_Z") else 0
    if s[i:i + 1] == "N": i += 1
    comps = []
    while i < len(s) and s[i].isdigit():
        j = i
        while j < len(s) and s[j].isdigit(): j += 1
        n = int(s[i:j])
        comps.append(s[j:j + n]); i = j + n
        if s[i:i + 1] == "I":   # template arguments: the name is complete
            break
    comps = [c for c in comps if c and not c.startswith("_GLOBAL__N")]
    f = comps[-1] if comps else k[:60]
    _fam_cache[k] = f
    return f


EXPERT_RE = re.compile(r"exl3|moe|expert", re.I)


def is_expert(fam):
    return bool(EXPERT_RE.search(fam))


def wins(a, flag="--win"):
    out = []
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            n, lo, hi = a[i + 1].split(":")
            out.append((n, int(lo), int(hi)))
    return out


def union(iv):
    """Total length of the union of [s, e) intervals."""
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None: tot += cur_e - cur_s
    return tot


def read_passes(path, W):
    """{window: {pass: {"rec": records, "rows": [(s, e, family)]}}} for the passes whose dispatches start in a window."""
    P = {wn: defaultdict(lambda: {"rec": 0, "rows": []}) for wn, _, _ in W}
    with opn(path) as f:
        for r in csv.DictReader(f):
            s, e = r.get("start_ns"), r.get("end_ns")
            if not s or not e: continue
            s, e = int(s), int(e)
            for wn, lo, hi in W:
                if lo <= s <= hi:
                    p = P[wn][int(r["pass"])]
                    p["rec"] = int(r["records"])
                    p["rows"].append((s, e, family(r["kernel"])))
    return P


def med(v):
    return statistics.median(v) if v else None


def tmean(v, cap):
    v = [x for x in v if cap is None or x <= cap]
    return statistics.mean(v) if v else None


def analyse(passes):
    """One window's passes -> kinds, the trunk kind, step structure, kernel families a pass by kind."""
    if not passes: return {"error": "no stamped pass in the window"}
    ps = []
    for pid, p in passes.items():
        rows = p["rows"]
        if not rows: continue
        t0, t1 = min(r[0] for r in rows), max(r[1] for r in rows)
        fam = defaultdict(lambda: [0, 0.0])
        for s, e, fm in rows:
            fam[fm][0] += 1; fam[fm][1] += (e - s) / 1e3
        ps.append({"id": pid, "rec": p["rec"], "t0": t0, "t1": t1, "wall": (t1 - t0) / 1e3,
                   "busy": union([(r[0], r[1]) for r in rows]) / 1e3, "disp": len(rows), "fam": fam,
                   "iv": [(r[0], r[1]) for r in rows]})
    ps.sort(key=lambda x: x["t0"])
    # the window's first and last passes may be cut: leave them out when there are enough
    ps = ps[1:-1] if len(ps) > 6 else ps
    bykind = defaultdict(list)
    for p in ps: bykind[p["rec"]].append(p)
    top = max(len(v) for v in bykind.values())
    trunk = max(k for k, v in bykind.items() if len(v) >= 0.25 * top)
    kinds = {}
    for k, v in sorted(bykind.items(), key=lambda kv: -len(kv[1])):
        fam = defaultdict(lambda: [0.0, 0.0])
        for p in v:
            for fm, (n, us) in p["fam"].items():
                fam[fm][0] += n; fam[fm][1] += us
        kinds[str(k)] = {"passes": len(v), "wall_us": med([p["wall"] for p in v]), "busy_us": med([p["busy"] for p in v]),
                         "dispatches": med([p["disp"] for p in v]),
                         "fam_us": {fm: round(x[1] / len(v), 2) for fm, x in fam.items()},
                         "fam_n": {fm: round(x[0] / len(v), 2) for fm, x in fam.items()}}
    # steps: trunk start to the next trunk start
    tr = [p for p in ps if p["rec"] == trunk]
    steps = []
    j = 0
    for a, b in zip(tr, tr[1:]):
        inside = []
        while j < len(ps) and ps[j]["t0"] < a["t0"]: j += 1
        k = j
        while k < len(ps) and ps[k]["t0"] < b["t0"]:
            inside.append(ps[k]); k += 1
        period = (b["t0"] - a["t0"]) / 1e3
        iv = []
        for p in inside: iv.extend((max(s, a["t0"]), min(e, b["t0"])) for s, e in p["iv"] if s < b["t0"])
        iv = [(s, e) for s, e in iv if e > s]
        other = defaultdict(lambda: [0, 0.0])
        for p in inside:
            if p is a: continue
            other[str(p["rec"])][0] += 1; other[str(p["rec"])][1] += p["wall"]
        spans = [(max(p["t0"], a["t0"]), min(p["t1"], b["t0"])) for p in inside]
        spans = [(s, e) for s, e in spans if e > s]
        steps.append({"period": period, "trunk_wall": a["wall"], "trunk_busy": a["busy"],
                      "idle": period - union(iv) / 1e3, "between": period - union(spans) / 1e3, "other": other,
                      "gap_after_trunk": ((min((p["t0"] for p in inside if p is not a), default=b["t0"]) - a["t1"]) / 1e3)})
    pm = med([s["period"] for s in steps])
    cap = 3 * pm if pm else None
    keep = [s for s in steps if cap is None or s["period"] <= cap]
    okinds = sorted({k for s in keep for k in s["other"]})
    other = {}
    for k in okinds:
        other[k] = {"per_step": statistics.mean(s["other"].get(k, [0, 0.0])[0] for s in keep) if keep else None,
                    "wall_us_step": statistics.mean(s["other"].get(k, [0, 0.0])[1] for s in keep) if keep else None}
    return {"trunk": trunk, "passes": len(ps), "kinds": kinds,
            "steps": len(steps), "steps_kept": len(keep),
            "period_us_med": pm, "period_us_mean": tmean([s["period"] for s in steps], cap),
            "trunk_wall_us_med": med([s["trunk_wall"] for s in keep]), "trunk_wall_us_mean": tmean([s["trunk_wall"] for s in keep], None),
            "trunk_busy_us_mean": tmean([s["trunk_busy"] for s in keep], None),
            "idle_us_med": med([s["idle"] for s in keep]), "idle_us_mean": tmean([s["idle"] for s in keep], None),
            "between_us_mean": tmean([s["between"] for s in keep], None),
            "gap_after_trunk_us_med": med([s["gap_after_trunk"] for s in keep]),
            "other_by_kind": other}


def fam_cmp(a, b, n=14):
    """Trunk kernel families: device us a pass a vs b (the largest differences first) and the one-sided ones."""
    fa = a["kinds"][str(a["trunk"])]["fam_us"]; fb = b["kinds"][str(b["trunk"])]["fam_us"]
    na = a["kinds"][str(a["trunk"])]["fam_n"]; nb = b["kinds"][str(b["trunk"])]["fam_n"]
    keys = set(fa) | set(fb)
    d = sorted(((k, fa.get(k, 0.0), fb.get(k, 0.0)) for k in keys), key=lambda x: -abs(x[1] - x[2]))
    exp_a = sum(v for k, v in fa.items() if is_expert(k)); exp_b = sum(v for k, v in fb.items() if is_expert(k))
    oth_a = sum(v for k, v in fa.items() if not is_expert(k)); oth_b = sum(v for k, v in fb.items() if not is_expert(k))
    return {"expert_us": [exp_a, exp_b], "other_us": [oth_a, oth_b],
            "top": [(k, round(x, 1), round(y, 1), na.get(k, 0), nb.get(k, 0)) for k, x, y in d[:n]],
            "only_here": sorted(k for k in fa if k not in fb and fa[k] >= 1.0),
            "only_ref": sorted(k for k in fb if k not in fa and fb[k] >= 1.0)}


def stamps(a):
    path, W = a[1], wins(a)
    P = read_passes(path, W)
    out = {wn: analyse(P[wn]) for wn, _, _ in W}
    ref = None
    for i, x in enumerate(a):
        if x == "--ref" and i + 1 < len(a):
            nm, rest = a[i + 1].split(":", 1)
            rp, lo, hi = rest.rsplit(":", 2)
            RP = read_passes(rp, [(nm, int(lo), int(hi))])
            ref = (nm, analyse(RP[nm]))
            out["ref:" + nm] = ref[1]
    if ref and "error" not in ref[1]:
        for wn, _, _ in W:
            if "error" not in out[wn]: out[wn]["vs_ref"] = fam_cmp(out[wn], ref[1])
    return out


def fmt_stamps(o):
    L = []
    for wn, x in o.items():
        if "error" in x: L.append(f"{wn}: {x['error']}"); continue
        t = x["kinds"][str(x["trunk"])]
        kinds = "; ".join(f"{k} records: {v['passes']} passes, wall {v['wall_us']:.0f} busy {v['busy_us']:.0f} us, {v['dispatches']:.0f} dispatches"
                          for k, v in list(x["kinds"].items())[:5])
        oth = ", ".join(f"kind {k} {v['per_step']:.2f} a step {v['wall_us_step']:.0f} us" for k, v in x["other_by_kind"].items()
                        if v["per_step"] and v["per_step"] >= 0.02)
        L.append(f"{wn}: {x['steps']} steps ({x['steps_kept']} kept): period {x['period_us_med']:.0f} us median / {x['period_us_mean']:.0f} mean; "
                 f"trunk ({x['trunk']} records) wall {x['trunk_wall_us_mean']:.0f} busy {x['trunk_busy_us_mean']:.0f} us; other passes a step: {oth or 'none'}; "
                 f"card 0 idle {x['idle_us_mean']:.0f} us a step (median {x['idle_us_med']:.0f}; between passes {x['between_us_mean']:.0f}), trunk end -> next pass {x['gap_after_trunk_us_med']:.0f} us | kinds: {kinds}")
        topf = sorted(t["fam_us"].items(), key=lambda kv: -kv[1])[:10]
        L.append(f"{wn} trunk kernel families (device us a pass): " + ", ".join(f"{k} {v:.0f} ({t['fam_n'][k]:.0f})" for k, v in topf))
        for k, v in list(x["kinds"].items())[:4]:
            if k == str(x["trunk"]) or v["passes"] < 0.25 * x["steps"]: continue
            topk = sorted(v["fam_us"].items(), key=lambda kv: -kv[1])[:10]
            L.append(f"{wn} kind {k} kernel families (device us a pass): " + ", ".join(f"{fm} {us:.0f} ({v['fam_n'][fm]:.0f})" for fm, us in topk))
        if "vs_ref" in x:
            c = x["vs_ref"]
            L.append(f"{wn} trunk vs ref (device us a pass, here / ref): expert kernels {c['expert_us'][0]:.0f} / {c['expert_us'][1]:.0f} "
                     f"({c['expert_us'][0] - c['expert_us'][1]:+.0f}), the rest {c['other_us'][0]:.0f} / {c['other_us'][1]:.0f} "
                     f"({c['other_us'][0] - c['other_us'][1]:+.0f}); largest: " + ", ".join(f"{k} {x_:.0f}/{y:.0f} ({na:.0f}/{nb:.0f})" for k, x_, y, na, nb in c["top"])
                     + f"; only here: {', '.join(c['only_here']) or '-'}; only ref: {', '.join(c['only_ref']) or '-'}")
    return L


# ---------------------------------------------------------------------------------------------------------------- ops
def ops(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ds_opprof_draft import tables, cls, ORDER
    path, W = a[1], wins(a)
    refj = None
    if "--ref-json" in a:
        try: refj = json.load(open(a[a.index("--ref-json") + 1]))
        except (OSError, ValueError) as e: refj = {"error": str(e)}
    T = tables(path)
    out = {}
    for wn, lo, hi in W:
        res = {}
        for rk in sorted({r for _, r, _ in T}):
            mine = [(n, t) for n, r, t in T if r == rk]
            before = [t for n, t in mine if n <= lo]
            after = [t for n, t in mine if n <= hi]
            if not after: continue
            A, B = (before[-1] if before else {}), after[-1]
            d = {}
            for idx, (c, ms, op, geom, w, wt) in B.items():
                p = A.get(idx, (0, 0.0, "", "", "", 0.0))
                if c - p[0] > 0: d[idx] = (c - p[0], ms - p[1], op, geom, w, wt - p[5])
            if not d: continue
            heads = [i for i, x in d.items() if x[2].startswith("logits_gemm")]
            hd = min(heads) if heads else 1 << 30
            steps = d[min(d)][0]
            n = max(1, steps)
            byc = defaultdict(float); dr_ops = defaultdict(lambda: [0, 0.0]); tr_ops = defaultdict(lambda: [0, 0.0, 0.0])
            for i, (c, ms, op, g, w, wt) in d.items():
                if i > hd and op.startswith(("sample", "row_topk", "pair_argmax")):
                    byc["sample"] += ms; continue
                if i > hd:
                    byc["draft_" + cls(op)] += ms
                    key = op + " " + (re.sub(r"\.\d+\.weight$|\.weight$", "", w) if w != "-" else "-")
                    dr_ops[key][0] += c; dr_ops[key][1] += ms
                    continue
                byc[cls(op)] += ms   # the head (i == hd) counts in dense_gemm, as ds_opprof.py counts it
                tr_ops[op][0] += c; tr_ops[op][1] += ms; tr_ops[op][2] += wt
            tr = sum(v for k, v in byc.items() if not k.startswith("draft_") and k != "sample")
            dr = sum(v for k, v in byc.items() if k.startswith("draft_"))
            res[str(rk)] = {"steps": steps, "device_ms_step": sum(x[1] for x in d.values()) / n,
                            "wait_ms_step": sum(x[5] for x in d.values()) / n,
                            "trunk_ms_step": tr / n, "drafter_ms_step": dr / n,
                            "by_class_ms_step": {k: v / n for k, v in byc.items()},
                            "drafter_ops": {k: {"calls_step": v[0] / n, "ms_step": v[1] / n, "us_call": 1000 * v[1] / max(1, v[0])}
                                            for k, v in dr_ops.items()},
                            "trunk_ops": {k: {"calls_step": v[0] / n, "ms_step": v[1] / n, "wait_ms_step": v[2] / n}
                                          for k, v in tr_ops.items()}}
        if not res: out[wn] = {"error": "no complete profile table in the window"}; continue
        if refj and "summary" in refj:
            cmpc = {}
            for k in ORDER:
                here = [res[r]["by_class_ms_step"].get(k, 0.0) for r in res]
                there = [refj["summary"][r]["by_class_ms"].get(k, 0.0) / max(1, refj["summary"][r]["steps"]) for r in refj["summary"]]
                cmpc[k] = {"here_mean": statistics.mean(here), "ref_mean": statistics.mean(there),
                           "here_r0": res.get("0", {}).get("by_class_ms_step", {}).get(k, 0.0),
                           "ref_r0": refj["summary"].get("0", {}).get("by_class_ms", {}).get(k, 0.0) / max(1, refj["summary"].get("0", {}).get("steps", 1))}
            r0 = res.get("0")
            rs = refj["summary"].get("0", {}).get("steps", 1) or 1
            rops = {k: {"calls_step": v[0] / rs, "ms_step": v[1] / rs} for k, v in (refj.get("rank0_by_op") or {}).items()}
            opd = []
            if r0:
                for k in set(r0["trunk_ops"]) | set(rops):
                    h = r0["trunk_ops"].get(k, {}); t = rops.get(k, {})
                    opd.append((k, h.get("ms_step", 0.0), t.get("ms_step", 0.0), h.get("calls_step", 0.0), t.get("calls_step", 0.0)))
            opd.sort(key=lambda x: -abs(x[1] - x[2]))
            refsteps = [refj["summary"][r]["steps"] for r in refj["summary"]]
            out[wn] = {"ranks": res, "vs_ref": {"classes": cmpc,
                       "trunk_ms_step": [statistics.mean(res[r]["trunk_ms_step"] for r in res),
                                         statistics.mean(sum(refj["summary"][r]["by_class_ms"].values()) / max(1, refj["summary"][r]["steps"]) for r in refj["summary"])],
                       "rank0_ops": opd[:16],
                       "only_here": sorted(k for k in (r0 or {}).get("trunk_ops", {}) if k not in rops),
                       "only_ref": sorted(k for k in rops if k not in (r0 or {}).get("trunk_ops", {})),
                       "ref_steps": refsteps, "ref_wall_rank0": refj.get("wall_rank0")}}
        else:
            out[wn] = {"ranks": res}
    return out


def fmt_ops(o):
    L = []
    for wn, x in o.items():
        if "error" in x: L.append(f"{wn}: {x['error']}"); continue
        R = x["ranks"]
        for rk, r in sorted(R.items()):
            cl = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["by_class_ms_step"].items(), key=lambda kv: -kv[1]) if v >= 0.05)
            L.append(f"{wn} rank {rk}: {r['steps']} steps, device {r['device_ms_step']:.2f} + lane waits {r['wait_ms_step']:.2f} ms a step "
                     f"(PROFILED) = trunk {r['trunk_ms_step']:.2f} + drafter {r['drafter_ms_step']:.2f} (+ sampling); by class: {cl}")
        r0 = R.get("0") or next(iter(R.values()))
        dro = sorted(r0["drafter_ops"].items(), key=lambda kv: -kv[1]["ms_step"])
        L.append(f"{wn} rank 0 drafter ops (calls a step: ~1 = the context pass, ~0 = the query pass), ms a step: " + "; ".join(
            f"{k} {v['calls_step']:.2f}/step {v['us_call']:.0f} us = {v['ms_step']:.3f}" for k, v in dro[:16]))
        if "vs_ref" in x:
            v = x["vs_ref"]
            L.append(f"{wn} trunk vs NST 0 ref (PROFILED ms a step, mean of ranks here / ref): total {v['trunk_ms_step'][0]:.2f} / {v['trunk_ms_step'][1]:.2f} "
                     f"({v['trunk_ms_step'][0] - v['trunk_ms_step'][1]:+.2f}); " + ", ".join(
                         f"{k} {c['here_mean']:.2f}/{c['ref_mean']:.2f} ({c['here_mean'] - c['ref_mean']:+.2f})" for k, c in v["classes"].items()
                         if max(c["here_mean"], c["ref_mean"]) >= 0.05))
            L.append(f"{wn} rank 0 trunk ops vs ref (ms a step here/ref, calls a step here/ref), largest differences: " + "; ".join(
                f"{k} {h:.3f}/{t:.3f} ({h - t:+.3f}; {ch:.1f}/{ct:.1f})" for k, h, t, ch, ct in v["rank0_ops"])
                + f" | trunk ops only here: {', '.join(v['only_here']) or '-'} | only ref: {', '.join(v['only_ref']) or '-'}")
    return L


# ---------------------------------------------------------------------------------------------------------------- summary
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None


def pct(x, y):
    return None if x is None or not y else 100.0 * (x / y - 1.0)


def f(v, n=1):
    return "-" if v is None else format(v, ".%df" % n)


def fp(v):
    return "-" if v is None else "%+.1f %%" % v


def summary(a):
    def arg(k, d=None): return a[a.index(k) + 1] if k in a else d
    me, gz = rows(arg("--blocks")), rows(arg("--nst0"))
    good = [r for r in me if r.get("ok")]
    g = lambda r, *p: (lambda x: x)(_dig(r, p))
    def arm(name, *p): return mean([_dig(r["res"], p) for r in good if r.get("arm") == name])
    def nst0(*p): return mean([_dig(r["res"], p) for r in gz if r.get("ok") and r.get("arm") == "B"])
    def nst0_c1(k): return mean([next((x.get(k) for x in (_dig(r["res"], ("c1", "rows")) or []) if x.get("prompt") == "essay"), None)
                                 for r in gz if r.get("ok") and r.get("arm") == "B"])
    out = {"phases": {}}
    L = []
    for ph, msk, tpk, tsk in (("c8", ("c8", "mb", "ms_decode_step"), ("c8", "mb", "agg_dec_tps"), ("c8", "mb", "tok_per_decode_step")),
                              ("c4", ("c4", "mb", "ms_decode_step"), ("c4", "mb", "agg_dec_tps"), ("c4", "mb", "tok_per_decode_step")),
                              ("c1", ("essay", "probe", "ms_per_decode_step"), ("essay", "probe", "decode_tps"), ("essay", "probe", "tokens_per_step"))):
        P16, P4k = arm("A", *msk), arm("B", *msk)
        T16, T4k = arm("A", *tpk), arm("B", *tpk)
        K16, K4k = arm("A", *tsk), arm("B", *tsk)
        z16, z4k = arm("A", msk[0], "win", "zero_share"), arm("B", msk[0], "win", "zero_share")
        pr16, pr4k = arm("A", msk[0], "win", "probe_share"), arm("B", msk[0], "win", "probe_share")
        if ph == "c1": N0, NT = nst0_c1("ms_per_decode_step"), nst0_c1("decode_tps")
        else: N0, NT = nst0(ph, "ms_decode_step"), nst0(ph, "agg_dec_tps")
        # pairs (P4096 / P16) for the ms a step
        pairs = []
        for x, y in zip(me[0::2], me[1::2]):
            if x.get("ok") and y.get("ok") and x["arm"] != y["arm"]:
                ra, rb = (x, y) if x["arm"] == "A" else (y, x)
                va, vb = _dig(ra["res"], msk), _dig(rb["res"], msk)
                if va and vb: pairs.append(va / vb)
        ci = (T95.get(len(pairs) - 1, 1.96) * statistics.stdev(pairs) / math.sqrt(len(pairs))) if len(pairs) > 1 else None
        hit = (arm("B", msk[0], "win", "hit_pct_rank0"), arm("A", msk[0], "win", "hit_pct_rank0"))
        miss = (arm("B", msk[0], "win", "missed_step_rank0"), arm("A", msk[0], "win", "missed_step_rank0"))
        strm = (arm("B", msk[0], "win", "stream_MB_step"), arm("A", msk[0], "win", "stream_MB_step"))
        vram = arm("B", msk[0], "win", "vram_units_end")
        out["phases"][ph] = {"P4096_ms": P4k, "P16_ms": P16, "NST0_ms": N0, "P4096_vs_NST0_pct": pct(P4k, N0),
                             "P16_vs_P4096_pct": pct(P16, P4k), "pairs_P16_over_P4096": pairs, "ci95": ci,
                             "P4096_tps": T4k, "P16_tps": T16, "NST0_tps": NT, "P4096_tok_step": K4k, "P16_tok_step": K16,
                             "zero_share": [z4k, z16], "probe_share": [pr4k, pr16], "hit_pct": hit, "missed_step": miss,
                             "stream_MB_step": strm, "vram_units": vram}
        L.append(f"{ph}: ms a decode step P4096 {f(P4k, 2)} vs NST 0 {f(N0, 2)} ({fp(pct(P4k, N0))}); P16 {f(P16, 2)} vs P4096 ({fp(pct(P16, P4k))}; "
                 f"pairs {' '.join('%+.1f' % (100 * (p - 1)) for p in pairs) or '-'}; t95 +- {f(None if ci is None else 100 * ci, 1)} %) | "
                 f"tok/s P4096 {f(T4k)} P16 {f(T16)} NST 0 {f(NT)} ({fp(pct(T4k, NT))} / {fp(pct(T16, NT))}) | tok a step P4096 {f(K4k, 2)} P16 {f(K16, 2)} | "
                 f"steps at depth 0 P4096 {f(None if z4k is None else 100 * z4k)} % / P16 {f(None if z16 is None else 100 * z16)} %, probe steps "
                 f"{f(None if pr4k is None else 100 * pr4k)} / {f(None if pr16 is None else 100 * pr16)} % | experts (rank 0, P4096 / P16): hit {f(hit[0], 2)} / {f(hit[1], 2)} %, "
                 f"missed {f(miss[0], 2)} / {f(miss[1], 2)} a step, stream {f(strm[0])} / {f(strm[1])} MB a step, VRAM units {f(vram, 0)}")
    # NST 0's expert counters (bz dsd-d0 stats per phase, main image, gg_gemv stock; placement is not gg_gemv's)
    ref_ph = {}
    sp = arg("--nst0-stats")
    if sp and os.path.exists(sp):
        for l in open(sp):
            if l.startswith("PH {"):
                try: d = json.loads(l[3:]); ref_ph[d["phase"]] = d
                except ValueError: pass
    if ref_ph:
        L.append("NST 0 experts (bz dsd-d0, /stats per phase): " + "; ".join(
            f"{p}: hit {d.get('hit_pct_rank0')} %, stream {d.get('stream_MB_step')} MB a step, VRAM {d.get('vram_units_end')} units"
            for p, d in ref_ph.items() if p in ("probe", "c4", "c8")))
    out["nst0_experts"] = ref_ph
    # the c8 attribution
    S = None
    try: S = json.load(open(arg("--stamps"))) if arg("--stamps") else None
    except (OSError, ValueError): S = None
    O = None
    try: O = json.load(open(arg("--ops"))) if arg("--ops") else None
    except (OSError, ValueError): O = None
    c8 = out["phases"]["c8"]
    att = {}
    if c8["P4096_ms"] and c8["NST0_ms"]:
        att["clean_delta_ms"] = c8["P4096_ms"] - c8["NST0_ms"]
    if c8["P16_ms"] and c8["P4096_ms"] and c8["P16_tps"] and c8["P4096_tps"]:
        # probes: the throughput P16 loses against P4096, as ms per emitted token
        att["probe_tps_pct"] = pct(c8["P16_tps"], c8["P4096_tps"])
        att["probe_ms_per_token"] = [1000.0 / c8["P16_tps"] * 8, 1000.0 / c8["P4096_tps"] * 8]
    if S:
        g2, g0 = S.get("g2") or {}, S.get("g0") or {}
        ref = next((v for k, v in S.items() if k.startswith("ref:")), None)
        ctx_kind = None
        if "error" not in g2 and not [k for k, v in (g2.get("other_by_kind") or {}).items() if (v.get("per_step") or 0) >= 0.5]:
            att["ctx_note"] = ("no second pass kind at >= 0.5 a step in the gg_gemv 2 window: the drafter's context pass is not played "
                               "from a tape (stamps time played passes only), so its time is inside 'between passes'")
        if "error" not in g2 and g2.get("other_by_kind"):
            ctx_kind = max(g2["other_by_kind"].items(), key=lambda kv: kv[1]["per_step"] or 0)
            att["ctx_pass_kind"] = ctx_kind[0]; att["ctx_pass_per_step"] = ctx_kind[1]["per_step"]
            att["ctx_pass_wall_ms"] = (ctx_kind[1]["wall_us_step"] or 0) / 1000.0
            att["g2_period_ms"] = (g2.get("period_us_mean") or 0) / 1000.0
            att["g2_idle_ms"] = (g2.get("idle_us_mean") or 0) / 1000.0
        if ref and "error" not in ref and "error" not in g0 and g0.get("vs_ref"):
            v = g0["vs_ref"]
            att["g0_period_ms"] = (g0.get("period_us_mean") or 0) / 1000.0; att["ref_period_ms"] = (ref.get("period_us_mean") or 0) / 1000.0
            att["trunk_wall_delta_ms"] = ((g0.get("trunk_wall_us_mean") or 0) - (ref.get("trunk_wall_us_mean") or 0)) / 1000.0
            att["expert_dev_delta_ms"] = (v["expert_us"][0] - v["expert_us"][1]) / 1000.0
            att["other_dev_delta_ms"] = (v["other_us"][0] - v["other_us"][1]) / 1000.0
            att["idle_delta_ms"] = ((g0.get("idle_us_mean") or 0) - (ref.get("idle_us_mean") or 0)) / 1000.0
            att["between_delta_ms"] = ((g0.get("between_us_mean") or 0) - (ref.get("between_us_mean") or 0)) / 1000.0
            att["g0_between_ms"] = (g0.get("between_us_mean") or 0) / 1000.0; att["ref_between_ms"] = (ref.get("between_us_mean") or 0) / 1000.0
            att["stamped_delta_ms"] = att["g0_period_ms"] - att["ref_period_ms"]
            g0ctx = max(g0["other_by_kind"].items(), key=lambda kv: kv[1]["per_step"] or 0) if g0.get("other_by_kind") else None
            att["g0_ctx_pass_wall_ms"] = (g0ctx[1]["wall_us_step"] or 0) / 1000.0 if g0ctx else 0.0
    if O:
        for wn in ("g2", "g0"):
            x = O.get(wn) or {}
            if "ranks" not in x: continue
            R = x["ranks"]
            att[f"prof_{wn}_drafter_ms"] = statistics.mean(r["drafter_ms_step"] for r in R.values())
            if "vs_ref" in x:
                c = x["vs_ref"]["classes"]
                att[f"prof_{wn}_experts_delta_ms"] = c["experts"]["here_mean"] - c["experts"]["ref_mean"]
                att[f"prof_{wn}_trunk_delta_ms"] = x["vs_ref"]["trunk_ms_step"][0] - x["vs_ref"]["trunk_ms_step"][1]
    if "clean_delta_ms" in att and (att.get("ctx_pass_wall_ms") is not None or att.get("expert_dev_delta_ms") is not None):
        att["remainder_ms"] = att["clean_delta_ms"] - sum(att.get(k) or 0.0 for k in ("ctx_pass_wall_ms", "expert_dev_delta_ms", "between_delta_ms"))
    out["attribution_c8"] = att
    L.append("c8 attribution: clean P4096 - NST 0 %s ms a step (%s vs %s); probes: P16 aggregate tok/s %s vs P4096 (%s vs %s ms per 8 tokens); "
             "context pass %s ms a step on card 0 (stamped, gg_gemv 2; pass kind %s, %s a step; at gg_gemv 0 %s ms); "
             "expert-miss: trunk expert kernels %s ms device a step vs NST 0 (stamped, gg_gemv 0 = the same forms), the other trunk kernels %s ms, trunk pass wall %s ms; "
             "between passes (host / sync) %s ms (%s vs %s ms a step); remainder of the clean difference after context pass + expert kernels + between %s ms | "
             "stamped step %s vs %s ms (%s) | PROFILED cross-check: drafter ops %s ms device a step (gg_gemv 2; %s at 0), trunk experts class %s ms vs NST 0, trunk total %s ms (gg_gemv 0)" % (
                 f(att.get("clean_delta_ms"), 2), f(c8["P4096_ms"], 2), f(c8["NST0_ms"], 2), fp(att.get("probe_tps_pct")),
                 f((att.get("probe_ms_per_token") or [None, None])[0], 1), f((att.get("probe_ms_per_token") or [None, None])[1], 1),
                 f(att.get("ctx_pass_wall_ms"), 2), att.get("ctx_pass_kind", "-"), f(att.get("ctx_pass_per_step"), 2), f(att.get("g0_ctx_pass_wall_ms"), 2),
                 f(att.get("expert_dev_delta_ms"), 2), f(att.get("other_dev_delta_ms"), 2), f(att.get("trunk_wall_delta_ms"), 2),
                 f(att.get("between_delta_ms"), 2), f(att.get("g0_between_ms"), 2), f(att.get("ref_between_ms"), 2), f(att.get("remainder_ms"), 2),
                 f(att.get("g0_period_ms"), 2), f(att.get("ref_period_ms"), 2), "-" if att.get("stamped_delta_ms") is None else "%+.2f" % att["stamped_delta_ms"],
                 f(att.get("prof_g2_drafter_ms"), 2), f(att.get("prof_g0_drafter_ms"), 2),
                 f(att.get("prof_g0_experts_delta_ms"), 2), f(att.get("prof_g0_trunk_delta_ms"), 2))
             + (" | NOTE " + att["ctx_note"] if att.get("ctx_note") else ""))
    return out, L


def _dig(x, p):
    for k in p:
        if not isinstance(x, dict): return None
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else (x if isinstance(x, list) else None)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("stamps", "ops", "summary"):
        print(__doc__); return 2
    label = a[a.index("--label") + 1] if "--label" in a else ""
    if a[0] == "stamps":
        o = stamps(a); L = fmt_stamps(o)
    elif a[0] == "ops":
        o = ops(a); L = fmt_ops(o)
    else:
        o, L = summary(a)
    if "--json" in a:
        with open(a[a.index("--json") + 1], "w") as fh: json.dump(o, fh)
    for l in L: print(f"D0AP {label} {a[0]} {l}")
    print("D0AP " + json.dumps({"label": label, "mode": a[0]})[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
