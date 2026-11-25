#!/usr/bin/env python3
"""ds_opprof_nc.py (12.11.2026, queue-ar/dsmain/nc, ds-main-port NC-profile builder) -- a COPY of queue-ar/dsmain/ds_opprof.py
(md5 in the item header; the original is not changed) for dsm-ncprof.sh's profiled start, with:
  * `docker logs -t` input (the receipt time leading each line is stripped and kept) and .gz files;
  * windows by TIME (NAME:T0:T1, epoch s): per rank the last complete table ("total" line) at or before T0 is subtracted
    from the last one at or before T1; the window's STEPS are counted from the step log (RADIANCE_LOG_STEPS=1: the last
    "step K" line before a table's header is the step the table stands behind; RADIANCE_PROFILE_EVERY divides K), not from
    the calls of the lowest-index op (lever BI runs a mixed step's dense launches in segments, so op calls != steps);
  * two classes of the DSpark drafter by op instance (the coordinator's 12.11. scope): dspark_ctx = the drafter's context
    pass (arch/deepseek41/deepseek41.cpp dsp_context: rope_table, the dspark.fc GEMM, the enc norm, per draft layer the kv
    GEMM / norm / rope / round trip / store -- the op instances after the trunk's logits op and before the drafter's own
    embed_lookup), dspark_query = the drafter's query pass (that embed_lookup up to its last dspark.* / draft-layer op);
  * breakdowns (ms a step): mhc by op (mhc_mix, mhc_collapse_rmsnorm, mhc_post_collapse_rmsnorm, ...), Engram (the
    engram_kv GEMM -- counted in dense_gemm in the class table -- vs embed_lookup_planes / ngram_ids / token_map /
    stream_gate_add), the indexer by op (index_score, index_topk, index_candidates, ...), dspark_ctx by op;
  * lane waits split: the STAGER waits ('wait us' of each trunk layer's first expert op: the prefill stager's copy waits,
    dsm-pipeprof.sh's header) vs the rest;
  * wall from the step log (a step's wall = the next step line's receipt time - its own; steps followed by a table dump,
    K % --every == 0, left out) or given (--wall NAME:MS); host remainder = wall - device - lane waits;
  * differences: --diff NEW:W2:W1 = (W2's totals - W2's decode steps x W1's per-decode-step) / W2's non-decode steps;
    --intervals NEW:W2 = inside W2, the dump intervals with exactly one non-decode step minus their decode steps x the
    mean of W2's decode-only intervals (both ESTIMATES, labelled).
Classes otherwise ds_opprof.py's: experts, moe_other, gather_ar, collective, attention, indexer, mhc, dense_gemm, embed_engram,
norm_elem, other. PROFILED: every number is from synchronised steps; shares and us a call are the reading, not a speed.

  ds_opprof_nc.py LOG[.gz] --win NAME:T0:T1 [...] [--wall NAME:MS] [--diff NEW:W2:W1] [--intervals NEW:W2] [--every 8]
                  [--label L] [--json OUT]
  ds_opprof_nc.py selftest
Standard library only."""
import calendar, gzip, json, re, statistics, sys, time
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HDR = re.compile(r"per-op timing, rank (\d+)")
ORDER = ["experts", "moe_other", "gather_ar", "collective", "attention", "indexer", "mhc", "dense_gemm", "embed_engram",
         "norm_elem", "other", "dspark_ctx", "dspark_query"]
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")
BLK = re.compile(r"^blk\.(\d+)\.")


def cls(op):   # ds_opprof.py's, unchanged
    o = op.lower()
    if o.startswith("moe_exl3") or "exl3" in o or o.startswith("moe_gemm"): return "experts"
    if o.startswith("moe_gather_all_reduce"): return "gather_ar"
    if o.startswith(("all_reduce", "all_gather", "ar_", "reduce_scatter")) or "all_reduce" in o: return "collective"
    if o.startswith(("moe_", "router", "swiglu")): return "moe_other"
    if o.startswith("index_"): return "indexer"
    if o.startswith("mhc_"): return "mhc"
    if "attn" in o or o.startswith(("kv_store", "entry_store", "ds41_rope", "rope_rows", "rmsnorm_rope", "qsa_",
                                    "softmax_pool", "latent")): return "attention"
    if o.startswith(("gemm", "logits")): return "dense_gemm"
    if o.startswith(("embed_lookup", "ngram", "token_map", "stream_gate")): return "embed_engram"
    if o.startswith(("rmsnorm", "fill", "cast", "rope_table", "norm")): return "norm_elem"
    return "other"


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def ts_split(line):
    m = TSRE.match(line)
    if not m: return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z": t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


def parse(path):
    """steps [(t, K, phase, n_tok)], tables {rank: [(t_end, K, {idx: (calls, ms, wait_ms)})]}, meta {idx: (op, geom, weight)}."""
    steps, tables, meta = [], defaultdict(list), {}
    cur, rank, curK, lastt = None, None, None, None
    with opn(path) as f:
        for l in f:
            t, rest = ts_split(l.rstrip("\n"))
            if t is not None: lastt = t
            if cur is None and "n_seq=" in rest:
                m = STEP.search(rest)
                if m and t is not None:
                    steps.append((t, int(m.group(1)), m.group(2), int(m.group(4)))); curK = int(m.group(1))
                    continue
            m = HDR.search(rest)
            if m:
                cur, rank, hk = {}, int(m.group(1)), curK
                continue
            if cur is None: continue
            r = ROW.match(rest)
            if r:
                idx = int(r.group(9))
                if idx not in meta: meta[idx] = (r.group(7), r.group(8), r.group(10))
                c = int(r.group(1))
                cur[idx] = (c, float(r.group(2)), float(r.group(6)) * c / 1000.0)
            elif rest.strip().startswith("total"):
                tables[rank].append((t if t is not None else lastt, hk, cur)); cur = None
    return steps, tables, meta


def classify(meta):
    """idx -> class, with the DSpark drafter's context / query passes split off by op instance; and the stager-wait ops."""
    ce = {i: cls(op) for i, (op, g, w) in meta.items()}
    lg = sorted(i for i, (op, g, w) in meta.items() if op.startswith("logits"))
    info = {"trunk_end": None, "ctx": None, "query": None, "ctx_ok": False}
    if lg:
        te = lg[0]
        nl = [int(m.group(1)) for i, (op, g, w) in meta.items() if i < te for m in [BLK.match(w)] if m]
        ntr = (max(nl) + 1) if nl else 10 ** 6
        dr = [i for i, (op, g, w) in meta.items() if i > te and (w.startswith("dspark.") or (BLK.match(w) and int(BLK.match(w).group(1)) >= ntr))]
        if dr:
            last = max(dr)
            qs = min((i for i, (op, g, w) in meta.items() if te < i <= last and op.startswith("embed_lookup")), default=None)
            ce_hi = qs if qs is not None else last + 1
            ctx = [i for i in meta if te < i < ce_hi]
            qry = [i for i in meta if qs is not None and qs <= i <= last]
            for i in ctx: ce[i] = "dspark_ctx"
            for i in qry: ce[i] = "dspark_query"
            info.update(trunk_end=te, ctx=[min(ctx), max(ctx)] if ctx else None, query=[qs, last] if qry else None, n_trunk_layers=ntr,
                        ctx_ok=any(meta[i][2].startswith("dspark.fc") for i in ctx))
    first = {}
    for i, (op, g, w) in meta.items():
        m = BLK.match(w)
        if m and ce[i] == "experts":
            L = int(m.group(1))
            if L not in first or i < first[L]: first[L] = i
    return ce, first, info


def sub_of(i, meta, ce):
    """the coordinator's breakdowns: (group, key) or None."""
    op, g, w = meta[i]
    c = ce[i]
    if c == "mhc": return ("mhc", op)
    if c == "indexer": return ("indexer", op)
    if "engram_kv" in w: return ("engram", "engram_kv_gemm")
    if op.startswith(("embed_lookup_planes", "ngram", "token_map", "stream_gate")): return ("engram", op)
    if c == "dspark_ctx": return ("dspark_ctx", op + ("(" + w + ")" if w != "-" else ""))
    return None


def delta(A, B):
    d = {}
    for idx, (c, ms, wt) in B.items():
        p = A.get(idx, (0, 0.0, 0.0))
        if c - p[0] > 0: d[idx] = (c - p[0], ms - p[1], wt - p[2])
    return d


def summarize(d, meta, ce, first, nsteps):
    n = max(1, nsteps)
    byc, sub, wait, sw, perL = defaultdict(float), defaultdict(float), 0.0, 0.0, {}
    ctx_calls = 0
    for idx, (c, ms, wt) in d.items():
        byc[ce[idx]] += ms
        wait += wt
        s = sub_of(idx, meta, ce)
        if s: sub["%s:%s" % s] += ms
        if ce[idx] == "dspark_ctx" and meta[idx][2].startswith("dspark.fc"): ctx_calls += c
    for L, idx in first.items():
        if idx in d:
            sw += d[idx][2]; perL[L] = d[idx][2] / n
    dev = sum(byc.values())
    return {"steps": nsteps, "device_ms": dev / n, "wait_ms": wait / n, "stager_wait_ms": sw / n, "other_wait_ms": (wait - sw) / n,
            "by_class_ms": {k: v / n for k, v in byc.items()}, "sub_ms": {k: v / n for k, v in sub.items()},
            "ctx_calls_per_step": ctx_calls / n, "stager_wait_by_layer": perL}


def combine(x, y, fx=1.0, fy=1.0):
    """fx * x + fy * y over the per-step dicts (x, y: summarize() results)."""
    out = {}
    for k in ("device_ms", "wait_ms", "stager_wait_ms", "other_wait_ms", "ctx_calls_per_step"):
        out[k] = fx * x.get(k, 0.0) + fy * y.get(k, 0.0)
    for k in ("by_class_ms", "sub_ms", "stager_wait_by_layer"):
        ks = set(x.get(k, {})) | set(y.get(k, {}))
        out[k] = {z: fx * x.get(k, {}).get(z, 0.0) + fy * y.get(k, {}).get(z, 0.0) for z in ks}
    return out


def ranked(s, wall):
    parts = sorted(((k, v) for k, v in s["by_class_ms"].items() if abs(v) >= 0.005), key=lambda kv: -kv[1])
    parts += [("stager_waits", s["stager_wait_ms"]), ("other_lane_waits", s["other_wait_ms"])]
    base = wall if wall else (s["device_ms"] + s["wait_ms"])
    if wall: parts.append(("host_remainder", wall - s["device_ms"] - s["wait_ms"]))
    parts.sort(key=lambda kv: -kv[1])
    return "; ".join("%s %.2f ms (%.0f %%)" % (k, v, 100.0 * v / base if base else 0) for k, v in parts) + \
        " | device %.2f + lane waits %.2f%s ms a step" % (s["device_ms"], s["wait_ms"], (" of wall %.2f" % wall) if wall else " (no wall)")


def breakdown(s):
    g = defaultdict(list)
    for k, v in sorted(s["sub_ms"].items(), key=lambda kv: -kv[1]):
        a, b = k.split(":", 1)
        g[a].append("%s %.3f" % (b, v))
    return " | ".join("%s: %s" % (a, ", ".join(v)) for a, v in g.items())


def step_mix(steps, k0, k1):
    ss = [s for s in steps if k0 < s[1] <= k1]
    mix = defaultdict(int)
    for s in ss: mix[s[2]] += 1
    return ss, dict(mix)


def walls(steps, ss, every):
    """{phase: [wall ms]} for the steps ss (the next step line's time - own), steps followed by a dump left out."""
    pos = {s[1]: i for i, s in enumerate(steps)}
    out = defaultdict(list)
    for s in ss:
        i = pos.get(s[1])
        if i is None or i + 1 >= len(steps) or (every and s[1] % every == 0): continue
        dt = steps[i + 1][0] - s[0]
        if 0 <= dt <= 5.0: out[s[2]].append(1000.0 * dt)
    return out


def decode_mean(tl, AB, steps, meta, ce, first):
    """the per-step mean of the decode-only dump intervals between the tables AB = (A, B): (mean, intervals, n)."""
    A, B = AB
    tl = [x for x in tl if x[0] is not None and A[0] <= x[0] <= B[0] and x[1] is not None]
    ivl = []
    for p, q in zip(tl, tl[1:]):
        ss, mix = step_mix(steps, p[1], q[1])
        ivl.append((mix, summarize(delta(p[2], q[2]), meta, ce, first, q[1] - p[1]), q[1] - p[1], ss))
    decs = [(x, n) for mix, x, n, ss in ivl if set(mix) == {"decode"}]
    if not decs: return None, ivl, 0
    tot = sum(n for x, n in decs)
    dmean = combine({}, {}, 0, 0)
    for x, n in decs: dmean = combine(dmean, x, 1.0, n / tot)
    return dmean, ivl, len(decs)


def detail(d, meta, ce, nsteps):
    """ds_opprof.py's rank tail: expert ops by op and by layer (us a call), the top ops (ms a step, us a call, lane wait)."""
    n = max(1, nsteps)
    byop, per, byname = defaultdict(lambda: [0, 0.0]), [], defaultdict(lambda: [0, 0.0, 0.0])
    for idx, (c, ms, wt) in d.items():
        op, g, w = meta[idx]
        if ce[idx] == "experts":
            byop[op][0] += c; byop[op][1] += ms
            m = BLK.match(w)
            per.append((1000 * ms / c, op, m.group(1) if m else w))
        k = op if ce[idx] not in ("dspark_ctx", "dspark_query") else ce[idx] + ":" + op
        byname[k][0] += c; byname[k][1] += ms; byname[k][2] += wt
    out = ["expert ops: " + ", ".join("%s %.1f ms / %d calls = %.1f us" % (op, v[1], v[0], 1000 * v[1] / v[0]) for op, v in sorted(byop.items(), key=lambda kv: -kv[1][1]))]
    for op in sorted(byop):
        us = sorted(p[0] for p in per if p[1] == op)
        if not us: continue
        slow = ", ".join("blk %s %.0f" % (p[2], p[0]) for p in sorted((p for p in per if p[1] == op), key=lambda x: -x[0])[:5])
        out.append("%s us a call by layer: min %.1f p25 %.1f p50 %.1f p75 %.1f max %.1f (slowest: %s)" % (
            op, us[0], us[len(us) // 4], us[len(us) // 2], us[3 * len(us) // 4], us[-1], slow))
    out.append("top ops (ms a step, us a call, lane wait ms a step): " + "; ".join("%s %.3f ms %.1f us w %.3f" % (op, v[1] / n, 1000 * v[1] / v[0], v[2] / n)
                                                                              for op, v in sorted(byname.items(), key=lambda kv: -kv[1][1])[:12]))
    return " | ".join(out)


def main(argv):
    if argv and argv[0] == "selftest": return selftest()
    if not argv: print(__doc__); return 2
    path, wins, wall, diffs, ivs, every, label, jout = argv[0], [], {}, [], [], 8, "", None
    i = 1
    while i < len(argv):
        k = argv[i]
        if k == "--win": wins.append(argv[i + 1]); i += 2
        elif k == "--wall": n, v = argv[i + 1].rsplit(":", 1); wall[n] = float(v); i += 2
        elif k == "--diff": diffs.append(argv[i + 1]); i += 2
        elif k == "--intervals": ivs.append(argv[i + 1]); i += 2
        elif k == "--every": every = int(argv[i + 1]); i += 2
        elif k == "--label": label = argv[i + 1]; i += 2
        elif k == "--json": jout = argv[i + 1]; i += 2
        else: i += 1
    steps, tables, meta = parse(path)
    ce, first, info = classify(meta)
    out = {"label": label, "profiled": True, "drafter": info, "windows": {}}
    lines, bad = [], []
    if not tables: print("DSOPNC %s: VIRHE no complete profile table in %s" % (label, path)); return 1
    W = {}
    for w in wins:
        nm, t0, t1 = w.split(":"); t0, t1 = float(t0), float(t1)
        per, ranks_ok = {}, []
        for rk, tl in sorted(tables.items()):
            A = [x for x in tl if x[0] is not None and x[0] <= t0]
            B = [x for x in tl if x[0] is not None and x[0] <= t1]
            if not B or not A or B[-1] is A[-1] or A[-1][1] is None or B[-1][1] is None: continue
            per[rk] = (A[-1], B[-1])
            ranks_ok.append(rk)
        if len(ranks_ok) < 4: bad.append("%s: tables on %d of 4 ranks" % (nm, len(ranks_ok)))
        if not per: continue
        rk0 = min(per)
        k0, k1 = per[rk0][0][1], per[rk0][1][1]
        ss, mix = step_mix(steps, k0, k1)
        if not ss: bad.append("%s: no step-log line between the tables (K %s..%s)" % (nm, k0, k1))
        if any(per[r][0][1] != k0 or per[r][1][1] != k1 for r in per): bad.append("%s: ranks' tables stand behind different steps" % nm)
        if every and (k0 % every or k1 % every): bad.append("%s: table steps %s / %s not multiples of %s" % (nm, k0, k1, every))
        wl = walls(steps, ss, every)
        res = {"k0": k0, "k1": k1, "steps": k1 - k0, "mix": mix, "n_tok_mean": round(statistics.mean([s[3] for s in ss]), 1) if ss else None,
               "wall_ms_by_phase": {p: round(statistics.mean(v), 2) for p, v in wl.items() if v}, "ranks": {}}
        dd = {}
        for rk, (A, B) in per.items():
            d = delta(A[2], B[2]); dd[rk] = d
            s = summarize(d, meta, ce, first, k1 - k0)
            calls0 = d[min(d)][0] if d else 0
            s["op_call_steps"] = calls0
            res["ranks"][rk] = s
            if rk == min(per):
                res["rank0_detail"] = detail(d, meta, ce, k1 - k0)
                sw = sorted(s["stager_wait_by_layer"].items(), key=lambda kv: -kv[1])
                res["stager_wait_layers"] = "%d layers with a stager wait > 0.1 ms a step; top: %s" % (
                    sum(1 for L, v in sw if v > 0.1), ", ".join("blk %s %.2f ms" % (L, v) for L, v in sw[:6]))
        W[nm] = (res, dd, per)
        out["windows"][nm] = res
    # the given walls, then the derived windows
    def finish(nm, res, wl_ms):
        for rk, s in res["ranks"].items():
            s["wall_ms"] = wl_ms
            s["host_remainder_ms"] = (wl_ms - s["device_ms"] - s["wait_ms"]) if wl_ms else None
            s["ranked"] = ranked(s, wl_ms)
        r0 = res["ranks"].get(min(res["ranks"])) if res["ranks"] else None
        if r0:
            res["breakdown_line"] = " || ".join("rank %s: %s | dspark_ctx %.3f ms a step, %.2f context passes a step over %s rows a step" % (
                rk, breakdown(s), s["by_class_ms"].get("dspark_ctx", 0.0), s["ctx_calls_per_step"], res.get("rows")) for rk, s in sorted(res["ranks"].items()))
            top = sorted(r0["by_class_ms"].items(), key=lambda kv: -kv[1])[:3]
            res["top0"] = ", ".join("%s %.1f ms" % kv for kv in top)
    for nm, (res, dd, per) in W.items():
        wl = wall.get(nm)
        if wl is None:
            ph = [p for p in res["wall_ms_by_phase"]]
            if len(ph) == 1: wl = res["wall_ms_by_phase"][ph[0]]
        res["rows"] = res.get("n_tok_mean")
        finish(nm, res, wl)
    for spec in diffs:
        new, w2, w1 = spec.split(":")
        if w2 not in W or w1 not in W: bad.append("diff %s: window missing" % new); continue
        r2, r1 = W[w2][0], W[w1][0]
        nd = sum(v for k, v in r2["mix"].items() if k != "decode"); ndec = r2["mix"].get("decode", 0)
        if not nd: bad.append("diff %s: no non-decode step in %s" % (new, w2)); continue
        res = {"estimate": "W2 totals - W2 decode steps x W1 per decode step, over W2's non-decode steps", "nd_steps": nd, "decode_steps": ndec, "ranks": {}}
        w1clean = not (r1["mix"].get("MIXED", 0) or r1["mix"].get("prefill", 0))
        if not w1clean:
            res["note"] = "W1 was not decode only (%s): its decode-only dump intervals are the reference" % r1["mix"]
        for rk in r2["ranks"]:
            if rk not in r1["ranks"]: continue
            ref = r1["ranks"][rk] if w1clean else decode_mean(tables[rk], W[w1][2][rk], steps, meta, ce, first)[0]
            if ref is None: bad.append("diff %s: rank %s has no decode-only reference in %s" % (new, rk, w1)); continue
            s = combine(r2["ranks"][rk], ref, (r2["steps"]) / nd, -ndec / nd)
            res["ranks"][rk] = s
        k0, k1 = r2["k0"], r2["k1"]
        ss, _ = step_mix(steps, k0, k1)
        ndw = [v for p, v in walls(steps, ss, every).items() if p != "decode" for v in v]
        res["rows"] = round(statistics.mean([s[3] for s in ss if s[2] != "decode"]), 1) if any(s[2] != "decode" for s in ss) else None
        finish(new, res, statistics.mean(ndw) if ndw else None)
        res["wall_note"] = "the non-decode steps' own wall (step log, %d of %d steps without a dump after them)" % (len(ndw), nd)
        out["windows"][new] = res
    for spec in ivs:
        new, w2 = spec.split(":")
        if w2 not in W: bad.append("intervals %s: window missing" % new); continue
        res2, dd, per = W[w2]
        res = {"estimate": "inside %s: the dump intervals with one non-decode step minus their decode steps x the decode-interval mean" % w2, "ranks": {}}
        nci = deci = 0
        rows = []
        for rk, (A, B) in per.items():
            dmean, ivl, ndi = decode_mean(tables[rk], (A, B), steps, meta, ce, first)
            if dmean is None: continue
            ones = []
            for mix, x, n, ss in ivl:
                if sum(v for k, v in mix.items() if k != "decode") == 1:
                    nde = mix.get("decode", 0)
                    ones.append(combine(x, dmean, n, -nde))
                    if rk == min(per): rows += [s[3] for s in ss if s[2] != "decode"]
            if not ones: continue
            m = combine({}, {}, 0, 0)
            for x in ones: m = combine(m, x, 1.0, 1.0 / len(ones))
            res["ranks"][rk] = m
            if rk == min(per): nci, deci = len(ones), ndi
        res["intervals_one_newcomer"], res["intervals_decode"] = nci, deci
        res["rows"] = round(statistics.mean(rows), 1) if rows else None
        wd = out["windows"].get(new.replace("_iv", "_diff"), {})
        finish(new, res, wd.get("ranks", {}).get(min(wd["ranks"]), {}).get("wall_ms") if wd.get("ranks") else None)
        if not res["ranks"]: bad.append("intervals %s: no usable interval" % new)
        out["windows"][new] = res
    for nm, res in out["windows"].items():
        rs = res.get("ranks") or {}
        lines.append("%s%s: %s" % (nm, " (%s)" % res["estimate"] if res.get("estimate") else " (%s steps %s, rows %s)" % (res.get("steps"), res.get("mix"), res.get("rows")),
                     " | ".join("rank %s: %s" % (rk, s.get("ranked")) for rk, s in sorted(rs.items()))))
    if not info.get("ctx_ok"): lines.append("dspark_ctx: NOT FOUND (no dspark.fc op after the trunk's logits op) -- its ms are not split off")
    out["guard"] = bad
    msg = "DSOPNC %s (PROFILED, not a speed; drafter ctx ops #%s, query #%s): %s%s" % (label, info.get("ctx"), info.get("query"), " || ".join(lines),
                                                                                      (" -- VIRHE " + "; ".join(bad)) if bad else "")
    print(msg)
    out["line"] = msg
    if jout:
        with open(jout, "w") as f: json.dump(out, f, default=str)
    return 0 if not bad else 2


def selftest():
    import shutil, tempfile
    d = tempfile.mkdtemp(prefix="dsopnc-")
    try: return _selftest(d)
    finally: shutil.rmtree(d, ignore_errors=True)   # DRYRUN writes nothing that stays


def _selftest(d):
    import os
    def ts(t):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(t))) + ".%09dZ" % int(round((t - int(t)) * 1e9))
    ops = [(1, "embed_lookup_ggml", "token_embd.weight"), (2, "mhc_mix", "blk.0.hc_attn_fn.weight"), (3, "mhc_collapse_rmsnorm", "blk.0.attn_norm.weight"),
           (4, "index_score", "blk.0.indexer.weight"), (5, "index_topk", "-"), (6, "ngram_ids", "blk.0.engram_mult"),
           (7, "gemm_nt_ggml", "blk.0.engram_kv.weight"), (8, "router_scored_topk", "blk.0.glm_router.correction"),
           (9, "moe_exl3_gate_up", "blk.0.ffn_gate_up_exps.0.weight"), (10, "moe_exl3_down", "blk.0.ffn_down_exps.0.weight"),
           (11, "all_reduce", "-"), (12, "moe_exl3_gate_up", "blk.1.ffn_gate_up_exps.0.weight"), (13, "mhc_post_collapse_rmsnorm", "output_norm.weight"),
           (14, "logits_gemm_ggml", "output.weight"), (15, "rope_table", "-"), (16, "gemm_nt", "dspark.fc.weight"),
           (17, "kv_store", "-"), (18, "embed_lookup_ggml", "token_embd.weight"), (19, "gemm_nt", "blk.2.attn_q_a.weight"),
           (20, "dflash_select", "dspark.markov_w1.weight"), (21, "sub_i32", "-")]
    # per call ms and wait us by op idx; the drafter's ctx (#15-17) and query (#18-20)
    per_ms = {i: 0.1 * i for i, _, _ in ops}
    per_w = {i: (50.0 if i in (9, 12) else 1.0) for i, _, _ in ops}
    def table(rank, calls, t):
        L = ["%s I[%d] per-op timing, rank %d. Not comparable with an unprofiled run:" % (ts(t), rank, rank)]
        for i, op, w in ops:
            c = calls
            L.append("%s   %10d %12.3f %10.2f   dev          - %10.2f  %s  n=1  #%d %s" % (ts(t), c, per_ms[i] * c, per_ms[i] * 1000, per_w[i], op, i, w))
        L.append("%s   %10s %12.3f   (waited %.3f ms besides)" % (ts(t), "total", 1.0, 0.0))
        return L
    lines, t, K = [], 1794480000.0, 0
    def step(ph, n_tok):
        nonlocal t, K
        K += 1
        lines.append("%s I step %d %s n_seq=1 n_tok=%d  [0] slot=0 q=%d ctx=0" % (ts(t), K, ph, n_tok, n_tok)); t += 0.5
        if K % 8 == 0:
            for r in range(4): lines.extend(table(r, K, t))
            t += 0.01
    for _ in range(24): step("prefill", 1099)
    t_alone = t
    for _ in range(16): step("decode", 8)
    t_w1 = t
    for k in range(16): step("MIXED" if k == 3 else "decode", 1107 if k == 3 else 8)
    t_w2 = t
    p = os.path.join(d, "e.log.gz")
    with gzip.open(p, "wt") as f: f.write("\n".join(lines) + "\n")
    js = os.path.join(d, "o.json")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main([p, "--win", "alone:%f:%f" % (1794480000.0 + 4.1, t_alone), "--win", "w1:%f:%f" % (t_alone, t_w1), "--win", "w2:%f:%f" % (t_w1, t_w2),
                   "--wall", "alone:600", "--diff", "nc_diff:w2:w1", "--intervals", "nc_iv:w2", "--every", "8", "--json", js])
    o = json.load(open(js))
    a0 = o["windows"]["alone"]["ranks"]["0"]
    nd = o["windows"]["nc_diff"]["ranks"]["0"]
    # every op is called once a step at 0.1 x idx ms, so per step: ctx = 1.5 + 1.6 + 1.7, query = 1.8 + 1.9 + 2.0; the
    # stager waits = #9 + #12 = 0.1 ms a step; the difference of identical per-step tables is the same per-step table
    ok = rc == 0 and o["drafter"]["ctx"] == [15, 17] and o["drafter"]["query"] == [18, 20] and o["drafter"]["ctx_ok"] \
        and abs(a0["by_class_ms"]["dspark_ctx"] - 4.8) < 1e-6 and abs(a0["by_class_ms"]["dspark_query"] - 5.7) < 1e-6 \
        and abs(a0["stager_wait_ms"] - 0.1) < 1e-9 and a0["steps"] == 16 and abs(a0["sub_ms"]["engram:engram_kv_gemm"] - 0.7) < 1e-9 \
        and abs(a0["sub_ms"]["indexer:index_topk"] - 0.5) < 1e-9 and abs(a0["sub_ms"]["mhc:mhc_post_collapse_rmsnorm"] - 1.3) < 1e-9 \
        and abs(nd["by_class_ms"]["experts"] - a0["by_class_ms"]["experts"]) < 1e-6 and o["windows"]["nc_iv"]["intervals_one_newcomer"] == 1 \
        and abs(o["windows"]["nc_iv"]["ranks"]["0"]["device_ms"] - a0["device_ms"]) < 1e-6 and a0["wall_ms"] == 600 \
        and o["windows"]["alone"]["mix"] == {"prefill": 16} and "host_remainder" in a0["ranked"]
    print("SELFTEST " + ("OK: time windows, step counts from the step log, dspark_ctx / dspark_query split, stager waits, breakdowns, W2 - W1 difference, intervals"
                         if ok else "FAILED: rc %s %s" % (rc, buf.getvalue()[:1500])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
