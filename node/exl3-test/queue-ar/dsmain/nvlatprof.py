#!/usr/bin/env python3
"""nvlatprof.py (10.11.2026, queue-ar dsmain, ds-main-port nvlat agent; item 57-p4zyzzzzz-ds-hszzz-nvlat's analyser, body
dsm-nvlat.sh) -- where the NVFP4 latent's decode cost goes: arm A (RADIANCE_DS41_LATENT unset, bf16 latent) against arm B
(nvfp4), each from its own start. Three modes:

  nvlatprof.py stamps --a CSV[.gz]:T0:T1 --b CSV[.gz]:T0:T1 [--label L] [--json OUT]
      RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; card 0, host
      CLOCK_MONOTONIC ns), the passes whose dispatches start in each side's window. The TRUNK kind (records) is
      d0attrprof.py's rule (the most records among the kinds with >= 25 % of the most frequent kind's passes). Per trunk
      pass: wall, busy, dispatches, the CHAIN (k1prof.py's walk: back from the last end, each dispatch's predecessor the
      one that ended last at or before its start; the chain's run + its gaps = the wall), and per kernel (family + its
      template arguments) count, device us, EXPOSED us (running with nothing else of the pass), idle-before us and its
      run ON THE CHAIN. Steps: trunk start to the next trunk start (over 3x the median left out). Then B against A: wall,
      chain run / gaps, step; per kernel family and per kernel the device and on-chain us a pass, largest differences first,
      and the kernels only one side has.
  nvlatprof.py ops --a LOG:L0:L1 --b LOG:L0:L1 [--label L] [--json OUT]
      --profile-ops tables (ds_opprof.py's parser and classes, imported read-only from beside this file), each side's
      window as engine-log line numbers: per rank the device ms and lane waits a step, by class, and per op (name +
      latent / format parameters of its geometry) calls a step and us a call; then B against A (rank 0 and the mean of the
      ranks), largest differences first, and the ops only one side has. PROFILED: synchronised steps, not a speed.
  nvlatprof.py summary DIR
      The item's summary from DIR's files (clean readings, stamps / ops JSON): per reading ms a step A / B and %, the
      stamped critical-path difference against the profiled device-sum difference, the kernels only in B.
Prints "NVLP {json}" (the numbers) and readable "NVLP <label> ..." lines. Standard library only."""
import bisect, csv, glob, gzip, json, os, re, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def opn(p):
    return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")


_kc = {}


def kname(k):
    """A mangled kernel name -> 'family<args>' (the last name component, its integer / bool template arguments)."""
    v = _kc.get(k)
    if v is not None: return v
    s, i, comps, targs = k, 0, [], ""
    i = 2 if s.startswith("_Z") else 0
    if s[i:i + 1] == "N": i += 1
    while i < len(s) and s[i].isdigit():
        j = i
        while j < len(s) and s[j].isdigit(): j += 1
        n = int(s[i:j]); comps.append(s[j:j + n]); i = j + n
        if s[i:i + 1] == "I":
            mm = re.match(r"I((?:L[a-z]n?\d+E)+)", s[i:])
            a = re.findall(r"L([a-z])(n?\d+)E", mm.group(1)) if mm else []
            targs = "<" + ",".join(("b" + x if t == "b" else x.replace("n", "-")) for t, x in a) + ">" if a else "<..>"
            break
    comps = [c for c in comps if c and not c.startswith("_GLOBAL__N")]
    v = (comps[-1] if comps else k[:60]) + targs
    _kc[k] = v
    return v


def fam(kn):
    return kn.split("<", 1)[0]


def side(a, flag):
    x = a[a.index(flag) + 1]
    p, lo, hi = x.rsplit(":", 2)
    return p, int(lo), int(hi)


# ------------------------------------------------------------------------------------------------------------ stamps
def read_passes(path, lo, hi):
    P = defaultdict(lambda: {"rec": 0, "rows": []})
    with opn(path) as f:
        for r in csv.DictReader(f):
            s, e = r.get("start_ns"), r.get("end_ns")
            if not s or not e: continue
            s, e = int(s), int(e)
            if lo <= s <= hi:
                p = P[int(r["pass"])]
                p["rec"] = int(r["records"])
                p["rows"].append((s, e, r.get("ordered") == "1", kname(r["kernel"])))
    return P


def pstats(rows):
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    n = len(rows)
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    ev = []
    for i, r in enumerate(rows):
        ev.append((r[0], 1, i)); ev.append((r[1], -1, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    active, busy, exposed, last = set(), 0, [0] * n, None
    for t, kind, i in ev:
        if last is not None and active:
            busy += t - last
            if len(active) == 1: exposed[next(iter(active))] += t - last
        if kind == 1: active.add(i)
        else: active.discard(i)
        last = t
    idle_before, run_end = [0] * n, rows[0][0]
    for i, r in enumerate(rows):
        if i and r[0] > run_end: idle_before[i] = r[0] - run_end
        run_end = max(run_end, r[1])
    by_end = sorted(range(n), key=lambda i: rows[i][1])
    ends = [rows[i][1] for i in by_end]
    cur = max(range(n), key=lambda i: rows[i][1])
    chain, gaps = {cur}, 0
    while True:
        j = bisect.bisect_right(ends, rows[cur][0]) - 1
        if j < 0: break
        p = by_end[j]
        if p in chain: break
        gaps += max(0, rows[cur][0] - rows[p][1])
        chain.add(p); cur = p
    per = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0, 0.0])   # n, dev, exposed, idle_before, chain_n, chain_run (us)
    for i, r in enumerate(rows):
        q = per[r[3]]
        q[0] += 1; q[1] += (r[1] - r[0]) / 1e3; q[2] += exposed[i] / 1e3; q[3] += idle_before[i] / 1e3
        if i in chain: q[4] += 1; q[5] += (r[1] - r[0]) / 1e3
    return {"wall": (t1 - t0) / 1e3, "busy": busy / 1e3, "disp": n, "ordered": sum(int(r[2]) for r in rows),
            "chain_n": len(chain), "chain_run": sum(rows[i][1] - rows[i][0] for i in chain) / 1e3, "chain_gaps": gaps / 1e3,
            "t0": t0, "t1": t1, "per": per}


def med(v):
    return statistics.median(v) if v else None


def analyse(P):
    ps = [dict(pstats(p["rows"]), rec=p["rec"]) for p in P.values() if p["rows"]]
    if not ps: return {"error": "no stamped pass in the window"}
    ps.sort(key=lambda x: x["t0"])
    ps = ps[1:-1] if len(ps) > 6 else ps
    kinds = defaultdict(int)
    for p in ps: kinds[p["rec"]] += 1
    top = max(kinds.values())
    trunk = max(k for k, v in kinds.items() if v >= 0.25 * top)
    tr = [p for p in ps if p["rec"] == trunk]
    per_ = [b["t0"] - a["t0"] for a, b in zip(tr, tr[1:])]
    pm = med(per_)
    keep = [i for i, x in enumerate(per_) if pm is None or x <= 3 * pm]
    trk = [tr[i] for i in keep] or tr
    m = lambda k: med([p[k] for p in trk])
    agg = defaultdict(lambda: [0.0] * 6)
    for p in trk:
        for kn, q in p["per"].items():
            for j in range(6): agg[kn][j] += q[j]
    nt = max(1, len(trk))
    kern = {kn: {"n": v[0] / nt, "dev": v[1] / nt, "exposed": v[2] / nt, "idle_before": v[3] / nt, "chain_n": v[4] / nt,
                 "chain_run": v[5] / nt} for kn, v in agg.items()}
    fams = defaultdict(lambda: defaultdict(float))
    for kn, v in kern.items():
        for k2, x in v.items(): fams[fam(kn)][k2] += x
    return {"trunk_records": trunk, "kinds": {str(k): v for k, v in kinds.items()}, "trunk_passes": len(trk),
            "step_us_med": (pm / 1e3) if pm else None,
            "step_us_mean": statistics.mean(per_[i] for i in keep) / 1e3 if keep else None,
            "wall_us": m("wall"), "busy_us": m("busy"), "disp": m("disp"), "ordered": m("ordered"),
            "chain_n": m("chain_n"), "chain_run_us": m("chain_run"), "chain_gaps_us": m("chain_gaps"),
            "kern": kern, "fam": {k: dict(v) for k, v in fams.items()}}


def cmp_keys(A, B, key, n=14, thr=1.0):
    ka, kb = A[key], B[key]
    ks = set(ka) | set(kb)
    z = {"n": 0, "dev": 0.0, "exposed": 0.0, "idle_before": 0.0, "chain_n": 0, "chain_run": 0.0}
    rows = []
    for k in ks:
        a, b = ka.get(k, z), kb.get(k, z)
        rows.append({"k": k, "dev_a": a["dev"], "dev_b": b["dev"], "chain_a": a["chain_run"], "chain_b": b["chain_run"],
                     "exp_a": a["exposed"], "exp_b": b["exposed"], "idle_a": a["idle_before"], "idle_b": b["idle_before"],
                     "n_a": a["n"], "n_b": b["n"]})
    rows.sort(key=lambda r: -abs(r["dev_b"] - r["dev_a"]))
    return {"top_dev": rows[:n], "top_chain": sorted(rows, key=lambda r: -abs(r["chain_b"] - r["chain_a"]))[:n],
            "only_b": sorted(r["k"] for r in rows if r["n_a"] == 0 and r["dev_b"] >= thr),
            "only_a": sorted(r["k"] for r in rows if r["n_b"] == 0 and r["dev_a"] >= thr),
            "dev_sum": [sum(v["dev"] for v in ka.values()), sum(v["dev"] for v in kb.values())]}


def stamps(a):
    pa, la, ha = side(a, "--a"); pb, lb, hb = side(a, "--b")
    A = analyse(read_passes(pa, la, ha)); B = analyse(read_passes(pb, lb, hb))
    out = {"A": A, "B": B}
    if "error" not in A and "error" not in B:
        out["fam"] = cmp_keys(A, B, "fam"); out["kern"] = cmp_keys(A, B, "kern")
    return out


def fmt_stamps(o):
    L = []
    for s in ("A", "B"):
        x = o[s]
        if "error" in x: L.append(f"{s}: {x['error']}"); continue
        L.append(f"{s}: {x['trunk_passes']} trunk passes ({x['trunk_records']} records; kinds {x['kinds']}), step {x['step_us_med'] or 0:.0f} us "
                 f"median, wall {x['wall_us']:.0f} busy {x['busy_us']:.0f} us, {x['disp']:.0f} dispatches ({x['ordered']:.0f} ordered), chain "
                 f"{x['chain_n']:.0f} dispatches = run {x['chain_run_us']:.0f} + gaps {x['chain_gaps_us']:.0f} us")
    if "fam" in o:
        A, B = o["A"], o["B"]
        f = o["fam"]
        L.append(f"B - A: step {(B['step_us_med'] or 0) - (A['step_us_med'] or 0):+.0f} us, wall {B['wall_us'] - A['wall_us']:+.0f}, chain run "
                 f"{B['chain_run_us'] - A['chain_run_us']:+.0f}, chain gaps {B['chain_gaps_us'] - A['chain_gaps_us']:+.0f}, dispatches "
                 f"{B['disp'] - A['disp']:+.0f}; device sum {f['dev_sum'][0]:.0f} -> {f['dev_sum'][1]:.0f} us a pass ({f['dev_sum'][1] - f['dev_sum'][0]:+.0f})")
        L.append("families by device us a pass A/B (n a pass), largest differences: " + "; ".join(
            f"{r['k']} {r['dev_a']:.1f}/{r['dev_b']:.1f} ({r['dev_b'] - r['dev_a']:+.1f}; on chain {r['chain_a']:.1f}/{r['chain_b']:.1f}; "
            f"exposed {r['exp_a']:.1f}/{r['exp_b']:.1f}; {r['n_a']:.0f}/{r['n_b']:.0f})" for r in f["top_dev"][:10]))
        L.append("families by ON-CHAIN us a pass A/B, largest differences: " + "; ".join(
            f"{r['k']} {r['chain_a']:.1f}/{r['chain_b']:.1f} ({r['chain_b'] - r['chain_a']:+.1f})" for r in f["top_chain"][:10]))
        k = o["kern"]
        L.append("kernels by device us a pass A/B, largest differences: " + "; ".join(
            f"{r['k']} {r['dev_a']:.1f}/{r['dev_b']:.1f} ({r['dev_b'] - r['dev_a']:+.1f}; {r['n_a']:.0f}/{r['n_b']:.0f})" for r in k["top_dev"][:10])
            + f" | kernels only in B: {', '.join(k['only_b']) or '-'} | only in A: {', '.join(k['only_a']) or '-'}")
    return L


# ------------------------------------------------------------------------------------------------------------ ops
def opkey(op, geom):
    g = " ".join(x for x in re.findall(r"(?:latent_format|format|key_format)=\S+", geom))
    return op + (" [" + g + "]" if g else "")


def ops_side(T, cls, lo, hi):
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
        steps = d[min(d)][0]
        n = max(1, steps)
        byc = defaultdict(float); byo = defaultdict(lambda: [0, 0.0, 0.0])
        for c, ms, op, g, w, wt in d.values():
            byc[cls(op)] += ms
            k = opkey(op, g)
            byo[k][0] += c; byo[k][1] += ms; byo[k][2] += wt
        res[str(rk)] = {"steps": steps, "device_ms_step": sum(x[1] for x in d.values()) / n,
                        "wait_ms_step": sum(x[5] for x in d.values()) / n,
                        "by_class_ms_step": {k: v / n for k, v in byc.items()},
                        "ops": {k: {"calls_step": v[0] / n, "ms_step": v[1] / n, "us_call": 1000 * v[1] / max(1, v[0]),
                                    "wait_ms_step": v[2] / n} for k, v in byo.items()}}
    return res


def ops(a):
    sys.path.insert(0, HERE)
    from ds_opprof import tables, cls
    out = {}
    for s, flag in (("A", "--a"), ("B", "--b")):
        p, lo, hi = side(a, flag)
        out[s] = ops_side(tables(p), cls, lo, hi) or {"error": "no complete profile table in the window"}
    A, B = out["A"], out["B"]
    if "error" not in A and "error" not in B:
        rks = sorted(set(A) & set(B))
        mean = lambda S, f: statistics.mean(f(S[r]) for r in rks) if rks else 0.0
        keys = set()
        for r in rks: keys |= set(A[r]["ops"]) | set(B[r]["ops"])
        rows = []
        for k in keys:
            g = lambda S, r, f: (S[r]["ops"].get(k) or {}).get(f, 0.0)
            rows.append({"k": k, "ms_a": mean(A, lambda x: (x["ops"].get(k) or {}).get("ms_step", 0.0)),
                         "ms_b": mean(B, lambda x: (x["ops"].get(k) or {}).get("ms_step", 0.0)),
                         "r0_a": g(A, rks[0], "ms_step"), "r0_b": g(B, rks[0], "ms_step"),
                         "us_a": g(A, rks[0], "us_call"), "us_b": g(B, rks[0], "us_call"),
                         "calls_a": g(A, rks[0], "calls_step"), "calls_b": g(B, rks[0], "calls_step")})
        rows.sort(key=lambda r: -abs(r["ms_b"] - r["ms_a"]))
        cl = set()
        for r in rks: cl |= set(A[r]["by_class_ms_step"]) | set(B[r]["by_class_ms_step"])
        out["cmp"] = {"ranks": rks, "device": [mean(A, lambda x: x["device_ms_step"]), mean(B, lambda x: x["device_ms_step"])],
                      "wait": [mean(A, lambda x: x["wait_ms_step"]), mean(B, lambda x: x["wait_ms_step"])],
                      "classes": {c: [mean(A, lambda x: x["by_class_ms_step"].get(c, 0.0)), mean(B, lambda x: x["by_class_ms_step"].get(c, 0.0))]
                                  for c in cl},
                      "ops": rows[:20],
                      "only_b": sorted(r["k"] for r in rows if r["calls_a"] == 0 and r["calls_b"] > 0),
                      "only_a": sorted(r["k"] for r in rows if r["calls_b"] == 0 and r["calls_a"] > 0)}
    return out


def fmt_ops(o):
    L = []
    for s in ("A", "B"):
        x = o[s]
        if "error" in x: L.append(f"{s}: {x['error']}"); continue
        for rk, r in sorted(x.items()):
            L.append(f"{s} rank {rk}: {r['steps']} steps, device {r['device_ms_step']:.2f} + lane waits {r['wait_ms_step']:.2f} ms a step (PROFILED); "
                     + ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["by_class_ms_step"].items(), key=lambda kv: -kv[1]) if v >= 0.05))
    if "cmp" in o:
        c = o["cmp"]
        L.append(f"B - A (PROFILED ms a step, mean of ranks {c['ranks']}): device {c['device'][0]:.2f} -> {c['device'][1]:.2f} ({c['device'][1] - c['device'][0]:+.2f}), "
                 f"lane waits {c['wait'][0]:.2f} -> {c['wait'][1]:.2f} ({c['wait'][1] - c['wait'][0]:+.2f}); classes: " + ", ".join(
                     f"{k} {v[0]:.2f}/{v[1]:.2f} ({v[1] - v[0]:+.2f})" for k, v in sorted(c["classes"].items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])) if max(v) >= 0.05))
        L.append("ops by ms a step A/B (mean of ranks; rank 0 us a call A/B, calls a step A/B), largest differences: " + "; ".join(
            f"{r['k']} {r['ms_a']:.3f}/{r['ms_b']:.3f} ({r['ms_b'] - r['ms_a']:+.3f}; {r['us_a']:.1f}/{r['us_b']:.1f} us; {r['calls_a']:.1f}/{r['calls_b']:.1f})"
            for r in c["ops"][:12]) + f" | ops only in B: {', '.join(c['only_b']) or '-'} | only in A: {', '.join(c['only_a']) or '-'}")
    return L


# ------------------------------------------------------------------------------------------------------------ summary
def jl(p, pre):
    try: return [json.loads(l[len(pre):]) for l in open(p) if l.startswith(pre)]
    except OSError: return []


def pc(a, b):
    return f"{100.0 * (b / a - 1):+.1f} %" if a and b else "-"


def summary(a):
    O = a[1]
    out, num = [], {}
    def clean(arm, what):
        fs = sorted(glob.glob(os.path.join(O, f"p1{arm}-{what}*.txt")))
        if what == "c1s":
            v = [r["ms_per_decode_step"] for f in fs for r in jl(f, "J ") if r.get("ms_per_decode_step")]
            t = [r["decode_tps"] for f in fs for r in jl(f, "J ") if r.get("decode_tps")]
            return (statistics.mean(v) if v else None, statistics.mean(t) if t else None, len(v))
        if what == "c1L":
            v = [r["ms_per_decode_step"] for f in fs for r in jl(f, "NVM ") if r.get("ms_per_decode_step")]
            return (statistics.mean(v) if v else None, None, len(v))
        if what == "c8":
            r = [x for f in fs for x in jl(f, "MB ")]
            return (statistics.mean(x["ms_decode_step"] for x in r) if r else None, statistics.mean(x["agg_dec_tps"] for x in r) if r else None, len(r))
        if what == "copy":
            fs = sorted(glob.glob(os.path.join(O, f"p1{arm}-c1L-cold*.txt")))
            r = [x for f in fs for x in jl(f, "J ") if x.get("test") == "copy" and x.get("run") == 1]
            tt = [x["ttft"] for x in r if x.get("ttft") and (x.get("cached") in (0, None))]
            pt = [((x.get("usage") or {}).get("prompt_tokens") or 0) for x in r]
            return (statistics.mean(tt) if tt else None, (pt[0] / statistics.mean(tt)) if tt and pt and pt[0] else None, len(tt))
        if what == "pf16":
            r = [x for f in glob.glob(os.path.join(O, "guard", f"nv-{arm.lower()}plain-warm-pf.txt")) for x in jl(f, "J ")]
            tt = [x["ttft"] for x in r if x.get("ttft")]
            tp = [x["prefill_tps"] for x in r if x.get("prefill_tps")]
            return (statistics.mean(tt) if tt else None, statistics.mean(tp) if tp else None, len(tt))
        return (None, None, 0)
    names = [("c1s", "1 user short prompt (stepprobe essay,code 512), ms a step"), ("c1L", "1 user ~31k prompt (the copy's decode), ms a step"),
             ("c8", "mb c8 45 s, ms a decode step"), ("copy", "~31k cold prefill (the copy's TTFT), s"), ("pf16", "16k cold prefill (the warm-up's), s")]
    parts = []
    for w, desc in names:
        A, B = clean("A", w), clean("B", w)
        num[w] = {"A": A, "B": B}
        extra = ""
        if w in ("c1s", "c8") and A[1] and B[1]: extra = f"; tok/s {A[1]:.1f} -> {B[1]:.1f} ({pc(A[1], B[1])})"
        if w in ("copy", "pf16") and A[1] and B[1]: extra = f"; prefill tok/s {A[1]:.0f} -> {B[1]:.0f} ({pc(A[1], B[1])})"
        if A[0] and B[0]:
            parts.append(f"{desc}: A {A[0]:.3f} / B {B[0]:.3f} ({B[0] - A[0]:+.3f}, {pc(A[0], B[0])}; n {A[2]}/{B[2]}){extra}")
        else:
            parts.append(f"{desc}: A {A[0]} / B {B[0]} (missing)")
    out.append("CLEAN (unprofiled, unstamped; A = bf16 latent, B = nvfp4): " + " | ".join(parts))
    for w in ("short", "long"):
        s = None
        try: s = json.load(open(os.path.join(O, f"st-{w}.json")))
        except (OSError, ValueError): pass
        p = None
        try: p = json.load(open(os.path.join(O, f"op-{w}.json")))
        except (OSError, ValueError): pass
        line = f"{w.upper()} prompt: "
        if s and "fam" in s:
            A, B = s["A"], s["B"]; f = s["fam"]
            line += (f"STAMPED (card 0, ~8 % inflated) step {A['step_us_med']:.0f} -> {B['step_us_med']:.0f} us ({B['step_us_med'] - A['step_us_med']:+.0f}), "
                     f"critical path (chain run + gaps = the trunk wall) {A['wall_us']:.0f} -> {B['wall_us']:.0f} ({B['wall_us'] - A['wall_us']:+.0f}; run "
                     f"{B['chain_run_us'] - A['chain_run_us']:+.0f}, gaps {B['chain_gaps_us'] - A['chain_gaps_us']:+.0f}), device sum {f['dev_sum'][1] - f['dev_sum'][0]:+.0f} us; "
                     "largest families (device / on chain B - A us): " + ", ".join(
                         f"{r['k']} {r['dev_b'] - r['dev_a']:+.1f} / {r['chain_b'] - r['chain_a']:+.1f}" for r in f["top_dev"][:5])
                     + f"; kernels only in B: {', '.join(s['kern']['only_b']) or '-'}")
            num["st_" + w] = {"step": [A["step_us_med"], B["step_us_med"]], "wall": [A["wall_us"], B["wall_us"]], "dev": f["dev_sum"]}
        else:
            line += "STAMPED: none"
        if p and "cmp" in p:
            c = p["cmp"]
            line += (f" || PROFILED device {c['device'][0]:.2f} -> {c['device'][1]:.2f} ms a step ({c['device'][1] - c['device'][0]:+.2f}); largest ops: " + ", ".join(
                f"{r['k']} {r['ms_b'] - r['ms_a']:+.3f} ms ({r['us_a']:.1f} -> {r['us_b']:.1f} us a call)" for r in c["ops"][:4])
                + f"; ops only in B: {', '.join(c['only_b']) or '-'}")
            num["op_" + w] = {"device": c["device"]}
        else:
            line += " || PROFILED: none"
        if ("st_" + w) in num and ("op_" + w) in num and num.get("c1s" if w == "short" else "c1L", {}).get("A", (None,))[0]:
            cw = num["c1s" if w == "short" else "c1L"]
            d_clean = 1000 * ((cw["B"][0] or 0) - (cw["A"][0] or 0))
            d_wall = num["st_" + w]["wall"][1] - num["st_" + w]["wall"][0]
            d_dev = 1000 * (num["op_" + w]["device"][1] - num["op_" + w]["device"][0])
            line += f" || clean {d_clean:+.0f} us a step vs stamped critical path {d_wall:+.0f} us vs profiled device sum {d_dev:+.0f} us"
        out.append(line)
    try:
        idt = open(os.path.join(O, "identity.txt")).read().split("\n")
        out.append("IDENTITY: " + "; ".join(l for l in idt if l.strip()))
    except OSError:
        out.append("IDENTITY: none")
    return out, num


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("stamps", "ops", "summary"):
        print(__doc__); return 2
    label = a[a.index("--label") + 1] if "--label" in a else a[0]
    if a[0] == "summary":
        L, o = summary(a)
    else:
        o = stamps(a) if a[0] == "stamps" else ops(a)
        L = fmt_stamps(o) if a[0] == "stamps" else fmt_ops(o)
    if "--json" in a:
        with open(a[a.index("--json") + 1], "w") as fh: json.dump(o, fh)
    for l in L: print(f"NVLP {label} {l}")
    print("NVLP " + json.dumps({"label": label, "mode": a[0], "res": o}, default=str)[:200000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
