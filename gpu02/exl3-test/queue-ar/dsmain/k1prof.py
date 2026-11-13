#!/usr/bin/env python3
"""k1prof.py (10.11.2026, queue-ar dsmain, ds41-k1 agent; item 57-p4zyzzzzz-ds-ha-k1prof's analyser) -- where lever K1's
bench saving went. Two readings, each A (the K1 switches 0: today's ops) against B (1: the folds):

  k1prof.py stamps STAMPS.csv --win A:T0:T1 --win B:T0:T1 [--json OUT] [--label L]
      RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; card 0, host
      CLOCK_MONOTONIC ns), the trunk passes (the longest kind by records) whose first dispatch starts in each window. A pass
      (one decode step's trunk at c1): its wall (first start .. last end), busy (the union of its dispatches), idle
      (wall - busy), dispatches and ordered ones (barrier bit), and the time between passes. Per kernel class of the folds
          kvnorm   norm_rope_rt_kernel<.., KV=0>   (the window key's / index key's norm + rope + round trip)
          kvfold   norm_rope_rt_kernel<.., KV=1>   (fold c)
          kvstore  kv_store_kernel
          attn     ds41_attn_dec_kernel<.., ORT=0> attnfold ds41_attn_dec_kernel<.., ORT=1> (fold d)
          ropert   rope_rt_kernel<..>              (rope_rows_roundtrip: o, latent, index query)
          fill1    fill_kernel<uint16_t>           (the MoE shared gate's ones, and other bf16 fills; fold f)
      per pass: count, device us (sum of durations), EXPOSED us (time it runs with no other dispatch of the pass running:
      what its own run can cost the pass), IDLE-BEFORE us (the card idle right before it starts: a dependent launch's gap
      that it pays), how many are ordered (barrier bit) and how many lie on the pass's CHAIN (walking back from the last
      end: each dispatch's predecessor is the one that ended last before it started; the chain is the serial path that
      sets the wall, its time = its dispatches' run + the gaps between them). The fold's chain saving estimate is
      A's (exposed + idle-before) of the launches the fold removes, against the wall's change.
  k1prof.py ops ENGINE_LOG --win A:L0:L1 --win B:L0:L1 [--json OUT] [--label L] [--rank 0]
      --profile-ops tables (ds_opprof.py's parser, beside this file): per window and rank, for the ops of the folds
      (rmsnorm_rope_roundtrip[_kv_store], kv_store, attn_sparse_sink[_rope_roundtrip], rope_rows_roundtrip, fill n 1 bf16)
      calls a step, us a call, lane-wait us a call; the window's device ms and lane waits a step. PROFILED: synchronised
      steps, the per-op cost in the engine, not a speed.
Prints "K1PROF {json}" (the numbers) and readable lines. Standard library only."""
import csv, json, os, re, statistics, sys
from collections import defaultdict

CLASSES = ["kvnorm", "kvfold", "kvstore", "attn", "attnfold", "ropert", "fill1"]
R_NORM = re.compile(r"norm_rope_rt_kernelILb[01]ELj\d+E(?:Lb([01])E)?E")
R_ATTN = re.compile(r"ds41_attn_dec_kernelILi\d+ELi\d+ELi\d+ELi\d+ELb[01]ELb[01]E(?:Lb([01])E)?E")


def kclass(k):
    if "norm_rope_rt_kernel" in k:
        m = R_NORM.search(k)
        return "kvfold" if m and m.group(1) == "1" else "kvnorm"
    if "kv_store_kernel" in k: return "kvstore"
    if "ds41_attn_dec_kernel" in k:
        m = R_ATTN.search(k)
        return "attnfold" if m and m.group(1) == "1" else "attn"
    if "rope_rt_kernel" in k: return "ropert"
    if re.search(r"fill_kernelIt", k): return "fill1"
    return None


def wins(a):
    out = []
    for i, x in enumerate(a):
        if x == "--win":
            n, lo, hi = a[i + 1].split(":")
            out.append((n, int(lo), int(hi)))
    return out


def pass_stats(rows):
    """rows: [start, end, ordered, kernel] of one pass. -> per-pass numbers and per-class sums."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    n = len(rows)
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    # union (busy) and the exposed time of each dispatch: a sweep over start / end events
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
    # idle right before each dispatch: nothing of the pass running at its start
    idle_before, run_end = [0] * n, rows[0][0]
    for i, r in enumerate(rows):
        if i and r[0] > run_end: idle_before[i] = r[0] - run_end
        run_end = max(run_end, r[1])
    # the chain: back from the last end, each step to the dispatch that ended last at or before this one's start
    by_end = sorted(range(n), key=lambda i: rows[i][1])
    ends = [rows[i][1] for i in by_end]
    import bisect
    cur = max(range(n), key=lambda i: rows[i][1])
    chain, gaps = set([cur]), 0
    while True:
        j = bisect.bisect_right(ends, rows[cur][0]) - 1
        if j < 0: break
        p = by_end[j]
        if p in chain: break
        gaps += max(0, rows[cur][0] - rows[p][1])
        chain.add(p)
        cur = p
    per = {c: {"n": 0, "dev": 0.0, "exposed": 0.0, "idle_before": 0.0, "ordered": 0, "on_chain": 0} for c in CLASSES}
    for i, r in enumerate(rows):
        c = kclass(r[3])
        if not c: continue
        p = per[c]
        p["n"] += 1; p["dev"] += (r[1] - r[0]) / 1e3; p["exposed"] += exposed[i] / 1e3
        p["idle_before"] += idle_before[i] / 1e3; p["ordered"] += int(r[2]); p["on_chain"] += int(i in chain)
    chain_run = sum(rows[i][1] - rows[i][0] for i in chain)
    return {"wall": (t1 - t0) / 1e3, "busy": busy / 1e3, "idle": (t1 - t0 - busy) / 1e3, "disp": n,
            "ordered": sum(int(r[2]) for r in rows), "chain_n": len(chain), "chain_run": chain_run / 1e3,
            "chain_gaps": gaps / 1e3, "t0": t0, "t1": t1, "per": per}


def stamps(a):
    path = a[1]
    W = wins(a)
    passes = defaultdict(list)   # (window, pass) -> rows
    kinds = defaultdict(lambda: defaultdict(set))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r["start_ns"] or not r["end_ns"]: continue
            s, e = int(r["start_ns"]), int(r["end_ns"])
            for wn, lo, hi in W:
                if lo <= s <= hi:
                    key = (wn, int(r["pass"]))
                    passes[key].append((s, e, r["ordered"] == "1", r["kernel"]))
                    kinds[wn][int(r["records"])].add(int(r["pass"]))
    out = {}
    for wn, lo, hi in W:
        if not kinds[wn]:
            out[wn] = {"error": "no stamped pass in the window"}; continue
        trunk = max(kinds[wn])   # the longest kind: the trunk pass
        ps = sorted(kinds[wn][trunk])
        st = [pass_stats(passes[(wn, p)]) for p in ps if passes[(wn, p)]]
        # complete passes only: the window's first and last may be cut
        st = st[1:-1] if len(st) > 4 else st
        between = [b["t0"] - a_["t1"] for a_, b in zip(st, st[1:])]
        med = lambda k: statistics.median(x[k] for x in st) if st else None
        per = {}
        for c in CLASSES:
            per[c] = {k: statistics.mean(x["per"][c][k] for x in st) for k in ("n", "dev", "exposed", "idle_before",
                                                                             "ordered", "on_chain")} if st else {}
        out[wn] = {"records": trunk, "passes": len(st), "wall_us": med("wall"), "busy_us": med("busy"),
                   "idle_us": med("idle"), "dispatches": med("disp"), "ordered": med("ordered"),
                   "chain_n": med("chain_n"), "chain_run_us": med("chain_run"), "chain_gaps_us": med("chain_gaps"),
                   "between_us": statistics.median(between) / 1e3 if between else None,
                   "step_us": (med("wall") or 0) + (statistics.median(between) / 1e3 if between else 0), "per": per}
    return out


def ops(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ds_opprof import tables
    path, W = a[1], wins(a)
    rank = int(a[a.index("--rank") + 1]) if "--rank" in a else None
    T = tables(path)
    want = ("rmsnorm_rope_roundtrip", "kv_store", "attn_sparse_sink", "rope_rows_roundtrip", "fill")
    out = {}
    for wn, lo, hi in W:
        res = {}
        for rk in sorted({r for _, r, _ in T}):
            if rank is not None and rk != rank: continue
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
            agg = defaultdict(lambda: [0, 0.0, 0.0])
            for c, ms, op, geom, w, wt in d.values():
                if not op.startswith(want): continue
                if op.startswith("fill") and not ("n=1 " in geom + " " and "bf16" in geom): continue
                k = op if not op.startswith("fill") else "fill(n=1,bf16)"
                agg[k][0] += c; agg[k][1] += ms; agg[k][2] += wt
            res[rk] = {"steps": steps, "device_ms_step": sum(x[1] for x in d.values()) / max(1, steps),
                       "wait_ms_step": sum(x[5] for x in d.values()) / max(1, steps),
                       "ops": {k: {"calls_step": v[0] / max(1, steps), "us_call": 1000 * v[1] / max(1, v[0]),
                                   "wait_us_call": 1000 * v[2] / max(1, v[0]),
                                   "us_step": 1000 * v[1] / max(1, steps)} for k, v in agg.items()}}
        out[wn] = res or {"error": "no complete profile table in the window"}
    return out


def fmt_stamps(o):
    lines = []
    for wn, x in o.items():
        if "error" in x: lines.append(f"{wn}: {x['error']}"); continue
        lines.append(f"{wn}: {x['passes']} trunk passes ({x['records']} records): wall {x['wall_us']:.0f} us, busy "
                     f"{x['busy_us']:.0f}, idle {x['idle_us']:.0f}, {x['dispatches']:.0f} dispatches ({x['ordered']:.0f} "
                     f"ordered), chain {x['chain_n']:.0f} dispatches = run {x['chain_run_us']:.0f} + gaps "
                     f"{x['chain_gaps_us']:.0f} us, between passes {x['between_us'] or 0:.0f} us -> step {x['step_us']:.0f} us")
        for c in CLASSES:
            p = x["per"].get(c) or {}
            if not p or p["n"] < 0.5: continue
            lines.append(f"   {c:9s} {p['n']:.1f} a pass: device {p['dev']:.1f} us, exposed {p['exposed']:.1f}, idle "
                         f"before {p['idle_before']:.1f}, ordered {p['ordered']:.1f}, on the chain {p['on_chain']:.1f}")
    if "A" in o and "B" in o and "error" not in o["A"] and "error" not in o["B"]:
        A, B = o["A"], o["B"]
        rem = lambda x, cs: sum((x["per"].get(c) or {}).get(k, 0) for c in cs for k in ("exposed", "idle_before"))
        dev = lambda x, cs: sum((x["per"].get(c) or {}).get("dev", 0) for c in cs)
        a_sep, b_fold = ["kvnorm", "kvstore", "attn", "ropert", "fill1"], ["kvnorm", "kvfold", "attnfold", "ropert", "fill1"]
        lines.append(f"A -> B: wall {A['wall_us']:.0f} -> {B['wall_us']:.0f} us ({B['wall_us'] - A['wall_us']:+.0f}), step "
                     f"{A['step_us']:.0f} -> {B['step_us']:.0f} ({B['step_us'] - A['step_us']:+.0f}), dispatches "
                     f"{A['dispatches']:.0f} -> {B['dispatches']:.0f}, chain gaps {A['chain_gaps_us']:.0f} -> "
                     f"{B['chain_gaps_us']:.0f}; the fold classes' device {dev(A, a_sep):.0f} -> {dev(B, b_fold):.0f} us a pass, "
                     f"exposed + idle-before {rem(A, a_sep):.0f} -> {rem(B, b_fold):.0f} us a pass")
    return lines


def main():
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("stamps", "ops"):
        print(__doc__); return 2
    o = stamps(a) if a[0] == "stamps" else ops(a)
    label = a[a.index("--label") + 1] if "--label" in a else ""
    if "--json" in a:
        with open(a[a.index("--json") + 1], "w") as f: json.dump(o, f)
    if a[0] == "stamps":
        for l in fmt_stamps(o): print(f"K1PROF {label} stamps {l}")
    else:
        for wn, res in o.items():
            if "error" in res: print(f"K1PROF {label} ops {wn}: {res['error']}"); continue
            for rk, x in sorted(res.items()):
                print(f"K1PROF {label} ops {wn} rank {rk}: {x['steps']} steps, device {x['device_ms_step']:.2f} ms + lane waits "
                      f"{x['wait_ms_step']:.2f} ms a step (PROFILED); " + "; ".join(
                          f"{k} {v['calls_step']:.1f}/step {v['us_call']:.1f} us (wait {v['wait_us_call']:.1f}) = {v['us_step']:.0f} us/step"
                          for k, v in sorted(x["ops"].items())))
    print("K1PROF " + json.dumps({"label": label, "mode": a[0], "res": o})[:200000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
