#!/usr/bin/env python3
"""profmoe.py RUNDIR EVERY [PHASES] (6.11.2026) -- post-analysis of a --profile-ops run (after-sweep.sh prof0/prcfg dirs, 3.11.
runs/prof0): per marks.txt window, ALL ranks: routed-MoE op time per layer and rank, all-reduce time per rank, and the
expert-parallel imbalance (a layer waits for its slowest rank: the all-reduce after the MoE absorbs the wait). Groups the MoE
ops by their geometry string (op + encoding: iq1_s / exl3 2-bit / 2.5-bit ...) so the 2.5-bit and 2-bit EXL3 layers stand apart.
Not run by after-sweep.sh; read only. Steps per window = EVERY x complete dumps of the rank between the marks."""
import os, re, sys
from collections import defaultdict
ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")


def dumps(path):
    """-> {rank: [(line_end, {(op, idx, wname): (ms, calls, geom, site)})]}"""
    out, cur, rk = defaultdict(list), None, None
    for i, l in enumerate(open(path, errors="replace"), 1):
        if l.startswith("per-op timing, rank "):
            rk = int(re.match(r"per-op timing, rank (\d+)", l).group(1)); cur = {}; continue
        if cur is None:
            continue
        m = ROW.match(l)
        if m:
            cur[(m.group(7), int(m.group(9)), m.group(10))] = (float(m.group(2)), int(m.group(1)), m.group(8), m.group(4))
        elif l.strip().startswith("total"):
            out[rk].append((i, cur)); cur = None
    return out


def win(ds, l0, l1):
    n = lambda line: sum(1 for ln, _ in ds if ln <= line)
    i0, i1 = n(l0), n(l1)
    a = ds[i0 - 1][1] if i0 else {}
    b = ds[i1 - 1][1] if i1 else {}
    return {k: (v[0] - a.get(k, (0, 0))[0], v[1] - a.get(k, (0, 0))[1], v[2], v[3]) for k, v in b.items()
            if v[1] - a.get(k, (0, 0))[1] > 0}, i1 - i0


rundir, every = sys.argv[1], int(sys.argv[2])
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
D = dumps(os.path.join(rundir, "engine.log"))
marks = [l.split() for l in open(os.path.join(rundir, "marks.txt")) if len(l.split()) >= 3]
lay = lambda w: int(re.search(r"blk\.(\d+)\.", w).group(1)) if re.search(r"blk\.(\d+)\.", w) else -1
for (p0, l0, _), (p1, l1, _) in zip(marks, marks[1:]):
    if want and p1 not in want:
        continue
    per = {}  # rank -> (delta, steps)
    for r, ds in sorted(D.items()):
        d, nd = win(ds, int(l0), int(l1))
        if nd:
            per[r] = (d, nd * every)
    if not per:
        print(f"== {p1}: no dumps"); continue
    print(f"\n== {p1}: ranks {sorted(per)}, steps {[per[r][1] for r in sorted(per)]}")
    moe = defaultdict(lambda: defaultdict(float))   # (layer, class) -> rank -> ms/step
    ar, tot, host = defaultdict(float), defaultdict(float), defaultdict(float)
    for r, (d, st) in per.items():
        for (op, idx, w), (ms, calls, geom, site) in d.items():
            tot[r] += ms / st
            if site == "host":
                host[r] += ms / st
            if op.startswith("all_reduce"):
                ar[r] += ms / st
            if op.startswith("moe_") and ("gemm" in op or "exl3" in op):
                cls = op + " " + " ".join(t for t in geom.split() if t.split("=")[0] in ("type", "dtype", "codebook", "N", "K"))
                moe[(lay(w), cls)][r] += ms / st
    print("   rank: op total / all-reduce / host-site ms per step: " + "; ".join(f"r{r} {tot[r]:.2f} / {ar[r]:.2f} / {host[r]:.2f}" for r in sorted(per)))
    bycls = defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # cls -> n layers, sum mean, sum max, sum min (ms/step)
    for (l, cls), rr in moe.items():
        v = [rr.get(r, 0.0) for r in sorted(per)]
        g = bycls[cls]; g[0] += 1; g[1] += sum(v) / len(v); g[2] += max(v); g[3] += min(v)
    imb = 0.0
    for cls, (n, sm, mx, mn) in sorted(bycls.items(), key=lambda x: -x[1][2]):
        imb += mx - sm
        print(f"   {cls[:90]:90s} layers {n:3d}: per layer mean-rank {1000 * sm / n:7.1f} us, slowest-rank {1000 * mx / n:7.1f} us, fastest {1000 * mn / n:7.1f} us; "
              f"sum over layers mean {sm:.2f} / max {mx:.2f} ms/step")
    print(f"   expert-parallel imbalance (sum over layers and projections of slowest-rank minus mean-rank MoE time): {imb:.2f} ms/step"
          f" -- an upper bound for what balanced expert placement could save, before the all-reduce absorbs it")
