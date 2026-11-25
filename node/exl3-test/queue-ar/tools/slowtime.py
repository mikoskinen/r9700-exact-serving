#!/usr/bin/env python3
"""slowtime.py STAMPS.csv[.gz] GM.txt --from NS --to NS [--kind RECORDS] -- the expert slow state against the card's own
gpu_metrics in time (8.11.2026, queue-ar; post-hoc for item 91). Card k = STAMPS.csv(.gz).c<k> (card 0 = STAMPS.csv), rank
k in gmsample.py's file (both in the engine's rank order). For each card and each complete verify pass in the window: the
number of slow layers (moe_down_dec > 1.6 x its layer's p10), the wall, and the gpu_metrics samples inside the pass
(power, sclk, the PPT0 / TEMP_HOTSPOT bits). Prints, per card, slow vs fast passes (<= 2 vs >= 10 slow layers): mean power,
sclk p50, PPT0 share, junction, and the correlation of the slow-layer count with power / sclk over the passes; then the
same pass on the other cards (is pass p slow everywhere?). Standard library only."""
import csv, glob, gzip, io, os, statistics, sys
from collections import defaultdict


def opn(p):
    return io.TextIOWrapper(gzip.open(p), newline="") if p.endswith(".gz") else open(p, newline="")


def files(base):
    b = base[:-3] if base.endswith(".gz") else base
    out = []
    for k in range(16):
        cand = [b + (f".c{k}" if k else ""), b + (f".c{k}" if k else "") + ".gz"]
        if k:
            cand = [b + f".c{k}", b + f".c{k}.gz", b + ".gz" + f".c{k}"]
        f = next((c for c in cand if os.path.exists(c)), None)
        if not f:
            if k == 0:
                continue
            break
        out.append((k, f))
    return out


def passes(path, kind, tf, tt):
    P = defaultdict(list)
    for r in csv.DictReader(opn(path)):
        if kind is not None and r["records"] != kind:
            continue
        try:
            s, e = int(r["start_ns"]), int(r["end_ns"])
        except (TypeError, ValueError):
            continue
        if s > 0 and e >= s:
            P[(int(r["records"]), int(r["pass"]))].append((int(r["seq"]), s, e, r["kernel"]))
    if kind is None:
        kmax = max(k for k, _ in P) if P else None
        P = {k: v for k, v in P.items() if k[0] == kmax}
    full = max((len(v) for v in P.values()), default=0)
    res = {}
    for (rec, p), v in P.items():
        if len(v) != full:
            continue
        v.sort()
        if tf <= v[0][1] <= tt:
            res[p] = v
    return res


def corr(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--kind"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    base, gm = a[0], a[1]
    tf, tt = int(opt.get("--from", 0)), int(opt.get("--to", 1 << 62))
    kind = opt.get("--kind")
    G = defaultdict(list)
    for line in open(gm):
        p = line.split()
        if len(p) >= 13 and p[0] == "M":
            G[int(p[1])].append((int(p[2]), int(p[5]), int(p[9]), int(p[7]), int(p[12], 16)))
    for r in G:
        G[r].sort()
    import bisect
    per = {}
    for k, f in files(base):
        ps = passes(f, kind, tf, tt)
        if not ps:
            print(f"card {k}: no passes"); continue
        dn = {p: [(x[2] - x[1]) for x in v if "moe_down_dec" in x[3]] for p, v in ps.items()}
        L = len(next(iter(dn.values())))
        p10 = [sorted(d[l] for d in dn.values())[len(dn) // 10] for l in range(L)]
        g = G.get(k, []); gt = [x[0] for x in g]
        rows = []
        for p, v in ps.items():
            t0, t1 = v[0][1], max(x[2] for x in v)
            ns = sum(1 for l in range(L) if dn[p][l] > 1.6 * p10[l])
            i0, i1 = bisect.bisect_left(gt, t0), bisect.bisect_right(gt, t1)
            smp = g[i0:i1]
            if not smp and g:   # 5 ms samples, a pass is ~25-35 ms: the nearest one if it is within 10 ms
                j = min(max(0, i0), len(g) - 1)
                smp = [g[j]] if abs(g[j][0] - t0) <= 10_000_000 else []
            pw = statistics.mean(x[1] for x in smp) if smp else float("nan")
            sc = statistics.mean(x[2] for x in smp) if smp else float("nan")
            hot = statistics.mean(x[3] for x in smp) if smp else float("nan")
            ppt = statistics.mean(1.0 if x[4] & 1 else 0.0 for x in smp) if smp else float("nan")
            if smp:
                rows.append((p, ns, (t1 - t0) / 1e3, pw, sc, hot, ppt, len(smp)))
        if not rows:
            print(f"card {k}: {len(ps)} passes but no gpu_metrics samples within them"); continue
        per[k] = {r[0]: r for r in rows}
        fast = [r for r in rows if r[1] <= 2]; slow = [r for r in rows if r[1] >= 10]
        def m(rs, i):
            return statistics.mean(r[i] for r in rs) if rs else float("nan")
        print(f"card {k}: {len(rows)} passes, slow layers mean {statistics.mean(r[1] for r in rows):.1f} of {L}; "
              f"fast (<=2) n={len(fast)} wall {m(fast, 2):.0f} us W {m(fast, 3):.0f} sclk {m(fast, 4):.0f} hot {m(fast, 5):.1f} PPT0 {100 * m(fast, 6):.0f}% | "
              f"slow (>=10) n={len(slow)} wall {m(slow, 2):.0f} us W {m(slow, 3):.0f} sclk {m(slow, 4):.0f} hot {m(slow, 5):.1f} PPT0 {100 * m(slow, 6):.0f}% | "
              f"corr(slow layers, W) {corr([r[1] for r in rows], [r[3] for r in rows]):+.2f}, (slow, sclk) {corr([r[1] for r in rows], [r[4] for r in rows]):+.2f}, "
              f"samples a pass {statistics.mean(r[7] for r in rows):.1f}")
    ks = sorted(per)
    if len(ks) > 1:
        common = set.intersection(*(set(per[k]) for k in ks))
        both = [[per[k][p][1] for k in ks] for p in sorted(common)]
        if both:
            print("same pass across cards: corr of slow-layer counts " +
                  " ".join(f"c0-c{k}: {corr([b[0] for b in both], [b[i] for b in both]):+.2f}" for i, k in enumerate(ks) if i))


if __name__ == "__main__":
    main()
