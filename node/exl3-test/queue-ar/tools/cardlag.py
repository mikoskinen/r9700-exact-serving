#!/usr/bin/env python3
"""cardlag.py STAMPS.csv [--card C] [--from NS --to NS] [--kind RECORDS] [--layers N] [--label L] [--json OUT]
(9.11.2026, queue-ar, item 111) -- how much ONE card delays a TP step, from RADIANCE_AQL_STAMP_CARDS=all stamps (card 0's
file is STAMPS.csv, card k's STAMPS.csv.c<k>; all in the host's CLOCK_MONOTONIC domain, the same pass numbers on every card).

For the longest pass kind in the window (the verify pass in decode; --kind picks another), over the passes complete on
every card, the k-th all-reduce of a pass is the same collective on every card (rankskew.py's matching). For card c and
collective k with arrival (start) times s_0..s_{W-1}:
  lag_c(k) = max(0, s_c - max_{d != c} s_d)   how much later c arrives than the latest of the others: the time every other
                                              rank waits for c alone, i.e. the step's critical path beyond what the
                                              next-latest card would have set (if c were never later than it)
Per card: how often it arrives last, the summed lag a pass (median and mean over passes, us) and its share of the pass's
wall (first start to last end, median over cards), and the rank-time the others spend waiting for it ((W-1) x lag).
For --card C (default 1: 83:00 in the c3 / 83 / 86 / 03 rank order): the lag by the work before the collective (moe / gdn /
attn / other), per layer (--layers N, default 48: collective k belongs to layer k * N // collectives-a-pass), and the
collectives with the largest median lag. Standard library only."""
import csv, glob, json, os, re, statistics, sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stampcat import classify  # noqa: E402


def load(path, tf, tt):
    P = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        try:
            s, e = int(r["start_ns"]), int(r["end_ns"])
        except (TypeError, ValueError, KeyError):
            continue
        if s <= 0 or e < s:
            continue
        P[(r["pass"], r["records"])].append((int(r["seq"]), s, e, r["kernel"]))
    out = defaultdict(dict)
    for (p, rec), rows in P.items():
        rows.sort()
        if tf <= rows[0][1] <= tt:
            out[int(rec)][int(p)] = rows
    return out


def ars(rows):
    """The pass's collectives in order: (start, end, the class of the work before it), as rankskew.py classifies them."""
    res = []
    for i, x in enumerate(rows):
        if classify(x[3]) != "allreduce":
            continue
        j = i - 1
        while j > 0 and classify(rows[j][3]) in ("other",) and ("aql_copy" in rows[j][3] or "gather" in rows[j][3]):
            j -= 1
        prev = rows[j][3] if j >= 0 else ""
        src = ("moe" if re.search(r"moe|gemm_fp8a8|gather", prev) and any("moe_down" in rows[q][3] for q in range(max(0, i - 6), i))
               else "gdn" if any("gdn" in rows[q][3] for q in range(max(0, i - 4), i))
               else "attn" if any(re.search(r"attn|qsa", rows[q][3]) for q in range(max(0, i - 6), i)) else "other")
        res.append((x[1], x[2], src))
    return res


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--kind", "--label", "--json", "--card", "--layers"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    base = a[0]
    tf = int(opt.get("--from", 0)); tt = int(opt.get("--to", 1 << 62))
    files = [base] + sorted(glob.glob(base + ".c*"), key=lambda f: int(f.rsplit(".c", 1)[1]))
    label = opt.get("--label", base)
    focus = int(opt.get("--card", 1)); nlay = int(opt.get("--layers", 48))
    data = [load(f, tf, tt) for f in files]
    W = len(files)
    kinds = set(data[0].keys())
    for d in data[1:]:
        kinds &= set(d.keys())
    if W < 2 or not kinds:
        print(f"{label}: no pass kind on every card ({W} file(s))"); return
    kind = int(opt["--kind"]) if "--kind" in opt else max(kinds)
    common = set(data[0][kind].keys())
    for d in data[1:]:
        common &= set(d[kind].keys())
    full = [max(len(v) for v in d[kind].values()) for d in data]
    passes = sorted(p for p in common if all(len(d[kind][p]) == full[c] for c, d in enumerate(data)))
    if not passes:
        print(f"{label}: kind records={kind}, no pass complete on every card"); return
    walls, lag, last = [], defaultdict(list), Counter()
    per_k = defaultdict(list); by_src = defaultdict(list); by_src_last = Counter(); by_src_n = Counter()
    nar = None
    for p in passes:
        rows = [d[kind][p] for d in data]
        walls.append(statistics.median(max(x[2] for x in r) - r[0][1] for r in rows))
        A = [ars(r) for r in rows]
        n = min(len(x) for x in A)
        nar = n if nar is None else min(nar, n)
        pl = [0.0] * W
        for k in range(n):
            s = [A[c][k][0] for c in range(W)]
            last[s.index(max(s))] += 1
            for c in range(W):
                l = max(0.0, s[c] - max(s[d] for d in range(W) if d != c))
                pl[c] += l
                if c == focus:
                    per_k[k].append(l)
                    src = A[c][k][2]
                    by_src[src].append(l); by_src_n[src] += 1
                    if l > 0: by_src_last[src] += 1
        for c in range(W):
            lag[c].append(pl[c])
    wall = statistics.median(walls)
    tot = sum(last.values())
    out = {"label": label, "cards": W, "kind": kind, "passes": len(passes), "collectives": nar, "wall_us": wall / 1e3,
           "card": {}, "focus": focus}
    print(f"{label}: {W} cards, kind records={kind}, {len(passes)} passes, {nar} collectives a pass, pass wall median "
          f"{wall / 1e3:.0f} us")
    for c in range(W):
        md, mn = statistics.median(lag[c]), statistics.mean(lag[c])
        out["card"][c] = {"last_share": last[c] / tot, "lag_median_us": md / 1e3, "lag_mean_us": mn / 1e3,
                          "lag_share_of_wall": mn / wall if wall else 0.0}
        print(f"  card {c}: last to arrive {100 * last[c] / tot:4.1f} % of collectives | its own lag a pass median "
              f"{md / 1e3:6.1f} mean {mn / 1e3:6.1f} us = {100 * mn / wall:4.1f} % of the pass | the other {W - 1} ranks wait "
              f"{(W - 1) * mn / 1e3:6.1f} rank-us a pass for it")
    print(f"  card {focus} by the work before the collective: " + "; ".join(
        f"{s} {by_src_n[s] // len(passes)} a pass, last {100 * by_src_last[s] / max(1, by_src_n[s]):4.1f} %, mean lag "
        f"{statistics.mean(by_src[s]) / 1e3:5.1f} us" for s in sorted(by_src)))
    if nar and nlay > 0:
        lay = defaultdict(float)
        for k in range(nar):
            lay[k * nlay // nar] += statistics.mean(per_k[k])
        vals = [lay[l] / 1e3 for l in range(nlay)]
        q = sorted(vals)
        print(f"  card {focus} lag per layer (mean us a pass, {nar} collectives over {nlay} layers): median "
              f"{statistics.median(vals):.1f}, p90 {q[int(0.9 * (len(q) - 1))]:.1f}, max {q[-1]:.1f} (layer "
              f"{vals.index(q[-1])}); layers 0-{nlay - 1}: " + " ".join(f"{v:.0f}" for v in vals))
        out["focus_layers_us"] = vals
    top = sorted(range(nar or 0), key=lambda k: -statistics.median(per_k[k]))[:6]
    srcs = ars(data[focus][kind][passes[0]])
    print(f"  card {focus} largest median lag: " + "; ".join(
        f"#{k} ({srcs[k][2] if k < len(srcs) else '?'}) {statistics.median(per_k[k]) / 1e3:.1f} us" for k in top))
    out["focus_by_src"] = {s_: statistics.mean(v) / 1e3 for s_, v in by_src.items()}
    if "--json" in opt:
        json.dump(out, open(opt["--json"], "w"))



if __name__ == "__main__":
    main()
