#!/usr/bin/env python3
"""rankskew.py STAMPS.csv [--from NS --to NS] [--kind RECORDS] [--label L] [--json OUT]
Per-rank view of RADIANCE_AQL_STAMP_CARDS=all stamps (8.11.2026, queue-ar, item 91): card 0's file is STAMPS.csv, card k's
STAMPS.csv.c<k> (whichever exist). All cards stamp the same pass numbers (lockstep) in the host's CLOCK_MONOTONIC domain.

For the longest pass kind in the window (the verify pass; --kind picks another), over the passes complete on every card:
  per card     wall, busy, gaps; exposed time (covered by no other dispatch) of experts / linear / attention / allreduce /
               other (stampcat.py's classes); the expert slow state: a layer's moe_down_dec counts as slow when it runs
               > 1.6 x its own p10 over the window (the 8.11. s3/s4 finding: fast or fast + ~105 us), share of slow layers
  collectives  the k-th collective of a pass is the same all-reduce on every card. Arrival skew: wait_r = (latest start
               over the cards) - start_r, i.e. how long card r sat in it for the last card to arrive; intrinsic =
               end_r - latest start. Per card: summed wait and intrinsic a pass (median over passes), and how often each card
               arrives last, by the class of the work before the collective (moe / gdn / attn / other).
  slow-state coincidence: for each (pass, layer), how many cards are slow at once (0..W), against what independent cards
               with the same per-card shares would give.
Standard library only."""
import csv, glob, json, os, re, statistics, sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stampcat import classify, union  # noqa: E402


def load(path, kind, tf, tt):
    P = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        if kind is not None and r["records"] != kind:
            continue
        try:
            s, e = int(r["start_ns"]), int(r["end_ns"])
        except (TypeError, ValueError):
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


def exposed(rows):
    """Per-dispatch exposed ns (its interval minus the part any other dispatch covers), neighbours +-8."""
    res = []
    for i, (_, s, e, _) in enumerate(rows):
        cov = []
        for j in range(max(0, i - 8), min(len(rows), i + 9)):
            if j != i:
                a, b = max(s, rows[j][1]), min(e, rows[j][2])
                if b > a:
                    cov.append((a, b))
        res.append((e - s) - union(cov))
    return res


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--kind", "--label", "--json"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    base = a[0]
    tf = int(opt.get("--from", 0)); tt = int(opt.get("--to", 1 << 62))
    files = [base] + sorted(glob.glob(base + ".c*"), key=lambda f: int(f.rsplit(".c", 1)[1]))
    label = opt.get("--label", base)
    data = [load(f, None, tf, tt) for f in files]
    W = len(files)
    kinds = set(data[0].keys())
    for d in data[1:]:
        kinds &= set(d.keys())
    if not kinds:
        print(f"{label}: no pass kind on every card ({W} files)"); return
    kind = int(opt["--kind"]) if "--kind" in opt else max(kinds)
    common = set(data[0][kind].keys())
    for d in data[1:]:
        common &= set(d[kind].keys())
    full = [max(len(v) for v in d[kind].values()) for d in data]
    passes = sorted(p for p in common if all(len(d[kind][p]) == full[c] for c, d in enumerate(data)))
    print(f"{label}: {W} card file(s), kind records={kind}, {len(passes)} passes complete on every card "
          f"(dispatches per card {full})")
    if not passes:
        return
    out = {"label": label, "cards": W, "kind": kind, "passes": len(passes), "card": [], "ar": {}, "coincide": {}}
    # per card summaries + slow layers
    slow = []   # slow[c][pass] = list of bool per layer
    for c, d in enumerate(data):
        st = defaultdict(list)
        downs = defaultdict(list)
        for p in passes:
            rows = d[kind][p]
            dn = [(x[2] - x[1]) for x in rows if "moe_down_dec" in x[3]]
            downs[p] = dn
        L = len(next(iter(downs.values())))
        p10 = [sorted(downs[p][l] for p in passes)[len(passes) // 10] for l in range(L)] if L else []
        sl = {p: [downs[p][l] > 1.6 * p10[l] for l in range(L)] for p in passes}
        slow.append(sl)
        for p in passes:
            rows = d[kind][p]
            t0 = rows[0][1]; t1 = max(x[2] for x in rows)
            busy = union([(x[1], x[2]) for x in rows])
            ex = exposed(rows)
            cl = defaultdict(float)
            for x, e in zip(rows, ex):
                cl[classify(x[3])] += e
            st["wall"].append(t1 - t0); st["busy"].append(busy); st["gaps"].append(t1 - t0 - busy)
            for k in ("experts", "linear", "attention", "allreduce", "other"):
                st[k].append(cl[k])
        med = {k: statistics.median(v) / 1e3 for k, v in st.items()}
        share = (sum(sum(v) for v in sl.values()) / max(1, len(passes) * L)) if L else 0.0
        med["slow_share"] = share
        out["card"].append(med)
        print(f"  card {c}: wall {med['wall']:.0f} busy {med['busy']:.0f} gaps {med['gaps']:.0f} | exposed experts "
              f"{med['experts']:.0f} linear {med['linear']:.0f} attention {med['attention']:.0f} allreduce "
              f"{med['allreduce']:.0f} other {med['other']:.0f} us | slow layers {100 * share:.1f} %")
    # collectives: k-th collective across cards
    def ars(rows):
        res = []
        for i, x in enumerate(rows):
            if classify(x[3]) == "allreduce":
                j = i - 1
                while j > 0 and classify(rows[j][3]) in ("other",) and ("aql_copy" in rows[j][3] or "gather" in rows[j][3]):
                    j -= 1
                prev = rows[j][3] if j >= 0 else ""
                src = ("moe" if re.search(r"moe|gemm_fp8a8|gather", prev) and any("moe_down" in rows[q][3] for q in range(max(0, i - 6), i))
                       else "gdn" if any("gdn" in rows[q][3] for q in range(max(0, i - 4), i))
                       else "attn" if any(re.search(r"attn|qsa", rows[q][3]) for q in range(max(0, i - 6), i)) else "other")
                res.append((x[1], x[2], src))
        return res
    waits = defaultdict(list); intr = defaultdict(list)
    last = defaultdict(Counter); bysrc_wait = defaultdict(lambda: defaultdict(list))
    nar = None
    for p in passes:
        A = [ars(d[kind][p]) for d in data]
        n = min(len(x) for x in A)
        nar = n if nar is None else min(nar, n)
        pw = [0.0] * W; pi = [0.0] * W
        for k in range(n):
            starts = [A[c][k][0] for c in range(W)]
            ms = max(starts)
            lc = starts.index(ms)
            src = A[0][k][2]
            last[src][lc] += 1
            for c in range(W):
                pw[c] += ms - starts[c]
                pi[c] += max(0, A[c][k][1] - ms)
                bysrc_wait[src][c].append(ms - starts[c])
        for c in range(W):
            waits[c].append(pw[c]); intr[c].append(pi[c])
    print(f"  collectives a pass: {nar}; per card, median over passes of the summed wait for the last arrival / the "
          f"intrinsic part (end - last start), us:")
    for c in range(W):
        print(f"    card {c}: wait {statistics.median(waits[c]) / 1e3:7.1f}  intrinsic {statistics.median(intr[c]) / 1e3:7.1f}")
        out["ar"][f"card{c}"] = {"wait_us": statistics.median(waits[c]) / 1e3, "intrinsic_us": statistics.median(intr[c]) / 1e3}
    for src in sorted(last):
        tot = sum(last[src].values())
        mw = [statistics.mean(bysrc_wait[src][c]) / 1e3 for c in range(W)]
        print(f"    after {src:5s}: last to arrive " + " ".join(f"c{c} {100 * last[src][c] / tot:4.1f}%" for c in range(W)) +
              " | mean wait " + " ".join(f"c{c} {mw[c]:5.1f}" for c in range(W)) + f" us ({tot // len(passes)} a pass)")
        out["ar"][f"after_{src}"] = {"last_share": [last[src][c] / tot for c in range(W)], "mean_wait_us": mw}
    # coincidence of slow layers across cards
    if W > 1 and slow[0]:
        L = len(next(iter(slow[0].values())))
        cnt = Counter()
        for p in passes:
            for l in range(L):
                cnt[sum(1 for c in range(W) if slow[c][p][l])] += 1
        tot = sum(cnt.values())
        sh = [out["card"][c]["slow_share"] for c in range(W)]
        # independent expectation
        exp = [1.0]
        for s in sh:
            nxt = [0.0] * (len(exp) + 1)
            for i, v in enumerate(exp):
                nxt[i] += v * (1 - s); nxt[i + 1] += v * s
            exp = nxt
        print("  slow layers at once (pass x layer): " + " ".join(f"{k}: {100 * cnt[k] / tot:4.1f}% (indep {100 * exp[k]:4.1f}%)" for k in range(W + 1)))
        out["coincide"] = {"observed": [cnt[k] / tot for k in range(W + 1)], "independent": exp}
    if "--json" in opt:
        json.dump(out, open(opt["--json"], "w"))


if __name__ == "__main__":
    main()
