#!/usr/bin/env python3
"""gapsize.py STAMPS.csv[.gz] [--from NS --to NS] [--label L] [--json OUT] (9.11.2026, queue-ar, items 115/117) -- does the
idle gap in front of a DEPENDENT launch (barrier bit, same queue as the dispatch it waits for) grow with the kernels around
it? From RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records), inside the trunk
pass kind (the longest tape) only, played stretches only (next.play == prev.play, so host work between stretches is out):
for every ordered dispatch whose predecessor on its queue ended last before it started, gap = start(next) - end(prev).
Binned by the PREVIOUS kernel's duration and, separately, by the NEXT kernel's duration: count, median and mean gap,
and the share of gaps under 2 us. A gap that grows with the previous kernel points at its drain (the release at its end,
the tail); one that grows with the next at its ramp. Also the ordered dispatches' median gap per pass and their count.
Standard library only."""
import csv, gzip, json, statistics, sys
from collections import defaultdict

BINS = [(0, 5e3, "<5us"), (5e3, 20e3, "5-20us"), (20e3, 100e3, "20-100us"), (100e3, 1e18, ">100us")]


def opener(p):
    return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")


def binof(d):
    for lo, hi, n in BINS:
        if lo <= d < hi:
            return n
    return BINS[-1][2]


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 2
    path, lo, hi, label, jout = a[0], None, None, "", None
    i = 1
    while i < len(a):
        if a[i] == "--from": lo = int(a[i + 1]); i += 2
        elif a[i] == "--to": hi = int(a[i + 1]); i += 2
        elif a[i] == "--label": label = a[i + 1]; i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        else: i += 1
    passes = defaultdict(list)
    recs = {}
    with opener(path) as f:
        for r in csv.DictReader(f):
            try:
                s, e = int(r["start_ns"]), int(r["end_ns"])
            except (ValueError, KeyError, TypeError):
                continue
            if lo is not None and s < lo: continue
            if hi is not None and s > hi: continue
            p = r["pass"]
            passes[p].append((int(r["seq"]), s, e, r["queue"], r["ordered"] == "1", r["chained"] == "1", r["play"], r["kernel"]))
            recs[p] = int(r["records"])
    if not passes:
        print(f"GAPSIZE {label}: no stamped passes in the window"); return 1
    trunk = max(recs.values())
    byprev, bynext = defaultdict(list), defaultdict(list)
    per_pass_n, per_pass_med = [], []
    for p, rows in passes.items():
        if recs[p] != trunk:
            continue
        rows.sort(key=lambda x: (x[1], x[0]))
        lastq = {}     # queue -> (end, start, play) of its latest-ending dispatch so far
        n_here, gaps_here = 0, []
        for seq, s, e, q, ordd, ch, play, k in rows:
            pv = lastq.get(q)
            if ordd and not ch and pv is not None and pv[2] == play and s >= pv[0]:
                g = s - pv[0]
                byprev[binof(pv[0] - pv[1])].append(g)
                bynext[binof(e - s)].append(g)
                n_here += 1; gaps_here.append(g)
            if pv is None or e > pv[0]:
                lastq[q] = (e, s, play)
        per_pass_n.append(n_here)
        if gaps_here: per_pass_med.append(statistics.median(gaps_here))

    def summ(d):
        out = {}
        for _, _, n in BINS:
            v = d.get(n, [])
            if not v: continue
            out[n] = {"n": len(v), "median_us": round(statistics.median(v) / 1e3, 2), "mean_us": round(sum(v) / len(v) / 1e3, 2),
                      "under2us_pct": round(100.0 * sum(1 for x in v if x < 2e3) / len(v), 1)}
        return out
    res = {"label": label, "trunk_records": trunk, "trunk_passes": len(per_pass_n),
           "ordered_gaps_a_pass": round(statistics.median(per_pass_n), 1) if per_pass_n else 0,
           "median_gap_us": round(statistics.median(per_pass_med) / 1e3, 2) if per_pass_med else None,
           "by_prev_duration": summ(byprev), "by_next_duration": summ(bynext)}
    if jout:
        with open(jout, "w") as f: json.dump(res, f)
    fm = lambda d: " ".join(f"{k}: n {v['n']} med {v['median_us']} mean {v['mean_us']} <2us {v['under2us_pct']}%" for k, v in d.items())
    print(f"GAPSIZE {label}: trunk {trunk} records, {res['trunk_passes']} passes, {res['ordered_gaps_a_pass']} same-queue ordered gaps a pass, "
          f"median {res['median_gap_us']} us | by prev kernel: {fm(res['by_prev_duration'])} | by next kernel: {fm(res['by_next_duration'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
