#!/usr/bin/env python3
"""idxbal.py ENGINE_LOG --from LINE [--to LINE] [--label L] [--limit 5] (10.11.2026, queue-ar dsmain, ds41-idxb, lever I1)

The indexer's per-rank work in a window of a --profile-ops run: for each rank, the window's device ms of index_score (all
of its row_part instances summed), index_topk, index_candidates and the row split's selection all_gather (the f32 gather
with numel=, deepseek41's sel_gather), each rank's last complete table at or before --from subtracted from its last one
at or before --to (the tables are cumulative; Ctx::dump_profile). Steps = the calls of the rank's lowest-index op.

Why summed per rank and not matched per op instance: under the row split a row_part instance (#index) runs on the rank
that holds that run only (rank r takes runs [r q, r q + q)), so per-instance means compare one rank's real run with
another rank's empty or short call; a per-instance max-rank sum then reads as a 2:2:1:1 skew that is not there (the
10.11. dsm-225-prof 128k window: per-instance r0 138 / r1 145 / r2 68 / r3 68 ms "a step", per rank summed 4135 / 4148 /
4195 / 4260 ms). The rank sums are what each rank spends before the gather that joins them.

Prints one IDXBAL line: per rank ms (and ms a step) of each, the index_score spread (max / min - 1 and max / mean - 1),
and PASS when max / min - 1 <= --limit %, else OVER. Then the runs a rank: per rank the row_part instances that ran
(calls > 0 in the window, summed over layers) and their ms. Standard library only."""
import argparse, json, re, sys
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HDR = re.compile(r"per-op timing, rank (\d+)")


def tables(path):
    """[(end_line, rank, {index: (calls, ms, op, geom)})] in log order (complete tables only)."""
    out, cur, rank = [], None, None
    with open(path, errors="replace") as f:
        for n, l in enumerate(f, 1):
            m = HDR.search(l)
            if m:
                cur, rank = {}, int(m.group(1)); continue
            if cur is None: continue
            r = ROW.match(l)
            if r:
                cur[int(r.group(9))] = (int(r.group(1)), float(r.group(2)), r.group(7), r.group(8))
            elif l.strip().startswith("total"):
                out.append((n, rank, cur)); cur = None
    return out


def last_at(tabs, rank, line):
    best = {}
    for n, r, t in tabs:
        if r == rank and n <= line: best = t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="l0", type=int, required=True)
    ap.add_argument("--to", dest="l1", type=int, default=1 << 62)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=float, default=5.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    tabs = tables(a.log)
    ranks = sorted({r for _, r, _ in tabs})
    res = {"label": a.label, "ranks": {}}
    for rk in ranks:
        t0, t1 = last_at(tabs, rk, a.l0), last_at(tabs, rk, a.l1)
        d = {}
        for i, (c, ms, op, geom) in t1.items():
            c0, ms0 = t0.get(i, (0, 0.0, op, geom))[:2]
            if c - c0 > 0: d[i] = (c - c0, ms - ms0, op, geom)
        steps = d[min(d)][0] if d else 0
        sums = defaultdict(float); parts = defaultdict(lambda: [0, 0.0])
        for i, (c, ms, op, geom) in d.items():
            if op == "index_score":
                sums["index_score"] += ms
                pm = re.search(r"row_part=(\d+)", geom)
                p = parts[int(pm.group(1)) if pm else -1]; p[0] += c; p[1] += ms
            elif op in ("index_topk", "index_candidates"):
                sums[op] += ms
            elif op == "all_gather" and "numel=" in geom and "dtype=f32" in geom:
                sums["sel_gather"] += ms
        res["ranks"][rk] = {"steps": steps, **{k: round(v, 1) for k, v in sums.items()},
                            "parts": {p: [v[0], round(v[1], 1)] for p, v in sorted(parts.items())}}
    sc = [res["ranks"][r].get("index_score", 0.0) for r in ranks]
    if not sc or min(sc) <= 0:
        print(f"IDXBAL {a.label}: no index_score in the window ({len(tabs)} tables)"); return 1
    spread = max(sc) / min(sc) - 1; over_mean = max(sc) / (sum(sc) / len(sc)) - 1
    res.update(spread_pct=round(100 * spread, 2), max_over_mean_pct=round(100 * over_mean, 2),
               verdict="PASS" if 100 * spread <= a.limit else "OVER")
    fmt = lambda r, k: "%s %.1f ms (%.2f a step)" % (k, res["ranks"][r].get(k, 0.0), res["ranks"][r].get(k, 0.0) / max(1, res["ranks"][r]["steps"]))
    print(f"IDXBAL {a.label}: index_score per rank " + " / ".join("%.1f" % s for s in sc) + " ms in the window; spread "
          f"max/min {100 * spread:+.2f} %, max/mean {100 * over_mean:+.2f} % -> {res['verdict']} (limit {a.limit:g} %) | "
          + " | ".join(f"rank {r} ({res['ranks'][r]['steps']} steps): " + ", ".join(fmt(r, k) for k in
                       ("index_score", "index_topk", "index_candidates", "sel_gather")) for r in ranks))
    for r in ranks:
        pr = res["ranks"][r]["parts"]
        big = {p: v for p, v in pr.items() if v[1] >= 0.05 * max(x[1] for x in pr.values())}
        print(f"IDXRUNS {a.label} rank {r}: row_parts with work " + " ".join(f"{p}:{v[1]:.0f}ms/{v[0]}" for p, v in big.items())
              + f" (all parts that ran: {len(pr)})")
    if a.json:
        json.dump(res, open(a.json, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
