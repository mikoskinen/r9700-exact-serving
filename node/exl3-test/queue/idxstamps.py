#!/usr/bin/env python3
"""idxstamps.py STAMPS.CSV [--gap S] [--last N] (6.11.2026) -- the indexer's share of recorded decode passes, from
RADIANCE_AQL_STAMP's CSV (card 0; one line a dispatch of every Nth played pass: pass,seq,queue,kernel,ordered,chained,
start_ns,end_ns,play,records -- core/device/aql.cpp "dispatch timestamps").

Passes are split into segments where the next pass starts more than --gap seconds (default 2) after the previous one
ended (a prefill, which is not played, or idle time between requests): onem.py's depths come out as consecutive segments.
Per segment, over the passes of its most common tape length (the decode pass) and only complete ones:
  wall      first start to last end of the pass (median ms)
  busy      union of all dispatch intervals (two at once count once)
  idx       summed dispatch time of the index_score kernels a pass, their EXCLUSIVE time (no other dispatch of the pass
            running at the same moment: what shortens the pass when they get shorter) and the 8 dispatches in issue
            order (layers 2, 8, 14, 20, 24, 28, 32, 36 on DeepSeek-V4.1), us
  topk/cand index_topk (both forms) and index_candidates, summed (and exclusive) ms a pass
  top       the kernels with the largest summed time a pass
One "IS {json}" line a segment, then a readable block. Standard library only; reads the CSV, writes nothing."""
import csv
import json
import statistics
import sys
from collections import defaultdict


def union_len(iv):
    tot, a0, b0 = 0, None, None
    for a, b in sorted(iv):
        if b0 is None or a > b0:
            if b0 is not None:
                tot += b0 - a0
            a0, b0 = a, b
        elif b > b0:
            b0 = b
    if b0 is not None:
        tot += b0 - a0
    return tot


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def exclusive(mine, others):
    """Length of union(mine) not covered by union(others)."""
    m, o = merge(mine), merge(others)
    tot, j = 0, 0
    for a, b in m:
        cur = a
        while j < len(o) and o[j][1] <= cur:
            j += 1
        k = j
        while cur < b:
            if k >= len(o) or o[k][0] >= b:
                tot += b - cur
                break
            if o[k][0] > cur:
                tot += o[k][0] - cur
            cur = max(cur, o[k][1])
            k += 1
    return tot


def short(k):
    for key in ("index_score", "index_topk_split1", "index_topk_split2", "index_topk", "index_candidates",
                "attn_sparse", "attn_dec", "ar_", "all_reduce", "allreduce", "exl3r4d_gemm", "exl3r4d_prep",
                "moe_gemm", "gemm"):
        if key in k:
            i = k.find(key)
            return k[i:i + 48].split("(")[0]
    return k[:48]


def main():
    path = sys.argv[1]
    gap = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 2.0
    last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 0
    passes = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                passes[int(r["pass"])].append((int(r["seq"]), r["kernel"], int(r["start_ns"]), int(r["end_ns"]),
                                               int(r["records"])))
            except (ValueError, KeyError, TypeError):
                passes.setdefault(int(r.get("pass") or -1), []).append(None)
    order = sorted(passes, key=lambda p: min((x[2] for x in passes[p] if x), default=0))
    segs, cur, prev_end = [], [], None
    for p in order:
        rows = [x for x in passes[p] if x]
        if not rows:
            continue
        s0, e0 = min(x[2] for x in rows), max(x[3] for x in rows)
        if prev_end is not None and s0 - prev_end > gap * 1e9:
            segs.append(cur)
            cur = []
        cur.append(p)
        prev_end = e0
    if cur:
        segs.append(cur)
    if last:
        segs = segs[-last:]
    for si, seg in enumerate(segs):
        kinds = defaultdict(int)
        for p in seg:
            rows = passes[p]
            if rows and all(rows) and rows[0]:
                kinds[rows[0][4]] += 1
        if not kinds:
            continue
        kind = max(kinds, key=lambda k: kinds[k])
        full = [p for p in seg if all(passes[p]) and passes[p][0][4] == kind]
        n_long = max((len(passes[p]) for p in full), default=0)
        full = [p for p in full if len(passes[p]) == n_long]
        if not full:
            continue
        wall, busy, isum, iexc, tsum, texc, csum, per = [], [], [], [], [], [], [], defaultdict(list)
        kern = defaultdict(float)
        for p in full:
            rows = sorted(passes[p])
            iv = [(x[2], x[3]) for x in rows]
            wall.append((max(b for _, b in iv) - min(a for a, _ in iv)) / 1e6)
            busy.append(union_len(iv) / 1e6)
            isc = [(x[2], x[3]) for x in rows if "index_score" in x[1]]
            itk = [(x[2], x[3]) for x in rows if "index_topk" in x[1]]
            ica = [(x[2], x[3]) for x in rows if "index_candidates" in x[1]]
            oth_i = [(x[2], x[3]) for x in rows if "index_score" not in x[1]]
            oth_t = [(x[2], x[3]) for x in rows if "index_topk" not in x[1]]
            isum.append(sum(b - a for a, b in isc) / 1e6)
            iexc.append(exclusive(isc, oth_i) / 1e6)
            tsum.append(sum(b - a for a, b in itk) / 1e6)
            texc.append(exclusive(itk, oth_t) / 1e6)
            csum.append(sum(b - a for a, b in ica) / 1e6)
            for i, (a, b) in enumerate(isc):
                per[i].append((b - a) / 1e3)
            for x in rows:
                kern[short(x[1])] += (x[3] - x[2]) / 1e6
        med = lambda v: round(statistics.median(v), 3) if v else None
        top = sorted(kern.items(), key=lambda kv: -kv[1])[:10]
        d = {"segment": si, "passes": len(full), "of": len(seg), "records": kind, "dispatches": n_long,
             "wall_ms": med(wall), "busy_ms": med(busy), "idle_pct": round(100 * (1 - med(busy) / med(wall)), 1) if med(wall) else None,
             "index_score_ms": med(isum), "index_score_excl_ms": med(iexc),
             "index_score_us": [round(statistics.median(per[i]), 1) for i in sorted(per)],
             "index_topk_ms": med(tsum), "index_topk_excl_ms": med(texc), "index_candidates_ms": med(csum),
             "top_ms_a_pass": [[k, round(v / len(full), 3)] for k, v in top]}
        print("IS " + json.dumps(d))
        print(f"  segment {si}: {len(full)} of {len(seg)} passes ({n_long} dispatches), wall {d['wall_ms']} ms, busy "
              f"{d['busy_ms']} ms (idle {d['idle_pct']} %); index_score {d['index_score_ms']} ms a pass, exclusive "
              f"{d['index_score_excl_ms']} ms; per dispatch us {d['index_score_us']}; index_topk {d['index_topk_ms']} "
              f"(excl {d['index_topk_excl_ms']}), index_candidates {d['index_candidates_ms']} ms")


if __name__ == "__main__":
    main()
