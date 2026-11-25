#!/usr/bin/env python3
"""stampcat.py STAMPS.csv [LABEL] [--json OUT] [--from NS --to NS] -- a recorded pass's time on card 0 split by what runs, from RADIANCE_AQL_STAMP's
CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; one line a dispatch of every Nth pass played on card
0). 8.11.2026, queue-ar, for 87-fn-tp3.sh's stamp arms (Flash-Next at TP2 / TP3 / TP4).

A pass kind is its tape length (`records`): at one stream with MTP 3 the verify pass (4 rows) is the long one and the two
draft passes (1 row) are short; at eight streams the verify pass has 32 rows. Per kind, the median over its complete passes of
  wall      first start to last end
  busy      the union of every queue's dispatch intervals (two queues at once count once)
  gaps      wall - busy: the card idle inside the pass (dispatch gaps, waits for the host or another queue)
  by class  the union of each class's intervals, the classes by kernel name:
              allreduce  ar_* / all_reduce / *_Nrank* (the collectives, fused ones included: ar_hc_write,
                         ar_gather_hc_write, ar_rmsnorm_quant_*; their time includes waiting for the slowest rank)
              attention  attention / qsa / gdn / delta-net / conv / rope / kv store kernels
              experts    moe / expert / iqr / exl3 / routed GEMMs and the router
              linear     the other GEMMs (dense projections, logits)
              other      the rest (norms, elementwise, hc_read, sampling, copies)
            Classes can overlap in time (two queues), so they need not sum to busy; `other` is busy minus the union of the
            named classes' intervals.
  ar count  collective dispatches a pass, their median duration, count x median and their summed duration (the excess of
            the sum over count x median is the few collectives that wait for a late rank)
--from / --to keep the passes whose first dispatch starts in [NS, NS] (host CLOCK_MONOTONIC ns: the stamps' clock), so one
file splits by load phase.
A pass with fewer lines than the longest of its kind (the stamp pool ran out) or a line without times is left out and counted.
The top kernel names of each class are printed, so the classification can be checked. Standard library only."""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict

CLASSES = [
    ("allreduce", re.compile(r"(^|[^a-z])ar_|all_?reduce|allreduce|[0-9]rank|oneshot|twoshot|reduce_scatter|all_gather", re.I)),
    ("attention", re.compile(r"attn|attention|qsa|gdn|delta|conv1d|causal_conv|rope|kv_store|paged|flash|decode_gq|softmax_pool", re.I)),
    ("experts", re.compile(r"moe|expert|iqr|exl3|route|router|topk|gather_rows|scatter", re.I)),
    ("linear", re.compile(r"gemm|gemv|linear|matmul|logits", re.I)),
]


def classify(name):
    for c, rx in CLASSES:
        if rx.search(name or ""):
            return c
    return "other"


def union(iv):
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


def main():
    argv = sys.argv[1:]
    opt = {}
    for k in ("--json", "--from", "--to"):
        if k in argv:
            i = argv.index(k)
            opt[k] = argv[i + 1]
            del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    jout = opt.get("--json")
    t_from = int(opt["--from"]) if "--from" in opt else None
    t_to = int(opt["--to"]) if "--to" in opt else None
    path = args[0]
    label = args[1] if len(args) > 1 else path
    passes = defaultdict(list)
    bad = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s, e = int(r["start_ns"]), int(r["end_ns"])
            except (TypeError, ValueError):
                bad += 1
                continue
            if s <= 0 or e < s:
                bad += 1
                continue
            passes[(r["pass"], r["records"])].append((s, e, r["kernel"], r["queue"]))
    kinds = defaultdict(list)
    lo = min((min(s for s, _, _, _ in rows) for rows in passes.values()), default=0)
    hi = max((max(e for _, e, _, _ in rows) for rows in passes.values()), default=0)
    inwin = 0
    for (p, rec), rows in passes.items():
        t0 = min(s for s, _, _, _ in rows)
        if (t_from is not None and t0 < t_from) or (t_to is not None and t0 > t_to):
            continue
        inwin += 1
        kinds[int(rec)].append(rows)
    if t_from is not None or t_to is not None:
        print(f"{label}: window [{t_from}, {t_to}] ns holds {inwin} of {len(passes)} passes (the file's stamps span "
              f"[{lo}, {hi}] ns; CLOCK_MONOTONIC on the host is the same domain when the window holds passes)")
    out = {"label": label, "bad_lines": bad, "kinds": []}
    for rec in sorted(kinds, reverse=True):
        lst = kinds[rec]
        full = max(len(x) for x in lst)
        keep = [x for x in lst if len(x) == full]
        if not keep:
            continue
        stats = defaultdict(list)
        names = defaultdict(lambda: defaultdict(float))
        for rows in keep:
            t0 = min(s for s, _, _, _ in rows)
            t1 = max(e for _, e, _, _ in rows)
            wall = t1 - t0
            busy = union([(s, e) for s, e, _, _ in rows])
            per = defaultdict(list)
            for s, e, k, q in rows:
                c = classify(k)
                per[c].append((s, e))
                names[c][k] += (e - s) / len(keep)
            named = union([iv for c in ("allreduce", "attention", "experts", "linear") for iv in per[c]])
            stats["wall"].append(wall)
            stats["busy"].append(busy)
            stats["gaps"].append(wall - busy)
            for c in ("allreduce", "attention", "experts", "linear"):
                stats[c].append(union(per[c]))
            stats["other"].append(busy - named)
            ars = [e - s for s, e in per["allreduce"]]
            stats["ar_n"].append(len(ars))
            stats["ar_med"].append(statistics.median(ars) if ars else 0)
            stats["ar_sum"].append(sum(ars))
            stats["dispatches"].append(len(rows))
        med = {k: statistics.median(v) for k, v in stats.items()}
        w = med["wall"] or 1
        kind = {"records": rec, "passes": len(keep), "dropped": len(lst) - len(keep), "dispatches": med["dispatches"],
                "wall_us": med["wall"] / 1e3, "busy_us": med["busy"] / 1e3, "gaps_us": med["gaps"] / 1e3,
                "allreduce_us": med["allreduce"] / 1e3, "attention_us": med["attention"] / 1e3,
                "experts_us": med["experts"] / 1e3, "linear_us": med["linear"] / 1e3, "other_us": med["other"] / 1e3,
                "ar_count": med["ar_n"], "ar_median_us": med["ar_med"] / 1e3,
                "ar_count_x_median_us": med["ar_n"] * med["ar_med"] / 1e3, "ar_sum_us": med["ar_sum"] / 1e3,
                "top": {c: sorted(names[c].items(), key=lambda kv: -kv[1])[:4] for c in names}}
        out["kinds"].append(kind)
        print(f"{label} kind records={rec}: {len(keep)} passes ({len(lst) - len(keep)} incomplete), {med['dispatches']:.0f} dispatches; "
              f"wall {kind['wall_us']:.1f} us = busy {kind['busy_us']:.1f} + gaps {kind['gaps_us']:.1f} ({100 * med['gaps'] / w:.1f} %); "
              f"allreduce {kind['allreduce_us']:.1f} us ({100 * med['allreduce'] / w:.1f} %: {med['ar_n']:.0f} collectives, median {kind['ar_median_us']:.1f} us, "
              f"sum {kind['ar_sum_us']:.1f} us -- above count x median = waits for the slowest rank), "
              f"attention {kind['attention_us']:.1f} ({100 * med['attention'] / w:.1f} %), experts {kind['experts_us']:.1f} "
              f"({100 * med['experts'] / w:.1f} %), linear {kind['linear_us']:.1f} ({100 * med['linear'] / w:.1f} %), "
              f"other {kind['other_us']:.1f} ({100 * med['other'] / w:.1f} %)")
        for c in ("allreduce", "attention", "experts", "linear", "other"):
            if names[c]:
                top = sorted(names[c].items(), key=lambda kv: -kv[1])[:4]
                print(f"    {c:9s} top: " + "; ".join(f"{k[:70]} {v / 1e3:.1f} us" for k, v in top))
    if bad:
        print(f"{label}: {bad} line(s) without times left out")
    if jout:
        with open(jout, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
