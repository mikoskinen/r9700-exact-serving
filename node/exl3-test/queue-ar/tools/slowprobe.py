#!/usr/bin/env python3
"""slowprobe.py STAMPS.csv[.gz] --from NS --to NS [--ref-us US] [--ratio R] [--mb MBFILE] [--ms-max MS] -- is this
process in the expert slow state? (9.11.2026, queue-ar, 93b). From card 0's stamps in the window (a short mb c8 probe:
32-row verify passes at TP4), the median over complete passes of the summed moe_down_dec time a pass, against a fast
reference (91 d4a at c8: 2758 us a pass; 87 s4 in the slow state: 4338).
DECISION (9.11. 05:xx, after 93b's p4m4 probe read "slow down 3627 us (1.32 of ref)" at mb c8 24.9 ms -- PR #396 changes
the expert down kernels, so the stock kernel's reference does not apply to every image): the mb file's c8 ms_decode_step
decides when it is there -- fast <= --ms-fast (26.8), slow >= --ms-max (28.5; fast TP4 lines 24.9-25.6 ms at c8, slow
30.3-31.0), "between" otherwise (no restart; the stamp ratio is printed beside it). Without an mb line the stamps decide
(slow when median > R x ref, default 1.25, at least 8 passes). Prints one line:
  "fast|slow|between|unknown (by mb|stamps) | mb c8 <ms> ms <tok/s> tok/s | down <median> us (<x> of ref, <n> passes)"
Exit status 0 fast, 1 slow, 2 unknown / between. Standard library only; tolerates a file still being written."""
import csv, gzip, io, json, statistics, sys
from collections import defaultdict


def opn(p):
    return io.TextIOWrapper(gzip.open(p), newline="") if p.endswith(".gz") else open(p, newline="")


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--ref-us", "--ratio", "--mb", "--ms-max", "--ms-fast"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    path = a[0]
    tf, tt = int(opt.get("--from", 0)), int(opt.get("--to", 1 << 62))
    ref = float(opt.get("--ref-us", 2758)); ratio = float(opt.get("--ratio", 1.25))
    msmax = max(float(opt.get("--ms-max", 28.5)), 28.5); msfast = float(opt.get("--ms-fast", 26.8))
    P = defaultdict(list)
    try:
        for r in csv.DictReader(opn(path)):
            try:
                s, e = int(r["start_ns"]), int(r["end_ns"])
            except (TypeError, ValueError, KeyError):
                continue
            if s <= 0 or e < s:
                continue
            P[(int(r["records"]), r["pass"])].append((s, e, r["kernel"]))
    except (OSError, EOFError, csv.Error):
        pass
    kinds = defaultdict(list)
    for (rec, p), v in P.items():
        t0 = min(x[0] for x in v)
        if tf <= t0 <= tt:
            kinds[rec].append(v)
    med = None; n = 0
    if kinds:
        rec = max(kinds)
        full = max(len(v) for v in kinds[rec])
        sums = [sum(e - s for s, e, k in v if "moe_down_dec" in k) / 1e3 for v in kinds[rec] if len(v) == full]
        sums = [x for x in sums if x > 0]
        n = len(sums)
        if n:
            med = statistics.median(sums)
    ms = tps = None
    if "--mb" in opt:
        try:
            for line in open(opt["--mb"]):
                if line.startswith("MB "):
                    d = json.loads(line[3:]); ms = d.get("ms_decode_step"); tps = d.get("agg_dec_tps")
        except (OSError, ValueError):
            pass
    if ms is not None and ms > 0:
        by = "mb"
        st = "fast" if ms <= msfast else ("slow" if ms >= msmax else "between")
    elif med is not None and n >= 8:
        by = "stamps"
        st = "slow" if med > ratio * ref else "fast"
    else:
        by = "-"; st = "unknown"
    print(f"{st} (by {by}) | mb c8 {ms} ms {tps} tok/s | down {med if med is None else round(med)} us "
          f"({'-' if med is None else round(med / ref, 2)} of ref {ref:.0f}, {n} passes)")
    sys.exit({"fast": 0, "slow": 1}.get(st, 2))


if __name__ == "__main__":
    main()
