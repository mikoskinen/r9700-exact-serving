#!/usr/bin/env python3
"""opwin.py ENGINE_LOG --win LABEL:FROM:TO [...] [--ops REGEX] [--top N] [--json OUT] (12.11.2026, queue-ar, item 204) -- per window
(engine-log line numbers) of a --profile-ops run (RADIANCE_PROFILE_EVERY 1), each rank's tables subtracted (opprof.tables),
then per op name + the geometry keys that tell twins apart (bi_dec, split_rows, kc_rows, max_seqs) the calls, device ms,
us a call and lane-wait ms, summed over instances, for the ops matching REGEX (default hc_read|attn|gate_quant); plus the
window's device ms and steps. Rank 0 and the slowest rank by device ms. Standard library only."""
import json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opprof import tables   # noqa: E402
from spprof import window   # noqa: E402

KEYS = ("bi_dec", "split_rows", "kc_rows", "max_seqs", "q_len")


def tag(geom):
    out = []
    for k in KEYS:
        m = re.search(r"\b%s=(\S+)" % k, geom)
        if m: out.append("%s=%s" % (k, m.group(1)))
    return " ".join(out)


def summ(d, rx):
    ops = defaultdict(lambda: [0, 0.0, 0.0, 0])
    dev = 0.0
    for i, (c, ms, op, geom, w, wt) in d.items():
        dev += ms
        if not rx.search(op): continue
        k = (op + " " + tag(geom)).strip()
        e = ops[k]; e[0] += c; e[1] += ms; e[2] += wt; e[3] += 1
    return dev, {k: {"calls": v[0], "ms": round(v[1], 2), "us_call": round(1000 * v[1] / max(1, v[0]), 1),
                     "wait_ms": round(v[2], 2), "inst": v[3]} for k, v in sorted(ops.items(), key=lambda t: -t[1][1])}


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return 2
    path, wins, rx, jout, top = a[0], [], re.compile(r"hc_read|attn|gate_quant"), None, 0
    i = 1
    while i < len(a):
        if a[i] == "--win": wins.append(a[i + 1]); i += 2
        elif a[i] == "--ops": rx = re.compile(a[i + 1]); i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        elif a[i] == "--top": top = int(a[i + 1]); i += 2
        else: i += 1
    T = tables(path)
    res = {}
    for w in wins:
        lab, lo, hi = w.split(":")[:3]
        d, steps = window(T, int(lo), int(hi))
        per = {r: summ(v, rx) for r, v in d.items()}
        if not per: print("OPWIN %s: no tables" % lab); continue
        slow = max(per, key=lambda r: per[r][0])
        res[lab] = {"steps": steps, "ranks": {r: {"dev_ms": round(p[0], 1), "ops": p[1]} for r, p in per.items()}}
        for r in sorted({min(per), slow}):
            dev, ops = per[r]
            print("OPWIN %s rank %d: steps %s dev %.0f ms (%.2f ms a step) | %s" % (
                lab, r, steps.get(r), dev, dev / max(1, steps.get(r) or 1),
                "; ".join("%s: %d calls %.1f ms %.1f us wait %.1f" % (k, v["calls"], v["ms"], v["us_call"], v["wait_ms"])
                          for k, v in (list(ops.items())[:top] if top else ops.items()))))
    if jout:
        with open(jout, "w") as f: json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
