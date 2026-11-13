#!/usr/bin/env python3
"""opdiff.py OPWIN.json --arm NAME=LAB[,LAB...] [...] --base NAME [--top 15] (13.11.2026, queue-ar, item 218) -- per-op attribution
between arms of a profiled run: opwin.py's --json (windows of the same work, e.g. one seeded mbt2 generation each; --per-table divides each window by its table count when the step
counts differ), each arm the
mean over its windows of every op's device ms / calls a window, on rank 0 and on the rank with the most device ms; then each arm
against --base: the window's device ms (and its delta, %), per op NAME (geometry tags dropped: a twin and its stock op fold
together) the delta ms, calls and us a call, the largest |delta| first; and per op KEY (name + geometry tags) the same, so a twin
appearing and the stock op disappearing show apart. Lines "OPDIFF ...". Standard library only."""
import json, sys
from collections import defaultdict


PER_TABLE = False   # --per-table: each window's numbers / its table count (windows of unequal step counts)


def arm_mean(R, labs, rank_sel):
    """{key: [calls, ms, wait]} mean over the windows, the device ms mean, the windows used"""
    acc, dev, n = defaultdict(lambda: [0.0, 0.0, 0.0]), 0.0, 0
    for lab in labs:
        w = R.get(lab)
        if not w or not w.get("ranks"): continue
        rk = w["ranks"]
        r = "0" if rank_sel == "0" and "0" in rk else max(rk, key=lambda k: rk[k]["dev_ms"])
        f = 1.0 / max(1, (w.get("steps") or {}).get(r) or 1) if PER_TABLE else 1.0
        n += 1; dev += rk[r]["dev_ms"] * f
        for k, v in rk[r]["ops"].items():
            e = acc[k]; e[0] += v["calls"] * f; e[1] += v["ms"] * f; e[2] += v["wait_ms"] * f
    if not n: return None, None, 0
    return {k: [v[0] / n, v[1] / n, v[2] / n] for k, v in acc.items()}, dev / n, n


def by_name(d):
    out = defaultdict(lambda: [0.0, 0.0, 0.0])
    for k, v in d.items():
        e = out[k.split(" ")[0]]; e[0] += v[0]; e[1] += v[1]; e[2] += v[2]
    return out


def fmt_rows(b, a, top):
    keys = set(b) | set(a)
    rows = []
    for k in keys:
        cb, mb, _ = b.get(k, (0, 0, 0)); ca, ma, _ = a.get(k, (0, 0, 0))
        rows.append((ma - mb, k, mb, ma, cb, ca))
    rows.sort(key=lambda t: -abs(t[0]))
    us = lambda m, c: ("%.1f" % (1000 * m / c)) if c else "-"
    return "; ".join("%s %+.1f ms (%.1f -> %.1f, calls %.0f -> %.0f, us %s -> %s)" % (k, d, mb, ma, cb, ca, us(mb, cb), us(ma, ca))
                     for d, k, mb, ma, cb, ca in rows[:top])


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"): print(__doc__); return 2
    R = json.load(open(a[0]))
    arms, base, top = {}, None, 15
    i = 1
    while i < len(a):
        if a[i] == "--arm": n, l = a[i + 1].split("=", 1); arms[n] = [x for x in l.split(",") if x]; i += 2
        elif a[i] == "--base": base = a[i + 1]; i += 2
        elif a[i] == "--top": top = int(a[i + 1]); i += 2
        elif a[i] == "--per-table": global PER_TABLE; PER_TABLE = True; i += 1
        else: i += 1
    if base not in arms: print("OPDIFF VIRHE --base %s is not an --arm" % base); return 1
    for rank_sel in ("0", "max"):
        B, devb, nb = arm_mean(R, arms[base], rank_sel)
        if B is None: print("OPDIFF VIRHE base %s: no windows with tables" % base); return 1
        for name, labs in arms.items():
            if name == base: continue
            A, deva, na = arm_mean(R, labs, rank_sel)
            if A is None: print("OPDIFF %s vs %s rank %s: no windows with tables" % (name, base, rank_sel)); continue
            print(("OPDIFF %s vs %s rank %s (windows %d / %d): device ms a " + ("table (PROFILE_EVERY steps)" if PER_TABLE else "window") + " %.1f -> %.1f (%+.1f, %+.2f %%) | by op: %s") % (
                name, base, rank_sel, na, nb, devb, deva, deva - devb, 100 * (deva - devb) / max(1e-9, devb),
                fmt_rows(by_name(B), by_name(A), top)))
            print("OPDIFF %s vs %s rank %s by key: %s" % (name, base, rank_sel, fmt_rows(B, A, top)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
