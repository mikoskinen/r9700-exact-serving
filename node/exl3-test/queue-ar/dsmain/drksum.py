#!/usr/bin/env python3
"""drksum.py DIR (10.11.2026, queue-ar dsmain, ds41-drk) -- the summary of 57-p4zyzzzzz-ds-p0-drkbench's outputs: mxcheck.txt /
bfcheck.txt (r4d_selftest --mxcheck / --bfcheck: the drafter's kernel forms vs the stock kernels, byte for byte) and mx-rot /
mx-cold / bf-rot / bf-cold.txt (--perfmx / --perfbf: MXSUM / BFSUM lines).

Prints DKCHK (the check verdicts), DKMX a (projection, mode) row of stock / rule / best us over the token counts, DKBF a (shape,
mode) row, and DKROUND: a drafter round's kernel time at 1 / 2 / 4 / 8 users, stock vs the rule (3 stages x (gate_up + down)
at 5 rows a user; 3 stages x the stage's linears (q_a, q_b, kv, 2 x o_a, o_b, shared gate_up, down) at 5 rows a user plus
main_proj at the verify rows, 6 a user) -- an ESTIMATE from one card's warm / cold kernel times, not a served measurement."""
import os, re, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
def lines(f):
    p = os.path.join(d, f)
    return open(p, errors="replace").read().splitlines() if os.path.exists(p) else []
for f in ("mxcheck.txt", "bfcheck.txt"):
    for l in lines(f):
        if l.startswith(("MXVERDICT", "BFVERDICT")): print("DKCHK " + l)
    bad = [l for l in lines(f) if l.startswith(("MXCHECK", "BFCHECK")) and "DIFFERS" in l]
    for l in bad[:10]: print("DKCHK " + l[:400])
mx = {}
for mode in ("rot", "cold"):
    for l in lines(f"mx-{mode}.txt"):
        m = re.match(r"MXSUM (\S+) tokens (\d+) (\w+): stock ([\d.]+) us \| rule ([-\d.]+) \| best (\S+) ([\d.]+)", l)
        if m: mx[(m.group(1), int(m.group(2)), m.group(3))] = (float(m.group(4)), float(m.group(5)), m.group(6), float(m.group(7)))
for proj in ("gate_up", "down"):
    for mode in ("rot", "cold"):
        ks = sorted(k[1] for k in mx if k[0] == proj and k[2] == mode)
        if ks: print(f"DKMX {proj} {mode} (stock/rule/best us): " + " ".join("t%d %.0f/%.0f/%.0f" % (t, mx[(proj, t, mode)][0], mx[(proj, t, mode)][1], mx[(proj, t, mode)][3]) for t in ks))
bf = {}
for mode in ("rot", "cold"):
    for l in lines(f"bf-{mode}.txt"):
        m = re.match(r"BFSUM (\S+) (\d+)x(\d+) M=(\d+) (\w+): stock ([\d.]+) us .*\| rule ([-\d.]+) \| best (\S+) ([\d.]+)", l)
        if m: bf[(m.group(1), int(m.group(4)), m.group(5))] = (float(m.group(6)), float(m.group(7)), m.group(8), float(m.group(9)))
for sh in sorted({k[0] for k in bf}):
    for mode in ("rot", "cold"):
        ks = sorted(k[1] for k in bf if k[0] == sh and k[2] == mode)
        if ks: print(f"DKBF {sh} {mode} (stock/rule/best us): " + " ".join("M%d %.1f/%.1f/%.1f" % (M, bf[(sh, M, mode)][0], bf[(sh, M, mode)][1], bf[(sh, M, mode)][3]) for M in ks))
stage = (("q_a", 1), ("q_b", 1), ("kv", 1), ("o_a", 2), ("o_b", 1), ("shexp_gate_up", 1), ("shexp_down", 1))
for mode in ("rot", "cold"):
    out = []
    for users in (1, 2, 4, 8):
        t, v = 5 * users, 6 * users
        a = b = 0.0; miss = []
        for proj in ("gate_up", "down"):
            r = mx.get((proj, t, mode))
            if r: a += 3 * r[0]; b += 3 * (r[1] if r[1] > 0 else r[3])
            else: miss.append(f"{proj}@{t}")
        for sh, k in stage:
            r = bf.get((sh, t, mode))
            if r: a += 3 * k * r[0]; b += 3 * k * (r[1] if r[1] > 0 else r[3])
            else: miss.append(f"{sh}@{t}")
        r = bf.get(("main_proj", v, mode))
        if r: a += r[0]; b += r[1] if r[1] > 0 else r[3]
        else: miss.append(f"main_proj@{v}")
        out.append("%d user%s %.2f -> %.2f ms (%s)" % (users, "" if users == 1 else "s", a / 1e3, b / 1e3, "missing " + ",".join(miss) if miss else "complete"))
    print(f"DKROUND {mode} (ESTIMATE: drafter MoE + bf16 linears a round, stock -> the rule): " + "; ".join(out))
