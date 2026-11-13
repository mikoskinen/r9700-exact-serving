#!/usr/bin/env python3
"""kbsum.py DIR (10.11.2026, queue-ar dsmain) -- the rebuilt expert-kernel table from the dsmoe-*.txt files of
57-p4zyzzzzz-ds-bzz-bench (r4d_selftest --perfdsmoe) and the dense-GEMV summary from gg-*.txt (--perfgg).
Expert rows: per format and M, us a gate_up + down layer (the slowest of 4 ranks' rows, expert parallel 96 of 384, top 6),
MiB an expert, bits a weight, GB/s of the touched experts; EXL3 2.25 = (11 x K2.5 + 29 x K2) / 40 layers, 2.9 = (35 x K3 +
5 x K2) / 40 (the containers' per-layer K). Then the residency model (stated in the output): VRAM for routed experts at
TP4 with today's KV / pool = 15.63 GiB a card (RELEASE-PLAN, the x117-best524 split); share r = 4 x 15.63 / total;
the share of routed READS that hit VRAM from a Zipf popularity over 384 experts a layer whose exponent is fitted to item
a's EXL3 2.25 hit (97.9 % at r = 0.46); estimated expert ms a step at 1 / 4 / 8 users = 40 layers x (the measured layer
time at that M + the missed experts' bytes over the host link at 25 GB/s a card)."""
import json, math, os, re, sys

D = sys.argv[1]
LAYER = re.compile(r"^DSMOE (\S+) layer M=(\d+) gate_up \+ down ([\d.]+) us.*?(?:([\d.]+) MiB an expert: ([\d.]+) bits a weight)?\)?$")
PROJ = re.compile(r"^DSMOE (\S+) (gate_up|down) M=(\d+) (?:\S+ )?([\d.]+) us ([\d.]+) GB/s")
ROUTE = re.compile(r"slowest rank \d+ gets (\d+) routed rows over (\d+) experts")
def load(tag):
    out, route = {}, {}
    for f in sorted(os.listdir(D)):
        m = re.match(r"dsmoe-%s-m(\d+)\.txt$" % tag, f)
        if not m: continue
        M = int(m.group(1))
        for l in open(os.path.join(D, f), errors="replace"):
            r = ROUTE.search(l)
            if r: route[M] = (int(r.group(1)), int(r.group(2)))
            x = LAYER.match(l.strip())
            if x:
                fmt = x.group(1); us = float(x.group(3))
                mib = float(x.group(4)) if x.group(4) else None; bpw = float(x.group(5)) if x.group(5) else None
                if us > 0: out[(fmt, M)] = {"us": us, "mib": mib, "bpw": bpw}
    return out, route
cold, route = load("cold")
warm, _ = load("warm")
for M in sorted({m for _, m in cold}):
    for name, mix in (("exl3_2.25", {"exl3k2h": 11, "exl3k2": 29}), ("exl3_2.9", {"exl3k3": 35, "exl3k2": 5})):
        if all((k, M) in cold for k in mix):
            us = sum(cold[(k, M)]["us"] * n for k, n in mix.items()) / 40.0
            mib = sum(cold[(k, M)]["mib"] * n for k, n in mix.items()) / 40.0
            cold[(name, M)] = {"us": us, "mib": mib, "bpw": mib * 1048576 * 8 / (4608.0 * 5120 + 5120.0 * 2304)}
TOTAL = {"mx": 268.95, "exl3_2.25": 135.90, "exl3_2.9": 182.59}
for f in ("w4", "iqr4", "iqr3", "iqr2", "exl3k2", "exl3k2h", "exl3k3"):
    k = next((v for (ff, m), v in cold.items() if ff == f and v.get("mib")), None)
    if k: TOTAL.setdefault(f, k["mib"] * 384 * 40 / 1024.0)
VRAM = 4 * 15.63
def capture(r, s, N=384):
    k = max(1, int(round(r * N)))
    h = lambda n: sum(1.0 / (i ** s) for i in range(1, n + 1))
    return h(min(k, N)) / h(N)
lo, hi = 0.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    if capture(0.46, mid) < 0.979: lo = mid
    else: hi = mid
S = (lo + hi) / 2
def at(fmt, M):
    if (fmt, M) in cold: return cold[(fmt, M)]["us"]
    if M == 8 and (fmt, 4) in cold and (fmt, 16) in cold:
        a, b = cold[(fmt, 4)]["us"], cold[(fmt, 16)]["us"]; return a + (b - a) * (8 - 4) / 12.0
    return None
def rexp(M):
    if M in route: return route[M][1]
    if M == 8 and 4 in route and 16 in route: return route[4][1] + (route[16][1] - route[4][1]) * 4 / 12.0
    return None
print("KBSUM model: Zipf exponent %.3f fitted to EXL3 2.25's 97.9 %% read hit at r 0.46; VRAM for experts %.2f GiB (4 cards)" % (S, VRAM))
fmts = ["mx", "w4", "iqr4", "iqr3", "iqr2", "exl3_2.25", "exl3_2.9", "exl3k2", "exl3k2h", "exl3k3"]
for f in fmts:
    rows = [(M, cold[(f, M)]) for M in (1, 4, 16, 64) if (f, M) in cold]
    if not rows: continue
    tot = TOTAL.get(f); r = min(1.0, VRAM / tot) if tot else None; hit = capture(r, S) if r else None
    mib = rows[0][1].get("mib"); bpw = rows[0][1].get("bpw")
    lay = " ".join("M%d %.1f us" % (M, v["us"]) for M, v in rows)
    # MB / us x 1000 = GB/s (10.11. fix: the line divided by 1000 again and printed MB/us)
    gbs = " ".join("M%d %.0f GB/s" % (M, (rexp(M) or 0) * (mib or 0) * 1.048576 / v["us"] * 1000) for M, v in rows if mib)
    est = []
    for users, M in ((1, 1), (4, 4), (8, 8)):
        u = at(f, M); e = rexp(M)
        if u is None or e is None or mib is None or hit is None: est.append("%du n/a" % users); continue
        miss_ms = 40 * e * (1 - hit) * mib * 1.048576 / 25.0          # MB / (GB/s) = ms
        est.append("%du %.1f ms (kernels %.1f + link %.1f)" % (users, 40 * u / 1000 + miss_ms, 40 * u / 1000, miss_ms))
    warm1 = warm.get((f, 1), {}).get("us")
    print("KBSUM %-10s | layer cold %s | warm M1 %s | %.2f MiB an expert, %.2f bpw | %s | model %s GiB, VRAM share %s, read hit %s | est. expert ms a step %s" % (
        f, lay, ("%.1f us" % warm1) if warm1 else "-", mib or 0, bpw or 0, gbs, ("%.1f" % tot) if tot else "?",
        ("%.0f %%" % (100 * r)) if r else "?", ("%.1f %%" % (100 * hit)) if hit else "?", "; ".join(est)))
# dense GEMV (lever 1): rule vs the best pinned form a shape / type / M
best = []
for f in sorted(os.listdir(D)):
    if not f.startswith("gg-"): continue
    for l in open(os.path.join(D, f), errors="replace"):
        if not l.startswith("GGPERF "): continue
        head, _, rest = l.partition(":")
        forms = re.findall(r" (rule|r\dg\d+) ([\d.]+) us (\d+) GB/s( OUTPUT DIFFERS)?;", rest)
        if not forms: continue
        rule = next((float(u) for n, u, g, d in forms if n == "rule"), None)
        b = min(forms, key=lambda x: float(x[1]))
        diff = [n for n, u, g, d in forms if d]
        best.append((head[7:], rule, b[0], float(b[1]), int(b[2]), diff))
for h, rule, bn, bu, bg, diff in best:
    print("GGSUM %s | rule %.1f us | best %s %.1f us %d GB/s (%+.1f %%)%s" % (h, rule or 0, bn, bu, bg, (bu / rule - 1) * 100 if rule else 0,
          (" | OUTPUT DIFFERS in " + ",".join(diff)) if diff else " | bytes match"))
