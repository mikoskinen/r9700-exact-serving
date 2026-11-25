#!/usr/bin/env python3
"""mrsum.py DIR (10.11.2026, queue-ar dsmain, lever D2) -- the summary of 57-p4zyzzzzz-ds-g1-mrbench's outputs:
mrcheck.txt (r4d_selftest --mrcheck: the multi-row decode-band form vs the stock path, byte for byte, M 1..48) and
mr-rot.txt / mr-cold.txt (--perfmr: MRSUM a (type, shape, M)).

Prints:
  MRTAB <type> <shape> <mode>: M=<m> <stock>/<stock best>/<mr best> us ... (one line a (type, shape, mode))
  MRCOST <mode>: how flat the multi-row form is -- per shape, mr best at M=8 / 24 / 48 over its M=1 (stock's beside)
  MRCHK  the byte check's verdict and MRINFO (a stock row alone vs inside an M-row call)
  MRFINAL one verdict line: bytes, speedups vs the stock rule and the best pinned stock form at M 1 / 8 / 24 / 48
"""
import math
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
sumre = re.compile(r"^MRSUM (\S+) (\S+) (\d+)x(\d+) M=(\d+) (\w+): stock ([\d.]+) us \| stock best (\S+) ([\d.]+) \| "
                   r"mr rule ([-\d.]+) \| mr best (\S+) ([\d.]+) \[(.*?)\] \| x([\d.]+) vs stock, x([\d.]+) vs stock best.*bytes (\w+)")
rows = {}
for mode in ("rot", "cold"):
    p = os.path.join(d, f"mr-{mode}.txt")
    if not os.path.exists(p):
        continue
    for line in open(p, errors="replace"):
        m = sumre.match(line)
        if not m:
            continue
        ty, sh, N, K, M, md = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)
        rows[(ty, sh, md, M)] = dict(stock=float(m.group(7)), sbest_nm=m.group(8), sbest=float(m.group(9)),
                                    rule=float(m.group(10)), best_nm=m.group(11), best=float(m.group(12)),
                                    form=m.group(13), bytes=m.group(16))

keys = sorted({(k[0], k[1], k[2]) for k in rows})
for ty, sh, md in keys:
    ms = sorted(k[3] for k in rows if k[:3] == (ty, sh, md))
    cells = []
    for M in ms:
        r = rows[(ty, sh, md, M)]
        cells.append(f"M{M} {r['stock']:.1f}/{r['sbest']:.1f}/{r['best']:.1f}")
    print(f"MRTAB {ty} {sh} {md} (stock/stock best/mr best us): " + " ".join(cells))

for md in ("rot", "cold"):
    parts = []
    for ty, sh, mm in keys:
        if mm != md or (ty, sh, md, 1) not in rows:
            continue
        r1 = rows[(ty, sh, md, 1)]
        seg = []
        for M in (8, 24, 48):
            if (ty, sh, md, M) in rows:
                r = rows[(ty, sh, md, M)]
                seg.append(f"{M}: mr x{r['best'] / r1['best']:.2f} stock x{r['stock'] / r1['stock']:.2f}")
        parts.append(f"{ty} {sh} [{'; '.join(seg)}]")
    if parts:
        print(f"MRCOST {md} (cost at M over cost at M=1): " + " | ".join(parts))

chk = os.path.join(d, "mrcheck.txt")
nchk = nbad = 0
if os.path.exists(chk):
    for line in open(chk, errors="replace"):
        if line.startswith("MRCHECK "):
            nchk += 1
            if "DIFFERS" in line:
                nbad += 1
                print(line.rstrip()[:600])
        if line.startswith("MRVERDICT check"):
            print("MRCHK " + line.rstrip())
    print(f"MRCHK {nchk} (type, shape) lines, {nbad} with a difference")

for md in ("rot", "cold"):
    out = []
    for M in (1, 8, 24, 48):
        sel = [r for k, r in rows.items() if k[2] == md and k[3] == M]
        if not sel:
            continue
        g1 = math.exp(sum(math.log(r["stock"] / r["best"]) for r in sel) / len(sel))
        g2 = math.exp(sum(math.log(r["sbest"] / r["best"]) for r in sel) / len(sel))
        out.append(f"M={M} x{g1:.2f} vs stock, x{g2:.2f} vs stock best ({len(sel)} shapes)")
    diff = sum(1 for k, r in rows.items() if k[2] == md and r["bytes"] != "identical")
    if out:
        print(f"MRFINAL {md}: bytes {'identical' if not diff else f'DIFFER at {diff}'}; geomean " + "; ".join(out))
