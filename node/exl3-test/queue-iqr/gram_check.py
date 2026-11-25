#!/usr/bin/env python3
"""gram_check.py CALIB_DIR LO HI -- the expert calibration files of layers [LO, HI) are complete: per layer 4 ranks' pooled
gate_up (k 5120) and down (k 2304) RADGRAM2 squares and 384 per-expert down files (k 2304, packed layout 1). Prints one line:
files, GB, the per-expert rows (min / median / max) and how many experts are under libquant's 4 x K = 9216 rows (those
quantise against the layer's pooled down Gram). Exit 1 when a file is missing or malformed."""
import os, struct, sys
d, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bad, n, tot, rows = [], 0, 0, []
def hdr(p, k, lay):
    global n, tot
    try:
        with open(p, "rb") as f: h = f.read(32)
        sz = os.path.getsize(p)
    except OSError: bad.append("missing " + os.path.basename(p)); return None
    if h[:8] != b"RADGRAM2": bad.append("magic " + os.path.basename(p)); return None
    kk, rk, r, dom, l = struct.unpack_from("<IIQII", h, 8)
    want = 32 + 4 * (k * (k + 1) // 2 if lay else k * k)
    if kk != k or l != lay or sz != want: bad.append(f"shape {os.path.basename(p)} k {kk} layout {l} size {sz}"); return None
    n += 1; tot += sz; return r
for L in range(lo, hi):
    for r in range(4):
        hdr(f"{d}/gram.r{r}.blk.{L}.ffn_gate_up_exps.bin", 5120, 0); hdr(f"{d}/gram.r{r}.blk.{L}.ffn_down_exps.bin", 2304, 0)
    for e in range(384):
        x = hdr(f"{d}/gram.r{e // 96}.blk.{L}.ffn_down_exps.{e}.bin", 2304, 1)
        if x is not None: rows.append(x)
rows.sort()
under = sum(1 for x in rows if x < 9216)
med = rows[len(rows) // 2] if rows else 0
print(f"layers {lo}-{hi - 1}: {n} files {tot / 1e9:.2f} GB; per-expert rows min {rows[0] if rows else 0} median {med} "
      f"max {rows[-1] if rows else 0}; {under} of {len(rows)} experts under 9216 rows -> pooled down Gram; "
      f"{len(bad)} problems{': ' + '; '.join(bad[:5]) if bad else ''}")
sys.exit(1 if bad else 0)
