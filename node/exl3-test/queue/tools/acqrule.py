#!/usr/bin/env python3
"""acqrule.py ENGINE_LOG [--line 128] [--w] [--us 0.9] -- how many played dispatches could drop their agent-scope acquire under the
address rule, from the engine's TAPEDUMP lines (RADIANCE_DEBUG_TAPE_DUMP=1: each recorded pass's dispatches once a key,
with the byte ranges its op reads and writes; core/runtime/ctx.cpp haz_mark).

The rule (6.11.2026): an acquire invalidates every CU's caches; between two acquires (a window) a CU can only hold a
stale line if that line was READ (cached) and WRITTEN (from another CU) inside the window. A dispatch may skip its
acquire iff none of the lines it reads is both read and written in the current window. A dispatch that keeps its
acquire starts a new window (its own reads and writes). An op without ranges (opaque: collectives, unbounded operands,
no schema) keeps its acquire and leaves the window unknown, so the next dispatch keeps its own too. Ranges are widened
to whole cache lines (--line, 128 B). --w: rule-w (core-acq), every line written in the window is risky too. --us: an
agent acquire's cost (gapbench 75: ~0.9 us); the us column = skippable ordered dispatches x it (a pass's bound).
Each tape is labelled with its "begin draft D tok T" line (pass kind, tokens) where the dump has one.
Events (waits) do not invalidate. Per tape (rank, recorded pass): dispatches,
ordered ones, and those the rule lets skip, split ordered / unordered."""
import re, sys, collections
LINE = 128
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--line']
if '--line' in sys.argv: LINE = int(sys.argv[sys.argv.index('--line') + 1])
WMODE = '--w' in sys.argv   # rule-w (core-acq): every line written in the window is risky as well
rx = re.compile(r'TAPEDUMP r(\d+) (\d+) q(\d+) (event|hop(-?\d+) op(-?\d+) decl(-?\d+) (\S+) (\S+) (U|O \S+)(?: with hop\d+)?(.*))')
def rng(s):
    out = []
    for tok in s.split():
        a, b = tok.split('+')
        a, b = int(a, 16), int(b, 16)
        if b <= 0: continue
        out.append((a // LINE, (a + b - 1) // LINE + 1))
    return out
def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out
def inter(x, y):
    i = j = 0; out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b: out.append((a, b))
        if x[i][1] < y[j][1]: i += 1
        else: j += 1
    return out
bx = re.compile(r'TAPEDUMP r(\d+) begin draft (-?\d+) tok (\d+)')
US = float(sys.argv[sys.argv.index('--us') + 1]) if '--us' in sys.argv else 0.9   # gapbench 75: an agent acquire
tapes = collections.defaultdict(list)   # (rank, nth) -> records
label, pend = {}, {}
cur = {}
for line in open(args[0], errors='replace'):
    b = bx.search(line)
    if b:
        pend[int(b.group(1))] = "d%s/t%s" % (b.group(2), b.group(3)); continue
    m = rx.search(line)
    if not m: continue
    rank, idx = int(m.group(1)), int(m.group(2))
    if idx == 0 or rank not in cur:
        cur[rank] = cur.get(rank, -1) + 1
        label[(rank, cur[rank])] = pend.pop(rank, "?")
    tapes[(rank, cur[rank])].append(m)
tot = collections.Counter()
print(f"{'tape':>10} {'pass':>12} {'disp':>6} {'ordered':>8} {'skip-ord':>9} {'skip-unord':>10} {'opaque':>7} {'us@%.1f' % US:>8}  first ops")
for key in sorted(tapes):
    recs = tapes[key]
    hop_rng = {}   # own op -> (rd, wr) from its first record
    for m in recs:
        if m.group(4) == 'event': continue
        hop, rest = int(m.group(5)), m.group(11) or ''
        if ' rd' in rest and hop not in hop_rng:
            rd, wr = rest.split(' wr')[0].split(' rd', 1)[1], rest.split(' wr', 1)[1] if ' wr' in rest else ''
            hop_rng[hop] = (merge(rng(rd)), merge(rng(wr)))
    R, W, RW, unknown = [], [], [], True   # a pass starts after host work: keep the first acquire
    n = no = so = su = nop = 0
    names = []
    for m in recs:
        if m.group(4) == 'event': continue
        n += 1
        hop, why = int(m.group(5)), m.group(10)
        ordered = why.startswith('O')
        no += ordered
        if len(names) < 3: names.append(m.group(8))
        r = hop_rng.get(hop)
        if r is None:   # opaque, a no-op record or an op with no ranges
            nop += 1
            R, W, RW, unknown = [], [], [], True
            continue
        rd, wr = r
        if unknown or inter(rd, RW):
            R, W = rd, wr
            RW = list(wr) if WMODE else merge(inter(rd, wr))
            unknown = False
            continue
        if ordered: so += 1
        else: su += 1
        RW = merge(RW + inter(rd, W) + inter(wr, R) + (list(wr) if WMODE else inter(rd, wr)))
        R, W = merge(R + rd), merge(W + wr)
    tot['n'] += n; tot['o'] += no; tot['so'] += so; tot['su'] += su; tot['op'] += nop
    print(f"r{key[0]}#{key[1]:<7} {label.get(key, '?'):>12} {n:6d} {no:8d} {so:9d} {su:10d} {nop:7d} {so * US:8.1f}  {' '.join(names)[:60]}")
print(f"ALL: {tot['n']} dispatches, {tot['o']} ordered; the rule skips the acquire of {tot['so']} ordered ({100*tot['so']/max(1,tot['o']):.1f} %) "
      f"and {tot['su']} unordered; {tot['op']} opaque / no-range records keep it")
