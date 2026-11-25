#!/usr/bin/env python3
"""kbsum2.py DIR (12.11.2026, ds-main-port kbench builder; standard library only) -- the summary of 57-p4zyzzzzz-ds-r00zzz-cz-qkbench
from DIR/ix-136k.txt, ix-64k.txt, ix-16k.txt, ix-stress.txt (kb2_ix: lever IX, index_score) and DIR/m1.txt (kb2_m1: lever M1,
one-pass mHC). Prints lines the body logs
as "RESULT xkb2 <line>":
  CHECKS     every expected byte check ran (compared bytes > 0, the variant's own run flag seen) and every exact form is
             byte-identical to LIB (the image's own libr4d.so); else VIRHE lines naming what is missing / different
  IXT        a (layer kind, context) row: LIB us a 256-row run (TFLOP/s on live entries, GB/s compulsory), STK, the best exact
             form, NI, the fp8 forms
  IXLIM      the limiter at 136k: the ablations of the stock form (no WMMA / no epilogue rounding / no unpack / no store), the
             dead-region share (stock vs NI), VGPRs and workgroups a CU
  IXF8       the fp8 forms: speed, and whether they hold the stock bytes on real-like and on stress data
  IXEST      ESTIMATE rank 0's index_score ms over a 129k prefill (vs the profile) and a 136k prefill at TP4 (4 of a step's
             16 runs), stock vs the best exact form a layer kind, PSK off
  M1 / M1EST the mHC pair and the fused prototype at 4096 / 8 / 1 rows; ESTIMATE of the 136k saving (77 foldable pairs a step)
Kinds: r2 = layers 2 / 8 / 14 (ratio 2, width 262145), r1 = layer 20 (ratio 1, width 524289, the candidate layer: its
index_candidates reads the whole width, so NI does not apply there), r1k = layers 24 / 28 / 32 / 36 (ratio 1 + keep)."""
import os, re, sys

D = sys.argv[1]
KV = re.compile(r"(\w+)=([^\s]+)")
CTXS = [139264, 65536, 16384]
KIND_LAYERS = {"r2": 3, "r1": 1, "r1k": 4}
PROFILE_129K = {"r2": "493 / 485 / 532 (layers 2 / 8 / 14)", "r1": "911 (layer 20)", "r1k": "412-435 (layers 24-36)"}
IX_EXACT = ["LIB", "STK", "NI", "V128", "EB256", "EB512", "PF", "PF512", "R2", "LDS", "LDSR2", "LIB_BF16KEY"]
IX_EXACT_K = ["LIB", "STK", "NI", "V1024", "KPF", "KR2", "K256", "K256PF"]
IX_F8 = ["F8R", "F8B"]
IX_ABL = ["A_NOWMMA", "A_NOEPI", "A_NOUNPACK", "A_NOSTORE", "A_NOWMMA_NOEPI"]
M1_FORMS = ["LIB_PCN", "LIB_MIX", "STK_PCN", "STK_MIX", "LIB_PAIR", "F1R8", "F1R4"]
M1_ROWS = [4096, 8, 1]
out, virhe = [], []


def kv(line):
    return dict(KV.findall(line))


def read(name):
    p = os.path.join(D, name)
    return open(p, errors="replace").read().splitlines() if os.path.exists(p) else None


IX_PARTS = ["ix-136k.txt", "ix-64k.txt", "ix-16k.txt", "ix-stress.txt"]   # the body's four kb2_ix runs, in order
ix = []
for name in IX_PARTS:
    part = read(name)
    if part is None:
        virhe.append("VIRHE %s missing" % name)
        continue
    if not any(l.startswith("KB2 IXDONE") for l in part):
        virhe.append("VIRHE kb2_ix %s did not finish (no IXDONE; last: %s)" % (name, part[-1][:200] if part else "-"))
    ix += part
m1 = read("m1.txt")
if m1 is None:
    virhe.append("VIRHE m1.txt missing")
    m1 = []
for l in ix + m1:
    if "VIRHE" in l or l.startswith("KB2 FATAL"):
        virhe.append("VIRHE bench: " + l[:300])
if m1 and not any(l.startswith("KB2 M1DONE") for l in m1):
    virhe.append("VIRHE kb2_m1 did not finish (no M1DONE; last: %s)" % (m1[-1][:200] if m1 else "-"))

# ---------------------------------------------------------------- IX checks
chk = {}   # (form, ctx, kind, data) -> dict
for l in ix:
    if l.startswith("KB2 IXCHK "):
        f = l.split()[2]
        d = kv(l)
        chk[(f, int(d.get("ctx", 0)), d.get("kind"), d.get("data"))] = (d, l)
expect = []
for c in CTXS:
    for k in ("r2", "r1"):
        expect += [(f, c, k, "real") for f in IX_EXACT + IX_F8 + IX_ABL]
    if c >= 32768:
        expect += [(f, c, "r1k", "real") for f in IX_EXACT_K]
for k in ("r1", "r2"):
    expect += [(f, 65536, k, "stress") for f in IX_EXACT + IX_F8 if f != "LIB_BF16KEY"]
nrun = nsame = nabl = 0
for e in expect:
    f = e[0]
    if e not in chk:
        virhe.append("VIRHE byte check did not run: %s ctx=%d kind=%s data=%s" % e)
        continue
    d, l = chk[e]
    if f in IX_ABL:
        nabl += 1
        if d.get("flag") != "ran":
            virhe.append("VIRHE ablation %s ctx=%d kind=%s did not run its own kernel (flag %s)" % (f, e[1], e[2], d.get("flag")))
        continue
    if int(d.get("compared_bytes", "0")) <= 0:
        virhe.append("VIRHE byte check compared nothing: " + l[:200])
        continue
    if d.get("flag") not in ("ran", "n/a") or (f not in ("LIB", "STK", "NI", "LIB_BF16KEY") and d.get("flag") != "ran"):
        virhe.append("VIRHE %s ctx=%d kind=%s data=%s: its kernel did not run (flag %s; a fallback?)" % (f, e[1], e[2], e[3], d.get("flag")))
        continue
    nrun += 1
    if f in IX_F8:
        continue
    if d.get("verdict") == "SAME":
        nsame += 1
    else:
        virhe.append("VIRHE exact form %s ctx=%d kind=%s data=%s differs from LIB: %s words (first row,col %s)" %
                     (f, e[1], e[2], e[3], d.get("diff_words"), d.get("first")))

# ---------------------------------------------------------------- IX timing
tm = {}   # (form, ctx, kind) -> med_us
tline = {}
for l in ix:
    if l.startswith("KB2 IXT ") and "med_us=" in l:
        f = l.split()[2]
        d = kv(l)
        key = (f, int(d["ctx"]), d["kind"])
        tm[key] = float(d["med_us"])
        tline[key] = d
cases = {}
for l in ix:
    if l.startswith("KB2 IXCASE "):
        d = kv(l)
        if d.get("data") == "real":
            cases[(int(d["ctx"]), d["kind"])] = d
attrs = {}
for l in ix + m1:
    if l.startswith("KB2 ATTR "):
        d = kv(l)
        attrs[l.split()[2]] = d


def same(f, c, k):
    v = chk.get((f, c, k, "real"))
    return v is not None and v[0].get("verdict") == "SAME"


def exact_forms(kind):
    base = IX_EXACT_K if kind == "r1k" else IX_EXACT
    fs = [f for f in base if f not in ("LIB", "LIB_BF16KEY")]
    if kind == "r1":
        fs = [f for f in fs if f != "NI"]   # layer 20's index_candidates reads the whole width
    return fs


def pct(a, b):
    return "%+.1f %%" % ((a - b) / b * 100.0) if a and b else "n/a"


def fmt(f, c, k):
    t = tm.get((f, c, k))
    if t is None:
        return "%s n/a" % f
    d = tline[(f, c, k)]
    return "%s %.0f us (%s TFLOP/s, %s GB/s)" % (f, t, d.get("live_tflops"), d.get("compulsory_gbs"))


best_at = {}
for k in ("r2", "r1", "r1k"):
    for c in CTXS:
        if (c, k) not in cases:
            continue
        lib = tm.get(("LIB", c, k))
        cand = [(tm[(f, c, k)], f) for f in exact_forms(k) if (f, c, k) in tm and f not in ("STK", "NI") and same(f, c, k)]
        cand.sort()
        bt = cand[0] if cand else None
        cs = cases[(c, k)]
        parts = ["%s ctx=%d (run 7 of a 4096-row step, M 256, width %s, reach %s, live %.3g entries)" %
                 (k, c, cs.get("C"), cs.get("reach"), float(cs.get("live", 0)))]
        parts.append(fmt("LIB", c, k))
        parts.append("STK %s" % ("%.0f us (%s vs LIB)" % (tm[("STK", c, k)], pct(tm[("STK", c, k)], lib)) if ("STK", c, k) in tm else "n/a"))
        if bt:
            parts.append("best exact %s %.0f us (%s)" % (bt[1], bt[0], pct(bt[0], lib)))
        if k != "r1" and ("NI", c, k) in tm:
            parts.append("NI %.0f us (%s)" % (tm[("NI", c, k)], pct(tm[("NI", c, k)], lib)))
        elif ("NI", c, k) in tm:
            parts.append("NI %.0f us (%s; not for layer 20)" % (tm[("NI", c, k)], pct(tm[("NI", c, k)], lib)))
        if k != "r1k":
            q = tm.get(("F8QPREP", c, k))
            for f in IX_F8:
                if (f, c, k) in tm:
                    t = tm[(f, c, k)] + (q / 16.0 if q else 0.0)
                    parts.append("%s %.0f us incl. q prep/16 (%s, %s)" % (f, t, pct(t, lib), "bytes SAME" if same(f, c, k) else "bytes DIFFER"))
            if ("LIB_BF16KEY", c, k) in tm:
                parts.append("bf16 keys %.0f us (%s)" % (tm[("LIB_BF16KEY", c, k)], pct(tm[("LIB_BF16KEY", c, k)], lib)))
        rest = ["%s %.0f" % (f, tm[(f, c, k)]) for f in exact_forms(k) if (f, c, k) in tm and f not in ("STK", "NI")]
        parts.append("all exact (us): " + ", ".join(rest))
        out.append("IXT " + " | ".join(parts))

# ---------------------------------------------------------------- IX limiter (136k, r1 and r2)
for k in ("r1", "r2"):
    c = 139264
    base = tm.get(("V128", c, k))
    stk = tm.get(("STK", c, k))
    if not base or not stk:
        continue
    ab = []
    for f, lbl in (("A_NOWMMA", "no WMMA"), ("A_NOEPI", "no epilogue rounding"), ("A_NOUNPACK", "no mx4 unpack"),
                   ("A_NOSTORE", "no score store"), ("A_NOWMMA_NOEPI", "no WMMA + no epilogue")):
        if (f, c, k) in tm:
            ab.append((tm[(f, c, k)], lbl))
    dead = tm.get(("NI", c, k))
    s = "IXLIM %s ctx=%d: stock form STK %.0f us, V128 (same geometry, mx4 at compile time) %.0f us; ablations of V128: %s" % (
        k, c, stk, base, ", ".join("%s %.0f us (%s)" % (lbl, t, pct(t, base)) for t, lbl in ab))
    if dead:
        s += "; dead region (the -inf writes past the run's reach) %.0f us = %.0f %% of STK" % (stk - dead, (stk - dead) / stk * 100)
    cs = cases.get((c, k))
    if cs:
        live = float(cs["live"])
        s += "; live %.3g entries -> STK %.1f TFLOP/s" % (live, live * 8192 / (stk * 1e-6) / 1e12)
    a = attrs.get("ix_stock_wm<2,false>")
    if a:
        s += "; stock kernel %s VGPRs, %s workgroups (%s waves) a CU" % (a.get("regs"), a.get("wg_per_cu"), a.get("waves_per_cu"))
    if ab:
        drops = {lbl: (base - t) / base for t, lbl in ab}
        both = drops.get("no WMMA + no epilogue", 0)
        if both < 0.3:
            lim = "neither the WMMA nor the epilogue: removing both saves %.0f %% -> loads / latency / stores" % (both * 100)
        else:
            top = max(drops.items(), key=lambda x: x[1])
            lim = "the largest single cost is '%s' (%.0f %% of V128); WMMA + epilogue together %.0f %%" % (top[0], top[1] * 100, both * 100)
        s += "; LIMITER (reading of the ablations): " + lim
    out.append(s)
if attrs:
    out.append("IXATTR " + " | ".join("%s %s VGPR %s wg/CU" % (n, a.get("regs"), a.get("wg_per_cu")) for n, a in sorted(attrs.items())))

# ---------------------------------------------------------------- IX fp8
for f in IX_F8:
    parts = []
    for (ff, c, k, dd), (d, l) in sorted(chk.items()):
        if ff == f:
            parts.append("%s %s ctx=%d: %s diff words %s" % (k, dd, c, d.get("verdict"), d.get("diff_words")))
    unrep = [kv(l).get("f8_unrepresentable") for l in ix if l.startswith("KB2 IXCASE ") and "data=stress" in l]
    mine = [e for e in chk if e[0] == f]
    exact_all = bool(mine) and all(chk[e][0].get("verdict") == "SAME" for e in mine)
    out.append("IXF8 %s (%s): %s -> %s; stress cases f8 unrepresentable (rebased/block) %s" % (
        f, "rebased: one exponent a 128-vector, the stock K slicing" if f == "F8R" else "the audit's block dot: 32-blocks, scaled partials",
        "; ".join(parts), ("the stock bytes in every case run" if exact_all else "NOT byte-exact vs today's scores (a non-exact option, user decision only)") if mine else "NO CHECKS",
        ",".join(u for u in unrep if u)))


# ---------------------------------------------------------------- IX estimate
def curve(f, k):
    """per-run us as a function of the run's middle position x (run 7 of a step ending at ctx: x = ctx - 2176)"""
    pts = sorted((c - 2176, tm[(f, c, k)]) for c in CTXS if (f, c, k) in tm)
    return pts


def at(pts, x):
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if x <= x1 or i == len(pts) - 2:
            return max(0.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return pts[-1][1]


def prefill_ms(f, k, P):
    steps = P / 4096.0
    tot = 0.0
    s = 0
    while s < steps:
        frac = min(1.0, steps - s)
        x = 4096 * s + 512                       # rank 0's runs: rows 0..1023 of the step
        kk = k
        if k == "r1k" and x < 32768:            # keep_min_live: the plain op below 32768 live positions
            kk = "r1"
            ff = f if f in exact_forms("r1") or f == "LIB" else "LIB"
        else:
            ff = f
        pts = curve(ff, kk)
        if not pts:
            return None
        tot += 4 * at(pts, x) * frac
        s += 1
    return tot / 1000.0


for P, tag in ((132096, "129k (the profile's prefill; check)"), (139264, "136k")):
    parts, tot_lib, tot_best = [], 0.0, 0.0
    ok = True
    for k in ("r2", "r1", "r1k"):
        lib = prefill_ms("LIB", k, P)
        if lib is None:
            ok = False
            continue
        cands = []
        for f in exact_forms(k):
            if f in ("STK", "NI"):
                continue
            if not all(same(f, c, k) for c in CTXS if (f, c, k) in tm):
                continue
            v = prefill_ms(f, k, P)
            if v is not None:
                cands.append((v, f))
        cands.sort()
        best = cands[0] if cands else (lib, "LIB")
        ni = prefill_ms("NI", k, P) if k != "r1" else None
        n = KIND_LAYERS[k]
        tot_lib += n * lib
        tot_best += n * best[0]
        s = "%s x%d: LIB %.0f ms a layer" % (k, n, lib)
        if P == 132096:
            s += " (profile %s)" % PROFILE_129K[k]
        s += ", best exact %s %.0f (%s)" % (best[1], best[0], pct(best[0], lib))
        if ni is not None:
            s += ", NI %.0f (%s)" % (ni, pct(ni, lib))
        parts.append(s)
    if ok or parts:
        out.append("IXEST ESTIMATE rank 0, %s, PSK off, TP4: %s | all 8 index layers LIB %.2f s -> best exact a kind %.2f s (%.2f s, %s)" % (
            tag, " ; ".join(parts), tot_lib / 1000, tot_best / 1000, (tot_best - tot_lib) / 1000, pct(tot_best, tot_lib)))

# ---------------------------------------------------------------- M1
mchk, mt = {}, {}
for l in m1:
    if l.startswith("KB2 M1CHK "):
        d = kv(l)
        mchk[(l.split()[2], int(d["rows"]))] = (d, l)
    if l.startswith("KB2 M1T "):
        d = kv(l)
        mt[(l.split()[2], int(d["rows"]))] = float(d["med_us"])
for r in M1_ROWS:
    for f in M1_FORMS:
        if (f, r) not in mchk:
            virhe.append("VIRHE M1 byte check did not run: %s rows=%d" % (f, r))
            continue
        d, l = mchk[(f, r)]
        if int(d.get("compared_bytes", "0")) <= 0:
            virhe.append("VIRHE M1 byte check compared nothing: " + l[:200])
        elif f.startswith("F1") and d.get("flags") != "ran":
            virhe.append("VIRHE M1 %s rows=%d: its kernels did not run" % (f, r))
        elif d.get("verdict") != "SAME":
            virhe.append("VIRHE M1 %s rows=%d differs from LIB: %s" % (f, r, l[l.find("(diff"):][:300]))
        else:
            nrun += 1
            nsame += 1
mw = [l for l in m1 if l.startswith("KB2 M1W ")]
for r in M1_ROWS:
    pcn, mix, pair = mt.get(("LIB_PCN", r)), mt.get(("LIB_MIX", r)), mt.get(("LIB_PAIR", r))
    if not pcn or not mix:
        continue
    two = pcn + mix
    s = "M1 rows=%d: LIB pcn %.1f us + mix %.1f us = %.1f us (back to back %s us); STK pcn %s / mix %s" % (
        r, pcn, mix, two, "%.1f" % pair if pair else "n/a", mt.get(("STK_PCN", r)), mt.get(("STK_MIX", r)))
    for f in ("F1R8", "F1R4"):
        t = mt.get((f, r))
        if t:
            ok = mchk.get((f, r), ({}, ""))[0].get("verdict") == "SAME"
            s += " | %s %.1f us (%s vs the pair; K1 %s us, finish %s us, norm %s us; bytes %s)" % (
                f, t, pct(t, pair or two), mt.get((f + "_K1", r)), mt.get(("F1_FIN", r)), mt.get(("F1_K2", r)),
                "SAME" if ok else "DIFFER")
    if mw and r == 4096:
        s += " | " + mw[0][len("KB2 "):]
    out.append(s)
pair = mt.get(("LIB_PAIR", 4096)) or ((mt.get(("LIB_PCN", 4096)) or 0) + (mt.get(("LIB_MIX", 4096)) or 0))
fb = min([(mt[(f, 4096)], f) for f in ("F1R8", "F1R4") if (f, 4096) in mt and mchk.get((f, 4096), ({}, ""))[0].get("verdict") == "SAME"] or [(None, None)])
if pair and fb[0]:
    sv = 34 * 77 * (pair - fb[0]) / 1e6
    out.append("M1EST ESTIMATE 136k prefill (34 steps of 4096 rows, 77 foldable pairs a step, every rank): pair %.1f us -> %s %.1f us = %.2f s %s a prefill (the audit's range 0.47-1.46 s)%s" % (
        pair, fb[1], fb[0], abs(sv), "saved" if sv > 0 else "LOST (the exact fused form is slower)",
        "" if sv > 0 else "; exact order makes the fusion slower"))

# ---------------------------------------------------------------- verdict
out.insert(0, "CHECKS %s: %d byte checks ran with their own kernels (%d exact forms byte-identical to LIB, %d ablations ran), %d VIRHE" % (
    "ok" if not virhe else "VIRHE", nrun, nsame, nabl, len(virhe)))
for l in out:
    print(l)
for v in virhe:
    print(v)
