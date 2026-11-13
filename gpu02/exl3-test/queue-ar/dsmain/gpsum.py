#!/usr/bin/env python3
"""gpsum.py OUTDIR (10.11.2026, queue-ar/dsmain, lever GPF agent) -- the summary of the xgp bench item (runs/xdsm/gpbench):
gpcheck.txt (r4d_selftest --gpcheck) and gpperf.txt (--perfgp).

Lines:
  GPCHK    the byte check's verdict and its per-(type, shape) DIFFERS lines (if any)
  GPROW    a (type, shape) at each M: stock us / TFLOP/s, best identical form, speedup, and every form's speedup (x stock)
  GPABLT   the stock geometry's ablations as % of the stock time (what the cut part costs; outputs differ by design)
  GPEST    ESTIMATE of rank 0's 128k-step dense GEMM time with the best form a shape (M 4096 speedups) against the profile
           of dsm-225-prof-pf128 (rank 0, gemm_nt_ggml 254.3 ms a step averaged over its 40 steps, 33 of them prefill), and
           the 136k cold TTFT saving it implies (x 40 steps x 136/129.2k tokens); one-card rot timing, so an estimate
  GPFORM   the best single form for every shape (one code) and its estimate, for a uniform gg_pf value; and
           "GPFORM table-4096+2048: <table>": the A/B item's RADIANCE_DS41_GG_PF_TABLE (per shape the identical form with the
           least us at M 4096 + 2048 that beats the stock's sum by >= 1 % and loses at neither M by > 0.5 %)
Standard library only."""
import math
import re
import sys
from collections import defaultdict

# rank 0's gemm_nt_ggml ms a step (40-step window of dsm-225-prof-pf128, decode steps included), by (type, shape)
PROF = {
    ("q6_K", "attn_q_b"): 32.02, ("q5_K", "attn_q_b"): 14.81,
    ("q6_K", "engram_kv"): 31.31,
    ("q6_K", "attn_output_b"): 30.11, ("q5_K", "attn_output_b"): 14.11,
    ("q6_K", "shexp_gate_up"): 23.66, ("q8_0", "shexp_gate_up"): 10.50,
    ("q6_K", "attn_q_a"): 19.27, ("q5_K", "attn_q_a"): 8.98,
    ("q6_K", "attn_output_a"): 25.10, ("q5_K", "attn_output_a"): 11.65,
    ("q6_K", "shexp_down"): 14.06, ("q8_0", "shexp_down"): 6.23,
    ("q6_K", "attn_kv"): 8.47, ("q5_K", "attn_kv"): 3.99,
}
STEPS, TOK_PROF, TOK_136 = 40, 129220, 136000

PERF = re.compile(r"^GPPERF (\S+) (\S+) (\d+)x(\d+) M=(\d+) \([\d.]+ MiB\):(.*)$")
ENT = re.compile(r"\s*(\S+) (-?[\d.]+) \(([\d.]+) TF\)( \(ablation: differs\)| OUTPUT DIFFERS)?")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    out = []
    try:
        chk = open(d + "/gpcheck.txt", errors="replace").read().splitlines()
    except OSError:
        chk = []
    v = [l for l in chk if l.startswith("GPVERDICT")]
    bad = [l for l in chk if l.startswith("GPCHECK") and "DIFFERS" in l]
    out.append("GPCHK " + (v[-1] if v else "no GPVERDICT line (check did not finish)") +
               (" | " + " | ".join(b[:200] for b in bad) if bad else ""))
    try:
        perf = open(d + "/gpperf.txt", errors="replace").read().splitlines()
    except OSError:
        perf = []
    rows = {}   # (ty, shape, M) -> {form: (us, tf, flag)}
    for l in perf:
        m = PERF.match(l)
        if not m:
            continue
        ty, sh, M = m.group(1), m.group(2), int(m.group(5))
        r = {}
        for part in m.group(6).split(";"):
            e = ENT.match(part)
            if e:
                r[e.group(1)] = (float(e.group(2)), float(e.group(3)), (e.group(4) or "").strip())
        rows[(ty, sh, M)] = r
    if not rows:
        out.append("GPROW none (no GPPERF lines)")
        out.extend(f32sum(chk, perf))   # a GPF-b-only bench (gg types 'none') still gets its lines
        print("\n".join(out))
        return 1
    forms = sorted({f for r in rows.values() for f in r if f != "stock" and "abl" not in f})
    best_at = {}
    for (ty, sh, M), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
        st = r.get("stock", (0, 0, ""))[0]
        if st <= 0:
            continue
        cand = [(us, f) for f, (us, tf, fl) in r.items() if f != "stock" and "abl" not in f and not fl and us > 0]
        bu, bf = min(cand) if cand else (st, "stock")
        if bu >= st:
            bu, bf = st, "stock"
        best_at[(ty, sh, M)] = (bf, st / bu)
        fx = " ".join("%s x%.3f%s" % (f, st / r[f][0], "!" if r[f][2] else "") for f in forms if f in r and r[f][0] > 0)
        out.append("GPROW %s %s M=%d: stock %.1f us %.1f TF | best %s %.1f us x%.3f | %s" % (
            ty, sh, M, st, r["stock"][1], bf, bu, st / bu, fx))
        ab = " ".join("%s %+.1f %%" % (f.split("-")[-1], 100.0 * (r[f][0] / st - 1.0)) for f in sorted(r) if "abl" in f and r[f][0] > 0)
        if ab:
            out.append("GPABLT %s %s M=%d: stock %.1f us | %s (abl1 no chain-init / min LDS reads, abl2 no B decode, abl4 no per-group fma)" % (
                ty, sh, M, st, ab))
    # estimate at M 4096 (the 128k step's rows)
    Mx = max(M for (_, _, M) in rows)
    tot = sum(PROF.values())
    new = 0.0
    parts = []
    for (ty, sh), ms in PROF.items():
        bf, sp = best_at.get((ty, sh, Mx), ("stock", 1.0))
        new += ms / sp
        parts.append("%s %s %.2f->%.2f (%s)" % (ty, sh, ms, ms / sp, bf))
    sav = tot - new
    out.append("GPEST ESTIMATE rank 0 128k step (40-step avg): gemm_nt_ggml %.1f -> %.1f ms (-%.1f ms, -%.1f %% of the 1206 ms step); "
               "136k cold TTFT -%.2f s (of ~50.2-51.6 s) | %s" % (tot, new, sav, 100.0 * sav / 1206.4,
                                                                   sav * STEPS * TOK_136 / TOK_PROF / 1000.0, "; ".join(parts)))
    # the best uniform form
    bestu = None
    for f in forms:
        n = 0.0
        ok = True
        for (ty, sh), ms in PROF.items():
            r = rows.get((ty, sh, Mx))
            if not r or f not in r or r[f][2] or r[f][0] <= 0:
                ok = False
                break
            n += ms * r[f][0] / r["stock"][0]
        if ok and (bestu is None or n < bestu[1]):
            bestu = (f, n)
    if bestu:
        f, n = bestu
        code = re.match(r"m(\d)n(\d)w(\d)(f?)$", f)
        cv = (100 * int(code.group(1)) + 10 * int(code.group(2)) + int(code.group(3)) // 4 + (4 if code.group(4) else 0)) if code else 0
        out.append("GPFORM best single form %s (gg_pf=%d): %.1f -> %.1f ms (-%.1f ms) ESTIMATE" % (f, cv, tot, n, tot - n))
    # the A/B item's table: per (type, shape) the identical form with the least us at M 4096 + M 2048 together (lone prefill
    # steps run 4096 rows, steps beside decode up to ~2112), taken where that sum beats the stock's by >= 1 % and the form is
    # not slower than the stock at either M by more than 0.5 % (no loss in the mixed steps)
    names = {}
    for (ty, sh, M), r in rows.items():
        names.setdefault((ty, sh), {})[M] = r
    ms = sorted({M for (_, _, M) in rows}, reverse=True)[:2]
    tab = []
    for (ty, sh), byM in sorted(names.items()):
        if not all(M in byM for M in ms):
            continue
        dims = None
        for l in perf:
            m = PERF.match(l)
            if m and m.group(1) == ty and m.group(2) == sh:
                dims = "%sx%s" % (m.group(3), m.group(4))
                break
        st = [byM[M].get("stock", (0, 0, ""))[0] for M in ms]
        if min(st) <= 0 or not dims:
            continue
        best = None
        for f in forms:
            v = [byM[M].get(f) for M in ms]
            if any(x is None or x[2] or x[0] <= 0 for x in v):
                continue
            if any(x[0] > 1.005 * s0 for x, s0 in zip(v, st)):
                continue
            tot = sum(x[0] for x in v)
            if tot <= 0.99 * sum(st) and (best is None or tot < best[1]):
                best = (f, tot)
        if best:
            mm = re.match(r"m(\d)n(\d)w(\d)(f?)$", best[0])
            if mm:
                code = 100 * int(mm.group(1)) + 10 * int(mm.group(2)) + int(mm.group(3)) // 4 + (4 if mm.group(4) else 0)
                tab.append("%s:%s=%d" % (ty, dims, code))
    out.append("GPFORM table-4096+2048: %s" % (",".join(tab) if tab else "-"))
    out.extend(f32sum(chk, perf))
    print("\n".join(out))
    return 0


F32PERF = re.compile(r"^GPF32PERF (\S+) (\d+)x(\d+) (\S+) M=(\d+): stock (-?[\d.]+) \(([\d.]+) TF\);(.*)$")
F32ENT = re.compile(r"\s*f(\d+) (-?[\d.]+) \(([\d.]+) TF\)( OUTPUT DIFFERS)?")
# rank 0's gemm_nt_f32 ms a step (40-step window of dsm-225-prof-pf128): router N 96 and the compressor's kv | gate N 1024
F32PROF = {"router": 16.25, "compressor_kvg": 14.58}


def f32sum(chk, perf):
    """lever GPF-b's lines: GPF32CHK (the byte check), GPF32ROW (shape, storage, M: stock / every form), GPF32EST (ESTIMATE of
    the router + compressor ms a 128k step with the A/B's form, f32 storage as the stack serves it), and
    "GPF32FORM ab: <code>" -- the single form the A/B runs (the least summed us over every shape, storage and M, at least
    2 % under the stock's sum, slower than the stock nowhere by more than 0.5 %), or '-'."""
    o = []
    v = [l for l in chk if l.startswith("GPF32VERDICT")]
    o.append("GPF32CHK " + (v[-1] if v else "no GPF32VERDICT line"))
    rows = {}
    for l in perf:
        m = F32PERF.match(l)
        if not m:
            continue
        r = {"stock": (float(m.group(6)), float(m.group(7)), "")}
        for part in m.group(8).split(";"):
            e = F32ENT.match(part)
            if e:
                r[int(e.group(1))] = (float(e.group(2)), float(e.group(3)), (e.group(4) or "").strip())
        rows[(m.group(1), m.group(4), int(m.group(5)))] = r
    if not rows:
        o.append("GPF32ROW none (no GPF32PERF lines)")
        o.append("GPF32FORM ab: -")
        return o
    forms = sorted({f for r in rows.values() for f in r if f != "stock"})
    for (sh, st, M), r in sorted(rows.items()):
        s0 = r["stock"][0]
        o.append("GPF32ROW %s %s M=%d: stock %.1f us %.2f TF | %s" % (sh, st, M, s0, r["stock"][1], " ".join(
            "f%d x%.3f%s" % (f, s0 / r[f][0], "!" if r[f][2] else "") for f in forms if f in r and r[f][0] > 0 and s0 > 0)))
    best = None
    for f in forms:
        ok, tot, stot = True, 0.0, 0.0
        for r in rows.values():
            s0 = r["stock"][0]
            x = r.get(f)
            if not x or x[2] or x[0] <= 0 or s0 <= 0 or x[0] > 1.005 * s0:
                ok = False
                break
            tot += x[0]; stot += s0
        if ok and tot <= 0.98 * stot and (best is None or tot < best[1]):
            best = (f, tot, stot)
    Mx = max(M for (_, _, M) in rows)
    if best:
        f = best[0]
        parts, sav = [], 0.0
        for sh, ms in F32PROF.items():
            r = rows.get((sh, "f32-w", Mx))
            if r and f in r and r[f][0] > 0:
                new = ms * r[f][0] / r["stock"][0]
                sav += ms - new
                parts.append("%s %.2f -> %.2f ms" % (sh, ms, new))
        o.append("GPF32EST ESTIMATE rank 0 128k step (40-step avg), f32 storage, form f%d: %s; -%.1f ms a step, 136k cold TTFT -%.2f s" % (
            f, "; ".join(parts), sav, sav * STEPS * TOK_136 / TOK_PROF / 1000.0))
        o.append("GPF32FORM ab: %d (summed us x%.3f vs stock)" % (f, best[2] / best[1]))
    else:
        o.append("GPF32FORM ab: - (no form 2 %% under the stock's summed time without a loss anywhere)")
    return o


if __name__ == "__main__":
    sys.exit(main())
