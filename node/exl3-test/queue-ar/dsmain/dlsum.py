#!/usr/bin/env python3
"""dlsum.py (10.11.2026, queue-ar dsmain, ds-main-port VRAM agent) -- the summary of item 57-p4zyzzzzz-ds-hczzzz-dlend (body
dsm-dlend.sh): lever DL (the drafter's lend) in process, arms A (draft_lend 0, back-off 0 = today's D7), B (lend 1, back-off
3), C (lend 0, back-off 3 = the back-off's own share). Reads dlblocks.py's rows; prints

  ARM X ...    per metric mean +- t95 over the arm's ready, good blocks;
  CMP X vs A   pooled (mean X / mean A) and pairs (inside each group of len(arms) consecutive blocks holding A and X once:
               X / A; ABC|CBA balances linear drift), mean +- t95, per metric;
  REF ...      the arms vs NST 0 + D2 (gz gg2-gemv2), vramsweep T0 and hcy-depth0b B (the DL agent's base);
  SUMMARY ...  one line.
Metrics: tok/s (c1 essay / code / edit, c4 / c8 aggregate + p50), ms a decode step, zero cycles a decode step (depth-0 share),
resident VRAM units, rank-0 hit, missed a step, stream MB a step, lend (lent share, lent units max, lends, reclaims, transit ms).
  dlsum.py BLOCKS.jsonl IDENTITY.txt
Standard library only."""
import json, math, statistics, sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
REF = {"NST 0 + D2 (gz gg2-gemv2)": {"1u essay": 62.5, "1u code": 64.7, "c4 aggregate": 131.7, "c8 aggregate": 170.0},
       "vramsweep T0 (NST 0, turnover 0)": {"1u essay": 65.14, "1u code": 67.16, "c4 aggregate": 134.0, "c8 aggregate": 169.5},
       "hcy-depth0b B (D7, the DL agent's base)": {"1u essay": 57.7, "1u code": 90.3, "c4 aggregate": 118.7, "c8 aggregate": 146.7}}
TARGET = {"1u essay": 90.0, "1u code": 90.0, "1u edit": 90.0, "c4 aggregate": 160.0, "c8 aggregate": 200.0}


def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def path(*ks):
    def f(res):
        x = res
        for k in ks:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return num(x)
    return f


def ratio(nk, dk_phase):   # draft counter (ggblocksdz2 draftph) / the phase's decode steps (vsblocks xstats)
    def f(res):
        n = path("draftph", dk_phase, nk)(res); d = path(dk_phase, "xstats", "decode_steps")(res)
        return n / d if n is not None and d else None
    return f


M = []
for nm, key in (("1u essay", ("c1", "essay")), ("1u code", ("c1", "code")), ("1u edit", ("edit",))):
    M.append((nm, path(*key, "decode_tps")))
for p in ("c4", "c8"):
    M += [("%s aggregate" % p, path(p, "agg_dec_tps")), ("%s p50" % p, path(p, "per_stream_dec_p50"))]
MS = [("1u essay ms", path("c1", "essay", "ms_per_decode_step")), ("1u code ms", path("c1", "code", "ms_per_decode_step")),
      ("1u edit ms", path("edit", "ms_per_decode_step")), ("c4 ms", path("c4", "ms_decode_step")), ("c8 ms", path("c8", "ms_decode_step"))]
X = []
for p, ph in (("c1", "c1"), ("c4", "c4"), ("c8", "c8")):
    X += [("%s depth-0 share" % p, ratio("zero_cycles", ph)), ("%s units" % p, path(ph, "xstats", "vram_units_end")),
          ("%s hit %%" % p, path(ph, "xstats", "hit_pct_rank0")), ("%s missed/step" % p, path(ph, "xstats", "missed_step_rank0")),
          ("%s stream MB/step" % p, path(ph, "xstats", "stream_MB_step"))]
for p in ("c4", "c8"):
    X += [("%s lent share" % p, path(p, "xlend", "s_lent_share")), ("%s lent units max" % p, path(p, "xlend", "s_lent_units_max")),
          ("%s lends" % p, path(p, "xlend", "lends")), ("%s reclaims" % p, path(p, "xlend", "reclaims")),
          ("%s transit ms" % p, path(p, "xlend", "transit_ms")), ("%s lent plain steps" % p, path(p, "xlend", "lent_plain_steps"))]


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None)


def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)


def good(r): return r.get("ok") and ((r.get("res") or {}).get("c1") or {}).get("xready", {}).get("ok", True)


def main():
    try: rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    except (OSError, ValueError): rows = []
    try: idl = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    except OSError: idl = []
    arms = []
    for r in rows:
        if r.get("arm") not in arms: arms.append(r.get("arm"))
    val = lambda a, g: [v for v in (g(r["res"]) for r in rows if good(r) and r.get("arm") == a) if v is not None]
    mean = {}
    for a in arms:
        parts = []
        for nm, g in M + MS + X:
            m, h = ci(val(a, g)); mean[(a, nm)] = m
            parts.append("%s %s%s" % (nm, f1(m, 2 if "share" in nm or "ms" in nm or "missed" in nm else 1), "" if h is None else " +- " + f1(h, 2 if "share" in nm or "ms" in nm else 1)))
        nb = sum(1 for r in rows if r.get("arm") == a); ng = sum(1 for r in rows if r.get("arm") == a and good(r))
        print("ARM %s %s (%d blocks, %d good): %s" % (a, json.dumps(next((r.get("values") for r in rows if r.get("arm") == a), {})), nb, ng, "; ".join(parts)))
    k = len(arms)
    groups = [rows[i:i + k] for i in range(0, len(rows) - k + 1, k)] if k else []
    cmp_line = []
    for x in [a for a in arms if a != "A"]:
        out = []
        for nm, g in M + MS:
            pr = []
            for grp in groups:
                ra = [r for r in grp if r.get("arm") == "A"]; rx = [r for r in grp if r.get("arm") == x]
                if len(ra) == 1 and len(rx) == 1 and good(ra[0]) and good(rx[0]):
                    va, vx = g(ra[0]["res"]), g(rx[0]["res"])
                    if va and vx is not None: pr.append(vx / va)
            m, h = ci(pr); ma, mx = mean.get(("A", nm)), mean.get((x, nm))
            out.append("%s pooled %s, pairs %s" % (nm, "%+.1f %%" % ((mx / ma - 1) * 100) if ma and mx is not None else "-",
                       "-" if m is None else "%+.1f %%%s (n %d)" % ((m - 1) * 100, "" if h is None else " +- %.1f" % (h * 100), len(pr))))
        for nm, g in X:
            ma, mx = mean.get(("A", nm)), mean.get((x, nm))
            out.append("%s %s -> %s" % (nm, f1(ma, 2), f1(mx, 2)))
        print("CMP %s vs A: %s" % (x, "; ".join(out)))
        cmp_line.append("%s vs A: %s" % (x, ", ".join(o for o in out if o.split(" pooled")[0] in ("1u essay", "1u code", "1u edit", "c4 aggregate", "c8 aggregate", "c4 ms", "c8 ms")
                                                     or o.startswith(("c4 units", "c8 units", "c8 lent share", "c8 lent units max", "c4 depth-0", "c8 depth-0")))))
    rl = []
    for lab, ref in REF.items():
        rl.append("%s: %s" % (lab, ", ".join("%s %s" % (nm, " / ".join("%s %s (%s)" % (a, f1(mean.get((a, nm))), "%+.1f %%" % ((mean[(a, nm)] / v - 1) * 100) if mean.get((a, nm)) else "-")
                                                                       for a in arms)) + " vs %g" % v for nm, v in ref.items())))
    print("REF " + " | ".join(rl))
    tl = ", ".join("%s %s" % (nm, " / ".join("%s %s%s" % (a, f1(mean.get((a, nm))), "" if mean.get((a, nm)) is None else
                   (" MET" if mean[(a, nm)] >= t else " (+%.1f %% needed)" % ((t / mean[(a, nm)] - 1) * 100))) for a in arms)) for nm, t in TARGET.items())
    s = " || ".join(cmp_line) or "no comparison"
    s += " || means: " + "; ".join("%s %s" % (a, ", ".join("%s %s" % (nm, f1(mean.get((a, nm)))) for nm, _ in M if nm.endswith(("essay", "code", "edit", "aggregate")))) for a in arms)
    s += " || targets: " + tl + " || refs: " + " | ".join(rl)
    s += " || blocks %d (%d good, %d not ready)" % (len(rows), sum(1 for r in rows if good(r)), sum(1 for r in rows if not ((r.get("res") or {}).get("c1") or {}).get("xready", {}).get("ok", True)))
    s += " || identity: " + (", ".join(idl) if idl else "none recorded")
    print("SUMMARY " + s)


if __name__ == "__main__":
    main()
