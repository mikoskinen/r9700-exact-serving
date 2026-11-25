#!/usr/bin/env python3
"""vssum.py (10.11.2026, queue-ar dsmain, ds-main-port VRAM agent) -- the summary of item 57-p4zyzzzzz-ds-hxz-vramstack
(body dsm-vramstack.sh). Reads the body's plan file (one "PLAN key=value ..." line per NST configuration), the block rows
(vlblocks.py; the OFF side's from hx-stack's ggblocksdz2 or an own OFF start), the 136k deep json files, the /stats start
snapshots and identity.txt; prints per configuration

  NST <n> START LEVERS ...  the ON start's base arm (start-time levers only: V at NST 0, A at NST 5) vs the OFF side, per
                            metric value (mean +- t95), OFF reference, % difference, distance to target; resident units ON vs
                            OFF per phase (measured delta vs the table's audit estimate) and the vramsweep slope's predicted
                            ms delta vs the measured one; the 136k cold TTFT;
  NST <n> EVERYTHING ...    the ON start's fullest arm (D) vs the OFF side (headline metrics);
  PAIRS <n> X vs Y ...      in-process arms (fn4): pooled (mean X / mean Y) and pairs inside each group of len(arms)
                            consecutive measured blocks (ABD|DBA), mean +- t95;
  DL <n> arm ...            per arm: c1 essay zero / probe cycles (the essay-only phase), c4 / c8 depth-0 share and zero
                            moves, lent share, lent units max, time to the first lent sample and to 95 % of the maximum,
                            borrow cycles, lends / reclaims, VRAM units max during the phase;
  SUMMARY ...               one line.
Settle blocks (vlblocks "settle") and blocks never ready are left out; off_skip=N drops the OFF side's first N blocks (the
first-c4-after-start artefact, so both sides lose their first block).
  vssum.py PLANFILE IDENTITY.txt DEEPREF.json
Standard library only."""
import json, math, statistics, sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
# vramsweep 10.11. (xvs, T150 vs T0, NST 0, ds41-gg 8b311fcc): ms a step per 100 resident routed units a rank
SLOPE = {"c1": 0.409, "c4": 1.309, "c8": 2.034}
TARGET = {"1u essay": 90.0, "1u code": 90.0, "1u edit": 90.0, "c4 aggregate": 160.0, "c8 aggregate": 200.0}


def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except (OSError, ValueError): return []


def js(p):
    try: return json.load(open(p))
    except Exception: return {}


def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def path(*ks):
    def f(res):
        x = res
        for k in ks:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return num(x)
    return f


def c1(name, k):   # vlblocks / ggblocksdz2 (c1 keyed by prompt) and the older ggblocks (c1.rows)
    def f(res):
        c = res.get("c1") or {}
        if isinstance(c.get(name), dict): return num(c[name].get(k))
        return num(next((x.get(k) for x in (c.get("rows") or []) if x.get("prompt") == name), None))
    return f


def c1ms(res):
    v = [x for x in (c1("essay", "ms_per_decode_step")(res), c1("code", "ms_per_decode_step")(res)) if x is not None]
    return sum(v) / len(v) if v else None


def share(ph):
    def f(res):
        n = path("draftph", ph, "zero_cycles")(res); d = path(ph, "xstats", "decode_steps")(res)
        return n / d if n is not None and d else None
    return f


M = [("1u essay", c1("essay", "decode_tps")), ("1u code", c1("code", "decode_tps")), ("1u edit", path("edit", "decode_tps")),
     ("c4 aggregate", path("c4", "agg_dec_tps")), ("c4 p50", path("c4", "per_stream_dec_p50")),
     ("c8 aggregate", path("c8", "agg_dec_tps")), ("c8 p50", path("c8", "per_stream_dec_p50"))]
MS = [("c1", c1ms), ("c4", path("c4", "ms_decode_step")), ("c8", path("c8", "ms_decode_step"))]
HEAD = ("1u essay", "1u code", "1u edit", "c4 aggregate", "c8 aggregate")
DL = [("c1 essay zero", path("draftph", "essaypos", "zero_cycles")), ("c1 essay probe", path("draftph", "essaypos", "probe_cycles"))]
for p in ("c4", "c8"):
    DL += [("%s depth-0 share" % p, share(p)), ("%s zero moves" % p, path("draftph", p, "zero_moves")),
           ("%s lent share" % p, path(p, "xlend", "s_lent_share")), ("%s lent max" % p, path(p, "xlend", "s_lent_units_max")),
           ("%s first lent s" % p, path(p, "xlend", "s_t_first_lent")), ("%s 95%% of max s" % p, path(p, "xlend", "s_t_to_95pct_max")),
           ("%s borrow cycles" % p, path(p, "xlend", "borrow_cycles")), ("%s lends" % p, path(p, "xlend", "lends")),
           ("%s reclaims" % p, path(p, "xlend", "reclaims")), ("%s units max" % p, path(p, "xlend", "s_vram_units_max")),
           ("%s hit %%" % p, path(p, "xstats", "hit_pct_rank0")), ("%s stream MB" % p, path(p, "xstats", "stream_MB_step"))]


def good(r):
    return r.get("ok") and not r.get("settle") and (((r.get("res") or {}).get("c1") or {}).get("xready") or {}).get("ok", True)


def vals(rs, arm, get): return [v for v in (get(r["res"]) for r in rs if good(r) and r.get("arm") == arm) if v is not None]


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None)


def mean(v): return statistics.mean(v) if v else None
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pct(v, r): return ("%+.1f %%" % ((v / r - 1) * 100)) if v is not None and r else "-"


def tgt(v, t, low=False):
    if t is None or v is None: return ""
    if low:
        return "; target <= %g s: %s" % (t, "MET (%.1f s under)" % (t - v) if v <= t else "%.1f s over, %.1f %% less needed" % (v - t, (1 - t / v) * 100))
    return "; target >= %g: %s" % (t, "MET (+%.1f)" % (v - t) if v >= t else "%.1f short, +%.1f %% needed" % (t - v, (t / v - 1) * 100))


def plan(p):
    out = []
    try:
        for l in open(p):
            if l.startswith("PLAN "): out.append(dict(kv.split("=", 1) for kv in l[5:].split() if "=" in kv))
    except OSError: pass
    return out


def start_units(p):
    for t in (js(p).get("experts") or {}).get("tiers") or []:
        if t.get("name") == "vram": return t.get("units")
    return None


def fnum(s):
    try: return float(s)
    except (TypeError, ValueError): return None


def main():
    pl, ident, dref = plan(sys.argv[1]), sys.argv[2], js(sys.argv[3])
    try: idl = [l.strip() for l in open(ident) if l.strip()]
    except OSError: idl = []
    rt = num(dref.get("ttft"))
    cold = lambda d: num(d.get("ttft")) if d.get("cached") == 0 else None
    summ = []
    for P in pl:
        nst, src = P.get("nst"), P.get("off_src", "none")
        arms = [x for x in P.get("arms", "V").split(",") if x]
        base, full, offarm = P.get("base", arms[0]), P.get("full", arms[-1]), P.get("off_arm", "O")
        on = rows(P.get("on_blocks", ""))
        off = rows(P.get("off_blocks", "")) if src in ("hx", "own") else []
        skip = int(fnum(P.get("off_skip")) or 0)
        if skip: off = [r for i, r in enumerate([r for r in off if r.get("arm") == offarm]) if i >= skip]
        exp_st, exp_fn = fnum(P.get("expected_st")), fnum(P.get("expected_fn"))
        # ---- start-time levers: the base arm vs OFF
        parts, sp = [], []
        for nm, get in M:
            v = vals(on, base, get); m, h = ci(v); r = mean(vals(off, offarm, get))
            parts.append("%s %s%s (n %d) vs OFF %s (n %d, %s)%s" % (nm, f1(m), "" if h is None else " +- " + f1(h), len(v), f1(r),
                         len(vals(off, offarm, get)), pct(m, r), tgt(m, TARGET.get(nm))))
            if nm in HEAD: sp.append("%s %s (OFF %s, %s%s)" % (nm, f1(m), f1(r), pct(m, r), tgt(m, TARGET.get(nm))))
        u, um = [], []
        su0 = start_units(P.get("off_stats0", "")) if src == "hx" else None
        for (pn, msget), xp in zip(MS, ("c1", "c4", "c8")):
            uon = mean(vals(on, base, path(xp, "xstats", "vram_units_end")))
            uoff = su0 if src == "hx" else mean(vals(off, offarm, path(xp, "xstats", "vram_units_end")))
            du = (uon - uoff) if uon is not None and uoff is not None else None
            mon, moff = mean(vals(on, base, msget)), mean(vals(off, offarm, msget))
            dm = (mon - moff) if mon is not None and moff is not None else None
            pred = -SLOPE[pn] * du / 100 if du is not None else None
            u.append("%s: units %s vs OFF %s%s = %s (audit %s); hit %s vs %s %%, missed %s vs %s, stream %s vs %s MB a step; ms %s vs %s = %s measured, %s by the slope" % (
                pn, f1(uon, 0), f1(uoff, 0), " (OFF at start)" if src == "hx" else "", "-" if du is None else "%+.0f" % du, f1(exp_st, 0),
                f1(mean(vals(on, base, path(xp, "xstats", "hit_pct_rank0"))), 2), f1(mean(vals(off, offarm, path(xp, "xstats", "hit_pct_rank0"))), 2),
                f1(mean(vals(on, base, path(xp, "xstats", "missed_step_rank0"))), 2), f1(mean(vals(off, offarm, path(xp, "xstats", "missed_step_rank0"))), 2),
                f1(mean(vals(on, base, path(xp, "xstats", "stream_MB_step")))), f1(mean(vals(off, offarm, path(xp, "xstats", "stream_MB_step")))),
                f1(mon, 2), f1(moff, 2), "-" if dm is None else "%+.2f" % dm, "-" if pred is None else "%+.2f" % pred))
            um.append("%s %s u (est %s), %s ms (slope %s)" % (pn, "-" if du is None else "%+.0f" % du, f1(exp_st, 0),
                      "-" if dm is None else "%+.2f" % dm, "-" if pred is None else "%+.2f" % pred))
        ton, toff = cold(js(P.get("on_deep", ""))), cold(js(P.get("off_deep", "")))
        dl = "136k cold TTFT %s s vs OFF %s (%s), vs dsm-225 %s (%s)%s" % (f1(ton, 2), f1(toff, 2), pct(ton, toff), f1(rt, 2), pct(ton, rt), tgt(ton, 35.0, True))
        fnl = [x for x in P.get("fn", "-").split(",") if x and x != "-"]
        stl = ",".join(x for x in P.get("levers", "").split(",") if x and x not in fnl) or "-"
        print("NST %s START LEVERS [%s] (arm %s of %s) vs OFF (%s, arm %s%s): %s || %s || %s || units at start ON %s, OFF %s" % (
            nst, stl, base, P.get("on_tag"), src, offarm, ", first %d OFF block(s) dropped" % skip if skip else "",
            "; ".join(parts), dl, " | ".join(u), f1(start_units(P.get("on_stats0", "")), 0), f1(su0, 0) if src == "hx" else "-"))
        line = "%s start levers [%s] (%s vs OFF %s): %s; %s; units %s" % (nst, stl, base, src, ", ".join(sp), dl, "; ".join(um))
        # ---- everything: the fullest arm vs OFF
        if full != base:
            ev = []
            for nm, get in M:
                if nm not in HEAD: continue
                m, h = ci(vals(on, full, get)); r = mean(vals(off, offarm, get))
                ev.append("%s %s%s vs OFF %s (%s)%s" % (nm, f1(m), "" if h is None else " +- " + f1(h), f1(r), pct(m, r), tgt(m, TARGET.get(nm))))
            print("NST %s EVERYTHING (arm %s: start levers + %s) vs OFF (%s): %s" % (nst, full, P.get("fn", "-"), src, "; ".join(ev)))
            line += "; everything (%s) vs OFF: %s" % (full, ", ".join(ev))
        # ---- in-process arms
        if len(arms) > 1:
            meas = [r for r in on if not r.get("settle")]
            k = len(arms)
            groups = [meas[i:i + k] for i in range(0, len(meas) - k + 1, k)]
            pl_ = []
            for xi, x in enumerate(arms):
                for y in arms[:xi]:
                    out = []
                    for nm, get in M + [("ms " + n, g) for n, g in MS]:
                        pr = []
                        for grp in groups:
                            ry = [r for r in grp if r.get("arm") == y]; rx = [r for r in grp if r.get("arm") == x]
                            if len(ry) == 1 and len(rx) == 1 and good(ry[0]) and good(rx[0]):
                                vy, vx = get(ry[0]["res"]), get(rx[0]["res"])
                                if vy and vx is not None: pr.append(vx / vy)
                        m, h = ci(pr); my, mx = mean(vals(on, y, get)), mean(vals(on, x, get))
                        out.append("%s pooled %s, pairs %s" % (nm, pct(mx, my), "-" if m is None else "%+.1f %%%s (n %d)" % (
                            (m - 1) * 100, "" if h is None else " +- %.1f" % (h * 100), len(pr))))
                    print("PAIRS %s %s vs %s (%s vs %s): %s" % (nst, x, y, json.dumps(next((r.get("values") for r in on if r.get("arm") == x), {})),
                                                            json.dumps(next((r.get("values") for r in on if r.get("arm") == y), {})), "; ".join(out)))
                    pl_.append("%s vs %s: %s" % (x, y, ", ".join(o for o in out if o.split(" pooled")[0] in HEAD + ("ms c4", "ms c8"))))
            for x in arms:
                print("DL %s arm %s: %s" % (nst, x, "; ".join("%s %s" % (nm, f1(mean(vals(on, x, g)), 2 if "share" in nm else 1)) for nm, g in DL)))
            line += "; in process: " + " | ".join(pl_)
            line += "; DL: " + " | ".join("%s c1 essay zero %s, c4 share %s moves %s lent %s, c8 share %s lent %s max %s first %s s 95%% %s s borrow %s" % (
                x, f1(mean(vals(on, x, DL[0][1])), 0), f1(mean(vals(on, x, share("c4"))), 2), f1(mean(vals(on, x, path("draftph", "c4", "zero_moves"))), 0),
                f1(mean(vals(on, x, path("c4", "xlend", "s_lent_share"))), 2), f1(mean(vals(on, x, share("c8"))), 2),
                f1(mean(vals(on, x, path("c8", "xlend", "s_lent_share"))), 2), f1(mean(vals(on, x, path("c8", "xlend", "s_lent_units_max"))), 0),
                f1(mean(vals(on, x, path("c8", "xlend", "s_t_first_lent")))), f1(mean(vals(on, x, path("c8", "xlend", "s_t_to_95pct_max")))),
                f1(mean(vals(on, x, path("c8", "xlend", "borrow_cycles"))), 0)) for x in arms)
            line += " (fn4 audit estimate %s units)" % f1(exp_fn, 0)
        nset = sum(1 for r in on if r.get("settle")); bad = sum(1 for r in on if not r.get("settle") and not good(r))
        line += "; blocks ON %d (+%d settle, %d left out), OFF %d (%s)" % (len(on) - nset, nset, bad, len(off), src)
        summ.append(line)
    s = " || ".join(summ) if summ else "no configuration measured"
    s += " || identity: " + (", ".join(idl) if idl else "none recorded")
    s += " || targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s, 136k <= 35 s; slope = vramsweep T150 vs T0 (NST 0)"
    print("SUMMARY " + s)


if __name__ == "__main__":
    main()
