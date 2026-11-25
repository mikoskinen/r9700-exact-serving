#!/usr/bin/env python3
"""s5ctr.py (12.11.2026, queue-ar/dsmain/stack5, ds-main-port stack5 builder) -- the COUNTER GUARD of 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5
(coordinator 12.11.: no silent lever). Standard library only; s5blocks.py imports it from its own directory, the body calls it.

A reading = two snapshots, each the full GET /stats and GET /admin/fn4 (taken AFTER the arm's switches were read back), and
the arm's lever list. Counters (the int8 image's names):
  psk     /stats psk: latched, planned_steps, cut_passes, rows_skipped (= rows_skipped_l21_29 + rows_skipped_l30_39) grow and
          plan_mismatch does not. NEVER layers_skipped (a loaded step need not skip a layer whole).
  pskp    the light form for readings without a long prefill (T 1 sdprobe): latched, planned_steps grow.
  l20     /stats psk: l20_cut_passes, l20_rows_skipped grow.
  pfctx   /stats dspark_pfctx: passes, rows_skipped grow.
  engdd   /stats engdd: passes, rows_deduped grow.
  noa     GET /admin/fn4 noa: pieces, rows grow (pf_steps_* count every step with prefill rows, NOA or not: never a guard).
  sa      GET /admin/fn4 sahb.ranks[]: short_steps and short_units grow on EVERY rank.
  hb      sahb.ranks[]: hb_reserve = R on the named ranks and 0 elsewhere (a gauge, at the end), hb_reserved > 0 on the named ranks.
  hbr     the reserve only (hb_reserve = R on the named ranks): readings too short for the heat to fill it.
  tail    /stats psk: tail_only_latched grows, tail_only_refused does not (ds41_psk 4 + ds41_psk_cr 1).
  cr      /stats psk: cr_hits, cr_passes grow, cr_fallbacks does not (a planned prefix hit replayed).
  crseen  /stats psk: cr_hits + cr_not_needed + cr_cheaper grow (lever PSK-CR looked at a planned prefix hit: replayed it, a snapshot
          sat at its grid point, or resuming at the snapshot below was the cheaper exact path (ds41-pskcr2)), cr_fallbacks does not
          -- the turn chain's hits under S5T.
FORBID (a lever OFF in the arm moves nothing): psk latched / planned_steps / cut_passes / rows_skipped +0; l20_cut_passes +0;
  dspark_pfctx passes +0; engdd passes +0; noa admitted / pieces / rows / displaced / waited / steps +0; sa short_steps +0 on every
  rank; hb hb_reserve 0 on every rank and hb_promotions +0; cr / tail cr_hits, cr_passes, tail_only_latched +0.

  s5ctr.py guard --need psk,l20 --forbid sa,hb --st0 A.json --st1 B.json --fn0 A.json --fn1 B.json [--hb 128 --hb-ranks 1,2]
      -> "OK | deltas" or "FAIL why | deltas" (rc 0 either way; rc 2 on unreadable input)
  s5ctr.py selftest
"""
import argparse, json, sys

LEVERS = ("psk", "pskp", "l20", "pfctx", "engdd", "noa", "sa", "hb", "hbr", "tail", "cr", "crseen")
NOAK = ("admitted", "pieces", "rows", "displaced", "waited", "steps")


def num(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict): return 0
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def compact(st, fn):
    """the counters this guard reads, from one /stats + one GET /admin/fn4"""
    st, fn = st or {}, fn or {}
    c = {"psk_latched": num(st, "psk", "latched"), "psk_planned_steps": num(st, "psk", "planned_steps"),
         "psk_cut_passes": num(st, "psk", "cut_passes"),
         "psk_rows_skipped": num(st, "psk", "rows_skipped_l21_29") + num(st, "psk", "rows_skipped_l30_39"),
         "psk_layers_skipped": num(st, "psk", "layers_skipped"), "psk_plan_mismatch": num(st, "psk", "plan_mismatch"),
         "l20_cut_passes": num(st, "psk", "l20_cut_passes"), "l20_rows_skipped": num(st, "psk", "l20_rows_skipped"),
         "cr_hits": num(st, "psk", "cr_hits"), "cr_passes": num(st, "psk", "cr_passes"),
         "cr_rows_replayed": num(st, "psk", "cr_rows_replayed"), "cr_fallbacks": num(st, "psk", "cr_fallbacks"),
         "cr_not_needed": num(st, "psk", "cr_not_needed"), "cr_cheaper": num(st, "psk", "cr_cheaper"),
         "cr_end_snapshots": num(st, "psk", "cr_end_snapshots"),
         "tail_only_latched": num(st, "psk", "tail_only_latched"), "tail_only_refused": num(st, "psk", "tail_only_refused"),
         "fallback_resumes": num(st, "psk", "fallback_resumes"),
         "pfctx_passes": num(st, "dspark_pfctx", "passes"), "pfctx_rows_skipped": num(st, "dspark_pfctx", "rows_skipped"),
         "engdd_passes": num(st, "engdd", "passes"), "engdd_rows_deduped": num(st, "engdd", "rows_deduped"),
         "engdd_rows_pooled": num(st, "engdd", "rows_pooled")}
    for k in NOAK: c["noa_" + k] = num(fn, "noa", k)
    ranks = {}
    for r in ((fn.get("sahb") or {}).get("ranks") or []):
        if isinstance(r, dict) and isinstance(r.get("rank"), int):
            ranks[r["rank"]] = {k: num(r, k) for k in ("short_steps", "short_units", "hb_reserve", "hb_reserved", "hb_promotions")}
    c["sahb"] = ranks
    c["modes"] = {"psk": (st.get("psk") or {}).get("mode"), "k": (st.get("psk") or {}).get("k"), "cr": (st.get("psk") or {}).get("cr"),
                  "pfctx": (st.get("dspark_pfctx") or {}).get("mode")}
    return c


GAUGE = ("hb_reserve", "hb_reserved")


def delta(a, b):
    d = {k: b[k] - a.get(k, 0) for k in b if isinstance(b[k], (int, float))}
    d["sahb"] = {}
    for r, y in b["sahb"].items():
        x = a["sahb"].get(r, {})
        d["sahb"][r] = {k: (y[k] if k in GAUGE else y[k] - x.get(k, 0)) for k in y}
    d["modes"] = b.get("modes")
    return d


def guard(d, need, forbid, hb=0, hb_ranks=()):
    why = []
    g = lambda k: d.get(k, 0)   # noqa: E731
    sa = d.get("sahb") or {}
    for lv in need:
        if lv == "psk":
            for k in ("psk_latched", "psk_planned_steps", "psk_cut_passes", "psk_rows_skipped"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
            if g("psk_plan_mismatch") != 0: why.append("psk_plan_mismatch %+d" % g("psk_plan_mismatch"))
        elif lv == "pskp":
            for k in ("psk_latched", "psk_planned_steps"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
        elif lv == "l20":
            for k in ("l20_cut_passes", "l20_rows_skipped"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
        elif lv == "pfctx":
            for k in ("pfctx_passes", "pfctx_rows_skipped"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
        elif lv == "engdd":
            for k in ("engdd_passes", "engdd_rows_deduped"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
        elif lv == "noa":
            for k in ("noa_pieces", "noa_rows"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
        elif lv == "sa":
            if len(sa) < 4: why.append("sahb ranks %d (want 4)" % len(sa))
            for r, v in sorted(sa.items()):
                if v.get("short_steps", 0) <= 0 or v.get("short_units", 0) <= 0:
                    why.append("SA did nothing on rank %s (short_steps +%d, units +%d)" % (r, v.get("short_steps", 0), v.get("short_units", 0)))
        elif lv in ("hb", "hbr"):
            if len(sa) < 4: why.append("sahb ranks %d (want 4)" % len(sa))
            for r, v in sorted(sa.items()):
                want = hb if r in hb_ranks else 0
                if v.get("hb_reserve", 0) != want: why.append("rank %s hb_reserve %s, want %s" % (r, v.get("hb_reserve"), want))
                if lv == "hb" and r in hb_ranks and v.get("hb_reserved", 0) <= 0: why.append("HB holds nothing on rank %s" % r)
        elif lv == "tail":
            if g("tail_only_latched") <= 0: why.append("tail_only_latched +%d" % g("tail_only_latched"))
            if g("tail_only_refused") != 0: why.append("tail_only_refused %+d" % g("tail_only_refused"))
        elif lv == "crseen":
            if g("cr_hits") + g("cr_not_needed") + g("cr_cheaper") <= 0:
                why.append("PSK-CR saw no planned hit (cr_hits +%d, cr_not_needed +%d, cr_cheaper +%d)" % (g("cr_hits"), g("cr_not_needed"), g("cr_cheaper")))
            if g("cr_fallbacks") != 0: why.append("cr_fallbacks %+d" % g("cr_fallbacks"))
        elif lv == "cr":
            for k in ("cr_hits", "cr_passes"):
                if g(k) <= 0: why.append("%s +%d" % (k, g(k)))
            if g("cr_fallbacks") != 0: why.append("cr_fallbacks %+d" % g("cr_fallbacks"))
    for lv in forbid:
        ks = {"psk": ("psk_latched", "psk_planned_steps", "psk_cut_passes", "psk_rows_skipped"), "pskp": ("psk_latched", "psk_planned_steps"),
              "l20": ("l20_cut_passes",), "pfctx": ("pfctx_passes",), "engdd": ("engdd_passes",),
              "noa": tuple("noa_" + k for k in NOAK), "cr": ("cr_hits", "cr_passes", "tail_only_latched"),
              "tail": ("tail_only_latched",)}.get(lv, ())
        for k in ks:
            if g(k) != 0: why.append("%s moved %+d with the lever off" % (k, g(k)))
        if lv == "sa":
            for r, v in sorted(sa.items()):
                if v.get("short_steps", 0) != 0: why.append("rank %s armed %d short passes with stage_short 0" % (r, v["short_steps"]))
        if lv in ("hb", "hbr"):
            for r, v in sorted(sa.items()):
                if v.get("hb_reserve", 0) != 0 or v.get("hb_promotions", 0) != 0:
                    why.append("rank %s hb_reserve %s, hb_promotions +%s with place_hb 0" % (r, v.get("hb_reserve"), v.get("hb_promotions")))
    return why


def line(d):
    sa = d.get("sahb") or {}
    show = ("psk_latched", "psk_planned_steps", "psk_cut_passes", "psk_rows_skipped", "psk_layers_skipped", "psk_plan_mismatch",
            "l20_cut_passes", "l20_rows_skipped", "pfctx_passes", "pfctx_rows_skipped", "engdd_passes", "engdd_rows_deduped",
            "noa_admitted", "noa_pieces", "noa_rows", "noa_displaced", "noa_waited", "cr_hits", "cr_passes", "cr_not_needed", "cr_cheaper", "cr_end_snapshots", "cr_fallbacks",
            "tail_only_latched", "tail_only_refused", "fallback_resumes")
    s = ", ".join("%s %+d" % (k, d.get(k, 0)) for k in show if d.get(k, 0))
    s += " | sahb " + " ".join("r%s ss %+d su %+d res %d held %d pr %+d" % (r, v.get("short_steps", 0), v.get("short_units", 0),
                                                                            v.get("hb_reserve", 0), v.get("hb_reserved", 0), v.get("hb_promotions", 0))
                                for r, v in sorted(sa.items()))
    return (s or "nothing moved") + " | modes %s" % json.dumps(d.get("modes"))


def check(st0, st1, fn0, fn1, need, forbid, hb=0, hb_ranks=()):
    d = delta(compact(st0, fn0), compact(st1, fn1))
    why = guard(d, need, forbid, hb, hb_ranks)
    return ("OK" if not why else "FAIL " + "; ".join(why)), d


def rd(p):
    with open(p) as f: return json.load(f)


def lst(s):
    v = [x for x in (s or "").split(",") if x]
    bad = [x for x in v if x not in LEVERS]
    if bad: raise SystemExit("unknown lever(s) %s (know %s)" % (bad, ",".join(LEVERS)))
    return v


def selftest():
    def snap(lat=0, cut=0, rows=0, l20=0, pf=0, ed=0, noa=0, ss=0, res=0, held=0, pr=0, tail=0, cr=0):
        st = {"psk": {"mode": 2, "k": 8, "latched": lat, "planned_steps": lat, "cut_passes": cut, "rows_skipped_l21_29": rows,
                      "rows_skipped_l30_39": rows, "l20_cut_passes": l20, "l20_rows_skipped": l20 * 10, "tail_only_latched": tail,
                      "cr_hits": cr, "cr_passes": cr, "layers_skipped": 0, "plan_mismatch": 0},
              "dspark_pfctx": {"mode": 1, "passes": pf, "rows_skipped": pf * 5}, "engdd": {"passes": ed, "rows_deduped": ed * 7}}
        fn = {"noa": {"admitted": noa, "pieces": noa, "rows": noa * 100, "pf_steps_1": 99},
              "sahb": {"ranks": [{"rank": r, "short_steps": ss, "short_units": ss * 3, "hb_reserve": res if r in (1, 2) else 0,
                                  "hb_reserved": held if r in (1, 2) else 0, "hb_promotions": pr} for r in range(4)]}}
        return st, fn
    ok = True
    a0, f0 = snap(); a1, f1 = snap(lat=2, cut=5, rows=9, l20=1, pf=2, ed=3, noa=4)
    r, d = check(a0, a1, f0, f1, ["psk", "l20", "pfctx", "engdd", "noa"], ["sa", "hb", "cr"])
    ok &= r == "OK"
    r, d = check(a0, a1, f0, f1, [], ["psk", "l20", "pfctx", "engdd", "noa", "sa", "hb", "cr"])
    ok &= r.startswith("FAIL") and "psk_latched" in r and "noa_pieces" in r
    a2, f2 = snap(lat=2, cut=5, rows=9, l20=1, pf=2, ed=3, noa=4, ss=6, res=128, held=40, pr=7)
    r, d = check(a0, a2, f0, f2, ["psk", "l20", "pfctx", "engdd", "noa", "sa", "hb"], ["cr"], 128, (1, 2))
    ok &= r == "OK"
    r, d = check(a0, a2, f0, f2, ["psk"], ["sa", "hb"], 128, (1, 2))
    ok &= r.startswith("FAIL") and "short passes" in r and "hb_reserve" in r
    r, d = check(a0, a0, f0, f0, [], ["psk", "l20", "pfctx", "engdd", "noa", "sa", "hb", "cr"])
    ok &= r == "OK"
    a3, f3 = snap(lat=1, cut=1, rows=1, tail=1, cr=2)
    r, d = check(a0, a3, f0, f3, ["psk", "tail", "cr", "crseen"], ["sa"])
    ok &= r == "OK"
    r, d = check(a0, a1, f0, f1, ["crseen"], [])
    ok &= r.startswith("FAIL") and "PSK-CR saw no" in r
    a4 = json.loads(json.dumps(a1)); a4["psk"]["plan_mismatch"] = 1
    r, d = check(a0, a4, f0, f1, ["psk"], [])
    ok &= "plan_mismatch" in r
    print("SELFTEST OK" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("guard", "selftest"))
    ap.add_argument("--need", default="")
    ap.add_argument("--forbid", default="")
    ap.add_argument("--st0"); ap.add_argument("--st1"); ap.add_argument("--fn0"); ap.add_argument("--fn1")
    ap.add_argument("--hb", type=int, default=0)
    ap.add_argument("--hb-ranks", default="1,2")
    a = ap.parse_args()
    if a.mode == "selftest": return selftest()
    try:
        st0, st1, fn0, fn1 = rd(a.st0), rd(a.st1), rd(a.fn0), rd(a.fn1)
    except Exception as e:
        print("FAIL unreadable snapshot (%s)" % str(e)[:200]); return 2
    r, d = check(st0, st1, fn0, fn1, lst(a.need), lst(a.forbid), a.hb, tuple(int(x) for x in a.hb_ranks.split(",") if x))
    print("%s | %s" % (r, line(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
