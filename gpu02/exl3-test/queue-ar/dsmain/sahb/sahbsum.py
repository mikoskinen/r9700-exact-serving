#!/usr/bin/env python3
"""sahbsum.py (12.11.2026, queue-ar/dsmain/sahb, ds-main-port sahb builder) -- the readings and guards of dsm-sahb.sh (item
57-p4zyzzzzz-ds-r00zzz-cz-usahb, levers SA / HB on image ds41-sahb). Standard library only.

  sahbsum.py pick --run NAME=ARMKIND:NCA.json:PRE_FN4.json:POST_FN4.json [...] --default K [--json OUT]
      the alone screen: per run ncsteps.py alone's JSON (step wall from /stats, staged / stream bytes a prompt) and the fn4
      GET "sahb" snapshots around it. ARMKIND B (stage_short 0) or P<K> (stage_short on, stage_pick K). Guard: a B run arms
      nothing (short_steps +0 on every rank), a P run arms every prompt on every rank (short_steps + prompts) and copies
      units; ncsteps' own guard empty. The pick = the P run with the lowest mean step wall (ties: the lower K); no clean P
      run: --default. Prints "PICK K" then one "SCREEN ..." line.
  sahbsum.py block --name N --arm A --want WANT.json --pre-fn4 F --post-fn4 F --pre-st F --post-st F --c8 NCC.json
                   --c1 STEPPROBE.txt --sa 0|1 --hb R --hb-ranks 1,2 [--json OUT]
      one ABBA block: the 1u stepprobe (essay / code), the mb c8 window (ncsteps.py c8's JSON: agg_dec_tps, decode_share,
      the newcomer step wall from the step log, non-decode / decode step ms, staged / streamed GB a non-decode step), the fn4
      "sahb" deltas per rank, per-rank streamed GB a non-decode step (/stats experts.ranks[].stream_bytes), the PSK counters.
      COUNTER GUARD (else VIRHE and no speed reading for the block): every switch of WANT applied, nothing pending (pre and
      post); SA arms: short_steps and short_units grow on every rank, non-SA arms: short_steps +0 on every rank; HB arms:
      hb_reserve = R on the named ranks and 0 elsewhere, hb_reserved > 0 on the named ranks; non-HB arms: hb_reserve 0 and
      hb_promotions +0 everywhere; PSK on the corrected counters: psk.latched, psk.planned_steps, psk.cut_passes grow,
      psk.plan_mismatch +0 (never layers_skipped); the c8 window clean (ncsteps guard empty, 0 errors); both 1u prompts.
  sahbsum.py report --block FILE.json [...] [--prof PROF.json] [--screen SCREEN.json] [--ident IDENT.txt] [--ref TEXT]
      per arm the clean blocks' means, every arm vs B (pooled and ABBA-adjacent pairs), the START P per-rank table
      (experts / collective / dense ms a newcomer step per rank and arm), identity; "RESULT xsahb ..." lines.
  sahbsum.py selftest
"""
import argparse, json, statistics, sys

sys.dont_write_bytecode = True
KEYS = ("short_steps", "short_layers", "short_units", "short_bytes", "pick_kept", "pick_dropped", "big_steps", "expect_units",
        "expect_bytes", "expect_sum_units", "expect_n", "hb_reserve", "hb_reserved", "hb_resident", "hb_promotions",
        "sa_recall_bytes")


def rd(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return {}


def ranks(g):
    return {r.get("rank"): r for r in ((g.get("sahb") or {}).get("ranks") or [])}


def dsahb(a, b):
    """per rank: deltas of the cumulative counters, end values of the level ones."""
    ra, rb = ranks(a), ranks(b)
    out = {}
    for k in sorted(rb):
        x, y = ra.get(k) or {}, rb[k]
        out[k] = {c: ((y.get(c) or 0) - (x.get(c) or 0)) if c not in ("hb_reserve", "hb_reserved", "hb_resident", "expect_units",
                                                                     "expect_bytes") else (y.get(c) or 0) for c in KEYS}
    return out


def applied(g, want):
    sw = g.get("switches") or {}
    bad = {k: sw.get(k) for k, v in want.items() if k != "epoch_bump" and sw.get(k) != v}
    return bad, g.get("pending") or {}


def num(d, *p):
    x = d
    for k in p:
        if not isinstance(x, dict): return 0
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None


def r(x, n=1):
    return None if x is None else round(x, n)


def pct(a, b):
    return None if a is None or b in (None, 0) else round(100.0 * (a / b - 1.0), 1)


# ---------------------------------------------------------------- pick
def cmd_pick(a):
    rows, out = [], {}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        kind, nca, pre, post = rest.split(":")
        d, g0, g1 = rd(nca), rd(pre), rd(post)
        ds = dsahb(g0, g1)
        n = d.get("ok") or 0
        guard = list(d.get("guard") or [])
        if not ds: guard.append("no fn4 sahb counters")
        for k, v in ds.items():
            if kind == "B" and v["short_steps"] != 0: guard.append("rank %s armed %d short passes under B" % (k, v["short_steps"]))
            if kind != "B" and (v["short_steps"] < n or v["short_units"] <= 0):
                guard.append("rank %s: %d short passes / %d units for %d prompts" % (k, v["short_steps"], v["short_units"], n))
        sm = (d.get("step_ms") or {}).get("mean")
        r0 = ds.get(0) or {}
        row = {"name": name, "kind": kind, "n": n, "step_ms": sm, "staged_gb_r0": r((r0.get("short_bytes") or 0) / max(1, n) / 1e9, 3),
               "units_r0": r((r0.get("short_units") or 0) / max(1, n), 1), "kept_r0": r((r0.get("pick_kept") or 0) / max(1, n), 1),
               "dropped_r0": r((r0.get("pick_dropped") or 0) / max(1, n), 1), "expect_r0": r0.get("expect_units"),
               "stream_gb": (d.get("link_stream_gb") or {}).get("mean"), "rank_stream_gb": d.get("rank_stream_gb"), "guard": guard}
        rows.append(row)
    ok = [x for x in rows if x["kind"].startswith("P") and not x["guard"] and x["step_ms"] is not None]
    best = min(ok, key=lambda x: (x["step_ms"], int(x["kind"][1:]))) if ok else None
    k = int(best["kind"][1:]) if best else a.default
    bms = mean([x["step_ms"] for x in rows if x["kind"] == "B" and not x["guard"]])
    out = {"pick": k, "fallback": best is None, "b_step_ms": r(bms), "rows": rows}
    print("PICK %d" % k)
    line = "SCREEN alone newcomers (one ~1.1k prompt a step, unprofiled): " + " | ".join(
        "%s %s: step %s ms (%s vs B) over %d, rank 0 staged %s units %s GB a step (pick kept %s / left in place %s), expect %s, link stream %s GB, per-rank stream %s%s" % (
            x["name"], x["kind"], r(x["step_ms"]), ("%+.1f %%" % pct(x["step_ms"], bms)) if x["kind"] != "B" and pct(x["step_ms"], bms) is not None else "-",
            x["n"], x["units_r0"], x["staged_gb_r0"], x["kept_r0"], x["dropped_r0"], x["expect_r0"], x["stream_gb"], x["rank_stream_gb"],
            (" -- VIRHE " + "; ".join(x["guard"])[:300]) if x["guard"] else "") for x in rows) + \
        " => stage_pick %d%s" % (k, " (DEFAULT: no clean P run)" if best is None else " (the fastest clean P run)")
    print(line)
    out["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f)
    return 0


# ---------------------------------------------------------------- block
def c1read(p):
    res = {}
    try:
        for l in open(p):
            if l.startswith("J "):
                try: d = json.loads(l[2:])
                except ValueError: continue
                res[d.get("prompt")] = d
    except OSError:
        pass
    return res


def cmd_block(a):
    want = json.loads(open(a.want).read()) if a.want else {}
    g0, g1, s0, s1 = rd(a.pre_fn4), rd(a.post_fn4), rd(a.pre_st), rd(a.post_st)
    c8 = rd(a.c8)
    c1 = c1read(a.c1)
    hbr = set(int(x) for x in a.hb_ranks.split(",") if x.strip() != "")
    guard = []
    for nm, g in (("pre", g0), ("post", g1)):
        bad, pend = applied(g, want)
        if not g: guard.append("no fn4 GET %s" % nm)
        elif bad or pend: guard.append("pins not applied (%s): %s pending %s" % (nm, bad, pend))
    ds = dsahb(g0, g1)
    if len(ds) < 4: guard.append("fn4 sahb counters for %d ranks" % len(ds))
    for k, v in ds.items():
        if a.sa:
            if v["short_steps"] <= 0 or v["short_units"] <= 0:
                guard.append("SA did nothing on rank %s (short passes +%d, units +%d)" % (k, v["short_steps"], v["short_units"]))
        elif v["short_steps"] != 0:
            guard.append("rank %s armed %d short passes with stage_short 0" % (k, v["short_steps"]))
        if a.hb > 0:
            wr = a.hb if k in hbr else 0
            if v["hb_reserve"] != wr: guard.append("rank %s hb_reserve %s, want %s" % (k, v["hb_reserve"], wr))
            if k in hbr and v["hb_reserved"] <= 0: guard.append("HB holds nothing on rank %s (hb_reserved 0)" % k)
        else:
            if v["hb_reserve"] != 0 or v["hb_promotions"] != 0:
                guard.append("rank %s: hb_reserve %s, hb_promotions +%s with place_hb 0" % (k, v["hb_reserve"], v["hb_promotions"]))
    pk = lambda d, c: num(d, "psk", c)   # noqa: E731
    psk = {c: pk(s1, c) - pk(s0, c) for c in ("latched", "planned_steps", "cut_passes", "plan_mismatch", "rows_skipped_l30_39")}
    if not isinstance(s1.get("psk"), dict): guard.append("no /stats psk counters")
    else:
        for c in ("latched", "planned_steps", "cut_passes"):
            if psk[c] <= 0: guard.append("psk.%s +%s (PSK design 2 pinned but silent)" % (c, psk[c]))
        if psk["plan_mismatch"] != 0: guard.append("psk.plan_mismatch +%s" % psk["plan_mismatch"])
    cg = c8.get("guard")
    if not c8: guard.append("no c8 reading")
    elif cg: guard.append("c8: " + "; ".join(cg)[:200])
    mb = c8.get("mb") or {}
    if (mb.get("errors") or 0) > 0: guard.append("c8 %s errors" % mb.get("errors"))
    for p in ("essay", "code"):
        if p not in c1: guard.append("no 1u %s line" % p)
    st = c8.get("stats") or {}
    nd = (st.get("prefill") or 0) + (st.get("mixed") or 0)
    # per-rank streamed GB a non-decode step (window /stats delta over the c8 phase only is ncsteps'; here the block's, labelled)
    ra = {x.get("rank"): x for x in (s0.get("experts") or {}).get("ranks") or []}
    rb = {x.get("rank"): x for x in (s1.get("experts") or {}).get("ranks") or []}
    nd_blk = (num(s1, "steps_prefill") + num(s1, "steps_mixed")) - (num(s0, "steps_prefill") + num(s0, "steps_mixed"))
    rank_stream = [r(((rb[k].get("stream_bytes") or 0) - ((ra.get(k) or {}).get("stream_bytes") or 0)) / max(1, nd_blk) / 1e9, 3) for k in sorted(rb)]
    nc = c8.get("newcomers") or {}
    sa0 = ds.get(0) or {}
    out = {"name": a.name, "arm": a.arm, "ok": not guard, "guard": guard,
           "essay_tps": (c1.get("essay") or {}).get("decode_tps"), "code_tps": (c1.get("code") or {}).get("decode_tps"),
           "essay_ms": (c1.get("essay") or {}).get("ms_per_decode_step"), "code_ms": (c1.get("code") or {}).get("ms_per_decode_step"),
           "c8_tps": mb.get("agg_dec_tps"), "c8_share_mb": mb.get("decode_share"), "c8_share": st.get("decode_share"),
           "nc_wall_p50": (nc.get("wall_ms") or {}).get("p50"), "nc_wall_mean": (nc.get("wall_ms") or {}).get("mean"), "newcomers": nc.get("n"),
           "nc_mixed": nc.get("mixed"), "nd_ms": st.get("nondecode_ms_mean"), "dec_ms": st.get("decode_ms_mean"),
           "staged_gb_nd": (st.get("per_nd") or {}).get("link_staged_gb"), "stream_gb_nd": (st.get("per_nd") or {}).get("link_stream_gb"),
           "stream_gb_nd_reg": ((c8.get("regression") or {}).get("link_stream_gb") or {}).get("per_nd"),
           "staged_gb_nd_reg": ((c8.get("regression") or {}).get("link_staged_gb") or {}).get("per_nd"),
           "rank_stream_gb_nd": rank_stream, "nd_steps_block": nd_blk, "nd_steps_c8": nd,
           "sa_r0": {k: sa0.get(k) for k in KEYS}, "sahb": ds, "psk": psk}
    s = out
    line = "BLOCK %s arm %s: 1u essay %s / code %s tok/s (%s / %s ms a step) | c8 %s tok/s, decode_share %s (MB %s), newcomer step p50 %s / mean %s ms (%s newcomers, %s MIXED), non-decode %s ms, decode %s ms | a non-decode step: staged %s GB, stream %s GB (regression %s / %s), per-rank stream %s GB | rank 0 SA: +%s short passes, %s units, %s GB, pick kept %s / left %s, expect %s | HB per rank reserve %s held %s resident %s promotions +%s | recall %s GB | psk latched +%s planned +%s cut +%s mismatch +%s%s" % (
        a.name, a.arm, s["essay_tps"], s["code_tps"], s["essay_ms"], s["code_ms"], s["c8_tps"], s["c8_share"], s["c8_share_mb"], s["nc_wall_p50"],
        s["nc_wall_mean"], s["newcomers"], s["nc_mixed"], s["nd_ms"], s["dec_ms"], s["staged_gb_nd"], s["stream_gb_nd"], s["staged_gb_nd_reg"],
        s["stream_gb_nd_reg"], rank_stream, sa0.get("short_steps"), sa0.get("short_units"), r((sa0.get("short_bytes") or 0) / 1e9, 2),
        sa0.get("pick_kept"), sa0.get("pick_dropped"), sa0.get("expect_units"),
        [ds[k]["hb_reserve"] for k in sorted(ds)], [ds[k]["hb_reserved"] for k in sorted(ds)], [ds[k]["hb_resident"] for k in sorted(ds)],
        [ds[k]["hb_promotions"] for k in sorted(ds)], r(sum(ds[k]["sa_recall_bytes"] for k in ds) / 1e9, 2),
        psk["latched"], psk["planned_steps"], psk["cut_passes"], psk["plan_mismatch"],
        (" -- VIRHE " + "; ".join(guard)[:600] + " (no speed reading from this block)") if guard else "")
    print(line)
    out["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f)
    return 0 if not guard else 2


# ---------------------------------------------------------------- report
MET = (("c8_tps", "c8 agg_dec_tps", 1, +1), ("c8_share", "c8 decode_share", 3, +1), ("nc_wall_p50", "newcomer step p50 ms", 1, -1),
       ("nd_ms", "non-decode step ms", 1, -1), ("dec_ms", "decode step ms (c8)", 2, -1), ("essay_tps", "1u essay tok/s", 2, +1),
       ("code_tps", "1u code tok/s", 2, +1), ("staged_gb_nd", "staged GB a non-decode step", 2, 0),
       ("stream_gb_nd_reg", "streamed GB a non-decode step (regression)", 2, -1))


def cmd_report(a):
    bl = [rd(p) for p in a.block]
    bl = [b for b in bl if b]
    arms = []
    for b in bl:
        if b["arm"] not in arms: arms.append(b["arm"])
    good = [b for b in bl if b.get("ok")]
    bad = [b for b in bl if not b.get("ok")]
    lines = []
    for key, lab, nd, sgn in MET:
        base = mean([b.get(key) for b in good if b["arm"] == "B"])
        parts = []
        for arm in arms:
            v = [b.get(key) for b in good if b["arm"] == arm]
            m = mean(v)
            # ABBA-adjacent pairs: each block of ARM against the nearest B block
            pr = []
            for i, b in enumerate(bl):
                if b["arm"] != arm or arm == "B" or not b.get("ok") or b.get(key) is None: continue
                near = sorted((abs(j - i), j) for j, c in enumerate(bl) if c["arm"] == "B" and c.get("ok") and c.get(key) is not None)
                if near: pr.append(pct(b[key], bl[near[0][1]][key]))
            parts.append("%s %s (n %d%s%s)" % (arm, r(m, nd), len([x for x in v if x is not None]),
                                              "" if arm == "B" or pct(m, base) is None else ", %+.1f %% vs B" % pct(m, base),
                                              "" if not pr else ", pairs " + "/".join("%+.1f" % x for x in pr if x is not None)))
        lines.append("RESULT xsahb summary %s: %s" % (lab, " | ".join(parts)))
    # per-rank stream a non-decode step and the counters
    for arm in arms:
        g = [b for b in good if b["arm"] == arm]
        if not g: continue
        rs = [mean([b["rank_stream_gb_nd"][i] for b in g if len(b.get("rank_stream_gb_nd") or []) > i]) for i in range(4)]
        sa = g[0].get("sahb") or {}
        lines.append("RESULT xsahb summary arm %s counters (clean blocks %d): per-rank streamed GB a non-decode step (block /stats, decode "
                     "steps' share included) %s | rank 0 SA a block: short passes %s, units %s, %s GB, pick kept %s / left in place %s | HB reserve %s "
                     "held %s resident %s promotions %s (per rank, first clean block) | SA recall %s GB a block" % (
                         arm, len(g), [r(x, 3) for x in rs], mean([(b.get("sa_r0") or {}).get("short_steps") for b in g]),
                         mean([(b.get("sa_r0") or {}).get("short_units") for b in g]),
                         r(mean([((b.get("sa_r0") or {}).get("short_bytes") or 0) / 1e9 for b in g]), 2),
                         mean([(b.get("sa_r0") or {}).get("pick_kept") for b in g]), mean([(b.get("sa_r0") or {}).get("pick_dropped") for b in g]),
                         [sa[k]["hb_reserve"] for k in sorted(sa)], [sa[k]["hb_reserved"] for k in sorted(sa)],
                         [sa[k]["hb_resident"] for k in sorted(sa)], [sa[k]["hb_promotions"] for k in sorted(sa)],
                         r(mean([sum((b.get("sahb") or {})[k]["sa_recall_bytes"] for k in (b.get("sahb") or {})) / 1e9 for b in g]), 2)))
    if bad:
        lines.append("RESULT xsahb summary: VIRHE %d block(s) failed the counter guard and give no speed reading: %s" % (
            len(bad), " | ".join("%s %s: %s" % (b["name"], b["arm"], "; ".join(b.get("guard") or [])[:300]) for b in bad)))
    # START P: per-rank experts / collective / dense ms a newcomer step, per arm
    pf = rd(a.prof) if a.prof else {}
    w = pf.get("windows") or {}
    for arm in arms:
        x = w.get(arm) or {}
        rk = x.get("ranks") or {}
        if not rk: continue
        cell = []
        for k in sorted(rk, key=lambda z: int(z)):
            c = (rk[k] or {}).get("by_class_ms") or {}
            cell.append("r%s experts %s coll %s dense %s" % (k, r(c.get("experts")), r(c.get("collective")), r(c.get("dense_gemm"))))
        bx = (w.get("B") or {}).get("ranks") or {}
        mx = lambda rr: max(((rr[k] or {}).get("by_class_ms") or {}).get("experts") or 0 for k in rr) if rr else None   # noqa: E731
        lines.append("RESULT xsahb profiled alone newcomer step arm %s (PROFILED, %s steps, wall %s ms): %s | slowest rank's experts %s ms (B %s)" % (
            arm, x.get("steps"), r((x.get("wall_ms_by_phase") or {}).get("prefill")), "; ".join(cell), r(mx(rk)), r(mx(bx))))
    if a.screen:
        sc = rd(a.screen)
        if sc.get("line"): lines.append("RESULT xsahb " + sc["line"][:3000])
    if a.ident:
        try:
            idl = [l.strip() for l in open(a.ident) if l.strip()]
        except OSError:
            idl = []
        nd_ = [l for l in idl if "IDENTICAL" not in l]
        lines.append("RESULT xsahb identity: %d compares, %s" % (len(idl), "all IDENTICAL" if idl and not nd_ else
                                                                  ("NOT ALL: " + " | ".join(nd_)[:900]) if idl else "none"))
    if a.ref: lines.append("RESULT xsahb references (cross-start, labelled): " + a.ref)
    for l in lines: print(l)
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    import os, tempfile
    d = tempfile.mkdtemp()
    def w(n, o):
        p = os.path.join(d, n)
        with open(p, "w") as f: json.dump(o, f)
        return p
    def g(sw, ss, units, res, held, prom):
        return {"switches": sw, "pending": {}, "sahb": {"ranks": [dict({k: 0 for k in KEYS}, rank=i, short_steps=ss, short_units=units,
                                                                         hb_reserve=res if i in (1, 2) else 0, hb_reserved=held if i in (1, 2) else 0,
                                                                         hb_promotions=prom) for i in range(4)]}}
    want = {"stage_short": 1025, "stage_pick": 1, "place_hb": 0}
    sw = dict(want)
    pre = w("g0.json", g(sw, 0, 0, 0, 0, 0)); post = w("g1.json", g(sw, 16, 900, 0, 0, 0))
    st0 = w("s0.json", {"steps_prefill": 0, "steps_mixed": 0, "psk": {"latched": 1, "planned_steps": 3, "cut_passes": 1, "plan_mismatch": 0},
                        "experts": {"ranks": [{"rank": i, "stream_bytes": 0} for i in range(4)]}})
    st1 = w("s1.json", {"steps_prefill": 0, "steps_mixed": 16, "psk": {"latched": 9, "planned_steps": 40, "cut_passes": 9, "plan_mismatch": 0},
                        "experts": {"ranks": [{"rank": i, "stream_bytes": 16e9 * (1.0 + 0.1 * i)} for i in range(4)]}})
    c8 = w("c8.json", {"guard": [], "mb": {"agg_dec_tps": 185.0, "decode_share": 0.84, "errors": 0},
                       "stats": {"prefill": 0, "mixed": 16, "decode_share": 0.84, "nondecode_ms_mean": 420.0, "decode_ms_mean": 35.0,
                                 "per_nd": {"link_staged_gb": 4.0, "link_stream_gb": 9.0}},
                       "newcomers": {"n": 16, "mixed": 16, "wall_ms": {"p50": 410.0, "mean": 415.0}}, "regression": {}})
    c1 = os.path.join(d, "c1.txt")
    with open(c1, "w") as f:
        f.write('J {"prompt": "essay", "decode_tps": 66.1, "ms_per_decode_step": 15.1}\nJ {"prompt": "code", "decode_tps": 68.0, "ms_per_decode_step": 14.8}\n')
    wf = w("want.json", want)
    ns = argparse.Namespace(name="b1", arm="SA", want=wf, pre_fn4=pre, post_fn4=post, pre_st=st0, post_st=st1, c8=c8, c1=c1, sa=1, hb=0,
                            hb_ranks="1,2", json=os.path.join(d, "b1.json"))
    rc1 = cmd_block(ns)
    # CONTROL: the same counters under a B arm (stage_short 0) must fail the guard
    ns2 = argparse.Namespace(**dict(vars(ns), name="b2", arm="B", sa=0, json=os.path.join(d, "b2.json")))
    rc2 = cmd_block(ns2)
    # CONTROL: HB named but no reserve applied
    ns3 = argparse.Namespace(**dict(vars(ns), name="b3", arm="SH", hb=128, json=os.path.join(d, "b3.json")))
    rc3 = cmd_block(ns3)
    # a pick screen
    nb = w("nb.json", {"ok": 8, "guard": [], "step_ms": {"mean": 590.0}})
    np1 = w("np1.json", {"ok": 8, "guard": [], "step_ms": {"mean": 450.0}})
    np0 = w("np0.json", {"ok": 8, "guard": [], "step_ms": {"mean": 640.0}})
    gb0, gb1 = w("gb0.json", g({}, 0, 0, 0, 0, 0)), w("gb1.json", g({}, 0, 0, 0, 0, 0))
    gp0, gp1 = w("gp0.json", g({}, 0, 0, 0, 0, 0)), w("gp1.json", g({}, 8, 400, 0, 0, 0))
    gq0, gq1 = w("gq0.json", g({}, 8, 400, 0, 0, 0)), w("gq1.json", g({}, 16, 1000, 0, 0, 0))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_pick(argparse.Namespace(run=["s1=B:%s:%s:%s" % (nb, gb0, gb1), "s2=P0:%s:%s:%s" % (np0, gq0, gq1), "s3=P1:%s:%s:%s" % (np1, gp0, gp1)],
                                    default=1, json=None))
        cmd_report(argparse.Namespace(block=[os.path.join(d, "b1.json"), os.path.join(d, "b2.json")], prof=None, screen=None, ident=None, ref=None))
    o = buf.getvalue()
    ok = rc1 == 0 and rc2 == 2 and rc3 == 2 and "PICK 1" in o and "RESULT xsahb summary c8 agg_dec_tps" in o and "VIRHE 1 block" in o
    print("SELFTEST " + ("OK: a clean SA block passes; CONTROLS: the same counters under B and an HB arm without a reserve fail; the "
                         "screen picks the fastest clean P run; the report" if ok else "FAILED rc %s %s %s | %s" % (rc1, rc2, rc3, o[-500:])))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("pick"); p.add_argument("--run", action="append", default=[]); p.add_argument("--default", type=int, default=1)
    p.add_argument("--json")
    p = sub.add_parser("block")
    for k in ("--name", "--arm", "--want", "--pre-fn4", "--post-fn4", "--pre-st", "--post-st", "--c8", "--c1"): p.add_argument(k, required=True)
    p.add_argument("--sa", type=int, required=True); p.add_argument("--hb", type=int, required=True); p.add_argument("--hb-ranks", default="1,2")
    p.add_argument("--json")
    p = sub.add_parser("report"); p.add_argument("--block", action="append", default=[]); p.add_argument("--prof"); p.add_argument("--screen")
    p.add_argument("--ident"); p.add_argument("--ref")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "pick": return cmd_pick(a)
    if a.cmd == "block": return cmd_block(a)
    if a.cmd == "report": return cmd_report(a)
    if a.cmd == "selftest": return selftest()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
