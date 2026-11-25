#!/usr/bin/env python3
"""yllsum.py (12.11.2026, queue-ar/dsmain/yll, the A3 / A5 builder for ds-main-port; from ypfx/ypsum.py, arms generalised) -- the
LL / CHN / A10 screen's summary (57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yll.sh, body yll/dsm-yll.sh). Standard library only; reads the
body's output dir:

  yllsum.py OUTDIR [--dec-th 1.0] [--sw 1] [--ref-nc 538.2] [--ref-share 20.0] [--ref-dec 36.18] [--ref-agg 184.0]
      OUTDIR/mb.jsonl (mbx's rows: n, arm, agg_dec_tps, decode_share, ms_decode_step), OUTDIR/split-<block>.txt (p12win.py split's
      P12SPLIT line: (a) decode steps' ms a step, (b) newcomer / MIXED steps' count, share, ms a step), OUTDIR/rows-<block>.json
      (gt48), OUTDIR/fn-<block>-0/1.json (GET /admin/fn4: cap steps_gt48, the "arll" and "chain" counters), OUTDIR/blocks.jsonl
      (the 1u reads: lever U1 / U1W, arm), OUTDIR/cold.jsonl (coldx: depth, arm, TTFT), OUTDIR/identity.txt.
      Arms: B (ar_ll 0, chain_ds41 0, exl3_x3d_pf 0 = int12), C (ar_ll on), D (chain_ds41 1), E (both), P (exl3_x3d_pf 1, A10); E and P
      only by knob since the 12.11. trims (A10 rides on yd2m): their lines, A10's decode guard and its rule print only where they ran.
      Every reading as the values, B as the reference, X vs B in %. Lines "YLLSUM ...": per arm c8 (agg, decode ms a step from the step
      log and the MB line, decode_share, newcomer ms a step and share), the WIRE PROOF (LL launches / chained launches a block from
      the counters: C / E must show LL, D / E chained launches, B none), c16 (B and the best arm: agg, decode ms, > 48-row steps
      = 0), 1u essay confirm (tok/s, ms a step), cold 136k (B, P), A10's decode guard (P's c8 decode step not above B's by more
      than --dec-th %), A10's rule ("joins the stack if 136k TTFT >= 1 % faster OR newcomer step >= 15 ms faster": PASS / NOT),
      identity (IDENTICAL / compared), the staggered-arrival checks (ld-ARM vs ARM's a1, strict), one headline. 12.11. trimmed plan: arms
      B C D P (E only by knob), c16 B C.
  yllsum.py selftest"""
import json, os, re, statistics, sys, tempfile

DEC = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) ms a step, ([\d.]+|-) rows a step")
NEW = re.compile(r"\(b\) newcomer / MIXED (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) rows a step, ([\d.]+|-) ms a step")
ARMS = ("B", "C", "D", "E", "P")
DESC = {"B": "B = int12 (ar_ll 0, chain_ds41 0, exl3_x3d_pf 0)", "C": "C = ar_ll (LL to 48 rows)", "D": "D = chain_ds41 1",
        "E": "E = ar_ll + chain_ds41", "P": "P = exl3_x3d_pf 1 (A10)"}


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def jl(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError: pass
    return out


def jload(p):
    try: return json.load(open(p))
    except (OSError, ValueError): return None


def split_of(path):
    """P12SPLIT -> dict(dec_ms, dec_steps, nc_steps, nc_share, nc_ms, nc_rows) or None"""
    try: txt = open(path).read()
    except OSError: return None
    l = next((x for x in txt.splitlines() if x.startswith("P12SPLIT ") and "VIRHE" not in x), None)
    if not l: return None
    d, n = DEC.search(l), NEW.search(l)
    if not d or not n: return None
    return dict(dec_steps=int(d.group(1)), dec_ms=fnum(d.group(4)), nc_steps=int(n.group(1)), nc_share=fnum(n.group(3)),
                nc_rows=fnum(n.group(4)), nc_ms=fnum(n.group(5)))


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def pct(c, b):
    return 100.0 * (c / b - 1.0) if (c is not None and b) else None


def f(v, nd=2):
    return ("%." + str(nd) + "f") % v if v is not None else "-"


def fp(v):
    return "%+.2f %%" % v if v is not None else "-"


def sl(v, nd=1):
    return "/".join(f(z, nd) for z in v) or "-"


def cdelta(a0, a1, key, fields):
    x0, x1 = (a0 or {}).get(key), (a1 or {}).get(key)
    if not isinstance(x1, dict): return None
    x0 = x0 if isinstance(x0, dict) else {}
    return {k: (x1.get(k) or 0) - (x0.get(k) or 0) for k in fields}


def blocks(o, n):
    """the mb blocks of c<n>: [dict(name, arm, mb, split, gt48, gt48cap, arll, chain)] in run order"""
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        if r.get("n") != n or r.get("arm") not in ARMS: continue
        nm = r.get("name")
        sp = split_of(os.path.join(o, "split-%s.txt" % nm))
        rw = jload(os.path.join(o, "rows-%s.json" % nm)) or {}
        g = None
        try: g = list((rw.get("windows") or {}).values())[0].get("gt48")
        except (IndexError, AttributeError): g = None
        a0, a1 = jload(os.path.join(o, "fn-%s-0.json" % nm)) or {}, jload(os.path.join(o, "fn-%s-1.json" % nm)) or {}
        c0, c1 = a0.get("cap") or {}, a1.get("cap")
        gc = (c1.get("steps_gt48", 0) - c0.get("steps_gt48", 0)) if isinstance(c1, dict) else None
        out.append(dict(name=nm, arm=r["arm"], mb=r, split=sp, gt48=g, gt48cap=gc,
                        arll=cdelta(a0, a1, "arll", ("oneshot_launches", "twoshot_launches", "rows", "gather_launches", "declined",
                                                     "plain_launches", "ll_all_ranks")),
                        chain=cdelta(a0, a1, "chain", ("issued_chained", "played_chained", "no_barrier"))))
    return out


def arm_vals(bl, a):
    x = [b for b in bl if b["arm"] == a]
    sp = lambda k: [b["split"][k] if b["split"] else None for b in x]
    mb = lambda k: [b["mb"].get(k) for b in x]
    return dict(n=len(x), names=[b["name"] for b in x], nc_ms=sp("nc_ms"), nc_share=sp("nc_share"), dec_ms=sp("dec_ms"),
                agg=mb("agg_dec_tps"), share=mb("decode_share"), mbdec=mb("ms_decode_step"), err=mb("errors"),
                gt48=[b["gt48"] for b in x], gt48cap=[b["gt48cap"] for b in x], arll=[b["arll"] for b in x], chain=[b["chain"] for b in x])


def wire(v):
    """'LL a / chained b' over an arm's blocks (sums), or '-'"""
    ll = [d.get("ll_all_ranks") for d in v["arll"] if d]
    dec = [d.get("declined") for d in v["arll"] if d]
    ch = [(d.get("issued_chained") or 0) + (d.get("played_chained") or 0) for d in v["chain"] if d]
    nb = [d.get("no_barrier") for d in v["chain"] if d]
    return dict(ll=sum(ll) if ll else None, declined=sum(dec) if dec else None, chained=sum(ch) if ch else None,
                no_barrier=sum(nb) if nb else None, nblocks=v["n"])


def mb_summary(o, n, a, out):
    bl = blocks(o, n)
    if not bl:
        out.append("YLLSUM c%d: no block (mb.jsonl has no c%d row)" % (n, n)); return {}
    V = {x: arm_vals(bl, x) for x in ARMS}
    B = V["B"]
    if not B["n"]:
        out.append("YLLSUM c%d: VIRHE no B block (the reference)" % n); return {}
    nb, db, ab, hb, sb = mean(B["nc_ms"]), mean(B["dec_ms"]), mean(B["agg"]), mean(B["share"]), mean(B["nc_share"])
    ref = ""
    if n == 8:
        ref = " | int12 reference (prof1): newcomer %s ms a step = %s %%, decode %s ms a step, agg %s" % (f(a.ref_nc, 1), f(a.ref_share, 1), f(a.ref_dec), f(a.ref_agg, 1))
    out.append("YLLSUM c%d blocks in order: %s | B: agg %s (mean %s) tok/s, decode ms a step (step log) %s (mean %s), MB ms a decode step %s, decode_share %s, "
               "newcomer ms a step %s (mean %s), share %s %%%s" % (
                   n, " ".join(b["name"] for b in bl), sl(B["agg"]), f(ab, 1), sl(B["dec_ms"], 2), f(db), sl(B["mbdec"], 2), sl(B["share"], 3),
                   sl(B["nc_ms"]), f(nb, 1), sl(B["nc_share"]), ref))
    res = {}
    for x in ARMS[1:]:
        X = V[x]
        if not X["n"]: continue
        ax, dx, nx, hx, sx = mean(X["agg"]), mean(X["dec_ms"]), mean(X["nc_ms"]), mean(X["share"]), mean(X["nc_share"])
        w = wire(X)
        res[x] = dict(agg=pct(ax, ab), dec=pct(dx, db), nc=pct(nx, nb), ncb=nb, ncx=nx, aggx=ax, decx=dx, w=w)
        out.append("YLLSUM c%d %s vs B: agg %s (mean %s) tok/s: %s | decode ms a step (step log) %s (mean %s): %s (%s ms) | MB ms a decode step %s | "
                   "decode_share %s (mean %s vs %s) | newcomer ms a step %s (mean %s): %s (%s ms), share %s %% | request errors %s | wire: LL launches %s "
                   "(declined %s), chained launches %s (no barrier %s) over %d block(s)" % (
                       n, DESC[x], sl(X["agg"]), f(ax, 1), fp(pct(ax, ab)), sl(X["dec_ms"], 2), f(dx), fp(pct(dx, db)),
                       ("%+.2f" % (dx - db)) if (dx is not None and db is not None) else "-", sl(X["mbdec"], 2), sl(X["share"], 3), f(hx, 3), f(hb, 3),
                       sl(X["nc_ms"]), f(nx, 1), fp(pct(nx, nb)), ("%+.1f" % (nx - nb)) if (nx is not None and nb is not None) else "-",
                       sl(X["nc_share"]), sum(e or 0 for e in X["err"]), w["ll"], w["declined"], w["chained"], w["no_barrier"], w["nblocks"]))
    # the wire proof: B none, C / E LL, D / E chained
    wb = wire(B)
    bad = []
    if a.sw:
        if wb["ll"] or wb["chained"]: bad.append("B moved an LL / chained counter (%s / %s)" % (wb["ll"], wb["chained"]))
        for x in ("C", "E"):
            if V[x]["n"] and not (res.get(x, {}).get("w") or {}).get("ll"): bad.append("%s ran no LL launch" % x)
            if V[x]["n"] and (res.get(x, {}).get("w") or {}).get("declined"): bad.append("%s declined %s" % (x, res[x]["w"]["declined"]))
        for x in ("D", "E"):
            if V[x]["n"] and not (res.get(x, {}).get("w") or {}).get("chained"): bad.append("%s chained no launch" % x)
        for x in ("C", "P"):
            if V[x]["n"] and (res.get(x, {}).get("w") or {}).get("chained"): bad.append("%s chained launches" % x)
        if V["P"]["n"] and (res.get("P", {}).get("w") or {}).get("ll"): bad.append("P ran LL launches")
    out.append("YLLSUM c%d wire proof (GET /admin/fn4 arll / chain deltas a block): %s" % (
        n, ("VIRHE " + "; ".join(bad)) if bad else ("OK: B none; " + ", ".join("%s LL %s / chained %s" % (x, res[x]["w"]["ll"], res[x]["w"]["chained"]) for x in res)
                                                     if a.sw else "INFO (SW=0 mock: no counters)")))
    if n == 8 and V["P"]["n"]:
        d = res["P"]["dec"]
        if d is None: out.append("YLLSUM c8 A10 decode guard: VIRHE not read (B %s / P %s ms a decode step)" % (f(db), f(res["P"]["decx"])))
        elif d > a.dec_th: out.append("YLLSUM c8 A10 decode guard: VIRHE P's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(res["P"]["decx"]), f(db), fp(d), f(a.dec_th, 1)))
        else: out.append("YLLSUM c8 A10 decode guard: OK P's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(res["P"]["decx"]), f(db), fp(d), f(a.dec_th, 1)))
    if n == 16:
        g = [z for x in ARMS for z in V[x]["gt48"]]; gc = [z for x in ARMS for z in V[x]["gt48cap"]]
        badg = any(z for z in g if z) or any(z for z in gc if z)
        unread = any(z is None for z in g)
        out.append("YLLSUM c16 > 48-row decode steps (cap 48): step log %s, GET cap steps_gt48 %s%s" % (
            " ".join("%s %s" % (b["name"], b["gt48"]) for b in bl), " ".join("%s %s" % (b["name"], b["gt48cap"]) for b in bl),
            (" -- VIRHE > 48-row steps under the cap" if a.sw else " -- INFO (SW=0 mock: no row cap emulated)") if badg else
            (" -- VIRHE a block's step log not read" if (unread and a.sw) else " -- OK 0")))
    return res


def u1_summary(o, out):
    R = [r for r in jl(os.path.join(o, "blocks.jsonl")) if r.get("lever") == "U1" and r.get("arm") in ARMS]
    if not R:
        out.append("YLLSUM 1u: no read (blocks.jsonl has no U1 row)"); return {}
    t = lambda a: [(r.get("j") or {}).get("decode_tps") for r in R if r["arm"] == a]
    m = lambda a: [(r.get("j") or {}).get("ms_per_decode_step") for r in R if r["arm"] == a]
    tb = mean(t("B")); res = {}
    parts = ["B %s tok/s (mean %s), %s ms a step" % (sl(t("B"), 2), f(tb), sl(m("B"), 2))]
    for x in ARMS[1:]:
        if not t(x): continue
        res[x] = pct(mean(t(x)), tb)
        parts.append("%s %s (mean %s): %s, %s ms a step" % (x, sl(t(x), 2), f(mean(t(x))), fp(res[x]), sl(m(x), 2)))
    out.append("YLLSUM 1u essay confirm (one discarded + back-to-back reads an arm, stepprobe essay, T 0): " + " | ".join(parts))
    return res


def cold_summary(o, out):
    R = [r for r in jl(os.path.join(o, "cold.jsonl")) if r.get("arm") in ARMS]
    res = {}
    for dep in sorted({r.get("depth") for r in R if r.get("depth")}):
        x = [r for r in R if r.get("depth") == dep]
        ok = [r for r in x if r.get("status") == 200 and not r.get("cached")]
        t = lambda a: [r.get("ttft") for r in ok if r["arm"] == a]
        nd = lambda a: [r.get("nondecode_ms") for r in ok if r["arm"] == a]
        tb = mean(t("B"))
        line = "YLLSUM cold %s (deepq, cold after a reset, prompt %s tok; order %s): TTFT B %s s, non-decode ms a step B %s" % (
            dep, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})), "".join(r["arm"] for r in sorted(x, key=lambda r: r.get("i", 0))),
            sl(t("B"), 2), sl(nd("B"), 1))
        for a in ARMS[1:]:
            if not t(a): continue
            res[(dep, a)] = (tb, mean(t(a)), pct(mean(t(a)), tb))
            line += " | %s TTFT %s s: %s, non-decode ms %s" % (a, sl(t(a), 2), fp(pct(mean(t(a)), tb)), sl(nd(a), 1))
        notc = [r.get("name") for r in x if r.get("cached")]
        bad = [r.get("name") for r in x if r.get("status") != 200]
        line += (" -- NOT COLD (left out): %s" % ", ".join(notc)) if notc else ""
        line += (" -- VIRHE request errors: %s" % ", ".join(bad)) if bad else ""
        out.append(line)
    if not R: out.append("YLLSUM cold: no read (cold.jsonl has no row)")
    return res


def ident_summary(o, out):
    try: L = [l.split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    L = [(k, re.sub(r"^warm-up ", "", v.strip())) for k, v in (x if len(x) == 2 else (x[0], "") for x in L) if k.startswith(("yll-", "cold-"))]
    same = [k for k, v in L if v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v]
    diff = [k for k, v in L if k not in same]
    out.append("YLLSUM identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (len(same), len(L), (" -- VIRHE differ / not compared: %s" % ", ".join(diff)) if diff else ""))
    return len(same), len(L)


def stag_summary(o, out):
    """the staggered-arrival checks (identity.txt lines yll-ld<ARM>-vs-a1-<ARM>): strict, a difference is a VIRHE"""
    try: L = [l.strip().split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.startswith("yll-ld")]
    except OSError: L = []
    if not L:
        out.append("YLLSUM staggered (loadid: decode streams + a fresh prefill a probe, strict vs the arm's a1): not run"); return
    bad = [k for k, v in (x if len(x) == 2 else (x[0], "") for x in L) if not (v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v)]
    out.append("YLLSUM staggered (loadid: decode streams + a fresh prefill a probe, strict vs the arm's a1): %s%s" % (
        " | ".join("%s %s" % (x[0][4:], (x[1] if len(x) == 2 else "")[:80]) for x in L), (" -- VIRHE differ / not compared: %s" % ", ".join(bad)) if bad else " -- OK"))


def a10_rule(s8, cs, out, depth=136000):
    p = (s8 or {}).get("P") or {}
    c = cs.get((depth, "P"))
    ttft_ok = c is not None and c[2] is not None and c[2] <= -1.0
    nc_ok = p.get("ncb") is not None and p.get("ncx") is not None and (p["ncb"] - p["ncx"]) >= 15.0
    known = c is not None or p.get("ncx") is not None
    if not known: return   # no arm P read here (12.11.: A10 rides on yd2m): no A10 line
    out.append("YLLSUM A10 rule (joins the stack if %dk TTFT >= 1 %% faster OR c8 newcomer step >= 15 ms faster): %s | %dk TTFT B %s s -> P %s s: %s | "
               "c8 newcomer step B %s ms -> P %s ms: %s ms (%s)" % (
                   depth // 1000, ("PASS" if (ttft_ok or nc_ok) else "NOT") if known else "NOT READ", depth // 1000, f(c[0] if c else None),
                   f(c[1] if c else None), fp(c[2] if c else None), f(p.get("ncb"), 1), f(p.get("ncx"), 1),
                   ("%+.1f" % (p["ncx"] - p["ncb"])) if (p.get("ncb") is not None and p.get("ncx") is not None) else "-", fp(p.get("nc"))))


def run(argv):
    o = argv[0]
    a = type("A", (), {})()
    kv = dict(zip(argv[1::2], argv[2::2]))
    a.dec_th = float(kv.get("--dec-th", 1.0)); a.sw = kv.get("--sw", "1") == "1"
    a.ref_nc, a.ref_share = float(kv.get("--ref-nc", 538.2)), float(kv.get("--ref-share", 20.0))
    a.ref_dec, a.ref_agg = float(kv.get("--ref-dec", 36.18)), float(kv.get("--ref-agg", 184.0))
    out = []
    s8 = mb_summary(o, 8, a, out)
    s16 = mb_summary(o, 16, a, out)
    u1 = u1_summary(o, out)
    cs = cold_summary(o, out)
    a10_rule(s8, cs, out)
    stag_summary(o, out)
    ni, nt = ident_summary(o, out)
    out.append("YLLSUM headline (vs B = int12): c8 agg %s | c8 decode step %s | 1u %s | c16 agg %s | identity %d / %d" % (
        ", ".join("%s %s" % (x, fp(v.get("agg"))) for x, v in s8.items()) or "-",
        ", ".join("%s %s" % (x, fp(v.get("dec"))) for x, v in s8.items()) or "-",
        ", ".join("%s %s" % (x, fp(v)) for x, v in u1.items()) or "-",
        ", ".join("%s %s" % (x, fp(v.get("agg"))) for x, v in s16.items()) or "-", ni, nt))
    for l in out: print(l)
    return 0


def best_arm(o, among="CDE"):
    """the arm of `among` with the highest mean c8 agg (mb.jsonl), for the c16 block; C when none was read"""
    R = [r for r in jl(os.path.join(o, "mb.jsonl")) if r.get("n") == 8 and r.get("arm") in among and r.get("agg_dec_tps") is not None]
    m = {x: mean([r["agg_dec_tps"] for r in R if r["arm"] == x]) for x in among}
    m = {x: v for x, v in m.items() if v is not None}
    return max(m, key=m.get) if m else among[0]


def selftest():
    d = tempfile.mkdtemp(prefix="yllsum-")
    mb = []
    seq = [("B", 540.0, 36.2, 184.0), ("C", 538.0, 35.6, 186.0), ("D", 541.0, 36.3, 183.5), ("E", 537.0, 35.7, 186.2), ("P", 515.0, 36.2, 185.0),
           ("P", 517.0, 36.25, 185.2), ("E", 539.0, 35.7, 186.4), ("D", 540.0, 36.2, 183.9), ("C", 538.5, 35.5, 186.1), ("B", 538.0, 36.2, 184.4)]
    for i, (a, nc, dec, agg) in enumerate(seq, 1):
        nm = "c8-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=8, arm=a, sched="S", agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec + 0.1, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms = 0.0 %% | agg 184.0 tok/s\n" % (nm, dec, nc))
        json.dump({"windows": {nm: {"gt48": 0}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
        ll = 900 if a in "CE" else 0
        ch = 300 if a in "DE" else 0
        json.dump({"cap": {"steps_gt48": 0}, "arll": {"ll_all_ranks": 100, "declined": 0}, "chain": {"issued_chained": 7, "played_chained": 0, "no_barrier": 0}},
                  open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 0}, "arll": {"ll_all_ranks": 100 + ll, "declined": 0}, "chain": {"issued_chained": 7, "played_chained": ch, "no_barrier": ch // 2}},
                  open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    for i, a in enumerate("BE", 1):
        nm = "c16-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=16, arm=a, sched="S", agg_dec_tps=300.0 + i, decode_share=0.9, ms_decode_step=55.0, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 500 steps 30000 ms = 85.7 %%, 55.00 ms a step, 16.80 rows a step, 16.6 tokens a step | (b) newcomer / MIXED 0 steps "
            "0 ms = 0.0 %%, - rows a step, - ms a step | (c) host gaps 0 ms\n" % nm)
        json.dump({"windows": {nm: {"gt48": 0 if a == "B" else 2}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 3}, "arll": {"ll_all_ranks": 0}, "chain": {"issued_chained": 0, "played_chained": 0, "no_barrier": 0}},
                  open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 3}, "arll": {"ll_all_ranks": 50 if a == "E" else 0}, "chain": {"issued_chained": 0, "played_chained": 9 if a == "E" else 0,
                   "no_barrier": 0}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        for i, (a, t) in enumerate([("B", 91.0), ("B", 91.2), ("C", 92.0), ("C", 92.4), ("D", 90.0), ("E", 92.1)]):
            fo.write(json.dumps(dict(i=i, lever="U1", arm=a, j=dict(decode_tps=t, ms_per_decode_step=17.3))) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        for i, (a, t) in enumerate([("B", 24.2), ("P", 23.8)], 1):
            fo.write(json.dumps(dict(name="cold-136000-%d-%s" % (i, a), depth=136000, arm=a, i=i, ttft=t, prompt_tokens=136650, cached=0, status=200, nondecode_ms=600.0)) + "\n")
    open(os.path.join(d, "identity.txt"), "w").write("yll-n5 warm-up IDENTICAL: 3 answers\nyll-a1-C-vs-ref12-a1 IDENTICAL: 20 of 20 answers\nyll-gs16-E-vs-a1-E DIFFERENT: q3\n"
                                                     "cold-136000-2-P-vs-cold-136000-1-B IDENTICAL (900 chars)\nyll-ldB-vs-a1-B IDENTICAL: 20 of 20 answers\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--dec-th", "1.0"])
    t = buf.getvalue()
    why = []
    if "YLLSUM c8 C = ar_ll (LL to 48 rows) vs B: agg 186.0/186.1 (mean 186.1) tok/s: +1.00 %" not in t: why.append("c8 C agg")
    if "decode ms a step (step log) 35.60/35.50 (mean 35.55): -1.80 % (-0.65 ms)" not in t: why.append("c8 C dec")
    if "wire proof (GET /admin/fn4 arll / chain deltas a block): OK" not in t: why.append("wire proof ok")
    if "YLLSUM c8 A10 decode guard: OK" not in t: why.append("A10 guard")
    if "YLLSUM A10 rule" not in t or ": PASS |" not in t: why.append("A10 rule PASS (newcomer -23 ms)")
    if "-- VIRHE > 48-row steps under the cap" not in t: why.append("c16 gt48 caught")
    if "YLLSUM c16 wire proof (GET /admin/fn4 arll / chain deltas a block): OK" not in t: why.append("c16 wire proof")
    if "4 of 5 IDENTICAL -- VIRHE differ / not compared: yll-gs16-E-vs-a1-E" not in t: why.append("identity")
    if "YLLSUM staggered (loadid: decode streams + a fresh prefill a probe, strict vs the arm's a1): ldB-vs-a1-B IDENTICAL: 20 of 20 answers -- OK" not in t: why.append("staggered")
    if "C 92.00/92.40 (mean 92.20): +1.21 %" not in t: why.append("1u")
    if best_arm(d) != "E": why.append("best arm E")
    # the wire proof catches a C block without LL
    json.dump({"cap": {"steps_gt48": 0}, "arll": {"ll_all_ranks": 100, "declined": 0}, "chain": {"issued_chained": 7, "played_chained": 0, "no_barrier": 0}},
              open(os.path.join(d, "fn-c8-2-S-C-1.json"), "w"))
    json.dump({"cap": {"steps_gt48": 0}, "arll": {"ll_all_ranks": 100, "declined": 0}, "chain": {"issued_chained": 7, "played_chained": 0, "no_barrier": 0}},
              open(os.path.join(d, "fn-c8-9-S-C-1.json"), "w"))
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): run([d, "--dec-th", "1.0"])
    if "wire proof (GET /admin/fn4 arll / chain deltas a block): VIRHE C ran no LL launch" not in buf2.getvalue(): why.append("wire proof virhe")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: c8 per arm vs B, the wire proof (OK and VIRHE), A10's decode guard and rule, c16 > 48 rows caught, 1u, cold, identity, staggered, best arm")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": sys.exit(selftest())
    if len(sys.argv) > 2 and sys.argv[1] == "best": print(best_arm(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CDE")); sys.exit(0)
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    sys.exit(run(sys.argv[1:]))
