#!/usr/bin/env python3
"""ydsum.py (12.11.2026, queue-ar/dsmain/yd2m, the A4 builder for ds-main-port) -- the A4 screen's C-vs-B summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m.sh, body yd2m/dsm-yd2m.sh; = xpush/xpsum.py with A4's arms and readings). Standard library
only; reads the body's output dir:

  ydsum.py OUTDIR [--dec-th 1.0] [--sw 1] [--ref-nc 538.2] [--ref-share 20.0] [--ref-dec 36.18] [--ref-agg 184.0]
      OUTDIR/mb.jsonl (mbx's rows: n, arm, agg_dec_tps, decode_share, ms_decode_step), OUTDIR/split-<block>.txt (p12win.py split's
      P12SPLIT line: (a) decode steps' ms a step and rows a step, (b) newcomer / MIXED steps), OUTDIR/rows-<block>.json (gt48),
      OUTDIR/fn-<block>-0/1.json (GET cap steps_gt48), OUTDIR/blocks.jsonl (the 1u essay reads: lever U1 / U1W, arm), OUTDIR/p0-perf.txt
      (the P0 bench's M16GATE / M16EST lines), OUTDIR/identity.txt. Arms B (gg_gemv_m16 0 = int12's D2 forms) and C (gg_gemv_m16 1 = the
      P0 table's A4 forms); every reading as the values, B as the reference, C vs B in %. Lines "YDSUM ...": c8 / c16 (decode ms a
      step = the lever's target, rows a step, agg, decode_share, newcomer ms a step), the c8 no-regression guard (C's decode-step ms
      not above B's by more than --dec-th %), the 1u essay reads (decode tok/s, ms a decode step), the P0 gate (bench), identity, one
      headline.
  ydsum.py selftest"""
import glob, json, os, re, statistics, sys, tempfile

DEC = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) ms a step, ([\d.]+|-) rows a step")
NEW = re.compile(r"\(b\) newcomer / MIXED (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) rows a step, ([\d.]+|-) ms a step")


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
    return dict(dec_steps=int(d.group(1)), dec_ms=fnum(d.group(4)), dec_rows=fnum(d.group(5)), nc_steps=int(n.group(1)),
                nc_share=fnum(n.group(3)), nc_rows=fnum(n.group(4)), nc_ms=fnum(n.group(5)))


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def pct(c, b):
    return 100.0 * (c / b - 1.0) if (c is not None and b) else None


def f(v, nd=2):
    return ("%." + str(nd) + "f") % v if v is not None else "-"


def fp(v):
    return "%+.2f %%" % v if v is not None else "-"


def blocks(o, n):
    """the mb blocks of c<n>: [(name, arm, mb row, split, gt48 step log, gt48 cap)] in run order"""
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        if r.get("n") != n or r.get("arm") not in ("B", "C", "P"): continue
        nm = r.get("name")
        sp = split_of(os.path.join(o, "split-%s.txt" % nm))
        rw = jload(os.path.join(o, "rows-%s.json" % nm)) or {}
        g = None
        try: g = list((rw.get("windows") or {}).values())[0].get("gt48")
        except (IndexError, AttributeError): g = None
        a0, a1 = jload(os.path.join(o, "fn-%s-0.json" % nm)) or {}, jload(os.path.join(o, "fn-%s-1.json" % nm)) or {}
        c0, c1 = a0.get("cap") or {}, a1.get("cap")
        gc = (c1.get("steps_gt48", 0) - c0.get("steps_gt48", 0)) if isinstance(c1, dict) else None
        out.append((nm, r["arm"], r, sp, g, gc))
    return out


def arm_line(bl, a):
    x = [b for b in bl if b[1] == a]
    v = lambda k: [b[3][k] if b[3] else None for b in x]
    m = lambda k: [b[2].get(k) for b in x]
    return dict(n=len(x), names=[b[0] for b in x], nc_ms=v("nc_ms"), nc_share=v("nc_share"), nc_steps=v("nc_steps"), dec_ms=v("dec_ms"), dec_rows=v("dec_rows"),
                agg=m("agg_dec_tps"), share=m("decode_share"), mbdec=m("ms_decode_step"), err=m("errors"),
                gt48=[b[4] for b in x], gt48cap=[b[5] for b in x])


def sl(v, nd=1):
    return "/".join(f(z, nd) for z in v) or "-"


def mb_summary(o, n, a, out):
    bl = blocks(o, n)
    if not bl:
        out.append("YDSUM c%d: no block (mb.jsonl has no c%d B / C row)" % (n, n)); return None
    B, C = arm_line(bl, "B"), arm_line(bl, "C")
    nb, nc = mean(B["nc_ms"]), mean(C["nc_ms"])
    db, dc = mean(B["dec_ms"]), mean(C["dec_ms"])
    ab, ac = mean(B["agg"]), mean(C["agg"])
    sb, sc = mean(B["nc_share"]), mean(C["nc_share"])
    hb, hc = mean(B["share"]), mean(C["share"])
    ref = ""
    if n == 8:
        ref = " | int12 reference (prof1): decode %s ms a step at 8.47 rows, newcomer %s ms a step = %s %%, agg %s" % (f(a.ref_dec), f(a.ref_nc, 1), f(a.ref_share, 1), f(a.ref_agg, 1))
    if not C["n"]:
        out.append("YDSUM c%d: no C block (A4 not screened: its gate stopped A4 after P0, or the order has no C) -- B blocks %s: decode ms a step %s, "
                   "newcomer ms a step %s, agg %s" % (n, " ".join(B["names"]), sl(B["dec_ms"], 2), sl(B["nc_ms"]), sl(B["agg"])))
    else:
      out.append("YDSUM c%d decode steps (the lever's target; step log, B = gg_gemv_m16 0, C = gg_gemv_m16 1; blocks %s): ms a step B %s (mean %s) -> C %s (mean %s): %s "
               "(%s ms); rows a step B %s / C %s%s" % (
                   n, " ".join(b[0] for b in bl), sl(B["dec_ms"], 2), f(db), sl(C["dec_ms"], 2), f(dc), fp(pct(dc, db)),
                   ("%+.2f" % (dc - db)) if (db is not None and dc is not None) else "-", sl(B["dec_rows"]), sl(C["dec_rows"]), ref))
      out.append("YDSUM c%d throughput: agg B %s (mean %s) -> C %s (mean %s) tok/s: %s; decode_share B %s -> C %s; MB ms a decode step B %s -> C %s; "
               "newcomer / MIXED ms a step B %s (mean %s) -> C %s (mean %s): %s, share B %s %% -> C %s %%; request errors B %s / C %s" % (
                   n, sl(B["agg"]), f(ab, 1), sl(C["agg"]), f(ac, 1), fp(pct(ac, ab)), sl(B["share"], 3), sl(C["share"], 3), sl(B["mbdec"]), sl(C["mbdec"]),
                   sl(B["nc_ms"]), f(nb, 1), sl(C["nc_ms"]), f(nc, 1), fp(pct(nc, nb)), sl(B["nc_share"]), sl(C["nc_share"]),
                   sum(e or 0 for e in B["err"]), sum(e or 0 for e in C["err"])))
    P = arm_line(bl, "P")
    pres = None
    if P["n"]:
        npm, dpm, apm = mean(P["nc_ms"]), mean(P["dec_ms"]), mean(P["agg"])
        out.append("YDSUM c%d A10 arm P (exl3_x3d_pf 1, gg_gemv_m16 0) vs B: newcomer / MIXED ms a step B %s (mean %s) -> P %s (mean %s): %s (%s ms); share B %s %% -> "
                   "P %s %%; decode ms a step B %s -> P %s (mean %s): %s; agg B %s -> P %s (mean %s): %s; blocks %s" % (
                       n, sl(B["nc_ms"]), f(nb, 1), sl(P["nc_ms"]), f(npm, 1), fp(pct(npm, nb)), ("%+.1f" % (npm - nb)) if (npm is not None and nb is not None) else "-",
                       sl(B["nc_share"]), sl(P["nc_share"]), sl(B["dec_ms"], 2), sl(P["dec_ms"], 2), f(dpm), fp(pct(dpm, db)), sl(B["agg"]), sl(P["agg"]), f(apm, 1),
                       fp(pct(apm, ab)), " ".join(P["names"])))
        pres = dict(ncb=nb, ncp=npm, nc=pct(npm, nb), dec=pct(dpm, db), agg=pct(apm, ab))
    if n == 8 and C["n"]:
        d = pct(dc, db)
        if d is None: out.append("YDSUM c8 no-regression guard: VIRHE not read (B %s / C %s ms a decode step)" % (f(db), f(dc)))
        elif d > a.dec_th: out.append("YDSUM c8 no-regression guard: VIRHE C's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(dc), f(db), fp(d), f(a.dec_th, 1)))
        else: out.append("YDSUM c8 no-regression guard: OK C's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(dc), f(db), fp(d), f(a.dec_th, 1)))
    if n == 16 and C["n"]:
        g = [z for z in B["gt48"] + C["gt48"]]; gc = [z for z in B["gt48cap"] + C["gt48cap"]]
        bad = any(z for z in g if z) or any(z for z in gc if z)
        unread = any(z is None for z in g)
        out.append("YDSUM c16 > 48-row decode steps (cap 48): step log B %s / C %s, GET cap steps_gt48 B %s / C %s%s" % (
            "/".join(str(z) for z in B["gt48"]) or "-", "/".join(str(z) for z in C["gt48"]) or "-", "/".join(str(z) for z in B["gt48cap"]) or "-",
            "/".join(str(z) for z in C["gt48cap"]) or "-",
            (" -- VIRHE > 48-row steps under the cap" if a.sw else " -- INFO (SW=0 mock: no row cap emulated)") if bad else
            (" -- VIRHE a block's step log not read" if (unread and a.sw) else " -- OK 0")))
    return dict(nc=pct(nc, nb), agg=pct(ac, ab), dec=pct(dc, db), ncb=nb, ncc=nc, db=db, dc=dc, p=pres)


def u1_summary(o, out):
    """the 1u essay reads (blocks.jsonl lever U1, the warm-up U1W left out): decode tok/s, ms a decode step, stream MB a step, by arm"""
    R = [r for r in jl(os.path.join(o, "blocks.jsonl")) if r.get("lever") in ("U1", "U1W") and r.get("arm") in ("B", "C")]
    if not R:
        out.append("YDSUM 1u: no read (blocks.jsonl has no U1 row)"); return None
    g = lambda a, k: [(r.get("j") or {}).get(k) for r in R if r["arm"] == a and r["lever"] == "U1"]
    s = lambda a: [(r.get("st") or {}).get("stream_mb_step") for r in R if r["arm"] == a and r["lever"] == "U1"]
    tb, tc = mean(g("B", "decode_tps")), mean(g("C", "decode_tps"))
    mb_, mc = mean(g("B", "ms_per_decode_step")), mean(g("C", "ms_per_decode_step"))
    out.append("YDSUM 1u essay (stepprobe essay 512, T 0; a discarded warm-up read per arm; order %s): decode tok/s B %s (mean %s) -> C %s (mean %s): %s; "
               "ms a decode step B %s -> C %s: %s; tokens a step B %s / C %s; stream MB a step B %s / C %s (a delta is read only at matched stream MB)" % (
                   "".join(r["arm"] + ("w" if r["lever"] == "U1W" else "") for r in R), sl(g("B", "decode_tps"), 2), f(tb), sl(g("C", "decode_tps"), 2), f(tc), fp(pct(tc, tb)),
                   sl(g("B", "ms_per_decode_step"), 2), sl(g("C", "ms_per_decode_step"), 2), fp(pct(mc, mb_)), sl(g("B", "tokens_per_step")), sl(g("C", "tokens_per_step")),
                   sl(s("B")), sl(s("C"))))
    return pct(tc, tb)


def cold_summary(o, out):
    """lever A10's cold reads (cold.jsonl: B then P at each depth): TTFT, prefill tok/s, non-decode ms a step; P vs B"""
    R = [r for r in jl(os.path.join(o, "cold.jsonl")) if r.get("arm") in ("B", "P")]
    res = {}
    for dep in sorted({r.get("depth") for r in R if r.get("depth")}):
        x = [r for r in R if r.get("depth") == dep]
        ok = [r for r in x if r.get("status") == 200 and not r.get("cached")]
        t = lambda a: [r.get("ttft") for r in ok if r["arm"] == a]
        nd = lambda a: [r.get("nondecode_ms") for r in ok if r["arm"] == a]
        tb, tp = mean(t("B")), mean(t("P"))
        res[dep] = (tb, tp, pct(tp, tb))
        bad = [r.get("name") for r in x if r.get("status") != 200]
        notc = [r.get("name") for r in x if r.get("cached")]
        out.append("YDSUM A10 cold %s (deepq after a prefix-cache reset, prompt %s tok, B then P): TTFT B %s s -> P %s s: %s | non-decode ms a step B %s -> P %s: %s%s%s" % (
            dep, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})), sl(t("B"), 2), sl(t("P"), 2), fp(pct(tp, tb)), sl(nd("B"), 1), sl(nd("P"), 1),
            fp(pct(mean(nd("P")), mean(nd("B")))), (" -- NOT COLD (left out): %s" % ", ".join(notc)) if notc else "", (" -- VIRHE request errors: %s" % ", ".join(bad)) if bad else ""))
    return res


def a10_verdict(s8, cold, a, out):
    p = (s8 or {}).get("p") or {}
    nb, np_ = p.get("ncb"), p.get("ncp")
    dnc = (nb - np_) if (nb is not None and np_ is not None) else None
    c = cold.get(136000) or (cold[max(cold)] if cold else None)
    tt = c[2] if c else None
    if dnc is None and tt is None:
        out.append("YDSUM A10 verdict: NOT READ (no P block and no P cold read: A10's P0 failed or YDM_A10=0)"); return None
    ok = (tt is not None and tt <= -a.a10_ttft) or (dnc is not None and dnc >= a.a10_nc)
    out.append("YDSUM A10 verdict (A10 joins the stack if 136k TTFT >= %s %% faster OR newcomer step >= %s ms faster): %s -- 136k TTFT B %s s -> P %s s (%s); "
               "c8 newcomer step B %s ms -> P %s ms (%s ms, %s)" % (
                   f(a.a10_ttft, 1), f(a.a10_nc, 1), "PASS" if ok else "NOT", f(c[0] if c else None), f(c[1] if c else None), fp(tt), f(nb, 1), f(np_, 1),
                   ("%+.1f" % (np_ - nb)) if dnc is not None else "-", fp(p.get("nc"))))
    return ok


def gate_summary(o, out, sw=True):
    """the P0 bench (r4d_selftest --perfm16 on one card): M16EST + the M16GATE lines of the c8 profile's shapes at M 8 / 16"""
    try: L = open(os.path.join(o, "p0-perf.txt")).read().splitlines()
    except OSError: L = []
    est = next((l for l in L if l.startswith("M16EST")), None)
    gates = [l for l in L if l.startswith("M16GATE")]
    nform = sum(1 for l in L if l.startswith("M16FORM "))
    out.append("YDSUM P0 gate (bench, one card, kernel us a call, two passes): %s | %d table entries%s" % (
        est[7:].strip() if est else "no M16EST line", nform, "" if est else (" -- VIRHE the bench did not finish" if sw else " -- INFO (SW=0 mock: no GPU bench)")))
    for l in gates: out.append("YDSUM P0 " + l)
    return nform


def ident_summary(o, out):
    try: L = [l.split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    # identw's line is "<tag> warm-up <result>" (the warm-up answers vs the stock reference): its result after "warm-up "
    L = [(k, re.sub(r"^warm-up ", "", v.strip())) for k, v in (x if len(x) == 2 else (x[0], "") for x in L) if k.startswith("ydm-")]
    same = [k for k, v in L if v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v]
    diff = [k for k, v in L if k not in same]
    out.append("YDSUM identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (len(same), len(L), (" -- VIRHE differ / not compared: %s" % ", ".join(diff)) if diff else ""))
    return len(same), len(L)


def run(argv):
    o = argv[0]
    a = type("A", (), {})()
    kv = dict(zip(argv[1::2], argv[2::2]))
    a.dec_th = float(kv.get("--dec-th", 1.0)); a.sw = kv.get("--sw", "1") == "1"
    a.ref_nc, a.ref_share = float(kv.get("--ref-nc", 538.2)), float(kv.get("--ref-share", 20.0))
    a.ref_dec, a.ref_agg = float(kv.get("--ref-dec", 36.18)), float(kv.get("--ref-agg", 184.0))
    a.a10_nc, a.a10_ttft = float(kv.get("--a10-nc-ms", 15)), float(kv.get("--a10-ttft-pct", 1))
    out = []
    s8 = mb_summary(o, 8, a, out)
    s16 = mb_summary(o, 16, a, out)
    u1 = u1_summary(o, out)
    cold = cold_summary(o, out)
    av = a10_verdict(s8, cold, a, out)
    nf = gate_summary(o, out, a.sw)
    ni, nt = ident_summary(o, out)
    out.append("YDSUM headline (C = A4 forms vs B = int12's D2 forms): c8 decode step %s (%s -> %s ms), c8 agg %s | c16 decode step %s, c16 agg %s | 1u %s | "
               "table %d entries | A10 %s | identity %d / %d" % (
                   fp((s8 or {}).get("dec")), f((s8 or {}).get("db")), f((s8 or {}).get("dc")), fp((s8 or {}).get("agg")), fp((s16 or {}).get("dec")),
                   fp((s16 or {}).get("agg")), fp(u1), nf, "-" if av is None else ("PASS" if av else "NOT"), ni, nt))
    for l in out: print(l)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="ydsum-")
    mb = []
    for i, (a, dec, agg) in enumerate([("B", 36.20, 184.0), ("C", 35.10, 188.0), ("C", 35.30, 187.4), ("B", 36.10, 184.4)], 1):
        nm = "c8-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=8, arm=a, sched="S", agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec + 0.1, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, 538.0 ms a step | (c) host gaps 0 ms = 0.0 %% (idle > 3 s 0 + window edges 0) | agg 184.0 tok/s\n" % (nm, dec))
        json.dump({"windows": {nm: {"gt48": 0}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
    for i, (a, nc) in ((5, ("P", 520.0)), (6, ("P", 518.0))):
        nm = "c8-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=8, arm=a, sched="S", agg_dec_tps=186.0, decode_share=0.81, ms_decode_step=36.2, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, 36.20 ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 19.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms\n" % (nm, nc))
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        fo.write(json.dumps(dict(name="cold-136000-1-B", depth=136000, arm="B", i=1, ttft=30.0, prompt_tokens=136100, cached=0, status=200, nondecode_ms=900.0)) + "\n")
        fo.write(json.dumps(dict(name="cold-136000-2-P", depth=136000, arm="P", i=2, ttft=29.9, prompt_tokens=136100, cached=0, status=200, nondecode_ms=890.0)) + "\n")
    for i, a in enumerate("BC", 1):
        nm = "c16-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=16, arm=a, sched="S", agg_dec_tps=300.0 + i, decode_share=0.9, ms_decode_step=55.0, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 500 steps 30000 ms = 85.7 %%, %.2f ms a step, 16.80 rows a step, 16.6 tokens a step | (b) newcomer / MIXED 0 steps "
            "0 ms = 0.0 %%, - rows a step, - ms a step | (c) host gaps 0 ms\n" % (nm, 55.0 if a == "B" else 53.0))
        json.dump({"windows": {nm: {"gt48": 0 if a == "B" else 2}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 5}}, open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 5}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        for lv, a, t in (("U1W", "C", 80.0), ("U1", "C", 91.0), ("U1", "C", 91.2), ("U1", "C", 91.1), ("U1W", "B", 90.0), ("U1", "B", 91.0), ("U1", "B", 90.8), ("U1", "B", 91.0)):
            fo.write(json.dumps(dict(lever=lv, arm=a, j=dict(prompt="essay", decode_tps=t, ms_per_decode_step=17.3, tokens_per_step=1.58), st=dict(stream_mb_step=20.0))) + "\n")
    open(os.path.join(d, "p0-perf.txt"), "w").write("M16GATE q6_K attn_q_b(fold a) M=8: d2 38.3 us -> 30.0 us (H4R2wfit-xp), -21.7 %; c8 calls a step 27\n"
                                                    "M16EST rank 0 (bench us): c8 at M 8 d2 8000 us -> 7000 us a step (1000 us saved); c16 at M 16 d2 14000 us -> 12500 us (1500 us saved)\n"
                                                    "M16FORM q6_K 8192 1280 8 8 0 4 2 0 -1 1\n")
    open(os.path.join(d, "identity.txt"), "w").write("ydm-n5 warm-up IDENTICAL: 3 answers\nydm-a1-C-vs-a1-B IDENTICAL: 20 of 20 answers\nydm-gs16-C-vs-gs16-B DIFFERENT: q3\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--dec-th", "1.0"])
    t = buf.getvalue()
    why = []
    if "ms a step B 36.20/36.10 (mean 36.15) -> C 35.10/35.30 (mean 35.20): -2.63 % (-0.95 ms)" not in t: why.append("c8 decode")
    if "YDSUM c8 no-regression guard: OK" not in t: why.append("guard ok")
    if "-- VIRHE > 48-row steps under the cap" not in t: why.append("c16 gt48 caught")
    if "decode tok/s B 91.00/90.80/91.00 (mean 90.93) -> C 91.00/91.20/91.10 (mean 91.10): +0.18 %" not in t: why.append("1u")
    if "1000 us saved" not in t or "1 table entries" not in t: why.append("gate")
    if "2 of 3 IDENTICAL -- VIRHE differ / not compared: ydm-gs16-C-vs-gs16-B" not in t: why.append("identity")
    if "c8 decode step -2.63 % (36.15 -> 35.20 ms)" not in t: why.append("headline")
    if "newcomer / MIXED ms a step B 538.0/538.0 (mean 538.0) -> P 520.0/518.0 (mean 519.0): -3.53 % (-19.0 ms)" not in t: why.append("A10 c8")
    if "YDSUM A10 verdict (A10 joins the stack if 136k TTFT >= 1.0 % faster OR newcomer step >= 15.0 ms faster): PASS -- 136k TTFT B 30.00 s -> P 29.90 s (-0.33 %)" not in t: why.append("A10 verdict")
    if "TTFT B 30.00 s -> P 29.90 s: -0.33 %" not in t: why.append("A10 cold")
    for nm, dec in (("c8-2-S-C", 36.9), ("c8-3-S-C", 36.9)):
        p = os.path.join(d, "split-%s.txt" % nm); s = open(p).read(); open(p, "w").write(re.sub(r"= 80.0 %, [\d.]+ ms a step", "= 80.0 %%, %.2f ms a step" % dec, s))
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3): run([d, "--dec-th", "1.0"])
    if "YDSUM c8 no-regression guard: VIRHE" not in buf3.getvalue(): why.append("guard virhe")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    d2 = tempfile.mkdtemp(prefix="ydsum-a10-")
    with open(os.path.join(d2, "mb.jsonl"), "w") as fo:
        for i, (a, nc) in enumerate([("P", 520.0), ("B", 540.0), ("B", 538.0), ("P", 518.0)], 1):
            nm = "c8-%d-S-%s" % (i, a)
            fo.write(json.dumps(dict(name=nm, n=8, arm=a, sched="S", agg_dec_tps=184.0, decode_share=0.8, ms_decode_step=36.2, errors=0)) + "\n")
            open(os.path.join(d2, "split-%s.txt" % nm), "w").write(
                "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, 36.20 ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
                "6997 ms = 20.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms\n" % (nm, nc))
    buf4 = io.StringIO()
    with contextlib.redirect_stdout(buf4): run([d2, "--sw", "1"])
    t4 = buf4.getvalue()
    if "YDSUM c8: no C block" not in t4 or "no-regression guard" in t4 or "A10 verdict (A10 joins the stack if 136k TTFT >= 1.0 % faster OR newcomer step >= 15.0 ms faster): PASS" not in t4:
        why.append("A10 alone")
        print(t4)
    if why:
        print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: A10 alone (no C block: no guard, the A10 rule line); c8 decode C vs B / no-regression guard (OK and VIRHE), c16 > 48 rows caught, 1u by arm, the P0 gate lines, identity count, headline")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": sys.exit(selftest())
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    sys.exit(run(sys.argv[1:]))
