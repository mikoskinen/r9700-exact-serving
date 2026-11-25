#!/usr/bin/env python3
"""xpsum.py (12.11.2026, queue-ar/dsmain/xpush, the XPUSH builder for ds-main-port) -- the XPUSH screen's C-vs-B summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-xpush.sh, body xpush/dsm-xpush.sh). Standard library only; reads the body's output dir:

  xpsum.py OUTDIR [--dec-th 1.0] [--sw 1] [--ref-nc 538.2] [--ref-share 20.0] [--ref-dec 36.18] [--ref-agg 184.0]
      OUTDIR/mb.jsonl (mbx's rows: n, arm, agg_dec_tps, decode_share, ms_decode_step), OUTDIR/split-<block>.txt (p12win.py split's
      P12SPLIT line: (a) decode steps' ms a step, (b) newcomer / MIXED steps' count, share, ms a step), OUTDIR/rows-<block>.json
      (p12win.py rows: gt48), OUTDIR/fn-<block>-0/1.json (GET cap steps_gt48), OUTDIR/cold.jsonl (coldx: depth, arm, TTFT),
      OUTDIR/identity.txt. Arms B (ar_xpush 0) and C (ar_xpush 1); every reading as the values, B as the reference, C vs B in %.
      Lines "XPSUM ...": c8 / c16 (newcomer ms a step and share, decode ms a step, agg, decode_share; c16: > 48-row steps, which
      must be 0 under the cap), the decode guard (C's c8 decode-step ms vs B's: a VIRHE past --dec-th %), cold per depth (TTFT,
      prefill tok/s, non-decode ms a step, cached), identity (IDENTICAL / compared), one headline.
  xpsum.py selftest"""
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


def blocks(o, n):
    """the mb blocks of c<n>: [(name, arm, mb row, split, gt48 step log, gt48 cap)] in run order"""
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        if r.get("n") != n or r.get("arm") not in ("B", "C"): continue
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
    return dict(n=len(x), names=[b[0] for b in x], nc_ms=v("nc_ms"), nc_share=v("nc_share"), nc_steps=v("nc_steps"), dec_ms=v("dec_ms"),
                agg=m("agg_dec_tps"), share=m("decode_share"), mbdec=m("ms_decode_step"), err=m("errors"),
                gt48=[b[4] for b in x], gt48cap=[b[5] for b in x])


def sl(v, nd=1):
    return "/".join(f(z, nd) for z in v) or "-"


def mb_summary(o, n, a, out):
    bl = blocks(o, n)
    if not bl:
        out.append("XPSUM c%d: no block (mb.jsonl has no c%d B / C row)" % (n, n)); return None
    B, C = arm_line(bl, "B"), arm_line(bl, "C")
    nb, nc = mean(B["nc_ms"]), mean(C["nc_ms"])
    db, dc = mean(B["dec_ms"]), mean(C["dec_ms"])
    ab, ac = mean(B["agg"]), mean(C["agg"])
    sb, sc = mean(B["nc_share"]), mean(C["nc_share"])
    hb, hc = mean(B["share"]), mean(C["share"])
    ref = ""
    if n == 8:
        ref = " | int12 reference (prof1): newcomer %s ms a step = %s %%, decode %s ms a step, agg %s" % (f(a.ref_nc, 1), f(a.ref_share, 1), f(a.ref_dec), f(a.ref_agg, 1))
    out.append("XPSUM c%d newcomer / MIXED steps (the wall split, B = ar_xpush 0, C = ar_xpush 1; blocks %s): ms a step B %s (mean %s) -> C %s (mean %s): %s; "
               "share of the window B %s %% -> C %s %% (%s points); steps a block B %s / C %s%s" % (
                   n, " ".join(b[0] for b in bl), sl(B["nc_ms"]), f(nb, 1), sl(C["nc_ms"]), f(nc, 1), fp(pct(nc, nb)), sl(B["nc_share"]), sl(C["nc_share"]),
                   ("%+.1f" % (sc - sb)) if (sb is not None and sc is not None) else "-", "/".join(str(z) for z in B["nc_steps"]) or "-",
                   "/".join(str(z) for z in C["nc_steps"]) or "-", ref))
    out.append("XPSUM c%d throughput: agg B %s (mean %s) -> C %s (mean %s) tok/s: %s; decode_share B %s -> C %s; decode ms a step (step log) B %s (mean %s) -> "
               "C %s (mean %s): %s; MB ms a decode step B %s -> C %s; request errors B %s / C %s" % (
                   n, sl(B["agg"]), f(ab, 1), sl(C["agg"]), f(ac, 1), fp(pct(ac, ab)), sl(B["share"], 3), sl(C["share"], 3), sl(B["dec_ms"]), f(db),
                   sl(C["dec_ms"]), f(dc), fp(pct(dc, db)), sl(B["mbdec"]), sl(C["mbdec"]), sum(e or 0 for e in B["err"]), sum(e or 0 for e in C["err"])))
    if n == 8:
        d = pct(dc, db)
        if d is None: out.append("XPSUM c8 decode guard: VIRHE not read (B %s / C %s ms a decode step)" % (f(db), f(dc)))
        elif d > a.dec_th: out.append("XPSUM c8 decode guard: VIRHE C's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(dc), f(db), fp(d), f(a.dec_th, 1)))
        else: out.append("XPSUM c8 decode guard: OK C's decode step %s ms vs B's %s ms = %s (limit +%s %%)" % (f(dc), f(db), fp(d), f(a.dec_th, 1)))
    if n == 16:
        g = [z for z in B["gt48"] + C["gt48"]]; gc = [z for z in B["gt48cap"] + C["gt48cap"]]
        bad = any(z for z in g if z) or any(z for z in gc if z)
        unread = any(z is None for z in g)
        out.append("XPSUM c16 > 48-row decode steps (cap 48): step log B %s / C %s, GET cap steps_gt48 B %s / C %s%s" % (
            "/".join(str(z) for z in B["gt48"]) or "-", "/".join(str(z) for z in C["gt48"]) or "-", "/".join(str(z) for z in B["gt48cap"]) or "-",
            "/".join(str(z) for z in C["gt48cap"]) or "-",
            (" -- VIRHE > 48-row steps under the cap" if a.sw else " -- INFO (SW=0 mock: no row cap emulated)") if bad else
            (" -- VIRHE a block's step log not read" if (unread and a.sw) else " -- OK 0")))
    return dict(nc=pct(nc, nb), agg=pct(ac, ab), dec=pct(dc, db), ncb=nb, ncc=nc)


def cold_summary(o, out):
    R = [r for r in jl(os.path.join(o, "cold.jsonl")) if r.get("arm") in ("B", "C")]
    res = {}
    for dep in sorted({r.get("depth") for r in R if r.get("depth")}):
        x = [r for r in R if r.get("depth") == dep]
        ok = [r for r in x if r.get("status") == 200 and not r.get("cached")]
        t = lambda a: [r.get("ttft") for r in ok if r["arm"] == a]
        tps = lambda a: [(r["prompt_tokens"] - (r.get("cached") or 0)) / r["ttft"] for r in ok if r["arm"] == a and r.get("ttft") and r.get("prompt_tokens")]
        nd = lambda a: [r.get("nondecode_ms") for r in ok if r["arm"] == a]
        tb, tc = mean(t("B")), mean(t("C"))
        res[dep] = pct(tc, tb)
        notc = [r.get("name") for r in x if r.get("cached")]
        bad = [r.get("name") for r in x if r.get("status") != 200]
        out.append("XPSUM cold %s (deepq, cold after a reset, prompt %s tok; order %s): TTFT B %s s (mean %s) -> C %s s (mean %s): %s | prefill tok/s B %s -> C %s: %s | "
                   "non-decode ms a step B %s -> C %s: %s%s%s" % (
                       dep, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})), "".join(r["arm"] for r in sorted(x, key=lambda r: r.get("i", 0))),
                       sl(t("B"), 2), f(tb), sl(t("C"), 2), f(tc), fp(pct(tc, tb)), f(mean(tps("B")), 0), f(mean(tps("C")), 0), fp(pct(mean(tps("C")), mean(tps("B")))),
                       sl(nd("B"), 1), sl(nd("C"), 1), fp(pct(mean(nd("C")), mean(nd("B")))),
                       (" -- NOT COLD (left out): %s" % ", ".join(notc)) if notc else "", (" -- VIRHE request errors: %s" % ", ".join(bad)) if bad else ""))
    if not R: out.append("XPSUM cold: no read (cold.jsonl has no B / C row)")
    return res


def ident_summary(o, out):
    try: L = [l.split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    # identw's line is "<tag> warm-up <result>" (the warm-up answers vs the stock reference): its result after "warm-up "
    L = [(k, re.sub(r"^warm-up ", "", v.strip())) for k, v in (x if len(x) == 2 else (x[0], "") for x in L) if k.startswith(("xpu-", "cold-"))]
    same = [k for k, v in L if v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v]
    diff = [k for k, v in L if k not in same]
    out.append("XPSUM identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (len(same), len(L), (" -- VIRHE differ / not compared: %s" % ", ".join(diff)) if diff else ""))
    return len(same), len(L)


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
    cs = cold_summary(o, out)
    ni, nt = ident_summary(o, out)
    out.append("XPSUM headline (C = XPUSH vs B = lockstep): c8 newcomer step %s, c8 agg %s, c8 decode step %s | c16 newcomer step %s, agg %s | cold TTFT %s | identity %d / %d" % (
        fp((s8 or {}).get("nc")), fp((s8 or {}).get("agg")), fp((s8 or {}).get("dec")), fp((s16 or {}).get("nc")), fp((s16 or {}).get("agg")),
        ", ".join("%s %s" % (k, fp(v)) for k, v in sorted(cs.items())) or "-", ni, nt))
    for l in out: print(l)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="xpsum-")
    mb, sp = [], {}
    for i, (a, nc, dec, agg) in enumerate([("B", 540.0, 36.2, 184.0), ("C", 520.0, 36.1, 186.0), ("C", 522.0, 36.3, 186.4), ("B", 538.0, 36.2, 184.4)], 1):
        nm = "c8-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=8, arm=a, sched="S", agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec + 0.1, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms = 0.0 %% (idle > 3 s 0 + window edges 0) | agg 184.0 tok/s\n" % (nm, dec, nc))
        json.dump({"windows": {nm: {"gt48": 0}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
    for i, a in enumerate("BC", 1):
        nm = "c16-%d-S-%s" % (i, a)
        mb.append(dict(name=nm, n=16, arm=a, sched="S", agg_dec_tps=300.0 + i, decode_share=0.9, ms_decode_step=55.0, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 500 steps 30000 ms = 85.7 %%, 55.00 ms a step, 16.80 rows a step, 16.6 tokens a step | (b) newcomer / MIXED 0 steps "
            "0 ms = 0.0 %%, - rows a step, - ms a step | (c) host gaps 0 ms\n" % nm)
        json.dump({"windows": {nm: {"gt48": 0 if a == "B" else 2}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 5}}, open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        json.dump({"cap": {"steps_gt48": 5}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        for i, (a, t) in enumerate([("B", 12.0), ("C", 10.0), ("C", 10.2), ("B", 12.2)], 1):
            fo.write(json.dumps(dict(name="cold-64000-%d-%s" % (i, a), depth=64000, arm=a, i=i, ttft=t, prompt_tokens=65650, cached=0, status=200, nondecode_ms=470.0 - (a == "C") * 60)) + "\n")
    open(os.path.join(d, "identity.txt"), "w").write("xpu-n5 warm-up IDENTICAL: 3 answers\nxpu-a1-C-vs-a1-B IDENTICAL: 20 of 20 answers\nxpu-gs16-C-vs-gs16-B DIFFERENT: q3\n"
                                                     "cold-64000-2-C-vs-cold-64000-1-B IDENTICAL (900 chars)\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--dec-th", "1.0"])
    t = buf.getvalue()
    why = []
    if "ms a step B 540.0/538.0 (mean 539.0) -> C 520.0/522.0 (mean 521.0): -3.34 %" not in t: why.append("c8 newcomer")
    if "XPSUM c8 decode guard: OK" not in t: why.append("decode guard ok")
    if "TTFT B 12.00/12.20 s (mean 12.10) -> C 10.00/10.20 s (mean 10.10): -16.53 %" not in t: why.append("cold")
    if "-- VIRHE > 48-row steps under the cap" not in t: why.append("c16 gt48 caught")
    if "3 of 4 IDENTICAL -- VIRHE differ / not compared: xpu-gs16-C-vs-gs16-B" not in t: why.append("identity")
    # the guard fires past the threshold
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): run([d, "--dec-th", "0.1"])
    if "XPSUM c8 decode guard: VIRHE" in buf2.getvalue(): pass
    else:
        # C mean 36.2 vs B 36.2: 0 %; make C slower to see the VIRHE
        for nm, dec in (("c8-2-S-C", 36.9), ("c8-3-S-C", 36.9)):
            p = os.path.join(d, "split-%s.txt" % nm); s = open(p).read(); open(p, "w").write(re.sub(r"= 80.0 %, [\d.]+ ms a step", "= 80.0 %%, %.2f ms a step" % dec, s))
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3): run([d, "--dec-th", "1.0"])
        if "XPSUM c8 decode guard: VIRHE" not in buf3.getvalue(): why.append("decode guard virhe")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: c8 newcomer / decode guard (OK and VIRHE), c16 > 48 rows caught, cold TTFT C vs B, identity count")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": sys.exit(selftest())
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    sys.exit(run(sys.argv[1:]))
