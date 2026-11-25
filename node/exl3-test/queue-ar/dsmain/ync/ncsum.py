#!/usr/bin/env python3
"""ncsum.py (12.11.2026, queue-ar/dsmain/ync, the ds41-nc builder for ds-main-port) -- the A9 / A11 newcomer screen's summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync.sh, body ync/dsm-ync.sh; ydsum.py's readers with the nc arms). Standard library only.

  ncsum.py OUTDIR [--dec-th 1.0] [--sw 1] [--ref-nc 538.2] [--ref-dec 36.18] [--ref-agg 184.0] [--ref-c16 613.9]
      OUTDIR/mb.jsonl (mbx rows: n, arm, agg_dec_tps, decode_share, ms_decode_step), OUTDIR/split-<block>.txt (p12win.py split:
      (a) decode ms a step / rows, (b) newcomer / MIXED steps: count, share, rows, ms a step), OUTDIR/rows-<block>.json (gt48),
      OUTDIR/fn-<block>-0/1.json (GET cap steps_gt48), OUTDIR/cold.jsonl, OUTDIR/p0-tab.txt (nctab.py's NCTAB lines), identity.txt.
      Arms: B = ds41_bi 2 + gg_pf 1 (int12 / ref12), A = A9 (ds41_bi 6), G = A11 (gg_pf 2: the P0 table's ranged entries), N = both.
      Lines "NCSUM ...": per arm vs B on c8 -- NEWCOMER / MIXED ms a step (THE TARGET), decode ms a step (the guard: not above B's by
      more than --dec-th %), agg, decode_share; the c8 guard; c16 B vs the best arm; cold B vs the best arm; the P0 table; identity;
      a headline.
  ncsum.py best OUTDIR [--dec-th 1.0] [--exclude "A G"]
      the best arm so far by c8's newcomer ms a step (arms whose decode-step guard holds and that are not excluded; B when none
      beats B): one line "BEST <arm> <newcomer ms delta vs B>".
  ncsum.py selftest"""
import json, os, re, statistics, sys, tempfile

DEC = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) ms a step, ([\d.]+|-) rows a step")
NEW = re.compile(r"\(b\) newcomer / MIXED (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) rows a step, ([\d.]+|-) ms a step")
ARMS = ("B", "A", "G", "N")
NAME = {"B": "B (ds41_bi 2, gg_pf 1 = ref12)", "A": "A9 (ds41_bi 6)", "G": "A11 (gg_pf 2)", "N": "A9+A11 (ds41_bi 6, gg_pf 2)"}


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


def sl(v, nd=1):
    return "/".join(f(z, nd) for z in v) or "-"


def blocks(o, n):
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        if r.get("n") != n or r.get("arm") not in ARMS: continue
        nm = r.get("name")
        sp = split_of(os.path.join(o, "split-%s.txt" % nm))
        rw = jload(os.path.join(o, "rows-%s.json" % nm)) or {}
        try: g = list((rw.get("windows") or {}).values())[0].get("gt48")
        except (IndexError, AttributeError): g = None
        a0, a1 = jload(os.path.join(o, "fn-%s-0.json" % nm)) or {}, jload(os.path.join(o, "fn-%s-1.json" % nm)) or {}
        c0, c1 = a0.get("cap") or {}, a1.get("cap")
        gc = (c1.get("steps_gt48", 0) - c0.get("steps_gt48", 0)) if isinstance(c1, dict) else None
        out.append((nm, r["arm"], r, sp, g, gc))
    return out


def arm_of(bl, a):
    x = [b for b in bl if b[1] == a]
    v = lambda k: [b[3][k] if b[3] else None for b in x]
    m = lambda k: [b[2].get(k) for b in x]
    d = dict(n=len(x), names=[b[0] for b in x], nc_ms=v("nc_ms"), nc_share=v("nc_share"), nc_steps=v("nc_steps"), nc_rows=v("nc_rows"),
             dec_ms=v("dec_ms"), dec_rows=v("dec_rows"), agg=m("agg_dec_tps"), share=m("decode_share"), err=m("errors"),
             gt48=[b[4] for b in x], gt48cap=[b[5] for b in x])
    d.update(m_nc=mean(d["nc_ms"]), m_dec=mean(d["dec_ms"]), m_agg=mean(d["agg"]))
    return d


def ranking(o, dec_th, exclude=()):
    bl = blocks(o, 8)
    B = arm_of(bl, "B")
    res = {}
    for a in ARMS[1:]:
        X = arm_of(bl, a)
        if not X["n"] or a in exclude: continue
        d = pct(X["m_dec"], B["m_dec"])
        res[a] = dict(nc=X["m_nc"], dnc=(X["m_nc"] - B["m_nc"]) if (X["m_nc"] is not None and B["m_nc"] is not None) else None,
                      dec=d, guard=(d is not None and d <= dec_th))
    return B, res


def best(o, dec_th, exclude=()):
    B, res = ranking(o, dec_th, exclude)
    ok = [(v["dnc"], a) for a, v in res.items() if v["guard"] and v["dnc"] is not None and v["dnc"] < 0]
    if not ok: return "B", None
    ok.sort()
    return ok[0][1], ok[0][0]


def c8_lines(o, a, out):
    bl = blocks(o, 8)
    if not bl:
        out.append("NCSUM c8: no block (mb.jsonl has no c8 row)"); return {}
    B = arm_of(bl, "B")
    out.append("NCSUM c8 B %s (blocks %s): newcomer / MIXED %s ms a step (mean %s; %s steps, %s rows a step, share %s %%), decode %s ms a step "
               "(mean %s) at %s rows, agg %s (mean %s), decode_share %s | int12 prof1 reference (another start): newcomer %s ms, decode %s ms, agg %s" % (
                   NAME["B"], " ".join(B["names"]), sl(B["nc_ms"]), f(B["m_nc"], 1), sl(B["nc_steps"], 0), sl(B["nc_rows"], 0), sl(B["nc_share"]),
                   sl(B["dec_ms"], 2), f(B["m_dec"]), sl(B["dec_rows"]), sl(B["agg"]), f(B["m_agg"], 1), sl(B["share"], 3), f(a.ref_nc, 1), f(a.ref_dec), f(a.ref_agg, 1)))
    res = {}
    for x in ARMS[1:]:
        X = arm_of(bl, x)
        if not X["n"]: continue
        dnc = (X["m_nc"] - B["m_nc"]) if (X["m_nc"] is not None and B["m_nc"] is not None) else None
        dd = pct(X["m_dec"], B["m_dec"])
        g = "not read" if dd is None else ("OK" if dd <= a.dec_th else "VIRHE")
        out.append("NCSUM c8 %s vs B (blocks %s): NEWCOMER / MIXED ms a step B %s (mean %s) -> %s %s (mean %s): %s (%s ms) [THE TARGET] | decode ms a step "
                   "B %s -> %s (mean %s): %s [guard <= +%s %%: %s] | agg B %s -> %s (mean %s): %s | decode_share B %s -> %s | newcomer share B %s %% -> %s %% | "
                   "request errors %s" % (
                       NAME[x], " ".join(X["names"]), sl(B["nc_ms"]), f(B["m_nc"], 1), x, sl(X["nc_ms"]), f(X["m_nc"], 1), fp(pct(X["m_nc"], B["m_nc"])),
                       ("%+.1f" % dnc) if dnc is not None else "-", f(B["m_dec"]), sl(X["dec_ms"], 2), f(X["m_dec"]), fp(dd), f(a.dec_th, 1), g,
                       f(B["m_agg"], 1), sl(X["agg"]), f(X["m_agg"], 1), fp(pct(X["m_agg"], B["m_agg"])), sl(B["share"], 3), sl(X["share"], 3),
                       sl(B["nc_share"]), sl(X["nc_share"]), sum(e or 0 for e in X["err"])))
        res[x] = dict(dnc=dnc, nc=pct(X["m_nc"], B["m_nc"]), dec=dd, agg=pct(X["m_agg"], B["m_agg"]), guard=g)
    return res


def c16_lines(o, a, out):
    bl = blocks(o, 16)
    if not bl:
        out.append("NCSUM c16: no block"); return None
    B = arm_of(bl, "B")
    xs = [x for x in ARMS[1:] if arm_of(bl, x)["n"]]
    g = [z for b in bl for z in (b[4],)]; gc = [b[5] for b in bl]
    bad = any(z for z in g if z) or any(z for z in gc if z)
    unread = any(z is None for z in g)
    r = None
    for x in xs:
        X = arm_of(bl, x)
        out.append("NCSUM c16 %s vs B (ncmb3 1x16, cap 48; blocks %s / %s): newcomer / MIXED ms a step B %s -> %s %s: %s (%s ms) | decode ms a step B %s -> %s: %s | "
                   "agg B %s -> %s: %s | decode_share B %s -> %s | newcomer share B %s %% -> %s %% | int12 reference (another start): newcomer %s ms" % (
                       NAME[x], " ".join(B["names"]), " ".join(X["names"]), f(B["m_nc"], 1), x, f(X["m_nc"], 1), fp(pct(X["m_nc"], B["m_nc"])),
                       ("%+.1f" % (X["m_nc"] - B["m_nc"])) if (X["m_nc"] is not None and B["m_nc"] is not None) else "-", f(B["m_dec"]), f(X["m_dec"]),
                       fp(pct(X["m_dec"], B["m_dec"])), f(B["m_agg"], 1), f(X["m_agg"], 1), fp(pct(X["m_agg"], B["m_agg"])), sl(B["share"], 3), sl(X["share"], 3),
                       sl(B["nc_share"]), sl(X["nc_share"]), f(a.ref_c16, 1)))
        r = dict(arm=x, nc=pct(X["m_nc"], B["m_nc"]), agg=pct(X["m_agg"], B["m_agg"]))
    if not xs:
        out.append("NCSUM c16 B only (blocks %s): newcomer %s ms, decode %s ms, agg %s" % (" ".join(B["names"]), f(B["m_nc"], 1), f(B["m_dec"]), f(B["m_agg"], 1)))
    out.append("NCSUM c16 > 48-row decode steps (cap 48): step log %s, GET cap steps_gt48 %s%s" % (
        "/".join(str(z) for z in g) or "-", "/".join(str(z) for z in gc) or "-",
        (" -- VIRHE > 48-row steps under the cap" if a.sw else " -- INFO (SW=0 mock)") if bad else (" -- VIRHE a block's step log not read" if (unread and a.sw) else " -- OK 0")))
    return r


def cold_lines(o, out):
    R = [r for r in jl(os.path.join(o, "cold.jsonl")) if r.get("arm") in ARMS]
    res = {}
    for dep in sorted({r.get("depth") for r in R if r.get("depth")}):
        x = [r for r in R if r.get("depth") == dep]
        ok = [r for r in x if r.get("status") == 200 and not r.get("cached")]
        arms = [a for a in ARMS if any(r["arm"] == a for r in x)]
        t = lambda a: [r.get("ttft") for r in ok if r["arm"] == a]
        nd = lambda a: [r.get("nondecode_ms") for r in ok if r["arm"] == a]
        tb = mean(t("B"))
        parts = []
        for a in arms:
            if a == "B": continue
            parts.append("%s %s s (%s), non-decode ms a step %s (%s)" % (a, sl(t(a), 2), fp(pct(mean(t(a)), tb)), sl(nd(a), 1), fp(pct(mean(nd(a)), mean(nd("B"))))))
            res[(dep, a)] = pct(mean(t(a)), tb)
        bad = [r.get("name") for r in x if r.get("status") != 200]
        notc = [r.get("name") for r in x if r.get("cached")]
        out.append("NCSUM cold %s (deepq after a prefix-cache reset, prompt %s tok, order %s): TTFT B %s s, non-decode ms a step %s | %s%s%s" % (
            dep, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})), " ".join(r.get("arm", "?") for r in x), sl(t("B"), 2), sl(nd("B"), 1),
            " | ".join(parts) or "no other arm", (" -- NOT COLD (left out): %s" % ", ".join(notc)) if notc else "",
            (" -- VIRHE request errors: %s" % ", ".join(bad)) if bad else ""))
    return res


def tab_lines(o, out, sw=True):
    try: L = open(os.path.join(o, "p0-tab.txt")).read().splitlines()
    except OSError: L = []
    t = next((l for l in L if l.startswith("NCTABLE ")), None)
    ent = [] if (not t or t.strip() == "NCTABLE -") else t[8:].strip().split(",")
    out.append("NCSUM P0 A11 table (r4d_selftest --perfgp on one card, every identical form vs int12's pick, summed over the range's Ms): %d ranged entries%s" % (
        len(ent), (": " + ",".join(ent)) if ent else ("" if t else (" -- VIRHE no NCTABLE line" if sw else " -- INFO (SW=0 mock)"))))
    for l in L:
        if l.startswith("NCTAB ") and "ENTRY" in l: out.append("NCSUM P0 " + l[:400])
    return len(ent)


def ident_lines(o, out):
    try: L = [l.split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    L = [(k, re.sub(r"^warm-up ", "", v.strip())) for k, v in (x if len(x) == 2 else (x[0], "") for x in L) if k.startswith("ync-")]
    same = [k for k, v in L if v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v]
    diff = [k for k, v in L if k not in same]
    out.append("NCSUM identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (len(same), len(L), (" -- VIRHE differ / not compared: %s" % ", ".join(diff)) if diff else ""))
    return len(same), len(L)


def run(argv):
    o = argv[0]
    kv = dict(zip(argv[1::2], argv[2::2]))
    a = type("A", (), {})()
    a.dec_th = float(kv.get("--dec-th", 1.0)); a.sw = kv.get("--sw", "1") == "1"
    a.ref_nc, a.ref_dec, a.ref_agg = float(kv.get("--ref-nc", 538.2)), float(kv.get("--ref-dec", 36.18)), float(kv.get("--ref-agg", 184.0))
    a.ref_c16 = float(kv.get("--ref-c16", 613.9))
    out = []
    r8 = c8_lines(o, a, out)
    bst, bd = best(o, a.dec_th)
    r16 = c16_lines(o, a, out)
    cold = cold_lines(o, out)
    nt = tab_lines(o, out, a.sw)
    ni, nn = ident_lines(o, out)
    out.append("NCSUM headline: c8 newcomer ms a step vs B -- %s | best by c8 %s%s | c16 %s | cold %s | A11 table %d entries | identity %d / %d" % (
        " ; ".join("%s %s (%s ms, decode %s, guard %s)" % (x, fp(v["nc"]), ("%+.1f" % v["dnc"]) if v["dnc"] is not None else "-", fp(v["dec"]), v["guard"])
                   for x, v in r8.items()) or "no lever block",
        bst, (" (%+.1f ms)" % bd) if bd is not None else "", ("%s newcomer %s, agg %s" % (r16["arm"], fp(r16["nc"]), fp(r16["agg"]))) if r16 else "-",
        " ; ".join("%s %s %s" % (d, x, fp(v)) for (d, x), v in sorted(cold.items())) or "-", nt, ni, nn))
    for l in out: print(l)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="ncsum-")
    mb = []
    order = [("B", 538.0, 36.2, 184.0), ("A", 528.0, 36.25, 186.0), ("G", 533.0, 36.2, 185.0), ("N", 522.0, 36.3, 187.0),
             ("G", 534.0, 36.2, 185.2), ("A", 529.0, 36.2, 186.2), ("B", 539.0, 36.1, 184.2)]
    for i, (x, nc, dec, agg) in enumerate(order, 1):
        nm = "c8-%d-S-%s" % (i, x)
        mb.append(dict(name=nm, n=8, arm=x, sched="S", agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms\n" % (nm, dec, nc))
    for i, x in enumerate("BN", 1):
        nm = "c16-%d-S-%s" % (i, x)
        mb.append(dict(name=nm, n=16, arm=x, sched="S", agg_dec_tps=218.0 + 3 * i, decode_share=0.72, ms_decode_step=55.0, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 500 steps 25000 ms = 71.6 %%, 55.00 ms a step, 16.00 rows a step | (b) newcomer / MIXED 16 steps "
            "9822 ms = 28.1 %%, 1118 rows a step, %.1f ms a step | (c) host gaps 0 ms\n" % (nm, 614.0 if x == "B" else 598.0))
        json.dump({"windows": {nm: {"gt48": 0}}}, open(os.path.join(d, "rows-%s.json" % nm), "w"))
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        for dep, x, t in ((64000, "B", 20.0), (64000, "N", 19.6), (136000, "B", 50.0), (136000, "N", 49.5)):
            fo.write(json.dumps(dict(name="cold-%d-%s" % (dep, x), depth=dep, arm=x, ttft=t, prompt_tokens=dep + 100, cached=0, status=200, nondecode_ms=900.0)) + "\n")
    open(os.path.join(d, "p0-tab.txt"), "w").write("NCTAB q6_K attn_output_b 5120x2048: int12's pick 426 ... -> ENTRY | x\nNCTABLE q6_K:5120x2048@1000-1536=0\n")
    open(os.path.join(d, "identity.txt"), "w").write("ync-n5 warm-up IDENTICAL: 3 answers\nync-a1-A-vs-a1-B IDENTICAL: 20 of 20\nync-ld-A-vs-a1-A DIFFERENT at char 40\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d])
    t = buf.getvalue()
    why = []
    if "NEWCOMER / MIXED ms a step B 538.0/539.0 (mean 538.5) -> A 528.0/529.0 (mean 528.5): -1.86 % (-10.0 ms)" not in t: why.append("c8 A")
    if "guard <= +1.0 %: OK" not in t: why.append("guard ok")
    if "-> N 522.0 (mean 522.0): -3.06 % (-16.5 ms)" not in t: why.append("c8 N")
    if "best by c8 N (-16.5 ms)" not in t: why.append("best")
    if "newcomer / MIXED ms a step B 614.0 -> N 598.0: -2.61 % (-16.0 ms)" not in t: why.append("c16")
    if "NCSUM cold 136000" not in t or "N 49.50 s (-1.00 %)" not in t: why.append("cold")
    if "1 ranged entries: q6_K:5120x2048@1000-1536=0" not in t: why.append("tab")
    if "2 of 3 IDENTICAL -- VIRHE differ / not compared: ync-ld-A-vs-a1-A" not in t: why.append("identity")
    b, bd = best(d, 1.0, exclude=("N",))
    if b != "A": why.append("best exclude N -> %s" % b)
    # a guard failure: N's decode +2 % -> not best
    p = os.path.join(d, "split-c8-4-S-N.txt"); s = open(p).read(); open(p, "w").write(s.replace("36.30 ms a step", "36.95 ms a step"))
    b, _ = best(d, 1.0)
    if b != "A": why.append("guard excludes N -> %s" % b)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): run([d])
    if "guard <= +1.0 %: VIRHE" not in buf2.getvalue(): why.append("guard virhe")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: c8 newcomer target per arm vs B, decode guard (OK / VIRHE), best (+ exclude, guard), c16 best vs B, cold, P0 table, identity, headline")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    if argv[0] == "best":
        kv = dict(zip(argv[2::2], argv[3::2]))
        b, bd = best(argv[1], float(kv.get("--dec-th", 1.0)), tuple(kv.get("--exclude", "").split()))
        print("BEST %s %s" % (b, ("%+.1f" % bd) if bd is not None else "-"))
        return 0
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
