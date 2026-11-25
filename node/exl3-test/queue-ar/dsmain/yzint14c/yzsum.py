#!/usr/bin/env python3
"""yzsum.py (12.11.2026, queue-ar/dsmain/yzint14c, the int14c builder for ds-main-port) -- the DS base-candidate screen's summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c.sh, body yzint14c/dsm-yzint14c.sh; ync/ncsum.py's readers + yswl's read set). Standard
library only.

  yzsum.py OUTDIR [--dec-th 1.0] [--sw 1] [--swlg 75]
      OUTDIR/mb.jsonl (mbx rows: name <c8|c16>-<i>-S-<arm>, n, arm, agg_dec_tps, decode_share, ms_decode_step; i < 10 = step 2, the
      per-lever reads; i > 10 = step 3, the all-on reads), OUTDIR/split-<block>.txt (p12win.py split: (a) decode ms a step, (b) newcomer
      / MIXED ms a step), OUTDIR/c16cap.jsonl (the CAP48-method c16 blocks), OUTDIR/blocks.jsonl (resprobe reads: lever XC / XE / XW
      / U1W / U1, arm, j = the J line, st = the /stats deltas), OUTDIR/cold.jsonl, OUTDIR/identity.txt, OUTDIR/p0.json.
      Arms: B = int12 / ref12; A = A9 (ds41_bi 6); G = A11 (gg_pf 2); C = A4 (gg_gemv_m16 1); W = SWL (place_swap_lent 1 +
      heat_min_gain_x100 SWLG); Z = ALL. Lines "YZSUM ...": one DECISION line a lever reading and an ALL reading (values, B the
      reference, the % difference, the identity / staggered outcome), the int12 references as INFO, a TABLE line, a headline.
  yzsum.py selftest"""
import json, os, re, statistics, sys, tempfile

DEC = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) ms a step, ([\d.]+|-) rows a step")
NEW = re.compile(r"\(b\) newcomer / MIXED (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) rows a step, ([\d.]+|-) ms a step")
NAME = re.compile(r"^c(\d+)-(\d+)-S-([A-Z]+)$")
ARMS = ("B", "A", "G", "C", "W", "Z")
# the int12 references (other starts; INFO only, never re-run): LEVERS.md PROF12 prof1 / RES12 / ref12 / XPUSH screen
REF = dict(c8=184.0, c16=217.8, c16cap=300.7, u1=91.01, cold136=24.57, nc8=538.2, dec8=36.18, nc16=613.9)
XPUSH_INFO = ("XPUSH screen 12.11. (int13, in-process B C C B): c8 agg +0.32 %, c8 newcomer step -0.72 %, decode step -0.64 %; c16 agg "
              "-0.27 %; cold 64k -1.76 %, 136k -1.71 %; AR4XP 2048 rows -26.7 %")


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
    return dict(dec_ms=fnum(d.group(4)), dec_rows=fnum(d.group(5)), nc_steps=int(n.group(1)), nc_share=fnum(n.group(3)), nc_ms=fnum(n.group(5)))


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


def mbblocks(o, n, sec=None):
    """mb.jsonl rows of c<n> -> [(name, arm, i, row, split)]; sec 'lev' (i < 10) / 'all' (i > 10) / None (both)"""
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        m = NAME.match(r.get("name") or "")
        if not m or int(m.group(1)) != n or r.get("n") != n: continue
        i, a = int(m.group(2)), m.group(3)
        if sec == "lev" and i > 10: continue
        if sec == "all" and i <= 10: continue
        out.append((r["name"], a, i, r, split_of(os.path.join(o, "split-%s.txt" % r["name"]))))
    return out


def armstat(bl, a):
    x = [b for b in bl if b[1] == a]
    v = lambda k: [b[4][k] if b[4] else None for b in x]
    m = lambda k: [b[3].get(k) for b in x]
    d = dict(n=len(x), names=[b[0] for b in x], nc=v("nc_ms"), dec=v("dec_ms"), dec_mb=m("ms_decode_step"), agg=m("agg_dec_tps"),
             share=m("decode_share"), err=m("errors"))
    d.update(m_nc=mean(d["nc"]), m_dec=mean(d["dec"]) if any(z is not None for z in d["dec"]) else mean(d["dec_mb"]), m_agg=mean(d["agg"]))
    return d


def idmap(o):
    try: L = [l.rstrip("\n").split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    out = {}
    for x in L:
        k, v = (x[0], x[1] if len(x) > 1 else "")
        if not k.startswith("yzi-"): continue
        v = re.sub(r"^warm-up ", "", v.strip())
        out[k[4:]] = "IDENTICAL" if (v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v) else ("DIFFERENT: " + v[:120])
    return out


def idsay(ids, *keys):
    got = [(k, ids.get(k)) for k in keys]
    if all(v is None for _, v in got): return "not run"
    bad = ["%s %s" % (k, v or "not compared") for k, v in got if v != "IDENTICAL"]
    return ("IDENTICAL (%s)" % ", ".join(k for k, _ in got if _ is not None)) if not bad else ("VIRHE " + "; ".join(bad))


def reads(o, levers, arm, sett=None):
    R = [r for r in jl(os.path.join(o, "blocks.jsonl")) if r.get("lever") in levers and r.get("arm") == arm and (sett is None or r.get("set") == sett)]
    j = lambda k: [(r.get("j") or {}).get(k) for r in R]
    s = lambda k: [(r.get("st") or {}).get(k) for r in R]
    return dict(n=len(R), tps=j("decode_tps"), ms=j("ms_per_decode_step"), tok=j("tokens_per_step"), md5=j("text_md5"), mb=s("stream_mb_step"),
                m_tps=mean(j("decode_tps")), m_ms=mean(j("ms_per_decode_step")), m_mb=mean(s("stream_mb_step")))


def run(argv):
    o = argv[0]
    kv = dict(zip(argv[1::2], argv[2::2]))
    dec_th = float(kv.get("--dec-th", 1.0)); sw = kv.get("--sw", "1") == "1"; swlg = kv.get("--swlg", "75")
    out, row = [], []
    ids = idmap(o)
    p0 = jload(os.path.join(o, "p0.json")) or {}
    out.append("YZSUM P0: XPUSH %s | A4 %s | A11 %s | levers %s" % (p0.get("xpush", "-"), p0.get("a4", "-"), p0.get("a11", "-"), p0.get("levers", "-")))
    gate = ["a1-Z-vs-ref12-a1", "a1-B-vs-ref12-a1", "a1-Z-vs-a1-B", "gs16-Z-vs-a1-Z", "ld-Z-vs-a1-Z", "ld-A-vs-a1-B"]
    gate += sorted(k for k in ids if re.match(r"cold-1[0-9]{3}-Z-vs-B$", k))
    stopped = any(k in ids and ids[k] != "IDENTICAL" for k in gate)
    out.append("YZSUM IDENTITY GATE (strict, T 0 greedy; before every speed block): %s" % " | ".join("%s %s" % (k, ids.get(k, "not run")) for k in gate))
    # ---- step 2: the per-lever reads
    c8l = mbblocks(o, 8, "lev"); B8 = armstat(c8l, "B")
    for a, lab in (("A", "A9 (ds41_bi 6)"), ("G", "A11 (gg_pf 2, the P0's ranged entries)")):
        X = armstat(c8l, a)
        if not X["n"]: out.append("YZSUM DECISION %s c8: not run" % lab); continue
        dd = pct(X["m_dec"], B8["m_dec"]); g = "-" if dd is None else ("OK" if dd <= dec_th else "VIRHE")
        idt = idsay(ids, "ld-A-vs-a1-B") if a == "A" else idsay(ids, *[k for k in ids if re.match(r"cold-1[0-9]{3}-Z-vs-B$", k)] or ["cold-1100-Z-vs-B"])
        out.append("YZSUM DECISION %s | c8 newcomer / MIXED ms a step (the target) B %s (%s) -> %s %s (%s): %s (%s ms) | decode ms a step B %s -> %s: %s "
                   "[guard <= +%.1f %%: %s] | agg B %s -> %s: %s | %s | identity: %s" % (
                       lab, f(B8["m_nc"], 1), sl(B8["nc"]), a, f(X["m_nc"], 1), sl(X["nc"]), fp(pct(X["m_nc"], B8["m_nc"])),
                       ("%+.1f" % (X["m_nc"] - B8["m_nc"])) if (X["m_nc"] is not None and B8["m_nc"] is not None) else "-",
                       f(B8["m_dec"]), f(X["m_dec"]), fp(dd), dec_th, g, f(B8["m_agg"], 1), f(X["m_agg"], 1), fp(pct(X["m_agg"], B8["m_agg"])),
                       "blocks " + " ".join(B8["names"] + X["names"]), idt))
        row.append("%s c8 newcomer %s, agg %s" % ("A9" if a == "A" else "A11", fp(pct(X["m_nc"], B8["m_nc"])), fp(pct(X["m_agg"], B8["m_agg"]))))
    c16l = mbblocks(o, 16, "lev"); B16 = armstat(c16l, "B"); C16 = armstat(c16l, "C")
    if C16["n"]:
        out.append("YZSUM DECISION A4 (gg_gemv_m16 1, yd2m's 95-line table) | c16 fixed harness (ncmb3 1x16, cap 48, 35 s): agg B %s -> C %s: %s | decode ms a step "
                   "B %s -> %s: %s | newcomer ms B %s -> %s: %s | blocks %s | identity: P0 --mrcheck16 %s; in Z: %s" % (
                       f(B16["m_agg"], 1), f(C16["m_agg"], 1), fp(pct(C16["m_agg"], B16["m_agg"])), f(B16["m_dec"]), f(C16["m_dec"]), fp(pct(C16["m_dec"], B16["m_dec"])),
                       f(B16["m_nc"], 1), f(C16["m_nc"], 1), fp(pct(C16["m_nc"], B16["m_nc"])), " ".join(B16["names"] + C16["names"]), p0.get("a4", "-"),
                       idsay(ids, "a1-Z-vs-ref12-a1", "gs16-Z-vs-a1-Z")))
        row.append("A4 c16 fixed %s" % fp(pct(C16["m_agg"], B16["m_agg"])))
    else: out.append("YZSUM DECISION A4 c16 fixed harness: not run")
    cap = jl(os.path.join(o, "c16cap.jsonl"))
    cb, cc = [r for r in cap if r.get("arm") == "B"], [r for r in cap if r.get("arm") == "C"]
    if cb and cc:
        ab, ac = mean([r.get("agg_dec_tps") for r in cb]), mean([r.get("agg_dec_tps") for r in cc])
        db, dc = mean([r.get("ms_decode_step") for r in cb]), mean([r.get("ms_decode_step") for r in cc])
        out.append("YZSUM DECISION A4 | c16 CAP48 method (cap48/ncmb.py phase c16, decode only): agg B %s -> C %s: %s | ms a decode step B %s -> %s: %s | "
                   "tok a step B %s -> %s | steps_gt48 %s | INFO int12 prof1 %s: B %s, C %s" % (
                       f(ab, 1), f(ac, 1), fp(pct(ac, ab)), f(db), f(dc), fp(pct(dc, db)), sl([r.get("tok_per_decode_step") for r in cb], 2),
                       sl([r.get("tok_per_decode_step") for r in cc], 2), "/".join(str(r.get("gt48")) for r in cap), REF["c16cap"],
                       fp(pct(ab, REF["c16cap"])), fp(pct(ac, REF["c16cap"]))))
        row.append("A4 c16 CAP48 %s" % fp(pct(ac, ab)))
    else: out.append("YZSUM DECISION A4 c16 CAP48 method: not run")
    sb, sw_ = reads(o, ("XE",), "B"), reads(o, ("XE",), "W")
    if sb["n"] and sw_["n"]:
        wb, ww = reads(o, ("XW",), "B"), reads(o, ("XW",), "W")
        out.append("YZSUM DECISION SWL (place_swap_lent 1 + heat_min_gain_x100 %s) | switched essay (after code) tok/s B %s (%s) -> W %s (%s): %s | ms a step B %s -> %s: "
                   "%s | stream MB a step B %s -> %s | warm essay tok/s B %s -> W %s: %s (ms %s -> %s) | answers %s | yswl (another start): switched +0.8 %%, c8 +1.5 %%" % (
                       swlg, f(sb["m_tps"]), sl(sb["tps"], 2), f(sw_["m_tps"]), sl(sw_["tps"], 2), fp(pct(sw_["m_tps"], sb["m_tps"])), f(sb["m_ms"]), f(sw_["m_ms"]),
                       fp(pct(sw_["m_ms"], sb["m_ms"])), f(sb["m_mb"], 1), f(sw_["m_mb"], 1), f(wb["m_tps"]), f(ww["m_tps"]), fp(pct(ww["m_tps"], wb["m_tps"])),
                       f(wb["m_ms"]), f(ww["m_ms"]), "IDENTICAL" if len(set(x for x in sb["md5"] + sw_["md5"] + wb["md5"] + ww["md5"] if x)) <= 1 else "VIRHE essay answers differ"))
        row.append("SWL switched essay %s" % fp(pct(sw_["m_tps"], sb["m_tps"])))
    else: out.append("YZSUM DECISION SWL lever reads: not run (dropped by the budget, or SWL off)")
    out.append("YZSUM DECISION XPUSH (ar_xpush 1) | no read of its own here (P0 %s; in Z) | INFO %s" % (p0.get("xpush", "-"), XPUSH_INFO))
    # ---- step 3: the all-on reads
    c8a = mbblocks(o, 8, "all"); BA, ZA = armstat(c8a, "B"), armstat(c8a, "Z")
    if ZA["n"]:
        dd = pct(ZA["m_dec"], BA["m_dec"])
        out.append("YZSUM DECISION ALL (Z) | c8 (ncmb3 35 s, P12SPLIT): agg B %s (%s) -> Z %s (%s): %s | newcomer ms a step B %s -> %s: %s (%s ms) | decode ms a step "
                   "B %s -> %s: %s | decode_share B %s -> %s | blocks %s | INFO int12 prof1 c8 %.1f: B %s, Z %s" % (
                       f(BA["m_agg"], 1), sl(BA["agg"]), f(ZA["m_agg"], 1), sl(ZA["agg"]), fp(pct(ZA["m_agg"], BA["m_agg"])), f(BA["m_nc"], 1), f(ZA["m_nc"], 1),
                       fp(pct(ZA["m_nc"], BA["m_nc"])), ("%+.1f" % (ZA["m_nc"] - BA["m_nc"])) if (ZA["m_nc"] is not None and BA["m_nc"] is not None) else "-",
                       f(BA["m_dec"]), f(ZA["m_dec"]), fp(dd), sl(BA["share"], 3), sl(ZA["share"], 3), " ".join(BA["names"] + ZA["names"]), REF["c8"],
                       fp(pct(BA["m_agg"], REF["c8"])), fp(pct(ZA["m_agg"], REF["c8"]))))
        row.append("ALL c8 %s" % fp(pct(ZA["m_agg"], BA["m_agg"])))
    else: out.append("YZSUM DECISION ALL c8: not run%s" % (" (STOPPED at the identity gate)" if stopped else ""))
    c16a = mbblocks(o, 16); B16a, Z16 = armstat(c16a, "B"), armstat(c16a, "Z")
    if Z16["n"]:
        out.append("YZSUM DECISION ALL (Z) | c16 fixed harness: agg B %s -> Z %s: %s | newcomer ms B %s -> %s: %s | decode ms a step B %s -> %s: %s | blocks %s | "
                   "INFO int12 c16 fixed %.1f: B %s, Z %s" % (
                       f(B16a["m_agg"], 1), f(Z16["m_agg"], 1), fp(pct(Z16["m_agg"], B16a["m_agg"])), f(B16a["m_nc"], 1), f(Z16["m_nc"], 1),
                       fp(pct(Z16["m_nc"], B16a["m_nc"])), f(B16a["m_dec"]), f(Z16["m_dec"]), fp(pct(Z16["m_dec"], B16a["m_dec"])),
                       " ".join(B16a["names"] + Z16["names"]), REF["c16"], fp(pct(B16a["m_agg"], REF["c16"])), fp(pct(Z16["m_agg"], REF["c16"]))))
        row.append("ALL c16 %s" % fp(pct(Z16["m_agg"], B16a["m_agg"])))
    else: out.append("YZSUM DECISION ALL c16: not run")
    ub, uz = reads(o, ("U1",), "B"), reads(o, ("U1",), "Z")
    if ub["n"] and uz["n"]:
        out.append("YZSUM DECISION ALL (Z) | 1u essay (one discarded warm-up + %d / %d back to back): tok/s B %s (%s) -> Z %s (%s): %s | ms a step B %s -> %s: %s | "
                   "stream MB a step B %s -> %s | answers %s | INFO int12 prof1 1u %.2f: B %s, Z %s" % (
                       ub["n"], uz["n"], f(ub["m_tps"]), sl(ub["tps"], 2), f(uz["m_tps"]), sl(uz["tps"], 2), fp(pct(uz["m_tps"], ub["m_tps"])), f(ub["m_ms"]),
                       f(uz["m_ms"]), fp(pct(uz["m_ms"], ub["m_ms"])), f(ub["m_mb"], 1), f(uz["m_mb"], 1),
                       "IDENTICAL" if len(set(x for x in ub["md5"] + uz["md5"] if x)) <= 1 else "VIRHE essay answers differ", REF["u1"],
                       fp(pct(ub["m_tps"], REF["u1"])), fp(pct(uz["m_tps"], REF["u1"]))))
        row.append("ALL 1u %s" % fp(pct(uz["m_tps"], ub["m_tps"])))
    else: out.append("YZSUM DECISION ALL 1u essay: not run")
    C = [r for r in jl(os.path.join(o, "cold.jsonl")) if (r.get("depth") or 0) > 2000]   # the A11 in-range reads (1100 / 1300) are identity only
    for dep in sorted({r.get("depth") for r in C}):
        x = [r for r in C if r.get("depth") == dep and r.get("status") == 200 and not r.get("cached")]
        tb, tz = mean([r.get("ttft") for r in x if r.get("arm") == "B"]), mean([r.get("ttft") for r in x if r.get("arm") == "Z"])
        ref = REF["cold136"] if dep == 136000 else None
        out.append("YZSUM DECISION ALL (Z) | cold %d (deepq after a prefix-cache reset, %s tok): TTFT B %s s -> Z %s s: %s | non-decode ms a step B %s -> Z %s | answers %s%s" % (
            dep, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})) or "-", f(tb), f(tz), fp(pct(tz, tb)),
            f(mean([r.get("nondecode_ms") for r in x if r.get("arm") == "B"]), 1), f(mean([r.get("nondecode_ms") for r in x if r.get("arm") == "Z"]), 1),
            ids.get("cold-%d-Z-vs-B" % dep, "not compared"), (" | INFO int12 %.2f s: B %s, Z %s" % (ref, fp(pct(tb, ref)), fp(pct(tz, ref)))) if ref else ""))
        row.append("ALL cold %dk %s" % (dep // 1000, fp(pct(tz, tb))))
    if not C: out.append("YZSUM DECISION ALL cold: not run")
    nid = len(ids); nok = sum(1 for v in ids.values() if v == "IDENTICAL")
    out.append("YZSUM identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (nok, nid, (" -- VIRHE differ / not compared: %s" % ", ".join(k for k, v in ids.items() if v != "IDENTICAL")) if nok < nid else ""))
    out.append("YZSUM TABLE " + " || ".join(row) if row else "YZSUM TABLE no speed reading%s" % (" (STOPPED at the identity gate: NEEDS USER)" if stopped else ""))
    out.append("YZSUM headline: %s | identity %d / %d%s" % ("; ".join(row) or "no speed reading", nok, nid, " | STOPPED at the identity gate (no speed block): NEEDS USER" if stopped else ""))
    for l in out: print(l)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="yzsum-")
    mb = []
    def blk(nm, n, arm, agg, dec, nc):
        mb.append(dict(name=nm, n=n, arm=arm, sched="S", agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 13 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, %.1f ms a step | (c) host gaps 0 ms\n" % (nm, dec, nc))
    for i, (x, nc, dec, agg) in enumerate([("B", 538.0, 36.2, 184.0), ("A", 528.0, 36.25, 186.0), ("G", 533.0, 36.2, 185.0), ("B", 539.0, 36.1, 184.2)], 1):
        blk("c8-%d-S-%s" % (i, x), 8, x, agg, dec, nc)
    for i, (x, nc, dec, agg) in enumerate([("B", 537.0, 36.2, 185.0), ("Z", 520.0, 35.9, 189.0), ("Z", 521.0, 35.9, 189.2), ("B", 537.0, 36.2, 185.2)], 11):
        blk("c8-%d-S-%s" % (i, x), 8, x, agg, dec, nc)
    blk("c16-1-S-B", 16, "B", 218.0, 55.0, 614.0); blk("c16-2-S-C", 16, "C", 221.0, 54.2, 612.0); blk("c16-4-S-Z", 16, "Z", 224.0, 54.0, 598.0)
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "c16cap.jsonl"), "w") as fo:
        fo.write(json.dumps(dict(name="c16cap-1-B", arm="B", agg_dec_tps=300.0, ms_decode_step=55.5, tok_per_decode_step=16.8, gt48="0")) + "\n")
        fo.write(json.dumps(dict(name="c16cap-2-C", arm="C", agg_dec_tps=304.0, ms_decode_step=54.8, tok_per_decode_step=16.8, gt48="0")) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        for lv, a, tps, ms, mb_ in (("XC", "B", 134, 27.6, 340), ("XE", "B", 79.0, 19.9, 173), ("XE", "B", 79.4, 19.8, 170), ("XW", "B", 89.3, 17.6, 28),
                                    ("XE", "W", 79.8, 19.7, 158), ("XE", "W", 79.9, 19.7, 159), ("XW", "W", 89.4, 17.6, 27),
                                    ("U1W", "B", 85, 18.5, 60), ("U1", "B", 90.5, 17.4, 11), ("U1", "B", 90.7, 17.4, 11), ("U1W", "Z", 85, 18.4, 60), ("U1", "Z", 91.5, 17.2, 10), ("U1", "Z", 91.7, 17.2, 10)):
            fo.write(json.dumps(dict(lever=lv, arm=a, j=dict(decode_tps=tps, ms_per_decode_step=ms, tokens_per_step=1.58, text_md5="e" if lv != "XC" else "c"), st=dict(stream_mb_step=mb_))) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        for dep, x, t in ((1100, "B", 0.4), (1100, "Z", 0.4), (136000, "B", 24.6), (136000, "Z", 24.1)):
            fo.write(json.dumps(dict(name="cold-%d-%s" % (dep, x), depth=dep, arm=x, ttft=t, prompt_tokens=dep + 1135, cached=0, status=200, nondecode_ms=420.0)) + "\n")
    json.dump(dict(xpush="PASS", a4="PASS", a11="3 entries", levers="XPUSH 1 A4 1 A11 1 A9 1 SWL 1"), open(os.path.join(d, "p0.json"), "w"))
    open(os.path.join(d, "identity.txt"), "w").write("".join("yzi-%s IDENTICAL: 20 of 20 answers\n" % k for k in (
        "a1-Z-vs-ref12-a1", "a1-B-vs-ref12-a1", "a1-Z-vs-a1-B", "gs16-Z-vs-a1-Z", "cold-1100-Z-vs-B", "ld-Z-vs-a1-Z", "ld-A-vs-a1-B", "cold-136000-Z-vs-B")))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--swlg", "75"])
    t = buf.getvalue()
    why = []
    if "B 538.5 (538.0/539.0) -> A 528.0 (528.0): -1.95 % (-10.5 ms)" not in t: why.append("A9")
    if "[guard <= +1.0 %: OK]" not in t: why.append("guard")
    if "-> G 533.0 (533.0): -1.02 %" not in t: why.append("A11")
    if "agg B 218.0 -> C 221.0: +1.38 %" not in t: why.append("A4 c16")
    if "agg B 300.0 -> C 304.0: +1.33 %" not in t: why.append("A4 cap48")
    if "switched essay (after code) tok/s B 79.20 (79.00/79.40) -> W 79.85 (79.80/79.90): +0.82 %" not in t: why.append("SWL")
    if "c8 (ncmb3 35 s, P12SPLIT): agg B 185.1 (185.0/185.2) -> Z 189.1 (189.0/189.2): +2.16 %" not in t: why.append("ALL c8")
    if "agg B 218.0 -> Z 224.0: +2.75 %" not in t: why.append("ALL c16")
    if "tok/s B 90.60 (90.50/90.70) -> Z 91.60 (91.50/91.70): +1.10 %" not in t: why.append("ALL 1u")
    if "TTFT B 24.60 s -> Z 24.10 s: -2.03 %" not in t or "INFO int12 24.57 s" not in t: why.append("cold")
    if "8 of 8 IDENTICAL" not in t or "STOPPED" in t: why.append("identity")
    # the stop path: a1-Z differs -> STOPPED, no speed rows read as such
    open(os.path.join(d, "identity.txt"), "w").write("yzi-a1-Z-vs-ref12-a1 DIFFERENT at char 40 (100 vs 100 chars)\nyzi-a1-B-vs-ref12-a1 IDENTICAL: 20 of 20\n")
    for p in ("mb.jsonl", "c16cap.jsonl", "blocks.jsonl", "cold.jsonl"): os.remove(os.path.join(d, p))
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): run([d])
    t2 = buf2.getvalue()
    if "STOPPED at the identity gate" not in t2 or "a1-Z-vs-ref12-a1 DIFFERENT" not in t2 or "1 of 2 IDENTICAL -- VIRHE" not in t2: why.append("stop path")
    if why:
        print(t); print(t2); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: A9 / A11 c8 newcomer + guard, A4 c16 fixed + CAP48, SWL switched / warm essay, ALL c8 / c16 / 1u / cold vs B (+ int12 INFO), identity gate, the stop path")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
