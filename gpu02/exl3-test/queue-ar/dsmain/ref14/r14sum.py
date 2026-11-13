#!/usr/bin/env python3
"""r14sum.py (12.11.2026, queue-ar/dsmain/ref14, a builder for ds-main-port) -- the int14 reference read's summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14.sh, body ref14/dsm-ref14.sh). Standard library only.

  r14sum.py OUTDIR [--sw 1] [--essay-md5 47a39dac581a] [--code-md5 c5e7c82e3e11] [--gens 2]
      OUTDIR/identity.txt (keys xref14-*), blocks.jsonl (resprobe reads: lever U1W / U1 = the warm essay set, XC / XE = code then the
      switched essay; arm O), mb.jsonl (mbx rows: c1-<i>-S-O, c4-w<i>-S-wO (discarded) / c4-<i>-S-O, c8-<i>-S-O (35 s, INFO),
      c8-<i>-G-<B|Z> (the work-based gens window, in-process B Z Z B), c16-<i>-S-O), split-<block>.txt (p12win.py split: (a) decode
      ms a step ONLY -- its newcomer mean is not used), rows-<block>.json (the step-log row buckets), ncm-<set>.json (ncmatch.py: the
      matched-prompt newcomer sums: c8g, c8t, c16, c4, c1), c16cap.jsonl, cold.jsonl.
      Lines "REF14 ...": every reading with its value, the spread (two or more blocks / reads), the target, the int12 references
      (prof1 / ref12 / res12 / yzint14c B, other starts: INFO with the % difference) and yzint14c's ALL arm Z (the provisional
      int14 zero point, INFO); a TABLE line and a headline.
  r14sum.py selftest"""
import json, os, re, statistics, sys, tempfile

DEC = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([\d.]+) %, ([\d.]+|-) ms a step, ([\d.]+|-) rows a step")
NAME = re.compile(r"^c(\d+)-(w?\d+)-([SG])-(w?[A-Z]+)$")
TARGET = dict(u1=90.0, c4=160.0, c8=200.0, cold136=35.0)
# the int12 references (other starts; INFO only, never re-run) and yzint14c's Z (the provisional int14 zero point). Sources:
# prof1 = RESULT xp1 (int12, copy path, MAXSEQS 16); ref12 = RESULT xref12 (int12, copy path, MAXSEQS 8); res12 = RESULT xres
# BASE-lf0 (int12, ngshare, MAXSEQS 16); xpu = the XPUSH screen's B (int13 at ar_xpush 0); yzint14c = RESULT yzi (int14c, ngshare,
# MAXSEQS 16): B = every lever at int12's value, Z = ALL; its c8 newcomer matched sums are ncmatch.py's on its step log
# (c8-11/14-S-B vs c8-12/13-S-Z, 13 prompts: 533.8 / 521.7 ms a newcomer step).
REF = {
    "u1": [("prof1", 91.01), ("res12 warm", 90.58), ("yzint14c B", 89.22)], "u1Z": 89.65,
    "sw": [("ref12", 82.24), ("res12", 81.66), ("yzint14c B", 78.72)], "swW": ("yzint14c W (SWL)", 81.00),
    "c1": [("ref12", 124.90)], "c4": [("ref12", 173.20)], "c4p50": [("ref12", 57.65)],
    "c8": [("prof1", 184.0), ("ref12", 184.50), ("res12", 183.10), ("yzint14c B", 185.3)], "c8Z": 185.6,
    "c8nc": [("yzint14c B", 533.8)], "c8ncZ": 521.7, "c8dec": [("prof1", 36.18), ("yzint14c B", 35.01)], "c8decZ": 34.62,
    "c8p50": [("prof1", 34.10), ("yzint14c B", 34.59)], "c8p50Z": 34.14,
    "c16": [("res12", 217.8), ("yzint14c B", 219.4)], "c16Z": 229.0, "c16dec": [("res12", 57.75), ("yzint14c B", 57.32)], "c16decZ": 56.58,
    "c16p50": [("yzint14c B", 52.52)], "c16p50Z": 51.83,
    "cap": [("prof1", 300.7), ("yzint14c B", 296.7)], "capC": ("yzint14c C (A4)", 304.7),
    "cold64": [("ref12", 11.93)], "cold136": [("ref12", 24.66), ("xpu B", 24.57), ("yzint14c B", 25.03)], "cold136Z": 24.57,
}


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


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def spread(v):
    v = [x for x in v if x is not None]
    return 100.0 * (max(v) - min(v)) / statistics.mean(v) if len(v) > 1 and statistics.mean(v) else None


def pct(c, b):
    return 100.0 * (c / b - 1.0) if (c is not None and b) else None


def f(v, nd=2):
    return ("%." + str(nd) + "f") % v if v is not None else "-"


def fp(v):
    return "%+.2f %%" % v if v is not None else "-"


def sl(v, nd=1):
    return "/".join(f(z, nd) for z in v) or "-"


def sp(v):
    s = spread(v)
    return ("spread %.2f %%" % s) if s is not None else "spread - (one)"


def refs(key, val, nd=2):
    out = ["%s %s (%s)" % (lb, f(r, nd), fp(pct(val, r))) for lb, r in REF.get(key, [])]
    return "INFO int12: " + (", ".join(out) or "-")


def zref(key, val, nd=2, lab="yzint14c Z"):
    r = REF.get(key)
    return "| INFO %s (provisional zero point) %s (%s)" % (lab, f(r, nd), fp(pct(val, r))) if r is not None else ""


def tgt(val, t, lower=False):
    if val is None: return "target %s %g: -" % ("<=" if lower else ">=", t)
    ok = val <= t if lower else val >= t
    return "target %s %g: %s (%s)" % ("<=" if lower else ">=", t, "MET" if ok else "%.2f short" % abs(val - t), fp(pct(val, t)))


def split_dec(o, nm):
    try: txt = open(os.path.join(o, "split-%s.txt" % nm)).read()
    except OSError: return None
    l = next((x for x in txt.splitlines() if x.startswith("P12SPLIT ") and "VIRHE" not in x), None)
    d = DEC.search(l) if l else None
    return fnum(d.group(4)) if d else None


def p50(o, nm):
    """the dominant row bucket of the block's decode steps -> (bucket, p50 ms, share)"""
    d = jload(os.path.join(o, "rows-%s.json" % nm))
    try: w = list(d["windows"].values())[0]["buckets"]
    except (TypeError, KeyError, IndexError, AttributeError): return None
    b = max(w.items(), key=lambda kv: kv[1].get("n") or 0)
    return (b[0], b[1].get("p50"), b[1].get("share")) if (b[1].get("n") or 0) > 0 else None


def mbset(o, n, sched, arm=None):
    out = []
    for r in jl(os.path.join(o, "mb.jsonl")):
        m = NAME.match(r.get("name") or "")
        if not m or int(m.group(1)) != n or m.group(3) != sched: continue
        if arm is not None and m.group(4) != arm: continue
        out.append(dict(name=r["name"], arm=m.group(4), r=r, dec=split_dec(o, r["name"]), p50=p50(o, r["name"])))
    return out


def ncm(o, label):
    return jload(os.path.join(o, "ncm-%s.json" % label))


def ncm_arm(d, a):
    return ((d or {}).get("arms") or {}).get(a) or {}


def ncm_say(d, arms):
    if not d: return "newcomer matched-prompt sum: not read (no ncm json)"
    k = len(d.get("matched") or [])
    s = []
    for a in arms:
        r = ncm_arm(d, a)
        if not r: continue
        s.append("%s %s ms (%s; %s ms a newcomer step%s)%s" % (a, f(r.get("mean_sum_ms"), 1), sl(r.get("sums") or []), f(r.get("mean_step_ms"), 1),
                 (", spread %.2f %%" % r["spread_pct"]) if r.get("spread_pct") is not None else "",
                 (" -> %s vs %s %+.1f ms (%+.2f %%, %+.2f ms a step)" % (a, r["vs"], r["d_ms"], r["d_pct"], r["d_step_ms"])) if "vs" in r else ""))
    return "newcomer matched-prompt sum (ncmatch %s: %d prompts in every block's window, unclipped walls) %s%s" % (
        d.get("label"), k, " | ".join(s) or "-", (" -- VIRHE fewer than %d matched prompts" % (d.get("min") or 4)) if k < (d.get("min") or 4) else "")


def idmap(o):
    try: L = [l.rstrip("\n").split(" ", 1) for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: L = []
    out = {}
    for x in L:
        k, v = (x[0], x[1] if len(x) > 1 else "")
        if not k.startswith("xref14-"): continue
        v = re.sub(r"^warm-up ", "", v.strip())
        out[k[7:]] = "IDENTICAL" if (v.startswith("IDENTICAL") and "DIFFER" not in v and "differ" not in v) else ("DIFFERENT: " + v[:120])
    return out


def reads(o, levers, arm="O"):
    R = [r for r in jl(os.path.join(o, "blocks.jsonl")) if r.get("lever") in levers and r.get("arm") == arm]
    j = lambda k: [(r.get("j") or {}).get(k) for r in R]
    s = lambda k: [(r.get("st") or {}).get(k) for r in R]
    return dict(n=len(R), tps=j("decode_tps"), ms=j("ms_per_decode_step"), tok=j("tokens_per_step"), md5=j("text_md5"), mb=s("stream_mb_step"),
                pro=s("promotions"), m_tps=mean(j("decode_tps")), m_ms=mean(j("ms_per_decode_step")), m_tok=mean(j("tokens_per_step")), m_mb=mean(s("stream_mb_step")))


def md5say(v, want, sw):
    got = sorted(set(x for x in v if x))
    if not got: return "answers: not read"
    if not sw: return "answers %s (INFO, R14_SW=0 mock: yzint14c's %s is a real-run reference)" % ("/".join(got), want)
    return ("answers IDENTICAL to yzint14c's (%s)" % want) if got == [want] else ("VIRHE answers %s, yzint14c's %s (strict, T 0 greedy)" % ("/".join(got), want))


def run(argv):
    o = argv[0]
    kv = dict(zip(argv[1::2], argv[2::2]))
    sw = kv.get("--sw", "1") == "1"; emd5 = kv.get("--essay-md5", "47a39dac581a"); cmd5 = kv.get("--code-md5", "c5e7c82e3e11"); gens = kv.get("--gens", "2")
    out, row = [], []
    ids = idmap(o)
    gate = ["n5", "a1-O-vs-yzint14c-a1-Z", "a1-O-vs-ref12-a1", "gs16-O-vs-a1-O"] + sorted(k for k in ids if re.match(r"cold-1[0-9]{3}-vs-yzint14c-Z$", k)) + ["ld-O-vs-a1-O"]
    stopped = any(ids.get(k) not in (None, "IDENTICAL") for k in gate) or not all(k in ids for k in ("n5", "a1-O-vs-yzint14c-a1-Z", "gs16-O-vs-a1-O", "ld-O-vs-a1-O"))
    out.append("REF14 IDENTITY GATE (strict, T 0 greedy; before every speed block): %s" % " | ".join("%s %s" % ("n5 warm-up vs stock" if k == "n5" else k, ids.get(k, "not run")) for k in gate))
    # ---- 1u
    u, uw = reads(o, ("U1",)), reads(o, ("U1W",))
    if u["n"]:
        out.append("REF14 1u essay warm (one discarded + %d back to back, resprobe essay 512, T 0): %s tok/s (reads %s, %s), %s ms a step, %s tok a step, stream %s MB a step "
                   "(reads %s), promotions a read %s | discarded warm-up (the residency switch after the identity loads) %s tok/s at %s MB a step | %s | %s | %s %s" % (
                       u["n"], f(u["m_tps"]), sl(u["tps"], 2), sp(u["tps"]), f(u["m_ms"]), f(u["m_tok"]), f(u["m_mb"], 1), sl(u["mb"]), "/".join(str(x) for x in u["pro"]),
                       sl(uw["tps"], 2), sl(uw["mb"]), md5say(u["md5"] + uw["md5"], emd5, sw), tgt(u["m_tps"], TARGET["u1"]), refs("u1", u["m_tps"]), zref("u1Z", u["m_tps"])))
        row.append("1u %s" % f(u["m_tps"]))
    else: out.append("REF14 1u essay warm: not run%s" % (" (STOPPED at the identity gate)" if stopped else ""))
    s, c = reads(o, ("XE",)), reads(o, ("XC",))
    if s["n"]:
        w = REF["swW"]
        out.append("REF14 1u switched essay (right after a code read, x%d): %s tok/s (reads %s, %s), %s ms a step, stream %s MB a step (reads %s) | code reads (INFO) %s tok/s (%s), "
                   "%s MB a step | essay %s; code %s | %s | INFO %s %.2f (%s)" % (
                       s["n"], f(s["m_tps"]), sl(s["tps"], 2), sp(s["tps"]), f(s["m_ms"]), f(s["m_mb"], 1), sl(s["mb"]), f(c["m_tps"]), sl(c["tps"], 2), f(c["m_mb"], 1),
                       md5say(s["md5"], emd5, sw), md5say(c["md5"], cmd5, sw), refs("sw", s["m_tps"]), w[0], w[1], fp(pct(s["m_tps"], w[1]))))
        row.append("1u switched %s" % f(s["m_tps"]))
    else: out.append("REF14 1u switched essay: not run")
    # ---- c8: the work-based gens set (B Z Z B) = the c8 reading; the 35 s blocks INFO
    g = mbset(o, 8, "G"); gB, gZ = [x for x in g if x["arm"] == "B"], [x for x in g if x["arm"] == "Z"]
    if gZ:
        aB, aZ = [x["r"].get("agg_dec_tps") for x in gB], [x["r"].get("agg_dec_tps") for x in gZ]
        dB, dZ = [x["dec"] for x in gB], [x["dec"] for x in gZ]
        pB, pZ = [(x["p50"] or (None, None, None))[1] for x in gB], [(x["p50"] or (None, None, None))[1] for x in gZ]
        mZ = mean(aZ)
        out.append("REF14 c8 WORK-BASED (harness --window gens --gens %s: a whole number of newcomer steps, both edges in decode-only time; in-process B Z Z B, B = every int14 "
                   "lever at 0 = int12 bytes, Z = the int14 pin): agg B %s (%s, %s) -> Z %s (%s, %s): %s | %s | decode ms a step (P12SPLIT (a)) B %s -> Z %s: %s | "
                   "row-bucket p50 B %s -> Z %s: %s | decode_share B %s -> Z %s | windows %s s | %s | %s %s" % (
                       gens, f(mean(aB), 1), sl(aB), sp(aB), f(mZ, 1), sl(aZ), sp(aZ), fp(pct(mZ, mean(aB))), ncm_say(ncm(o, "c8g"), ("B", "Z")),
                       f(mean(dB)), f(mean(dZ)), fp(pct(mean(dZ), mean(dB))), f(mean(pB), 3), f(mean(pZ), 3), fp(pct(mean(pZ), mean(pB))),
                       sl([x["r"].get("decode_share") for x in gB], 3), sl([x["r"].get("decode_share") for x in gZ], 3), sl([x["r"].get("window_s") for x in g]),
                       tgt(mZ, TARGET["c8"]), refs("c8", mZ, 1), zref("c8Z", mZ, 1)))
        row.append("c8 (gens) Z %s vs B %s" % (f(mZ, 1), fp(pct(mZ, mean(aB)))))
    else: out.append("REF14 c8 WORK-BASED (gens, B Z Z B): not run%s" % (" (STOPPED at the identity gate)" if stopped else ""))
    t8 = mbset(o, 8, "S", "O")
    if t8:
        a = [x["r"].get("agg_dec_tps") for x in t8]; d_ = [x["dec"] for x in t8]; p_ = [(x["p50"] or (None, None, None))[1] for x in t8]
        nc = ncm(o, "c8t"); r = ncm_arm(nc, "O")
        out.append("REF14 c8 35 s (INFO: the older numbers' window; it holds 13 or 14 newcomers, so its agg carries a ~1 %% newcomer quantum): agg %s (%s, %s) | %s | decode ms a "
                   "step %s | 5-8 row p50 %s | %s %s | INFO yzint14c's 13-prompt matched newcomer B %s / Z %s ms a step: O %s / %s" % (
                       f(mean(a), 1), sl(a), sp(a), ncm_say(nc, ("O",)), f(mean(d_)), f(mean(p_), 3), refs("c8", mean(a), 1), zref("c8Z", mean(a), 1),
                       f(REF["c8nc"][0][1], 1), f(REF["c8ncZ"], 1), fp(pct(r.get("mean_step_ms"), REF["c8nc"][0][1])), fp(pct(r.get("mean_step_ms"), REF["c8ncZ"]))))
        row.append("c8 35 s %s" % f(mean(a), 1))
    else: out.append("REF14 c8 35 s: not run")
    # ---- c16 fixed + CAP48
    c16 = mbset(o, 16, "S", "O")
    if c16:
        a = [x["r"].get("agg_dec_tps") for x in c16]; d_ = [x["dec"] for x in c16]; p_ = [(x["p50"] or (None, None, None))[1] for x in c16]
        out.append("REF14 c16 fixed harness (ncmb3 1x16, cap 48, 35 s): agg %s (%s, %s) | %s | decode ms a step %s (INFO int12: %s) | 16-row p50 %s (INFO yzint14c B %s, Z %s) | %s %s" % (
            f(mean(a), 1), sl(a), sp(a), ncm_say(ncm(o, "c16"), ("O",)), f(mean(d_)), ", ".join("%s %.2f" % kv for kv in REF["c16dec"]) + ", yzint14c Z %.2f" % REF["c16decZ"],
            f(mean(p_), 3), f(REF["c16p50"][0][1]), f(REF["c16p50Z"]), refs("c16", mean(a), 1), zref("c16Z", mean(a), 1)))
        row.append("c16 fixed %s" % f(mean(a), 1))
    else: out.append("REF14 c16 fixed: not run")
    cap = [r for r in jl(os.path.join(o, "c16cap.jsonl")) if r.get("arm") == "O"]
    if cap:
        a = [r.get("agg_dec_tps") for r in cap]
        out.append("REF14 c16 CAP48 method (cap48/ncmb.py phase c16, decode only): agg %s (%s, %s), %s ms a decode step, %s tok a step, steps_gt48 %s | %s | INFO %s %.1f (%s)" % (
            f(mean(a), 1), sl(a), sp(a), f(mean([r.get("ms_decode_step") for r in cap])), f(mean([r.get("tok_per_decode_step") for r in cap])),
            "/".join(str(r.get("gt48")) for r in cap), refs("cap", mean(a), 1), REF["capC"][0], REF["capC"][1], fp(pct(mean(a), REF["capC"][1]))))
        row.append("c16 CAP48 %s" % f(mean(a), 1))
    else: out.append("REF14 c16 CAP48: not run")
    # ---- c4 / c1
    c4 = mbset(o, 4, "S", "O"); c4w = mbset(o, 4, "S", "wO")
    if c4:
        a = [x["r"].get("agg_dec_tps") for x in c4]; acc = [x["r"].get("acceptance") for x in c4]
        p_ = ["%s %s (%s)" % (x["p50"][0], f(x["p50"][1], 3), f(x["p50"][2], 2)) if x["p50"] else "-" for x in c4]
        p16 = [x["p50"][1] for x in c4 if x["p50"] and x["p50"][0] == "16"]
        sa = spread(acc)
        out.append("REF14 c4 (fixed harness, the same prompts each block, %d discarded warm-up, %d block(s)): agg %s (%s, %s) | acceptance %s (%s) | row-bucket p50 a block %s "
                   "(16-row %s; INFO ref12 %.2f: %s) | decode ms a step %s | %s | %s | %s" % (
                       len(c4w), len(c4), f(mean(a), 1), sl(a), sp(a), sl(acc, 3), ("matched, spread %.2f %%" % sa) if sa is not None and sa < 1.0 else
                       (("NOTE mix differs, spread %.2f %%" % sa) if sa is not None else "one block"), "; ".join(p_), f(mean(p16), 3), REF["c4p50"][0][1],
                       fp(pct(mean(p16), REF["c4p50"][0][1])), f(mean([x["dec"] for x in c4])), ncm_say(ncm(o, "c4"), ("O",)), tgt(mean(a), TARGET["c4"]), refs("c4", mean(a), 1)))
        row.append("c4 %s" % f(mean(a), 1))
    else: out.append("REF14 c4: not run (budget, or the identity gate)")
    c1 = mbset(o, 1, "S", "O")
    if c1:
        a = [x["r"].get("agg_dec_tps") for x in c1]
        out.append("REF14 c1 (fixed harness, %d block(s)): agg %s (%s, %s), acceptance %s, %s ms a decode step, TTFT p50 %s s | dominant row bucket %s | %s" % (
            len(c1), f(mean(a), 1), sl(a), sp(a), sl([x["r"].get("acceptance") for x in c1], 3), f(mean([x["r"].get("ms_decode_step") for x in c1])),
            f(mean([x["r"].get("ttft_p50") for x in c1])), "; ".join("%s %s (%s)" % (x["p50"][0], f(x["p50"][1], 3), f(x["p50"][2], 2)) if x["p50"] else "-" for x in c1),
            refs("c1", mean(a), 1)))
        row.append("c1 %s" % f(mean(a), 1))
    else: out.append("REF14 c1: not run (budget, or the identity gate)")
    # ---- cold
    C = [r for r in jl(os.path.join(o, "cold.jsonl")) if (r.get("depth") or 0) > 2000]
    for dep in sorted({r.get("depth") for r in C}):
        x = [r for r in C if r.get("depth") == dep and r.get("status") == 200 and not r.get("cached")]
        t = mean([r.get("ttft") for r in x])
        key = "cold136" if dep == 136000 else ("cold64" if dep == 64000 else None)
        idk = sorted(k for k in ids if k.startswith("cold-%d-vs-" % dep))
        out.append("REF14 cold %dk (deepq after a prefix-cache reset, %s tok): TTFT %s s (%s), %s prefill tok/s, non-decode %s ms a step | answers %s | %s%s%s" % (
            dep // 1000, "/".join(sorted({str(r.get("prompt_tokens")) for r in x})) or "-", f(t), sl([r.get("ttft") for r in x], 2),
            f(mean([(r["prompt_tokens"] - (r.get("cached") or 0)) / r["ttft"] for r in x if r.get("ttft") and r.get("prompt_tokens")]), 0),
            f(mean([r.get("nondecode_ms") for r in x]), 1), ", ".join("%s %s" % (k, ids[k]) for k in idk) or ("INFO (R14_SW=0 mock: real-run references not compared)" if not sw else "not compared"),
            (tgt(t, TARGET["cold136"], lower=True) + " | ") if dep == 136000 else "", refs(key, t) if key else "INFO int12: -", (" " + zref("cold136Z", t)) if dep == 136000 else ""))
        row.append("cold %dk %s s" % (dep // 1000, f(t)))
    if not C: out.append("REF14 cold: not run")
    nid = len(ids); nok = sum(1 for v in ids.values() if v == "IDENTICAL")
    out.append("REF14 identity (strict, T 0 greedy): %d of %d IDENTICAL%s" % (nok, nid, (" -- VIRHE differ / not compared: %s" % ", ".join(k for k, v in ids.items() if v != "IDENTICAL")) if nok < nid else ""))
    out.append("REF14 TABLE " + " || ".join(row) if row else "REF14 TABLE no speed reading%s" % (" (STOPPED at the identity gate: NEEDS USER)" if stopped else ""))
    out.append("REF14 headline: int14 zero point -- %s | identity %d / %d%s" % ("; ".join(row) or "no speed reading", nok, nid, " | STOPPED at the identity gate (no speed block): NEEDS USER" if stopped else ""))
    for l in out: print(l)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="r14sum-")
    mb = []
    def blk(nm, n, arm, agg, dec, acc=0.7, bucket=("5-8", 34.1)):
        mb.append(dict(name=nm, n=n, arm=arm, sched=nm.split("-")[2], agg_dec_tps=agg, decode_share=0.8, ms_decode_step=dec, acceptance=acc, window_s=45.9, ttft_p50=0.6, errors=0))
        open(os.path.join(d, "split-%s.txt" % nm), "w").write(
            "P12SPLIT %s (window 35.0 s): (a) decode 774 steps 28003 ms = 80.0 %%, %.2f ms a step, 8.47 rows a step, 8.32 tokens a step | (b) newcomer / MIXED 14 steps "
            "6997 ms = 20.0 %%, 1114 rows a step, 499.8 ms a step | (c) host gaps 0 ms\n" % (nm, dec))
        json.dump({"windows": {nm: {"buckets": {bucket[0]: {"n": 700, "share": 0.9, "p50": bucket[1]}, ">16": {"n": 9, "share": 0.01, "p50": 160.0}}}}},
                  open(os.path.join(d, "rows-%s.json" % nm), "w"))
    for i, (a, agg, dec) in enumerate((("B", 185.0, 35.0), ("Z", 187.5, 34.6), ("Z", 187.7, 34.6), ("B", 185.2, 35.0)), 1):
        blk("c8-%d-G-%s" % (i, a), 8, a, agg, dec)
    blk("c8-1-S-O", 8, "O", 185.6, 34.6); blk("c8-2-S-O", 8, "O", 187.4, 34.6)
    blk("c16-1-S-O", 16, "O", 228.0, 56.6, bucket=("16", 51.8)); blk("c16-2-S-O", 16, "O", 229.4, 56.5, bucket=("16", 51.9))
    blk("c4-w1-S-wO", 4, "wO", 170.0, 62.0, bucket=("16", 57.0)); blk("c4-1-S-O", 4, "O", 175.0, 62.5, 0.71, ("16", 57.4)); blk("c4-2-S-O", 4, "O", 174.0, 62.6, 0.705, ("16", 57.5))
    blk("c1-1-S-O", 1, "O", 126.0, 26.9, 0.777, ("5-8", 29.6))
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in mb: fo.write(json.dumps(r) + "\n")
    json.dump(dict(label="c8g", matched=[[i, 0] for i in range(16)], arms=dict(B=dict(sums=[8540.0, 8542.0], mean_sum_ms=8541.0, mean_step_ms=533.8, spread_pct=0.02),
              Z=dict(sums=[8346.0, 8348.0], mean_sum_ms=8347.0, mean_step_ms=521.7, spread_pct=0.02, vs="B", d_ms=-194.0, d_pct=-2.27, d_step_ms=-12.13))), open(os.path.join(d, "ncm-c8g.json"), "w"))
    json.dump(dict(label="c8t", matched=[[i, 0] for i in range(13)], arms=dict(O=dict(sums=[6782.0, 6780.0], mean_sum_ms=6781.0, mean_step_ms=521.6, spread_pct=0.03))),
              open(os.path.join(d, "ncm-c8t.json"), "w"))
    with open(os.path.join(d, "c16cap.jsonl"), "w") as fo:
        fo.write(json.dumps(dict(name="c16cap-1-O", arm="O", agg_dec_tps=305.0, ms_decode_step=52.4, tok_per_decode_step=16.0, gt48="0")) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        for lv, tps, ms, mb_, md in (("U1W", 86.0, 18.3, 60.0, "e"), ("U1", 89.6, 17.6, 1.0, "e"), ("U1", 89.7, 17.6, 0.0, "e"), ("U1", 89.65, 17.6, 0.0, "e"),
                                     ("XC", 131.0, 28.0, 360.0, "c"), ("XE", 81.0, 19.5, 135.0, "e"), ("XC", 132.0, 27.9, 350.0, "c"), ("XE", 81.1, 19.4, 136.0, "e")):
            fo.write(json.dumps(dict(lever=lv, arm="O", j=dict(decode_tps=tps, ms_per_decode_step=ms, tokens_per_step=1.58, text_md5=md), st=dict(stream_mb_step=mb_, promotions=5))) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo:
        for dep, t in ((1100, 0.4), (64000, 11.7), (136000, 24.6)):
            fo.write(json.dumps(dict(name="cold-%d-1-O" % dep, depth=dep, arm="O", ttft=t, prompt_tokens=dep + 1135, cached=0, status=200, nondecode_ms=420.0)) + "\n")
    open(os.path.join(d, "identity.txt"), "w").write("xref14-n5 warm-up IDENTICAL: 3 answers\n" + "".join("xref14-%s IDENTICAL: 20 of 20 answers\n" % k for k in (
        "a1-O-vs-yzint14c-a1-Z", "a1-O-vs-ref12-a1", "gs16-O-vs-a1-O", "cold-1100-vs-yzint14c-Z", "ld-O-vs-a1-O", "cold-64000-vs-ref12", "cold-136000-vs-yzint14c-Z")))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--essay-md5", "e", "--code-md5", "c"])
    t = buf.getvalue()
    why = []
    if "agg B 185.1 (185.0/185.2, spread 0.11 %) -> Z 187.6 (187.5/187.7, spread 0.11 %): +1.35 %" not in t: why.append("c8 gens")
    if "B 8541.0 ms (8540.0/8542.0; 533.8 ms a newcomer step, spread 0.02 %) | Z 8347.0 ms" not in t or "Z vs B -194.0 ms (-2.27 %, -12.13 ms a step)" not in t: why.append("c8 matched")
    if "499.8" in t: why.append("P12SPLIT newcomer mean used")
    if "target >= 200: 12.40 short" not in t: why.append("c8 target")
    if "1u essay warm (one discarded + 3 back to back" not in t or "89.65 tok/s (reads 89.60/89.70/89.65" not in t or "answers IDENTICAL to yzint14c's (e)" not in t: why.append("1u")
    if "switched essay (right after a code read, x2): 81.05" not in t: why.append("switched")
    if "c4 (fixed harness, the same prompts each block, 1 discarded warm-up, 2 block(s)): agg 174.5" not in t or "matched, spread 0.71 %" not in t or "16-row 57.450" not in t: why.append("c4")
    if "c16 fixed harness (ncmb3 1x16, cap 48, 35 s): agg 228.7" not in t or "c16 CAP48 method (cap48/ncmb.py phase c16, decode only): agg 305.0" not in t: why.append("c16")
    if "REF14 cold 136k" not in t or "target <= 35: MET" not in t or "cold-136000-vs-yzint14c-Z IDENTICAL" not in t: why.append("cold")
    if "8 of 8 IDENTICAL" not in t or "STOPPED" in t: why.append("identity")
    open(os.path.join(d, "identity.txt"), "w").write("xref14-n5 warm-up IDENTICAL: 3 answers\nxref14-a1-O-vs-yzint14c-a1-Z DIFFERENT at char 40 (100 vs 100 chars)\n")
    for p in ("mb.jsonl", "c16cap.jsonl", "blocks.jsonl", "cold.jsonl"): os.remove(os.path.join(d, p))
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2): run([d])
    t2 = buf2.getvalue()
    if "STOPPED at the identity gate" not in t2 or "a1-O-vs-yzint14c-a1-Z DIFFERENT" not in t2 or "1 of 2 IDENTICAL -- VIRHE" not in t2: why.append("stop path")
    if why:
        print(t); print(t2); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: c8 work-based B Z Z B (agg, matched-prompt newcomer sum, decode, bucket, target), c8 35 s INFO, c16 fixed + CAP48, c4 matched mix + 16-row p50, c1, 1u warm + switched (answers), cold + target, identity gate, the stop path")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
