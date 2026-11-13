#!/usr/bin/env python3
"""ref12sum.py (12.11.2026, queue-ar/dsmain/ref12, the int11 builder) -- the int12 reference screen's summary
(57-p4zyzzzzz-ds-r92zzzzzz-ref12.sh, body dsm-ref12.sh). Standard library only.

  ref12sum.py summary --blocks B.jsonl --steps STEPS.json --mb MB.jsonl --cold COLD.jsonl --idf IDENTITY.txt --i11 I11DIR
      the 1u reads (lever U1, arm O; warm-up reads U1W left out): per prompt (essay / code) the mean tok/s of the warm reads, tokens
      a step, ms a step (stepprobe), stream MB / h2d MB a step and promotions a read, ms a step by verify rows (u1sum.py steps);
      the MX12 confirm pair (lever MXC: M0 vs O); mb c1 / c4 / c8 (arm O, warm-ups "w*" left out); cold 64k / 136k TTFT; identity;
      the TABLE against the targets (1u essay >= 90 tok/s, c4 >= 160, c8 >= 200, 136k TTFT <= 35 s) and int11's O numbers (the
      int11 1u screen's runs/xdsm/i11u: summary.txt "U1 BASE", mb.jsonl arm O). Lines "REF12 ..." and "SUMMARY NEW DS REFERENCE (int12) ...".
  ref12sum.py selftest"""
import io, json, os, re, statistics, sys, tempfile

TARGETS = {"1u essay": (90.0, ">="), "c4": (160.0, ">="), "c8": (200.0, ">="), "cold 136k": (35.0, "<=")}
ROWCLS = ["1", "2", "3", "4+"]


def rd(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except ValueError: pass
    except (OSError, TypeError): pass
    return out


def opt(a, k):
    return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def f(v, fm="%.2f"):
    return fm % v if v is not None else "-"


def pct(x, y):
    return 100.0 * (x / y - 1) if x is not None and y else None


def i11refs(d):
    r = {}
    try:
        s = open(os.path.join(d, "summary.txt")).read()
        m = re.search(r"U1 BASE[^\n]*?essay ([0-9.]+) tok/s", s); r["1u essay"] = float(m.group(1)) if m else None
        m = re.search(r"U1 BASE[^\n]*?code ([0-9.]+) tok/s", s); r["1u code"] = float(m.group(1)) if m else None
    except OSError: pass
    mb = [x for x in rd(os.path.join(d, "mb.jsonl")) if x.get("arm") == "O" and x.get("sched", "S") == "S" and x.get("agg_dec_tps")]
    for n, k in ((1, "c1"), (4, "c4"), (8, "c8")):
        r[k] = mean([x["agg_dec_tps"] for x in mb if x.get("n") == n])
    return r


def summary(a):
    B = rd(opt(a, "--blocks")); MB = rd(opt(a, "--mb")); C = rd(opt(a, "--cold"))
    try: ST = json.load(open(opt(a, "--steps")))
    except (OSError, ValueError, TypeError): ST = {}
    I = i11refs(opt(a, "--i11") or "")
    out, val = [], {}
    for p in ("essay", "code"):
        rs = [b for b in B if b.get("lever") == "U1" and b.get("arm") == "O" and (b.get("j") or {}).get("prompt") == p]
        if not rs: out.append("REF12 1u %s: no warm read" % p); continue
        tps = [b["j"].get("decode_tps") for b in rs]
        by = {c: [] for c in ROWCLS}
        for b in rs:
            for c, v in ((ST.get(str(b["i"])) or {}).get("_raw") or {}).items(): by.setdefault(c, []).extend(v)
        val["1u " + p] = mean(tps)
        out.append("REF12 1u %s (stepprobe %s 512, T 0; %d warm reads after one discarded): %s tok/s (reads %s), %s tok/step, %s ms/step | stream %s MB / h2d %s MB a step, promotions %s a read | step log ms by verify rows %s" % (
            p, p, len(rs), f(mean(tps)), "/".join(f(x) for x in tps), f(mean(b["j"].get("tokens_per_step") for b in rs), "%.3f"),
            f(mean(b["j"].get("ms_per_decode_step") for b in rs)), f(mean((b.get("st") or {}).get("stream_mb_step") for b in rs), "%.1f"),
            f(mean((b.get("st") or {}).get("h2d_mb_step") for b in rs), "%.1f"), "/".join(str((b.get("st") or {}).get("promotions")) for b in rs),
            " ".join("%s:%s" % (c, f(statistics.mean(by[c])) if by.get(c) else "-") for c in ROWCLS)))
    mx = [b for b in B if b.get("lever") == "MXC"]
    if mx:
        e = {b["arm"]: b for b in mx}
        o, z = e.get("O"), e.get("M0")
        if o and z:
            out.append("REF12 MX12 confirm (one warm essay pair, same start): mix_wv %s %s tok/s (stream %s MB a step) vs mix_wv 0 %s tok/s (stream %s MB a step): %s" % (
                (o.get("pin") or {}).get("ds41_mix_wv", "base"), f(o["j"].get("decode_tps")), f((o.get("st") or {}).get("stream_mb_step"), "%.1f"),
                f(z["j"].get("decode_tps")), f((z.get("st") or {}).get("stream_mb_step"), "%.1f"), "%+.2f %%" % pct(o["j"].get("decode_tps"), z["j"].get("decode_tps")) if z["j"].get("decode_tps") else "-"))
    for n, k in ((1, "c1"), (4, "c4"), (8, "c8")):
        xs = [x for x in MB if x.get("n") == n and x.get("arm") == "O" and x.get("agg_dec_tps")]
        if not xs: out.append("REF12 mb %s: no block" % k); continue
        val[k] = mean(x["agg_dec_tps"] for x in xs)
        out.append("REF12 mb %s (fixed harness, the SAME prompts each block (mbt3 --prompt-set fixed, reset per block), %d block(s)%s): %s tok/s (blocks %s), acceptance %s (blocks %s), %s tok / %s ms a decode step, per-stream p50 %s, TTFT p50 %s / p90 %s s, decode share %s" % (
            k, len(xs), ", after one discarded warm-up" if n == 4 else "", f(val[k]), "/".join(f(x["agg_dec_tps"]) for x in xs),
            f(mean(x.get("acceptance") for x in xs), "%.3f"), "/".join(f(x.get("acceptance"), "%.3f") for x in xs), f(mean(x.get("tok_per_decode_step") for x in xs)),
            f(mean(x.get("ms_decode_step") for x in xs)), f(mean(x.get("per_stream_dec_p50") for x in xs)),
            f(mean(x.get("ttft_p50") for x in xs)), f(mean(x.get("ttft_p90_win") for x in xs)), f(mean(x.get("decode_share") for x in xs), "%.3f")))
        acc = [x.get("acceptance") for x in xs if x.get("acceptance") is not None]
        if n in (1, 4) and len(acc) >= 2:
            dv = 100.0 * (max(acc) - min(acc)) / statistics.mean(acc) if statistics.mean(acc) else None
            out.append("REF12 %s mix check (TESTING.md 12.11. c4 METHOD FIX: the blocks' acceptance must differ by < 1 %%): acceptance %s, spread %s -> %s" % (
                k, "/".join(f(x, "%.3f") for x in acc), "%.2f %%" % dv if dv is not None else "-", "matched" if dv is not None and dv < 1.0 else "NOTE mix not matched"))
    for c in C:
        dep = c.get("depth"); k = "cold 136k" if dep and dep >= 100000 else "cold 64k"
        cold_ok = c.get("status") == 200 and not c.get("cached")
        if cold_ok: val[k] = c.get("ttft")
        out.append("REF12 %s (deepq %s, cold after a reset): TTFT %s s, prompt %s tok, cached %s, status %s%s" % (k, dep, f(c.get("ttft")), c.get("prompt_tokens"), c.get("cached"), c.get("status"),
                   "" if cold_ok else " -- not a cold reading"))
    idl = []
    try:
        for l in open(opt(a, "--idf")): idl.append(l.strip())
    except (OSError, TypeError): pass
    for l in idl: out.append("REF12 identity %s" % l[:300])
    rows = []
    for k in ("1u essay", "1u code", "c1", "c4", "c8", "cold 64k", "cold 136k"):
        lab = "c4 (warm, matched-mix: the same prompts each block)" if k == "c4" else k
        v = val.get(k); t = TARGETS.get(k); r = I.get(k)
        tt = "-"
        if t and v is not None:
            ok = v >= t[0] if t[1] == ">=" else v <= t[0]
            tt = "%s %s: %s (%s)" % (t[1], f(t[0], "%g"), "MET" if ok else "%s short" % f(abs(v - t[0])), "%+.1f %%" % pct(v, t[0]))
        rows.append("%s %s %s | target %s | int11 O %s (%s)" % (lab, f(v), "s" if k.startswith("cold") else "tok/s", tt, f(r), "%+.2f %%" % pct(v, r) if (v is not None and r) else "-"))
    for l in out: print(l)
    for l in rows: print("REF12 TABLE " + l)
    print("SUMMARY NEW DS REFERENCE (int12): " + " || ".join(rows))
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="ref12sum-")
    try:
        os.makedirs(os.path.join(d, "i11"))
        open(os.path.join(d, "i11", "summary.txt"), "w").write("x U1 BASE (the O blocks, ..): essay 77.69 tok/s (blocks ..) | code 127.76 tok/s (blocks ..)\n")
        with open(os.path.join(d, "i11", "mb.jsonl"), "w") as g:
            for n, v in ((1, 110.9), (4, 162.6), (4, 157.7), (8, 180.4)): g.write(json.dumps(dict(n=n, arm="O", sched="S", agg_dec_tps=v)) + "\n")
            g.write(json.dumps(dict(n=4, arm="wO", sched="S", agg_dec_tps=99.0)) + "\n")
        bf = os.path.join(d, "b.jsonl")
        with open(bf, "w") as g:
            for i, (lv, arm, p, t) in enumerate((("U1W", "O", "essay", 70.0), ("U1", "O", "essay", 80.0), ("U1", "O", "code", 130.0), ("U1", "O", "essay", 82.0),
                                                ("MXC", "M0", "essay", 79.0), ("MXC", "O", "essay", 81.0))):
                g.write(json.dumps(dict(i=i, lever=lv, arm=arm, j=dict(prompt=p, decode_tps=t, tokens_per_step=1.6, ms_per_decode_step=20.0),
                                        st=dict(stream_mb_step=100.0, h2d_mb_step=50.0, promotions=10), pin={"ds41_mix_wv": 12} if arm == "O" else {"ds41_mix_wv": 0})) + "\n")
        sj = os.path.join(d, "s.json"); json.dump({"1": {"_raw": {"1": [17.0, 17.2], "2": [20.0]}}, "3": {"_raw": {"1": [17.4]}}}, open(sj, "w"))
        mf = os.path.join(d, "mb.jsonl")
        with open(mf, "w") as g:
            for n, arm, v, ac in ((4, "wO", 140.0, 0.6), (4, "O", 165.0, 0.700), (4, "O", 167.0, 0.703), (8, "O", 190.0, 0.5), (1, "O", 112.0, 0.70), (1, "O", 113.0, 0.75)):
                g.write(json.dumps(dict(n=n, arm=arm, agg_dec_tps=v, acceptance=ac, tok_per_decode_step=9.0, ms_decode_step=60.0)) + "\n")
        cf = os.path.join(d, "c.jsonl")
        with open(cf, "w") as g:
            g.write(json.dumps(dict(depth=64000, ttft=11.5, status=200, cached=0, prompt_tokens=64000)) + "\n")
            g.write(json.dumps(dict(depth=136000, ttft=33.0, status=200, cached=0, prompt_tokens=136000)) + "\n")
        idf = os.path.join(d, "id.txt"); open(idf, "w").write("xref12-a1-vs-int11-O IDENTICAL: 20 of 20 answers\n")
        so = sys.stdout; sink = io.StringIO(); sys.stdout = sink
        try: summary(["summary", "--blocks", bf, "--steps", sj, "--mb", mf, "--cold", cf, "--idf", idf, "--i11", os.path.join(d, "i11")])
        finally: sys.stdout = so
        t = sink.getvalue()
        assert "REF12 1u essay (stepprobe essay 512, T 0; 2 warm reads after one discarded): 81.00 tok/s (reads 80.00/82.00)" in t, t
        assert "1:17.20 2:20.00" in t and "REF12 MX12 confirm" in t and "+2.53 %" in t, t
        assert "c4 (warm, matched-mix: the same prompts each block) 166.00 tok/s | target >= 160: MET (+3.8 %) | int11 O 160.15 (+3.65 %)" in t, t
        assert "REF12 c4 mix check" in t and "-> matched" in t and "REF12 c1 mix check" in t and "NOTE mix not matched" in t, t
        assert "cold 136k 33.00 s | target <= 35: MET" in t and "1u essay 81.00 tok/s | target >= 90: 9.00 short" in t, t
        assert "SUMMARY NEW DS REFERENCE (int12): " in t and "identity xref12-a1-vs-int11-O IDENTICAL" in t, t
        print("SELFTEST OK (summary on synthetic data)")
        return 0
    except AssertionError as e:
        print("SELFTEST FAIL %s" % str(e)[:2000]); return 1
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("summary", "selftest"): print(__doc__); return 2
    return selftest() if a[0] == "selftest" else summary(a)


if __name__ == "__main__":
    sys.exit(main())
