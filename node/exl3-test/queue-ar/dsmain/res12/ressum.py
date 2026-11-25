#!/usr/bin/env python3
"""ressum.py (12.11.2026, queue-ar/dsmain/res12, ds-main-port residency screen) -- the summary of dsm-res12.sh. No GPU, no server.

  ressum.py summary --dir OUT [--base BASE-lf0]   reads OUT/blocks.jsonl (one line a 1u read: lever XCW / XEW / XC / XE / XW, set,
                                                  stepprobe J, /stats deltas), OUT/mb.jsonl (the c8 blocks), OUT/sets.jsonl (set ->
                                                  its c8 block), OUT/c8/<block>-mix.json (c8mix signature) and each read's answer
                                                  file; prints RES12 lines:
      RES12 SET ...      a set: SWITCHED essay (XE: essay after code) tok/s, ms a step, stream / h2d MB a step, promotions,
                         readmits, starved_slab / starved_pool (rank 0's mover), lent fills / fills_lf; WARM essay (XW: essay after
                         essay); the warm-ups (XEW / XCW) and the switched code reads (XC) beside; c8 agg / ms a decode step /
                         decode_share / acceptance / mix
      RES12 VERDICT ...  a set vs the base set: "helps the switch by X % at c8 Y %" -- X = the switched essay's ms a step cut
                         (positive = faster), Y = the c8 agg change (positive = more tok/s); the warm essay and the stream MB beside
      RES12 IDENTITY ... every read's answer vs the first read of the same prompt in the base set (else the first anywhere):
                         IDENTICAL or VIRHE DIFFERENT (placement cannot change a T 0 answer); the switched essays across arms first
      RES12 TABLE ...    one compact line
  ressum.py selftest                               synthetic files in a temp dir; asserts the lines.
"""
import argparse, json, os, sys, tempfile


def jl(p):
    out = []
    try:
        with open(p) as f:
            for l in f:
                l = l.strip()
                if l:
                    try: out.append(json.loads(l))
                    except ValueError: pass
    except OSError:
        pass
    return out


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


def f(v, d=2):
    return ("%%.%df" % d) % v if isinstance(v, (int, float)) else "-"


def pct(new, old, inv=False):
    """% change new vs old; inv: the cut of a time (positive = faster)."""
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or not old:
        return None
    return 100.0 * ((old - new) / old if inv else (new / old - 1.0))


def fp(v):
    return ("%+.1f %%" % v) if isinstance(v, (int, float)) else "-"


def agg_reads(rs):
    j = lambda k: mean([(r.get("j") or {}).get(k) for r in rs])
    s = lambda k: mean([(r.get("st") or {}).get(k) for r in rs])
    return dict(n=len(rs), tps=j("decode_tps"), ms=j("ms_per_decode_step"), tok_step=j("tokens_per_step"), acc=j("acceptance"),
                mb=s("stream_mb_step"), h2d=s("h2d_mb_step"), prom=s("promotions"), dem=s("demotions"), readm=s("readmits"),
                ss=s("starved_slab"), sp=s("starved_pool"), fills=s("fills"), flf=s("fills_lf"),
                tps_each=[(r.get("j") or {}).get("decode_tps") for r in rs], ms_each=[(r.get("j") or {}).get("ms_per_decode_step") for r in rs],
                mb_each=[(r.get("st") or {}).get("stream_mb_step") for r in rs])


def rtext(a):
    def g(x): return "/".join(f(v) for v in x) if x else "-"
    return ("%d read(s) %s tok/s (%s), %s ms a step (%s), %s tok a step, stream %s MB (%s) / h2d %s MB a step, promotions %s, "
            "readmits %s, starved_slab %s / pool %s, lent fills %s (lf %s)") % (
        a["n"], f(a["tps"]), g(a["tps_each"]), f(a["ms"]), g(a["ms_each"]), f(a["tok_step"]), f(a["mb"], 1), g(a["mb_each"]),
        f(a["h2d"], 1), f(a["prom"], 0), f(a["readm"], 0), f(a["ss"], 0), f(a["sp"], 0), f(a["fills"], 0), f(a["flf"], 0))


def load_text(d, r):
    p = r.get("text") or ""
    for q in (p, os.path.join(d, os.path.basename(p))):
        try:
            with open(q) as fh:
                x = json.load(fh).get("reads") or []
            if x: return x[0]
        except (OSError, ValueError):
            continue
    return None


def summary(d, base, out=sys.stdout):
    B = jl(os.path.join(d, "blocks.jsonl")); M = jl(os.path.join(d, "mb.jsonl")); S = jl(os.path.join(d, "sets.jsonl"))
    mb = {m.get("name"): m for m in M}
    order = []
    for s in S:
        if s.get("set") not in order: order.append(s.get("set"))
    for r in B:
        if r.get("set") and r.get("set") not in order: order.append(r.get("set"))
    sets = {}
    for name in order:
        rs = sorted([r for r in B if r.get("set") == name], key=lambda r: r.get("i", 0))
        meta = next((s for s in S if s.get("set") == name), {})
        c8n = meta.get("c8"); c8 = mb.get(c8n) if c8n else None
        mix = None
        if c8n:
            try:
                with open(os.path.join(d, "c8", c8n + "-mix.json")) as fh: mix = json.load(fh).get("signature")
            except (OSError, ValueError): mix = None
        by = lambda lv: [r for r in rs if r.get("lever") == lv]
        sets[name] = dict(meta=meta, sw=agg_reads(by("XE")), warm=agg_reads(by("XW")), ew=agg_reads(by("XEW")), cw=agg_reads(by("XCW")),
                          code=agg_reads(by("XC")), c8=c8, c8n=c8n, mix=mix, n=len(rs))
    for name in order:
        x = sets[name]; c8 = x["c8"] or {}
        print("RES12 SET %s (k %s, start %s, place_lf %s, %d reads): SWITCHED essay (after code) %s || WARM essay (after essay) %s || essay W "
              "(discarded, after code W) %s ms, %s MB a step, starved_slab %s || code W %s ms, %s MB || code after essay %s ms, %s MB a step || c8 %s: "
              "agg %s tok/s, %s ms a decode step, %s tok a step, decode_share %s, acceptance %s, TTFT p50 %s s, %s errors, mix %s" % (
                  name, x["meta"].get("k", "-"), x["meta"].get("start", "-"), x["meta"].get("lf", "-"), x["n"], rtext(x["sw"]), rtext(x["warm"]),
                  f(x["ew"]["ms"]), f(x["ew"]["mb"], 1), f(x["ew"]["ss"], 0), f(x["cw"]["ms"]), f(x["cw"]["mb"], 1), f(x["code"]["ms"]),
                  f(x["code"]["mb"], 1), x["c8n"] or "-", f(c8.get("agg_dec_tps")), f(c8.get("ms_decode_step")), f(c8.get("tok_per_decode_step")),
                  f(c8.get("decode_share"), 3), f(c8.get("acceptance"), 3), f(c8.get("ttft_p50")), c8.get("errors", "-"), json.dumps(x["mix"]) if x["mix"] is not None else "-"), file=out)
    b = sets.get(base)
    if b is None:
        print("RES12 VERDICT: VIRHE no base set %s (sets: %s) -- no verdict" % (base, ", ".join(order) or "none"), file=out)
    else:
        bc = b["c8"] or {}
        for name in order:
            if name == base: continue
            x = sets[name]; c8 = x["c8"] or {}
            xs = pct(x["sw"]["ms"], b["sw"]["ms"], inv=True); y = pct(c8.get("agg_dec_tps"), bc.get("agg_dec_tps"))
            how = "in-process, the same start as the base" if x["meta"].get("start") == b["meta"].get("start") else "cross-start, start-to-start spread ~1-1.5 %"
            mixv = "MATCH" if x["mix"] is not None and x["mix"] == b["mix"] else ("DIFFERS" if x["mix"] is not None and b["mix"] is not None else "n/a")
            print("RES12 VERDICT %s vs %s (%s): switched essay %s -> %s ms a step (%s), %s -> %s tok/s (%s), stream %s -> %s MB a step, "
                  "starved_slab %s -> %s, readmits %s -> %s | warm essay %s -> %s ms a step (%s), stream %s -> %s MB | c8 agg %s -> %s tok/s (%s), "
                  "%s -> %s ms a decode step (%s), decode_share %s -> %s, mix %s%s => helps the switch by %s at c8 %s" % (
                      name, base, how, f(b["sw"]["ms"]), f(x["sw"]["ms"]), fp(xs), f(b["sw"]["tps"]), f(x["sw"]["tps"]), fp(pct(x["sw"]["tps"], b["sw"]["tps"])),
                      f(b["sw"]["mb"], 1), f(x["sw"]["mb"], 1), f(b["sw"]["ss"], 0), f(x["sw"]["ss"], 0), f(b["sw"]["readm"], 0), f(x["sw"]["readm"], 0),
                      f(b["warm"]["ms"]), f(x["warm"]["ms"]), fp(pct(x["warm"]["ms"], b["warm"]["ms"], inv=True)), f(b["warm"]["mb"], 1), f(x["warm"]["mb"], 1),
                      f(bc.get("agg_dec_tps")), f(c8.get("agg_dec_tps")), fp(y), f(bc.get("ms_decode_step")), f(c8.get("ms_decode_step")),
                      fp(pct(c8.get("ms_decode_step"), bc.get("ms_decode_step"), inv=True)), f(bc.get("decode_share"), 3), f(c8.get("decode_share"), 3), mixv,
                      " (c8 delta under 2 % with a different mix: INCONCLUSIVE)" if mixv == "DIFFERS" and y is not None and abs(y) < 2 else "",
                      ("%+.1f %%" % xs) if xs is not None else "- %", ("%+.1f %%" % y) if y is not None else "- %"), file=out)
    # identity: every read's answer vs the first of its prompt in the base set (else the first anywhere)
    allr = sorted(B, key=lambda r: r.get("i", 0))
    bad = []; nread = {}; refs = {}; txt = {}
    for r in allr:
        t = load_text(d, r); p = (r.get("j") or {}).get("prompt")
        if t is None:
            bad.append("%s (no answer file)" % r.get("name")); continue
        txt[r.get("name")] = t; nread[p] = nread.get(p, 0) + 1
        if p not in refs or (refs[p][0] != base and r.get("set") == base):
            refs[p] = (r.get("set"), r.get("name"), t)   # the first read of the prompt in the base set, else the first anywhere
    diff = {}
    for r in allr:
        p = (r.get("j") or {}).get("prompt"); t = txt.get(r.get("name"))
        if t is None or p not in refs: continue
        ref = refs[p][2]
        if (t.get("content") or "") != (ref.get("content") or "") or t.get("finish") != ref.get("finish"):
            a, c = t.get("content") or "", ref.get("content") or ""
            k = next((i for i, (u, v) in enumerate(zip(a, c)) if u != v), min(len(a), len(c)))
            diff.setdefault(p, []).append("%s [%s] at char %d (%d vs %d chars)" % (r.get("name"), r.get("set"), k, len(a), len(c)))
    sw = [r for r in allr if r.get("lever") == "XE"]
    swd = [x for x in diff.get("essay", []) if any(x.startswith(r.get("name") + " ") for r in sw)]
    swsets = sorted(set(r.get("set") for r in sw))
    print("RES12 IDENTITY switched essays across arms (%d reads in %d sets: %s) vs %s: %s" % (
        len(sw), len(swsets), ", ".join(swsets), refs.get("essay", ("-", "-"))[1],
        "IDENTICAL" if sw and not swd and "essay" in refs else ("VIRHE DIFFERENT: " + "; ".join(swd[:6]) if swd else "VIRHE no switched essay answer")), file=out)
    for p in sorted(refs):
        print("RES12 IDENTITY %s (every read, T 0 greedy, %d reads) vs %s [%s]: %s" % (
            p, nread.get(p, 0), refs[p][1], refs[p][0], "IDENTICAL" if not diff.get(p) else "VIRHE DIFFERENT %d: %s" % (len(diff[p]), "; ".join(diff[p][:6]))), file=out)
    if bad:
        print("RES12 IDENTITY: VIRHE %d read(s) without an answer file: %s" % (len(bad), "; ".join(bad[:6])), file=out)
    print("RES12 TABLE set | switched essay tok/s / ms a step / stream MB a step / starved_slab | warm essay tok/s / ms / MB | c8 agg / ms a step || " + " || ".join(
        "%s | %s / %s / %s / %s | %s / %s / %s | %s / %s" % (n, f(sets[n]["sw"]["tps"]), f(sets[n]["sw"]["ms"]), f(sets[n]["sw"]["mb"], 1), f(sets[n]["sw"]["ss"], 0),
                                                            f(sets[n]["warm"]["tps"]), f(sets[n]["warm"]["ms"]), f(sets[n]["warm"]["mb"], 1),
                                                            f((sets[n]["c8"] or {}).get("agg_dec_tps")), f((sets[n]["c8"] or {}).get("ms_decode_step"))) for n in order), file=out)
    return 0


def selftest():
    import io
    d = tempfile.mkdtemp(prefix="ressum-")
    os.makedirs(os.path.join(d, "c8"))
    B, M, S = [], [], []
    i = 0
    plan = [("BASE-lf0", "BASE", 0, 1, 19.2, 136.0, 521, 17.4, 27.0, 184.5, "Essay A"),
            ("BASE-lf1", "BASE", 1, 2, 18.5, 110.0, 400, 17.4, 26.0, 185.0, "Essay A"),
            ("GAIN2-lf0", "GAIN2", 0, 3, 19.8, 140.0, 600, 17.5, 27.5, 182.0, "Essay B")]
    for (sn, st, lf, k, swms, swmb, ss, wms, wmb, c8, etext) in plan:
        a = "LF1" if lf else "O"
        S.append(dict(k=k, set=sn, start=st, lf=lf, arm=a, c8="c8-%d-S-%s" % (k, a)))
        for lv, p, ms, mbs, s_ in (("XCW", "code", 27.7, 370, 500), ("XEW", "essay", swms + 0.5, swmb + 10, ss), ("XC", "code", 27.7, 360, 490),
                                   ("XE", "essay", swms, swmb, ss), ("XC", "code", 27.6, 362, 489), ("XE", "essay", swms, swmb, ss),
                                   ("XW", "essay", wms, wmb, 92), ("XW", "essay", wms, wmb, 90)):
            nm = "b%d-%s-%s" % (i, lv, a)
            tp = os.path.join(d, nm + "-text.json")
            with open(tp, "w") as fh:
                json.dump({"reads": [{"prompt": p, "status": 200, "finish": "length", "tokens": 512, "content": (etext if p == "essay" else "Code X")}]}, fh)
            B.append(dict(i=i, lever=lv, arm=a, set=sn, k=k, start=st, name=nm, text="/nowhere/" + nm + "-text.json",
                          j=dict(prompt=p, tokens=512, decode_tps=round(1000 * 1.58 / ms, 2), decode_steps=324.0, ms_per_decode_step=ms, tokens_per_step=1.58, acceptance=0.698),
                          st=dict(stream_mb_step=mbs, h2d_mb_step=mbs * 0.6, promotions=700, readmits=690, starved_slab=s_, starved_pool=60, fills=1700, fills_lf=1000 if lf else 0)))
            i += 1
        M.append(dict(name="c8-%d-S-%s" % (k, a), n=8, arm=a, sched="S", agg_dec_tps=c8, ms_decode_step=36.0 * 184.5 / c8, decode_share=0.803, tok_per_decode_step=8.28, acceptance=0.428, ttft_p50=0.6, errors=0))
        with open(os.path.join(d, "c8", "c8-%d-S-%s-mix.json" % (k, a)), "w") as fh: json.dump({"signature": [[], 13]}, fh)
    for n, L in (("blocks.jsonl", B), ("mb.jsonl", M), ("sets.jsonl", S)):
        with open(os.path.join(d, n), "w") as fh:
            for x in L: fh.write(json.dumps(x) + "\n")
    buf = io.StringIO(); summary(d, "BASE-lf0", out=buf); o = buf.getvalue()
    L = o.splitlines()
    assert sum(1 for l in L if l.startswith("RES12 SET ")) == 3, o
    v = [l for l in L if l.startswith("RES12 VERDICT BASE-lf1 vs BASE-lf0")]
    assert v and "in-process" in v[0] and "helps the switch by +3.6 % at c8 +0.3 %" in v[0], v
    g = [l for l in L if l.startswith("RES12 VERDICT GAIN2-lf0 vs BASE-lf0")]
    assert g and "cross-start" in g[0] and "helps the switch by -3.1 % at c8 -1.4 %" in g[0] and "mix MATCH" in g[0], g
    idl = [l for l in L if l.startswith("RES12 IDENTITY switched essays")]
    assert idl and "VIRHE DIFFERENT" in idl[0] and "[GAIN2-lf0]" in idl[0], idl     # GAIN2's essay text differs on purpose
    ce = [l for l in L if l.startswith("RES12 IDENTITY code")]
    assert ce and ": IDENTICAL" in ce[0] and "9 reads" in ce[0] and "b0-XCW-O [BASE-lf0]" in ce[0], ce
    assert any(l.startswith("RES12 TABLE ") for l in L), o
    # the same with every answer equal: IDENTICAL, no VIRHE anywhere
    for r in B:
        tp = os.path.join(d, r["name"] + "-text.json")
        x = json.load(open(tp)); x["reads"][0]["content"] = "Essay A" if r["j"]["prompt"] == "essay" else "Code X"; json.dump(x, open(tp, "w"))
    buf = io.StringIO(); summary(d, "BASE-lf0", out=buf); o2 = buf.getvalue()
    assert "VIRHE" not in o2 and "RES12 IDENTITY switched essays across arms (6 reads in 3 sets" in o2 and ": IDENTICAL" in o2, o2
    # no base set -> a VIRHE verdict line, no crash
    buf = io.StringIO(); summary(d, "NOPE-lf0", out=buf); assert "VIRHE no base set" in buf.getvalue()
    # an empty dir -> no crash
    e = tempfile.mkdtemp(prefix="ressum-e-"); buf = io.StringIO(); summary(e, "BASE-lf0", out=buf)
    import shutil; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(e, ignore_errors=True)
    print("SELFTEST OK ressum.py (sets, verdicts in-process / cross-start, identity same / different, table, no base, empty dir)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default="BASE-lf0")
    a = ap.parse_args()
    return summary(a.dir, a.base)


if __name__ == "__main__":
    sys.exit(main() or 0)
