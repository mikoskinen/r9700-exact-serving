#!/usr/bin/env python3
"""swlsum.py (12.11.2026, queue-ar/dsmain/yswl, a builder for ds-main-port) -- the summary of dsm-yswl.sh (the SWL / L2 screen on
int12 + ds41-swl: ONE start, the arms in-process by fn4 place_swap_lent / heat_min_gain_x100). No GPU, no server. From res12/ressum.py
(same inputs and identity rule; arms instead of start sets, c8 blocks per arm, the per-read SWL counters).

  swlsum.py summary --dir OUT [--base B] [--ref "19.30:81.66:17.41:90.58:183.1"]
      OUT/blocks.jsonl  one line a 1u read (lever XC = code, XE = SWITCHED essay (after code), XW = WARM essay (after essay); set = the
                        arm; j = resprobe's J; st = the /stats deltas: stream / h2d MB a step, starved_slab, promotions (rank 0),
                        fills_r0 / swaps_r0 (rank 0's lent fills / SWL promotions), direct (= promotions - fills_r0), misses_r0)
      OUT/mb.jsonl      the c8 blocks (name c8-K-S-ARM, arm); OUT/sets.jsonl (arm -> its c8 blocks, in run order)
      OUT/c8/<b>-split.txt  p12win.py split's P12SPLIT line per c8 block; OUT/c8/<b>-mix.json the c8mix signature
    prints SWL lines:
      SWL ARM ...      an arm: its c8 block(s) (agg, ms a decode step, decode_share, the wall split, mix) || SWITCHED essays || WARM
                       essay || code reads -- each read set with tok/s, ms a step, stream / h2d MB a step, starved_slab, direct
                       promotions, lent fills, SWL lent promotions (swaps), misses (rank 0)
      SWL VERDICT ...  an arm vs the base arm B in-process (the same start): switched essay ms a step / tok/s, warm essay, c8 agg and ms
                       a decode step (each arm's blocks averaged: B L1 L1 B is ABBA) => "switch X %, warm W %, c8 Y %"
      SWL INFO ...     res12 BASE-lf0 (another start, not re-run): switched 19.30 ms / 81.66 tok/s, warm 17.41 / 90.58, c8 183.1 --
                       this start's B beside it (cross-start: ~1-1.5 % spread)
      SWL IDENTITY ... every read's answer byte-compared with the first read of its prompt in B (else the first anywhere): IDENTICAL or
                       VIRHE DIFFERENT (placement cannot change a T 0 answer); the switched essays across arms first
      SWL C8PAIR ...   the c8 B-vs-B and L1-vs-L1 spreads (the ABBA halves)
      SWL TABLE ...    one compact line
  swlsum.py selftest
"""
import argparse, json, os, re, sys, tempfile


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


KEYS = ("stream_mb_step", "h2d_mb_step", "starved_slab", "starved_pool", "promotions", "direct", "fills_r0", "swaps_r0", "fills", "swaps",
        "misses_r0", "readmits")


def agg_reads(rs):
    j = lambda k: mean([(r.get("j") or {}).get(k) for r in rs])
    s = lambda k: mean([(r.get("st") or {}).get(k) for r in rs])
    a = dict(n=len(rs), tps=j("decode_tps"), ms=j("ms_per_decode_step"), tok_step=j("tokens_per_step"), acc=j("acceptance"))
    for k in KEYS: a[k] = s(k)
    a["tps_each"] = [(r.get("j") or {}).get("decode_tps") for r in rs]
    a["ms_each"] = [(r.get("j") or {}).get("ms_per_decode_step") for r in rs]
    a["mb_each"] = [(r.get("st") or {}).get("stream_mb_step") for r in rs]
    return a


def rtext(a):
    def g(x): return "/".join(f(v) for v in x) if x else "-"
    return ("%d read(s) %s tok/s (%s), %s ms a step (%s), %s tok a step, stream %s MB (%s) / h2d %s MB a step, starved_slab %s, "
            "direct promotions %s (promotions %s - lent fills %s, rank 0), SWL lent promotions %s, misses %s (rank 0), readmits %s") % (
        a["n"], f(a["tps"]), g(a["tps_each"]), f(a["ms"]), g(a["ms_each"]), f(a["tok_step"]), f(a["stream_mb_step"], 1), g(a["mb_each"]),
        f(a["h2d_mb_step"], 1), f(a["starved_slab"], 0), f(a["direct"], 0), f(a["promotions"], 0), f(a["fills_r0"], 0), f(a["swaps_r0"], 0),
        f(a["misses_r0"], 0), f(a["readmits"], 0))


SPLIT = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = ([0-9.]+) %, (\S+) ms a step, (\S+) rows a step.*?\| \(b\) newcomer / MIXED (\d+) steps "
                   r"(\d+) ms = ([0-9.]+) %, (\S+) rows a step, (\S+) ms a step \| \(c\) host gaps (\d+) ms = ([0-9.]+) %")


def split_of(d, b):
    try:
        with open(os.path.join(d, "c8", b + "-split.txt")) as fh:
            for l in fh:
                if l.startswith("P12SPLIT "):
                    m = SPLIT.search(l)
                    if m:
                        return dict(dec_n=int(m.group(1)), dec_pct=float(m.group(3)), dec_ms=m.group(4), nc_n=int(m.group(6)),
                                    nc_pct=float(m.group(8)), nc_ms=m.group(10), gap_pct=float(m.group(12)))
    except OSError:
        pass
    return None


def mix_of(d, b):
    try:
        with open(os.path.join(d, "c8", b + "-mix.json")) as fh: return json.load(fh).get("signature")
    except (OSError, ValueError):
        return None


def c8text(d, b, m):
    s = split_of(d, b)
    return "%s agg %s tok/s, %s ms / %s tok a decode step, decode_share %s, acceptance %s, %s errors, split %s, mix %s" % (
        b, f(m.get("agg_dec_tps")), f(m.get("ms_decode_step")), f(m.get("tok_per_decode_step")), f(m.get("decode_share"), 3),
        f(m.get("acceptance"), 3), m.get("errors", "-"),
        ("(a) decode %d steps %s %% at %s ms, (b) newcomer / MIXED %d steps %s %% at %s ms, (c) gaps %s %%" % (
            s["dec_n"], f(s["dec_pct"], 1), s["dec_ms"], s["nc_n"], f(s["nc_pct"], 1), s["nc_ms"], f(s["gap_pct"], 1))) if s else "-",
        json.dumps(mix_of(d, b)) if mix_of(d, b) is not None else "-")


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


def summary(d, base, ref, out=sys.stdout):
    B = jl(os.path.join(d, "blocks.jsonl")); M = jl(os.path.join(d, "mb.jsonl")); S = jl(os.path.join(d, "sets.jsonl"))
    mb = {m.get("name"): m for m in M}
    order, c8s, desc = [], {}, {}
    for s in S:
        a = s.get("arm")
        if a not in order: order.append(a)
        if s.get("c8"): c8s.setdefault(a, []).append(s.get("c8"))
        if s.get("desc"): desc.setdefault(a, s.get("desc"))
    for r in B:
        if r.get("set") and r.get("set") not in order: order.append(r.get("set"))
    arms = {}
    for a in order:
        rs = sorted([r for r in B if r.get("set") == a], key=lambda r: r.get("i", 0))
        by = lambda lv: [r for r in rs if r.get("lever") == lv]
        cb = [b for b in c8s.get(a, []) if b in mb]
        arms[a] = dict(sw=agg_reads(by("XE")), warm=agg_reads(by("XW")), code=agg_reads(by("XC")), n=len(rs), c8=cb,
                       agg=mean([mb[b].get("agg_dec_tps") for b in cb]), ms=mean([mb[b].get("ms_decode_step") for b in cb]),
                       share=mean([mb[b].get("decode_share") for b in cb]))
    for a in order:
        x = arms[a]
        print("SWL ARM %s (%s, %d reads, %d c8 block(s)): c8 %s || SWITCHED essay (after code) %s || WARM essay (after essay) %s || code %s" % (
            a, desc.get(a, "-"), x["n"], len(x["c8"]), " ; ".join(c8text(d, b, mb[b]) for b in x["c8"]) or "-", rtext(x["sw"]),
            rtext(x["warm"]), rtext(x["code"])), file=out)
    for a in order:
        cb = arms[a]["c8"]
        if len(cb) >= 2:
            v = [mb[b].get("agg_dec_tps") for b in cb]; w = [mb[b].get("ms_decode_step") for b in cb]
            sig = [mix_of(d, b) for b in cb]
            print("SWL C8PAIR %s: %s agg %s tok/s (spread %s), ms a decode step %s; mix %s" % (
                a, " / ".join(cb), " / ".join(f(x) for x in v), fp(pct(max(v), min(v))) if all(isinstance(x, (int, float)) for x in v) and v else "-",
                " / ".join(f(x) for x in w), "MATCH" if sig[0] is not None and all(s == sig[0] for s in sig) else "DIFFERS or n/a"), file=out)
    b = arms.get(base)
    if b is None or (not b["sw"]["n"] and not b["c8"]):
        print("SWL VERDICT: VIRHE no base arm %s (arms: %s) -- no verdict" % (base, ", ".join(order) or "none"), file=out)
    else:
        for a in order:
            if a == base: continue
            x = arms[a]
            xs = pct(x["sw"]["ms"], b["sw"]["ms"], inv=True); xw = pct(x["warm"]["ms"], b["warm"]["ms"], inv=True)
            y = pct(x["agg"], b["agg"]); ym = pct(x["ms"], b["ms"], inv=True)
            mixv = "n/a"
            bs = [mix_of(d, k) for k in b["c8"]]; xsg = [mix_of(d, k) for k in x["c8"]]
            if bs and xsg and all(s is not None for s in bs + xsg): mixv = "MATCH" if all(s == bs[0] for s in xsg + bs) else "DIFFERS"
            print("SWL VERDICT %s vs %s (in-process, one start; c8 %d vs %d block(s)%s): switched essay %s -> %s ms a step (%s), %s -> %s tok/s (%s), "
                  "stream %s -> %s MB / h2d %s -> %s MB a step, starved_slab %s -> %s, direct promotions %s -> %s, lent fills %s -> %s, SWL lent "
                  "promotions %s -> %s, misses %s -> %s | warm essay %s -> %s ms a step (%s), %s -> %s tok/s, stream %s -> %s MB | c8 agg %s -> %s "
                  "tok/s (%s), %s -> %s ms a decode step (%s), decode_share %s -> %s, mix %s%s => switch %s, warm %s, c8 %s" % (
                      a, base, len(x["c8"]), len(b["c8"]), ", ABBA" if len(x["c8"]) >= 2 and len(b["c8"]) >= 2 else "",
                      f(b["sw"]["ms"]), f(x["sw"]["ms"]), fp(xs), f(b["sw"]["tps"]), f(x["sw"]["tps"]), fp(pct(x["sw"]["tps"], b["sw"]["tps"])),
                      f(b["sw"]["stream_mb_step"], 1), f(x["sw"]["stream_mb_step"], 1), f(b["sw"]["h2d_mb_step"], 1), f(x["sw"]["h2d_mb_step"], 1),
                      f(b["sw"]["starved_slab"], 0), f(x["sw"]["starved_slab"], 0), f(b["sw"]["direct"], 0), f(x["sw"]["direct"], 0),
                      f(b["sw"]["fills_r0"], 0), f(x["sw"]["fills_r0"], 0), f(b["sw"]["swaps_r0"], 0), f(x["sw"]["swaps_r0"], 0),
                      f(b["sw"]["misses_r0"], 0), f(x["sw"]["misses_r0"], 0),
                      f(b["warm"]["ms"]), f(x["warm"]["ms"]), fp(xw), f(b["warm"]["tps"]), f(x["warm"]["tps"]), f(b["warm"]["stream_mb_step"], 1),
                      f(x["warm"]["stream_mb_step"], 1), f(b["agg"]), f(x["agg"]), fp(y), f(b["ms"]), f(x["ms"]), fp(ym), f(b["share"], 3),
                      f(x["share"], 3), mixv,
                      " (c8 delta under 2 % with a different mix: INCONCLUSIVE)" if mixv == "DIFFERS" and y is not None and abs(y) < 2 else "",
                      fp(xs), fp(xw), fp(y)), file=out)
    try:
        r = [float(v) for v in ref.split(":")]
    except ValueError:
        r = []
    if len(r) == 5 and b is not None:
        print("SWL INFO res12 BASE-lf0 (another start, 12.11. 17:54, NOT re-run; cross-start spread ~1-1.5 %%): switched essay %s ms / %s tok/s, "
              "warm %s ms / %s tok/s, c8 %s tok/s -- this start's %s: switched %s ms (%s) / %s tok/s, warm %s ms (%s) / %s tok/s, c8 %s tok/s (%s)" % (
                  f(r[0]), f(r[1]), f(r[2]), f(r[3]), f(r[4], 1), base, f(b["sw"]["ms"]), fp(pct(b["sw"]["ms"], r[0], inv=True)), f(b["sw"]["tps"]),
                  f(b["warm"]["ms"]), fp(pct(b["warm"]["ms"], r[2], inv=True)), f(b["warm"]["tps"]), f(b["agg"], 1), fp(pct(b["agg"], r[4]))), file=out)
    # identity: every read's answer vs the first read of its prompt in the base arm (else the first anywhere), byte for byte
    allr = sorted(B, key=lambda r: r.get("i", 0))
    bad = []; nread = {}; refs = {}; txt = {}
    for r in allr:
        t = load_text(d, r); p = (r.get("j") or {}).get("prompt")
        if t is None:
            bad.append("%s (no answer file)" % r.get("name")); continue
        txt[r.get("name")] = t; nread[p] = nread.get(p, 0) + 1
        if p not in refs or (refs[p][0] != base and r.get("set") == base):
            refs[p] = (r.get("set"), r.get("name"), t)
    diff = {}
    for r in allr:
        p = (r.get("j") or {}).get("prompt"); t = txt.get(r.get("name"))
        if t is None or p not in refs: continue
        ref_ = refs[p][2]
        if (t.get("content") or "") != (ref_.get("content") or "") or t.get("finish") != ref_.get("finish") or t.get("tokens") != ref_.get("tokens"):
            a, c = t.get("content") or "", ref_.get("content") or ""
            k = next((i for i, (u, v) in enumerate(zip(a, c)) if u != v), min(len(a), len(c)))
            diff.setdefault(p, []).append("%s [%s] at char %d (%d vs %d chars)" % (r.get("name"), r.get("set"), k, len(a), len(c)))
    sw = [r for r in allr if r.get("lever") == "XE"]
    swd = [x for x in diff.get("essay", []) if any(x.startswith(r.get("name") + " ") for r in sw)]
    swarms = sorted(set(r.get("set") for r in sw))
    print("SWL IDENTITY switched essays across arms (%d reads in %d arms: %s) vs %s: %s" % (
        len(sw), len(swarms), ", ".join(swarms), refs.get("essay", ("-", "-"))[1],
        "IDENTICAL" if sw and not swd and "essay" in refs else ("VIRHE DIFFERENT: " + "; ".join(swd[:6]) if swd else "VIRHE no switched essay answer")), file=out)
    for p in sorted(refs, key=str):
        print("SWL IDENTITY %s (every read, T 0 greedy, %d reads, across %d arm(s)) vs %s [%s]: %s" % (
            p, nread.get(p, 0), len(set(r.get("set") for r in allr if (r.get("j") or {}).get("prompt") == p)), refs[p][1], refs[p][0],
            "IDENTICAL" if not diff.get(p) else "VIRHE DIFFERENT %d: %s" % (len(diff[p]), "; ".join(diff[p][:6]))), file=out)
    if bad:
        print("SWL IDENTITY: VIRHE %d read(s) without an answer file: %s" % (len(bad), "; ".join(bad[:6])), file=out)
    print("SWL TABLE arm | switched essay tok/s / ms a step / stream MB / starved_slab / direct / swaps | warm essay tok/s / ms / MB | c8 agg / ms a "
          "decode step (blocks) || " + " || ".join(
              "%s | %s / %s / %s / %s / %s / %s | %s / %s / %s | %s / %s (%d)" % (
                  a, f(arms[a]["sw"]["tps"]), f(arms[a]["sw"]["ms"]), f(arms[a]["sw"]["stream_mb_step"], 1), f(arms[a]["sw"]["starved_slab"], 0),
                  f(arms[a]["sw"]["direct"], 0), f(arms[a]["sw"]["swaps_r0"], 0), f(arms[a]["warm"]["tps"]), f(arms[a]["warm"]["ms"]),
                  f(arms[a]["warm"]["stream_mb_step"], 1), f(arms[a]["agg"]), f(arms[a]["ms"]), len(arms[a]["c8"])) for a in order), file=out)
    return 0


def selftest():
    import io, shutil
    d = tempfile.mkdtemp(prefix="swlsum-")
    os.makedirs(os.path.join(d, "c8"))
    B, M, S = [], [], []
    i = 0
    # arm, k, c8 agg, switched ms, switched MB, starved, swaps, warm ms, essay text
    plan = [("B", 1, 183.0, 19.30, 146.5, 516, 0, 17.41, "Essay A"), ("L1", 2, 184.0, 18.60, 120.0, 40, 470, 17.30, "Essay A"),
            ("L1G075", 5, 182.0, 18.50, 118.0, 60, 520, 17.35, "Essay B")]
    for (a, k, c8, swms, swmb, ss, swp, wms, etext) in plan:
        S.append(dict(k=k, set=a, arm=a, c8="c8-%d-S-%s" % (k, a), desc="desc " + a, kind="arm"))
        for lv, p, ms, mbs in (("XC", "code", 27.7, 370), ("XE", "essay", swms, swmb), ("XC", "code", 27.6, 362), ("XE", "essay", swms, swmb),
                               ("XW", "essay", wms, 18.0)):
            nm = "b%d-%s-%s" % (i, lv, a)
            with open(os.path.join(d, nm + "-text.json"), "w") as fh:
                json.dump({"reads": [{"prompt": p, "status": 200, "finish": "length", "tokens": 512, "content": (etext if p == "essay" else "Code X")}]}, fh)
            B.append(dict(i=i, lever=lv, arm=a, set=a, k=k, start="BASE", name=nm, text="/nowhere/" + nm + "-text.json",
                          j=dict(prompt=p, tokens=512, decode_tps=round(1000 * 1.58 / ms, 2), decode_steps=324.0, ms_per_decode_step=ms, tokens_per_step=1.58,
                                 acceptance=0.698),
                          st=dict(stream_mb_step=mbs, h2d_mb_step=mbs * 0.6, promotions=700, readmits=690, starved_slab=ss, starved_pool=0, fills_r0=450,
                                  swaps_r0=swp, direct=250 + swp, misses_r0=1100, fills=1800, swaps=4 * swp)))
            i += 1
        M.append(dict(name="c8-%d-S-%s" % (k, a), n=8, arm=a, sched="S", agg_dec_tps=c8, ms_decode_step=36.0 * 183.0 / c8, decode_share=0.80,
                      tok_per_decode_step=8.28, acceptance=0.43, ttft_p50=0.6, errors=0))
        with open(os.path.join(d, "c8", "c8-%d-S-%s-mix.json" % (k, a)), "w") as fh: json.dump({"signature": [[], 16]}, fh)
        with open(os.path.join(d, "c8", "c8-%d-S-%s-split.txt" % (k, a)), "w") as fh:
            fh.write("P12SPLIT c8-%d-S-%s (window 35.0 s): (a) decode 800 steps 28000 ms = 80.0 %%, 36.00 ms a step, 8.10 rows a step, 8.28 tokens a step | "
                     "(b) newcomer / MIXED 16 steps 7000 ms = 20.0 %%, 1096 rows a step, 437.5 ms a step | (c) host gaps 0 ms = 0.0 %% (idle > 3 s 0)\n" % (k, a))
    # the ABBA halves: L1 (k 3) and B (k 4), c8 only
    for (a, k, c8) in (("L1", 3, 184.4), ("B", 4, 183.4)):
        S.append(dict(k=k, set=a, arm=a, c8="c8-%d-S-%s" % (k, a), kind="c8"))
        M.append(dict(name="c8-%d-S-%s" % (k, a), n=8, arm=a, sched="S", agg_dec_tps=c8, ms_decode_step=36.0 * 183.0 / c8, decode_share=0.80,
                      tok_per_decode_step=8.28, acceptance=0.43, ttft_p50=0.6, errors=0))
        with open(os.path.join(d, "c8", "c8-%d-S-%s-mix.json" % (k, a)), "w") as fh: json.dump({"signature": [[], 16]}, fh)
    for n, L in (("blocks.jsonl", B), ("mb.jsonl", M), ("sets.jsonl", S)):
        with open(os.path.join(d, n), "w") as fh:
            for x in L: fh.write(json.dumps(x) + "\n")
    buf = io.StringIO(); summary(d, "B", "19.30:81.66:17.41:90.58:183.1", out=buf); o = buf.getvalue()
    L = o.splitlines()
    assert sum(1 for l in L if l.startswith("SWL ARM ")) == 3, o
    a1 = [l for l in L if l.startswith("SWL ARM L1 ")]
    assert a1 and "2 c8 block(s)" in a1[0] and "(a) decode 800 steps 80.0 % at 36.00 ms" in a1[0] and "SWL lent promotions 470" in a1[0], a1
    v = [l for l in L if l.startswith("SWL VERDICT L1 vs B")]
    assert v and "ABBA" in v[0] and "switch +3.6 %" in v[0] and "c8 +0.5 %" in v[0] and "starved_slab 516 -> 40" in v[0] and "mix MATCH" in v[0], v
    g = [l for l in L if l.startswith("SWL VERDICT L1G075 vs B")]
    assert g and "c8 1 vs 2 block(s))" in g[0] and "c8 -0.7 %" in g[0], g
    assert any(l.startswith("SWL C8PAIR B: c8-1-S-B / c8-4-S-B") and "MATCH" in l for l in L), o
    info = [l for l in L if l.startswith("SWL INFO res12 BASE-lf0")]
    assert info and "this start's B: switched 19.30 ms (+0.0 %)" in info[0], info
    idl = [l for l in L if l.startswith("SWL IDENTITY switched essays")]
    assert idl and "VIRHE DIFFERENT" in idl[0] and "[L1G075]" in idl[0], idl     # L1G075's essay differs on purpose
    ce = [l for l in L if l.startswith("SWL IDENTITY code")]
    assert ce and ": IDENTICAL" in ce[0] and "6 reads, across 3 arm(s)" in ce[0] and "b0-XC-B [B]" in ce[0], ce
    assert any(l.startswith("SWL TABLE ") for l in L), o
    for r in B:   # every answer equal: IDENTICAL, no VIRHE anywhere
        tp = os.path.join(d, r["name"] + "-text.json")
        x = json.load(open(tp)); x["reads"][0]["content"] = "Essay A" if r["j"]["prompt"] == "essay" else "Code X"; json.dump(x, open(tp, "w"))
    buf = io.StringIO(); summary(d, "B", "19.30:81.66:17.41:90.58:183.1", out=buf); o2 = buf.getvalue()
    assert "VIRHE" not in o2 and "SWL IDENTITY switched essays across arms (6 reads in 3 arms" in o2 and ": IDENTICAL" in o2, o2
    tp = os.path.join(d, B[-1]["name"] + "-text.json")   # one token count differs, same text: still a difference
    x = json.load(open(tp)); x["reads"][0]["tokens"] = 511; json.dump(x, open(tp, "w"))
    buf = io.StringIO(); summary(d, "B", "19.30:81.66:17.41:90.58:183.1", out=buf); assert "VIRHE DIFFERENT" in buf.getvalue()
    buf = io.StringIO(); summary(d, "NOPE", "", out=buf); assert "VIRHE no base arm" in buf.getvalue()
    e = tempfile.mkdtemp(prefix="swlsum-e-"); buf = io.StringIO(); summary(e, "B", "x", out=buf)
    shutil.rmtree(d, ignore_errors=True); shutil.rmtree(e, ignore_errors=True)
    print("SELFTEST OK swlsum.py (arms, c8 split / ABBA pairs, verdicts vs B, res12 INFO, identity same / different / token count, table, no base, empty dir)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default="B")
    ap.add_argument("--ref", default="19.30:81.66:17.41:90.58:183.1")
    a = ap.parse_args()
    return summary(a.dir, a.base, a.ref)


if __name__ == "__main__":
    sys.exit(main() or 0)
