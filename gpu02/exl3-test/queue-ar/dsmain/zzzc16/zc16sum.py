#!/usr/bin/env python3
"""zc16sum.py (13.11.2026, queue-ar/dsmain/zzzc16, a builder for ds-main-port) -- the summary of THE int16 CONFIRMATION (item
57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16.sh, body zzzc16/dsm-zzzc16.sh): int16 = int15 + fn4 ds41_rfuse 2 + route_copy_ov 1
(ds41_mix_fin 0) as arm F (= the pin O) against B = int15 (the three fusion switches 0), in-process on ONE start. New; the step-log readers
(read_steps / wall / the newcomer test) are zzzfuse/fusesum.py's (md5 2a785be2), verbatim.

  zc16sum.py OUT --log ENGINE_LOG [--sw 1] [--tag xzc16-n5] [--post 10] [--want B=0/0/0,F=2/0/1] [--xb 2] [--swlg 75] [--rfdecl 2]
             [--u1n 3] [--c4gens 3] [--essay-md5 47a39dac581a] [--cold-depth 136000] [--cold-ref-md5 3623688c] [--stopped 0] [--json J]
      OUT = the item's out dir: identity.txt; fn-TAG-pin.json / st-TAG-pin.json (rankrb's snapshot right after THE int16 PIN was posted);
      fn-TAG-eng-0/1.json (pin16's engagement probe); fuse-readback.txt (fusechk, one line an arm()); blocks.jsonl (blockx's 1u reads,
      set U1-<block>); mb.jsonl (mbx's c4 gens blocks c4-<i>-G-<arm>, the warm-up c4-w1-G-w<arm>); fn-NAME-0/1.json + st-NAME-0/1.json
      around every read / block; cold.jsonl (coldx). Prints lines "INT16 ..."; a guard failure is a "VIRHE" inside its line:
        INT16 PIN         the start's FUSE-R declare line ("now 2", form 2 at every trunk router), the pin read back on GET /admin/fn4
                          (the fusion switches, exl3_x3d, the int14 levers, nothing pending) and on the 4 ranks (/stats), the engagement
                          probe's "fuse" / "rco" deltas, the per-block read-backs (fuse-readback.txt)
        INT16 IDENTITY    every strict comparison (warm-up, a1-F vs ref12 / zzzfuse a1-B, gs16-F, staggered ld-F)
        INT16 1u BLOCK    per ABBA block: tok/s a read + mean, ms a step, stream MB a step, the discarded warm-up, answers, switches,
                          engagement, ranks
        INT16 1u ARM / 1u F vs B / 1u VERDICT   the mean of each arm's reads, F vs B % with the same-arm spread, "1u int16 = X tok/s (>= 90)"
        INT16 c4 BLOCK    per gens block: agg, tokens a step, acceptance, the step-log decode p50 (all decode steps) + the full-row bucket,
                          rows a step, the q histogram / depth-0 share, newcomer steps in the window; switches, engagement, ranks
        INT16 c4 F vs B / c4 INFO   arm means, F vs B % with the spread, block separation; zrq14 / ref12 INFO ("not the same method")
        INT16 136k        TTFT, the answer vs ref14's (identity.txt), INFO vs ref14 24.48 s
        INT16 SUMMARY     one line: identity, the pin read-back, the 1u verdict, c4 gens, 136k -- values, reference, % difference
  zc16sum.py selftest
Standard library only. Read only (writes only --json)."""
import bisect, gzip, io, json, os, re, statistics, sys, tempfile, zlib
from datetime import datetime, timezone

TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")
STEP = re.compile(r" step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
SEG = re.compile(r"\[\d+\] slot=(\d+) q=(\d+) ctx=(\d+)")
QMIN = 16      # a prefill segment has more rows than any decode / verify segment (draft depth <= 15)
NCROWS = 512   # a non-decode step of >= 512 rows is a newcomer step even when its prefill entry is not printed
IDLE = 3.0     # a step followed by > 3 s of nothing is a host gap, not a wall
U1TARGET = 90.0
# references (other starts, INFO only): ref14 = int14 (MAXSEQS 16) 136k cold TTFT; zrq14 c4 (MAXSEQS 16, 20 s TIME windows): B = int12
# bytes, Z = int14; ref12 c4 (int12, MAXSEQS 8, 20 s time windows); zzzfuse 1u (single order B then F): B 89.86, F 90.46
REF136 = ("ref14 (int14, another start)", 24.48)
C4REF = (("zrq14 B (int12 bytes, MAXSEQS 16, 20 s time windows)", 175.30), ("zrq14 Z (int14, MAXSEQS 16, 20 s time windows)", 178.05),
         ("ref12 (int12, MAXSEQS 8, 20 s time windows)", 173.20))
U1REF = (("zzzfuse B (int15, single order)", 89.86), ("zzzfuse F (int16, single order)", 90.46))
IDKEYS = ("n5 warm-up", "a1-F-vs-ref12-a1", "a1-F-vs-zzzfuse-a1-B", "gs16-F-vs-a1-F", "ld-F-vs-a1-F")
FUSEK = ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches")
RCOK = ("recordings", "copies", "freed")
PINK = ("ds41_rfuse", "ds41_mix_fin", "route_copy_ov", "exl3_x3d", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100",
        "heat_hyst_x100", "sched_row_cap", "mhc_fuse", "draft_depth0", "ds41_gq", "ds41_e2", "ds41_mix_wv")


def opn(p):
    with open(p, "rb") as f: data = f.read()
    out = []
    while data:
        if data[:2] == b"\x1f\x8b":
            d = zlib.decompressobj(31); out.append(d.decompress(data)); out.append(d.flush()); data = d.unused_data
        else: out.append(data); break
    return io.StringIO(b"".join(out).decode("utf-8", errors="replace"))


def tsec(m):
    t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return t + (float("0." + m.group(2)) if m.group(2) else 0.0)


def read_log(path):
    """(steps, the FUSE-R declare line) from the engine log (docker logs -t, gzip or plain)"""
    out, decl = [], None
    if not path or not os.path.exists(path): return out, decl
    for l in opn(path):
        if decl is None and "lever FUSE-R (ds41_rfuse" in l: decl = l.strip()[:400]
        m = TSRE.match(l)
        if not m or " step " not in l: continue
        s = STEP.search(l.rstrip("\n"))
        if not s: continue
        out.append((tsec(m), int(s.group(1)), s.group(2), int(s.group(3)), int(s.group(4)), [tuple(int(x) for x in g) for g in SEG.findall(s.group(5))]))
    return out, decl


def rdj(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None


def jl(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if not l: continue
            try: out.append(json.loads(l))
            except ValueError: pass
    except OSError: pass
    return out


def is_nc(s): return s[2] != "decode" and (any(g[1] > QMIN for g in s[5]) or s[4] >= NCROWS)


def wall(steps, j):
    if j is None or j + 1 >= len(steps): return None
    d = steps[j + 1][0] - steps[j][0]
    return 1000.0 * d if 0.0 <= d <= IDLE else None


def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def nums(v): return [x for x in v if num(x)]
def p50(v): return round(statistics.median(v), 2) if v else None
def mean(v): return round(statistics.mean(v), 2) if v else None
def f(v, n=2): return ("%." + str(n) + "f") % v if num(v) else "-"
def pc(a, b): return 100.0 * (a / b - 1.0) if num(a) and num(b) and b else None
def fp(v): return ("%+.2f %%" % v) if num(v) else "-"
def spread(v): return round(100.0 * (max(v) - min(v)) / statistics.mean(v), 2) if len(v) > 1 and statistics.mean(v) else None


def sep(a, b):
    """block values of two arms: SEPARATED when every block of one arm is below every block of the other"""
    if not a or not b: return "-"
    return "SEPARATED" if max(a) < min(b) or max(b) < min(a) else "OVERLAP"


def fusesw(fn):
    w = (fn or {}).get("switches") or {}
    return (w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d")) if fn else None


def ctr(f0, f1):
    """"fuse" / "rco" deltas between two GET /admin/fn4 snapshots (+ rco in_force at the end), or None"""
    if not f0 or not f1 or not isinstance(f1.get("fuse"), dict): return None
    a, b = f0.get("fuse") or {}, f1["fuse"]
    d = {k: (b.get(k) or 0) - (a.get(k) or 0) for k in FUSEK}
    ra, rb = f0.get("rco") or {}, f1.get("rco") or {}
    for k in RCOK: d["rco_" + k] = (rb.get(k) or 0) - (ra.get(k) or 0)
    d["rco_in_force"] = rb.get("in_force")
    return d


# ds41_rfuse 2's documented fallback (ds41-fuse rad_block_moe_fp8.h pass(): form 2 at a MoE pass of <= 48 rows, form 1 at 49 .. 170 rows
# (1024 / top_k, the sort's one workgroup), the separate ops above). Under ds41_bi 6 a MoE pass is one BI segment (deepseek41.cpp: the
# decode segment in pieces of <= kBiMoe 16 rows -> always form 2; each PREFILL segment whole). So form 1 runs only in passes over a prefill
# segment of 49 .. 170 rows: a short prompt / prefill piece, or a PSK-cut layer (from psk_first on a prefill segment runs only its last
# psk_rows = max(C_l - ahead, kPskBase 160) rows, so a few late layers of a newcomer step land in 160 .. 170). EXPECTED (INFO) when the span
# prefilled something; a form-1 pass in a span with no prefill (decode / verify rows only: <= 48, form 2's) or more than KF1 a prefilled
# request (40 MoE layers x 2 segments) is UNEXPECTED (VIRHE).
KF1 = 80


def f1fallback(want, d, prefills):
    """'' or the INFO text for ds41_rfuse 2's expected form-1 fallback in a span with `prefills` prefilled requests"""
    if not d or want[0] != 2 or not d.get("rf_form1_passes") or not prefills or d["rf_form1_passes"] > KF1 * prefills: return ""
    return (" | form 1 +%d (EXPECTED fallback, INFO: MoE passes over a prefill segment of 49 .. 170 rows -- a short prefill or a PSK-cut layer; "
            "%d prefilled request(s) in the span)" % (d["rf_form1_passes"], prefills))


def engaged(want, d, prefills=0):
    """[problems]: an arm running (rf, mf, rc) whose block / read moved the counters d (a recorded pass counts at recording; arm() bumps the epoch);
    `prefills`: the prefilled requests in the span (0 = none or unknown: a form-1 pass is then UNEXPECTED)"""
    if d is None: return ["no GET \"fuse\" / \"rco\" counters around it"]
    rf, mf, rc = want
    bad = []
    if rf == 2 and not (d["rf_form2_passes"] > 0 and d["rgs_launches"] > 0): bad.append("ds41_rfuse 2 but form 2 did not run (passes +%s, launches +%s)" % (d["rf_form2_passes"], d["rgs_launches"]))
    if rf == 0 and (d["rf_form1_passes"] or d["rf_form2_passes"] or d["rgs_launches"]): bad.append("ds41_rfuse 0 but a fused router ran (form1 +%s, form2 +%s)" % (d["rf_form1_passes"], d["rf_form2_passes"]))
    if rf == 2 and d["rf_form1_passes"] and not f1fallback(want, d, prefills):
        bad.append("ds41_rfuse 2 but form 1 ran (+%s) %s" % (d["rf_form1_passes"], "in a span with no prefill (decode / verify passes are <= 48 rows: form 2's)" if not prefills
                                                             else "beyond the fallback's bound (%d x %d prefilled requests)" % (KF1, prefills)))
    if mf == 0 and d["mixfin_launches"]: bad.append("ds41_mix_fin 0 but FIN launches +%s" % d["mixfin_launches"])
    if rc == 1 and not (d.get("rco_in_force") == 1 and d["rco_recordings"] > 0 and d["rco_freed"] > 0): bad.append("route_copy_ov 1 but no recording freed a copy (in_force %s, recordings +%s, freed +%s)" % (d.get("rco_in_force"), d["rco_recordings"], d["rco_freed"]))
    if rc == 0 and (d.get("rco_in_force") or d["rco_recordings"]): bad.append("route_copy_ov 0 but rco in force / recording (%s, +%s)" % (d.get("rco_in_force"), d["rco_recordings"]))
    return bad


def estr(d): return ("form2 +%s rgs +%s form1 +%s mixfin +%s | rco in_force %s rec +%s freed +%s" % (d["rf_form2_passes"], d["rgs_launches"], d["rf_form1_passes"], d["mixfin_launches"], d.get("rco_in_force"), d["rco_recordings"], d["rco_freed"])) if d else "n/a"


def ranks_bad(ranks, swlg):
    """the 4 ranks' SWL values (GET /stats experts.ranks): place_swl 1, heat_min_gain_x100 SWLG, heat_hyst_x100 125, place_lf 0"""
    if not ranks: return ["no ranks in /stats"]
    bad = [] if len(ranks) == 4 else ["%d ranks (want 4)" % len(ranks)]
    for r in ranks:
        for k, v in (("place_swl", 1), ("heat_min_gain_x100", swlg), ("heat_hyst_x100", 125), ("place_lf", 0)):
            if r.get(k) != v: bad.append("rank %s %s %s (want %s)" % (r.get("rank"), k, r.get(k), v))
    return bad


# ---------------------------------------------------------------- the pin read-back
def pin(out, tag, decl, sw, xb, swlg, rfdecl, want16):
    fp_, st = rdj(os.path.join(out, "fn-%s-pin.json" % tag)), rdj(os.path.join(out, "st-%s-pin.json" % tag))
    e0, e1 = rdj(os.path.join(out, "fn-%s-eng-0.json" % tag)), rdj(os.path.join(out, "fn-%s-eng-1.json" % tag))
    bad, info = [], []
    m = re.search(r"now (\d+)\).*form 2, [^)]*\) at (\d+)", decl or "")
    if not decl: bad.append("no FUSE-R declare line in the engine log")
    elif not m or int(m.group(1)) != rfdecl or int(m.group(2)) <= 0: bad.append("the FUSE-R declare line is not 'now %d' with form 2 at > 0 layers" % rfdecl)
    w = (fp_ or {}).get("switches") or {}
    exp = dict(zip(("ds41_rfuse", "ds41_mix_fin", "route_copy_ov"), want16), exl3_x3d=xb, ds41_bi=6, gg_pf=2, ar_xpush=1, gg_gemv_m16=1, place_swap_lent=1,
               heat_min_gain_x100=swlg, heat_hyst_x100=125)
    if not fp_: bad.append("no GET /admin/fn4 snapshot after the pin (fn-%s-pin.json)" % tag)
    else:
        bad += ["%s %s (want %s)" % (k, w.get(k), v) for k, v in exp.items() if w.get(k) != v]
        if fp_.get("pending"): bad.append("pending %s" % fp_.get("pending"))
    rk = ((st or {}).get("experts") or {}).get("ranks") or []
    bad += ranks_bad(rk, swlg)
    d = ctr(e0, e1)
    eb = engaged(want16, d, prefills=1)   # the probe: one prompt
    if d is not None and not ((e1 or {}).get("fuse") or {}).get("rf_declared"): eb.append("fuse rf_declared %s" % ((e1 or {}).get("fuse") or {}).get("rf_declared"))
    bad += eb
    rb = []
    try:
        for l in open(os.path.join(out, "fuse-readback.txt")):
            mm = re.search(r" arm (\S+) ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d (.*) \(want (.*)\)\s*$", l)
            if mm: rb.append((mm.group(1), mm.group(2).strip(), mm.group(3).strip()))
    except OSError: pass
    mis = [x for x in rb if x[1] != x[2]]
    if mis: bad.append("%d of %d per-block read-backs differ (%s)" % (len(mis), len(rb), "; ".join("%s %s want %s" % x for x in mis[:4])))
    if not sw: info, bad = bad, []
    txt = ("INT16 PIN (THE int16 PIN = the int15 pin + ds41_rfuse %d + ds41_mix_fin %d + route_copy_ov %d, posted as O right after the start and read back): start declare '%s' | "
           "GET /admin/fn4 after the post: %s, pending %s | GET /stats ranks: %s | engagement over the probe (arm O, epoch_bump, + 64 tok): %s | per-block read-backs "
           "(fusechk before every block): %d checks, %d mismatches%s" % (
               want16[0], want16[1], want16[2], (decl or "NONE")[-260:], " ".join("%s %s" % (k, w.get(k)) for k in PINK), (fp_ or {}).get("pending"),
               "; ".join("r%s swl %s mg %s hy %s lf %s" % (r.get("rank"), r.get("place_swl"), r.get("heat_min_gain_x100"), r.get("heat_hyst_x100"), r.get("place_lf")) for r in rk) or "none",
               estr(d), len(rb), len(mis), (" -- VIRHE " + "; ".join(bad)) if bad else ((" | INFO (mock, ZC16_SW=0) " + "; ".join(info[:6])) if info else " -- OK")))
    return dict(ok=not bad, bad=bad, checks=len(rb), mismatches=len(mis), engagement=d, switches={k: w.get(k) for k in PINK}), txt


# ---------------------------------------------------------------- identity
def identity(out):
    rows = []
    try:
        for l in open(os.path.join(out, "identity.txt")):
            l = l.strip()
            if l: rows.append(l)
    except OSError: pass
    got = {}
    for k in IDKEYS:
        hit = [r for r in rows if r.startswith("xzc16-" + k + " ") or r.startswith("xzc16-" + k + ":")]
        got[k] = hit[-1][len("xzc16-" + k):].strip() if hit else None
    ok = [k for k in IDKEYS if got[k] and got[k].startswith("IDENTICAL")]
    return got, ok, rows


# ---------------------------------------------------------------- 1u ABBA
def u1(out, want, xb, swlg, emd5, sw):
    R = [r for r in jl(os.path.join(out, "blocks.jsonl")) if r.get("lever") in ("U1W", "U1")]
    blocks = {}
    for r in R:
        k = r.get("k") or 0
        blocks.setdefault(k, []).append(r)
    res = dict(blocks=[], arms={})
    for k in sorted(blocks):
        rs = blocks[k]
        arm = rs[0].get("arm")
        W = [r for r in rs if r.get("lever") == "U1W"]; U = [r for r in rs if r.get("lever") == "U1"]
        g = lambda r, x: (r.get("j") or {}).get(x)
        s = lambda r, x: (r.get("st") or {}).get(x)
        bad, info = [], []
        if any(r.get("arm") != arm for r in rs): bad.append("mixed arms in one block %s" % sorted({r.get("arm") for r in rs}))
        wa = tuple(want.get(arm, (None, None, None)))
        for r in W + U:
            nm = r.get("name") or ""
            f0, f1 = rdj(os.path.join(out, "fn-%s-0.json" % nm)), rdj(os.path.join(out, "fn-%s-1.json" % nm))
            x = fusesw(f1)
            if x is None or tuple(x) != wa + (xb,): (bad if sw else info).append("%s switches %s (want %s)" % (nm, list(x) if x else None, list(wa + (xb,))))
            if g(r, "text_md5") != emd5: (bad if sw else info).append("%s answer md5 %s (want %s)" % (nm, g(r, "text_md5"), emd5))
            if sw: bad += ["%s %s" % (nm, e) for e in engaged(wa, ctr(f0, f1), prefills=1)] if None not in wa else ["%s arm %s not in --want" % (nm, arm)]   # a read: its one prompt
            rk = [dict(rank=i, place_swl=a, heat_min_gain_x100=b, heat_hyst_x100=c, place_lf=d) for i, (a, b, c, d) in enumerate(zip(s(r, "place_swl") or [], s(r, "mg") or [], s(r, "hy") or [], s(r, "place_lf") or []))]
            rkb = ranks_bad(rk, swlg)
            if rkb: (bad if sw else info).append("%s ranks: %s" % (nm, "; ".join(rkb[:3])))
            if g(r, "status") not in (200, None) or not (g(r, "tokens") or 0) > 0: bad.append("%s request status %s tokens %s" % (nm, g(r, "status"), g(r, "tokens")))
        tps, ms, tok, mb = nums(g(r, "decode_tps") for r in U), nums(g(r, "ms_per_decode_step") for r in U), nums(g(r, "tokens_per_step") for r in U), nums(s(r, "stream_mb_step") for r in U)
        b = dict(k=k, arm=arm, n=len(U), tps=tps, tps_mean=mean(tps), ms=ms, ms_mean=mean(ms), tok=mean(tok), mb=mb, mb_mean=mean(mb), acc=mean(nums(g(r, "acceptance") for r in U)),
                 w_tps=g(W[0], "decode_tps") if W else None, w_mb=s(W[0], "stream_mb_step") if W else None, md5=[g(r, "text_md5") for r in W + U], bad=bad, info=info)
        res["blocks"].append(b)
    for a in ("B", "F"):
        bl = [b for b in res["blocks"] if b["arm"] == a]
        tps = [v for b in bl for v in b["tps"]]
        ms = [v for b in bl for v in b["ms"]]
        mb = [v for b in bl for v in b["mb"]]
        bm = nums(b["tps_mean"] for b in bl)
        res["arms"][a] = dict(blocks=[b["k"] for b in bl], n=len(tps), tps=mean(tps), tps_reads=tps, spread=spread(tps), block_means=bm, block_spread=spread(bm),
                              ms=mean(ms), mb=mean(mb), tok=mean(nums(b["tok"] for b in bl)))
    return res


# ---------------------------------------------------------------- c4 gens blocks
def c4(out, steps, want, xb, swlg, sw, gens):
    ts = [s[0] for s in steps]
    R = [r for r in jl(os.path.join(out, "mb.jsonl")) if r.get("n") == 4 and r.get("sched") == "G"]
    res = dict(blocks=[], arms={})
    for r in R:
        nm, arm = r["name"], r["arm"]
        mk = r.get("marks") or []
        if len(mk) < 2:
            mk = sorted(x.get("t") for x in jl(os.path.join(out, "c8", nm + "-reqs.jsonl")) if x.get("ev") == "mark")[:2]
        warm = arm.startswith("w")
        b = dict(name=nm, arm=arm, warm=warm, agg=r.get("agg_dec_tps"), tok=r.get("tok_per_decode_step"), ms_h=r.get("ms_decode_step"), acc=r.get("acceptance"),
                 window_s=r.get("window_s"), requests=r.get("requests"), errors=r.get("errors"), decode_share=r.get("decode_share"), ttft_p50=r.get("ttft_p50"))
        bad, info = [], []
        if len(mk) < 2 or not num(mk[0]) or not num(mk[1]): bad.append("no window marks"); b["win"] = None
        else:
            w0, w1 = mk[0], mk[1]; b["win"] = [w0, w1]
            i0, i1 = bisect.bisect_left(ts, w0), bisect.bisect_right(ts, w1)
            dec = [j for j in range(i0, i1) if steps[j][2] == "decode"]
            wl = [(j, wall(steps, j)) for j in dec]; wl = [(j, x) for j, x in wl if x is not None]
            allw = [x for _, x in wl]
            q = {}; segs = 0; d0 = 0; seqs = []; rows = []
            for j in dec:
                st = steps[j]; seqs.append(st[3]); rows.append(st[4])
                for g in st[5]:
                    segs += 1; key = str(g[1]) if g[1] <= 5 else "6+"
                    q[key] = q.get(key, 0) + 1
                    if g[1] == 1: d0 += 1
            rc = {}
            for j, x in wl:
                if steps[j][3] == 4: rc[steps[j][4]] = rc.get(steps[j][4], 0) + 1
            b.update(n_dec=len(allw), dec_p50=p50(allw), dec_mean=mean(allw), seqs=mean(seqs), rows=mean(rows), q=q, segs=segs,
                     d0=round(100.0 * d0 / segs, 1) if segs else None, rowc=rc, n_nc=sum(1 for j in range(i0, i1) if is_nc(steps[j])))
            if not allw: bad.append("no decode step in the window (engine log truncated or not captured?)")
            if gens and b["n_nc"] != 4 * gens: b["nc_note"] = "%d newcomer steps in the window (gens x 4 = %d expected)" % (b["n_nc"], 4 * gens)
        f0, f1 = rdj(os.path.join(out, "fn-%s-0.json" % nm)), rdj(os.path.join(out, "fn-%s-1.json" % nm))
        s1 = rdj(os.path.join(out, "st-%s-1.json" % nm))
        a = arm[1:] if warm else arm
        wa = tuple(want.get(a, (None, None, None)))
        x = fusesw(f1); b["sw"] = list(x) if x else None
        b["eng"] = ctr(f0, f1)
        if None in wa: bad.append("arm %s not in --want" % a)
        else:
            if x is None or tuple(x) != wa + (xb,): (bad if sw else info).append("switches %s at the block's end (want %s)" % (list(x) if x else None, list(wa + (xb,))))
            if sw: bad += engaged(wa, b["eng"], prefills=(b.get("requests") or 0) + 8)   # the window's requests + the warm-up / tail admissions (4 streams)
            b["f1"] = f1fallback(wa, b["eng"], (b.get("requests") or 0) + 8)
        rkb = ranks_bad(((s1 or {}).get("experts") or {}).get("ranks") or [], swlg)
        if rkb: (bad if sw else info).append("ranks: " + "; ".join(rkb[:3]))
        if (b.get("errors") or 0) != 0 or not (b.get("requests") or 0) > 0: bad.append("request errors %s of %s requests" % (b.get("errors"), b.get("requests")))
        b["bad"], b["info"] = bad, info
        res["blocks"].append(b)
    pool = {}
    for b in res["blocks"]:
        if not b["warm"]:
            for k, v in (b.get("rowc") or {}).items(): pool[k] = pool.get(k, 0) + v
    rows = max(pool, key=lambda k: pool[k]) if pool else None
    for b in res["blocks"]:
        if b.get("win") and rows is not None:
            i0, i1 = bisect.bisect_left(ts, b["win"][0]), bisect.bisect_right(ts, b["win"][1])
            w = [wall(steps, j) for j in range(i0, i1) if steps[j][2] == "decode" and steps[j][3] == 4 and steps[j][4] == rows]
            w = [v for v in w if v is not None]
            b["bkt_rows"], b["bkt_p50"], b["bkt_n"] = rows, p50(w), len(w)
    for a in ("B", "F"):
        bl = [b for b in res["blocks"] if b["arm"] == a]
        ag, dp, bk = nums(b["agg"] for b in bl), nums(b.get("dec_p50") for b in bl), nums(b.get("bkt_p50") for b in bl)
        qq = {}
        for b in bl:
            for k, v in (b.get("q") or {}).items(): qq[k] = qq.get(k, 0) + v
        res["arms"][a] = dict(blocks=[b["name"] for b in bl], aggs=ag, agg=mean(ag), agg_spread=spread(ag), dec_p50s=dp, dec_p50=mean(dp), dec_spread=spread(dp),
                              bkt_p50s=bk, bkt_p50=mean(bk), bkt_spread=spread(bk), bkt_rows=rows, tok=mean(nums(b["tok"] for b in bl)), acc=mean(nums(b["acc"] for b in bl)),
                              d0=mean(nums(b.get("d0") for b in bl)), q=qq, domq=max(qq, key=lambda k: qq[k]) if qq else None)
    return res


def run(argv):
    out, log, sw, tag, post, xb, swlg, rfdecl, u1n, gens, emd5, cdep, cmd5, stopped, jout = None, None, 1, "xzc16-n5", 10, 2, 75, 2, 3, 3, "47a39dac581a", 136000, "3623688c", 0, None
    wspec = "B=0/0/0,F=2/0/1"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--log", "--sw", "--tag", "--post", "--want", "--xb", "--swlg", "--rfdecl", "--u1n", "--c4gens", "--essay-md5", "--cold-depth", "--cold-ref-md5", "--stopped", "--json"):
            v = argv[i + 1]; i += 2
            if a == "--log": log = v
            elif a == "--sw": sw = int(v)
            elif a == "--tag": tag = v
            elif a == "--post": post = int(v)
            elif a == "--want": wspec = v
            elif a == "--xb": xb = int(v)
            elif a == "--swlg": swlg = int(v)
            elif a == "--rfdecl": rfdecl = int(v)
            elif a == "--u1n": u1n = int(v)
            elif a == "--c4gens": gens = int(v)
            elif a == "--essay-md5": emd5 = v
            elif a == "--cold-depth": cdep = int(v)
            elif a == "--cold-ref-md5": cmd5 = v
            elif a == "--stopped": stopped = int(v)
            elif a == "--json": jout = v
        else: out = a; i += 1
    if not out:
        print("INT16 VIRHE usage: OUT --log LOG"); return 2
    want = {}
    for e in wspec.split(","):
        k, v = e.split("=")
        want[k] = tuple(int(x) for x in v.split("/"))
    want16 = want.get("F", (2, 0, 1))
    steps, decl = read_log(log)
    nbad = 0
    # PIN
    P, txt = pin(out, tag, decl, sw, xb, swlg, rfdecl, want16)
    print(txt); nbad += len(P["bad"])
    # IDENTITY
    got, ok, rows = identity(out)
    idbad = [k for k in IDKEYS if not (got[k] or "").startswith("IDENTICAL")]
    print("INT16 IDENTITY (strict, T 0 greedy, under F = THE int16 PIN): %s -> %d of %d IDENTICAL%s" % (
        " | ".join("%s %s" % (k, (got[k] or "NOT RUN")[:90]) for k in IDKEYS), len(ok), len(IDKEYS),
        (" -- VIRHE " + ", ".join(idbad) + (" (the item STOPPED before the speed blocks)" if stopped else "")) if idbad else ""))
    nbad += len(idbad)
    res = dict(pin=P, identity=dict(got=got, ok=len(ok), n=len(IDKEYS)), stopped=stopped)
    # 1u ABBA
    U = u1(out, want, xb, swlg, emd5, sw)
    res["u1"] = U
    for b in U["blocks"]:
        print("INT16 1u BLOCK %s %s (%s, ds41_rfuse/ds41_mix_fin/route_copy_ov %s; one discarded + %d back to back, resprobe essay 512, T 0): reads %s tok/s, mean %s | ms a step %s (mean %s) | "
              "stream MB a step %s (mean %s) | %s tok a step, acceptance %s | discarded warm-up %s tok/s at %s MB a step | answers %s%s" % (
                  b["k"], b["arm"], "int15" if b["arm"] == "B" else "int16", "/".join(str(v) for v in want.get(b["arm"], ())), b["n"], "/".join(f(v) for v in b["tps"]), f(b["tps_mean"]),
                  "/".join(f(v) for v in b["ms"]), f(b["ms_mean"]), "/".join(f(v, 1) for v in b["mb"]), f(b["mb_mean"], 1), f(b["tok"]), f(b["acc"], 3), f(b["w_tps"]), f(b["w_mb"], 1),
                  ("IDENTICAL to ref14's essay answer (%s) on all %d reads" % (emd5, len(b["md5"]))) if all(m == emd5 for m in b["md5"]) else "md5 %s" % b["md5"],
                  (" -- VIRHE " + "; ".join(b["bad"][:6])) if b["bad"] else ((" | INFO (mock) " + "; ".join(b["info"][:3])) if b["info"] else " | switches / engagement / ranks OK")))
        nbad += len(b["bad"])
        if b["n"] != u1n: print("INT16 1u BLOCK %s %s: NOTE %d reads (want %d)" % (b["k"], b["arm"], b["n"], u1n))
    ub, uf = U["arms"].get("B") or {}, U["arms"].get("F") or {}
    for a, A in (("B", ub), ("F", uf)):
        if A.get("n"):
            print("INT16 1u ARM %s (%s): %s tok/s = the mean of its %d reads (%s; spread %s %%) | block means %s (spread %s %%) | %s ms a step, %s MB a step, %s tok a step" % (
                a, "int15" if a == "B" else "int16", f(A["tps"]), A["n"], "/".join(f(v) for v in A["tps_reads"]), f(A["spread"]), "/".join(f(v) for v in A["block_means"]),
                f(A["block_spread"]), f(A["ms"]), f(A["mb"], 1), f(A["tok"])))
    d1 = pc(uf.get("tps"), ub.get("tps"))
    if ub.get("n") and uf.get("n"):
        msp = max(nums([ub.get("spread"), uf.get("spread")]) or [0.0])
        dm = abs((uf.get("mb") or 0) - (ub.get("mb") or 0))
        print("INT16 1u F vs B (ABBA, in-process): %s -> %s tok/s (%s), %s -> %s ms a step (%s) | same-arm read spread B %s %% / F %s %%: the delta is %s | block means B %s / F %s (%s) | "
              "stream MB a step %s / %s (%s)" % (
                  f(ub["tps"]), f(uf["tps"]), fp(d1), f(ub["ms"]), f(uf["ms"]), fp(pc(uf["ms"], ub["ms"])), f(ub["spread"]), f(uf["spread"]),
                  "OUTSIDE the larger same-arm spread" if num(d1) and abs(d1) > msp else "INSIDE the larger same-arm spread (%s %%)" % f(msp),
                  "/".join(f(v) for v in ub["block_means"]), "/".join(f(v) for v in uf["block_means"]), sep(ub["block_means"], uf["block_means"]),
                  f(ub["mb"], 1), f(uf["mb"], 1), "matched residency" if dm <= 20.0 else "NOT matched (> 20 MB a step apart): INCONCLUSIVE"))
        print("INT16 1u VERDICT: 1u int16 = %s tok/s (>= %.0f: %s) | B (int15) %s tok/s | F vs B %s | INFO (zzzfuse, single order B then F, NOT ABBA): %s" % (
            f(uf["tps"]), U1TARGET, "YES" if uf["tps"] >= U1TARGET else "NO", f(ub["tps"]), fp(d1),
            ", ".join("%s %s (this %s %s)" % (lab, f(v), "F" if "F (" in lab else "B", fp(pc(uf["tps"] if "F (" in lab else ub["tps"], v))) for lab, v in U1REF)))
    elif not stopped: print("INT16 1u: VIRHE no ABBA reads of both arms (B %s, F %s reads)" % (ub.get("n"), uf.get("n"))); nbad += 1
    # c4 gens
    C = c4(out, steps, want, xb, swlg, sw, gens)
    res["c4"] = C
    for b in C["blocks"]:
        print("INT16 c4 BLOCK %s (%s%s, gens %d): agg %s tok/s, window %s s, %s requests, %s errors, %s tok / %s ms a decode step (harness), acceptance %s, decode_share %s, TTFT p50 %s s | "
              "step log: %s decode steps, p50 %s / mean %s ms | full-row bucket (n_seq 4, n_tok %s): %s steps p50 %s ms | %s seqs / %s rows a step | q %s, depth-0 share %s %% | "
              "newcomer steps in the window %s%s | switches at the end %s, %s%s" % (
                  b["name"], b["arm"], ", a DISCARDED warm-up block: never read" if b["warm"] else (" = int15" if b["arm"] == "B" else " = int16"), gens, f(b["agg"], 1), b.get("window_s"),
                  b.get("requests"), b.get("errors"), f(b["tok"]), f(b["ms_h"], 1), f(b["acc"], 3), b.get("decode_share"), b.get("ttft_p50"), b.get("n_dec"), f(b.get("dec_p50")),
                  f(b.get("dec_mean")), b.get("bkt_rows", "-"), b.get("bkt_n", "-"), f(b.get("bkt_p50")), f(b.get("seqs")), f(b.get("rows")), json.dumps(b.get("q"), sort_keys=True),
                  f(b.get("d0"), 1), b.get("n_nc"), (" [NOTE %s]" % b["nc_note"]) if b.get("nc_note") else "", b.get("sw"), estr(b.get("eng")) + b.get("f1", ""),
                  (" -- VIRHE " + "; ".join(b["bad"][:6])) if b["bad"] else ((" | INFO (mock) " + "; ".join(b["info"][:3])) if b["info"] else "")))
        nbad += len(b["bad"])
    cb, cf = C["arms"].get("B") or {}, C["arms"].get("F") or {}
    for a, A in (("B", cb), ("F", cf)):
        if A.get("aggs"):
            print("INT16 c4 ARM %s (%s): agg %s tok/s (blocks %s, spread %s %%) | decode p50 %s ms (blocks %s, spread %s %%) | full-row bucket (%s rows) p50 %s ms (blocks %s, spread %s %%) | "
                  "%s tok a step, acceptance %s | dominant q %s, depth-0 share %s %%, q %s" % (
                      a, "int15" if a == "B" else "int16", f(A["agg"]), "/".join(f(v, 1) for v in A["aggs"]), f(A["agg_spread"]), f(A["dec_p50"]), "/".join(f(v) for v in A["dec_p50s"]),
                      f(A["dec_spread"]), A.get("bkt_rows"), f(A["bkt_p50"]), "/".join(f(v) for v in A["bkt_p50s"]), f(A["bkt_spread"]), f(A["tok"]), f(A["acc"], 3), A.get("domq"),
                      f(A["d0"], 1), json.dumps(A.get("q"), sort_keys=True)))
    ca = pc(cf.get("agg"), cb.get("agg")); cdp = pc(cf.get("dec_p50"), cb.get("dec_p50")); cbk = pc(cf.get("bkt_p50"), cb.get("bkt_p50"))
    if cb.get("aggs") and cf.get("aggs"):
        asp = max(nums([cb.get("agg_spread"), cf.get("agg_spread")]) or [0.0])
        print("INT16 c4 F vs B (gens window, in-process, the same fixed prompts every block): agg %s -> %s tok/s (%s; same-arm spread B %s %% / F %s %%: %s) | decode p50 %s -> %s ms (%s, "
              "block p50s %s) | full-row bucket (%s rows) p50 %s -> %s ms (%s, %s) | tok a step %s / %s, acceptance %s / %s, depth-0 share %s / %s %%" % (
                  f(cb["agg"]), f(cf["agg"]), fp(ca), f(cb["agg_spread"]), f(cf["agg_spread"]), "OUTSIDE the larger spread" if num(ca) and abs(ca) > asp else "INSIDE the larger spread (%s %%)" % f(asp),
                  f(cb["dec_p50"]), f(cf["dec_p50"]), fp(cdp), sep(cb["dec_p50s"], cf["dec_p50s"]), cb.get("bkt_rows"), f(cb["bkt_p50"]), f(cf["bkt_p50"]), fp(cbk),
                  sep(cb["bkt_p50s"], cf["bkt_p50s"]), f(cb["tok"]), f(cf["tok"]), f(cb["acc"], 3), f(cf["acc"], 3), f(cb["d0"], 1), f(cf["d0"], 1)))
        print("INT16 c4 INFO (other starts, NOT THE SAME METHOD: time windows, not the gens window): %s | this B %s, F %s" % (
            " | ".join("%s %s (this F %s)" % (lab, f(v), fp(pc(cf["agg"], v))) for lab, v in C4REF), f(cb["agg"]), f(cf["agg"])))
    elif not stopped and C["blocks"]: print("INT16 c4: VIRHE no measured gens block of both arms"); nbad += 1
    elif not stopped: print("INT16 c4: not run (skipped by the budget, ZC16_C4=0, or the start did not come up)")
    # 136k
    cr = [r for r in jl(os.path.join(out, "cold.jsonl")) if r.get("depth") == cdep]
    ci = [r for r in rows if r.startswith("xzc16-cold%d-F-vs-" % cdep)]
    if cr:
        r = cr[-1]
        cid = "; ".join(ci) if ci else "not compared"
        cbad = [] if (r.get("status") == 200 and not r.get("cached")) else ["status %s, cached %s" % (r.get("status"), r.get("cached"))]
        if not ci or not all(" IDENTICAL" in x for x in ci): (cbad if sw else []).append("the answer is not IDENTICAL to ref14's (md5 %s): %s" % (cmd5, cid))
        print("INT16 136k (cold deepq %d after a prefix-cache reset, F, %s greedy tokens): TTFT %s s, prompt %s tok, cached %s, status %s | %s vs %s %s s: %s (not the same start) | answer: %s%s" % (
            cdep, r.get("tokens"), f(r.get("ttft")), r.get("prompt_tokens"), r.get("cached"), r.get("status"), "INFO", REF136[0], f(REF136[1]), fp(pc(r.get("ttft"), REF136[1])),
            cid, (" -- VIRHE " + "; ".join(cbad)) if cbad else ""))
        nbad += len(cbad)
        res["c136"] = dict(ttft=r.get("ttft"), d_pct=pc(r.get("ttft"), REF136[1]), ident=ci)
    else: print("INT16 136k: not run (ZC16_COLD=0, dropped by the budget, the identity stop, or no answer)")
    # SUMMARY
    hl = []
    hl.append("identity %d/%d IDENTICAL%s" % (len(ok), len(IDKEYS), "" if not idbad else " (VIRHE: %s)" % ", ".join(idbad)))
    hl.append("pin read-back %s (fn4 every switch + 4 ranks + declare 'now %d' + rco / fuse engagement; %d per-block checks, %d mismatches)" % (
        ("OK" if P["ok"] else "VIRHE") if sw else "INFO (mock, ZC16_SW=0: not served)", rfdecl, P["checks"], P["mismatches"]))
    if ub.get("n") and uf.get("n"):
        hl.append("1u int16 = %s tok/s (>= %.0f: %s) vs B (int15) %s tok/s (%s; read spreads %s / %s %%)" % (f(uf["tps"]), U1TARGET, "YES" if uf["tps"] >= U1TARGET else "NO", f(ub["tps"]), fp(d1), f(ub["spread"]), f(uf["spread"])))
    if cb.get("aggs") and cf.get("aggs"):
        hl.append("c4 gens F vs B: agg %s -> %s tok/s (%s; spreads %s / %s %%), decode p50 %s -> %s ms (%s, %s), full-row p50 %s" % (
            f(cb["agg"]), f(cf["agg"]), fp(ca), f(cb["agg_spread"]), f(cf["agg_spread"]), f(cb["dec_p50"]), f(cf["dec_p50"]), fp(cdp), sep(cb["dec_p50s"], cf["dec_p50s"]), fp(cbk)))
    if res.get("c136"):
        hl.append("136k TTFT %s s (vs %s %s s: %s), answer %s" % (f(res["c136"]["ttft"]), REF136[0], f(REF136[1]), fp(res["c136"]["d_pct"]),
                                                               "IDENTICAL to ref14's" if res["c136"]["ident"] and all(" IDENTICAL" in x for x in res["c136"]["ident"]) else "NOT identical / not compared"))
    print("INT16 SUMMARY%s: %s | %d guard finding(s)%s" % (" (STOPPED at the identity gate)" if stopped else "", " | ".join(hl), nbad, " -- VIRHE see the lines above" if nbad else ""))
    if jout:
        with open(jout, "w") as fo: json.dump(res, fo, default=str)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="zc16sum-")
    try:
        return _selftest(d)
    finally:
        __import__("shutil").rmtree(d, ignore_errors=True)   # the temp dir goes at once (gpu02's /tmp is a tmpfs: MOCK SANDBOX RULE's inodes)


def _selftest(d):
    import contextlib
    os.makedirs(os.path.join(d, "c8"))
    t00 = 1794517000.0
    L = []
    def iso(t):
        x = "%.9f" % (t - int(t))
        return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "." + x[2:] + "Z"
    L.append(iso(t00 - 50) + " deepseek41: lever FUSE-R (ds41_rfuse, RADIANCE_DS41_RFUSE, now 2) declared: beside the router of 40 MoE layers -- router_scored_topk_scatter (form 1, the selection + the sort) at 40, router_gemm_f32_scored_topk_scatter (form 2, the projection too, <= 48 rows) at 40")
    ctr_ = dict(rf_declared=3, rf_layers=40, rf_form1_passes=0, rf_form2_passes=0, mixfin_launches=0, rgs_launches=0)
    rco = dict(recordings=0, copies=0, freed=0)
    ranks = lambda g=75: {"experts": {"ranks": [{"rank": r, "place_swl": 1, "heat_min_gain_x100": g, "heat_hyst_x100": 125, "place_lf": 0} for r in range(4)]}}
    def snap(nm, sw4, run=None, pend=None):
        json.dump({"switches": dict(ds41_rfuse=0, ds41_mix_fin=0, route_copy_ov=0, exl3_x3d=2), "fuse": dict(ctr_), "rco": dict(rco, in_force=0)}, open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        rf, mf, rc = (run if run else sw4[:3])
        if rf == 2: ctr_["rf_form2_passes"] += 560; ctr_["rgs_launches"] += 2240
        if rc == 1: rco["recordings"] += 14; rco["copies"] += 970; rco["freed"] += 970
        w = dict(ds41_rfuse=sw4[0], ds41_mix_fin=sw4[1], route_copy_ov=sw4[2], exl3_x3d=sw4[3], ds41_bi=6, gg_pf=2, ar_xpush=1, gg_gemv_m16=1, place_swap_lent=1, heat_min_gain_x100=75, heat_hyst_x100=125)
        json.dump({"switches": w, "fuse": dict(ctr_), "rco": dict(rco, in_force=sw4[2]), "pending": pend or {}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
        json.dump(ranks(), open(os.path.join(d, "st-%s-1.json" % nm), "w"))
    # the pin + engagement
    snap("xzc16-n5-eng", (2, 0, 1, 2))
    os.rename(os.path.join(d, "fn-xzc16-n5-eng-1.json"), os.path.join(d, "fn-xzc16-n5-eng-1.json.tmp"))
    os.rename(os.path.join(d, "fn-xzc16-n5-eng-1.json.tmp"), os.path.join(d, "fn-xzc16-n5-eng-1.json"))
    json.dump(json.load(open(os.path.join(d, "fn-xzc16-n5-eng-1.json"))), open(os.path.join(d, "fn-xzc16-n5-pin.json"), "w"))
    json.dump(ranks(), open(os.path.join(d, "st-xzc16-n5-pin.json"), "w"))
    rb = open(os.path.join(d, "fuse-readback.txt"), "w")
    with open(os.path.join(d, "identity.txt"), "w") as fo:
        fo.write("xzc16-n5 warm-up IDENTICAL: 3 answers\n")
        for k in IDKEYS[1:]: fo.write("xzc16-%s IDENTICAL: 20 of 20 answers\n" % k)
    # 1u ABBA
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        i = 0
        for k, arm, tps in ((1, "B", (80.0, 89.8, 89.9, 89.85)), (2, "F", (90.3, 90.5, 90.45, 90.4)), (3, "F", (90.4, 90.5, 90.55, 90.5)), (4, "B", (89.9, 89.9, 89.95, 90.0))):
            sw4 = (0, 0, 0, 2) if arm == "B" else (2, 0, 1, 2)
            for n, t in enumerate(tps):
                lv = "U1W" if n == 0 else "U1"; nm = "b%d-%s-%s" % (i, lv, arm); i += 1
                snap(nm, sw4); rb.write("2026-11-13 01:00:00 arm %s ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d %s (want %s)\n" % (arm, " ".join(map(str, sw4)), " ".join(map(str, sw4))))
                fo.write(json.dumps(dict(i=i, lever=lv, arm=arm, name=nm, set="U1-%d" % k, k=k, j=dict(prompt="essay", decode_tps=t, ms_per_decode_step=round(1000 * 1.58 / t, 2), tokens_per_step=1.58,
                         acceptance=0.698, status=200, tokens=512, text_md5="e"), st=dict(stream_mb_step=50.0 if n == 0 else 1.0, place_swl=[1] * 4, mg=[75] * 4, hy=[125] * 4, place_lf=[0] * 4))) + "\n")
    # c4 gens blocks: warm-up wF, then B F F B
    rows = []
    t = t00
    for i_, arm, dec_ms in (("w1", "wF", 61.0), ("1", "B", 60.2), ("2", "F", 59.9), ("3", "F", 59.95), ("4", "B", 60.25)):
        nm = "c4-%s-G-%s" % (i_, arm)
        a = arm[1:] if arm.startswith("w") else arm
        sw4 = (0, 0, 0, 2) if a == "B" else (2, 0, 1, 2)
        w0 = t + 5.0; step = 0; nc = 0; since = 99; post = 0
        while not (nc >= 12 and post >= 20):
            if nc >= 12: post += 1
            if since >= 30 and t > w0 + 0.3 and nc < 12:
                L.append("%s I step %d MIXED n_seq=4 n_tok=1066  [0] slot=1 q=4 ctx=900  [1] slot=2 q=4 ctx=900  [2] slot=3 q=4 ctx=900  [3] slot=4 q=1054 ctx=0" % (iso(t), step)); t += 0.45; nc += 1; since = 0
            else:
                since += 1
                qd = 1 if step % 50 == 7 else 4
                L.append("%s I step %d decode n_seq=4 n_tok=%d  [0] slot=1 q=%d ctx=900  [1] slot=2 q=4 ctx=900  [2] slot=3 q=4 ctx=900  [3] slot=4 q=4 ctx=900" % (iso(t), step, 12 + qd, qd)); t += dec_ms / 1000.0
            step += 1
        w1 = t - 0.2; t += 10.0
        rows.append(dict(name=nm, n=4, arm=arm, sched="G", agg_dec_tps=round(176.0 * 60.2 / dec_ms, 1), tok_per_decode_step=13.2, ms_decode_step=dec_ms, acceptance=0.73, marks=[w0, w1],
                         window_s=round(w1 - w0, 1), requests=12, errors=0, decode_share=0.8, ttft_p50=0.6))
        snap(nm, sw4); rb.write("2026-11-13 01:00:00 arm %s ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d %s (want %s)\n" % (a, " ".join(map(str, sw4)), " ".join(map(str, sw4))))
    rb.close()
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "cold.jsonl"), "w") as fo: fo.write(json.dumps(dict(status=200, prompt_tokens=137135, cached=0, ttft=24.3, tokens=64, depth=136000, name="cold-136000-1-F", arm="F")) + "\n")
    with open(os.path.join(d, "identity.txt"), "a") as fo: fo.write("xzc16-cold136000-F-vs-ref14 IDENTICAL (293 chars)\n")
    lp = os.path.join(d, "eng.log.gz")
    with gzip.open(lp, "wt") as fo: fo.write("\n".join(L) + "\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = run([d, "--log", lp, "--sw", "1", "--essay-md5", "e", "--json", os.path.join(d, "o.json")])
    t1 = buf.getvalue()
    o = json.load(open(os.path.join(d, "o.json")))
    why = []
    if rc != 0: why.append("rc %s" % rc)
    if not o["pin"]["ok"]: why.append("pin %s" % o["pin"]["bad"])
    u = o["u1"]["arms"]
    if abs((u["F"]["tps"] or 0) - 90.48) > 0.01 or abs((u["B"]["tps"] or 0) - 89.9) > 0.01 or u["F"]["n"] != 6: why.append("1u arms %s" % u)
    if "1u int16 = 90.48 tok/s (>= 90: YES)" not in t1 or "B (int15) 89.90 tok/s" not in t1: why.append("1u verdict")
    if "matched residency" not in t1: why.append("1u residency")
    c = o["c4"]["arms"]
    if not c.get("B") or abs((c["B"]["dec_p50"] or 0) - 60.22) > 0.05 or abs((c["F"]["dec_p50"] or 0) - 59.92) > 0.05: why.append("c4 decode p50 %s / %s" % (c.get("B", {}).get("dec_p50"), c.get("F", {}).get("dec_p50")))
    if c.get("F", {}).get("bkt_rows") != 16 or c["F"].get("domq") != "4": why.append("c4 bucket / q %s %s" % (c.get("F", {}).get("bkt_rows"), c.get("F", {}).get("domq")))
    bl = {b["name"]: b for b in o["c4"]["blocks"]}
    if not bl or bl["c4-1-G-B"].get("n_nc") != 12 or not (0 < (bl["c4-1-G-B"].get("d0") or 0) < 5): why.append("c4 newcomers / depth-0 %s" % [(b.get("n_nc"), b.get("d0")) for b in bl.values()])
    if any(b["bad"] for b in o["c4"]["blocks"]) or any(b["bad"] for b in o["u1"]["blocks"]): why.append("a block flagged %s" % [b["bad"] for b in o["c4"]["blocks"] + o["u1"]["blocks"] if b["bad"]])
    for s_ in ("INT16 PIN", "INT16 IDENTITY", "5 of 5 IDENTICAL", "INT16 c4 F vs B", "SEPARATED", "INT16 c4 INFO (other starts, NOT THE SAME METHOD", "INT16 136k", "answer IDENTICAL to ref14's",
               "INT16 SUMMARY", "0 guard finding(s)", "a DISCARDED warm-up block"):
        if s_ not in t1: why.append("missing %r" % s_)
    # the guards: an F 1u read whose form 2 never ran, a B c4 block whose switch read back 2, a rank with heat_min_gain_x100 150, a pin rank wrong, a differing identity
    with open(os.path.join(d, "blocks.jsonl"), "a") as fo:
        snap("b99-U1-F", (2, 0, 1, 2), run=(0, 0, 0))
        fo.write(json.dumps(dict(lever="U1", arm="F", name="b99-U1-F", k=3, j=dict(decode_tps=90.5, ms_per_decode_step=17.4, tokens_per_step=1.58, text_md5="e", status=200, tokens=512),
                                 st=dict(stream_mb_step=0.0, place_swl=[1] * 4, mg=[150] * 4, hy=[125] * 4, place_lf=[0] * 4))) + "\n")
    f1 = json.load(open(os.path.join(d, "fn-c4-4-G-B-1.json"))); f1["switches"]["ds41_rfuse"] = 2; json.dump(f1, open(os.path.join(d, "fn-c4-4-G-B-1.json"), "w"))
    json.dump(ranks(150), open(os.path.join(d, "st-xzc16-n5-pin.json"), "w"))
    with open(os.path.join(d, "identity.txt"), "a") as fo: fo.write("xzc16-ld-F-vs-a1-F DIFFERENT at char 17 (4 vs 4 answers)\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--log", lp, "--sw", "1", "--essay-md5", "e"])
    t2 = buf.getvalue()
    for s_ in ("form 2 did not run", "heat_min_gain_x100 150 (want 75)", "switches [2, 0, 0, 2] at the block's end (want [0, 0, 0, 2])", "VIRHE ld-F-vs-a1-F", "INT16 PIN (THE int16 PIN"):
        if s_ not in t2: why.append("guard %r did not fire" % s_)
    if " -- VIRHE" not in [l for l in t2.split("\n") if l.startswith("INT16 PIN")][0]: why.append("pin guard")
    # the mock (sw 0): mismatches are INFO, never VIRHE
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--log", lp, "--sw", "0", "--essay-md5", "e"])
    t3 = buf.getvalue()
    if "INFO (mock" not in t3 or "VIRHE" in [l for l in t3.split("\n") if l.startswith("INT16 PIN")][0] or "pin read-back INFO (mock" not in t3: why.append("mock INFO")
    # the identity stop: no speed data -> no VIRHE for the missing sections
    d2 = os.path.join(d, "stop"); os.makedirs(d2)
    with open(os.path.join(d2, "identity.txt"), "w") as fo: fo.write("xzc16-n5 warm-up IDENTICAL: 3 answers\nxzc16-a1-F-vs-ref12-a1 DIFFERENT at char 3 (20 vs 20 answers)\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = run([d2, "--log", lp, "--sw", "1", "--stopped", "1"])
    t4 = buf.getvalue()
    if rc != 0 or "STOPPED at the identity gate" not in t4 or "INT16 1u: VIRHE" in t4 or "INT16 c4: VIRHE" in t4: why.append("stop path %s" % t4[-400:])
    # the form-1 fallback rule (ds41_rfuse 2): with a prefill INFO, with none or past the bound VIRHE; never under ds41_rfuse 1 / 0's own rules
    dd = dict(rf_form1_passes=19, rf_form2_passes=1920, rgs_launches=7680, mixfin_launches=0, rco_in_force=1, rco_recordings=14, rco_copies=970, rco_freed=970)
    if engaged((2, 0, 1), dd, prefills=20) or "EXPECTED fallback" not in f1fallback((2, 0, 1), dd, 20): why.append("expected form-1 fallback flagged / not noted")
    if not any("no prefill" in x for x in engaged((2, 0, 1), dd, prefills=0)): why.append("form 1 in a decode-only span not flagged")
    if not any("beyond the fallback" in x for x in engaged((2, 0, 1), dict(dd, rf_form1_passes=KF1 * 3 + 1), prefills=3)): why.append("form 1 past the bound not flagged")
    if f1fallback((1, 0, 0), dd, 20) or f1fallback((2, 0, 1), dict(dd, rf_form1_passes=0), 20): why.append("f1fallback outside ds41_rfuse 2's fallback")
    if why:
        print(t1); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: the form-1 fallback rule (expected with a prefill: INFO; none / past the bound: VIRHE), the pin read-back (declare 'now 2', fn4 switches, 4 ranks, engagement, per-block read-backs), identity 5 keys, 1u ABBA B F F B "
          "(per block, arm means of 6 reads, F vs B, the 90 verdict, residency), c4 gens wF + B F F B (step-log decode p50, the 16-row bucket, q histogram / "
          "depth-0 share, 12 newcomers a window, separation), 136k vs ref14, the guards (form 2 not run, a stale rank value, a wrong switch, a differing "
          "identity), the mock's INFO, the identity-stop path")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
