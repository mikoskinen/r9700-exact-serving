#!/usr/bin/env python3
"""zq14sum.py (13.11.2026, queue-ar/dsmain/zrq14, a builder for ds-main-port) -- the summary of zrq14 (ref14's shortfall vs ref12:
int14 or MAXSEQS 16 or the block window?), item 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14.sh, body zrq14/dsm-zrq14.sh. Standard library only.

  zq14sum.py wins OUTDIR START
      prints p12win.py rows' --win arguments (one per line, "--win" then "NAME:T0:T1") for START's mb blocks (mb.jsonl names
      c<n>-<START>-<i>-S-<arm>, their two marks) and 1u reads (blocks.jsonl rows with start START, their utc0 / utc1).
  zq14sum.py run OUTDIR [--sw 1]
      OUTDIR/mb.jsonl (mbx rows), blocks.jsonl (blockx rows), rows-<START>.json (p12win.py rows over START's engine log), the
      fn-<name>-0/1.json + st-<name>-0/1.json snapshots (GET /admin/fn4 + GET /stats at each block's / read's edges), gate2.json
      (start 2's rule, values, decision), identity.txt. Lines "ZQ14 ...":
        ZQ14 BLOCK / ZQ14 READ  every block and read: tok/s, tokens a step, ms a step, acceptance; the step log's decode steps, rows a
                                step, ms p50 / mean by row bucket, the depth-0 share (sequence-steps verifying 1 row) and the q histogram;
                                the draft controller over the block (GET /admin/fn4 draft / accept / lend / cap deltas, the switches in
                                force; GET /stats draft_conf cuts and the per-position acceptance)
        ZQ14 c1 / c4 / 1u       Z vs B in-process (values, %, spread); the window bridge (Z at ref14's 35 s vs Z at ref12's 20 s);
                                B @ MAXSEQS 16 vs B @ 8 (start 2); ref12 / prof1 / ref14 as INFO
        ZQ14 GATE / VERDICT     start 2's rule and values; the one-line verdict per reading and overall
  zq14sum.py selftest
START is m16 (start 1, MAXSEQS 16) or m8 (start 2, MAXSEQS 8)."""
import json, os, re, statistics, sys, tempfile

# references (other starts: INFO only, never re-run). ref12 = RESULT xref12 (int12, MAXSEQS 8, c1 / c4 20 s windows); ref14 =
# RESULT xref14 (int14 = Z, MAXSEQS 16, c1 / c4 35 s windows); prof1 = RESULT xp1 (int12 = B bytes, MAXSEQS 16, 1u warm essay).
REF = dict(c1=("ref12", 124.90), c4=("ref12", 173.20), u1=("prof1", 91.01), u1w=("ref12 warm", 90.46), u1s=("ref12 switched", 82.24))
R14 = dict(c1=114.2, c4=167.4, u1=89.85)
NAME = re.compile(r"^c(\d+)-(m\d+)-(\w+)-S-(w?[A-Z]+)$")
DRAFTK = ("zero_cycles", "probe_cycles", "zero_moves", "occ_cycles", "occ_moves", "hybrid_lookup", "hybrid_head", "hybrid_unahead")
LENDK = ("lent_cycles", "lent_plain_steps", "lent_probe_steps", "transit_steps", "lends", "reclaims", "borrow_cycles", "hyst_holds")
CAPK = ("steps_capped", "drafts_trimmed", "ahead_capped", "steps_over", "steps_gt48")
CONFK = ("rows_bounded", "rows_gated", "cuts", "cut_drafts", "cut_full")
SWK = ("draft_depth0", "draft_occ", "draft_occ_k", "draft_adaptive", "draft_lend", "draft_probe", "draft_probe_backoff", "sched_row_cap",
       "dspark_conf", "dspark_conf_rows", "ds41_bi", "gg_gemv_m16")
BUCKETS = ["1-4", "5-8", "9-15", "16", ">16"]


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


def jload(p):
    try: return json.load(open(p))
    except Exception: return None


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mean(v):
    v = [x for x in v if num(x)]
    return statistics.mean(v) if v else None


def spread(v):
    v = [x for x in v if num(x)]
    if len(v) < 2: return None
    m = statistics.mean(v)
    return 100.0 * (max(v) - min(v)) / m if m else None


def pct(c, b):
    return 100.0 * (c - b) / b if num(c) and num(b) and b else None


def f(v, nd=2):
    return ("%%.%df" % nd) % v if num(v) else "-"


def fp(v):
    return ("%+.2f %%" % v) if num(v) else "-"


def sl(v, nd=1):
    return "/".join(f(x, nd) for x in v) if v else "-"


def sp(v):
    s = spread(v)
    return ("%.2f %%" % s) if s is not None else "- (one)"


# ---------------------------------------------------------------- inputs
def mbrows(o):
    """mb.jsonl rows by name (the last row of a name wins), with (n, start, i, arm) parsed from the name."""
    d = {}
    for r in jl(os.path.join(o, "mb.jsonl")):
        m = NAME.match(r.get("name") or "")
        if not m: continue
        r["_n"], r["_start"], r["_i"], r["_arm"] = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        d[r["name"]] = r
    return d


def reads(o):
    return [r for r in jl(os.path.join(o, "blocks.jsonl")) if isinstance(r.get("j"), dict)]


def rowsw(o, start):
    d = jload(os.path.join(o, "rows-%s.json" % start)) or {}
    return d.get("windows") or {}


def wins(argv):
    o, start = argv[0], argv[1]
    out = []
    for nm, r in sorted(mbrows(o).items(), key=lambda kv: (kv[1].get("marks") or [0])[0]):
        mk = r.get("marks") or []
        if r["_start"] == start and len(mk) >= 2 and num(mk[0]) and num(mk[1]): out.append("%s:%.6f:%.6f" % (nm, mk[0], mk[1]))
    for r in reads(o):
        if r.get("start") == start and num(r.get("utc0")) and num(r.get("utc1")): out.append("%s:%.6f:%.6f" % (r.get("name"), r["utc0"], r["utc1"]))
    for w in out: print("--win"); print(w)
    return 0


def ctrl(o, nm):
    """the draft controller over one block / read: GET /admin/fn4 + GET /stats deltas between the NAME-0 / NAME-1 snapshots."""
    f0, f1 = jload(os.path.join(o, "fn-%s-0.json" % nm)), jload(os.path.join(o, "fn-%s-1.json" % nm))
    s0, s1 = jload(os.path.join(o, "st-%s-0.json" % nm)), jload(os.path.join(o, "st-%s-1.json" % nm))
    if not isinstance(f1, dict): return None
    f0 = f0 if isinstance(f0, dict) else {}

    def dd(a, b, keys):
        a, b = a or {}, b or {}
        return {k: b.get(k) - (a.get(k) or 0) for k in keys if num(b.get(k))}
    c = dict(draft=dd(f0.get("draft"), f1.get("draft"), DRAFTK),
             accept=dd(f0.get("accept"), f1.get("accept"), ("head_drafted", "head_accepted", "lookup_drafted", "lookup_accepted")),
             lend=dd(f0.get("lend"), f1.get("lend"), LENDK), lend_state={k: (f1.get("lend") or {}).get(k) for k in ("state", "backoff_k", "probe_interval")},
             cap=dd(f0.get("cap"), f1.get("cap"), CAPK), sw={k: (f1.get("switches") or {}).get(k) for k in SWK}, epoch=f1.get("epoch"))
    if isinstance(s1, dict):
        s0 = s0 if isinstance(s0, dict) else {}
        c0, c1 = ((s0.get("draft_conf") or {}).get("conf") or {}), ((s1.get("draft_conf") or {}).get("conf") or {})
        c["conf"] = dd(c0, c1, CONFK)
        h0, h1 = c0.get("cut_hist") or [], c1.get("cut_hist") or []
        if h1: c["conf"]["cut_hist"] = [h1[i] - (h0[i] if i < len(h0) else 0) for i in range(len(h1))]
        p0, p1 = s0.get("draft_positions") or [], s1.get("draft_positions") or []
        pos = []
        for i, b in enumerate(p1):
            a = p0[i] if i < len(p0) else {}
            r_, ac = (b.get("reached") or 0) - (a.get("reached") or 0), (b.get("accepted") or 0) - (a.get("accepted") or 0)
            pos.append([r_, round(ac / r_, 3) if r_ else None])
        c["pos"] = pos
    return c


def ctrl_say(c):
    if not c: return "controller: no GET /admin/fn4 snapshots"
    a = c["accept"]
    hd, ha = a.get("head_drafted"), a.get("head_accepted")
    s = "controller over the block: draft %s | accept head %s/%s (%s) lookup %s/%s | lend %s state %s backoff_k %s | cap %s" % (
        " ".join("%s +%s" % (k, v) for k, v in c["draft"].items()), hd, ha, f(ha / hd, 3) if num(hd) and hd and num(ha) else "-",
        a.get("lookup_drafted"), a.get("lookup_accepted"), " ".join("%s +%s" % (k, v) for k, v in c["lend"].items() if v),
        c["lend_state"].get("state"), c["lend_state"].get("backoff_k"), " ".join("%s +%s" % (k, v) for k, v in c["cap"].items()))
    if "conf" in c:
        s += " | draft_conf %s" % " ".join("%s +%s" % (k, v) for k, v in c["conf"].items())
    if c.get("pos"):
        s += " | per-position acceptance %s" % " ".join("p%d %s (n %s)" % (i + 1, f(r, 3), n) for i, (n, r) in enumerate(c["pos"]) if n)
    s += " | in force: %s, epoch %s" % (" ".join("%s %s" % (k, v) for k, v in c["sw"].items()), c.get("epoch"))
    return s


def rows_say(w):
    if not w: return "step log: no window"
    if not w.get("decode_steps"): return "step log: VIRHE no decode step in the window"
    b = w.get("buckets") or {}
    return ("step log: %d decode steps, %s sequences / %s rows a step, ms p50 %s mean %s | %s | depth-0 share %s (sequence-steps verifying 1 row), "
            "steps all depth 0 %s, q %s" % (
                w["decode_steps"], w.get("seqs_mean"), w.get("rows_mean"), w.get("ms_p50"), w.get("ms_mean"),
                " | ".join("%s n %d (%.0f %%) p50 %s mean %s" % (k, b[k]["n"], 100 * b[k]["share"], b[k]["p50"], b[k]["mean"]) for k in BUCKETS if (b.get(k) or {}).get("n")),
                ("%.1f %%" % (100 * w["depth0_share"])) if num(w.get("depth0_share")) else "-",
                ("%.1f %%" % (100 * w["all_depth0_steps"])) if num(w.get("all_depth0_steps")) else "-", json.dumps(w.get("q_hist"), sort_keys=True)))


def domq(w):
    """the dominant q (verify rows a sequence) of a window, or None"""
    q = (w or {}).get("q_hist") or {}
    return max(q, key=lambda k: q[k]) if q else None


# ---------------------------------------------------------------- the verdict
def attrib(gap, comps):
    """gap: ref14's % vs its reference; comps: [(label, d %, threshold %)] -> the component that explains it, or 'unexplained'.
    A component explains when it has the gap's sign, clears its threshold (the noise) and is >= 30 % of the gap."""
    if not num(gap): return "unexplained (no reference gap)", []
    ok = [(lab, d, t) for lab, d, t in comps if num(d) and d * gap > 0 and abs(d) >= t and abs(d) >= 0.3 * abs(gap)]
    if not ok: return "unexplained", []
    ok.sort(key=lambda x: -abs(x[1]))
    return ok[0][0] + ("" if len(ok) == 1 else " (+ " + ", ".join(x[0] for x in ok[1:]) + ")"), ok


def run(argv):
    o = argv[0]
    sw = "1"
    if "--sw" in argv: sw = argv[argv.index("--sw") + 1]
    mb, rd = mbrows(o), reads(o)
    rw = {s: rowsw(o, s) for s in ("m16", "m8")}
    gate = jload(os.path.join(o, "gate2.json")) or {}
    L = []
    # every block / read
    for nm, r in sorted(mb.items(), key=lambda kv: (kv[1].get("marks") or [0])[0]):
        w = rw.get(r["_start"], {}).get(nm)
        L.append("ZQ14 BLOCK %s c%d %s arm %s%s (MAXSEQS %s, window %s s, %s requests, %s errors): agg %s tok/s, %s tok / %s ms a decode step, acceptance %s, "
                 "decode_share %s | %s | %s" % (
                     r["_start"], r["_n"], nm, r["_arm"], " (DISCARDED warm-up)" if r["_arm"].startswith("w") else "", r["_start"][1:], r.get("window_s"),
                     r.get("requests"), r.get("errors"), r.get("agg_dec_tps"), r.get("tok_per_decode_step"), r.get("ms_decode_step"), r.get("acceptance"),
                     r.get("decode_share"), rows_say(w), ctrl_say(ctrl(o, nm))))
    for r in rd:
        j, st, nm = r["j"], r.get("st") or {}, r.get("name")
        w = rw.get(r.get("start"), {}).get(nm)
        L.append("ZQ14 READ %s %s set %s arm %s%s (1u essay, MAXSEQS %s): %s tok/s, %s tok / %s ms a decode step, acceptance %s, answer %s | stream %s MB a step, "
                 "promotions %s | %s | %s" % (
                     r.get("start"), nm, r.get("set"), r.get("arm"), " (DISCARDED warm-up read)" if r.get("lever") == "U1W" else "", (r.get("start") or "m?")[1:],
                     j.get("decode_tps"), j.get("tokens_per_step"), j.get("ms_per_decode_step"), j.get("acceptance"), j.get("text_md5"),
                     st.get("stream_mb_step"), st.get("promotions"), rows_say(w), ctrl_say(ctrl(o, nm))))

    def blk(n, start, arm, i=None):
        return [r for nm, r in sorted(mb.items()) if r["_n"] == n and r["_start"] == start and r["_arm"] == arm and r["_i"][:1].isdigit() and (i is None or r["_i"] == i)]

    def bridge(n):
        return [r for nm, r in sorted(mb.items()) if r["_n"] == n and r["_start"] == "m16" and r["_i"].startswith("x")]

    def rset(s):
        return [r for r in rd if r.get("set") == s and r.get("lever") == "U1"]

    res = {}
    for n in (1, 4):
        k = "c%d" % n
        B = [r.get("agg_dec_tps") for r in blk(n, "m16", "B")]; Z = [r.get("agg_dec_tps") for r in blk(n, "m16", "Z")]
        B8 = [r.get("agg_dec_tps") for r in blk(n, "m8", "B")]; X = [r.get("agg_dec_tps") for r in bridge(n)]
        mB, mZ, mB8, mX = mean(B), mean(Z), mean(B8), mean(X)
        tps = lambda rs: f(mean([x.get("tok_per_decode_step") for x in rs]))
        mss = lambda rs: f(mean([x.get("ms_decode_step") for x in rs]))
        acc = lambda rs: f(mean([x.get("acceptance") for x in rs]), 3)
        qd = lambda rs, st: "/".join(str(domq(rw.get(st, {}).get(x["name"]))) for x in rs) or "-"
        d0 = lambda rs, st: "/".join(f(100 * (rw.get(st, {}).get(x["name"]) or {}).get("depth0_share"), 1) if num((rw.get(st, {}).get(x["name"]) or {}).get("depth0_share")) else "-" for x in rs) or "-"
        rB, rZ, rB8, rX = blk(n, "m16", "B"), blk(n, "m16", "Z"), blk(n, "m8", "B"), bridge(n)
        thr = max(2 * max([x for x in (spread(B), spread(Z)) if x is not None] or [0.25]), 0.5)
        res[k] = dict(B=mB, Z=mZ, B8=mB8, X=mX, thr=thr, dint=pct(mZ, mB), dwin=pct(mX, mZ), dms=pct(mB, mB8))
        rname, rv = REF[k]
        L.append("ZQ14 %s Z vs B (in-process, MAXSEQS 16, ref12's %s s window, the same fixed prompts each block): B %s (%s, spread %s) -> Z %s (%s, spread %s): %s "
                 "(noise bar %.2f %%) | tok a step B %s / Z %s, ms a step B %s / Z %s, acceptance B %s / Z %s | dominant q a block B %s / Z %s, depth-0 %% B %s / Z %s" % (
                     k, f(mean([x.get("window_s") for x in rB + rZ]), 0), f(mB), sl(B), sp(B), f(mZ), sl(Z), sp(Z), fp(pct(mZ, mB)), thr, tps(rB), tps(rZ), mss(rB), mss(rZ),
                     acc(rB), acc(rZ), qd(rB, "m16"), qd(rZ, "m16"), d0(rB, "m16"), d0(rZ, "m16")))
        if X:
            L.append("ZQ14 %s WINDOW BRIDGE (Z at ref14's %s s window vs Z at ref12's window, in-process): %s (%s) vs %s: %s | tok a step %s vs %s, acceptance %s vs %s, "
                     "dominant q %s | INFO ref14 (Z, 35 s, another start) %s: bridge %s" % (
                         k, f(mean([x.get("window_s") for x in rX]), 0), f(mX), sl(X), f(mZ), fp(pct(mX, mZ)), tps(rX), tps(rZ), acc(rX), acc(rZ), qd(rX, "m16"),
                         f(R14[k], 1), fp(pct(mX, R14[k]))))
        if B8:
            L.append("ZQ14 %s B @ MAXSEQS 16 vs B @ MAXSEQS 8 (two fresh starts, same image / pin / window): %s (%s) vs %s (%s, spread %s): %s (cross-start bar 1.5 %%) | "
                     "tok a step %s vs %s, ms a step %s vs %s, acceptance %s vs %s | dominant q a block @16 %s / @8 %s, depth-0 %% @16 %s / @8 %s" % (
                         k, f(mB), sl(B), f(mB8), sl(B8), sp(B8), fp(pct(mB, mB8)), tps(rB), tps(rB8), mss(rB), mss(rB8), acc(rB), acc(rB8),
                         qd(rB, "m16"), qd(rB8, "m8"), d0(rB, "m16"), d0(rB8, "m8")))
        L.append("ZQ14 %s INFO vs the references (other starts): %s %s (same window): B@16 %s, Z@16 %s%s | ref14 %s (Z, 35 s window): Z@16 (20 s) %s%s" % (
            k, rname, f(rv), fp(pct(mB, rv)), fp(pct(mZ, rv)), (", B@8 %s" % fp(pct(mB8, rv))) if num(mB8) else "", f(R14[k], 1), fp(pct(mZ, R14[k])),
            (", Z@16 at 35 s %s" % fp(pct(mX, R14[k]))) if num(mX) else ""))
    # 1u
    B1, Z1, B2, B8 = ([r["j"].get("decode_tps") for r in rset(s)] for s in ("B1", "Z1", "B2", "B8"))
    mB1, mZ1, mB2, mB8u = mean(B1), mean(Z1), mean(B2), mean(B8)
    mBb = mean([x for x in (mB1, mB2) if num(x)])
    smb = lambda s: sl([(r.get("st") or {}).get("stream_mb_step") for r in rset(s)])
    thr = max(2 * max([x for x in (spread(B1), spread(Z1), spread(B2)) if x is not None] or [0.25]), 0.5)
    res["u1"] = dict(B=mBb, Z=mZ1, B8=mB8u, thr=thr, dint=pct(mZ1, mBb), dms=pct(mB1, mB8u))
    L.append("ZQ14 1u essay Z vs B (in-process, MAXSEQS 16, one discarded read before B and before Z, then B again x%d): B %s (%s, spread %s) -> Z %s (%s, spread %s) "
             "-> B again %s (%s, spread %s) | Z vs the B mean %s %s (noise bar %.2f %%), B drift %s | stream MB a step B %s / Z %s / B again %s" % (
                 len(B2), f(mB1), sl(B1, 2), sp(B1), f(mZ1), sl(Z1, 2), sp(Z1), f(mB2), sl(B2, 2), sp(B2), f(mBb), fp(pct(mZ1, mBb)), thr, fp(pct(mB2, mB1)),
                 smb("B1"), smb("Z1"), smb("B2")))
    if B8:
        L.append("ZQ14 1u essay B @ MAXSEQS 16 vs B @ MAXSEQS 8 (two fresh starts): %s (%s) vs %s (%s, spread %s): %s (cross-start bar 1.5 %%) | stream MB a step %s vs %s" % (
            f(mB1), sl(B1, 2), f(mB8u), sl(B8, 2), sp(B8), fp(pct(mB1, mB8u)), smb("B1"), smb("B8")))
    L.append("ZQ14 1u INFO vs the references (other starts): prof1 91.01 (int12 = B, MAXSEQS 16): B@16 %s, Z@16 %s%s | ref14 89.85 (Z): Z@16 %s | ref12 warm 90.46 "
             "(MAXSEQS 8, one read): B@16 %s | ref12 switched 82.24: not read here (warm reads only) | target >= 90: B %s, Z %s" % (
                 fp(pct(mB1, 91.01)), fp(pct(mZ1, 91.01)), (", B@8 %s" % fp(pct(mB8u, 91.01))) if num(mB8u) else "", fp(pct(mZ1, 89.85)), fp(pct(mB1, 90.46)),
                 "MET" if num(mB1) and mB1 >= 90 else "short", "MET" if num(mZ1) and mZ1 >= 90 else "short"))
    # the gate
    if gate:
        L.append("ZQ14 GATE start 2 (MAXSEQS 8, B only): rule '%s' | values c1 B %s, c4 B %s | %s" % (
            gate.get("rule"), f(gate.get("c1")), f(gate.get("c4")), gate.get("decision")))
    # identity
    idl = []
    try: idl = [l.split()[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "DIFFER" not in l.upper() else "DIFFERENT/not compared") for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: pass
    L.append("ZQ14 IDENTITY (strict, T 0 greedy): %s" % (" | ".join(idl) or "no line"))
    # the verdict
    gaps = dict(c1=pct(R14["c1"], REF["c1"][1]), c4=pct(R14["c4"], REF["c4"][1]), u1=pct(R14["u1"], REF["u1"][1]))
    vs, labs = [], []
    for k in ("c1", "c4", "u1"):
        r = res.get(k) or {}
        comps = [("int14 effect", r.get("dint"), r.get("thr") or 0.5)]
        # 1u: ref14's gap is vs prof1, which ran MAXSEQS 16 too -> MAXSEQS cannot explain it (B16/B8 printed as INFO only)
        if k != "u1": comps += [("MAXSEQS-16 policy effect", r.get("dms"), 1.5), ("window effect", r.get("dwin"), r.get("thr") or 0.5)]
        lab, ok = attrib(gaps[k], comps)
        labs.append(lab.split(" (+")[0])
        vs.append("%s %s (ref14 gap %s; Z/B %s, B16/B8 %s%s)" % (k if k != "u1" else "1u", lab, fp(gaps[k]), fp(r.get("dint")), fp(r.get("dms")),
                                                               ", Z35/Z20 %s" % fp(r.get("dwin")) if k != "u1" else " INFO: prof1 ran MAXSEQS 16 as well"))
    ex = [x for x in labs if x != "unexplained"]
    overall = "unexplained" if not ex else (ex[0] if all(x == ex[0] for x in ex) and len(ex) == len(labs) else "mixed: " + " / ".join(labs))
    L.append("ZQ14 VERDICT %s -- %s%s" % (overall, " | ".join(vs), "" if sw == "1" else " | MOCK (ZQ14_SW=0): NOT A READING"))
    L.append("ZQ14 TABLE c1 B %s Z %s%s%s || c4 B %s Z %s%s%s || 1u B %s Z %s B again %s%s" % (
        f(res["c1"]["B"]), f(res["c1"]["Z"]), (" Z35 %s" % f(res["c1"]["X"])) if num(res["c1"]["X"]) else "", (" B8 %s" % f(res["c1"]["B8"])) if num(res["c1"]["B8"]) else "",
        f(res["c4"]["B"]), f(res["c4"]["Z"]), (" Z35 %s" % f(res["c4"]["X"])) if num(res["c4"]["X"]) else "", (" B8 %s" % f(res["c4"]["B8"])) if num(res["c4"]["B8"]) else "",
        f(mB1), f(mZ1), f(mB2), (" B8 %s" % f(mB8u)) if num(mB8u) else ""))
    for l in L: print(l)
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    d = tempfile.mkdtemp(prefix="zq14st-")
    mb, bl, rows = [], [], {"m16": {}, "m8": {}}
    t = 1000.0

    def addmb(n, start, i, arm, tps, win=20.0, q="3"):
        nonlocal t
        nm = "c%d-%s-%s-S-%s" % (n, start, i, arm)
        mb.append(dict(name=nm, n=n, arm=arm, sched="S", agg_dec_tps=tps, tok_per_decode_step=3.5, ms_decode_step=27.0, acceptance=0.75, marks=[t, t + win],
                       window_s=win, requests=4, errors=0, decode_share=0.86))
        rows[start][nm] = dict(decode_steps=600, seqs_mean=float(n), rows_mean=4.6, ms_p50=28.0, ms_mean=27.0, depth0_share=0.02, all_depth0_steps=0.02,
                               q_hist={q: 500, "1": 10}, buckets={"1-4": dict(n=200, share=0.33, p50=22.4, mean=22.3), "5-8": dict(n=400, share=0.67, p50=29.7, mean=30.3)})
        for k in (0, 1):
            json.dump({"draft": {x: 10 * k for x in DRAFTK}, "accept": {"head_drafted": 1000 * k, "head_accepted": 700 * k, "lookup_drafted": 0, "lookup_accepted": 0},
                       "lend": {"lent_cycles": 0, "state": 0, "backoff_k": 0}, "cap": {x: 0 for x in CAPK}, "switches": {"draft_depth0": 3, "draft_occ": 6}, "epoch": 3},
                      open(os.path.join(d, "fn-%s-%d.json" % (nm, k)), "w"))
            json.dump({"draft_conf": {"conf": dict(rows_bounded=5 * k, rows_gated=2 * k, cuts=k, cut_drafts=k, cut_full=0, cut_hist=[k, 0])},
                       "draft_positions": [{"reached": 100 * k, "accepted": 80 * k}]}, open(os.path.join(d, "st-%s-%d.json" % (nm, k)), "w"))
        t += 60
    for i, a, v in (("1", "B", 125.0), ("2", "Z", 125.6), ("3", "Z", 125.4), ("4", "B", 124.8)): addmb(1, "m16", i, a, v)
    addmb(1, "m16", "x35", "Z", 114.5, 35.0)
    addmb(4, "m16", "w", "wB", 166.0)
    for i, a, v in (("1", "B", 167.0), ("2", "Z", 167.5), ("3", "Z", 167.2), ("4", "B", 167.3)): addmb(4, "m16", i, a, v)
    addmb(4, "m16", "x35", "Z", 166.9, 35.0)
    for i, v in (("1", 124.9), ("2", 125.0)): addmb(1, "m8", i, "B", v)
    addmb(4, "m8", "w", "wB", 172.0, q="4")
    for i, v in (("1", 173.0), ("2", 173.4)): addmb(4, "m8", i, "B", v, q="4")
    nb = 0
    for st, s, lv, a, v in [("m16", "B1", "U1W", "B", 80.0)] + [("m16", "B1", "U1", "B", 91.0)] * 3 + [("m16", "Z1", "U1W", "Z", 85.0)] + [("m16", "Z1", "U1", "Z", 89.9)] * 3 + \
            [("m16", "B2", "U1", "B", 90.9)] * 3 + [("m8", "B8", "U1W", "B", 80.0)] + [("m8", "B8", "U1", "B", 91.1)] * 3:
        nm = "b%d-%s-%s" % (nb, lv, a); nb += 1
        bl.append(dict(i=nb, lever=lv, arm=a, set=s, start=st, name=nm, utc0=t, utc1=t + 6, j=dict(decode_tps=v, tokens_per_step=1.58, ms_per_decode_step=17.4,
                                                                                                   acceptance=0.7, text_md5="47a39dac581a"), st=dict(stream_mb_step=10.0, promotions=5)))
        t += 10
    with open(os.path.join(d, "mb.jsonl"), "w") as fh:
        for r in mb: fh.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fh:
        for r in bl: fh.write(json.dumps(r) + "\n")
    for s in rows: json.dump({"windows": rows[s]}, open(os.path.join(d, "rows-%s.json" % s), "w"))
    json.dump(dict(rule="c1 B mean < 121.0 OR c4 B mean < 170.0", c1=124.9, c4=167.15, decision="RUN (c4 B 167.15 < 170.0)"), open(os.path.join(d, "gate2.json"), "w"))
    open(os.path.join(d, "identity.txt"), "w").write("xzq14-a1-B-vs-ref12-a1 IDENTICAL (20 of 20)\nxzq14-a1-Z-vs-ref12-a1 IDENTICAL (20 of 20)\n")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--sw", "1"])
    out = buf.getvalue()
    wb = io.StringIO()
    with contextlib.redirect_stdout(wb): wins([d, "m16"])
    w = wb.getvalue().split()
    ok = out.count("ZQ14 BLOCK") == len(mb) and out.count("ZQ14 READ") == len(bl) and "ZQ14 VERDICT" in out and "ZQ14 TABLE" in out
    ok = ok and "c1 window effect" in out and "c4 MAXSEQS-16 policy effect" in out and "1u int14 effect" in out
    ok = ok and "ZQ14 GATE" in out and w.count("--win") == sum(1 for r in mb if "-m16-" in r["name"]) + sum(1 for r in bl if r["start"] == "m16")
    ok = ok and "depth-0 share 2.0 %" in out and "per-position acceptance p1 0.800" in out and "draft_conf rows_bounded +5" in out
    for fn in os.listdir(d): os.remove(os.path.join(d, fn))
    os.rmdir(d)
    if not ok:
        print(out[-3000:])
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST OK: %d blocks + %d reads summarised; verdict per reading (c1 window, c4 MAXSEQS, 1u int14 on the synthetic set); wins, gate, identity lines" % (len(mb), len(bl)))
    return 0


def main(argv):
    if not argv: print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    if argv[0] == "wins": return wins(argv[1:])
    if argv[0] == "run": return run(argv[1:])
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
