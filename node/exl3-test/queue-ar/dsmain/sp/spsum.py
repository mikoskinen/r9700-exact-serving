#!/usr/bin/env python3
"""spsum.py (12.11.2026, queue-ar/dsmain/sp, ds-main-port int9 builder) -- the readings of 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp
(lever SP on the ds41-int9 image: fn4 stage_full_min / stage_full_pick, the prefill stager at the FULL plan). Standard library only.

  spsum.py cell --reqs R.jsonl --stats S.jsonl --fn0 A.json --fn1 B.json --arm NAME --size LABEL --sw N,K --json OUT
      one cell of alone one-token prompts (ncone.py's REQS / STATS: /stats snapshot j and j + 1 bracket prompt j): per prompt the
      prompt / cached tokens, the reply time, the prefill step time (/stats: (step_ns - step_ns_decode) over steps_prefill +
      steps_mixed), staged GB (all ranks: link.staged_bytes; rank 0: experts.staged_bytes) and streamed GB per rank
      (experts.ranks[].stream_bytes); the cell's GET "sp" deltas per rank. Freshness: every prompt cached == 0 (else VIRHE).
      Prints "CELL {json}".
  spsum.py need --sw N,K --pts P1,P2,..      the counter guard's need / forbid for a cell of lone prompts of those sizes under
      stage_full_min N / stage_full_pick K: a prompt of <= 2112 tokens never reaches the full plan; one of 2113 .. 4160 is ONE
      full-plan step of its size; a longer one's first step is a full-plan chunk of 4096 .. 4160 rows (the rest below). A step
      under N streams (spstream), else arms (spfull; + sppick when K > 0). A first chunk with N inside 4097 .. 4160 is left
      out of both lists (not asserted). Always pskl. Prints "need/forbid".
  spsum.py pick --cell SIZE:ARM:CELL.json [...] --sizes S1,S2,S3
      the screen's verdict: per size the median prefill step ms of B / S (stage_full_min above the size) / 1 / 2 (stage_full_pick
      1 / 2) and the faster SP form; then ONE composite pin for the later readings: stage_full_min N* = streaming below the first
      size whose fastest SP form is a pick (the midpoint between that size's and the one below's median prompt; 4161 when the
      stream form wins every size; 0 when a pick wins the smallest), stage_full_pick K* = the pick that wins most sizes at or
      above N* (by summed medians; 0 when streaming wins there). Prints "SCREEN ..." and "PICK N K KP" (KP = the faster pick at
      the largest size, the cold 64k's pick arm).
  spsum.py steps --log ENGINE.log.gz --t0 T0 --t1 T1   (coordinator 12.11.: the c8 cell) the step log (RADIANCE_LOG_STEPS=1,
      docker logs -t receipt times; ncsteps.py's regexes and wall rule: a step's wall = the next step line's time - its own, gaps
      over 3 s dropped) inside [T0, T1]: per class -- decode, MIXED / prefill at most 2112 rows (the span level), and the
      CO-BATCHED FULL-PLAN steps (MIXED / prefill of 2113 .. 4160 rows: c8 newcomers co-batched under NOA 2) -- n, rows range, wall
      ms p50 / mean. Prints "STEPS {json}".
  spsum.py summary --out DIR       the item's final lines.
  spsum.py selftest
"""
import argparse, calendar, gzip, json, math, os, re, statistics, sys, time

sys.dont_write_bytecode = True
SPAN, FULL, CHUNK = 2112, 4160, 4096
NAME = {"B": "B", "S": "SP-S", "1": "SP-P1", "2": "SP-P2", "X": "SP-best", "P": "SP-P"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
       12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def num(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict): return 0
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def rows(p):
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


def rd(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return {}


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)


def f1(x, n=1):
    return "-" if x is None else format(x, ".%df" % n)


def pc(a, b):
    return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"


def sp_ranks(fn):
    out = {}
    for r in ((fn or {}).get("sp") or {}).get("ranks") or []:
        if isinstance(r, dict) and isinstance(r.get("rank"), int): out[r["rank"]] = r
    return out


def sp_delta(f0, f1_):
    a, b = sp_ranks(f0), sp_ranks(f1_)
    keys = ("full_steps", "full_rows", "full_bytes", "full_pick_kept", "full_pick_dropped", "stream_steps", "stream_rows")
    return {r: {k: num(b[r], k) - num(a.get(r, {}), k) for k in keys} for r in sorted(b)}


# ------------------------------------------------------------------ steps (ncsteps.py's regexes, read only copies)
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")


def ts_split(line):
    m = TSRE.match(line)
    if not m: return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z": t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


def steps(path, t0, t1, idle=3.0):
    rs = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for l in f:
            if "n_seq=" not in l: continue
            t, rest = ts_split(l.rstrip("\n"))
            m = STEP.search(rest)
            if m and t is not None: rs.append(dict(t=t, ph=m.group(2), n_tok=int(m.group(4))))
    for a, b in zip(rs, rs[1:]):
        d = b["t"] - a["t"]
        a["wall"] = 1000.0 * d if 0 <= d <= idle else None
    win = [r for r in rs if t0 <= r["t"] <= t1 and r.get("wall") is not None]
    def cls(r):
        if r["ph"] == "decode": return "decode"
        return "full" if r["n_tok"] > SPAN else "span"
    out = {}
    for c in ("decode", "span", "full"):
        v = [r for r in win if cls(r) == c]
        w = [r["wall"] for r in v]
        out[c] = dict(n=len(v), rows_min=min((r["n_tok"] for r in v), default=None), rows_max=max((r["n_tok"] for r in v), default=None),
                      ms_p50=round(med(w), 1) if w else None, ms_mean=round(statistics.mean(w), 1) if w else None)
    out["steps"] = len(win)
    return out


# ------------------------------------------------------------------ cell
def cell(reqs, stats, fn0, fn1, arm, size, sw):
    rq = [r for r in rows(reqs)]
    st = {r.get("k"): r.get("s") or {} for r in rows(stats)}
    per, bad = [], []
    for r in rq:
        j = r.get("j")
        a, b = st.get(j), st.get(j + 1) if isinstance(j, int) else None
        if r.get("err") or not a or not b:
            bad.append("prompt %s: %s" % (j, r.get("err") or "no /stats bracket")); continue
        d = lambda *k: num(b, *k) - num(a, *k)   # noqa: E731
        nst = d("steps_prefill") + d("steps_mixed")
        pf_ns = d("step_ns") - d("step_ns_decode")
        ra, rb = {x.get("rank"): x for x in num_list(a)}, {x.get("rank"): x for x in num_list(b)}
        strm = [round((num(rb.get(k, {}), "stream_bytes") - num(ra.get(k, {}), "stream_bytes")) / 1e9, 3) for k in sorted(rb)]
        per.append(dict(j=j, prompt=r.get("prompt"), cached=r.get("cached"), t_resp=r.get("t_resp"), ttft=r.get("ttft"),
                        pf_steps=nst, step_ms=round(pf_ns / 1e6 / nst, 1) if nst > 0 else None,
                        pf_ms=round(pf_ns / 1e6, 1) if nst > 0 else None,
                        staged_gb=round(d("link", "staged_bytes") / 1e9, 3), staged_gb_r0=round(d("experts", "staged_bytes") / 1e9, 3),
                        stream_gb=strm))
    for p in per:
        if (p.get("cached") or 0) != 0: bad.append("not fresh: prompt %s cached %s" % (p["j"], p["cached"]))
    spd = sp_delta(rd(fn0), rd(fn1))
    pts = [p["prompt"] for p in per if p.get("prompt")]
    sm = [p["pf_ms"] for p in per]
    out = dict(arm=arm, size=size, sw=sw, n=len(per), errors=len(rq) - len(per), prompts=pts,
               prompt_med=med(pts), prompt_min=min(pts) if pts else None, prompt_max=max(pts) if pts else None,
               pf_ms_med=med(sm), pf_ms_mean=round(statistics.mean([x for x in sm if x is not None]), 1) if any(x is not None for x in sm) else None,
               step_ms_med=med([p["step_ms"] for p in per]), pf_steps=[p["pf_steps"] for p in per],
               t_resp_med=med([p["t_resp"] for p in per]),
               staged_gb_med=med([p["staged_gb"] for p in per]), staged_gb_r0_med=med([p["staged_gb_r0"] for p in per]),
               stream_gb_rank_med=[med([p["stream_gb"][k] for p in per if len(p["stream_gb"]) > k]) for k in range(4)],
               sp=spd, per=per, guard=bad)
    return out


def num_list(s):
    return ((s or {}).get("experts") or {}).get("ranks") or []


# ------------------------------------------------------------------ need
def full_steps_of(pt):
    """the full-plan step sizes a lone fresh prompt of pt tokens runs, and whether any is uncertain (chunking unknown)"""
    if not pt or pt <= SPAN: return [], False
    if pt <= FULL: return [pt], False
    return [CHUNK], True   # the first chunk: 4096 .. 4160 rows (the scheduler's split), the rest below the full plan


def need(n, k, pts):
    ns = nf = unsure_s = unsure_f = 0
    for pt in pts:
        st, unsure = full_steps_of(pt)
        for r in st:
            lo, hi = (CHUNK, FULL) if unsure else (r, r)   # a long prompt's first chunk: 4096 .. 4160 rows
            if n > 0 and n > hi: ns += 1
            elif n == 0 or n <= lo: nf += 1
            else: unsure_s += 1; unsure_f += 1
    nd, fb = ["pskl"], []
    if ns: nd.append("spstream")
    elif not unsure_s: fb.append("spstream")
    if nf: nd.append("spfull")
    elif not unsure_f: fb.append("spfull")
    if nf and k > 0: nd.append("sppick")
    elif not unsure_f or k == 0: fb.append("sppick")
    return ",".join(nd) + "/" + ",".join(fb)


# ------------------------------------------------------------------ pick
def pick(cells, sizes):
    by = {}
    for c in cells:
        sz, arm, p = c.split(":", 2)
        d = rd(p)
        if d: by.setdefault(sz, {}).setdefault(arm, []).append(d)
    lines, win, meds = [], {}, {}
    for sz in sizes:
        arms = by.get(sz, {})
        m = {}
        for a, ds in arms.items():
            v = [x for d in ds for x in [p.get("pf_ms") for p in d.get("per", [])] if x is not None]
            pt = [x for d in ds for x in d.get("prompts", [])]
            stg = [x for d in ds for x in [p.get("staged_gb") for p in d.get("per", [])] if x is not None]
            sr = [x for d in ds for p in d.get("per", []) for x in p.get("stream_gb", [])[:1]]
            gd = [g for d in ds for g in d.get("guard", [])]
            m[a] = dict(ms=med(v), ci=ci(v), n=len(v), pt=med(pt), ptr=(min(pt) if pt else None, max(pt) if pt else None),
                        staged=med(stg), stream_r0=med(sr), guard=gd)
        meds[sz] = m
        sp = [(m[a]["ms"], a) for a in ("S", "1", "2") if a in m and m[a]["ms"] is not None]
        win[sz] = min(sp)[1] if sp else None
        b = m.get("B", {}).get("ms")
        cells_txt = []
        for a in ("B", "S", "1", "2"):
            if a not in m: continue
            x = m[a]
            cells_txt.append("%s %s ms%s (n %d; %s; staged %s GB all ranks, stream r0 %s GB)%s" % (
                NAME[a], f1(x["ms"]), "" if a == "B" else " " + pc(x["ms"], b), x["n"],
                "+- %s" % f1(x["ci"][1]) if x["ci"][1] is not None else "-", f1(x["staged"], 2), f1(x["stream_r0"], 2),
                (" -- VIRHE " + "; ".join(x["guard"])[:200]) if x["guard"] else ""))
        pts = m.get("B", {}).get("ptr") or (None, None)
        lines.append("size %s (prompts %s..%s tok): %s -> fastest SP form %s%s" % (
            sz, pts[0], pts[1], " | ".join(cells_txt), NAME.get(win[sz], "-"),
            (" (%s vs B)" % pc(m[win[sz]]["ms"], b)) if win[sz] and b else ""))
    # the composite: streaming below the first size a pick wins
    N, K = 4161, 0
    first_pick = next((i for i, sz in enumerate(sizes) if win.get(sz) in ("1", "2")), None)
    if first_pick is not None:
        if first_pick == 0:
            N = 0
        else:
            lo = meds[sizes[first_pick - 1]].get("B", {}).get("pt") or 0
            hi = meds[sizes[first_pick]].get("B", {}).get("pt") or 0
            N = int((lo + hi) / 2) if lo and hi else 0
        above = sizes[first_pick:]
        tot = {k: sum((meds[s].get(k, {}).get("ms") or 1e12) for s in above) for k in ("1", "2")}
        K = 1 if tot["1"] <= tot["2"] else 2
    last = sizes[-1] if sizes else None
    kp = 1
    if last and meds.get(last):
        a1, a2 = meds[last].get("1", {}).get("ms"), meds[last].get("2", {}).get("ms")
        kp = 2 if a1 is not None and a2 is not None and a2 < a1 else 1
    lines.append("composite SP-best: stage_full_min %d, stage_full_pick %d (%s); the cold 64k's pick arm SP-P = stage_full_pick %d" % (
        N, K, "stream every full-plan step" if N == 4161 else ("pick K at every full-plan step" if N == 0 else
                                                              "stream below %d rows, pick %d at / above" % (N, K)), kp))
    return lines, N, K, kp, meds


# ------------------------------------------------------------------ summary
def summary(o):
    out = []
    try:
        scr = [l.rstrip("\n") for l in open(os.path.join(o, "screen.out")) if l.startswith("SCREEN ")]
    except OSError:
        scr = []
    out += scr
    # alone tails: B vs X per size
    al = {}
    for c in rows(os.path.join(o, "cells.jsonl")):
        if c.get("kind") != "tail": continue
        d = rd(c.get("json", ""))
        if not d: continue
        al.setdefault(c["size"], {}).setdefault(c["arm"], []).append(d)
    for sz, arms in al.items():
        parts = []
        b = None
        for a in ("B", "X"):
            ds = arms.get(a, [])
            v = [p.get("pf_ms") for d in ds for p in d.get("per", []) if p.get("pf_ms") is not None]
            tr = [p.get("t_resp") for d in ds for p in d.get("per", []) if p.get("t_resp") is not None]
            pt = [x for d in ds for x in d.get("prompts", [])]
            stg = [p.get("staged_gb") for d in ds for p in d.get("per", []) if p.get("staged_gb") is not None]
            m, h, n = ci(v)
            md = med(v)
            if a == "B": b = md
            gd = [g for d in ds for g in d.get("guard", [])]
            parts.append("%s prefill %s ms median (mean %s +- %s, n %d), reply %s s, staged %s GB%s%s" % (
                NAME[a], f1(md), f1(m), f1(h), n, f1(med(tr), 3), f1(med(stg), 2),
                "" if a == "B" else " (%s vs B)" % pc(md, b), (" -- VIRHE " + "; ".join(gd)[:160]) if gd else ""))
        pts = [x for ds in arms.values() for d in ds for x in d.get("prompts", [])]
        out.append("TAILS size %s (prompts %s..%s tok): %s" % (sz, min(pts) if pts else "-", max(pts) if pts else "-", " | ".join(parts)))
    # the c8 cells (coordinator 12.11.): B vs SP-best, mb c8 at temperature 0, dur 60 warm 15, twice an arm
    c8 = rows(os.path.join(o, "c8.jsonl"))
    if c8:
        for r in c8:
            st = r.get("steps") or {}
            fu = st.get("full") or {}
            out.append("C8 %s (%s): agg %s tok/s, decode_share %s | co-batched full-plan steps %s (%s..%s rows) %s ms p50 / %s mean | span-level non-decode %s at %s ms p50 | decode %s ms p50 | staged %s GB, sp full %s / stream %s | %s" % (
                r.get("name"), NAME.get(r.get("arm"), r.get("arm")), r.get("agg_dec_tps"), r.get("decode_share"), fu.get("n"), fu.get("rows_min"),
                fu.get("rows_max"), fu.get("ms_p50"), fu.get("ms_mean"), (st.get("span") or {}).get("n"), (st.get("span") or {}).get("ms_p50"),
                (st.get("decode") or {}).get("ms_p50"), r.get("staged_gb"), r.get("sp_full"), r.get("sp_stream"), str(r.get("guard"))[:60]))
        parts = []
        for a in ("B", "X"):
            v = [r for r in c8 if r.get("arm") == a]
            if not v: continue
            parts.append("%s agg %s tok/s, decode_share %s, co-batched step %s ms (mean of the cells' p50s)" % (
                NAME[a], f1(statistics.mean([r["agg_dec_tps"] for r in v if r.get("agg_dec_tps") is not None]) if any(r.get("agg_dec_tps") is not None for r in v) else None),
                f1(statistics.mean([r["decode_share"] for r in v if r.get("decode_share") is not None]) if any(r.get("decode_share") is not None for r in v) else None, 3),
                f1(statistics.mean([((r.get("steps") or {}).get("full") or {}).get("ms_p50") for r in v
                                    if ((r.get("steps") or {}).get("full") or {}).get("ms_p50") is not None]) if any(((r.get("steps") or {}).get("full") or {}).get("ms_p50") is not None for r in v) else None)))
        out.append("C8 B vs SP-best: " + " | ".join(parts))
    # the chain
    ch = rows(os.path.join(o, "chain.jsonl"))
    if ch:
        by = {}
        for r in ch: by.setdefault((r.get("chain"), r.get("arm")), []).append(r)
        for (cn, a), rs in sorted(by.items()):
            out.append("CHAIN %s (%s): %s" % (cn, NAME.get(a, a), " | ".join(
                "t%s prompt %s cached %s tail %s TTFT %s s, staged %s GB, sp full %s / stream %s" % (
                    r.get("turn"), r.get("prompt_tokens"), r.get("cached"),
                    (r.get("prompt_tokens") or 0) - (r.get("cached") or 0), r.get("ttft"), r.get("staged_gb"), r.get("sp_full"),
                    r.get("sp_stream")) for r in sorted(rs, key=lambda x: x.get("turn") or 0))))
        for t in sorted({r.get("turn") for r in ch}):
            if t == 1: continue
            vb = [r.get("ttft") for r in ch if r.get("turn") == t and r.get("arm") == "B"]
            vx = [r.get("ttft") for r in ch if r.get("turn") == t and r.get("arm") == "X"]
            mb, mx = med(vb), med(vx)
            out.append("CHAIN turn %s TTFT: B %s s, SP-best %s s (%s)%s" % (t, f1(mb, 3), f1(mx, 3), pc(mx, mb),
                       "" if mb is None or mx is None else " = %+.0f ms" % (1000 * (mx - mb))))
    # the cold 64k
    dp = rows(os.path.join(o, "deep.jsonl"))
    if dp:
        by = {}
        for r in dp: by.setdefault(r.get("arm"), []).append(r)
        b = med([r.get("ttft") for r in by.get("B", [])])
        bc = med([r.get("nondecode_ms") for r in by.get("B", [])])
        parts = []
        for a in ("B", "S", "P"):
            rs = by.get(a, [])
            if not rs: continue
            t_, c_ = med([r.get("ttft") for r in rs]), med([r.get("nondecode_ms") for r in rs])
            parts.append("%s TTFT %s s (%s) chunk %s ms (%s), %s chunks, staged %s GB, guard %s" % (
                NAME[a], f1(t_, 2), "-" if a == "B" else pc(t_, b), f1(c_), "-" if a == "B" else pc(c_, bc),
                med([r.get("nondecode_steps") for r in rs]), f1(med([r.get("staged_gb") for r in rs]), 1),
                "/".join(str(r.get("guard", "?"))[:4] for r in rs)))
        out.append("COLD 64k (deepq, cold after a reset, 512 greedy tokens): " + " | ".join(parts))
    try:
        ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError:
        ident = []
    bad = [l for l in ident if not ("IDENTICAL" in l and "differ" not in l)]
    try:
        guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
    except OSError:
        guards = []
    gb = [g for g in guards if " FAIL" in g or "VIRHE" in g]
    out.append("IDENTITY (strict, every SP answer vs B): %s || COUNTER GUARD: %s" % (
        ("IDENTICAL at every compare (%d)" % len(ident)) if ident and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:900] if ident else "none"),
        ("OK at every reading (%d)" % len(guards)) if guards and not gb else (("FAILED: " + "; ".join(gb))[:900] if guards else "none")))
    return out


# ------------------------------------------------------------------ selftest
def selftest():
    import tempfile
    ok, why = True, []
    if need(4161, 0, [2300, 2400]) != "pskl,spstream/spfull,sppick": ok = False; why.append("need S %s" % need(4161, 0, [2300, 2400]))
    if need(0, 0, [2300, 3900]) != "pskl,spfull/spstream,sppick": ok = False; why.append("need B")
    if need(0, 2, [3000]) != "pskl,spfull,sppick/spstream": ok = False; why.append("need P")
    if need(4161, 0, [1500, 1600]) != "pskl/spstream,spfull,sppick": ok = False; why.append("need span")
    if need(3450, 1, [3000, 3900]) != "pskl,spstream,spfull,sppick/": ok = False; why.append("need X %s" % need(3450, 1, [3000, 3900]))
    if need(4161, 0, [5000]) != "pskl,spstream/spfull,sppick": ok = False; why.append("need 5k S %s" % need(4161, 0, [5000]))
    if need(4100, 1, [5000]) != "pskl/": ok = False; why.append("need 5k unsure %s" % need(4100, 1, [5000]))
    if need(0, 1, [5000]) != "pskl,spfull,sppick/spstream": ok = False; why.append("need 5k P %s" % need(0, 1, [5000]))
    with tempfile.TemporaryDirectory() as d:
        def w(n, x):
            p = os.path.join(d, n)
            with open(p, "w") as f:
                if isinstance(x, list): f.write("".join(json.dumps(y) + "\n" for y in x))
                else: json.dump(x, f)
            return p
        def snap(k, ns, sb, st):
            return {"k": k, "s": {"steps_prefill": k, "steps_mixed": 0, "step_ns": ns, "step_ns_decode": 0,
                                  "experts": {"staged_bytes": sb, "ranks": [{"rank": r, "stream_bytes": st} for r in range(4)]},
                                  "link": {"staged_bytes": 4 * sb}}}
        rq = w("r.jsonl", [dict(j=0, prompt=2300, cached=0, t_resp=1.1, ttft=1.0), dict(j=1, prompt=2310, cached=0, t_resp=1.2, ttft=1.1)])
        stt = w("s.jsonl", [snap(0, 0, 0, 0), snap(1, 1050 * 10**6, 12 * 10**9, 0), snap(2, 2110 * 10**6, 24 * 10**9, 10**9)])
        fa = w("f0.json", {"sp": {"ranks": [{"rank": r, "full_steps": 0, "full_bytes": 0} for r in range(4)]}})
        fb = w("f1.json", {"sp": {"ranks": [{"rank": r, "full_steps": 2, "full_bytes": 24 * 10**9} for r in range(4)]}})
        c = cell(rq, stt, fa, fb, "B", "2.3k", "0,0")
        if not (c["n"] == 2 and c["per"][0]["pf_ms"] == 1050.0 and c["per"][1]["pf_ms"] == 1060.0 and c["per"][0]["staged_gb"] == 48.0
                and c["per"][1]["stream_gb"] == [1.0] * 4 and c["sp"][3]["full_steps"] == 2 and not c["guard"]):
            ok = False; why.append("cell %s" % json.dumps(c)[:300])
        rq2 = w("r2.jsonl", [dict(j=0, prompt=2300, cached=64, t_resp=1.1)])
        if not cell(rq2, stt, fa, fb, "B", "2.3k", "0,0")["guard"]: ok = False; why.append("freshness")
        cells = []
        for sz, base in (("2.3k", 1000), ("3k", 1200), ("3.9k", 1500)):
            for a, f in (("B", 1.0), ("S", 0.8 if sz != "3.9k" else 0.95), ("1", 0.85 if sz != "3.9k" else 0.9), ("2", 0.9)):
                pt = {"2.3k": 2300, "3k": 3000, "3.9k": 3900}[sz]
                cells.append("%s:%s:%s" % (sz, a, w("c-%s-%s.json" % (sz, a), dict(prompts=[pt] * 4, per=[dict(pf_ms=base * f, staged_gb=1, stream_gb=[1, 1, 1, 1])] * 4, guard=[]))))
        lines, N, K, kp, _ = pick(cells, ["2.3k", "3k", "3.9k"])
        if not (N == 3450 and K == 1 and kp == 1 and "fastest SP form SP-S" in lines[0] and "SP-P1" in lines[2]): ok = False; why.append("pick %s %s %s %s" % (N, K, kp, lines))
        cells2 = [c.replace(":S:", ":S:") for c in cells if ":1:" not in c and ":2:" not in c]
        _, N2, K2, _, _ = pick(cells2, ["2.3k", "3k", "3.9k"])
        if not (N2 == 4161 and K2 == 0): ok = False; why.append("pick all-stream %s %s" % (N2, K2))
        lg = os.path.join(d, "e.log.gz")
        with gzip.open(lg, "wt") as f:
            ls = [("2026-11-12T08:00:00.000000000Z", "decode", 8), ("2026-11-12T08:00:00.050000000Z", "MIXED", 2400),
                  ("2026-11-12T08:00:01.150000000Z", "decode", 8), ("2026-11-12T08:00:01.200000000Z", "MIXED", 1200),
                  ("2026-11-12T08:00:01.800000000Z", "decode", 8), ("2026-11-12T08:00:09.000000000Z", "decode", 8)]
            for t, ph, n in ls: f.write("%s I[0] step 1 %s n_seq=8 n_tok=%d  [0] slot=0 q=1 ctx=9\n" % (t, ph, n))
        t0 = calendar.timegm(time.strptime("2026-11-12T08:00:00", "%Y-%m-%dT%H:%M:%S"))
        st = steps(lg, t0, t0 + 100)
        if not (st["full"]["n"] == 1 and st["full"]["ms_p50"] == 1100.0 and st["span"]["ms_p50"] == 600.0 and st["decode"]["n"] == 2 and st["steps"] == 4):
            ok = False; why.append("steps %s" % st)
    print("SELFTEST OK (need rules: span / B / S / P / composite / 5k; cell readings + freshness; the screen's pick + composite; the step log's classes)" if ok
          else "SELFTEST FAIL " + "; ".join(why))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cell", "need", "pick", "steps", "summary", "selftest"))
    ap.add_argument("--log"); ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--reqs"); ap.add_argument("--stats"); ap.add_argument("--fn0"); ap.add_argument("--fn1")
    ap.add_argument("--arm", default=""); ap.add_argument("--size", default=""); ap.add_argument("--sw", default="0,0")
    ap.add_argument("--json"); ap.add_argument("--pts", default="")
    ap.add_argument("--cell", action="append", default=[]); ap.add_argument("--sizes", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode == "selftest": return selftest()
    if a.mode == "need":
        n, k = (int(x) for x in a.sw.split(","))
        print(need(n, k, [int(x) for x in a.pts.split(",") if x.strip().isdigit()]))
        return 0
    if a.mode == "cell":
        c = cell(a.reqs, a.stats, a.fn0, a.fn1, a.arm, a.size, a.sw)
        if a.json:
            with open(a.json, "w") as f: json.dump(c, f)
        print("CELL " + json.dumps({k: v for k, v in c.items() if k != "per"}))
        return 0
    if a.mode == "pick":
        lines, N, K, kp, meds = pick(a.cell, [x for x in a.sizes.split(",") if x])
        if a.json:
            with open(a.json, "w") as f: json.dump(dict(N=N, K=K, KP=kp, meds={s: {k: {kk: vv for kk, vv in v.items() if kk != "ci"} for k, v in m.items()} for s, m in meds.items()}), f)
        for l in lines: print("SCREEN " + l)
        print("PICK %d %d %d" % (N, K, kp))
        return 0
    if a.mode == "summary":
        for l in summary(a.out): print(l)
        return 0
    if a.mode == "steps":
        print("STEPS " + json.dumps(steps(a.log, a.t0, a.t1)))
        return 0


if __name__ == "__main__":
    sys.exit(main())
