#!/usr/bin/env python3
"""u1sum.py (12.11.2026, queue-ar/dsmain/1u, the int11 builder) -- the readings of the int11 1u screen
(57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u.sh, body dsm-1u.sh). Standard library only.

  u1sum.py steps ENGINE_LOG(.gz) BLOCKS.jsonl --out STEPS.json
      `docker logs -t` step lines (RADIANCE_LOG_STEPS=1: "<utc>Z I step N decode n_seq=1 n_tok=R ..."): per block window
      [utc0, utc1] the 1u decode steps (n_seq 1) followed by step N + 1 inside the window; a step's ms = the next step line's
      time - its own (<= 400 ms); by verify rows R = n_tok: 1 / 2 / 3 / 4+ -> n, p50, mean; all steps' mean; rows a step.
  u1sum.py stamps STAMPS.csv(.gz) BLOCKS.jsonl --out STAMPS.json [--steps STEPS.json]
      RADIANCE_AQL_STAMP's CSV (card 0; every Nth pass; host CLOCK_MONOTONIC ns): the passes whose first dispatch starts in a
      block's window [mono0, mono1], per pass kind (records): wall (first start .. last end), busy (union), idle (wall - busy),
      idle-before (the card idle right before a dispatch, nothing of the pass running: a dependent launch's gap), dispatches,
      ordered, the chain's gaps (k1prof.py's walk), dispatches off the pass's main queue (lane 1), and per kernel family (the
      mangled name's base identifier) device us and calls a pass. The pass ids give passes a step (with the step count).
  u1sum.py summary BLOCKS.jsonl --steps STEPS.json --stamps STAMPS.json [--ref [code:]NAME=TPS ..]
      per lever (MX / MX12 / GQ / E2 / F4: on vs the O blocks of its group, paired in plan order; C1T: C50 / C40 / C30):
      the essay read's and (when present: "jc") the code read's tok/s and tokens a step, acceptance, ms a step at 1 / 2 / 3 / 4+ rows, the stamped trunk pass and the lever's target kernel
      families (us a pass); the 1u read of the O blocks (1U-HG: idle share, idle-before share, chain gaps, step period vs pass
      wall = the host round trip estimate; 1U-DQ: the passes' all-reduce and MoE families, card 0 only). Lines "U1 ..." and one
      "SUMMARY ..." line. Numbers only.
  u1sum.py selftest
      synthetic step log, stamps and blocks through all three -> "SELFTEST OK" or the failure."""
import bisect, csv, gzip, io, json, os, re, statistics, sys, tempfile
from datetime import datetime, timezone
from collections import defaultdict

ROWCLS = ["1", "2", "3", "4+"]
RX_STEP = re.compile(r"^(\S+?)Z? I step (\d+) (\w+) n_seq=(\d+) n_tok=(\d+)")


def rd_blocks(p):
    out = []
    for l in open(p):
        l = l.strip()
        if not l: continue
        try: out.append(json.loads(l))
        except ValueError: pass
    return out


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def ts_utc(s):
    s = s.rstrip("Z")
    if "." in s:
        a, b = s.split(".", 1)
        s = a + "." + b[:6]
        fm = "%Y-%m-%dT%H:%M:%S.%f"
    else:
        fm = "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fm).replace(tzinfo=timezone.utc).timestamp()


def rcls(r):
    return str(r) if r <= 3 else "4+"


def q(v, p):
    if not v: return None
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


# ------------------------------------------------------------------ steps
def steps(a):
    logf, blkf = a[1], a[2]
    out = a[a.index("--out") + 1] if "--out" in a else None
    B = rd_blocks(blkf)
    S = []
    with opn(logf) as f:
        for l in f:
            m = RX_STEP.match(l)
            if not m: continue
            try: t = ts_utc(m.group(1))
            except ValueError: continue
            S.append((int(m.group(2)), t, m.group(3), int(m.group(4)), int(m.group(5))))
    S.sort(key=lambda x: x[1])
    ts = [x[1] for x in S]
    res = {}
    for b in B:
        k = str(b.get("i"))
        u0, u1 = b.get("utc0"), b.get("utc1")
        if u0 is None or u1 is None: res[k] = {"error": "no window"}; continue
        lo, hi = bisect.bisect_left(ts, u0), bisect.bisect_right(ts, u1)
        W = S[lo:hi]
        by = defaultdict(list); allms = []; rows = []
        for x, y in zip(W, W[1:]):
            if x[2] != "decode" or x[3] != 1 or y[0] != x[0] + 1: continue
            dt = (y[1] - x[1]) * 1000.0
            if dt <= 0 or dt > 400: continue
            by[rcls(x[4])].append(dt); allms.append(dt); rows.append(x[4])
        r = {"steps": len(allms), "steps_in_window": len(W), "mean_ms": round(statistics.mean(allms), 3) if allms else None,
             "rows_a_step": round(statistics.mean(rows), 3) if rows else None, "ms": {}, "_raw": {c: [round(v, 3) for v in by[c]] for c in ROWCLS}}
        for c in ROWCLS:
            v = by[c]
            r["ms"][c] = {"n": len(v), "p50": round(statistics.median(v), 3) if v else None,
                          "mean": round(statistics.mean(v), 3) if v else None}
        res[k] = r
    if out:
        with open(out, "w") as f: json.dump(res, f)
    for k, r in res.items():
        if "error" in r: print("U1STEPS block %s: %s" % (k, r["error"])); continue
        print("U1STEPS block %s: %d 1u decode steps, mean %s ms, %s rows a step | %s" % (k, r["steps"], r["mean_ms"], r["rows_a_step"],
              " | ".join("%s rows n %d p50 %s mean %s" % (c, r["ms"][c]["n"], r["ms"][c]["p50"], r["ms"][c]["mean"]) for c in ROWCLS)))
    return 0


# ------------------------------------------------------------------ stamps
def fam(k):
    """the base identifier of an Itanium-mangled kernel name (_ZN12_GLOBAL__N_115gg_embed_kernelILi2EE.. -> gg_embed_kernel)"""
    s = k.strip()
    if not s.startswith("_Z"): return s[:48] or "?"
    i = 2
    if s[i:i + 1] == "N": i += 1
    names = []
    while i < len(s) and s[i].isdigit():
        j = i
        while j < len(s) and s[j].isdigit(): j += 1
        n = int(s[i:j]); names.append(s[j:j + n]); i = j + n
        if i < len(s) and s[i] in "IE": break
    names = [x for x in names if not x.startswith("_GLOBAL")]
    return names[-1] if names else s[:48]


def wv_of(k):
    """mix_split_kernel<HC, R, WV, TS, FIN>'s WV from the mangled name (ILi4ELi1ELi8ELi8ELb0EE -> 8)"""
    m = re.search(r"mix_split_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", k)
    return int(m.group(3)) if m else None


def pass_stats(rows):
    """rows: (start, end, ordered, kernel, queue) of one pass -> numbers (k1prof.py's walk, per family)"""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    n = len(rows)
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    ev = []
    for i, r in enumerate(rows):
        ev.append((r[0], 1, i)); ev.append((r[1], -1, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    active, busy, last = set(), 0, None
    for t, kind, i in ev:
        if last is not None and active: busy += t - last
        if kind == 1: active.add(i)
        else: active.discard(i)
        last = t
    idle_before, run_end = [0] * n, rows[0][0]
    for i, r in enumerate(rows):
        if i and r[0] > run_end: idle_before[i] = r[0] - run_end
        run_end = max(run_end, r[1])
    by_end = sorted(range(n), key=lambda i: rows[i][1])
    ends = [rows[i][1] for i in by_end]
    cur = max(range(n), key=lambda i: rows[i][1])
    chain, gaps = set([cur]), 0
    while True:
        j = bisect.bisect_right(ends, rows[cur][0]) - 1
        if j < 0: break
        p = by_end[j]
        if p in chain: break
        gaps += max(0, rows[cur][0] - rows[p][1])
        chain.add(p); cur = p
    qc = defaultdict(int)
    for r in rows: qc[r[4]] += 1
    mainq = max(qc, key=qc.get)
    per = defaultdict(lambda: [0, 0.0, 0.0])   # family -> [calls, device us, idle-before us]
    wv = defaultdict(int)
    for i, r in enumerate(rows):
        f = fam(r[3]); p = per[f]
        p[0] += 1; p[1] += (r[1] - r[0]) / 1e3; p[2] += idle_before[i] / 1e3
        if f == "mix_split_kernel":
            w = wv_of(r[3])
            if w: wv[w] += 1
    return {"wall": (t1 - t0) / 1e3, "busy": busy / 1e3, "idle": (t1 - t0 - busy) / 1e3, "idle_before": sum(idle_before) / 1e3,
            "disp": n, "ordered": sum(int(r[2]) for r in rows), "chain_gaps": gaps / 1e3, "lane1": n - qc[mainq],
            "per": {k: list(v) for k, v in per.items()}, "mix_wv": dict(wv)}


def agg(st):
    """per-pass stats of one kind -> medians / means"""
    if not st: return {}
    med = lambda k: round(statistics.median(x[k] for x in st), 2)
    mean = lambda k: round(statistics.mean(x[k] for x in st), 2)
    fams = defaultdict(lambda: [0.0, 0.0, 0.0])
    for x in st:
        for f, v in x["per"].items():
            a = fams[f]; a[0] += v[0]; a[1] += v[1]; a[2] += v[2]
    n = len(st)
    wv = defaultdict(int)
    for x in st:
        for w, c in x["mix_wv"].items(): wv[str(w)] += c
    return {"passes": n, "wall_us": med("wall"), "busy_us": med("busy"), "idle_us": med("idle"), "idle_before_us": mean("idle_before"),
            "dispatches": med("disp"), "ordered": med("ordered"), "chain_gaps_us": med("chain_gaps"), "lane1": mean("lane1"),
            "fam": {f: {"calls": round(v[0] / n, 2), "us": round(v[1] / n, 2), "idle_before_us": round(v[2] / n, 2)} for f, v in fams.items()},
            "mix_wv": dict(wv)}


def stamps(a):
    path, blkf = a[1], a[2]
    out = a[a.index("--out") + 1] if "--out" in a else None
    stp = {}
    if "--steps" in a:
        try: stp = json.load(open(a[a.index("--steps") + 1]))
        except (OSError, ValueError): stp = {}
    B = [b for b in rd_blocks(blkf) if b.get("mono0") is not None and b.get("mono1") is not None]
    W = sorted((int(b["mono0"]), int(b["mono1"]), str(b["i"])) for b in B)
    starts = [w[0] for w in W]
    res = {str(b["i"]): {"kinds": {}, "pass_ids": None} for b in B}
    if not os.path.exists(path):
        print("U1STAMP no stamp file %s" % path)
        if out:
            with open(out, "w") as f: json.dump({"error": "no stamp file"}, f)
        return 0
    open_p = {}    # pass -> [block, records, rows]
    done = defaultdict(lambda: defaultdict(list))
    pids = defaultdict(list)
    first_of = {}
    nrows = 0

    def fin(p):
        b, rec, rows = open_p.pop(p)
        if b is not None and rows: done[b][rec].append(pass_stats(rows))
    with opn(path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        if not hdr: return 0
        ix = {h: i for i, h in enumerate(hdr)}
        for r in rdr:
            try:
                p = int(r[ix["pass"]]); s = int(r[ix["start_ns"]]); e = int(r[ix["end_ns"]])
            except (ValueError, IndexError, KeyError):
                continue
            nrows += 1
            if p not in open_p:
                # the pass's block: by its first dispatch's start
                j = bisect.bisect_right(starts, s) - 1
                blk = W[j][2] if j >= 0 and W[j][0] <= s <= W[j][1] else None
                open_p[p] = [blk, int(r[ix["records"]] or 0), []]
                if blk is not None: pids[blk].append(p)
                while len(open_p) > 6: fin(min(open_p))
            ent = open_p[p]
            if ent[0] is not None:
                ent[2].append((s, e, r[ix["ordered"]] == "1", r[ix["kernel"]], r[ix["queue"]]))
    for p in list(open_p): fin(p)
    for k in res:
        kinds = done.get(k, {})
        res[k]["kinds"] = {str(rec): agg(st) for rec, st in kinds.items()}
        ps = sorted(pids.get(k, []))
        if ps:
            sw = (stp.get(k) or {}).get("steps_in_window")
            res[k]["pass_ids"] = {"first": ps[0], "last": ps[-1], "stamped": len(ps), "every": (ps[-1] - ps[0]) / (len(ps) - 1) if len(ps) > 1 else None,
                                  "passes_a_step": round((ps[-1] - ps[0]) / sw, 3) if sw else None}
        if kinds:
            res[k]["trunk"] = str(max(kinds, key=lambda rec: (rec, len(kinds[rec]))))
    if out:
        with open(out, "w") as f: json.dump(res, f)
    for k, r in res.items():
        if not r["kinds"]: print("U1STAMP block %s: no stamped pass in the window" % k); continue
        t = r["kinds"].get(r.get("trunk"), {})
        print("U1STAMP block %s: kinds %s; trunk (%s records) %d passes: wall %s us, busy %s, idle %s, idle-before %s, %s dispatches, lane-1 %s" % (
            k, {kk: v.get("passes") for kk, v in r["kinds"].items()}, r.get("trunk"), t.get("passes", 0), t.get("wall_us"), t.get("busy_us"),
            t.get("idle_us"), t.get("idle_before_us"), t.get("dispatches"), t.get("lane1")))
    print("U1STAMP rows read %d" % nrows)
    return 0


# ------------------------------------------------------------------ summary
TARGET = {   # the families a lever's kernel change shows in (card 0)
    "MX": ["mix_split_kernel", "mix_finish_kernel"], "MX12": ["mix_split_kernel", "mix_finish_kernel"],
    "GQ": ["gg_mr_kernel", "gg_mr_pair_kernel"], "E2": ["gg_mr_kernel"], "F4": ["gg_mr_kernel", "swiglu_clamped_kernel"],
}


def pct(a, b):
    return 100.0 * (a / b - 1) if a is not None and b else None


def fmtp(x):
    return "%+.2f %%" % x if x is not None else "-"


def stamped_of(SM, ids):
    """pool the trunk kind's per-block aggregates of `ids` (weighted by passes)"""
    tot = 0; acc = defaultdict(float); fams = defaultdict(lambda: [0.0, 0.0]); wv = defaultdict(int)
    for i in ids:
        r = SM.get(str(i)) or {}
        t = (r.get("kinds") or {}).get(r.get("trunk") or "", {})
        n = t.get("passes") or 0
        if not n: continue
        tot += n
        for k in ("wall_us", "busy_us", "idle_us", "idle_before_us", "dispatches", "chain_gaps_us", "lane1"):
            acc[k] += (t.get(k) or 0) * n
        for f, v in (t.get("fam") or {}).items():
            fams[f][0] += v["us"] * n; fams[f][1] += v["calls"] * n
        for w, c in (t.get("mix_wv") or {}).items(): wv[w] += c
    if not tot: return None
    o = {k: v / tot for k, v in acc.items()}
    o["passes"] = tot
    o["fam"] = {f: (v[0] / tot, v[1] / tot) for f, v in fams.items()}
    o["mix_wv"] = dict(wv)
    return o


def steps_pool(ST, ids):
    by = defaultdict(list); allv = []
    for i in ids:
        r = ST.get(str(i)) or {}
        for c, v in (r.get("_raw") or {}).items():
            by[c] += v; allv += v
    return {c: (len(by[c]), statistics.median(by[c]) if by[c] else None, statistics.mean(by[c]) if by[c] else None) for c in ROWCLS}, (statistics.mean(allv) if allv else None)


def summary(a):
    B = rd_blocks(a[1])
    ST = json.load(open(a[a.index("--steps") + 1])) if "--steps" in a and os.path.exists(a[a.index("--steps") + 1]) else {}
    SM = json.load(open(a[a.index("--stamps") + 1])) if "--stamps" in a and os.path.exists(a[a.index("--stamps") + 1]) else {}
    if "error" in SM: SM = {}
    groups = defaultdict(list)
    for b in B: groups[b.get("lever")].append(b)
    out, sm = [], []
    j = lambda b, k: (b.get("j") or {}).get(k)
    jc = lambda b, k: (b.get("jc") or {}).get(k)   # the code read of the block (stepprobe essay,code), when present

    def pread(on, off, seq, g):
        """one prompt's on vs off: means, delta, pairs (each on block against the O block before it), O-O spread"""
        a1 = [g(b, "decode_tps") for b in on if g(b, "decode_tps")]; a0 = [g(b, "decode_tps") for b in off if g(b, "decode_tps")]
        if not a1 or not a0: return None
        pr, lo = [], None
        for b in seq:
            if b["arm"] == "O": lo = b
            elif lo is not None and g(b, "decode_tps") and g(lo, "decode_tps"): pr.append(pct(g(b, "decode_tps"), g(lo, "decode_tps"))); lo = None
        m1, m0 = statistics.mean(a1), statistics.mean(a0)
        t1 = statistics.mean(g(b, "tokens_per_step") or 0 for b in on); t0 = statistics.mean(g(b, "tokens_per_step") or 0 for b in off)
        sp = 100.0 * (max(a0) - min(a0)) / m0 if len(a0) > 1 else None
        return m0, m1, pct(m1, m0), pr, sp, t0, t1
    for lever, bl in groups.items():
        bl = [b for b in bl if b.get("j")]
        if not bl: out.append("U1 %s: no block with a reading" % lever); continue
        if lever == "C1T":
            arms = defaultdict(list)
            for b in bl: arms[b["arm"]].append(b)
            ref = arms.get("C50") or []
            rt = statistics.mean(j(b, "decode_tps") for b in ref) if ref else None
            parts = []
            for arm in sorted(arms, key=lambda x: -int(x[1:]) if x[1:].isdigit() else 0):
                xs = arms[arm]
                tps = statistics.mean(j(b, "decode_tps") for b in xs)
                tok = statistics.mean(j(b, "tokens_per_step") for b in xs)
                acc = statistics.mean((j(b, "acceptance") or 0) for b in xs)
                rows, mean_ms = steps_pool(ST, [b["i"] for b in xs])
                n_all = sum(v[0] for v in rows.values()) or 1
                cx = [jc(b, "decode_tps") for b in xs if jc(b, "decode_tps")]
                cr = [jc(b, "decode_tps") for b in ref if jc(b, "decode_tps")]
                cpart = (", code %.2f tok/s (%s vs C50)" % (statistics.mean(cx), fmtp(pct(statistics.mean(cx), statistics.mean(cr)) if cr else None))) if cx else ""
                parts.append("%s: essay %.2f tok/s (%s vs C50; blocks %s), %.3f tok/step, acceptance %.3f%s, step %s ms, rows mix %s" % (
                    arm, tps, fmtp(pct(tps, rt)), "/".join("%.2f" % j(b, "decode_tps") for b in xs), tok, acc, cpart,
                    "%.2f" % mean_ms if mean_ms else "-", " ".join("%s:%.0f%%" % (c, 100.0 * rows[c][0] / n_all) for c in ROWCLS)))
                if arm != "C50" and rt: sm.append("C1T %s %s" % (arm, fmtp(pct(tps, rt))))
            out.append("U1 C1T (dspark_conf threshold; dspark_conf_rows as the base): " + " | ".join(parts))
            continue
        on = [b for b in bl if b["arm"] != "O"]; off = [b for b in bl if b["arm"] == "O"]
        if not on or not off: out.append("U1 %s: on %d / off %d blocks -- not compared" % (lever, len(on), len(off))); continue
        seq = sorted(bl, key=lambda b: b["i"])
        pe = pread(on, off, seq, j); pc = pread(on, off, seq, jc)
        toff, ton, d, pairs, spread, tokoff, tokon = pe
        codel = ""
        if pc:
            codel = " | code %.2f -> %.2f tok/s (%s; pairs %s; O-O spread %s), tok/step %.3f -> %.3f" % (
                pc[0], pc[1], fmtp(pc[2]), "/".join(fmtp(x) for x in pc[3]) or "-", "%.2f %%" % pc[4] if pc[4] is not None else "-", pc[5], pc[6])
        rows_on, ms_on = steps_pool(ST, [b["i"] for b in on]); rows_off, ms_off = steps_pool(ST, [b["i"] for b in off])
        rr = []
        for c in ROWCLS:
            a1, a0 = rows_on[c], rows_off[c]
            if a1[0] and a0[0]: rr.append("%s rows %.2f -> %.2f ms p50 (%+.2f; n %d/%d)" % (c, a0[1], a1[1], a1[1] - a0[1], a0[0], a1[0]))
        s1, s0 = stamped_of(SM, [b["i"] for b in on]), stamped_of(SM, [b["i"] for b in off])
        stl = "stamped: n/a"
        if s1 and s0:
            tf = TARGET.get(lever, [])
            fl = []
            for f in tf:
                x0, x1 = s0["fam"].get(f, (0.0, 0.0)), s1["fam"].get(f, (0.0, 0.0))
                fl.append("%s %.1f -> %.1f us a pass (%.1f -> %.1f calls)" % (f, x0[0], x1[0], x0[1], x1[1]))
            allf = set(s0["fam"]) | set(s1["fam"])
            dl = sorted(((s1["fam"].get(f, (0, 0))[0] - s0["fam"].get(f, (0, 0))[0], f) for f in allf), key=lambda x: -abs(x[0]))[:5]
            stl = ("stamped trunk pass (card 0; %d / %d passes): wall %.0f -> %.0f us (%+.0f), busy %.0f -> %.0f, idle-before %.0f -> %.0f, dispatches %.0f -> %.0f, lane-1 %.1f -> %.1f | target %s | largest family moves %s%s" % (
                s0["passes"], s1["passes"], s0["wall_us"], s1["wall_us"], s1["wall_us"] - s0["wall_us"], s0["busy_us"], s1["busy_us"],
                s0["idle_before_us"], s1["idle_before_us"], s0["dispatches"], s1["dispatches"], s0["lane1"], s1["lane1"], "; ".join(fl) or "-",
                ", ".join("%s %+.1f" % (f, d) for d, f in dl), (" | mix_split WV calls %s -> %s" % (s0["mix_wv"], s1["mix_wv"])) if lever.startswith("MX") else ""))
        out.append("U1 %s (on %s vs O): essay %.2f -> %.2f tok/s (%s; pairs %s; O-O spread %s), tok/step %.3f -> %.3f%s | step %s -> %s ms | %s | %s" % (
            lever, on[0]["arm"], toff, ton, fmtp(d), "/".join(fmtp(x) for x in pairs) or "-", "%.2f %%" % spread if spread is not None else "-",
            tokoff, tokon, codel, "%.2f" % ms_off if ms_off else "-", "%.2f" % ms_on if ms_on else "-", "; ".join(rr) or "rows: n/a", stl))
        sm.append("%s essay %s (pairs %s)%s" % (lever, fmtp(d), "/".join(fmtp(x) for x in pairs), (" code %s" % fmtp(pc[2])) if pc else ""))
    # the new base: every O block (all new levers off) pooled, against the references the caller names (--ref NAME=TPS)
    ob = [b for b in B if b.get("arm") == "O" and b.get("j")]
    if ob:
        tps = [j(b, "decode_tps") for b in ob]
        rows0, ms0b = steps_pool(ST, [b["i"] for b in ob])
        ctps = [jc(b, "decode_tps") for b in ob if jc(b, "decode_tps")]
        refs = []
        for k, x in enumerate(a):
            if x == "--ref" and k + 1 < len(a) and "=" in a[k + 1]:
                nm, v = a[k + 1].split("=", 1)
                use = ctps if nm.startswith("code:") else tps
                if not use: continue
                try: refs.append("%s %.2f (%s)" % (nm, float(v), fmtp(pct(statistics.mean(use), float(v)))))
                except ValueError: pass
        out.append("U1 BASE (the O blocks, every new lever off, %d blocks): essay %.2f tok/s (blocks %s; spread %s), %.3f tok/step, %.2f ms/step (stepprobe)%s, step log %s ms, rows %s | vs %s" % (
            len(ob), statistics.mean(tps), "/".join("%.2f" % v for v in tps), "%.2f %%" % (100.0 * (max(tps) - min(tps)) / statistics.mean(tps)) if len(tps) > 1 else "-",
            statistics.mean(j(b, "tokens_per_step") for b in ob), statistics.mean(j(b, "ms_per_decode_step") or 0 for b in ob),
            (" | code %.2f tok/s (blocks %s), %.3f tok/step" % (statistics.mean(ctps), "/".join("%.2f" % v for v in ctps), statistics.mean(jc(b, "tokens_per_step") or 0 for b in ob if jc(b, "decode_tps")))) if ctps else "",
            "%.2f" % ms0b if ms0b else "-", " ".join("%s:%s" % (c, "%.2f" % rows0[c][1] if rows0[c][1] else "-") for c in ROWCLS), "; ".join(refs) or "-"))
    # the 1u read (1U-HG + 1U-DQ) from the O blocks
    oids = [b["i"] for b in B if b.get("arm") == "O" and b.get("j")]
    hg = "U1 HG/DQ (the O blocks, card 0 stamps): n/a (no stamps)"
    if SM and oids:
        s0 = stamped_of(SM, oids)
        rows0, ms0 = steps_pool(ST, oids)
        pas = [((SM.get(str(i)) or {}).get("pass_ids") or {}).get("passes_a_step") for i in oids]
        pas = [x for x in pas if x]
        kinds = defaultdict(lambda: [0, 0.0])
        for i in oids:
            for rec, v in ((SM.get(str(i)) or {}).get("kinds") or {}).items():
                kinds[rec][0] += v.get("passes", 0); kinds[rec][1] += (v.get("wall_us") or 0) * v.get("passes", 0)
        if s0:
            nk = sum(v[0] for v in kinds.values()) or 1
            mean_wall = sum(v[1] for v in kinds.values()) / nk
            ppstep = statistics.mean(pas) if pas else None
            dev_step = mean_wall * ppstep if ppstep else None
            host = 1000.0 * ms0 - dev_step if (ms0 and dev_step) else None
            ar = sorted(((v[0], f, v[1]) for f, v in s0["fam"].items() if "_ar_" in f or f.startswith("r4d_ar") or "gather" in f), reverse=True)[:6]
            moe = sorted(((v[0], f, v[1]) for f, v in s0["fam"].items() if any(x in f for x in ("exl3", "moe", "mxfp4", "mx_mr", "swiglu"))), reverse=True)[:6]
            other = []
            for i in oids:
                r = SM.get(str(i)) or {}
                for rec, v in (r.get("kinds") or {}).items():
                    if rec == r.get("trunk"): continue
                    other.append((rec, v))
            okinds = defaultdict(list)
            for rec, v in other: okinds[rec].append(v)
            ol = []
            for rec, vs in okinds.items():
                n = sum(v.get("passes", 0) for v in vs) or 1
                fams = defaultdict(float)
                for v in vs:
                    for f, x in (v.get("fam") or {}).items(): fams[f] += x["us"] * v.get("passes", 0)
                top = sorted(((u / n, f) for f, u in fams.items()), reverse=True)[:5]
                ol.append("kind %s records: %d passes, wall %.0f us, top %s" % (rec, n, statistics.mean(v.get("wall_us") or 0 for v in vs),
                          ", ".join("%s %.0f" % (f, u) for u, f in top)))
            hg = ("U1 HG (the O blocks, %d trunk passes stamped, card 0): trunk wall %.0f us, busy %.0f, idle %.0f (%.1f %% of the wall), idle-before %.0f us (%.1f %%), chain gaps %.0f us, %.0f dispatches | step period %s ms (step log) vs %s passes a step x mean pass wall %.0f us -> host round trip estimate %s us a step | DQ (card 0 only; per-rank needs a profiled start, not in this one): all-reduce families %s; MoE families %s%s" % (
                s0["passes"], s0["wall_us"], s0["busy_us"], s0["wall_us"] - s0["busy_us"], 100.0 * (s0["wall_us"] - s0["busy_us"]) / s0["wall_us"] if s0["wall_us"] else 0,
                s0["idle_before_us"], 100.0 * s0["idle_before_us"] / s0["wall_us"] if s0["wall_us"] else 0, s0["chain_gaps_us"], s0["dispatches"],
                "%.2f" % ms0 if ms0 else "-", "%.2f" % ppstep if ppstep else "-", mean_wall, "%.0f" % host if host is not None else "-",
                "; ".join("%s %.0f us a pass (%.1f calls, %.1f us a call)" % (f, u, c, u / c if c else 0) for u, f, c in ar) or "-",
                "; ".join("%s %.0f us a pass (%.1f calls)" % (f, u, c) for u, f, c in moe) or "-",
                (" | other pass kinds: " + " | ".join(ol)) if ol else ""))
    out.append(hg)
    for l in out: print(l)
    print("SUMMARY 1u int11: %s" % ("; ".join(sm) or "-"))
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    d = tempfile.mkdtemp(prefix="u1sum-")
    try:
        t0 = 1794487000.0
        blocks = [{"i": 0, "lever": "MX", "arm": "O", "utc0": t0, "utc1": t0 + 10, "mono0": 1000, "mono1": 10 ** 10,
                   "j": {"decode_tps": 75.0, "tokens_per_step": 1.56, "acceptance": 0.7}, "jc": {"decode_tps": 120.0, "tokens_per_step": 2.5}},
                  {"i": 1, "lever": "MX", "arm": "MX", "utc0": t0 + 20, "utc1": t0 + 30, "mono0": 2 * 10 ** 10, "mono1": 3 * 10 ** 10,
                   "j": {"decode_tps": 76.5, "tokens_per_step": 1.56, "acceptance": 0.7}, "jc": {"decode_tps": 121.2, "tokens_per_step": 2.5}},
                  {"i": 2, "lever": "C1T", "arm": "C50", "utc0": t0 + 40, "utc1": t0 + 50, "j": {"decode_tps": 75.0, "tokens_per_step": 1.5, "acceptance": 0.6}},
                  {"i": 3, "lever": "C1T", "arm": "C40", "utc0": t0 + 60, "utc1": t0 + 70, "j": {"decode_tps": 75.6, "tokens_per_step": 1.6, "acceptance": 0.6}}]
        bf = os.path.join(d, "blocks.jsonl")
        with open(bf, "w") as f:
            for b in blocks: f.write(json.dumps(b) + "\n")
        lf = os.path.join(d, "engine.log.gz")
        with gzip.open(lf, "wt") as f:
            n = 1
            for b in blocks:
                t = b["utc0"] + 0.5
                for k in range(40):
                    rows = 1 + (k % 3)
                    ts = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"
                    f.write("%s I step %d decode n_seq=1 n_tok=%d  [0] slot=0 q=%d ctx=%d\n" % (ts, n, rows, rows, 100 + n))
                    t += (17.0 + 3.0 * (rows - 1) + (0.5 if b["arm"] == "MX" and rows == 1 else 0)) / 1000.0
                    n += 1
        sj = os.path.join(d, "steps.json")
        sink = io.StringIO(); so = sys.stdout; sys.stdout = sink
        try: steps(["steps", lf, bf, "--out", sj])
        finally: sys.stdout = so
        S = json.load(open(sj))
        assert S["0"]["ms"]["1"]["n"] > 5 and abs(S["0"]["ms"]["2"]["p50"] - 20.0) < 0.01, S["0"]
        assert abs(S["1"]["ms"]["1"]["p50"] - 17.5) < 0.01, S["1"]["ms"]
        cf = os.path.join(d, "stamps.csv")
        K1 = "_ZN12_GLOBAL__N_116mix_split_kernelILi4ELi1ELi8ELi8ELb0EEEvPKtlPKflPfiilPjNS_6MixOutE"
        K2 = "_ZN12_GLOBAL__N_116mix_split_kernelILi4ELi1ELi24ELi8ELb0EEEvPKtlPKflPfiilPjNS_6MixOutE"
        K3 = "_Z32r4d_ar_oneshot_wide_exact_kernelI14__hip_bfloat16Li8ELi4EEvPK15HIP_vector_typeIiLj4EE"
        with open(cf, "w") as f:
            f.write("pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records\n")
            for p, base, k in ((17, 5000, K1), (34, 9000, K1), (51, 2 * 10 ** 10 + 100, K2), (68, 2 * 10 ** 10 + 5000, K2)):
                s = base
                for seq in range(6):
                    kk = k if seq % 2 == 0 else K3
                    dur = 10000 if kk == K3 else (14000 if k == K1 else 9000)
                    f.write("%d,%d,1034,%s,1,0,%d,%d,0,1396\n" % (p, seq, kk, s, s + dur))
                    s += dur + 2000
        mj = os.path.join(d, "stamps.json")
        sys.stdout = sink
        try: stamps(["stamps", cf, bf, "--out", mj, "--steps", sj])
        finally: sys.stdout = so
        M = json.load(open(mj))
        t = M["0"]["kinds"]["1396"]
        assert t["passes"] == 2 and t["fam"]["mix_split_kernel"]["calls"] == 3 and abs(t["fam"]["mix_split_kernel"]["us"] - 42.0) < 1e-6, t
        assert abs(t["idle_before_us"] - 10.0) < 1e-6 and t["mix_wv"] == {"8": 6}, t
        assert M["1"]["kinds"]["1396"]["mix_wv"] == {"24": 6}
        sys.stdout = sink
        try: summary(["summary", bf, "--steps", sj, "--stamps", mj, "--ref", "base1113=74.0", "--ref", "code:base1113=100.0"])
        finally: sys.stdout = so
        txt = sink.getvalue()
        assert "U1 MX (on MX vs O): essay 75.00 -> 76.50 tok/s (+2.00 %" in txt and "code 120.00 -> 121.20 tok/s (+1.00 %" in txt, txt[-3000:]
        assert "mix_split_kernel 42.0 -> 27.0 us a pass" in txt, txt[-3000:]
        assert "C40: essay 75.60 tok/s (+0.80 % vs C50" in txt, txt[-3000:]
        assert "U1 BASE (the O blocks, every new lever off, 1 blocks): essay 75.00 tok/s" in txt and "base1113 74.00 (+1.35 %)" in txt and "code:base1113 100.00 (+20.00 %)" in txt, txt[-3000:]
        assert "U1 HG (the O blocks, 2 trunk passes stamped" in txt and "SUMMARY 1u int11: MX essay +2.00 %" in txt, txt[-3000:]
        assert fam(K3) == "r4d_ar_oneshot_wide_exact_kernel" and fam(K1) == "mix_split_kernel" and wv_of(K2) == 24
        print("SELFTEST OK (steps, stamps, summary on synthetic data)")
        return 0
    except AssertionError as e:
        print("SELFTEST FAIL %s" % str(e)[:1500])
        return 1
    finally:
        for fn in os.listdir(d): os.unlink(os.path.join(d, fn))
        os.rmdir(d)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("steps", "stamps", "summary", "selftest"):
        print(__doc__); return 2
    if a[0] == "selftest": return selftest()
    return {"steps": steps, "stamps": stamps, "summary": summary}[a[0]](a)


if __name__ == "__main__":
    sys.exit(main())
