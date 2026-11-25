#!/usr/bin/env python3
"""p12win.py (12.11.2026, queue-ar/dsmain/prof12, the int11 builder) -- the int12 profile item's readings
(57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof.sh, body prof12/dsm-prof12.sh). Standard library only; nc/ds_opprof_nc.py is IMPORTED
READ ONLY (--opnc PATH: its table parser, op classes, the decode-only dump-interval mean, the step walls); not changed.

  p12win.py confirm BLOCKS.jsonl [--lever U1] [--expect-tps 90] [--expect-mb 27] [--ref-tps 82.24] [--ref-mb 136.6]
      start 1's 1u reads (blockx rows of LEVER): per read decode tok/s, tokens / ms a step, acceptance, stream / h2d MB a step,
      promotions; their mean and spread; vs the coordinator's expectation (~90 tok/s at ~27 MB a step) and vs ref12's essay
      (an essay right after a code read). One line "P12CONFIRM ...".
  p12win.py rows LOG(.gz) --win NAME:T0:T1 [...] [--every 8] [--st NAME:ST0.json:ST1.json:U0:U1 ...] [--json OUT]
      per window, the `docker logs -t` step lines (RADIANCE_LOG_STEPS=1) of its DECODE steps: the row mix by a step's rows
      (n_tok: 1-4 / 5-8 / 9-15 / 16 / > 16: steps, share, p50 / mean ms), the sequences a step, the depth-0 share (a sequence's
      verify rows q = 1: no draft verified that step; also the steps where every sequence has q = 1), single-sequence steps' ms by
      q (1 / 2 / 3 / 4+), the decode steps over 48 rows (n_tok > 48: the CAP48 count) and the largest; a step's ms = the next step line's time - its own (<= 5 s; with --every N the steps followed by a table
      dump, K % N == 0, left out). --st: the /stats deltas between two snapshots (link stream / h2d bytes, expert promotions,
      staged bytes) and per decode step over the step-log decode steps inside [U0, U1]. Lines "P12ROWS ...".
  p12win.py split LOG(.gz) --win NAME:T0:T1 [--every 8] [--idle 3] [--sw0 W0.json --sw1 W1.json] [--mbline MB.txt] [--target 200]
                  [--ref "ref12 c8:8.28:36.1:0.803:184.5"]
      the c8 window's WALL split (the NC-row method, nc/ncsteps.py's wall rule): the window [T0, T1] tiled by the step lines -- a step
      owns [its line, the next line] (clipped to the window; a stretch over --idle s is a gap): (a) decode steps: count, total ms,
      share, ms a step, rows a step, tokens a step (the MB line); (b) newcomer-prefill / MIXED steps: count, total ms, share, rows a
      step, ms a step; (c) host gaps: idle stretches + the window's edges outside any step, and (beside it) the step-log walls minus
      the engine's own step timer (/stats step_ns between the window's snapshots W0 / W1: "between the step timers"); with --every N
      (profiled) the steps followed by a table dump are their own class (d). agg tok/s, decode_share (MB line), the decode rate while
      decoding (tokens / ms a decode step) and what --target needs: decode_share >= target / rate at this step, or a rate >= target /
      share at this share; the --ref basis printed beside. Lines "P12SPLIT ...".
  p12win.py prof LOG(.gz) --opnc ds_opprof_nc.py --win NAME:T0:T1 [...] [--every 8] [--json OUT]
      PROFILED windows (--profile-ops, RADIANCE_PROFILE_EVERY N): per rank the tables at or before T0 and T1, and between them
      ONLY the decode-only dump intervals (an interval with a prefill / MIXED step left out: the c8 newcomers' prefills, the 1u
      read's own prefill), their per-step mean (ds_opprof_nc.decode_mean) -> ms a decode step by class: drafter ctx + query,
      MoE experts, moe_other, attention + indexer, mhc, collectives / gather_ar, dense + norms (dense_gemm, norm_elem,
      embed_engram, other), lane waits (stager / other), host remainder (= the decode steps' wall from the step log - device -
      lane waits); rank 0's sub-breakdowns (mhc / indexer / Engram / dspark_ctx by op) and top ops. Lines "P12PROF ...",
      "P12SUB ...", "P12TOP ...". Inside the newcomer steps SEPARATELY (an ESTIMATE, ds_opprof_nc.py --intervals' rule: the dump
      intervals with exactly one non-decode step minus their decode steps x the decode-only mean): "P12PROFNC ..." per rank. PROFILED: every number from synchronised steps -- not a speed.
  p12win.py fit PROF.json [--win NAME ...] [--json OUT]
      "ms per added row by class": per rank, a least-squares line of each class's ms a decode step (prof's decode-only mean) against
      the window's mean rows a step over the windows (1u ~1.6, c8 ~8, c4 ~16, c16 ~48 rows) -> the slope (ms per added row) and the
      intercept (ms at 0 rows), rank 0's numbers and the spread over the ranks; classes EXL3 experts, dense GEMM, collectives (AR /
      gather_ar), attention + indexer, mhc, drafter ctx+query, moe_other, norms / other, lane waits, host remainder, wall. Also inside
      each window over its decode-only dump intervals when their rows spread >= 2 rows. Lines "P12FIT ..." (one table), "P12FITW ...".
  p12win.py selftest"""
import gzip, importlib.util, json, os, re, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone

BUCKETS = ["1-4", "5-8", "9-15", "16", ">16"]
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")
STEP = re.compile(r" step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
QRE = re.compile(r"\bq=(\d+)")
GROUPS = [("drafter ctx+query", ("dspark_ctx", "dspark_query")), ("MoE experts", ("experts",)), ("moe_other", ("moe_other",)),
          ("attention + indexer", ("attention", "indexer")), ("mhc", ("mhc",)), ("collectives / gather_ar", ("collective", "gather_ar")),
          ("dense + norms", ("dense_gemm", "norm_elem", "embed_engram", "other"))]


def bucket(r):
    return "1-4" if r <= 4 else "5-8" if r <= 8 else "9-15" if r <= 15 else "16" if r == 16 else ">16"


def opn(p):   # by the gzip magic, not the name; gzip members and any plain text after them (a mock capture appends plain lines)
    import io, zlib
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


def read_steps(path):
    """[(t, K, phase, n_seq, n_tok, [q, ...])] in log order."""
    out = []
    with opn(path) as f:
        for l in f:
            m = TSRE.match(l)
            if not m or " step " not in l: continue
            s = STEP.search(l.rstrip("\n"))
            if not s: continue
            out.append((tsec(m), int(s.group(1)), s.group(2), int(s.group(3)), int(s.group(4)), [int(x) for x in QRE.findall(s.group(5))]))
    return out


def pq(v, p):
    if not v: return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))], 3)


def args_of(argv, multi=("--win", "--st")):
    a, i = {k: [] for k in multi}, 0
    pos = []
    while i < len(argv):
        k = argv[i]
        if k.startswith("--") and i + 1 < len(argv):
            if k in multi: a[k].append(argv[i + 1])
            else: a[k] = argv[i + 1]
            i += 2
        else: pos.append(k); i += 1
    return a, pos


def win_of(s):
    nm, t0, t1 = s.rsplit(":", 2)
    return nm, float(t0), float(t1)


# ---------------------------------------------------------------- confirm
def confirm(argv):
    a, pos = args_of(argv)
    lever = a.get("--lever", "U1")
    rows = []
    for l in open(pos[0]):
        l = l.strip()
        if not l: continue
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("lever") == lever and (d.get("j") or {}).get("prompt") == "essay": rows.append(d)
    if not rows:
        print("P12CONFIRM VIRHE no %s essay read in %s" % (lever, pos[0])); return 1
    J = [r["j"] for r in rows]; S = [r.get("st") or {} for r in rows]
    tps = [j.get("decode_tps") for j in J if j.get("decode_tps") is not None]
    mbs = [s.get("stream_mb_step") for s in S if s.get("stream_mb_step") is not None]
    h2d = [s.get("h2d_mb_step") for s in S if s.get("h2d_mb_step") is not None]
    pro = [s.get("promotions") for s in S if s.get("promotions") is not None]
    mean = lambda v: round(statistics.mean(v), 2) if v else None
    m_tps, m_mb = mean(tps), mean(mbs)
    m_tps = round(m_tps, 2) if m_tps is not None else None
    spread = round(100.0 * (max(tps) - min(tps)) / m_tps, 2) if len(tps) > 1 and m_tps else 0.0
    et, em = float(a.get("--expect-tps", 90)), float(a.get("--expect-mb", 27))
    rt, rm = a.get("--ref-tps"), a.get("--ref-mb")
    pct = lambda x, y: ("%+.1f %%" % (100.0 * (x - y) / y)) if x is not None and y else "-"
    line = ("P12CONFIRM 1u essay back to back (%d warm read(s) after one discarded, stepprobe essay, T 0): %s tok/s (reads %s, spread %.2f %%), "
            "%s tok / %s ms a step, acceptance %s | stream %s MB / h2d %s MB a step (reads %s), promotions a read %s | expect ~%g tok/s at ~%g MB a step: "
            "tok/s %s, stream %s" % (
                len(rows), ("%.2f" % m_tps) if m_tps is not None else "-", "/".join("%.2f" % x for x in tps), spread, mean([j.get("tokens_per_step") for j in J if j.get("tokens_per_step") is not None]),
                mean([j.get("ms_per_decode_step") for j in J if j.get("ms_per_decode_step") is not None]), mean([j.get("acceptance") for j in J if j.get("acceptance") is not None]),
                m_mb, mean(h2d), "/".join("%.1f" % x for x in mbs), "/".join(str(x) for x in pro), et, em, pct(m_tps, et),
                ("%+.1f MB (%s)" % (m_mb - em, pct(m_mb, em))) if m_mb is not None else "-"))
    if rt: line += " | ref12's essay (right after a code read) %s tok/s%s: %s" % (rt, (" / %s MB a step" % rm) if rm else "", pct(m_tps, float(rt)))
    print(line)
    return 0


# ---------------------------------------------------------------- rows
def rows(argv):
    a, pos = args_of(argv)
    every = int(a.get("--every", 0) or 0)
    steps = read_steps(pos[0])
    out, rc = {"windows": {}}, 0
    nxt = {}
    for i in range(len(steps) - 1): nxt[i] = steps[i + 1][0] - steps[i][0]
    sts = {}
    for s in a["--st"]:
        nm, s0, s1, u0, u1 = s.split(":")
        sts[nm] = (s0, s1, float(u0), float(u1))
    for w in a["--win"]:
        nm, t0, t1 = win_of(w)
        idx = [i for i, s in enumerate(steps) if t0 <= s[0] <= t1 and s[2] == "decode"]
        res = {"t0": t0, "t1": t1, "decode_steps": len(idx)}
        if not idx:
            print("P12ROWS %s: VIRHE no decode step line in [%.3f, %.3f]" % (nm, t0, t1)); rc = 1; out["windows"][nm] = res; continue
        ms = {i: 1000.0 * nxt[i] for i in idx if i in nxt and 0 <= nxt[i] <= 5.0 and not (every and steps[i][1] % every == 0)}
        bk = defaultdict(list); bn = defaultdict(int)
        for i in idx:
            b = bucket(steps[i][4]); bn[b] += 1
            if i in ms: bk[b].append(ms[i])
        qs = [q for i in idx for q in steps[i][5]]
        nseq = [steps[i][3] for i in idx]
        alld0 = sum(1 for i in idx if steps[i][5] and all(q == 1 for q in steps[i][5]))
        qh = defaultdict(int)
        for q in qs: qh[str(q) if q < 6 else "6+"] += 1
        byq = defaultdict(list)
        for i in idx:
            if steps[i][3] == 1 and steps[i][5] and i in ms: byq[str(min(steps[i][5][0], 4)) if steps[i][5][0] < 4 else "4+"].append(ms[i])
        res.update(buckets={b: dict(n=bn[b], share=round(bn[b] / len(idx), 4), p50=pq(bk[b], 0.5), mean=round(statistics.mean(bk[b]), 3) if bk[b] else None) for b in BUCKETS},
                   seqs_mean=round(statistics.mean(nseq), 2), rows_mean=round(statistics.mean(steps[i][4] for i in idx), 2),
                   depth0_share=round(sum(1 for q in qs if q == 1) / len(qs), 4) if qs else None, all_depth0_steps=round(alld0 / len(idx), 4),
                   q_hist=dict(qh), ms_p50=pq(list(ms.values()), 0.5), ms_mean=round(statistics.mean(ms.values()), 3) if ms else None, ms_n=len(ms),
                   by_q={k: dict(n=len(v), mean=round(statistics.mean(v), 2)) for k, v in sorted(byq.items())},
                   gt48=sum(1 for i in idx if steps[i][4] > 48), rows_max=max(steps[i][4] for i in idx), rows_p50=pq([steps[i][4] for i in idx], 0.5))
        if nm in sts:
            s0, s1, u0, u1 = sts[nm]
            try:
                A, B = json.load(open(s0)), json.load(open(s1))
                g = lambda d, x, y: ((d.get(x) or {}).get(y) or 0)
                nd = sum(1 for s in steps if u0 <= s[0] <= u1 and s[2] == "decode")
                sb, hb = g(B, "link", "stream_bytes") - g(A, "link", "stream_bytes"), g(B, "link", "h2d_bytes") - g(A, "link", "h2d_bytes")
                res["st"] = dict(stream_mb=round(sb / 1e6, 1), h2d_mb=round(hb / 1e6, 1), decode_steps_span=nd,
                                 stream_mb_step=round(sb / 1e6 / nd, 2) if nd else None, h2d_mb_step=round(hb / 1e6 / nd, 2) if nd else None,
                                 promotions=int(g(B, "experts", "promotions") - g(A, "experts", "promotions")),
                                 staged_mb=round((g(B, "experts", "staged_bytes") - g(A, "experts", "staged_bytes")) / 1e6, 1))
            except (OSError, ValueError) as e:
                res["st"] = {"error": str(e)[:120]}
        out["windows"][nm] = res
        b = res["buckets"]
        st = res.get("st")
        print("P12ROWS %s: %d decode steps (%d timed), %.2f sequences / %.2f rows a step, step ms p50 %s mean %s | rows %s | depth-0 share %s "
              "(sequence-steps with verify rows q = 1), steps all depth 0 %s, q %s | rows > 48: %d steps (most %s, p50 %s)%s%s" % (
                  nm, len(idx), len(ms), res["seqs_mean"], res["rows_mean"], res["ms_p50"], res["ms_mean"],
                  " | ".join("%s n %d (%.0f %%) p50 %s mean %s ms" % (k, b[k]["n"], 100 * b[k]["share"], b[k]["p50"], b[k]["mean"]) for k in BUCKETS if b[k]["n"]),
                  "%.1f %%" % (100 * res["depth0_share"]) if res["depth0_share"] is not None else "-", "%.1f %%" % (100 * res["all_depth0_steps"]),
                  json.dumps(res["q_hist"], sort_keys=True), res["gt48"], res["rows_max"], res["rows_p50"],
                  (" | single-sequence ms by q: " + ", ".join("%s: %s (n %d)" % (k, v["mean"], v["n"]) for k, v in res["by_q"].items())) if res["by_q"] else "",
                  (" | /stats over [%.1f, %.1f]: stream %s MB / h2d %s MB (%s / %s MB a decode step over %s decode steps), promotions %s, staged %s MB" % (
                      sts[nm][2], sts[nm][3], st.get("stream_mb"), st.get("h2d_mb"), st.get("stream_mb_step"), st.get("h2d_mb_step"), st.get("decode_steps_span"),
                      st.get("promotions"), st.get("staged_mb")) if st and "error" not in st else (" | /stats: %s" % st.get("error") if st else ""))))
    if a.get("--json"):
        with open(a["--json"], "w") as f: json.dump(out, f)
    return rc


# ---------------------------------------------------------------- split
def split(argv):
    a, pos = args_of(argv)
    every, idle, target = int(a.get("--every", 0) or 0), float(a.get("--idle", 3.0)), float(a.get("--target", 200))
    steps = read_steps(pos[0])
    mb = {}
    if a.get("--mbline"):
        try:
            for l in open(a["--mbline"]):
                if l.startswith("MB "): mb = json.loads(l[3:]); break
        except (OSError, ValueError): mb = {}
    sw = None
    if a.get("--sw0") and a.get("--sw1"):
        try: sw = (json.load(open(a["--sw0"])), json.load(open(a["--sw1"])))
        except (OSError, ValueError): sw = None
    rc = 0
    for w in a["--win"]:
        nm, t0, t1 = win_of(w)
        W = 1000.0 * (t1 - t0)
        acc = {"decode": [0, 0.0, 0], "newcomer": [0, 0.0, 0], "dump": [0, 0.0, 0]}   # steps, ms, rows
        gap_idle = 0.0; covered = 0.0
        for i, s in enumerate(steps[:-1]):
            b0, b1 = s[0], steps[i + 1][0]
            lo, hi = max(b0, t0), min(b1, t1)
            if hi <= lo: continue
            ms = 1000.0 * (hi - lo); covered += ms
            if b1 - b0 > idle: gap_idle += ms; continue
            c = "dump" if (every and s[1] % every == 0) else ("decode" if s[2] == "decode" else "newcomer")
            if t0 <= b0 <= t1: acc[c][0] += 1; acc[c][2] += s[4]
            acc[c][1] += ms
        edge = max(0.0, W - covered)
        if not any(acc[k][0] for k in acc):
            print("P12SPLIT %s: VIRHE no step line in [%.3f, %.3f]" % (nm, t0, t1)); rc = 1; continue
        sh = lambda v: "%.1f %%" % (100.0 * v / W) if W else "-"
        per = lambda k: (acc[k][1] / acc[k][0]) if acc[k][0] else None
        rows_ = lambda k: (acc[k][2] / acc[k][0]) if acc[k][0] else None
        tps = mb.get("tok_per_decode_step"); msd = mb.get("ms_decode_step"); agg = mb.get("agg_dec_tps"); share = mb.get("decode_share")
        rate = (1000.0 * tps / msd) if tps and msd else None
        stl = ""
        if sw:
            A, B = sw
            g = lambda d, k: d.get(k) or 0
            sn = (g(B, "step_ns") - g(A, "step_ns")) / 1e6; snd = (g(B, "step_ns_decode") - g(A, "step_ns_decode")) / 1e6
            nd = g(B, "steps_decode") - g(A, "steps_decode"); nn = (g(B, "steps_mixed") - g(A, "steps_mixed")) + (g(B, "steps_prefill") - g(A, "steps_prefill"))
            wsw = (g(B, "now_ms") - g(A, "now_ms")) or W
            walls = acc["decode"][1] + acc["newcomer"][1] + acc["dump"][1]
            stl = (" | /stats over the window's snapshots (%.0f ms): decode %d steps %.0f ms (%s, %.2f ms a step), non-decode %d steps %.0f ms (%s), outside the "
                   "engine's step timer %.0f ms (%s); step-log walls - step_ns = %.0f ms (between the step timers)" % (
                       wsw, nd, snd, "%.1f %%" % (100.0 * snd / wsw) if wsw else "-", snd / nd if nd else 0.0, nn, sn - snd,
                       "%.1f %%" % (100.0 * (sn - snd) / wsw) if wsw else "-", wsw - sn, "%.1f %%" % (100.0 * (wsw - sn) / wsw) if wsw else "-", walls - sn))
        need = ""
        if rate and share:
            need = (" | decoding at %.1f tok/s (%.2f tok / %.1f ms a decode step) x decode_share %.3f = %.1f (MB agg %s) | %g needs decode_share >= %.3f at "
                    "this step, or decode >= %.1f tok/s at this share" % (rate, tps, msd, share, rate * share, agg, target, target / rate, target / share))
        ref = ""
        if a.get("--ref"):
            try:
                lb, rt, rm, rs, ra = a["--ref"].rsplit(":", 4); rt, rm, rs, ra = float(rt), float(rm), float(rs), float(ra)
                rr = 1000.0 * rt / rm
                ref = " | basis %s: %.2f tok / %.1f ms = %.1f tok/s while decoding x decode_share %.3f = agg %.1f; %g needs decode_share >= %.3f at that step, or decode >= %.1f tok/s at that share" % (
                    lb, rt, rm, rr, rs, ra, target, target / rr, target / rs)
            except ValueError: ref = " | basis: unreadable --ref"
        print("P12SPLIT %s (window %.1f s): (a) decode %d steps %.0f ms = %s, %s ms a step, %s rows a step, %s tokens a step | (b) newcomer / MIXED %d steps %.0f ms = %s, "
              "%s rows a step, %s ms a step | (c) host gaps %.0f ms = %s (idle > %g s %.0f + window edges %.0f)%s | agg %s tok/s, decode_share %s%s%s%s" % (
                  nm, W / 1000.0, acc["decode"][0], acc["decode"][1], sh(acc["decode"][1]), ("%.2f" % per("decode")) if per("decode") else "-",
                  ("%.2f" % rows_("decode")) if rows_("decode") else "-", tps, acc["newcomer"][0], acc["newcomer"][1], sh(acc["newcomer"][1]),
                  ("%.0f" % rows_("newcomer")) if rows_("newcomer") else "-", ("%.1f" % per("newcomer")) if per("newcomer") else "-",
                  gap_idle + edge, sh(gap_idle + edge), idle, gap_idle, edge,
                  (" | (d) steps before a profile table dump %d steps %.0f ms = %s" % (acc["dump"][0], acc["dump"][1], sh(acc["dump"][1]))) if every else "",
                  agg, share, need, ref, stl))
    return rc


# ---------------------------------------------------------------- prof
def load_opnc(path):
    sp = importlib.util.spec_from_file_location("ds_opprof_nc_ro", path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def prof(argv):
    a, pos = args_of(argv)
    every = int(a.get("--every", 8) or 8)
    opnc = a.get("--opnc") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nc", "ds_opprof_nc.py")
    M = load_opnc(opnc)
    M.opn = opn   # its parse() reads through this module's opener (gzip by magic)
    steps, tables, meta = M.parse(pos[0])
    ce, first, info = M.classify(meta)
    out, rc = {"drafter": info, "windows": {}}, 0
    if not tables:
        print("P12PROF VIRHE no complete per-op table in %s (was the start --profile-ops?)" % pos[0]); return 1
    if not info.get("ctx_ok"): print("P12PROF NOTE dspark_ctx not found (no dspark.fc op after the trunk's logits op): the drafter's ms not split off")
    for w in a["--win"]:
        nm, t0, t1 = win_of(w)
        res = {"t0": t0, "t1": t1, "ranks": {}}
        notes = []
        for rk, tl in sorted(tables.items()):
            tl = [x for x in tl if x[0] is not None and x[1] is not None]
            A = [x for x in tl if x[0] <= t0]; B = [x for x in tl if x[0] <= t1]
            if not A or not B or A[-1] is B[-1]: notes.append("rank %s: no table pair around the window" % rk); continue
            dmean, ivl, ndi = M.decode_mean(tables[rk], (A[-1], B[-1]), steps, meta, ce, first)
            if dmean is None: notes.append("rank %s: no decode-only dump interval (%d intervals)" % (rk, len(ivl))); continue
            dec = [(mix, x, n, ss) for mix, x, n, ss in ivl if set(mix) == {"decode"}]
            pts = []
            for mix, x, n, ss in dec:
                wl_ = (M.walls(steps, ss, every).get("decode") or [])
                pts.append(dict(rows=round(statistics.mean([z[3] for z in ss]), 2) if ss else None, steps=n, by_class_ms=x["by_class_ms"], wait_ms=x["wait_ms"],
                                device_ms=x["device_ms"], wall_ms=round(statistics.mean(wl_), 3) if wl_ else None))
            nsteps = sum(n for _, _, n, _ in dec)
            ssd = [s for _, _, _, ss in dec for s in ss]
            wl = M.walls(steps, ssd, every).get("decode") or []
            wall = statistics.mean(wl) if wl else None
            bc = dmean["by_class_ms"]
            parts = [(g, sum(bc.get(k, 0.0) for k in ks)) for g, ks in GROUPS]
            s = dict(intervals=len(dec), intervals_all=len(ivl), steps=nsteps, rows=round(statistics.mean([x[3] for x in ssd]), 2) if ssd else None,
                     wall_ms=round(wall, 3) if wall else None, wall_n=len(wl), device_ms=dmean["device_ms"], wait_ms=dmean["wait_ms"],
                     stager_wait_ms=dmean["stager_wait_ms"], other_wait_ms=dmean["other_wait_ms"], by_class_ms=bc, groups=dict(parts),
                     host_remainder_ms=(wall - dmean["device_ms"] - dmean["wait_ms"]) if wall else None, sub_ms=dmean["sub_ms"],
                     ctx_ms=bc.get("dspark_ctx", 0.0), query_ms=bc.get("dspark_query", 0.0), ctx_calls_per_step=dmean["ctx_calls_per_step"], points=pts)
            res["ranks"][str(rk)] = s
            ones, nrows = [], []
            for mix, x, n, ss in ivl:
                if sum(v for k, v in mix.items() if k != "decode") == 1:
                    ones.append(M.combine(x, dmean, n, -mix.get("decode", 0)))
                    nrows += [z[3] for z in ss if z[2] != "decode"]
            if ones:
                mo = M.combine({}, {}, 0, 0)
                for x in ones: mo = M.combine(mo, x, 1.0, 1.0 / len(ones))
                nwl = M.walls(steps, [z for mix, x, n, ss in ivl if sum(v for k, v in mix.items() if k != "decode") == 1 for z in ss if z[2] != "decode"], every)
                nw = [v for p_, vs in nwl.items() for v in vs]
                res.setdefault("newcomer", {})[str(rk)] = dict(intervals=len(ones), rows=round(statistics.mean(nrows), 1) if nrows else None, by_class_ms=mo["by_class_ms"],
                                                               device_ms=mo["device_ms"], wait_ms=mo["wait_ms"], stager_wait_ms=mo["stager_wait_ms"],
                                                               wall_ms=round(statistics.mean(nw), 2) if nw else None, wall_n=len(nw),
                                                               groups={g: sum(mo["by_class_ms"].get(k, 0.0) for k in ks) for g, ks in GROUPS})
            if rk == min(tables):
                dsum, ns = {}, 0
                for p, q in zip([x for x in tables[rk] if x[0] is not None and A[-1][0] <= x[0] <= B[-1][0] and x[1] is not None],
                                [x for x in tables[rk] if x[0] is not None and A[-1][0] <= x[0] <= B[-1][0] and x[1] is not None][1:]):
                    ss, mix = M.step_mix(steps, p[1], q[1])
                    if set(mix) != {"decode"}: continue
                    for idx, v in M.delta(p[2], q[2]).items():
                        o = dsum.get(idx, (0, 0.0, 0.0)); dsum[idx] = (o[0] + v[0], o[1] + v[1], o[2] + v[2])
                    ns += q[1] - p[1]
                res["top0"] = M.detail(dsum, meta, ce, ns) if dsum else ""
                res["sub0"] = M.breakdown(dmean)
        res["notes"] = notes
        out["windows"][nm] = res
        if not res["ranks"]:
            print("P12PROF %s: VIRHE no profile window (%s)" % (nm, "; ".join(notes) or "no table")); rc = 1; continue
        if len(res["ranks"]) < 4: notes.append("ranks %d of 4" % len(res["ranks"]))
        for rk, s in sorted(res["ranks"].items(), key=lambda kv: int(kv[0])):
            tot = s["wall_ms"] or (s["device_ms"] + s["wait_ms"])
            items = list(s["groups"].items()) + [("lane waits", s["wait_ms"])] + ([("host remainder", s["host_remainder_ms"])] if s["host_remainder_ms"] is not None else [])
            pc = lambda v: ("%.0f %%" % (100.0 * v / tot)) if tot else "-"
            g = s["groups"]; bc = s["by_class_ms"]
            print("P12PROF %s rank %s (PROFILED, not a speed; %d decode-only dump intervals of %d, %d steps, %s rows a step): wall %s ms a decode step "
                  "(step log, %d steps; steps before a dump left out) | drafter ctx+query %.2f ms %s (ctx %.2f + query %.2f) | MoE experts %.2f %s | "
                  "moe_other %.2f %s | attention + indexer %.2f %s (attn %.2f / idx %.2f) | mhc %.2f %s | collectives / gather_ar %.2f %s (coll %.2f / gar %.2f) | "
                  "dense + norms %.2f %s (dense_gemm %.2f / norm_elem %.2f / engram %.2f / other %.2f) | lane waits %.2f %s (stager %.2f / other %.2f) | "
                  "host remainder %s | device %.2f ms | ranked: %s" % (
                      nm, rk, s["intervals"], s["intervals_all"], s["steps"], s["rows"], ("%.2f" % s["wall_ms"]) if s["wall_ms"] else "-", s["wall_n"],
                      g["drafter ctx+query"], pc(g["drafter ctx+query"]), s["ctx_ms"], s["query_ms"], g["MoE experts"], pc(g["MoE experts"]),
                      g["moe_other"], pc(g["moe_other"]), g["attention + indexer"], pc(g["attention + indexer"]), bc.get("attention", 0.0), bc.get("indexer", 0.0),
                      g["mhc"], pc(g["mhc"]), g["collectives / gather_ar"], pc(g["collectives / gather_ar"]), bc.get("collective", 0.0), bc.get("gather_ar", 0.0),
                      g["dense + norms"], pc(g["dense + norms"]), bc.get("dense_gemm", 0.0), bc.get("norm_elem", 0.0), bc.get("embed_engram", 0.0), bc.get("other", 0.0),
                      s["wait_ms"], pc(s["wait_ms"]), s["stager_wait_ms"], s["other_wait_ms"],
                      ("%.2f ms %s" % (s["host_remainder_ms"], pc(s["host_remainder_ms"]))) if s["host_remainder_ms"] is not None else "- (no wall)",
                      s["device_ms"], "; ".join("%s %.2f" % kv for kv in sorted(items, key=lambda kv: -kv[1]))))
        for rk, s in sorted((res.get("newcomer") or {}).items(), key=lambda kv: int(kv[0])):
            g = s["groups"]
            print("P12PROFNC %s rank %s (PROFILED; newcomer step ESTIMATE: %d dump intervals with one non-decode step minus their decode steps x the decode-only "
                  "mean; %s rows a newcomer step; its own wall %s ms from the step log over %d steps) ms a newcomer step: %s | lane waits %.2f (stager %.2f) | device %.2f" % (
                      nm, rk, s["intervals"], s["rows"], s["wall_ms"], s["wall_n"], " | ".join("%s %.2f" % (k, g[k]) for k, _ in GROUPS),
                      s["wait_ms"], s["stager_wait_ms"], s["device_ms"]))
        if res.get("sub0"): print("P12SUB %s rank 0 (ms a decode step by op): %s" % (nm, res["sub0"][:2500]))
        if res.get("top0"): print("P12TOP %s rank 0: %s" % (nm, res["top0"][:2500]))
        if notes: print("P12PROF %s: NOTE %s" % (nm, "; ".join(notes)))
    if a.get("--json"):
        with open(a["--json"], "w") as f: json.dump(out, f, default=str)
    return rc


# ---------------------------------------------------------------- fit
FITC = [("EXL3 experts", ("experts",)), ("dense GEMM", ("dense_gemm",)), ("collectives (AR / gather_ar)", ("collective", "gather_ar")),
        ("attention + indexer", ("attention", "indexer")), ("mhc", ("mhc",)), ("drafter ctx+query", ("dspark_ctx", "dspark_query")),
        ("moe_other", ("moe_other",)), ("norms / other", ("norm_elem", "embed_engram", "other"))]


def cls_vals(bc, wait, wall, device):
    v = {k: sum(bc.get(c, 0.0) for c in cs) for k, cs in FITC}
    v["lane waits"] = wait
    v["host remainder"] = (wall - device - wait) if wall else None
    v["wall"] = wall
    return v


def lsq(pts):
    """least squares y = a + b x over [(x, y)] -> (a, b, r2) or None (fewer than 2 distinct x)."""
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    if len(pts) < 2 or len(set(round(x, 6) for x, _ in pts)) < 2: return None
    n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts); sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx; a = my - b * mx
    sst = sum((y - my) ** 2 for _, y in pts); sse = sum((y - a - b * x) ** 2 for x, y in pts)
    return a, b, (1.0 - sse / sst) if sst > 0 else 1.0


def fit(argv):
    """p12win.py fit PROF.json [--win NAME ...]: ms per added row by class -- across the windows (each window's decode-only mean at its
    mean rows a step) and inside each window (its decode-only dump intervals, when their rows spread over >= 2 rows)."""
    a, pos = args_of(argv)
    d = json.load(open(pos[0]))
    W = d.get("windows") or {}
    names = a["--win"] or [k for k in W]
    names = [n for n in names if (W.get(n) or {}).get("ranks")]
    if len(names) < 2:
        print("P12FIT VIRHE fewer than 2 profiled windows with a reading (%s)" % ", ".join(names)); return 1
    ranks = sorted({rk for n in names for rk in W[n]["ranks"]}, key=int)
    keys = [k for k, _ in FITC] + ["lane waits", "host remainder", "wall"]
    res = {}
    for rk in ranks:
        pts = {k: [] for k in keys}
        for n in names:
            s = W[n]["ranks"].get(rk)
            if not s: continue
            v = cls_vals(s["by_class_ms"], s["wait_ms"], s.get("wall_ms"), s["device_ms"])
            for k in keys: pts[k].append((s.get("rows"), v[k]))
        res[rk] = {k: lsq(pts[k]) for k in keys}
    r0 = ranks[0]
    print("P12FIT ms per added row by class (PROFILED, not a speed; least squares of a class's ms a decode step on the window's mean rows a step, "
          "decode-only dump intervals; windows %s; rank %s's numbers, the spread over ranks %s)" % (
              ", ".join("%s %.1f rows" % (n, W[n]["ranks"][r0]["rows"]) for n in names if r0 in W[n]["ranks"] and W[n]["ranks"][r0].get("rows") is not None), r0, "/".join(ranks)))
    out = {"windows": names, "ranks": {}, "within": {}}
    for k in keys:
        f0 = res[r0][k]
        bs = [res[rk][k][1] for rk in ranks if res[rk][k]]; as_ = [res[rk][k][0] for rk in ranks if res[rk][k]]
        if not f0:
            print("P12FIT %-30s | no fit (fewer than 2 distinct rows or no value)" % k); continue
        print("P12FIT %-30s | %+.3f ms per added row (ranks %.3f .. %.3f) | intercept %.2f ms at 0 rows (ranks %.2f .. %.2f) | r2 %.2f | at 1.6 / 8 / 16 / 48 rows: %s" % (
            k, f0[1], min(bs), max(bs), f0[0], min(as_), max(as_), f0[2], " / ".join("%.1f" % (f0[0] + f0[1] * x) for x in (1.6, 8, 16, 48))))
    for rk in ranks: out["ranks"][rk] = {k: (list(v) if v else None) for k, v in res[rk].items()}
    for n in names:
        s = W[n]["ranks"].get(r0) or {}
        P = s.get("points") or []
        xs = [p["rows"] for p in P if p.get("rows") is not None]
        if len(P) < 3 or not xs or max(xs) - min(xs) < 2.0:
            print("P12FITW %s rank %s: no within-window fit (%d decode-only intervals, rows %s .. %s)" % (n, r0, len(P), min(xs) if xs else "-", max(xs) if xs else "-")); continue
        fw = {}
        for k in keys:
            fw[k] = lsq([(p["rows"], cls_vals(p["by_class_ms"], p["wait_ms"], p.get("wall_ms"), p["device_ms"])[k]) for p in P])
        out["within"][n] = {k: (list(v) if v else None) for k, v in fw.items()}
        print("P12FITW %s rank %s (%d decode-only intervals, rows %.1f .. %.1f a step; ms per added row, intercept): %s" % (
            n, r0, len(P), min(xs), max(xs), " | ".join("%s %+.3f / %.2f" % (k, fw[k][1], fw[k][0]) for k in keys if fw[k])))
    if a.get("--json"):
        with open(a["--json"], "w") as f: json.dump(out, f)
    return 0


def fit_selftest(d):
    """synthetic: class c = a_c + b_c x rows on 4 windows (1.6, 8, 16, 48 rows) and 4 ranks (rank r's slope + 0.01 r); one window's
    intervals spread 40 .. 56 rows -> the fit returns every a_c / b_c, the rank spread, and the within-window slope."""
    A = {"experts": (2.0, 1.5), "dense_gemm": (1.0, 0.5), "collective": (0.5, 0.1), "gather_ar": (0.3, 0.2), "attention": (0.4, 0.3),
         "indexer": (0.1, 0.05), "mhc": (0.2, 0.1), "dspark_ctx": (0.5, 0.0), "dspark_query": (1.0, 0.0), "moe_other": (0.3, 0.2),
         "norm_elem": (0.2, 0.1), "embed_engram": (0.0, 0.0), "other": (0.0, 0.0)}
    def bc(x, r): return {c: a_ + (b + 0.01 * r) * x for c, (a_, b) in A.items()}
    W = {}
    for n, x in (("u", 1.6), ("c4", 16.0), ("c8", 8.0), ("c16", 48.0)):
        rk = {}
        for r in range(4):
            b_ = bc(x, r); dev = sum(b_.values()); wait = 1.0 + 0.05 * x
            pts = [dict(rows=xx, steps=8, by_class_ms=bc(xx, r), wait_ms=1.0 + 0.05 * xx, device_ms=sum(bc(xx, r).values()), wall_ms=sum(bc(xx, r).values()) + 1.0 + 0.05 * xx + 3.0)
                   for xx in ((40.0, 44.0, 52.0, 56.0) if n == "c16" else (x, x, x))]
            rk[str(r)] = dict(rows=x, by_class_ms=b_, wait_ms=wait, device_ms=dev, wall_ms=dev + wait + 3.0, points=pts)
        W[n] = {"ranks": rk}
    p = os.path.join(d, "fit.json"); json.dump({"windows": W}, open(p, "w"))
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = fit([p, "--json", os.path.join(d, "fo.json")])
    o = json.load(open(os.path.join(d, "fo.json"))); r0 = o["ranks"]["0"]; r3 = o["ranks"]["3"]
    ok = rc == 0 and abs(r0["EXL3 experts"][1] - 1.5) < 1e-9 and abs(r0["EXL3 experts"][0] - 2.0) < 1e-9 and abs(r3["EXL3 experts"][1] - 1.53) < 1e-9 \
        and abs(r0["collectives (AR / gather_ar)"][1] - 0.3) < 1e-9 and abs(r0["drafter ctx+query"][1]) < 1e-9 and abs(r0["drafter ctx+query"][0] - 1.5) < 1e-9 \
        and abs(r0["host remainder"][0] - 3.0) < 1e-9 and abs(r0["host remainder"][1]) < 1e-9 and abs(r0["lane waits"][1] - 0.05) < 1e-9 \
        and abs(o["within"]["c16"]["dense GEMM"][1] - 0.5) < 1e-9 and "c4" not in o["within"] and "(ranks 1.500 .. 1.530)" in buf.getvalue()
    return ok, buf.getvalue()


# ---------------------------------------------------------------- selftest
def selftest():
    import contextlib, io, shutil, tempfile, time
    d = tempfile.mkdtemp(prefix="p12win-")
    try:
        def ts(t):
            return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(t))) + ".%09dZ" % int(round((t - int(t)) * 1e9))
        ops = [(1, "embed_lookup_ggml", "token_embd.weight"), (2, "mhc_mix", "blk.0.hc_attn_fn.weight"), (3, "index_score", "blk.0.indexer.weight"),
               (4, "gemm_nt_ggml", "blk.0.attn_q_a.weight"), (5, "router_scored_topk", "blk.0.glm_router.correction"),
               (6, "moe_exl3_gate_up", "blk.0.ffn_gate_up_exps.0.weight"), (7, "all_reduce", "-"), (8, "moe_gather_all_reduce", "-"),
               (9, "attn_decode", "blk.0.attn.weight"), (10, "rmsnorm", "blk.0.attn_norm.weight"), (11, "logits_gemm_ggml", "output.weight"),
               (12, "rope_table", "-"), (13, "gemm_nt", "dspark.fc.weight"), (14, "embed_lookup_ggml", "token_embd.weight"),
               (15, "gemm_nt", "blk.2.attn_q_a.weight")]
        per_ms = {i: 0.1 * i for i, _, _ in ops}

        def table(rank, calls, t):
            L = ["%s I[%d] per-op timing, rank %d. Not comparable with an unprofiled run:" % (ts(t), rank, rank)]
            for i, op, w in ops:
                L.append("%s   %10d %12.3f %10.2f   dev          - %10.2f  %s  n=1  #%d %s" % (ts(t), calls, per_ms[i] * calls, per_ms[i] * 1000, 2.0, op, i, w))
            L.append("%s   %10s %12.3f   (waited %.3f ms besides)" % (ts(t), "total", 1.0, 0.0))
            return L
        lines, t, K = [], 1794480000.0, 0
        mark = {}

        def step(ph, qs, dt=0.05):
            nonlocal t, K
            K += 1
            ents = "".join("  [%d] slot=%d q=%d ctx=100" % (j, j, q) for j, q in enumerate(qs))
            lines.append("%s I step %d %s n_seq=%d n_tok=%d%s" % (ts(t), K, ph, len(qs), sum(qs), ents)); t += dt
            if K % 8 == 0:
                for r in range(4): lines.extend(table(r, K, t))
                t += 0.01
        # a c8-like window: 8 sequences, 3 of them drafting 1 (q 2), one newcomer prefill in the middle
        for _ in range(8): step("prefill", [900], 0.2)
        mark["c8_0"] = t
        for k in range(32): step("decode", [2, 2, 2, 1, 1, 1, 1, 1])
        step("MIXED", [1100, 1, 1, 1, 1, 1, 1, 1], 0.4)
        for k in range(31): step("decode", [2, 2, 2, 1, 1, 1, 1, 1])
        mark["c8_1"] = t
        # a 1u window: one sequence, q 3
        for _ in range(8): step("prefill", [600], 0.2)
        mark["u_0"] = t
        for _ in range(40): step("decode", [3], 0.02)
        mark["u_1"] = t
        p = os.path.join(d, "e.log.gz")
        with gzip.open(p, "wt") as f: f.write("\n".join(lines) + "\n")
        s0, s1 = os.path.join(d, "s0.json"), os.path.join(d, "s1.json")
        json.dump({"link": {"stream_bytes": 1000000000, "h2d_bytes": 5e8}, "experts": {"promotions": 10, "staged_bytes": 0}}, open(s0, "w"))
        json.dump({"link": {"stream_bytes": 1000000000 + 40 * 27e6, "h2d_bytes": 5e8 + 40 * 13e6}, "experts": {"promotions": 30, "staged_bytes": 2e6}}, open(s1, "w"))
        here = os.path.dirname(os.path.abspath(__file__))
        opnc = os.path.join(here, "..", "nc", "ds_opprof_nc.py")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r1 = rows([p, "--win", "c8:%f:%f" % (mark["c8_0"], mark["c8_1"]), "--win", "u:%f:%f" % (mark["u_0"] - 0.005, mark["u_1"]), "--every", "8",
                       "--st", "u:%s:%s:%f:%f" % (s0, s1, mark["u_0"] - 0.005, mark["u_1"]), "--json", os.path.join(d, "r.json")])
            r2 = prof([p, "--opnc", opnc, "--win", "c8:%f:%f" % (mark["c8_0"], mark["c8_1"]), "--win", "u:%f:%f" % (mark["u_0"] - 0.005, mark["u_1"]),
                       "--every", "8", "--json", os.path.join(d, "p.json")])
            bj = os.path.join(d, "b.jsonl")
            with open(bj, "w") as f:
                for x in (80.0, 90.0, 91.0):
                    f.write(json.dumps({"lever": "U1", "j": {"prompt": "essay", "decode_tps": x, "tokens_per_step": 1.6, "ms_per_decode_step": 19.0, "acceptance": 0.7},
                                        "st": {"stream_mb_step": 27.0, "h2d_mb_step": 10.0, "promotions": 5}}) + "\n")
                f.write(json.dumps({"lever": "U1W", "j": {"prompt": "essay", "decode_tps": 10.0}, "st": {}}) + "\n")
            mbl = os.path.join(d, "mb.txt")
            with open(mbl, "w") as f: f.write("x\nMB " + json.dumps({"tok_per_decode_step": 8.28, "ms_decode_step": 36.1, "agg_dec_tps": 184.5, "decode_share": 0.803}) + "\n")
            w0, w1 = os.path.join(d, "w0.json"), os.path.join(d, "w1.json")
            json.dump({"now_ms": 0, "step_ns": 0, "step_ns_decode": 0, "steps_decode": 0, "steps_mixed": 0, "steps_prefill": 0}, open(w0, "w"))
            json.dump({"now_ms": 3600, "step_ns": 3.4e9, "step_ns_decode": 3.0e9, "steps_decode": 63, "steps_mixed": 1, "steps_prefill": 0}, open(w1, "w"))
            r4 = split([p, "--win", "c8:%f:%f" % (mark["c8_0"], mark["c8_1"]), "--every", "8", "--mbline", mbl, "--sw0", w0, "--sw1", w1,
                        "--ref", "ref12 c8:8.28:36.1:0.803:184.5"])
            r5 = split([p, "--win", "c8u:%f:%f" % (mark["c8_0"], mark["c8_1"])])
            r3 = confirm([bj, "--lever", "U1", "--expect-tps", "90", "--expect-mb", "27", "--ref-tps", "82.24", "--ref-mb", "136.6"])
        o = buf.getvalue()
        R = json.load(open(os.path.join(d, "r.json")))["windows"]; P = json.load(open(os.path.join(d, "p.json")))["windows"]
        c8, u = R["c8"], R["u"]
        pc8, pu = P["c8"]["ranks"]["0"], P["u"]["ranks"]["0"]
        # per step every op once at 0.1 x idx ms: ctx = #12 + #13 = 2.5, query = #14 + #15 = 2.9; experts #6 = 0.6; collectives #7 + #8 = 1.5;
        # attention + indexer #9 + #3 = 1.2; dense + norms #4 + #10 + #11 + #1 = 2.6 (+ moe_other #5 0.5, mhc #2 0.2) -> device 12.0
        sp = [l for l in o.splitlines() if l.startswith("P12SPLIT c8 ")]
        spu = [l for l in o.splitlines() if l.startswith("P12SPLIT c8u ")]
        nc0 = P["c8"].get("newcomer", {}).get("0", {})
        ok = r4 == 0 and r5 == 0 and len(sp) == 1 and "needs decode_share >= 0.872" in sp[0] and "decode >= 249.1 tok/s" in sp[0] and "(d) steps before" in sp[0] \
            and "(b) newcomer / MIXED 1 steps 400 ms" in spu[0] and "(a) decode 63 steps" in spu[0] and "between the step timers" in sp[0] \
            and nc0.get("intervals") == 1 and abs(nc0.get("device_ms", 0) - 12.0) < 1e-6 and "P12PROFNC c8 rank 0" in o
        fok, fout = fit_selftest(d)
        if not fok: o += "\nFIT FAILED\n" + fout
        ok = ok and fok and r1 == 0 and r2 == 0 and r3 == 0 and c8["decode_steps"] == 63 and c8["buckets"]["9-15"]["n"] == 63 and abs(c8["depth0_share"] - 5 / 8) < 1e-9 \
            and c8["all_depth0_steps"] == 0 and u["buckets"]["1-4"]["n"] == 40 and u["by_q"]["3"]["n"] > 30 and abs(u["st"]["stream_mb_step"] - 27.0) < 0.01 \
            and u["st"]["promotions"] == 20 and abs(pc8["groups"]["drafter ctx+query"] - 5.4) < 1e-6 and abs(pc8["groups"]["collectives / gather_ar"] - 1.5) < 1e-6 \
            and abs(pc8["groups"]["MoE experts"] - 0.6) < 1e-6 and abs(pc8["groups"]["attention + indexer"] - 1.2) < 1e-6 and abs(pc8["device_ms"] - 12.0) < 1e-6 \
            and pc8["intervals"] < pc8["intervals_all"] and pu["intervals"] == pu["intervals_all"] and pu["host_remainder_ms"] is not None \
            and "P12CONFIRM" in o and "87.00 tok/s" in o and "-3.3 %" in o and "P12TOP c8 rank 0" in o and "P12SUB u rank 0" in o
        print("SELFTEST " + ("OK: rows (buckets, depth-0 share, by q, > 48 rows, /stats per decode step), split (decode / newcomer / gaps / dumps, the needed share and rate), fit (ms per added row by class across and within windows, rank spread), prof (decode-only intervals, classes, host remainder, top ops, newcomer estimate), confirm"
                             if ok else "FAILED: %s %s %s %s %s\n%s" % (r1, r2, r3, r4, r5, "\n".join(l[:600] for l in o.splitlines() if not l.startswith(("P12PROF c8 rank", "P12PROF u rank"))))))
        return 0 if ok else 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main(argv):
    if not argv: print(__doc__); return 2
    m = {"confirm": confirm, "rows": rows, "split": split, "prof": prof, "fit": fit}
    if argv[0] == "selftest": return selftest()
    if argv[0] in m: return m[argv[0]](argv[1:])
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
