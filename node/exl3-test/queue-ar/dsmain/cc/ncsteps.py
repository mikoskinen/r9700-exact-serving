#!/usr/bin/env python3
"""ncsteps.py (12.11.2026, queue-ar/dsmain/nc, ds-main-port NC-profile builder) -- the readings of dsm-ncprof.sh from the
engine's step log, /stats and the miss log. Standard library only; reads .gz or plain files.

The engine log is `docker logs -t` (each line led by the daemon's receipt time, RFC 3339, UTC), with RADIANCE_LOG_STEPS=1:
"step K prefill|decode|MIXED n_seq=.. n_tok=..  [i] slot=S q=Q ctx=C" (core/engine.cpp:1222, printed when step K is
planned, before it runs; the regexes are queue-ar/dsmain/longid.py's STEP / ENT). A step's wall = the receipt time of the
next step's line minus its own (UNPROFILED in START U; a gap over --idle s is idle, not a step). /stats are ncone.py's
compact copies (or full ones, compacted here).

  ncsteps.py alone --log L --reqs R.jsonl --stats S.jsonl --tag T [--skip-mod 8] [--label X] [--json OUT]
      one-token ALONE newcomers (ncone.py): per prompt its step lines (phase, n_tok, K), TTFT, the /stats delta between
      the snapshots around it (steps, step ms = step_ns, staged units / bytes / copies / layers, h2d / link staged /
      stream bytes, per-rank stream bytes, slot misses, snapshots). --skip-mod M: prompts whose step K % M == 0 (a
      --profile-ops table dump follows that step) are left out of the wall statistics (counted apart).
  ncsteps.py c8 --log L --reqs R.jsonl --tag T --sa W0.json --sb W1.json [--samples SAMP.jsonl] [--mbline F] [--skip-mod 8]
      the window between ncmb.py's two marks: every step by class (decode / MIXED / prefill) with wall p50 / mean / sum;
      /stats deltas (steps_prefill / steps_mixed / steps_decode / steps_held, step_ns, step_ns_decode); staging per
      non-decode step (window delta / non-decode steps, and a least-squares split over the 2 s samples inside the window:
      bytes = a x non-decode steps + b x decode steps); each newcomer sent in the window: the steps from its arrival to its first
      token, its own prefill step (the entry with q >= 256 and ctx <= 64), that step's phase, the decode rows that rode
      in it, the wait before it; decode_share / agg_dec_tps from the MB line.
  ncsteps.py miss --file MISS.log[.gz] --log L --win NAME:T0:T1 [...] [--json OUT]
      RADIANCE_DEBUG_MISS_LOG (core/place/misslog.h): per window the trunk lines by kind and rank -- live experts (the
      rank's experts with a nonzero routing count, summed over layers = distinct experts touched), vram / host-pool /
      other, the stager armed share, host-pool KiB; the windows' step ranges from the step log.
  ncsteps.py ld --log L --runs D.jsonl --pskld stack4/pskld.py [--grid 2048] [--json OUT]
      section (d), D7 under a long prefill: per run (c4 + one cold ~64k deepq probe, stack4's pskld.py --load; the body writes
      one line a run: name, x X1 | X2, t0, t1, longid = pskld's LONGID json, md = the /metrics deltas, text = the probe's answer)
      the step log in [t0, t1] (MIXED / prefill / decode steps) and the probe's chunks (pskld.py's find_chunks, loaded READ
      ONLY): sizes, boundaries off the --grid, MIXED vs prefill, the decode rows / drafted rows riding each chunk step, its wall;
      X2 vs X1 in % per reading; identity (every run's answer vs the first X1's, X1 vs X1, X2 vs X2) with the chunk boundaries
      compared (a difference reported, labelled class (ii) where the boundaries differ); guards: cold (cached 0), status 200, no
      c4 error, step lines, chunks found, X2 drafted ~0 (<= max(800, 10 % of X1's mean): depth 0 forced keeps only the probes).
  ncsteps.py report --in NAME=FILE.json [...] [--ref-c8 TEXT]      the item's final "RESULT xnc" lines (printed).
  ncsteps.py selftest
"""
import calendar, gzip, importlib.util, json, math, os, re, statistics, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")    # longid.py's
ENT = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")                           # longid.py's
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")
FRESH_Q, FRESH_CTX = 256, 64


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def ts_split(line):
    m = TSRE.match(line)
    if not m: return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z": t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


def read_steps(path, idle=3.0):
    """[{t, K, ph, n_seq, n_tok, ents: [(slot, q, ctx)], wall_ms}] in log order; wall = next line's time - own."""
    out = []
    with opn(path) as f:
        for l in f:
            if "n_seq=" not in l: continue
            t, rest = ts_split(l.rstrip("\n"))
            m = STEP.search(rest)
            if not m or t is None: continue
            ents = [(int(s), int(q), int(c)) for _, s, q, c in ENT.findall(m.group(5))]
            out.append({"t": t, "K": int(m.group(1)), "ph": m.group(2), "n_seq": int(m.group(3)), "n_tok": int(m.group(4)), "ents": ents})
    for a, b in zip(out, out[1:]):
        d = b["t"] - a["t"]
        a["wall_ms"] = 1000.0 * d if 0 <= d <= idle else None
    if out: out[-1]["wall_ms"] = None
    return out


def load_compact():
    spec = importlib.util.spec_from_file_location("ncone", os.path.join(HERE, "ncone.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.compact


def jl(path, tag=None):
    rows = []
    try:
        with opn(path) as f:
            for l in f:
                l = l.strip()
                if not l: continue
                try: d = json.loads(l)
                except ValueError: continue
                if tag is None or d.get("tag") == tag: rows.append(d)
    except OSError:
        pass
    return rows


def num(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict): return 0
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def sdelta(A, B):
    """compact /stats B - A."""
    g = lambda *p: num(B, *p) - num(A, *p)   # noqa: E731
    ra, rb = {r.get("rank"): r for r in (A.get("experts") or {}).get("ranks") or []}, {r.get("rank"): r for r in (B.get("experts") or {}).get("ranks") or []}
    rs, miss, mcls = [], [], {}
    for k in sorted(rb, key=lambda x: (x is None, x)):
        x, y = ra.get(k) or {}, rb[k]
        rs.append((num(y, "stream_bytes") - num(x, "stream_bytes")) / 1e9)
        mm = 0
        for c, v in (y.get("miss") or {}).items():
            dv = (v or 0) - ((x.get("miss") or {}).get(c) or 0)
            mm += dv; mcls[c] = mcls.get(c, 0) + dv
        miss.append(mm)
    return {"steps": g("steps"), "prefill": g("steps_prefill"), "mixed": g("steps_mixed"), "decode": g("steps_decode"), "held": g("steps_held"),
            "step_ms": g("step_ns") / 1e6, "step_ms_decode": g("step_ns_decode") / 1e6, "ckpts": g("checkpoints_written"),
            "staged_units": g("experts", "staged_units"), "staged_gb_r0": g("experts", "staged_bytes") / 1e9,
            "stage_copies": g("experts", "stage_copies"), "stage_layers": g("experts", "stage_layers"),
            "stage_overflow": g("experts", "stage_overflow"), "routed_r0": g("experts", "routed"),
            "routed_res_r0": g("experts", "routed_res"), "h2d_gb": g("link", "h2d_bytes") / 1e9,
            "link_staged_gb": g("link", "staged_bytes") / 1e9, "link_stream_gb": g("link", "stream_bytes") / 1e9,
            "ar_gb": g("link", "ar_bytes") / 1e9, "rank_stream_gb": [round(x, 3) for x in rs], "rank_miss": miss, "miss_by_class": mcls}


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v: return None
    k = (len(v) - 1) * p / 100.0
    f = int(math.floor(k)); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def stat(v, nd=1):
    v = [x for x in v if x is not None]
    if not v: return {"n": 0}
    return {"n": len(v), "p50": round(pct(v, 50), nd), "mean": round(statistics.mean(v), nd), "sum": round(sum(v), nd),
            "min": round(min(v), nd), "max": round(max(v), nd)}


def r3(x, n=3):
    return None if x is None else round(x, n)


# ---------------------------------------------------------------- alone
def cmd_alone(a):
    steps = read_steps(a.log, a.idle)
    reqs = sorted(jl(a.reqs, a.tag), key=lambda d: d.get("j", 0))
    snaps = {d["k"]: d for d in jl(a.stats, a.tag)}
    rows, guard = [], []
    for r in reqs:
        j = r.get("j")
        t0, te = r.get("t0"), r.get("t_end")
        st = [s for s in steps if t0 is not None and te is not None and t0 - 0.02 <= s["t"] <= te + 0.2]
        A, B = (snaps.get(j) or {}).get("s"), (snaps.get(j + 1) or {}).get("s")
        dl = sdelta(A, B) if isinstance(A, dict) and isinstance(B, dict) and "error" not in A and "error" not in B else None
        row = {"j": j, "prompt": r.get("prompt"), "cached": r.get("cached"), "ttft": r3(r.get("ttft")), "t_resp": r3(r.get("t_resp")),
               "err": r.get("err"), "log_steps": [(s["K"], s["ph"], s["n_tok"]) for s in st], "st": dl}
        row["dump_step"] = bool(a.skip_mod and any(s["K"] % a.skip_mod == 0 for s in st))
        rows.append(row)
    ok = [x for x in rows if not x["err"]]
    use = [x for x in ok if not x["dump_step"]]
    S = lambda k, rr=use: [x["st"][k] for x in rr if x["st"]]   # noqa: E731
    if not steps: guard.append("no step-log line in the engine log (RADIANCE_LOG_STEPS)")
    if len(ok) < len(rows) or not ok: guard.append("%d of %d prompts failed" % (len(rows) - len(ok), len(rows)))
    one = [x for x in ok if len(x["log_steps"]) == 1 and x["log_steps"][0][1] == "prefill"]
    if len(one) < len(ok): guard.append("%d of %d prompts not exactly one prefill step in the log (%s)" % (
        len(ok) - len(one), len(ok), "; ".join("j%s %s" % (x["j"], x["log_steps"]) for x in ok if x not in one)[:300]))
    st1 = [x for x in ok if x["st"] and x["st"]["steps"] == 1]
    if len(st1) < len(ok): guard.append("/stats steps delta != 1 for %d of %d prompts" % (len(ok) - len(st1), len(ok)))
    if not any(x["st"] and x["st"]["steps"] > 0 for x in ok): guard.append("/stats step counters did not move")
    bad = [x for x in ok if not (x["prompt"] and 900 <= x["prompt"] <= 1300 and (x["cached"] or 0) <= FRESH_CTX)]
    if bad: guard.append("not fresh ~1.1k prompts: %s" % ", ".join("j%s %s/%s cached" % (x["j"], x["prompt"], x["cached"]) for x in bad)[:300])
    summ = {"label": a.label, "tag": a.tag, "n": len(rows), "ok": len(ok), "dump_excluded": len(ok) - len(use),
            "prompt": stat([x["prompt"] for x in ok], 0), "cached_max": max([(x["cached"] or 0) for x in ok] or [0]),
            "ttft_s": stat([x["ttft"] for x in ok], 3), "t_resp_s": stat([x["t_resp"] for x in ok], 3),
            "phases": sorted({s[1] for x in ok for s in x["log_steps"]}), "n_tok": stat([s[2] for x in ok for s in x["log_steps"]], 0),
            "step_ms": stat(S("step_ms"), 1), "staged_units": stat(S("staged_units"), 1), "staged_gb_r0": stat(S("staged_gb_r0"), 3),
            "stage_copies": stat(S("stage_copies"), 1), "stage_layers": stat(S("stage_layers"), 1), "stage_overflow": stat(S("stage_overflow"), 1),
            "h2d_gb": stat(S("h2d_gb"), 3), "link_staged_gb": stat(S("link_staged_gb"), 3), "link_stream_gb": stat(S("link_stream_gb"), 3),
            "ar_gb": stat(S("ar_gb"), 3), "ckpts": stat(S("ckpts"), 2),
            "routed_r0": stat(S("routed_r0"), 0), "routed_res_r0": stat(S("routed_res_r0"), 0),
            "rank_stream_gb": [r3(statistics.mean(v)) for v in zip(*[x["st"]["rank_stream_gb"] for x in use if x["st"]])] if use else [],
            "rank_miss": [r3(statistics.mean(v), 1) for v in zip(*[x["st"]["rank_miss"] for x in use if x["st"]])] if use else [],
            "guard": guard, "rows": rows}
    line = "NCA %s: %d prompts (%d ok, %d dump steps left out of the wall) prompt %s tok (cached max %s) | TTFT p50 %s s, reply p50 %s s | %s step(s) a prompt (%s, n_tok p50 %s) | step wall (/stats step_ns) p50 %s mean %s ms | staged a step: %s units, %s GB rank 0, link staged %s GB (4 ranks), copies %s, layers %s, overflow %s | h2d %s GB, link stream %s GB, per-rank stream %s GB, slot misses %s | snapshots %s%s" % (
        a.label, len(rows), len(ok), summ["dump_excluded"], summ["prompt"].get("p50"), summ["cached_max"], summ["ttft_s"].get("p50"),
        summ["t_resp_s"].get("p50"), "1" if len(one) == len(ok) else "NOT 1", "/".join(summ["phases"]), summ["n_tok"].get("p50"),
        summ["step_ms"].get("p50"), summ["step_ms"].get("mean"), summ["staged_units"].get("mean"), summ["staged_gb_r0"].get("mean"),
        summ["link_staged_gb"].get("mean"), summ["stage_copies"].get("mean"), summ["stage_layers"].get("mean"), summ["stage_overflow"].get("mean"),
        summ["h2d_gb"].get("mean"), summ["link_stream_gb"].get("mean"), summ["rank_stream_gb"], summ["rank_miss"], summ["ckpts"].get("mean"),
        (" -- VIRHE " + "; ".join(guard)) if guard else "")
    print(line)
    summ["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(summ, f)
    return 0 if not guard else 2


# ---------------------------------------------------------------- c8
def lsq2(rows):
    """y = a x1 + b x2 (no intercept) by the normal equations; rows [(x1, x2, y)]."""
    s11 = sum(x1 * x1 for x1, x2, y in rows); s22 = sum(x2 * x2 for x1, x2, y in rows); s12 = sum(x1 * x2 for x1, x2, y in rows)
    s1y = sum(x1 * y for x1, x2, y in rows); s2y = sum(x2 * y for x1, x2, y in rows)
    det = s11 * s22 - s12 * s12
    if len(rows) < 3 or abs(det) < 1e-9: return None, None
    return (s1y * s22 - s2y * s12) / det, (s2y * s11 - s1y * s12) / det


def cmd_c8(a):
    compact = load_compact()
    steps = read_steps(a.log, a.idle)
    rr = jl(a.reqs, a.tag)
    marks = sorted([d for d in rr if d.get("ev") == "mark"], key=lambda d: d["k"])
    guard = []
    if len(marks) < 2:
        print("NCC %s: VIRHE no window marks in %s" % (a.label, a.reqs)); return 2
    t0, t1 = marks[0]["t"], marks[1]["t"]
    W = [s for s in steps if t0 <= s["t"] < t1]
    if not W: guard.append("no step-log line in the window")
    dump = lambda s: bool(a.skip_mod) and s["K"] % a.skip_mod == 0   # noqa: E731
    cls = {}
    for ph in ("decode", "MIXED", "prefill"):
        ss = [s for s in W if s["ph"] == ph]
        cls[ph] = {"steps": len(ss), "wall_ms": stat([s.get("wall_ms") for s in ss if not dump(s)], 1),
                   "n_tok": stat([s["n_tok"] for s in ss], 0), "n_seq": stat([s["n_seq"] for s in ss], 1)}
    try:
        A, B = compact(json.load(open(a.sa))), compact(json.load(open(a.sb)))
        dl = sdelta(A, B)
    except Exception as e:
        dl = None; guard.append("no /stats pair (%s)" % str(e)[:100])
    nd_st = (dl["prefill"] + dl["mixed"]) if dl else 0
    if dl and dl["steps"] <= 0: guard.append("/stats step counters did not move")
    nd_log = cls["MIXED"]["steps"] + cls["prefill"]["steps"]
    if dl and abs(nd_st - nd_log) > 2: guard.append("non-decode steps /stats %d vs step log %d" % (nd_st, nd_log))
    stx = {}
    if dl:
        stx = {"steps": dl["steps"], "prefill": dl["prefill"], "mixed": dl["mixed"], "decode": dl["decode"], "held": dl["held"],
               "nondecode_ms_mean": r3((dl["step_ms"] - dl["step_ms_decode"]) / nd_st, 1) if nd_st else None,
               "decode_ms_mean": r3(dl["step_ms_decode"] / dl["decode"], 1) if dl["decode"] else None,
               "decode_share": r3(dl["step_ms_decode"] / dl["step_ms"]) if dl["step_ms"] else None,
               "per_nd": {k: r3(dl[k] / nd_st, 3) if nd_st else None for k in ("staged_units", "staged_gb_r0", "stage_copies", "stage_layers",
                                                                                  "stage_overflow", "h2d_gb", "link_staged_gb", "link_stream_gb", "ckpts")},
               "per_step": {k: r3(dl[k] / dl["steps"], 4) if dl["steps"] else None for k in ("link_stream_gb", "h2d_gb", "staged_gb_r0")},
               "window": {k: (r3(v, 3) if isinstance(v, float) else v) for k, v in dl.items()}}
    # the 2 s samples: bytes = a x non-decode steps + b x decode steps
    reg = {}
    sm = sorted((d for d in jl(a.samples) if t0 - 2.5 <= d.get("t", 0) <= t1 + 2.5), key=lambda d: d.get("t", 0)) if a.samples else []
    if len(sm) >= 4:   # the samples around the window only (steady state; the closed loop's first burst left out)
        iv = []
        for p, q in zip(sm, sm[1:]):
            d = sdelta(p["s"], q["s"])
            iv.append((d["prefill"] + d["mixed"], d["decode"], d))
        for k in ("staged_gb_r0", "link_staged_gb", "link_stream_gb", "h2d_gb", "staged_units"):
            x, y = lsq2([(n, m, d[k]) for n, m, d in iv])
            reg[k] = {"per_nd": r3(x, 3), "per_decode": r3(y, 4)}
        reg["intervals"] = len(iv); reg["nd_steps"] = sum(n for n, m, d in iv)
    # newcomers
    done = [d for d in rr if d.get("ev") == "done" and d.get("t_send") is not None and t0 <= d["t_send"] < t1]
    nc = []
    for d in sorted(done, key=lambda x: x["t_send"]):
        if d.get("err"): continue
        tf = (d["t0"] + d["ttft"]) if d.get("t0") and d.get("ttft") else None
        span = [s for s in steps if d["t_send"] <= s["t"] <= (tf if tf else d["t_send"] + 30)]
        own = next((s for s in span if any(q >= FRESH_Q and c <= FRESH_CTX for _, q, c in s["ents"])), None)
        before = [s for s in span if own is None or s["t"] < own["t"]]
        x = {"w": d.get("w"), "prompt": d.get("prompt"), "cached": d.get("cached"), "ttft": r3(d.get("ttft")),
             "steps_to_first": len(span), "found": own is not None}
        if own:
            # its slot (the fresh entry whose q is closest to prompt - cached); later chunks of the same prompt = that slot again
            # with a prefill-sized q before its first token (a 1099-token prompt is expected in ONE chunk)
            want_q = (d.get("prompt") or 0) - (d.get("cached") or 0)
            fr = [(abs(q - want_q), sl) for sl, q, c in own["ents"] if q >= FRESH_Q and c <= FRESH_CTX]
            slot = min(fr)[1] if fr else None
            more = [s for s in span if s["t"] > own["t"] and any(sl == slot and q > 16 for sl, q, c in s["ents"])]
            x.update(K=own["K"], ph=own["ph"], n_tok=own["n_tok"], n_seq=own["n_seq"], wall_ms=r3(own.get("wall_ms"), 1),
                     slot=slot, chunks=1 + len(more), later_chunk_phases="".join(s["ph"][0] for s in more),
                     dump=dump(own), fresh=sum(1 for _, q, c in own["ents"] if q >= FRESH_Q and c <= FRESH_CTX),
                     rode=sum(1 for _, q, c in own["ents"] if q < FRESH_Q),
                     wait_steps=len(before), wait_ms=r3(1000 * (own["t"] - d["t_send"]), 1),
                     wait_phases="".join(s["ph"][0] for s in before))
        nc.append(x)
    found = [x for x in nc if x["found"]]
    if done and len(found) < len(nc): guard.append("%d of %d newcomers without their prefill step in the log" % (len(nc) - len(found), len(nc)))
    badf = [x for x in nc if not (x["prompt"] and 900 <= x["prompt"] <= 1300 and (x["cached"] or 0) <= FRESH_CTX)]
    if badf: guard.append("not fresh ~1.1k newcomers: %s" % ", ".join("%s/%s" % (x["prompt"], x["cached"]) for x in badf)[:200])
    mb = {}
    if a.mbline:
        try:
            for l in opn(a.mbline):
                if l.startswith("MB "): mb = json.loads(l[3:]); break
        except (OSError, ValueError): pass
    if a.mbline and (not mb or "error" in mb): guard.append("no MB line")
    nwall = [x["wall_ms"] for x in found if not x.get("dump")]
    summ = {"label": a.label, "tag": a.tag, "window_s": r3(t1 - t0, 1), "classes": cls, "stats": stx, "regression": reg,
            "newcomers": {"n": len(nc), "found": len(found), "mixed": sum(1 for x in found if x["ph"] == "MIXED"),
                          "prefill": sum(1 for x in found if x["ph"] == "prefill"),
                          "shared_step": sum(1 for x in found if x["fresh"] > 1),
                          "multi_chunk": sum(1 for x in found if x["chunks"] > 1),
                          "rode": stat([x["rode"] for x in found], 2), "rode_any": sum(1 for x in found if x["rode"] > 0),
                          "wall_ms": stat(nwall, 1), "wait_steps": stat([x["wait_steps"] for x in found], 1),
                          "wait_ms": stat([x["wait_ms"] for x in found], 1), "ttft_s": stat([x["ttft"] for x in nc], 3),
                          "prompt": stat([x["prompt"] for x in nc], 0), "cached_max": max([(x["cached"] or 0) for x in nc] or [0]),
                          "rows": nc},
            "mb": {k: mb.get(k) for k in ("agg_dec_tps", "decode_share", "ms_decode_step", "per_stream_dec_p50", "ttft_p50", "requests",
                                          "errors", "steps", "prefill_tok", "tok_per_decode_step", "sends_in_window", "done_in_window")},
            "guard": guard}
    n = summ["newcomers"]
    line = "NCC %s: window %s s, %d steps in the log: decode %d (wall p50 %s / mean %s ms), MIXED %d (p50 %s / mean %s ms, n_tok p50 %s), prefill %d (p50 %s / mean %s ms) | /stats: prefill %s mixed %s decode %s held %s; non-decode %s ms mean, decode %s ms mean, decode_share %s | newcomers %d (%d found): %d MIXED / %d prefill, %d sharing a step, %d in more than one chunk; decode rows riding p50 %s (any: %d of %d); its step wall p50 %s ms; arrival -> its step p50 %s steps / %s ms; TTFT p50 %s s | staged a non-decode step %s units, %s GB r0, link staged %s GB, layers %s, h2d %s GB; regression a non-decode step: link staged %s GB, stream %s GB (decode step: stream %s GB) | MB agg_dec_tps %s decode_share %s%s" % (
        a.label, summ["window_s"], len(W), cls["decode"]["steps"], cls["decode"]["wall_ms"].get("p50"), cls["decode"]["wall_ms"].get("mean"),
        cls["MIXED"]["steps"], cls["MIXED"]["wall_ms"].get("p50"), cls["MIXED"]["wall_ms"].get("mean"), cls["MIXED"]["n_tok"].get("p50"),
        cls["prefill"]["steps"], cls["prefill"]["wall_ms"].get("p50"), cls["prefill"]["wall_ms"].get("mean"),
        stx.get("prefill"), stx.get("mixed"), stx.get("decode"), stx.get("held"), stx.get("nondecode_ms_mean"), stx.get("decode_ms_mean"),
        stx.get("decode_share"), n["n"], n["found"], n["mixed"], n["prefill"], n["shared_step"], n["multi_chunk"], n["rode"].get("p50"), n["rode_any"], n["found"],
        n["wall_ms"].get("p50"), n["wait_steps"].get("p50"), n["wait_ms"].get("p50"), n["ttft_s"].get("p50"),
        (stx.get("per_nd") or {}).get("staged_units"), (stx.get("per_nd") or {}).get("staged_gb_r0"), (stx.get("per_nd") or {}).get("link_staged_gb"),
        (stx.get("per_nd") or {}).get("stage_layers"), (stx.get("per_nd") or {}).get("h2d_gb"),
        (reg.get("link_staged_gb") or {}).get("per_nd"), (reg.get("link_stream_gb") or {}).get("per_nd"), (reg.get("link_stream_gb") or {}).get("per_decode"),
        summ["mb"].get("agg_dec_tps"), summ["mb"].get("decode_share"), (" -- VIRHE " + "; ".join(guard)) if guard else "")
    print(line)
    summ["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(summ, f)
    return 0 if not guard else 2


# ---------------------------------------------------------------- miss log
def cmd_miss(a):
    steps = read_steps(a.log, a.idle)
    wins = []
    for w in a.win:
        nm, x, y = w.split(":")
        ks = [s["K"] for s in steps if float(x) <= s["t"] < float(y)]
        kind = {s["K"]: s["ph"] for s in steps if float(x) <= s["t"] < float(y)}
        wins.append((nm, set(ks), kind))
    hdr, agg = {}, {}
    with opn(a.file) as f:
        for l in f:
            if l.startswith("#"):
                m = re.match(r"# rank (\d+) expert_units (\d+)", l)
                if m: hdr[int(m.group(1))] = int(m.group(2))
                m = re.match(r"# model \S+ arch \S+ tp (\d+) routed_layers (\d+) trunk_layers (\d+) experts (\d+)", l)
                if m: hdr["trunk_layers"], hdr["experts"] = int(m.group(3)), int(m.group(4))
                continue
            p = l.split()
            if len(p) < 16: continue
            try:
                step, kind, rank = int(p[0]), p[1], int(p[2])
                tok, st, live, v, h, o, hk = int(p[3]), int(p[6]), int(p[8]), int(p[9]), int(p[10]), int(p[11]), int(p[13])
            except ValueError:
                continue
            if kind == "draft": continue
            k = (step, rank)
            c = agg.setdefault(k, {"kind": kind, "tok": tok, "st": 0, "live": 0, "v": 0, "h": 0, "o": 0, "hk": 0})
            c["st"] = max(c["st"], st); c["live"] += live; c["v"] += v; c["h"] += h; c["o"] += o; c["hk"] += hk
    out = {"hdr": hdr, "windows": {}}
    lines = []
    for nm, ks, kind in wins:
        res = {}
        for (step, rank), c in agg.items():
            if step not in ks: continue
            key = "%s|%d" % (kind.get(step, c["kind"]), rank)
            r = res.setdefault(key, {"n": 0, "live": [], "v": [], "h": [], "o": [], "hk": [], "st": 0, "tok": []})
            r["n"] += 1; r["st"] += c["st"]
            for f in ("live", "v", "h", "o", "hk", "tok"): r[f].append(c[f])
        o2 = {}
        for key, r in sorted(res.items()):
            ph, rank = key.split("|")
            units = hdr.get(int(rank))
            ml = statistics.mean(r["live"])
            o2[key] = {"steps": r["n"], "live_mean": round(ml, 1), "of_units": units, "live_pct": round(100 * ml / units, 1) if units else None,
                       "vram_mean": round(statistics.mean(r["v"]), 1), "host_mean": round(statistics.mean(r["h"]), 1),
                       "other_mean": round(statistics.mean(r["o"]), 1), "host_mib_mean": round(statistics.mean(r["hk"]) / 1024, 1),
                       "staged_share": round(r["st"] / r["n"], 3), "tok_p50": pct(r["tok"], 50)}
        out["windows"][nm] = {"steps_in_log": len(ks), "per_kind_rank": o2}
        lines.append("%s (%d steps): %s" % (nm, len(ks), " | ".join("%s rank %s: %s steps, live %s of %s (%s %%), vram %s host %s (%s MiB) other %s, stager armed %s" % (
            k.split("|")[0], k.split("|")[1], v["steps"], v["live_mean"], v["of_units"], v["live_pct"], v["vram_mean"], v["host_mean"],
            v["host_mib_mean"], v["other_mean"], v["staged_share"]) for k, v in o2.items())))
    bad = [nm for nm, ks, kind in wins if not ks]
    line = "NCM distinct experts touched (miss log, trunk lines, per step and rank): " + " || ".join(lines) + (" -- VIRHE no step in %s" % bad if bad else "")
    print(line)
    out["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f)
    return 0 if not bad and agg else 2


# ---------------------------------------------------------------- (d) D7 under a long prefill
def textcmp(p, q):
    try: a, b = open(p, errors="replace").read(), open(q, errors="replace").read()
    except OSError as e: return "not compared (%s)" % str(e)[:80]
    if a == b: return "IDENTICAL (%d chars)" % len(a)
    f = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return "DIFFERENT at char %d (%d vs %d chars)" % (f, len(a), len(b))


def cmd_ld(a):
    spec = importlib.util.spec_from_file_location("pskld_ro", a.pskld)
    pk = importlib.util.module_from_spec(spec); spec.loader.exec_module(pk)
    steps = read_steps(a.log, a.idle)
    runs = jl(a.runs)
    guard, rows = [], []
    for r in runs:
        li, md = r.get("longid") or {}, r.get("md") or {}
        W = [x for x in steps if r["t0"] <= x["t"] <= r["t1"]]
        byk = {x["K"]: x for x in W}
        mix = {ph: sum(1 for x in W if x["ph"] == ph) for ph in ("decode", "MIXED", "prefill")}
        n = li.get("prompt_tokens") or 0
        ch = pk.find_chunks([(x["K"], x["ph"], x["n_seq"], x["n_tok"], x["ents"]) for x in W], n) if n else None
        row = {"name": r["name"], "x": r["x"], "longid": li, "md": md, "mix": mix, "text": r.get("text")}
        if ch:
            cs = ch["chunks"]
            pre = cs   # find_chunks stops at the prompt's end: every entry is a prefill chunk (the last may be a short tail)
            sz = {}
            for c in pre: sz[c[1]] = sz.get(c[1], 0) + 1
            off = sum(1 for c in pre[:-1] if (c[0] + c[1]) % a.grid)
            dec_rows, dec_tok, walls = [], [], []
            for c in pre:
                st = byk.get(c[5])
                if not st: continue
                d = [q for sl, q, cx in st["ents"] if q <= 16 and cx > 0]
                dec_rows.append(len(d)); dec_tok.append(sum(d))
                walls.append(st.get("wall_ms"))
            row["chunks"] = {"n": len(pre), "starts": [c[0] for c in pre], "sizes": sorted(sz.items(), key=lambda kv: -kv[1])[:6],
                             "off_grid": off, "mixed": sum(1 for c in pre if c[2] == "MIXED"), "prefill": sum(1 for c in pre if c[2] == "prefill"),
                             "dec_rows_mean": r3(statistics.mean(dec_rows), 2) if dec_rows else None,
                             "dec_tok_mean": r3(statistics.mean(dec_tok), 2) if dec_tok else None,
                             "wall_ms": stat([w for w in walls if w is not None], 1)}
        else:
            row["chunks"] = None
            guard.append("%s: the probe's chunks not found in the step log (%d lines in the window)" % (r["name"], len(W)))
        if not W: guard.append("%s: no step-log line in its window" % r["name"])
        if li.get("status") != 200: guard.append("%s: probe status %s" % (r["name"], li.get("status")))
        if li.get("cached") not in (0, None) or li.get("cached") is None: guard.append("%s: probe not cold (cached %s)" % (r["name"], li.get("cached")))
        if (li.get("c4_errors") or 0) > 0: guard.append("%s: %s c4 errors" % (r["name"], li.get("c4_errors")))
        if not md or "decode_steps" not in md: guard.append("%s: no /metrics deltas" % r["name"])
        rows.append(row)
    X = {x: [r for r in rows if r["x"] == x] for x in ("X1", "X2")}
    if not X["X1"] or not X["X2"]: guard.append("runs X1 %d, X2 %d" % (len(X["X1"]), len(X["X2"])))
    G = [("probe TTFT s", lambda r: r["longid"].get("ttft"), 2), ("c4 agg tok/s (prefill window)", lambda r: r["longid"].get("c4_pf_agg_tps"), 1),
         ("c4 per-stream p50", lambda r: r["longid"].get("c4_pf_stream_tps_p50"), 1), ("TPOT p50 ms", lambda r: r["longid"].get("tpot_pf_p50"), 1),
         ("TPOT p90 ms", lambda r: r["longid"].get("tpot_pf_p90"), 1), ("decode steps", lambda r: r["md"].get("decode_steps"), 0),
         ("ms a decode step", lambda r: r["md"].get("ms_decode_step"), 1), ("tok a decode step", lambda r: r["md"].get("tok_per_decode_step"), 2),
         ("drafted", lambda r: r["md"].get("draft_tok"), 0), ("accepted", lambda r: r["md"].get("draft_acc"), 0),
         ("non-decode steps", lambda r: r["md"].get("nondecode_steps"), 0), ("non-decode ms mean", lambda r: r["md"].get("nondecode_ms"), 1),
         ("probe chunks", lambda r: (r["chunks"] or {}).get("n"), 1), ("chunks off the grid", lambda r: (r["chunks"] or {}).get("off_grid"), 1),
         ("decode rows a chunk step", lambda r: (r["chunks"] or {}).get("dec_rows_mean"), 2),
         ("decode tokens a chunk step", lambda r: (r["chunks"] or {}).get("dec_tok_mean"), 2),
         ("chunk-step wall ms", lambda r: ((r["chunks"] or {}).get("wall_ms") or {}).get("mean"), 1)]
    def mean(x, f):
        v = [f(r) for r in X[x] if f(r) is not None]
        return statistics.mean(v) if v else None
    cells, agg = [], {}
    for nm, f, nd in G:
        m1, m2 = mean("X1", f), mean("X2", f)
        agg[nm] = {"X1": m1, "X2": m2, "pct": r3(100.0 * (m2 / m1 - 1), 1) if m1 and m2 is not None else None,
                   "runs": {r["name"]: f(r) for r in rows}}
        cells.append("%s %s -> %s (%s)" % (nm, "-" if m1 is None else ("%.*f" % (nd, m1)), "-" if m2 is None else ("%.*f" % (nd, m2)),
                                           "%+.1f %%" % agg[nm]["pct"] if agg[nm]["pct"] is not None else "-"))
    d1, d2 = mean("X1", lambda r: r["md"].get("draft_tok")), mean("X2", lambda r: r["md"].get("draft_tok"))
    lim = max(800.0, 0.1 * (d1 or 0.0))
    bad2 = [r["name"] for r in X["X2"] if (r["md"].get("draft_tok") or 0) > lim]
    if bad2: guard.append("depth 0 forced did not hold: %s drafted > %.0f" % (", ".join("%s %s" % (r["name"], r["md"].get("draft_tok")) for r in X["X2"] if r["name"] in bad2), lim))
    note = " (X1 drafted little, %.0f a run: a weak contrast)" % d1 if d1 is not None and d1 < 2000 else ""
    ident = []
    if X["X1"]:
        ref = X["X1"][0]
        def lab(r, q):
            c = textcmp(q["text"], r["text"])
            sb = (r["chunks"] or {}).get("starts"), (q["chunks"] or {}).get("starts")
            if "IDENTICAL" not in c and sb[0] and sb[1] and sb[0] != sb[1]: c += " -- the chunk boundaries differ (class (ii), reported)"
            elif "IDENTICAL" not in c: c += " -- reported, not hidden"
            return "%s vs %s: %s" % (r["name"], q["name"], c)
        for r in rows:
            if r is not ref: ident.append(lab(r, ref))
        if len(X["X2"]) > 1: ident.append(lab(X["X2"][1], X["X2"][0]))
    sizes = " / ".join("%s: %s" % (r["name"], ", ".join("%dx%d" % (k, v) for k, v in (r["chunks"] or {}).get("sizes") or [])) for r in rows)
    line_d = "(d) D7 under a long prefill (arm S, c4 + one cold ~64k deepq probe, ABBA %s; X1 draft_depth0 1 -> X2 2, means, X2 vs X1): %s | X2 drafted %s a run (limit %.0f)%s | chunk sizes (q x n): %s | identity: %s%s" % (
        "".join("1" if r["x"] == "X1" else "2" for r in rows), "; ".join(cells), "-" if d2 is None else "%.0f" % d2, lim, note, sizes,
        "; ".join(ident) or "-", (" -- VIRHE " + "; ".join(guard)) if guard else "")
    print("NCD " + line_d)
    out = {"rows": rows, "agg": agg, "identity": ident, "guard": guard, "line_d": line_d}
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, default=str)
    return 0 if not guard else 2


# ---------------------------------------------------------------- report
def cmd_report(a):
    J = {}
    for x in a.inp:
        k, p = x.split("=", 1)
        try: J[k] = json.load(open(p))
        except Exception: J[k] = None
    L = []
    pr = lambda s: L.append("RESULT xnc " + s)   # noqa: E731
    def alone_s(d):
        if not d: return "no reading"
        return "%s prompts, %s tok, step %s ms (/stats; p50 %s), TTFT p50 %s s; staged %s units = %s GB rank 0 (link staged %s GB all ranks, %s copies, %s layers), h2d %s GB, stream %s GB%s" % (
            d.get("ok"), (d.get("prompt") or {}).get("p50"), (d.get("step_ms") or {}).get("mean"), (d.get("step_ms") or {}).get("p50"),
            (d.get("ttft_s") or {}).get("p50"), (d.get("staged_units") or {}).get("mean"), (d.get("staged_gb_r0") or {}).get("mean"),
            (d.get("link_staged_gb") or {}).get("mean"), (d.get("stage_copies") or {}).get("mean"), (d.get("stage_layers") or {}).get("mean"),
            (d.get("h2d_gb") or {}).get("mean"), (d.get("link_stream_gb") or {}).get("mean"), " VIRHE" if d.get("guard") else "")
    def c8_s(d):
        if not d: return "no reading"
        s, r = d.get("stats") or {}, d.get("regression") or {}
        pn = s.get("per_nd") or {}
        return "a non-decode step (window / %s steps): staged %s units, %s GB rank 0, link staged %s GB, %s layers, h2d %s GB; 2 s regression (%s intervals): link staged %s GB, stream %s GB a non-decode step vs stream %s GB a decode step%s" % (
            (s.get("prefill") or 0) + (s.get("mixed") or 0), pn.get("staged_units"), pn.get("staged_gb_r0"), pn.get("link_staged_gb"), pn.get("stage_layers"),
            pn.get("h2d_gb"), r.get("intervals"), (r.get("link_staged_gb") or {}).get("per_nd"), (r.get("link_stream_gb") or {}).get("per_nd"),
            (r.get("link_stream_gb") or {}).get("per_decode"), " VIRHE" if d.get("guard") else "")
    prof = J.get("prof") or {}
    pw = prof.get("windows") or {}
    def waits(nm):
        w = pw.get(nm) or {}
        return ", ".join("r%s %.1f / %.1f ms" % (rk, v.get("stager_wait_ms") or 0.0, v.get("wait_ms") or 0.0) for rk, v in sorted((w.get("ranks") or {}).items())) or "-"
    miss = J.get("miss") or {}
    def missl(nm, ph):
        w = ((miss.get("windows") or {}).get(nm) or {}).get("per_kind_rank") or {}
        return ", ".join("r%s %s of %s (%s %%), host-pool %s (%s MiB), armed %s" % (k.split("|")[1], v["live_mean"], v["of_units"], v["live_pct"], v["host_mean"],
                         v["host_mib_mean"], v["staged_share"]) for k, v in sorted(w.items()) if k.split("|")[0] == ph) or "-"
    pr("(a) STAGING a ~1.1k newcomer step -- ALONE (unprofiled, START U): S %s | A %s" % (alone_s(J.get("ua_S")), alone_s(J.get("ua_A"))))
    pr("(a) STAGING at c8 (unprofiled, START U): S %s | A %s" % (c8_s(J.get("uc_S")), c8_s(J.get("uc_A"))))
    pr("(a) START P arm S (PROFILED, same counters): ALONE %s | W1 (8 streams decode only) %s | W2 (mb c8) %s" % (alone_s(J.get("pa")), c8_s(J.get("pw1")), c8_s(J.get("pw2"))))
    pr("(a) LANE WAITS a newcomer step (PROFILED, START P arm S; stager waits before each layer's first expert op / all lane waits, ms a step): ALONE %s | at c8 by difference %s | distinct experts touched (miss log, START P): ALONE prefill %s | at c8 MIXED %s | decode (W1) %s" % (
        waits("alone"), waits("nc_diff"), missl("alone", "prefill"), missl("w2", "MIXED"), missl("w1", "decode")))
    for arm in ("S", "A"):
        d = J.get("uc_" + arm) or {}
        c, s, n, m = d.get("classes") or {}, d.get("stats") or {}, d.get("newcomers") or {}, d.get("mb") or {}
        if not d: pr("(b) c8 arm %s: no reading" % arm); continue
        pr("(b) c8 arm %s (unprofiled, step log + /stats): newcomers %s: %s MIXED / %s prefill (%s sharing a step, %s in more than one chunk); decode rows riding a newcomer step p50 %s (in %s of %s); newcomer step wall p50 %s / mean %s ms vs decode step p50 %s / mean %s ms (/stats: non-decode %s ms, decode %s ms); arrival -> its step p50 %s steps / %s ms; TTFT p50 %s s | log: MIXED %s, prefill %s, decode %s; /stats mixed %s prefill %s held %s | decode_share %s, agg_dec_tps %s (stack4 c8 %s)%s" % (
            arm, n.get("n"), n.get("mixed"), n.get("prefill"), n.get("shared_step"), n.get("multi_chunk"), (n.get("rode") or {}).get("p50"), n.get("rode_any"), n.get("found"),
            (n.get("wall_ms") or {}).get("p50"), (n.get("wall_ms") or {}).get("mean"), (c.get("decode") or {}).get("wall_ms", {}).get("p50"),
            (c.get("decode") or {}).get("wall_ms", {}).get("mean"), s.get("nondecode_ms_mean"), s.get("decode_ms_mean"),
            (n.get("wait_steps") or {}).get("p50"), (n.get("wait_ms") or {}).get("p50"), (n.get("ttft_s") or {}).get("p50"),
            (c.get("MIXED") or {}).get("steps"), (c.get("prefill") or {}).get("steps"), (c.get("decode") or {}).get("steps"),
            s.get("mixed"), s.get("prefill"), s.get("held"), m.get("decode_share"), m.get("agg_dec_tps"), a.ref_c8, " -- VIRHE " + "; ".join(d.get("guard")) if d.get("guard") else ""))
    d2 = J.get("pw2") or {}
    if d2:
        n2, c2 = d2.get("newcomers") or {}, d2.get("classes") or {}
        pr("(b) second reading, START P W2 (PROFILED arm S, %s s window): newcomers %s: %s MIXED / %s prefill, decode rows riding p50 %s; log MIXED %s prefill %s decode %s%s" % (
            d2.get("window_s"), n2.get("n"), n2.get("mixed"), n2.get("prefill"), (n2.get("rode") or {}).get("p50"), (c2.get("MIXED") or {}).get("steps"),
            (c2.get("prefill") or {}).get("steps"), (c2.get("decode") or {}).get("steps"), " -- VIRHE " + "; ".join(d2.get("guard")) if d2.get("guard") else ""))
    for nm, what in (("alone", "ALONE newcomer step (PROFILED, START P arm S)"), ("nc_diff", "at-c8 newcomer step BY DIFFERENCE W2 - W2 decode steps x W1's decode step (PROFILED, an ESTIMATE)"),
                     ("nc_iv", "at-c8 newcomer step BY DIFFERENCE inside W2 (the dump intervals with one newcomer minus the W2 decode-interval mean; PROFILED, an ESTIMATE)")):
        w = pw.get(nm)
        if not w: pr("(c) %s: no reading" % what); continue
        if w.get("note"): pr("(c) %s note: %s" % (what, w.get("note")))
        for rk, v in sorted((w.get("ranks") or {}).items()):
            pr("(c) %s rank %s: %s" % (what, rk, v.get("ranked")))
        pr("(c) %s breakdowns (ms a step; dspark_ctx rows = the step's rows): %s" % (what, w.get("breakdown_line")))
        if w.get("rank0_detail"): pr("(c) %s rank 0 detail: stager waits by layer: %s | %s" % (what, w.get("stager_wait_layers"), w.get("rank0_detail")))
    if prof.get("guard"): pr("(c) profile guard: VIRHE " + "; ".join(prof["guard"]))
    ld = J.get("ld") or {}
    pr(ld.get("line_d") or "(d) D7 under a long prefill: no reading")
    sm = []
    for arm in ("S", "A"):
        d = J.get("uc_" + arm) or {}; n = d.get("newcomers") or {}
        sm.append("%s: %s of %s newcomer steps MIXED, %s ms vs decode %s ms" % (arm, n.get("mixed"), n.get("found"), (n.get("wall_ms") or {}).get("p50"),
                  ((d.get("classes") or {}).get("decode") or {}).get("wall_ms", {}).get("p50")))
    al = J.get("ua_S") or {}
    top = (pw.get("alone") or {}).get("top0") or "-"
    pr("summary: c8 newcomer steps %s | ALONE S newcomer step %s ms, staged %s GB rank 0 (%s link GB) | profiled ALONE rank 0 top: %s" % (
        " | ".join(sm), (al.get("step_ms") or {}).get("mean"), (al.get("staged_gb_r0") or {}).get("mean"), (al.get("link_staged_gb") or {}).get("mean"), top))
    for l in L: print(l)
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    import shutil, tempfile
    d = tempfile.mkdtemp(prefix="ncsteps-")
    try: return _selftest(d)
    finally: shutil.rmtree(d, ignore_errors=True)   # DRYRUN writes nothing that stays


def _selftest(d):
    def ts(t):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(t))) + ".%09dZ" % int(round((t - int(t)) * 1e9))
    T = 1794480000.0
    lines, t = [], T
    # 3 decode steps (40 ms), one MIXED newcomer step (600 ms, 7 decode rows + 1099), 2 decode, then an alone prefill later
    for K in range(1, 4):
        lines.append("%s I step %d decode n_seq=8 n_tok=8 " % (ts(t), K) + " ".join("  [%d] slot=%d q=1 ctx=%d" % (i, i, 500 + K) for i in range(8))); t += 0.040
    lines.append("%s I step 4 MIXED n_seq=8 n_tok=1106 " % ts(t) + " ".join("  [%d] slot=%d q=1 ctx=600" % (i, i) for i in range(7)) + "  [7] slot=7 q=1099 ctx=0"); t += 0.600
    for K in (5, 6):
        lines.append("%s I step %d decode n_seq=8 n_tok=8  [0] slot=0 q=1 ctx=700" % (ts(t), K)); t += 0.040
    lines.append("%s I unrelated line n_seq=nothing" % ts(t))
    ta = t + 5.0
    lines.append("%s I step 7 prefill n_seq=1 n_tok=1099  [0] slot=3 q=1099 ctx=0" % ts(ta + 0.01))
    log = os.path.join(d, "e.log.gz")
    with gzip.open(log, "wt") as f: f.write("\n".join(lines) + "\n")
    st = lambda n, nd, sb, ns: {"steps": n, "steps_prefill": 0, "steps_mixed": nd, "steps_decode": n - nd, "steps_held": 0, "step_ns": ns,   # noqa: E731
                                "step_ns_decode": ns - nd * 600_000_000, "experts": {"staged_units": 1000 * nd, "staged_bytes": sb, "stage_layers": 40 * nd,
                                "ranks": [{"rank": 0, "stream_bytes": 0, "miss": {"cold": 0}}]}, "link": {"staged_bytes": 4 * sb, "stream_bytes": 10**8 * n}}
    reqs = os.path.join(d, "r.jsonl")
    with open(reqs, "w") as f:
        f.write(json.dumps({"tag": "c", "ev": "mark", "k": 0, "t": T - 0.001}) + "\n")
        f.write(json.dumps({"tag": "c", "ev": "send", "t": T + 0.10, "w": 7}) + "\n")
        f.write(json.dumps({"tag": "c", "ev": "done", "t_send": T + 0.10, "t0": T + 0.10, "ttft": 0.62, "w": 7, "prompt": 1099, "cached": 0, "n": 512, "err": None}) + "\n")
        f.write(json.dumps({"tag": "c", "ev": "mark", "k": 1, "t": T + 0.85}) + "\n")
    sa, sb = os.path.join(d, "a.json"), os.path.join(d, "b.json")
    json.dump(st(0, 0, 0, 0), open(sa, "w")); json.dump(st(6, 1, 9.6e9, 760_000_000), open(sb, "w"))
    samp = os.path.join(d, "s.jsonl")
    with open(samp, "w") as f:
        for i, (n, nd) in enumerate([(0, 0), (50, 0), (100, 1), (150, 1), (200, 3), (250, 3)]):
            f.write(json.dumps({"t": T + i, "s": load_compact()(st(n, nd, 9.6e9 * nd, 0))}) + "\n")
    mbf = os.path.join(d, "mb.txt")
    open(mbf, "w").write('MB {"agg_dec_tps": 172.2, "decode_share": 0.784}\n')
    class A: pass
    a = A(); a.log, a.reqs, a.tag, a.sa, a.sb, a.samples, a.mbline, a.skip_mod, a.idle, a.label, a.json = log, reqs, "c", sa, sb, samp, mbf, 0, 3.0, "t", os.path.join(d, "c.json")
    rc = cmd_c8(a)
    c = json.load(open(a.json))
    n = c["newcomers"]
    ok = rc == 0 and c["classes"]["MIXED"]["steps"] == 1 and c["classes"]["decode"]["steps"] == 5 and abs(c["classes"]["MIXED"]["wall_ms"]["p50"] - 600) < 1 \
        and n["mixed"] == 1 and n["rode"]["p50"] == 7 and n["wait_steps"]["p50"] == 0 and n["multi_chunk"] == 0 and n["rows"][0]["slot"] == 7 and c["stats"]["per_nd"]["staged_units"] == 1000 \
        and abs(c["regression"]["link_staged_gb"]["per_nd"] - 38.4) < 0.01 and c["mb"]["agg_dec_tps"] == 172.2 and c["stats"]["nondecode_ms_mean"] == 600.0
    # alone
    reqa, sta = os.path.join(d, "ra.jsonl"), os.path.join(d, "sa.jsonl")
    with open(reqa, "w") as f:
        f.write(json.dumps({"tag": "u", "j": 0, "t0": ta, "ttft": 0.65, "t_resp": 0.66, "t_end": ta + 0.66, "prompt": 1099, "cached": 0, "n": 1, "err": None}) + "\n")
    s0, s1 = load_compact()(st(6, 1, 9.6e9, 760_000_000)), load_compact()(st(7, 2, 19.2e9, 1_360_000_000))
    s1["steps_prefill"], s1["steps_mixed"] = 1, 1
    with open(sta, "w") as f:
        f.write(json.dumps({"tag": "u", "k": 0, "t": ta - 0.5, "s": s0}) + "\n"); f.write(json.dumps({"tag": "u", "k": 1, "t": ta + 1.1, "s": s1}) + "\n")
    b = A(); b.log, b.reqs, b.stats, b.tag, b.skip_mod, b.idle, b.label, b.json = log, reqa, sta, "u", 8, 3.0, "u", os.path.join(d, "u.json")
    rc2 = cmd_alone(b)
    u = json.load(open(b.json))
    ok = ok and rc2 == 0 and u["step_ms"]["mean"] == 600.0 and u["staged_units"]["mean"] == 1000 and u["rows"][0]["log_steps"] == [[7, "prefill", 1099]]
    # miss
    ml = os.path.join(d, "m.log")
    with open(ml, "w") as f:
        f.write("# radiance-miss-log/1\n# model m arch deepseek41 tp 4 routed_layers 43 trunk_layers 40 experts 96 started_unix 1 pid 2\n# rank 0 expert_units 3936 at_start vram 2900 host 1036 other 0 unit_bytes_mean 9700000\n")
        f.write("4 mixed 0 1106 8 2 1 0 3800 2800 1000 0 -1 9500000 0 0:95.25,95.25\n4 mixed 0 1106 8 2 1 0 40 30 10 0 -1 95000 0 2:30.10\n4 draft 0 8 8 1 0 0 30 30 0 0 0 0 0 40:10\n")
    m = A(); m.file, m.log, m.win, m.idle, m.json = ml, log, ["w2:%f:%f" % (T - 1, T + 1)], 3.0, os.path.join(d, "m.json")
    rc3 = cmd_miss(m)
    mm = json.load(open(m.json))["windows"]["w2"]["per_kind_rank"]["MIXED|0"]
    ok = ok and rc3 == 0 and mm["live_mean"] == 3840 and mm["host_mean"] == 1010 and mm["staged_share"] == 1.0
    # (d): two runs (X1, X2), a 4100-token probe in chunks 2048 / 2048 / 4 beside decode rows
    T2 = T + 100.0
    ll, runs = [], os.path.join(d, "runs.jsonl")
    for k, (x, base, dt) in enumerate((("X1", T2, 6), ("X2", T2 + 50, 1))):
        tt, K0 = base, 100 + 10 * k
        plan = [("MIXED", [(0, 1 + dt - 1 if dt > 1 else 1, 500), (1, dt, 600), (3, 2048, 0)]), ("decode", [(0, 1, 501), (1, 1, 601)]),
                ("MIXED", [(0, dt, 502), (1, 1, 602), (3, 2048, 2048)]), ("MIXED", [(0, 1, 503), (1, 1, 603), (3, 4, 4096)]),
                ("decode", [(0, 1, 504), (1, 1, 604), (3, 1, 4100)])]
        for i, (ph, ents) in enumerate(plan):
            ll.append("%s I step %d %s n_seq=%d n_tok=%d " % (ts(tt), K0 + i, ph, len(ents), sum(q for _, q, _ in ents)) +
                      " ".join("  [%d] slot=%d q=%d ctx=%d" % (j, sl, q, c) for j, (sl, q, c) in enumerate(ents)))
            tt += 0.5 if ph != "decode" else 0.04
        txt = os.path.join(d, "%s.txt" % x); open(txt, "w").write("the same answer")
        with open(runs, "a") as f:
            f.write(json.dumps({"name": "d-%d%s" % (k + 1, x), "x": x, "t0": base - 0.01, "t1": tt + 0.01, "text": txt,
                                "longid": {"status": 200, "prompt_tokens": 4100, "cached": 0, "ttft": 40.0 if x == "X1" else 30.0,
                                           "c4_pf_agg_tps": 25.0, "c4_pf_stream_tps_p50": 6.0, "tpot_pf_p50": 50.0, "tpot_pf_p90": 800.0, "c4_errors": 0},
                                "md": {"decode_steps": 900, "ms_decode_step": 55.0 if x == "X1" else 30.0, "tok_per_decode_step": 9.0,
                                       "draft_tok": 6000 if x == "X1" else 120, "draft_acc": 4000 if x == "X1" else 80,
                                       "nondecode_steps": 33, "nondecode_ms": 900.0}}) + "\n")
    ldl = os.path.join(d, "ld.log.gz")
    with gzip.open(ldl, "wt") as f: f.write("\n".join(ll) + "\n")
    q = A(); q.log, q.runs, q.pskld, q.grid, q.idle, q.json = ldl, runs, os.path.join(HERE, "..", "stack4", "pskld.py"), 2048, 3.0, os.path.join(d, "ld.json")
    rc4 = cmd_ld(q)
    lj = json.load(open(q.json))
    ok = ok and rc4 == 0 and lj["agg"]["probe TTFT s"]["pct"] == -25.0 and lj["rows"][0]["chunks"]["n"] == 3 and lj["rows"][0]["chunks"]["off_grid"] == 0 \
        and lj["rows"][0]["chunks"]["mixed"] == 3 and lj["rows"][0]["chunks"]["dec_tok_mean"] > lj["rows"][1]["chunks"]["dec_tok_mean"] and "IDENTICAL" in lj["identity"][0]
    # report renders
    r = A(); r.inp, r.ref_c8 = ["uc_S=%s" % a.json, "ua_S=%s" % b.json, "miss=%s" % m.json, "ld=%s" % q.json], "A 170.1, S 173.4"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): cmd_report(r)
    ok = ok and buf.getvalue().count("RESULT xnc ") >= 7 and "1 MIXED / 0 prefill" in buf.getvalue() and "RESULT xnc (d) D7 under a long prefill" in buf.getvalue()
    print("SELFTEST " + ("OK: c8 (classes, newcomer MIXED with 7 decode rows riding, /stats per non-decode step, regression), alone, miss, ld (chunks via pskld.py, X2 vs X1, identity, guard), report" if ok
                         else "FAILED: c8 rc %s %s | alone rc %s %s | miss rc %s %s | ld rc %s %s" % (rc, json.dumps(c)[:600], rc2, json.dumps(u)[:300], rc3, mm, rc4, json.dumps(lj)[:800])))
    return 0 if ok else 1


def main():
    import argparse
    if len(sys.argv) > 1 and sys.argv[1] == "selftest": return selftest()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("alone"); p.add_argument("--log", required=True); p.add_argument("--reqs", required=True); p.add_argument("--stats", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--skip-mod", type=int, default=0); p.add_argument("--idle", type=float, default=3.0)
    p.add_argument("--label", default=""); p.add_argument("--json")
    p = sub.add_parser("c8"); p.add_argument("--log", required=True); p.add_argument("--reqs", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--sa", required=True); p.add_argument("--sb", required=True); p.add_argument("--samples"); p.add_argument("--mbline")
    p.add_argument("--skip-mod", type=int, default=0); p.add_argument("--idle", type=float, default=3.0); p.add_argument("--label", default=""); p.add_argument("--json")
    p = sub.add_parser("miss"); p.add_argument("--file", required=True); p.add_argument("--log", required=True); p.add_argument("--win", action="append", default=[])
    p.add_argument("--idle", type=float, default=3.0); p.add_argument("--json")
    p = sub.add_parser("ld"); p.add_argument("--log", required=True); p.add_argument("--runs", required=True); p.add_argument("--pskld", required=True)
    p.add_argument("--grid", type=int, default=2048); p.add_argument("--idle", type=float, default=3.0); p.add_argument("--json")
    p = sub.add_parser("report"); p.add_argument("--in", dest="inp", action="append", default=[]); p.add_argument("--ref-c8", default="172.2")
    a = ap.parse_args()
    return {"alone": cmd_alone, "c8": cmd_c8, "miss": cmd_miss, "ld": cmd_ld, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
