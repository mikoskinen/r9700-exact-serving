#!/usr/bin/env python3
"""ccsum.py (12.11.2026, queue-ar/dsmain/cc, ds-main-port cc builder) -- the /stats sampler and the readings of dsm-cc.sh
(the EXISTING Claude Code simulation, cc_sim3.py natural mode, under a bf16 vs an nvfp4 compressed latent at MAXSEQS 8; tag
"RESULT xcc "). NOT a load generator: the load is queue-ar/tools/ccsim/cc_sim3.py run through ccsim-exl3-turns.sh; this file
only samples /stats and reads what the run wrote. cc_sim3.py is loaded READ ONLY (importlib) for its summarize_turns / pct
and, offline, its session builder (the trace check); ncsteps.py beside this file is an UNCHANGED copy of nc/ncsteps.py
(0a6e0ed2: read_steps = the step log of RADIANCE_LOG_STEPS=1, lsq2 = the least-squares split). Standard library only;
reads .gz or plain files.

  ccsum.py sample --url U --every S --out F.jsonl [--max-s N]
      GET /stats every S s (compact: kv pool tokens used / cached / total, the step counters, experts per rank: vram units,
      pinned, kv-lent / arena-lent held, stream bytes; link stream / staged / h2d bytes; psk / pfctx / engdd counters) until
      SIGTERM or N s. One json line a sample ({"t": unix, ...}); a failed GET writes {"t", "err"}.
  ccsum.py trace --ccsim cc_sim3.py --seed N --users U [--minctx 50000] [--ref RUN.jsonl]
      OFFLINE (no server, writes nothing): cc_sim3's own session builder (Sim.new_session, default context settings, preage on,
      109 tools, cpt 3.5) for each user's FIRST session -> its prompt estimate (cc_sim's chars / 3.5); with --ref (an earlier
      run of the same seed on this model) the server-counted first prompts beside them. Prints "TRACE ..." (OK when the
      longest first prompt >= minctx).
  ccsum.py run --cc CC.jsonl --samples S.jsonl --log ENGINE.log.gz --tag T --lat L --ccsim cc_sim3.py [--minctx 50000]
               [--nst 5] --json OUT
      one cc run: cc_sim3.summarize_turns over the kept turns + TTFT p50 / p90, decode tok/s per stream (p50 / p90) and
      aggregate, completed work per minute (requests, generated / uncached-prompt tokens; kept window and whole run), errors,
      the context distribution (guard: max prompt >= minctx, else VIRHE); from the samples the KV pool fill % over time,
      resident expert units a rank, kv-lent GiB a rank (unit = experts.bytes / n_units), the streamed GB a non-decode step
      (least squares over the sample intervals: stream = a x non-decode steps + b x decode steps) and per newcomer; from the
      step log the steps by class and each newcomer's prefill run (a slot's consecutive entries with q > 1 + nst). Prints
      "RESULT xcc ..." lines.
  ccsum.py report --dir O --runs "b8-1 n8-1 ..." [--order-note TEXT]
      the item's summary: one table line a run, the pairs (adjacent runs of different latent; with two runs a mode, the mode
      means and both pairs), guards. Prints "RESULT xcc ..." lines.
  ccsum.py selftest
"""
import argparse, gzip, importlib.util, json, math, os, signal, statistics, sys, time, types, urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1024.0 ** 3


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ncs():
    return load(os.path.join(HERE, "ncsteps.py"), "ncsteps")


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def jl(path):
    out = []
    try:
        with opn(path) as f:
            for l in f:
                l = l.strip()
                if not l: continue
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError:
        pass
    return out


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v: return None
    k = (len(v) - 1) * p / 100.0
    f = int(math.floor(k)); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def rnd(x, n=1):
    return None if x is None else round(x, n)


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def pc(a, b):
    """a vs b in %"""
    return None if a is None or b is None or not b else round(100.0 * (a - b) / b, 1)


def gv(d, *path):
    """a number at d[path...] (0 when absent)"""
    x = d
    for k in path:
        x = x.get(k) if isinstance(x, dict) else None
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


# ---------------------------------------------------------------- the sampler
def compact(st, t):
    e, ln = st.get("experts") or {}, st.get("link") or {}
    keep = ("running", "waiting", "in_flight", "preempted", "preemptions", "total_requests", "kv_util", "kv_tokens_used",
            "kv_tokens_total", "kv_tokens_cached", "kv_bytes_used", "kv_bytes_total", "kv_bytes_carved", "steps", "steps_prefill",
            "steps_mixed", "steps_decode", "steps_held", "step_ns", "step_ns_decode", "prefix_hit_rate", "prefix_evictions",
            "decode_tps", "draft_accept", "checkpoints_written")
    out = {"t": t}
    out.update({k: st.get(k) for k in keep if k in st})
    ex = {k: e.get(k) for k in ("n_units", "bytes", "staged_units", "staged_bytes", "stage_copies", "stage_layers", "promotions",
                                "demotions", "routed", "routed_res", "starved_slab", "starved_pool", "refused") if k in e}
    ex["tiers"] = {x.get("name"): x.get("units") for x in e.get("tiers") or [] if isinstance(x, dict)}
    rk = []
    for r in e.get("ranks") or []:
        lent = r.get("lent") or {}
        rk.append({"rank": r.get("rank"), "vram": r.get("vram"), "pinned": r.get("pinned"), "trunk_vram": r.get("trunk_vram"),
                   "draft_vram": r.get("draft_vram"), "stream_bytes": r.get("stream_bytes"),
                   "kv_held": (lent.get("kv") or {}).get("held"), "kv_covered": (lent.get("kv") or {}).get("covered"),
                   "arena_held": (lent.get("arena") or {}).get("held"), "dslab_held": (lent.get("dslab") or {}).get("held")})
    ex["ranks"] = rk
    out["experts"] = ex
    out["link"] = {k: ln.get(k) for k in ("h2d_bytes", "staged_bytes", "stream_bytes", "ar_bytes") if k in ln}
    p = st.get("psk") or {}
    out["psk"] = {k: p.get(k) for k in ("latched", "tail_only_latched", "cut_passes", "plan_mismatch", "cr_hits", "cr_passes",
                                         "cr_fallbacks", "cr_not_needed", "l20_cut_passes", "fallback_resumes") if k in p}
    out["lev"] = {"pfctx_passes": (st.get("dspark_pfctx") or {}).get("passes"), "engdd_passes": (st.get("engdd") or {}).get("passes")}
    return out


def cmd_sample(a):
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *x: stop.__setitem__(0, True))
    t_end = time.time() + a.max_s; n = bad = 0
    with open(a.out, "a") as f:
        while not stop[0] and time.time() < t_end:
            t0 = time.time()
            try:
                with urllib.request.urlopen(a.url.rstrip("/") + "/stats", timeout=5) as r:
                    st = json.load(r)
                f.write(json.dumps(compact(st, round(t0, 3))) + "\n"); n += 1
            except Exception as e:  # noqa: BLE001 -- a failed GET is a gap in the timeline, not a stop
                f.write(json.dumps({"t": round(t0, 3), "err": str(e)[:160]}) + "\n"); bad += 1
            f.flush()
            while not stop[0] and time.time() < t0 + a.every:
                time.sleep(0.2)
    print("SAMPLE %d samples, %d failed GETs -> %s" % (n, bad, a.out))
    return 0


# ---------------------------------------------------------------- the trace (offline)
TRACE_DEFAULTS = dict(route="hash", users=8, duration=600.0, warmup=120.0, drain=180.0, ramp=20.0, seed=1, run_id="20261112-000000",
                      ctx_cap=170000, start_min=20000, start_max=40000, preage=True, shared_prefix=0, tools=109, gen_median=350.0,
                      gen_sigma=1.0, gen_max=13000, gen_mode="natural", ref=None, reflen_eos="stop", tool_median=300.0, tool_sigma=1.2,
                      tool_min=20, tool_max=8000, think_median=0.8, think_sigma=1.0, think_min=0.2, think_max=90.0,
                      effort="xhigh,medium", drop_reasoning=False, temperature=0.0, cpt=3.5, model="", key_env="", api_key="",
                      read_timeout=900.0, progress=60.0, tag="", out="", summarize="", turns=0, warmup_turns=0, cpt_fixed=True,
                      pair=None, trace_seeded=True, record=False, replay="", replay_closed=False, pair_replay=None)


def first_sessions(ccsim, seed, users, **over):
    """each user's first session as cc_sim3 builds it (offline): [(user, prompt estimate, preaged turns)]"""
    cc = load(ccsim, "cc_sim3_ro")
    d = dict(TRACE_DEFAULTS); d.update(seed=seed, users=users); d.update(over)
    a = types.SimpleNamespace(**d)
    cc.load_corpus()
    tools = cc.build_tools(a.tools) if a.tools > 0 else []
    sim = cc.Sim(a, ["http://127.0.0.1:1"], "m", "", tools, os.devnull)
    out = []
    for u in range(users):
        s = sim.new_session(u, 0)
        out.append((u, sim.est(s["chars"] + sim.tools_chars), s["preaged"]))
    return out


def cmd_trace(a):
    try:
        fs = first_sessions(a.ccsim, a.seed, a.users)
    except Exception as e:  # noqa: BLE001
        print("TRACE VIRHE cc_sim3's session builder did not run offline (%s: %s)" % (type(e).__name__, str(e)[:200])); return 2
    ref = {}
    if a.ref:
        for r in sorted((r for r in jl(a.ref) if "turn" in r and r.get("prompt")), key=lambda r: r["t"]):
            ref.setdefault(r["user"], r["prompt"])
    est = [p for _, p, _ in fs]
    ok = max(est) >= a.minctx
    print("TRACE %s seed %d, %d users, first sessions (cc_sim3 preage: U(start 20-40k, 0.95 x cap 170000), estimate at 3.5 chars a token)"
          ": %s | p50 %s, max %s, >= %d: %d of %d%s" % (
              "OK" if ok else "VIRHE", a.seed, a.users,
              ", ".join("u%d %.1fk%s" % (u, p / 1000.0, (" (this model %.1fk)" % (ref[u] / 1000.0)) if u in ref else "") for u, p, _ in fs),
              int(pct(est, 50)), max(est), a.minctx, sum(1 for p in est if p >= a.minctx), len(est),
              (" | the server-counted first prompts of %s: %s" % (os.path.basename(a.ref), [ref.get(u) for u in range(a.users)])) if ref else ""))
    return 0 if ok else 2


# ---------------------------------------------------------------- one run
def read_cc(path):
    cfg, recs = None, []
    for d in jl(path):
        if "config" in d: cfg = d["config"]
        elif "t" in d and "url" in d: recs.append(d)
    return cfg, recs


def prefill_runs(steps, nst):
    """each newcomer's prefill run on its slot: the slot's consecutive entries with q > 1 + nst, closed by the slot's first
    decode entry (q <= 1 + nst). [{t0, t1, steps, tokens, wall_ms, phases, ctx0}]"""
    lim, open_, done = 1 + nst, {}, []
    for s in steps:
        seen = set()
        for slot, q, ctx in s["ents"]:
            seen.add(slot)
            if q > lim:
                r = open_.get(slot)
                if r is None:
                    r = open_[slot] = {"t0": s["t"], "steps": 0, "tokens": 0, "wall_ms": 0.0, "phases": "", "ctx0": ctx, "nowall": 0}
                r["steps"] += 1; r["tokens"] += q; r["phases"] += s["ph"][0]; r["t1"] = s["t"]
                if s.get("wall_ms") is not None: r["wall_ms"] += s["wall_ms"]
                else: r["nowall"] += 1
            elif slot in open_:
                done.append(open_.pop(slot))
    return done


def cmd_run(a):
    out, guard = [], []
    cfg, recs = read_cc(a.cc)
    if not cfg or not recs:
        print("RESULT xcc %s cc: VIRHE no cc_sim records in %s" % (a.tag, a.cc)); return 2
    cc = load(a.ccsim, "cc_sim3_ro")
    wt = cfg.get("warmup_turns") or 0
    s = cc.summarize_turns(recs, wt, types.SimpleNamespace(users=cfg.get("users"), turns=cfg.get("turns")))
    t_zero = time.mktime(time.strptime(cfg["start"], "%Y-%m-%d %H:%M:%S"))
    good = lambda r: not r.get("err") and not r.get("aborted") and r.get("ttft") is not None   # noqa: E731
    kept = [r for r in recs if r.get("uturn", 0) > wt]
    kok = [r for r in kept if good(r)]
    allok = [r for r in recs if good(r)]
    w0, w1 = (s["window_s"] if kept else [0.0, 0.0])
    span = max(1e-9, w1 - w0)
    mk = max(r["t"] + (r.get("total") or 0) for r in recs)
    tt = [r["ttft"] for r in kok]; dec = [r["decode"] for r in kok if r.get("decode")]
    unc = lambda r: (r.get("prompt") or 0) - (r.get("cached") or 0)   # noqa: E731
    gen_k = sum(r.get("gen") or 0 for r in kok); unc_k = sum(unc(r) for r in kok)
    gen_a = sum(r.get("gen") or 0 for r in allok); unc_a = sum(unc(r) for r in allok)
    per_user = {}
    for r in recs: per_user.setdefault(r["user"], []).append(r)
    turns_done = {u: sum(1 for r in v if good(r)) for u, v in per_user.items()}
    want = cfg.get("turns") or 0
    short = sorted(u for u in range(cfg.get("users") or 0) if turns_done.get(u, 0) < want)
    errs = [r for r in recs if r.get("err")]; abo = [r for r in recs if r.get("aborted")]
    first = sorted((r for r in recs if r.get("uturn") == 1), key=lambda r: r["user"])
    P = [r.get("prompt") for r in recs if r.get("prompt")]
    pmax = max(P or [0])
    if pmax < a.minctx: guard.append("the trace did not reach a long context: max prompt %d < %d" % (pmax, a.minctx))
    if not kok: guard.append("no kept request answered")
    R = {"tag": a.tag, "lat": a.lat, "cfg": {k: cfg.get(k) for k in ("users", "turns", "warmup_turns", "seed", "gen_mode", "temperature",
                                                                        "trace_seeded", "cpt_fixed", "effort", "ctx_cap", "duration", "run_id", "start")},
         "summary": s, "kept_n": len(kept), "kept_ok": len(kok), "all_n": len(recs), "all_ok": len(allok),
         # percentiles as cc_sim3's own (nearest rank), so p50 matches the wrapper's "RESULT cc_sim" line
         "ttft_p50": rnd(cc.pct(tt, 50), 2) if tt else None, "ttft_p90": rnd(cc.pct(tt, 90), 2) if tt else None, "ttft_max": rnd(max(tt or [0]), 2),
         "dec_p50": rnd(cc.pct(dec, 50)) if dec else None, "dec_p90": rnd(cc.pct(dec, 90)) if dec else None,
         "dec_p10": rnd(cc.pct(dec, 10)) if dec else None, "dec_mean": rnd(mean(dec)),
         "agg_tps": s.get("agg_gen_tps"), "inflight": s.get("mean_inflight"),
         "kept_window_s": [rnd(w0), rnd(w1)], "makespan_s": rnd(mk),
         "work_kept": {"req_min": rnd(len(kok) / span * 60, 2), "gen_tok_min": rnd(gen_k / span * 60, 0), "unc_tok_min": rnd(unc_k / span * 60, 0)},
         "work_all": {"req_min": rnd(len(allok) / max(1e-9, mk) * 60, 2), "gen_tok_min": rnd(gen_a / max(1e-9, mk) * 60, 0),
                      "unc_tok_min": rnd(unc_a / max(1e-9, mk) * 60, 0), "requests": len(allok), "gen_tok": gen_a, "unc_tok": unc_a},
         "errors": len(errs), "aborted": len(abo), "first_err": (errs[0].get("err") if errs else ""),
         "incomplete_users": short, "turns_done": turns_done,
         "first_turns": {"prompt": [r.get("prompt") for r in first], "ttft": [r.get("ttft") for r in first]},
         "ctx": {"p10": rnd(pct(P, 10), 0), "p50": rnd(pct(P, 50), 0), "p90": rnd(pct(P, 90), 0), "max": pmax,
                 "ge50k": sum(1 for p in P if p >= 50000), "ge100k": sum(1 for p in P if p >= 100000), "n": len(P)},
         "draft_acc": s.get("draft_acc_mean"), "cache_hit_pct": s.get("cache_hit_pct"), "gen_p50": s.get("gen_p50"), "gen_mean": s.get("gen_mean")}
    # ---- the samples
    sm = sorted((d for d in jl(a.samples) if "err" not in d and t_zero - 15 <= d["t"] <= t_zero + mk + 15), key=lambda d: d["t"])
    nerr = sum(1 for d in jl(a.samples) if "err" in d)
    ka, kb = t_zero + w0, t_zero + w1
    ks = [d for d in sm if ka <= d["t"] <= kb]
    if len(sm) < 3: guard.append("only %d /stats samples" % len(sm))
    fill = lambda d: 100.0 * (d.get("kv_tokens_used") or 0) / d["kv_tokens_total"] if d.get("kv_tokens_total") else None   # noqa: E731
    cfill = lambda d: 100.0 * (d.get("kv_tokens_cached") or 0) / d["kv_tokens_total"] if d.get("kv_tokens_total") else None   # noqa: E731
    ub = None
    for d in sm:
        e = d.get("experts") or {}
        if e.get("bytes") and e.get("n_units"): ub = e["bytes"] / e["n_units"]; break
    ranks = sorted({r["rank"] for d in sm for r in (d.get("experts") or {}).get("ranks") or []})
    rv = lambda d, k: [next((x.get(k) for x in (d.get("experts") or {}).get("ranks") or [] if x["rank"] == r), None) for r in ranks]   # noqa: E731
    def series(dd, f):
        return [f(d) for d in dd]
    def five(v, nd=1):
        v2 = [x for x in v if x is not None]
        return {"start": rnd(v[0], nd) if v else None, "min": rnd(min(v2), nd) if v2 else None, "p50": rnd(pct(v2, 50), nd),
                "mean": rnd(mean(v2), nd), "max": rnd(max(v2), nd) if v2 else None, "end": rnd(v[-1], nd) if v else None}
    S = {"n": len(sm), "kept_n": len(ks), "failed_gets": nerr, "unit_bytes": ub}
    if sm:
        S["fill_used_pct"] = {"all": five(series(sm, fill)), "kept": five(series(ks, fill))}
        S["fill_cached_pct"] = {"all": five(series(sm, cfill)), "kept": five(series(ks, cfill))}
        S["kv_tokens_total"] = sm[0].get("kv_tokens_total")
        S["kv_used_max_tok"] = max((d.get("kv_tokens_used") or 0) for d in sm)
        S["vram_units"] = {"rank%d" % r: {"all": five([v[i] for v in series(sm, lambda d: rv(d, "vram"))], 0),
                                          "kept": five([v[i] for v in series(ks, lambda d: rv(d, "vram"))], 0)} for i, r in enumerate(ranks)}
        S["vram_units_mean_ranks"] = {"all": five([mean(v) for v in series(sm, lambda d: rv(d, "vram"))], 1),
                                      "kept": five([mean(v) for v in series(ks, lambda d: rv(d, "vram"))], 1)}
        if ub:
            gl = lambda d: [None if x is None else x * ub / GIB for x in rv(d, "kv_held")]   # noqa: E731
            S["kv_lent_gib"] = {"rank%d" % r: {"all": five([v[i] for v in series(sm, gl)], 2), "kept": five([v[i] for v in series(ks, gl)], 2)}
                                for i, r in enumerate(ranks)}
            S["kv_lent_gib_mean_ranks"] = {"all": five([mean(v) for v in series(sm, gl)], 2), "kept": five([mean(v) for v in series(ks, gl)], 2)}
            ga = lambda d: [None if x is None else x * ub / GIB for x in rv(d, "arena_held")]   # noqa: E731
            S["arena_lent_gib_mean_ranks"] = {"all": five([mean(v) for v in series(sm, ga)], 2)}
        # the timeline: one row a minute since the run's start
        tl = {}
        for d in sm:
            m = int((d["t"] - t_zero) // 60)
            tl.setdefault(m, []).append(d)
        S["timeline"] = [{"min": m, "fill_used_pct": rnd(mean([fill(d) for d in v]), 1), "fill_cached_pct": rnd(mean([cfill(d) for d in v]), 1),
                          "vram_units_mean": rnd(mean([mean(rv(d, "vram")) for d in v]), 0),
                          "kv_lent_gib_mean": rnd(mean([mean([x * ub / GIB for x in rv(d, "kv_held") if x is not None]) for d in v]), 2) if ub else None,
                          "running": rnd(mean([d.get("running") for d in v]), 1), "waiting": rnd(mean([d.get("waiting") for d in v]), 1)}
                         for m, v in sorted(tl.items())]
        # the streamed GB a non-decode step: least squares over the kept window's sample intervals (all if too few)
        N = ncs()
        use = ks if len(ks) >= 6 else sm
        iv = []
        g = gv
        for p, q in zip(use, use[1:]):
            nd = (g(q, "steps_prefill") + g(q, "steps_mixed")) - (g(p, "steps_prefill") + g(p, "steps_mixed"))
            dc = g(q, "steps_decode") - g(p, "steps_decode")
            iv.append((nd, dc, (g(q, "link", "stream_bytes") - g(p, "link", "stream_bytes")) / 1e9,
                       (g(q, "link", "staged_bytes") - g(p, "link", "staged_bytes")) / 1e9))
        aS, bS = N.lsq2([(x, y, z) for x, y, z, _ in iv]); aT, bT = N.lsq2([(x, y, w) for x, y, _, w in iv])
        A0, B0 = use[0], use[-1]
        nd_w = (g(B0, "steps_prefill") + g(B0, "steps_mixed")) - (g(A0, "steps_prefill") + g(A0, "steps_mixed"))
        dc_w = g(B0, "steps_decode") - g(A0, "steps_decode")
        st_w = (g(B0, "link", "stream_bytes") - g(A0, "link", "stream_bytes")) / 1e9
        S["stream"] = {"basis": "kept window" if use is ks else "whole run (too few kept samples)", "intervals": len(iv),
                       "nd_steps": nd_w, "decode_steps": dc_w, "stream_gb": rnd(st_w, 1),
                       "gb_per_nd_step_lsq": rnd(aS, 3), "gb_per_decode_step_lsq": rnd(bS, 4),
                       "staged_gb_per_nd_step_lsq": rnd(aT, 3), "gb_per_nd_step_ratio": rnd(st_w / nd_w, 3) if nd_w else None,
                       "rank_stream_gb": [rnd(((next((x.get("stream_bytes") for x in B0["experts"]["ranks"] if x["rank"] == r), 0) or 0)
                                                - (next((x.get("stream_bytes") for x in A0["experts"]["ranks"] if x["rank"] == r), 0) or 0)) / 1e9, 1) for r in ranks]}
        S["preemptions"] = (sm[-1].get("preemptions") or 0) - (sm[0].get("preemptions") or 0)
        S["levers"] = {k: (g(sm[-1], *k.split(".")) - g(sm[0], *k.split("."))) for k in (
            "psk.latched", "psk.tail_only_latched", "psk.cut_passes", "psk.plan_mismatch", "psk.cr_hits", "psk.cr_passes", "psk.cr_fallbacks",
            "psk.cr_not_needed", "psk.l20_cut_passes", "lev.pfctx_passes", "lev.engdd_passes")}
    else:
        guard.append("no /stats sample inside the run")
    R["samples"] = S
    # ---- the step log
    L = {}
    try:
        N = ncs()
        steps = N.read_steps(a.log, a.idle)
        W = [x for x in steps if ka <= x["t"] <= kb]
        if not W: guard.append("no step-log line in the kept window")
        for ph in ("decode", "MIXED", "prefill"):
            ss = [x for x in W if x["ph"] == ph]
            L[ph] = {"steps": len(ss), "wall_ms_p50": rnd(pct([x.get("wall_ms") for x in ss], 50)), "wall_ms_mean": rnd(mean([x.get("wall_ms") for x in ss])),
                     "n_tok_p50": rnd(pct([x["n_tok"] for x in ss], 50), 0)}
        drows = [sum(1 for sl, q, c in x["ents"] if q <= 1 + a.nst and c > 0) for x in W if x["ph"] == "decode"]
        L["decode_seqs_p50"] = rnd(pct([x["n_seq"] for x in W if x["ph"] == "decode"], 50), 1)
        L["decode_rows_p50"] = rnd(pct([x["n_tok"] for x in W if x["ph"] == "decode"], 50), 0)
        L["decode_entries_p50"] = rnd(pct(drows, 50), 1)
        runs = [r for r in prefill_runs(steps, a.nst) if ka <= r["t0"] <= kb]
        L["newcomers"] = {"n": len(runs), "steps_p50": rnd(pct([r["steps"] for r in runs], 50), 1), "steps_mean": rnd(mean([r["steps"] for r in runs]), 2),
                          "steps_max": max([r["steps"] for r in runs] or [0]), "tokens_p50": rnd(pct([r["tokens"] for r in runs], 50), 0),
                          "tokens_mean": rnd(mean([r["tokens"] for r in runs]), 0), "wall_s_p50": rnd(pct([r["wall_ms"] / 1000 for r in runs], 50), 2),
                          "wall_s_mean": rnd(mean([r["wall_ms"] / 1000 for r in runs]), 2),
                          "mixed_share": rnd(sum(r["phases"].count("M") for r in runs) / max(1, sum(r["steps"] for r in runs)), 3),
                          "ctx0_p50": rnd(pct([r["ctx0"] for r in runs], 50), 0)}
        if len(runs) < 0.5 * len(kok): guard.append("the step log shows %d prefill runs for %d kept requests" % (len(runs), len(kok)))
        st = S.get("stream") or {}
        if st.get("gb_per_nd_step_lsq") is not None and L["newcomers"]["steps_mean"] is not None:
            L["newcomers"]["gb_mean"] = rnd(st["gb_per_nd_step_lsq"] * L["newcomers"]["steps_mean"], 2)
            L["newcomers"]["gb_p50"] = rnd(st["gb_per_nd_step_lsq"] * (L["newcomers"]["steps_p50"] or 0), 2)
        if st.get("nd_steps") is not None and kok:
            L["nd_steps_per_kept_request_stats"] = rnd(st["nd_steps"] / len(kok), 2)
    except Exception as e:  # noqa: BLE001
        guard.append("step log not read (%s: %s)" % (type(e).__name__, str(e)[:120]))
    R["steplog"] = L
    R["guard"] = guard
    # ---- the lines
    fu, vu, kl, stv, nw = (S.get("fill_used_pct") or {}), (S.get("vram_units_mean_ranks") or {}), (S.get("kv_lent_gib_mean_ranks") or {}), (S.get("stream") or {}), (L.get("newcomers") or {})
    out.append("%s cc (latent %s; %d users x %s turns, first %s excluded; %s, T %s, trace-seeded %s, seed %s): %d of %d kept answered | "
               "TTFT p50 %s / p90 %s / max %s s | decode a stream p50 %s / p90 %s (p10 %s) tok/s, aggregate %s tok/s, in flight %s | "
               "work a minute (kept window %s-%s s): %s requests, %s generated tok, %s uncached prompt tok | whole run (makespan %s s): %s requests, %s gen tok, %s uncached tok a minute | "
               "errors %d, aborted %d%s | first turns (cold, preaged) TTFT max %s s | gen p50 %s mean %s, cache hit %s %%, draft acceptance %s%s" % (
                   a.tag, a.lat, cfg.get("users"), want, wt, cfg.get("gen_mode"), cfg.get("temperature"), cfg.get("trace_seeded"), cfg.get("seed"), len(kok), len(kept),
                   R["ttft_p50"], R["ttft_p90"], R["ttft_max"], R["dec_p50"], R["dec_p90"], R["dec_p10"], R["agg_tps"], R["inflight"],
                   R["kept_window_s"][0], R["kept_window_s"][1], R["work_kept"]["req_min"], R["work_kept"]["gen_tok_min"], R["work_kept"]["unc_tok_min"],
                   R["makespan_s"], R["work_all"]["req_min"], R["work_all"]["gen_tok_min"], R["work_all"]["unc_tok_min"],
                   len(errs), len(abo), (" (first: %s)" % str(R["first_err"])[:120]) if errs else "",
                   rnd(max([x for x in R["first_turns"]["ttft"] if x is not None] or [0]), 1), R["gen_p50"], R["gen_mean"], R["cache_hit_pct"], R["draft_acc"],
                   (" | INCOMPLETE: users %s did not finish %d turns before the cap (work a minute still valid; pairs over common requests)" % (short, want)) if short else ""))
    out.append("%s contexts: prompt p10 %s / p50 %s / p90 %s / max %s tokens over %d requests (>= 50k: %d, >= 100k: %d); first turns %s" % (
        a.tag, R["ctx"]["p10"], R["ctx"]["p50"], R["ctx"]["p90"], R["ctx"]["max"], R["ctx"]["n"], R["ctx"]["ge50k"], R["ctx"]["ge100k"], R["first_turns"]["prompt"]))
    if sm:
        out.append("%s pool / experts (/stats every ~%s s, %d samples, %d in the kept window, %d failed): KV pool %s tokens; fill (used) %% start %s / p50 %s / max %s / end %s "
                   "(kept p50 %s, max %s), cached %% max %s | expert units in VRAM a rank (mean of ranks) start %s / min %s / kept mean %s / end %s; by rank kept mean %s | "
                   "kv-lent GiB a rank (mean of ranks) start %s / min %s / kept mean %s / end %s | preemptions %s" % (
                       a.tag, a.every, len(sm), len(ks), nerr, S.get("kv_tokens_total"),
                       fu["all"]["start"], fu["all"]["p50"], fu["all"]["max"], fu["all"]["end"], fu["kept"]["p50"], fu["kept"]["max"],
                       (S.get("fill_cached_pct") or {}).get("all", {}).get("max"),
                       vu["all"]["start"], vu["all"]["min"], vu["kept"]["mean"], vu["all"]["end"],
                       [S["vram_units"][k]["kept"]["mean"] for k in sorted(S["vram_units"])],
                       (kl.get("all") or {}).get("start"), (kl.get("all") or {}).get("min"), (kl.get("kept") or {}).get("mean"), (kl.get("all") or {}).get("end"),
                       S.get("preemptions")))
        tlv = S.get("timeline") or []
        step = max(1, len(tlv) // 12)
        out.append("%s timeline (minute: fill used %% / expert units a rank / kv-lent GiB a rank / running+waiting): %s" % (
            a.tag, " | ".join("%d: %s / %s / %s / %s+%s" % (x["min"], x["fill_used_pct"], x["vram_units_mean"], x["kv_lent_gib_mean"], x["running"], x["waiting"])
                              for x in tlv[::step])))
        out.append("%s streamed (%s, %s intervals): %s GB over %s non-decode + %s decode steps; least squares %s GB a non-decode step, %s GB a decode step "
                   "(ratio %s GB a non-decode step incl. decode streaming); staged %s GB a non-decode step; by rank %s GB | newcomers (step log, prefill runs "
                   "q > %d in the kept window) %s: steps p50 %s / mean %s / max %s, tokens p50 %s, wall p50 %s s, MIXED share %s, cached prefix p50 %s -> streamed "
                   "%s GB a newcomer (mean; p50 %s) | steps: decode %s (wall p50 %s ms, %s seqs, %s rows), MIXED %s (p50 %s ms, n_tok p50 %s), prefill %s (p50 %s ms)" % (
                       a.tag, stv.get("basis"), stv.get("intervals"), stv.get("stream_gb"), stv.get("nd_steps"), stv.get("decode_steps"),
                       stv.get("gb_per_nd_step_lsq"), stv.get("gb_per_decode_step_lsq"), stv.get("gb_per_nd_step_ratio"), stv.get("staged_gb_per_nd_step_lsq"),
                       stv.get("rank_stream_gb"), 1 + a.nst, nw.get("n"), nw.get("steps_p50"), nw.get("steps_mean"), nw.get("steps_max"), nw.get("tokens_p50"),
                       nw.get("wall_s_p50"), nw.get("mixed_share"), nw.get("ctx0_p50"), nw.get("gb_mean"), nw.get("gb_p50"),
                       (L.get("decode") or {}).get("steps"), (L.get("decode") or {}).get("wall_ms_p50"), L.get("decode_seqs_p50"), L.get("decode_rows_p50"),
                       (L.get("MIXED") or {}).get("steps"), (L.get("MIXED") or {}).get("wall_ms_p50"), (L.get("MIXED") or {}).get("n_tok_p50"),
                       (L.get("prefill") or {}).get("steps"), (L.get("prefill") or {}).get("wall_ms_p50")))
        out.append("%s lever counters over the run (information): %s" % (a.tag, S.get("levers")))
    out.append("%s guard: %s" % (a.tag, ("VIRHE " + "; ".join(guard)) if guard else "OK (records, kept answers, max prompt %d >= %d, samples, step log)" % (pmax, a.minctx)))
    R["lines"] = out
    if a.json:
        with open(a.json, "w") as f: json.dump(R, f)
    for l in out: print("RESULT xcc " + l)
    return 0 if not guard else 2


# ---------------------------------------------------------------- the report
KEYS = [("TTFT p50 s", lambda r: r.get("ttft_p50")), ("TTFT p90 s", lambda r: r.get("ttft_p90")),
        ("decode p50 tok/s a stream", lambda r: r.get("dec_p50")), ("aggregate tok/s", lambda r: r.get("agg_tps")),
        ("requests a min (kept)", lambda r: (r.get("work_kept") or {}).get("req_min")),
        ("gen tok a min (kept)", lambda r: (r.get("work_kept") or {}).get("gen_tok_min")),
        ("requests a min (run)", lambda r: (r.get("work_all") or {}).get("req_min")),
        ("makespan s", lambda r: r.get("makespan_s")),
        ("KV fill max %", lambda r: (((r.get("samples") or {}).get("fill_used_pct") or {}).get("all") or {}).get("max")),
        ("expert units a rank (kept mean)", lambda r: (((r.get("samples") or {}).get("vram_units_mean_ranks") or {}).get("kept") or {}).get("mean")),
        ("expert units a rank (min)", lambda r: (((r.get("samples") or {}).get("vram_units_mean_ranks") or {}).get("all") or {}).get("min")),
        ("kv-lent GiB a rank (kept mean)", lambda r: (((r.get("samples") or {}).get("kv_lent_gib_mean_ranks") or {}).get("kept") or {}).get("mean")),
        ("streamed GB a non-decode step", lambda r: ((r.get("samples") or {}).get("stream") or {}).get("gb_per_nd_step_lsq")),
        ("streamed GB a newcomer", lambda r: ((r.get("steplog") or {}).get("newcomers") or {}).get("gb_mean")),
        ("errors", lambda r: r.get("errors"))]


def cmd_report(a):
    runs = a.runs.split()
    R = {}
    for t in runs:
        try:
            with open(os.path.join(a.dir, t + "-cc.json")) as f: R[t] = json.load(f)
        except (OSError, ValueError):
            pass
    lines = []
    for t in runs:
        r = R.get(t)
        if not r:
            lines.append("table %s: no reading (the start or its cc run did not complete)" % t); continue
        lines.append("table %s (%s): %s%s" % (t, r.get("lat"), " | ".join("%s %s" % (k, f(r)) for k, f in KEYS),
                                              (" | guard VIRHE: " + "; ".join(r["guard"])) if r.get("guard") else ""))
    done = [t for t in runs if t in R]
    pairs = [(x, y) for x, y in zip(done, done[1:]) if R[x].get("lat") != R[y].get("lat")]
    for x, y in pairs:
        n, b = (y, x) if R[y].get("lat") == "nvfp4" else (x, y)
        lines.append("pair %s vs %s (nvfp4 vs bf16, adjacent starts, %s first%s): %s" % (
            n, b, x, a.order_note and ", " + a.order_note or "",
            " | ".join("%s %s vs %s (%s %%)" % (k, f(R[n]), f(R[b]), pc(f(R[n]), f(R[b]))) for k, f in KEYS)))
    lat = {}
    for t in done: lat.setdefault(R[t].get("lat"), []).append(t)
    if len(lat.get("nvfp4", [])) >= 2 and len(lat.get("bf16", [])) >= 2:
        mm = lambda ts, f: mean([f(R[t]) for t in ts])   # noqa: E731
        lines.append("modes (mean of %s vs mean of %s, B A A B with fresh starts): %s" % (
            "+".join(lat["nvfp4"]), "+".join(lat["bf16"]),
            " | ".join("%s %s vs %s (%s %%)" % (k, rnd(mm(lat["nvfp4"], f), 2), rnd(mm(lat["bf16"], f), 2), pc(mm(lat["nvfp4"], f), mm(lat["bf16"], f)))
                       for k, f in KEYS)))
    nv = sum(1 for t in done if R[t].get("guard"))
    lines.append("runs: %d of %d with a reading (%s), %d with a guard VIRHE; NO identity claim (natural mode: replies may differ between runs; "
                 "nvfp4's exactness is KV4's identity result, not this item's)" % (len(done), len(runs), " ".join(done) or "none", nv))
    for l in lines: print("RESULT xcc " + l)
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="ccsum-st-")
    try:
        start = "2026-11-12 10:00:00"; tz = time.mktime(time.strptime(start, "%Y-%m-%d %H:%M:%S"))
        cfg = {"config": {"users": 2, "turns": 4, "warmup_turns": 1, "seed": 1, "gen_mode": "natural", "temperature": 0.0, "trace_seeded": True,
                          "cpt_fixed": True, "effort": "xhigh,medium", "ctx_cap": 170000, "duration": 1500.0, "run_id": "x", "start": start}}
        recs = []
        for u in range(2):
            t = 1.0 + u
            for k in range(1, 5):
                p = 60000 + 30000 * u + 2000 * k
                recs.append({"t": t, "url": "http://x", "user": u, "session": "u%ds0" % u, "turn": k, "uturn": k, "prompt": p, "prompt_est": p,
                             "cached": 0 if k == 1 else p - 3000, "gen": 400, "reasoning": 300, "ttft": 20.0 if k == 1 else 2.0 + k * 0.1,
                             "total": 22.0 if k == 1 else 18.0, "decode": 25.0 + k, "finish": "tool_calls", "draft_acc": 0.6, "think": 0.8})
                t += 20.0
        cc = os.path.join(d, "cc.jsonl")
        with open(cc, "w") as f:
            for x in [cfg] + recs: f.write(json.dumps(x) + "\n")
        sp = os.path.join(d, "s.jsonl")
        ndi, dci = [0, 5, 1, 7, 2, 4, 6, 3, 8, 2], [0, 95, 120, 80, 100, 90, 110, 70, 130, 85]
        with open(sp, "w") as f:
            for i in range(10):
                ND, DC = sum(ndi[:i + 1]), sum(dci[:i + 1])
                f.write(json.dumps(compact({"kv_tokens_used": 100000 * i, "kv_tokens_total": 2656334, "kv_tokens_cached": 1000 * i, "running": 2, "waiting": 0,
                                            "steps": ND + DC, "steps_prefill": ND // 2, "steps_mixed": ND - ND // 2, "steps_decode": DC, "preemptions": 0,
                                            "experts": {"n_units": 3936, "bytes": 38285475840,
                                                        "ranks": [{"rank": r, "vram": 3000 - 10 * i, "pinned": 900, "stream_bytes": 1e9 * i,
                                                                   "lent": {"kv": {"held": 700 - 20 * i}, "arena": {"held": 300}}} for r in range(4)]},
                                            "link": {"stream_bytes": 5e9 * ND + 0.1e9 * DC, "staged_bytes": 1e9 * ND},
                                            "psk": {"latched": i}, "dspark_pfctx": {"passes": i}, "engdd": {"passes": 2 * i}}, round(tz + 10 * i, 3))) + "\n")
            f.write(json.dumps({"t": tz + 101, "err": "timed out"}) + "\n")
        lg = os.path.join(d, "e.log")
        with open(lg, "w") as f:
            tt = tz + 21.0
            for k in range(40):
                ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(tt)) + ".%06dZ" % int((tt % 1) * 1e6)
                if k % 10 == 0:
                    f.write("%s I step %d MIXED n_seq=2 n_tok=4102  [0] slot=0 q=6 ctx=900  [1] slot=1 q=4096 ctx=%d\n" % (ts, k, 60000 + k))
                elif k % 10 == 1:
                    f.write("%s I step %d MIXED n_seq=2 n_tok=1006  [0] slot=0 q=6 ctx=906  [1] slot=1 q=1000 ctx=%d\n" % (ts, k, 64096 + k))
                else:
                    f.write("%s I step %d decode n_seq=2 n_tok=12  [0] slot=0 q=6 ctx=910  [1] slot=1 q=6 ctx=%d\n" % (ts, k, 65100 + k))
                tt += 0.05
        ns = argparse.Namespace(cc=cc, samples=sp, log=lg, tag="b8-1", lat="bf16", ccsim=os.environ.get("CCSUM_CCSIM", os.path.join(HERE, "..", "..", "tools", "ccsim", "cc_sim3.py")),
                                minctx=50000, nst=5, json=os.path.join(d, "b8-1-cc.json"), idle=3.0, every=10)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = cmd_run(ns)
        o = buf.getvalue()
        R = json.load(open(ns.json))
        assert rc == 0, (rc, R["guard"], o[:2000])
        assert R["kept_ok"] == 6 and R["all_ok"] == 8, R["kept_ok"]
        assert R["ttft_p50"] == 2.3 and R["errors"] == 0 and R["ctx"]["max"] == 98000, (R["ttft_p50"], R["ctx"])
        assert abs(R["samples"]["stream"]["gb_per_nd_step_lsq"] - 5.0) < 1e-6 and abs(R["samples"]["stream"]["gb_per_decode_step_lsq"] - 0.1) < 1e-6, R["samples"]["stream"]
        assert R["samples"]["failed_gets"] == 1 and R["samples"]["fill_used_pct"]["all"]["max"] == round(100.0 * 900000 / 2656334, 1)
        assert R["steplog"]["newcomers"]["n"] == 4 and R["steplog"]["newcomers"]["steps_p50"] == 2.0 and R["steplog"]["newcomers"]["tokens_p50"] == 5096, R["steplog"]
        assert abs(R["steplog"]["newcomers"]["gb_mean"] - 10.0) < 1e-6
        assert "RESULT xcc b8-1 guard: OK" in o, o[-400:]
        # the guard: a short trace is VIRHE
        ns2 = argparse.Namespace(**dict(vars(ns), minctx=200000, json=os.path.join(d, "x.json")))
        with contextlib.redirect_stdout(io.StringIO()): rc2 = cmd_run(ns2)
        assert rc2 == 2 and "long context" in json.load(open(ns2.json))["guard"][0]
        # the report (a second run as nvfp4)
        R2 = dict(R); R2["tag"] = "n8-1"; R2["lat"] = "nvfp4"; R2["ttft_p50"] = 2.0
        json.dump(R2, open(os.path.join(d, "n8-1-cc.json"), "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): cmd_report(argparse.Namespace(dir=d, runs="b8-1 n8-1", order_note="A / B once"))
        o2 = buf.getvalue()
        assert "pair n8-1 vs b8-1" in o2 and "TTFT p50 s 2.0 vs 2.3 (-13.0 %)" in o2, o2[:1500]
        for t, l in (("b8-2", "bf16"), ("n8-2", "nvfp4")):
            x = dict(R); x["tag"] = t; x["lat"] = l; json.dump(x, open(os.path.join(d, t + "-cc.json"), "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): cmd_report(argparse.Namespace(dir=d, runs="n8-1 b8-1 b8-2 n8-2", order_note=""))
        o3 = buf.getvalue()
        assert o3.count("RESULT xcc pair ") == 2 and "modes (mean of n8-1+n8-2 vs mean of b8-1+b8-2" in o3, o3[:2500]
        # prefill runs: a slot reused after its decode starts a new run
        P = prefill_runs([{"t": 0, "ph": "prefill", "ents": [(3, 4096, 0)], "wall_ms": 100}, {"t": 1, "ph": "MIXED", "ents": [(3, 500, 4096)], "wall_ms": 50},
                          {"t": 2, "ph": "decode", "ents": [(3, 6, 4596)], "wall_ms": 30}, {"t": 3, "ph": "MIXED", "ents": [(3, 2000, 9000)], "wall_ms": 40},
                          {"t": 4, "ph": "decode", "ents": [(3, 1, 11000)], "wall_ms": 30}], 5)
        assert [(r["steps"], r["tokens"]) for r in P] == [(2, 4596), (1, 2000)], P
    finally:
        for f in os.listdir(d): os.remove(os.path.join(d, f))
        os.rmdir(d)
    print("SELFTEST OK: run (kept turns, TTFT p50 / p90, decode, work a minute, contexts + the long-context guard, samples: fill / units / "
          "kv-lent / least-squares streamed GB a non-decode step, failed GETs, step log: classes + prefill runs + GB a newcomer), report "
          "(table, adjacent pairs, B A A B mode means), prefill runs")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("--url", required=True); p.add_argument("--every", type=float, default=10.0)
    p.add_argument("--out", required=True); p.add_argument("--max-s", type=float, default=7200.0)
    p = sub.add_parser("trace"); p.add_argument("--ccsim", required=True); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--users", type=int, default=8); p.add_argument("--minctx", type=int, default=50000); p.add_argument("--ref", default="")
    p = sub.add_parser("run"); p.add_argument("--cc", required=True); p.add_argument("--samples", required=True); p.add_argument("--log", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--lat", required=True); p.add_argument("--ccsim", required=True)
    p.add_argument("--minctx", type=int, default=50000); p.add_argument("--nst", type=int, default=5); p.add_argument("--idle", type=float, default=3.0)
    p.add_argument("--every", type=float, default=10.0); p.add_argument("--json")
    p = sub.add_parser("report"); p.add_argument("--dir", required=True); p.add_argument("--runs", required=True); p.add_argument("--order-note", default="")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest": return selftest()
    return {"sample": cmd_sample, "trace": cmd_trace, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
