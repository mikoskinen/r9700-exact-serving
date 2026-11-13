#!/usr/bin/env python3
"""kv4sum.py (12.11.2026, queue-ar/dsmain/kv4, ds-main-port kv4 builder) -- the readings of dsm-kv4.sh (lever KV4: the
packed FP4 compressed latent, RADIANCE_DS41_LATENT=nvfp4, as a capacity lever; tag "RESULT xkv4 "). Standard library only;
reads .gz or plain files; never writes beside itself.

  kv4sum.py cap --log START.log --stats STATS.json --tag T --lat bf16|nvfp4 --ms N --mml M --hp MIB --json OUT
      the start's capacity: the latent / index-key literal (deepseek41's indexer note), the planner's split (VRAM budget:
      experts / kv cache / elastic / prefill staging / arena), "kv: the pool backs P% of the worst case (S sequences all at
      T tokens), X full-length sessions", "kv: the cache can grow to G a card -- N tokens of context, X sessions at M --
      ...; up to L is lent to the experts", the movers' slab / pinned slots, the host pool; /stats kv_tokens_total /
      kv_bytes_carved (bytes a token = carved / tokens) and experts (unit bytes, per rank vram / pinned / trunk / lent).
      Prints "CAP {json}" (guard: the latent literal is the asked mode, the pool lines and four mover lines are there).
  kv4sum.py hp --b CAP_b8.json --n CAP_n8.json --hp MIB
      the expert-heavy bf16 start's --host-pool-mib: HP - (n8's experts - b8's experts) when the nvfp4 planner moved
      >= 0.25 GiB to the experts (core/mem/vram_budget.cpp's lends split), else HP - what nvfp4 frees at b8's token
      capacity (kv x (1 - bytes a token nvfp4 / bf16)). Prints "HP <MiB> <why>" (>= 4096).
  kv4sum.py psk --a ST0.json --b ST1.json --deepq OUT
      the PSK guard on the corrected /stats psk counters over one cold deepq prompt: latched, planned_steps, cut_passes grow,
      plan_mismatch +0 (layers_skipped, ckpt_dropped: information only); the prompt cold (cached 0), status 200.
  kv4sum.py bi48 --log ENGINE.log.gz --win NAME:T0:T1 [...] [--nst 5] --json OUT
      the step log (RADIANCE_LOG_STEPS=1, `docker logs -t`): per window the steps, the decode rows a step (entries with
      q <= 1 + nst and ctx > 0), steps with > 48 / > 32 decode rows, the most; and deepseek41's "lever BI stats" lines
      (cumulative "steps with > 48 decode rows", printed every 512 segmented steps) at the window's edges.
  kv4sum.py report --dir O --starts "b8 n8 ..." [--ref TEXT]
      every "RESULT xkv4 ..." summary line (printed): capacity per start, MAXSEQS 8 / 12 / 16 per mode, the smaller-pool
      prices, speed per start, the pairs (nvfp4 vs bf16 at MAXSEQS 8 and 16, expert-heavy vs today, equal expert VRAM),
      the c16 cell, the >48-row BI check, identity, guards.
  kv4sum.py selftest
"""
import argparse, calendar, gzip, json, math, os, re, statistics, sys, time

sys.dont_write_bytecode = True
GIB = 1024.0 ** 3
SZ = r"([\d.]+ (?:B|KiB|MiB|GiB|TiB))"
UNIT = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4}
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")      # longid.py's / ncsteps.py's
ENT = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")
BIST = re.compile(r"lever BI stats: (\d+) segmented steps, .*?(\d+) steps with > 48 decode rows; ds41_bi (\d+)")


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def size(s):
    if not s: return None
    v, u = s.split()
    return float(v) * UNIT[u]


def gib(s):
    b = size(s)
    return None if b is None else round(b / GIB, 3)


def jload(p):
    try:
        with opn(p) as f: return json.load(f)
    except (OSError, ValueError):
        return None


def jl(p):
    rows = []
    try:
        with opn(p) as f:
            for l in f:
                l = l.strip()
                if not l: continue
                try: rows.append(json.loads(l))
                except ValueError: pass
    except OSError:
        pass
    return rows


def num(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict): return None
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None


def f(x, nd=1):
    return "-" if x is None else format(x, ".%df" % nd)


def pc(a, b):
    return "%+.1f %%" % ((a / b - 1) * 100) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else "-"


def ts_split(line):
    m = TSRE.match(line)
    if not m: return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z": t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


# ---------------------------------------------------------------- cap
def ranks_of(st):
    out = []
    for r in ((st or {}).get("experts") or {}).get("ranks") or []:
        lent = r.get("lent") or {}
        out.append({"rank": r.get("rank"), "vram": r.get("vram"), "pinned": r.get("pinned"), "trunk_vram": r.get("trunk_vram"),
                    "draft_vram": r.get("draft_vram"), "lent_kv_held": (lent.get("kv") or {}).get("held"),
                    "lent_arena_held": (lent.get("arena") or {}).get("held")})
    return out


def parse_cap(text, st, a):
    c = {"tag": a.tag, "lat": a.lat, "ms": a.ms, "mml": a.mml, "hp": a.hp, "guard": []}
    m = re.search(r"latent (bf16|nvfp4) \((\d+) bytes an entry; RADIANCE_DS41_LATENT\)", text)
    c["lat_lit"], c["lat_bytes"] = (m.group(1), int(m.group(2))) if m else (None, None)
    if c["lat_lit"] != a.lat: c["guard"].append("latent literal %s, asked %s" % (c["lat_lit"], a.lat))
    m = re.search(r"keys (bf16|mxfp4) \((\d+) bytes an entry; RADIANCE_DS41_IDXKEY\)", text)
    c["keys"], c["key_bytes"] = (m.group(1), int(m.group(2))) if m else (None, None)
    m = re.search(r"kv: the pool backs (\d+)% of the worst case \((\d+) sequences all at (\d+) tokens\), ([\d.]+) full-length sessions", text)
    if m: c.update(pool_pct=int(m.group(1)), pool_seqs=int(m.group(2)), pool_tok=int(m.group(3)), pool_sessions=float(m.group(4)))
    else: c["guard"].append("no 'kv: the pool backs' line")
    m = re.search(r"kv: the cache can grow to " + SZ + r" a card -- (\d+) tokens of context, ([\d.]+) sessions at (\d+) -- .*?; "
                  + SZ + r" of that is per-sequence state, and (?:up to " + SZ + r" is lent|nothing is lent)", text)
    if m:
        c.update(kv_gib=gib(m.group(1)), kv_tokens=int(m.group(2)), kv_sessions=float(m.group(3)), kv_mlen=int(m.group(4)),
                 state_mib=round(size(m.group(5)) / 1024 ** 2, 2), lent_gib=gib(m.group(6)) if m.group(6) else 0.0)
    else: c["guard"].append("no 'kv: the cache can grow to' line")
    blk = text[text.find("VRAM budget, per card"):] if "VRAM budget, per card" in text else ""
    for k, nm in (("arena_gib", "activation arena"), ("elastic_gib", "elastic"), ("exp_gib", "experts"),
                  ("kvc_gib", "kv cache"), ("stage_gib", "prefill staging")):
        m = re.search(r"^\s+" + nm + r"\s+" + SZ, blk, re.M)
        c[k] = gib(m.group(1)) if m else None
    if c.get("exp_gib") is None: c["guard"].append("no VRAM budget experts line")
    m = re.search(r"host_pool\s+" + SZ + " budget", text)
    c["hostpool_gib"] = gib(m.group(1)) if m else None
    mv = re.findall(r"I\[(\d)\] mover: (\d+) slab slots \((\d+) shadow[^)]*\), (\d+) pinned pool slots, " + SZ + " static vram", text)
    c["movers"] = [{"rank": int(r), "slab": int(s), "shadow": int(sh), "pinned": int(p), "static_gib": gib(sv)} for r, s, sh, p, sv in mv]
    if len(c["movers"]) != 4: c["guard"].append("%d mover lines (want 4)" % len(c["movers"]))
    m = re.search(r"pinned by this process ([\d.]+) GiB", text)
    c["pinned_proc_gib"] = float(m.group(1)) if m else None
    m = re.search(r"checkpoint slots: (\d+) .*?\(" + SZ + r"\)", text)
    c["ckpt_slots"], c["ckpt_gib"] = (int(m.group(1)), gib(m.group(2))) if m else (None, None)
    m = re.search(r"kv tiers: .*?\(" + SZ + r", (\d+) entries, (\d+) tokens of context\)", text)
    c["tier_gib"], c["tier_tokens"] = (gib(m.group(1)), int(m.group(3))) if m else (None, None)
    st = st or {}
    c["st_kv_tokens_total"], c["st_kv_bytes_carved"] = num(st, "kv_tokens_total"), num(st, "kv_bytes_carved")
    c["bpt"] = round(c["st_kv_bytes_carved"] / c["st_kv_tokens_total"], 1) if c["st_kv_tokens_total"] and c["st_kv_bytes_carved"] else None
    e = st.get("experts") or {}
    c["n_units"], c["unit_bytes"] = e.get("n_units"), (round(e["bytes"] / e["n_units"]) if e.get("n_units") and e.get("bytes") else None)
    c["ranks"] = ranks_of(st)
    if not c["ranks"]: c["guard"].append("no /stats experts.ranks")
    return c


def cmd_cap(a):
    try:
        with opn(a.log) as fh: text = fh.read()
    except OSError as e:
        print("CAP %s: VIRHE no start log (%s)" % (a.tag, e)); return 2
    c = parse_cap(text, jload(a.stats), a)
    rv = [r.get("vram") for r in c["ranks"]]
    line = ("CAP %s: latent %s (%s B an entry), keys %s | MAXSEQS %s MML %s HOSTPOOL %s MiB | VRAM experts %s GiB, kv cache %s GiB "
            "(elastic %s, staging %s) | pool backs %s %% of %s x %s, %s full-length sessions | cache %s GiB = %s tokens, %s sessions at %s, "
            "%s B a token (/stats), lent up to %s GiB | units a rank vram %s (unit %s MB) %s" % (
                c["tag"], c["lat_lit"], c["lat_bytes"], c["keys"], c["ms"], c["mml"], c["hp"], c.get("exp_gib"), c.get("kvc_gib"),
                c.get("elastic_gib"), c.get("stage_gib"), c.get("pool_pct"), c.get("pool_seqs"), c.get("pool_tok"),
                c.get("pool_sessions"), c.get("kv_gib"), c.get("kv_tokens"), c.get("kv_sessions"), c.get("kv_mlen"), c.get("bpt"),
                c.get("lent_gib"), rv, f(c["unit_bytes"] / 1e6 if c["unit_bytes"] else None, 2),
                ("-- VIRHE " + "; ".join(c["guard"])) if c["guard"] else ""))
    c["line"] = line
    print(line)
    if a.json:
        with open(a.json, "w") as fh: json.dump(c, fh)
    return 0 if not c["guard"] else 2


# ---------------------------------------------------------------- hp
def cmd_hp(a):
    b, n = jload(a.b), jload(a.n)
    if not b or not n or b.get("exp_gib") is None or n.get("exp_gib") is None:
        print("HP none no b8 / n8 capacity"); return 2
    d = n["exp_gib"] - b["exp_gib"]
    if d >= 0.25:
        hp, why = a.hp - int(round(d * 1024)), "n8's planner moved %.2f GiB to the experts (n8 %.2f vs b8 %.2f GiB)" % (d, n["exp_gib"], b["exp_gib"])
    elif b.get("kvc_gib") and b.get("bpt") and n.get("bpt"):
        d = b["kvc_gib"] * (1 - n["bpt"] / b["bpt"])
        hp, why = a.hp - int(round(d * 1024)), "fallback: n8's planner kept the split; what nvfp4 frees at b8's tokens %.2f GiB" % d
    else:
        print("HP none no split delta"); return 2
    hp = max(hp, 4096)
    print("HP %d %s" % (hp, why))
    return 0


# ---------------------------------------------------------------- psk
def cmd_psk(a):
    A, B = (jload(a.a) or {}).get("psk"), (jload(a.b) or {}).get("psk")
    dq = {}
    try:
        for l in opn(a.deepq):
            if l.startswith("DEEPQ "): dq = json.loads(l[6:]); break
    except (OSError, ValueError):
        pass
    why = []
    if not isinstance(A, dict) or not isinstance(B, dict):
        print("FAIL no /stats psk object"); return 2
    dl = {k: (B.get(k) or 0) - (A.get(k) or 0) for k in B if isinstance(B.get(k), (int, float))}
    for k in ("latched", "planned_steps", "cut_passes"):
        if dl.get(k, 0) <= 0: why.append("%s did not grow (%+d)" % (k, dl.get(k, 0)))
    if dl.get("plan_mismatch", 0) != 0: why.append("plan_mismatch grew (%+d)" % dl["plan_mismatch"])
    if dq.get("status") != 200: why.append("deepq status %s" % dq.get("status"))
    if dq.get("cached"): why.append("deepq not cold (cached %s)" % dq.get("cached"))
    print(("OK" if not why else "FAIL " + "; ".join(why)) + " | psk mode %s k %s | deltas %s | deepq %s tok, cached %s, TTFT %s s" % (
        B.get("mode"), B.get("k"), ", ".join("%s %+d" % (k, v) for k, v in sorted(dl.items()) if v), dq.get("prompt_tokens"),
        dq.get("cached"), dq.get("ttft")))
    return 0 if not why else 2


# ---------------------------------------------------------------- bi48
def read_log(path):
    steps, bis = [], []
    with opn(path) as fh:
        for l in fh:
            if "n_seq=" not in l and "lever BI stats" not in l: continue
            t, rest = ts_split(l.rstrip("\n"))
            if t is None: continue
            m = STEP.search(rest)
            if m:
                steps.append({"t": t, "K": int(m.group(1)), "ph": m.group(2), "n_tok": int(m.group(4)),
                              "ents": [(int(s), int(q), int(c)) for _, s, q, c in ENT.findall(m.group(5))]})
                continue
            m = BIST.search(rest)
            if m: bis.append((t, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return steps, bis


def bi48_windows(steps, bis, wins, nst):
    out = {}
    for nm, t0, t1 in wins:
        W = [s for s in steps if t0 <= s["t"] < t1]
        dec = [sum(q for _, q, c in s["ents"] if q <= nst + 1 and c > 0) for s in W]
        before = [b for b in bis if b[0] <= t0]
        upto = [b for b in bis if b[0] <= t1]
        b0, b1 = (before[-1] if before else None), (upto[-1] if upto else None)
        out[nm] = {"steps": len(W), "decode_rows_max": max(dec) if dec else 0,
                   "steps_gt48": sum(1 for d in dec if d > 48), "steps_gt32": sum(1 for d in dec if d > 32),
                   "mixed_gt48": sum(1 for s, d in zip(W, dec) if d > 48 and s["ph"] == "MIXED"),
                   "decode_rows_p50": sorted(dec)[len(dec) // 2] if dec else 0,
                   "bistats_gt48_delta": (b1[2] - (b0[2] if b0 else 0)) if b1 else None,
                   "bistats_seg_delta": (b1[1] - (b0[1] if b0 else 0)) if b1 else None,
                   "bistats_lines": len(bis), "ds41_bi": b1[3] if b1 else None}
    return out


def cmd_bi48(a):
    try:
        steps, bis = read_log(a.log)
    except OSError as e:
        print("BI48 VIRHE no engine log (%s)" % e); return 2
    wins = []
    for w in a.win:
        nm, t0, t1 = w.rsplit(":", 2)
        wins.append((nm, float(t0), float(t1)))
    out = bi48_windows(steps, bis, wins, a.nst)
    print("BI48 " + " | ".join("%s: %d steps, %d with > 48 decode rows (%d MIXED), %d > 32, most %d, p50 %d; BI stats > 48 +%s (granular 512 segmented steps)" % (
        k, v["steps"], v["steps_gt48"], v["mixed_gt48"], v["steps_gt32"], v["decode_rows_max"], v["decode_rows_p50"], v["bistats_gt48_delta"]) for k, v in out.items()))
    if a.json:
        with open(a.json, "w") as fh: json.dump(out, fh)
    return 0


# ---------------------------------------------------------------- report
def win_rows(o, tag, kind):
    """the mb windows of a start: [(name, ncsteps json, w0 /stats, w1 /stats)]"""
    rs, k = [], 1
    while os.path.exists(os.path.join(o, "%s-%s-%d.json" % (tag, kind, k))) or os.path.exists(os.path.join(o, "%s-%s-%d.txt" % (tag, kind, k))):
        nm = "%s-%s-%d" % (tag, kind, k)
        rs.append((nm, jload(os.path.join(o, nm + ".json")), jload(os.path.join(o, nm + "-st-w0.json")), jload(os.path.join(o, nm + "-st-w1.json"))))
        k += 1
    return rs


def wsum(rows):
    """mean over a start's windows of one kind; the window's guard list kept"""
    keys = {"agg": ("mb", "agg_dec_tps"), "p50": ("mb", "per_stream_dec_p50"), "share": ("mb", "decode_share"),
            "msdec": ("mb", "ms_decode_step"), "ttft": ("mb", "ttft_p50"), "wall": ("newcomers", "wall_ms", "p50"),
            "mixed_p50": ("classes", "MIXED", "wall_ms", "p50"), "stream_nd": ("stats", "per_nd", "link_stream_gb"),
            "staged_nd": ("stats", "per_nd", "staged_gb_r0"), "lstaged_nd": ("stats", "per_nd", "link_staged_gb"),
            "reg_stream_nd": ("regression", "link_stream_gb", "per_nd"), "st_share": ("stats", "decode_share"),
            "nc_n": ("newcomers", "found"), "nc_mixed": ("newcomers", "mixed")}
    out = {k: mean([num(j or {}, *p) for _, j, _, _ in rows]) for k, p in keys.items()}
    out["vals"] = {k: [num(j or {}, *p) for _, j, _, _ in rows] for k, p in keys.items() if k in ("agg", "wall", "share")}
    vr, kh, per = [], [], {}
    for _, _, w0, w1 in rows:
        for w in (w0, w1):
            rk = ranks_of(w)
            if rk:
                vr.append(mean([r["vram"] for r in rk])); kh.append(mean([r["lent_kv_held"] for r in rk]))
                for r in rk: per.setdefault(r["rank"], []).append(r["vram"])
    out["vram_units"], out["kv_lent_units"] = mean(vr), mean(kh)
    out["vram_by_rank"] = [None if mean(v) is None else int(round(mean(v))) for _, v in sorted(per.items(), key=lambda x: (x[0] is None, x[0]))]
    # streamed GB a rank a non-decode step: the window's per-rank stream delta over its non-decode steps (decode steps' streaming included)
    rs = []
    for _, j, _, _ in rows:
        w = ((j or {}).get("stats") or {}).get("window") or {}
        nd = (w.get("prefill") or 0) + (w.get("mixed") or 0)
        if nd and w.get("rank_stream_gb"): rs.append([x / nd for x in w["rank_stream_gb"]])
    out["rank_stream_nd"] = [round(mean([r[i] for r in rs if len(r) > i]), 3) for i in range(max((len(r) for r in rs), default=0))]
    out["guards"] = ["%s: %s" % (nm, "; ".join(j.get("guard") or [])) for nm, j, _, _ in rows if j is None or j.get("guard")]
    out["n"] = len(rows)
    return out


def u1sum(o, tag):
    v = {"essay": [], "code": []}
    k = 1
    while os.path.exists(os.path.join(o, "%s-1u-%d.txt" % (tag, k))):
        try:
            for l in open(os.path.join(o, "%s-1u-%d.txt" % (tag, k))):
                if l.startswith("J "):
                    d = json.loads(l[2:])
                    for p in v:
                        if (d.get("prompt") or "").startswith(p): v[p].append(d.get("decode_tps"))
        except (OSError, ValueError):
            pass
        k += 1
    return {p: mean(x) for p, x in v.items()}, v


def cmd_report(a):
    o, starts = a.dir, a.starts.split()
    caps = {t: jload(os.path.join(o, "%s-cap.json" % t)) for t in starts}
    deep = {(d.get("tag"), d.get("name")): d for d in jl(os.path.join(o, "deep.jsonl"))}
    load = {(d.get("tag"), d.get("name")): d for d in jl(os.path.join(o, "load.jsonl"))}
    bi = {t: jload(os.path.join(o, "%s-bi48.json" % t)) or {} for t in starts}
    S, lines = {}, []
    out = lambda s: lines.append("RESULT xkv4 " + s)   # noqa: E731
    # capacity
    for t in starts:
        c = caps.get(t)
        if not c: out("capacity %s: not run / no capacity file" % t); continue
        rk = c.get("ranks") or []
        out("capacity %s (latent %s, MAXSEQS %s, MML %s, HOSTPOOL %s MiB): VRAM experts %s GiB, kv cache %s GiB, staging %s GiB | pool backs %s %% of %s x %s = %s full-length sessions | cache %s tokens = %s sessions at %s, %s B a token incl. state (official FP4 main KV 890 B), lent to experts up to %s GiB | units a rank at start: vram %s (trunk %s, kv-lent held %s, arena-lent held %s), pinned %s; mover slab slots %s; unit %s MB%s" % (
            t, c.get("lat_lit"), c.get("ms"), c.get("mml"), c.get("hp"), c.get("exp_gib"), c.get("kvc_gib"), c.get("stage_gib"),
            c.get("pool_pct"), c.get("pool_seqs"), c.get("pool_tok"), c.get("pool_sessions"), c.get("kv_tokens"), c.get("kv_sessions"),
            c.get("kv_mlen"), c.get("bpt"), c.get("lent_gib"), [r.get("vram") for r in rk], [r.get("trunk_vram") for r in rk],
            [r.get("lent_kv_held") for r in rk], [r.get("lent_arena_held") for r in rk], [r.get("pinned") for r in rk],
            [m.get("slab") for m in c.get("movers") or []], f(c["unit_bytes"] / 1e6 if c.get("unit_bytes") else None, 2),
            (" -- VIRHE " + "; ".join(c["guard"])) if c.get("guard") else ""))
    b8, n8 = caps.get("b8"), caps.get("n8")
    ub = next((c["unit_bytes"] for c in caps.values() if c and c.get("unit_bytes")), None)
    units = lambda g: None if g is None or not ub else int(round(g * GIB / ub))   # noqa: E731
    for t in starts:
        c = caps.get(t)
        if c and b8 and t != "b8" and c.get("exp_gib") is not None and b8.get("exp_gib") is not None:
            d = c["exp_gib"] - b8["exp_gib"]
            out("split %s vs b8: experts %+.2f GiB (= %+s units a rank at %s MB), kv cache %+.2f GiB, full-length sessions %s vs %s, mean vram units a rank at start %s vs %s" % (
                t, d, units(d), f(ub / 1e6 if ub else None, 2), (c.get("kvc_gib") or 0) - (b8.get("kvc_gib") or 0), c.get("kv_sessions"),
                b8.get("kv_sessions"), f(mean([r.get("vram") for r in c.get("ranks") or []]), 0), f(mean([r.get("vram") for r in b8.get("ranks") or []]), 0)))
    # MAXSEQS 8 / 12 / 16 per mode from the pool numbers (the planner's lends split, core/mem/vram_budget.cpp)
    room = b8.get("kvc_gib") if b8 else None   # bf16 at MAXSEQS 8 is room-bound (its ceiling 8 x 524288 is above the room)
    for mode, ref in (("bf16", b8 or caps.get("b16")), ("nvfp4", n8 or caps.get("n16"))):
        if not ref or not ref.get("bpt"): continue
        cells = []
        for ms in (8, 12, 16):
            cap = ms * 524288 * ref["bpt"] / GIB
            kv = min(cap, room) if room else None
            got = next((c for c in caps.values() if c and c.get("lat_lit") == mode and c.get("ms") == ms and c.get("mml") == 524288 and c.get("hp") == (b8 or {}).get("hp")), None)
            meas = (" [measured: kv %s GiB, %s sessions, %s %%]" % (got.get("kvc_gib"), got.get("kv_sessions"), got.get("pool_pct"))) if got else " [from the pool numbers, no start]"
            if kv is None:
                cells.append("MAXSEQS %d: ceiling %.2f GiB%s" % (ms, cap, meas)); continue
            ses = kv * GIB / (524288 * ref["bpt"])
            cells.append("MAXSEQS %d: ceiling %.2f GiB -> kv %.2f GiB, %.1f full-length sessions (%.0f %% of the worst case), experts %+.2f GiB (%+s units) vs the room%s" % (
                ms, cap, kv, ses, 100 * min(1.0, ses / ms), room - kv, units(room - kv), meas))
        out("capacity by MAXSEQS, %s latent (%s B a token, MML 524288, room %s GiB = b8's cache): %s" % (mode, ref["bpt"], room, " | ".join(cells)))
    # smaller-pool prices at c8 (bf16): MML 262144, one full-length session (the planner's floor), the c8 working set
    if b8 and b8.get("bpt") and room:
        bpt = b8["bpt"]
        k262 = 8 * 262144 * bpt / GIB
        k1 = 524288 * bpt / GIB
        ws = []
        for _, j, _, _ in win_rows(o, "b8", "c8"):
            for r in ((j or {}).get("newcomers") or {}).get("rows") or []:
                if r.get("prompt"): ws.append(r["prompt"])
        wtok = 8 * ((mean(ws) or 1100) + 512)
        out("price of a smaller KV pool at c8 (bf16, b8's %s B a token, room %s GiB): MML 262144 -> ceiling %.2f GiB, experts %+.2f GiB (%+s units a rank) | one 524288 session (the planner's floor at MML 524288) %.2f GiB -> experts %+.2f GiB (%+s units) | the c8 working set ~%d tokens = %.3f GiB (8 x (prompt %s + 512 gen)) | --prefix-cache-mib 86016 (RAM KV tier, %s GiB / %s tokens) and --checkpoint-slots 512 (%s GiB) are pinned HOST memory, 0 VRAM: they only bound how far --host-pool-mib can grow (pinned by the process %s GiB)" % (
            bpt, room, k262, room - k262, units(room - k262), k1, room - k1, units(room - k1), wtok, wtok * bpt / GIB, f(mean(ws), 0),
            b8.get("tier_gib"), b8.get("tier_tokens"), b8.get("ckpt_gib"), b8.get("pinned_proc_gib")))
    # speed per start
    for t in starts:
        c8, c16 = wsum(win_rows(o, t, "c8")), wsum(win_rows(o, t, "c16"))
        u1, u1v = u1sum(o, t)
        d136, g16, ld = deep.get((t, "d136")) or {}, deep.get((t, "g16")) or {}, load.get((t, "ld")) or {}
        S[t] = {"c8": c8, "c16": c16, "u1": u1, "d136": d136, "ld": ld}
        if not (c8["n"] or c16["n"] or any(u1.values()) or d136): continue
        parts = []
        if any(u1.values()): parts.append("1u essay %s / code %s tok/s (%s / %s)" % (f(u1["essay"]), f(u1["code"]), u1v["essay"], u1v["code"]))
        for nm, w in (("c8", c8), ("c16", c16)):
            if not w["n"]: continue
            parts.append("%s (%d window(s)): agg %s tok/s %s, per-stream p50 %s, decode_share %s (/stats %s), decode step %s ms, TTFT p50 %s s | newcomer step wall p50 %s ms (MIXED p50 %s), %s newcomers (%s MIXED) | a non-decode step: streamed %s GB (by rank %s), staged r0 %s GB, link staged %s GB (regression stream %s) | vram units a rank %s %s, kv-lent held %s%s" % (
                nm, w["n"], f(w["agg"]), w["vals"]["agg"], f(w["p50"]), f(w["share"], 3), f(w["st_share"], 3), f(w["msdec"]), f(w["ttft"], 2),
                f(w["wall"]), f(w["mixed_p50"]), f(w["nc_n"], 1), f(w["nc_mixed"], 1), f(w["stream_nd"], 3), w["rank_stream_nd"], f(w["staged_nd"], 3),
                f(w["lstaged_nd"], 3), f(w["reg_stream_nd"], 3), f(w["vram_units"], 0), w["vram_by_rank"], f(w["kv_lent_units"], 0),
                (" -- VIRHE " + " / ".join(w["guards"])) if w["guards"] else ""))
        if d136: parts.append("cold %s tok TTFT %s s (cached %s)" % (d136.get("prompt_tokens"), d136.get("ttft"), d136.get("cached")))
        if g16: parts.append("cold %s tok TTFT %s s" % (g16.get("prompt_tokens"), g16.get("ttft")))
        if ld: parts.append("c4 + cold 64k: probe TTFT %s s, c4 agg %s tok/s / p50 %s in the prefill window, TPOT p90 %s ms" % (
            ld.get("ttft"), ld.get("c4_pf_agg_tps"), ld.get("c4_pf_stream_tps_p50"), ld.get("tpot_pf_p90")))
        out("speed %s: %s" % (t, " || ".join(parts)))
    # pairs
    PAIRS = [("n8", "b8", "nvfp4 vs bf16, MAXSEQS 8 (each the planner's split)"), ("x8", "b8", "bf16 expert-heavy (HOSTPOOL) vs bf16 today"),
             ("n8", "x8", "nvfp4 vs bf16 at equal expert VRAM (MAXSEQS 8)"), ("m8", "b8", "bf16 MML 262144 vs MML 524288"),
             ("n16", "b16", "nvfp4 vs bf16, MAXSEQS 16 (same split)")]
    for x, y, lab in PAIRS:
        X, Y = S.get(x), S.get(y)
        if not X or not Y: continue
        cells = []
        for nm, g in (("1u essay", lambda s: s["u1"]["essay"]), ("1u code", lambda s: s["u1"]["code"]),
                      ("c8 agg", lambda s: s["c8"]["agg"]), ("c8 p50", lambda s: s["c8"]["p50"]), ("c8 decode_share", lambda s: s["c8"]["share"]),
                      ("c8 newcomer wall ms", lambda s: s["c8"]["wall"]), ("c8 streamed GB a non-decode step", lambda s: s["c8"]["stream_nd"]),
                      ("c8 vram units", lambda s: s["c8"]["vram_units"]), ("c16 agg", lambda s: s["c16"]["agg"]), ("c16 p50", lambda s: s["c16"]["p50"]),
                      ("c16 decode_share", lambda s: s["c16"]["share"]), ("c16 newcomer wall ms", lambda s: s["c16"]["wall"]),
                      ("c16 streamed GB a non-decode step", lambda s: s["c16"]["stream_nd"]), ("136k TTFT s", lambda s: s["d136"].get("ttft")),
                      ("loaded probe TTFT s", lambda s: s["ld"].get("ttft")), ("loaded c4 agg", lambda s: s["ld"].get("c4_pf_agg_tps"))):
            a_, b_ = g(X), g(Y)
            if isinstance(a_, (int, float)) and isinstance(b_, (int, float)):
                cells.append("%s %s vs %s (%s)" % (nm, f(a_, 3 if "share" in nm or "GB" in nm else 1), f(b_, 3 if "share" in nm or "GB" in nm else 1), pc(a_, b_)))
        if cells: out("pair %s vs %s (%s): %s" % (x, y, lab, " | ".join(cells)))
    # the c16 cell + the >48-row BI check
    for t in ("b16", "n16"):
        if t not in S or not S[t]["c16"]["n"]: continue
        w = S[t]["c16"]
        b = bi.get(t) or {}
        bl = " | ".join("%s: %s steps with > 48 decode rows of %s (most %s; BI stats +%s)" % (k, v.get("steps_gt48"), v.get("steps"), v.get("decode_rows_max"),
                                                                                              v.get("bistats_gt48_delta")) for k, v in sorted(b.items()))
        out("c16 cell %s (latent %s, the first measured 10+ user number): agg %s tok/s, per-stream p50 %s, decode_share %s, newcomer step wall p50 %s ms | >48-row reach: %s" % (
            t, (caps.get(t) or {}).get("lat_lit"), f(w["agg"]), f(w["p50"]), f(w["share"], 3), f(w["wall"]), bl or "-"))
    # identity
    try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError: ident = []
    ok = lambda l: "IDENTICAL" in l and "differ" not in l and "not compared" not in l   # noqa: E731
    strict = [l for l in ident if l.startswith("S:")]
    b48 = [l for l in ident if l.startswith("B48:")]
    info = [l for l in ident if l.startswith("I:")]
    bad = [l for l in strict if not ok(l)]
    out("identity strict (nvfp4 vs bf16 and every answer vs its alone / reference; exact by code): %s" % (
        ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:1500] if strict else "none")))
    reach = {t: sum((v.get("steps_gt48") or 0) for k, v in (bi.get(t) or {}).items() if k.startswith("lid16")) for t in ("b16", "n16")}
    out("identity at MAXSEQS 16 with > 48 decode rows (coordinator 11.11.: BI is claimed at a MAXSEQS only with 20 / 20 there and > 48-row steps reached): %s | reached in lid16: %s" % (
        "; ".join(l.split(" ")[0][4:] + " " + ("IDENTICAL" if ok(l) else "DIFFERS: " + " ".join(l.split(" ")[1:])[:200]) for l in b48) or "none", reach))
    if info: out("identity information (cross-start, labelled): %s" % "; ".join(l[2:][:200] for l in info))
    try: gl = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
    except OSError: gl = []
    gbad = [l for l in gl if " OK" not in l.split("|")[0]]
    out("guards: %s" % (("OK (%d)" % len(gl)) if gl and not gbad else ("VIRHE " + " || ".join(gbad))[:1500] if gl else "none"))
    for l in lines: print(l)
    return 0


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    import shutil, tempfile
    d = tempfile.mkdtemp(prefix="kv4sum-")
    try:
        return selftest_in(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)   # a selftest leaves nothing behind (DRYRUN writes nothing)


def selftest_in(d):
    log = ("radiance: deepseek41 indexer: candidate restriction on from 32768 positions, keys mxfp4 (80 bytes an entry; RADIANCE_DS41_IDXKEY), "
           "latent nvfp4 (288 bytes an entry; RADIANCE_DS41_LATENT), compressor tail bound\n"
           "I kv: the pool backs 100% of the worst case (8 sequences all at 524288 tokens), 8.0 full-length sessions; a larger --host-pool-mib gives the cache more\n"
           "I VRAM budget, per card:\n  activation arena     2.59 GiB   computed\n  elastic             25.62 GiB   the experts\n"
           "    experts           20.90 GiB   82% of it\n    kv cache           3.70 GiB   14% of it\n    prefill staging    1.02 GiB   out of\n"
           "vram_weights        20.00 GiB budget,        0 B used,  20.00 GiB free\nhost_pool       19.13 GiB budget,        0 B used,  19.13 GiB free\n"
           + "".join("I[%d] mover: 2215 slab slots (48 shadow, 0 more kept free out of a loan), 2232 pinned pool slots, 2.65 GiB static vram\n" % r for r in range(4))
           + "I checkpoint slots: 512 linear-state + sliding-window snapshot(s) of 25.33 MiB in pinned host memory (12.66 GiB), none of it on the card\n"
           "I kv: the cache can grow to 3.70 GiB a card -- 4194304 tokens of context, 8.0 sessions at 524288 -- with every block it lends the experts taken back; "
           "1.31 MiB of that is per-sequence state, and up to 3.50 GiB is lent to the experts while no request holds it\n")
    st = {"kv_tokens_total": 4194304, "kv_bytes_carved": 3.7 * GIB, "experts": {"n_units": 3936, "bytes": 38285475840,
          "ranks": [{"rank": r, "vram": 3300, "pinned": 636, "trunk_vram": 3204, "draft_vram": 96, "lent": {"kv": {"held": 300}, "arena": {"held": 200}}} for r in range(4)]},
          "psk": {"mode": 2, "k": 4, "latched": 0, "planned_steps": 0, "cut_passes": 0, "plan_mismatch": 0}}
    open(os.path.join(d, "s.log"), "w").write(log)
    json.dump(st, open(os.path.join(d, "st.json"), "w"))
    ns = argparse.Namespace(tag="n8", lat="nvfp4", ms=8, mml=524288, hp=19456)
    c = parse_cap(log, st, ns)
    assert c["lat_lit"] == "nvfp4" and c["lat_bytes"] == 288 and c["pool_pct"] == 100 and c["kv_tokens"] == 4194304, c
    assert c["exp_gib"] == 20.9 and c["kvc_gib"] == 3.7 and len(c["movers"]) == 4 and not c["guard"], c
    assert c["lent_gib"] == 3.5 and c["ckpt_slots"] == 512 and c["unit_bytes"] == 9727001, c
    c2 = parse_cap(log.replace("latent nvfp4 (288", "latent bf16 (1024"), st, ns)
    assert c2["guard"] and "latent literal" in c2["guard"][0], c2
    b = dict(c, tag="b8", lat="bf16", lat_lit="bf16", exp_gib=17.34, kvc_gib=7.26, bpt=2852.0)
    json.dump(b, open(os.path.join(d, "b8-cap.json"), "w")); json.dump(c, open(os.path.join(d, "n8-cap.json"), "w"))
    # psk
    st1 = json.loads(json.dumps(st)); st1["psk"].update(latched=1, planned_steps=8, cut_passes=7)
    json.dump(st1, open(os.path.join(d, "st1.json"), "w"))
    open(os.path.join(d, "dq.out"), "w").write('DEEPQ {"status": 200, "prompt_tokens": 16123, "cached": 0, "ttft": 5.1}\n')
    # bi48: three steps in the window, one with 96 decode rows (16 x 6), one MIXED 90 + a 1100-row newcomer, one 16-row
    eng = ("2026-11-12T10:00:00.100000000Z step 10 decode n_seq=16 n_tok=96" + "".join("  [%d] slot=%d q=6 ctx=%d" % (i, i, 500 + i) for i in range(16)) + "\n"
           "2026-11-12T10:00:00.200000000Z step 11 MIXED n_seq=16 n_tok=1190" + "".join("  [%d] slot=%d q=6 ctx=%d" % (i, i, 600 + i) for i in range(15))
           + "  [15] slot=15 q=1100 ctx=0\n"
           "2026-11-12T10:00:00.300000000Z radiance: deepseek41: lever BI stats: 1024 segmented steps, 9 MoE passes a layer in them (2.00 a step), 0 cu read-backs, 700 steps with > 48 decode rows; ds41_bi 2\n"
           "2026-11-12T10:00:00.400000000Z step 12 decode n_seq=16 n_tok=16" + "".join("  [%d] slot=%d q=1 ctx=%d" % (i, i, 700 + i) for i in range(16)) + "\n"
           "2026-11-12T10:00:05.000000000Z step 13 decode n_seq=1 n_tok=6  [0] slot=0 q=6 ctx=9\n")
    with gzip.open(os.path.join(d, "e.log.gz"), "wt") as fh: fh.write(eng)
    steps, bis = read_log(os.path.join(d, "e.log.gz"))
    t0 = calendar.timegm(time.strptime("2026-11-12T10:00:00", "%Y-%m-%dT%H:%M:%S"))
    w = bi48_windows(steps, bis, [("lid16-x", t0, t0 + 1.0)], 5)["lid16-x"]
    assert w["steps"] == 3 and w["steps_gt48"] == 2 and w["mixed_gt48"] == 1 and w["decode_rows_max"] == 96 and w["bistats_gt48_delta"] == 700, w
    # report over the synthetic dir
    json.dump({"lid16-a": w}, open(os.path.join(d, "n16-bi48.json"), "w"))
    open(os.path.join(d, "identity.txt"), "w").write("S:a1-n8-vs-a1-b8 IDENTICAL (20 of 20)\nB48:lid16-n16-vs-a1-n16 IDENTICAL (20 of 20)\nI:ld-x DIFFERENT at char 5\n")
    open(os.path.join(d, "guard.txt"), "w").write("n8 psk OK | x\nn8 modes OK\n")
    win = {"mb": {"agg_dec_tps": 170.5, "per_stream_dec_p50": 21.0, "decode_share": 0.79, "ms_decode_step": 36.0, "ttft_p50": 0.7},
           "newcomers": {"found": 30, "mixed": 30, "wall_ms": {"p50": 581.5}}, "classes": {"MIXED": {"wall_ms": {"p50": 580.0}}},
           "stats": {"per_nd": {"link_stream_gb": 7.1, "staged_gb_r0": 0.0, "link_staged_gb": 0.0}, "decode_share": 0.79,
                     "window": {"prefill": 0, "mixed": 30, "rank_stream_gb": [210.0, 212.0, 211.0, 213.0]}},
           "regression": {"link_stream_gb": {"per_nd": 6.9}}, "guard": []}
    json.dump(win, open(os.path.join(d, "n8-c8-1.json"), "w")); json.dump(st, open(os.path.join(d, "n8-c8-1-st-w0.json"), "w"))
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_hp = cmd_hp(argparse.Namespace(b=os.path.join(d, "b8-cap.json"), n=os.path.join(d, "n8-cap.json"), hp=19456))
        rc_psk = cmd_psk(argparse.Namespace(a=os.path.join(d, "st.json"), b=os.path.join(d, "st1.json"), deepq=os.path.join(d, "dq.out")))
        rc_r = cmd_report(argparse.Namespace(dir=d, starts="b8 n8 n16", ref=""))
    txt = buf.getvalue()
    assert rc_hp == 0 and "HP 15811 " in txt, txt
    assert rc_psk == 0 and txt.count("OK | psk mode 2") == 1, txt
    assert rc_r == 0 and "RESULT xkv4 capacity n8" in txt and "identity strict" in txt and "IDENTICAL at every compare (1)" in txt, txt
    assert "capacity by MAXSEQS, nvfp4 latent" in txt and "guards: OK (2)" in txt, txt
    assert "speed n8: c8 (1 window(s)): agg 170.5 tok/s" in txt and "streamed 7.100 GB (by rank [7.0, 7.067, 7.033, 7.1])" in txt and "vram units a rank 3300 [3300, 3300, 3300, 3300]" in txt, txt
    print("SELFTEST OK: cap parse (literal guard, split, pool, movers, lent), hp (15811 MiB for +3.56 GiB), psk guard, bi48 (decode rows, MIXED, BI stats delta), report (capacity, MAXSEQS, prices, a c8 window with per-rank units / streamed GB, identity, guards)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cap"); p.add_argument("--log", required=True); p.add_argument("--stats", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--lat", required=True); p.add_argument("--ms", type=int, required=True); p.add_argument("--mml", type=int, required=True)
    p.add_argument("--hp", type=int, required=True); p.add_argument("--json")
    p = sub.add_parser("hp"); p.add_argument("--b", required=True); p.add_argument("--n", required=True); p.add_argument("--hp", type=int, required=True)
    p = sub.add_parser("psk"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--deepq", required=True)
    p = sub.add_parser("bi48"); p.add_argument("--log", required=True); p.add_argument("--win", action="append", default=[]); p.add_argument("--nst", type=int, default=5)
    p.add_argument("--json")
    p = sub.add_parser("report"); p.add_argument("--dir", required=True); p.add_argument("--starts", required=True); p.add_argument("--ref", default="")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"cap": cmd_cap, "hp": cmd_hp, "psk": cmd_psk, "bi48": cmd_bi48, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
