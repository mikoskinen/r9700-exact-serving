#!/usr/bin/env python3
"""aqlcp.py (13.11.2026, queue-ar/dsmain/zzzaql, a builder for ds-main-port; item 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzaql's
analyser) -- the AQL-stamped CRITICAL-PATH read of the c8 decode step on card 0 (the ha-k1prof / hy-z0prof method: RADIANCE_AQL_STAMP=1
times every dispatch of every pass played on card 0; queue-ar/dsmain/k1prof.py's pass sweep, extended to price the idle by class).
Standard library only.

  aqlcp.py stamps STAMPS.csv[.gz] --win NAME:M0:M1 [...] [--nc 10] [--stall-us 1000] [--json OUT] [--label L]
      RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; card 0, CLOCK_MONOTONIC ns).
      Per window [M0, M1] (monotonic ns): the TRUNK passes (the most common `records` among passes of >= 300 dispatches: the decode
      tape), a DECODE STEP = one trunk pass's start to the next one's (a period over 3 x the median, or a large non-trunk pass in it,
      is a newcomer / MIXED step: not a decode step; the trunk passes after it are counted 1, 2, .. as POST-NEWCOMER steps). Per
      decode step, over every stamped dispatch in [its start, the next trunk start) (its trunk pass + any small pass between):
        busy  = the union of the dispatches' run times;  idle = the rest (card 0 runs nothing of the step);
        each idle interval is priced by the dispatch b that ENDS it and p = the dispatch issued right before b in b's tape (seq):
          dep      same play, same queue, seq(b) = seq(p) + 1: a DEPENDENT-LAUNCH gap (barrier packet -> next start), whole;
          rec      same play, records between (seq jump >= 2: an event record / wait packet) or another queue: the launch floor F
                   is a LAUNCH gap, the excess a LANE / CROSS-QUEUE WAIT (seq jump 2 alone may be a record riding on p: then
                   its gap is ~F and its excess ~0);
          play     b starts another play of the pass (Ctx::play_pass plays a pass in stretches between its HOST OPS): F launch,
                   the excess HOST;
          between  b is the first dispatch of a pass (the next trunk pass or a small pass): F launch, the excess HOST (step
                   scheduling, sampling, drafter host work, the next pass's issue);
          a single idle interval >= --stall-us is OTHER (a stall: counted apart, its position logged).
        F = the median of the window's `dep` gaps (the launch floor).
        AR WAIT = time inside the collectives (r4d_ar_* / all-reduce / XPUSH kernels) above their TRANSFER FLOOR, the 5th percentile of
        that dispatch (same tape position) over the window: card 0 waiting for the slowest peer, busy on the card but idle work.
        GPU busy on the critical path = busy - AR wait; critical-path idle = launch + lane + host + other + AR wait; wall = busy + idle.
      Per window and pooled: steady decode steps vs the first --nc steps after a newcomer (kept separate, and by index 1..nc);
      where the idle sits (class, previous kernel -> next kernel, us a step); the collectives (count, device, floor, wait; the share
      of collectives where card 0 waited <= 2 us above the floor = card 0 the last rank to arrive). Lines "AQLCP ...".
  aqlcp.py clock CLOCKFILE  -> "OFFSET_NS DRIFT_US": CLOCKFILE lines "UTC_S MONO_NS" (one process each: utc and monotonic together)
  aqlcp.py monowin REQS.jsonl CLOCKFILE -> "M0:M1": the harness block's two marks (UTC, time.time()) on the monotonic clock
  aqlcp.py ranks NAME ST0.json ST1.json -> "AQLRANKS {json}": /stats experts.ranks[] deltas per decode step (misses = routed -
      routed_res, routed, stream MB) and steps_decode -- per-rank INFO (the stamps are card 0's only)
  aqlcp.py summary --stamps JSON [--split NAME=P12SPLIT.txt ...] [--twin NAME=P12SPLIT.txt ...] [--ranks AQLRANKS.txt]
                   [--prof2 TEXT] [--ncm NCMATCH.txt] [--min-steps 200] [--label c8]
      the item's lines "AQL ...": the summary ("AQL c8: wall X ms a decode step, critical-path idle Y ms (Z %) = launch gaps .. / lane
      waits .. / host .. / AR wait .. / other .."), the post-newcomer split, the per-rank reading, the stamped vs unstamped step
      (stamped step-log decode ms vs the twin's), prof2 as INFO and the verdict "launch / sync levers worth >= 1 ms: YES / NO"
      (the pool launch + lane + host after removing the WHOLE stamping cost (stamped - unstamped step) from the launch gaps).
  aqlcp.py synth --out CSV --win M0:M1 [...] [--seed 1]   (MOCK ONLY: a small synthetic stamps CSV over the windows, so the mock
      runs every parser; never on a real run)
  aqlcp.py selftest
"""
import bisect, collections, csv, gzip, io, json, math, os, random, re, statistics, sys

AR_RE = re.compile(r"(^|_)ar_|all_?reduce|xpush|arcoded", re.I)


def short(k):
    """the kernel's base name from its mangled symbol (fall back to the raw text)"""
    m = re.match(r"_ZN?(?:12_GLOBAL__N_1)?(?:\d+r4d_moesort12_GLOBAL__N_1)?(\d+)", k)
    if m:
        n = int(m.group(1))
        s = k[m.end():m.end() + n]
        if s:
            return s
    return k[:40]


def is_ar(name):
    return bool(AR_RE.search(name))


def opener(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), newline="")
    return open(path, newline="")


def wins_of(a):
    out = []
    for i, x in enumerate(a):
        if x == "--win":
            n, lo, hi = a[i + 1].split(":")
            out.append((n, int(float(lo)), int(float(hi))))
    return out


def opt(a, k, d=None):
    return a[a.index(k) + 1] if k in a else d


def multi(a, k):
    return [a[i + 1] for i, x in enumerate(a) if x == k]


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(q * (len(v) - 1) + 0.5)))]


# ---------------------------------------------------------------- load
def load(path, W, margin_ns=2_000_000_000):
    """rows per window: pass -> [(s, e, seq, queue, kid, ordered, play, records)]; kernel names interned"""
    names, kid = [], {}
    per = {w[0]: collections.defaultdict(list) for w in W}
    lo_all = min(w[1] for w in W) - margin_ns
    hi_all = max(w[2] for w in W) + margin_ns
    total = untimed = kept = 0
    with opener(path) as f:
        r = csv.reader(f)
        hdr = next(r, None)
        if not hdr or hdr[:2] != ["pass", "seq"]:
            raise SystemExit("aqlcp: %s is not a RADIANCE_AQL_STAMP CSV (header %s)" % (path, hdr))
        for row in r:
            total += 1
            if len(row) < 10:
                continue
            if not row[6] or not row[7]:
                untimed += 1
                continue
            s = int(row[6])
            if s < lo_all or s > hi_all:
                continue
            e = int(row[7])
            k = row[3]
            i = kid.get(k)
            if i is None:
                i = kid[k] = len(names)
                names.append(k)
            t = (s, e, int(row[1]), int(row[2]), i, row[4] == "1", int(row[8]), int(row[9]))
            p = int(row[0])
            for wn, lo, hi in W:
                if lo - margin_ns <= s <= hi + margin_ns:
                    per[wn][p].append(t)
                    kept += 1
    return per, names, {"rows": total, "untimed": untimed, "kept": kept}


# ---------------------------------------------------------------- one window
def analyse(wn, lo, hi, passes, names, nc=10, stall_ns=1_000_000):
    snames = [short(n) for n in names]
    arid = {i for i, n in enumerate(snames) if is_ar(n)}
    P = []
    for p, rows in passes.items():
        s0 = min(x[0] for x in rows)
        if not (lo <= s0 <= hi):
            continue
        P.append({"pass": p, "start": s0, "end": max(x[1] for x in rows), "records": rows[0][7], "n": len(rows), "rows": rows})
    if not P:
        return {"error": "no stamped pass in the window"}
    P.sort(key=lambda x: x["start"])
    big = collections.Counter(x["records"] for x in P if x["n"] >= 300)
    if not big:
        return {"error": "no pass of >= 300 dispatches in the window (no decode tape stamped)"}
    TR = big.most_common(1)[0][0]
    kinds = {str(k): {"passes": sum(1 for x in P if x["records"] == k),
                      "dispatches": statistics.median([x["n"] for x in P if x["records"] == k]),
                      "wall_us": statistics.median([(x["end"] - x["start"]) / 1e3 for x in P if x["records"] == k])}
             for k in sorted({x["records"] for x in P})}
    # every large kind whose median wall is under 2 x the main decode tape's is a decode tape too (another row level); a longer large
    # kind (a recorded MIXED / prefill pass) marks a non-decode step
    trw = kinds[str(TR)]["wall_us"]
    TRS = {k for k in big if kinds[str(k)]["wall_us"] < 2 * trw}
    T = [x for x in P if x["records"] in TRS]
    if len(T) < 3:
        return {"error": "fewer than 3 trunk passes in the window", "kinds": kinds}
    bigother = sorted(x["start"] for x in P if x["records"] not in TRS and x["n"] >= 300)
    small = [x for x in P if x["records"] not in TRS and x["n"] < 300]
    # a pass stamped in part (RADIANCE_AQL_STAMP ran out of signals: "signals are in flight") is not priced (its missing dispatches
    # would read as idle); it still counts as a decode step for the newcomer index
    mn = {k: statistics.median([x["n"] for x in T if x["records"] == k]) for k in TRS}
    for x in T:
        x["partial"] = x["n"] < 0.95 * mn[x["records"]]
    smst = [x["start"] for x in small]
    per_ = [b["start"] - a["start"] for a, b in zip(T, T[1:])]
    med = statistics.median(per_)
    # newcomer boundaries and the post-newcomer index of every trunk pass
    # before the window's first newcomer the index is unknown, but the k-th trunk pass of the window is at least k steps after
    # any newcomer before the window: k > nc -> steady (its index is set to k); the first nc stay unknown (in neither set)
    idx = [None] * len(T)
    nbound = 0
    seen = False
    valid = [False] * (len(T) - 1)
    for i in range(len(T) - 1):
        a, b = T[i], T[i + 1]
        j = bisect.bisect_right(bigother, a["start"])
        has_big = j < len(bigother) and bigother[j] < b["start"]
        valid[i] = (b["start"] - a["start"]) <= 3 * med and not has_big
        if idx[i] is None and not seen and i + 1 > nc:
            idx[i] = i + 1
        if not valid[i]:
            nbound += 1
            seen = True
            idx[i + 1] = 1
        elif idx[i] is not None:
            idx[i + 1] = idx[i] + 1
    # AR floors: per (seq) over every trunk pass of the window
    ard = collections.defaultdict(list)
    for x in T:
        for r in x["rows"]:
            if r[4] in arid:
                ard[r[2]].append(r[1] - r[0])
    arfloor = {s: pct(v, 0.05) for s, v in ard.items()}
    # pass 1: the idle intervals of every decode step, typed
    steps = []
    npartial = 0
    for i in range(len(T) - 1):
        if not valid[i]:
            continue
        if T[i]["partial"]:
            npartial += 1
            continue
        a, b = T[i], T[i + 1]
        S, E = a["start"], b["start"]
        D = list(a["rows"])
        lo_i = bisect.bisect_left(smst, S)
        hi_i = bisect.bisect_left(smst, E)
        nsmall = 0
        for sp in small[lo_i:hi_i]:
            D.extend(sp["rows"])
            nsmall += 1
        # previous dispatch in issue order (seq) within its own pass
        prev = {}
        # group by pass identity: rows of `a` and of each small pass
        groups = [a["rows"]] + [sp["rows"] for sp in small[lo_i:hi_i]]
        for g in groups:
            gs = sorted(g, key=lambda r: r[2])
            for k, r in enumerate(gs):
                prev[id(r)] = gs[k - 1] if k else None
        D.sort(key=lambda r: (r[0], r[1]))
        run_end = S
        gaps = []   # (type, g, prev kernel, next kernel)
        for r in D:
            if r[0] >= E:
                break
            if r[0] > run_end:
                g = r[0] - run_end
                p = prev[id(r)]
                if p is None:
                    t = "between"
                elif r[6] != p[6]:
                    t = "play"
                elif r[3] != p[3] or r[2] - p[2] >= 2:
                    t = "rec"
                else:
                    t = "dep"
                gaps.append((t, g, p[4] if p is not None else -1, r[4]))
            run_end = max(run_end, r[1])
        if E > run_end:
            gaps.append(("between", E - run_end, D[-1][4] if D else -1, b["rows"][0][4] if b["rows"] else -1))
        idle = sum(g[1] for g in gaps)
        ar_dev = ar_fl = ar_wait = 0
        ar_n = ar_last = 0
        for r in a["rows"]:
            if r[4] in arid:
                d = r[1] - r[0]
                f = arfloor.get(r[2], d)
                ar_dev += d
                ar_fl += min(f, d)
                w = max(0, d - f)
                ar_wait += w
                ar_n += 1
                ar_last += int(w <= 2000)
        steps.append({"i": i, "idx": idx[i], "period": E - S, "wall": a["end"] - a["start"], "idle": idle,
                      "busy": (E - S) - idle, "gaps": gaps, "ar_dev": ar_dev, "ar_floor": ar_fl, "ar_wait": ar_wait,
                      "ar_n": ar_n, "ar_last": ar_last, "nsmall": nsmall, "disp": a["n"]})
    if not steps:
        return {"error": "no decode step (every trunk-to-trunk period is a newcomer boundary)", "kinds": kinds}
    deps = [g[1] for s in steps for g in s["gaps"] if g[0] == "dep" and g[1] < 50_000]
    F = statistics.median(deps) if deps else 4000
    # pass 2: price every step
    where = collections.defaultdict(float)
    stalls = []
    for s in steps:
        c = {"launch": 0.0, "lane": 0.0, "host": 0.0, "other": 0.0}
        n = collections.Counter()
        hb = hp = 0.0
        for t, g, pk, nk in s["gaps"]:
            n[t] += 1
            if g >= stall_ns:
                c["other"] += g
                stalls.append({"type": t, "us": round(g / 1e3, 1), "prev": snames[pk] if pk >= 0 else "-", "next": snames[nk] if nk >= 0 else "-"})
                continue
            if t == "dep":
                c["launch"] += g
                cls = "launch"
            else:
                fl = min(g, F)
                c["launch"] += fl
                ex = g - fl
                cls = "lane" if t == "rec" else "host"
                c[cls] += ex
                if t == "play":
                    hp += ex
                elif t == "between":
                    hb += ex
            if s["idx"] is None or s["idx"] > nc:
                where[(t, snames[pk] if pk >= 0 else "-", snames[nk] if nk >= 0 else "-")] += g
        s["c"] = c
        s["n"] = dict(n)
        s["host_play"] = hp
        s["host_between"] = hb
        s["cp_idle"] = c["launch"] + c["lane"] + c["host"] + c["other"] + s["ar_wait"]
        s["gpu_cp"] = s["busy"] - s["ar_wait"]
        del s["gaps"]
    steady = [s for s in steps if s["idx"] is not None and s["idx"] > nc]
    post = [s for s in steps if s["idx"] is not None and s["idx"] <= nc]
    unknown = [s for s in steps if s["idx"] is None]
    nsteady = len(steady)
    out = {"records": TR, "trunk_kinds": sorted(TRS), "partial_steps_skipped": npartial, "kinds": kinds, "trunk_passes": len(T), "decode_steps": len(steps), "newcomer_boundaries": nbound,
           "median_period_us": med / 1e3, "launch_floor_us": F / 1e3, "nc": nc,
           "steady": agg(steady), "post": agg(post), "before_first_newcomer": agg(unknown),
           "post_by_index": {str(k): agg([s for s in post if s["idx"] == k]) for k in range(1, nc + 1)},
           "where_us_step": [{"type": k[0], "prev": k[1], "next": k[2], "us_step": round(v / 1e3 / max(1, nsteady + len(unknown)), 2)}
                             for k, v in sorted(where.items(), key=lambda kv: -kv[1])[:12]],
           "stalls": stalls[:20], "stall_count": len(stalls),
           "ar_families": sorted({snames[i] for i in arid}),
           "steps": [{k: (round(v / 1e3, 1) if isinstance(v, (int, float)) and k in ("period", "wall", "idle", "busy", "ar_wait", "cp_idle", "gpu_cp") else v)
                      for k, v in s.items() if k in ("idx", "period", "wall", "idle", "busy", "ar_wait", "cp_idle", "gpu_cp")} for s in steps]}
    return out


def agg(S):
    """means (and a few medians) in ms a step over a set of priced decode steps"""
    if not S:
        return {"steps": 0}
    m = lambda f: statistics.mean(f(s) for s in S) / 1e6
    md = lambda f: statistics.median(f(s) for s in S) / 1e6
    n = collections.Counter()
    for s in S:
        n.update(s["n"])
    arn = sum(s["ar_n"] for s in S)
    return {"steps": len(S), "wall_ms": m(lambda s: s["period"]), "wall_p50_ms": md(lambda s: s["period"]),
            "trunk_wall_ms": m(lambda s: s["wall"]), "busy_ms": m(lambda s: s["busy"]), "idle_ms": m(lambda s: s["idle"]),
            "gpu_cp_ms": m(lambda s: s["gpu_cp"]), "cp_idle_ms": m(lambda s: s["cp_idle"]),
            "cp_idle_p50_ms": md(lambda s: s["cp_idle"]),
            "launch_ms": m(lambda s: s["c"]["launch"]), "lane_ms": m(lambda s: s["c"]["lane"]), "host_ms": m(lambda s: s["c"]["host"]),
            "host_play_ms": m(lambda s: s["host_play"]), "host_between_ms": m(lambda s: s["host_between"]),
            "other_ms": m(lambda s: s["c"]["other"]), "ar_wait_ms": m(lambda s: s["ar_wait"]), "ar_dev_ms": m(lambda s: s["ar_dev"]),
            "ar_floor_ms": m(lambda s: s["ar_floor"]), "ar_n_step": arn / len(S),
            "ar_last_share": (sum(s["ar_last"] for s in S) / arn) if arn else None,
            "gaps_step": {k: v / len(S) for k, v in n.items()}, "small_passes_step": statistics.mean(s["nsmall"] for s in S),
            "dispatches": statistics.median(s["disp"] for s in S)}


def pool(res, key):
    """the windows' sets pooled: a step-weighted mean of every *_ms field"""
    sets = [r[key] for r in res.values() if isinstance(r, dict) and key in r and r[key].get("steps")]
    N = sum(s["steps"] for s in sets)
    if not N:
        return {"steps": 0}
    o = {"steps": N}
    for k in sets[0]:
        if k.endswith("_ms") or k in ("ar_n_step", "small_passes_step", "dispatches"):
            vals = [(s[k], s["steps"]) for s in sets if s.get(k) is not None]
            o[k] = sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None
    sh = [(s["ar_last_share"], s["ar_n_step"] * s["steps"]) for s in sets if s.get("ar_last_share") is not None]
    o["ar_last_share"] = sum(v * w for v, w in sh) / sum(w for _, w in sh) if sh else None
    g = collections.Counter()
    for s in sets:
        for k, v in s.get("gaps_step", {}).items():
            g[k] += v * s["steps"]
    o["gaps_step"] = {k: v / N for k, v in g.items()}
    return o


def f2(x, d=2):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "-"


def splitline(a):
    if not a.get("steps"):
        return "no decode step"
    return ("%d steps: wall %s ms a decode step (p50 %s), GPU busy on the critical path %s ms, critical-path idle %s ms (%s %%) = launch gaps %s / "
            "lane waits %s / host %s (play boundaries %s + between passes %s) / AR wait %s / other %s | collectives %s a step: device %s ms, floor %s, "
            "card 0 last to arrive at %s %% | idle gaps a step %s | trunk wall %s ms, %s dispatches, small passes %s a step"
            % (a["steps"], f2(a["wall_ms"]), f2(a.get("wall_p50_ms")), f2(a["gpu_cp_ms"]), f2(a["cp_idle_ms"]),
               f2(100 * a["cp_idle_ms"] / a["wall_ms"], 1), f2(a["launch_ms"]), f2(a["lane_ms"]), f2(a["host_ms"]), f2(a["host_play_ms"]),
               f2(a["host_between_ms"]), f2(a["ar_wait_ms"]), f2(a["other_ms"]), f2(a["ar_n_step"], 1), f2(a["ar_dev_ms"]), f2(a["ar_floor_ms"]),
               f2(100 * a["ar_last_share"], 1) if a.get("ar_last_share") is not None else "-",
               json.dumps({k: round(v, 1) for k, v in sorted(a.get("gaps_step", {}).items())}), f2(a.get("trunk_wall_ms")),
               f2(a.get("dispatches"), 0), f2(a.get("small_passes_step"), 2)))


def stamps_cmd(a):
    path = a[1]
    W = wins_of(a)
    if not W:
        raise SystemExit("aqlcp stamps: no --win")
    nc = int(opt(a, "--nc", 10))
    stall = int(float(opt(a, "--stall-us", 1000)) * 1000)
    per, names, cnt = load(path, W)
    res = {}
    for wn, lo, hi in W:
        res[wn] = analyse(wn, lo, hi, per[wn], names, nc=nc, stall_ns=stall)
    o = {"windows": res, "counts": cnt, "pooled": {"steady": pool(res, "steady"), "post": pool(res, "post")}, "nc": nc}
    label = opt(a, "--label", "")
    if "--json" in a:
        with open(opt(a, "--json"), "w") as f:
            json.dump(o, f)
    print("AQLCP %s file: %d rows (%d untimed), %d kept in the windows" % (label, cnt["rows"], cnt["untimed"], cnt["kept"]))
    for wn, r in res.items():
        if "error" in r:
            print("AQLCP %s %s: %s" % (label, wn, r["error"]))
            continue
        print("AQLCP %s %s: trunk records %s (decode kinds %s), %d trunk passes, %d decode steps priced (%d stamped in part, skipped), %d newcomer boundaries, median period %.0f us, launch floor F %.2f us | kinds %s | AR families %s"
              % (label, wn, r["records"], r["trunk_kinds"], r["trunk_passes"], r["decode_steps"], r["partial_steps_skipped"], r["newcomer_boundaries"], r["median_period_us"],
                 r["launch_floor_us"], json.dumps(r["kinds"])[:600], ",".join(r["ar_families"])[:300]))
        print("AQLCP %s %s steady (> %d steps after a newcomer): %s" % (label, wn, nc, splitline(r["steady"])))
        print("AQLCP %s %s post-newcomer (steps 1..%d): %s" % (label, wn, nc, splitline(r["post"])))
        if r["before_first_newcomer"].get("steps"):
            print("AQLCP %s %s the window's first %d steps before its first newcomer (index unknown, in neither set): %d steps, wall %s ms" % (label, wn, nc, r["before_first_newcomer"]["steps"], f2(r["before_first_newcomer"]["wall_ms"])))
        print("AQLCP %s %s where the idle sits (steady + unknown, us a step): %s" % (label, wn, "; ".join("%s %s -> %s %.1f" % (w["type"], w["prev"], w["next"], w["us_step"]) for w in r["where_us_step"][:10])))
        if r["stall_count"]:
            print("AQLCP %s %s stalls >= %d us: %d (%s)" % (label, wn, stall // 1000, r["stall_count"], "; ".join("%s %s->%s %.0f us" % (s["type"], s["prev"], s["next"], s["us"]) for s in r["stalls"][:6])))
    for k in ("steady", "post"):
        print("AQLCP %s pooled %s: %s" % (label, k, splitline(o["pooled"][k])))
    print("AQLCP " + json.dumps({"label": label, "pooled": o["pooled"]})[:20000])
    return 0


# ---------------------------------------------------------------- clocks, windows, ranks
def clock_cmd(a):
    pts = []
    for l in open(a[1]):
        f = l.split()
        if len(f) >= 2:
            pts.append(int(f[1]) - int(round(float(f[0]) * 1e9)))
    if not pts:
        raise SystemExit("aqlcp clock: no pair in %s" % a[1])
    off = sum(pts) / len(pts)
    print("%d %.1f" % (off, (max(pts) - min(pts)) / 1e3))
    return 0


def monowin_cmd(a):
    m = sorted(json.loads(l)["t"] for l in open(a[1]) if l.strip() and json.loads(l).get("ev") == "mark")
    if len(m) < 2:
        raise SystemExit("aqlcp monowin: fewer than 2 marks in %s" % a[1])
    pts = []
    for l in open(a[2]):
        f = l.split()
        if len(f) >= 2:
            pts.append(int(f[1]) - int(round(float(f[0]) * 1e9)))
    if not pts:
        raise SystemExit("aqlcp monowin: no clock pair in %s" % a[2])
    off = sum(pts) / len(pts)
    print("%d:%d" % (int(m[0] * 1e9 + off), int(m[1] * 1e9 + off)))
    return 0


def ranks_cmd(a):
    name, p0, p1 = a[1], a[2], a[3]
    try:
        s0, s1 = json.load(open(p0)), json.load(open(p1))
    except (OSError, ValueError) as e:
        print("AQLRANKS " + json.dumps({"name": name, "error": str(e)[:200]}))
        return 0
    steps = (s1.get("steps_decode") or 0) - (s0.get("steps_decode") or 0)
    R0 = {r.get("rank"): r for r in (s0.get("experts") or {}).get("ranks") or []}
    out = []
    for r in (s1.get("experts") or {}).get("ranks") or []:
        q = R0.get(r.get("rank"), {})
        d = lambda k: (r.get(k) or 0) - (q.get(k) or 0)
        routed, res = d("routed"), d("routed_res")
        out.append({"rank": r.get("rank"), "misses_step": (routed - res) / steps if steps else None,
                    "routed_step": routed / steps if steps else None, "stream_mb_step": d("stream_bytes") / 1e6 / steps if steps else None,
                    "hit": res / routed if routed else None, "vram": r.get("vram")})
    print("AQLRANKS " + json.dumps({"name": name, "decode_steps": steps, "ranks": out}))
    return 0


# ---------------------------------------------------------------- summary
SPLIT_RE = re.compile(r"\(a\) decode (\d+) steps (\d+) ms = [^,]+, ([\d.]+) ms a step")


def read_split(p):
    try:
        for l in open(p):
            m = SPLIT_RE.search(l)
            if m and l.startswith("P12SPLIT"):
                return int(m.group(1)), float(m.group(2)), float(m.group(3))
    except OSError:
        pass
    return None


def wmean(v):
    v = [x for x in v if x]
    n = sum(x[0] for x in v)
    return (sum(x[1] for x in v) / n if n else None), n


def summary_cmd(a):
    label = opt(a, "--label", "c8")
    try:
        o = json.load(open(opt(a, "--stamps")))
    except (OSError, ValueError, TypeError) as e:
        print("AQL %s: VIRHE no stamps analysis (%s)" % (label, str(e)[:200]))
        return 0
    minst = int(opt(a, "--min-steps", 200))
    S, Pn = o["pooled"]["steady"], o["pooled"]["post"]
    sp = [(n, read_split(f)) for n, f in (x.split("=", 1) for x in multi(a, "--split"))]
    tw = [(n, read_split(f)) for n, f in (x.split("=", 1) for x in multi(a, "--twin"))]
    st_ms, st_n = wmean([(x[0], x[1]) for _, x in sp if x])
    tw_ms, tw_n = wmean([(x[0], x[1]) for _, x in tw if x])
    d = (st_ms - tw_ms) if (st_ms is not None and tw_ms is not None) else None
    prof2 = opt(a, "--prof2", "")
    tot = S.get("steps", 0) + Pn.get("steps", 0)
    if not S.get("steps"):
        print("AQL %s: VIRHE no steady decode step priced (%s)" % (label, json.dumps({k: v.get("error") for k, v in o["windows"].items() if isinstance(v, dict) and "error" in v})[:400]))
        return 0
    if tot < minst:
        print("AQL %s: VIRHE only %d decode steps stamped (want >= %d)" % (label, tot, minst))
    W = S["wall_ms"]
    line = ("AQL %s: wall %.2f ms a decode step, critical-path idle %.2f ms (%.1f %%) = launch gaps %.2f / lane waits %.2f / host %.2f / AR wait %.2f / other %.2f"
            % (label, W, S["cp_idle_ms"], 100 * S["cp_idle_ms"] / W, S["launch_ms"], S["lane_ms"], S["host_ms"], S["ar_wait_ms"], S["other_ms"]))
    line += (" | GPU busy on the critical path %.2f ms | card 0, STAMPED, %d steady decode steps (> %d after a newcomer) of %d stamped"
             % (S["gpu_cp_ms"], S["steps"], o.get("nc", 10), tot))
    if d is not None:
        line += (" | step log decode ms: stamped %.2f (%d steps) vs unstamped twin %.2f (%d steps, %s) = %+.2f ms (%+.1f %%: %s)"
                 % (st_ms, st_n, tw_ms, tw_n, ",".join(n for n, x in tw if x), d, 100 * d / tw_ms,
                    "the stamping cost (cross-start)" if d >= 0 else "stamped NOT slower: no stamping cost measurable cross-start, none removed"))
    elif st_ms is not None:
        line += " | step log decode ms stamped %.2f (%d steps); unstamped twin: none read" % (st_ms, st_n)
    if prof2:
        line += " | INFO prof2 (PROFILED, --profile-ops, no recorded pass): " + prof2
    print(line)
    if Pn.get("steps"):
        print("AQL %s post-newcomer (the %d decode steps right after a newcomer, %d steps): wall %.2f ms a decode step (steady %.2f, %+.2f), critical-path idle %.2f ms (%.1f %%) = launch gaps %.2f / lane waits %.2f / host %.2f / AR wait %.2f / other %.2f | GPU busy on the critical path %.2f ms"
              % (label, o.get("nc", 10), Pn["steps"], Pn["wall_ms"], W, Pn["wall_ms"] - W, Pn["cp_idle_ms"], 100 * Pn["cp_idle_ms"] / Pn["wall_ms"],
                 Pn["launch_ms"], Pn["lane_ms"], Pn["host_ms"], Pn["ar_wait_ms"], Pn["other_ms"], Pn["gpu_cp_ms"]))
        byi = collections.defaultdict(list)
        for r in o["windows"].values():
            for k, v in (r.get("post_by_index") or {}).items():
                if v.get("steps"):
                    byi[int(k)].append(v)
        curve = []
        for k in sorted(byi):
            n = sum(v["steps"] for v in byi[k])
            curve.append("%d: %.2f/%.2f" % (k, sum(v["wall_ms"] * v["steps"] for v in byi[k]) / n, sum(v["cp_idle_ms"] * v["steps"] for v in byi[k]) / n))
        print("AQL %s post-newcomer by index (step: wall / critical-path idle ms): %s" % (label, " ".join(curve)))
    else:
        print("AQL %s post-newcomer: no newcomer-adjacent decode step priced" % label)
    # per-rank
    rk = []
    rp = opt(a, "--ranks")
    if rp:
        try:
            for l in open(rp):
                if l.startswith("AQLRANKS "):
                    rk.append(json.loads(l[9:]))
        except (OSError, ValueError):
            pass
    agg_r = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for x in rk:
        n = x.get("decode_steps") or 0
        for r in x.get("ranks") or []:
            if r.get("misses_step") is None:
                continue
            q = agg_r[r["rank"]]
            q[0] += r["misses_step"] * n
            q[1] += (r.get("stream_mb_step") or 0) * n
            q[2] += n
    rtxt = " ".join("r%s %.1f / %.0f MB" % (k, v[0] / v[2], v[1] / v[2]) for k, v in sorted(agg_r.items()) if v[2])
    slow = max(agg_r.items(), key=lambda kv: kv[1][0] / kv[1][2] if kv[1][2] else -1)[0] if agg_r else None
    ls = S.get("ar_last_share")
    print("AQL %s per-rank: card 0 is the only stamped rank (RADIANCE_AQL_STAMP times card 0's passes: aql.cpp StampRun::begin skips a tape whose first stream is not ordinal 0) -- card 0's split above; "
          "card 0 waits %.2f ms a step in %.1f collectives above their transfer floor (= how far the slowest peer runs behind it) and is the LAST to arrive (<= 2 us above the floor) at %s %% of them "
          "(an even share of 4 ranks is 25 %%) -> %s | /stats per rank over the stamped c8 blocks (misses / stream a decode step, INFO): %s%s"
          % (label, S["ar_wait_ms"], S["ar_n_step"] or 0, f2(100 * ls, 1) if ls is not None else "-",
             ("card 0 is the slowest rank at most collectives" if ls is not None and ls >= 0.5 else
              "the slowest rank is a peer at %s %% of the collectives, card 0 at %s %%%s" % (f2(100 * (1 - ls), 1), f2(100 * ls, 1),
              " (card 0 the straggler more often than an even share)" if ls > 0.3 else (" (card 0 less often than an even share)" if ls < 0.2 else "")) if ls is not None else "no collective read"),
             rtxt or "none read", (" -> most misses: rank %s" % slow) if slow is not None else ""))
    # where
    wh = collections.defaultdict(float)
    nst = 0
    for r in o["windows"].values():
        if not isinstance(r, dict) or "error" in r:
            continue
        n = r["steady"].get("steps", 0) + r["before_first_newcomer"].get("steps", 0)
        nst += n
        for w in r.get("where_us_step") or []:
            wh[(w["type"], w["prev"], w["next"])] += w["us_step"] * n
    if nst:
        print("AQL %s where the idle sits (us a steady step, by the gap's type and prev -> next kernel): %s"
              % (label, "; ".join("%s %s -> %s %.0f" % (k[0], k[1], k[2], v / nst) for k, v in sorted(wh.items(), key=lambda kv: -kv[1])[:10])))
    # verdict
    pool_ = S["launch_ms"] + S["lane_ms"] + S["host_ms"]
    corr = pool_ - max(0.0, d) if d is not None else None
    use = corr if corr is not None else pool_
    yes = use >= 1.0
    print("AQL %s verdict: launch / sync levers worth >= 1 ms: %s -- the idle such levers address (launch gaps + lane waits + host) %.2f ms a step stamped%s; "
          "AR wait %.2f ms a step is rank skew (balance levers, not launch / sync); other %.2f ms (stalls >= 1 ms)%s"
          % (label, "YES" if yes else "NO", pool_,
             (", %.2f ms after removing the whole stamping cost (%+.2f ms) from the launch gaps (the conservative figure, used)" % (corr, d)) if corr is not None else " (no unstamped twin: the stamped figure is used, an upper bound)",
             S["ar_wait_ms"], S["other_ms"],
             "" if not Pn.get("steps") else "; post-newcomer steps %.2f ms" % (Pn["launch_ms"] + Pn["lane_ms"] + Pn["host_ms"] - (max(0.0, d) if d is not None else 0))))
    ncm = opt(a, "--ncm")
    if ncm:
        try:
            for l in open(ncm):
                if l.startswith("NCMATCH") and "matched prompts" in l:
                    print("AQL %s INFO newcomer steps (stamped, ncmatch): %s" % (label, l.strip()[8:600]))
        except OSError:
            pass
    return 0


# ---------------------------------------------------------------- mock data and selftest
def synth_cmd(a):
    out = opt(a, "--out")
    W = [(n, lo, hi) for n, lo, hi in wins_of(a)]
    rnd = random.Random(int(opt(a, "--seed", 1)))
    K = ["_ZN12_GLOBAL__N_112gg_mr_kernelILi2ELi2ELi1ELi6EtLi0EEEvNS_8GgMrArgsE", "_Z19exl3r4d_gemm_kernelILi21ELi2ELi2ELb0ELb0ELi0ELb0EEv11Exl3r4dGemm",
         "_Z32r4d_ar_oneshot_wide_exact_kernelI14__hip_bfloat16Li8ELi4EEvPK15HIP_vector_typeIiLj4EE", "_ZN12_GLOBAL__N_116mix_split_kernelILi4ELi1ELi8ELi8ELb0EEEvPKtlPKflPfi",
         "_ZN12_GLOBAL__N_110aql_copy16EPKDv4_jPS0_m", "_ZN12_GLOBAL__N_119norm_rope_rt_kernelILb1ELj1024ELb1ELb0EEEvPKtjPKvPt"]
    pn = 100
    with open(out, "w") as f:
        f.write("pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records\n")
        for wn, lo, hi in W:
            t = lo - 200_000_000
            k = 0
            while t < hi + 200_000_000:
                k += 1
                pn += 1
                if k % 60 == 0:   # a newcomer: an unstamped ~500 ms MIXED step
                    t += 500_000_000
                    continue
                s = t
                play = 0
                for i in range(320):
                    kern = K[i % len(K)]
                    if i % 12 == 11:
                        play += 1
                        s += 6000 + rnd.randint(0, 3000)
                    else:
                        s += 4000 + rnd.randint(0, 600)
                    d = 70_000 + rnd.randint(0, 20000) if i % len(K) != 2 else 20000 + rnd.randint(0, 60000)
                    f.write("%d,%d,1035,%s,%d,0,%d,%d,%d,350\n" % (pn, i + (i // 12), kern, 1, s, s + d, play))
                    s += d
                t = s + 200_000
                if k % 7 == 0:   # a small pass
                    pn += 1
                    for i in range(6):
                        t += 5000
                        f.write("%d,%d,1035,%s,1,0,%d,%d,0,18\n" % (pn, i, K[0], t, t + 30000))
                        t += 30000
                    t += 100_000
    print("AQLCP synth: MOCK stamps (synthetic, not a reading) -> %s" % out)
    return 0


def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="aqlcp-")
    p = os.path.join(d, "s.csv")
    lo, hi = 10_000_000_000, 14_000_000_000
    synth_cmd(["synth", "--out", p, "--win", "w:%d:%d" % (lo, hi)])
    per, names, cnt = load(p, [("w", lo, hi)])
    r = analyse("w", lo, hi, per["w"], names, nc=10)
    assert "error" not in r, r
    assert r["records"] == 350 and r["decode_steps"] > 50 and r["newcomer_boundaries"] >= 1, (r["records"], r["decode_steps"], r["newcomer_boundaries"])
    s = r["steady"]
    assert s["steps"] > 20 and abs(s["wall_ms"] - (s["busy_ms"] + s["idle_ms"])) < 1e-6, s
    assert abs(s["cp_idle_ms"] - (s["launch_ms"] + s["lane_ms"] + s["host_ms"] + s["other_ms"] + s["ar_wait_ms"])) < 1e-6
    assert abs(s["wall_ms"] - (s["gpu_cp_ms"] + s["cp_idle_ms"])) < 1e-6
    assert 3.5 < r["launch_floor_us"] < 5.0 and s["host_play_ms"] > 0 and s["ar_wait_ms"] > 0 and s["lane_ms"] > 0, (r["launch_floor_us"], s)
    assert r["post"]["steps"] > 0 and set(r["post_by_index"]) == {str(k) for k in range(1, 11)}
    j = os.path.join(d, "a.json")
    with open(j, "w") as f:
        json.dump({"windows": {"w": r}, "pooled": {"steady": pool({"w": r}, "steady"), "post": pool({"w": r}, "post")}, "nc": 10}, f)
    sp = os.path.join(d, "split.txt")
    with open(sp, "w") as f:
        f.write("P12SPLIT c8-1-S-O (window 35.0 s): (a) decode 805 steps 27823 ms = 79.5 %, 34.56 ms a step, 8.00 rows a step\n")
    sp2 = os.path.join(d, "split2.txt")
    with open(sp2, "w") as f:
        f.write("P12SPLIT c8-1-S-O (window 35.0 s): (a) decode 800 steps 29600 ms = 79.5 %, 37.00 ms a step, 8.00 rows a step\n")
    assert read_split(sp) == (805, 27823.0, 34.56)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        summary_cmd(["summary", "--stamps", j, "--split", "c8-1=" + sp2, "--twin", "ref14-c8-1=" + sp, "--prof2", "x", "--min-steps", "10"])
    finally:
        sys.stdout = old
    t = buf.getvalue()
    assert "AQL c8: wall " in t and "critical-path idle" in t and "verdict: launch / sync levers worth >= 1 ms: " in t and "the stamping cost" in t, t
    assert "VIRHE" not in t, t
    c = os.path.join(d, "clock.txt")
    with open(c, "w") as f:
        f.write("100.000000 1000000000\n101.000000 2000000100\n")
    rq = os.path.join(d, "reqs.jsonl")
    with open(rq, "w") as f:
        f.write(json.dumps({"ev": "mark", "t": 100.5}) + "\n" + json.dumps({"ev": "mark", "t": 100.75}) + "\n")
    buf = io.StringIO()
    sys.stdout = buf
    try:
        monowin_cmd(["monowin", rq, c])
    finally:
        sys.stdout = old
    m0, m1 = (int(x) for x in buf.getvalue().strip().split(":"))
    assert abs(m0 - 1_500_000_050) < 10 and abs(m1 - 1_750_000_050) < 10, (m0, m1)
    assert short("_ZN12_GLOBAL__N_112gg_mr_kernelILi2E") == "gg_mr_kernel" and is_ar("r4d_ar_gather_twoshot_kernel") and not is_ar("gg_mr_kernel")
    for x in os.listdir(d):
        os.remove(os.path.join(d, x))
    os.rmdir(d)
    print("SELFTEST OK aqlcp.py (synthetic stamps: %d decode steps, F %.2f us, steady idle %.2f ms of %.2f)" % (r["decode_steps"], r["launch_floor_us"], s["cp_idle_ms"], s["wall_ms"]))
    return 0


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    c = a[0]
    if c == "selftest":
        return selftest()
    if c == "stamps" and len(a) >= 2:
        return stamps_cmd(a)
    if c == "clock" and len(a) >= 2:
        return clock_cmd(a)
    if c == "monowin" and len(a) >= 3:
        return monowin_cmd(a)
    if c == "ranks" and len(a) >= 4:
        return ranks_cmd(a)
    if c == "summary":
        return summary_cmd(a)
    if c == "synth":
        return synth_cmd(a)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
