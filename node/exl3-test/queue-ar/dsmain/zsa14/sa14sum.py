#!/usr/bin/env python3
"""sa14sum.py (12.11.2026, queue-ar/dsmain/zsa14, a builder for ds-main-port) -- the zsa14 summary: lever SA (fn4 stage_short N +
stage_pick K: the prefill stager armed for newcomer-sized steps at the span's shared level) re-read on int14 with the fixed method
(TESTING.md 12.11. "WINDOW QUANTUM"): WORK-based c8 / c16 blocks (harness gens window: the same requests in every block, agg =
tokens / time), newcomer readings on MATCHED-PROMPT sums (ref14/ncmatch.py's json for c8; for c16 this file's own tie, because the
engine prints only a step's first 8 plan entries and the c16 newcomer is planned after 15 decode rows), decode-step p50 over ALL
decode steps of the window and over the POST decode steps right after each newcomer step (the refill-miss risk: an armed step
recalls ~1 GiB of arena loan a rank), the link / slot counters of the window (/stats at the marks) and lever SA's GET "sahb"
counters of the block (staged bytes a step), and a steady-state cycle model per arm (decode ms x steps + matched newcomer ms).

  sa14sum.py OUT --log ENGINE_LOG [--sw 1] [--post 10] [--base B] [--c8gens 2] [--c16gens 1] [--json OUT.json]
      OUT = the item's out dir: mb.jsonl (mbx's rows), c8/NAME-reqs.jsonl, c8/NAME-st-w0.json / -st-w1.json (/stats at the two
      window marks), fn-NAME-0.json / fn-NAME-1.json (GET /admin/fn4 around the block), ncm-c8g.json (ncmatch.py, optional).
      Only the gens blocks (sched G) are read. Prints lines "SA14 ..."; a guard failure is a "VIRHE" inside its line.
  sa14sum.py selftest
Standard library only. Read only (writes only --json)."""
import bisect, gzip, io, json, os, re, statistics, sys, tempfile, zlib
from datetime import datetime, timezone

TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")
STEP = re.compile(r" step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
SEG = re.compile(r"\[\d+\] slot=(\d+) q=(\d+) ctx=(\d+)")
QMIN = 16      # a prefill segment has more rows than any decode / verify segment (draft depth <= 15)
NCROWS = 512   # a non-decode step of >= 512 rows is a newcomer step even when its prefill entry is not printed (c16: entry > 8)
IDLE = 3.0     # a step followed by > 3 s of nothing is a host gap, not a wall


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


def read_steps(path):
    out = []
    for l in opn(path):
        m = TSRE.match(l)
        if not m or " step " not in l: continue
        s = STEP.search(l.rstrip("\n"))
        if not s: continue
        out.append((tsec(m), int(s.group(1)), s.group(2), int(s.group(3)), int(s.group(4)), [tuple(int(x) for x in g) for g in SEG.findall(s.group(5))]))
    return out


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


def is_nc(s):
    return s[2] != "decode" and (any(g[1] > QMIN for g in s[5]) or s[4] >= NCROWS)


def wall(steps, j):
    if j is None or j + 1 >= len(steps): return None
    d = steps[j + 1][0] - steps[j][0]
    return 1000.0 * d if 0.0 <= d <= IDLE else None


def p50(v): return round(statistics.median(v), 2) if v else None
def mean(v): return round(statistics.mean(v), 2) if v else None
def f(v, n=2): return ("%." + str(n) + "f") % v if isinstance(v, (int, float)) else "-"
def pc(a, b): return 100.0 * (a / b - 1.0) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None
def fp(v): return ("%+.2f %%" % v) if isinstance(v, (int, float)) else "-"


def window_steps(steps, ts, w0, w1, post):
    """decode walls of the window (all / the POST decode steps right after each newcomer step / the rest) + its newcomer steps"""
    i0, i1 = bisect.bisect_left(ts, w0), bisect.bisect_right(ts, w1)
    nc, postidx = [], set()
    for j in range(i0, i1):
        if not is_nc(steps[j]): continue
        nc.append(j)
        c, k = 0, j + 1
        while k < len(steps) and c < post:
            if is_nc(steps[k]): break
            if steps[k][2] == "decode": postidx.add(k); c += 1
            k += 1
    dec = [(j, wall(steps, j)) for j in range(i0, i1) if steps[j][2] == "decode"]
    dec = [(j, w) for j, w in dec if w is not None]
    allw = [w for _, w in dec]
    pw = [w for w in (wall(steps, k) for k in sorted(postidx)) if w is not None]
    ow = [w for j, w in dec if j not in postidx]
    ncw = [w for w in (wall(steps, j) for j in nc) if w is not None]
    return dict(n_dec=len(allw), dec_p50=p50(allw), dec_mean=mean(allw), post_n=len(pw), post_p50=p50(pw), post_mean=mean(pw),
                other_p50=p50(ow), other_mean=mean(ow), n_nc=len(nc), nc_sum=round(sum(ncw), 1), nc_mean=mean(ncw),
                nc_rows=[steps[j][4] for j in nc])


def tie(steps, ts, done):
    """each done request -> (key, its newcomer step index or None): the last newcomer step between its send and its first token"""
    out = []
    for r in done:
        a = (r.get("t_send") or r.get("t0") or r["t_first"] - 30.0) - 0.05
        j = bisect.bisect_right(ts, r["t_first"]) - 1
        hit = None
        while j >= 0 and ts[j] >= a:
            if is_nc(steps[j]): hit = j; break
            j -= 1
        out.append(((r.get("w"), r.get("k")), hit))
    return out


def matched(steps, ts, blocks):
    """blocks: [(name, arm, win, done)] -> {name: (sum_ms, n_steps)}, n_matched (prompts whose newcomer step starts inside every window)"""
    inw = {}
    for nm, arm, win, done in blocks:
        d = {}
        for key, j in tie(steps, ts, done):
            if j is not None and win[0] <= steps[j][0] <= win[1]: d[key] = j
        inw[nm] = d
    keys = None
    for nm in inw: keys = set(inw[nm]) if keys is None else keys & set(inw[nm])
    keys = keys or set()
    out = {}
    for nm in inw:
        js = sorted({inw[nm][k] for k in keys})
        ws = [wall(steps, j) for j in js]
        out[nm] = (round(sum(w for w in ws if w is not None), 1), len(js))
    return out, len(keys)


def link(s0, s1):
    if not s0 or not s1: return None
    g = lambda s, *k: (lambda v: v if isinstance(v, (int, float)) else 0)(
        __import__("functools").reduce(lambda x, y: x.get(y) if isinstance(x, dict) else None, k, s))
    d = lambda *k: g(s1, *k) - g(s0, *k)
    r0 = {r.get("rank"): r for r in ((s0.get("experts") or {}).get("ranks") or [])}
    miss = drop = fills = routed = rres = 0
    for r in ((s1.get("experts") or {}).get("ranks") or []):
        a = r0.get(r.get("rank")) or {}
        miss += g(r, "slots", "miss", "arena") - g(a, "slots", "miss", "arena")
        drop += g(r, "lent", "arena", "drop_recall") - g(a, "lent", "arena", "drop_recall")
        fills += g(r, "lent", "arena", "fills") - g(a, "lent", "arena", "fills")
        routed += g(r, "routed") - g(a, "routed"); rres += g(r, "routed_res") - g(a, "routed_res")
    return dict(staged_gb=round(d("link", "staged_bytes") / 1e9, 2), stream_gb=round(d("link", "stream_bytes") / 1e9, 2),
                h2d_gb=round(d("link", "h2d_bytes") / 1e9, 2), arena_miss=int(miss), arena_drop_recall=int(drop), arena_fills=int(fills),
                hit_pct=round(100.0 * rres / routed, 2) if routed else None)


def sahb(f0, f1):
    if not f0 or not f1: return None
    a = {r.get("rank"): r for r in ((f0.get("sahb") or {}).get("ranks") or [])}
    b = ((f1.get("sahb") or {}).get("ranks") or [])
    if not b: return None
    o = dict(short_steps_r0=0, big_steps_r0=0, short_gb=0.0, pick_kept=0, pick_dropped=0, expect_gb=0.0, recall_gib=0.0)
    for r in b:
        x = a.get(r.get("rank")) or {}
        dl = lambda k: (r.get(k) or 0) - (x.get(k) or 0)
        if r.get("rank") == 0: o["short_steps_r0"] = int(dl("short_steps")); o["big_steps_r0"] = int(dl("big_steps"))
        o["short_gb"] += dl("short_bytes") / 1e9; o["pick_kept"] += int(dl("pick_kept")); o["pick_dropped"] += int(dl("pick_dropped"))
        o["expect_gb"] += dl("expect_bytes") / 1e9; o["recall_gib"] += dl("sa_recall_bytes") / 2 ** 30
    o["gb_per_armed_step"] = round(o["short_gb"] / o["short_steps_r0"], 2) if o["short_steps_r0"] else None
    o["recall_gib_per_armed_step"] = round(o["recall_gib"] / o["short_steps_r0"], 3) if o["short_steps_r0"] else None
    for k in ("short_gb", "expect_gb", "recall_gib"): o[k] = round(o[k], 2)
    o["switches"] = {k: ((f1.get("switches") or {}).get(k)) for k in ("stage_short", "stage_pick")}
    return o


def run(argv):
    out, log, sw, post, base, jout, gens = None, None, 1, 10, "B", None, {8: 2, 16: 1}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--log": log = argv[i + 1]; i += 2
        elif a == "--sw": sw = int(argv[i + 1]); i += 2
        elif a == "--post": post = int(argv[i + 1]); i += 2
        elif a == "--base": base = argv[i + 1]; i += 2
        elif a == "--json": jout = argv[i + 1]; i += 2
        elif a == "--c8gens": gens[8] = int(argv[i + 1]); i += 2
        elif a == "--c16gens": gens[16] = int(argv[i + 1]); i += 2
        else: out = a; i += 1
    if not out or not log:
        print("SA14 VIRHE usage: OUT --log LOG"); return 2
    steps = read_steps(log) if os.path.exists(log) else []
    if not steps:
        print("SA14 VIRHE no step line in %s (RADIANCE_LOG_STEPS=1, docker logs -t)" % log); return 1
    ts = [s[0] for s in steps]
    R = [r for r in jl(os.path.join(out, "mb.jsonl")) if r.get("sched") == "G" and r.get("n") in (8, 16)]
    if not R:
        print("SA14 VIRHE no gens block in %s/mb.jsonl" % out); return 1
    ncm = rdj(os.path.join(out, "ncm-c8g.json")) or {}
    res = dict(blocks=[], arms={}, c16={})
    for r in R:
        nm, arm, n = r["name"], r["arm"], r["n"]
        mk = r.get("marks") or []
        rq = jl(os.path.join(out, "c8", nm + "-reqs.jsonl"))
        done = [x for x in rq if x.get("ev") == "done" and not x.get("err") and x.get("t_first") is not None]
        if len(mk) < 2:
            mk = sorted(x.get("t") for x in rq if x.get("ev") == "mark")[:2]
        b = dict(name=nm, arm=arm, n=n, agg=r.get("agg_dec_tps"), window_s=r.get("window_s"), requests=r.get("requests"),
                 errors=r.get("errors"), decode_share=r.get("decode_share"), ttft_p50=r.get("ttft_p50"), acceptance=r.get("acceptance"))
        bad = []
        if len(mk) < 2:
            bad.append("no window marks"); b.update(win=None)
        else:
            b.update(win=[mk[0], mk[1]], **window_steps(steps, ts, mk[0], mk[1], post))
            want = gens.get(n, 0) * n
            if want and b["n_nc"] != want: b["nc_note"] = "%d newcomer steps in the window (gens x %d = %d expected)" % (b["n_nc"], n, want)
        b["done"] = done
        b["link"] = link(rdj(os.path.join(out, "c8", nm + "-st-w0.json")), rdj(os.path.join(out, "c8", nm + "-st-w1.json")))
        b["sahb"] = sahb(rdj(os.path.join(out, "fn-%s-0.json" % nm)), rdj(os.path.join(out, "fn-%s-1.json" % nm)))
        if (b.get("errors") or 0) != 0 or not (b.get("requests") or 0) > 0: bad.append("request errors %s of %s requests" % (b.get("errors"), b.get("requests")))
        sa = b["sahb"]
        if sa is None: bad.append("no GET /admin/fn4 \"sahb\" counters around the block")
        elif arm == base and sa["short_steps_r0"] != 0: bad.append("SA armed %d step(s) under %s (stage_short %s)" % (sa["short_steps_r0"], base, sa["switches"].get("stage_short")))
        elif arm != base and sa["short_steps_r0"] == 0: bad.append("path did not run: SA armed 0 steps under %s (stage_short %s, stage_pick %s; big_steps %s)" % (arm, sa["switches"].get("stage_short"), sa["switches"].get("stage_pick"), sa["big_steps_r0"]))
        elif arm != base and (sa["switches"].get("stage_pick") or 0) > 0 and sa["pick_kept"] + sa["pick_dropped"] == 0: bad.append("stage_pick %s but no unit kept / dropped" % sa["switches"].get("stage_pick"))
        b["bad"] = bad
        if b.get("win") and b.get("n_nc") and b.get("agg") and b.get("window_s"):
            b["tok_per_nc"] = round(b["agg"] * b["window_s"] / b["n_nc"], 1)
            b["dec_per_nc"] = round(b["n_dec"] / b["n_nc"], 2)
        res["blocks"].append(b)
    # own matched-prompt sums per set (c8: a cross-check of ncmatch.py; c16: the reading)
    for n in (8, 16):
        bl = [b for b in res["blocks"] if b["n"] == n and b.get("win")]
        if len(bl) < 2: continue
        own, nk = matched(steps, ts, [(b["name"], b["arm"], b["win"], b["done"]) for b in bl])
        for b in bl: b["own_sum"], b["own_steps"], b["own_matched"] = own[b["name"]][0], own[b["name"]][1], nk
    for b in res["blocks"]:
        sa, lk = b.get("sahb") or {}, b.get("link") or {}
        print("SA14 BLOCK %s (%s, c%d gens): agg %s tok/s, window %s s, %s requests, %s errors, decode_share %s, TTFT p50 %s s, acceptance %s | "
              "decode steps %s: p50 %s / mean %s ms | %d after each newcomer: %s steps p50 %s / mean %s ms; the rest p50 %s / mean %s ms | "
              "newcomer steps in the window %s (rows %s), unclipped %s ms (%s ms a step)%s | own matched %s prompts: %s ms over %s steps | "
              "window link: staged %s GB, stream %s GB, h2d %s GB (all ranks; per newcomer %s / %s / %s GB), arena slot misses %s, arena drop_recall %s, "
              "fills %s, routed-resident %s %% | block sahb: SA armed %s steps (rank 0; big %s), staged %s GB all ranks = %s GB an armed step, pick kept %s / "
              "dropped %s, expect %s GB, loan recalled %s GiB (%s GiB an armed step) | switches %s%s" % (
                  b["name"], b["arm"], b["n"], b.get("agg"), b.get("window_s"), b.get("requests"), b.get("errors"), b.get("decode_share"),
                  b.get("ttft_p50"), b.get("acceptance"), b.get("n_dec"), f(b.get("dec_p50")), f(b.get("dec_mean")), post, b.get("post_n"),
                  f(b.get("post_p50")), f(b.get("post_mean")), f(b.get("other_p50")), f(b.get("other_mean")), b.get("n_nc"),
                  "%s..%s" % (min(b["nc_rows"]), max(b["nc_rows"])) if b.get("nc_rows") else "-", b.get("nc_sum"), f(b.get("nc_mean"), 1),
                  (" [NOTE %s]" % b["nc_note"]) if b.get("nc_note") else "", b.get("own_matched", "-"), b.get("own_sum", "-"), b.get("own_steps", "-"),
                  lk.get("staged_gb"), lk.get("stream_gb"), lk.get("h2d_gb"),
                  f(lk["staged_gb"] / b["n_nc"]) if lk and b.get("n_nc") else "-", f(lk["stream_gb"] / b["n_nc"]) if lk and b.get("n_nc") else "-",
                  f(lk["h2d_gb"] / b["n_nc"]) if lk and b.get("n_nc") else "-", lk.get("arena_miss"), lk.get("arena_drop_recall"), lk.get("arena_fills"),
                  lk.get("hit_pct"), sa.get("short_steps_r0"), sa.get("big_steps_r0"), sa.get("short_gb"), sa.get("gb_per_armed_step"), sa.get("pick_kept"),
                  sa.get("pick_dropped"), sa.get("expect_gb"), sa.get("recall_gib"), sa.get("recall_gib_per_armed_step"), json.dumps(sa.get("switches")),
                  (" -- VIRHE " + "; ".join(b["bad"])) if b["bad"] else ""))
    # c8: per arm
    c8 = [b for b in res["blocks"] if b["n"] == 8]
    arms = []
    for b in c8:
        if b["arm"] not in arms: arms.append(b["arm"])
    na = ncm.get("arms") or {}
    for a in arms:
        x = [b for b in c8 if b["arm"] == a]
        ag = [b["agg"] for b in x if isinstance(b.get("agg"), (int, float))]
        A = dict(blocks=[b["name"] for b in x], aggs=ag, agg=mean(ag), spread_pct=round(100.0 * (max(ag) - min(ag)) / statistics.mean(ag), 2) if len(ag) > 1 else None,
                 dec_p50=mean([b["dec_p50"] for b in x if b.get("dec_p50") is not None]), dec_mean=mean([b["dec_mean"] for b in x if b.get("dec_mean") is not None]),
                 post_p50=mean([b["post_p50"] for b in x if b.get("post_p50") is not None]), other_p50=mean([b["other_p50"] for b in x if b.get("other_p50") is not None]),
                 own_step=mean([b["own_sum"] / b["own_steps"] for b in x if b.get("own_steps")]),
                 ncm_step=(na.get(a) or {}).get("mean_step_ms"), ncm_sum=(na.get(a) or {}).get("mean_sum_ms"), ncm_spread=(na.get(a) or {}).get("spread_pct"),
                 dec_per_nc=mean([b["dec_per_nc"] for b in x if b.get("dec_per_nc")]), tok_per_nc=mean([b["tok_per_nc"] for b in x if b.get("tok_per_nc")]),
                 staged_gb=mean([b["link"]["staged_gb"] for b in x if b.get("link")]), stream_gb=mean([b["link"]["stream_gb"] for b in x if b.get("link")]),
                 h2d_gb=mean([b["link"]["h2d_gb"] for b in x if b.get("link")]), n_nc=mean([b["n_nc"] for b in x if b.get("n_nc")]),
                 arena_drop=mean([b["link"]["arena_drop_recall"] for b in x if b.get("link")]), arena_miss=mean([b["link"]["arena_miss"] for b in x if b.get("link")]),
                 armed_gb=mean([b["sahb"]["gb_per_armed_step"] for b in x if b.get("sahb") and b["sahb"].get("gb_per_armed_step")]),
                 armed=sum((b.get("sahb") or {}).get("short_steps_r0") or 0 for b in x))
        res["arms"][a] = A
    B0 = res["arms"].get(base) or {}
    for a in arms:
        A = res["arms"][a]
        nstep = A["ncm_step"] if A.get("ncm_step") is not None else A.get("own_step")
        ndec = B0.get("dec_per_nc") or A.get("dec_per_nc")
        A["cycle_ms"] = round(ndec * A["dec_mean"] + nstep, 1) if ndec and A.get("dec_mean") is not None and nstep is not None else None
        A["model_agg"] = round(1000.0 * (B0.get("tok_per_nc") or 0) / A["cycle_ms"], 1) if A.get("cycle_ms") and B0.get("tok_per_nc") else None
    sp = [res["arms"][a]["spread_pct"] for a in arms if res["arms"][a].get("spread_pct") is not None]
    msp = max(sp) if sp else None
    for a in arms:
        A = res["arms"][a]
        d = pc(A.get("agg"), B0.get("agg")) if a != base else None
        A["d_pct"] = round(d, 2) if d is not None else None
        nb = (B0.get("ncm_step") if B0.get("ncm_step") is not None else B0.get("own_step"))
        ns = (A.get("ncm_step") if A.get("ncm_step") is not None else A.get("own_step"))
        print("SA14 ARM c8 %s: agg %s tok/s (blocks %s%s)%s | decode p50 %s ms (mean %s), %d-after-newcomer p50 %s ms, the rest p50 %s ms%s | newcomer matched "
              "(ncmatch.py) %s ms a step (sum %s ms, spread %s %%), own tie %s ms a step%s | window: %s newcomer steps, %s decode steps a newcomer, %s tokens a "
              "newcomer, staged %s / stream %s / h2d %s GB (all ranks; a newcomer %s / %s / %s), arena drop_recall %s, arena misses %s | SA armed %s steps, %s GB "
              "staged an armed step (all ranks)" % (
                  a, f(A.get("agg")), "/".join(f(v) for v in A["aggs"]), (", spread %s %%" % f(A["spread_pct"])) if A.get("spread_pct") is not None else "",
                  (" = %s vs %s" % (fp(d), base)) if d is not None else "", f(A.get("dec_p50")), f(A.get("dec_mean")), post, f(A.get("post_p50")),
                  f(A.get("other_p50")), (" (vs %s: decode p50 %s ms, after-newcomer p50 %s ms)" % (base, f((A["dec_p50"] or 0) - (B0.get("dec_p50") or 0)),
                  f((A["post_p50"] or 0) - (B0.get("post_p50") or 0)))) if a != base and A.get("dec_p50") is not None and B0.get("dec_p50") is not None and A.get("post_p50") is not None and B0.get("post_p50") is not None else "",
                  f(A.get("ncm_step"), 1), f(A.get("ncm_sum"), 1), f(A.get("ncm_spread")), f(A.get("own_step"), 1),
                  (" (vs %s: %s ms a newcomer step, %s)" % (base, f(ns - nb, 1), fp(pc(ns, nb)))) if a != base and ns is not None and nb is not None else "",
                  f(A.get("n_nc"), 1), f(A.get("dec_per_nc"), 1), f(A.get("tok_per_nc"), 0), f(A.get("staged_gb")), f(A.get("stream_gb")), f(A.get("h2d_gb")),
                  f(A["staged_gb"] / A["n_nc"]) if A.get("staged_gb") is not None and A.get("n_nc") else "-",
                  f(A["stream_gb"] / A["n_nc"]) if A.get("stream_gb") is not None and A.get("n_nc") else "-",
                  f(A["h2d_gb"] / A["n_nc"]) if A.get("h2d_gb") is not None and A.get("n_nc") else "-",
                  f(A.get("arena_drop"), 0), f(A.get("arena_miss"), 0), A.get("armed"), f(A.get("armed_gb"))))
    for a in arms:
        A = res["arms"][a]
        md = pc(A.get("model_agg"), B0.get("model_agg")) if a != base else None
        print("SA14 MODEL c8 %s: cycle a newcomer = %s decode steps (%s's) x %s ms (mean) + %s ms (matched newcomer) = %s ms -> %s tokens / cycle = %s tok/s "
              "(measured %s)%s" % (a, f(B0.get("dec_per_nc") or A.get("dec_per_nc"), 1), base, f(A.get("dec_mean")),
                                   f(A.get("ncm_step") if A.get("ncm_step") is not None else A.get("own_step"), 1), f(A.get("cycle_ms"), 1), f(B0.get("tok_per_nc"), 0),
                                   f(A.get("model_agg"), 1), f(A.get("agg")),
                                   (" | model %s vs %s, measured %s" % (fp(md), base, fp(A.get("d_pct")))) if a != base else ""))
    # c16
    c16 = [b for b in res["blocks"] if b["n"] == 16]
    c16b = [b for b in c16 if b["arm"] == base]
    for b in c16:
        if b["arm"] == base: continue
        x0 = c16b[0] if c16b else None
        ns = b["own_sum"] / b["own_steps"] if b.get("own_steps") else None
        n0 = x0["own_sum"] / x0["own_steps"] if x0 and x0.get("own_steps") else None
        res["c16"][b["arm"]] = dict(d_pct=pc(b.get("agg"), x0.get("agg") if x0 else None))
        print("SA14 C16 %s vs %s (one gens block each): agg %s -> %s tok/s (%s) | decode p50 %s -> %s ms, %d-after-newcomer p50 %s -> %s ms | own matched (%s prompts) "
              "%s -> %s ms a newcomer step (%s) | SA armed %s steps, %s GB an armed step | window staged %s -> %s GB, stream %s -> %s GB" % (
                  b["arm"], base, f(x0.get("agg")) if x0 else "-", f(b.get("agg")), fp(pc(b.get("agg"), x0.get("agg") if x0 else None)),
                  f(x0.get("dec_p50")) if x0 else "-", f(b.get("dec_p50")), post, f(x0.get("post_p50")) if x0 else "-", f(b.get("post_p50")),
                  b.get("own_matched", "-"), f(n0, 1), f(ns, 1), fp(pc(ns, n0)), (b.get("sahb") or {}).get("short_steps_r0"),
                  f((b.get("sahb") or {}).get("gb_per_armed_step")), f(((x0 or {}).get("link") or {}).get("staged_gb")), f((b.get("link") or {}).get("staged_gb")),
                  f(((x0 or {}).get("link") or {}).get("stream_gb")), f((b.get("link") or {}).get("stream_gb"))))
    if not c16: print("SA14 C16: no c16 gens block (skipped by the budget or not run)")
    nbad = sum(len(b["bad"]) for b in res["blocks"])
    parts = []
    for a in arms:
        A = res["arms"][a]
        parts.append("%s %s%s" % (a, f(A.get("agg")), (" (%s)" % fp(A.get("d_pct"))) if a != base else ""))
    ncp = []
    for a in arms:
        A = res["arms"][a]; s = A.get("ncm_step") if A.get("ncm_step") is not None else A.get("own_step")
        ncp.append("%s %s" % (a, f(s, 1)))
    best = max((a for a in arms if a != base and res["arms"][a].get("agg") is not None), key=lambda a: res["arms"][a]["agg"], default=None)
    bd = res["arms"][best]["d_pct"] if best else None
    tag = ("within the same-arm spread x2 (%s %%): INCONCLUSIVE" % f(2 * msp)) if best and msp is not None and bd is not None and abs(bd) < 2 * msp else (
          ("outside the same-arm spread x2 (%s %%)" % f(2 * msp)) if best and msp is not None and bd is not None else "no spread (one block an arm)")
    print("SA14 HEADLINE c8 work-based agg: %s tok/s | best SA arm %s %s vs %s, %s | matched newcomer ms a step: %s | decode p50 ms: %s | %d-after-newcomer "
          "p50 ms: %s | c16 %s | %s guard finding(s)%s" % (
              " / ".join(parts), best or "-", fp(bd), base, tag, " / ".join(ncp),
              " / ".join("%s %s" % (a, f(res["arms"][a].get("dec_p50"))) for a in arms), post,
              " / ".join("%s %s" % (a, f(res["arms"][a].get("post_p50"))) for a in arms),
              " / ".join("%s %s" % (a, fp(v["d_pct"])) for a, v in res["c16"].items()) or "-", nbad, " -- VIRHE see the BLOCK lines" if nbad else ""))
    res["best"] = best
    if jout:
        for b in res["blocks"]: b.pop("done", None)
        with open(jout, "w") as fo: json.dump(res, fo)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="sa14sum-")
    os.makedirs(os.path.join(d, "c8"))
    t00 = 1794517000.0
    L = []
    def iso(t):
        x = "%.9f" % (t - int(t))
        return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "." + x[2:] + "Z"
    def stats(staged, stream, drop):
        return {"link": {"staged_bytes": staged, "stream_bytes": stream, "h2d_bytes": staged + stream},
                "experts": {"ranks": [{"rank": r, "routed": 1000, "routed_res": 990, "slots": {"miss": {"arena": drop}}, "lent": {"arena": {"drop_recall": drop, "fills": drop}}} for r in range(4)]}}
    def fn(steps, gb, sshort, spick):
        return {"switches": {"stage_short": sshort, "stage_pick": spick},
                "sahb": {"ranks": [{"rank": r, "short_steps": steps, "short_bytes": gb * 1e9 / 4, "pick_kept": 10 * steps if spick else 0,
                                    "pick_dropped": 2 * steps if spick else 0, "big_steps": 16, "expect_bytes": 0, "sa_recall_bytes": steps * 2 ** 30} for r in range(4)]}}
    rows = []
    def blk(i, arm, n, t0, nc_ms, post_ms, dec_ms, n_nc, armed):
        """decode steps of dec_ms (the first 10 after each newcomer post_ms), a newcomer (one MIXED step of nc_ms) after every 41 decode steps"""
        nm = "c%d-%d-G-%s" % (n, i, arm)
        rq = []; nct = []
        t = t0; step = 0; k = 0; after = 30
        while len(nct) < n_nc + 2:
            if after >= 41:
                segs = "".join("  [%d] slot=%d q=1 ctx=900" % (e, e + 1) for e in range(min(n - 1, 8)))
                if n <= 8: segs += "  [%d] slot=%d q=1050 ctx=0" % (n - 1, n)
                L.append("%s I step %d MIXED n_seq=%d n_tok=%d%s" % (iso(t), step, n, n - 1 + 1050, segs))
                rq.append(dict(ev="done", w=k % n, k=k // n + 1, t_send=t - 0.3, t_first=t + nc_ms / 1000.0 + 0.002, prompt=1050, cached=0, err=None))
                nct.append(t); t += nc_ms / 1000.0; k += 1; after = 0
            else:
                L.append("%s I step %d decode n_seq=%d n_tok=%d  [0] slot=1 q=1 ctx=900" % (iso(t), step, n, n))
                t += (post_ms if after < 10 else dec_ms) / 1000.0; after += 1
            step += 1
        w0, w1 = nct[0] - 0.4, nct[n_nc - 1] + 0.5
        rq.append(dict(ev="mark", k=0, t=w0)); rq.append(dict(ev="mark", k=1, t=w1))
        with open(os.path.join(d, "c8", nm + "-reqs.jsonl"), "w") as fo:
            for r in rq: fo.write(json.dumps(r) + "\n")
        span = w1 - w0
        rows.append(dict(name=nm, n=n, arm=arm, sched="G", agg_dec_tps=round(n_nc * 512 / span, 1), window_s=round(span, 1), marks=[w0, w1],
                         requests=n_nc, errors=0, decode_share=0.8))
        json.dump(stats(0, 0, 0), open(os.path.join(d, "c8", nm + "-st-w0.json"), "w"))
        json.dump(stats(armed * 3e9, n_nc * 7e9, 5 * armed), open(os.path.join(d, "c8", nm + "-st-w1.json"), "w"))
        sp = 0 if arm == "B" else (1 if arm == "S1" else 2)
        json.dump(fn(0, 0, 1025 if sp else 0, sp), open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        json.dump(fn(armed, armed * 3.0, 1025 if sp else 0, sp), open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    blk(1, "B", 8, t00, 533.0, 34.0, 34.0, 16, 0)
    blk(2, "S1", 8, t00 + 100, 503.0, 36.0, 34.0, 16, 18)
    blk(3, "S2", 8, t00 + 200, 493.0, 36.0, 34.0, 16, 18)
    blk(4, "S2", 8, t00 + 300, 493.0, 36.0, 34.0, 16, 18)
    blk(5, "S1", 8, t00 + 400, 503.0, 36.0, 34.0, 16, 18)
    blk(6, "B", 8, t00 + 500, 533.0, 34.0, 34.0, 16, 0)
    blk(1, "B", 16, t00 + 600, 560.0, 45.0, 45.0, 16, 0)
    blk(2, "S2", 16, t00 + 700, 520.0, 47.0, 45.0, 16, 0)     # S2 at c16 armed 0 steps: the engagement guard must fire
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    lp = os.path.join(d, "eng.log.gz")
    with gzip.open(lp, "wt") as fo: fo.write("\n".join(L) + "\n")
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = run([d, "--log", lp, "--sw", "1", "--post", "10", "--json", os.path.join(d, "o.json")])
    t = buf.getvalue()
    o = json.load(open(os.path.join(d, "o.json")))
    why = []
    A = o["arms"]
    if rc != 0: why.append("rc %s" % rc)
    if set(A) != {"B", "S1", "S2"}: why.append("arms %s" % list(A))
    else:
        if abs((A["S2"].get("own_step") or 0) - 493.0) > 1.0 or abs((A["B"].get("own_step") or 0) - 533.0) > 1.0: why.append("own newcomer step %s / %s" % (A["S2"].get("own_step"), A["B"].get("own_step")))
        if abs((A["S2"].get("post_p50") or 0) - 36.0) > 0.5 or abs((A["B"].get("post_p50") or 0) - 34.0) > 0.5: why.append("post p50 %s / %s" % (A["S2"].get("post_p50"), A["B"].get("post_p50")))
        if abs((A["S1"].get("other_p50") or 0) - 34.0) > 0.5: why.append("other p50 %s" % A["S1"].get("other_p50"))
        if A["S2"].get("armed_gb") != 3.0: why.append("GB an armed step %s" % A["S2"].get("armed_gb"))
        if not A["S2"].get("model_agg") or not A["B"].get("model_agg") or A["S2"]["model_agg"] <= A["B"]["model_agg"]: why.append("model %s / %s" % (A["S2"].get("model_agg"), A["B"].get("model_agg")))
    if o.get("best") != "S2": why.append("best %s" % o.get("best"))
    bl = {b["name"]: b for b in o["blocks"]}
    if [b["n_nc"] for b in o["blocks"]][:6] != [16] * 6: why.append("newcomers in window %s" % [b["n_nc"] for b in o["blocks"]])
    if not any("path did not run" in x for x in bl["c16-2-G-S2"]["bad"]): why.append("c16 S2 engagement guard did not fire")
    if any(bl[n]["bad"] for n in bl if n.startswith("c8-")): why.append("a c8 block flagged %s" % [bl[n]["bad"] for n in bl if n.startswith("c8-")])
    if "SA14 HEADLINE" not in t or "SA14 MODEL c8 S2" not in t or "SA14 C16 S2 vs B" not in t: why.append("lines")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: gens blocks B S1 S2 S2 S1 B + c16 B / S2 -- decode p50 all / after-newcomer / rest, own matched newcomer sums, "
          "sahb staged GB an armed step, cycle model, best arm S2, the engagement guard fires on an SA arm that armed 0 steps")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
