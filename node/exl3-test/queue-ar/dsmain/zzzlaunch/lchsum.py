#!/usr/bin/env python3
"""lchsum.py (13.11.2026, queue-ar/dsmain/zzzlaunch, a builder for ds-main-port) -- the zzzlaunch summary: lever RCO (core fn4
route_copy_ov 1: a routing report's copy recorded with its ranges, so the dispatch after it may play unordered; ds41-launch 0f65666)
read in-process on THE int15 PIN (the int14 pin + exl3_x3d 2) against route_copy_ov 0 (int15 exactly). A COPY of zzk25s/k25sum.py
(eac39a76, itself zsa14/sa14sum.py's copy; its step-log readers, gens windows, own matched-prompt tie, link counters, cycle model and
1u sets kept verbatim) with the per-block switch read back changed from exl3_x3d (K25U's) to route_copy_ov (B 0, Z 1) plus a pin guard
(exl3_x3d 2 = the int15 pin at every block's end), the references moved to the int15 zero point (zzk25s's Z blocks), lines "LCH ...".
WORK-based c8 / c16 blocks (harness gens window: the same requests in every block, agg = tokens / time); newcomer readings on
MATCHED-PROMPT sums; decode-step p50 over ALL decode steps of the window and the full-row bucket, block separation. The int15
references are another start: INFO only.

  lchsum.py OUT --log ENGINE_LOG [--sw 1] [--post 10] [--base B] [--zarm Z] [--xb 0] [--xz 1] [--xpin 2] [--c8gens 2] [--c16gens 1]
            [--essay-md5 47a39dac581a] [--json OUT.json]
      OUT = the item's out dir: mb.jsonl (mbx's rows), c8/NAME-reqs.jsonl, c8/NAME-st-w0.json / -st-w1.json (/stats at the two
      window marks), fn-NAME-1.json (GET /admin/fn4 after the block), ncm-c8g.json (ncmatch.py, optional), blocks.jsonl (blockx's 1u
      reads, optional). Only the gens blocks (sched G) are read. Prints lines "LCH ..."; a guard failure is a "VIRHE" inside its line.
      --xb / --xz: route_copy_ov read back at a block's end under the base / the lever arm; --xpin: exl3_x3d (the int15 pin) at every block's end.
  lchsum.py selftest
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



# ---------------------------------------------------------------- zzk25s's own: K25U's switch read back, the modal-row bucket, the 1u sets, the references
REFS = (("zzk25s c8 gens Z = the int15 zero point (unstamped start)", "c8", 191.60), ("zzk25s c16 gens Z (int15)", "c16", 236.70),
        ("zzk25s 1u essay Z (int15)", "u1", 89.91))


def x3dsw(p):   # zzzlaunch: the LEVER's switch read back at a block's end (route_copy_ov)
    d = rdj(p)
    if not d: return None
    return (d.get("switches") or {}).get("route_copy_ov")


def pinsw(p):   # zzzlaunch: the int15 pin's exl3_x3d at a block's end
    d = rdj(p)
    if not d: return None
    return (d.get("switches") or {}).get("exl3_x3d")


def rowcount(steps, ts, w0, w1, n):
    """{n_tok: count} over the window's decode steps with n_seq == n (the full-row candidates)"""
    i0, i1 = bisect.bisect_left(ts, w0), bisect.bisect_right(ts, w1)
    c = {}
    for j in range(i0, i1):
        s = steps[j]
        if s[2] == "decode" and s[3] == n and wall(steps, j) is not None: c[s[4]] = c.get(s[4], 0) + 1
    return c


def bucket(steps, ts, w0, w1, n, rows):
    """p50 / count of the window's decode steps with n_seq == n and n_tok == rows (TESTING.md 13.11. lesson 1: the full-row bucket)"""
    i0, i1 = bisect.bisect_left(ts, w0), bisect.bisect_right(ts, w1)
    w = [wall(steps, j) for j in range(i0, i1) if steps[j][2] == "decode" and steps[j][3] == n and steps[j][4] == rows]
    w = [x for x in w if x is not None]
    return p50(w), len(w)


def sep(a, b):
    """block p50s of two arms: SEPARATED when every block of one arm is below every block of the other"""
    if not a or not b: return "-"
    return "SEPARATED" if max(a) < min(b) or max(b) < min(a) else "OVERLAP"


def u1sum(out, arms, want, emd5, sw, xpin=2):
    R = jl(os.path.join(out, "blocks.jsonl"))
    res = {}
    for a in arms:
        W = [r for r in R if r.get("lever") == "U1W" and r.get("arm") == a]
        U = [r for r in R if r.get("lever") == "U1" and r.get("arm") == a]
        if not U and not W: continue
        g = lambda r, k: (r.get("j") or {}).get(k)
        s = lambda r, k: (r.get("st") or {}).get(k)
        num = lambda v: [x for x in v if isinstance(x, (int, float))]
        md = [g(r, "text_md5") for r in W + U]
        x = [x3dsw(os.path.join(out, "fn-%s-1.json" % r.get("name"))) if r.get("name") else None for r in W + U]
        xp = [pinsw(os.path.join(out, "fn-%s-1.json" % r.get("name"))) if r.get("name") else None for r in W + U]
        bad, info = [], []
        if any(v != want.get(a) for v in x): (bad if sw else info).append("route_copy_ov read back %s after the reads (want %s)" % (x, want.get(a)))
        if any(v != xpin for v in xp): (bad if sw else info).append("exl3_x3d read back %s after the reads (want %s: the int15 pin)" % (xp, xpin))
        if any(m != emd5 for m in md): (bad if sw else info).append("answers %s differ from ref14's essay answer %s (strict, T 0 greedy)" % (md, emd5))
        tps, ms, tok = num(g(r, "decode_tps") for r in U), num(g(r, "ms_per_decode_step") for r in U), num(g(r, "tokens_per_step") for r in U)
        mb, acc = num(s(r, "stream_mb_step") for r in U), num(g(r, "acceptance") for r in U)
        res[a] = dict(n=len(U), tps=mean(tps), tps_reads=tps, spread=round(100.0 * (max(tps) - min(tps)) / statistics.mean(tps), 2) if len(tps) > 1 else None,
                      ms=mean(ms), ms_reads=ms, tok=mean(tok), acc=mean(acc), mb=mean(mb), mb_reads=mb, prom=[s(r, "promotions") for r in U],
                      w_tps=g(W[0], "decode_tps") if W else None, w_mb=s(W[0], "stream_mb_step") if W else None, md5=md, x3d=x, bad=bad, info=info)
    return res


def run(argv):
    out, log, sw, post, base, zarm, jout, gens, xb, xz, emd5, xpin = None, None, 1, 10, "B", "Z", None, {8: 2, 16: 1}, 0, 1, "47a39dac581a", 2
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--log": log = argv[i + 1]; i += 2
        elif a == "--sw": sw = int(argv[i + 1]); i += 2
        elif a == "--post": post = int(argv[i + 1]); i += 2
        elif a == "--base": base = argv[i + 1]; i += 2
        elif a == "--zarm": zarm = argv[i + 1]; i += 2
        elif a == "--xb": xb = int(argv[i + 1]); i += 2
        elif a == "--xz": xz = int(argv[i + 1]); i += 2
        elif a == "--xpin": xpin = int(argv[i + 1]); i += 2
        elif a == "--essay-md5": emd5 = argv[i + 1]; i += 2
        elif a == "--json": jout = argv[i + 1]; i += 2
        elif a == "--c8gens": gens[8] = int(argv[i + 1]); i += 2
        elif a == "--c16gens": gens[16] = int(argv[i + 1]); i += 2
        else: out = a; i += 1
    if not out or not log:
        print("LCH VIRHE usage: OUT --log LOG"); return 2
    want = {base: xb, zarm: xz}
    U1 = u1sum(out, (base, zarm), want, emd5, sw, xpin)
    steps = read_steps(log) if os.path.exists(log) else []
    if not steps:
        print("LCH VIRHE no step line in %s (RADIANCE_LOG_STEPS=1, docker logs -t)" % log); return 1
    ts = [s[0] for s in steps]
    R = [r for r in jl(os.path.join(out, "mb.jsonl")) if r.get("sched") == "G" and r.get("n") in (8, 16)]
    if not R:
        print("LCH VIRHE no gens block in %s/mb.jsonl" % out); return 1
    ncm = rdj(os.path.join(out, "ncm-c8g.json")) or {}
    res = dict(blocks=[], arms={}, c16={}, u1=U1)
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
            b["rowc"] = rowcount(steps, ts, mk[0], mk[1], n)
            want_nc = gens.get(n, 0) * n
            if want_nc and b["n_nc"] != want_nc: b["nc_note"] = "%d newcomer steps in the window (gens x %d = %d expected)" % (b["n_nc"], n, want_nc)
        b["done"] = done
        b["link"] = link(rdj(os.path.join(out, "c8", nm + "-st-w0.json")), rdj(os.path.join(out, "c8", nm + "-st-w1.json")))
        b["x3d"] = x3dsw(os.path.join(out, "fn-%s-1.json" % nm))
        b["pin"] = pinsw(os.path.join(out, "fn-%s-1.json" % nm))
        if (b.get("errors") or 0) != 0 or not (b.get("requests") or 0) > 0: bad.append("request errors %s of %s requests" % (b.get("errors"), b.get("requests")))
        if arm not in want: bad.append("arm %s is neither %s nor %s" % (arm, base, zarm))
        elif sw and b["x3d"] != want[arm]: bad.append("route_copy_ov %s at the block's end (GET /admin/fn4; want %s under %s)" % (b["x3d"], want[arm], arm))
        if sw and b["pin"] != xpin: bad.append("exl3_x3d %s at the block's end (GET /admin/fn4; want %s: the int15 pin)" % (b["pin"], xpin))
        b["bad"] = bad
        if b.get("win") and b.get("n_nc") and b.get("agg") and b.get("window_s"):
            b["tok_per_nc"] = round(b["agg"] * b["window_s"] / b["n_nc"], 1)
            b["dec_per_nc"] = round(b["n_dec"] / b["n_nc"], 2)
        res["blocks"].append(b)
    # the full-row bucket per concurrency: the modal n_tok of the n_seq == n decode steps over every block of that concurrency
    for n in (8, 16):
        pool = {}
        for b in res["blocks"]:
            if b["n"] == n:
                for k, v in (b.get("rowc") or {}).items(): pool[k] = pool.get(k, 0) + v
        if not pool: continue
        rows = max(pool, key=lambda k: pool[k])
        for b in res["blocks"]:
            if b["n"] == n and b.get("win"):
                b["bkt_rows"] = rows; b["bkt_p50"], b["bkt_n"] = bucket(steps, ts, b["win"][0], b["win"][1], n, rows)
    # own matched-prompt sums per set (c8: a cross-check of ncmatch.py; c16: the reading)
    for n in (8, 16):
        bl = [b for b in res["blocks"] if b["n"] == n and b.get("win")]
        if len(bl) < 2: continue
        own, nk = matched(steps, ts, [(b["name"], b["arm"], b["win"], b["done"]) for b in bl])
        for b in bl: b["own_sum"], b["own_steps"], b["own_matched"] = own[b["name"]][0], own[b["name"]][1], nk
    for b in res["blocks"]:
        lk = b.get("link") or {}
        print("LCH BLOCK %s (%s, c%d gens): agg %s tok/s, window %s s, %s requests, %s errors, decode_share %s, TTFT p50 %s s, acceptance %s | "
              "decode steps %s: p50 %s / mean %s ms | full-row bucket (n_seq %d, n_tok %s): %s steps p50 %s ms | %d after each newcomer: %s steps p50 %s / "
              "mean %s ms; the rest p50 %s / mean %s ms | newcomer steps in the window %s (rows %s), unclipped %s ms (%s ms a step)%s | own matched %s "
              "prompts: %s ms over %s steps | window link: staged %s GB, stream %s GB, h2d %s GB (all ranks), arena slot misses %s, arena drop_recall %s, "
              "fills %s, routed-resident %s %% | route_copy_ov at the block's end %s (want %s), exl3_x3d %s%s" % (
                  b["name"], b["arm"], b["n"], b.get("agg"), b.get("window_s"), b.get("requests"), b.get("errors"), b.get("decode_share"),
                  b.get("ttft_p50"), b.get("acceptance"), b.get("n_dec"), f(b.get("dec_p50")), f(b.get("dec_mean")), b["n"], b.get("bkt_rows", "-"),
                  b.get("bkt_n", "-"), f(b.get("bkt_p50")), post, b.get("post_n"), f(b.get("post_p50")), f(b.get("post_mean")), f(b.get("other_p50")),
                  f(b.get("other_mean")), b.get("n_nc"), "%s..%s" % (min(b["nc_rows"]), max(b["nc_rows"])) if b.get("nc_rows") else "-", b.get("nc_sum"),
                  f(b.get("nc_mean"), 1), (" [NOTE %s]" % b["nc_note"]) if b.get("nc_note") else "", b.get("own_matched", "-"), b.get("own_sum", "-"),
                  b.get("own_steps", "-"), lk.get("staged_gb"), lk.get("stream_gb"), lk.get("h2d_gb"), lk.get("arena_miss"), lk.get("arena_drop_recall"),
                  lk.get("arena_fills"), lk.get("hit_pct"), b.get("x3d"), want.get(b["arm"]), b.get("pin"), (" -- VIRHE " + "; ".join(b["bad"])) if b["bad"] else ""))
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
                 dec_p50s=[b["dec_p50"] for b in x if b.get("dec_p50") is not None], bkt_p50s=[b["bkt_p50"] for b in x if b.get("bkt_p50") is not None],
                 dec_p50=mean([b["dec_p50"] for b in x if b.get("dec_p50") is not None]), dec_mean=mean([b["dec_mean"] for b in x if b.get("dec_mean") is not None]),
                 bkt_p50=mean([b["bkt_p50"] for b in x if b.get("bkt_p50") is not None]), bkt_rows=next((b.get("bkt_rows") for b in x if b.get("bkt_rows")), None),
                 post_p50=mean([b["post_p50"] for b in x if b.get("post_p50") is not None]), other_p50=mean([b["other_p50"] for b in x if b.get("other_p50") is not None]),
                 own_step=mean([b["own_sum"] / b["own_steps"] for b in x if b.get("own_steps")]),
                 ncm_step=(na.get(a) or {}).get("mean_step_ms"), ncm_sum=(na.get(a) or {}).get("mean_sum_ms"), ncm_spread=(na.get(a) or {}).get("spread_pct"),
                 dec_per_nc=mean([b["dec_per_nc"] for b in x if b.get("dec_per_nc")]), tok_per_nc=mean([b["tok_per_nc"] for b in x if b.get("tok_per_nc")]),
                 stream_gb=mean([b["link"]["stream_gb"] for b in x if b.get("link")]), h2d_gb=mean([b["link"]["h2d_gb"] for b in x if b.get("link")]),
                 n_nc=mean([b["n_nc"] for b in x if b.get("n_nc")]))
        A["dec_spread_pct"] = round(100.0 * (max(A["dec_p50s"]) - min(A["dec_p50s"])) / statistics.mean(A["dec_p50s"]), 2) if len(A["dec_p50s"]) > 1 else None
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
        A["dec_d_pct"] = pc(A.get("dec_p50"), B0.get("dec_p50")) if a != base else None
        A["bkt_d_pct"] = pc(A.get("bkt_p50"), B0.get("bkt_p50")) if a != base else None
        nb = (B0.get("ncm_step") if B0.get("ncm_step") is not None else B0.get("own_step"))
        ns = (A.get("ncm_step") if A.get("ncm_step") is not None else A.get("own_step"))
        print("LCH ARM c8 %s (route_copy_ov %s): agg %s tok/s (blocks %s%s)%s | decode p50 %s ms (blocks %s, spread %s %%; mean %s)%s | full-row bucket (8 seqs, %s rows) "
              "p50 %s ms (blocks %s)%s | %d-after-newcomer p50 %s ms, the rest p50 %s ms | newcomer matched (ncmatch.py) %s ms a step (sum %s ms, spread %s %%), "
              "own tie %s ms a step%s | window: %s newcomer steps, %s decode steps a newcomer, %s tokens a newcomer, stream %s / h2d %s GB (all ranks)" % (
                  a, want.get(a), f(A.get("agg")), "/".join(f(v) for v in A["aggs"]), (", spread %s %%" % f(A["spread_pct"])) if A.get("spread_pct") is not None else "",
                  (" = %s vs %s" % (fp(d), base)) if d is not None else "", f(A.get("dec_p50")), "/".join(f(v) for v in A["dec_p50s"]), f(A.get("dec_spread_pct")),
                  f(A.get("dec_mean")), (" = %s ms, %s vs %s (%s)" % (f((A["dec_p50"] or 0) - (B0.get("dec_p50") or 0)), fp(A["dec_d_pct"]), base, sep(A["dec_p50s"], B0.get("dec_p50s"))))
                  if a != base and A.get("dec_p50") is not None and B0.get("dec_p50") is not None else "", A.get("bkt_rows", "-"), f(A.get("bkt_p50")),
                  "/".join(f(v) for v in A["bkt_p50s"]), (" = %s vs %s (%s)" % (fp(A["bkt_d_pct"]), base, sep(A["bkt_p50s"], B0.get("bkt_p50s")))) if a != base and A.get("bkt_d_pct") is not None else "",
                  post, f(A.get("post_p50")), f(A.get("other_p50")), f(A.get("ncm_step"), 1), f(A.get("ncm_sum"), 1), f(A.get("ncm_spread")), f(A.get("own_step"), 1),
                  (" (vs %s: %s ms a newcomer step, %s)" % (base, f(ns - nb, 1), fp(pc(ns, nb)))) if a != base and ns is not None and nb is not None else "",
                  f(A.get("n_nc"), 1), f(A.get("dec_per_nc"), 1), f(A.get("tok_per_nc"), 0), f(A.get("stream_gb")), f(A.get("h2d_gb"))))
    for a in arms:
        A = res["arms"][a]
        md = pc(A.get("model_agg"), B0.get("model_agg")) if a != base else None
        print("LCH MODEL c8 %s: cycle a newcomer = %s decode steps (%s's) x %s ms (mean) + %s ms (matched newcomer) = %s ms -> %s tokens / cycle = %s tok/s "
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
        res["c16"][b["arm"]] = dict(d_pct=pc(b.get("agg"), x0.get("agg") if x0 else None), dec_d_pct=pc(b.get("dec_p50"), x0.get("dec_p50") if x0 else None),
                                    bkt_d_pct=pc(b.get("bkt_p50"), x0.get("bkt_p50") if x0 else None))
        print("LCH C16 %s vs %s (one gens block each): agg %s -> %s tok/s (%s) | decode p50 %s -> %s ms (%s) | full-row bucket (16 seqs, %s rows: %s / %s steps) "
              "p50 %s -> %s ms (%s) | %d-after-newcomer p50 %s -> %s ms | own matched (%s prompts) %s -> %s ms a newcomer step (%s) | window stream %s -> %s GB" % (
                  b["arm"], base, f(x0.get("agg")) if x0 else "-", f(b.get("agg")), fp(pc(b.get("agg"), x0.get("agg") if x0 else None)),
                  f(x0.get("dec_p50")) if x0 else "-", f(b.get("dec_p50")), fp(res["c16"][b["arm"]]["dec_d_pct"]), b.get("bkt_rows", "-"),
                  (x0 or {}).get("bkt_n", "-"), b.get("bkt_n", "-"), f((x0 or {}).get("bkt_p50")), f(b.get("bkt_p50")), fp(res["c16"][b["arm"]]["bkt_d_pct"]),
                  post, f(x0.get("post_p50")) if x0 else "-", f(b.get("post_p50")), b.get("own_matched", "-"), f(n0, 1), f(ns, 1), fp(pc(ns, n0)),
                  f(((x0 or {}).get("link") or {}).get("stream_gb")), f((b.get("link") or {}).get("stream_gb"))))
    if not c16: print("LCH C16: no c16 gens block (skipped by the budget or not run)")
    # 1u essay sets
    for a in (base, zarm):
        u = U1.get(a)
        if not u: print("LCH 1u essay %s: not run" % a); continue
        print("LCH 1u essay %s (route_copy_ov %s; one discarded + %d back to back, resprobe essay 512, T 0): %s tok/s (reads %s, spread %s %%), %s ms a step (reads %s), "
              "%s tok a step, acceptance %s, stream %s MB a step (reads %s), promotions a read %s | discarded warm-up %s tok/s at %s MB a step | answers %s | "
              "route_copy_ov read back after each read %s%s" % (
                  a, want.get(a), u["n"], f(u["tps"]), "/".join(f(v) for v in u["tps_reads"]), f(u["spread"]), f(u["ms"]), "/".join(f(v) for v in u["ms_reads"]),
                  f(u["tok"]), f(u["acc"], 3), f(u["mb"], 1), "/".join(f(v, 1) for v in u["mb_reads"]), "/".join(str(v) for v in u["prom"]), f(u["w_tps"]),
                  f(u["w_mb"], 1), ("IDENTICAL to ref14's essay answer (%s) on all %d reads" % (emd5, len(u["md5"]))) if all(m == emd5 for m in u["md5"]) else "md5 %s" % u["md5"],
                  u["x3d"], (" -- VIRHE " + "; ".join(u["bad"])) if u["bad"] else ((" | INFO (mock) " + "; ".join(u["info"])) if u["info"] else "")))
    ub, uz = U1.get(base), U1.get(zarm)
    u1d = None
    if ub and uz:
        u1d = pc(uz.get("tps"), ub.get("tps"))
        dm = abs((uz.get("mb") or 0) - (ub.get("mb") or 0))
        same = ub["md5"] == uz["md5"]
        print("LCH 1u %s vs %s: %s -> %s tok/s (%s), %s -> %s ms a step (%s), tok a step %s / %s | stream MB a step %s / %s (%s) | spread %s %% / %s %% | "
              "answers %s%s" % (
                  zarm, base, f(ub.get("tps")), f(uz.get("tps")), fp(u1d), f(ub.get("ms")), f(uz.get("ms")), fp(pc(uz.get("ms"), ub.get("ms"))), f(ub.get("tok")),
                  f(uz.get("tok")), f(ub.get("mb"), 1), f(uz.get("mb"), 1), "matched residency" if dm <= 20.0 else "NOT matched (> 20 MB a step apart): the tok/s delta is INCONCLUSIVE",
                  f(ub.get("spread")), f(uz.get("spread")), "the same in both arms" if same else "DIFFER between the arms",
                  "" if same else " -- VIRHE the 1u answers differ between %s and %s (strict, T 0 greedy)" % (base, zarm)))
    # the int15 references (another start): INFO
    rv = {"c8": (res["arms"].get(base) or {}).get("agg"), "c16": (c16b[0].get("agg") if c16b else None), "u1": (ub or {}).get("tps")}
    print("LCH REFS (INFO: another start, unstamped -- this item stamps one pass in N on card 0; the in-process B is the reading's base): %s" % " | ".join(
        "%s %s (this B %s: %s)" % (lab, f(v), f(rv.get(k)), fp(pc(rv.get(k), v))) for lab, k, v in REFS))
    nbad = sum(len(b["bad"]) for b in res["blocks"]) + sum(len(u["bad"]) for u in U1.values()) + (1 if ub and uz and ub["md5"] != uz["md5"] else 0)
    Z = res["arms"].get(zarm) or {}
    tag = ("within the same-arm agg spread x2 (%s %%): INCONCLUSIVE on agg" % f(2 * msp)) if msp is not None and Z.get("d_pct") is not None and abs(Z["d_pct"]) < 2 * msp else (
          ("outside the same-arm agg spread x2 (%s %%)" % f(2 * msp)) if msp is not None and Z.get("d_pct") is not None else "no spread (one block an arm)")
    nb = B0.get("ncm_step") if B0.get("ncm_step") is not None else B0.get("own_step")
    nz = Z.get("ncm_step") if Z.get("ncm_step") is not None else Z.get("own_step")
    cz = res["c16"].get(zarm) or {}
    print("LCH HEADLINE %s (route_copy_ov %s) vs %s (route_copy_ov %s), in-process on the int15 pin | c8 work-based agg %s -> %s tok/s (%s, %s) | c8 decode p50 %s -> %s ms (%s, "
          "block p50s %s) | c8 full-row p50 %s -> %s ms (%s) | c8 matched newcomer %s -> %s ms a step (%s) | c8 model %s | c16 agg %s, decode p50 %s, full-row p50 %s | "
          "1u %s -> %s tok/s (%s) | %d guard finding(s)%s" % (
              zarm, xz, base, xb, f(B0.get("agg")), f(Z.get("agg")), fp(Z.get("d_pct")), tag, f(B0.get("dec_p50")), f(Z.get("dec_p50")), fp(Z.get("dec_d_pct")),
              sep(Z.get("dec_p50s"), B0.get("dec_p50s")), f(B0.get("bkt_p50")), f(Z.get("bkt_p50")), fp(Z.get("bkt_d_pct")), f(nb, 1), f(nz, 1), fp(pc(nz, nb)),
              fp(pc(Z.get("model_agg"), B0.get("model_agg"))), fp(cz.get("d_pct")), fp(cz.get("dec_d_pct")), fp(cz.get("bkt_d_pct")),
              f((ub or {}).get("tps")), f((uz or {}).get("tps")), fp(u1d), nbad, " -- VIRHE see the BLOCK / 1u lines" if nbad else ""))
    if jout:
        for b in res["blocks"]: b.pop("done", None)
        with open(jout, "w") as fo: json.dump(res, fo)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="lchsum-")
    os.makedirs(os.path.join(d, "c8"))
    t00 = 1794517000.0
    L = []
    def iso(t):
        x = "%.9f" % (t - int(t))
        return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "." + x[2:] + "Z"
    def stats(stream):
        return {"link": {"staged_bytes": 0, "stream_bytes": stream, "h2d_bytes": stream / 3},
                "experts": {"ranks": [{"rank": r, "routed": 1000, "routed_res": 990, "slots": {"miss": {"arena": 0}}, "lent": {"arena": {"drop_recall": 0, "fills": 0}}} for r in range(4)]}}
    rows = []
    def blk(i, arm, n, t0, nc_ms, dec_ms, n_nc, x3d):
        """decode steps of dec_ms (n_seq n, n_tok n), a newcomer (one MIXED step of nc_ms) after every 41 decode steps; GET exl3_x3d x3d at the end"""
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
                t += dec_ms / 1000.0; after += 1
            step += 1
        w0, w1 = nct[0] - 0.4, nct[n_nc - 1] + 0.5
        rq.append(dict(ev="mark", k=0, t=w0)); rq.append(dict(ev="mark", k=1, t=w1))
        with open(os.path.join(d, "c8", nm + "-reqs.jsonl"), "w") as fo:
            for r in rq: fo.write(json.dumps(r) + "\n")
        span = w1 - w0
        rows.append(dict(name=nm, n=n, arm=arm, sched="G", agg_dec_tps=round(n_nc * 512 / span, 1), window_s=round(span, 1), marks=[w0, w1],
                         requests=n_nc, errors=0, decode_share=0.8))
        json.dump(stats(0), open(os.path.join(d, "c8", nm + "-st-w0.json"), "w"))
        json.dump(stats(n_nc * 7e9), open(os.path.join(d, "c8", nm + "-st-w1.json"), "w"))
        json.dump({"switches": {"route_copy_ov": x3d, "exl3_x3d": 2}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    blk(1, "B", 8, t00, 508.0, 34.0, 16, 0)
    blk(2, "Z", 8, t00 + 100, 508.0, 33.6, 16, 1)
    blk(3, "Z", 8, t00 + 200, 508.0, 33.6, 16, 1)
    blk(4, "B", 8, t00 + 300, 508.0, 34.0, 16, 0)
    blk(1, "B", 16, t00 + 400, 540.0, 52.0, 16, 0)
    blk(2, "Z", 16, t00 + 500, 540.0, 51.5, 16, 0)     # Z's c16 block read back route_copy_ov 0: the switch guard must fire
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        i = 0
        for arm, x3d, reads in (("B", 0, ((79.3, 19.9, 180.0), (89.7, 17.6, 20.0), (89.9, 17.55, 6.0), (89.8, 17.57, 1.0))),
                                ("Z", 1, ((89.0, 17.8, 3.0), (90.6, 17.4, 2.0), (90.7, 17.38, 1.0), (90.8, 17.37, 0.5)))):
            for k, (tps, ms, mb) in enumerate(reads):
                nm = "b%d-%s-%s" % (i, "U1W" if k == 0 else "U1", arm); i += 1
                fo.write(json.dumps(dict(i=i, lever="U1W" if k == 0 else "U1", arm=arm, name=nm, j=dict(prompt="essay", decode_tps=tps, ms_per_decode_step=ms,
                         tokens_per_step=1.58, acceptance=0.698, status=200, tokens=512, text_md5="e"), st=dict(stream_mb_step=mb, promotions=3))) + "\n")
                json.dump({"switches": {"route_copy_ov": x3d, "exl3_x3d": 2}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    lp = os.path.join(d, "eng.log.gz")
    with gzip.open(lp, "wt") as fo: fo.write("\n".join(L) + "\n")
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = run([d, "--log", lp, "--sw", "1", "--post", "10", "--essay-md5", "e", "--json", os.path.join(d, "o.json")])
    t = buf.getvalue()
    o = json.load(open(os.path.join(d, "o.json")))
    why = []
    A = o["arms"]
    if rc != 0: why.append("rc %s" % rc)
    if set(A) != {"B", "Z"}: why.append("arms %s" % list(A))
    else:
        if abs((A["Z"].get("dec_p50") or 0) - 33.6) > 0.05 or abs((A["B"].get("dec_p50") or 0) - 34.0) > 0.05: why.append("decode p50 %s / %s" % (A["Z"].get("dec_p50"), A["B"].get("dec_p50")))
        if A["Z"].get("bkt_rows") != 8 or abs((A["Z"].get("bkt_p50") or 0) - 33.6) > 0.05: why.append("full-row bucket %s %s" % (A["Z"].get("bkt_rows"), A["Z"].get("bkt_p50")))
        if abs((A["Z"].get("own_step") or 0) - 508.0) > 1.0: why.append("own newcomer step %s" % A["Z"].get("own_step"))
        if not A["Z"].get("model_agg") or not A["B"].get("model_agg") or A["Z"]["model_agg"] <= A["B"]["model_agg"]: why.append("model %s / %s" % (A["Z"].get("model_agg"), A["B"].get("model_agg")))
        if not A["Z"].get("d_pct") or A["Z"]["d_pct"] <= 0: why.append("Z agg delta %s" % A["Z"].get("d_pct"))
    bl = {b["name"]: b for b in o["blocks"]}
    if [b["n_nc"] for b in o["blocks"]][:4] != [16] * 4: why.append("newcomers in window %s" % [b["n_nc"] for b in o["blocks"]])
    if not any("route_copy_ov 0 at the block's end" in x for x in bl["c16-2-G-Z"]["bad"]): why.append("c16 Z switch guard did not fire")
    if any(bl[n]["bad"] for n in bl if n.startswith("c8-")): why.append("a c8 block flagged %s" % [bl[n]["bad"] for n in bl if n.startswith("c8-")])
    u = o["u1"]
    if abs((u["B"].get("tps") or 0) - 89.8) > 0.01 or abs((u["Z"].get("tps") or 0) - 90.7) > 0.01 or u["B"]["bad"] or u["Z"]["bad"]: why.append("1u %s" % u)
    if "LCH 1u Z vs B: 89.80 -> 90.70 tok/s (+1.00 %)" not in t or "matched residency" not in t: why.append("1u line")
    if "SEPARATED" not in t or "LCH REFS (INFO" not in t: why.append("separation / refs")
    if "LCH HEADLINE" not in t or "LCH MODEL c8 Z" not in t or "LCH C16 Z vs B" not in t or "1 guard finding(s)" not in t: why.append("lines")
    # a 1u answer that differs, and a Z read whose switch read back 1: both VIRHE
    with open(os.path.join(d, "blocks.jsonl"), "a") as fo:
        fo.write(json.dumps(dict(lever="U1", arm="Z", name="b9-U1-Z", j=dict(decode_tps=90.0, ms_per_decode_step=17.4, tokens_per_step=1.58, text_md5="x"), st=dict(stream_mb_step=0.0))) + "\n")
    json.dump({"switches": {"route_copy_ov": 0, "exl3_x3d": 1}}, open(os.path.join(d, "fn-b9-U1-Z-1.json"), "w"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--log", lp, "--sw", "1", "--essay-md5", "e"])
    t2 = buf.getvalue()
    if "differ from ref14's essay answer" not in t2 or "route_copy_ov read back" not in t2 or "exl3_x3d read back" not in t2 or "DIFFER between the arms" not in t2: why.append("1u guards %s" % [l for l in t2.split("\n") if "1u" in l])
    __import__("shutil").rmtree(d, ignore_errors=True)   # the temp dir goes at once (gpu02's /tmp is a tmpfs: MOCK SANDBOX RULE's inodes)
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: gens blocks B Z Z B + c16 B / Z -- decode p50 all / full-row bucket / after-newcomer / rest, block separation, own matched "
          "newcomer sums, cycle model, the route_copy_ov read-back guard fires on a Z block that read back 0, the int15 pin guard (exl3_x3d 2) on a 1u read "
          "that read back 1, 1u sets B / Z (discarded warm-up apart, answer md5 + switch guards, residency match), the int15 references INFO")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
