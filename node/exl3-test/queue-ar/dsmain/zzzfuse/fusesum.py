#!/usr/bin/env python3
"""fusesum.py (13.11.2026, queue-ar/dsmain/zzzfuse, a fusion builder for ds-main-port) -- the zzzfuse summary: levers FUSE-R (fn4
ds41_rfuse 1 / 2: the router chain in fewer launches), FUSE-M (libr4d fn4 ds41_mix_fin 1: mhc_mix's one-launch FIN form) and RCO (core fn4
route_copy_ov 1, ds41-launch 0f65666: the routing copy recorded with its ranges) read in-process on THE int15 PIN against B (all 0). A COPY of zzk25s/k25sum.py (md5 eac39a76; its step-log readers, gens windows, own matched-prompt tie,
link counters and cycle model kept verbatim) with K25U's exl3_x3d read back replaced by the fusion switches' (GET /admin/fn4 switches
ds41_rfuse / ds41_mix_fin / exl3_x3d at every block's end, the arm's values from --want) and the GET "fuse" counter deltas between the
block's two snapshots (engagement: a form an arm runs must move its counter, a form it does not run must not; a recorded pass counts at
recording and every block re-records), the single-fusion arms (one c8 block each: decode p50 vs the base arm's blocks), the int15
references. Arms: --base B (both 0), --all F (the ALL set); every other arm in --want is read as a single.

  fusesum.py OUT --log ENGINE_LOG [--sw 1] [--post 10] [--base B] [--all F] [--want B=0/0/0,F=2/0/1,Y=2/1/1,R1=1/0/0,R2=2/0/0,M=0/1/0,R=0/0/1] [--xb 2]
             [--c8gens 2] [--c16gens 1] [--essay-md5 47a39dac581a] [--json OUT.json]
      OUT = the item's out dir: mb.jsonl (mbx's rows), c8/NAME-reqs.jsonl, c8/NAME-st-w0.json / -st-w1.json, fn-NAME-0.json / fn-NAME-1.json
      (GET /admin/fn4 before / after the block), ncm-c8g.json (optional), blocks.jsonl (blockx's 1u reads, optional). Prints lines
      "ZFUSE ..."; a guard failure is a "VIRHE" inside its line.
  fusesum.py selftest
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



# ---------------------------------------------------------------- zzzfuse's own: the fusion switches read back, the counters, the references
REFS = (("zzk25s c8 gens Z (= int15, B Z Z B)", "c8", 191.60), ("zzk25s c16 gens Z (= int15)", "c16", 236.70), ("zzk25s 1u essay Z (= int15)", "u1", 89.91))
FUSEK = ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches", "mixfin_rows")
RCOK = ("recordings", "copies", "freed")


def fusesw(p):
    """(ds41_rfuse, ds41_mix_fin, route_copy_ov, exl3_x3d) read back in a GET /admin/fn4 snapshot, or None"""
    d = rdj(p)
    if not d: return None
    w = d.get("switches") or {}
    return (w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d"))


def fusectr(p0, p1):
    """GET "fuse" and "rco" counter deltas between two snapshots (+ rco in_force at the end), or None"""
    a, b = rdj(p0), rdj(p1)
    if not a or not b or not isinstance(b.get("fuse"), dict): return None
    fa, fb = a.get("fuse") or {}, b.get("fuse")
    out = {k: (fb.get(k) or 0) - (fa.get(k) or 0) for k in FUSEK}
    ra, rb = a.get("rco") or {}, b.get("rco") or {}
    for k in RCOK: out["rco_" + k] = (rb.get(k) or 0) - (ra.get(k) or 0)
    out["rco_in_force"] = rb.get("in_force")
    return out


def engaged(want, d):
    """the engagement guard: [problems] for an arm running (rf, mf, rc) whose block moved the counters d"""
    if d is None: return ["no GET \"fuse\" counters around the block"]
    rf, mf, rc = want
    bad = []
    if rc == 1 and not (d.get("rco_in_force") == 1 and d.get("rco_recordings", 0) > 0 and d.get("rco_freed", 0) > 0): bad.append("route_copy_ov 1 but no recording freed a copy (in_force %s, recordings +%s, freed +%s)" % (d.get("rco_in_force"), d.get("rco_recordings"), d.get("rco_freed")))
    if rc == 0 and (d.get("rco_in_force") or d.get("rco_recordings")): bad.append("route_copy_ov 0 but rco in force / recording (%s, +%s)" % (d.get("rco_in_force"), d.get("rco_recordings")))
    if rf == 2 and not (d["rf_form2_passes"] > 0 and d["rgs_launches"] > 0): bad.append("ds41_rfuse 2 but form 2 did not run (passes +%s, launches +%s)" % (d["rf_form2_passes"], d["rgs_launches"]))
    if rf == 1 and not d["rf_form1_passes"] > 0: bad.append("ds41_rfuse 1 but form 1 did not run (passes +%s)" % d["rf_form1_passes"])
    if rf == 1 and d["rf_form2_passes"] > 0: bad.append("ds41_rfuse 1 but form 2 ran (+%s)" % d["rf_form2_passes"])
    if rf == 0 and (d["rf_form1_passes"] or d["rf_form2_passes"] or d["rgs_launches"]): bad.append("ds41_rfuse 0 but a fused router ran (%s)" % d)
    if mf == 1 and not d["mixfin_launches"] > 0: bad.append("ds41_mix_fin 1 but no FIN launch (+%s)" % d["mixfin_launches"])
    if mf == 0 and d["mixfin_launches"]: bad.append("ds41_mix_fin 0 but FIN launches +%s" % d["mixfin_launches"])
    return bad


def fstr(d): return ("/".join("+%s" % d[k] for k in FUSEK) + " | rco in_force %s rec/copies/freed %s" % (d.get("rco_in_force"), "/".join("+%s" % d.get("rco_" + k) for k in RCOK))) if d else "n/a"


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


def u1sum(out, arms, want, emd5, sw, xb):
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
        x = [fusesw(os.path.join(out, "fn-%s-1.json" % r.get("name"))) if r.get("name") else None for r in W + U]
        wa = tuple(want.get(a, (None, None, None))) + (xb,)
        bad, info = [], []
        if any(v is None or tuple(v) != wa for v in x): (bad if sw else info).append("ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d read back %s after the reads (want %s)" % (x, list(wa)))
        if any(m != emd5 for m in md): (bad if sw else info).append("answers %s differ from ref14's essay answer %s (strict, T 0 greedy)" % (md, emd5))
        tps, ms, tok = num(g(r, "decode_tps") for r in U), num(g(r, "ms_per_decode_step") for r in U), num(g(r, "tokens_per_step") for r in U)
        mb, acc = num(s(r, "stream_mb_step") for r in U), num(g(r, "acceptance") for r in U)
        res[a] = dict(n=len(U), tps=mean(tps), tps_reads=tps, spread=round(100.0 * (max(tps) - min(tps)) / statistics.mean(tps), 2) if len(tps) > 1 else None,
                      ms=mean(ms), ms_reads=ms, tok=mean(tok), acc=mean(acc), mb=mean(mb), mb_reads=mb, prom=[s(r, "promotions") for r in U],
                      w_tps=g(W[0], "decode_tps") if W else None, w_mb=s(W[0], "stream_mb_step") if W else None, md5=md, sw=x, bad=bad, info=info)
    return res


def run(argv):
    out, log, sw, post, base, zarm, jout, gens, xb, emd5 = None, None, 1, 10, "B", "F", None, {8: 2, 16: 1}, 2, "47a39dac581a"
    wspec = "B=0/0/0,F=2/0/1,Y=2/1/1,R1=1/0/0,R2=2/0/0,M=0/1/0,R=0/0/1"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--log": log = argv[i + 1]; i += 2
        elif a == "--sw": sw = int(argv[i + 1]); i += 2
        elif a == "--post": post = int(argv[i + 1]); i += 2
        elif a == "--base": base = argv[i + 1]; i += 2
        elif a == "--all": zarm = argv[i + 1]; i += 2
        elif a == "--want": wspec = argv[i + 1]; i += 2
        elif a == "--xb": xb = int(argv[i + 1]); i += 2
        elif a == "--essay-md5": emd5 = argv[i + 1]; i += 2
        elif a == "--json": jout = argv[i + 1]; i += 2
        elif a == "--c8gens": gens[8] = int(argv[i + 1]); i += 2
        elif a == "--c16gens": gens[16] = int(argv[i + 1]); i += 2
        else: out = a; i += 1
    if not out or not log:
        print("ZFUSE VIRHE usage: OUT --log LOG"); return 2
    want = {}
    for e in wspec.split(","):
        k, v = e.split("=")
        want[k] = tuple(int(x) for x in v.split("/"))
    U1 = u1sum(out, (base, zarm), want, emd5, sw, xb)
    steps = read_steps(log) if os.path.exists(log) else []
    if not steps:
        print("ZFUSE VIRHE no step line in %s (RADIANCE_LOG_STEPS=1, docker logs -t)" % log); return 1
    ts = [s[0] for s in steps]
    R = [r for r in jl(os.path.join(out, "mb.jsonl")) if r.get("sched") == "G" and r.get("n") in (8, 16)]
    if not R:
        print("ZFUSE VIRHE no gens block in %s/mb.jsonl" % out); return 1
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
        b["sw"] = fusesw(os.path.join(out, "fn-%s-1.json" % nm))
        b["fuse"] = fusectr(os.path.join(out, "fn-%s-0.json" % nm), os.path.join(out, "fn-%s-1.json" % nm))
        if (b.get("errors") or 0) != 0 or not (b.get("requests") or 0) > 0: bad.append("request errors %s of %s requests" % (b.get("errors"), b.get("requests")))
        if arm not in want: bad.append("arm %s is not in --want %s" % (arm, wspec))
        elif sw:
            wa = tuple(want[arm]) + (xb,)
            if b["sw"] is None or tuple(b["sw"]) != wa: bad.append("ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d %s at the block's end (GET /admin/fn4; want %s under %s)" % (b["sw"], list(wa), arm))
            bad += engaged(want[arm], b["fuse"])
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
        print("ZFUSE BLOCK %s (%s, c%d gens): agg %s tok/s, window %s s, %s requests, %s errors, decode_share %s, TTFT p50 %s s, acceptance %s | "
              "decode steps %s: p50 %s / mean %s ms | full-row bucket (n_seq %d, n_tok %s): %s steps p50 %s ms | %d after each newcomer: %s steps p50 %s / "
              "mean %s ms; the rest p50 %s / mean %s ms | newcomer steps in the window %s (rows %s), unclipped %s ms (%s ms a step)%s | own matched %s "
              "prompts: %s ms over %s steps | window link: staged %s GB, stream %s GB, h2d %s GB (all ranks), arena slot misses %s, arena drop_recall %s, "
              "fills %s, routed-resident %s %% | ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d at the block's end %s (want %s) | fuse counters f1/f2/rgs/mixfin/rows %s%s" % (
                  b["name"], b["arm"], b["n"], b.get("agg"), b.get("window_s"), b.get("requests"), b.get("errors"), b.get("decode_share"),
                  b.get("ttft_p50"), b.get("acceptance"), b.get("n_dec"), f(b.get("dec_p50")), f(b.get("dec_mean")), b["n"], b.get("bkt_rows", "-"),
                  b.get("bkt_n", "-"), f(b.get("bkt_p50")), post, b.get("post_n"), f(b.get("post_p50")), f(b.get("post_mean")), f(b.get("other_p50")),
                  f(b.get("other_mean")), b.get("n_nc"), "%s..%s" % (min(b["nc_rows"]), max(b["nc_rows"])) if b.get("nc_rows") else "-", b.get("nc_sum"),
                  f(b.get("nc_mean"), 1), (" [NOTE %s]" % b["nc_note"]) if b.get("nc_note") else "", b.get("own_matched", "-"), b.get("own_sum", "-"),
                  b.get("own_steps", "-"), lk.get("staged_gb"), lk.get("stream_gb"), lk.get("h2d_gb"), lk.get("arena_miss"), lk.get("arena_drop_recall"),
                  lk.get("arena_fills"), lk.get("hit_pct"), b.get("sw"), list(want.get(b["arm"], ())) + [xb], fstr(b.get("fuse")), (" -- VIRHE " + "; ".join(b["bad"])) if b["bad"] else ""))
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
        print("ZFUSE ARM c8 %s (ds41_rfuse/ds41_mix_fin/route_copy_ov %s): agg %s tok/s (blocks %s%s)%s | decode p50 %s ms (blocks %s, spread %s %%; mean %s)%s | full-row bucket (8 seqs, %s rows) "
              "p50 %s ms (blocks %s)%s | %d-after-newcomer p50 %s ms, the rest p50 %s ms | newcomer matched (ncmatch.py) %s ms a step (sum %s ms, spread %s %%), "
              "own tie %s ms a step%s | window: %s newcomer steps, %s decode steps a newcomer, %s tokens a newcomer, stream %s / h2d %s GB (all ranks)" % (
                  a, "/".join(str(v) for v in want.get(a, ())), f(A.get("agg")), "/".join(f(v) for v in A["aggs"]), (", spread %s %%" % f(A["spread_pct"])) if A.get("spread_pct") is not None else "",
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
        print("ZFUSE MODEL c8 %s: cycle a newcomer = %s decode steps (%s's) x %s ms (mean) + %s ms (matched newcomer) = %s ms -> %s tokens / cycle = %s tok/s "
              "(measured %s)%s" % (a, f(B0.get("dec_per_nc") or A.get("dec_per_nc"), 1), base, f(A.get("dec_mean")),
                                   f(A.get("ncm_step") if A.get("ncm_step") is not None else A.get("own_step"), 1), f(A.get("cycle_ms"), 1), f(B0.get("tok_per_nc"), 0),
                                   f(A.get("model_agg"), 1), f(A.get("agg")),
                                   (" | model %s vs %s, measured %s" % (fp(md), base, fp(A.get("d_pct")))) if a != base else ""))
    # the singles: every c8 arm but the base and ALL, its block(s) against the base arm's blocks
    res["singles"] = {}
    for a in arms:
        if a in (base, zarm): continue
        A, bp = res["arms"][a], (B0.get("dec_p50s") or [])
        d = A.get("dec_d_pct")
        pos = "-" if not bp or A.get("dec_p50") is None else ("BELOW B's blocks" if A["dec_p50"] < min(bp) else "ABOVE B's blocks" if A["dec_p50"] > max(bp) else "INSIDE B's range")
        res["singles"][a] = dict(dec_p50=A.get("dec_p50"), dec_d_pct=d, agg_d_pct=A.get("d_pct"), pos=pos)
        print("ZFUSE SINGLE c8 %s (ds41_rfuse/ds41_mix_fin/route_copy_ov %s; %d block): decode p50 %s ms vs %s %s ms (blocks %s) = %s ms, %s -- %s | full-row p50 %s (%s) | "
              "agg %s tok/s (%s) | after-newcomer p50 %s ms, the rest %s ms" % (
                  a, "/".join(str(v) for v in want.get(a, ())), len(A["blocks"]), f(A.get("dec_p50")), base, f(B0.get("dec_p50")), "/".join(f(v) for v in bp),
                  f((A.get("dec_p50") or 0) - (B0.get("dec_p50") or 0)) if A.get("dec_p50") is not None and B0.get("dec_p50") is not None else "-", fp(d), pos,
                  f(A.get("bkt_p50")), fp(A.get("bkt_d_pct")), f(A.get("agg")), fp(A.get("d_pct")), f(A.get("post_p50")), f(A.get("other_p50"))))
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
        print("ZFUSE C16 %s vs %s (one gens block each): agg %s -> %s tok/s (%s) | decode p50 %s -> %s ms (%s) | full-row bucket (16 seqs, %s rows: %s / %s steps) "
              "p50 %s -> %s ms (%s) | %d-after-newcomer p50 %s -> %s ms | own matched (%s prompts) %s -> %s ms a newcomer step (%s) | window stream %s -> %s GB" % (
                  b["arm"], base, f(x0.get("agg")) if x0 else "-", f(b.get("agg")), fp(pc(b.get("agg"), x0.get("agg") if x0 else None)),
                  f(x0.get("dec_p50")) if x0 else "-", f(b.get("dec_p50")), fp(res["c16"][b["arm"]]["dec_d_pct"]), b.get("bkt_rows", "-"),
                  (x0 or {}).get("bkt_n", "-"), b.get("bkt_n", "-"), f((x0 or {}).get("bkt_p50")), f(b.get("bkt_p50")), fp(res["c16"][b["arm"]]["bkt_d_pct"]),
                  post, f(x0.get("post_p50")) if x0 else "-", f(b.get("post_p50")), b.get("own_matched", "-"), f(n0, 1), f(ns, 1), fp(pc(ns, n0)),
                  f(((x0 or {}).get("link") or {}).get("stream_gb")), f((b.get("link") or {}).get("stream_gb"))))
    if not c16: print("ZFUSE C16: no c16 gens block (skipped by the budget or not run)")
    # 1u essay sets
    for a in (base, zarm):
        u = U1.get(a)
        if not u: print("ZFUSE 1u essay %s: not run" % a); continue
        print("ZFUSE 1u essay %s (ds41_rfuse/ds41_mix_fin/route_copy_ov %s; one discarded + %d back to back, resprobe essay 512, T 0): %s tok/s (reads %s, spread %s %%), %s ms a step (reads %s), "
              "%s tok a step, acceptance %s, stream %s MB a step (reads %s), promotions a read %s | discarded warm-up %s tok/s at %s MB a step | answers %s | "
              "ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d read back after each read %s%s" % (
                  a, "/".join(str(v) for v in want.get(a, ())), u["n"], f(u["tps"]), "/".join(f(v) for v in u["tps_reads"]), f(u["spread"]), f(u["ms"]), "/".join(f(v) for v in u["ms_reads"]),
                  f(u["tok"]), f(u["acc"], 3), f(u["mb"], 1), "/".join(f(v, 1) for v in u["mb_reads"]), "/".join(str(v) for v in u["prom"]), f(u["w_tps"]),
                  f(u["w_mb"], 1), ("IDENTICAL to ref14's essay answer (%s) on all %d reads" % (emd5, len(u["md5"]))) if all(m == emd5 for m in u["md5"]) else "md5 %s" % u["md5"],
                  u["sw"], (" -- VIRHE " + "; ".join(u["bad"])) if u["bad"] else ((" | INFO (mock) " + "; ".join(u["info"])) if u["info"] else "")))
    ub, uz = U1.get(base), U1.get(zarm)
    u1d = None
    if ub and uz:
        u1d = pc(uz.get("tps"), ub.get("tps"))
        dm = abs((uz.get("mb") or 0) - (ub.get("mb") or 0))
        same = ub["md5"] == uz["md5"]
        print("ZFUSE 1u %s vs %s: %s -> %s tok/s (%s), %s -> %s ms a step (%s), tok a step %s / %s | stream MB a step %s / %s (%s) | spread %s %% / %s %% | "
              "answers %s%s" % (
                  zarm, base, f(ub.get("tps")), f(uz.get("tps")), fp(u1d), f(ub.get("ms")), f(uz.get("ms")), fp(pc(uz.get("ms"), ub.get("ms"))), f(ub.get("tok")),
                  f(uz.get("tok")), f(ub.get("mb"), 1), f(uz.get("mb"), 1), "matched residency" if dm <= 20.0 else "NOT matched (> 20 MB a step apart): the tok/s delta is INCONCLUSIVE",
                  f(ub.get("spread")), f(uz.get("spread")), "the same in both arms" if same else "DIFFER between the arms",
                  "" if same else " -- VIRHE the 1u answers differ between %s and %s (strict, T 0 greedy)" % (base, zarm)))
    # the int14 references (other starts): INFO
    rv = {"c8": (res["arms"].get(base) or {}).get("agg"), "c16": (c16b[0].get("agg") if c16b else None), "u1": (ub or {}).get("tps")}
    print("ZFUSE REFS (INFO: other starts; the in-process B is the reading's base; B = int15): %s" % " | ".join(
        "%s %s (this B %s: %s)" % (lab, f(v), f(rv.get(k)), fp(pc(rv.get(k), v))) for lab, k, v in REFS))
    nbad = sum(len(b["bad"]) for b in res["blocks"]) + sum(len(u["bad"]) for u in U1.values()) + (1 if ub and uz and ub["md5"] != uz["md5"] else 0)
    Z = res["arms"].get(zarm) or {}
    tag = ("within the same-arm agg spread x2 (%s %%): INCONCLUSIVE on agg" % f(2 * msp)) if msp is not None and Z.get("d_pct") is not None and abs(Z["d_pct"]) < 2 * msp else (
          ("outside the same-arm agg spread x2 (%s %%)" % f(2 * msp)) if msp is not None and Z.get("d_pct") is not None else "no spread (one block an arm)")
    nb = B0.get("ncm_step") if B0.get("ncm_step") is not None else B0.get("own_step")
    nz = Z.get("ncm_step") if Z.get("ncm_step") is not None else Z.get("own_step")
    cz = res["c16"].get(zarm) or {}
    sg = " | singles c8 decode p50: " + ", ".join("%s %s (%s)" % (k, fp(v.get("dec_d_pct")), v.get("pos")) for k, v in res["singles"].items()) if res["singles"] else ""
    print("ZFUSE HEADLINE %s (ds41_rfuse/ds41_mix_fin/route_copy_ov %s) vs %s (%s), in-process on THE int15 PIN | c8 work-based agg %s -> %s tok/s (%s, %s) | c8 decode p50 %s -> %s ms (%s, "
          "block p50s %s) | c8 full-row p50 %s -> %s ms (%s) | c8 matched newcomer %s -> %s ms a step (%s) | c8 model %s | c16 agg %s, decode p50 %s, full-row p50 %s | "
          "1u %s -> %s tok/s (%s)%s | %d guard finding(s)%s" % (
              zarm, "/".join(str(v) for v in want.get(zarm, ())), base, "/".join(str(v) for v in want.get(base, ())), f(B0.get("agg")), f(Z.get("agg")), fp(Z.get("d_pct")), tag, f(B0.get("dec_p50")), f(Z.get("dec_p50")), fp(Z.get("dec_d_pct")),
              sep(Z.get("dec_p50s"), B0.get("dec_p50s")), f(B0.get("bkt_p50")), f(Z.get("bkt_p50")), fp(Z.get("bkt_d_pct")), f(nb, 1), f(nz, 1), fp(pc(nz, nb)),
              fp(pc(Z.get("model_agg"), B0.get("model_agg"))), fp(cz.get("d_pct")), fp(cz.get("dec_d_pct")), fp(cz.get("bkt_d_pct")),
              f((ub or {}).get("tps")), f((uz or {}).get("tps")), fp(u1d), sg, nbad, " -- VIRHE see the BLOCK / 1u lines" if nbad else ""))
    if jout:
        for b in res["blocks"]: b.pop("done", None)
        with open(jout, "w") as fo: json.dump(res, fo)
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="fusesum-")
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
    ctr = dict((k, 0) for k in FUSEK)
    rco = dict((k, 0) for k in RCOK)
    def fn(nm, sw3, run=None):
        """GET snapshots around a block: -0 before, -1 after with the counters moved by the forms `run` (rf, mf) ran"""
        json.dump({"switches": {"ds41_rfuse": 0, "ds41_mix_fin": 0, "route_copy_ov": 0, "exl3_x3d": 2}, "fuse": dict(ctr), "rco": dict(rco, in_force=0)}, open(os.path.join(d, "fn-%s-0.json" % nm), "w"))
        rf, mf, rc = run if run else sw3[:3]
        if rf == 1: ctr["rf_form1_passes"] += 7
        if rf == 2: ctr["rf_form2_passes"] += 7; ctr["rgs_launches"] += 280
        if mf == 1: ctr["mixfin_launches"] += 560; ctr["mixfin_rows"] += 4480
        if rc == 1: rco["recordings"] += 7; rco["copies"] += 560; rco["freed"] += 555
        json.dump({"switches": {"ds41_rfuse": sw3[0], "ds41_mix_fin": sw3[1], "route_copy_ov": sw3[2], "exl3_x3d": sw3[3]}, "fuse": dict(ctr), "rco": dict(rco, in_force=sw3[2])}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
    def blk(i, arm, n, t0, nc_ms, dec_ms, n_nc, sw3, run=None):
        """decode steps of dec_ms (n_seq n, n_tok n), a newcomer (one MIXED step of nc_ms) after every 41 decode steps"""
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
        fn(nm, sw3, run)
    blk(1, "B", 8, t00, 508.0, 33.7, 16, (0, 0, 0, 2))
    blk(2, "F", 8, t00 + 100, 508.0, 33.4, 16, (2, 0, 1, 2))
    blk(3, "F", 8, t00 + 200, 508.0, 33.4, 16, (2, 0, 1, 2))
    blk(4, "B", 8, t00 + 300, 508.0, 33.75, 16, (0, 0, 0, 2))
    blk(5, "R2", 8, t00 + 400, 508.0, 33.42, 16, (2, 0, 0, 2))
    blk(6, "R1", 8, t00 + 500, 508.0, 33.58, 16, (1, 0, 0, 2))
    blk(7, "M", 8, t00 + 600, 508.0, 34.0, 16, (0, 1, 0, 2))
    blk(8, "R", 8, t00 + 650, 508.0, 33.5, 16, (0, 0, 1, 2))
    blk(1, "B", 16, t00 + 700, 540.0, 51.5, 16, (0, 0, 0, 2))
    blk(2, "F", 16, t00 + 800, 540.0, 51.1, 16, (2, 0, 1, 2), run=(0, 0, 1))   # F's c16 block: the switch reads 2 but form 2 never ran -> the engagement guard fires
    with open(os.path.join(d, "mb.jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "blocks.jsonl"), "w") as fo:
        i = 0
        for arm, sw3, reads in (("B", (0, 0, 0, 2), ((79.3, 19.9, 180.0), (89.7, 17.6, 20.0), (89.9, 17.55, 6.0), (89.8, 17.57, 1.0))),
                                ("F", (2, 0, 1, 2), ((89.0, 17.8, 3.0), (90.6, 17.4, 2.0), (90.7, 17.38, 1.0), (90.8, 17.37, 0.5)))):
            for k, (tps, ms, mb) in enumerate(reads):
                nm = "b%d-%s-%s" % (i, "U1W" if k == 0 else "U1", arm); i += 1
                fo.write(json.dumps(dict(i=i, lever="U1W" if k == 0 else "U1", arm=arm, name=nm, j=dict(prompt="essay", decode_tps=tps, ms_per_decode_step=ms,
                         tokens_per_step=1.58, acceptance=0.698, status=200, tokens=512, text_md5="e"), st=dict(stream_mb_step=mb, promotions=3))) + "\n")
                json.dump({"switches": {"ds41_rfuse": sw3[0], "ds41_mix_fin": sw3[1], "route_copy_ov": sw3[2], "exl3_x3d": sw3[3]}}, open(os.path.join(d, "fn-%s-1.json" % nm), "w"))
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
    if set(A) != {"B", "F", "R2", "R1", "M", "R"}: why.append("arms %s" % list(A))
    else:
        if abs((A["F"].get("dec_p50") or 0) - 33.4) > 0.05 or abs((A["B"].get("dec_p50") or 0) - 33.73) > 0.05: why.append("decode p50 %s / %s" % (A["F"].get("dec_p50"), A["B"].get("dec_p50")))
        if A["F"].get("bkt_rows") != 8 or abs((A["F"].get("bkt_p50") or 0) - 33.4) > 0.05: why.append("full-row bucket %s %s" % (A["F"].get("bkt_rows"), A["F"].get("bkt_p50")))
        if abs((A["F"].get("own_step") or 0) - 508.0) > 1.0: why.append("own newcomer step %s" % A["F"].get("own_step"))
        if not A["F"].get("d_pct") or A["F"]["d_pct"] <= 0: why.append("F agg delta %s" % A["F"].get("d_pct"))
    sg = o.get("singles") or {}
    if (sg.get("R") or {}).get("pos") != "BELOW B's blocks" or (sg.get("R2") or {}).get("pos") != "BELOW B's blocks" or (sg.get("M") or {}).get("pos") != "ABOVE B's blocks" or (sg.get("R1") or {}).get("pos") != "BELOW B's blocks": why.append("singles %s" % sg)
    bl = {b["name"]: b for b in o["blocks"]}
    if [b["n_nc"] for b in o["blocks"]][:4] != [16] * 4: why.append("newcomers in window %s" % [b["n_nc"] for b in o["blocks"]])
    if not any("form 2 did not run" in x for x in bl["c16-2-G-F"]["bad"]): why.append("c16 F engagement guard did not fire %s" % bl["c16-2-G-F"]["bad"])
    if any(bl[n]["bad"] for n in bl if n.startswith("c8-")): why.append("a c8 block flagged %s" % [bl[n]["bad"] for n in bl if n.startswith("c8-")])
    u = o["u1"]
    if abs((u["B"].get("tps") or 0) - 89.8) > 0.01 or abs((u["F"].get("tps") or 0) - 90.7) > 0.01 or u["B"]["bad"] or u["F"]["bad"]: why.append("1u %s" % u)
    if "ZFUSE 1u F vs B: 89.80 -> 90.70 tok/s (+1.00 %)" not in t or "matched residency" not in t: why.append("1u line")
    if "SEPARATED" not in t or "ZFUSE REFS (INFO" not in t or "ZFUSE SINGLE c8 R2" not in t: why.append("separation / refs / singles")
    if "ZFUSE HEADLINE" not in t or "ZFUSE MODEL c8 F" not in t or "ZFUSE C16 F vs B" not in t or "1 guard finding(s)" not in t: why.append("lines")
    # a 1u answer that differs, and an F read whose switch read back 0: both VIRHE
    with open(os.path.join(d, "blocks.jsonl"), "a") as fo:
        fo.write(json.dumps(dict(lever="U1", arm="F", name="b9-U1-F", j=dict(decode_tps=90.0, ms_per_decode_step=17.4, tokens_per_step=1.58, text_md5="x"), st=dict(stream_mb_step=0.0))) + "\n")
    json.dump({"switches": {"ds41_rfuse": 0, "ds41_mix_fin": 0, "route_copy_ov": 0, "exl3_x3d": 2}}, open(os.path.join(d, "fn-b9-U1-F-1.json"), "w"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run([d, "--log", lp, "--sw", "1", "--essay-md5", "e"])
    t2 = buf.getvalue()
    if "differ from ref14's essay answer" not in t2 or "read back" not in t2 or "DIFFER between the arms" not in t2: why.append("1u guards %s" % [l for l in t2.split("\n") if "1u" in l])
    __import__("shutil").rmtree(d, ignore_errors=True)   # the temp dir goes at once (gpu02's /tmp is a tmpfs: MOCK SANDBOX RULE's inodes)
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: gens blocks B F F B + singles R2 R1 M R + c16 B / F -- decode p50 all / full-row bucket / after-newcomer / rest, block separation, "
          "own matched newcomer sums, cycle model, singles against B's range, the switch / engagement guards (an F block whose form 2 never ran), "
          "1u sets B / F (discarded warm-up apart, answer md5 + switch guards, residency match), the int15 references INFO")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
