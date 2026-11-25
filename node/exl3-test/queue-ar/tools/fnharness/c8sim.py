#!/usr/bin/env python3
"""c8sim.py (12.11.2026, queue-ar/dsmain/harness, ds-main-port harness builder) -- NO-GPU simulation of the mb c8 closed loop
against the DS engine's step scheduler, to compare arrival schedules by the co-batch mix they give block after block.

Engine model (calibrated on 12.11.'s step logs, runs/xdsm/sp c8-engine.log.gz + runs/xdsm/kv4 *-engine.log.gz; `calib` re-derives
the numbers): steps run back to back; a step is planned DELTA after the previous one ends (decode: 0.1-0.5 ms, after a MIXED /
prefill step 0.2-3 ms; an idle engine plans 0.3-2 ms after the first arrival); planning admits the queued requests that ARRIVED
before that instant, FIFO, whole, while they fit MNBT 4160 rows minus the decode rows (stop at the first misfit); every running
request gains one token a step (the newcomer's first token comes with its prefill step); wall: decode 16 + 2.55 x rows ms
(+- 3 ms), MIXED / prefill 224.5 + 0.361 x prefill tokens ms (+- 60 ms) -- the logs: 8 rows 36.4 ms mean, 1 / 2 / 3 / 4 fresh
prompts 618 / 1184 / 1402 / 1772 ms. Client: a request that ends at a step's end sends the next one 0.6-2.5 ms later (measured
done -> send 0.28 ms p50 + the HTTP hop). Prompt tokens per (worker, request) = mb.py's on gpu02 (the KV4 / SP logs; k >= 3
drawn 919..1386 by a fixed hash). Optional: the start-of-server drafting probe of the FIRST block after a start (logs: 6-9
decode steps at q 3 / 6 right after the first burst -> 8-26 extra tokens a stream, periods 487-505 instead of 513) and
low-occupancy drafting (D7 / draft_occ: each decode row gains 0-4 tokens a step once <= OCC requests have run for more than
--hold steps; --hold 0 = the pessimistic case, drafting on every low-occupancy step).

Schedules (one block = a fresh closed loop on an idle server, window [warm, dur] by step-plan time, then a drain):
  mb      mb.py: 8 threads start together (sends spread over 4.5 ms), max_tokens 512 (all 296 requests ran to 512 on 12.11.)
  settle  option 2: the same, with a discarded settling window S s before [warm, dur] (one closed loop: S + dur)
  settle2 option 2b: a separate discarded closed loop of S s first (it eats the probe), drain, then an mb block
  fixed   option 1 (mbt2.py --sched fixed): lead + groups released on first-token events (each group arrives during the
          previous group's prefill step), group j's FIRST request max_tokens = gen - j x spacing, later ones gen; ignore_eos
Per block: the newcomer steps in the window (non-decode steps with fresh prompts) -> co-batched (>= 2 fresh) sizes and the
single count; the signature (co-batched sizes sorted, singles) is compared block to block.

  c8sim.py run [--modes mb,settle,settle2,fixed] [--runs 200] [--blocks 4] [--probe-first 1] [--occ 0] [--settle 60]
               [--dur 60 --warm 15] [--pattern 1x8] [--seed 1] [--wall-scale 1.0] [--json OUT]
  c8sim.py armscan [--scales 0.95,1.0,1.05] ...     the fixed schedule's signature under arms 5 % slower / faster
  c8sim.py calib --log ENGINE.log.gz --win T0:T1 [...]   the step-wall fits from a real step log (prints the constants)
  c8sim.py selftest
Standard library only.
"""
import argparse, calendar, collections, gzip, json, random, re, statistics, sys, time, zlib

sys.dont_write_bytecode = True
MNBT = 4160
PROMPT = {0: [1048, 935, 1021, 1247, 1170, 958, 994, 947], 1: [1280, 1180, 999, 1129, 1386, 919, 991, 1010],
          2: [1271, 1089, 971, 1139, 1157, 938, 1020, 1108]}


def ptok(w, k):
    if k in PROMPT and w < 8: return PROMPT[k][w]
    h = (w * 2654435761 + k * 40503) & 0xFFFFFFFF
    return 919 + h % 468


class Engine:
    def __init__(self, rng, probe=False, occ=0, scale=1.0, mnbt=MNBT):
        self.rng, self.probe, self.occ, self.scale, self.mnbt = rng, probe, occ, scale, mnbt
        self.steps = []          # (t_plan, wall, n_fresh, rows, kind)
        self.dec_steps = 0

    def dwall(self, rows):
        return max(0.008, (16.0 + 2.55 * rows + self.rng.gauss(0, 3.0)) / 1000.0) * self.scale

    def mwall(self, pf):
        return max(0.25, (224.5 + 0.361 * pf + self.rng.gauss(0, 60.0)) / 1000.0) * self.scale


def simulate(mode, rng, n=8, gen=512, dur=60.0, warm=15.0, settle=60.0, probe=False, occ=0, scale=1.0, pattern=None,
             seed=1, delay=0.1, spacing=None, keep_trace=False, window="time", gens=2, edge=0.5, hold=32):
    """one block -> dict(sig, newcomer sizes in window, ...)."""
    eng = Engine(rng, probe, occ, scale)
    t_end_block = (settle + dur) if mode == "settle" else dur
    if window == "gens": t_end_block = 200.0
    w0, w1 = ((settle + warm, settle + dur) if mode == "settle" else (warm, dur))
    # requests: dict(w, k, P, L, ta, tok, t_first, grp)
    queue, running = [], []
    arrivals = []            # (ta, req) future arrivals, sorted
    k_of = collections.defaultdict(int)
    stop_t = t_end_block
    first_tok_ev = {}        # group id -> first-token time of its first request
    pend_release = []        # (cond, group) for fixed mode
    groups, first_max = None, {}
    if mode == "fixed":
        sizes = parse_pattern(pattern or "1x%d" % n, n)
        order = list(range(n)); random.Random(seed).shuffle(order)
        groups, p = [], 0
        for s in sizes: groups.append(order[p:p + s]); p += s
        m = len(groups); sp = spacing or gen // m
        for j, g in enumerate(groups):
            for wk in g: first_max[wk] = gen - j * sp
    def new_req(wk, ta):
        k = k_of[wk]; k_of[wk] += 1
        L = first_max.get(wk, gen) if (mode == "fixed" and k == 0) else gen
        return dict(w=wk, k=k, P=ptok(wk, k), L=L, ta=ta, tok=0, t_first=None, grp=None)
    # initial arrivals
    if mode in ("mb", "settle", "settle2"):
        for i in range(n):
            arrivals.append((rng.uniform(0, 0.0045), new_req(i, 0)))     # 12.11.: the 8 first sends spread 4.1-5.0 ms
    else:
        # lead group 0 at t=0 (a primer if it has > 1 member), group 1 at +delay, group j>=2 at first token of group j-2 + delay
        primer = len(groups[0]) > 1
        if primer:
            arrivals.append((0.0, dict(w=-1, k=0, P=1050, L=1, ta=0.0, tok=0, t_first=None, grp=-1)))
        for j, g in enumerate(groups):
            pend_release.append((j, g))
    arrivals.sort(key=lambda x: x[0])
    released = set()
    t = 0.0
    def release_fixed(now):
        out = []
        primer = len(groups[0]) > 1
        for j, g in pend_release:
            if j in released: continue
            if not primer:
                if j == 0: tr = 0.0
                elif j == 1: tr = delay
                else: tr = (first_tok_ev.get(j - 2) + delay) if (j - 2) in first_tok_ev else None
            else:
                ref = j - 2 if j >= 1 else None
                if j == 0: tr = delay
                else: tr = (first_tok_ev.get(ref) + delay) if ref in first_tok_ev else None
            if tr is not None and tr <= now + 1e9:
                out.append((tr, j, g))
        return out
    trace = []
    pending_ev = []          # fixed: scheduled releases (time, j, group)
    idle = True
    t_plan = None
    last_kind = "decode"
    drafted_left = 0
    probe_armed = probe
    nc_seen = 0
    low = 0
    while True:
        if mode == "fixed":
            for tr, j, g in release_fixed(t):
                if j not in released:
                    released.add(j)
                    for wk in g:
                        r = new_req(wk, tr); r["grp"] = j; r["ta"] = tr
                        arrivals.append((tr, r))
            arrivals.sort(key=lambda x: x[0])
        # move arrived requests into the queue
        if not running and not queue:
            if not arrivals:
                if mode == "fixed" and len(released) < len(groups):
                    # waiting on an event that cannot come (should not happen)
                    break
                break
            # idle: plan at first arrival + delta_idle
            ta0 = arrivals[0][0]
            t = max(t, ta0)
            t_plan = t + rng.uniform(0.0003, 0.002)
        else:
            t_plan = t + (rng.uniform(0.0001, 0.0005) if last_kind == "decode" else rng.uniform(0.0002, 0.003))
        while arrivals and arrivals[0][0] <= t_plan:
            queue.append(arrivals.pop(0)[1])
        # admit FIFO while fits
        room = eng.mnbt - len(running)
        adm = []
        while queue and queue[0]["P"] <= room:
            r = queue.pop(0); adm.append(r); room -= r["P"]
        if not adm and not running:
            t = t_plan
            continue
        if adm:
            wall = eng.mwall(sum(r["P"] for r in adm)); kind = "mixed"
        else:
            wall = eng.dwall(len(running)); kind = "decode"
        te = t_plan + wall
        fresh = [r for r in adm if r["w"] >= 0]
        eng.steps.append((t_plan, wall, len(fresh), sum(r["P"] for r in adm) + len(running), kind, [(r["w"], r["k"]) for r in fresh]))
        # tokens
        extra = 0
        if kind == "decode" and probe_armed and len(running) >= 4 and drafted_left == 0 and not getattr(eng, "probed", False):
            drafted_left = 7; eng.probed = True
        low = low + 1 if (occ and len(running) <= occ) else 0
        for r in running:
            g = 1
            if kind == "decode":
                if drafted_left > 0: g += rng.randint(0, 4)
                elif occ and low > hold: g += rng.randint(0, 4)
            r["tok"] += g
        if kind == "decode" and drafted_left > 0: drafted_left -= 1
        for r in adm:
            r["tok"] = 1; r["t_first"] = te
            if mode == "fixed" and r["grp"] is not None and r["k"] == 0 and r["grp"] not in first_tok_ev:
                first_tok_ev[r["grp"]] = te
            if r.get("grp") == -1:
                first_tok_ev[-1] = te
        running.extend(adm)
        done = [r for r in running if r["tok"] >= r["L"]]
        running = [r for r in running if r["tok"] < r["L"]]
        for r in done:
            if r["w"] < 0: continue
            tn = te + rng.uniform(0.0006, 0.0025)
            if tn < stop_t:
                arrivals.append((tn, new_req(r["w"], tn)))
        arrivals.sort(key=lambda x: x[0])
        t = te
        last_kind = kind
        if t > stop_t + 60: break
        if window == "gens" and mode == "fixed" and any(k >= 1 for _, k in eng.steps[-1][5]) and te >= warm:
            nc_seen += 1
            if nc_seen == gens * len(groups) + 1:
                stop_t = min(stop_t, eng.steps[-1][0] + eng.steps[-1][1] + edge)
    if window == "gens":   # mbt2.py's: from the first newcomer step whose first token comes after warm
        ncs = [x for x in eng.steps if any(k >= 1 for _, k in x[5]) and x[0] + x[1] >= warm]
        mg = len(groups) if groups else n
        if len(ncs) > gens * mg:
            w0, w1 = ncs[0][0] + ncs[0][1] + edge, ncs[gens * mg][0] + ncs[gens * mg][1] + edge
        else:
            w0, w1 = 0.0, -1.0
    win = [s for s in eng.steps if w0 <= s[0] < w1]
    nc = [s for s in win if s[2] > 0]
    cob = sorted([s[2] for s in nc if s[2] > 1], reverse=True)
    single = sum(1 for s in nc if s[2] == 1)
    dec = [s for s in win if s[4] == "decode"]
    busy = sum(s[1] for s in win)
    out = dict(sig=(tuple(cob), single), cob=cob, single=single, newcomers=sum(s[2] for s in nc),
               decode_share=round(sum(s[1] for s in dec) / busy, 3) if busy else None,
               groups_seq="".join(str(s[2]) for s in eng.steps if s[2] > 0))
    if keep_trace: out["steps"] = eng.steps
    return out


def parse_pattern(p, n):
    p = (p or "").strip()
    if not p: return [1] * n
    m = re.fullmatch(r"(\d+)x(\d+)", p)
    if m: s = [int(m.group(1))] * int(m.group(2))
    else: s = [int(x) for x in p.split(",") if x.strip()]
    if sum(s) != n or any(x < 1 or x > 3 for x in s):
        raise SystemExit("pattern %r: group sizes 1..3 summing to %d" % (p, n))
    return s


def run_mode(mode, a, scale=1.0):
    sigs_by_block = collections.defaultdict(collections.Counter)
    same12 = same_all = 0
    ex = []
    for run in range(a.runs):
        rng = random.Random(a.seed0 * 1000003 + run * 7919 + zlib.crc32(mode.encode()) % 1000)
        sigs = []
        probe_left = bool(a.probe_first)
        for b in range(a.blocks):
            m = mode
            pr = probe_left
            if mode == "settle2":
                # the discarded loop eats the probe; the block itself starts from idle
                simulate("mb", rng, dur=a.settle, warm=0, probe=pr, occ=a.occ, scale=scale, hold=a.hold)
                pr = False; m = "mb"
            r = simulate(m, rng, dur=a.dur, warm=a.warm, settle=a.settle, probe=pr, occ=a.occ, scale=scale, hold=a.hold,
                         pattern=a.pattern, seed=a.seed, spacing=a.spacing,
                         window=(a.window if m == "fixed" else "time"), gens=a.gens, edge=a.edge)
            probe_left = False
            sigs.append(r["sig"])
            sigs_by_block[b][r["sig"]] += 1
            if run < 3: ex.append((run, b, r["groups_seq"][:40], r["sig"], r["decode_share"]))
        same12 += sigs[0] == sigs[1] if len(sigs) > 1 else 0
        same_all += len(set(sigs)) == 1
    return sigs_by_block, same12, same_all, ex


def fmt_sig(s):
    cob, single = s
    return "%d co-batched %s + %d single" % (len(cob), "[" + ",".join(map(str, cob)) + "]" if cob else "", single)


def cmd_run(a):
    out = {}
    for mode in a.modes.split(","):
        sb, s12, sall, ex = run_mode(mode, a, a.wall_scale)
        allsig = collections.Counter()
        for b in sb: allsig.update(sb[b])
        print("SIM %s: %d runs x %d blocks (probe in block 1: %s, occ %s hold %s%s): block1 == block2 in %d / %d runs (%.0f %%), all %d blocks equal in %d / %d (%.0f %%); distinct signatures %d" % (
            mode, a.runs, a.blocks, bool(a.probe_first), a.occ, a.hold, (", settle %s s" % a.settle) if mode.startswith("settle") else
            (", pattern %s, window %s" % (a.pattern or "1x8", a.window if a.window == "time" else "gens %d" % a.gens)) if mode == "fixed" else "", s12, a.runs, 100.0 * s12 / a.runs, a.blocks, sall, a.runs,
            100.0 * sall / a.runs, len(allsig)))
        for b in sorted(sb):
            top = sb[b].most_common(4)
            print("    block %d: %s" % (b + 1, " | ".join("%s x%d" % (fmt_sig(s), c) for s, c in top)))
        for e in ex[:6]:
            print("    example run %d block %d: newcomer groups by step %s ... -> %s, decode_share %s" % (e[0], e[1] + 1, e[2], fmt_sig(e[3]), e[4]))
        out[mode] = {"block1_eq_block2": s12, "all_equal": sall, "runs": a.runs, "distinct": len(allsig),
                     "per_block": {b + 1: [[fmt_sig(s), c] for s, c in sb[b].most_common(6)] for b in sb}}
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, indent=1)
    return 0


def cmd_armscan(a):
    res = {}
    for sc in [float(x) for x in a.scales.split(",")]:
        sb, s12, sall, ex = run_mode("fixed", a, sc)
        allsig = collections.Counter()
        for b in sb: allsig.update(sb[b])
        res[sc] = allsig
        print("ARMSCAN fixed pattern %s wall x%.2f: %s" % (a.pattern or "1x8", sc, " | ".join("%s x%d" % (fmt_sig(s), c) for s, c in allsig.most_common(4))))
    return 0


# ----------------------------------------------------------------- calibration from a real step log
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
ENT = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")


def read_steps(path, idle=3.0):
    out = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for l in f:
            if "n_seq=" not in l: continue
            m0 = TSRE.match(l)
            if not m0: continue
            t = calendar.timegm(time.strptime(m0.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m0.group(2)) if m0.group(2) else 0.0)
            m = STEP.search(l[m0.end():].rstrip("\n"))
            if not m: continue
            out.append({"t": t, "ph": m.group(2), "n_seq": int(m.group(3)), "n_tok": int(m.group(4)),
                        "ents": [(int(s), int(q), int(c)) for _, s, q, c in ENT.findall(m.group(5))]})
    for x, y in zip(out, out[1:]):
        d = y["t"] - x["t"]; x["wall"] = 1000.0 * d if 0 <= d <= idle else None
    return out


def cmd_calib(a):
    dec = collections.defaultdict(list); mix = []
    for lg, wins in zip(a.log, a.win):
        st = read_steps(lg)
        for w in wins.split(","):
            t0, t1 = (float(x) for x in w.split(":"))
            for s in st:
                if not (t0 <= s["t"] < t1) or s.get("wall") is None: continue
                if s["ph"] == "decode" and all(q == 1 for _, q, c in s["ents"]): dec[s["n_seq"]].append(s["wall"])
                elif s["ph"] != "decode":
                    mix.append((sum(q for _, q, c in s["ents"] if q >= 256), sum(1 for _, q, c in s["ents"] if q >= 256), s["wall"]))
    for nrow in sorted(dec):
        v = dec[nrow]
        if len(v) >= 20: print("CALIB decode %d rows: n %d mean %.1f ms sd %.1f" % (nrow, len(v), statistics.mean(v), statistics.pstdev(v)))
    if len(mix) >= 3:
        xs, ys = [m[0] for m in mix], [m[2] for m in mix]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs)); c = my - b * mx
        print("CALIB MIXED/prefill: n %d wall = %.1f + %.4f x prefill tokens ms, residual sd %.1f" % (
            len(mix), c, b, statistics.pstdev([y - (c + b * x) for x, y in zip(xs, ys)])))
        for nf in range(1, 6):
            v = [m for m in mix if m[1] == nf]
            if v: print("CALIB   %d fresh: n %d, prefill %.0f tok, wall %.0f ms" % (nf, len(v), statistics.mean(m[0] for m in v), statistics.mean(m[2] for m in v)))
    return 0


def cmd_selftest(a):
    ok, why = True, []
    for sd in (5, 6, 7):
        r = simulate("fixed", random.Random(sd), pattern="1x8", window="gens", probe=True, occ=6, hold=0)
        if r["sig"] != ((), 16): ok = False; why.append("fixed 1x8 (probe, occ 6 hold 0) gave %s" % (r["sig"],))
        r2 = simulate("fixed", random.Random(sd), pattern="1,2,2,3", window="gens")
        if r2["sig"] != ((3, 3, 2, 2, 2, 2), 2): ok = False; why.append("fixed 1,2,2,3 (no drafting) gave %s" % (r2["sig"],))
    sigs = {simulate("mb", random.Random(sd))["sig"] for sd in range(12)}
    if len(sigs) < 2: ok = False; why.append("mb gave one signature over 12 seeds (the race is not modelled)")
    if parse_pattern("3,3,2", 8) != [3, 3, 2] or parse_pattern("1x8", 8) != [1] * 8: ok = False; why.append("parse_pattern")
    print("SELFTEST " + ("OK (fixed 1x8 = 16 singles under probe + drafting, 1,2,2,3 = [3,3,2,2,2,2] + 2 without drafting, mb varies "
                         "with the race: %d signatures over 12 seeds)" % len(sigs) if ok else "FAILED: " + "; ".join(why)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("run", "armscan"):
        p = sub.add_parser(nm)
        p.add_argument("--modes", default="mb,settle,settle2,fixed")
        p.add_argument("--runs", type=int, default=200)
        p.add_argument("--blocks", type=int, default=4)
        p.add_argument("--probe-first", type=int, default=1)
        p.add_argument("--occ", type=int, default=0, help="low-occupancy drafting: rows gain 0-4 tokens while <= OCC run")
        p.add_argument("--hold", type=int, default=32, help="... after HOLD consecutive low-occupancy steps (0 = at once)")
        p.add_argument("--settle", type=float, default=60.0)
        p.add_argument("--dur", type=float, default=60.0)
        p.add_argument("--warm", type=float, default=15.0)
        p.add_argument("--pattern", default="1x8")
        p.add_argument("--spacing", type=int, default=None)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--seed0", type=int, default=1)
        p.add_argument("--wall-scale", type=float, default=1.0)
        p.add_argument("--scales", default="0.95,1.0,1.05")
        p.add_argument("--window", choices=("gens", "time"), default="gens", help="fixed mode's window (mbt2.py default gens)")
        p.add_argument("--gens", type=int, default=2)
        p.add_argument("--edge", type=float, default=0.5)
        p.add_argument("--json")
    p = sub.add_parser("calib")
    p.add_argument("--log", action="append", required=True)
    p.add_argument("--win", action="append", required=True, help="T0:T1[,T0:T1...] per --log")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"run": cmd_run, "armscan": cmd_armscan, "calib": cmd_calib, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
