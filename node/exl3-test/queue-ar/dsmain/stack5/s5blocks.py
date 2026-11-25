#!/usr/bin/env python3
"""s5blocks.py (12.11.2026, queue-ar dsmain/stack5, ds-main-port stack5 builder; the driver of 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5) =
s4blocks.py (md5 0a2cca92, copied; that file untouched) + three things, everything else s4blocks.py's:
  READ-BACK   after a block's POST (fixed + arm + epoch_bump) a 4-token nudge (/completion, temperature 0) applies it at a step, then
              GET /admin/fn4 is polled (<= --readback-s, default 30 s) until every switch of the block is applied and nothing is
              pending; a block not read back is flagged (ok false) and runs no phase.
  COUNTER GUARD  the block's baseline -- the full GET /stats + GET /admin/fn4 -- is taken AFTER the read-back, the end snapshot
              after its last phase (files --snap-dir/LABEL-bI-ARM-{0,1}-{st,fn4}.json); s5ctr.py (beside this file, imported)
              checks the arm's levers: --guard 'ARM=need,need/forbid,forbid' (s5ctr.py's lever names). A failed guard flags the
              block (ok false: no speed reading from it) and prints "GUARD i ARM FAIL ..."; else "GUARD i ARM OK ...".
  C8 TAPS     --c8-ncmb NCMB --c8-dir DIR: the c8 phase runs through nc/ncmb.py --mode phase (mb.phase_c itself at --mbtemp, as
              mbt.py runs it, + per-request taps, the full /stats at the window's edges, 2 s /stats samples) with its files in DIR
              (LABEL-bI-ARM-c8-{reqs.jsonl,st-w0.json,st-w1.json,samp.jsonl} + the MB line in LABEL-bI-ARM-c8.txt), so the body
              reads the c8 newcomer share and the newcomer step wall from the step log (nc/ncsteps.py c8, read only). The MB line is
              the c8 result, as before.

s4blocks.py (11.11.2026, queue-ar dsmain/stack4, ds-main-port stack3 builder; the driver of 57-p4zyzzzzz-ds-r00zy-stack4) =
occ2blocks.py (md5 cb31614d, copied; that file untouched) + --t1-prompts (the T 1 probe's prompts, default essay,code; stack4 runs
essay,code,edit) and the T 1 edit's metrics. Everything else is occ2blocks.py's.

occ2blocks.py (11.11.2026, queue-ar dsmain/occ2, ds-main-port stack3 builder; the driver of 57-p4zyzzzzz-ds-r02z-z0occ2) =
s3blocks.py (md5 e64f2ddd, copied; that file untouched) + lever Z0-O's readings:
  the 2 s sampler also reads /stats steps_decode and GET /admin/fn4's "draft" zero_cycles / occ_cycles / occ_moves, so each
  phase gets the DEPTH-0 SHARE of every 2 s interval (zero cycles / decode steps in it; intervals with no decode step left
  out): mean / sd / min / max / n in res["statsph"][phase]["d0_2s"], and the whole-phase share (zero cycles / decode steps)
  in res["statsph"][phase]["d0_share"]; the draft counters' line also prints occ_cycles / occ_moves; one "OCC" line a block.
Everything else is s3blocks.py's.

s3blocks.py (11.11.2026, queue-ar dsmain/stack3, ds-main-port stack3 builder; the driver of the stack3 item) = dlwblocks.py
(md5 d8915fe6, copied; that file untouched) + more phases and two counters:
  mbc1 / pf    mb.py's c1 (one closed-loop stream) and pf (4 streams + one fresh ~32k prefill) beside c4 / c8;
  --mbtemp T   every mb phase (mbc1, c4, c8, pf) at temperature T through mbt.py (mb.py loaded read only, its stream() given
               {"temperature": T}); unset = mb.py itself (the server-default sampling, temperature 1, as before);
  t1           sdprobe.py --mode sampled (the server-default sampling, temperature 1, --seed) --thinking on --effort medium,
               essay / code, 1 run: the 1-user probe at T 1 with thinking as real traffic sends it (lever SD's measure);
  per phase    also GET /admin/fn4's "sample" counters (lever SD: rows_sampled / rows_greedy / rounds_bound / ...) as deltas,
               res["sampleph"], and the lever BI stats are the body's (engine log).
The c1 / edit phases stay stepprobe / editprobe at temperature 0 (the stack2 numbers' probes). Everything else is dlwblocks.py's.

dlwblocks.py (11.11.2026, queue-ar dsmain/dlw, lever DLW follow-up) = z0blocks.py (md5 f7a22af5, copied; that file untouched)
+ --phases (a subset of c1,essaypos,edit,c4,c8; default all) + the lend integrated over every phase: the deltas of GET /admin/fn4's
lend counters (lent_cycles, lent_plain_steps / _ms, lent_probe_steps, transit_steps / _ms, lends, reclaims) and a 2-second
sampler of /stats experts (routed units in VRAM) and the lend's lent_units during the phase (mean / min / max, n), since the
values at a phase's end are a point reading (the iejzz-dlw summary read a lend in progress at one A start's c4 end as the arm's
lent units). Everything else is z0blocks.py's.

z0blocks.py (11.11.2026, queue-ar dsmain, ds-main-port Z0 agent; the driver of 57-p4zyzzzzz-ds-iej-z0ab) = ggblocksdz2.py
(md5 at copy: see the item; that file untouched) + per phase the experts and the lend: /stats snapshots around every phase
(routed units in VRAM at the end, rank-0 hit %, missed units and host-stream MB a step, decode steps and ms a decode step)
and GET /admin/fn4's lend object (lent units at the end, lends / reclaims / transit steps in the phase) into
res["statsph"][phase]; one "STATS" line a block. Everything else is ggblocksdz2's.

ggblocksdz2.py (10.11.2026, queue-ar dsmain, ds41-gg agent) -- ggblocksdz.py (queued items run it, left as it is) with
ARMS instead of one switch: each arm is a set of runtime switches (POST /admin/fn4), so an arm can move several at once
(the combined drafting policy: draft_hybrid 1 + draft_depth0 1). Every block the phases a drafter needs: stepprobe essay,code 512 (c1),
an essay-only stepprobe between two /stats snapshots (acceptance by draft position), editprobe (an edit re-emitting a
~2.5k-token region), mb c4 and mb c8 (--dur / --warm), the scheduler's draft counters (GET /admin/fn4 "draft") around every
phase.

  ggblocksdz2.py --url URL --mb-dir DIR --edit EDITPROBE --label L --arm 'A={"draft_hybrid": 0}' --arm 'B={...}'
                 --order ABBAC [--pair A,B] [--dur 75] [--warm 10] [--gen 512] [--fixed '{}'] [--out FILE.jsonl]

A block = POST {fixed..., arm..., "epoch_bump": 1}, the phases, then GET /admin/fn4: a block whose switches (fixed or
the arm's) are not as set (GET reports the applied value), with a change pending, whose epoch did not move or whose POST
was refused is flagged and left out. Per metric and arm: the good blocks' values, mean and t 95 % half-width over blocks;
for --pair X,Y (default the first two arms): consecutive (X, Y) / (Y, X) blocks are pairs, each pair's Y/X, their mean
and t 95 %, and the pooled mean(Y) / mean(X). One "BLOCK" line a block, one "GGB {json}" line. Standard library only."""
import argparse, json, math, os, statistics, subprocess, sys, threading, time, urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s5ctr  # noqa: E402  (beside this file)

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def jlines(out, prefix="J "):
    rows = []
    for l in out.splitlines():
        if l.startswith(prefix):
            try: rows.append(json.loads(l[len(prefix):]))
            except ValueError: pass
    return rows


def stepprobe(a, prompts):
    p = subprocess.run([sys.executable, "stepprobe.py", prompts, str(a.gen)], cwd=a.mb_dir, capture_output=True,
                       text=True, timeout=1500)
    return {r.get("prompt"): r for r in jlines(p.stdout)}


def essaypos(a):
    try: s0 = http(a.url + "/stats")
    except Exception: s0 = {}
    r = stepprobe(a, "essay").get("essay", {})
    try: s1 = http(a.url + "/stats")
    except Exception: s1 = {}
    pa, pb = s0.get("draft_positions") or [], s1.get("draft_positions") or []
    pos = []
    for k, q in enumerate(pb):
        r0 = pa[k] if k < len(pa) else {"reached": 0, "accepted": 0}
        re_, ac = q.get("reached", 0) - r0.get("reached", 0), q.get("accepted", 0) - r0.get("accepted", 0)
        pos.append({"p": k + 1, "reached": re_, "accepted": ac, "rate": (ac / re_) if re_ else None})
    return {"probe": r, "positions": pos}


def editprobe(a):
    p = subprocess.run([sys.executable, "-I", a.edit, "9000", "1500", "7"], cwd=a.mb_dir, capture_output=True, text=True,
                       timeout=1500)
    rows = jlines(p.stdout)
    return rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def mb(a, phase, tag):
    if phase == "c8" and a.c8_ncmb:
        base = os.path.join(a.c8_dir, tag)
        cmd = [sys.executable, "-I", "-B", a.c8_ncmb, "--mb", os.path.join(a.mb_dir, "mb.py"), "--mode", "phase", "--n", "8",
               "--tag", tag, "--dur", str(a.dur), "--warm", str(a.warm), "--reqs", base + "-reqs.jsonl", "--stats-prefix", base + "-st",
               "--samples", base + "-samp.jsonl"] + ([] if a.mbtemp is None else ["--temperature", str(a.mbtemp)])
        p = subprocess.run(cmd, cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 900)
        try:
            with open(base + ".txt", "w") as f: f.write(p.stdout + p.stderr)
        except OSError: pass
        rows = jlines(p.stdout, "MB ")
        r = rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}
        r["ncmb"] = base
        return r
    if a.mbtemp is None:
        cmd = [sys.executable, "mb.py", "--tag", tag, "--phases", phase, "--dur", str(a.dur), "--warm", str(a.warm)]
    else:
        cmd = [sys.executable, "-I", "-B", a.mbt, "--mb", "mb.py", "--temperature", str(a.mbtemp), "--tag", tag, "--phases", phase,
               "--dur", str(a.dur), "--warm", str(a.warm)]
    p = subprocess.run(cmd, cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 900)
    rows = jlines(p.stdout, "MB ")
    return rows[0] if rows else {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def t1probe(a):
    p = subprocess.run([sys.executable, "-I", "-B", a.sdprobe, "--mode", "sampled", "--seed", str(a.seed), "--runs", "1",
                        "--thinking", "on", "--effort", "medium", "--prompts", a.t1_prompts, "--tag", a.label],
                       cwd=a.mb_dir, capture_output=True, text=True, timeout=1500)
    rows = {r.get("prompt"): r for r in jlines(p.stdout)}
    return rows or {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def samplec(a):
    try: return http(a.url + "/admin/fn4").get("sample") or {}
    except Exception: return {}


def draftc(a):
    try: return http(a.url + "/admin/fn4").get("draft") or {}
    except Exception: return {}


def snapx(a):
    try: st = http(a.url + "/stats")
    except Exception: st = {}
    try: g = http(a.url + "/admin/fn4")
    except Exception: g = {}
    return st, g.get("lend") or {}


LENDK = ("lent_cycles", "lent_plain_steps", "lent_plain_ms", "lent_probe_steps", "lent_probe_ms", "transit_ms")


class Sampler:
    """/stats experts' VRAM units and the lend's lent_units every `every` seconds while a phase runs."""
    def __init__(self, a, every=2.0):
        self.a, self.every, self.v, self.l, self.z = a, every, [], [], []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(self.every):
            try:
                st = http(self.a.url + "/stats", timeout=10)
                t = {x.get("name"): x.get("units") for x in st.get("experts", {}).get("tiers", [])}
                if isinstance(t.get("vram"), (int, float)): self.v.append(t["vram"])
            except Exception: st = {}
            try:
                gg = http(self.a.url + "/admin/fn4", timeout=10)
                g = gg.get("lend") or {}
                if isinstance(g.get("lent_units"), (int, float)): self.l.append(g["lent_units"])
                d = gg.get("draft") or {}
                if isinstance(st.get("steps_decode"), (int, float)) and isinstance(d.get("zero_cycles"), (int, float)):
                    self.z.append((st["steps_decode"], d["zero_cycles"], d.get("occ_cycles") or 0, d.get("occ_moves") or 0))
            except Exception: pass

    def __enter__(self):
        self.t.start(); return self

    def __exit__(self, *e):
        self.stop.set(); self.t.join(timeout=15)

    def result(self):
        f = lambda v: {"mean": round(statistics.mean(v), 1), "min": min(v), "max": max(v), "n": len(v)} if v else None
        sh = []
        for (s0, z0, _, _), (s1, z1, _, _) in zip(self.z, self.z[1:]):
            if s1 > s0: sh.append(max(0.0, min(1.0, (z1 - z0) / (s1 - s0))))
        d0 = ({"mean": round(statistics.mean(sh), 3), "sd": round(statistics.stdev(sh), 3) if len(sh) > 1 else None,
               "min": round(min(sh), 3), "max": round(max(sh), 3), "n": len(sh)} if sh else None)
        return {"vram_units_sampled": f(self.v), "lent_units_sampled": f(self.l), "d0_2s": d0}


def xdelta(s0, s1):
    (a, l0), (b, l1) = s0, s1
    if not a or not b: return {"error": "no /stats"}
    ea, eb, la, lb = a.get("experts", {}), b.get("experts", {}), a.get("link", {}), b.get("link", {})
    ds = b.get("steps", 0) - a.get("steps", 0); dd = b.get("steps_decode", 0) - a.get("steps_decode", 0)
    dr = eb.get("routed", 0) - ea.get("routed", 0); drr = eb.get("routed_res", 0) - ea.get("routed_res", 0)
    n = max(1, ds)
    tiers = {x.get("name"): x.get("units") for x in eb.get("tiers", [])}
    return {"steps": ds, "decode_steps": dd,
            "ms_decode_step": round((b.get("step_ns_decode", 0) - a.get("step_ns_decode", 0)) / max(1, dd) / 1e6, 2) if dd > 0 else None,
            "hit_pct_rank0": round(100.0 * drr / dr, 2) if dr else None, "missed_step_rank0": round((dr - drr) / n, 2),
            "stream_MB_step": round((lb.get("stream_bytes", 0) - la.get("stream_bytes", 0)) / n / 1e6, 1),
            "vram_units_end": tiers.get("vram"),
            "lent_units_end": l1.get("lent_units"), "lend_state_end": l1.get("state"),
            "lends": (l1.get("lends") or 0) - (l0.get("lends") or 0), "reclaims": (l1.get("reclaims") or 0) - (l0.get("reclaims") or 0),
            "transit_steps": (l1.get("transit_steps") or 0) - (l0.get("transit_steps") or 0),
            "hyst_holds": (l1.get("hyst_holds") or 0) - (l0.get("hyst_holds") or 0) if "hyst_holds" in l1 else None,
            **{k: (l1.get(k) or 0) - (l0.get(k) or 0) for k in LENDK if k in l1}}


SPH = {}
SMP = {}


def nudge(a):
    try: http(a.url + "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0}, timeout=120)
    except Exception: pass


def readback(a, want):
    """poll GET /admin/fn4 until every switch of `want` is applied and nothing is pending -> (ok, the last GET)"""
    t_end, st = time.time() + a.readback_s, {}
    while True:
        try: st = http(a.url + "/admin/fn4")
        except Exception: st = {}
        sw = st.get("switches", {})
        if st and all(sw.get(k) == v for k, v in want.items()) and not st.get("pending"): return True, st
        if time.time() > t_end: return False, st
        nudge(a); time.sleep(1.0)


def fullsnap(a, path):
    out = {}
    for nm, ep in (("st", "/stats"), ("fn4", "/admin/fn4")):
        try: out[nm] = http(a.url + ep, timeout=30)
        except Exception: out[nm] = {}
        if a.snap_dir:
            try:
                with open("%s-%s.json" % (path, nm), "w") as f: json.dump(out[nm], f)
            except OSError: pass
    return out


def gspec(a, arm):
    """--guard ARM=need,need/forbid,forbid -> (need, forbid); an arm not named: no guard ([], [])"""
    for x in a.guard:
        k, v = x.split("=", 1)
        if k == arm:
            nd, _, fb = v.partition("/")
            return s5ctr.lst(nd), s5ctr.lst(fb)
    return [], []


def phase(a, dph, name, f):
    """f() between two reads of the draft counters; their delta into dph[name]; /stats + lend deltas into SPH[name]."""
    x0 = snapx(a)
    d0, q0 = draftc(a), samplec(a)
    with Sampler(a) as smp:
        r = f()
    d1, q1 = draftc(a), samplec(a)
    x1 = snapx(a)
    dph[name] = {k: (d1.get(k) or 0) - (d0.get(k) or 0) for k in d1}
    SMP[name] = {k: (q1.get(k) or 0) - (q0.get(k) or 0) for k in q1 if k != "capable"}
    SPH[name] = dict(xdelta(x0, x1), **smp.result())
    return r


def dfmt(d):
    if not d: return "-"
    return "zero %s probe %s moves %s hyb %s/%s/%s occ %s/%s" % (d.get("zero_cycles"), d.get("probe_cycles"), d.get("zero_moves"),
                                                    d.get("hybrid_lookup"), d.get("hybrid_head"), d.get("hybrid_unahead"),
                                                    d.get("occ_cycles"), d.get("occ_moves"))


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


# (label, getter over a block's res)
def g(path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f


METRICS = []
for ph in ("c4", "c8"):
    METRICS.append(("%s d0 share" % ph, g(("statsph", ph, "d0_share"))))
    METRICS.append(("%s d0 share 2s sd" % ph, g(("statsph", ph, "d0_2s", "sd"))))
for nm, key in (("essay", ("c1", "essay")), ("code", ("c1", "code")), ("edit", ("edit",))):
    for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc"), ("ms_per_decode_step", "ms/step")):
        METRICS.append(("c1 %s %s" % (nm, ml), g(key + (mk,))))
for nm in ("essay", "code", "edit"):
    for mk, ml in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc")):
        METRICS.append(("t1 %s %s" % (nm, ml), g(("t1", nm, mk))))
for ph in ("mbc1", "c4", "c8"):
    for mk, ml in (("agg_dec_tps", "aggregate tok/s"), ("per_stream_dec_p50", "per-stream p50"), ("tok_per_decode_step", "tok/step"),
                   ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
        METRICS.append(("%s %s" % (ph, ml), g((ph, mk))))
for mk, ml in (("prefill_tps", "prefill tok/s"), ("prefill_ttft", "prefill TTFT s"), ("decode_kept", "decode kept"),
               ("ms_step_during", "ms/step during")):
    METRICS.append(("pf %s" % ml, g(("pf", mk))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--pair")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--out")
    ap.add_argument("--phases", default="c1,edit,mbc1,c4,c8,pf")
    ap.add_argument("--mbtemp", type=float, default=None)
    ap.add_argument("--mbt", default=None)
    ap.add_argument("--sdprobe", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--t1-prompts", default="essay,code")
    ap.add_argument("--guard", action="append", default=[])
    ap.add_argument("--hb", type=int, default=0)
    ap.add_argument("--hb-ranks", default="1,2")
    ap.add_argument("--readback-s", type=float, default=30.0)
    ap.add_argument("--snap-dir", default="")
    ap.add_argument("--c8-ncmb", default="")
    ap.add_argument("--c8-dir", default="")
    a = ap.parse_args()
    if a.c8_ncmb and not a.c8_dir:
        print("GGB " + json.dumps({"error": "--c8-ncmb needs --c8-dir"}))
        return 1
    hbr = tuple(int(x) for x in a.hb_ranks.split(",") if x)
    for x in a.guard: gspec(a, x.split("=", 1)[0])   # unknown lever names fail here, before any block
    phs = [p for p in a.phases.split(",") if p]
    bad = [p for p in phs if p not in ("c1", "essaypos", "edit", "t1", "mbc1", "c4", "c8", "pf")]
    if bad or (a.mbtemp is not None and not a.mbt) or ("t1" in phs and not a.sdprobe):
        print("GGB " + json.dumps({"error": "phases %s unknown, or --mbtemp without --mbt, or t1 without --sdprobe" % bad}))
        return 1
    fixed = json.loads(a.fixed)
    arms, names = {}, []
    for x in a.arm:
        k, v = x.split("=", 1)
        arms[k] = json.loads(v)
        names.append(k)
    if any(c not in arms for c in a.order):
        print("GGB " + json.dumps({"error": "order %s names an arm not given (%s)" % (a.order, names)}))
        return 1
    px, py = (a.pair.split(",") if a.pair else (names + names)[:2])
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        want = dict(fixed, **arms[arm])
        s0 = http(a.url + "/admin/fn4")
        e0 = s0.get("epoch")
        try:
            http(a.url + "/admin/fn4", dict(want, epoch_bump=1))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        nudge(a)
        rb, rbst = readback(a, want) if posted is True else (False, {})
        sp = os.path.join(a.snap_dir, "%s-b%d%s" % (a.label, i, arm)) if a.snap_dir else ""
        snap0 = fullsnap(a, sp + "-0")
        dph = {}
        if not rb:   # not read back: no phase runs (the counters would mix two arms)
            phs_b = []
        else:
            phs_b = phs
        res = {"c1": phase(a, dph, "c1", lambda: stepprobe(a, "essay,code")) if "c1" in phs_b else {},
               "essaypos": phase(a, dph, "essaypos", lambda: essaypos(a)) if "essaypos" in phs_b else {"probe": {}, "positions": []},
               "edit": phase(a, dph, "edit", lambda: editprobe(a)) if "edit" in phs_b else {},
               "t1": phase(a, dph, "t1", lambda: t1probe(a)) if "t1" in phs_b else {}}
        for ph in ("mbc1", "c4", "c8", "pf"):
            mp = "c1" if ph == "mbc1" else ph
            res[ph] = phase(a, dph, ph, lambda ph=ph, mp=mp: mb(a, mp, f"{a.label}-b{i}{arm}-{ph}")) if ph in phs_b else {"skipped": 1}
        res["draftph"] = dph
        res["statsph"] = dict(SPH); SPH.clear()
        res["sampleph"] = dict(SMP); SMP.clear()
        for ph, spd in res["statsph"].items():
            z = (res["draftph"].get(ph) or {}).get("zero_cycles") or 0; dd = spd.get("decode_steps") or 0
            spd["d0_share"] = round(z / dd, 3) if dd else None
            spd["occ_cycles"] = (res["draftph"].get(ph) or {}).get("occ_cycles")
            spd["occ_moves"] = (res["draftph"].get(ph) or {}).get("occ_moves")
        snap1 = fullsnap(a, sp + "-1")
        nd, fb = gspec(a, arm)
        gr, gd = s5ctr.check(snap0["st"], snap1["st"], snap0["fn4"], snap1["fn4"], nd, fb, a.hb, hbr)
        if not rb: gr = "FAIL the switches were not read back within %.0f s (%s)" % (a.readback_s, json.dumps({k: (rbst.get("switches") or {}).get(k) for k, v in want.items() if (rbst.get("switches") or {}).get(k) != v})[:300])
        res["ctr"] = {"guard": gr, "need": nd, "forbid": fb, "delta": gd, "line": s5ctr.line(gd), "readback": rb}
        st = http(a.url + "/admin/fn4")
        sw = st.get("switches", {})
        ok = posted is True and rb and gr == "OK" and all(sw.get(k) == v for k, v in want.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) \
            and not any(isinstance(res.get(p), dict) and ("error" in res[p] or (res[p].get("errors") or 0) > 0) for p in ("mbc1", "c4", "c8", "pf", "t1"))
        row = {"label": a.label, "block": i, "arm": arm, "values": arms[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok, "res": res,
               "t": round(time.time() - t_start, 1)}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        c1 = res["c1"]; ed = res.get("edit") or {}; t1 = res.get("t1") or {}
        pos = " ".join("p%d %s" % (p["p"], "-" if p["rate"] is None else "%.2f" % p["rate"]) for p in res["essaypos"]["positions"])
        cx = lambda ph: "%s %s tok/s p50 %s %s tok/step acc %s %s ms" % (ph, res[ph].get("agg_dec_tps"), res[ph].get("per_stream_dec_p50"),
                                                                    res[ph].get("tok_per_decode_step"), res[ph].get("acceptance"), res[ph].get("ms_decode_step"))
        print("BLOCK %d %s %s ok=%s | mb temperature %s | draft %s | essay %s tok/s %s tok/step acc %s | code %s tok/s %s tok/step acc %s | essay by position %s | edit %s tok/s %s tok/step acc %s echo %s | t1 essay %s tok/s %s tok/step acc %s think %s ch, code %s tok/s %s tok/step acc %s | %s | %s | %s | pf %s tok/s TTFT %s s decode kept %s %s ms/step during | sample %s" % (
            i, arm, json.dumps(arms[arm]), ok, "server default" if a.mbtemp is None else a.mbtemp, " / ".join("%s %s" % (k, dfmt(v)) for k, v in dph.items()) or "-",
            c1.get("essay", {}).get("decode_tps"), c1.get("essay", {}).get("tokens_per_step"), c1.get("essay", {}).get("acceptance"),
            c1.get("code", {}).get("decode_tps"), c1.get("code", {}).get("tokens_per_step"), c1.get("code", {}).get("acceptance"),
            pos or "-", ed.get("decode_tps"), ed.get("tokens_per_step"), ed.get("acceptance"), ed.get("echo_ratio"),
            (t1.get("essay") or {}).get("decode_tps"), (t1.get("essay") or {}).get("tokens_per_step"), (t1.get("essay") or {}).get("acceptance"),
            (t1.get("essay") or {}).get("reasoning_chars"), (t1.get("code") or {}).get("decode_tps"), (t1.get("code") or {}).get("tokens_per_step"),
            (t1.get("code") or {}).get("acceptance"), cx("mbc1"), cx("c4"), cx("c8"),
            res["pf"].get("prefill_tps"), res["pf"].get("prefill_ttft"), res["pf"].get("decode_kept"), res["pf"].get("ms_step_during"),
            " / ".join("%s %s" % (k, " ".join("%s %+d" % (x, y) for x, y in v.items() if y)) for k, v in res["sampleph"].items() if any(v.values())) or "-"), flush=True)
        if t1 and "error" not in t1:
            print("T1 %d %s | %s" % (i, arm, " | ".join("%s %s tok/s %s tok/step acc %s think %s ch %s tok" % (k, v.get("decode_tps"), v.get("tokens_per_step"),
                  v.get("acceptance"), v.get("reasoning_chars"), v.get("tokens")) for k, v in t1.items() if isinstance(v, dict))), flush=True)
        print("GUARD %d %s %s | need [%s] forbid [%s] | %s" % (i, arm, gr, ",".join(nd), ",".join(fb), s5ctr.line(gd)), flush=True)
        print("OCC %d %s | %s" % (i, arm, " | ".join("%s d0 share %s (2 s: %s) occ_cycles %s occ_moves %s zero_moves %s" % (
            k, v.get("d0_share"), v.get("d0_2s"), v.get("occ_cycles"), v.get("occ_moves"), (res["draftph"].get(k) or {}).get("zero_moves"))
            for k, v in res["statsph"].items())), flush=True)
        print("STATS %d %s | %s" % (i, arm, " | ".join("%s vram %s (sampled %s) hit %s %% missed %s stream %s MB/st lent %s (sampled %s) lends %s reclaims %s lent_cycles %s lent_plain_steps %s of %s decode steps transit %s" % (
            k, v.get("vram_units_end"), v.get("vram_units_sampled"), v.get("hit_pct_rank0"), v.get("missed_step_rank0"), v.get("stream_MB_step"),
            v.get("lent_units_end"), v.get("lent_units_sampled"), v.get("lends"), v.get("reclaims"), v.get("lent_cycles"), v.get("lent_plain_steps"),
            v.get("decode_steps"), v.get("transit_steps")) for k, v in res["statsph"].items())), flush=True)
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if {rows[j]["arm"], rows[j + 1]["arm"]} == {px, py}]
    out = {"label": a.label, "arms": arms, "fixed": fixed, "order": a.order, "pair": [px, py], "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1), "metrics": {}}
    for nm, get in METRICS:
        per = {}
        for k in names:
            v = [get(r["res"]) for r in rows if r["ok"] and r["arm"] == k and get(r["res"]) is not None]
            m, h = ci(v)
            per[k] = {"values": v, "mean": m, "t95": h}
        pr = []
        for x, y in pairs:
            if not (x["ok"] and y["ok"]): continue
            rx, ry = (x, y) if x["arm"] == px else (y, x)
            gx, gy = get(rx["res"]), get(ry["res"])
            if gx and gy is not None: pr.append(gy / gx)
        m, h = ci(pr)
        mx, my = per.get(px, {}).get("mean"), per.get(py, {}).get("mean")
        out["metrics"][nm] = {"arms": per, "pairs": pr, "pair_mean": m, "ci95": h, "pooled": (my / mx) if mx and my is not None else None}
    print("GGB " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
