#!/usr/bin/env python3
"""mbt3.py (12.11.2026, queue-ar/dsmain/harness, the int11 builder; = mbt2.py 8b8e1094 + a FIXED PROMPT SET reset per block, TESTING.md
12.11. "c4 METHOD FIX"; mbt2.py is not changed) -- mb.py's c<N> at a fixed temperature with a FIXED, SEEDED ARRIVAL SCHEDULE, so the
newcomer co-batch mix is the same in every block and arm (c8 block-mix note 12.11.), AND THE SAME PROMPTS.

WHY mbt3: mbt2's worker i sends request k as "# req <time_ns> <i>" + mb.text(.., i * 11 + k) + TASKS[(i + k) % ..]: the
schedule and the (i, k) sequence restart with every block (a fresh process), but the time_ns nonce on the first line makes
every prompt -- and so every greedy answer and its draft acceptance -- different from block to block (c4lev 12.11.: E2 "+5.8 %"
was all acceptance, O 0.666 vs E2 0.744 at the same ~63.5 ms a step). mbt3 --prompt-set fixed (DEFAULT) writes the first line
"# req fixed-s<seed>-k<k> <i>" instead (the worker index stays field 4, as ncmb2's --sched mb reads it): request (i, k) is the
same text in every block, so at T 0 the same answer; --reset-per-block 1 (DEFAULT) POSTs /reset_prefix_cache when the phase
starts, so no block reuses another's cached prompts (mbt2's nonce did that by never repeating a prompt). The primer is
"# primer fixed-s<seed>" likewise. --prompt-set nonce = mbt2's prompts exactly.

WHY: mb.py's closed loop starts all N streams at once on an idle server. Which of them share the first prefill steps is a
race (12.11. logs: the first step took 1, 2 or 3 of the 8), every request then runs exactly 512 tokens = 513 steps, so the
groups formed in that burst come back together generation after generation (SP 2X 233233233, 3X 134134134); the start-of-
server drafting probe of the first block after a start gives each stream 8-26 extra tokens and desynchronises them into
singles (SP 1B, KV4 every c8-1). Co-batched newcomer steps move decode_share +3-4 % and TTFT p50 +1-2 s, so the block's mix,
not the arm, sets ~2 % of c8. A settling window cannot fix this (no attractor: the burst's groups persist; c8sim.py).

--sched fixed (DEFAULT):
  * streams in groups (--pattern, default 1xN = every newcomer alone); which worker is in which group = a --seed permutation;
  * release on first-token events: the lead group at t 0 (a primer request of max_tokens 1 first when the lead group has
    more than one member), group 1 --release-delay s later, group j >= 2 --release-delay s after group j-2's first token
    (= the start of group j-1's prefill step) -> every group arrives DURING the previous group's prefill step and is admitted
    whole in the next one (groups <= 3 prompts of mb's ~0.9-1.4k tokens fit MNBT 4160);
  * max_tokens: group j's FIRST request gen - j x spacing (spacing = gen // groups), every later one gen (+- --len-jitter,
    seeded; default 0), ignore_eos (default on; mb's prompts reach 512 at T 0 anyway: 296 / 296 requests on 12.11.) ->
    in the step domain the groups' newcomer steps sit spacing - 1 steps apart for good;
  * window --window gens (default): from --edge s after the first token of the first newcomer step (a request after a
    stream's first) that ends after --warm s, to --edge s after the --gens x groups-th newcomer step after it: exactly
    --gens x groups newcomer steps inside, both edges in decode-only time (a time window cuts the newcomer row at an
    arbitrary point: 15 vs 16 single steps = ~1.3 % of agg tok/s); with 1x8, dur ~46 s, ending ~60 s after the start;
    --window time = mb's [--warm, --dur].
--sched mb: mb.phase_c itself (the old loop; for comparison), with the client-side mix added.
Phases pf / turns run mb.py's own functions (at --temperature).

Output: mb.py's "MB {json}" line a phase + temperature, sched, pattern, seed, window, first_max, prompt_set, reset (the HTTP code
of the block's prefix-cache reset), release (per group: the
release time and its first token, s from the block start), mix_client (newcomer steps in the window by first-token time:
co-batched count / sizes, single count; a co-batched step's requests get their first token at the same instant), short
(requests that ended before max_tokens), window_timeout. --reqs writes ncmb.py-style rows (send / done / mark) for c8mix.py.

  mbt3.py --mb ~/gpu/ds41/mb.py --temperature 0 --tag T --phases c8 [--sched fixed|mb] [--pattern 1x8 | 1,2,2,3]
          [--prompt-set fixed|nonce] [--reset-per-block 1|0]
          [--seed 1] [--window gens|time] [--gens 2] [--edge 0.5] [--dur 60 --warm 15] [--gen 512] [--ctx 1000]
          [--ignore-eos 1] [--release-delay 0.1] [--spacing N] [--len-jitter 0] [--reqs R.jsonl] [--url http://127.0.0.1:8090]
  mbt3.py --mb ~/gpu/ds41/mb.py --selftest      offline: mb.py loads READ ONLY, has what this wrapper calls; plan / pattern; two
          blocks' prompt sequences under --prompt-set fixed identical (ids, order, n_predict, text), under nonce not.
mb.py is loaded READ ONLY (importlib, no bytecode); nothing of it changes on disk. Standard library only.
"""
import argparse, importlib.util, inspect, json, random, re, statistics, sys, threading, time, urllib.request

sys.dont_write_bytecode = True
NEED = ("stream", "phase_c", "phase_pf", "phase_turns", "closed_loop", "metrics", "mdelta", "text", "TASKS", "URL")


def load(path, name="mb"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_pattern(p, n):
    p = (p or "").strip().lower()
    if not p or p in ("solo", "1xn"): return [1] * n
    m = re.fullmatch(r"(\d+)x(\d+)", p)
    s = [int(m.group(1))] * int(m.group(2)) if m else [int(x) for x in p.split(",") if x.strip()]
    if sum(s) != n or any(x < 1 or x > 3 for x in s):
        raise ValueError("pattern %r: group sizes 1..3 summing to %d" % (p, n))
    return s


def make_plan(n, pattern, seed, gen, spacing=None):
    """-> [{"j", "workers", "first_max"}]: the seeded worker permutation cut into the pattern's groups."""
    sizes = parse_pattern(pattern, n)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    m = len(sizes)
    sp = spacing or gen // m
    if gen - (m - 1) * sp < 16:
        raise ValueError("spacing %d x %d groups leaves the last group's first request < 16 tokens of %d" % (sp, m, gen))
    out, p = [], 0
    for j, s in enumerate(sizes):
        out.append({"j": j, "workers": order[p:p + s], "first_max": gen - j * sp})
        p += s
    return out, sp


def gen_of(a, i, k):
    if not a.len_jitter: return a.gen
    return a.gen + random.Random((a.seed * 1000003 + i * 7919 + k) & 0xFFFFFFFF).randint(-a.len_jitter, a.len_jitter)


class Stamps(list):
    """mb.stream's stamps list: calls back once, at the first token (mb.stream appends one time.time() per text chunk)."""
    def __init__(self, cb=None):
        super().__init__()
        self.cb, self.first = cb, None

    def append(self, x):
        if self.first is None:
            self.first = x
            if self.cb:
                try: self.cb(x)
                except Exception: pass
        super().append(x)


def cobatch(firsts, tol=0.005):
    """first-token times -> step groups (a co-batched step's requests get their first token at the same instant)."""
    fs = sorted(firsts)
    groups, cur = [], []
    for t in fs:
        if cur and t - cur[-1] > tol:
            groups.append(cur); cur = []
        cur.append(t)
    if cur: groups.append(cur)
    sizes = [len(g) for g in groups]
    return {"newcomer_steps": len(sizes), "cobatched": sum(1 for s in sizes if s > 1),
            "cob_sizes": sorted([s for s in sizes if s > 1], reverse=True), "single": sum(1 for s in sizes if s == 1),
            "newcomers": sum(sizes)}


class Recorder:
    def __init__(self):
        self.lock, self.rows, self.recs = threading.Lock(), [], []

    def add(self, d):
        with self.lock: self.rows.append(d)

    def dump(self, path, tag):
        if not path: return
        with self.lock, open(path, "a") as f:
            for d in self.rows: f.write(json.dumps(dict(d, tag=tag)) + "\n")


def prompt_of(mb, a, i, k):
    """request k of worker i: --prompt-set fixed = the same text in every block (no time nonce); nonce = mbt2.py's"""
    if getattr(a, "prompt_set", "fixed") == "nonce":
        head = f"# req {time.time_ns()} {i}\n"
    else:
        head = f"# req fixed-s{a.seed}-k{k} {i}\n"
    return head + mb.text(int(a.ctx * 3.9), i * 11 + k) + "\n\n" + mb.TASKS[(i + k) % len(mb.TASKS)]


def primer_of(mb, a):
    head = f"# primer {time.time_ns()}\n" if getattr(a, "prompt_set", "fixed") == "nonce" else f"# primer fixed-s{a.seed}\n"
    return head + mb.text(int(a.ctx * 3.9), 997) + "\n\nSay OK."


def prompt_seq(mb, a, plan, kmax):
    """the block's request sequence as the workers would send it: [(i, k, group, max_tokens, sha1(prompt)[:16])], k < kmax"""
    import hashlib
    out = []
    for g in plan:
        for i in g["workers"]:
            for k in range(kmax):
                mt = g["first_max"] if k == 0 else gen_of(a, i, k)
                out.append((i, k, g["j"], mt, hashlib.sha1(prompt_of(mb, a, i, k).encode()).hexdigest()[:16]))
    return out


def reset_cache(mb):
    """POST /reset_prefix_cache (the block's start); the HTTP code, or the error"""
    try:
        req = urllib.request.Request(mb.URL + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except Exception as e:
        return "error %s" % str(e)[:80]


def phase_fixed(mb, n, a, rec, drain=True):
    """the fixed schedule's closed loop + window; drain=False returns right after the window (ncmb2.py --mode win: hard exit)."""
    plan, sp = make_plan(n, a.pattern if a.pattern else "1x%d" % n, a.seed, a.gen, a.spacing)
    m = len(plan)
    primer = len(plan[0]["workers"]) > 1
    stop = threading.Event()
    first_ev = {j: threading.Event() for j in range(-1, m)}
    first_t = {}
    rel_ev = {j: threading.Event() for j in range(m)}
    rel_t = {}
    cond = threading.Condition()
    nc = []                       # (t_first, j, k) a group's newcomer step from its second request on (k >= 1), once per (j, k)
    seen = set()
    recs = []
    ft = []                       # every request's first-token time, at the callback (win mode exits before requests end)
    rcode = reset_cache(mb) if getattr(a, "reset_per_block", 1) else None
    rec.add({"ev": "reset", "code": rcode, "prompt_set": getattr(a, "prompt_set", "fixed")})
    t_start = time.time()

    def on_first(j, k):
        def cb(t):
            ft.append(t)
            if k == 0 and not first_ev[j].is_set():
                first_t[j] = t; first_ev[j].set()
            if k >= 1:
                with cond:
                    if (j, k) not in seen:
                        seen.add((j, k)); nc.append((t, j, k)); cond.notify_all()
        return cb

    def worker(i, g):
        rel_ev[g["j"]].wait()
        k = 0
        while not stop.is_set():
            q = prompt_of(mb, a, i, k)
            mt = g["first_max"] if k == 0 else gen_of(a, i, k)
            st = Stamps(on_first(g["j"], k))
            ts = time.time()
            rec.add({"ev": "send", "t": ts, "w": i, "k": k, "j": g["j"], "max_tokens": mt})
            try:
                r = mb.stream([{"role": "user", "content": q}], mt, stamps=st,
                              extra=({"ignore_eos": True} if a.ignore_eos else None))
                r["w"], r["k"], r["j"], r["max_tokens"], r["t_send"], r["t_first"], r["t_end"] = i, k, g["j"], mt, ts, st.first, time.time()
                r.pop("text", None)
                recs.append(r)
                rec.add({"ev": "done", "t_send": ts, "w": i, "k": k, "j": g["j"], "t0": r.get("t0"), "ttft": r.get("ttft"),
                         "t_first": st.first, "t_end": r["t_end"], "prompt": r.get("prompt"), "cached": r.get("cached"),
                         "n": r.get("n"), "max_tokens": mt, "err": None})
            except Exception as e:
                recs.append(dict(err=str(e)[:200], t0=ts, w=i, k=k))
                rec.add({"ev": "done", "t_send": ts, "w": i, "k": k, "j": g["j"], "err": str(e)[:200], "t_end": time.time()})
                if stop.wait(1.0): break
            k += 1

    th = []
    for g in plan:
        for i in g["workers"]:
            t = threading.Thread(target=worker, args=(i, g), daemon=True); t.start(); th.append(t)
    timeouts = []
    # releases
    if primer:
        def prim():
            q = primer_of(mb, a)
            st = Stamps(lambda t: (first_t.__setitem__(-1, t), first_ev[-1].set()))
            try: mb.stream([{"role": "user", "content": q}], 1, effort="none", stamps=st)
            except Exception as e: timeouts.append("primer: %s" % str(e)[:100])
            first_ev[-1].set()
        tp = threading.Thread(target=prim, daemon=True); tp.start()
        time.sleep(a.release_delay)
    rel_t[0] = time.time() - t_start; rel_ev[0].set()
    for j in range(1, m):
        if j == 1 and not primer:
            time.sleep(max(0.0, a.release_delay - (time.time() - t_start - rel_t[0])))
        else:
            ref = j - 2
            if not first_ev[ref].wait(a.max_wait):
                timeouts.append("no first token of group %d" % ref)
            time.sleep(a.release_delay)
        rel_t[j] = time.time() - t_start; rel_ev[j].set()
    # window
    wto = []
    if a.window == "gens":
        need = 1 + a.gens * m
        tw = t_start + a.warm
        late = lambda: sorted(x[0] for x in nc if x[0] >= tw)          # noqa: E731  newcomer steps after the warm-up
        with cond:
            if not cond.wait_for(lambda: len(late()) >= 1, timeout=a.warm + a.max_wait): wto.append("no newcomer after the warm-up")
            ta = late()[0] if late() else time.time()
        time.sleep(max(0.0, ta + a.edge - time.time()))
        b = mb.metrics(); tw0 = time.time(); rec.add({"ev": "mark", "k": 0, "t": tw0})
        with cond:
            if not cond.wait_for(lambda: len(late()) >= need, timeout=a.max_wait + a.gens * 60):
                wto.append("only %d of %d newcomer steps" % (len(late()), need))
            tb = late()[need - 1] if len(late()) >= need else time.time()
        time.sleep(max(0.0, tb + a.edge - time.time()))
        e = mb.metrics(); tw1 = time.time(); rec.add({"ev": "mark", "k": 1, "t": tw1})
    else:
        time.sleep(max(0.0, t_start + a.warm - time.time()))
        b = mb.metrics(); tw0 = time.time(); rec.add({"ev": "mark", "k": 0, "t": tw0})
        time.sleep(max(0.0, t_start + a.dur - time.time()))
        e = mb.metrics(); tw1 = time.time(); rec.add({"ev": "mark", "k": 1, "t": tw1})
    stop.set()
    if drain:
        for t in th:
            t.join(timeout=600)
    res = summarize(mb, n, b, e, tw0, tw1, list(recs), list(ft))
    res.update(sched="fixed", pattern=",".join(str(len(g["workers"])) for g in plan), seed=a.seed, spacing=sp,
               window=a.window, gens=a.gens if a.window == "gens" else None, edge=a.edge if a.window == "gens" else None,
               groups=[g["workers"] for g in plan], first_max=[g["first_max"] for g in plan], primer=primer,
               ignore_eos=bool(a.ignore_eos), len_jitter=a.len_jitter, prompt_set=getattr(a, "prompt_set", "fixed"), reset=rcode,
               release=[[round(rel_t.get(j, -1), 3), round(first_t[j] - t_start, 3) if j in first_t else None] for j in range(m)],
               window_timeout="; ".join(timeouts + wto) or None)
    return res


def summarize(mb, n, b, e, tw0, tw1, recs, ft=None):
    win = [r for r in recs if not r.get("err") and tw0 <= r["t0"] < tw1]
    dec = [r["dec"] for r in win if r.get("dec")]
    tt = [r["ttft"] for r in win if r.get("ttft")]
    d = mb.mdelta(b, e)
    firsts = [t for t in (ft if ft is not None else [r.get("t_first") for r in recs if not r.get("err")]) if t and tw0 <= t < tw1]
    short = sum(1 for r in recs if not r.get("err") and r.get("max_tokens") and (r.get("n") or 0) < r["max_tokens"])
    return dict(phase="c%d" % n, window_s=round(tw1 - tw0, 1), agg_dec_tps=round(d["dec_tok"] / (tw1 - tw0), 1),
                agg_gen_tps_q=round(d["gen_tok"] / (tw1 - tw0), 1), per_stream_dec_p50=round(statistics.median(dec), 1) if dec else None,
                ttft_p50=round(statistics.median(tt), 2) if tt else None, requests=len(win), errors=sum(1 for r in recs if r.get("err")),
                mix_client=cobatch(firsts), short=short, **d)


def phase_mb(mb, n, a, rec):
    """mb.phase_c itself (the old loop); mb.metrics is wrapped for the two window marks, the stream wrapper in main() records
    each request's first token -> the client-side mix of the same window."""
    orig = mb.metrics
    marks = []

    def metrics():
        x = orig()
        t = time.time(); marks.append(t); rec.add({"ev": "mark", "k": len(marks) - 1, "t": t})
        return x

    mb.metrics = metrics
    try:
        res = mb.phase_c(n, a)
    finally:
        mb.metrics = orig
    with rec.lock:
        rows = [d for d in rec.rows if d.get("ev") == "done" and not d.get("err")]
    if len(marks) >= 2:
        firsts = [d["t_first"] for d in rows if d.get("t_first") and marks[0] <= d["t_first"] < marks[1]]
        res["mix_client"] = cobatch(firsts)
    res["short"] = sum(1 for d in rows if (d.get("n") or 0) < (d.get("max_tokens") or 0))
    res.update(sched="mb", window="time")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True)
    ap.add_argument("--url", default=None, help="server URL (default mb.py's http://127.0.0.1:8090)")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--phases", default="c8")
    ap.add_argument("--sched", choices=("fixed", "mb"), default="fixed")
    ap.add_argument("--pattern", default="", help="group sizes 1..3, e.g. 1x8 (default 1xN) or 1,2,2,3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spacing", type=int, default=None, help="tokens between groups' first requests (default gen // groups)")
    ap.add_argument("--len-jitter", type=int, default=0, help="later requests' max_tokens gen +- J (seeded); 0 = all gen (A/B)")
    ap.add_argument("--ignore-eos", type=int, default=1)
    ap.add_argument("--release-delay", type=float, default=0.1)
    ap.add_argument("--window", choices=("gens", "time"), default="gens")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--edge", type=float, default=0.5)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--warm", type=float, default=15)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--reqs", default="")
    ap.add_argument("--prompt-set", choices=("fixed", "nonce"), default="fixed")
    ap.add_argument("--reset-per-block", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb)
    miss = [x for x in NEED if not hasattr(mb, x)]
    if a.selftest:
        ok = not miss and "extra" in inspect.signature(mb.stream).parameters and "stamps" in inspect.signature(mb.stream).parameters
        why = []
        try:
            p1, sp = make_plan(8, "1x8", 1, 512)
            p2, _ = make_plan(8, "1x8", 1, 512)
            p3, sp3 = make_plan(8, "1,2,2,3", 7, 512)
            if p1 != p2 or sp != 64 or [g["first_max"] for g in p1] != [512 - 64 * j for j in range(8)]: ok = False; why.append("plan 1x8")
            if sorted(w for g in p1 for w in g["workers"]) != list(range(8)): ok = False; why.append("plan workers")
            if [len(g["workers"]) for g in p3] != [1, 2, 2, 3] or sp3 != 128: ok = False; why.append("plan 1,2,2,3")
            for bad in ("4,4", "1x7", "0,8"):
                try: parse_pattern(bad, 8); ok = False; why.append("pattern %s accepted" % bad)
                except ValueError: pass
            c = cobatch([1.0, 1.001, 1.002, 2.0, 3.0, 3.003])
            if c["cobatched"] != 2 or c["cob_sizes"] != [3, 2] or c["single"] != 1: ok = False; why.append("cobatch %s" % c)
            s = Stamps(lambda t: why.append("cb") if False else None); s.append(5.0); s.append(6.0)
            if s.first != 5.0 or list(s) != [5.0, 6.0]: ok = False; why.append("stamps")
            # the fixed prompt set: two blocks (two plans, two prompt walks, apart in time) send the same sequence
            import types
            for pat, nn in (("1x4", 4), ("1x8", 8), ("1x1", 1)):
                fa = types.SimpleNamespace(seed=1, gen=512, len_jitter=0, ctx=1000, prompt_set="fixed")
                b1 = prompt_seq(mb, fa, make_plan(nn, pat, 1, 512)[0], 6); time.sleep(0.002)
                b2 = prompt_seq(mb, fa, make_plan(nn, pat, 1, 512)[0], 6)
                if b1 != b2 or len(b1) != nn * 6: ok = False; why.append("fixed %s: blocks differ" % pat)
                if len(set(x[4] for x in b1)) != len(b1): ok = False; why.append("fixed %s: a repeated prompt in a block" % pat)
                na = types.SimpleNamespace(seed=1, gen=512, len_jitter=0, ctx=1000, prompt_set="nonce")
                n1 = prompt_seq(mb, na, make_plan(nn, pat, 1, 512)[0], 2); time.sleep(0.002)
                n2 = prompt_seq(mb, na, make_plan(nn, pat, 1, 512)[0], 2)
                if [x[:4] for x in n1] != [x[:4] for x in n2] or n1 == n2: ok = False; why.append("nonce %s: not mbt2's (ids equal, texts differ)" % pat)
            if prompt_of(mb, fa, 3, 0).split("\n", 1)[0].split()[3] != "3": ok = False; why.append("worker index not field 4")
        except Exception as e:
            ok = False; why.append("exception %s" % e)
        print("SELFTEST " + ("OK: mb.py %s loads, %s present, stream() takes stamps + extra; plan 1x8 / 1,2,2,3, pattern guard, "
                             "co-batch grouping, first-token stamps; --prompt-set fixed: two blocks' sequences identical (ids, order, "
                             "n_predict, prompt sha1) at 1x1 / 1x4 / 1x8, no repeat inside a block; nonce = mbt2's (texts differ)" % (a.mb, ", ".join(NEED)) if ok else
                             "FAILED: missing %s %s" % (miss, "; ".join(why))))
        return 0 if ok else 1
    if miss or a.temperature is None:
        print("MB " + json.dumps({"error": "mb.py lacks %s or no --temperature" % miss}), flush=True)
        return 1
    if a.url: mb.URL = a.url.rstrip("/")
    rec = Recorder()
    orig, temp = mb.stream, a.temperature

    def stream(msgs, max_tokens, effort="medium", stamps=None, extra=None):
        e = dict(extra or {})
        e["temperature"] = temp
        if a.sched == "mb" and stamps is None:           # the old loop: record the first token for the client-side mix
            st, ts = Stamps(), time.time()
            r = orig(msgs, max_tokens, effort=effort, stamps=st, extra=e)
            rec.add({"ev": "done", "t_send": ts, "t_first": st.first, "n": r.get("n"), "max_tokens": max_tokens, "prompt": r.get("prompt"),
                     "cached": r.get("cached"), "t0": r.get("t0"), "ttft": r.get("ttft"), "t_end": time.time(), "err": None})
            return r
        return orig(msgs, max_tokens, effort=effort, stamps=stamps, extra=e)

    mb.stream = stream
    with urllib.request.urlopen(mb.URL + "/v1/models", timeout=60) as r:
        mb.MODEL = json.load(r)["data"][0]["id"]
    for ph in a.phases.split(","):
        if ph.startswith("c"):
            res = phase_fixed(mb, int(ph[1:]), a, rec) if a.sched == "fixed" else phase_mb(mb, int(ph[1:]), a, rec)
        elif ph == "pf":
            res = mb.phase_pf(a)
        else:
            res = mb.phase_turns(a)
        print("MB " + json.dumps(dict(tag=a.tag, temperature=temp, **res)), flush=True)
    rec.dump(a.reqs, a.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
