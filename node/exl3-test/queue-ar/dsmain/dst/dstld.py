#!/usr/bin/env python3
"""dstld.py (11.11.2026, lever DST item 57-p4zyzzzzz-ds-r04z-dst) = pskld.py (queue-ar/dsmain/psk, md5 dca443b6, copied, the
original not edited) + the decode users' readings split by request, nothing else changed:
  * pskld's tpot_* / tpot_pf_* count every gap between two token stamps of a stream, and a stream's stamps run across its
    requests (mb.closed_loop's stamps_by): the gap between a request's last token and the next request's first is that
    request's TTFT (its ~1000-token prompt prefilled beside the long probe), not a decode stall. r00z's 6.66 / 4.92 s
    "max TPOT" were those gaps (the engine's own ttft=6.65 s / 4.91 s of the c4 requests sent in the window).
  * here, over the probe's prefill window [sent, first token]: tpot_in_* = gaps INSIDE a request only (p50 / p90 / p99 /
    max), turn_* = the request turnovers (gap count, max, and the c4 requests sent in the window: their TTFT p50 / max,
    prompt tokens), beside pskld's unchanged all-gap tpot_pf_* (comparable to r00z);
  * --stamps FILE: every stream's token stamps and request start times (t0, ttft, prompt) as JSON, for later analysis.
  * --newc K --newc-at S --newc-tok T --newc-out FILE (with --load): a FIXED newcomer set -- K deterministic ~--ctx-token
    prompts (no nonce: newc_payload, mb.py's text at seeds 901.., its TASKS), sent together S s after the probe, T greedy
    tokens each (temperature 0, effort none; --newc-long M: the last one M x as long) -> FILE: per newcomer its prompt
    tokens, cached, TTFT, text; info gets
    newc_ttft_p50 / p90 / max. `dstld.py newc-alone --newc K ...` runs the same set one by one with nothing else
    running (the identity reference for each newcomer's answer).
  * turn_ttft_p90 beside p50 / max.
pskld.py's own text:
pskld.py (11.11.2026, ds41-psk item 57-p4zyzzzzz-ds-r00z-psk) = longid2.py (queue-ar/dsmain, md5 at copy time in the item log) +
three knobs, nothing else changed: --depth / --ntok (deepq.py's fixed prompt at another depth / answer length: the same filler,
offset and task, so a 64k probe is deepq.py's request at depth 64000 byte for byte), --pf-tok 0 (no extra fresh prefill under
--load: the probe is the only prefill), and the c4 streams' readings over the PROBE'S PREFILL window [probe sent, its first
token] (pf_win_s, c4_pf_tokens, c4_pf_agg_tps, c4_pf_stream_tps_p50, tpot_pf_p50 / p90 / max) beside longid2's whole-probe TPOT.
longid2.py's own text:
longid2.py (= longid.py 978e2549 + the c4 streams' token arrival times under --load: TPOT of the decode users WHILE the
probe runs -- every inter-token gap of every stream inside [probe sent, probe done], p50 / p90 / p99 / max in ms -- in OUT.json
as tpot_*; for lever BI's class (ii) chunking options, 11.11.2026, ds-main-port BI agent). longid.py's own text:
longid.py (11.11.2026, queue-ar/dsmain, ds-main-port) -- lever BI, class (ii) check: LONG prompts alone vs under load, and
where their prefill chunks fell. Driver of dsm-bilid7.sh's starts l0 / l5 (item 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid).
Reuses ds41_tests.py (text_chars, chat, cached; the copy test's exact request) and mb.py (c4 closed-loop streams, the fresh
~32k prefill prompt, stream) READ ONLY via importlib; nothing of theirs changes.

  longid.py run --url URL --tests ds41_tests.py --mb mb.py --which copy|deep --out OUT.json --text OUT.txt
                [--load] [--conc 4] [--warm 10] [--offset 0.5] [--pf-tok 32768]
      copy = ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --offset 0 (dsm-stack2.sh copyt's request;
             the text written is the answer's content, as copy<tag>-run1.txt);
      deep = deepq.py's fixed ~135000-token prompt (CPT 3.95, offset 1200, "# deepq fixed prompt v1"), 512 greedy tokens,
             ignore_eos, thinking off; the text written is content (+ "\\n[reasoning]\\n" + reasoning), as deepq.py's OUT.
      --load: mb c4 streams (mb.py closed_loop: ctx 1000, gen 512) start, --warm s later the probe is sent, --offset s after it ONE
      fresh ~--pf-tok prefill (mb.py's nonce-first prompt, 1 token out); the c4 streams run until the probe is done. Without
      --load the probe runs alone. OUT.json: status, prompt_tokens, cached, ttft, tokens, finish, seconds (+ the prefill's
      offset / TTFT / prompt / cached and the c4 requests and errors under --load). Prints "LONGID {json}".
  longid.py chunks STEPS.log --prompt N [--json OUT.json]
      STEPS.log = the engine log lines of the probe's window (RADIANCE_LOG_STEPS=1: "step K PHASE n_seq=.. n_tok=..  [i] slot=S
      q=Q ctx=C ..."). The probe's prefill = the EARLIEST run in the window (a slot's entry with ctx 0 or ctx == q starts a run)
      whose cumulative q reaches N (or N - 1: the last prompt token may ride the first decode step). Prints "CHUNKS {json}":
      the chunk list [start, q, phase, n_seq, n_tok], the boundaries, how many chunks ran in MIXED steps.
  longid.py cmpchunks A.json B.json
      two CHUNKS results: "SAME BOUNDARIES (n chunks)" or the first differing chunk (index, A's and B's start / q).
  longid.py selftest --tests ds41_tests.py --mb mb.py
      offline: both helpers load and have what this driver calls; the chunk parser on a synthetic log (alone, loaded with a
      competing prefill of the same length started later, the N - 1 case, ctx-is-end semantics). No server, writes nothing.
Standard library only.
"""
import argparse, importlib.util, json, re, sys, threading, time

STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
ENT = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")
DEEP_CPT, DEEP_OFF = 3.95, 1200
DEEP_TASK = ("\n\nYou have read the source files above. Write a long, detailed technical review of this code: explain what "
             "the main modules do, how they depend on each other, which functions are the most complex and why, and "
             "propose concrete refactorings with example code.")
COPY_INSTR = ("Copy the text between <<<BEGIN>>> and <<<END>>> exactly, character for character, including whitespace "
              "and blank lines. Output only the copied text, without the markers and without any comment.\n\n<<<BEGIN>>>\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def copy_payload(t, chars=2400, offset=0, prefix_chars=120000, max_tokens=900):
    # ds41_tests.cmd_copy's request, byte for byte
    src = t.text_chars(chars, offset=offset)
    pre = ("Background material, for context only (do not copy this part):\n<<<CONTEXT>>>\n"
           + t.text_chars(prefix_chars, offset=offset + 300) + "\n<<<END CONTEXT>>>\n\n")
    msgs = [{"role": "user", "content": pre + COPY_INSTR + src + "\n<<<END>>>"}]
    return {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": "none"}


def deep_payload(t, depth=135000, ntok=512):
    # deepq.py's request, byte for byte
    body = t.text_chars(int(depth * DEEP_CPT), offset=DEEP_OFF)
    msgs = [{"role": "user", "content": "# deepq fixed prompt v1\n" + body + DEEP_TASK}]
    return {"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True}


def newc_msgs(mb, k, ctx):
    # the fixed newcomer k: deterministic (no nonce), so the alone reference and the loaded run send the same bytes
    return [{"role": "user", "content": "# newc %d\n" % k + mb.text(int(ctx * 3.9), 901 + k) + "\n\n" + mb.TASKS[k % len(mb.TASKS)]}]


def newc_ctx(a, k):
    # --newc-long M: the last fixed newcomer is M x --ctx tokens (a multi-piece prompt for sched_bypass_total)
    return a.ctx * (a.newc_long if k == a.newc - 1 else 1)


def newc_one(mb, k, ctx, ntok, out):
    try:
        r = mb.stream(newc_msgs(mb, k, ctx), ntok, effort="none", extra={"temperature": 0})
        out[k] = dict(k=k, t0=r["t0"], ttft=r["ttft"], prompt=r["prompt"], cached=r["cached"], n=r["n"], text=r["text"])
    except Exception as e:
        out[k] = dict(k=k, err=str(e)[:300])


def newc_alone(a):
    t = load(a.tests, "ds41_tests_ro")
    mb = load(a.mb, "mb_ro")
    mb.URL = a.url.rstrip("/")
    t.URL = mb.URL
    mb.MODEL = t.model_id()
    out = {}
    for k in range(a.newc):
        newc_one(mb, k, newc_ctx(a, k), a.newc_tok, out)
    res = [out[k] for k in range(a.newc)]
    with open(a.newc_out, "w") as f:
        json.dump(res, f)
    bad = [r for r in res if r.get("err") or not r.get("text")]
    print("NEWC alone %d prompts, prompt tokens %s, TTFT %s s%s" % (len(res), [r.get("prompt") for r in res],
          [round(r["ttft"], 2) for r in res if r.get("ttft")], (" -- VIRHE %d failed" % len(bad)) if bad else ""), flush=True)
    return 1 if bad else 0


def run(a):
    t = load(a.tests, "ds41_tests_ro")
    url = a.url.rstrip("/")
    t.URL = url
    t.MODEL = t.model_id()
    payload = copy_payload(t) if a.which == "copy" else deep_payload(t, a.depth, a.ntok)
    res, pf = {}, {}
    stop, recs, th = threading.Event(), [], []
    mb = None
    if a.load:
        mb = load(a.mb, "mb_ro")
        mb.URL = url
        mb.MODEL = t.MODEL
        stamps = [[] for _ in range(a.conc)]
        th = mb.closed_loop(a.conc, 0, a.ctx, a.gen, stop, recs, stamps_by=stamps)
        time.sleep(a.warm)

    def probe():
        res["t0"] = time.time()
        try:
            res["r"] = t.chat(dict(payload), timeout=7200)
        except Exception as e:
            res["err"] = str(e)[:300]
        res["t1"] = time.time()

    def prefill():
        q = f"# pf {time.time_ns()}\n" + mb.text(int(a.pf_tok * 3.9), 777) + "\n\nSummarise the code above in one sentence."
        try:
            r = mb.stream([{"role": "user", "content": q}], 1, effort="none")
            pf.update(t0=r["t0"], ttft=r["ttft"], prompt=r["prompt"], cached=r["cached"])
        except Exception as e:
            pf["err"] = str(e)[:300]

    t_start = time.time()
    tp = threading.Thread(target=probe)
    tp.start()
    tf = None
    if a.load and a.pf_tok > 0:
        time.sleep(a.offset)
        tf = threading.Thread(target=prefill)
        tf.start()
    nres, tn = {}, []
    if a.load and a.newc > 0:
        time.sleep(max(0.0, a.newc_at - (a.offset if a.pf_tok > 0 else 0.0)))
        tn = [threading.Thread(target=newc_one, args=(mb, k, newc_ctx(a, k), a.newc_tok, nres)) for k in range(a.newc)]
        for x in tn:
            x.start()
    tp.join()
    if tf:
        tf.join()
    for x in tn:
        x.join()
    t_end = time.time()
    if a.load:
        stop.set()
        for x in th:
            x.join(timeout=900)
    r = res.get("r") or {}
    u = r.get("usage") or {}
    if a.which == "copy":
        text = r.get("content") or ""
    else:
        text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
    with open(a.text, "w") as f:
        f.write(text)
    info = dict(which=a.which, load=bool(a.load), status=r.get("status"), prompt_tokens=u.get("prompt_tokens"),
                cached=t.cached(u) if u else None, ttft=round(r.get("ttft") or 0, 2), tokens=u.get("completion_tokens"),
                finish=r.get("finish"), chars=len(text), seconds=round(res.get("t1", t_end) - res.get("t0", t_start), 2),
                err=res.get("err") or r.get("raw_err"))
    if a.load:
        c4 = [x for x in recs if not x.get("err") and t_start <= x["t0"] < t_end]
        w0, w1 = res.get("t0", t_start), res.get("t1", t_end)
        gaps = sorted(1000.0 * (b - x) for st in stamps for x, b in zip(st, st[1:]) if w0 <= x and b <= w1)
        def q(f):
            return round(gaps[min(len(gaps) - 1, int(f * len(gaps)))], 1) if gaps else None
        info.update(tpot_n=len(gaps), tpot_p50=q(0.50), tpot_p90=q(0.90), tpot_p99=q(0.99),
                    tpot_max=round(gaps[-1], 1) if gaps else None)
        # the probe's prefill window: [sent, sent + TTFT]
        p0 = res.get("t0")
        p1 = p0 + r["ttft"] if p0 and r.get("ttft") else None
        if p0 and p1 and p1 > p0:
            win = p1 - p0
            per = [sum(1 for x in st if p0 <= x <= p1) for st in stamps]
            pg = sorted(1000.0 * (b - x) for st in stamps for x, b in zip(st, st[1:]) if p0 <= x and b <= p1)
            def qp(f):
                return round(pg[min(len(pg) - 1, int(f * len(pg)))], 1) if pg else None
            sp = sorted(n / win for n in per)
            info.update(pf_win_s=round(win, 2), c4_pf_tokens=sum(per), c4_pf_agg_tps=round(sum(per) / win, 1),
                        c4_pf_stream_tps_p50=round(sp[len(sp) // 2], 1) if sp else None, c4_pf_streams=per,
                        tpot_pf_n=len(pg), tpot_pf_p50=qp(0.50), tpot_pf_p90=qp(0.90), tpot_pf_max=round(pg[-1], 1) if pg else None)
            # dstld: the gaps split by request. A gap (x, b] of stream i is a turnover when one of stream i's requests was
            # sent inside it (closed loop: the next request is sent right after the last token of the one before).
            starts = [sorted(r["t0"] for r in recs if r.get("w") == i and r.get("t0")) for i in range(a.conc)]
            gin, gturn = [], []
            for i, st in enumerate(stamps):
                for x, b in zip(st, st[1:]):
                    if not (p0 <= x and b <= p1):
                        continue
                    (gturn if any(x < t <= b for t in starts[i]) else gin).append(1000.0 * (b - x))
            gin.sort()
            def qi(f):
                return round(gin[min(len(gin) - 1, int(f * len(gin)))], 1) if gin else None
            newc = [r for r in recs if r.get("w") is not None and not r.get("err") and p0 <= r["t0"] <= p1]
            nt = sorted(1000.0 * r["ttft"] for r in newc if r.get("ttft"))
            info.update(turn_ttft_p90=round(nt[min(len(nt) - 1, int(0.9 * len(nt)))], 1) if nt else None)
            info.update(tpot_in_n=len(gin), tpot_in_p50=qi(0.50), tpot_in_p90=qi(0.90), tpot_in_p99=qi(0.99),
                        tpot_in_max=round(gin[-1], 1) if gin else None,
                        turn_n=len(gturn), turn_gap_max=round(max(gturn), 1) if gturn else None,
                        turn_req_n=len(newc), turn_ttft_p50=round(nt[len(nt) // 2], 1) if nt else None,
                        turn_ttft_max=round(nt[-1], 1) if nt else None,
                        turn_prompts=[r.get("prompt") for r in newc])
        if a.newc > 0:
            nr = [nres.get(k, dict(k=k, err="no result")) for k in range(a.newc)]
            with open(a.newc_out, "w") as f:
                json.dump(nr, f)
            ntt = sorted(1000.0 * x["ttft"] for x in nr if x.get("ttft"))
            info.update(newc_n=len(nr), newc_err=sum(1 for x in nr if x.get("err")),
                        newc_sent_after_s=round(min(x["t0"] for x in nr if x.get("t0")) - res["t0"], 2) if any(x.get("t0") for x in nr) and res.get("t0") else None,
                        newc_ttft_p50=round(ntt[len(ntt) // 2], 1) if ntt else None,
                        newc_ttft_p90=round(ntt[min(len(ntt) - 1, int(0.9 * len(ntt)))], 1) if ntt else None,
                        newc_ttft_max=round(ntt[-1], 1) if ntt else None, newc_prompts=[x.get("prompt") for x in nr],
                        newc_cached=[x.get("cached") for x in nr])
        if a.stamps:
            with open(a.stamps, "w") as f:
                json.dump(dict(probe_t0=res.get("t0"), probe_ttft=r.get("ttft"), probe_t1=res.get("t1"), stamps=stamps,
                               requests=[dict(w=x.get("w"), t0=x.get("t0"), ttft=x.get("ttft"), n=x.get("n"),
                                              prompt=x.get("prompt"), err=x.get("err")) for x in recs]), f)
        info.update(pf_off=round(pf["t0"] - res["t0"], 2) if pf.get("t0") and res.get("t0") else None,
                    pf_ttft=round(pf["ttft"], 2) if pf.get("ttft") else None, pf_prompt=pf.get("prompt"), pf_cached=pf.get("cached"),
                    pf_err=pf.get("err"), c4_requests=len(c4), c4_errors=sum(1 for x in recs if x.get("err")))
    with open(a.out, "w") as f:
        json.dump(info, f)
    print("LONGID " + json.dumps(info), flush=True)
    ok = r.get("status") == 200 and not res.get("err") and (not a.load or (not pf.get("err") and info["c4_errors"] == 0))
    return 0 if ok else 1


def parse_steps(lines):
    out = []
    for l in lines:
        m = STEP.search(l)
        if not m:
            continue
        ents = [(int(s), int(q), int(c)) for _, s, q, c in ENT.findall(m.group(5))]
        out.append((int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), ents))
    return out


def find_chunks(steps, n):
    """earliest run reaching n (or n - 1) prompt tokens: [(start, q, phase, n_seq, n_tok, step)], target, semantics"""
    runs = {}      # slot -> current run dict
    done = []      # runs that reached n - 1 or n (first step, chunks, cum, semantics), each at its FIRST reach
    for st, ph, ns, nt, ents in steps:
        for slot, q, ctx in ents:
            r = runs.get(slot)
            cont = r is not None and not r["closed"] and (
                (r["sem"] == "start" and ctx == r["cum"]) or (r["sem"] == "end" and ctx == r["cum"] + q))
            if not cont:
                if ctx == 0 and q > 0:
                    r = runs[slot] = dict(first=st, cum=0, chunks=[], sem="start", closed=False)
                elif ctx == q and q > 0:
                    r = runs[slot] = dict(first=st, cum=0, chunks=[], sem="end", closed=False)
                else:
                    if r is not None:
                        r["closed"] = True
                    continue
            r["chunks"].append((r["cum"], q, ph, ns, nt, st))
            r["cum"] += q
            if r["cum"] in (n - 1, n):
                done.append((r["first"], list(r["chunks"]), r["cum"], r["sem"]))
                r["closed"] = True
            elif r["cum"] > n:
                r["closed"] = True
    if not done:
        return None
    first, ch, cum, sem = min(done, key=lambda d: d[0])
    return dict(first_step=first, chunks=ch, reached=cum, semantics=sem)


def chunks(a):
    with open(a.log, errors="replace") as f:
        steps = parse_steps(f)
    r = find_chunks(steps, a.prompt)
    if r is None:
        res = dict(ok=False, prompt=a.prompt, steps_in_window=len(steps), why="no run reaches the prompt length")
    else:
        ch = r["chunks"]
        res = dict(ok=True, prompt=a.prompt, reached=r["reached"], semantics=r["semantics"], steps_in_window=len(steps),
                   n_chunks=len(ch), boundaries=[c[0] for c in ch], q=[c[1] for c in ch],
                   mixed_chunks=sum(1 for c in ch if c[2] == "MIXED"), max_seq_in_chunk_steps=max(c[3] for c in ch),
                   chunks=[list(c) for c in ch])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f)
    if res["ok"]:
        print("CHUNKS %d chunks over %d prompt tokens (reached %d, ctx = %s), %d in MIXED steps, up to %d sequences a step; starts %s"
              % (res["n_chunks"], a.prompt, res["reached"], res["semantics"], res["mixed_chunks"], res["max_seq_in_chunk_steps"],
                 short(res["boundaries"])), flush=True)
    else:
        print("CHUNKS none: %s (%d step lines in the window)" % (res["why"], len(steps)), flush=True)
    return 0 if res["ok"] else 1


def short(b, k=6):
    return str(b) if len(b) <= 2 * k else "%s ... %s" % (str(b[:k])[:-1], str(b[-k:])[1:])


def cmpchunks(a):
    try:
        x, y = json.load(open(a.a)), json.load(open(a.b))
    except (OSError, ValueError) as e:
        print("CHUNKCMP not compared (%s)" % e)
        return 1
    if not x.get("ok") or not y.get("ok"):
        print("CHUNKCMP not compared (no chunk list: %s / %s)" % (x.get("ok"), y.get("ok")))
        return 1
    bx, by = list(zip(x["boundaries"], x["q"])), list(zip(y["boundaries"], y["q"]))
    if bx == by:
        print("CHUNKCMP SAME BOUNDARIES (%d chunks)" % len(bx))
        return 0
    k = next((i for i, (p, r) in enumerate(zip(bx, by)) if p != r), min(len(bx), len(by)))
    pa = bx[k] if k < len(bx) else None
    pb = by[k] if k < len(by) else None
    print("CHUNKCMP DIFFERENT BOUNDARIES: %d vs %d chunks, first difference at chunk %d (start, q): %s vs %s; B's chunks in MIXED steps %d"
          % (len(bx), len(by), k, pa, pb, y.get("mixed_chunks", 0)))
    return 1


def selftest(a):
    t, mb = load(a.tests, "ds41_tests_ro"), load(a.mb, "mb_ro")
    need_t = [n for n in ("text_chars", "chat", "cached", "model_id") if not callable(getattr(t, n, None))]
    need_m = [n for n in ("closed_loop", "stream", "text") if not callable(getattr(mb, n, None))]
    if need_t or need_m:
        print("SELFTEST FAIL missing ds41_tests %s mb %s" % (need_t, need_m))
        return 1
    import inspect
    if "stamps_by" not in inspect.signature(mb.closed_loop).parameters:
        print("SELFTEST FAIL mb.closed_loop takes no stamps_by (the TPOT stamps)")
        return 1
    p = copy_payload(t)
    d = deep_payload(t)
    if not p["messages"][0]["content"].startswith("Background material") or not d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n"):
        print("SELFTEST FAIL payload shape")
        return 1
    # synthetic logs
    def L(st, ph, ents):
        return "I step %d %s n_seq=%d n_tok=%d" % (st, ph, len(ents), sum(q for _, q, _ in ents)) + "".join(
            "  [%d] slot=%d q=%d ctx=%d" % (i, s, q, c) for i, (s, q, c) in enumerate(ents))
    N = 10000
    alone = [L(1, "prefill", [(0, 4160, 0)]), L(2, "prefill", [(0, 4160, 4160)]), L(3, "prefill", [(0, 1680, 8320)]),
             L(4, "decode", [(0, 1, 10000)])]
    r = find_chunks(parse_steps(alone), N)
    assert r and [c[0] for c in r["chunks"]] == [0, 4160, 8320] and r["reached"] == N, r
    # loaded: c4 decode on slots 1-4, the probe on slot 0, a competing same-length prefill on slot 5 starting later
    ld = [L(1, "decode", [(1, 1, 500), (2, 1, 700)]),
          L(2, "MIXED", [(1, 1, 501), (2, 1, 701), (0, 4158, 0)]),
          L(3, "MIXED", [(1, 1, 502), (0, 2000, 4158), (5, 2158, 0)]),
          L(4, "MIXED", [(1, 1, 503), (0, 3842, 6158), (5, 316, 2158)]),
          L(5, "MIXED", [(1, 1, 504), (5, 4158, 2474)]),
          L(6, "MIXED", [(1, 1, 505), (5, 3368, 6632)])]
    r = find_chunks(parse_steps(ld), N)
    assert r and r["first_step"] == 2 and [c[0] for c in r["chunks"]] == [0, 4158, 6158], r
    # N - 1: the last prompt token rides the first decode step
    nm1 = [L(1, "prefill", [(0, 4160, 0)]), L(2, "prefill", [(0, 5839, 4160)]), L(3, "decode", [(0, 1, 9999)])]
    r = find_chunks(parse_steps(nm1), N)
    assert r and r["reached"] in (N - 1, N), r
    # ctx = the END of the chunk
    end = [L(1, "prefill", [(0, 4160, 4160)]), L(2, "prefill", [(0, 4160, 8320)]), L(3, "prefill", [(0, 1680, 10000)])]
    r = find_chunks(parse_steps(end), N)
    assert r and r["semantics"] == "end" and [c[0] for c in r["chunks"]] == [0, 4160, 8320], r
    # dstld: the fixed newcomers are the same bytes every time, and differ from each other
    m1, m2 = newc_msgs(mb, 0, 1000), newc_msgs(mb, 0, 1000)
    assert m1 == m2 and m1 != newc_msgs(mb, 1, 1000) and len(m1[0]["content"]) > 3000, "newc payload"
    if "extra" not in inspect.signature(mb.stream).parameters:
        print("SELFTEST FAIL mb.stream takes no extra (temperature 0 for the fixed newcomers)")
        return 1
    # dstld: the gap split -- a turnover is the gap a request of the same stream was sent in
    st = [1.0, 1.03, 1.06, 2.0, 2.03, 8.7, 8.73]
    starts = [5.0]
    gin = [round(1000 * (b - x), 1) for x, b in zip(st, st[1:]) if not any(x < t <= b for t in starts)]
    gturn = [round(1000 * (b - x), 1) for x, b in zip(st, st[1:]) if any(x < t <= b for t in starts)]
    assert gin == [30.0, 30.0, 940.0, 30.0, 30.0] and gturn == [6670.0], (gin, gturn)
    print("SELFTEST OK (helpers load; payloads; chunk parser: alone, loaded with a later same-length prefill, N - 1, ctx-is-end; "
          "dstld's gap split: in-request vs turnover)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "chunks", "cmpchunks", "selftest", "newc-alone"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="")
    ap.add_argument("--tests")
    ap.add_argument("--mb")
    ap.add_argument("--which", choices=("copy", "deep"))
    ap.add_argument("--out")
    ap.add_argument("--text")
    ap.add_argument("--json")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--pf-tok", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--depth", type=int, default=135000)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--stamps", default="")
    ap.add_argument("--newc", type=int, default=0)
    ap.add_argument("--newc-at", type=float, default=15.0)
    ap.add_argument("--newc-tok", type=int, default=64)
    ap.add_argument("--newc-out", default="")
    ap.add_argument("--newc-long", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "run":
        sys.exit(run(a))
    if a.mode == "chunks":
        a.log = a.files[0]
        sys.exit(chunks(a))
    if a.mode == "newc-alone":
        sys.exit(newc_alone(a))
    if a.mode == "cmpchunks":
        a.a, a.b = a.files[0], a.files[1]
        sys.exit(cmpchunks(a))
    sys.exit(selftest(a))
