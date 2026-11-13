#!/usr/bin/env python3
"""dstld2.py (12.11.2026, ds-main-port noa builder; lever NOA's DST item 57-p4zyzzzzz-ds-r00zzz-cz-tdst2) = dstld.py
(queue-ar/dsmain/dst, md5 24c2506f, COPIED, the original not edited): its payloads, step-log parser, chunk tools and the
window readings kept; the decode load and the newcomers changed for the coordinator's design rules (12.11.):
  * the decode load (--load): ONE long stream started first (--long-tok greedy tokens, ignore_eos, temperature 0, its own
    nonce: it decodes through the whole prefill, so >= 1 decode stream is alive in the window; closed when the run ends)
    + --conc - 1 mb.py closed-loop streams (ctx --ctx, gen --gen: the turnovers, server-default sampling as dstld's c4);
    without --load NO decode stream at all (the no-decode case, on purpose);
  * newcomers INSIDE the probe's prefill window: --newc-at "3,8,14" (s after the probe is sent) and --newc-ctx
    "1100,1330,4400" (mb.py token sizes), --newc-tok greedy tokens each (temperature 0, effort none); every prompt carries
    --nonce (the item's: the same bytes in every cell and in the alone reference, fresh to the server; the item resets the
    prefix cache before every run, so each must read cached 0 -- the guard checks it);
  * readings as dstld's (window [probe sent, its first token]: c4 tokens / s over ALL decode streams, TPOT inside a request,
    turnovers and their TTFT) + newc_ttft_p50 / max / each, newc_cached, newc_prompts, long_alive (the long stream's
    stamps span the window), long_tokens;
  * c8id / c8id-alone: a deterministic c8 burst -- 8 workers x --reqs requests of ~--ctx tokens, --tok greedy tokens,
    ignore_eos, temperature 0, prompts "# c8id NONCE i j" (equal lengths: the workers turn over together, waves of 8
    newcomers) -- and the same prompts one by one with nothing else running; cmpc8 A B compares the answers.

  dstld2.py run --url URL --tests ds41_tests.py --mb mb.py --depth N --ntok T [--load --conc 4 --warm 10 --long-tok 4096]
            [--newc-at 3,8,14 --newc-ctx 1100,1330,4400 --newc-tok 64 --nonce X --newc-out F] --out OUT.json --text OUT.txt
  dstld2.py newc-alone --url URL --tests .. --mb .. --newc-ctx .. --newc-tok .. --nonce X --newc-out F
  dstld2.py c8id | c8id-alone --url URL --tests .. --mb .. --conc 8 --reqs 2 --ctx 1000 --tok 64 --nonce X --c8-out F
  dstld2.py cmpc8 A.json B.json      dstld2.py chunks STEPS.log --prompt N [--json OUT]      dstld2.py selftest --tests .. --mb ..
Standard library only.
"""
import argparse, importlib.util, json, re, sys, threading, time, urllib.request

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


def newc_ctxs(a):
    return [int(x) for x in str(a.newc_ctx).split(",") if x.strip()]


def newc_ats(a):
    return [float(x) for x in str(a.newc_at).split(",") if x.strip()]


def newc_msgs(mb, nonce, k, ctx):
    # newcomer k: the item's nonce + mb.py's text at seed 901 + k: the same bytes in every cell and alone, fresh to the server
    return [{"role": "user", "content": "# newc %s %d\n" % (nonce, k) + mb.text(int(ctx * 3.9), 901 + k) + "\n\n" + mb.TASKS[k % len(mb.TASKS)]}]


def newc_one(mb, a, k, ctx, out):
    try:
        r = mb.stream(newc_msgs(mb, a.nonce, k, ctx), a.newc_tok, effort="none", extra={"temperature": 0})
        out[k] = dict(k=k, t0=r["t0"], ttft=r["ttft"], prompt=r["prompt"], cached=r["cached"], n=r["n"], text=r["text"])
    except Exception as e:
        out[k] = dict(k=k, err=str(e)[:300])


def setup(a):
    t = load(a.tests, "ds41_tests_ro")
    mb = load(a.mb, "mb_ro")
    url = a.url.rstrip("/")
    t.URL = url
    mb.URL = url
    t.MODEL = t.model_id()
    mb.MODEL = t.MODEL
    return t, mb


def newc_alone(a):
    t, mb = setup(a)
    out = {}
    cs = newc_ctxs(a)
    for k, c in enumerate(cs):
        newc_one(mb, a, k, c, out)
    res = [out[k] for k in range(len(cs))]
    with open(a.newc_out, "w") as f:
        json.dump(res, f)
    bad = [r for r in res if r.get("err") or not r.get("text")]
    print("NEWC alone %d prompts, prompt tokens %s, cached %s, TTFT %s s%s" % (len(res), [r.get("prompt") for r in res],
          [r.get("cached") for r in res], [round(r["ttft"], 2) for r in res if r.get("ttft")],
          (" -- VIRHE %d failed" % len(bad)) if bad else ""), flush=True)
    return 1 if bad else 0


def long_stream(mb, a, st, stop, out):
    """the long decode stream: greedy, ignore_eos, --long-tok tokens; read until the run stops it (the connection is closed:
    the server cancels the request at its next commit)"""
    p = {"model": mb.MODEL, "max_tokens": a.long_tok, "stream": True, "stream_options": {"include_usage": True},
         "temperature": 0, "ignore_eos": True, "reasoning_effort": "none",
         "messages": [{"role": "user", "content": "# long %d\n" % time.time_ns() + mb.text(int(a.ctx * 3.9), 4242) + "\n\n" + mb.TASKS[1]}]}
    req = urllib.request.Request(mb.URL + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    out["t0"] = time.time()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if stop.is_set():
                    break
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                for c in ev.get("choices", []):
                    d = c.get("delta", {})
                    if (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or ""):
                        st.append(time.time())
                        n += 1
        out["ended"] = "stopped" if stop.is_set() else "finished"
    except Exception as e:
        out["err"] = str(e)[:200]
    out["n"] = n
    out["t1"] = time.time()


def q_of(v, f):
    return round(v[min(len(v) - 1, int(f * len(v)))], 1) if v else None


def run(a):
    t, mb = setup(a)
    payload = deep_payload(t, a.depth, a.ntok)
    res = {}
    stop, recs, th = threading.Event(), [], []
    nst = a.conc if a.load else 0
    stamps = [[] for _ in range(nst)]
    lres = {}
    if a.load:
        tl = threading.Thread(target=long_stream, args=(mb, a, stamps[0], stop, lres), daemon=True)
        tl.start()
        th.append(tl)
        if a.conc > 1:
            th += mb.closed_loop(a.conc - 1, 0, a.ctx, a.gen, stop, recs, stamps_by=stamps[1:])
        time.sleep(a.warm)

    def probe():
        res["t0"] = time.time()
        try:
            res["r"] = t.chat(dict(payload), timeout=7200)
        except Exception as e:
            res["err"] = str(e)[:300]
        res["t1"] = time.time()

    t_start = time.time()
    tp = threading.Thread(target=probe)
    tp.start()
    nres, tn = {}, []
    cs, ats = newc_ctxs(a), newc_ats(a)
    for k, (c, at) in enumerate(zip(cs, ats)):
        time.sleep(max(0.0, t_start + at - time.time()))
        x = threading.Thread(target=newc_one, args=(mb, a, k, c, nres))
        x.start()
        tn.append(x)
    tp.join()
    for x in tn:
        x.join()
    t_end = time.time()
    stop.set()
    for x in th:
        x.join(timeout=900)
    r = res.get("r") or {}
    u = r.get("usage") or {}
    text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
    with open(a.text, "w") as f:
        f.write(text)
    info = dict(which="deep", load=bool(a.load), status=r.get("status"), prompt_tokens=u.get("prompt_tokens"),
                cached=t.cached(u) if u else None, ttft=round(r.get("ttft") or 0, 2), tokens=u.get("completion_tokens"),
                finish=r.get("finish"), chars=len(text), seconds=round(res.get("t1", t_end) - res.get("t0", t_start), 2),
                err=res.get("err") or r.get("raw_err"))
    p0 = res.get("t0")
    p1 = p0 + r["ttft"] if p0 and r.get("ttft") else None
    if a.load and p0 and p1 and p1 > p0:
        win = p1 - p0
        per = [sum(1 for x in st if p0 <= x <= p1) for st in stamps]
        pg = sorted(1000.0 * (b - x) for st in stamps for x, b in zip(st, st[1:]) if p0 <= x and b <= p1)
        sp = sorted(n / win for n in per)
        info.update(pf_win_s=round(win, 2), c4_pf_tokens=sum(per), c4_pf_agg_tps=round(sum(per) / win, 1),
                    c4_pf_stream_tps_p50=round(sp[len(sp) // 2], 1) if sp else None, c4_pf_streams=per,
                    tpot_pf_n=len(pg), tpot_pf_p50=q_of(pg, 0.5), tpot_pf_p90=q_of(pg, 0.9), tpot_pf_max=round(pg[-1], 1) if pg else None)
        # dstld's split: a gap of stream i is a turnover when one of its requests was sent inside it (stream 0 = the long one)
        starts = [[]] + [sorted(x["t0"] for x in recs if x.get("w") == i - 1 and x.get("t0")) for i in range(1, nst)]
        gin, gturn = [], []
        for i, st in enumerate(stamps):
            for x, b in zip(st, st[1:]):
                if not (p0 <= x and b <= p1):
                    continue
                (gturn if any(x < s <= b for s in starts[i]) else gin).append(1000.0 * (b - x))
        gin.sort()
        turn = [x for x in recs if x.get("w") is not None and not x.get("err") and p0 <= x["t0"] <= p1]
        nt = sorted(1000.0 * x["ttft"] for x in turn if x.get("ttft"))
        ls = stamps[0]
        info.update(tpot_in_n=len(gin), tpot_in_p50=q_of(gin, 0.5), tpot_in_p90=q_of(gin, 0.9), tpot_in_p99=q_of(gin, 0.99),
                    tpot_in_max=round(gin[-1], 1) if gin else None, turn_n=len(gturn),
                    turn_gap_max=round(max(gturn), 1) if gturn else None, turn_req_n=len(turn),
                    turn_ttft_p50=round(nt[len(nt) // 2], 1) if nt else None, turn_ttft_p90=q_of(nt, 0.9),
                    turn_ttft_max=round(nt[-1], 1) if nt else None, turn_prompts=[x.get("prompt") for x in turn],
                    long_alive=bool(ls and ls[0] <= p0 and ls[-1] >= p1), long_tokens=lres.get("n"), long_err=lres.get("err"),
                    long_win_max_gap=round(max([1000.0 * (b - x) for x, b in zip(ls, ls[1:]) if p0 <= x and b <= p1] or [0]), 1))
    if cs:
        nr = [nres.get(k, dict(k=k, err="no result")) for k in range(len(cs))]
        if a.newc_out:
            with open(a.newc_out, "w") as f:
                json.dump(nr, f)
        ntt = sorted(1000.0 * x["ttft"] for x in nr if x.get("ttft"))
        info.update(newc_n=len(nr), newc_err=sum(1 for x in nr if x.get("err")),
                    newc_sent_after_s=[round(x["t0"] - p0, 2) if x.get("t0") and p0 else None for x in nr],
                    newc_ttft_p50=round(ntt[len(ntt) // 2], 1) if ntt else None, newc_ttft_max=round(ntt[-1], 1) if ntt else None,
                    newc_ttft=[round(1000.0 * x["ttft"], 1) if x.get("ttft") else None for x in nr],
                    newc_prompts=[x.get("prompt") for x in nr], newc_cached=[x.get("cached") for x in nr],
                    newc_in_window=[bool(x.get("t0") and p1 and x["t0"] < p1) for x in nr])
    if a.stamps:
        with open(a.stamps, "w") as f:
            json.dump(dict(probe_t0=p0, probe_ttft=r.get("ttft"), probe_t1=res.get("t1"), stamps=stamps, long=lres,
                           requests=[dict(w=x.get("w"), t0=x.get("t0"), ttft=x.get("ttft"), n=x.get("n"), prompt=x.get("prompt"),
                                          err=x.get("err")) for x in recs]), f)
    info.update(c4_requests=len([x for x in recs if not x.get("err")]), c4_errors=sum(1 for x in recs if x.get("err")))
    with open(a.out, "w") as f:
        json.dump(info, f)
    print("LONGID " + json.dumps(info), flush=True)
    ok = r.get("status") == 200 and not res.get("err") and info["c4_errors"] == 0 and not info.get("newc_err")
    return 0 if ok else 1


def c8_msgs(mb, nonce, i, j, ctx):
    return [{"role": "user", "content": "# c8id %s %d %d\n" % (nonce, i, j) + mb.text(int(ctx * 3.9), 500 + 7 * i + j) + "\n\n" + mb.TASKS[(i + j) % len(mb.TASKS)]}]


def c8_one(mb, a, i, j):
    try:
        r = mb.stream(c8_msgs(mb, a.nonce, i, j, a.ctx), a.tok, effort="none", extra={"temperature": 0, "ignore_eos": True})
        return dict(i=i, j=j, t0=r["t0"], ttft=r["ttft"], prompt=r["prompt"], cached=r["cached"], n=r["n"], text=r["text"])
    except Exception as e:
        return dict(i=i, j=j, err=str(e)[:300])


def c8id(a, alone):
    t, mb = setup(a)
    out = []
    if alone:
        for i in range(a.conc):
            for j in range(a.reqs):
                out.append(c8_one(mb, a, i, j))
    else:
        lock = threading.Lock()
        def worker(i):
            for j in range(a.reqs):
                x = c8_one(mb, a, i, j)
                with lock:
                    out.append(x)
        ws = [threading.Thread(target=worker, args=(i,)) for i in range(a.conc)]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
    out.sort(key=lambda x: (x["i"], x["j"]))
    with open(a.c8_out, "w") as f:
        json.dump(out, f)
    bad = [x for x in out if x.get("err") or not x.get("text")]
    tt = sorted(x["ttft"] for x in out if x.get("ttft"))
    print("C8ID %s %d requests (%d x %d), prompt tokens %s..%s, cached %s, TTFT p50 %s max %s s%s" % (
          "alone" if alone else "burst", len(out), a.conc, a.reqs, min([x.get("prompt") or 0 for x in out] or [0]),
          max([x.get("prompt") or 0 for x in out] or [0]), sorted({x.get("cached") for x in out}, key=str),
          round(tt[len(tt) // 2], 2) if tt else None, round(tt[-1], 2) if tt else None,
          (" -- VIRHE %d failed" % len(bad)) if bad else ""), flush=True)
    return 1 if bad else 0


def cmpc8(a):
    try:
        x = {(r["i"], r["j"]): r for r in json.load(open(a.a))}
        y = {(r["i"], r["j"]): r for r in json.load(open(a.b))}
    except (OSError, ValueError, KeyError) as e:
        print("not compared (%s)" % type(e).__name__)
        return 1
    bad = [k for k in sorted(x) if not x[k].get("text") or (y.get(k) or {}).get("text") != x[k]["text"]]
    print(("IDENTICAL (%d answers)" % len(x)) if x and not bad else ("DIFFERENT: %d of %d %s" % (len(bad), len(x), bad[:8])))
    return 0 if x and not bad else 1


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
    if "stamps_by" not in inspect.signature(mb.closed_loop).parameters or "extra" not in inspect.signature(mb.stream).parameters:
        print("SELFTEST FAIL mb.closed_loop takes no stamps_by or mb.stream no extra")
        return 1
    d = deep_payload(t)
    if not d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n"):
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
    # dstld2: newcomers and the c8 burst are the same bytes for the same nonce, distinct per index, distinct per nonce
    m1, m2 = newc_msgs(mb, "N1", 0, 1100), newc_msgs(mb, "N1", 0, 1100)
    assert m1 == m2 and m1 != newc_msgs(mb, "N1", 1, 1100) and m1 != newc_msgs(mb, "N2", 0, 1100) and len(m1[0]["content"]) > 3000, "newc"
    c1 = c8_msgs(mb, "N1", 3, 1, 1000)
    assert c1 == c8_msgs(mb, "N1", 3, 1, 1000) and c1 != c8_msgs(mb, "N1", 3, 0, 1000) and c1 != c8_msgs(mb, "N1", 2, 1, 1000), "c8id"
    class A:
        newc_ctx, newc_at = "1100,1330,4400", "3,8,14"
    assert newc_ctxs(A) == [1100, 1330, 4400] and newc_ats(A) == [3.0, 8.0, 14.0], "lists"
    # dstld: the gap split -- a turnover is the gap a request of the same stream was sent in
    st = [1.0, 1.03, 1.06, 2.0, 2.03, 8.7, 8.73]
    starts = [5.0]
    gin = [round(1000 * (b - x), 1) for x, b in zip(st, st[1:]) if not any(x < t <= b for t in starts)]
    gturn = [round(1000 * (b - x), 1) for x, b in zip(st, st[1:]) if any(x < t <= b for t in starts)]
    assert gin == [30.0, 30.0, 940.0, 30.0, 30.0] and gturn == [6670.0], (gin, gturn)
    print("SELFTEST OK (helpers load; payloads; chunk parser: alone, loaded with a later same-length prefill, N - 1, ctx-is-end; "
          "dstld's gap split; dstld2's newcomer and c8id prompts fixed per nonce)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "chunks", "cmpchunks", "selftest", "newc-alone", "c8id", "c8id-alone", "cmpc8"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="")
    ap.add_argument("--tests")
    ap.add_argument("--mb")
    ap.add_argument("--out")
    ap.add_argument("--text")
    ap.add_argument("--json")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--long-tok", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=64000)
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--stamps", default="")
    ap.add_argument("--newc-at", default="")
    ap.add_argument("--newc-ctx", default="")
    ap.add_argument("--newc-tok", type=int, default=64)
    ap.add_argument("--newc-out", default="")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--reqs", type=int, default=2)
    ap.add_argument("--tok", type=int, default=64)
    ap.add_argument("--c8-out", default="")
    a = ap.parse_args()
    if a.mode == "run":
        sys.exit(run(a))
    if a.mode == "chunks":
        a.log = a.files[0]
        sys.exit(chunks(a))
    if a.mode == "newc-alone":
        sys.exit(newc_alone(a))
    if a.mode in ("c8id", "c8id-alone"):
        sys.exit(c8id(a, a.mode == "c8id-alone"))
    if a.mode == "cmpchunks":
        a.a, a.b = a.files[0], a.files[1]
        sys.exit(cmpchunks(a))
    if a.mode == "cmpc8":
        a.a, a.b = a.files[0], a.files[1]
        sys.exit(cmpc8(a))
    sys.exit(selftest(a))
