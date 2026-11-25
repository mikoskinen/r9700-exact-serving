#!/usr/bin/env python3
"""longid2.py (= longid.py 978e2549 + the c4 streams' token arrival times under --load: TPOT of the decode users WHILE the
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


def run(a):
    t = load(a.tests, "ds41_tests_ro")
    url = a.url.rstrip("/")
    t.URL = url
    t.MODEL = t.model_id()
    payload = copy_payload(t) if a.which == "copy" else deep_payload(t)
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
    if a.load:
        time.sleep(a.offset)
        tf = threading.Thread(target=prefill)
        tf.start()
    tp.join()
    if tf:
        tf.join()
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
    print("SELFTEST OK (helpers load; payloads; chunk parser: alone, loaded with a later same-length prefill, N - 1, ctx-is-end)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "chunks", "cmpchunks", "selftest"))
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
    a = ap.parse_args()
    if a.mode == "run":
        sys.exit(run(a))
    if a.mode == "chunks":
        a.log = a.files[0]
        sys.exit(chunks(a))
    if a.mode == "cmpchunks":
        a.a, a.b = a.files[0], a.files[1]
        sys.exit(cmpchunks(a))
    sys.exit(selftest(a))
