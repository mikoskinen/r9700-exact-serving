#!/usr/bin/env python3
"""cap48probe.py (12.11.2026, queue-ar/dsmain/cap48, ds-main-port cap48 builder) -- the T 1 identity driver of dsm-cap48.sh
(item 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48, tag xc48). A COPY of stack4/loadid.py (ee7d9663, unchanged beside it) with
the probe request at a given temperature and a FIXED PER-PROMPT SEED, and one more mode (set). greedyset.py's 20 prompts and
its request shape, and mb.py's streams / newcomer prompt / metrics, are loaded READ ONLY (importlib); nothing of theirs changes.

THE REQUEST (ask_t): greedyset.py's /v1/completions body -- {"model", "prompt", "max_tokens", "temperature", "top_p": 1,
"stream": false} -- with "temperature": T and, at T > 0, "seed": SEED_BASE + i (the sampler's draw is a function of (seed,
output position): core/server/oai.cpp "seed", core/sample/sampler.cpp p.pos = n_out + k, so a draft row's draw is the same
with or without speculation). At T 0 the body is greedyset.py's own, byte for byte (no seed key; selftest checks it).
Falls back to /v1/chat/completions (reasoning_effort none) as greedyset.py does.

  cap48probe.py set --url URL --gs GREEDYSET.py --out OUT.jsonl [--conc 1] [--temperature 1] [--seed-base 1000]
                    [--max-tokens 192] [--timeout 300]
      the 20 prompts, --conc in flight (1 = ALONE, one at a time; 16 = the gs16 shape), each with ask_t. OUT = greedyset.py's
      line shape {i, endpoint, text, finish, tokens} (+ temperature, seed). Last line "SET {json}". Exit 0 = 20 answered.
  cap48probe.py run ... (= loadid.py run: --conc mb streams (closed loop, ctx 1000, gen 512, server-default sampling), --warm s
      later the 20 probes ONE AT A TIME with ask_t, --offset s after each a fresh ~--pf-tok newcomer prefill (--pf-tok 0: none))
      + --temperature / --seed-base. Last line "LOADID {json}".
  cap48probe.py cmp --url URL A.jsonl B.jsonl [--json OUT.json] [--label L]   (= loadid.py cmp: byte for byte + the first
      differing char / token; "CMPLINE ...")
  cap48probe.py selftest --gs GREEDYSET.py --mb MB.py
      offline: loadid.py's checks + ask_t's body at T 0 == greedyset.ask's (both captured from a stub post), at T 1 + the seed.
Standard library only.
"""
import argparse, importlib.util, json, statistics, sys, threading, time, urllib.error, urllib.request

sys.dont_write_bytecode = True   # the helpers are loaded read only: no __pycache__ beside them


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def first_diff(a, b):
    """index of the first differing element of two sequences; None when equal"""
    if a == b:
        return None
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def tokenize(url, s):
    try:
        req = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": s}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        t = d.get("tokens")
        return [x if isinstance(x, int) else x.get("id") for x in t] if isinstance(t, list) else None
    except Exception:
        return None


def rd(f):
    try:
        return {d["i"]: d for d in map(json.loads, open(f)) if "i" in d}
    except OSError:
        return {}


def cmp(a):
    x, y = rd(a.a), rd(a.b)
    key = lambda d: (d.get("text"), d.get("finish"), d.get("reasoning"))  # noqa: E731 (greedyset.py --cmp's key)
    ids = sorted(set(x) | set(y))
    diff, missing = [], [i for i in ids if i not in x or i not in y]
    for i in ids:
        if i in missing or key(x[i]) == key(y[i]):
            continue
        ta, tb = x[i].get("text") or "", y[i].get("text") or ""
        c = first_diff(ta, tb)
        ka, kb = (tokenize(a.url.rstrip("/"), ta), tokenize(a.url.rstrip("/"), tb)) if a.url else (None, None)
        k = first_diff(ka, kb) if ka is not None and kb is not None else None
        e = dict(i=i, char=c, tok=k, len_a=len(ta), len_b=len(tb), ntok_a=len(ka) if ka else None, ntok_b=len(kb) if kb else None,
                 fin_a=x[i].get("finish"), fin_b=y[i].get("finish"))
        if c is not None:
            e.update(at_a=ta[max(0, c - 12):c + 24], at_b=tb[max(0, c - 12):c + 24])
        elif x[i].get("reasoning") != y[i].get("reasoning"):
            e["what"] = "reasoning"
        else:
            e["what"] = "finish"
        if "mixed" in y[i]:
            e.update(mixed_b=y[i].get("mixed"), ov_frac_b=y[i].get("ov_frac"))
        diff.append(e)
    r = dict(label=a.label, a=a.a, b=a.b, n_a=len(x), n_b=len(y), n=len(ids), missing=missing, n_diff=len(diff),
             identical=bool(x) and bool(y) and not diff and not missing, diff=diff)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, ensure_ascii=False)
    print("CMP " + json.dumps(r, ensure_ascii=False))
    if not x or not y:
        line = "not compared (answers %d / %d)" % (len(x), len(y))
    elif r["identical"]:
        line = "IDENTICAL: %d of %d answers" % (len(ids), len(ids))
    else:
        line = "%d of %d answers differ%s: %s" % (len(diff), len(ids), (", %d missing %s" % (len(missing), missing)) if missing else "",
               ", ".join("p%d char %s tok %s%s" % (e["i"], e["char"], "-" if e["tok"] is None else e["tok"],
                                                     " (%s)" % e["what"] if e.get("what") else "") for e in diff) or "-")
    print("CMPLINE " + line)
    return 0 if r["identical"] else 1


def ask_t(gs, url, model, i, mt, timeout, mode, temp, seed):
    """greedyset.ask with the temperature and, at T > 0, the seed (greedyset.py's body otherwise, key for key)"""
    p = gs.PROMPTS[i]
    temp = int(temp) if float(temp).is_integer() else temp   # 0, not 0.0: greedyset.py's body byte for byte
    if mode[0] != "chat":
        body = {"model": model, "prompt": p, "max_tokens": mt, "temperature": temp, "top_p": 1, "stream": False}
        if temp > 0:
            body["seed"] = seed
        try:
            d = gs.post(url, "/v1/completions", body, timeout)
            c = d["choices"][0]
            return {"i": i, "endpoint": "completions", "text": c.get("text"), "finish": c.get("finish_reason"),
                    "tokens": (d.get("usage") or {}).get("completion_tokens")}
        except urllib.error.HTTPError as e:
            if e.code not in (400, 404, 405, 422, 501): raise
            mode[0] = "chat"
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": temp, "top_p": 1,
            "stream": False, "reasoning_effort": "none"}
    if temp > 0:
        body["seed"] = seed
    d = gs.post(url, "/v1/chat/completions", body, timeout)
    c = d["choices"][0]
    m = c.get("message") or {}
    return {"i": i, "endpoint": "chat", "text": m.get("content"), "reasoning": m.get("reasoning_content"),
            "finish": c.get("finish_reason"), "tokens": (d.get("usage") or {}).get("completion_tokens")}


def set_run(a):
    gs = load(a.gs, "greedyset_ro")
    url = a.url.rstrip("/")
    model = gs.model_id(url, 10)
    res, err, lock, nxt, mode = {}, [], threading.Lock(), [0], ["completions"]
    t0 = time.time()

    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= len(gs.PROMPTS): return
            try:
                r = ask_t(gs, url, model, i, a.max_tokens, a.timeout, mode, a.temperature, a.seed_base + i)
                r.update(temperature=a.temperature, seed=(a.seed_base + i) if a.temperature > 0 else None)
            except Exception as e:
                r = None; err.append("%d: %s" % (i, str(e)[:160]))
            with lock:
                if r: res[i] = r
    ts = [threading.Thread(target=worker) for _ in range(max(1, a.conc))]
    for t in ts: t.start()
    for t in ts: t.join()
    with open(a.out, "w") as f:
        for i in sorted(res): f.write(json.dumps(res[i], ensure_ascii=False) + "\n")
    s = dict(answered=len(res), of=len(gs.PROMPTS), conc=a.conc, temperature=a.temperature, seed_base=a.seed_base,
             max_tokens=a.max_tokens, endpoint=mode[0], tokens=sum((r.get("tokens") or 0) for r in res.values()),
             secs=round(time.time() - t0, 1), errors=err[:5])
    print("SET " + json.dumps(s), flush=True)
    return 0 if len(res) == len(gs.PROMPTS) else 1


def run(a):
    gs, mb = load(a.gs, "greedyset_ro"), load(a.mb, "mb_ro")
    url = a.url.rstrip("/")
    mb.URL = url
    model = gs.model_id(url, 30)
    mb.MODEL = model
    n_pr = len(gs.PROMPTS)
    stop, recs = threading.Event(), []
    t_start = time.time()
    th = mb.closed_loop(a.conc, 0, a.ctx, a.gen, stop, recs)
    time.sleep(a.warm)
    m_first = mb.metrics()
    mode, out, errs = ["completions"], [], []
    g = lambda m0, m1, k: int(m1.get(k, 0) - m0.get(k, 0))  # noqa: E731
    for i in range(n_pr):
        q = f"# pf {time.time_ns()}\n" + mb.text(int(a.pf_tok * 3.9), 777) + "\n\nSummarise the code above in one sentence."
        pr, pf = {}, {}

        def probe():
            pr["t0"] = time.time()
            try:
                pr.update(ask_t(gs, url, model, i, a.max_tokens, a.timeout, mode, a.temperature, a.seed_base + i))
            except Exception as e:
                pr["err"] = str(e)[:200]
            pr["t1"] = time.time()

        def prefill():
            try:
                r = mb.stream([{"role": "user", "content": q}], 1, effort="none")
                pf.update(t0=r["t0"], ttft=r["ttft"], prompt=r["prompt"], cached=r["cached"])
            except Exception as e:
                pf["err"] = str(e)[:200]
            pf["t1"] = time.time()

        busy0 = sum(1 for r in recs if r.get("err"))
        m0 = mb.metrics()
        tp = threading.Thread(target=probe)
        tp.start()
        tf = None
        if a.pf_tok > 0:
            time.sleep(a.offset)
            tf = threading.Thread(target=prefill)
            tf.start()
        tp.join()
        m1 = mb.metrics()
        if tf:
            tf.join()
        if pr.get("err") or pf.get("err"):
            errs.append("p%d: %s" % (i, pr.get("err") or "prefill " + str(pf.get("err"))))
        rec = {k: pr.get(k) for k in ("i", "endpoint", "text", "finish", "tokens")}
        if "reasoning" in pr:
            rec["reasoning"] = pr["reasoning"]
        rec["i"] = i
        rec.update(temperature=a.temperature, seed=(a.seed_base + i) if a.temperature > 0 else None)
        p0, p1 = pr["t0"], pr["t1"]
        f0 = pf.get("t0")
        f1 = (f0 + pf["ttft"]) if (f0 and pf.get("ttft")) else pf.get("t1")
        ov = max(0.0, min(p1, f1) - max(p0, f0)) if f0 and f1 else 0.0
        rec.update(probe_s=round(p1 - p0, 2), pf_off=round(f0 - p0, 2) if f0 else None, pf_ttft=round(pf["ttft"], 2) if pf.get("ttft") else None,
                   pf_prompt=pf.get("prompt"), pf_cached=pf.get("cached"), ov_s=round(ov, 2), ov_frac=round(ov / max(1e-9, p1 - p0), 3),
                   steps=g(m0, m1, "radiance:engine_steps_total"), mixed=g(m0, m1, "radiance:engine_steps_mixed_total"),
                   prefill_steps=g(m0, m1, "radiance:engine_steps_prefill_total"), c4_errors=sum(1 for r in recs if r.get("err")) - busy0)
        out.append(rec)
        print("PROBE p%d: %s tok %s, %.1f s | prefill +%s s, %s tok, cached %s, TTFT %s s, in flight %.0f %% of the probe | steps %d, mixed %d"
              % (i, rec["tokens"], rec["finish"], rec["probe_s"], rec["pf_off"], rec["pf_prompt"], rec["pf_cached"], rec["pf_ttft"],
                 100 * rec["ov_frac"], rec["steps"], rec["mixed"]), flush=True)
    m_last = mb.metrics()
    t_end = time.time()
    stop.set()
    for t in th:
        t.join(timeout=600)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if r.get("text") is not None]
    c4ok = [r for r in recs if not r.get("err") and t_start <= r["t0"] < t_end]
    c4err = [r for r in recs if r.get("err")]
    pft = [r["pf_ttft"] for r in out if r.get("pf_ttft")]
    ovf = [r["ov_frac"] for r in out]
    mx = [r["mixed"] for r in out]
    cold = sum(1 for r in out if r.get("pf_cached") == 0)
    d = mb.mdelta(m_first, m_last)
    s = dict(answered=len(ok), of=n_pr, max_tokens=a.max_tokens, offset_s=a.offset, pf_tok_asked=a.pf_tok, conc=a.conc,
             temperature=a.temperature, seed_base=a.seed_base,
             pf_prompt_mean=int(statistics.mean([r["pf_prompt"] for r in out if r.get("pf_prompt")] or [0])), pf_cold=cold,
             pf_ttft_mean=round(statistics.mean(pft), 2) if pft else None, pf_ttft_min=min(pft) if pft else None, pf_ttft_max=max(pft) if pft else None,
             ov_frac_min=min(ovf) if ovf else None, ov_frac_median=round(statistics.median(ovf), 3) if ovf else None,
             mixed_total=sum(mx), mixed_min=min(mx) if mx else None, mixed_median=statistics.median(mx) if mx else None,
             probes_without_mixed=[r["i"] for r in out if not r["mixed"]],
             probe_s_mean=round(statistics.mean([r["probe_s"] for r in out]), 2) if out else None,
             c4_requests_done=len(c4ok), c4_errors=len(c4err), c4_err_first=(c4err[0].get("err") if c4err else None),
             window_s=round(t_end - t_start, 1), window=d, errors=errs[:5])
    print("LOADID " + json.dumps(s), flush=True)
    return 0 if len(ok) == n_pr and (cold == n_pr or a.pf_tok <= 0) and not c4err and not errs else 1


def selftest(a):
    gs, mb = load(a.gs, "greedyset_ro"), load(a.mb, "mb_ro")
    bad = []
    if len(getattr(gs, "PROMPTS", [])) != 20:
        bad.append("greedyset.PROMPTS != 20")
    for m, names in ((gs, ("ask", "model_id", "cmp")), (mb, ("closed_loop", "stream", "text", "metrics", "mdelta"))):
        bad += ["%s.%s missing" % (m.__name__, n) for n in names if not callable(getattr(m, n, None))]
    for (x, y), want in ((("abc", "abc"), None), (("abc", "abd"), 2), (("ab", "abc"), 2), (([1, 2, 3], [1, 2, 4]), 2), (([], [5]), 0)):
        if first_diff(x, y) != want:
            bad.append("first_diff(%r, %r) = %r, want %r" % (x, y, first_diff(x, y), want))
    seen = []
    real_post = gs.post
    gs.post = lambda url, path, body, timeout: (seen.append((path, json.dumps(body, sort_keys=True))) or
                                                {"choices": [{"text": "x", "finish_reason": "length"}], "usage": {"completion_tokens": 1}})
    try:
        gs.ask("http://stub", "m", 3, 192, 5, ["completions"])
        ask_t(gs, "http://stub", "m", 3, 192, 5, ["completions"], 0.0, 1003)
        ask_t(gs, "http://stub", "m", 3, 192, 5, ["completions"], 1.0, 1003)
    finally:
        gs.post = real_post
    if len(seen) != 3 or seen[0] != seen[1]:
        bad.append("ask_t at T 0 is not greedyset.ask's body (%r vs %r)" % (seen[0] if seen else None, seen[1] if len(seen) > 1 else None))
    else:
        b1 = json.loads(seen[2][1])
        if not (b1.get("temperature") == 1.0 and b1.get("seed") == 1003 and b1.get("top_p") == 1 and b1.get("prompt") == gs.PROMPTS[3]):
            bad.append("ask_t at T 1: %r" % b1)
    q = mb.text(int(32768 * 3.9), 777)
    if len(q) < 32768 * 3.9 * 0.99:
        bad.append("mb.text gave %d chars" % len(q))
    print("SELFTEST " + ("OK: greedyset.py %d prompts, mb.py helpers present, diff logic, ask_t T 0 body = greedyset.ask's, T 1 + seed, pf text %d chars" % (len(gs.PROMPTS), len(q))
                         if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "set", "cmp", "selftest"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="")
    ap.add_argument("--gs")
    ap.add_argument("--mb")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--pf-tok", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed-base", type=int, default=1000)
    a = ap.parse_args()
    if a.mode == "cmp":
        a.a, a.b = a.files[:2]
        sys.exit(cmp(a))
    if a.mode == "set":
        sys.exit(set_run(a))
    sys.exit(run(a) if a.mode == "run" else selftest(a))
