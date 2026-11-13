#!/usr/bin/env python3
"""loadid.py (11.11.2026, queue-ar/dsmain, ds-main-port) -- the driver of dsm-loadid.sh (item 57-p4zyzzzzz-ds-iej-z0ad-loadid,
tag xlid): IDENTITY UNDER LOAD. Reuses greedyset.py (its 20 prompts and its exact request, ask()) and mb.py (its c4 closed-loop
streams, its fresh ~32k prefill prompt, /metrics) READ ONLY, loaded from their paths with importlib; nothing of theirs changes.

  loadid.py run --url URL --gs GREEDYSET.py --mb MB.py --out OUT.jsonl [--conc 4] [--warm 10] [--offset 0.5] [--pf-tok 32768]
                [--max-tokens 192] [--ctx 1000] [--gen 512] [--timeout 300]
      mb c4 streams (mb.py closed_loop: ctx 1000, gen 512, reasoning_effort medium, server-default sampling, as the stack items'
      `mb.py --phases c4`) start, --warm s later the 20 greedy prompts go ONE AT A TIME, each with greedyset.py's own request
      (/v1/completions, temperature 0, top_p 1, --max-tokens); --offset s after each probe is sent, ONE fresh ~--pf-tok prefill
      (mb.py's pf prompt: a time_ns nonce first, so nothing of it is cached; 1 token out, reasoning_effort none) starts, so steps
      with a prefill chunk ride along the probe's decode. The next probe starts when both are done. OUT = greedyset.py's line
      shape {i, endpoint, text, finish, tokens} + per probe: probe seconds, the prefill's offset / TTFT / prompt / cached, the
      seconds and share of the probe's life the prefill was in flight, engine steps and MIXED steps (radiance:engine_steps_
      mixed_total) during the probe. Last line "LOADID {json}". Exit 0 = 20 answered, 20 prefills cold (cached 0), no c4 error.
  loadid.py cmp --url URL A.jsonl B.jsonl [--json OUT.json] [--label L]
      byte-for-byte per prompt (text, finish, reasoning, as greedyset.py --cmp), and for each differing one the first differing
      character and token (both texts through the server's POST /tokenize {"content": text}; token = first index where the two
      id lists differ; "-" when the server cannot tokenize). Prints "CMP {json}" and one "CMPLINE ..." line.
  loadid.py selftest --gs GREEDYSET.py --mb MB.py
      offline: both helpers load, have what this driver calls (20 prompts, ask, model_id, closed_loop, stream, text, metrics,
      mdelta), the diff logic on fixed pairs. No server, writes nothing.
Standard library only.
"""
import argparse, importlib.util, json, statistics, sys, threading, time, urllib.request

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
                pr.update(gs.ask(url, model, i, a.max_tokens, a.timeout, mode))
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
        time.sleep(a.offset)
        tf = threading.Thread(target=prefill)
        tf.start()
        tp.join()
        m1 = mb.metrics()
        tf.join()
        if pr.get("err") or pf.get("err"):
            errs.append("p%d: %s" % (i, pr.get("err") or "prefill " + str(pf.get("err"))))
        rec = {k: pr.get(k) for k in ("i", "endpoint", "text", "finish", "tokens")}
        if "reasoning" in pr:
            rec["reasoning"] = pr["reasoning"]
        rec["i"] = i
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
             pf_prompt_mean=int(statistics.mean([r["pf_prompt"] for r in out if r.get("pf_prompt")] or [0])), pf_cold=cold,
             pf_ttft_mean=round(statistics.mean(pft), 2) if pft else None, pf_ttft_min=min(pft) if pft else None, pf_ttft_max=max(pft) if pft else None,
             ov_frac_min=min(ovf) if ovf else None, ov_frac_median=round(statistics.median(ovf), 3) if ovf else None,
             mixed_total=sum(mx), mixed_min=min(mx) if mx else None, mixed_median=statistics.median(mx) if mx else None,
             probes_without_mixed=[r["i"] for r in out if not r["mixed"]],
             probe_s_mean=round(statistics.mean([r["probe_s"] for r in out]), 2) if out else None,
             c4_requests_done=len(c4ok), c4_errors=len(c4err), c4_err_first=(c4err[0].get("err") if c4err else None),
             window_s=round(t_end - t_start, 1), window=d, errors=errs[:5])
    print("LOADID " + json.dumps(s), flush=True)
    return 0 if len(ok) == n_pr and cold == n_pr and not c4err and not errs else 1


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
    q = mb.text(int(32768 * 3.9), 777)
    if len(q) < 32768 * 3.9 * 0.99:
        bad.append("mb.text gave %d chars" % len(q))
    print("SELFTEST " + ("OK: greedyset.py %d prompts, mb.py helpers present, diff logic, pf text %d chars" % (len(gs.PROMPTS), len(q))
                         if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "cmp", "selftest"))
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
    a = ap.parse_args()
    if a.mode == "cmp":
        a.a, a.b = a.files[:2]
        sys.exit(cmp(a))
    sys.exit(run(a) if a.mode == "run" else selftest(a))
