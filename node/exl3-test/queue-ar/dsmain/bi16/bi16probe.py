#!/usr/bin/env python3
"""bi16probe.py (12.11.2026, queue-ar/dsmain/bi16, ds-main-port bi16 builder) -- the driver of dsm-bi16.sh (item
57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16, tag "RESULT xbi16 "): 1-2 greedy probes ALONE and UNDER a pure decode load of N
closed-loop streams, so lever BI's tracer (RADIANCE_DS41_BITRACE) holds the probe's rows alone and in steps with > 48 decode
rows. Made from stack4/loadid.py (ee7d9663: its helper loading, run() shape, cmp()'s first-difference logic); greedyset.py
(the 20 prompts and its exact request, ask()) and mb.py (closed_loop streams, metrics) are loaded READ ONLY with importlib,
nothing of theirs changes. Differences to loadid.py run: only the probes named by --idx (not all 20), no newcomer prefill
beside the probe (pure decode load; the streams' own turnover prefills still ride along), per-probe wall times kept for
the step-log count.

  bi16probe.py tok --url URL --gs GREEDYSET.py --idx 9,8 --out PROBES.json
      each probe prompt's token ids through the server's POST /tokenize {"content": prompt} (the trace analyzer finds the
      probe's sequences by them). Prints "TOK {json}".
  bi16probe.py alone --url URL --gs GREEDYSET.py --idx 9,8 --out OUT.jsonl [--max-tokens 192] [--timeout 300]
      the probes one at a time, nothing else in flight (greedyset.py's ask(): /v1/completions, temperature 0, top_p 1).
      OUT = greedyset.py's line shape {i, endpoint, text, finish, tokens} + t0 / t1. Last line "ALONE {json}".
  bi16probe.py load --url URL --gs GREEDYSET.py --mb MB.py --idx 9,8 --out OUT.jsonl [--conc 15] [--warm 10] [--ctx 1000]
                    [--gen 512] [--gap 1] [--max-tokens 192] [--timeout 300]
      --conc mb.py closed_loop streams (ctx 1000, gen 512, reasoning_effort medium, server-default sampling: the lid16
      streams), --warm s later each probe ONE AT A TIME (the same request as alone), --gap s between probes; then the
      streams stop (each finishes its request: the next phase must run alone). Per probe: t0 / t1, engine steps and MIXED
      steps (/metrics deltas). Last line "LOADP {json}". Exit 0 = every probe answered, no stream error.
  bi16probe.py cmp A.jsonl B.jsonl --idx 9,8 [--json OUT.json] [--label L] [--url URL]
      byte for byte per named probe (text, finish, reasoning: greedyset.py --cmp's key), the first differing character and
      token (POST /tokenize when --url is given). Prints "CMP {json}" and "CMPLINE ...". Exit 0 = identical.
  bi16probe.py selftest --gs GREEDYSET.py --mb MB.py
      offline: both helpers load and have what this driver calls; the diff logic on fixed pairs; --idx parsing. No server.
Standard library only.
"""
import argparse, importlib.util, json, sys, threading, time, urllib.request

sys.dont_write_bytecode = True   # the helpers are loaded read only: no __pycache__ beside them


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def idx_list(s, n):
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k < 0 or k >= n or k in out:
            raise ValueError("--idx: %r is not a distinct prompt index 0..%d" % (x, n - 1))
        out.append(k)
    if not out:
        raise ValueError("--idx names no prompt")
    return out


def first_diff(a, b):
    """index of the first differing element of two sequences; None when equal (loadid.py's)"""
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


def cmd_tok(a):
    gs = load(a.gs, "greedyset_ro")
    ids = idx_list(a.idx, len(gs.PROMPTS))
    url = a.url.rstrip("/")
    out = {"probes": []}
    for i in ids:
        t = tokenize(url, gs.PROMPTS[i])
        out["probes"].append({"i": i, "prompt": gs.PROMPTS[i], "ids": t})
    with open(a.out, "w") as f:
        json.dump(out, f)
    ok = all(p["ids"] for p in out["probes"])
    print("TOK " + json.dumps({"ok": ok, "probes": [{"i": p["i"], "n": len(p["ids"] or [])} for p in out["probes"]]}))
    return 0 if ok else 1


def ask_one(gs, url, model, i, mt, timeout, mode):
    r = {"i": i}
    t0 = time.time()
    try:
        r.update(gs.ask(url, model, i, mt, timeout, mode))
    except Exception as e:
        r["err"] = str(e)[:200]
    r["t0"], r["t1"] = t0, time.time()
    return r


def rec_of(pr):
    rec = {k: pr.get(k) for k in ("i", "endpoint", "text", "finish", "tokens")}
    if "reasoning" in pr:
        rec["reasoning"] = pr["reasoning"]
    rec.update(t0=round(pr["t0"], 3), t1=round(pr["t1"], 3), probe_s=round(pr["t1"] - pr["t0"], 2))
    if pr.get("err"):
        rec["err"] = pr["err"]
    return rec


def cmd_alone(a):
    gs = load(a.gs, "greedyset_ro")
    ids = idx_list(a.idx, len(gs.PROMPTS))
    url = a.url.rstrip("/")
    model = gs.model_id(url, 30)
    mode, out = ["completions"], []
    for i in ids:
        pr = ask_one(gs, url, model, i, a.max_tokens, a.timeout, mode)
        out.append(rec_of(pr))
        print("PROBE p%d alone: %s tok %s, %.1f s%s" % (i, pr.get("tokens"), pr.get("finish"), pr["t1"] - pr["t0"],
                                                        " ERR " + pr["err"] if pr.get("err") else ""), flush=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if r.get("text") is not None]
    print("ALONE " + json.dumps(dict(answered=len(ok), of=len(ids), idx=ids, max_tokens=a.max_tokens,
                                     probes=[{"i": r["i"], "t0": r["t0"], "t1": r["t1"], "tokens": r.get("tokens")} for r in out])),
          flush=True)
    return 0 if len(ok) == len(ids) else 1


def cmd_load(a):
    gs, mb = load(a.gs, "greedyset_ro"), load(a.mb, "mb_ro")
    ids = idx_list(a.idx, len(gs.PROMPTS))
    url = a.url.rstrip("/")
    mb.URL = url
    model = gs.model_id(url, 30)
    mb.MODEL = model
    stop, recs = threading.Event(), []
    t_start = time.time()
    th = mb.closed_loop(a.conc, 0, a.ctx, a.gen, stop, recs)
    time.sleep(a.warm)
    m_first = mb.metrics()
    mode, out = ["completions"], []
    g = lambda m0, m1, k: int(m1.get(k, 0) - m0.get(k, 0))  # noqa: E731
    for n, i in enumerate(ids):
        if n:
            time.sleep(a.gap)
        busy0 = sum(1 for r in recs if r.get("err"))
        m0 = mb.metrics()
        pr = ask_one(gs, url, model, i, a.max_tokens, a.timeout, mode)
        m1 = mb.metrics()
        rec = rec_of(pr)
        rec.update(steps=g(m0, m1, "radiance:engine_steps_total"), mixed=g(m0, m1, "radiance:engine_steps_mixed_total"),
                   decode_steps=g(m0, m1, "radiance:engine_steps_decode_total"),
                   stream_errors=sum(1 for r in recs if r.get("err")) - busy0)
        out.append(rec)
        print("PROBE p%d loaded: %s tok %s, %.1f s | steps %d (decode %d, mixed %d)%s" % (
            i, rec["tokens"], rec["finish"], rec["probe_s"], rec["steps"], rec["decode_steps"], rec["mixed"],
            " ERR " + pr["err"] if pr.get("err") else ""), flush=True)
    m_last = mb.metrics()
    t_end = time.time()
    stop.set()
    for t in th:
        t.join(timeout=900)
    t_stop = time.time()
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if r.get("text") is not None and not r.get("err")]
    sok = [r for r in recs if not r.get("err")]
    serr = [r for r in recs if r.get("err")]
    alive = sum(1 for t in th if t.is_alive())
    d = mb.mdelta(m_first, m_last)
    s = dict(answered=len(ok), of=len(ids), idx=ids, conc=a.conc, ctx=a.ctx, gen=a.gen, warm_s=a.warm, max_tokens=a.max_tokens,
             probes=[{"i": r["i"], "t0": r["t0"], "t1": r["t1"], "tokens": r.get("tokens"), "steps": r["steps"],
                      "decode_steps": r["decode_steps"], "mixed": r["mixed"]} for r in out],
             streams_done=len(sok), stream_errors=len(serr), stream_err_first=(serr[0].get("err") if serr else None),
             streams_alive_after_join=alive, window_s=round(t_end - t_start, 1), drain_s=round(t_stop - t_end, 1),
             t_start=round(t_start, 3), t_end=round(t_end, 3), t_stop=round(t_stop, 3), window=d)
    print("LOADP " + json.dumps(s), flush=True)
    return 0 if len(ok) == len(ids) and not serr and not alive else 1


def cmd_cmp(a):
    x, y = rd(a.files[0]), rd(a.files[1])
    ids = [int(v) for v in str(a.idx).split(",") if v.strip()]
    key = lambda d: (d.get("text"), d.get("finish"), d.get("reasoning"))  # noqa: E731 (greedyset.py --cmp's key)
    diff, missing = [], [i for i in ids if i not in x or i not in y]
    for i in ids:
        if i in missing or key(x[i]) == key(y[i]):
            continue
        ta, tb = x[i].get("text") or "", y[i].get("text") or ""
        c = first_diff(ta, tb)
        url = (a.url or "").rstrip("/")
        ka, kb = (tokenize(url, ta), tokenize(url, tb)) if url else (None, None)
        k = first_diff(ka, kb) if ka is not None and kb is not None else None
        e = dict(i=i, char=c, tok=k, len_a=len(ta), len_b=len(tb), fin_a=x[i].get("finish"), fin_b=y[i].get("finish"))
        if c is not None:
            e.update(at_a=ta[max(0, c - 12):c + 24], at_b=tb[max(0, c - 12):c + 24])
        elif x[i].get("reasoning") != y[i].get("reasoning"):
            e["what"] = "reasoning"
        else:
            e["what"] = "finish"
        diff.append(e)
    r = dict(label=a.label, a=a.files[0], b=a.files[1], idx=ids, n=len(ids), missing=missing, n_diff=len(diff),
             identical=bool(x) and bool(y) and not diff and not missing, diff=diff)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, ensure_ascii=False)
    print("CMP " + json.dumps(r, ensure_ascii=False))
    if missing:
        line = "not compared: %d of %d probes missing %s" % (len(missing), len(ids), missing)
    elif r["identical"]:
        line = "IDENTICAL: %d of %d probes" % (len(ids), len(ids))
    else:
        line = "%d of %d probes differ: %s" % (len(diff), len(ids), ", ".join(
            "p%d char %s tok %s%s" % (e["i"], e["char"], "-" if e["tok"] is None else e["tok"], " (%s)" % e["what"] if e.get("what") else "")
            for e in diff))
    print("CMPLINE " + line)
    return 0 if r["identical"] else 1


def cmd_selftest(a):
    gs, mb = load(a.gs, "greedyset_ro"), load(a.mb, "mb_ro")
    bad = []
    if len(getattr(gs, "PROMPTS", [])) != 20:
        bad.append("greedyset.PROMPTS != 20")
    for m, names in ((gs, ("ask", "model_id")), (mb, ("closed_loop", "metrics", "mdelta", "stream", "text"))):
        bad += ["%s.%s missing" % (m.__name__, n) for n in names if not callable(getattr(m, n, None))]
    for (x, y), want in ((("abc", "abc"), None), (("abc", "abd"), 2), (("ab", "abc"), 2), (([1, 2, 3], [1, 2, 4]), 2)):
        if first_diff(x, y) != want:
            bad.append("first_diff(%r, %r) = %r, want %r" % (x, y, first_diff(x, y), want))
    try:
        if idx_list("9,8", 20) != [9, 8]:
            bad.append("idx_list")
    except ValueError as e:
        bad.append(str(e))
    for s in ("20", "9,9", ""):
        try:
            idx_list(s, 20)
            bad.append("idx_list(%r) accepted" % s)
        except ValueError:
            pass
    import os, tempfile
    d = tempfile.mkdtemp(prefix="bi16probe-")
    fa, fb = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
    with open(fa, "w") as f:
        f.write(json.dumps({"i": 9, "text": " Canberra.", "finish": "length"}) + "\n" + json.dumps({"i": 8, "text": " x", "finish": "stop"}) + "\n")
    with open(fb, "w") as f:
        f.write(json.dumps({"i": 9, "text": " Canberra!", "finish": "length"}) + "\n" + json.dumps({"i": 8, "text": " x", "finish": "stop"}) + "\n")
    ns = argparse.Namespace(files=[fa, fb], idx="9,8", json=None, label="t", url="")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_cmp(ns)
    if rc != 1 or "1 of 2 probes differ: p9 char 9" not in buf.getvalue():
        bad.append("cmp: %r" % buf.getvalue()[-200:])
    for fn in (fa, fb):
        os.unlink(fn)
    os.rmdir(d)
    print("SELFTEST " + ("OK: greedyset.py %d prompts, mb.py helpers present, diff logic, --idx parsing, cmp" % len(gs.PROMPTS)
                         if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tok", "alone", "load", "cmp", "selftest"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="")
    ap.add_argument("--gs")
    ap.add_argument("--mb")
    ap.add_argument("--idx", default="9,8")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    ap.add_argument("--conc", type=int, default=15)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--gap", type=float, default=1)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    fn = {"tok": cmd_tok, "alone": cmd_alone, "load": cmd_load, "cmp": cmd_cmp, "selftest": cmd_selftest}[a.mode]
    if a.mode == "cmp" and len(a.files) != 2:
        ap.error("cmp needs A.jsonl B.jsonl")
    sys.exit(fn(a))
