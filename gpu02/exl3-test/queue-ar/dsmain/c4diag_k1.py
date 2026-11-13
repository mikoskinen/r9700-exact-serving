#!/usr/bin/env python3
"""c4diag_k1.py (10.11.2026, queue-ar dsmain, ds41-k1 agent; item 57-p4zyzzzzz-ds-n-attnfold's c4-floor phase) -- the c4
greedy set (queue/tools/greedyset.py's 20 prompts, greedy, max_tokens 192, four in flight) with every prompt PREFILLED FIRST,
so that, where the server keeps the prompts' prefix, the c4 run's steps are decode rows only (no prompt chunk shares a step
with another request's decode rows):

  c4diag_k1.py URL OUT.jsonl --gs PATH/greedyset.py [--conc 4] [--max-tokens 192] [--timeout 300]

  phase 1  each prompt alone (one request in flight), max_tokens 1: a prefill-only step a prompt (its one token comes from
           the prefill's logits, no decode step)
  phase 2  the same 20 requests as greedyset.py --conc 4 (same prompts, same parameters), four in flight; each answer's
           usage is kept: cached = prompt_tokens_details.cached_tokens (or cached_tokens) against prompt_tokens

OUT has greedyset.py's rows {i, endpoint, text, finish, tokens} (+ prompt_tokens, cached, p1_cached), so greedyset.py --cmp
compares it with any c4 / c1 file. Prints one "C4DIAG {json}" line: answered, the phase-2 cached counts (min / max of
prompt_tokens - cached: 0 or 1 = the prefix was kept, the continuation's prompt work is at most its last token), and the
requests whose prefix was not kept. Standard library only; greedyset.py is loaded from --gs (its PROMPTS and ask())."""
import importlib.util, json, sys, threading, urllib.request


def load_gs(path):
    spec = importlib.util.spec_from_file_location("greedyset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cached(u):
    if not u:
        return None
    d = u.get("prompt_tokens_details") or {}
    return d.get("cached_tokens", u.get("cached_tokens"))


def complete(gs, url, model, p, mt, timeout):
    d = gs.post(url, "/v1/completions", {"model": model, "prompt": p, "max_tokens": mt, "temperature": 0, "top_p": 1,
                                         "stream": False}, timeout)
    c = d["choices"][0]
    u = d.get("usage") or {}
    return c, u


def main():
    a = sys.argv[1:]
    def opt(k, dflt):
        return type(dflt)(a[a.index(k) + 1]) if k in a else dflt
    pos = [x for j, x in enumerate(a) if not x.startswith("--") and (j == 0 or not a[j - 1].startswith("--"))]
    if len(pos) < 2 or "--gs" not in a:
        print(__doc__); return 2
    url, out = pos[0].rstrip("/"), pos[1]
    gs = load_gs(opt("--gs", ""))
    conc, mt, timeout = opt("--conc", 4), opt("--max-tokens", 192), opt("--timeout", 300)
    model = gs.model_id(url, 10)
    n = len(gs.PROMPTS)
    p1, err = {}, []
    for i, p in enumerate(gs.PROMPTS):   # phase 1: one at a time
        try:
            _, u = complete(gs, url, model, p, 1, timeout)
            p1[i] = {"prompt_tokens": u.get("prompt_tokens"), "cached": cached(u)}
        except Exception as e:
            err.append("p1 %d: %s" % (i, e))
    res, lock, nxt = {}, threading.Lock(), [0]
    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= n: return
            try:
                c, u = complete(gs, url, model, gs.PROMPTS[i], mt, timeout)
                r = {"i": i, "endpoint": "completions", "text": c.get("text"), "finish": c.get("finish_reason"),
                     "tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"), "cached": cached(u),
                     "p1_cached": (p1.get(i) or {}).get("cached")}
            except Exception as e:
                r = None; err.append("p2 %d: %s" % (i, e))
            with lock:
                if r: res[i] = r
    ts = [threading.Thread(target=worker) for _ in range(max(1, conc))]
    for t in ts: t.start()
    for t in ts: t.join()
    with open(out, "w") as f:
        for i in sorted(res): f.write(json.dumps(res[i], ensure_ascii=False) + "\n")
    gap = [r["prompt_tokens"] - r["cached"] for r in res.values()
           if isinstance(r.get("prompt_tokens"), int) and isinstance(r.get("cached"), int)]
    unknown = sum(1 for r in res.values() if not isinstance(r.get("cached"), int))
    not_kept = sorted(r["i"] for r in res.values()
                      if isinstance(r.get("prompt_tokens"), int) and isinstance(r.get("cached"), int)
                      and r["prompt_tokens"] - r["cached"] > 1)
    s = {"answered": len(res), "of": n, "conc": conc, "max_tokens": mt, "phase1": len(p1),
         "uncached_min": min(gap) if gap else None, "uncached_max": max(gap) if gap else None,
         "cached_unknown": unknown, "prefix_not_kept": not_kept,
         "prefix_kept": bool(gap) and unknown == 0 and not not_kept, "errors": err[:8]}
    print("C4DIAG " + json.dumps(s))
    return 0 if len(res) == n else 1


if __name__ == "__main__":
    sys.exit(main())
