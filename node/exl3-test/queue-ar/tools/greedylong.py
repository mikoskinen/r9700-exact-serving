#!/usr/bin/env python3
"""greedylong.py URL OUT.jsonl [--conc N] [--max-tokens 128] [--timeout 600]  |  greedylong.py --cmp A.jsonl B.jsonl

fn4 12.11.2026 (lever BI chunking): greedyset.py with 10 LONG fixed prompts (~2.3k..9.5k tokens, deterministic text: an
LCG over a fixed vocabulary, no host files), so each prompt's prefill runs in several chunks / pipelined pieces and its
tail chunk lands at varied heights -- the case where a prompt's chunk boundaries can move with the load. Same request,
same JSON lines, same --cmp; fn-loadid.py run takes it as --gs. Standard library only."""
import json, sys, threading, urllib.request

_V = ("the system reads each record and writes a short report about its state while the queue holds new work for the next "
      "worker so that every task keeps its order and nothing waits longer than the limit set by the operator during the "
      "night shift when the load is low and the cache stays warm for the morning batch of requests from the clients").split()
def _text(n_words, seed):
    x = seed * 2654435761 % (2 ** 32) or 1
    out = []
    for k in range(n_words):
        x = (1103515245 * x + 12345) % (2 ** 31)
        out.append(_V[x % len(_V)])
        if k % 17 == 16: out.append("item %d." % (k // 17))
    return " ".join(out)
_LEN = [1700, 2150, 2600, 3050, 3500, 3950, 4550, 5200, 6100, 7100]   # words (~1.35 tokens a word)
import os as _os
PROMPTS = ["Here is a log.\n\n" + _text(n, i + 1) + "\n\nQuestion: which item numbers appear in the last two lines above, "
           "and what does the log say happens at night? Answer in two sentences.\nAnswer:" for i, n in enumerate(_LEN)]
if _os.environ.get("GL_N", "").isdigit() and int(_os.environ["GL_N"]) > 0:   # a subset (the first GL_N prompts)
    PROMPTS = PROMPTS[:int(_os.environ["GL_N"])]

def post(url, path, body, timeout):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def model_id(url, timeout):
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=timeout) as r:
            d = json.loads(r.read())
        return (d.get("data") or [{}])[0].get("id") or "default"
    except Exception:
        return "default"

def ask(url, model, i, mt, timeout, mode):
    p = PROMPTS[i]
    if mode[0] != "chat":
        try:
            d = post(url, "/v1/completions", {"model": model, "prompt": p, "max_tokens": mt, "temperature": 0,
                                              "top_p": 1, "stream": False}, timeout)
            c = d["choices"][0]
            return {"i": i, "endpoint": "completions", "text": c.get("text"), "finish": c.get("finish_reason"),
                    "tokens": (d.get("usage") or {}).get("completion_tokens")}
        except urllib.error.HTTPError as e:
            if e.code not in (400, 404, 405, 422, 501): raise
            mode[0] = "chat"
    d = post(url, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": mt,
                                           "temperature": 0, "top_p": 1, "stream": False, "reasoning_effort": "none"},
             timeout)
    c = d["choices"][0]
    m = c.get("message") or {}
    return {"i": i, "endpoint": "chat", "text": m.get("content"), "reasoning": m.get("reasoning_content"),
            "finish": c.get("finish_reason"), "tokens": (d.get("usage") or {}).get("completion_tokens")}

def run(url, out, conc, mt, timeout):
    model = model_id(url, 10)
    res, err, lock, nxt, mode = {}, [], threading.Lock(), [0], ["completions"]
    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= len(PROMPTS): return
            try:
                r = ask(url, model, i, mt, timeout, mode)
            except Exception as e:
                r = None; err.append("%d: %s" % (i, e))
            with lock:
                if r: res[i] = r
    ts = [threading.Thread(target=worker) for _ in range(max(1, conc))]
    for t in ts: t.start()
    for t in ts: t.join()
    with open(out, "w") as f:
        for i in sorted(res): f.write(json.dumps(res[i], ensure_ascii=False) + "\n")
    print("greedylong: %d of %d answered (conc %d, max_tokens %d, endpoint %s)%s" % (len(res), len(PROMPTS), conc, mt, mode[0],
          "; errors: " + "; ".join(err)[:400] if err else ""))
    return 0 if len(res) == len(PROMPTS) else 1

def cmp(a, b):
    def rd(f):
        try: return {d["i"]: d for d in map(json.loads, open(f))}
        except OSError: return {}
    x, y = rd(a), rd(b)
    if not x or not y:
        print("no answers (%d / %d)" % (len(x), len(y))); return 1
    bad = [i for i in sorted(set(x) | set(y)) if (x.get(i, {}).get("text"), x.get(i, {}).get("finish"), x.get(i, {}).get("reasoning"))
           != (y.get(i, {}).get("text"), y.get(i, {}).get("finish"), y.get(i, {}).get("reasoning"))]
    print(("IDENTICAL: %d answers" % len(x)) if not bad else "DIFFERENT: " + " ".join(map(str, bad)) + " (of %d / %d)" % (len(x), len(y)))
    return 0 if not bad else 1

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--cmp":
        sys.exit(cmp(a[1], a[2]))
    def opt(k, d):
        return type(d)(a[a.index(k) + 1]) if k in a else d
    pos = [x for j, x in enumerate(a) if not x.startswith("--") and (j == 0 or not a[j - 1].startswith("--"))]
    if len(pos) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(run(pos[0].rstrip("/"), pos[1], opt("--conc", 1), opt("--max-tokens", 128), opt("--timeout", 600)))
