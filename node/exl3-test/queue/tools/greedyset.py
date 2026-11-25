#!/usr/bin/env python3
"""greedyset.py URL OUT.jsonl [--conc N] [--max-tokens 192] [--timeout 300]  |  greedyset.py --cmp A.jsonl B.jsonl

20 fixed prompts, greedy (temperature 0), each answer saved as one JSON line {i, endpoint, text, finish, tokens}, for a
byte-for-byte A/B of an engine switch that must not change the output (83-acqrule-ab: RADIANCE_AQL_ACQ off vs rule).
--conc 1: one request at a time; --conc 4: four in flight (the c4 batch shape), the same 20 prompts.
Uses /v1/completions (raw prompt, no chat template or reasoning); falls back to /v1/chat/completions with
reasoning_effort none if the first is refused. Standard library only. Exit 0 = all 20 answered.
--cmp: prints "IDENTICAL: 20 answers" or "DIFFERENT: i i ..." (text or finish differs), exit 0 / 1."""
import json, sys, threading, urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n",
    "The three primary colors of light are",
    "Q: What is 17 * 23? Show the steps.\nA:",
    "Translate to Finnish: 'The weather is cold today, but the sun is shining.'\nFinnish:",
    "List five prime numbers greater than 100:\n1.",
    "def quicksort(arr):\n    \"\"\"Sort a list with quicksort.\"\"\"\n",
    "Explain in two sentences why the sky is blue.\n",
    "A haiku about autumn rain:\n",
    "SELECT name, COUNT(*) FROM orders",
    "The capital of Australia is",
    "// C++: a function that reverses a std::string in place\n#include <string>\n",
    "Summarize the plot of Romeo and Juliet in one paragraph.\n",
    "Q: If a train travels 120 km in 1.5 hours, what is its average speed?\nA:",
    "#!/bin/bash\n# Count the lines of every .txt file in the current directory\n",
    "The difference between TCP and UDP is",
    "Write a JSON object describing a book with title, author and year:\n",
    "In mathematics, a group is",
    "fn main() {\n    // Rust: print the numbers 1 to 10\n",
    "Three tips for writing clear technical documentation:\n1.",
    "The derivative of sin(x) * x^2 is",
]

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
    print("greedyset: %d of %d answered (conc %d, max_tokens %d, endpoint %s)%s" % (len(res), len(PROMPTS), conc, mt, mode[0],
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
    sys.exit(run(pos[0].rstrip("/"), pos[1], opt("--conc", 1), opt("--max-tokens", 192), opt("--timeout", 300)))
