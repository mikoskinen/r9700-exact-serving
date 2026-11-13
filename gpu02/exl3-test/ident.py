#!/usr/bin/env python3
"""ident.py save TAG | cmp TAG_A TAG_B -- temperature-0 identity across restarts: 3 prompts (essay, code, factual), greedy, thinking
off, 256 tokens, full text saved to runs/ident-TAG.json; cmp prints one line with identical count and first differing char."""
import json, os, sys, importlib.util
R = os.path.expanduser("~/gpu/ds41/exl3-test/runs")
if sys.argv[1] == "save":
    spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py")); t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    t.MODEL = t.model_id()
    qs = {"essay": t.PROMPTS["essay"], "code": t.PROMPTS["code"], "factual": "What is the capital of Australia, and in which year did it officially become the capital? Answer in two sentences."}
    out = {}
    for k, q in qs.items():
        r = t.chat({"messages": [{"role": "user", "content": q}], "max_tokens": 256, "temperature": 0.0, "reasoning_effort": "none"})
        out[k] = {"status": r["status"], "content": r.get("content") or r.get("raw_err") or "", "usage": r.get("usage")}
    json.dump(out, open(f"{R}/ident-{sys.argv[2]}.json", "w"), ensure_ascii=False)
    print("IDENT saved " + sys.argv[2] + ": " + ", ".join(f"{k} {len(v['content'])} chars" for k, v in out.items()))
else:
    a, b = (json.load(open(f"{R}/ident-{x}.json")) for x in sys.argv[2:4])
    rows = []
    for k in a:
        x, y = a[k]["content"], b.get(k, {}).get("content", "")
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))
        rows.append((k, x == y, first))
    print("IDENT %s vs %s: %d/%d identical; %s" % (sys.argv[2], sys.argv[3], sum(r[1] for r in rows), len(rows),
          "; ".join(f"{k} {'same' if s else 'first diff at char ' + str(f)}" for k, s, f in rows)))
