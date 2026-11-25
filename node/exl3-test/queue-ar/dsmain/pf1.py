#!/usr/bin/env python3
"""pf1.py TAG DEPTH N [OFFSET] (10.11.2026, queue-ar/dsmain, ds41-pipe agent; for dsm-pipeprof.sh) -- N fresh prefills of about
DEPTH tokens back to back on the DS41 server (:8090, DS41_URL), each answered with ONE token (max_tokens 1, temperature 0,
thinking off, cache_prompt off, a nonce first): every step the engine runs for them is a lone prefill step -- no decode
step follows a prompt, so a --profile-ops window that brackets them holds prefill steps only. The filler is
ds41_tests.text_chars (stdlib code) at 3.95 chars a token, offset OFFSET + 17 * i. Prints one J line a prompt
(prompt tokens, cached, seconds to the token, tok/s). PROFILED when the server profiles: not a speed."""
import importlib.util, json, os, sys, time
spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
tag, depth, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
off0 = int(sys.argv[4]) if len(sys.argv) > 4 else 300
t.MODEL = t.model_id()
for i in range(n):
    body = t.text_chars(int(depth * 3.95), offset=(off0 + 17 * i) % len(t.SRC))
    q = f"# pf1 {tag} {i} {time.time_ns()}\n" + body + "\n\nSummarise what the code above does in one sentence."
    r = t.chat({"messages": [{"role": "user", "content": q}], "max_tokens": 1, "temperature": 0.0,
                "cache_prompt": False, "reasoning_effort": "none"}, timeout=3600)
    u = r.get("usage") or {}
    pt = u.get("prompt_tokens")
    s = r.get("ttft") or r.get("t_total")
    print("J " + json.dumps(dict(test="pf1", tag=tag, i=i, status=r.get("status"), prompt_tokens=pt, cached=t.cached(u),
                                 s=round(s, 3) if s else None, prefill_tps=round(pt / s, 1) if (pt and s) else None,
                                 finish=r.get("finish"), err=r.get("raw_err"))), flush=True)
