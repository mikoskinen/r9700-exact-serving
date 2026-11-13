#!/usr/bin/env python3
"""deepq.py OUT [DEPTH] [TOKENS] (7.11.2026, queue-ar; item 79's X79_DEEP step) -- a deep exactness probe on the DS41 server
(:8090, DS41_URL): ONE fixed prompt of about DEPTH tokens (default 135000) -- the stdlib-code filler of ds41_tests.text_chars
at a fixed offset and a fixed chars/token (3.95, no calibration request, no nonce), so every arm and every run sends the
same bytes -- and a free-text task, answered greedily (temperature 0, thinking off) for TOKENS tokens (default 512,
ignore_eos so the length is fixed). Writes OUT (the answer's text) and OUT.json (prompt tokens, cached, TTFT, tokens,
finish). A copy test reproduces its source whatever small numeric differences an arm makes; free text after a long
prompt diverges at the first token whose logits move, so two arms' OUT files are equal only if every step agreed."""
import json, os, sys, importlib.util
spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
out = sys.argv[1]
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 135000
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 512
CPT, OFF = 3.95, 1200
t.MODEL = t.model_id()
body = t.text_chars(int(depth * CPT), offset=OFF)
task = ("\n\nYou have read the source files above. Write a long, detailed technical review of this code: explain what "
        "the main modules do, how they depend on each other, which functions are the most complex and why, and "
        "propose concrete refactorings with example code.")
msgs = [{"role": "user", "content": "# deepq fixed prompt v1\n" + body + task}]
r = t.chat({"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True},
           timeout=7200)
u = r.get("usage") or {}
text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
open(out, "w").write(text)
info = dict(status=r.get("status"), prompt_tokens=u.get("prompt_tokens"), cached=t.cached(u), ttft=round(r.get("ttft") or 0, 2),
            tokens=u.get("completion_tokens"), finish=r.get("finish"), chars=len(text), err=r.get("raw_err"))
json.dump(info, open(out + ".json", "w"))
print("DEEPQ " + json.dumps(info), flush=True)
sys.exit(0 if r.get("status") == 200 else 1)
