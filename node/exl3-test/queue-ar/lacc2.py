#!/usr/bin/env python3
"""lacc2.py STAGE OUTDIR [DEPTH=32768] [TOKENS=256] (9.11.2026, queue-ar; item 113b) -- MTP draft acceptance on a COLD
prompt against the same prompt resumed from the prefix cache, on the test server (DS41_URL, default :8090). One request a
call, greedy (temperature 0, thinking off, ignore_eos so every stage decodes TOKENS tokens); around it the engine's own
counters from /metrics, so the line is that request's alone (run one stage at a time, nothing else on the server):
  cold     a fresh prompt: "# lacc2 <nonce>" + ~DEPTH tokens of ds41_tests' stdlib-code filler + task A
  suffix   the same prompt with task B instead of A (a prefix hit up to the task)
  turn     cold's conversation + its answer + a follow-up (a hit over the prompt AND the answer, as a coding agent's turn)
  turn2    turn's conversation + its answer + another follow-up
The nonce (LACC2_NONCE, else OUTDIR's name) keeps every server's first prompt cold. Prints one line:
  LACC2 {"stage", "prompt_tokens", "cached", "ttft", "tokens", "drafts", "accepted", "acceptance", "decode_steps",
         "tokens_per_step", "preempted", "status"}
and writes the answer to OUTDIR/lacc2-<stage>.txt (turn / turn2 read the earlier answers from there)."""
import importlib.util, json, os, sys, urllib.request

spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
stage, out = sys.argv[1], sys.argv[2]
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
ntok = int(sys.argv[4]) if len(sys.argv) > 4 else 256
URL = os.environ.get("DS41_URL", "http://127.0.0.1:8090").rstrip("/")
nonce = os.environ.get("LACC2_NONCE") or os.path.basename(os.path.normpath(out))
KEYS = ("radiance:draft_tokens_total", "radiance:draft_accepted_total", "radiance:engine_steps_decode_total",
        "radiance:engine_steps_mixed_total", "radiance:decode_tokens_total", "radiance:num_requests_preempted",
        "radiance:attn_cached_tokens_total")


def metrics():
    m = {k: 0.0 for k in KEYS}
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if not line or line[0] == "#":
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in m:
                    try:
                        m[name] += float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
    except Exception:  # noqa: BLE001
        pass
    return m


t.MODEL = t.model_id()
body = t.text_chars(int(depth * 3.95), offset=2400)
head = f"# lacc2 {nonce}\n"
taskA = ("\n\nYou have read the source files above. Explain in detail what the three largest functions do and how "
         "they could be simplified, with example code.")
taskB = ("\n\nYou have read the source files above. List every class defined above with a one-paragraph description "
         "of its responsibilities and its main methods.")
follow1 = "Good. Now rewrite the most complex function you described as a cleaner version, with comments."
follow2 = "Now write unit tests for the rewritten function, covering the edge cases you can think of."


def rd(name):
    with open(os.path.join(out, f"lacc2-{name}.txt")) as f:
        return f.read()


if stage == "cold":
    msgs = [{"role": "user", "content": head + body + taskA}]
elif stage == "suffix":
    msgs = [{"role": "user", "content": head + body + taskB}]
elif stage == "turn":
    msgs = [{"role": "user", "content": head + body + taskA}, {"role": "assistant", "content": rd("cold")},
            {"role": "user", "content": follow1}]
elif stage == "turn2":
    msgs = [{"role": "user", "content": head + body + taskA}, {"role": "assistant", "content": rd("cold")},
            {"role": "user", "content": follow1}, {"role": "assistant", "content": rd("turn")},
            {"role": "user", "content": follow2}]
else:
    sys.exit(f"lacc2: unknown stage {stage}")

m0 = metrics()
r = t.chat({"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True},
           timeout=1800)
m1 = metrics()
d = {k: m1[k] - m0[k] for k in KEYS}
u = r.get("usage") or {}
text = r.get("content") or ""
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f"lacc2-{stage}.txt"), "w") as f:
    f.write(text)
drafts, acc = d["radiance:draft_tokens_total"], d["radiance:draft_accepted_total"]
steps = d["radiance:engine_steps_decode_total"] + d["radiance:engine_steps_mixed_total"]
info = dict(stage=stage, prompt_tokens=u.get("prompt_tokens"), cached=t.cached(u), ttft=round(r.get("ttft") or 0, 2),
            tokens=u.get("completion_tokens"), drafts=drafts, accepted=acc,
            acceptance=round(acc / drafts, 3) if drafts else None, decode_steps=steps,
            tokens_per_step=round(d["radiance:decode_tokens_total"] / steps, 2) if steps else None,
            preempted=d["radiance:num_requests_preempted"], status=r.get("status"))
print("LACC2 " + json.dumps(info), flush=True)
sys.exit(0 if r.get("status") == 200 else 1)
