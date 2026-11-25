#!/usr/bin/env python3
"""concacc.py N OUTDIR [DEPTH=16384] [TOKENS=256] [TURNS=3] [STAGGER_S=2.0] (9.11.2026, queue-ar; item 113d) -- MTP draft
acceptance under CONCURRENT multi-turn sessions with prefix-cache hits, on the test server (DS41_URL, default :8090). A
compact, deterministic stand-in for the Claude Code replay: N sessions in N threads, session i starting i x STAGGER_S
after the first, so new turns arrive while others decode (mixed prefill + decode steps). Each session: a cold ~DEPTH-token
prompt ("# concacc <nonce> session <i>" + ds41_tests' stdlib-code filler at a per-session offset + a task), then TURNS
warm turns (the conversation so far + its last answer + a follow-up: a prefix hit over everything before), every request
greedy (temperature 0, thinking off) for TOKENS tokens with ignore_eos. Around the whole window the engine's own counters
from /metrics. The nonce: CONCACC_NONCE, else OUTDIR's name. Prints one line:
  CONCACC {"n", "requests", "errors", "prompt_tokens", "cached", "cached_share", "ttft_p50", "ttft_max", "wall_s",
           "drafts", "accepted", "acceptance", "decode_steps", "mixed_steps", "tokens_per_step", "preempt_forced",
           "preempted_gauge_delta"}
Under concurrency a request's own share of the engine's global draft counters is not separable, so the counters are the
window's; the per-request stats (cached tokens, TTFT) go to OUTDIR/concacc-n<N>.json."""
import importlib.util, json, os, statistics, sys, threading, time, urllib.request

spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
N, out = int(sys.argv[1]), sys.argv[2]
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
ntok = int(sys.argv[4]) if len(sys.argv) > 4 else 256
turns = int(sys.argv[5]) if len(sys.argv) > 5 else 3
stagger = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
URL = os.environ.get("DS41_URL", "http://127.0.0.1:8090").rstrip("/")
nonce = os.environ.get("CONCACC_NONCE") or os.path.basename(os.path.normpath(out))
KEYS = ("radiance:draft_tokens_total", "radiance:draft_accepted_total", "radiance:engine_steps_decode_total",
        "radiance:engine_steps_mixed_total", "radiance:decode_tokens_total", "radiance:preempt_forced_total",
        "radiance:num_requests_preempted")
TASK = ("\n\nYou have read the source files above. Explain in detail what the three largest functions do and how they "
        "could be simplified, with example code.")
FOLLOW = ["Good. Now rewrite the most complex function you described as a cleaner version, with comments.",
          "Now write unit tests for the rewritten function, covering the edge cases you can think of.",
          "Summarise the changes you proposed as a short commit message and a list of risks.",
          "Which of the remaining functions would you refactor next, and how?"]


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
results = []
lock = threading.Lock()


def session(i):
    time.sleep(i * stagger)
    body = t.text_chars(int(depth * 3.95), offset=4000 + 977 * i)
    msgs = [{"role": "user", "content": f"# concacc {nonce} session {i}\n" + body + TASK}]
    for k in range(turns + 1):
        r = t.chat({"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none",
                    "ignore_eos": True}, timeout=3600)
        u = r.get("usage") or {}
        rec = dict(session=i, turn=k, status=r.get("status"), prompt_tokens=u.get("prompt_tokens"),
                   cached=t.cached(u), ttft=r.get("ttft"), tokens=u.get("completion_tokens"))
        with lock:
            results.append(rec)
        if r.get("status") != 200:
            return
        msgs = msgs + [{"role": "assistant", "content": r.get("content") or ""},
                       {"role": "user", "content": FOLLOW[k % len(FOLLOW)]}]


m0 = metrics(); t0 = time.monotonic()
th = [threading.Thread(target=session, args=(i,)) for i in range(N)]
for x in th:
    x.start()
for x in th:
    x.join()
wall = time.monotonic() - t0; m1 = metrics()
d = {k: m1[k] - m0[k] for k in KEYS}
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f"concacc-n{N}.json"), "w") as f:
    json.dump(results, f)
ok = [r for r in results if r["status"] == 200]
pt = sum(r["prompt_tokens"] or 0 for r in ok); ca = sum(r["cached"] or 0 for r in ok)
tt = [r["ttft"] for r in ok if r["ttft"] is not None]
steps = d["radiance:engine_steps_decode_total"] + d["radiance:engine_steps_mixed_total"]
dr, ac = d["radiance:draft_tokens_total"], d["radiance:draft_accepted_total"]
info = dict(n=N, requests=len(results), errors=len(results) - len(ok), prompt_tokens=pt, cached=ca,
            cached_share=round(ca / pt, 3) if pt else None, ttft_p50=round(statistics.median(tt), 2) if tt else None,
            ttft_max=round(max(tt), 2) if tt else None, wall_s=round(wall, 1), drafts=dr, accepted=ac,
            acceptance=round(ac / dr, 3) if dr else None, decode_steps=d["radiance:engine_steps_decode_total"],
            mixed_steps=d["radiance:engine_steps_mixed_total"],
            tokens_per_step=round(d["radiance:decode_tokens_total"] / steps, 2) if steps else None,
            preempt_forced=d["radiance:preempt_forced_total"], preempted_gauge_delta=d["radiance:num_requests_preempted"])
print("CONCACC " + json.dumps(info), flush=True)
sys.exit(0 if not info["errors"] else 1)
