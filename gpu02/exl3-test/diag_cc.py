#!/usr/bin/env python3
"""diag_cc.py URL OUTDIR (5.11.2026; user OK "joo" to a ~5-min diagnosis) -- two cc_sim-built Claude Code requests (cc_sim's own
session builder: seed 1 sessions u0 / u1, preaged to its steady-state context, 109 tools, natural generation up to 13000 tokens,
effort xhigh / medium) sent at once to one server. Saves each reply (reasoning + content + tool calls) to OUTDIR and prints one
DIAG line per reply with repetition stats, to tell a loop from long reasoning."""
import collections, json, os, sys, threading, time, types
sys.path.insert(0, os.path.expanduser("~/gpu/bench"))
import cc_sim as C

url, out = sys.argv[1], sys.argv[2]
C.load_corpus()   # cc_sim main() does this before building sessions (fmt/file_lines pick from CORPUS)
os.makedirs(out, exist_ok=True)
a = types.SimpleNamespace(seed=1, start_min=20000, start_max=40000, ctx_cap=170000, preage=True, gen_median=350, gen_sigma=1.0,
                          gen_max=13000, effort="xhigh", shared_prefix=0, cpt=3.5, drop_reasoning=False, gen_mode="natural",
                          temperature=None, tool_median=300, tool_sigma=1.2, tool_min=20, tool_max=8000, route="hash",
                          run_id="diag-1105")
sim = C.Sim(a, [url], C.model_id(url, ""), "", C.build_tools(109), os.path.join(out, "diag.jsonl"))


def dup_share(text, n=8):
    w = text.split()
    sh = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n))]
    return round(1 - len(set(sh)) / len(sh), 3) if sh else 0.0


def one(u, effort):
    s = sim.new_session(u, 0)
    if effort:
        s["effort"] = effort
    effort = s["effort"]
    r = C.stream_chat(url, "", sim.payload(s, a.gen_max), time.time() + 1500, 1200)
    with open(os.path.join(out, f"u{u}-{effort}.txt"), "w") as f:
        f.write(r["reasoning"] + "\n===== CONTENT =====\n" + r["content"] + "\n===== TOOL CALLS =====\n" + json.dumps(r["tool_calls"]))
    rs = r["reasoning"]; lines = [l for l in rs.splitlines() if l.strip()]
    top = collections.Counter(lines).most_common(1)
    half = len(rs) // 2
    print("DIAG " + json.dumps(dict(
        u=u, effort=effort, prompt=r["usage"].get("prompt_tokens"), gen=r["usage"].get("completion_tokens"),
        finish=r["finish"], err=r["err"], secs=round(r["t1"] - r["t0"], 1), reasoning_chars=len(rs),
        content_chars=len(r["content"]), tool_calls=[(t["name"], t["arguments"][:100]) for t in r["tool_calls"]],
        dup8_all=dup_share(rs), dup8_first_half=dup_share(rs[:half]), dup8_second_half=dup_share(rs[half:]),
        top_line=(top[0][1], top[0][0][:100]) if top else None, tail=rs[-300:])), flush=True)


if len(sys.argv) > 3:   # N users at once (sessions u0..uN-1, effort as cc_sim picks it), max_tokens argv[4]
    n, a.gen_max = int(sys.argv[3]), int(sys.argv[4])
    ts = [threading.Thread(target=one, args=(u, None)) for u in range(n)]
else:
    ts = [threading.Thread(target=one, args=(0, "xhigh")), threading.Thread(target=one, args=(1, "medium"))]
for t in ts:
    t.start()
for t in ts:
    t.join()
