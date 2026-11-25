#!/usr/bin/env python3
"""perpos.py URL ENGINE_LOG_CMD_OUT OUT (10.11.2026, queue-ar dsmain) -- per-position DSpark acceptance for one prompt at a
time: sends a fixed prompt (PERPOS_PROMPT essay|code) to /completion (greedy, n_predict PERPOS_N, return_tokens), then
reads the engine's -v lines "drafter '...': sequence S proposes d1 .. d5" that appeared while it ran (the caller passes the
engine log written after the request) and replays them against the reply's ids: a block proposed after anchor position p
predicts ids[p+1 .. p+5]; its accepted run is the leading matches, then the anchor moves by run + 1. Prints one PERPOS line:
blocks, tokens a step, accepted per position (1..5) and the conditional acceptance P(pos k | pos k-1 accepted)."""
import json, re, sys, urllib.request

PROMPTS = {
    # 10.11.: a raw /completion of the instruction-style essay prompt ended after 1 token (no chat template on this
    # route; the code prompt kept going). The essay is now a text continuation, which the model writes on.
    "essay": "Why Cities Should Invest in Public Libraries\n\nAn essay in five parts: an introduction, three arguments "
             "with examples, and a conclusion.\n\nIntroduction\n\nIn most cities the public library",
    "code": "Write a complete, well-commented Python module that implements an LRU cache class with get, put and "
            "delete methods, type hints, and a small unittest suite that covers eviction order and capacity edge cases.",
}


def main():
    if len(sys.argv) < 4:
        print(__doc__); return 2
    url, mode = sys.argv[1], sys.argv[2]
    import os
    which = os.environ.get("PERPOS_PROMPT", "essay"); n = int(os.environ.get("PERPOS_N", "384"))
    if mode == "send":   # perpos.py URL send OUT.json
        body = {"prompt": PROMPTS[which], "n_predict": n, "temperature": 0.0, "return_tokens": True}
        req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            rep = json.loads(r.read().decode())
        json.dump(rep, open(sys.argv[3], "w"))
        print("PERPOS_SENT " + json.dumps({"prompt": which, "tokens": len(rep.get("tokens") or [])}))
        return 0
    # perpos.py - score REPLY.json LOGFILE
    rep = json.load(open(sys.argv[3])); ids = rep.get("tokens") or []
    pat = re.compile(r"drafter '[^']*': sequence (\d+) proposes((?: -?\d+)+)")
    blocks = [list(map(int, m.group(2).split())) for m in map(pat.search, open(sys.argv[4], errors="replace")) if m]
    def replay(p0):
        p, acc_pos, seen_pos, runs = p0, [0] * 5, [0] * 5, []
        for d in blocks:
            if p + 1 >= len(ids): break
            run = 0
            for j, t in enumerate(d[:5]):
                if p + 1 + j >= len(ids): break
                seen_pos[j] += 1
                if t == ids[p + 1 + j]:
                    acc_pos[j] += 1; run += 1
                else:
                    break
            runs.append(run)
            p += run + 1
        return acc_pos, seen_pos, runs
    # the first block's anchor: the prefill's token (0) unless the first decode step proposed nothing (1)
    best = max((replay(p0) + (p0,) for p0 in (0, 1)), key=lambda r: (sum(r[0]), -r[3]))
    acc_pos, seen_pos, runs, p0 = best
    cond = [round(acc_pos[k] / seen_pos[k], 3) if seen_pos[k] else None for k in range(5)]
    tps = round(sum(r + 1 for r in runs) / len(runs), 3) if runs else None
    print("PERPOS " + json.dumps({"prompt": os.environ.get("PERPOS_PROMPT", "?"), "reply_tokens": len(ids), "blocks_in_log": len(blocks),
                                  "blocks_scored": len(runs), "first_anchor": p0, "tokens_a_step": tps, "accepted_by_pos": acc_pos,
                                  "conditional_by_pos": cond}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
