#!/usr/bin/env python3
"""daheld.py (11.11.2026, queue-ar/dsmain/da, lever DA fidelity item, tag xdaf) -- SCORE A DRAFTER WITH THE ENGINE on the pilot's
held-out generated documents: the same documents the port's simulated acceptance was scored on (datrain.py split_docs: the
validation documents + ~10 % of the rest, random.Random(1) in kld.json order), the generated kinds only (val-*, gen-*: target
greedy text; the public ones are human text, where the target's own continuation is not the document). Standard library only.

  daheld.py run URL LOG KLDIR OUT [--max-docs N] [--held 0.1] [--seed 1]
      per document: the prompt = the ids up to the chat template's assistant prefix (found by /tokenize of
      "<｜Assistant｜></think>"), POST /completion with those ids, greedy, n_predict = the reply's length, return_tokens; the
      engine's -v proposals of the request (the followed LOG) replayed against the reply (first anchor 0 or 1, as daanat.py);
      the reply must equal the document's own reply (temperature 0, exact) -> OUT-rows.jsonl, then per kind: tokens a step
      (fixed window 5, replay), conditional acceptance p1..p5, the engine's accepted count, identity (equal replies).
  daheld.py selftest
Prints "DAHELD {json}" lines."""
import json
import os
import random
import re
import struct
import sys
import time
import urllib.request

PAT = re.compile(r"drafter '[^']*': sequence (\d+) proposes((?: -?\d+)+)")


def http(url, body=None, timeout=900):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def held_docs(kldir, held=0.1, seed=1):
    """datrain.py's split_docs, in its order and with its draws"""
    m = json.load(open(os.path.join(kldir, "kld.json")))
    raw = open(os.path.join(kldir, "tokens.i32"), "rb").read()
    toks = struct.unpack("<%di" % (len(raw) // 4), raw)
    rnd = random.Random(seed)
    out, off = [], 0
    for d in m["docs"]:
        ids = list(toks[off:off + d["tokens"]])
        off += d["tokens"]
        k = d["source"].split(":")[0]
        if k.startswith("val"):
            h = True
        else:
            h = rnd.random() < held
        if h and (k.startswith("val") or k.startswith("gen")):
            out.append({"source": d["source"], "kind": k, "ids": ids})
    return out


def replay(ids, blocks, p0):
    p, acc, reach, runs = p0, [0] * 5, [0] * 5, []
    for d in blocks:
        if p + 1 >= len(ids):
            break
        run = 0
        for j, t in enumerate(d[:5]):
            if p + 1 + j >= len(ids):
                break
            reach[j] += 1
            if t != ids[p + 1 + j]:
                break
            acc[j] += 1
            run += 1
        runs.append(run)
        p += run + 1
    return acc, reach, runs


def logsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def run(a):
    url = a.url.rstrip("/")
    # 12.11. fix (ds-main-port): {"content": ...} does not parse the special tokens (the 13:23 run matched 0 of 192 docs: "no
    # assistant prefix"). The marker is now rendered the way dagen.py rendered the documents' prompts -- /tokenize with messages
    # and reasoning_effort none -- and taken as the template's tail after the user text (its last two ids, <｜Assistant｜></think>).
    ref = "Hello there"
    tail = [int(t) for t in http(url + "/tokenize", {"messages": [{"role": "user", "content": ref}], "reasoning_effort": "none"})["tokens"]]
    body = [int(t) for t in http(url + "/tokenize", {"content": ref})["tokens"]]
    cut = -1   # the template's tail = everything after the LAST occurrence of the user text's own ids
    for i in range(len(tail) - len(body), -1, -1):
        if body and tail[i:i + len(body)] == body:
            cut = i + len(body)
            break
    marker = tail[cut:] if 0 <= cut < len(tail) else tail[-2:]
    marker = [t for t in marker if t not in body] or tail[-2:]
    print("DAHELD " + json.dumps({"marker": marker}), flush=True)
    docs = held_docs(a.kldir, a.held, a.seed)
    if a.max_docs:
        docs = docs[:a.max_docs]
    rows = []
    with open(a.out + "-rows.jsonl", "w") as fo:
        for d in docs:
            ids, pl = d["ids"], -1
            for i in range(len(ids) - len(marker), -1, -1):
                if ids[i:i + len(marker)] == marker:
                    pl = i + len(marker)
                    break
            if pl <= 0 or pl >= len(ids) - 6:
                print("DAHELD " + json.dumps({"skip": d["source"], "why": "no assistant prefix"}), flush=True)
                continue
            want = ids[pl:]
            o0 = logsize(a.log)
            t0 = time.time()
            r = http(url + "/completion", {"prompt": ids[:pl], "n_predict": len(want), "temperature": 0.0, "return_tokens": True})
            dt = time.time() - t0
            time.sleep(a.settle)
            with open(a.log, "rb") as f:
                f.seek(o0)
                chunk = f.read(max(0, logsize(a.log) - o0)).decode(errors="replace")
            props = {}
            for mm in PAT.finditer(chunk):
                props.setdefault(mm.group(1), []).append([int(x) for x in mm.group(2).split()])
            seq, blocks = max(props.items(), key=lambda kv: len(kv[1])) if props else (None, [])
            got = [int(t) for t in (r.get("tokens") or [])]
            best = max(((replay(got, blocks, p0), p0) for p0 in (0, 1)), key=lambda x: (sum(x[0][0]), -x[1]))
            (acc, reach, runs), p0 = best
            tm = r.get("timings") or {}
            row = {"source": d["source"], "kind": d["kind"], "prompt": pl, "reply": len(got), "same": got == want,
                   "first_diff": next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None if len(got) == len(want) else min(len(got), len(want))),
                   "blocks": len(blocks), "acc": acc, "reach": reach, "runs": runs, "s": round(dt, 2),
                   "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"), "seqs": len(props)}
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            rows.append(row)
            print("DAHELD " + json.dumps({k: row[k] for k in ("source", "prompt", "reply", "same", "blocks", "s")}), flush=True)
    summ = {}
    for k in sorted({r["kind"] for r in rows}) + ["all-gen"]:
        x = [r for r in rows if (r["kind"] == k if k != "all-gen" else True)]
        acc = [sum(r["acc"][j] for r in x) for j in range(5)]
        reach = [sum(r["reach"][j] for r in x) for j in range(5)]
        runs = [v for r in x for v in r["runs"]]
        summ[k] = {"docs": len(x), "tokens_a_step": round(sum(v + 1 for v in runs) / max(1, len(runs)), 4),
                   "cond": [round(acc[j] / reach[j], 4) if reach[j] else None for j in range(5)], "blocks": len(runs),
                   "identical": sum(1 for r in x if r["same"]), "engine_accepted": sum(r["draft_n_accepted"] or 0 for r in x)}
    print("DAHELDSUM " + json.dumps(summ), flush=True)
    return 0


def selftest():
    acc, reach, runs = replay([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                              [[11, 77, 1, 1, 1], [13, 14, 15, 16, 66], [18, 5, 0, 0, 0], [20, 21, 0, 0, 0]], 0)
    ok = runs == [1, 4, 1, 2] and acc == [4, 2, 1, 1, 0] and reach == [4, 4, 1, 1, 1]
    print("DAHELD " + json.dumps({"selftest": "ok" if ok else "FAIL", "acc": acc, "reach": reach, "runs": runs}))
    return 0 if ok else 1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run")
    r.add_argument("url"); r.add_argument("log"); r.add_argument("kldir"); r.add_argument("out")
    r.add_argument("--max-docs", type=int, default=0); r.add_argument("--held", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=1); r.add_argument("--settle", type=float, default=1.5)
    sp.add_parser("selftest")
    a = ap.parse_args()
    return {"run": run, "selftest": lambda a: selftest()}.get(a.cmd, lambda a: 2)(a)


if __name__ == "__main__":
    sys.exit(main())
