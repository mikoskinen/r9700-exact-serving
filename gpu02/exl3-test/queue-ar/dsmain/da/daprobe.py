#!/usr/bin/env python3
"""daprobe.py (11.11.2026, queue-ar/dsmain/da, lever DA pilot + item iej-sampled) -- the 1-user probe at a chosen sampling: the
stepprobe essay and code (ds41_tests.py PROMPTS) and editprobe's edit (the stdlib region of 9000 chars at offset 7, one
docstring added, 1500 tokens), reasoning none, c1. Per request the /metrics deltas as stepprobe.py / editprobe.py take them:
decode tok/s ((completion - 1) / (last chunk - first chunk)), decode steps, ms a step, tokens a step, draft acceptance.

  daprobe.py --mode sampled|greedy [--seed S] [--runs N] [--prompts essay,code,edit] [--gen 512] [--edit-gen 1500] [--tag T]

  sampled  the SERVER-DEFAULT sampling: no temperature / top_p / top_k in the request (the engine's defaults, temperature 1),
           and --seed S (the engine takes a per-request seed: core/server/oai.cpp "seed"; the sampler's draw is a function
           of (seed, position)), so a run is reproducible; N runs repeat the same seeded request (timing spread).
  greedy   temperature 0 (the comparison rule; stepprobe's).
One "J {json}" line a request, one "PROBE {json}" summary: per prompt mean and t95 over the runs. Standard library only;
reads ds41_tests.py from ~/gpu/ds41 (URL :8090)."""
import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import urllib.request

spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
KEYS = ["radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total", "radiance:draft_tokens_total",
        "radiance:draft_accepted_total", "radiance:decode_tokens_total"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def m():
    txt = urllib.request.urlopen(t.URL + "/metrics", timeout=60).read().decode()
    o = {}
    for k in KEYS:
        mm = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)
        o[k] = float(mm.group(2)) if mm else None
    return o


def edit_prompt(chars=9000, off=7):
    """editprobe.py's request, built the same way"""
    src = t.text_chars(chars, offset=off)
    src = src[:src.rfind("\n") + 1] if "\n" in src else src
    defs = [(mm.start(), mm.group(1)) for mm in re.finditer(r"^[ \t]*def ([A-Za-z_]\w*)\(", src, re.M)]
    mid = [d for d in defs if len(src) // 4 <= d[0] <= 3 * len(src) // 4]
    pick = (mid or defs or [(0, None)])[0][1]
    if pick:
        task = ("Add a one-line docstring to the function `%s` (its first definition below). Output the whole "
                "file with that one change and everything else exactly as it is, character for character, including "
                "comments and blank lines. Output only the file: no code fences, no explanation." % pick)
    else:
        task = ("Add a one-line comment `# reviewed` as the first line. Output the whole file with that one change and "
                "everything else exactly as it is, character for character. Output only the file: no code fences, no "
                "explanation.")
    return task + "\n\n" + src


def one(name, q, n, a):
    p = {"messages": [{"role": "user", "content": q}], "max_tokens": n, "reasoning_effort": "none"}
    if a.mode == "greedy":
        p["temperature"] = 0.0
    if a.seed is not None:
        p["seed"] = a.seed
    b = m()
    r = t.chat(p)
    e = m()
    d = {k.split(":")[1]: (e[k] - b[k]) if (e[k] is not None and b[k] is not None) else None for k in KEYS}
    u = r.get("usage") or {}
    k = u.get("completion_tokens") or 0
    dt = (r.get("t_last") or 0) - (r.get("t_first") or 0)
    steps = d["engine_steps_decode_total"] or 0
    return dict(prompt=name, mode=a.mode, seed=a.seed, status=r.get("status"), finish=r.get("finish"), tokens=k,
                decode_tps=round((k - 1) / dt, 2) if dt > 0 and k > 1 else None, decode_steps=steps,
                ms_per_decode_step=round(1000 * (d["engine_step_seconds_decode_total"] or 0) / steps, 2) if steps else None,
                tokens_per_step=round(k / steps, 3) if steps else None,
                acceptance=round((d["draft_accepted_total"] or 0) / max(1, d["draft_tokens_total"] or 0), 3),
                drafted=d["draft_tokens_total"], chars=len(r.get("content") or ""), head=(r.get("content") or "")[:80])


def ci(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    mu = statistics.mean(v)
    return mu, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sampled", "greedy"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--prompts", default="essay,code,edit")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--edit-gen", type=int, default=1500)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t.MODEL = t.model_id()
    qs = {"essay": (t.PROMPTS["essay"], a.gen), "code": (t.PROMPTS["code"], a.gen), "edit": (edit_prompt(), a.edit_gen)}
    names = a.prompts.split(",")
    rows = []
    for k in range(a.runs):
        for nm in names:
            q, n = qs[nm]
            r = one(nm, q, n, a)
            r["run"] = k
            rows.append(r)
            print("J " + json.dumps(r, ensure_ascii=False), flush=True)
    summ = {"tag": a.tag, "mode": a.mode, "seed": a.seed, "runs": a.runs}
    for nm in names:
        x = [r for r in rows if r["prompt"] == nm and r["status"] == 200]
        for f in ("decode_tps", "tokens_per_step", "acceptance", "ms_per_decode_step"):
            mu, h = ci([r[f] for r in x])
            summ["%s_%s" % (nm, f)] = [None if mu is None else round(mu, 3), None if h is None else round(h, 3)]
        summ["%s_n" % nm] = len(x)
    print("PROBE " + json.dumps(summ), flush=True)
    return 0 if all(r["status"] == 200 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
