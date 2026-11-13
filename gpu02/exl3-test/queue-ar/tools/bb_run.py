#!/usr/bin/env python3
"""bb_run.py --bb-root DIR --deps DIR [--thinking off|server] -- BETTERBENCH ARGS...   (8.11.2026, queue-ar)

Runs BetterBench (https://github.com/GGZ14/BetterBench, Apache-2.0; read before use: it talks only to --endpoint, plus a
GitHub tag check that --no-update-check turns off, and runs `rocm-smi` for its environment record) from a source tree
(--bb-root) with numpy from an unpacked wheel (--deps), under `python3 -I` (no user site, no PYTHON* variables), so nothing
is installed anywhere.

--thinking off (the default) adds "reasoning_effort": "none" to every chat request BetterBench sends: Radiance's chat
endpoint takes that as thinking off (GUIDE: "none and minimal turn thinking off"), which is how the Radiance docs' BetterBench
figures were taken. BetterBench itself has no switch for it and its run path passes no extra request fields, so the one
function that builds the payload (betterbench.client._build_payload) is wrapped; every request goes through it.
--thinking server leaves the requests as BetterBench sends them (the server's --reasoning-effort decides).

After the run, `--summary RESULTS.json` prints one BB line (decode per category and weighted, prefill per depth,
concurrency aggregate per level, the reasoning count), and `--check RESULTS.json` (a separate call) prints how many single-stream results carried any reasoning text
(reasoning_chars > 0): 0 is thinking off as asked.
"""
import json
import os
import sys


def check(path):
    d = json.load(open(path))
    n = r = 0
    for cat, recs in (d.get("single_stream") or {}).items():
        for x in recs:
            n += 1
            r += 1 if (x.get("reasoning_chars") or 0) > 0 else 0
    print(f"BBCHECK {json.dumps({'results': n, 'with_reasoning': r, 'thinking_off': n > 0 and r == 0})}")
    return 0 if n and r == 0 else 1


def summary(path):
    """One line: the single-stream decode median per category and weighted as BetterBench's report weights them, the
    prefill sweep's median tok/s per depth, the concurrency sweep's aggregate tok/s per level, and the thinking check."""
    import statistics as st
    d = json.load(open(path))
    w = (d.get("config") or {}).get("weights") or {}
    cats, ttfa = {}, {}
    for cat, recs in (d.get("single_stream") or {}).items():
        v = [x["decode_tps"] for x in recs if x.get("ok") and x.get("decode_tps")]
        if v:
            cats[cat] = round(st.median(v), 1)
        t = [x["ttft_ms"] for x in recs if x.get("ok") and x.get("ttft_ms")]
        if t:
            ttfa[cat] = round(st.median(t), 1)
    ws = sum(w.get(c, 0) for c in cats)
    weighted = round(sum(cats[c] * w.get(c, 0) for c in cats) / ws, 1) if ws else None
    pf = {str(x["target_depth"]): round(st.median(x["pp_tps"]), 1) for x in (d.get("prefill") or []) if x.get("pp_tps")}
    cc = {str(x["level"]): round(x["aggregate_tps"], 1) for x in (d.get("concurrency") or []) if x.get("aggregate_tps")}
    n = sum(len(r) for r in (d.get("single_stream") or {}).values())
    r = sum(1 for recs in (d.get("single_stream") or {}).values() for x in recs if (x.get("reasoning_chars") or 0) > 0)
    out = {"version": d.get("betterbench_version"), "corpus": d.get("corpus_version"), "decode_weighted_tps": weighted,
           "decode_tps_by_cat": cats, "ttft_ms_by_cat": ttfa, "prefill_tps_by_depth": pf, "concurrency_agg_tps": cc,
           "with_reasoning": f"{r}/{n}", "notes": (d.get("env") or {}).get("notes")}
    print("BB " + json.dumps(out))
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "--check":
        return check(a[1])
    if a and a[0] == "--summary":
        return summary(a[1])
    opts = {"--bb-root": None, "--deps": None, "--thinking": "off"}
    while a and a[0] != "--":
        k = a.pop(0)
        if k not in opts or not a:
            sys.exit(f"bb_run.py: unknown or incomplete option {k}")
        opts[k] = a.pop(0)
    if a and a[0] == "--":
        a.pop(0)
    root, deps = opts["--bb-root"], opts["--deps"]
    if not root or not os.path.isdir(os.path.join(root, "betterbench")):
        sys.exit(f"bb_run.py: --bb-root {root!r} has no betterbench/ package")
    if deps:
        sys.path.insert(0, os.path.abspath(deps))
    sys.path.insert(0, os.path.abspath(root))
    import betterbench
    import betterbench.client as client
    from betterbench import cli
    if opts["--thinking"] == "off":
        orig = client._build_payload

        def build(*args, **kw):
            p = orig(*args, **kw)
            p["reasoning_effort"] = "none"
            return p
        client._build_payload = build
    elif opts["--thinking"] != "server":
        sys.exit("bb_run.py: --thinking takes off or server")
    if "--no-update-check" not in a:
        a = a[:1] + ["--no-update-check"] + a[1:]
    print(f"bb_run: BetterBench {betterbench.__version__} corpus {getattr(betterbench, 'CORPUS_VERSION', '?')} from {root}, "
          f"thinking {opts['--thinking']}, args {' '.join(a)}", flush=True)
    return cli.main(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
