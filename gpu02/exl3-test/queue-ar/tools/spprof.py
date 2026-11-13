#!/usr/bin/env python3
"""spprof.py ENGINE_LOG --win LABEL:FROM:TO[:PROMPT_TOKENS[:TTFT_S]] ... [--piece 2048] [--json OUT] (12.11.2026, queue-ar,
item 201) -- SP-hc's separating profile from the engine's --profile-ops tables (RADIANCE_PROFILE_EVERY 1: one cumulative
table a rank a step). For each window (engine-log line numbers) each rank's last complete table at or before FROM is
subtracted from its last one at or before TO (opprof.tables), then per rank: the collectives an op (calls, device ms,
lane-wait ms), hc_read split into the piece form and SP's kc-pinned quarter twin (geometry with kc_rows), hc_write,
hc_codes, the device ms by class, the lane-wait ms by class and the top waiting ops; per piece = / ceil(tokens / piece).
Profiled steps synchronise: shares, us a call and waits are the reading, not tok/s. Standard library only."""
import json, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opprof import tables, cls   # noqa: E402


def window(T, lo, hi):
    """{rank: {index: (calls, ms, op, geom, weight, wait_ms)}} of the window, and the steps (tables) a rank."""
    out, steps = {}, {}
    for r in sorted({rk for _, rk, _ in T}):
        mine = [(n, t) for n, rk, t in T if rk == r]
        a = [t for n, t in mine if n <= lo]
        b = [t for n, t in mine if n <= hi]
        if not b: continue
        ta, tb = (a[-1] if a else {}), b[-1]
        steps[r] = sum(1 for n, _ in mine if lo < n <= hi)
        d = {}
        for i, v in tb.items():
            u = ta.get(i)
            c = v[0] - (u[0] if u else 0)
            if c <= 0: continue
            d[i] = (c, v[1] - (u[1] if u else 0.0), v[2], v[3], v[4], v[5] - (u[5] if u else 0.0))
        out[r] = d
    return out, steps


def summ(d, pieces):
    S = {"coll": defaultdict(lambda: [0, 0.0, 0.0]), "cls_ms": defaultdict(float), "cls_wait": defaultdict(float)}
    rd = {"piece": [0, 0.0, 0, 0.0], "sp": [0, 0.0, 0, 0.0]}   # calls, ms, instances, wait ms
    hw, codes, waits = [0, 0.0, 0.0], [0, 0.0, 0.0], []
    sp_calls = []
    for i, (c, ms, op, geom, w, wt) in d.items():
        k = cls(op)
        S["cls_ms"][k] += ms
        S["cls_wait"][k] += wt
        waits.append((wt, op, geom[:60], c))
        if k == "collective" or op.endswith(("_rows",)) and ("scatter" in op or "gather" in op):
            e = S["coll"][op]; e[0] += c; e[1] += ms; e[2] += wt
        if op == "hc_read":
            key = "sp" if "kc_rows" in geom else "piece"
            e = rd[key]; e[0] += c; e[1] += ms; e[2] += 1; e[3] += wt
            if key == "sp": sp_calls.append(c)
        elif op == "hc_write":
            hw[0] += c; hw[1] += ms; hw[2] += wt
        elif op == "hc_codes":
            codes[0] += c; codes[1] += ms; codes[2] += wt
    waits.sort(reverse=True)
    pc = max(1, pieces)
    f = lambda x: round(x, 2)
    return {
        "dev_ms": f(sum(S["cls_ms"].values())), "wait_ms": f(sum(S["cls_wait"].values())),
        "cls_ms": {k: f(v) for k, v in sorted(S["cls_ms"].items(), key=lambda t: -t[1])},
        "cls_wait_ms": {k: f(v) for k, v in sorted(S["cls_wait"].items(), key=lambda t: -t[1]) if v >= 0.5},
        "coll": {op: {"calls": c, "ms": f(ms), "wait_ms": f(wt), "ms_per_piece": f(ms / pc), "us_call": f(1000 * ms / c)}
                 for op, (c, ms, wt) in sorted(S["coll"].items(), key=lambda t: -t[1][1])},
        "hc_read_piece": {"calls": rd["piece"][0], "inst": rd["piece"][2], "ms": f(rd["piece"][1]),
                          "ms_per_piece": f(rd["piece"][1] / pc), "us_call": f(1000 * rd["piece"][1] / max(1, rd["piece"][0])),
                          "wait_ms": f(rd["piece"][3])},
        "hc_read_sp": {"calls": rd["sp"][0], "inst": rd["sp"][2], "ms": f(rd["sp"][1]), "ms_per_piece": f(rd["sp"][1] / pc),
                       "us_call": f(1000 * rd["sp"][1] / max(1, rd["sp"][0])), "wait_ms": f(rd["sp"][3]),
                       "calls_per_inst_min_max": [min(sp_calls), max(sp_calls)] if sp_calls else None},
        "hc_write": {"calls": hw[0], "ms": f(hw[1]), "ms_per_piece": f(hw[1] / pc), "us_call": f(1000 * hw[1] / max(1, hw[0])),
                     "wait_ms": f(hw[2])},
        "hc_codes": {"calls": codes[0], "ms": f(codes[1]), "ms_per_piece": f(codes[1] / pc), "wait_ms": f(codes[2])},
        "top_wait": [{"wait_ms": f(w), "op": op, "geom": g, "calls": c} for w, op, g, c in waits[:6]],
    }


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return 2
    path, wins, piece, jout = a[0], [], 2048, None
    i = 1
    while i < len(a):
        if a[i] == "--win": wins.append(a[i + 1]); i += 2
        elif a[i] == "--piece": piece = int(a[i + 1]); i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        else: i += 1
    T = tables(path)
    res = {}
    for w in wins:
        p = w.split(":")
        lab, lo, hi = p[0], int(p[1]), int(p[2])
        tok = int(p[3]) if len(p) > 3 and p[3] else 0
        ttft = float(p[4]) if len(p) > 4 and p[4] else None
        pieces = math.ceil(tok / piece) if tok else 0
        d, steps = window(T, lo, hi)
        res[lab] = {"tokens": tok, "pieces": pieces, "ttft_s": ttft, "steps": steps,
                    "ranks": {r: summ(v, pieces) for r, v in d.items()}}
        for r, s in res[lab]["ranks"].items():
            print("SPPROF %s rank %d: pieces %d steps %s ttft %s | dev %.1f ms wait %.1f ms | coll %s | rd piece %s | rd sp %s | "
                  "write %s | codes %s | cls %s | waits %s | top wait %s" % (
                      lab, r, pieces, steps.get(r), ttft, s["dev_ms"], s["wait_ms"],
                      json.dumps(s["coll"]), json.dumps(s["hc_read_piece"]), json.dumps(s["hc_read_sp"]),
                      json.dumps(s["hc_write"]), json.dumps(s["hc_codes"]), json.dumps(s["cls_ms"]),
                      json.dumps(s["cls_wait_ms"]), json.dumps(s["top_wait"][:4])))
    if jout:
        with open(jout, "w") as f: json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
