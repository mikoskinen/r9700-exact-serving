#!/usr/bin/env python3
"""profchunks.py ENGINE_LOG [RANK] [--csv OUT] (test117, 6.11.2026) -- where a prefill chunk's time grows with context.

Input: the docker log of a profiled server run with --profile-ops, RADIANCE_PROFILE_EVERY=1 and the -profsync image (its dump header
names the step's phase and n_tok: 'per-op timing, rank 0 (step 812 prefill n_tok 8192, ...)'), driven by fresh single prefills one
after the other (ds41_tests.py prefill --depths ...). With RADIANCE_PROFILE_SYNC=1 every device op is timed on the host clock between
lane syncs, so chunk ops an event pair would miss are counted too.

Consecutive cumulative dumps of RANK differ by one step. Every prefill step is one chunk; a chunk's context start is its index within
its request times the full chunk size (the first chunk of a request is the one after any non-prefill step). Per chunk: device ms by
op group (op name, plus the layer's index-op family). Then, per group, a least-squares line ms = a + b * ctx_start over the full-size
chunks: b / chunk_tokens is the group's growth in ns per context token per prompt token -- the figure to compare with the measured
~0.9 ns (UD-IQ1_S 16k 2,844 -> 644k 1,081 tok/s). Standard library only.
"""
import csv
import re
import sys
from collections import defaultdict

HDR = re.compile(r"^per-op timing, rank (\d+)(?: \(step (\d+) (\w+) n_tok (\d+),)?")
ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
GROUPS = [  # (group, op-name regex, weight regex or None); first match wins
    ("index_score", r"^index_score", None), ("index_topk", r"^index_topk", None),
    ("index_candidates", r"^index_candidates", None), ("attn_sparse_sink", r"^attn_sparse", None),
    ("compressor", r"qsa|softmax_pool|compress", None), ("kv_store", r"entry_store|kv_store", None),
    ("all_reduce", r"all_reduce|allreduce|all_gather|reduce_scatter", None),
    ("moe_routed", r"^moe_", None), ("mhc", r"mhc", None), ("router", r"router", None),
    ("gemm_indexer", r"gemm|gemv", r"indexer"), ("gemm_attn", r"gemm|gemv", r"attn_(q_a|q_b|kv|output)"),
    ("gemm_shared", r"gemm|gemv", r"shexp"), ("gemm_engram", r"gemm|gemv", r"engram"),
    ("gemm_compressor", r"gemm|gemv", r"compressor"), ("gemm_other", r"gemm|gemv", None),
    ("rope_quant", r"rope|roundtrip|quant", None), ("engram_host", r"embed_lookup|ngram|token_map", None),
]


def group(op, w):
    for g, orx, wrx in GROUPS:
        if re.search(orx, op) and (wrx is None or re.search(wrx, w)):
            return g
    return "other"


def dumps(path, rank):
    out, cur, meta = [], None, None
    with open(path, errors="replace") as f:
        for l in f:
            m = HDR.match(l)
            if m:
                if int(m.group(1)) == rank:
                    cur = {}
                    meta = (int(m.group(2)) if m.group(2) else -1, m.group(3) or "?", int(m.group(4)) if m.group(4) else -1)
                else:
                    cur = None
                continue
            if cur is None:
                continue
            r = ROW.match(l)
            if r:
                if r.group(4) == "dev" or "--host" in sys.argv:
                    cur[(r.group(7), r.group(9), r.group(10))] = float(r.group(2))
            elif l.strip().startswith("total"):
                out.append((meta, cur))
                cur = None
    return out


def fit(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main():
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
    ds = dumps(path, rank)
    chunks, idx, full = [], 0, 0
    for (ma, a), (mb, b) in zip(ds, ds[1:]):
        step, phase, ntok = mb
        if phase != "prefill":
            idx = 0
            continue
        g = defaultdict(float)
        for k, v in b.items():
            d = v - a.get(k, 0.0)
            if d > 0:
                g[group(k[0], k[2])] += d
        full = max(full, ntok)
        chunks.append({"step": step, "n_tok": ntok, "chunk": idx, "groups": g})
        idx += 1
    if not chunks:
        print("no prefill steps with a phase header (the -profsync image and RADIANCE_PROFILE_EVERY=1 are needed)")
        return
    for c in chunks:
        c["ctx_start"] = c["chunk"] * full
    names = sorted({k for c in chunks for k in c["groups"]}, key=lambda k: -sum(c["groups"].get(k, 0) for c in chunks))
    if "--csv" in sys.argv:
        with open(sys.argv[sys.argv.index("--csv") + 1], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "n_tok", "chunk", "ctx_start", "total_ms"] + names)
            for c in chunks:
                w.writerow([c["step"], c["n_tok"], c["chunk"], c["ctx_start"], round(sum(c["groups"].values()), 3)]
                           + [round(c["groups"].get(k, 0.0), 3) for k in names])
    fc = [c for c in chunks if c["n_tok"] == full]
    xs = [c["ctx_start"] for c in fc]
    print(f"# {path} rank {rank}: {len(chunks)} prefill chunks ({len(fc)} of {full} tokens), ctx_start 0..{max(xs) if xs else 0}")
    print(f"{'group':18s} {'ms@ctx0':>9s} {'ms/chunk per 100k ctx':>22s} {'ns per ctx-token/token':>23s} {'share@0':>8s} {'share@max':>9s}")
    tot0, totm = fit(xs, [sum(c["groups"].values()) for c in fc])
    xmax = max(xs) if xs else 0
    for k in names + ["TOTAL"]:
        ys = [sum(c["groups"].values()) if k == "TOTAL" else c["groups"].get(k, 0.0) for c in fc]
        a, b = fit(xs, ys)
        at0, atm = a, a + b * xmax
        t0, tm = tot0, tot0 + totm * xmax
        print(f"{k:18s} {a:9.1f} {b * 1e5:22.1f} {b / full * 1e6 if full else 0:23.3f} {at0 / t0 if t0 else 0:8.1%} {atm / tm if tm else 0:9.1%}")


if __name__ == "__main__":
    main()
