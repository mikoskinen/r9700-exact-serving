#!/usr/bin/env python3
"""ds_opprof.py ENGINE_LOG [--from LINE --to LINE] [--label L] [--json OUT] [--stats A.json B.json]
(10.11.2026, queue-ar dsmain; = tools/opprof.py of item 120 with DeepSeek-V4.1's op classes)

Reads the engine's --profile-ops tables (Ctx::dump_profile: "per-op timing, rank N." then one row an op instance: calls,
total ms, us/call, site, join us, wait us, op, geometry, #index, first weight), cumulative since start. For a window given
as engine-log line numbers, each rank's last complete table at or before --from is subtracted from its last one at or
before --to, so the result is the window's device time an op. Steps in the window = the calls of the rank's
lowest-index op (one call a pass; MTP is off on the DS line).

Classes (DeepSeek-V4.1 TP4 graph): experts (moe_exl3_*, moe_gemm*), moe_other (router, scatter, gather, swiglu),
gather_ar (moe_gather_all_reduce: the fused gather + exact all-reduce), collective (all_reduce, all_gather, ar_*),
attention (attn_*, kv_store, entry_store, rope rows, qsa_*, softmax_pool), indexer (index_*), mhc (mhc_*), dense_gemm
(gemm_*: projections, engram_kv, compressor, head), embed_engram (embed_lookup_*, ngram_ids, token_map, stream_gate_add),
norm_elem (rmsnorm, fill, cast, rope_table), other.

Output: per rank device ms and ms a step by class (share of the rank's device time), the lane waits ('wait us': the op's
lane idle before it started -- stream waits and issue delays, in no op's time) a step, and with --stats (two /stats
snapshots around the window) rank 0's wall ms a step and the remainder wall - device - waits (host issue + the profiler's
own synchronisation). Then rank 0's expert ops by op and by layer (us a call) and its top ops. PROFILED: every number is
from synchronised steps; shares and us a call are the reading, not a speed. Standard library only."""
import json, re, sys
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HDR = re.compile(r"per-op timing, rank (\d+)")
ORDER = ["experts", "moe_other", "gather_ar", "collective", "attention", "indexer", "mhc", "dense_gemm", "embed_engram",
         "norm_elem", "other"]


def cls(op):
    o = op.lower()
    if o.startswith("moe_exl3") or "exl3" in o or o.startswith("moe_gemm"): return "experts"
    if o.startswith("moe_gather_all_reduce"): return "gather_ar"
    if o.startswith(("all_reduce", "all_gather", "ar_", "reduce_scatter")) or "all_reduce" in o: return "collective"
    if o.startswith(("moe_", "router", "swiglu")): return "moe_other"
    if o.startswith("index_"): return "indexer"
    if o.startswith("mhc_"): return "mhc"
    if "attn" in o or o.startswith(("kv_store", "entry_store", "ds41_rope", "rope_rows", "rmsnorm_rope", "qsa_",
                                    "softmax_pool", "latent")): return "attention"
    if o.startswith(("gemm", "logits")): return "dense_gemm"
    if o.startswith(("embed_lookup", "ngram", "token_map", "stream_gate")): return "embed_engram"
    if o.startswith(("rmsnorm", "fill", "cast", "rope_table", "norm")): return "norm_elem"
    return "other"


def tables(path):
    """[(line_no_of_table_end, rank, {index: (calls, ms, op, geom, weight, wait_ms)})] in log order."""
    out, cur, rank = [], None, None
    with open(path, errors="replace") as f:
        for n, l in enumerate(f, 1):
            m = HDR.search(l)
            if m:
                if cur is not None: out.append((n - 1, rank, cur))
                cur, rank = {}, int(m.group(1)); continue
            if cur is None: continue
            r = ROW.match(l)
            if r:
                cur[int(r.group(9))] = (int(r.group(1)), float(r.group(2)), r.group(7), r.group(8), r.group(10),
                                        float(r.group(6)) * int(r.group(1)) / 1000.0)
            elif l.strip().startswith("total"):
                out.append((n, rank, cur)); cur = None
    return out


def stats_wall(a, b):
    try:
        A, B = json.load(open(a)), json.load(open(b))
    except Exception:
        return None
    ds = B.get("steps", 0) - A.get("steps", 0)
    if ds <= 0: return None
    return {"steps": ds, "ms_step": (B.get("step_ns", 0) - A.get("step_ns", 0)) / ds / 1e6,
            "decode_steps": B.get("steps_decode", 0) - A.get("steps_decode", 0),
            "prefill_mixed_steps": (B.get("steps_prefill", 0) + B.get("steps_mixed", 0)) - (A.get("steps_prefill", 0) + A.get("steps_mixed", 0))}


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return 2
    path, lo, hi, label, jout, st = a[0], 0, 1 << 60, "", None, None
    i = 1
    while i < len(a):
        if a[i] == "--from": lo = int(a[i + 1]); i += 2
        elif a[i] == "--to": hi = int(a[i + 1]); i += 2
        elif a[i] == "--label": label = a[i + 1]; i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        elif a[i] == "--stats": st = (a[i + 1], a[i + 2]); i += 3
        else: i += 1
    T = tables(path)
    res = {}
    for rk in sorted({r for _, r, _ in T}):
        mine = [(n, t) for n, r, t in T if r == rk]
        before = [t for n, t in mine if n <= lo]
        after = [t for n, t in mine if n <= hi]
        if not after: continue
        A = before[-1] if before else {}
        B = after[-1]
        d = {}
        for idx, (c, ms, op, geom, w, wt) in B.items():
            p = A.get(idx, (0, 0.0, "", "", "", 0.0))
            if c - p[0] > 0: d[idx] = (c - p[0], ms - p[1], op, geom, w, wt - p[5])
        if d: res[rk] = d
    if not res:
        print(f"DSOPPROF {label}: no complete profile table in the window (PROFILED)"); return 1
    summ = {}
    for rk, d in res.items():
        byc = defaultdict(float)
        for c, ms, op, g, w, wt in d.values(): byc[cls(op)] += ms
        steps = d[min(d)][0]
        summ[rk] = {"steps": steps, "device_ms": sum(byc.values()), "wait_ms": sum(x[5] for x in d.values()),
                    "by_class_ms": dict(byc)}
    wall = stats_wall(*st) if st else None
    out = [f"DSOPPROF {label} (PROFILED, not a speed): device time a rank, ms a step (share):"]
    for rk, s in sorted(summ.items()):
        n = max(1, s["steps"]); tot = s["device_ms"] or 1.0
        cl = ", ".join(f"{k} {s['by_class_ms'][k] / n:.2f} ({100 * s['by_class_ms'][k] / tot:.0f} %)"
                       for k in ORDER if s["by_class_ms"].get(k))
        out.append(f" | rank {rk}: {s['steps']} steps, device {s['device_ms']:.1f} ms in the window = {s['device_ms'] / n:.2f} ms + lane waits {s['wait_ms'] / n:.2f} ms a step = {cl}")
    rk0 = sorted(res)[0]
    if wall:
        s = summ[rk0]; n = max(1, s["steps"])
        rem = wall["ms_step"] - (s["device_ms"] + s["wait_ms"]) / n
        out.append(f" | rank {rk0} wall (/stats, {wall['steps']} steps: {wall['decode_steps']} decode, {wall['prefill_mixed_steps']} prefill/mixed) "
                   f"{wall['ms_step']:.2f} ms a step; remainder after device + waits {rem:.2f} ms ({100 * rem / wall['ms_step']:.0f} %: host issue + profiler syncs)")
    d0 = res[rk0]
    byop = defaultdict(lambda: [0, 0.0])
    for c, ms, op, g, w, wt in d0.values():
        if cls(op) == "experts": byop[op][0] += c; byop[op][1] += ms
    out.append(f" | rank {rk0} expert ops: " + ", ".join(f"{op} {v[1]:.1f} ms / {v[0]} calls = {1000 * v[1] / v[0]:.1f} us"
                                                    for op, v in sorted(byop.items(), key=lambda kv: -kv[1][1])))
    per = [(1000 * ms / c, op, re.sub(r"\D*(\d+).*", r"\1", w) if re.search(r"\d", w) else w) for c, ms, op, g, w, wt in d0.values() if cls(op) == "experts"]
    for op in sorted(byop):
        us = sorted(p[0] for p in per if p[1] == op)
        if not us: continue
        slow = ", ".join(f"blk {p[2]} {p[0]:.0f}" for p in sorted((p for p in per if p[1] == op), key=lambda x: -x[0])[:5])
        out.append(f" | {op} us a call by layer: min {us[0]:.1f} p25 {us[len(us) // 4]:.1f} p50 {us[len(us) // 2]:.1f} "
                   f"p75 {us[3 * len(us) // 4]:.1f} max {us[-1]:.1f} (slowest: {slow})")
    byname = defaultdict(lambda: [0, 0.0, 0.0])
    for c, ms, op, g, w, wt in d0.values():
        byname[op][0] += c; byname[op][1] += ms; byname[op][2] += wt
    n0 = max(1, summ[rk0]["steps"])
    out.append(f" | rank {rk0} top ops (ms a step, us a call, lane wait ms a step): " + "; ".join(
        f"{op} {v[1] / n0:.3f} ms {1000 * v[1] / v[0]:.1f} us w {v[2] / n0:.3f}" for op, v in sorted(byname.items(), key=lambda kv: -kv[1][1])[:12]))
    print("".join(out))
    if jout:
        with open(jout, "w") as f:
            json.dump({"label": label, "profiled": True, "summary": {str(k): v for k, v in summ.items()}, "wall_rank0": wall,
                       "rank0_experts_by_op": {k: v for k, v in byop.items()},
                       "rank0_by_op": {k: v for k, v in byname.items()}}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
