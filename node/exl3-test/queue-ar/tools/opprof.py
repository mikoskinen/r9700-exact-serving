#!/usr/bin/env python3
"""opprof.py ENGINE_LOG [--from LINE --to LINE] [--label L] [--json OUT] (9.11.2026, queue-ar, item 120) -- the engine's
--profile-ops tables (Ctx::dump_profile: "per-op timing, rank N." then one row an op instance: calls, total ms, us/call,
site, join us, wait us, op, geometry, #index, first weight), cumulative since start. For a window given as engine-log line
numbers, each rank's last complete table at or before --from is subtracted from its last one at or before --to, so the
result is the window's device time an op. Grouped by class (EXL3 expert GEMMs and their prep, other MoE, attention,
delta-net, gated residual, GEMMs, collectives, other), per rank and summed; the EXL3 ops also by layer (their first
weight) with us a call, so layers of another K show as another cluster. Standard library only."""
import json, re, sys
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HDR = re.compile(r"per-op timing, rank (\d+)\.")


def cls(op):
    o = op.lower()
    if "exl3" in o: return "exl3_expert"
    if o.startswith(("moe_gemm", "gated_quant", "gated_had", "moe_down")): return "moe_expert"
    if o.startswith(("moe_", "router")): return "moe_other"
    if "attn" in o or o.startswith(("qk_norm", "qkv_", "qsa_", "kv_store", "rope")): return "attention"
    if o.startswith("gdn") or "delta" in o or o.startswith("conv"): return "deltanet"
    if o.startswith(("hc_", "mtp_enter")): return "gated_residual"
    if o.startswith("ar_") or "all_reduce" in o or o.startswith(("all_gather", "reduce")): return "collective"
    if o.startswith(("gemm", "logits")): return "gemm"
    return "other"


def tables(path):
    """[(line_no_of_table_end, rank, {index: (calls, ms, op, geom, weight)})] in log order."""
    out, cur, rank, start = [], None, None, 0
    with open(path, "rb") as g: gz = g.read(2) == b"\x1f\x8b"   # 13.11.: a gzipped capture too (docker logs -f | gzip)
    import gzip
    with (gzip.open(path, "rt", errors="replace") if gz else open(path, errors="replace")) as f:
        for n, l in enumerate(f, 1):
            m = HDR.search(l)
            if m:
                if cur is not None: out.append((n - 1, rank, cur))
                cur, rank = {}, int(m.group(1)); continue
            if cur is None: continue
            r = ROW.match(l)
            if r:
                cur[int(r.group(9))] = (int(r.group(1)), float(r.group(2)), r.group(7), r.group(8), r.group(10),
                                        float(r.group(6)) * int(r.group(1)) / 1000.0)   # wait ms in total
            elif l.strip().startswith("total"):
                out.append((n, rank, cur)); cur = None
    return out


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return 2
    path, lo, hi, label, jout = a[0], 0, 1 << 60, "", None
    i = 1
    while i < len(a):
        if a[i] == "--from": lo = int(a[i + 1]); i += 2
        elif a[i] == "--to": hi = int(a[i + 1]); i += 2
        elif a[i] == "--label": label = a[i + 1]; i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        else: i += 1
    T = tables(path)
    ranks = sorted({r for _, r, _ in T})
    res = {"label": label, "ranks": {}}
    for rk in ranks:
        mine = [(n, t) for n, r, t in T if r == rk]
        before = [t for n, t in mine if n <= lo]
        after = [t for n, t in mine if n <= hi]
        if not after: continue
        A = before[-1] if before else {}
        B = after[-1]
        d = {}
        for idx, (c, ms, op, geom, w, wt) in B.items():
            a = A.get(idx, (0, 0.0, "", "", "", 0.0))
            if c - a[0] > 0: d[idx] = (c - a[0], ms - a[1], op, geom, w, wt - a[5])
        res["ranks"][rk] = d
    if not res["ranks"]:
        print(f"OPPROF {label}: no complete profile table in the window"); return 1
    summ = {}
    waits = {}
    for rk, d in res["ranks"].items():
        byc = defaultdict(float)
        for c, ms, op, g, w, wt in d.values(): byc[cls(op)] += ms
        tot = sum(byc.values())
        summ[rk] = (tot, dict(byc))
        waits[rk] = sum(x[5] for x in d.values())
    out = [f"OPPROF {label}: window device time a rank (ms, share):"]
    for rk, (tot, byc) in sorted(summ.items()):
        out.append(f" rank {rk} {tot:.1f} ms + lane waits {waits[rk]:.1f} ms = " + ", ".join(f"{k} {v:.1f} ({100 * v / tot:.0f} %)" for k, v in sorted(byc.items(), key=lambda kv: -kv[1])))
    # EXL3 ops on rank 0 (or the first rank): by op, then by layer with us a call
    rk0 = sorted(res["ranks"])[0]
    d0 = res["ranks"][rk0]
    byop = defaultdict(lambda: [0, 0.0])
    for c, ms, op, g, w, wt in d0.values():
        if cls(op) == "exl3_expert": byop[op][0] += c; byop[op][1] += ms
    out.append(f" | rank {rk0} EXL3 ops: " + ", ".join(f"{op} {v[1]:.1f} ms / {v[0]} calls = {1000 * v[1] / v[0]:.1f} us" for op, v in sorted(byop.items(), key=lambda kv: -kv[1][1])))
    per = sorted(((1000 * ms / c, op, re.sub(r"\D*(\d+).*", r"\1", w) if re.search(r"\d", w) else w, g) for c, ms, op, g, w, wt in d0.values() if cls(op) == "exl3_expert"), key=lambda x: (x[1], x[0]))
    for op in sorted(byop):
        us = [p[0] for p in per if p[1] == op]
        if not us: continue
        us.sort()
        out.append(f" | {op} us a call by layer: min {us[0]:.1f} p25 {us[len(us) // 4]:.1f} p50 {us[len(us) // 2]:.1f} p75 {us[3 * len(us) // 4]:.1f} max {us[-1]:.1f}"
                   f" (slowest layers: {', '.join(p[2] + ' ' + format(p[0], '.0f') for p in sorted((p for p in per if p[1] == op), key=lambda x: -x[0])[:5])})")
    top = sorted(d0.values(), key=lambda x: -x[1])[:8]
    out.append(f" | rank {rk0} top ops: " + "; ".join(f"{op} {ms:.1f} ms {1000 * ms / c:.1f} us x{c} [{g[:50]}]" for c, ms, op, g, w, wt in top))
    print("".join(out))
    if jout:
        with open(jout, "w") as f:
            json.dump({"label": label, "summary": {str(k): v for k, v in summ.items()}, "lane_wait_ms": {str(k): v for k, v in waits.items()},
                       "rank0_exl3_by_op": {k: v for k, v in byop.items()}}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
