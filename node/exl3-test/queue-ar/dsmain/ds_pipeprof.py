#!/usr/bin/env python3
"""ds_pipeprof.py --a LOG FROM TO --b LOG FROM TO [--sa STA STB] [--sb STA STB] [--label L] [--json OUT] [--table OUT]
(10.11.2026, queue-ar/dsmain, ds41-pipe agent; for dsm-pipeprof.sh) -- the pipelined prefill (B, RADIANCE_DS41_PIPE=1) against
the serial step (A) from two --profile-ops runs, op instance by op instance, ms a prefill step a rank.

Parsing = ds_opprof.py's (Ctx::dump_profile tables, cumulative since start; a window is a rank's last complete table at or
before TO minus its last at or before FROM). Steps in a window = the calls of the rank's embedding lookup (one a pass). Each
op instance gets a ROLE from its op, geometry, first weight and -- for the collectives -- the nearest preceding
non-collective op in index order (an all_reduce after attn_output_b is the attention's, after the MoE's ops the MoE's, the
first one the embedding's; `overlap=1` in the geometry is the lane form B declares; an all_gather with `row=` is the
router's split, without it the indexer's). Roles are grouped as B runs them:
  beside   ops lane 0 runs while a lane-1 all-reduce may be in flight: the output projection, mix_a, the MoE pass (router
           selection, sort, EXL3 experts, shared expert, gather), the next post + collapse + norm
  control  ops no lane-1 work runs beside in B (attention core, indexer, q / kv / compressor / indexer GEMMs, Engram,
           mix_f, the ffn post + collapse + norm, the router's own GEMM): their B / A should be 1.00
  ar       the all-reduces (serial in A, the lane form in B) and the gathers (alone on lane 1 in B)
For every role: ms a step A -> B (B / A), calls a step, and lane waits ('wait us': the op's lane idle before it started --
joins, stream waits, the stager's copy waits before a layer's first expert op). For the all-reduces also each instance's
floor (the fastest rank's us a call) and skew (mean - floor: waiting for the slowest rank). With --sa / --sb (/stats around
each window): rank 0's wall ms a prefill step, staged GB a step, link h2d GB a step, AR calls and GB a step.
PROFILED: synchronised steps, event pairs around every launch; shares and ratios are the reading, not a speed.
Standard library only."""
import json, re, sys
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HDR = re.compile(r"per-op timing, rank (\d+)")


def tables(path):
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
                cur[int(r.group(9))] = (int(r.group(1)), float(r.group(2)), r.group(7), r.group(8).strip(), r.group(10),
                                        float(r.group(6)) * int(r.group(1)) / 1000.0)
            elif l.strip().startswith("total"):
                out.append((n, rank, cur)); cur = None
    return out


def window(path, lo, hi):
    T = tables(path)
    res, full = {}, {}
    for rk in sorted({r for _, r, _ in T}):
        mine = [(n, t) for n, r, t in T if r == rk]
        before = [t for n, t in mine if n <= lo]
        after = [t for n, t in mine if n <= hi]
        if not after: continue
        A = before[-1] if before else {}
        B = after[-1]
        full[rk] = mine[-1][1]
        d = {}
        for idx, (c, ms, op, g, w, wt) in B.items():
            p = A.get(idx, (0, 0.0, "", "", "", 0.0))
            if c - p[0] > 0: d[idx] = (c - p[0], ms - p[1], op, g, w, wt - p[5])
        if d: res[rk] = d
    return res, full


COLL = ("all_reduce", "all_gather", "moe_gather_all_reduce")


def roles(full):
    """index -> role, from one rank's complete op list (index order)."""
    out, prev = {}, None
    for idx in sorted(full):
        c, ms, op, g, w, wt = full[idx]
        if op in COLL:
            if op == "all_gather":
                out[idx] = "gather_router" if "row=" in g else "gather_sel"
            elif op == "moe_gather_all_reduce":
                out[idx] = "gather_ar_fused"
            else:
                lane = "overlap=1" in g
                if prev is None or prev[0].startswith("embed_lookup"):
                    k = "ar_embed"
                elif "attn_output_b" in prev[2]:
                    k = "ar_attn"
                else:
                    k = "ar_moe"
                out[idx] = k + ("_lane" if lane else ("" if k == "ar_embed" else "_serial"))
            continue
        out[idx] = role(op, g, w)
        prev = (op, g, w)
    return out


def role(op, g, w):
    if op.startswith("moe_exl3_gate_up"): return "exp_gate_up"
    if op.startswith("moe_exl3_down"): return "exp_down"
    if op.startswith("moe_gemm"): return "exp_other"
    if op == "router_scored_topk" or op.startswith("router_"): return "router_select"
    if op.startswith("moe_scatter"): return "moe_scatter"
    if op.startswith("moe_gather"): return "moe_gather"
    if op.startswith("swiglu"): return "shared_swiglu"
    if op == "gemm_nt_f32":
        return "router_part" if "glm_router" in w else ("comp_gemm" if "compressor" in w else "gemm_f32_other")
    if op.startswith("gemm") or op.startswith("logits"):
        for key, r in (("shexp", "shared_gemm"), ("attn_output_a", "o_a"), ("attn_output_b", "o_b"),
                       ("indexer", "idx_gemm"), ("attn_q_a", "q_a"), ("attn_q_b", "q_b"), ("attn_kv", "kv_gemm"),
                       ("compressor", "comp_gemm"), ("engram_kv", "engram_gemm"), ("output.weight", "head")):
            if key in w: return r
        return "gemm_other"
    if op == "mhc_mix": return "mix_a" if "hc_attn" in w else ("mix_f" if "hc_ffn" in w else "mix_other")
    if op == "mhc_post_collapse_rmsnorm":
        return "pcn_f" if "ffn_norm" in w else ("pcn_a" if "attn_norm" in w else "pcn_fin")
    if op == "mhc_collapse_rmsnorm": return "cn"
    if op.startswith("mhc_"): return op
    if op.startswith("index_"): return "indexer"
    if "attn" in op or op.startswith(("kv_store", "entry_store", "ds41_rope", "rope_rows", "rmsnorm_rope", "qsa_",
                                      "softmax_pool", "quant_roundtrip")): return "attention"
    if op.startswith(("embed_lookup", "ngram", "token_map", "stream_gate", "cast")): return "embed_engram"
    if op == "fill": return "fill"
    if op in ("rmsnorm", "rope_table", "gather_rows"): return "norm_elem"
    return "other:" + op


BESIDE = ["o_a", "o_b", "mix_a", "router_select", "moe_scatter", "exp_gate_up", "exp_down", "shared_gemm",
          "shared_swiglu", "moe_gather", "fill", "pcn_a", "pcn_fin"]
CONTROL = ["attention", "indexer", "q_a", "q_b", "kv_gemm", "idx_gemm", "comp_gemm", "engram_gemm", "embed_engram",
           "mix_f", "pcn_f", "cn", "mhc_post", "router_part", "head", "gemm_other", "norm_elem"]
AR = ["ar_embed", "ar_attn_serial", "ar_attn_lane", "ar_moe_serial", "ar_moe_lane", "gather_router", "gather_sel",
      "gather_ar_fused"]


def per_role(res, rl):
    """rank -> role -> [ms, calls, wait_ms], and rank -> steps."""
    out, steps = {}, {}
    for rk, d in res.items():
        acc = defaultdict(lambda: [0.0, 0, 0.0])
        emb = [c for idx, (c, ms, op, g, w, wt) in d.items() if op.startswith("embed_lookup") and "per_layer" not in w
               and "token_embd" in w]
        steps[rk] = max(emb) if emb else d[min(d)][0]
        for idx, (c, ms, op, g, w, wt) in d.items():
            r = rl.get(idx, role(op, g, w))
            a = acc[r]; a[0] += ms; a[1] += c; a[2] += wt
        out[rk] = acc
    return out, steps


def ar_floor(res, rl, want):
    """the instances of roles `want`: mean over instances of floor us a call (min over ranks) and skew (mean - min)."""
    inst = defaultdict(dict)
    for rk, d in res.items():
        for idx, (c, ms, op, g, w, wt) in d.items():
            if rl.get(idx) in want and c > 0: inst[idx][rk] = 1000.0 * ms / c
    fl, sk = [], []
    for idx, v in inst.items():
        if len(v) < 2: continue
        m = min(v.values()); fl.append(m); sk.append(sum(v.values()) / len(v) - m)
    return (sum(fl) / len(fl) if fl else None, sum(sk) / len(sk) if sk else None, len(fl))


def stats(a, b):
    try:
        A, B = json.load(open(a)), json.load(open(b))
    except Exception:
        return None
    ds = B.get("steps", 0) - A.get("steps", 0)
    if ds <= 0: return None
    pm = (B.get("steps_prefill", 0) + B.get("steps_mixed", 0)) - (A.get("steps_prefill", 0) + A.get("steps_mixed", 0))
    ea, eb, la, lb = A.get("experts", {}), B.get("experts", {}), A.get("link", {}), B.get("link", {})
    G = 1e9
    return {"steps": ds, "prefill_steps": pm, "ms_step": (B.get("step_ns", 0) - A.get("step_ns", 0)) / ds / 1e6,
            "staged_GB_step": (eb.get("staged_bytes", 0) - ea.get("staged_bytes", 0)) / ds / G,
            "stage_copies_step": (eb.get("stage_copies", 0) - ea.get("stage_copies", 0)) / ds,
            "stage_overflow": eb.get("stage_overflow", 0) - ea.get("stage_overflow", 0),
            "h2d_GB_step": (lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)) / ds / G,
            "ar_calls_step": (lb.get("ar_calls", 0) - la.get("ar_calls", 0)) / ds,
            "ar_GB_step": (lb.get("ar_bytes", 0) - la.get("ar_bytes", 0)) / ds / G,
            "hit_pct": 100.0 * (eb.get("routed_res", 0) - ea.get("routed_res", 0)) / max(1, eb.get("routed", 0) - ea.get("routed", 0))}


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return 2
    arg = {"label": "", "json": None, "table": None, "sa": None, "sb": None}
    i = 0
    while i < len(a):
        k = a[i]
        if k in ("--a", "--b"): arg[k[2:]] = (a[i + 1], int(a[i + 2]), int(a[i + 3])); i += 4
        elif k in ("--sa", "--sb"): arg[k[2:]] = (a[i + 1], a[i + 2]); i += 3
        elif k in ("--label", "--json", "--table"): arg[k[2:]] = a[i + 1]; i += 2
        else: i += 1
    R, S, RL, RES = {}, {}, {}, {}
    for arm in ("a", "b"):
        res, full = window(*arg[arm])
        RES[arm] = res
        if not res:
            print(f"DSPIPEPROF {arg['label']}: no complete profile table in arm {arm.upper()}'s window (PROFILED)"); return 1
        rk0 = min(full)
        RL[arm] = roles(full[rk0])
        R[arm], S[arm] = per_role(res, RL[arm])
    ranks = sorted(set(R["a"]) & set(R["b"]))
    allroles = sorted(set(r for arm in R for rk in R[arm] for r in R[arm][rk]))

    def ms(arm, r, rk=None, k=0):   # a role's ms (k 0), calls (1) or wait ms (2) a step, one rank or the ranks' mean
        rs = [rk] if rk is not None else ranks
        v = [R[arm][x][r][k] / max(1, S[arm][x]) if r in R[arm][x] else 0.0 for x in rs]
        return sum(v) / len(v)

    def grp(arm, names, k=0, rk=None): return sum(ms(arm, r, rk, k) for r in names)
    lines = []
    for r in BESIDE + CONTROL + AR + [x for x in allroles if x not in BESIDE + CONTROL + AR]:
        if not any(r in R[arm][x] for arm in R for x in R[arm]): continue
        ma, mb = ms("a", r), ms("b", r)
        per = " ".join(f"r{x} {ms('a', r, x):.2f}->{ms('b', r, x):.2f}" for x in ranks)
        lines.append(f"{r:16s} A {ma:8.2f}  B {mb:8.2f} ms/step  {('x%.3f' % (mb / ma)) if ma > 0.005 else '   -  ':>7s}  "
                     f"calls {ms('a', r, k=1):6.1f} -> {ms('b', r, k=1):6.1f}  wait {ms('a', r, k=2):6.2f} -> {ms('b', r, k=2):6.2f}  [{per}]")
    dev = {arm: sum(ms(arm, r) for r in allroles) for arm in R}
    wt = {arm: sum(ms(arm, r, k=2) for r in allroles) for arm in R}
    # lane 0 = every role but the collectives B issues on lane 1 (in A those run on lane 0 too)
    l0 = {arm: [r for r in allroles if arm == "a" or r not in AR] for arm in R}
    l0dev = {arm: sum(ms(arm, r) for r in l0[arm]) for arm in R}
    l0wt = {arm: sum(ms(arm, r, k=2) for r in l0[arm]) for arm in R}
    l1dev = sum(ms("b", r) for r in AR)
    l1wt = sum(ms("b", r, k=2) for r in AR)
    ex = [r for r in allroles if r not in BESIDE + CONTROL + AR]
    st = {arm: (stats(*arg["s" + arm]) if arg["s" + arm] else None) for arm in R}
    fa = {k: ar_floor(RES[arm], RL[arm], {want}) for k, (arm, want) in {
        "attn_serial": ("a", "ar_attn_serial"), "moe_serial": ("a", "ar_moe_serial"),
        "attn_lane": ("b", "ar_attn_lane"), "moe_lane": ("b", "ar_moe_lane")}.items()}
    o = [f"DSPIPEPROF {arg['label']} (PROFILED, not a speed; ranks' mean, ms a prefill step; steps A {S['a']} B {S['b']}):"]
    for arm in ("a", "b"):
        s = st[arm]
        o.append(f" | {arm.upper()}: lane 0 device {l0dev[arm]:.1f} + waits {l0wt[arm]:.1f} ms"
                 + (f", lane 1 (the collectives) device {l1dev:.1f} + waits {l1wt:.1f} ms" if arm == "b" else " (every op on lane 0)")
                 + (f"; /stats step_ns {s['ms_step']:.1f} ms a step (not wall: steps overlap their issue) over {s['steps']} steps ({s['prefill_steps']} prefill), rank 0 staged "
                    f"{s['staged_GB_step']:.2f} GB / {s['stage_copies_step']:.0f} copies a step (overflow {s['stage_overflow']}), "
                    f"h2d (all ranks) {s['h2d_GB_step']:.2f} GB a step, rank 0 AR {s['ar_calls_step']:.0f} calls {s['ar_GB_step']:.2f} GB "
                    f"a step, rank 0 expert hit {s['hit_pct']:.1f} %" if s else "")
                 + f"; first expert op's lane wait (joins + the stager's copy waits) {ms(arm, 'exp_gate_up', k=2):.1f} ms a step")
    o.append(f" | beside (lane 0 while a lane-1 AR may run in B): A {grp('a', BESIDE):.1f} -> B {grp('b', BESIDE):.1f} "
             f"(x{grp('b', BESIDE) / max(1e-9, grp('a', BESIDE)):.3f}), waits {grp('a', BESIDE, 2):.1f} -> {grp('b', BESIDE, 2):.1f}"
             f"; of it experts {grp('a', ['exp_gate_up', 'exp_down']):.1f} -> {grp('b', ['exp_gate_up', 'exp_down']):.1f} "
             f"(x{grp('b', ['exp_gate_up', 'exp_down']) / max(1e-9, grp('a', ['exp_gate_up', 'exp_down'])):.3f}; fn4 one-card "
             f"micro: 2 x 2048 rows = x1.12 of one 4096 pass), dense o_a+o_b+shared {grp('a', ['o_a', 'o_b', 'shared_gemm']):.1f} -> "
             f"{grp('b', ['o_a', 'o_b', 'shared_gemm']):.1f}, mhc mix_a+pcn_a {grp('a', ['mix_a', 'pcn_a', 'pcn_fin']):.1f} -> "
             f"{grp('b', ['mix_a', 'pcn_a', 'pcn_fin']):.1f}")
    o.append(f" | control (nothing beside in B; expect x1.00): A {grp('a', CONTROL):.1f} -> B {grp('b', CONTROL):.1f} "
             f"(x{grp('b', CONTROL) / max(1e-9, grp('a', CONTROL)):.3f}), waits {grp('a', CONTROL, 2):.1f} -> {grp('b', CONTROL, 2):.1f}")
    o.append(f" | all-reduce kernel time: attention serial {ms('a', 'ar_attn_serial'):.1f} -> lane {ms('b', 'ar_attn_lane'):.1f}, "
             f"MoE serial {ms('a', 'ar_moe_serial'):.1f} -> lane {ms('b', 'ar_moe_lane'):.1f}, embedding {ms('a', 'ar_embed'):.1f} -> "
             f"{ms('b', 'ar_embed'):.1f}; floor / skew us a call (instances): "
             + ", ".join(f"{k} {v[0]:.0f} / {v[1]:.0f} ({v[2]})" for k, v in fa.items() if v[0] is not None))
    o.append(f" | gathers alone on lane 1 in B: router {ms('a', 'gather_router'):.2f} -> {ms('b', 'gather_router'):.2f} "
             f"(calls {ms('a', 'gather_router', k=1):.0f} -> {ms('b', 'gather_router', k=1):.0f}), indexer {ms('a', 'gather_sel'):.2f} -> "
             f"{ms('b', 'gather_sel'):.2f}; extra ops: moe_scatter {ms('a', 'moe_scatter'):.2f} -> {ms('b', 'moe_scatter'):.2f} "
             f"(calls {ms('a', 'moe_scatter', k=1):.0f} -> {ms('b', 'moe_scatter', k=1):.0f}: 2 pieces + the whole-step count), "
             f"router part {ms('a', 'router_part'):.2f} -> {ms('b', 'router_part'):.2f}")
    top = sorted(allroles, key=lambda r: -(ms("b", r, k=2) - ms("a", r, k=2)))[:8]
    o.append(" | lane-wait growth B - A (ms a step, by the role that waited): "
             + ", ".join(f"{r} {ms('a', r, k=2):.1f}->{ms('b', r, k=2):.1f}" for r in top))
    if ex: o.append(" | other roles: " + ", ".join(f"{r} {ms('a', r):.2f}->{ms('b', r):.2f}" for r in ex))
    print("".join(o))
    if arg["table"]:
        with open(arg["table"], "w") as f:
            f.write("\n".join(o) + "\n\n" + "\n".join(lines) + "\n")
    if arg["json"]:
        with open(arg["json"], "w") as f:
            json.dump({"label": arg["label"], "profiled": True, "steps": S, "stats": st, "device": dev, "waits": wt,
                       "lane0_device": l0dev, "lane0_waits": l0wt, "lane1_device_b": l1dev, "lane1_waits_b": l1wt,
                       "roles": {arm: {str(rk): {r: v for r, v in R[arm][rk].items()} for rk in R[arm]} for arm in R},
                       "ar_floor_skew": fa}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
