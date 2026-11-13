#!/usr/bin/env python3
"""z0prof.py (11.11.2026, queue-ar dsmain, ds-main-port Z0 agent; item 57-p4zyzzzzz-ds-hy-z0prof's analyser, body
dsm-z0prof.sh) -- lever Z0 (depth-0 parity): where a depth-0 step of the NST 5 stack2 server spends the time an NST 0 step of
the stack2 NST 0 server does not. Standard library only; imports ds_opprof_draft.tables (read-only, beside this file).

  z0prof.py ar ENGINE_LOG --win NAME:L0:L1 [--win ...] [--json OUT] [--label L]
      --profile-ops tables (cumulative; each window = engine-log line numbers, the last complete table of a rank at or
      before L0 subtracted from its last at or before L1). Every collective op instance (all_reduce*, moe_gather_all_reduce,
      all_gather*) per rank: us a call; the instance's class by the op before it in rank 0's order (attn = after
      attn_output_b, moe = after the MoE / the fused gather, other); per class and rank ms a step; floor = the fastest
      rank's time (the last to arrive waits least: ~ the transfer), above-floor = waiting for the slowest rank (lower bound);
      per rank: how often it is the fastest at an instance (= arrives last = the straggler). PROFILED (synchronised ops).
  z0prof.py opscmp --a OPS_A.json:WIN --b OPS_B.json:WIN [--json OUT] [--label L]
      Two d0attrprof.py ops JSONs (A = NST 5 depth 0, B = NST 0): per class the trunk's ms a step per rank and the rank mean,
      A - B, sorted by |difference|; A's drafter ops ms a step and sampling; rank 0's trunk ops A vs B by name (largest
      differences). PROFILED.
  z0prof.py summary --wins-n5 F.jsonl --wins-n0 F.jsonl [--stamps-c4 F.json] [--stamps-c8 F.json] [--opscmp-c4 F.json]
                    [--opscmp-c8 F.json] [--ar-n5 F.json] [--ar-n0 F.json] [--slope-c4 1.3] [--slope-c8 2.0] [--json OUT]
      The ranking: per load (c4, c8) the clean depth-0 window (draft_depth0 2, draft_probe 4096) against the NST 0 window
      (/stats ms a decode step, tok/s, tok a step; stamped both), the natural windows (stack2's pin: probes 16 + back-off 3,
      D7 choosing) against both, then per candidate the ms a step from each reading: stamps (card 0: trunk kernel classes
      a pass, the other pass kinds a step, card idle / between passes) and the profile (every rank, by class; the AR
      per rank), /stats (hit, missed, stream, VRAM units). Ranked by the stamped ms where there is one, else the profiled.
      Prints "Z0 rank ..." lines and "Z0 summary ...".
  z0prof.py selftest
      Synthetic tables / windows through ar, opscmp and summary (CPU, no files written outside a temp dir)."""
import json, math, os, re, statistics, sys, tempfile
from collections import defaultdict

AR_OP = re.compile(r"^(all_reduce|moe_gather_all_reduce|all_gather|ar_|reduce_scatter)")
# kernel families of the stamps (card 0) into classes; AR first: the fused gather + all-reduce kernel names 'moe' too
FAM_CLASSES = [("ar", re.compile(r"_ar_|all_?reduce|allreduce|twoshot|oneshot|all_?gather", re.I)),
               ("experts", re.compile(r"exl3|moe_gemm|expert", re.I)),
               ("moe_mix", re.compile(r"^mix_|router|scatter|swiglu", re.I)),
               ("fill", re.compile(r"^fill", re.I)),
               ("attention", re.compile(r"attn|rope|kv_store|qsa|softmax_pool|latent|index", re.I)),
               ("dense", re.compile(r"gemm|gemv|gg_mr|gg_", re.I)),
               ("norm_elem", re.compile(r"norm|cast|quant|collapse|mhc|contract", re.I))]


def famclass(f):
    for c, rx in FAM_CLASSES:
        if rx.search(f): return c
    return "other"


def wins(a, flag="--win"):
    out = []
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            n, lo, hi = a[i + 1].split(":")
            out.append((n, int(lo), int(hi)))
    return out


def arg(a, k, d=None):
    return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d


def jload(p):
    try: return json.load(open(p))
    except Exception: return None


def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []


def f(v, n=2):
    return "-" if v is None else format(v, ".%df" % n)


def pct(x, y):
    return None if x is None or not y else 100.0 * (x / y - 1.0)


def fp(v):
    return "-" if v is None else "%+.1f %%" % v


# ------------------------------------------------------------------------------------------------------------- ar
def _tables(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ds_opprof_draft import tables
    return tables(path)


def window_delta(T, lo, hi):
    """{rank: {idx: (calls, ms, op, geom, w, wait_ms)}} over the window."""
    out = {}
    for rk in sorted({r for _, r, _ in T}):
        mine = [(n, t) for n, r, t in T if r == rk]
        before = [t for n, t in mine if n <= lo]
        after = [t for n, t in mine if n <= hi]
        if not after: continue
        A, B = (before[-1] if before else {}), after[-1]
        d = {}
        for idx, (c, ms, op, geom, w, wt) in B.items():
            p = A.get(idx, (0, 0.0, "", "", "", 0.0))
            if c - p[0] > 0: d[idx] = (c - p[0], ms - p[1], op, geom, w, wt - p[5])
        if d: out[rk] = d
    return out


def ar(a, T=None):
    path = a[1]
    T = T if T is not None else _tables(path)
    out = {}
    for wn, lo, hi in wins(a):
        D = window_delta(T, lo, hi)
        if not D or 0 not in D: out[wn] = {"error": "no complete profile table on rank 0 in the window"}; continue
        heads = [i for i, x in D[0].items() if x[2].startswith("logits_gemm")]
        hd = min(heads) if heads else 1 << 30
        steps = D[0][min(D[0])][0]
        n = max(1, steps)
        order = sorted(D[0].items())
        prev, last = {}, None
        for idx, x in order:
            if AR_OP.match(x[2]): prev[idx] = last
            else: last = (x[2], x[4])
        inst = [idx for idx, x in order if AR_OP.match(x[2]) and idx <= hd]   # the trunk's collectives
        dinst = [idx for idx, x in order if AR_OP.match(x[2]) and idx > hd]   # the drafter's (after the head)
        def klass(idx):
            if D[0][idx][2].startswith("moe_gather_all_reduce"): return "moe(gather_ar)"
            p = prev.get(idx)
            if p is None: return "embed"
            if "attn_output_b" in (p[1] or ""): return "attn"
            if p[0].startswith("moe") or "ffn" in (p[1] or "") or "shexp" in (p[1] or ""): return "moe"
            return "other:" + p[0]
        ranks = sorted(D)
        agg = defaultdict(lambda: {"n": 0, "mean_ms": 0.0, "floor_ms": 0.0, "max_ms": 0.0, "rank_ms": defaultdict(float),
                                   "last": defaultdict(int)})
        for idx in inst + dinst:
            v = {}
            for r in ranks:
                x = D[r].get(idx)
                if x and x[0] > 0: v[r] = x[1] / n      # ms a step on rank r
            if len(v) < len(ranks): continue
            c = ("draft:" if idx in dinst else "") + klass(idx)
            g = agg[c]; g["n"] += 1
            g["mean_ms"] += statistics.mean(v.values()); g["floor_ms"] += min(v.values()); g["max_ms"] += max(v.values())
            for r, ms in v.items(): g["rank_ms"][r] += ms
            g["last"][min(v, key=v.get)] += 1
        res = {}
        tot = {"n": 0, "mean_ms": 0.0, "floor_ms": 0.0, "max_ms": 0.0, "rank_ms": defaultdict(float), "last": defaultdict(int)}
        for c, g in agg.items():
            res[c] = {"n": g["n"], "mean_ms": g["mean_ms"], "floor_ms": g["floor_ms"], "above_floor_ms": g["mean_ms"] - g["floor_ms"],
                      "max_ms": g["max_ms"], "rank_ms": {str(r): v for r, v in g["rank_ms"].items()},
                      "last_arrival": {str(r): v for r, v in g["last"].items()}}
            if not c.startswith("draft:"):
                for k in ("n", "mean_ms", "floor_ms", "max_ms"): tot[k] += g[k]
                for r, v in g["rank_ms"].items(): tot["rank_ms"][r] += v
                for r, v in g["last"].items(): tot["last"][r] += v
        res["TRUNK"] = {"n": tot["n"], "mean_ms": tot["mean_ms"], "floor_ms": tot["floor_ms"], "above_floor_ms": tot["mean_ms"] - tot["floor_ms"],
                        "max_ms": tot["max_ms"], "rank_ms": {str(r): v for r, v in tot["rank_ms"].items()},
                        "last_arrival": {str(r): v for r, v in tot["last"].items()}}
        # per rank: the trunk's non-collective device ms a step (what a rank does before it arrives) -> the skew's source
        work = {}
        for r in ranks:
            w = sum(x[1] for i, x in D[r].items() if i <= hd and not AR_OP.match(x[2])) / n
            e = sum(x[1] for i, x in D[r].items() if i <= hd and (x[2].startswith("moe_exl3") or "exl3" in x[2].lower() or x[2].startswith("moe_gemm"))) / n
            work[str(r)] = {"work_ms": w, "experts_ms": e}
        out[wn] = {"steps": steps, "classes": res, "rank_work": work}
    return out


def fmt_ar(o):
    L = []
    for wn, x in o.items():
        if "error" in x: L.append(f"{wn}: {x['error']}"); continue
        parts = []
        for c, g in sorted(x["classes"].items(), key=lambda kv: (kv[0] != "TRUNK", -kv[1]["mean_ms"])):
            parts.append(f"{c} n {g['n']}: mean {g['mean_ms']:.2f} floor {g['floor_ms']:.2f} above-floor {g['above_floor_ms']:.2f} ms a step; per rank "
                         + " ".join(f"r{r} {v:.2f}" for r, v in sorted(g["rank_ms"].items()))
                         + "; fastest (arrives last) " + " ".join(f"r{r} {v}" for r, v in sorted(g["last_arrival"].items())))
        wk = " ".join(f"r{r} {v['work_ms']:.2f} (experts {v['experts_ms']:.2f})" for r, v in sorted(x["rank_work"].items()))
        L.append(f"{wn}: {x['steps']} steps (PROFILED) | " + " | ".join(parts) + f" | non-collective trunk work a step per rank: {wk}")
    return L


# ------------------------------------------------------------------------------------------------------------- opscmp
def opscmp(a):
    pa, wa = arg(a, "--a").rsplit(":", 1); pb, wb = arg(a, "--b").rsplit(":", 1)
    A, B = (jload(pa) or {}).get(wa) or {}, (jload(pb) or {}).get(wb) or {}
    if "ranks" not in A or "ranks" not in B:
        return {"error": "missing window %s in %s or %s in %s" % (wa, pa, wb, pb)}
    RA, RB = A["ranks"], B["ranks"]
    ranks = sorted(set(RA) & set(RB))
    classes = sorted({k for r in ranks for k in RA[r]["by_class_ms_step"]} | {k for r in ranks for k in RB[r]["by_class_ms_step"]})
    cmp = {}
    for k in classes:
        if k.startswith("draft_") or k == "sample": continue
        ha = {r: RA[r]["by_class_ms_step"].get(k, 0.0) for r in ranks}; hb = {r: RB[r]["by_class_ms_step"].get(k, 0.0) for r in ranks}
        cmp[k] = {"a_mean": statistics.mean(ha.values()), "b_mean": statistics.mean(hb.values()),
                  "delta_mean": statistics.mean(ha.values()) - statistics.mean(hb.values()),
                  "per_rank": {r: [ha[r], hb[r], ha[r] - hb[r]] for r in ranks}}
    trunk = {r: [RA[r]["trunk_ms_step"], RB[r]["trunk_ms_step"]] for r in ranks}
    dev = {r: [RA[r]["device_ms_step"], RB[r]["device_ms_step"], RA[r]["wait_ms_step"], RB[r]["wait_ms_step"]] for r in ranks}
    drafter = {r: RA[r]["drafter_ms_step"] for r in ranks}
    samp = {r: [RA[r]["by_class_ms_step"].get("sample", 0.0), RB[r]["by_class_ms_step"].get("sample", 0.0)] for r in ranks}
    r0a, r0b = RA.get("0", {}).get("trunk_ops", {}), RB.get("0", {}).get("trunk_ops", {})
    ops = []
    for k in set(r0a) | set(r0b):
        x, y = r0a.get(k, {}), r0b.get(k, {})
        ops.append((k, x.get("ms_step", 0.0), y.get("ms_step", 0.0), x.get("calls_step", 0.0), y.get("calls_step", 0.0)))
    ops.sort(key=lambda t: -abs(t[1] - t[2]))
    dro = sorted(((k, v["ms_step"], v["calls_step"], v["us_call"]) for k, v in RA.get("0", {}).get("drafter_ops", {}).items()), key=lambda t: -t[1])
    return {"ranks": ranks, "steps": [RA[ranks[0]]["steps"], RB[ranks[0]]["steps"]] if ranks else None, "classes": cmp,
            "trunk_ms": trunk, "device": dev, "drafter_ms": drafter, "sample_ms": samp, "rank0_ops": ops[:20], "drafter_ops_r0": dro[:12]}


def fmt_opscmp(o, label):
    if "error" in o: return [o["error"]]
    L = []
    cl = sorted(o["classes"].items(), key=lambda kv: -abs(kv[1]["delta_mean"]))
    L.append("classes (PROFILED ms a step, rank mean A/B (A-B) [per rank A-B]): " + "; ".join(
        f"{k} {v['a_mean']:.2f}/{v['b_mean']:.2f} ({v['delta_mean']:+.2f}) [" + " ".join(f"r{r} {d[2]:+.2f}" for r, d in sorted(v["per_rank"].items())) + "]"
        for k, v in cl if max(v["a_mean"], v["b_mean"]) >= 0.03))
    L.append("trunk ms a step per rank A/B: " + " ".join(f"r{r} {v[0]:.2f}/{v[1]:.2f} ({v[0] - v[1]:+.2f})" for r, v in sorted(o["trunk_ms"].items()))
             + " | device + lane waits A/B: " + " ".join(f"r{r} {v[0]:.2f}+{v[2]:.2f} / {v[1]:.2f}+{v[3]:.2f}" for r, v in sorted(o["device"].items()))
             + " | drafter ops A: " + " ".join(f"r{r} {v:.3f}" for r, v in sorted(o["drafter_ms"].items()))
             + " | sampling A/B: " + " ".join(f"r{r} {v[0]:.3f}/{v[1]:.3f}" for r, v in sorted(o["sample_ms"].items()))
             + " | steps A/B %s" % o["steps"])
    L.append("rank 0 trunk ops A/B (ms a step; calls a step), largest differences: " + "; ".join(
        f"{k} {x:.3f}/{y:.3f} ({x - y:+.3f}; {cx:.1f}/{cy:.1f})" for k, x, y, cx, cy in o["rank0_ops"][:16]))
    if o["drafter_ops_r0"]:
        L.append("rank 0 drafter ops at depth 0 (ms a step, calls a step, us a call): " + "; ".join(
            f"{k} {ms:.3f} ({c:.2f}, {u:.0f})" for k, ms, c, u in o["drafter_ops_r0"]))
    return L


# ------------------------------------------------------------------------------------------------------------- summary
def win(rs, label):
    r = [x for x in rs if x.get("label") == label]
    return r[-1] if r else None


def wv(w, *p):
    x = w
    for k in p:
        if not isinstance(x, dict): return None
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def stamp_classes(x):
    """A stamps window (d0attrprof.py analyse) -> trunk kernel classes (device us a pass), period / idle / between, other kinds."""
    if not x or "error" in x: return None
    t = x["kinds"][str(x["trunk"])]
    cls = defaultdict(float)
    for fm, us in t["fam_us"].items(): cls[famclass(fm)] += us
    oth = {k: v for k, v in (x.get("other_by_kind") or {}).items()}
    return {"period_ms": (x.get("period_us_mean") or 0) / 1e3, "trunk_wall_ms": (x.get("trunk_wall_us_mean") or 0) / 1e3,
            "trunk_busy_ms": (x.get("trunk_busy_us_mean") or 0) / 1e3, "idle_ms": (x.get("idle_us_mean") or 0) / 1e3,
            "between_ms": (x.get("between_us_mean") or 0) / 1e3, "cls_ms": {k: v / 1e3 for k, v in cls.items()},
            "other_ms": sum((v.get("wall_us_step") or 0) for v in oth.values()) / 1e3,
            "other": {k: [v.get("per_step"), (v.get("wall_us_step") or 0) / 1e3] for k, v in oth.items()},
            "fam_us": t["fam_us"], "fam_n": t["fam_n"], "steps": x.get("steps_kept")}


def summary(a):
    W5, W0 = rows(arg(a, "--wins-n5")), rows(arg(a, "--wins-n0"))
    ST = {"c4": jload(arg(a, "--stamps-c4")) if arg(a, "--stamps-c4") else None,
          "c8": jload(arg(a, "--stamps-c8")) if arg(a, "--stamps-c8") else None}
    OC = {"c4": jload(arg(a, "--opscmp-c4")) if arg(a, "--opscmp-c4") else None,
          "c8": jload(arg(a, "--opscmp-c8")) if arg(a, "--opscmp-c8") else None}
    AR5, AR0 = (jload(arg(a, "--ar-n5")) if arg(a, "--ar-n5") else None), (jload(arg(a, "--ar-n0")) if arg(a, "--ar-n0") else None)
    slope = {"c4": float(arg(a, "--slope-c4", "1.3")), "c8": float(arg(a, "--slope-c8", "2.0"))}
    out, L = {}, []
    for ph in ("c4", "c8"):
        d0, n0 = win(W5, "z0-n5-stamp-%sd0" % ph), win(W0, "z0-n0-stamp-%s" % ph)
        nats = [w for w in (win(W5, "z0-n5-stamp-%snat" % ph), win(W5, "z0-n5-stamp-%snat2" % ph)) if w]
        o = {"d0": {}, "n0": {}, "nat": []}
        for k, w in (("d0", d0), ("n0", n0)):
            if not w: continue
            o[k] = {"ok": w.get("ok"), "tps": wv(w, "mb", "agg_dec_tps"), "ms_mb": wv(w, "mb", "ms_decode_step"), "tok_step": wv(w, "mb", "tok_per_decode_step"),
                    "ms_stats": wv(w, "win", "ms_decode_step"), "zero": wv(w, "win", "zero_share"), "probe": wv(w, "win", "probe_share"),
                    "hit": wv(w, "win", "hit_pct_rank0"), "missed": wv(w, "win", "missed_step_rank0"), "stream": wv(w, "win", "stream_MB_step"),
                    "vram": wv(w, "win", "vram_units_end"), "played": wv(w, "win", "passes_step", "played"),
                    "lent": wv(w, "win", "draft", "lent_units") if wv(w, "win", "draft", "lent_units") is not None else None}
        for w in nats:
            o["nat"].append({"label": w.get("label"), "ok": w.get("ok"), "tps": wv(w, "mb", "agg_dec_tps"), "ms_mb": wv(w, "mb", "ms_decode_step"),
                             "tok_step": wv(w, "mb", "tok_per_decode_step"), "ms_stats": wv(w, "win", "ms_decode_step"),
                             "zero": wv(w, "win", "zero_share"), "probe": wv(w, "win", "probe_share"),
                             "moves": wv(w, "win", "draft", "zero_moves"), "hit": wv(w, "win", "hit_pct_rank0"), "vram": wv(w, "win", "vram_units_end")})
        D, N = o["d0"], o["n0"]
        gap = (D.get("ms_stats") - N.get("ms_stats")) if D.get("ms_stats") and N.get("ms_stats") else None
        L.append(f"{ph} (stamped, /stats): depth 0 forced (draft_depth0 2, P4096) {f(D.get('ms_stats'))} ms a decode step, {f(D.get('tps'), 1)} tok/s, "
                 f"{f(D.get('tok_step'))} tok a step, depth-0 share {f(D.get('zero'), 3)}, probe {f(D.get('probe'), 3)} | NST 0 {f(N.get('ms_stats'))} ms, "
                 f"{f(N.get('tps'), 1)} tok/s, {f(N.get('tok_step'))} tok a step | depth 0 - NST 0 {f(gap)} ms a step ({fp(pct(D.get('ms_stats'), N.get('ms_stats')))}), "
                 f"tok/s {fp(pct(D.get('tps'), N.get('tps')))} | experts rank 0 depth 0 / NST 0: hit {f(D.get('hit'))} / {f(N.get('hit'))} %, missed "
                 f"{f(D.get('missed'))} / {f(N.get('missed'))} a step, stream {f(D.get('stream'), 1)} / {f(N.get('stream'), 1)} MB a step, VRAM units "
                 f"{f(D.get('vram'), 0)} / {f(N.get('vram'), 0)}, passes played a step {f(D.get('played'))} / {f(N.get('played'))}")
        for x in o["nat"]:
            L.append(f"{ph} natural ({x['label']}: stack2's pin, probes 16 + back-off 3, D7 choosing): {f(x['tps'], 1)} tok/s ({fp(pct(x['tps'], D.get('tps')))} vs "
                     f"depth 0 forced, {fp(pct(x['tps'], N.get('tps')))} vs NST 0), {f(x['ms_stats'])} ms a decode step, {f(x['tok_step'])} tok a step, depth-0 share "
                     f"{f(x['zero'], 3)}, probe share {f(x['probe'], 3)}, D7 moves {x['moves']}, hit {f(x['hit'])} %, VRAM units {f(x['vram'], 0)}")
        # the candidates
        cand = []
        s = ST.get(ph) or {}
        sd, sn = stamp_classes(s.get("%sd0" % ph)), stamp_classes(next((v for k, v in s.items() if k.startswith("ref:")), None))
        sdel = None
        if sd and sn:
            sdel = {k: sd["cls_ms"].get(k, 0.0) - sn["cls_ms"].get(k, 0.0) for k in set(sd["cls_ms"]) | set(sn["cls_ms"])}
            L.append(f"{ph} stamps (card 0, both stamped): step period depth 0 {sd['period_ms']:.2f} vs NST 0 {sn['period_ms']:.2f} ms ({sd['period_ms'] - sn['period_ms']:+.2f}); "
                     f"trunk wall {sd['trunk_wall_ms']:.2f} / {sn['trunk_wall_ms']:.2f} ({sd['trunk_wall_ms'] - sn['trunk_wall_ms']:+.2f}); other passes a step "
                     f"{sd['other_ms']:.2f} / {sn['other_ms']:.2f} ms {sd['other']}; card idle {sd['idle_ms']:.2f} / {sn['idle_ms']:.2f}; between passes "
                     f"{sd['between_ms']:.2f} / {sn['between_ms']:.2f} | trunk kernel classes device ms a pass depth 0 / NST 0 (diff): " + ", ".join(
                         f"{k} {sd['cls_ms'].get(k, 0):.2f}/{sn['cls_ms'].get(k, 0):.2f} ({v:+.2f})" for k, v in sorted(sdel.items(), key=lambda kv: -abs(kv[1]))))
            famd = sorted(((k, sd["fam_us"].get(k, 0.0), sn["fam_us"].get(k, 0.0), sd["fam_n"].get(k, 0), sn["fam_n"].get(k, 0))
                           for k in set(sd["fam_us"]) | set(sn["fam_us"])), key=lambda t: -abs(t[1] - t[2]))
            L.append(f"{ph} stamps trunk kernel families depth 0 / NST 0 (device us a pass; dispatches a pass), largest differences: " + ", ".join(
                f"{k} [{famclass(k)}] {x:.0f}/{y:.0f} ({x - y:+.0f}; {nx:.0f}/{ny:.0f})" for k, x, y, nx, ny in famd[:16]))
        oc = OC.get(ph) or {}
        ocl = oc.get("classes") or {}
        def pdel(*ks): return sum(ocl.get(k, {}).get("delta_mean", 0.0) for k in ks) if ocl else None
        a5 = ((AR5 or {}).get("%sd0" % ph) or {}).get("classes", {}).get("TRUNK")
        a0 = ((AR0 or {}).get(ph) or {}).get("classes", {}).get("TRUNK")
        # (name, stamped ms, profiled ms, note)
        if sdel is not None or ocl:
            cand.append(("expert kernels (unlent slots / hit / stream)", sdel.get("experts", 0.0) if sdel else None, pdel("experts"),
                         f"hit {f(D.get('hit'))} vs {f(N.get('hit'))} %, missed {f(D.get('missed'))} vs {f(N.get('missed'))} a step, stream {f(D.get('stream'), 1)} vs {f(N.get('stream'), 1)} MB, VRAM units {f(D.get('vram'), 0)} vs {f(N.get('vram'), 0)}"))
            cand.append(("all-reduce / gather-AR (AR wait, rank skew)", sdel.get("ar", 0.0) if sdel else None, pdel("gather_ar", "collective"),
                         "profiled AR above-floor (waiting) %s vs %s ms a step; per-rank AR depth 0 %s / NST 0 %s" % (
                             f(a5.get("above_floor_ms")) if a5 else "-", f(a0.get("above_floor_ms")) if a0 else "-",
                             " ".join("r%s %.2f" % (r, v) for r, v in sorted((a5 or {}).get("rank_ms", {}).items())) or "-",
                             " ".join("r%s %.2f" % (r, v) for r, v in sorted((a0 or {}).get("rank_ms", {}).items())) or "-")))
            cand.append(("MoE mix / router (moe_other)", sdel.get("moe_mix", 0.0) if sdel else None, pdel("moe_other"), ""))
            cand.append(("fill kernels (buffer clears: arena / verify sizing)", sdel.get("fill", 0.0) if sdel else None, None, "stamps only (profile counts fills in norm_elem)"))
            cand.append(("dense GEMM / GEMV (forms keyed by step shape)", sdel.get("dense", 0.0) if sdel else None, pdel("dense_gemm"), ""))
            cand.append(("attention / indexer / rope / kv", sdel.get("attention", 0.0) if sdel else None, pdel("attention", "indexer"), ""))
            cand.append(("norms / elementwise / mhc", sdel.get("norm_elem", 0.0) if sdel else None, pdel("norm_elem", "mhc", "embed_engram", "other"), ""))
            cand.append(("other pass kinds a step (drafter context / tap / walk)", (sd["other_ms"] - sn["other_ms"]) if sd and sn else None,
                         statistics.mean(oc["drafter_ms"].values()) if oc.get("drafter_ms") else None, "stamps: other pass kinds' wall a step; profile: drafter ops device ms a step"))
            cand.append(("host / between passes (scheduler, hybrid lookups)", (sd["between_ms"] - sn["between_ms"]) if sd and sn else None, None,
                         "card 0 time outside every pass; idle %s vs %s ms" % (f(sd["idle_ms"]) if sd else "-", f(sn["idle_ms"]) if sn else "-")))
        nat = o["nat"]
        if nat and D.get("tps") and N.get("tps"):
            best = max(nat, key=lambda x: x["tps"] or 0)
            worst = min(nat, key=lambda x: x["tps"] or 1e9)
            # probes + D7's choice as ms per emitted token x tokens a forced step: comparable to the per-step rows
            def mspt(t): return 1000.0 / t if t else None
            pr = (mspt(worst["tps"]) - mspt(D["tps"])) * (D.get("tok_step") or 0) if worst["tps"] else None
            cand.append(("probes + D7 choice (natural vs depth 0 forced)", pr, None,
                         "natural %s tok/s (zero share %s, moves %s) .. %s vs forced %s; as ms a forced step's tokens" % (
                             " / ".join(f(x["tps"], 1) for x in nat), " / ".join(f(x["zero"], 2) for x in nat), " / ".join(str(x["moves"]) for x in nat),
                             f(best["tps"], 1), f(D.get("tps"), 1))))
        if D.get("vram") is not None and N.get("vram") is not None:
            du = N["vram"] - D["vram"]
            cand.append(("VRAM units vs NST 0 (est. via the vramsweep slope)", None, None,
                         "%+d units a rank (NST 0 %s, depth 0 %s) x %.1f ms / 100 units = est. %+.2f ms (an estimate, not a reading)" % (
                             du, N["vram"], D["vram"], slope[ph], du * slope[ph] / 100.0)))
        cand.sort(key=lambda c: -abs(c[1] if c[1] is not None else (c[2] if c[2] is not None else 0.0)))
        for i, (nm, sm, pm, note) in enumerate(cand, 1):
            L.append(f"{ph} rank {i}: {nm}: stamped {f(sm)} ms a step, profiled {f(pm)} ms a step{(' | ' + note) if note else ''}")
        if sd and sn:
            acc = sum(c[1] for c in cand if c[1] is not None and not c[0].startswith("probes"))
            L.append(f"{ph} stamped decomposition: period difference {sd['period_ms'] - sn['period_ms']:+.2f} ms = trunk kernel classes {sum(sdel.values()):+.2f} "
                     f"+ other pass kinds {sd['other_ms'] - sn['other_ms']:+.2f} + between passes {sd['between_ms'] - sn['between_ms']:+.2f} + rest "
                     f"{(sd['period_ms'] - sn['period_ms']) - acc:+.2f} (overlap / busy vs wall)")
        out[ph] = {"windows": o, "candidates": cand, "gap_ms_stats": gap}
    return out, L


# ------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    td = tempfile.mkdtemp(prefix="z0prof-st-")
    log = os.path.join(td, "engine.log")
    def table(rank, calls, scale):
        L = ["per-op timing, rank %d." % rank]
        ops = [(1, "gemm_nt", "blk.0.attn_output_b.weight", 0.10), (2, "all_reduce", "-", 0.05 + 0.02 * (3 - rank)),
               (3, "moe_exl3_gemm", "blk.0.ffn_gate_up_exps.weight", 0.30 * scale), (4, "moe_gather_all_reduce", "-", 0.06 + 0.01 * rank),
               (5, "logits_gemm_ggml", "output.weight", 0.20)]
        for idx, op, w, ms in ops:
            tot = ms * calls
            L.append("  %d %.3f %.3f dev - 0.000 %s geom #%d %s" % (calls, tot, 1000 * ms, op, idx, w))
        L.append("  total")
        return L
    lines = []
    for calls in (16, 32):
        for r in range(4): lines += table(r, calls, 1.0 + 0.1 * r)
    open(log, "w").write("\n".join(lines) + "\n")
    n1 = len(lines) // 2
    o = ar(["ar", log, "--win", "w:%d:%d" % (n1, len(lines))])
    w = o["w"]
    assert w["steps"] == 16, w
    assert abs(w["classes"]["attn"]["floor_ms"] - 0.05) < 1e-6, w["classes"]
    assert w["classes"]["attn"]["last_arrival"] == {"3": 1}, w["classes"]["attn"]
    assert "moe(gather_ar)" in w["classes"], w["classes"]
    # opscmp
    def ops_json(scale):
        r = {}
        for k in range(4):
            r[str(k)] = {"steps": 100, "device_ms_step": 10 * scale, "wait_ms_step": 1.0, "trunk_ms_step": 9 * scale, "drafter_ms_step": 0.5 * (scale > 1),
                         "by_class_ms_step": {"experts": 5 * scale, "collective": 2.0, "sample": 0.1}, "drafter_ops": {"gemm_nt dspark.fc": {"calls_step": 1, "ms_step": 0.3, "us_call": 300}},
                         "trunk_ops": {"moe_exl3_gemm": {"calls_step": 40, "ms_step": 5 * scale, "wait_ms_step": 0}}}
        return {"x": {"ranks": r}}
    pa, pb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
    json.dump(ops_json(1.2), open(pa, "w")); json.dump(ops_json(1.0), open(pb, "w"))
    oc = opscmp(["opscmp", "--a", pa + ":x", "--b", pb + ":x"])
    assert abs(oc["classes"]["experts"]["delta_mean"] - 1.0) < 1e-9, oc["classes"]
    fmt_opscmp(oc, "t")
    # summary
    w5, w0 = os.path.join(td, "w5.jsonl"), os.path.join(td, "w0.jsonl")
    def wrow(label, tps, ms, z):
        return {"label": label, "ok": True, "mb": {"agg_dec_tps": tps, "ms_decode_step": ms, "tok_per_decode_step": 8.0},
                "win": {"ms_decode_step": ms, "zero_share": z, "probe_share": 0.01, "hit_pct_rank0": 97.0, "missed_step_rank0": 10.0,
                        "stream_MB_step": 400.0, "vram_units_end": 2900, "passes_step": {"played": 1.0}, "draft": {"zero_moves": 3}}}
    with open(w5, "w") as fh:
        for r in (wrow("z0-n5-stamp-c8nat", 172.0, 38.3, 0.98), wrow("z0-n5-stamp-c8d0", 176.0, 37.0, 1.0)): fh.write(json.dumps(r) + "\n")
    with open(w0, "w") as fh: fh.write(json.dumps(wrow("z0-n0-stamp-c8", 180.0, 35.9, None)) + "\n")
    def st(per, fams):
        return {"trunk": 10, "kinds": {"10": {"fam_us": fams, "fam_n": {k: 1 for k in fams}}}, "period_us_mean": per, "trunk_wall_us_mean": per - 500,
                "trunk_busy_us_mean": per - 900, "idle_us_mean": 400, "between_us_mean": 200, "other_by_kind": {}, "steps_kept": 100}
    sp = os.path.join(td, "s8.json")
    json.dump({"c8d0": st(37000, {"exl3r4d_gemm_kernel": 15000, "r4d_ar_twoshot_wide_exact_kernel": 8000, "fill_kernel": 2500}),
               "ref:n0c8": st(35900, {"exl3r4d_gemm_kernel": 14500, "r4d_ar_twoshot_wide_exact_kernel": 7600, "fill_kernel": 1900})}, open(sp, "w"))
    o, L = summary(["summary", "--wins-n5", w5, "--wins-n0", w0, "--stamps-c8", sp])
    assert o["c8"]["gap_ms_stats"] is not None and abs(o["c8"]["gap_ms_stats"] - 1.1) < 1e-9, o["c8"]
    assert famclass("r4d_ar_twoshot_wide_exact_kernel") == "ar" and famclass("exl3r4d_gemm_kernel") == "experts" and famclass("fill_kernel") == "fill"
    assert famclass("moe_gather_all_reduce_kernel") == "ar" and famclass("gg_mr_kernel") == "dense"
    names = [c[0] for c in o["c8"]["candidates"]]
    assert names.index("fill kernels (buffer clears: arena / verify sizing)") < names.index("expert kernels (unlent slots / hit / stream)"), names
    assert names[0].startswith("probes"), names
    for p in os.listdir(td): os.remove(os.path.join(td, p))
    os.rmdir(td)
    print("Z0 selftest OK (ar: floor / last-arrival / gather_ar class; opscmp class deltas; summary gap + stamp classes + ranking)")
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("ar", "opscmp", "summary", "selftest"):
        print(__doc__); return 2
    if a[0] == "selftest": return selftest()
    label = arg(a, "--label", "")
    if a[0] == "ar":
        o = ar(a); L = fmt_ar(o)
    elif a[0] == "opscmp":
        o = opscmp(a); L = fmt_opscmp(o, label)
    else:
        o, L = summary(a)
    if "--json" in a:
        with open(arg(a, "--json"), "w") as fh: json.dump(o, fh)
    for l in L: print(f"Z0 {label} {a[0]} {l}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
