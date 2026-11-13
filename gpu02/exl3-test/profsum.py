#!/usr/bin/env python3
"""profsum.py (6.11.2026) -- after-sweep.sh's --profile-ops / --live summaries, on the 3.11. parser (~/gpu/ds41/profparse.py:
dumps(), cat()). Steps in a window = EVERY x the rank's complete dumps between the two marks (a dump comes every EVERY engine
steps, idle adds none), so op ms per step needs no request-side step count. marks.txt lines: "phase logline epoch" (3.11. form).
  profsum.py opstep RUNDIR MARKSFILE M0 M1 EVERY [RANK]  -> "OPSTEP {json}": device / host-site op ms per step between two marks
  profsum.py phases RUNDIR EVERY [RANK]                  -> per consecutive marks.txt window: category ms/step, routed-MoE split
                                                            (exl3 / ggml / other), top ops, host-site ms; one "PS {json}" per window
  profsum.py live RUNDIR                                  -> per marks.txt window, from the --live key=value lines: expert units per
                                                            tier (vram/pinned/host/ssd/mapped), resident %, hit %, promotions,
                                                            link bytes per step (ar, h2d, d2h, stream, ssd), prefill vs decode
                                                            steps apart; one "LV {json}" per window
Read-only on RUNDIR."""
import json, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/gpu/ds41"))
from profparse import dumps, cat  # noqa: E402


def read_marks(path):
    out = []
    for l in open(path):
        p = l.split()
        if len(p) >= 3:
            out.append((p[0], int(p[1]), int(p[2])))
    return out


def read_metrics(rundir):
    """marks-metrics.txt: "phase steps_total steps_decode step_s_total step_s_decode" (after-sweep.sh mark())."""
    m = {}
    try:
        for l in open(os.path.join(rundir, "marks-metrics.txt")):
            p = l.split()
            if len(p) >= 5:
                try:
                    m[p[0]] = [float(x) for x in p[1:5]]
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return m


def n_at(ds, line):
    n = 0
    for ln, _ in ds:
        if ln > line:
            break
        n += 1
    return n


def window(ds, l0, l1, every):
    i0, i1 = n_at(ds, l0), n_at(ds, l1)
    a = ds[i0 - 1][1] if i0 > 0 else {}
    b = ds[i1 - 1][1] if i1 > 0 else {}
    d = {}
    for k, v in b.items():
        z = a.get(k, (0.0, 0, v[2], 0.0))
        if v[0] - z[0] > 0 or v[1] - z[1] > 0:
            d[k] = (v[0] - z[0], v[1] - z[1], v[2], v[3] - z[3])
    return d, (i1 - i0) * every


def summarize(d, steps):
    s = max(1, steps)
    dev = sum(v[0] for v in d.values() if v[2] == "dev")
    host = sum(v[0] for v in d.values() if v[2] == "host")
    waits = sum(v[3] for v in d.values())
    cats, moe, byop = defaultdict(float), defaultdict(float), defaultdict(lambda: [0.0, 0, ""])
    for (op, idx, w), (ms, calls, site, wt) in d.items():
        cats[cat(op, w)] += ms
        if op.startswith("moe_"):
            moe["exl3" if "exl3" in op else "ggml" if "ggml" in op else "other(" + op + ")"] += ms
        e = byop[op]; e[0] += ms; e[1] += calls; e[2] = site
    r = {"steps": steps, "op_ms_step": round((dev + host) / s, 3), "dev_ms_step": round(dev / s, 3), "host_ms_step": round(host / s, 3),
         "wait_ms_step": round(waits / s, 3),
         "cats": {k: round(v / s, 3) for k, v in sorted(cats.items(), key=lambda x: -x[1])},
         "moe": {k: round(v / s, 3) for k, v in sorted(moe.items(), key=lambda x: -x[1])},
         "top": [[op, round(ms / s, 3), round(calls / s, 2), site] for op, (ms, calls, site) in sorted(byop.items(), key=lambda x: -x[1][0])[:12]]}
    return r


def main_opstep(rundir, marksfile, m0, m1, every, rank):
    mk = {p: ln for p, ln, _ in read_marks(marksfile)}
    ds = dumps(os.path.join(rundir, "engine.log"), rank)
    d, steps = window(ds, mk[m0], mk[m1], every)
    r = summarize(d, steps)
    if steps <= 0:   # no complete dump between the marks (fastcheck failed or too short): never read as a fast instance
        r = dict(r, op_ms_step=999.0, dev_ms_step=999.0)
    print("OPSTEP " + json.dumps({k: r[k] for k in ("steps", "op_ms_step", "dev_ms_step", "host_ms_step", "wait_ms_step")}))


def main_phases(rundir, every, rank):
    mk = read_marks(os.path.join(rundir, "marks.txt"))
    met = read_metrics(rundir)
    ds = dumps(os.path.join(rundir, "engine.log"), rank)
    print(f"# {rundir}: rank {rank}, {len(ds)} complete dumps, every {every} steps; marks {[(p, l) for p, l, _ in mk]}")
    for (p0, l0, t0), (p1, l1, t1) in zip(mk, mk[1:]):
        d, steps = window(ds, l0, l1, every)
        if steps <= 0:
            print(f"\n== {p1}: no complete dump in the window ({t1 - t0} s)"); continue
        r = summarize(d, steps); r["phase"] = p1; r["secs"] = t1 - t0
        if p0 in met and p1 in met:
            a, b = met[p0], met[p1]
            r["eng_steps"], r["eng_decode_steps"] = b[0] - a[0], b[1] - a[1]
            r["wall_ms_step"] = round(1000 * (b[2] - a[2]) / max(1, b[0] - a[0]), 2)
            r["wall_ms_decode_step"] = round(1000 * (b[3] - a[3]) / max(1, b[1] - a[1]), 2)
        print(f"\n== {p1}: {steps} steps from dumps ({t1 - t0} s), op {r['op_ms_step']:.2f} ms/step (device {r['dev_ms_step']:.2f}, host-site "
              f"{r['host_ms_step']:.2f}), lane waits {r['wait_ms_step']:.2f} ms/step"
              + (f"; engine steps {r['eng_steps']:.0f} (decode {r['eng_decode_steps']:.0f}), profiled wall {r['wall_ms_step']} ms/step" if "eng_steps" in r else ""))
        tot = max(1e-9, r["op_ms_step"])
        for k, v in r["cats"].items():
            print(f"   {k:20s} {v:9.3f} ms/step {100 * v / tot:5.1f} %")
        print("   routed MoE split: " + ", ".join(f"{k} {v:.3f}" for k, v in r["moe"].items()))
        for op, ms, calls, site in r["top"]:
            print(f"   {op:40s} {site:4s} {ms:9.3f} ms/step  {calls:8.2f} calls/step")
        print("PS " + json.dumps(r))


def parse_live(line):
    head, _, link = line.partition(" link ")
    main, _, exp = head.partition(" experts ")
    kv = lambda s: dict(t.split("=", 1) for t in s.split() if "=" in t)
    num = lambda x: float(re.sub(r"[^0-9.eE+-]", "", x) or 0) if x is not None else 0.0
    m, e, k = kv(main), kv(exp), kv(link)
    return {"step": num(m.get("step")), "run": num(m.get("run")), "prefill_tps": num(m.get("prefill_tps")), "decode_tps": num(m.get("decode_tps")),
            "tiers": {t: num(e.get(t)) for t in ("vram", "pinned", "host", "ssd", "mapped") if t in e},
            "resident": num(e.get("resident")), "hit": num(e.get("hit")) if "hit" in e else None,
            "promo": num(e.get("promo")), "demo": num(e.get("demo")),
            "ar": num(k.get("ar")), "h2d": num(k.get("h2d")), "d2h": num(k.get("d2h")), "stream": num(k.get("stream")), "ssd": num(k.get("ssd"))}


def main_live(rundir):
    recs = []
    for i, l in enumerate(open(os.path.join(rundir, "engine.log"), errors="replace"), 1):
        if l.startswith("live step="):
            try:
                recs.append((i, parse_live(l.rstrip("\n"))))
            except Exception:
                pass
    mk = read_marks(os.path.join(rundir, "marks.txt"))
    print(f"# {rundir}: {len(recs)} live lines")
    B = ("ar", "h2d", "d2h", "stream", "ssd")
    for (p0, l0, t0), (p1, l1, t1) in zip(mk, mk[1:]):
        w = [r for ln, r in recs if l0 < ln <= l1]
        if len(w) < 2:
            print(f"== {p1}: no live lines"); continue
        a, b = w[0], w[-1]; st = max(1.0, b["step"] - a["step"]); dt = max(1, t1 - t0)
        r = {"phase": p1, "steps": b["step"] - a["step"], "secs": t1 - t0,
             "per_step_MB": {x: round((b[x] - a[x]) / st / 1e6, 2) for x in B},
             "per_s_GBps": {x: round((b[x] - a[x]) / dt / 1e9, 3) for x in B},
             "tiers_end": b["tiers"], "resident_end": b["resident"],
             "hit_mean": round(sum(x["hit"] for x in w if x["hit"] is not None) / max(1, sum(1 for x in w if x["hit"] is not None)), 2),
             "promo": b["promo"] - a["promo"], "demo": b["demo"] - a["demo"]}
        agg = {"prefill": defaultdict(float), "decode": defaultdict(float)}
        for x, y in zip(w, w[1:]):
            ds = y["step"] - x["step"]
            if ds <= 0:
                continue
            g = agg["prefill" if y["prefill_tps"] > 500 else "decode"]; g["steps"] += ds
            for k in B:
                g[k] += y[k] - x[k]
        r["split"] = {k: dict({"steps": g["steps"]}, **{x: round(g[x] / g["steps"] / 1e6, 2) for x in B}) for k, g in agg.items() if g["steps"]}
        print(f"== {p1}: {r['steps']:.0f} steps in ~{r['secs']} s; MB/step " + ", ".join(f"{x} {v}" for x, v in r["per_step_MB"].items())
              + f"; tiers {r['tiers_end']} resident {r['resident_end']} % hit(mean) {r['hit_mean']} %; promo {r['promo']:.0f} demo {r['demo']:.0f}")
        for k, g in r["split"].items():
            print(f"     {k:8s}: " + ", ".join(f"{x} {v}" for x, v in g.items()))
        print("LV " + json.dumps(r))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "opstep":
        main_opstep(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    elif mode == "phases":
        main_phases(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "live":
        main_live(sys.argv[2])
    else:
        sys.exit(__doc__)
