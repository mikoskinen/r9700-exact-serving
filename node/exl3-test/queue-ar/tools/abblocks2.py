#!/usr/bin/env python3
"""abblocks2.py -- abblocks.py (in-process A/B blocks over the fn4 runtime switches, POST /admin/fn4, 9.11.2026,
queue-ar, item 121) + a PREFILL window in every block (item 129, 9.11.2026): after the block's mb window (decode
first, TESTING.md 9.11.: no soak, the mb window before the prefill window inside a block, A B B A), fresh prefills
of --pf-depths (ds41_tests.py prefill, nonce first, --cpt 3.6 --calibrate 0 --reps 1, in --mb-dir), each block's
prefill tok/s per depth with the window's thermal line (~/gpu/temps/<day>.csv full-power rows, per card median
junction / sclk, as heatsoak.sh thermwin). A round's prefill ratio per depth = (B1 + B2) / (A1 + A2) of prefill_tps.
abblocks.py itself is unchanged (a running item reads it).

  abblocks2.py --url URL --mb-dir DIR --label L --phase c16 --dur 20 --warm 6 --rounds 3
               --a '{"hc_writevec": 0}' --b '{"hc_writevec": 1}' [--pf-depths 16384,65536,131072]
               [--out FILE.jsonl] [--names A,B]

One server, one process: each round is A B B A (the order cancels a linear drift), a block = POST the arm's switches
(every switch named in either arm is set explicitly in both, so no block inherits the previous one's, plus epoch_bump,
so every block starts on fresh recordings whether or not a switch changed), then one mb.py
window (--phases PHASE --dur --warm; the switch lands at the first step release of its warm-up), then GET /admin/fn4 to
check the switches the window ran under (values and the epoch). Reading: a round's ratio = (B1 + B2) / (A1 + A2) of
agg_dec_tps (and of ms_decode_step); the item's = their mean with a t 95 % interval over rounds. A block whose GET shows
other values than its arm's, or a pending change, is flagged and left out. Prints one "ABB {json}" line; every block
appends a row to --out. Standard library only; mb.py runs as a subprocess in --mb-dir.
12.11.2026 (quick screens): --no-mb skips the mb window (a prefill-only lever: the target metric only); the output gains
"sep" per metric -- "B>A" when every B block beats every A block, "B<A" the other way, else "overlap" (a guide beside the size and direction,
not a gate), and the A / B block values."""
import argparse, json, math, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def mb(a, tag):
    cmd = [sys.executable, "mb.py", "--tag", tag, "--phases", a.phase, "--dur", str(a.dur), "--warm", str(a.warm)]
    if a.mbt2:   # 12.11.: the fixed, seeded arrival schedule (dsmain harness mbt2.py, copied to tools/fnharness)
        n = "".join(ch for ch in a.phase if ch.isdigit()) or "8"
        cmd = [sys.executable, a.mbt2, "--mb", "mb.py", "--temperature", "0", "--tag", tag, "--phases", a.phase,
               "--pattern", "1x" + n, "--seed", str(a.seed), "--gens", str(a.gens), "--warm", str(a.warm)]
    p = subprocess.run(cmd, cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 600)
    for l in p.stdout.splitlines():
        if l.startswith("MB "):
            try:
                return json.loads(l[3:])
            except ValueError:
                pass
    return {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def prefill(a, depths):
    """fresh prefills of each depth -> ({target: {"tps", "ttft", "tok"}}, t0, t1)"""
    t0 = time.time()
    cmd = [sys.executable, "ds41_tests.py", "prefill", "--depths", ",".join(str(d) for d in depths), "--cpt", "3.6",
           "--calibrate", "0", "--reps", "1"]
    out = {}
    try:
        p = subprocess.run(cmd, cwd=a.mb_dir, capture_output=True, text=True, timeout=300 + 60 * len(depths))
        for l in p.stdout.splitlines():
            if not l.startswith("J "): continue
            try: d = json.loads(l[2:])
            except ValueError: continue
            if d.get("test") == "prefill" and d.get("status") == 200 and d.get("prefill_tps"):
                out[str(d.get("target"))] = {"tps": d["prefill_tps"], "ttft": d.get("ttft"), "tok": d.get("prompt_tokens")}
    except subprocess.TimeoutExpired:
        pass
    return out, t0, time.time()


def therm(t0, t1):
    """per card median junction C / sclk MHz over the window's full-power rows (power >= 195 W)"""
    import os
    days = {time.strftime("%Y%m%d", time.localtime(t)) for t in (t0, t1)}
    rows = {}
    for d in sorted(days):
        p = os.path.expanduser(f"~/gpu/temps/{d}.csv")
        if not os.path.exists(p): continue
        for l in open(p):
            f = l.strip().split(",")
            if len(f) < 7: continue
            try: t, pci, j, w, s = float(f[0]), f[1], int(f[2]), int(f[4]), int(f[6])
            except ValueError: continue
            if t0 <= t <= t1 and w >= 195: rows.setdefault(pci, []).append((j, s))
    if not rows: return "no full-power rows"
    return " ".join(f"{pci}:{statistics.median(x[0] for x in v):.0f}C/{statistics.median(x[1] for x in v):.0f}MHz(n{len(v)})"
                    for pci, v in sorted(rows.items()))


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    h = T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))
    return m, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--phase", default="c16")
    ap.add_argument("--dur", type=int, default=20)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--names", default="A,B")
    ap.add_argument("--out")
    ap.add_argument("--pf-depths", default="", help="comma list: fresh prefills after every block's mb window")
    ap.add_argument("--no-mb", action="store_true", help="no mb window: the blocks are the prefills only")
    ap.add_argument("--mbt2", default="", help="path of mbt2.py: the block's window on the fixed, seeded schedule")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gens", type=int, default=1)
    a = ap.parse_args()
    depths = [int(x) for x in a.pf_depths.split(",") if x.strip()]
    A, B = json.loads(a.a), json.loads(a.b)
    keys = sorted(set(A) | set(B))
    base = http(a.url + "/admin/fn4").get("switches", {})
    full = {"A": {k: A.get(k, base.get(k, 0)) for k in keys}, "B": {k: B.get(k, base.get(k, 0)) for k in keys}}
    na, nb = a.names.split(",")
    rows, rounds, pf_rounds = [], [], []
    t_start = time.time()
    for r in range(a.rounds):
        got = {"A": [], "B": []}
        gotp = {"A": [], "B": []}
        for arm in ("A", "B", "B", "A"):
            cfg = full[arm]
            http(a.url + "/admin/fn4", dict(cfg, epoch_bump=1))   # every block re-records: A/A pairs as A/B
            res = {} if a.no_mb else mb(a, f"{a.label}-r{r}-{arm}{len(got[arm])}")
            pf, pt0, pt1 = prefill(a, depths) if depths else ({}, None, None)
            st = http(a.url + "/admin/fn4")
            sw = st.get("switches", {})
            ok = all(sw.get(k) == v for k, v in cfg.items()) and not st.get("pending")
            row = {"label": a.label, "round": r, "arm": arm, "name": na if arm == "A" else nb, "switches": cfg,
                   "applied": sw, "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok,
                   "tps": res.get("agg_dec_tps"), "ms": res.get("ms_decode_step"), "mb": res, "t": time.time() - t_start,
                   "pf": pf, "pf_therm": therm(pt0, pt1) if depths else None}
            rows.append(row)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if ok and row["tps"]: got[arm].append((row["tps"], row["ms"]))
            if ok and depths: gotp[arm].append(pf)
            if a.no_mb: got[arm].append(None)
        if not a.no_mb and len(got["A"]) == 2 and len(got["B"]) == 2:
            ta = sum(x[0] for x in got["A"]); tb = sum(x[0] for x in got["B"])
            ma = sum(x[1] or 0 for x in got["A"]); mbs = sum(x[1] or 0 for x in got["B"])
            rounds.append({"round": r, "tps_ratio": tb / ta, "ms_ratio": (mbs / ma) if ma else None,
                           "A": [x[0] for x in got["A"]], "B": [x[0] for x in got["B"]]})
        if depths and len(gotp["A"]) == 2 and len(gotp["B"]) == 2:
            pr = {}
            for d in depths:
                va = [x.get(str(d), {}).get("tps") for x in gotp["A"]]; vb = [x.get(str(d), {}).get("tps") for x in gotp["B"]]
                if all(va) and all(vb): pr[str(d)] = {"ratio": sum(vb) / sum(va), "A": va, "B": vb}
            pf_rounds.append({"round": r, "depths": pr})
    m, h = ci([x["tps_ratio"] for x in rounds])
    mm, hm = ci([x["ms_ratio"] for x in rounds if x["ms_ratio"]])
    out = {"label": a.label, "phase": a.phase, "A": full["A"], "B": full["B"], "names": [na, nb],
           "rounds": rounds, "blocks": len(rows), "blocks_bad": sum(1 for x in rows if not x["ok"]),
           "tps_ratio": m, "tps_ci95": h, "ms_ratio": mm, "ms_ci95": hm,
           "A_tps_mean": statistics.mean([v for x in rounds for v in x["A"]]) if rounds else None,
           "B_tps_mean": statistics.mean([v for x in rounds for v in x["B"]]) if rounds else None,
           "wall_s": round(time.time() - t_start, 1)}
    if depths:
        pfo = {}
        for d in depths:
            rr = [x["depths"][str(d)] for x in pf_rounds if str(d) in x["depths"]]
            m2, h2 = ci([x["ratio"] for x in rr])
            pfo[str(d)] = {"ratio": m2, "ci95": h2, "rounds": [x["ratio"] for x in rr],
                           "A_mean": statistics.mean([v for x in rr for v in x["A"]]) if rr else None,
                           "B_mean": statistics.mean([v for x in rr for v in x["B"]]) if rr else None}
        out["pf"] = pfo
    def sep(va, vb, higher=True):
        if not va or not vb: return None
        if min(vb) > max(va): return "B>A" if higher else "B<A"
        if max(vb) < min(va): return "B<A" if higher else "B>A"
        return "overlap"
    ab = [v for x in rounds for v in x["A"]]; bb = [v for x in rounds for v in x["B"]]
    out["sep"] = {"tps": sep(ab, bb)}
    out["A_blocks"], out["B_blocks"] = ab, bb
    for d, v in (out.get("pf") or {}).items():
        rr = [x["depths"][d] for x in pf_rounds if d in x["depths"]]
        va = [t for x in rr for t in x["A"]]; vb = [t for x in rr for t in x["B"]]
        v["sep"] = sep(va, vb); v["A_blocks"], v["B_blocks"] = va, vb
    print("ABB " + json.dumps(out))
    return 0 if (rounds or pf_rounds) else 1


if __name__ == "__main__":
    sys.exit(main())
