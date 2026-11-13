#!/usr/bin/env python3
"""abblocks.py -- in-process A/B blocks over the fn4 runtime switches (POST /admin/fn4, 9.11.2026, queue-ar, item 121).

  abblocks.py --url URL --mb-dir DIR --label L --phase c16 --dur 20 --warm 6 --rounds 3
              --a '{"hc_normwave": 0}' --b '{"hc_normwave": 1}' [--out FILE.jsonl] [--names A,B]

One server, one process: each round is A B B A (the order cancels a linear drift), a block = POST the arm's switches
(every switch named in either arm is set explicitly in both, so no block inherits the previous one's, plus epoch_bump,
so every block starts on fresh recordings whether or not a switch changed), then one mb.py
window (--phases PHASE --dur --warm; the switch lands at the first step release of its warm-up), then GET /admin/fn4 to
check the switches the window ran under (values and the epoch). Reading: a round's ratio = (B1 + B2) / (A1 + A2) of
agg_dec_tps (and of ms_decode_step); the item's = their mean with a t 95 % interval over rounds. A block whose GET shows
other values than its arm's, or a pending change, is flagged and left out. Prints one "ABB {json}" line; every block
appends a row to --out. Standard library only; mb.py runs as a subprocess in --mb-dir."""
import argparse, json, math, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def mb(a, tag):
    cmd = [sys.executable, "mb.py", "--tag", tag, "--phases", a.phase, "--dur", str(a.dur), "--warm", str(a.warm)]
    p = subprocess.run(cmd, cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 600)
    for l in p.stdout.splitlines():
        if l.startswith("MB "):
            try:
                return json.loads(l[3:])
            except ValueError:
                pass
    return {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


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
    a = ap.parse_args()
    A, B = json.loads(a.a), json.loads(a.b)
    keys = sorted(set(A) | set(B))
    base = http(a.url + "/admin/fn4").get("switches", {})
    full = {"A": {k: A.get(k, base.get(k, 0)) for k in keys}, "B": {k: B.get(k, base.get(k, 0)) for k in keys}}
    na, nb = a.names.split(",")
    rows, rounds = [], []
    t_start = time.time()
    for r in range(a.rounds):
        got = {"A": [], "B": []}
        for arm in ("A", "B", "B", "A"):
            cfg = full[arm]
            http(a.url + "/admin/fn4", dict(cfg, epoch_bump=1))   # every block re-records: A/A pairs as A/B
            res = mb(a, f"{a.label}-r{r}-{arm}{len(got[arm])}")
            st = http(a.url + "/admin/fn4")
            sw = st.get("switches", {})
            ok = all(sw.get(k) == v for k, v in cfg.items()) and not st.get("pending")
            row = {"label": a.label, "round": r, "arm": arm, "name": na if arm == "A" else nb, "switches": cfg,
                   "applied": sw, "epoch": st.get("epoch"), "applied_step": st.get("applied_step"), "ok": ok,
                   "tps": res.get("agg_dec_tps"), "ms": res.get("ms_decode_step"), "mb": res, "t": time.time() - t_start}
            rows.append(row)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if ok and row["tps"]: got[arm].append((row["tps"], row["ms"]))
        if len(got["A"]) == 2 and len(got["B"]) == 2:
            ta = sum(x[0] for x in got["A"]); tb = sum(x[0] for x in got["B"])
            ma = sum(x[1] or 0 for x in got["A"]); mbs = sum(x[1] or 0 for x in got["B"])
            rounds.append({"round": r, "tps_ratio": tb / ta, "ms_ratio": (mbs / ma) if ma else None,
                           "A": [x[0] for x in got["A"]], "B": [x[0] for x in got["B"]]})
    m, h = ci([x["tps_ratio"] for x in rounds])
    mm, hm = ci([x["ms_ratio"] for x in rounds if x["ms_ratio"]])
    out = {"label": a.label, "phase": a.phase, "A": full["A"], "B": full["B"], "names": [na, nb],
           "rounds": rounds, "blocks": len(rows), "blocks_bad": sum(1 for x in rows if not x["ok"]),
           "tps_ratio": m, "tps_ci95": h, "ms_ratio": mm, "ms_ci95": hm,
           "A_tps_mean": statistics.mean([v for x in rounds for v in x["A"]]) if rounds else None,
           "B_tps_mean": statistics.mean([v for x in rounds for v in x["B"]]) if rounds else None,
           "wall_s": round(time.time() - t_start, 1)}
    print("ABB " + json.dumps(out))
    return 0 if rounds else 1


if __name__ == "__main__":
    sys.exit(main())
