#!/usr/bin/env python3
"""ggblocks3.py (10.11.2026, queue-ar dsmain, ds41-slot agent) -- ggblocks2.py with each ARM a set of switches (--arm-a /
--arm-b JSON objects, so an arm may flip several, e.g. today's sum + contiguous vs the slot-order sum + a placement table) and
--miss-log: the RADIANCE_DEBUG_MISS_LOG file's last step at each block's start and end goes into its row ("steps"), so
epplace.py measure --from-step --to-step reads each block's measured skew. Otherwise ggblocks2.py (below) as it is.

ggblocks2.py (10.11.2026, queue-ar dsmain, ds41-gg D1) -- ggblocks.py (which a queued item runs, so it is left as it
is) + --fixed: switches held at a value in every block of both arms (POSTed with the block's own, and checked in the
block's GET), so blocks over one switch run with the others pinned. In-process A/B blocks over one runtime switch (POST
/admin/fn4) with every load point in each block: c1 (stepprobe.py essay,code 512: one user, greedy), mb c4 and mb c8
(closed loop, --dur / --warm), all three inside each block, so they see the same switch epoch and thermal state.

  ggblocks2.py --url URL --mb-dir DIR --label L --switch gg_gemv --a 0 --b 1 --order ABBAAB [--dur 75] [--warm 10]
               [--gen 512] [--fixed '{"aql_acq": 0}'] [--out FILE.jsonl]

A block = POST {switch: value, fixed..., "epoch_bump": 1} (every block starts on fresh recordings, so A/A pairs as A/B), then
stepprobe (both prompts), mb c4, mb c8, then GET /admin/fn4: a block whose switch (or a fixed one) is not as set -- GET
reports the applied value, from its owner --, or with a change still pending, or whose epoch did not move, or whose
POST was refused, is flagged and left out. Pairs are the order's consecutive (A, B) or (B, A)
blocks (A B | B A | A B for ABBAAB). Per load point and metric: A mean, B mean over the good blocks, the pooled B/A
(mean B / mean A), each pair's B/A and their t 95 % interval. Metrics: c1 decode tok/s and ms a decode step (the mean of
essay and code), c4 / c8 aggregate decode tok/s (agg_dec_tps), per-stream decode p50 (per_stream_dec_p50) and ms a
decode step. Prints one "GGB {json}" line; every block appends a row to --out. Standard library only."""
import argparse, json, math, statistics, subprocess, sys, time, urllib.request

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def http(url, body=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stepprobe(a):
    p = subprocess.run([sys.executable, "stepprobe.py", "essay,code", str(a.gen)], cwd=a.mb_dir, capture_output=True,
                       text=True, timeout=1200)
    rows = []
    for l in p.stdout.splitlines():
        if l.startswith("J "):
            try: rows.append(json.loads(l[2:]))
            except ValueError: pass
    if len(rows) != 2:
        return {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}
    return {"tps": statistics.mean(r["decode_tps"] for r in rows), "ms": statistics.mean(r["ms_per_decode_step"] for r in rows),
            "rows": rows}


def mb(a, phase, tag):
    p = subprocess.run([sys.executable, "mb.py", "--tag", tag, "--phases", phase, "--dur", str(a.dur), "--warm", str(a.warm)],
                       cwd=a.mb_dir, capture_output=True, text=True, timeout=a.dur + a.warm + 900)
    for l in p.stdout.splitlines():
        if l.startswith("MB "):
            try: return json.loads(l[3:])
            except ValueError: pass
    return {"error": (p.stdout[-300:] + p.stderr[-300:]).strip()}


def last_step(path):
    """the last complete line's step of a miss log, or None"""
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 65536))
            lines = f.read().decode(errors="replace").splitlines()
        for ln in reversed(lines[:-1] if lines else []):
            p = ln.split()
            if len(p) == 16 and p[0].lstrip("-").isdigit():
                return int(p[0])
    except OSError:
        return None
    return None


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    if len(v) < 2: return m, None
    return m, T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v))


METRICS = [("c1", "tps", "decode tok/s"), ("c1", "ms", "ms a step"),
           ("c4", "agg_dec_tps", "aggregate tok/s"), ("c4", "per_stream_dec_p50", "per-stream p50"), ("c4", "ms_decode_step", "ms a step"),
           ("c8", "agg_dec_tps", "aggregate tok/s"), ("c8", "per_stream_dec_p50", "per-stream p50"), ("c8", "ms_decode_step", "ms a step")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--mb-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm-a", required=True, help="arm A's switches, a JSON object")
    ap.add_argument("--arm-b", required=True, help="arm B's switches, a JSON object")
    ap.add_argument("--miss-log")
    ap.add_argument("--order", default="ABBAAB")
    ap.add_argument("--dur", type=int, default=75)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--fixed", default="{}")
    ap.add_argument("--out")
    a = ap.parse_args()
    fixed = json.loads(a.fixed)
    val = {"A": json.loads(a.arm_a), "B": json.loads(a.arm_b)}
    rows, t_start = [], time.time()
    for i, arm in enumerate(a.order):
        e0 = http(a.url + "/admin/fn4").get("epoch")
        s0 = last_step(a.miss_log)
        try:
            http(a.url + "/admin/fn4", dict(fixed, **dict(val[arm], epoch_bump=1)))
            posted = True
        except Exception as e:   # a refused body (400) queues nothing: the block is flagged
            posted = str(e)[:200]
        res = {"c1": stepprobe(a)}
        for ph in ("c4", "c8"):
            res[ph] = mb(a, ph, f"{a.label}-b{i}{arm}-{ph}")
        st = http(a.url + "/admin/fn4")
        sw = st.get("switches", {})
        ok = posted is True and all(sw.get(k) == v for k, v in val[arm].items()) and all(sw.get(k) == v for k, v in fixed.items()) \
            and not st.get("pending") and (st.get("epoch") or 0) > (e0 or 0) and not any("error" in res[p] for p in res)
        row = {"label": a.label, "block": i, "arm": arm, "value": val[arm], "fixed": fixed, "posted": posted, "applied": sw,
               "epoch": st.get("epoch"),
               "applied_step": st.get("applied_step"), "ok": ok, "res": res, "t": round(time.time() - t_start, 1),
               "steps": [s0, last_step(a.miss_log)]}
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        print("BLOCK %d %s %s ok=%s c1 %.2f tok/s %.2f ms | c4 %s tok/s p50 %s %s ms | c8 %s tok/s p50 %s %s ms" % (
            i, arm, json.dumps(val[arm]), ok, res["c1"].get("tps") or 0, res["c1"].get("ms") or 0,
            res["c4"].get("agg_dec_tps"), res["c4"].get("per_stream_dec_p50"), res["c4"].get("ms_decode_step"),
            res["c8"].get("agg_dec_tps"), res["c8"].get("per_stream_dec_p50"), res["c8"].get("ms_decode_step")), flush=True)
    pairs = [(rows[j], rows[j + 1]) for j in range(0, len(rows) - 1, 2) if rows[j]["arm"] != rows[j + 1]["arm"]]
    out = {"label": a.label, "A": val["A"], "B": val["B"], "fixed": fixed, "order": a.order, "blocks": len(rows),
           "blocks_bad": sum(1 for r in rows if not r["ok"]), "wall_s": round(time.time() - t_start, 1), "metrics": {}}
    for ph, k, nm in METRICS:
        va = [r["res"][ph].get(k) for r in rows if r["ok"] and r["arm"] == "A" and r["res"][ph].get(k)]
        vb = [r["res"][ph].get(k) for r in rows if r["ok"] and r["arm"] == "B" and r["res"][ph].get(k)]
        pr = []
        for x, y in pairs:
            if not (x["ok"] and y["ok"]): continue
            ra, rb = (x, y) if x["arm"] == "A" else (y, x)
            if ra["res"][ph].get(k) and rb["res"][ph].get(k): pr.append(rb["res"][ph][k] / ra["res"][ph][k])
        m, h = ci(pr)
        out["metrics"][f"{ph} {nm}"] = {"A": statistics.mean(va) if va else None, "B": statistics.mean(vb) if vb else None,
                                        "pooled": (statistics.mean(vb) / statistics.mean(va)) if va and vb else None,
                                        "pairs": pr, "pair_mean": m, "ci95": h}
    print("GGB " + json.dumps(out))
    return 0 if out["blocks_bad"] < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
