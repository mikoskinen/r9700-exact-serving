#!/usr/bin/env python3
"""z0steps.py (11.11.2026, queue-ar dsmain, ds-main-port Z0 agent; analyser of 57-p4zyzzzzz-ds-hy-z0prof2) -- the stamped c4 / c8
windows of hy-z0prof BY STEP CLASS, not only decode: RADIANCE_AQL_STAMP's CSV (card 0, every played pass, host CLOCK_MONOTONIC
ns) per window, every pass by KIND (its tape's record count): passes, wall sum / mean, busy mean, top kernel families; the
DECODE kind = the most frequent kind with >= 1000 records (the trunk's decode pass); the rest are prefill / mixed trunk passes
(more records than decode), drafter passes (context refills, query passes at probes) and small passes. Card-0 idle time (no
pass running) is attributed to the pass that ends it: the gap before a decode pass, before a non-decode trunk pass (host work
around a prefill / mixed step: admission, the lookup index, the lend's state, the step's preparation) and before a small pass.
Per window: total ms, decode-kind wall, non-decode trunk wall, drafter / small passes wall, idle by class -- each also per
request (--req NAME:N, mb.py's requests in the window) and per non-decode trunk pass. PLAYED passes only (an eager pass is not
stamped; a non-decode step whose pass was never recorded shows as idle).

  z0steps.py CSV[.gz] --win NAME:T0:T1 [--win ...] [--req NAME:N ...] [--json OUT] [--label L]
  z0steps.py selftest
Prints "Z0S <label> ..." lines. Standard library only; reads the CSV once."""
import csv, gzip, json, os, re, sys, tempfile
from collections import defaultdict


def opn(p):
    return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")


_fc = {}


def family(k):
    f = _fc.get(k)
    if f is not None: return f
    s = k; i = 2 if s.startswith("_Z") else 0
    if s[i:i + 1] == "N": i += 1
    comps = []
    while i < len(s) and s[i].isdigit():
        j = i
        while j < len(s) and s[j].isdigit(): j += 1
        n = int(s[i:j]); comps.append(s[j:j + n]); i = j + n
        if s[i:i + 1] == "I": break
    comps = [c for c in comps if c and not c.startswith("_GLOBAL__N")]
    f = comps[-1] if comps else k[:60]
    _fc[k] = f
    return f


def wins(a, flag):
    out = []
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a): out.append(a[i + 1])
    return out


def read(path, W):
    P = {n: defaultdict(lambda: {"rec": 0, "iv": [], "fam": defaultdict(float)}) for n, _, _ in W}
    with opn(path) as f:
        for r in csv.DictReader(f):
            s, e = r.get("start_ns"), r.get("end_ns")
            if not s or not e: continue
            s, e = int(s), int(e)
            for n, lo, hi in W:
                if lo <= s <= hi:
                    p = P[n][int(r["pass"])]
                    p["rec"] = int(r["records"]); p["iv"].append((s, e)); p["fam"][family(r["kernel"])] += (e - s) / 1e3
    return P


def union(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        elif e > ce: ce = e
    if ce is not None: tot += ce - cs
    return tot


def analyse(passes, lo, hi):
    ps = []
    for pid, p in passes.items():
        if not p["iv"]: continue
        t0, t1 = min(x[0] for x in p["iv"]), max(x[1] for x in p["iv"])
        ps.append({"rec": p["rec"], "t0": t0, "t1": t1, "wall": (t1 - t0) / 1e6, "busy": union(p["iv"]) / 1e6, "fam": p["fam"]})
    if not ps: return {"error": "no stamped pass"}
    ps.sort(key=lambda x: x["t0"])
    cnt = defaultdict(int)
    for p in ps: cnt[p["rec"]] += 1
    big = [k for k in cnt if k >= 1000]
    dec = max(big, key=lambda k: cnt[k]) if big else max(cnt, key=lambda k: cnt[k])
    def cls(rec):
        if rec == dec: return "decode"
        if rec > dec: return "trunk_other"      # prefill / mixed trunk passes (more records than a decode pass)
        if rec >= 0.5 * dec: return "trunk_other"
        return "small"                          # drafter context refills, query passes, sampling, staging
    kinds = {}
    for k in sorted(cnt, key=lambda k: -cnt[k]):
        v = [p for p in ps if p["rec"] == k]
        fam = defaultdict(float)
        for p in v:
            for fm, us in p["fam"].items(): fam[fm] += us
        kinds[str(k)] = {"class": cls(k), "passes": len(v), "wall_ms_sum": sum(p["wall"] for p in v), "wall_ms_mean": sum(p["wall"] for p in v) / len(v),
                         "busy_ms_mean": sum(p["busy"] for p in v) / len(v),
                         "top": sorted(((fm, us / len(v) / 1e3) for fm, us in fam.items()), key=lambda t: -t[1])[:8]}
    # idle: card 0 time inside [first pass start, last pass end] covered by no pass, attributed to the pass that ends the gap
    idle = defaultdict(float); idle_n = defaultdict(int)
    end = ps[0]["t1"]
    for p in ps[1:]:
        if p["t0"] > end:
            c = cls(p["rec"]); idle[c] += (p["t0"] - end) / 1e6; idle_n[c] += 1
        end = max(end, p["t1"])
    span = (end - ps[0]["t0"]) / 1e6
    wall = defaultdict(float); n = defaultdict(int)
    for p in ps: wall[cls(p["rec"])] += p["wall"]; n[cls(p["rec"])] += 1
    return {"decode_kind": dec, "span_ms": span, "wall_ms": dict(wall), "passes": dict(n), "idle_ms": dict(idle), "idle_gaps": dict(idle_n),
            "kinds": kinds, "busy_ms": union([(p["t0"], p["t1"]) for p in ps]) / 1e6}


def fmt(name, x, req):
    if "error" in x: return ["%s: %s" % (name, x["error"])]
    w, n, i = x["wall_ms"], x["passes"], x["idle_ms"]
    to = w.get("trunk_other", 0.0); no = n.get("trunk_other", 0)
    L = ["%s: span %.0f ms; decode kind %s: %d passes %.0f ms (%.2f a pass); non-decode trunk passes %d, %.0f ms (%s a pass); small passes %d, %.0f ms; "
         "card idle before decode passes %.0f ms (%d gaps, %.2f a pass), before non-decode trunk passes %.0f ms (%d gaps%s), before small passes %.0f ms%s" % (
             name, x["span_ms"], x["decode_kind"], n.get("decode", 0), w.get("decode", 0.0), w.get("decode", 0.0) / max(1, n.get("decode", 0)),
             no, to, "%.1f" % (to / no) if no else "-", n.get("small", 0), w.get("small", 0.0),
             i.get("decode", 0.0), x["idle_gaps"].get("decode", 0), i.get("decode", 0.0) / max(1, n.get("decode", 0)),
             i.get("trunk_other", 0.0), x["idle_gaps"].get("trunk_other", 0), ", %.1f a pass" % (i.get("trunk_other", 0.0) / no) if no else "",
             i.get("small", 0.0),
             (" | per request (%d): non-decode trunk %.0f ms + idle before them %.0f ms + small passes %.0f ms = %.0f ms" % (
                 req, to / req, i.get("trunk_other", 0.0) / req, w.get("small", 0.0) / req, (to + i.get("trunk_other", 0.0) + w.get("small", 0.0)) / req)) if req else "")]
    for k, v in list(x["kinds"].items())[:8]:
        L.append("%s kind %s [%s]: %d passes, wall %.2f ms mean (%.0f ms sum), busy %.2f; top families (ms a pass): %s" % (
            name, k, v["class"], v["passes"], v["wall_ms_mean"], v["wall_ms_sum"], v["busy_ms_mean"], ", ".join("%s %.2f" % (f, ms) for f, ms in v["top"][:6])))
    return L


def main():
    a = sys.argv[1:]
    if a[:1] == ["selftest"]: return selftest()
    if not a: print(__doc__); return 2
    W = []
    for x in wins(a, "--win"):
        n, lo, hi = x.split(":"); W.append((n, int(lo), int(hi)))
    req = {}
    for x in wins(a, "--req"):
        n, v = x.split(":"); req[n] = int(v)
    P = read(a[0], W)
    out = {n: analyse(P[n], lo, hi) for n, lo, hi in W}
    lab = a[a.index("--label") + 1] if "--label" in a else ""
    if "--json" in a:
        with open(a[a.index("--json") + 1], "w") as fh: json.dump(out, fh)
    for n, _, _ in W:
        for l in fmt(n, out[n], req.get(n)): print("Z0S %s %s" % (lab, l))
    return 0


def selftest():
    td = tempfile.mkdtemp(prefix="z0steps-st-"); p = os.path.join(td, "s.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["pass", "seq", "queue", "kernel", "ordered", "chained", "start_ns", "end_ns", "play", "records"])
        t = 1000
        for i in range(10):          # decode passes: 1200 records, 2 dispatches, 10 ms, 1 ms gap
            w.writerow([i, 0, 0, "_Z8kernAaaav", 1, 0, t, t + 4_000_000, 0, 1200]); w.writerow([i, 1, 0, "_Z8kernBbbbv", 1, 0, t + 4_000_000, t + 10_000_000, 0, 1200])
            t += 11_000_000
        w.writerow([10, 0, 0, "_Z8kernCcccv", 1, 0, t + 5_000_000, t + 105_000_000, 0, 1280]); t += 105_000_000   # 1 ms gap + 5 ms idle, then a 100 ms prefill pass
        w.writerow([11, 0, 0, "_Z8kernDdddv", 1, 0, t, t + 2_000_000, 0, 18])
    P = read(p, [("w", 0, 10**12)])
    x = analyse(P["w"], 0, 10**12)
    assert x["decode_kind"] == 1200 and x["passes"]["decode"] == 10 and x["passes"]["trunk_other"] == 1 and x["passes"]["small"] == 1, x
    assert abs(x["wall_ms"]["trunk_other"] - 100.0) < 1e-6 and abs(x["idle_ms"]["trunk_other"] - 6.0) < 1e-6 and abs(x["idle_ms"]["decode"] - 9.0) < 1e-6, x
    fmt("w", x, 1)
    os.remove(p); os.rmdir(td)
    print("Z0S selftest OK (kinds by records, decode / non-decode trunk / small classes, idle attributed to the pass ending the gap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
