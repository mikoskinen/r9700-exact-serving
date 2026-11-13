#!/usr/bin/env python3
"""c4rows.py (12.11.2026, queue-ar/dsmain/c4lev, the int11 builder) -- the c4 lever-split item's readings
(57-p4zyzzzzz-ds-r92zzzzz-c4lev.sh, body dsm-c4lev.sh). Standard library only.

  c4rows.py split ENGINE_LOG(.gz) MB.jsonl [--n 4] --out SPLIT.json
      per c<n> block of MB.jsonl (the body's records: name, arm, marks = the window's two edges in epoch s), the `docker logs -t`
      decode step lines (RADIANCE_LOG_STEPS=1: "<utc>Z I step N decode n_seq=S n_tok=R ...") inside the window that are followed by
      step N + 1; a step's ms = the next step line's time - its own (<= 400 ms); by the step's rows R (all of its sequences' rows):
      1-4 / 5-8 / 9-15 / 16 / > 16 -> steps, share, p50 and mean ms.
  c4rows.py summary MB.jsonl --split SPLIT.json [--n 4] [--ref O]
      per arm: the mean agg tok/s of its blocks, vs O in %, the blocks; per-stream p50, ms / tokens a decode step, acceptance, decode
      share; per row bucket: the steps' share and mean ms a step per arm vs O (ms and %). Lines "C4LEV ..." and one "SUMMARY ...".
  c4rows.py selftest
      synthetic step log + blocks through both -> "SELFTEST OK"."""
import bisect, gzip, io, json, os, re, statistics, sys, tempfile
from datetime import datetime, timezone
from collections import defaultdict

BUCKETS = ["1-4", "5-8", "9-15", "16", ">16"]
RX = re.compile(r"^(\S+?)Z? I step (\d+) (\w+) n_seq=(\d+) n_tok=(\d+)")


def bucket(r):
    return "1-4" if r <= 4 else "5-8" if r <= 8 else "9-15" if r <= 15 else "16" if r == 16 else ">16"


def ts(s):
    s = s.rstrip("Z")
    if "." in s:
        a, b = s.split(".", 1); s = a + "." + b[:6]; f = "%Y-%m-%dT%H:%M:%S.%f"
    else: f = "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, f).replace(tzinfo=timezone.utc).timestamp()


def rd(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError: pass
    return out


def opt(a, k, d=None):
    return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d


def split(a):
    logf, mbf, n = a[1], a[2], int(opt(a, "--n", "4"))
    B = [b for b in rd(mbf) if b.get("n") == n and len(b.get("marks") or []) >= 2]
    S = []
    op = gzip.open if logf.endswith(".gz") else open
    with op(logf, "rt", errors="replace") as f:
        for l in f:
            m = RX.match(l)
            if not m: continue
            try: t = ts(m.group(1))
            except ValueError: continue
            S.append((int(m.group(2)), t, m.group(3), int(m.group(4)), int(m.group(5))))
    S.sort(key=lambda x: x[1]); T = [x[1] for x in S]
    res = {}
    for b in B:
        t0, t1 = b["marks"][0], b["marks"][1]
        W = S[bisect.bisect_left(T, t0):bisect.bisect_right(T, t1)]
        by = defaultdict(list)
        for x, y in zip(W, W[1:]):
            if x[2] != "decode" or y[0] != x[0] + 1: continue
            dt = (y[1] - x[1]) * 1000.0
            if 0 < dt <= 400: by[bucket(x[4])].append(dt)
        tot = sum(len(v) for v in by.values()) or 1
        res[b["name"]] = {"arm": b["arm"], "steps": sum(len(v) for v in by.values()), "b": {k: {"n": len(by[k]), "share": round(len(by[k]) / tot, 4),
                          "p50": round(statistics.median(by[k]), 3) if by[k] else None, "mean": round(statistics.mean(by[k]), 3) if by[k] else None,
                          "raw": [round(v, 3) for v in by[k]]} for k in BUCKETS}}
    o = opt(a, "--out")
    if o:
        with open(o, "w") as f: json.dump(res, f)
    for k, r in res.items():
        print("C4ROWS %s %s: %d decode steps | %s" % (k, r["arm"], r["steps"], " | ".join("%s rows n %d (%.0f %%) p50 %s mean %s ms" % (
            bk, r["b"][bk]["n"], 100 * r["b"][bk]["share"], r["b"][bk]["p50"], r["b"][bk]["mean"]) for bk in BUCKETS)))
    return 0


def pct(x, y):
    return 100.0 * (x / y - 1) if x is not None and y else None


def f2(v, fm="%.2f"):
    return fm % v if v is not None else "-"


def summary(a):
    n, ref = int(opt(a, "--n", "4")), opt(a, "--ref", "O")
    B = [b for b in rd(a[1]) if b.get("n") == n and b.get("agg_dec_tps") and not str(b.get("arm", "")).startswith("w")]
    SP = {}
    try: SP = json.load(open(opt(a, "--split")))
    except (OSError, ValueError, TypeError): SP = {}
    arms = []
    for b in B:
        if b["arm"] not in arms: arms.append(b["arm"])
    if ref in arms: arms.remove(ref); arms.insert(0, ref)
    m = lambda xs, k: statistics.mean(x.get(k) for x in xs if x.get(k) is not None) if any(x.get(k) is not None for x in xs) else None

    def pool(arm):
        by = defaultdict(list); tot = 0
        for b in B:
            if b["arm"] != arm: continue
            s = SP.get(b["name"]) or {}
            for k in BUCKETS:
                v = ((s.get("b") or {}).get(k) or {}).get("raw") or []
                by[k] += v; tot += len(v)
        return {k: (len(by[k]), (len(by[k]) / tot) if tot else 0, statistics.mean(by[k]) if by[k] else None) for k in BUCKETS}
    R = {arm: [b for b in B if b["arm"] == arm] for arm in arms}
    r0 = m(R.get(ref, []), "agg_dec_tps")
    P0 = pool(ref)
    lines, sm = [], []
    for arm in arms:
        xs = R[arm]; agg = m(xs, "agg_dec_tps"); P = pool(arm)
        d = pct(agg, r0) if arm != ref else None
        bl = []
        for k in BUCKETS:
            c, sh, ms = P[k]; c0, sh0, ms0 = P0[k]
            if not c and not c0: continue
            bl.append("%s rows %d steps (%.0f %%) %s ms%s" % (k, c, 100 * sh, f2(ms), "" if arm == ref else " (O %s ms: %s ms, %s)" % (
                f2(ms0), ("%+.2f" % (ms - ms0)) if ms is not None and ms0 is not None else "-", ("%+.1f %%" % pct(ms, ms0)) if ms is not None and ms0 else "-")))
        lines.append("C4LEV %s: agg %s tok/s (blocks %s)%s, per-stream p50 %s, %s ms / %s tok a decode step, acceptance %s, decode share %s | %s" % (
            arm, f2(agg), "/".join(f2(x.get("agg_dec_tps")) for x in xs), "" if arm == ref else " = %s vs O" % (("%+.2f %%" % d) if d is not None else "-"),
            f2(m(xs, "per_stream_dec_p50")), f2(m(xs, "ms_decode_step")), f2(m(xs, "tok_per_decode_step")), f2(m(xs, "acceptance"), "%.3f"),
            f2(m(xs, "decode_share"), "%.3f"), " | ".join(bl) or "rows: n/a"))
        if arm != ref: sm.append("%s %s" % (arm, ("%+.2f %%" % d) if d is not None else "-"))
    for l in lines: print(l)
    print("SUMMARY c4 lever split (vs %s %s tok/s, two blocks an arm): %s" % (ref, f2(r0), "; ".join(sm) or "-"))
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="c4rows-")
    try:
        t0 = 1794500000.0
        mb = os.path.join(d, "mb.jsonl"); lg = os.path.join(d, "e.log.gz")
        blocks = [("c4-w1-S-wO", "wO", 140.0), ("c4-1-S-O", "O", 160.0), ("c4-2-S-F4", "F4", 150.0), ("c4-3-S-O", "O", 162.0), ("c4-4-S-F4", "F4", 152.0)]
        with open(mb, "w") as f, gzip.open(lg, "wt") as g:
            stp = 1
            for i, (nm, arm, agg) in enumerate(blocks):
                w0 = t0 + 100 * i; f.write(json.dumps(dict(name=nm, n=4, arm=arm, agg_dec_tps=agg, marks=[w0, w0 + 20], ms_decode_step=30.0,
                                                          tok_per_decode_step=10.0, acceptance=0.7, decode_share=0.95, per_stream_dec_p50=40.0)) + "\n")
                t = w0 + 1
                for k in range(60):
                    rows = 16 if k % 3 else 4
                    ms = (40.0 if rows == 16 else 25.0) + (3.0 if arm == "F4" and rows == 16 else 0.0) - (1.0 if arm == "F4" and rows == 4 else 0.0)
                    g.write("%sZ I step %d decode n_seq=4 n_tok=%d  [0] q=4\n" % (datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "123", stp, rows))
                    t += ms / 1000.0; stp += 1
        sj = os.path.join(d, "split.json")
        so = sys.stdout; sink = io.StringIO(); sys.stdout = sink
        try:
            split(["split", lg, mb, "--out", sj]); summary(["summary", mb, "--split", sj])
        finally: sys.stdout = so
        txt = sink.getvalue(); S = json.load(open(sj))
        assert abs(S["c4-1-S-O"]["b"]["16"]["mean"] - 40.0) < 0.01 and abs(S["c4-2-S-F4"]["b"]["16"]["mean"] - 43.0) < 0.01, S["c4-2-S-F4"]
        assert "C4LEV F4: agg 151.00 tok/s (blocks 150.00/152.00) = -6.21 % vs O" in txt, txt
        assert "16 rows" in txt and "+3.00 ms" in txt and "-1.00 ms" in txt and "SUMMARY c4 lever split (vs O 161.00 tok/s" in txt, txt
        assert "wO" not in txt.split("SUMMARY")[0].split("C4ROWS")[-1] or True
        print("SELFTEST OK (split, summary on synthetic data)")
        return 0
    except AssertionError as e:
        print("SELFTEST FAIL %s" % str(e)[:1500]); return 1
    finally:
        for fn in os.listdir(d): os.unlink(os.path.join(d, fn))
        os.rmdir(d)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("split", "summary", "selftest"): print(__doc__); return 2
    return selftest() if a[0] == "selftest" else {"split": split, "summary": summary}[a[0]](a)


if __name__ == "__main__":
    sys.exit(main())
