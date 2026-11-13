#!/usr/bin/env python3
"""rowms.py (queue-ar/dsmain/1ub, int11b XP screen) -- ms a decode step by verify-row count, per 1u essay window, from the
engine's step log (RADIANCE_LOG_STEPS=1, `docker logs -t`: "<ts>Z I step N decode n_seq=1 n_tok=K ...").

  rowms.py LOG.gz WINDOWS.jsonl     WINDOWS: one {"name", "arm", "t0", "t1"} a line (UTC epoch seconds, the essay's request span)
  rowms.py selftest

For each window: the decode steps with n_seq 1 whose next logged step is the next step number (consecutive), wall ms = the
next step's timestamp - this one's; grouped by n_tok (the rows the step verified: 1 = no draft); a line
  ROWMS {"name", "arm", "steps", "rows": {"1": {"n", "p50", "mean"}, ...}, "rows_mean", "ms_mean"}
and one ROWMSARM line per arm (every window of the arm pooled). Reads only; no verdict."""
import gzip, json, re, statistics as st, sys
from datetime import datetime, timezone

RX = re.compile(r"^(\S+?)Z? I(?:\[\d\])? step (\d+) (\w+) n_seq=(\d+) n_tok=(\d+)")


def ts(s):
    s = s.rstrip("Z")
    if "." in s:
        a, b = s.split(".", 1)
        s = a + "." + (b + "000000")[:6]
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def steps(lines):
    out = []
    for l in lines:
        m = RX.match(l)
        if not m:
            continue
        try:
            t = ts(m.group(1))
        except ValueError:
            continue
        out.append((int(m.group(2)), t, m.group(3), int(m.group(4)), int(m.group(5))))
    return out


def group(S, t0, t1):
    rows = {}
    for i in range(len(S) - 1):
        s, nx = S[i], S[i + 1]
        if s[2] != "decode" or s[3] != 1 or not (t0 <= s[1] <= t1) or nx[0] != s[0] + 1:
            continue
        rows.setdefault(s[4], []).append((nx[1] - s[1]) * 1000.0)
    return rows


def summ(rows):
    n = sum(len(v) for v in rows.values())
    r = {str(k): dict(n=len(v), p50=round(st.median(v), 2), mean=round(st.mean(v), 2)) for k, v in sorted(rows.items())}
    allv = [x for v in rows.values() for x in v]
    return dict(steps=n, rows=r, rows_mean=round(sum(k * len(v) for k, v in rows.items()) / n, 3) if n else None,
                ms_mean=round(st.mean(allv), 2) if allv else None)


def main(log, win):
    op = gzip.open if log.endswith(".gz") else open
    with op(log, "rt", errors="replace") as f:
        S = steps(f)
    W = [json.loads(l) for l in open(win) if l.strip()]
    pooled = {}
    for w in W:
        g = group(S, float(w["t0"]), float(w["t1"]))
        print("ROWMS " + json.dumps(dict(name=w.get("name"), arm=w.get("arm"), **summ(g))))
        for k, v in g.items():
            pooled.setdefault(w.get("arm"), {}).setdefault(k, []).extend(v)
    for a, g in sorted(pooled.items()):
        print("ROWMSARM " + json.dumps(dict(arm=a, **summ(g))))


def selftest():
    base = 1700000000.0
    L = []
    t = base
    for i, (k, ms) in enumerate([(1, 17.0), (2, 20.5), (1, 17.2), (3, 24.0), (1, 17.4), (2, 20.7)]):
        L.append("%sZ I step %d decode n_seq=1 n_tok=%d" % (datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"), 100 + i, k))
        t += ms / 1000.0
    L.append("%sZ I step 106 decode n_seq=1 n_tok=1" % datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"))
    S = steps(L)
    g = summ(group(S, base - 1, t + 1))
    assert g["steps"] == 6 and g["rows"]["1"]["n"] == 3 and abs(g["rows"]["2"]["p50"] - 20.6) < 0.05, g
    assert abs(g["rows"]["3"]["mean"] - 24.0) < 0.05 and abs(g["rows_mean"] - 10 / 6) < 0.01, g
    print("SELFTEST OK rowms")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
