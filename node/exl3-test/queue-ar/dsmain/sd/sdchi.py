#!/usr/bin/env python3
"""sdchi.py (11.11.2026, queue-ar/dsmain/sd, lever SD item) -- the on-card distribution check: a fixed prompt, N seeded requests
at the server-default sampling (no temperature / top_p / top_k sent: the engine's temperature 1), the first two generated
tokens' text as the category; two such samples (lever SD on vs non-speculative) compared by a chi-square test of homogeneity.

  sdchi.py collect --out F.jsonl [--url U] [--n 2000] [--seed0 1] [--conc 4] [--max-tokens 2] [--prompt P]
  sdchi.py compare A.jsonl B.jsonl [--min-exp 5]

collect: POST /v1/completions {prompt, max_tokens, seed} for seeds seed0 .. seed0 + n - 1, --conc at a time; one JSON line a
  request (seed, status, text, finish, completion_tokens), then "COLLECT {json}" (n ok, distinct categories, seconds).
  The second generated token is the first a drafter proposes (the first comes from the prefill's own sampling), so with
  max_tokens 2 every category is (the prefill's draw, the verify's row-0 token); max_tokens is not a draft cap (the verify
  carries the window and the commit cuts), so the coupled row is reached.
compare: the 2 x K contingency table over the categories (a category whose expected count is below --min-exp in either
  sample is pooled into "other"), Pearson chi-square, df = K' - 1, p-value from the regularised upper incomplete gamma;
  "CHI {json}" (chi2, df, p, n_a, n_b, cells, the ten most common categories with both counts). Disjoint seed ranges make the
  two samples independent. Standard library only."""
import argparse
import json
import math
import sys
import threading
import time
import urllib.request

PROMPT = "Here is a list of random fruits: apple, banana,"


def gammq(a, x):
    """Q(a, x) = Gamma(a, x) / Gamma(a), the chi-square tail at 2x with 2a degrees of freedom."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(5000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 5000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def collect(a):
    seeds = list(range(a.seed0, a.seed0 + a.n))
    out, lock, idx = [], threading.Lock(), [0]
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if idx[0] >= len(seeds):
                    return
                s = seeds[idx[0]]
                idx[0] += 1
            row = {"seed": s}
            try:
                st, d = post(a.url + "/v1/completions", {"prompt": a.prompt, "max_tokens": a.max_tokens, "seed": s})
                ch = (d.get("choices") or [{}])[0]
                row.update(status=st, text=ch.get("text"), finish=ch.get("finish_reason"),
                           completion_tokens=(d.get("usage") or {}).get("completion_tokens"))
            except Exception as e:   # noqa: BLE001 -- recorded, the request is left out
                row.update(status=-1, error=str(e)[:200])
            with lock:
                out.append(row)

    th = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    out.sort(key=lambda r: r["seed"])
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if r.get("status") == 200 and r.get("text") is not None]
    print("COLLECT " + json.dumps({"out": a.out, "n": len(out), "ok": len(ok), "distinct": len({r["text"] for r in ok}),
                                   "seconds": round(time.time() - t0, 1), "prompt": a.prompt, "max_tokens": a.max_tokens,
                                   "seed0": a.seed0}, ensure_ascii=False), flush=True)
    return 0 if len(ok) == len(out) else 1


def load(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return [r["text"] for r in rows if r.get("status") == 200 and r.get("text") is not None]


def compare(a):
    xa, xb = load(a.a), load(a.b)
    na, nb = len(xa), len(xb)
    if not na or not nb:
        print("CHI " + json.dumps({"error": "empty sample", "n_a": na, "n_b": nb}))
        return 1
    ca, cb = {}, {}
    for t in xa:
        ca[t] = ca.get(t, 0) + 1
    for t in xb:
        cb[t] = cb.get(t, 0) + 1
    n = na + nb
    keep, other_a, other_b = [], 0, 0
    for k in set(ca) | set(cb):
        tot = ca.get(k, 0) + cb.get(k, 0)
        if tot * na / n >= a.min_exp and tot * nb / n >= a.min_exp:
            keep.append(k)
        else:
            other_a += ca.get(k, 0)
            other_b += cb.get(k, 0)
    cells = [(ca.get(k, 0), cb.get(k, 0)) for k in keep]
    if other_a + other_b > 0:
        cells.append((other_a, other_b))
    chi = 0.0
    for oa, ob in cells:
        tot = oa + ob
        ea, eb = tot * na / n, tot * nb / n
        chi += (oa - ea) ** 2 / ea + (ob - eb) ** 2 / eb
    df = len(cells) - 1
    p = gammq(df / 2.0, chi / 2.0) if df > 0 else 1.0
    top = sorted(keep, key=lambda k: -(ca.get(k, 0) + cb.get(k, 0)))[:10]
    print("CHI " + json.dumps({"chi2": round(chi, 3), "df": df, "p": round(p, 4), "n_a": na, "n_b": nb,
                               "cells": len(cells), "pooled_a": other_a, "pooled_b": other_b,
                               "top": [[k, ca.get(k, 0), cb.get(k, 0)] for k in top]}, ensure_ascii=False), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8090")
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--seed0", type=int, default=1)
    c.add_argument("--conc", type=int, default=4)
    c.add_argument("--max-tokens", type=int, default=2)
    c.add_argument("--prompt", default=PROMPT)
    m = sp.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--min-exp", type=float, default=5.0)
    m.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return collect(a) if a.cmd == "collect" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
