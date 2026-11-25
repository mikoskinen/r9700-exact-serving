#!/usr/bin/env python3
"""oracle.py ref|check URL IDS OUT (10.11.2026, queue-ar dsmain) -- the verify-path harness's client (Radiance
--debug-draft-oracle, tests/engine_e2e_test.cpp's check): one fixed text prompt to the native /completion endpoint,
greedy (temperature 0), ignore_eos, N tokens (ORACLE_N, default 256), return_tokens.
  ref    on a plain server (no speculation): write the reply's token ids to IDS (the server's oracle file) and the
         reply to OUT
  check  on a server started with --spec-ngram K --debug-draft-oracle IDS (prompt lookup proposes from those ids with
         30 % of the drafts corrupted, so verifies accept and reject at every depth): the reply's ids must be IDS's
Prints one ORACLE line (JSON). Standard library only."""
import json, os, sys, urllib.request

PROMPT = ("The following is a careful technical explanation of how a hash table resolves collisions, written for a "
          "reader who knows arrays and linked lists but has never implemented one. It covers separate chaining, open "
          "addressing with linear and quadratic probing, the load factor, resizing, and why a good hash function "
          "matters, with a short worked example after each part.\n\n")


def post(url, body):
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ("ref", "check"):
        print(__doc__); return 2
    mode, url, ids_path, out = sys.argv[1:]
    n = int(os.environ.get("ORACLE_N", "256"))
    rep = post(url, {"prompt": PROMPT, "n_predict": n, "temperature": 0.0, "ignore_eos": True, "return_tokens": True})
    with open(out, "w") as f: json.dump(rep, f)
    toks = rep.get("tokens") or []
    tm = rep.get("timings") or {}
    if mode == "ref":
        with open(ids_path, "w") as f: f.write(" ".join(str(t) for t in toks) + "\n")
        print("ORACLE " + json.dumps({"mode": "ref", "tokens": len(toks)}))
        return 0 if toks else 1
    ref = [int(x) for x in open(ids_path).read().split()]
    first = next((i for i, (a, b) in enumerate(zip(ref, toks)) if a != b), None if len(ref) == len(toks) else min(len(ref), len(toks)))
    ok = ref == toks
    print("ORACLE " + json.dumps({"mode": "check", "identical": ok, "tokens": len(toks), "ref_tokens": len(ref), "first_diff": first,
                                  "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
