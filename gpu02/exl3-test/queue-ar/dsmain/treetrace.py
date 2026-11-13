#!/usr/bin/env python3
"""treetrace.py ENGINE_LOG ORACLE_IDS [N_VOCAB] (11.11.2026, queue-ar dsmain, ds-main-port T1 diagnosis) -- reads the engine's
RADIANCE_DEBUG_TREE_STEPS lines ("tree trace: step S seq Q out L ctx C nspec N rows R | ids .. | pos .. | spec_parent .. |
draft .. | parent .. | target .. | acc A path ..") against the plain run's ids R (the oracle file) and says where a tree
verify goes wrong, in five readings, one TRACE line each:
  harness   the main chain's nodes 1..K (K = the first node whose parent is the root again, else 5) carry R[L + d - 1] or the
            harness's corruption (R + 1) mod V at depth d: else the oracle's ids are misaligned with the output;
  staging   NOT A READING (11.11.): the trace reads the batch's card arrays after the drafter's passes restaged them (tree3's
            passing or20t0 "mismatched" in every step, positions = the query pass's block), so it is printed as information only;
  root      target[0] = R[L] (the row that sees only committed context), steps before the first divergence;
  rows      every node whose path's drafts are R[L .. L + d - 1] exactly: its target must be R[L + d] (what the chain computes
            there), counted by depth and by main chain / branch, ONLY in the steps up to and including the first divergence
            (after it the context is no longer the plain run's, so R says nothing about a row) -- one STEP line each;
  accept    acceptance as accept_tree's rule over the logged targets vs the logged acc (a host-rule check).
Then FIRSTBAD: the first step where the output stops matching R, with that step's line. Standard library only."""
import re, sys


def ints(s):
    s = s.strip()
    if s in ("", "chain"): return None if s == "chain" else []
    return [int(x) for x in s.split()]


def parse(path):
    out = []
    pat = re.compile(r"tree trace: step (\d+) seq (\d+) out (\d+) ctx (-?\d+) nspec (\d+) rows (\d+) \| ids (.*?) \| pos (.*?) \| "
                     r"spec_parent (.*?) \| draft (.*?) \| parent (.*?) \| target (.*?) \| acc (\d+) path ?(.*)$")
    for line in open(path, errors="replace"):
        m = pat.search(line.rstrip("\n"))
        if not m: continue
        g = m.groups()
        out.append({"step": int(g[0]), "out": int(g[2]), "ctx": int(g[3]), "n": int(g[4]), "rows": int(g[5]),
                    "ids": ints(g[6]), "pos": ints(g[7]), "sp": ints(g[8]), "draft": ints(g[9]), "par": ints(g[10]),
                    "target": ints(g[11]), "acc": int(g[12]), "path": ints(g[13]) or []})
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    T = parse(sys.argv[1])
    R = [int(x) for x in open(sys.argv[2]).read().split()]
    V = int(sys.argv[3]) if len(sys.argv) > 3 else 129280
    if not T:
        print("TRACE none: no 'tree trace:' lines in %s (RADIANCE_DEBUG_TREE_STEPS unset, or no tree step ran)" % sys.argv[1]); return 1
    h_bad = st_bad = root_ok = root_bad = acc_bad = 0
    rows = {}                # (kind, depth) -> [ok, bad]
    first_bad_row = None
    out_ok_until = None      # the first output length L whose step emitted a token != R
    ex = {"harness": None, "staging": None}
    first_bad_L = None
    for t in T:
        L, d, tg = t["out"], t["draft"], t["target"]
        em = [tg[0]] if t["acc"] == 0 else [d[q - 1] for q in t["path"]] + [tg[t["path"][-1]]]
        if any(L + i < len(R) and em[i] != R[L + i] for i in range(len(em))):
            first_bad_L = L; break
    for t in T:
        L, n, d, tg, par = t["out"], t["n"], t["draft"], t["target"], t["par"]
        pre = first_bad_L is None or L <= first_bad_L
        P = [None] + [(par[j - 1] if par is not None else j - 1) for j in range(1, n + 1)]
        for j in range(1, n + 1):
            if P[j] is None or P[j] < 0 or P[j] >= j: P[j] = j - 1
        dep = [0] * (n + 1)
        for j in range(1, n + 1): dep[j] = dep[P[j]] + 1
        K = next((j - 1 for j in range(2, n + 1) if P[j] == 0), n)
        # harness
        hb = [j for j in range(1, min(K, n) + 1) if L + dep[j] - 1 < len(R) and d[j - 1] not in (R[L + dep[j] - 1], (R[L + dep[j] - 1] + 1) % V)]
        if hb:
            h_bad += 1
            if ex["harness"] is None: ex["harness"] = "step %d L %d: main nodes %s carry %s, R[L..] = %s" % (t["step"], L, hb[:5], [d[j - 1] for j in hb[:5]], R[L:L + 5])
        # staging
        sb = []
        ids, pos, sp = t["ids"], t["pos"], t["sp"]
        if ids and len(ids) == n + 1:
            if L >= 1 and L - 1 < len(R) and (out_ok_until is None) and ids[0] != R[L - 1]: sb.append("ids[0] %d != anchor R[L-1] %d" % (ids[0], R[L - 1]))
            if ids[1:] != d: sb.append("ids[1..] != draft")
        if sp and len(sp) == n + 1:
            if sp[0] != -1: sb.append("spec_parent[0] %d" % sp[0])
            if [x - 0 for x in sp[1:]] != P[1:]: sb.append("spec_parent %s vs parents %s" % (sp[1:8], P[1:8]))
        if pos and len(pos) == n + 1:
            want = [t["ctx"] + dep[j] for j in range(n + 1)]
            if pos != want: sb.append("pos %s vs ctx + depth %s" % (pos[:8], want[:8]))
        if sb:
            st_bad += 1
            if ex["staging"] is None: ex["staging"] = "step %d: %s" % (t["step"], "; ".join(sb))
        # root (before the output diverges)
        if L < len(R) and out_ok_until is None:
            if tg[0] == R[L]: root_ok += 1
            else: root_bad += 1
        # rows on a correct path (pre-divergence steps only)
        srow = []
        for j in range(1, n + 1 if pre else 1):
            path, q = [], j
            while q > 0: path.append(q); q = P[q]
            path.reverse()
            dd = dep[j]
            if L + dd >= len(R): continue
            if all(d[q - 1] == R[L + dep[q] - 1] for q in path):
                kind = "main" if path[0] == 1 else "branch"
                c = rows.setdefault((kind, dd), [0, 0])
                srow.append("%d(%s d%d):%s" % (j, kind[0], dd, "ok" if tg[j] == R[L + dd] else "BAD %d != %d" % (tg[j], R[L + dd])))
                if tg[j] == R[L + dd]: c[0] += 1
                else:
                    c[1] += 1
                    if first_bad_row is None:
                        first_bad_row = "step %d L %d node %d (%s, depth %d): target %d, the chain's R[L+d] %d" % (t["step"], L, j, kind, dd, tg[j], R[L + dd])
        if pre: print("STEP %d out %d ctx %d acc %d path %s | root %s | %s" % (t["step"], L, t["ctx"], t["acc"], t["path"], "ok" if L < len(R) and tg[0] == R[L] else "BAD", " ".join(srow) or "-"))
        # acceptance by the rule
        cur, acc = 0, 0
        while True:
            nxt = next((j for j in range(cur + 1, n + 1) if P[j] == cur and d[j - 1] == tg[cur]), None)
            if nxt is None: break
            acc += 1; cur = nxt
        if acc != t["acc"]: acc_bad += 1
        # the emitted tokens vs R
        emitted = [tg[0]] if t["acc"] == 0 else [d[q - 1] for q in t["path"]] + [tg[t["path"][-1]]]
        if out_ok_until is None and any(L + i < len(R) and emitted[i] != R[L + i] for i in range(len(emitted))):
            out_ok_until = L
            print("FIRSTBAD step %d at output %d: emitted %s, R %s | acc %d path %s | targets on the path %s" % (
                t["step"], L, emitted, R[L:L + len(emitted)], t["acc"], t["path"], [tg[q] for q in [0] + t["path"]]))
    print("TRACE steps %d (first output %d, last %d)" % (len(T), T[0]["out"], T[-1]["out"]))
    print("TRACE harness: %d of %d steps carry main-chain ids that are neither R nor its corruption%s" % (h_bad, len(T), (" -- e.g. " + ex["harness"]) if ex["harness"] else ""))
    print("TRACE staging (information only, not a reading: card arrays read after the drafter passes restaged them): %d of %d steps differ%s" % (st_bad, len(T), (" -- e.g. " + ex["staging"]) if ex["staging"] else ""))
    print("TRACE root: target[0] = R[L] in %d, not in %d steps before the first divergence" % (root_ok, root_bad))
    print("TRACE rows on a correct path, steps up to the first divergence (target = the chain's next id): %s%s" % (
        ", ".join("%s d%d %d ok %d bad" % (k[0], k[1], v[0], v[1]) for k, v in sorted(rows.items())) or "none",
        (" -- first bad: " + first_bad_row) if first_bad_row else ""))
    print("TRACE accept: the logged acc differs from accept_tree's rule over the logged targets in %d steps" % acc_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
