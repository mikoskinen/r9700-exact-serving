#!/usr/bin/env python3
"""epplace.py (10.11.2026, queue-ar dsmain, ds41-slot agent, levers D3s / D3c / F1) -- expert placement files for
RADIANCE_DS41_EP_PLACE / _EP_PLACE2 (deepseek41.cpp, "Lever D3s"): one line "L <layer> <mask>" a layer, the mask one hex
digit a model expert (bit r = rank r holds a decode copy), and optionally "C <layer> <c0,c1,...>" (each expert's load units
for the replicated experts' least-loaded dispatch, default 16). A layer the file does not name keeps the contiguous split
(expert e on rank e // (E / 4)). Every rank still holds its home run for prefill; a decode copy off home is an EXTRA expert
on that rank (a movement unit like any other: VRAM or the pinned host pool).

  epplace.py scramble  --experts 384 --layers 0-39 --per-rank 6 --seed 1 -o F    invariance test: per layer and rank, 6 experts
                                                                                  of the other ranks moved onto it (one copy)
  epplace.py replicate --stats S.json --hot 32 [--kind decode] -o F               lever F1: the H hottest experts a layer on
                                                                                  every rank (least-loaded copy per step)
  epplace.py solve     --stats S.json --rows 8 [--budget 12] [--hot 0] -o F       lever D3c: a per-layer placement that lowers
                       [--A 35 --B 18.8 --M 400 --slots 1763] [--draws 300]       the expected slowest-rank MoE time (below)
  epplace.py eval      --stats S.json --rows 8 [--place F ...]                    the model's numbers for each file (and contiguous)
  epplace.py measure   --miss MISS.log [--from-step N] [--kind decode]           measured skew from a RADIANCE_DEBUG_MISS_LOG
  epplace.py extras    --place F [--place F2] --experts 384                       each rank's extra experts (as deepseek41 declares them)

The cost model (queue-ar/tools/balsim.py, the caller's balrows.py): a rank's MoE time in a layer is A + B x (its live experts
resident in VRAM) + (B + M) x (its live experts missed, read from the host pool), 0 with none live; the step's slowest rank
sets the layer's time, the others wait at the all-reduce. A step of R rows draws each token's top-k without replacement in
proportion to the layer's decode counts (--stats: a radiance-expert-stats file, `counts.<kind>[layer][model expert]`,
recorded at the contiguous placement, where every count is the model's). Residency: each rank keeps its --slots hottest units
(by the decode traffic each copy gets) in VRAM, as an ideal heat engine would. Replicated experts go, per step, to the copy
the kernel's rule picks (moe_scatter_placed: single-copy loads first, then the replicated ones in ascending id each to the
least-loaded copy by cost units, ties the first in rotation from id % 4), so the model dispatches exactly as the card does.

solve (D3c): per layer, from the contiguous split (plus --hot replication), a greedy local search: take the rank with the
largest expected time and try moving each of its live-heavy experts to each other rank (the expert's decode copy leaves home,
an extra on the destination, at most --budget extras a rank a layer), keep the move that lowers the layer's mean slowest-rank
time over the drawn steps most, until none does. Costs (C lines) from the expected miss rate: 16 x (B + M p_miss) / B.
Iterate in rounds by re-running with the stats of the new placement's run (routing frequencies do not depend on placement)
and the measured skew (measure) as the check. Standard library only."""
import argparse, json, math, random, re, sys

W = 4


def read_stats(path, kind):
    j = json.load(open(path))
    C = j["counts"][kind]
    tk = j["top_k"]
    return C, tk, j.get("tp", W)


def parse_layers(s, n):
    if s in (None, "", "all"):
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l < n]


def write_place(path, masks, costs=None, head=""):
    with open(path, "w") as f:
        if head:
            for h in head.splitlines():
                f.write("# " + h + "\n")
        for l in sorted(masks):
            f.write("L %d %s\n" % (l, "".join("0123456789abcdef"[m] for m in masks[l])))
            if costs and l in costs:
                f.write("C %d %s\n" % (l, ",".join(str(int(c)) for c in costs[l])))


def read_place(path):
    masks, costs = {}, {}
    for ln in open(path):
        ln = ln.split("#")[0].split()
        if len(ln) != 3:
            continue
        if ln[0] == "L":
            masks[int(ln[1])] = [int(ch, 16) for ch in ln[2]]
        elif ln[0] == "C":
            costs[int(ln[1])] = [int(x) for x in ln[2].split(",")]
    return masks, costs


def contiguous(E):
    nh = E // W
    return [1 << (e // nh) for e in range(E)]


def extras_of(mask_tables, E):
    """each rank's extras as deepseek41's slot_cfg declares them: the union over the tables, sorted, padded to an even
    count with the smallest model expert not already held."""
    nh = E // W
    out = []
    for r in range(W):
        xs = set()
        for m in mask_tables:
            for e in range(E):
                if (m[e] >> r) & 1 and e // nh != r:
                    xs.add(e)
        xs = sorted(xs)
        if (nh + len(xs)) % 2:
            for e in range(E):
                if e // nh != r and e not in xs:
                    xs = sorted(xs + [e])
                    break
        out.append(xs)
    return out


# ---------------------------------------------------------------- the model

class Model:
    def __init__(self, C, tk, layers, rows, draws, A, B, M, slots, seed=1):
        self.C, self.tk, self.L, self.R = C, tk, layers, rows
        self.A, self.B, self.M, self.S = A, B, M, slots
        self.E = len(C[layers[0]])
        rnd = random.Random(seed)
        self.D = {}
        for l in layers:
            w = C[l]
            idx = list(range(self.E))
            tot = float(sum(w)) or 1.0
            cum, acc = [], 0.0
            for x in w:
                acc += x
                cum.append(acc / tot)
            def pick():
                u = rnd.random()
                lo, hi = 0, self.E - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cum[mid] < u: lo = mid + 1
                    else: hi = mid
                return lo
            steps = []
            for _ in range(draws):
                s = set()
                for _ in range(rows):
                    ch = set()
                    while len(ch) < min(tk[l], self.E):
                        ch.add(pick())
                    s |= ch
                steps.append(sorted(s))
            self.D[l] = steps

    def dispatch(self, mask, cost, active):
        """the kernel's rule: -> {expert: rank}"""
        ld = [0] * W
        own = {}
        rep = []
        for e in active:
            m = mask[e]
            if bin(m).count("1") == 1:
                r = (m & -m).bit_length() - 1
                own[e] = r
                ld[r] += cost[e]
            else:
                rep.append(e)
        for e in sorted(rep):
            m, best = mask[e], -1
            for s in range(W):
                r = (e + s) % W
                if (m >> r) & 1 and (best < 0 or ld[r] < ld[best]):
                    best = r
            own[e] = best
            ld[best] += cost[e]
        return own

    def traffic(self, masks, costs):
        """per (rank, layer, expert): the share of steps the copy is live, from the drawn steps"""
        T = {}
        for l in self.L:
            n = len(self.D[l])
            for s in self.D[l]:
                for e, r in self.dispatch(masks[l], costs[l], s).items():
                    T[(r, l, e)] = T.get((r, l, e), 0.0) + 1.0 / n
        return T

    def resident(self, T):
        res = [set() for _ in range(W)]
        for r in range(W):
            u = sorted(((v, l, e) for (q, l, e), v in T.items() if q == r), reverse=True)
            res[r] = {(l, e) for _, l, e in u[:self.S]}
        return res

    def layer_time(self, l, mask, cost, res, missfree=False):
        tm = tw = 0.0
        for s in self.D[l]:
            own = self.dispatch(mask, cost, s)
            t = [0.0] * W
            n = [0] * W
            for e, r in own.items():
                n[r] += 1
                t[r] += self.B + (0.0 if missfree or (l, e) in res[r] else self.M)
            t = [(self.A + x) if k else 0.0 for x, k in zip(t, n)]
            tm += max(t)
            tw += max(t) - sum(t) / W
        k = len(self.D[l])
        return tm / k, tw / k

    def total(self, masks, costs, missfree=False):
        res = self.resident(self.traffic(masks, costs))
        tm = tw = 0.0
        for l in self.L:
            a, b = self.layer_time(l, masks[l], costs[l], res, missfree)
            tm += a
            tw += b
        return tm / 1000.0, tw / 1000.0, res


def cmd_scramble(a):
    rnd = random.Random(a.seed)
    E, nh = a.experts, a.experts // W
    masks = {}
    for l in parse_layers(a.layers, 4096):
        m = contiguous(E)
        moved = set()
        for r in range(W):
            cand = [e for e in range(E) if e // nh != r and e not in moved]
            for e in rnd.sample(cand, a.per_rank):
                m[e] = 1 << r
                moved.add(e)
        masks[l] = m
    write_place(a.o, masks, head="epplace.py scramble --experts %d --layers %s --per-rank %d --seed %d: per layer and rank, "
                "%d experts of the other ranks moved onto it (one decode copy)" % (E, a.layers, a.per_rank, a.seed, a.per_rank))
    xs = extras_of([masks[l] for l in masks][:1], E)
    print("EPPLACE scramble %d layers, %d experts moved a layer, extras a rank a layer %s -> %s" %
          (len(masks), W * a.per_rank, [len(x) for x in xs], a.o))


def hottest(C, l, h):
    return sorted(range(len(C[l])), key=lambda e: -C[l][e])[:h]


def cmd_replicate(a):
    C, tk, _ = read_stats(a.stats, a.kind)
    layers = [l for l in parse_layers(a.layers, len(C)) if sum(C[l]) > 0]
    masks = {}
    for l in layers:
        m = contiguous(len(C[l]))
        for e in hottest(C, l, a.hot):
            m[e] = (1 << W) - 1
        masks[l] = m
    write_place(a.o, masks, head="epplace.py replicate --stats %s --hot %d: lever F1, the %d hottest experts a layer (%s "
                "counts) on every rank" % (a.stats, a.hot, a.hot, a.kind))
    print("EPPLACE replicate %d layers, %d experts a layer on every rank (%d extras a rank a layer) -> %s" %
          (len(masks), a.hot, a.hot * (W - 1) // W, a.o))


def cmd_solve(a):
    C, tk, _ = read_stats(a.stats, a.kind)
    layers = [l for l in parse_layers(a.layers, len(C)) if sum(C[l]) > 0]
    mdl = Model(C, tk, layers, a.rows, a.draws, a.A, a.B, a.M, a.slots, a.seed)
    E, nh = mdl.E, mdl.E // W
    masks = {l: contiguous(E) for l in layers}
    for l in layers:
        for e in hottest(C, l, a.hot):
            masks[l][e] = (1 << W) - 1
    costs = {l: [16] * E for l in layers}
    t0, w0, res = mdl.total({l: contiguous(E) for l in layers}, costs)
    tr, wr, res = mdl.total(masks, costs)
    moves = 0
    if a.method == "lpt":
        # balance each layer's EXPECTED rank load: p_e = P(expert e live in a step of R rows) x (B, + M if not resident);
        # repeatedly move the expert off the most loaded rank (to the least loaded) that most lowers the larger of the two,
        # within the extras budget; deterministic, so it does not chase the draws' noise
        for l in layers:
            m = masks[l]
            tot = float(sum(C[l])) or 1.0
            p = [1.0 - (1.0 - min(1.0, tk[l] * C[l][e] / tot)) ** a.rows for e in range(E)]
            def cost_of(e, r):
                return p[e] * (mdl.B + (0.0 if (l, e) in res[r] else mdl.M))
            while True:
                load = [0.0] * W
                for e in range(E):
                    if bin(m[e]).count("1") == 1:
                        r = (m[e] & -m[e]).bit_length() - 1
                        load[r] += cost_of(e, r)
                    else:   # a replicated expert: its load spread over its copies
                        rs = [r for r in range(W) if (m[e] >> r) & 1]
                        for r in rs:
                            load[r] += cost_of(e, r) / len(rs)
                src = max(range(W), key=lambda r: load[r]); dst = min(range(W), key=lambda r: load[r])
                gap = load[src] - load[dst]
                xs = extras_of([m], E)
                best = (0.0, None)
                for e in range(E):
                    if m[e] != 1 << src:
                        continue
                    if e // nh != dst and len(xs[dst]) >= a.budget:
                        continue
                    c = cost_of(e, src)
                    gain = min(c, gap - c)   # how much the larger of the two loads drops
                    if gain > best[0] + 1e-12:
                        best = (gain, e)
                if best[1] is None or best[0] < a.min_gain * (sum(load) / W):
                    break
                e = best[1]
                if (l, e) in res[src]:
                    res[src].discard((l, e)); res[dst].add((l, e))
                m[e] = 1 << dst
                moves += 1
        a.rounds = 0
    for rnd_ in range(a.rounds):
        for l in layers:
            m = masks[l]
            base, _ = mdl.layer_time(l, m, costs[l], res)
            while True:
                # the rank with the largest expected time: the live experts' sum of B (+ M if missed) a step
                load = [0.0] * W
                live = {}
                for s in mdl.D[l]:
                    for e, r in mdl.dispatch(m, costs[l], s).items():
                        load[r] += mdl.B + (0.0 if (l, e) in res[r] else mdl.M)
                        live[e] = live.get(e, 0) + 1
                src = max(range(W), key=lambda r: load[r])
                xs = extras_of([m], E)
                cand = [e for e in sorted(live, key=lambda e: -live[e])
                        if bin(m[e]).count("1") == 1 and (m[e] >> src) & 1][:a.cand]
                best = (base, None, None)
                for e in cand:
                    for dst in range(W):
                        if dst == src:
                            continue
                        if e // nh != dst and len(xs[dst]) >= a.budget:
                            continue
                        old = m[e]
                        m[e] = 1 << dst
                        # the copy keeps its tier: the heat engine follows the traffic to the new rank
                        was = (l, e) in res[src]
                        if was:
                            res[src].discard((l, e)); res[dst].add((l, e))
                        t, _ = mdl.layer_time(l, m, costs[l], res)
                        if was:
                            res[dst].discard((l, e)); res[src].add((l, e))
                        m[e] = old
                        if t < best[0] - 1e-9:
                            best = (t, e, dst)
                if best[1] is None:
                    break
                e, dst = best[1], best[2]
                if (l, e) in res[src]:
                    res[src].discard((l, e)); res[dst].add((l, e))
                m[e] = 1 << dst
                base = best[0]
                moves += 1
        _, _, res = mdl.total(masks, costs)   # residency follows the new traffic, then another pass
    # the dispatch's costs from each copy's expected miss rate
    T = mdl.traffic(masks, costs)
    for l in layers:
        cl = []
        for e in range(E):
            ranks = [r for r in range(W) if (masks[l][e] >> r) & 1]
            pm = sum(1.0 for r in ranks if (l, e) not in res[r]) / max(1, len(ranks))
            cl.append(max(1, int(round(16.0 * (mdl.B + mdl.M * pm) / mdl.B))))
        costs[l] = cl
    t1, w1, _ = mdl.total(masks, costs)
    xs = [extras_of([masks[l]], E) for l in layers]
    per_rank = [sum(len(x[r]) for x in xs) for r in range(W)]
    head = ("epplace.py solve --stats %s --rows %d --budget %d --hot %d --A %g --B %g --M %g --slots %d --draws %d: lever D3c, "
            "%d moves; model slowest-rank MoE %.2f -> %.2f ms a step (contiguous %.2f), AR wait %.2f -> %.2f (contiguous %.2f); "
            "extras a rank %s" % (a.stats, a.rows, a.budget, a.hot, a.A, a.B, a.M, a.slots, a.draws, moves, tr, t1, t0, wr, w1,
                                  w0, per_rank))
    write_place(a.o, masks, costs if a.costs else None, head=head)
    print("EPPLACE solve " + json.dumps({"rows": a.rows, "moves": moves, "contig_ms": round(t0, 3), "start_ms": round(tr, 3),
                                         "solved_ms": round(t1, 3), "contig_wait_ms": round(w0, 3),
                                         "solved_wait_ms": round(w1, 3), "extras_a_rank": per_rank, "out": a.o}))


def cmd_eval(a):
    C, tk, _ = read_stats(a.stats, a.kind)
    layers = [l for l in parse_layers(a.layers, len(C)) if sum(C[l]) > 0]
    E = len(C[layers[0]])
    for rows in [int(x) for x in str(a.rows).split(",")]:
        mdl = Model(C, tk, layers, rows, a.draws, a.A, a.B, a.M, a.slots, a.seed)
        tabs = [("contiguous", {l: contiguous(E) for l in layers}, {l: [16] * E for l in layers})]
        for p in a.place or []:
            mk, ck = read_place(p)
            tabs.append((p, {l: mk.get(l, contiguous(E)) for l in layers}, {l: ck.get(l, [16] * E) for l in layers}))
        for name, mk, ck in tabs:
            tm, tw, _ = mdl.total(mk, ck)
            fm, fw, _ = mdl.total(mk, ck, missfree=True)
            xs = [extras_of([mk[l]], E) for l in layers]
            print("EPEVAL rows %d %-40s slowest-rank MoE %.2f ms, AR wait %.2f ms (no misses: %.2f / %.2f); extras a rank %s" %
                  (rows, name[-40:], tm, tw, fm, fw, [sum(len(x[r]) for x in xs) for r in range(W)]))


def cmd_measure(a):
    """per (step, layer): the live experts a rank (vram + host + other) -> mean over steps and layers of max - mean,
    the missed ones a step, by rank."""
    rows = {}
    for ln in open(a.miss, errors="replace"):
        if ln.startswith("#"):
            continue
        f = ln.split()
        if len(f) != 16:
            continue
        step, kind, rank = int(f[0]), f[1], int(f[2])
        if step < a.from_step or (a.to_step is not None and step > a.to_step) or (a.kind and kind != a.kind):
            continue
        first, _, packed = f[15].partition(":")
        for i, x in enumerate(packed.split(",")):
            if x == "-":
                continue
            p = [int(v) for v in x.split(".")] + [0, 0]
            rows.setdefault((step, int(first) + i), {})[rank] = (p[0], p[1] + p[2])
    skew = live = miss = 0.0
    n = 0
    per_rank_miss = [0.0] * W
    for (step, l), d in rows.items():
        if len(d) < W:
            continue
        lv = [d[r][0] + d[r][1] for r in range(W)]
        skew += max(lv) - sum(lv) / W
        live += sum(lv)
        miss += sum(d[r][1] for r in range(W))
        for r in range(W):
            per_rank_miss[r] += d[r][1]
        n += 1
    steps = len({s for s, _ in rows})
    if not n:
        print("EPMEASURE " + json.dumps({"error": "no complete (step, layer) rows", "file": a.miss}))
        return
    print("EPMEASURE " + json.dumps({"file": a.miss, "kind": a.kind, "steps": steps, "layer_rows": n,
                                     "live_a_layer": round(live / n, 3), "skew_live_a_layer": round(skew / n, 3),
                                     "skew_live_a_step": round(skew / max(1, steps), 2),
                                     "missed_a_step": round(miss / max(1, steps), 2),
                                     "missed_a_step_by_rank": [round(x / max(1, steps), 2) for x in per_rank_miss]}))


def cmd_extras(a):
    tabs = [read_place(p)[0] for p in a.place]
    layers = sorted(set().union(*[set(t) for t in tabs])) if tabs else []
    tot = [0] * W
    for l in layers:
        xs = extras_of([contiguous(a.experts)] + [t.get(l, contiguous(a.experts)) for t in tabs], a.experts)
        for r in range(W):
            tot[r] += len(xs[r])
    print("EPEXTRAS " + json.dumps({"files": a.place, "layers": len(layers), "extras_a_rank": tot,
                                    "extras_max": max(tot) if tot else 0}))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("scramble"); p.add_argument("--experts", type=int, default=384); p.add_argument("--layers", default="0-39")
    p.add_argument("--per-rank", type=int, default=6); p.add_argument("--seed", type=int, default=1); p.add_argument("-o", required=True)
    for name in ("replicate", "solve", "eval"):
        q = sp.add_parser(name)
        q.add_argument("--stats", required=True); q.add_argument("--kind", default="decode"); q.add_argument("--layers", default="all")
        if name == "replicate":
            q.add_argument("--hot", type=int, default=32); q.add_argument("-o", required=True)
        else:
            q.add_argument("--rows", default="8"); q.add_argument("--draws", type=int, default=300)
            q.add_argument("--A", type=float, default=35.0); q.add_argument("--B", type=float, default=18.8)
            q.add_argument("--M", type=float, default=400.0); q.add_argument("--slots", type=int, default=1763)
            q.add_argument("--seed", type=int, default=1)
        if name == "solve":
            q.add_argument("--budget", type=int, default=12); q.add_argument("--hot", type=int, default=0)
            q.add_argument("--cand", type=int, default=12); q.add_argument("--rounds", type=int, default=2)
            q.add_argument("--costs", type=int, default=1); q.add_argument("-o", required=True)
            q.add_argument("--method", default="lpt", choices=["lpt", "greedy"])
            q.add_argument("--min-gain", type=float, default=0.002)
        if name == "eval":
            q.add_argument("--place", action="append")
    p = sp.add_parser("measure"); p.add_argument("--miss", required=True); p.add_argument("--from-step", type=int, default=0)
    p.add_argument("--to-step", type=int, default=None)
    p.add_argument("--kind", default="decode")
    p = sp.add_parser("extras"); p.add_argument("--place", action="append", default=[]); p.add_argument("--experts", type=int, default=384)
    a = ap.parse_args()
    if a.cmd == "solve":
        a.rows = int(a.rows)
    {"scramble": cmd_scramble, "replicate": cmd_replicate, "solve": cmd_solve, "eval": cmd_eval,
     "measure": cmd_measure, "extras": cmd_extras}[a.cmd](a)


if __name__ == "__main__":
    main()
