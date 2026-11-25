#!/usr/bin/env python3
"""gapcat.py STAMPS.csv [--from NS --to NS] [--label L] [--json OUT] [--top N] (9.11.2026, queue-ar, item 115) -- where a
card idles inside and between recorded passes, from RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,
start_ns,end_ns,play,records; one line a dispatch, host CLOCK_MONOTONIC ns). Streams the file pass by pass.

Inside a pass (dispatches in start order over the union of their intervals), every idle stretch is a gap, charged to the
dispatch that STARTS after it (`next`) and the one whose end it follows (`prev`, the last to finish):
  stretch    next.play != prev.play: the host's work between two played stretches (host ops, table refreshes, routing
             reports, KV staging) -- Ctx::play_pass
  queue      next on another queue than prev: a cross-queue dependency (barrier-AND, the collective queue, a lane)
  chained    next is a chained form (waits in its prologue)
  barrier    next carries the barrier bit: it starts only after prev finished and its caches drained -- launch latency
             of a dependent kernel
  unordered  next has no barrier bit, yet the queue idled: nothing was submitted (host behind) or an implicit dependency
Between passes: the idle time from one pass's last end to the next pass's first start on this card, by the two passes'
kinds (records counts; the longest kind is the trunk's verify pass). Per kind: passes, median wall / busy / gaps, dispatch
counts (ordered, chained); for the trunk kind the categories ranked by us a pass, the gap-length histogram and the kernel
pairs that cost the most. Standard library only."""
import csv, json, os, statistics, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stampcat import classify  # noqa: E402

BINS = [(0, 2e3, "<2us"), (2e3, 5e3, "2-5us"), (5e3, 10e3, "5-10us"), (10e3, 50e3, "10-50us"), (50e3, 1e18, ">50us")]


def short(k):
    """A kernel's own name from its Itanium mangling: the last component of a nested name, the anonymous
    namespace (_GLOBAL__N_1) skipped; plain names as they are."""
    k = k or "?"
    if k.startswith("_Z"):
        i = 2
        nested = i < len(k) and k[i] == "N"
        if nested:
            i += 1
            while i < len(k) and k[i] in "rVKL":
                i += 1
        last = None
        while i < len(k) and k[i].isdigit():
            j = i
            while j < len(k) and k[j].isdigit():
                j += 1
            n = int(k[i:j]); comp = k[j:j + n]; i = j + n
            if comp != "_GLOBAL__N_1":
                last = comp
            if not nested:
                break
        if last:
            k = last
    return k[:40]


def analyse(rows):
    """rows: (seq, start, end, kernel, queue, ordered, chained, play) of one pass. -> dict."""
    rows.sort(key=lambda r: (r[1], r[0]))
    t0 = rows[0][1]
    cov, last = rows[0][2], rows[0]
    busy = rows[0][2] - rows[0][1]
    cats, pairs, hist = Counter(), Counter(), Counter()
    ncat = Counter()
    for r in rows[1:]:
        s, e = r[1], r[2]
        if s > cov:
            g = s - cov
            if r[7] != last[7]: c = "stretch"
            elif r[4] != last[4]: c = "queue"
            elif r[6]: c = "chained"
            elif r[5]: c = "barrier"
            else: c = "unordered"
            cats[c] += g; ncat[c] += 1
            pairs[f"{short(last[3])} > {short(r[3])}"] += g
            for lo, hi, nm in BINS:
                if lo <= g < hi:
                    hist[nm] += g; break
            busy += e - s
        elif e > cov:
            busy += e - cov
        if e >= cov:
            cov, last = e, r
    wall = cov - t0
    return {"t0": t0, "t1": cov, "wall": wall, "busy": busy, "gaps": wall - busy, "cats": cats, "ncat": ncat,
            "pairs": pairs, "hist": hist, "n": len(rows), "ordered": sum(1 for r in rows if r[5]),
            "chained": sum(1 for r in rows if r[6])}


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--label", "--json", "--top"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    path = a[0]
    tf = int(opt.get("--from", 0)); tt = int(opt.get("--to", 1 << 62)); top = int(opt.get("--top", 8))
    label = opt.get("--label", path)
    per_kind = defaultdict(list)          # records -> [analysis]
    seqp = []                             # every pass in file order: (t0, t1, kind, n)
    cur, cur_key = [], None

    def flush():
        if not cur:
            return
        r = analyse(cur)
        kind = cur_key[1]
        seqp.append((r["t0"], r["t1"], kind, r["n"]))
        if tf <= r["t0"] <= tt:
            per_kind[kind].append(r)

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["pass"]), int(row["records"]))
                s, e = int(row["start_ns"]), int(row["end_ns"])
            except (TypeError, ValueError, KeyError):
                continue
            if s <= 0 or e < s:
                continue
            if key != cur_key:
                flush(); cur.clear(); cur_key = key
            cur.append((int(row["seq"]), s, e, row["kernel"], row["queue"], row["ordered"] == "1",
                        row["chained"] == "1", row["play"]))
    flush()
    if not per_kind:
        print(f"{label}: no stamped pass in the window"); return
    # a pass stamped in part (the signal pool ran short) is left out: only passes with every dispatch of their kind
    full = {k: max(x["n"] for x in v) for k, v in per_kind.items()}
    part = {k: sum(1 for x in v if x["n"] < full[k]) for k, v in per_kind.items()}
    per_kind = {k: [x for x in v if x["n"] == full[k]] for k, v in per_kind.items()}
    inter = defaultdict(list)             # (kind_prev, kind_next) -> [gap ns], complete passes only
    for (a0, a1, ka, na), (b0, b1, kb, nb) in zip(seqp, seqp[1:]):
        if tf <= b0 <= tt and na == full.get(ka) and nb == full.get(kb) and a1 < b0 and b0 - a1 < 50e6:
            inter[(ka, kb)].append(b0 - a1)
    if any(part.values()):
        print(f"{label}: passes stamped in part, left out: " + ", ".join(f"kind {k} {n}" for k, n in part.items() if n))
    trunk = max(per_kind)
    names = {trunk: "trunk"}
    out = {"label": label, "kinds": {}}
    print(f"{label}: pass kinds by records (dispatches): " + ", ".join(
        f"{k}{' (trunk)' if k == trunk else ''}: {len(v)} passes" for k, v in sorted(per_kind.items(), reverse=True)))
    for k, v in sorted(per_kind.items(), reverse=True):
        med = lambda key: statistics.median(x[key] for x in v) / 1e3
        d = {"passes": len(v), "wall_us": med("wall"), "busy_us": med("busy"), "gaps_us": med("gaps"),
             "dispatches": statistics.median(x["n"] for x in v), "ordered": statistics.median(x["ordered"] for x in v),
             "chained": statistics.median(x["chained"] for x in v)}
        out["kinds"][k] = d
        print(f"  kind {k}{' trunk' if k == trunk else ''}: {len(v)} passes, wall {d['wall_us']:.0f} busy {d['busy_us']:.0f} "
              f"gaps {d['gaps_us']:.0f} us ({100 * d['gaps_us'] / max(1e-9, d['wall_us']):.1f} %), dispatches "
              f"{d['dispatches']:.0f} (barrier bit {d['ordered']:.0f}, chained {d['chained']:.0f})")
    v = per_kind[trunk]
    P = len(v)
    cats, ncat, pairs, hist = Counter(), Counter(), Counter(), Counter()
    for x in v:
        cats.update(x["cats"]); ncat.update(x["ncat"]); pairs.update(x["pairs"]); hist.update(x["hist"])
    tot = sum(cats.values()) / P / 1e3
    print(f"  trunk gap sources, us a pass (of {tot:.0f}): " + "; ".join(
        f"{c} {cats[c] / P / 1e3:.0f} ({ncat[c] / P:.0f} gaps, mean {cats[c] / max(1, ncat[c]) / 1e3:.1f} us)"
        for c, _ in cats.most_common()))
    print("  trunk gap lengths, us a pass: " + "; ".join(f"{nm} {hist[nm] / P / 1e3:.0f}" for _, _, nm in BINS))
    print(f"  trunk costliest kernel pairs (prev > next), us a pass: " + "; ".join(
        f"{p} {g / P / 1e3:.0f}" for p, g in pairs.most_common(top)))
    out["trunk"] = {"cats_us": {c: cats[c] / P / 1e3 for c in cats}, "gaps_n": {c: ncat[c] / P for c in ncat},
                    "hist_us": {nm: hist[nm] / P / 1e3 for nm in hist},
                    "pairs_us": {p: g / P / 1e3 for p, g in pairs.most_common(30)}}
    if inter:
        print("  between passes (this card idle from one pass's end to the next's start), median us: " + "; ".join(
            f"{a}>{b} {statistics.median(g) / 1e3:.0f} (n {len(g)})" for (a, b), g in sorted(inter.items(), key=lambda kv: -len(kv[1]))[:8]))
        out["between_us"] = {f"{a}>{b}": statistics.median(g) / 1e3 for (a, b), g in inter.items()}
    if "--json" in opt:
        json.dump(out, open(opt["--json"], "w"))


if __name__ == "__main__":
    main()
