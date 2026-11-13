#!/usr/bin/env python3
"""bitrace.py (11.11.2026, queue-ar/dsmain, ds41-bi, lever BI) -- reads lever BI's engine tracer (RADIANCE_DS41_BITRACE=<dir>:
<dir>/bt-r<rank>-<pid>.bin + .names, written by deepseek41's bt_flush) and names, for each greedy probe, the FIRST op whose
output row differs between the probe run ALONE and the same probe run UNDER LOAD (and the control: alone again after).

  bitrace.py analyze DIR [--json OUT.json] [--min-prompt 4]
      Every request is a sequence (its engine seq id). A probe is a prompt that occurs in three or more sequences (alone,
      under load, alone again: the run order); the c4 streams' and the long prefills' prompts occur once. For each probe its
      occurrences are ordered by first step: a1, ld, a2 (more: the extra ones are listed). Per (sequence, position) the LAST
      row computed at that position is kept (a rejected verify row is recomputed later). ld vs a1 (and a2 vs a1): positions
      ascending; the first position whose token differs ends the walk (the row before it holds the cause); at each position
      with equal tokens the points are compared in issue order (h0, xn, qr, q, kvn, iq, iw, sel, o, oa, attn_out, h1, ffn_in,
      router, ids, ew, sout, moe_out a layer, then final, logits), per rank; the first differing (position, point, rank) is the
      first diverging op. Its step's shape is printed beside it (rows T, decode rows DT, sequences S, prefill sequences, the
      row's class decode / prefill and its sequence's q_len), alone and loaded.
      Prints one "BITRACE p<k> ..." line a probe and "BITRACE summary: ..." (first-divergence counts by op and by step class).
  bitrace.py selftest
      writes a synthetic trace (two ranks, three runs of one probe, one point differing at a known place) to a temp dir and
      checks the analyzer finds it. No server.
Standard library only.
"""
import argparse, collections, glob, json, os, re, shutil, struct, sys, tempfile

HDR = struct.Struct("<4siq8iQ")   # magic, rank, step, T, S, NO, NP, nsel, DT, SD, n_spec, names key


def read_names(path):
    """names key -> [(name, kind)]"""
    out, key, want = {}, None, 0
    try:
        f = open(path)
    except OSError:
        return out
    for line in f:
        p = line.split()
        if not p:
            continue
        if p[0] == "N" and len(p) == 3:
            key, want = int(p[1], 16), int(p[2])
            out[key] = []
        elif key is not None and len(out[key]) < want:
            out[key].append((p[0], int(p[1]) if len(p) > 1 else 0))
    return out


def read_bin(path, names):
    """yields one dict a step: step, rank, T, S, NO, DT, SD, sid, ql, cl, rows [(r, pos, tok, sq, hashes)] and names"""
    data = open(path, "rb").read()
    o, n = 0, len(data)
    while o + HDR.size <= n:
        magic, rank, step, T, S, NO, NP, nsel, DT, SD, nspec, key = HDR.unpack_from(data, o)
        if magic != b"BIT2":
            raise ValueError("%s: bad record at %d" % (path, o))
        o += HDR.size
        def arr(k):
            nonlocal o
            v = struct.unpack_from("<%di" % k, data, o) if k > 0 else ()
            o += 4 * k
            return v
        cu, sid, ql, cl, oid = arr(S + 1), arr(S), arr(S), arr(S), arr(NO)
        rows = []
        for _ in range(nsel):
            r, pos, tok, sq = struct.unpack_from("<4i", data, o)
            o += 16
            h = struct.unpack_from("<%dQ" % NP, data, o) if NP else ()
            o += 8 * NP
            rows.append((r, pos, tok, sq, h))
        nm = names.get(key)
        yield dict(step=step, rank=rank, T=T, S=S, NO=NO, DT=DT, SD=SD, nspec=nspec, cu=cu, sid=sid, ql=ql, cl=cl,
                   rows=rows, names=[x[0] for x in nm] if nm and len(nm) == NP else ["p%d" % i for i in range(NP)])


def load(d):
    """rank -> request instance -> {pos: (step, tok, hashes dict, shape)}; instance -> first step.
    The engine's seq_ids are SLOTS, reused by later requests (trace3: 6 slots over a whole start), so a request instance is
    (slot, n): a slot's rows from a step where it computes position 0 (a new prompt's first chunk) to the next such step.
    Every rank sees the same steps and rows, so the numbering agrees across ranks."""
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    first = {}
    for b in sorted(glob.glob(os.path.join(d, "bt-r*-*.bin"))):
        names = read_names(b[:-4] + ".names")
        inst = {}   # slot -> its current instance number in this rank's file
        for st in read_bin(b, names):
            S, SD = st["S"], st["SD"]
            npf = S - SD
            starts = {st["sid"][sq] for (r, pos, tok, sq, h) in st["rows"] if pos == 0 and 0 <= sq < S}
            for slot in starts:
                inst[slot] = inst.get(slot, -1) + 1
            for (r, pos, tok, sq, h) in st["rows"]:
                if sq < 0 or sq >= S:
                    continue
                slot = st["sid"][sq]
                sid = (slot, inst.get(slot, 0))
                shape = dict(T=st["T"], DT=st["DT"], S=S, npf=npf, cls="decode" if sq < SD else "prefill",
                             qlen=st["ql"][sq], step=st["step"], nspec=st["nspec"])
                prev = per[st["rank"]][sid].get(pos)
                if prev is None or st["step"] >= prev[0]:
                    per[st["rank"]][sid][pos] = (st["step"], tok, dict(zip(st["names"], h)), shape, st["names"])
                first[sid] = min(first.get(sid, st["step"]), st["step"])
    return per, first


def prompts(per):
    """seq id -> its prompt key: the tokens at positions 0.. of the rows computed in its first step(s) with q_len > 1"""
    r0 = min(per) if per else None
    out = {}
    if r0 is None:
        return out
    for sid, rows in per[r0].items():
        if 0 not in rows:
            continue
        toks, p = [], 0
        while p in rows and rows[p][3]["cls"] == "prefill" and len(toks) < 64:
            toks.append(rows[p][1])
            p += 1
        if toks:
            out[sid] = tuple(toks)
    return out


def compare(per, a, b):
    """first divergence of sequence b against a: dict or None (identical over the common positions)"""
    best = None
    tokdiff = None
    ranks = sorted(per)
    r0 = ranks[0]
    common = sorted(set(per[r0][a]) & set(per[r0][b]))
    for pos in common:
        ta, tb = per[r0][a][pos][1], per[r0][b][pos][1]
        if ta != tb:
            tokdiff = pos
            break
        for rk in ranks:
            ra, rb = per[rk][a].get(pos), per[rk][b].get(pos)
            if ra is None or rb is None:
                continue
            for i, nm in enumerate(rb[4]):
                if nm not in ra[2]:
                    continue
                if ra[2][nm] != rb[2][nm]:
                    cand = (pos, i, rk, nm, ra[3], rb[3])
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                        best = cand
                    break
        if best is not None and best[0] == pos:
            break
    if best is None and tokdiff is None:
        return None
    out = {"token_diff_pos": tokdiff, "positions_compared": len(common)}
    if best is not None:
        pos, i, rk, nm, sa, sb = best
        out.update(pos=pos, point=nm, op=re.sub(r"^L\d+\.", "", nm), point_index=i, rank=rk, alone=sa, loaded=sb)
    return out


def shape_s(s):
    return "T %d (decode rows %d, %d seqs, %d prefill seqs; row %s, its q_len %d%s)" % (
        s["T"], s["DT"], s["S"], s["npf"], s["cls"], s["qlen"], ", n_spec %d" % s["nspec"] if s["nspec"] else "")


def analyze(a):
    per, first = load(a.dir)
    if not per:
        print("BITRACE: no trace records in %s" % a.dir)
        return 1
    pk = prompts(per)
    groups = collections.defaultdict(list)
    for sid, key in pk.items():
        if len(key) >= a.min_prompt:
            groups[key].append(sid)
    probes = [sorted(v, key=lambda s: first[s]) for v in groups.values() if len(v) >= 3]
    probes.sort(key=lambda v: first[v[0]])
    res, cnt_op, cnt_cls = [], collections.Counter(), collections.Counter()
    for k, seqs in enumerate(probes):
        a1, ld, a2 = seqs[0], seqs[1], seqs[2]
        d = compare(per, a1, ld)
        c = compare(per, a1, a2)
        e = dict(probe=k, seqs=seqs, prompt_len=len(next(key for key, v in groups.items() if v and sorted(v, key=lambda s: first[s]) == seqs)),
                 ld_vs_a1=d, a2_vs_a1=c)
        res.append(e)
        if d is None:
            line = "IDENTICAL over every traced row and point"
        elif "point" in d:
            line = "FIRST DIVERGENCE pos %d point %s (rank %d)%s | loaded step %s | alone step %s" % (
                d["pos"], d["point"], d["rank"], "" if d["token_diff_pos"] is None else ", tokens differ from pos %d" % d["token_diff_pos"],
                shape_s(d["loaded"]), shape_s(d["alone"]))
            cnt_op[d["op"]] += 1
            cnt_cls["%s row, %s" % (d["loaded"]["cls"], "mixed step" if d["loaded"]["npf"] and d["loaded"]["DT"] else
                                    "prefill step" if d["loaded"]["npf"] else "decode step")] += 1
        else:
            line = "tokens differ from pos %d with no differing traced point before it (an untraced op or cache entry)" % d["token_diff_pos"]
            cnt_op["(untraced)"] += 1
        print("BITRACE p%d seqs %s: %s | control a2 vs a1: %s" % (
            k, seqs[:3], line, "IDENTICAL" if c is None else "DIFFERS (pos %s, %s)" % (c.get("pos", c["token_diff_pos"]), c.get("point", "token"))))
    print("BITRACE summary: %d probes (%d identical under load); first-divergence op: %s; step class: %s" % (
        len(res), sum(1 for e in res if e["ld_vs_a1"] is None), dict(cnt_op.most_common()), dict(cnt_cls.most_common())))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    return 0


def selftest(_a):
    d = tempfile.mkdtemp(prefix="bitrace-")
    names = ["L00.h0", "L00.xn", "L00.q", "final", "logits"]
    kinds = [0, 0, 0, 0, 1]
    key = 0x1234
    for rank in (0, 1):
        with open(os.path.join(d, "bt-r%d-1.names" % rank), "w") as f:
            f.write("N %016x %d\n" % (key, len(names)))
            for nm, kd in zip(names, kinds):
                f.write("%s %d\n" % (nm, kd))
        with open(os.path.join(d, "bt-r%d-1.bin" % rank), "wb") as f:
            step = 0
            prompt = [5, 6, 7, 8]
            for run in range(3):
                sid = 7   # one slot reused by every run (the engine's seq_ids are slots)
                # prefill step: the prompt, alone (run 0, 2) or beside another sequence's prefill (run 1)
                rows = [(p, prompt[p]) for p in range(4)]
                other = run == 1
                S = 2 if other else 1
                cu = [0, 4, 8] if other else [0, 4]
                sids = [sid, 999] if other else [sid]
                recs = []
                for r, (p, t) in enumerate(rows):
                    h = [hash((p, t, nm)) & 0xffffffffffffffff for nm in names]
                    recs.append((r, p, t, 0, h))
                if other:
                    for r in range(4):
                        recs.append((4 + r, r, 50 + r, 1, [7] * len(names)))
                f.write(HDR.pack(b"BIT2", rank, step, cu[-1], S, 1, len(names), len(recs), 0, 0, 0, key))
                f.write(struct.pack("<%di" % (S + 1), *cu))
                f.write(struct.pack("<%di" % S, *sids))
                f.write(struct.pack("<%di" % S, *([4] * S)))
                f.write(struct.pack("<%di" % S, *([0] * S)))
                f.write(struct.pack("<1i", 3))
                for (r, p, t, sq, h) in recs:
                    f.write(struct.pack("<4i", r, p, t, sq))
                    f.write(struct.pack("<%dQ" % len(h), *h))
                step += 1
                # decode steps: positions 4..6, run 1 differs at position 5's L00.q on rank 1
                for p in range(4, 7):
                    t = 20 + p
                    h = [hash((p, t, nm)) & 0xffffffffffffffff for nm in names]
                    if run == 1 and p >= 5 and rank == 1:
                        h[2] ^= 1
                    f.write(HDR.pack(b"BIT2", rank, step, 1, 1, 1, len(names), 1, 1, 1, 0, key))
                    f.write(struct.pack("<2i", 0, 1))
                    f.write(struct.pack("<1i", sid))
                    f.write(struct.pack("<1i", 1))
                    f.write(struct.pack("<1i", p))
                    f.write(struct.pack("<1i", 0))
                    f.write(struct.pack("<4i", 0, p, t, 0))
                    f.write(struct.pack("<%dQ" % len(h), *h))
                    step += 1
    per, first = load(d)
    pk = prompts(per)
    seqs = sorted([s for s, k in pk.items() if k == (5, 6, 7, 8)], key=lambda s: first[s])
    r = compare(per, seqs[0], seqs[1])
    c = compare(per, seqs[0], seqs[2])
    shutil.rmtree(d, ignore_errors=True)
    ok = seqs == [(7, 0), (7, 1), (7, 2)] and r and r.get("pos") == 5 and r.get("point") == "L00.q" and r.get("rank") == 1 and c is None
    print("SELFTEST %s (probe seqs %s, divergence %s, control %s)" % ("OK" if ok else "FAILED", seqs,
          {k: r.get(k) for k in ("pos", "point", "rank")} if r else None, c))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "selftest"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--min-prompt", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if not a.dir:
        ap.error("analyze needs DIR")
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
