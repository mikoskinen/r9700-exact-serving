#!/usr/bin/env python3
"""fn-bitrace.py (11.11.2026, fn4 BI; = the DS agent's queue-ar/dsmain/bitrace.py, its record format, adapted) -- reads fn4 BI's
Flash-Next engine tracer (RADIANCE_FN4_BITRACE=<dir>: <dir>/bt-r<rank>-<pid>.bin + .names, written by qwen4exp's bt_flush; the
sequence ids in it are request serials, not slots) and names, for each greedy probe and each LOADED run of it, the FIRST op whose
output row differs from the probe's first ALONE run (the last occurrence is the control: alone again).
fn4 changes: any number of loaded runs between the first and the last occurrence (--labels names them, in run order);
hashes kept as tuples with one name index a names key (memory); a summary per loaded run (identical count, first-divergence
point class and layer, step class).

  fn-bitrace.py analyze DIR [--json OUT.json] [--min-prompt 4] [--labels c4,pf]   (the DS text below: its single-ld form)
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


_NAMES = {}


def names_of(nm, NP):
    """one shared names list (and its index) a names key"""
    k = (id(nm), NP)
    if k not in _NAMES:
        lst = [x[0] for x in nm] if nm and len(nm) == NP else ["p%d" % i for i in range(NP)]
        _NAMES[k] = (lst, {n: i for i, n in enumerate(lst)})
    return _NAMES[k]


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
        lst, idx = names_of(names.get(key), NP)
        yield dict(step=step, rank=rank, T=T, S=S, NO=NO, DT=DT, SD=SD, nspec=nspec, cu=cu, sid=sid, ql=ql, cl=cl,
                   rows=rows, names=lst, idx=idx)


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
                    per[st["rank"]][sid][pos] = (st["step"], tok, (h, st["idx"]), shape, st["names"])
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


SKIP = set()     # 13.11.: --skip NAMES -- point base names (no "Lnn.") left out of the walk (is a later point clean?)
CENSUS = None    # 13.11.: --census -- {base name: probes-loaded pairs with a difference there at ANY compared position}


def compare(per, a, b):
    """first divergence of sequence b against a: dict or None (identical over the common positions)"""
    best = None
    seen = set()
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
            ha, ia = ra[2]
            hb = rb[2][0]
            for i, nm in enumerate(rb[4]):
                base = re.sub(r"^L\d+\.", "", nm)
                if base in SKIP:
                    continue
                j = ia.get(nm)
                if j is None:
                    continue
                if ha[j] != hb[i]:
                    seen.add(base if CENSUS is not None else nm)
                    cand = (pos, i, rk, nm, ra[3], rb[3])
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                        best = cand
                    if CENSUS is None:
                        break
        if best is not None and best[0] == pos and CENSUS is None:
            break
    if CENSUS is not None:
        for nm in seen: CENSUS[nm] = CENSUS.get(nm, 0) + 1
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
    labels = [x for x in (a.labels or "").split(",") if x]
    nld = collections.Counter(len(v) - 2 for v in probes)
    res = []
    cnt = collections.defaultdict(lambda: dict(n=0, ident=0, op=collections.Counter(), layer=collections.Counter(),
                                               cls=collections.Counter(), ctrl_bad=0))
    gnames = [x for x in (a.groups or "").split(",") if x]
    for k, seqs in enumerate(probes):
      # --group N: a probe's occurrences in arms of N runs each (in run order), each arm its own a1 / loaded / control
      chunks = [seqs] if a.group <= 0 else [seqs[i:i + a.group] for i in range(0, len(seqs), a.group)]
      for gi, chunk in enumerate(chunks):
        if len(chunk) < 3:
            continue
        gpre = "" if a.group <= 0 else (gnames[gi] if gi < len(gnames) else "g%d" % gi) + "/"
        a1, a2, lds = chunk[0], chunk[-1], chunk[1:-1]
        c = compare(per, a1, a2)
        e = dict(probe=k, arm=gpre[:-1], seqs=chunk, loaded={}, a2_vs_a1=c)
        for j, ld in enumerate(lds):
            lab = gpre + (labels[j] if j < len(labels) else "ld%d" % (j + 1))
            d = compare(per, a1, ld)
            e["loaded"][lab] = d
            s = cnt[lab]
            s["n"] += 1
            if c is not None:
                s["ctrl_bad"] += 1
            if d is None:
                s["ident"] += 1
                line = "IDENTICAL over every traced row and point"
            elif "point" in d:
                line = "FIRST DIVERGENCE pos %d point %s (rank %d)%s | loaded step %s | alone step %s" % (
                    d["pos"], d["point"], d["rank"], "" if d["token_diff_pos"] is None else ", tokens differ from pos %d" % d["token_diff_pos"],
                    shape_s(d["loaded"]), shape_s(d["alone"]))
                s["op"][d["op"]] += 1
                mm = re.match(r"^L(\d+)\.", d["point"])
                s["layer"]["L%s" % mm.group(1) if mm else d["point"]] += 1
                s["cls"]["%s row, %s" % (d["loaded"]["cls"], "mixed step" if d["loaded"]["npf"] and d["loaded"]["DT"] else
                                         "prefill step" if d["loaded"]["npf"] else "decode step")] += 1
            else:
                line = "tokens differ from pos %d with no differing traced point before it (an untraced op or cache entry)" % d["token_diff_pos"]
                s["op"]["(untraced)"] += 1
            print("BITRACE p%d %s seqs %s: %s" % (k, lab, [a1, ld], line))
        print("BITRACE p%d %scontrol a2 vs a1 (seqs %s): %s" % (k, gpre, [a1, a2], "IDENTICAL" if c is None else "DIFFERS (pos %s, %s)" % (
            c.get("pos", c["token_diff_pos"]), c.get("point", "token"))))
        res.append(e)
    for lab, s in cnt.items():
        print("BITRACE summary %s: %d probes, %d identical vs alone (control differs in %d); first-divergence point: %s; layer: %s; "
              "step class: %s" % (lab, s["n"], s["ident"], s["ctrl_bad"], dict(s["op"].most_common()),
                                  dict(s["layer"].most_common(8)), dict(s["cls"].most_common())))
    print("BITRACE summary: %d probes (occurrences: %s)" % (len(res), dict(nld)))
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
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze(argparse.Namespace(dir=d, json=None, min_prompt=4, labels="ld", group=0, groups=""))
    out = buf.getvalue()
    shutil.rmtree(d, ignore_errors=True)
    an = ("BITRACE summary ld: 1 probes, 0 identical vs alone (control differs in 0); first-divergence point: {'q': 1}" in out
          and "point L00.q (rank 1)" in out)
    ok = seqs == [(7, 0), (7, 1), (7, 2)] and r and r.get("pos") == 5 and r.get("point") == "L00.q" and r.get("rank") == 1 and c is None and an
    if not an:
        print(out)
    print("SELFTEST %s (probe seqs %s, divergence %s, control %s)" % ("OK" if ok else "FAILED", seqs,
          {k: r.get(k) for k in ("pos", "point", "rank")} if r else None, c))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "selftest"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--min-prompt", type=int, default=4)
    ap.add_argument("--labels", default="", help="the loaded runs' names in run order (comma list)")
    ap.add_argument("--group", type=int, default=0, help="occurrences an arm (one process, arms switched in process)")
    ap.add_argument("--groups", default="", help="the arms' names in run order (comma list), with --group")
    ap.add_argument("--skip", default="", help="point base names (comma list) left out of the walk")
    ap.add_argument("--census", action="store_true", help="also count, per point base name, the compared pairs differing there anywhere")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if not a.dir:
        ap.error("analyze needs DIR")
    global SKIP, CENSUS
    SKIP = set(x for x in (a.skip or "").split(",") if x)
    if a.census: CENSUS = {}
    rc = analyze(a)
    if CENSUS is not None:
        print("BITRACE census (compared pairs with a difference at the point, any position): %s" % (
            ", ".join("%s %d" % (k, v) for k, v in sorted(CENSUS.items(), key=lambda t: -t[1])) or "none"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
