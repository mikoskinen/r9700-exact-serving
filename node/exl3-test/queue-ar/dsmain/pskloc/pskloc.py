#!/usr/bin/env python3
"""pskloc.py (11.11.2026, queue-ar/dsmain/pskloc, PSK cone-exactness localiser) -- reads lever BI's engine tracer
(RADIANCE_DS41_BITRACE=<dir>: <dir>/bt-r<rank>-<pid>.bin + .names, deepseek41's bt_flush; format = bitrace2.py's) for ONE
start in which the same long prompt was sent cold once per arm (prefix reset between), and names, for every arm against the
reference arm (the first A), the FIRST (chunk, layer, op) whose output differs on a row the arm's cut must leave exact.

  pskloc.py analyze DIR --order AB [--cut B=21:160:128,C=25:160:128] [--pairs B:C] [--min-prompt 1000] [--json OUT]
      Request instances are found as bitrace2.py does (slot, n); the instances with a prompt of >= --min-prompt positions
      are the arms' runs, in --order (first step order). The reference is the first A. --cut gives an arm's cone: first cut
      layer : base rows : rows a layer (C_l = base + (n_layer - 1 - l) * step, layer l >= first, a prefill chunk of more
      than C_l rows cut to its last C_l). Arms not named in --cut are uncut (every row trusted). TRUSTED ROWS of a point at a
      cut layer l, chunk end e: the pre-attention points (h0 xn qr q kvn iq iw sel) at rows >= e - C_l; every other point
      (o oa attn_out h1 ffn_in router ids ew sout moe_out) at rows >= e - C_l + (window - 1) -- the rows whose window keys at
      l were all written by the cone; `final` as layer n_layer - 1's post points; `logits` at the chunk's output row. A row a
      cut layer leaves stale is never compared.
      Prints per arm: "PSKLOC <arm> chunks ..." (the chunking, vs the reference), "PSKLOC <arm> FIRST ..." (the first
      differing trusted point: chunk, rows, layer.op, ranks, rows differing / trusted, their offsets from the chunk end), "PSKLOC
      <arm> layers ..." (the first chunk with a difference: per layer the first differing op and its row count), "PSKLOC <arm>
      window ..." (kvn of the last `window` rows of every chunk at layers >= 21), "PSKLOC <arm> taps ..." (h0 of the last
      window + 1 rows at layers 37-39), "PSKLOC <arm> logits ...", "PSKLOC <arm> decode ..." (the first decode position whose
      token differs), and one "PSKLOC summary" line.
  pskloc.py analyze DIR --labels "A B XB16000 LA LB" --pairs B:A,LB:LA --cut B=21:160:128,LB=21:160:128 [--min-prompt N]
      The long prompts sharing the most common first 32 tokens (the deepq prompts) are the labelled runs in step order (the
      last len(labels) of them); each pair X:Y compares run X against run Y with X's cut, as above.
  pskloc.py selftest
      a synthetic trace (two ranks, arms A and B, one cut layer, one stale row and one real difference) -> the analyzer must
      report the real one and ignore the stale one. No server.
Standard library only."""
import argparse, collections, glob, json, os, re, shutil, struct, sys, tempfile

HDR = struct.Struct("<4siq8iQ")   # magic, rank, step, T, S, NO, NP, nsel, DT, SD, n_spec, names key
PRE = {"h0", "xn", "qr", "q", "kvn", "iq", "iw", "sel"}
NL, WIN = 40, 128


def read_names(path):
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


def load(d):
    """rank -> inst -> pos -> (step, tok, cls, blob(memoryview of NP u64), names tuple, T, S, qlen); inst -> first step"""
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    first = {}
    keep = []   # the files' bytes (blobs are views into them)
    for b in sorted(glob.glob(os.path.join(d, "bt-r*-*.bin"))):
        names = read_names(b[:-4] + ".names")
        data = open(b, "rb").read()
        keep.append(data)
        mv = memoryview(data)
        o, n = 0, len(data)
        inst = {}
        ntup = {}
        while o + HDR.size <= n:
            magic, rank, step, T, S, NO, NP, nsel, DT, SD, nspec, key = HDR.unpack_from(data, o)
            if magic != b"BIT2":
                raise ValueError("%s: bad record at %d" % (b, o))
            o += HDR.size
            def arr(k):
                nonlocal o
                v = struct.unpack_from("<%di" % k, data, o) if k > 0 else ()
                o += 4 * k
                return v
            cu, sid, ql, cl, oid = arr(S + 1), arr(S), arr(S), arr(S), arr(NO)
            if key not in ntup:
                nm = names.get(key)
                ntup[key] = tuple(x[0] for x in nm) if nm and len(nm) == NP else tuple("p%d" % i for i in range(NP))
            nt = ntup[key]
            rows = []
            for _ in range(nsel):
                r, pos, tok, sq = struct.unpack_from("<4i", data, o)
                o += 16
                rows.append((r, pos, tok, sq, mv[o:o + 8 * NP]))
                o += 8 * NP
            starts = {sid[sq] for (r, pos, tok, sq, h) in rows if pos == 0 and 0 <= sq < S}
            for slot in starts:
                inst[slot] = inst.get(slot, -1) + 1
            for (r, pos, tok, sq, h) in rows:
                if sq < 0 or sq >= S:
                    continue
                slot = sid[sq]
                iid = (slot, inst.get(slot, 0))
                prev = per[rank][iid].get(pos)
                if prev is None or step >= prev[0]:
                    per[rank][iid][pos] = (step, tok, "decode" if sq < SD else "prefill", h, nt, T, S, ql[sq])
                first[iid] = min(first.get(iid, step), step)
    return per, first, keep


def hashes(blob):
    return struct.unpack("<%dQ" % (len(blob) // 8), blob)


def prompt_len(rows):
    p = 0
    while p in rows and rows[p][2] == "prefill":
        p += 1
    return p


def chunks(rows, plen):
    """[(start, end, step, T)] of the prefill steps, by step"""
    by = collections.OrderedDict()
    for p in range(plen):
        st, T = rows[p][0], rows[p][5]
        if st not in by:
            by[st] = [p, p + 1, T]
        else:
            by[st][0] = min(by[st][0], p)
            by[st][1] = max(by[st][1], p + 1)
    return [(v[0], v[1], st, v[2]) for st, v in sorted(by.items())]


def parse_cut(s):
    out = {}
    for e in (s or "").split(","):
        e = e.strip()
        if not e:
            continue
        a, v = e.split("=")
        f, b, w = (int(x) for x in v.split(":"))
        out[a] = (f, b, w)
    return out


def layer_of(name):
    m = re.match(r"^L(\d+)\.(.+)$", name)
    if m:
        return int(m.group(1)), m.group(2)
    return (NL - 1 if name == "final" else NL), name


def trusted_lo(cut, n, e, layer, op):
    """the first trusted position of a point in a chunk [e - n, e) under `cut` (None: uncut arm); None = no trusted row"""
    s = e - n
    if cut is None:
        return s
    f, b, w = cut
    if layer >= NL:   # logits: the output row only (compared where both hold one)
        return e - 1
    if layer < f:
        return s
    C = b + (NL - 1 - layer) * w
    if n <= C:
        return s
    if op in PRE:
        return e - C
    return e - C + WIN - 1


def compare(per, ref, inst, cut):
    ranks = sorted(per)
    r0 = ranks[0]
    ra, rb = per[r0][ref], per[r0][inst]
    pa, pb = prompt_len(ra), prompt_len(rb)
    ca, cb = chunks(ra, pa), chunks(rb, pb)
    res = dict(prompt=[pa, pb], chunks_ref=[(c[0], c[1], c[3]) for c in ca], chunks=[(c[0], c[1], c[3]) for c in cb])
    res["same_chunking"] = [(c[0], c[1]) for c in ca] == [(c[0], c[1]) for c in cb]
    toks = [p for p in range(min(pa, pb)) if ra[p][1] != rb[p][1]]
    res["prompt_token_diff"] = toks[:3]
    first = None
    perlayer = {}
    window, taps, logits = [], [], []
    for ci, (s, e, stb, Tb) in enumerate(cb):
        n = e - s
        # every point of the chunk: name -> list of (pos, rank) differing among trusted rows, and trusted counts
        diff = collections.defaultdict(set)
        order = None
        los = None
        for rk in ranks:
            A, B = per[rk].get(ref, {}), per[rk].get(inst, {})
            for p in range(s, e):
                xa, xb = A.get(p), B.get(p)
                if xa is None or xb is None:
                    continue
                if order is None:
                    order = list(xb[4])
                    los = {nm: trusted_lo(cut, n, e, *layer_of(nm)) for nm in order}
                if xa[4] == xb[4] and xa[3] == xb[3]:
                    continue
                ha, hb = hashes(xa[3]), hashes(xb[3])
                ia = {nm: i for i, nm in enumerate(xa[4])}
                for j, nm in enumerate(xb[4]):
                    lo = los.get(nm)
                    if lo is None or p < lo:
                        continue
                    i = ia.get(nm)
                    if i is None:
                        continue
                    if nm == "logits" and (ha[i] == 0 or hb[j] == 0):
                        continue
                    if ha[i] != hb[j]:
                        diff[nm].add((p, rk))
        if order is None:
            continue
        # window keys (kvn, last WIN rows) at layers >= 21 and taps (h0 at 37-39, last WIN + 1 rows)
        wl = [L for L in range(21, NL) if any(e - WIN <= p for (p, rk) in diff.get("L%02d.kvn" % L, ()))]
        window.append((ci, wl[0] if wl else None, len(wl)))
        tl = [L for L in (37, 38, 39) if any(e - WIN - 1 <= p for (p, rk) in diff.get("L%02d.h0" % L, ()))]
        taps.append((ci, tl))
        if "logits" in diff:
            logits.append(ci)
        for nm in order:
            if diff.get(nm):
                L, op = layer_of(nm)
                pts = sorted(diff[nm])
                offs = sorted({e - p for (p, rk) in pts})
                f = dict(chunk=ci, start=s, end=e, rows=n, T=Tb, point=nm, layer=L, op=op,
                         ranks=sorted({rk for (p, rk) in pts}), rows_diff=len({p for (p, rk) in pts}),
                         trusted=max(0, e - max(s, los[nm])) if nm != "logits" else 1, off_min=offs[0], off_max=offs[-1],
                         cut_C=(None if cut is None or L >= NL or L < cut[0] else cut[1] + (NL - 1 - L) * cut[2]))
                if first is None:
                    first = f
                break
        if first is not None and first["chunk"] == ci and not perlayer:
            for nm in order:
                L, op = layer_of(nm)
                if diff.get(nm) and L not in perlayer:
                    perlayer[L] = "%s:%d" % (op, len({p for (p, rk) in diff[nm]}))
    res["first"] = first
    res["layers"] = perlayer
    res["window"] = window
    res["taps"] = taps
    res["logits_diff_chunks"] = logits
    dec = [p for p in sorted(set(ra) & set(rb)) if p >= max(pa, pb) and ra[p][1] != rb[p][1]]
    res["decode_token_diff"] = dec[0] if dec else None
    res["decode_compared"] = len([p for p in set(ra) & set(rb) if p >= max(pa, pb)])
    return res


def analyze(a):
    per, first, _keep = load(a.dir)
    if not per:
        print("PSKLOC: no trace records in %s" % a.dir)
        return 1
    r0 = min(per)
    insts = [i for i in per[r0] if prompt_len(per[r0][i]) >= a.min_prompt]
    insts.sort(key=lambda i: first[i])
    if a.labels:
        return analyze_labels(a, per, first, insts)
    order = list(a.order)
    # the arms are the LAST len(order) long prompts (the start's warm-up prefill comes first and is ignored)
    skipped = insts[:-len(order)] if len(insts) > len(order) else []
    insts = insts[len(skipped):]
    print("PSKLOC instances: %d long prompts (>= %d positions) %s for order %s; ignored (earlier: the warm-up) %s" % (
        len(insts), a.min_prompt, [(i, prompt_len(per[r0][i])) for i in insts], a.order,
        [(i, prompt_len(per[r0][i])) for i in skipped]))
    if len(insts) != len(order):
        print("PSKLOC VIRHE: %d long prompts traced, %d arms in the order -- matching the first %d" % (
            len(insts), len(order), min(len(insts), len(order))))
    cuts = parse_cut(a.cut)
    if "A" not in order:
        print("PSKLOC VIRHE: no A arm")
        return 1
    ref = insts[order.index("A")]
    out, summ = [], []
    jobs = [(k, arm, inst, ref, "A%d" % (order.index("A") + 1)) for k, (arm, inst) in enumerate(zip(order, insts)) if inst != ref]
    for pr in (a.pairs or "").split(","):   # extra pairs X:Y -- the first X run against the first Y run
        if ":" not in pr:
            continue
        x, y = pr.strip().split(":")
        if x in order and y in order and order.index(x) < len(insts) and order.index(y) < len(insts):
            jobs.append((order.index(x), x, insts[order.index(x)], insts[order.index(y)], "%s%d" % (y, order.index(y) + 1)))
    for (k, arm, inst, rf, rtag) in jobs:
        cut = cuts.get(arm)
        r = compare(per, rf, inst, cut)
        r.update(arm=arm, run=k, inst=inst, cut=cut, ref=rtag)
        out.append(r)
        tag = "%s%d-vs-%s" % (arm, k + 1, rtag)
        print("PSKLOC %s chunks: %d (rows %s)%s; prompt %s; prompt tokens %s" % (
            tag, len(r["chunks"]), [c[2] for c in r["chunks"]][:12],
            "" if r["same_chunking"] else " -- VIRHE chunking differs from the reference: %s" % r["chunks_ref"][:12],
            r["prompt"], "identical" if not r["prompt_token_diff"] else "DIFFER at %s" % r["prompt_token_diff"]))
        f = r["first"]
        if f is None:
            line = "IDENTICAL on every trusted row of every traced point (%s)" % ("cut %s" % (cut,) if cut else "uncut")
        else:
            line = ("chunk %d [%d, %d) (%d rows, step T %d): %s -- %d of %d trusted rows differ, offsets from the chunk end "
                    "%d..%d, ranks %s%s" % (f["chunk"], f["start"], f["end"], f["rows"], f["T"], f["point"], f["rows_diff"],
                                           f["trusted"], f["off_min"], f["off_max"], f["ranks"],
                                           "" if f["cut_C"] is None else ", the layer's cone %d rows" % f["cut_C"]))
        print("PSKLOC %s FIRST %s" % (tag, line))
        if r["layers"]:
            print("PSKLOC %s layers (that chunk, per layer the first differing op:rows): %s" % (
                tag, " ".join("L%d=%s" % (L, v) for L, v in sorted(r["layers"].items()))))
        print("PSKLOC %s window kvn last %d rows, layers 21-39, per chunk (first differing layer, layers differing): %s" % (
            tag, WIN, " ".join("c%d:%s" % (c, "ok" if L is None else "L%d/%d" % (L, n)) for c, L, n in r["window"])))
        print("PSKLOC %s taps h0 L37-39 last %d rows per chunk: %s" % (
            tag, WIN + 1, " ".join("c%d:%s" % (c, "ok" if not t else ",".join("L%d" % x for x in t)) for c, t in r["taps"])))
        print("PSKLOC %s logits: %s; decode: %s" % (
            tag, "identical" if not r["logits_diff_chunks"] else "DIFFER in chunks %s" % r["logits_diff_chunks"],
            "%d positions compared, tokens identical" % r["decode_compared"] if r["decode_token_diff"] is None
            else "first differing token at position %d" % r["decode_token_diff"]))
        summ.append("%s %s" % (tag, "IDENTICAL" if f is None else "%s (chunk %d, %d rows)" % (f["point"], f["chunk"], f["rows_diff"])))
    print("PSKLOC summary: %s" % " | ".join(summ))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    return 0


def report(tag, r, cut):
    print("PSKLOC %s chunks: %d (rows %s)%s; prompt %s; prompt tokens %s" % (
        tag, len(r["chunks"]), [c[2] for c in r["chunks"]][:12],
        "" if r["same_chunking"] else " -- VIRHE chunking differs from the reference: %s" % r["chunks_ref"][:12],
        r["prompt"], "identical" if not r["prompt_token_diff"] else "DIFFER at %s" % r["prompt_token_diff"]))
    f = r["first"]
    if f is None:
        line = "IDENTICAL on every trusted row of every traced point (%s)" % ("cut %s" % (cut,) if cut else "uncut")
    else:
        line = ("chunk %d [%d, %d) (%d rows, step T %d): %s -- %d of %d trusted rows differ, offsets from the chunk end "
                "%d..%d, ranks %s%s" % (f["chunk"], f["start"], f["end"], f["rows"], f["T"], f["point"], f["rows_diff"],
                                       f["trusted"], f["off_min"], f["off_max"], f["ranks"],
                                       "" if f["cut_C"] is None else ", the layer's cone %d rows" % f["cut_C"]))
    print("PSKLOC %s FIRST %s" % (tag, line))
    if r["layers"]:
        print("PSKLOC %s layers (that chunk, per layer the first differing op:rows): %s" % (
            tag, " ".join("L%d=%s" % (L, v) for L, v in sorted(r["layers"].items()))))
    print("PSKLOC %s window kvn last %d rows, layers 21-39, per chunk (first differing layer, layers differing): %s" % (
        tag, WIN, " ".join("c%d:%s" % (c, "ok" if L is None else "L%d/%d" % (L, n)) for c, L, n in r["window"])))
    print("PSKLOC %s taps h0 L37-39 last %d rows per chunk: %s" % (
        tag, WIN + 1, " ".join("c%d:%s" % (c, "ok" if not t else ",".join("L%d" % x for x in t)) for c, t in r["taps"])))
    print("PSKLOC %s logits: %s; decode: %s" % (
        tag, "identical" if not r["logits_diff_chunks"] else "DIFFER in chunks %s" % r["logits_diff_chunks"],
        "%d positions compared, tokens identical" % r["decode_compared"] if r["decode_token_diff"] is None
        else "first differing token at position %d" % r["decode_token_diff"]))
    return "%s %s" % (tag, "IDENTICAL" if f is None else "%s (chunk %d, %d rows)" % (f["point"], f["chunk"], f["rows_diff"]))


def analyze_labels(a, per, first, insts):
    """--labels: the long prompts sharing the most common first 32 prompt tokens (the deepq prompts; the warm-up and the
    load streams' prompts differ) are the labelled runs, in step order (the last len(labels) of them)."""
    r0 = min(per)
    labels = a.labels.split()
    key = {i: tuple(per[r0][i][p][1] for p in range(32)) for i in insts}
    groups = collections.defaultdict(list)
    for i in insts:
        groups[key[i]].append(i)
    best = max(groups.values(), key=lambda v: (len(v), first[v[-1]])) if groups else []
    other = [i for i in insts if i not in best]
    runs = best[-len(labels):]
    print("PSKLOC instances: %d long prompts (>= %d positions); the deepq group %d %s -> labels %s; other long prompts %s" % (
        len(insts), a.min_prompt, len(best), [(i, prompt_len(per[r0][i])) for i in best], labels,
        [(i, prompt_len(per[r0][i])) for i in other]))
    if len(best) != len(labels):
        print("PSKLOC VIRHE: %d deepq prompts traced, %d labels -- the last %d matched" % (len(best), len(labels), len(runs)))
    lab = dict(zip(labels[len(labels) - len(runs):], runs))
    cuts = parse_cut(a.cut)
    out, summ = [], []
    for pr in (a.pairs or "").replace(" ", ",").split(","):
        if ":" not in pr:
            continue
        x, y = pr.split(":")
        if x not in lab or y not in lab:
            print("PSKLOC VIRHE: pair %s: no run labelled %s" % (pr, x if x not in lab else y))
            continue
        cut = cuts.get(x)
        r = compare(per, lab[y], lab[x], cut)
        r.update(arm=x, ref=y, inst=lab[x], cut=cut)
        out.append(r)
        summ.append(report("%s-vs-%s" % (x, y), r, cut))
    print("PSKLOC summary: %s" % " | ".join(summ))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    return 0


def selftest(_a):
    d = tempfile.mkdtemp(prefix="pskloc-")
    names = ["L00.h0", "L00.kvn", "L00.o", "L39.h0", "L39.kvn", "L39.o", "L39.moe_out", "final", "logits"]
    kinds = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    key = 0x77
    P = 400
    for rank in (0, 1):
        with open(os.path.join(d, "bt-r%d-1.names" % rank), "w") as f:
            f.write("N %016x %d\n" % (key, len(names)))
            for nm, kd in zip(names, kinds):
                f.write("%s %d\n" % (nm, kd))
        with open(os.path.join(d, "bt-r%d-1.bin" % rank), "wb") as f:
            step = 0
            for run in range(2):   # A then B, one slot
                for (s, e) in ((0, 200), (200, P)):
                    n = e - s
                    recs = []
                    for r in range(n):
                        p = s + r
                        h = [(hash((p, nm, rank)) & 0xffffffffffffff) | 1 for nm in names]
                        h[-1] = h[-1] if p == e - 1 else 0
                        if run == 1:
                            # stale: L39.o at a row before the trusted range (cone 160, trusted from e - 33)
                            if p == e - 100:
                                h[5] ^= 2
                            # real: L39.moe_out at the last row of the second chunk, rank 1
                            if p == P - 1 and rank == 1:
                                h[6] ^= 4
                        recs.append((r, p, 1000 + p, 0, h))
                    f.write(HDR.pack(b"BIT2", rank, step, n, 1, 1 if e == P else 0, len(names), len(recs), 0, 0, 0, key))
                    f.write(struct.pack("<2i", 0, n))
                    f.write(struct.pack("<1i", 5))
                    f.write(struct.pack("<1i", n))
                    f.write(struct.pack("<1i", s))
                    if e == P:
                        f.write(struct.pack("<1i", n - 1))
                    for (r, p, t, sq, h) in recs:
                        f.write(struct.pack("<4i", r, p, t, sq))
                        f.write(struct.pack("<%dQ" % len(h), *h))
                    step += 1
    per, first, _k = load(d)
    insts = sorted([i for i in per[0] if prompt_len(per[0][i]) >= 100], key=lambda i: first[i])
    r = compare(per, insts[0], insts[1], (21, 160, 128))
    shutil.rmtree(d, ignore_errors=True)
    f = r["first"] or {}
    ok = len(insts) == 2 and f.get("point") == "L39.moe_out" and f.get("chunk") == 1 and f.get("ranks") == [1] and \
        f.get("rows_diff") == 1 and f.get("off_min") == 1 and r["same_chunking"]
    print("SELFTEST %s (instances %s, first %s)" % ("OK" if ok else "FAILED", insts,
                                                   {k: f.get(k) for k in ("chunk", "point", "ranks", "rows_diff", "off_min")}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "selftest"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--order", default="AB")
    ap.add_argument("--cut", default="B=21:160:128")
    ap.add_argument("--min-prompt", type=int, default=1000)
    ap.add_argument("--pairs", default="", help="extra comparisons X:Y (arm X's run against arm Y's), comma-separated")
    ap.add_argument("--labels", default="", help="labels of the deepq runs in step order (then --pairs X:Y by label)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if not a.dir:
        ap.error("analyze needs DIR")
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
