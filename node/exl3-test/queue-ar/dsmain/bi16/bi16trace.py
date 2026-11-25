#!/usr/bin/env python3
"""bi16trace.py (12.11.2026, queue-ar/dsmain/bi16, ds-main-port bi16 builder) -- = bitrace2.py (cbbd1852, itself bitrace.py
c141541a + --min-occ; copied, the original unchanged) + what item 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16 reads: the probes
found by their token ids, each step's decode-row count (DT) beside every row, the first diverging op's CLASS, the
independent flips of its layer, a small extract of the rows around it, and a step-log counter.

  bi16trace.py analyze DIR [--probes PROBES.json] [--order a1,ld,a2] [--json OUT.json] [--extract OUT.json.gz]
                           [--min-prompt 4] [--min-occ 3]
      bitrace.py's reading (its text below) over the tracer files of DIR (bt-r<rank>-<pid>.bin + .names): requests are
      (slot, n) instances, the probe's prompt is its first prefill rows' tokens, per (sequence, position) the LAST row
      computed there is kept, ld vs a1 / a2 vs a1 walk the positions ascending and the points in issue order (h0, xn, qr, q,
      kvn, iq, iw, sel, o, oa, attn_out, h1, ffn_in, router, ids, ew, sout, moe_out a layer, then final, logits) per rank;
      the first differing (position, point, rank) is the first diverging op. Added here:
        * --probes (bi16probe.py tok): a probe = the occurrences of a prompt whose key is the probe's ids (a leading BOS
          either side allowed); without it, or when the ids match nothing, every prompt seen >= --min-occ times (bitrace2.py's rule). The occurrences in
          first-step order take the --order labels (a1, ld, a2: alone, loaded, alone again).
        * DT: every step a probe's sequence had rows in (rank 0's records: T, DT = the step's decode rows, S, prefill
          sequences) -> per run: steps, DT max, steps with DT > 48, the first position computed in a > 48 step.
        * the first op's class by what writes the point in step_bi (int8 7e03384, deepseek41.cpp:4288-5003): dense GEMM
          over the WHOLE decode segment (qr q_a, kvn kv_g -> fused key norm <= 32, iw i_w, q q_b / K1 fold a, iq i_q),
          dense GEMM in <= 48-row pieces (oa o_a, attn_out o_b, logits head), attention over the whole segment (o:
          attn_sparse_sink), indexer over the segment's runs (sel: index_score / topk), fused norm in <= 32-row pieces (xn,
          h1, ffn_in, final; h0 = the previous layer's post + next collapse), MoE in <= 16-row passes (router, ids, ew,
          sout, moe_out).
        * the first op's layer at the first position on its rank: every point equal / differing, and the INDEPENDENT flips
          = differing points whose traced inputs (xn -> qr, kvn, iw; qr -> q, iq; iq + iw -> sel; q + kvn + sel -> o; ...)
          are all equal: each one an op that flipped by itself (e.g. qr AND kvn: two whole-segment GEMMs, not one).
      Prints "BI16 p<i> ..." a probe, "BI16 summary: ..." and "BI16GUARD {json}" (records, probes found, the loaded runs'
      > 48-row steps, the a2-vs-a1 control, divergences found). --extract: positions first-divergence -2 .. +2 of every
      run, every rank, every point (a few hundred kB gzipped; the raw files can then go).
  bi16trace.py steps --log ENGINE.log[.gz] --win NAME:T0:T1 [--win ...] [--json OUT.json]
      the engine's step log (RADIANCE_LOG_STEPS=1, docker logs -t) in each window: steps, decode steps, decode steps with
      n_tok > 48 (a decode step's n_tok IS its decode rows), the largest, MIXED steps (their decode rows cannot be summed
      from the line: it lists at most 8 sequences -- the reason kv4sum.py's bi48 read 'most 48'), and the 'lever BI stats'
      > 48 counter's delta. Prints "STEPS {json}" and one "STEPLINE ..." line.
  bi16trace.py selftest
      a synthetic trace (two ranks; one probe alone, loaded in 50-row decode steps with qr AND kvn flipped at layer 0 on
      rank 1 from position 5, alone again) + a synthetic step log; checks the probe match (BOS offset), the first op, its
      class, the independent flips, the DT reading, the control, the extract and the step counter. No server.
Standard library only.

bitrace.py's own text (unchanged reading):
  bitrace.py analyze DIR [--json OUT.json] [--min-prompt 4]
      Every request is a sequence (its engine seq id). A probe is a prompt that occurs in three or more sequences (alone,
      under load, alone again: the run order); the c4 streams' and the long prefills' prompts occur once. For each probe its
      occurrences are ordered by first step: a1, ld, a2 (more: the extra ones are listed). Per (sequence, position) the LAST
      row computed at that position is kept (a rejected verify row is recomputed later). ld vs a1 (and a2 vs a1): positions
      ascending; the first position whose token differs ends the walk (the row before it holds the cause); at each position
      with equal tokens the points are compared in issue order, per rank; the first differing (position, point, rank) is the
      first diverging op. Its step's shape is printed beside it (rows T, decode rows DT, sequences S, prefill sequences, the
      row's class decode / prefill and its sequence's q_len), alone and loaded.
"""
import argparse, calendar, collections, glob, gzip, json, os, re, shutil, struct, sys, tempfile, time

HDR = struct.Struct("<4siq8iQ")   # magic, rank, step, T, S, NO, NP, nsel, DT, SD, n_spec, names key

# what writes each point in step_bi (int8 7e03384) -> (class, the op as issued)
OPCLASS = {
    "h0": ("other", "layer input: embedding / expand at layer 0, else the previous layer's post_f + next collapse (pcn_a, fused norm <= 32-row pieces)"),
    "xn": ("fused norm", "collapse + attention norm (cn_a, <= 32-row pieces)"),
    "qr": ("dense GEMM (whole decode segment)", "q_a: gemm_nt_ggml over the segment's rows (lin_issue(L.q_a, s.n))"),
    "kvn": ("dense GEMM (whole decode segment)", "kv_g: gemm_nt_ggml over the segment's rows, then the fused window-key norm (<= 32-row pieces)"),
    "iw": ("dense GEMM (whole decode segment)", "i_w: the indexer weight projection over the segment's rows"),
    "q": ("dense GEMM (whole decode segment)", "q_b / K1 fold a (q_bf: q norm + wq_b) over the segment's rows"),
    "iq": ("dense GEMM (whole decode segment)", "i_q: the index query over the segment's rows"),
    "sel": ("indexer", "index_score / candidates / topk over the segment's runs (run rows = the segment up to irows)"),
    "o": ("attention", "attn_sparse_sink (+ K1 fold d) over the whole decode segment"),
    "oa": ("dense GEMM (<= 48-row pieces)", "o_a groups, rows(kBiBand)"),
    "attn_out": ("dense GEMM (<= 48-row pieces)", "o_b, rows(kBiBand)"),
    "h1": ("fused norm", "attention all-reduce (exact, whole) + post / collapse / ffn norm (pcn_f, <= 32-row pieces)"),
    "ffn_in": ("fused norm", "the ffn norm (pcn_f / cn_f, <= 32-row pieces)"),
    "router": ("MoE", "router (gemm_nt_f32) per MoE pass (<= 16-row passes at ds41_bi 2)"),
    "ids": ("MoE", "top-k per MoE pass"),
    "ew": ("MoE", "expert weights per MoE pass"),
    "sout": ("MoE", "shared expert arm per MoE pass"),
    "moe_out": ("MoE", "routed experts (EXL3) + gather + all-reduce per MoE pass"),
    "final": ("fused norm", "final norm (pcn_fin / cn_fin, <= 32-row pieces)"),
    "logits": ("dense GEMM (<= 48-row pieces)", "the head (logits_gemm_ggml, <= 48-row launches)"),
}
# a point's traced inputs at the same layer and position (caches written by earlier positions are not traced)
DEPS = {"xn": ["h0"], "qr": ["xn"], "kvn": ["xn"], "iw": ["xn"], "q": ["qr"], "iq": ["qr"], "sel": ["iq", "iw"],
        "o": ["q", "kvn", "sel"], "oa": ["o"], "attn_out": ["oa"], "h1": ["attn_out", "h0"], "ffn_in": ["h1"],
        "router": ["ffn_in"], "ids": ["router"], "ew": ["router"], "sout": ["ffn_in"], "moe_out": ["ffn_in", "ids", "ew", "sout"]}


def split_point(nm):
    m = re.match(r"^L(\d+)\.(.+)$", nm)
    return (int(m.group(1)), m.group(2)) if m else (None, nm)


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
        need = HDR.size + 4 * (S + 1 + 3 * S + NO) + nsel * (16 + 8 * NP)
        if o + need > n:   # a record cut by the container's stop (the last step): ignored
            break
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
    """rank -> request instance -> {pos: (step, tok, hashes dict, shape, names)}; instance -> first step; instance -> rank 0's
    steps [(step, T, DT, S, npf, positions)]. Requests are (slot, n) instances (bitrace.py's reading)."""
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    first = {}
    steps = collections.defaultdict(dict)   # (rank, sid) -> step -> [T, DT, S, npf, [positions]]
    nrec = 0
    for b in sorted(glob.glob(os.path.join(d, "bt-r*-*.bin"))):
        names = read_names(b[:-4] + ".names")
        inst = {}
        for st in read_bin(b, names):
            nrec += 1
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
                e = steps[(st["rank"], sid)].setdefault(st["step"], [st["T"], st["DT"], S, npf, []])
                e[4].append(pos)
    return per, first, steps, nrec


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


def key_matches(key, ids):
    """the trace's prompt key is the probe's ids (a leading BOS on either side allowed)"""
    if not ids:
        return False
    for c in (tuple(ids), tuple(ids[1:])):
        if not c:
            continue
        for k in (tuple(key), tuple(key[1:])):
            if k == c:
                return True
    return False


def layer_view(per, rk, a, b, pos, layer):
    """at (rank, pos): every point of `layer` equal / differing (a vs b) and the independent flips"""
    ra, rb = per[rk][a].get(pos), per[rk][b].get(pos)
    if ra is None or rb is None:
        return None
    st = {}
    for nm in rb[4]:
        L, op = split_point(nm)
        if L != layer or nm not in ra[2]:
            continue
        st[op] = ra[2][nm] != rb[2][nm]
    if layer is not None and layer > 0:   # h0 at layer l is the previous layer's output
        pass
    indep = [op for op, dif in st.items() if dif and all(st.get(dp) is False for dp in DEPS.get(op, [])) and
             (DEPS.get(op) or op == "h0")]
    return dict(differ=[op for op, dif in st.items() if dif], equal=[op for op, dif in st.items() if not dif], independent=indep)


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
        layer, op = split_point(nm)
        cls, what = OPCLASS.get(op, ("other", op))
        out.update(pos=pos, point=nm, op=op, layer=layer, point_index=i, rank=rk, alone=sa, loaded=sb, cls=cls, what=what,
                   layer_view=layer_view(per, rk, a, b, pos, layer))
        # every rank's first differing point at this position (the same op on every rank, or a rank-local one)
        rks = {}
        for r in ranks:
            ra, rb = per[r][a].get(pos), per[r][b].get(pos)
            if ra is None or rb is None:
                continue
            for nm2 in rb[4]:
                if nm2 in ra[2] and ra[2][nm2] != rb[2][nm2]:
                    rks[r] = nm2
                    break
        out["ranks_first"] = rks
    return out


def dt_stats(steps, rank, sid):
    s = steps.get((rank, sid), {})
    dts = [v[1] for v in s.values()]
    gt = sorted((k for k, v in s.items() if v[1] > 48))
    first_pos = min((min(s[k][4]) for k in gt), default=None)
    return dict(steps=len(s), dt_max=max(dts) if dts else None, gt48=len(gt), first_gt48_pos=first_pos,
                t_max=max((v[0] for v in s.values()), default=None))


def shape_s(s):
    return "T %d (decode rows DT %d%s, %d seqs, %d prefill seqs; row %s, its q_len %d%s)" % (
        s["T"], s["DT"], " > 48" if s["DT"] > 48 else "", s["S"], s["npf"], s["cls"], s["qlen"],
        ", n_spec %d" % s["nspec"] if s["nspec"] else "")


def extract_rows(per, sid, lo, hi):
    out = {}
    for rk in sorted(per):
        rows = per[rk].get(sid, {})
        out[str(rk)] = {str(p): dict(step=v[0], tok=v[1], shape=v[3], points={k: "%016x" % h for k, h in v[2].items()})
                        for p, v in sorted(rows.items()) if lo <= p <= hi}
    return out


def analyze(a):
    per, first, steps, nrec = load(a.dir)
    if not per:
        print("BI16 VIRHE no trace records in %s" % a.dir)
        print("BI16GUARD " + json.dumps(dict(records=0, probes_found=0)))
        return 1
    order = [x for x in a.order.split(",") if x]
    pk = prompts(per)
    probes = []   # [(label, [sids in first-step order])]
    want = None
    if a.probes:
        try:
            want = json.load(open(a.probes)).get("probes")
        except (OSError, ValueError):
            want = None
    if want and all(p.get("ids") for p in want):
        for p in want:
            sids = sorted([s for s, k in pk.items() if key_matches(k, p["ids"])], key=lambda s: first[s])
            probes.append(("p%d" % p["i"], sids))
        if any(not v for _, v in probes):   # /tokenize's ids are not the trace's prompt rows: bitrace2.py's grouping instead
            print("BI16 note: probe ids %s matched no traced prompt for %s -- every prompt seen >= %d times instead" % (
                [p["i"] for p in want], [k for k, v in probes if not v], a.min_occ))
            probes = []
    if not probes:
        groups = collections.defaultdict(list)
        for sid, key in pk.items():
            if len(key) >= a.min_prompt:
                groups[key].append(sid)
        for k, v in sorted(groups.items(), key=lambda kv: min(first[s] for s in kv[1])):
            if len(v) >= a.min_occ:
                probes.append(("k%d" % len(probes), sorted(v, key=lambda s: first[s])))
    ranks = sorted(per)
    r0 = ranks[0]
    res, cnt_op, cnt_cls, dt_div = [], collections.Counter(), collections.Counter(), collections.Counter()
    ext = {}
    g = dict(records=nrec, ranks=len(ranks), probes_asked=len(want) if want else None, probes_found=0, runs_ok=0,
             ld_gt48=0, control_identical=0, control_n=0, divergences=0, div_gt48=0)
    for label, sids in probes:
        lab = {order[k] if k < len(order) else "x%d" % k: s for k, s in enumerate(sids)}
        e = dict(probe=label, seqs=sids, labels={k: list(v) for k, v in lab.items()},
                 dt={k: dt_stats(steps, r0, v) for k, v in lab.items()})
        if len(sids) < 2 or "a1" not in lab:
            print("BI16 %s: VIRHE %d occurrence(s) in the trace (want %s) -- not compared" % (label, len(sids), ",".join(order)))
            res.append(e)
            continue
        g["probes_found"] += 1
        if len(sids) >= len(order):
            g["runs_ok"] += 1
        a1 = lab["a1"]
        e["cmp"] = {}
        for k, s in lab.items():
            if k == "a1":
                continue
            e["cmp"][k] = compare(per, a1, s)
        ld = e["cmp"].get("ld")
        ctl = e["cmp"].get("a2", "absent") if "a2" in lab else "absent"
        if e["dt"].get("ld", {}).get("gt48"):
            g["ld_gt48"] += 1
        if ctl != "absent":
            g["control_n"] += 1
            if ctl is None:
                g["control_identical"] += 1
        if ld is None:
            line = "loaded IDENTICAL to alone over every traced row and point"
        elif "point" in ld:
            g["divergences"] += 1
            if ld["loaded"]["DT"] > 48:
                g["div_gt48"] += 1
            lv = ld.get("layer_view") or {}
            line = ("FIRST DIVERGENCE pos %d point %s (rank %d; first point a rank %s)%s | class: %s = %s | loaded step %d: %s | "
                    "alone step %d: %s | L%02d at pos %d rank %d: differ [%s], equal [%s], INDEPENDENT flips [%s]") % (
                ld["pos"], ld["point"], ld["rank"], ld.get("ranks_first"),
                "" if ld["token_diff_pos"] is None else ", tokens differ from pos %d" % ld["token_diff_pos"],
                ld["cls"], ld["what"], ld["loaded"]["step"], shape_s(ld["loaded"]), ld["alone"]["step"], shape_s(ld["alone"]),
                ld["layer"] if ld["layer"] is not None else -1, ld["pos"], ld["rank"], " ".join(lv.get("differ", [])),
                " ".join(lv.get("equal", [])), " ".join(lv.get("independent", [])))
            cnt_op[ld["point"]] += 1
            cnt_cls[ld["cls"]] += 1
            dt_div["DT %d" % ld["loaded"]["DT"]] += 1
        else:
            line = "tokens differ from pos %d with no differing traced point before it (an untraced op or cache entry)" % ld["token_diff_pos"]
            cnt_op["(untraced)"] += 1
        d1, dl = e["dt"].get("a1", {}), e["dt"].get("ld", {})
        print("BI16 %s seqs %s: %s | loaded run's rows: %s steps, DT max %s, %s steps > 48 (first row in one: pos %s) | alone: %s "
              "steps, DT max %s | control a2 vs a1: %s" % (
                  label, ", ".join("%s=%s" % (k, tuple(v)) for k, v in lab.items()), line, dl.get("steps"), dl.get("dt_max"),
                  dl.get("gt48"), dl.get("first_gt48_pos"), d1.get("steps"), d1.get("dt_max"),
                  "absent" if ctl == "absent" else "IDENTICAL" if ctl is None else "DIFFERS (pos %s, %s)" % (
                      ctl.get("pos", ctl["token_diff_pos"]), ctl.get("point", "token"))))
        res.append(e)
        if a.extract:
            p0 = ld["pos"] if ld and "pos" in ld else (ld["token_diff_pos"] if ld else 0) or 0
            ext[label] = {k: extract_rows(per, s, p0 - 2, p0 + 2) for k, s in lab.items()}
            ext[label]["_center"] = p0
    print("BI16 summary: %d probe(s) found of %s, %d with every run; first diverging point %s; class %s; loaded step at the "
          "divergence %s; loaded runs with > 48-row steps %d; control a2 vs a1 identical %d of %d; %d trace records, %d ranks" % (
              g["probes_found"], g["probes_asked"] if g["probes_asked"] is not None else "(no --probes)", g["runs_ok"],
              dict(cnt_op.most_common()), dict(cnt_cls.most_common()), dict(dt_div.most_common()), g["ld_gt48"],
              g["control_identical"], g["control_n"], nrec, len(ranks)))
    print("BI16GUARD " + json.dumps(g))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(guard=g, probes=res), f, indent=1, default=str)
    if a.extract:
        with gzip.open(a.extract, "wt") as f:
            json.dump(ext, f, default=str)
    return 0


# ---------------------------------------------------------------- the step log
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")      # kv4sum.py's / ncsteps.py's
BIST = re.compile(r"lever BI stats: (\d+) segmented steps, .*?(\d+) steps with > 48 decode rows; ds41_bi (\d+)")
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")


def ts_split(line):
    m = TSRE.match(line)
    if not m:
        return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z":
        t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


def steps_cmd(a):
    op = gzip.open if a.log.endswith(".gz") else open
    st, bis = [], []
    try:
        with op(a.log, "rt", errors="replace") as fh:
            for l in fh:
                if "n_seq=" not in l and "lever BI stats" not in l:
                    continue
                t, rest = ts_split(l.rstrip("\n"))
                if t is None:
                    continue
                m = STEP.search(rest)
                if m:
                    st.append((t, m.group(2), int(m.group(3)), int(m.group(4))))
                    continue
                m = BIST.search(rest)
                if m:
                    bis.append((t, int(m.group(1)), int(m.group(2))))
    except OSError as e:
        print("STEPLINE VIRHE no engine log (%s)" % e)
        return 2
    out = {}
    for w in a.win:
        nm, t0, t1 = w.rsplit(":", 2)
        t0, t1 = float(t0), float(t1)
        W = [s for s in st if t0 <= s[0] < t1]
        dec = [s[3] for s in W if s[1] == "decode"]
        mx = [s for s in W if s[1] == "MIXED"]
        b0 = [b for b in bis if b[0] <= t0]
        b1 = [b for b in bis if b[0] <= t1]
        out[nm] = dict(steps=len(W), decode=len(dec), decode_gt48=sum(1 for x in dec if x > 48),
                       decode_max=max(dec) if dec else 0, decode_p50=sorted(dec)[len(dec) // 2] if dec else 0,
                       mixed=len(mx), mixed_nseq_max=max((s[2] for s in mx), default=0),
                       bistats_gt48_delta=(b1[-1][2] - (b0[-1][2] if b0 else 0)) if b1 else None)
    print("STEPS " + json.dumps(out))
    print("STEPLINE " + " | ".join("%s: %d steps, decode %d (%d with > 48 rows = n_tok, most %d, p50 %d), MIXED %d (up to %d "
                                   "sequences; decode rows not summable from the 8-entry line); BI stats > 48 +%s" % (
                                       k, v["steps"], v["decode"], v["decode_gt48"], v["decode_max"], v["decode_p50"], v["mixed"],
                                       v["mixed_nseq_max"], v["bistats_gt48_delta"]) for k, v in out.items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
    return 0


# ---------------------------------------------------------------- selftest
def selftest(_a):
    d = tempfile.mkdtemp(prefix="bi16trace-")
    names = ["L00.h0", "L00.xn", "L00.qr", "L00.q", "L00.kvn", "L00.iq", "L00.iw", "L00.sel", "L00.o", "final", "logits"]
    kinds = [0] * 10 + [1]
    key = 0x1234
    prompt = [5, 6, 7, 8]
    for rank in (0, 1):
        with open(os.path.join(d, "bt-r%d-1.names" % rank), "w") as f:
            f.write("N %016x %d\n" % (key, len(names)))
            for nm, kd in zip(names, kinds):
                f.write("%s %d\n" % (nm, kd))
        with open(os.path.join(d, "bt-r%d-1.bin" % rank), "wb") as f:
            step = 0
            for run in range(3):
                sid = 7
                recs = [(r, p, prompt[p], 0, [hash((p, prompt[p], nm)) & 0xffffffffffffffff for nm in names]) for r, p in
                        enumerate(range(4))]
                f.write(HDR.pack(b"BIT2", rank, step, 4, 1, 1, len(names), len(recs), 0, 0, 0, key))
                f.write(struct.pack("<2i", 0, 4))
                f.write(struct.pack("<1i", sid))
                f.write(struct.pack("<1i", 4))
                f.write(struct.pack("<1i", 0))
                f.write(struct.pack("<1i", 3))
                for (r, p, t, sq, h) in recs:
                    f.write(struct.pack("<4i", r, p, t, sq))
                    f.write(struct.pack("<%dQ" % len(h), *h))
                step += 1
                for p in range(4, 7):
                    t = 20 + p
                    h = [hash((p, t, nm)) & 0xffffffffffffffff for nm in names]
                    DT = 50 if run == 1 else 1
                    if run == 1 and p >= 5 and rank == 1:
                        for nm in ("L00.qr", "L00.kvn", "L00.q", "L00.iq", "L00.sel", "L00.o", "final", "logits"):
                            h[names.index(nm)] ^= 1
                    f.write(HDR.pack(b"BIT2", rank, step, DT, 1, 1, len(names), 1, DT, 1, 0, key))
                    f.write(struct.pack("<2i", 0, DT))
                    f.write(struct.pack("<1i", sid))
                    f.write(struct.pack("<1i", DT))
                    f.write(struct.pack("<1i", p))
                    f.write(struct.pack("<1i", 0))
                    f.write(struct.pack("<4i", 0, p, t, 0))
                    f.write(struct.pack("<%dQ" % len(h), *h))
                    step += 1
    pj = os.path.join(d, "probes.json")
    with open(pj, "w") as f:
        json.dump({"probes": [{"i": 9, "ids": [1] + prompt}]}, f)   # tokenize with a BOS the trace does not carry
    ns = argparse.Namespace(dir=d, probes=pj, order="a1,ld,a2", json=os.path.join(d, "o.json"),
                            extract=os.path.join(d, "x.json.gz"), min_prompt=4, min_occ=3)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = analyze(ns)
    txt = buf.getvalue()
    j = json.load(open(ns.json))
    e = j["probes"][0]
    ld = e["cmp"]["ld"]
    x = json.load(gzip.open(ns.extract, "rt"))
    pj2 = os.path.join(d, "probes2.json")   # ids that match nothing: the fallback must still find the probe
    with open(pj2, "w") as f:
        json.dump({"probes": [{"i": 9, "ids": [99, 98, 97]}]}, f)
    nf = argparse.Namespace(dir=d, probes=pj2, order="a1,ld,a2", json=os.path.join(d, "o2.json"), extract=None, min_prompt=4,
                            min_occ=3)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze(nf)
    j2 = json.load(open(nf.json))
    fb_ok = j2["guard"]["probes_found"] == 1 and j2["probes"][0]["cmp"]["ld"]["point"] == "L00.qr"
    lg = os.path.join(d, "e.log")
    with open(lg, "w") as f:
        f.write("2026-11-12T07:51:21.000000000Z I step 1 decode n_seq=16 n_tok=96  [0] slot=0 q=6 ctx=10\n")
        f.write("2026-11-12T07:51:22.000000000Z I step 2 decode n_seq=8 n_tok=48  [0] slot=0 q=6 ctx=16\n")
        f.write("2026-11-12T07:51:23.000000000Z I step 3 MIXED n_seq=16 n_tok=1179  [0] slot=0 q=6 ctx=22\n")
        f.write("2026-11-12T07:51:30.000000000Z I step 4 decode n_seq=16 n_tok=96  [0] slot=0 q=6 ctx=28\n")
    t0 = calendar.timegm(time.strptime("2026-11-12T07:51:21", "%Y-%m-%dT%H:%M:%S"))
    sa = argparse.Namespace(log=lg, win=["w:%d:%d" % (t0, t0 + 5)], json=os.path.join(d, "s.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        steps_cmd(sa)
    s = json.load(open(sa.json))["w"]
    shutil.rmtree(d, ignore_errors=True)
    lv = ld.get("layer_view") or {}
    ok = (rc == 0 and ld and ld.get("pos") == 5 and ld.get("point") == "L00.qr" and ld.get("rank") == 1 and
          ld.get("cls", "").startswith("dense GEMM (whole") and sorted(lv.get("independent", [])) == ["kvn", "qr"] and
          "o" in lv.get("differ", []) and "xn" in lv.get("equal", []) and e["cmp"]["a2"] is None and
          e["dt"]["ld"]["dt_max"] == 50 and e["dt"]["ld"]["gt48"] == 3 and e["dt"]["a1"]["dt_max"] == 1 and
          j["guard"]["ld_gt48"] == 1 and j["guard"]["control_identical"] == 1 and j["guard"]["div_gt48"] == 1 and
          x["p9"]["_center"] == 5 and "L00.qr" in x["p9"]["ld"]["1"]["5"]["points"] and
          s["decode"] == 2 and s["decode_gt48"] == 1 and s["mixed"] == 1 and s["mixed_nseq_max"] == 16 and
          "BI16 summary" in txt and "BI16GUARD" in txt and fb_ok)
    print("SELFTEST %s (divergence %s, class %s, independent %s, ld DT %s, control %s, id fallback %s, steps %s)" % (
        "OK" if ok else "FAILED", {k: ld.get(k) for k in ("pos", "point", "rank")} if ld else None, ld.get("cls") if ld else None,
        lv.get("independent"), e["dt"].get("ld"), e["cmp"].get("a2"), fb_ok, s))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("analyze", "steps", "selftest"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--probes")
    ap.add_argument("--order", default="a1,ld,a2")
    ap.add_argument("--extract")
    ap.add_argument("--min-prompt", type=int, default=4)
    ap.add_argument("--min-occ", type=int, default=3)
    ap.add_argument("--log")
    ap.add_argument("--win", action="append", default=[])
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "steps":
        if not a.log or not a.win:
            ap.error("steps needs --log and --win")
        return steps_cmd(a)
    if not a.dir:
        ap.error("analyze needs DIR")
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
