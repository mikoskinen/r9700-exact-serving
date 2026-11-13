#!/usr/bin/env python3
"""pl20.py (12.11.2026, queue-ar/dsmain/pskl20, ds-main-port PSK-L20/DS builder) -- the readings of dsm-pskl20.sh (item
57-p4zyzzzzz-ds-r00zzz-cz-pskl20): levers PSK-L20 (fn4 ds41_psk_l20) and PSK-DS (fn4 dspark_pfctx) on the ds41-pskl20 image.
Standard library only. Reuses queue-ar/dsmain/nc/ncsteps.py READ ONLY (importlib; read_steps / stat, its step-log regexes:
"step K prefill|decode|MIXED n_seq=.. n_tok=..  [i] slot=S q=Q ctx=C", the wall of a step = the next step line's receipt time
minus its own, from `docker logs -t`).

  pl20.py sdelta A.json B.json                 /stats B - A: snapshots, staged GB, the psk counters (design 2's and PSK-L20's),
                                               dspark_pfctx's -> one JSON line
  pl20.py guard ARM DELTAJSON                  the counter guard (standing rule: a lever that silently does nothing = VIRHE):
                                               design 2's latched / planned_steps / ckpt_dropped / layers_skipped grow and
                                               plan_mismatch does not, in every arm; L / X: l20_cut_passes and l20_rows_skipped
                                               grow; B / D: no l20 counter moves; D / X: dspark_pfctx passes and rows_skipped
                                               grow; B / L: no pfctx counter moves -> "OK ..." or "FAIL ..."
  pl20.py steps --log ENGINE.gz --marks MARKS.jsonl [--warm 15]
                                               per mb block window [t0 + warm, t1] (the marks the body wrote): steps by class,
                                               newcomer steps (an entry with q >= 256 and ctx <= 64: a fresh prompt's prefill,
                                               ncsteps' FRESH_Q / FRESH_CTX) with their wall p50 / mean, MIXED / decode walls
                                               -> one "PL20STEPS {json}" line a block
  pl20.py acc ACC.jsonl                        the drafter check: every (prompt, run) of a deterministic greedy NST 5 pin
                                               (draft_adaptive 0, depth 0 off, occ 0, hybrid 0, dspark_conf 0) -- the answer and
                                               the request's draft tokens / accepted (/metrics deltas) equal to the first B run's
                                               -> "ACC {json}" + a verdict line
  pl20.py texts --pskld P --tests T --doffs a,b,c ...   offline: every text offset the item uses starts with a distinct block
  pl20.py summary --out DIR                    the item's final lines (RESULT xpl20 summary)
  pl20.py selftest --ncsteps NCSTEPS.py        offline: the step parser on a synthetic log, the guard rules, sdelta, acc
"""
import argparse, gzip, importlib.util, json, math, os, re, statistics, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
NCS_DEFAULT = os.path.join(HERE, "..", "nc", "ncsteps.py")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
NAME = {"B": "B", "L": "L", "D": "D", "X": "LD"}
PSKD2 = ("latched", "planned_steps", "ckpt_dropped", "layers_skipped")
L20 = ("l20_cut_passes", "l20_rows_skipped", "l20_cone_empty", "l20_layers_not_staged")
PFC = ("passes", "rows_skipped", "rows_run")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rd_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return {}


def rows(p):
    out = []
    try:
        with open(p) as f:
            for l in f:
                l = l.strip()
                if not l: continue
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError:
        pass
    return out


def num(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict): return 0
        x = x.get(k)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def sdelta(A, B):
    g = lambda *p: num(B, *p) - num(A, *p)   # noqa: E731
    d = {"ckpts": int(g("checkpoints_written")), "staged_gb": round(g("link", "staged_bytes") / 1e9, 2),
         "staged_gb_r0": round(g("experts", "staged_bytes") / 1e9, 2), "stage_layers_r0": int(g("experts", "stage_layers")),
         "ar_gb": round(g("link", "ar_bytes") / 1e9, 2), "has_psk": "psk" in B, "has_pfctx": "dspark_pfctx" in B,
         "pfctx_mode": num(B, "dspark_pfctx", "mode") if "dspark_pfctx" in B else None}
    for k in PSKD2 + ("plan_mismatch", "cut_passes", "rows_skipped_l21_29", "rows_skipped_l30_39", "stager_layers_skipped",
                      "passes_ended_early", "fallback_resumes") + L20:
        d["psk_" + k] = int(g("psk", k))
    for k in PFC:
        d["pfctx_" + k] = int(g("dspark_pfctx", k))
    return d


def guard(arm, d):
    why = []
    if not d.get("has_psk") or not d.get("has_pfctx"):
        why.append("no /stats psk or dspark_pfctx object (not the pskl20 image?)")
    for k in PSKD2:
        if d.get("psk_" + k, 0) <= 0: why.append("design 2's %s did not grow (%+d)" % (k, d.get("psk_" + k, 0)))
    if d.get("psk_plan_mismatch", 0) != 0: why.append("plan_mismatch grew (%+d)" % d["psk_plan_mismatch"])
    l20 = arm in ("L", "X")
    pfc = arm in ("D", "X")
    if l20:
        for k in ("l20_cut_passes", "l20_rows_skipped"):
            if d.get("psk_" + k, 0) <= 0: why.append("PSK-L20 %s did not grow (%+d)" % (k, d.get("psk_" + k, 0)))
    else:
        for k in L20:
            if d.get("psk_" + k, 0) != 0: why.append("PSK-L20 %s moved in arm %s (%+d)" % (k, NAME.get(arm, arm), d["psk_" + k]))
    if pfc:
        for k in ("passes", "rows_skipped"):
            if d.get("pfctx_" + k, 0) <= 0: why.append("PSK-DS %s did not grow (%+d)" % (k, d.get("pfctx_" + k, 0)))
    else:
        for k in PFC:
            if d.get("pfctx_" + k, 0) != 0: why.append("PSK-DS %s moved in arm %s (%+d)" % (k, NAME.get(arm, arm), d["pfctx_" + k]))
    ev = ", ".join("%s %+d" % (k, v) for k, v in sorted(d.items()) if isinstance(v, int) and not isinstance(v, bool) and v and
                   (k.startswith("psk_") or k.startswith("pfctx_")))
    return ("OK" if not why else "FAIL " + "; ".join(why)) + " | arm %s deltas: %s" % (NAME.get(arm, arm), ev or "none")


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def steps_report(ncs, log, marks, warm):
    st = ncs.read_steps(log)
    fq, fc = getattr(ncs, "FRESH_Q", 256), getattr(ncs, "FRESH_CTX", 64)
    out = []
    for mk in marks:
        t0, t1 = mk["t0"] + warm, mk["t1"]
        win = [s for s in st if t0 <= s["t"] < t1]
        nc = [s for s in win if any(q >= fq and c <= fc for _, q, c in s["ents"])]
        mx = [s for s in win if s["ph"] == "MIXED" and s not in nc]
        dc = [s for s in win if s["ph"] == "decode"]
        w = lambda ss: [s.get("wall_ms") for s in ss if s.get("wall_ms") is not None]   # noqa: E731
        out.append({"name": mk.get("name"), "arm": mk.get("arm"), "steps": len(win), "newcomer": ncs.stat(w(nc)),
                    "newcomer_ntok": ncs.stat([s["n_tok"] for s in nc], 0), "mixed_other": ncs.stat(w(mx)),
                    "decode": ncs.stat(w(dc)), "prefill_only": sum(1 for s in win if s["ph"] == "prefill")})
    return out


def acc_check(rs):
    """rs: rows {arm, run, prompt, text, draft_tok, draft_acc} -> (verdict, per-row compares)."""
    def txt(p):
        try:
            with open(p) as f: return f.read()
        except OSError: return None
    ref = {}
    for r in rs:
        if r.get("arm") == "B" and r["prompt"] not in ref: ref[r["prompt"]] = r
    res, bad, bb = [], [], []
    for r in rs:
        b = ref.get(r["prompt"])
        if b is None or b is r: continue
        same_t = txt(r.get("text") or "") == txt(b.get("text") or "") and txt(b.get("text") or "") is not None
        same_d = (r.get("draft_tok"), r.get("draft_acc")) == (b.get("draft_tok"), b.get("draft_acc"))
        line = "%s run %s %s: text %s, drafted / accepted %s / %s vs B %s / %s" % (
            NAME.get(r["arm"], r["arm"]), r.get("run"), r["prompt"], "IDENTICAL" if same_t else "DIFFERENT", r.get("draft_tok"),
            r.get("draft_acc"), b.get("draft_tok"), b.get("draft_acc"))
        res.append(line)
        if not (same_t and same_d): (bb if r["arm"] == "B" else bad).append(line)
    drafted = sum(1 for r in rs if (r.get("draft_tok") or 0) > 0)
    if not rs or drafted == 0:
        v = "VIRHE no drafting in the deterministic runs (nothing compared)"
    elif bb:
        v = "INCONCLUSIVE: B against B differs (the pin is not deterministic here): " + "; ".join(bb)[:600]
    elif bad:
        v = "VIRHE proposals / acceptance differ: " + "; ".join(bad)[:900]
    else:
        v = "EQUAL: every run's answer and drafted / accepted counts equal the first B run's (%d compares, B vs B included)" % len(res)
    return v, res


def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)


def f1(x, n=1):
    return "-" if x is None else format(x, ".%df" % n)


def pc(a, b):
    return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"


def summary(o):
    lines = []
    # ---- the counter guard and the arms' pins
    try:
        guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
    except OSError:
        guards = []
    gbad = [g for g in guards if " FAIL " in (" " + g + " ") or g.startswith("FAIL") or "VIRHE" in g]
    lines.append("GUARD %s" % ((("VIRHE " + " || ".join(gbad))[:1500]) if gbad else ("OK (%d checks): " % len(guards) + " || ".join(guards))[:1500]))
    # ---- every cold run's counters (TTFT runs): the arm's lever moved, the other did not
    dp = rows(os.path.join(o, "deep.jsonl"))
    ld = rows(os.path.join(o, "load.jsonl"))
    mbr = rows(os.path.join(o, "mb.jsonl"))
    cbad, inert = [], []
    for r in dp + ld + mbr:
        a = r.get("arm")
        if a not in NAME: continue
        l20m, pfm = r.get("psk_l20_rows_skipped", 0), r.get("pfctx_rows_skipped", 0)
        pre = (r.get("prompt_tokens") or 0) - (r.get("cached") or 0)
        kind = r.get("kind")
        # must move: PSK-L20 where a prefilled prompt spans more than layer 20's cone (2592 rows); PSK-DS where a pass of
        # more than 300 rows ran (every chunk of such a prompt, and mb c8's ~1k-token newcomers)
        if a in ("L", "X"):
            if kind in ("ttft", "fixed", "hitcold", "guard", "load") and pre > 2592 and l20m <= 0:
                cbad.append("%s: PSK-L20 rows 0 (prefilled %d)" % (r.get("name"), pre))
            if kind == "mb" and l20m <= 0: inert.append(r.get("name"))
        if a in ("D", "X") and ((kind != "mb" and pre > 300) or kind == "mb") and pfm <= 0:
            cbad.append("%s: PSK-DS rows 0" % r.get("name"))
        if a in ("B", "D") and l20m != 0: cbad.append("%s: PSK-L20 rows %+d in arm %s" % (r.get("name"), l20m, NAME[a]))
        if a in ("B", "L") and pfm != 0: cbad.append("%s: PSK-DS rows %+d in arm %s" % (r.get("name"), pfm, NAME[a]))
        if r.get("psk_plan_mismatch", 0): cbad.append("%s: plan_mismatch %+d" % (r.get("name"), r["psk_plan_mismatch"]))
    lines.append("COUNTERS every run: %s%s" % (
        ("VIRHE " + "; ".join(cbad))[:1200] if cbad else "OK (%d runs: each arm's lever moved where it applies, the other did not)" % len(dp + ld + mbr),
        (" | PSK-L20 inert in the mb c8 blocks (no prompt past layer 20's 2592-row cone): %s" % ", ".join(inert)) if inert else ""))
    # ---- cold TTFT ABBA (fresh distinct texts)
    tt = [r for r in dp if r.get("kind") == "ttft"]
    for dep in sorted({r["depth"] for r in tt}, reverse=True):
        s = [r for r in tt if r["depth"] == dep]
        mB = ci([r.get("ttft") for r in s if r["arm"] == "B"])[0]
        cells = []
        for a in "BLDX":
            m, h, n = ci([r.get("ttft") for r in s if r["arm"] == a])
            if m is None: continue
            c = "%s %s%s s (n %d)" % (NAME[a], f1(m, 2), "" if h is None else " +- " + f1(h, 2), n)
            if a != "B": c += " %s vs B" % pc(m, mB)
            cells.append(c)
        pr = []
        for p, q in zip(s, s[1:]):
            if "B" in (p["arm"], q["arm"]) and p["arm"] != q["arm"] and p.get("ttft") and q.get("ttft"):
                b, x = (p, q) if p["arm"] == "B" else (q, p)
                pr.append("%s %+.1f" % (NAME[x["arm"]], 100 * (x["ttft"] / b["ttft"] - 1)))
        cold = all((r.get("cached") or 0) == 0 for r in s)
        stg = " | staged GB " + ", ".join("%s %s" % (NAME[a], f1(ci([r.get("staged_gb") for r in s if r["arm"] == a])[0], 0)) for a in "BLDX")
        l20r = " | L20 rows skipped / cone-empty passes / layers not staged a run: " + ", ".join(
            "%s %s / %s / %s" % (NAME[a], f1(ci([r.get("psk_l20_rows_skipped") for r in s if r["arm"] == a])[0], 0),
                                f1(ci([r.get("psk_l20_cone_empty") for r in s if r["arm"] == a])[0], 1),
                                f1(ci([r.get("psk_l20_layers_not_staged") for r in s if r["arm"] == a])[0], 1)) for a in "LX")
        pfr = " | pfctx rows skipped a run: " + ", ".join("%s %s" % (NAME[a], f1(ci([r.get("pfctx_rows_skipped") for r in s if r["arm"] == a])[0], 0)) for a in "DX")
        lines.append("TTFT cold %d (%s tok, fresh distinct texts): %s | adjacent pairs vs B %s%s%s%s%s" % (
            dep, s[0].get("prompt_tokens"), " | ".join(cells), "/".join(pr) or "-", "" if cold else " -- VIRHE a run was not cold", stg, l20r, pfr))
    # ---- mb c8 ABBA
    mb = rows(os.path.join(o, "mb.jsonl"))
    sp = {r.get("name"): r for r in rows(os.path.join(o, "steps.jsonl"))}
    if mb:
        mB = {k: ci([r.get(k) for r in mb if r["arm"] == "B"])[0] for k in ("agg_dec_tps", "decode_share")}
        cells = []
        for a in "BLDX":
            ra = [r for r in mb if r["arm"] == a]
            if not ra: continue
            m, h, n = ci([r.get("agg_dec_tps") for r in ra])
            ds = ci([r.get("decode_share") for r in ra])[0]
            nw = ci([((sp.get(r["name"]) or {}).get("newcomer") or {}).get("p50") for r in ra])[0]
            nm = ci([((sp.get(r["name"]) or {}).get("newcomer") or {}).get("mean") for r in ra])[0]
            nn = sum(((sp.get(r["name"]) or {}).get("newcomer") or {}).get("n", 0) for r in ra)
            dw = ci([((sp.get(r["name"]) or {}).get("decode") or {}).get("p50") for r in ra])[0]
            c = "%s agg %s%s tok/s%s, decode_share %s, newcomer step wall p50 %s / mean %s ms (%d steps), decode step p50 %s ms" % (
                NAME[a], f1(m), "" if h is None else " +- " + f1(h), "" if a == "B" else " (%s vs B)" % pc(m, mB["agg_dec_tps"]),
                f1(ds, 3), f1(nw), f1(nm), nn, f1(dw))
            cells.append(c)
        lines.append("MB c8 T 0 (dur 60 warm 15, %d blocks): %s" % (len(mb), " | ".join(cells)))
    # ---- the drafter check
    av = [l.strip() for l in open(os.path.join(o, "acc.txt"))] if os.path.exists(os.path.join(o, "acc.txt")) else []
    lines.append("DRAFTER (deterministic greedy NST 5 pin): %s" % (av[0] if av else "not run"))
    # ---- identity
    try:
        ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError:
        ident = []
    info = [l for l in ident if l.split(" ")[0].startswith(("info-", "ld-"))]
    strict = [l for l in ident if l not in info]
    bad = [l for l in strict if not ("IDENTICAL" in l and "differ" not in l)]
    lab = lambda ls: "; ".join(l.split(" ")[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "differ" not in l else "differs") for l in ls)  # noqa: E731
    lines.append("IDENTITY strict: %s || information (class (ii) at sched_bi_chunk 0 where the probe's chunks differ; B vs bz d0): %s" % (
        ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:1500] if strict else "none"),
        lab(info) or "-"))
    return lines


def selftest(ncs_path):
    ok, why = True, []
    ncs = load(ncs_path, "ncsteps_ro")
    import tempfile
    d = tempfile.mkdtemp(prefix="pl20st-")
    p = os.path.join(d, "e.log")
    with open(p, "w") as f:
        t = 1762900000.0
        lines = [(0.00, "step 10 decode n_seq=8 n_tok=48  [0] slot=0 q=6 ctx=5000"),
                 (0.04, "step 11 MIXED n_seq=8 n_tok=1090  [0] slot=0 q=6 ctx=5006 [7] slot=7 q=1048 ctx=0"),
                 (0.20, "step 12 decode n_seq=8 n_tok=48  [0] slot=0 q=6 ctx=5012"),
                 (0.24, "step 13 MIXED n_seq=8 n_tok=500  [0] slot=0 q=6 ctx=5018 [7] slot=7 q=458 ctx=4096"),
                 (0.40, "step 14 decode n_seq=8 n_tok=48  [0] slot=0 q=6 ctx=5024")]
        import time as _t
        for dt, l in lines:
            ts = _t.strftime("%Y-%m-%dT%H:%M:%S", _t.gmtime(t + dt)) + ".%06dZ" % int(round(((t + dt) % 1) * 1e6))
            f.write("%s I %s\n" % (ts, l))
    r = steps_report(ncs, p, [{"name": "x", "arm": "B", "t0": t - 1, "t1": t + 1}], 0)[0]
    if not (r["steps"] == 5 and r["newcomer"].get("n") == 1 and abs(r["newcomer"]["p50"] - 160.0) < 1.0 and r["mixed_other"].get("n") == 1
            and r["decode"].get("n") == 2):
        ok = False; why.append("steps %s" % json.dumps(r))
    A = {"checkpoints_written": 1, "psk": {k: 0 for k in PSKD2 + L20 + ("plan_mismatch",)}, "dspark_pfctx": {"mode": 0, "passes": 0, "rows_skipped": 0, "rows_run": 0}}
    Bx = json.loads(json.dumps(A))
    for k in PSKD2: Bx["psk"][k] = 3
    Bx["psk"]["l20_cut_passes"] = 2; Bx["psk"]["l20_rows_skipped"] = 900
    Bx["dspark_pfctx"].update(passes=4, rows_skipped=8000, rows_run=640)
    dx = sdelta(A, Bx)
    if not guard("X", dx).startswith("OK") or not guard("B", dx).startswith("FAIL") or not guard("L", dx).startswith("FAIL") \
            or not guard("D", dx).startswith("FAIL"):
        ok = False; why.append("guard X/B/L/D %s / %s / %s / %s" % (guard("X", dx)[:60], guard("B", dx)[:60], guard("L", dx)[:60], guard("D", dx)[:60]))
    B0 = json.loads(json.dumps(A))
    for k in PSKD2: B0["psk"][k] = 3
    if not guard("B", sdelta(A, B0)).startswith("OK") or not guard("D", sdelta(A, B0)).startswith("FAIL"):
        ok = False; why.append("guard B0")
    if not guard("B", sdelta(A, A)).startswith("FAIL"):
        ok = False; why.append("guard: design 2 not moving must fail")
    t1, t2 = os.path.join(d, "a.txt"), os.path.join(d, "b.txt")
    open(t1, "w").write("same"); open(t2, "w").write("same")
    rs = [{"arm": "B", "run": 1, "prompt": "p", "text": t1, "draft_tok": 10, "draft_acc": 7},
          {"arm": "D", "run": 1, "prompt": "p", "text": t2, "draft_tok": 10, "draft_acc": 7},
          {"arm": "B", "run": 2, "prompt": "p", "text": t2, "draft_tok": 10, "draft_acc": 7}]
    if not acc_check(rs)[0].startswith("EQUAL"): ok = False; why.append("acc equal")
    rs[1]["draft_acc"] = 6
    if not acc_check(rs)[0].startswith("VIRHE"): ok = False; why.append("acc differ")
    rs[1]["draft_acc"] = 7; rs[2]["draft_acc"] = 5
    if not acc_check(rs)[0].startswith("INCONCLUSIVE"): ok = False; why.append("acc B vs B")
    import shutil
    shutil.rmtree(d, ignore_errors=True)   # a DRYRUN writes nothing that stays
    print("SELFTEST OK (step windows / newcomer walls, guard rules for B / L / D / LD, sdelta, the drafter compare)" if ok
          else "SELFTEST FAIL " + "; ".join(why))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sdelta", "guard", "steps", "acc", "texts", "summary", "selftest"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ncsteps", default=NCS_DEFAULT)
    ap.add_argument("--log")
    ap.add_argument("--marks")
    ap.add_argument("--warm", type=float, default=15.0)
    ap.add_argument("--out")
    ap.add_argument("--pskld")
    ap.add_argument("--tests")
    ap.add_argument("--doffs", default="")
    a = ap.parse_args()
    if a.mode == "sdelta":
        print(json.dumps(sdelta(rd_json(a.args[0]), rd_json(a.args[1]))))
        return 0
    if a.mode == "guard":
        print(guard(a.args[0], json.loads(a.args[1] or "{}")))
        return 0
    if a.mode == "steps":
        ncs = load(a.ncsteps, "ncsteps_ro")
        for r in steps_report(ncs, a.log, rows(a.marks), a.warm): print("PL20STEPS " + json.dumps(r))
        return 0
    if a.mode == "acc":
        v, res = acc_check(rows(a.args[0]))
        for l in res: print("ACC " + l)
        print(v)
        return 0
    if a.mode == "texts":
        pl, t = load(a.pskld, "pskld_ro"), load(a.tests, "ds41_tests_ro")
        offs = [int(x) for x in a.doffs.split(",") if x.strip()]
        firsts = {}
        for o in offs:
            pl.DEEP_OFF = o
            b = pl.deep_payload(t, 2000, 16)["messages"][0]["content"][:3000]
            firsts.setdefault(b, []).append(o)
        dup = [v for v in firsts.values() if len(v) > 1]
        print(("SELFTEST OK (%d text offsets, every first block distinct)" % len(offs)) if not dup else "SELFTEST FAIL shared first blocks: %s" % dup)
        return 0 if not dup else 1
    if a.mode == "summary":
        for l in summary(a.out): print(l)
        return 0
    return selftest(a.ncsteps)


if __name__ == "__main__":
    sys.exit(main())
