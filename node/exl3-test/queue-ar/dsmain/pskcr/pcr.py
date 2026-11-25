#!/usr/bin/env python3
"""pcr.py (12.11.2026, queue-ar/dsmain/pskcr, ds-main-port PSK-CR builder) -- the requests and readings of dsm-pskcr.sh (item
57-p4zyzzzzz-ds-r00zzz-cz-scr): lever PSK-CR (fn4 ds41_psk_cr, exact cone replay on a prefix hit; ds41_psk 4 = tail-only
anchors) on the ds41-pskcr image. Standard library only. Reuses pskld.py (the pskl20 copy, md5 at copy time in the item log)
READ ONLY via importlib: its run() does the request, the TTFT / usage readings and, with --load, the c4 decode streams beside
the probe; this file only hands it another payload (as p2bq.py hands it another DEEP_OFF).

  pcr.py run --pskld P --tests T --url URL --spec SPEC --ntok N --out J --text TXT [--load --mb MB]
      SPEC = deep:DEPTH:OFF          deepq.py's request (pskld.deep_payload) on text offset OFF (p2bq.py's construction: depths
                                     of one offset share their body, so a shorter depth after a longer one is a prefix hit);
             head:H:HOFF:TT:TOFF     a SHARED HEAD: "# deepq fixed prompt v1\\n" + H tokens of text HOFF, then TT tokens of text
                                     TOFF (each prompt of a head its own TOFF), then deepq's task -- prompts with one HOFF share
                                     exactly the header + head;
             turn:DEPTH:OFF:ANSFILE  a FOLLOW-UP TURN: deep:DEPTH:OFF's user message, the assistant's answer from ANSFILE (its
                                     content, the reasoning part dropped), then FOLLOW -- a hit on the earlier turn's prompt.
      Prints "LONGID {json}" (pskld.run's: status, prompt_tokens, cached, ttft, tokens, finish, chars, seconds [+ load readings]).
  pcr.py sdelta A.json B.json      /stats B - A: snapshots, staged GB, design 2's / PSK-L20's / PSK-DS's / PSK-CR's counters
  pcr.py guard ARM DELTAJSON [L20 PF]  the counter guard (a lever that silently does nothing = VIRHE): in every arm design 2's
                                     latched / planned_steps grow and plan_mismatch does not; C / T: cr_hits, cr_rows_replayed,
                                     cr_passes, cr_layer_rows_run, cr_layer_rows_skipped, cr_chains_skipped grow, no cr fallback,
                                     no cr_mixed_passes (alone), mode cr 1; T also tail_only_latched grows, tail_only_refused does
                                     not; B: every cr / tail-only counter 0, mode cr 0, and its hit fell back (fallback_resumes);
                                     L20 / PF 1: l20_cut_passes / pfctx passes grow in every arm -> "OK ..." or "FAIL ..."
  pcr.py want ARM DEPTH [K]        the cached count a hit of DEPTH must report: C / T the grid point under it (the cone replay's
                                     end, 2048 + k x 4096), B K's anchor below it (0 when none)
  pcr.py expect ARM PT SNAPS [K]   from the hit's own prompt_tokens PT (its match lies in [PT - 300, PT - 40]: deepq's task, the
                                     draft refill) and the snapshots its text holds (SNAPS, comma list): "WANT REPLAY" -- C / T the
                                     grid point under the match unless a snapshot sits at or above it (then that snapshot, no
                                     replay); B the deepest snapshot under the match (0 none); "" when a grid point or a snapshot
                                     lies inside the match's range (not asserted)
  pcr.py snaps ARM CACHED PT [K] [CKPT_AT]  the snapshots a request that resumed at CACHED and prefilled PT tokens wrote: B / C
                                     every K-th grid point past CACHED, T none; the tail snapshot (the grid point under PT - 1, past
                                     2048) where past CACHED; checkpoint_at where given
  pcr.py texts --pskld P --tests T --doffs a,b,...  offline: every text offset used starts with a distinct block
  pcr.py summary --out DIR [--k K]  the item's final lines (RESULT xcr summary)
  pcr.py selftest --pskld P --tests T   offline: the three payloads (deep = pskld's byte for byte; a head's prompts share
                                     exactly header + head and differ after it; a turn carries the deep message first), sdelta,
                                     the guard rules, want()
"""
import argparse, importlib.util, json, math, os, statistics, sys

sys.dont_write_bytecode = True
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
NAME = {"B": "B", "C": "C", "T": "T"}
HDR = "# deepq fixed prompt v1\n"
FOLLOW = ("\n\nThank you. Now list the three most important concrete improvements you would make to this code, each in one "
          "short paragraph.")
PSKD2 = ("latched", "planned_steps")
CRK = ("cr_hits", "cr_rows_replayed", "cr_resume_tokens", "cr_fallbacks", "cr_fallback_unplanned", "cr_fallback_geometry",
       "cr_fallback_adopt", "cr_not_needed", "tail_only_latched", "tail_only_refused", "cr_passes", "cr_layer_rows_run",
       "cr_layer_rows_skipped", "cr_chains_skipped", "cr_mixed_passes")
CR_MUST = ("cr_hits", "cr_rows_replayed", "cr_passes", "cr_layer_rows_run", "cr_layer_rows_skipped", "cr_chains_skipped")
L20 = ("l20_cut_passes", "l20_rows_skipped")
PFC = ("passes", "rows_skipped", "rows_run")
GO, GS = 2048, 4096   # the production grid: --checkpoint-interval 2048, --prefill-span 2


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


# ------------------------------------------------------------------ payloads
def body(pl, t, toks, off):
    return t.text_chars(int(toks * pl.DEEP_CPT), offset=off)


def answer_of(path):
    try:
        with open(path) as f: s = f.read()
    except OSError:
        return None
    return s.split("\n[reasoning]\n")[0]


def payload(pl, t, spec, ntok):
    f = spec.split(":")
    if f[0] == "deep" and len(f) == 3:
        pl.DEEP_OFF = int(f[2])
        return pl.deep_payload(t, int(f[1]), ntok)
    if f[0] == "head" and len(f) == 5:
        H, ho, TT, to = int(f[1]), int(f[2]), int(f[3]), int(f[4])
        msg = HDR + body(pl, t, H, ho) + "\n\n" + body(pl, t, TT, to) + pl.DEEP_TASK
        return {"messages": [{"role": "user", "content": msg}], "max_tokens": ntok, "temperature": 0.0,
                "reasoning_effort": "none", "ignore_eos": True}
    if f[0] == "turn" and len(f) == 4:
        pl.DEEP_OFF = int(f[2])
        d = pl.deep_payload(t, int(f[1]), ntok)
        a = answer_of(f[3])
        if a is None: raise SystemExit("turn: no answer file %s" % f[3])
        d["messages"] = list(d["messages"]) + [{"role": "assistant", "content": a}, {"role": "user", "content": FOLLOW.strip()}]
        return d
    raise SystemExit("bad --spec %r (deep:DEPTH:OFF | head:H:HOFF:TT:TOFF | turn:DEPTH:OFF:ANSFILE)" % spec)


def spec_depth(spec):
    f = spec.split(":")
    try:
        return int(f[1]) + (int(f[3]) if f[0] == "head" else 0)
    except (IndexError, ValueError):
        return 0


# ------------------------------------------------------------------ readings
def sdelta(A, B):
    g = lambda *p: num(B, *p) - num(A, *p)   # noqa: E731
    d = {"ckpts": int(g("checkpoints_written")), "staged_gb": round(g("link", "staged_bytes") / 1e9, 2),
         "staged_gb_r0": round(g("experts", "staged_bytes") / 1e9, 2), "stage_layers_r0": int(g("experts", "stage_layers")),
         "has_psk": "psk" in B, "has_cr": isinstance(B.get("psk"), dict) and "cr_hits" in B["psk"],
         "cr_mode": num(B, "psk", "cr") if isinstance(B.get("psk"), dict) else None,
         "psk_mode": num(B, "psk", "mode") if isinstance(B.get("psk"), dict) else None}
    for k in PSKD2 + ("ckpt_dropped", "layers_skipped", "plan_mismatch", "cut_passes", "rows_skipped_l21_29",
                      "rows_skipped_l30_39", "stager_layers_skipped", "passes_ended_early", "fallback_resumes",
                      "fallback_tokens") + L20 + CRK:
        d["psk_" + k] = int(g("psk", k))
    for k in PFC:
        d["pfctx_" + k] = int(g("dspark_pfctx", k))
    return d


def guard(arm, d, l20=0, pf=0):
    why = []
    if not d.get("has_psk") or not d.get("has_cr"):
        why.append("no /stats psk cr_* counters (not the pskcr image?)")
    for k in PSKD2:
        if d.get("psk_" + k, 0) <= 0: why.append("design 2's %s did not grow (%+d)" % (k, d.get("psk_" + k, 0)))
    if d.get("psk_plan_mismatch", 0) != 0: why.append("plan_mismatch grew (%+d)" % d["psk_plan_mismatch"])
    if arm in ("C", "T"):
        for k in CR_MUST:
            if d.get("psk_" + k, 0) <= 0: why.append("PSK-CR %s did not grow (%+d)" % (k, d.get("psk_" + k, 0)))
        for k in ("cr_fallbacks", "cr_mixed_passes"):
            if d.get("psk_" + k, 0) != 0: why.append("PSK-CR %s moved (%+d)" % (k, d["psk_" + k]))
        if d.get("cr_mode") != 1: why.append("/stats psk cr is %s, not 1" % d.get("cr_mode"))
        if arm == "T":
            if d.get("psk_tail_only_latched", 0) <= 0: why.append("tail_only_latched did not grow")
            if d.get("psk_tail_only_refused", 0) != 0: why.append("tail_only_refused moved")
            if d.get("psk_mode") != 4: why.append("/stats psk mode is %s, not 4" % d.get("psk_mode"))
        else:
            if d.get("psk_tail_only_latched", 0) != 0: why.append("tail_only_latched moved in C")
    else:
        for k in CRK:
            if d.get("psk_" + k, 0) != 0: why.append("PSK-CR %s moved in arm B (%+d)" % (k, d["psk_" + k]))
        if d.get("cr_mode") not in (0, None): why.append("/stats psk cr is %s in arm B" % d.get("cr_mode"))
        if d.get("psk_fallback_resumes", 0) <= 0: why.append("B's hit did not fall back to its anchor (fallback_resumes %+d)" %
                                                             d.get("psk_fallback_resumes", 0))
    if l20:
        for k in L20:
            if d.get("psk_" + k, 0) <= 0: why.append("PSK-L20 %s did not grow (XCR_L20 1)" % k)
    if pf and d.get("pfctx_passes", 0) <= 0: why.append("PSK-DS passes did not grow (XCR_PF 1)")
    ev = ", ".join("%s %+d" % (k, v) for k, v in sorted(d.items()) if isinstance(v, int) and not isinstance(v, bool) and v and
                   (k.startswith("psk_") or k.startswith("pfctx_")))
    return ("OK" if not why else "FAIL " + "; ".join(why)) + " | arm %s deltas: %s" % (NAME.get(arm, arm), ev or "none")


def gfloor(p):
    return 0 if p < GO else GO + (p - GO) // GS * GS


def want(arm, depth, k=4):
    """the cached count a hit whose match lies just under DEPTH reports (the item's depths keep >= 400 tokens from a grid
    point): C / T the grid point under it (a K-th point is an anchor: the same position), B K's anchor below it"""
    at = gfloor(depth)
    if arm in ("C", "T"): return at
    anch = [GO + GS * i for i in range(0, 300) if (i + 1) % k == 0 and GO + GS * i <= depth]
    return anch[-1] if anch else 0


def anchors(k, lo, hi):
    """K's grid anchors q with lo < q < hi"""
    return [GO + GS * i for i in range(0, 400) if (i + 1) % k == 0 and lo < GO + GS * i < hi]


def snaps(arm, cached, pt, k=4, ckpt_at=0):
    out = [] if arm == "T" else anchors(k, cached, pt)
    tail = gfloor(pt - 1)
    if tail > GO and tail > cached: out.append(tail)
    if ckpt_at and cached < ckpt_at < pt: out.append(ckpt_at)
    return sorted(set(out))


def expect(arm, pt, sn, k=4):
    lo, hi = pt - 300, pt - 40
    if gfloor(lo) != gfloor(hi) or any(lo <= x <= hi for x in sn): return None, None
    m = hi
    below = [x for x in sn if x <= m]
    best = max(below) if below else 0
    if arm in ("C", "T"):
        at = gfloor(m)
        return (best, False) if best >= at else (at, at > 0)
    return best, False


def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)


def f1(x, n=1):
    return "-" if x is None else format(x, ".%df" % n)


def pc(a, b):
    return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"


def summary(o, k=4):
    lines = []
    try:
        guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
    except OSError:
        guards = []
    gbad = [g for g in guards if " FAIL " in (" " + g + " ") or g.startswith("FAIL") or "VIRHE" in g]
    lines.append("GUARD %s" % ((("VIRHE " + " || ".join(gbad))[:1500]) if gbad else ("OK (%d checks): " % len(guards) + " || ".join(guards))[:1500]))
    dp = rows(os.path.join(o, "deep.jsonl"))
    # every run's counters: C / T hits replayed, B never; no plan mismatch, no cr fallback anywhere
    cbad = []
    for r in dp:
        a, kind = r.get("arm"), r.get("kind")
        if a not in NAME: continue
        if r.get("psk_plan_mismatch", 0): cbad.append("%s: plan_mismatch %+d" % (r.get("name"), r["psk_plan_mismatch"]))
        if a == "B" and any(r.get("psk_" + x, 0) for x in CRK): cbad.append("%s: a PSK-CR counter moved in B" % r.get("name"))
        if a in ("C", "T") and r.get("psk_cr_fallbacks", 0): cbad.append("%s: cr_fallbacks %+d" % (r.get("name"), r["psk_cr_fallbacks"]))
        if a in ("C", "T") and kind in ("hit", "head", "turn", "loadhit") and r.get("want_replay") and r.get("psk_cr_hits", 0) <= 0:
            cbad.append("%s: no cone replay on a hit that needed one" % r.get("name"))
    lines.append("COUNTERS every run: %s" % (("VIRHE " + "; ".join(cbad))[:1200] if cbad else "OK (%d runs)" % len(dp)))
    # hits: cached = want, TTFT per depth per arm, the replay rows
    hits = [r for r in dp if r.get("kind") in ("hit", "turn", "head", "loadhit")]
    hbad = [r for r in hits if r.get("want") is not None and r.get("cached") != r.get("want")]
    lines.append("HIT RESUMES: %s" % (("VIRHE cached != expected: " + "; ".join("%s cached %s want %s" % (r["name"], r.get("cached"), r.get("want")) for r in hbad))[:1200]
                                     if hbad else "every hit resumed where expected (%d: C / T at the grid point under the match by a cone replay, B at K%d's anchor below)" % (len(hits), k)))
    groups = {}
    for r in hits:
        groups.setdefault((r.get("kind"), r.get("label") or str(r.get("depth"))), []).append(r)
    for (kind, lab), s in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        cells = []
        mB = ci([r.get("ttft") for r in s if r["arm"] == "B"])[0]
        for a in "BCT":
            ra = [r for r in s if r["arm"] == a]
            if not ra: continue
            m, h, n = ci([r.get("ttft") for r in ra])
            c = "%s TTFT %s%s s (n %d, cached %s, prompt %s, replayed %s tok)" % (
                a, f1(m, 2), "" if h is None else " +- " + f1(h, 2), n, "/".join(str(r.get("cached")) for r in ra),
                ra[0].get("prompt_tokens"), "/".join(str(r.get("psk_cr_rows_replayed", 0)) for r in ra))
            if a != "B" and mB: c += " %s vs B (%+.2f s)" % (pc(m, mB), m - mB)
            cells.append(c)
        lines.append("HIT %s %s: %s" % (kind, lab, " | ".join(cells)))
    # cold TTFT ABBA (T vs B)
    tt = [r for r in dp if r.get("kind") == "ttft"]
    for dep in sorted({r["depth"] for r in tt}, reverse=True):
        s = [r for r in tt if r["depth"] == dep]
        mB = ci([r.get("ttft") for r in s if r["arm"] == "B"])[0]
        cells = []
        for a in "BCT":
            m, h, n = ci([r.get("ttft") for r in s if r["arm"] == a])
            if m is None: continue
            c = "%s %s%s s (n %d)" % (a, f1(m, 2), "" if h is None else " +- " + f1(h, 2), n)
            if a != "B": c += " %s vs B" % pc(m, mB)
            cells.append(c)
        pr = []
        for p, q in zip(s, s[1:]):
            if "B" in (p["arm"], q["arm"]) and p["arm"] != q["arm"] and p.get("ttft") and q.get("ttft"):
                b, x = (p, q) if p["arm"] == "B" else (q, p)
                pr.append("%s %+.1f" % (x["arm"], 100 * (x["ttft"] / b["ttft"] - 1)))
        cold = all((r.get("cached") or 0) == 0 for r in s)
        stg = " | snapshots a run " + ", ".join("%s %s" % (a, f1(ci([r.get("ckpts") for r in s if r["arm"] == a])[0], 1)) for a in "BT") + \
              " | staged GB " + ", ".join("%s %s" % (a, f1(ci([r.get("staged_gb") for r in s if r["arm"] == a])[0], 0)) for a in "BT")
        lines.append("TTFT cold %d (%s tok, fresh distinct texts): %s | adjacent pairs vs B %s%s%s" % (
            dep, s[0].get("prompt_tokens"), " | ".join(cells), "/".join(pr) or "-", "" if cold else " -- VIRHE a run was not cold", stg))
    try:
        ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError:
        ident = []
    info = [l for l in ident if l.split(" ")[0].startswith(("info-", "ld-"))]
    strict = [l for l in ident if l not in info]
    bad = [l for l in strict if not ("IDENTICAL" in l and "differ" not in l)]
    lab = lambda ls: "; ".join(l.split(" ")[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "differ" not in l else "differs") for l in ls)  # noqa: E731
    lines.append("IDENTITY strict: %s || information (class (ii) under load where the probe's chunks are not the grid's): %s" % (
        ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:1500] if strict else "none"),
        lab(info) or "-"))
    return lines


# ------------------------------------------------------------------ selftest
def selftest(a):
    ok, why = True, []
    pl, t = load(a.pskld, "pskld_ro"), load(a.tests, "ds41_tests_ro")
    # deep = pskld's own, byte for byte
    pl.DEEP_OFF = 300
    ref = json.dumps(pl.deep_payload(t, 9000, 64), sort_keys=True)
    if json.dumps(payload(pl, t, "deep:9000:300", 64), sort_keys=True) != ref: ok = False; why.append("deep != pskld's")
    # a head's prompts share exactly header + head
    h1 = payload(pl, t, "head:6000:500:1500:510", 64)["messages"][0]["content"]
    h2 = payload(pl, t, "head:6000:500:1500:511", 64)["messages"][0]["content"]
    hd = HDR + body(pl, t, 6000, 500) + "\n\n"
    cp = os.path.commonprefix([h1, h2])
    if not (h1.startswith(hd) and h2.startswith(hd) and len(cp) < len(hd) + 400 and h1 != h2 and h1.endswith(pl.DEEP_TASK)):
        ok = False; why.append("head prompts (common %d, head %d)" % (len(cp), len(hd)))
    # a turn: the deep message first, then the answer and the follow-up
    import tempfile
    d = tempfile.mkdtemp(prefix="pcrst-")
    af = os.path.join(d, "a.txt")
    with open(af, "w") as f: f.write("ANSWER\n[reasoning]\nTHOUGHTS")
    tm = payload(pl, t, "turn:9000:300:" + af, 64)["messages"]
    pl.DEEP_OFF = 300
    if not (len(tm) == 3 and tm[0] == pl.deep_payload(t, 9000, 64)["messages"][0] and tm[1] == {"role": "assistant", "content": "ANSWER"}
            and tm[2]["role"] == "user"):
        ok = False; why.append("turn payload")
    import shutil
    shutil.rmtree(d, ignore_errors=True)   # a DRYRUN writes nothing that stays
    # sdelta + guard rules
    A = {"checkpoints_written": 1, "psk": dict({k: 0 for k in PSKD2 + CRK + L20 + ("plan_mismatch", "fallback_resumes")}, cr=0, mode=2),
         "dspark_pfctx": {"mode": 1, "passes": 0, "rows_skipped": 0, "rows_run": 0}}
    C = json.loads(json.dumps(A))
    for k in PSKD2 + CR_MUST + L20: C["psk"][k] = 3
    C["psk"]["cr"] = 1
    C["dspark_pfctx"]["passes"] = 2
    dc = sdelta(A, C)
    T = json.loads(json.dumps(C)); T["psk"]["tail_only_latched"] = 2; T["psk"]["mode"] = 4
    dt = sdelta(A, T)
    Bm = json.loads(json.dumps(A))
    for k in PSKD2 + L20: Bm["psk"][k] = 3
    Bm["psk"]["fallback_resumes"] = 1
    Bm["dspark_pfctx"]["passes"] = 2
    db = sdelta(A, Bm)
    checks = [(guard("C", dc, 1, 1), "OK"), (guard("T", dt, 1, 1), "OK"), (guard("B", db, 1, 1), "OK"),
              (guard("T", dc, 1, 1), "FAIL"), (guard("B", dc, 1, 1), "FAIL"), (guard("C", db, 1, 1), "FAIL"),
              (guard("C", dt, 1, 1), "FAIL"), (guard("B", sdelta(A, A)), "FAIL")]
    for i, (g, w) in enumerate(checks):
        if not g.startswith(w): ok = False; why.append("guard %d: %s" % (i, g[:120]))
    Bx = json.loads(json.dumps(Bm)); Bx["psk"]["cr_hits"] = 1
    if not guard("B", sdelta(A, Bx)).startswith("FAIL"): ok = False; why.append("guard: a cr counter in B must fail")
    # want(): the grid point under the hit (C / T), K4's anchor below (B)
    w = [want("C", 40000), want("T", 20000), want("C", 26000), want("C", 5000), want("C", 46000), want("B", 40000),
         want("B", 20000), want("B", 5000), want("B", 46000), want("C", 12000), want("B", 12000)]
    if w != [38912, 18432, 22528, 2048, 43008, 30720, 14336, 0, 30720, 10240, 0]: ok = False; why.append("want %s" % w)
    # expect / snaps on the hit block's sequence: cold 64k under B (anchors 14336 30720 47104, tail 63488), then hits
    sb = snaps("B", 0, 64040, 4)
    if sb != [14336, 30720, 47104, 63488]: ok = False; why.append("snaps B %s" % sb)
    if snaps("T", 0, 64040, 4) != [63488] or snaps("C", 38912, 41030, 4) != []: ok = False; why.append("snaps T / C")
    if snaps("B", 30720, 41030, 4) != [38912] or snaps("C", 22528, 28100, 4, 24600) != [24600, 26624]: ok = False; why.append("snaps hit")
    ex = [expect("C", 41030, sb), expect("B", 41030, sb), expect("T", 41030, [63488]), expect("B", 20550, sb + [38912]),
          expect("B", 24640, sb + [38912, 18432]), expect("C", 24640, sb), expect("C", 4150, sb), expect("B", 64040, sb),
          expect("C", 64040, sb), expect("C", 38990, sb), expect("C", 22600, sb + [22528])]
    want_ex = [(38912, True), (30720, False), (38912, True), (14336, False), (18432, False), (22528, True), (2048, True),
               (63488, False), (63488, False), (None, None), (None, None)]
    if ex != want_ex: ok = False; why.append("expect %s" % ex)
    if spec_depth("head:25000:1:3000:2") != 28000 or spec_depth("deep:9000:3") != 9000: ok = False; why.append("spec_depth")
    print("SELFTEST OK (deep = pskld's payload, a head's prompts share header + head only, a follow-up turn, sdelta, the guard "
          "rules for B / C / T, want())" if ok else "SELFTEST FAIL " + "; ".join(why))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "sdelta", "guard", "want", "expect", "snaps", "texts", "summary", "selftest"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--pskld")
    ap.add_argument("--tests")
    ap.add_argument("--url", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--text")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--mb")
    ap.add_argument("--doffs", default="")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "sdelta":
        print(json.dumps(sdelta(rd_json(a.args[0]), rd_json(a.args[1]))))
        return 0
    if a.mode == "guard":
        x = a.args + ["0", "0"]
        print(guard(x[0], json.loads(x[1] or "{}"), int(x[2]), int(x[3])))
        return 0
    if a.mode == "want":
        print(want(a.args[0], int(a.args[1]), int(a.args[2]) if len(a.args) > 2 else a.k))
        return 0
    if a.mode == "summary":
        for l in summary(a.out, a.k): print(l)
        return 0
    if a.mode == "expect":
        sn = [int(x) for x in (a.args[2] if len(a.args) > 2 else "").split(",") if x.strip()]
        w, r = expect(a.args[0], int(a.args[1]), sn, int(a.args[3]) if len(a.args) > 3 else a.k)
        print("" if w is None else "%d %d" % (w, 1 if r else 0))
        return 0
    if a.mode == "snaps":
        x = a.args + ["4", "0"]
        print(",".join(str(v) for v in snaps(x[0], int(x[1]), int(x[2]), int(x[3]), int(x[4]))))
        return 0
    if a.mode == "texts":
        pl, t = load(a.pskld, "pskld_ro"), load(a.tests, "ds41_tests_ro")
        offs = [int(x) for x in a.doffs.split(",") if x.strip()]
        firsts = {}
        for o in offs:
            firsts.setdefault(body(pl, t, 800, o)[:3000], []).append(o)
        dup = [v for v in firsts.values() if len(v) > 1]
        print(("SELFTEST OK (%d text offsets, every first block distinct)" % len(offs)) if not dup else "SELFTEST FAIL shared first blocks: %s" % dup)
        return 0 if not dup else 1
    if a.mode == "selftest":
        return selftest(a)
    pl = load(a.pskld, "pskld_ro")
    t = load(a.tests, "ds41_tests_ro")
    p = payload(pl, t, a.spec, a.ntok)
    pl.deep_payload = lambda _t, _d, _n: p   # pskld.run asks for its deep payload: this one
    ns = argparse.Namespace(url=a.url, tests=a.tests, mb=a.mb, which="deep", out=a.out, text=a.text, load=bool(a.load), conc=4,
                            warm=10, offset=0.5, pf_tok=0, ctx=1000, gen=512, depth=spec_depth(a.spec), ntok=a.ntok)
    return pl.run(ns)


if __name__ == "__main__":
    sys.exit(main())
