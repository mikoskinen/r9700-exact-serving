#!/usr/bin/env python3
"""cap48sum.py (12.11.2026, queue-ar/dsmain/cap48, ds-main-port cap48 builder) -- the parser of dsm-cap48.sh (item
57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48, tag xc48). Standard library only; reads, never writes beside --json / stdout.

  capd   --fn0 A.json --fn1 B.json
         lever CAP48's counters between two GET /admin/fn4 snapshots ("cap": in_force, steps_capped, drafts_trimmed,
         ahead_capped, steps_over, steps_gt48, decode_steps as deltas; max_rows the gauge at B). One line.
  steps  --log ENGINE.log.gz --win NAME:T0:T1 [...] [--json OUT]
         the step log (RADIANCE_LOG_STEPS=1, `docker logs -t` receipt times) a window: steps, decode steps, decode steps
         with > 48 rows (a decode step's n_tok IS its decode rows -- never the line's first 8 entries, kv4sum's mistake),
         the most / p50, MIXED steps (their decode rows are NOT summable from the line: the fn4 counter reads them), prefill
         steps; deepseek41's 'lever BI stats' > 48-row counter's delta (granular: 512 segmented steps). Last line STEPLINE.
  refs   --log RUNLOG    the cross-start references from runs/ds41-exl3.log: kv4 n16 c16 / c8 ("RESULT xkv4 speed n16:") and
         stack5 S5T c8 ("RESULT xs5 ARMS T 0", the last). "REFS OK ..." or "REFS MISSING ...".
  report --dir O --tag TAG --phases 'NAME:ARM:KIND ...' --cap N --log RUNLOG
         "RESULT xc48 ..." lines: identity (strict B48 / S, the cap-off control's counts, information), the steps and the
         counters a window with their guards, speed vs the references (CROSS-START, labelled), freshness, summary.
  selftest
"""
import argparse, datetime, gzip, json, os, re, statistics, sys

CAPK = ("steps_capped", "drafts_trimmed", "ahead_capped", "steps_over", "steps_gt48", "decode_steps")
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
BIST = re.compile(r"lever BI stats: (\d+) segmented steps, .*?(\d+) steps with > 48 decode rows; ds41_bi (\d+)")
TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z ")
ON_ARMS, OFF_ARMS = "SDZ", "dz"
LOADED = ("gs0", "lid0", "gs1", "lid1")
ALONE = ("alone0", "set1")


def jload(p):
    try:
        with open(p) as f: return json.load(f)
    except (OSError, ValueError): return None


def capc(fn):
    c = (fn or {}).get("cap") if isinstance(fn, dict) else None
    return c if isinstance(c, dict) else None


def capd(a, b):
    """the deltas between two GET /admin/fn4 snapshots' cap blocks; None when either has none"""
    ca, cb = capc(a), capc(b)
    if ca is None or cb is None: return None
    d = {k: (cb.get(k) or 0) - (ca.get(k) or 0) for k in CAPK}
    d.update(in_force=cb.get("in_force"), max_rows=cb.get("max_rows"))
    return d


def capline(d):
    if d is None: return "no cap counters (GET /admin/fn4 has no \"cap\")"
    return "in force %s: steps_capped +%d, drafts_trimmed +%d, ahead_capped +%d, steps_over +%d, steps_gt48 +%d, decode_steps +%d, max_rows %s" % (
        d["in_force"], d["steps_capped"], d["drafts_trimmed"], d["ahead_capped"], d["steps_over"], d["steps_gt48"], d["decode_steps"], d["max_rows"])


def tsec(line):
    m = TS.match(line)
    if not m: return None
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    return t + (float("0" + m.group(2)) if m.group(2) else 0.0)


def read_log(path):
    steps, bis = [], []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if "step " not in line and "lever BI stats" not in line: continue
            t = tsec(line)
            if t is None: continue
            m = STEP.search(line)
            if m:
                steps.append((t, int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))))
                continue
            b = BIST.search(line)
            if b: bis.append((t, int(b.group(1)), int(b.group(2)), int(b.group(3))))
    return steps, bis


def windows(steps, bis, wins):
    out = {}
    for nm, t0, t1 in wins:
        W = [s for s in steps if t0 <= s[0] < t1]
        dec = [s[4] for s in W if s[2] == "decode"]
        mx = [s for s in W if s[2] == "MIXED"]
        b0 = [b for b in bis if b[0] <= t0]
        b1 = [b for b in bis if b[0] <= t1]
        out[nm] = dict(steps=len(W), decode=len(dec), decode_gt48=sum(1 for x in dec if x > 48), decode_max=max(dec) if dec else 0,
                       decode_p50=int(statistics.median(dec)) if dec else 0, mixed=len(mx), mixed_nseq_max=max((s[3] for s in mx), default=0),
                       prefill=sum(1 for s in W if s[2] == "prefill"),
                       bistats_gt48_delta=(b1[-1][2] - (b0[-1][2] if b0 else 0)) if b1 else None, bistats_lines=len(bis))
    return out


def cmd_capd(a):
    print("CAPD " + capline(capd(jload(a.fn0), jload(a.fn1))))
    return 0


def cmd_steps(a):
    try:
        steps, bis = read_log(a.log)
    except OSError as e:
        print("STEPLINE VIRHE no engine log (%s)" % e); return 2
    wins = []
    for w in a.win or []:
        nm, t0, t1 = w.rsplit(":", 2)
        wins.append((nm, float(t0), float(t1)))
    out = windows(steps, bis, wins)
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f)
    print("STEPS " + json.dumps(out))
    print("STEPLINE %d step lines in the log | " % len(steps) + " | ".join(
        "%s: %d steps, decode %d (%d with > 48 rows = n_tok, most %d, p50 %d), MIXED %d (up to %d sequences; decode rows from the counter), "
        "prefill %d; BI stats > 48 +%s" % (k, v["steps"], v["decode"], v["decode_gt48"], v["decode_max"], v["decode_p50"], v["mixed"],
                                          v["mixed_nseq_max"], v["prefill"], v["bistats_gt48_delta"]) for k, v in out.items()))
    return 0


def refs(log):
    r = {}
    try:
        lines = open(log, errors="replace").read().splitlines()
    except OSError:
        return r
    for l in lines:
        if "RESULT xkv4 speed n16:" in l:
            m = re.search(r"c16 \(\d+ window\(s\)\): agg ([\d.]+) tok/s \[[^\]]*\], per-stream p50 ([\d.]+)", l)
            if m: r["kv4_c16"], r["kv4_c16_p50"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"c8 \(\d+ window\(s\)\): agg ([\d.]+) tok/s \[[^\]]*\], per-stream p50 ([\d.]+)", l)
            if m: r["kv4_c8"], r["kv4_c8_p50"] = float(m.group(1)), float(m.group(2))
        if "RESULT xs5 ARMS T 0" in l:
            parts = l.split("|")
            for i, p in enumerate(parts):
                if p.strip().startswith("c8: A "):
                    for q in parts[i:i + 4]:
                        m = re.search(r"S5T ([\d.]+) \+- ([\d.]+)", q)
                        if m: r["s5t_c8"], r["s5t_c8_t95"] = float(m.group(1)), float(m.group(2)); break
                if p.strip().startswith("c8 p50: A "):
                    for q in parts[i:i + 4]:
                        m = re.search(r"S5T ([\d.]+) \+- ([\d.]+)", q)
                        if m: r["s5t_c8_p50"] = float(m.group(1)); break
    return r


def cmd_refs(a):
    r = refs(a.log)
    need = ("kv4_c16", "kv4_c16_p50", "s5t_c8")
    miss = [k for k in need if k not in r]
    print(("REFS OK" if not miss else "REFS MISSING %s" % miss) + " | kv4 n16 c16 %s tok/s (p50 %s), kv4 n16 c8 %s (p50 %s) | stack5 S5T c8 %s +- %s (p50 %s)" % (
        r.get("kv4_c16"), r.get("kv4_c16_p50"), r.get("kv4_c8"), r.get("kv4_c8_p50"), r.get("s5t_c8"), r.get("s5t_c8_t95"), r.get("s5t_c8_p50")))
    return 0 if not miss else 1


def pct(x, ref):
    return "%+.1f %%" % (100.0 * (x - ref) / ref) if x is not None and ref else "-"


def first_json(path, prefix):
    try:
        for l in open(path, errors="replace"):
            if l.startswith(prefix): return json.loads(l[len(prefix):])
    except (OSError, ValueError):
        return None
    return None


def cmd_report(a):
    O, tag = a.dir, a.tag
    ph = [p.split(":") for p in a.phases.split()]
    ident = {}
    try:
        for l in open(os.path.join(O, "identity.txt")):
            k, _, v = l.strip().partition(" ")
            ident[k] = v
    except OSError:
        pass
    st = jload(os.path.join(O, "%s-steps.json" % tag)) or {}
    rf = refs(a.log)
    out, virhe = [], []
    # ---- (a) identity
    grp = {"B48": [], "S": [], "C": [], "I": []}
    for k, v in ident.items():
        g = k.split(":", 1)[0]
        if g in grp: grp[g].append("%s %s" % (k.split(":", 1)[1], v))
        elif g.startswith("S"): grp["S"].append("%s %s" % (k, v))
    strict_bad = [k for k, v in ident.items() if k.split(":", 1)[0] in ("B48", "S") and not v.startswith("IDENTICAL")]
    out.append("IDENTITY under the cap (strict, B48: loaded vs alone, T 0 greedy and T 1 seeded with draft_sample 0): %s || S: %s || "
               "CONTROL cap 0 (counts, the > 48-row BI failure the cap removes): %s || INFORMATION: %s || verdict: %s" % (
                   "; ".join(grp["B48"]) or "-", "; ".join(grp["S"]) or "-", "; ".join(grp["C"]) or "-", "; ".join(grp["I"]) or "-",
                   "every strict compare IDENTICAL" if not strict_bad and grp["B48"] else
                   ("NOT IDENTICAL: %s (flagged where compared)" % ", ".join(strict_bad) if strict_bad else "no strict compare ran")))
    # ---- (b) steps + counters a window
    tot = dict(capped=0, trimmed=0)
    for row in ph:
        if len(row) != 3: continue
        nm, arm, kind = row
        w = "%s-%s" % (tag, nm)
        d = capd(jload(os.path.join(O, w + "-0-fn4.json")), jload(os.path.join(O, w + "-1-fn4.json")))
        s = st.get(nm)
        if d is None and s is None: continue   # not run (skipped / stopped)
        g = []
        if not a.capsw:
            d = None   # XC48_CAPSW=0: a server without the switch (the mock) -- the counters and the > 48 guard are not checked
        elif d is None: g.append("no cap counters")
        else:
            on = arm in ON_ARMS
            if on and d["steps_gt48"] != 0: g.append("steps_gt48 +%d under the cap" % d["steps_gt48"])
            if on and d["steps_over"] != 0: g.append("steps_over +%d" % d["steps_over"])
            if on and kind in LOADED and arm in "DZ" and (d["steps_capped"] <= 0 or d["drafts_trimmed"] <= 0): g.append("the cap did not act (DET at 16 sequences)")
            if kind in ALONE and d["steps_capped"] != 0: g.append("an alone window capped %d steps" % d["steps_capped"])
            if not on and d["steps_capped"] != 0: g.append("cap 0 capped %d steps" % d["steps_capped"])
            if d["in_force"] != (a.cap if on else 0): g.append("cap in force %s" % d["in_force"])
            if on:
                tot["capped"] += d["steps_capped"]; tot["trimmed"] += d["drafts_trimmed"]
        if a.capsw and s is not None and arm in ON_ARMS and s.get("decode_gt48"): g.append("%d decode steps over 48 rows in the step log" % s["decode_gt48"])
        if s is None: g.append("no step-log window")
        note = ""
        if d is not None and arm in OFF_ARMS and kind in LOADED:
            note = " (control: %s)" % ("reached > 48 rows in %d steps" % d["steps_gt48"] if d["steps_gt48"] > 0 else "did NOT reach > 48 rows -- uninformative")
        line = "STEPS %s (%s, arm %s): %s | step log: %s%s" % (
            nm, kind, arm, capline(d),
            ("%d steps, decode %d (> 48 rows %d, most %d, p50 %d), MIXED %d (up to %d sequences), prefill %d; BI stats > 48 +%s" % (
                s["steps"], s["decode"], s["decode_gt48"], s["decode_max"], s["decode_p50"], s["mixed"], s["mixed_nseq_max"], s["prefill"],
                s["bistats_gt48_delta"])) if s else "-", note)
        if g:
            line += " -- VIRHE " + "; ".join(g)
            virhe.append(nm)
        out.append(line)
        # freshness + driver results
        txt = os.path.join(O, w + ".txt")
        if kind in ("lid0", "lid1"):
            j = first_json(txt, "LOADID ")
            if not j: out.append("FRESH %s: VIRHE no LOADID line" % nm); virhe.append(nm + "-fresh")
            elif j.get("pf_cold") != j.get("of") or j.get("answered") != j.get("of") or j.get("c4_errors"):
                out.append("FRESH %s: VIRHE newcomers cold %s of %s, answered %s, stream errors %s" % (nm, j.get("pf_cold"), j.get("of"), j.get("answered"), j.get("c4_errors")))
                virhe.append(nm + "-fresh")
        if kind in ("mb16", "mb8"):
            n = c = 0
            try:
                for l in open(os.path.join(O, w + "-reqs.jsonl")):
                    r = json.loads(l)
                    if r.get("ev") == "done" and not r.get("err"):
                        n += 1; c += 1 if (r.get("cached") or 0) > 0 else 0
            except (OSError, ValueError):
                n = -1
            if n <= 0 or c:
                out.append("FRESH %s: VIRHE %s requests done, %d with cached > 0" % (nm, n, c)); virhe.append(nm + "-fresh")
    # ---- (c) speed
    for nm, ref, refp, rlab in (("c16", rf.get("kv4_c16"), rf.get("kv4_c16_p50"), "kv4 n16 c16 (int8 image, S5 pin, MAXSEQS 16 nvfp4, 2 windows)"),
                                ("c8", rf.get("s5t_c8"), rf.get("s5t_c8_p50"), "stack5 S5T c8 (int8 image, MAXSEQS 8 bf16, 4 blocks, +- %s)" % rf.get("s5t_c8_t95"))):
        j = first_json(os.path.join(O, "%s-%s.txt" % (tag, nm)), "MB ")
        if not j: continue
        x, p = j.get("agg_dec_tps"), j.get("per_stream_dec_p50")
        extra = ""
        if nm == "c8" and rf.get("kv4_c8"):
            extra = "; kv4 n16 c8 %s (p50 %s; MAXSEQS 16 nvfp4, S5, 1 window) -> %s" % (rf["kv4_c8"], rf.get("kv4_c8_p50"), pct(x, rf["kv4_c8"]))
        d = capd(jload(os.path.join(O, "%s-%s-0-fn4.json" % (tag, nm))), jload(os.path.join(O, "%s-%s-1-fn4.json" % (tag, nm))))
        out.append("SPEED %s (cap on, arm S = S5T + cap %s, T 0, dur %s warm %s): agg %s tok/s, per-stream p50 %s, decode step %s ms, %s tok/step, decode_share %s, TTFT p50 %s s, %s requests %s errors | cap %s | CROSS-START (labelled, not an A/B): %s %s tok/s (p50 %s) -> %s%s" % (
            nm, a.cap, a.dur, a.warm, x, p, j.get("ms_decode_step"), j.get("tok_per_decode_step"), j.get("decode_share"), j.get("ttft_p50"), j.get("requests"),
            j.get("errors"), capline(d), rlab, ref, refp, pct(x, ref), extra))
    if not a.capsw:
        out.append("NOTE XC48_CAPSW=0: no sched_row_cap was posted and no cap counter was checked -- a run on a server without the switch "
                   "(the no-GPU mock); not a lever reading")
    out.append("SUMMARY cap %s: identity %s | > 48-row steps under the cap: %s | the cap trimmed %d drafts in %d steps (cap-on windows) | guards %s" % (
        a.cap, "strict IDENTICAL" if not strict_bad and grp["B48"] else ("NOT IDENTICAL: " + ", ".join(strict_bad) if strict_bad else "not run"),
        ("none" if not any(n for n in virhe if not n.endswith("-fresh")) else "see VIRHE windows %s" % virhe) if a.capsw else "NOT CHECKED (XC48_CAPSW=0)",
        tot["trimmed"], tot["capped"],
        "OK" if not virhe else "VIRHE in %s" % ", ".join(virhe)))
    for l in out: print("RESULT xc48 " + l)
    return 0


def selftest(_a):
    import tempfile
    bad = []
    d = tempfile.mkdtemp(prefix="cap48sum-")
    def wj(p, o):
        with open(os.path.join(d, p), "w") as f: json.dump(o, f)
    def fn(c, steps_capped=0, trimmed=0, gt48=0, dec=0, mx=0):
        return {"switches": {"sched_row_cap": c}, "cap": {"in_force": c, "steps_capped": steps_capped, "drafts_trimmed": trimmed, "ahead_capped": 0,
                                                          "steps_over": 0, "steps_gt48": gt48, "max_rows": mx, "decode_steps": dec}}
    # capd
    x = capd(fn(48, 1, 3, 0, 10, 40), fn(48, 6, 99, 0, 30, 48))
    if not x or x["steps_capped"] != 5 or x["drafts_trimmed"] != 96 or x["max_rows"] != 48 or x["in_force"] != 48: bad.append("capd %r" % x)
    if capd({"switches": {}}, fn(48)) is not None: bad.append("capd without a cap block")
    # the step log: a decode step's n_tok, MIXED not summed, windows by receipt time, BI stats delta
    lg = os.path.join(d, "e.log.gz")
    with gzip.open(lg, "wt") as f:
        f.write("2026-11-12T07:51:20.500000000Z radiance: deepseek41: lever BI stats: 512 segmented steps, 3 MoE passes, 7 steps with > 48 decode rows; ds41_bi 2\n")
        f.write("2026-11-12T07:51:21.000000000Z I step 1 decode n_seq=16 n_tok=96  [0] slot=0 q=6 ctx=10\n")
        f.write("2026-11-12T07:51:22.000000000Z I step 2 decode n_seq=16 n_tok=48  [0] slot=0 q=3 ctx=16\n")
        f.write("2026-11-12T07:51:23.000000000Z I step 3 MIXED n_seq=17 n_tok=1148  [0] slot=0 q=3 ctx=22\n")
        f.write("2026-11-12T07:51:24.000000000Z radiance: deepseek41: lever BI stats: 1024 segmented steps, 3 MoE passes, 9 steps with > 48 decode rows; ds41_bi 2\n")
        f.write("2026-11-12T07:51:30.000000000Z I step 4 decode n_seq=16 n_tok=48  [0] slot=0 q=3 ctx=28\n")
    t0 = datetime.datetime(2026, 11, 12, 7, 51, 20, 900000, tzinfo=datetime.timezone.utc).timestamp()
    steps, bis = read_log(lg)
    w = windows(steps, bis, [("lid16", t0, t0 + 5), ("c16", t0 + 5, t0 + 20)])
    if len(steps) != 4 or len(bis) != 2: bad.append("read_log %d steps %d bis" % (len(steps), len(bis)))
    a1 = w["lid16"]
    if not (a1["steps"] == 3 and a1["decode"] == 2 and a1["decode_gt48"] == 1 and a1["decode_max"] == 96 and a1["mixed"] == 1 and a1["mixed_nseq_max"] == 17
            and a1["bistats_gt48_delta"] == 2): bad.append("window lid16 %r" % a1)
    if not (w["c16"]["steps"] == 1 and w["c16"]["decode_gt48"] == 0): bad.append("window c16 %r" % w["c16"])
    # refs
    rl = os.path.join(d, "run.log")
    with open(rl, "w") as f:
        f.write("2026-11-12 08:15:50 RESULT xkv4 speed n16: c8 (1 window(s)): agg 174.5 tok/s [174.5], per-stream p50 24.8, decode_share 0.811 | x | "
                "c16 (2 window(s)): agg 230.5 tok/s [225.7, 235.3], per-stream p50 15.7, decode_share 0.799 (/stats 0.799)\n")
        f.write("2026-11-12 06:56:27 RESULT xs5 ARMS T 0 (12 blocks, 0 flagged): 1u essay: A 74.8 +- 2.3 || c8: A 168.5 +- 0.7; >= 200: 31.5 short | S5 173.2 +- 2.0 (+2.8 % vs A) "
                "| S5T 173.6 +- 2.6 (+3.0 % vs A); >= 200: 26.4 short | S5 vs A +2.8 % || c8 p50: A 22.9 +- 0.4 | S5 23.5 +- 0.5 | S5T 23.7 +- 0.5 (+3.6 % vs A)\n")
    r = refs(rl)
    if not (r.get("kv4_c16") == 230.5 and r.get("kv4_c16_p50") == 15.7 and r.get("kv4_c8") == 174.5 and r.get("s5t_c8") == 173.6
            and r.get("s5t_c8_t95") == 2.6 and r.get("s5t_c8_p50") == 23.7): bad.append("refs %r" % r)
    # report: a clean run and a failing one
    with open(os.path.join(d, "identity.txt"), "w") as f:
        f.write("S:a1-vs-d0 IDENTICAL: 20 of 20 answers\nB48:lid16-vs-a1 IDENTICAL: 20 of 20 answers\nC:lid16o-vs-a1 8 of 20 answers differ: p1 char 3\n")
    on = dict(a1, decode_gt48=0, decode_max=48, decode_p50=48)
    wj("c16-steps.json", {"lid16": on, "lid16o": a1, "a1": dict(on, decode_max=6, decode_p50=6), "c16": on})
    wj("c16-lid16-0-fn4.json", fn(48, 0, 0, 0, 0)); wj("c16-lid16-1-fn4.json", fn(48, 100, 3000, 0, 400, 48))
    wj("c16-lid16o-0-fn4.json", fn(0, 100, 3000, 0, 400, 48)); wj("c16-lid16o-1-fn4.json", fn(0, 100, 3000, 300, 800, 96))
    wj("c16-a1-0-fn4.json", fn(48)); wj("c16-a1-1-fn4.json", fn(48, 0, 0, 0, 200, 6))
    lj = {"answered": 20, "of": 20, "pf_cold": 20, "c4_errors": 0}
    for n in ("lid16", "lid16o"):
        with open(os.path.join(d, "c16-%s.txt" % n), "w") as f: f.write("PROBE x\nLOADID " + json.dumps(lj) + "\n")
    with open(os.path.join(d, "c16-c16.txt"), "w") as f:
        f.write("MB " + json.dumps({"agg_dec_tps": 233.0, "per_stream_dec_p50": 15.9, "ms_decode_step": 55.0, "requests": 16, "errors": 0}) + "\n")
    with open(os.path.join(d, "c16-c16-reqs.jsonl"), "w") as f:
        f.write(json.dumps({"ev": "done", "cached": 0, "err": None}) + "\n")
    wj("c16-c16-0-fn4.json", fn(48)); wj("c16-c16-1-fn4.json", fn(48, 2, 10, 0, 600, 48))
    import io, contextlib
    def rep(phases):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_report(argparse.Namespace(dir=d, tag="c16", phases=phases, cap=48, capsw=1, dur=60, warm=15, log=rl))
        return buf.getvalue()
    o = rep("a1:D:alone0 lid16:D:lid0 lid16o:d:lid0 c16:S:mb16")
    if "VIRHE" in o or "every strict compare IDENTICAL" not in o or "reached > 48 rows in 300 steps" not in o or "+1.1 %" not in o:
        bad.append("report clean: %s" % o[:1500])
    wj("c16-lid16-1-fn4.json", fn(48, 0, 0, 5, 400, 96))   # under the cap: > 48 rows and nothing capped
    o = rep("a1:D:alone0 lid16:D:lid0")
    if "steps_gt48 +5 under the cap" not in o or "the cap did not act" not in o or "VIRHE in lid16" not in o:
        bad.append("report failing: %s" % o[:1500])
    print("SELFTEST " + ("OK: capd, the step log by receipt time (decode n_tok, MIXED counted apart, BI stats delta), refs, report guards (clean / failing)"
                         if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capd", "steps", "refs", "report", "selftest"))
    ap.add_argument("--fn0"); ap.add_argument("--fn1")
    ap.add_argument("--log"); ap.add_argument("--win", action="append"); ap.add_argument("--json")
    ap.add_argument("--dir"); ap.add_argument("--tag", default="c16"); ap.add_argument("--phases", default=""); ap.add_argument("--cap", type=int, default=48)
    ap.add_argument("--capsw", type=int, default=1)
    ap.add_argument("--dur", type=int, default=60); ap.add_argument("--warm", type=int, default=15)
    a = ap.parse_args()
    return {"capd": cmd_capd, "steps": cmd_steps, "refs": cmd_refs, "report": cmd_report, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
