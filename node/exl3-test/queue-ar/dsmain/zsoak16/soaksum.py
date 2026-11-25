#!/usr/bin/env python3
"""soaksum.py (13.11.2026, queue-ar/dsmain/zsoak16, a builder for ds-main-port) -- the helper of the DS int16 SOAK body
dsm-zsoak16.sh (item 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16, log tag "RESULT xzs16 "). Standard library only;
reads only, never writes outside the output dir it is given (summary --json); no server calls.

  soaksum.py pick --log RESULTLOG
      THE PIN AUTO-CHOICE (ds-main-port 13.11.): the LAST int16 confirmation run in the log (from its last
      "RESULT xzc16 <item>: start IMAGE=" line to the end): any "RESULT xzc16 identity ..." line whose result is not IDENTICAL
      (VIRHE / differ / DIFFERENT / not compared), or its STOP line ("IDENTITY GATE: VIRHE", "STOPPED at the identity gate",
      "identity SKIPPED by the budget") -> "PICK int15 because ..."; every identity line IDENTICAL -> "PICK int16 ...";
      no start line, or no identity line in that run -> "PICK int16 MISSING ..." (the xzc16 result is missing: int16).
  soaksum.py tel --csv TELEMETRY.csv --t0 EPOCH --t1 EPOCH
      one line: each card (chip UUID) over the window -- power W, junction C, memory C, sclk MHz (min-max), the sclk levels
      seen, fan rpm (min-max), samples.
  soaksum.py summary DIR --cycles soak-cycles.jsonl --tel telemetry.csv [--log ENGINE.log.gz] --pin int16|int15
             --pickwhy TEXT --sw 1|0 --min-cycles 6 [--templog-dir ~/gpu/temps] [--json OUT.json]
      the "SOAK16 ..." lines: PIN, ALONE (a1 + its cross-run compares), one CYCLE line each (n of 20 identical, the first
      differing char / token, pin held, load, INFO agg tok/s + step-log decode p50, telemetry), CONTROL (a2 vs a1), TOTALS,
      TELEMETRY per card (min / max over the soak; templog cross-check by PCI), VERDICT "EXACT UNDER LOAD over N cycles /
      X min: YES / NO".
  soaksum.py selftest
      offline, in memory: the picker on pass / fail / stop / missing / rerun logs, the step-log parser and p50, the
      telemetry reader, the cycle verdict logic. Prints "SELFTEST OK ..." or "SELFTEST FAIL ...".
"""
import argparse, calendar, gzip, io, json, os, re, statistics, sys, time

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------------------------------------------- the pick
RE_START = re.compile(r" RESULT xzc16 \S+: start IMAGE=")
RE_ID = re.compile(r" RESULT xzc16 identity (\S+)(?: \([^)]*\))?: (.*)$")
RE_STOP = re.compile(r"IDENTITY GATE: VIRHE|STOPPED at the identity gate|identity SKIPPED")


def pick(lines):
    """-> (pin, kind, why): pin int16 / int15; kind OK / FAIL / MISSING"""
    xs = [l.rstrip("\n") for l in lines if " RESULT xzc16 " in l]
    st = [k for k, l in enumerate(xs) if RE_START.search(l)]
    if not st:
        return "int16", "MISSING", "the xzc16 result is missing: no 'RESULT xzc16 <item>: start' line in the log"
    run = xs[st[-1]:]
    t0 = run[0][:19]
    ids, bad = [], []
    for l in run:
        m = RE_ID.search(l)
        if not m:
            continue
        res = m.group(2)
        ids.append(m.group(1))
        if not res.startswith("IDENTICAL") or re.search(r"VIRHE|differ|DIFFER|not compared", res):
            bad.append("%s: %s" % (m.group(1), res[:160]))
    stops = [l[20:].replace("RESULT xzc16 ", "")[:200] for l in run if RE_STOP.search(l)]
    if bad or stops:
        why = "; ".join((["identity %s" % b for b in bad[:2]] + ["STOP line '%s'" % s for s in stops[:1]]))
        return "int15", "FAIL", "the int16 identity gate failed in the xzc16 run started %s (%d identity line(s), %d not IDENTICAL, %d STOP line(s)): %s" % (
            t0, len(ids), len(bad), len(stops), why)
    if not ids:
        return "int16", "MISSING", "the xzc16 result is missing: its run started %s has no identity line (%d xzc16 lines)" % (t0, len(run))
    gate = any("IDENTITY GATE: every strict check IDENTICAL" in l for l in run)
    done = any(re.search(r" RESULT xzc16 \S+: done ", l) for l in run)
    return "int16", "OK", "the xzc16 run started %s: %d identity line(s), all IDENTICAL (%s)%s%s" % (
        t0, len(ids), ", ".join(ids[:8]), "; gate line IDENTICAL" if gate else "; no gate line (yet)", "; done" if done else "; no done line (yet)")


# ---------------------------------------------------------------------------------------------------------------- engagement
FUSEK = ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches")
RCOK = ("recordings", "copies", "freed")


KF1 = 80   # form-1 passes allowed a prefilled request (40 MoE layers x 2 segments; zzzc16/zc16sum.py 7f241a37's bound)


def form1_expected(pin, mv, prefills):
    """THE RULE HOOK for FUSE-R form-1 passes under the int16 pin (ds-main-port 13.11., verified in code: rad_block_moe_fp8.h:2288-2290,
    bounds :2000 / :2019; deepseek41 kBiMoe 16): under ds41_rfuse 2 form 2 serves MoE passes of <= 48 rows and form 1 those of 49 .. 170
    (MoeFP8::pass). Under ds41_bi 6 a decode pass is <= 16 rows, so form 1 can only come from a prefill segment of 49 .. 170 rows (a short
    prefill, or a PSK-cut layer at 160 .. 170). A form-1 delta over a span is INFO ("expected fallback", with its count) when the span
    prefilled >= 1 request and form 1 <= KF1 80 x the prefilled requests; form 1 > 0 in a decode-only span, or above that bound, is VIRHE.
    (The reference implementation: zzzc16/zc16sum.py 7f241a37 f1fallback / engaged.) -> (allowed, note)"""
    f1 = mv.get("rf_form1_passes") or 0
    if not f1:
        return True, ""
    if not prefills:
        return False, "form 1 +%d in a span with no prefill (decode / verify passes are <= 48 rows: form 2's)" % f1
    if f1 > KF1 * prefills:
        return False, "form 1 +%d beyond the fallback's bound (%d x %d prefilled requests)" % (f1, KF1, prefills)
    return True, "form 1 +%d (EXPECTED fallback, INFO: MoE passes over a prefill segment of 49 .. 170 rows; %d prefilled request(s) in the span, bound %d)" % (f1, prefills, KF1 * prefills)


def fuse_check(pin, a, b, xb, recorded=True, prefills=0):
    """GET /admin/fn4 snapshots a (before) / b (after) of a span -> (bad list, text). int16: ds41_rfuse / ds41_mix_fin / route_copy_ov /
    exl3_x3d 2 / 0 / 1 / xb, fuse.rf_declared / rf_layers > 0, rco in force, mixfin 0, form 2 + rgs launches MOVE in every span (ds-main-port:
    "Form 2 must still run (> 0) in every F span"), form 1 per form1_expected(prefills); recorded (an epoch bump before the span: every pass
    re-records) also rco recordings / freed > 0 (without one they are INFO). int15: 0 / 0 / 0 / xb, no fused form, rco not in force."""
    w = b.get("switches") or {}
    fa, fb, ra, rb = a.get("fuse") or {}, b.get("fuse"), a.get("rco") or {}, b.get("rco")
    want = (2, 0, 1, xb) if pin == "int16" else (0, 0, 0, xb)
    sw = (w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d"))
    bad, info = [], []
    if sw != want: bad.append("switches %s (want %s)" % (list(sw), list(want)))
    if not isinstance(fb, dict): bad.append("no \"fuse\" in GET /admin/fn4")
    if not isinstance(rb, dict): bad.append("no \"rco\" in GET /admin/fn4")
    fb, rb = fb or {}, rb or {}
    if not (fb.get("rf_declared") or 0) > 0 or not (fb.get("rf_layers") or 0) > 0:
        bad.append("fuse rf_declared %s rf_layers %s (want > 0)" % (fb.get("rf_declared"), fb.get("rf_layers")))
    mv = {k: (fb.get(k) or 0) - (fa.get(k) or 0) for k in FUSEK}
    mv.update({"rco_" + k: (rb.get(k) or 0) - (ra.get(k) or 0) for k in RCOK})
    if pin == "int16":
        if rb.get("in_force") != 1: bad.append("rco in_force %s (want 1)" % rb.get("in_force"))
        if mv["mixfin_launches"]: bad.append("mixfin_launches +%s (want 0: ds41_mix_fin 0)" % mv["mixfin_launches"])
        for k in ("rf_form2_passes", "rgs_launches"):
            if not mv[k] > 0: bad.append("%s +%s (want > 0: form 2 must run in every int16 span)" % (k, mv[k]))
        rco = ["%s +%s" % (k, mv[k]) for k in ("rco_recordings", "rco_freed") if not mv[k] > 0]
        if rco and recorded: bad.append("route_copy_ov 1 but no recording freed a copy (%s)" % ", ".join(rco))
        elif rco: info.append("rco %s (INFO: no epoch bump in the span, a recorded pass counts once, at recording)" % ", ".join(rco))
        ok1, note = form1_expected(pin, mv, prefills)
        if not ok1: bad.append(note)
        elif note: info.append(note)
    else:
        if rb.get("in_force"): bad.append("rco in_force %s (want 0: route_copy_ov 0)" % rb.get("in_force"))
        for k in ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches", "rco_recordings"):
            if mv[k]: bad.append("%s +%s (want 0: the int15 pin runs no fused form)" % (k, mv[k]))
    txt = "ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d %s, rco in_force %s, fuse rf_declared %s rf_layers %s | %s | prefilled %d%s" % (
        "/".join(str(x) for x in sw), rb.get("in_force"), fb.get("rf_declared"), fb.get("rf_layers"),
        ", ".join("%s +%s" % kv for kv in mv.items()), prefills, (" | " + "; ".join(info)) if info else "")
    return bad, txt


# ---------------------------------------------------------------------------------------------------------------- telemetry
TEL_HDR = "epoch,uuid,pci,power_w,junction_c,mem_c,edge_c,sclk_mhz,sclk_level,fan_rpm"


def tel_rows(lines):
    out = []
    for l in lines:
        f = l.strip().split(",")
        if len(f) < 10 or f[0] == "epoch":
            continue
        try:
            out.append(dict(t=float(f[0]), uuid=f[1], pci=f[2], p=int(f[3]), j=int(f[4]), m=int(f[5]), e=int(f[6]),
                            s=int(f[7]), lv=f[8], fan=int(f[9]) if f[9].lstrip("-").isdigit() else None))
        except ValueError:
            continue
    return out


def tel_cards(rows, t0=None, t1=None):
    by = {}
    for r in rows:
        if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1):
            by.setdefault(r["uuid"], []).append(r)
    res = {}
    for u, v in sorted(by.items()):
        g = lambda k: [x[k] for x in v if x[k] is not None]  # noqa: E731
        fans = g("fan")
        res[u] = dict(pci=v[0]["pci"], n=len(v), p=(min(g("p")), max(g("p")), round(statistics.mean(g("p")))),
                      j=(min(g("j")), max(g("j"))), m=(min(g("m")), max(g("m"))), s=(min(g("s")), max(g("s"))),
                      lv=sorted(set(x["lv"] for x in v)), fan=(min(fans), max(fans)) if fans else None)
    return res


def tel_line(cards):
    if not cards:
        return "no telemetry samples in the window"
    return " | ".join("%s (%s): %s-%s W (mean %s), Tj %s-%s C, Tmem %s-%s C, sclk %s-%s MHz lvl %s, fan %s rpm, n %d" % (
        u, c["pci"], c["p"][0], c["p"][1], c["p"][2], c["j"][0], c["j"][1], c["m"][0], c["m"][1], c["s"][0], c["s"][1],
        "/".join(c["lv"]), ("%s-%s" % c["fan"]) if c["fan"] else "n/a", c["n"]) for u, c in cards.items())


def templog(dirp, t0, t1, pci2uuid):
    """~/gpu/temps/<day>.csv (templog.sh, 10 s: epoch,pci,junction,mem,power,fan,sclk) over [t0, t1] by PCI -> {uuid: dict}"""
    by = {}
    for d in sorted({time.strftime("%Y%m%d", time.localtime(t)) for t in (t0, t1)}):
        p = os.path.join(dirp, d + ".csv")
        try:
            fh = open(p)
        except OSError:
            continue
        for l in fh:
            f = l.strip().split(",")
            if len(f) < 7:
                continue
            try:
                t, pci, j, m, w, s = float(f[0]), f[1], int(f[2]), int(f[3]), int(f[4]), int(f[6])
            except ValueError:
                continue
            if t0 <= t <= t1:
                by.setdefault(pci, []).append((j, m, w, s))
    out = {}
    for pci, v in by.items():
        u = pci2uuid.get(pci, "pci-" + pci)
        out[u] = dict(pci=pci, n=len(v), j=(min(x[0] for x in v), max(x[0] for x in v)), m=(min(x[1] for x in v), max(x[1] for x in v)),
                      p=(min(x[2] for x in v), max(x[2] for x in v)), s=(min(x[3] for x in v), max(x[3] for x in v)))
    return out


# ---------------------------------------------------------------------------------------------------------------- step log
RE_STEP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z .*?\bstep (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")


def tsec(a, frac):
    return calendar.timegm(time.strptime(a, "%Y-%m-%dT%H:%M:%S")) + (float("0" + frac) if frac else 0.0)


def steps(lines):
    """-> [(t, kind, n_seq, n_tok)] in log order"""
    out = []
    for l in lines:
        m = RE_STEP.match(l)
        if m:
            out.append((tsec(m.group(1), m.group(2)), m.group(4), int(m.group(5)), int(m.group(6))))
    return out


def decode_p50(st, t0, t1, min_seq=0):
    """p50 ms of the decode steps starting in [t0, t1] (a step's ms = the next step line's time minus its own), count"""
    ms = [1000.0 * (st[k + 1][0] - st[k][0]) for k in range(len(st) - 1)
          if st[k][1] == "decode" and t0 <= st[k][0] <= t1 and st[k][2] >= min_seq]
    return (round(statistics.median(ms), 2) if ms else None), len(ms)


def readlines(p):
    if not p or not os.path.exists(p):
        return []
    try:
        if p.endswith(".gz"):
            with gzip.open(p, "rt", errors="replace") as f:
                try:
                    return f.read().splitlines()
                except EOFError:
                    return []
        with open(p, errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def first_line(p, prefix):
    for q in (p, p + ".gz"):
        for l in readlines(q):
            if l.startswith(prefix):
                return l[len(prefix):]
    return None


def jload(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def answers(p):
    out = {}
    for l in readlines(p):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if isinstance(d, dict) and "i" in d:
            out[d["i"]] = d
    return out


# ---------------------------------------------------------------------------------------------------------------- one cycle
def cyc_eval(cmpd, ans):
    """cmp json (loadid.py cmp) + the cycle's answers -> dict: identical n, differing, missing, errors, first diff text"""
    if not isinstance(cmpd, dict):
        return dict(ok=False, n=0, ident=0, ndiff=None, err=[], missing=[], first="not compared (no cmp json)")
    n = cmpd.get("n") or 0
    err = sorted(i for i, d in ans.items() if d.get("text") is None)
    diff = [d for d in (cmpd.get("diff") or []) if d.get("i") not in err]
    missing = list(cmpd.get("missing") or [])
    ident = n - len(cmpd.get("diff") or []) - len(missing)
    first = None
    if diff:
        d = diff[0]
        first = "p%s char %s tok %s (%s vs %s chars%s)%s" % (d.get("i"), d.get("char"), "-" if d.get("tok") is None else d.get("tok"),
                                                             d.get("len_a"), d.get("len_b"), (", %s" % d["what"]) if d.get("what") else "",
                                                             (" alone %r | load %r" % (d.get("at_a"), d.get("at_b"))) if d.get("at_a") is not None else "")
    ok = bool(cmpd.get("identical")) and not err and n == 20
    return dict(ok=ok, n=n, ident=ident, ndiff=len(diff), err=err, missing=missing, first=first)


def summary(a):
    D = a.dir
    cyc = [json.loads(l) for l in readlines(a.cycles) if l.strip().startswith("{")]
    trows = tel_rows(readlines(a.tel))
    st = steps(readlines(a.log)) if a.log else []
    out, J = [], dict(pin=a.pin, cycles=[])
    pr = lambda s: out.append("SOAK16 " + s)  # noqa: E731
    pr("PIN = %s (%s)%s" % (a.pin, a.pickwhy, "" if a.sw == "1" else " | MOCK RUN ZS16_SW=0: NOT A READING"))
    a1 = answers(os.path.join(D, "xzs16-a1.jsonl"))
    xr = []
    for nm in ("a1-vs-ref12-a1", "a1-vs-zzzc16"):
        c = jload(os.path.join(D, "xzs16-cmp-%s.json" % nm))
        xr.append("%s %s" % (nm, ("IDENTICAL %d of %d" % (c.get("n"), c.get("n")) if c.get("identical") else "DIFFERENT (%s of %s differ, missing %s)" % (c.get("n_diff"), c.get("n"), c.get("missing"))) if c else "not compared"))
        J[nm] = bool(c and c.get("identical"))
    pr("ALONE a1 (greedy-20 c1, T 0, 192 tok): %d of 20 answered | strict cross-run: %s%s" % (
        sum(1 for d in a1.values() if d.get("text") is not None), "; ".join(xr), "" if a.sw == "1" else " (INFO under the mock)"))
    tot = dict(n=0, probes=0, ident=0, diff=0, err=0, missing=0, held=0, engaged=0, bad=[])
    allok = True
    for c in cyc:
        i = c["cycle"]
        cm = jload(os.path.join(D, "xzs16-cmp-c%d.json" % i))
        ans = answers(os.path.join(D, "xzs16-c%d.jsonl" % i))
        e = cyc_eval(cm, ans)
        ld = first_line(os.path.join(D, "xzs16-c%d.txt" % i), "LOADID ")
        try:
            ld = json.loads(ld) if ld else {}
        except ValueError:
            ld = {}
        w = ld.get("window") or {}
        warm = float(c.get("warm") or 0)
        u0, u1 = float(c["u0"]), float(c["u1"])
        p0, p1 = u0 + warm, (u0 + float(ld["window_s"])) if ld.get("window_s") else u1
        dur = max(1e-6, p1 - p0)
        agg = round(w["dec_tok"] / dur, 1) if w.get("dec_tok") else None
        p50, nst = decode_p50(st, p0, p1) if st else (None, 0)
        p50f, nstf = decode_p50(st, p0, p1, min_seq=13) if st else (None, 0)
        tc = tel_cards(trows, u0, u1)
        held = c.get("pin_ok") == 1
        engc = c.get("eng_ok", 1) == 1
        tot["n"] += 1; tot["probes"] += e["n"]; tot["ident"] += e["ident"]; tot["diff"] += e["ndiff"] or 0
        tot["err"] += len(e["err"]); tot["missing"] += len(e["missing"]); tot["held"] += 1 if held else 0; tot["engaged"] += 1 if engc else 0
        cok = e["ok"] and held and engc
        if not cok:
            tot["bad"].append(i)
        allok = allok and cok
        pr("CYCLE %d: %s%s | pin %s | engagement %s | load: %s of %s answered, c%s-load %s requests / %s errors, prefills cold %s, pf TTFT mean %s s, probe in flight median %s %%, mixed steps in probes min %s | INFO agg %s tok/s (engine decode tokens over the probe phase %.0f s), step-log decode p50 %s ms (%d steps; rows >= 13: %s ms, %d) | warm %s s, cycle %.0f s | %s" % (
            i, ("IDENTICAL %d of 20" % e["ident"]) if e["ok"] else ("VIRHE %d of %d identical, %d differ%s%s" % (
                e["ident"], e["n"], e["ndiff"] or 0, (", %d errors %s" % (len(e["err"]), e["err"])) if e["err"] else "", (", missing %s" % e["missing"]) if e["missing"] else "")),
            (" -- first diff %s" % e["first"]) if e["first"] else "",
            "held" if held else "VIRHE %s" % (c.get("pin_txt") or "not held")[:200],
            ("%s (%d prefilled, %s)" % ("OK" if a.sw == "1" else "INFO (mock)", c.get("prefills") or 0, (c.get("eng") or "").split(" | ")[1][:160] if " | " in (c.get("eng") or "") else "")) if engc else "VIRHE %s" % re.sub(r"^VIRHE ", "", c.get("eng") or "no line")[-300:],
            ld.get("answered"), ld.get("of"), ld.get("conc"), ld.get("c4_requests_done"), ld.get("c4_errors"), ld.get("pf_cold"), ld.get("pf_ttft_mean"),
            round(100 * (ld.get("ov_frac_median") or 0)), ld.get("mixed_min"), agg, dur, p50, nst, p50f, nstf, c.get("warm"), u1 - u0, tel_line(tc)))
        J["cycles"].append(dict(cycle=i, ok=cok, identical=e["ident"], n=e["n"], differ=e["ndiff"], errors=e["err"], missing=e["missing"],
                                first=e["first"], pin_held=held, engaged=engc, eng=c.get("eng"), agg_tps=agg, decode_p50_ms=p50, decode_steps=nst, decode_p50_rows13_ms=p50f,
                                u0=u0, u1=u1, warm=c.get("warm"), tel=tc))
    a2 = jload(os.path.join(D, "xzs16-cmp-a2-vs-a1.json"))
    a2ok = bool(a2 and a2.get("identical"))
    pr("CONTROL a2 (greedy-20 c1 alone after the cycles) vs a1: %s" % (("IDENTICAL %d of %d" % (a2.get("n"), a2.get("n"))) if a2ok else (
        "VIRHE %s of %s differ, missing %s" % (a2.get("n_diff"), a2.get("n"), a2.get("missing")) if a2 else "VIRHE not compared")))
    t_first = min((float(c["u0"]) for c in cyc), default=None)
    t_last = max((float(c["u1"]) for c in cyc), default=None)
    mins = round((t_last - t_first) / 60.0, 1) if cyc else 0.0
    pr("TOTALS: %d cycles x 20 probes = %d probes under load; %d identical, %d differing, %d request errors, %d missing; pin held %d of %d cycles; engaged (form 2 ran, form 1 within the rule) %d of %d; cycles not clean: %s; soak window %.1f min%s" % (
        tot["n"], tot["probes"], tot["ident"], tot["diff"], tot["err"], tot["missing"], tot["held"], tot["n"], tot["engaged"], tot["n"], tot["bad"] or "none", mins,
        (" -- NOTE %d cycles < %d (the budget or a stop cut the soak)" % (tot["n"], a.min_cycles)) if tot["n"] < a.min_cycles else ""))
    soak = tel_cards(trows, t_first, t_last) if cyc else {}
    p2u = {c["pci"]: u for u, c in soak.items()}
    tl = templog(os.path.expanduser(a.templog_dir), t_first, t_last, {k[5:10] if len(k) > 10 else k: v for k, v in p2u.items()}) if cyc and a.templog_dir else {}
    for u, c in soak.items():
        x = tl.get(u)
        pr("TELEMETRY %s (%s) over the soak: power %s-%s W (mean %s), junction %s-%s C, memory %s-%s C, sclk %s-%s MHz (levels %s), fan %s rpm, %d samples | templog cross-check: %s" % (
            u, c["pci"], c["p"][0], c["p"][1], c["p"][2], c["j"][0], c["j"][1], c["m"][0], c["m"][1], c["s"][0], c["s"][1], "/".join(c["lv"]),
            ("%s-%s" % c["fan"]) if c["fan"] else "n/a", c["n"],
            ("junction %s-%s C, memory %s-%s C, power %s-%s W, sclk %s-%s MHz, %d rows" % (x["j"] + x["m"] + x["p"] + x["s"] + (x["n"],))) if x else "no rows"))
    if cyc and not soak:
        pr("TELEMETRY: VIRHE no telemetry samples over the soak window")
    hot = os.path.exists(os.path.join(D, "soak.hot"))
    eng = {}
    for l in readlines(a.engage) if a.engage else []:
        k, _, r = l.partition(" ")
        eng[k] = r
    es = eng.get("xzs16-n5"), eng.get("xzs16-end")
    eok = lambda r: bool(r) and r.startswith("OK")  # noqa: E731
    bad_ = "VIRHE " if a.sw == "1" else "INFO (ZS16_SW=0 mock: no fuse / rco counters) "
    pr("ENGAGEMENT (soaksum.py fuse_check + form1_expected, an epoch-bumped 64-token probe): start %s | end %s" % tuple(
        ("OK " + r[3:300]) if eok(r) else (bad_ + re.sub(r"^VIRHE ", "", r or "no line")[:300]) for r in es))
    engok = a.sw != "1" or (eok(es[0]) and eok(es[1]))
    yes = bool(cyc) and allok and a2ok and not hot and engok
    J.update(totals=tot, control_identical=a2ok, minutes=mins, verdict="YES" if yes else "NO", hot=hot, telemetry=soak, engagement=dict(start=es[0], end=es[1], ok=engok))
    pr("VERDICT: EXACT UNDER LOAD over %d cycles / %.1f min: %s%s%s" % (
        tot["n"], mins, "YES" if yes else "NO", "" if yes else " (%s)" % "; ".join(x for x in (
            "no cycle ran" if not cyc else "", ("cycles not clean %s (identity / pin / engagement: see the CYCLE lines)" % tot["bad"]) if tot["bad"] else "",
            "" if a2ok else "the a2 control is not IDENTICAL", "the temperature guard stopped the soak" if hot else "",
            "" if engok else "the engagement probe failed (start or end)") if x),
        "" if a.sw == "1" else " | MOCK RUN: NOT A READING"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(J, f)
    print("\n".join(out))
    return 0


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    bad = []
    S = "2026-11-13 02:51:13 RESULT xzc16 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16: start IMAGE=radiance-local:x (4419a17d60a7) | ..."
    ok1 = "2026-11-13 02:57:00 RESULT xzc16 identity xzc16-a1-F-vs-ref12-a1 (xref12-a1.jsonl vs xzc16-a1-F.jsonl): IDENTICAL: 20 of 20 answers"
    ok2 = "2026-11-13 02:57:11 RESULT xzc16 identity xzc16-gs16-F-vs-a1-F (xzc16-a1-F.jsonl vs xzc16-gs16-F.jsonl): IDENTICAL: 20 of 20 answers"
    okc = "2026-11-13 03:10:00 RESULT xzc16 identity xzc16-cold136000-F-vs-ref14 (cold-136000-1-O.txt vs cold-136000-1-F.txt, deepq 136000, 64 greedy tokens): IDENTICAL (412 chars)"
    gate = "2026-11-13 02:59:00 RESULT xzc16 IDENTITY GATE: every strict check IDENTICAL (5 lines) -> the speed blocks run | B {} | F {}"
    dif = "2026-11-13 02:59:00 RESULT xzc16 identity xzc16-ld-F-vs-a1-F (xzc16-a1-F.jsonl vs xzc16-ld-F.jsonl): 1 of 20 answers differ: p3 char 57 tok 12 -- reported, not hidden"
    vir = "2026-11-13 02:59:00 RESULT xzc16 identity ld-F-vs-a1-F: VIRHE 1 of 20 answers differ: p3 char 57 tok 12 (strict, T 0 greedy)"
    stp = "2026-11-13 02:59:01 RESULT xzc16 IDENTITY GATE: VIRHE strict identity failed [ld-F-vs-a1-F] -> the item STOPS before every speed block"
    cdf = "2026-11-13 03:10:00 RESULT xzc16 identity xzc16-cold136000-F-vs-ref14 (cold-136000-1-O.txt vs cold-136000-1-F.txt, deepq 136000, 64 greedy tokens): DIFFERENT at char 7 (412 vs 400 chars) -- reported, not hidden"
    noise = "2026-11-13 02:40:00 RESULT xzs16 identity something: DIFFERENT"
    fsay = "2026-11-13 02:59:22 RESULT xzc16 identity: ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d read back before each block (GET /admin/fn4, arm:values): O:2/0/1/2"
    cases = [
        ([], "int16", "MISSING"),
        ([noise], "int16", "MISSING"),
        ([S], "int16", "MISSING"),
        ([S, ok1, ok2, gate], "int16", "OK"),
        ([S, ok1, ok2, fsay, gate], "int16", "OK"),
        ([S, ok1, ok2, gate, okc], "int16", "OK"),
        ([S, ok1, dif, vir, stp], "int15", "FAIL"),
        ([S, ok1, stp], "int15", "FAIL"),
        ([S, ok1, ok2, gate, cdf], "int15", "FAIL"),
        ([S, ok1, dif, vir, stp, S, ok1, ok2, gate], "int16", "OK"),        # a re-run: only the last run counts
        ([S, ok1, ok2, gate, S], "int16", "MISSING"),                        # the last run has no identity line yet
        ([S, ok1, ok2, "2026-11-13 02:59:00 RESULT xzc16 identity SKIPPED by the budget: VIRHE"], "int15", "FAIL"),
    ]
    for k, (ls, pin, kind) in enumerate(cases):
        p, kd, why = pick(ls)
        if (p, kd) != (pin, kind):
            bad.append("pick case %d -> %s %s (want %s %s): %s" % (k, p, kd, pin, kind, why))
    # step log: 4 decode steps 40 ms apart, one MIXED
    L = ["2026-11-13T03:00:00.000000000Z I step 1 decode n_seq=14 n_tok=40  [0] slot=1 q=3 ctx=100",
         "2026-11-13T03:00:00.040000000Z I step 2 decode n_seq=14 n_tok=40",
         "2026-11-13T03:00:00.080000000Z I step 3 MIXED n_seq=15 n_tok=2088",
         "2026-11-13T03:00:00.380000000Z I step 4 decode n_seq=12 n_tok=30",
         "2026-11-13T03:00:00.410000000Z I step 5 decode n_seq=14 n_tok=40",
         "garbage line", "2026-11-13T03:00:00.450000000Z I step 6 prefill n_seq=1 n_tok=4096"]
    st = steps(L)
    t0 = calendar.timegm(time.strptime("2026-11-13T03:00:00", "%Y-%m-%dT%H:%M:%S"))
    if len(st) != 6:
        bad.append("steps parsed %d (want 6)" % len(st))
    p50, n = decode_p50(st, t0, t0 + 1)
    if n != 4 or abs(p50 - 40.0) > 0.01:
        bad.append("decode p50 %s n %s (want 40.0, 4)" % (p50, n))
    p50f, nf = decode_p50(st, t0, t0 + 1, min_seq=13)
    if nf != 3:
        bad.append("rows>=13 count %s (want 3)" % nf)
    # telemetry
    T = [TEL_HDR, "100.0,GPU-aa,0000:03:00.0,210,63,56,50,2831,1,2600", "105.0,GPU-aa,0000:03:00.0,190,65,57,51,2700,1,2650",
         "105.0,GPU-bb,0000:48:00.0,212,61,55,49,2845,1,", "200.0,GPU-aa,0000:03:00.0,10,40,40,40,600,0,2000", "bad,row"]
    tr = tel_rows(T)
    if len(tr) != 4:
        bad.append("tel rows %d (want 4)" % len(tr))
    tc = tel_cards(tr, 99, 150)
    if tc.get("GPU-aa", {}).get("p") != (190, 210, 200) or tc["GPU-aa"]["j"] != (63, 65) or tc["GPU-bb"]["fan"] is not None or tc["GPU-aa"]["lv"] != ["1"]:
        bad.append("tel cards %s" % tc)
    if "GPU-aa (0000:03:00.0): 190-210 W" not in tel_line(tc):
        bad.append("tel line %s" % tel_line(tc))
    # cycle eval
    ans = {i: {"i": i, "text": "x"} for i in range(20)}
    e = cyc_eval({"n": 20, "identical": True, "diff": [], "missing": []}, ans)
    if not e["ok"] or e["ident"] != 20:
        bad.append("cyc_eval identical %s" % e)
    e = cyc_eval({"n": 20, "identical": False, "diff": [{"i": 3, "char": 57, "tok": 12, "len_a": 100, "len_b": 101, "at_a": "ab", "at_b": "ac"}], "missing": []}, ans)
    if e["ok"] or e["ident"] != 19 or e["ndiff"] != 1 or not e["first"].startswith("p3 char 57 tok 12"):
        bad.append("cyc_eval differ %s" % e)
    ans2 = dict(ans); ans2[5] = {"i": 5, "text": None}
    e = cyc_eval({"n": 20, "identical": False, "diff": [{"i": 5, "char": 0, "tok": 0, "len_a": 10, "len_b": 0}], "missing": []}, ans2)
    if e["ok"] or e["err"] != [5] or e["ndiff"] != 0:
        bad.append("cyc_eval error %s" % e)
    if cyc_eval(None, ans)["ok"]:
        bad.append("cyc_eval None ok")
    # engagement (fuse_check + the form1_expected hook, ds-main-port's rule)
    def fnj(rf, rc, f1, f2, rgs, mixf, rec, freed, inf):
        return {"switches": {"ds41_rfuse": rf, "ds41_mix_fin": 0, "route_copy_ov": rc, "exl3_x3d": 2},
                "fuse": {"rf_declared": 3, "rf_layers": 40, "rf_form1_passes": f1, "rf_form2_passes": f2, "rgs_launches": rgs, "mixfin_launches": mixf},
                "rco": {"in_force": inf, "recordings": rec, "copies": freed, "freed": freed}}
    z16, z15 = fnj(2, 1, 0, 0, 0, 0, 0, 0, 1), fnj(0, 0, 0, 0, 0, 0, 0, 0, 0)
    ecases = [   # (pin, after, recorded, prefills, n bad, INFO expected)
        ("int16", fnj(2, 1, 0, 360, 1440, 0, 4, 90, 1), True, 1, 0, False),       # the zzzc16 start probe
        ("int16", fnj(2, 1, 19, 2360, 9000, 0, 4, 90, 1), True, 20, 0, True),     # form 1 +19 with 20 prefills: expected fallback (INFO)
        ("int16", fnj(2, 1, 19, 2360, 9000, 0, 4, 90, 1), False, 0, 1, False),    # form 1 in a decode-only span: VIRHE
        ("int16", fnj(2, 1, 81, 2360, 9000, 0, 4, 90, 1), True, 1, 1, False),     # form 1 above 80 x 1: VIRHE
        ("int16", fnj(2, 1, 80, 2360, 9000, 0, 4, 90, 1), True, 1, 0, True),      # at the bound: INFO
        ("int16", fnj(2, 1, 19, 0, 0, 0, 4, 90, 1), True, 20, 2, True),           # form 1 while form 2 did not move: form 2 / rgs VIRHE
        ("int16", fnj(2, 1, 0, 360, 1440, 0, 0, 0, 1), True, 1, 1, False),        # recorded span, rco did not record: VIRHE
        ("int16", fnj(2, 1, 0, 360, 1440, 0, 0, 0, 1), False, 50, 0, True),       # soak cycle without an epoch bump: rco INFO
        ("int16", fnj(2, 1, 0, 0, 0, 0, 0, 0, 1), False, 50, 2, True),            # form 2 must run in every span, recorded or not
        ("int16", fnj(2, 1, 0, 360, 1440, 5, 4, 90, 1), True, 1, 1, False),       # mixfin moved
        ("int16", fnj(0, 0, 0, 360, 1440, 0, 4, 90, 1), True, 1, 1, False),       # wrong switches
        ("int15", fnj(0, 0, 0, 0, 0, 0, 0, 0, 0), True, 1, 0, False),
        ("int15", fnj(0, 0, 0, 360, 1440, 0, 0, 0, 0), False, 50, 2, False),      # a fused form ran under int15
    ]
    for k, (pin, bb_, rec, npf, nbad, inf) in enumerate(ecases):
        bb, txt = fuse_check(pin, z16 if pin == "int16" else z15, bb_, 2, recorded=rec, prefills=npf)
        if len(bb) != nbad or (inf and "INFO" not in txt):
            bad.append("fuse_check case %d -> %d bad %s (want %d%s): %s" % (k, len(bb), bb, nbad, ", INFO" if inf else "", txt[-200:]))
    print("SELFTEST " + ("OK: pick %d cases (pass / fail / stop / missing / rerun / cold / budget), step-log p50, telemetry, cycle eval, engagement %d cases (form1_expected hook)" % (len(cases), len(ecases)) if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pick", "tel", "engage", "summary", "selftest"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--fn0")
    ap.add_argument("--fn1")
    ap.add_argument("--xb", type=int, default=2)
    ap.add_argument("--recorded", type=int, default=1)
    ap.add_argument("--prefills", type=int, default=0)
    ap.add_argument("--log")
    ap.add_argument("--csv")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--cycles")
    ap.add_argument("--tel")
    ap.add_argument("--pin", default="int16")
    ap.add_argument("--pickwhy", default="")
    ap.add_argument("--sw", default="1")
    ap.add_argument("--min-cycles", type=int, default=6)
    ap.add_argument("--templog-dir", default="")
    ap.add_argument("--json")
    ap.add_argument("--engage")
    a = ap.parse_args()
    if a.mode == "selftest":
        sys.exit(selftest())
    if a.mode == "pick":
        pin, kind, why = pick(readlines(a.log))
        print("PICK %s %s%s" % (pin, "" if kind == "OK" else kind + " ", why))
        sys.exit(0)
    if a.mode == "tel":
        print("TEL " + tel_line(tel_cards(tel_rows(readlines(a.csv)), a.t0, a.t1)))
        sys.exit(0)
    if a.mode == "engage":
        b, txt = fuse_check(a.pin, jload(a.fn0) or {}, jload(a.fn1) or {}, a.xb, recorded=a.recorded == 1, prefills=a.prefills)
        print(("ENGAGE OK " if not b else "ENGAGE VIRHE ") + txt + ("" if not b else " -- " + "; ".join(b)))
        sys.exit(0)
    sys.exit(summary(a))
