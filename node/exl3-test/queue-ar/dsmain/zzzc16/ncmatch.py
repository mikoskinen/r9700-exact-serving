#!/usr/bin/env python3
"""ncmatch.py (12.11.2026, queue-ar/dsmain/ref14, a builder for ds-main-port) -- the MATCHED-PROMPT newcomer reading (coordinator 12.11.
method fix after yzint14c): prof12/p12win.py split counts a newcomer step that STARTS inside the window as a whole step but clips
its wall at the window end, and a 35 s window holds a whole number of newcomers (~1.5 % of the window each), so its "newcomer ms a
step" (and the block's agg) move with how many newcomers an arm pulls inside (A9 in yzint14c: (6930.8 + ~5) / 14, not 6925 / 13).
This helper reads the SAME fixed prompts' newcomer-step walls instead: every request (w, k) of the fixed prompt set (harness/mbt3.py
--prompt-set fixed: request (w, k) is the same text in every block) is tied to ITS newcomer step in the engine step log (the last
non-decode step with a prefill segment q > 16 that starts between the request's send and its first token), the step's wall =
the NEXT step line's time minus its own (not clipped by any window), and only the prompts whose newcomer step starts inside the
window of EVERY block of the set are summed. prof12/p12win.py is not changed (other items read it in place). Standard library only.

  ncmatch.py LOG --set LABEL --block NAME:ARM:REQS.jsonl [--block ...] [--json OUT.json] [--min 4]
      LOG = the engine log captured with `docker logs -t` (gzip or plain, p12win.py's reader); REQS = the block's ncmb3 / mbt3 rows
      (send / done / mark). Prints one line "NCMATCH LABEL: ..." (the matched prompts, each block's sum over them, each arm's mean of
      its blocks' sums and ms a newcomer step, every arm vs the FIRST arm named: ms and %) and per block "NCMATCH LABEL block ...";
      fewer than --min matched prompts (default 4) is a VIRHE (a set whose newcomer sum is a reading); --min 1 for INFO sets.
  ncmatch.py selftest"""
import gzip, io, json, os, re, statistics, sys, tempfile, zlib
from datetime import datetime, timezone

TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")
STEP = re.compile(r" step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
SEG = re.compile(r"\[\d+\] slot=(\d+) q=(\d+) ctx=(\d+)")
QMIN = 16          # a prefill segment: more rows than any decode / verify segment (draft depth <= 15)
IDLE = 3.0         # a step followed by > 3 s of nothing is a host gap, not a wall (p12win.py's --idle)


def opn(p):   # gzip members and any plain text after them (p12win.py's reader; a mock capture appends plain lines)
    with open(p, "rb") as f: data = f.read()
    out = []
    while data:
        if data[:2] == b"\x1f\x8b":
            d = zlib.decompressobj(31); out.append(d.decompress(data)); out.append(d.flush()); data = d.unused_data
        else: out.append(data); break
    return io.StringIO(b"".join(out).decode("utf-8", errors="replace"))


def tsec(m):
    t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return t + (float("0." + m.group(2)) if m.group(2) else 0.0)


def read_steps(path):
    """[(t, step, phase, n_seq, n_tok, [(slot, q, ctx), ..])] in log order"""
    out = []
    for l in opn(path):
        m = TSRE.match(l)
        if not m or " step " not in l: continue
        s = STEP.search(l.rstrip("\n"))
        if not s: continue
        out.append((tsec(m), int(s.group(1)), s.group(2), int(s.group(3)), int(s.group(4)), [tuple(int(x) for x in g) for g in SEG.findall(s.group(5))]))
    return out


def read_reqs(path):
    marks, done = [], []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("ev") == "mark": marks.append(r.get("t"))
        elif r.get("ev") == "done" and not r.get("err") and r.get("t_first") is not None: done.append(r)
    marks = sorted(x for x in marks if x is not None)
    return (marks[0], marks[1]) if len(marks) >= 2 else None, done


def tie(steps, done):
    """each done request -> (key, step index, q, prompt) of its newcomer step (or None)"""
    ts = [s[0] for s in steps]
    import bisect
    out = []
    for r in done:
        a = (r.get("t_send") or r.get("t0") or r["t_first"] - 30.0) - 0.05
        b = r["t_first"]
        j = bisect.bisect_right(ts, b) - 1
        hit = None
        while j >= 0 and ts[j] >= a:
            s = steps[j]
            if s[2] != "decode":
                big = [g for g in s[5] if g[1] > QMIN]
                if big: hit = (j, max(g[1] for g in big)); break
            j -= 1
        out.append(((r.get("w"), r.get("k")), hit[0] if hit else None, hit[1] if hit else None, r.get("prompt")))
    return out


def wall(steps, j):
    if j is None or j + 1 >= len(steps): return None
    d = steps[j + 1][0] - steps[j][0]
    return 1000.0 * d if 0.0 <= d <= IDLE else None


def run(argv):
    log, label, blocks, jout, mn = None, "set", [], None, 4
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--set": label = argv[i + 1]; i += 2
        elif a == "--block": blocks.append(argv[i + 1]); i += 2
        elif a == "--json": jout = argv[i + 1]; i += 2
        elif a == "--min": mn = int(argv[i + 1]); i += 2
        else: log = a; i += 1
    if not log or not blocks:
        print("NCMATCH %s: VIRHE usage: LOG --set LABEL --block NAME:ARM:REQS [...]" % label); return 2
    steps = read_steps(log)
    if not steps:
        print("NCMATCH %s: VIRHE no step line in %s" % (label, log)); return 1
    B = []
    for spec in blocks:
        nm, arm, rq = spec.split(":", 2)
        win, done = read_reqs(rq) if os.path.exists(rq) else (None, [])
        if not win:
            print("NCMATCH %s block %s: VIRHE no window marks in %s" % (label, nm, rq)); continue
        T = tie(steps, done)
        inwin = {}
        untied = 0
        for key, j, q, pr in T:
            if j is None: untied += 1; continue
            if win[0] <= steps[j][0] <= win[1]: inwin[key] = (j, q, pr)
        B.append(dict(name=nm, arm=arm, win=win, inwin=inwin, untied=untied, ndone=len(done)))
    if not B:
        print("NCMATCH %s: VIRHE no block with a window" % label); return 1
    keys = set(B[0]["inwin"])
    for b in B[1:]: keys &= set(b["inwin"])
    keys = sorted(keys, key=lambda k: (B[0]["inwin"][k][0]))
    res = dict(label=label, matched=[list(k) for k in keys], blocks=[], arms={}, min=mn)
    for b in B:
        js = sorted({b["inwin"][k][0] for k in keys})          # a co-batched step counted once
        ws = [wall(steps, j) for j in js]
        bad = sum(1 for w in ws if w is None)
        s = sum(w for w in ws if w is not None)
        allw = [wall(steps, v[0]) for v in b["inwin"].values()]
        qd = [v[1] - (v[2] or 0) for v in b["inwin"].values() if v[2]]
        per = {"%s,%s" % k: round(wall(steps, b["inwin"][k][0]) or 0.0, 1) for k in keys}
        d = dict(name=b["name"], arm=b["arm"], inwin=len(b["inwin"]), matched=len(keys), steps=len(js), sum_ms=round(s, 1),
                 mean_ms=round(s / len(js), 1) if js else None, nowall=bad, untied=b["untied"], done=b["ndone"],
                 sum_all_inwin_ms=round(sum(w for w in allw if w is not None), 1), q_minus_prompt=[min(qd), max(qd)] if qd else None, per=per)
        res["blocks"].append(d)
    arms = []
    for d in res["blocks"]:
        if d["arm"] not in arms: arms.append(d["arm"])
    for a in arms:
        x = [d for d in res["blocks"] if d["arm"] == a]
        sums = [d["sum_ms"] for d in x]
        m = statistics.mean(sums)
        res["arms"][a] = dict(blocks=[d["name"] for d in x], sums=sums, mean_sum_ms=round(m, 1),
                              mean_step_ms=round(m / x[0]["steps"], 1) if x[0]["steps"] else None,
                              spread_pct=round(100.0 * (max(sums) - min(sums)) / m, 2) if len(sums) > 1 and m else None)
    a0 = arms[0]
    for a in arms[1:]:
        r0, r1 = res["arms"][a0]["mean_sum_ms"], res["arms"][a]["mean_sum_ms"]
        res["arms"][a].update(vs=a0, d_ms=round(r1 - r0, 1), d_pct=round(100.0 * (r1 / r0 - 1.0), 2) if r0 else None,
                              d_step_ms=round((r1 - r0) / len(keys), 2) if keys else None)
    for d in res["blocks"]:
        print("NCMATCH %s block %s (%s): %d newcomer steps in its window (%d done, %d not tied to a step), %d matched -> %.1f ms over %d step(s) (%s ms a step)%s "
              "| every in-window newcomer step, unclipped: %.1f ms | q - prompt %s" % (
                  label, d["name"], d["arm"], d["inwin"], d["done"], d["untied"], d["matched"], d["sum_ms"], d["steps"], d["mean_ms"],
                  (" -- VIRHE %d matched step(s) without a wall" % d["nowall"]) if d["nowall"] else "", d["sum_all_inwin_ms"], d["q_minus_prompt"]))
    parts = []
    for a in arms:
        r = res["arms"][a]
        parts.append("%s %.1f ms (%s; %s ms a newcomer step%s)%s" % (
            a, r["mean_sum_ms"], "/".join("%.1f" % v for v in r["sums"]), r["mean_step_ms"],
            (", spread %.2f %%" % r["spread_pct"]) if r.get("spread_pct") is not None else "",
            (" | %s vs %s: %+.1f ms (%+.2f %%, %+.2f ms a step)" % (a, r["vs"], r["d_ms"], r["d_pct"], r["d_step_ms"])) if "vs" in r else ""))
    few = len(keys) < mn
    print("NCMATCH %s: %d matched prompts (newcomer step inside EVERY block's window; in-window per block %s) -- the same prompts' newcomer-step walls "
          "(next step line - its own, unclipped) | %s%s" % (label, len(keys), "/".join(str(d["inwin"]) for d in res["blocks"]), " || ".join(parts),
                                                            (" -- VIRHE fewer than %d matched prompts" % mn) if few else ""))
    if jout: json.dump(res, open(jout, "w"))
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix="ncmatch-")
    t00 = 1794517000.0
    L = []
    def iso(t):
        f = "%.9f" % (t - int(t))
        return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "." + f[2:] + "Z"
    def blk(nm, arm, t0, nc_ms, n_nc, edge_in):
        """decode steps of 35 ms, a newcomer (one MIXED step, nc_ms) every ~2.5 s; window from 0.4 s to 0.3 s after the n_nc-th newcomer
        step's start, or (edge_in) 3 ms after the (n_nc + 1)-th one's start (it starts inside, its wall runs past the end)"""
        rq = []; nct = []
        t = t0; step = 0; k = 0
        tnext = t0 + 0.5
        while t < t0 + 2.5 * (n_nc + 3):
            if t >= tnext:
                L.append("%s I step %d MIXED n_seq=8 n_tok=1007  [0] slot=1 q=1 ctx=900  [7] slot=8 q=996 ctx=0" % (iso(t), step))
                rq.append(dict(ev="done", w=k % 8, k=k // 8, t_send=t - 0.3, t_first=t + nc_ms / 1000.0 + 0.002, prompt=1018, cached=0, err=None))
                nct.append(t); t += nc_ms / 1000.0; tnext += 2.5; k += 1
            else:
                L.append("%s I step %d decode n_seq=8 n_tok=8  [0] slot=1 q=1 ctx=900" % (iso(t), step)); t += 0.035
            step += 1
        rq.append(dict(ev="mark", k=0, t=t0 + 0.4)); rq.append(dict(ev="mark", k=1, t=(nct[n_nc] + 0.003) if edge_in else (nct[n_nc - 1] + 0.3)))
        p = os.path.join(d, nm + "-reqs.jsonl")
        with open(p, "w") as f:
            for r in rq: f.write(json.dumps(r) + "\n")
        return "%s:%s:%s" % (nm, arm, p)
    b1 = blk("c8-1-G-B", "B", t00, 533.0, 13, False)
    b2 = blk("c8-2-G-Z", "Z", t00 + 100, 518.0, 13, False)
    b3 = blk("c8-3-G-Z", "Z", t00 + 200, 518.0, 13, True)   # a 14th newcomer starts 3 ms before this window's end
    b4 = blk("c8-4-G-B", "B", t00 + 300, 533.0, 13, False)
    lp = os.path.join(d, "eng.log.gz")
    with gzip.open(lp, "wt") as f: f.write("\n".join(L) + "\n")
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf): rc = run([lp, "--set", "c8g", "--block", b1, "--block", b2, "--block", b3, "--block", b4, "--json", os.path.join(d, "o.json")])
    t = buf.getvalue()
    o = json.load(open(os.path.join(d, "o.json")))
    why = []
    if rc != 0: why.append("rc %s" % rc)
    if len(o["matched"]) != 13: why.append("matched %d (want 13)" % len(o["matched"]))
    if [b["inwin"] for b in o["blocks"]] != [13, 13, 14, 13]: why.append("in-window %s" % [b["inwin"] for b in o["blocks"]])
    if abs(o["arms"]["B"]["mean_sum_ms"] - 13 * 533.0) > 1.0 or abs(o["arms"]["Z"]["mean_sum_ms"] - 13 * 518.0) > 1.0: why.append("sums %s" % o["arms"])
    if abs(o["arms"]["Z"]["d_step_ms"] + 15.0) > 0.1: why.append("d_step %s" % o["arms"]["Z"].get("d_step_ms"))
    if "13 matched prompts" not in t or "Z vs B: -195.0 ms" not in t: why.append("line")
    if why:
        print(t); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: newcomer steps tied by send / first token, unclipped walls, the matched set excludes the 14th newcomer one block pulled inside (13 of 13 / 13 / 14 / 13), arm sums, Z vs B per step")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
