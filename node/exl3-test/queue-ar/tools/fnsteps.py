#!/usr/bin/env python3
"""fnsteps.py LOG(.gz) --blocks BLOCKS.jsonl --label L[:SEQS:ROWS] [...] [--json OUT] (13.11.2026, queue-ar, item 215) -- small
decode levers judged on the STEP LOG of the full-row bucket (coordinator 12.11.: 64-row steps at c16, 32-row at c8), from a
`docker logs -t` engine log with RADIANCE_LOG_STEPS=1. A step's ms = the next step line's time - its own (the next line must be
step N + 1 and within 5 s). Each abblocks2 block of label L owns the steps from its applied_step (the step its switches landed
at) up to the next "fn4: switches ... applied at step" line of the log. Per block: the decode steps with n_seq = SEQS and
n_tok = ROWS (default 16 / 64; c8 -> 8 / 32): count, share of the block's decode steps, p50, mean, 10-90 % trimmed mean. Per
arm the mean of the block medians and of the trimmed means; B/A of both; per round (A B B A) (B1 + B2) / (A1 + A2) of the
medians; "sep" B<A / B>A when every B block's median is below / above every A block's, else overlap. Also the engagement
counter (RADIANCE_FN4_FORMCTR=1): the cumulative "fn4 formctr dev D ..." lines, the last one before the block's "applied at
step" line and the last one before the next applied line (a switch dumps the counts just before it lands), per block the calls a
form took a device in between (forms recorded during the block).
Lines "FNSTEPS ...", "FNSTEPSB ...", "FNFORM ...". Standard library only."""
import gzip, json, re, statistics, sys
from collections import defaultdict

STEP = re.compile(r"^(\S+)\s+I step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
APPL = re.compile(r"^(\S+)\s+I fn4: switches .* applied at step (\d+) \(epoch (\d+)\)")
FORM = re.compile(r"^(\S+)\s+fn4 formctr dev (\d+) readv (-?\d+) hc_bi (-?\d+):(.*)$")
FITEM = re.compile(r"\s*(\w+) (\d+)/([0-9.]+)/(\d+) bd (\d+);")


def opn(p):
    with open(p, "rb") as f: magic = f.read(2)
    return gzip.open(p, "rt", errors="replace") if magic == b"\x1f\x8b" else open(p, errors="replace")


def tsec(s):
    # 2026-11-12T20:01:34.123456789Z -> seconds (day-relative is enough inside one log; keep the date for safety)
    m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)(?:\.(\d+))?", s)
    if not m: return None
    y, mo, d, h, mi, se = (int(x) for x in m.groups()[:6])
    frac = float("0." + m.group(7)) if m.group(7) else 0.0
    return ((d * 24 + h) * 60 + mi) * 60 + se + frac


def pq(v, p):
    if not v: return None
    v = sorted(v); k = (len(v) - 1) * p; i = int(k); j = min(i + 1, len(v) - 1)
    return round(v[i] + (v[j] - v[i]) * (k - i), 3)


def trim(v, f=0.1):
    if not v: return None
    v = sorted(v); k = int(len(v) * f)
    w = v[k:len(v) - k] or v
    return round(statistics.mean(w), 3)


def read_log(path):
    steps, appl, forms = [], [], []   # steps: (line, step, phase, n_seq, n_tok, t); appl: (line, step, epoch); forms: (line, t, dev, readv, hcbi, {form: calls})
    with opn(path) as f:
        for n, l in enumerate(f):
            l = l.rstrip("\n")
            m = STEP.match(l)
            if m:
                steps.append((n, int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)), tsec(m.group(1)))); continue
            m = APPL.match(l)
            if m: appl.append((n, int(m.group(2)), int(m.group(3)))); continue
            m = FORM.match(l)
            if m:
                fm = {k: (int(c), float(r), int(mx), int(bd)) for k, c, r, mx, bd in FITEM.findall(m.group(5))}
                forms.append((n, tsec(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), fm))
    return steps, appl, forms


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"): print(__doc__); return 2
    path, blocks, labels, jout = a[0], None, [], None
    i = 1
    while i < len(a):
        if a[i] == "--blocks": blocks = a[i + 1]; i += 2
        elif a[i] == "--label": labels.append(a[i + 1]); i += 2
        elif a[i] == "--json": jout = a[i + 1]; i += 2
        else: i += 1
    steps, appl, forms = read_log(path)
    if not steps: print("FNSTEPS VIRHE no step lines (RADIANCE_LOG_STEPS=1 and docker logs -t?)"); return 1
    # a step's ms
    ms = {}
    for k in range(len(steps) - 1):
        s, t = steps[k], steps[k + 1]
        if t[1] == s[1] + 1 and s[5] is not None and t[5] is not None and 0 < t[5] - s[5] <= 5.0:
            ms[s[1]] = 1000.0 * (t[5] - s[5])
    bounds = sorted({st for _, st, _ in appl})
    recs = [json.loads(l) for l in open(blocks) if l.strip()] if blocks else []
    out = {}
    for spec in labels:
        p = spec.split(":")
        lab = p[0]; seqs = int(p[1]) if len(p) > 1 and p[1] else 16; rows = int(p[2]) if len(p) > 2 and p[2] else 4 * seqs
        bl = [r for r in recs if r.get("label") == lab and r.get("ok") and r.get("applied_step") is not None]
        res = {"seqs": seqs, "rows": rows, "blocks": []}
        for r in bl:
            s0 = int(r["applied_step"]); nxt = [b for b in bounds if b > s0]; s1 = nxt[0] if nxt else 1 << 40
            dec = [s for s in steps if s0 <= s[1] < s1 and s[2] == "decode"]
            full = [ms[s[1]] for s in dec if s[3] == seqs and s[4] == rows and s[1] in ms]
            # engagement: formctr lines by the "applied at step" lines (a step's log line comes before the switches it runs
            # under land, and the counts up to a switch are dumped just before its applied line): the block's counts = the last
            # dump before its applied line -> the last dump before the next applied line
            l0 = next((ln for ln, st, _ in appl if st == s0), None); l1 = next((ln for ln, st, _ in appl if st > s0), 1 << 40)
            fdel = {}
            if l0 is not None and forms:
                for dev in sorted({f[2] for f in forms}):
                    fb = [f for f in forms if f[2] == dev and f[0] < l0]; fi = [f for f in forms if f[2] == dev and l0 < f[0] < l1]
                    base = fb[-1][5] if fb else {}; last = fi[-1][5] if fi else base
                    d = {k: v[0] - base.get(k, (0,))[0] for k, v in last.items() if v[0] - base.get(k, (0,))[0] > 0}
                    fdel[dev] = {"readv": (fi[-1][3] if fi else (fb[-1][3] if fb else None)), "calls": d}
            b = {"arm": r.get("arm"), "round": r.get("round"), "epoch": r.get("epoch"), "step0": s0, "step1": s1 if s1 < 1 << 40 else None,
                 "readv": (r.get("applied") or {}).get("hc_readv"), "decode_steps": len(dec), "full_n": len(full),
                 "full_share": round(len(full) / max(1, len(dec)), 3), "p50": pq(full, 0.5), "mean": round(statistics.mean(full), 3) if full else None,
                 "trim": trim(full), "mb_tps": r.get("tps"), "mb_ms": r.get("ms"), "forms": fdel}
            res["blocks"].append(b)
        A = [b for b in res["blocks"] if b["arm"] == "A" and b["p50"] is not None]
        B = [b for b in res["blocks"] if b["arm"] == "B" and b["p50"] is not None]
        if A and B:
            ma, mb_ = statistics.mean(b["p50"] for b in A), statistics.mean(b["p50"] for b in B)
            ta, tb = statistics.mean(b["trim"] for b in A), statistics.mean(b["trim"] for b in B)
            rounds = defaultdict(lambda: {"A": [], "B": []})
            for b in res["blocks"]:
                if b["p50"] is not None: rounds[b["round"]][b["arm"]].append(b["p50"])
            rr = [round(sum(v["B"]) / sum(v["A"]), 4) for k, v in sorted(rounds.items()) if len(v["A"]) == 2 and len(v["B"]) == 2]
            sep = "B<A" if max(b["p50"] for b in B) < min(b["p50"] for b in A) else "B>A" if min(b["p50"] for b in B) > max(b["p50"] for b in A) else "overlap"
            res.update(A_p50=round(ma, 3), B_p50=round(mb_, 3), ratio_p50=round(mb_ / ma, 4), A_trim=round(ta, 3), B_trim=round(tb, 3),
                       ratio_trim=round(tb / ta, 4), round_ratios=rr, sep=sep)
        out[lab] = res
        print("FNSTEPS %s (n_seq %d, %d rows): full-row step ms B/A p50 %s trim %s %s | A p50 %s trim %s | B p50 %s trim %s | rounds %s | blocks %s" % (
            lab, seqs, rows, res.get("ratio_p50"), res.get("ratio_trim"), res.get("sep"), res.get("A_p50"), res.get("A_trim"),
            res.get("B_p50"), res.get("B_trim"), res.get("round_ratios"),
            " ".join("%s:%s/%s(n%d,%.0f%%)" % (b["arm"], b["p50"], b["trim"], b["full_n"], 100 * b["full_share"]) for b in res["blocks"])))
        for b in res["blocks"]:
            print("FNSTEPSB %s %s r%s epoch %s steps %s-%s readv %s: full-row n %d (%.0f %% of %d decode) p50 %s mean %s trim %s | mb %s tok/s %s ms" % (
                lab, b["arm"], b["round"], b["epoch"], b["step0"], b["step1"], b["readv"], b["full_n"], 100 * b["full_share"], b["decode_steps"],
                b["p50"], b["mean"], b["trim"], b["mb_tps"], b["mb_ms"]))
            for dev, f in sorted(b["forms"].items()):
                print("FNFORM %s %s r%s dev %d readv %s: %s" % (lab, b["arm"], b["round"], dev, f["readv"],
                      " ".join("%s %d" % (k, v) for k, v in sorted(f["calls"].items())) or "no new calls"))
    if forms:
        last = {}
        for f in forms: last[f[2]] = f
        for dev, f in sorted(last.items()):
            print("FNFORM total dev %d readv %d hc_bi %d: %s" % (dev, f[3], f[4], " ".join("%s %d/%.1f/%d bd %d" % (k, *v) for k, v in sorted(f[5].items()))))
    else:
        print("FNFORM none (no fn4 formctr lines: RADIANCE_FN4_FORMCTR=1 on an image with the counter?)")
    if jout:
        with open(jout, "w") as f: json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
