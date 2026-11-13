#!/usr/bin/env python3
"""c8mix.py (12.11.2026, queue-ar/dsmain/harness, ds-main-port harness builder) -- the newcomer co-batch MIX of an mb c8 block
from the engine step log, printed next to the block's agg tok/s (c8 block-mix note 12.11.; the SP item's "c8 step walls").

The step log = `docker logs -t` of a RADIANCE_LOG_STEPS=1 server: "step K prefill|decode|MIXED n_seq= n_tok=  [i] slot= q= ctx="
(regexes and wall rule copied from nc/ncsteps.py, read only: a step's wall = the next step line's receipt time - its own,
gaps over --idle s dropped). Window = the block's two marks (mbt2.py / ncmb.py --reqs rows {"ev": "mark", "k": 0 | 1}) or
--t0 / --t1. In the window, by plan time:
  newcomer step = a non-decode step with a FRESH entry (q >= 256, ctx <= 64; ncsteps.py's rule);
    single     = one fresh prompt in the step,   co-batched = two or more (sizes = fresh prompts a step);
    per class: count, rows (n_tok) min-max, wall mean / p50 ms;
  other non-decode steps (no fresh entry: chunks of a longer prompt), decode steps (count, wall mean), DRAFTED decode steps
  (an entry with q > 1: the start-of-server probe / low-occupancy drafting -- they shift streams in the step domain);
  span / full = non-decode steps of <= 2112 / 2113..4160 rows (the SP item's classes).
Signature = (co-batched sizes sorted, single count). Two blocks' mixes MATCH when the signatures are equal.

  c8mix.py block --log L [--reqs R.jsonl --tag T | --t0 T0 --t1 T1] [--mbline F.txt] [--label X] [--json OUT]
      (VIRHE only when the window has no step line; drafted decode steps / chunk steps are a NOTE)
      -> "C8MIX X: agg A tok/s, decode_share D | co-batched N [sizes] rows a-b wall m ms | single N rows a-b wall m ms |
          other non-decode N | decode N (m ms), drafted N | span N / full N | window W s"
  c8mix.py compare --in NAME=BLOCK.json [...]   every block's signature vs the first one's -> "C8MIXCMP ... MATCH | DIFFERS";
      with the agg tok/s delta and, under 2 % with differing mixes, "INCONCLUSIVE (mix)" (TESTING.md c8 block-mix rule)
  c8mix.py selftest
Standard library only; reads .gz or plain files.
"""
import argparse, calendar, gzip, json, math, re, statistics, sys, time

sys.dont_write_bytecode = True
STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")    # ncsteps.py's (longid.py's)
ENT = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")
TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) ")
FRESH_Q, FRESH_CTX, SPAN, FULL = 256, 64, 2112, 4160


def opn(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def ts_split(line):
    m = TSRE.match(line)
    if not m: return None, line
    t = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    z = m.group(3)
    if z != "Z": t -= (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[4:6]) * 60)
    return t, line[m.end():]


def read_steps(path, idle=3.0, t0=None, t1=None):
    out = []
    with opn(path) as f:
        for l in f:
            if "n_seq=" not in l: continue
            t, rest = ts_split(l.rstrip("\n"))
            m = STEP.search(rest)
            if not m or t is None: continue
            if t1 is not None and t > t1 + 3 * idle + 5: break
            out.append({"t": t, "K": int(m.group(1)), "ph": m.group(2), "n_seq": int(m.group(3)), "n_tok": int(m.group(4)),
                        "ents": [(int(s), int(q), int(c)) for _, s, q, c in ENT.findall(m.group(5))]})
    for a, b in zip(out, out[1:]):
        d = b["t"] - a["t"]
        a["wall_ms"] = 1000.0 * d if 0 <= d <= idle else None
    if out: out[-1]["wall_ms"] = None
    return out


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v: return None
    k = (len(v) - 1) * p / 100.0
    f = int(math.floor(k)); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def cls_stat(ss):
    w = [s.get("wall_ms") for s in ss if s.get("wall_ms") is not None]
    return {"n": len(ss), "rows_min": min((s["n_tok"] for s in ss), default=None), "rows_max": max((s["n_tok"] for s in ss), default=None),
            "wall_mean": round(statistics.mean(w), 1) if w else None, "wall_p50": round(pct(w, 50), 1) if w else None}


def mix(steps, t0, t1):
    W = [s for s in steps if t0 <= s["t"] < t1]
    fresh = lambda s: sum(1 for _, q, c in s["ents"] if q >= FRESH_Q and c <= FRESH_CTX)   # noqa: E731
    nd = [s for s in W if s["ph"] != "decode"]
    single = [s for s in nd if fresh(s) == 1]
    cob = [s for s in nd if fresh(s) >= 2]
    other = [s for s in nd if fresh(s) == 0]
    dec = [s for s in W if s["ph"] == "decode"]
    drafted = [s for s in dec if any(q > 1 for _, q, c in s["ents"])]
    sizes = sorted((fresh(s) for s in cob), reverse=True)
    out = {"window_s": round(t1 - t0, 1), "steps": len(W), "single": cls_stat(single), "cobatched": dict(cls_stat(cob), sizes=sizes),
           "other_nondecode": cls_stat(other), "decode": cls_stat(dec), "drafted_decode": len(drafted),
           "span": sum(1 for s in nd if s["n_tok"] <= SPAN), "full": sum(1 for s in nd if SPAN < s["n_tok"] <= FULL),
           "newcomers": sum(fresh(s) for s in nd), "sequence": "".join(str(fresh(s)) for s in nd if fresh(s))}
    out["signature"] = [sizes, len(single)]
    return out


def marks_of(path, tag):
    mk = {}
    try:
        with opn(path) as f:
            for l in f:
                try: d = json.loads(l)
                except ValueError: continue
                if d.get("ev") == "mark" and (tag is None or d.get("tag") == tag): mk[d["k"]] = d["t"]
    except OSError:
        pass
    return mk


def mbline(path):
    try:
        for l in opn(path):
            if l.startswith("MB "):
                d = json.loads(l[3:])
                if str(d.get("phase", "")).startswith("c"): return d
    except (OSError, ValueError):
        pass
    return {}


def fmt(label, r, mb):
    c, s, d = r["cobatched"], r["single"], r["decode"]
    return ("C8MIX %s: agg %s tok/s, decode_share %s, TTFT p50 %s s | co-batched %d %s rows %s-%s wall %s ms mean | single %d rows %s-%s wall %s ms mean"
            " | other non-decode %d | decode %d (%s ms mean), drafted %d | span %d / full %d | newcomers %d (%s) | window %s s%s") % (
        label, mb.get("agg_dec_tps"), mb.get("decode_share"), mb.get("ttft_p50"), c["n"], c["sizes"], c["rows_min"], c["rows_max"], c["wall_mean"],
        s["n"], s["rows_min"], s["rows_max"], s["wall_mean"], r["other_nondecode"]["n"], d["n"], d["wall_mean"], r["drafted_decode"],
        r["span"], r["full"], r["newcomers"], r["sequence"][:40], r["window_s"],
        ((" | NOTE " + "; ".join(r["note"])) if r.get("note") else "") + ((" -- VIRHE " + "; ".join(r["guard"])) if r.get("guard") else ""))


def cmd_block(a):
    t0, t1 = a.t0, a.t1
    if a.reqs:
        mk = marks_of(a.reqs, a.tag)
        if 0 in mk and 1 in mk: t0, t1 = mk[0], mk[1]
    if t0 is None or t1 is None:
        print("C8MIX %s: VIRHE no window (no marks k 0 / 1 for tag %s in %s and no --t0 / --t1)" % (a.label, a.tag, a.reqs)); return 2
    steps = read_steps(a.log, a.idle, t0, t1)
    r = mix(steps, t0, t1)
    guard, note = [], []
    if not r["steps"]: guard.append("no step-log line in the window (RADIANCE_LOG_STEPS=1?)")
    if r["drafted_decode"]: note.append("%d drafted decode steps (probe / drafting at c8: streams shift in the step domain, the fixed mix can drift)" % r["drafted_decode"])
    if r["other_nondecode"]["n"]: note.append("%d non-decode steps without a fresh prompt" % r["other_nondecode"]["n"])
    r["guard"], r["note"] = guard, note
    mb = mbline(a.mbline) if a.mbline else {}
    r["mb"] = {k: mb.get(k) for k in ("agg_dec_tps", "decode_share", "ttft_p50", "requests", "per_stream_dec_p50", "sched", "pattern", "window", "mix_client")}
    r["label"], r["t0"], r["t1"] = a.label, t0, t1
    line = fmt(a.label, r, mb)
    print(line)
    r["line"] = line
    if a.json:
        with open(a.json, "w") as f: json.dump(r, f)
    return 0


def cmd_compare(a):
    blocks = []
    for x in a.inp:
        nm, p = x.split("=", 1)
        blocks.append((nm, json.load(open(p))))
    ref_nm, ref = blocks[0]
    rc = 0
    for nm, b in blocks:
        same = b["signature"] == ref["signature"]
        ta, tb = (ref.get("mb") or {}).get("agg_dec_tps"), (b.get("mb") or {}).get("agg_dec_tps")
        dl = (100.0 * (tb / ta - 1)) if ta and tb else None
        verdict = "MATCH" if same else ("DIFFERS" + (" -- INCONCLUSIVE (mix): c8 delta %+.1f %% under 2 %% with a different co-batch mix" % dl
                                                      if dl is not None and abs(dl) < 2.0 else ""))
        if not same: rc = 1
        print("C8MIXCMP %s vs %s: co-batched %s + %d single vs %s + %d single | agg %s vs %s (%s) | %s" % (
            nm, ref_nm, b["signature"][0], b["signature"][1], ref["signature"][0], ref["signature"][1], tb, ta,
            ("%+.2f %%" % dl) if dl is not None else "-", verdict))
    return rc


def cmd_selftest(a):
    import os, tempfile
    ok, why = True, []
    base = calendar.timegm((2026, 11, 12, 10, 0, 0, 0, 0, 0))
    lines, t, K = [], base, 0

    def step(ph, ents, wall):
        nonlocal t, K
        lines.append("%s.%06dZ I step %d %s n_seq=%d n_tok=%d" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(t))), int((t % 1) * 1e6), K, ph,
                     len(ents), sum(q for _, q, _ in ents)) + "".join("  [%d] slot=%d q=%d ctx=%d" % (i, s, q, c) for i, (s, q, c) in enumerate(ents)))
        t += wall; K += 1
    dec8 = [(i, 1, 1000 + i) for i in range(8)]
    step("decode", dec8, 0.036)
    step("MIXED", dec8[:7] + [(8, 1100, 0)], 0.600)                       # single
    step("decode", dec8, 0.036)
    step("MIXED", dec8[:5] + [(9, 1000, 0), (10, 1200, 0), (11, 900, 0)], 1.4)   # co-batched 3
    step("decode", [(0, 6, 1000)] + dec8[1:], 0.050)                      # drafted
    step("MIXED", dec8[:6] + [(12, 1100, 0), (13, 1000, 0)], 1.1)          # co-batched 2
    step("prefill", [(14, 2048, 2048)], 0.8)                               # other (a later chunk)
    step("decode", dec8, 0.036)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "e.log")
    with open(p, "w") as f: f.write("\n".join(lines) + "\n")
    r = mix(read_steps(p), base, t)
    if r["single"]["n"] != 1 or r["cobatched"]["n"] != 2 or r["cobatched"]["sizes"] != [3, 2]: ok = False; why.append("classes %s" % r["signature"])
    if r["drafted_decode"] != 1 or r["other_nondecode"]["n"] != 1: ok = False; why.append("drafted / other")
    if abs((r["single"]["wall_mean"] or 0) - 600.0) > 1.0: ok = False; why.append("single wall %s" % r["single"]["wall_mean"])
    if r["full"] != 1 or r["span"] != 3: ok = False; why.append("span / full %s / %s" % (r["span"], r["full"]))
    rq = os.path.join(d, "r.jsonl")
    with open(rq, "w") as f:
        f.write(json.dumps({"ev": "mark", "k": 0, "t": base, "tag": "x"}) + "\n" + json.dumps({"ev": "mark", "k": 1, "t": t, "tag": "x"}) + "\n")
    if marks_of(rq, "x") != {0: base, 1: t}: ok = False; why.append("marks")
    r["guard"], r["note"] = [], ["n"]
    ln = fmt("t", r, {"agg_dec_tps": 1.0})
    if not ln.startswith("C8MIX t: agg 1.0") or "| NOTE n" not in ln: ok = False; why.append("line format")
    for f_ in (p, rq): os.remove(f_)
    os.rmdir(d)
    print("SELFTEST " + ("OK (single / co-batched sizes / drafted / other / wall / span-full / marks / line)" if ok else "FAILED: " + "; ".join(why)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("block")
    p.add_argument("--log", required=True)
    p.add_argument("--reqs")
    p.add_argument("--tag")
    p.add_argument("--t0", type=float)
    p.add_argument("--t1", type=float)
    p.add_argument("--mbline")
    p.add_argument("--label", default="")
    p.add_argument("--idle", type=float, default=3.0)
    p.add_argument("--json")
    p = sub.add_parser("compare")
    p.add_argument("--in", dest="inp", action="append", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"block": cmd_block, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
