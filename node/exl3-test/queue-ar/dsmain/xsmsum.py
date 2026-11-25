#!/usr/bin/env python3
"""xsmsum.py (10.11.2026, queue-ar dsmain, ds-main-port small-levers agent) -- the summary of item 57-p4zyzzzzz-ds-*-small (body
dsm-small.sh). Reads the body's plan file (one "PLAN key=value ..." line a start), the block rows (vlblocks.py jsonl), the 136k
deep json files and identity.txt; prints

  START <nst> B vs A ...        the combined levers in process: per metric A and B (mean, n), pooled B / A, the ABBA pairs'
                                B / A (mean +- t95, n), B's distance to the target; the c4 / c8 draft counters per arm (NST 5);
  DEEP <nst> ...                the 136k cold prefill under A and B: TTFT (cached must be 0), B vs A, both vs dsm-225;
  LOO <nst> lever ...           leave-one-out (when run): (B - lever) vs the adjacent B blocks, per metric on the phases run;
  ENV <nst> ...                 a follow-up start's B (env levers on) vs the main start's B (cross-start, labelled so);
  STAMP ...                     the stamped c4 screen of dspark_conf (W0 0, W1 1, W2 101, W3 0): per window ms a step by
                                kernel family group (d0attrprof.py stamps JSON: every pass kind, device us a pass x passes a
                                step), card 0 idle a step, the step period, tok a step and tok/s (the window rows); W1 / W2
                                against the mean of W0 and W3;
  CTX ...                       the context-skip refill check under forced depth 0 (when run): ctxskip 1 vs 0, essay / code
                                tok/s, acceptance, probe cycles;
  IDENTITY ...                  identity.txt counted;
  SUMMARY ...                   one line.
  xsmsum.py PLANFILE IDENTITY.txt DEEPREF.json
  xsmsum.py --decide BLOCKS.jsonl LOSS_PCT PHASES   -> "LOSS phases=<p,..> (<metric pooled %>...)" or "LOSS none (...)": a
                                headline metric where B loses >= LOSS_PCT pooled AND every A / B pair shows the loss
Settle blocks and blocks not ok are left out. Standard library only."""
import json, math, re, statistics, sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
TARGET = {"1u essay": 90.0, "1u code": 90.0, "1u edit": 90.0, "c4 aggregate": 160.0, "c8 aggregate": 200.0}


def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except (OSError, ValueError): return []


def js(p):
    try: return json.load(open(p))
    except Exception: return {}


def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def path(*ks):
    def f(res):
        x = res
        for k in ks:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return num(x)
    return f


def c1(name, k):
    def f(res):
        c = res.get("c1") or {}
        return num(c[name].get(k)) if isinstance(c.get(name), dict) else None
    return f


# (name, getter, phase it comes from, higher is better)
M = [("1u essay", c1("essay", "decode_tps"), "c1", True), ("1u code", c1("code", "decode_tps"), "c1", True),
     ("1u edit", path("edit", "decode_tps"), "edit", True),
     ("c4 aggregate", path("c4", "agg_dec_tps"), "c4", True), ("c4 p50", path("c4", "per_stream_dec_p50"), "c4", True),
     ("c8 aggregate", path("c8", "agg_dec_tps"), "c8", True), ("c8 p50", path("c8", "per_stream_dec_p50"), "c8", True),
     ("ms essay", c1("essay", "ms_per_decode_step"), "c1", False), ("ms code", c1("code", "ms_per_decode_step"), "c1", False),
     ("ms c4", path("c4", "ms_decode_step"), "c4", False), ("ms c8", path("c8", "ms_decode_step"), "c8", False),
     ("essay acc", c1("essay", "acceptance"), "c1", True), ("code acc", c1("code", "acceptance"), "c1", True)]
HEAD = ("1u essay", "1u code", "1u edit", "c4 aggregate", "c8 aggregate")
DC = [("%s %s" % (p, k), path("draftph", p, k)) for p in ("c1", "c4", "c8") for k in ("zero_cycles", "probe_cycles", "zero_moves")]


def good(r): return r.get("ok") and not r.get("settle")


def vals(rs, arm, get): return [v for v in (get(r["res"]) for r in rs if good(r) and r.get("arm") == arm) if v is not None]


def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None)


def mean(v): return statistics.mean(v) if v else None
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pct(v, r): return ("%+.2f %%" % ((v / r - 1) * 100)) if v is not None and r else "-"


def tgt(v, t, low=False):
    if t is None or v is None: return ""
    if low:
        return "; target <= %g s: %s" % (t, "MET" if v <= t else "%.1f s over" % (v - t))
    return "; target >= %g: %s" % (t, "MET" if v >= t else "%.1f short (+%.1f %% needed)" % (t - v, (t / v - 1) * 100))


def pairs(rs, x, y, get):
    """x / y per adjacent (x, y) block pair: the measured blocks in order, cut into pairs holding one x and one y."""
    meas = [r for r in rs if not r.get("settle")]
    out, i = [], 0
    while i + 1 < len(meas):
        a, b = meas[i], meas[i + 1]
        arms = {a.get("arm"), b.get("arm")}
        if arms == {x, y}:
            rx, ry = (a, b) if a.get("arm") == x else (b, a)
            if good(rx) and good(ry):
                vx, vy = get(rx["res"]), get(ry["res"])
                if vx is not None and vy: out.append(vx / vy)
            i += 2
        else:
            i += 1
    return out


def compare(rs, x, y, metrics, phases=None):
    """per metric: y mean, x mean, pooled x / y, pairs; [(name, line, pooled ratio or None, better bool)]"""
    out = []
    for nm, get, ph, hib in metrics:
        if phases and ph not in phases: continue
        vy, vx = vals(rs, y, get), vals(rs, x, get)
        if not vy and not vx: continue
        my, mx = mean(vy), mean(vx)
        pr = pairs(rs, x, y, get); pm, ph_ = ci(pr)
        pooled = mx / my if mx is not None and my else None
        line = "%s %s %s (n %d) vs %s %s (n %d): pooled %s, pairs %s" % (
            nm, x, f1(mx, 2), len(vx), y, f1(my, 2), len(vy), pct(mx, my),
            "-" if pm is None else "%+.2f %%%s (n %d: %s)" % ((pm - 1) * 100, "" if ph_ is None else " +- %.2f" % (ph_ * 100),
                                                             len(pr), " ".join("%+.2f" % ((p - 1) * 100) for p in pr)))
        out.append((nm, line, pooled, hib, pr))
    return out


GROUPS = [("all-reduce", re.compile(r"(^|_)ar_|all_reduce|allreduce", re.I)),
          ("expert kernels", re.compile(r"exl3|moe|expert", re.I)),
          ("attention / indexer", re.compile(r"attn|index|idx|qsa|sparse|topk|sel_", re.I)),
          ("dense GEMM", re.compile(r"gemm|gemv|gg_mr|gg_tiled|gg_wm|matmul", re.I))]


def group(fam):
    for g, rx in GROUPS:
        if rx.search(fam): return g
    return "other"


def stamp(P, summ):
    sj, wr = js(P.get("stamps", "")), rows(P.get("win", ""))
    tag = P.get("tag", "sm-stamp")
    conf = {"w0": 0, "w1": 1, "w2": 101, "w3": 0}
    out = {}
    for w in ("w0", "w1", "w2", "w3"):
        x = sj.get(w) or {}
        row = next((r for r in reversed(wr) if r.get("label") == "%s-%s" % (tag, w)), {})
        mb = row.get("mb") or {}
        steps = x.get("steps") or 0
        g = {}
        for k, v in (x.get("kinds") or {}).items():
            n = v.get("passes") or 0
            for fam, us in (v.get("fam_us") or {}).items():
                g[group(fam)] = g.get(group(fam), 0.0) + us * n
        if steps: g = {k: v / steps / 1000.0 for k, v in g.items()}
        out[w] = {"groups": g, "idle": (x.get("idle_us_mean") or 0) / 1000.0 if x else None, "period": (x.get("period_us_mean") or 0) / 1000.0 if x else None,
                  "steps": steps, "tps": num(mb.get("agg_dec_tps")), "tok_step": num(mb.get("tok_per_decode_step")), "ms_dec": num(mb.get("ms_decode_step"))}
        if x or mb:
            print("STAMP c4 %s dspark_conf %d: period %s ms a step (%d steps), card 0 idle %s ms; %s | tok a step %s, %s ms a decode step, %s tok/s" % (
                w, conf[w], f1(out[w]["period"], 2), steps, f1(out[w]["idle"], 2),
                ", ".join("%s %s ms" % (k, f1(g.get(k), 2)) for k in [n for n, _ in GROUPS] + ["other"]),
                f1(out[w]["tok_step"], 2), f1(out[w]["ms_dec"], 2), f1(out[w]["tps"], 1)))
    base = [out[w] for w in ("w0", "w3") if out.get(w) and out[w]["steps"]]
    if not base: return
    def bm(key, sub=None):
        v = [(b["groups"].get(sub) if sub else b[key]) for b in base]
        v = [x for x in v if x is not None]
        return statistics.mean(v) if v else None
    for w in ("w1", "w2"):
        o = out.get(w)
        if not o or not o["steps"]: continue
        parts = []
        for k in [n for n, _ in GROUPS] + ["other"]:
            a, b = o["groups"].get(k), bm(None, k)
            if a is not None and b is not None: parts.append("%s %+.2f ms" % (k, a - b))
        di = (o["idle"] - bm("idle")) if o["idle"] is not None and bm("idle") is not None else None
        dp = (o["period"] - bm("period")) if o["period"] is not None and bm("period") is not None else None
        line = "dspark_conf %d vs 0 (mean of w0 / w3) at c4: period %s ms a step (%s), card 0 idle %s ms; by family %s; tok a step %s; tok/s %s" % (
            conf[w], "-" if dp is None else "%+.2f" % dp, pct(o["period"], bm("period")), "-" if di is None else "%+.2f" % di, ", ".join(parts),
            pct(o["tok_step"], bm("tok_step")), pct(o["tps"], bm("tps")))
        print("STAMP c4 %s %s" % (w, line))
        summ.append("stamped c4 %s" % line)


def plan(p):
    out = []
    try:
        for l in open(p):
            if l.startswith("PLAN "): out.append(dict(kv.split("=", 1) for kv in l[5:].split() if "=" in kv))
    except OSError: pass
    return out


def decide(blocks, loss, phases):
    rs = rows(blocks)
    lost, why = [], []
    for nm, line, pooled, hib, pr in compare(rs, "B", "A", [m for m in M if m[0] in HEAD], phases.split(",")):
        if pooled is None: continue
        d = (pooled - 1) * 100 if hib else (1 - pooled) * 100
        same = bool(pr) and all((p < 1.0) if hib else (p > 1.0) for p in pr)
        why.append("%s %+.2f %% (pairs %s)" % (nm, d, "/".join("%+.1f" % ((p - 1) * 100) for p in pr) or "-"))
        if d <= -loss and same:
            ph = next(m[2] for m in M if m[0] == nm)
            if ph not in lost: lost.append(ph)
    print("LOSS phases=%s (%s)" % (",".join(lost) if lost else "none", "; ".join(why) or "no B / A blocks"))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--decide":
        return decide(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    pl, ident, dref = plan(sys.argv[1]), sys.argv[2], js(sys.argv[3])
    try: idl = [l.strip() for l in open(ident) if l.strip()]
    except OSError: idl = []
    rt = num(dref.get("ttft"))
    summ = []
    mainB = {}
    for P in pl:
        nst, kind = P.get("nst"), P.get("kind")
        if kind == "stamp":
            stamp(P, summ); continue
        rs = rows(P.get("blocks", ""))
        if kind == "main":
            mainB[nst] = rs
            cmp = compare(rs, "B", "A", M)
            for nm, line, pooled, hib, pr in cmp:
                if nm in TARGET:
                    line += tgt(mean(vals(rs, "B", next(m[1] for m in M if m[0] == nm))), TARGET.get(nm))
                print("START %s B vs A (levers %s, in process): %s" % (nst, P.get("levers"), line))
            if nst == "n5":
                for arm in ("A", "B"):
                    print("START %s draft counters arm %s (sum over blocks): %s" % (nst, arm, "; ".join(
                        "%s %s" % (n, f1(sum(vals(rs, arm, g)) if vals(rs, arm, g) else None, 0)) for n, g in DC)))
            hl = ", ".join("%s %s" % (nm, "-" if pooled is None else "%+.2f %%%s" % ((pooled - 1) * 100,
                           "" if len(pr) < 2 else " (pairs %s)" % "/".join("%+.1f" % ((p - 1) * 100) for p in pr)))
                           for nm, line, pooled, hib, pr in cmp if nm in HEAD)
            bad = sum(1 for r in rs if not r.get("ok"))
            summ.append("%s B vs A: %s%s" % (nst, hl or "no blocks", "; %d block(s) not ok" % bad if bad else ""))
            # deep
            dA, dB = js(P.get("deepA", "")), js(P.get("deepB", ""))
            if dA or dB:
                cold = lambda d: num(d.get("ttft")) if d.get("cached") == 0 else None
                ta, tb = cold(dA), cold(dB)
                print("DEEP %s 136k cold prefill: A TTFT %s s (cached %s), B %s s (cached %s): B vs A %s; vs dsm-225 %s: A %s, B %s%s" % (
                    nst, f1(ta, 2), dA.get("cached"), f1(tb, 2), dB.get("cached"), pct(tb, ta), f1(rt, 2), pct(ta, rt), pct(tb, rt),
                    tgt(tb, 35.0, True)))
                summ.append("%s 136k TTFT A %s / B %s s (%s)" % (nst, f1(ta, 2), f1(tb, 2), pct(tb, ta)))
            # leave-one-out
            lo = rows(P.get("loo", ""))
            if lo:
                levers = P.get("levers", "").split(",")
                ph = sorted({ph for r in lo for ph in (r.get("res") or {}) if ph in ("c1", "edit", "c4", "c8")})
                lsum = []
                for k, lv in enumerate(levers, 1):
                    cm = compare(lo, str(k), "B", M, ph)
                    for nm, line, pooled, hib, pr in cm:
                        print("LOO %s %s (B without %s) vs B: %s" % (nst, k, lv, line))
                    lsum.append("-%s: %s" % (lv, ", ".join("%s %s" % (nm, pct(pooled, 1.0) if pooled else "-") for nm, line, pooled, hib, pr in cm if nm in HEAD)))
                summ.append("%s leave-one-out (B - lever vs B): %s" % (nst, " | ".join(lsum)))
            # context-skip refill check
            cx = rows(P.get("ctx", ""))
            if cx:
                cm = compare(cx, "B", "A", [m for m in M if m[2] == "c1"])
                for nm, line, pooled, hib, pr in cm:
                    print("CTX %s forced depth 0, draft_ctxskip 1 (B) vs 0 (A): %s" % (nst, line))
                for arm in ("A", "B"):
                    print("CTX %s arm %s c1 draft counters: %s" % (nst, arm, "; ".join(
                        "%s %s" % (k, f1(sum(vals(cx, arm, path("draftph", "c1", k))) if vals(cx, arm, path("draftph", "c1", k)) else None, 0))
                        for k in ("zero_cycles", "probe_cycles"))))
                summ.append("%s refill check (forced depth 0, skip on vs off): %s" % (nst, ", ".join(
                    "%s %s" % (nm, pct(pooled, 1.0) if pooled else "-") for nm, line, pooled, hib, pr in cm if nm in ("essay acc", "code acc", "1u essay", "1u code"))))
        else:
            ref = mainB.get(nst, [])
            both = [dict(r, arm="E") for r in rs] + [r for r in ref if r.get("arm") == "B"]
            cm = compare(both, "E", "B", M)
            for nm, line, pooled, hib, pr in cm:
                print("ENV %s env levers %s on top of B (follow-up start, E) vs the main start's B (CROSS-START, pooled only): %s" % (nst, P.get("env"), re.sub(r", pairs .*$", "", line)))
            def units(f):
                for t in ((js(f).get("experts") or {}).get("tiers") or []):
                    if t.get("name") == "vram": return t.get("units")
                return None
            ue, ub = units(P.get("stats0", "")), units(P.get("mainstats0", ""))
            print("ENV %s routed units in VRAM at start (rank 0 /stats): env start %s vs main start %s (%s)" % (
                nst, ue, ub, "-" if ue is None or ub is None else "%+d" % (ue - ub)))
            dB = js(P.get("deepB", ""))
            if dB:
                print("ENV %s 136k cold prefill under B + env: TTFT %s s (cached %s) vs dsm-225 %s s (%s)" % (
                    nst, f1(num(dB.get("ttft")), 2), dB.get("cached"), f1(rt, 2), pct(num(dB.get("ttft")), rt)))
            summ.append("%s env [%s] units %s vs %s at start" % (nst, P.get("env"), ue, ub))
            summ.append("%s env [%s] (cross-start) vs B: %s" % (nst, P.get("env"), ", ".join(
                "%s %s" % (nm, pct(pooled, 1.0) if pooled else "-") for nm, line, pooled, hib, pr in cm if nm in HEAD)))
    n_id = sum(1 for l in idl if " IDENTICAL" in l or l.endswith(" CLEAN"))
    n_diff = sum(1 for l in idl if " DIFFERENT" in l or " VIRHE" in l)
    n_nc = sum(1 for l in idl if "not compared" in l)
    print("IDENTITY %d identical / clean, %d different, %d not compared: %s" % (n_id, n_diff, n_nc, " | ".join(idl)[:2500]))
    summ.append("identity %d identical, %d DIFFERENT, %d not compared" % (n_id, n_diff, n_nc))
    print("SUMMARY " + " || ".join(summ))


if __name__ == "__main__":
    main()
