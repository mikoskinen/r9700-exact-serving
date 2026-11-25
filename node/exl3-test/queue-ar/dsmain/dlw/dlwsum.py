#!/usr/bin/env python3
"""dlwsum.py (11.11.2026, queue-ar/dsmain/dlw, lever DLW) -- the summary of 57-p4zyzzzzz-ds-iejzz-dlw (dsm-dlw.sh) and of the
follow-up 57-p4zyzzzzz-ds-iejzzb-dlwb (dsm-dlwb.sh, dlwblocks.py: + per phase the sampled VRAM / lent units (mean of a 2 s
sampler), the lent share (lent plain steps / decode steps) and the lends; absent fields print "-"): per start
dlw-<k><ARM> its z0blocks.py block (<tag>-blocks.jsonl), its sampled probe (<tag>-samp.txt, daprobe.py J lines) and its
/stats at start (<tag>-stats0.json); per arm the mean over its starts, then B / A and C / A (A = the mean of the A starts) and,
with two A starts, the last A over the first (drift between starts).

  dlwsum.py OUTDIR ORDER        one "DLW ..." line a section on stdout
  dlwsum.py --selftest          the same over fabricated starts in a temporary directory; exit 0 when the lines come out

Standard library only."""
import json
import os
import statistics
import sys
import tempfile

C1 = [("essay", ("c1", "essay")), ("code", ("c1", "code")), ("edit", ("edit",))]
MB = ["c4", "c8"]


def get(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def load(o, tag):
    st = {"tag": tag, "block": None, "samp": {}, "units0": None}
    try:
        rows = [json.loads(l) for l in open(os.path.join(o, tag + "-blocks.jsonl")) if l.strip()]
        st["block"] = rows[-1] if rows else None
    except (OSError, ValueError):
        pass
    try:
        for l in open(os.path.join(o, tag + "-samp.txt")):
            if l.startswith("J "):
                r = json.loads(l[2:])
                st["samp"].setdefault(r.get("prompt"), []).append(r)
    except (OSError, ValueError):
        pass
    try:
        s = json.load(open(os.path.join(o, tag + "-stats0.json")))
        st["units0"] = {x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}
    except (OSError, ValueError, AttributeError):
        pass
    return st


def metrics(st):
    """name -> value for one start."""
    m = {}
    b = st["block"]
    res = b.get("res", {}) if b else {}
    for nm, path in C1:
        for k, lab in (("decode_tps", "tok/s"), ("tokens_per_step", "tok/step"), ("acceptance", "acc")):
            m["t0 %s %s" % (nm, lab)] = get(res, path + (k,))
    for p in (res.get("essaypos") or {}).get("positions") or []:
        if p.get("p", 9) <= 5:
            m["t0 essay p%d" % p["p"]] = p.get("rate")
    for ph in MB:
        for k, lab in (("agg_dec_tps", "tok/s"), ("ms_decode_step", "ms/step"), ("tok_per_decode_step", "tok/step"),
                       ("acceptance", "acc"), ("ttft_p50", "TTFT p50")):
            m["%s %s" % (ph, lab)] = get(res, (ph, k))
        sp = (res.get("statsph") or {}).get(ph) or {}
        m["%s VRAM units" % ph] = sp.get("vram_units_end")
        m["%s lent units" % ph] = sp.get("lent_units_end")
        m["%s hit %%" % ph] = sp.get("hit_pct_rank0")
        m["%s stream MB/step" % ph] = sp.get("stream_MB_step")
        sv, sl = sp.get("vram_units_sampled") or {}, sp.get("lent_units_sampled") or {}
        m["%s VRAM units sampled" % ph] = sv.get("mean")
        m["%s lent units sampled" % ph] = sl.get("mean")
        lp, dsx = sp.get("lent_plain_steps"), sp.get("decode_steps")
        m["%s lent share" % ph] = (lp / dsx) if isinstance(lp, (int, float)) and dsx else None
        m["%s lends" % ph] = sp.get("lends")
        dp = (res.get("draftph") or {}).get(ph) or {}
        ds = sp.get("decode_steps")
        z = dp.get("zero_cycles")
        m["%s depth-0 share" % ph] = (z / ds) if isinstance(z, (int, float)) and ds else None
    for nm in ("essay", "code", "edit"):
        v = [r.get("decode_tps") for r in st["samp"].get(nm, []) if r.get("decode_tps")]
        t = [r.get("tokens_per_step") for r in st["samp"].get(nm, []) if r.get("tokens_per_step")]
        m["sampled %s tok/s" % nm] = statistics.mean(v) if v else None
        m["sampled %s tok/step" % nm] = statistics.mean(t) if t else None
    u = st["units0"] or {}
    m["start VRAM units"] = u.get("vram")
    m["block ok"] = 1.0 if b and b.get("ok") else 0.0 if b else None
    return m


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None


def fmt(x):
    if x is None:
        return "-"
    if abs(x) >= 100:
        return "%.1f" % x
    if abs(x) >= 1:
        return "%.2f" % x
    return "%.3f" % x


def summary(o, order):
    starts = []
    for k, arm in enumerate(order):
        st = load(o, "dlw-%d%s" % (k, arm))
        starts.append((arm, st, metrics(st)))
    names = []
    for _, _, m in starts:
        for n in m:
            if n not in names:
                names.append(n)
    arms = [a for a in "ABC" if any(s[0] == a for s in starts)]
    per = {a: {n: mean([m.get(n) for x, _, m in starts if x == a]) for n in names} for a in arms}
    out = []
    for a in arms:
        tags = [st["tag"] for x, st, _ in starts if x == a]
        out.append("DLW arm %s (%s): %s" % (a, "+".join(tags), "; ".join("%s %s" % (n, fmt(per[a][n])) for n in names)))
    for a in ("B", "C"):
        if a not in per or "A" not in per:
            continue
        parts = []
        for n in names:
            x, y = per["A"][n], per[a][n]
            if x is None or y is None:
                continue
            if "units" in n or "share" in n or "acc" in n or n.startswith("t0 essay p") or "hit" in n:
                parts.append("%s %s vs %s (%+.3g)" % (n, fmt(y), fmt(x), y - x))
            elif x:
                parts.append("%s %s vs %s (%+.1f %%)" % (n, fmt(y), fmt(x), 100.0 * (y / x - 1)))
        out.append("DLW %s vs A: %s" % (a, "; ".join(parts)))
    a_starts = [m for x, _, m in starts if x == "A"]
    if len(a_starts) >= 2:
        f, l = a_starts[0], a_starts[-1]
        parts = []
        for n in names:
            x, y = f.get(n), l.get(n)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x and "tok/s" in n:
                parts.append("%s %+.1f %%" % (n, 100.0 * (y / x - 1)))
        out.append("DLW drift last A / first A (tok/s): %s" % "; ".join(parts))
    return out


def selftest():
    d = tempfile.mkdtemp(prefix="dlwsum-")
    for k, arm in enumerate("ABCA"):
        f = 1.0 + 0.02 * "ABC".index(arm)
        res = {"c1": {"essay": {"decode_tps": 60 * f, "tokens_per_step": 2.0, "acceptance": 0.5},
                      "code": {"decode_tps": 70 * f, "tokens_per_step": 3.0, "acceptance": 0.7}},
               "essaypos": {"positions": [{"p": 1, "rate": 0.6}, {"p": 2, "rate": 0.5}]},
               "edit": {"decode_tps": 90 * f, "tokens_per_step": 4.0, "acceptance": 0.8}}
        for ph in MB:
            res[ph] = {"agg_dec_tps": 130 * f, "ms_decode_step": 25.0 / f, "tok_per_decode_step": 1.1, "acceptance": 0.4,
                       "ttft_p50": 1.0}
        res["statsph"] = {ph: {"vram_units_end": 3000 + 20 * "ABC".index(arm), "lent_units_end": 190, "hit_pct_rank0": 97.5,
                               "stream_MB_step": 400.0, "decode_steps": 1000} for ph in MB}
        res["draftph"] = {ph: {"zero_cycles": 900} for ph in MB}
        with open(os.path.join(d, "dlw-%d%s-blocks.jsonl" % (k, arm)), "w") as fo:
            fo.write(json.dumps({"ok": True, "res": res}) + "\n")
        with open(os.path.join(d, "dlw-%d%s-samp.txt" % (k, arm)), "w") as fo:
            for p in ("essay", "code", "edit"):
                fo.write("J " + json.dumps({"mode": "sampled", "prompt": p, "decode_tps": 50 * f, "tokens_per_step": 1.8}) + "\n")
        with open(os.path.join(d, "dlw-%d%s-stats0.json" % (k, arm)), "w") as fo:
            json.dump({"experts": {"tiers": [{"name": "vram", "units": 3000 + 20 * "ABC".index(arm)}]}}, fo)
    lines = summary(d, "ABCA")
    ok = (len(lines) == 6 and lines[3].startswith("DLW B vs A") and "+2.0 %" in lines[3] and "+20" in lines[3]
          and lines[5].startswith("DLW drift"))
    print("\n".join(lines))
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    for l in summary(sys.argv[1], sys.argv[2]):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
