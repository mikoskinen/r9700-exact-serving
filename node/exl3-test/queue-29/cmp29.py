#!/usr/bin/env python3
"""cmp29.py LOG [NEW_TAG] [REF_TAG] -- one RESULT line: the 2.9 bpw row (default tag x29-best524) against the 2.25 bpw row with
the same image and flags (default x117-best524), every value as new / ref (+-x.x %). cc_sim u4 has no 524k 2.25 row (that item was
killed at 13:54 on 6.11.), so it is compared with the 6.11. morning window's 2.25 bpw cc_sim (05:31, MML 262144, acc9ae58 EXL3
image) and marked so. The last matching line of each kind wins. Missing values print as '-'."""
import json, re, sys

log = sys.argv[1]
new = sys.argv[2] if len(sys.argv) > 2 else "x29-best524"
ref = sys.argv[3] if len(sys.argv) > 3 else "x117-best524"
lines = [l.rstrip("\n") for l in open(log, errors="replace") if " RESULT " in l]


def last(pat):
    r = re.compile(pat)
    hit = None
    for l in lines:
        if r.search(l): hit = l
    return hit


def restart(tag):
    l = last(r"RESULT restart-pr266-%s: healthy" % re.escape(tag))
    if not l: return {}
    g = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, l))
    return {"start s": g(r"healthy in (\d+) s"), "VRAM experts GiB": g(r"VRAM experts ([\d.]+) GiB"),
            "KV GiB/card": g(r"KV ([\d.]+) GiB a card"), "sessions": g(r"([\d.]+) sessions at"),
            "pinned GiB": g(r"pinned by this process ([\d.]+) GiB")}


def guard(tag):
    l = last(r"RESULT guard (x117-)?%s: " % re.escape(tag))
    m = re.search(r"1-user ([\d.]+) ms/step", l or "")
    return {"guard ms": float(m.group(1))} if m else {}


def probe(tag):
    l = last(r"RESULT pr266 %s probe " % re.escape(tag))
    out = {}
    for name in ("essay", "code"):
        m = re.search(name + r" ([\d.]+) tok/s ([\d.]+) ms/step", l or "")
        if m: out[name + " tok/s"], out[name + " ms"] = float(m.group(1)), float(m.group(2))
    return out


def mb(tag, phase, keys):
    l = last(r"RESULT pr266 %s mb %s " % (re.escape(tag), phase))
    if not l: return {}
    try: d = json.loads(l[l.index("{"):])
    except ValueError: return {}
    return {"%s %s" % (phase, k): d.get(k) for k in keys if d.get(k) is not None}


def ccsim(tag):
    l = last(r"RESULT cc_sim %s u4 " % re.escape(tag))
    if not l: return {}
    g = lambda p: (lambda m: m.groups() if m else None)(re.search(p, l))
    out = {}
    t = g(r"ttft p50/p95/max ([\d.]+)/([\d.]+)/([\d.]+) s")
    if t: out["cc u4 TTFT p50"], out["cc u4 TTFT p95"] = float(t[0]), float(t[1])
    for k, p in (("cc u4 tok/s/agent", r"gen/agent p50 ([\d.]+) tok/s"), ("cc u4 total tok/s", r"total gen ([\d.]+) tok/s"),
                 ("cc u4 ms/decode step", r"([\d.]+) ms/decode step"), ("cc u4 err", r"err (\d+)")):
        m = g(p)
        if m: out[k] = float(m[0])
    return out


def row(tag, cc_tag):
    d = {}
    d.update(restart(tag)); d.update(guard(tag)); d.update(probe(tag))
    d.update(mb(tag, "c4", ("agg_dec_tps", "ms_decode_step", "ttft_p50")))
    d.update(mb(tag, "c8", ("agg_dec_tps", "ms_decode_step", "ttft_p50")))
    d.update(mb(tag, "pf", ("prefill_tps", "prefill_ttft", "decode_kept")))
    d.update(mb(tag, "turns", ("cold_ttft", "ttft_hit_mean")))
    d.update(ccsim(cc_tag))
    return d


a = row(new, new)
b = row(ref, ref)
cc_note = ""
if not any(k.startswith("cc ") for k in b):
    b.update(ccsim("best-s8"))
    cc_note = " [cc_sim ref: 6.11. morning best-s8 u4, MML 262144]"
parts = []
for k in list(dict.fromkeys(list(a) + list(b))):
    x, y = a.get(k), b.get(k)
    if x is None and y is None: continue
    pct = " (%+.1f %%)" % (100.0 * (x - y) / y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and y else ""
    parts.append("%s %s / %s%s" % (k, "-" if x is None else x, "-" if y is None else y, pct))
print("RESULT x29 compare %s vs %s (2.9 bpw / 2.25 bpw): %s%s" % (new, ref, "; ".join(parts) or "no rows", cc_note))
