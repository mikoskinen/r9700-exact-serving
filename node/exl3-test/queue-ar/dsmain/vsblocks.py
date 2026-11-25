#!/usr/bin/env python3
"""vsblocks.py (10.11.2026, queue-ar dsmain, ds-main-port VRAM agent; item 57-p4zyzzzzz-ds-hxz-vramstack, body dsm-vramstack.sh)
-- ggblocksdz2.py (loaded read only from beside this file; not edited) with the routed experts' /stats counters around every
phase. Each phase's result dict gains "xstats" = the delta between a /stats snapshot before and after the phase: steps, decode
steps, engine ms a decode step, VRAM units at the start / end (experts tier "vram"), rank-0 routed / missed a step and hit %,
promotions a step, stream MB a step (zero-copy host reads), h2d MB a step (the counters dsm-vramsweep.sh's statsum reads).
One "VSX {json}" line a phase (label, block, phase, counters). Arguments, BLOCK / GGB lines and the jsonl rows: exactly
ggblocksdz2.py's (the rows only gain the xstats keys). Standard library only."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ggblocksdz2", os.path.join(HERE, "ggblocksdz2.py"))
gg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gg)


def snap(url):
    try: return gg.http(url + "/stats", timeout=15)
    except Exception: return None


def units(s):
    for t in ((s or {}).get("experts") or {}).get("tiers") or []:
        if t.get("name") == "vram": return t.get("units")
    return None


def delta(a, b):
    if not a or not b: return {"error": "no /stats snapshot"}
    ea, eb = a.get("experts") or {}, b.get("experts") or {}
    la, lb = a.get("link") or {}, b.get("link") or {}
    n = lambda d, k: d.get(k) or 0
    ds = n(b, "steps") - n(a, "steps")
    dd = n(b, "steps_decode") - n(a, "steps_decode")
    dr = n(eb, "routed") - n(ea, "routed")
    drr = n(eb, "routed_res") - n(ea, "routed_res")
    per = lambda x, r=2: round(x / ds, r) if ds > 0 else None
    return {"steps": ds, "decode_steps": dd,
            "ms_decode_step": round((n(b, "step_ns_decode") - n(a, "step_ns_decode")) / dd / 1e6, 3) if dd > 0 else None,
            "vram_units_start": units(a), "vram_units_end": units(b),
            "hit_pct_rank0": round(100.0 * drr / dr, 3) if dr > 0 else None,
            "routed_step_rank0": per(dr), "missed_step_rank0": per(dr - drr, 3),
            "promotions_step": per(n(eb, "promotions") - n(ea, "promotions"), 3),
            "stream_MB_step": per((n(lb, "stream_bytes") - n(la, "stream_bytes")) / 1e6),
            "h2d_MB_step": per((n(lb, "h2d_bytes") - n(la, "h2d_bytes")) / 1e6)}


_phase = gg.phase
_block = {"i": -1}


def phase(a, dph, name, f):
    if name == "c1": _block["i"] += 1          # every block starts with the c1 phase (ggblocksdz2.main)
    s0 = snap(a.url)
    r = _phase(a, dph, name, f)
    x = delta(s0, snap(a.url))
    if isinstance(r, dict): r["xstats"] = x
    print("VSX " + json.dumps(dict({"label": a.label, "block": _block["i"], "phase": name}, **x)), flush=True)
    return r


gg.phase = phase

if __name__ == "__main__":
    sys.exit(gg.main())
