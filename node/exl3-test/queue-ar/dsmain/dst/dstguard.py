#!/usr/bin/env python3
"""dstguard.py (11.11.2026, lever DST item 57-p4zyzzzzz-ds-r04z-dst) -- did a cell's levers act? (coordinator: a cell whose
lever silently did nothing gets a VIRHE and no speed reading)

  dstguard.py check --cell CELL.json --bic B --grid G --il IL --mt MT --chunks CHUNKS.json --alone ALONE_CHUNKS.json
                    --fn4a GET0.json --fn4b GET1.json --stats-a STATS0.json --stats-b STATS1.json --psk-paths "a.b c.d"
                    [--psk-zero "e.f"]
      CELL = the cell's switches (sched_interleave, sched_mixed_tok, ds41_psk, sched_short_bypass, sched_bi_chunk when not
      the kept one); CHUNKS / ALONE_CHUNKS = longid's chunk lists (the probe loaded / the same prompt alone); GET0/1 =
      /admin/fn4 before / after the cell; STATS0/1 = /stats before / after. Prints "GUARD OK [what was checked]" or
      "GUARD VIRHE reason; reason ..." (exit 0 either way; 2 when the inputs do not parse).
      bypass (sched_short_bypass > 0): fn4 "dst".bypassed grew over the cell.
      D (the kept sched_bi_chunk 3 / 4, the cell not overriding it): every loaded chunk end of the probe is one of alone's
        piece ends, or a grid point (multiple of G) inside alone's piece with >= 128 rows on both sides; and at least one
        such grid cut ran in a MIXED step (decode rows present).
      budget (sched_mixed_tok < 2112): no MIXED chunk of the probe is longer than the budget.
      interleave (decode rows present): the median count of steps between two consecutive probe chunks in MIXED steps
        is at least sched_interleave + 1.
      PSK (ds41_psk >= 1): every /stats counter named in --psk-paths (dotted paths) exists and grew over the cell, and
        every one named in --psk-zero exists and did not grow (design 2: psk.latched / planned_steps / layers_skipped
        grow, psk.plan_mismatch stays).
  dstguard.py selftest
Standard library only.
"""
import argparse, json, statistics, sys


def load(p):
    with open(p) as f:
        return json.load(f)


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None


def check(cell, bic, grid, il, mt, ch, al, g0, g1, s0, s1, psk_paths, psk_zero=""):
    bad, ok = [], []
    # bypass
    if (cell.get("sched_short_bypass") or 0) > 0:
        a = ((g0 or {}).get("dst") or {}).get("bypassed")
        b = ((g1 or {}).get("dst") or {}).get("bypassed")
        if a is None or b is None:
            bad.append("bypass: fn4 GET has no dst.bypassed")
        elif b - a <= 0:
            bad.append("bypass: sched_short_bypass %s admitted nothing (dst.bypassed %s -> %s)" % (cell.get("sched_short_bypass"), a, b))
        else:
            ok.append("bypass +%d" % (b - a))
    chunks = (ch or {}).get("chunks") if (ch or {}).get("ok") else None
    if chunks is None:
        bad.append("no chunk list for the probe")
        chunks = []
    prompt = (ch or {}).get("prompt") or 0
    # D
    kept_d = bic in (3, 4) and cell.get("sched_bi_chunk") is None
    if kept_d and chunks:
        if not (al or {}).get("ok"):
            bad.append("D: no alone chunk list to compare with")
        else:
            ab = list(zip(al["boundaries"], al["q"]))
            ends = {b + q for b, q in ab}
            def piece(pos):
                for b, q in ab:
                    if b <= pos < b + q:
                        return b, b + q
                return None
            off, cuts = [], 0
            for c in chunks:
                b, q, ph = c[0], c[1], c[2]
                e = b + q
                if e in ends or e >= prompt - 1:
                    continue
                pc = piece(b)
                if pc and e % grid == 0 and q >= 128 and pc[1] - e >= 128 and pc[0] <= b < e < pc[1]:
                    cuts += ph == "MIXED"
                    continue
                off.append((b, q))
            if off:
                bad.append("D: %d chunk(s) end neither at alone's piece end nor on the %d grid >= 128 a side: %s" % (len(off), grid, off[:4]))
            elif cuts == 0:
                bad.append("D: no grid cut ran beside decode rows (sched_bi_chunk %d did not act)" % bic)
            else:
                ok.append("D %d grid cuts in MIXED steps" % cuts)
    # budget
    if mt < 2112 and chunks:
        mx = max([c[1] for c in chunks if c[2] == "MIXED"] or [0])
        if mx > mt:
            bad.append("budget: a MIXED chunk of %d rows > sched_mixed_tok %d" % (mx, mt))
        else:
            ok.append("budget max MIXED chunk %d" % mx)
    # interleave
    ms = [c[5] for c in chunks if c[2] == "MIXED" and len(c) > 5]
    if len(ms) >= 4:
        gaps = [b - a for a, b in zip(ms, ms[1:])]
        med = statistics.median(gaps)
        if med < il + 1:
            bad.append("interleave: median %g steps between chunks < sched_interleave %d + 1" % (med, il))
        else:
            ok.append("interleave median gap %g" % med)
    # PSK
    if (cell.get("ds41_psk") or 0) >= 1:
        for p in psk_paths.split():
            a, b = dig(s0 or {}, p), dig(s1 or {}, p)
            if a is None or b is None:
                bad.append("PSK: /stats has no %s" % p)
            elif b - a <= 0:
                bad.append("PSK: %s did not grow (%s -> %s)" % (p, a, b))
            else:
                ok.append("PSK %s +%g" % (p, b - a))
        for p in psk_zero.split():
            a, b = dig(s0 or {}, p), dig(s1 or {}, p)
            if a is None or b is None:
                bad.append("PSK: /stats has no %s" % p)
            elif b - a != 0:
                bad.append("PSK: %s moved (%s -> %s)" % (p, a, b))
            else:
                ok.append("PSK %s 0" % p)
    return bad, ok


def cmd_check(a):
    try:
        cell = json.loads(a.cell) if a.cell.strip().startswith("{") else load(a.cell)
        ch = load(a.chunks) if a.chunks else None
        al = load(a.alone) if a.alone else None
        g0 = load(a.fn4a) if a.fn4a else None
        g1 = load(a.fn4b) if a.fn4b else None
        s0 = load(a.stats_a) if a.stats_a else None
        s1 = load(a.stats_b) if a.stats_b else None
    except (OSError, ValueError) as e:
        print("GUARD VIRHE inputs do not parse (%s)" % e)
        return 2
    bad, ok = check(cell, a.bic, a.grid, a.il, a.mt, ch, al, g0, g1, s0, s1, a.psk_paths, a.psk_zero)
    print(("GUARD VIRHE " + "; ".join(bad)) if bad else ("GUARD OK [%s]" % ", ".join(ok)))
    return 0


def selftest():
    al = dict(ok=True, boundaries=[0, 2048, 6144, 10240], q=[2048, 4096, 4096, 2000], prompt=12240)
    def ch(lst):
        return dict(ok=True, prompt=12240, chunks=lst)
    good = ch([[0, 2048, "MIXED", 5, 2052, 10], [2048, 2048, "MIXED", 5, 2052, 19], [4096, 2048, "MIXED", 5, 2052, 28],
               [6144, 2048, "MIXED", 5, 2052, 37], [8192, 2048, "MIXED", 5, 2052, 46], [10240, 2000, "MIXED", 5, 2004, 55]])
    g0, g1 = {"dst": {"bypassed": 3}}, {"dst": {"bypassed": 5}}
    s0, s1 = {"psk": {"cut_steps": 1, "rows": 10, "plan_mismatch": 0}}, {"psk": {"cut_steps": 4, "rows": 90, "plan_mismatch": 0}}
    cell = {"sched_interleave": 8, "sched_mixed_tok": 0, "ds41_psk": 2, "sched_short_bypass": 1280}
    bad, ok = check(cell, 3, 2048, 8, 2112, good, al, g0, g1, s0, s1, "psk.cut_steps psk.rows", "psk.plan_mismatch")
    assert not bad and len(ok) == 6, (bad, ok)
    bad, ok = check(cell, 3, 2048, 8, 2112, good, al, g0, g1, s0, {"psk": {"cut_steps": 4, "rows": 90, "plan_mismatch": 2}},
                    "psk.cut_steps psk.rows", "psk.plan_mismatch")
    assert bad == ["PSK: psk.plan_mismatch moved (0 -> 2)"], bad
    # an off-grid end, a bypass that did nothing, a PSK counter missing, an interleave not acting, a budget not acting
    badc = ch([[0, 2048, "MIXED", 5, 2052, 10], [2048, 1936, "MIXED", 5, 2052, 11], [3984, 2160, "MIXED", 5, 2052, 12],
               [6144, 2048, "MIXED", 5, 2052, 13], [8192, 2048, "MIXED", 5, 2052, 14], [10240, 2000, "MIXED", 5, 2004, 15]])
    bad, ok = check(dict(cell, sched_mixed_tok=1088), 3, 1024, 8, 1088, badc, al, g0, g0, s0, {"psk": {"rows": 90}},
                    "psk.cut_steps psk.rows")
    kinds = sorted({b.split(":")[0] for b in bad})
    assert kinds == ["D", "PSK", "budget", "bypass", "interleave"], bad
    # today's chunking (sched_bi_chunk 0 override): no D check; PSK 0: no PSK check; no bypass: no bypass check
    bad, ok = check({"sched_interleave": 8, "ds41_psk": 0, "sched_bi_chunk": 0}, 3, 2048, 8, 2112, badc, al, None, None,
                    None, None, "psk.cut_steps")
    assert [b.split(":")[0] for b in bad] == ["interleave"], bad
    print("SELFTEST OK (bypass, D grid / alone ends / a MIXED cut, budget, interleave, PSK counters; overrides skip checks)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "selftest"))
    ap.add_argument("--cell", default="{}")
    ap.add_argument("--bic", type=int, default=0)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--il", type=int, default=8)
    ap.add_argument("--mt", type=int, default=2112)
    ap.add_argument("--chunks", default="")
    ap.add_argument("--alone", default="")
    ap.add_argument("--fn4a", default="")
    ap.add_argument("--fn4b", default="")
    ap.add_argument("--stats-a", default="")
    ap.add_argument("--stats-b", default="")
    ap.add_argument("--psk-paths", default="")
    ap.add_argument("--psk-zero", default="")
    a = ap.parse_args()
    sys.exit(selftest() if a.mode == "selftest" else cmd_check(a))
