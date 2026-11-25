#!/usr/bin/env python3
"""dstguard2.py (12.11.2026, ds-main-port noa builder; item 57-p4zyzzzzz-ds-r00zzz-cz-tdst2) = dstguard.py (queue-ar/dsmain/dst,
md5 ab79e7a5, COPIED, the original not edited): its D-grid, interleave and PSK checks kept; the coordinator's guard for the
NOA item added. A run whose lever did not act, or whose pins / load are not what the cell says, is a VIRHE (no speed reading).

  dstguard2.py check --mode l|n|c8 --cell CELL.json --want PIN.json --bic B --grid G --il IL --chunks CH.json --alone ACH.json
                     --fn4a GET0.json --fn4b GET1.json --stats-a S0.json --stats-b S1.json --info RUN.json --md MDELTA.json
                     --psk-paths "a.b .." --psk-zero "c.d .."
      mode l = decode alive (the long stream + c4 turnovers + newcomers in the window), n = no decode stream (on purpose),
      c8 = mb c8 + the c8 identity burst (no probe: no chunk checks).
      pins: every switch in PIN (the pin + the cell; sched_mixed_tok 0 = 2112) read back from GET before AND after the run.
      NOA (cell sched_noa > 0): noa.pieces and noa.rows grew; else noa admitted / pieces / rows / displaced / waited / steps
        did not move (pf_steps_* count every step and are not checked).
      bypass (cell sched_short_bypass > 0): mode l -> dst.bypassed grew; mode n -> information only (no held step without
        decode rows: the gap NOA covers); sched_short_bypass 0 -> dst.bypassed and dst.bypass_refused did not move.
      drafting pinned off: the window's draft tokens <= max(64, 2 % of its decode tokens) (MD = mdelta's JSON).
      PSK (cell ds41_psk >= 1): every --psk-paths counter grew, every --psk-zero one stayed (never layers_skipped: loaded
        chunks run MIXED, no layer is skipped whole).
      newcomers (RUN has newc_cached): every one fresh (cached 0) and answered (newc_err 0); mode l: the long stream alive
        through the window (long_alive).
      D (kept sched_bi_chunk 3 / 4, mode l / n): the probe's chunk ends on alone's piece ends or the G grid >= 128 rows a
        side; mode l also >= 1 such grid cut in a MIXED step. interleave (mode l): as dstguard.py.
      Prints "GUARD OK [...]" or "GUARD VIRHE reason; ..." (exit 0; 2 when the inputs do not parse).
  dstguard2.py selftest
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


def check(mode, cell, want, bic, grid, il, ch, al, g0, g1, s0, s1, info, md, psk_paths, psk_zero=""):
    bad, ok = [], []
    # pins read back, before and after
    for nm, g in (("before", g0), ("after", g1)):
        sw = (g or {}).get("switches") or {}
        miss = []
        for k, v in (want or {}).items():
            vv = 2112 if (k == "sched_mixed_tok" and v == 0) else v
            if sw.get(k) != vv:
                miss.append("%s=%s (want %s)" % (k, sw.get(k), vv))
        if miss:
            bad.append("pins %s the run: %s" % (nm, ", ".join(miss[:6])))
    if not bad:
        ok.append("pins %d read back" % len(want or {}))
    def dd(g0_, g1_, sect, key):
        a = ((g0_ or {}).get(sect) or {}).get(key)
        b = ((g1_ or {}).get(sect) or {}).get(key)
        return None if a is None or b is None else b - a
    # NOA
    noa = int(cell.get("sched_noa") or 0)
    if noa > 0:
        p, r = dd(g0, g1, "noa", "pieces"), dd(g0, g1, "noa", "rows")
        if p is None or r is None:
            bad.append("NOA: fn4 GET has no noa counters")
        elif p <= 0 or r <= 0:
            bad.append("NOA: sched_noa %d planned nothing (noa.pieces +%s, rows +%s)" % (noa, p, r))
        else:
            ok.append("NOA pieces +%d rows +%d admitted +%s displaced +%s waited +%s" % (p, r, dd(g0, g1, "noa", "admitted"),
                      dd(g0, g1, "noa", "displaced"), dd(g0, g1, "noa", "waited")))
    else:
        mv = {k: dd(g0, g1, "noa", k) for k in ("admitted", "pieces", "rows", "displaced", "waited", "steps")}
        if any(v is None for v in mv.values()):
            bad.append("NOA: fn4 GET has no noa counters")
        elif any(v != 0 for v in mv.values()):
            bad.append("NOA: counters moved with sched_noa 0: %s" % {k: v for k, v in mv.items() if v})
        else:
            ok.append("NOA 0")
    # bypass
    byp = int(cell.get("sched_short_bypass") or 0)
    b_, rf = dd(g0, g1, "dst", "bypassed"), dd(g0, g1, "dst", "bypass_refused")
    if b_ is None or rf is None:
        bad.append("bypass: fn4 GET has no dst counters")
    elif byp > 0:
        if mode == "l" and b_ <= 0:
            bad.append("bypass: sched_short_bypass %d admitted nothing with decode rows alive (dst.bypassed +%d, refused +%d)" % (byp, b_, rf))
        else:
            ok.append("bypass +%d refused +%d%s" % (b_, rf, " (no-decode: information)" if mode != "l" else ""))
    elif b_ != 0 or rf != 0:
        bad.append("bypass: counters moved with sched_short_bypass 0 (+%d / +%d)" % (b_, rf))
    else:
        ok.append("bypass 0")
    # drafting pinned off
    dt, dn = (md or {}).get("draft_tok"), (md or {}).get("decode_tok")
    if dt is None or dn is None:
        bad.append("draft: no draft / decode token deltas")
    elif dt > max(64, 0.02 * dn):
        bad.append("draft: %d draft tokens over %d decode tokens under the depth-0 pin" % (dt, dn))
    else:
        ok.append("draft %d / %d" % (dt, dn))
    # newcomers fresh, the long stream alive
    if info is not None and "newc_cached" in info:
        cc = info.get("newc_cached") or []
        if info.get("newc_err"):
            bad.append("newcomers: %s failed" % info.get("newc_err"))
        if not cc or any(c != 0 for c in cc):
            bad.append("newcomers: not fresh (cached %s)" % cc)
        else:
            ok.append("newcomers fresh x%d" % len(cc))
    if mode == "l":
        if not (info or {}).get("long_alive"):
            bad.append("load: the long decode stream did not span the prefill window (%s tokens, err %s)" % (
                (info or {}).get("long_tokens"), (info or {}).get("long_err")))
        else:
            ok.append("long stream alive")
    if mode in ("l", "n"):
        chunks = (ch or {}).get("chunks") if (ch or {}).get("ok") else None
        if chunks is None:
            bad.append("no chunk list for the probe")
            chunks = []
        prompt = (ch or {}).get("prompt") or 0
        # D
        kept_d = cell.get("sched_bi_chunk", bic) in (3, 4)   # dstguard2: the cell names its sched_bi_chunk
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
        if mode == "n":   # without decode rows a grid cut need not ride a MIXED step
            bad = [b for b in bad if not b.startswith("D: no grid cut ran beside decode rows")]
        # interleave
        ms = [] if mode != "l" else [c[5] for c in chunks if c[2] == "MIXED" and len(c) > 5]
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
        want = json.loads(a.want) if a.want.strip().startswith("{") else (load(a.want) if a.want else {})
        ch = load(a.chunks) if a.chunks else None
        al = load(a.alone) if a.alone else None
        g0 = load(a.fn4a) if a.fn4a else None
        g1 = load(a.fn4b) if a.fn4b else None
        s0 = load(a.stats_a) if a.stats_a else None
        s1 = load(a.stats_b) if a.stats_b else None
        info = load(a.info) if a.info else None
        md = json.loads(a.md) if a.md.strip().startswith("{") else (load(a.md) if a.md else None)
    except (OSError, ValueError) as e:
        print("GUARD VIRHE inputs do not parse (%s)" % e)
        return 2
    bad, ok = check(a.mode, cell, want, a.bic, a.grid, a.il, ch, al, g0, g1, s0, s1, info, md, a.psk_paths, a.psk_zero)
    print(("GUARD VIRHE " + "; ".join(bad)) if bad else ("GUARD OK [%s]" % ", ".join(ok)))
    return 0


def selftest():
    al = dict(ok=True, boundaries=[0, 2048, 6144, 10240], q=[2048, 4096, 4096, 2000], prompt=12240)
    def ch(lst):
        return dict(ok=True, prompt=12240, chunks=lst)
    good = ch([[0, 2048, "MIXED", 5, 2052, 10], [2048, 2048, "MIXED", 5, 2052, 19], [4096, 2048, "MIXED", 5, 2052, 28],
               [6144, 2048, "MIXED", 5, 2052, 37], [8192, 2048, "MIXED", 5, 2052, 46], [10240, 2000, "MIXED", 5, 2004, 55]])
    sw = {"sched_noa": 2, "sched_bi_chunk": 3, "sched_short_bypass": 0, "sched_bypass_total": 0, "sched_interleave": 8,
          "sched_mixed_tok": 2112, "ds41_psk": 2, "ds41_psk_k": 4, "draft_depth0": 2, "draft_probe": 4096}
    want = dict(sw, sched_mixed_tok=0)
    N0 = {"admitted": 0, "pieces": 0, "rows": 0, "displaced": 0, "waited": 0, "steps": 0, "pf_steps_1": 5}
    N1 = {"admitted": 3, "pieces": 5, "rows": 6849, "displaced": 0, "waited": 0, "steps": 5, "pf_steps_1": 30}
    D0 = {"bypassed": 0, "bypass_refused": 0}
    g0, g1 = {"switches": sw, "noa": N0, "dst": D0}, {"switches": sw, "noa": N1, "dst": D0}
    s0 = {"psk": {"cut_passes": 1, "rows": 10, "plan_mismatch": 0}}
    s1 = {"psk": {"cut_passes": 4, "rows": 90, "plan_mismatch": 0}}
    info = {"newc_cached": [0, 0, 0], "newc_err": 0, "long_alive": True, "long_tokens": 900}
    md = {"draft_tok": 5, "decode_tok": 2000}
    cell = {"sched_noa": 2, "sched_short_bypass": 0, "ds41_psk": 2}
    P, Z = "psk.cut_passes psk.rows", "psk.plan_mismatch"
    bad, ok = check("l", cell, want, 3, 2048, 8, good, al, g0, g1, s0, s1, info, md, P, Z)
    assert not bad, bad
    # a pin not read back, NOA silent, drafting on, a newcomer cached, the long stream dead
    g1b = {"switches": dict(sw, draft_depth0=1), "noa": N0, "dst": D0}
    bad, ok = check("l", cell, want, 3, 2048, 8, good, al, g0, g1b, s0, s1, dict(info, newc_cached=[0, 512, 0], long_alive=False),
                    {"draft_tok": 900, "decode_tok": 2000}, P, Z)
    kinds = sorted({b.split(":")[0].split(" ")[0] for b in bad})
    assert kinds == ["NOA", "draft", "load", "newcomers", "pins"], bad
    # NOA counters moving with sched_noa 0; bypass silent with decode rows (VIRHE) vs no-decode (information)
    cz = {"sched_noa": 0, "sched_short_bypass": 1280, "ds41_psk": 2}
    wz = dict(want, sched_noa=0, sched_short_bypass=1280)
    swz = dict(sw, sched_noa=0, sched_short_bypass=1280)
    bad, ok = check("l", cz, wz, 3, 2048, 8, good, al, {"switches": swz, "noa": N0, "dst": D0},
                    {"switches": swz, "noa": N1, "dst": D0}, s0, s1, info, md, P, Z)
    assert sorted({b.split(":")[0] for b in bad}) == ["NOA", "bypass"], bad
    bad, ok = check("n", cz, wz, 3, 2048, 8, good, al, {"switches": swz, "noa": N0, "dst": D0},
                    {"switches": swz, "noa": N0, "dst": D0}, s0, s1, {"newc_cached": [0], "newc_err": 0}, md, P, Z)
    assert not bad and any("information" in o for o in ok), (bad, ok)
    # c8: no chunk checks; PSK counters must still grow
    bad, ok = check("c8", cell, want, 3, 2048, 8, None, None, g0, g1, s0, s0, None, md, P, Z)
    assert [b.split(":")[0] for b in bad] == ["PSK", "PSK"], bad
    # D off-grid end in mode n is still a VIRHE; the "no MIXED cut" one is not
    offg = ch([[0, 2048, "prefill", 1, 2048, 1], [2048, 1936, "prefill", 1, 1936, 2], [3984, 2160, "prefill", 1, 2160, 3],
               [6144, 4096, "prefill", 1, 4096, 4], [10240, 2000, "prefill", 1, 2000, 5]])
    bad, ok = check("n", cell, want, 3, 2048, 0, offg, al, g0, g1, s0, s1, {"newc_cached": [0], "newc_err": 0}, md, P, Z)
    assert [b.split(":")[0] for b in bad] == ["D"], bad
    onl = ch([[0, 2048, "prefill", 1, 2048, 1], [2048, 4096, "prefill", 1, 4096, 2], [6144, 4096, "prefill", 1, 4096, 3],
              [10240, 2000, "prefill", 1, 2000, 4]])
    bad, ok = check("n", cell, want, 3, 2048, 0, onl, al, g0, g1, s0, s1, {"newc_cached": [0], "newc_err": 0}, md, P, Z)
    assert not bad, bad
    print("SELFTEST OK (pins before / after, NOA grew / stayed, bypass l / n / off, draft pin, newcomers fresh, long stream, "
          "D grid (n: no MIXED cut needed), interleave, PSK counters; c8 skips chunk checks)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "selftest"))
    ap.add_argument("--mode", default="l", choices=("l", "n", "c8"))
    ap.add_argument("--cell", default="{}")
    ap.add_argument("--want", default="")
    ap.add_argument("--bic", type=int, default=0)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--il", type=int, default=8)
    ap.add_argument("--chunks", default="")
    ap.add_argument("--alone", default="")
    ap.add_argument("--fn4a", default="")
    ap.add_argument("--fn4b", default="")
    ap.add_argument("--stats-a", default="")
    ap.add_argument("--stats-b", default="")
    ap.add_argument("--info", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--psk-paths", default="")
    ap.add_argument("--psk-zero", default="")
    a = ap.parse_args()
    sys.exit(selftest() if a.cmd == "selftest" else cmd_check(a))
