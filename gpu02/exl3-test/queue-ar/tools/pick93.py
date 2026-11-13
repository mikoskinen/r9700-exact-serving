#!/usr/bin/env python3
"""pick93.py LOG -- 93's automatic choice (X93_WINNER=auto, 8.11.2026, queue-ar): from the run log's RESULT lines of 91
(57-p4zy) and 92 (57-p4zyz), the candidate arm with the highest mb c16 aggregate decode tok/s among those whose gate passed
(a "RESULT fn4 guard fn4-<arm>: fast..." line) and that logged no VIRHE. Candidates: 92's w4a / p4a / p4w and 91's d4t; the
power arms are not candidates (9.11.: power is decided -- +3-4 % at best, card 03:00's memory at 96 C from 260 W). The
reference is 91's d4a (same line, image and caps). The newest line per arm wins (a rerun replaces an earlier run). Prints
one line: "WINNER <93 arm> | <why>". If nothing beats d4a at c16 by >= 2 %, the best-c16 candidate is still the pick (so
the A/B lands) and the line says so. No usable candidate (92 / 91 have not run, or every candidate failed) -> the
fail-safe "WINNER w4a" (no power arm). Standard library only; reads LOG only."""
import re, sys

CANDS = ["w4a", "p4a", "p4w", "d4t"]   # 9.11.: power arms excluded (decided: +3-4 % at best; 03:00 mem 96 C from 260 W)
REF = "d4a"
FAILSAFE = "w4a"


def main():
    log = sys.argv[1]
    c16, gate, virhe = {}, {}, {}
    try:
        lines = open(log, errors="replace").read().splitlines()
    except OSError as e:
        print(f"WINNER {FAILSAFE} | fail-safe: the log {log} cannot be read ({e})"); return
    arms = CANDS + [REF]
    rx_mb = re.compile(r"RESULT fn4 fn4-([a-z0-9-]+) mb c16: .*?\"agg_dec_tps\": ([0-9.]+)")
    rx_g = re.compile(r"RESULT fn4 guard fn4-([a-z0-9-]+): (\S+)")
    rx_start = re.compile(r"RESULT fn4 fn4-([a-z0-9-]+): start ")
    for ln in lines:
        m = rx_start.search(ln)
        if m and m.group(1) in arms:   # a new run of the arm: forget the older one's results
            a = m.group(1); c16.pop(a, None); gate.pop(a, None); virhe.pop(a, None)
            continue
        m = rx_mb.search(ln)
        if m and m.group(1) in arms:
            c16[m.group(1)] = float(m.group(2)); continue
        m = rx_g.search(ln)
        if m and m.group(1) in arms:
            gate[m.group(1)] = m.group(2).rstrip(","); continue
        if "VIRHE" in ln:
            for a in arms:
                if re.search(rf"fn4-{re.escape(a)}(\b|:)", ln) and not re.search(rf"fn4-{re.escape(a)}-p", ln):
                    virhe[a] = ln.split("RESULT", 1)[-1][:160]
    ok = {a: v for a, v in c16.items() if a in CANDS and gate.get(a, "").startswith("fast") and a not in virhe}
    bad = {a: (gate.get(a, "no gate line"), virhe.get(a, "")) for a in CANDS if a in c16 and a not in ok}
    seen = ", ".join(f"{a} {c16[a]:.1f}" for a in arms if a in c16) or "none"
    if not ok:
        print(f"WINNER {FAILSAFE} | fail-safe: no candidate with a passed gate, no VIRHE and an mb c16 line "
              f"(c16 seen: {seen}; excluded: {bad or '-'})"); return
    best = max(ok, key=lambda a: ok[a])
    arm = best
    ref = c16.get(REF)
    if ref:
        gain = (ok[best] / ref - 1) * 100
        why = (f"{best} c16 {ok[best]:.1f} tok/s vs d4a {ref:.1f} ({gain:+.1f} %)" +
               ("" if gain >= 2 else " -- nothing beats d4a by >= 2 %, the best-c16 arm runs anyway"))
    else:
        why = f"{best} c16 {ok[best]:.1f} tok/s (no d4a c16 line to compare)"
    print(f"WINNER {arm} | {why}; candidates {', '.join(f'{a} {v:.1f}' for a, v in sorted(ok.items(), key=lambda kv: -kv[1]))}"
          f"{'; excluded ' + str(bad) if bad else ''}")


if __name__ == "__main__":
    main()
