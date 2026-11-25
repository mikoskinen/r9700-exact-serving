#!/usr/bin/env python3
"""compare.py [RUNSDIR] -- the EXL3 window's numbers next to the 3.11. UD-IQ1_S TP4 references (same harness, same flags), as a
Markdown table with % differences (positive = EXL3 higher). Reads window.sh's outputs: probe.txt (stepprobe), mb-*.txt (mb.py),
prefill.txt (ds41_tests prefill), onem.txt (onem.py), cc_sim RESULT lines in ds41-exl3.log. Missing phases print '-'."""
import glob, json, os, re, statistics, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/gpu/ds41/exl3-test/runs")
def lines(f, pre):
    try: return [json.loads(l[len(pre):]) for l in open(os.path.join(R, f)) if l.startswith(pre)]
    except FileNotFoundError: return []
rows = []  # (name, unit, ref, val, higher_is_better, note)
p = {x["prompt"]: x for x in lines("probe.txt", "J ")}
rows += [("1 käyttäjä, essee", "tok/s", 62.6, p.get("essay", {}).get("decode_tps"), True, "3.11. 15,9 ms/askel"),
         ("1 käyttäjä, koodi", "tok/s", 68.5, p.get("code", {}).get("decode_tps"), True, "3.11. 14,5 ms/askel"),
         ("1 käyttäjä, askel (koodi)", "ms", 14.5, p.get("code", {}).get("ms_per_decode_step"), False, "")]
mb = {}
for f in glob.glob(os.path.join(R, "mb-*.txt")):
    for x in lines(os.path.basename(f), "MB "): mb[x["phase"]] = x
rows += [("4 virtaa, dekoodi yhteensä", "tok/s", 125.7, mb.get("c4", {}).get("agg_dec_tps"), True, "3.11. sama kokoonpano"),
         ("8 virtaa, dekoodi yhteensä", "tok/s", 164.0, mb.get("c8", {}).get("agg_dec_tps"), True, "3.11. ckint 2048 ilman span/lomitusta"),
         ("31k esitäyttö 4 dekooderin rinnalla", "tok/s", 1475, mb.get("pf", {}).get("prefill_tps"), True, "3.11. TTFT 21,3 s"),
         ("56k istunto, kylmä TTFT", "s", 27.8, mb.get("turns", {}).get("cold_ttft"), False, ""),
         ("jatkovuoro, TTFT ka", "s", 2.16, mb.get("turns", {}).get("ttft_hit_mean"), False, "3.11. uudelleenlaskenta 3 318")]
pf = lines("prefill.txt", "J ")
def pfv(lo, hi):
    v = [x["prefill_tps"] for x in pf if x.get("prefill_tps") and lo <= (x.get("prompt_tokens") or 0) <= hi]
    return round(statistics.mean(v), 1) if v else None
rows += [("kylmä esitäyttö 16k", "tok/s", 2844, pfv(12000, 20000), True, "aamun liput: DSpark 5, ckint 8192"),
         ("kylmä esitäyttö ~59k", "tok/s", 2641, pfv(50000, 70000), True, ""),
         ("kylmä esitäyttö ~140k", "tok/s", 2316, pfv(120000, 160000), True, "")]
om = lines("onem.txt", "OM ")
def omv(lo, hi, k="prefill_tps"):
    v = [x.get(k) for x in om if x.get("test") == "depth" and lo <= (x.get("prompt_tokens") or 0) <= hi]
    return v[0] if v else None
hit = [x for x in om if x.get("test") == "hit-turn"]
rows += [("kylmä esitäyttö 266k (1M-tila)", "tok/s", 1805, omv(240000, 290000), True, ""),
         ("kylmä esitäyttö 501k", "tok/s", 1287, omv(470000, 530000), True, ""),
         ("kylmä esitäyttö ~644k", "tok/s", 1081, omv(600000, 690000), True, ""),
         ("jatkovuoro syvimmässä istunnossa", "s", 8.01, hit[0].get("ttft") if hit else None, False, "3.11. 644k: uudelleenlaskenta 4 421")]
cc = {}
try:
    for l in open(os.path.join(R, "ds41-exl3.log")):
        m = re.search(r"RESULT cc_sim (\S+) u(\d+) \[.*?ttft p50/p95/max ([0-9.]+)/.*?gen/agent p50 ([0-9.]+) tok/s, total gen ([0-9.]+)", l)
        if m: cc[int(m.group(2))] = tuple(float(m.group(i)) for i in (3, 4, 5))
except FileNotFoundError: pass
for u, ref in ((4, (3.92, 29.1, 92.5)), (8, (4.31, 14.4, 94.5))):
    v = cc.get(u, (None, None, None))
    rows += [(f"Claude Code -simulaatio {u} käyttäjää, TTFT p50", "s", ref[0], v[0], False, ""),
             (f"Claude Code -simulaatio {u} käyttäjää, per agentti", "tok/s", ref[1], v[1], True, ""),
             (f"Claude Code -simulaatio {u} käyttäjää, yhteensä", "tok/s", ref[2], v[2], True, "")]
def fmt(x): return "-" if x is None else (f"{x:,.2f}" if x < 100 else f"{x:,.1f}" if x < 1000 else f"{x:,.0f}").replace(",", " ").replace(".", ",")
print("| mittari | yksikkö | UD-IQ1_S 3.11. | EXL3 | ero | huom. |\n|---|---|---:|---:|---:|---|")
for name, unit, ref, val, hib, note in rows:
    d = "-" if val in (None, 0) else ("%+.1f %%" % (100 * (val - ref) / ref)).replace(".", ",")
    print(f"| {name} | {unit} | {fmt(float(ref))} | {fmt(val if val is None else float(val))} | {d} | {note} |")
