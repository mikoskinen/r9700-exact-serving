#!/usr/bin/env python3
"""compare-after.py [RUNSDIR] (6.11.2026) -- after-sweep.sh's EXL3 numbers next to their references as one Markdown table
(value, reference, % difference; positive = EXL3 higher). Reference per row (column "viite"):
  3.11.   = the 3.11. UD-IQ1_S runs, re-parsed from ~/gpu/ds41/runs/{live0,prof0,profpf,onem*.txt,ds41-1103.log} with the same code
  RAM     = this window's EXL3 run in RAM Engram mode (window.sh probe / mb-c4 / mb-turns, cap ref140) for the NVMe rows
  prof0   = EXL3 prof0 (host pool 19456, Engram RAM) for the PR-config rows (host pool 4096, Engram disk)
  PR      = the PR author's published UD-IQ1_S numbers (deepseek41-wip.md, 62 GiB host, unprofiled)
Profile rows use profsum.py (steps from the dumps) on both sides. Missing values print '-'. Read only."""
import json, os, re, subprocess, sys
H = os.path.expanduser("~")
R = sys.argv[1] if len(sys.argv) > 1 else H + "/gpu/ds41/exl3-test/runs"
A, O = R + "/after-sweep", H + "/gpu/ds41/runs"
PS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profsum.py")
rows = []


def add(name, unit, ref, val, viite, note=""):
    rows.append((name, unit, ref, val, viite, note))


def pref(f, pre):
    try:
        return [json.loads(l[len(pre):]) for l in open(f, errors="replace") if l.startswith(pre)]
    except (FileNotFoundError, ValueError):
        return []


def first(f, pre):
    x = pref(f, pre)
    return x[0] if x else {}


def g(d, k):
    return d.get(k) if isinstance(d, dict) else None


def probe(f):
    return {x["prompt"]: x for x in pref(f, "J ")}


def sub(mode, d, *a):
    if not os.path.exists(os.path.join(d, "engine.log")):
        return {}
    try:
        out = subprocess.run([sys.executable, PS, mode, d] + [str(x) for x in a], capture_output=True, text=True, timeout=900).stdout
    except Exception:
        return {}
    tag = "PS " if mode == "phases" else "LV "
    return {r["phase"]: r for r in (json.loads(l[3:]) for l in out.splitlines() if l.startswith(tag))}


def cc(logf, tag, u):
    rx = re.compile(r"RESULT cc_sim %s u%d \[.*?\]: req/min ([0-9.]+) inflight ([0-9.]+) ttft p50/p95/max ([0-9.]+)/([0-9.]+)/([0-9.]+) s, "
                    r"gen/agent p50 ([0-9.]+) tok/s, total gen ([0-9.]+) tok/s, mean call ([-0-9.]+) s, prompt p50/max (\d+)/(\d+), cache ([0-9.]+)%%"
                    r".*?decode steps (\d+), ([0-9.]+) ms/decode step" % (re.escape(tag), u))
    last = None
    try:
        for l in open(logf, errors="replace"):
            m = rx.search(l)
            if m:
                last = m
    except FileNotFoundError:
        pass
    if not last:
        return {}
    v = [float(x) for x in last.groups()]
    return dict(req_min=v[0], ttft50=v[2], ttft95=v[3], agent=v[5], total=v[6], call=v[7], prompt50=v[8], cache=v[10], ms_dec=v[12])


def om(f):
    d = pref(f, "OM ")
    return [x for x in d if x.get("test") == "depth"], [x for x in d if x.get("test") == "hit-turn"]


# ---- live0 (unprofiled, --live): wall numbers and link traffic
for nm, a, b in (("live0", A + "/live0", O + "/live0"),):
    pe, po = probe(a + "/1u.txt"), probe(b + "/1u.txt")
    for k in ("essay", "code"):
        add(f"live0 1 käyttäjä {k}", "tok/s", g(po.get(k, {}), "decode_tps"), g(pe.get(k, {}), "decode_tps"), "3.11.")
        add(f"live0 1 käyttäjä {k}, askel", "ms", g(po.get(k, {}), "ms_per_decode_step"), g(pe.get(k, {}), "ms_per_decode_step"), "3.11.")
    for ph, key in (("c4", "ms_decode_step"), ("c4", "agg_dec_tps"), ("c8", "ms_decode_step"), ("c8", "agg_dec_tps"), ("pf", "ms_step_during"), ("pf", "prefill_tps")):
        add(f"live0 {ph} {key}", "", g(first(b + f"/{ph}.txt", "MB "), key), g(first(a + f"/{ph}.txt", "MB "), key), "3.11.",
            "3.11. live0 ei ajanut c8:aa" if ph == "c8" else "")
    le, lo = sub("live", a), sub("live", b)
    for ph in ("1u", "c4", "c8", "pf"):
        for k in ("stream", "ar", "h2d"):
            add(f"live0 {ph} linkki {k}", "MB/askel", g(g(lo.get(ph), "per_step_MB"), k), g(g(le.get(ph), "per_step_MB"), k), "3.11.")
        add(f"live0 {ph} asiantuntijaosuma VRAM", "%", g(lo.get(ph), "hit_mean"), g(le.get(ph), "hit_mean"), "3.11.")

# ---- prof0 (profiled, every 32): wall + op categories
pe, po = probe(A + "/prof0/1u.txt"), probe(O + "/prof0/1u.txt")
for k in ("essay", "code"):
    add(f"prof0 1 käyttäjä {k}, profiloitu askel", "ms", g(po.get(k, {}), "ms_per_decode_step"), g(pe.get(k, {}), "ms_per_decode_step"), "3.11.")
for ph, key in (("c4", "ms_decode_step"), ("c8", "ms_decode_step"), ("pf", "ms_step_during")):
    add(f"prof0 {ph} {key} (profiloitu)", "ms", g(first(O + f"/prof0/{ph}.txt", "MB "), key), g(first(A + f"/prof0/{ph}.txt", "MB "), key), "3.11.")
pse, pso = sub("phases", A + "/prof0", 32, 0), sub("phases", O + "/prof0", 32, 0)
for ph in ("1u", "c4", "c8", "pf", "deep"):
    e, o = pse.get(ph, {}), pso.get(ph, {})
    if not e and not o:
        continue
    for k, lab in (("op_ms_step", "op-aika yht."), ("dev_ms_step", "laiteoperaatiot"), ("host_ms_step", "isäntäoperaatiot"), ("wait_ms_step", "kaistan odotus")):
        add(f"prof0 {ph} {lab}", "ms/askel", g(o, k), g(e, k), "3.11.")
    cats = list(dict.fromkeys(list((e.get("cats") or {}).keys()) + list((o.get("cats") or {}).keys())))
    for c in cats:
        add(f"prof0 {ph} {c}", "ms/askel", (o.get("cats") or {}).get(c), (e.get("cats") or {}).get(c), "3.11.")
    for c, v in (e.get("moe") or {}).items():
        add(f"prof0 {ph} reititetty MoE: {c}", "ms/askel", (o.get("moe") or {}).get(c), v, "3.11.")

# ---- profpf (every step)
for key, f, lab in (("ms_per_step", "fc.txt", "profiloitu 1 käyttäjä"), ("prefill_tps", "alone.txt", "31k esitäyttö yksin"), ("prefill_tps", "pf.txt", "31k esitäyttö + 4 dekooderia")):
    pre = "FC " if f == "fc.txt" else "MB " if f == "pf.txt" else "J "
    add(f"profpf {lab}", key, g(first(O + "/profpf/" + f, pre), key), g(first(A + "/profpf/" + f, pre), key), "3.11.",
        "3.11. arvo todennäköisesti hitaasta tilasta" if f == "fc.txt" else "profiloitu")

# ---- prcfg (PR author's serve line) vs EXL3 prof0 and the PR author's numbers
pc, p0 = probe(A + "/prcfg/1u.txt"), probe(A + "/prof0/1u.txt")
for k, prv in (("essay", 77.4), ("code", 77.2)):
    add(f"PR-kokoonpano 1 käyttäjä {k} (profiloitu)", "tok/s", prv, g(pc.get(k, {}), "decode_tps"), "PR", "PR: UD-IQ1_S, profiloimaton, NST auto?")
    add(f"PR-kokoonpano 1 käyttäjä {k} (profiloitu)", "tok/s", g(p0.get(k, {}), "decode_tps"), g(pc.get(k, {}), "decode_tps"), "prof0")
add("PR-kokoonpano 16k esitäyttö (profiloitu)", "tok/s", 3746, g(first(A + "/prcfg/pf16k.txt", "J "), "prefill_tps"), "PR", "PR: UD-IQ1_S, profiloimaton")
psc = sub("phases", A + "/prcfg", 32, 0)
for ph in ("1u", "c4", "pf"):
    e, o = psc.get(ph, {}), pse.get(ph, {})
    if not e:
        continue
    for k, lab in (("op_ms_step", "op-aika yht."), ("dev_ms_step", "laiteoperaatiot"), ("host_ms_step", "isäntäoperaatiot (mapped/host-taso)"), ("wait_ms_step", "kaistan odotus")):
        add(f"PR-kokoonpano {ph} {lab}", "ms/askel", g(o, k), g(e, k), "prof0")
    for c in (e.get("cats") or {}):
        add(f"PR-kokoonpano {ph} {c}", "ms/askel", (o.get("cats") or {}).get(c), e["cats"][c], "prof0")
lc, l0 = sub("live", A + "/prcfg"), sub("live", A + "/live0")
for ph in ("1u", "c4", "pf"):
    for k in ("stream", "ssd", "h2d"):
        add(f"PR-kokoonpano {ph} linkki {k}", "MB/askel", g(g(l0.get(ph), "per_step_MB"), k), g(g(lc.get(ph), "per_step_MB"), k), "live0 EXL3")
    add(f"PR-kokoonpano {ph} asiantuntijaosuma VRAM", "%", g(l0.get(ph), "hit_mean"), g(lc.get(ph), "hit_mean"), "live0 EXL3",
        "tasot loppu: " + json.dumps(g(lc.get(ph), "tiers_end") or {}))

# ---- 1M probe, extra, fable-1m cc_sim
de, he = om(A + "/onem/onem.txt"); do, ho = om(O + "/onem.txt")
for lo, hi, lab in ((240000, 300000, "~266k"), (470000, 540000, "~501k"), (600000, 690000, "~644k")):
    e = next((x for x in de if lo <= (x.get("prompt_tokens") or 0) <= hi), {}); o = next((x for x in do if lo <= (x.get("prompt_tokens") or 0) <= hi), {})
    add(f"1M-tila kylmä esitäyttö {lab}", "tok/s", g(o, "prefill_tps"), g(e, "prefill_tps"), "3.11.", f"EXL3 {g(e, 'prompt_tokens')} tok")
    add(f"1M-tila dekoodiaskel {lab}", "ms", g(o, "ms_per_decode_step"), g(e, "ms_per_decode_step"), "3.11.")
add("1M-tila jatkovuoro syvimmässä", "s", g(ho[0] if ho else {}, "ttft"), g(he[0] if he else {}, "ttft"), "3.11.")
de, he = om(A + "/onem/onem-extra.txt"); do, ho = om(O + "/onem-728k.txt")
add("1M lisä ~700k esitäyttö", "tok/s", g(do[0] if do else {}, "prefill_tps"), g(de[0] if de else {}, "prefill_tps"), "3.11.")
add("1M lisä ~700k dekoodiaskel", "ms", g(do[0] if do else {}, "ms_per_decode_step"), g(de[0] if de else {}, "ms_per_decode_step"), "3.11.")
add("1M lisä ~700k jatkovuoro", "s", g(ho[0] if ho else {}, "ttft"), g(he[0] if he else {}, "ttft"), "3.11.")
ce, co = cc(R + "/ds41-exl3.log", "as-fable-1m", 4), cc(O + "/ds41-1103.log", "fable-1m", 4)
for k, lab, unit in (("ttft50", "TTFT p50", "s"), ("ttft95", "TTFT p95", "s"), ("agent", "per agentti p50", "tok/s"), ("total", "yhteensä", "tok/s"),
                     ("ms_dec", "dekoodiaskel", "ms"), ("cache", "välimuistiosuma", "%"), ("prompt50", "kehote p50", "tok")):
    add(f"fable-1m cc_sim 4 käyttäjää {lab}", unit, co.get(k), ce.get(k), "3.11.")

# ---- Engram on NVMe vs RAM (same window, same config)
pn, pr = probe(A + "/nvme/probe.txt"), probe(R + "/probe.txt")
for k in ("essay", "code"):
    add(f"NVMe-Engram 1 käyttäjä {k}", "tok/s", g(pr.get(k, {}), "decode_tps"), g(pn.get(k, {}), "decode_tps"), "RAM")
for f, key in (("mb-c4.txt", "ms_decode_step"), ("mb-c4.txt", "agg_dec_tps"), ("mb-turns.txt", "cold_ttft"), ("mb-turns.txt", "ttft_hit_mean")):
    add(f"NVMe-Engram {f[3:-4]} {key}", "", g(first(R + "/" + f, "MB "), key), g(first(A + "/nvme/" + f, "MB "), key), "RAM")
dn, hn = om(A + "/nvme/onem140.txt"); dr, hr = om(A + "/cap/onem140.txt")
for key, lab, unit in (("prefill_tps", "~140k kylmä esitäyttö", "tok/s"), ("ttft", "~140k TTFT", "s"), ("ms_per_decode_step", "~140k dekoodiaskel", "ms")):
    add(f"NVMe-Engram {lab}", unit, g(dr[0] if dr else {}, key), g(dn[0] if dn else {}, key), "RAM", "RAM-viite: cap-instanssi (MAXSEQS 16, MML 524288)")
add("NVMe-Engram ~140k jatkovuoro", "s", g(hr[0] if hr else {}, "ttft"), g(hn[0] if hn else {}, "ttft"), "RAM", "RAM-viite: cap-instanssi")

# ---- capacity (no 3.11. run above 8 users: reference = 3.11. u8 confirm run, for scale only)
c8o = cc(O + "/ds41-1103.log", "best-s8", 8)
for u in (12, 16):
    c = cc(R + "/ds41-exl3.log", "as-cap-s16", u)
    for k, lab, unit in (("ttft50", "TTFT p50", "s"), ("ttft95", "TTFT p95", "s"), ("agent", "per agentti p50", "tok/s"), ("total", "yhteensä", "tok/s"),
                         ("ms_dec", "dekoodiaskel", "ms"), ("req_min", "pyyntöä/min", "")):
        add(f"kapasiteetti u{u} (MAXSEQS 16, MML 524288) {lab}", unit, c8o.get(k), c.get(k), "3.11. u8", "viite on 8 käyttäjää: vain mittakaava")


def fmt(x):
    if x is None:
        return "-"
    x = float(x)
    return (f"{x:,.2f}" if abs(x) < 100 else f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}").replace(",", " ").replace(".", ",")


print("| mittari | yksikkö | viite | viitearvo | EXL3 | ero | huom. |\n|---|---|---|---:|---:|---:|---|")
for name, unit, ref, val, viite, note in rows:
    if ref is None and val is None:
        continue
    d = "-" if val is None or not ref else ("%+.1f %%" % (100 * (float(val) - float(ref)) / float(ref))).replace(".", ",")
    print(f"| {name} | {unit} | {viite} | {fmt(ref)} | {fmt(val)} | {d} | {note} |")
