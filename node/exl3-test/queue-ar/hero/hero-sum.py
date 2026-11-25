#!/usr/bin/env python3
"""hero-sum.py RUNDIR [RUNDIR ...] (14.11.2026) -- the summary of hero-ds.sh runs: every results cell with the raw lines it
comes from, the switch read-back, the identity lines, and temperatures / power per section. Standard library only.
Writes RUNDIR/hero-sum.txt and RUNDIR/hero-sum.json and prints the text.

Cells (all from the run folder's own files):
  1 user, prose   mean decode_tps of u1.jsonl labels essay-b1-r1..3 and essay-b2-r1..3 (the -w reads are discarded warm-ups)
  1 user, code    mean decode_tps of u1.jsonl labels code-r1..3
  N users         mean agg_dec_tps of mb.jsonl names cN-b1 and cN-b2 (c4-w is a discarded warm-up block; c1-g1 and c4-g1 are
                  single gens-1 blocks, read only for the comparison with other runs)
  prefill D       cold.jsonl depth D: prompt_tokens / ttft (cached must be 0)
TTFT p90 of a block is recomputed from mb/<name>-reqs.jsonl the way the harness does it (requests whose first token falls
between the block's two marks). Temperatures and power: temps.csv rows between marks.jsonl's start / end of each block.
"""
import gzip, json, os, re, statistics, sys

CELLS_U1 = [("1 user, prose", "essay", ["essay-b1-r1", "essay-b1-r2", "essay-b1-r3", "essay-b2-r1", "essay-b2-r2", "essay-b2-r3"]),
            ("1 user, code", "code", ["code-r1", "code-r2", "code-r3"])]
CELLS_MB = [("4 users", 4, ["c4-b1", "c4-b2"]), ("8 users", 8, ["c8-b1", "c8-b2"]), ("16 users", 16, ["c16-b1", "c16-b2"]),
            ("24 users", 24, ["c24-b1", "c24-b2"]), ("32 users", 32, ["c32-b1", "c32-b2"])]
INFO_MB = [("1 stream, mb gens 1 (comparison only)", 1, ["c1-g1"]), ("4 users, mb gens 1 (comparison only)", 4, ["c4-g1"])]
CELLS_PF = [("prefill 16k", 16000), ("prefill 64k", 64000), ("prefill 128k", 130000), ("prefill 136k", 136000)]


def jl(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError: pass
    return out


def p90_win(reqs):
    rows, marks = [], []
    for r in reqs:
        if r.get("ev") == "mark": marks.append(r.get("t"))
        elif r.get("ev") == "done" and not r.get("err") and r.get("ttft") is not None: rows.append(r)
    if len(marks) >= 2: rows = [r for r in rows if marks[0] <= (r.get("t_first") or 0) <= marks[1]]
    tt = sorted(r["ttft"] for r in rows)
    return (tt[min(len(tt) - 1, int(0.9 * (len(tt) - 1) + 0.5))] if tt else None), len(tt)


def temps_between(rows, t0, t1):
    by = {}
    for r in rows:
        if t0 <= r[0] <= t1: by.setdefault(r[1], []).append(r)
    out = {}
    for pci, rs in sorted(by.items()):
        out[pci] = dict(n=len(rs), power_mean_w=round(statistics.mean(x[2] for x in rs), 1), power_max_w=max(x[2] for x in rs),
                        junction_max_c=max(x[3] for x in rs), memory_max_c=max(x[4] for x in rs), fan_rpm_max=max(x[5] for x in rs),
                        sclk_mhz_median=statistics.median(x[6] for x in rs))
    return out


def summ(d):
    S = dict(run=os.path.basename(os.path.normpath(d)), cells=[], info=[], identity=[], switches={}, sections=[], notes=[])
    log = open(os.path.join(d, "run.log")).read().splitlines() if os.path.exists(os.path.join(d, "run.log")) else []
    marks = jl(os.path.join(d, "marks.jsonl"))
    tm = {}
    for m in marks: tm.setdefault((m["section"], m["name"]), {})[m["ev"]] = m["t"]
    trows = []
    try:
        for l in open(os.path.join(d, "temps.csv")):
            p = l.strip().split(",")
            if len(p) == 7 and p[0][0].isdigit():
                trows.append((float(p[0]), p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5] or 0), int(p[6] or 0)))
    except OSError: pass

    u1 = {r.get("label"): r for r in jl(os.path.join(d, "u1.jsonl"))}
    for cell, prompt, labels in CELLS_U1:
        vals = [u1[l]["decode_tps"] for l in labels if l in u1 and u1[l].get("status") == 200]
        if not vals: continue
        t0 = min(tm.get(("u1", l), {}).get("start", 1e18) for l in labels); t1 = max(tm.get(("u1", l), {}).get("end", 0) for l in labels)
        S["cells"].append(dict(cell=cell, value=round(statistics.mean(vals), 2), unit="tok/s", reads=vals, n=len(vals),
                               spread_pct=round(100 * (max(vals) - min(vals)) / statistics.mean(vals), 2),
                               ms_per_step=round(statistics.mean(u1[l]["ms_per_decode_step"] for l in labels if l in u1), 2),
                               tok_per_step=round(statistics.mean(u1[l]["tokens_per_step"] for l in labels if l in u1), 2),
                               acceptance=round(statistics.mean(u1[l]["acceptance"] for l in labels if l in u1), 3),
                               text_md5=sorted({u1[l].get("text_md5") for l in labels if l in u1}),
                               discarded=[u1[w]["decode_tps"] for w in u1 if w.startswith(prompt) and w.endswith("w") and "decode_tps" in u1[w]],
                               source="u1.jsonl labels " + ", ".join(labels) + " (raw: u1/<label>.txt J line, u1/<label>-text.json)",
                               temps=temps_between(trows, t0, t1)))
    mb = {r.get("name"): r for r in jl(os.path.join(d, "mb.jsonl"))}
    for group, cells in (("cells", CELLS_MB), ("info", INFO_MB)):
        for cell, n, names in cells:
            rs = [mb[x] for x in names if x in mb]
            if not rs: continue
            vals = [r["agg_dec_tps"] for r in rs]
            p90s = [p90_win(jl(os.path.join(d, "mb", x + "-reqs.jsonl")))[0] for x in names if x in mb]
            t0 = min(tm.get(("mb", x), {}).get("start", 1e18) for x in names); t1 = max(tm.get(("mb", x), {}).get("end", 0) for x in names)
            S[group].append(dict(cell=cell, value=round(statistics.mean(vals), 2), unit="tok/s", blocks=vals, n=len(vals),
                                 spread_pct=round(100 * (max(vals) - min(vals)) / statistics.mean(vals), 2) if len(vals) > 1 else 0.0,
                                 gens=rs[0].get("gens"), ms_decode_step=[r.get("ms_decode_step") for r in rs],
                                 tok_per_decode_step=[r.get("tok_per_decode_step") for r in rs], acceptance=[r.get("acceptance") for r in rs],
                                 ttft_p50_s=[r.get("ttft_p50") for r in rs], ttft_p90_s=p90s,
                                 per_stream_p50=[r.get("per_stream_dec_p50") for r in rs], requests=[r.get("requests") for r in rs],
                                 errors=[r.get("errors") for r in rs],
                                 source="mb.jsonl names " + ", ".join(names) + " (raw: mb/<name>.txt MB line, mb/<name>-reqs.jsonl)",
                                 temps=temps_between(trows, t0, t1)))
    cold = {r.get("depth"): r for r in jl(os.path.join(d, "cold.jsonl"))}
    for cell, dep in CELLS_PF:
        r = cold.get(dep)
        if not r or not r.get("ttft"): continue
        toks = r["prompt_tokens"] - (r.get("cached") or 0)
        t = tm.get(("cold", str(dep)), {})
        S["cells"].append(dict(cell=cell, value=round(toks / r["ttft"], 1), unit="prefill tok/s", ttft_s=r["ttft"], prompt_tokens=r["prompt_tokens"],
                               cached=r.get("cached"), status=r.get("status"), depth_arg=dep,
                               source="cold.jsonl depth %d (raw: cold/cold-%d.out DEEPQ line, cold/cold-%d.txt.json)" % (dep, dep, dep),
                               temps=temps_between(trows, t.get("start", 0), t.get("end", 0))))
    try: S["identity"] = open(os.path.join(d, "identity.txt")).read().splitlines()
    except OSError: pass
    for l in log:
        m = re.search(r"switches (at start \(before switches\.sh\)|after switches\.sh|at the end): (.*)$", l)
        if m: S["switches"][m.group(1)] = m.group(2)
        if re.search(r"switches \d+ of \d+", l) and "DIFFERENT" in l: S["notes"].append("switch drift: " + l)
        if " COOL " in l: S["sections"].append(l.split(" ", 2)[2])
        for key in ("HEALTHY after", "check answer:", "run/switches.sh exit", "engine log: error lines", "POWER", "VIRHE", "start lines:", "ld16:"):
            if key in l: S["notes"].append(l[20:])
    try:
        eng = gzip.open(os.path.join(d, "engine.log.gz"), "rt", errors="replace").read().splitlines()
        S["engine"] = dict(lines=len(eng), settle=[x[31:300] for x in eng if "settled at" in x][:8],
                           headroom_warn=[x[31:300] for x in eng if "this engine allows" in x][:4],
                           errors=sum(1 for x in eng if re.search(r"device error|hipError|HSA_STATUS_ERROR|memory access fault|page fault|out of memory|segfault|did not load", x, re.I)))
    except OSError: pass
    if trows:
        S["whole_run_temps"] = temps_between(trows, 0, 1e18)
    return S


def text(S):
    o = ["%s" % S["run"], ""]
    o.append("| cell | value | detail | source |")
    o.append("|---|---|---|---|")
    for c in S["cells"] + S["info"]:
        if c["unit"] == "prefill tok/s":
            det = "TTFT %.2f s, prompt %d tok, cached %s, status %s" % (c["ttft_s"], c["prompt_tokens"], c["cached"], c["status"])
        elif "reads" in c:
            det = "reads %s (spread %.2f %%), %.2f ms / %.2f tok a step, acceptance %.3f, discarded warm-ups %s, answer md5 %s" % (
                "/".join("%.2f" % v for v in c["reads"]), c["spread_pct"], c["ms_per_step"], c["tok_per_step"], c["acceptance"], c["discarded"], ",".join(c["text_md5"]))
        else:
            det = "blocks %s (gens %s, spread %.2f %%), ms a decode step %s, tok a step %s, acceptance %s, TTFT p50 %s / p90 %s s, per-stream p50 %s, requests %s, errors %s" % (
                "/".join("%.1f" % v for v in c["blocks"]), c["gens"], c["spread_pct"], c["ms_decode_step"], c["tok_per_decode_step"], c["acceptance"],
                c["ttft_p50_s"], c["ttft_p90_s"], c["per_stream_p50"], c["requests"], c["errors"])
        tt = c.get("temps") or {}
        th = "; ".join("%s %.0f W avg / %d W max, j %d / m %d C" % (k, v["power_mean_w"], v["power_max_w"], v["junction_max_c"], v["memory_max_c"]) for k, v in tt.items())
        o.append("| %s | %s %s | %s | %s |" % (c["cell"], c["value"], c["unit"], det + (" | " + th if th else ""), c["source"]))
    o += ["", "switches:"] + ["  %s: %s" % kv for kv in S["switches"].items()]
    o += ["", "identity:"] + ["  " + x for x in S["identity"]]
    o += ["", "sections (cool-down):"] + ["  " + x for x in S["sections"]]
    o += ["", "notes:"] + ["  " + x for x in S["notes"]]
    if S.get("engine"): o += ["", "engine log: %s" % json.dumps(S["engine"])]
    if S.get("whole_run_temps"): o += ["", "whole run temps: %s" % json.dumps(S["whole_run_temps"])]
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    for d in sys.argv[1:]:
        S = summ(d)
        json.dump(S, open(os.path.join(d, "hero-sum.json"), "w"), indent=1)
        t = text(S)
        open(os.path.join(d, "hero-sum.txt"), "w").write(t)
        sys.stdout.write(t)
