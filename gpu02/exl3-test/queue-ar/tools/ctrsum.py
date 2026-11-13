#!/usr/bin/env python3
"""ctrsum.py (10.11.2026, fn4 E6) -- per-kernel hardware-counter summary of rocprofiler-sdk counter_collection.csv files.

  ctrsum.py ROOT CONFIG   -> one line per kernel of the config: mean per dispatch of every counter collected over its
                             passes (ROOT/CONFIG-g*/**/*counter_collection.csv), the derived shares and a verdict
  ctrsum.py ROOT --table  -> every config's kernels as one table (ROOT/summary.tsv) and the verdict lines on stdout

Shares (per dispatch, means): VALUBusy / MemUnitBusy / OccupancyPercent / MeanOccupancyPerActiveCU / LdsUtil /
ValuPipeIssueUtil / L2CacheHit as rocprofiler-sdk derives them for the architecture; wait = SQ_WAIT_ANY / SQ_WAVE_CYCLES,
issue-wait = SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES, vmem = SQ_INST_CYCLES_VMEM / SQ_WAVE_CYCLES, valu-cyc =
SQ_INST_CYCLES_VALU / SQ_WAVE_CYCLES; VALU / LDS / SALU / texture-load instructions a wave; TCP miss rate.
Verdict (heuristic, stated with the numbers it rests on): ALU-bound if VALUBusy or ValuPipeIssueUtil >= 60 %;
memory-bound if MemUnitBusy >= 60 %; latency-bound if wait >= 50 % with VALUBusy < 40 % and MemUnitBusy < 50 %
(occupancy-limited too when OccupancyPercent < 40 %); otherwise mixed.
"""
import csv, glob, math, os, re, statistics, sys


def short(name):
    m = re.search(r"(exl3r4d_[A-Za-z0-9_]+|moe_gemm_[A-Za-z0-9_]+|gemm_[A-Za-z0-9_]+|[A-Za-z_][A-Za-z0-9_]*kernel[A-Za-z0-9_]*)", name or "")
    s = m.group(1) if m else (name or "?")
    t = re.search(r"I(L[a-z]\d+E|Li\d+E)+", name or "")      # template arguments, when the name is mangled
    return (s + ("<" + ",".join(re.findall(r"L[a-z](\d+)E", t.group(0))) + ">" if t else ""))[:80]


def load(files):
    per = {}     # kernel -> counter -> [values]; plus "_dur" (ns) and "_vgpr"
    for f in files:
        try:
            rd = csv.DictReader(open(f, newline=""))
        except OSError:
            continue
        cols = rd.fieldnames or []
        kc = next((c for c in cols if c.lower() == "kernel_name"), None)
        nc = next((c for c in cols if c.lower() == "counter_name"), None)
        vc = next((c for c in cols if c.lower() == "counter_value"), None)
        sc = next((c for c in cols if c.lower() == "start_timestamp"), None)
        ec = next((c for c in cols if c.lower() == "end_timestamp"), None)
        gc = next((c for c in cols if c.lower() in ("vgpr_count", "arch_vgpr_count")), None)
        dc = next((c for c in cols if c.lower() == "dispatch_id"), None)
        if not (kc and nc and vc):
            continue
        seen = set()
        for r in rd:
            k = short(r[kc]); d = per.setdefault(k, {})
            try:
                d.setdefault(r[nc], []).append(float(r[vc]))
            except ValueError:
                continue
            key = (f, r.get(dc) if dc else None)
            if key not in seen:
                seen.add(key)
                if sc and ec:
                    try:
                        d.setdefault("_dur", []).append(float(r[ec]) - float(r[sc]))
                    except ValueError:
                        pass
                if gc:
                    try:
                        d.setdefault("_vgpr", []).append(float(r[gc]))
                    except ValueError:
                        pass
    return per


def summarize(per):
    out = []
    for k, d in sorted(per.items()):
        m = {c: statistics.mean(v) for c, v in d.items() if v}
        n = len(d.get("_dur", [])) or max((len(v) for v in d.values()), default=0)
        wc = m.get("SQ_WAVE_CYCLES")
        sh = lambda c: (100.0 * m[c] / wc) if wc and c in m else None   # noqa: E731
        waves = m.get("SQ_WAVES")
        pw = lambda c: (m[c] / waves) if waves and c in m else None     # noqa: E731
        row = {
            "kernel": k, "dispatches": n, "us": (m["_dur"] / 1000.0) if "_dur" in m else None, "vgpr": m.get("_vgpr"),
            "VALUBusy": m.get("VALUBusy"), "ValuPipeIssueUtil": m.get("ValuPipeIssueUtil"), "MemUnitBusy": m.get("MemUnitBusy"),
            "Occupancy%": m.get("OccupancyPercent"), "OccActiveCU": m.get("MeanOccupancyPerActiveCU"), "LdsUtil": m.get("LdsUtil"),
            "LDSBankConflict": m.get("LDSBankConflict"), "L2Hit": m.get("L2CacheHit"),
            "wait%": sh("SQ_WAIT_ANY"), "issuewait%": sh("SQ_WAIT_INST_ANY"), "vmem%": sh("SQ_INST_CYCLES_VMEM"),
            "valucyc%": sh("SQ_INST_CYCLES_VALU"),
            "valu/wave": pw("SQ_INSTS_VALU"), "lds/wave": pw("SQ_INSTS_LDS"), "salu/wave": pw("SQ_INSTS_SALU"),
            "texld/wave": pw("SQ_INSTS_TEX_LOAD"), "waves": waves,
            "tcpmiss%": (100.0 * m["TCP_REQ_MISS"] / m["TCP_REQ"]) if m.get("TCP_REQ") and "TCP_REQ_MISS" in m else None,
            "fetchGBs": (m["FetchSize"] * 1e3 / m["_dur"]) if m.get("_dur") and "FetchSize" in m else None,   # FetchSize KB
        }
        vb, mb, w, oc = row["VALUBusy"], row["MemUnitBusy"], row["wait%"], row["Occupancy%"]
        vi = row["ValuPipeIssueUtil"]
        if (vb is not None and vb >= 60) or (vi is not None and vi >= 60):
            v = "ALU-bound"
        elif mb is not None and mb >= 60:
            v = "memory-bound"
        elif w is not None and w >= 50 and (vb is None or vb < 40) and (mb is None or mb < 50):
            v = "latency-bound" + (" + occupancy-limited" if oc is not None and oc < 40 else "")
        elif vb is None and mb is None and w is None:
            v = "no counters"
        else:
            v = "mixed"
        row["verdict"] = v
        out.append(row)
    return out


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def line(cfg, r):
    keys = ["us", "vgpr", "VALUBusy", "ValuPipeIssueUtil", "MemUnitBusy", "Occupancy%", "OccActiveCU", "wait%", "issuewait%",
            "vmem%", "valucyc%", "LdsUtil", "LDSBankConflict", "L2Hit", "tcpmiss%", "fetchGBs", "valu/wave", "lds/wave"]
    return f"{cfg} {r['kernel']} ({r['dispatches']} dispatches): " + ", ".join(f"{k} {fmt(r[k])}" for k in keys) + f" -> {r['verdict']}"


def main():
    root = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] != "--table":
        cfg = sys.argv[2]
        files = glob.glob(os.path.join(root, cfg + "-g*", "**", "*counter_collection.csv"), recursive=True)
        rows = summarize(load(files))
        if not rows:
            print(f"{cfg}: no counter rows ({len(files)} csv files)")
        for r in rows:
            print(line(cfg, r))
        return
    cfgs = sorted({os.path.basename(p).rsplit("-g", 1)[0] for p in glob.glob(os.path.join(root, "*-g*")) if os.path.isdir(p)})
    with open(os.path.join(root, "summary.tsv"), "w") as t:
        hdr = None
        for c in cfgs:
            files = glob.glob(os.path.join(root, c + "-g*", "**", "*counter_collection.csv"), recursive=True)
            for r in summarize(load(files)):
                if hdr is None:
                    hdr = ["config"] + list(r.keys()); t.write("\t".join(hdr) + "\n")
                t.write("\t".join([c] + [fmt(r[k], 3) for k in hdr[1:]]) + "\n")
                print(line(c, r))


if __name__ == "__main__":
    main()
