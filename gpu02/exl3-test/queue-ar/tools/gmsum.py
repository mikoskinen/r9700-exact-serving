#!/usr/bin/env python3
"""gmsum.py GM.txt LABEL --from NS --to NS [--json OUT] -- one window of gmsample.py's file, per rank: samples, power (median, p90),
sclk p10/p50/p90, current uclk and fclk / mclk level shares, temperatures (max), and the share of samples with each
ASIC-independent throttler bit set (amdgpu SMU_THROTTLER_*: 0 PPT0, 1 PPT1, 4 SPL, 16 TDC_GFX, 17 TDC_SOC, 18 TDC_MEM,
22 EDC_GFX, 32 TEMP_GPU, 34 TEMP_MEM, 35 TEMP_EDGE, 36 TEMP_HOTSPOT, 38 TEMP_VR_GFX, 40 TEMP_VR_MEM0, 44 VRHOT0, 57 FIT).
Standard library only."""
import sys
from collections import Counter, defaultdict

NAMES = {0: "PPT0", 1: "PPT1", 2: "PPT2", 3: "PPT3", 4: "SPL", 5: "FPPT", 6: "SPPT", 16: "TDC_GFX", 17: "TDC_SOC",
         18: "TDC_MEM", 19: "TDC_VDD", 22: "EDC_GFX", 23: "APCC", 32: "TEMP_GPU", 33: "TEMP_CORE", 34: "TEMP_MEM",
         35: "TEMP_EDGE", 36: "TEMP_HOTSPOT", 37: "TEMP_SOC", 38: "TEMP_VR_GFX", 39: "TEMP_VR_SOC", 40: "TEMP_VR_MEM0",
         41: "TEMP_VR_MEM1", 44: "VRHOT0", 45: "VRHOT1", 56: "PPM", 57: "FIT"}


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else 0


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--from", "--to", "--json"):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1] if k == "--json" else int(a[i + 1]); del a[i:i + 2]
    path, label = a[0], (a[1] if len(a) > 1 else a[0])
    tf, tt = opt.get("--from", 0), opt.get("--to", 1 << 62)
    M = defaultdict(list); F = defaultdict(list)
    for line in open(path):
        p = line.split()
        if not p or p[0] == "#":
            continue
        try:
            r, t = int(p[1]), int(p[2])
        except (IndexError, ValueError):
            continue
        if not (tf <= t <= tt):
            continue
        if p[0] == "M" and len(p) >= 13:   # M rank mono gfx% umc% W edge hot mem sclk uclk avg_sclk indep_hex
            M[r].append([int(x) for x in p[3:12]] + [int(p[12], 16)])
        elif p[0] == "F" and len(p) >= 5:
            F[r].append((int(p[3]), int(p[4])))
    out = []; js = {}
    for r in sorted(set(M) | set(F)):
        m = M[r]
        if not m:
            continue
        pw = [x[2] for x in m]; sc = [x[6] for x in m]
        uc = Counter(x[7] for x in m)
        bits = Counter()
        for x in m:
            v = x[9]
            for b in range(64):
                if v >> b & 1:
                    bits[b] += 1
        fc = Counter(f for f, _ in F[r]); mc = Counter(c for _, c in F[r])
        n = len(m)
        js[r] = {"n": n, "power_mean_w": sum(pw) / n, "power_p50_w": q(pw, .5), "sclk_p50": q(sc, .5), "sclk_p10": q(sc, .1),
                 "uclk_top": uc.most_common(1)[0][0], "fclk_top": fc.most_common(1)[0][0] if F[r] else None,
                 "edge_max": max(x[3] for x in m), "hot_max": max(x[4] for x in m), "mem_max": max(x[5] for x in m),
                 "hot_mean": sum(x[4] for x in m) / n,
                 "throttle_share": {NAMES.get(b, str(b)): v / n for b, v in bits.items()}}
        out.append(f"r{r}: n={n} power {q(pw, .5)}/{q(pw, .9)} W, sclk {q(sc, .1)}/{q(sc, .5)}/{q(sc, .9)}, umc% p50 "
                   f"{q([x[1] for x in m], .5)}, uclk " + ",".join(f"{k}:{100 * v / n:.0f}%" for k, v in uc.most_common(3)) +
                   (", fclk " + ",".join(f"{k}:{100 * v / len(F[r]):.0f}%" for k, v in fc.most_common(3)) if F[r] else "") +
                   (", mclk " + ",".join(f"{k}:{100 * v / len(F[r]):.0f}%" for k, v in mc.most_common(2)) if F[r] else "") +
                   f", T max edge/hot/mem {max(x[3] for x in m)}/{max(x[4] for x in m)}/{max(x[5] for x in m)} C, throttle " +
                   (",".join(f"{NAMES.get(b, b)}:{100 * v / n:.0f}%" for b, v in bits.most_common(5)) or "none"))
    print(f"{label} gpu_metrics: " + " | ".join(out) if out else f"{label} gpu_metrics: no samples in the window")
    if "--json" in opt:
        import json
        json.dump({"label": label, "ranks": js}, open(opt["--json"], "w"))


if __name__ == "__main__":
    main()
