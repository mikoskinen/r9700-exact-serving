#!/usr/bin/env python3
"""pwsum.py O TAG [TAG...] -- item 91's per-arm table (8.11.2026, queue-ar): for each arm tag (fn4-d4a, fn4-d4a-p300, ...)
the power cap it ran at ($O/<tag>-cap.txt), mb c4 / c8 / c16 (aggregate decode tok/s, ms/step), the slow-layer share per
card at c8 ($O/<tag>-c8-rank.json), and from $O/<tag>-c8-gm.json the mean power, sclk p50, PPT0 / TEMP_HOTSPOT throttler
share and the max junction / mem temperatures (over the cards). One line an arm. Standard library only."""
import json, os, sys


def mb(O, tag, p):
    try:
        for line in open(os.path.join(O, f"{tag}-{p}.txt")):
            if line.startswith("MB "):
                return json.loads(line[3:])
    except (OSError, ValueError):
        pass
    return {}


def js(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def main():
    O, tags = sys.argv[1], sys.argv[2:]
    for tag in tags:
        try:
            cap = open(os.path.join(O, f"{tag}-cap.txt")).read().strip()
        except OSError:
            cap = "?"
        parts = [f"{tag} cap {cap}"]
        for p in ("c4", "c8", "c16"):
            d = mb(O, tag, p)
            if d:
                parts.append(f"{p} {d.get('agg_dec_tps')} tok/s {d.get('ms_decode_step')} ms")
        r = js(os.path.join(O, f"{tag}-c8-rank.json"))
        if r.get("card"):
            parts.append("slow c8 " + "/".join(f"{100 * c.get('slow_share', 0):.1f}" for c in r["card"]) + " %")
            co = r.get("coincide", {}).get("observed")
            if co:
                parts.append("slow-at-once " + "/".join(f"{100 * x:.0f}" for x in co) + " %")
        g = js(os.path.join(O, f"{tag}-c8-gm.json")).get("ranks", {})
        if g:
            v = list(g.values())
            n = len(v)
            parts.append(f"W mean {sum(x['power_mean_w'] for x in v) / n:.0f} sclk p50 {sum(x['sclk_p50'] for x in v) / n:.0f} "
                         f"PPT0 {100 * sum(x['throttle_share'].get('PPT0', 0) for x in v) / n:.0f} % "
                         f"HOTSPOT {100 * sum(x['throttle_share'].get('TEMP_HOTSPOT', 0) for x in v) / n:.0f} % "
                         f"Tmax hot/mem {max(x['hot_max'] for x in v)}/{max(x['mem_max'] for x in v)} C")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
