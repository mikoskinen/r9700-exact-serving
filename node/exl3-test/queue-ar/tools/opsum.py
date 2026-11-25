#!/usr/bin/env python3
"""opsum.py ENGINE.log [LABEL] -- the last --profile-ops table of every rank in an engine log ("per-op timing, rank N."),
summed by op class (8.11.2026, queue-ar, item 95): per rank the total device ms and the share of each class, plus the top
ops; and the spread of each class over the ranks (max - min, ms) so a replicated or unbalanced class shows. Classes by op
name: collective (all_reduce, ar_*, all_gather, reduce_scatter), hc (hc_read / hc_write / hc_* / mtp_enter), experts (moe_*,
router_*, gated_*quant*, moe_gather), attention (attn*, qsa*, rope, kv), gdn (gdn_*), dense (gemm_*, logits_gemm), other.
Rows marked `host` are left out of the device totals (they time host-site launches). Standard library only."""
import re, sys
from collections import defaultdict

CLS = [("collective", re.compile(r"^(all_reduce|ar_|all_gather|reduce_scatter)")),
       ("hc", re.compile(r"^(hc_|mtp_enter)")),
       ("experts", re.compile(r"^(moe_|router_|gated_|moe_gather)")),
       ("attention", re.compile(r"^(attn|qsa|qk_norm|rope|kv_)")),
       ("gdn", re.compile(r"^gdn_")),
       ("dense", re.compile(r"^(gemm|logits_gemm)"))]


def cls(op):
    for c, rx in CLS:
        if rx.search(op):
            return c
    return "other"


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    tabs = {}
    cur = None
    rx_head = re.compile(r"per-op timing, rank (\d+)\.")
    rx_row = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+\S+\s+\S+\s+(\S+)")
    for line in open(path, errors="replace"):
        m = rx_head.search(line)
        if m:
            cur = int(m.group(1)); tabs[cur] = []; continue
        if cur is None:
            continue
        m = rx_row.match(line)
        if m:
            tabs[cur].append((int(m.group(1)), float(m.group(2)), m.group(4), m.group(5)))
        elif line.strip() and not line.startswith(" ") and not line[0].isdigit() and tabs[cur]:
            cur = None   # the table ended
    if not tabs:
        print(f"{label}: no per-op table in {path}"); return
    per = {}
    for r in sorted(tabs):
        by = defaultdict(float); ops = defaultdict(float)
        for calls, ms, site, op in tabs[r]:
            if site != "dev":
                continue
            by[cls(op)] += ms; ops[op] += ms
        tot = sum(by.values()) or 1.0
        per[r] = by
        top = sorted(ops.items(), key=lambda kv: -kv[1])[:6]
        print(f"{label} rank {r}: device {tot:.0f} ms | " + " ".join(f"{c} {100 * by[c] / tot:.1f}%" for c, _ in CLS + [("other", 0)]) +
              " | top " + "; ".join(f"{o} {v:.0f}" for o, v in top))
    if len(per) > 1:
        print(f"{label} spread over ranks (max - min ms): " + " ".join(
            f"{c} {max(p[c] for p in per.values()) - min(p[c] for p in per.values()):.0f}" for c, _ in CLS + [("other", 0)]))


if __name__ == "__main__":
    main()
