#!/usr/bin/env python3
"""kl_line.py REPORT.json [REF_REPORT_FOR_DELTA] -- one RESULT line from radiance's KL report (core/kld.cpp report): mean,
median, p99, p99.9, top-1, PPL ratio, then each source's mean / top-1."""
import json, sys
r = json.load(open(sys.argv[1])); a = r["all"]; k = a["kld"]
s = (f"{a['positions']} pos ({r.get('ref_form')}): KL mean {k['mean']:.5f} median {k['median']:.6f} p99 {k['p99']:.4f} "
     f"p99.9 {k['p99_9']:.4f} max {k['max']:.3f}; top-1 {100 * a['top1_agreement']:.2f}%; PPL x{a['ppl']['ratio']:.4f}")
s += " | " + "; ".join(f"{n} {v['kld']['mean']:.5f}/{100 * v['top1_agreement']:.1f}%" for n, v in sorted(r.get("by_source", {}).items()))
print(s)
