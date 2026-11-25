#!/bin/bash
# ccsim-exl3.sh USERS TAG -- one cc_sim run on :8090 with CCARGS (default = the 3.11. confirm args: --duration 660 --warmup 300
# --effort xhigh,medium) + RESULT line (summary, engine metrics, mean call). = ~/gpu/ds41/ccsim-ds41c.sh writing to this kit's log.
set -u
U=$1; TAG=$2; . "$(dirname "$0")/env.sh"; CCARGS=${CCARGS:---duration 660 --warmup 300 --effort xhigh,medium --gen-mode forced --seed 1}   # 6.11. 21:30 user: forced lengths (TESTING.md standard)
B=$HOME/gpu/bench; TS=$(date +%m%d-%H%M); OUT=$R/ccsim-$TS-$TAG-u$U
cd $B
curl -s $URL/metrics > $OUT.metrics-before.txt
timeout 3000 python3 cc_sim.py --urls $URL --users $U $CCARGS --tag ds41exl3-$TAG-u$U --out $OUT.jsonl > $OUT.txt 2>&1
echo "cc_sim exit $?" >> $OUT.txt
curl -s $URL/metrics > $OUT.metrics-after.txt
python3 $D/mdelta.py $OUT.metrics-before.txt $OUT.metrics-after.txt > $OUT.mdelta.txt 2>&1
python3 - $OUT.jsonl $U "$TAG" "$CCARGS" "$(grep DERIVED $OUT.mdelta.txt | cut -c9-)" <<'PY' | while read -r line; do echo "$(date '+%F %T') $line" >> $L; echo "$line"; done
import json, statistics, sys, importlib.util, os
f, U, tag, args, eng = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
cfg, recs = None, []
for l in open(f):
    d = json.loads(l)
    if "config" in d: cfg = d["config"]
    elif "t" in d and "url" in d: recs.append(d)
w0, w1 = cfg["warmup"], cfg["duration"]
win = [r for r in recs if w0 <= r["t"] < w1 and not r.get("err") and r.get("ttft") is not None]
mean_total = statistics.mean(r["total"] for r in win) if win else None
spec = importlib.util.spec_from_file_location("cc", os.path.expanduser("~/gpu/bench/cc_sim.py")); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
s = cc.summarize(recs, w0, w1)
print("RESULT cc_sim %s u%d [%s]: req/min %s inflight %s ttft p50/p95/max %s/%s/%s s, gen/agent p50 %s tok/s, total gen %s tok/s, mean call %.1f s, "
      "prompt p50/max %s/%s, cache %s%%, uncached p50 %s, gen p50/mean %s/%s, err %s, cold starts in window %s | engine: %s" % (
      tag, U, args, s["req_per_min"], s["mean_inflight"], s["ttft_p50"], s["ttft_p95"], s["ttft_max"], s["decode_p50"], s["agg_gen_tps"],
      mean_total or -1, s["prompt_p50"], s["prompt_max"], s["cache_hit_pct"], s["uncached_p50"], s["gen_p50"], s["gen_mean"], s["errors"],
      s["cold_starts"], eng))
PY
echo "CCSIM DONE $OUT"
