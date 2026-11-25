#!/bin/bash
# 47-exl3-prefill-prof.sh (queue item, 6.11.2026) -- why prefill slows with context (UD-IQ1_S 3.11.: 16k 2,844 -> 644k 1,081
# tok/s, ~0.9 ns per context token per prompt token; EXL3 6.11.: 16.5k 2,343 -> 137k 1,978). Flags: MML 524288, MAXSEQS 4, NST 0,
# ckint 8192, MNBT 8192, no RAM tier, host pool 19456, ngram ram.
#   1. the 1.17 EXL3 image, gated (24 ms): fresh 16k, then 140k + 256k (ds41_tests.py prefill; 256k has no earlier number on this model)
#   2. the -profsync image, --profile-ops, RADIANCE_PROFILE_EVERY=1, RADIANCE_PROFILE_SYNC=1 (every device op timed on the host
#      clock between lane syncs; the dump header names each step's phase and n_tok): 16k first -- it must reach >= 0.33 x step 1's
#      16k tok/s, else one restart (SYNC costs an unknown share) -- then 140k + 256k. profchunks.py (cards 0 and 3): every prefill
#      chunk's device ms by op group, and per group a line over the chunk's context start = its growth in ns per context token per
#      prompt token. Hypothesis to test: the growth is the indexer (index_score on every card: the indexer weights are not sharded,
#      and layers 24/28/32/36 score the whole context although their top-k reads only layer 20's 2,048 candidate blocks).
# Estimate ~30 min (2-3 starts; 16k+140k+256k ~4 min unprofiled, profiled ~1.5-3x; the per-step dumps make the log ~0.5-1 GB).
# DRYRUN=1 bash 47-exl3-prefill-prof.sh : prints, writes nothing. Needs lib-117.sh, lib-pr266.sh, ds41-117-tp4.yaml, profchunks.py.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
P=$O/prefill-prof
pf16v() { python3 -c 'import json,sys; v=[d["prefill_tps"] for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J ")) if d.get("prefill_tps") and (d.get("prompt_tokens") or 0) < 30000]; print(v[0] if v else 0)' "$1" 2>/dev/null || echo 0; }
log "RESULT x117 47-prefill-prof: start IMAGE=$IMAGE / $X3P"
for im in $IMAGE $X3P; do [ "$DRY" = 1 ] || docker image inspect $im > /dev/null 2>&1 || { log "RESULT x117 47: VIRHE image $im not on gpu02"; exit 1; }; done
mkd $O $STATSDIR $P
export HOSTPOOL=19456 NGRAM=ram CKINT=8192 MNBT=8192 NST=0 MAXSEQS=4 MML=524288 WIRE=exact XARGS=
gate x117-pf-plain 24; rc=$?
[ $rc = 1 ] && { log "RESULT x117 47: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT x117 47: VIRHE slow state after one restart -> skipped"; down; exit 1; }
run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --reps 1 > $O/x117-pf-plain.txt 2>&1)"
run "(cd $D && timeout 2400 python3 ds41_tests.py prefill --depths 140000,256000 --cpt 3.95 --reps 1 >> $O/x117-pf-plain.txt 2>&1)"
[ "$DRY" = 1 ] || log "RESULT x117 47 prefill unprofiled (MML 524288 MAXSEQS 4 NST 0 ckint 8192, fresh) [guard $(cat $O/guard-x117-pf-plain.txt 2>/dev/null)]: $(pfline $O/x117-pf-plain.txt) | ref $REF_X3_MORNING; 3.11. UD-IQ1_S 1M mode ckint 2048: 266k 1805, 501k 1287, 644k 1081"
stopfile && { down; exit 0; }
export IMAGE=$X3P XARGS="--profile-ops" PROFEVERY=1 PROFSYNC=1
for try in 1 2; do
  restart x117-pf-prof || { log "RESULT x117 47: VIRHE profiled instance not healthy"; exit 1; }
  run "rm -f $P/engine.log; : > $P/marks.txt"
  follow $P; mark start $P
  run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --reps 1 > $P/prefill.txt 2>&1)"; mark pf16 $P
  [ "$DRY" = 1 ] && break
  slower "$(pf16v $O/x117-pf-plain.txt)" "$(pf16v $P/prefill.txt)" 3.0 || break
  log "RESULT x117 47 profiled try $try: 16k $(pf16v $P/prefill.txt) tok/s < 0.33 x unprofiled $(pf16v $O/x117-pf-plain.txt) -> slow state suspected$([ $try = 1 ] && echo ', restart')"
  [ $try = 1 ] && unfollow
done
run "(cd $D && timeout 3600 python3 ds41_tests.py prefill --depths 140000,256000 --cpt 3.95 --reps 1 >> $P/prefill.txt 2>&1)"; mark pf $P
unfollow
down
run "python3 $QD/profchunks.py $P/engine.log 0 --csv $P/chunks-r0.csv > $P/profchunks-r0.txt 2>&1; python3 $QD/profchunks.py $P/engine.log 3 --csv $P/chunks-r3.csv > $P/profchunks-r3.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT x117 47 prefill profiled (-profsync, every step, SYNC): $(pfline $P/prefill.txt) | growth by op group, card 0 (group ms@ctx0 ms-per-100k-ctx ns/ctx-token share@0 share@max): $(sed -n '3,10p' $P/profchunks-r0.txt | tr -s ' ' | tr '\n' ';' | cut -c1-900) -- runs/x117/prefill-prof/"
log "RESULT x117 47-prefill-prof: done"
