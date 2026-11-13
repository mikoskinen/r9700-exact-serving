#!/bin/bash
# 46-exl3-mtp5-prof.sh (queue item, 6.11.2026) -- where the DSpark NST 5 round's 45-53 ms go (plain step 19.4 ms; the Volta 8x V100
# build's round is ~23 ms with every expert in VRAM). 6.11. sweep 'base' flags: NST 5, ckint 8192, MAXSEQS 8, RAM KV tier, no
# span/IL, host pool 19456, ngram ram.
#   1. 1.17 EXL3 image with RADIANCE_AQL_STAMP=10: every dispatch of one recorded pass in 10 on card 0 gets a completion stamp,
#      barriers and chains unchanged (--profile-ops plays no recorded pass). stepprobe essay/code 512 + mb c4, server down (the CSV
#      is written at exit), stampgaps.py: per kind of pass (draft / verify / decode) wall, busy, idle share, launch gaps, kernels.
#      Gate 70 ms (NST 5 rounds 45-53; the slow state ~2x).
#   2. the -profsync image with --profile-ops (dump every 32, event timing): stepprobe + mb c4 75 s, profsum.py phases + profmoe.py.
#      Not gated by fastcheck (profiling slows it): the profiled code ms/round must be <= 3.0 x step 1's (EXL3 profiling costs ~2.5x
#      consistently, the slow state ~4x), else one restart; logged and kept after that.
# The plain NST 0 profile to read the rounds against is after-sweep prof0 (queue/10-prof-retry.sh); not re-run here.
# Estimate ~32 min (2-3 starts ~4 min; warm-up 1.5; probe 1 x2; c4 3.6 + 1.5; parsing 1).
# DRYRUN=1 bash 46-exl3-mtp5-prof.sh : prints, writes nothing. Needs lib-117.sh, lib-pr266.sh, ds41-117-tp4.yaml, stampgaps.py in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
P=$O/mtp5-prof
log "RESULT x117 46-mtp5-prof: start IMAGE=$IMAGE / $X3P"
for im in $IMAGE $X3P; do [ "$DRY" = 1 ] || docker image inspect $im > /dev/null 2>&1 || { log "RESULT x117 46: VIRHE image $im not on gpu02"; exit 1; }; done
mkd $O $STATSDIR $P
run "rm -f $STATSDIR/stamps.csv"
export HOSTPOOL=19456 NGRAM=ram CKINT=8192 MNBT=8192 NST=5 MAXSEQS=8 MML=524288 WIRE=exact XARGS="$RAMTIER"
export AQLSTAMP=10
gate x117-mtp5-stamp 70; rc=$?
[ $rc = 1 ] && { log "RESULT x117 46: VIRHE stamped instance not healthy"; exit 1; }
[ $rc = 2 ] && { log "RESULT x117 46: VIRHE stamped instance slow after one restart -> skipped"; down; exit 1; }
probe x117-mtp5-stamp
mbp x117-mtp5-stamp c4
down
run "python3 $QD/stampgaps.py $STATSDIR/stamps.csv x117-mtp5 --top 15 > $O/x117-mtp5-stampgaps.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT x117 46 stamped: $(head -c 1500 $O/x117-mtp5-stampgaps.txt | tr '\n' ' ' | tr -s ' ') -- runs/x117/x117-mtp5-stampgaps.txt | ref $REF_X3_NST5"
stopfile && exit 0
export AQLSTAMP= PROFEVERY=32 IMAGE=$X3P XARGS="$RAMTIER --profile-ops"
REFMS=$(codems $O/x117-mtp5-stamp-probe.txt)
for try in 1 2; do
  restart x117-mtp5-prof || { log "RESULT x117 46: VIRHE profiled instance not healthy"; exit 1; }
  run "rm -f $P/engine.log; : > $P/marks.txt"
  follow $P; mark start $P
  run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $P/1u.txt 2>&1)"; mark 1u $P
  [ "$DRY" = 1 ] && break
  slower "$(codems $P/1u.txt)" "$REFMS" 3.0 || break
  log "RESULT x117 46 profiled try $try: code $(codems $P/1u.txt) ms/round > 3.0 x stamped $REFMS -> slow state suspected$([ $try = 1 ] && echo ', restart')"
  [ $try = 1 ] && unfollow
done
run "(cd $D && timeout 1200 python3 mb.py --tag x117-mtp5-prof --phases c4 --dur 75 --warm 15 > $P/c4.txt 2>&1)"; mark c4 $P
unfollow
down
run "python3 $T/profsum.py phases $P 32 0 > $P/profsum.txt 2>&1; python3 $T/profmoe.py $P 32 1u,c4 > $P/profmoe.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT x117 46 profiled (-profsync image, --profile-ops every 32, event timing): 1u $(python3 $D/sweepfmt.py probe $P/1u.txt 2>&1 | cut -c1-300); c4 $(grep -o '"ms_decode_step": [0-9.]*' $P/c4.txt) | profsum: $(grep -iE 'ms/step|total' $P/profsum.txt 2>/dev/null | head -6 | tr '\n' ' ' | cut -c1-900) -- runs/x117/mtp5-prof/"
log "RESULT x117 46-mtp5-prof: done"
