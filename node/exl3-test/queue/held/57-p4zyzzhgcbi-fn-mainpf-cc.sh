#!/bin/bash
# 126b-fn-mainpf-cc.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~55 min) --
# 126's two main-line arms as a cc_sim pair in REFLEN mode (cc_sim2 --gen-mode reflen, validated by 123 just before):
#   mpA-rl  main c16-sw + F1n, interleave 0, nearest, MNBT 2048     mpB-rl  + MNBT 4096 + --prefill-span 2
# cc_sim u16 --turns 15 --warmup-turns 3, every request held to the length 118's k4n-n run generated for it (the same
# work for both arms; --reflen-eos stop), trace-seeded, greedy; engine stats around each run; pair mpB-rl vs mpA-rl and
# each run's REFLEN classes. = 93n-fn-tp4ab.sh beside this file with X93_CC_ONLY=1.
# DRYRUN=1 bash 126b-fn-mainpf-cc.sh : prints every command, runs and writes nothing.
set -u
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
CC2=; for f in "$H/tools/ccsim/cc_sim2.py" "$H/../tools/ccsim/cc_sim2.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/ccsim/cc_sim2.py"; do [ -e "$f" ] && { CC2=$f; break; }; done
REF=${X126_REF:-$(ls -t $R/ccsim-*-fn4-k4n-n-u16.jsonl 2>/dev/null | head -1)}
B93=; for f in "$H/93n-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93n-fn-tp4ab.sh"; do [ -e "$f" ] && { B93=$f; break; }; done
if [ "$DRY" != 1 ]; then
  [ -n "$CC2" ] && [ -n "$B93" ] || { log "RESULT fn4 126b: VIRHE cc_sim2.py or 93n-fn-tp4ab.sh missing"; exit 1; }
  [ -s "$REF" ] || { log "RESULT fn4 126b: VIRHE no reference run (118's k4n-n jsonl) -- '$REF'"; exit 1; }
fi
log "RESULT fn4 126b-fn-mainpf-cc: reference $REF, cc_sim2 $CC2 ($(md5sum $CC2 2>/dev/null | cut -c1-8))"
export ARMS=${X126B_ARMS:-"mpA-rl mpB-rl"} X93_CC_ONLY=1 X93_REF=${X93_REF:-mpA-rl} X93_REF_RERUN=0 X93_HANG428=0 X93_LACC=0 X93_PROF16=0
export X93_CCSIM=$CC2 X93_ITEM=${X93_ITEM:-126b-fn-mainpf-cc}
export X93_CCARGS=${X93_CCARGS:-"--duration 3000 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode reflen --reflen-eos stop --ref $REF --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
DRYRUN=$DRY bash "$B93"; rc=$?
for a in $ARMS; do
  f=$(ls -t $R/ccsim-*-fn4-$a-u16.txt 2>/dev/null | head -1)
  if [ "$DRY" = 1 ]; then echo "+ REFLEN line of $R/ccsim-*-fn4-$a-u16.txt"
  else log "RESULT fn4 126b $a reflen classes: $( [ -s "$f" ] && grep -m1 '^REFLEN' $f | cut -c1-300 || echo "no run output")"; fi
done
log "RESULT fn4 126b-fn-mainpf-cc: done (93n exit $rc)"
