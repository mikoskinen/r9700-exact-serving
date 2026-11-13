#!/bin/bash
# 123-fn-ccreflen.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~75 min) --
# cc_sim pairs that do the SAME WORK with natural drafting: --gen-mode reflen (tools/ccsim/cc_sim2.py).
# WHY: 118's natural mode fixed the acceptance (k4n-n 0.913, k4nA-n 0.953) but the arms did different work (gen mean 1300
# vs 1919 tokens, prompt p50 78k vs 44k): greedy outputs diverge with the batch shape, and natural lengths follow them.
# reflen: every request's max_tokens = the length the reference run generated for the same (user, user turn), no
# ignore_eos (--reflen-eos stop): a reply that stops earlier stops ("short", logged), one that would run on is cut at
# the reference length ("cut": it would have run past the reference's natural stop); trace-seeded as before, so the tool
# results and the turn structure are the reference's. Reference = 118's k4n-n run (the newest
# runs/ccsim-*-fn4-k4n-n-u16.jsonl, X123_REF overrides).
# ARMS (93l X93_CC_ONLY: gate, slow-state probe, cc_sim u16 --turns 15 --warmup-turns 3 reflen + engine stats):
#   k4n-rl    1.15.1 + nearest (the reference's own line)          k4n-rl2   the same again: the reflen noise floor
#   k4nA-rl   + --draft-window adaptive
# PAIRS (cc_sim --pair): k4n-rl2 vs k4n-rl (floor), k4nA-rl vs k4n-rl; and each run's REFLEN classes (exact / cut / short
# shares, shortfall tokens).
# DRYRUN=1 bash 123-fn-ccreflen.sh : prints every command, runs and writes nothing.
set -u
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
CC2=; for f in "$H/tools/ccsim/cc_sim2.py" "$H/../tools/ccsim/cc_sim2.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/ccsim/cc_sim2.py" "$HOME/gpu/ds41/exl3-test/tools/ccsim/cc_sim2.py"; do [ -e "$f" ] && { CC2=$f; break; }; done
REF=${X123_REF:-$(ls -t $R/ccsim-*-fn4-k4n-n-u16.jsonl 2>/dev/null | head -1)}
B93=; for f in "$H/93l-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93l-fn-tp4ab.sh"; do [ -e "$f" ] && { B93=$f; break; }; done
if [ "$DRY" != 1 ]; then
  [ -n "$CC2" ] && [ -n "$B93" ] || { log "RESULT fn4 123: VIRHE cc_sim2.py or 93l-fn-tp4ab.sh missing"; exit 1; }
  [ -s "$REF" ] || { log "RESULT fn4 123: VIRHE no reference run (118's k4n-n jsonl) -- '$REF'"; exit 1; }
fi
log "RESULT fn4 123-fn-ccreflen: reference $REF ($( [ -s "$REF" ] && grep -c '"turn"' $REF || echo '?') requests), cc_sim2 $CC2 ($(md5sum $CC2 2>/dev/null | cut -c1-8))"
export ARMS=${X123_ARMS:-"k4n-rl k4n-rl2 k4nA-rl"} X93_CC_ONLY=1 X93_REF=${X93_REF:-k4n-rl} X93_REF_RERUN=0 X93_HANG428=0 X93_LACC=0 X93_PROF16=0
export X93_CCSIM=$CC2 X93_ITEM=${X93_ITEM:-123-fn-ccreflen}
export X93_CCARGS=${X93_CCARGS:-"--duration 3000 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode reflen --reflen-eos stop --ref $REF --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
DRYRUN=$DRY bash "$B93"; rc=$?
for a in $ARMS; do
  f=$(ls -t $R/ccsim-*-fn4-$a-u16.txt 2>/dev/null | head -1)
  if [ "$DRY" = 1 ]; then echo "+ REFLEN line of $R/ccsim-*-fn4-$a-u16.txt"
  else log "RESULT fn4 123 $a reflen classes: $( [ -s "$f" ] && grep -m1 '^REFLEN' $f | cut -c1-300 || echo "no run output")"; fi
done
log "RESULT fn4 123-fn-ccreflen: done (93l exit $rc)"
