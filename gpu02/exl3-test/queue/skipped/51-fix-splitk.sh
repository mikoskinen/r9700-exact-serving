#!/bin/bash
# 51-fix-splitk.sh (queue item, 6.11.2026) -- the EXL3 decode split-K rules on TP4: radiance-local:ds41-exl3-117-<x3sk>-
# <exl3sk>-splitk (engine ds41-x3sk: the EXL3 ops get ep_ranks when RADIANCE_EXL3_SPLITK=1; radiance-exl3 exl3-splitk,
# interface v9: E1 down split-K 4 at one token / 2 at two, E2 gate_up split-K 2 kept at one token under EP). Only 1-2
# token launches change (1-user decode); c4/c8 and the MTP verify (6 tokens) keep today's split-K.
#   1. RADIANCE_EXL3_SPLITK=1, the fin-best line at MML 524288, gated (21 ms): stepprobe essay/code 512, mb c4/c8 (refs
#      x117-best524: 19.71 ms, essay 47.7 / code 50.6 tok/s, c4 95.0, c8 123.4) and the short copy test (fidelity: the
#      split-K reorders fp32 sums, so the bits of 1-2 token steps may move; greedy text may differ from today's, the copy
#      must stay exact)
#   2. the same image with RADIANCE_EXL3_SPLITK unset only if step 1's guard is not faster than 19.2 ms (to tell a
#      no-op from a regression in the same image; otherwise no off arm: 19.54-19.71 ms is today's number)
# Expected (PERF-REPORT model, not measured): gate_up ~74 -> ~46 us and down ~48 -> ~17 us a layer at one token ->
# about -2.0 to -2.4 ms a 1-user step (19.7 -> ~17.5 ms, essay/code ~+12 %); c4/c8 unchanged.
# Estimate ~25 min (1-2 starts, probe 1, mb c4/c8 7, copy 2).
# DRYRUN=1 bash 51-fix-splitk.sh : prints, writes nothing.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
export IMAGE=$X3SK
log "RESULT xfix 51-splitk: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))"
imgcheck $IMAGE
mkd $O $STATSDIR
best_vars
export RADIANCE_EXL3_SPLITK=1
gate xfix-sk 21; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 51: VIRHE not healthy ($(desc)) -- check the start log for a refused ep_ranks (v9 plugin?)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 51: VIRHE slow state after one restart -> skipped"; down; exit 1; }
run "docker logs $CN 2>&1 | grep -m1 'EP split-K rules' > $O/xfix-sk-mode.txt"
[ "$DRY" = 1 ] || log "RESULT xfix 51 mode: $(cat $O/xfix-sk-mode.txt | cut -c1-200)"
probe xfix-sk
for p in c4 c8; do stopfile && break; mbp xfix-sk $p; done
run "(cd $D && timeout 900 python3 ds41_tests.py copy --chars 6000 --max-tokens 2400 --runs 1 --tag=-xfix-sk --outdir $O > $O/xfix-sk-copy.txt 2>&1)"
[ "$DRY" = 1 ] || log "RESULT xfix 51 copy (6k chars, greedy): $(grep '^J ' $O/xfix-sk-copy.txt | head -1 | cut -c3-300)"
run "docker logs $CN > $O/xfix-sk-engine.log 2>&1"
log "RESULT xfix 51: ref $REF_BEST524"
unset RADIANCE_EXL3_SPLITK
ms=$(cut -d' ' -f2- $O/guard-xfix-sk.txt 2>/dev/null | awk '{print $NF}')
if [ "$DRY" = 1 ] || awk -v m="${ms:-99}" 'BEGIN{exit !(m > 19.2)}'; then
  gate xfix-sk-off 21 && probe xfix-sk-off
fi
down
log "RESULT xfix 51-splitk: done"
