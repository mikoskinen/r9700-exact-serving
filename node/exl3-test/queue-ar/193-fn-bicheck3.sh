#!/bin/bash
# 193-fn-bicheck3.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~9 min) -- fn4 BI:
# THE TWO FIXES FROM 180's TRACE, checked with the same tracer. Image radiance-local:c16-x3ab (fn4-x3 b67cb1c0 = c16-x3bs +
# 5777709a bi_seg 2: the lm_head at each output row's alone height class + b67cb1c0 att_bi: a step's prefill sequences'
# dense attention each in a launch shaped as its alone step's; plugin set plug-dual2 as c16-x3bs). Two bodies, unchanged
# but for env (md5 below), one after the other from queue-ar:
#   187-fn-smoke.sh  73c5cdd9  the dry run through warm-up + greedy c1 under today's and the new set + c4 (~2.5 min)
#   180-fn-bitrace.sh 3790f227  ONE traced start, phase sg2 = exl3_bi 0 + exl3_dual 1 + hc_bi 2 + bi_seg 2 + att_bi 1, exact wire:
#                     alone / c4 / c4 + a fresh 16k prefill / alone again, text compares + first divergence per probe (~5 min)
# 180 starts only when 187 wrote this image's id (its own gate). Reading: c4 and pf vs alone want 20/20 IDENTICAL; where not,
# the tracer names the next op (decode rows' split count by sequence count is the expected one at long contexts / c16).
# DRYRUN=1 bash 193-fn-bicheck3.sh : each body's own dry run.
set -u
QA=$HOME/gpu/ds41/exl3-test/queue-ar; L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
DRY=${DRYRUN:-0}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
IMG=${X193_IMAGE:-radiance-local:c16-x3ab}
chk() { [ -e $QA/$1 ] && [ "$(md5sum < $QA/$1 | cut -c1-8)" = $2 ] || { log "RESULT fn4 193: VIRHE $1 missing or not md5 $2"; exit 1; }; }
log "RESULT fn4 193-fn-bicheck3: start IMAGE=$IMG -- 187 smoke (second set bi_seg 2 + att_bi 1) then 180 bitrace arm bi2"
chk 187-fn-smoke.sh 73c5cdd9; chk 180-fn-bitrace.sh 3790f227
( cd $QA && DRYRUN=$DRY X187_IMAGE=$IMG X187_SW2='{"hc_bi": 2, "exl3_dual": 1, "bi_seg": 2, "att_bi": 1}' bash $QA/187-fn-smoke.sh ); rc=$?
[ $rc = 0 ] || log "RESULT fn4 193: 187 exit $rc"
( cd $QA && DRYRUN=$DRY X180_IMAGE=$IMG X180_ARMS=bi2 bash $QA/180-fn-bitrace.sh ); rc=$?
[ $rc = 0 ] || log "RESULT fn4 193: 180 exit $rc"
log "RESULT fn4 193-fn-bicheck3: done"
