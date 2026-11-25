#!/bin/bash
# 57-p4zyzzzzz-ds-e-hold.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288; ~25 min) --
# held item 57-p5-onsh (225:onsh = arena levels + stager ons + RADIANCE_ARENA_HOLD_MS 5000 on the old tuned2-shp image,
# window cc_sim) converted to the main image (coordinator 10.11.): the hold lever on today's stack = the t3 line on
# radiance-local:ds41-main-e736c019-5c13e555 + RADIANCE_ARENA_HOLD_MS=5000 (arm 225h), NO cc_sim (user 10.11.), no
# BetterBench. Same phases as 57-p4zyzzzzz-ds-a-main's arm (gate, modes, warm-up answers, slow-state probe, probe, mb c4 / c8,
# THEN pf / turns, i1 copy, deep); compared with that item's dsm-225 (the main t3 arm, hold unset = 2000 ms) and with 105's
# tuned3l; identity (copy, deep) vs dsm-225 and vs tuned3l (the hold only moves arena loans: exact by construction).
# Runs after 57-p4zyzzzzz-ds-a-main (its dsm-225 files are the reference). DRYRUN=1 bash ... : prints, writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
export XDSM_ARMS=${XDSM_ARMS:-225h} XDSM_BB=${XDSM_BB:-none} XDSM_GAP=${XDSM_GAP:-} XDSM_PROF=${XDSM_PROF:-} XDSM_SHORT=${XDSM_SHORT:-} XDSM_HOLD_MS=${XDSM_HOLD_MS:-5000}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-serve.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || [ -s $HOME/gpu/ds41/exl3-test/runs/xdsm/guard-dsm-225.txt ] || { echo "$(date '+%F %T') RESULT xdsm hold: VIRHE no dsm-225 reference arm (57-p4zyzzzzz-ds-a-main did not run its 225 arm) -- not starting" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
