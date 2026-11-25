#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzz-c4lev.sh (queue item, STAGED 12.11.2026 by the int11 builder for ds-main-port / the coordinator; sorts after
# -r92zz-ngreuse and -r92zzzz-fn-p2xfix in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN (or
# the one ATTACH claims); ~15 min estimate (~17-18 with an n-gram fill start), budget XC4_BUDGET_MIN 18 so all 12 ABBA blocks run (a block past it skipped and logged; MX24 only if it fits), NO
# evict) -- THE c4 LEVER SPLIT on int11: which lever carries the screen's c4 +7.7 % (N 172.6 vs O 162.6 / 157.7)? F4 at the 16-row
# drafting steps is the suspect. Tag "RESULT xc4lev ", out runs/xdsm/c4lev, body queue-ar/dsmain/c4lev/dsm-c4lev.sh (a COPY of
# 1u/dsm-1u.sh 58fad99d; + c4lev/c4rows.py; ngshare.sh sourced; ATTACH-patched by attach/patch-body.sh). NOT QUEUED.
# IMAGE radiance-local:ds41-int11-d3569104-5c13e555-l (7a2d85b1a13e).
# ONE START = the screen's O base (stack5's + LOG_STEPS=1, ENGDD=0, MHC_FUSE=0, HB ranks 1,2, GQ=3 / E2=1 declared; sched_row_cap 48 by
# fn4; KVR / SP off; no P0, no stamping) on the ngshare path (fill or reuse; the tables stay for the next item). Arms in process (fn4,
# read back): O, F4 (ds41_shexp_lane 1), E2 (ds41_e2 1), GQ (ds41_gq 3), MX (ds41_mix_wv 12), N (all on, mix_wv 12; hold 0); MX24
# (ds41_mix_wv 24) only if the budget allows. ONE discarded c4 warm-up (O), then two c4 blocks an arm on the fixed 1x4 schedule (gens 2),
# never adjacent: O F4 E2 GQ MX N N MX GQ E2 F4 O. Per block agg, per-stream p50, ms / tok a decode step, acceptance, decode share, the
# step log by rows (1-4 / 5-8 / 9-15 / 16 / > 16). Summary per arm vs O, per row bucket ms vs O. Guards: read back, 0 request errors,
# the i11 / cbh counters, the warm-up answers.
# XC4_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzz-c4lev.sh : prints every command; checks the
# image literals, every helper (+ selftests), the ngramshare files, both yamls, the sidecars, the disk; no server, no GPU, writes nothing.
export XC4_IMAGE=${XC4_IMAGE:-radiance-local:ds41-int11-d3569104-5c13e555-l} XC4_ITEM=${XC4_ITEM:-57-p4zyzzzzz-ds-r92zzzzz-c4lev}
export XC4_OUT=${XC4_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/c4lev} XC4_SW=${XC4_SW:-1} XC4_MXV=${XC4_MXV:-12}
export XC4_C4ORDER=${XC4_C4ORDER-O F4 E2 GQ MX N N MX GQ E2 F4 O} XC4_C4ORDERX=${XC4_C4ORDERX-O F4 E2 GQ MX MX24 N N MX24 MX GQ E2 F4 O}
export XC4_C4GENS=${XC4_C4GENS:-2} XC4_C4WARM=${XC4_C4WARM:-6} XC4_DISKMIN_MB=${XC4_DISKMIN_MB:-11000} XC4_BUDGET_MIN=${XC4_BUDGET_MIN:-18}
export XC4_SIDECAR=${XC4_SIDECAR:-dspark-dlw-q8_0.rad} XC4_SIDESHA=${XC4_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/c4lev/dsm-c4lev.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xc4lev: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
# ATTACH-SIG: SIGONLY=1 DRYRUN=1 bash <this item> prints its start signature (queue-ar/dsmain/attach/attach.sh)
export ATTACH=${ATTACH:-1} KEEP=${KEEP:-1}
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B; rc=$?
# 12.11. ds-main-port: FN items follow -> remove the n-gram tables (~103 GiB /dev/shm) after this item (no radiance container running then)
[ "${DRYRUN:-0}" = 1 ] || { sleep 5; $HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngram-shm-cleanup.sh; }
exit $rc
