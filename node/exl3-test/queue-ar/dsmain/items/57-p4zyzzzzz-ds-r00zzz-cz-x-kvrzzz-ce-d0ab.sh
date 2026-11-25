#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-ce-d0ab.sh (queue item, STAGED 13.11.2026 by the ds-main-port base builder for the coordinator;
# sorts after -cz-x-kvrzzz-cd-cap48 and before -cz-x-kvrzzzz-i9b in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~15 min estimate, budget XS5_BUDGET_MIN 17 (B extra / identity skipped and logged past it), NO evict)
# -- THE c4 DEPTH-0 CONFIRMATION, tag "RESULT xd0 ", out runs/xdsm/d0ab, body queue-ar/dsmain/d0ab/dsm-d0ab.sh (a COPY of
# base/dsm-baseline.sh 6be30f5e; + d0blocks.py = s5blocks.py + --settle, s5ctr.py copy, the coordinator's c4split.py). NOT QUEUED.
# ONE START exactly the baseline's (int9 0cb6fd31, stack5's NST 5 start, S5T pins, KVR / SP off). Arms at runtime (fn4, read back):
# A = draft_depth0 1 + draft_lend 1 (baseline), B = draft_depth0 0 (drafting always on), C = draft_depth0 1 + draft_lend 0.
# Order A B C C B A, mb c4 only at T 0, dur 30 warm 10, 5 s settle after each read-back; c4 step split (c4split.py); ONE B extra block
# mb c1 + mb c8 (c8 drafting check, occ rule); identity: greedy set at conc 4 under A / B / C, B vs A and C vs A strict (VIRHE on any
# difference); summary B vs A %, C vs A %, base1113 S5T and the pre-move stack5 S5T c4 150.6.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-ce-d0ab.sh : prints every command; checks the image literals, every helper (+
# selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XS5_IMAGE=${XS5_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l} XS5_ITEM=${XS5_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-ce-d0ab}
export XS5_OUT=${XS5_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/d0ab}
export XS5_T=1 XS5_X=0 XS5_PSKK=8 XS5_BIC=3 XS5_NOA=2 XS5_OCC=6 XS5_OCCK=32 XS5_DSA=85
export XS5_ORDER0=${XS5_ORDER0:-ABCCBA} XS5_ORDER1="" XS5_PORDER=A XS5_LORDER=""
export XS5_DUR=${XS5_DUR:-30} XS5_WARM=${XS5_WARM:-10} XD0_SETTLE=${XD0_SETTLE:-5} XS5_EDIT=1 XS5_GDEP=${XS5_GDEP:-16000}
export XS5_DEPTHS="135000 64000" XS5_LDEPTH=64000 XS5_CDEPTH=48000 XS5_CNTOK=256
export XS5_DISKMIN_MB=${XS5_DISKMIN_MB:-11000} XS5_BUDGET_MIN=${XS5_BUDGET_MIN:-17}
export XS5_SIDECAR=dspark-dlw-q8_0.rad XS5_SIDESHA=f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/d0ab/dsm-d0ab.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xd0: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
