#!/bin/bash
# 57-p4zyzzzzz-ds-r02z-z0occ2.sh (queue item, 11.11.2026, ds-main-port stack3 builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r02-z0occ, before 57-p4zyzzzzz-ds-r03-sdtemp (C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~45 min, NO evict) -- LEVER Z0-O FOLLOW-UP, tag "RESULT xz0o2 ", out runs/xdsm/z0occ2, body
# queue-ar/dsmain/occ2/dsm-z0occ2.sh (+ occ2blocks.py = s3blocks.py + the 2 s depth-0 share, mbt.py, copies of loadid.py /
# sdprobe.py / editprobe.py). Image radiance-local:ds41-int5-283c7a16-5c13e555-l (18f79fe8b7ac, XO2_IMAGE; ds41-int5 283c7a1 = ds41-int4 cea53a4 + ds41-z0
# 8f171a4; hostcheck: every switch unset = int4 = int3).
# WHY (coordinator 11.11.): z0occ's draft_occ 3 holds depth 0 from 3 decoding sequences up (drafting off at c4), but stack3 shows
# drafting wins at c4 (T 0 P 149.5 vs NST 0 138.5; T 1 S 142.1) and loses at c8 (170.6 / 157.6 vs 184.6): draft_occ 6 / 8 hold
# depth 0 only near c8.
# ONE NST 5 START (stack3's S-arm start: stack2 NST 5 env, ds41_bi 2 pinned, draft_sample XO2_DS pinned, --draft-coupling on, DLW
# q8_0 sidecar; + z0occ's start words: ring 129, RADIANCE_DRAFT_LEND_WEIGHTS=0, rungs 8,48; Z0 switches 0, draft_occ_k XO2_OCCK),
# in process: arms A = draft_occ 0, 6 = draft_occ 6, 8 = draft_occ 8; T 0 blocks A 6 8 8 6 A (mb c4 / c8 at temperature 0), T 1
# blocks A 6 8 8 6 A (mb c4 / c8 at the server default); per block and phase: agg tok/s, p50, tok/step, acceptance, ms/step, the
# depth-0 share (phase and 2 s intervals: mean / sd / min / max), occ_cycles / occ_moves, VRAM / lent units sampled; identity at
# T 0 under 6 (alone vs bz d0, loaded vs alone). Summary: 6 / 8 vs A per phase and temperature (ABBA pairs), the depth-0 share sd
# per arm; refs stack3 NST 0 (T 0: c4 138.5 / c8 184.6) and hxzzy-stack2 NST 0 (T 1), cross-start.
# STAGED in queue-ar/dsmain/items/ -- copy into queue/ only on the coordinator's word (after r02 z0occ lands; XO2_OCCK / XO2_DS
# may be set from z0occ / r03 sdtemp then).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r02z-z0occ2.sh : prints every command; checks the image literals, every helper, the sidecars, the
# yaml; no server, no GPU, writes nothing.
export XO2_IMAGE=${XO2_IMAGE:-radiance-local:ds41-int5-283c7a16-5c13e555-l} XO2_ITEM=${XO2_ITEM:-57-p4zyzzzzz-ds-r02z-z0occ2}
export XO2_OUT=${XO2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/z0occ2}
export XO2_ORDER=${XO2_ORDER:-A6886A} XO2_T1ORDER=${XO2_T1ORDER-A6886A} XO2_DUR=${XO2_DUR:-45} XO2_WARM=${XO2_WARM:-10}
export XO2_DS=${XO2_DS:-1} XO2_OCCK=${XO2_OCCK:-32} XO2_IDARM=${XO2_IDARM:-6}
export XO2_SIDECAR=${XO2_SIDECAR:-dspark-dlw-q8_0.rad} XO2_SIDESHA=${XO2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/occ2/dsm-z0occ2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0o2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
