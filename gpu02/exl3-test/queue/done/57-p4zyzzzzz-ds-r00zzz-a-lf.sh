#!/bin/bash
# 57-p4zyzzzzz-ds-r06z-lf.sh (queue item, 11.11.2026, ds-main-port LF builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r06-slotsum, before 57-p4zyzzzzz-ds-r07-bi-gz-dual (C and en_US order); gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~45 min, NO evict) -- LEVER LF (lent-slot fill) + the slot-class counters, tag "RESULT xlf ",
# out runs/xdsm/lf, body queue-ar/dsmain/lf/dsm-lf.sh (+ lfblocks.py = occ2blocks.py + per-rank slot classes / lenders, copies of
# mbt.py / loadid.py / sdprobe.py / editprobe.py). Image radiance-local:ds41-lf-d583904e-5c13e555-l (e921126b6273, XLF_IMAGE; ds41-lf d583904 = ds41-int5 283c7a1 + lever LF + /stats slot classes;
# hostcheck: unset = int5).
# WHY (coordinator 11.11., LEVERS "LF lent-slot proactive fill"): z0occ c8 NST 5 depth 0 173.2 vs NST 0 182.0 (-5 %), ~100 fewer
# trunk units a rank; runtime-lent units (DL, Z0-W) moved neither hit nor tok/s. Lent slots are filled only by Mover::fill, hottest
# pooled first by a ~14-step heat with no admission bar (= the experts just missed); LF ranks that fill by a long-memory decode heat.
# ONE NST 5 START (z0occ2's: stack3's S-arm start + the Z0 start words), draft_occ 3 / k 32 pinned in every arm (c4 / c8 hold depth
# 0, the slab lent); in process: arms A = place_lf 0, B = place_lf 1; T 0 blocks ABBAABBA, T 1 blocks ABBA (mb c4 / c8); per block
# and phase: tok/s, p50, depth-0 share, per rank hit / misses / trunk units, slot classes' hits and misses, lenders' held / covered,
# fills (lf / hot), drops by cause; identity at T 0 under B (alone vs bz d0, loaded vs alone). Summary: B vs A (pairs), lent-slot
# hits a unit A vs B, c8 vs z0occ NST 0 182.0 (cross-start).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r06z-lf.sh : prints every command; checks the image literals, every helper, the sidecars, the yaml;
# no server, no GPU, writes nothing.
export XLF_IMAGE=${XLF_IMAGE:-radiance-local:ds41-lf-d583904e-5c13e555-l} XLF_ITEM=${XLF_ITEM:-57-p4zyzzzzz-ds-r06z-lf}
export XLF_OUT=${XLF_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/lf}
export XLF_ORDER=${XLF_ORDER:-ABBAABBA} XLF_T1ORDER=${XLF_T1ORDER-ABBA} XLF_DUR=${XLF_DUR:-45} XLF_WARM=${XLF_WARM:-10}
export XLF_DS=${XLF_DS:-1} XLF_OCC=${XLF_OCC:-3} XLF_OCCK=${XLF_OCCK:-32} XLF_IDARM=${XLF_IDARM:-B}
export XLF_SIDECAR=${XLF_SIDECAR:-dspark-dlw-q8_0.rad} XLF_SIDESHA=${XLF_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/lf/dsm-lf.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xlf: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
