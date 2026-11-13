#!/bin/bash
# 57-p4zyzzzzz-ds-r00zz-psk2.sh (queue item, 11.11.2026, ds-main-port PSK design-2 builder for the coordinator; after r00z (done) /
# r00y-pskloc / r00zy-stack4, before r04-bi-gyz-chunk (C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~28 min, NO evict) -- LEVER PSK DESIGN 2 (anchors), tag "RESULT xpsk2 ", out runs/xdsm/psk2, body
# queue-ar/dsmain/psk2/dsm-psk2.sh (= r00z's dsm-psk.sh + design 2's arms / hit sets / counters; + pskld.py, r00z's copy byte for
# byte). Image XP2_IMAGE = radiance-local:ds41-psk2-d6fcc241-5c13e555-l (747d9ae8dc90; ds41-psk2 d6fcc24 = ds41-psk 4d4da8e + design 2 + ds41-psk-fix cf7bbc7 (the indexer deal): fn4 ds41_psk 2 / 3
# + ds41_psk_k; hostcheck: unset and ds41_psk 0 / 1 = design 1's image, unset = int5).
# WHY (coordinator + ds-main-port 11.11.): design 1 (r00z) cut 136k cold 50.46 -> 43.48 s; its cut layers stay staging-bound
# (280 MB a layer a chunk). Design 2 runs layers 21-39 only at anchor chunk ends (tail, every K-th grid point, checkpoint_at) and
# skips them -- compute, collectives, expert staging -- elsewhere; snapshots only at anchors, a hit between them falls back to the
# anchor below. This item is the exactness gate and the K choice: A / D1 / K4 / K8 in one NST 5 start.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zz-psk2.sh : prints every command; checks the image literals, every helper, the sidecars,
# the yaml; no server, no GPU, writes nothing.
export XP2_IMAGE=${XP2_IMAGE:-radiance-local:ds41-psk2-d6fcc241-5c13e555-l} XP2_ITEM=${XP2_ITEM:-57-p4zyzzzzz-ds-r00zz-psk2}
export XP2_OUT=${XP2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/psk2}
export XP2_ORDER=${XP2_ORDER:-AD4884DA} XP2_LORDER=${XP2_LORDER-D88D} XP2_HITS=${XP2_HITS:-90000 58000 20000}
export XP2_DS=${XP2_DS:-1}
export XP2_SIDECAR=${XP2_SIDECAR:-dspark-dlw-q8_0.rad} XP2_SIDESHA=${XP2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/psk2/dsm-psk2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpsk2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
