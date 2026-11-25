#!/bin/bash
# 57-p4zyzzzzzzzzzz-bi-a-opcheck.sh (queue item, 11.11.2026, ds41-bi agent for ds-main-port, lever BI; sorts after
# 57-p4zyzzzzzzzzz-da-f-fidelity (en_US and C order); gpu02, ONE card (HIP_VISIBLE_DEVICES=0), no server, every radiance
# container DOWN; ~15-25 min) -- LEVER BI PER-OP ON-CARD BYTE COMPARE (tag "RESULT xbio ", out runs/xdsm/biop, body
# queue-ar/dsmain/dsm-biop.sh). USER RULING 11.11.: an answer that changes under load vs alone is a BUG.
# The image's r4d_selftest --bicheck: every row-wise op of the DeepSeek trunk at its TP4 shapes, a row ALONE (M = 1) vs the same row
# inside an M-row launch (M 1..64, 65..4160), byte for byte, the kernel row the engine's band takes at M; arms stock (form switches
# unset) and stack (gg_gemv 2, gg_pf 1 + table, f32_pf 25). One line a case: INVARIANT or the first flip M. Measurement only.
# DRYRUN=1 bash 57-p4zyzzzzzzzzzz-bi-a-opcheck.sh : prints every command; checks the image (r4d_selftest's --bicheck literals)
# and every helper; no GPU, writes nothing. Knobs (edit the export line before moving into queue/; the runner passes no
# environment): XBIO_IMAGE, XBIO_ARMS, XBIO_CLASSES, XBIO_MS.
export XBIO_IMAGE=${XBIO_IMAGE:-radiance-local:ds41-bi-060dff08-5c13e555-l} XBIO_ITEM=${XBIO_ITEM:-57-p4zyzzzzzzzzzz-bi-a-opcheck}
export XBIO_ARMS=${XBIO_ARMS:-stock stack} XBIO_CLASSES=${XBIO_CLASSES:-all}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-biop.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbio: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
