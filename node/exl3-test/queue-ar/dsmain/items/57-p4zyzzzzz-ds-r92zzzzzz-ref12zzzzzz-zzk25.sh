#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25.sh (queue item, STAGED 13.11.2026, K25U builder for ds-main-port; sorts after
# -r92zzzzzz-ref12zzzzzz-zsa14 and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order; gpu02, ONE card BY CHIP UUID
# (ZZK_UUID, default GPU-dbba5c851cfef7b3), every radiance container DOWN, no server, no engine start, no compose; ~3-5 min
# expected, bounded at 10 min) -- lever K25U's card checks: the ds41-k25u exl3r4d (K = 2.5's decode band with unpacked trellis
# states; fn4 exl3_x3d 2 / 3 / 4 = X3D-F + gate_up G3 / G2D / UB96, down U4 at each) against the int14 image's shipped plugin
# (579d0bc), side by side in one process: the fn4 hooks; h' and y byte for byte at K 2 / 2.5, M 1-16 at every switch value
# 0..4 (408 rows a variant + the 96-expert geometry); K 2.5 gate_up / down GEMM time at EP4 rank geometry, M 1-16, arms off /
# G3 / G2D / UB96 (down off / U4) in balanced blocks x3. The last log lines state the byte verdict and the best gate_up value.
# Tag "RESULT zzk25 ", out runs/xdsm/zzk25, body queue-ar/dsmain/zzk25/dsm-zzk25.sh, binaries zzk25/bin (k25probe,
# exl3r4d-k25u.so; md5s in zzk25/bin/MD5SUMS). Measurement only. NOT QUEUED: ds-main-port / the coordinator queues it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25.sh : prints every command; checks the binaries, both images, the
# plugin hashes, the card and the disk; starts nothing GPU-side, writes nothing.
export ZZK_IMAGE=${ZZK_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l}
export ZZK_OUT=${ZZK_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzk25}
export ZZK_UUID=${ZZK_UUID:-GPU-dbba5c851cfef7b3}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzk25/dsm-zzk25.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT zzk25: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
