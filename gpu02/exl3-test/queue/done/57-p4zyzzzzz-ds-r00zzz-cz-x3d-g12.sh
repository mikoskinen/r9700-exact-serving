#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12.sh (queue item, STAGED 12.11.2026, X3D-F builder for ds-main-port; gpu02, ONE card (logical
# HIP index CARD, default 0; no PCI address / card order used), every radiance container DOWN, no server, no engine start,
# no compose; ~2-4 min expected, bounded at 8 min) -- X3D-F gates 1 + 2: the ds41-x3d exl3r4d (K = 2's decode from one
# word pair, fn4 switch exl3_x3d) against the image's shipped 5c13e55, side by side in one process: h' and y byte for
# byte at K 2 / 2.5, M 1-16 (OLD vs NEW off vs NEW on), the fn4 hooks, and K 2 gate_up / down GEMM time at EP4 rank
# geometry, M 1-16, ABBA x3 (off vs on). Tag "RESULT x3dg12 ", out runs/xdsm/x3dg12, body queue-ar/dsmain/x3d/dsm-x3dprobe.sh,
# binaries x3d/bin (x3dprobe, exl3r4d-x3d.so; md5s in x3d/bin/MD5SUMS). Measurement only. NOT QUEUED: ds-main-port queues it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12.sh : prints every command; checks the binaries, the image and the plugin
# hash; starts nothing GPU-side, writes nothing.
export X3D_IMAGE=${X3D_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
export X3D_OUT=${X3D_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/x3dg12}
export CARD=${CARD:-0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/x3d/dsm-x3dprobe.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT x3dg12: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. wait up to 60 s for a previous item s container (ds-main-port)
bash $B
