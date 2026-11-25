#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12b.sh (queue item, STAGED 12.11.2026, X3D-F builder for ds-main-port; gpu02, ONE card (logical
# HIP index CARD, default 0; no PCI address / card order used), every radiance container DOWN, no server, no engine start,
# no compose; ~2-4 min expected, bounded at 8 min) -- X3D-F gates 1 + 2 again after g12's gate_up regression (+29 %): the
# ds41-x3d exl3r4d with the X3D decode of a step's tiles phase-ordered (decode_frags_x3d: the MATS = 2 K loop's per-tile
# alignbit -> pk_mul -> mad -> sad -> fma chains, each stalling on its own latency, now run phase by phase over both tiles),
# against the image's shipped 5c13e55, side by side in one process: h' and y byte for byte at K 2 / 2.5, M 1-16 (OLD vs NEW
# off vs NEW on), the fn4 hooks, and K 2 gate_up / down GEMM time at EP4 rank geometry, M 1-16, ABBA x3 (off vs on). The
# last log line states the verdict PASS / FAIL from the numbers. Tag "RESULT x3dg12b ", out runs/xdsm/x3dg12b, body
# queue-ar/dsmain/x3d/dsm-x3dprobe-b.sh, binaries x3d/bin-b (x3dprobe, exl3r4d-x3d.so; md5s in x3d/bin-b/MD5SUMS).
# Measurement only. NOT QUEUED: ds-main-port queues it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12b.sh : prints every command; checks the binaries, the image and the plugin
# hash; starts nothing GPU-side, writes nothing.
export X3D_IMAGE=${X3D_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
export X3D_OUT=${X3D_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/x3dg12b}
export CARD=${CARD:-0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/x3d/dsm-x3dprobe-b.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT x3dg12b: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. wait up to 60 s for a previous item s container (ds-main-port)
export X3D_GATE2_PCT=${X3D_GATE2_PCT:-10}   # 12.11. ds-main-port: the coordinator s gate (K loop >= ~10 %), strict
bash $B
