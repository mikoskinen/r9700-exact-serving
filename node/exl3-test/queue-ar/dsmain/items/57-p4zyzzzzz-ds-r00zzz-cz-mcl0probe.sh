#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-mcl0probe.sh (queue item, STAGED 12.11.2026, MCL builder for the coordinator; gpu02, ONE card
# (logical HIP index CARD, default 0; no PCI address / card order used), every radiance container DOWN, no server, no engine
# start, no evict; ~1 min expected, bounded at 5 min) -- MCL gate G0 (exact CPU lane for host-held decode experts): the card
# measurements an exact CPU copy of the EXL3 decode band needs -- the v_wmma_f32_16x16x16_f16 internal sum (batteries), the
# v_exp_f32 table on |x| < 1, the mix / division / silu rounding, and the SHIPPED exl3r4d kernels' h' and y on synthetic
# DS-shaped experts at K 2 / 2.5, M 1..16 (dlopen of the int9 image's own plugin). Tag "RESULT mcl0 ", out runs/xdsm/mcl0
# (~180 MB), body queue-ar/dsmain/mcl/dsm-mclprobe.sh, binary mcl/bin/mclprobe (md5 in mcl/bin/MD5SUMS). Measurement only;
# the G0 verdict is offline (mclcheck analyze on fedora-homeamd, after the out dir is copied back). NOT QUEUED: the coordinator
# queues it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-mcl0probe.sh : prints every command; checks the binary, the image and the plugin hash;
# starts nothing GPU-side, writes nothing.
export MCL0_IMAGE=${MCL0_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
export MCL0_OUT=${MCL0_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/mcl0}
export CARD=${CARD:-0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/mcl/dsm-mclprobe.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT mcl0: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
bash $B
