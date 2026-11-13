#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-qkbench.sh (queue item, STAGED 12.11.2026, ds-main-port kbench builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-pskl20 and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, ONE card (CARD, default
# 0), every radiance container DOWN, no server, no evict; ~3-6 min (estimate), bounded at 13.5 min) -- SINGLE-CARD KERNEL BENCHES for
# audit levers 7 (IX: index_score -> exact faster forms / the fp8 block dot) and 4 (M1: one-pass mHC in prefill), tag
# "RESULT xkb2 ", out runs/xdsm/kbench2, body queue-ar/dsmain/kb/dsm-kbench.sh (+ kb/kbsum2.py, binaries kb/bin/kb2_ix + kb2_m1
# built from ds41-int7 0391a6e's libr4d sources copied to kb/src, real mHC tensors kb/data), image the int7 image (its HIP runtime
# and its libr4d.so = LIB, the production kernels through their real entry points). NOT QUEUED: the coordinator queues it.
# WHY (coordinator 12.11., dsm-225-prof-pf128-op rank 0, PSK off): index_score 125.3 ms a prefill step (8.6 %), ~17 TFLOP/s on
# live entries; mhc_mix 79.6 + mhc_post_collapse_rmsnorm 61.4 ms a step. The item answers: index_score's time a 256-row run at
# 136k / 64k / 16k for ratio 2 / ratio 1 / ratio 1 + keep, its TFLOP/s and GB/s and the limiter (ablations, VGPRs / occupancy, the
# dead-region writes); exact variants with a byte check vs LIB; the fp8 forms' speed and whether they hold today's bytes; mhc_mix
# and mhc_post_collapse_rmsnorm at 4096 / 8 / 1 rows and the fused exact-order prototype's bytes and time vs the pair.
# GUARD: a byte check that did not run, a variant whose own kernel did not run (run flag), an exact form that differs from LIB:
# VIRHE (kbsum2.py CHECKS).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-qkbench.sh : prints every command; checks the binaries, data, parser (md5s) and the image;
# starts nothing, writes nothing.
export XKB2_IMAGE=${XKB2_IMAGE:-radiance-local:ds41-int7-0391a6ec-5c13e555-l}
export XKB2_OUT=${XKB2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kbench2}
export CARD=${CARD:-0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/kb/dsm-kbench.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT xkb2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
bash $B
