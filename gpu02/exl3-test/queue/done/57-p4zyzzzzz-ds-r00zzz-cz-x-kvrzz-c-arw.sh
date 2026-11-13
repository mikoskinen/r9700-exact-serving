#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-c-arw.sh (queue item, STAGED 12.11.2026, ds-main-port arw builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16 and before 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc in C and en_US order; gpu02, ALL FOUR cards,
# every radiance container DOWN, no server, no evict; ~2-3 min (estimate), bounded at 12 min) -- THE CODED ALL-REDUCE WIRE AT DS
# SHAPES: is libr4d's coded wire for the exact four-rank two-shot (~12.2 bits a bf16 value; selected only at overlap 1, r4d_ar_entry.hip
# 1061-1068, which DS never declares) byte-identical to the plain wire DS runs, and faster? Tag "RESULT xarw ", out runs/xdsm/arw,
# body queue-ar/dsmain/arw/dsm-arw.sh (+ arw/arwsum.py, binary arw/bin/arw built from arw/bench/arw.hip by arw/build.sh in
# radiance-ds41-build:main-e736c019), image the int9 image (its libr4d.so, md5 pinned).
# P0 the image's r4d_selftest --case ar_twoshot_4rank_exact_lane (libr4d's plain + coded vs the fp32 fold at its own sizes, 4 cards);
# P1 arw: P (ar_twoshot_4rank_exact, DS's row) vs C (the lane row's coded launch, 4 x 1024) and C8 / C16 / C24 at T x 5120 bf16,
# T = 1100 / 1500 / 2048 / 4096 / 8 / 16 / 48, every rank, in place + out of place, patterns bin / act / actz, the wire seen on every
# launch; 100 interleaved timed launches a variant (median, p10, p90) + back to back; P2 arwsum.py: one RESULT line a shape,
# SUMMARY (coded vs plain % a shape, identity verdict), ESTIMATE (the 128k step).
# GUARD (VIRHE, the reading stops): any byte plain vs coded on any rank, a launch on the wrong wire, a missing comparison / timing,
# the selftest case not clean.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-c-arw.sh : prints every command; checks the binary, parser, image and library
# (md5s), r4d_selftest --help and arw --selftest in the image with no GPU device; starts nothing on a GPU, writes nothing.
export XARW_IMAGE=${XARW_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
export XARW_OUT=${XARW_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/arw}
export XARW_DISKMIN_MB=${XARW_DISKMIN_MB:-11000}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/arw/dsm-arw.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT xarw: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
bash $B
