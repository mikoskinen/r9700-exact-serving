#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16z-rfst.sh (queue item, STAGED 13.11.2026, ds41-fuse builder for ds-main-port;
# sorts right after -r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16 in C and en_US order; gpu02, ONE card BY CHIP UUID (RFST_UUID, default
# GPU-dbba5c851cfef7b3), every radiance container DOWN, no server, no engine start, no compose; ~10-30 s expected, the selftest
# bounded at 300 s) -- lever FUSE-R's form 1 at 49..170 rows (ds41_rfuse 2's prefill-segment fallback) byte for byte vs the stock
# chain: r4d_selftest --case rfuse of image radiance-local:ds41-fuse-e231d919-38b88ec6-st2 (be78eec84ba0 = 4419a17d60a7 + the
# selftest of ds41-fuse dcee256; engine / libr4d / deepseek41 byte-identical to 4419a17d60a7).
# The verdict line "RESULT rfst: RFST M 49..170: PASS|FAIL (rc, ok n/151, FAIL n, 450 of 450, form-1-only rows n/42)"; a FAIL also
# says "int16 soak result VOID: fall back to int15" and logs every FAIL line; a guard refusal says NOT RUN (int16 UNCONFIRMED).
# Out runs/xdsm/rfst, body queue-ar/dsmain/rfst/dsm-rfst.sh. Measurement only. NOT QUEUED: ds-main-port / the coordinator queues it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16z-rfst.sh : prints every command; checks the image, its selftest,
# the card and the disk; starts nothing GPU-side, writes nothing.
export RFST_IMAGE=${RFST_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-st2}
export RFST_OUT=${RFST_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/rfst}
export RFST_UUID=${RFST_UUID:-GPU-dbba5c851cfef7b3}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/rfst/dsm-rfst.sh
if [ ! -e $B ]; then
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: $B MISSING"; exit 1; fi
  echo "$(date '+%F %T') RESULT rfst: RFST M 49..170: NOT RUN (VIRHE $B missing) -- int16 soak result UNCONFIRMED until this item runs" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
