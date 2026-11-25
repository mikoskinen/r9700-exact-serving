#!/bin/bash
# mockrun-yswl.sh NAME: the SWL screen under queue-ar/tools/mockrad (XSW_SW=0, short knobs: the start, the staggered B / L1 checks, every
# arm's c8 block + reads, the ABBA halves, the c8 splits, the summary run; the ngshare keeper / cleanup on sandbox paths)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-yswl-m}
exec ./mockrun.sh --name $n --timeout 2700 --env XSW_SW=0 --env XSW_EGEN=128 --env XSW_C8WIN=8 --env XSW_C8WARM=4 --env XSW_BUDGET_MIN=120 \
  --env XSW_LDPF=2048 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl.sh
