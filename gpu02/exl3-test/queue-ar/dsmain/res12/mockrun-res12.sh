#!/bin/bash
# mockrun-res12.sh NAME: the residency screen under queue-ar/tools/mockrad (XRS_SW=0, short knobs: every start, read set, c8 block,
# parser and the summary run; the ngshare keeper / cleanup on sandbox paths)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-res-m}
exec ./mockrun.sh --name $n --timeout 2700 --env XRS_SW=0 --env XRS_EGEN=128 --env XRS_C8WIN=8 --env XRS_C8WARM=4 --env XRS_BUDGET_MIN=120 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzz-res.sh
