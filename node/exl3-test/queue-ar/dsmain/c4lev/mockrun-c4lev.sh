#!/bin/bash
# mockrun-c4lev.sh NAME: the c4 lever-split item under queue-ar/tools/mockrad (XC4_SW=0, short knobs; the ngshare keeper on sandbox paths)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-c4l-m}
exec ./mockrun.sh --name $n --timeout 1500 --env XC4_SW=0 --env XC4_C4GENS=1 --env XC4_C4WARM=3 --env XC4_BUDGET_MIN=120 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  items/57-p4zyzzzzz-ds-r92zzzzz-c4lev.sh
