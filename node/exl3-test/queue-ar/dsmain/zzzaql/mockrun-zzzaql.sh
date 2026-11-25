#!/bin/bash
# mockrun-zzzaql.sh NAME [--fault ...]: the AQL-stamped c8 read under queue-ar/tools/mockrad (ZQA_SW=0, short knobs; every stage runs
# once: start + pin + read-back, the warm-up identity gate, 2 c8 blocks (8 s windows), the split / ncmatch / ranks / monowin parsers,
# aqlcp.py on SYNTHETIC stamps (the mock server writes none), the summary; the ngshare keeper / cleanup on sandbox paths).
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-zqa-m}; shift
exec ./mockrun.sh --name $n --timeout 2400 --env ZQA_SW=0 --env ZQA_C8WIN=8 --env ZQA_C8WARM=4 --env ZQA_BUDGET_MIN=120 --env ZQA_MIN_STEPS=50 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzaql.sh
