#!/bin/bash
# mockrun-zsa14.sh NAME [--fault ...]: the zsa14 lever-SA screen under queue-ar/tools/mockrad (SA14_SW=0, short knobs; every stage runs once:
# identity (warm-up, a1 B / S1 / S2 vs ref12, gs16-S2, staggered ld-S2 + SA engagement), c8 work-based B S1 S2 S2 S1 B (gens 1), c16 B + the
# better SA arm (gens 1), ncmatch / steps.py / sa14sum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-zsa14.sh sa14-stop --fault differ=stage_pick:2 (S2's answers differ -> the identity gate).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-sa14-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env SA14_SW=0 --env SA14_C8WARM=4 --env SA14_BUDGET_MIN=120 --env SA14_LDPF=2048 --env SA14_C8GN=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14.sh
