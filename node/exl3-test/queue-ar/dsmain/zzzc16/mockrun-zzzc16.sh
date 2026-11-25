#!/bin/bash
# mockrun-zzzc16.sh NAME [--fault ...]: the int16 confirmation under queue-ar/tools/mockrad (ZC16_SW=0, short knobs; every stage runs once:
# the start + the pin read-back (INFO under the mock), identity under F (warm-up, a1-F vs ref12 / zzzfuse a1-B, gs16-F, staggered ld-F),
# 1u ABBA B F F B (one discarded + 1 each), c4 gens (gens 1) one discarded + B F F B, cold (a small depth) vs ref14's answer (the mock's text
# differs: logged INFO under ZC16_SW=0), zc16sum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-zzzc16.sh zc16-stop --env ZC16_ZFA1=/tmp/mockrad/zc16-stop/home/gpu/ds41/exl3-test/queue-ar/dsmain/zzzc16/mock-ref/xzfuse-a1-B-altered.jsonl
# (a1-F vs an altered copy of zzzfuse's a1-B -> the identity gate STOPS; the mock replays the greedy-set answers, so a --fault differ= never reaches them).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-zc16-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env ZC16_SW=0 --env ZC16_C8WARM=4 --env ZC16_C4WARM=4 --env ZC16_BUDGET_MIN=120 --env ZC16_LDPF=2048 --env ZC16_C4GN=1 --env ZC16_U1N=1 \
  --env ZC16_CDEPTH=16000 --env ZC16_COLDEST=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16.sh
