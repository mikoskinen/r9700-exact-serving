#!/bin/bash
# mockrun-zzzfuse.sh NAME [--fault ...]: the zzzfuse fusion screen under queue-ar/tools/mockrad (ZFU_SW=0, short knobs; every stage runs
# once: P0 (fails under the mock: docker --device is not emulated; logged, the mock goes on), identity (warm-up, a1 B / Y vs ref12, gs16-Y,
# staggered ld-Y), c8 work-based B F F B (gens 1), singles R2 R1 M, c16 B / F (gens 1), 1u essay B / F (one discarded + 1 each), ncmatch /
# steps.py / fusesum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-zzzfuse.sh zfu-stop --fault differ=ds41_bi (every arm's answers differ from ref12's a1 -> the identity gate).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-zfu-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env ZFU_SW=0 --env ZFU_C8WARM=4 --env ZFU_BUDGET_MIN=120 --env ZFU_LDPF=2048 --env ZFU_C8GN=1 --env ZFU_U1N=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse.sh
