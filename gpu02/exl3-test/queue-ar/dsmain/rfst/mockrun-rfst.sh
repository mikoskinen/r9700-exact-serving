#!/bin/bash
# mockrun-rfst.sh NAME [pass|fail|shim] [mockrun options]: the rfst item under queue-ar/tools/mockrad (RFST_MOCK=1). pass / fail feed
# rfst/mock/pass.txt / fail.txt (151 ok + 450 of 450; the same with one FAIL at base 96, 449 of 450) as the selftest's output; shim lets
# the GPU run go to the mock's docker shim (refused, exit 125 -> the FAIL path with no ok lines).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-rfst-m}; v=${2:-pass}; shift; [ $# -gt 0 ] && shift
M=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/rfst/mock
case $v in pass|fail) X=(--env RFST_MOCK_OUT=$M/$v.txt) ;; shim) X=() ;; *) echo "pass|fail|shim"; exit 2 ;; esac
exec ./mockrun.sh --name $n --timeout 300 --env RFST_MOCK=1 "${X[@]}" "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16z-rfst.sh
