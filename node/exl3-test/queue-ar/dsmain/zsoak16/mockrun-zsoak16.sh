#!/bin/bash
# mockrun-zsoak16.sh NAME [mockrun args ...]: the DS int16 soak under queue-ar/tools/mockrad (ZS16_SW=0, short knobs; every stage runs once:
# the pin auto-choice, the start + the pin read-back (INFO under the mock), a1 + the cross-run compares (INFO), 3 cycles of loadid.py at c12
# + 2k prefills, the telemetry sampler on a FAKE sysfs tree under the output dir (never the real cards), a2 + the control, the end
# engagement probe, soaksum.py summary, gzip; the ngshare keeper / cleanup on sandbox paths).
#   clean:       mockrun-zsoak16.sh zs16-m1
#   one cycle differs + the int15 auto-choice (the mock replays the greedy set, so --no-replay lets the differ fault reach it):
#                mockrun-zsoak16.sh zs16-m2 --no-replay --env ZS16_MOCKDIFF=2 \
#                  --env ZS16_PICKLOG=/tmp/mockrad/zs16-m2/home/gpu/ds41/exl3-test/queue-ar/dsmain/zsoak16/mock-ref/xzc16-fail.log
#                -> cycle 2 VIRHE (recorded), cycle 3 runs, verdict NO; the warm-up vs stock line differs too (no replay).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-zs16-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env ZS16_SW=0 --env ZS16_BUDGET_MIN=120 --env ZS16_CYCLES=3 --env ZS16_CYCLES_MIN=3 \
  --env ZS16_CYCLE_S=30 --env ZS16_WARM=3 --env ZS16_WARM_MIN=2 --env ZS16_WARM_MAX=10 --env ZS16_PFTOK=2048 --env ZS16_TELS=2 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16.sh
