#!/bin/bash
# mockrun-kvres.sh NAME [mockrun args ...]: the KV-reservation trade-off item under queue-ar/tools/mockrad (KVR_SW=0, short knobs; every stage runs
# once a start: two starts (K35 / K60), the start + the pin read-back (INFO under the mock), the placement read-back (the mock serves a real 19456
# start log: KVPLACE INFO), identity (warm-up, a1-F vs ref12 / zzzc16 a1-F), one c8 gens block (gens 1), one c16 gens block (gens 1), 1u essay
# (one discarded + 1), the engine log capture, ncmatch.py / steps.py, kvsum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
#   clean:   mockrun-kvres.sh kv-m1
#   STOP:    mockrun-kvres.sh kv-stop --env KVR_ZCA1=/tmp/mockrad/kv-stop/home/gpu/ds41/exl3-test/queue-ar/dsmain/kvres/mock-ref/xzc16-a1-F-altered.jsonl
#            (a1-F vs an altered copy of zzzc16's a1-F -> each start's identity gate STOPS its speed reads; the next start still runs)
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-kv-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env KVR_SW=0 --env KVR_C8WARM=4 --env KVR_BUDGET_MIN=120 --env KVR_C8GN=1 --env KVR_C16GN=1 --env KVR_C8B=1 \
  --env KVR_U1N=1 --env KVR_STARTEST=5 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres.sh
