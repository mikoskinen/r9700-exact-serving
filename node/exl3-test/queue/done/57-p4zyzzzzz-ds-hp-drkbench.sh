#!/bin/bash
# 57-p4zyzzzzz-ds-p0-drkbench.sh (queue item, 10.11.2026, ds41-drk agent; gpu02, ONE card (CARD, default 0), every radiance
# container DOWN, no server; ~15-25 min) -- the DSpark drafter's kernels on one card, from the ds41-drk image's own r4d_selftest
# and libr4d: (1) --mxcheck: drafter_mx's multi-row forms vs the stock MXFP4 grouped GEMM (gemv band to 16 tokens, tiled kernel
# past it and on the tiled row below), gate_up 4608x5120 / down 5120x2304, 32 experts a rank, top 3, tokens 1..48, drafter-like
# routings with and without the dropped tail, stacked experts and a pointer table, every form, BYTE FOR BYTE; (2) --bfcheck:
# drafter_bf's prefetching form vs the stock skinny bf16 GEMM at the drafter's linears (q_a, q_b, kv, o_a, o_b, shared gate_up /
# down at both widths, main_proj 5120x15360), M 1..48, every form, byte for byte; (3) --perfmx at 5 / 10 / 15 / 20 / 30 / 40
# tokens (1-8 users x 5 rows) and (4) --perfbf at M 5 6 10 12 20 24 40 48, each warm (rotated copies) and cold (256 MiB read
# before every launch): stock, the rule, pinned forms, bytes compared. -> drksum.py: DKCHK, DKMX / DKBF tables, DKROUND (an
# ESTIMATE of a drafter round's kernel time at 1 / 2 / 4 / 8 users, stock -> the rule).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-p0-drkbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
O=${XDKB_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/drkbench}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XDK_IMAGE:-radiance-local:ds41-drk-a22a5dfa-5c13e555-l}
CARD=${CARD:-0}
ST=/opt/radiance/bin/r4d_selftest; LIB=/opt/radiance/share/radiance/kernels/libr4d.so
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xdsm drkbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xdsm drkbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -e $H/drksum.py ] || { log "RESULT xdsm drkbench: VIRHE $H/drksum.py missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xdsm drkbench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
  [ -e $H/drksum.py ] && echo "  (dry) $H/drksum.py present" || echo "  (dry) $H/drksum.py MISSING"
  docker image inspect "$IMAGE" > /dev/null 2>&1 && echo "  (dry) image $IMAGE present ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19))" || echo "  (dry) image $IMAGE MISSING"
fi
log "RESULT xdsm drkbench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD -> $O"
SEL="--entrypoint $ST $IMAGE $LIB"
run "gpurun dsdk-$$-mxc 1800 $DEV $SEL --mxcheck > $O/mxcheck.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xdsm drkbench mxcheck (rc $rc): $(grep -m1 '^MXVERDICT' $O/mxcheck.txt | cut -c1-300)"
run "gpurun dsdk-$$-bfc 1800 $DEV $SEL --bfcheck > $O/bfcheck.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xdsm drkbench bfcheck (rc $rc): $(grep -m1 '^BFVERDICT' $O/bfcheck.txt | cut -c1-300)"
run "gpurun dsdk-$$-mxr 1500 $DEV $SEL --perfmx 5,10,15,20,30,40 100 > $O/mx-rot.txt 2>&1"
run "gpurun dsdk-$$-mxk 1800 $DEV $SEL --cold 256 --perfmx 5,10,15,20,30,40 50 > $O/mx-cold.txt 2>&1"
run "gpurun dsdk-$$-bfr 1800 $DEV $SEL --perfbf 5,6,10,12,20,24,40,48 100 > $O/bf-rot.txt 2>&1"
run "gpurun dsdk-$$-bfk 2400 $DEV $SEL --cold 256 --perfbf 5,6,10,12,20,24,40,48 50 > $O/bf-cold.txt 2>&1"
run "python3 -I $H/drksum.py $O > $O/drksum.txt 2>&1"
if [ "$DRY" != 1 ]; then
  grep -h -E '^(MXSUM|BFSUM)' $O/mx-rot.txt $O/mx-cold.txt $O/bf-rot.txt $O/bf-cold.txt
  cat $O/drksum.txt
  while IFS= read -r l; do log "RESULT xdsm drkbench ${l:0:1500}"; done < <(grep -E '^(DKCHK|DKROUND)' $O/drksum.txt; grep -h -E '^(MXVERDICT|BFVERDICT)' $O/mx-rot.txt $O/mx-cold.txt $O/bf-rot.txt $O/bf-cold.txt)
  grep -q "OUTPUT DIFFERS" $O/mx-rot.txt $O/mx-cold.txt $O/bf-rot.txt $O/bf-cold.txt && log "RESULT xdsm drkbench: VIRHE a form's output differs from the stock kernel's"
fi
log "RESULT xdsm drkbench: done -- $O"
