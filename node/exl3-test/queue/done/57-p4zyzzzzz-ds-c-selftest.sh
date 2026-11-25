#!/bin/bash
# 57-p4zyzzzzz-ds-c-selftest.sh (queue item, 10.11.2026, ds-main-port agent; ~3 min) -- held item 57-p6a-t3x-selftest (86a)
# converted to the main image (coordinator 10.11.): the same four-rank selftest cases on radiance-local:ds41-main-e736c019-
# 5c13e555's own r4d_selftest + libr4d (ds41-main carries ds41-t3x's pcnm chain merged with main's libr4d), output in
# runs/t3x-selftest-main ("86a CLEAN" gates 57-p4zyzzzzz-ds-d-t3xab). Everything below is 86a unchanged but the defaults.
# 86a-t3x-selftest.sh (queue item, 8.11.2026, queue-ar; gpu02, ALL FOUR cards, every radiance container DOWN, no server; ~3 min)
# -- 80-arpcn-selftest (7.11.: 47 ok, 0 FAIL on the arpcn image) again on the tuned3x image's libr4d, before 86b serves the
# fused collective: ds41-t3x is the pcnm chain cherry-picked onto ds41-lat (argather + the nvfp4 fix), with r4d_rows.cpp's
# releasing list merged by hand (ar_mhc_pcn_4rank beside moe_gather_ar_4rank), so the four-rank rows are checked on this build:
#   ar_mhc_post_collapse_rmsnorm_4rank  four rank threads, one card each, the fused collective twice on the same inputs against
#                                       x folded in ascending rank order + the single-card mhc_post_collapse_rmsnorm_bf16 row
#   ar_oneshot_4rank_exact, ar_twoshot_4rank_exact, ar_hc_write_4rank, ar_gather_hc_write_4rank   the rows sharing its
#                                       instance and protocol, as regressions
# The selftest is the image's own /opt/radiance/bin/r4d_selftest (built with that libr4d, ds41-t3x 67c9cc3).
# Hang protection: each case is its own process in its own named container under gpurun (timeout -k, SIGKILL, docker rm -f);
# the kernels' only wait is the one-shot handshake, which traps past R4D_AR_SPIN_MAX.
# Out: $O/<case>.txt, $O/SUMMARY.txt ("86a CLEAN" when every case has rc 0, >= 1 ok and 0 FAIL; 86b-t3x-ab.sh needs it).
# DRYRUN=1 bash 86a-t3x-selftest.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
O=${X86_SELFTEST:-$HOME/gpu/ds41/exl3-test/runs/t3x-selftest-main}
L=${X86_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X79_T3_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
CASES=${X86_CASES:-"ar_mhc_post_collapse_rmsnorm_4rank ar_oneshot_4rank_exact ar_twoshot_4rank_exact ar_hc_write_4rank ar_gather_hc_write_4rank"}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT t3x-selftest: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT t3x-selftest: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT t3x-selftest: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"; rm -f "$O/SUMMARY.txt"
else
  echo "+ mkdir -p $O"
fi
log "RESULT t3x-selftest: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) cases [$CASES] -> $O"
bad=0
for c in $CASES; do
  run "gpurun st-$$-${c:0:14} 600 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/$c.txt 2>&1"
  rc=$?
  if [ "$DRY" != 1 ]; then
    nf=$(grep -c 'FAIL' $O/$c.txt); nok=$(grep -cE '^ *(ok|PASS)' $O/$c.txt); ns=$(grep -ciE '^ *skip' $O/$c.txt)
    { [ $rc = 0 ] && [ $nf = 0 ] && [ $nok -gt 0 ]; } || bad=$((bad + 1))
    r="$c: rc $rc, $nok ok, $nf FAIL, $ns skip$([ $nf -gt 0 ] && echo " -- first: $(grep -m1 FAIL $O/$c.txt | cut -c1-300)")$({ [ $rc = 124 ] || [ $rc = 137 ]; } && echo ' -- STOPPED by the limit (hang)')"
    echo "$r" >> $O/SUMMARY.txt; log "RESULT t3x-selftest $r"
  fi
done
[ "$DRY" = 1 ] && { log "RESULT t3x-selftest: dry run done"; exit 0; }
if [ $bad = 0 ]; then echo "86a CLEAN" >> $O/SUMMARY.txt; log "RESULT t3x-selftest: done -- 86a CLEAN ($O)"
else echo "86a NOT CLEAN ($bad case(s))" >> $O/SUMMARY.txt; log "RESULT t3x-selftest: VIRHE $bad case(s) not clean -> 86b will not start ($O)"; exit 1; fi
