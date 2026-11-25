#!/bin/bash
# 57-p4zyzzzzz-ds-b-moebench.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, ONE card (CARD, default 0), every
# radiance container DOWN, no server; ~15 min) -- coordinator 10.11. (the user asks whether iq4r would be faster than the
# official 4-bit for DeepSeek; this is the kernel half, the other half is bytes over PCIe): DeepSeek-V4.1-Flash's
# routed-expert GEMMs at its TP4 layout -- expert parallel, 96 whole experts a rank of 384, top 6, the slowest rank's rows
# -- gate_up 4608 x 5120 (parts 2) and down 5120 x 2304, M = 1 / 4 / 16 / 64 tokens a step, COLD (256 MiB read before
# every launch, median of 50), on the MAIN image's libr4d (radiance-local:ds41-main-e736c019-5c13e555 = PR #266 head +
# main 451b136; IQR-V4 kernels and moe_gemm_mxfp4 incl. main's 220cb3f / b954585 small-M forms):
#   mx    moe_gemm_mxfp4a8 (the official release's E2M1 + E8M0/32 experts): the rule's form, then pinned forms
#         (r4d_moe_mxfp4_pin_small 2 never small-M, 1 small-M, 3 small-M K256x4), each compared byte for byte
#   iqrB  moe_gemm_iqr over IQR-V4 trellis experts at B = 4 (IQ4R) and, for reference, 3 (IQ3R) and 2 (IQ2R)
# us a call and GB/s of the touched experts' bytes, a gate_up + down layer sum per format.
# The bench is r4d_selftest --perfdsmoe of branch ds41-main-bench e39a758 (ds41-main + the bench function only; built
# in radiance-ds41-build:9c7ce498, same toolchain as main's Dockerfile), mounted from queue-ar/dsmain/bin; the kernels are
# the image's /opt/radiance/share/radiance/kernels/libr4d.so.
# Reference (6.11., 57-j fmt-micro, older IQR kernels, DENSE one-matrix GEMV, not this MoE dispatch): gate_up [4608 x 5120]
# M=1 IQR4 32.5 us vs gemm_mxfp4a8_gemv 23.4 us; down M=1 IQR4 23.1 vs 13.7 us.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-b-moebench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
B=${XDSB_BIN:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/bin}
O=${XDSB_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/moebench}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XDSB_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
CARD=${CARD:-0}
MS=${XDSB_MS:-"1 4 16 64"}; FMTS=${XDSB_FMTS:-mx,iqr4,iqr3,iqr2}
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
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
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xdsm moebench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xdsm moebench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/r4d_selftest-dsmoe" ] || { log "RESULT xdsm moebench: VIRHE $B/r4d_selftest-dsmoe missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xdsm moebench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT xdsm moebench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD bench $(md5sum $B/r4d_selftest-dsmoe 2>/dev/null | cut -c1-8) M [$MS] formats $FMTS, cold 256 MiB -> $O"
for m in $MS; do
  f=$O/dsmoe-m$m.txt
  run "gpurun dsmoe-$$-m$m 1200 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-dsmoe $IMAGE $R4D --cold 256 --perfdsmoe $m 50 $FMTS > $f 2>&1"
  rc=$?
  [ "$DRY" = 1 ] && continue
  log "RESULT xdsm moebench M=$m (rc $rc): $(grep -m1 'slowest rank' $f | grep -oE 'gets [0-9]+ routed rows over [0-9]+ experts') | $(grep '^DSMOE' $f | sed 's/^DSMOE //' | tr '\n' ';' | cut -c1-1500)"
  grep -q "OUTPUT DIFFERS" $f && log "RESULT xdsm moebench M=$m: VIRHE a pinned MXFP4 form's output differs from the rule's ($f)"
done
[ "$DRY" = 1 ] || log "RESULT xdsm moebench summary (us a gate_up + down layer, the rank's rows): $(for m in $MS; do echo "M=$m: $(grep -E '^DSMOE .* layer ' $O/dsmoe-m$m.txt 2>/dev/null | sed -E 's/^DSMOE ([a-z0-9]+) layer M=[0-9]+ gate_up \+ down ([0-9.]+) us.*/\1 \2/' | tr '\n' ',' )"; done | tr '\n' ' ')"
log "RESULT xdsm moebench: done -- $O"
