#!/bin/bash
# 57-p4zyzzzzz-ds-bzz-bench.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, ONE card (CARD, default 0), every radiance
# container DOWN, no server; ~20 min) -- replaces 57-p4zyzzzzz-ds-b-moebench (its MXFP4 arm never ran: the bench passed the
# rank's rows where moe_gemm_q takes the step's whole slot buffers; fixed). Two benches on the main image's kernels:
# (1) DeepSeek-V4.1's routed-expert GEMMs as the served engine runs them (expert parallel: 96 whole experts a rank of 384,
#     top 6, the SLOWEST rank's rows; gate_up 4608 x 5120 + down 5120 x 2304 a layer), M = 1 / 4 / 16 / 64 COLD (256 MiB read
#     before every launch, median of 50) and M = 1 WARM, formats:
#       mx     moe_gemm_q mxfp4a8 (the official release's experts): the rule, then main's small-M forms pinned
#              (r4d_moe_mxfp4_pin_small 2 / 1 / 3), each byte-compared with the rule
#       w4     moe_gemm_q w4a8h (the plain signed 4-bit grid, bf16 scale per 128 of K: the form family of Qwen's iq4r .rad)
#       iqrB   moe_gemm_iqr over IQR-V4 trellis experts, B = 4 / 3 / 2 (the computed 2^16-state trellis: decode-bound)
#       exl3   exl3r4d's moe_exl3_gate_up (prep inside) + moe_exl3_down at K 2 / 2.5 / 3, as deepseek41 issues them
#              (decode band _m16 to 16 tokens, prefill band _pf above); 2.25 = 11 K2.5 + 29 K2 layers, 2.9 = 35 K3 + 5 K2
#     -> kbsum.py: us a layer, MiB an expert, bits a weight, GB/s, model size, the VRAM share at TP4 (15.63 GiB a card for
#        experts today), the read hit from a Zipf fitted to item a's EXL3 2.25 hit, and an estimated expert ms a step at
#        1 / 4 / 8 users = 40 x (measured layer time + missed bytes over the link at 25 GB/s). Byte checks: MXFP4's pinned
#        forms vs the rule; IQR / w4 / EXL3 have no reference path in this harness (said in the table).
# (2) lever 1's one-card bench: gemm_nt_ggml's decode band at DeepSeek's dense TP4 shapes (q5_K / q6_K / q8_0; M 1 / 4 / 8),
#     the rule vs every pinned rows-a-wave / grid form (r4d_gg_gemv_pin), outputs byte-compared (exact by construction).
# Binaries (queue-ar/dsmain/bin): r4d_selftest-bench (ds41-main-bench 25ee316) and libr4d-bench.so (= the image's libr4d
# e736c019 sources + the test-only r4d_gg_gemv_pin hook; every other kernel the same sources); EXL3 from the image.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-bzz-bench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
B=${XDSB_BIN:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/bin}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
O=${XDSB_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kbench}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XDSB_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
CARD=${CARD:-0}
MS=${XDSB_MS:-"1 4 16 64"}; FMTS=${XDSB_FMTS:-mx,w4,iqr4,iqr3,iqr2}
X3=/opt/radiance/share/radiance/kernels/exl3r4d.so
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD -e DSMOE_EXL3=$X3 -e DSMOE_EXL3_K=exl3k2,exl3k2h,exl3k3"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xdsm kbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xdsm kbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in r4d_selftest-bench libr4d-bench.so; do [ -e "$B/$f" ] || { log "RESULT xdsm kbench: VIRHE $B/$f missing"; exit 1; }; done
  [ -e $H/kbsum.py ] || { log "RESULT xdsm kbench: VIRHE $H/kbsum.py missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xdsm kbench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT xdsm kbench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD bench $(md5sum $B/r4d_selftest-bench 2>/dev/null | cut -c1-8) lib $(md5sum $B/libr4d-bench.so 2>/dev/null | cut -c1-8) M [$MS] formats $FMTS + exl3 k2/k2h/k3 -> $O"
for m in $MS; do
  run "gpurun dsk-$$-m$m 1500 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-bench $IMAGE /b/libr4d-bench.so --cold 256 --perfdsmoe $m 50 $FMTS > $O/dsmoe-cold-m$m.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xdsm kbench cold M=$m (rc $?): $(grep -m1 'slowest rank' $O/dsmoe-cold-m$m.txt | grep -oE 'gets [0-9]+ routed rows over [0-9]+ experts') | $(grep -E '^DSMOE .* layer ' $O/dsmoe-cold-m$m.txt | sed 's/^DSMOE //; s/ gate_up + down / /' | cut -c1-90 | tr '\n' ';' | cut -c1-1200)"
  grep -q "OUTPUT DIFFERS" $O/dsmoe-cold-m$m.txt 2>/dev/null && log "RESULT xdsm kbench cold M=$m: VIRHE a pinned MXFP4 form's output differs from the rule's"
done
run "gpurun dsk-$$-w1 1500 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-bench $IMAGE /b/libr4d-bench.so --perfdsmoe 1 50 $FMTS > $O/dsmoe-warm-m1.txt 2>&1"
run "gpurun dsk-$$-gg 2400 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-bench $IMAGE /b/libr4d-bench.so --perfgg 1,4,8 100 q5_K,q6_K,q8_0 > $O/gg-rot.txt 2>&1"
[ "$DRY" = 1 ] || grep -q "OUTPUT DIFFERS" $O/gg-rot.txt && [ "$DRY" != 1 ] && log "RESULT xdsm kbench gemv: VIRHE a pinned form's output differs from the rule's ($O/gg-rot.txt)"
run "python3 -I $H/kbsum.py $O > $O/kbsum.txt 2>&1"
if [ "$DRY" != 1 ]; then
  while IFS= read -r l; do log "RESULT xdsm kbench ${l:0:1500}"; done < <(grep -E '^(KBSUM|GGSUM)' $O/kbsum.txt)
fi
log "RESULT xdsm kbench: done -- $O"
