#!/bin/bash
# 57-p4zyzzzzz-ds-g1-mrbench.sh (queue item, 10.11.2026, lever D2 agent, branch ds41-mr; gpu02, ONE card (CARD, default 0),
# every radiance container DOWN, no server; ~15-25 min) -- lever D2's one-card bench: gemm_nt_ggml / logits_gemm_ggml's
# decode band (M <= 48) in the multi-row form (libr4d/r4d_ds41_ggml_mr.hip: every weight block dequantised ONCE for all
# the call's activation rows, H weight rows a wave side by side, MT rows a block, slabs of staged activations; its bytes
# must be the stock kernel's), against the stock path, at DeepSeek-V4.1-Flash's dense TP4 shapes.
# (1) --mrcheck: the form vs the stock path BYTE FOR BYTE at q5_K / q6_K / q8_0 x every dense shape (attn_q_b 8192x1280,
#     attn_output_b 5120x2048, attn_q_a 1280x5120, attn_output_a 1024x4096, attn_kv 512x5120, shexp gate_up / down at
#     rank 0's width 1536x5120 / 5120x768 and ranks 1-3's 1024x5120 / 5120x512, engram_kv 25600x6144) and the head
#     (logits_gemm_ggml 32320x5120 q6_K, f32 out), M = 1..48, 14 forms each (every (H, R) pair at its largest, most at a small
#     MT cap, the wave64 unit, the rule); random blocks with finite scales (incl. f16 subnormals and near-top values),
#     activations with zeros, subnormals, an inf row and an overflow row. MRINFO: does the stock path give row m of an
#     M-row call the bytes it gives the row alone (decides whether drafting's verify rows can differ from plain decode).
# (2) --perfmr warm (rotated weight copies past 256 MiB, mean of 100) and (3) cold (256 MiB read before each launch,
#     median of 50): at the 2.25 .rad's (type, shape) pairs (attention q5_K/q6_K, shexp q6_K/q8_0, engram_kv q6_K, head
#     q6_K), M 1 2 3 4 6 8 12 16 24 32 48: the stock rule, the 6 pinned stock forms (r4d_gg_gemv_pin rows a wave 1/2/4 x
#     grid 256/1024), and 13 multi-row forms; every output byte-compared with the stock rule's. MRSUM a (type, shape, M).
# -> mrsum.py: MRTAB (a (type, shape, mode) row of stock / stock best / mr best us), MRCOST (how flat: mr's cost at M 8 /
#    24 / 48 over its M=1, stock's beside), MRCHK, MRFINAL (bytes; geomean speedups at M 1 / 8 / 24 / 48).
# Binaries (queue-ar/dsmain/bin): r4d_selftest-mr and libr4d-mr.so (ds41-mr: the image's libr4d e736c019 sources + the
# --perfgg pin hook of ds41-main-bench + lever D2's units and the gg_gemm hook, OFF unless pinned; every other kernel the
# same sources), run in the main image.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-g1-mrbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
B=${XMR_BIN:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/bin}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
O=${XMR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/mrbench}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XMR_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
CARD=${CARD:-0}
MS=${XMR_MS:-1,2,3,4,6,8,12,16,24,32,48}
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
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xdsm mrbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xdsm mrbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in r4d_selftest-mr libr4d-mr.so; do [ -e "$B/$f" ] || { log "RESULT xdsm mrbench: VIRHE $B/$f missing"; exit 1; }; done
  [ -e $H/mrsum.py ] || { log "RESULT xdsm mrbench: VIRHE $H/mrsum.py missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xdsm mrbench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
  for f in r4d_selftest-mr libr4d-mr.so; do [ -e "$B/$f" ] && echo "  (dry) $B/$f present, md5 $(md5sum $B/$f | cut -c1-8)" || echo "  (dry) $B/$f MISSING"; done
  [ -e $H/mrsum.py ] && echo "  (dry) $H/mrsum.py present" || echo "  (dry) $H/mrsum.py MISSING"
  docker image inspect "$IMAGE" > /dev/null 2>&1 && echo "  (dry) image $IMAGE present" || echo "  (dry) image $IMAGE MISSING"
fi
log "RESULT xdsm mrbench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD bench $(md5sum $B/r4d_selftest-mr 2>/dev/null | cut -c1-8) lib $(md5sum $B/libr4d-mr.so 2>/dev/null | cut -c1-8) M [$MS] -> $O"
# (1) the byte check
run "gpurun dsmr-$$-chk 2400 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-mr $IMAGE /b/libr4d-mr.so --mrcheck q5_K,q6_K,q8_0 > $O/mrcheck.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xdsm mrbench check (rc $rc): $(grep -m1 '^MRVERDICT check' $O/mrcheck.txt | cut -c1-400)"
[ "$DRY" = 1 ] || { grep -q '^MRVERDICT check' $O/mrcheck.txt || log "RESULT xdsm mrbench check: VIRHE no verdict line ($(tail -2 $O/mrcheck.txt | tr '\n' ' ' | cut -c1-300))"; }
# (2) warm, (3) cold timing
run "gpurun dsmr-$$-rot 2400 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-mr $IMAGE /b/libr4d-mr.so --perfmr $MS 100 > $O/mr-rot.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xdsm mrbench rot (rc $rc): $(grep -m1 '^MRVERDICT' $O/mr-rot.txt | cut -c1-400)"
run "gpurun dsmr-$$-cold 3000 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-mr $IMAGE /b/libr4d-mr.so --cold 256 --perfmr $MS 50 > $O/mr-cold.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xdsm mrbench cold (rc $rc): $(grep -m1 '^MRVERDICT' $O/mr-cold.txt | cut -c1-400)"
run "python3 -I $H/mrsum.py $O > $O/mrsum.txt 2>&1"
if [ "$DRY" != 1 ]; then
  grep -h '^MRSUM' $O/mr-rot.txt $O/mr-cold.txt
  cat $O/mrsum.txt
  while IFS= read -r l; do log "RESULT xdsm mrbench ${l:0:1500}"; done < <(grep -E '^(MRCHK|MRCOST|MRFINAL)' $O/mrsum.txt)
  grep -q "OUTPUT DIFFERS" $O/mr-rot.txt $O/mr-cold.txt && log "RESULT xdsm mrbench: VIRHE a form's output differs from the stock rule's (mr-rot / mr-cold)"
  grep -q "DIFFERS" $O/mrcheck.txt && log "RESULT xdsm mrbench: VIRHE the byte check found differences ($O/mrcheck.txt)"
fi
log "RESULT xdsm mrbench: done -- $O"
