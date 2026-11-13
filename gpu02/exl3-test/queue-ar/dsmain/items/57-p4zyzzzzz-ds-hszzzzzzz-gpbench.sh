#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzz-gpbench.sh (queue item, 10.11.2026, lever GPF agent, branch ds41-gpf; gpu02, ONE card (CARD, default 0),
# every radiance container DOWN, no server; ~15-30 min) -- lever GPF's one-card bench: gemm_nt_ggml's PREFILL rows (above the
# decode band: gg_wm_kernel, the WMMA kernel with the scale factored out; rank 0's 128k step spends 254 of 1206 ms there at
# 49-65 TFLOP/s profiled) in other tile geometries with the stock kernel's per-output arithmetic (libr4d/r4d_ds41_ggml_pf.hip:
# MT 16-row tiles a block, NT 16-column tiles a wave, WN waves a block; the same WMMA fragments / chains / scale fmas in the same
# order), from the ds41-gpf image's own r4d_selftest and libr4d:
# (1) --gpcheck: every form vs the stock WMMA kernel BYTE FOR BYTE at every DeepSeek-V4.1 dense TP4 shape (attn_q_b 8192x1280,
#     attn_output_b 5120x2048, attn_q_a 1280x5120, attn_output_a 1024x4096, attn_kv 512x5120, shexp gate_up / down at rank 0's
#     1536x5120 / 5120x768 and ranks 1-3's 1024x5120 / 5120x512, engram_kv 25600x6144) in q5_K / q6_K / q8_0 and the head
#     (32320x5120 q6_K, f32 out), M 49 63 64 100 1000 2047 2048 4095 4096 4097 4160; random blocks with finite scales (incl.
#     f16 subnormals and near-top values), activations with zeros, subnormals, inf rows and overflow rows.
# (2) --perfgp at M 4096 / 2048 (rotated weight copies past 256 MiB, mean of 30): the stock kernel, every form, and three
#     ablations of the stock geometry whose bytes differ by design (abl1 no chain-init / min group-sum LDS reads, abl2 no B
#     decode, abl4 no per-group fma): what each part of the kernel costs. Bytes of every form compared with the stock.
# (3) lever GPF-b (same two runs): gemm_nt_f32's prefill rows (the router N 96 / 384 and the compressor's kv | gate N 1024,
#     K 5120) in r4d_ds41_f32pf.hip's tiled forms 20..25 (the stock 32 lane-strided fmaf chains + xor butterfly) vs the stock
#     gemm_nt_f32, f32 and bf16 weight storage: GPF32CHECK byte for byte at M 49..4160, GPF32PERF at M 4096 / 2048.
# -> gpsum.py: GPCHK, GPROW (a (type, shape, M): stock / best / every form's speedup), GPABLT, GPEST (ESTIMATE of rank 0's
#    128k-step gemm_nt_ggml ms with the best form a shape against dsm-225-prof-pf128, and the 136k TTFT saving), GPFORM (the
#    best single form), GPTAB (the RADIANCE_DS41_GG_PF_TABLE string for the in-process A/B item xgq); GPF32CHK, GPF32ROW,
#    GPF32EST (ESTIMATE of the router + compressor ms a 128k step), "GPF32FORM ab" (the f32_pf form xgq switches).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzz-gpbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
O=${XGP_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/gpbench}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XGP_IMAGE:-radiance-local:ds41-gpf-edc9a9be-5c13e555-l}
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
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xgp gpbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xgp gpbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -e $H/gpsum.py ] || { log "RESULT xgp gpbench: VIRHE $H/gpsum.py missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xgp gpbench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
  [ -e $H/gpsum.py ] && echo "  (dry) $H/gpsum.py present, md5 $(md5sum $H/gpsum.py | cut -c1-8)" || echo "  (dry) $H/gpsum.py MISSING"
  if docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "  (dry) image $IMAGE present ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19))"
    echo "  (dry) image check (no device, no network): $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c "grep -qzF r4d_gg_pf_pin $LIB && echo 'libr4d has r4d_gg_pf_pin' || echo 'libr4d LACKS r4d_gg_pf_pin'; grep -qzF -- --gpcheck $ST && echo 'selftest has --gpcheck' || echo 'selftest LACKS --gpcheck'; grep -qzF gg_pf /opt/radiance/bin/radiance && echo 'radiance has gg_pf' || echo 'radiance LACKS gg_pf'; grep -qzF r4d_f32pf_pin $LIB && echo 'libr4d has r4d_f32pf_pin' || echo 'libr4d LACKS r4d_f32pf_pin'; grep -qzF f32_pf /opt/radiance/bin/radiance && echo 'radiance has f32_pf' || echo 'radiance LACKS f32_pf'" 2>&1 | tr '\n' ';')"
  else echo "  (dry) image $IMAGE MISSING"; fi
fi
log "RESULT xgp gpbench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD -> $O"
SEL="--entrypoint $ST $IMAGE $LIB"
run "gpurun dsgp-$$-chk 2400 $DEV $SEL --gpcheck > $O/gpcheck.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xgp gpbench check (rc $rc): $(grep -m1 '^GPVERDICT' $O/gpcheck.txt | cut -c1-400) | $(grep -m1 '^GPF32VERDICT' $O/gpcheck.txt | cut -c1-300)"
[ "$DRY" = 1 ] || grep -q '^GPVERDICT' $O/gpcheck.txt || log "RESULT xgp gpbench check: VIRHE no verdict line ($(tail -2 $O/gpcheck.txt | tr '\n' ' ' | cut -c1-300))"
run "gpurun dsgp-$$-perf 2400 $DEV $SEL --perfgp 4096,2048 30 > $O/gpperf.txt 2>&1"; rc=$?
[ "$DRY" = 1 ] || log "RESULT xgp gpbench perf (rc $rc): $(grep -m1 '^GPVERDICT' $O/gpperf.txt | cut -c1-400)"
run "python3 -I $H/gpsum.py $O > $O/gpsum.txt 2>&1"
if [ "$DRY" != 1 ]; then
  cat $O/gpsum.txt
  while IFS= read -r l; do log "RESULT xgp gpbench ${l:0:1800}"; done < <(grep -E '^(GPCHK|GPROW|GPABLT|GPEST|GPFORM|GPF32CHK|GPF32ROW|GPF32EST|GPF32FORM)' $O/gpsum.txt; grep -h -E '^(GPTAB|GPF32FORM best)' $O/gpperf.txt)
  grep -q "OUTPUT DIFFERS" $O/gpperf.txt && log "RESULT xgp gpbench: VIRHE a form's output differs from the stock kernel's (gpperf)"
  grep -q "DIFFERS" $O/gpcheck.txt && log "RESULT xgp gpbench: VIRHE the byte check found differences ($O/gpcheck.txt)"
fi
log "RESULT xgp gpbench: done -- $O"
