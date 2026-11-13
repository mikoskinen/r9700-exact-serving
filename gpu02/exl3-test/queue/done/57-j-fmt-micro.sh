#!/bin/bash
# 70-fmt-micro.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN) -- per-expert decode cost of
# the routed-expert formats Radiance has kernels for, at DeepSeek V4.1 Flash's expert shapes (gate_up [4608, 5120],
# down [5120, 2304]), M = 1 and 4:
#   A. perfmoe (MoE dispatch, EP-4 routing over 96 experts a rank, top-6, slowest rank, ~2.9 live experts): EXL3 at
#      K 2 / 2.5 / 3 / 4 (exl3r4d-gu25.so: today's kernels + the 2.5-bit PF fix) and, on the same routing, libr4d's
#      moe_gemm_w4a8 (w4nl64a8h: 4-bit codebook, E4M3 scale per 64, int8 activations) via --r4d
#   B. r4d_selftest --perfiqr (dense GEMV/GEMM of ONE expert-shaped matrix, the IQR-V4 decode the MoE kernel shares:
#      computed 2^16-state trellis, f32 row scale, int8 rotated activations) at B = 2..6 bits; each call also times the
#      dense MXFP4 rows (gemm_mxfp4a8_*: the official release's expert format) at the same shape
# GGUF I-quants (IQ1_S / IQ2_XXS) are not here: their engine numbers exist (3.11. profile: gate_up 48.6 / 77.1 us a layer
# a card at 1 user) and the ds41 GGML kernel reads only Q8_0 Q5_K Q6_K IQ4_NL IQ2_XXS IQ1_S.
# Reading: A gives us a dispatch (divide by 'live experts a dispatch' in each file for us an expert); B gives us for one
# expert matrix with the whole card on it (a lower bound of the per-expert cost; compare formats at equal shape).
# Inputs (bin/ beside this file): perfmoe (gu25 build), exl3r4d-gu25.so. IMAGE: a 1.17 serving image (its libr4d.so
# and r4d_selftest are used). DRYRUN=1 bash 70-fmt-micro.sh : prints, runs and writes nothing. Estimate ~10 min.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${FMT_BIN:-$Q/bin-gu25}
O=${FMT_OUT:-$HOME/gpu/ds41/exl3-test/runs/fmt-micro}
L=${FMT_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${FMT_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 300 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
res() { grep -m1 "^RESULT $2" "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){printf "%s us", $(i-1)}; for(i=1;i<=NF;i++) if($i=="GB/s"){printf " %s GB/s", $(i-1); exit}}'; }
live() { grep -m1 'live experts a dispatch' "$1" 2>/dev/null | grep -oE '[0-9.]+ live' | cut -d' ' -f1; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT fmt-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT fmt-micro: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-gu25.so; do [ -e "$B/$f" ] || { log "RESULT fmt-micro: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT fmt-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT fmt-micro: start IMAGE=$IMAGE card $CARD -> $O"
# A. MoE dispatch: EXL3 K 2 / 2.5 / 3 / 4, and libr4d w4nl64a8h on the K=2 runs' routing
for op in gate_up down; do for m in 1 4; do
  line=""
  for k in 2 2.5 3 4; do
    f=$O/A-$op-k$k-m$m.txt
    xr=""; [ "$k" = 2 ] && xr="--r4d $R4D"
    run "docker run --rm $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-gu25.so $xr --op $op --M $m --bits $k $GEOM > $f 2>&1"
    [ "$DRY" = 1 ] || line="$line EXL3 K$k $(res $f exl3r4d);"
    [ "$DRY" = 1 ] || [ "$k" != 2 ] || line="$line w4nl64a8h $(res $f libr4d);"
  done
  [ "$DRY" = 1 ] || log "RESULT fmt-micro A $op M=$m (live experts $(live $O/A-$op-k2-m$m.txt)):$line"
done; done
# B. one expert matrix, dense: IQR-V4 B bits (+ the MXFP4 rows the same call times)
for sh in "4608 5120 gate_up" "5120 2304 down"; do
  set -- $sh; N=$1 K=$2 nm=$3
  for m in 1 4; do
    line=""
    for b in 2 3 4 5 6; do
      f=$O/B-$nm-iqr$b-m$m.txt
      run "docker run --rm $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE $R4D --perfiqr $N $K $m $b > $f 2>&1"
      [ "$DRY" = 1 ] || line="$line IQR$b $(grep -m1 -E 'iqr k[0-9]' $f | grep -oE '[0-9.]+ *us' | head -1);"
    done
    [ "$DRY" = 1 ] || log "RESULT fmt-micro B $nm [$N x $K] M=$m:$line | MXFP4 rows: $(grep -E 'gemm_mxfp4a8' $O/B-$nm-iqr4-m$m.txt | tr -s ' ' | cut -c1-60 | tr '\n' ';' | cut -c1-400)"
  done
done
log "RESULT fmt-micro: done -- $O (A: perfmoe files; B: r4d_selftest --perfiqr outputs, each with the MXFP4 rows)"
