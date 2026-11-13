#!/bin/bash
# 80-iqr-gptq-rate.sh (6.11.2026; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN) -- how fast libquant's
# card path (lq_tcq_gpu.hip, what rad-convert --gpus runs) does GPTQ + the computed trellis over DeepSeek-V4.1's routed
# experts, to size the IQR conversion (15,360 experts of gate_up [4608 x 5120] + down [5120 x 2304] per base).
# Workstation CPU reference (3950X, 12 threads, same libquant): gate_up u2 L15 one expert 781.3 s (9 runs = 9 busy threads),
# planes fnv in logs; projected ~7,000 h for one base on gpu02's 16 cores -> the cards are the only way.
# Weights: layer 3's experts 0-7, decoded from the release's MXFP4 (model-00006) to f32, gate_up = [w1; w3]; Grams synthetic
# (seeded, energies over two decades; timing only -- the trellis's cost does not depend on the Gram's values).
# The harness (iqr-bin/iqr_moe_rate) runs libquant's own quantise loop (quant_fixture.h) inside the serving image, whose
# libquant it loads; IQR_GPUS is rad-convert's --gpus (rad_quant_devices). A weight under 32 runs goes whole to the next
# listed card in turn, so listing each card 2-3 times runs 2-3 weights a card at once (one weight = 9 workgroups at
# gate_up r512, 20 at down r256, 5 at down r1024; a card has 64 CUs).
# Forms (the V4 recipes' expert rules, tcq_colf=1 = the kRowF32 form moe_gemm_iqr is built for):
#   gu2  gate_up u2 tcq=15 r512 gain 1.04     gu3  gate_up u3 tcq=16 r512 gain 1.08
#   dn2  down    u2 tcq=15 r256 gain 1.04     dn3  down    u3 tcq=16 r256 gain 1.08     dn4  down u4 tcq=16 r1024 gain 1.12
# Phases: A gu2 one card one weight (latency, x2); B gu2 cards x1/x2/x3 (8/16/24 weights); C dn3 x1/x2/x3; D gu3, dn2 x2 and
# x3; E dn4 x3. Each run has a 1500 s deadline (docker stop; finished weights are still in its log).
# Per run: RESULT line with s a weight (wall / weights), us a weight row, and the weights' fnv (A's gu2 expert 0 is compared
# with the CPU run's fnv: the card path promises the host's bytes).
# Inputs beside this file: iqr-bin/iqr_moe_rate (md5 d64a7153...), iqr-l3/ (gu.0-7.f32, dn.0-7.f32, calib/; 1.4 GB).
# DRYRUN=1 bash 80-iqr-gptq-rate.sh : prints, runs and writes nothing. Estimate 40-90 min (unknown card rate; capped 1500 s a run).
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${IQR_BIN:-$Q/iqr-bin}
D=${IQR_DATA:-$Q/iqr-l3}
O=${IQR_OUT:-$HOME/gpu/ds41/exl3-test/runs/iqr-gptq-rate}
L=${IQR_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${IQR_IMAGE:-radiance-local:ds41-exl3-117-c528cdff-f2f3c3f7-mx4l}
CPU_FNV_GU2_E0=${CPU_FNV_GU2_E0:-c047d2ea4812afe2}   # workstation CPU run of gu2 expert 0 (778.8 s, 12 threads)
DEADLINE=${DEADLINE:-1500}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --memory 64g"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
COMMON="codes=CODES,tcq=TCQ,tcq_code=mad,tcq_rows=ROWS,block=1x*,scale=f32,tcq_colf=1,transform=fwht128,tcq_gain=GAIN,damp=0.01,tcq_order=1"
opts() { case $1 in   # the rule's options for form $1
  gu2) echo "$COMMON" | sed 's/CODES/u2/;s/TCQ/15/;s/ROWS/512/;s/GAIN/1.04/' ;;
  gu3) echo "$COMMON" | sed 's/CODES/u3/;s/TCQ/16/;s/ROWS/512/;s/GAIN/1.08/' ;;
  dn2) echo "$COMMON" | sed 's/CODES/u2/;s/TCQ/15/;s/ROWS/256/;s/GAIN/1.04/' ;;
  dn3) echo "$COMMON" | sed 's/CODES/u3/;s/TCQ/16/;s/ROWS/256/;s/GAIN/1.08/' ;;
  dn4) echo "$COMMON" | sed 's/CODES/u4/;s/TCQ/16/;s/ROWS/1024/;s/GAIN/1.12/' ;; esac; }
rows() { case $1 in gu*) echo 4608 ;; dn*) echo 5120 ;; esac; }
# go NAME FORM GPUS WORKERS EXPERTS -- one harness run, detached, with a deadline
go() {
  local name=$1 form=$2 gpus=$3 workers=$4 ex=$5 proj=${2:0:2} n w0 t0 t1 st
  [ "$proj" = gu ] || proj=dn
  local cmd="docker run -d --name iqr-rate-$name $DEV -e OMP_NUM_THREADS=32 -e IQR_GPUS=$gpus -e RAD_TESTPLUGIN_HOME=/opt/radiance/share/radiance -v $B:/b:ro -v $D:/d:ro --entrypoint /b/iqr_moe_rate $IMAGE /d 3 $proj $ex '$(opts $form)' $workers"
  if [ "$DRY" = 1 ]; then echo "+ $cmd ; wait <= ${DEADLINE}s ; docker logs > $O/$name.txt ; docker rm"; return; fi
  docker rm -f iqr-rate-$name > /dev/null 2>&1
  t0=$(date +%s); eval "$cmd" > /dev/null || { log "RESULT iqr-gptq-rate $name: VIRHE docker run failed"; return; }
  st=done
  while [ "$(docker inspect -f '{{.State.Running}}' iqr-rate-$name 2>/dev/null)" = true ]; do
    [ $(( $(date +%s) - t0 )) -ge $DEADLINE ] && { docker stop -t 5 iqr-rate-$name > /dev/null; st=TIMEOUT; break; }
    sleep 5
  done
  t1=$(date +%s); docker logs iqr-rate-$name > $O/$name.txt 2>&1; docker rm iqr-rate-$name > /dev/null 2>&1
  n=$(grep -c "status 0," $O/$name.txt)
  local tot; tot=$(awk '/^total /{print $2}' $O/$name.txt); [ -n "$tot" ] || tot=$((t1 - t0))
  local per=$(awk -v t=$tot -v n=$n 'BEGIN{ if (n > 0) printf "%.2f", t / n; else print "NA" }')
  local usrow=$(awk -v t=$tot -v n=$n -v r=$(rows $form) 'BEGIN{ if (n > 0) printf "%.0f", t / n / r * 1e6; else print "NA" }')
  local lat=$(awk '/status 0,/{for(i=1;i<=NF;i++) if($i=="s,"){s+=$(i-1); c++}} END{ if (c) printf "%.1f", s / c; else print "NA" }' $O/$name.txt)
  local bad=$(grep -c "status -" $O/$name.txt)
  log "RESULT iqr-gptq-rate $name: $st form $form cards [$gpus] workers $workers, $n weights ok ($bad failed) in ${tot}s -> $per s a weight ($usrow us a weight row), mean per-weight latency $lat s; $(grep -m1 'rad_quant_devices' $O/$name.txt)"
  grep -m2 -E "status -|refus|cannot|VIRHE|error" $O/$name.txt | sed "s/^/  $name: /" | while read -r x; do log "  $x"; done
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT iqr-gptq-rate: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT iqr-gptq-rate: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/iqr_moe_rate" ] || { log "RESULT iqr-gptq-rate: VIRHE $B/iqr_moe_rate missing"; exit 1; }
  for f in gu.0.f32 gu.7.f32 dn.0.f32 dn.7.f32 calib/gram.r0.blk.3.ffn_gate_up_exps.bin calib/gram.r0.blk.3.ffn_down_exps.7.bin; do
    [ -s "$D/$f" ] || { log "RESULT iqr-gptq-rate: VIRHE $D/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT iqr-gptq-rate: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT iqr-gptq-rate: start IMAGE=$IMAGE bin $(md5sum $B/iqr_moe_rate 2>/dev/null | cut -c1-8) data $D -> $O"
E8=0,1,2,3,4,5,6,7; E16=$E8,$E8; E24=$E16,$E8
# A: one weight on one card, twice (latency; expert 0's fnv vs the CPU's)
go A-gu2-c1 gu2 0 1 0,1
if [ "$DRY" != 1 ] && [ -n "$CPU_FNV_GU2_E0" ]; then
  f=$(grep -m1 "ffn_gate_up_exps.0.weight: status 0" $O/A-gu2-c1.txt | sed 's/.*fnv //')
  [ "$f" = "$CPU_FNV_GU2_E0" ] && log "RESULT iqr-gptq-rate exact: card gu2 expert 0 fnv $f == CPU's" \
                               || log "RESULT iqr-gptq-rate exact: card gu2 expert 0 fnv '$f' != CPU's $CPU_FNV_GU2_E0"
fi
# B: gate_up u2, one / two / three weights a card
go B-gu2-x1 gu2 0,1,2,3 4 $E8
go B-gu2-x2 gu2 0,1,2,3,0,1,2,3 8 $E16
go B-gu2-x3 gu2 0,1,2,3,0,1,2,3,0,1,2,3 12 $E24
# C: down u3
go C-dn3-x1 dn3 0,1,2,3 4 $E8
go C-dn3-x2 dn3 0,1,2,3,0,1,2,3 8 $E16
go C-dn3-x3 dn3 0,1,2,3,0,1,2,3,0,1,2,3 12 $E24
# D: the other two widths at two and three a card
go D-gu3-x2 gu3 0,1,2,3,0,1,2,3 8 $E16
go D-gu3-x3 gu3 0,1,2,3,0,1,2,3,0,1,2,3 12 $E24
go D-dn2-x2 dn2 0,1,2,3,0,1,2,3 8 $E16
go D-dn2-x3 dn2 0,1,2,3,0,1,2,3,0,1,2,3 12 $E24
# E: down u4 at 1024-row runs (5 workgroups a weight)
go E-dn4-x3 dn4 0,1,2,3,0,1,2,3,0,1,2,3 12 $E24
log "RESULT iqr-gptq-rate: done -> $O (per-weight lines in each run's .txt; projected hours per base = 15360 x (s a weight gate_up + s a weight down) / 3600)"
