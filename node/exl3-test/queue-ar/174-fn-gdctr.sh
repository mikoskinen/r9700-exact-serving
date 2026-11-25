#!/bin/bash
# 174-fn-gdctr.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- E10b's CAUSE CHECK (coordinator 11.11.: a counter reading before the redesign). 172: the DS variant's fused chain is
# exact but slower than the split pair at every M (K2 M4 +5.7 %, M16 +0.6 %); the static metadata says the fused kernel
# runs its down units at the gate_up path's register budget (fused <2,2> 182 / <2.5,2.5> 180 / <4,4> 184 VGPRs vs the
# split down 92 / 96 / 94). Counters per kernel (rocprofiler-sdk 1.4.1, 144b's harness, run as root for the SQ counters):
# g1 GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES (mean resident waves = WAVE_CYCLES / BUSY_CYCLES), g5 VALUBusy
# MemUnitBusy OccupancyPercent MeanOccupancyPerActiveCU, g2 SQ_INSTS_VALU SQ_INST_CYCLES_VALU SQ_INSTS_SALU SQ_INSTS_LDS
# -- on perfmoe --gd (both the split chain and the fused op each pass): DS EP4 rank geometry K 2 at M 4 and 8, K 2.5 at M 4,
# and Flash-Next's EP4 K4 at M 16 (where the fused op wins, 153 D / 156b). perfmoe b4ab7889 + exl3r4d-dsgd 2c66cff3
# (bin-dsgd), image c16-x3c. Then tools/ctrsum.py per kernel. Measurement only.
# DRYRUN=1 bash 174-fn-gdctr.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=${X174_OUT:-$R/fn4/gdctr}
B=; for c in "$Q/bin-dsgd" "$Q/../queue-ar/bin-dsgd" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-dsgd"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-dsgd}
PS=; for c in "$Q/prof-sdk" "$Q/../queue-ar/prof-sdk" "$HOME/gpu/ds41/exl3-test/queue-ar/prof-sdk"; do [ -z "$PS" ] && [ -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ] && PS=$(cd "$c" && pwd); done
TL=; for c in "$Q/tools" "$Q/../queue-ar/tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -z "$TL" ] && [ -e "$c/ctrsum.py" ] && TL=$(cd "$c" && pwd); done
IMAGE=${X174_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
CARD=${CARD:-0}
TIMC="--warm-ms 500 --iters ${X174_ITERS:-2} --reps ${X174_REPS:-2} --pooled 0 --ceiling 640"
DSG="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 96 --topk 6 --hidden 5120 --width 2304 --chain --chain-band _m16 --rand-data --gd $TIMC"
FNG="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 128 --topk 10 --hidden 2560 --width 640 --group 4 --group-mix 0.53 --chain --chain-band _m16 --rand-data --gd --bits 4 $TIMC"
# as the item's user with the card groups, so the counter files the tool writes under $O are ours (not root's)
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD -w /tmp"   # root (144b)   # -w: the tool writes ./.rocprofv3 (the image's WORKDIR /data is root's: 144's first run died on it)
P=/opt/rocm/core-10.1
PENV="-v $PS:$P:ro -e LD_LIBRARY_PATH=$P/lib -e LD_PRELOAD=$P/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so:$P/lib/librocprofiler-sdk.so -e ROCP_TOOL_LIBRARIES=$P/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so -e ROCPROFILER_REGISTER_LIBRARY=$P/lib/librocprofiler-sdk.so.1.4.1 -e ROCPROFILER_LIBRARY_CTOR=1 -e ROCPROF_COUNTER_COLLECTION=1 -e ROCPROF_OUTPUT_FORMAT=csv -e ROCPROF_OUTPUT_FILE_NAME=run -e ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=exl3r4d|moe_gemm"
GROUPS_=("GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES" "SQ_INSTS_VALU SQ_INST_CYCLES_VALU SQ_INSTS_SALU SQ_INSTS_LDS"
         "SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_INST_CYCLES_VMEM SQ_INSTS_TEX_LOAD" "TA_TA_BUSY TCP_REQ TCP_REQ_MISS GL2C_HIT GL2C_MISS"
         "VALUBusy MemUnitBusy OccupancyPercent MeanOccupancyPerActiveCU")
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUT GROUP(0 = plain) perfmoe-args...
  local out=$1 g=$2; shift 2
  local cn=fn-ctr-$$-$(basename $out .txt) pe="" od=""
  if [ "$g" != 0 ]; then
    od=$O/$(basename $out .txt)
    pe="$PENV -v $od:/out -e ROCPROF_OUTPUT_PATH=/out -e 'ROCPROF_COUNTERS=pmc: ${GROUPS_[$((g - 1))]}'"
    run "rm -rf $od; mkdir -p $od; chmod 777 $od"
  fi
  run "timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV ${pe//|/\\|} -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-dsgd.so $* > $out 2>&1; rc=\$?; docker rm -f $cn > /dev/null 2>&1; [ \$rc = 0 ] || echo \"\$(date '+%F %T') RESULT fn4 174 ctr: run $(basename $out) rc \$rc: \$(grep -vE '^W[0-9]' $out | tail -2 | tr '\n' ' ' | cut -c1-300)\" | tee -a $L"
}
# the configs: name|args
CFGS=(
  "ds-k2-m4|$DSG --bits 2 --limit 10 --M 4"
  "ds-k2-m8|$DSG --bits 2 --limit 10 --M 8"
  "ds-k25-m4|$DSG --bits 2.5 --limit 10 --M 4"
  "fn-k4-m16|$FNG --M 16"
)
log "RESULT fn4 174 ctr 174-fn-gdctr: start IMAGE=$IMAGE ($IMGID) card $CARD bin $B prof-sdk $PS -> $O"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 174 ctr: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 174 ctr: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 174 ctr: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  [ -n "$PS" ] && [ -e "$PS/lib/libhsa-amd-aqlprofile64.so.1" ] && [ -e "$PS/share/rocprofiler-sdk/config.yaml" ] || { log "RESULT fn4 174 ctr: VIRHE prof-sdk bundle missing or incomplete ($PS)"; exit 1; }
  [ -n "$TL" ] || { log "RESULT fn4 174 ctr: VIRHE tools/ctrsum.py missing"; exit 1; }
  for i in $(seq 1 12); do [ -z "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 5; done   # KFD idle
  mkdir -p $O
fi
# STEP 0: the tool must produce counter rows
pm $O/s0-ds-k2-m4-g1.txt 1 ${CFGS[0]#*|}
if [ "$DRY" != 1 ]; then
  n0=$(find $O/s0-ds-k2-m4-g1 -name '*counter_collection.csv' -exec cat {} + 2>/dev/null | grep -c exl3r4d)
  if [ "${n0:-0}" -lt 1 ]; then
    log "RESULT fn4 174 ctr: VIRHE step 0 -- no counter rows for the EXL3 kernel ($(ls -R $O/s0-ds-k2-m4-g1 2>/dev/null | tr '\n' ' ' | cut -c1-200) | $(grep -E 'E[0-9]{4}|rror|fail' $O/s0-ds-k2-m4-g1.txt | head -3 | tr '\n' ' ' | cut -c1-500))"
    exit 1
  fi
  log "RESULT fn4 174 ctr step 0: $n0 counter rows for the EXL3 kernel (g1) -- the profiler works on the card"
fi
for c in "${CFGS[@]}"; do
  nm=${c%%|*}; args=${c#*|}
  pm $O/$nm-plain.txt 0 $args
  [ "$DRY" = 1 ] || log "RESULT fn4 174 ctr $nm plain (perfmoe's own timing, no profiler): $(grep -iE 'us|GB/s' $O/$nm-plain.txt | grep -vE '^#|^W[0-9]' | tail -3 | tr -s ' ' | tr '\n' '|' | cut -c1-500)"
  for g in 1 2 5; do pm $O/$nm-g$g.txt $g $args; done
  [ "$DRY" = 1 ] || log "RESULT fn4 174 ctr $nm counters: $(timeout 60 python3 -I $TL/ctrsum.py $O $nm 2>&1 | tr '\n' '|' | cut -c1-1500)"
done
run "timeout 120 python3 -I $TL/ctrsum.py $O --table > $O/verdicts.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 174 ctr 174 verdicts (kernel -> bound): $(sed -E 's/ \([0-9]+ dispatches\):.*-> / -> /' $O/verdicts.txt | tr '\n' ';' | cut -c1-1200) | table $O/summary.tsv"
run "timeout -k 5 120 docker run --rm --init --name fn-ctr-chown-$$ --network none -v $O:/o --entrypoint /bin/sh $IMAGE -c 'chown -R $(id -u):$(id -g) /o'"
log "RESULT fn4 174 ctr 174-fn-gdctr: done"
