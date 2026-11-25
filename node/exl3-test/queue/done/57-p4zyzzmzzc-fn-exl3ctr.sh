#!/bin/bash
# 144-fn-exl3ctr.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- E6 (coordinator 10.11.: the user asked whether EXL3 is optimised to the end -- it is not): a HARDWARE-COUNTER profile
# of the EXL3 expert kernels, so their bound (ALU / latency / occupancy / memory) is read from counters instead of
# inferred (90b: K4 EP-4 M16 gate_up 91.2 us at 75 % of 640 GB/s, VALU floor 43 %; fmtmicro's cache-resident control ran
# +10 % SLOWER when resident, so bytes alone do not explain it).
# Harness = 90b's arm H / I: perfmoe (bin-gu25, md5 b088bd01) with exl3r4d-gu25c.so (aa4703c9: gu25c, the K4 kernels the
# -fnx line serves) and the 1.17 line's libr4d in radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7 (6e63a9ccd389), Flash-Next
# shapes, MTP-verify routing (--group 4 --group-mix 0.53):
#   x-m16 / x-m64   EXL3 K4, EP-4 slowest rank (128 experts, --tp 1 --ep 4 --ep-rank -1), gate_up and down ALONE, at M 16
#                   (4 users x MTP 3 verify rows: c4) and M 64 (c16)  -- the production EXL3 line
#   i-m16 / i-m64   production iq4r beside it: TP4 --mod4 rank 0 (libr4d moe_gemm_w4a8_mod4 + exl3 K4 mod4), same M
# Per config: one plain run (perfmoe's own us, the sanity check against 90b), then seven counter passes, one run each,
# through rocprofiler-sdk 1.4.1 (queue-ar/prof-sdk: the tool library, the sdk, aqlprofile and the counter definitions
# from radiance-build-c16 -- the SAME ROCm build as the image: libamdhip64 cd4e0b14 / libhsa-runtime64 d89a964a in both),
# mounted at /opt/rocm/core-10.1 and preloaded exactly as rocprofv3 would (its env, captured 10.11.), kernel filter
# exl3r4d|moe_gemm, csv out:
#   g1 GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES      g2 SQ_INSTS_VALU SQ_INST_CYCLES_VALU SQ_INSTS_SALU SQ_INSTS_LDS
#   g3 SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_INST_CYCLES_VMEM SQ_INSTS_TEX_LOAD   g4 TA_TA_BUSY TCP_REQ TCP_REQ_MISS GL2C_HIT GL2C_MISS
#   g5 VALUBusy MemUnitBusy OccupancyPercent MeanOccupancyPerActiveCU      g6 LdsUtil LDSBankConflict ValuPipeIssueUtil L2CacheHit
#   g7 FetchSize GPUBusy WriteUnitStalled SQ_INSTS_SMEM
# (the gfx12 counters rocprofiler-sdk 1.4.1 defines; a group the card refuses is logged and skipped). Then tools/ctrsum.py:
# per kernel the means per dispatch, the shares and a verdict line, plus runs/fn-exl3ctr/summary.tsv. STEP 0: one g1 run of
# x-m16 gate_up must produce counter rows, or the item stops (VIRHE, the tool's stderr). Measurement only, nothing exact
# to check. Every run bounded (timeout -k 15 300, --init, named, rm -f), --network none.
# DRYRUN=1 bash 144-fn-exl3ctr.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=${X144_OUT:-$R/fn-exl3ctr}
B=${X144_BIN:-}
for c in "$Q/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/bin-gu25"; do
  [ -z "$B" ] && [ -e "$c/perfmoe" ] && [ -e "$c/exl3r4d-gu25c.so" ] && B=$(cd "$c" && pwd)
done
B=${B:-$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25}
PS=; for c in "$Q/prof-sdk" "$Q/../queue-ar/prof-sdk" "$HOME/gpu/ds41/exl3-test/queue-ar/prof-sdk"; do [ -z "$PS" ] && [ -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ] && PS=$(cd "$c" && pwd); done
TL=; for c in "$Q/tools" "$Q/../queue-ar/tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -z "$TL" ] && [ -e "$c/ctrsum.py" ] && TL=$(cd "$c" && pwd); done
IMAGE=${X144_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}; IMGID=6e63a9ccd389
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
CARD=${CARD:-0}
FN="--experts 512 --topk 10 --hidden 2560 --width 640 --warm-ms 500 --iters ${X144_ITERS:-2} --reps ${X144_REPS:-2} --pooled 0 --ceiling 640"
VER="--group 4 --group-mix 0.53"
EPG=${FN/--experts 512/--experts 128}; EPA="--tp 1 --ep 4 --ep-rank -1 --h-rot 1"
# as the item's user with the card groups, so the counter files the tool writes under $O are ours (not root's)
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"   # -w: the tool writes ./.rocprofv3 (the image's WORKDIR /data is root's: 144's first run died on it)
P=/opt/rocm/core-10.1
PENV="-v $PS:$P:ro -e LD_LIBRARY_PATH=$P/lib -e LD_PRELOAD=$P/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so:$P/lib/librocprofiler-sdk.so -e ROCP_TOOL_LIBRARIES=$P/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so -e ROCPROFILER_REGISTER_LIBRARY=$P/lib/librocprofiler-sdk.so.1.4.1 -e ROCPROFILER_LIBRARY_CTOR=1 -e ROCPROF_COUNTER_COLLECTION=1 -e ROCPROF_OUTPUT_FORMAT=csv -e ROCPROF_OUTPUT_FILE_NAME=run -e ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=exl3r4d|moe_gemm"
GROUPS_=("GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES" "SQ_INSTS_VALU SQ_INST_CYCLES_VALU SQ_INSTS_SALU SQ_INSTS_LDS"
         "SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_INST_CYCLES_VMEM SQ_INSTS_TEX_LOAD" "TA_TA_BUSY TCP_REQ TCP_REQ_MISS GL2C_HIT GL2C_MISS"
         "VALUBusy MemUnitBusy OccupancyPercent MeanOccupancyPerActiveCU" "LdsUtil LDSBankConflict ValuPipeIssueUtil L2CacheHit"
         "FetchSize GPUBusy WriteUnitStalled SQ_INSTS_SMEM")
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUT GROUP(0 = plain) perfmoe-args...
  local out=$1 g=$2; shift 2
  local cn=fn-ctr-$$-$(basename $out .txt) pe="" od=""
  if [ "$g" != 0 ]; then
    od=$O/$(basename $out .txt)
    pe="$PENV -v $od:/out -e ROCPROF_OUTPUT_PATH=/out -e 'ROCPROF_COUNTERS=pmc: ${GROUPS_[$((g - 1))]}'"
    run "rm -rf $od; mkdir -p $od"
  fi
  run "timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV ${pe//|/\\|} -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-gu25c.so --r4d $R4D $* > $out 2>&1; rc=\$?; docker rm -f $cn > /dev/null 2>&1; [ \$rc = 0 ] || echo \"\$(date '+%F %T') RESULT fn-ctr: run $(basename $out) rc \$rc: \$(grep -vE '^W[0-9]' $out | tail -2 | tr '\n' ' ' | cut -c1-300)\" | tee -a $L"
}
# the configs: name|args
CFGS=(
  "x-m16-gu|$EPA --op gate_up --a-rot 1 --exl3-row exl3r4d_moe_gate_up_m16 --M 16 --bits 4 $VER $EPG"
  "x-m16-dn|$EPA --op down --exl3-row exl3r4d_moe_down_m16 --M 16 --bits 4 $VER $EPG"
  "x-m64-gu|$EPA --op gate_up --exl3-row exl3r4d_moe_gate_up_pf --M 64 --bits 4 $VER $EPG"
  "x-m64-dn|$EPA --op down --exl3-row exl3r4d_moe_down_pf --M 64 --bits 4 $VER $EPG"
  "i-m16-gu|--tp 4 --mod4 --rank 0 --h-rot 1 --op gate_up --a-rot 1 --exl3-row exl3r4d_moe_gate_up_mod4_m16 --M 16 --bits 4 $VER $FN"
  "i-m16-dn|--tp 4 --mod4 --rank 0 --h-rot 1 --op down --exl3-row exl3r4d_moe_down_mod4_m16 --M 16 --bits 4 $VER $FN"
  "i-m64-gu|--tp 4 --mod4 --rank 0 --h-rot 1 --op gate_up --exl3-row exl3r4d_moe_gate_up_mod4_pf --M 64 --bits 4 $VER $FN"
  "i-m64-dn|--tp 4 --mod4 --rank 0 --h-rot 1 --op down --exl3-row exl3r4d_moe_down_mod4_pf --M 64 --bits 4 $VER $FN"
)
log "RESULT fn-ctr 144-fn-exl3ctr: start IMAGE=$IMAGE ($IMGID) card $CARD bin $B prof-sdk $PS -> $O"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn-ctr: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn-ctr: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  for f in perfmoe exl3r4d-gu25c.so; do [ -e "$B/$f" ] || { log "RESULT fn-ctr: VIRHE $B/$f missing"; exit 1; }; done
  [ "$(md5sum "$B/exl3r4d-gu25c.so" | cut -c1-8)" = aa4703c9 ] && [ "$(md5sum "$B/perfmoe" | cut -c1-8)" = b088bd01 ] || { log "RESULT fn-ctr: VIRHE bin md5s (perfmoe b088bd01, exl3r4d-gu25c aa4703c9)"; exit 1; }
  [ -n "$PS" ] && [ -e "$PS/lib/libhsa-amd-aqlprofile64.so.1" ] && [ -e "$PS/share/rocprofiler-sdk/config.yaml" ] || { log "RESULT fn-ctr: VIRHE prof-sdk bundle missing or incomplete ($PS)"; exit 1; }
  [ -n "$TL" ] || { log "RESULT fn-ctr: VIRHE tools/ctrsum.py missing"; exit 1; }
  for i in $(seq 1 12); do [ -z "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 5; done   # KFD idle
  mkdir -p $O
fi
# STEP 0: the tool must produce counter rows
pm $O/s0-x-m16-gu-g1.txt 1 ${CFGS[0]#*|}
if [ "$DRY" != 1 ]; then
  n0=$(find $O/s0-x-m16-gu-g1 -name '*counter_collection.csv' -exec cat {} + 2>/dev/null | grep -c exl3r4d)
  if [ "${n0:-0}" -lt 1 ]; then
    log "RESULT fn-ctr: VIRHE step 0 -- no counter rows for the EXL3 kernel ($(ls -R $O/s0-x-m16-gu-g1 2>/dev/null | tr '\n' ' ' | cut -c1-200) | $(grep -E 'E[0-9]{4}|rror|fail' $O/s0-x-m16-gu-g1.txt | head -3 | tr '\n' ' ' | cut -c1-500))"
    exit 1
  fi
  log "RESULT fn-ctr step 0: $n0 counter rows for the EXL3 kernel (g1) -- the profiler works on the card"
fi
for c in "${CFGS[@]}"; do
  nm=${c%%|*}; args=${c#*|}
  pm $O/$nm-plain.txt 0 $args
  [ "$DRY" = 1 ] || log "RESULT fn-ctr $nm plain (perfmoe's own timing, no profiler): $(grep -iE 'us|GB/s' $O/$nm-plain.txt | grep -vE '^#|^W[0-9]' | tail -3 | tr -s ' ' | tr '\n' '|' | cut -c1-500)"
  for g in 1 2 3 4 5 6 7; do pm $O/$nm-g$g.txt $g $args; done
  [ "$DRY" = 1 ] || log "RESULT fn-ctr $nm counters: $(timeout 60 python3 -I $TL/ctrsum.py $O $nm 2>&1 | tr '\n' '|' | cut -c1-1500)"
done
run "timeout 120 python3 -I $TL/ctrsum.py $O --table > $O/verdicts.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn-ctr 144 verdicts (kernel -> bound): $(sed -E 's/ \([0-9]+ dispatches\):.*-> / -> /' $O/verdicts.txt | tr '\n' ';' | cut -c1-1200) | table $O/summary.tsv"
log "RESULT fn-ctr 144-fn-exl3ctr: done"
