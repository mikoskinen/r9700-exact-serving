#!/bin/bash
# 57-pmc-cap.sh (6.11.2026 coordinator; from 14-exl3-bottleneck.sh phase F only) -- 54 phase F read every counter as 0 except SQ_BUSY_CYCLES
# (perf_event_paranoid 4: KFD PMC/profiler access is likely capability-gated). Reruns F with --cap-add PERFMON, then --privileged if still 0.
# exl3r4d's decode GEMMs at DeepSeek V4.1's EP-4 shapes, and which split-K each arm really launched.
# Background (12-splitk-microbench, 6.11. 17:5x): split-K 2/4 changed gate_up/down by 0..-5 % where the model said
# -45/-65 %, and at every M the dispatch reads ~210-310 GB/s of a 680 GB/s stream ceiling; time grows linearly with the
# slowest rank's live experts (gate_up K2: ~27 us + 18.8 us an expert; down K2: ~8 + 10.4 us). Three hypotheses:
#   L  latency chain / occupancy: too few waves in flight per CU, each wave's K loop a dependent chain (the model's)
#   C  per-CU throughput: the trellis decode's VALU (or LDS / WMMA issue) saturates the active CUs
#   M  memory: DRAM or L2 bandwidth / request rate of the codes' access pattern
# Steps (each perfmoe run ~5-15 s; same geometry, routing and binaries as 12):
#   A  which split-K launched: AMD_LOG_LEVEL=3 HIP API log of one dispatch an arm -> the exl3r4d GEMM's block size
#      (gate_up 256 threads = split-K 1, 512 = 2; down 128 = 1, 256 = 2, 512 = 4). Settles "did sk1 engage".
#   B  cache-hot control: --layers 1 --pools 1 (the routed experts, 17-32 MB, stay in L2 + the 64 MB MALL) vs the cold
#      default (48 layers x 2 pools, every dispatch from DRAM). Hot ~= cold -> not M; hot << cold -> M.
#   C  K sweep at fixed workgroups: gate_up --hidden 1280/2560/5120 (k-steps 80/160/320): the per-k-step slope and the
#      fixed part, base vs split-K. Split-K should halve the slope under L; under C or M the slope stays.
#   D  CU scaling: --cu-pct 50 vs 100 at M=4 (enough workgroups for both): time x2 -> per-CU throughput (C or L1/LDS);
#      ~same -> DRAM-bound (M).
#   E  libr4d's moe_gemm_w4a8 on the same routing (--r4d, a cheap-decode 4-bit kernel): its GB/s is what this card's
#      memory path gives an expert GEMM at these shapes -- if it reads 2x exl3r4d's, the difference is decode (C).
#   F  hardware counters, if a rocprofv3 is found (PROF_IMAGE, else rocm/dev-ubuntu-24.04:latest, else
#      radiance-ds41-build:9c7ce498): tests/gpu/pmc.sh's three groups -- SQ_INSTS_VALU / SQ_WAVE_CYCLES / SQ_WAIT_INST_ANY /
#      SQ_BUSY_CYCLES, VALUBusy / ValuPipeIssueUtil / MemUnitBusy / MeanOccupancyPerActiveCU / WAVE_DEP_WAIT /
#      WAVE_ISSUE_WAIT, FETCH_SIZE / GL2C hit-miss / TCP_REQ / SQ_INST_CYCLES_VALU|VMEM -- plus --kernel-trace (workgroup
#      size per dispatch, the step-A answer from the profiler). Reading: VALUBusy / ValuPipeIssueUtil > ~70 % -> C;
#      SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES high with VALUBusy low -> L or M; MemUnitBusy high + FETCH_SIZE / time near the
#      stream ceiling -> M. SQ_INSTS_VALU / (weights decoded) = VALU instructions a weight (the decode's cost).
#      rocprofv3 needs libdw.so.1 (absent from the build image, tests/gpu/pmc.sh): PMC_LIBDW = a host directory with it.
# Inputs: $B/perfmoe, $B/exl3r4d-splitk.so, $B/exl3r4d-base.so (12's bin/), IMAGE for steps A-E (any 1.17 serving image:
# its ROCm runtime and /opt/radiance/share/radiance/kernels/libr4d.so for step E).
# DRYRUN=1 bash 14-exl3-bottleneck.sh : prints every command, runs and writes nothing. Estimate ~15 min (F +10 min).
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3SK_BIN:-$Q/bin}
O0=${X3BN_OUT:-$HOME/gpu/ds41/exl3-test/runs/exl3-pmc-cap}; O=$O0
L=${X3SK_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3SK_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
res() { grep -m1 '^RESULT exl3r4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){printf "%s us", $(i-1)} ; for(i=1;i<=NF;i++) if($i=="GB/s"){printf " %s GB/s", $(i-1); exit}}'; }
resr4d() { grep -m1 '^RESULT libr4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){printf "%s us", $(i-1)} ; for(i=1;i<=NF;i++) if($i=="GB/s"){printf " %s GB/s", $(i-1); exit}}'; }
pm() {   # pm OUTFILE "extra env" perfmoe-args...
  local out=$1 env=$2; shift 2
  run "docker run --rm $DEV $env -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE $* > $out 2>&1"
}
arm() { case $1 in base) echo "--exl3 /b/exl3r4d-base.so" ;; sk0) echo "--exl3 /b/exl3r4d-splitk.so" ;;
                   sk1) echo "--exl3 /b/exl3r4d-splitk.so --ep-param" ;; esac; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT exl3-bottleneck: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # 6.11.: KFD teardown race (11)
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT exl3-bottleneck: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-splitk.so exl3r4d-base.so; do [ -e "$B/$f" ] || { log "RESULT exl3-bottleneck: VIRHE $B/$f missing"; exit 1; }; done
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT exl3-pmc-cap: start IMAGE=$IMAGE card $CARD -> $O"


runF() {   # runF NAME "docker flags"
  CAPN=$1 CAPF=$2 O=$O0/$1; run "mkdir -p $O"
  # F. counters (rocprofv3), when an image has it
  PI=""
  if [ "$DRY" != 1 ]; then
    for im in ${PROF_IMAGE:-} rocm/dev-ubuntu-24.04:latest radiance-ds41-build:9c7ce498; do
      [ -n "$im" ] && docker image inspect $im > /dev/null 2>&1 && \
        docker run --rm --network none --entrypoint sh $im -c 'command -v rocprofv3' > /dev/null 2>&1 && { PI=$im; break; }
    done
  else
    PI=${PROF_IMAGE:-rocm/dev-ubuntu-24.04:latest}
  fi
  if [ -z "$PI" ]; then
    log "RESULT exl3-pmc-cap F[$CAPN]: skipped -- no image with rocprofv3 (set PROF_IMAGE; PMC_LIBDW if it lacks libdw.so.1)"
  else
    G1="SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_TEX_LOAD SQ_INSTS_LDS"
    G2="WAVE_DEP_WAIT WAVE_ISSUE_WAIT ValuPipeIssueUtil VALUBusy MemUnitBusy MeanOccupancyPerActiveCU"
    G3="TCP_REQ TCP_REQ_MISS FETCH_SIZE SQ_INST_CYCLES_VMEM SQ_INST_CYCLES_VALU TA_BUSY_avr"
    G4="GL2C_HIT_sum GL2C_MISS_sum GL2C_EA_RDREQ_sum"   # GL2C limit 4 counters per pass (6.11. agent)
    LD=""; [ -n "${PMC_LIBDW:-}" ] && LD="-v $PMC_LIBDW:/libdw:ro -e LD_LIBRARY_PATH=/libdw"
    # the serving image's libr4d.so is not in the profiler image: exl3r4d only (perfmoe runs without --r4d)
    for pt in "gate_up 2 1 base" "gate_up 2 1 sk1" "gate_up 2.5 1 base" "down 2 1 base" "down 2 1 sk1" "gate_up 2 4 base"; do
      set -- $pt; op=$1 k=$2 m=$3 a=$4
      g=0
      for grp in "$G1" "$G2" "$G3" "$G4" "TRACE"; do
        g=$((g + 1)); d=/out/F-$op-k$k-m$m-$a-g$g
        if [ "$grp" = TRACE ]; then pmc="--kernel-trace"; else pmc="--pmc $grp"; fi
        run "docker run --rm $DEV $CAPF $LD -v $B:/b:ro -v $O:/out --entrypoint sh $PI -c \"timeout 300 rocprofv3 $pmc" \
            "--kernel-include-regex 'exl3r4d' --output-format csv -d $d -o run -- /b/perfmoe $(arm $a) --op $op --M $m" \
            "--bits $k --experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --layers 8 --iters 1" \
            "--reps 1 --warm-ms 0\" > $O/F-$op-k$k-m$m-$a-g$g.log 2>&1 || echo FAILED >> $O/F-$op-k$k-m$m-$a-g$g.log"
      done
      [ "$DRY" = 1 ] || log "RESULT exl3-pmc-cap F[$CAPN] $op K=$k M=$m $a: counters in $O/F-$op-k$k-m$m-$a-g{1,2,3,4} (csv), trace g5; $(grep -l FAILED $O/F-$op-k$k-m$m-$a-g*.log 2>/dev/null | wc -l) failed passes"
    done
  fi
  
}
nz() { python3 -c "import csv,sys; print(sum(float(r['Counter_Value'])!=0 for r in csv.DictReader(open(sys.argv[1])) if r['Counter_Name']=='SQ_INSTS_VALU'))" $O0/$1/F-gate_up-k2-m1-base-g1/run_counter_collection.csv 2>/dev/null || echo 0; }
runF perfmon "--cap-add PERFMON"
if [ "$DRY" = 1 ] || [ "$(nz perfmon)" = 0 ]; then
  [ "$DRY" = 1 ] || log "RESULT exl3-pmc-cap: PERFMON still 0 for SQ_INSTS_VALU -> --privileged"
  runF priv "--privileged"
  [ "$DRY" = 1 ] || log "RESULT exl3-pmc-cap: privileged nonzero SQ_INSTS_VALU dispatches $(nz priv)"
else
  log "RESULT exl3-pmc-cap: PERFMON works, nonzero SQ_INSTS_VALU dispatches $(nz perfmon)"
fi
log "RESULT exl3-pmc-cap: done ($O0)"
