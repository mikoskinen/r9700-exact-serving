#!/bin/bash
# 161-fn-dnread.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- WHAT THE EXL3 DOWN KERNEL'S FIXED COST IS (the user, via the coordinator: the decode-band down loses on its intercept
# -- lab READING 5: 10.3 vs libr4d 1.7 us a dispatch -- and memory-side fixes are refuted: PF 4/6, streamed K, FU_PF_DN).
# One card, perfmoe (exl3-main37 e345bb0) at Flash-Next's EP4 rank geometry (128 experts of 512, top-10, hidden 2560,
# width 640, --ep 4 --ep-rank -1, h_rot 1, the 4-bit group mix), op down, row exl3r4d_moe_down_m16, M = 1 / 2 / 4 / 16:
#  * the stock .so (= c16-x3c's) beside libr4d's moe_gemm_w4a8 on the same routing (--r4d): the gap a dispatch;
#  * STAGE DELETION ARMS (EXL3R4D_EXIT_STAGE, the lab's own reading points in exl3r4d_gemm_kernel): 99 the dispatch alone
#    (launch + this grid, every workgroup returns at entry), 1 + the expert search (find_run), 2 + the plane pointers,
#    3 + the A side (the h rows' staging), 4 + the K loop (no epilogue; the lab notes say a stage-4 exit re-schedules the
#    K loop, so 4 is an upper bound on the loop); the stock run adds the epilogue (output rotation, svh, stores, zero rows);
#    EXL3R4D_ABLATE_DECODE (the K loop without the trellis decode);
#  * EXL3R4D_WGSTAMP=3 (entry / exit / hardware slot a workgroup) at M 1 / 4 / 16: workgroups, live ones, the dispatch span,
#    the first round's size (slots resident at once), rounds, the ramp (entry spread of the first round), the per-workgroup
#    time (median / p90), WGPs used -- launch + ramp vs rounds vs per-workgroup latency.
#  Static (code object, no GPU; in the RESULT line): the stock down kernel (exl3r4d_gemm_kernel<4, 2, 1, ...>) holds
#  94 VGPRs, 68 SGPRs, 140 B static LDS.
# Bins (queue-ar/bin-dnr): perfmoe + exl3r4d-{stock,stage99,stage1,stage2,stage3,stage4,nodecode,wgs}.so, md5s checked.
# DRYRUN=1 bash 161-fn-dnread.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/dnread
B=; for c in "$Q/bin-dnr" "$Q/../queue-ar/bin-dnr" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-dnr"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-dnr}
IMAGE=${X161_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --op down --exl3-row exl3r4d_moe_down_m16 --bits 4 --group 4 --group-mix 0.53 --experts 128 --topk 10 --hidden 2560 --width 640 --pooled 0 --ceiling 640 --warm-ms 500"
TIM="--iters ${X161_ITERS:-20} --reps ${X161_REPS:-5}"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
VARS="stock stage99 stage1 stage2 stage3 stage4 nodecode"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-dnr-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -E 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/' || echo NA; }
log "RESULT fn4 161-fn-dnread: start IMAGE=$IMAGE card HIP $CARD bins $B GEO '$GEO' $TIM | static: stock down kernel 94 VGPRs, 68 SGPRs, 140 B static LDS"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 161: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 161: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 161: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O
fi
for M in 1 2 4 16; do
  for v in $VARS; do
    if [ $v = stock ]; then pm m$M-$v $v $GEO $TIM --M $M --r4d $R4D; else pm m$M-$v $v $GEO $TIM --M $M; fi
  done
  if [ "$DRY" != 1 ]; then
    line="M $M: libr4d $(usl m$M-stock 'libr4d') us | exl3r4d stock $(usl m$M-stock 'exl3r4d')"
    for v in stage99 stage1 stage2 stage3 stage4 nodecode; do line="$line | $v $(usl m$M-$v 'exl3r4d')"; done
    log "RESULT fn4 161 down $line (us a dispatch; stage deltas: 99 launch+grid, 1-99 search, 2-1 planes, 3-2 A side, 4-3 K loop, stock-4 epilogue)"
  fi
done
for M in 1 4 16; do
  pm w$M wgs $GEO --iters 2 --reps 2 --M $M --wgstamps /o/w$M.csv
  [ "$DRY" = 1 ] || log "RESULT fn4 161 timeline M $M: $(python3 - $O/w$M.csv <<'PY' 2>&1 | cut -c1-600
import csv, sys, statistics as st
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["op"] == "down"]
if not rows: print("no down rows"); sys.exit()
e = [int(r["entry"]) for r in rows]; x = [int(r["exit"]) for r in rows]; hw = [int(r["hw"]) for r in rows]
t0 = min(e); first_exit = min(v for v in x if v > 0)
live = [r for r, h in zip(rows, hw) if h >> 31]
dur = sorted((int(r["exit"]) - int(r["entry"])) / 100.0 for r in live if int(r["exit"]) > 0)
fr = [v for v in e if v < first_exit]
wgp = len({h & 0x7fffffff for h in hw})
print("%d workgroups (%d live), span %.2f us, first round %d (ramp %.2f us), rounds ~%.1f, per-WG median %.2f p90 %.2f us, %d slots (HW_ID1 distinct)" % (
    len(rows), len(live), (max(x) - t0) / 100.0, len(fr), (max(fr) - t0) / 100.0 if fr else 0, len(rows) / max(1, len(fr)),
    st.median(dur) if dur else 0, dur[int(0.9 * (len(dur) - 1))] if dur else 0, wgp))
PY
)"
done
log "RESULT fn4 161-fn-dnread: done"
