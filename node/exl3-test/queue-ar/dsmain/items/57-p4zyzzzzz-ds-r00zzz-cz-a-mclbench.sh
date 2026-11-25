#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-a-mclbench.sh (queue item, 12.11.2026, ds-main-port; MCL G1 input, user GO via the coordinator):
# HOST-ONLY gpu02 CPU decode throughput bench (no GPU, no container, cards idle): the MCL builder's static Zen2 binary decbench
# (queue-ar/dsmain/mcl/bin/decbench md5 5c5648ad; floor = AVX2 trellis decode + FMA, exact = + the exact WMMA-chain emulation),
# threads 1 / 8 / 12 / 16 / 32, M 1 / 2 / 4, K 2 / 2.5, 1.5 s each -> runs/xdsm/mcl1/decbench-gpu02.txt (~1.5-2 min).
# Guard: refuses while any radiance container runs (cards must be idle); the first 4 DECBENCH lines must say 0 differ / 0 wrong.
# Sorts after r00zzz-cz-a-base, before r00zzz-cz-mcl0probe (C and en_US). DRYRUN=1: prints, runs nothing.
set -u
T=$HOME/gpu/ds41/exl3-test; LOG=$T/runs/ds41-exl3.log; B=$T/queue-ar/dsmain/mcl/bin; O=$T/runs/xdsm/mcl1
log() { echo "$(date '+%F %T') RESULT xmclb $*" | tee -a $LOG; }
if [ "${DRYRUN:-0}" = 1 ]; then
  echo "+ docker ps (refuse if a radiance container runs); (cd $B && md5sum -c MD5SUMS); mkdir -p $O"
  echo "+ $B/decbench --threads 1,8,12,16,32 --m 1,2,4 --k 2,2.5 --secs 1.5 --pool 48 > $O/decbench-gpu02.txt"
  [ -x $B/decbench ] && echo "DRYRUN: 0 MISSING" || echo "DRYRUN: MISSING decbench"; exit 0
fi
if docker ps --format '{{.Names}}' | grep -qiE '^radiance'; then log "VIRHE a radiance container is running -- host bench refused (cards must be idle)"; exit 0; fi
( cd $B && md5sum -c MD5SUMS >/dev/null 2>&1 ) || { log "VIRHE MD5SUMS mismatch in $B"; exit 0; }
mkdir -p $O
log "start: host-only decbench, governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null), $(nproc) cpus"
timeout 600 $B/decbench --threads 1,8,12,16,32 --m 1,2,4 --k 2,2.5 --secs 1.5 --pool 48 > $O/decbench-gpu02.txt 2>&1; rc=$?
grep -qE "DECBENCH VIRHE" $O/decbench-gpu02.txt && log "VIRHE decbench self-check failed (see $O/decbench-gpu02.txt)"
log "done rc $rc: $(grep -E '^DECBENCH' $O/decbench-gpu02.txt | grep -E 't16|threads 16|16 ' | head -4 | tr '\n' ' ' | cut -c1-600) -- $O/decbench-gpu02.txt"
