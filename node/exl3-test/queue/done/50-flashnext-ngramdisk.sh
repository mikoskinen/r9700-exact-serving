#!/bin/bash
# 50-flashnext-ngramdisk.sh (6.11.2026; user: "flash nextiä voisi tämän osalta testata ennen tuotantoa") -- Qwen3.8 Flash-Next
# production config with the n-gram table on disk instead of /dev/shm RAM: the 5.11. stock396 single-server set 1:1 (stock
# image, qdpy.yaml c3+03 :8090, MNBT 2048, u12 forced seed 3), only EXTRA="--ngram-placement disk" added (last value wins).
# Page cache evicted first (cold disk). Reference = stock396 (5.11.: 1-user 16.74 ms, c4 381.7, c8 523.6, pf 4923.7, cc_sim u12).
# Production is down (chain-after); qdpy goes down at the end, chain-after.sh restores production afterwards.
set -u
C=$HOME/radiance-1.15.1/compose; Q=$HOME/gpu/ds41/runs/q; D=$HOME/gpu/ds41; T=$HOME/gpu/ds41/exl3-test
QL=$HOME/gpu/ds41/runs/qwen-tp4-1103.log; XL=$T/runs/ds41-exl3.log
log() { echo "$(date '+%F %T') $*" | tee -a $QL >> $XL; }
n0=$(wc -l < $QL)
log "RESULT fnngd single-server test: start (Flash-Next stock 1.15.1, qdpy.yaml c3+03 :8090, n-gram on disk, u12 forced seed 3; reference stock396 5.11.)"
python3 $Q/evict.py > /dev/null 2>&1; log "RESULT fnngd evict: $(free -g | awk '/Mem:/{print "free " $4 " GiB, buff/cache " $6 " GiB"}')"
export RADIANCE_IMAGE=stilldeadcode/radiance:1.15.1 QF=qdpy.yaml QC=radiance-qdpy QPORT=8090 QURL=http://127.0.0.1:8090 GUARD_MS=21 MAXSEQS=16 HP=16384 PREFIX=61440 MNBT=2048 EXTRA="--ngram-placement disk"
cd $Q && timeout 1200 ./qgate.sh fnngd || { log "RESULT fnngd: VIRHE qgate failed -> rest skipped"; cd $C && docker compose -f qdpy.yaml down > /dev/null 2>&1; exit 1; }
log "RESULT fnngd start check: $(docker inspect -f '{{.Config.Image}}' radiance-qdpy 2>&1); $(docker logs radiance-qdpy 2>&1 | grep -m2 -i -E 'n-?gram' | tr '\n' ' ' | cut -c1-300)"
CCARGS="--duration 660 --warmup 300 --effort xhigh,medium --gen-mode forced --seed 3" timeout 1500 ./qccsim.sh 12 fnngd
timeout 1200 ./qgate.sh fnngdb || log "RESULT fnngd: VIRHE qgate fnngdb failed"
O=$Q/sw-fnngd; mkdir -p $O
for p in c4 c8 pf turns; do
  timeout 1800 python3 $D/mb.py --tag fnngd --phases $p --dur 200 --warm 20 > $O/$p.txt 2>&1
  log "RESULT fnngd quick $p: $(python3 $Q/qfmt.py $p $O/$p.txt $Q/sw-stock396/$p.txt stock396 2>&1 | tail -1)"
done
cd $C && docker compose -f qdpy.yaml down > /dev/null 2>&1
tail -n +$((n0 + 1)) $QL | grep -E "RESULT (restart-fnngd|guard fnngd|cc_sim fnngd)" >> $XL
log "RESULT fnngd: done; qdpy down (production restore by chain-after.sh)"
