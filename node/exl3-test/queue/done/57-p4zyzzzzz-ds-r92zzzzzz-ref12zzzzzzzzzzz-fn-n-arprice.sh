#!/bin/bash
# 228-fn-arprice.sh (queue item, 13.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~15 min) -- PRICING a coded exact
# RS / AG for the prefill's critical comm lane (coordinator 13.11. after 209): r4d_selftest --perfar4 at 2048x2560 and 4096x2560 (a piece's
# message, 10.5 / 21 MB), 300 launches, on radiance-local:c16-x3sp22 (+ the SP rows at overlap 1, the lane's knobs), the lane's CU budget
# RADIANCE_FN4_AR_LANE_BLOCKS 0 (= 4) / 8 / 16 / 24 x RADIANCE_FN4_AR_LANE_THREADS 0 / 512: per row us / GB/s of the coded exact lane AR
# (ar_twoshot_4rank_exact_lane, ~12.2 bits a value on the wire) and the plain lane AR (coded 0), +-xpush, against SP's raw RS (xpush) and
# AG at the same budget -- the coded form pays only where its encode / decode keeps up with the wire.
# DRYRUN=1 bash 228-fn-arprice.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${X228_IMAGE:-radiance-local:c16-x3sp22}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS=$C3,$C83,$C86,$C03
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 228-fn-arprice: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) --perfar4 2048x2560,4096x2560 300, lane blocks 0/8/16/24 x threads 0/512"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 228: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 228: VIRHE image $IMG missing"; exit 1; }
fi
mkdir -p $O
for lb in 0 8 16 24; do
  for lt in 0 512; do
    f=$O/228-ar-b$lb-t$lt.txt
    if [ "$DRY" = 1 ]; then echo "+ docker run ... -e RADIANCE_FN4_AR_LANE_BLOCKS=$lb -e RADIANCE_FN4_AR_LANE_THREADS=$lt r4d_selftest --perfar4 2048x2560,4096x2560 300 > $f"; continue; fi
    timeout -k 15 600 docker run --rm --init --name fn4-228-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS -e RADIANCE_FN4_AR_LANE_BLOCKS=$lb -e RADIANCE_FN4_AR_LANE_THREADS=$lt \
      --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so --perfar4 2048x2560,4096x2560 300 > $f 2>&1
    rc=$?; docker rm -f fn4-228-$$ > /dev/null 2>&1
    log "RESULT fn4 228 blocks $lb threads $lt (exit $rc): $(grep -E '^ *(2048x2560|4096x2560) ' $f | sed 's/^ *//' | grep -oE '^[0-9x]+ \([0-9.]+ MB\)|AR coded exact \(lane[^;]*|AR lane plain \(coded 0\)[^;]*|SP RS raw xpush \(lane\)[^;]*|SP AG raw \(lane\)[^;]*|AR two-shot exact xpush [0-9][^;]*' | tr '\n' ';' | cut -c1-1400)"
  done
done
log "RESULT fn4 228-fn-arprice: done ($O/228-ar-*.txt)"
