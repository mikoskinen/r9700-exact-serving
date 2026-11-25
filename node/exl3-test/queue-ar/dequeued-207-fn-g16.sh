#!/bin/bash
# 207-fn-g16.sh (queue item, 12.11.2026, queue-ar; gpu02, one card, every radiance container DOWN; <= 3 min) -- hc_bi 5's G16
# (form G's kernel A' at 16 rows a block, hc_readv bit 6, fn4-x3-sp3 860c69a0) against the 8-row form G it must equal byte
# for byte, and its time: r4d_selftest --perfhcv M at 12 / 16 / 32 / 64 rows with R4D_SELFTEST_BI_DEC=1, hc_readv bit 6
# off (R4D_SELFTEST_READV_OR 0) and on (64); each run first compares every output byte of G16 vs form G. 204 reference:
# bi_dec M16 28.3 / M32 41.3 / M64 70.0 us vs today's forms 21.5 / 36.7 / 56.5.
# DRYRUN=1 bash 207-fn-g16.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${X207_IMAGE:-radiance-local:c16-x3sp8}
C3=GPU-dbba5c851cfef7b3
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 207-fn-g16: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) r4d_selftest --perfhcv M 12 / 16 / 32 / 64, bi_dec, readv bit 6 off / on, card $C3"
if [ "$DRY" = 1 ]; then
  for m in 12 16 32 64; do for o in 0 64; do echo "+ docker run ... -e R4D_SELFTEST_BI_DEC=1 -e R4D_SELFTEST_READV_OR=$o r4d_selftest --perfhcv $m 500"; done; done
  log "RESULT fn4 207-fn-g16: done"; exit 0
fi
docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 207: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 207: VIRHE image $IMG missing"; exit 1; }
mkdir -p $O
for m in 12 16 32 64; do
  for o in 0 64; do
    g=$O/207-g16-m$m-or$o.txt
    timeout -k 15 240 docker run --rm --init --name fn4-207-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 -e R4D_SELFTEST_BI_DEC=1 -e R4D_SELFTEST_READV_OR=$o \
      --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so --perfhcv $m 500 > $g 2>&1
    docker rm -f fn4-207-$$ > /dev/null 2>&1
    log "RESULT fn4 207 M $m readv-or $o: $(grep -E 'G16 \(hc_readv 64\)|hc_read_e4m3 M=.*warm' $g | sed 's/^ *//' | tr '\n' ' ' | cut -c1-500)"
  done
done
log "RESULT fn4 207-fn-g16: done"
