#!/bin/bash
# 210-fn-zsplit.sh (queue item, 12.11.2026, queue-ar; gpu02, one card, every radiance container DOWN; <= 3 min) -- SP-hc's
# quarter read with the PF2 column split (hc_readv bit 7, fn4-x3-sp3 eea7bc35): r4d_selftest --perfhcv 512 with the K split
# pinned to 2048 (R4D_SELFTEST_KC_ROWS, the quarter of a 2048-row piece), the outputs of bit 7 (and bits 7 + 1, the 32-row
# tile) against today's (R4D_SELFTEST_CMP_OR: every byte), and the times with bit 7 on (R4D_SELFTEST_READV_OR 128) vs off;
# 202 reference: pinned 512 rows 210.8-219.9 us (down 74.9-98.2), own split 172.7, the 2048-row piece 592.
# DRYRUN=1 bash 210-fn-zsplit.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${X210_IMAGE:-radiance-local:c16-x3sp10}
C3=GPU-dbba5c851cfef7b3
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 210-fn-zsplit: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) r4d_selftest --perfhcv 512, kc pin 2048, column split vs not, card $C3"
hv() {   # hv LABEL M KC CMP_OR READV_OR
  local g=$O/210-$1.txt
  if [ "$DRY" = 1 ]; then echo "+ docker run ... -e R4D_SELFTEST_KC_ROWS=$3 -e R4D_SELFTEST_CMP_OR=$4 -e R4D_SELFTEST_READV_OR=$5 r4d_selftest --perfhcv $2 500"; return 0; fi
  timeout -k 15 240 docker run --rm --init --name fn4-210-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 -e R4D_SELFTEST_KC_ROWS=$3 -e R4D_SELFTEST_CMP_OR=$4 \
    -e R4D_SELFTEST_READV_OR=$5 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --perfhcv $2 500 > $g 2>&1
  docker rm -f fn4-210-$$ > /dev/null 2>&1
  log "RESULT fn4 210 $1: $(grep -E 'hc_readv [0-9]+ vs 0|hc_read_e4m3 M=.*warm|PF2 stages' $g | sed 's/^ *//' | tr '\n' ' ' | cut -c1-900)"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 210: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 210: VIRHE image $IMG missing"; exit 1; }
  mkdir -p $O
fi
hv q512-z-off 512 2048 128 0
hv q512-z-on 512 2048 130 128
hv p2048-z 2048 0 128 128
log "RESULT fn4 210-fn-zsplit: done"
