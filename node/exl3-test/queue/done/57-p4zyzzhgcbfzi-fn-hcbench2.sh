#!/bin/bash
# 130-fn-hcbench2.sh (queue item, 9.11.2026, queue-ar; gpu02, ONE card (X130_CARD, default c3:00), every radiance
# container DOWN; ~15-25 min) -- the gated residual's kernels on one card, nothing beside them, at Flash-Next's shape,
# on the main line's image radiance-local:c16-sw3 (fn4-c16 fd18c52: c16-sw2 + hc_write's eight-wide form behind
# hc_writevec + the selftest's --perfhcw). Replaces 102-fn-hcbench.sh (1.15.1 images, rad-kbench; left in held/).
# WHY (127, mpB prefill profile, rank 0, 2048-row pieces, compute lane): hc_read 4105 ms of the lane's 13879 ms at
# 128k (29.6 %): 726 us a call (ffn, rotate 1; ~72 MB, ~27 GFLOP -> 100 GB/s, 37 TFLOPS) and 561 us (attn), measured
# beside the all-reduce lane; hc_write 1176 ms (8.5 %), ~210 us a 2048-row call (94.4 MB -> 450 GB/s). Decode c16:
# hc_read 79.8-81.6 us a call (iq4r) / 90-92 (EXL3), hc_write 17.7 us. This sizes both without the second lane:
#   A  r4d_selftest --perfhc M (hc_read_e4m3: every prefill-tile form that serves M, byte-compared, timed as one op;
#      then the decode form against the launcher's rule) at M = 16 33 48 64 512 2048 4096, --cold 256 (weights from
#      memory, as the engine reads them), under three environments: plain (no fn4 switch), F1n
#      (RADIANCE_FN4_HC_NORMWAVE=1, the main line), F1n + F1a v2 (+ RADIANCE_FN4_HC_FUSEDOWN=2)
#   B  r4d_selftest --perfhcw M (hc_write plain vs eight-wide, GB/s, bytes compared) at M = 16 48 64 512 2048 4096,
#      --cold 256
# Read-only on the cards' settings (no power / clock change); output runs/fn4/hcb130/*.txt + RESULT lines.
# DRYRUN=1 bash 130-fn-hcbench2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4/hcb130; L=$R/ds41-exl3.log
C3=GPU-dbba5c851cfef7b3
CARD=${X130_CARD:-$C3}
IMG=${X130_IMAGE:-radiance-local:c16-sw3}
MS_HC=${X130_M_HC:-"16 33 48 64 512 2048 4096"}; MS_HCW=${X130_M_HCW:-"16 48 64 512 2048 4096"}; COLD=${X130_COLD:-256}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
st() {   # st NAME ENVS ARGS... : r4d_selftest of the image's libr4d on the one card -> $O/NAME.txt
  local nm=$1 envs=$2 e ev=""; shift 2
  for e in $envs; do ev="$ev -e $e"; done
  if [ "$DRY" = 1 ]; then
    echo "+ timeout -k 15 600 docker run --rm --init --name fn4-130-$nm --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARD$ev --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so $* > $O/$nm.txt"
    return 0
  fi
  timeout -k 15 600 docker run --rm --init --name fn4-130-$nm --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARD $ev --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so "$@" > $O/$nm.txt 2>&1
  echo "exit $?" >> $O/$nm.txt
  docker rm -f fn4-130-$nm > /dev/null 2>&1
}
log "RESULT fn4 130-fn-hcbench2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) card $CARD, hc_read M [$MS_HC], hc_write M [$MS_HCW], --cold $COLD"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 130: VIRHE a radiance container is running -- the card is not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 130: VIRHE image $IMG missing"; exit 1; }
  mkdir -p $O
  trap 'docker ps -a --format "{{.Names}}" | grep -E "^fn4-130-" | xargs -r docker rm -f > /dev/null 2>&1' EXIT
fi
for env in "plain|" "F1n|RADIANCE_FN4_HC_NORMWAVE=1" "F1nfd2|RADIANCE_FN4_HC_NORMWAVE=1 RADIANCE_FN4_HC_FUSEDOWN=2"; do
  en=${env%%|*}; ev=${env#*|}
  for m in $MS_HC; do
    [ -e $R/sweep.stop ] && break 2
    st hc-$en-m$m "$ev" --cold $COLD --perfhc $m
    [ "$DRY" = 1 ] || log "RESULT fn4 130 hc_read $en M=$m: $(grep -vE '^ *$' $O/hc-$en-m$m.txt | grep -iE 'us|FAIL|exit' | sed 's/^ *//' | head -8 | tr -s ' ' | tr '\n' '|' | cut -c1-900)"
  done
done
for m in $MS_HCW; do
  [ -e $R/sweep.stop ] && break
  st hcw-m$m "" --cold $COLD --perfhcw $m
  [ "$DRY" = 1 ] || log "RESULT fn4 130 hc_write M=$m: $(grep -E '^hc_write|FAIL|exit' $O/hcw-m$m.txt | sed 's/^ *//' | head -3 | tr -s ' ' | tr '\n' '|' | cut -c1-500)"
done
log "RESULT fn4 130-fn-hcbench2: done | $O/*.txt"
