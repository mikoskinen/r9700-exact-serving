#!/bin/bash
# 155-fn-hcread.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card at a time, every radiance container DOWN; ~15 min) --
# the iq4r round's first readings (coordinator 10.11., from 151's per-op profile), no server:
#  P-1  WHERE PF2's DOWN PRODUCT LOSES (hc_read at 2048 rows: 586 us a call on 151's critical rank, 28 % of its prefill
#       lane; the stack's 32-row down product ~146 us of 480 alone). r4d_selftest --perfhcd (fn4-c16 8f5a2895): the
#       32-row down product timed with its B fragment loads, its A loads, its WMMAs and its slice barrier DELETED one by
#       one and together (r4d_hc_pf2_abl, measurement only; wrong output) + the stock 64-row form, cold (256 MiB flushed
#       before every launch), 2048 and 512 rows, card c3:00. Reading: B loads cost = stock - noB etc.; if "no WMMA" is
#       near stock the product is load-bound (B re-reads: 64 row blocks x ~6.9 MB of fragments a call from L2), if "no B"
#       drops most the lever is fewer B re-reads (B through LDS shared by more rows, the same chains).
#  SKEW PROBE  151 at 128k: rank 0 (c3:00) hc_read 3308 ms vs ranks 1-3 3684-3763 (-10..-12 %), gemm_nt_q 1362 vs
#       1462-1533. --perfhcv 2048 (the hc_read forms + PF2's stages, cold) on EACH card alone: same kernel, same bytes --
#       a card-to-card spread here is silicon / clock under the 210 W cap (not a code lever); none = placement or
#       co-running in the server (a code lever).
#  first: --case hc_read_e4m3 on c3:00 (every hc_read form byte for byte, so the template change kept the kernels).
# Image radiance-local:c16-p1 (fn4-c16 8f5a2895 = c16-sw12's tree + the reading arms). Power untouched.
# DRYRUN=1 bash 155-fn-hcread.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4/hcread; L=$R/ds41-exl3.log
IMG=${X155_IMAGE:-radiance-local:c16-p1}; IMGID=d411227c4c15
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
st() {   # st CARD OUT TIMEOUT selftest-args...
  local card=$1 out=$2 to=$3; shift 3
  if [ "$DRY" = 1 ]; then echo "+ timeout -k 15 $to docker run ... -e ROCR_VISIBLE_DEVICES=$card r4d_selftest $IMG libr4d.so $* > $out"; return 0; fi
  timeout -k 15 $to docker run --rm --init --name fn4-155-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$card --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so "$@" > $out 2>&1
  local rc=$?; docker rm -f fn4-155-$$ > /dev/null 2>&1; return $rc
}
log "RESULT fn4 155-fn-hcread: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19))"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 155: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 155: VIRHE image $IMG is not $IMGID"; exit 1; }
  mkdir -p $O
fi
# ---- byte checks first
f=$O/155-selftest-hc_read_e4m3.txt
st $C3 $f 1500 --case hc_read_e4m3; rc=$?
if [ "$DRY" != 1 ]; then
  np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f)
  log "RESULT fn4 155 selftest hc_read_e4m3 (c3:00): exit $rc, $np ok / $nf FAIL / $ns skip$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -3 | sed 's/^ *//' | cut -c1-200 | tr '\n' '|')")"
  case $rc in 0|2) ;; *) log "RESULT fn4 155: VIRHE the selftest exited $rc -- no timings"; exit 1 ;; esac
  [ $nf -gt 0 ] && { log "RESULT fn4 155: VIRHE hc_read forms differ -- no timings"; exit 1; }
fi
# ---- P-1: the down product's deletion arms (c3:00)
for m in 2048 512; do
  f=$O/155-perfhcd-m$m.txt
  st $C3 $f 900 --cold 256 --perfhcd $m 30
  [ "$DRY" = 1 ] || log "RESULT fn4 155 P-1 down product M=$m (c3:00, cold 256 MiB): $(grep -E 'PF2 down|FAIL|skip' $f | sed 's/^ *//; s/PF2 down M=[0-9]* //' | tr -s ' ' | tr '\n' '|' | cut -c1-1400)"
done
# ---- the skew probe: the same hc_read on each card alone
for cd in c3:$C3 83:$C83 86:$C86 03:$C03; do
  nm=${cd%%:*}; card=${cd#*:}
  f=$O/155-perfhcv-$nm.txt
  st $card $f 900 --cold 256 --perfhcv 2048 30
  [ "$DRY" = 1 ] || log "RESULT fn4 155 skew probe card $nm:00 alone (hc_read M=2048 cold): $(grep -E '^hc_read_e4m3 M=|PF2 stages M=2048 hc_readv (2|7)' $f | sed 's/^ *//' | tr -s ' ' | tr '\n' '|' | cut -c1-700)"
done
log "RESULT fn4 155-fn-hcread: done"
