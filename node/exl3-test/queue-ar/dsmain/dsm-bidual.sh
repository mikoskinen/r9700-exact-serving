#!/bin/bash
# dsm-bidual.sh (11.11.2026, queue-ar/dsmain, ds41-bi, lever BI-e) -- NOT a queue item; the body of the BI-e byte-check item.
# Log tag "RESULT xbid ", out runs/xdsm/bidual (XBID_OUT). LEVER BI-e'S ON-CARD BYTE CHECK (= dsm-biop.sh's frame, other modes).
# One card (HIP_VISIBLE_DEVICES=0), no server, every radiance container DOWN. The image's r4d_selftest, per arm (libr4d reads its
# form switches at load, so one process a mode an arm):
#   --dualcheck  gemm_nt_ggml_dual (ONE launch: the decode group's blocks in lever D2's multi-row form + the prefill group's in
#                the WMMA form) vs the two gemm_nt_ggml launches the engine issues today, byte for byte over every output, at
#                DeepSeek-V4.1-Flash's dense TP4 shapes / types, M0 (decode rows, XBID_M0) x M1 (prefill rows, XBID_M1),
#                both block orders; + DUALTIME (the pair vs the two launches, M0 4 / 8 / 24 / 48 x M1 2112 / 4160);
#   --mrcheck    lever D2's form vs the stock decode kernel (its kernel is now gg_mr_body behind gg_mr_kernel: the refactor's
#                bytes), q5_K / q6_K / q8_0, M 1..48;
#   --gpcheck    lever GPF's forms vs the stock WMMA kernel (gg_pf_body behind gg_pf_kernel: the same), M 2048 / 4096 + tails.
# Arms: stock (every libr4d form switch unset) and stack (RADIANCE_DS41_GG_GEMV=2, _GG_PF=1 + the 7-entry table, _F32_PF=25).
# PASS = every DUALCHECK line IDENTICAL, every MRCHECK / GPCHECK line the forms' "same" verdict, rc 0. ~15-25 min.
# DRYRUN=1 bash dsm-bidual.sh : prints every command; checks the image's r4d_selftest literals and the helpers; no GPU.
# Knobs: XBID_IMAGE (required), XBID_ITEM, XBID_OUT, XBID_ARMS ("stock stack"), XBID_MODES ("dual mr gp"), XBID_M0 (decode rows,
#   1,2,3,4,5,6,8,12,16,17,24,30,48), XBID_M1 (prefill rows, 49,100,2112,4160).
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
O=${XBID_OUT:-$R/xdsm/bidual}
IMAGE=${XBID_IMAGE:?set XBID_IMAGE}
ITEM=${XBID_ITEM:-dsm-bidual}
ARMS=${XBID_ARMS:-stock stack}; MODES=${XBID_MODES:-dual mr gp}
M0L=${XBID_M0:-1,2,3,4,5,6,8,12,16,17,24,30,48}; M1L=${XBID_M1:-49,100,2112,4160}
case "$M0L$M1L" in *[!0-9,]*) echo "XBID_M0 / XBID_M1: comma lists of row counts"; exit 1 ;; esac
for a in $ARMS; do case $a in stock|stack) ;; *) echo "XBID_ARMS takes stock / stack (got '$a')"; exit 1 ;; esac; done
for md in $MODES; do case $md in dual|mr|gp) ;; *) echo "XBID_MODES takes dual / mr / gp (got '$md')"; exit 1 ;; esac; done
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (dsm-attnfold2.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
armenv() {   # armenv stock|stack: the docker -e words of the arm
  case $1 in
    stock) echo "" ;;
    stack) echo "-e RADIANCE_DS41_GG_GEMV=2 -e RADIANCE_DS41_GG_PF=1 -e RADIANCE_DS41_GG_PF_TABLE=$GPTAB -e RADIANCE_DS41_F32_PF=25" ;;
  esac
}
onearm() {   # onearm ARM MODE: one selftest mode in one process, its verdict lines logged
  local a=$1 md=$2 rc t0=$(date +%s) l flag pat
  case $md in
    dual) flag="--dualcheck $M0L $M1L 20"; pat='^(DUALCHECK|DUALTIME|lever BI-e)' ;;
    mr)   flag="--mrcheck"; pat='^(MRCHECK|  FAIL)' ;;
    gp)   flag="--gpcheck"; pat='^(GPCHECK|  FAIL)' ;;
  esac
  run "gpurun xbidual-$$-$a-$md 2700 $DEV $(armenv $a) --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so $flag > $O/$md-$a.txt 2>&1"
  rc=$?
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xbid $a $(echo "$l" | cut -c1-600)"; done < <(grep -E "$pat" $O/$md-$a.txt)
  log "RESULT xbid $a $md: rc $rc, $(( $(date +%s) - t0 )) s$([ $rc = 0 ] || echo " -- VIRHE (failures or a crash: $(tail -3 $O/$md-$a.txt | tr '\n' ' ' | cut -c1-400))")"
  echo "$a $md $rc" >> $O/rc.txt
}
summary() {   # summary: PASS / FAIL over every arm and mode
  [ "$DRY" = 1 ] && { echo "+ RESULT xbid summary: PASS | FAIL (DUALCHECK identical, MRCHECK / GPCHECK same, rc 0)"; return 0; }
  local nd nx bad
  nd=$(cat $O/dual-*.txt 2>/dev/null | grep -c '^DUALCHECK .*: IDENTICAL')
  nx=$(cat $O/dual-*.txt 2>/dev/null | grep -c '^DUALCHECK .*: DIFF')
  bad=$(awk '$3 != 0' $O/rc.txt 2>/dev/null | tr '\n' ';')
  if [ "$nx" = 0 ] && [ -z "$bad" ] && [ "$nd" -gt 0 ]; then
    log "RESULT xbid summary: PASS -- $nd (case, M1) identical, 0 differ; every mode rc 0 ($(cat $O/rc.txt | tr '\n' ';'))"
  else
    log "RESULT xbid summary: FAIL -- $nd identical, $nx differ; nonzero rc: ${bad:-none}"
  fi
}
log "RESULT xbid $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] modes [$MODES] -> $O"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xbid: VIRHE image $IMAGE not on this host"; exit 1; }
LITS="--dualcheck|DUALCHECK summary|lever BI-e dual check|DUALTIME|--mrcheck|--gpcheck|r4d_gg_dual_pin"
MISS=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'S=/opt/radiance/bin/r4d_selftest; K=/opt/radiance/share/radiance/kernels/libr4d.so; IFS="|"; for l in $0; do grep -qzF -- "$l" $S || echo "MISS $l"; done; for l in gemm_nt_ggml_dual "dual decode H4"; do grep -qzF -- "$l" $K || echo "MISS libr4d:$l"; done; echo CHECKED' "$LITS" 2>&1)
case "$MISS" in CHECKED*) ;; *) log "RESULT xbid: VIRHE $IMAGE's r4d_selftest lacks --dualcheck: $(echo "$MISS" | grep MISS | tr '\n' ';' | cut -c1-400)"; exit 1 ;; esac
HMISS=""
for fn in log run mkd gpurun armenv onearm summary; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker python3 timeout grep awk; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
[ -z "$HMISS" ] || { log "RESULT xbid: VIRHE helpers do not resolve:$HMISS"; exit 1; }
if [ "$DRY" = 1 ]; then
  echo "+ image check: $IMAGE's r4d_selftest carries --dualcheck / --mrcheck / --gpcheck and libr4d's r4d_gg_dual_pin -- OK; helpers resolve -- OK"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses beside one)"
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xbid: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xbid: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
fi
mkd $O
run ": > $O/rc.txt"
for a in $ARMS; do for md in $MODES; do onearm $a $md; done; done
summary
log "RESULT xbid $ITEM: done -- $O"
