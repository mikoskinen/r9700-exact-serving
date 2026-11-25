#!/bin/bash
# 82-idx-bench-t3.sh (queue item, 7.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN; ~2-3 min) -- the
# indexer kernels of the tuned3 libr4d checked on the card at real contexts: 62's step 0 (idx2/idx2b: index_score with keep
# and index_topk(keep) against the unrestricted kernels) and 67's step 0 (MXFP4 index keys: the packed cache against bf16),
# both on the libr4d the tuned3 image ships (ds41-t3ag 390664b = tuned2's libr4d byte for byte + moe_gather_all_reduce).
# ds41idx_bench (queue-r2 src/ds41idx_bench.cpp, md5 c5108fe5) dlopens the image's libr4d, builds one sequence's index cache
# at ctx, runs both forms and compares on the device: a kept, passed position's score bit-identical, every other restricted
# score -inf, the two selections identical (--mx4 1: every packed-key score bit-identical to bf16). MISMATCH = VIRHE, exit 1.
#   62 cases: --ctx 16384 / 135000 / 322000 / 524000 --rows 256 (the prefill band, WMMA); --ctx 135000 / 322000 / 524000
#             --rows 1 (the decode row); --ctx 322000 --rows 4
#   67 cases: --mx4 1 at --ctx 135000 / 322000 / 524000, --rows 256 and 1
# References (11, -idx image, 6.11.): restricted index_score 256 rows 0.91 / 2.07 / 2.91 / 3.72 ms, index_topk(keep) 0.21 /
# 0.82 / 1.74 / 2.74 ms at 16k / 135k / 322k / 524k; decode row restricted 233.9 us (135k), 163.7 (322k), topk 25.9 / 65.1 us.
# Every run bounded (timeout -k 15 600, --init, named, docker rm -f). Guards: no radiance container, KFD idle (60 s), the
# bench binary (md5 c5108fe5) and the image present.
# DRYRUN=1 bash 82-idx-bench-t3.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${IDXB_BIN:-}
for c in "$Q/bin" "$HOME/gpu/ds41/exl3-test/queue/done/bin" "$HOME/gpu/ds41/exl3-test/queue/bin"; do
  [ -z "$B" ] && [ -x "$c/ds41idx_bench" ] && B=$(cd "$c" && pwd)
done
B=${B:-$HOME/gpu/ds41/exl3-test/queue/done/bin}
O=${IDXB_OUT:-$HOME/gpu/ds41/exl3-test/runs/idxb-t3}
L=${IDXB_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${IDXB_IMAGE:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab}
CARD=${CARD:-0}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
bench() {   # bench NAME ARGS...
  local n=$1; shift
  local cn=idxb-$$-$RANDOM
  run "timeout -k 15 600 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint /b/ds41idx_bench $IMAGE $* > $O/$n.txt 2>&1; docker rm -f $cn > /dev/null 2>&1"
  [ "$DRY" = 1 ] || log "$(grep -m1 -E '^RESULT ds41idx' $O/$n.txt | sed 's/^RESULT ds41idx/RESULT idxb-t3 ds41idx/' || echo "RESULT idxb-t3 $n: VIRHE $(tail -2 $O/$n.txt | tr '\n' ' ' | cut -c1-300)")"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT idxb-t3: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT idxb-t3: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/ds41idx_bench" ] || { log "RESULT idxb-t3: VIRHE $B/ds41idx_bench missing"; exit 1; }
  [ "$(md5sum "$B/ds41idx_bench" | cut -c1-8)" = c5108fe5 ] || { log "RESULT idxb-t3: VIRHE $B/ds41idx_bench is not md5 c5108fe5 (the --mx4 build)"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT idxb-t3: VIRHE image $IMAGE missing"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT idxb-t3: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD bench $B/ds41idx_bench -> $O"
for c in 16384 135000 322000 524000; do bench wmma-$c --ctx $c --rows 256; done
for c in 135000 322000 524000; do bench dec-$c --ctx $c --rows 1; done
bench dec4-322000 --ctx 322000 --rows 4
for c in 135000 322000 524000; do for r in 256 1; do bench mx4-c$c-r$r --ctx $c --rows $r --mx4 1; done; done
[ "$DRY" = 1 ] && { log "RESULT idxb-t3: dry run done"; exit 0; }
n=$(ls $O/*.txt | wc -l); bad=$(grep -l MISMATCH $O/*.txt 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
nores=$(for f in $O/*.txt; do grep -q '^RESULT ds41idx' $f || basename $f; done | tr '\n' ' ')
if [ -n "$bad" ]; then log "RESULT idxb-t3: VIRHE MISMATCH on the card in: $bad"; exit 1; fi
[ -n "$nores" ] && log "RESULT idxb-t3: VIRHE no RESULT line in: $nores"
log "RESULT idxb-t3: done -- $n cases, no MISMATCH${nores:+ (missing: $nores)} | refs 11 (-idx): restricted index_score 256 rows 0.91 / 2.07 / 2.91 / 3.72 ms, topk(keep) 0.21 / 0.82 / 1.74 / 2.74 ms at 16k / 135k / 322k / 524k; decode row 233.9 us (135k), 163.7 (322k)"
