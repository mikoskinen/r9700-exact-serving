#!/bin/bash
# 67-mx4-keys.sh (queue item, 6.11.2026) -- MXFP4 index keys on TP4: radiance-local:ds41-exl3-117-<ds41-mx4 sha8>-f2f3c3f7-mx4
# = the idx2b engine + RADIANCE_DS41_IDXKEY=mxfp4: the index caches (layers 2/8/14 at ratio 2, 20 at ratio 1; layers
# 24-36 read layer 20's) hold each 128-value key as 80 bytes (64 B e2m1 codes, 4 ue8m0 scales, pad) instead of 256 B
# bf16. Exact: the keys are quant_roundtrip(fp4_e2m1, ue8m0, 32) values, which the packing reproduces bit for bit
# (docs/OPS.md "MXFP4 index keys"; emulator: every index_score form packed vs bf16 bit-identical). Unset = bf16 keys
# (idx2b's behaviour). Expected (arithmetic): index caches 640 -> 200 B a token (2.5 entries a token, every card), at
# MAXSEQS 8 x MML 524288 ~1.8 GiB a card less KV (lent to the experts while idle), index_score +-5 %.
#   0. one card, before any start (skipped if a radiance container runs): ds41idx_bench --mx4 1 at ctx 135k / 322k /
#      524k, 256 rows and 1 row: packed vs bf16 keys EXACT/MISMATCH on the card + index_score times. MISMATCH stops.
#   1. RADIANCE_DS41_IDXKEY=mxfp4, the 47 flags (MML 524288 MAXSEQS 4 NST 0 ckint 8192), gated (24 ms): mode line, the
#      startup "kv cache" line, the identity copy -> byte comparison with 50's bf16 off arm (must be IDENTICAL), the cold
#      set, onem 135000,262144, stepprobe
#   2. the same at the fin-best line (MAXSEQS 8, MML 524288), gated (21 ms): the "kv cache" line (vs 62's bf16 start log
#      at the same flags, runs/xfix/start-xfix-idx2-best524.log), stepprobe, mb c4 / c8
# References: 62 (idx2 / idx2b, bf16 keys) when it has run; else -idx (50): cold 2322.0 / 2381.9 / 2251.9, deep 135k
# 35.18 tok/s, 266k 33.82; fin-best 19.72 ms, essay 47.7 / code 50.6, c4 95.5, c8 123.4.
# Estimate ~60 min (bench 3, 2 gated starts, ident 2, cold 5, onem 6, probe 2, mb c4/c8 6).
# DRYRUN=1 bash 67-mx4-keys.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh), ds41-fix2-tp4.yaml
# (passes RADIANCE_DS41_IDXKEY), bin/ds41idx_bench (queue-r2's, with --mx4), 50's runs/xfix/copy-xfix-idx-off-run1.txt.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
# gpurun NAME SECONDS docker-run-args...: a bounded, named container, removed even if its kernel hangs (timeout -k sends
# SIGKILL to the CLI after 15 s; --init so PID 1 takes SIGTERM; docker rm -f ends the container whatever the CLI did).
gpurun() {
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3MX4=${X3MX4:-radiance-local:ds41-exl3-117-8505d5ff-f2f3c3f7-mx4}
export IMAGE=$X3MX4
QB=$(cd "$(dirname "$0")" && pwd)/bin; [ -x $QB/ds41idx_bench ] || QB=$QD/bin
kvline() { grep -m1 -E '^I +kv cache +[0-9.]+ GiB a card' "$1" 2>/dev/null | sed 's/^I *//' | tr -s ' '; }
log "RESULT xfix 67-mx4: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q RADIANCE_DS41_IDXKEY $Y || { log "RESULT xfix 67: VIRHE $Y does not pass RADIANCE_DS41_IDXKEY"; exit 1; }
mkd $O $STATSDIR $O/mx4-micro
# 0. one card
if [ "$DRY" != 1 ] && docker ps --format '{{.Names}}' | grep -qi radiance; then
  log "RESULT xfix 67 bench: skipped (a radiance container runs)"
else
  [ "$DRY" = 1 ] || for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  for cr in "135000 256" "322000 256" "524000 256" "135000 1" "322000 1" "524000 1"; do
    set -- $cr
    run "gpurun mx4b-$$-$RANDOM 900 --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0" \
        "-v $QB:/b:ro --entrypoint /b/ds41idx_bench $IMAGE --ctx $1 --rows $2 --mx4 1 > $O/mx4-micro/c$1-r$2.txt 2>&1"
    [ "$DRY" = 1 ] || log "$(grep -m1 '^RESULT ds41idx-mx4' $O/mx4-micro/c$1-r$2.txt || echo "RESULT ds41idx-mx4 c$1 r$2: VIRHE $(tail -2 $O/mx4-micro/c$1-r$2.txt | tr '\n' ' ')")"
  done
  if [ "$DRY" != 1 ] && grep -q MISMATCH $O/mx4-micro/*.txt; then log "RESULT xfix 67: VIRHE mxfp4 keys MISMATCH on the card -> stopped"; exit 1; fi
fi
stopfile && exit 0
export RADIANCE_DS41_IDXKEY=mxfp4
# 1. the 47 flags
pfflags
gate xfix-mx4 24; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 67: VIRHE not healthy ($(desc)) -- see start-xfix-mx4.log"; unset RADIANCE_DS41_IDXKEY; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 67: VIRHE slow state after one restart -> skipped"; down; unset RADIANCE_DS41_IDXKEY; exit 1; }
run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xfix-mx4-mode.txt"
[ "$DRY" = 1 ] || log "RESULT xfix 67 mode: $(cut -c1-260 $O/xfix-mx4-mode.txt) | $(kvline $O/start-xfix-mx4.log)"
ident xfix-mx4
if [ "$DRY" = 1 ] || [ -e $O/copy-xfix-idx-off-run1.txt ]; then
  identcmp xfix-idx-off xfix-mx4 || { log "RESULT xfix 67: VIRHE mxfp4 keys changed the output -> stopped"; run "docker logs $CN > $O/xfix-mx4-engine.log 2>&1"; down; unset RADIANCE_DS41_IDXKEY; exit 1; }
else
  log "RESULT xfix 67: no 50 off-arm copy ($O/copy-xfix-idx-off-run1.txt) -- identity not checked"
fi
cold xfix-mx4
stopfile && { down; unset RADIANCE_DS41_IDXKEY; exit 0; }
deep xfix-mx4 135000,262144
probe xfix-mx4
run "docker logs $CN > $O/xfix-mx4-engine.log 2>&1"
stopfile && { down; unset RADIANCE_DS41_IDXKEY; exit 0; }
# 2. fin-best
best_vars
gate xfix-mx4-best524 21; rc=$?
if [ $rc = 0 ]; then
  [ "$DRY" = 1 ] || log "RESULT xfix 67 kv at fin-best (MAXSEQS 8, MML 524288): mxfp4 $(kvline $O/start-xfix-mx4-best524.log) | bf16 (62) $(kvline $O/start-xfix-idx2-best524.log)"
  probe xfix-mx4-best524
  for p in c4 c8; do stopfile && break; mbp xfix-mx4-best524 $p; done
  run "docker logs $CN > $O/xfix-mx4-best524-engine.log 2>&1"
else
  log "RESULT xfix 67 best: skipped (gate rc $rc)"
fi
unset RADIANCE_DS41_IDXKEY
down
log "RESULT xfix 67-mx4: done"
