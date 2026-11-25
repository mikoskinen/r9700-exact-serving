#!/bin/bash
# 62-fix-idx2.sh (queue item, 6.11.2026) -- the indexer's second step on TP4: radiance-local:ds41-exl3-117-21a69ead-f2f3c3f7-idx2
# = the -idx image (engine ds41-idx b4624774, exl3 f2f3c3f7) with libr4d.so from branch ds41-idx2 21a69ead:
#   - index_topk with keep walks only the kept blocks (an LDS list of up to 4096 a row; the same candidates in the same
#     order, so the same selection); its decode form (<= 16 rows) deals the kept blocks over its chunks;
#   - the decode band's index_score with keep scores the row's kept positions compacted (16384 at DeepSeek's 2048 x 8),
#     not every 32-position wave that holds one; a kept entry is the unrestricted kernel's code (same bits).
# deepseek41.so is unchanged (RADIANCE_DS41_INDEX as in -idx: keep,split by default).
#   0. one card, before any start (skipped if a radiance container runs): ds41idx_bench on the idx2 libr4d (11's cases):
#      EXACT/MISMATCH + index_score / index_topk(keep) times against 11's -idx numbers. A MISMATCH stops the item.
#   1. the 47 flags (MML 524288 MAXSEQS 4 NST 0 ckint 8192), gated (24 ms): the identity copy (~145k) -> byte comparison
#      with 50's off arm (runs/xfix/copy-xfix-idx-off-run1.txt; must be IDENTICAL), the cold prefill set, onem.py
#      135000,262144 (the same order as 50's on arm: gate, ident, cold, deep), stepprobe
#   2. the fin-best line at MML 524288 (lib-117 best_vars), gated (21 ms): stepprobe, mb c4 / c8
# References: -idx (50, coordinator 6.11.): cold 16k 2322.0 / 135k 2381.9 / 322k 2251.9 tok/s; deep 135k prefill 2493.2,
# decode 35.18 tok/s 28.8 ms/step; 266k prefill 2404.9, decode 33.82 tok/s 30.4 ms/step; fin-best guard 19.72 ms,
# essay 47.7 / code 50.6 tok/s, c4 95.5, c8 123.4 tok/s. Base 6.11. (abf7f9a0): cold 2402.2 / 2002.2 / 1522.3; deep off
# 135k 35.9 / 266k 34.0 tok/s.
# Expected (arithmetic, not measured): prefill +1-3 % at 322k (index_topk(keep) 1.74 -> ~0.2 ms a 256-row run, 4 layers,
# 8 runs a rank a chunk); decode at 266k -0.5..-1 ms/step IF the keep layers' index_score is on the critical path (item
# 64 measures that); short context unchanged.
# Estimate ~70 min (bench 2, 2 gated starts ~5 each, ident ~2, cold ~5, onem ~6, probe 1, mb c4/c8 ~6).
# DRYRUN=1 bash 62-fix-idx2.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh, ds41-fix-tp4.yaml),
# bin/ds41idx_bench (queue-fix's), and 50's runs/xfix/copy-xfix-idx-off-run1.txt.
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
X3IDX2=${X3IDX2:-radiance-local:ds41-exl3-117-528e5e06-f2f3c3f7-idx2b}   # 6.11. 19.00: idx2b
export IMAGE=$X3IDX2
QB=$(cd "$(dirname "$0")" && pwd)/bin; [ -x $QB/ds41idx_bench ] || QB=$QD/bin
REF_IDX="-idx 6.11.: cold 16k 2322.0 / 135k 2381.9 / 322k 2251.9 tok/s; deep 135k 2493.2 pf, 35.18 tok/s 28.8 ms; 266k 2404.9 pf, 33.82 tok/s 30.4 ms; fin-best 19.72 ms, essay 47.7 / code 50.6, c4 95.5, c8 123.4"
log "RESULT xfix 62-idx2: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))"
imgcheck $IMAGE
mkd $O $STATSDIR $O/idx2-micro
# 0. one card: the bench on the idx2 libr4d
if [ "$DRY" != 1 ] && docker ps --format '{{.Names}}' | grep -qi radiance; then
  log "RESULT xfix 62 bench: skipped (a radiance container runs)"
else
  [ "$DRY" = 1 ] || for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  bench2() {   # bench2 NAME ARGS...
    local n=$1; shift
    run "gpurun idxb-$$-$RANDOM 900 --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0" \
        "-v $QB:/b:ro --entrypoint /b/ds41idx_bench $IMAGE $* > $O/idx2-micro/$n.txt 2>&1"
    [ "$DRY" = 1 ] || log "$(grep -m1 '^RESULT ds41idx' $O/idx2-micro/$n.txt | sed 's/^RESULT ds41idx/RESULT ds41idx2/' || echo "RESULT ds41idx2 $n: VIRHE $(tail -2 $O/idx2-micro/$n.txt | tr '\n' ' ')")"
  }
  if [ "$DRY" = 1 ] || [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ]; then
    for c in 16384 135000 322000 524000; do bench2 wmma-$c --ctx $c --rows 256; done
    for c in 135000 322000 524000; do bench2 dec-$c --ctx $c --rows 1; done
    bench2 dec4-322000 --ctx 322000 --rows 4
    if [ "$DRY" != 1 ] && grep -q MISMATCH $O/idx2-micro/*.txt; then
      log "RESULT xfix 62: VIRHE ds41idx2 MISMATCH on the card -> stopped"; exit 1
    fi
    log "RESULT xfix 62 bench refs (11, -idx, 6.11.): restricted index_score 256 rows 0.91 / 2.07 / 2.91 / 3.72 ms, index_topk(keep) 0.21 / 0.82 / 1.74 / 2.74 ms at 16k/135k/322k/524k; decode row restricted 233.9 us (135k), 163.7 (322k), topk 25.9 / 65.1 us"
  else
    log "RESULT xfix 62 bench: skipped (a KFD client after 60 s)"
  fi
fi
stopfile && exit 0
# 1. the 47 flags
pfflags
gate xfix-idx2 24; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 62: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 62: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
run "docker logs $CN 2>&1 | grep -m2 'deepseek41 indexer' > $O/xfix-idx2-mode.txt"
[ "$DRY" = 1 ] || log "RESULT xfix 62 mode: $(cat $O/xfix-idx2-mode.txt | tr '\n' ' ')"
ident xfix-idx2
if [ "$DRY" = 1 ] || [ -e $O/copy-xfix-idx-off-run1.txt ]; then
  identcmp xfix-idx-off xfix-idx2 || { log "RESULT xfix 62: VIRHE idx2 changed the output -> stopped"; run "docker logs $CN > $O/xfix-idx2-engine.log 2>&1"; down; exit 1; }
else
  log "RESULT xfix 62: no 50 off-arm copy to compare with ($O/copy-xfix-idx-off-run1.txt) -- identity not checked"
fi
cold xfix-idx2
stopfile && { down; exit 0; }
deep xfix-idx2 135000,262144
probe xfix-idx2
run "docker logs $CN > $O/xfix-idx2-engine.log 2>&1"
log "RESULT xfix 62 47-flags done | ref $REF_IDX"
stopfile && { down; exit 0; }
# 2. the fin-best line at MML 524288
best_vars
gate xfix-idx2-best524 21; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 62 best: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 62 best: VIRHE slow state after one restart -> skipped"; down; exit 1; }
probe xfix-idx2-best524
for p in c4 c8; do stopfile && break; mbp xfix-idx2-best524 $p; done
run "docker logs $CN > $O/xfix-idx2-best524-engine.log 2>&1"
log "RESULT xfix 62 best: done | ref $REF_BEST524 | $REF_IDX"
down
log "RESULT xfix 62-idx2: done"
