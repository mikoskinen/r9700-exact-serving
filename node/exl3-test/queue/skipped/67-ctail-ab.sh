#!/bin/bash
# 72-ctail-ab.sh (queue item, 6.11.2026) -- the compressor tail's chain over the step's bound (radiance-local:ds41-exl3-117-
# <ds41-ctail sha8>-f2f3c3f7-ctail = the mx4l image with deepseek41.so of ds41-ctail 7af8fee; its indexer note names the
# tail mode, checked per arm). At the kv sources with ratio 2 (layers 2, 8, 14) the chain after qsa_work --
# softmax_pool, rmsnorm, rope_table, the index key GEMM, norms,
# ropes, round trips, two entry stores -- ran over m.work = 8200 rows every step, a decode step's one token included.
# Now min(m.work, max(ceil(T / 2) + S, MAXSEQS)) rows (8 at one token, MAXSEQS 8). RADIANCE_DS41_CTAIL=full restores
# the old rows in the same image. Output: the same row by row except where a kernel's form changes with the pass height
# (the index key GEMM at 8 vs 8200 rows), so the identity copy decides.
# Both arms at the fin-best line (MAXSEQS 8, MML 524288), gated (21 ms), the packed formats off:
#   full   RADIANCE_DS41_CTAIL=full: identity copy (~145k), stepprobe, mb c4
#   bound  default: identity copy -> byte comparison with the full arm, stepprobe, mb c4
# Expected (another agent's count, not measured): -0.2..-0.4 ms a decode step.
# Estimate ~30 min (2 gated starts, 2 copies, 2 probes, 2 c4).
# DRYRUN=1 bash 72-ctail-ab.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh) and ds41-fix2-tp4.yaml
# (passes RADIANCE_DS41_CTAIL).
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3CT=${X3CT:-radiance-local:ds41-exl3-117-7af8fee3-f2f3c3f7-ctail}
export IMAGE=$X3CT
log "RESULT xfix 72-ctail: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q RADIANCE_DS41_CTAIL $Y || { log "RESULT xfix 72: VIRHE $Y does not pass RADIANCE_DS41_CTAIL"; exit 1; }
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
best_vars
for arm in full bound; do
  if [ $arm = full ]; then export RADIANCE_DS41_CTAIL=full; else unset RADIANCE_DS41_CTAIL; fi
  gate xct-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 72: VIRHE $arm arm not healthy ($(desc))"; unset RADIANCE_DS41_CTAIL; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 72: VIRHE $arm arm slow twice -> stopped"; down; unset RADIANCE_DS41_CTAIL; exit 1; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xct-$arm-mode.txt"
  [ "$DRY" = 1 ] || grep -q "compressor tail $arm" $O/xct-$arm-mode.txt || {
    log "RESULT xfix 72: VIRHE the $arm arm's serve log does not say 'compressor tail $arm' (see $O/xct-$arm-mode.txt)"
    down; unset RADIANCE_DS41_CTAIL; exit 1; }
  ident xct-$arm
  probe xct-$arm
  mbp xct-$arm c4
  run "docker logs $CN > $O/xct-$arm-engine.log 2>&1"
  stopfile && { down; unset RADIANCE_DS41_CTAIL; exit 0; }
done
unset RADIANCE_DS41_CTAIL
identcmp xct-full xct-bound || log "RESULT xfix 72: the bound changed the greedy output (expected only through the index key GEMM's form) -- judge before shipping"
down
log "RESULT xfix 72-ctail: done | ref $REF_BEST524"
