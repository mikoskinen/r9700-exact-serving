#!/bin/bash
# 76-fz1-tp4.sh (queue item, 6.11.2026 night) -- gate_up's prep and GEMM as ONE launch in the engine (exl3r4d of exl3-117-fz1
# e2ee477 = fa0e958 + D18 at every K, its K loop pinned: no 0-wait) against the two dispatches, in ONE image
# (RADIANCE_EXL3_FUSE1=0 keeps the two). The played tape waits ~3.76 us before every ordered dispatch (AQL stamps), and
# gate_up's prep is one of them a MoE layer (prep_row 4.94 us + 3.7..6.3 us gap): -0.15..-0.3 ms a step expected.
# Image radiance-local:ds41-exl3-117-7af8fee3-e2ee4775-tuned2-h7e342763 (b42ff7e2d98b): the tuned2 gu25c image (ctail deepseek41.so, mx4l libs, the
# 7e342763 heat binary) with exl3r4d.so of exl3-117-fz1; every other env unset (the pieces' defaults: indexer
# keep,split, bf16 keys / latent, compressor tail bound, heat off).
# Both arms at the fin-best line (MAXSEQS 8, MML 524288), gated (21 ms), on the 2.25 bpw .rad (K2 / K2.5 experts):
#   two   RADIANCE_EXL3_FUSE1=0: identity copy (~145k), stepprobe, mb c4
#   one   default: identity copy -> byte comparison with two, stepprobe, mb c4
# The container env is checked after each gate (docker inspect), so a yaml without the switch cannot run two equal arms.
# Estimate ~30 min. DRYRUN=1 bash 76-fz1-tp4.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh)
# and ds41-fix2-tp4.yaml (passes RADIANCE_EXL3_FUSE1).
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3FZ=${X3FZ:-radiance-local:ds41-exl3-117-7af8fee3-e2ee4775-tuned2-h7e342763}
export IMAGE=$X3FZ
log "RESULT xfix 76-fz1: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q RADIANCE_EXL3_FUSE1 $Y || { log "RESULT xfix 76: VIRHE $Y does not pass RADIANCE_EXL3_FUSE1"; exit 1; }
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_EXL3_FUSE1 \
      RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN
best_vars
for arm in two one; do
  if [ $arm = two ]; then export RADIANCE_EXL3_FUSE1=0; else unset RADIANCE_EXL3_FUSE1; fi
  gate xfz-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 76: VIRHE $arm arm not healthy ($(desc))"; unset RADIANCE_EXL3_FUSE1; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 76: VIRHE $arm arm slow twice -> stopped"; down; unset RADIANCE_EXL3_FUSE1; exit 1; }
  if [ "$DRY" != 1 ]; then
    ev=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_EXL3_FUSE1=' || true)
    { [ $arm = two ] && [ "$ev" = "RADIANCE_EXL3_FUSE1=0" ]; } || { [ $arm = one ] && [ -z "$ev" -o "$ev" = "RADIANCE_EXL3_FUSE1=" ]; } || {
      log "RESULT xfix 76: VIRHE the $arm arm's container env says '${ev:-no RADIANCE_EXL3_FUSE1}' -> stopped"; down; unset RADIANCE_EXL3_FUSE1; exit 1; }
  fi
  ident xfz-$arm
  probe xfz-$arm
  mbp xfz-$arm c4
  run "docker logs $CN > $O/xfz-$arm-engine.log 2>&1"
  stopfile && { down; unset RADIANCE_EXL3_FUSE1; exit 0; }
done
unset RADIANCE_EXL3_FUSE1
identcmp xfz-two xfz-one || log "RESULT xfix 76: VIRHE the single launch changed the greedy output (must be exact)"
down
log "RESULT xfix 76-fz1: done | ref $REF_BEST524"
