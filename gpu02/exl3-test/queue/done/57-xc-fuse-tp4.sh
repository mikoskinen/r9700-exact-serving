#!/bin/bash
# 77-fuse-tp4.sh (queue item, 6.11.2026 night) -- two exact dispatch fusions of the decode chain, each behind its own
# off-switch, in ONE image (radiance-local:ds41-exl3-117-<ds41-rsc sha8>-f2f3c3f7-fuse = the ctail image with libr4d.so
# and deepseek41.so of ds41-rsc 1fb143a, on ctail 7af8fee):
#   router  router_scored_topk + moe_scatter (+ the ungated shared gate's fill) as router_scored_topk_scatter: one
#           ordered dispatch fewer a MoE layer (43 a pass)          off: RADIANCE_DS41_ROUTER_FUSE=0
#   mix     mhc_mix's finish in the split launch's last workgroup a row group: one fewer a mix (80 a pass)
#                                                                    off: R4D_DS41_MIX_FUSE=0
# Expected from gapbench (75: an unstamped ordered dispatch ~1.5 us of its own beside the kernel) plus the removed
# kernels' time: router ~3 us x 43, mix ~3.5 us x 80 -> ~0.4 ms a step together (a count, not measured).
# Arms (same image, fin-best line, MAXSEQS 8, MML 524288, 2.25 bpw .rad, gated 21 ms):
#   off     both off: identity copy, stepprobe, mb c4
#   on      both on (default): the same, its copy against off's byte for byte
#   router  the router fusion only: the same (mix's share = on - router), its copy against off's
# The indexer note names the router mode ("router selection+sort fused|separate") and the container env the mix
# switch (docker inspect); a mismatch stops the item.
# Estimate ~45 min. ARMS="off on" skips the attribution arm. DRYRUN=1 bash 77-fuse-tp4.sh : prints, writes nothing.
# Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh) and ds41-fix2-tp4.yaml (passes RADIANCE_DS41_ROUTER_FUSE, R4D_DS41_MIX_FUSE).
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3FU=${X3FU:-radiance-local:ds41-exl3-117-1fb143a5-f2f3c3f7-fuse}
export IMAGE=$X3FU
ARMS=${ARMS:-"off on router"}
log "RESULT xfix 77-fuse: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] compose $Y"
imgcheck $IMAGE
if [ "$DRY" != 1 ]; then
  for v in RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; do
    grep -q "^ *$v:" $Y || { log "RESULT xfix 77: VIRHE $Y does not pass $v"; exit 1; }
  done
fi
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE
best_vars
for arm in $ARMS; do
  case $arm in
    off)    export RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0; want_r=separate; want_m=0 ;;
    on)     unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; want_r=fused; want_m= ;;
    router) unset RADIANCE_DS41_ROUTER_FUSE; export R4D_DS41_MIX_FUSE=0; want_r=fused; want_m=0 ;;
    *)      log "RESULT xfix 77: unknown arm $arm"; continue ;;
  esac
  gate xfu-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 77: VIRHE $arm arm not healthy ($(desc))"; unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 77: VIRHE $arm arm slow twice -> stopped"; down; unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit 1; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xfu-$arm-mode.txt"
  if [ "$DRY" != 1 ]; then
    mv_=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^R4D_DS41_MIX_FUSE=' | cut -d= -f2- || true)
    if ! grep -q "router selection+sort $want_r" $O/xfu-$arm-mode.txt || [ "${mv_}" != "$want_m" ]; then
      log "RESULT xfix 77: VIRHE the $arm arm runs router '$(grep -o 'router selection+sort [a-z]*' $O/xfu-$arm-mode.txt)', mix env '${mv_}' (want $want_r / '$want_m') -> stopped"
      down; unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit 1
    fi
  fi
  ident xfu-$arm
  probe xfu-$arm
  mbp xfu-$arm c4
  run "docker logs $CN > $O/xfu-$arm-engine.log 2>&1"
  stopfile && { down; unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit 0; }
done
unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE
for a in on router; do
  case " $ARMS " in *" off "*) case " $ARMS " in *" $a "*) identcmp xfu-off xfu-$a || log "RESULT xfix 77: VIRHE the $a arm changed the greedy output (must be exact)" ;; esac ;; esac
done
down
log "RESULT xfix 77-fuse: done | ref $REF_BEST524"
