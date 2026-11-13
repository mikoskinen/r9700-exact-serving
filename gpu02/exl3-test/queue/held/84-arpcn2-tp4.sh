#!/bin/bash
# 84-arpcn2-tp4.sh (queue item, 7.11.2026 night; ONLY AFTER 80 (0 FAIL) and 81 (exact, on vs off)) -- fusion 2 at both
# sites: RADIANCE_DS41_AR_PCN=2 = 81's attention site plus the MoE site: the MoE block's all-reduce and the
# mhc_post_collapse_rmsnorm behind it (the next layer's post + attention collapse + attn norm, or the last layer's +
# the final norm) as one ar_mhc_post_collapse_rmsnorm_4rank collective (the same row 80 tested); the block leaves its
# all-reduce to it (MoeFP8::ar_out, kArOutExact) at the decode/verify steps the fused op takes.
# Image radiance-local:ds41-exl3-117-<ds41-pcnm sha8>-f2f3c3f7-arpcn2 = the arpcn image with deepseek41.so of ds41-pcnm
# (libr4d.so is pcnw's: no kernel change).
# Expected: 81's gain at the attention site again at up to 42 MoE sites (not behind an Engram layer) + the final one (a count, not measured).
# Arms (same image, fin-best line, MAXSEQS 8, MML 524288, 2.25 bpw .rad, gated 21 ms):
#   off   switch unset: identity copy, greedyset (20 greedy prompts) c1 + c4 twice, stepprobe, mb c4
#   both  RADIANCE_DS41_AR_PCN=2: the same (c4 once); against off byte for byte: copy, c1 answers, c4 answers (where
#         off's two c4 runs agree). 81's on arm (=1) is the attention site alone: not re-run here.
# The indexer note must say "attention ... fused at N layers" and "MoE all-reduce + next collapse-norm fused at M
# layers" (both >= 1) for both, "separate at 0" twice for off; else VIRHE + down.
# Hang protection as 81: the only wait is the one-shot handshake (R4D_AR_SPIN_MAX, then a trap); /health after the gate
# and every step; a dead server ends the item with logs saved and the compose DOWN.
# Estimate ~35 min. DRYRUN=1 bash 84-arpcn2-tp4.sh : prints, writes nothing.
# Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh), ds41-fix2-tp4.yaml (passes RADIANCE_DS41_AR_PCN) and tools/greedyset.py.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3AP=${X3A2:-radiance-local:ds41-exl3-117-00a831c7-f2f3c3f7-arpcn2}
export IMAGE=$X3AP
ARMS=${ARMS:-"off both"}
FUSEBASE=${FUSEBASE:-on}
GS=$QD/tools/greedyset.py; [ -e $GS ] || GS=$(cd "$(dirname "$0")" && pwd)/tools/greedyset.py
log "RESULT xfix 84-arpcn2: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] router/mix fusions $FUSEBASE compose $Y"
imgcheck $IMAGE
if [ "$DRY" != 1 ]; then
  grep -q "^ *RADIANCE_DS41_AR_PCN:" $Y || { log "RESULT xfix 84: VIRHE $Y does not pass RADIANCE_DS41_AR_PCN"; exit 1; }
  [ -e $GS ] || { log "RESULT xfix 84: VIRHE tools/greedyset.py missing"; exit 1; }
fi
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE RADIANCE_DS41_AR_PCN
[ "$FUSEBASE" = off ] && export RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0
best_vars
quit() { run "docker logs $CN > $O/xa2-$1-engine.log 2>&1"; down; unset RADIANCE_DS41_AR_PCN RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit ${2:-1}; }
# alive ARM STEP: the server still answers /health and its container runs; else the item ends here
alive() {
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200 (else: logs, down, exit)"; return 0; }
  local st code
  st=$(docker inspect -f '{{.State.Status}} exit {{.State.ExitCode}}' $CN 2>/dev/null || echo gone)
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT xfix 84: VIRHE the $1 arm's server is not healthy after $2 (container: $st, /health $code; last E lines: $(docker logs $CN 2>&1 | grep -E '^E |trap|abort|HSA_STATUS' | tail -3 | tr '\n' ' ' | cut -c1-500)) -> logs saved, down, stopped"
  quit $1
}
# gset TAG CONC N: the 20 greedy prompts at concurrency CONC into $O/TAG-gs-N.jsonl
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-$3.jsonl --conc $2 --max-tokens 192 > $O/$1-gs-$3.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "+ greedyset --cmp $1 $2" >&2; echo IDENTICAL; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
for arm in $ARMS; do
  case $arm in
    off) unset RADIANCE_DS41_AR_PCN; want=separate ;;
    both) export RADIANCE_DS41_AR_PCN=2; want=fused ;;
    *)   log "RESULT xfix 84: unknown arm $arm"; continue ;;
  esac
  gate xa2-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 84: VIRHE $arm arm not healthy ($(desc))"; quit $arm; }
  alive $arm gate
  [ $rc = 2 ] && { log "RESULT xfix 84: VIRHE $arm arm slow twice -> stopped"; quit $arm; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xa2-$arm-mode.txt"
  if [ "$DRY" != 1 ]; then
    got=$(grep -o 'post-collapse-norm [a-z]* at [0-9]* layers' $O/xa2-$arm-mode.txt)
    gm=$(grep -o 'next collapse-norm [a-z]* at [0-9]* layers' $O/xa2-$arm-mode.txt); nm=$(echo "$gm" | awk '{print $5}')
    ev=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_DS41_AR_PCN=' | cut -d= -f2- || true)
    n=$(echo "$got" | awk '{print $4}')
    if [ -z "$got" ] || [ -z "$gm" ] || ! echo "$got" | grep -q "norm $want " || ! echo "$gm" | grep -q "norm $want " \
       || { [ $want = fused ] && { [ "${n:-0}" -lt 1 ] || [ "${nm:-0}" -lt 1 ]; }; } || { [ $want = separate ] && { [ "${n:-1}" != 0 ] || [ "${nm:-1}" != 0 ]; }; }; then
      log "RESULT xfix 84: VIRHE the $arm arm runs '$got' / '$gm' (env '${ev}'; want $want) -> stopped"
      quit $arm
    fi
    log "RESULT xfix 84 $arm: attention $got, MoE $gm (env RADIANCE_DS41_AR_PCN='${ev}')"
  fi
  ident xa2-$arm;   alive $arm "the identity copy"
  gset xa2-$arm 1 c1; alive $arm "greedyset c1"
  gset xa2-$arm 4 c4; alive $arm "greedyset c4"
  [ $arm = off ] && { gset xa2-$arm 4 c4b; alive $arm "greedyset c4 (second)"; }
  probe xa2-$arm;   alive $arm "the stepprobe"
  mbp xa2-$arm c4;  alive $arm "mb c4"
  run "docker logs $CN > $O/xa2-$arm-engine.log 2>&1"
  stopfile && quit $arm 0
done
unset RADIANCE_DS41_AR_PCN
case " $ARMS " in *" off "*) case " $ARMS " in *" both "*)
  identcmp xa2-off xa2-both || log "RESULT xfix 84: VIRHE the fused collective changed the greedy output (must be exact)"
  c1=$(gcmp $O/xa2-off-gs-c1.jsonl $O/xa2-both-gs-c1.jsonl); c4=$(gcmp $O/xa2-off-gs-c4.jsonl $O/xa2-both-gs-c4.jsonl)
  fl=$(gcmp $O/xa2-off-gs-c4.jsonl $O/xa2-off-gs-c4b.jsonl)
  log "RESULT xfix 84 both vs off: greedy c1 $c1 | c4 $c4 (off's own c4 floor: $fl)"
  case "$c1" in *DIFFERENT*) log "RESULT xfix 84: VIRHE the fused collective changed greedy answers (must be exact)" ;; esac
  case "$fl" in IDENTICAL*) case "$c4" in *DIFFERENT*) log "RESULT xfix 84: VIRHE the fused collective changed the c4 answers (off's two c4 runs agree)" ;; esac ;; esac ;; esac ;; esac
down
unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE
log "RESULT xfix 84-arpcn2: done | ref $REF_BEST524"
