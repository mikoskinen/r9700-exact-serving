#!/bin/bash
# 81-arpcn-tp4.sh (queue item, 6.11.2026 night; AFTER 80-arpcn-selftest passed: 0 FAIL, no stop) -- fusion 2 in the
# engine: the attention all-reduce + mhc_post_collapse_rmsnorm (mhc_post -> collapse -> ffn rmsnorm) as ONE exact
# four-rank collective (row ar_mhc_post_collapse_rmsnorm_4rank, libr4d of ds41-pcnw ce26b61), behind its off-switch:
#   RADIANCE_DS41_AR_PCN=1  the fused collective at every layer that has both (decode steps, T <= max_seqs*(1+spec))
#   unset (default)          the pair as before (all-reduce, then mhc_post_collapse_rmsnorm)
# Image radiance-local:ds41-exl3-117-<ds41-pcnw sha8>-f2f3c3f7-arpcn = the fuse image (77) with libr4d.so and
# deepseek41.so of ds41-pcnw; router + mix fusions stay at the image default in BOTH arms (FUSEBASE=off sets both to 0
# in both arms, if 77 found them wrong).
# Expected: one ordered dispatch fewer an attention site (~1.5 us own cost, 75) plus the norm kernel's time minus the
# epilogue's time in the collective; ~3-5 us x 43 layers -> ~0.15-0.2 ms a step (a count, not measured).
# Arms (same image, fin-best line, MAXSEQS 8, MML 524288, 2.25 bpw .rad, gated 21 ms):
#   off   switch unset: identity copy, tools/greedyset.py (20 fixed greedy prompts) at c1 and twice at c4 (the c4 batch's
#         own run-to-run floor), stepprobe, mb c4
#   on    RADIANCE_DS41_AR_PCN=1: the same (greedyset c4 once); against off byte for byte: the copy, the 20 c1 answers,
#         the 20 c4 answers (counted where off's two c4 runs agree)
# The indexer note names the mode and the count ("attention all-reduce + post-collapse-norm fused at N layers" /
# "separate at 0 layers"); a mismatch stops the item.
# Hang protection: the collective's only wait is the one-shot handshake (R4D_AR_SPIN_MAX, then a trap = the server
# exits); every client step has lib-fix's timeout; after the gate and after every step the server's /health is checked
# and a dead or hung server ends the item with its log saved and the compose DOWN (no further arm).
# Estimate ~30 min. DRYRUN=1 bash 81-arpcn-tp4.sh : prints, writes nothing.
# Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh), ds41-fix2-tp4.yaml (passes RADIANCE_DS41_AR_PCN) and tools/greedyset.py.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3AP=${X3AP:-radiance-local:ds41-exl3-117-ce26b611-f2f3c3f7-arpcn}
export IMAGE=$X3AP
ARMS=${ARMS:-"off on"}
FUSEBASE=${FUSEBASE:-on}
GS=$QD/tools/greedyset.py; [ -e $GS ] || GS=$(cd "$(dirname "$0")" && pwd)/tools/greedyset.py
log "RESULT xfix 81-arpcn: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] router/mix fusions $FUSEBASE compose $Y"
imgcheck $IMAGE
if [ "$DRY" != 1 ]; then
  grep -q "^ *RADIANCE_DS41_AR_PCN:" $Y || { log "RESULT xfix 81: VIRHE $Y does not pass RADIANCE_DS41_AR_PCN"; exit 1; }
  [ -e $GS ] || { log "RESULT xfix 81: VIRHE tools/greedyset.py missing"; exit 1; }
fi
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE RADIANCE_DS41_AR_PCN
[ "$FUSEBASE" = off ] && export RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0
best_vars
quit() { run "docker logs $CN > $O/xap-$1-engine.log 2>&1"; down; unset RADIANCE_DS41_AR_PCN RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE; exit ${2:-1}; }
# alive ARM STEP: the server still answers /health and its container runs; else the item ends here
alive() {
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200 (else: logs, down, exit)"; return 0; }
  local st code
  st=$(docker inspect -f '{{.State.Status}} exit {{.State.ExitCode}}' $CN 2>/dev/null || echo gone)
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT xfix 81: VIRHE the $1 arm's server is not healthy after $2 (container: $st, /health $code; last E lines: $(docker logs $CN 2>&1 | grep -E '^E |trap|abort|HSA_STATUS' | tail -3 | tr '\n' ' ' | cut -c1-500)) -> logs saved, down, stopped"
  quit $1
}
# gset TAG CONC N: the 20 greedy prompts at concurrency CONC into $O/TAG-gs-N.jsonl
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-$3.jsonl --conc $2 --max-tokens 192 > $O/$1-gs-$3.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "+ greedyset --cmp $1 $2" >&2; echo IDENTICAL; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
for arm in $ARMS; do
  case $arm in
    off) unset RADIANCE_DS41_AR_PCN; want=separate ;;
    on)  export RADIANCE_DS41_AR_PCN=1; want=fused ;;
    *)   log "RESULT xfix 81: unknown arm $arm"; continue ;;
  esac
  gate xap-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 81: VIRHE $arm arm not healthy ($(desc))"; quit $arm; }
  alive $arm gate
  [ $rc = 2 ] && { log "RESULT xfix 81: VIRHE $arm arm slow twice -> stopped"; quit $arm; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xap-$arm-mode.txt"
  if [ "$DRY" != 1 ]; then
    got=$(grep -o 'post-collapse-norm [a-z]* at [0-9]* layers' $O/xap-$arm-mode.txt)
    ev=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_DS41_AR_PCN=' | cut -d= -f2- || true)
    n=$(echo "$got" | awk '{print $4}')
    if [ -z "$got" ] || ! echo "$got" | grep -q "norm $want " || { [ $want = fused ] && [ "${n:-0}" -lt 1 ]; } || { [ $want = separate ] && [ "${n:-1}" != 0 ]; }; then
      log "RESULT xfix 81: VIRHE the $arm arm runs '$got' (env '${ev}'; want $want) -> stopped"
      quit $arm
    fi
    log "RESULT xfix 81 $arm: $got (env RADIANCE_DS41_AR_PCN='${ev}')"
  fi
  ident xap-$arm;   alive $arm "the identity copy"
  gset xap-$arm 1 c1; alive $arm "greedyset c1"
  gset xap-$arm 4 c4; alive $arm "greedyset c4"
  [ $arm = off ] && { gset xap-$arm 4 c4b; alive $arm "greedyset c4 (second)"; }
  probe xap-$arm;   alive $arm "the stepprobe"
  mbp xap-$arm c4;  alive $arm "mb c4"
  run "docker logs $CN > $O/xap-$arm-engine.log 2>&1"
  stopfile && quit $arm 0
done
unset RADIANCE_DS41_AR_PCN
case " $ARMS " in *" off "*) case " $ARMS " in *" on "*)
  identcmp xap-off xap-on || log "RESULT xfix 81: VIRHE the fused collective changed the greedy output (must be exact)"
  c1=$(gcmp $O/xap-off-gs-c1.jsonl $O/xap-on-gs-c1.jsonl); c4=$(gcmp $O/xap-off-gs-c4.jsonl $O/xap-on-gs-c4.jsonl)
  fl=$(gcmp $O/xap-off-gs-c4.jsonl $O/xap-off-gs-c4b.jsonl)
  log "RESULT xfix 81 on vs off: greedy c1 $c1 | c4 $c4 (off's own c4 floor: $fl)"
  case "$c1" in *DIFFERENT*) log "RESULT xfix 81: VIRHE the fused collective changed greedy answers (must be exact)" ;; esac
  case "$fl" in IDENTICAL*) case "$c4" in *DIFFERENT*) log "RESULT xfix 81: VIRHE the fused collective changed the c4 answers (off's two c4 runs agree)" ;; esac ;; esac ;; esac ;; esac
down
unset RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE
log "RESULT xfix 81-arpcn: done | ref $REF_BEST524"
