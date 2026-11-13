#!/bin/bash
# 73-qlds-ab.sh (queue item, 6.11.2026) -- the decode index_score's q staged in LDS (radiance-local:ds41-exl3-117-<ds41-qlds
# sha8>-f2f3c3f7-qlds = the ctail image with libr4d.so of ds41-qlds f6f767f). The scalar index_score kernels (the decode
# band, <= 48 rows a pass: layers 2/8/14/20 unrestricted, 24/28/32/36 kept) read q by scalar loads and spilled 206-266 SGPRs
# through lane moves; their 4-head group loop was ~2400 instructions for 512 FMAs. Now q is in LDS as f32 (same values),
# read in chunks with a one-chunk prefetch: 0 SGPR spills in the plain kernels (40-54 left in the kept ones, outside the
# score loop), 0 VGPR spills, ~1070 instructions a group (device ISA). Same products, same order: same bits.
# R4D_DS41_QLDS=0 restores the scalar form in the same image.
# As 64-idx-decode-stamps (the 47 flags: MML 524288 MAXSEQS 4 NST 0 ckint 8192 MNBT 8192, RADIANCE_DS41_INDEX default
# keep,split; packed formats off; ctail = the image's default, bound), per arm:
#   ql      (default): restart + gate (24 ms), identity copy (~145k), onem.py 135000,262144 with AQL stamps every 8th pass
#   ql1     the ql arm with EVERY pass stamped (AQLSTAMP=1), no copy: against ql (1 in 8) at the same load, the stamp's
#           own cost a decode step (the deep lines' ms/step) and a stamped pass's wall
#   scalar  R4D_DS41_QLDS=0: as ql
# then the two copies byte for byte, and idxstamps.py per arm: the decode pass's wall / busy, every index_score dispatch's
# time and its exclusive part at 135k / 262k / the hit turn. The arm's container env is checked (docker inspect) so a yaml
# without R4D_DS41_QLDS cannot run two scalar arms. Read with 64's numbers (same flags, idx image).
# Estimate ~65 min (4 gated starts, 3 copies, onem 135k+262k four times; 7.11.: + the off arm). ARMS="ql scalar" skips the stamp-cost arm.
# DRYRUN=1 bash 73-qlds-ab.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh), ds41-fix2-tp4.yaml
# (passes R4D_DS41_QLDS) and idxstamps.py in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3QL=${X3QL:-radiance-local:ds41-exl3-117-f6f767f6-f2f3c3f7-qlds}
export IMAGE=$X3QL
P=$O/qlds
ARMS=${ARMS:-"ql ql1 scalar off"}   # 7.11. queue-ar: + off (64-idx-decode-stamps' question on today's kernels)
EVERY=${STAMP_EVERY:-8}
IS=$QD/idxstamps.py; [ -e $IS ] || IS=$(cd "$(dirname "$0")" && pwd)/idxstamps.py
log "RESULT xfix 73-qlds: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] stamp 1 pass in $EVERY compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q R4D_DS41_QLDS $Y || { log "RESULT xfix 73: VIRHE $Y does not pass R4D_DS41_QLDS"; exit 1; }
case " $ARMS " in *" off "*) [ "$DRY" = 1 ] || grep -q "^ *RADIANCE_DS41_INDEX:" $Y || { log "RESULT xfix 73: VIRHE $Y does not pass RADIANCE_DS41_INDEX (off arm)"; exit 1; } ;; esac
[ "$DRY" = 1 ] || [ -e $IS ] || { log "RESULT xfix 73: VIRHE idxstamps.py missing"; exit 1; }
mkd $O $P
unset RADIANCE_DS41_INDEX RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL R4D_DS41_QLDS
pfflags
for arm in $ARMS; do
  ev=$EVERY
  case $arm in
    ql)     unset R4D_DS41_QLDS ;;
    ql1)    unset R4D_DS41_QLDS; ev=1 ;;   # the stamp's own cost: every pass stamped, the same load as ql's
    scalar) export R4D_DS41_QLDS=0 ;;
    off)    export R4D_DS41_QLDS=0 RADIANCE_DS41_INDEX=off ;;   # the unrestricted, unsplit indexer on the scalar kernels (64's off arm)
    *)      log "RESULT xfix 73: unknown arm $arm"; continue ;;
  esac
  export STATSDIR=$P/$arm AQLSTAMP=$ev
  mkd $STATSDIR
  run "rm -f $STATSDIR/stamps.csv"
  gate xql-$arm 24; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 73: VIRHE $arm arm not healthy ($(desc))"; unset AQLSTAMP R4D_DS41_QLDS; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 73: VIRHE $arm arm slow twice -> skipped"; down; continue; }
  if [ "$DRY" != 1 ]; then
    qv=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^R4D_DS41_QLDS=' || true)
    { [ $arm = scalar -o $arm = off ] && [ "$qv" = "R4D_DS41_QLDS=0" ]; } || { [ $arm != scalar ] && [ $arm != off ] && [ -z "$qv" -o "$qv" = "R4D_DS41_QLDS=" ]; } || {
      log "RESULT xfix 73: VIRHE the $arm arm's container env says '${qv:-no R4D_DS41_QLDS}' -> stopped"; down; unset AQLSTAMP R4D_DS41_QLDS; exit 1; }
  fi
  if [ "$DRY" != 1 ] && [ $arm = off ]; then
    iv=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_DS41_INDEX=' || true)
    [ "$iv" = "RADIANCE_DS41_INDEX=off" ] || { log "RESULT xfix 73: VIRHE the off arm's container env says '${iv:-no RADIANCE_DS41_INDEX}' -> stopped"; down; unset AQLSTAMP R4D_DS41_QLDS RADIANCE_DS41_INDEX; exit 1; }
  fi
  [ $arm = ql1 ] || ident xql-$arm
  deep xql-$arm 135000,262144
  run "docker logs $CN > $P/$arm-engine.log 2>&1"
  down   # the stamp writer flushes at exit
  run "python3 $IS $STATSDIR/stamps.csv --last 3 > $P/$arm-stamps.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xfix 73 $arm (R4D_DS41_QLDS=${R4D_DS41_QLDS-unset} RADIANCE_DS41_INDEX=${RADIANCE_DS41_INDEX-unset}, stamp 1 pass in $ev): 135k / 262k / hit-turn: $(grep '^  segment' $P/$arm-stamps.txt | cut -c3- | tr '\n' '|' | cut -c1-1400)"
  unset RADIANCE_DS41_INDEX
  stopfile && { unset AQLSTAMP R4D_DS41_QLDS; exit 0; }
done
unset AQLSTAMP R4D_DS41_QLDS RADIANCE_DS41_INDEX
case " $ARMS " in *" ql "*) case " $ARMS " in *" scalar "*) identcmp xql-scalar xql-ql || log "RESULT xfix 73: VIRHE q in LDS changed the greedy output (must be exact)" ;; esac ;; esac
case " $ARMS " in *" off "*) case " $ARMS " in *" scalar "*) identcmp xql-off xql-scalar || log "RESULT xfix 73: VIRHE the restricted indexer changed the greedy output (must be exact)" ;; esac ;; esac
log "RESULT xfix 73-qlds: done -- $P/{ql,scalar,off}-stamps.txt, stamps.csv per arm | off = the unrestricted indexer (64's question: the keep layers' index_score with keep vs without, on today's kernels)"
