#!/bin/bash
# dsm-ngsweep.sh (11.11.2026, queue-ar/dsmain, ds-main-port T1 diagnosis) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-*-tree4.sh.
# THE VERIFY ROW-COUNT SWEEP, NO TREE. tree3 (11.11. 04:25-04:37) localised the T1 oracle failure to one row: the tree oracle
# (3 x 5 forced, 21 verify rows) emits the plain run's tokens for 11 steps (every row on a correct path -- main chain and
# branches, after branch commits too -- has the chain's target), then at step 12 (output 37) main-chain node 2's row gives
# 84443 where the plain run gives 2052, while node 3 (its child, which reads everything node 2 reads) is right; the same
# tree path at dspark_tree 0 (6 verify rows) is IDENTICAL. Hypothesis to test: not the tree path, but the trunk's kernels
# are not row-count invariant above some verify height (engine.cpp: "split-K depends on M, so logits differ slightly"; the
# gg_gemv / skinny forms cover small M), so a 21-row verify flips a near-tie the 1-row plain run did not.
# Starts, each the tree image with the tree UNSET (its graph = ds41-int c05de27's, hostcheck) on the stack env, NO draft head
# (--spec-ngram K leaves the DSpark head out), --debug-draft-oracle (prompt lookup's chain = the plain run's ids, 30 %
# corrupted): ng20 (21-row chain verifies, the tree's height), ng12 (13), ng8 (9), ng5 (6 rows, the DSpark NST 5 height),
# one oracle.py check each (256 greedy ids vs bz d0's): IDENTICAL / DIFFERENT (first diff), drafts and accepted.
# Reading: ng20 DIFFERENT -> the 21-row verify is not the 1-row decode (not T1's code); ng20 IDENTICAL -> T1-specific.
# ~16 min. Tag "RESULT xng", out runs/xdsm/ngsweep. XNG_IMAGE, XNG_KS (default "20 12 8 5").
# DRYRUN=1: prints every command, checks the image and the helpers, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-tree3-tp4.yaml
O=$R/xdsm/ngsweep; export STATSDIR=$O/stats
RD=$R/xdsm/draft
ORC=$HD/oracle.py
export IMAGE=${XNG_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l}
ITEM=${XNG_ITEM:-57-p4zyzzzzz-ds-tree4}
KS=${XNG_KS:-"20 12 8 5"}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DS41_DSPARK_TREE RADIANCE_DS41_DSPARK_TREE_B RADIANCE_DS41_DSPARK_TREE_L RADIANCE_DS41_DSPARK_TREE_FORCE
  RADIANCE_DEBUG_ORACLE_TREE RADIANCE_DEBUG_TREE_STEPS"
stackenv() {   # the tree items' start env (dsm-tree.sh treeenv) WITHOUT the tree
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
trap 'down' EXIT
trap 'exit 143' TERM INT
ngarm() {   # ngarm K: a start with --spec-ngram K --debug-draft-oracle, one oracle check
  local k=$1 tag=ng$1 r ee
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST= XARGS="$XARGS --live --no-prefix-cache --spec-ngram $k --debug-draft-oracle /stats/oracle-ids.txt"
  log "RESULT xng $tag: start IMAGE=$IMAGE MODEL=$MODEL tree unset, no draft head, --spec-ngram $k (chain verifies of $((k + 1)) rows) + --debug-draft-oracle $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xng $tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-600)"; echo "$tag START-FAILED" >> $O/sweep.txt; return 1; }
  run "timeout 900 python3 -I $ORC check $URL $STATSDIR/oracle-ids.txt $O/$tag-check1.json > $O/$tag-check1.txt 2>&1"
  if [ "$DRY" != 1 ]; then
    r=$(grep -m1 '^ORACLE' $O/$tag-check1.txt | cut -c8- | cut -c1-300)
    ee=$(docker logs $CN 2>&1 | grep -E '^E ' | head -2 | tr '\n' ' ' | cut -c1-400)
    log "RESULT xng $tag harness lines: $(docker logs $CN 2>&1 | grep -iE -- '--debug-draft-oracle|--spec-ngram|speculation: |draft trees' | sed 's/^[A-Z] *//' | sort -u | head -5 | tr '\n' '|' | cut -c1-900)"
    case "$r" in
      *'"identical": true'*) echo "$tag IDENTICAL" >> $O/sweep.txt; log "RESULT xng $tag: IDENTICAL to d0's ids at $((k + 1))-row chain verifies ($r)" ;;
      '') echo "$tag ERROR" >> $O/sweep.txt; log "RESULT xng $tag: VIRHE ERROR -- no reply ($(tail -1 $O/$tag-check1.txt | cut -c1-200)) | engine: ${ee:-no E line}" ;;
      *) echo "$tag DIFFERENT" >> $O/sweep.txt; log "RESULT xng $tag: DIFFERENT from d0's ids at $((k + 1))-row chain verifies ($r) | engine: ${ee:-no E line}" ;;
    esac
  fi
  keeplog $tag; down
}
log "RESULT xng $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | --spec-ngram sweep [$KS] with --debug-draft-oracle, tree unset, no draft head, the stack env | bz d0's ids"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xng: VIRHE image $IMAGE not on this host"; exit 1; }
if [ "$DRY" = 1 ]; then
  hmiss=""
  for fn in log run mkd desc best_vars serve_args restart evict down stopfile; do [ "$(type -t $fn)" = function ] || hmiss="$hmiss $fn()"; done
  for f in $Y $ORC $RD/oracle-ids.txt $MODELDIR/$M225 $C; do [ -e "$f" ] || hmiss="$hmiss $f"; done
  python3 -I -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $ORC 2>/dev/null || hmiss="$hmiss $ORC(syntax)"
  [ -z "$hmiss" ] && echo "+ HELPERS OK: log run mkd desc best_vars serve_args restart evict down stopfile, yaml, oracle.py, d0 ids, model, compose dir" \
                  || { echo "+ HELPERS MISSING:$hmiss"; exit 1; }
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xng: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -s $RD/oracle-ids.txt ] || { log "RESULT xng: VIRHE no $RD/oracle-ids.txt"; exit 1; }
fi
mkd $O $STATSDIR
[ "$DRY" = 1 ] || : > $O/sweep.txt
run "cp $RD/oracle-ids.txt $STATSDIR/oracle-ids.txt"
evict
for k in $KS; do stopfile && break; ngarm $k; done
unset $SWITCHES
down
log "RESULT xng summary: $(tr '\n' ' ' 2>/dev/null < $O/sweep.txt) (chain verify rows = K + 1; the T1 tree verifies 21 rows; tree3: tree path at 6 rows IDENTICAL, at 21 rows DIFFERENT at token 39)"
log "RESULT xng $ITEM: done -- $O"
