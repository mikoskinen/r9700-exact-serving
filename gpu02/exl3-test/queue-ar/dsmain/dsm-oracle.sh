#!/bin/bash
# dsm-oracle.sh (10.11.2026, queue-ar/dsmain, ds41-gg agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-h-oracle.sh.
# The verify-path oracle BESIDE THE DSPARK HEAD on deepseek41 (bz's oracle start died: --spec-ngram is refused on a container
# that declares a draft head, and with --num-speculative-tokens 5 the harness only filled proposals the head left empty).
# The ds41-gg image (XGG_IMAGE) changes only the harness: with --debug-draft-oracle and a head drafting, the oracle's ids (the
# same 30 % corruption hash) REPLACE each proposal the head publishes, at the head's own count, so every verify runs at the
# head's shape (6 rows for DSpark's block of 5) and accepts / rejects where the harness says; under the harness no step is
# issued ahead of its commit (the proposal must be on the host before the verify is built), so this is not a speed.
# EXL3 2.25, TP4, MML 524288, the t3 env and serve line of dsm-serve.sh, NST 5, --no-prefix-cache.
# STARTS (XGH_ARMS, default "o5 o5a"):
#   o5   --num-speculative-tokens 5 --debug-draft-oracle /stats/oracle-ids.txt
#   o5a  o5 + --draft-window adaptive (each sequence's window, so the verify shape moves 2..6 rows)
# EACH: restart (no gate: the reply is the measurement), the harness lines in the serve log (the scheduler's "every
#   proposal of the draft head ... replaced" and the engine's "beside the draft head 'dspark': no step is issued ahead"),
#   then oracle.py check: one fixed prompt, greedy, ignore_eos, 256 tokens, its ids against bz's d0 (NST 0, main image)
#   reference runs/xdsm/draft/oracle-ids.txt -- IDENTICAL is the pass; draft_n > draft_n_accepted > 0 shows the harness
#   drafted and was rejected. A second check request on the same server (the passes now played from their recordings).
# NO cc_sim. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-oracle.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-gg-tp4.yaml
O=$R/xdsm/gg-oracle; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 reference: oracle-ids.txt / oracle-ref.json
export IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4d70490d-5c13e555-l}
ITEM=${XGH_ITEM:-57-p4zyzzzzz-ds-h-oracle}; ARMS=${XGH_ARMS-"o5 o5a"}
ORC=$HD/oracle.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV"
t3env() {   # 105's t3 arm, every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
trap 'down' EXIT
trap 'exit 143' TERM INT
ocheck() {   # ocheck TAG N: one oracle.py check -> the result line
  run "timeout 900 python3 -I $ORC check $URL $STATSDIR/oracle-ids.txt $O/$1-check$2.json > $O/$1-check$2.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  local r; r=$(grep -m1 '^ORACLE' $O/$1-check$2.txt | cut -c8- | cut -c1-300)
  case "$r" in
    *'"identical": true'*)
      case "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); n, a = d.get("draft_n") or 0, d.get("draft_n_accepted") or 0; print("ok" if n > a > 0 else "vacuous")' "$r" 2>/dev/null)" in
        ok) log "RESULT xgh $1 check $2: IDENTICAL to d0's ids, the harness drafted and was rejected ($r)" ;;
        *)  log "RESULT xgh $1 check $2: IDENTICAL to d0's ids but VIRHE vacuous -- no rejected draft (draft_n <= accepted or 0): the harness did not drive the verify ($r)" ;;
      esac ;;
    *) log "RESULT xgh $1 check $2: VIRHE DIFFERENT from d0's ids ($r) -- the 6-row verify path is not exact beside the head (or the harness misfired)" ;;
  esac
}
oarm() {   # oarm o5|o5a
  local a=$1 tag=gh-$1
  t3env; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --no-prefix-cache --debug-draft-oracle /stats/oracle-ids.txt"
  case $a in o5) ;; o5a) export XARGS="$XARGS --draft-window adaptive" ;; *) log "RESULT xgh: VIRHE unknown arm $a"; return 1 ;; esac
  log "RESULT xgh $tag: start IMAGE=$IMAGE MODEL=$MODEL NST 5 + --debug-draft-oracle (the DSpark head's proposals replaced) $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xgh $tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  if [ "$DRY" = 1 ]; then echo "+ RESULT xgh $tag harness lines (scheduler + engine)"
  else
    log "RESULT xgh $tag harness lines: $(docker logs $CN 2>&1 | grep -E -- '--debug-draft-oracle|DSpark drafter|speculation: ' | sed 's/^[A-Z] *//' | head -5 | tr '\n' '|' | cut -c1-900)"
    docker logs $CN 2>&1 | grep -q "every proposal of the draft head" && docker logs $CN 2>&1 | grep -q "beside the draft head 'dspark': no step is issued ahead" \
      || log "RESULT xgh $tag: VIRHE the serve log lacks the harness-beside-a-head lines (an image without the change?)"
  fi
  ocheck $tag 1
  ocheck $tag 2
  keeplog $tag; down
  log "RESULT xgh $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xgh $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] reference $RD/oracle-ids.txt (bz d0, NST 0, main image)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xgh: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xgh: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && [ -e $ORC ] || { log "RESULT xgh: VIRHE $Y or $ORC missing"; exit 1; }
  [ -s $RD/oracle-ids.txt ] || { log "RESULT xgh: VIRHE no reference ids $RD/oracle-ids.txt"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xgh: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "every proposal of the draft head" /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xgh: VIRHE $IMAGE's engine has no harness-beside-a-head change"; exit 1; }
fi
mkd $O $STATSDIR
run "cp $RD/oracle-ids.txt $STATSDIR/oracle-ids.txt"
done_arms=""
for a in $ARMS; do stopfile && break; oarm $a && done_arms="$done_arms $a"; done
unset $SWITCHES
down
log "RESULT xgh $ITEM: done -- arms [$done_arms ] $O"
