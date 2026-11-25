#!/bin/bash
# 61-exl3-29-tuned.sh (chain-after queue item, prepared 6.11.2026 in queue-ar, NOT run) -- 60-exl3-29.sh's set on the LibertAI EXL3
# 2.9 bpw .rad with our TUNED engine instead of the stock 1.17 EXL3 image, same serve line (x117-best524: TP4 exact,
# expert_tiered, MML 524288, NST 0, MAXSEQS 8, ngram ram, RAM KV tier 84 GiB, ckint 2048 + span 2 + IL 8, host pool HP29):
# "1.17 vs tuned" at the same flags. Lives beside 60 in gpu02 ~/gpu/ds41/exl3-test/queue-29/ with ds41-tuned-tp4.yaml; uses
# lib-29.sh / cmp29.py there (same lookup as 60). Every piece is exact by itself; run only the pieces that passed their own
# TP4 item, chosen by PIECES (default all four; drop a word to drop a piece):
#   idx   idx2b's indexer (candidate restriction + row split, thresholds 2 x candidates)   off -> RADIANCE_DS41_INDEX=off
#   mx4   MXFP4 index keys                                    (item 67)   on -> RADIANCE_DS41_IDXKEY=mxfp4
#   heat  heat-engine prefill share                           (item 70)   on -> RADIANCE_HEAT_PREFILL=$HEATP (default 0)
#   gu25  exl3r4d gate_up half-K prefetch 3                   (63, 66)    on -> image ...-5dd4abf3-tuned-..., off -> ...-f2f3c3f7-...
# Images (build/tuned/assemble.sh, no-GPU checks in build/tuned/verify-*: 0 plugin refusals, host start declares the same 2202
# ops as the mx4 image, 0 unresolved moe_exl3 / index_* / all_gather bands):
#   with gu25     radiance-local:ds41-exl3-117-8505d5ff-5dd4abf3-tuned-h058faddc   c5107389b814
#   without gu25  radiance-local:ds41-exl3-117-8505d5ff-f2f3c3f7-tuned-h058faddc   72c4c726a810
#   = the mx4 image (ds41-mx4 8505d5ff: libr4d/libref/libavx/deepseek41.so) + bin/radiance of ds41-ar 058faddc (+ the gu25 build's
#   exl3r4d/exl3ref/exl3_quant, interface v9; this deepseek41.so passes no ep_ranks, so E1/E2 split-K stay off).
# Steps = 60's: pre-checks, evict, gate (GATE29 30 ms), warm-up answers vs the 2.25's, IDENTITY: the warm-up answers (greedy,
# thinking off, ds41_tests.py correct --only-q) byte for byte vs 60's on the same .rad + flags (must match: every piece is exact;
# a difference is logged as VIRHE and the measurements still run), probe, mb c4 / c8 / pf / turns, cc_sim u4, engine log,
# cmp29.py line TAG vs x29-best524 (60) -- the comparison asked for -- then down.
# ~45 min (60's estimate). DRYRUN=1 bash 61-exl3-29-tuned.sh : prints, writes nothing. Run after 60 (its answers and rows are the
# reference); without them the identity line says so and cmp29 prints '-' for the reference.
H29="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$H29/lib-29.sh" "$H29/../lib-29.sh" "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
for YT in "$H29/ds41-tuned-tp4.yaml" "$H29/../ds41-tuned-tp4.yaml" "$K29/ds41-tuned-tp4.yaml"; do [ -e "$YT" ] && break; done
[ -e "$YT" ] || { echo "ds41-tuned-tp4.yaml not found"; exit 1; }
Y=$YT
CMP=$K29/cmp29.py
PIECES=${PIECES-"idx mx4 heat gu25"}; HEATP=${HEATP:-0}
has() { case " $PIECES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
X3T_GU=${X3T_GU:-radiance-local:ds41-exl3-117-8505d5ff-5dd4abf3-tuned-h058faddc}
X3T_NOGU=${X3T_NOGU:-radiance-local:ds41-exl3-117-8505d5ff-f2f3c3f7-tuned-h058faddc}
if has gu25; then export IMAGE=$X3T_GU; else export IMAGE=$X3T_NOGU; fi
unset RADIANCE_DS41_INDEX RADIANCE_DS41_IDXKEY RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP
has idx || export RADIANCE_DS41_INDEX=off
has mx4 && export RADIANCE_DS41_IDXKEY=mxfp4
has heat && export RADIANCE_HEAT_PREFILL=$HEATP
TAG=x29-t-$(echo $PIECES | tr ' ' '-'); [ -n "$PIECES" ] || TAG=x29-t-none
REFTAG=x29-best524
ENVS="RADIANCE_DS41_INDEX=${RADIANCE_DS41_INDEX-unset} RADIANCE_DS41_IDXKEY=${RADIANCE_DS41_IDXKEY-unset} RADIANCE_HEAT_PREFILL=${RADIANCE_HEAT_PREFILL-unset}"
log "RESULT x29 61-exl3-29-tuned: start PIECES [$PIECES] IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) $ENVS MODEL=$MODELDIR/$MODEL HP29=$HP29 GATE29=$GATE29 yaml $Y"
if [ "$DRY" != 1 ]; then
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT x29 61: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -f $MODELDIR/$MODEL ] || { log "RESULT x29 61: VIRHE $MODELDIR/$MODEL missing"; exit 1; }
  [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT x29 61: VIRHE $MODELDIR/$MODEL.sha256ok missing (copy not verified)"; exit 1; }
  [ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT x29 61: VIRHE production is running"; exit 1; }
  [ -s $O/guard/$REFTAG-warm-q.txt ] || log "RESULT x29 61: note: no 60 warm-up answers ($O/guard/$REFTAG-warm-q.txt); identity not checked"
fi
# identq NEW REF: every warm-up answer (content, finish) of NEW byte for byte against REF's, by test name
identq() {
  [ "$DRY" = 1 ] && { echo "+ RESULT x29 $1 identity: $O/guard/$1-warm-q.txt vs $O/guard/$2-warm-q.txt"; return 0; }
  [ -s $O/guard/$2-warm-q.txt ] || return 0
  local r
  r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = []
for t in b:
    x, y = a.get(t, {}), b[t]
    cx, cy = x.get("content") or "", y.get("content") or ""
    if cx != cy or x.get("finish") != y.get("finish"):
        i = next((k for k, (p, q) in enumerate(zip(cx, cy)) if p != q), min(len(cx), len(cy)))
        bad.append("%s at char %d (%d vs %d chars)" % (t, i, len(cx), len(cy)))
n = sum(len(b[t].get("content") or "") for t in b)
print(("IDENTICAL: %d answers, %d chars" % (len(b), n)) if not bad and b else ("DIFFERENT: " + "; ".join(bad) if b else "no reference answers"))' \
     $O/guard/$1-warm-q.txt $O/guard/$2-warm-q.txt 2>&1 | cut -c1-600)
  case $r in IDENTICAL*) log "RESULT x29 $1 identity vs 60 ($2, stock 1.17, same .rad + flags): $r" ;;
             *) log "RESULT x29 $1 identity vs 60 ($2): VIRHE $r -- a piece is not exact on this .rad; measurements continue" ;; esac
}
mkd $O $STATSDIR
evict
best_vars
gate $TAG $GATE29; rc=$?
[ $rc = 1 ] && { log "RESULT x29 61: VIRHE not healthy ($(desc)); start log $O/start-$TAG.log: $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; exit 1; }
[ $rc = 2 ] && { log "RESULT x29 61: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
[ "$DRY" = 1 ] || log "RESULT x29 61 $TAG modes: $(docker logs $CN 2>&1 | grep -E 'deepseek41 indexer|heat engine: a step' | head -2 | cut -c1-260 | tr '\n' '|')"
warmq $TAG
identq $TAG $REFTAG
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
run "python3 $CMP $L $TAG $REFTAG >> $O/$TAG-cmp.txt 2>&1"
[ "$DRY" = 1 ] || log "$(tail -1 $O/$TAG-cmp.txt | cut -c1-1500) [tuned PIECES $PIECES vs stock 1.17]"
log "RESULT x29 61 $TAG: done | ref 60 $REFTAG (stock 1.17, same .rad + flags) | $REF_X117_524"
unset RADIANCE_DS41_INDEX RADIANCE_DS41_IDXKEY RADIANCE_HEAT_PREFILL
down
log "RESULT x29 61-exl3-29-tuned: done"
