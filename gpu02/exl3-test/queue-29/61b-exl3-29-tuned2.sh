#!/bin/bash
# 61b-exl3-29-tuned2.sh (chain-after queue item, 6.11.2026 evening, queue-r2; = queue-ar 61-exl3-29-tuned.sh with the tuned2 images
# and three more pieces) -- 60-exl3-29.sh's set on the LibertAI EXL3 2.9 bpw .rad with our TUNED engine instead of the stock 1.17
# EXL3 image, same serve line (x117-best524: TP4 exact, expert_tiered, MML 524288, NST 0, MAXSEQS 8, ngram ram, RAM KV tier 84 GiB,
# ckint 2048 + span 2 + IL 8, host pool HP29): "1.17 vs tuned" at the same flags. Lives beside 60 in gpu02
# ~/gpu/ds41/exl3-test/queue-29/ with ds41-tuned2-tp4.yaml (passes every switch below); uses lib-29.sh / cmp29.py there. Every
# piece is exact by itself; run only the pieces that passed their own A/B, chosen by PIECES (default "idx mx4 heat gu25"; latent,
# ctail and gu25c are off until their A/Bs pass -- add the word to turn one on):
#   idx    idx2b's indexer (candidate restriction + row split)  (62, 65)   off -> RADIANCE_DS41_INDEX=off
#   mx4    MXFP4 index keys                                     (67)       on  -> RADIANCE_DS41_IDXKEY=mxfp4
#   latent nvfp4 compressed latent                              (68)       on  -> RADIANCE_DS41_LATENT=nvfp4
#   ctail  the compressor tail over the step's bound            (72)       on  = unset (the image's default); off -> RADIANCE_DS41_CTAIL=full
#   heat   dual heat + decode pin (ds41-ar 7e342763)            (70, 72-heat) on -> RADIANCE_HEAT_PREFILL_DECAY=$HEATDECAY (0.7)
#          RADIANCE_HEAT_DECODE_PIN=$HEATPIN (0.8), RADIANCE_HEAT_PREFILL unset; off -> all RADIANCE_HEAT_* unset (as before). The
#          other agent's 2.25 bpw result: 266k prefill 2436 tok/s (p1 2512), deep decode 25.0 ms/step (p1 28.6), IDENTICAL
#   gu25   exl3r4d 12-byte tile load at K2.5 (gate_up, PF 4)    (57-l)     image ...-67151b18-tuned2-...
#   gu25c  the same at K2.5 AND K3                              (57-l K3)  image ...-fa0e9580-tuned2-... (implies gu25's K2.5 load)
#   neither gu25 nor gu25c                                                 image ...-f2f3c3f7-tuned2-... (exl3 f2f3c3f7 plugins)
# NOTE the 2.9 .rad's experts are K = 3 (layers 0-17, 23-39) and K = 2 (18-22) (lib-29.sh): no K2.5 tensor, so gu25 alone runs
# the same expert kernels as the f2f3c3f7 image on this .rad; gu25c is the EXL3 piece that acts here (K3 gate_up: 40 -> 16 loads
# a K loop, six s_wait_loadcnt 0 -> none; fmt-micro K3 gate_up M1 133.85 us vs K2 81.41).
# Images (build/tuned/assemble-tuned2.sh; no-GPU checks build/tuned/verify-tuned2.log: 0 plugin refusals, rad-schemas 2493 lines /
# 16 moe_exl3 as the mx4l base, host start 258 E lines and 0 unresolved moe_exl3 / index_* / all_gather bands in each, the
# indexer note naming keys / latent / compressor tail as set):
#   gu25c   radiance-local:ds41-exl3-117-7af8fee3-fa0e9580-tuned2-h7e342763   11a7d3ebe622
#   gu25    radiance-local:ds41-exl3-117-7af8fee3-67151b18-tuned2-h7e342763   4ecf4d3de300
#   none    radiance-local:ds41-exl3-117-7af8fee3-f2f3c3f7-tuned2-h7e342763   2acbef1172b2
#   (the same three with the 058faddc heat binary: ...-tuned2-h058faddc 732a2b7dcdea / 44729226edb4 / 5c37cd194e1e;
#   X3T2_H=058faddc with HEATMODE=share for the prefill-share heat piece)
#   = the mx4l image (ds41-mx4l c528cdff: libr4d/libref/libavx: idx2b indexer + MXFP4 keys + nvfp4 latent, each behind its env)
#   + deepseek41.so of ds41-ctail 7af8fee + bin/radiance of ds41-ar 7e342763 (heat: prefill share, move budget, dual heat, decode
#   pin, each behind its env; all unset = the engine as before) + exl3r4d.so of the EXL3 choice
#   (interface v8 throughout). Another heat binary: build/tuned/assemble-tuned2.sh <its image> -> ...-tuned2-h<sha8>, then queue
#   this item with X3T2_H=<sha8> (and that binary's RADIANCE_HEAT_* in the shell; the yaml passes them all).
# Steps = 61's: pre-checks, evict, gate (GATE29 30 ms), the serve log's indexer note against PIECES (keys / latent / compressor
# tail; a mismatch stops the item), warm-up answers vs the 2.25's, IDENTITY vs 60's answers (same .rad + flags; must match: every
# piece is exact; a difference is logged as VIRHE and the measurements still run), probe, mb c4 / c8 / pf / turns, cc_sim u4,
# engine log, cmp29.py TAG vs x29-best524 (60), then down.
# ~45 min (60's estimate). DRYRUN=1 bash 61b-exl3-29-tuned2.sh : prints, writes nothing. Run after 60.
H29="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$H29/lib-29.sh" "$H29/../lib-29.sh" "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
for YT in "$H29/ds41-tuned2-tp4.yaml" "$H29/../ds41-tuned2-tp4.yaml" "$K29/ds41-tuned2-tp4.yaml"; do [ -e "$YT" ] && break; done
[ -e "$YT" ] || { echo "ds41-tuned2-tp4.yaml not found"; exit 1; }
Y=$YT
CMP=$K29/cmp29.py
PIECES=${PIECES-"idx mx4 heat gu25"}; HEATP=${HEATP:-0}
HEATMODE=${HEATMODE:-dualpin}   # dualpin (7e342763 binaries): RADIANCE_HEAT_PREFILL_DECAY + _DECODE_PIN; share: RADIANCE_HEAT_PREFILL
HEATDECAY=${HEATDECAY:-0.7}; HEATPIN=${HEATPIN:-0.8}
has() { case " $PIECES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for w in $PIECES; do case $w in idx|mx4|latent|ctail|heat|gu25|gu25c) ;; *) log "RESULT x29 61b: VIRHE unknown piece '$w' in PIECES [$PIECES]"; exit 1 ;; esac; done
X3T2_H=${X3T2_H:-7e342763}   # the heat binary's label in the tuned2 tags (assemble-tuned2.sh)
X3T2_C=${X3T2_C:-radiance-local:ds41-exl3-117-7af8fee3-fa0e9580-tuned2-h$X3T2_H}
X3T2_B=${X3T2_B:-radiance-local:ds41-exl3-117-7af8fee3-67151b18-tuned2-h$X3T2_H}
X3T2_0=${X3T2_0:-radiance-local:ds41-exl3-117-7af8fee3-f2f3c3f7-tuned2-h$X3T2_H}
if has gu25c; then export IMAGE=$X3T2_C; elif has gu25; then export IMAGE=$X3T2_B; else export IMAGE=$X3T2_0; fi
unset RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT \
      RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN \
      RADIANCE_HEAT_DECODE_PIN
has idx || export RADIANCE_DS41_INDEX=off
has mx4 && export RADIANCE_DS41_IDXKEY=mxfp4
has latent && export RADIANCE_DS41_LATENT=nvfp4
has ctail || export RADIANCE_DS41_CTAIL=full
if has heat; then
  case $HEATMODE in
    dualpin) export RADIANCE_HEAT_PREFILL_DECAY=$HEATDECAY RADIANCE_HEAT_DECODE_PIN=$HEATPIN ;;
    share)   export RADIANCE_HEAT_PREFILL=$HEATP ;;
    *)       log "RESULT x29 61b: VIRHE HEATMODE '$HEATMODE' (dualpin | share)"; exit 1 ;;
  esac
fi
TAG=x29-t2-$(echo $PIECES | tr ' ' '-'); [ -n "$PIECES" ] || TAG=x29-t2-none
REFTAG=x29-best524
ENVS="RADIANCE_DS41_INDEX=${RADIANCE_DS41_INDEX-unset} RADIANCE_DS41_IDXKEY=${RADIANCE_DS41_IDXKEY-unset} RADIANCE_DS41_LATENT=${RADIANCE_DS41_LATENT-unset} RADIANCE_DS41_CTAIL=${RADIANCE_DS41_CTAIL-unset} RADIANCE_HEAT_PREFILL=${RADIANCE_HEAT_PREFILL-unset}"
HEATENV=$(env | grep -E '^RADIANCE_(HEAT_(CREDIT_CAP|MOVES|PREFILL_DECAY|PREFILL_GAIN|DECODE_PIN)|PLACE_TURNOVER)=' | sort | tr '\n' ' ')
# what the serve log's indexer note must say for these pieces
XKEY=$(has mx4 && echo mxfp4 || echo bf16); XLAT=$(has latent && echo nvfp4 || echo bf16); XCT=$(has ctail && echo bound || echo full)
log "RESULT x29 61b-exl3-29-tuned2: start PIECES [$PIECES] IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) $ENVS ${HEATENV}MODEL=$MODELDIR/$MODEL HP29=$HP29 GATE29=$GATE29 yaml $Y"
if [ "$DRY" != 1 ]; then
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT x29 61b: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN; do
    grep -q "^ *$v:" $Y || { log "RESULT x29 61b: VIRHE $Y does not pass $v"; exit 1; }
  done
  [ -f $MODELDIR/$MODEL ] || { log "RESULT x29 61b: VIRHE $MODELDIR/$MODEL missing"; exit 1; }
  [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT x29 61b: VIRHE $MODELDIR/$MODEL.sha256ok missing (copy not verified)"; exit 1; }
  [ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT x29 61b: VIRHE production is running"; exit 1; }
  [ -s $O/guard/$REFTAG-warm-q.txt ] || log "RESULT x29 61b: note: no 60 warm-up answers ($O/guard/$REFTAG-warm-q.txt); identity not checked"
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
[ $rc = 1 ] && { log "RESULT x29 61b: VIRHE not healthy ($(desc)); start log $O/start-$TAG.log: $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; exit 1; }
[ $rc = 2 ] && { log "RESULT x29 61b: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
if [ "$DRY" != 1 ]; then
  ML=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer')
  log "RESULT x29 61b $TAG modes: $(echo "$ML" | sed 's/^radiance: //' | cut -c1-420) | $(docker logs $CN 2>&1 | grep -m1 'heat engine: a step' | cut -c1-200)"
  for want in "keys $XKEY " "latent $XLAT " "compressor tail $XCT "; do
    case "$ML" in *"$want"*) ;; *) log "RESULT x29 61b: VIRHE the serve log's indexer note lacks '$want' for PIECES [$PIECES] -> stopped"; down; exit 1 ;; esac
  done
fi
warmq $TAG
identq $TAG $REFTAG
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
run "python3 $CMP $L $TAG $REFTAG >> $O/$TAG-cmp.txt 2>&1"
[ "$DRY" = 1 ] || log "$(tail -1 $O/$TAG-cmp.txt | cut -c1-1500) [tuned2 PIECES $PIECES vs stock 1.17]"
log "RESULT x29 61b $TAG: done | ref 60 $REFTAG (stock 1.17, same .rad + flags) | $REF_X117_524"
unset RADIANCE_DS41_INDEX RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_HEAT_PREFILL \
      RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN
down
log "RESULT x29 61b-exl3-29-tuned2: done"
