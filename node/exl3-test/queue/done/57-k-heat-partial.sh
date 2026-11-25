#!/bin/bash
# 70-heat-prefill.sh (queue item, 6.11.2026, queue-ar) -- what slows 1-user decode after a long cold prompt: the heat engine
# turning the resident experts towards the prompt's (a transient), or finished sessions' KV held in VRAM taking back the
# experts' KV loan (lasts until the KV is evicted)? And does RADIANCE_HEAT_PREFILL=0 remove the first? EXL3 2.25 bpw, TP4,
# MML 524288.
# Evidence so far: prof0 (6.11.) decode right after a 243k cold prefill: routed EXL3 8.3 ms/step, MoE all-reduce wait 10.7
# (all 4 ranks alike), falling to 5.0 / 4.4-5.4 within ~200 steps; at the same depth after the hit turn (KV still held) = 1u.
# Nothing else moved (gemm, index_score within 0.1 ms). The heat engine credits every prefill chunk up to its cap (512
# selections a rank and layer, ~340 decode steps' worth at TP4) and decays the decode heat once a chunk. The other agent's
# reading of 50 (probe 22.2/22.4 vs 19.8 after 135k+266k, 47 flags, no RAM tier): those sessions' KV (~1.3 GiB/card) stays in
# VRAM and takes ~150 of the 705 loaned expert slots until evicted.
# Image X3HEAT = -idx's kernels/plugins + bin/radiance of ds41-ar 058faddc: RADIANCE_HEAT_PREFILL=p makes a step with prefill
# rows count as its decode rows + p x its prefill rows (decay and credit); unset = the -idx engine, bit for bit. Placement does
# not change outputs -> exact; arm p0 checks the copy output byte for byte against 50's -idx output.
#
# A (47 flags: MAXSEQS 4 NST0 ckint 8192, no RAM tier, + --live), arms ARMS_A (default "p1 p0"; p1 = env unset, p0 = 0):
#   gate (24 ms); P0 probe fresh; onem 262144 (82k calibration, a 266k cold prompt + 256 tokens, its hit turn);
#   P1 probe at once (essay = decode steps 0-511 after, code = 512-1023);
#   fastcheck 384 (an unrelated short request: 384 more decode steps; the heat has recovered, the 266k KV is still held);
#   P2 probe;  POST /reset_prefix_cache (the held KV back to the pool, re-lendable to experts) + fastcheck 384;  P3 probe;
#   p0 only: ident (copy over ~143k, byte compare vs runs/xfix/copy-xfix-idx-run1.txt).
#   Reading: onem's 266k decode ms/step (its 256 tokens right after the prompt) p1 vs p0 = the transient itself (prof0: it
#   lasted ~200 steps, so P1 holds only its tail plus the hit turn's chunk); P1 > P2 = what is left of it; P2 > P3 (and the
#   live vram count rising after the reset) = the KV loan; p0's onem decode / P1 at p1's P2 level = the share removes it.
# B (fin-best: MAXSEQS 8, RAM tier --prefix-cache-mib 86016, ckint 2048, span 2, IL 8, + --live, production-like), arms ARMS_B
#   (default "p0"): gate (21 ms); P0; onem 262144; P1; fastcheck 384; P2; mb c4, c8 (MB="" skips; refs 50 best524 c4 95.5
#   tok/s 35.6 ms, c8 123.4 tok/s 55.6 ms); cc_sim u4; probe after.
# Engine log followed with marks; profsum.py live per phase: expert tiers (vram = slab + KV loan), hit %, promotions, decode-
# step stream MB (experts read across the link) -> RESULT "xar ... live".
# Refs (no earlier number has this sequence; p1 is its control): REF_50 (deep 266k 30.3 ms/step, probe after a longer set
# 22.2/22.4), REF_BEST (probe 21.0/19.8), REF_55 (cc_sim u4 79.2 tok/s total, 22.7 / agent, TTFT p50 3.94 s; 58 ccref on -idx
# if it has run).
# Time: A ~14 min an arm (+3 for p0's copy), B ~35 min (27 without mb) -> ~66 min with the defaults. DRYRUN=1 bash 70-heat-prefill.sh prints,
# writes nothing. Needs lib-ar.sh + ds41-ar-tp4.yaml (+ lib-fix.sh, lib-117.sh, lib-pr266.sh) in queue/, image X3HEAT on gpu02.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
export IMAGE=$X3HEAT
ARMS_A=${ARMS_A-"p03 p01 c64"}; ARMS_B=${ARMS_B-""}; MB=${MB-"c4 c8"}
log "RESULT xar 70-heat: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A [$ARMS_A] B [$ARMS_B]"
imgcheck $IMAGE
[ "$DRY" = 1 ] || [ -e $IDREF ] || log "RESULT xar 70: note: $IDREF missing, the copy is kept but not compared"
mkd $O
evict
setarm() { unset RADIANCE_HEAT_CREDIT_CAP; case $1 in p0) export RADIANCE_HEAT_PREFILL=0 ;; p1) unset RADIANCE_HEAT_PREFILL ;; p01) export RADIANCE_HEAT_PREFILL=0.1 ;; p03) export RADIANCE_HEAT_PREFILL=0.3 ;; c64) unset RADIANCE_HEAT_PREFILL; export RADIANCE_HEAT_CREDIT_CAP=64 ;; *) return 1 ;; esac; }
# the shared sequence: P0, one 266k cold prompt, P1, 384 unrelated steps, P2 (marks into $P)
seq_deep() {   # TAG
  xprobe $1-p0 "P0 fresh"; mark p0 $P
  xdeep $1 262144; mark onem $P
  xprobe $1-p1 "P1 at once after the 266k prompt + its hit turn"; mark p1 $P
  fcrun $1-fc1 "(384 unrelated decode steps)"; mark fc1 $P
  xprobe $1-p2 "P2 after 384 more steps (KV still held)"; mark p2 $P
}
for arm in $ARMS_A; do
  setarm $arm || { log "RESULT xar 70: unknown arm $arm"; continue; }
  P=$O/heat-$arm-pf; mkd $P
  pfflags; export XARGS="--live"
  gate xar-$arm 24; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 70 A $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 70 A $arm: VIRHE slow twice -> skipped"; else
    heatmode xar-$arm
    follow $P; mark start $P
    seq_deep xar-$arm
    resetpc xar-$arm; fcrun xar-$arm-fc2 "after reset_prefix_cache"; mark reset $P
    xprobe xar-$arm-p3 "P3 after reset_prefix_cache + 384 steps"; mark p3 $P
    if [ $arm = p0 ] && ! stopfile; then
      ident xar-$arm; mark ident $P
      [ -e $IDREF ] && identref xar-$arm
    fi
    unfollow
    run "docker logs $CN > $P/engine-full.log 2>&1"
    livesum $P xar-$arm-pf
  fi
  stopfile && { down; exit 0; }
done
for arm in $ARMS_B; do
  setarm $arm || { log "RESULT xar 70: unknown arm $arm"; continue; }
  P=$O/heat-$arm-best; mkd $P
  best_vars; export XARGS="$XARGS --live"
  gate xar-$arm-best 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 70 B $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 70 B $arm: VIRHE slow twice -> skipped"; else
    heatmode xar-$arm-best
    follow $P; mark start $P
    seq_deep xar-$arm-best
    for ph in $MB; do stopfile && break; mbp xar-$arm-best $ph; mark mb-$ph $P; done
    if ! stopfile; then
      cc xar-$arm-best; mark ccsim $P
      xprobe xar-$arm-best-cc "after cc_sim u4"; mark probe-cc $P
    fi
    unfollow
    run "docker logs $CN > $P/engine-full.log 2>&1"
    livesum $P xar-$arm-best
    [ "$DRY" = 1 ] || log "RESULT xar 70 B $arm cc_sim u4: $(grep -h '^RESULT cc_sim' $O/xar-$arm-best-ccsim-u4.out 2>/dev/null | cut -c1-330) | ref $REF_55"
  fi
  stopfile && { down; exit 0; }
done
unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP
down
log "RESULT xar 70-heat: done -- $O/heat-<arm>-{pf,best}/live.txt, probes / onem / fastchecks / copy in $O | refs: $REF_50"
