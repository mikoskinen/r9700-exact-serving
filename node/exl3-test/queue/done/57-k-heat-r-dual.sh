#!/bin/bash
# 74-heat-dual.sh (queue item, 6.11.2026, queue-ar) -- the dual heat: keep a prompt's experts resident while it prefills (p1's
# prefill rate) and give the VRAM back to the decode set within tens of steps after it (p0's decode). EXL3 2.25 bpw, TP4,
# MML 524288.
# Why (70, 47 flags, onem 262144: prefill / deep decode / code probe right after / decode-step stream after):
#   p1 (one heat, as shipped)   2512 tok/s / 28.6 ms/step / 21.4 ms / 177 MB
#   p0 (prefill share 0)        1736 (-31 %) / 23.5 (-18 %) / 20.6 / 154; gate 21.3 vs 19.7 (the warm-up prefill heats nothing)
#   p0.3 / p0.1 / cap64 sit at p1's numbers: the scalar share is a cliff, not a dial.
# The fix (ds41-ar bb654442): RADIANCE_HEAT_PREFILL_DECAY=d turns on two heats per unit, ranked by their sum. Decode rows credit
# the decode heat (decaying 0.95 with them, untouched by a prompt); prefill rows credit a prefill heat that decays by d at every
# step. While a prompt prefills its chunks hold the prefill heat at ~1/(1-d) chunk credits (d 0.7: ~17 for a median prompt
# expert against <= ~4 for the hottest decode experts), so the ranking follows the prompt as p1's does; after it the prefill
# heat falls below the decode heat in ~5-8 steps (0.7^n) and the moves put the decode set back. Exact (placement only).
# Image X3DUAL = radiance-local:ds41-exl3-117-bb654442-f2f3c3f7-idx-heat (3f614141e1eb): -idx kernels/plugins + bin/radiance of
# ds41-ar bb654442 (058faddc's share + 7cecf223's move budget + the dual heat; all unset = the -idx engine).
#   A (47 flags MAXSEQS 4 NST0 ckint 8192, no RAM tier, + --live), arms ARMS_A (default "d70 d85"):
#     d70 = RADIANCE_HEAT_PREFILL_DECAY 0.7, d85 = 0.85 (slower: steadier prompt residency, ~2x the return)
#     gate (23 ms); P0 probe fresh; onem 262144 (82k calibration, 266k cold prompt + 256 tokens, hit turn); P1 probe at once;
#     fastcheck 384; P2; POST /reset_prefix_cache + fastcheck 384; P3; first arm only: ident (copy over ~143k vs 50 -idx).
#   B (fin-best: MAXSEQS 8, RAM tier, ckint 2048, span 2, IL 8, + --live), arms ARMS_B (default "d70"): gate (23 ms); P0;
#     onem 262144; P1; fastcheck 384; P2; cc_sim u4; probe after.
#   live.txt per phase (profsum.py live): vram expert count, hit %, promotions, decode-step stream MB.
# Refs, no re-runs: 70 p1 / p0 above (arm A, same sequence and flags; their RESULT lines in runs/ds41-exl3.log), 70 arm B p0
# (fin-best p0 + cc_sim), 55 cc_sim u4 (79.2 tok/s total, 22.7 / agent, TTFT 3.94 s), 50 best524 (probe 21.0 / 19.8).
# Reading: d70 wins if onem prefill ~2500 tok/s (p1) AND deep decode / P1 near p0's (23.5 / 20.6) and the gate near 19.7.
# ~16 min an A arm (+3 for the copy), ~27 min B. DRYRUN=1 bash 74-heat-dual.sh prints, writes nothing.
# Needs lib-ar.sh + ds41-ar-tp4.yaml (updated: passes RADIANCE_HEAT_PREFILL_DECAY / _GAIN) in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
X3DUAL=${X3DUAL:-radiance-local:ds41-exl3-117-bb654442-f2f3c3f7-idx-heat}
export IMAGE=$X3DUAL
ARMS_A=${ARMS_A-"d70 d85"}; ARMS_B=${ARMS_B-"d70"}
REF70="70 A (47 flags): p1 prefill 2512 tok/s, deep 28.6 ms/step, code probe after 21.4 ms, stream after 177 MB/step; p0 1736 / 23.5 / 20.6 / 154, gate 21.3"
log "RESULT xar 74-dual: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A [$ARMS_A] B [$ARMS_B]"
imgcheck $IMAGE
[ "$DRY" = 1 ] || [ -e $IDREF ] || log "RESULT xar 74: note: $IDREF missing, the copy is kept but not compared"
mkd $O
evict
setarm() {
  unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER RADIANCE_HEAT_PREFILL_GAIN
  case $1 in d70) export RADIANCE_HEAT_PREFILL_DECAY=0.7 ;; d85) export RADIANCE_HEAT_PREFILL_DECAY=0.85 ;;
             d50) export RADIANCE_HEAT_PREFILL_DECAY=0.5 ;; *) return 1 ;; esac
}
seq_deep() {   # TAG -- 70's sequence
  xprobe $1-p0 "P0 fresh"; mark p0 $P
  xdeep $1 262144; mark onem $P
  xprobe $1-p1 "P1 at once after the 266k prompt + its hit turn"; mark p1 $P
  fcrun $1-fc1 "(384 unrelated decode steps)"; mark fc1 $P
  xprobe $1-p2 "P2 after 384 more steps (KV still held)"; mark p2 $P
}
first=1
for arm in $ARMS_A; do
  setarm $arm || { log "RESULT xar 74: unknown arm $arm"; continue; }
  P=$O/heat-$arm-pf; mkd $P
  pfflags; export XARGS="--live"
  gate xar-$arm 23; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 74 A $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 74 A $arm: VIRHE slow twice -> skipped"; else
    heatmode xar-$arm
    follow $P; mark start $P
    seq_deep xar-$arm
    resetpc xar-$arm; fcrun xar-$arm-fc2 "after reset_prefix_cache"; mark reset $P
    xprobe xar-$arm-p3 "P3 after reset_prefix_cache + 384 steps"; mark p3 $P
    if [ $first = 1 ] && ! stopfile; then
      ident xar-$arm; mark ident $P
      [ -e $IDREF ] && identref xar-$arm
    fi
    first=0
    unfollow
    run "docker logs $CN > $P/engine-full.log 2>&1"
    livesum $P xar-$arm-pf
    [ "$DRY" = 1 ] || log "RESULT xar 74 A $arm vs refs: $REF70"
  fi
  stopfile && { down; exit 0; }
done
for arm in $ARMS_B; do
  setarm $arm || { log "RESULT xar 74: unknown arm $arm"; continue; }
  P=$O/heat-$arm-best; mkd $P
  best_vars; export XARGS="$XARGS --live"
  gate xar-$arm-best 23; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 74 B $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 74 B $arm: VIRHE slow twice -> skipped"; else
    heatmode xar-$arm-best
    follow $P; mark start $P
    seq_deep xar-$arm-best
    if ! stopfile; then
      cc xar-$arm-best; mark ccsim $P
      xprobe xar-$arm-best-cc "after cc_sim u4"; mark probe-cc $P
    fi
    unfollow
    run "docker logs $CN > $P/engine-full.log 2>&1"
    livesum $P xar-$arm-best
    [ "$DRY" = 1 ] || log "RESULT xar 74 B $arm cc_sim u4: $(grep -h '^RESULT cc_sim' $O/xar-$arm-best-ccsim-u4.out 2>/dev/null | cut -c1-330) | refs 55 79.2 total, 22.7 / agent, TTFT 3.94 s; 70 B p0 in this log"
  fi
  stopfile && { down; exit 0; }
done
unset RADIANCE_HEAT_PREFILL_DECAY
down
log "RESULT xar 74-dual: done -- $O/heat-<arm>-{pf,best}/live.txt | refs: $REF70"
