#!/bin/bash
# 75-heat-pin.sh (queue item, 6.11.2026, queue-ar) -- dual heat + decode pin: the prompt's experts still come in while it
# prefills, but only into the part of the resident set the decode core does not hold, so decode after a long prompt does
# not have to win its set back. EXL3 2.25 bpw, TP4, MML 524288.
# Why (74 d70, 47 flags, onem 262144): prefill 2500.8 tok/s and gate 19.68 ms (both fixed) but deep decode 26.8 ms/step (p0
# 23.5): the prompt promoted 2151 units / demoted 2254 -- the whole resident set -- and after it only 1-2.5 promotions a step
# came back (live: +6..+25 a second). The guard needs hot >= cold x 1.25 + min_gain 1.5, and at TP4 one token credits ~1.5
# selections a rank: of the 2249-unit resident budget only the 141 hottest units a rank (33 % of decode selections) ever reach
# a steady decode heat of 1.5 (cc-x117 routing; the 2249th sits at 0.08), so the rest return only on bursts of routing.
# The fix (ds41-ar 7e342763): RADIANCE_HEAT_DECODE_PIN=f -- under dual heat the decode-hottest fraction f of each slab class's
# residents is no promotion's victim (thresholds once a step). Lowering min_gain instead was rejected: the replay
# (tools/heatreplay.py, the policy on cc-x117 routing) loses 1.8 points of steady decode hit at 0.4 / 0.1.
# Replay, 266k prompt (33 chunks), 1 rank of 2249 slots: prompt selections resident during it / decode hit over the 256 steps
# after it -- p1 95.7 / 90.2, p0 87.6 / 95.2, d70 95.6 / 92.5, d70 + pin 0.8 94.2 / 94.3, pin 0.9 94.0 / 94.6. Mapped linearly
# onto 70/74's measured pairs (1.02 ms a decode-hit point, 95.5 tok/s a prompt-resident point): pin 0.8 ~2370 tok/s prefill and
# ~24.4 ms/step deep decode, pin 0.9 ~2350 / ~24.1 (d70 2500.8 / 26.8, p0 1736 / 23.5). Exact (placement only).
# Image X3PIN = radiance-local:ds41-exl3-117-7e342763-f2f3c3f7-idx-heat (0748b9ad3577): 74's image + the pin (all unset = the
# -idx engine).
#   A (47 flags MAXSEQS 4 NST0 ckint 8192, no RAM tier, + --live), arms ARMS_A (default "p80 p90"): RADIANCE_HEAT_PREFILL_DECAY
#     0.7 + RADIANCE_HEAT_DECODE_PIN 0.8 / 0.9. 70's sequence: gate (23 ms); P0 fresh; onem 262144; P1; fastcheck 384; P2;
#     reset_prefix_cache + fastcheck; P3; first arm only: ident (copy over ~143k vs 50 -idx).
#   B (fin-best + --live), arms ARMS_B (default "p80"): gate (23 ms); P0; onem 262144; P1; fastcheck 384; P2; cc_sim u4; probe.
# Refs, no re-runs: 70 p1 / p0 and 74 d70 (arm A; RESULT lines in runs/ds41-exl3.log), 74 B d70 cc_sim (if it ran), 55 cc_sim u4.
# ~16 min an A arm (+3 for the copy), ~27 min B. DRYRUN=1 bash 75-heat-pin.sh prints, writes nothing.
# Needs lib-ar.sh + ds41-ar-tp4.yaml (updated: passes RADIANCE_HEAT_DECODE_PIN) in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
X3PIN=${X3PIN:-radiance-local:ds41-exl3-117-7e342763-f2f3c3f7-idx-heat}
export IMAGE=$X3PIN
ARMS_A=${ARMS_A-"p80 p90"}; ARMS_B=${ARMS_B-"p80"}
REF70="70/74 A (47 flags): p1 prefill 2512 tok/s, deep 28.6 ms/step, code probe after 21.4 ms, stream after 177 MB/step; p0 1736 / 23.5 / 20.6 / 154, gate 21.3; d70 2500.8 / 26.8 / 21.2, gate 19.68"
log "RESULT xar 75-pin: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A [$ARMS_A] B [$ARMS_B]"
imgcheck $IMAGE
[ "$DRY" = 1 ] || [ -e $IDREF ] || log "RESULT xar 75: note: $IDREF missing, the copy is kept but not compared"
mkd $O
evict
setarm() {
  unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER RADIANCE_HEAT_PREFILL_GAIN
  export RADIANCE_HEAT_PREFILL_DECAY=0.7
  case $1 in p80) export RADIANCE_HEAT_DECODE_PIN=0.8 ;; p90) export RADIANCE_HEAT_DECODE_PIN=0.9 ;;
             p70) export RADIANCE_HEAT_DECODE_PIN=0.7 ;; *) return 1 ;; esac
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
  setarm $arm || { log "RESULT xar 75: unknown arm $arm"; continue; }
  P=$O/heat-$arm-pf; mkd $P
  pfflags; export XARGS="--live"
  gate xar-$arm 23; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 75 A $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 75 A $arm: VIRHE slow twice -> skipped"; else
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
    [ "$DRY" = 1 ] || log "RESULT xar 75 A $arm vs refs: $REF70"
  fi
  stopfile && { down; exit 0; }
done
for arm in $ARMS_B; do
  setarm $arm || { log "RESULT xar 75: unknown arm $arm"; continue; }
  P=$O/heat-$arm-best; mkd $P
  best_vars; export XARGS="$XARGS --live"
  gate xar-$arm-best 23; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 75 B $arm: VIRHE not healthy ($(desc))"; exit 1; }
  if [ $rc = 2 ]; then log "RESULT xar 75 B $arm: VIRHE slow twice -> skipped"; else
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
    [ "$DRY" = 1 ] || log "RESULT xar 75 B $arm cc_sim u4: $(grep -h '^RESULT cc_sim' $O/xar-$arm-best-ccsim-u4.out 2>/dev/null | cut -c1-330) | refs 55 79.2 total, 22.7 / agent, TTFT 3.94 s; 70 B p0 in this log"
  fi
  stopfile && { down; exit 0; }
done
unset RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN
down
log "RESULT xar 75-pin: done -- $O/heat-<arm>-{pf,best}/live.txt | refs: $REF70"
