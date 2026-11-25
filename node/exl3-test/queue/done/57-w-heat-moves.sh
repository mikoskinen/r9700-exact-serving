#!/bin/bash
# 72-heat-moves.sh (queue item, 6.11.2026, queue-ar) -- more experts in VRAM from the KV loan by a smaller heat-engine move budget,
# paired with the prefill share (70's p0), at the fin-best line (EXL3 2.25 bpw, TP4, MML 524288, RAM tier).
# Why: the mover keeps 3 x moves - 32 lent slots a slab class free for swaps to land in; at the shipped 64 moves and two slab
# classes that is 320 slots, 2.84 GiB a card of KV loan holding no expert (start log: "1778 slab slots (64 shadow, 320 more kept
# free out of a loan)"). Resident with the KV idle = slab - shadow + loan - 320 = 1714 + 535 = 2249 (live0 measured 2283).
# RADIANCE_HEAT_MOVES=16 -> 16 kept free a class: +288 resident slots; 32 -> 64 a class: +192. Model (tools/balsim.py, cc-x117
# decode routing): 2249 -> 2537 slots = slowest-rank routed MoE 8.01 -> 6.58 ms/step (-1.4 ms; ideal hit 96.4 -> 98.0 %).
# Risk: a quarter of the swaps a step re-converges 4x slower after the routing shifts (upstream Qwen replay: 77.1 % hit at 64,
# 74.1 % at 32) -- which is why it pairs with RADIANCE_HEAT_PREFILL=0 (prompts no longer shift the heat).
# Image X3HEAT2 = radiance-local:ds41-exl3-117-7cecf223-f2f3c3f7-idx-heat (6e71ab6409bc): -idx's kernels/plugins + bin/radiance
# of ds41-ar 7cecf223 (058faddc's prefill share + RADIANCE_HEAT_MOVES / RADIANCE_PLACE_TURNOVER; all unset = the -idx engine).
#   arms (ARMS, default "p0m16 p0m16n4160"; optional p0m32):
#     p0m16       RADIANCE_HEAT_PREFILL=0 RADIANCE_HEAT_MOVES=16
#     p0m16n4160  the same + --max-num-batched-tokens 4160 (71's m4160: arena -1.75 GiB, all-reduce scratch -0.34 GiB; the
#                 shared budget 2112 still holds a 2048-row interval chunk + 64 decode rows, the lone span chunk 4096)
#     p0m32       RADIANCE_HEAT_PREFILL=0 RADIANCE_HEAT_MOVES=32
#   each: gate (21 ms), probe, mb c4 / c8 / pf / turns, cc_sim u4 (CC=0 skips), probe after; --live: profsum.py live per phase
#   (vram expert count = resident, hit %, promotions, decode-step stream MB); a RESULT line with the mover / heat start lines.
# Refs (no re-runs): 70 arm B (fin-best + RADIANCE_HEAT_PREFILL=0: probe, onem, mb c4/c8, cc_sim u4), 71 m4160 (MNBT 4160 alone),
# 50 xfix-idx-best524 (probe 21.0 / 19.8 ms; c4 95.5 tok/s 35.6 ms; c8 123.4 55.6 ms; pf 1260.3 kept 0.231; turns cold 28.67 hit
# 2.08 s), 55 cc_sim u4 (79.2 tok/s total, 22.7 / agent, TTFT 3.94 s).
# Exact: placement only. ~27 min an arm. DRYRUN=1 bash 72-heat-moves.sh prints, writes nothing. Needs lib-ar.sh + ds41-ar-tp4.yaml.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
X3HEAT2=${X3HEAT2:-radiance-local:ds41-exl3-117-7e342763-f2f3c3f7-idx-heat}   # 6.11. 21:00 coordinator: dual heat (bb654442 carries 7cecf223 moves too)
export IMAGE=$X3HEAT2
ARMS=${ARMS-"p80m16n4160 p80n4160"}; CC=${CC:-1}
log "RESULT xar 72-moves: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS]"
imgcheck $IMAGE
mkd $O
evict
for arm in $ARMS; do
  best_vars
  unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER RADIANCE_HEAT_CREDIT_CAP RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN RADIANCE_HEAT_DECODE_PIN
  case $arm in
    p0m16)      export RADIANCE_HEAT_PREFILL=0 RADIANCE_HEAT_MOVES=16 ;;
    p0m16n4160) export RADIANCE_HEAT_PREFILL=0 RADIANCE_HEAT_MOVES=16 MNBT=4160 ;;
    p0m32)      export RADIANCE_HEAT_PREFILL=0 RADIANCE_HEAT_MOVES=32 ;;
    d70m16)     export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_MOVES=16 ;;
    d70m16n4160) export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_MOVES=16 MNBT=4160 ;;
    p80m16n4160) export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 MNBT=4160 ;;
    p80n4160)   export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 MNBT=4160 ;;
    *) log "RESULT xar 72: unknown arm $arm"; continue ;;
  esac
  export XARGS="$XARGS --live"
  t=xar-$arm
  gate $t ${GATEB:-23}; rc=$?   # 6.11. 20:00: heat p0 starts at ~21.3 ms (warm-up no longer heats VRAM)
  [ $rc = 1 ] && { log "RESULT xar 72 $arm: VIRHE not healthy ($(desc))"; continue; }
  [ $rc = 2 ] && { log "RESULT xar 72 $arm: VIRHE slow twice -> skipped"; continue; }
  [ "$DRY" = 1 ] || log "RESULT xar 72 $arm start: $(grep -m1 -E '^I\[0\] mover:' $O/start-$t.log | cut -c1-110) | $(grep -m1 -E '^I\[0\] heat engine: [0-9]+ movable' $O/start-$t.log | cut -c1-90) | $(grep -m1 'prefill share' $O/start-$t.log | cut -c1-80) | $(grep -E '^  activation arena|^    (experts|kv cache) ' $O/start-$t.log | tr -s ' ' | cut -c1-40 | tr '\n' ';')"
  P=$O/$t; mkd $P
  follow $P; mark start $P
  probe $t; mark probe $P
  for ph in c4 c8 pf turns; do stopfile && break; mbp $t $ph; mark mb-$ph $P; done
  if [ "$CC" = 1 ] && ! stopfile; then
    cc $t; mark ccsim $P
    xprobe $t-aftercc "after cc_sim u4"; mark probe-cc $P
  fi
  unfollow
  run "docker logs $CN > $P/engine-full.log 2>&1"
  livesum $P $t
  [ "$DRY" = 1 ] || log "RESULT xar 72 $arm cc_sim u4: $(grep -h '^RESULT cc_sim' $O/$t-ccsim-u4.out 2>/dev/null | cut -c1-300) | ref 55 79.2 total, 22.7 / agent, TTFT 3.94 s; 70 B p0 and 71 m4160 in this log"
  stopfile && { down; exit 0; }
done
unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER RADIANCE_HEAT_PREFILL_DECAY
down
log "RESULT xar 72-moves: done"
