#!/bin/bash
# 71-vram-mnbt.sh (queue item, 6.11.2026, queue-ar) -- flag-only VRAM for experts at the fin-best line (EXL3 2.25 bpw, TP4,
# MML 524288): a smaller --max-num-batched-tokens. At MNBT 8192 the activation arena is 4.31 GiB a card (0.83 fixed + 0.425 MiB a
# token) and the all-reduce scratch ~0.69 GiB (8 slots of MNBT x 5120 bf16, inside "already held"), but no step carries more
# than one 4096-row chunk (ckint 2048 x span 2) + 8 decode rows. The catch is --prefill-span's shared budget
# (core/sched/geometry.cpp prefill_shared_tok = max(ckint, MNBT - (span-1) x ckint)): a step that also decodes may take only that,
# so below MNBT 6144 + decode rows the chunk under load drops to 2048 (6.11. sweep n0mnbt4096, MML 262144: pf under load 1065.9
# vs 1284.4 tok/s; but experts 17.44 -> 19.68 GiB, arena 3.90 -> 2.16 GiB, held 946.8 -> 592.8 MiB, c4 31.9 vs 34.5 ms, c8 50.0
# vs 54.5 ms). At MML 524288 the freed memory goes to the KV pool and reaches the experts through the loan while the KV is idle.
#   arms (ARMS, default "m6208 m4160"):
#     m6208    MNBT 6208: shared budget 4160 = the 4096 chunk + 64 decode rows, prefill unchanged; frees ~1.07 GiB a card
#              (arena -0.90, all-reduce scratch -0.17) -> model (tools/balsim.py, cc-x117 routing) +118 slots ~ -0.65 ms/step
#     m4160    MNBT 4160: frees ~2.09 GiB (arena -1.75, scratch -0.34) -> +230 slots ~ -1.2 ms/step; chunks under load 2048
#     m4160hp  (optional) m4160 + HOSTPOOL 17408: the same memory in the expert slab itself instead of the loan
#   Image X3HEAT with the heat knob unset (= the -idx engine, so 50's xfix-idx-best524 rows are the reference) + --live.
#   each: gate (21 ms), probe, mb c4 / c8 / pf / turns, cc_sim u4 (CC=0 skips), probe after; a RESULT line gives the VRAM split.
# Refs (50 xfix-idx-best524, MNBT 8192, same image libs): guard 19.72; probe essay 21.0 / code 19.8 ms; c4 95.5 tok/s 35.6 ms;
# c8 123.4 tok/s 55.6 ms; pf 1260.3 tok/s kept 0.231 step 175.4 ms; turns cold 28.67 hit 2.08 s; start log: arena 4.31 GiB,
# held 978.82 MiB, experts 15.63 GiB, KV 7.92 GiB. cc_sim u4 ref: 55 (base image) 79.2 tok/s total, 22.7 / agent, TTFT 3.94 s.
# ~25 min an arm (12 without cc_sim). DRYRUN=1 bash 71-vram-mnbt.sh prints, writes nothing. Needs lib-ar.sh + ds41-ar-tp4.yaml.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
export IMAGE=$X3HEAT
unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP
ARMS=${ARMS-"m6208 m4160"}; CC=${CC:-1}
REF50="50 xfix-idx-best524 (MNBT 8192): probe 21.0 / 19.8 ms; c4 95.5 tok/s 35.6 ms; c8 123.4 55.6 ms; pf 1260.3 tok/s kept 0.231; turns cold 28.67 hit 2.08 s; arena 4.31 GiB, held 978.82 MiB, experts 15.63 GiB, KV 7.92 GiB"
log "RESULT xar 71-vram: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS]"
imgcheck $IMAGE
mkd $O
evict
for arm in $ARMS; do
  best_vars
  case $arm in
    m6208)   export MNBT=6208 ;;
    m4160)   export MNBT=4160 ;;
    m4160hp) export MNBT=4160 HOSTPOOL=17408 ;;
    *) log "RESULT xar 71: unknown arm $arm"; continue ;;
  esac
  export XARGS="$XARGS --live"
  t=xar-$arm
  gate $t 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xar 71 $arm: VIRHE not healthy ($(desc))"; continue; }
  [ $rc = 2 ] && { log "RESULT xar 71 $arm: VIRHE slow twice -> skipped"; continue; }
  [ "$DRY" = 1 ] || log "RESULT xar 71 $arm vram: $(grep -E '^  (already held|activation arena|step staging)|^    (experts|kv cache) ' $O/start-$t.log | tr -s ' ' | cut -c1-60 | tr '\n' ';') $(grep -m1 -E '^I\[0\] mover:' $O/start-$t.log | cut -c1-110) | $(grep -m1 'slots can be lent' $O/start-$t.log | cut -c3-90)"
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
  [ "$DRY" = 1 ] || log "RESULT xar 71 $arm cc_sim u4: $(grep -h '^RESULT cc_sim' $O/$t-ccsim-u4.out 2>/dev/null | cut -c1-300) | ref 55 79.2 total, 22.7 / agent, TTFT 3.94 s"
  stopfile && { down; exit 0; }
done
down
log "RESULT xar 71-vram: done | ref $REF50"
