#!/bin/bash
# 84-iqr-speed.sh (6.11.2026; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN; run AFTER 83-iqr-selftest
# passed) -- the 6.11. standard speed set on the IQR containers (DeepSeek-V4.1 routed experts as IQR-V4 computed trellis,
# moe_gemm_iqr; ds41-iqrmoe b77f7a85 image) with the x117-best524 serve line: TP4 exact, expert_tiered, --max-model-len 524288,
# MTP off (NST 0), MAXSEQS 8, ngram ram, RAM KV tier 84 GiB, ckint 2048 + span 2 + IL 8. The host pool per container keeps
# ~15.6 GiB of experts on each card (= the EXL3 2.25 row's 15.63): experts a card - pool + 0.65 GiB.
#   iqr215 136.1 GiB (34.0 a card) hp 19456 | iqr225 144.2 (36.1) hp 21504 | iqr2l 179.5 (44.9) hp 30720 | iqr3s 200.7 (50.2) hp 36096
# MODELS (default "iqr215 iqr225"): each: evict, gate (GATE_IQR ms, default 30), warm-up answers (ds41_tests.py correct --only-q,
# written by check), stepprobe essay/code 512, mb c4 / c8, cc_sim u4 (ccsim-exl3.sh, 660 s), engine log, down.
# References (runs/ds41-exl3.log): 6.11. x117-best524 EXL3 2.25 (hp 19456): guard 19.71 ms; essay 47.7 / code 50.6 tok/s; c4 95.0
# tok/s 35.8 ms; c8 123.4 56.1 ms; and the x29 rows of 60-exl3-29 (EXL3 2.9, hp 31488).
# DRYRUN=1 bash 84-iqr-speed.sh : prints only. Estimate ~35 min a model (start ~5 min x 1-2, probe 1, mb ~8, cc_sim 11, rest 3).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
for LIB in "$H/lib-iqr.sh" "$H/../lib-iqr.sh" "$HOME/gpu/ds41/exl3-test/queue-iqr/lib-iqr.sh"; do [ -e "$LIB" ] && break; done
. "$LIB" || { echo "lib-iqr.sh not found"; exit 1; }
MODELS=${MODELS:-iqr215 iqr225}
GATE_IQR=${GATE_IQR:-30}
guards iqr-speed || exit 1
if [ "$DRY" != 1 ]; then
  grep -q "RESULT iqr-selftest moe_gemm_iqr_ds41: [0-9]* ok, 0 FAIL" $L || { log "RESULT iqr-speed: VIRHE 83-iqr-selftest has not passed (no '0 FAIL' line for moe_gemm_iqr_ds41)"; exit 1; }
fi
log "RESULT iqr-speed: start MODELS='$MODELS'"
for m in $MODELS; do
  stopfile && break
  have $m || continue
  TAG=$m-best524
  export HOSTPOOL=$HP NGRAM=ram CKINT=2048 MNBT=8192 NST=0 MAXSEQS=8 MML=524288 WIRE=exact \
         XARGS="$RAMTIER --prefill-span 2 --prefill-interleave 8"
  evict
  gate $TAG $GATE_IQR; rc=$?
  [ $rc = 1 ] && { log "RESULT iqr-speed $m: VIRHE not healthy ($(desc)): $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; continue; }
  [ $rc = 2 ] && { log "RESULT iqr-speed $m: VIRHE slow state after one restart -> measurements skipped"; down; continue; }
  [ "$DRY" = 1 ] || log "RESULT iqr-speed $m warm-up answers: $(grep '^J ' $O/guard/$TAG-warm-q.txt 2>/dev/null | python3 -c 'import json,sys; print(" | ".join("%s %s: %r" % (d.get("test"), d.get("status"), (d.get("content") or "")[:60]) for d in (json.loads(l[2:]) for l in sys.stdin)))' 2>&1 | cut -c1-700)"
  probe $TAG
  for p in c4 c8; do stopfile && break; mbp $TAG $p; done
  stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"
  [ "$DRY" = 1 ] || log "RESULT iqr-speed $m cc_sim u4: $(grep -E 'TTFT|tok/s' $O/$TAG-ccsim-u4.out 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-500)"
  run "docker logs $CN > $O/$TAG-engine.log 2>&1"
  down
done
log "RESULT iqr-speed: done"
