#!/bin/bash
# 42-best-hp12.sh = 41-best-nst0.sh (MTP off) with host pool 12288 and --max-model-len 524288 (user 6.11. 14.55: "vaikka"): experts in VRAM ~17.7 GiB/card as on 3.11. instead of 8.6
# 41-best-nst0.sh = 40-best.sh with NST=0 forced after best.env (user 6.11. 13.55: "kiinnostaa mtp pois tämänhetken parhaalla")
# 40-best.sh (queue item, 6.11.2026) -- the combination 30-tune512.sh chose (runs/pr266/best.env; queue/40-best.override.env,
# if present, is read instead -- same variables) on the stock PR #266 head image and the UD-IQ1_S .rad, with the full standard
# set: gate, stepprobe essay/code 512, mb c4/c8/pf/turns, 2 x cold ~16k prefill, then cc_sim u4 only (3.11. confirm args:
# --duration 660 --warmup 300 --effort xhigh,medium; user: no u8). If the combination does not start, the single row with the
# highest c4 (FB_* in best.env) is run instead and logged as such.
# Estimate ~35 min (start + warm-up 4.5; probe 1; mb 10.3; 16k 1; cc_sim 11 + start-up margin).
# DRYRUN=1 bash 40-best.sh : prints, writes nothing.
. "$(cd "$(dirname "$0")" && pwd)/../lib-pr266.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-pr266.sh" || { echo "lib-pr266.sh not found"; exit 1; }
TAG=best-hp12
guard_lim() { echo 24; }   # 6.11.: NST 0 on this head is not the 3.11. 15 ms class (18 ms limit); 24 as for the EXL3 instances
B=$O/best.env; [ -e $QD/40-best.override.env ] && B=$QD/40-best.override.env
if [ ! -e $B ]; then
  [ "$DRY" = 1 ] && { echo "+ (dry) $B not there yet: would source it"; B=/dev/null; } || { log "RESULT pr266 42-best-hp12: VIRHE no $O/best.env (30-tune512 not run?)"; exit 1; }
fi
. $B
NST=0; HOSTPOOL=12288; MML=524288; export HOSTPOOL NGRAM CKINT MNBT NST MAXSEQS MML XARGS   # 41: the chosen combination with MTP off (user 6.11.)
log "RESULT pr266 42-best-hp12: start from $B: $(desc)"
evict
gate $TAG $(guard_lim); rc=$?
if [ $rc = 1 ] && [ -n "${FB_TAG:-}" ]; then
  log "RESULT pr266 42-best-hp12: the combination did not start -> fallback to the single best row $FB_TAG"
  HOSTPOOL=$FB_HOSTPOOL NGRAM=$FB_NGRAM CKINT=$FB_CKINT MNBT=$FB_MNBT NST=$FB_NST MAXSEQS=$FB_MAXSEQS MML=$FB_MML XARGS=$FB_XARGS
  export HOSTPOOL NGRAM CKINT MNBT NST MAXSEQS MML XARGS
  gate $TAG $(guard_lim); rc=$?
fi
[ $rc = 1 ] && { log "RESULT pr266 42-best-hp12: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT pr266 42-best-hp12: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || pf16 $TAG
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 pr266-best-hp12 > $O/best-hp12-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
log "RESULT pr266 42-best-hp12: done ($(desc)) | refs: $REF_311_NST0 || 3.11. cc_sim u4 fin-best: TTFT p50/p95 3.92/7.41 s, 29.1 tok/s/agent, 92.5 total || $REF_BAKON"
