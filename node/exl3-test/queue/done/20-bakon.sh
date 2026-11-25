#!/bin/bash
# 20-bakon.sh (queue item, 6.11.2026) -- the stock PR #266 head image (radiance-local:ds41-pr266-9c7ce498, Radiance 1.17.0 base,
# no local changes) with bakon's model and his TP4 serve line from docs/deepseek41-wip.md:
#   --tp 4 --tp-wire exact --placement expert_tiered --host-pool-mib 4096 --ngram-placement disk
#   --max-num-batched-tokens 8192 --checkpoint-interval 8192 --kv-cache-dtype bf16
# no --num-speculative-tokens (auto = DSpark), no --max-num-seqs / --max-model-len (engine defaults 256 / training context).
# --row-prefetch / --mapped-rewarm: documented in that file but not parsed by this head's config.cpp (row read-ahead on, rewarm
# off in code), so not passed. Model ~/models/ds41/deepseek-v4.1-flash-1017-uncensored.rad (3.11. UD-IQ1_S, 269.65 GB): this
# head reads it (same format version 2; rad-info of the identical file on the workstation passes, header + tail bytes md5-equal).
# One addition: --expert-stats /stats/bakon.json (counts from the routing histograms the core already copies back: no device
# work, no sync) feeds 30-tune512's --expert-profile rows.
# Fallback (logged as a deviation): host pool 4096 OOM-stopped startup on 3.11. (acc9ae58, MAXSEQS 4: 3.76 GiB/rank of experts
# left over), so if it does not start, 8192 and then 12288 are tried; the host pool that started is written to runs/pr266/base.env.
# Steps: evict, gate (DSpark limit 60 ms), stepprobe essay/code 512, mb c4/c8/pf/turns, 2 x cold ~16k prefill.
# Estimate 20-26 min (one start ~3 min + warm-up 1.5; probe 1; c4 3.6; c8 3.7; pf 1.5; turns 1.5; 16k 1; +6 min per fallback).
# DRYRUN=1 bash 20-bakon.sh : prints, writes nothing.
. "$(cd "$(dirname "$0")" && pwd)/../lib-pr266.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-pr266.sh" || { echo "lib-pr266.sh not found"; exit 1; }
TAG=bakon
log "RESULT pr266 20-bakon: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) MODEL=$MODELDIR/$MODEL $(stat -c %s $MODELDIR/$MODEL 2>/dev/null) B"
[ "$DRY" = 1 ] || docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT pr266 20-bakon: VIRHE image $IMAGE not on gpu02"; exit 1; }
[ "$DRY" = 1 ] || [ -f $MODELDIR/$MODEL ] || { log "RESULT pr266 20-bakon: VIRHE $MODELDIR/$MODEL missing"; exit 1; }
mkd $O $STATSDIR
evict
ok=0
for HOSTPOOL in 4096 8192 12288; do
  export HOSTPOOL NGRAM=disk CKINT=8192 MNBT=8192 NST= MAXSEQS= MML= XARGS="--expert-stats /stats/bakon.json"
  gate $TAG $(guard_lim); rc=$?
  [ $rc = 0 ] && { ok=1; break; }
  [ $rc = 2 ] && { log "RESULT pr266 20-bakon: VIRHE slow state after one restart (host pool $HOSTPOOL) -> measurements skipped"; down; exit 1; }
  log "RESULT pr266 20-bakon: host pool $HOSTPOOL did not start -> next"
done
[ $ok = 1 ] || { log "RESULT pr266 20-bakon: VIRHE no host pool of 4096/8192/12288 started"; down; exit 1; }
[ "$HOSTPOOL" = 4096 ] || log "RESULT pr266 20-bakon: DEVIATION from bakon's line: --host-pool-mib $HOSTPOOL (4096 did not start)"
run "printf 'HOSTPOOL=%s\n' $HOSTPOOL > $O/base.env"
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || pf16 $TAG
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
log "RESULT pr266 20-bakon: done (host pool $HOSTPOOL) | refs: $REF_BAKON || $REF_311_NST5 || $REF_311_NST0"
