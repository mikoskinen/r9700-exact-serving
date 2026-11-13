#!/bin/bash
# 68-mx4l-latent.sh (queue item, 6.11.2026) -- the packed compressed latent + MXFP4 index keys on TP4:
# radiance-local:ds41-exl3-117-<ds41-mx4l sha8>-f2f3c3f7-mx4l = the mx4 engine + RADIANCE_DS41_LATENT=nvfp4: the
# compressed latent caches (kv sources 2/8/14 at ratio 2, 20 at ratio 1) hold each 512-value entry as 288 bytes (256 B
# e2m1 codes, 32 e4m3 scales, one a block of 16) instead of 1024 B bf16; with RADIANCE_DS41_IDXKEY=mxfp4 the index keys
# are 80 B instead of 256 B. Exact: the latent is quant_roundtrip(fp4_e2m1, e4m3, 16) output and the packer searches
# each block's e4m3 scale until every value comes back bit for bit (docs/OPS.md "Packed compressed latent"; emulator:
# attn_sparse_sink packed vs bf16 bit-identical on the decode band's direct and tiled forms, the scalar kernel and WMMA,
# ratio 1 and 2, subnormal-scale blocks included, libref too). Either switch unset = that cache in bf16.
# Expected (arithmetic, 2.5 compressed entries a token on every card for each cache): latent 2560 -> 720 B a token, index
# 640 -> 200 B, together 3200 -> 920 B a token: at MAXSEQS 8 x MML 524288 (4.19M tokens) ~13.4 -> ~3.9 GB of KV a card,
# ~9.5 GB a card back to the experts (lent while idle) or to more sequences; MAXSEQS 16 at MML 524288 then needs ~8 GB.
#   1. both on, the 47 flags (MML 524288 MAXSEQS 4 NST 0 ckint 8192), gated (24 ms): mode line, the startup "kv cache"
#      line, the identity copy -> byte comparison with 50's bf16 off arm (must be IDENTICAL), the cold set, onem
#      135000,262144, stepprobe
#   2. both on at the fin-best line (MAXSEQS 8, MML 524288), gated (21 ms): the "kv cache" line against 62's bf16 start
#      log and 67's keys-only start log at the same flags, stepprobe, mb c4 / c8
#   3. the capacity probe: both on, the fin-best line at MAXSEQS 16 (MML 524288), gated: the "kv cache" line (its
#      "N sessions at 524288") and a probe; if MAXSEQS 16 does not start, MAXSEQS 12
# References: 67 (mxfp4 keys), 62 (idx2, bf16) when they have run; else -idx (50): cold 2322.0 / 2381.9 / 2251.9 tok/s,
# deep 135k 35.18 tok/s, 266k 33.82; fin-best 19.72 ms, essay 47.7 / code 50.6, c4 95.5, c8 123.4.
# Estimate ~70 min (3-4 gated starts, ident 2, cold 5, onem 6, probes 3, mb c4/c8 6).
# DRYRUN=1 bash 68-mx4l-latent.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh),
# ds41-fix2-tp4.yaml (passes RADIANCE_DS41_IDXKEY and RADIANCE_DS41_LATENT), 50's runs/xfix/copy-xfix-idx-off-run1.txt.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3MX4L=${X3MX4L:-radiance-local:ds41-exl3-117-c528cdff-f2f3c3f7-mx4l}
export IMAGE=$X3MX4L
kvline() { grep -m1 -E '^I +kv cache +[0-9.]+ GiB a card' "$1" 2>/dev/null | sed 's/^I *//' | tr -s ' '; }
log "RESULT xfix 68-mx4l: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q RADIANCE_DS41_LATENT $Y || { log "RESULT xfix 68: VIRHE $Y does not pass RADIANCE_DS41_LATENT"; exit 1; }
mkd $O $STATSDIR
export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_DS41_LATENT=nvfp4
quit() { unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT; exit ${1:-0}; }
# 1. the 47 flags
pfflags
gate xfix-mx4l 24; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 68: VIRHE not healthy ($(desc)) -- see start-xfix-mx4l.log"; quit 1; }
[ $rc = 2 ] && { log "RESULT xfix 68: VIRHE slow state after one restart -> skipped"; down; quit 1; }
run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/xfix-mx4l-mode.txt"
[ "$DRY" = 1 ] || log "RESULT xfix 68 mode: $(cut -c1-320 $O/xfix-mx4l-mode.txt) | $(kvline $O/start-xfix-mx4l.log)"
ident xfix-mx4l
if [ "$DRY" = 1 ] || [ -e $O/copy-xfix-idx-off-run1.txt ]; then
  identcmp xfix-idx-off xfix-mx4l || { log "RESULT xfix 68: VIRHE the packed caches changed the output -> stopped"; run "docker logs $CN > $O/xfix-mx4l-engine.log 2>&1"; down; quit 1; }
else
  log "RESULT xfix 68: no 50 off-arm copy ($O/copy-xfix-idx-off-run1.txt) -- identity not checked"
fi
cold xfix-mx4l
stopfile && { down; quit 0; }
deep xfix-mx4l 135000,262144
probe xfix-mx4l
run "docker logs $CN > $O/xfix-mx4l-engine.log 2>&1"
stopfile && { down; quit 0; }
# 2. fin-best
best_vars
gate xfix-mx4l-best524 21; rc=$?
if [ $rc = 0 ]; then
  [ "$DRY" = 1 ] || log "RESULT xfix 68 kv at fin-best (MAXSEQS 8, MML 524288): both packed $(kvline $O/start-xfix-mx4l-best524.log) | keys only (67) $(kvline $O/start-xfix-mx4-best524.log) | bf16 (62) $(kvline $O/start-xfix-idx2-best524.log)"
  probe xfix-mx4l-best524
  for p in c4 c8; do stopfile && break; mbp xfix-mx4l-best524 $p; done
  run "docker logs $CN > $O/xfix-mx4l-best524-engine.log 2>&1"
else
  log "RESULT xfix 68 best: skipped (gate rc $rc)"
fi
stopfile && { down; quit 0; }
# 3. capacity: MAXSEQS 16 at MML 524288 (12 if 16 does not start)
for ms in 16 12; do
  best_vars; export MAXSEQS=$ms
  gate xfix-mx4l-cap$ms 21; rc=$?
  [ "$DRY" = 1 ] || log "RESULT xfix 68 capacity MAXSEQS $ms MML 524288 (both packed): gate rc $rc | $(kvline $O/start-xfix-mx4l-cap$ms.log) | $(grep -m1 -E 'sessions at' $O/start-xfix-mx4l-cap$ms.log 2>/dev/null | sed 's/^I *//' | tr -s ' ' | cut -c1-200)"
  if [ $rc = 0 ]; then probe xfix-mx4l-cap$ms; break; fi
done
down
log "RESULT xfix 68-mx4l: done"
quit 0
