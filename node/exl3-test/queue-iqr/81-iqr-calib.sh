#!/bin/bash
# 81-iqr-calib.sh (6.11.2026; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN) -- GPTQ's Hessians for the
# DeepSeek-V4.1 routed experts, recorded on the reference (the official release's MXFP4 experts as shipped, q8_0 dense, iq4_nl
# Engram: ds41-ref-q8dense-iq4engram.rad) over the calibration corpus (corpus/calib.jsonl: 1,238,874 tokens, 390 documents,
# DS41 chat template; agentic 45% / chat 20% / wiki en+fi 20% / code 15%; disjoint from the KL corpus).
# The engine's --calib-corpus mode (prefill only, prefix cache and speculation off, no preemption) with the expert tap of the
# iqrmoe image (ds41-iqrmoe b77f7a85: writes into --calib-dir once the whole corpus is through; per-expert down Grams packed).
# Per recorded layer and rank: the pooled gate_up Gram [5120^2] of the block input, the pooled down Gram [2304^2] and each of
# its 96 experts' own down Grams; files gram.r<rank>.blk.<L>.ffn_{gate_up,down}_exps[.<e>].bin (RADGRAM2), rad-convert's
# calib=DIR. Layers come in chunks (LAYERS=lo:hi, default 0:20; run again with 20:40) sized to the disk:
#   disk  4.583 GB a layer (10 layers 45.8 GB, 20 layers 91.7 GB, all 40 183.3 GB)
#   RAM   4.08 GB a layer over the 4 ranks (20 layers 81.6 GB) + 8.2 GB pinned staging, beside the host pool and Engram
#   VRAM  126 MB a layer a rank + 2.04 GB scratch -> the host pool HP_CAL (default 58368 MiB = the reference's 53760 + 4608)
# Engram from disk (NGRAM_OS=disk) to keep 103 GiB of RAM free: placement does not change the statistics.
# Steps: guards, inputs (image, .rad + .sha256ok, corpus, free disk >= chunk + 10 GB), one --calib-corpus run (deadline
# CAL_DEADLINE, default 5400 s), gram_check.py over the chunk (files, GB, per-expert rows; experts under 9216 rows fall back to
# the pooled down Gram). Outputs: $WORK/calib (= runs/iqr/work/calib unless IQRWORK is set), runs/iqr/calib-<lo>-<hi>.log.
# DRYRUN=1 bash 81-iqr-calib.sh : prints, runs and writes nothing. Estimate per chunk 20-45 min (start ~6 min for 417 GB;
# 1.24M tokens of prefill at an unmeasured 600-1500 tok/s; every step copies 2.04 GB a recorded layer a rank to the host).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
for LIB in "$H/lib-iqr.sh" "$H/../lib-iqr.sh" "$HOME/gpu/ds41/exl3-test/queue-iqr/lib-iqr.sh"; do [ -e "$LIB" ] && break; done
. "$LIB" || { echo "lib-iqr.sh not found"; exit 1; }
LAYERS=${LAYERS:-0:20}; LO=${LAYERS%:*}; HI=${LAYERS#*:}; TAG=calib-$LO-$HI
NEED=$(( (HI - LO) * CAL_LAYER_B + 10000000000 ))
guards iqr-calib || exit 1
model ref; HP=${HP_CAL:-58368}
if [ "$DRY" != 1 ]; then
  have ref || { log "RESULT iqr-calib $TAG: VIRHE reference not here"; exit 1; }
  [ -s $CORPUS/calib.jsonl ] || { log "RESULT iqr-calib $TAG: VIRHE $CORPUS/calib.jsonl missing"; exit 1; }
  mkdir -p $WORK/calib
  a=$(df -B1 --output=avail $WORK | tail -1 | tr -d ' ')
  [ "$a" -ge $NEED ] || { log "RESULT iqr-calib $TAG: VIRHE $a B free on $WORK < $NEED B (layers $LAYERS: $(( (HI - LO) * CAL_LAYER_B / 1000000000 )) GB + 10 GB)"; exit 1; }
  n0=$(ls $WORK/calib | grep -cE "blk\.($(seq -s'|' $LO $((HI - 1))))\.ffn_" || true)
  [ "$n0" = 0 ] || { log "RESULT iqr-calib $TAG: VIRHE $WORK/calib already holds $n0 files of layers $LAYERS (move them away first)"; exit 1; }
fi
oneshot_vars; export NGRAM=${NGRAM_OS:-disk}
export XARGS="--calib-corpus /corpus/calib.jsonl --calib-dir /work/calib --calib-layers $LAYERS"
log "RESULT iqr-calib $TAG: start IMAGE=$IMAGE MODEL=$MODELDIR/$MODEL $(desc) corpus $(md5sum $CORPUS/calib.jsonl 2>/dev/null | cut -c1-8) -> $WORK/calib"
oneshot $TAG ${CAL_DEADLINE:-5400}; rc=$?
[ "$DRY" = 1 ] && { run "python3 -I $KIT/gram_check.py $WORK/calib $LO $HI"; exit 0; }
cl=$(grep -E 'calib: [0-9]+ documents, [0-9]+ tokens in' $O/$TAG.log | tail -1 | cut -c1-200)
wr=$(grep -m1 'calibration wrote layer Hessians' $O/$TAG.log | cut -c1-200)
if [ $rc != 0 ]; then
  log "RESULT iqr-calib $TAG: VIRHE $ONESHOT_ST exit $ONESHOT_EC after $ONESHOT_S s: $(errs $TAG) | $cl"
  exit 1
fi
chk=$(python3 -I $KIT/gram_check.py $WORK/calib $LO $HI 2>&1); ck=$?
log "RESULT iqr-calib $TAG: done in $ONESHOT_S s; $cl; ${wr:-no 'wrote' line}; $chk; disk free $(df -h --output=avail $WORK | tail -1 | tr -d ' ')"
[ $ck = 0 ] || exit 1
