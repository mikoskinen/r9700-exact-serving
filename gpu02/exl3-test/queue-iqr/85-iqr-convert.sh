#!/bin/bash
# 85-iqr-convert.sh (6.11.2026; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN; NEEDS THE USER'S GO for
# the GPU time) -- one IQR mix's GPTQ + computed trellis on the cards (rad-convert --gpus, libquant's lq_tcq_gpu.hip, the host
# loop's bytes): routed experts of layers 0-39 from the release's MXFP4 (decoded to f32) against the calibration Grams, every
# other weight byte-copied from the --reuse containers (the reference .rad: q8_0 dense, iq4_nl Engram, DSpark MXFP4 experts),
# and every expert an earlier mix already wrote under the same rule copied from that mix (--reuse it too).
# MIX (default iqr225; recipes/ds41-<MIX>.recipe), GPUS (default each card twice, 0,1,2,3,0,1,2,3: set from 80-iqr-gptq-rate's
# best x1/x2/x3 row), SRC (the release's hf-src dir; its Engram shards model-0004[78] may be the skeleton's sparse files, since
# --reuse supplies every weight that reads them), CALIB (the Grams dir of 81-iqr-calib), REUSE (space-separated containers,
# default the reference), OUTDIR (default ~/models/ds41-iqr). The run resumes from <out>.journal when one is there.
# Disk at once (GB): SRC 307 (without Engram shards) + CALIB 183.3 + reference 417.1 + output (iqr215 274.5, iqr225 283.2,
# iqr2l 321.0, iqr3s 343.8; control iqr215-iq1strunk 272.4) -- the check below refuses to start without output + 20 GB free.
# DRYRUN=1 bash 85-iqr-convert.sh : prints only. Estimate: from 80-iqr-gptq-rate (s a weight x 384 experts x the mix's new
# (layer, form) units: iqr225 80 units; then iqr2l +38, iqr3s +28, iqr215 +15), + I/O.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
for LIB in "$H/lib-iqr.sh" "$H/../lib-iqr.sh" "$HOME/gpu/ds41/exl3-test/queue-iqr/lib-iqr.sh"; do [ -e "$LIB" ] && break; done
. "$LIB" || { echo "lib-iqr.sh not found"; exit 1; }
MIX=${MIX:-iqr225}; GPUS=${GPUS:-0,1,2,3,0,1,2,3}
SRC=${SRC:-$HOME/models/ds41-release/hf-src/DeepSeek-V4.1-Flash-1017}
CALIBD=${CALIB:-$WORK/calib}
model ref; REFRAD=$MODELDIR/$MODEL
REUSE=${REUSE:-$REFRAD}
OUTDIR=${OUTDIR:-$IQRMODELS}; OUT=ds41-$MIX.rad
RECIPE=$KIT/recipes/ds41-$MIX.recipe
ENG=${ENGRAM_TABLES:-$KIT/engram-tables.safetensors}
case $MIX in iqr215) OB=274500000000 ;; iqr225) OB=283200000000 ;; iqr2l) OB=321000000000 ;; iqr3s) OB=343800000000 ;;
             iqr215-iq1strunk) OB=272400000000 ;; *) echo "MIX?"; exit 1 ;; esac
guards iqr-convert || exit 1
if [ "$DRY" != 1 ]; then
  for f in $RECIPE $ENG $SRC/model.safetensors.index.json $SRC/tokenizer.json $KIT/chat_template.jinja; do
    [ -s $f ] || { log "RESULT iqr-convert $MIX: VIRHE $f missing"; exit 1; }; done
  for r in $REUSE; do [ -s $r ] || { log "RESULT iqr-convert $MIX: VIRHE --reuse $r missing"; exit 1; }; done
  ng=$(ls $CALIBD 2>/dev/null | grep -c '^gram\.r.*ffn_down_exps\.[0-9]*\.bin$')
  [ "$ng" -ge 15360 ] || { log "RESULT iqr-convert $MIX: VIRHE $CALIBD holds $ng per-expert down Grams of 15,360"; exit 1; }
  mkdir -p $OUTDIR
  a=$(df -B1 --output=avail $OUTDIR | tail -1 | tr -d ' ')
  [ -f $OUTDIR/$OUT.journal ] || [ "$a" -ge $(( OB + 20000000000 )) ] || { log "RESULT iqr-convert $MIX: VIRHE $a B free < $OB + 20 GB"; exit 1; }
fi
M="-v $SRC:/c/hf-src/DeepSeek-V4.1-Flash-1017:ro -v $ENG:/c/engram-tables.safetensors:ro -v $RECIPE:/c/recipe:ro -v $KIT/chat_template.jinja:/c/chat_template.jinja:ro -v $CALIBD:/calib:ro -v $OUTDIR:/out"
RA=""; i=0; for r in $REUSE; do M="$M -v $r:/reuse$i.rad:ro"; RA="$RA --reuse /reuse$i.rad"; i=$((i + 1)); done
A="hf-src/DeepSeek-V4.1-Flash-1017 --arch deepseek41 --add engram-tables.safetensors --recipe recipe --tokenizer hf-src/DeepSeek-V4.1-Flash-1017/tokenizer.json --chat-template chat_template.jinja $RA --gpus $GPUS -v"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --memory 200g --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
log "RESULT iqr-convert $MIX: start IMAGE=$IQRIMAGE gpus $GPUS recipe $(md5sum $RECIPE 2>/dev/null | cut -c1-8) calib $CALIBD reuse '$REUSE' -> $OUTDIR/$OUT"
drun iqr-convert-plan 900 $O/convert-$MIX-plan.log $DEV $M -e CALIB=/calib -w /c --entrypoint rad-convert $IQRIMAGE $A --plan-only -o /out/plan-only.rad \
  || { log "RESULT iqr-convert $MIX: VIRHE plan-only: $(grep -E '^E ' $O/convert-$MIX-plan.log | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ "$DRY" = 1 ] || log "RESULT iqr-convert $MIX plan: $(grep -m1 'would be written' $O/convert-$MIX-plan.log)"
t0=$(date +%s)
drun iqr-convert-$MIX ${CONV_DEADLINE:-86400} $O/convert-$MIX.log $DEV $M -e CALIB=/calib -w /c --entrypoint rad-convert $IQRIMAGE $A -o /out/$OUT; rc=$?
[ "$DRY" = 1 ] && exit 0
if [ $rc != 0 ]; then
  log "RESULT iqr-convert $MIX: VIRHE after $(( $(date +%s) - t0 )) s ($(grep -c 'progress' $O/convert-$MIX.log) progress lines; last $(grep progress $O/convert-$MIX.log | tail -1 | cut -c1-120)): $(grep -E '^E ' $O/convert-$MIX.log | tail -3 | tr '\n' ' ' | cut -c1-400); --resume continues from the journal"
  exit 1
fi
(cd $OUTDIR && sha256sum $OUT > $OUT.sha256 && cp $OUT.sha256 $OUT.sha256ok)
log "RESULT iqr-convert $MIX: done in $(( $(date +%s) - t0 )) s: $(grep -E '^I wrote' $O/convert-$MIX.log | cut -c1-200); reused: $(grep -E '^I --reuse' $O/convert-$MIX.log | cut -c1-200); sha256 $(cut -c1-16 $OUTDIR/$OUT.sha256)"
