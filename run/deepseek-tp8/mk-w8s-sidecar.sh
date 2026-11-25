#!/bin/bash
# Builds the weight sidecar p30t-i8-w8s.rad that the DeepSeek commands in "Results" load
# (RADIANCE_WEIGHT_SIDECAR in run/deepseek-tp8/*.env): the 8-bit twin sidecar p30t-i8-twin.rad
# (run/deepseek-tp8/mk-twin-sidecar.sh) copied unchanged, plus an int8 twin of the drafter's Markov
# table (w8s.recipe).
#
# Usage: MODEL_DIR=~/models/ds41-exl3 run/deepseek-tp8/mk-w8s-sidecar.sh
#
# Run it once on each machine, after mk-twin-sidecar.sh, while no server runs there. It reads the model
# file and p30t-i8-twin.rad from MODEL_DIR and writes MODEL_DIR/p30t-i8-w8s.rad. It uses 4 CPU cores,
# no GPU, and takes about a minute. A sidecar serves only the model file it was written from.
set -uo pipefail
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder with the DeepSeek model files}
MODEL_FILE=${MODEL_FILE:-deepseek-v4.1-flash-1017-exl3-2.25bpw.rad}
KEEP=${KEEP:-p30t-i8-twin.rad}
OUT=${OUT:-p30t-i8-w8s.rad}
IMG=${IMG:-ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -s "$MODEL_DIR/$MODEL_FILE" ] || { echo "no model file $MODEL_DIR/$MODEL_FILE"; exit 1; }
[ -s "$MODEL_DIR/$KEEP" ] || { echo "no $MODEL_DIR/$KEEP (run mk-twin-sidecar.sh first)"; exit 1; }
[ -e "$MODEL_DIR/$OUT" ] && [ "${FORCE:-0}" != 1 ] && { echo "$MODEL_DIR/$OUT exists (FORCE=1 overwrites)"; exit 1; }
docker run --rm --network none --cpus 4 --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR":/models -v "$HERE":/rcp:ro \
  --entrypoint /opt/radiance/bin/rad-sidecar "$IMG" \
  "/models/$MODEL_FILE" -o "/models/$OUT.tmp" --recipe /rcp/w8s.recipe \
  --twin 'dspark.markov_w2.weight' --keep "/models/$KEEP" -v > "$MODEL_DIR/${OUT%.rad}.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$MODEL_DIR/$OUT.tmp" ]; then
  rm -f "$MODEL_DIR/$OUT.tmp"
  echo "rad-sidecar exit $rc; tail of $MODEL_DIR/${OUT%.rad}.log:"; tail -5 "$MODEL_DIR/${OUT%.rad}.log"
  exit 1
fi
mv -f "$MODEL_DIR/$OUT.tmp" "$MODEL_DIR/$OUT"
echo "$MODEL_DIR/$OUT: $(stat -c %s "$MODEL_DIR/$OUT") bytes"
