#!/bin/bash
# Builds the 8-bit twin sidecar p30t-i8-twin.rad that the DeepSeek commands in "DeepSeek on two
# machines" load (RADIANCE_WEIGHT_SIDECAR in run/deepseek-tp8/deepseek.env).
#
# Usage: MODEL_DIR=~/models/ds41-exl3 run/deepseek-tp8/mk-twin-sidecar.sh
#
# Run it once on each machine while no server runs there. It reads the model file and the drafter
# sidecar dspark-dlw-q8_0.rad from MODEL_DIR and writes MODEL_DIR/p30t-i8-twin.rad
# (7,502,987,264 bytes; the script wants 12 GB free). It takes 5 to 7 minutes on 16 CPU cores and
# uses no GPU.
#
# The sidecar holds every dense linear weight of layers 0-39 (attention, Engram and the shared
# expert, 562 entries) decoded from the model file and requantised to int8 codes with one bf16
# scale per 128 weights. Each is written as a twin that the engine keeps beside the model file's
# own weight and uses for prompt rows only (RADIANCE_DS41_TWIN); decode rows keep the model file's
# weights. The drafter sidecar's 43 entries are copied in unchanged. A sidecar serves only the
# model file it was written from, so MODEL_FILE must be the file the server loads.
set -uo pipefail
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder with the DeepSeek model files}
MODEL_FILE=${MODEL_FILE:-deepseek-v4.1-flash-1017-exl3-2.25bpw.rad}
DLW=${DLW:-dspark-dlw-q8_0.rad}
OUT=${OUT:-p30t-i8-twin.rad}
IMG=${IMG:-ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-0735a8e2}
CPUS=${CPUS:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
RCP=p30-i8-all.recipe
[ "$(md5sum < "$HERE/$RCP" | cut -c1-8)" = 6ca81f90 ] || { echo "$HERE/$RCP is not the published recipe"; exit 1; }
[ -s "$MODEL_DIR/$MODEL_FILE" ] || { echo "no model file $MODEL_DIR/$MODEL_FILE"; exit 1; }
[ -s "$MODEL_DIR/$DLW" ] || { echo "no drafter sidecar $MODEL_DIR/$DLW"; exit 1; }
[ -e "$MODEL_DIR/$OUT" ] && [ "${FORCE:-0}" != 1 ] && { echo "$MODEL_DIR/$OUT exists (FORCE=1 overwrites)"; exit 1; }
FREE=$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 12 ] || { echo "only $FREE GB free in $MODEL_DIR (12 GB needed)"; exit 1; }
t0=$(date +%s)
nice -n 19 docker run --rm --network none --cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR":/models -v "$HERE":/rcp:ro \
  --entrypoint /opt/radiance/bin/rad-sidecar "$IMG" \
  "/models/$MODEL_FILE" -o "/models/$OUT.tmp" --requant --keep "/models/$DLW" \
  --twin 'blk.[0-9].*' --twin 'blk.[1-3][0-9].*' --recipe "/rcp/$RCP" > "$MODEL_DIR/${OUT%.rad}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  rm -f "$MODEL_DIR/$OUT.tmp"
  echo "rad-sidecar exit $rc after $(( $(date +%s) - t0 )) s; tail of $MODEL_DIR/${OUT%.rad}.log:"
  tail -5 "$MODEL_DIR/${OUT%.rad}.log"
  exit 1
fi
mv -f "$MODEL_DIR/$OUT.tmp" "$MODEL_DIR/$OUT"
NT=$(grep -c '#pf ' "$MODEL_DIR/${OUT%.rad}.log")
echo "$MODEL_DIR/$OUT: $(stat -c %s "$MODEL_DIR/$OUT") bytes in $(( $(date +%s) - t0 )) s, $NT twins (want 562)"
echo "$(tail -1 "$MODEL_DIR/${OUT%.rad}.log" | cut -c1-200)"
