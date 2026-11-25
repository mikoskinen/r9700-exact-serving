#!/bin/bash
# rad-convert the official release body (sparse release dir: headers + non-trunk-expert tensors) with the
# current container's recipe, --reuse of the current container for everything not in CHANGED (the EXL3 experts),
# --imatrix the body imatrix (Unsloth's + the plugin's fused entries, experts' entries dropped).
#   ob-convert.sh CHANGED_LIST OUTNAME [extra rad-convert args]
set -u
M=$HOME/models/ds41-ob
IMG=${IMG:-radiance-local:ds41-fin-5e86593e-4a67be32-l}
CH=$(realpath "$1"); OUT=$2; shift 2
LOG=$HOME/ob-tools/convert-$OUT-$(date +%m%d-%H%M).log
mkdir -p $M/out $M/emptyexl3
args=(hf-src/DeepSeek-V4.1-Flash-1017 --add engram-tables.safetensors --recipe recipe
      --tokenizer hf-src/DeepSeek-V4.1-Flash-1017/tokenizer.json -v --chat-template chat_template.jinja
      --imatrix imatrix-body.gguf --reuse /base.rad --changed changed.txt -o /out/$OUT "$@")
echo "$(date '+%F %T') rad-convert ${args[*]}  [IMG $IMG, changed $CH: $(wc -l < $CH) tensors]" | tee "$LOG"
NAME=obconv-${OUT%.rad}
( peak=0; while sleep 5; do id=$(docker inspect -f '{{.Id}}' $NAME 2>/dev/null) || { [ $peak -gt 0 ] && break; continue; }
  v=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/system.slice/docker-$id.scope/memory.stat 2>/dev/null || echo 0); [ "${v:-0}" -gt $peak ] && peak=$v
  echo "$peak" > $HOME/ob-tools/memory-peak-$OUT.txt; done ) &
SAMP=$!
t0=$(date +%s)
docker run --rm --name $NAME --network none --user "$(id -u):$(id -g)" \
  -v $M/rel/hf-src/DeepSeek-V4.1-Flash-1017:/c/hf-src/DeepSeek-V4.1-Flash-1017:ro \
  -v $M/emptyexl3:/c/turboderp/DeepSeek-V4.1-Flash-1017-exl3:ro \
  -v $M/in/engram-tables.safetensors:/c/engram-tables.safetensors:ro -v $M/in/deepseek41-exl3-full225.recipe:/c/recipe:ro \
  -v $M/in/chat_template.jinja:/c/chat_template.jinja:ro -v $M/in/imatrix-body.gguf:/c/imatrix-body.gguf:ro \
  -v "$CH":/c/changed.txt:ro -v $M/cur/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad:/base.rad:ro -v $M/out:/out \
  -w /c -e EXL3_SRC=turboderp/DeepSeek-V4.1-Flash-1017-exl3 --entrypoint rad-convert "$IMG" "${args[@]}" >> "$LOG" 2>&1
rc=$?; kill $SAMP 2>/dev/null
echo "$(date '+%F %T') exit $rc after $(( $(date +%s) - t0 )) s; container peak anon memory $(( $(cat $HOME/ob-tools/memory-peak-$OUT.txt 2>/dev/null || echo 0) / 1048576 )) MiB" | tee -a "$LOG"; exit $rc
