#!/bin/bash
# Does task A`s extension (ds41-vis a5efdb0, rad-convert --in-place with dfbc908) apply to the official-body layout?
# Run on a throwaway copy of the new file; the copy is deleted afterwards.
M=$HOME/models/ds41-ob; IMG=radiance-local:ds41-vis-a5efdb0f; X=deepseek-v4.1-flash-1017-exl3-2.25bpw-official-vistest.rad
LOG=$HOME/ob-tools/vistest-convert.log; t0=$(date +%s)
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v $M/rel/hf-src/DeepSeek-V4.1-Flash-1017:/c/hf-src/DeepSeek-V4.1-Flash-1017:ro -v $M/emptyexl3:/c/turboderp/DeepSeek-V4.1-Flash-1017-exl3:ro \
  -v $M/in/engram-tables.safetensors:/c/engram-tables.safetensors:ro -v $M/in/deepseek41-exl3-full225.recipe:/c/recipe:ro \
  -v $M/in/chat_template-vis.jinja:/c/chat_template.jinja:ro -v $M/vistest:/out -w /c -e EXL3_SRC=turboderp/DeepSeek-V4.1-Flash-1017-exl3 \
  --entrypoint rad-convert $IMG hf-src/DeepSeek-V4.1-Flash-1017 --add engram-tables.safetensors --recipe recipe \
  --tokenizer hf-src/DeepSeek-V4.1-Flash-1017/tokenizer.json -v --chat-template chat_template.jinja --reuse /out/$X --in-place > $LOG 2>&1
echo "rc $? $(( $(date +%s)-t0 )) s"; grep -v "^D \|radiance: dbg" $LOG | grep "^E \|^W \|in place\|in-place\|wrote\|added\|kept" | cut -c1-300 | head -12; ls -la $M/vistest
