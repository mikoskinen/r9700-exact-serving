#!/bin/bash
# 109-fn-tp4new.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h: two TP4 arms
# ~55 min each + p15w ~10 min) -- newer Radiance on Flash-Next TP4 (the user 9.11.: "testaa myös radiancen uudet muutokset"),
# and Radiance #428 (a TP all-reduce barrier deadlock under one cold ~131k-token prefill, reported on 1.17.0 TP2 wht6).
#   r17n  v1.17.0 (radiance-local:1.17.0-src: built from the tag in radiance-build, the upstream Dockerfile's configure
#         and strip, over the stock 1.15.1 image's runtime) on the k4n line: TP4 all_vram MML 262144 MAXSEQS 16 MNBT 2048,
#         MTP 3, --prefix-resume nearest
#   rmn   main 451b136 (radiance-local:main-451b136), the same line; carries e2e9416 (qwen4exp: chained launches with the
#         MTP drafter -- ON by itself, no flag, so no rmnC arm), 7f5218e (decode / verify / draft passes recorded, keyed by
#         the split cap, #402), 7cad854 (gqa16 decode takes an eight-row verify), a583046 (--prefill-interleave auto:1)
#   p15w  #428 alone on the production-like 1.15.1 TP2 wht6 serve line (fn-tp2.yaml: 83+86, MML 524288 yarn:2,
#         expert_tiered 16384, test port 8090); production's yaml is only read and diffed
# = 93e-fn-tp4ab.sh beside this file with ARMS="r17n rmn p15w" X93_REF=k4n (106's rows, no baseline re-run)
# X93_HANG428=1 X93_ITEM=109-fn-tp4new. Per TP4 arm: gate, slow-state probe (+ restart), produp + probe, mb c4 / c8 / c16
# (+ engine stats: verified / accepted drafts a step, acceptance, verify rows a step, decode step ms, pass play share),
# prefill 16k + fresh 64k / 128k, the 33k copy test, mb pf, BetterBench, cc_sim u16 (+ engine stats), the #428 check; then
# each vs k4n: cmparm, cc_sim --pair, warm-up answers (information only: 1.16.0 changed the softmax, #382).
# The .rad: loads unchanged (no reconvert) -- the format code changed only in checks (rad-info of both builds reads the
# production container's header); a load failure shows as the arm's gate VIRHE.
# DRYRUN=1 bash 109-fn-tp4new.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"r17n rmn p15w"} X93_REF=${X93_REF:-k4n} X93_ITEM=${X93_ITEM:-109-fn-tp4new} X93_HANG428=${X93_HANG428:-1}
for f in "$H/93e-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93e-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 109-fn-tp4new: VIRHE 93e-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
