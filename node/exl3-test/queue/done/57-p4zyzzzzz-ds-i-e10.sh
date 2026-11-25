#!/bin/bash
# 57-p4zyzzzzz-ds-i-e10.sh (queue item, 11.11.2026, ds-main-port E10 agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~55 min: a one-card GD-CHECK ~6 min, then two starts ~22-25 min each) -- LEVER D5 = E10 on DeepSeek-V4.1: the
# decode band's fused EXL3 chain moe_exl3_gate_up_down (interface v11 with the SwiGLU `limit`; exl3r4d's fused launch at K 2 /
# 2.5, fn4-dsgd 1a1d429 / a3e0f1c) in place of the split gate_up / down at passes of 3..16 tokens, A/B'd IN ONE PROCESS a start
# on the ds41-e10 image (ds41-int2 745ce76 + the arch hook 1d28b33; exl3-ds-e10 76e24c4): B = e10_gd 1, A = e10_gd 0 (today's
# ops), RADIANCE_DS41_E10_GD=1 at start declares the fused op. P0 (one card, no server): perfmoe-e10 --gd GD-CHECK at DS's EP4
# geometry, limit 10, K 2 and 2.5, M 3-16 (+ a forced clamp) -- a failure stops the item. Then NST 0 (stack N pin) and NST 5
# (D7 + hybrid + drafter_mx 1) starts: blocks A B B A (stepprobe essay,code, editprobe, mb c4 / c8 75 s), identity B vs A
# strict (28k copy + 20-prompt greedy c1) and vs bz d0. Sorts after fn4's 172 (hxzy-fn-dsgd: the same kernel's GD-CHECK
# with a fallback count). Nothing goes into a production line without the user's go.
# Tags "RESULT xe10 ", "RESULT xe10 summary: ...", "... done"; out runs/xdsm/e10. Body queue-ar/dsmain/dsm-e10.sh.
# IMAGE radiance-local:ds41-e10-1d28b33c-76e24c49-l (sha256:3914f475365a).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-i-e10.sh : prints every command; checks the image, the files and the helpers (no server, no
# GPU), writes nothing.
export XE10_ITEM=${XE10_ITEM:-57-p4zyzzzzz-ds-i-e10}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-e10.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xe10: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
