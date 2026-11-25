#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzz-sd.sh (queue item, 11.11.2026, ds-main-port SD agent; gpu02, all 4 cards; every radiance container DOWN;
# ~33 min) -- LEVER SD, SAMPLED DRAFTING (the user's GO 11.11.: distribution-exact, NOT byte-identical to non-speculative
# sampling at temperature > 0; behind fn4 draft_sample, default 0): at temperature > 0 the DSpark W1 walk samples its drafts
# x ~ Q (the eight best keys of the whole vocabulary, softmax at the request's temperature) and the verify keeps x with
# probability min(1, P/Q), else draws from norm(max(0, P - Q)); temperature 0 drafts and verifies as today. One NST 5 start with
# stack2's n5 env + --draft-coupling on: ABBA A = draft_sample 0 / B = 1 (1u essay / code / edit at the server default, seed
# 1234, thinking on; mb c4 / c8), a C = 70 screen, the on-card chi-square (2000 x 2 seeded requests, SD vs non-speculative,
# p-value), identity at temperature 0 under B and A (28k copy + greedy set vs bz d0). Body queue-ar/dsmain/sd/dsm-sd.sh (its
# header). Tag "RESULT xsd ", out runs/xdsm/sd. IMAGE radiance-local:ds41-sd-397608f6-5c13e555-l (sha256:24c9cf8eccce) = ds41-int3 1116248a
# + ds41-sd 397608f (37ac893 lever SD + 397608f test fix). No cc_sim, no 2.9.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzz-sd.sh : prints every command; checks the image, every helper; no server, no GPU.
export SD_IMAGE=${SD_IMAGE:-radiance-local:ds41-sd-397608f6-5c13e555-l} SD_ITEM=${SD_ITEM:-57-p4zyzzzzz-ds-iejzzz-sd}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/sd/dsm-sd.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsd: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
