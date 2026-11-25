#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzuz-sdtemp.sh (queue item, 11.11.2026, ds-main-port; gpu02, all 4 cards; every radiance container DOWN; ~50 min;
# sorts right after 57-p4zyzzzzz-ds-iejzzzu-z0occ, before 57-p4zyzzzzz-ds-iekz-fnloadid in C and en_US) -- LEVER SD DRAFTER-TEMPERATURE
# SCREEN (speed lever for the 1-user prose target; the user's GO on SD 11.11.: distribution-exact, not byte-identical at T > 0):
# the C screen block of iejzzz-sd showed 1u essay +6.3 % at draft_sample 70 vs A (n 1). One NST 5 start on the int4 stack image
# with stack3's S arm words (ds41_bi 2 in the pin, RADIANCE_DS41_BI=0 at start, the DLW q8_0 sidecar, --draft-coupling on,
# RADIANCE_DRAFT_SAMPLE=0 at start): chi-square gate under draft_sample 70 (fixed window 5) vs non-speculative, 2000 x 2 seeded;
# then blocks B H S E E S H B (B = draft_sample 1 = Q at T, H = 50, S = 70, E = 85: Q at T x v/100), each = 1u essay / code /
# edit at the server default T 1, seed 1234, thinking on + mb c4 / c8 (75 s); identity at T 0 under B and S (28k copy + greedy
# set vs bz d0). Body queue-ar/dsmain/sd/dsm-sdtemp.sh (= dsm-sd.sh 4d2def10 + these changes). Tag "RESULT xsdt ", out
# runs/xdsm/sdtemp. IMAGE radiance-local:ds41-int4-cea53a40-5c13e555-l (482004322f11). No cc_sim, no 2.9.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzuz-sdtemp.sh : prints every command; checks the image, every helper; no server, no GPU.
export SD_IMAGE=${SD_IMAGE:-radiance-local:ds41-int4-cea53a40-5c13e555-l} SD_ITEM=${SD_ITEM:-57-p4zyzzzzz-ds-iejzzzuz-sdtemp}
export SD_OUT=${SD_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/sdtemp}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/sd/dsm-sdtemp.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsdt: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
