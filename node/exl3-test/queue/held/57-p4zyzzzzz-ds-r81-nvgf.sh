#!/bin/bash
# 57-p4zyzzzzz-ds-iz-nvgf.sh (queue item, 11.11.2026, ds-main-port NVG-fold agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~50 min: the A / B start ~36 (gate 5, ABBA c1 / c4 / c8 4 x 4, the 136k under A and B ~6, identity
# ~6), the profiled start ~14) -- LEVER NVG-fold on radiance-local:ds41-nvgf-0e19833a-5c13e555-l (branch ds41-nvgf 0e19833 =
# ds41-nvlat b12aca4 + the fold; body dsm-nvgf.sh, log tag "RESULT xnf ", out runs/xdsm/nvgf).
# Lever NVG unpacks the packed latent's selected entries in each of the 38 attentions (a gather launch a call); the 38 read only 8
# selections. The fold: index_topk_nvg unpacks each selection once inside the selection's own launch, the gathered attentions read
# the rows. Exact (tests/r4d_emu test_ds41nvgf, 204 checks; a code mutation caught). Hypothesis: c1 -0.2 .. -0.35 ms a step.
#   ONE START, NST 0, the NVG item's arm-B env on both arms (stack env + RADIANCE_DS41_LATENT=nvfp4 + RADIANCE_DS41_NVGATHER=16) +
#   RADIANCE_DS41_NVGATHER_FOLD=16: the start log must say the fold took 8 of 8 index sources and 38 + 38 of 38 layers (else the
#   start stops); in process ABBA ds41_nvgfold 0 (A = lever NVG = hszzzzz-nvg's B) / 1 (B), c1 / c4 / c8 (mb 75 s); the 136k cold
#   prefill under A and B; identity per arm (28k copy + greedy c1 vs bz d0) and B vs A.
#   PROFILED START (--profile-ops, not a speed): windows short / long / c8 under fold 1 then 0 -> nvlatprof.py ops per window.
#   No cc_sim, no 2.9. Nothing goes into a production line from here.
# XNF_IMAGE, XNF_NST, XNF_ORDER, XNF_DUR, XNF_PHASES0, XNF_DEEP, XNF_PROF, XNF_BASEENV, XNF_ROWS: set them by editing the export line
# below before moving this file into queue/ (the queue runner passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iz-nvgf.sh : prints every command; checks the helpers, the image and the yaml (no server, no GPU),
# writes nothing.
export XNF_IMAGE=${XNF_IMAGE:-radiance-local:ds41-nvgf-0e19833a-5c13e555-l} XNF_ITEM=${XNF_ITEM:-57-p4zyzzzzz-ds-iz-nvgf}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-nvgf.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xnf: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
