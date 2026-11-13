#!/bin/bash
# 57-p4zyzzzzz-ds-r-confhead.sh (queue item, 10.11.2026, ds41-acc agent, lever C1; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~45 min) -- DeepSeek-V4.1's DSpark CONFIDENCE HEAD as a verify gate. The release's
# mtp.2.confidence_head.proj (in no container: the sidecar ~/models/ds41-exl3/dspark-conf.f32 beside the .rad, sha256
# a97178fa...) gives each draft a logit from the head's pre-norm hidden and the walk's predecessor (inference/model.py
# forward_head); the engine's dspark_conf switch cuts each sequence's verify to the longest prefix with sigmoid >= 0.5
# (exllamav3's rule). EXACT: it changes how many drafts a verify carries, never a proposal or a token. One start (DSpark NST 5
# --draft-window adaptive, gg_gemv 2), in-process blocks dspark_conf A 0 / B 1 (the rule replaces the window's stop) / C 150
# (the rule capped by the window), ABCCBA: c1 essay / code / edit (tokens a step, acceptance, ms a step, drafts a step, essay
# by position), mb c4 / c8; one OBSERVE block (calibration over today's chain); identity (28k copy, greedy set) under B and C
# vs bz d0. No cc_sim, no 2.9, no BetterBench. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-acc-1f2f03b5-5c13e555-l (XAC_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r-confhead.sh : prints every command, runs and writes nothing.
export XAC_IMAGE=${XAC_IMAGE:-radiance-local:ds41-acc-1f2f03b5-5c13e555-l} XAC_MODE=conf
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-acc.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xac: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
