#!/bin/bash
# 57-p4zyzzzzz-ds-hz-ngram.sh (queue item, 10.11.2026, ds41-gg agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~25 min) -- DeepSeek-V4.1 with prompt-lookup (n-gram) drafting ALONE, the DSpark head neither declared,
# loaded nor placed (the ds41-gg image's core passes RadBuildCtx::draft_head_off again under --spec-ngram): one arm,
# --spec-ngram 5 --draft-window adaptive (--num-speculative-tokens unset), the t3 env and serve line of dsm-draft.sh.
# Gate, modes (the lookup-alone speculation line, no DSpark line, the routed experts in VRAM vs bz's d0 / d5), warm-up
# answers vs stock, probe essay / code (1 user), editprobe (1 user: an edit that re-emits a ~2.5k-token region; tokens a
# step, acceptance, echo check), mb c4 / c8 (aggregate, per-stream p50, tokens a step, acceptance), the 28k copy, the
# greedy set c1; identity vs bz's d0 files; a comparison line vs bz's d0 / d5a numbers (no re-runs). Lookup only
# proposes and the trunk verifies, so the greedy text should not move; a difference is reported, not hidden. Nothing goes
# into a production line without the user's go. IMAGE radiance-local:ds41-gg-d5bed91f-5c13e555-l (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hz-ngram.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-d5bed91f-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-ngram.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgn: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
