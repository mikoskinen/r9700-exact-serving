#!/bin/bash
# 57-p4zyzzzzz-ds-iej-z0ab.sh (queue item, 11.11.2026, ds-main-port Z0 agent; sorts right after 57-p4zyzzzzz-ds-iei-da-c-measure,
# before j-shexplane; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~36 min) -- LEVER Z0 (depth-0 parity)
# STEP 3, THE A/B (tag "RESULT xz0b ", out runs/xdsm/z0ab, body queue-ar/dsmain/dsm-z0ab.sh, driver z0blocks.py = ggblocksdz2.py
# + per-phase experts / lend). ONE NST 5 start of the ds41-z0 image (radiance-local:ds41-z0-61cbb39b-5c13e555-l, sha256:7fae0a36e8b9: ds41-int3 1116248 + Z0-H 32b96af + Z0-A bc1f764 + 61cbb39),
# hxzzy-stack2's NST 5 env and pin, with the arena rungs 8,48 declared at start (RADIANCE_DS41_ARENA_DECODE_ROWS=8,48) and the
# context-skip ring at the drafter's window (RADIANCE_DRAFT_CTXSKIP_SLOTS=129, +23 MiB a rank); with arena_d0 0 every decode /
# verify step still takes the 48-row rung and the skip still refills every 32 steps = stack2's behaviour. IN PROCESS, blocks A B C C B A
# (1u essay / code (stepprobe), essay by position, edit, mb c4 / c8 75 s; /stats experts + GET lend around every phase):
#   A = stack2's pin + draft_lend_hyst 0 + arena_d0 0 (today);
#   B = + arena_d0 1 (lever Z0-A: a depth-0 step takes the 8-row rung, +0.16 GiB a rank lent to the experts, its refills
#       deferred to probes / prefills / the ring's end so they never pull it back to 48) + draft_lend_hyst
#       20 (lever Z0-H: leaving a lent depth 0 needs 20 % more yield);
#   C = B + ds41_argather 4 + ds41_qnfold 0 + ds41_swfold 0 (SCREEN, pin only: NST 0's AR form and no K1 a / e folds; the
#       profile's AR-form item and the folds item whose two readings disagreed).
# Identity: the 28k copy + the 20-prompt greedy set at c1 vs bz d0 under A, B and B with depth 0 forced (draft_depth0 2, probes
# every 64: the 8-row rung, the lend and the refills at c1). Reference: hxzzy-stack2's NST 0 blocks (int3, same env; no re-run).
# Summary: B vs A pairs (t95), C vs B, each vs stack2 NST 0 (c4 138.5, c8 180.3), VRAM units / hit / stream / lent per phase.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Exact: placement and D7's choice only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-z0ab.sh : prints every command; checks the image (+ the Z0 literals), every helper (files,
# parse, functions), the sidecar and the yaml; no server, no GPU, writes nothing.
export XZ0B_IMAGE=${XZ0B_IMAGE:-radiance-local:ds41-z0-61cbb39b-5c13e555-l} XZ0B_ITEM=${XZ0B_ITEM:-57-p4zyzzzzz-ds-iej-z0ab}
export XZ0B_ORDER=${XZ0B_ORDER:-ABCCBA} XZ0B_ARENA_ROWS=${XZ0B_ARENA_ROWS:-8,48} XZ0B_LOGLITS=${XZ0B_LOGLITS:-lever,Z0-A,set,up|ring,of,129,steps}
export XZ0B_STARTENV=${XZ0B_STARTENV-RADIANCE_DRAFT_CTXSKIP_SLOTS=129}
export XZ0B_ARMA=${XZ0B_ARMA:-'{"draft_lend_hyst": 0, "arena_d0": 0}'}
export XZ0B_ARMB=${XZ0B_ARMB:-'{"draft_lend_hyst": 20, "arena_d0": 1}'}
export XZ0B_ARMC=${XZ0B_ARMC-'{"draft_lend_hyst": 20, "arena_d0": 1, "ds41_argather": 4, "ds41_qnfold": 0, "ds41_swfold": 0}'}
export XZ0B_IDF=${XZ0B_IDF-'{"draft_depth0": 2, "draft_probe": 64}'}
export XZ0B_IMGLITS=${XZ0B_IMGLITS-'B|arena_d0 takes 1 (on) or 0 (off)
B|draft_lend_hyst takes 0 (off) .. 100
B|lever Z0-A set up
B|hyst_holds
B|RADIANCE_DRAFT_CTXSKIP_SLOTS'}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0ab.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0b: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
