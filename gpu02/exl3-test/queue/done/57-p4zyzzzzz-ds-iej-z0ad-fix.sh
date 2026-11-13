#!/bin/bash
# 57-p4zyzzzzz-ds-iej-z0ad-fix.sh (queue item, 11.11.2026, ds-main-port Z0 agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~45 min) -- LEVER Z0, THE A/B AGAIN (iej-z0ab and iej-z0ac-w stopped at their start guard: "activation arena: no
# level for 8-token steps -- the sizing declare failed"; ds41-z0 c0cb43c + 3737580 fix it: the head's query rows Q and the drafter block's router geometry in the sizing declare). Tag "RESULT xz0b ", out runs/xdsm/z0ab3, body
# queue-ar/dsmain/dsm-z0ab3.sh (= dsm-z0ab2.sh + the guard printing the arena warnings + a no-device check of the 8-row level +
# the bases below), driver z0blocks.py.
# FIRST, NO DEVICE (z0lvl8): a host-backend TP4 NST 5 --debug-graph start of the 2.25 bpw .rad with this item's NST 5 env must
# accept the 8-token arena level, else VIRHE and nothing runs. Then ONE NST 5 start of radiance-local:ds41-z0-3737580d-5c13e555-l
# (ds41-int3 1116248 + Z0-H + Z0-A + Z0-W + the 8-row fixes c0cb43c + 3737580), hxzzy-stack2's NST 5 env and pin + the arena rungs 8,48 + the
# 129-slot context-skip ring + RADIANCE_DRAFT_LEND_WEIGHTS=0 (Z0-W declared, off). IN PROCESS, blocks A B D C C D B A (1u essay /
# code, essay by position, edit, mb c4 / c8 75 s; /stats experts + GET lend incl. w_* around every phase):
#   A = today (draft_lend_hyst 0, arena_d0 0, draft_lend_w 0);
#   B = arena_d0 1 (a depth-0 step takes the 8-row rung) + draft_lend_hyst 20;
#   D = B + draft_lend_w 1 (lever Z0-W: the drafter's query-pass weights lent with its slab);
#   C = D + ds41_argather 4 + ds41_qnfold 0 + ds41_swfold 0 (screen, pin only).
# Bases: B vs A, D vs B, C vs D (pairs adjacent, t95), each vs stack2's NST 0 blocks. Identity: the 28k copy + the 20-prompt greedy
# set at c1 vs bz d0 under A, B, and B + draft_lend_w 1 with depth 0 forced (probes every 64). Experts / lend per phase per arm.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Exact: placement and D7's choice only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-z0ad-fix.sh : prints every command; checks the image (+ the Z0 literals), every helper (files,
# parse, functions), the sidecar and the yaml, and runs the no-device 8-row level check; no server, no GPU, writes nothing.
export XZ0B_IMAGE=${XZ0B_IMAGE:-radiance-local:ds41-z0-3737580d-5c13e555-l} XZ0B_ITEM=${XZ0B_ITEM:-57-p4zyzzzzz-ds-iej-z0ad-fix}
export XZ0B_ORDER=${XZ0B_ORDER:-ABDCCDBA} XZ0B_ARENA_ROWS=${XZ0B_ARENA_ROWS:-8,48}
export XZ0B_LOGLITS=${XZ0B_LOGLITS:-lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:}
export XZ0B_STARTENV=${XZ0B_STARTENV-RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0}
export XZ0B_ARMA=${XZ0B_ARMA:-'{"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0}'}
export XZ0B_ARMB=${XZ0B_ARMB:-'{"draft_lend_hyst": 20, "arena_d0": 1, "draft_lend_w": 0}'}
export XZ0B_ARMD=${XZ0B_ARMD-'{"draft_lend_hyst": 20, "arena_d0": 1, "draft_lend_w": 1}'}
export XZ0B_ARMC=${XZ0B_ARMC-'{"draft_lend_hyst": 20, "arena_d0": 1, "draft_lend_w": 1, "ds41_argather": 4, "ds41_qnfold": 0, "ds41_swfold": 0}'}
export XZ0B_IDF=${XZ0B_IDF-'{"draft_depth0": 2, "draft_probe": 64, "draft_lend_w": 1}'}
export XZ0B_IMGLITS=${XZ0B_IMGLITS-'B|arena_d0 takes 1 (on) or 0 (off)
B|draft_lend_hyst takes 0 (off) .. 100
B|draft_lend_w takes 1 (on) or 0 (off)
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-W: %zu stretch(es)
B|lever Z0-A set up
B|hyst_holds
B|w_lent_units
B|RADIANCE_DRAFT_CTXSKIP_SLOTS'}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0ab3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0b: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
