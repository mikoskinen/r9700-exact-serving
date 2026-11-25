#!/bin/bash
# 57-p4zyzzzzz-ds-iej-z0ac-w.sh (queue item, 11.11.2026, ds-main-port Z0 agent; sorts right after 57-p4zyzzzzz-ds-iej-z0ab (en_US and C order), before
# iek-fn-x3arm / iz-nvgf / j-shexplane; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~36 min) -- LEVER Z0 FOLLOW-UP A/B
# (tag "RESULT xz0w ", out runs/xdsm/z0ab2, body queue-ar/dsmain/dsm-z0ab2.sh = dsm-z0ab.sh + the new registry tail, driver
# z0blocks.py). ONE NST 5 start of radiance-local:ds41-z0-db9d30a9-5c13e555-l (sha256:de54c5376128: ds41-int3 1116248 + Z0-H 32b96af + Z0-A bc1f764 / 61cbb39 + Z0-W 7d8d3d9 / db9d30a),
# hxzzy-stack2's NST 5 env and pin + the arena rungs 8,48 + the 129-slot context-skip ring (as iej-z0ab) + RADIANCE_DRAFT_LEND_
# WEIGHTS=0 (lever Z0-W declared, off: the drafter's query-pass weights get a pinned host copy and are carved as a fourth lender;
# nothing moves while draft_lend_w is 0). IN PROCESS, blocks A B C C B A (1u essay / code, essay by position, edit, mb c4 / c8
# 75 s; /stats experts + GET lend, incl. Z0-W's w_* counters, around every phase):
#   A = stack2's pin + draft_lend_hyst 0 + arena_d0 0 + draft_lend_w 0 (today);
#   B = arena_d0 1 + draft_lend_hyst 20 (= iej-z0ab's B);
#   C = B + draft_lend_w 1 (lever Z0-W: ~0.3-0.4 GiB a rank of the drafter's q / o / shared expert / router / Markov weights
#       lent to the trunk's experts while depth 0 holds and the slab is lent, recalled + copied back before a probe's head).
# Z0-F (the ~30 empty lent drafter slots) is NOT an arm: the second reading (mover code: fill takes the lowest free slot, the
# working reserve -- 16 shadow slots a trunk class -- stays free at the highest slots, which are the drafter's segments, carved
# last; 221 - 32 = 189 vs the measured 188-192) says they are the trunk's working reserve, which NST 0 keeps too.
# Identity: the 28k copy + the 20-prompt greedy set at c1 vs bz d0 under A, B, and B + draft_lend_w 1 with depth 0 forced
# (draft_depth0 2, probes every 64: a weight recall + copy-back at every probe). Reference: hxzzy-stack2's NST 0 blocks.
# Summary: B vs A, C vs B (pairs, t95), each vs stack2 NST 0, VRAM units / hit / stream / lent per phase per arm.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Exact: placement and D7's choice only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-z0ac-w.sh : prints every command; checks the image (+ the Z0 literals), every helper (files,
# parse, functions), the sidecar and the yaml; no server, no GPU, writes nothing.
export XZ0B_IMAGE=${XZ0B_IMAGE:-radiance-local:ds41-z0-db9d30a9-5c13e555-l} XZ0B_ITEM=${XZ0B_ITEM:-57-p4zyzzzzz-ds-iej-z0ac-w}
export XZ0B_ORDER=${XZ0B_ORDER:-ABCCBA} XZ0B_ARENA_ROWS=${XZ0B_ARENA_ROWS:-8,48}
export XZ0B_LOGLITS=${XZ0B_LOGLITS:-lever,Z0-A,set,up|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:}
export XZ0B_STARTENV=${XZ0B_STARTENV-RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0}
export XZ0B_ARMA=${XZ0B_ARMA:-'{"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0}'}
export XZ0B_ARMB=${XZ0B_ARMB:-'{"draft_lend_hyst": 20, "arena_d0": 1, "draft_lend_w": 0}'}
export XZ0B_ARMC=${XZ0B_ARMC-'{"draft_lend_hyst": 20, "arena_d0": 1, "draft_lend_w": 1}'}
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
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0ab2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0w: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
