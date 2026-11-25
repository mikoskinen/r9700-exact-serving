#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzu-z0occ.sh (queue item, 11.11.2026, ds-main-port Z0 agent; sorts right after 57-p4zyzzzzz-ds-iejzzzt-tree6, before iek-fn-x3arm (the
# lead 11.11.: a stable depth 0 at c4 is the c4 parity lever, so early); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~55 min) --
# LEVER Z0-O, D7 LOAD STABILITY + THE PER-RANK READING (tag "RESULT xz0o ", out runs/xdsm/z0occ, body
# queue-ar/dsmain/dsm-z0occ.sh = dsm-z0ab3.sh + driver z0blocks2.py (per-rank units / hit / misses / stream) + occsum()).
# FIRST, NO DEVICE (z0lvl8): the 8-token arena level must be accepted, else VIRHE and nothing runs.
# Image radiance-local:ds41-z0-8f171a49-5c13e555-l (ds41-int3 1116248 + Z0-H/A/W + the 8-row fixes + Z0-O a1ca3c9: fn4
# draft_occ / draft_occ_k, depth 0 by sustained occupancy, default off + 8f171a4: /stats experts.ranks).
# (1) NST 0 start (stack2's N pin), blocks N N (1u essay / code, essay by position, edit, mb c4 / c8 75 s) with per-rank
#     /stats around every phase and units sampled every 2 s through c4 / c8 -- the same-item NST 0 reference and its
#     per-rank units / misses; no 136k prefill.
# (2) ONE NST 5 start (hxzzy-stack2's NST 5 env and pin + 8,48 rungs + ring 129 + RADIANCE_DRAFT_LEND_WEIGHTS=0, as
#     z0ad-fix's A), IN PROCESS blocks A B B A A B B A:
#       A = today (D7 decides; every Z0 switch 0);
#       B = draft_occ 3, draft_occ_k 32 (depth 0 once >= 3 decode rows held 32 cycles, no probe there; D7 below 3 rows).
#     B vs A pairs (t95); per block and arm the depth-0 share (zero cycles / decode steps) at c4 / c8 / 1u, mean / sd /
#     min / max; per rank (r0..r3) hit %, missed units a step, stream MB a step, VRAM units (trunk / lent) per phase,
#     NST 5 A / B vs NST 0. Identity: 28k copy + 20-prompt greedy set at c1 vs bz d0 under A, B, and B + draft_occ 1 /
#     draft_occ_k 8 (depth 0 held by occupancy at one row).
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Exact: D7's choice of depth only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzu-z0occ.sh : prints every command; checks the image (+ the Z0-O literals), every helper (files,
# parse, functions), the sidecar and the yaml, and runs the no-device 8-row level check; no server, no GPU, writes nothing.
export XZ0B_IMAGE=${XZ0B_IMAGE:-radiance-local:ds41-z0-8f171a49-5c13e555-l} XZ0B_ITEM=${XZ0B_ITEM:-57-p4zyzzzzz-ds-iejzzzu-z0occ}
export XZ0B_OUT=${XZ0B_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/z0occ}
export XZ0B_NST0=${XZ0B_NST0:-1} XZ0B_ORDER0=${XZ0B_ORDER0:-NN}
export XZ0B_ORDER=${XZ0B_ORDER:-ABBAABBA} XZ0B_ARENA_ROWS=${XZ0B_ARENA_ROWS:-8,48}
export XZ0B_LOGLITS=${XZ0B_LOGLITS:-lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:}
export XZ0B_STARTENV=${XZ0B_STARTENV-RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0}
export XZ0B_ARMA=${XZ0B_ARMA:-'{"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 0, "draft_occ_k": 32}'}
export XZ0B_ARMB=${XZ0B_ARMB:-'{"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 3, "draft_occ_k": 32}'}
export XZ0B_ARMC=${XZ0B_ARMC-} XZ0B_ARMD=${XZ0B_ARMD-}
export XZ0B_IDF=${XZ0B_IDF-'{"draft_occ": 1, "draft_occ_k": 8}'}
export XZ0B_IMGLITS=${XZ0B_IMGLITS-'B|draft_occ takes 0 (off) .. 64
B|draft_occ_k takes 1 .. 4096
B|occ_cycles
B|draft_lend_hyst takes 0 (off) .. 100
B|arena_d0 takes 1 (on) or 0 (off)
B|draft_lend_w takes 1 (on) or 0 (off)
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|draft_lent_units
B|trunk_vram
B|RADIANCE_DRAFT_CTXSKIP_SLOTS'}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0occ.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0o: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
