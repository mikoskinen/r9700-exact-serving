#!/bin/bash
# 57-p4zyzzzzz-ds-pz-vramsweep.sh (queue item, 10.11.2026, DS VRAM-budget audit agent; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~33 min) -- routed-expert residency at decode on today's DS line (EXL3 2.25, NST 0, t3 env,
# gg_gemv 2 + aql_acq 2 switched on in process), by the one exact start-time knob that moves it without a build:
# RADIANCE_PLACE_TURNOVER (lent slots a slab class keeps free for swaps; today 16 a class = 32 kept free, 2948 routed units
# a rank in VRAM at decode). Arms (XVS_ARMS):
#   T0    the lever: 0 kept free -> +32 resident units a rank (exact: placement only)
#   T150  the calibration: 300 kept free -> -268 units vs today; T150 vs T0 = today's ms a step per resident expert at
#         c1 / c4 / c8, which turns every VRAM lever that needs a build (arena decode level, shadow slots, xprefetch ring,
#         router bf16, ...) from GiB into ms
# T16 (today's default) is NOT run (TESTING rule 2): T0's own gain is read as the slope x 32 units; XVS_ARMS="T0 T150 T16"
# adds it only with the coordinator's go. Per arm, decode before prefill: gate, modes (+ the mover's kept-free count),
# state probe, the stack switch, warm-up answers, probe c1 (essay, code), mb c4, mb c8, fresh 136k prefill, the 28k copy,
# the greedy set c1, identity vs bz's d0 (runs/xdsm/draft), per-phase /stats (VRAM units, rank-0 hit, misses, promotions);
# then T150 vs T0 with the slope, each vs the gz gg2-gemv2 blocks (information), identity T150 vs T0. No cc_sim, no 2.9.
# Body: queue-ar/dsmain/dsm-vramsweep.sh (helpers copied from dsm-span4 aa91b3f1 / dsm-draftmr 055848b7 / dsm-gemvacq e0507cb5).
# DRYRUN=1 bash ... : prints, runs and writes nothing.
export XVS_IMAGE=${XVS_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l} XVS_ARMS=${XVS_ARMS:-T0 T150}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-vramsweep.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xvs: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
