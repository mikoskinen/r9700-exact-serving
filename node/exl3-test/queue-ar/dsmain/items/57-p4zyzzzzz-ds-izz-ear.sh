#!/bin/bash
# 57-p4zyzzzzz-ds-izz-ear.sh (queue item, 11.11.2026, ds-main-port EAR agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~80 min: n0 ~42 (gate 5, ABBA c1 / c4 / c8 4 x 4, prefill blocks ABFFBA 16k / 64k ~6, the 136k under A and B ~6,
# identity ~6), n5 ~36 (gate 5, settle 2, ABBA c1 / c4 / c8 4 x 5, identity ~6); +5 min a start whose host pool is raised) --
# LEVER EAR on radiance-local:ds41-ear-c5527b78-5c13e555-l (branch ds41-small-ear c5527b7 = ds41-small 2b03903 + EAR; body
# dsm-ear.sh, log tag "RESULT xea ", out runs/xdsm/ear).
# The token embedding without its all-reduce: each rank holds the whole q6_K table (pinned in host memory beside its shard) and
# writes the hc copies as row + 0 -- the exact all-reduce's bytes (its -0.0 comes out +0.0; tests/r4d_emu test_ds41emb on the real
# table). Hypothesis: ~35-50 us a pass at decode; prefill: 17.5 MB read a rank at 4160 rows vs ~64 + 64 MB of all-reduce.
#   Both arms RADIANCE_DS41_EMB_RAM=1 + RADIANCE_DS41_EMB_REP=4160 (declares the lever for every height); START n0 and START n5 (the
#   small item's DSpark config): the start log must show the lever EAR note (n5: trunk and the DSpark query pass) and both tables
#   pinned on 4 ranks (else the start stops); the routed experts' tiers vs the ds41-small reference start -- if the host tier
#   shrank, ONE restart with --host-pool-mib 19974 (19456 + 518), logged "host pool raised" and named in the summary; in process
#   ABBA ds41_embrep 0 (A) / 4160 (B), c1 / c4 / c8; at n0 fresh prefill blocks 16k / 64k under 0 / 4160 / 48 (the decode-only
#   fallback), order ABFFBA, and the 136k cold prefill under A and B; identity per arm (28k copy + greedy c1 vs bz d0) and B vs A.
#   No cc_sim, no 2.9. Nothing goes into a production line from here.
# XEA_IMAGE, XEA_NST, XEA_ORDER, XEA_DUR, XEA_PHASES0, XEA_PHASES5, XEA_DEEP, XEA_PFORDER, XEA_PFDEPTHS, XEA_BASEENV,
# XEA_DRAFTER_MX: set them by editing the export line below before moving this file into queue/ (the queue runner passes no
# environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-izz-ear.sh : prints every command; checks the helpers, the image and the yaml (no server, no GPU),
# writes nothing.
export XEA_IMAGE=${XEA_IMAGE:-radiance-local:ds41-ear-c5527b78-5c13e555-l} XEA_ITEM=${XEA_ITEM:-57-p4zyzzzzz-ds-izz-ear}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-ear.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xea: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
