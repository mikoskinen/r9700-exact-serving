#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzb-dlwb.sh (queue item, 11.11.2026, ds-main-port DLW agent at the lead's request; sorts right after the
# item running at 07:40 UTC (iej-z0ad-loadid), ahead of iejzzz-sd; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~40 min) -- LEVER DLW FOLLOW-UP (tag "RESULT xdlwb ", out runs/xdsm/dlwb, body queue-ar/dsmain/dlw/dsm-dlwb.sh, driver
# dlwblocks.py, summary dlwsum.py). B (the drafter's dense linears as q8_0 from the weight sidecar dspark-dlw-q8_0.rad) vs A
# (none), cross-start A B A, NST 5, hxzzy-stack2's env as it ran and its pin, image radiance-local:ds41-dlw-df860485-5c13e555-l
# (sha256:867d4364fb7e; unset = int3). iejzz-dlw read the lend at each phase's END only (B / C "lent 0" while their counters show
# lends 5 / reclaims 5 at c4); this item reads it over every phase: the lend counters' deltas (lent_cycles, lent plain steps of
# the decode steps, transit) and a 2 s sampler of the routed VRAM units and the lent units. Per start one block: c1 essay / code
# (temperature 0), mb c4 / c8 natural 75 s; + the 1-user probe at the server-default sampling (seed 1234, 1 run); identity on B
# (28k copy + 20-prompt greedy set c1 vs bz d0). NO cc_sim, no 2.9. No power / host / fan settings; production compose read only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzb-dlwb.sh : prints every command; checks the image, every helper, the sidecar and the yaml;
# no server, no GPU, writes nothing.
export DLW_IMAGE=${DLW_IMAGE:-radiance-local:ds41-dlw-df860485-5c13e555-l} DLW_ITEM=${DLW_ITEM:-57-p4zyzzzzz-ds-iejzzb-dlwb} DLW_ORDER=${DLW_ORDER:-ABA}
export DLW_SHA_Q8=${DLW_SHA_Q8:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dlw/dsm-dlwb.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdlwb: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
