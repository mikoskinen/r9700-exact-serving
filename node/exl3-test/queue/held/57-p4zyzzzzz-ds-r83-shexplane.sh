#!/bin/bash
# 57-p4zyzzzzz-ds-j-shexplane.sh (queue item, 10.11.2026, ds41-d3 agent, lever F4; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~60 min) -- the MoE shared expert on the second lane beside the routed EXL3 experts at
# decode / verify steps (<= 64 rows), joined back before the gather; A/B'd IN ONE PROCESS by the switch ds41_shexp_lane
# (0 = one lane, today = A; 1 = the lane = B; RADIANCE_DS41_SHEXP_LANE=1 at start) on today's line (ARGATHER on at 4 rows,
# pinned): blocks A B B A A B (stepprobe c1 -- the headline, est. -0.5..-1.2 ms a 1-user step --, mb c4, mb c8), the c4
# greedy set B vs A against A's own floor, identity (the 28k copy and the c1 greedy set under B and A vs bz's dsd-d0 files and
# B vs A). Exact by construction. NST 0. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-d3-c03c57d9-5c13e555-l (XD3_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-j-shexplane.sh : prints every command, runs and writes nothing.
export XD3_IMAGE=${XD3_IMAGE:-radiance-local:ds41-d3-c03c57d9-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-shexplane.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xd3: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
