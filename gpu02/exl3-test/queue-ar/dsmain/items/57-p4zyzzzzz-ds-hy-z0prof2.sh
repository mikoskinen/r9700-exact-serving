#!/bin/bash
# 57-p4zyzzzzz-ds-hy-z0prof2.sh (queue item, 11.11.2026, ds-main-port Z0 agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~16 min) -- LEVER Z0 STEP 1, THE PROFILED HALF AGAIN (tag "RESULT xz0 ", out runs/xdsm/z0prof, body
# queue-ar/dsmain/dsm-z0prof2.sh). hy-z0prof's two --profile-ops starts stopped at the stack2 modes check ("rule-w is not
# what the recorded passes play": --profile-ops records no pass, core/runtime/ctx.cpp tape_on_ needs !profiling_); its
# stamped starts ran. Here: n5prof (NST 5 stack2, forced depth 0, lend engaged) and n0prof (NST 0 stack2), each c4 / c8 mb
# windows + two PREFILL windows (8 fresh ~1170-token prompts, max_tokens 1, one / four at a time) -> per op and per rank at
# NST 5 vs NST 0 for decode, mixed and prefill steps, per-rank all-reduce wait; plus the stamped half read BY STEP CLASS
# (z0steps.py on hy-z0prof's CSVs, CPU) and the /stats split per window; then the ranking. rule-w is replaced by the
# profile sync in the profiled starts (logged; not a speed). Image radiance-local:ds41-int3-1116248a-5c13e555-l
# (e19c28ce372f). NO cc_sim, no 2.9, no identity (nothing changes output). Production compose read only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hy-z0prof2.sh : prints every command; checks the image, every helper (files, parse,
# functions, tools, z0prof.py / z0pf.py / z0steps.py selftests), hy-z0prof's stamped inputs, the sidecar and the yaml; no
# server, no GPU, writes nothing.
export XZ0_IMAGE=${XZ0_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l} XZ0_ITEM=${XZ0_ITEM:-57-p4zyzzzzz-ds-hy-z0prof2}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0prof2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
