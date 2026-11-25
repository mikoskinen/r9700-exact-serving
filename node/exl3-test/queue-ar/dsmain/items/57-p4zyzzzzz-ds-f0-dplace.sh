#!/bin/bash
# 57-p4zyzzzzz-ds-f0-dplace.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~22 min) -- drafting lever D4: the DSpark drafter's routed experts pinned in the host pool
# (--draft-expert-placement host) so the trunk keeps the ~232 VRAM expert units the drafter takes under the default auto
# placement (bz 10.11.: probe hit 98.2 -> 97.3 %, host stream 86 -> 490 MB a step at NST 5). Placement only = the same
# bytes; output-exact by construction (identity still checked). ONE arm, d5ah = bz's d5a (NST 5 + --draft-window adaptive)
# + --draft-expert-placement host, on bz's image and env: gate, probe (essay / code), mb c4 / c8, 28k copy, greedy set c1,
# identity vs bz's d0 files, then the numbers vs bz's d5a (no baseline re-run; the same O dir runs/xdsm/draft). No profile,
# no oracle (the oracle needs the ds41-gg core fix: item h). Host start checked GPU-free 10.11.: the flag parses and the
# drafter declares. IMAGE radiance-local:ds41-main-e736c019-5c13e555 (XDSM_IMAGE; bz's).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-f0-dplace.sh : prints every command, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
export XDSD_ITEM=57-p4zyzzzzz-ds-f0-dplace XDSD_ARMS=${XDSD_ARMS-d5ah} XDSD_PROF=${XDSD_PROF-0} XDSD_ORACLE=${XDSD_ORACLE-0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-draft.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsd: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || [ -s $HOME/gpu/ds41/exl3-test/runs/xdsm/draft/copy-dsd-d5a-i1-run1.txt ] || { echo "$(date '+%F %T') RESULT xdsd f0-dplace: VIRHE no bz d5a reference in runs/xdsm/draft -- not starting" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
