#!/bin/bash
# 57-p4zyzzzzz-ds-kb-idxbal.sh (queue item, 10.11.2026, ds41-idxb agent, lever I1; gpu02, all 4 cards, TP4, MML 524288, NST 0,
# every radiance container DOWN; ~40 min) -- DeepSeek-V4.1 Flash EXL3 2.25 bpw: the indexer row split's BALANCED DEAL, an
# exact lever A/B'd IN ONE PROCESS (POST /admin/fn4 idx_split: A = 0 the plain deal, today's issue byte for byte; B = 1 an
# equal block of rows a rank where the 4 ranks do not divide a step's score runs -- a mixed step of a 2048-row chunk beside
# decode rows is 9 runs dealt 3 / 3 / 3 / 0 under A, 12 runs 3 a rank under B; a lone 4096 / 2048-row prefill step is the same
# under both). The 2:2:1:1 index_score skew of the dsm-225-prof 128k window is an opsplit.py artefact (per rank summed it is
# 4135 / 4148 / 4195 / 4260 ms, +3.0 %): EXPECTED fresh lone prefill +-0, mb pf (mixed steps) ~+0.3..1 %.
# IMAGE radiance-local:ds41-idxb-d30fd7d9-5c13e555-l (XIDX_IMAGE) = radiance-local:ds41-gg-d5bed91f-5c13e555-l + ds41-idxb's
#   bin/radiance (the fn4 registry knows idx_split) and deepseek41.so (workstation build/idxb/{run,assemble}.sh;
#   /opt/radiance/share/radiance/idxb-PROVENANCE.txt). t3 env, the dsm 225 serve line.
# PHASES (queue-ar/dsmain/dsm-idxbal.sh): start 1 -- gate, modes (+ the deal note), switch note (2 refused, 1 / 0 applied),
#   warm-up answers vs stock, decode check mb c4 30 s at 0 and 1 (must not move), fresh prefill 16k / 64k / 136k in blocks
#   A B B A (thermal line each), mb pf A B B A, identity: the 28k copy + the 136k deep answer at 1 cold vs today's files
#   (runs/xdsm copy-dsm-225-i1-run1.txt, deep-dsm-225.txt); start 2 (--profile-ops, PROFILED) -- the copy + deep at 0 cold
#   (vs B, vs today's files), then mb pf at 0, mb pf at 1, a fresh 128k prompt at 1: index_score per rank summed (idxbal.py;
#   acceptance within ~5 % across ranks with the balance on). No cc_sim, no 2.9. Nothing goes into a production line
#   without the user's go.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-kb-idxbal.sh : prints every command, runs and writes nothing.
export XIDX_IMAGE=${XIDX_IMAGE:-radiance-local:ds41-idxb-d30fd7d9-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-idxbal.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xidx: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
