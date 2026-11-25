#!/bin/bash
# 57-p4zyzzzzz-ds-a-main.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~80 min) -- DeepSeek-V4.1 Flash EXL3 2.25 (+ a short 2.9) bpw on the Radiance MAIN line (PR #266 head 671acfa =
# main 451b136 + the deepseek41 plugin) with every DS tuning switch, vs the tuned3l line (1.17.0 base) on the same .rad
# files and env. User 10.11.: "tee". No cc_sim (user 10.11.).
# IMAGE radiance-local:ds41-main-e736c019-5c13e555 (XDSM_IMAGE; build/main/{build,run-exl3,assemble}.sh on the workstation):
#   ds41-main e736c019 (PR head 671acfa + ds41-lat + ds41-ar + deepseekv41 line + ds41-t3x + ds41-ar-acq) built clean by
#   upstream's Dockerfile (host ctest stage), + exl3-main 5c13e555 (x3c fa0e958 kernels, ABI 37 pin).
# PHASES (queue-ar/dsmain/dsm-serve.sh; user 10.11.: 2.25 is the DeepSeek line, 2.9 only a short arm): the 2.25 arm --
#   gate, modes, warm-up answers vs stock, slow-state probe (mb c8 45 s), probe, mb c4, mb c8, THEN mb pf, mb turns, i1 copy
#   + deep answer (identity byte for byte vs 105's tuned3l files), BetterBench --quick (1/2/4/8); a stamped 2.25 start
#   (gapcat dependent-launch split vs 117); a --profile-ops 2.25 start (c4 / c8 30 s + one fresh 128k prompt, per rank by
#   op class; PROFILED = not a speed); last a SHORT 2.9 arm (gate, warm-up answers, state probe, probe, mb c4 / c8, i1 copy).
# Time: 225 arm ~38 min, gaps ~9 min, profile ~12 min, 29 short arm ~18 min = ~80 min.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-a-main.sh : prints every command, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-serve.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
