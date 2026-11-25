#!/bin/bash
# 57-p4zyzzzzz-ds-r00y-pskloc.sh (queue item, 11.11.2026, PSK cone-exactness localiser for ds-main-port; sorts before r04 / r05 /
# r06 (C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~16 min, NO evict) -- tag
# "RESULT xpsl ", out runs/xdsm/pskloc, body queue-ar/dsmain/pskloc/dsm-pskloc.sh (= psk/dsm-psk.sh's start + lever BI's engine
# tracer) + pskloc.py. Image radiance-local:ds41-pskfix-77eefeae-5c13e555-l (ds41-psk-fix 77eefea = ds41-psk 4d4da8e + the
# diagnostic switch value ds41_psk 2; layered on the ds41-psk a35c69a5b9c9 image, deepseek41.so only; it carries ds41-bi's
# tracer RADIANCE_DS41_BITRACE; ds41_psk 0 / 1 / unset = the ds41-psk image's).
# WHY (ds-main-port 11.11., r00z): ds41_psk 1 changes the alone 136k / 64k answers (4096-row pure prefill chunks, layers 21-39
# cut) and the alone 16k (2048-row pure prefill chunks: under 16384 tokens nothing spans; layers 25-39 cut), while the loaded 64k
# (2048-row mixed chunks, layers 25-39 cut) is IDENTICAL to A. Which op, at which layer, first
# differs on a row the cone must leave exact? ONE NST 5 start = r00z's + the tracer (positions < 8192), arms A (ds41_psk 0),
# B (1: the cone), C (2: the same psk_only step_bi path, no layer cut -- coordinator 11.11.: C == A -> the cut is the defect,
# C == B -> the step_bi path), each a ~7.9k-token deepq prompt (depth 8000 < the 16384 span threshold: 2048-row pure prefill
# chunks) cold (prefix reset), 512 greedy tokens; pskloc.py names per arm vs A, and B vs C, the first (chunk, layer, op)
# differing on a trusted row (the cone's exact rows; C uncut: every row).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00y-pskloc.sh : prints every command; checks the image literals, every helper; no server, no GPU.
export XPL_IMAGE=${XPL_IMAGE:-radiance-local:ds41-pskfix-77eefeae-5c13e555-l} XPL_ITEM=${XPL_ITEM:-57-p4zyzzzzz-ds-r00y-pskloc}
export XPL_PROV=${XPL_PROV:-pskfix-PROVENANCE.txt} XPL_PAIRS=${XPL_PAIRS-B:C}
export XPL_OUT=${XPL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/pskloc}
export XPL_ORDER=${XPL_ORDER:-ABC} XPL_ARMVALS=${XPL_ARMVALS:-A=0 B=1 C=2} XPL_CUT=${XPL_CUT:-B=21:160:128}
export XPL_DEPTH=${XPL_DEPTH:-8000} XPL_NTOK=${XPL_NTOK:-512} XPL_PMAX=${XPL_PMAX:-8192} XPL_DEEP136=${XPL_DEEP136-}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/pskloc/dsm-pskloc.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpsl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
