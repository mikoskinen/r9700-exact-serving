#!/bin/bash
# 57-p4zyzzzzzzzzzz-dsm-fnloadid.sh (queue item, 11.11.2026, ds-main-port; LAST in the queue: sorts after
# 57-p4zyzzzzzzzzz-da-f-fidelity in en_US and C order; gpu02, all 4 cards, every radiance container DOWN; ~32 min) --
# FLASH-NEXT IDENTITY UNDER LOAD (tag "RESULT xfnlid ", out runs/xdsm/fnloadid, body queue-ar/dsmain/fn/dsm-fnloadid.sh, driver
# fnloadid.py = DS xlid's loadid.py verbatim). DS's xlid found 6 of 20 greedy answers differ under c4 + a fresh ~32k prefill vs
# alone (control 20 / 20); the user rules an answer changing under load vs alone is a bug -> does Flash-Next have it too?
# THREE STARTS, one a stack: x3 = the fn4 EXL3 line (c16-x3arm 73d564388db8, 173 x3arm's A: X3CODES + exl3_gd 1, map 0 + dz 0,
# fn-lidtp4.yaml = fn-tp4zq.yaml); s3 = the fn4 iq4r stack 3 (c16-rf 37f8739a4a60, 170 M-fix's env, same yaml); prod =
# production's configuration, one DP half (stilldeadcode/radiance:1.15.1 cccbc0bd6bfe, fn-lidprod.yaml = fn-prodres.yaml:
# flashnext-prod-dp-a.yaml's serve line token for token except --host / --port / --api-key-env, own /dev/shm n-gram dir).
# In each: the 20-prompt greedy set c1 ALONE (temperature 0, 192 tok), then each prompt under mb c4 + one fresh ~32k prefill at
# +0.5 s, then alone again (control); byte for byte ld vs a1, a2 vs a1, vs fn4's greedy references (156b x3gd2 on / 146 acq off)
# with each differing prompt's first differing char and token. NO cc_sim. No power / host / fan settings; production compose
# files read only. Measurement only.
# DRYRUN=1 bash 57-p4zyzzzzzzzzzz-dsm-fnloadid.sh : prints every command; checks the three images (ids + their literals,
# no device, no network), every helper (files, parse, fnloadid.py selftest, every function via type -t, tools), both yamls
# (docker compose config per start), the prod serve line vs production's file, models, references; no server, no GPU, writes nothing.
# Knobs (edit the export line before moving into queue/; the queue runner passes no environment): XFNLID_STARTS, XFNLID_OFFSET,
# XFNLID_PFTOK, XFNLID_CONC, XFNLID_WARM, XFNLID_MT, XFNLID_CTRL, XFNLID_NGRAM.
export XFNLID_STARTS=${XFNLID_STARTS:-x3 s3 prod} XFNLID_ITEM=${XFNLID_ITEM:-57-p4zyzzzzzzzzzz-dsm-fnloadid}
export XFNLID_OFFSET=${XFNLID_OFFSET:-0.5} XFNLID_PFTOK=${XFNLID_PFTOK:-32768} XFNLID_CTRL=${XFNLID_CTRL:-1}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/fn/dsm-fnloadid.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xfnlid: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
