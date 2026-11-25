#!/bin/bash
# 57-p4zyzzzzz-ds-iej-z0ad-loadid.sh (queue item, 11.11.2026, ds-main-port; sorts right after 57-p4zyzzzzz-ds-iej-z0ac-w and before
# 57-p4zyzzzzz-ds-iejzz-dlw (en_US and C order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~27 min) --
# IDENTITY UNDER LOAD (tag "RESULT xlid ", out runs/xdsm/loadid, body queue-ar/dsmain/dsm-loadid.sh, driver loadid.py).
# The EXL3 routed-expert GEMMs take the prefill form above 16 tokens a step (equal to the decode kernels only to 4e-3 rel L2):
# c4 / c8 drafting verifies and mixed steps with a prefill chunk. Every identity check so far was 1-user; this measures load.
# TWO STARTS of radiance-local:ds41-int3-1116248a-5c13e555-l (e19c28ce372f) with hxzzy-stack2's env, serve line and pins:
#   n0 = NST 0 stack2 N; n5 = NST 5 stack2 A (walk 1, small n5 set, VRAM levers, DL lend 1 + back-off 3, D7, hybrid 1, drafter_mx 1).
# In each: the 20-prompt greedy set c1 ALONE (greedyset.py, temperature 0, 192 tok), then the same 20 prompts EACH under load
# (mb.py c4 streams + one fresh ~32k-token prefill started 0.5 s after the probe, so mixed steps ride along its decode), then
# the set alone again (control). Byte for byte: loaded vs alone (same start), loaded vs bz d0, alone vs bz d0, control vs alone;
# each differing prompt's first differing char and token. One summary line per start + "RESULT xlid summary:".
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Measurement only, no lever changes.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-z0ad-loadid.sh : prints every command; checks the image (its literals), every helper (files,
# parse, loadid.py selftest, every function via type -t), the sidecar and the yaml; no server, no GPU, writes nothing.
# Knobs (edit the export line before moving into queue/; the queue runner passes no environment): XLID_IMAGE, XLID_STARTS,
# XLID_OFFSET, XLID_PFTOK, XLID_CONC, XLID_WARM, XLID_MT, XLID_CTRL.
export XLID_IMAGE=${XLID_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l} XLID_ITEM=${XLID_ITEM:-57-p4zyzzzzz-ds-iej-z0ad-loadid}
export XLID_STARTS=${XLID_STARTS:-n0 n5} XLID_OFFSET=${XLID_OFFSET:-0.5} XLID_PFTOK=${XLID_PFTOK:-32768} XLID_CTRL=${XLID_CTRL:-1}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-loadid.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xlid: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
