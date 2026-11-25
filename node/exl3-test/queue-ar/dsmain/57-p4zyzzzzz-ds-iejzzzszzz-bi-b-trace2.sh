#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzz-bi-b-trace2.sh (queue item, 11.11.2026, ds41-bi agent for ds-main-port, lever BI; RERUN of bi-b-trace, which stopped 09:03 UTC on the rule-w recorded-pass check (the tracer records no pass by design; the body now checks rule-w in the env + the tracer lines instead, as z0prof2), and prints the traced loaded-vs-alone count beside the untraced iej-z0ad loadid's (does the tracer reproduce the bug); sorts right after
# the running queue head, before 57-p4zyzzzzz-ds-iejzzzt-tree6 (C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~40 min) -- LEVER BI'S
# ENGINE TRACER UNDER LOAD (tag "RESULT xbil ", out runs/xdsm/bitrace2, body queue-ar/dsmain/dsm-bilid.sh with XBIL_TRACE=1,
# analyzer bitrace.py). The loadid measurement (iej-z0ad: 6 / 20 NST 0, 11 / 20 NST 5 answers differ under load) repeated on the
# ds41-bi image with RADIANCE_DS41_BITRACE on: every traced op's output row fingerprinted (bi_rowhash) for rows at positions
# < 256, alone (a1), under mb c4 + a fresh ~32k prefill (ld), alone again (a2); then per probe the FIRST op whose row differs
# loaded vs alone, with both steps' shapes (rows, decode rows, prefill sequences, the row's class). Two starts, stack2's env,
# serve line and pins (n0 NST 0, n5 NST 5). Tracing: passes not recorded and one sync a step (slower steps; identity is
# unaffected: a fresh issue is the recorded pass's dispatches). NO cc_sim, no 2.9. Measurement only, no lever changes.
# DRYRUN=1 bash 57-p4zyzzzzzzzzzz-bi-b-trace.sh : prints every command; checks the image (its literals incl. bi_rowhash),
# every helper (files, parse, loadid.py and bitrace.py selftests, every function via type -t), the sidecar and the yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-060dff08-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzz-bi-b-trace2}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bitrace2} XBIL_STARTS=${XBIL_STARTS:-n0 n5}
export XBIL_TRACE=${XBIL_TRACE:-1} XBIL_PMAX=${XBIL_PMAX:-256} XBIL_C4RR=${XBIL_C4RR:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
