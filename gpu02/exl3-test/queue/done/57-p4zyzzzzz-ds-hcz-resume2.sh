#!/bin/bash
# 57-p4zyzzzzz-ds-hcz-resume2.sh (queue item, 10.11.2026, ds41 resume agent, finding R1 = correctness; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~16 min, ~23 with the fix start) -- WHY did hcz's resumes of d1's chain differ from
# cold while every chain written after the prefix-cache reset resumed exactly? Separates three candidates: H1 the first long
# prefill of a start (first-sighting passes), H2 pool pressure (only d1 dropped entries mid-prefill), H3 the shared 2-token head
# entry (BOS, <User>) first published by the warm-up's 27-token question and kept by the cache (core/mem/prefix.cpp insert keeps
# the first publisher's bytes when snapshots exist) while d1's snapshots were computed from d1's own 2048-row bytes.
# Same image and serve line as hcz (ds41-gg-8b311fcc, gg_gemv 0 = today's arithmetic, t3 env, NST 0, MNBT 4160, --live).
# S1 as hcz (warm-up + fastcheck, RADIANCE_DEBUG_TIER_VERIFY=1 = host-side round-trip hashes only): nonce 136k cold as the FIRST
# long prefill + its hit, then deep cold + hit. S2 no warm-up: deep cold as the first request + hit; /reset_prefix_cache; the
# warm-up's three short questions alone; deep cold + hit. S3 only with XHR_FIX_IMAGE (the H3 fix): as S1 on the fix image
# (nonce + deep, cold + hit, and deep cold vs hcz's stock cold). Every request greedy, thinking off, 512 tokens, logprobs top 2
# (resume_ho.py, unchanged); tier counters per request; every pair byte-compared; b2 / f4 also against hcz's d2 (H3 predicts
# byte-identical). Body queue-ar/dsmain/dsm-resume2.sh, driver resume_ho.py, compose ds41-xhr-tp4.yaml. Output
# runs/xdsm/resume2/, log tag "RESULT xhr ". Sorts after -hcy-depth0b and before -hd-hybrid under the runner's sort.
# No cc_sim. Nothing goes into a production line from here.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hcz-resume2.sh : prints every command, runs and writes nothing.
export XHR_IMAGE=${XHR_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l} XHR_ITEM=${XHR_ITEM:-57-p4zyzzzzz-ds-hcz-resume2}
export XHR_FIX_IMAGE=${XHR_FIX_IMAGE-radiance-local:ds41-resfix-38950b02-5c13e555-l}   # S3: the H3 fix (ds41-resfix-gg 38950b0 = hcz image + core/mem fix); XHR_FIX_IMAGE= skips S3
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-resume2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xhr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
