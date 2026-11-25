#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvr.sh (queue item, STAGED 12.11.2026 by the ds-main-port kvr builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-w-kv4 and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4, MML
# 524288, every radiance container DOWN; ~33 min (estimate; budget XKVR_BUDGET_MIN 45), NO evict) -- LEVER KVR (the host tier's
# capacity x4, DPA-DESIGN-20261112.md spin-off P0), tag "RESULT xkvr ", out runs/xdsm/kvr, body queue-ar/dsmain/kvr/dsm-kvr.sh (from
# COPIES of stack4/dsm-stack4.sh 21ce8c8e, stack4/s4blocks.py 0a2cca92, stack4/mbt.py 94db30da, stack4/editprobe.py 96a6b775,
# stack4/loadid.py ee7d9663, dsmain/resume_ho.py b30dab40, all beside it; originals unchanged) + kvrsum.py (tier lines, hit counters,
# TIER_VERIFY lines, report). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-kvr-d5a0c602-5c13e555-l (ds41-kvr d5a0c60 = ds41-int7-pskglue 5093ce9 + lever KVR), layered on the int7 image.
# LEVER KVR: RADIANCE_DS41_KV_REPLICATED=1 (start-time; a declare-time layout, no fn4 switch) declares deepseek41's prefix-cached
# groups kv_index_r1/_r2 + kv_compressed_r1/_r2 rad_kv_group_replicated: a host-tier slot holds them once (rank 0 writes, every rank
# reads back) -- 5.39 KiB instead of 21.56 KiB a 2-token entry, 32.7M instead of 8.17M tokens in the 84 GiB tier. Exact by code:
# every rank stores the same bytes (replicated stores off the exact-wire residual stream); storage only, the program unchanged.
# THREE STARTS (XKVR_ORDER "A B V"), each stack4's NST 5 start (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start")
# on the kvr image under stack4's S pins: A KVR off, B KVR on, V KVR on + RADIANCE_DEBUG_TIER_VERIFY=1. Per start: (b) the host
# tier's capacity (start log tier lines + /sessions), (a) a hit served FROM THE HOST TIER (cold nonce 32k prompt -> its VRAM copies
# dropped (polled on /sessions, filler prompts if not) -> the same prompt: promoted entries / restores) vs the cold, text + token
# ids + logprobs, (c) A / B only: 2 blocks of 1u essay / code + mb c8 at T 0 (dur 60 warm 15; cross-start, labelled), (d) the
# greedy set alone. Strict compares across starts (a1, the cold, the hit) and vs bz d0 / stack4's a1-S.
# GUARD: the env read back from the container, the tier slot quartered (B / V) / one window a rank (A) and tokens x4, the four
# groups named; the hit's /sessions counters moved; V: no replicated mismatch, no round-trip mismatch. Otherwise VIRHE.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvr.sh : prints every command; checks the image literals, every helper (+ selftests),
# the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
export XKVR_IMAGE=${XKVR_IMAGE:-radiance-local:ds41-kvr-d5a0c602-5c13e555-l} XKVR_ITEM=${XKVR_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvr}
export XKVR_OUT=${XKVR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kvr}
export XKVR_ORDER=${XKVR_ORDER:-A B V} XKVR_BLOCKS=${XKVR_BLOCKS:-2} XKVR_DUR=${XKVR_DUR:-60} XKVR_WARM=${XKVR_WARM:-15}
export XKVR_TDEPTH=${XKVR_TDEPTH:-32000} XKVR_TTOK=${XKVR_TTOK:-256} XKVR_DROPWAIT=${XKVR_DROPWAIT:-60} XKVR_FILL=${XKVR_FILL:-2}
export XKVR_DISKMIN_MB=${XKVR_DISKMIN_MB:-11000} XKVR_BUDGET_MIN=${XKVR_BUDGET_MIN:-45}
export XKVR_SIDECAR=${XKVR_SIDECAR:-dspark-dlw-q8_0.rad} XKVR_SIDESHA=${XKVR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/kvr/dsm-kvr.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xkvr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
