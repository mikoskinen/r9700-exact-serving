#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc.sh (queue item, STAGED 12.11.2026 by the ds-main-port cc builder for the coordinator; USER-ASKED;
# sorts after 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16 and before 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b, -cz-xz-fnsmoke2 and -d-slotsum,
# in C and en_US order; gpu02, all 4 cards, TP4, every radiance container DOWN; ~53 min A / B (estimate; <= ~71 at the cc cap; budget
# XCC_BUDGET_MIN 77: a start that would end past it is skipped and logged), NO evict) -- the EXISTING Claude Code simulation under a
# bf16 vs an nvfp4 compressed latent, tag "RESULT xcc ", out runs/xdsm/cc, body queue-ar/dsmain/cc/dsm-cc.sh (start functions COPIED
# from kv4/dsm-kv4.sh f89a6cc2, the S5T pin from stack5/dsm-stack5.sh 63649414; originals unchanged) + ccsum.py (the /stats sampler and
# the readings) + ncsteps.py / ncone.py (= nc/ncsteps.py 0a6e0ed2 / nc/ncone.py a24570f8, unchanged copies). NO new harness: cc_sim3.py (queue-ar/tools/ccsim, 2512bb16)
# through the kit's ccsim-exl3-turns.sh (0e00acd2), both read only. NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int9-0cb6fd31-5c13e555-l (9af3dddd556c; int9 = int8 7e03384 + KVR + SP) -- XCC_IMAGE=radiance-local:ds41-int8-
# 7e033848-5c13e555-l is the fallback (then XCC_KVR auto = 0). KVR ON in both starts (coordinator 12.11.; RADIANCE_DS41_KV_REPLICATED=1
# via the env file, read back: container env + 'lever KVR declared' + the tier's 'held once a slot' line, bf16 5.39 KiB); SP 0.
# RUNS (XCC_ORDER): "b8 n8" = A / B ONCE, b8 first, labelled (order / drift not cancelled); "n8 b8 b8 n8" = B A A B with fresh starts
# (~1 h 45 min, <= ~2 h 20 min; set XCC_ORDER and nothing else -- the budget follows the run count). b8 = latent bf16, n8 =
# RADIANCE_DS41_LATENT=nvfp4; both MAXSEQS 8, HOSTPOOL 19456, MML 524288, NST 5, kv4's start words (+ RADIANCE_LOG_STEPS=1, NVG 48),
# pin = stack4's fixed pin + S5T (ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 4 k 8, ds41_psk_cr 1, ds41_psk_l20 1,
# dspark_pfctx 1, ds41_engdd 1, sched_noa 2, sched_bi_chunk 3).
# LOAD per start: prefix-cache reset, cc_sim3 --users 8 --turns 32 --warmup-turns 3 --gen-mode natural --temperature 0 --trace-seeded
# --cpt-fixed --seed 1 --effort xhigh,medium (--duration 1500 safety cap): cc_sim's own seed-1 sessions, first sessions 33.5-141.9k
# tokens on this model (p50 ~78k, 632k live), growing ~1-2k a turn, cap 170k then a new 20-40k session; the SAME seed / trace set in
# every start. /stats sampled every 10 s, the step log (RADIANCE_LOG_STEPS=1) captured.
# READINGS: TTFT p50 / p90, decode tok/s a stream + aggregate, completed work a minute (requests, tokens), errors; KV pool fill % over
# time, expert units in VRAM a rank, kv-lent GiB a rank (timeline); streamed GB a non-decode step and a newcomer; the n8 vs b8 pair in %.
# NO identity claims (natural mode). GUARDS (VIRHE): latent read back, KVR read back, pins read back (before and after cc), the trace
# reaching >= 50k-token prompts, records / samples / step-log lines; disk XCC_DISKMIN_MB 11000 before every start.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc.sh : prints every command; checks the image literals, every helper (+ selftests,
# cc_sim3 offline: --help + its session builder = the trace check), the wrapper, the sidecars, the yaml, the disk; no server, no GPU,
# no request to :8090, writes nothing.
export XCC_IMAGE=${XCC_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l} XCC_ITEM=${XCC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc}
export XCC_OUT=${XCC_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/cc} XCC_ORDER=${XCC_ORDER:-b8 n8} XCC_KVR=${XCC_KVR:-auto}
export XCC_USERS=${XCC_USERS:-8} XCC_TURNS=${XCC_TURNS:-32} XCC_WTURNS=${XCC_WTURNS:-3} XCC_SEED=${XCC_SEED:-1} XCC_CAP=${XCC_CAP:-1500}
export XCC_EFFORT=${XCC_EFFORT:-xhigh,medium} XCC_SAMP=${XCC_SAMP:-10} XCC_MINCTX=${XCC_MINCTX:-50000}
export XCC_MAXSEQS=${XCC_MAXSEQS:-8} XCC_MML=${XCC_MML:-524288} XCC_HP=${XCC_HP:-19456} XCC_NVG=${XCC_NVG:-48}
export XCC_DISKMIN_MB=${XCC_DISKMIN_MB:-11000} XCC_EST_MIN=${XCC_EST_MIN:-26}
export XCC_SIDECAR=${XCC_SIDECAR:-dspark-dlw-q8_0.rad} XCC_SIDESHA=${XCC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/cc/dsm-cc.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xcc: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
