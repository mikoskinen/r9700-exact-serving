#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-w-kv4.sh (queue item, STAGED 12.11.2026 by the ds-main-port kv4 builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-v-scr3 and 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5, before 57-p4zyzzzzz-ds-r00zzz-cz-x-fnhcbi and
# 57-p4zyzzzzz-ds-r00zzz-d-slotsum, in C and en_US order; gpu02, all 4 cards, TP4, every radiance container DOWN; ~83 min
# (estimate; budget XKV_BUDGET_MIN 120: a start that would end past it is skipped and logged), NO evict) -- LEVER KV4: the packed
# FP4 compressed latent (RADIANCE_DS41_LATENT=nvfp4, start-level) as the CAPACITY lever for 10+ users, tag "RESULT xkv4 ", out
# runs/xdsm/kv4, body queue-ar/dsmain/kv4/dsm-kv4.sh (from COPIES of stack4/dsm-stack4.sh 21ce8c8e, stack4/mbt.py 94db30da,
# nc/ncsteps.py 0a6e0ed2 (+ nc/ncmb.py ded624e5 / nc/ncone.py a24570f8, which ncsteps.py reads / loads); originals unchanged)
# + kv4sum.py. NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int7-0391a6ec-5c13e555-l (b54f9224a056; ds41-int7 0391a6e, the image stack4 ran) with stack4 S's pins
# (XKV_PINS auto -> s4: ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 2 k 4) + sched_bi_chunk 3 (option D +
# its over-budget cut 0ea8cec = ds41-bi db15e15). No ds41-int8 image was on gpu02 when this was staged; for it set
# XKV_IMAGE=radiance-local:ds41-int8-... (XKV_PINS auto -> s5 = stack5's S5: sched_bi_chunk 3, sched_noa 2, ds41_psk 2 k 8,
# ds41_psk_l20 1, dspark_pfctx 1, ds41_engdd 1 (+ RADIANCE_DS41_ENGDD=0 at start), draft_sample 85, draft_occ 6, ds41_bi 2,
# place_lf 0; XKV_SPIN overrides the JSON) and run its DRYRUN first.
# EXACT BY CODE (deepseek41.cpp:2025-2031 / :4232-4242 / :4868-4870, r4d_ds41_qrt.h:110-137, r4d_ds41_sparse.hip:2734-2771):
# every stored latent row is quant_roundtrip(fp4_e2m1, e4m3, 16) output, which the packer reproduces bit for bit; window keys,
# index keys (mxfp4 in both modes), compressor tails, the drafter's window are not packed. Identity nvfp4 vs bf16 strict.
# STARTS b8 n8 x8 n16 b16 m8 (bf16 / nvfp4 at MAXSEQS 8; bf16 at n8's expert VRAM via --host-pool-mib; nvfp4 / bf16 at MAXSEQS
# 16 = the c16 cell + the > 48-row BI check; bf16 MML 262144 = the smaller-pool price), RADIANCE_DS41_NVGATHER=48 and
# RADIANCE_LOG_STEPS=1 in every start. Readings: capacity (split, tokens, 524288 sessions, units a rank), MAXSEQS 8 / 12 / 16
# per mode, speed (1u, mb c8 / c16 at T 0 with the newcomer step wall and the streamed GB a non-decode step, cold 136k, c4 +
# cold 64k), identity (alone / c8 / loaded / c16 / > 48 rows), counter guards (latent read back, pins read back, PSK).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-w-kv4.sh : prints every command; checks the image literals, every helper (+ selftests),
# the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
export XKV_IMAGE=${XKV_IMAGE:-radiance-local:ds41-int8-7e033848-5c13e555-l} XKV_ITEM=${XKV_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-w-kv4}
export XKV_OUT=${XKV_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kv4}
export XKV_PINS=${XKV_PINS:-auto} XKV_BIC=${XKV_BIC:-3} XKV_NVG=${XKV_NVG:-48} XKV_STARTS=${XKV_STARTS:-b8 n8 x8 n16 b16 m8}
export XKV_HP=${XKV_HP:-19456} XKV_XHP=${XKV_XHP:-auto} XKV_MMLX=${XKV_MMLX:-262144}
export XKV_DUR=${XKV_DUR:-60} XKV_WARM=${XKV_WARM:-15} XKV_C8N=${XKV_C8N:-2} XKV_C16N=${XKV_C16N:-2} XKV_1UN=${XKV_1UN:-2}
export XKV_LIDCONC=${XKV_LIDCONC:-14} XKV_LIDPF=${XKV_LIDPF:-1100} XKV_DISKMIN_MB=${XKV_DISKMIN_MB:-11000} XKV_BUDGET_MIN=${XKV_BUDGET_MIN:-120}
export XKV_SIDECAR=${XKV_SIDECAR:-dspark-dlw-q8_0.rad} XKV_SIDESHA=${XKV_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/kv4/dsm-kv4.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xkv4: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
