#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp.sh (queue item, STAGED 12.11.2026 by the ds-main-port int9 builder for the coordinator; sorts
# after 57-p4zyzzzzz-ds-r00zzz-cz-x-kvr and before 57-p4zyzzzzz-ds-r00zzz-cz-xz-fnsmoke2 / 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and
# en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~32 min estimate (budget XSP_BUDGET_MIN 40: a
# section that would start past it is skipped), NO evict) -- LEVER SP (the prefill stager at the FULL plan), tag "RESULT xsp ", out
# runs/xdsm/sp, body queue-ar/dsmain/sp/dsm-sp.sh (from COPIES of stack5/dsm-stack5.sh 63649414 and s5ctr.py 256dfd9a -> spctr.py,
# sahb/ncone.py a24570f8 -> ncone.py, pskcr/pcr3.py ac7e1658 -> spchain.py; + spsum.py; originals unchanged). NOT QUEUED: the
# coordinator queues it.
# IMAGE radiance-local:ds41-int9-0cb6fd31-5c13e555-l (ds41-int9 0cb6fd3 = ds41-int8 7e03384 + ds41-kvr d5a0c60 (lever KVR,
# RADIANCE_DS41_KV_REPLICATED, declare-time) + lever SP: fn4 stage_full_min (a full-plan step below N rows streams its pooled experts
# in-kernel) / stage_full_pick (an armed full-plan step copies only the units the last K stager-size steps routed), appended after
# place_hb; GET /admin/fn4 "sp"), layered on the int8 image; both SP switches 0 and KVR unset = the int8 image (build/int9 hostcheck).
# ONE NST 5 START = stack5's (stack4's + RADIANCE_LOG_STEPS=1 + RADIANCE_DS41_ENGDD=0 + RADIANCE_PLACE_HB_RANKS=1,2), KVR env OFF (only
# SP is measured). DS BASE PIN = stack5's S5T (D3 + NOA 2 + ds41_psk 4 tail-only k 8 + ds41_psk_cr 1 + L20 + PF + E1 + SD 85 + occ 6 +
# bi 2). Arms in process: B (SP off) / SP-S (stage_full_min 4161: stream every full-plan step) / SP-P1, SP-P2 (stage_full_pick 1 / 2) /
# SP-best (the screen's composite) / SP-P (the faster pick at the largest size).
# Readings: (a) stager screen -- fresh one-token alone prompts at ~2.3k / 3k / 3.9k tokens, B S 1 2 2 1 S B x 4 (8 an arm a size):
# prefill step ms, staged / streamed GB per rank, GET "sp"; the faster SP form per size -> SP-best; (b) alone tails 1.5k / 2.5k / 3.5k /
# 5k, B X X B x 4; (e) c8 (coordinator 12.11., KV4's co-batched 2162-3290-row newcomer steps arm the full-plan stager): mb c8 T 0
# (stack4/mbt.py) dur 60 warm 15, B X X B -- agg tok/s, decode_share, the co-batched full-plan step ms from the step log; (c) a CC-like 4-turn chain (16k cold + 3 x (64-token answer + ~2.3k tokens of tool text): tails over 2112 rows),
# B X X B, the same text each: TTFT a turn; (d) cold 64k (deepq) B S P P S B: TTFT, prefill chunk ms. Identity strict (every SP
# answer vs B: chains turn by turn, 64k, an alone set at 2.2k / 3k / 3.9k / 5k under B S P X); counter guard (spctr.py) on every
# reading; freshness on cached == 0 only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp.sh : prints every command; checks the image literals, every helper (+ selftests),
# the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XSP_IMAGE=${XSP_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l} XSP_ITEM=${XSP_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp}
export XSP_OUT=${XSP_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/sp}
export XSP_SIZES="${XSP_SIZES:-2.3k:2200 3k:2900 3.9k:3800}" XSP_SORDER=${XSP_SORDER:-BS1221SB} XSP_SCN=${XSP_SCN:-4}
export XSP_TSIZES="${XSP_TSIZES:-1.5k:1400 2.5k:2400 3.5k:3400 5k:4900}" XSP_TORDER=${XSP_TORDER:-BXXB}
export XSP_CORDER=${XSP_CORDER:-BXXB} XSP_CDEPTH=${XSP_CDEPTH:-16000} XSP_CTT=${XSP_CTT:-2300} XSP_CNTOK=${XSP_CNTOK:-64}
export XSP_DDEPTH=${XSP_DDEPTH:-64000} XSP_DORDER=${XSP_DORDER:-BSPPSB} XSP_IDEPTHS="${XSP_IDEPTHS:-2200 3000 3900 5000}"
export XSP_C8ORDER=${XSP_C8ORDER:-BXXB} XSP_C8DUR=${XSP_C8DUR:-60} XSP_C8WARM=${XSP_C8WARM:-15}
export XSP_SFULL=${XSP_SFULL:-4161} XSP_DISKMIN_MB=${XSP_DISKMIN_MB:-11000} XSP_BUDGET_MIN=${XSP_BUDGET_MIN:-40}
export XSP_SIDECAR=${XSP_SIDECAR:-dspark-dlw-q8_0.rad} XSP_SIDESHA=${XSP_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/sp/dsm-sp.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsp: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
