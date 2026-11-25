#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-tdst2.sh (queue item, STAGED 12.11.2026 by the ds-main-port noa builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-scr and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4, MML
# 524288, every radiance container DOWN; ~45 min (estimate; budget XDT_BUDGET_MIN 70), NO evict) -- THE NEXT DST ITEM: LEVER NOA,
# tag "RESULT xdst2 ", out runs/xdsm/dst2, body queue-ar/dsmain/dst2/dsm-dst2.sh (= a copy of dst/dsm-dst.sh 659b86c7 with the cells,
# load, guard and summary replaced; + dstld2.py = a copy of dst/dstld.py 24c2506f, dstguard2.py = a copy of dst/dstguard.py ab79e7a5,
# mbt.py = a copy of stack4/mbt.py 94db30da). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-noa-0f0294be-5c13e555-l (a615f42d9f3a; ds41-noa 0f0294b off ds41-int7-pskglue 5093ce9: 3f7fc59 a recycled slot clears the
# bypass's per-request state + the NOA commit: fn4 sched_noa 0 / 1 / 2, GET "noa"), layered on the int7 image.
# WHY (r00zzz-c-dst, 12.11.): under the kept chunking D3 a short newcomer waits for the WHOLE long prefill (it cannot take a partial
# piece beside the 2048-row grid chunk); the bypass acts only on held steps, which need decode rows. NOA: a prefill D turns away
# beside another prefill takes D's own piece (alone's whole, else a grid cut >= 128 rows a side) at the step's max_tok room
# (sched_noa 2: as many as fit; over max_tok: waits once, then displaces the long chunk one step, bounded). Exact as D is.
# ONE NST 5 START (stack4's start, runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start") + RADIANCE_LOG_STEPS=1;
# pin in every cell: PSK design 2 k 4, ds41_bi 2, interleave 8, budget 2112, DRAFTING PINNED OFF (draft_depth0 2 + draft_probe 4096).
# CELLS: A = D3 base, N = D3 + NOA 2, Y = D3 + bypass (interleave 8, 1280 + total 4096), NY = D3 + NOA + bypass, Z = sched_bi_chunk 0
# (r00z-like, not exact). LOADS: l = decode ALIVE (a 4096-token greedy ignore_eos stream through the whole prefill + 3 closed-loop
# c4 streams with turnovers) + one cold 64k probe + 3 nonce'd fresh newcomers (~1100 / ~1330 / ~4400 tokens) at +3 / +8 / +14 s
# inside its prefill window; n = the same probe + newcomers with NO decode stream (on purpose); c8 = mb c8 as stack4 (mbt.py T 0,
# dur 60 warm 15) + a c8 identity burst (coordinator 12.11.: cells A / N / Z). ABBA: l A N Y NY Z | c8 A N Z | n A N Y NY Z |
# n Z NY Y N A | c8 Z N A | l Z NY Y N A (the no-decode mirror is dropped past budget - 15 min).
# READINGS: newcomer TTFT p50 / max, c4 turnover TTFT p50 / max, TPOT in a request p50 / p90 / max, long TTFT, c4 tok/s in the
# window, mixed step ms, noa / bypass counters; c8: agg_dec_tps, decode_share, non-decode (newcomer) step ms, newcomers a step;
# identity of the long answer, every newcomer and the c8 burst vs alone (strict under D3, information under Z).
# GUARD (dstguard2.py): pins read back; PSK latched / planned_steps / cut_passes / rows_skipped_* grow, plan_mismatch stays (never
# layers_skipped); NOA counters grow in N / NY and stay 0 elsewhere; the bypass counters likewise (no-decode: information); draft
# tokens ~0 under the pin; newcomers fresh; the long stream alive. Otherwise VIRHE, no speed reading.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-tdst2.sh : prints every command; checks the image literals, every helper, the sidecars,
# the yaml; no server, no GPU, writes nothing.
export XDT_IMAGE=${XDT_IMAGE:-radiance-local:ds41-noa-0f0294be-5c13e555-l} XDT_ITEM=${XDT_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-tdst2}
export XDT_OUT=${XDT_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/dst2}
export XDT_NOA=${XDT_NOA:-2} XDT_BYPASS=${XDT_BYPASS:-1280} XDT_BYPTOT=${XDT_BYPTOT:-4096} XDT_IL=${XDT_IL:-8}
export XDT_PSKV=${XDT_PSKV:-2} XDT_PSKK=${XDT_PSKK:-4} XDT_LDEPTH=${XDT_LDEPTH:-64000} XDT_NTOK=${XDT_NTOK:-128}
export XDT_NEWC_AT=${XDT_NEWC_AT:-3,8,14} XDT_NEWC_CTX=${XDT_NEWC_CTX:-1100,1330,4400} XDT_NEWC_TOK=${XDT_NEWC_TOK:-64}
export XDT_LONG_TOK=${XDT_LONG_TOK:-4096} XDT_CELLS_L=${XDT_CELLS_L:-A N Y NY Z} XDT_CELLS_N=${XDT_CELLS_N:-A N Y NY Z}
export XDT_CELLS_C8=${XDT_CELLS_C8:-A N Z} XDT_C8_DUR=${XDT_C8_DUR:-60} XDT_C8_WARM=${XDT_C8_WARM:-15} XDT_C8_REQS=${XDT_C8_REQS:-2}
export XDT_C8_TOK=${XDT_C8_TOK:-64} XDT_BUDGET_MIN=${XDT_BUDGET_MIN:-70} XDT_DS=${XDT_DS:-1} XDT_PROBE=${XDT_PROBE:-4096}
export XDT_PSK_STATS=${XDT_PSK_STATS:-psk.latched psk.planned_steps psk.cut_passes psk.rows_skipped_l21_29 psk.rows_skipped_l30_39}
export XDT_PSK_STATS_C8=${XDT_PSK_STATS_C8:-psk.latched psk.planned_steps psk.cut_passes psk.rows_skipped_l30_39}
export XDT_PSK_ZERO=${XDT_PSK_ZERO:-psk.plan_mismatch}
export XDT_SIDECAR=${XDT_SIDECAR:-dspark-dlw-q8_0.rad} XDT_SIDESHA=${XDT_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dst2/dsm-dst2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdst2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
