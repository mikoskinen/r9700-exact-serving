#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-c-dst.sh (queue item, STAGED 11.11.2026, ds-main-port DST agent for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-b-stack4 (the coordinator's order r00zz -> a-lf -> b-stack4 -> c-dst) in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~115-150 min (estimate), NO evict) -- LEVER DST (decode stall during prefill), tag
# "RESULT xdst ", out runs/xdsm/dst, body queue-ar/dsmain/dst/dsm-dst.sh (+ dstld.py = pskld.py + the decode users' gaps split
# by request). NOT QUEUED: it goes to queue/ only after r00zz and the r04 readout, on the merged image (ds41-int5 + PSK design 2
# + the PSK fix + the KEPT sched_bi_chunk policy (ds41-bi 3d19a2c + 49989ab + db15e15) + ds41-dst + the PSK glue ds41-int7-pskglue
# 0391a6e: design 2 latched on every chunk path, /stats "psk"), with XDS_IMAGE / XDS_BIC / XDS_PSKV
# set below to that image, the kept option and PSK design 2's value, and a DRYRUN=1 of it passed with that image.
# WHY (coordinator 11.11.): c4 decode streams drop to 26.6 tok/s (vs ~145-150) while one cold 64k prompt prefills (r00z A;
# TPOT p90 892 ms = the 2048-row mixed step every 9th token; the 6.66 s "max TPOT" = a c4 request's TTFT at its turnover).
# Levers (ds41-dst, core fn4, read at every step planned): sched_interleave (--prefill-interleave's count, 8 today) and
# sched_mixed_tok (the shared budget beside decode, 2112 today; 1088 / 576 -> 1024 / 512-row mixed chunks on their grid
# under sched_bi_chunk 3 / 4). ONE NST 5 START = r00z's + RADIANCE_LOG_STEPS=1; cells interleave 8 / 16 / 32 x budget
# 2112 / 1088 / 576 x PSK 0 / on, two blocks (the second reversed, budget-capped); readings in the prefill window: c4 tok/s,
# TPOT inside a request p50 / p90 / p99 / max, the c4 turnovers' TTFT, mixed / decode step ms, the probe's TTFT and chunks;
# identity loaded vs alone (same PSK) and vs the base (alone, PSK 0, sched_bi_chunk 0). The two dearest corners (16 / 32 at
# 576) are left out (XDS_SKIP). BYPASS SECTION (coordinator): sched_short_bypass 0 / 1280 / 2112 at (8, 2112) and the best
# interleave at 2112, PSK 0 / on, ABBA, each beside c4 + a fixed set of 4 ~1000-token newcomers sent at +15 s: their TTFT and
# the c4 turnovers' TTFT p50 / p90 / max, TPOT in a request p50 / p99 / max, long TTFT, c4 tok/s, each newcomer's answer vs alone;
# the bypass cells run sched_bypass_total 4096 (piece by piece), the 4th newcomer is ~3000 tokens (two alone pieces).
# GUARD (dstguard.py): a run whose lever did not act (bypass, D grid cuts, budget, interleave, PSK design 2's /stats "psk"
# counters) is a VIRHE with no speed reading.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-c-dst.sh : prints every command; checks the image literals, every helper, the sidecars, the
# yaml; no server, no GPU, writes nothing.
# 12.11. (coordinator, l21 guard): psk.layers_skipped counts only WHOLE-layer skips; with decode rows / a bypassed newcomer in every chunk step no layer is skipped whole while the cut acts (l21: cut_passes +49, rows_skipped +560548 / +731520, passes_ended_early 0) -> a PSK cell must grow latched / planned_steps / cut_passes / rows_skipped_*.
export XDS_PSK_STATS=${XDS_PSK_STATS:-psk.latched psk.planned_steps psk.cut_passes psk.rows_skipped_l21_29 psk.rows_skipped_l30_39}
export XDS_IMAGE=${XDS_IMAGE:-radiance-local:ds41-int7-0391a6ec-5c13e555-l} XDS_ITEM=${XDS_ITEM:-57-p4zyzzzzz-ds-r00zzz-c-dst}
export XDS_OUT=${XDS_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/dst}
export XDS_BIC=${XDS_BIC:-3} XDS_PSKV=${XDS_PSKV:-2} XDS_PSKK=${XDS_PSKK:-8} XDS_ILS=${XDS_ILS:-8 16 32} XDS_MTS=${XDS_MTS:-2112 1088 576}
export XDS_BLOCKS=${XDS_BLOCKS:-2} XDS_BUDGET_MIN=${XDS_BUDGET_MIN:-150} XDS_LDEPTH=${XDS_LDEPTH:-64000} XDS_DS=${XDS_DS:-1}
export XDS_SKIP=${XDS_SKIP-16:576 32:576} XDS_BYPASS=${XDS_BYPASS-1280 2112} XDS_BYPIL=${XDS_BYPIL:-auto} XDS_NEWC=${XDS_NEWC:-4}
export XDS_NEWC_AT=${XDS_NEWC_AT:-15} XDS_NEWC_TOK=${XDS_NEWC_TOK:-64} XDS_BYP_RESERVE_MIN=${XDS_BYP_RESERVE_MIN:-20}
export XDS_NEWC_LONG=${XDS_NEWC_LONG:-3} XDS_BYPTOT=${XDS_BYPTOT:-4096}
export XDS_SIDECAR=${XDS_SIDECAR:-dspark-dlw-q8_0.rad} XDS_SIDESHA=${XDS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dst/dsm-dst.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdst: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
