#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-pskl20.sh (queue item, STAGED 12.11.2026, ds-main-port PSK-L20/DS builder for the coordinator; NOT
# queued: the coordinator queues it after a second reading. Sorts after 57-p4zyzzzzz-ds-r00zzz-cz-ncprof and before
# 57-p4zyzzzzz-ds-r00zzz-d-slotsum in LANG=en_US.UTF-8 and LC_ALL=C order; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~57 min (estimate), NO evict) -- LEVERS PSK-L20 + PSK-DS (ARCH-AUDIT-20261112 levers 1 and 2), tag
# "RESULT xpl20 ", out runs/xdsm/pskl20, body queue-ar/dsmain/pskl20/dsm-pskl20.sh (+ pl20.py; copies of stack4's mbt.py /
# loadid.py / pskld.py and psk2b's p2bq.py; nc/ncsteps.py read only).
# WHY: the official model is a causal encoder-decoder -- layers 0-19 the encoder, 20-39 the decoder, the decoder's global KV
# (latent + index keys) projected from layer 20's attention input -- so PSK's "every row through the last source" is one
# layer too conservative (lever 1, PSK-L20: layer 20's query / indexer / attention / MoE / window store over its cone C_20 =
# 2592 rows, its collapse + norm / compressor / latent + index-key stores over every row; estimate 136k cold K4 33.25 ->
# ~31 s, and non-anchor chunks stage none of layer 20's experts); and the DSpark context pass runs over every trunk row while
# the drafter can only read each sequence's last window + 1 = 129 (lever 2, PSK-DS: estimate -0.3..-0.45 s at 136k, and a
# lighter c8 newcomer step, lever NC).
# DESIGN: image ds41-pskl20 (ds41-int7-pskglue 5093ce9 + 25aab56: fn4 ds41_psk_l20 / dspark_pfctx appended, 44 switches, both 0
# = today; /stats psk.l20_* + dspark_pfctx {mode, passes, rows_skipped, rows_run}). ONE NST 5 START = stack4's (its start line
# runs/ds41-exl3.log 2026-11-11 21:51:34) + RADIANCE_LOG_STEPS=1 (docker logs -t captured, gzipped). Arms in process on stack4's
# fixed pin + S's pins {ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 2, ds41_psk_k 4}: B = l20 0 pfctx 0, L = l20
# 1, D = pfctx 1, LD = both; every arm's switches read back. Readings: COUNTER GUARD (L / LD move the l20 counters, D / LD the
# pfctx counters, B neither, design 2's counters in every arm; a failure = VIRHE, no speed reading); the DRAFTER CHECK (a
# deterministic greedy NST 5 pin, B D LD LD D B on 2k / 8k / 16k texts: answers + drafted / accepted counts equal to the first
# B); MB c8 at T 0 (stack4's mbt.py, dur 60 warm 15) B L D LD LD D L B: agg_dec_tps, decode_share, newcomer step wall from the
# step log; COLD TTFT 136k / 64k / 16k B L D LD LD D L B on fresh distinct texts; IDENTITY vs B: greedy set alone, deepq's fixed
# 136k (+ vs dsm-225), the greedy set at conc 8 (c8 newcomers), c4 + a cold 64k probe (strict where the probe's chunks match B's,
# else class (ii) information at sched_bi_chunk 0), prefix hits on each arm's own text (resume at the K4 anchor below asserted)
# vs B's cold answers on that text.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-pskl20.sh : prints every command; checks the image literals, every helper (+ selftests,
# the text offsets), the sidecars, the yaml; no server, no GPU, writes nothing.
export XPL_IMAGE=${XPL_IMAGE:-radiance-local:ds41-pskl20-25aab56c-5c13e555-l} XPL_ITEM=${XPL_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-pskl20}
export XPL_OUT=${XPL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/pskl20}
export XPL_PSKK=${XPL_PSKK:-4} XPL_OCC=${XPL_OCC:-6} XPL_OCCK=${XPL_OCCK:-32} XPL_DSA=${XPL_DSA:-85}
export XPL_TORDER=${XPL_TORDER:-BLDXXDLB} XPL_MORDER=${XPL_MORDER-BLDXXDLB} XPL_AORDER=${XPL_AORDER-BDXXDB}
export XPL_DEPTHS=${XPL_DEPTHS:-135000 64000 16000} XPL_TNTOK=${XPL_TNTOK:-64} XPL_DUR=${XPL_DUR:-60} XPL_WARM=${XPL_WARM:-15}
export XPL_DISKMIN_MB=${XPL_DISKMIN_MB:-11000}
export XPL_SIDECAR=${XPL_SIDECAR:-dspark-dlw-q8_0.rad} XPL_SIDESHA=${XPL_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/pskl20/dsm-pskl20.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpl20: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
