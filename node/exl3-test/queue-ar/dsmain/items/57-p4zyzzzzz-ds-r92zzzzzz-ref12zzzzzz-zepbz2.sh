# DROPPED 12.11. — do not queue (coordinator: lever EPB dropped; staging files kept in place)
#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepbz2.sh (queue item, STAGED 12.11.2026 by the EPB builder for ds-main-port / the coordinator --
# NOT QUEUED by the builder; sorts right after -ref12zzzzzz-zepb in C and en_US order -- the name "-zepb2" would sort BEFORE "-zepb" in
# en_US (a digit before a letter once the '.' is ignored), hence "-zepbz2"; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~37 min (start 1 ~22, start 2 ~15), budget XE2_BUDGET_MIN 40, NO evict) -- LEVER EPB FOLLOW-UP (user 12.11.: MEASURE ONLY):
# zepb's O arm with the hot-8 extras declared read c8 128.7 tok/s / 56.1 ms a step vs prof1's 184.0 / 36.2 (lost residency), so:
# START 1 = the slot-order sum with NO extras (RADIANCE_DS41_SLOTSUM=16, no placement file): resident units vs prof1 (VIRHE if lower
# by > 1 %), arms O / S in process: P0 + P0b S (strict), c8 O S S O, c16 newcomer block O / S, 1u essay O / S, staggered S;
# START 2 (budget permitting) = hot-4 extras only (zepb/place/hot4.place as table 1): resident units printed, arms O / E4: P0b E4
# (+ E4 a1 = start 1's S a1: placement invariance), c8 O E4 E4 O, 1u essay E4 / O. No profiled start. Same image as zepb
# (radiance-local:ds41-epb-f637982e-579d0bc2-l 5dc295f627ec), ref12's pins, MAXSEQS 16, the copy path, no ATTACH. Tag "RESULT xep2 ",
# out runs/xdsm/epb2/s1 + /s2, body queue-ar/dsmain/zepb2/dsm-zepb2.sh (a COPY of zepb/dsm-zepb.sh).
# INCIDENT GUARD (zepb c8-5-S-S): ncmb3 --max-wait 8, a per-block guard (requests > 0, window >= 90 %; else VIRHE + that arm's blocks
# skipped), zepb2/streamdiag.py re-sends the incident's two requests under O / S / E4 (INFO).
# XE2_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepbz2.sh : prints every command.
export XE2_IMAGE=${XE2_IMAGE:-radiance-local:ds41-epb-f637982e-579d0bc2-l} XE2_ITEM=${XE2_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepbz2}
export XE2_OUT=${XE2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/epb2} XE2_SW=${XE2_SW:-1}
export XE2_MX=${XE2_MX:-0} XE2_F4=${XE2_F4:-0} XE2_F4G=${XE2_F4G:-0} XE2_MXCONFIRM=${XE2_MXCONFIRM:-0} XE2_GSCONC=${XE2_GSCONC:-16} XE2_MAXSEQS=${XE2_MAXSEQS:-16}
export XE2_EGEN=${XE2_EGEN:-512} XE2_NREADS=${XE2_NREADS:-3} XE2_PLUGMD5=${XE2_PLUGMD5:-9823cdde} XE2_C48REF=${XE2_C48REF:-223.2}
export XE2_C8WIN=${XE2_C8WIN:-35} XE2_C8WARM=${XE2_C8WARM:-10} XE2_C16DUR=${XE2_C16DUR:-30} XE2_C16WARM=${XE2_C16WARM:-10}
export XE2_DISKMIN_MB=${XE2_DISKMIN_MB:-11000} XE2_BUDGET_MIN=${XE2_BUDGET_MIN:-40} XE2_START2=${XE2_START2:-1} XE2_S2MIN=${XE2_S2MIN:-15}
export XE2_SLOTSUM=${XE2_SLOTSUM:-16} XE2_PLACE1=${XE2_PLACE1:-hot4.place}
export XE2_SIDECAR=${XE2_SIDECAR:-dspark-dlw-q8_0.rad} XE2_SIDESHA=${XE2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zepb2/dsm-zepb2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xep2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (restart() runs it again before each compose up)
bash $B
