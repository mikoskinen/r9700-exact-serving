#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb.sh (queue item, STAGED 12.11.2026 by the EPB builder for ds-main-port / the coordinator --
# NOT QUEUED by the builder; sorts after -ref12zzzzz-res, -ref12zzzzzz-xpush and the planned -ref12zzzzzz-yd2m / -yll / -ychn / -ypfx
# in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~45-50 min, budget XEP_BUDGET_MIN 55, NO evict)
# -- LEVER EPB (user 12.11. via the coordinator: "tasapainotuksen saa mitata" = MEASURE ONLY; nothing into a base / merge / reference):
# the rank-independent slot-order MoE sum + least-loaded hot-expert replication (ds41-slot 4cf3501 D3s ported onto int12 bf8f557 as
# branch ds41-epb, image radiance-local:ds41-epb-*), RADIANCE_DS41_SLOTSUM=16 (every MoE pass of <= 16 rows: the decode band, also a
# mixed step's decode pieces -- batch-invariant within the arm), tables hot8 / hot4 (epplace.py replicate from cc-x117.json), in process
# by fn4 ds41_slotsum / ds41_place (63 / 64). ref12's start and pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4,
# exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH / XP 0), MAXSEQS 16, the COPY PATH, no ATTACH. Tag "RESULT xepb ", out
# runs/xdsm/epb (+ /prof), body queue-ar/dsmain/zepb/dsm-zepb.sh (a COPY of prof12/dsm-prof1.sh + prof2's profiled summary).
# P0 (4 cards, r4d_selftest slot-sum + placed sort) and P0b (E: a1 = a2, gs16 = a1; O's a1 = ref12's a1; E != O) STOP the item on a
# failure. READINGS: c8 ABBA O E E O + S + E4 (ncmb3 fixed, 35 s, P12SPLIT), c16 CAP48 method E / O + c16 newcomer block E, 1u essay
# E / O (W + 3), staggered arrivals E vs a1E (strict), E a1 vs ref12 a1 (INFO); START B PROFILED c8 E: gather_ar(_slot) us a call vs prof2.
# XEP_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb.sh : prints every command;
# checks the image literals (int9 .. int12 + EPB, the plugin md5), every helper (+ selftests), the placement files, the yaml, the disk.
export XEP_IMAGE=${XEP_IMAGE:-radiance-local:ds41-epb-f637982e-579d0bc2-l} XEP_ITEM=${XEP_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb}
export XEP_OUT=${XEP_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/epb} XEP_SW=${XEP_SW:-1}
export XEP_MX=${XEP_MX:-0} XEP_F4=${XEP_F4:-0} XEP_F4G=${XEP_F4G:-0} XEP_MXCONFIRM=${XEP_MXCONFIRM:-0} XEP_GSCONC=${XEP_GSCONC:-16} XEP_MAXSEQS=${XEP_MAXSEQS:-16}
export XEP_EGEN=${XEP_EGEN:-512} XEP_NREADS=${XEP_NREADS:-3} XEP_PLUGMD5=${XEP_PLUGMD5:-9823cdde} XEP_C48REF=${XEP_C48REF:-223.2}
export XEP_C8WIN=${XEP_C8WIN:-35} XEP_C8WARM=${XEP_C8WARM:-10} XEP_C16DUR=${XEP_C16DUR:-30} XEP_C16WARM=${XEP_C16WARM:-10}
export XEP_DISKMIN_MB=${XEP_DISKMIN_MB:-11000} XEP_PDISK_MB=${XEP_PDISK_MB:-12000} XEP_BUDGET_MIN=${XEP_BUDGET_MIN:-55}
export XEP_SLOTSUM=${XEP_SLOTSUM:-16} XEP_PLACE1=${XEP_PLACE1:-hot8.place} XEP_PLACE2=${XEP_PLACE2:-hot4.place}
export XEP_SIDECAR=${XEP_SIDECAR:-dspark-dlw-q8_0.rad} XEP_SIDESHA=${XEP_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zepb/dsm-zepb.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xepb: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (restart() runs it again before each compose up)
bash $B
