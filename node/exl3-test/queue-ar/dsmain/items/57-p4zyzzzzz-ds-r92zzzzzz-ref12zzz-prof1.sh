#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof1.sh (queue item, STAGED 12.11.2026 ~17:00 UTC by the int11 builder for ds-main-port / the
# coordinator -- NOT QUEUED by the builder; sorts after -r92zzzzzz-ref12zz-fn-bi5 in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~11 min estimate, budget XP1_BUDGET_MIN 20 (a section past it skipped, logged), NO evict)
# -- PROF START 1 (UNPROFILED) on int12: ref12's start and pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4,
# exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH / XPREFETCH 0), MAXSEQS 16 (ref12 ran at 8). Tag "RESULT xp1 ", out
# runs/xdsm/prof12/p1, body queue-ar/dsmain/prof12/dsm-prof1.sh (a minimal COPY of ref12/dsm-ref12.sh 82eee978) + prof12/p12win.py.
# The COPY PATH: no ngshare, no ATTACH. READINGS: 1u essay 512 (one discarded warm-up + 3 back to back: tok/s, stream MB / promotions
# a read, ms by verify rows); ONE mb c8 block (harness/ncmb3.py fixed prompt set, 35 s) + its WALL SPLIT (decode / newcomer-MIXED /
# host gaps, agg, decode_share, what 200 tok/s needs, the ref12 basis); ONE c16 block by the CAP48 method (cap48/ncmb.py phase c16,
# T 0, dur 30 warm 10: agg, decode_share, > 48-row steps, steps_capped, drafts_trimmed, vs CAP48 223.2); identity gs16 vs a1 strict
# (all 16 in flight, cap 48; > 48-row steps must be 0) + a1 vs ref12's a1. PROF START 2 (profiled windows) is a separate item.
# XP1_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof1.sh : prints every command;
# checks the image literals (int9 .. int12, the plugin md5), every helper (+ selftests), the sidecars, the yaml, the disk.
export XP1_IMAGE=${XP1_IMAGE:-radiance-local:ds41-int12-6f3e70df-579d0bc2-l} XP1_ITEM=${XP1_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof1}
export XP1_OUT=${XP1_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/prof12/p1} XP1_SW=${XP1_SW:-1}
export XP1_MX=${XP1_MX:-0} XP1_F4=${XP1_F4:-0} XP1_F4G=${XP1_F4G:-0} XP1_MXCONFIRM=${XP1_MXCONFIRM:-0} XP1_GSCONC=${XP1_GSCONC:-16} XP1_MAXSEQS=${XP1_MAXSEQS:-16}
export XP1_EGEN=${XP1_EGEN:-512} XP1_NREADS=${XP1_NREADS:-3} XP1_PLUGMD5=${XP1_PLUGMD5:-9823cdde} XP1_C48REF=${XP1_C48REF:-223.2}
export XP1_C8WIN=${XP1_C8WIN:-35} XP1_C8WARM=${XP1_C8WARM:-10} XP1_C16DUR=${XP1_C16DUR:-30} XP1_C16WARM=${XP1_C16WARM:-10}
export XP1_DISKMIN_MB=${XP1_DISKMIN_MB:-11000} XP1_BUDGET_MIN=${XP1_BUDGET_MIN:-20}
export XP1_SIDECAR=${XP1_SIDECAR:-dspark-dlw-q8_0.rad} XP1_SIDESHA=${XP1_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/prof12/dsm-prof1.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xp1: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
