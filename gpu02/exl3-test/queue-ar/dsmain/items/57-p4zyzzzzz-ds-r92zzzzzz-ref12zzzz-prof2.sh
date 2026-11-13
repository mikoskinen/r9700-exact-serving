#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzz-prof2.sh (queue item, STAGED 12.11.2026 by the int11 builder for ds-main-port / the coordinator --
# NOT QUEUED by the builder; sorts after -r92zzzzzz-ref12zzz-prof1 in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~15 min estimate, budget XP2_BUDGET_MIN 30 (a section past it skipped, logged), NO evict) -- PROF START 2
# (PROFILED: --profile-ops + RADIANCE_PROFILE_EVERY 8, every op synchronised: NOT A SPEED) on int12: ref12's start and pins exactly
# (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH / XPREFETCH 0),
# MAXSEQS 16 (= prof1; ref12 ran at 8). Tag "RESULT xp2 ", out runs/xdsm/prof12/p2, body queue-ar/dsmain/prof12/dsm-prof2.sh (a COPY of
# prof12/dsm-prof1.sh + i9p's profiled-start pattern) + prof12/p12win.py (nc/ds_opprof_nc.py imported read only). The COPY PATH.
# WINDOWS: c16 (CAP48 method: cap48/ncmb.py phase c16, T 0, dur 30 warm 10), c8 (harness/ncmb3.py fixed prompts, 35 s), c4 (ncmb3,
# 20 s), 1u switch (code 512 for residency, then essay 512 = the window), 1u warm (essay right after essay). Per window and rank: ms a
# decode step by class (decode-only dump intervals), the newcomer steps apart (estimate), the row mix / depth-0 share, stream MB
# (1u), the c8 / c16 wall split; ONE table "ms per added row by class" (P12FIT, warm / c8 / c4 / c16) with the intercept, rank 0 and
# the rank spread, + within-window fits.
# XP2_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzz-prof2.sh : prints every command;
# checks the image literals (int9 .. int12, the plugin md5), every helper (+ selftests), the sidecars, the yaml, the disk.
export XP2_IMAGE=${XP2_IMAGE:-radiance-local:ds41-int12-6f3e70df-579d0bc2-l} XP2_ITEM=${XP2_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzz-prof2}
export XP2_OUT=${XP2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/prof12/p2} XP2_SW=${XP2_SW:-1}
export XP2_MX=${XP2_MX:-0} XP2_F4=${XP2_F4:-0} XP2_F4G=${XP2_F4G:-0} XP2_MXCONFIRM=${XP2_MXCONFIRM:-0} XP2_GSCONC=${XP2_GSCONC:-16} XP2_MAXSEQS=${XP2_MAXSEQS:-16}
export XP2_EGEN=${XP2_EGEN:-512} XP2_PEVERY=${XP2_PEVERY:-8} XP2_C4=${XP2_C4:-1} XP2_PLUGMD5=${XP2_PLUGMD5:-9823cdde} XP2_C48REF=${XP2_C48REF:-223.2}
export XP2_C8WIN=${XP2_C8WIN:-35} XP2_C8WARM=${XP2_C8WARM:-10} XP2_C4WIN=${XP2_C4WIN:-20} XP2_C4WARM=${XP2_C4WARM:-8} XP2_C16DUR=${XP2_C16DUR:-30} XP2_C16WARM=${XP2_C16WARM:-10}
export XP2_DISKMIN_MB=${XP2_DISKMIN_MB:-11000} XP2_BUDGET_MIN=${XP2_BUDGET_MIN:-30}
export XP2_SIDECAR=${XP2_SIDECAR:-dspark-dlw-q8_0.rad} XP2_SIDESHA=${XP2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/prof12/dsm-prof2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xp2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
