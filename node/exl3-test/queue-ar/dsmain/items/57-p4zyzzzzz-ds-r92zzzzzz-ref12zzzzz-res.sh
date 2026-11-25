#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzz-res.sh (queue item, STAGED 12.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzz-prof2 in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~34 min estimate, budget XRS_BUDGET_MIN 40 (a start or a read set past it skipped, logged), NO evict) --
# RESIDENCY SCREEN on int12 (zero-build): placement / heat knobs only (exact by construction) against ref12's topic-switch cost (a 1u
# essay right after a code read streams ~136 MB a step and reads ~82 tok/s; essay after essay 27 MB / 90.46; starved_slab +521 vs
# +92). Four FRESH starts on n-gram REUSE (tools/ngramshare/ngshare.sh: the first start fills the tables, the rest reuse them; the
# tables are CLEANED at the end with ngram-shm-cleanup.sh, FN items follow): BASE (ref12's start; its read set runs twice, fn4
# place_lf 0 then 1 = lever LF in-process), GAIN2 (RADIANCE_HEAT_PREFILL_GAIN=2.0), GAIN05 (=0.5), SHADOW32
# (RADIANCE_PLACE_SHADOW_SLOTS=32, VA's 16 undone). ref12's pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4,
# exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH / XPREFETCH 0), MAXSEQS 16 (= prof1). Tag "RESULT xres ", out runs/xdsm/res12,
# body queue-ar/dsmain/res12/dsm-res12.sh (a COPY of prof12/dsm-prof1.sh f64daf3a) + res12/resprobe.py (stepprobe + the answer) +
# res12/ressum.py (summary + identity). A READ SET: resprobe 512 T 0 code W, essay W, code, essay, code, essay (SWITCHED), essay,
# essay (WARM) -- per read stream / h2d MB a step, promotions, readmits, starved_slab / starved_pool deltas, ms a step -- then ONE c8
# block (harness/ncmb3.py --prompt-set fixed --reset-per-block 1, 35 s) = the c8 GUARD. BASE-lf0 only: + ONE c16 block on the same
# fixed harness (1x16, 35 s, MAXSEQS 16, cap 48) + its P12SPLIT wall split (prof12/p12win.py): agg, decode_share, newcomer-step ms
# a step and share, > 48-row steps (must be 0). SUMMARY per arm vs BASE-lf0: switched essay
# ms a step / tok/s + stream MB, warm essay ms a step, c8 agg + ms a step, "helps the switch by X % at c8 Y %". GUARDS: every switch /
# start word read back (env + the engine's heat / mover lines + place_lf on every rank), 0 request errors, greedy identity of every
# read's answer across arms / sets, the compose guard. DISK (TESTING.md 12.11.): guard >= 11000 MB free before every start; one gz
# engine log a start; at the end the start logs / c8 samples gzipped and the /stats + fn4 snapshots bundled (snapshots.tar.gz).
# HOLD (coordinator 12.11.): staged only, not in queue/ until the coordinator says.
# XRS_SW=0 = a mock run (queue-ar/tools/mockrad; res12/mockrun-res12.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzz-res.sh :
# prints every command; checks the image literals (int9 .. int12, the plugin md5), every helper (+ selftests), the ngramshare files,
# the sidecars, both yamls, the disk.
export XRS_IMAGE=${XRS_IMAGE:-radiance-local:ds41-int12-6f3e70df-579d0bc2-l} XRS_ITEM=${XRS_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzz-res}
export XRS_OUT=${XRS_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/res12} XRS_SW=${XRS_SW:-1}
export XRS_MX=${XRS_MX:-0} XRS_F4=${XRS_F4:-0} XRS_F4G=${XRS_F4G:-0} XRS_MXCONFIRM=${XRS_MXCONFIRM:-0} XRS_MAXSEQS=${XRS_MAXSEQS:-16}
export XRS_ARMS="${XRS_ARMS-BASE GAIN2 GAIN05 SHADOW32}" XRS_GAIN2=${XRS_GAIN2:-2.0} XRS_GAIN05=${XRS_GAIN05:-0.5} XRS_SHADOW=${XRS_SHADOW:-32} XRS_BASELF=${XRS_BASELF:-01}
export XRS_EGEN=${XRS_EGEN:-512} XRS_PLUGMD5=${XRS_PLUGMD5:-9823cdde} XRS_C8WIN=${XRS_C8WIN:-35} XRS_C8WARM=${XRS_C8WARM:-10}
export XRS_DISKMIN_MB=${XRS_DISKMIN_MB:-11000} XRS_BUDGET_MIN=${XRS_BUDGET_MIN:-40} XRS_START_MIN=${XRS_START_MIN:-6} XRS_SET_MIN=${XRS_SET_MIN:-3}
export XRS_SIDECAR=${XRS_SIDECAR:-dspark-dlw-q8_0.rad} XRS_SIDESHA=${XRS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/res12/dsm-res12.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xres: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
