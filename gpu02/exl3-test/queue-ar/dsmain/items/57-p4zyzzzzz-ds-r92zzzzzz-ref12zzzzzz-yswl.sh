#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl.sh (queue item, STAGED 12.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-xpush (and -yll, -ypfx) in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~20 min estimate, budget XSW_BUDGET_MIN 22 (a section past it skipped, logged: L1G100
# first, then the staggered L1 check), NO evict) -- THE SWL / L2 SCREEN on int12 + ds41-swl (radiance-local:ds41-swl-29c003e8-579d0bc2-l
# = int12 bf8f557 + fn4 place_swap_lent 63 / heat_min_gain_x100 64 / heat_hyst_x100 65, bin/radiance only; unset = int12's graph and
# start log). ONE FRESH start (ref12's pins exactly incl. RADIANCE_HEAT_MOVES=16, RADIANCE_PLACE_TURNOVER=0, shadow 16, KVR=1; MAXSEQS
# 16; copy path + n-gram reuse (ngshare.sh, the tables cleaned at the end), no attach), the arms IN-PROCESS by fn4, read back in
# GET /admin/fn4 and on every rank (/stats experts.ranks[].place_swl / heat_min_gain_x100):
#   B (int12: place_swap_lent 0, min_gain 150) | L1 (SWL: place_swap_lent 1) | L1G075 (+ heat_min_gain_x100 75) | L1G100 (100, budget).
# PLAN: STAGGERED identity under B (a1B = the greedy set c1 alone, T 0, 192 tok; stack4/loadid.py run: c4 streams + the 20 probes, each
# with a fresh 8k prefill joining it -> ldB vs a1B STRICT); then per arm its c8 block FIRST (harness/ncmb3.py fixed prompts, reset per
# block, 35 s, + P12SPLIT) and code, essay, code, essay (2 SWITCHED essays) + essay (1 WARM essay), resprobe 512 T 0, with stream / h2d
# MB a step, starved_slab, direct promotions, lent fills, SWL lent promotions, misses (rank 0) a read; the c8 blocks run B L1 L1 B
# (ABBA); then L1G075; STAGGERED under L1 (budget: a1L1 + ldL1 vs a1L1, a1L1 vs a1B, strict); L1G100 (budget, last).
# SUMMARY (yswl/swlsum.py, "RESULT xswl SWL ..."): an ARM line each, C8PAIR, VERDICT each arm vs the in-process B ("switch X %, warm W %,
# c8 Y %"), INFO vs res12 BASE-lf0 (19.30 ms / 81.66 switched, 17.41 / 90.58 warm, c8 183.1; NOT re-run), IDENTITY: every read's
# answer byte-compared across arms (a difference = VIRHE); the staggered checks' lines (a difference = VIRHE, the item goes on).
# Body queue-ar/dsmain/yswl/dsm-yswl.sh (a COPY of res12/dsm-res12.sh 08873cb8) + yswl/swlsum.py + yswl/resprobe.py (= res12's).
# GUARDS: every switch / start word read back, 0 request errors, the compose guard; DISK guard >= 11000 MB; logs gzipped at the end.
# XSW_SW=0 = a mock run (queue-ar/tools/mockrad; yswl/mockrun-yswl.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl.sh :
# prints every command; checks the image literals (int9 .. int12 + ds41-swl, the plugin md5), every helper (+ selftests), the ngramshare
# files, the sidecars, both yamls, the disk.
export XSW_IMAGE=${XSW_IMAGE:-radiance-local:ds41-swl-29c003e8-579d0bc2-l} XSW_ITEM=${XSW_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl}
export XSW_OUT=${XSW_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/yswl} XSW_SW=${XSW_SW:-1}
export XSW_MX=${XSW_MX:-0} XSW_F4=${XSW_F4:-0} XSW_F4G=${XSW_F4G:-0} XSW_MXCONFIRM=${XSW_MXCONFIRM:-0} XSW_MAXSEQS=${XSW_MAXSEQS:-16}
export XSW_ARMS="${XSW_ARMS-BASE}" XSW_C16=${XSW_C16:-0} XSW_STAG=${XSW_STAG:-2} XSW_ABBA=${XSW_ABBA:-1} XSW_G100=${XSW_G100:-1}
export XSW_LDPF=${XSW_LDPF:-8192} XSW_LDCONC=${XSW_LDCONC:-4}
export XSW_EGEN=${XSW_EGEN:-512} XSW_PLUGMD5=${XSW_PLUGMD5:-9823cdde} XSW_C8WIN=${XSW_C8WIN:-35} XSW_C8WARM=${XSW_C8WARM:-10}
export XSW_DISKMIN_MB=${XSW_DISKMIN_MB:-11000} XSW_BUDGET_MIN=${XSW_BUDGET_MIN:-22} XSW_START_MIN=${XSW_START_MIN:-6} XSW_SET_MIN=${XSW_SET_MIN:-2}
export XSW_C8_MIN=${XSW_C8_MIN:-1} XSW_STAG_MIN=${XSW_STAG_MIN:-3}
export XSW_SIDECAR=${XSW_SIDECAR:-dspark-dlw-q8_0.rad} XSW_SIDESHA=${XSW_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/yswl/dsm-yswl.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xswl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
