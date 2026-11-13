#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-scr.sh (queue item, STAGED 12.11.2026, ds-main-port PSK-CR builder for the coordinator; NOT queued: the
# coordinator queues it after a second reading. Sorts after 57-p4zyzzzzz-ds-r00zzz-cz-reng and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum
# in LANG=en_US.UTF-8 and LC_ALL=C order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~40 min (estimate,
# budget <= 60), NO evict) -- LEVER PSK-CR (ARCH-AUDIT-20261112 lever 3: exact cone replay on a prefix hit), tag "RESULT xcr ", out
# runs/xdsm/pskcr, body queue-ar/dsmain/pskcr/dsm-pskcr.sh (+ pcr.py; pskld.py = pskl20's copy).
# WHY: under design 2 a prefix hit with no snapshot at the grid point under it falls back to the anchor below and recomputes up
# to K chunks (measured K4 worst +3.04 s, K8 mean +4.11 s); the shared head's 2nd and 3rd prompt fall back too (branch_at). With
# ds41_psk_cr 1 the hit resumes AT that grid point: the cached global KV (latent + index keys, every row's) is adopted, the
# sequence starts at the grid point under it less 5152 rows with no state, and its grid chunks up to it run EVERY layer only over
# its cone (layer l: the last 160 + (39 - l) x 128 rows) with no global-KV store -- the window state there is a cold prefill's.
# Estimate ~0.7-1.0 s a hit; it also allows tail-only anchors (ds41_psk 4: cold = K8's 31.4 s or better).
# DESIGN: image ds41-pskcr (= ds41-pskl20 25aab56 + e9e529d: fn4 ds41_psk_cr appended (45 switches), ds41_psk accepts 4, /stats psk
# cr_* + tail_only_*; default 0 = the pskl20 image). ONE NST 5 START = stack4's (its start line runs/ds41-exl3.log 2026-11-11
# 21:51:34) + RADIANCE_LOG_STEPS=1. Arms in process on stack4's fixed pin + S's pins: B = ds41_psk 2 k 4 cr 0, C = ds41_psk 2 k 4
# cr 1, T = ds41_psk 4 (tail-only) cr 1; + ds41_psk_l20 XCR_L20 and dspark_pfctx XCR_PF in EVERY arm (default 1).
#   >>> XCR_L20 / XCR_PF: THE COORDINATOR SETS THEM FROM 57-p4zyzzzzz-ds-r00zzz-cz-pskl20's RESULT BEFORE THIS RUN (0 where that
#   >>> item's lever did not pass: identity or its counter guard).
# Readings: COUNTER GUARD (C / T move the cr counters, B none; pins read back; a failure = VIRHE, no speed reading); HIT IDENTITY
# vs B's cold prefill of the same full text (each arm its own text: mid-prompt hits at grid points with no anchor, the
# first-chunk replay, the tail, a follow-up turn, the shared head's 2nd / 3rd / 4th prompt), alone and under c4 load; HIT TTFT
# (C / T vs B's fallback, two rounds); COLD TTFT 136k / 64k ABBA (T vs B).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-scr.sh : prints every command; checks the image literals, every helper (+ selftests,
# the text offsets), the sidecars, the yaml; no server, no GPU, writes nothing.
export XCR_IMAGE=${XCR_IMAGE:-radiance-local:ds41-pskcr-e9e529d2-5c13e555-l} XCR_ITEM=${XCR_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-scr}
export XCR_OUT=${XCR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/pskcr}
# 12.11. ds-main-port: L20 / PF default 0 = PSK-CR measured ALONE on the pskl20 image (exactness first: a failure is attributable; r00zzz-cz-pskl20 runs before this item but its result is not read before queueing); the L20 + PF + CR + E1 stack goes to the next stack confirmation.
export XCR_L20=${XCR_L20:-0} XCR_PF=${XCR_PF:-0}
export XCR_PSKK=${XCR_PSKK:-4} XCR_OCC=${XCR_OCC:-6} XCR_OCCK=${XCR_OCCK:-32} XCR_DSA=${XCR_DSA:-85}
export XCR_HDEP=${XCR_HDEP:-64000} XCR_HITS=${XCR_HITS:-40960 20480 24576 4096} XCR_LHIT=${XCR_LHIT:-45056}
export XCR_HEAD=${XCR_HEAD:-24576} XCR_HTAIL=${XCR_HTAIL:-3000} XCR_DEPTHS=${XCR_DEPTHS:-135000 64000} XCR_TORDER=${XCR_TORDER:-TBBT}
export XCR_ROUND2=${XCR_ROUND2:-1} XCR_DISKMIN_MB=${XCR_DISKMIN_MB:-11000}
export XCR_SIDECAR=${XCR_SIDECAR:-dspark-dlw-q8_0.rad} XCR_SIDESHA=${XCR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/pskcr/dsm-pskcr.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xcr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
