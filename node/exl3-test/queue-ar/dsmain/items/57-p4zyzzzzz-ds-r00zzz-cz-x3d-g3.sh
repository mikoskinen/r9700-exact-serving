#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g3.sh (queue item, STAGED 12.11.2026 by the X3D-F builder for ds-main-port; sorts after
# -cz-x3d-g12b in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN (or the one ATTACH claims);
# ~24 min estimate, budget X3DS_BUDGET_MIN 28 (a reading past it skipped and logged), NO evict) -- X3D-F GATE 3 (the engine port):
# exl3_x3d 0 vs 1 in ONE start, tag "RESULT xx3d ", out runs/xdsm/x3d, body queue-ar/dsmain/x3d/dsm-x3d.sh (from a COPY of
# ngreuse/dsm-ngreuse.sh 92204f57 + arc's arm pieces; ATTACH-patched by attach/patch-body.sh). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int10x-04882136-579d0bc2-l (f5d557c1b16d; = the int10 image 0a60fffd3b3c + a layer of bin/radiance, r4d_selftest,
# rad-sidecar, libr4d / libavx / libref, deepseek41.so from ds41-x3d-eng10 0488213 (fn4 exl3_x3d 57, a plugin switch; kSwitches 58)
# and kernels/exl3r4d.so = the ds41-x3d 579d0bc plugin (md5 9823cdde: g12b's bytes); exl3_x3d 0 at load = the int10 image).
# ONE START = the ngreuse / spread base (stack5's + RADIANCE_DS41_MHC_FUSE=0, RADIANCE_LOG_STEPS=1, RADIANCE_DS41_ENGDD=0, HB ranks
# 1,2; KVR / SP off; sched_row_cap 48 by fn4) on the ngshare path (tools/ngramshare/ngshare.sh: fill or reuse). BASE PIN = S5T,
# draft_sample 0, SP 0, mhc_fuse 1, IX off, draft_depth0 3, sched_row_cap 48, ds41_arcoded 0. Arms in process (fn4, read back before
# every reading): B exl3_x3d 0, C exl3_x3d 1. Readings (TESTING.md 12.11. block rules): identity B; c4 warm-up (discarded); c8 B, C
# (fixed schedule, ~35 s time window, one a arm); c4 C, B (one a arm); c1 B C C B; 1u essay512 B C C B; cold 64k B, C (TTFT +
# the answers byte for byte); identity C (greedy set C vs B strict). Summary: C vs B % a reading, identity, INFO B vs int10's own
# numbers (ngreuse / spread), "VERDICT x3d ...". Guards: exl3_x3d served (0 at start, 2 / -1 refused), switch read back, 0 request
# errors, identity (a difference = VIRHE). The n-gram tables stay in /dev/shm/ds41ng (X3DS_NGCLEAN=1 cleans them).
# X3DS_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 .. int10x switch posted, a "VIRHE X3DS_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g3.sh : prints every command; checks the image literals (int9 .. int10x) and the
# plugin's sha256, every helper (+ selftests), the ngramshare files and both yamls, the sidecars, the disk; no server, no GPU,
# writes nothing.
export X3DS_IMAGE=${X3DS_IMAGE:-radiance-local:ds41-int10x-04882136-579d0bc2-l} X3DS_ITEM=${X3DS_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x3d-g3}
export X3DS_OUT=${X3DS_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/x3d} X3DS_SW=${X3DS_SW:-1}
export X3DS_C8DUR=${X3DS_C8DUR:-45} X3DS_C8WARM=${X3DS_C8WARM:-10} X3DS_DUR=${X3DS_DUR:-30} X3DS_WARM=${X3DS_WARM:-10}
export X3DS_GENS1=${X3DS_GENS1:-6} X3DS_GENS4=${X3DS_GENS4:-2} X3DS_EGEN=${X3DS_EGEN:-512} X3DS_DEPTH=${X3DS_DEPTH:-64000} X3DS_NTOK=${X3DS_NTOK:-64}
export X3DS_MHC=${X3DS_MHC:-1} X3DS_D0=${X3DS_D0:-3} X3DS_CAP=${X3DS_CAP:-48} X3DS_SPN=${X3DS_SPN:-0} X3DS_SPK=${X3DS_SPK:-0} X3DS_NGCLEAN=${X3DS_NGCLEAN:-0}
export X3DS_DISKMIN_MB=${X3DS_DISKMIN_MB:-11000} X3DS_BUDGET_MIN=${X3DS_BUDGET_MIN:-28}
export X3DS_PLUGSHA=${X3DS_PLUGSHA:-4245da82106e56c1a1e9b7f58221b2cf9af1b79c8e5eefc2628d5ba0c052ad5a}
export X3DS_SIDECAR=${X3DS_SIDECAR:-dspark-dlw-q8_0.rad} X3DS_SIDESHA=${X3DS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/x3d/dsm-x3d.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xx3d: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
# ATTACH-SIG: SIGONLY=1 DRYRUN=1 bash <this item> prints its start signature (queue-ar/dsmain/attach/attach.sh)
export ATTACH=${ATTACH:-1} KEEP=${KEEP:-1}
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
export X3DS_NGCLEAN=1   # 12.11. ds-main-port: clean the n-gram tables at the end unless the next DS ngshare item is queued right behind (FN items follow)
bash $B
