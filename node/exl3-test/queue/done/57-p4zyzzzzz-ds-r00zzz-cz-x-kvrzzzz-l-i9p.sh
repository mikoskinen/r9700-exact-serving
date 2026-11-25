#!/bin/bash
# RE-QUEUE 12.11. ~12:15 (ds-main-port, coordinator): ONLY the profiled 1u NST 5 essay round of i9b (its pstart failed 12:12:30 on the ghost compose container); body i9b/dsm-i9p.sh (copy of dsm-i9b.sh without dstart / summary); out runs/xdsm/i9p; + compose guard; ~7.5 min
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b.sh (queue item, STAGED 12.11.2026 by the ds-main-port int9b builder for the coordinator,
# RESHAPED 12.11. per the coordinator: ONE SCREEN START + one short PROFILED start; sorts after 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-kvd
# and before 57-p4zyzzzzz-ds-r00zzz-cz-xz-fnsmoke2 / 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~30 min estimate (screen budget XI9_BUDGET_MIN 25, total XI9_TOTAL_MIN 36: a late section
# is skipped, logged), NO evict) -- LEVERS M1 (one-pass mHC: pcn_a + mix_a as one op), IX-LDSR2 (index_score keys in LDS, two rows a
# wave) and IX-NI (no -inf dead-region writes; index_candidates reads below reach), tag "RESULT xi9b ", out runs/xdsm/i9b, body
# queue-ar/dsmain/i9b/dsm-i9p.sh (from a COPY of stack5/dsm-stack5.sh 63649414; helper i9b/i9bctr.py = a copy of stack5/s5ctr.py;
# originals unchanged). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int9b-f119c561-5c13e555-l (v2: ds41-int9b f119c56 = v1 6bee6f7b + r4d_selftest's IX cases on stress data;
# ds41-int9 0cb6fd3 + the three levers behind fn4 mhc_fuse / ix_lds / ix_ni after stage_full_pick, GET /admin/fn4 "i9b";
# RADIANCE_DS41_MHC_FUSE declares M1; every new switch 0 and RADIANCE_DS41_MHC_FUSE unset = the int9 image).
# P0 (CARD 0, no server, before the start): r4d_selftest --case mhc_pcn_mix / ix_lds / ix_ni; a failure stops the item.
# SCREEN START = stack5's start + RADIANCE_DS41_MHC_FUSE=0 + RADIANCE_DS41_KV_REPLICATED=1 (read back: env, the KVR "5.39 KiB a slot"
# line, the M1 declare line). BASE = stack5's S5T with draft_sample 0 + SP pinned 0 / 0 (SP off: its item gave no SP value) -- "draft_sample
# 0 + KVR=1 base per the coordinator". Each lever on its target only, ABAB, 2 blocks a side: M / L / N cold 64k TTFT (B X B X a lever,
# 64 tokens, a reset before each; NOT 136k); N also mb c8 T 0 on the c8 block-mix harness (ncmb2.py / mbt2.py fixed schedule 1x8
# seed 1, gens 2 = 16 newcomer steps a block, ~85 s) B N N B: c8mix.py per block (mix + newcomer step), the B-vs-B mix check (blocks 1 / 4):
# the N-vs-B delta only when they MATCH, else INCONCLUSIVE (mix); a NEW c8 reference (not comparable with old-loop c8); ONE 1u essay T 1
# block under the base (the cost of draft_sample 0, cross-start vs SD 85); exactness: the greedy set alone, all three ON vs OFF, text +
# token ids. Guards: switch read back + 0 request errors only (counters INFO). Output: a VERDICT line a lever (mean delta, block
# spreads, beyond / within spread), the c8 line, the T1 line, the identity line, a summary.
# PROFILED START (at the end): the same start + base + --profile-ops (RADIANCE_PROFILE_EVERY 8), ONE 1u NST 5 essay round -> the
# drafting round's per-op ms split (drafter / verify trunk / MoE / attention / mhc / collectives / lane waits / host gaps), ranked
# (nc/ds_opprof_nc.py, read only). Not a speed, not comparable with the screen.
# XI9_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 / int9b switch posted, P0 not fatal, a "VIRHE XI9_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b.sh : prints every command; checks the image literals (int9 + KVR + profiler +
# int9b), every helper (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XI9_IMAGE=${XI9_IMAGE:-radiance-local:ds41-int9b-f119c561-5c13e555-l} XI9_ITEM=${XI9_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-l-i9p}
export XI9_OUT=${XI9_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/i9p} XI9_SW=${XI9_SW:-1}
export XI9_LEVERS=${XI9_LEVERS:-MLN} XI9_SDEPTH=${XI9_SDEPTH:-64000} XI9_SNTOK=${XI9_SNTOK:-64} XI9_SORDER=${XI9_SORDER:-BXBX}
export XI9_C8ORDER=${XI9_C8ORDER:-BNNB} XI9_C8DUR=${XI9_C8DUR:-30} XI9_C8WARM=${XI9_C8WARM:-10} XI9_C8GENS=${XI9_C8GENS:-2} XI9_T1=${XI9_T1:-1} XI9_IDN=${XI9_IDN:-1}
export XI9_PROF=${XI9_PROF:-1} XI9_PEVERY=${XI9_PEVERY:-8} XI9_PGEN=${XI9_PGEN:-512} XI9_PSKIP=${XI9_PSKIP:-2}
export XI9_MHC=${XI9_MHC:-1} XI9_SPN=${XI9_SPN:-0} XI9_SPK=${XI9_SPK:-0}
export XI9_P0MIN="${XI9_P0MIN:-mhc_pcn_mix:7:2 ix_lds:6:1 ix_ni:6:2}"
export XI9_DISKMIN_MB=${XI9_DISKMIN_MB:-11000} XI9_BUDGET_MIN=${XI9_BUDGET_MIN:-25} XI9_TOTAL_MIN=${XI9_TOTAL_MIN:-12}
export XI9_SIDECAR=${XI9_SIDECAR:-dspark-dlw-q8_0.rad} XI9_SIDESHA=${XI9_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/i9b/dsm-i9p.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xi9b: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
