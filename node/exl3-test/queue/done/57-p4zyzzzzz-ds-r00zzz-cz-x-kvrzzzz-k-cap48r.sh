#!/bin/bash
# RE-QUEUE 12.11. ~12:03 as -kvrzzzz-k-cap48r (after i9b, d0ab): 11:59:48 run failed in 5 s on the ghost compose container 622b5b7a9e64 (Recreate path); + compose guard
# RE-RUN 12.11. ~11:52 as -cg-cap48r (ds-main-port): the 11:19 run hung in its start on the lost card 83:00 (906 s); same item c3f401b4, new name sorts after cf-arw2 (user order d0ab -> arw2 -> CAP48 -> i9b); old output in runs/xdsm/cap48/run1-hung-1119/
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48.sh (queue item, STAGED 12.11.2026 by the ds-main-port cap48 builder for the coordinator;
# sorts after 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc and before 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b / -cz-xz-fnsmoke2 / -d-slotsum in
# C and en_US order; gpu02, all 4 cards, TP4, every radiance container DOWN; ~29 min (estimate; budget XC48_BUDGET_MIN 35), NO evict)
# -- LEVER CAP48: a scheduler cap of 48 decode rows a step (drafts trimmed, the deepest first; prefill rows not counted), so every
# decode piece stays in the <= 48-row kernel forms whose rows are batch-invariant (kv4 n16: 9 / 20 and 8 / 20 answers changed under
# load above 48 decode rows). Tag "RESULT xc48 ", out runs/xdsm/cap48, body queue-ar/dsmain/cap48/dsm-cap48.sh (from COPIES of
# kv4/dsm-kv4.sh f89a6cc2 and stack5/dsm-stack5.sh 63649414) + cap48probe.py (copy of stack4/loadid.py ee7d9663 + T / seed / set)
# + cap48sum.py + ncmb.py / ncone.py (copies of kv4's); originals unchanged. NOT QUEUED: the coordinator queues it only if the
# BI16 trace (57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16) confirms the > 48-row cause.
# IMAGE radiance-local:ds41-cap48-e5acfbcf-5c13e555-l (05acc2fd8f3c; = the int9 image radiance-local:ds41-int9-0cb6fd31-5c13e555-l + bin/radiance
# of ds41-cap48 e5acfbcf: fn4 sched_row_cap (52, after stage_full_pick; kSwitches 53) / RADIANCE_DECODE_ROW_CAP, GET "cap" counters;
# unset = int9: graph and start log identical, build/cap48 hostcheck).
# ONE START c16 = kv4's n16 (nvfp4 latent, MAXSEQS 16, HOSTPOOL 19456, MML 524288, NST 5, NVG 48) with stack5's start words +
# RADIANCE_DECODE_ROW_CAP=48; the S5T pin; in process: S = S5T + cap 48 (mb c16 / c8), D = S5T + DET (fixed depth 5) + cap 48
# (T 0 identity), Z = D + draft_sample 0 (T 1 seeded identity), d / z = D / Z with cap 0 (the in-process OFF control).
# READINGS: (a) identity alone vs loaded (gs16 = conc 16; lid16 = each prompt under 14 mb streams + a fresh ~1100 newcomer) at T 0
# and T 1 (seed 1000 + i), byte for byte, cap on (strict) and cap off (the control's count); (b) a window's steps (step log n_tok)
# + the cap counters (steps_capped, drafts_trimmed, steps_gt48 = 0 under the cap) + BI stats; (c) mb c16 / c8 at T 0, cross-start vs
# kv4 n16 c16 230.5 tok/s and stack5 S5T c8 173.6 (labelled). Guards: pins read back, the cap counters move only where it acts,
# PSK on the corrected counters (cold 16k), fresh newcomers (cached 0), disk >= 11000 MB.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48.sh : prints every command; checks the image literals, every helper
# (+ selftests), the references, the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
# 12.11. SCREEN MODE (coordinator / user: screen first, confirm only winners): identity T0 alone vs loaded at conc 16 (cap on) + the
#   cap-off control count, the n_tok > 48 step count from the step log, mb c16 cap on (D) vs off (d) ABAB at dur 30 warm 10;
#   T1 and the lid16 / cap-0 T1 controls move to the confirmation start. c8 omitted: at 8 sequences x (1 + 5) = 48 rows the cap
#   never trims (cost 0 by construction).
export XC48_PHASES=${XC48_PHASES:-"a1:D:alone0 gs16:D:gs0 gs16o:d:gs0 c16a:D:mb16 c16b:d:mb16 c16c:D:mb16 c16d:d:mb16"}
export XC48_DUR=${XC48_DUR:-30} XC48_WARM=${XC48_WARM:-10} XC48_BUDGET_MIN=${XC48_BUDGET_MIN:-16}
export XC48_IMAGE=${XC48_IMAGE:-radiance-local:ds41-cap48-e5acfbcf-5c13e555-l} XC48_ITEM=${XC48_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48}
export XC48_OUT=${XC48_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/cap48}
export XC48_CAP=${XC48_CAP:-48} XC48_CAPSW=${XC48_CAPSW:-1} XC48_INFO85=${XC48_INFO85:-0} XC48_BIC=${XC48_BIC:-3} XC48_NVG=${XC48_NVG:-48} XC48_HP=${XC48_HP:-19456} XC48_MS=${XC48_MS:-16}
export XC48_LIDC=${XC48_LIDC:-14} XC48_LIDPF=${XC48_LIDPF:-1100} XC48_SEEDB=${XC48_SEEDB:-1000} XC48_MT=${XC48_MT:-192}
export XC48_DUR=${XC48_DUR:-60} XC48_WARM=${XC48_WARM:-15} XC48_GDEP=${XC48_GDEP:-16000} XC48_DISKMIN_MB=${XC48_DISKMIN_MB:-11000} XC48_BUDGET_MIN=${XC48_BUDGET_MIN:-35}
export XC48_SIDECAR=${XC48_SIDECAR:-dspark-dlw-q8_0.rad} XC48_SIDESHA=${XC48_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
[ -n "${XC48_PHASES-}" ] && export XC48_PHASES || unset XC48_PHASES   # empty = the body's default (+ the info85 pair under XC48_INFO85=1)
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/cap48/dsm-cap48.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xc48: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (ghost 622b5b7a9e64)
bash $B
