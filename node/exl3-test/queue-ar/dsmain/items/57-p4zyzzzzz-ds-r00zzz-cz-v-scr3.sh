#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-v-scr3.sh (queue item, STAGED 12.11.2026, ds-main-port pskcr2 builder for the coordinator; NOT queued:
# the coordinator queues it. Sorts after 57-p4zyzzzzz-ds-r00zzz-cz-uzzz-usahb and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in
# LANG=en_US.UTF-8 and LC_ALL=C order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~27 min (estimate,
# budget <= ~35), NO evict) -- LEVER PSK-CR's fix ds41-pskcr2 (d4540a8 on ds41-pskcr e9e529d), tag "RESULT xcr ", out
# runs/xdsm/pskcr3, body queue-ar/dsmain/pskcr/dsm-pskcr3.sh (+ pcr3.py; pskld.py = the pskl20 copy). = r00zzz-cz-tz-scr2's
# body dsm-pskcr2.sh f2b81d07 + pcr2.py a7c03dc3 copied (neither edited).
# WHY (r00zzz-cz-tz-scr2): a request served by a cone replay kept no snapshot (Scheduler::psk_ckpt dropped the replay's end H, and a
# replayed prompt's tail snapshot is H), so every next turn of the conversation replayed again (HIT turn:40960 B 1.21 s, C / T
# 2.64 / 2.67 s); hit 24576 C +0.76 s (the same cause: 20480 replayed to 18432 without a snapshot) and hit 4096 C +0.32 s (a
# replay of the first chunk is slower than its cold chunk; a grid point one step over a snapshot: two replay passes vs one chunk).
# DESIGN: image ds41-pskcr2 (psk_ckpt keeps the snapshot at a replay's end; psk_cr_resume replays only more than one grid step over
# the snapshot the hit has, else that snapshot / cold; /stats psk cr_cheaper, cr_end_snapshots). ONE NST 5 START = stack4's +
# RADIANCE_LOG_STEPS=1; arms in process B (ds41_psk 2 k 4, cr 0) / C (2 k 4, cr 1) / T (4 tail-only, cr 1); XCR_L20 / XCR_PF 0.
# Every arm on its own text with a nonce line.
# Readings: COUNTER GUARD (baseline after each arm switch; C / T replay + take the end snapshot, B none); HIT CASES 40960 / 20480 /
# 24576 / 4096 (scr2's round-1 order) per arm vs B's cold refs; TURN CHAINS (chR: cold 64k, turn 1 a 43500 hit -- C / T replay
# once --, turns 2-4 previous + answer + new message; chA: turn 1 cold 48k, turns 2-4): per turn cached / replays (C / T: 0
# after turn 1, else VIRHE) / snapshots / TTFT, identity vs B's cold ref of the same full text; COLD TTFT T B B T at 135000.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-v-scr3.sh : prints every command; checks the image literals, every helper (+ selftests,
# the text offsets), the sidecars, the yaml; no server, no GPU, writes nothing.
export XCR_IMAGE=${XCR_IMAGE:-radiance-local:ds41-pskcr2-d4540a8e-5c13e555-l} XCR_ITEM=${XCR_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-v-scr3}
export XCR_OUT=${XCR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/pskcr3}
# L20 / PF 0 = PSK-CR measured alone (as scr2): a failure is attributable.
export XCR_L20=${XCR_L20:-0} XCR_PF=${XCR_PF:-0}
export XCR_PSKK=${XCR_PSKK:-4} XCR_OCC=${XCR_OCC:-6} XCR_OCCK=${XCR_OCCK:-32} XCR_DSA=${XCR_DSA:-85}
export XCR_HDEP=${XCR_HDEP:-64000} XCR_HITS=${XCR_HITS:-40960 20480 24576 4096} XCR_CRD1=${XCR_CRD1:-43500} XCR_CAD=${XCR_CAD:-48000}
export XCR_NTURN=${XCR_NTURN:-4} XCR_DEPTHS=${XCR_DEPTHS:-135000} XCR_TORDER=${XCR_TORDER:-TBBT} XCR_DISKMIN_MB=${XCR_DISKMIN_MB:-11000}
export XCR_SIDECAR=${XCR_SIDECAR:-dspark-dlw-q8_0.rad} XCR_SIDESHA=${XCR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/pskcr/dsm-pskcr3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xcr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
