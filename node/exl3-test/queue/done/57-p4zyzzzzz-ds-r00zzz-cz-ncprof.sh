#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-ncprof.sh (queue item, STAGED 12.11.2026, ds-main-port NC-profile builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-c-dst and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~28 min (estimate; budget <= 30), NO evict) -- THE NEWCOMER (NC) PROFILE, tag "RESULT xnc ",
# out runs/xdsm/ncprof, body queue-ar/dsmain/nc/dsm-ncprof.sh (+ ncone.py, ncmb.py, ncsteps.py, ds_opprof_nc.py = a copy of
# ds_opprof.py with docker-time windows, step counts from the step log, dspark_ctx / dspark_query classes and the mhc / Engram /
# indexer splits). NOT QUEUED: the coordinator queues it after checking. Image radiance-local:ds41-int7-0391a6ec-5c13e555-l
# (b54f9224a056; ds41-int7 0391a6e).
# WHY (coordinator 12.11.): mb c8 reads 172 tok/s steady (stack4 T 0); decode alone ~220 tok/s a decode-second, but decode_share
# 0.784-0.791: each ~1099-token newcomer costs one non-decode step of ~0.61 s (stack4 arm A: 16 such steps for 16 requests, 9.7 s
# of 45), 1099 tokens at ~1.8k tok/s vs 3.4-4.3k tok/s for a 136k prefill; alone TTFT 0.62-0.70 s, the same. c8 >= 200 tok/s
# needs <= ~0.25 s a newcomer, or the newcomer overlapped with decode.
# QUESTIONS: (a) STAGING -- expert units / GB the ~1.1k step stages / streams, the compute lane's wait for it, distinct experts
# touched (hypothesis to test: staging-bound); (b) at c8 a MIXED step (decode rows ride) or a pure prefill step -- by the code
# (scheduler.cpp:2601-2721 decode rows planned first, :2807-2852 the newcomer admitted into the same plan, :2993 MIXED; no hold:
# interleave only after an unfinished chunk :3009-3012, span only alone >= 16384 :2538-2554, budget 2112 geometry.cpp:40, NST 5
# rows mix :2557, issue-ahead stops while a request waits :3264): MIXED, the decode rows ride, in a step as long as the prefill;
# the step log confirms; (c) ranked ms per class of the newcomer step per rank + host remainder.
# DESIGN: START U (unprofiled, stack4's NST 5 start + RADIANCE_LOG_STEPS=1, docker logs -t): arm S then A in process -- ALONE 12
# one-token ~1.1k prompts + one mb c8 window as stack4 (T 0, dur 60 warm 15, the 2 s sampler); START P (profiled every 16 +
# RADIANCE_DEBUG_MISS_LOG, arm S): ALONE 16 + 32 one-token prompts, W1 = 8 streams decode only (20 s), W2 = mb c8 (40 s),
# newcomer ms by difference (labelled ESTIMATE). Every 16, not 8 (stated in the body: every 8 over the c8 phases is ~230 MB of
# raw log). GUARD: step lines / 4-rank tables in every window, /stats counters moved, pins read back, newcomers fresh, one step
# a prompt -> else VIRHE. Disk: >= 11000 MB free before each start; engine logs gzipped only.
# SECTION (d) (coordinator 12.11., in START U after the arms, ~6.5 min): D7 under a long prefill -- arm S + draft_depth0 1 (X1,
# today) / 2 (X2, depth 0 forced), ABBA (1221): stack4's loaded probe (pskld.py --load: c4 + one cold ~64k deepq prompt, a
# prefix-cache reset before each run; the same fixed prompt every run so X2's answer compares with X1's), probe TTFT, c4 tok/s
# and per-stream p50 in the prefill window, TPOT p50 / p90, /metrics deltas (decode steps, ms / tok a decode step, drafted /
# accepted, non-decode steps + mean ms), the step log's MIXED / prefill steps and the probe's chunk sizes (off the 2048 grid?);
# guard X2 drafted ~0 + pins read back + cold; identity reported; "RESULT xnc (d) D7 under a long prefill: ..." X2 vs X1 in %.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-ncprof.sh : prints every command; checks the image literals, every helper (+ selftests),
# the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
export XNC_IMAGE=${XNC_IMAGE:-radiance-local:ds41-int7-0391a6ec-5c13e555-l} XNC_ITEM=${XNC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-ncprof}
export XNC_OUT=${XNC_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ncprof}
export XNC_UARMS=${XNC_UARMS:-S A} XNC_ALONE=${XNC_ALONE:-12} XNC_PWARM=${XNC_PWARM:-16} XNC_PALONE=${XNC_PALONE:-32}
export XNC_DUR=${XNC_DUR:-60} XNC_WARM=${XNC_WARM:-15} XNC_EVERY=${XNC_EVERY:-16} XNC_W1=${XNC_W1:-4096:20:40} XNC_W2=${XNC_W2:-512:15:55}
export XNC_DISKMIN_MB=${XNC_DISKMIN_MB:-11000} XNC_P=${XNC_P:-1} XNC_D=${XNC_D:-1} XNC_DORDER=${XNC_DORDER:-1221} XNC_LDEPTH=${XNC_LDEPTH:-64000}
export XNC_SIDECAR=${XNC_SIDECAR:-dspark-dlw-q8_0.rad} XNC_SIDESHA=${XNC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/nc/dsm-ncprof.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xnc: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
