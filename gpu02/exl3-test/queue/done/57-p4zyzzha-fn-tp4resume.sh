#!/bin/bash
# 106-fn-tp4resume.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~1 h 50 min for the
# two arms) -- why 99's pipelined prefill (m4a: MNBT 4096 --prefill-span 2) lost the u16 cc_sim (total gen 218.4 vs 230.2
# tok/s -5.1 %, TTFT p50 0.96 vs 0.67 s, uncached prompt p50 3108 vs 1914 tokens, cache 96.3 vs 97.4 %) while winning every
# microbench (prefill +8.1..+10.2 %), and the param-only fix.
# CAUSE (1.15.1 source, core/sched/geometry.cpp + scheduler.cpp): the default --prefix-resume exact lets a hit resume only
# where a cold prefill of the prompt starts a chunk (ChunkGeometry::grid). With a span, a lone prefill's chunks are span x
# interval long from the first interval on, so grid = 2 x 2048 = 4096 from origin 2048 (geometry.cpp: g.grid = h * I,
# g.grid_origin = I), and the prompt's tail checkpoint is grid_floor(end - 1) (Scheduler::tail_checkpoint, exact branch) --
# up to 4096 tokens before the end instead of up to 2048 at span 1. Every multi-turn cc turn resumes from the previous
# prompt's tail checkpoint, so it re-prefills ~1 interval more on average: +1194 tokens uncached p50 measured, ~+1024
# expected. plan_chunk also gives a spanned chunk one checkpoint, at its end. Exact cannot keep a 2048 grid with 4096-token
# steps (a hit must recompute the cold run's own chunks), so the fix is --prefix-resume nearest: its tail checkpoint is the
# prompt's end rounded down to the quantum (taken when at least a quarter interval past the last boundary), independent of
# the span -- finer than stock's exact grid too.
# EXACTNESS: nearest changes no arithmetic; it drops exact's guarantee that a prefix hit's output equals a cold run of the
# same prompt bit for bit (the recomputed tail is chunked from a different start, and a kernel may round a row differently
# at another chunk length). Output vs stock is still the same function; hit-vs-cold byte identity is what is given up.
#   m4n   m4a + --prefix-resume nearest (TP4 all_vram MML 262144 MNBT 4096 --prefill-span 2)
#   k4n   the stock line + --prefix-resume nearest (nearest's own effect at span 1)
# = 93c-fn-tp4ab.sh (93b + these arms + X93_EXTRA_PAIRS) beside this file with ARMS="m4n k4n" X93_REF=k4s (the night's fast
# k4sf is the reference both arms are paired with) X93_EXTRA_PAIRS="m4a:m4n" (m4n against 99's m4a: the same prefill line,
# exact vs nearest) X93_ITEM=106-fn-tp4resume. Phases as 93b: gate, slow-state probe (+ restart), produp + probe, mb c4 / c8
# / c16, prefill 16k + fresh 64k / 128k, mb pf, BetterBench, cc_sim u16 fixed-work trace-seeded; state-aware pairs.
# DRYRUN=1 bash 106-fn-tp4resume.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"m4n k4n"} X93_REF=${X93_REF:-k4s} X93_ITEM=${X93_ITEM:-106-fn-tp4resume} X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"m4a:m4n"}
for f in "$H/93c-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93c-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 106-fn-tp4resume: VIRHE 93c-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
