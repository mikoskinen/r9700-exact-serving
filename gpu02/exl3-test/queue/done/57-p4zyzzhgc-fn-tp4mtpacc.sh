#!/bin/bash
# 113-fn-tp4mtpacc.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h 20 min) --
# why main 451b136's MTP drafts are accepted less under agent load. 109b rmnI (main + --prefill-interleave 0 + nearest):
# mb fine (c4 / c8 / c16 562.2 / 754.1 / 958.6, mb pf 6416 tok/s, copy IDENTICAL, mb acceptance 0.66-0.68) but cc_sim u16
# draft acceptance 0.259 (verified 16.52 / accepted 4.28 a step, 15289 decode steps) against 1.15.1 k4n ~0.354 and 1.17.0
# r17n 0.350; total gen 223.4 vs 242.3. Source reading (v1.17.0..451b136, qwen4exp + the draft path):
#   7f5218e removed 1.17.0's guard in qwen4exp's pass_key: "MTP owns another attention path on the same KV group. The
#           present batch key cannot prove whether that head's attention row takes a split grid, so keep only those passes
#           live" (if draft_pass != 0 return 0) -- draft passes are now recorded and replayed, keyed by the TRUNK's split
#           cap level (ctx / 48 / 8 up to 64: it moves every 384 tokens below 24.6k); long contexts only -> fits "mb fine,
#           agent load bad"
#   e2e9416 chain_pays 1: chained launches on with the MTP drafter (start log: "chained launches: on"; 1.17.0: off); with
#           b0a58b5 every kernel of the step at <= 32 rows -- every draft round -- waits on its predecessor in its prologue,
#           QSA's indexer (sparse only above 2051 tokens of context) included
#   not the drafter's: 7cad854 (gqa8 / gqa16 kernels; Flash-Next runs gqa6), d6109d7 (the merge of VERIFY rows only; the
#           history pass's used row is its last, unchanged), 036bd8f (libr3d, RDNA3), dccc632 (a block drafter's hrow)
# ARMS "rmnIk0d rmnIc0 rmnI2 r17n2" (93g): main + interleave 0 with qwen4exp_fp8.so rebuilt out of tree against main's own
# package -- k0d: 1.17.0's draft-pass guard back (radiance-local:main-451b136-k0d), c0: chain_pays 0
# (radiance-local:main-451b136-c0) -- then main's own plugin (rmnI2) and 1.17.0 (r17n2). Per arm: gate, slow-state probe,
# produp + probe, mb c4 / c8 / c16 (+ engine stats), the acceptance-by-context probe (1k / 32k at 1 user / 32k at 4 users,
# engine stats each), the copy test; cc_sim u16 (+ engine stats) on k0d and c0 only (rmnI's from 109b, r17n's from 109);
# no BetterBench, no prefill sweep. Pairs: every arm vs 109b's rmnI (X93_REF), the two candidates also vs 109's r17n and
# each other. A fix shows as cc_sim acceptance back at ~0.35 with rmnI's decode / prefill numbers kept.
# DRYRUN=1 bash 113-fn-tp4mtpacc.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"rmnIk0d rmnIc0 rmnI2 r17n2"} X93_REF=${X93_REF:-rmnI} X93_ITEM=${X93_ITEM:-113-fn-tp4mtpacc} X93_HANG428=0
export X93_LACC=${X93_LACC:-1} X93_SKIP_PF=${X93_SKIP_PF:-1} X93_SKIP_BB=${X93_SKIP_BB:-1} X93_CC_ARMS=${X93_CC_ARMS:-"rmnIk0d rmnIc0"}
export X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"r17n:rmnIk0d r17n:rmnIc0 rmnIc0:rmnIk0d"}
for f in "$H/93g-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93g-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 113-fn-tp4mtpacc: VIRHE 93g-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
