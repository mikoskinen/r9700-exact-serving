#!/bin/bash
# 108-fn-tp4pmin.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h 30 min, three
# arms) -- speculative draft p_min on the k4n line (llama.cpp's --draft-p-min: a low-confidence draft is never verified).
# IMAGE radiance-local:1.15.1-pmin-47e859a-stampcards (X93_PMIN_IMAGE) = the stampcards image + bin/radiance from branch
# fn4-pmin 47e859a (on fn4-stamp-cards c0cbe5a): env RADIANCE_DRAFT_PMIN, unset/empty = off = stock code paths.
# With p_min set (core/engine_draft.cpp pmin_compute, engine.cpp issue_ahead):
#   - every draft round's reranked top-8 exact logits of the drafting rows (`mtp_dval`, + ids) are read back on rank 0's
#     stream; a proposal's probability = column 0's share of the softmax over those 8 (an upper bound of the full-vocab
#     probability; llama.cpp also reads its draft sampler's top-k);
#   - a sequence's chain stops at its first round below p_min (0 to 3 drafts kept);
#   - the kept total is rounded UP to a multiple of RADIANCE_DRAFT_PMIN_BUCKET (X93_PMIN_BUCKET, 8) with the dropped
#     drafts most likely to be reached and accepted, so the verify pass keeps few recorded shapes (24 kept per card);
#   - the step issued ahead WAITS for the drafter's last round (stock issues it behind the drafter) -- the host build +
#     issue of the next step is no longer hidden; the engine line reports the wait;
#   - a sequence cut to none gets no prompt-lookup fallback; under --draft-coupling or without a top-R head it stays off.
# EXACTNESS: no arithmetic changes; the verify rows' count changes, and the 1.15.1 kernels compute each row independently
# of M (accept.h: speculative text matches a non-speculative run at temperature 0), so greedy text must be unchanged --
# checked by the warm-up answers vs 106's k4n and the 33k copy test vs 95's stock m2, byte for byte.
#   k4nP5 / k4nP7 / k4nP9   k4n + RADIANCE_DRAFT_PMIN 0.5 / 0.7 / 0.9
# = 93d-fn-tp4ab.sh beside this file with ARMS="k4nP5 k4nP7 k4nP9" X93_REF=k4n (106's rows, no baseline re-run)
# X93_ITEM=108-fn-tp4pmin. Per arm the engine's counters around mb c4 / c8 / c16 and cc_sim ("engine stats": verified
# drafts / accepted a step, acceptance, verify rows a step, recorded-pass play share) and its own "draft p_min" line
# (proposed vs verified drafts, refilled, rows cut to none, wait a step).
# DRYRUN=1 bash 108-fn-tp4pmin.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"k4nP5 k4nP7 k4nP9"} X93_REF=${X93_REF:-k4n} X93_ITEM=${X93_ITEM:-108-fn-tp4pmin}
for f in "$H/93d-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93d-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 108-fn-tp4pmin: VIRHE 93d-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
