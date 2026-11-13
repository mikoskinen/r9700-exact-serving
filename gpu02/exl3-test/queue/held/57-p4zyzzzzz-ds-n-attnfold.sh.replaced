#!/bin/bash
# 57-p4zyzzzzz-ds-n-attnfold.sh (queue item, 10.11.2026, ds41-k1 agent, lever K1; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~2 h) -- lever K1's exact launch folds on DeepSeek-V4.1's attention path, A/B'd IN ONE
# PROCESS by three switches (POST /admin/fn4, 0 = today's ops = A, 1 = the fold = B; all three 1 at start):
#   ds41_kvfold   (c) rmsnorm_rope_roundtrip + kv_store -> one launch writing the window cache
#   ds41_ofold    (d) attn_sparse_sink + rope_rows_roundtrip -> the inverse rope + fp8 round trip in the attention's epilogue
#   ds41_onefill  (f) the MoE shared gate's ones filled once a step, not once a layer
# on today's line (t3 env, ARGATHER on at 4 pinned, F4 off). First the image's r4d_selftest holds each fold to the rows it
# folds, byte for byte, on card 0 (a FAIL stops the item); then blocks A B B A A B (stepprobe c1 -- the headline, est.
# -1.2 ms a 1-user step profiled --, mb c4, mb c8) for all three together, then each alone; the c4-floor diagnosis (the c4
# greedy set run to run, normal vs every prompt prefilled first, at K1 0; once prefilled at K1 1); identity last (the 28k copy,
# the c1 greedy set, a fresh 16k prefill and the 136k deep answer under B and A vs today's files and B vs A). Exact by
# construction. NST 0. No cc_sim. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-k1-1d1f782b-5c13e555-l (XK1_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-n-attnfold.sh : prints every command, runs and writes nothing.
export XK1_IMAGE=${XK1_IMAGE:-radiance-local:ds41-k1-1d1f782b-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-attnfold.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xk1: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
