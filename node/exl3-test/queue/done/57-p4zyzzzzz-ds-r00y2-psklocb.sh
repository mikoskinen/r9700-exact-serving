#!/bin/bash
# 57-p4zyzzzzz-ds-r00y2-psklocb.sh (queue item, 11.11.2026, PSK cone-exactness localiser for ds-main-port; at the HEAD of the queue:
# sorts first in C and en_US; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~20 min, NO evict) -- tag
# "RESULT xpsl ", out runs/xdsm/psklocb, body queue-ar/dsmain/pskloc/dsm-psklocb.sh (= dsm-pskloc.sh + alone / loaded deepq runs +
# pskloc.py's labelled pairs; loaded runs via psk/pskld.py). Image radiance-local:ds41-pskfix-10202880-5c13e555-l (ds41-psk-fix
# 1020288 = ds41-psk 4d4da8e + 77eefea ds41_psk 2 (diagnostic: the step_bi path, no cut) + cf7bbc7 THE FIX (a cut layer's indexer
# keeps the step's row-split deal: index_keep is rank-local) + 1020288 ds41_psk 3 (diagnostic: the cut with ds41-psk's re-dealt
# indexer); deepseek41.so layered on the ds41-psk a35c69a5b9c9 image; unset / ds41_psk 0 = int5, hostcheck IDENTICAL).
# WHY (ds-main-port / coordinator 11.11., r00y): r00y's trace put ds41_psk 1's change at L28.sel -- the cut re-deals the indexer's
# row split while the candidate layer's keep is rank-local. Here: is the fix exact (B == A on every trusted traced row, the 8k /
# 16k / 64k / 136k answers == A), and why did r00z's loaded 64k look exact (D = the old deal under load, traced vs loaded A)?
# ONE NST 5 START = r00z's + the tracer (positions < 4096): traced A (ds41_psk 0) / B (1, the fix) at depth 8000 (2048-row pure
# prefill chunks); alone B at 16k / 136k / 64k vs r00z's A answers / dsm-225; loaded (c4 + one cold 64k deepq prompt) A, B, D
# (3, the old deal); pskloc.py pairs B:A, LB:LA, LD:LA.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00y2-psklocb.sh : prints every command; checks the image literals, every helper; no server, no GPU.
export XPL_IMAGE=${XPL_IMAGE:-radiance-local:ds41-pskfix-10202880-5c13e555-l} XPL_ITEM=${XPL_ITEM:-57-p4zyzzzzz-ds-r00y2-psklocb}
export XPL_PROV=${XPL_PROV:-pskfix-PROVENANCE.txt} XPL_OUT=${XPL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/psklocb}
export XPL_ORDER=${XPL_ORDER:-AB} XPL_ARMVALS=${XPL_ARMVALS:-A=0 B=1 C=2 D=3} XPL_DEPTH=${XPL_DEPTH:-8000} XPL_NTOK=${XPL_NTOK:-512}
export XPL_PMAX=${XPL_PMAX:-4096} XPL_DEEPS=${XPL_DEEPS-B:16000 B:135000 B:64000} XPL_LOADS=${XPL_LOADS-A B D} XPL_LDEPTH=${XPL_LDEPTH:-64000}
export XPL_CUT=${XPL_CUT:-B=21:160:128,LB=21:160:128,LD=21:160:128} XPL_PAIRS=${XPL_PAIRS:-B:A,LB:LA,LD:LA}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/pskloc/dsm-psklocb.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpsl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
