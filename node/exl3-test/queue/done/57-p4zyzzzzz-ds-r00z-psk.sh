#!/bin/bash
# 57-p4zyzzzzz-ds-r00z-psk.sh (queue item, 11.11.2026, ds-main-port PSK agent for the coordinator; at the HEAD of the queue: sorts
# before 57-p4zyzzzzz-ds-r02z-z0occ2 / -r04-bi-gyz-chunk / -r05-pipeprof (C and en_US order); gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~28 min, NO evict) -- LEVER PSK (prefill skip above the last cache source), tag "RESULT xpsk ",
# out runs/xdsm/psk, body queue-ar/dsmain/psk/dsm-psk.sh (+ pskld.py = longid2.py + --depth / --ntok / --pf-tok 0 / the prefill
# window's c4 readings). Image radiance-local:ds41-psk-4d4da8e9-5c13e555-l (a35c69a5b9c9, XPS_IMAGE; ds41-psk = ds41-int5 283c7a1 + lever PSK:
# fn4 ds41_psk, RADIANCE_DS41_PSK; hostcheck: unset = int5 at NST 0 and NST 5).
# WHY (coordinator 11.11.): the 136k cold TTFT is 50.03 s, target <= 35 s; layers 21-39 persist only their 128-key windows and the
# 37-39 taps, so a prefill chunk runs them over its end cone only (C_l = 160 + (39 - l) x 128 rows). Profile-based estimate:
# -29 % device time alone (4096-row chunks), -17 % in 2048-row mixed chunks.
# ONE NST 5 START (the z0occ2 start + RADIANCE_DS41_PSK=0), in process: A = ds41_psk 0, B = ds41_psk 1. Cold TTFT ABBA at
# 136k / 64k / 16k (deepq.py's fixed prompt, a prefix-cache reset each) + acceptance; resume exactness under B (a 64k hit on the
# 136k run's snapshot, the 136k on its tail snapshot); 28k copy A / B and the greedy set c1 under B vs bz d0; c4 decode + one
# cold 64k prompt ABBA (mixed-step ms, c4 tok/s in the prefill window, TTFT, the loaded answer vs alone).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00z-psk.sh : prints every command; checks the image literals, every helper, the sidecars, the
# yaml; no server, no GPU, writes nothing.
export XPS_IMAGE=${XPS_IMAGE:-radiance-local:ds41-psk-4d4da8e9-5c13e555-l} XPS_ITEM=${XPS_ITEM:-57-p4zyzzzzz-ds-r00z-psk}
export XPS_OUT=${XPS_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/psk}
export XPS_ORDER=${XPS_ORDER:-ABBA} XPS_LORDER=${XPS_LORDER-ABBA} XPS_DEPTHS=${XPS_DEPTHS:-135000 64000 16000} XPS_LDEPTH=${XPS_LDEPTH:-64000}
export XPS_DS=${XPS_DS:-1}
export XPS_SIDECAR=${XPS_SIDECAR:-dspark-dlw-q8_0.rad} XPS_SIDESHA=${XPS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/psk/dsm-psk.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpsk: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
