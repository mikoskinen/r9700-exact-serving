#!/bin/bash
# 57-p4zyzzzzz-ds-r00zz-psk2b.sh (queue item, 11.11.2026, ds-main-port PSK design-2 builder for the coordinator; right after
# r00zz-psk2, before r00zzz-a-lf (C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~17 min,
# NO evict) -- LEVER PSK DESIGN 2, the anchor fallback for real: tag "RESULT xpsk2b ", out runs/xdsm/psk2b, body
# queue-ar/dsmain/psk2b/dsm-psk2b.sh (= r00zz's dsm-psk2.sh start / helpers + per-K blocks on own texts) + p2bq.py (pskld.py's
# deep request at another SRC file offset) + pskld.py (r00z's copy). Image radiance-local:ds41-psk2-d6fcc241-5c13e555-l
# (747d9ae8dc90; ds41-psk2 d6fcc24, r00zz's).
# WHY (coordinator 11.11.): r00zz's K hits resumed at A / D1's snapshots of the same prefix (/reset_prefix_cache keeps checkpoint
# records), so the fallback to the anchor below was not exercised and its TTFT cost not measured. Here each K runs on its own text:
# cold 136k, then hits at 90000 / 58000 / 20000 (+ the tail) must resume at the anchor below (K8 63488 / 30720 / 0, K4 79872 /
# 47104 / 14336) else VIRHE; A on the same text gives the hit-TTFT reference (its own grid snapshots) and cold answers at each
# depth for identity. K for stack4 is picked from this item.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zz-psk2b.sh : prints every command; checks the image, every helper, the sidecars, the yaml;
# no server, no GPU, writes nothing.
export XP2_IMAGE=${XP2_IMAGE:-radiance-local:ds41-psk2-d6fcc241-5c13e555-l} XP2_ITEM=${XP2_ITEM:-57-p4zyzzzzz-ds-r00zz-psk2b}
export XP2_OUT=${XP2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/psk2b}
export XP2_HITS=${XP2_HITS:-90000 58000 20000} XP2_KS=${XP2_KS:-4 8} XP2_DOFF4=${XP2_DOFF4:-200} XP2_DOFF8=${XP2_DOFF8:-400}
export XP2_DS=${XP2_DS:-1}
export XP2_SIDECAR=${XP2_SIDECAR:-dspark-dlw-q8_0.rad} XP2_SIDESHA=${XP2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/psk2b/dsm-psk2b.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpsk2b: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
