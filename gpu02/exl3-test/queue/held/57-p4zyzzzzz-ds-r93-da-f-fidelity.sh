#!/bin/bash
# RE-QUEUE 12.11. ~13:22 as r93 (ds-main-port, coordinator): the 13:13 r91 run failed its readback start in 1 s (13:19:47, restart-pr266-daf-rb) on the ghost compose container 622b5b7a9e64 -- r91 was queued before compose-guard; + compose guard; runs after r92 ccfill
# 57-p4zyzzzzzzzzz-da-f-fidelity.sh (queue item, 11.11.2026, ds-main-port DA agent at the coordinator's request; gpu02, all 4 cards;
# every radiance container DOWN; ~40 min) -- LEVER DA: WHERE DO THE PORT AND THE ENGINE DIVERGE. The pilot's adapted drafter B lost
# in the engine (iei anatomy, fixed 5, T=0: prose 2.210 vs A 2.260, code 4.233 vs 4.395, fi 1.714 vs 1.741) where the port had
# simulated +4.2 %. Body queue-ar/dsmain/da/dsm-da3.sh fidelity: (1) the port's numerics variants vs the engine's proposals and the
# port's prediction for A / B / X on the held-out generated documents (no server, 4 cards); (2) a readback of B's overlay from the
# card (-vv, RADIANCE_DUMP_BUF); (3) the ENGINE's score of A / B (best.ovl) / X (ext.ovl, the external fine-tune) on those same
# documents (stack2 env + fixed window 5, T=0, -v; replies must equal the documents); then the KL dump is deleted. Tag
# "RESULT xdaf ", out runs/xdsm/da. IMAGE radiance-local:ds41-da-bd2caa0c-5c13e555-l (sha256:561f4fbc60e3). No training, no cc_sim.
# DRYRUN=1 bash 57-p4zyzzzzzzzzz-da-f-fidelity.sh : prints every command; checks the images, every helper, the inputs.
export DAP_IMAGE=${DAP_IMAGE:-radiance-local:ds41-da-bd2caa0c-5c13e555-l} DAP_ITEM=${DAP_ITEM:-57-p4zyzzzzzzzzz-da-f-fidelity}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/da/dsm-da3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdaf: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B fidelity
