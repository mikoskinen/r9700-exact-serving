#!/bin/bash
# 57-p4zyzzzzz-ds-o-eport.sh (queue item, 10.11.2026, ds41 eport agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~65 min: a one-card perfmoe byte check ~5 min, then two starts ~27-30 min each) -- fn4's EXL3 expert-kernel levers ported to the DeepSeek-V4.1
# line (fn4's commits, no re-implementation), A/B'd IN ONE PROCESS a start on the ds41-eport image: B = exl3_dz 1 (fn4 DZ:
# the spare workgroups zero the dead rows directly instead of the integer-modulo scan; decode + prefill band) + exl3_mapshare 1
# (fn4 E9: down reuses gate_up's routing map; prefill band, passes of 17+ tokens), A = both 0 (today's paths); gg_gemv 2
# (lever D2) on in both arms. Start nst0 (NST 0: c1 / c4 / c8 are decode-band passes, B vs A = DZ) and start dspark (NST 5
# adaptive: c4 / c8 verify in the prefill band, B vs A = DZ + E9). Blocks A B B A (POST /admin/fn4 {exl3_dz, exl3_mapshare,
# gg_gemv 2, epoch_bump}; stepprobe essay,code 512, mb c4 / c8 75 s), the B-vs-A summary per load point with t95, identity
# last (28k copy on a cold prompt + 20-prompt greedy set at c1, B vs A byte for byte and vs bz dsd-d0). E10 (the fused
# gate_up_down) is not ported: exl3r4d's fused launch is K 4 only and takes no SwiGLU limit (DS: K 2 / 2.5, limit 10).
# Nothing goes into a production line without the user's go. IMAGE XEP_IMAGE (default in the body), XEP_STARTS="nst0" for
# the decode start alone (~30 min).
# First P0 (one card, HIP 0, no server): the image's perfmoe-eport, chain y A vs B byte for byte at DS geometry M 4-48 (a
# difference stops the item) + DZ's dispatch us. XEP_P0=0 skips it.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-o-eport.sh : prints every command, runs and writes nothing.
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-eport.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xep: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
