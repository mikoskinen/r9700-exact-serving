#!/bin/bash
# 57-p4zyzzzzz-ds-hy-stack2.sh (queue item, 11.11.2026, ds-main-port; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~47 min, ~24 with XS2_NST0=0) -- THE DS STACK CONFIRMATION 2 (tag xs2, out runs/xdsm/stack2) on the ds41-int3 image: =
# 57-p4zyzzzzz-ds-hx-stack.sh (copied; hx-stack's files untouched) with body dsm-stack2.sh (= dsm-stack.sh + the int3 levers) and
# yaml ds41-stack2-tp4.yaml (+ env_file ${XS2_ENVFILE}). ds41-int3 = ds41-small 2b03903 (int2 + I1, conf + dspark_conf_rows, K1 a/e,
# ctx skip, KVWIN) + ds41-gpf edc9a9b (gg_pf / f32_pf) + ds41-nvlat b12aca4 (NVG, unused here) + ds41-walk eecd93a (dspark_walk);
# every new lever off unless set. TWO STARTS, each with hx-stack's start env (RADIANCE_DS41_GG_GEMV=2, RADIANCE_AQL_ACQ=rule-w,
# K1 c/d/f declared, turnover 0) + the start's lever env (written to <tag>.env and exported), the pins in process every block:
#   start 1 (XS2_NST0=1): NST 0, N = hx-stack's N + gg_pf 1 (RADIANCE_DS41_GG_PF=1 + RADIANCE_DS41_GG_PF_TABLE=XS2_GGPF_TABLE at
#     start) + f32_pf 25 (RADIANCE_DS41_F32_PF=25); blocks N N N N;
#   start 2: DSpark NST 5 --draft-window adaptive, A = the same + draft_adaptive 1, D7 (draft_depth0 1), hybrid 1 (match 8),
#     drafter_mx 1, dspark_walk 1 (RADIANCE_DS41_DSPARK_WALK=1 at start), the small n5 set: ds41_argather 48
#     (RADIANCE_DS41_ARGATHER_MAX=48), idx_split 1, ds41_qnfold 1 + ds41_swfold 1 (RADIANCE_DS41_QNFOLD=1 / _SWFOLD=1),
#     draft_ctxskip 1 (RADIANCE_DRAFT_CTXSKIP=0), dspark_conf 1 + dspark_conf_rows 1 (RADIANCE_DS41_DSPARK_CONF=/models/dspark-conf.f32);
#     blocks A A A A.
#   HELD OFF (one edit each in the export line, once hxz-vramstack lands): XS2_VRAM (empty; "all" or names of arena bf16 emb shadow
#     xpoff), XS2_LEND (0; 1 = lever DL declared off at the NST 5 start + draft_lend 1, draft_probe_backoff 3 pinned), XS2_BORROW
#     (0; 1 = + draft_lend_borrow 1, needs XS2_LEND=1).
# In each start: decode blocks (stepprobe essay / code (c1), editprobe, mb c4 / c8 75 s), one 136k cold prefill (TTFT + answer vs
# dsm-225), identity (28k copy + greedy c1 vs bz d0). Summary as hx-stack's (vs gz gg2-gemv2 and the targets) + each arm vs
# hx-stack's same arm (cross-start).
# IMAGE radiance-local:ds41-int3-1116248a-5c13e555-l (XS2_IMAGE; ds41-int3 1116248, image e19c28ce372f); XS2_NST0, STACK_ARMS, HYB, XS2_DRAFTER_MX, XS2_VRAMENV, XS2_GGPF,
# XS2_GGPF_TABLE, XS2_F32PF, XS2_WALK, XS2_SMALL5, XS2_VRAM, XS2_LEND, XS2_BORROW. Set them by editing the export line below before
# moving this file into queue/ (the queue runner passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hy-stack2.sh : prints every command; checks the image, every helper, the sidecar and the yaml (no
# server, no GPU), writes nothing.
export XS2_IMAGE=${XS2_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l} XS2_NST0=${XS2_NST0:-1} STACK_ARMS=${STACK_ARMS:-stack} HYB=${HYB:-1} XS2_DRAFTER_MX=${XS2_DRAFTER_MX:-1}
export XS2_VRAMENV=${XS2_VRAMENV-RADIANCE_PLACE_TURNOVER=0} XS2_GGPF=${XS2_GGPF:-1} XS2_F32PF=${XS2_F32PF:-25} XS2_WALK=${XS2_WALK:-1} XS2_SMALL5=${XS2_SMALL5:-1}
export XS2_GGPF_TABLE=${XS2_GGPF_TABLE-q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422}
export XS2_VRAM=${XS2_VRAM-} XS2_LEND=${XS2_LEND:-1} XS2_BORROW=${XS2_BORROW:-0} XS2_VRAM=${XS2_VRAM:-all}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-stack2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xs2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
