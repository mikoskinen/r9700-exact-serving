#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse.sh (queue item, STAGED 13.11.2026 by a fusion builder for ds-main-port / the coordinator --
# NOT QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zzzaql and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order (the
# brief's name "...-zzzfuse" sorts AFTER the fn-* items under en_US, which ignores the hyphens: "...zzzzzzzzzfuse" > "...zzzzzzzzzfn"); gpu02,
# all 4 cards (P0: one card by chip UUID), TP4, MML 524288, every radiance container DOWN; ~25 min estimate, budget ZFU_BUDGET_MIN 32 for the
# sections (a section past it skipped, logged), <= 35 with the summary; NO evict, NO build) -- THE DEPENDENT-LAUNCH FUSION SCREEN on THE
# DS BASE int15: lever FUSE-R (fn4 ds41_rfuse 1 = the router's selection + sort in one launch, 2 = the projection + selection + sort in one
# launch: the GEMM's last workgroup ranks and sorts) and lever FUSE-M (libr4d fn4 ds41_mix_fin 1: mhc_mix's one-launch FIN form at <= 48
# rows), each the stock kernels' code with only the launch boundary removed (exact: bytes held by r4d_selftest / r4d_emu), + lever RCO
# (core fn4 route_copy_ov 1, the launch builder's ds41-launch 0f65666, merged: the routing report's copy recorded with its ranges, so the
# next dispatch starts unordered beside it; launch mechanics only). IMAGE
# radiance-local:ds41-fuse-<head8>-38b88ec6-l = int15 (bc91c460c13f) + ds41-fuse's radiance / r4d_selftest / libr4d.so / deepseek41.so
# (every switch 0, RADIANCE_DS41_RFUSE / RADIANCE_AQL_ROUTE_OV unset = int15: hostcheck), THE int15 PIN, + RADIANCE_DS41_RFUSE=0 (FUSE-R declared, off), MAXSEQS 16,
# ONE FRESH START (ngshare, no ATTACH, compose guard). 0 P0 on one card (GPU-dbba5c851cfef7b3): r4d_selftest --case rfuse / mixfin -- bytes
# (a FAIL drops the form: NEEDS USER) + UNSTAMPED per-call times, the stock launches back to back vs fused (FUSEPERF); GATE: a form must
# beat the pair by >= 1.0 us at M 8 warm and cold (not slower cold at M 1 / 16), else it is dropped; NO form left -> the item STOPS after P0
# (~2-3 min, no start; ZFU_P0ONLY=1 = always stop after P0, a kbench-only run); 1 IDENTITY FIRST under B and Y
# (every exact form on + RCO): a1 vs ref12's a1 (+ a1-B vs zzk25s a1-Z), gs16-Y, STRICT staggered ld-Y -- ANY difference: VIRHE and STOP;
# 2 c8 WORK-BASED B F F B (F = ALL: ds41_rfuse 2 + route_copy_ov 1; FUSE-M not in F, predicted to lose); 3 singles R2 R1 R M one c8 block
# each; 4 c16 B / F; 5 1u
# essay B / F. Judged on the STEP-LOG decode p50 (all decode steps) + c8 gens agg (zzzfuse/fusesum.py, lines "ZFUSE ..."). Tag "RESULT
# xzfuse ", out runs/xdsm/zzzfuse, body queue-ar/dsmain/zzzfuse/dsm-zzzfuse.sh (a COPY of zzk25s/dsm-zzk25s.sh abbf540f made by
# zzzfuse/mkbody.py). ZFU_SW=0 = a mock run (queue-ar/tools/mockrad; zzzfuse/mockrun-zzzfuse.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse.sh : prints every command; checks the image literals (int9 .. int15 + the
# fuse image's), the K25U plugin md5, every helper (+ selftests), the tables' and steps.py's md5s, the ngramshare files, the sidecars, the
# yamls, the disk.
export ZFU_IMAGE=${ZFU_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-l} ZFU_ITEM=${ZFU_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse}
export ZFU_OUT=${ZFU_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzzfuse} ZFU_SW=${ZFU_SW:-1}
export ZFU_MX=${ZFU_MX:-0} ZFU_F4=${ZFU_F4:-0} ZFU_F4G=${ZFU_F4G:-0} ZFU_MXCONFIRM=${ZFU_MXCONFIRM:-0} ZFU_GSCONC=${ZFU_GSCONC:-16} ZFU_MAXSEQS=${ZFU_MAXSEQS:-16}
export ZFU_PLUGMD5=${ZFU_PLUGMD5:-7b2f66d6} ZFU_SWL=${ZFU_SWL:-1} ZFU_SWLG=${ZFU_SWLG:-75} ZFU_XPMIN=${ZFU_XPMIN:-1048576}
export ZFU_A4MD5=${ZFU_A4MD5:-d55ad354} ZFU_A4N=${ZFU_A4N:-95} ZFU_A11MD5=${ZFU_A11MD5:-8e4fc84e} ZFU_A11NE=${ZFU_A11NE:-8}
export ZFU_XB=${ZFU_XB:-2} ZFU_XREFUSE=${ZFU_XREFUSE:-5} ZFU_GORDER=${ZFU_GORDER-B F F B} ZFU_SINGLES=${ZFU_SINGLES-R2 R1 R M} ZFU_ALLMIX=${ZFU_ALLMIX:-0} ZFU_ALLRCO=${ZFU_ALLRCO:-1} ZFU_C8GN=${ZFU_C8GN:-2}
export ZFU_C16N=${ZFU_C16N:-1} ZFU_C16GN=${ZFU_C16GN:-1} ZFU_C16ARMS=${ZFU_C16ARMS-B F} ZFU_POST=${ZFU_POST:-10} ZFU_IDARMS=${ZFU_IDARMS-B Y} ZFU_GSARM=${ZFU_GSARM:-Y} ZFU_LDARM=${ZFU_LDARM:-Y}
export ZFU_1U=${ZFU_1U:-1} ZFU_1UARMS=${ZFU_1UARMS-B F} ZFU_U1N=${ZFU_U1N:-3} ZFU_ESSAY_MD5=${ZFU_ESSAY_MD5:-47a39dac581a}
export ZFU_P0_UUID=${ZFU_P0_UUID:-GPU-dbba5c851cfef7b3} ZFU_P0SEC=${ZFU_P0SEC:-300} ZFU_P0MARGIN_US=${ZFU_P0MARGIN_US:-1.0} ZFU_P0STOP=${ZFU_P0STOP:-1} ZFU_P0ONLY=${ZFU_P0ONLY:-0}
export ZFU_C8WARM=${ZFU_C8WARM:-10} ZFU_LDCONC=${ZFU_LDCONC:-4} ZFU_LDPF=${ZFU_LDPF:-8192} ZFU_DISKMIN_MB=${ZFU_DISKMIN_MB:-11000} ZFU_BUDGET_MIN=${ZFU_BUDGET_MIN:-32}
export ZFU_STPMD5=${ZFU_STPMD5:-569fd59b}
export ZFU_SIDECAR=${ZFU_SIDECAR:-dspark-dlw-q8_0.rad} ZFU_SIDESHA=${ZFU_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzzfuse/dsm-zzzfuse.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xzfuse: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
