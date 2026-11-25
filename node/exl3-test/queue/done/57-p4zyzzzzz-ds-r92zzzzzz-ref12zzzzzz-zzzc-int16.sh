#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zzzb-fuse and before every -r92zzzzzz-ref12zzzzzzzzz*-fn-* item in C and en_US
# order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~17 min estimate, budget ZC16_BUDGET_MIN 23 for the sections (a
# section past it skipped, logged; the cold 136k goes first), <= 25 with the summary; NO evict, NO build, NO P0) -- THE int16 CONFIRMATION
# (coordinator 13.11., after zzzfuse: F exact and never worse): int16 = int15 + fn4 ds41_rfuse 2 (FUSE-R form 2) + core fn4 route_copy_ov 1
# (RCO), ds41_mix_fin 0. IMAGE radiance-local:ds41-fuse-e231d919-38b88ec6-l (4419a17d60a7) = int15 (K25U, bc91c460c13f) + ds41-fuse e231d91's
# radiance / r4d_selftest / libr4d.so / deepseek41.so. ONE FRESH START (ngshare, no ATTACH, compose guard), MAXSEQS 16, the int15 start words +
# RADIANCE_DS41_RFUSE=2 (FUSE-R declared and started at form 2); THE int16 PIN posted as O right after it and READ BACK (GET /admin/fn4 every
# switch, the 4 ranks on /stats, the start log's declare lines, the "rco" / "fuse" engagement counters; a mismatch = VIRHE, nothing read).
# Arms in-process: B = int15 (the three fusion switches 0), F = int16. 1 IDENTITY FIRST under F (strict): a1-F vs ref12's a1 AND vs zzzfuse's
# a1-B, gs16-F vs a1-F, staggered ld-F vs a1-F -- ANY difference: VIRHE and STOP; 2 1u ABBA B F F B (each block 1 discarded + 3 essay reads):
# "1u int16 = X tok/s (>= 90: YES / NO)" + B; 3 c4 gens window (gens 3) B F F B after one discarded c4 block; 4 optional cold 136k under F vs
# ref14's answer. Refusal probes exl3_x3d 5 (TESTING.md 13.11. NEW DS BASE int15 / K25U SWITCH-VALUE NOTE). Summary zzzc16/zc16sum.py, lines
# "INT16 ...". Tag "RESULT xzc16 ", out runs/xdsm/zzzc16, body queue-ar/dsmain/zzzc16/dsm-zzzc16.sh (a COPY of zzzfuse/dsm-zzzfuse.sh 2a0f352d
# made by zzzc16/mkbody16.py). ZC16_SW=0 = a mock run (queue-ar/tools/mockrad; zzzc16/mockrun-zzzc16.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16.sh : prints every command; checks the image literals (int9 .. int15 + the
# fuse image's), the K25U plugin md5, every helper (+ selftests), the tables' and steps.py's md5s, the references, the ngramshare files, the
# sidecars, the yamls, the disk.
export ZC16_IMAGE=${ZC16_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-l} ZC16_ITEM=${ZC16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16}
export ZC16_OUT=${ZC16_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzzc16} ZC16_SW=${ZC16_SW:-1}
export ZC16_MX=${ZC16_MX:-0} ZC16_F4=${ZC16_F4:-0} ZC16_F4G=${ZC16_F4G:-0} ZC16_MXCONFIRM=${ZC16_MXCONFIRM:-0} ZC16_GSCONC=${ZC16_GSCONC:-16} ZC16_MAXSEQS=${ZC16_MAXSEQS:-16}
export ZC16_PLUGMD5=${ZC16_PLUGMD5:-7b2f66d6} ZC16_SWL=${ZC16_SWL:-1} ZC16_SWLG=${ZC16_SWLG:-75} ZC16_XPMIN=${ZC16_XPMIN:-1048576}
export ZC16_A4MD5=${ZC16_A4MD5:-d55ad354} ZC16_A4N=${ZC16_A4N:-95} ZC16_A11MD5=${ZC16_A11MD5:-8e4fc84e} ZC16_A11NE=${ZC16_A11NE:-8}
export ZC16_XB=${ZC16_XB:-2} ZC16_XREFUSE=${ZC16_XREFUSE:-5} ZC16_RFDECL=${ZC16_RFDECL:-2} ZC16_POST=${ZC16_POST:-10}
export ZC16_1U=${ZC16_1U:-1} ZC16_U1ABBA=${ZC16_U1ABBA-B F F B} ZC16_U1N=${ZC16_U1N:-3} ZC16_ESSAY_MD5=${ZC16_ESSAY_MD5:-47a39dac581a}
export ZC16_C4=${ZC16_C4:-1} ZC16_C4ABBA=${ZC16_C4ABBA-B F F B} ZC16_C4GN=${ZC16_C4GN:-3} ZC16_C4WARMUP=${ZC16_C4WARMUP:-1} ZC16_C4WARMARM=${ZC16_C4WARMARM:-F}
export ZC16_COLD=${ZC16_COLD:-1} ZC16_CDEPTH=${ZC16_CDEPTH:-136000} ZC16_COLDEST=${ZC16_COLDEST:-2} ZC16_R14CMD5=${ZC16_R14CMD5:-3623688c}
export ZC16_LDCONC=${ZC16_LDCONC:-4} ZC16_LDPF=${ZC16_LDPF:-8192} ZC16_DISKMIN_MB=${ZC16_DISKMIN_MB:-11000} ZC16_BUDGET_MIN=${ZC16_BUDGET_MIN:-23}
export ZC16_STPMD5=${ZC16_STPMD5:-569fd59b}
export ZC16_SIDECAR=${ZC16_SIDECAR:-dspark-dlw-q8_0.rad} ZC16_SIDESHA=${ZC16_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzzc16/dsm-zzzc16.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xzc16: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
