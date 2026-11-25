#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts AFTER ...-zsoak16.sh and ...-zsoak16z-rfst.sh in C and en_US order); gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~23 min estimate (two fresh starts ~11.5 + ~9.5, summary ~1; budget KVR_BUDGET_MIN 42: a start begins only if
# elapsed + KVR_STARTEST 11 fit; NO evict, NO build, NO P0) -- THE KV-RESERVATION vs EXPERT-RESIDENCY TRADE-OFF on the DS candidate int16
# (MEASUREMENT ONLY; ds-main-port 13.11.): one FRESH START per arm = zzzc16's int16 start (image radiance-local:ds41-fuse-e231d919-38b88ec6-l
# (4419a17d60a7), RADIANCE_DS41_RFUSE=2 declared, ngshare, no ATTACH, compose guard, MAXSEQS 16, THE int16 PIN posted as O and READ BACK) with
# ONLY --host-pool-mib moved (the knob, radiance-int14c: planner.cpp:103 expert_vram_floor = all experts + spares + 2 x the largest unit - the
# host room; engine_bringup.cpp:2250-2255 host room = --host-pool-mib; vram_budget.cpp:189-199 experts = the floor, KV cache = elastic - floor):
# K35 16800 (KV pool ~-35 %) and K60 14900 (~-60 %); K0 19456 NOT re-run (zzzc16 / zzzfuse references). Per start: the placement read-back
# (kvsum.py place: the knob in force, owned / KV- / arena-lendable slots a rank, KV tokens + sessions; VIRHE -> no reading), identity (warm-up,
# a1-F vs ref12 a1 + zzzc16 a1-F; ANY difference = VIRHE, that start's speed reads STOP), c8 gens x 2, c16 gens x 1, 1u essay 1 discarded + 3.
# Refusal probes exl3_x3d 5. Summary kvres/kvsum.py, lines "KVRES ...". Tag "RESULT xkvr ", out runs/xdsm/kvres, body queue-ar/dsmain/kvres/
# dsm-kvres.sh (a COPY of zzzc16/dsm-zzzc16.sh 00cfc8b8 made by kvres/mkbodykv.py). KVR_SW=0 = a mock run (queue-ar/tools/mockrad;
# kvres/mockrun-kvres.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres.sh : prints every command; checks the image literals, the K25U plugin
# md5, every helper (+ selftests), zzzc16's tables and steps.py md5s, the references, the ngramshare files, the sidecars, the yamls, the disk.
export KVR_IMAGE=${KVR_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-l} KVR_ITEM=${KVR_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres}
export KVR_OUT=${KVR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kvres} KVR_SW=${KVR_SW:-1}
export KVR_ARMS=${KVR_ARMS-K35:16800 K60:14900} KVR_C8B=${KVR_C8B:-2} KVR_C16B=${KVR_C16B:-1} KVR_1U=${KVR_1U:-1} KVR_STARTEST=${KVR_STARTEST:-11}
export KVR_C8GN=${KVR_C8GN:-2} KVR_C16GN=${KVR_C16GN:-1} KVR_U1N=${KVR_U1N:-3}
export KVR_MX=${KVR_MX:-0} KVR_F4=${KVR_F4:-0} KVR_F4G=${KVR_F4G:-0} KVR_MXCONFIRM=${KVR_MXCONFIRM:-0} KVR_GSCONC=${KVR_GSCONC:-16} KVR_MAXSEQS=${KVR_MAXSEQS:-16}
export KVR_PLUGMD5=${KVR_PLUGMD5:-7b2f66d6} KVR_SWL=${KVR_SWL:-1} KVR_SWLG=${KVR_SWLG:-75} KVR_XPMIN=${KVR_XPMIN:-1048576}
export KVR_A4MD5=${KVR_A4MD5:-d55ad354} KVR_A4N=${KVR_A4N:-95} KVR_A11MD5=${KVR_A11MD5:-8e4fc84e} KVR_A11NE=${KVR_A11NE:-8}
export KVR_XB=${KVR_XB:-2} KVR_XREFUSE=${KVR_XREFUSE:-5} KVR_RFDECL=${KVR_RFDECL:-2} KVR_POST=${KVR_POST:-10}
export KVR_ESSAY_MD5=${KVR_ESSAY_MD5:-47a39dac581a} KVR_COLD=${KVR_COLD:-0} KVR_C4=${KVR_C4:-0}   # zzzc16's c4 / cold sections: not in this item
export KVR_STPMD5=${KVR_STPMD5:-569fd59b} KVR_DISKMIN_MB=${KVR_DISKMIN_MB:-11000} KVR_BUDGET_MIN=${KVR_BUDGET_MIN:-42}
export KVR_SIDECAR=${KVR_SIDECAR:-dspark-dlw-q8_0.rad} KVR_SIDESHA=${KVR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/kvres/dsm-kvres.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xkvr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
