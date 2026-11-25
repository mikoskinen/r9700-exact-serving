#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts AFTER 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-fn-s-s9soak.sh (FN 231) in C and en_US order: the same 12 z's,
# then "-z" vs "-f"); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~48 min (<= 50 min GPU time; budget ZS16_BUDGET_MIN
# 47 for starting a cycle; NO evict, NO build, NO P0) -- THE DS int16 SOAK (ds-main-port 13.11.): EXACT UNDER LOAD over a long window for
# the production candidate. ONE FRESH START = the int16 confirmation's (zzzc16: image radiance-local:ds41-fuse-e231d919-38b88ec6-l
# (4419a17d60a7), RADIANCE_DS41_RFUSE=2 declared, ngshare, no ATTACH, compose guard, MAXSEQS 16), THE PIN posted right after it and READ
# BACK (GET /admin/fn4 every switch, the 4 ranks on /stats, the declare lines, the engagement probe; a mismatch = VIRHE, nothing read).
# PIN AUTO-CHOICE at the start: the last "RESULT xzc16" run in runs/ds41-exl3.log -- any identity line not IDENTICAL or its STOP line ->
# THE int15 PIN (ds41_rfuse 0, route_copy_ov 0), else THE int16 PIN (ds41_rfuse 2, route_copy_ov 1); missing -> int16; one line
# "SOAK PIN = ...". Refusal probes exl3_x3d 5. Readings: a1 = greedy-20 c1 alone (T 0, 192 tok) + strict vs ref12's a1 and zzzc16's
# a1-F / a1-B; up to 8 cycles (~280 s each) of stack4/loadid.py run --conc 12 (mb c12 load) + the 20 probes each joined 0.5 s in by a
# fresh ~8k-token prefill, strict vs a1 each cycle (a difference = VIRHE, recorded, the soak continues); telemetry every 5 s a card by
# chip UUID (read-only sysfs); a2 control alone; the end engagement probe. Summary zsoak16/soaksum.py, lines "SOAK16 ...", verdict
# "EXACT UNDER LOAD over N cycles / X min: YES / NO". Tag "RESULT xzs16 ", out runs/xdsm/zsoak16, body
# queue-ar/dsmain/zsoak16/dsm-zsoak16.sh (a COPY of zzzc16/dsm-zzzc16.sh 00cfc8b8 made by zsoak16/mkbodys16.py). ZS16_SW=0 = a mock run
# (queue-ar/tools/mockrad; zsoak16/mockrun-zsoak16.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16.sh : prints every command; checks the image literals, the K25U plugin
# md5, every helper (+ selftests), zzzc16's tables and steps.py md5s, the references, the ngramshare files, the sidecars, the yamls, the disk.
export ZS16_IMAGE=${ZS16_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-l} ZS16_ITEM=${ZS16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16}
export ZS16_OUT=${ZS16_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zsoak16} ZS16_SW=${ZS16_SW:-1}
export ZS16_MX=${ZS16_MX:-0} ZS16_F4=${ZS16_F4:-0} ZS16_F4G=${ZS16_F4G:-0} ZS16_MXCONFIRM=${ZS16_MXCONFIRM:-0} ZS16_GSCONC=${ZS16_GSCONC:-16} ZS16_MAXSEQS=${ZS16_MAXSEQS:-16}
export ZS16_PLUGMD5=${ZS16_PLUGMD5:-7b2f66d6} ZS16_SWL=${ZS16_SWL:-1} ZS16_SWLG=${ZS16_SWLG:-75} ZS16_XPMIN=${ZS16_XPMIN:-1048576}
export ZS16_A4MD5=${ZS16_A4MD5:-d55ad354} ZS16_A4N=${ZS16_A4N:-95} ZS16_A11MD5=${ZS16_A11MD5:-8e4fc84e} ZS16_A11NE=${ZS16_A11NE:-8}
export ZS16_XB=${ZS16_XB:-2} ZS16_XREFUSE=${ZS16_XREFUSE:-5} ZS16_RFDECL=${ZS16_RFDECL:-2} ZS16_POST=${ZS16_POST:-10}
export ZS16_1U=${ZS16_1U:-0} ZS16_C4=${ZS16_C4:-0} ZS16_COLD=${ZS16_COLD:-0}   # zzzc16's speed / cold sections: not in the soak
export ZS16_STPMD5=${ZS16_STPMD5:-569fd59b} ZS16_DISKMIN_MB=${ZS16_DISKMIN_MB:-11000} ZS16_BUDGET_MIN=${ZS16_BUDGET_MIN:-47}
export ZS16_SIDECAR=${ZS16_SIDECAR:-dspark-dlw-q8_0.rad} ZS16_SIDESHA=${ZS16_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
export ZS16_CYCLES=${ZS16_CYCLES:-8} ZS16_CYCLES_MIN=${ZS16_CYCLES_MIN:-6} ZS16_CYCLE_S=${ZS16_CYCLE_S:-280} ZS16_WARM=${ZS16_WARM:-60}
export ZS16_LOADCONC=${ZS16_LOADCONC:-12} ZS16_PFTOK=${ZS16_PFTOK:-8192} ZS16_OFFSET=${ZS16_OFFSET:-0.5} ZS16_TELS=${ZS16_TELS:-5}
export ZS16_TJMAX=${ZS16_TJMAX:-100} ZS16_TMMAX=${ZS16_TMMAX:-96}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zsoak16/dsm-zsoak16.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xzs16: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
