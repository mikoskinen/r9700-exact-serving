#!/bin/bash
# 126-fn-mainpf.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~70 min, two arms)
# -- the pipelined prefill step on the MAIN line, to make it part of the main-line base (it is meant for production).
# So far measured on 1.15.1 only (99 / 106: m4n vs k4n mb c4/c8/c16 536.5/739.3/953.0 vs 525.5/738.5/953.7; prefill
# 16k / 128k +6.9 / +9.8 %; cc TTFT vs k4sf 0.828 vs 0.849, forced mode).
#   mpA  radiance-local:c16-sw (main 451b136 + chain_pays 0 + the fn4 fixes) with RADIANCE_FN4_HC_NORMWAVE=1 (F1n, the
#        current best stack), --prefill-interleave 0 --prefix-resume nearest, MNBT 2048
#   mpB  the same + MNBT 4096 + --prefill-span 2
# = 93n-fn-tp4ab.sh beside this file with X93_SCREEN_PF=1: gate, slow-state probe, mb c4 / c8 / c16 with engine stats,
# prefill 16k + fresh 64k / 128k, mb pf, the 33k copy test (mpB byte for byte against mpA: X93_COPYREF), the c16
# card-0 profile; every restart line records the KV pool and VRAM headroom (MNBT 4096's larger activation arena).
# No automatic stop rule (TESTING.md): both arms run whole. The cc pair (reflen mode) is 126b, queued after 123.
# DRYRUN=1 bash 126-fn-mainpf.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${X126_ARMS:-"mpA mpB"} X93_REF=${X93_REF:-mpA} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-mpA}
export X93_SCREEN_PF=1 X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-126-fn-mainpf}
for f in "$H/93n-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93n-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 126-fn-mainpf: VIRHE 93n-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
