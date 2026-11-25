#!/bin/bash
# 124-fn-x3pf.sh (queue item, 9.11.2026, queue-ar; EXL3 track E2; gpu02, all 4 cards, every radiance container DOWN;
# ~30 min, one arm) -- the pipelined prefill step on the Flash-Next EXL3 4.05 line: flags only.
#   e4rP  122's e4r (-fnx4: recorded passes, TP4 + EP4, --prefix-resume nearest) + MNBT 4096 + --prefill-span 2 (the
#         iq4r line's CANDIDATE: prefill 16k / 128k +6.9 / +9.8 %, decode -0..-2 %, exact; 99 / 106)
# TARGET METRIC FIRST (TESTING.md 9.11. loop): prefill 16k warm + fresh 64k / 128k and mb pf (31k beside 4 decoders);
# the decode screen beside it (gate, slow-state probe, mb c4 / c8 / c16 with engine stats) to show what it costs there.
# IDENTITY: the warm-up answers and the 33k copy test byte for byte against 122's e4r (X93_REF / X93_COPYREF = e4r, no
# re-run; the pipelined step is exact on the iq4r line). The c16 window's card-0 profile stays on (X93_PROF16).
# = 93m-fn-tp4ab.sh beside this file with X93_SCREEN_PF=1.
# DRYRUN=1 bash 124-fn-x3pf.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${X124_ARMS:-"e4rP"} X93_REF=${X93_REF:-e4r} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-e4r}
export X93_SCREEN_PF=1 X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-124-fn-x3pf}
for f in "$H/93m-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93m-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 124-fn-x3pf: VIRHE 93m-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
