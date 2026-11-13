#!/bin/bash
# 57-p4zyzzzzz-ds-f2-span4.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~25 min) -- prefill lever S4: a lone prefill step of 8192 rows instead of 4096 (--prefill-span 4 at ckint
# 2048, MNBT 8256; beside decode rows the shared budget stays max(2048, 8256 - 3 x 2048) = 2112, as today). Each EXL3 expert
# is then read once per 8192 prompt rows instead of per 4096 (the 136k profile: experts 22 % of a chunk). Exact by the same
# per-row launch rules as lever 7 (dense > 48 rows per row, norms a workgroup a row, EXL3 map form, exact-wire AR per element)
# -- checked here byte for byte against f-pipe's serial arm A (same image, same day: no baseline re-run). History: DS never ran
# 8192-row prefill steps (6.11. MNBT 8192 still cut at ckint 2048 x span 2 = 4096); Flash-Next's analog (P2) was not exact
# only because of its hc split-K / ti8 wire, which DS does not use. One arm S on the f-pipe image with RADIANCE_DS41_PIPE unset:
# gate, modes, warm-up answers, mb c4 / c8 (decode: the larger arena costs expert VRAM), fresh 16k / 64k / 136k, mb pf, the
# copy, the 136k deep answer; then S vs A (% and identity). 10.11. + arm SP = S + RADIANCE_DS41_PIPE=1 (lever 7 lost 11-13 % at
# span 2 with pieces of 2048; at span 4 its pieces are 4096 = the MoE pass size today): SP vs S and SP vs f-pipe B. ~45 min.
# Runs after 57-p4zyzzzzz-ds-f-pipe (needs its pipe-A and pipe-B files).
# DRYRUN=1 bash ... : prints, runs and writes nothing.
export XPIPE_IMAGE=${XPIPE_IMAGE:-radiance-local:ds41-pipe-fd215624-5c13e555-l} XSPAN_ARMS=${XSPAN_ARMS:-S SP}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-span4.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpipe: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || [ -s $HOME/gpu/ds41/exl3-test/runs/xpipe/copy-pipe-A-i1-run1.txt ] || { echo "$(date '+%F %T') RESULT xpipe f2-span4: VIRHE no f-pipe arm A files in runs/xpipe -- not starting" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
