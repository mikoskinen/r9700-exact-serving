#!/bin/bash
# 122-fn-x3rec.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~20 min, one arm) --
# Flash-Next EXL3 4.05 with RECORDED decode passes: the -fnx4 image = -fnx3 + qwen4exp_fp8.so / qwen4exp_x3.so of ds41-fnx
# c4e546a, the port of upstream main's issue #402 pass key (verify and draft passes keyed by their split cap; -fnx3's 1.17
# base answered 0, so 116's e4 issued every decode pass live: c16 window 0 played of 63712). The split laws the key
# relies on (rad_prefill_split.h, r4d_attn_prefill_split.h, r4d_attn_paged_h256_gqa6*.hip, r4d_decode_splits, the
# gated attention and MTP blocks) are byte-identical to main 451b136's; a wrong key would replay a wrong split count, so:
# BYTE IDENTITY against live issue: the warm-up answers and the 33k copy test (cold) against 116's e4 (the same line,
# every pass live) byte for byte (X93_COPYREF=e4, X93_REF=e4: no e4 re-run).
#   e4r  -fnx4, TP4 + EP4, --prefix-resume nearest (116's e4 line)
# = 93l-fn-tp4ab.sh beside this file with X93_SCREEN=1: gate, slow-state probe, mb c4 / c8 / c16 with engine stats
# (passes played vs issued live), the copy test, and -- recorded passes are what the AQL stamps time -- the c16 window's
# card-0 stampcat + gapcat + gapsize (X93_PROF16; every 20th pass): where EXL3 spends a decode step, kernels by class
# and the gaps. Compared with 116's e4 rows (mb, warm-up answers, copy).
# DRYRUN=1 bash 122-fn-x3rec.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${X122_ARMS:-"e4r"} X93_REF=${X93_REF:-e4} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-e4}
export X93_SCREEN=1 X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-122-fn-x3rec}
for f in "$H/93l-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93l-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 122-fn-x3rec: VIRHE 93l-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
