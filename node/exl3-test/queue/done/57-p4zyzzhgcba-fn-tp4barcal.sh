#!/bin/bash
# 57-p4zyzzhgcba wrapper (9.11.2026): barrier-cost calibration before any launch-fusion build -- 94 (cbf980a4) with arms
# d4a (unchanged) and o4a (the barrier bit on every played dispatch, ~275 more dependent launches a step), stamps off.
# Delta step ms / ~275 = the unstamped cost of one dependent launch (115 measured ~4.4 us stamped). ~25 min.
export ARMS="d4a o4a" X91_STAMP_EVERY=1000000 X91_CARDS=""
exec bash $HOME/gpu/ds41/exl3-test/queue-ar/94-fn-tp4noovl.sh
