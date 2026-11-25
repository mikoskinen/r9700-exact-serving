#!/bin/bash
# 6.11. 10.05: prof0 was skipped (guard 1.85 x 19.56 = 36.2 ms; profiled device op 48.6 / 47.9 ms on both tries = consistent
# EXL3 profiling overhead ~2.5x, not the random slow state, which would be ~4x). Re-run prof0 + profpf with PROF_GUARD_X 3.0.
cd ~/gpu/ds41/exl3-test && STEPS=prof0,profpf PROF_GUARD_X=3.0 BUDGET_MIN=60 ./after-sweep.sh
