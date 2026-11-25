# env.sh -- the one place for the window's variables (sourced by every script here). Edit IMAGE/MODEL after the build.
T=$HOME/gpu/ds41/exl3-test                 # this kit on gpu02
R=$T/runs                                  # outputs
L=$R/ds41-exl3.log                         # RESULT lines (one per step, like ~/gpu/ds41/runs/ds41-1103.log)
C=$HOME/radiance-1.15.1/compose            # compose project dir (.env lives here; production files too)
D=$HOME/gpu/ds41                           # the 3.11. harness (ds41_tests.py, fastcheck.py, stepprobe.py, mb.py, onem.py, mdelta.py)
Q=$HOME/gpu/ds41/runs/q                    # produp.sh, prodfc.py, evict.py, qual/qneedle.py
export IMAGE=${IMAGE:-radiance-local:ds41-exl3-9e8ccb64-3684c0b1}
export MODEL=${MODEL:-deepseek-v4.1-flash-1017-exl3-2.25bpw.rad}
MODELDIR=$HOME/models/ds41-exl3
# 3.11. confirm config ("fin-best"): NST 0, checkpoint 2048 + prefill span 2, interleave 8, MAXSEQS 8, RAM KV tier 84 GiB, ngram ram,
# TP4 exact, expert_tiered. Host pool raised for the EXL3 experts (~33.9 GiB/rank Stage A, ~35.7 full vs ~28 UD-IQ1_S).
export HOSTPOOL=${HOSTPOOL:-19456} MAXSEQS=${MAXSEQS:-8} NST=${NST:-0} CKINT=${CKINT:-2048} MNBT=${MNBT:-8192} KVD=${KVD:-bf16}
export WIRE=${WIRE:-exact} MML=${MML:-262144}
RAMTIER="--prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all"
BEST2="--prefill-span 2 --prefill-interleave 8"
export EXTRA=${EXTRA-$RAMTIER} EXTRA2=${EXTRA2-$BEST2}
GUARD_MS=${GUARD_MS:-21}                   # 1-user NST0 gate: UD-IQ1_S fast 15-16.5 ms, EXL3 expected 17-19.5 ms
PROD_GATE_MS=21                            # produp.sh's own gate (prodfc.py), production fast ~15.0 ms
MEMFREE_MIN_GB=${MEMFREE_MIN_GB:-150}      # after production down + evict
FREE_MIN_GB=${FREE_MIN_GB:-100}            # 'available' while production runs (copy steps)
URL=http://127.0.0.1:8090
mkdir -p $R
