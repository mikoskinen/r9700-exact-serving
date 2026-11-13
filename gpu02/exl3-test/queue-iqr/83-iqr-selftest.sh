#!/bin/bash
# 83-iqr-selftest.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN) -- moe_gemm_iqr (libr4d's routed
# MoE over IQR trellis experts, r4d_moe_iqr.hip: decode GEMV + prefill tile) against the int8 dot of the same decoded values,
# FIRST at DeepSeek-V4.1's shapes, before any IQR container is served: r4d_selftest --case moe_gemm_iqr_ds41 (branch ds41-iqrmoe
# 4d0e7a5: gate_up [4608 x 5120] u2 L15 / u3 L16 on 512-row runs, down [5120 x 2304] u2 L15 / u3 L16 on 256-row runs and u4 L16
# on 1024; M 1 / 9 / 64 = one token, a run crossing the GEMV's 8-token group, the prefill tile; two experts, top-2), then baKon's
# own small cases (--case moe_gemm_iqr: M 2/9 GEMV, M 64/40/32 tile) and the dense IQR rows (--case gemm_iqr) as the control.
# The libr4d.so is the iqrmoe image's own (b77f7a85: moe_gemm_iqr unchanged from 9c7ce498); the selftest binary is mounted.
# Tolerance: |y - ref| <= 2^-7 |ref| + 2^-8 rms (the case's own rule), FAIL lines name the first mismatch.
# Inputs: ../queue-iqr/iqr-bin/r4d_selftest (md5 59946c0b...). DRYRUN=1 prints only. Estimate ~10 min (host references
# at M=64: ~5 s a case on the EPYC).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
for LIB in "$H/lib-iqr.sh" "$H/../lib-iqr.sh" "$HOME/gpu/ds41/exl3-test/queue-iqr/lib-iqr.sh"; do [ -e "$LIB" ] && break; done
. "$LIB" || { echo "lib-iqr.sh not found"; exit 1; }
B=$KIT/iqr-bin; CARD=${CARD:-0}; OS=$O/selftest
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
guards iqr-selftest || exit 1
if [ "$DRY" != 1 ]; then
  [ -x $B/r4d_selftest ] || { log "RESULT iqr-selftest: VIRHE $B/r4d_selftest missing"; exit 1; }
  docker image inspect $IQRIMAGE > /dev/null 2>&1 || { log "RESULT iqr-selftest: VIRHE image $IQRIMAGE missing"; exit 1; }
fi
mkd $OS
log "RESULT iqr-selftest: start $IQRIMAGE card $CARD selftest $(md5sum $B/r4d_selftest 2>/dev/null | cut -c1-8)"
for c in moe_gemm_iqr_ds41 moe_gemm_iqr gemm_iqr; do
  drun iqr-selftest-$c 1800 $OS/$c.txt $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest $IQRIMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c
  [ "$DRY" = 1 ] && continue
  okn=$(grep -c '^  ok ' $OS/$c.txt); fl=$(grep -c 'FAIL' $OS/$c.txt)
  log "RESULT iqr-selftest $c: $okn ok, $fl FAIL$( [ $fl -gt 0 ] && echo ": $(grep -m3 FAIL $OS/$c.txt | tr '\n' ' ' | cut -c1-500)")"
done
log "RESULT iqr-selftest: done -> $OS"
