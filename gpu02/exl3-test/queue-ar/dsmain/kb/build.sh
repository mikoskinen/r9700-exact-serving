#!/bin/bash
# build.sh [ix|m1|all] (12.11.2026, ds-main-port kbench builder) -- builds kb/bin/kb2_ix and kb/bin/kb2_m1 on THIS host's CPU in
# radiance-ds41-build:main-e736c019 (the toolchain the ds41-int7 image's libr4d was built with: /opt/rocm-repeatable clang++,
# libr4d's unit flags -O3 -std=gnu++17 --offload-arch=gfx1201 -ffp-contract=off -g0), no GPU, no network. Sources: kb/src
# (ds41-int7 0391a6e libr4d + abi, copied verbatim; ORIGIN.txt, SRC-MD5SUMS) and kb/bench. Then MD5SUMS.
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
WHAT=${1:-all}
IMG=radiance-ds41-build:main-e736c019
timeout 1800 nice -n 19 docker run --rm --network none --cpus ${CPUS:-6} --memory 10g -e WHAT=$WHAT -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) -v $K:/kb --entrypoint bash $IMG -c '
set -e
CXX=/opt/rocm-repeatable/lib/llvm/bin/clang++
FL="-x hip --rocm-path=/opt/rocm -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1201 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-but-set-variable -ffp-contract=off -g0 --no-offload-compress -D__HIP_PLATFORM_AMD__=1 -DRAD_HAVE_HIP=1 -I/kb/src/libr4d -I/kb/src/abi -I/kb/bench -I/build/rad_include"
LK="--rocm-path=/opt/rocm --hip-link -O3 -ldl"
cd /tmp
for t in ix m1; do
  [ "$WHAT" = all ] || [ "$WHAT" = $t ] || continue
  echo "[build] $(date +%T) kb2_$t"
  $CXX $FL -c /kb/bench/kb2_$t.hip -o kb2_$t.o 2>&1 | grep -E "error|warning" | grep -v "Wunused" | head -40 || true
  [ -s kb2_$t.o ] || { echo "[build] kb2_$t.o missing"; exit 1; }
  $CXX $LK kb2_$t.o -o /kb/bin/kb2_$t
  echo "[build] $(date +%T) kb2_$t linked"
done
cd /kb/bin && md5sum kb2_* > MD5SUMS
chown -R $HOST_UID:$HOST_GID /kb/bin
'
cat $K/bin/MD5SUMS
