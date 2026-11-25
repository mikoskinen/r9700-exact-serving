#!/bin/bash
# build.sh (12.11.2026, ds-main-port arw builder) -- builds arw/bin/arw on THIS host's CPU in radiance-ds41-build:main-e736c019
# (the toolchain the ds41-int9 image's libr4d was built with: /opt/rocm-repeatable clang++, libr4d's unit flags -O3 -std=gnu++17
# --offload-arch=gfx1201 -ffp-contract=off -g0), no GPU, no network. Sources: arw/bench/arw.hip and arw/src/abi (ds41-int9 0cb6fd3
# abi/rad_abi.h, rad_types.h, rad_encoding.h, copied verbatim; ORIGIN.txt, SRC-MD5SUMS). Linked -rdynamic: the image's libr4d.so binds
# its PLT calls to r4d_ar_twoshot_4rank_exact / _coded / r4d_ar_twoshot_coded_ok to the harness's wrappers (arw.hip's header).
# No libr4d at link time (dlopen at run time). Then bin/MD5SUMS.
set -euo pipefail
A=$(cd "$(dirname "$0")" && pwd)
IMG=radiance-ds41-build:main-e736c019
timeout 1200 nice -n 19 docker run --rm --network none --cpus ${CPUS:-6} --memory 8g -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -v $A:/arw --entrypoint bash $IMG -c '
set -e
CXX=/opt/rocm-repeatable/lib/llvm/bin/clang++
FL="-x hip --rocm-path=/opt/rocm -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1201 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -ffp-contract=off -g0 --no-offload-compress -D__HIP_PLATFORM_AMD__=1 -I/arw/src/abi"
LK="--rocm-path=/opt/rocm --hip-link -O3 -ldl -lpthread -rdynamic"
cd /tmp
echo "[build] $(date +%T) arw"
$CXX $FL -c /arw/bench/arw.hip -o arw.o 2>&1 | grep -E "error|warning" | head -60 || true
[ -s arw.o ] || { echo "[build] arw.o missing"; exit 1; }
$CXX $LK arw.o -o /arw/bin/arw
echo "[build] $(date +%T) arw linked"
cd /arw/bin && md5sum arw > MD5SUMS
chown -R $HOST_UID:$HOST_GID /arw/bin
'
cat $A/bin/MD5SUMS
