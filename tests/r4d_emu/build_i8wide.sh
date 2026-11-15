#!/bin/bash
# build_i8wide.sh SRC OUT -- the fn4 BI wide emulator test (test_i8wide.cpp) by hand, with the r4d_emu flags
# (tests/CMakeLists.txt): libr4d's r4d_gemm_fp8a8.hip compiled as C++ on the SIMT emulator.
set -eu
S=$1; O=$2; mkdir -p $O
CXX=${EMU_CXX:-/opt/rocm-repeatable/lib/llvm/bin/clang++}
F="-std=c++20 -O2 -ffp-contract=off -Wno-psabi -I$S/tests/r4d_emu/emu -I$S/tests/r4d_emu -I$S/abi -I$S/libr4d"
$CXX $F -x c++ -c $S/libr4d/r4d_gemm_fp8a8.hip -o $O/gemm_fp8a8.o &
$CXX $F -c $S/tests/r4d_emu/emu/emu.cpp -o $O/emu.o &
$CXX $F -c $S/tests/r4d_emu/cfg_stub.cpp -o $O/cfg_stub.o &
$CXX $F -c $S/tests/r4d_emu/test_i8wide.cpp -o $O/test_i8wide.o &
wait
$CXX $O/test_i8wide.o $O/gemm_fp8a8.o $O/emu.o $O/cfg_stub.o -ldl -o $O/r4d_emu_i8wide
echo built $O/r4d_emu_i8wide
