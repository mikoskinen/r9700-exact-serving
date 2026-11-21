// r4d_ika_x.h -- lever CIK-X (tp8 loop R49c, PREFILL round 49, 21.11.2026): R4D_IKA_ENTRY_X() is the entry of the IKA families whose
// CHAIN-compiled body changes beyond the entry itself (PREFILL NOTE 13:20 21.11.: the chain wait's atomic poll is a memory clobber to
// LLVM's uniform-load analysis, so loads of pointers read from memory after it stop being scalar -- more vector loads, VGPRs,
// spills; tp8/devpf/r49/cgscan.py). In a CIK-X build (CMake R4D_IKA_CHAIN_X, needs R4D_IKA_CHAIN; set on the files that include
// this header) they keep IKA's plain entry and r4d_ika_nochain_kernels() names them (r4d_ika_nochain.inc), so the engine never
// makes them chain-capable (core/device/aql.cpp: cikcap false = issued / played as plain IKA kernels, ordered). Every other build:
// R4D_IKA_ENTRY() exactly, and the list is empty. The list and the macro's uses must name the same kernels: a plain-compiled kernel
// the engine thought chain-capable would run unordered without waiting (tp8/devpf/r49/cikxcheck.py checks it).
#pragma once
#include "r4d_ika.h"
#ifndef R4D_IKA_CHAIN_X
#define R4D_IKA_CHAIN_X 0
#endif
#if R4D_IKA && R4D_IKA_CHAIN && R4D_IKA_CHAIN_X && defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
#define R4D_IKA_ENTRY_X() __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); const R4dIkaRel r4d_ika_rel_ {}; (void)r4d_ika_rel_
#else
#define R4D_IKA_ENTRY_X() R4D_IKA_ENTRY()
#endif
