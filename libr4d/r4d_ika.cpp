// r4d_ika.cpp -- lever IKA's list (r4d_ika.h): the kernels of this libr4d that acquire and release themselves. The engine reads it
// through dlsym on this object (core/device/aql.cpp ika_kernel) and matches an identifier as the mangled name's <length><name>.
// Empty unless built with R4D_IKA (then every kernel ika_apply.py converted, r4d_ika_list.inc).
#ifndef R4D_IKA
#define R4D_IKA 0
#endif
namespace {
const char* const kIka[] = {
#if R4D_IKA
#include "r4d_ika_list.inc"
#endif
    nullptr};
}  // namespace
extern "C" __attribute__((visibility("default"))) const char* const* r4d_ika_kernels(int* n) {
    if (n) *n = (int)(sizeof(kIka) / sizeof(kIka[0])) - 1;
    return kIka;
}
/* Lever CIK (DRAFTER track round 62): 1 when every listed kernel's entry is chain-capable (R4D_IKA_CHAIN, r4d_ika.h); the engine
 * (RADIANCE_AQL_CIK=1) chains this object's listed kernels only then (aql.cpp cik_module). */
#ifndef R4D_IKA_CHAIN
#define R4D_IKA_CHAIN 0
#endif
extern "C" __attribute__((visibility("default"))) int r4d_ika_chain(void) { return (R4D_IKA && R4D_IKA_CHAIN) ? 1 : 0; }
/* Lever CIK-X (tp8 loop R49c): the listed kernels whose entry is IKA's plain one even in this CHAIN build (r4d_ika_x.h
 * R4D_IKA_ENTRY_X, r4d_ika_nochain.inc); the engine keeps them off the chain (aql.cpp cikcap). Empty unless built with
 * R4D_IKA + R4D_IKA_CHAIN + R4D_IKA_CHAIN_X. */
#ifndef R4D_IKA_CHAIN_X
#define R4D_IKA_CHAIN_X 0
#endif
namespace {
const char* const kIkaNoChain[] = {
#if R4D_IKA && R4D_IKA_CHAIN && R4D_IKA_CHAIN_X
#include "r4d_ika_nochain.inc"
#endif
    nullptr};
}  // namespace
extern "C" __attribute__((visibility("default"))) const char* const* r4d_ika_nochain_kernels(int* n) {
    if (n) *n = (int)(sizeof(kIkaNoChain) / sizeof(kIkaNoChain[0])) - 1;
    return kIkaNoChain;
}
