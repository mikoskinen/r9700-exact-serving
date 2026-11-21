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
