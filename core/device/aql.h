/* aql.h -- declarations shared by the AQL backend's translation units.
 *
 * aql_code.cpp reads code objects (offload bundle, ELF note, msgpack metadata), aql.cpp owns the
 * queues and all submission, and aql_blit.hip holds the backend's own copy and fill kernels.
 * aql_blit.hip is compiled as HIP, so this header must not pull in HSA.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace rad {
namespace aql {

/* ------------------------------------------------------------------ code objects */

/* One kernel argument as laid out in the code object's metadata. `hidden` names the implicit
 * arguments the compiler appends; explicit parameters map in order onto hipLaunchKernel's
 * `kernelParams`. */
enum class Hidden : uint8_t {
    None,           /* an explicit parameter */
    Pad,            /* hidden_none: bytes the compiler reserved and reads nothing from */
    BlockCountX, BlockCountY, BlockCountZ,
    GroupSizeX, GroupSizeY, GroupSizeZ,
    RemainderX, RemainderY, RemainderZ,
    GlobalOffsetX, GlobalOffsetY, GlobalOffsetZ,
    GridDims,
    DynamicLds,
    Unsupported,    /* anything else: a kernel that wants it is refused by name */
};

struct KernelArg {
    uint32_t offset = 0;
    uint32_t size   = 0;
    Hidden   hidden = Hidden::None;
    std::string kind;              /* the metadata's value_kind, for the refusal message */
};

struct KernelMeta {
    std::string symbol;            /* "<name>.kd" */
    uint32_t kernarg_size = 0;
    uint32_t group_size   = 0;     /* static LDS */
    uint32_t private_size = 0;     /* static scratch */
    bool     dynamic_stack = false;
    std::vector<KernelArg> args;
};

/* Selects the card's code object from a clang offload bundle, or fails with `*why` naming what
 * the bundle holds and what the card accepts. `isas` lists the card's accepted targets in the
 * runtime's order of preference ("amdgcn-amd-amdhsa--gfx1036", then the family's
 * "amdgcn-amd-amdhsa--gfx10-3-generic"), each with its modes ("gfx942:sramecc+:xnack-"). The
 * bundle has no total length, so it is walked entry by entry. Compressed bundles are refused;
 * nothing in this tree builds one. */
bool bundle_code_object(const void* bundle, const std::vector<std::string>& isas, const void** out,
                        size_t* out_size, std::string* why);

/* Every kernel described in an AMDGPU code object's metadata note. */
bool code_object_kernels(const void* elf, size_t size, std::vector<KernelMeta>* out,
                         std::string* why);

/* ------------------------------------------------------------------ fence policy
 *
 * Extra fence scope for the next dispatch on this thread, beyond the agent scope every kernel
 * gets. Set by the backend around its own blits; cleared by the dispatch that consumes it. */
/* kFenceAside (tp8 loop W1, lever WAITK's wait kernel): a packet that only schedules. It takes the agent-scope acquire
 * every live dispatch keeps, but not the acquire bookkeeping (the system acquire owed after a host wait or a lap stays
 * with the next real packet, which then gets exactly the scope it would have got without the aside packet), and it
 * never completes a pending rad_ride (the next real dispatch does, as without it). */
enum : uint8_t { kFenceAcquireSystem = 1, kFenceReleaseSystem = 2, kFenceAside = 4 };
void set_next_fence(uint8_t f);

/* ------------------------------------------------------------------ lever WAITK
 *
 * RADIANCE_AQL_WAITK (DRAFTER track round 49; reviewed + made exact-by-construction in tp8 loop W1): a wait whose
 * signal is still pending when it is enqueued gets a one-wave wait kernel (aql_blit.hip) on the waiting stream ahead of
 * its unchanged barrier-AND, so the packet processor never holds a pending barrier (one costs every other queue of its
 * CP pipe ~+40 us a packet, FINDINGS DRAFTER note 00:18 21.11.). The start value: unset / empty / 0 / negative / not a
 * number = off, 1 = event waits (rad_event_wait), 2 or more = event + ride waits (rad_ride_wait). In process: fn4 aql_waitk
 * (switch 106, tp8 loop W1s) 0 / 1 / 2 replaces the start value from the step it is applied at. */
inline int waitk_mode_of(const char* env) {
    if (!env || !*env) return 0;
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end == env || v <= 0) return 0;
    return v >= 2 ? 2 : 1;
}
/* What one wait does: kWaitkOff = only the barrier, as without the lever (the mode does not cover this kind of wait,
 * `need` 1 an event wait / 2 a ride wait, or this thread is recording a tape: a recorded wait kernel would poll a
 * retired signal at play, and a played tape's waits go through rad_event_wait again, which adds the kernel then);
 * kWaitkMet = only the barrier (the signal `value` read at enqueue is 0 already); kWaitkKernel = the wait kernel, then
 * the barrier. */
enum : int { kWaitkOff = 0, kWaitkMet = 1, kWaitkKernel = 2 };
inline int waitk_plan(int mode, int need, bool capturing, int64_t value) {
    if (mode < need || capturing) return kWaitkOff;
    return value == 0 ? kWaitkMet : kWaitkKernel;
}
/* Since start: wait kernels issued, waits met at enqueue, waits the mode covers but a capture kept plain. */
struct WaitkCounts { int64_t kernels = 0, met = 0, captured = 0; };
WaitkCounts waitk_counts();

/* ------------------------------------------------------------------ test hooks */

/* Whether the signal behind a rad_ride token could be handed out again now (tests/device_test.cpp). */
bool ride_reusable(void* token);
/* tests/device_test.cpp: replaces RADIANCE_AQL_WAITK's start value for this process; -1 = the start value again. */
void waitk_set_for_test(int mode);
/* The mode the next wait runs at: the test value, else fn4 aql_waitk once set (tp8 loop W1s), else RADIANCE_AQL_WAITK's start value
 * (0 in a build without the AQL backend). */
int waitk_mode_now();

}  // namespace aql
}  // namespace rad

struct RadCopyRange;

/* Blit kernel launchers, defined in aql_blit.hip. Each issues one kernel onto `stream` through
 * hipLaunchKernel, which the backend intercepts. The caller guarantees alignment: the 16-byte
 * forms need both addresses and the length 16-byte aligned. */
extern "C" {
/* `link`: either side is pinned host memory, so use one narrow workgroup instead of a full grid
 * (see aql_blit.hip). */
int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream, bool link = false);
int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                        uint64_t width, uint64_t height, void* stream);
int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream);
/* Lever WAITK: one wave polls the int64 at `value` (an amd_signal_t's value) until it reads 0. */
int rad_aql_blit_wait(const void* value, void* stream);
/* rad_memcpy_ranges_async's kernel: `prefix` a multiple of 16, both bases 16-byte aligned. */
int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix, const RadCopyRange* ranges,
                        int n, void* stream);
}
