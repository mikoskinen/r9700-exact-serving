/* r4d_ar_sdma.hip -- fn4 SDMA-AR (12.11.2026): the exact four-rank (any world <= 8) two-shot all-reduce with the
 * peer-to-peer bytes moved by each card's SDMA copy engine and the compute units doing only the fp32 fold.
 *
 * Why: after the riser move (202) the exact two-shot moves 17.96 GB/s a rank a direction at 42 MB while one card's link
 * carries 24-26 GB/s; its reduce-scatter hop (CU stores pushed into the peers' fine-grained scratch, a system-scope
 * fence a wave) runs at 14.4 GB/s, 1.5-1.7x the all-gather hop for the same bytes. A copy engine writes at the link
 * rate without compute units and without per-wave fences: its packets complete in order, so one FENCE packet after a
 * copy publishes it.
 *
 * The protocol (rank r, launch value v = ++seq; flags are words of r's own uncached VRAM, one 64-byte line each):
 *   stream  k_flag:  global_wb (system) then go1 = v          (the input, written through L2 by earlier kernels, is in VRAM)
 *   SDMA    POLL go1 == v; per peer p (ring order from r + 1): COPY in[shard p] -> p's scratch A, slot r; FENCE p.A[r] = v
 *   stream  k_wait:  A[p] == v for every peer p, then global_inv (system)
 *   stream  k_fold:  out[shard r] = bf16(fp32 0 + x_0 + x_1 + ... + x_{W-1})   (ascending rank: own input, slots A[p]),
 *                    every block global_wb (system), the last block go2 = v
 *   SDMA    POLL go2 == v; per peer p: COPY out[shard r] -> p's scratch B, slot r; FENCE p.B[r] = v; FENCE done = v
 *   stream  k_wait:  B[p] == v for every peer, done == v, then global_inv (system)
 *   stream  k_assemble: out[shard p] = B slot p for every peer p
 * The fold is the CU two-shot's (r4d_ar_twoshot_Nrank_exact.hip reduce2: acc = 0.0f, += r4d_ar_to_f of rank 0..W-1,
 * r4d_ar_from_f) on the same even shard table (r4d_ar_shards.h), so every output byte equals today's exact two-shot's.
 * The scratch is written only by peers' copy engines and read only after the wait's invalidate; no compute unit of this
 * card writes it, so no dirty L2 line can land over a copied word. `out` is written only by this card's kernels.
 * Reuse: a rank's next launch sends its first hop only after its stream passed this launch's B wait, i.e. after every
 * owner's fold of this launch read the slots; so one slot set suffices (no parity).
 *
 * The queue is the driver's user-mode SDMA queue (AMDKFD_IOC_CREATE_QUEUE), as core/device/aql.cpp's DMA lane opens it
 * (the same ring / write-pointer rules, packet formats and list-number room check), opened here per (card, engine) so
 * the selftest and a plugin-side all-reduce can drive it without the engine's backend.
 *
 * Test entry points (r4d_selftest --perfsdma / --sdmaar4) are extern "C"; nothing in the engine calls them yet. */
#include "rad_abi.h"
#include "r4d.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_shards.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <dirent.h>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {

/* ------------------------------------------------------------------ the KFD SDMA queue */

constexpr uint32_t kSdmaCopy    = 1;           /* COPY, sub-op LINEAR */
constexpr uint32_t kSdmaFence   = 5;
constexpr uint32_t kSdmaPoll    = 8;           /* POLL_REGMEM */
constexpr uint64_t kSdmaMaxCopy = 0x3fffe0;    /* the largest count a linear copy carries */
constexpr uint64_t kRingBytes   = (uint64_t)1 << 20;
constexpr size_t   kDoorbellPage = 8192;
constexpr uint32_t kEnds        = 4096;

struct KfdCreateQueue {
    uint64_t ring_base_address, write_pointer_address, read_pointer_address, doorbell_offset;
    uint32_t ring_size, gpu_id, queue_type, queue_percentage, queue_priority, queue_id;
    uint64_t eop_buffer_address, eop_buffer_size, ctx_save_restore_address;
    uint32_t ctx_save_restore_size, ctl_stack_size, sdma_engine_id, metadata_ring_size;
};
static_assert(sizeof(KfdCreateQueue) == 96, "the KFD 1.22 create-queue layout");
constexpr uint32_t kKfdQueueSdmaByEngine = 0x4;
constexpr unsigned long kKfdCreateQueue = _IOWR(AMDKFD_IOCTL_BASE, 0x02, KfdCreateQueue);

std::once_flag g_kfd_once;
int g_kfd = -1;
int kfd() {
    std::call_once(g_kfd_once, [] { g_kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC); });
    return g_kfd;
}

std::string sfmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string sfmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

/* The driver's gpu_id and render minor of HIP device `dev`, from the topology node whose PCI location matches. */
bool topo_find(int dev, uint32_t* gpu_id, int* minor, std::string* why) {
    char bus[64] = { 0 };
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess ||
        std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) {
        *why = sfmt("no PCI address for HIP device %d", dev);
        return false;
    }
    const unsigned long long want = (b << 8) | (d << 3) | f;
    for (int n = 0; n < 256; ++n) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%d/properties", n);
        FILE* fp = std::fopen(path, "r");
        if (!fp) { if (n > 0) break; continue; }
        char key[128];
        unsigned long long v = 0, loc = ~0ull, domain = ~0ull, mn = ~0ull;
        while (std::fscanf(fp, "%127s %llu", key, &v) == 2) {
            if (!std::strcmp(key, "location_id")) loc = v;
            else if (!std::strcmp(key, "domain")) domain = v;
            else if (!std::strcmp(key, "drm_render_minor")) mn = v;
        }
        std::fclose(fp);
        if (loc != want || domain != dom) continue;
        std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", n);
        fp = std::fopen(path, "r");
        unsigned id = 0;
        const bool got = fp && std::fscanf(fp, "%u", &id) == 1;
        if (fp) std::fclose(fp);
        if (!got || !id || mn == ~0ull) break;
        *gpu_id = id;
        *minor = (int)mn;
        return true;
    }
    *why = sfmt("no KFD topology node for %s", bus);
    return false;
}

struct Bo {
    char* va = nullptr;
    size_t size = 0;
    uint64_t handle = 0;
    char* res = nullptr;
    size_t res_size = 0;
};

void bo_free(uint32_t gpu_id, Bo* b) {
    if (b->handle) {
        uint32_t ids[1] = { gpu_id };
        kfd_ioctl_unmap_memory_from_gpu_args u{};
        u.handle = b->handle;
        u.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
        u.n_devices = 1;
        ioctl(kfd(), AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &u);
        kfd_ioctl_free_memory_of_gpu_args fr{};
        fr.handle = b->handle;
        ioctl(kfd(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
    }
    if (b->res) munmap(b->res, b->res_size);
    *b = Bo{};
}

/* A GTT buffer of card `gpu_id` at one address for the CPU and the card (aql.cpp dma_bo). */
bool bo_alloc(uint32_t gpu_id, int minor, size_t size, Bo* b, std::string* why) {
    const size_t al = (size_t)2 << 20;
    b->res_size = size + al;
    void* r = mmap(nullptr, b->res_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) { b->res = nullptr; *why = "no address range"; return false; }
    b->res = (char*)r;
    b->va = (char*)(((uintptr_t)r + al - 1) & ~(uintptr_t)(al - 1));
    b->size = size;
    kfd_ioctl_alloc_memory_of_gpu_args a{};
    a.va_addr = (uint64_t)(uintptr_t)b->va;
    a.size = size;
    a.gpu_id = gpu_id;
    a.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
              KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
    if (ioctl(kfd(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a) < 0) {
        *why = sfmt("the driver refused a %zu-byte GTT buffer: %s", size, std::strerror(errno));
        bo_free(gpu_id, b);
        return false;
    }
    b->handle = a.handle;
    uint32_t ids[1] = { gpu_id };
    kfd_ioctl_map_memory_to_gpu_args m{};
    m.handle = a.handle;
    m.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
    m.n_devices = 1;
    if (ioctl(kfd(), AMDKFD_IOC_MAP_MEMORY_TO_GPU, &m) < 0) {
        *why = sfmt("the driver would not map a GTT buffer: %s", std::strerror(errno));
        bo_free(gpu_id, b);
        return false;
    }
    char want[64];
    std::snprintf(want, sizeof want, "/dev/dri/renderD%d", minor);
    if (DIR* d = opendir("/proc/self/fd")) {
        while (dirent* e = readdir(d)) {
            char link[300], tgt[128];
            std::snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
            const ssize_t n = readlink(link, tgt, sizeof tgt - 1);
            if (n <= 0) continue;
            tgt[n] = 0;
            if (std::strcmp(tgt, want)) continue;
            const int fd = std::atoi(e->d_name);
            if (mmap(b->va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, a.mmap_offset) != MAP_FAILED) {
                closedir(d);
                return true;
            }
        }
        closedir(d);
    }
    *why = sfmt("no open %s maps the card's buffers", want);
    bo_free(gpu_id, b);
    return false;
}

struct SdmaQ {
    std::mutex mu;
    int dev = -1, engine = 0;
    uint32_t gpu_id = 0, qid = 0;
    Bo ring, ptrs;
    uint32_t* rb = nullptr;
    volatile uint64_t* rptr = nullptr;
    volatile uint64_t* wptr = nullptr;
    volatile uint64_t* db = nullptr;
    void* db_page = nullptr;
    volatile uint32_t* fin = nullptr;   /* the number of the last finished list (a word of `ptrs`) */
    uint64_t w = 0, seen = 0;
    uint32_t lists = 0;
    uint64_t ends[kEnds] = { 0 };
};

uint64_t q_read(const SdmaQ* q) {
    uint64_t r = *q->rptr;
    const uint32_t k = *q->fin;
    if (k && q->lists - k < kEnds) r = std::max(r, q->ends[k % kEnds]);
    return std::max(r, q->seen);
}

int q_put(SdmaQ* q, const uint32_t* dw, uint64_t n) {
    q->seen = q_read(q);
    const uint64_t cap = kRingBytes / 4;
    uint64_t at = (q->w / 4) & (cap - 1);
    const uint64_t pad = at + n > cap ? cap - at : 0;
    const uint64_t need = (pad + n) * 4;
    if (q->w + need - q->seen > kRingBytes) {
        const auto t0 = std::chrono::steady_clock::now();
        while (q->w + need - (q->seen = q_read(q)) > kRingBytes) {
            _mm_pause();
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) return RAD_E_DEVICE;
        }
    }
    for (uint64_t i = 0; i < pad; ++i) q->rb[at + i] = 0;
    q->w += pad * 4;
    at = (q->w / 4) & (cap - 1);
    for (uint64_t i = 0; i < n; ++i) q->rb[at + i] = dw[i];
    q->w += n * 4;
    return RAD_OK;
}

int q_poll(SdmaQ* q, const void* addr, uint32_t v) {
    const uint64_t a = (uint64_t)(uintptr_t)addr;
    const uint32_t p[6] = { kSdmaPoll | (3u << 28) /* equal */ | (1u << 31) /* memory */, (uint32_t)a, (uint32_t)(a >> 32),
                            v, 0xffffffffu, 10u | (0xfffu << 16) /* retry forever */ };
    return q_put(q, p, 6);
}

int q_copy(SdmaQ* q, void* dst, const void* src, uint64_t bytes) {
    int rc = RAD_OK;
    for (uint64_t o = 0; rc == RAD_OK && o < bytes; o += kSdmaMaxCopy) {
        const uint64_t len = std::min(kSdmaMaxCopy, bytes - o);
        const uint64_t sa = (uint64_t)(uintptr_t)src + o, da = (uint64_t)(uintptr_t)dst + o;
        const uint32_t cp[7] = { kSdmaCopy, (uint32_t)(len - 1), 0u, (uint32_t)sa, (uint32_t)(sa >> 32), (uint32_t)da,
                                 (uint32_t)(da >> 32) };
        rc = q_put(q, cp, 7);
    }
    return rc;
}

int q_fence(SdmaQ* q, void* addr, uint32_t v) {
    const uint64_t a = (uint64_t)(uintptr_t)addr;
    const uint32_t f[4] = { kSdmaFence | (3u << 16) /* uncached */, (uint32_t)a, (uint32_t)(a >> 32), v };
    return q_put(q, f, 4);
}

/* Ends the list with its number (the room check's lower bound on what the engine read) and rings the doorbell. */
int q_kick(SdmaQ* q) {
    const uint32_t k = q->lists + 1;
    int rc = q_fence(q, (void*)q->fin, k);
    if (rc == RAD_OK) {
        q->ends[k % kEnds] = q->w;
        q->lists = k;
    }
    _mm_sfence();
    *q->wptr = q->w;
    _mm_sfence();
    *q->db = q->w;
    return rc;
}

void q_close(SdmaQ* q) {
    if (!q) return;
    if (q->qid || q->db_page) {
        kfd_ioctl_destroy_queue_args d{};
        d.queue_id = q->qid;
        ioctl(kfd(), AMDKFD_IOC_DESTROY_QUEUE, &d);
    }
    if (q->db_page) munmap(q->db_page, kDoorbellPage);
    bo_free(q->gpu_id, &q->ptrs);
    bo_free(q->gpu_id, &q->ring);
    delete q;
}

SdmaQ* q_open(int dev, int engine, std::string* why) {
    if (kfd() < 0) { *why = sfmt("/dev/kfd: %s", std::strerror(errno)); return nullptr; }
    std::unique_ptr<SdmaQ> q(new SdmaQ());
    q->dev = dev;
    q->engine = engine;
    int minor = -1;
    if (!topo_find(dev, &q->gpu_id, &minor, why)) return nullptr;
    if (!bo_alloc(q->gpu_id, minor, kRingBytes, &q->ring, why)) return nullptr;
    if (!bo_alloc(q->gpu_id, minor, 4096, &q->ptrs, why)) { bo_free(q->gpu_id, &q->ring); return nullptr; }
    std::memset(q->ring.va, 0, kRingBytes);
    std::memset(q->ptrs.va, 0, 4096);
    q->rb = (uint32_t*)q->ring.va;
    q->rptr = (volatile uint64_t*)q->ptrs.va;
    q->wptr = (volatile uint64_t*)(q->ptrs.va + 64);
    q->fin = (volatile uint32_t*)(q->ptrs.va + 128);
    KfdCreateQueue a{};
    a.ring_base_address = (uint64_t)(uintptr_t)q->ring.va;
    a.ring_size = (uint32_t)kRingBytes;
    a.read_pointer_address = (uint64_t)(uintptr_t)q->rptr;
    a.write_pointer_address = (uint64_t)(uintptr_t)q->wptr;
    a.gpu_id = q->gpu_id;
    a.queue_type = kKfdQueueSdmaByEngine;
    a.sdma_engine_id = (uint32_t)engine;
    a.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE;
    a.queue_priority = KFD_MAX_QUEUE_PRIORITY;
    if (ioctl(kfd(), kKfdCreateQueue, &a) < 0) {
        *why = sfmt("the driver refused an SDMA queue on engine %d: %s", engine, std::strerror(errno));
        bo_free(q->gpu_id, &q->ptrs);
        bo_free(q->gpu_id, &q->ring);
        return nullptr;
    }
    q->qid = a.queue_id;
    void* m = mmap(nullptr, kDoorbellPage, PROT_READ | PROT_WRITE, MAP_SHARED, kfd(),
                   (off_t)(a.doorbell_offset & ~(uint64_t)(kDoorbellPage - 1)));
    if (m == MAP_FAILED) {
        *why = sfmt("the SDMA doorbell could not be mapped: %s", std::strerror(errno));
        q_close(q.release());
        return nullptr;
    }
    q->db_page = m;
    q->db = (volatile uint64_t*)((char*)m + (a.doorbell_offset & (kDoorbellPage - 1)));
    return q.release();
}

/* ------------------------------------------------------------------ kernels */

constexpr int kFlagStride = 16;   /* words: one 64-byte line a flag */
enum : int { kFA = 0, kFB = 8, kFGo1 = 16, kFGo2 = 17, kFDone = 18, kFCtr = 19, kFErr = 20, kFWords = 24 };

__device__ __forceinline__ unsigned int ld_sys(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_sys(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* Writes back this card's L2 (system scope), then the flag. */
__global__ void k_sdma_flag(unsigned int* fl, int idx, unsigned int v) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        st_sys(fl + idx * kFlagStride, v);
    }
}

/* Waits until every flag of `mask` (bit i = flag index i) holds v, then invalidates this card's L2 (system scope).
 * A wait past ~20 s records v | 0x80000000 in the error word and returns (the host sees it; the bytes are wrong). */
__global__ void k_sdma_wait(unsigned int* fl, unsigned int mask, unsigned int v) {
    const int t = threadIdx.x;
    if (t < 32 && ((mask >> t) & 1u)) {
        const unsigned int* p = fl + t * kFlagStride;
        const unsigned long long t0 = wall_clock64();
        while (ld_sys(p) != v) {
            __builtin_amdgcn_s_sleep(1);
            if (wall_clock64() - t0 > 2000000000ull) { st_sys(fl + kFErr * kFlagStride, v | 0x80000000u); break; }
        }
    }
    __syncthreads();
    __threadfence_system();
}

struct FoldSrc { const int4* p[8]; };

/* out[w] = the CU two-shot's fold of word w of this rank's shard: fp32 from 0, ascending rank, bf16 at the end. Every
 * block writes back L2 (system) after its stores; the last block to finish sets go2. */
template <typename T, int WS>
__global__ void __launch_bounds__(256) k_sdma_fold(FoldSrc src, int4* __restrict__ out, int n16, unsigned int* fl,
                                                    unsigned int v) {
    constexpr int LANES = 16 / sizeof(T);
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n16; k += gridDim.x * blockDim.x) {
        float acc[LANES];
#pragma unroll
        for (int l = 0; l < LANES; ++l) acc[l] = 0.0f;
#pragma unroll
        for (int r = 0; r < WS; ++r) {
            const int4 x = src.p[r][k];
            const T* px = reinterpret_cast<const T*>(&x);
#pragma unroll
            for (int l = 0; l < LANES; ++l) acc[l] += r4d_ar_to_f(px[l]);
        }
        int4 o;
        T* po = reinterpret_cast<T*>(&o);
#pragma unroll
        for (int l = 0; l < LANES; ++l) r4d_ar_from_f(acc[l], po[l]);
        out[k] = o;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int* ctr = fl + kFCtr * kFlagStride;
        const unsigned int prev = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev == gridDim.x - 1) {
            st_sys(ctr, 0u);
            __threadfence_system();
            st_sys(fl + kFGo2 * kFlagStride, v);
        }
    }
}

struct AsmSeg { const int4* src[8]; int off[8]; int len[8]; };

/* out[shard p] = B slot p for each peer p (blockIdx.y = the segment). */
__global__ void __launch_bounds__(256) k_sdma_assemble(AsmSeg s, int4* __restrict__ out) {
    const int g = blockIdx.y;
    const int4* __restrict__ a = s.src[g];
    int4* __restrict__ o = out + s.off[g];
    const int n = s.len[g];
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) o[k] = a[k];
}

/* VRAM-bound load for the contention case: dst = src, `reps` times. */
__global__ void __launch_bounds__(256) k_sdma_stream(const int4* __restrict__ src, int4* __restrict__ dst, long long n16,
                                                      int reps) {
    for (int r = 0; r < reps; ++r)
        for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x; k < n16; k += (long long)gridDim.x * blockDim.x)
            dst[k] = src[k];
}

/* The check's values: per (rank, element, salt) a bf16 with a spread of exponents, both signs and some +-0. */
__device__ __forceinline__ __hip_bfloat16 gen_val(int rank, long long i, unsigned int salt) {
    unsigned int h = (unsigned int)i * 0x9E3779B1u ^ (unsigned int)(i >> 32) * 0x85EBCA77u ^ (unsigned int)rank * 0xC2B2AE3Du ^
                     salt * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    const unsigned int lo = h & 0xffu;
    float x;
    if (lo == 0u) x = -0.0f;
    else if (lo == 1u) x = 0.0f;
    else x = ldexpf(1.0f + (float)((h >> 8) & 0xffffu) / 65536.0f, (int)((h >> 24) & 15u) - 8) * ((h >> 28) & 1u ? -1.0f : 1.0f);
    return (__hip_bfloat16)x;
}

__global__ void __launch_bounds__(256) k_sdma_gen(__hip_bfloat16* __restrict__ buf, long long n, int rank, unsigned int salt) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        buf[i] = gen_val(rank, i, salt);
}

/* Counts the elements of `out` that differ (bitwise) from the fp32 ascending-rank fold of the generated inputs. */
__global__ void __launch_bounds__(256) k_sdma_check(const unsigned short* __restrict__ out, long long n, int world,
                                                     unsigned int salt, unsigned long long* bad) {
    unsigned long long b = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < world; ++r) acc += (float)gen_val(r, i, salt);
        const __hip_bfloat16 w = (__hip_bfloat16)acc;
        unsigned short wb;
        __builtin_memcpy(&wb, &w, 2);
        b += out[i] != wb;
    }
    if (b) atomicAdd(bad, b);
}

/* ------------------------------------------------------------------ the all-reduce context */

struct Rv {
    std::mutex m;
    std::condition_variable cv;
    int arrived = 0, epoch = 0, failed = 0;
    char* scr_a[8] = { nullptr };
    char* scr_b[8] = { nullptr };
    unsigned int* fl[8] = { nullptr };
    int dev[8] = { 0 };
};
Rv g_rv;

int rv_barrier(int world) {
    std::unique_lock<std::mutex> lk(g_rv.m);
    const int e = g_rv.epoch;
    if (++g_rv.arrived == world) {
        g_rv.arrived = 0;
        g_rv.epoch++;
        g_rv.cv.notify_all();
        return RAD_OK;
    }
    return g_rv.cv.wait_for(lk, std::chrono::seconds(60), [&] { return g_rv.epoch != e; }) ? RAD_OK : RAD_E_STATE;
}

struct SdmaAr {
    int world = 4, rank = 0, dev = 0, scr_mode = 1;
    SdmaQ* q = nullptr;
    long long max_numel = 0;
    size_t slot_bytes = 0;
    char* scr_a = nullptr;      /* world slots: slot s written by rank s's engine (first hop) */
    char* scr_b = nullptr;      /* world slots: slot s = rank s's folded shard (second hop) */
    unsigned int* fl = nullptr; /* kFWords lines, uncached */
    char* peer_a[8] = { nullptr };
    char* peer_b[8] = { nullptr };
    unsigned int* peer_fl[8] = { nullptr };
    unsigned int seq = 0;
    int fold_blocks = 0;
};

void* dev_alloc(size_t bytes, int mode) {
    void* p = nullptr;
    hipError_t e;
    if (mode == 0) e = hipMalloc(&p, bytes);
    else e = hipExtMallocWithFlags(&p, bytes, mode == 2 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained);
    if (e != hipSuccess) { (void)hipGetLastError(); return nullptr; }
    return p;
}

void shard_table(long long n16, int world, int* off) {
    const long long base = n16 / world, rem = n16 % world;
    for (int s = 0; s <= world; ++s) off[s] = (int)(s * base + std::min<long long>(s, rem));
}

}  // namespace

/* ------------------------------------------------------------------ test entry points */

extern "C" {

void* r4d_sdma_open(int dev, int engine, char* why, int cap) {
    std::string w;
    int cur = -1;
    (void)hipGetDevice(&cur);
    SdmaQ* q = q_open(dev, engine, &w);
    if (!q && why && cap > 0) std::snprintf(why, (size_t)cap, "%s", w.c_str());
    return q;
}
void r4d_sdma_close(void* q) { q_close((SdmaQ*)q); }
int r4d_sdma_copy(void* q, void* dst, const void* src, unsigned long long bytes) {
    std::lock_guard<std::mutex> lk(((SdmaQ*)q)->mu);
    return q_copy((SdmaQ*)q, dst, src, bytes);
}
int r4d_sdma_fence(void* q, void* addr, unsigned int v) {
    std::lock_guard<std::mutex> lk(((SdmaQ*)q)->mu);
    return q_fence((SdmaQ*)q, addr, v);
}
int r4d_sdma_poll(void* q, const void* addr, unsigned int v) {
    std::lock_guard<std::mutex> lk(((SdmaQ*)q)->mu);
    return q_poll((SdmaQ*)q, addr, v);
}
int r4d_sdma_kick(void* q) {
    std::lock_guard<std::mutex> lk(((SdmaQ*)q)->mu);
    return q_kick((SdmaQ*)q);
}

int r4d_sdma_k_stream(hipStream_t st, void* dst, const void* src, long long bytes, int reps, int blocks) {
    k_sdma_stream<<<blocks > 0 ? blocks : 1024, 256, 0, st>>>((const int4*)src, (int4*)dst, bytes / 16, reps);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int r4d_sdma_k_gen(hipStream_t st, void* buf, long long n, int rank, unsigned int salt) {
    k_sdma_gen<<<1024, 256, 0, st>>>((__hip_bfloat16*)buf, n, rank, salt);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int r4d_sdma_k_check(hipStream_t st, const void* out, long long n, int world, unsigned int salt, unsigned long long* bad) {
    k_sdma_check<<<1024, 256, 0, st>>>((const unsigned short*)out, n, world, salt, bad);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* Collective: every rank of `world` calls it (one thread a card, the current device = `dev`). scr_mode: 0 coarse
 * (hipMalloc), 1 fine-grained (today's AR scratch), 2 uncached. */
void* r4d_sdma_ar_create(int world, int rank, int dev, int engine, long long max_numel, int scr_mode, char* why, int cap) {
    auto fail = [&](const std::string& s) -> void* {
        if (why && cap > 0) std::snprintf(why, (size_t)cap, "%s", s.c_str());
        return nullptr;
    };
    if (world < 2 || world > 8 || rank < 0 || rank >= world || max_numel <= 0 || max_numel % 8) {
        rv_barrier(world); rv_barrier(world);
        return fail("bad arguments");
    }
    std::unique_ptr<SdmaAr> a(new SdmaAr());
    a->world = world; a->rank = rank; a->dev = dev; a->max_numel = max_numel; a->scr_mode = scr_mode;
    std::string w;
    bool ok = hipSetDevice(dev) == hipSuccess;
    if (ok) a->q = q_open(dev, engine, &w);
    ok = ok && a->q;
    const long long n16 = max_numel * 2 / 16;
    a->slot_bytes = (size_t)(((n16 + world - 1) / world) * 16 + 4095) / 4096 * 4096;
    if (ok) {
        a->scr_a = (char*)dev_alloc(a->slot_bytes * world, scr_mode);
        a->scr_b = (char*)dev_alloc(a->slot_bytes * world, scr_mode);
        a->fl = (unsigned int*)dev_alloc(kFWords * kFlagStride * 4, 2);
        ok = a->scr_a && a->scr_b && a->fl;
        if (!ok) w = "device allocation failed";
        else ok = hipMemset(a->fl, 0, kFWords * kFlagStride * 4) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
    }
    int nsm = 0;
    if (hipDeviceGetAttribute(&nsm, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess) nsm = 64;
    a->fold_blocks = std::max(nsm * 4, 64);
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        g_rv.scr_a[rank] = ok ? a->scr_a : nullptr;
        g_rv.scr_b[rank] = ok ? a->scr_b : nullptr;
        g_rv.fl[rank] = ok ? a->fl : nullptr;
        g_rv.dev[rank] = dev;
        if (!ok) g_rv.failed++;
    }
    if (rv_barrier(world) != RAD_OK) ok = false;
    int failed;
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        failed = g_rv.failed;
        for (int r = 0; r < world; ++r) {
            a->peer_a[r] = g_rv.scr_a[r];
            a->peer_b[r] = g_rv.scr_b[r];
            a->peer_fl[r] = g_rv.fl[r];
        }
    }
    if (ok && !failed)
        for (int r = 0; r < world; ++r) {
            if (r == rank || g_rv.dev[r] == dev) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(g_rv.dev[r], 0);
            if (e == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
            else if (e != hipSuccess) { ok = false; w = std::string("peer access: ") + hipGetErrorString(e); }
        }
    if (rv_barrier(world) != RAD_OK) ok = false;
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        failed = g_rv.failed;
    }
    if (!ok || failed) {
        if (a->q) q_close(a->q);
        for (void* p : { (void*)a->scr_a, (void*)a->scr_b, (void*)a->fl }) if (p) (void)hipFree(p);
        return fail(w.empty() ? "another rank failed" : w);
    }
    return a.release();
}

void r4d_sdma_ar_destroy(void* p) {
    SdmaAr* a = (SdmaAr*)p;
    if (!a) return;
    q_close(a->q);
    for (void* x : { (void*)a->scr_a, (void*)a->scr_b, (void*)a->fl }) if (x) (void)hipFree(x);
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        g_rv.failed = 0;
    }
    delete a;
}

/* The error word (nonzero: a wait timed out at that launch value | 0x80000000). */
unsigned int r4d_sdma_ar_error(void* p) {
    SdmaAr* a = (SdmaAr*)p;
    unsigned int v = 0;
    if (a && hipMemcpy(&v, a->fl + kFErr * kFlagStride, 4, hipMemcpyDeviceToHost) != hipSuccess) v = 0xffffffffu;
    return v;
}

/* One exact all-reduce of `numel` bf16 (numel % 8 == 0, <= max_numel) from `in` into `out` (may equal `in`) on `st`.
 * mode 0 the all-reduce; 1 the reduce-scatter half alone (first hop + fold: out[shard rank] only); 2 the all-gather half
 * alone (out[shard rank] taken as folded: second hop + assemble). The halves are timed only (SP's raw RS / AG). */
int r4d_sdma_ar_launch(void* p, const void* in, void* out, long long numel, hipStream_t st, int mode) {
    SdmaAr* a = (SdmaAr*)p;
    if (!a || numel <= 0 || numel % 8 || numel > a->max_numel || mode < 0 || mode > 2) return RAD_E_SHAPE;
    const int W = a->world, r = a->rank;
    if (W != 2 && W != 3 && W != 4 && W != 8) return RAD_E_SHAPE;
    const bool hop1 = mode != 2, hop2 = mode != 1;
    const long long n16 = numel * 2 / 16;
    int off[9];
    shard_table(n16, W, off);
    const unsigned int v = ++a->seq;
    const int4* in4 = (const int4*)in;
    int4* out4 = (int4*)out;
    unsigned int* fl = a->fl;
    k_sdma_flag<<<1, 64, 0, st>>>(fl, hop1 ? kFGo1 : kFGo2, v);
    int rc = RAD_OK;
    {
        SdmaQ* q = a->q;
        std::lock_guard<std::mutex> lk(q->mu);
        if (hop1) {
            rc = q_poll(q, fl + kFGo1 * kFlagStride, v);
            for (int i = 1; rc == RAD_OK && i < W; ++i) {
                const int pr = (r + i) % W;
                const uint64_t len = (uint64_t)(off[pr + 1] - off[pr]) * 16;
                if (len) rc = q_copy(q, a->peer_a[pr] + (size_t)r * a->slot_bytes, in4 + off[pr], len);
                if (rc == RAD_OK) rc = q_fence(q, a->peer_fl[pr] + (kFA + r) * kFlagStride, v);
            }
        }
        if (hop2) {
            if (rc == RAD_OK) rc = q_poll(q, fl + kFGo2 * kFlagStride, v);
            const uint64_t mylen = (uint64_t)(off[r + 1] - off[r]) * 16;
            for (int i = 1; rc == RAD_OK && i < W; ++i) {
                const int pr = (r + i) % W;
                if (mylen) rc = q_copy(q, a->peer_b[pr] + (size_t)r * a->slot_bytes, out4 + off[r], mylen);
                if (rc == RAD_OK) rc = q_fence(q, a->peer_fl[pr] + (kFB + r) * kFlagStride, v);
            }
        }
        if (rc == RAD_OK) rc = q_fence(q, fl + kFDone * kFlagStride, v);
        const int krc = q_kick(q);
        if (rc == RAD_OK) rc = krc;
    }
    if (rc != RAD_OK) return rc;
    unsigned int mask_a = 0, mask_b = 0;
    for (int pr = 0; pr < W; ++pr)
        if (pr != r) { mask_a |= 1u << (kFA + pr); mask_b |= 1u << (kFB + pr); }
    if (hop1) {
        k_sdma_wait<<<1, 64, 0, st>>>(fl, mask_a, v);
        FoldSrc src{};
        for (int s = 0; s < W; ++s)
            src.p[s] = s == r ? in4 + off[r] : (const int4*)(a->scr_a + (size_t)s * a->slot_bytes);
        const int myn = off[r + 1] - off[r];
        const int fb = std::max(1, std::min(a->fold_blocks, (myn + 255) / 256));
        switch (W) {
            case 2: k_sdma_fold<__hip_bfloat16, 2><<<fb, 256, 0, st>>>(src, out4 + off[r], myn, fl, v); break;
            case 3: k_sdma_fold<__hip_bfloat16, 3><<<fb, 256, 0, st>>>(src, out4 + off[r], myn, fl, v); break;
            case 4: k_sdma_fold<__hip_bfloat16, 4><<<fb, 256, 0, st>>>(src, out4 + off[r], myn, fl, v); break;
            default: k_sdma_fold<__hip_bfloat16, 8><<<fb, 256, 0, st>>>(src, out4 + off[r], myn, fl, v); break;
        }
    }
    k_sdma_wait<<<1, 64, 0, st>>>(fl, (hop2 ? mask_b : 0u) | (1u << kFDone), v);
    if (hop2) {
        AsmSeg seg{};
        int g = 0, most = 0;
        for (int pr = 0; pr < W; ++pr) {
            if (pr == r) continue;
            seg.src[g] = (const int4*)(a->scr_b + (size_t)pr * a->slot_bytes);
            seg.off[g] = off[pr];
            seg.len[g] = off[pr + 1] - off[pr];
            most = std::max(most, seg.len[g]);
            ++g;
        }
        const int ab = std::max(1, std::min(a->fold_blocks / std::max(1, g) + 1, (most + 255) / 256));
        k_sdma_assemble<<<dim3(ab, g), 256, 0, st>>>(seg, out4);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  // extern "C"
