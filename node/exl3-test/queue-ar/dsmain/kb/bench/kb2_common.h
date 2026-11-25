// kb2_common.h (12.11.2026, ds-main-port kbench builder) -- the one-card kernel bench's shared host pieces: HIP checks,
// a fixed-seed RNG, host bf16, event timing (median / p10 / p90), a device byte compare, the production libr4d.so
// loaded by dlopen (LIB: the image's own kernels through their real launch entry), RadArgs builders.
#pragma once
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define KB_CK(x)                                                                                     \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::printf("KB2 FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::fflush(stdout);                                                                     \
            std::exit(3);                                                                            \
        }                                                                                            \
    } while (0)

namespace kb {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 0x1234567ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uni() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() {
        double u1 = uni(), u2 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

inline uint16_t h_bf16(float f) {   // round to nearest even (no NaN in the bench's inputs)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float h_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    KB_CK(hipMalloc((void**)&p, n * sizeof(T) + 256));
    KB_CK(hipMemset(p, 0, n * sizeof(T) + 256));
    return p;
}
template <class T>
T* dup(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    KB_CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

/* ---- timing: one event pair a launch, reps after warm-ups; median, p10, p90 in us */
struct Tm {
    double med = 0, p10 = 0, p90 = 0, mean = 0;
    int n = 0;
};
template <class F>
Tm time_it(hipStream_t s, int warm, int reps, F&& launch) {
    for (int i = 0; i < warm; ++i) launch();
    KB_CK(hipStreamSynchronize(s));
    std::vector<hipEvent_t> ev(2 * (size_t)reps);
    for (auto& e : ev) KB_CK(hipEventCreate(&e));
    for (int i = 0; i < reps; ++i) {
        KB_CK(hipEventRecord(ev[2 * i], s));
        launch();
        KB_CK(hipEventRecord(ev[2 * i + 1], s));
    }
    KB_CK(hipStreamSynchronize(s));
    std::vector<double> t((size_t)reps);
    for (int i = 0; i < reps; ++i) {
        float ms = 0;
        KB_CK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]));
        t[i] = ms * 1000.0;
    }
    for (auto& e : ev) KB_CK(hipEventDestroy(e));
    std::vector<double> u = t;
    std::sort(u.begin(), u.end());
    Tm r;
    r.n = reps;
    r.med = u[(size_t)reps / 2];
    r.p10 = u[(size_t)(reps * 0.1)];
    r.p90 = u[std::min((size_t)reps - 1, (size_t)(reps * 0.9))];
    double s0 = 0;
    for (double x : u) s0 += x;
    r.mean = s0 / reps;
    return r;
}

/* ---- device byte compare: 32-bit words of a [rows, cols] region at a row pitch (in words) */
__global__ void kb_cmp32(const uint32_t* a, const uint32_t* b, long pitch_a, long pitch_b, long rows, long cols,
                         unsigned long long* cnt, unsigned long long* first) {
    const long tot = rows * cols;
    unsigned long long local = 0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const long r = i / cols, c = i - r * cols;
        if (a[r * pitch_a + c] != b[r * pitch_b + c]) {
            ++local;
            atomicMin(first, (unsigned long long)i);
        }
    }
    if (local) atomicAdd(cnt, local);
}
__global__ void kb_cmp16(const uint16_t* a, const uint16_t* b, long n, unsigned long long* cnt,
                         unsigned long long* first) {
    unsigned long long local = 0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        if (a[i] != b[i]) {
            ++local;
            atomicMin(first, (unsigned long long)i);
        }
    if (local) atomicAdd(cnt, local);
}
__global__ void kb_fill32(uint32_t* p, long pitch, long rows, long cols, uint32_t v) {
    const long tot = rows * cols;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const long r = i / cols, c = i - r * cols;
        p[r * pitch + c] = v;
    }
}
/* every word of a region equal to v? (count of those that are not) */
__global__ void kb_count_ne32(const uint32_t* p, long pitch, long rows, long c0, long c1, uint32_t v,
                              unsigned long long* cnt) {
    const long cols = c1 - c0, tot = rows * cols;
    unsigned long long local = 0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const long r = i / cols, c = c0 + (i - r * cols);
        if (p[r * pitch + c] != v) ++local;
    }
    if (local) atomicAdd(cnt, local);
}

struct Cmp {
    unsigned long long diff = 0, first = ~0ull, compared = 0;
};
struct Cmper {
    unsigned long long* d = nullptr;   /* allocated at first use (no HIP call before the arguments are read) */
    void need() {
        if (!d) d = dalloc<unsigned long long>(4);
    }
    Cmp words(const void* a, const void* b, long pa, long pb, long rows, long cols, hipStream_t s) {
        need();
        unsigned long long h[2] = { 0, ~0ull };
        KB_CK(hipMemcpyAsync(d, h, 16, hipMemcpyHostToDevice, s));
        kb_cmp32<<<2048, 256, 0, s>>>((const uint32_t*)a, (const uint32_t*)b, pa, pb, rows, cols, d, d + 1);
        KB_CK(hipGetLastError());
        KB_CK(hipMemcpyAsync(h, d, 16, hipMemcpyDeviceToHost, s));
        KB_CK(hipStreamSynchronize(s));
        Cmp c;
        c.diff = h[0];
        c.first = h[1];
        c.compared = (unsigned long long)rows * (unsigned long long)cols * 4ull;
        return c;
    }
    Cmp halfs(const void* a, const void* b, long n, hipStream_t s) {
        need();
        unsigned long long h[2] = { 0, ~0ull };
        KB_CK(hipMemcpyAsync(d, h, 16, hipMemcpyHostToDevice, s));
        kb_cmp16<<<2048, 256, 0, s>>>((const uint16_t*)a, (const uint16_t*)b, n, d, d + 1);
        KB_CK(hipGetLastError());
        KB_CK(hipMemcpyAsync(h, d, 16, hipMemcpyDeviceToHost, s));
        KB_CK(hipStreamSynchronize(s));
        Cmp c;
        c.diff = h[0];
        c.first = h[1];
        c.compared = (unsigned long long)n * 2ull;
        return c;
    }
    unsigned long long count_ne(const void* p, long pitch, long rows, long c0, long c1, uint32_t v, hipStream_t s) {
        need();
        unsigned long long h = 0;
        KB_CK(hipMemcpyAsync(d, &h, 8, hipMemcpyHostToDevice, s));
        kb_count_ne32<<<2048, 256, 0, s>>>((const uint32_t*)p, pitch, rows, c0, c1, v, d);
        KB_CK(hipGetLastError());
        KB_CK(hipMemcpyAsync(&h, d, 8, hipMemcpyDeviceToHost, s));
        KB_CK(hipStreamSynchronize(s));
        return h;
    }
};

/* ---- the production library (LIB) */
typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);
struct Lib {
    void* h = nullptr;
    std::string path;
    bool open(const char* p) {
        path = p ? p : "";
        h = p ? dlopen(p, RTLD_NOW | RTLD_LOCAL) : nullptr;
        if (!h) std::printf("KB2 LIB dlopen %s failed: %s\n", path.c_str(), dlerror());
        return h != nullptr;
    }
    template <class T>
    T sym(const char* n) {
        void* s = h ? dlsym(h, n) : nullptr;
        if (!s) std::printf("KB2 LIB symbol %s missing in %s\n", n, path.c_str());
        return (T)s;
    }
};

inline RadTensor tens(void* data, uint32_t dt, std::initializer_list<int64_t> shape,
                      std::initializer_list<int64_t> stride = {}) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = data;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    if (stride.size()) {
        i = 0;
        for (int64_t s : stride) t.stride[i++] = s;
    } else {
        int64_t acc = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) {
            t.stride[k] = acc;
            acc *= t.shape[k];
        }
    }
    return t;
}
inline RadParam pint(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
inline RadParam pf64(const char* k, double v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}
inline RadParam pstr(const char* k, const char* v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}
inline RadArgs args(const std::vector<RadTensor>& t, const std::vector<RadParam>& p, void* scratch = nullptr,
                    int64_t sb = 0) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = scratch;
    a.scratch_bytes = sb;
    a.kv_used_max_h = -1;
    a.world_size = 1;
    return a;
}

/* kernel attributes: VGPRs, LDS, max resident workgroups a CU at the launch's block */
template <class K>
void attr(const char* name, K kern, int block, size_t dyn_lds) {
    hipFuncAttributes fa;
    std::memset(&fa, 0, sizeof fa);
    const hipError_t e = hipFuncGetAttributes(&fa, (const void*)kern);
    int occ = -1;
    const hipError_t e2 = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, block, dyn_lds);
    std::printf("KB2 ATTR %s regs=%d lds_static=%zu lds_dyn=%zu block=%d wg_per_cu=%d waves_per_cu=%d%s\n", name,
                fa.numRegs, (size_t)fa.sharedSizeBytes, dyn_lds, block, occ, occ >= 0 ? occ * block / 32 : -1,
                (e != hipSuccess || e2 != hipSuccess) ? " (attr query failed)" : "");
}

}  // namespace kb
