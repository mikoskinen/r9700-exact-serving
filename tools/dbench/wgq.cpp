/* wgq (DRAFTER track, 18.11.2026 round 5) -- one card: does a small elementwise launch's time depend on its workgroup
 * COUNT? abench a5 measured quant_roundtrip at 256 and 512 workgroups of 256 threads at ~23-24 us against 7-13 us at
 * 128 / 192 / 320 / 640 / 1024 (the same kernel, other shapes). Here a trivial kernel: each thread loads E bytes of its
 * own slice of a buffer, adds one, stores (in place or to a second buffer), grid G x 1 (or G/Y x Y) of T threads, K
 * launches back to back between two events, median of `iters`.
 * usage: wgq [--t 256] [--e 2] [--iters 50] [--grids 128,256,...] [--y 1] [--inplace 1] */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

template <int E>
__global__ void touch(const unsigned char* __restrict__ a, unsigned char* __restrict__ b, long per_block) {
    const long blk = (long)blockIdx.y * gridDim.x + blockIdx.x;
    const long off = blk * per_block + (long)threadIdx.x * E;
    if constexpr (E == 2) { unsigned short v = *(const unsigned short*)(a + off); *(unsigned short*)(b + off) = (unsigned short)(v + 1); }
    if constexpr (E == 16) { uint4 v = *(const uint4*)(a + off); v.x += 1; *(uint4*)(b + off) = v; }
}
__global__ void empty_k() {}

static std::vector<int> ints(const char* s) { std::vector<int> v; while (s && *s) { v.push_back(std::atoi(s)); s = std::strchr(s, ','); if (s) ++s; } return v; }

int main(int argc, char** argv) {
    int T = 256, E = 2, iters = 50, Y = 1, inplace = 1;
    std::vector<int> grids = { 32, 64, 128, 192, 224, 240, 248, 256, 264, 288, 320, 384, 448, 480, 504, 512, 520, 576, 640, 768, 1024, 2048 };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)""; };
        if (k == "--t") T = std::atoi(nx());
        else if (k == "--e") E = std::atoi(nx());
        else if (k == "--iters") iters = std::atoi(nx());
        else if (k == "--grids") grids = ints(nx());
        else if (k == "--y") Y = std::atoi(nx());
        else if (k == "--inplace") inplace = std::atoi(nx());
    }
    int dev; CK(hipGetDevice(&dev)); hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, dev));
    std::printf("wgq: %s, %d CUs; T %d threads, %d bytes a thread, y %d, %s, K 20 a sample, median of %d\n", pr.name,
                pr.multiProcessorCount, T, E, Y, inplace ? "in place" : "a to b", iters);
    const long per = (long)T * E;
    const long maxb = (long)*std::max_element(grids.begin(), grids.end()) * per;
    unsigned char *a, *b; CK(hipMalloc(&a, maxb)); CK(hipMalloc(&b, maxb)); CK(hipMemset(a, 1, maxb)); CK(hipMemset(b, 1, maxb));
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int K = 20;
    auto meas = [&](auto launch) {
        std::vector<float> v;
        for (int it = 0; it < iters + 3; ++it) {
            CK(hipEventRecord(e0, st));
            for (int k = 0; k < K; ++k) launch();
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) v.push_back(ms * 1000.0f / K);
        }
        std::sort(v.begin(), v.end());
        return std::make_pair(v[v.size() / 2], v[v.size() * 9 / 10]);
    };
    auto em = meas([&]() { hipLaunchKernelGGL(empty_k, dim3(1), dim3(32), 0, st); });
    std::printf("  empty kernel, 1 workgroup: %.2f us a launch\n", em.first);
    for (int G : grids) {
        const int gy = Y > 1 && G % Y == 0 ? Y : 1, gx = G / gy;
        unsigned char* dst = inplace ? a : b;
        auto r = meas([&]() {
            if (E == 16) hipLaunchKernelGGL(touch<16>, dim3(gx, gy), dim3(T), 0, st, a, dst, per);
            else hipLaunchKernelGGL(touch<2>, dim3(gx, gy), dim3(T), 0, st, a, dst, per);
        });
        std::printf("  %5d workgroups (%d x %d) = %6d waves: %7.2f us (p90 %.2f)\n", G, gx, gy, G * ((T + 31) / 32), r.first, r.second);
    }
    return 0;
}
