/* gstream (DRAFTER track 20.11.2026 round 30): the bytes-once floor of ONE small streaming call at a given grid -- how fast
 * the card reads B bytes (the drafter's dense q8_0 weights: 8.9-11.1 MB a call) cold, with G blocks of 256 threads, each
 * block a contiguous chunk, U 16-byte loads in flight a thread. Timed by the card: every block stamps wall_clock64()
 * (100 MHz) at entry and exit; span = last exit - first entry (no launch / event overhead). Before every timed launch a
 * 256 MiB read evicts L2 + the 64 MiB LLC. Prints the p50 span over R launches and GB/s.
 *   gstream [R] [sizes MB,..] [grids,..] [unrolls,..] */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ __launch_bounds__(256) void flush(const uint4* p, size_t n, uint32_t* o) {
    uint32_t a = 0;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) o[0] = a;
}
template <int U>
__global__ __launch_bounds__(256) void rd(const uint4* p, long long n16, uint32_t* o, unsigned long long* st) {
    const unsigned long long t0 = wall_clock64();
    const long long per = (n16 + gridDim.x - 1) / gridDim.x;
    const long long b0 = (long long)blockIdx.x * per, b1 = std::min(b0 + per, n16);
    uint32_t a = 0;
    long long i = b0 + threadIdx.x;
    for (; i + (U - 1) * 256 < b1; i += U * 256) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = p[i + u * 256];
#pragma unroll
        for (int u = 0; u < U; ++u) a ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < b1; i += 256) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) o[0] = a;
    __syncthreads();
    if (threadIdx.x == 0) { st[2 * blockIdx.x] = t0; st[2 * blockIdx.x + 1] = wall_clock64(); }
}
static std::vector<double> dl(const char* s, const char* d) {
    std::vector<double> v; std::string m = s ? s : d; size_t p = 0;
    while (p < m.size()) { size_t q = m.find(',', p); if (q == std::string::npos) q = m.size(); v.push_back(std::atof(m.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}
int main(int argc, char** argv) {
    const int R = argc > 1 ? std::atoi(argv[1]) : 30;
    const std::vector<double> sizes = dl(argc > 2 ? argv[2] : nullptr, "9.75,11.14,8.91,136");
    const std::vector<double> grids = dl(argc > 3 ? argv[3] : nullptr, "40,56,64,80,112,128,160,224,256,512,1024");
    const std::vector<double> unr = dl(argc > 4 ? argv[4] : nullptr, "1,2,4,8");
    const size_t FB = 256ull << 20;
    uint4 *f, *p; uint32_t* o; unsigned long long* st;
    CK(hipMalloc((void**)&f, FB)); CK(hipMemset(f, 1, FB));
    CK(hipMalloc((void**)&p, 160ull << 20)); CK(hipMemset(p, 3, 160ull << 20));
    CK(hipMalloc((void**)&o, 64)); CK(hipMalloc((void**)&st, 2 * 4096 * 8));
    std::vector<unsigned long long> hs(2 * 4096);
    for (double mb : sizes) {
        const long long n16 = (long long)(mb * 1e6 / 16);
        for (double gd : grids) {
            const int G = (int)gd;
            std::string line;
            for (double ud : unr) {
                const int U = (int)ud;
                std::vector<double> sp;
                for (int r = 0; r < R + 2; ++r) {
                    flush<<<1024, 256>>>(f, FB / 16, o);
                    if (U == 1) rd<1><<<G, 256>>>(p, n16, o, st);
                    else if (U == 2) rd<2><<<G, 256>>>(p, n16, o, st);
                    else if (U == 4) rd<4><<<G, 256>>>(p, n16, o, st);
                    else rd<8><<<G, 256>>>(p, n16, o, st);
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(hs.data(), st, 2 * G * 8, hipMemcpyDeviceToHost));
                    unsigned long long a = ~0ull, b = 0;
                    for (int i = 0; i < G; ++i) { a = std::min(a, hs[2 * i]); b = std::max(b, hs[2 * i + 1]); }
                    if (r >= 2) sp.push_back((double)(b - a) / 100.0);   /* 100 MHz -> us */
                }
                std::sort(sp.begin(), sp.end());
                const double us = sp[sp.size() / 2];
                line += " U" + std::to_string(U) + " " + std::to_string(us).substr(0, 6) + " us " + std::to_string(mb * 1e3 / us).substr(0, 5) + " GB/s |";
            }
            std::printf("GSTREAM %.2f MB grid %4d:%s\n", mb, G, line.c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}
