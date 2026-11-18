/* peak (DRAFTER track 18.11.2026): this card's DRAM read rate (dwordx4 streaming, 1 GiB) and f32 mul+add issue rate
 * (separate mul and add as -ffp-contract=off emits them), to calibrate dbench's GB/s against the card. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); return 2; } } while (0)
__global__ __launch_bounds__(256) void rd(const uint4* p, size_t n, uint32_t* o) {
    uint32_t a = 0;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) o[0] = a;
}
__global__ __launch_bounds__(256) void fl(float* o, int it, float s) {
    float a[8]; for (int j = 0; j < 8; ++j) a[j] = threadIdx.x * 0.001f + j;
    for (int i = 0; i < it; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) { float p = a[j] * s; a[j] = a[j] + p; }
    float t = 0; for (int j = 0; j < 8; ++j) t += a[j];
    if (t == 1234.5f) o[0] = t;
}
int main() {
    const size_t B = 1ull << 30; uint4* p; uint32_t* o; float* fo;
    CK(hipMalloc((void**)&p, B)); CK(hipMemset(p, 1, B)); CK(hipMalloc((void**)&o, 64)); CK(hipMalloc((void**)&fo, 64));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int grid : { 256, 512, 1024, 2048 }) {
        for (int w = 0; w < 3; ++w) rd<<<grid, 256>>>(p, B / 16, o);
        CK(hipEventRecord(a)); for (int r = 0; r < 10; ++r) rd<<<grid, 256>>>(p, B / 16, o); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b)); std::printf("read grid %d: %.1f GB/s\n", grid, 10.0 * B / (ms * 1e6));
    }
    const int it = 20000, grid = 4096;
    for (int w = 0; w < 2; ++w) fl<<<grid, 256>>>(fo, it, 1e-7f);
    CK(hipEventRecord(a)); fl<<<grid, 256>>>(fo, it, 1e-7f); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    const double ops = 2.0 * 8 * it * (double)grid * 256;
    std::printf("f32 mul+add (separate): %.2f T ops/s = %.1f lanes a clock a CU at the reported clock\n", ops / (ms * 1e9), 0.0);
    return 0;
}
