// dma-check.cpp -- is a host<->device copy a DMA from the host buffer, or a CPU memcpy through a staging buffer?
// Every card at once, own stream per card, timed with GPU events per card (plus host wall clock), host buffer by type:
//   pinned   = hipHostMalloc (SDMA straight from the buffer)
//   register = malloc + hipHostRegister
//   malloc   = plain pageable malloc (HIP stages it through a pinned bounce buffer with the CPU)
// hipcc -O2 -o dma-check dma-check.cpp && ./dma-check pinned|register|malloc [GiB=1] [iters=10]   (31.10., gpu02)
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("ERR %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "pinned";
  size_t n = (size_t)(argc > 2 ? atof(argv[2]) * (1 << 30) : 1ull << 30); int iters = argc > 3 ? atoi(argv[3]) : 10;
  int nd = 0; CK(hipGetDeviceCount(&nd));
  // blocking sync: the waiting host thread sleeps instead of spinning, so CPU time shows only real copy work
  for (int i = 0; i < nd; i++) { CK(hipSetDevice(i)); CK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync)); }
  std::vector<void*> d(nd), h(nd); std::vector<hipStream_t> s(nd); std::vector<hipEvent_t> a(nd), b(nd);
  for (int i = 0; i < nd; i++) {
    CK(hipSetDevice(i)); CK(hipMalloc(&d[i], n)); CK(hipStreamCreateWithFlags(&s[i], hipStreamNonBlocking));
    CK(hipEventCreateWithFlags(&a[i], hipEventBlockingSync)); CK(hipEventCreateWithFlags(&b[i], hipEventBlockingSync));
    if (!strcmp(mode, "pinned")) CK(hipHostMalloc(&h[i], n, hipHostMallocDefault));
    else { h[i] = aligned_alloc(4096, n); memset(h[i], 1, n); if (!strcmp(mode, "register")) CK(hipHostRegister(h[i], n, hipHostRegisterDefault)); }
  }
  printf("mode %s, %d cards, %.2f GiB x %d per card, HSA_ENABLE_SDMA=%s\n", mode, nd, n / 1073741824.0, iters,
         getenv("HSA_ENABLE_SDMA") ? getenv("HSA_ENABLE_SDMA") : "(unset)");
  for (int dir = 0; dir < 2; dir++) {
    const char* nm = dir ? "D2H" : "H2D";
    auto enqueue = [&](int i) {
      for (int it = 0; it < iters; it++)
        CK(dir ? hipMemcpyAsync(h[i], d[i], n, hipMemcpyDeviceToHost, s[i]) : hipMemcpyAsync(d[i], h[i], n, hipMemcpyHostToDevice, s[i]));
    };
    for (int i = 0; i < nd; i++) { CK(hipSetDevice(i)); enqueue(i); CK(hipStreamSynchronize(s[i])); }  // warm
    for (int i = 0; i < nd; i++) {  // each card alone
      CK(hipSetDevice(i)); CK(hipEventRecord(a[i], s[i])); enqueue(i); CK(hipEventRecord(b[i], s[i])); CK(hipEventSynchronize(b[i]));
      float ms; CK(hipEventElapsedTime(&ms, a[i], b[i])); printf("  %s card %d alone   %.1f GB/s\n", nm, i, (double)n * iters / ms / 1e6);
    }
    auto t0 = std::chrono::steady_clock::now();  // all cards at once
    for (int i = 0; i < nd; i++) { CK(hipSetDevice(i)); CK(hipEventRecord(a[i], s[i])); }
    for (int i = 0; i < nd; i++) { CK(hipSetDevice(i)); enqueue(i); CK(hipEventRecord(b[i], s[i])); }
    double sum = 0;
    for (int i = 0; i < nd; i++) {
      CK(hipSetDevice(i)); CK(hipEventSynchronize(b[i])); float ms; CK(hipEventElapsedTime(&ms, a[i], b[i]));
      double g = (double)n * iters / ms / 1e6; sum += g; printf("  %s card %d at once %.1f GB/s (GPU events)\n", nm, i, g);
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  %s ALL %d cards: sum of per-card %.1f GB/s, wall-clock aggregate %.1f GB/s\n", nm, nd, sum, (double)n * iters * nd / wall / 1e9);
  }
  return 0;
}
