// bw.cpp: host<->device (pinned) and peer copy bandwidth on two cards, alone and concurrently. hipcc -O2 bw.cpp -o bw
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("ERR %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)
static const size_t SZ = 256ull << 20; static const int REPS = 20;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Dev { int id; void* d; void* d2; void* h; hipStream_t s; };
static double run(std::vector<std::function<void()>> jobs) {  // jobs in parallel threads, returns wall s
  double t0 = now(); std::vector<std::thread> th; for (auto& j : jobs) th.emplace_back(j); for (auto& t : th) t.join(); return now() - t0; }
static std::function<void()> copy(Dev& x, void* dst, const void* src, int dstdev, int srcdev) {
  return [=, &x]() { CK(hipSetDevice(x.id)); for (int r = 0; r < REPS; r++) {
    if (dstdev >= 0 && srcdev >= 0 && dstdev != srcdev) CK(hipMemcpyPeerAsync(dst, dstdev, src, srcdev, SZ, x.s));
    else CK(hipMemcpyAsync(dst, src, SZ, hipMemcpyDefault, x.s)); } CK(hipStreamSynchronize(x.s)); }; }
int main() {
  int n = 0; CK(hipGetDeviceCount(&n)); if (n < 2) { printf("need 2 devices\n"); return 1; }
  Dev g[2];
  for (int i = 0; i < 2; i++) { g[i].id = i; CK(hipSetDevice(i)); CK(hipMalloc(&g[i].d, SZ)); CK(hipMalloc(&g[i].d2, SZ));
    CK(hipHostMalloc(&g[i].h, SZ, hipHostMallocDefault)); CK(hipStreamCreate(&g[i].s));
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, i)); printf("dev %d: %s pci %02x:%02x\n", i, p.name, p.pciBusID, p.pciDeviceID); }
  int a01 = 0, a10 = 0; CK(hipDeviceCanAccessPeer(&a01, 0, 1)); CK(hipDeviceCanAccessPeer(&a10, 1, 0));
  printf("can access peer 0->1 %d, 1->0 %d\n", a01, a10);
  if (a01) { CK(hipSetDevice(0)); CK(hipDeviceEnablePeerAccess(1, 0)); } if (a10) { CK(hipSetDevice(1)); CK(hipDeviceEnablePeerAccess(0, 0)); }
  double GB = (double)SZ * REPS / 1e9;
  for (int w = 0; w < 2; w++) run({copy(g[0], g[0].d, g[0].h, -1, -1), copy(g[1], g[1].d, g[1].h, -1, -1)});  // warm-up
  for (int i = 0; i < 2; i++) {
    printf("card %d host->dev %.1f GB/s\n", i, GB / run({copy(g[i], g[i].d, g[i].h, -1, -1)}));
    printf("card %d dev->host %.1f GB/s\n", i, GB / run({copy(g[i], g[i].h, g[i].d, -1, -1)})); }
  printf("both host->dev concurrently %.1f GB/s total\n", 2 * GB / run({copy(g[0], g[0].d, g[0].h, -1, -1), copy(g[1], g[1].d, g[1].h, -1, -1)}));
  printf("both dev->host concurrently %.1f GB/s total\n", 2 * GB / run({copy(g[0], g[0].h, g[0].d, -1, -1), copy(g[1], g[1].h, g[1].d, -1, -1)}));
  printf("both cards h2d+d2h at once %.1f GB/s total\n", 4 * GB / run({copy(g[0], g[0].d, g[0].h, -1, -1), copy(g[1], g[1].d, g[1].h, -1, -1),
                                                                       [&]() { Dev t = g[0]; CK(hipSetDevice(0)); hipStream_t s; CK(hipStreamCreate(&s)); for (int r = 0; r < REPS; r++) CK(hipMemcpyAsync(g[0].h, g[0].d2, SZ, hipMemcpyDefault, s)); CK(hipStreamSynchronize(s)); },
                                                                       [&]() { CK(hipSetDevice(1)); hipStream_t s; CK(hipStreamCreate(&s)); for (int r = 0; r < REPS; r++) CK(hipMemcpyAsync(g[1].h, g[1].d2, SZ, hipMemcpyDefault, s)); CK(hipStreamSynchronize(s)); }}));
  printf("peer 0->1 %.1f GB/s\n", GB / run({copy(g[0], g[1].d, g[0].d, 1, 0)}));
  printf("peer 1->0 %.1f GB/s\n", GB / run({copy(g[1], g[0].d, g[1].d, 0, 1)}));
  printf("peer bidirectional %.1f GB/s total\n", 2 * GB / run({copy(g[0], g[1].d2, g[0].d, 1, 0), copy(g[1], g[0].d2, g[1].d, 0, 1)}));
  return 0;
}
