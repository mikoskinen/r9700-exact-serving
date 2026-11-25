// bw4.cpp: N-card version of bw.cpp: host<->device per card alone and all at once, peer copy for every ordered pair,
// and all cards sending to their neighbour at once. hipcc -O2 bw4.cpp -o bw4
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
static double run(std::vector<std::function<void()>> jobs) { double t0 = now(); std::vector<std::thread> th; for (auto& j : jobs) th.emplace_back(j); for (auto& t : th) t.join(); return now() - t0; }
int main() {
  int n = 0; CK(hipGetDeviceCount(&n)); printf("devices %d\n", n);
  std::vector<void*> d(n), d2(n), h(n); std::vector<hipStream_t> s(n);
  for (int i = 0; i < n; i++) { CK(hipSetDevice(i)); CK(hipMalloc(&d[i], SZ)); CK(hipMalloc(&d2[i], SZ)); CK(hipHostMalloc(&h[i], SZ, hipHostMallocDefault)); CK(hipStreamCreate(&s[i]));
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, i)); printf("dev %d pci %02x:%02x\n", i, p.pciBusID, p.pciDeviceID); }
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) { int a = 0; CK(hipDeviceCanAccessPeer(&a, i, j)); if (a) { CK(hipSetDevice(i)); hipDeviceEnablePeerAccess(j, 0); } else printf("no peer %d->%d\n", i, j); }
  auto cp = [&](int dev, void* dst, int dd, void* src, int sd) { return [=, &s]() { CK(hipSetDevice(dev)); for (int r = 0; r < REPS; r++) {
      if (dd >= 0 && sd >= 0) CK(hipMemcpyPeerAsync(dst, dd, src, sd, SZ, s[dev])); else CK(hipMemcpyAsync(dst, src, SZ, hipMemcpyDefault, s[dev])); } CK(hipStreamSynchronize(s[dev])); }; };
  double GB = (double)SZ * REPS / 1e9; std::vector<std::function<void()>> all;
  for (int i = 0; i < n; i++) all.push_back(cp(i, d[i], -1, h[i], -1)); run(all);  // warm-up
  for (int i = 0; i < n; i++) printf("card %d h2d %.1f d2h %.1f GB/s\n", i, GB / run({cp(i, d[i], -1, h[i], -1)}), GB / run({cp(i, h[i], -1, d[i], -1)}));
  printf("all %d cards h2d at once %.1f GB/s total\n", n, n * GB / run(all));
  all.clear(); for (int i = 0; i < n; i++) all.push_back(cp(i, h[i], -1, d[i], -1)); printf("all %d cards d2h at once %.1f GB/s total\n", n, n * GB / run(all));
  printf("peer matrix GB/s (row = from, col = to):\n");
  for (int i = 0; i < n; i++) { printf("  %d:", i); for (int j = 0; j < n; j++) { if (i == j) { printf("    -  "); continue; } printf(" %6.1f", GB / run({cp(i, d2[j], j, d[i], i)})); } printf("\n"); }
  all.clear(); for (int i = 0; i < n; i++) { int j = (i + 1) % n; all.push_back(cp(i, d2[j], j, d[i], i)); }
  printf("ring: every card -> next card at once %.1f GB/s total\n", n * GB / run(all));
  return 0;
}
