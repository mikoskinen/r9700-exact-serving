// cpulat.cpp (9.11.2026, queue-ar, item 112) -- which CPU core complexes are near which card. With BIOS NPS1 every card
// reports numa_node -1 and the CCD <-> IOD-quadrant <-> root-complex map is not exposed, so it is measured: from one CPU
// of every L3 group (CCX; /sys/devices/system/cpu/cpu*/cache/index3, CPU topology only -- no card sysfs is read), the
// thread pinned there times, on every card in turn, the host round trips the engine's issue path is made of:
//   kern   an empty kernel launch + hipStreamSynchronize (AQL packet write, doorbell MMIO, completion signal back)
//   d2h4   a 4-byte hipMemcpyAsync device -> pinned host + hipStreamSynchronize
// REPS (default 3000) of each after 200 warm-ups, one CPU x card pair at a time; prints the median and p10 in us, then per
// card the L3 groups ranked by the kern median and per CCD (two L3 groups a CCD on Zen 2: cores 4k..4k+3).
//   cpulat [REPS]
// Output lines: "LAT <op> l3=<id> cpu=<n> cpus=<list> card=<bdf> med <us> p10 <us>", "NEAR card=<bdf> ..." and
// "CCDMAP ..." (one line, machine-readable: card=<bdf>:<ccd order nearest first>:<kern medians per CCD>).
#include <hip/hip_runtime.h>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "cpulat: %s failed: %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

__global__ void nop_kernel(int* p) { if (p && threadIdx.x == 1024) p[0] = 1; }

using clk = std::chrono::steady_clock;

static std::string slurp(const std::string& f) {
    std::ifstream in(f);
    std::string s;
    std::getline(in, s);
    return s;
}

struct Stat { double med, p10; };

static Stat stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return Stat{v[v.size() / 2], v[v.size() / 10]};
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3000;
    int n = 0;
    CK(hipGetDeviceCount(&n));
    std::vector<std::string> bdf(n);
    for (int d = 0; d < n; ++d) {
        char bus[64] = {};
        CK(hipDeviceGetPCIBusId(bus, sizeof bus, d));
        bdf[d] = std::string(bus).substr(5, 7);
    }
    // L3 groups: id -> (first cpu, list)
    std::map<int, std::pair<int, std::string>> l3;
    const unsigned ncpu = std::thread::hardware_concurrency();
    for (unsigned c = 0; c < ncpu; ++c) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cache/index3/";
        const std::string id = slurp(base + "id");
        if (id.empty()) continue;
        const int i = std::atoi(id.c_str());
        if (!l3.count(i)) l3[i] = {(int)c, slurp(base + "shared_cpu_list")};
    }
    if (l3.empty()) { std::fprintf(stderr, "cpulat: no L3 topology under /sys/devices/system/cpu\n"); return 2; }
    std::printf("CPULAT cards %d (", n);
    for (int d = 0; d < n; ++d) std::printf("%s%d=%s", d ? " " : "", d, bdf[d].c_str());
    std::printf("), %zu L3 groups, %d reps\n", l3.size(), reps);
    std::fflush(stdout);

    std::map<std::pair<int, int>, Stat> kern, d2h;   // (l3 id, card)
    for (auto& [id, g] : l3) {
        const int cpu = g.first;
        for (int d = 0; d < n; ++d) {
            Stat sk{}, sd{};
            std::thread t([&] {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpu, &set);
                if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) {
                    std::fprintf(stderr, "cpulat: cannot pin to CPU %d\n", cpu); std::exit(2);
                }
                CK(hipSetDevice(d));
                hipStream_t s;
                CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
                int* dev = nullptr; int* host = nullptr;
                CK(hipMalloc(&dev, 64));
                CK(hipHostMalloc(&host, 64, hipHostMallocDefault));
                std::vector<double> vk, vd;
                vk.reserve(reps); vd.reserve(reps);
                for (int i = 0; i < 200 + reps; ++i) {
                    const auto t0 = clk::now();
                    hipLaunchKernelGGL(nop_kernel, dim3(1), dim3(64), 0, s, dev);
                    CK(hipStreamSynchronize(s));
                    const auto t1 = clk::now();
                    CK(hipMemcpyAsync(host, dev, 4, hipMemcpyDeviceToHost, s));
                    CK(hipStreamSynchronize(s));
                    const auto t2 = clk::now();
                    if (i >= 200) {
                        vk.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                        vd.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
                    }
                }
                if (sched_getcpu() != cpu) std::fprintf(stderr, "cpulat: thread left CPU %d\n", cpu);
                sk = stats(vk); sd = stats(vd);
                CK(hipFree(dev)); CK(hipHostFree(host)); CK(hipStreamDestroy(s));
            });
            t.join();
            kern[{id, d}] = sk; d2h[{id, d}] = sd;
            std::printf("LAT kern l3=%d cpu=%d cpus=%s card=%s med %.2f p10 %.2f\n", id, cpu, g.second.c_str(), bdf[d].c_str(), sk.med, sk.p10);
            std::printf("LAT d2h4 l3=%d cpu=%d cpus=%s card=%s med %.2f p10 %.2f\n", id, cpu, g.second.c_str(), bdf[d].c_str(), sd.med, sd.p10);
            std::fflush(stdout);
        }
    }
    // per card: L3 groups by kern median; CCD = L3 groups whose first cpu / 4 is equal (Zen 2: 4 cores a CCD, 2 a CCX)
    std::string ccdmap = "CCDMAP";
    for (int d = 0; d < n; ++d) {
        std::vector<std::pair<double, int>> v;
        for (auto& [id, g] : l3) v.push_back({kern[{id, d}].med, id});
        std::sort(v.begin(), v.end());
        std::printf("NEAR card=%s kern by L3 group:", bdf[d].c_str());
        for (auto& [m, id] : v) std::printf(" l3=%d %.2f", id, m);
        std::printf("\n");
        std::map<int, std::vector<double>> ccd;
        for (auto& [id, g] : l3) ccd[(g.first % 16) / 4].push_back(kern[{id, d}].med);
        std::vector<std::pair<double, int>> cv;
        for (auto& [c, ms] : ccd) {
            double a = 0; for (double x : ms) a += x;
            cv.push_back({a / ms.size(), c});
        }
        std::sort(cv.begin(), cv.end());
        ccdmap += " card=" + bdf[d] + ":";
        for (size_t k = 0; k < cv.size(); ++k) ccdmap += (k ? "," : "") + std::to_string(cv[k].second);
        ccdmap += ":";
        for (size_t k = 0; k < cv.size(); ++k) {
            char b[32]; std::snprintf(b, sizeof b, "%s%.2f", k ? "," : "", cv[k].first); ccdmap += b;
        }
    }
    std::printf("%s\n", ccdmap.c_str());
    std::printf("CPULAT done\n");
    return 0;
}
