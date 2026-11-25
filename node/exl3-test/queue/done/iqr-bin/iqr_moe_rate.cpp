/* iqr_moe_rate.cpp -- time libquant's GPTQ + trellis over real routed experts, exactly as
 * rad-convert runs one large weight (quant_fixture.h: the converter's own quantise loop).
 *
 *   iqr_moe_rate DIR LAYER PROJ EXPERTS OPTS [WORKERS]
 *     DIR      gu.<e>.f32 [4608, 5120] / dn.<e>.f32 [5120, 2304] and calib/ (RADGRAM2 files)
 *     PROJ     gu | dn
 *     EXPERTS  comma-separated expert numbers
 *     OPTS     the recipe rule's options, comma-separated k=v (calib=DIR/calib is added)
 *     WORKERS  weights quantised side by side, each with OMP threads / WORKERS (default 1, as
 *              rad-convert's host path runs its large weights one at a time)
 *   IQR_GPUS=0,1,2,3  the cards (rad-convert --gpus): libquant's rad_quant_devices hook, as
 *              rad-convert calls it. A weight of fewer than 32 runs goes whole to the next card
 *              in turn, so WORKERS > cards puts several weights on a card at once.
 *
 * Prints each weight's wall time and relative weight error, then the throughput.
 */
#include "quant_fixture.h"

#include <dlfcn.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

using namespace qfix;

static std::vector<float> load(const std::string& p, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(p, std::ios::binary);
    f.read((char*)v.data(), (std::streamsize)(n * 4));
    if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(2); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: see the header\n"); return 2; }
    const std::string dir = argv[1];
    const int layer = std::atoi(argv[2]);
    const std::string proj = argv[3];
    std::vector<int> ex;
    { std::stringstream s(argv[4]); std::string t; while (std::getline(s, t, ',')) ex.push_back(std::stoi(t)); }
    const std::string opts = argv[5];
    const int workers = argc > 6 ? std::atoi(argv[6]) : 1;
    const bool gu = proj == "gu";
    const int64_t N = gu ? 4608 : 5120, K = gu ? 5120 : 2304;
    const RadQuantizerInfo* q = quantizer("gptq");
    if (!q) { std::fprintf(stderr, "no libquant under %s\n", home().c_str()); return 2; }
    static const std::set<std::string> ints = { "tcq", "tcq_rows", "tcq_colf", "tcq_order", "dyn",
                                                "cd", "rescale", "act_order", "tcq_rb", "gpu" };
    static const std::set<std::string> f64s = { "tcq_gain", "damp" };
    auto make = [&]() {
        Opts o;
        std::stringstream s(opts);
        std::string kv;
        while (std::getline(s, kv, ',')) {
            const size_t e = kv.find('=');
            const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
            if (ints.count(k)) o.i(k.c_str(), std::stoll(v));
            else if (f64s.count(k)) o.f(k.c_str(), std::stod(v));
            else o.s(k.c_str(), v.c_str());
        }
        o.s("calib", (dir + "/calib").c_str());
        return o;
    };
    if (const char* gl = std::getenv("IQR_GPUS")) {
        std::vector<int> ids;
        std::stringstream s(gl);
        std::string t;
        while (std::getline(s, t, ',')) ids.push_back(std::stoi(t));
        const rad::Plugin* p = reg().plugin("libquant");
        const RadQuantDevicesFn fn =
            p && p->handle ? (RadQuantDevicesFn)dlsym(p->handle, "rad_quant_devices") : nullptr;
        const int st = fn ? fn(ids.data(), (int)ids.size()) : RAD_E_UNSUPPORTED;
        std::printf("cards %s: rad_quant_devices %d\n", gl, st);
        if (st != RAD_OK) return 3;
    }
    const int threads = omp_get_max_threads();
    std::printf("layer %d %s [%lld x %lld], %zu weight(s), %d worker(s) x %d thread(s): %s\n", layer,
                proj.c_str(), (long long)N, (long long)K, ex.size(), workers,
                std::max(1, threads / workers), opts.c_str());
    std::fflush(stdout);
    std::atomic<size_t> next{0};
    std::mutex mu;
    const auto t0 = std::chrono::steady_clock::now();
    auto work = [&] {
        omp_set_num_threads(std::max(1, threads / workers));
        for (size_t i; (i = next.fetch_add(1)) < ex.size();) {
            const int e = ex[i];
            const std::vector<float> w =
                load(dir + (gu ? "/gu." : "/dn.") + std::to_string(e) + ".f32", (size_t)(N * K));
            const std::string name = "blk." + std::to_string(layer) +
                                     (gu ? ".ffn_gate_up_exps." : ".ffn_down_exps.") +
                                     std::to_string(e) + ".weight";
            Opts o = make();
            const auto a = std::chrono::steady_clock::now();
            const Quantised r = quantise(q, o, name.c_str(), N, K, w);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            double err = -1;
            int64_t bytes = 0;
            uint64_t h = 0xcbf29ce484222325ull;
            for (auto& p : r.planes) { bytes += (int64_t)p.size(); h = fnv(p, h); }
            if (r.status == RAD_OK) {
                const std::vector<float> d = decode(r, N, K);
                if (!d.empty()) err = rel_err(w, d);
            }
            std::lock_guard<std::mutex> lk(mu);
            std::printf("  %s: status %d, %.1f s, %.4f bpw, rel. weight error %.6f, planes fnv %016llx\n",
                        name.c_str(), r.status, s, bytes * 8.0 / (double)(N * K), err,
                        (unsigned long long)h);
            std::fflush(stdout);
        }
    };
    std::vector<std::thread> th;
    for (int k = 1; k < workers; ++k) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    const double tot = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("total %.1f s for %zu weight(s): %.2f s a weight\n", tot, ex.size(), tot / (double)ex.size());
    return 0;
}
