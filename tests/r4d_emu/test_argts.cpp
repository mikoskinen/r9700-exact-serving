/* tests/r4d_emu: moe_gather_all_reduce's two-shot form (lever D3, r4d_ar_twoshot_Nrank_exact.hip
 * r4d_ar_gather_twoshot_kernel) on the host SIMT emulator -- its gather, byte for byte against the
 * kernels whose bytes it must keep.
 *
 * The two-shot kernel's phase 1 builds the block's slot list (r4d_ar_gather_list: every row's live
 * slots in ascending slot order, their weights and the shared gate) and gathers each word of its
 * range with r4d_ar_gather_word_list; the reduce and assemble after it are the plain two-shot's, and
 * the transport (peer pushes, flags) needs four cards, which r4d_selftest's
 * moe_gather_all_reduce_4rank case runs. Here the same list and words over a whole message, in one
 * block (as every block of the kernel builds the whole list), are held to
 *   - moe_gather's gather_kernel (r4d_moe_gather.h), what the unfused path runs before its
 *     all-reduce: the self-scan up to 512 slots, the prepass's inverse table above, f32 or bf16
 *     slot weights;
 *   - the one-shot fused form's word, r4d_ar_gather_word over the inverse table;
 * for each rank's partial of an expert-parallel routing (a rank's sort holds only its own experts'
 * slots, stable by local expert then slot, -1 past the live count; one slot in 31 dropped), at
 * DeepSeek-V4.1's shape (n 5120, top_k 6, 384 experts, f32 weights, the ungated shared arm: a gate
 * plane of ones folded as "none") from 1 to 64 rows, and at generic ones: bf16 weights, no shared
 * arm, a sigmoid gate (both sides take the emulator's __expf), top_k 8 and 10, up to the list's
 * 2048 slots. Then the fold the exact all-reduce makes of the four partials (fp32, ascending rank,
 * rounded once) is formed from both sides' partials and compared, so "the same partials" is shown
 * to be "the same reduced bytes". Guards past every output show an overrun. */
#include "rad_abi.h"
#include "r4d_moe_gather.h"
#include "r4d_ar_hc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint16_t kJunk = 0x7e57;

/* The two-shot kernel's phase-1 word loop over every word of the message, in one block. */
__global__ void list_kernel(R4dArGather G, int4* y4, int n16, int wpr) {
    __shared__ int   s_i[R4D_ARGL_TAB];
    __shared__ float s_w[R4D_ARGL_TAB];
    __shared__ int   s_n[R4D_ARGL_ROWS];
    __shared__ float s_g[R4D_ARGL_ROWS];
    r4d_ar_gather_list(G, s_i, s_w, s_n, s_g);
    const int K = G.top_k;
    for (int k = (int)threadIdx.x; k < n16; k += (int)blockDim.x) {
        const int row = k / wpr, i0 = (k - row * wpr) * 8;
        y4[k] = r4d_ar_gather_word_list(G, s_i + row * K, s_w + row * K, s_n[row], G.sh ? s_g[row] : 0.0f,
                                        row, i0);
    }
}

/* The one-shot fused form's word over the inverse table (r4d_ar_gather_word<true>). */
__global__ void inv_word_kernel(R4dArGather G, const int* inv, int4* y4, int n16, int wpr) {
    for (int k = (int)(blockIdx.x * blockDim.x + threadIdx.x); k < n16; k += (int)(blockDim.x * gridDim.x)) {
        const int row = k / wpr, i0 = (k - row * wpr) * 8;
        y4[k] = r4d_ar_gather_word<true>(G, inv, nullptr, nullptr, nullptr, row, 0, i0);
    }
}

float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

struct Shape { int M, n, k, E; bool f32w; int shared; };   /* shared: 0 none, 1 gate "none" (ones), 2 sigmoid */

void run_shape(const Shape& s, std::mt19937& rng, int& cases, int& fails) {
    const int M = s.M, n = s.n, k = s.k, T = M * k, El = s.E / 4, wpr = n / 8, n16 = M * wpr;
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    /* the replicated routing: distinct experts a token, one slot in 31 dropped */
    std::vector<int32_t> ex((size_t)T, -1);
    for (int tok = 0; tok < M; ++tok)
        for (int j = 0; j < k; ++j) {
            if (rng() % 31 == 0) continue;
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                e = (int32_t)(rng() % (uint32_t)s.E);
                dup = false;
                for (int q = 0; q < j; ++q) dup = dup || ex[(size_t)(tok * k + q)] == e;
            }
            ex[(size_t)(tok * k + j)] = e;
        }
    std::vector<float> ewf((size_t)T);
    std::vector<uint16_t> ewb((size_t)T);
    for (int i = 0; i < T; ++i) {
        ewf[(size_t)i] = (float)(rng() % 100000 + 1) / 40000.0f;
        ewb[(size_t)i] = f2b(ewf[(size_t)i]);
    }
    std::vector<std::vector<uint16_t>> part_list(4), part_ref(4);
    bool same = true;
    std::string why;
    for (int r = 0; r < 4; ++r) {
        std::vector<int32_t> sorted((size_t)T, -1), inv((size_t)T, -1);
        int pos = 0;
        for (int32_t e = r * El; e < (r + 1) * El; ++e)
            for (int i = 0; i < T; ++i)
                if (ex[(size_t)i] == e) sorted[(size_t)pos++] = i;
        for (int i = 0; i < pos; ++i) inv[(size_t)sorted[(size_t)i]] = i;
        std::vector<uint16_t> ye((size_t)T * n), sh, sg;
        for (auto& v : ye) v = f2b(ud(rng));
        if (s.shared) {
            sh.resize((size_t)M * n);
            sg.resize((size_t)M);
            for (auto& v : sh) v = f2b(ud(rng) * 2.0f);
            for (auto& v : sg) v = s.shared == 1 ? f2b(1.0f) : f2b(ud(rng));
        }
        const int sig = s.shared == 2;
        R4dArGather G;
        std::memset(&G, 0, sizeof G);
        G.ye = ye.data();
        G.ew = s.f32w ? reinterpret_cast<const unsigned short*>(ewf.data()) : ewb.data();
        G.sorted = sorted.data();
        G.sh = s.shared ? sh.data() : nullptr;
        G.sg = s.shared ? sg.data() : nullptr;
        G.ye_ld = n; G.ew_ld = k; G.sh_ld = n; G.sg_ld = 1;
        G.top_k = k; G.T = T; G.sig = sig; G.ew_f32 = s.f32w ? 1 : 0;

        /* the two-shot's list words, 8 guard words past the end */
        std::vector<uint16_t> yl((size_t)(n16 + 8) * 8, kJunk);
        R4D_EMIT(nullptr, list_kernel, dim3(1), dim3(256), 0, G, reinterpret_cast<int4*>(yl.data()), n16, wpr);
        /* the one-shot's word over the inverse table */
        std::vector<uint16_t> yi((size_t)(n16 + 8) * 8, kJunk);
        R4D_EMIT(nullptr, inv_word_kernel, dim3((unsigned)((n16 + 255) / 256)), dim3(256), 0, G, inv.data(),
                 reinterpret_cast<int4*>(yi.data()), n16, wpr);
        /* moe_gather's kernel, as its launcher picks: self-scan up to 512 slots, the prepass's table above */
        std::vector<uint16_t> yg((size_t)M * n + 64, kJunk);
        const bool small = T <= (int)r4d_moegather::kGatherSelfT;
        const int32_t* tab = small ? nullptr : inv.data();
        const int32_t* srt = small ? sorted.data() : nullptr;
        const void* ew = s.f32w ? (const void*)ewf.data() : (const void*)ewb.data();
        const unsigned gy = (unsigned)((n + 255) / 256);
        if (s.f32w)
            R4D_EMIT(nullptr, (r4d_moegather::gather_kernel<256, true>), dim3((unsigned)M, gy), dim3(256), 0,
                     ye.data(), ew, tab, yg.data(), (uint32_t)M, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                     (uint32_t)k, (uint32_t)n, srt, (uint32_t)T, G.sh, G.sg, (uint32_t)n, 1u, sig);
        else
            R4D_EMIT(nullptr, (r4d_moegather::gather_kernel<256, false>), dim3((unsigned)M, gy), dim3(256), 0,
                     ye.data(), ew, tab, yg.data(), (uint32_t)M, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                     (uint32_t)k, (uint32_t)n, srt, (uint32_t)T, G.sh, G.sg, (uint32_t)n, 1u, sig);
        auto first_diff = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
            for (size_t i = 0; i < (size_t)M * n; ++i) if (a[i] != b[i]) return (long)i;
            return -1L;
        };
        auto guards = [&](const std::vector<uint16_t>& a, size_t from) {
            for (size_t i = from; i < a.size(); ++i) if (a[i] != kJunk) return false;
            return true;
        };
        /* every element written: a kernel that stored nothing would leave the junk on both sides */
        auto unwritten = [&](const std::vector<uint16_t>& a) {
            long u = 0;
            for (size_t i = 0; i < (size_t)M * n; ++i) u += a[i] == kJunk;
            return u > (long)M * n / 100;
        };
        const long d1 = first_diff(yl, yg), d2 = first_diff(yl, yi);
        if (d1 >= 0 || d2 >= 0 || !guards(yl, (size_t)M * n) || !guards(yi, (size_t)M * n) ||
            !guards(yg, (size_t)M * n) || unwritten(yl) || unwritten(yg)) {
            same = false;
            char b[200];
            std::snprintf(b, sizeof b, " rank %d: vs moe_gather first diff %ld, vs the one-shot word %ld, guards %d%d%d, "
                          "unwritten %d%d", r, d1, d2, (int)guards(yl, (size_t)M * n), (int)guards(yi, (size_t)M * n),
                          (int)guards(yg, (size_t)M * n), (int)unwritten(yl), (int)unwritten(yg));
            why += b;
        }
        part_list[(size_t)r].assign(yl.begin(), yl.begin() + (long)M * n);
        part_ref[(size_t)r].assign(yg.begin(), yg.begin() + (long)M * n);
    }
    /* the exact all-reduce's fold of each side's partials */
    long fd = -1;
    for (long i = 0; i < (long)M * n && fd < 0; ++i) {
        float a = 0.0f, b = 0.0f;
        for (int r = 0; r < 4; ++r) { a += b2f(part_list[(size_t)r][(size_t)i]); b += b2f(part_ref[(size_t)r][(size_t)i]); }
        if (f2b(a) != f2b(b)) fd = i;
    }
    ++cases;
    if (!same || fd >= 0) {
        ++fails;
        std::printf("FAIL two-shot gather M=%d n=%d top_k=%d %s weights shared %d:%s; fold first diff %ld\n", M, n, k,
                    s.f32w ? "f32" : "bf16", s.shared, why.c_str(), fd);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(41);
    int cases = 0, fails = 0;
    std::vector<Shape> shapes;
    for (int M : { 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 47, 48, 64 }) shapes.push_back({ M, 5120, 6, 384, true, 1 });
    for (int M : { 3, 17, 40 }) {
        shapes.push_back({ M, 512, 8, 64, false, 0 });
        shapes.push_back({ M, 512, 8, 64, true, 0 });
        shapes.push_back({ M, 512, 8, 64, false, 1 });
        shapes.push_back({ M, 512, 8, 64, true, 2 });
        shapes.push_back({ M, 512, 8, 64, false, 2 });
    }
    shapes.push_back({ 100, 256, 8, 64, true, 1 });     /* 800 slots: moe_gather's prepass table */
    shapes.push_back({ 204, 256, 10, 80, false, 2 });   /* 2040 slots, near the list's bound */
    shapes.push_back({ 256, 128, 8, 32, true, 1 });     /* the list's row bound */
    for (const Shape& s : shapes) run_shape(s, rng, cases, fails);
    std::printf("%d case(s) (%d shapes x 4 ranks' partials, each three ways), %d failure(s)\n", cases,
                (int)shapes.size(), fails);
    return fails ? 1 : 0;
}
