/* tests/r4d_emu: lever P (ds41-p1pass) -- the MoE gather of a 17 .. 48-row decode step as ONE pass against today's
 * pieces of <= 16 rows, on the host SIMT emulator, byte for byte per token.
 *
 * Under ds41_moe1 a step that is one decode segment of 17 .. 48 rows sorts its routing ONCE (a rank's sort: its own
 * experts' slots, stable by local expert then flat slot index) and gathers ONCE -- moe_gather_all_reduce's two-shot form
 * (the step is one pass of <= RADIANCE_DS41_ARGATHER_MAX rows), or moe_gather + the all-reduce -- where today each
 * <= 16-row piece sorts and gathers its own rows (moe_gather) before one all-reduce over the step. The routed experts'
 * rows are the same bytes either way (exl3r4d's _dk rows; tools/dkcheck on the card), in a different ORDER: a row sits
 * at its place in the pass's sort. What must hold is that a token's partial does not depend on which other tokens share
 * its pass: its slots are folded in ascending sorted position, which is ascending local expert in either sort.
 *
 * For every rank of an expert-parallel routing at DeepSeek-V4.1's shape (384 experts, top 6, f32 weights, the ungated
 * shared arm; n 5120 at M 17 / 24 / 31 / 32 / 33 / 47 / 48, n 512 at every M 17 .. 48; one slot in 31 dropped; two
 * routings an M, the second with tokens in runs of 4 sharing five of their six experts -- a verify pass): each
 * (token, slot)'s expert row and weight are a function of (token, slot) alone, laid out in the pass's sorted order, and
 *   A  the ONE pass through the two-shot's list words (r4d_ar_gather_list / r4d_ar_gather_word_list),
 *   B  the ONE pass through moe_gather's gather_kernel,
 *   C  the pieces [0, 16), [16, 32), [32, 48) each through moe_gather's gather_kernel (today),
 * must give the same bytes for every token (A == C and B == C), and so the exact all-reduce's fold of the four ranks'
 * partials (fp32, ascending rank, rounded once) is the same.
 *
 *   r4d_emu_ds41p1 */
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
constexpr int kPiece = 16;

/* The two-shot kernel's phase-1 word loop over every word of the message, in one block (test_argts.cpp's). */
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

float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
/* element i of (token t, slot j)'s expert row / of token t's shared row: of (case seed, t, j, i) alone */
float val(uint64_t seed, int t, int j, int i) {
    const uint64_t h = mix(seed ^ mix((uint64_t)t * 1000003ull + (uint64_t)j * 7919ull + 1) ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull));
    return ((float)((h >> 40) & 0xffffffu) / 16777216.0f - 0.5f) * 8.0f;
}

struct Pass {   /* tokens [a, a + m) of the step as one pass on rank r: its sort and its operands */
    std::vector<int32_t> sorted, inv;
    std::vector<uint16_t> ye, sh, sg;
    std::vector<float> ew;
};
Pass make_pass(uint64_t seed, const std::vector<int32_t>& ex, int a, int m, int n, int k, int r, int El) {
    Pass p;
    const int T = m * k;
    p.sorted.assign((size_t)T, -1);
    p.inv.assign((size_t)T, -1);
    int pos = 0;
    for (int32_t e = r * El; e < (r + 1) * El; ++e)
        for (int i = 0; i < T; ++i)
            if (ex[(size_t)(a * k + i)] == e) p.sorted[(size_t)pos++] = i;
    for (int i = 0; i < pos; ++i) p.inv[(size_t)p.sorted[(size_t)i]] = i;
    p.ye.assign((size_t)T * n, kJunk);   /* the rows past the live count are never read */
    for (int q = 0; q < pos; ++q) {
        const int f = p.sorted[(size_t)q], t = a + f / k, j = f % k;
        for (int i = 0; i < n; ++i) p.ye[(size_t)q * n + i] = f2b(val(seed, t, j, i));
    }
    p.ew.resize((size_t)T);
    for (int f = 0; f < T; ++f) p.ew[(size_t)f] = 0.05f + std::fabs(val(seed ^ 0x77, a + f / k, f % k, 0)) / 3.0f;
    p.sh.resize((size_t)m * n);
    p.sg.assign((size_t)m, f2b(1.0f));
    for (int t = 0; t < m; ++t)
        for (int i = 0; i < n; ++i) p.sh[(size_t)t * n + i] = f2b(val(seed ^ 0x5151, a + t, 99, i) * 2.0f);
    return p;
}
R4dArGather args_of(const Pass& p, int m, int n, int k) {
    R4dArGather G;
    std::memset(&G, 0, sizeof G);
    G.ye = p.ye.data();
    G.ew = reinterpret_cast<const unsigned short*>(p.ew.data());
    G.sorted = p.sorted.data();
    G.sh = p.sh.data();
    G.sg = p.sg.data();
    G.ye_ld = n; G.ew_ld = k; G.sh_ld = n; G.sg_ld = 1;
    G.top_k = k; G.T = m * k; G.sig = 0; G.ew_f32 = 1;
    return G;
}
/* moe_gather's kernel over a pass, as its launcher picks (self-scan up to 512 slots): [m][n] bf16 */
std::vector<uint16_t> gather(const Pass& p, int m, int n, int k) {
    const int T = m * k;
    std::vector<uint16_t> y((size_t)m * n + 64, kJunk);
    const bool small = T <= (int)r4d_moegather::kGatherSelfT;
    R4D_EMIT(nullptr, (r4d_moegather::gather_kernel<256, true>), dim3((unsigned)m, (unsigned)((n + 255) / 256)), dim3(256), 0,
             p.ye.data(), (const void*)p.ew.data(), small ? nullptr : p.inv.data(), y.data(), (uint32_t)m, (uint32_t)n,
             (uint32_t)k, (uint32_t)n, (uint32_t)k, (uint32_t)n, small ? p.sorted.data() : nullptr, (uint32_t)T,
             p.sh.data(), p.sg.data(), (uint32_t)n, 1u, 0);
    for (size_t i = (size_t)m * n; i < y.size(); ++i)
        if (y[i] != kJunk) { std::printf("FAIL moe_gather wrote past its rows (m %d)\n", m); std::exit(1); }
    y.resize((size_t)m * n);
    return y;
}
std::vector<uint16_t> list_words(const Pass& p, int m, int n, int k) {
    const int wpr = n / 8, n16 = m * wpr;
    std::vector<uint16_t> y((size_t)(n16 + 8) * 8, kJunk);
    R4D_EMIT(nullptr, list_kernel, dim3(1), dim3(256), 0, args_of(p, m, n, k), reinterpret_cast<int4*>(y.data()), n16, wpr);
    for (size_t i = (size_t)m * n; i < y.size(); ++i)
        if (y[i] != kJunk) { std::printf("FAIL the list words wrote past their rows (m %d)\n", m); std::exit(1); }
    y.resize((size_t)m * n);
    return y;
}

void run(int M, int n, int grouped, std::mt19937& rng, int& cases, int& fails, long& tokens) {
    const int k = 6, E = 384, El = E / 4, T = M * k;
    const uint64_t seed = mix((uint64_t)M * 131 + (uint64_t)n * 7 + (uint64_t)grouped);
    std::vector<int32_t> ex((size_t)T, -1);
    for (int tok = 0; tok < M; ++tok)
        for (int j = 0; j < k; ++j) {
            if (grouped && tok % 4 != 0 && j < 5) {   /* a verify run: five of the six experts are the run's first token's */
                ex[(size_t)(tok * k + j)] = ex[(size_t)((tok - tok % 4) * k + j)];
                continue;
            }
            if (rng() % 31 == 0) continue;
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                e = (int32_t)(rng() % (uint32_t)E);
                dup = false;
                for (int q = 0; q < j; ++q) dup = dup || ex[(size_t)(tok * k + q)] == e;
            }
            ex[(size_t)(tok * k + j)] = e;
        }
    std::vector<std::vector<uint16_t>> pa(4), pb(4), pc(4);
    std::string why;
    for (int r = 0; r < 4; ++r) {
        const Pass full = make_pass(seed, ex, 0, M, n, k, r, El);
        pa[(size_t)r] = list_words(full, M, n, k);
        pb[(size_t)r] = gather(full, M, n, k);
        for (int a = 0; a < M; a += kPiece) {
            const int m = std::min(kPiece, M - a);
            const std::vector<uint16_t> y = gather(make_pass(seed, ex, a, m, n, k, r, El), m, n, k);
            pc[(size_t)r].insert(pc[(size_t)r].end(), y.begin(), y.end());
        }
        long unwritten = 0;
        for (uint16_t v : pc[(size_t)r]) unwritten += v == kJunk;
        for (int t = 0; t < M; ++t) {
            const bool da = std::memcmp(&pa[(size_t)r][(size_t)t * n], &pc[(size_t)r][(size_t)t * n], (size_t)n * 2) != 0;
            const bool db = std::memcmp(&pb[(size_t)r][(size_t)t * n], &pc[(size_t)r][(size_t)t * n], (size_t)n * 2) != 0;
            if ((da || db) && why.size() < 300) {
                char b[96];
                std::snprintf(b, sizeof b, " rank %d token %d:%s%s", r, t, da ? " list!=pieces" : "", db ? " gather!=pieces" : "");
                why += b;
            }
        }
        if (unwritten > (long)M * n / 100) why += " (pieces left unwritten)";
    }
    long fd = -1;   /* the exact all-reduce's fold of the four partials: ONE pass (list) vs the pieces */
    for (long i = 0; i < (long)M * n && fd < 0; ++i) {
        float a = 0.0f, c = 0.0f;
        for (int r = 0; r < 4; ++r) { a += b2f(pa[(size_t)r][(size_t)i]); c += b2f(pc[(size_t)r][(size_t)i]); }
        if (f2b(a) != f2b(c)) fd = i;
    }
    ++cases;
    tokens += 4L * M;
    const bool ok = why.empty() && fd < 0;
    if (!ok) ++fails;
    std::printf("%s lever P gather M=%d n=%d %s: ONE pass (two-shot list words, moe_gather) vs pieces of <= %d, 4 ranks%s%s\n",
                ok ? "ok  " : "FAIL", M, n, grouped ? "grouped" : "drawn", kPiece, why.c_str(),
                fd >= 0 ? "; the reduced fold differs" : "");
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::mt19937 rng(4117);
    int cases = 0, fails = 0;
    long tokens = 0;
    for (int M : { 17, 24, 31, 32, 33, 47, 48 })
        for (int g = 0; g < 2; ++g) run(M, 5120, g, rng, cases, fails, tokens);
    for (int M = 17; M <= 48; ++M)
        for (int g = 0; g < 2; ++g) run(M, 512, g, rng, cases, fails, tokens);
    std::printf("%d case(s), %ld rank-token partials compared two ways, %d failure(s)\n", cases, tokens, fails);
    return fails ? 1 : 0;
}
