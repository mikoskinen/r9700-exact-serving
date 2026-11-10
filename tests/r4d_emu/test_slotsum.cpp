/* tests/r4d_emu: lever D3s on the host SIMT emulator -- moe_gather_all_reduce_slot's slot-order sum
 * (r4d_ar_slotsum.hip, its words in r4d_ar_hc.h) and moe_scatter_placed's dispatch and sort
 * (r4d_moe_place.hip), byte for byte.
 *
 * The sum. The transport's handshakes need four cards (r4d_selftest's moe_gather_all_reduce_slot_4rank
 * case runs them); here every rank's phase 1 (r4d_ar_slot_table + r4d_ar_slot_push) writes into the
 * planes the kernels lay out -- the one-shot's K + 4 planes of the whole message, the two-shot's
 * K + 4 planes of each owner's shard -- and the reducer's word (r4d_ar_slot_sum_word) sums them, as
 * the kernels do after the handshake. Held to
 *   - the slot-order oracle on the host: acc = the ranks' bf16(shared * gate) in ascending rank
 *     order, then every slot's row times its weight in ascending slot order (fmaf with f32 weights,
 *     multiply then add with bf16 ones), rounded once;
 *   - ITSELF across placements: the same routing and the same expert rows served by the contiguous
 *     split, a scrambled one (every expert on a random rank), and a replicated one (some experts on
 *     two to four ranks, each slot served by the copy moe_scatter_placed's dispatch picks), so the
 *     slot rows sit on different ranks, in different sorted positions; one-shot and two-shot alike.
 *   - every plane word a live slot names written (junk left over would show), guards past y.
 * At DeepSeek-V4.1's shape (n 5120, top_k 6, 384 experts, f32 weights, the ungated shared arm: a gate
 * of ones) from 1 to 64 rows, and generic ones (bf16 weights, no shared arm, a sigmoid gate).
 *
 * The dispatch. For each placement and routing, every rank's place_scatter_kernel output: its slots
 * are exactly those r4d_place_owner (the host reference) gives it, every slot lands on exactly one
 * rank, its sort is stable by local expert then slot with offsets and counts that agree; and at the
 * contiguous placement (no extras) the bytes equal moe_scatter's own small sort (scatter_small_kernel
 * with expert_base), which today's EP layer runs. */
#include "rad_abi.h"
#include "r4d_ar_hc.h"
#include "r4d_ar_shards.h"
#include "r4d_moe_sort.h"
#include "r4d_moe_place.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint16_t kJunk = 0x7e57;
constexpr int WS = 4;

float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

/* One rank's phase 1 over the words [w0, w1) of shard o (offset `off`): what the kernels' loops do
 * before the handshake, into the planes at P (stride ps). One block, as each kernel block builds the
 * whole table. */
__global__ void push_kernel(R4dArGather G, int4* P, long ps, int off, int w0, int w1, int wpr, int rank) {
    __shared__ int   s_i[R4D_ARGL_TAB];
    __shared__ float s_w[R4D_ARGL_TAB];
    __shared__ float s_g[R4D_ARGL_ROWS];
    r4d_ar_slot_table(G, s_i, s_w, s_g);
    for (int k = w0 + (int)threadIdx.x; k < w1; k += (int)blockDim.x) {
        const int gw = off + k, row = gw / wpr, i0 = (gw - row * wpr) * 8;
        r4d_ar_slot_push(G, s_i, s_g, row, i0, rank, P, ps, k);
    }
}

/* The reducer's words [w0, w1) of shard `off` from its planes, into y4 at off + k. */
__global__ void sum_kernel(R4dArGather G, const int4* P, long ps, int off, int w0, int w1, int wpr, int4* y4) {
    __shared__ int   s_i[R4D_ARGL_TAB];
    __shared__ float s_w[R4D_ARGL_TAB];
    __shared__ float s_g[R4D_ARGL_ROWS];
    r4d_ar_slot_table(G, s_i, s_w, s_g);
    const int K = G.top_k;
    for (int k = w0 + (int)threadIdx.x; k < w1; k += (int)blockDim.x) {
        const int row = (off + k) / wpr;
        y4[off + k] = r4d_ar_slot_sum_word<WS>(P, ps, k, K, G.sh != nullptr, s_w + row * K, G.ew_f32);
    }
}

struct Shape { int M, n, k, E; bool f32w; int shared; };   /* shared: 0 none, 1 gate of ones, 2 sigmoid */

/* A placement: each model expert's copy mask over WS ranks, and each rank's extras. */
struct Place {
    const char* name;
    std::vector<int32_t> mask;
    std::vector<std::vector<int32_t>> extras;
};

std::string hexmask(const Place& p) {
    std::string s;
    for (int32_t m : p.mask) s += "0123456789abcdef"[m & 15];
    return s;
}
std::string commas(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

/* The three placements of E experts: contiguous; scrambled (each expert on one random rank, at home
 * where the draw says so); replicated (contiguous, plus every 5th expert copied to one to three more
 * ranks). Extras = an expert a rank holds that is not its home. */
std::vector<Place> placements(int E, std::mt19937& rng) {
    const int nh = E / WS;
    std::vector<Place> v;
    Place c{ "contiguous", std::vector<int32_t>((size_t)E), std::vector<std::vector<int32_t>>(WS) };
    for (int e = 0; e < E; ++e) c.mask[(size_t)e] = 1 << (e / nh);
    v.push_back(c);
    Place s{ "scrambled", std::vector<int32_t>((size_t)E), std::vector<std::vector<int32_t>>(WS) };
    for (int e = 0; e < E; ++e) s.mask[(size_t)e] = 1 << (int)(rng() % WS);
    v.push_back(s);
    Place r{ "replicated", std::vector<int32_t>((size_t)E), std::vector<std::vector<int32_t>>(WS) };
    for (int e = 0; e < E; ++e) {
        int32_t m = 1 << (e / nh);
        if (e % 5 == 0) m |= (int32_t)(rng() % 16);
        r.mask[(size_t)e] = m;
    }
    v.push_back(r);
    for (Place& p : v)
        for (int e = 0; e < E; ++e)
            for (int q = 0; q < WS; ++q)
                if (((p.mask[(size_t)e] >> q) & 1) && e / nh != q) p.extras[(size_t)q].push_back(e);
    return v;
}

/* Every rank's sort as the placed scatter's reference: the slots r4d_place_owner gives the rank, by
 * local expert then slot. Returns false (with `why`) when the table refuses. */
bool ref_sorts(const Place& p, int E, const std::vector<int32_t>& ex, std::vector<R4dPlaceTab>& tabs,
               std::vector<std::vector<int32_t>>& sorted, std::vector<int32_t>& owner, std::string& why) {
    const int nh = E / WS, T = (int)ex.size();
    tabs.assign(WS, R4dPlaceTab{});
    sorted.assign(WS, std::vector<int32_t>((size_t)T, -1));
    owner.assign((size_t)T, -1);
    const std::string mk = hexmask(p);
    for (int q = 0; q < WS; ++q) {
        const std::string xs = commas(p.extras[(size_t)q]);
        const long long nl = nh + (long long)p.extras[(size_t)q].size();
        if (r4d_place_parse(mk.c_str(), xs.c_str(), nullptr, E, nh, nl, WS, q, &tabs[(size_t)q], &why) != RAD_OK)
            return false;
    }
    r4d_place_owner(tabs[0], ex.data(), T, owner.data());
    for (int q = 0; q < WS; ++q) {
        const int32_t* loc = tabs[(size_t)q].tab.data() + E;
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (int i = 0; i < T; ++i)
            if (owner[(size_t)i] == q) mine.push_back({ loc[ex[(size_t)i]], i });
        std::stable_sort(mine.begin(), mine.end());
        for (size_t x = 0; x < mine.size(); ++x) sorted[(size_t)q][x] = mine[x].second;
    }
    return true;
}

void run_shape(const Shape& s, std::mt19937& rng, int& cases, int& fails) {
    const int M = s.M, n = s.n, k = s.k, T = M * k, E = s.E, wpr = n / 8, n16 = M * wpr;
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    /* the routing: k distinct experts a token, every slot live */
    std::vector<int32_t> ex((size_t)T, -1);
    for (int tok = 0; tok < M; ++tok)
        for (int j = 0; j < k; ++j) {
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                e = (int32_t)(rng() % (uint32_t)E);
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
    /* each slot's routed row: whichever rank computes it, these bytes */
    std::vector<uint16_t> row((size_t)T * n);
    for (auto& v : row) v = f2b(ud(rng));
    /* each rank's shared slice and the replicated gate */
    std::vector<std::vector<uint16_t>> sh(WS);
    std::vector<uint16_t> sg;
    if (s.shared) {
        for (auto& r : sh) { r.resize((size_t)M * n); for (auto& v : r) v = f2b(ud(rng) * 2.0f); }
        sg.resize((size_t)M);
        for (auto& v : sg) v = s.shared == 1 ? f2b(1.0f) : f2b(ud(rng));
    }
    const int sig = s.shared == 2;
    auto gate_of = [&](int tok) { const float v0 = b2f(sg[(size_t)tok]); return sig ? 1.0f / (1.0f + std::exp(-v0)) : v0; };

    /* the oracle */
    std::vector<uint16_t> want((size_t)M * n);
    for (int tok = 0; tok < M; ++tok)
        for (int e = 0; e < n; ++e) {
            float acc = 0.0f;
            if (s.shared)
                for (int r = 0; r < WS; ++r) acc += b2f(f2b(b2f(sh[(size_t)r][(size_t)(tok * n + e)]) * gate_of(tok)));
            for (int j = 0; j < k; ++j) {
                const float yv = b2f(row[(size_t)(tok * k + j) * n + e]);
                if (s.f32w) acc = std::fmaf(ewf[(size_t)(tok * k + j)], yv, acc);
                else { const float pr = b2f(ewb[(size_t)(tok * k + j)]) * yv; acc = acc + pr; }
            }
            want[(size_t)(tok * n + e)] = f2b(acc);
        }

    const std::vector<Place> pls = placements(E, rng);
    std::vector<uint16_t> first;   /* placement 0's bytes */
    std::string why;
    bool ok = true;
    for (const Place& pl : pls) {
        std::vector<R4dPlaceTab> tabs;
        std::vector<std::vector<int32_t>> sorted;
        std::vector<int32_t> owner;
        std::string pw;
        if (!ref_sorts(pl, E, ex, tabs, sorted, owner, pw)) { ok = false; why += std::string(" ") + pl.name + ": table " + pw; continue; }
        /* each rank's routed rows in its sorted order, its gather operands */
        std::vector<std::vector<uint16_t>> ye(WS);
        std::vector<R4dArGather> G(WS);
        for (int q = 0; q < WS; ++q) {
            ye[(size_t)q].assign((size_t)T * n, kJunk);
            for (int i = 0; i < T; ++i) {
                const int32_t f = sorted[(size_t)q][(size_t)i];
                if (f < 0) break;
                std::memcpy(&ye[(size_t)q][(size_t)i * n], &row[(size_t)f * n], (size_t)n * 2);
            }
            R4dArGather& g = G[(size_t)q];
            std::memset(&g, 0, sizeof g);
            g.ye = ye[(size_t)q].data();
            g.ew = s.f32w ? reinterpret_cast<const unsigned short*>(ewf.data()) : ewb.data();
            g.sorted = sorted[(size_t)q].data();
            g.sh = s.shared ? sh[(size_t)q].data() : nullptr;
            g.sg = s.shared ? sg.data() : nullptr;
            g.ye_ld = n; g.ew_ld = k; g.sh_ld = n; g.sg_ld = 1;
            g.top_k = k; g.T = T; g.sig = sig; g.ew_f32 = s.f32w ? 1 : 0;
        }
        const int planes1 = R4D_ARS_ONESHOT_PLANES(k, WS);
        /* one-shot: every rank writes the whole message into one set of planes (each rank's own copy
         * is the same bytes), then a reducer sums every word */
        std::vector<int4> P1((size_t)planes1 * n16);
        for (auto& w : P1) { uint16_t* h = reinterpret_cast<uint16_t*>(&w); for (int e = 0; e < 8; ++e) h[e] = kJunk; }
        for (int q = 0; q < WS; ++q)
            R4D_EMIT(nullptr, push_kernel, dim3(1), dim3(256), 0, G[(size_t)q], P1.data(), (long)n16, 0, 0, n16, wpr, q);
        long unw = 0;
        for (int j = 0; j < planes1; ++j) {
            if (j >= k && !s.shared) continue;
            for (int x = 0; x < n16; ++x) {
                const uint16_t* h = reinterpret_cast<const uint16_t*>(&P1[(size_t)j * n16 + x]);
                bool junk = true;
                for (int e = 0; e < 8; ++e) junk = junk && h[e] == kJunk;
                unw += junk;
            }
        }
        std::vector<uint16_t> y1((size_t)(n16 + 8) * 8, kJunk);
        R4D_EMIT(nullptr, sum_kernel, dim3(1), dim3(256), 0, G[0], P1.data(), (long)n16, 0, 0, n16, wpr,
                 reinterpret_cast<int4*>(y1.data()));
        /* two-shot: the planes of each owner's shard */
        long off[WS + 1];
        const long sh_max = r4d_ar_shard_table(n16, WS, nullptr, off);
        const int planes2 = R4D_ARS_TWOSHOT_PLANES(k, WS);
        std::vector<uint16_t> y2((size_t)(n16 + 8) * 8, kJunk);
        for (int o = 0; o < WS; ++o) {
            std::vector<int4> P2((size_t)planes2 * sh_max);
            for (auto& w : P2) { uint16_t* h = reinterpret_cast<uint16_t*>(&w); for (int e = 0; e < 8; ++e) h[e] = kJunk; }
            const int len = (int)(off[o + 1] - off[o]);
            for (int q = 0; q < WS; ++q)
                R4D_EMIT(nullptr, push_kernel, dim3(1), dim3(256), 0, G[(size_t)q], P2.data(), (long)sh_max,
                         (int)off[o], 0, len, wpr, q);
            R4D_EMIT(nullptr, sum_kernel, dim3(1), dim3(256), 0, G[(size_t)o], P2.data(), (long)sh_max, (int)off[o],
                     0, len, wpr, reinterpret_cast<int4*>(y2.data()));
        }
        long d1 = -1, d2 = -1, d0 = -1;
        for (long i = 0; i < (long)M * n; ++i) {
            if (d1 < 0 && y1[(size_t)i] != want[(size_t)i]) d1 = i;
            if (d2 < 0 && y2[(size_t)i] != want[(size_t)i]) d2 = i;
        }
        bool guards = true;
        for (size_t i = (size_t)M * n; i < y1.size(); ++i) guards = guards && y1[i] == kJunk && y2[i] == kJunk;
        if (first.empty()) first.assign(y1.begin(), y1.begin() + (long)M * n);
        else for (long i = 0; i < (long)M * n && d0 < 0; ++i) if (y1[(size_t)i] != first[(size_t)i]) d0 = i;
        if (d1 >= 0 || d2 >= 0 || d0 >= 0 || !guards || unw) {
            ok = false;
            char b[240];
            std::snprintf(b, sizeof b, " %s: one-shot vs oracle first diff %ld, two-shot %ld, vs placement 0 %ld, guards %d, "
                          "%ld plane words unwritten", pl.name, d1, d2, d0, (int)guards, unw);
            why += b;
        }
    }
    ++cases;
    if (!ok) {
        ++fails;
        std::printf("FAIL slot sum M=%d n=%d top_k=%d E=%d %s weights shared %d:%s\n", M, n, k, E, s.f32w ? "f32" : "bf16",
                    s.shared, why.c_str());
    }
}

/* The placed scatter: every rank's kernel output against the reference, and the contiguous table
 * against moe_scatter's own sort. */
void run_scatter(int M, int k, int E, std::mt19937& rng, int& cases, int& fails) {
    const int T = M * k, nh = E / WS;
    std::vector<int32_t> ex((size_t)T, -1);
    for (int tok = 0; tok < M; ++tok)
        for (int j = 0; j < k; ++j) {
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                /* skewed: a few hot experts, as real routing is */
                e = (int32_t)(rng() % 3 == 0 ? rng() % (uint32_t)std::max(8, E / 16) : rng() % (uint32_t)E);
                dup = false;
                for (int q = 0; q < j; ++q) dup = dup || ex[(size_t)(tok * k + q)] == e;
            }
            ex[(size_t)(tok * k + j)] = e;
        }
    std::string why;
    bool ok = true;
    for (const Place& pl : placements(E, rng)) {
        std::vector<R4dPlaceTab> tabs;
        std::vector<std::vector<int32_t>> sorted;
        std::vector<int32_t> owner;
        std::string pw;
        if (!ref_sorts(pl, E, ex, tabs, sorted, owner, pw)) { ok = false; why += std::string(" ") + pl.name + ": table " + pw; continue; }
        std::vector<int> seen((size_t)T, 0);
        for (int q = 0; q < WS; ++q) {
            const int nl = (int)tabs[(size_t)q].n_local;
            std::vector<int32_t> so((size_t)T + 4, 0x7e57), of((size_t)nl + 5, 0x7e57), cn((size_t)nl + 4, 0x7e57);
            const int rc = r4d_moe_scatter_placed_launch(ex.data(), tabs[(size_t)q].tab.data(), so.data(), of.data(),
                                                         cn.data(), T, nl, E, WS, q, nullptr);
            if (rc != RAD_OK) { ok = false; why += " launch rc " + std::to_string(rc); continue; }
            /* the sort as the reference, -1 past the live count, guards intact */
            long d = -1;
            for (int i = 0; i < T && d < 0; ++i) if (so[(size_t)i] != sorted[(size_t)q][(size_t)i]) d = i;
            bool g = so[(size_t)T] == 0x7e57 && of[(size_t)nl + 1] == 0x7e57 && cn[(size_t)nl] == 0x7e57;
            /* offsets / counts agree with the sort */
            const int32_t* loc = tabs[(size_t)q].tab.data() + E;
            bool oc = of[0] == 0;
            int live = 0;
            for (int l = 0; l < nl; ++l) { oc = oc && of[(size_t)l + 1] == of[(size_t)l] + cn[(size_t)l]; live += cn[(size_t)l]; }
            for (int i = 0; i < live; ++i) {
                const int32_t f = so[(size_t)i];
                if (f < 0 || f >= T) { oc = false; break; }
                ++seen[(size_t)f];
                const int32_t l = loc[ex[(size_t)f]];
                oc = oc && l >= 0 && i >= of[(size_t)l] && i < of[(size_t)l + 1];
            }
            if (d >= 0 || !g || !oc) {
                ok = false;
                char b[200];
                std::snprintf(b, sizeof b, " %s rank %d: sort first diff %ld, guards %d, offsets/counts %d", pl.name, q, d,
                              (int)g, (int)oc);
                why += b;
            }
            /* contiguous: moe_scatter's own small sort (today's EP layer) */
            if (std::string(pl.name) == "contiguous") {
                std::vector<int32_t> s2((size_t)T + 4, 0x7e57), o2((size_t)nh + 5, 0x7e57), c2((size_t)nh + 4, 0x7e57);
                r4d_moesort::Protect none{};
                R4D_EMIT(nullptr, (r4d_moesort::scatter_small_kernel<false>), dim3(1),
                         dim3(r4d_moesort::kScatWaves * r4d_moesort::kSortWave), 0, ex.data(), s2.data(), o2.data(),
                         c2.data(), (uint32_t)T, (uint32_t)nh, (int32_t)(q * nh), none, (int32_t*)nullptr);
                bool same = true;
                for (int i = 0; i < T; ++i) same = same && s2[(size_t)i] == so[(size_t)i];
                for (int l = 0; l <= nh; ++l) same = same && o2[(size_t)l] == of[(size_t)l];
                for (int l = 0; l < nh; ++l) same = same && c2[(size_t)l] == cn[(size_t)l];
                if (!same) { ok = false; why += " contiguous rank " + std::to_string(q) + ": differs from moe_scatter's sort"; }
            }
        }
        for (int i = 0; i < T; ++i)
            if (seen[(size_t)i] != 1) { ok = false; why += std::string(" ") + pl.name + ": slot " + std::to_string(i) + " served " + std::to_string(seen[(size_t)i]) + " times"; break; }
    }
    ++cases;
    if (!ok) {
        ++fails;
        std::printf("FAIL placed scatter M=%d top_k=%d E=%d:%s\n", M, k, E, why.c_str());
    }
}

}  // namespace

int main() {
    std::mt19937 rng(4141);
    int cases = 0, fails = 0;
    std::vector<Shape> shapes;
    for (int M : { 1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32, 48, 64 }) shapes.push_back({ M, 5120, 6, 384, true, 1 });
    for (int M : { 3, 17, 40 }) {
        shapes.push_back({ M, 512, 8, 64, false, 0 });
        shapes.push_back({ M, 512, 8, 64, true, 0 });
        shapes.push_back({ M, 512, 8, 64, false, 1 });
        shapes.push_back({ M, 512, 8, 64, true, 2 });
    }
    shapes.push_back({ 256, 128, 8, 64, true, 1 });     /* the slot table's row bound, 2048 slots */
    for (const Shape& s : shapes) run_shape(s, rng, cases, fails);
    const int sum_cases = cases;
    for (int M : { 1, 2, 4, 8, 16, 48, 64, 170 }) run_scatter(M, 6, 384, rng, cases, fails);
    for (int M : { 1, 5, 33, 128 }) run_scatter(M, 8, 64, rng, cases, fails);
    std::printf("%d slot-sum case(s) (%d shapes x 3 placements, one-shot and two-shot planes), %d placed-scatter "
                "case(s) (x 3 placements x 4 ranks), %d failure(s)\n", sum_cases, (int)shapes.size(), cases - sum_cases,
                fails);
    return fails ? 1 : 0;
}
