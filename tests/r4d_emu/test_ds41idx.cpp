/* tests/r4d_emu: DeepSeek-V4.1's indexer (r4d_ds41_sparse.hip) on the host SIMT emulator -- the
 * candidate restriction and the row split over the ranks.
 *
 * (a) index_score with `keep` scores only index_candidates' kept blocks. What a layer past the
 *     candidate layer selects, index_topk(score, keep), must be the same set, bit for bit, as from
 *     the unrestricted score deepseek41 issued before; every kept score must be the unrestricted
 *     score's bits and every other one -inf. Checked at the decode form (the scalar kernel) and the
 *     prefill form (WMMA, > 48 rows), at a power-of-two block (8, DeepSeek's) and another (3), ratio
 *     1 and 2, with unmapped blocks, rows of several sequences and exact ties.
 * (b) a step's runs (row_part) dealt over the ranks by arch/deepseek41/ds41_index_split.h, each
 *     rank's candidates, restricted scores and selections of its own runs, joined as the all_gather
 *     concatenates them: every row must be the unsplit runs' selection, at worlds 2, 3 and 4.
 * libref (RADIANCE_HOME's libref.so) is held to the kernels: its index_topk over the restricted
 * score and keep is the kernel's selection, its index_candidates the test's keep (a host port of
 * it), its index_score with keep the scalar kernel's bits, and at every kept position its own
 * unrestricted score.
 *
 * Contexts: 5,000- and 2,500-position rows on both forms; 300,000-position rows (past 2^18) on the
 * decode form; with --long also on the prefill form (49 rows, minutes on the emulator). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "../../arch/deepseek41/ds41_index_split.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_ds41_index_score(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_index_topk(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs*);
extern "C" int r4d_ds41_entry_store_mx4(const RadArgs*, RadStream);

namespace {

constexpr int kDim = 128;

RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = const_cast<void*>(p);
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
int64_t keep_words(int64_t width, int64_t block) { return ((width + block - 1) / block + 31) / 32; }

/* libref's launch for `op`, through the plugin's own exports, or null. */
struct Ref {
    RadLaunchFn score = nullptr, cand = nullptr, topk = nullptr, store = nullptr;
};
Ref load_libref() {
    Ref r;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return r;
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return r;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
    if (open && open() < 0) return r;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return r;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (!std::strcmp(k->op, "index_score")) r.score = k->launch;
        if (!std::strcmp(k->op, "index_candidates")) r.cand = k->launch;
        if (!std::strcmp(k->op, "index_topk")) r.topk = k->launch;
        if (!std::strcmp(k->op, "entry_store")) r.store = k->launch;
    }
    return r;
}

/* A step: sequences of rows ending at given positions, their paged index caches (one entry a
 * block of `ratio` slots, as deepseek41's index groups), the candidate layer's query and head
 * weights (q2, w2) and the scoring layer's (q, w). */
struct Fix {
    int H = 0, ratio = 1, S = 0, M = 0;
    int64_t C = 0, nblk = 0, nbt = 0;
    std::vector<uint16_t> q, w, q2, w2, kc;
    std::vector<int32_t> bt, pos, cu;
};

Fix make(uint32_t seed, int H, int ratio, const std::vector<std::pair<int, int64_t>>& seqs) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 9999);
    Fix f;
    f.H = H;
    f.ratio = ratio;
    f.S = (int)seqs.size();
    f.cu.push_back(0);
    std::vector<int64_t> ents;
    for (const auto& s : seqs) {
        for (int i = 0; i < s.first; ++i) f.pos.push_back((int32_t)(s.second - s.first + 1 + i));
        f.cu.push_back(f.cu.back() + s.first);
        ents.push_back((s.second + 1) / ratio);
        f.C = std::max<int64_t>(f.C, (s.second + 1) / ratio);
        f.nbt = std::max<int64_t>(f.nbt, (s.second + 1) / ratio + 2);
    }
    f.M = f.cu.back();
    int64_t total = 0;
    for (int64_t e : ents) total += e;
    f.nblk = total + 7;
    /* every entry its own block, the blocks shuffled over the cache; 0.05 % unmapped */
    std::vector<int32_t> perm((size_t)f.nblk);
    for (int64_t i = 0; i < f.nblk; ++i) perm[(size_t)i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    f.bt.assign((size_t)f.S * f.nbt, -1);
    int64_t used = 0;
    for (int s = 0; s < f.S; ++s)
        for (int64_t j = 0; j < ents[(size_t)s]; ++j)
            f.bt[(size_t)s * f.nbt + j] = pct(rng) < 5 ? -1 : perm[(size_t)used++];
    auto bf = [&] { return rad_f32_to_bf16(nd(rng)); };
    f.kc.resize((size_t)f.nblk * kDim);
    for (auto& v : f.kc) v = bf();
    /* exact ties: 1 % of a sequence's entries hold the previous entry's key */
    for (int s = 0; s < f.S; ++s)
        for (int64_t j = 1; j < ents[(size_t)s]; ++j) {
            const int32_t a = f.bt[(size_t)s * f.nbt + j - 1], b = f.bt[(size_t)s * f.nbt + j];
            if (a >= 0 && b >= 0 && pct(rng) < 100)
                std::memcpy(&f.kc[(size_t)b * kDim], &f.kc[(size_t)a * kDim], kDim * 2);
        }
    for (auto* v : { &f.q, &f.q2 }) {
        v->resize((size_t)f.M * H * kDim);
        for (auto& x : *v) x = bf();
    }
    for (auto* v : { &f.w, &f.w2 }) {
        v->resize((size_t)f.M * H);
        for (auto& x : *v) x = bf();
    }
    return f;
}

/* index_score of plane rows from step row `part * plane` (part < 0: the whole step), optionally
 * restricted to `keep` ([krows, KW] for the plane's rows). */
constexpr int kMx4Slots = 40;   /* a packed 128-value key: 80 bytes */
int score(RadLaunchFn fn, const Fix& f, bool second, int64_t plane, int part, const int32_t* keep,
          int64_t krows, int block, float* out, const uint16_t* packed = nullptr) {
    RadTensor t[8] = {
        tens(second ? f.q2.data() : f.q.data(), RAD_BF16, { f.M, (int64_t)f.H * kDim }),
        tens(second ? f.w2.data() : f.w.data(), RAD_BF16, { f.M, f.H }),
        packed ? tens(packed, RAD_BF16, { f.nblk, 1, f.ratio, kMx4Slots / f.ratio })
               : tens(f.kc.data(), RAD_BF16, { f.nblk, 1, f.ratio, kDim / f.ratio }),
        tens(f.bt.data(), RAD_I32, { f.S, f.nbt }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(out, RAD_F32, { plane, f.C }),
        keep ? tens(keep, RAD_I32, { krows, keep_words(f.C, block) }) : RadTensor{},
    };
    std::vector<RadParam> p = { pint("M", plane), pint("heads", f.H), pint("dim", kDim),
                                pint("ratio", f.ratio), pint("width", f.C),
                                pf64("wscale", 1.0 / std::sqrt((double)kDim * f.H)), pstr("dtype", "bf16") };
    if (part >= 0) p.push_back(pint("row_part", part));
    if (keep) p.push_back(pint("block", block));
    if (packed) p.push_back(pstr("key_format", "mxfp4"));
    RadArgs a{};
    a.t = t;
    a.n_t = 8;
    a.p = p.data();
    a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

/* libref's index_candidates, ported: block maxima (std::max, a NaN never replaces), the newest
 * reachable block pinned to +inf, the kb best kept unless -inf, ties to the lower block. The test
 * feeds keep from here: the kernel's ds_emit takes a wave ballot in a branch that lanes past the
 * last block skip, which the card allows and the emulator reports as a divergence. */
void host_candidates(const Fix& f, const float* sc, int64_t rows, const int32_t* pos, int block, int kb,
                     int32_t* keep) {
    const int64_t C = f.C, nb = (C + block - 1) / block, KW = keep_words(C, block);
    for (int64_t m = 0; m < rows; ++m) {
        std::vector<float> bs((size_t)nb, -INFINITY);
        for (int64_t j = 0; j < C; ++j) bs[(size_t)(j / block)] = std::max(bs[(size_t)(j / block)], sc[m * C + j]);
        const int64_t reach = ((int64_t)pos[m] + 1) / f.ratio;
        if (reach >= 1 && (reach - 1) / block < nb) bs[(size_t)((reach - 1) / block)] = INFINITY;
        std::vector<int64_t> idx((size_t)nb);
        for (int64_t i = 0; i < nb; ++i) idx[(size_t)i] = i;
        const int64_t k = std::min<int64_t>(kb, nb);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t x, int64_t y) {
            return bs[(size_t)x] > bs[(size_t)y] || (bs[(size_t)x] == bs[(size_t)y] && x < y);
        });
        for (int64_t w = 0; w < KW; ++w) keep[m * KW + w] = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t b = idx[(size_t)i];
            if (bs[(size_t)b] > -INFINITY) keep[m * KW + b / 32] |= (int32_t)(1u << (b % 32));
        }
    }
}

int candidates(RadLaunchFn fn, const Fix& f, const float* sc, int64_t rows, const int32_t* pos,
               int block, int kb, int32_t* keep) {
    RadTensor t[3] = { tens(sc, RAD_F32, { rows, f.C }), tens(pos, RAD_I32, { rows }),
                       tens(keep, RAD_I32, { rows, keep_words(f.C, block) }) };
    RadParam p[5] = { pint("M", rows), pint("width", f.C), pint("block", block), pint("topk_blocks", kb),
                      pint("ratio", f.ratio) };
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 5;
    return fn(&a, nullptr);
}

/* index_topk over [rows, C] with keep; `split` gives the r4d kernel its scratch (the
 * chunk-parallel form at <= 16 rows). */
int topk(RadLaunchFn fn, const Fix& f, const float* sc, int64_t rows, const int32_t* keep,
         const int32_t* pos, int block, int k, bool split, int32_t* sel) {
    RadTensor t[4] = { tens(sc, RAD_F32, { rows, f.C }),
                       keep ? tens(keep, RAD_I32, { rows, keep_words(f.C, block) }) : RadTensor{},
                       tens(pos, RAD_I32, { rows }), tens(sel, RAD_I32, { rows, k }) };
    RadParam p[5] = { pint("M", rows), pint("width", f.C), pint("topk", k), pint("ratio", f.ratio),
                      pint("block", block) };
    RadArgs a{};
    a.t = t;
    a.n_t = 4;
    a.p = p;
    a.n_p = keep ? 5 : 4;
    std::vector<uint64_t> scratch;
    if (split && fn == r4d_rad_ds41_index_topk) {
        const int64_t need = r4d_rad_ds41_index_topk_scratch(&a);
        scratch.assign((size_t)(need / 8 + 2), 0xA5A5A5A5A5A5A5A5ull);
        a.scratch = scratch.data();
        a.scratch_bytes = need;
    }
    return fn(&a, nullptr);
}

/* The selection index_topk must make, on the host: a row's candidates are its positions below
 * reach in a kept block with a score above -inf (a NaN is none), the k best by (score desc,
 * position asc), written in position order, -1 after. */
void host_topk(const Fix& f, const float* sc, int64_t rows, const int32_t* keep, const int32_t* pos, int block,
               int k, int32_t* sel) {
    const int64_t C = f.C, KW = keep_words(C, block);
    for (int64_t m = 0; m < rows; ++m) {
        const int64_t reach = std::max<int64_t>(0, std::min<int64_t>(C, ((int64_t)pos[m] + 1) / f.ratio));
        std::vector<int64_t> c;
        for (int64_t j = 0; j < reach; ++j) {
            const int64_t b = j / block;
            if (keep && !(((uint32_t)keep[m * KW + b / 32] >> (b % 32)) & 1u)) continue;
            const float v = sc[m * C + j];
            if (v > -INFINITY) c.push_back(j);
        }
        const int64_t n = std::min<int64_t>(k, (int64_t)c.size());
        std::partial_sort(c.begin(), c.begin() + n, c.end(), [&](int64_t x, int64_t y) {
            const float a = sc[m * C + x] + 0.0f, b = sc[m * C + y] + 0.0f;
            return a > b || (a == b && x < y);
        });
        std::sort(c.begin(), c.begin() + n);
        for (int64_t i = 0; i < k; ++i) sel[m * k + i] = i < n ? (int32_t)c[(size_t)i] : -1;
    }
}

int g_fail = 0, g_check = 0;
void expect(bool ok, const char* name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name, what, a, b);
}

/* (a) at one pass height: the whole step as one plane. */
void restriction(const char* name, const Fix& f, int block, int kb, int k, const Ref& ref) {
    const int64_t M = f.M, C = f.C, KW = keep_words(C, block);
    std::vector<float> sc20((size_t)(M * C)), scA((size_t)(M * C)), scB((size_t)(M * C));
    std::vector<int32_t> keep((size_t)(M * KW), 0x5A5A5A5A), keepR((size_t)(M * KW), 0);
    std::vector<int32_t> selA((size_t)(M * k), 0x55555555), selB((size_t)(M * k), 0x33333333);
    expect(score(r4d_rad_ds41_index_score, f, true, M, -1, nullptr, 0, block, sc20.data()) == RAD_OK,
           name, "candidate layer's index_score");
    host_candidates(f, sc20.data(), M, f.pos.data(), block, kb, keep.data());
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, nullptr, 0, block, scA.data()) == RAD_OK,
           name, "unrestricted index_score");
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, keep.data(), M, block, scB.data()) == RAD_OK,
           name, "restricted index_score");
    const bool split = M <= 16;
    expect(topk(r4d_rad_ds41_index_topk, f, scA.data(), M, keep.data(), f.pos.data(), block, k, split,
                selA.data()) == RAD_OK, name, "index_topk (unrestricted score)");
    expect(topk(r4d_rad_ds41_index_topk, f, scB.data(), M, keep.data(), f.pos.data(), block, k, split,
                selB.data()) == RAD_OK, name, "index_topk (restricted score)");
    /* the selections: the same set, in the same order, every row */
    int64_t diff = 0, nsel = 0;
    for (int64_t i = 0; i < M * k; ++i) {
        diff += selA[(size_t)i] != selB[(size_t)i];
        nsel += selB[(size_t)i] >= 0;
    }
    expect(diff == 0, name, "selection differs from the unrestricted score's", diff, M * k);
    /* and both are the host's selection over the unrestricted score (kept blocks only) */
    {
        std::vector<int32_t> selH((size_t)(M * k), 0x44444444);
        host_topk(f, scA.data(), M, keep.data(), f.pos.data(), block, k, selH.data());
        int64_t dh = 0;
        for (int64_t i = 0; i < M * k; ++i) dh += selH[(size_t)i] != selB[(size_t)i];
        expect(dh == 0, name, "index_topk differs from the host's selection", dh, M * k);
    }
    /* the scores: kept and passed = the unrestricted bits, else -inf */
    int64_t kept = 0, bad = 0, passed = 0;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t reach = std::min<int64_t>(C, ((int64_t)f.pos[(size_t)m] + 1) / f.ratio);
        for (int64_t j = 0; j < C; ++j) {
            const int64_t b = j / block;
            const bool kb_ = ((uint32_t)keep[(size_t)(m * KW + b / 32)] >> (b % 32)) & 1u;
            const float a = scA[(size_t)(m * C + j)], v = scB[(size_t)(m * C + j)];
            passed += j < reach;
            if (kb_ && j < reach) {
                ++kept;
                bad += bits(a) != bits(v);
            } else {
                bad += !(std::isinf(v) && v < 0);
            }
        }
    }
    expect(bad == 0, name, "restricted scores differ from the unrestricted at kept positions or are not -inf elsewhere", bad, kept);
    /* the same selection when the chunk-parallel form is not taken (or is) */
    {
        std::vector<int32_t> selC((size_t)(M * k), 0x11111111);
        expect(topk(r4d_rad_ds41_index_topk, f, scB.data(), M, keep.data(), f.pos.data(), block, k, !split,
                    selC.data()) == RAD_OK, name, "index_topk (other form)");
        int64_t d = 0;
        for (int64_t i = 0; i < M * k; ++i) d += selC[(size_t)i] != selB[(size_t)i];
        expect(d == 0, name, "the two index_topk forms disagree", d, M * k);
    }
    int64_t refmis = -1;
    if (ref.score && ref.cand && ref.topk) {
        expect(candidates(ref.cand, f, sc20.data(), M, f.pos.data(), block, kb, keepR.data()) == RAD_OK,
               name, "libref index_candidates");
        int64_t dk = 0;
        for (int64_t i = 0; i < M * KW; ++i) dk += keepR[(size_t)i] != keep[(size_t)i];
        expect(dk == 0, name, "libref's keep differs from the test's port of it", dk, M * KW);
        std::vector<int32_t> selR((size_t)(M * k), 0x22222222);
        expect(topk(ref.topk, f, scB.data(), M, keep.data(), f.pos.data(), block, k, false, selR.data()) == RAD_OK,
               name, "libref index_topk");
        int64_t ds = 0;
        for (int64_t i = 0; i < M * k; ++i) ds += selR[(size_t)i] != selB[(size_t)i];
        expect(ds == 0, name, "libref's index_topk over the restricted score differs", ds, M * k);
        std::vector<float> rA((size_t)(M * C)), rB((size_t)(M * C));
        expect(score(ref.score, f, false, M, -1, nullptr, 0, block, rA.data()) == RAD_OK, name, "libref index_score");
        expect(score(ref.score, f, false, M, -1, keep.data(), M, block, rB.data()) == RAD_OK, name,
               "libref index_score with keep");
        int64_t self = 0;
        refmis = 0;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t j = 0; j < C; ++j) {
                const float v = rB[(size_t)(m * C + j)];
                if (std::isinf(v) && v < 0) {
                    const float u = scB[(size_t)(m * C + j)];
                    self += !(std::isinf(u) && u < 0);
                    continue;
                }
                self += bits(v) != bits(rA[(size_t)(m * C + j)]);
                refmis += bits(v) != bits(scB[(size_t)(m * C + j)]);
            }
        expect(self == 0, name, "libref's keep path differs from its unrestricted path or the kernel's -inf set", self, kept);
        /* the decode form is libref's bits; the WMMA form moves a score only where the f32 dot sat at
         * a bf16 rounding boundary (its summation order) */
        if (M <= 48) expect(refmis == 0, name, "scalar kernel differs from libref", refmis, kept);
    }
    std::printf("%-48s rows %lld width %lld heads %d ratio %d block %d: kept %lld of %lld passed "
                "(%.1f %%), %lld selected; libref score mismatches %lld (%s form)\n",
                name, (long long)M, (long long)C, f.H, f.ratio, block, (long long)kept, (long long)passed,
                passed ? 100.0 * (double)kept / (double)passed : 0.0, (long long)nsel, (long long)refmis,
                M > 48 ? "WMMA" : "scalar");
}

/* (b) a step scored in runs of at most `irows` rows, the runs dealt over `world` ranks: each rank's
 * candidates, restricted scores and selections of its own runs, the parts concatenated as the
 * all_gather lays them, against the unsplit runs (world 1). */
void deal(const char* name, const Fix& f, int64_t irows, int block, int kb, int k) {
    const int64_t M = f.M, C = f.C, KW = keep_words(C, block);
    auto run_world = [&](int world, std::vector<int32_t>* out) {
        const Ds41IdxRuns r0 = ds41_idx_runs(M, irows, C, world, 0, true);
        const int64_t rows_all = world > 1 && r0.split ? (int64_t)world * r0.q * r0.R : M;
        out->assign((size_t)(std::max(rows_all, M) * k), 0x77777777);
        for (int rank = 0; rank < world; ++rank) {
            const Ds41IdxRuns r = ds41_idx_runs(M, irows, C, world, rank, true);
            std::vector<int32_t> keep((size_t)(M * KW), 0x0F0F0F0F);
            std::vector<int32_t> part((size_t)(r.q * r.R * k), 0x66666666);
            std::vector<float> plane((size_t)(r.R * C));
            for (int64_t p = r.p0; p < r.p1; ++p) {
                const int64_t rr0 = p * r.R, rows = std::min(r.R, M - rr0);
                expect(score(r4d_rad_ds41_index_score, f, true, r.R, (int)p, nullptr, 0, block, plane.data()) ==
                       RAD_OK, name, "run score (candidate layer)");
                host_candidates(f, plane.data(), rows, f.pos.data() + rr0, block, kb, keep.data() + rr0 * KW);
                expect(score(r4d_rad_ds41_index_score, f, false, r.R, (int)p, keep.data() + rr0 * KW, rows,
                             block, plane.data()) == RAD_OK, name, "run restricted score");
                int32_t* dst = r.split ? part.data() + (p - r.p0) * r.R * k : out->data() + rr0 * k;
                expect(topk(r4d_rad_ds41_index_topk, f, plane.data(), rows, keep.data() + rr0 * KW,
                            f.pos.data() + rr0, block, k, rows <= 16, dst) == RAD_OK, name, "run topk");
            }
            /* the all_gather: rank r's q R rows at r q R */
            if (r.split)
                std::memcpy(out->data() + (size_t)(rank * r.q * r.R * k), part.data(),
                            (size_t)(r.q * r.R * k) * 4);
            if (!r.split) break;   /* unsplit: one rank holds the whole step */
        }
    };
    std::vector<int32_t> one;
    run_world(1, &one);
    for (int world : { 2, 3, 4 }) {
        std::vector<int32_t> got;
        run_world(world, &got);
        const Ds41IdxRuns r = ds41_idx_runs(M, irows, C, world, 0, true);
        int64_t d = 0;
        for (int64_t i = 0; i < M * k; ++i) d += got[(size_t)i] != one[(size_t)i];
        char what[160];
        std::snprintf(what, sizeof what, "world %d (%lld runs of %lld rows, %lld a rank, split %d) differs "
                      "from the unsplit runs", world, (long long)r.n, (long long)r.R, (long long)r.q, (int)r.split);
        expect(d == 0, name, what, d, M * k);
        std::printf("%-48s world %d: %lld runs of %lld rows, %lld a rank, split %d, %lld of %lld entries differ\n",
                    name, world, (long long)r.n, (long long)r.R, (long long)r.q, (int)r.split, (long long)d,
                    (long long)(M * k));
    }
}

/* (c) MXFP4 index keys. The fixture's keys made quant_roundtrip(fp4_e2m1, ue8m0, 32) values (a host
 * port of libref's rule), packed by entry_store format mxfp4 (the kernel; libref's when loadable),
 * every packed key decoded on the host back to the stored bf16 bits, then every form of index_score
 * (decode band, the kept kernel, WMMA) over the packed cache against the bf16 cache: bit for bit. */
uint16_t host_mx4_bf16(unsigned c, unsigned e) {   /* the kernel's ds_mx4_bf16 */
    const unsigned m = c & 7u, sg = (c & 8u) << 12;
    if (!m) return (uint16_t)sg;
    const unsigned E = e + (m >> 1) - 1u;
    if (!E) return (uint16_t)(sg | 0x40u);
    return (uint16_t)(sg | (E << 7) | ((m & 1u) && m > 1u ? 0x40u : 0u));
}
void fp4_roundtrip_keys(Fix& f) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int64_t blk = 0; blk < f.nblk; ++blk)
        for (int b = 0; b < kDim / 32; ++b) {
            uint16_t* v = &f.kc[(size_t)(blk * kDim + b * 32)];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(rad_bf16_to_f32(v[i])));
            amax = std::max(amax, 6.0f * 0x1p-126f);
            uint32_t u;
            const float t = amax * (1.0f / 6.0f);
            std::memcpy(&u, &t, 4);
            const int k = (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
            const float scale = std::ldexp(1.0f, k);
            for (int i = 0; i < 32; ++i) {
                const float x = std::clamp(rad_bf16_to_f32(v[i]) / scale, -6.0f, 6.0f);
                unsigned best = 0;
                for (unsigned c = 1; c < 8; ++c) {
                    const float d = std::fabs(kMag[c] - std::fabs(x)), db = std::fabs(kMag[best] - std::fabs(x));
                    if (d < db || (d == db && (c & 1u) == 0u)) best = c;
                }
                v[i] = rad_f32_to_bf16((std::signbit(x) ? -1.0f : 1.0f) * kMag[best] * scale);
            }
        }
}
int store_mx4(RadLaunchFn fn, const Fix& f, uint16_t* packed) {
    std::vector<int32_t> page((size_t)f.nblk);
    for (int64_t i = 0; i < f.nblk; ++i) page[(size_t)i] = (int32_t)i;
    RadTensor t[3] = { tens(f.kc.data(), RAD_BF16, { f.nblk, kDim }), tens(page.data(), RAD_I32, { f.nblk }),
                       tens(packed, RAD_BF16, { f.nblk, 1, f.ratio, kMx4Slots / f.ratio }) };
    RadParam p[5] = { pint("M", f.nblk), pint("n", kDim), pint("block_size", f.ratio), pstr("dtype", "bf16"),
                      pstr("format", "mxfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 5;
    return fn(&a, nullptr);
}
void packed_keys(const char* name, Fix f, int block, int kb, const Ref& ref) {
    fp4_roundtrip_keys(f);
    const int64_t M = f.M, C = f.C, KW = keep_words(C, block);
    std::vector<uint16_t> pk((size_t)(f.nblk * kMx4Slots), 0x7777), pr((size_t)(f.nblk * kMx4Slots), 0x6666);
    expect(store_mx4(r4d_ds41_entry_store_mx4, f, pk.data()) == RAD_OK, name, "entry_store mxfp4");
    int64_t dec = 0;
    for (int64_t blk = 0; blk < f.nblk; ++blk) {
        const uint8_t* e = (const uint8_t*)&pk[(size_t)(blk * kMx4Slots)];
        for (int d = 0; d < kDim; ++d)
            dec += host_mx4_bf16((e[d / 2] >> (4 * (d & 1))) & 15u, e[kDim / 2 + d / 32]) != f.kc[(size_t)(blk * kDim + d)];
        for (int z = kDim / 2 + kDim / 32; z < 2 * kMx4Slots; ++z) dec += e[z] != 0;
    }
    expect(dec == 0, name, "a packed key does not decode to the stored bf16 bits", dec, f.nblk * kDim);
    if (ref.store) {
        expect(store_mx4(ref.store, f, pr.data()) == RAD_OK, name, "libref entry_store mxfp4");
        int64_t d = 0;
        for (size_t i = 0; i < pk.size(); ++i) d += pk[i] != pr[i];
        expect(d == 0, name, "libref's packed cache differs from the kernel's", d, (long long)pk.size());
    }
    std::vector<float> sc20((size_t)(M * C)), a1((size_t)(M * C)), a2((size_t)(M * C)), b1((size_t)(M * C)),
        b2((size_t)(M * C));
    std::vector<int32_t> keep((size_t)(M * KW), 0);
    expect(score(r4d_rad_ds41_index_score, f, true, M, -1, nullptr, 0, block, sc20.data()) == RAD_OK, name, "cand score");
    host_candidates(f, sc20.data(), M, f.pos.data(), block, kb, keep.data());
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, nullptr, 0, block, a1.data()) == RAD_OK, name, "bf16");
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, nullptr, 0, block, a2.data(), pk.data()) == RAD_OK, name,
           "mxfp4");
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, keep.data(), M, block, b1.data()) == RAD_OK, name,
           "bf16 kept");
    expect(score(r4d_rad_ds41_index_score, f, false, M, -1, keep.data(), M, block, b2.data(), pk.data()) == RAD_OK,
           name, "mxfp4 kept");
    int64_t da = 0, db = 0, live = 0;
    for (size_t i = 0; i < a1.size(); ++i) {
        da += bits(a1[i]) != bits(a2[i]);
        db += bits(b1[i]) != bits(b2[i]);
        live += a1[i] > -INFINITY;
    }
    expect(da == 0, name, "mxfp4 keys score differently from bf16 keys", da, (long long)a1.size());
    expect(db == 0, name, "mxfp4 keys score differently from bf16 keys (kept)", db, (long long)b1.size());
    int64_t dr = -1;
    if (ref.score) {
        std::vector<float> r1((size_t)(M * C)), r2((size_t)(M * C));
        expect(score(ref.score, f, false, M, -1, nullptr, 0, block, r1.data()) == RAD_OK, name, "libref bf16");
        expect(score(ref.score, f, false, M, -1, nullptr, 0, block, r2.data(), pk.data()) == RAD_OK, name,
               "libref mxfp4");
        dr = 0;
        for (size_t i = 0; i < r1.size(); ++i) dr += bits(r1[i]) != bits(r2[i]);
        expect(dr == 0, name, "libref: mxfp4 keys score differently from bf16 keys", dr, (long long)r1.size());
    }
    std::printf("%-48s rows %lld width %lld heads %d ratio %d: %lld keys packed (80 B each), %lld live scores; "
                "differ from bf16 keys: %lld (unrestricted), %lld (kept), libref %lld (%s form)\n",
                name, (long long)M, (long long)C, f.H, f.ratio, (long long)f.nblk, (long long)live, (long long)da,
                (long long)db, (long long)dr, M > 48 ? "WMMA" : "scalar");
}

}  // namespace

int main(int argc, char** argv) {
    const bool longrun = argc > 1 && !std::strcmp(argv[1], "--long");
    const Ref ref = load_libref();
    if (!ref.score || !ref.cand || !ref.topk)
        std::printf("note: libref not loadable (set RADIANCE_HOME): its checks are skipped\n");
    /* DeepSeek-V4.1-Flash's geometry: 32 heads of 128, top-512 positions, 2048 candidate blocks of
     * 8 at the candidate layer (ratio 1); smaller selections where the context is short. */
    {
        /* a prefill chunk of 64 rows at 4,936.. beside two decode rows of another sequence */
        const Fix f = make(11, 32, 1, { { 2, 3100 }, { 64, 4999 } });
        restriction("WMMA, chunk beside decode rows", f, 8, 64, 128, ref);
        restriction("WMMA, block 3", f, 3, 100, 96, ref);
    }
    {
        const Fix f = make(12, 20, 2, { { 52, 2499 * 2 + 1 } });
        restriction("WMMA, 20 heads, ratio 2", f, 8, 40, 64, ref);
    }
    {
        const Fix f = make(13, 32, 1, { { 1, 4800 }, { 3, 4999 } });
        restriction("scalar, decode rows", f, 8, 64, 128, ref);
        restriction("scalar, every block kept", f, 8, 4096, 128, ref);
    }
    {
        /* rows past 2^18: decode rows of two sequences, DeepSeek's block 8 / 2048 blocks / top 512 */
        const Fix f = make(14, 32, 1, { { 1, 262144 + 1000 }, { 2, 299999 } });
        restriction("scalar, 300k context", f, 8, 2048, 512, ref);
    }
    {
        /* index_topk over the kept blocks only: more kept blocks a row than the one-block-a-row
         * kernel lists (4500 > 4096: it walks the positions), the split form dealing them over
         * its chunks; and block 2 / block 1 with every block kept (the split's lists fit at block
         * 2, not at block 1: it walks the positions) */
        const Fix f = make(18, 16, 1, { { 3, 39999 } });
        restriction("kept walk, 4500 of 5000 blocks", f, 8, 4500, 256, ref);
        restriction("kept walk, block 2, all kept", f, 2, 20000, 128, ref);
        restriction("kept walk, block 1, 30000 kept", f, 1, 30000, 128, ref);
    }
    {
        /* (c): packed keys, both forms, ratio 1 and 2 */
        packed_keys("mxfp4 keys, scalar, decode rows", make(21, 32, 1, { { 1, 4800 }, { 3, 4999 } }), 8, 64, ref);
        packed_keys("mxfp4 keys, WMMA, 66 rows", make(22, 32, 1, { { 2, 3100 }, { 64, 4999 } }), 8, 64, ref);
        packed_keys("mxfp4 keys, WMMA, 20 heads, ratio 2", make(23, 20, 2, { { 52, 2499 * 2 + 1 } }), 8, 40, ref);
        packed_keys("mxfp4 keys, scalar, ratio 2", make(24, 32, 2, { { 3, 4001 } }), 8, 40, ref);
    }
    {
        /* (b): 200 prefill rows in runs of <= 64 (4 runs of 50, WMMA); and 130 rows in runs of
         * <= 48 (3 runs of 44, scalar) */
        const Fix f = make(15, 32, 1, { { 200, 6199 } });
        deal("row split, 200 rows, runs <= 64", f, 64, 8, 64, 128);
        const Fix g = make(16, 16, 1, { { 3, 2000 }, { 127, 5000 } });
        deal("row split, 130 rows, runs <= 48", g, 48, 8, 48, 96);
    }
    if (longrun) {
        const Fix f = make(17, 16, 1, { { 49, 299999 } });
        restriction("WMMA, 300k context (--long)", f, 8, 2048, 512, ref);
    }
    std::printf("ds41 indexer: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
