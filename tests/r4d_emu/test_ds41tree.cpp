/* tests/r4d_emu: a tree verify for DeepSeek-V4.1's DSpark (lever T1) -- the tree variants and the tree
 * bookkeeping (r4d_ds41_sparse.hip, r4d_qsa_work_bf16.hip, r4d_ngram_ids.hip, r4d_tree.hip) on the host
 * SIMT emulator, and libref's (when RADIANCE_HOME names a build) the same.
 *
 * The claim each part is held to is the tree verify's exactness: a tree row computes exactly what the
 * chain its path is computes. So every tree-aware op runs once over a tree step (a DSpark tree of a
 * main chain and alternative branches at position 1, depth-major, beside ordinary sequences) and once
 * per tree row over the chain-equivalent step -- that row's path laid out as an ordinary chain (its
 * ancestors' keys at consecutive slots, the in-path compressed entries in the cache at their real
 * blocks) -- and the row's outputs must be the chain's last row's, bit for bit:
 *   (a) tree_rows        the table, slot positions and marks against the tree written out
 *   (b) attn_sparse_sink_tree  ratio 0 / 1 / 2, window only and with selections, bf16 and nvfp4
 *                        latent, the decode band (<= 48 rows) and the head-group kernel (> 48)
 *   (c) index_score_tree ratio 1 / 2, with and without keep, bf16 and mxfp4 keys
 *   (d) qsa_work_tree    every tree item's staged keys, page / mpage / tpage, bpos vs qsa_work on the
 *                        chain; the ordinary sequence's items vs qsa_work
 *   (e) ngram_ids_tree   ids vs ngram_ids on the chain; the window write-back vs ngram_ids
 *   (f) tree_commit_rows / tree_commit_entries  after a tree step and the acceptance of a branch other
 *                        than the main one: the ring holds every reachable position's row, the
 *                        Engram window is the chain step's, the cache holds the path's entries and
 *                        nothing of a rejected branch
 *   (g) dspark_tree_assemble  B = 0..3 with missing alternatives, against the layout written out
 * Each kernel check is the emulated kernel's; each libref check the reference's (the two are not
 * compared with each other where their kernels' bits are not promised equal). */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_tree(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_index_score(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_index_score_tree(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_mx4(const RadArgs*, RadStream);
extern "C" int r4d_rad_qsa_work(const RadArgs*, RadStream);
extern "C" int r4d_rad_qsa_work_tree(const RadArgs*, RadStream);
extern "C" int r4d_rad_ngram_ids(const RadArgs*, RadStream);
extern "C" int r4d_rad_ngram_ids_tree(const RadArgs*, RadStream);
extern "C" int r4d_rad_tree_rows(const RadArgs*, RadStream);
extern "C" int r4d_rad_tree_commit_entries(const RadArgs*, RadStream);
extern "C" int r4d_rad_tree_commit_rows(const RadArgs*, RadStream);
extern "C" int r4d_rad_dspark_tree_assemble(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kNv = 144;     /* the latent; its packed entry in bf16 slots */
constexpr int kDi = 128, kMx = 40;     /* the index key; its packed entry in bf16 slots */
constexpr int kTW = 8;                 /* the row table's width */

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
RadTensor none() { return RadTensor{}; }
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
}

int call(RadLaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

/* libref's launches by op name (RADIANCE_HOME), null where not loadable. */
struct Ref {
    void* h = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    int (*cnt)(void) = nullptr;
    RadLaunchFn fn(const char* op) const {
        if (!h) return nullptr;
        for (int i = 0; i < cnt(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k && k->op && !std::strcmp(k->op, op)) return k->launch;
        }
        return nullptr;
    }
};
Ref load_libref() {
    Ref r;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return r;
    void* h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return r;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
    if (open && open() < 0) return r;
    r.cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    r.at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (r.cnt && r.at) r.h = h;
    return r;
}

/* quant_roundtrip(fp4_e2m1, e4m3, 16) on one block of 16, in place (libref's rule; the nvfp4 latent). */
void roundtrip16(uint16_t* v) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(rad_bf16_to_f32(v[i])));
    amax = std::max(amax, 6.0f * 0x1p-9f);
    const float scale = rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(amax / 6.0f));
    for (int i = 0; i < 16; ++i) {
        const float x = std::clamp(rad_bf16_to_f32(v[i]) / scale, -6.0f, 6.0f), ax = std::fabs(x);
        unsigned best = 0;
        for (unsigned c = 1; c < 8; ++c) {
            const float d = std::fabs(kMag[c] - ax), db = std::fabs(kMag[best] - ax);
            if (d < db || (d == db && (c & 1u) == 0u)) best = c;
        }
        v[i] = rad_f32_to_bf16((std::signbit(x) ? -1.0f : 1.0f) * kMag[best] * scale);
    }
}

/* ------------------------------------------------------------------ the tree
 * DSpark's layout: node 0 the root, nodes 1..K the main chain, then B branches of depth L at position 1,
 * depth-major (node K + (k-1) B + b + 1 is branch b at depth k). */
struct Tree {
    int K = 5, B = 3, L = 5;
    std::vector<int> par;                 /* par[j]: node j's parent (-1 the root) */
    std::vector<int> dep;
    int nodes() const { return (int)par.size(); }
    /* node j's ancestors by depth, j last */
    std::vector<int> path(int j) const {
        std::vector<int> p((size_t)dep[(size_t)j] + 1);
        for (int x = dep[(size_t)j]; x >= 0; --x) { p[(size_t)x] = j; j = par[(size_t)j]; }
        return p;
    }
};
Tree make_tree(int K, int B, int L) {
    Tree t;
    t.K = K;
    t.B = B;
    t.L = L;
    t.par.push_back(-1);
    t.dep.push_back(0);
    for (int j = 1; j <= K; ++j) { t.par.push_back(j - 1); t.dep.push_back(j); }
    for (int k = 1; k <= L; ++k)
        for (int b = 0; b < B; ++b) {
            const int n = K + (k - 1) * B + b + 1;
            t.par.push_back(k == 1 ? 0 : n - B);
            t.dep.push_back(k);
        }
    return t;
}

/* A step: sequences of rows; a tree sequence (its tree and anchor) or an ordinary one (positions p0..). */
struct Seq {
    bool tree = false;
    Tree t;
    int anchor = 0;            /* tree: the root's position; ordinary: the first row's */
    int rows = 0;
};
struct Step {
    std::vector<Seq> seqs;
    std::vector<int32_t> cu, pos, spar;   /* spar: RadBatch::spec_parent */
    int M = 0;
    /* tree row m: its sequence, node; ordinary: node -1 */
    std::vector<int> seq_of, node_of;
};
Step make_step(const std::vector<Seq>& seqs) {
    Step s;
    s.seqs = seqs;
    s.cu.push_back(0);
    for (size_t i = 0; i < seqs.size(); ++i) {
        const Seq& q = seqs[i];
        const int n = q.tree ? q.t.nodes() : q.rows;
        for (int j = 0; j < n; ++j) {
            s.pos.push_back(q.tree ? q.anchor + q.t.dep[(size_t)j] : q.anchor + j);
            s.spar.push_back(q.tree ? q.t.par[(size_t)j] : -2);
            s.seq_of.push_back((int)i);
            s.node_of.push_back(q.tree ? j : -1);
        }
        s.cu.push_back(s.cu.back() + n);
    }
    s.M = s.cu.back();
    return s;
}

/* tree_rows' expected table. */
std::vector<int32_t> expect_table(const Step& s) {
    std::vector<int32_t> tt((size_t)s.M * kTW, -1);
    for (int m = 0; m < s.M; ++m) {
        if (s.node_of[(size_t)m] < 0) continue;
        const Seq& q = s.seqs[(size_t)s.seq_of[(size_t)m]];
        const int b0 = s.cu[(size_t)s.seq_of[(size_t)m]];
        const std::vector<int> p = q.t.path(s.node_of[(size_t)m]);
        tt[(size_t)m * kTW] = (int32_t)(p.size() - 1);
        tt[(size_t)m * kTW + 1] = b0;
        for (size_t x = 0; x < p.size(); ++x) tt[(size_t)m * kTW + 2 + x] = b0 + p[x];
    }
    return tt;
}

/* ================================================================== (a) tree_rows */
void case_tree_rows(const Ref& ref) {
    const Step s = make_step({ Seq{ false, {}, 700, 3 }, Seq{ true, make_tree(5, 3, 5), 1000, 0 },
                               Seq{ false, {}, 90, 1 }, Seq{ true, make_tree(5, 0, 0), 77, 0 },
                               Seq{ true, make_tree(5, 2, 2), 5, 0 } });
    const std::vector<int32_t> want = expect_table(s);
    std::vector<int32_t> wspos((size_t)s.M), wmark((size_t)s.M);
    for (int m = 0; m < s.M; ++m) {
        const bool tr = want[(size_t)m * kTW] >= 0;
        const int r0 = want[(size_t)m * kTW + 1];
        wspos[(size_t)m] = tr ? s.pos[(size_t)r0] + (m - r0) : s.pos[(size_t)m];
        wmark[(size_t)m] = tr && want[(size_t)m * kTW] >= 1;
    }
    for (int k = 0; k < 2; ++k) {
        RadLaunchFn fn = k ? ref.fn("tree_rows") : r4d_rad_tree_rows;
        if (!fn) continue;
        const std::string nm = std::string("tree_rows (") + (k ? "libref" : "kernel") + ")";
        std::vector<int32_t> tt((size_t)s.M * kTW, 77), sp((size_t)s.M, 77), mk((size_t)s.M, 77);
        expect(call(fn, { tens(s.spar.data(), RAD_I32, { s.M }), tens(s.cu.data(), RAD_I32, { (int64_t)s.cu.size() }),
                          tens(s.pos.data(), RAD_I32, { s.M }), tens(tt.data(), RAD_I32, { s.M, kTW }),
                          tens(sp.data(), RAD_I32, { s.M }), tens(mk.data(), RAD_I32, { s.M }) },
                    { pint("M", s.M), pint("width", kTW) }) == RAD_OK, nm, "launch");
        int dt = 0, ds = 0, dm = 0;
        for (size_t i = 0; i < tt.size(); ++i) dt += tt[i] != want[i];
        for (int m = 0; m < s.M; ++m) { ds += sp[(size_t)m] != wspos[(size_t)m]; dm += mk[(size_t)m] != wmark[(size_t)m]; }
        expect(dt == 0, nm, "table differs", dt);
        expect(ds == 0, nm, "slot positions differ", ds);
        expect(dm == 0, nm, "marks differ", dm);
        /* a malformed parent (after the row, outside the sequence) is an ordinary row */
        std::vector<int32_t> bad = s.spar;
        const int t0 = s.cu[1];
        bad[(size_t)t0 + 3] = 9;          /* main node 3's parent after it */
        bad[(size_t)t0 + 7] = -5;
        expect(call(fn, { tens(bad.data(), RAD_I32, { s.M }), tens(s.cu.data(), RAD_I32, { (int64_t)s.cu.size() }),
                          tens(s.pos.data(), RAD_I32, { s.M }), tens(tt.data(), RAD_I32, { s.M, kTW }),
                          tens(sp.data(), RAD_I32, { s.M }), tens(mk.data(), RAD_I32, { s.M }) },
                    { pint("M", s.M), pint("width", kTW) }) == RAD_OK, nm, "launch (malformed)");
        expect(tt[(size_t)(t0 + 3) * kTW] == -1 && tt[(size_t)(t0 + 4) * kTW] == -1 && tt[(size_t)(t0 + 7) * kTW] == -1 &&
                   tt[(size_t)(t0 + 2) * kTW] == 2,
               nm, "a malformed parent (or one under it) is not an ordinary row");
        std::printf("%-40s %d rows (3 tree sequences, 2 ordinary): table / slots / marks differ %d / %d / %d\n",
                    nm.c_str(), s.M, dt, ds, dm);
    }
}

/* ================================================================== (b) attention, (c) index scores
 * A fixture over one step: window cache (blocks of 16, shuffled tables), compressed / index caches at
 * `ratio` (a block an entry, shuffled), the in-tree entries of every tree row of depth >= 1 that
 * completes a block (ratio > 1), and per row a selection biased to the newest entries (the in-path
 * ones). Entries are kept as bytes, `eb` an entry, so the bf16 and packed forms share the code. */
struct Fix {
    Step s;
    int H = 8, ratio = 0, win = 32, topk = 16, bs = 16;
    int dim = kD;                         /* 512 latent, 128 index */
    int eb = 0;                           /* entry bytes (window slot: dim * 2) */
    int64_t nwbt = 0, nwblk = 0, ncbt = 0, ncblk = 0, C = 0;
    std::vector<uint16_t> q, wc, w;       /* w: index head weights */
    std::vector<uint8_t> cc, tc;          /* compressed (or index) cache and in-tree entries, bytes */
    std::vector<int32_t> wbt, wused, cbt, sel, keep;
    std::vector<float> sink;
    int kw = 0;                           /* keep words a row, block 8 */
};

int64_t maxpos(const Step& s) {
    int64_t mx = 0;
    for (int32_t p : s.pos) mx = std::max<int64_t>(mx, p);
    return mx;
}

/* entry bytes from bf16 values: plain, or packed by the r4d entry_store (nvfp4 latent / mxfp4 key) */
std::vector<uint8_t> entries(const std::vector<uint16_t>& v, int64_t n, int dim, int ratio, const char* fmt) {
    if (!fmt) {
        std::vector<uint8_t> b(v.size() * 2);
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    }
    const bool nv = !std::strcmp(fmt, "nvfp4");
    const int slots = nv ? kNv : kMx;
    std::vector<uint16_t> pk((size_t)(n * slots), 0x5a5a);
    std::vector<int32_t> page((size_t)n);
    for (int64_t i = 0; i < n; ++i) page[(size_t)i] = (int32_t)i;
    const int rc = call(nv ? r4d_ds41_entry_store_nv4 : r4d_ds41_entry_store_mx4,
                        { tens(v.data(), RAD_BF16, { n, dim }), tens(page.data(), RAD_I32, { n }),
                          tens(pk.data(), RAD_BF16, { n, 1, ratio, slots / ratio }) },
                        { pint("M", n), pint("n", dim), pint("block_size", ratio), pstr("dtype", "bf16"),
                          pstr("format", fmt) });
    expect(rc == RAD_OK, "fixture", "entry_store packed");
    std::vector<uint8_t> b(pk.size() * 2);
    std::memcpy(b.data(), pk.data(), b.size());
    return b;
}

Fix make_fix(uint32_t seed, const Step& s, int H, int ratio, int dim, int win, int topk, const char* fmt,
             bool keep) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    Fix f;
    f.s = s;
    f.H = H;
    f.ratio = ratio;
    f.dim = dim;
    f.win = win;
    f.topk = topk;
    const int S = (int)s.seqs.size();
    auto bf = [&](float sc) { return rad_f32_to_bf16(nd(rng) * sc); };
    const int64_t maxp = maxpos(s) + 1 + 32;
    f.q.resize((size_t)s.M * H * dim);
    for (auto& x : f.q) x = bf(dim == kD ? 0.05f : 0.3f);
    f.w.resize((size_t)s.M * H);
    for (auto& x : f.w) x = bf(1.0f);
    /* window: slot positions 0 .. maxp, tables from position 0 */
    f.nwbt = (maxp + f.bs - 1) / f.bs + 1;
    f.nwblk = S * f.nwbt + 3;
    std::vector<int32_t> perm((size_t)f.nwblk);
    for (int64_t i = 0; i < f.nwblk; ++i) perm[(size_t)i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    f.wbt.assign((size_t)(S * f.nwbt), -1);
    for (int i = 0; i < S; ++i)
        for (int64_t b = 0; b < f.nwbt; ++b) f.wbt[(size_t)(i * f.nwbt + b)] = perm[(size_t)(i * f.nwbt + b)];
    f.wc.resize((size_t)(f.nwblk * f.bs * kD));
    for (auto& x : f.wc) x = bf(1.0f);
    for (int i = 0; i < S; ++i) {
        const Seq& q = s.seqs[(size_t)i];
        const int n = s.cu[(size_t)i + 1] - s.cu[(size_t)i];
        /* a tree sequence holds slot positions up to anchor + rows - 1 (row-indexed), an ordinary one its last row */
        f.wused.push_back(q.tree ? q.anchor + n : s.pos[(size_t)s.cu[(size_t)i + 1] - 1] + 1);
    }
    if (ratio > 0) {
        /* compressed entries: one a block of `ratio` slot positions */
        const int64_t nent = maxp / ratio + 2;
        f.ncbt = nent;
        f.ncblk = S * f.ncbt + 5;
        std::vector<int32_t> cperm((size_t)f.ncblk);
        for (int64_t i = 0; i < f.ncblk; ++i) cperm[(size_t)i] = (int32_t)i;
        std::shuffle(cperm.begin(), cperm.end(), rng);
        f.cbt.assign((size_t)(S * f.ncbt), -1);
        for (int i = 0; i < S; ++i)
            for (int64_t c = 0; c < f.ncbt; ++c) f.cbt[(size_t)(i * f.ncbt + c)] = cperm[(size_t)(i * f.ncbt + c)];
        auto vals = [&](int64_t n) {
            std::vector<uint16_t> v((size_t)(n * dim));
            for (auto& x : v) x = bf(dim == kD ? 1.0f : 0.5f);
            if (fmt && !std::strcmp(fmt, "nvfp4"))
                for (int64_t i = 0; i < n * dim / 16; ++i) roundtrip16(&v[(size_t)i * 16]);
            return v;
        };
        f.cc = entries(vals(f.ncblk), f.ncblk, dim, ratio, fmt);
        if (ratio > 1) f.tc = entries(vals(s.M), s.M, dim, ratio, fmt);
        f.eb = (int)(f.cc.size() / (size_t)f.ncblk);
        /* selections: the newest entries first (the in-path ones), then older, some -1 */
        f.sel.assign((size_t)s.M * topk, -1);
        for (int m = 0; m < s.M; ++m) {
            const int64_t reach = ((int64_t)s.pos[(size_t)m] + 1) / ratio;
            std::uniform_int_distribution<int64_t> any(0, std::max<int64_t>(0, reach - 1));
            for (int t = 0; t < topk; ++t) {
                int64_t c = t < 6 ? reach - 1 - t : any(rng);
                if (pct(rng) < 6 || c < 0) c = -1;
                f.sel[(size_t)m * topk + t] = (int32_t)c;
            }
        }
        f.C = maxp / ratio + 1;
        if (keep) {
            f.kw = (int)(((f.C + 7) / 8 + 31) / 32);
            f.keep.resize((size_t)s.M * f.kw);
            for (auto& x : f.keep) x = (int32_t)(rng() | rng() << 7);
        }
    }
    f.sink.resize((size_t)H);
    for (auto& x : f.sink) x = nd(rng);
    return f;
}

/* The chain-equivalent of tree row m: one sequence of its path's d + 1 rows at the anchor's positions, the
 * window and compressed caches with the path's keys / entries where a chain step leaves them (slot
 * position = position), the row's q, selection and keep as the last row's. */
Fix chain_of(const Fix& f, const std::vector<int32_t>& tt, int m) {
    const Step& s = f.s;
    const int si = s.seq_of[(size_t)m];
    const int d = tt[(size_t)m * kTW], r0 = tt[(size_t)m * kTW + 1], A = s.pos[(size_t)r0];
    Seq q;
    q.tree = false;
    q.anchor = A;
    q.rows = d + 1;
    Fix c = f;
    c.s = make_step({ q });
    const int M = d + 1;
    /* one sequence: the tree sequence's tables as row 0 */
    c.wbt.assign(f.wbt.begin() + (int64_t)si * f.nwbt, f.wbt.begin() + (int64_t)(si + 1) * f.nwbt);
    c.wused = { A + d + 1 };
    if (f.ratio > 0) c.cbt.assign(f.cbt.begin() + (int64_t)si * f.ncbt, f.cbt.begin() + (int64_t)(si + 1) * f.ncbt);
    auto wslot = [&](const Fix& g, int64_t sp) -> uint16_t* {
        const int64_t blk = g.wbt[(size_t)(sp / g.bs)];
        return const_cast<uint16_t*>(&g.wc[(size_t)((blk * g.bs + sp % g.bs) * kD)]);
    };
    auto centry = [&](const Fix& g, int64_t c_) -> uint8_t* {
        const int64_t blk = g.cbt[(size_t)c_];
        return const_cast<uint8_t*>(&g.cc[(size_t)(blk * g.eb)]);
    };
    Fix tbl = f;   /* the tree's tables as one sequence's, to read the tree caches */
    tbl.wbt = c.wbt;
    if (f.ratio > 0) tbl.cbt = c.cbt;
    for (int x = 1; x <= d; ++x) {
        const int an = tt[(size_t)m * kTW + 2 + x];
        std::memcpy(wslot(c, A + x), wslot(tbl, A + (an - r0)), kD * 2);
        if (f.ratio == 1) std::memcpy(centry(c, A + x), centry(tbl, A + (an - r0)), (size_t)f.eb);
        if (f.ratio > 1 && (A + x + 1) % f.ratio == 0)
            std::memcpy(centry(c, (A + x + 1) / f.ratio - 1), &f.tc[(size_t)an * f.eb], (size_t)f.eb);
    }
    const int64_t qw = (int64_t)f.H * f.dim;
    c.q.assign((size_t)M * qw, 0);
    c.w.assign((size_t)M * f.H, 0);
    for (int x = 0; x <= d; ++x) {
        const int an = tt[(size_t)m * kTW + 2 + x];
        std::memcpy(&c.q[(size_t)x * qw], &f.q[(size_t)an * qw], (size_t)qw * 2);
        std::memcpy(&c.w[(size_t)x * f.H], &f.w[(size_t)an * f.H], (size_t)f.H * 2);
    }
    if (f.ratio > 0) {
        c.sel.assign((size_t)M * f.topk, -1);
        std::copy(f.sel.begin() + (int64_t)m * f.topk, f.sel.begin() + (int64_t)(m + 1) * f.topk,
                  c.sel.begin() + (int64_t)d * f.topk);
    }
    if (!f.keep.empty()) {
        c.keep.assign((size_t)M * f.kw, 0);
        std::copy(f.keep.begin() + (int64_t)m * f.kw, f.keep.begin() + (int64_t)(m + 1) * f.kw,
                  c.keep.begin() + (int64_t)d * f.kw);
    }
    c.tc.clear();
    return c;
}

/* Ordinary sequence si alone (its rows, tables, q, selections, keep): the plain op over it runs the form
 * a decode step of it does (<= 48 rows: the decode band), the reference for its rows in a tree step. */
Fix seq_alone(const Fix& f, int si) {
    const Step& s = f.s;
    Fix c = f;
    c.s = make_step({ s.seqs[(size_t)si] });
    const int t0 = s.cu[(size_t)si], M = s.cu[(size_t)si + 1] - t0;
    c.wbt.assign(f.wbt.begin() + (int64_t)si * f.nwbt, f.wbt.begin() + (int64_t)(si + 1) * f.nwbt);
    c.wused = { f.wused[(size_t)si] };
    if (f.ratio > 0) c.cbt.assign(f.cbt.begin() + (int64_t)si * f.ncbt, f.cbt.begin() + (int64_t)(si + 1) * f.ncbt);
    const int64_t qw = (int64_t)f.H * f.dim;
    c.q.assign(f.q.begin() + (int64_t)t0 * qw, f.q.begin() + (int64_t)(t0 + M) * qw);
    c.w.assign(f.w.begin() + (int64_t)t0 * f.H, f.w.begin() + (int64_t)(t0 + M) * f.H);
    if (f.ratio > 0) c.sel.assign(f.sel.begin() + (int64_t)t0 * f.topk, f.sel.begin() + (int64_t)(t0 + M) * f.topk);
    if (!f.keep.empty()) c.keep.assign(f.keep.begin() + (int64_t)t0 * f.kw, f.keep.begin() + (int64_t)(t0 + M) * f.kw);
    c.tc.clear();
    return c;
}

int attn(RadLaunchFn fn, const Fix& f, const char* fmt, const std::vector<int32_t>* tt, std::vector<uint16_t>& out) {
    const Step& s = f.s;
    const int S = (int)s.seqs.size();
    const bool nv = fmt != nullptr;
    const int cs = nv ? kNv : kD;
    out.assign((size_t)s.M * f.H * kD, 0x7171);
    std::vector<RadTensor> t = {
        tens(f.q.data(), RAD_BF16, { s.M, (int64_t)f.H * kD }),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { S }),
        f.ratio > 0 ? tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, cs / f.ratio }) : none(),
        f.ratio > 0 ? tens(f.cbt.data(), RAD_I32, { S, f.ncbt }) : none(),
        f.ratio > 0 ? tens(f.sel.data(), RAD_I32, { s.M, f.topk }) : none(),
        tens(s.pos.data(), RAD_I32, { s.M }),
        tens(s.cu.data(), RAD_I32, { S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out.data(), RAD_BF16, { s.M, (int64_t)f.H * kD }),
    };
    if (tt) {
        t.push_back(tens(tt->data(), RAD_I32, { s.M, kTW }));
        t.push_back(f.ratio > 1 ? tens(f.tc.data(), RAD_BF16, { s.M, 1, f.ratio, cs / f.ratio }) : none());
    }
    std::vector<RadParam> p = { pint("M", s.M), pint("heads", f.H), pint("dim", kD), pint("window", f.win),
                                pint("ratio", f.ratio), pint("topk", f.ratio > 0 ? f.topk : 0),
                                pf64("scale", 1.0 / std::sqrt((double)kD)), pstr("dtype", "bf16") };
    if (nv) p.push_back(pstr("latent_format", "nvfp4"));
    return call(fn, t, p);
}

int iscore(RadLaunchFn fn, const Fix& f, const char* fmt, const std::vector<int32_t>* tt, std::vector<float>& out) {
    const Step& s = f.s;
    const int S = (int)s.seqs.size();
    const bool mx = fmt != nullptr;
    const int ks = mx ? kMx : kDi;
    out.assign((size_t)s.M * f.C, 7.0f);
    std::vector<RadTensor> t = {
        tens(f.q.data(), RAD_BF16, { s.M, (int64_t)f.H * kDi }),
        tens(f.w.data(), RAD_BF16, { s.M, f.H }),
        tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, ks / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { S, f.ncbt }),
        tens(s.pos.data(), RAD_I32, { s.M }),
        tens(s.cu.data(), RAD_I32, { S + 1 }),
        tens(out.data(), RAD_F32, { s.M, f.C }),
        f.keep.empty() ? none() : tens(f.keep.data(), RAD_I32, { s.M, f.kw }),
    };
    if (tt) {
        t.push_back(tens(tt->data(), RAD_I32, { s.M, kTW }));
        t.push_back(f.ratio > 1 ? tens(f.tc.data(), RAD_BF16, { s.M, 1, f.ratio, ks / f.ratio }) : none());
    }
    std::vector<RadParam> p = { pint("M", s.M), pint("heads", f.H), pint("dim", kDi), pint("ratio", f.ratio),
                                pint("width", f.C), pf64("wscale", 1.0 / std::sqrt((double)kDi * f.H)),
                                pstr("dtype", "bf16") };
    if (!f.keep.empty()) p.push_back(pint("block", 8));
    if (mx) p.push_back(pstr("key_format", "mxfp4"));
    return call(fn, t, p);
}

/* Every tree row against its chain: `run` gives a row's outputs (width `w` elements of type T). */
template <class T>
void tree_vs_chain(const std::string& name, const Fix& f, int64_t w,
                   const std::function<int(const Fix&, const std::vector<int32_t>*, std::vector<T>&)>& run) {
    const std::vector<int32_t> tt = expect_table(f.s);
    std::vector<T> ot;
    expect(run(f, &tt, ot) == RAD_OK, name, "tree launch");
    int64_t rows = 0, diff = 0, nz = 0;
    for (int m = 0; m < f.s.M; ++m) {
        if (tt[(size_t)m * kTW] < 0) continue;
        const Fix c = chain_of(f, tt, m);
        std::vector<T> oc;
        expect(run(c, nullptr, oc) == RAD_OK, name, "chain launch");
        const int d = tt[(size_t)m * kTW];
        diff += std::memcmp(&ot[(size_t)(m * w)], &oc[(size_t)(d * w)], (size_t)w * sizeof(T)) != 0;
        for (int64_t i = 0; i < w; ++i) {
            T z{};
            nz += std::memcmp(&ot[(size_t)(m * w + i)], &z, sizeof(T)) != 0;
        }
        ++rows;
    }
    /* ordinary rows: the plain op over their sequence alone (a decode step of it) gives the same outputs */
    int64_t odiff = 0;
    for (size_t si = 0; si < f.s.seqs.size(); ++si) {
        if (f.s.seqs[si].tree) continue;
        const Fix c = seq_alone(f, (int)si);
        std::vector<T> op;
        expect(run(c, nullptr, op) == RAD_OK, name, "plain launch");
        const int t0 = f.s.cu[si];
        for (int m = t0; m < f.s.cu[si + 1]; ++m)
            odiff += std::memcmp(&ot[(size_t)(m * w)], &op[(size_t)((m - t0) * w)], (size_t)w * sizeof(T)) != 0;
    }
    expect(diff == 0, name, "a tree row differs from its chain", diff, rows);
    expect(odiff == 0, name, "an ordinary row differs from the plain op over its sequence alone", odiff);
    expect(nz > rows * w / 4, name, "the outputs are mostly zero (the fixture reached no keys)", nz, rows * w);
    std::printf("%-58s %3d rows, %2lld tree rows vs their chains: %lld differ; ordinary rows vs the plain op alone: %lld differ\n",
                name.c_str(), f.s.M, (long long)rows, (long long)diff, (long long)odiff);
}

std::vector<Seq> tree_step(int trees) {
    std::vector<Seq> v;
    v.push_back(Seq{ false, {}, 401, 4 });
    for (int i = 0; i < trees; ++i) v.push_back(Seq{ true, make_tree(5, 3, 5), 300 + 37 * i + (i & 1), 0 });
    v.push_back(Seq{ true, make_tree(5, 0, 0), 222, 0 });   /* a chain-shaped tree (dspark_tree 0) */
    v.push_back(Seq{ false, {}, 511, 1 });
    return v;
}

void case_attn(const Ref& ref) {
    struct C { const char* nm; int trees, ratio, win; const char* fmt; };
    const C cs[] = {
        { "attn tree, window only, decode band", 1, 0, 32, nullptr },
        { "attn tree, ratio 1, decode band", 1, 1, 32, nullptr },
        { "attn tree, ratio 2, decode band", 1, 2, 32, nullptr },
        { "attn tree, ratio 2, nvfp4, decode band", 1, 2, 32, "nvfp4" },
        { "attn tree, ratio 1, nvfp4, decode band", 1, 1, 64, "nvfp4" },
        { "attn tree, ratio 2, > 48 rows (head-group kernel)", 3, 2, 32, nullptr },
        { "attn tree, ratio 1, nvfp4, > 48 rows", 3, 1, 32, "nvfp4" },
    };
    uint32_t seed = 101;
    for (const C& c : cs) {
        const Fix f = make_fix(seed++, make_step(tree_step(c.trees)), 8, c.ratio, kD, c.win, 16, c.fmt, false);
        for (int k = 0; k < 2; ++k) {
            RadLaunchFn plain = k ? ref.fn("attn_sparse_sink") : r4d_rad_ds41_attn_sparse_sink;
            RadLaunchFn tree = k ? ref.fn("attn_sparse_sink_tree") : r4d_rad_ds41_attn_sparse_sink_tree;
            if (!plain || !tree) continue;
            tree_vs_chain<uint16_t>(std::string(c.nm) + (k ? " (libref)" : " (kernel)"), f, (int64_t)f.H * kD,
                                    [&](const Fix& g, const std::vector<int32_t>* tt, std::vector<uint16_t>& o) {
                                        return attn(tt ? tree : plain, g, c.fmt, tt, o);
                                    });
        }
    }
}

void case_index(const Ref& ref) {
    struct C { const char* nm; int trees, ratio; const char* fmt; bool keep; };
    const C cs[] = {
        { "index_score tree, ratio 1", 1, 1, nullptr, false },
        { "index_score tree, ratio 2", 1, 2, nullptr, false },
        { "index_score tree, ratio 2, mxfp4", 1, 2, "mxfp4", false },
        { "index_score tree, ratio 1, mxfp4, keep", 1, 1, "mxfp4", true },
        { "index_score tree, ratio 2, keep", 1, 2, nullptr, true },
        { "index_score tree, ratio 2, mxfp4, > 48 rows", 3, 2, "mxfp4", false },
    };
    uint32_t seed = 201;
    for (const C& c : cs) {
        const Fix f = make_fix(seed++, make_step(tree_step(c.trees)), 4, c.ratio, kDi, 32, 16, c.fmt, c.keep);
        for (int k = 0; k < 2; ++k) {
            RadLaunchFn plain = k ? ref.fn("index_score") : r4d_rad_ds41_index_score;
            RadLaunchFn tree = k ? ref.fn("index_score_tree") : r4d_rad_ds41_index_score_tree;
            if (!plain || !tree) continue;
            tree_vs_chain<float>(std::string(c.nm) + (k ? " (libref)" : " (kernel)"), f, f.C,
                                 [&](const Fix& g, const std::vector<int32_t>* tt, std::vector<float>& o) {
                                     return iscore(tt ? tree : plain, g, c.fmt, tt, o);
                                 });
        }
    }
}

/* ================================================================== (d) qsa_work */
struct QOut {
    std::vector<float> stage;
    std::vector<int32_t> page, bpos, nc, mpage, tpage;
};
int qsa(RadLaunchFn fn, const Step& s, const std::vector<float>& k, const std::vector<float>& tail, int slots,
        const std::vector<int32_t>& sidx, const std::vector<int32_t>& bt, int nbt, int n, int ratio, int ring, int work,
        const std::vector<int32_t>* tt, QOut& o) {
    const int S = (int)s.seqs.size();
    o.stage.assign((size_t)work * ratio * n, -9.0f);
    o.page.assign((size_t)work, -7);
    o.bpos.assign((size_t)work, -7);
    o.nc.assign((size_t)S, -7);
    o.mpage.assign((size_t)work, -7);
    o.tpage.assign((size_t)work, -7);
    std::vector<RadTensor> t = {
        tens(k.data(), RAD_F32, { s.M, n }), tens(s.pos.data(), RAD_I32, { s.M }),
        tens(s.cu.data(), RAD_I32, { S + 1 }), tens(tail.data(), RAD_F32, { slots, 1, ring, n }),
        tens(sidx.data(), RAD_I32, { S, 1 }), tens(bt.data(), RAD_I32, { S, nbt }), none(),
        tens(o.stage.data(), RAD_F32, { (int64_t)work * ratio, n }), tens(o.page.data(), RAD_I32, { work }),
        tens(o.bpos.data(), RAD_I32, { work }), tens(o.nc.data(), RAD_I32, { S }),
    };
    if (tt) {
        t.push_back(tens(tt->data(), RAD_I32, { s.M, kTW }));
        t.push_back(tens(o.mpage.data(), RAD_I32, { work }));
        t.push_back(tens(o.tpage.data(), RAD_I32, { work }));
    }
    return call(fn, t, { pint("M", s.M), pint("n", n), pint("ratio", ratio), pint("ring", ring), pint("seqs", S),
                         pint("work", work), pstr("dtype", "f32") });
}

void case_qsa(const Ref& ref) {
    std::mt19937 rng(301);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int ratio : { 2, 4 }) {
        const int n = 8, ring = ratio + 20, work = 64, slots = 4;
        const Step s = make_step({ Seq{ false, {}, 403, 3 }, Seq{ true, make_tree(5, 3, 5), 300 + ratio - 1, 0 },
                                   Seq{ true, make_tree(5, 2, 3), 151, 0 } });
        const int S = (int)s.seqs.size();
        const std::vector<int32_t> tt = expect_table(s);
        std::vector<float> k((size_t)s.M * n), tail((size_t)slots * ring * n);
        for (auto& x : k) x = nd(rng);
        for (auto& x : tail) x = nd(rng);
        const std::vector<int32_t> sidx = { 2, 0, 3 };
        const int nbt = 600 / ratio;
        std::vector<int32_t> bt((size_t)S * nbt);
        for (auto& x : bt) x = (int32_t)(rng() % 5000);
        for (int kk = 0; kk < 2; ++kk) {
            RadLaunchFn plain = kk ? ref.fn("qsa_work") : r4d_rad_qsa_work;
            RadLaunchFn tree = kk ? ref.fn("qsa_work_tree") : r4d_rad_qsa_work_tree;
            if (!plain || !tree) continue;
            const std::string nm = std::string("qsa_work tree, ratio ") + std::to_string(ratio) + (kk ? " (libref)" : " (kernel)");
            QOut ot;
            expect(qsa(tree, s, k, tail, slots, sidx, bt, nbt, n, ratio, ring, work, &tt, ot) == RAD_OK, nm, "tree launch");
            /* the expected items: sequence 0 as qsa_work alone; each tree sequence's completing rows in order */
            int w = 0, items = 0, bad = 0;
            {
                const Step s0 = make_step({ s.seqs[0] });
                std::vector<float> k0(k.begin(), k.begin() + (int64_t)s.cu[1] * n);
                QOut o0;
                expect(qsa(plain, s0, k0, tail, slots, { sidx[0] }, std::vector<int32_t>(bt.begin(), bt.begin() + nbt), nbt, n,
                           ratio, ring, work, nullptr, o0) == RAD_OK, nm, "plain launch (ordinary)");
                for (int i = 0; i < work && o0.page[(size_t)i] != -1; ++i, ++w) {
                    bad += ot.page[(size_t)w] != o0.page[(size_t)i] || ot.mpage[(size_t)w] != o0.page[(size_t)i] ||
                           ot.tpage[(size_t)w] != -1 || ot.bpos[(size_t)w] != o0.bpos[(size_t)i] ||
                           std::memcmp(&ot.stage[(size_t)w * ratio * n], &o0.stage[(size_t)i * ratio * n],
                                       (size_t)ratio * n * 4) != 0;
                    ++items;
                }
                bad += ot.nc[0] != o0.nc[0];
            }
            for (int si = 1; si < S; ++si) {
                const int t0 = s.cu[(size_t)si], t1 = s.cu[(size_t)si + 1];
                int mx = 0;
                for (int m = t0; m < t1; ++m) {
                    mx = std::max(mx, s.pos[(size_t)m]);
                    if ((s.pos[(size_t)m] + 1) % ratio) continue;
                    /* the chain of row m's path: its rows, the same tail */
                    const int d = tt[(size_t)m * kTW], A = s.pos[(size_t)t0];
                    const Step sc = make_step({ Seq{ false, {}, A, d + 1 } });
                    std::vector<float> kc((size_t)(d + 1) * n);
                    for (int x = 0; x <= d; ++x)
                        std::memcpy(&kc[(size_t)x * n], &k[(size_t)tt[(size_t)m * kTW + 2 + x] * n], (size_t)n * 4);
                    QOut oc;
                    expect(qsa(plain, sc, kc, tail, slots, { sidx[(size_t)si] },
                               std::vector<int32_t>(bt.begin() + (int64_t)si * nbt, bt.begin() + (int64_t)(si + 1) * nbt), nbt,
                               n, ratio, ring, work, nullptr, oc) == RAD_OK, nm, "chain launch");
                    /* the chain's item for this block */
                    const int b = (s.pos[(size_t)m] + 1) / ratio - 1;
                    int ci = -1;
                    for (int i = 0; i < work && oc.page[(size_t)i] != -1; ++i)
                        if (oc.bpos[(size_t)i] == b * ratio) ci = i;
                    bad += ci < 0;
                    if (ci >= 0) {
                        const int32_t pg = oc.page[(size_t)ci];
                        bad += ot.page[(size_t)w] != pg || ot.bpos[(size_t)w] != b * ratio ||
                               ot.mpage[(size_t)w] != (d == 0 ? pg : -1) || ot.tpage[(size_t)w] != (d >= 1 ? m : -1) ||
                               std::memcmp(&ot.stage[(size_t)w * ratio * n], &oc.stage[(size_t)ci * ratio * n],
                                           (size_t)ratio * n * 4) != 0;
                    }
                    ++w;
                    ++items;
                }
                bad += ot.nc[(size_t)si] != (mx + 1) / ratio;
            }
            for (int i = w; i < work; ++i) bad += ot.page[(size_t)i] != -1 || ot.mpage[(size_t)i] != -1 || ot.tpage[(size_t)i] != -1;
            expect(bad == 0, nm, "an item differs from its chain's (or qsa_work's)", bad, items);
            std::printf("%-58s %3d rows, %2d items (staged keys, pages, positions) vs the chains / qsa_work: %d differ\n",
                        nm.c_str(), s.M, items, bad);
        }
    }
}

/* ================================================================== (e) ngram_ids */
int ngram(RadLaunchFn fn, const Step& s, const std::vector<int32_t>& tok, std::vector<int32_t>& state, int W,
          const std::vector<int32_t>& cidx, const std::vector<int32_t>& nacc, const std::vector<int64_t>& mult,
          const std::vector<int64_t>& vsz, const std::vector<int64_t>& off, int heads, int ngr, int eos,
          const std::vector<int32_t>* tt, std::vector<int32_t>& ids) {
    const int S = (int)s.seqs.size();
    std::vector<int32_t> hini((size_t)S, 1);
    ids.assign((size_t)s.M * heads, -3);
    std::vector<RadTensor> t = {
        tens(tok.data(), RAD_I32, { s.M }), tens(state.data(), RAD_I32, { (int64_t)(state.size() / W), W }),
        tens(s.cu.data(), RAD_I32, { S + 1 }), tens(mult.data(), RAD_I64, { ngr }),
        tens(vsz.data(), RAD_I64, { heads }), tens(off.data(), RAD_I64, { heads }),
        tens(cidx.data(), RAD_I32, { S }), tens(hini.data(), RAD_I32, { S }), tens(nacc.data(), RAD_I32, { S }),
        tens(ids.data(), RAD_I32, { s.M, heads }), none(),
    };
    if (tt) t.push_back(tens(tt->data(), RAD_I32, { s.M, kTW }));
    return call(fn, t, { pint("M", s.M), pint("heads", heads), pint("ngram", ngr), pint("eos", eos) });
}

struct NgCfg {
    int ngr = 4, heads = 6, eos = 1;
    std::vector<int64_t> mult, vsz, off;
};
NgCfg ng_cfg(std::mt19937& rng) {
    NgCfg c;
    for (int i = 0; i < c.ngr; ++i) c.mult.push_back((int64_t)(rng() % 100000) * 2 + 1);
    for (int h = 0; h < c.heads; ++h) { c.vsz.push_back(1000 + h * 37); c.off.push_back(h * 1100); }
    return c;
}

void case_ngram(const Ref& ref) {
    std::mt19937 rng(401);
    const NgCfg g = ng_cfg(rng);
    const int ctx = g.ngr - 1, W = ctx + 20, slots = 3;
    const Step s = make_step({ Seq{ false, {}, 50, 2 }, Seq{ true, make_tree(5, 3, 5), 900, 0 } });
    const int S = (int)s.seqs.size();
    const std::vector<int32_t> tt = expect_table(s);
    std::vector<int32_t> tok((size_t)s.M);
    for (auto& x : tok) x = 2 + (int32_t)(rng() % 500);
    tok[(size_t)s.cu[1] + 7] = g.eos;     /* an eos inside a branch: the shifts past it read eos */
    std::vector<int32_t> state0((size_t)slots * W);
    for (auto& x : state0) x = 2 + (int32_t)(rng() % 500);
    const std::vector<int32_t> cidx = { 2, 0 }, nacc = { 3, 2 };
    for (int k = 0; k < 2; ++k) {
        RadLaunchFn plain = k ? ref.fn("ngram_ids") : r4d_rad_ngram_ids;
        RadLaunchFn tree = k ? ref.fn("ngram_ids_tree") : r4d_rad_ngram_ids_tree;
        if (!plain || !tree) continue;
        const std::string nm = std::string("ngram_ids tree") + (k ? " (libref)" : " (kernel)");
        std::vector<int32_t> st = state0, ids;
        expect(ngram(tree, s, tok, st, W, cidx, nacc, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos, &tt, ids) == RAD_OK,
               nm, "tree launch");
        /* the write-back is ngram_ids' on the same rows */
        std::vector<int32_t> sp = state0, idp;
        expect(ngram(plain, s, tok, sp, W, cidx, nacc, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos, nullptr, idp) == RAD_OK,
               nm, "plain launch");
        int bad = 0, rows = 0;
        bad += st != sp;
        for (int m = 0; m < s.cu[1]; ++m)
            bad += std::memcmp(&ids[(size_t)m * g.heads], &idp[(size_t)m * g.heads], (size_t)g.heads * 4) != 0;
        for (int m = s.cu[1]; m < s.M; ++m) {
            const int d = tt[(size_t)m * kTW];
            const Step sc = make_step({ Seq{ false, {}, s.pos[(size_t)s.cu[1]], d + 1 } });
            std::vector<int32_t> tc((size_t)d + 1);
            for (int x = 0; x <= d; ++x) tc[(size_t)x] = tok[(size_t)tt[(size_t)m * kTW + 2 + x]];
            std::vector<int32_t> sc_ = state0, idc;
            expect(ngram(plain, sc, tc, sc_, W, { cidx[1] }, { nacc[1] }, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos,
                         nullptr, idc) == RAD_OK, nm, "chain launch");
            bad += std::memcmp(&ids[(size_t)m * g.heads], &idc[(size_t)d * g.heads], (size_t)g.heads * 4) != 0;
            ++rows;
        }
        expect(bad == 0, nm, "ids or the window differ", bad, rows);
        std::printf("%-58s %3d rows, %2d tree rows vs their chains + the window vs ngram_ids: %d differ\n", nm.c_str(),
                    s.M, rows, bad);
        (void)S;
    }
}

/* ================================================================== (f) the commits
 * One tree sequence (anchor A) beside an ordinary one: the tree step, then the acceptance of a path
 * through branch `br` (or the main chain, br -1) to depth `acc`, then the context pass's batch as
 * ADV_CTXTREE stages it (the root at A, path node t at A + t + 1, a rejected row at A with slot -1). */
struct Ctx {
    Step trunk;
    std::vector<int32_t> tt, tmark, pos, slot;   /* the context pass's positions and slots */
    std::vector<int> path;                       /* the accepted nodes, depth order (without the root) */
    int A = 0;
};
Ctx make_ctx(int br, int acc, int ratio_bs, const std::vector<int32_t>& bt_of_seq1) {
    Ctx c;
    c.trunk = make_step({ Seq{ false, {}, 77, 2 }, Seq{ true, make_tree(5, 3, 5), 300, 0 } });
    const Step& s = c.trunk;
    c.tt = expect_table(s);
    c.tmark.assign((size_t)s.M, 0);
    for (int m = 0; m < s.M; ++m) c.tmark[(size_t)m] = c.tt[(size_t)m * kTW] >= 1;
    const Tree& t = s.seqs[1].t;
    c.A = s.seqs[1].anchor;
    /* the accepted path: main nodes 1..acc, or branch br's nodes at depth 1..acc */
    for (int k = 1; k <= acc; ++k) c.path.push_back(br < 0 ? k : t.K + (k - 1) * t.B + br + 1);
    c.pos = s.pos;
    c.slot.assign((size_t)s.M, -1);
    for (int m = 0; m < s.cu[1]; ++m) c.slot[(size_t)m] = 1000 + m;   /* the ordinary sequence: stored */
    const int b0 = s.cu[1];
    auto slot_of = [&](int p) { return bt_of_seq1[(size_t)(p / ratio_bs)] * ratio_bs + p % ratio_bs; };
    for (int j = 0; j < t.nodes(); ++j) c.pos[(size_t)b0 + j] = c.A;
    c.slot[(size_t)b0] = slot_of(c.A);
    for (size_t x = 0; x < c.path.size(); ++x) {
        c.pos[(size_t)b0 + c.path[x]] = c.A + (int)x + 1;
        c.slot[(size_t)b0 + c.path[x]] = slot_of(c.A + (int)x + 1);
    }
    return c;
}

void case_commits(const Ref& ref) {
    std::mt19937 rng(501);
    struct P { int br, acc; };
    const P ps[] = { { 1, 4 }, { 2, 1 }, { 0, 5 }, { -1, 3 }, { 2, 0 } };
    for (const P& pa : ps) {
        const std::string tag = pa.br < 0 ? std::string("main chain") : "branch " + std::to_string(pa.br);
        /* ---- the ring (the compressor tail: ring ratio + 20 rows of 4 f32, ratio 2 and 4) */
        for (int ratio : { 2, 4 }) {
            const int ring = ratio + 20, n = 4, slots = 3;
            std::vector<int32_t> bt(400);
            for (auto& x : bt) x = (int32_t)(rng() % 3000);
            const Ctx c = make_ctx(pa.br, pa.acc, 16, bt);
            const Step& s = c.trunk;
            const int b0 = s.cu[1];
            /* positions' values: committed q < A as f(q); the tree step writes row j at (A + j) % ring (spos) */
            auto val = [&](int key, int e) { return (float)(key * 10 + e) + 0.25f; };
            std::vector<float> st0((size_t)slots * ring * n, -1.0f);
            const int si = 1;
            for (int q = c.A - ring + 1; q < c.A; ++q)
                for (int e = 0; e < n; ++e) st0[(size_t)(si * ring + q % ring) * n + e] = val(q, e);
            for (int j = 0; j < s.seqs[1].t.nodes(); ++j)
                for (int e = 0; e < n; ++e) st0[(size_t)(si * ring + (c.A + j) % ring) * n + e] = val(100000 + j, e);
            for (int k = 0; k < 2; ++k) {
                RadLaunchFn fn = k ? ref.fn("tree_commit_rows") : r4d_rad_tree_commit_rows;
                if (!fn) continue;
                const std::string nm = "tree_commit_rows ring r" + std::to_string(ratio) + ", " + tag + " to depth " +
                                       std::to_string(pa.acc) + (k ? " (libref)" : " (kernel)");
                std::vector<float> st = st0;
                const std::vector<int32_t> sidx = { 0, si };
                expect(call(fn, { tens(c.tmark.data(), RAD_I32, { s.M }), tens(c.pos.data(), RAD_I32, { s.M }),
                                  tens(s.cu.data(), RAD_I32, { 3 }), tens(c.slot.data(), RAD_I32, { s.M }),
                                  tens(sidx.data(), RAD_I32, { 2, 1 }), tens(st.data(), RAD_F32, { slots, 1, ring, n }) },
                            { pint("M", s.M), pint("n", n), pint("ring", ring) }) == RAD_OK, nm, "launch");
                /* every position the next step can read: q in [A' - ratio + 1, A' - 1], A' = A + acc + 1 (the
                 * ring is ratio + 20: the tree step's 21 rows never reach those committed before it) */
                int bad = 0;
                for (int q = c.A + pa.acc + 2 - ratio; q <= c.A + pa.acc; ++q) {
                    const int key = q < c.A ? q : q == c.A ? 100000 : 100000 + c.path[(size_t)(q - c.A - 1)];
                    for (int e = 0; e < n; ++e) bad += st[(size_t)(si * ring + q % ring) * n + e] != val(key, e);
                }
                /* slot 0 (the ordinary sequence, all its rows tmark 0) is untouched */
                for (int i = 0; i < ring * n; ++i) bad += st[(size_t)i] != st0[(size_t)i];
                expect(bad == 0, nm, "a reachable ring row is not the accepted path's (or a leftover)", bad);
                std::printf("%-58s ring %d: %d reachable rows wrong\n", nm.c_str(), ring, bad);
                (void)b0;
            }
        }
        /* ---- the Engram window: ngram_ids_tree, the commit, then vs ngram_ids over the accepted chain */
        {
            std::vector<int32_t> bt(400);
            for (auto& x : bt) x = (int32_t)(rng() % 3000);
            const Ctx c = make_ctx(pa.br, pa.acc, 16, bt);
            const Step& s = c.trunk;
            const NgCfg g = ng_cfg(rng);
            const int ctx = g.ngr - 1, W = ctx + 20, slots = 3;
            std::vector<int32_t> tok((size_t)s.M);
            for (auto& x : tok) x = 2 + (int32_t)(rng() % 500);
            std::vector<int32_t> state0((size_t)slots * W);
            for (auto& x : state0) x = 2 + (int32_t)(rng() % 500);
            const std::vector<int32_t> cidx = { 0, 2 }, nacc = { 2, 3 };
            for (int k = 0; k < 2; ++k) {
                RadLaunchFn tree = k ? ref.fn("ngram_ids_tree") : r4d_rad_ngram_ids_tree;
                RadLaunchFn plain = k ? ref.fn("ngram_ids") : r4d_rad_ngram_ids;
                RadLaunchFn com = k ? ref.fn("tree_commit_rows") : r4d_rad_tree_commit_rows;
                if (!tree || !plain || !com) continue;
                const std::string nm = "tree_commit_rows Engram window, " + tag + " to depth " + std::to_string(pa.acc) +
                                       (k ? " (libref)" : " (kernel)");
                std::vector<int32_t> st = state0, ids;
                expect(ngram(tree, s, tok, st, W, cidx, nacc, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos, &c.tt, ids) ==
                           RAD_OK, nm, "tree step");
                expect(call(com, { tens(c.tmark.data(), RAD_I32, { s.M }), tens(c.pos.data(), RAD_I32, { s.M }),
                                   tens(s.cu.data(), RAD_I32, { 3 }), tens(c.slot.data(), RAD_I32, { s.M }),
                                   tens(cidx.data(), RAD_I32, { 2 }), tens(st.data(), RAD_I32, { slots, W }) },
                             { pint("M", s.M), pint("n", 1), pint("ring", 0), pint("off", ctx - 1) }) == RAD_OK, nm,
                       "commit");
                /* the chain step of the accepted path (root + path) from the same window */
                const Step sc = make_step({ Seq{ false, {}, c.A, pa.acc + 1 } });
                std::vector<int32_t> tc((size_t)pa.acc + 1);
                tc[0] = tok[(size_t)s.cu[1]];
                for (int x = 0; x < pa.acc; ++x) tc[(size_t)x + 1] = tok[(size_t)s.cu[1] + c.path[(size_t)x]];
                std::vector<int32_t> stc = state0, idc;
                expect(ngram(plain, sc, tc, stc, W, { cidx[1] }, { nacc[1] }, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos,
                             nullptr, idc) == RAD_OK, nm, "chain step");
                /* what the next step reads: the first ctx - 1 + acc + 1 ids of the slot */
                int bad = 0;
                for (int i = 0; i < ctx + pa.acc; ++i) bad += st[(size_t)(2 * W + i)] != stc[(size_t)(2 * W + i)];
                /* the ordinary sequence's slot: ngram_ids' own write-back, untouched by the commit */
                std::vector<int32_t> sp = state0, idp;
                expect(ngram(plain, s, tok, sp, W, cidx, nacc, g.mult, g.vsz, g.off, g.heads, g.ngr, g.eos, nullptr, idp) ==
                           RAD_OK, nm, "plain step");
                for (int i = 0; i < W; ++i) bad += st[(size_t)i] != sp[(size_t)i];
                expect(bad == 0, nm, "the window the next step reads is not the chain step's", bad);
                std::printf("%-58s window %d: %d ids differ from the chain step's\n", nm.c_str(), W, bad);
            }
        }
        /* ---- the in-tree entries (ratio 2, the bf16 latent: 1024 bytes an entry) */
        {
            const int ratio = 2, sl = kD / ratio, nblk = 300;
            std::vector<int32_t> bt(400);
            std::vector<int32_t> perm(nblk);
            for (int i = 0; i < nblk; ++i) perm[(size_t)i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (size_t i = 0; i < bt.size(); ++i) bt[i] = perm[i % nblk];
            const Ctx c = make_ctx(pa.br, pa.acc, ratio, bt);
            const Step& s = c.trunk;
            std::vector<uint16_t> scratch((size_t)s.M * kD), cache0((size_t)nblk * kD);
            for (auto& x : scratch) x = (uint16_t)(rng() & 0x7fff);
            for (auto& x : cache0) x = (uint16_t)(rng() & 0x7fff);
            for (int k = 0; k < 2; ++k) {
                RadLaunchFn fn = k ? ref.fn("tree_commit_entries") : r4d_rad_tree_commit_entries;
                if (!fn) continue;
                const std::string nm = "tree_commit_entries, " + tag + " to depth " + std::to_string(pa.acc) +
                                       (k ? " (libref)" : " (kernel)");
                std::vector<uint16_t> cache = cache0;
                expect(call(fn, { tens(c.tmark.data(), RAD_I32, { s.M }), tens(c.pos.data(), RAD_I32, { s.M }),
                                  tens(c.slot.data(), RAD_I32, { s.M }), tens(scratch.data(), RAD_BF16, { s.M, 1, ratio, sl }),
                                  tens(cache.data(), RAD_BF16, { nblk, 1, ratio, sl }) },
                            { pint("M", s.M), pint("ratio", ratio) }) == RAD_OK, nm, "launch");
                /* expected: the blocks completing at path positions hold that node's entry, every other block unchanged */
                std::vector<uint16_t> want = cache0;
                const int b0 = s.cu[1];
                for (size_t x = 0; x < c.path.size(); ++x) {
                    const int p = c.A + (int)x + 1;
                    if ((p + 1) % ratio) continue;
                    std::memcpy(&want[(size_t)bt[(size_t)(p / ratio)] * kD], &scratch[(size_t)(b0 + c.path[x]) * kD], kD * 2);
                }
                int bad = 0;
                for (int b = 0; b < nblk; ++b) bad += std::memcmp(&want[(size_t)b * kD], &cache[(size_t)b * kD], kD * 2) != 0;
                expect(bad == 0, nm, "a cache block is not the accepted path's entry (or holds a rejected one)", bad);
                std::printf("%-58s %d blocks: %d differ\n", nm.c_str(), nblk, bad);
            }
        }
    }
}

/* ================================================================== (g) dspark_tree_assemble */
void case_assemble(const Ref& ref) {
    std::mt19937 rng(601);
    const int S = 3, K = 5, na = 3, P = 20;
    std::vector<int32_t> tok((size_t)S * K), alt((size_t)S * na * K);
    for (auto& x : tok) x = 10 + (int32_t)(rng() % 1000);
    for (auto& x : alt) x = 10 + (int32_t)(rng() % 1000);
    for (int l = 0; l < K; ++l) alt[(size_t)(1 * na * K + 2 * K + l)] = -1;   /* sequence 1 has no third branch */
    alt[(size_t)(2 * na * K + 0 * K + 3)] = -1;                              /* a hole inside a branch */
    for (int B = 0; B <= 3; ++B) {
        /* the layout written out */
        std::vector<int32_t> wp((size_t)S * P), wq((size_t)S * P);
        for (int s = 0; s < S; ++s)
            for (int n = 1; n <= P; ++n) {
                int t, p;
                if (n <= K) { t = tok[(size_t)(s * K + n - 1)]; p = n - 1; }
                else if (n - K - 1 < B * K) {
                    const int x = n - K - 1, k = x / B + 1, b = x % B;
                    t = alt[(size_t)(s * na * K + b * K + k - 1)];
                    if (t < 0) t = tok[(size_t)(s * K + k - 1)];
                    p = k == 1 ? 0 : n - B;
                } else { t = tok[(size_t)(s * K + K - 1)]; p = n - 1; }
                wp[(size_t)(s * P + n - 1)] = t;
                wq[(size_t)(s * P + n - 1)] = p;
            }
        for (int k = 0; k < 2; ++k) {
            RadLaunchFn fn = k ? ref.fn("dspark_tree_assemble") : r4d_rad_dspark_tree_assemble;
            if (!fn) continue;
            const std::string nm = "dspark_tree_assemble B " + std::to_string(B) + (k ? " (libref)" : " (kernel)");
            std::vector<int32_t> prop((size_t)S * P, -9), par((size_t)S * P, -9);
            expect(call(fn, { tens(tok.data(), RAD_I32, { S, K }), tens(alt.data(), RAD_I32, { S, na * K }),
                              tens(prop.data(), RAD_I32, { S, P }), tens(par.data(), RAD_I32, { S, P }) },
                        { pint("M", S), pint("K", K), pint("B", B), pint("pitch", P), pint("n_alt", na) }) == RAD_OK, nm,
                   "launch");
            int bad = 0;
            for (int i = 0; i < S * P; ++i) bad += prop[(size_t)i] != wp[(size_t)i] || par[(size_t)i] != wq[(size_t)i];
            /* every parent precedes its child, and a node's depth is at most K */
            for (int s = 0; s < S; ++s)
                for (int n = 1; n <= P; ++n) bad += par[(size_t)(s * P + n - 1)] < 0 || par[(size_t)(s * P + n - 1)] >= n;
            expect(bad == 0, nm, "the proposal or its parents differ from the layout", bad);
            std::printf("%-58s %d sequences x %d nodes: %d differ\n", nm.c_str(), S, P, bad);
        }
    }
}

}  // namespace

int main() {
    const Ref ref = load_libref();
    if (!ref.h) std::printf("note: libref not loadable (set RADIANCE_HOME): its checks are skipped\n");
    case_tree_rows(ref);
    case_attn(ref);
    case_index(ref);
    case_qsa(ref);
    case_ngram(ref);
    case_commits(ref);
    case_assemble(ref);
    std::printf("ds41 tree verify: %d checks, %d failure(s)%s\n", g_check, g_fail, ref.h ? "" : " (kernels only)");
    return g_fail ? 1 : 0;
}
