/* r4d_selftest_bi.inc -- lever BI (batch invariance, ds41-bi, 11.11.2026): r4d_selftest --bicheck.
 * Included once by r4d_selftest.cpp in front of main(); uses its helpers (find, dalloc, T, PI / PS / PF, mr_weights).
 *
 * USER RULING 11.11.: an answer that changes under load vs alone is a BUG. A row's bytes must not depend on what else is
 * in the launch. This mode measures it per op, on the card, at DeepSeek-V4.1-Flash's TP4 shapes (the declared graph of
 * ds41-int3 1116248: every row-wise op the trunk issues):
 *
 *   for M in the sweep: one launch of M rows, row i holding content i (a function of (case, operand, i, element) only),
 *   and for probe rows p in { 0, M/2, M-1 }: the output row p of the M-row launch against the same content run ALONE
 *   (M = 1), byte for byte. Weights / tables are the same bytes at every M.
 *
 * The kernel row is the one the engine's band would take at that M (the highest-priority row of the op whose
 * constraints hold at M; a row with an init hook is skipped), so the forms switch where they switch in the engine
 * (gemm_nt_ggml's decode kernel up to 48 rows and its WMMA forms above, gemm_nt's m16 / m64 / tiled rows, ...). The
 * libr4d switches read at load (RADIANCE_DS41_GG_GEMV, _GG_PF, _GG_PF_TABLE, _F32_PF) take the run's environment, so
 * one run under the stack's env and one under none cover both lines.
 *
 * One line a case:
 *   BICHECK <label> [<row at the flip>] M=<sweep>: INVARIANT | FLIPS at M=<first differing M> (<n> of <m> M differ:
 *     <list>) rows <rows by M band>
 * and a summary line "BICHECK summary: <n> cases, <k> invariant, <f> flip, <s> skipped". A FLIP is a measurement, not
 * a failure of the selftest (the exit status counts only launch / setup failures).
 *
 *   --bicheck [classes] [M list]   classes: comma list of gg,f32,bf16,norm,mhc,rope,elem,router,all (default all);
 *                                  M list: comma list (default 1..64,65,80,96,127,128,129,192,256,512,1024,2112,4160) */
#ifdef RAD_HAVE_HIP
namespace bi {

uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
uint64_t key(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return mix64(a * 0x9E3779B97F4A7C15ull ^ mix64(b + 0x632BE59BD9B4E019ull) ^ mix64(c * 0xD6E8FEB86659FD93ull + 7) ^
                 mix64(d + 0x8CB92BA72F3D8DD7ull));
}
float unif(uint64_t h) { return (float)((h >> 40) & 0xffffffu) / 16777216.0f; }
/* ~N(0, 1): four uniforms */
float normalish(uint64_t h) {
    const float s = unif(h) + unif(mix64(h ^ 1)) + unif(mix64(h ^ 2)) + unif(mix64(h ^ 3));
    return (s - 2.0f) * 1.7320508f;
}
uint16_t f32_to_f16(float f) {
    _Float16 h = (_Float16)f;
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

struct Case {
    std::string label, op, cls;
    std::vector<RadParam> ps;     /* without M */
    std::string ggtype;           /* a GGUF weight at operand 1 (native blocks), "" for none */
};

/* The op's schema, from the plugin. */
const RadOpSchema* schema(const Plugin& pl, const char* op) {
    for (int i = 0; i < pl.schema_count(); ++i)
        if (!std::strcmp(pl.schema_at(i)->op, op)) return pl.schema_at(i);
    return nullptr;
}
const RadParam* param(const std::vector<RadParam>& ps, const char* k) {
    for (const RadParam& p : ps)
        if (!std::strcmp(p.key, k)) return &p;
    return nullptr;
}
bool holds(const RadKernelInfo* r, const std::vector<RadParam>& ps) {
    for (int i = 0; i < r->n_constraints; ++i) {
        const RadConstraint& c = r->constraints[i];
        if (!std::strncmp(c.key, "device.", 7)) continue;   /* a machine fact: the card running this */
        const RadParam* p = param(ps, c.key);
        if (!p) return false;
        if (c.op == RAD_C_IN) {
            if (p->kind != RAD_P_STR || !p->sval) return false;
            const std::string set = std::string(" ") + (c.sval ? c.sval : "") + " ";
            if (set.find(std::string(" ") + p->sval + " ") == std::string::npos) return false;
            continue;
        }
        if (p->kind != RAD_P_INT) return false;
        const long long v = p->ival;
        if (c.op == RAD_C_EQ && v != c.ival) return false;
        if (c.op == RAD_C_LE && v > c.ival) return false;
        if (c.op == RAD_C_GE && v < c.ival) return false;
        if (c.op == RAD_C_DIV && (c.ival == 0 || v % c.ival)) return false;
    }
    return true;
}
/* The row the engine's band takes at these parameters: the highest priority among the op's device rows whose
 * constraints hold, the first declared on a tie. */
const RadKernelInfo* pick(const Plugin& pl, const char* op, const std::vector<RadParam>& ps) {
    const RadKernelInfo* best = nullptr;
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (std::strcmp(r->op, op) || r->domain != RAD_DOMAIN_DEVICE || !r->launch) continue;
        if (!holds(r, ps)) continue;
        if (!best || r->priority > best->priority) best = r;
    }
    return best;
}

std::vector<RadParam> with_m(const std::vector<RadParam>& ps, long long M) {
    std::vector<RadParam> o;
    o.push_back(PI("M", M));
    o.insert(o.end(), ps.begin(), ps.end());
    return o;
}

int64_t elems(const RadOpdDesc& d, int from) {
    int64_t n = 1;
    for (uint32_t i = (uint32_t)from; i < d.rank; ++i) n *= d.shape[i];
    return n;
}

struct Opd {
    RadOpdDesc da{}, db{};   /* at Ma and Mb */
    int role = 0;
    bool absent = false, row = false, gg = false;
    int64_t row_elems = 0;   /* a row operand's elements a row */
    size_t eb = 0;           /* bytes an element */
    void* dev = nullptr;     /* Mmax rows (a row operand) or the fixed bytes */
    void* work = nullptr;    /* INOUT: the launch's copy */
    std::vector<uint8_t> fixed;
};

/* One float value of a fill kind, as `dt`'s bytes. */
void put(uint8_t* at, uint32_t dt, const RadOpdDesc& d, uint64_t h) {
    float v = d.fill == RAD_FILL_SIGMOID ? 0.02f + 0.96f * unif(h) : normalish(h);
    switch (dt) {
    case RAD_F32: std::memcpy(at, &v, 4); break;
    case RAD_BF16: { const uint16_t b = h_to_bf16(v); std::memcpy(at, &b, 2); break; }
    case RAD_F16: { const uint16_t b = f32_to_f16(v); std::memcpy(at, &b, 2); break; }
    case RAD_F8E4M3: { uint8_t b = (uint8_t)(h >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; *at = b; break; }
    default: *at = (uint8_t)(h >> 24); break;
    }
}

/* Fill `n` elements; `ctx` keys the content (a row's content id, or ~0 for a fixed operand). */
bool fill(std::vector<uint8_t>& out, const RadOpdDesc& d, size_t eb, int64_t n, uint64_t seed, int opd, uint64_t ctx,
          int64_t prev_lead, std::string* why) {
    out.resize((size_t)n * eb);
    if (d.fill == RAD_FILL_INDEX) {
        if (d.flags & (RAD_OPD_F_IDX_UNIQUE | RAD_OPD_F_IDX_CU | RAD_OPD_F_IDX_PREFIX | RAD_OPD_F_IDX_HISTORY)) {
            *why = "an index operand with a structure flag";
            return false;
        }
        const int64_t mx = d.idx_max > 0 ? d.idx_max : prev_lead;
        for (int64_t j = 0; j < n; ++j) {
            const int32_t v = d.idx_const >= 0 ? (int32_t)d.idx_const
                                               : (int32_t)(mx > 0 ? key(seed, opd, ctx, j) % (uint64_t)mx : 0);
            std::memcpy(&out[(size_t)j * 4], &v, 4);
        }
        return true;
    }
    if (d.fill != RAD_FILL_NORMAL && d.fill != RAD_FILL_SIGMOID) {
        *why = fmt("fill kind %u", d.fill);
        return false;
    }
    for (int64_t j = 0; j < n; ++j) put(&out[(size_t)j * eb], d.dtype, d, key(seed, opd, ctx, j));
    return true;
}

struct Result {
    std::vector<long long> ms, flips;
    std::map<std::string, std::string> rows;   /* row name -> M range it served */
    std::string skip;
    int fails = 0;
};

/* Run one case over the sweep. */
Result run(const Plugin& pl, const Case& c, const std::vector<long long>& sweep) {
    Result R;
    const RadOpSchema* sc = schema(pl, c.op.c_str());
    if (!sc) { R.skip = "no schema"; return R; }
    const long long Mmax = *std::max_element(sweep.begin(), sweep.end());
    const long long Ma = 37, Mb = 53;
    const std::vector<RadParam> pa = with_m(c.ps, Ma), pb = with_m(c.ps, Mb);
    /* the descriptions: any row of the op that has a shape hook (they agree across rows of one op) */
    RadShapeFn sh = nullptr;
    for (int k = 0; k < pl.kernel_count() && !sh; ++k)
        if (!std::strcmp(pl.kernel_at(k)->op, c.op.c_str()) && pl.kernel_at(k)->opd_shape) sh = pl.kernel_at(k)->opd_shape;
    if (!sh) { R.skip = "no shape hook"; return R; }
    std::vector<Opd> od((size_t)sc->n_operands);
    const uint64_t seed = std::hash<std::string>{}(c.label);
    for (int i = 0; i < sc->n_operands; ++i) {
        Opd& o = od[(size_t)i];
        o.role = sc->operands[i].role;
        if (sh(pa.data(), (int)pa.size(), i, &o.da) != RAD_OK || sh(pb.data(), (int)pb.size(), i, &o.db) != RAD_OK) {
            R.skip = fmt("shape hook refused operand %d", i);
            return R;
        }
        if (o.da.flags & RAD_OPD_F_ABSENT) { o.absent = true; continue; }
        o.eb = esz(o.da.dtype);
        const bool lead_m = o.da.rank >= 1 && o.da.shape[0] == Ma && o.db.shape[0] == Mb;
        bool same = o.da.rank == o.db.rank;
        for (uint32_t d = lead_m ? 1 : 0; same && d < o.da.rank; ++d) same = o.da.shape[d] == o.db.shape[d];
        if (!same) { R.skip = fmt("operand %d (%s) scales with M other than as rows", i, sc->operands[i].name); return R; }
        o.row = lead_m;
        if (o.row && o.da.fill == RAD_FILL_INDEX && o.da.idx_max != o.db.idx_max) {
            R.skip = fmt("index operand %d's range scales with M", i);
            return R;
        }
        o.row_elems = o.row ? elems(o.da, 1) : 0;
        o.gg = !c.ggtype.empty() && i == 1;
    }
    /* allocate and fill: row operands at Mmax rows, row i = content i */
    for (int i = 0; i < sc->n_operands; ++i) {
        Opd& o = od[(size_t)i];
        if (o.absent) continue;
        std::string why;
        if (o.gg) {
            const MrTy* ty = nullptr;
            for (const MrTy& t : kMrTypes)
                if (c.ggtype == t.name) ty = &t;
            if (!ty) { R.skip = "GGUF type " + c.ggtype + " not generated"; return R; }
            const int64_t N = o.da.shape[0], K = o.da.shape[1];
            mr_weights(o.fixed, *ty, N, K, seed ^ 0x5bd1e995u);
            o.da.dtype = RAD_U8; o.da.rank = 2; o.da.shape[1] = K / ty->qk * ty->bb; o.eb = 1;
            o.dev = dalloc(o.fixed.size());
            if (!o.dev) { R.skip = "alloc"; return R; }
            st_memcpy(o.dev, o.fixed.data(), o.fixed.size(), hipMemcpyHostToDevice);
            continue;
        }
        const int64_t prev_lead = i > 0 && !od[(size_t)i - 1].absent && od[(size_t)i - 1].da.rank
                                      ? od[(size_t)i - 1].da.shape[0] : 0;
        if (o.row) {
            const size_t rb = (size_t)o.row_elems * o.eb;
            std::vector<uint8_t> all((size_t)Mmax * rb), one;
            for (long long r = 0; r < Mmax; ++r) {
                if (!fill(one, o.da, o.eb, o.row_elems, seed, i, (uint64_t)r, prev_lead, &why)) { R.skip = why; return R; }
                std::memcpy(&all[(size_t)r * rb], one.data(), rb);
            }
            o.dev = dalloc(all.size());
            if (o.role == RAD_OPD_INOUT) o.work = dalloc(all.size());
            if (!o.dev || (o.role == RAD_OPD_INOUT && !o.work)) { R.skip = "alloc"; return R; }
            if (o.role != RAD_OPD_OUT) st_memcpy(o.dev, all.data(), all.size(), hipMemcpyHostToDevice);
        } else {
            if (!fill(o.fixed, o.da, o.eb, elems(o.da, 0), seed, i, ~0ull, prev_lead, &why)) { R.skip = why; return R; }
            o.dev = dalloc(o.fixed.size());
            if (!o.dev) { R.skip = "alloc"; return R; }
            if (o.role != RAD_OPD_OUT) st_memcpy(o.dev, o.fixed.data(), o.fixed.size(), hipMemcpyHostToDevice);
        }
    }
    /* one launch: M rows starting at content row `first` (alone: M = 1, first = the content id) */
    auto launch = [&](long long M, long long first, std::string* rowname) -> int {
        const std::vector<RadParam> ps = with_m(c.ps, M);
        const RadKernelInfo* r = pick(pl, c.op.c_str(), ps);
        if (!r) return RAD_E_UNSUPPORTED;
        if (r->init) return RAD_E_UNSUPPORTED;
        if (rowname) *rowname = r->name;
        std::vector<RadTensor> ts;
        for (int i = 0; i < sc->n_operands; ++i) {
            Opd& o = od[(size_t)i];
            if (o.absent) { RadTensor t; std::memset(&t, 0, sizeof t); ts.push_back(t); continue; }
            std::vector<int64_t> shp(o.da.shape, o.da.shape + o.da.rank);
            void* base = o.dev;
            if (o.row) {
                shp[0] = M;
                const size_t rb = (size_t)o.row_elems * o.eb;
                base = (uint8_t*)o.dev + (size_t)first * rb;
                if (o.role == RAD_OPD_INOUT) {
                    st_memcpy(o.work, base, (size_t)M * rb, hipMemcpyDeviceToDevice);
                    base = o.work;
                } else if (o.role == RAD_OPD_OUT) {
                    st_memset(base, 0xA5, (size_t)M * rb);
                }
            } else if (o.role == RAD_OPD_OUT) {
                st_memset(base, 0xA5, o.fixed.size());
            }
            ts.push_back(T(base, o.da.dtype, shp));
        }
        return launch_row(r, ts, ps);
    };
    /* output bytes of row `row` of the last launch (in the OUT / INOUT row operands) */
    auto outrow = [&](long long first, long long row, std::vector<uint8_t>& v) {
        v.clear();
        for (int i = 0; i < sc->n_operands; ++i) {
            Opd& o = od[(size_t)i];
            if (o.absent || !o.row || (o.role != RAD_OPD_OUT && o.role != RAD_OPD_INOUT)) continue;
            const size_t rb = (size_t)o.row_elems * o.eb;
            const uint8_t* src = o.role == RAD_OPD_INOUT ? (const uint8_t*)o.work + (size_t)row * rb
                                                         : (const uint8_t*)o.dev + (size_t)(first + row) * rb;
            const size_t at = v.size();
            v.resize(at + rb);
            st_memcpy(&v[at], src, rb, hipMemcpyDeviceToHost);
        }
    };
    std::map<long long, std::vector<uint8_t>> alone;
    auto ref = [&](long long content) -> const std::vector<uint8_t>* {
        auto it = alone.find(content);
        if (it != alone.end()) return &it->second;
        if (launch(1, content, nullptr) != RAD_OK) return nullptr;
        std::vector<uint8_t> v;
        outrow(content, 0, v);
        return &(alone[content] = v);
    };
    bool any_out = false;
    for (const Opd& o : od) any_out |= !o.absent && o.row && (o.role == RAD_OPD_OUT || o.role == RAD_OPD_INOUT);
    if (!any_out) { R.skip = "no row output"; return R; }
    std::string band_row;
    long long band_lo = 0, prev_m = 0;
    for (long long M : sweep) {
        std::string rn;
        const int rc = launch(M, 0, &rn);
        if (rc != RAD_OK) {
            std::printf("  BIFAIL %s M=%lld launch rc %d (%s)\n", c.label.c_str(), M, rc, rn.c_str());
            ++R.fails;
            continue;
        }
        if (rn != band_row) {
            if (!band_row.empty()) R.rows[band_row] += fmt("%lld..%lld ", band_lo, prev_m);
            band_row = rn;
            band_lo = M;
        }
        prev_m = M;
        R.ms.push_back(M);
        std::vector<long long> probes = { 0, M / 2, M - 1 };
        std::sort(probes.begin(), probes.end());
        probes.erase(std::unique(probes.begin(), probes.end()), probes.end());
        std::vector<std::vector<uint8_t>> got(probes.size());
        for (size_t q = 0; q < probes.size(); ++q) outrow(0, probes[q], got[q]);
        bool differ = false;
        for (size_t q = 0; q < probes.size(); ++q) {
            const std::vector<uint8_t>* a = ref(probes[q]);
            if (!a) { ++R.fails; std::printf("  BIFAIL %s alone launch for content %lld\n", c.label.c_str(), probes[q]); continue; }
            if (*a != got[q]) differ = true;
        }
        if (differ) R.flips.push_back(M);
    }
    if (!band_row.empty()) R.rows[band_row] += fmt("%lld..%lld ", band_lo, prev_m);
    return R;
}

std::vector<Case> cases() {
    std::vector<Case> v;
    auto gg = [&](const char* lbl, long long N, long long K, const char* ty) {
        v.push_back({ fmt("gemm_nt_ggml %s %s %lldx%lld", lbl, ty, N, K), "gemm_nt_ggml", "gg",
                      { PI("N", N), PI("K", K), PS("type", ty), PS("dtype", "bf16") }, ty });
    };
    gg("attn_q_b", 8192, 1280, "q5_K");      gg("attn_q_b", 8192, 1280, "q6_K");
    gg("attn_output_b", 5120, 2048, "q5_K"); gg("attn_output_b", 5120, 2048, "q6_K");
    gg("attn_q_a", 1280, 5120, "q5_K");      gg("attn_q_a", 1280, 5120, "q6_K");
    gg("attn_output_a", 1024, 4096, "q5_K"); gg("attn_output_a", 1024, 4096, "q6_K");
    gg("attn_kv", 512, 5120, "q5_K");        gg("attn_kv", 512, 5120, "q6_K");
    gg("shexp_gate_up", 1536, 5120, "q6_K"); gg("shexp_gate_up", 1536, 5120, "q8_0");
    gg("shexp_down", 5120, 768, "q6_K");     gg("shexp_down", 5120, 768, "q8_0");
    gg("engram_kv", 25600, 6144, "q6_K");
    v.push_back({ "logits_gemm_ggml head q6_K 32320x5120", "logits_gemm_ggml", "gg",
                  { PI("n_vocab", 32320), PI("n_embd", 5120), PS("type", "q6_K"), PS("dtype", "bf16") }, "q6_K" });
    for (const char* ty : { "q5_K", "q6_K" })
        v.push_back({ fmt("rmsnorm_gemm_nt_ggml q_b %s 8192x1280 (K1 fold a)", ty), "rmsnorm_gemm_nt_ggml", "gg",
                      { PI("N", 8192), PI("K", 1280), PS("type", ty), PS("dtype", "bf16"), PF("eps", 1e-20) }, ty });
    for (const char* ty : { "q6_K", "q8_0" })
        v.push_back({ fmt("swiglu_clamped_gemm_nt_ggml shexp_down %s 5120x768 (K1 fold e)", ty),
                      "swiglu_clamped_gemm_nt_ggml", "gg",
                      { PI("N", 5120), PI("K", 768), PS("type", ty), PS("dtype", "bf16"), PF("limit", 10) }, ty });
    for (long long N : { 384LL, 96LL, 1024LL, 128LL, 32LL })
        v.push_back({ fmt("gemm_nt_f32 %lldx5120 (%s)", N, N == 384 || N == 96 ? "router" : "compressor kvg / score"),
                      "gemm_nt_f32", "f32", { PI("N", N), PI("K", 5120), PS("dtype", "bf16") }, "" });
    struct Bf { const char* lbl; long long N, K; };
    for (const Bf& b : { Bf{ "indexer attn_q_b", 4096, 1280 }, Bf{ "indexer proj", 32, 5120 },
                         Bf{ "indexer attn_k", 128, 512 }, Bf{ "compressor kv", 512, 5120 } })
        v.push_back({ fmt("gemm_nt bf16 %s %lldx%lld", b.lbl, b.N, b.K), "gemm_nt", "bf16",
                      { PI("N", b.N), PI("K", b.K), PS("dtype", "bf16") }, "" });
    for (long long n : { 5120LL, 1280LL, 512LL, 128LL })
        v.push_back({ fmt("rmsnorm n=%lld", n), "rmsnorm", "norm", { PI("n", n), PF("eps", 1e-20), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_mix n=5120 hc=4", "mhc_mix", "mhc",
                  { PI("n", 5120), PI("hc", 4), PI("iterations", 20), PF("eps", 1e-20), PF("hc_eps", 1e-6), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_collapse n=5120 hc=4", "mhc_collapse", "mhc", { PI("n", 5120), PI("hc", 4), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_collapse_rmsnorm n=5120 hc=4", "mhc_collapse_rmsnorm", "mhc",
                  { PI("n", 5120), PI("hc", 4), PF("eps", 1e-20), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_post n=5120 hc=4", "mhc_post", "mhc", { PI("n", 5120), PI("hc", 4), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_post_collapse_rmsnorm n=5120 hc=4", "mhc_post_collapse_rmsnorm", "mhc",
                  { PI("n", 5120), PI("hc", 4), PF("eps", 1e-20), PS("dtype", "bf16") }, "" });
    v.push_back({ "mhc_contract n=5120 hc=4", "mhc_contract", "mhc", { PI("n", 5120), PI("hc", 4), PS("dtype", "bf16") }, "" });
    v.push_back({ "stream_gate_add n=5120 hc=4", "stream_gate_add", "mhc",
                  { PI("n", 5120), PI("hc", 4), PF("eps", 1e-20), PF("gate_eps", 1e-6), PS("dtype", "bf16") }, "" });
    struct Rp { int nh, hd, off, inv; };
    for (const Rp& r : { Rp{ 16, 512, 448, 0 }, Rp{ 16, 512, 448, 1 }, Rp{ 1, 128, 64, 0 }, Rp{ 1, 512, 448, 0 }, Rp{ 32, 128, 64, 0 } })
        v.push_back({ fmt("ds41_rope_rows heads=%d dim=%d inverse=%d", r.nh, r.hd, r.inv), "ds41_rope_rows", "rope",
                      { PI("n_head", r.nh), PI("head_dim", r.hd), PI("rot", 64), PI("off", r.off), PS("mode", "gptj"),
                        PI("inverse", r.inv), PS("dtype", "bf16") }, "" });
    struct Qr { int n, block; const char *fmt_, *scale; };
    for (const Qr& q : { Qr{ 128, 32, "fp4_e2m1", "ue8m0" }, Qr{ 4096, 32, "fp4_e2m1", "ue8m0" }, Qr{ 512, 16, "fp4_e2m1", "e4m3" },
                         Qr{ 512, 32, "fp8_e4m3", "ue8m0" }, Qr{ 8192, 32, "fp8_e4m3", "ue8m0" } })
        v.push_back({ fmt("quant_roundtrip n=%d block=%d %s", q.n, q.block, q.fmt_), "quant_roundtrip", "rope",
                      { PI("n", q.n), PI("block", q.block), PS("format", q.fmt_), PS("scale", q.scale), PS("dtype", "bf16") }, "" });
    v.push_back({ "rmsnorm_rope_roundtrip n=128 fp4 (index key)", "rmsnorm_rope_roundtrip", "rope",
                  { PI("n", 128), PF("eps", 1e-20), PI("n_head", 1), PI("head_dim", 128), PI("rot", 64), PI("off", 64),
                    PS("mode", "gptj"), PI("inverse", 0), PI("block", 32), PS("format", "fp4_e2m1"), PS("scale", "ue8m0"),
                    PS("dtype", "bf16") }, "" });
    v.push_back({ "rmsnorm_rope_roundtrip n=512 fp8 (window key)", "rmsnorm_rope_roundtrip", "rope",
                  { PI("n", 512), PF("eps", 1e-20), PI("n_head", 1), PI("head_dim", 512), PI("rot", 64), PI("off", 448),
                    PS("mode", "gptj"), PI("inverse", 0), PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"),
                    PS("dtype", "bf16") }, "" });
    struct Rr { int nh, hd, off, inv, block; const char *fmt_, *scale; };
    for (const Rr& r : { Rr{ 16, 512, 448, 1, 32, "fp8_e4m3", "ue8m0" }, Rr{ 1, 512, 448, 0, 16, "fp4_e2m1", "e4m3" },
                         Rr{ 32, 128, 64, 0, 32, "fp4_e2m1", "ue8m0" } })
        v.push_back({ fmt("rope_rows_roundtrip heads=%d dim=%d inverse=%d %s", r.nh, r.hd, r.inv, r.fmt_), "rope_rows_roundtrip", "rope",
                      { PI("n_head", r.nh), PI("head_dim", r.hd), PI("rot", 64), PI("off", r.off), PS("mode", "gptj"),
                        PI("inverse", r.inv), PI("block", r.block), PS("format", r.fmt_), PS("scale", r.scale), PS("dtype", "bf16") }, "" });
    for (long long n : { 768LL, 2304LL })
        v.push_back({ fmt("swiglu_clamped n=%lld", n), "swiglu_clamped", "elem", { PI("n", n), PF("limit", 10), PS("dtype", "bf16") }, "" });
    for (long long n : { 5120LL, 20480LL, 6144LL })
        v.push_back({ fmt("cast bf16 n=%lld", n), "cast", "elem", { PI("n", n), PS("from", "bf16"), PS("to", "bf16") }, "" });
    v.push_back({ "softmax_pool n=512 ratio=2", "softmax_pool", "elem", { PI("n", 512), PI("ratio", 2), PS("dtype", "bf16") }, "" });
    v.push_back({ "router_scored_topk 384 top6 sqrtsoftplus", "router_scored_topk", "router",
                  { PI("n_expert", 384), PI("top_k", 6), PI("norm", 1), PF("route_scale", 1.5), PS("score", "sqrtsoftplus"),
                    PS("dtype", "f32") }, "" });
    return v;
}

std::vector<long long> default_sweep() {
    std::vector<long long> m;
    for (long long i = 1; i <= 64; ++i) m.push_back(i);
    for (long long i : { 65LL, 80LL, 96LL, 127LL, 128LL, 129LL, 192LL, 256LL, 512LL, 1024LL, 2112LL, 4160LL }) m.push_back(i);
    return m;
}

}  // namespace bi

static void ds_bi_check(const Plugin& pl, const char* classes, const char* mlist) {
    const std::string want = classes && *classes ? classes : "all";
    std::vector<long long> sweep;
    if (mlist && *mlist) {
        for (const char* s = mlist; *s;) {
            char* e = nullptr;
            const long long v = std::strtoll(s, &e, 10);
            if (e == s) break;
            if (v >= 1 && v <= 65535) sweep.push_back(v);
            s = *e == ',' ? e + 1 : e;
        }
    }
    if (sweep.empty()) sweep = bi::default_sweep();
    std::printf("lever BI per-op check: a row alone (M = 1) vs the same row inside an M-row launch, byte for byte; "
                "probe rows 0, M/2, M-1; %zu M values up to %lld; env RADIANCE_DS41_GG_GEMV=%s _GG_PF=%s _F32_PF=%s\n",
                sweep.size(), *std::max_element(sweep.begin(), sweep.end()),
                std::getenv("RADIANCE_DS41_GG_GEMV") ? std::getenv("RADIANCE_DS41_GG_GEMV") : "(unset)",
                std::getenv("RADIANCE_DS41_GG_PF") ? std::getenv("RADIANCE_DS41_GG_PF") : "(unset)",
                std::getenv("RADIANCE_DS41_F32_PF") ? std::getenv("RADIANCE_DS41_F32_PF") : "(unset)");
    int n = 0, inv = 0, flip = 0, sk = 0;
    for (const bi::Case& c : bi::cases()) {
        if (want != "all" && !mr_in(want.c_str(), c.cls.c_str())) continue;
        ++n;
        dfree_all();
        const bi::Result R = bi::run(pl, c, sweep);
        if (!R.skip.empty()) {
            ++sk;
            std::printf("BICHECK %s: SKIP (%s)\n", c.label.c_str(), R.skip.c_str());
            continue;
        }
        std::string rows;
        for (const auto& kv : R.rows) rows += kv.first + " M " + kv.second;
        if (R.fails) ++g_fail;
        if (R.flips.empty()) {
            ++inv;
            std::printf("BICHECK %s: INVARIANT at %zu of %zu M%s | rows %s\n", c.label.c_str(), R.ms.size(), sweep.size(),
                        R.fails ? fmt(" (%d launch failures)", R.fails).c_str() : "", rows.c_str());
        } else {
            ++flip;
            std::string l;
            for (size_t i = 0; i < R.flips.size(); ++i) l += fmt("%s%lld", i ? "," : "", R.flips[i]);
            std::printf("BICHECK %s: FLIPS at M=%lld (%zu of %zu M differ: %s)%s | rows %s\n", c.label.c_str(), R.flips[0],
                        R.flips.size(), R.ms.size(), l.c_str(), R.fails ? fmt(" (%d launch failures)", R.fails).c_str() : "",
                        rows.c_str());
        }
        std::fflush(stdout);
    }
    dfree_all();
    std::printf("BICHECK summary: %d cases, %d invariant, %d flip, %d skipped\n", n, inv, flip, sk);
}
#endif
