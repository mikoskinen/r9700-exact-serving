/* tests/r4d_emu: qfn-on-ds lever RGT (Qwen loop round 5, 21.11.2026) on the host SIMT emulator (the GPU twin: r4d_selftest
 * --perfrg's byte line) -- the fused decode router router_gemm_topk_scatter_inv with the ranking per 16-row tile (op param
 * rg_tail 1: each row tile's last block ranks the tile's rows and stages the slots' sort places in sorted_tok, the last tile
 * sorts) byte for byte against today's form (rg_tail 0: the grid's last block ranks every row and sorts):
 *   logits, gate, expert_ids, expert_w, sorted_tok, expert_offset, expert_count, inverse, each with a guard tail;
 *   Qwen's tp4 rank shape (512 experts, top 10, a 128-expert slice at base 128, norm 1, the shared gate) at a short K, and a
 *   small shape (64 experts, top 6, every expert local, base 0, no gate, norm 0), with a protected expert in the slice;
 *   M 17 / 33 / 64 / 96 / 128 (the Qwen shape) and 17 / 40 / 128 (the small one); random rows, tied rows (one row's logits
 *   all equal) and a NaN row; rg_mt 1 and 2 (two row tiles a block: the tile is blockIdx.y's rows); each form launched twice
 *   (the counters come back to zero: the second launch's bytes = the first's).
 * NEGATIVE CONTROLS: the same compare finds the change when norm flips (weights differ) and when one activation element
 * changes (logits differ); and a launch with rg_tail 1 at M <= 16 takes today's form (one tile: the rule's floor).
 *
 *   r4d_emu_qrgt [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_router_gemm_topk_scatter_inv(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_topk_scatter_init(const RadParam*, int, int, int, void**);
extern "C" void r4d_rad_router_gemm_topk_scatter_fini(void*);
extern "C" int r4d_rg_tail_set(int value);
extern "C" int r4d_rg_tail_get();

namespace {

int g_cases = 0, g_fail = 0;
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}
void check(bool ok, const std::string& what, const std::string& why = "") {
    ++g_cases;
    if (!ok) ++g_fail;
    std::printf("  %s  %s%s%s\n", ok ? "ok  " : "FAIL", what.c_str(), ok || why.empty() ? "" : ": ", ok ? "" : why.c_str());
}
uint64_t g_rs = 0x2545F4914F6CDD1Dull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
double uni() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
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
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }

constexpr int64_t TAIL = 16;   /* guard words past every output */

struct Shape { const char* name; int64_t ne, K, tk, nl, base; bool gate; int norm; const char* protect; };

/* one launch's outputs, poisoned 0x5a (guard tails included) before the launch */
struct Out {
    std::vector<uint16_t> lg, g, w;
    std::vector<int32_t> id, so, of, cn, inv;
    void init(int64_t M, const Shape& s) {
        const int64_t T = M * s.tk;
        lg.assign((size_t)(M * s.ne + TAIL), 0x5a5a); g.assign((size_t)(M + TAIL), 0x5a5a); w.assign((size_t)(T + TAIL), 0x5a5a);
        id.assign((size_t)(T + TAIL), 0x5a5a5a5a); so.assign((size_t)(T + TAIL), 0x5a5a5a5a);
        of.assign((size_t)(s.nl + 1 + TAIL), 0x5a5a5a5a); cn.assign((size_t)(s.nl + TAIL), 0x5a5a5a5a);
        inv.assign((size_t)(T + TAIL), 0x5a5a5a5a);
    }
};
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}
/* the first differing output, or "" */
std::string cmp(const Out& a, const Out& b) {
    struct { const char* n; long long d; } r[] = {
        { "logits", ndiff(a.lg, b.lg) }, { "gate", ndiff(a.g, b.g) }, { "expert_ids", ndiff(a.id, b.id) },
        { "expert_w", ndiff(a.w, b.w) }, { "sorted_tok", ndiff(a.so, b.so) }, { "expert_offset", ndiff(a.of, b.of) },
        { "expert_count", ndiff(a.cn, b.cn) }, { "inverse", ndiff(a.inv, b.inv) } };
    std::string s;
    for (auto& e : r)
        if (e.d) s += fmt("%s%s %lld", s.empty() ? "" : ", ", e.n, e.d);
    return s;
}

int run(const Shape& s, int64_t M, const std::vector<uint16_t>& x, const std::vector<uint16_t>& wr,
        const std::vector<uint16_t>& wg, void* inst, int tail, int mt, int norm, Out& o, int ku = 0, int tpb = -1) {
    o.init(M, s);
    const int64_t T = M * s.tk;
    std::vector<RadTensor> t = { tens(x.data(), RAD_BF16, { M, s.K }), tens(wr.data(), RAD_BF16, { s.ne, s.K }),
                                 tens(s.gate ? wg.data() : nullptr, RAD_BF16, { 1, s.K }),
                                 tens(o.lg.data(), RAD_BF16, { M, s.ne }), tens(o.id.data(), RAD_I32, { M, s.tk }),
                                 tens(o.w.data(), RAD_BF16, { M, s.tk }), tens(o.so.data(), RAD_I32, { T }),
                                 tens(o.of.data(), RAD_I32, { s.nl + 1 }), tens(o.cn.data(), RAD_I32, { s.nl }),
                                 tens(s.gate ? o.g.data() : nullptr, RAD_BF16, { M, 1 }), tens(o.inv.data(), RAD_I32, { T }) };
    std::vector<RadParam> p = { pint("M", M), pint("K", s.K), pint("n_expert", s.ne), pint("n_local", s.nl),
                                pint("top_k", s.tk), pint("norm", norm), pstr("dtype", "bf16"), pint("expert_base", s.base),
                                pint("rg_tail", tail) };
    if (mt > 1) p.push_back(pint("rg_mt", mt));
    if (ku > 0) p.push_back(pint("rg_ku", ku));     /* lever RGU (round 36): the op params; absent = the switch's form */
    if (tpb >= 0) p.push_back(pint("rg_tpb", tpb));
    if (s.protect) p.push_back(pstr("protect", s.protect));
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    a.instance = inst;
    if (!s.gate) {   /* operands 2 and 9 absent: r4d_opt reads a null data pointer as absent */
        t[2] = RadTensor{};
        t[9] = RadTensor{};
    }
    return r4d_rad_router_gemm_topk_scatter_inv(&a, nullptr);
}

void rows(std::vector<uint16_t>& x, int64_t M, int64_t K, int64_t ne, const std::vector<uint16_t>& wr) {
    x.resize((size_t)(M * K));
    for (auto& v : x) v = bf((float)((uni() * 2.0 - 1.0) * 2.0));
    /* row 3: a zero row -> every logit 0 (ties everywhere: the selection's ties go to the lower id) */
    if (M > 3) std::memset(&x[(size_t)(3 * K)], 0, (size_t)K * 2);
    /* row M - 2: a NaN element -> NaN logits */
    if (M > 4) x[(size_t)((M - 2) * K + 5)] = 0x7fc1;
    (void)ne; (void)wr;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const Shape shapes[] = {
        { "Qwen tp4 rank (512 experts, top 10, slice 128 at base 128, gate, norm 1, protect 7)", 512, 128, 10, 128, 128, true, 1, "7" },
        { "small (64 experts, top 6, all local, no gate, norm 0)", 64, 64, 6, 64, 0, false, 0, nullptr } };
    std::printf("r4d_emu_qrgt: lever RGT (fused decode router, ranking per row tile) vs today's form\n");
    /* the switch: 0 .. 128 applied, others refused; the bench param overrides it (every launch below passes rg_tail) */
    {
        const int a = r4d_rg_tail_set(65), b = r4d_rg_tail_get(), c = r4d_rg_tail_set(129), d = r4d_rg_tail_get(),
                  e = r4d_rg_tail_set(-1), f = r4d_rg_tail_set(0), g = r4d_rg_tail_get();
        check(a == 1 && b == 65 && c == -1 && d == 65 && e == -1 && f == 1 && g == 0,
              "switch rg_tail: 65 applied, 129 / -1 refused (value kept), 0 applied",
              fmt("%d %d %d %d %d %d %d", a, b, c, d, e, f, g));
    }
    for (const Shape& s : shapes) {
        std::printf("%s\n", s.name);
        void* inst = nullptr;
        const int irc = r4d_rad_router_gemm_topk_scatter_init(nullptr, 0, 0, 1, &inst);
        check(irc == RAD_OK && inst, "init takes an instance");
        if (!inst) continue;
        std::vector<uint16_t> wr((size_t)(s.ne * s.K)), wg((size_t)s.K), x;
        for (auto& v : wr) v = bf((float)((uni() * 2.0 - 1.0) * 0.08));
        for (auto& v : wg) v = bf((float)((uni() * 2.0 - 1.0) * 0.08));
        std::vector<int64_t> Ms = s.ne == 512 ? std::vector<int64_t>{ 17, 33, 64, 96, 128 } : std::vector<int64_t>{ 17, 40, 128 };
        if (quick) Ms = { 33, 128 };
        for (const int64_t M : Ms) {
            rows(x, M, s.K, s.ne, wr);
            for (const int mt : { 1, 2 }) {
                Out ref, got, again;
                const int r0 = run(s, M, x, wr, wg, inst, 0, mt, s.norm, ref);
                const int r1 = run(s, M, x, wr, wg, inst, 1, mt, s.norm, got);
                const int r2 = run(s, M, x, wr, wg, inst, 1, mt, s.norm, again);
                const std::string d1 = cmp(ref, got), d2 = cmp(got, again);
                check(r0 == RAD_OK && r1 == RAD_OK && r2 == RAD_OK && d1.empty() && d2.empty(),
                      fmt("M %lld rg_mt %d: RGT = today's form, every output byte with guard tails; again = the first", (long long)M, mt),
                      fmt("rc %d %d %d; vs today: %s; again: %s", r0, r1, r2, d1.c_str(), d2.c_str()));
            }
        }
        /* negative controls: the compare sees a changed weight rule and a changed activation */
        {
            const int64_t M = 33;
            rows(x, M, s.K, s.ne, wr);
            Out a, b, c;
            run(s, M, x, wr, wg, inst, 1, 1, s.norm, a);
            run(s, M, x, wr, wg, inst, 1, 1, 1 - s.norm, b);
            std::vector<uint16_t> x2 = x;
            x2[(size_t)(20 * s.K + 1)] ^= 0x0100;
            run(s, M, x2, wr, wg, inst, 1, 1, s.norm, c);
            const std::string dn = cmp(a, b), dx = cmp(a, c);
            check(dn.find("expert_w") != std::string::npos, "negative control: norm flipped -> expert_w differs", dn.empty() ? "no difference seen" : dn);
            check(dx.find("logits") != std::string::npos, "negative control: one activation element changed -> logits differ", dx.empty() ? "no difference seen" : dx);
        }
        /* the floor: M <= 16 takes today's form under rg_tail 1 (one tile) */
        {
            const int64_t M = 16;
            rows(x, M, s.K, s.ne, wr);
            Out a, b;
            const int ra = run(s, M, x, wr, wg, inst, 0, 1, s.norm, a), rb = run(s, M, x, wr, wg, inst, 1, 1, s.norm, b);
            check(ra == RAD_OK && rb == RAD_OK && cmp(a, b).empty(), "M 16: rg_tail 1 = today's form (one row tile)", cmp(a, b));
        }
        r4d_rad_router_gemm_topk_scatter_fini(inst);
    }
    /* lever RGU (Qwen loop round 36, QUEUE Q65): the GEMM's 20-step load batch (rg_ku 20) and the tile cap a block (rg_tpb 4 /
     * 2 / 1), alone and together, under today's tail and RGT, byte for byte against today's form -- at K 1280 (split 4: one
     * 20-step batch an item; the gate's split 20: 4 single steps) and K 2560 (two batches; the gate's 8 steps = today's batch);
     * then the switch's form bits (no op param: the switch decides) = the op params' bytes, and the switch's range. */
    {
        const Shape q[] = { { "Qwen tp4 rank at K 1280 (RGU)", 512, 1280, 10, 128, 128, true, 1, "7" },
                            { "Qwen tp4 rank at K 2560 (RGU)", 512, 2560, 10, 128, 128, true, 1, "7" } };
        for (const Shape& s : q) {
            std::printf("%s\n", s.name);
            void* inst = nullptr;
            if (r4d_rad_router_gemm_topk_scatter_init(nullptr, 0, 0, 1, &inst) != RAD_OK || !inst) { check(false, "init"); continue; }
            std::vector<uint16_t> wr((size_t)(s.ne * s.K)), wg((size_t)s.K), x;
            for (auto& v : wr) v = bf((float)((uni() * 2.0 - 1.0) * 0.08));
            for (auto& v : wg) v = bf((float)((uni() * 2.0 - 1.0) * 0.08));
            std::vector<int64_t> Ms = s.K == 1280 ? std::vector<int64_t>{ 5, 33, 128 } : std::vector<int64_t>{ 5, 64 };
            if (quick) Ms = { 33 };
            for (const int64_t M : Ms) {
                rows(x, M, s.K, s.ne, wr);
                for (const int tail : { 0, 1 }) {
                    Out ref;
                    const int r0 = run(s, M, x, wr, wg, inst, tail, 1, s.norm, ref, 8, 0);
                    struct A { int ku, tpb; } arms[] = { { 20, 0 }, { 40, 0 }, { 8, 4 }, { 8, 2 }, { 8, 1 }, { 20, 2 }, { 20, 1 },
                                                         { 40, 2 }, { 40, 1 } };
                    for (const A& a : arms) {
                        Out got, again;
                        const int r1 = run(s, M, x, wr, wg, inst, tail, 1, s.norm, got, a.ku, a.tpb);
                        const int r2 = run(s, M, x, wr, wg, inst, tail, 1, s.norm, again, a.ku, a.tpb);
                        const std::string d1 = cmp(ref, got), d2 = cmp(got, again);
                        check(r0 == RAD_OK && r1 == RAD_OK && r2 == RAD_OK && d1.empty() && d2.empty(),
                              fmt("M %lld rg_tail %d: rg_ku %d rg_tpb %d = today's form, every output byte; again = the first",
                                  (long long)M, tail, a.ku, a.tpb),
                              fmt("rc %d %d %d; vs today: %s; again: %s", r0, r1, r2, d1.c_str(), d2.c_str()));
                    }
                }
            }
            /* the switch's form fields: 512 + 2048 (ku 40, cap 2) with no op param = the op params' bytes; the switch back to 0 */
            {
                const int64_t M = 33;
                rows(x, M, s.K, s.ne, wr);
                Out a, b, c;
                run(s, M, x, wr, wg, inst, 0, 1, s.norm, a, 40, 2);
                const int sw = r4d_rg_tail_set(512 + 2048);
                run(s, M, x, wr, wg, inst, 0, 1, s.norm, b);
                (void)r4d_rg_tail_set(0);
                run(s, M, x, wr, wg, inst, 0, 1, s.norm, c);
                check(sw == 1 && cmp(a, b).empty() && cmp(a, c).empty(), "switch rg_tail 2560 (RGU: KU 40, cap 2) = the op params' bytes = form 0",
                      fmt("set %d; %s / %s", sw, cmp(a, b).c_str(), cmp(a, c).c_str()));
            }
            /* negative control at the wide form: one activation element changed -> logits differ */
            {
                const int64_t M = 33;
                rows(x, M, s.K, s.ne, wr);
                Out a, c;
                run(s, M, x, wr, wg, inst, 0, 1, s.norm, a, 20, 1);
                std::vector<uint16_t> x2 = x;
                x2[(size_t)(20 * s.K + 1)] ^= 0x0100;
                run(s, M, x2, wr, wg, inst, 0, 1, s.norm, c, 20, 1);
                const std::string dx = cmp(a, c);
                check(dx.find("logits") != std::string::npos, "negative control (rg_ku 20 rg_tpb 1): one activation changed -> logits differ",
                      dx.empty() ? "no difference seen" : dx);
            }
            r4d_rad_router_gemm_topk_scatter_fini(inst);
        }
        const int a = r4d_rg_tail_set(64 + 256), b = r4d_rg_tail_get(), c = r4d_rg_tail_set(129 + 256), d = r4d_rg_tail_get(),
                  e = r4d_rg_tail_set(128 + 512 + 3072), f = r4d_rg_tail_set(64 + 768), g = r4d_rg_tail_get(),
                  i = r4d_rg_tail_set(4096), h = r4d_rg_tail_set(0);
        check(a == 1 && b == 320 && c == -1 && d == 320 && e == 1 && f == -1 && g == 128 + 512 + 3072 && i == -1 && h == 1,
              "switch rg_tail with RGU's form: 320 applied, a row field 129 refused, 3712 applied, KU code 3 refused, 4096 refused, 0 applied",
              fmt("%d %d %d %d %d %d %d %d %d", a, b, c, d, e, f, g, i, h));
    }
    std::printf("%d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
