/* tests/r4d_emu: qfn-on-ds lever GNQ8 (Qwen loop round 7, 21.11.2026) on the host SIMT emulator -- gdn_gated_rmsnorm_i8g
 * (the GDN prefill gated norm writing quant_act_i8g's codes of its output in the same pass) byte for byte against the pair
 * it replaces, gdn_gated_rmsnorm then quant_act_i8g over its output:
 *   o (bf16), q (int8 codes), scale (f32 per (token, 128 columns)), each with a guard tail;
 *   Qwen's tp4 rank shape (12 v heads x 128 = 1536 a token) and a 3-head one; tokens 1 / 2 / 17 / 64 / 129; silu and
 *   sigmoid gates; the gate flat [tokens, n] and as the z columns of a wider [q|k|v|z] row (the engine's bcol_at view);
 *   random rows, a zero (token, head) row (scale 1, codes 0), a row of large values (the clamp), a NaN element;
 *   each launched twice (the second launch's bytes = the first's).
 * NEGATIVE CONTROLS: the same compare finds one changed x element, and a pitched code plane (q at a wider row pitch than
 * n: the fused op writes inside each row's n columns only, guard bytes between rows untouched).
 *
 *   r4d_emu_gnq8 */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_gdn_gated_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_gdn_gated_rmsnorm_i8g(const RadArgs*, RadStream);
extern "C" int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);

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
uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
double uni() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* a tensor of `shape` at row pitch `pitch` elements (0: dense) */
RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape, int64_t pitch = 0) {
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
    if (pitch && t.rank == 2) t.stride[0] = pitch;
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }

constexpr int64_t CH = 128, TAIL = 64;

struct Out {
    std::vector<uint16_t> o;
    std::vector<int8_t> q;
    std::vector<float> s;
};
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}
std::string cmp(const Out& a, const Out& b) {
    std::string s;
    const long long d0 = ndiff(a.o, b.o), d1 = ndiff(a.q, b.q), d2 = ndiff(a.s, b.s);
    if (d0) s += fmt("o %lld", d0);
    if (d1) s += fmt("%sq %lld", s.empty() ? "" : ", ", d1);
    if (d2) s += fmt("%sscale %lld", s.empty() ? "" : ", ", d2);
    return s;
}

struct Case {
    int64_t tok, heads;
    int act;          /* 0 silu, 1 sigmoid */
    bool zwide;       /* z = the z columns of a [q|k|v|z] row */
    int64_t qpitch;   /* q's row pitch (0: n) */
};

/* the inputs: x [tok, n], the z plane (flat [tok, n] or [tok, zw] with z at column zoff), w [128] */
struct In {
    std::vector<uint16_t> x, zp;
    std::vector<float> w;
    int64_t zw = 0, zoff = 0;
};
void make(const Case& c, In& in) {
    const int64_t n = c.heads * CH;
    in.x.resize((size_t)(c.tok * n));
    for (auto& v : in.x) v = bf((float)((uni() * 2.0 - 1.0) * 3.0));
    if (c.tok > 1) std::memset(&in.x[(size_t)(1 * n + (c.heads - 1) * CH)], 0, CH * 2);   /* a zero (token, head) row */
    if (c.tok > 2) for (int64_t i = 0; i < CH; ++i) in.x[(size_t)(2 * n + i)] = bf((float)(4096.0 * (uni() - 0.5)));
    if (c.tok > 3) in.x[(size_t)(3 * n + 7)] = 0x7fc1;                                    /* a NaN element */
    in.zw = c.zwide ? 3 * n + n : n;
    in.zoff = c.zwide ? 3 * n : 0;
    in.zp.resize((size_t)(c.tok * in.zw));
    for (auto& v : in.zp) v = bf((float)((uni() * 2.0 - 1.0) * 4.0));
    in.w.resize(CH);
    for (auto& v : in.w) v = (float)(0.5 + uni());
}
RadTensor ztens(const Case& c, const In& in) {
    const int64_t n = c.heads * CH;
    if (!c.zwide) return tens(in.zp.data(), RAD_BF16, { c.tok, n });
    /* the engine's bcol_at: [tok, n] columns at zoff of a zw-wide row */
    RadTensor t = tens(in.zp.data() + in.zoff, RAD_BF16, { c.tok, n });
    t.stride[0] = in.zw;
    return t;
}
std::vector<RadParam> nparams(const Case& c, bool fused) {
    std::vector<RadParam> p = { pint("M", fused ? c.tok : c.tok * c.heads), pint("channels", CH), pf64("eps", 1e-6),
                                pstr("act", c.act ? "sigmoid" : "silu") };
    if (fused) { p.push_back(pint("n", c.heads * CH)); p.push_back(pint("group", 128)); }
    return p;
}
int call(int (*fn)(const RadArgs*, RadStream), std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    return fn(&a, nullptr);
}
/* the pair: gdn_gated_rmsnorm into o, then quant_act_i8g over o */
int run_pair(const Case& c, const In& in, Out& r) {
    const int64_t n = c.heads * CH, qp = c.qpitch ? c.qpitch : n;
    r.o.assign((size_t)(c.tok * n + TAIL), 0x5a5a);
    r.q.assign((size_t)(c.tok * qp + TAIL), (int8_t)0x5a);
    r.s.assign((size_t)(c.tok * c.heads + TAIL), -77.0f);
    std::vector<RadTensor> t = { tens(in.x.data(), RAD_BF16, { c.tok, n }), ztens(c, in), tens(in.w.data(), RAD_F32, { CH }),
                                 tens(r.o.data(), RAD_BF16, { c.tok, n }) };
    std::vector<RadParam> p = nparams(c, false);
    int rc = call(r4d_rad_gdn_gated_rmsnorm, t, p);
    if (rc) return rc;
    std::vector<RadTensor> u = { tens(r.o.data(), RAD_BF16, { c.tok, n }), tens(r.q.data(), RAD_I8, { c.tok, n }, qp),
                                 tens(r.s.data(), RAD_F32, { c.tok, c.heads }) };
    std::vector<RadParam> pq = { pint("M", c.tok), pint("n", n), pint("group", 128), pstr("dtype", "bf16") };
    return call(r4d_rad_quant_act_i8g, u, pq);
}
int run_fused(const Case& c, const In& in, Out& r) {
    const int64_t n = c.heads * CH, qp = c.qpitch ? c.qpitch : n;
    r.o.assign((size_t)(c.tok * n + TAIL), 0x5a5a);
    r.q.assign((size_t)(c.tok * qp + TAIL), (int8_t)0x5a);
    r.s.assign((size_t)(c.tok * c.heads + TAIL), -77.0f);
    std::vector<RadTensor> t = { tens(in.x.data(), RAD_BF16, { c.tok, n }), ztens(c, in), tens(in.w.data(), RAD_F32, { CH }),
                                 tens(r.o.data(), RAD_BF16, { c.tok, n }), tens(r.q.data(), RAD_I8, { c.tok, n }, qp),
                                 tens(r.s.data(), RAD_F32, { c.tok, c.heads }) };
    std::vector<RadParam> p = nparams(c, true);
    return call(r4d_rad_gdn_gated_rmsnorm_i8g, t, p);
}

}  // namespace

int main() {
    std::printf("r4d_emu_gnq8: gdn_gated_rmsnorm_i8g vs gdn_gated_rmsnorm + quant_act_i8g\n");
    const int64_t toks[] = { 1, 2, 17, 64, 129 };
    for (int64_t heads : { (int64_t)12, (int64_t)3 })
        for (int64_t tok : toks)
            for (int act : { 0, 1 })
                for (bool zw : { false, true }) {
                    const Case c{ tok, heads, act, zw, 0 };
                    In in;
                    make(c, in);
                    Out a, b, b2;
                    const int ra = run_pair(c, in, a), rb = run_fused(c, in, b), rb2 = run_fused(c, in, b2);
                    const std::string what = fmt("heads %lld tok %lld %s z %s", (long long)heads, (long long)tok,
                                                 act ? "sigmoid" : "silu", zw ? "qkvz view" : "flat");
                    if (ra || rb || rb2) { check(false, what, fmt("rc pair %d fused %d / %d", ra, rb, rb2)); continue; }
                    const std::string d = cmp(a, b), d2 = cmp(b, b2);
                    check(d.empty() && d2.empty(), what, d.empty() ? "relaunch: " + d2 : d);
                }
    /* a pitched code plane: the guard bytes between rows stay untouched, and the rows equal the pair's */
    {
        const Case c{ 17, 12, 0, true, 12 * CH + 32 };
        In in;
        make(c, in);
        Out a, b;
        const int ra = run_pair(c, in, a), rb = run_fused(c, in, b);
        check(!ra && !rb && cmp(a, b).empty(), "pitched q (row pitch n + 32)", fmt("rc %d / %d: %s", ra, rb, cmp(a, b).c_str()));
    }
    /* NEGATIVE CONTROL: one x element changed -> the compare sees it */
    {
        const Case c{ 17, 12, 1, false, 0 };
        In in;
        make(c, in);
        Out a, b;
        run_pair(c, in, a);
        in.x[(size_t)(5 * 12 * CH + 300)] ^= 0x0100;
        run_fused(c, in, b);
        check(!cmp(a, b).empty(), "negative control: one x element changed is found", "compare saw no change");
    }
    /* the shape refusals: rows not a whole number of tokens' heads; a scale plane of the wrong width */
    {
        const Case c{ 4, 3, 0, false, 0 };
        In in;
        make(c, in);
        Out r;
        r.o.assign((size_t)(4 * 3 * CH + TAIL), 0);
        r.q.assign((size_t)(4 * 3 * CH + TAIL), 0);
        r.s.assign((size_t)(4 * 3 + TAIL), 0);
        std::vector<RadTensor> t = { tens(in.x.data(), RAD_BF16, { 4, 3 * CH }), ztens(c, in), tens(in.w.data(), RAD_F32, { CH }),
                                     tens(r.o.data(), RAD_BF16, { 4, 3 * CH }), tens(r.q.data(), RAD_I8, { 4, 3 * CH }),
                                     tens(r.s.data(), RAD_F32, { 4, 2 }) };
        std::vector<RadParam> p = nparams(c, true);
        check(call(r4d_rad_gdn_gated_rmsnorm_i8g, t, p) != 0, "refused: scale plane 2 wide for 3 heads");
    }
    std::printf("r4d_emu_gnq8: %d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
