/* tests/r4d_emu: qfn-on-ds lever EPD (Qwen loop round 41, QUEUE Q78) on the host SIMT emulator -- the GDN decode update with
 * the convolution folded in (gdn_conv_recurrent_update / _had, anchor state, fp32) with its per-token epilogue (row norm, bf16
 * output, the out projection's codes + scales) deferred past the token loop (r4d_gdn_epd(1)) against today's form
 * (r4d_gdn_epd(0)), byte for byte -- and lever QKS (round 41, QUEUE Q79; r4d_gdn_epd(2): EPD + each token's normalised q / k staged once):
 *   o (bf16), the codes (E4M3 = NORM 1, int8 = NORM 2, had_quant_act_i8g's rotated int8 = NORM 3) and their scales, the whole
 *   recurrent state pool (anchor and factor slots) and the conv state, over TWO steps (the second replays the factors the
 *   first wrote);
 *   Flash-Next's tp4 rank shape (12 value heads, 4 key heads: three readers a key head, so the last-arriver rewrite runs),
 *   the engine's state_index layout (columns 0-2 the sequence's anchor slot, 3-5 its factor slot = slot + max_seqs), uneven
 *   windows 5 / 4 / 8 / 1 / 2 with accepted prefixes 3 / 1 / 4 / 2 / 1, a padded last sequence (every slot -1), silu gate.
 * DISPATCH: r4d_gdn_epd_launches() must grow by one a call under EPD and not at all without it.
 * NEGATIVE CONTROL: the same compare finds one changed gate element (token 0, head 0, row 5) in the EPD run.
 *
 *   r4d_emu_gdnepd */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_gdn_conv_recurrent_update(const RadArgs*, RadStream);
extern "C" int r4d_rad_gdn_conv_recurrent_update_had(const RadArgs*, RadStream);
extern "C" void r4d_gdn_epd(int);
extern "C" long long r4d_gdn_epd_launches();

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
uint64_t g_rs = 0x51ED270B9E3779B9ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
float uni(float s) { return (float)((double)(rnd() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0) * s; }
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
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

constexpr int64_t K = 128, V = 128, H = 12, Hg = 4, W = 4, MQ = 8, SL = W - 2 + MQ;
constexpr int64_t N = 5, COLS = 6, SSLOTS = 2 * N, CSLOTS = 6;
constexpr int64_t CONV = 2 * Hg * K + H * V, INW = CONV + H * V;
const int32_t kLens[N] = { 5, 4, 8, 1, 2 }, kNacc[N] = { 3, 1, 4, 2, 1 }, kCidx[N] = { 3, 0, 1, 4, -1 };

/* the shared inputs (weights, the two steps' projection rows and a / b), one state set a run */
struct Shared {
    std::vector<uint16_t> w, x[2], ab[2];
    std::vector<float> alog, dtb, nw;
    std::vector<int32_t> cu, sidx, nacc, cidx;
    int64_t Td = 0;
};
struct Run {
    std::vector<float> st;
    std::vector<uint16_t> cs, o;
    std::vector<uint8_t> oq;
    std::vector<float> os;
};
void make(Shared& s) {
    s.cu.assign(1, 0);
    for (int64_t n = 0; n < N; ++n) { s.Td += kLens[n]; s.cu.push_back((int32_t)s.Td); }
    s.w.resize((size_t)(CONV * W));
    for (auto& e : s.w) e = bf(uni(0.5f));
    for (int st = 0; st < 2; ++st) {
        s.x[st].resize((size_t)(s.Td * INW));
        for (auto& e : s.x[st]) e = bf(uni(1.0f));
        s.ab[st].resize((size_t)(s.Td * 2 * H));
        for (auto& e : s.ab[st]) e = bf(uni(1.0f));
    }
    s.alog.resize(H); s.dtb.resize(H); s.nw.resize(V);
    for (int64_t h = 0; h < H; ++h) { s.alog[(size_t)h] = -2.0f + 0.25f * (float)h; s.dtb[(size_t)h] = -1.0f + 0.5f * (float)(h % 5); }
    for (auto& e : s.nw) e = 1.0f + uni(0.25f);
    s.sidx.resize((size_t)(N * COLS));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < COLS; ++c)   /* KVManager::state_index_row: column j names copy j * 2 / 6 at slot + copy * max_seqs */
            s.sidx[(size_t)(n * COLS + c)] = kCidx[n] < 0 ? -1 : (int32_t)(n + (c * 2 / COLS) * N);
    s.nacc.assign(kNacc, kNacc + N);
    s.cidx.assign(kCidx, kCidx + N);
}
void init(Run& r, const Shared& s, int form) {
    uint64_t keep = g_rs;
    g_rs = 0xC0FFEE1234567ull;   /* every run starts from the same state */
    r.st.resize((size_t)(SSLOTS * H * V * K));
    for (auto& e : r.st) e = uni(0.1f);
    for (int64_t n = 0; n < N; ++n)   /* the factor slots start zeroed, as after a prefill (marker absent) */
        std::memset(&r.st[(size_t)((n + N) * H * V * K)], 0, (size_t)(H * V * K) * 4);
    r.cs.resize((size_t)(CSLOTS * CONV * SL));
    for (auto& e : r.cs) e = bf(uni(1.0f));
    r.o.assign((size_t)(s.Td * H * V), 0);
    r.oq.assign((size_t)(s.Td * H * V), 0);
    r.os.assign((size_t)(s.Td * H), 0.0f);
    (void)form;
    g_rs = keep;
}
/* form 1 E4M3 codes (NORM 1), 2 int8 (NORM 2), 3 had_quant_act_i8g's rotated int8 (NORM 3); zpoke >= 0: that gate element + 1 ulp-ish */
int step(Run& r, const Shared& s, int form, int st, long long zpoke) {
    std::vector<uint16_t> x = s.x[st];
    if (zpoke >= 0) x[(size_t)(CONV + zpoke)] ^= 0x0100u;
    RadTensor tx = tens(x.data(), RAD_BF16, { s.Td, CONV });
    tx.stride[0] = INW;
    RadTensor tz = tens(x.data() + CONV, RAD_BF16, { s.Td, H * V });
    tz.stride[0] = INW;
    RadTensor ta = tens(s.ab[st].data(), RAD_BF16, { s.Td, H });
    ta.stride[0] = 2 * H;
    RadTensor tb = tens(s.ab[st].data() + H, RAD_BF16, { s.Td, H });
    tb.stride[0] = 2 * H;
    RadTensor none{};
    std::vector<RadTensor> t = {
        tx, tens(s.w.data(), RAD_BF16, { CONV, W }), none, tens(r.cs.data(), RAD_BF16, { CSLOTS, CONV, SL }),
        tens(s.cidx.data(), RAD_I32, { N }), ta, tb, tens(s.alog.data(), RAD_F32, { H }), tens(s.dtb.data(), RAD_F32, { H }),
        tens(r.st.data(), RAD_F32, { SSLOTS, H, V, K }), tens(s.sidx.data(), RAD_I32, { N, COLS }),
        tens(s.nacc.data(), RAD_I32, { N }), tens(s.cu.data(), RAD_I32, { N + 1 }), tz, tens(s.nw.data(), RAD_F32, { V }),
        tens(r.o.data(), RAD_BF16, { s.Td, H, V }), tens(r.oq.data(), form == 1 ? RAD_F8E4M3 : RAD_I8, { s.Td, H * V }),
        tens(r.os.data(), RAD_F32, { s.Td, H }) };
    std::vector<RadParam> p = { pint("q_len", MQ), pint("head_k", K), pint("head_v", V), pint("n_head_v", H),
                                pf64("scale", 0.088388), pf64("eps", 1e-6), pstr("act", "silu"), pstr("state_form", "anchor"),
                                pint("conv_width", W) };
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
    return form == 3 ? r4d_rad_gdn_conv_recurrent_update_had(&a, nullptr) : r4d_rad_gdn_conv_recurrent_update(&a, nullptr);
}
std::string cmp(const Run& a, const Run& b, int64_t real_tok) {
    std::string s;
    auto add = [&](const char* w, long long d) { if (d) s += fmt("%s%s %lld", s.empty() ? "" : ", ", w, d); };
    add("o", ndiff(a.o, b.o));
    /* the padded last sequence's rows are written by neither form */
    std::vector<uint8_t> qa(a.oq.begin(), a.oq.begin() + real_tok * H * V), qb(b.oq.begin(), b.oq.begin() + real_tok * H * V);
    add("codes", ndiff(qa, qb));
    add("scales", ndiff(a.os, b.os));
    add("state", ndiff(a.st, b.st));
    add("conv_state", ndiff(a.cs, b.cs));
    return s;
}

}  // namespace

int main() {
    std::printf("r4d_emu_gdnepd: lever EPD (the GDN decode update's epilogue after its token loop) vs today's form\n");
    Shared s;
    make(s);
    const int64_t real = s.cu[(size_t)N - 1];
    for (int form : { 3, 2, 1 }) {
        const char* fn = form == 3 ? "had (NORM 3, rotated int8)" : form == 2 ? "int8 (NORM 2)" : "E4M3 (NORM 1)";
        Run a, b, c, q;
        init(a, s, form); init(b, s, form); init(c, s, form); init(q, s, form);
        std::string why;
        long long moved_o = 0;
        for (int st = 0; st < 2 && why.empty(); ++st) {
            r4d_gdn_epd(0);
            const long long l0 = r4d_gdn_epd_launches();
            const int ra = step(a, s, form, st, -1);
            const long long l1 = r4d_gdn_epd_launches();
            r4d_gdn_epd(1);
            const int rb = step(b, s, form, st, -1);
            const long long l2 = r4d_gdn_epd_launches();
            /* NEGATIVE CONTROL run: EPD with one gate element changed at step 0 */
            const int rc = step(c, s, form, st, st == 0 ? 5 : -1);
            /* lever QKS (round 41, QUEUE Q79): EPD level 2 */
            r4d_gdn_epd(2);
            const long long l3 = r4d_gdn_epd_launches();
            const int rq = step(q, s, form, st, -1);
            const long long l4 = r4d_gdn_epd_launches();
            r4d_gdn_epd(-1);
            if (ra || rb || rc || rq) { why = fmt("step %d rc %d / %d / %d / %d", st, ra, rb, rc, rq); break; }
            check(l1 == l0 && l2 == l1 + 1 && l4 == l3 + 1, fmt("%s step %d: dispatch (EPD launches +%lld unset, +%lld EPD, +%lld QKS)", fn, st,
                                                               l1 - l0, l2 - l1, l4 - l3));
            const std::string d = cmp(a, b, real);
            if (!d.empty()) { why = fmt("step %d: %s", st, d.c_str()); break; }
            const std::string dq = cmp(a, q, real);
            if (!dq.empty()) { why = fmt("step %d (QKS): %s", st, dq.c_str()); break; }
            for (auto v : a.o) moved_o += v != 0;
            if (st == 0) {
                const std::string dc = cmp(a, c, real);
                check(!dc.empty(), fmt("%s NEGATIVE CONTROL: one changed gate element is seen (%s)", fn, dc.c_str()));
            }
        }
        check(why.empty() && moved_o > 0, fmt("%s: EPD and EPD + QKS = today's form, every byte, two steps (o non-zero %lld)", fn, moved_o), why);
    }
    std::printf("r4d_emu_gdnepd: %d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
