// arw.hip (12.11.2026, ds-main-port arw builder) -- libr4d's four-rank exact two-shot all-reduce: the PLAIN wire (the row DS
// resolves at prefill, ar_twoshot_4rank_exact, no `overlap`) against the CODED wire (the same entry at `overlap` 1, row
// ar_twoshot_4rank_exact_lane: r4d_ar_entry.hip 1061-1068 -> r4d_ar_twoshot_4rank_exact_coded, ~12.2 bits a bf16 value), at
// DeepSeek-V4.1's shapes (T x 5120 bf16), on FOUR cards, one host thread a rank in one process (as r4d_selftest's four-rank
// cases run them). The image's own libr4d.so is dlopen'ed; both wires go through the row's real launch entry with the row's
// own tunable defaults as launch parameters (what the core passes when nothing is tuned / pinned).
//
// Which wire ran is SEEN, not inferred: this executable exports r4d_ar_twoshot_4rank_exact, r4d_ar_twoshot_4rank_exact_coded
// and r4d_ar_twoshot_coded_ok (linked -rdynamic); libr4d.so calls them from r4d_ar_entry.hip through its PLT (JUMP_SLOT
// relocations), so the dynamic linker binds those calls here first; each wrapper counts the call a thread and forwards to
// libr4d's own definition (dlsym on the library handle). `--selftest` proves the binding with no device: it reads the
// library's GOT slots for the three names and checks they hold this executable's wrappers. The coded wrapper can also replace
// the entry's block count (the entry passes 4 x 1024): variants C8 / C16 / C24 = the same coded kernel at a wider grid
// (what a build could choose; libr4d: "the block count moves no byte").
//
// Per shape T and input pattern: every variant (P = plain, C = the entry's coded launch, C8 / C16 / C24) in place (DS's form)
// and out of place, every rank's output vs P's on that rank (BYTE-IDENTICAL or VIRHE and the reading stops) and vs the fp32
// ascending-rank fold rounded once to bf16 (libr4d's contract; a NOTE if it differs). Then timing on the timed patterns:
// warm-ups, `reps` isolated launches a variant (in place on a fresh copy, variants interleaved forward / backward a rep, a
// spin barrier before each), per rank one event pair a launch: "own" = the last arriver's time (min over ranks; the
// collective's own), "max" = the first arriver's (adds the barrier skew); median / p10 / p90 over reps; and back-to-back
// (`bb` launches out of place, no host sync between): the lockstep period (max over ranks).
// Patterns: bin = bf16 across binades (per 64-value group an exponent span 0..20 at a base 2^-40..2^40, so groups land on
// both sides of the coded form's 14-binade limit; zeros, subnormals, +inf on rank 0); act = activation-like (gaussian x a
// token scale 2^-5..2^1 x ~0.2 % outlier channels x16..x128, per rank independent); actz = act with rank 3 all zeros (an
// attention-zero rank's partial).
// Out: "ARW ..." lines (arwsum.py). Exit 0 = every comparison ran and P = every coded variant on every rank; 1 = VIRHE;
// 2 = usage; 3 = a HIP / setup failure.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "rad_abi.h"

#define ARW_CK(x)                                                                                    \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::printf("ARW FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::fflush(stdout);                                                                     \
            std::_Exit(3);                                                                           \
        }                                                                                            \
    } while (0)

/* ------------------------------------------------------------------ the three wrappers libr4d binds to */
typedef void (*PlainFn)(long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long,
                        long, long, long, long, long, long, long, long);
typedef void (*CodedFn)(long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long,
                        long, long, long, long, long, long, long);
typedef int (*OkFn)(long, long, long, long);
static PlainFn g_real_plain = nullptr;
static CodedFn g_real_coded = nullptr;
static OkFn g_real_ok = nullptr;

struct Probe {
    int plain_calls = 0, coded_calls = 0, ok_calls = 0, ok_last = -1;
    long plain_nb = -1, plain_nt = -1, plain_drain = -1, coded_nb_in = -1, coded_nb_used = -1, coded_nt = -1, coded_drain = -1;
};
static thread_local Probe tp;
static thread_local int tp_nb_override = 0;

extern "C" {
__attribute__((visibility("default"))) int r4d_ar_twoshot_coded_ok(long n_elem, long dtype, long world, long slot_stride16) {
    if (!g_real_ok) { std::printf("ARW FATAL coded_ok wrapper called before binding\n"); std::fflush(stdout); std::_Exit(3); }
    const int r = g_real_ok(n_elem, dtype, world, slot_stride16);
    ++tp.ok_calls;
    tp.ok_last = r;
    return r;
}
__attribute__((visibility("default"))) void r4d_ar_twoshot_4rank_exact(
    long peer_scratch0, long peer_scratch1, long peer_scratch2, long peer_flags0, long peer_flags1, long peer_flags2,
    long my_scratch, long my_flags, long seq_ctrs, long slot_stride16, long inp, long out, long n_elem, long dtype, long rank,
    long stream, long nblocks, long nthreads, long drain, long acq, long pub, long gate, long gate_ctr, long chain) {
    if (!g_real_plain) { std::printf("ARW FATAL plain wrapper called before binding\n"); std::fflush(stdout); std::_Exit(3); }
    ++tp.plain_calls;
    tp.plain_nb = nblocks;
    tp.plain_nt = nthreads;
    tp.plain_drain = drain;
    g_real_plain(peer_scratch0, peer_scratch1, peer_scratch2, peer_flags0, peer_flags1, peer_flags2, my_scratch, my_flags,
                 seq_ctrs, slot_stride16, inp, out, n_elem, dtype, rank, stream, nblocks, nthreads, drain, acq, pub, gate,
                 gate_ctr, chain);
}
__attribute__((visibility("default"))) void r4d_ar_twoshot_4rank_exact_coded(
    long peer_scratch0, long peer_scratch1, long peer_scratch2, long peer_flags0, long peer_flags1, long peer_flags2,
    long my_scratch, long my_flags, long seq_ctrs, long slot_stride16, long inp, long out, long n_elem, long rank, long stream,
    long nblocks, long nthreads, long drain, long acq, long pub, long gate, long gate_ctr, long chain) {
    if (!g_real_coded) { std::printf("ARW FATAL coded wrapper called before binding\n"); std::fflush(stdout); std::_Exit(3); }
    ++tp.coded_calls;
    tp.coded_nb_in = nblocks;
    if (tp_nb_override > 0) nblocks = tp_nb_override;
    tp.coded_nb_used = nblocks;
    tp.coded_nt = nthreads;
    tp.coded_drain = drain;
    g_real_coded(peer_scratch0, peer_scratch1, peer_scratch2, peer_flags0, peer_flags1, peer_flags2, my_scratch, my_flags,
                 seq_ctrs, slot_stride16, inp, out, n_elem, rank, stream, nblocks, nthreads, drain, acq, pub, gate, gate_ctr,
                 chain);
}
}

/* ------------------------------------------------------------------ host pieces */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static inline uint64_t h4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { return mix64(mix64(mix64(mix64(a) ^ b) ^ c) ^ d); }
/* r4d_selftest's h_bf16 / h_to_bf16 (round to nearest even; the inputs hold no NaN) */
static inline float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

enum Pat { PBIN = 0, PACT = 1, PACTZ = 2 };
static const char* pat_name(int p) { return p == PBIN ? "bin" : p == PACT ? "act" : "actz"; }
static int pat_of(const std::string& s) { return s == "bin" ? PBIN : s == "act" ? PACT : s == "actz" ? PACTZ : -1; }

static uint16_t gen_val(int pat, uint64_t sd, int r, long i, long N) {
    if (pat == PBIN) {
        const long g = i / 64;
        const int j = (int)(i % 64);
        const uint64_t hg = h4(sd, 0xB1, (uint64_t)g, (uint64_t)r);
        const int span = (int)(hg % 21), base = (int)((hg >> 8) % 81) - 40, kind = (int)((hg >> 16) % 256);
        const int pos = (int)((hg >> 24) % 64);
        if (kind < 4) return 0;                                                   /* an all-zero group on this rank */
        const uint64_t he = h4(sd, 0xB2, (uint64_t)i, (uint64_t)r);
        if (kind < 8 && j == pos) return (uint16_t)(1 + (he % 127));            /* a subnormal: the group goes raw */
        if (kind == 8 && r == 0 && j == pos) return (uint16_t)0x7f80;            /* +inf on rank 0: raw, sum inf */
        if ((he & 15) == 0) return 0;                                            /* an exact zero */
        const int e = 127 + base + (int)((he >> 4) % (uint64_t)(span + 1));
        const unsigned m = (unsigned)((he >> 12) & 0x7f), s = (unsigned)((he >> 20) & 1);
        return (uint16_t)((s << 15) | ((unsigned)e << 7) | m);
    }
    if (pat == PACTZ && r == 3) return 0;
    const long row = i / N, col = i % N;
    const uint64_t hr = h4(sd, 0xA1, (uint64_t)row, 0), hc = h4(sd, 0xA2, (uint64_t)col, 0);
    const float rs = std::ldexp(1.0f, (int)(hr % 7) - 5);
    const float cs = (hc % 512 == 0) ? (float)(16 << (int)((hc >> 9) % 4)) : 1.0f;
    const uint64_t he = h4(sd, 0xA3, (uint64_t)i, (uint64_t)r), he2 = mix64(he);
    const double u1 = (double)((he >> 11) + 1) * (1.0 / 9007199254740993.0);
    const double u2 = (double)(he2 >> 11) * (1.0 / 9007199254740992.0);
    const double gs = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return f2bf((float)(gs * rs * cs));
}

/* r4d_arc_enc's group rule restated (r4d_ar_twoshot_Nrank_exact.hip 247-279) -- for the raw-group share only; the
 * comparisons never use it: 64 values a group; raw if any subnormal / inf / NaN or an exponent range past 14. */
static bool arc_raw(const uint16_t* v) {
    unsigned emin = 255u, emax = 0u;
    bool bad = false;
    for (int j = 0; j < 64; ++j) {
        const unsigned u = v[j], e = (u >> 7) & 0xffu, m = u & 0x7fu;
        if (e) { emin = e < emin ? e : emin; emax = e > emax ? e : emax; }
        if ((e == 0u && m != 0u) || e == 255u) bad = true;
    }
    const unsigned base = emin == 255u ? 1u : emin;
    return bad || emax > base + 14u;
}
static void raw_share(const uint16_t* v, long n, long long& raw, long long& tot) {
    for (long g = 0; g + 64 <= n; g += 64) { raw += arc_raw(v + g) ? 1 : 0; ++tot; }
}

struct Diff { long long n = 0; long first = -1; };
static Diff cmp16(const uint16_t* a, const uint16_t* b, long n) {
    Diff d;
    if (!std::memcmp(a, b, (size_t)n * 2)) return d;
    for (long i = 0; i < n; ++i)
        if (a[i] != b[i]) { if (d.first < 0) d.first = i; ++d.n; }
    return d;
}

static inline void cpu_relax() {
#if defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__)
    __builtin_ia32_pause();
#endif
}
struct SpinBar {
    std::atomic<int> cnt{0}, gen{0};
    int n = 4;
    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            cnt.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        long spins = 0;
        while (gen.load(std::memory_order_acquire) == g) {
            if (++spins < 4000000) cpu_relax();
            else std::this_thread::yield();
        }
    }
};

static inline RadParam pint(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
static inline RadParam pstr(const char* k, const char* v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}
static inline RadTensor tens1(void* data, long n) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = data;
    t.dtype = RAD_BF16;
    t.rank = 1;
    t.shape[0] = n;
    t.stride[0] = 1;
    return t;
}

/* ------------------------------------------------------------------ configuration and shared state */
struct Var { std::string name; bool coded; int nb; };
struct Cfg {
    std::string lib = "/opt/radiance/share/radiance/kernels/libr4d.so";
    std::vector<long> rows = { 1100, 1500, 2048, 4096, 8, 16, 48 };
    long N = 5120, maxrows = 4160;
    int reps = 100, warm = 5, bb = 20;
    uint64_t seed = 20261112;
    std::vector<int> nbs = { 8, 16, 24 };
    std::vector<int> pats = { PBIN, PACT, PACTZ }, tpats = { PACT, PBIN };
    bool selftest = false;
};
static Cfg C;
static std::vector<Var> V;
static void* g_lib = nullptr;
static const RadKernelInfo* g_rowP = nullptr;   /* ar_twoshot_4rank_exact: DS's prefill row (no overlap) */
static const RadKernelInfo* g_rowL = nullptr;   /* ar_twoshot_4rank_exact_lane: the same entry at overlap 1 */
static SpinBar g_bar;
static std::atomic<int> g_bad{0}, g_setup_bad{0};
static std::vector<uint16_t> g_hx[4], g_hplain[4], g_hcur[4], g_horacle;
static Probe g_probe[4];
static int g_rc[4];
static std::vector<double> g_t[8][4];
static double g_bb[8][4];
static long long g_pathbad[4], g_launchbad[4];
static int g_wgp[4];
static long long g_oracle_notes = 0;

static std::vector<RadParam> init_params(bool overlap, long long maxn) {
    std::vector<RadParam> p = { pint("world_size", 4), pint("numel", maxn), pstr("dtype", "bf16"), pint("exact", 1) };
    if (overlap) p.push_back(pint("overlap", 1));
    return p;
}
static std::vector<RadParam> launch_params(const RadKernelInfo* ki, bool overlap, long long maxn) {
    std::vector<RadParam> p = init_params(overlap, maxn);
    for (int i = 0; i < ki->n_tunables; ++i) p.push_back(pint(ki->tunables[i].key, (long long)ki->tunables[i].deflt));
    return p;
}

/* ------------------------------------------------------------------ the GOT check (no device) */
static int got_bound(void* h, const char* sym, const void* want, int* found) {
    struct link_map* lm = nullptr;
    *found = 0;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || !lm) return -1;
    const ElfW(Addr) base = lm->l_addr;
    ElfW(Addr) jmprel = 0, pltrelsz = 0, rela = 0, relasz = 0, symtab = 0, strtab = 0;
    for (const ElfW(Dyn)* d = lm->l_ld; d->d_tag != DT_NULL; ++d) {
        switch (d->d_tag) {
            case DT_JMPREL: jmprel = d->d_un.d_ptr; break;
            case DT_PLTRELSZ: pltrelsz = d->d_un.d_val; break;
            case DT_RELA: rela = d->d_un.d_ptr; break;
            case DT_RELASZ: relasz = d->d_un.d_val; break;
            case DT_SYMTAB: symtab = d->d_un.d_ptr; break;
            case DT_STRTAB: strtab = d->d_un.d_ptr; break;
            default: break;
        }
    }
    auto fix = [&](ElfW(Addr) p) { return (p && p < base) ? p + base : p; };   /* glibc may or may not have relocated */
    jmprel = fix(jmprel); rela = fix(rela); symtab = fix(symtab); strtab = fix(strtab);
    if (!symtab || !strtab) return -1;
    int bound = 0;
    auto scan = [&](ElfW(Addr) tab, ElfW(Addr) sz) {
        const ElfW(Rela)* R = (const ElfW(Rela)*)tab;
        for (size_t k = 0; k < sz / sizeof(ElfW(Rela)); ++k) {
            const unsigned t = (unsigned)ELF64_R_TYPE(R[k].r_info);
            if (t != R_X86_64_JUMP_SLOT && t != R_X86_64_GLOB_DAT) continue;
            const ElfW(Sym)* s = (const ElfW(Sym)*)symtab + ELF64_R_SYM(R[k].r_info);
            if (std::strcmp((const char*)strtab + s->st_name, sym) != 0) continue;
            ++*found;
            if (*(void* const*)(base + R[k].r_offset) == want) ++bound;
        }
    };
    if (jmprel) scan(jmprel, pltrelsz);
    if (rela) scan(rela, relasz);
    return bound;
}

static bool load_lib() {
    g_lib = dlopen(C.lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) { std::printf("ARW VIRHE dlopen %s: %s\n", C.lib.c_str(), dlerror()); return false; }
    auto abi = (uint32_t(*)(void))dlsym(g_lib, "rad_plugin_abi_version");
    auto kc = (int (*)(void))dlsym(g_lib, "rad_kernel_count");
    auto ka = (const RadKernelInfo* (*)(int))dlsym(g_lib, "rad_kernel_at");
    g_real_plain = (PlainFn)dlsym(g_lib, "r4d_ar_twoshot_4rank_exact");
    g_real_coded = (CodedFn)dlsym(g_lib, "r4d_ar_twoshot_4rank_exact_coded");
    g_real_ok = (OkFn)dlsym(g_lib, "r4d_ar_twoshot_coded_ok");
    if (!abi || !kc || !ka || !g_real_plain || !g_real_coded || !g_real_ok) {
        std::printf("ARW VIRHE symbols missing in %s (abi %p count %p at %p plain %p coded %p ok %p)\n", C.lib.c_str(),
                    (void*)abi, (void*)kc, (void*)ka, (void*)g_real_plain, (void*)g_real_coded, (void*)g_real_ok);
        return false;
    }
    if ((void*)g_real_plain == (void*)&r4d_ar_twoshot_4rank_exact || (void*)g_real_coded == (void*)&r4d_ar_twoshot_4rank_exact_coded ||
        (void*)g_real_ok == (void*)&r4d_ar_twoshot_coded_ok) {
        std::printf("ARW VIRHE dlsym on the library handle returned this executable's wrapper\n");
        return false;
    }
    if (abi() != RAD_ABI_VERSION) {
        std::printf("ARW VIRHE ABI: library %u, harness headers %u\n", abi(), (unsigned)RAD_ABI_VERSION);
        return false;
    }
    for (int k = 0; k < kc(); ++k) {
        const RadKernelInfo* ki = ka(k);
        if (!ki || !ki->name) continue;
        if (!std::strcmp(ki->name, "ar_twoshot_4rank_exact")) g_rowP = ki;
        if (!std::strcmp(ki->name, "ar_twoshot_4rank_exact_lane")) g_rowL = ki;
    }
    if (!g_rowP || !g_rowL || !g_rowP->init || !g_rowP->launch || !g_rowL->init || !g_rowL->launch) {
        std::printf("ARW VIRHE rows: ar_twoshot_4rank_exact %s, ar_twoshot_4rank_exact_lane %s\n", g_rowP ? "found" : "MISSING",
                    g_rowL ? "found" : "MISSING");
        return false;
    }
    std::printf("ARW LIB %s abi %u rows %d: P=%s (prio %d, launch %p) L=%s (prio %d, launch %p, same entry %s)\n", C.lib.c_str(),
                abi(), kc(), g_rowP->name, g_rowP->priority, (void*)g_rowP->launch, g_rowL->name, g_rowL->priority,
                (void*)g_rowL->launch, g_rowP->launch == g_rowL->launch ? "yes" : "NO");
    for (const RadKernelInfo* ki : { g_rowP, g_rowL }) {
        std::string s;
        for (int i = 0; i < ki->n_tunables; ++i) s += std::string(" ") + ki->tunables[i].key + "=" + std::to_string((long long)ki->tunables[i].deflt);
        std::printf("ARW TUNE %s defaults:%s\n", ki->name, s.c_str());
    }
    bool okall = true;
    const struct { const char* n; const void* w; } syms[3] = {
        { "r4d_ar_twoshot_4rank_exact", (const void*)&r4d_ar_twoshot_4rank_exact },
        { "r4d_ar_twoshot_4rank_exact_coded", (const void*)&r4d_ar_twoshot_4rank_exact_coded },
        { "r4d_ar_twoshot_coded_ok", (const void*)&r4d_ar_twoshot_coded_ok },
    };
    for (const auto& s : syms) {
        int found = 0;
        const int b = got_bound(g_lib, s.n, s.w, &found);
        const bool good = found > 0 && b == found;
        okall = okall && good;
        std::printf("ARW INTERPOSE %s: %d GOT slot(s) in the library, %d bound to this harness -> %s\n", s.n, found, b < 0 ? 0 : b,
                    good ? "OK" : "VIRHE (the wire that runs cannot be seen)");
    }
    return okall;
}

/* ------------------------------------------------------------------ host-only self-check (--selftest, no device) */
static int selftest_host() {
    int bad = 0;
    auto chk = [&](bool c, const char* what) { if (!c) { ++bad; std::printf("ARW SELFTEST FAIL %s\n", what); } };
    /* the restated raw rule on r4d_selftest's ar_coded_value kinds: range 14 coded, 15 raw, subnormal raw, inf raw, zeros coded */
    std::vector<uint16_t> g(64, 0x3f80);
    chk(!arc_raw(g.data()), "uniform group coded");
    g[5] = (uint16_t)((127 + 14) << 7); chk(!arc_raw(g.data()), "exponent range 14 coded");
    g[5] = (uint16_t)((127 + 15) << 7); chk(arc_raw(g.data()), "exponent range 15 raw");
    g.assign(64, 0x3f80); g[17] = 0x0005; chk(arc_raw(g.data()), "subnormal raw");
    g.assign(64, 0x3f80); g[40] = 0x7f80; chk(arc_raw(g.data()), "inf raw");
    g.assign(64, 0); chk(!arc_raw(g.data()), "zeros coded");
    /* rounding: RNE as r4d_selftest's h_to_bf16 */
    chk(f2bf(1.0f) == 0x3f80, "bf16(1)");
    chk(f2bf(1.0f + std::ldexp(1.0f, -8)) == 0x3f80, "bf16 tie to even (down)");
    chk(f2bf(1.0f + 3.0f * std::ldexp(1.0f, -8)) == 0x3f82, "bf16 tie to even (up)");
    /* patterns deterministic, finite except bin's planted +inf, actz rank 3 zero */
    const long N = C.N, n = 4 * N;
    long long inf_seen = 0, nan_seen = 0, z3 = 0;
    for (int p : { PBIN, PACT, PACTZ })
        for (int r = 0; r < 4; ++r)
            for (long i = 0; i < n; ++i) {
                const uint64_t sd = mix64(C.seed ^ (4ull << 20) ^ (uint64_t)p);
                const uint16_t a = gen_val(p, sd, r, i, N), b = gen_val(p, sd, r, i, N);
                if (a != b) { ++bad; std::printf("ARW SELFTEST FAIL pattern %s not deterministic\n", pat_name(p)); goto next; }
                if (((a >> 7) & 0xff) == 0xff) { if (a & 0x7f) ++nan_seen; else ++inf_seen; }
                if (p == PACTZ && r == 3 && a != 0) ++z3;
            }
next:
    chk(nan_seen == 0, "no NaN in any pattern");
    chk(z3 == 0, "actz rank 3 all zeros");
    /* the library's own coded_ok (host) at every shape against this run's slot */
    const long long maxn = C.maxrows * C.N;
    const long slot16 = (long)((maxn * 2 + 15) / 16);
    for (long T : C.rows) {
        const long nn = T * C.N;
        const int ok = g_real_ok(nn, 0, 4, slot16);
        std::printf("ARW CODEDOK T=%ld n=%ld slot16=%ld (instance numel %lld) -> r4d_ar_twoshot_coded_ok %d\n", T, nn, slot16, maxn, ok);
    }
    std::printf("ARW SELFTEST %s (%d check(s) failed; inf planted %lld)\n", bad ? "FAIL" : "OK", bad, inf_seen);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ a rank */
static void rank_main(int r) {
    hipStream_t st = nullptr;
    ARW_CK(hipSetDevice(r));
    ARW_CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    const long long maxn = C.maxrows * C.N;
    std::vector<RadParam> ipP = init_params(false, maxn), ipL = init_params(true, maxn);
    std::vector<RadParam> lpP = launch_params(g_rowP, false, maxn), lpL = launch_params(g_rowL, true, maxn);
    void *instP = nullptr, *instL = nullptr;
    const int rcP = g_rowP->init(ipP.data(), (int)ipP.size(), r, 4, &instP);
    const int rcL = rcP == RAD_OK ? g_rowL->init(ipL.data(), (int)ipL.size(), r, 4, &instL) : RAD_E_STATE;
    if (rcP != RAD_OK || rcL != RAD_OK) {
        std::printf("ARW VIRHE rank %d init: P rc %d, L rc %d\n", r, rcP, rcL);
        g_setup_bad = 1;
    }
    void *x = nullptr, *y = nullptr, *w = nullptr;
    ARW_CK(hipMalloc(&x, (size_t)maxn * 2 + 256));
    ARW_CK(hipMalloc(&y, (size_t)maxn * 2 + 256));
    ARW_CK(hipMalloc(&w, (size_t)maxn * 2 + 256));
    hipEvent_t e0, e1;
    ARW_CK(hipEventCreate(&e0));
    ARW_CK(hipEventCreate(&e1));
    g_bar.wait();
    if (r == 0 && !g_setup_bad)
        std::printf("ARW INIT 4 ranks: P instance %s L instance (one instance a rank: the library's cache key is rank, world, dtype, "
                    "numel, host)\n", instP == instL ? "==" : "!=");
    if (g_setup_bad) goto fini;

    for (long T : C.rows) {
        const long n = T * C.N;
        for (int pat : C.pats) {
            const uint64_t sd = mix64(C.seed ^ ((uint64_t)T << 20) ^ (uint64_t)pat);
            g_hx[r].resize((size_t)n);
            for (long i = 0; i < n; ++i) g_hx[r][(size_t)i] = gen_val(pat, sd, r, i, C.N);
            ARW_CK(hipMemcpyAsync(x, g_hx[r].data(), (size_t)n * 2, hipMemcpyHostToDevice, st));
            ARW_CK(hipStreamSynchronize(st));
            g_hplain[r].resize((size_t)n);
            g_hcur[r].resize((size_t)n);
            g_bar.wait();
            if (r == 0) {
                g_horacle.resize((size_t)n);
                for (long i = 0; i < n; ++i) {
                    float acc = 0.0f;
                    for (int q = 0; q < 4; ++q) acc += bf2f(g_hx[q][(size_t)i]);
                    g_horacle[(size_t)i] = f2bf(acc);
                }
                long long r1 = 0, t1 = 0, r2 = 0, t2 = 0;
                for (int q = 0; q < 4; ++q) raw_share(g_hx[q].data(), n, r1, t1);
                raw_share(g_horacle.data(), n, r2, t2);
                std::printf("ARW PAT T=%ld n=%ld pat=%s raw_in_pct=%.3f raw_sum_pct=%.3f (r4d_arc_enc's raw rule restated: groups of 64 "
                            "sent raw, inputs of all ranks / the folded sum)\n", T, n, pat_name(pat), t1 ? 100.0 * r1 / t1 : 0.0,
                            t2 ? 100.0 * r2 / t2 : 0.0);
                std::fflush(stdout);
            }
            g_bar.wait();
            /* identity: every variant in place (DS's form) and out of place */
            for (size_t v = 0; v < V.size() && !g_bad; ++v) {
                for (int form = 0; form < 2 && !g_bad; ++form) {
                    const bool inplace = form == 0;
                    if (inplace) ARW_CK(hipMemcpyAsync(w, x, (size_t)n * 2, hipMemcpyDeviceToDevice, st));
                    else ARW_CK(hipMemsetAsync(y, 0xff, (size_t)n * 2, st));
                    ARW_CK(hipStreamSynchronize(st));
                    g_bar.wait();
                    tp = Probe();
                    tp_nb_override = V[v].nb;
                    RadTensor ts[2] = { tens1(inplace ? w : x, n), tens1(y, n) };
                    std::vector<RadParam>& lp = V[v].coded ? lpL : lpP;
                    RadArgs a;
                    std::memset(&a, 0, sizeof a);
                    a.t = ts; a.n_t = inplace ? 1 : 2; a.p = lp.data(); a.n_p = (int)lp.size();
                    a.instance = V[v].coded ? instL : instP; a.rank = r; a.world_size = 4; a.kv_used_max_h = -1; a.wgp_count = g_wgp[r];
                    g_rc[r] = (V[v].coded ? g_rowL : g_rowP)->launch(&a, (RadStream)st);
                    ARW_CK(hipStreamSynchronize(st));
                    tp_nb_override = 0;
                    g_probe[r] = tp;
                    std::vector<uint16_t>& dst = (v == 0 && inplace) ? g_hplain[r] : g_hcur[r];
                    ARW_CK(hipMemcpyAsync(dst.data(), inplace ? w : y, (size_t)n * 2, hipMemcpyDeviceToHost, st));
                    ARW_CK(hipStreamSynchronize(st));
                    g_bar.wait();
                    if (r == 0) {
                        bool rc_ok = true, path_ok = true;
                        long long dP = 0, dO = 0;
                        long fP = -1, fO = -1;
                        int fPr = -1, fOr = -1;
                        std::string paths;
                        for (int q = 0; q < 4; ++q) {
                            const Probe& p = g_probe[q];
                            rc_ok = rc_ok && g_rc[q] == RAD_OK;
                            const bool pq = V[v].coded ? (p.coded_calls == 1 && p.plain_calls == 0 && p.ok_last == 1)
                                                       : (p.plain_calls == 1 && p.coded_calls == 0);
                            path_ok = path_ok && pq;
                            if (q == 0)
                                paths = V[v].coded ? ("coded calls " + std::to_string(p.coded_calls) + " plain " + std::to_string(p.plain_calls) +
                                                      " coded_ok " + std::to_string(p.ok_last) + " nb " + std::to_string(p.coded_nb_in) + "->" +
                                                      std::to_string(p.coded_nb_used) + " nt " + std::to_string(p.coded_nt) + " drain " +
                                                      std::to_string(p.coded_drain))
                                                   : ("plain calls " + std::to_string(p.plain_calls) + " coded " + std::to_string(p.coded_calls) +
                                                      " nb " + std::to_string(p.plain_nb) + " nt " + std::to_string(p.plain_nt) + " drain " +
                                                      std::to_string(p.plain_drain));
                            const std::vector<uint16_t>& got = (v == 0 && inplace) ? g_hplain[q] : g_hcur[q];
                            if (!(v == 0 && inplace)) {
                                const Diff d = cmp16(g_hplain[q].data(), got.data(), n);
                                if (d.n && fP < 0) { fP = d.first; fPr = q; }
                                dP += d.n;
                            }
                            const Diff o = cmp16(g_horacle.data(), got.data(), n);
                            if (o.n && fO < 0) { fO = o.first; fOr = q; }
                            dO += o.n;
                        }
                        const bool same = rc_ok && path_ok && dP == 0;
                        std::printf("ARW ID T=%ld n=%ld pat=%s var=%s form=%s rc=%d,%d,%d,%d path=%s [%s] vsP_diff=%lld%s vsOracle_diff=%lld%s -> %s\n",
                                    T, n, pat_name(pat), V[v].name.c_str(), inplace ? "inplace" : "oop", g_rc[0], g_rc[1], g_rc[2], g_rc[3],
                                    path_ok ? "OK" : "WRONG", paths.c_str(), dP,
                                    fP >= 0 ? (" (first elem " + std::to_string(fP) + " rank " + std::to_string(fPr) + ")").c_str() : "",
                                    dO, fO >= 0 ? (" (first elem " + std::to_string(fO) + " rank " + std::to_string(fOr) + ")").c_str() : "",
                                    same ? (v == 0 && inplace ? "REF" : "IDENTICAL") : "VIRHE");
                        if (fP >= 0) {
                            std::string vals;
                            for (int q = 0; q < 4; ++q) {
                                char b[96];
                                std::snprintf(b, sizeof b, " r%d in %04x P %04x got %04x", q, g_hx[q][(size_t)fP], g_hplain[q][(size_t)fP],
                                              g_hcur[q][(size_t)fP]);
                                vals += b;
                            }
                            std::printf("ARW VIRHE first difference T=%ld pat=%s var=%s form=%s elem %ld (row %ld col %ld):%s oracle %04x\n", T,
                                        pat_name(pat), V[v].name.c_str(), inplace ? "inplace" : "oop", fP, fP / C.N, fP % C.N, vals.c_str(),
                                        g_horacle[(size_t)fP]);
                        }
                        if (dO) g_oracle_notes += dO;
                        if (!same) g_bad = 1;
                        std::fflush(stdout);
                    }
                    g_bar.wait();
                }
            }
            if (g_bad) goto fini;
            /* timing */
            if (std::find(C.tpats.begin(), C.tpats.end(), pat) == C.tpats.end()) { g_bar.wait(); continue; }
            for (size_t v = 0; v < V.size(); ++v) { g_t[v][r].assign((size_t)C.reps, 0.0); g_bb[v][r] = 0; }
            g_pathbad[r] = 0;
            g_launchbad[r] = 0;
            auto one = [&](size_t v, bool inplace, bool timed, int rep) {
                tp = Probe();
                tp_nb_override = V[v].nb;
                RadTensor ts[2] = { tens1(inplace ? w : x, n), tens1(y, n) };
                std::vector<RadParam>& lp = V[v].coded ? lpL : lpP;
                RadArgs a;
                std::memset(&a, 0, sizeof a);
                a.t = ts; a.n_t = inplace ? 1 : 2; a.p = lp.data(); a.n_p = (int)lp.size();
                a.instance = V[v].coded ? instL : instP; a.rank = r; a.world_size = 4; a.kv_used_max_h = -1; a.wgp_count = g_wgp[r];
                if (timed) ARW_CK(hipEventRecord(e0, st));
                const int rc = (V[v].coded ? g_rowL : g_rowP)->launch(&a, (RadStream)st);
                if (timed) ARW_CK(hipEventRecord(e1, st));
                tp_nb_override = 0;
                if (rc != RAD_OK) ++g_launchbad[r];
                const bool pq = V[v].coded ? (tp.coded_calls == 1 && tp.plain_calls == 0) : (tp.plain_calls == 1 && tp.coded_calls == 0);
                if (!pq) ++g_pathbad[r];
                if (timed) {
                    ARW_CK(hipEventSynchronize(e1));
                    float ms = 0;
                    ARW_CK(hipEventElapsedTime(&ms, e0, e1));
                    g_t[v][r][(size_t)rep] = ms * 1000.0;
                }
            };
            for (size_t v = 0; v < V.size(); ++v)
                for (int k = 0; k < C.warm; ++k) {
                    ARW_CK(hipMemcpyAsync(w, x, (size_t)n * 2, hipMemcpyDeviceToDevice, st));
                    ARW_CK(hipStreamSynchronize(st));
                    g_bar.wait();
                    one(v, true, false, 0);
                    ARW_CK(hipStreamSynchronize(st));
                }
            for (int i = 0; i < C.reps; ++i)
                for (size_t j = 0; j < V.size(); ++j) {
                    const size_t v = (i & 1) ? V.size() - 1 - j : j;
                    ARW_CK(hipMemcpyAsync(w, x, (size_t)n * 2, hipMemcpyDeviceToDevice, st));
                    ARW_CK(hipStreamSynchronize(st));
                    g_bar.wait();
                    one(v, true, true, i);
                }
            for (size_t v = 0; v < V.size(); ++v) {
                g_bar.wait();
                ARW_CK(hipEventRecord(e0, st));
                for (int k = 0; k < C.bb; ++k) one(v, false, false, 0);
                ARW_CK(hipEventRecord(e1, st));
                ARW_CK(hipEventSynchronize(e1));
                float ms = 0;
                ARW_CK(hipEventElapsedTime(&ms, e0, e1));
                g_bb[v][r] = ms * 1000.0 / C.bb;
            }
            g_bar.wait();
            if (r == 0) {
                long long pb = 0, lb = 0;
                for (int q = 0; q < 4; ++q) { pb += g_pathbad[q]; lb += g_launchbad[q]; }
                double pmed = 0;
                for (size_t v = 0; v < V.size(); ++v) {
                    std::vector<double> own((size_t)C.reps), mx((size_t)C.reps);
                    for (int i = 0; i < C.reps; ++i) {
                        double lo = 1e30, hi = 0;
                        for (int q = 0; q < 4; ++q) { lo = std::min(lo, g_t[v][q][(size_t)i]); hi = std::max(hi, g_t[v][q][(size_t)i]); }
                        own[(size_t)i] = lo;
                        mx[(size_t)i] = hi;
                    }
                    std::sort(own.begin(), own.end());
                    std::sort(mx.begin(), mx.end());
                    const double med = own[(size_t)C.reps / 2], p10 = own[(size_t)(C.reps * 0.1)],
                                 p90 = own[std::min((size_t)C.reps - 1, (size_t)(C.reps * 0.9))], mmed = mx[(size_t)C.reps / 2];
                    double bbv = 0;
                    for (int q = 0; q < 4; ++q) bbv = std::max(bbv, g_bb[v][q]);
                    if (v == 0) pmed = med;
                    std::printf("ARW TIME T=%ld n=%ld pat=%s var=%s own_med_us=%.1f own_p10_us=%.1f own_p90_us=%.1f max_med_us=%.1f bb_us=%.1f "
                                "reps=%d bb=%d algbw_gbs=%.2f vsP_pct=%+.2f\n", T, n, pat_name(pat), V[v].name.c_str(), med, p10, p90, mmed, bbv,
                                C.reps, C.bb, (double)n * 2 / (med * 1e-6) / 1e9, v == 0 ? 0.0 : 100.0 * (med / pmed - 1.0));
                }
                std::printf("ARW TIMECHK T=%ld pat=%s launches_not_ok=%lld wrong_path=%lld\n", T, pat_name(pat), lb, pb);
                if (lb || pb) {
                    std::printf("ARW VIRHE timed launches: %lld not RAD_OK, %lld on the wrong wire\n", lb, pb);
                    g_bad = 1;
                }
                std::fflush(stdout);
            }
            g_bar.wait();
            if (g_bad) goto fini;
        }
    }
fini:
    g_bar.wait();
    ARW_CK(hipStreamSynchronize(st));
    if (instL && g_rowL->fini) g_rowL->fini(instL);
    if (instP && g_rowP->fini) g_rowP->fini(instP);
    (void)hipFree(x);
    (void)hipFree(y);
    (void)hipFree(w);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    (void)hipStreamDestroy(st);
}

static std::vector<long> csvl(const char* s) {
    std::vector<long> v;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back(x);
        p = (*e == ',') ? e + 1 : e;
        if (!*e) break;
    }
    return v;
}
static void usage() {
    std::printf("usage: arw [--lib PATH] [--rows T,T,..] [--n 5120] [--maxrows 4160] [--reps 100] [--warm 5] [--bb 20] [--seed S]\n"
                "           [--nbs 8,16,24] [--pats bin,act,actz] [--tpats act,bin] [--selftest]\n");
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&]() -> const char* { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--lib") C.lib = need();
        else if (a == "--rows") C.rows = csvl(need());
        else if (a == "--n") C.N = std::atol(need());
        else if (a == "--maxrows") C.maxrows = std::atol(need());
        else if (a == "--reps") C.reps = std::atoi(need());
        else if (a == "--warm") C.warm = std::atoi(need());
        else if (a == "--bb") C.bb = std::atoi(need());
        else if (a == "--seed") C.seed = std::strtoull(need(), nullptr, 0);
        else if (a == "--nbs") { C.nbs.clear(); for (long x : csvl(need())) C.nbs.push_back((int)x); }
        else if (a == "--pats" || a == "--tpats") {
            std::vector<int>& dst = a == "--pats" ? C.pats : C.tpats;
            dst.clear();
            std::string s = need();
            size_t p0 = 0;
            while (p0 <= s.size()) {
                const size_t p1 = s.find(',', p0);
                const std::string t = s.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
                if (!t.empty()) { const int k = pat_of(t); if (k < 0) { usage(); return 2; } dst.push_back(k); }
                if (p1 == std::string::npos) break;
                p0 = p1 + 1;
            }
        } else if (a == "--selftest") C.selftest = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { usage(); return 2; }
    }
    for (int t : C.tpats)
        if (std::find(C.pats.begin(), C.pats.end(), t) == C.pats.end()) { std::printf("ARW usage: every --tpats entry must be in --pats\n"); return 2; }
    if (C.rows.empty() || C.reps < 10 || C.bb < 1 || C.N <= 0) { usage(); return 2; }
    for (long T : C.rows)
        if (T < 1 || T > C.maxrows || (T * C.N) % 1024) { std::printf("ARW usage: T=%ld outside 1..maxrows or T*n not a multiple of 1024\n", T); return 2; }
    V.push_back({ "P", false, 0 });
    V.push_back({ "C", true, 0 });
    for (int nb : C.nbs)
        if (nb > 0 && V.size() < 8) V.push_back({ "C" + std::to_string(nb), true, nb });
    std::string vs;
    for (const Var& v : V) vs += " " + v.name;
    std::string ps, tps, rs;
    for (int p : C.pats) ps += std::string(ps.empty() ? "" : ",") + pat_name(p);
    for (int p : C.tpats) tps += std::string(tps.empty() ? "" : ",") + pat_name(p);
    for (long T : C.rows) rs += (rs.empty() ? "" : ",") + std::to_string(T);
    std::printf("ARW CFG rows=%s n=%ld maxrows=%ld (instance numel %ld) reps=%d warm=%d bb=%d seed=%llu variants:%s pats=%s tpats=%s\n", rs.c_str(),
                C.N, C.maxrows, C.maxrows * C.N, C.reps, C.warm, C.bb, (unsigned long long)C.seed, vs.c_str(), ps.c_str(), tps.c_str());
    if (!load_lib()) { std::printf("ARW VIRHE setup (library / rows / interposition)\n"); return 1; }
    if (C.selftest) return selftest_host();

    int ndev = 0;
    ARW_CK(hipGetDeviceCount(&ndev));
    if (ndev < 4) { std::printf("ARW VIRHE %d HIP device(s) visible, the four-rank two-shot needs 4\n", ndev); return 1; }
    for (int d = 0; d < 4; ++d) {
        hipDeviceProp_t pr;
        ARW_CK(hipGetDeviceProperties(&pr, d));
        g_wgp[d] = pr.multiProcessorCount;
        char bus[64] = { 0 };
        (void)hipDeviceGetPCIBusId(bus, sizeof bus, d);
        std::printf("ARW DEV %d %s %s cus=%d\n", d, pr.gcnArchName, bus, pr.multiProcessorCount);
    }
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            if (a == b) continue;
            int can = 0;
            if (hipDeviceCanAccessPeer(&can, a, b) != hipSuccess || !can) {
                std::printf("ARW VIRHE no peer path %d -> %d (device.p2p)\n", a, b);
                return 1;
            }
        }
    std::printf("ARW PEER 4 cards, every pair peer-accessible\n");
    std::fflush(stdout);
    std::vector<std::thread> th;
    for (int r = 0; r < 4; ++r) th.emplace_back(rank_main, r);
    for (auto& t : th) t.join();
    if (g_setup_bad) { std::printf("ARW VIRHE setup on a rank\n"); return 1; }
    std::printf("ARW DONE %s oracle_diffs=%lld\n", g_bad ? "VIRHE" : "IDENTICAL", g_oracle_notes);
    std::fflush(stdout);
    return g_bad ? 1 : 0;
}
