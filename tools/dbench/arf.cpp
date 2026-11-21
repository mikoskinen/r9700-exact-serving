/* arf (DRAFTER track 21.11.2026 round 58, lever DAR b) -- ONE card: the fp8-wire two-shot all-reduce (libr4d/r4d_ar_fp8.h)
 * against the exact wide two-shot it would replace (r4d_ar_twoshot_wide_exact_body, the served form of the drafter's
 * [rows, 5120] bf16 reductions), with the four ranks as four kernels on four streams of ONE device (each rank's "peer"
 * scratch / flags are the other ranks' buffers in the same VRAM). The Xeon's two cards have no P2P, so this measures
 * the codec's cost and the data path, not the link: the wire's saving is the link's (bytes 56.25 %), to be read on the
 * pair.
 *
 *   codec   the device codec (hardware v_cvt_pk_fp8_f32 / v_cvt_pk_f32_fp8 / v_ldexp_f32) == the host codec r4d_arf_sw_*,
 *           byte for byte: every bf16 pattern as a group member beside 9 companion magnitudes (the scale's edges
 *           1.75 / 1.7578125, tiny, huge, 0), random f32 groups over 270 binades + narrow ones, every code x scale byte;
 *   ar      every rank's output of both forms at 1 .. 64 rows == the host fold (exact: fp32 ascending rank, one bf16
 *           rounding; fp8: dequant(quant(sum of dequant(quant(x_r))))) bit for bit, all four ranks equal, three repeats
 *           equal (deterministic), and row r the same at every message size (batch-invariant; a row's values are a
 *           function of (rank, row, column) only); the fp8 values' distance from the exact ones (rms / max relative);
 *   time    us a call (the slowest rank, median of N back-to-back launches) exact at the served grid (nb_for_shard: 2 x
 *           1024 at <= 64 rows) vs fp8 at nb 2 / 4 / 8 and 256 / 1024 threads.
 *
 * usage: arf [--rows 8,16,32,40,48] [--iters 200] [--no-codec] [--no-time] */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "r4d_ar_twoshot_Nrank_exact.hip"
#define R4D_ARF_BODY 1
#include "r4d_ar_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

constexpr int WS = 4;
constexpr int64_t kW = 5120;
int g_fail = 0, g_cases = 0;
void check(bool ok, const char* what, const std::string& detail = "") {
  ++g_cases; if (!ok) ++g_fail;
  std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : " -- ", detail.c_str());
  std::fflush(stdout);
}
uint16_t bf16_rne(float f) { uint32_t u; std::memcpy(&u, &f, 4); if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }
float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

/* ------------------------------------------------------------------ codec kernels */
__global__ void k_enc(const float* __restrict__ in, int2* __restrict__ d, unsigned char* __restrict__ sc, int G) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = in[(size_t)g * 8 + i];
  int2 dd; unsigned s; r4d_arf_enc8(v, dd, s);
  d[g] = dd; sc[g] = (unsigned char)s;
}
__global__ void k_dec(float* __restrict__ out) {   // 65536 threads: code = t & 255 in all 8 bytes, scale byte = t >> 8
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= 65536) return;
  const unsigned c = (unsigned)t & 255u, sb = (unsigned)t >> 8;
  const int w = (int)(c * 0x01010101u);
  float v[8]; r4d_arf_dec8(make_int2(w, w), sb, v);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[(size_t)t * 8 + i] = v[i];
}

void codec_tests() {
  std::mt19937_64 rng(0x5eed0058ull);
  // (a) every bf16 pattern beside companions (the scale's edges and extremes) and random members below them
  const float comp[9] = { 0.0f, 1.0f, 1.75f, 1.7578125f, 448.0f, 3.0e38f, 1.0e-30f, 6.0e-36f, -0.7f };
  std::vector<float> in;
  for (int ci = 0; ci < 9; ++ci) {
    const float M = bf16_f(bf16_rne(comp[ci]));
    int eM = 0; if (M != 0.0f) std::frexp(M, &eM);
    for (uint32_t b = 0; b < 65536; ++b) {
      float g[8];
      g[0] = bf16_f((uint16_t)b); g[1] = M;
      for (int i = 2; i < 8; ++i) {
        const int e = eM - (int)(rng() % 24);
        const float m = 1.0f + (float)(rng() % 128) / 128.0f;
        g[i] = bf16_f(bf16_rne(std::ldexp((rng() & 1) ? -m : m, e - 1)));
        if (rng() % 16 == 0) g[i] = (rng() & 1) ? -0.0f : 0.0f;
      }
      if (b & 1) std::swap(g[0], g[(b >> 1) % 8]);
      in.insert(in.end(), g, g + 8);
    }
  }
  const size_t Ga = in.size() / 8;
  // (b) random f32 groups: wide (any exponent 2^-140 .. 2^130) and narrow (one binade +- 3), plus sums of bf16 values
  for (int g = 0; g < 1000000; ++g) {
    const int mode = g % 3;
    const int e0 = (int)(rng() % 271) - 140;
    for (int i = 0; i < 8; ++i) {
      uint32_t u;
      if (mode == 0) { const int e = (int)(rng() % 271) - 140 + 127; u = (e <= 0 ? 0u : ((uint32_t)std::min(e, 254) << 23)) | (uint32_t)(rng() & 0x7FFFFF); }
      else if (mode == 1) { const int e = std::max(1, std::min(254, e0 + 127 + (int)(rng() % 7) - 3)); u = ((uint32_t)e << 23) | (uint32_t)(rng() & 0x7FFFFF); }
      else { float s = 0.0f; for (int r = 0; r < 4; ++r) { const float m = 1.0f + (float)(rng() % 128) / 128.0f; s += bf16_f(bf16_rne(std::ldexp((rng() & 1) ? -m : m, (int)(rng() % 9) - 4))); } std::memcpy(&u, &s, 4); }
      if (mode != 2 && (rng() & 1)) u |= 0x80000000u;
      float f; std::memcpy(&f, &u, 4); in.push_back(f);
    }
  }
  // edge groups: all zero, -0, an inf, a NaN, f32 subnormals, the largest finite
  const float edges[][8] = {
    { 0, 0, 0, 0, 0, 0, 0, 0 }, { -0.0f, 0, -0.0f, 0, 0, 0, 0, -0.0f }, { 1, 2, INFINITY, 4, 5, 6, 7, 8 },
    { 1, NAN, 3, 4, 5, 6, 7, 8 }, { 1e-40f, -1e-41f, 1e-45f, 0, 0, 0, 0, 2e-39f }, { 3.4e38f, -3.4e38f, 1, 0, 0, 0, 0, 0 },
    { 1.0e-38f, 1.2e-38f, 5.9e-39f, 0, 0, 0, 0, 0 } };
  for (auto& e : edges) in.insert(in.end(), e, e + 8);
  const int G = (int)(in.size() / 8);

  float* dIn; int2* dD; unsigned char* dS;
  CK(hipMalloc(&dIn, in.size() * 4)); CK(hipMalloc(&dD, (size_t)G * 8)); CK(hipMalloc(&dS, (size_t)G));
  CK(hipMemcpy(dIn, in.data(), in.size() * 4, hipMemcpyHostToDevice));
  k_enc<<<(G + 255) / 256, 256>>>(dIn, dD, dS, G);
  CK(hipDeviceSynchronize());
  std::vector<unsigned char> hd((size_t)G * 8), hs((size_t)G);
  CK(hipMemcpy(hd.data(), dD, hd.size(), hipMemcpyDeviceToHost));
  CK(hipMemcpy(hs.data(), dS, hs.size(), hipMemcpyDeviceToHost));
  long bad_a = 0, bad_b = 0, bad_e = 0; std::string first;
  for (int g = 0; g < G; ++g) {
    unsigned char c[8], s;
    r4d_arf_sw_enc8(&in[(size_t)g * 8], c, s);
    const bool ok = s == hs[(size_t)g] && std::memcmp(c, &hd[(size_t)g * 8], 8) == 0;
    if (!ok) {
      ((size_t)g < Ga ? bad_a : g < G - 7 ? bad_b : bad_e)++;
      if (first.empty()) {
        char t[512]; std::snprintf(t, sizeof t, "group %d: in %08x %08x %08x %08x %08x %08x %08x %08x sw s %02x %02x%02x%02x%02x%02x%02x%02x%02x hw s %02x %02x%02x%02x%02x%02x%02x%02x%02x", g,
          fbits(in[g*8]), fbits(in[g*8+1]), fbits(in[g*8+2]), fbits(in[g*8+3]), fbits(in[g*8+4]), fbits(in[g*8+5]), fbits(in[g*8+6]), fbits(in[g*8+7]),
          s, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], hs[g], hd[g*8], hd[g*8+1], hd[g*8+2], hd[g*8+3], hd[g*8+4], hd[g*8+5], hd[g*8+6], hd[g*8+7]);
        first = t;
      }
    }
  }
  char t[256];
  std::snprintf(t, sizeof t, "%zu bf16-pattern groups %ld differ, %d f32 groups %ld differ, 7 edge groups %ld differ", Ga, bad_a, G - 7 - (int)Ga, bad_b, bad_e);
  check(bad_a == 0 && bad_b == 0 && bad_e == 0, "codec encode: device (v_cvt_pk_fp8_f32 RNE) == host r4d_arf_sw_enc8", std::string(t) + (first.empty() ? "" : " | first " + first));

  float* dO; CK(hipMalloc(&dO, 65536 * 8 * 4));
  k_dec<<<256, 256>>>(dO);
  CK(hipDeviceSynchronize());
  std::vector<float> ho(65536 * 8);
  CK(hipMemcpy(ho.data(), dO, ho.size() * 4, hipMemcpyDeviceToHost));
  long bad_d = 0; first.clear();
  long ndec = 0;
  for (int tt = 0; tt < 65536; ++tt) {
    const int sb = tt >> 8;   /* the scale bytes the encoder produces: s in [-117, 120] and 0xFF */
    if (sb != 0xFF && (sb < 127 + R4D_ARF_SMIN || sb > 127 + 120)) continue;
    ++ndec;
    unsigned char c[8]; std::memset(c, tt & 255, 8);
    float v[8]; r4d_arf_sw_dec8(c, (unsigned char)sb, v);
    for (int i = 0; i < 8; ++i) if (fbits(v[i]) != fbits(ho[(size_t)tt * 8 + i]) && !(std::isnan(v[i]) && std::isnan(ho[(size_t)tt * 8 + i]))) {
      if (first.empty()) { char u[128]; std::snprintf(u, sizeof u, "code %02x scale %02x sw %08x hw %08x", tt & 255, tt >> 8, fbits(v[i]), fbits(ho[tt * 8 + i])); first = u; }
      ++bad_d; break;
    }
  }
  std::snprintf(t, sizeof t, "%ld code x scale pairs (256 codes x the produced scale bytes), %ld differ (NaN == NaN)", ndec, bad_d);
  check(bad_d == 0, "codec decode: device (v_cvt_pk_f32_fp8 + v_ldexp_f32) == host r4d_arf_sw_dec8", std::string(t) + (first.empty() ? "" : " | first " + first));
  CK(hipFree(dIn)); CK(hipFree(dD)); CK(hipFree(dS)); CK(hipFree(dO));
}

/* ------------------------------------------------------------------ the all-reduce on 4 ranks of one card */
__global__ void k_exact(const int4* in4, R4DArWidePeers<int4, WS - 1> peers, const int4* my_base, unsigned int* my_flags,
                        int4* out4, unsigned int* seq, R4DArShardOff<WS> sh, int slot16, int rank, int drain, int acq, int pub) {
  r4d_ar_twoshot_wide_exact_body<__hip_bfloat16, 8, WS, false, R4D_AR8_FORM_LOCKSTEP, false>(
      in4, peers, my_base, my_flags, out4, seq, sh, slot16, rank, drain, acq, pub, nullptr, nullptr, 1, nullptr);
}
__global__ void k_fp8(const int4* in4, R4DArWidePeers<int4, WS - 1> peers, const int4* my_base, unsigned int* my_flags,
                      int4* out4, unsigned int* seq, R4DArShardOff<WS> sh, int slot16, int rank, int drain, int acq, int pub) {
  r4d_ar_twoshot_fp8_body<WS>(in4, peers, my_base, my_flags, out4, seq, sh, slot16, rank, drain, acq, pub, nullptr, nullptr);
}

/* --phase: one rank's loops alone (no sequence take, no handshakes, slot 0; the "peers" are the other ranks' buffers on
 * this card), one stream: the exact wire's copies + fp32 fold vs the fp8 wire's codec + fold, the same grid */
__global__ void k_exact_nohs(const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
                             int4* __restrict__ out4, R4DArShardOff<WS> sh, int slot16, int rank) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int sh_max = sh.sh_max, w = (sh_max + nb - 1) / nb, w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
  int4* pa[NPEERS];
  for (int p = 0; p < NPEERS; ++p) pa[p] = peers.scratch[p] + (size_t)rank * slot16;
  for (int i = 0; i < NPEERS; ++i) {
    const int owner = peers.rank[i];
    const int4* s1 = in4 + sh.off[owner];
    const int len = sh.off[owner + 1] - sh.off[owner], lim = w1 < len ? w1 : len;
    for (int k = w0 + tid; k < lim; k += nt) pa[i][k] = s1[k];
  }
  __syncthreads();
  const int4* src[WS];
  for (int r = 0; r < WS; ++r) src[r] = r == rank ? in4 + sh.off[rank] : my_base + (size_t)r * slot16;
  int4* my_out = out4 + sh.off[rank];
  const int my_len = sh.off[rank + 1] - sh.off[rank], my_lim = w1 < my_len ? w1 : my_len;
  for (int k = w0 + tid; k < my_lim; k += nt) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      const int4 v = src[r][k];
      const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
      for (int l = 0; l < 8; ++l) acc[l] += (float)pv[l];
    }
    int4 o; __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
    for (int l = 0; l < 8; ++l) po[l] = (__hip_bfloat16)acc[l];
    my_out[k] = o;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;
  }
  __syncthreads();
  for (int i = 0; i < NPEERS; ++i) {
    const int owner = peers.rank[i];
    const int4* srcB = my_base + (size_t)owner * slot16 + sh_max;
    int4* dst = out4 + sh.off[owner];
    const int len = sh.off[owner + 1] - sh.off[owner], lim = w1 < len ? w1 : len;
    for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
  }
}
__global__ void k_fp8_nohs(const int4* in4, R4DArWidePeers<int4, WS - 1> peers, const int4* my_base, int4* out4,
                           R4DArShardOff<WS> sh, int slot16, int rank) {
  r4d_ar_twoshot_fp8_body<WS, false>(in4, peers, my_base, nullptr, out4, nullptr, sh, slot16, rank, 3, 0, 1, nullptr, nullptr);
}

/* gather form (moe_gather_all_reduce's two-shot): the kernels as libr4d launches them, plus the gather alone */
__global__ void k_gonly(int4* y4, R4dArGather G, int n16, int wpr) {
  __shared__ int   s_i[R4D_ARGL_TAB];
  __shared__ float s_w[R4D_ARGL_TAB];
  __shared__ int   s_n[R4D_ARGL_ROWS];
  __shared__ float s_g[R4D_ARGL_ROWS];
  r4d_ar_gather_list(G, s_i, s_w, s_n, s_g);
  const int K = G.top_k;
  for (int gw = blockIdx.x * blockDim.x + threadIdx.x; gw < n16; gw += gridDim.x * blockDim.x) {
    const int row = gw / wpr, i0 = (gw - row * wpr) * 8;
    y4[gw] = r4d_ar_gather_word_list(G, s_i + row * K, s_w + row * K, s_n[row], G.sh ? s_g[row] : 0.0f, row, i0);
  }
}

struct Ranks {
  int64_t max_rows = 64;
  long slot16 = 0;
  void* in[WS] = {}; void* out[WS] = {}; void* scratch[WS] = {}; unsigned int* flags[WS] = {}; unsigned int* seq[WS] = {};
  hipStream_t st[WS] = {};
  void init(int64_t rows) {
    max_rows = rows;
    const long n16 = rows * kW / 8, sh_max = (n16 + WS - 1) / WS;
    slot16 = 2 * sh_max + 16;
    for (int r = 0; r < WS; ++r) {
      CK(hipMalloc(&in[r], (size_t)rows * kW * 2)); CK(hipMalloc(&out[r], (size_t)rows * kW * 2));
      CK(hipMalloc(&scratch[r], (size_t)2 * WS * slot16 * 16));
      CK(hipMalloc(&flags[r], (size_t)R4D_AR_WIDE_MAX_BLOCKS * WS * 4)); CK(hipMemset(flags[r], 0, (size_t)R4D_AR_WIDE_MAX_BLOCKS * WS * 4));
      CK(hipMalloc(&seq[r], (size_t)(2 * R4D_AR_WIDE_MAX_BLOCKS + 4) * 4)); CK(hipMemset(seq[r], 0, (size_t)(2 * R4D_AR_WIDE_MAX_BLOCKS + 4) * 4));
      CK(hipStreamCreateWithFlags(&st[r], hipStreamNonBlocking));
    }
  }
  /* one all-reduce of `rows` rows on every rank (rank r on stream r) */
  void launch(bool fp8, int64_t rows, int nb, int nt) {
    const long n16 = rows * kW / 8;
    long off[WS + 1];
    const long sh_max = r4d_ar_shard_table(n16, WS, nullptr, off);
    R4DArShardOff<WS> sh{};
    for (int s = 0; s <= WS; ++s) sh.off[s] = (int)off[s];
    sh.sh_max = (int)sh_max;
    if (nb > sh_max) nb = (int)sh_max;
    const int drain = rows * kW >= 131072 ? 1 : 3;   // r4d_cfg_ar's rule (kArFenceFromElems at four ranks)
    for (int r = 0; r < WS; ++r) {
      R4DArWidePeers<int4, WS - 1> peers{};
      for (int i = 0; i < WS - 1; ++i) {
        const int pr = i < r ? i : i + 1;
        peers.scratch[i] = (int4*)scratch[pr]; peers.flags[i] = flags[pr]; peers.rank[i] = pr;
      }
      if (fp8) k_fp8<<<nb, nt, 0, st[r]>>>((const int4*)in[r], peers, (const int4*)scratch[r], flags[r], (int4*)out[r], seq[r], sh, (int)slot16, r, drain, 0, 1);
      else k_exact<<<nb, nt, 0, st[r]>>>((const int4*)in[r], peers, (const int4*)scratch[r], flags[r], (int4*)out[r], seq[r], sh, (int)slot16, r, drain, 0, 1);
    }
  }
  void sync() { for (int r = 0; r < WS; ++r) CK(hipStreamSynchronize(st[r])); }
};

/* activation-like: a hashed near-gaussian with a per-row scale 2^-4 .. 2^3 and 1 in 16 exact zeros, a function of
 * (rank, row, column) only */
float act(int rank, int64_t row, int64_t col) {
  uint64_t h = ((uint64_t)row * 0x9E3779B97F4A7C15ull + (uint64_t)col * 0xC2B2AE3D27D4EB4Full) ^ ((uint64_t)rank + 7) * 0xD1B54A32D192ED03ull;
  h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
  uint64_t r = (uint64_t)row * 0x94D049BB133111EBull + (uint64_t)rank * 0x2545F4914F6CDD1Dull; r ^= r >> 31;
  if ((h >> 60) == 0) return 0.0f;
  double s = 0.0;
  for (int k = 0; k < 4; ++k) s += (double)((h >> (13 * k)) & 0x1fff) / 4096.0 - 1.0;
  return (float)std::ldexp(s, (int)(r % 8) - 4);
}

void ar_tests(Ranks& R, const std::vector<int64_t>& rows_list) {
  const int64_t MR = R.max_rows;
  std::vector<std::vector<uint16_t>> hin(WS, std::vector<uint16_t>((size_t)MR * kW));
  for (int r = 0; r < WS; ++r)
    for (int64_t i = 0; i < MR * kW; ++i) hin[r][(size_t)i] = bf16_rne(act(r, i / kW, i % kW));
  // host references, per row (the same at every message size by construction)
  std::vector<uint16_t> ref_x((size_t)MR * kW), ref_f((size_t)MR * kW);
  for (int64_t g = 0; g < MR * kW / 8; ++g) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, accx[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int r = 0; r < WS; ++r) {
      float v[8]; for (int l = 0; l < 8; ++l) v[l] = bf16_f(hin[r][(size_t)g * 8 + l]);
      for (int l = 0; l < 8; ++l) accx[l] += v[l];
      unsigned char c[8], s; r4d_arf_sw_enc8(v, c, s);
      float d[8]; r4d_arf_sw_dec8(c, s, d);
      for (int l = 0; l < 8; ++l) acc[l] += d[l];
    }
    unsigned char c[8], s; r4d_arf_sw_enc8(acc, c, s);
    float d[8]; r4d_arf_sw_dec8(c, s, d);
    for (int l = 0; l < 8; ++l) { ref_f[(size_t)g * 8 + l] = bf16_rne(d[l]); ref_x[(size_t)g * 8 + l] = bf16_rne(accx[l]); }
  }
  for (int r = 0; r < WS; ++r) CK(hipMemcpy(R.in[r], hin[r].data(), (size_t)MR * kW * 2, hipMemcpyHostToDevice));
  std::vector<uint16_t> ho((size_t)MR * kW);
  for (int form = 0; form < 2; ++form) {
    const bool fp8 = form == 1;
    const std::vector<uint16_t>& ref = fp8 ? ref_f : ref_x;
    for (int64_t rows : rows_list) {
      for (int cfg = 0; cfg < (fp8 ? 3 : 1); ++cfg) {
        const int nb = fp8 ? (cfg == 0 ? 2 : cfg == 1 ? 4 : 8) : 2, nt = (fp8 && cfg == 2) ? 256 : 1024;
        long bad = 0, rep_diff = 0;
        uint64_t h0 = 0;
        for (int rep = 0; rep < 3; ++rep) {
          for (int r = 0; r < WS; ++r) CK(hipMemsetAsync(R.out[r], 0xA5, (size_t)rows * kW * 2, R.st[r]));
          R.sync();
          R.launch(fp8, rows, nb, nt);
          R.sync();
          uint64_t h = 1469598103934665603ull;
          for (int r = 0; r < WS; ++r) {
            CK(hipMemcpy(ho.data(), R.out[r], (size_t)rows * kW * 2, hipMemcpyDeviceToHost));
            for (int64_t i = 0; i < rows * kW; ++i) {
              if (ho[(size_t)i] != ref[(size_t)i]) ++bad;
              h = (h ^ ho[(size_t)i]) * 1099511628211ull;
            }
          }
          if (rep == 0) h0 = h; else if (h != h0) ++rep_diff;
        }
        char t[256];
        std::snprintf(t, sizeof t, "%s %lld rows, nb %d x %d: %ld of %lld elements x 3 repeats differ from the host fold; repeats differing %ld",
                      fp8 ? "fp8" : "exact", (long long)rows, nb, nt, bad, (long long)(rows * kW * WS), rep_diff);
        check(bad == 0 && rep_diff == 0, fp8 ? "ar fp8 == host fold (all ranks, deterministic, batch-invariant)" : "ar exact == host fold", t);
      }
    }
  }
  // the fp8 values' distance from the exact ones (over the largest message)
  double se = 0, sx = 0, mx = 0, mxr = 0;
  for (int64_t i = 0; i < MR * kW; ++i) {
    const double x = bf16_f(ref_x[(size_t)i]), f = bf16_f(ref_f[(size_t)i]);
    se += (f - x) * (f - x); sx += x * x;
    mx = std::max(mx, std::fabs(f - x));
    if (x != 0) mxr = std::max(mxr, std::fabs(f - x) / std::fabs(x));
  }
  std::printf("INFO fp8 vs exact over %lld rows x 5120 x 4 ranks of activation-like values: rms(d) / rms(x) %.5f, max |d| %.4g, max |d| / |x| %.4f\n",
              (long long)MR, std::sqrt(se / sx), mx, mxr);
}

/* the gather form: 4 ranks' routed outputs ye [M * K, 5120], weights ew [M, K], the sorted-row map, the shared arm sh
 * [M, 5120] and its gate sg [M, 1] (sigmoid), as the drafter's MoE gives them (top 3). fp8 gather two-shot == the
 * gather alone (k_gonly) + the wide fp8 two-shot, byte for byte; exact gather two-shot == the gather alone + the exact
 * wide two-shot (sanity). */
void gather_tests(Ranks& R, const std::vector<int64_t>& rows_list) {
  const int K = 3;
  std::mt19937_64 rng(0x6a7e0058ull);
  for (int64_t M : rows_list) {
    const int64_t T = M * K, n = kW;
    std::vector<void*> ye(WS), ew(WS), so(WS), shp(WS), sg(WS);
    std::vector<int> perm((size_t)T);
    for (int64_t i = 0; i < T; ++i) perm[(size_t)i] = (int)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int r = 0; r < WS; ++r) {
      std::vector<uint16_t> hy((size_t)T * n), hw((size_t)M * K), hs((size_t)M * n), hg((size_t)M);
      for (int64_t i = 0; i < T * n; ++i) hy[(size_t)i] = bf16_rne(act(r, i / n + 1000, i % n));
      for (int64_t i = 0; i < M * K; ++i) hw[(size_t)i] = bf16_rne(0.1f + 0.8f * (float)(rng() % 1000) / 1000.0f);
      for (int64_t i = 0; i < M * n; ++i) hs[(size_t)i] = bf16_rne(act(r, i / n + 5000, i % n));
      for (int64_t i = 0; i < M; ++i) hg[(size_t)i] = bf16_rne((float)((int)(rng() % 400) - 200) / 100.0f);
      CK(hipMalloc(&ye[r], hy.size() * 2)); CK(hipMemcpy(ye[r], hy.data(), hy.size() * 2, hipMemcpyHostToDevice));
      CK(hipMalloc(&ew[r], hw.size() * 2)); CK(hipMemcpy(ew[r], hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
      CK(hipMalloc(&so[r], perm.size() * 4)); CK(hipMemcpy(so[r], perm.data(), perm.size() * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&shp[r], hs.size() * 2)); CK(hipMemcpy(shp[r], hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
      CK(hipMalloc(&sg[r], hg.size() * 2)); CK(hipMemcpy(sg[r], hg.data(), hg.size() * 2, hipMemcpyHostToDevice));
    }
    auto gat = [&](int r) {
      R4dArGather G; std::memset(&G, 0, sizeof G);
      G.ye = (const unsigned short*)ye[r]; G.ew = (const unsigned short*)ew[r]; G.sorted = (const int*)so[r];
      G.sh = (const unsigned short*)shp[r]; G.sg = (const unsigned short*)sg[r];
      G.ye_ld = n; G.ew_ld = K; G.sh_ld = n; G.sg_ld = 1; G.top_k = K; G.T = (int)T; G.sig = 1; G.ew_f32 = 0;
      return G;
    };
    const long n16 = M * n / 8;
    long off[WS + 1];
    const long sh_max = r4d_ar_shard_table(n16, WS, nullptr, off);
    R4DArShardOff<WS> sh{};
    for (int q = 0; q <= WS; ++q) sh.off[q] = (int)off[q];
    sh.sh_max = (int)sh_max;
    const int drain = M * kW >= 131072 ? 1 : 3;
    std::vector<std::vector<uint16_t>> got(4, std::vector<uint16_t>((size_t)M * n));
    for (int arm = 0; arm < 4; ++arm) {   /* 0 exact gather two-shot, 1 gather + exact wide, 2 fp8 gather two-shot, 3 gather + fp8 wide */
      for (int r = 0; r < WS; ++r) CK(hipMemsetAsync(R.out[r], 0x5A, (size_t)M * n * 2, R.st[r]));
      R.sync();
      if (arm == 1 || arm == 3) {
        for (int r = 0; r < WS; ++r) k_gonly<<<8, 256, 0, R.st[r]>>>((int4*)R.in[r], gat(r), (int)n16, (int)(n / 8));
        R.sync();
        R.launch(arm == 3, M, arm == 3 ? 16 : 2, arm == 3 ? 256 : 1024);
        R.sync();
      } else {
        for (int r = 0; r < WS; ++r) {
          R4DArWidePeers<int4, WS - 1> peers{};
          for (int i = 0; i < WS - 1; ++i) { const int pr = i < r ? i : i + 1; peers.scratch[i] = (int4*)R.scratch[pr]; peers.flags[i] = R.flags[pr]; peers.rank[i] = pr; }
          if (arm == 0)
            r4d_ar_gather_twoshot_kernel<WS><<<2, 1024, 0, R.st[r]>>>((int4*)R.out[r], peers, (const int4*)R.scratch[r], R.flags[r], R.seq[r], sh,
                                                                     (int)(n / 8), (int)R.slot16, r, drain, 0, 1, gat(r), nullptr, nullptr);
          else
            r4d_ar_gather_twoshot_fp8_kernel<WS><<<16, 256, 0, R.st[r]>>>((int4*)R.out[r], peers, (const int4*)R.scratch[r], R.flags[r], R.seq[r], sh,
                                                                         (int)(n / 8), (int)R.slot16, r, drain, 0, 1, gat(r), nullptr, nullptr);
        }
        R.sync();
      }
      /* every rank's output must be the same bits; keep rank 0's after checking the others against it */
      std::vector<uint16_t> h((size_t)M * n);
      long rdiff = 0;
      for (int r = 0; r < WS; ++r) {
        CK(hipMemcpy(h.data(), R.out[r], h.size() * 2, hipMemcpyDeviceToHost));
        if (r == 0) got[(size_t)arm] = h; else rdiff += std::memcmp(h.data(), got[(size_t)arm].data(), h.size() * 2) != 0;
      }
      if (rdiff) check(false, "gather: the four ranks' outputs equal", "arm " + std::to_string(arm));
    }
    long dx = 0, df = 0;
    for (size_t i = 0; i < got[0].size(); ++i) { dx += got[0][i] != got[1][i]; df += got[2][i] != got[3][i]; }
    char t[200];
    std::snprintf(t, sizeof t, "M %lld x 5120, top 3 + shared arm: exact gather two-shot vs gather + exact AR %ld differ; fp8 gather two-shot vs gather + fp8 AR %ld differ", (long long)M, dx, df);
    check(dx == 0 && df == 0, "gather form: the fused launches == the gather + the all-reduce, byte for byte (both wires)", t);
    for (int r = 0; r < WS; ++r) { CK(hipFree(ye[r])); CK(hipFree(ew[r])); CK(hipFree(so[r])); CK(hipFree(shp[r])); CK(hipFree(sg[r])); }
  }
}

void time_tests(Ranks& R, const std::vector<int64_t>& rows_list, int iters) {
  struct Cfg { bool fp8; int nb, nt; };
  const Cfg cfgs[] = { { false, 2, 1024 }, { false, 4, 1024 }, { true, 2, 1024 }, { true, 4, 1024 }, { true, 4, 256 }, { true, 8, 256 } };
  std::vector<hipEvent_t> es((size_t)iters * WS), ee((size_t)iters * WS);
  for (auto& e : es) CK(hipEventCreate(&e));
  for (auto& e : ee) CK(hipEventCreate(&e));
  for (int64_t rows : rows_list) {
    std::printf("TIME rows %lld (%lld KB a rank's message):", (long long)rows, (long long)(rows * kW * 2 / 1024));
    for (const Cfg& c : cfgs) {
      for (int w = 0; w < 20; ++w) R.launch(c.fp8, rows, c.nb, c.nt);
      R.sync();
      for (int i = 0; i < iters; ++i) {
        for (int r = 0; r < WS; ++r) CK(hipEventRecord(es[(size_t)i * WS + r], R.st[r]));
        R.launch(c.fp8, rows, c.nb, c.nt);
        for (int r = 0; r < WS; ++r) CK(hipEventRecord(ee[(size_t)i * WS + r], R.st[r]));
      }
      R.sync();
      std::vector<float> d(iters);
      for (int i = 0; i < iters; ++i) {
        float lo = 1e30f, hi = -1e30f;
        for (int r = 0; r < WS; ++r) {
          float a, b;
          CK(hipEventElapsedTime(&a, es[0], es[(size_t)i * WS + r]));
          CK(hipEventElapsedTime(&b, es[0], ee[(size_t)i * WS + r]));
          lo = std::min(lo, a); hi = std::max(hi, b);
        }
        d[i] = (hi - lo) * 1000.0f;
      }
      float wall; CK(hipEventElapsedTime(&wall, es[0], ee[(size_t)(iters - 1) * WS + WS - 1]));
      std::sort(d.begin(), d.end());
      std::printf("  %s nb %d x %d p50 %.2f (p10 %.2f p90 %.2f, chain %.2f) us", c.fp8 ? "fp8" : "exact", c.nb, c.nt, d[iters / 2], d[iters / 10], d[iters * 9 / 10], wall * 1000.0f / iters);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  for (auto& e : es) CK(hipEventDestroy(e));
  for (auto& e : ee) CK(hipEventDestroy(e));
}

void phase_tests(Ranks& R, const std::vector<int64_t>& rows_list, int iters) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  struct Cfg { bool fp8; int nb, nt; };
  const Cfg cfgs[] = { { false, 2, 1024 }, { true, 2, 1024 }, { true, 4, 1024 }, { true, 4, 256 }, { true, 8, 256 }, { true, 16, 256 } };
  for (int64_t rows : rows_list) {
    const long n16 = rows * kW / 8;
    long off[WS + 1];
    const long sh_max = r4d_ar_shard_table(n16, WS, nullptr, off);
    R4DArShardOff<WS> sh{};
    for (int s = 0; s <= WS; ++s) sh.off[s] = (int)off[s];
    sh.sh_max = (int)sh_max;
    R4DArWidePeers<int4, WS - 1> peers{};
    for (int i = 0; i < WS - 1; ++i) { peers.scratch[i] = (int4*)R.scratch[i + 1]; peers.flags[i] = R.flags[i + 1]; peers.rank[i] = i + 1; }
    std::printf("PHASE rows %lld (one rank's loops, no handshakes):", (long long)rows);
    for (const Cfg& c : cfgs) {
      std::vector<float> d(iters);
      for (int i = -20; i < iters; ++i) {
        CK(hipEventRecord(a, R.st[0]));
        if (c.fp8) k_fp8_nohs<<<c.nb, c.nt, 0, R.st[0]>>>((const int4*)R.in[0], peers, (const int4*)R.scratch[0], (int4*)R.out[0], sh, (int)R.slot16, 0);
        else k_exact_nohs<<<c.nb, c.nt, 0, R.st[0]>>>((const int4*)R.in[0], peers, (const int4*)R.scratch[0], (int4*)R.out[0], sh, (int)R.slot16, 0);
        CK(hipEventRecord(b, R.st[0]));
        CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        if (i >= 0) d[i] = ms * 1000.0f;
      }
      std::sort(d.begin(), d.end());
      std::printf("  %s nb %d x %d %.2f us", c.fp8 ? "fp8" : "exact", c.nb, c.nt, d[iters / 2]);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
}

std::vector<int64_t> parse_list(const char* s) {
  std::vector<int64_t> v; std::string t(s); size_t p = 0;
  while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); v.push_back(std::atoll(t.substr(p, q - p).c_str())); p = q + 1; }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int64_t> trows = { 8, 16, 32, 40, 48 };
  const std::vector<int64_t> crows = { 1, 2, 3, 8, 16, 24, 32, 40, 48, 64 };
  int iters = 200; bool codec = true, tm = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) trows = parse_list(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-codec")) codec = false;
    else if (!std::strcmp(argv[i], "--no-time")) tm = false;
  }
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
  std::printf("arf: %s (%s), %d CUs; 4 ranks as 4 streams of this one card\n", pr.name, pr.gcnArchName, pr.multiProcessorCount);
  if (codec) codec_tests();
  Ranks R; R.init(64);
  ar_tests(R, crows);
  gather_tests(R, { 1, 8, 16, 32, 40, 48 });
  phase_tests(R, trows, iters);
  if (tm) time_tests(R, trows, iters);
  std::printf("arf: %d cases, %d failed\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
