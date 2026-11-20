/* r4d_ds41_ggml_layout.cpp -- the native-block GEMMs' weight layout: affine planes (a GGUF import)
 * relayouted into llama.cpp's blocks at load, through libquant's definition of them. Pure host.
 *
 * libref's hooks (libref/ref_ggml.cpp: ref_ggml_layout, ref_ggml_relayout, ref_ggml_unrelayout),
 * restated here because a plugin links nothing of another: the operand must carry exactly the
 * type's encoding with every plane selected in order; the stored form is u8 [rows, K / block *
 * block_bytes], the type's native blocks rows back to back (pack_of), tagged "ggml_<type>" like
 * libref's so a weight both libraries read is stored once. The unrelayout unpacks the blocks
 * back into the planes and writes the shared tables (an IQ grid). The row length is `K` for the
 * GEMMs and `n_embd` for the vocabulary ops. Only the types the kernels decode are served (pack
 * and dequant both written in libquant): q8_0, q5_K, q6_K, iq4_nl, iq2_xxs, iq1_s (r4d_ds41_ggml),
 * and mxfp4 (r4d_ds41_mxfp4: OCP MXFP4's planes, fp4_e2m1 codes and an e8m0 per 32, packed into
 * block_mxfp4 {e, qs[16]}, 17 bytes a block of 32). */
#include "r4d_plugin.h"

#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <cstring>
#include <vector>

namespace {

const lq::ggml::PlaneKind* gg_kind_named(const char* s) {
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t)) return k;
    }
    return nullptr;
}

/* Lever Q6IL (DRAFTER track 20.11.2026 round 44): type "q6_K_il16b" = q6_K's encoding and values, stored tile-interleaved (below). */
bool gg_il6_type(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return t && t->kind == RAD_P_STR && t->sval && !std::strcmp(t->sval, "q6_K_il16b");
}
/* Lever Q8IL (round 44): type "q8_0_il16b" = q8_0's encoding and values, stored tile-interleaved (q8il_pack below). */
bool gg_il8_type(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return t && t->kind == RAD_P_STR && t->sval && !std::strcmp(t->sval, "q8_0_il16b");
}

/* Lever Q5IL (round 45): type "q5_K_il16b" = q5_K's encoding and values, stored tile-interleaved (q5il_pack below). */
bool gg_il5_type(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return t && t->kind == RAD_P_STR && t->sval && !std::strcmp(t->sval, "q5_K_il16b");
}

const lq::ggml::PlaneKind* gg_kind_of(const RadParam* p, int n_p) {
    if (gg_il6_type(p, n_p)) return gg_kind_named("q6_K");
    if (gg_il8_type(p, n_p)) return gg_kind_named("q8_0");
    if (gg_il5_type(p, n_p)) return gg_kind_named("q5_K");
    const RadParam* t = rad_param_find(p, n_p, "type");
    return gg_kind_named(t && t->kind == RAD_P_STR ? t->sval : nullptr);
}

/* An integer parameter, a range read as its ceiling (libref's pint). */
int64_t gg_pint(const RadParam* p, int n_p, const char* key) {
    const RadParam* q = rad_param_find(p, n_p, key);
    if (!q) return 0;
    if (q->kind == RAD_P_RANGE) return (int64_t)q->ihi;
    return q->kind == RAD_P_INT ? (int64_t)q->ival : 0;
}

/* The selection must be exactly the type's encoding, every plane in order. */
int gg_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
             int n_planes, const lq::ggml::PlaneKind** out) {
    const lq::ggml::PlaneKind* k = gg_kind_of(p, n_p);
    if (operand != 1 || !k || !enc || !sel) return RAD_E_DTYPE;
    const RadEncoding want = lq::ggml::encoding_of(k->type);
    if (!rad_enc_equal(enc, &want) || n_planes != want.n_planes) return RAD_E_DTYPE;
    for (int i = 0; i < n_planes; ++i)
        if (sel[i] != i) return RAD_E_SHAPE;
    *out = k;
    return RAD_OK;
}

/* Plane i's row r, or null for a TABLE or per-tensor plane (pack and unpack do not touch those). */
uint8_t* gg_plane_row(const RadEncoding* enc, const RadTensor* planes, int i, int64_t r) {
    const RadEncPlane& pl = enc->plane[i];
    if (pl.kind == RAD_PLANE_TABLE || (pl.block[0] == 0 && pl.block[1] == 0)) return nullptr;
    const int64_t cols = planes[i].rank >= 2 ? planes[i].shape[1] : 0;
    return (uint8_t*)planes[i].data + r * rad_enc_row_bytes(pl.dtype, cols);
}

int64_t gg_rows(const RadTensor* planes) { return planes[0].rank >= 1 ? planes[0].shape[0] : 0; }

int64_t gg_row_len(const RadParam* p, int n_p) {
    const int64_t K = gg_pint(p, n_p, "K");
    return K > 0 ? K : gg_pint(p, n_p, "n_embd");
}

/* A stable tag per type (the RadLayout keeps the pointer). */
const char* gg_tag(const lq::ggml::PlaneKind* k) {
    static const char* const kTags[][2] = {
        { "q8_0", "ggml_q8_0" },     { "q5_K", "ggml_q5_K" },       { "q6_K", "ggml_q6_K" },
        { "iq4_nl", "ggml_iq4_nl" }, { "iq2_xxs", "ggml_iq2_xxs" }, { "iq1_s", "ggml_iq1_s" },
        { "mxfp4", "ggml_mxfp4" },
    };
    for (const auto& t : kTags)
        if (!std::strcmp(t[0], k->name)) return t[1];
    static thread_local char tag[32];
    snprintf(tag, sizeof tag, "ggml_%s", k->name);
    return tag;
}

/* Lever Q6IL: q6_K's native rows (210-byte blocks) <-> the tile-interleaved form IL-B (r4d_ds41_ggml.hip Wq6Il2): tile t (rows
 * 16 t .. 16 t + 15), slab s (half block s / 2, chunk c = s & 1) at (t S + s) 1696, S = K / 128; lane L = 16 h + r (r the tile's row):
 * ql bytes 64 c + 32 (q / 2) + 16 (q % 2) + 8 h + 0..7 at 256 q + 8 L (q 0..3), qh bytes 128 + 32 c + 16 q + 8 h + 0..7 at
 * 1024 + 256 q + 8 L (q 0..1), the 8 scales 192 + 8 c + 0..7 at 1536 + 8 r, d (208, 209) at 1664 + 2 r. The native bytes moved
 * (d in both slabs): 106 bytes a row a slab, rows a multiple of 16. */
constexpr int64_t kQ6Il = 1696, kQ6IlR = 106;
void q6il_pack(const uint8_t* nat, uint8_t* dst, int64_t rows, int64_t K) {
    const int64_t S = K / 128, rb = K / 256 * 210;
    for (int64_t t = 0; t < rows / 16; ++t)
        for (int64_t sl = 0; sl < S; ++sl) {
            uint8_t* d = dst + (t * S + sl) * kQ6Il;
            const int c = (int)(sl & 1);
            for (int L = 0; L < 32; ++L) {
                const int r = L & 15, h = L >> 4;
                const uint8_t* b = nat + (16 * t + r) * rb + (sl >> 1) * 210;
                for (int q = 0; q < 4; ++q) std::memcpy(d + 256 * q + 8 * L, b + 64 * c + 32 * (q >> 1) + 16 * (q & 1) + 8 * h, 8);
                for (int q = 0; q < 2; ++q) std::memcpy(d + 1024 + 256 * q + 8 * L, b + 128 + 32 * c + 16 * q + 8 * h, 8);
                if (h == 0) {
                    std::memcpy(d + 1536 + 8 * r, b + 192 + 8 * c, 8);
                    std::memcpy(d + 1664 + 2 * r, b + 208, 2);
                }
            }
        }
}
void q6il_unpack(const uint8_t* src, uint8_t* nat, int64_t rows, int64_t K) {
    const int64_t S = K / 128, rb = K / 256 * 210;
    for (int64_t t = 0; t < rows / 16; ++t)
        for (int64_t sl = 0; sl < S; ++sl) {
            const uint8_t* d = src + (t * S + sl) * kQ6Il;
            const int c = (int)(sl & 1);
            for (int L = 0; L < 32; ++L) {
                const int r = L & 15, h = L >> 4;
                uint8_t* b = nat + (16 * t + r) * rb + (sl >> 1) * 210;
                for (int q = 0; q < 4; ++q) std::memcpy(b + 64 * c + 32 * (q >> 1) + 16 * (q & 1) + 8 * h, d + 256 * q + 8 * L, 8);
                for (int q = 0; q < 2; ++q) std::memcpy(b + 128 + 32 * c + 16 * q + 8 * h, d + 1024 + 256 * q + 8 * L, 8);
                if (h == 0) {
                    std::memcpy(b + 192 + 8 * c, d + 1536 + 8 * r, 8);
                    if (c == 0) std::memcpy(b + 208, d + 1664 + 2 * r, 2);
                }
            }
        }
}

/* Lever Q8IL: q8_0's native rows (34-byte blocks) <-> its tile-interleaved form (r4d_ds41_ggml.hip Wq8Il): tile t, slab s (blocks 4 s ..
 * 4 s + 3) at (t S + s) 2176, S = K / 128; lane L = 16 h + r: block i's bytes 2 + 16 p + 8 h .. + 7 at 512 i + 256 p + 8 L, d (bytes 0, 1)
 * at 2048 + 8 r + 2 i. The native bytes moved, none duplicated: rows a multiple of 16, K of 128. */
constexpr int64_t kQ8Il = 2176;
void q8il_pack(const uint8_t* nat, uint8_t* dst, int64_t rows, int64_t K) {
    const int64_t S = K / 128, rb = K / 32 * 34;
    for (int64_t t = 0; t < rows / 16; ++t)
        for (int64_t sl = 0; sl < S; ++sl) {
            uint8_t* d = dst + (t * S + sl) * kQ8Il;
            for (int L = 0; L < 32; ++L) {
                const int r = L & 15, h = L >> 4;
                for (int i = 0; i < 4; ++i) {
                    const uint8_t* b = nat + (16 * t + r) * rb + (4 * sl + i) * 34;
                    for (int p = 0; p < 2; ++p) std::memcpy(d + 512 * i + 256 * p + 8 * L, b + 2 + 16 * p + 8 * h, 8);
                    if (h == 0) std::memcpy(d + 2048 + 8 * r + 2 * i, b, 2);
                }
            }
        }
}
void q8il_unpack(const uint8_t* src, uint8_t* nat, int64_t rows, int64_t K) {
    const int64_t S = K / 128, rb = K / 32 * 34;
    for (int64_t t = 0; t < rows / 16; ++t)
        for (int64_t sl = 0; sl < S; ++sl) {
            const uint8_t* d = src + (t * S + sl) * kQ8Il;
            for (int L = 0; L < 32; ++L) {
                const int r = L & 15, h = L >> 4;
                for (int i = 0; i < 4; ++i) {
                    uint8_t* b = nat + (16 * t + r) * rb + (4 * sl + i) * 34;
                    for (int p = 0; p < 2; ++p) std::memcpy(b + 2 + 16 * p + 8 * h, d + 512 * i + 256 * p + 8 * L, 8);
                    if (h == 0) std::memcpy(b, d + 2048 + 8 * r + 2 * i, 2);
                }
            }
        }
}

/* Lever Q5IL (round 45): q5_K's native rows (176-byte blocks) <-> its tile-interleaved form (r4d_ds41_ggml.hip Wq5Il): tile t, block b
 * at (t K / 256 + b) 2816; row r's block bytes 0..15 (d, dmin, scales) at 16 r; lane L = 16 h + r: qh bytes 16 i + 8 h .. + 7 (i 0..1) at
 * 256 + 256 i + 8 L, slab c's qs bytes 64 c + 32 (i >> 1) + 16 (i & 1) + 8 h .. + 7 (block bytes 48 + that, i 0..3) at 768 + 1024 c + 256 i
 * + 8 L. The native bytes moved, none duplicated: rows a multiple of 16, K of 256. */
constexpr int64_t kQ5Il = 2816;
void q5il_move(uint8_t* nat, uint8_t* il, int64_t rows, int64_t K, bool to_il) {
    const int64_t NB = K / 256, rb = NB * 176;
    auto mv = [&](uint8_t* a, uint8_t* n, size_t len) { if (to_il) std::memcpy(a, n, len); else std::memcpy(n, a, len); };
    for (int64_t t = 0; t < rows / 16; ++t)
        for (int64_t b = 0; b < NB; ++b) {
            uint8_t* d = il + (t * NB + b) * kQ5Il;
            for (int L = 0; L < 32; ++L) {
                const int r = L & 15, h = L >> 4;
                uint8_t* blk = nat + (16 * t + r) * rb + b * 176;
                if (h == 0) mv(d + 16 * r, blk, 16);
                for (int i = 0; i < 2; ++i) mv(d + 256 + 256 * i + 8 * L, blk + 16 + 16 * i + 8 * h, 8);
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 4; ++i) mv(d + 768 + 1024 * c + 256 * i + 8 * L, blk + 48 + 64 * c + 32 * (i >> 1) + 16 * (i & 1) + 8 * h, 8);
            }
        }
}
void q5il_pack(const uint8_t* nat, uint8_t* dst, int64_t rows, int64_t K) { q5il_move(const_cast<uint8_t*>(nat), dst, rows, K, true); }
void q5il_unpack(const uint8_t* src, uint8_t* nat, int64_t rows, int64_t K) { q5il_move(nat, const_cast<uint8_t*>(src), rows, K, false); }

}  // namespace

extern "C" int r4d_q5_K_il16b_pack(const void* native, void* dst, int64_t rows, int64_t K) {
    if (!native || !dst || rows <= 0 || rows % 16 || K <= 0 || K % 256) return RAD_E_SHAPE;
    q5il_pack((const uint8_t*)native, (uint8_t*)dst, rows, K);
    return RAD_OK;
}
extern "C" int r4d_q5_K_il16b_unpack(const void* src, void* native, int64_t rows, int64_t K) {
    if (!native || !src || rows <= 0 || rows % 16 || K <= 0 || K % 256) return RAD_E_SHAPE;
    q5il_unpack((const uint8_t*)src, (uint8_t*)native, rows, K);
    return RAD_OK;
}
extern "C" int r4d_q8_0_il16b_pack(const void* native, void* dst, int64_t rows, int64_t K) {
    if (!native || !dst || rows <= 0 || rows % 16 || K <= 0 || K % 128) return RAD_E_SHAPE;
    q8il_pack((const uint8_t*)native, (uint8_t*)dst, rows, K);
    return RAD_OK;
}
extern "C" int r4d_q8_0_il16b_unpack(const void* src, void* native, int64_t rows, int64_t K) {
    if (!native || !src || rows <= 0 || rows % 16 || K <= 0 || K % 128) return RAD_E_SHAPE;
    q8il_unpack((const uint8_t*)src, (uint8_t*)native, rows, K);
    return RAD_OK;
}

/* The selftest's (and tools') access to the move: native q6_K rows <-> IL-B, rows a multiple of 16, K a multiple of 256. */
extern "C" int r4d_q6k_il16b_pack(const void* native, void* dst, int64_t rows, int64_t K) {
    if (!native || !dst || rows <= 0 || rows % 16 || K <= 0 || K % 256) return RAD_E_SHAPE;
    q6il_pack((const uint8_t*)native, (uint8_t*)dst, rows, K);
    return RAD_OK;
}
extern "C" int r4d_q6k_il16b_unpack(const void* src, void* native, int64_t rows, int64_t K) {
    if (!native || !src || rows <= 0 || rows % 16 || K <= 0 || K % 256) return RAD_E_SHAPE;
    q6il_unpack((const uint8_t*)src, (uint8_t*)native, rows, K);
    return RAD_OK;
}

extern "C" int r4d_ds41_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                    const int* sel, const RadTensor* planes, int n_planes,
                                    RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    if (K <= 0 || K % k->block || rows <= 0) return RAD_E_SHAPE;
    std::memset(out, 0, sizeof *out);
    out->tag = gg_tag(k);
    out->dtype = RAD_U8;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = K / k->block * (int64_t)k->block_bytes;
    out->align = 64;
    if (gg_il6_type(p, n_p)) {   /* lever Q6IL: [rows, 106 K / 128], whole 16-row tiles */
        if (rows % 16) return RAD_E_SHAPE;
        out->tag = "r4d_q6k_il16b";
        out->shape[1] = K / 128 * kQ6IlR;
    }
    if (gg_il8_type(p, n_p)) {   /* lever Q8IL: the native size, whole 16-row tiles and 128-value slabs */
        if (rows % 16 || K % 128) return RAD_E_SHAPE;
        out->tag = "r4d_q8_0_il16b";
    }
    if (gg_il5_type(p, n_p)) {   /* lever Q5IL: the native size, whole 16-row tiles */
        if (rows % 16) return RAD_E_SHAPE;
        out->tag = "r4d_q5k_il16b";
    }
    out->bytes = out->shape[0] * out->shape[1];
    return RAD_OK;
}

extern "C" int r4d_ds41_ggml_relayout(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, void* dst,
                                      int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    const bool il6 = gg_il6_type(p, n_p), il8 = gg_il8_type(p, n_p), il5 = gg_il5_type(p, n_p), il = il6 || il8 || il5;
    if (K <= 0 || K % k->block || rows <= 0 || (il && rows % 16) || (il8 && K % 128) || dst_bytes < rows * (il6 ? K / 128 * kQ6IlR : rb))
        return RAD_E_SHAPE;
    const lq::ggml::PackFn pack = lq::ggml::pack_of(k->type);
    std::vector<uint8_t> nat(il ? (size_t)(rows * rb) : 0);   /* levers Q6IL / Q8IL: the native rows first, then moved */
    uint8_t* const out = il ? nat.data() : (uint8_t*)dst;
    std::vector<const uint8_t*> row((size_t)n_planes);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = gg_plane_row(enc, planes, i, r);
        pack(row.data(), nb, out + r * rb);
    }
    if (il6) q6il_pack(nat.data(), (uint8_t*)dst, rows, K);
    if (il8) q8il_pack(nat.data(), (uint8_t*)dst, rows, K);
    if (il5) q5il_pack(nat.data(), (uint8_t*)dst, rows, K);
    return RAD_OK;
}

extern "C" int r4d_ds41_ggml_unrelayout(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel, const void* src,
                                        int64_t src_bytes, const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = gg_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    const bool il6 = gg_il6_type(p, n_p), il8 = gg_il8_type(p, n_p), il5 = gg_il5_type(p, n_p), il = il6 || il8 || il5;
    if (K <= 0 || K % k->block || rows <= 0 || (il && rows % 16) || (il8 && K % 128) || src_bytes < rows * (il6 ? K / 128 * kQ6IlR : rb))
        return RAD_E_SHAPE;
    std::vector<uint8_t> nat(il ? (size_t)(rows * rb) : 0);   /* levers Q6IL / Q8IL: moved back to the native rows first */
    if (il6) q6il_unpack((const uint8_t*)src, nat.data(), rows, K);
    if (il8) q8il_unpack((const uint8_t*)src, nat.data(), rows, K);
    if (il5) q5il_unpack((const uint8_t*)src, nat.data(), rows, K);
    const uint8_t* const in = il ? nat.data() : (const uint8_t*)src;
    std::vector<uint8_t*> row((size_t)n_planes);
    std::vector<void*> all((size_t)n_planes);
    for (int i = 0; i < n_planes; ++i) all[(size_t)i] = planes[i].data;
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = gg_plane_row(enc, planes, i, r);
        k->unpack(in + r * rb, nb, row.data());
    }
    lq::ggml::write_shared(k->type, *enc, all.data());
    return RAD_OK;
}

/* gemm_nt_ggml_pair (lever 1U-GQ, ds41-int11): its weights are operands 1 and 3, each the type's native blocks at K, as
 * gemm_nt_ggml (operand 1) stores them -- the same tag and bytes, so one stored form serves both ops. */
extern "C" int r4d_ds41_ggml_pair_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                         const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    if (operand != 1 && operand != 3) return RAD_E_DTYPE;
    return r4d_ds41_ggml_layout(p, n_p, 1, enc, sel, planes, n_planes, out);
}
extern "C" int r4d_ds41_ggml_pair_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                           const int* sel, const RadTensor* planes, int n_planes, void* dst,
                                           int64_t dst_bytes) {
    if (operand != 1 && operand != 3) return RAD_E_DTYPE;
    return r4d_ds41_ggml_relayout(p, n_p, 1, enc, sel, planes, n_planes, dst, dst_bytes);
}
extern "C" int r4d_ds41_ggml_pair_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                             const int* sel, const void* src, int64_t src_bytes,
                                             const RadTensor* planes, int n_planes) {
    if (operand != 1 && operand != 3) return RAD_E_DTYPE;
    return r4d_ds41_ggml_unrelayout(p, n_p, 1, enc, sel, src, src_bytes, planes, n_planes);
}

/* moe_gemm_mxil (lever WIL, DRAFTER track 19.11.2026 round 27): GGUF MXFP4 blocks re-arranged into 16-row x 1-block
 * tiles of 272 bytes -- tile (row group g, block b) at (g nb + b) 272 holds, for lane L = r + 16 h of the WMMA form
 * (r the group's row, h 0 / 1), the block's code bytes qs[4 h .. 4 h + 3] and qs[8 + 4 h .. 8 + 4 h + 3] at 8 L, and the
 * 16 rows' E8M0 exponents at 256 + r (r4d_ds41_mxfp4_mr.hip, xm_wm_kernel's IL note). The same bytes as the native
 * blocks, moved: the stored size, the planes it is made from and the inverse are the native form's; the rows (an
 * expert's N, a table entry's weight) must be whole groups of 16. Tagged apart so no reader of native blocks gets it. */
namespace {
constexpr int64_t kIlTile = 272;
void il_pack(const uint8_t* blk, uint8_t* dst, int64_t rows, int64_t nb) {
    for (int64_t g = 0; g < rows / 16; ++g)
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t* t = dst + (g * nb + b) * kIlTile;
            for (int64_t r = 0; r < 16; ++r) {
                const uint8_t* q = blk + ((16 * g + r) * nb + b) * 17;
                for (int h = 0; h < 2; ++h) {
                    std::memcpy(t + 8 * (r + 16 * h), q + 1 + 4 * h, 4);
                    std::memcpy(t + 8 * (r + 16 * h) + 4, q + 9 + 4 * h, 4);
                }
                t[256 + r] = q[0];
            }
        }
}
void il_unpack(const uint8_t* src, uint8_t* blk, int64_t rows, int64_t nb) {
    for (int64_t g = 0; g < rows / 16; ++g)
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* t = src + (g * nb + b) * kIlTile;
            for (int64_t r = 0; r < 16; ++r) {
                uint8_t* q = blk + ((16 * g + r) * nb + b) * 17;
                for (int h = 0; h < 2; ++h) {
                    std::memcpy(q + 1 + 4 * h, t + 8 * (r + 16 * h), 4);
                    std::memcpy(q + 9 + 4 * h, t + 8 * (r + 16 * h) + 4, 4);
                }
                q[0] = t[256 + r];
            }
        }
}
bool il_type_ok(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return t && t->kind == RAD_P_STR && t->sval && !std::strcmp(t->sval, "mxfp4");
}
}  // namespace

extern "C" int r4d_ds41_mxil_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                    const RadTensor* planes, int n_planes, RadLayout* out) {
    if (!il_type_ok(p, n_p)) return RAD_E_DTYPE;
    const int rc = r4d_ds41_ggml_layout(p, n_p, operand, enc, sel, planes, n_planes, out);
    if (rc != RAD_OK) return rc;
    if (out->shape[0] % 16) return RAD_E_SHAPE;
    out->tag = "r4d_mxfp4_il16";
    return RAD_OK;
}

extern "C" int r4d_ds41_mxil_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    if (!il_type_ok(p, n_p)) return RAD_E_DTYPE;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    if (K <= 0 || K % 32 || rows <= 0 || rows % 16) return RAD_E_SHAPE;
    const int64_t nb = K / 32;
    if (dst_bytes < rows * nb * 17) return RAD_E_SHAPE;
    std::vector<uint8_t> blk((size_t)(rows * nb * 17));
    const int rc = r4d_ds41_ggml_relayout(p, n_p, operand, enc, sel, planes, n_planes, blk.data(), (int64_t)blk.size());
    if (rc != RAD_OK) return rc;
    il_pack(blk.data(), (uint8_t*)dst, rows, nb);
    return RAD_OK;
}

extern "C" int r4d_ds41_mxil_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                                        const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes) {
    if (!il_type_ok(p, n_p)) return RAD_E_DTYPE;
    const int64_t K = gg_row_len(p, n_p), rows = gg_rows(planes);
    if (K <= 0 || K % 32 || rows <= 0 || rows % 16) return RAD_E_SHAPE;
    const int64_t nb = K / 32;
    if (src_bytes < rows * nb * 17) return RAD_E_SHAPE;
    std::vector<uint8_t> blk((size_t)(rows * nb * 17));
    il_unpack((const uint8_t*)src, blk.data(), rows, nb);
    return r4d_ds41_ggml_unrelayout(p, n_p, operand, enc, sel, blk.data(), (int64_t)blk.size(), planes, n_planes);
}
