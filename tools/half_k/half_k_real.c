/* half_k_real.c -- the C side of tools/half_k/crosscheck.py on real half-integer EXL3 tensors:
 *   1. W_hat from the trellis (exl3_trellis_decode, the oracle) -> <name>.what_c.bin, fp16 [16 kt][16 nt]
 *   2. the canonical pair plane (exl3_record_to_codes) decodes to the same W_hat (exl3_decode_what_rows, transposed)
 *   3. exl3r4d's fast relayout of that plane gives back the source trellis byte for byte, and its inverse the plane
 *   4. W (suh, both rotations, svh) row RMS, for a look at plausibility
 * usage: half_k_real DIR NAME [NAME ...] */
#include "exl3_decode.h"
#include "exl3_layout.h"
#include "exl3r4d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* slurp(const char* path, size_t want) {
    FILE* f = fopen(path, "rb");
    void* p = malloc(want);
    if (!f || !p || fread(p, 1, want, f) != want) { fprintf(stderr, "read %s\n", path); exit(2); }
    fclose(f);
    return p;
}

int main(int argc, char** argv) {
    int fails = 0;
    for (int a = 2; a < argc; ++a) {
        char path[1024];
        long long kt, nt, w;
        snprintf(path, sizeof path, "%s/%s.meta", argv[1], argv[a]);
        FILE* f = fopen(path, "r");
        if (!f || fscanf(f, "%lld %lld %lld", &kt, &nt, &w) != 3) { fprintf(stderr, "meta %s\n", path); return 2; }
        fclose(f);
        const int ka = (int)((w - 8) / 16), bits = EXL3_HALF(ka);
        const int64_t K = kt * 16, N = nt * 16;   /* exllamav3's [k, n]: in features, out features */
        if (w != 8 * exl3_k2(bits)) { fprintf(stderr, "%s: not half-integer\n", argv[a]); return 2; }
        snprintf(path, sizeof path, "%s/%s.trellis.bin", argv[1], argv[a]);
        uint16_t* tr = slurp(path, (size_t)(kt * nt * w * 2));
        snprintf(path, sizeof path, "%s/%s.suh.bin", argv[1], argv[a]);
        uint16_t* suh = slurp(path, (size_t)K * 2);
        snprintf(path, sizeof path, "%s/%s.svh.bin", argv[1], argv[a]);
        uint16_t* svh = slurp(path, (size_t)N * 2);
        /* 1 */
        uint16_t* wh = malloc((size_t)(K * N) * 2);
        if (exl3_trellis_decode(tr, kt, nt, bits, EXL3_CB_MUL1, wh) != EXL3_OK) { fprintf(stderr, "decode\n"); return 2; }
        snprintf(path, sizeof path, "%s/%s.what_c.bin", argv[1], argv[a]);
        f = fopen(path, "wb");
        fwrite(wh, 2, (size_t)(K * N), f);
        fclose(f);
        /* 2: radiance's [rows = n, cols = k] pair plane */
        const int64_t rb = exl3_codes_row_bytes(K, bits);
        uint8_t* codes = calloc((size_t)(N * rb), 1);
        int64_t bad2 = 0;
        if (exl3_record_to_codes(tr, N, K, bits, codes, rb) != EXL3_OK) { fprintf(stderr, "to_codes\n"); return 2; }
        float* out = malloc((size_t)(16 * K) * sizeof *out);
        Exl3View v;
        memset(&v, 0, sizeof v);
        v.rows = N; v.cols = K; v.bits = bits; v.codebook = EXL3_CB_MUL1; v.codes = codes; v.suh = suh; v.svh = svh;
        for (int64_t r0 = 0; r0 < N; r0 += 16) {
            exl3_decode_what_rows(&v, r0, 16, out);
            for (int64_t i = 0; i < 16; ++i)
                for (int64_t k = 0; k < K; ++k) bad2 += exl3_half_from_double(out[i * K + k]) != wh[k * N + r0 + i];
        }
        /* 3: the device library's stored form (u32 words, the same bytes as the int16 trellis) and back */
        uint32_t* rec = malloc((size_t)(kt * nt * w * 2));
        exl3r4d_kx_to_record(codes, rb, N, K, bits, rec);
        const int bad3 = memcmp(rec, tr, (size_t)(kt * nt * w * 2)) != 0;
        uint8_t* back = calloc((size_t)(N * rb), 1);
        exl3r4d_kx_from_record(rec, N, K, bits, back, rb);
        const int bad3b = memcmp(back, codes, (size_t)(N * rb)) != 0;
        /* 4: W rows' RMS (the weight as exllamav3 reconstructs it) */
        float* wrow = malloc((size_t)(128 * K) * sizeof *wrow);
        double ss = 0, mx = 0;
        exl3_decode_rows(&v, 0, 128, wrow);
        for (int64_t i = 0; i < 128 * K; ++i) { ss += (double)wrow[i] * wrow[i]; if (fabs(wrow[i]) > mx) mx = fabs(wrow[i]); }
        printf("%s: K=%d.5 [k %lld, n %lld] plane-vs-trellis mismatches %lld; device record %s; inverse %s; "
               "W rows 0..127 rms %.5f max %.4f\n", argv[a], ka, (long long)K, (long long)N, (long long)bad2,
               bad3 ? "DIFFERS" : "== source trellis", bad3b ? "DIFFERS" : "== plane", sqrt(ss / (128.0 * K)), mx);
        fails += bad2 != 0 || bad3 || bad3b;
        free(tr); free(suh); free(svh); free(wh); free(codes); free(out); free(rec); free(back); free(wrow);
    }
    printf("half_k_real: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
