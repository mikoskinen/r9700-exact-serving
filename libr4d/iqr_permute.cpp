/* iqr-permute -- rewrite a sealed IQR container in the GEMM's lane order, so a file read is a
 * copy and the host pool can hold it. Every weight's planes are read, the CPU walk
 * (r4d_rad_relayout_iqr) writes the stored form, and the new container carries that as a
 * "stored" encoding the loader reads directly. The scale plane is stored the same way.
 *
 * usage: iqr-permute <old.rad> <new.rad>
 */
#include "core/format/radfile.h"
#include "libr4d/r4d_iqr.h"
#include "abi/rad_abi.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace r4d_iqr;
using namespace rad;

typedef int (*LayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                        int, RadLayout*);
typedef int (*RelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                          int, void*, int64_t);

static inline uint64_t get_bits_lane(const uint8_t* row, long long o, int n) {
    const uint8_t* b = row + (o >> 3);
    const int sh = (int)(o & 7);
    const int bytes = (sh + n + 7) >> 3;
    uint64_t w = 0;
    for (int i = 0; i < bytes; ++i) w |= (uint64_t)b[i] << (8 * i);
    return (w >> sh) & ((1ull << n) - 1ull);
}

static bool is_iqr_scheme(const RadEncoding& e) {
    return rad_enc_is(&e, "trellis");
}

static int verify_all(RadFile& old_file, const char* newpath, LayoutFn layout_fn,
                      RelayoutFn relayout_fn) {
    RadFile new_file;
    int rc2 = new_file.open(newpath);
    if (rc2 != RAD_OK) {
        fprintf(stderr, "verify open %s: %d\n", newpath, rc2);
        return 1;
    }
    int64_t n_checked = 0, n_bad = 0;
    for (int64_t i = 0; i < old_file.entry_count(); ++i) {
        const RadFileEntry& e = old_file.entry(i);
        const RadEncoding& enc = old_file.encoding(e);
        if (enc.n_planes <= 0 || !is_iqr_scheme(enc) || e.rank != 2) {
            const RadFileEntry* ne = new_file.find(old_file.str(e.name));
            if (!ne) { ++n_bad; continue; }
            bool bad = false;
            for (int j = 0; j < enc.n_planes; ++j) {
                const RadFilePlane& op = old_file.plane(e, j);
                const RadFilePlane& np = new_file.plane(*ne, j);
                if (op.bytes != np.bytes ||
                    std::memcmp(old_file.data(op), new_file.data(np), (size_t)op.bytes) != 0) {
                    bad = true;
                    break;
                }
            }
            if (bad) ++n_bad;
            ++n_checked;
            continue;
        }
        const int64_t n = e.shape[0], k = e.shape[1];
        const char* dt = (enc.transform[0] &&
                          !strncmp(enc.transform, "fwht128", RAD_ENC_STR)) ? "i8a8h" : "";
        RadParam p[] = {
            RAD_INT("N", n), RAD_INT("K", k), RAD_INT("group", 128), RAD_STR("dtype", dt),
        };
        int jc = -1, js = -1, jf = -1, j2 = -1;
        for (int j = 0; j < enc.n_planes; ++j) {
            if (!strncmp(enc.plane[j].role, "codes", RAD_ENC_STR)) jc = j;
            else if (!strncmp(enc.plane[j].role, "scale", RAD_ENC_STR)) js = j;
            else if (!strncmp(enc.plane[j].role, "scale.1", RAD_ENC_STR)) j2 = j;
            else if (!strncmp(enc.plane[j].role, "scale.col", RAD_ENC_STR)) jf = j;
        }
        if (js < 0) js = jc;
        RadTensor pl[RAD_ENC_MAX_PLANES];
        int sel[RAD_ENC_MAX_PLANES];
        for (int j = 0; j < enc.n_planes; ++j) {
            const RadFilePlane& plane = old_file.plane(e, j);
            int64_t pr = 1, pc = 1;
            rad_enc_plane_dims(&enc.plane[j], n, k, &pr, &pc);
            pl[j].data = (void*)old_file.data(plane);
            pl[j].dtype = enc.plane[j].dtype;
            pl[j].rank = 2;
            pl[j].shape[0] = pr;
            pl[j].shape[1] = pc;
            pl[j].stride[0] = pc;
            pl[j].stride[1] = 1;
            sel[j] = j;
        }
        RadLayout out;
        int rc = layout_fn(p, 4, 2, &enc, sel, pl, enc.n_planes, &out);
        if (rc != RAD_OK) { ++n_bad; continue; }
        const int64_t stored_bytes = out.bytes;
        std::vector<uint8_t> dst((size_t)stored_bytes);
        rc = relayout_fn(p, 4, 2, &enc, sel, pl, enc.n_planes, dst.data(), stored_bytes);
        if (rc != RAD_OK) { ++n_bad; continue; }
        int64_t pr = 1, pc = 1;
        rad_enc_plane_dims(&enc.plane[jc], n, k, &pr, &pc);
        const int bits = rad_dtype_bits(enc.plane[jc].dtype);
        const int64_t ncode = pr * pc;
        std::vector<uint8_t> udst((size_t)ncode);
        for (int64_t q = 0; q < ncode; ++q)
            udst[(size_t)q] = (uint8_t)get_bits_lane(dst.data(), q * bits, bits);
        int sel_s[3];
        int ns = 0;
        sel_s[ns++] = js;
        if (j2 >= 0) sel_s[ns++] = j2;
        if (jf >= 0) sel_s[ns++] = jf;
        RadTensor pl_s[3];
        for (int t = 0; t < ns; ++t) pl_s[t] = pl[sel_s[t]];
        RadLayout sout;
        rc = layout_fn(p, 4, 3, &enc, sel_s, pl_s, ns, &sout);
        if (rc != RAD_OK) { ++n_bad; continue; }
        const int64_t sbytes = sout.bytes;
        std::vector<uint8_t> sdst((size_t)sbytes);
        rc = relayout_fn(p, 4, 3, &enc, sel_s, pl_s, ns, sdst.data(), sbytes);
        if (rc != RAD_OK) { ++n_bad; continue; }
        const RadFileEntry* ne = new_file.find(old_file.str(e.name));
        if (!ne || ne->n_planes < 2) { ++n_bad; continue; }
        const RadFilePlane& np0 = new_file.plane(*ne, 0);
        const RadFilePlane& np1 = new_file.plane(*ne, 1);
        if ((int64_t)np0.bytes != ncode ||
            std::memcmp(udst.data(), new_file.data(np0), (size_t)ncode) != 0 ||
            (int64_t)np1.bytes != sbytes ||
            std::memcmp(sdst.data(), new_file.data(np1), (size_t)sbytes) != 0) {
            ++n_bad;
        }
        ++n_checked;
    }
    fprintf(stderr, "I verify: %lld checked, %lld bad\n", (long long)n_checked, (long long)n_bad);
    return n_bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <old.rad> <new.rad> [--verify]\n", argv[0]);
        return 2;
    }
    RadFile old_file;
    int rc = old_file.open(argv[1]);
    if (rc != RAD_OK) {
        fprintf(stderr, "open %s: %d\n", argv[1], rc);
        return 1;
    }
    const char* home = std::getenv("RADIANCE_HOME");
    std::string libpath = home ? std::string(home) + "/kernels/libr4d.so" : "libr4d.so";
    void* lib = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen %s: %s\n", libpath.c_str(), dlerror());
        return 1;
    }
    LayoutFn layout_fn = (LayoutFn)dlsym(lib, "r4d_rad_layout_iqr");
    RelayoutFn relayout_fn = (RelayoutFn)dlsym(lib, "r4d_rad_relayout_iqr");
    if (!layout_fn || !relayout_fn) {
        fprintf(stderr, "missing layout/relayout symbols in %s\n", libpath.c_str());
        return 1;
    }
    if (argc == 4 && std::strcmp(argv[3], "--verify-only") == 0)
        return verify_all(old_file, argv[2], layout_fn, relayout_fn);
    RadWriter writer;
    rc = writer.begin(argv[2]);
    if (rc != RAD_OK) {
        fprintf(stderr, "begin %s: %d\n", argv[2], rc);
        return 1;
    }
    const RadFileHeader& hdr = old_file.header();
    writer.set_arch(old_file.str(hdr.arch_id));
    writer.set_model_name(old_file.str(hdr.model_name));
    writer.set_quant(old_file.str(hdr.quant));
    writer.set_recipe(old_file.str(hdr.recipe));
    writer.set_created_by(old_file.str(hdr.created_by));
    for (int64_t i = 0; i < old_file.meta_count(); ++i) {
        const RadFileKV& kv = old_file.meta_at(i);
        const char* key = old_file.str(kv.key);
        if (kv.type == RAD_P_INT) writer.meta_i(key, kv.v.i);
        else if (kv.type == RAD_KV_F64) writer.meta_f(key, kv.v.f);
        else writer.meta_s(key, old_file.str(kv.v.s));
    }
    {
        std::vector<uint8_t> vocab((const uint8_t*)old_file.base() + hdr.vocab_off,
                                   (const uint8_t*)old_file.base() + hdr.vocab_off + hdr.vocab_bytes);
        RadVocabHeader* vh = (RadVocabHeader*)vocab.data();
        if (vh->tok_text_off)  vh->tok_text_off  -= hdr.vocab_off;
        if (vh->tok_score_off) vh->tok_score_off -= hdr.vocab_off;
        if (vh->tok_type_off)  vh->tok_type_off  -= hdr.vocab_off;
        if (vh->merge_off)     vh->merge_off     -= hdr.vocab_off;
        if (vh->step_off)      vh->step_off      -= hdr.vocab_off;
        if (vh->tok_text_off) {
            rad_stroff* st = (rad_stroff*)(vocab.data() + vh->tok_text_off);
            for (uint32_t i = 0; i < vh->n_tokens; ++i) {
                const char* s = (const char*)(old_file.base() + st[i]);
                st[i] = (rad_stroff)writer.intern_string(s);
            }
        }
        if (vh->step_off) {
            RadVocabStep* steps = (RadVocabStep*)(vocab.data() + vh->step_off);
            for (uint32_t i = 0; i < vh->n_steps; ++i) {
                steps[i].arg0 = (rad_stroff)writer.intern_string(
                    (const char*)(old_file.base() + steps[i].arg0));
                steps[i].arg1 = (rad_stroff)writer.intern_string(
                    (const char*)(old_file.base() + steps[i].arg1));
            }
        }
        if (vh->chat_template) {
            vh->chat_template = (rad_stroff)writer.intern_string(
                (const char*)(old_file.base() + vh->chat_template));
        }
        rc = writer.set_vocab_section(vocab.data(), (int64_t)hdr.vocab_bytes);
        if (rc != RAD_OK) {
            fprintf(stderr, "set_vocab_section %d\n", rc);
            return 1;
        }
    }
    if (hdr.reserved[RAD_HDR_AUX_VOCAB_BYTES]) {
        uint64_t aoff = hdr.reserved[RAD_HDR_AUX_VOCAB_OFF];
        uint64_t abytes = hdr.reserved[RAD_HDR_AUX_VOCAB_BYTES];
        std::vector<uint8_t> aux((const uint8_t*)old_file.base() + aoff,
                                 (const uint8_t*)old_file.base() + aoff + abytes);
        RadVocabHeader* ah = (RadVocabHeader*)aux.data();
        if (ah->tok_text_off)  ah->tok_text_off  -= aoff;
        if (ah->tok_score_off) ah->tok_score_off -= aoff;
        if (ah->tok_type_off)  ah->tok_type_off  -= aoff;
        if (ah->merge_off)     ah->merge_off     -= aoff;
        if (ah->step_off)      ah->step_off      -= aoff;
        rc = writer.set_aux_vocab_section(aux.data(), (int64_t)abytes);
        if (rc != RAD_OK) {
            fprintf(stderr, "set_aux_vocab_section %d\n", rc);
            return 1;
        }
    }
    for (int64_t i = 0; i < old_file.profile_count(); ++i) {
        const RadFileProfile& p = old_file.profile()[i];
        writer.add_profile(p.layer, p.expert, p.share);
    }
    int64_t n_weights = 0, n_stored = 0;
    for (int64_t i = 0; i < old_file.entry_count(); ++i) {
        const RadFileEntry& e = old_file.entry(i);
        const RadEncoding& enc = old_file.encoding(e);
        if (enc.n_planes <= 0 || !is_iqr_scheme(enc) || e.rank != 2 || e.expert < 0) {
            RadWriter::Weight w{};
            w.name = old_file.str(e.name);
            w.quantizer = old_file.str(e.quantizer);
            w.options = old_file.str(e.options);
            w.enc = enc;
            w.rank = e.rank;
            for (int d = 0; d < RAD_MAX_RANK; ++d) w.shape[d] = e.shape[d];
            w.layer = e.layer;
            w.expert = e.expert;
            w.slot = e.slot;
            const void* planes[RAD_ENC_MAX_PLANES];
            for (int j = 0; j < enc.n_planes; ++j)
                planes[j] = (void*)old_file.data(old_file.plane(e, j));
            rc = writer.add_weight(w, planes);
            if (rc < 0) {
                fprintf(stderr, "%s: copy add_weight %d\n", old_file.str(e.name), rc);
                return 1;
            }
            continue;
        }
        const int64_t n = e.shape[0], k = e.shape[1];
        const char* dt = (enc.transform[0] &&
                          !strncmp(enc.transform, "fwht128", RAD_ENC_STR)) ? "i8a8h" : "";
        RadParam p[] = {
            RAD_INT("N", n), RAD_INT("K", k), RAD_INT("group", 128), RAD_STR("dtype", dt),
        };
        RadTensor pl[RAD_ENC_MAX_PLANES];
        int sel[RAD_ENC_MAX_PLANES];
        for (int j = 0; j < enc.n_planes; ++j) {
            const RadFilePlane& plane = old_file.plane(e, j);
            int64_t pr = 1, pc = 1;
            rad_enc_plane_dims(&enc.plane[j], n, k, &pr, &pc);
            pl[j].data = (void*)old_file.data(plane);
            pl[j].dtype = enc.plane[j].dtype;
            pl[j].rank = 2;
            pl[j].shape[0] = pr;
            pl[j].shape[1] = pc;
            pl[j].stride[0] = pc;
            pl[j].stride[1] = 1;
            sel[j] = j;
        }
        RadLayout out;
        rc = layout_fn(p, 4, 2, &enc, sel, pl, enc.n_planes, &out);
        if (rc != RAD_OK) {
            fprintf(stderr, "%s: layout failed %d\n", old_file.str(e.name), rc);
            return 1;
        }
        const int64_t stored_bytes = out.bytes;
        std::vector<uint8_t> dst((size_t)stored_bytes);
        rc = relayout_fn(p, 4, 2, &enc, sel, pl, enc.n_planes, dst.data(), stored_bytes);
        if (rc != RAD_OK) {
            fprintf(stderr, "%s: relayout failed %d\n", old_file.str(e.name), rc);
            return 1;
        }
        int jc = -1, js = -1, jf = -1, j2 = -1;
        for (int j = 0; j < enc.n_planes; ++j) {
            if (!strncmp(enc.plane[j].role, "codes", RAD_ENC_STR)) jc = j;
            else if (!strncmp(enc.plane[j].role, "scale", RAD_ENC_STR)) js = j;
            else if (!strncmp(enc.plane[j].role, "scale.1", RAD_ENC_STR)) j2 = j;
            else if (!strncmp(enc.plane[j].role, "scale.col", RAD_ENC_STR)) jf = j;
        }
        if (js < 0) js = jc;
        int sel_s[3];
        int ns = 0;
        sel_s[ns++] = js;
        if (j2 >= 0) sel_s[ns++] = j2;
        if (jf >= 0) sel_s[ns++] = jf;
        RadTensor pl_s[3];
        for (int t = 0; t < ns; ++t) pl_s[t] = pl[sel_s[t]];
        RadLayout sout;
        rc = layout_fn(p, 4, 3, &enc, sel_s, pl_s, ns, &sout);
        if (rc != RAD_OK) {
            fprintf(stderr, "%s: scale layout failed %d\n", old_file.str(e.name), rc);
            return 1;
        }
        const int64_t sbytes = sout.bytes;
        std::vector<uint8_t> sdst((size_t)sbytes);
        rc = relayout_fn(p, 4, 3, &enc, sel_s, pl_s, ns, sdst.data(), sbytes);
        if (rc != RAD_OK) {
            fprintf(stderr, "%s: scale relayout failed %d\n", old_file.str(e.name), rc);
            return 1;
        }
        const uint32_t form_word = (uint32_t)(out.dtype & kWordMask);
        std::vector<uint8_t> dummy((size_t)n, 0);
        uint32_t word_bytes = form_word;
        RadEncoding stored_enc;
        rad_enc_clear(&stored_enc);
        rad_enc_copy_str(stored_enc.scheme, "stored");
        if (enc.transform[0]) rad_enc_copy_str(stored_enc.transform, enc.transform);
        rad_enc_add_plane(&stored_enc, "codes", RAD_U8, 1, k);
        rad_enc_add_table(&stored_enc, "lane", RAD_U8, 1, stored_bytes);
        rad_enc_add_table(&stored_enc, "scale", RAD_U8, 1, sbytes);
        rad_enc_add_table(&stored_enc, "form", RAD_U32, 1, 1);
        stored_enc.plane[3].block[0] = form_word;
        RadWriter::Weight w{};
        w.name = old_file.str(e.name);
        w.quantizer = old_file.str(e.quantizer);
        w.options = old_file.str(e.options);
        w.enc = stored_enc;
        w.rank = e.rank;
        for (int d = 0; d < RAD_MAX_RANK; ++d) w.shape[d] = e.shape[d];
        w.layer = e.layer;
        w.expert = e.expert;
        w.slot = e.slot;
        const void* planes[4] = { dummy.data(), dst.data(), sdst.data(), &word_bytes };
        rc = writer.add_weight(w, planes);
        if (rc < 0) {
            fprintf(stderr, "%s: add_weight %d\n", old_file.str(e.name), rc);
            return 1;
        }
        ++n_weights;
        if ((n_weights % 64) == 0)
            fprintf(stderr, "I permute %lld weights\n", (long long)n_weights);
    }
    rc = writer.finish();
    if (rc != RAD_OK) {
        fprintf(stderr, "finish: %d\n", rc);
        return 1;
    }
    fprintf(stderr, "I permute done: %lld weights\n", (long long)n_weights);
    if (argc == 4 && std::strcmp(argv[3], "--verify") == 0)
        return verify_all(old_file, argv[2], layout_fn, relayout_fn);
    return 0;
}
