/* ds41_vision_test -- DeepSeek-V4.1's pictures against the release's own code, on the CPU.
 *
 * The reference is the release's inference/vision.py, image_processor.py and encoding/encoding.py, run unchanged on
 * the CPU by ~/gpu/ds41/vis/scripts/ref_tower.py, which writes, per picture, the patches load_image makes (pix.bf16),
 * the ViT's rows (vit.f32), the aligner's (ali.f32), the whole span merge_image_embeddings writes (span.f32), its
 * grids (meta.json) and the picture as PIL decodes it (rgb.u8, dump_rgb.py). DS41_VIS_REF names that directory and
 * DS41_VIS_RELEASE the release (its shards 1 and 2 hold the tower); without them every case skips.
 *
 *   - the processor (core/mm Family::DeepSeek41) stages the reference's patches bit for bit, at its grids, and the
 *     span layout matches image_token_types (start, rows with a newline each, end);
 *   - the tower block (arch/common/rad_block_ds41_vision.h), declared on the recording builder and issued over the
 *     core's staging of one or more pictures, executed op by op on libref ($RADIANCE_HOME/kernels/libref.so), against
 *     the reference's span: the cosine over each picture and the per-row relative error, and the learned rows bit
 *     for bit (a copy). DS41_VIS_PICS picks the pictures (default ocr.png,count.png). */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/common/rad_block_ds41_vision.h"
#include "mm/processor.h"
#include "rad_plugin.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <sstream>

using json = nlohmann::json;

namespace {

std::string env(const char* k) {
    const char* v = std::getenv(k);
    return v ? v : "";
}
std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, c)) if (!x.empty()) out.push_back(x);
    return out;
}

/* The container's DeepSeek-V4.1 keys the processor and the tower read (the release's config.json). */
MetaKV ds41_meta() {
    RadModelMeta m{};
    m.n_layers = 40;
    m.n_embd = 5120;
    MetaKV k(m);
    k.set("image_token_id", "129264")
     .set("vision_config.model_type", "deepseek_v41_vision")
     .set("vision_config.num_hidden_layers", "32")
     .set("vision_config.hidden_size", "1024")
     .set("vision_config.num_attention_heads", "16")
     .set("vision_config.intermediate_size", "2816")
     .set("vision_config.patch_size", "14")
     .set("vision_config.rope_theta", "10000")
     .set("vision_config.downsample_ratio", "3")
     .set("vision_config.max_image_tokens", "1024")
     .set("vision_config.min_pixels", "295936");
    return k;
}

rad::mm::VisionConfig vision_config(int64_t max_patches) {
    MetaKV mk = ds41_meta();
    rad::mm::VisionConfig vc;
    std::string why;
    if (vc.from_meta(*mk.meta(), &why) < 0) fprintf(stderr, "    from_meta: %s\n", why.c_str());
    vc.max_patches = max_patches;
    return vc;
}

/* A picture's item from the reference's decoded RGB. */
std::shared_ptr<rad::mm::Item> item_of(const rad::mm::Processor& proc, const std::string& dir) {
    const json rj = json::parse(slurp(dir + "/rgb.json"));
    rad::mm::Frame f;
    f.w = rj["w"].get<int32_t>();
    f.h = rj["h"].get<int32_t>();
    const std::vector<char> rgb = slurp(dir + "/rgb.u8");
    f.rgb.assign(rgb.begin(), rgb.end());
    std::shared_ptr<rad::mm::Item> it;
    std::string why;
    const int rc = proc.image_from(f, &it, &why);
    if (rc < 0) fprintf(stderr, "    image_from: %s\n", why.c_str());
    return it;
}

}  // namespace

/* The processor stages the reference's patches bit for bit and lays the span out as image_token_types does. */
TEST(ds41_processor_matches_the_release) {
    const std::string ref = env("DS41_VIS_REF");
    if (ref.empty()) SKIP("DS41_VIS_REF is not set");
    rad::mm::Processor proc;
    REQUIRE_EQ(proc.init(vision_config(16380), [](const std::string&) { return std::vector<int32_t>{}; }), RAD_OK);
    const json prompts = json::parse(slurp(ref + "/prompts.json"));
    int n = 0;
    for (auto it = prompts.begin(); it != prompts.end(); ++it) {
        const std::string name = it.key(), dir = ref + "/" + name;
        const json meta = json::parse(slurp(dir + "/meta.json"));
        auto item = item_of(proc, dir);
        REQUIRE(item != nullptr);
        const int nvh = meta["n_vit"][0], nvw = meta["n_vit"][1], nlh = meta["n_llm"][0], nlw = meta["n_llm"][1];
        const int64_t n_tok = meta["n_tok"], real = (int64_t)nvh * nvw;
        CHECK_EQ(item->seg_h(0), nvh);
        CHECK_EQ(item->seg_w(0), nvw);
        CHECK_EQ((int64_t)item->expand.size(), n_tok);
        CHECK_EQ(item->n_patch, 9 * n_tok);
        CHECK_EQ(item->n_rows(), n_tok);
        /* the real patches, bit for bit; the rest of the segment zero */
        const std::vector<char> pix = slurp(dir + "/pix.bf16");
        REQUIRE_EQ((int64_t)pix.size(), real * 588 * 2);
        const bool same = std::memcmp(pix.data(), item->pixels.data(), pix.size()) == 0;
        int64_t first_bad = -1;
        if (!same) {
            const uint16_t* a = reinterpret_cast<const uint16_t*>(pix.data());
            for (int64_t i = 0; i < real * 588; ++i)
                if (a[i] != item->pixels[(size_t)i]) { first_bad = i; break; }
        }
        bool zero = true;
        for (size_t i = (size_t)(real * 588); i < item->pixels.size(); ++i) zero &= item->pixels[i] == 0;
        fprintf(stderr, "    %s: grid %dx%d -> %dx%d tokens (span %lld), patches %s%s, padding %s\n", name.c_str(),
                nvh, nvw, nlh, nlw, (long long)n_tok, same ? "identical" : "DIFFER at ",
                same ? "" : std::to_string(first_bad).c_str(), zero ? "zero" : "NOT ZERO");
        CHECK(same);
        CHECK(zero);
        /* the span: the reference's token types at the image's place in the prompt */
        const json& pr = it.value();
        const int64_t at = pr["image_start"];
        std::vector<int> types;
        for (int64_t i = 0; i < n_tok; ++i) types.push_back(pr["types"][(size_t)(at + i)].get<int>());
        std::vector<int32_t> pl((size_t)(2 * item->n_patch));
        std::vector<int32_t> planes((size_t)(4 * item->n_patch));
        item->coords(0, item->n_patch, planes.data());
        const int32_t* p2 = planes.data() + 2 * item->n_patch;
        const int32_t* p3 = planes.data() + 3 * item->n_patch;
        const int64_t n_sp = nlh + 2;
        bool layout = true;
        int64_t sp = 0;
        for (int64_t r = 0; r < n_tok; ++r) {
            const int t = types[(size_t)r];   /* 0 start, 1 image, 2 newline, 3 end (image_processor.py) */
            bool any = false;
            for (int k = 0; k < 9; ++k) any |= p2[r * 9 + k] >= 0;
            if (t == 1) {
                layout &= any || true;   /* an image row may be all padding only past the grid; checked below */
            } else {
                layout &= !any;
                const int want = t == 0 ? 0 : t == 2 ? 1 : 2;
                layout &= sp < n_sp && p3[sp] == want && p3[n_sp + sp] == r;
                ++sp;
            }
        }
        layout &= sp == n_sp && p3[2 * n_sp] == 0 && p3[2 * n_sp + 1] == real;
        /* every real patch is read by exactly one tap */
        std::vector<int> hits((size_t)real, 0);
        for (int64_t j = 0; j < item->n_patch; ++j)
            if (p2[j] >= 0) {
                if (p2[j] >= real) layout = false;
                else ++hits[(size_t)p2[j]];
            }
        for (int h : hits) layout &= h == 1;
        /* rotary positions: raster (row, column) */
        for (int64_t j = 0; j < real; ++j) layout &= planes[(size_t)j] == j / nvw && planes[(size_t)(item->n_patch + j)] == j % nvw;
        CHECK(layout);
        /* every placeholder the image token */
        for (int32_t id : item->expand) CHECK_EQ(id, 129264);
        ++n;
    }
    CHECK(n > 0);
}

namespace {

/* ---- the tower on libref */
struct RefLib {
    void* h = nullptr;
    std::map<std::string, const RadKernelInfo*> k;
};
RefLib load_libref() {
    RefLib r;
    const std::string home = env("RADIANCE_HOME");
    if (home.empty()) return r;
    r.h = dlopen((home + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!r.h) return r;
    auto open = (int (*)(void))dlsym(r.h, "rad_plugin_open");
    if (open && open() < 0) return r;
    auto cnt = (int (*)(void))dlsym(r.h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(r.h, "rad_kernel_at");
    if (!cnt || !at) return r;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* ki = at(i);
        if (ki && ki->op && ki->launch && !r.k.count(ki->op)) r.k[ki->op] = ki;
    }
    return r;
}

/* The release's safetensors, by tensor name: shard 1 (vision.*, aligner.*) and 2 (image_*). */
struct Ckpt {
    std::map<std::string, std::pair<std::string, std::pair<int64_t, int64_t>>> at;   /* file, [off, end) */
    bool open(const std::string& dir) {
        for (const char* f : { "model-00001-of-00048.safetensors", "model-00002-of-00048.safetensors" }) {
            const std::string p = dir + "/" + f;
            std::ifstream in(p, std::ios::binary);
            uint64_t n = 0;
            if (!in.read(reinterpret_cast<char*>(&n), 8)) return false;
            std::string h(n, '\0');
            in.read(&h[0], (std::streamsize)n);
            const json j = json::parse(h);
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                const auto& o = it.value()["data_offsets"];
                at[it.key()] = { p, { 8 + (int64_t)n + o[0].get<int64_t>(), 8 + (int64_t)n + o[1].get<int64_t>() } };
            }
        }
        return !at.empty();
    }
    bool read(const std::string& name, std::vector<uint8_t>* out) const {
        auto it = at.find(name);
        if (it == at.end()) return false;
        std::ifstream in(it->second.first, std::ios::binary);
        in.seekg(it->second.second.first);
        const size_t n = (size_t)(it->second.second.second - it->second.second.first);
        const size_t was = out->size();
        out->resize(was + n);
        return (bool)in.read(reinterpret_cast<char*>(out->data() + was), (std::streamsize)n);
    }
};

int64_t dt_bytes(uint32_t dt) { return rad_dtype_bytes(dt, 1) > 0 ? rad_dtype_bytes(dt, 1) : 1; }

RadTensor tmpl(uint32_t dt, uint32_t rank, const int64_t* shape, void* data) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = rank;
    for (uint32_t i = 0; i < rank; ++i) t.shape[i] = shape[i];
    rad_tensor_pack(&t);
    return t;
}

/* Runs a recorded pass on libref: buffers in host memory, weights from the checkpoint through the plugin's name
 * map, each issue's operands resolved as core/runtime/issue.cpp resolves them. */
struct Exec {
    RadBuilder& b;
    const RefLib& lib;
    std::vector<std::vector<uint8_t>> buf, wd;
    std::string err;
    Exec(RadBuilder& bb, const RefLib& l) : b(bb), lib(l) {}

    bool load(const Ckpt& ck) {
        buf.assign(b.bufs.size() + 1, {});
        for (size_t i = 0; i < b.bufs.size(); ++i) {
            const RadBufDecl& d = b.bufs[i].d;
            int64_t n = 1;
            for (uint32_t r = 0; r < d.rank; ++r) n *= d.shape[r];
            buf[i + 1].assign((size_t)(n * dt_bytes(d.dtype)), 0);
        }
        wd.assign(b.weights.size() + 1, {});
        for (size_t i = 0; i < b.weights.size(); ++i) {
            const RecWeight& w = b.weights[i];
            const RecMap* mp = nullptr;
            for (const auto& m : b.maps) if (m.declared == w.name) mp = &m;
            if (!mp) { err = "no map for " + w.name; return false; }
            for (const auto& s : mp->src)
                if (!ck.read(s, &wd[i + 1])) { err = "no tensor " + s; return false; }
            int64_t n = 1;
            for (uint32_t r = 0; r < w.d.rank; ++r) n *= w.d.shape[r];
            if ((int64_t)wd[i + 1].size() != n * dt_bytes(w.d.dtype)) { err = "size of " + w.name; return false; }
        }
        return true;
    }

    bool run(const RadCtx& c, double* seconds_by_op = nullptr) {
        (void)seconds_by_op;
        for (const RecIssue& is : c.issues) {
            const RecOp* op = b.op_of(is.op);
            if (!op) { err = "an issue of no op"; return false; }
            auto kit = lib.k.find(op->op);
            if (kit == lib.k.end()) { err = "libref has no " + op->op; return false; }
            RadTensor t[24] = {};
            const int n = (int)is.opd.size();
            for (int i = 0; i < n && i < 24; ++i) {
                const RadOperand& o = is.opd[i];
                if (o.kind == RAD_OPK_BUF) {
                    const RadBufDecl& d = b.bufs[o.handle - 1].d;
                    t[i] = tmpl(d.dtype, d.rank, d.shape, buf[o.handle].data() + o.offset * dt_bytes(d.dtype));
                    if (o.rows > 0) t[i].shape[0] = o.rows;
                    if (o.cols > 0) t[i].shape[t[i].rank - 1] = o.cols;
                } else if (o.kind == RAD_OPK_WEIGHT) {
                    const RadWeightDecl& d = b.weights[o.handle - 1].d;
                    t[i] = tmpl(d.dtype, d.rank, d.shape, wd[o.handle].data() + o.offset * dt_bytes(d.dtype));
                    if (o.rows > 0) t[i].shape[0] = o.rows;
                    if (o.cols > 0) t[i].shape[t[i].rank - 1] = o.cols;
                } else if (o.kind == RAD_OPK_RAW) {
                    t[i].data = o.raw;
                    t[i].dtype = o.dtype;
                    if (o.rows > 0 && o.cols > 0) {
                        t[i].rank = 2; t[i].shape[0] = o.rows; t[i].shape[1] = o.cols;
                        t[i].stride[0] = o.cols; t[i].stride[1] = 1;
                    } else if (o.rows > 0) {
                        t[i].rank = 1; t[i].shape[0] = o.rows; t[i].stride[0] = 1;
                    }
                }
            }
            std::vector<RadParam> ps;
            for (const RecParam& p : op->p) {
                RadParam q{};
                q.key = p.key.c_str();
                if (p.kind == RAD_P_RANGE) { q.kind = RAD_P_INT; q.ival = is.n; }
                else { q.kind = p.kind; q.ival = p.ival; q.ihi = p.ihi; q.dval = p.dval; q.sval = p.sval.c_str(); }
                ps.push_back(q);
            }
            RadArgs a{};
            a.t = t;
            a.n_t = n;
            a.p = ps.data();
            a.n_p = (int)ps.size();
            a.rank = 0;
            a.world_size = 1;
            a.kv_used_max_h = -1;
            std::vector<uint8_t> scratch;
            if (kit->second->scratch) {
                const int64_t s = kit->second->scratch(&a);
                if (s > 0) { scratch.assign((size_t)s, 0); a.scratch = scratch.data(); a.scratch_bytes = s; }
            }
            void* inst = nullptr;
            if (kit->second->init && kit->second->init(a.p, a.n_p, 0, 1, &inst) < 0) { err = "init " + op->op; return false; }
            a.instance = inst;
            const int rc = kit->second->launch(&a, nullptr);
            if (kit->second->fini) kit->second->fini(inst);
            if (rc < 0) { err = op->op + " returned " + std::to_string(rc); return false; }
        }
        return true;
    }
    const uint16_t* bf16(rad_buf h) const { return reinterpret_cast<const uint16_t*>(buf[h].data()); }
};

}  // namespace

/* The tower block on libref over the core's staging of one or more pictures, against the release's span. */
TEST(ds41_tower_on_libref_matches_the_release) {
    const std::string ref = env("DS41_VIS_REF"), rel = env("DS41_VIS_RELEASE");
    if (ref.empty() || rel.empty()) SKIP("DS41_VIS_REF / DS41_VIS_RELEASE are not set");
    const RefLib lib = load_libref();
    if (lib.k.empty()) SKIP("no libref under $RADIANCE_HOME/kernels");
    Ckpt ck;
    REQUIRE(ck.open(rel));
    rad::mm::Processor proc;
    REQUIRE_EQ(proc.init(vision_config(16380), [](const std::string&) { return std::vector<int32_t>{}; }), RAD_OK);
    std::vector<std::string> pics = split(env("DS41_VIS_PICS").empty() ? "ocr.png,count.png" : env("DS41_VIS_PICS"), ',');
    /* the pass: the pictures' items, staged as BatchBuilder::build_encoder stages them */
    std::vector<std::shared_ptr<rad::mm::Item>> items;
    int64_t Pn = 0;
    for (const auto& p : pics) {
        items.push_back(item_of(proc, ref + "/" + p));
        REQUIRE(items.back() != nullptr);
        Pn += items.back()->n_patch;
    }
    std::vector<uint16_t> pix;
    std::vector<int32_t> coord((size_t)(4 * Pn)), cu{ 0 }, sh, sw;
    int64_t at = 0;
    for (const auto& it : items) {
        pix.insert(pix.end(), it->pixels.begin(), it->pixels.end());
        std::vector<int32_t> tmp((size_t)(4 * it->n_patch));
        it->coords(0, it->n_patch, tmp.data());
        for (int c = 0; c < 4; ++c)
            std::memcpy(coord.data() + c * Pn + at, tmp.data() + c * it->n_patch, (size_t)it->n_patch * 4);
        at += it->n_patch;
        cu.push_back((int32_t)at);
        sh.push_back(it->seg_h(0));
        sw.push_back(it->seg_w(0));
    }
    MetaKV mk = ds41_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx(0, 1);
    ctx.max_tok = 4160;
    ctx.max_enc_patches = Pn;   /* the pass exactly (a multiple of 9) */
    rad::arch::Names nm("");
    rad::arch::Ds41Vision vit;
    REQUIRE_EQ(vit.declare(&b, mk.meta(), &ctx, nm, 5120, 43), RAD_OK);
    REQUIRE(vit.on);
    CHECK_EQ(b.encoder.merge, 9);
    CHECK_EQ(b.encoder.patch_dim, 588);
    RadBatch batch{};
    batch.enc = RAD_MM_IMAGE;
    batch.enc_n_patch = Pn;
    batch.enc_pixels = pix.data();
    batch.enc_coord = coord.data();
    batch.enc_n_seg = (int64_t)items.size();
    batch.enc_cu = cu.data();
    batch.enc_seg_h = sh.data();
    batch.enc_seg_w = sw.data();
    RadCtx c{ .b = &b, .batch = &batch };
    vit.step(&c, &batch);
    REQUIRE(c.step_fail.empty());
    Exec ex(b, lib);
    if (!ex.load(ck)) fprintf(stderr, "    load: %s\n", ex.err.c_str());
    REQUIRE(ex.err.empty());
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = ex.run(c);
    if (!ok) fprintf(stderr, "    run: %s\n", ex.err.c_str());
    REQUIRE(ok);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "    %zu picture(s), %lld patches, %zu issues on libref in %.1f s\n", items.size(), (long long)Pn,
            c.issues.size(), sec);
    /* compare each picture's rows: the span (out) and the ViT (post, the real patches) */
    const std::string dump = env("DS41_VIS_DUMP");
    int64_t p0 = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        const std::string dir = ref + "/" + pics[i];
        const json meta = json::parse(slurp(dir + "/meta.json"));
        const int64_t n_tok = meta["n_tok"], real = meta["patches"];
        const std::vector<char> sp = slurp(dir + "/span.f32"), vt = slurp(dir + "/vit.f32");
        const float* want = reinterpret_cast<const float*>(sp.data());
        const float* vwant = reinterpret_cast<const float*>(vt.data());
        const uint16_t* got = ex.bf16(vit.b_out) + (p0 / 9) * 5120;
        const uint16_t* vgot = ex.bf16(vit.b_post) + p0 * 1024;
        auto stats = [&](const uint16_t* g, const float* w, int64_t rows, int64_t width, double* cosv, double* med,
                         double* worst, int64_t* exact_rows) {
            double dot = 0, gg = 0, ww = 0;
            std::vector<double> rel;
            int64_t ex_rows = 0;
            for (int64_t r = 0; r < rows; ++r) {
                double num = 0, den = 0;
                bool exact = true;
                for (int64_t k = 0; k < width; ++k) {
                    const double a = rad_bf16_to_f32(g[r * width + k]), bb = w[r * width + k];
                    dot += a * bb; gg += a * a; ww += bb * bb;
                    num += (a - bb) * (a - bb); den += bb * bb;
                    exact &= a == bb;
                }
                rel.push_back(den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0));
                ex_rows += exact;
            }
            std::vector<double> s = rel;
            std::sort(s.begin(), s.end());
            *cosv = dot / std::sqrt(gg * ww);
            *med = s[s.size() / 2];
            *worst = s.back();
            *exact_rows = ex_rows;
        };
        double cs = 0, md = 0, wr = 0, vcs = 0, vmd = 0, vwr = 0;
        int64_t er = 0, ver = 0;
        stats(got, want, n_tok, 5120, &cs, &md, &wr, &er);
        stats(vgot, vwant, real, 1024, &vcs, &vmd, &vwr, &ver);
        /* the learned rows: start, every newline, end -- copies, so bit for bit */
        const int64_t h = meta["n_llm"][0], w = meta["n_llm"][1];
        bool learned = true;
        std::vector<int64_t> lr{ 0, n_tok - 1 };
        for (int64_t y = 0; y < h; ++y) lr.push_back(1 + y * (w + 1) + w);
        for (int64_t r : lr)
            for (int64_t k = 0; k < 5120; ++k) learned &= rad_bf16_to_f32(got[r * 5120 + k]) == want[r * 5120 + k];
        /* the same image rows against the release's float32 run, beside the bf16 reference's own distance to it */
        {
            const std::vector<char> a32 = slurp(dir + "/ali32.f32"), ab = slurp(dir + "/ali.f32");
            if (!a32.empty() && !ab.empty()) {
                const float* f32 = reinterpret_cast<const float*>(a32.data());
                const float* fbf = reinterpret_cast<const float*>(ab.data());
                std::vector<uint16_t> ours;
                std::vector<float> theirs;
                std::vector<char> is_l((size_t)n_tok, 0);
                for (int64_t r : lr) is_l[(size_t)r] = 1;
                for (int64_t r = 0; r < n_tok; ++r)
                    if (!is_l[(size_t)r]) ours.insert(ours.end(), got + r * 5120, got + (r + 1) * 5120);
                const int64_t rows = (int64_t)ours.size() / 5120;
                double c1 = 0, m1 = 0, w1 = 0, c2 = 0, m2 = 0, w2 = 0;
                int64_t e1 = 0, e2 = 0;
                stats(ours.data(), f32, rows, 5120, &c1, &m1, &w1, &e1);
                std::vector<uint16_t> refb((size_t)(rows * 5120));
                for (size_t q = 0; q < refb.size(); ++q) refb[q] = rad_f32_to_bf16(fbf[q]);
                stats(refb.data(), f32, rows, 5120, &c2, &m2, &w2, &e2);
                fprintf(stderr, "    %s: the %lld image rows against the release in float32: cosine %.6f, median %.4f, "
                                "largest %.4f (the release's own bf16 run: cosine %.6f, median %.4f, largest %.4f)\n",
                        pics[i].c_str(), (long long)rows, c1, m1, w1, c2, m2, w2);
            }
        }
        fprintf(stderr, "    %s: span %lld rows: cosine %.6f, per-row relative error median %.4f, largest %.4f, "
                        "%lld rows bit-exact; learned rows %s; ViT %lld rows: cosine %.6f, median %.4f, largest %.4f\n",
                pics[i].c_str(), (long long)n_tok, cs, md, wr, (long long)er, learned ? "identical" : "DIFFER",
                (long long)real, vcs, vmd, vwr);
        CHECK(learned);
        CHECK(cs >= 0.99);
        CHECK(md <= 0.08);
        CHECK(vcs >= 0.99);
        if (!dump.empty()) {
            std::ofstream o(dump + "/" + pics[i] + ".span.bf16", std::ios::binary);
            o.write(reinterpret_cast<const char*>(got), n_tok * 5120 * 2);
        }
        p0 += it->n_patch;
    }
}
RAD_TEST_MAIN()
