/* batch_wire.cpp -- the step batch as it crosses to a two-node pipeline's follower (tp8 S4; see
 * BatchBuilder::Wire in batch.h). Kept out of batch.cpp so the one-machine build path is the code
 * it was: build() only records where its span sits (last_off_ and friends). */
#include "batch.h"
#include "device/device.h"
#include "kvstream/bind.h"
#include "net/node_link.h"

#include <cstring>

namespace rad {

#define RAD_WIRE_TRY_NULL(expr)                                                                \
    do { int _s = (expr);                                                                      \
         if (_s < 0) { RAD_ERR("step wire: %s -> %s", #expr, rad_strerror(_s));                \
                       return nullptr; } } while (0)

template <class T>
static inline T* wire_on_rank(void* base_r, void* base_0, T* p0) {
    return p0 ? (T*)((char*)base_r + ((char*)p0 - (char*)base_0)) : nullptr;
}

uint64_t BatchBuilder::layout_hash() const {
    std::vector<int64_t> w;
    w.push_back(max_tok_);
    w.push_back(max_seqs_);
    w.push_back(max_ctx_);
    w.push_back(set_.host_bytes);
    w.push_back(set_.dev_bytes);
    w.push_back(set_.bt_base);
    w.push_back((int64_t)set_.groups.size());
    for (size_t gi = 0; gi < set_.groups.size(); ++gi) {
        const Group& g = set_.groups[gi];
        w.push_back(g.block_size);
        w.push_back(g.max_blocks);
        w.push_back(g.sidx_pitch);
        w.push_back(g.bt_fix);
        if (gi < kvgeom_.g.size()) {
            const KVGeom::G& L = kvgeom_.g[gi];
            w.push_back(L.paged ? 1 : 0);
            w.push_back((int64_t)L.kind);
            w.push_back(L.window);
        }
    }
    w.push_back(mm_on_);
    w.push_back(bidi_on_);
    w.push_back(lora_on_);
    w.push_back(pw_cap_);
    w.push_back(cfg_.n_spec);
    w.push_back(cfg_.kld() ? 1 : 0);
    w.push_back(derived_bytes_);
    for (const Derived& d : derived_) {
        w.push_back(d.fn_index);
        w.push_back(d.conv_grp);
        w.push_back(d.off);
        w.push_back(d.capacity);
    }
    /* Two CRC-32Cs, the second over the words in reverse order, as one 64-bit identity. */
    const uint32_t a = net::crc32c(w.data(), w.size() * sizeof(int64_t), 0);
    std::vector<int64_t> r(w.rbegin(), w.rend());
    const uint32_t b = net::crc32c(r.data(), r.size() * sizeof(int64_t), 0x9e3779b9u);
    return ((uint64_t)a << 32) | b;
}

bool BatchBuilder::wire_last(Wire* w, const char** span) const {
    if (out_last_ < 0 || last_off_ < 0 || last_enc_) return false;
    const Out& o = out_[out_last_];
    const RadBatch& B = o.batch[0];
    *w = Wire{};
    w->layout = layout_hash();
    w->phase = B.phase;
    w->step = B.step;
    w->n_tok = B.n_tok;
    w->n_seq = B.n_seq;
    w->n_out = B.n_out;
    w->n_draft_out = B.n_draft_out;
    w->n_seq_decode = B.n_seq_decode;
    w->n_tok_decode = B.n_tok_decode;
    w->n_ahead = B.n_ahead;
    w->n_bidi = B.n_bidi;
    w->n_pw = B.n_plugin_words;
    w->n_mm = B.n_mm_rows;
    w->max_q = B.max_q_len;
    w->ctx_bound = B.max_ctx_len;
    w->max_q_dec = B.max_q_len_decode;
    w->max_q_pf = B.max_q_len_prefill;
    w->draft_pass = B.draft_pass;
    w->n_spec = B.n_spec;
    w->n_ck = B.n_checkpoints;
    w->rope_mixed = B.rope_mixed;
    w->cvec_rows = B.control_vector;
    w->score_only = B.score_only;
    w->tree = B.spec_parent ? 1 : 0;
    w->lora_rows = B.lora_slot ? 1 : 0;
    w->n_groups = (int32_t)set_.groups.size();
    w->n_derived = (int32_t)derived_.size();
    if (w->n_groups > RAD_SCHED_MAX_KV_GROUPS || w->n_derived > kWireMaxDerived) return false;
    w->bt_base = set_.bt_base;
    w->used = last_used_;
    w->rng_at = last_rng_at_;
    w->n_rng = last_n_rng_;
    for (int gi = 0; gi < w->n_groups; ++gi) {
        w->pitch[gi] = pitch_[(size_t)gi];
        w->used_max[gi] = o.kv_used_max_h[(size_t)gi];
    }
    for (int i = 0; i < w->n_derived; ++i) w->derived_n[i] = derived_[(size_t)i].produced;
    *span = ring_.host + last_off_;
    return true;
}

/* Review F1's guard: every slot a replayed step writes (each group's slot mapping) and every block
 * id or state slot its table payload names lies inside this node's carve. The cap both nodes agree
 * on before the carve keeps it so; this makes a broken cap a refused step, never a store past the
 * pool. One pass over the step's rows a group and over the restaged table words (a decode step
 * restages a few, a new sequence its row). */
bool BatchBuilder::wire_in_carve(const Wire& w, const char* base) const {
    const Set& s = set_;
    const size_t G = s.groups.size();
    auto lim = [](const std::vector<int64_t>& v, size_t gi) { return gi < v.size() ? v[gi] : (int64_t)-1; };
    for (size_t gi = 0; gi < G; ++gi) {
        const int64_t sl = lim(wire_slot_lim_, gi);
        const int32_t* hs = s.groups[gi].h_slot;
        if (sl < 0 || !hs) continue;
        for (int64_t t = 0; t < w.n_tok; ++t)
            if ((int64_t)hs[t] >= sl) {
                RAD_ERR("step wire: step %d writes slot %d of KV group %zu, past this node's %lld: the "
                        "two nodes' KV carves differ (a broken cap), the step is refused", w.step, hs[t], gi,
                        (long long)sl);
                return false;
            }
    }
    const RadCopyRange* rng = reinterpret_cast<const RadCopyRange*>(base + w.rng_at);
    for (int64_t i = 0; i < w.n_rng; ++i) {
        const RadCopyRange& c = rng[i];
        if (c.dst < s.bt_base) continue;
        const int32_t* v = reinterpret_cast<const int32_t*>(base + c.src);
        const int64_t w0 = (c.dst - s.bt_base) / (int64_t)sizeof(int32_t);
        const int64_t n = c.bytes / (int64_t)sizeof(int32_t);
        size_t gi = 0;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t word = w0 + k;
            while (gi < G && word >= s.groups[gi].bt_fix + max_seqs_ * s.groups[gi].max_blocks) ++gi;
            if (gi >= G) break;
            if (word < s.groups[gi].bt_fix) continue;   /* the alignment between two groups' tables */
            const int64_t il = lim(wire_id_lim_, gi);
            if (il >= 0 && (int64_t)v[k] >= il) {
                RAD_ERR("step wire: step %d's block table names id %d in KV group %zu, past this node's %lld: "
                        "the two nodes' KV carves differ (a broken cap), the step is refused", w.step, v[k], gi,
                        (long long)il);
                return false;
            }
        }
    }
    return true;
}

/* The same fill as build()'s, from the leader's scalars: kept beside it rather than shared, so the
 * one-machine path stays exactly the code it was. */
const RadBatch* BatchBuilder::wire_replay(const Wire& w, const char* span, int64_t bytes, bool run) {
    if (w.magic != Wire{}.magic || w.version != Wire{}.version) {
        RAD_ERR("step wire: not a step batch of this build (magic %08x version %u)", w.magic, w.version);
        return nullptr;
    }
    if (w.layout != layout_hash()) {
        RAD_ERR("step wire: the leader's step batch is laid out differently (layout %016llx, this node "
                "%016llx): the two nodes' programs or options differ", (unsigned long long)w.layout,
                (unsigned long long)layout_hash());
        return nullptr;
    }
    if (bytes != w.used || w.used <= 0 || w.n_groups != (int32_t)set_.groups.size() ||
        w.n_derived != (int32_t)derived_.size() || w.n_seq > max_seqs_ || w.n_tok > max_tok_ ||
        w.n_seq < 0 || w.n_tok < 0 || w.bt_base != set_.bt_base || w.n_rng < 0 ||
        w.rng_at < w.bt_base || w.rng_at + w.n_rng * (int64_t)sizeof(RadCopyRange) > w.used) {
        RAD_ERR("step wire: step %d's header does not describe its %lld-byte span (%lld groups, %lld "
                "rows, %lld ranges)", w.step, (long long)bytes, (long long)w.n_groups,
                (long long)w.n_tok, (long long)w.n_rng);
        return nullptr;
    }
    if (w.n_mm > 0) {
        RAD_ERR("step wire: step %d carries encoder rows, which the two-node pipeline does not ship",
                w.step);
        return nullptr;
    }
    Set& s = set_;
    int64_t off = 0;
    RAD_WIRE_TRY_NULL(reserve(w.used, &off));
    std::memcpy(ring_.host + off, span, (size_t)w.used);
    carve(s, ring_.host + off);
    RadCopyRange* const rng = reinterpret_cast<RadCopyRange*>(ring_.host + off + w.rng_at);
    for (int64_t i = 0; i < w.n_rng; ++i) {
        const RadCopyRange& c = rng[i];
        if (c.src < 0 || c.dst < 0 || c.bytes < 0 || c.src + c.bytes > w.used ||
            c.dst + c.bytes > s.dev_bytes) {
            RAD_ERR("step wire: step %d's upload range %lld runs outside the span or the slab", w.step,
                    (long long)i);
            return nullptr;
        }
    }
    if (!wire_id_lim_.empty() && !wire_in_carve(w, ring_.host + off)) return nullptr;
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const int st = rad_memcpy_ranges_async(s.dev[r], ring_.host + off, s.bt_base, rng,
                                               (int)w.n_rng, ranks_[r].stream);
        if (st < 0) {
            RAD_ERR("step wire: rank %zu's upload -> %s", r, rad_strerror(st));
            return nullptr;
        }
    }
    if (run)
        for (size_t i = 0; i < derived_.size(); ++i) {
            Derived& d = derived_[i];
            d.produced = w.derived_n[i];
            if (d.produced < 0 || d.produced > d.capacity) {
                RAD_ERR("step wire: derivation '%s' produced %lld of %lld", d.name, (long long)d.produced,
                        (long long)d.capacity);
                return nullptr;
            }
            const int32_t* host = (const int32_t*)(s.h_derived + d.off);
            for (const RankIO& io : ranks_) {
                void* dst = d.handle < io.prog->buffers.size() ? io.prog->buffers[d.handle].ptr : nullptr;
                if (!dst) return nullptr;
                RAD_WIRE_TRY_NULL(rad_memcpy_async(dst, host, d.produced * (int64_t)sizeof(int32_t), io.stream));
            }
        }
    for (int gi = 0; gi < w.n_groups; ++gi) pitch_[(size_t)gi] = w.pitch[gi];

    Out& o = out_[out_next_];
    o.has_seqs = false;
    o.seqs.clear();
    for (int gi = 0; gi < w.n_groups; ++gi) o.kv_used_max_h[(size_t)gi] = w.used_max[gi];
    for (size_t gi = 0; gi < s.groups.size(); ++gi) {
        Group& g = s.groups[gi];
        const KVGeom::G& L = kvgeom_.g[gi];
        for (size_t r = 0; r < ranks_.size(); ++r) {
            void* base = s.dev[r];
            RadKVGroupBatch& b = o.kv[r][gi];
            b.group = g.g;
            b.slot_mapping = wire_on_rank(base, s.dev[0], g.d_slot);
            b.block_table = wire_on_rank(base, s.dev[0], g.d_bt);
            b.block_table_pitch = w.pitch[gi];
            b.seqused = wire_on_rank(base, s.dev[0], g.d_used);
            b.seqused_h = g.h_used;
            b.state_index = wire_on_rank(base, s.dev[0], g.d_sidx);
            b.state_index_pitch = g.sidx_pitch;
            b.block_size = L.block_size;
            b.max_blocks = w.pitch[gi];
            b.stream = kvstream_step_view(&o.stream[r][gi],
                                          ranks_[r].kv_streams ? ranks_[r].kv_streams[gi] : nullptr,
                                          ranks_[r].kv_streams != nullptr, ranks_[0].kv_base,
                                          ranks_[r].kv_base);
        }
    }
    for (size_t r = 0; r < ranks_.size(); ++r) {
        void* const base = s.dev[r];
        void* const b0   = s.dev[0];
        RadBatch& B = o.batch[r];
        B = RadBatch{};
        B.phase = w.phase;
        B.step = w.step;
        B.n_tok = w.n_tok;
        B.n_seq = w.n_seq;
        B.token_ids = wire_on_rank(base, b0, s.d_tok);
        B.positions = wire_on_rank(base, b0, s.d_pos);
        B.positions_h = s.h_pos;
        B.cu_seqlens = wire_on_rank(base, b0, s.d_cu);
        B.seq_ids = wire_on_rank(base, b0, s.d_sid);
        B.n_out = w.n_out;
        B.out_ids = w.n_out ? wire_on_rank(base, b0, s.d_out) : nullptr;
        B.score_only = w.score_only;
        B.n_draft_out = w.n_draft_out;
        B.draft_out_ids = w.n_draft_out ? wire_on_rank(base, b0, s.d_dout) : nullptr;
        B.q_lens = wire_on_rank(base, b0, s.d_q);
        B.ctx_lens = wire_on_rank(base, b0, s.d_ctx);
        B.max_q_len = w.max_q;
        B.n_seq_decode = w.n_seq_decode;
        B.n_tok_decode = w.n_tok_decode;
        B.max_q_len_decode = w.max_q_dec;
        B.max_q_len_prefill = w.max_q_pf;
        B.max_ctx_len = w.ctx_bound;
        B.kv_used_max_h = o.kv_used_max_h.data();
        B.n_kv_groups = (int)o.kv[r].size();
        B.kv = o.kv[r].data();
        B.draft_pass = w.draft_pass;
        B.n_spec = w.n_spec;
        B.num_accepted = cfg_.n_spec > 0 ? wire_on_rank(base, b0, s.d_acc) : nullptr;
        B.spec_parent = w.tree ? wire_on_rank(base, b0, s.d_parent) : nullptr;
        B.anchor_rows = w.draft_pass == 1 ? wire_on_rank(base, b0, s.d_arow) : nullptr;
        B.rope_pos = mm_on_ ? wire_on_rank(base, b0, s.d_rope) : nullptr;
        B.rope_mixed = w.rope_mixed;
        B.n_bidi = w.n_bidi;
        B.bidi_used = w.n_bidi ? wire_on_rank(base, b0, s.d_bidi) : nullptr;
        B.bidi_row  = w.n_bidi ? s.h_bidi + max_tok_ : nullptr;
        B.bidi_n    = w.n_bidi ? s.h_bidi + 2 * max_tok_ : nullptr;
        B.bidi_seq  = w.n_bidi ? s.h_bidi + 3 * max_tok_ : nullptr;
        B.lora_slot = w.lora_rows ? wire_on_rank(base, b0, s.d_lora) : nullptr;
        B.control_vector = w.cvec_rows;
        B.n_plugin_words = w.n_pw;
        B.plugin_words   = w.n_pw ? wire_on_rank(base, b0, s.d_pw) : nullptr;
        B.plugin_cu      = w.n_pw ? wire_on_rank(base, b0, s.d_pwcu) : nullptr;
        B.plugin_words_h = w.n_pw ? s.h_pw : nullptr;
        B.plugin_cu_h    = w.n_pw ? s.h_pwcu : nullptr;
        B.n_checkpoints = w.n_ck;
        B.checkpoint_tok = w.n_ck ? wire_on_rank(base, b0, s.d_cktok) : nullptr;
        B.checkpoint_slot = w.n_ck ? wire_on_rank(base, b0, s.d_ckslot) : nullptr;
        B.n_ahead = w.n_ahead;
    }
    const int ev = ev_free_.back();
    ev_free_.pop_back();
    for (size_t r = 0; r < ranks_.size(); ++r) {
        if (rad_event_record(ev_[(size_t)ev][r], ranks_[r].stream) < 0) {
            RAD_ERR("step wire: could not record rank %zu's staging event", r);
            ev_free_.push_back(ev);
            return nullptr;
        }
    }
    ring_.fifo[(ring_.first + ring_.live) % kFlights] = Flight{ off, off + w.used, ev };
    ++ring_.live;
    last_n_seq_ = w.n_seq;
    last_n_tok_ = w.n_tok;
    last_tree_  = w.tree != 0;
    last_off_ = off;
    last_used_ = w.used;
    last_rng_at_ = w.rng_at;
    last_n_rng_ = w.n_rng;
    last_enc_ = false;
    ring_.head = off + w.used;
    out_last_ = out_next_;
    out_next_ = (out_next_ + 1) % kOut;
    return &o.batch[0];
}

}  /* namespace rad */
