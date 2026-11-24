/* sched_fakes.h -- the fakes the scheduler tests run against: the device layer as host memory,
 * and core/mem's KV manager and prefix cache as small faithful models of their interfaces
 * (core/sched/mem_iface.h; see that header for what must be reconciled with core/mem).
 *
 * The device fakes are weak, so a linked core/device/ backend would replace them at link time; a
 * test binary checks that they are the ones in use (g_fake_device). A binary that runs on the
 * real device layer and wants only the fakes of core/mem (mem_test, which compares FakeKV with
 * KVManager) defines SCHED_FAKES_KV_ONLY first. Include this once per test binary.
 */
#pragma once
#include "rad_test.h"

#include "sched/scheduler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>

#ifndef SCHED_FAKES_KV_ONLY
/* Set by the fake rad_dev_alloc: a binary that reads it false runs on the real backend. */
static bool g_fake_device = false;

/* ================================================================== the device fakes */
/* Weak, so a linked-in real host backend replaces them. Device memory is host memory here, so the
 * assertions on cu_seqlens and the slot mapping read exactly the bytes a kernel would. */
extern "C" {

__attribute__((weak)) void* rad_dev_alloc(int64_t bytes, int) {
    g_fake_device = true;
    return std::calloc(1, (size_t)(bytes > 0 ? bytes : 1));
}
__attribute__((weak)) void rad_dev_free(void* p, int) { std::free(p); }

/* The builder binds a card before allocating a rank's staging slab. Stubbed because, if missing,
 * the linker would pull in dispatch.cpp.o, whose strong rad_dev_alloc/rad_memcpy_async would
 * replace every weak fake above and run this test on the real HIP backend with a null stream. */
__attribute__((weak)) int rad_dev_set(int) { return RAD_OK; }

__attribute__((weak)) int rad_memcpy_async(void* d, const void* s, int64_t n, RadStream) {
    if (n > 0) std::memcpy(d, s, (size_t)n);
    return RAD_OK;
}
/* The builder zeroes the encoder rows of every rank but the first. */
__attribute__((weak)) int rad_memset_async(void* d, int v, int64_t n, RadStream) {
    if (n > 0) std::memset(d, v, (size_t)n);
    return RAD_OK;
}
__attribute__((weak)) int rad_memcpy_ranges_async(void* d, const void* s, int64_t prefix,
                                                  const RadCopyRange* r, int n, RadStream) {
    if (prefix > 0) std::memcpy(d, s, (size_t)prefix);
    for (int i = 0; i < n; ++i)
        std::memcpy((char*)d + r[i].dst, (const char*)s + r[i].src, (size_t)r[i].bytes);
    return RAD_OK;
}
__attribute__((weak)) int rad_memcpy_list_async(const RadCopySeg* g, int n, unsigned, RadStream) {
    for (int i = 0; i < n; ++i)
        if (g[i].bytes > 0) std::memcpy(g[i].dst, g[i].src, (size_t)g[i].bytes);
    return RAD_OK;
}
__attribute__((weak)) int rad_link_hold(RadStream, int) { return RAD_OK; }
__attribute__((weak)) void* rad_link_gate(RadStream) { return nullptr; }

static int g_fake_event;
__attribute__((weak)) int  rad_event_create(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_local(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_as(RadEvent* e, unsigned) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) void rad_event_destroy(RadEvent) {}
__attribute__((weak)) int  rad_event_record(RadEvent, RadStream) { return RAD_OK; }
__attribute__((weak)) int  rad_event_sync(RadEvent) { return RAD_OK; }

}  /* extern "C" */
#endif  /* SCHED_FAKES_KV_ONLY */

/* ================================================================== fake core/mem */
namespace sched_fake {
using namespace rad;

/* A small but faithful block manager: a free list per group, refcounted blocks (an adopted
 * prefix-cache block stays out of the pool while the cache names it), and per-sequence state slots
 * for the linear and conv groups. Shaped like core/mem/kv.h (sequence-keyed, owns the tables). */
struct FakeKV : IKVManager {
    std::vector<KVGroupInfo> gs;
    KVGeom geom;
    int64_t max_ctx = 0;

    std::vector<std::vector<int32_t>> freelist;    /* per group */
    std::vector<std::vector<int32_t>> refcnt;      /* per group, per block */
    std::vector<int64_t> total;
    std::vector<std::vector<int32_t>> state_free;  /* per group */
    std::vector<int64_t> n_state_slots;            /* per group, 0 if paged */

    struct Seq {
        std::vector<std::vector<int32_t>> blocks;  /* per group */
        std::vector<int32_t> state;                /* per group, -1 when not stateful */
        int64_t n = 0;
        /* Per group, the lowest table entry changed since take_table_changes last read it, kept
         * as the real manager keeps it, so these tests run the builder's partial restage. */
        mutable std::vector<int64_t> dirty;
    };
    std::unordered_map<uint64_t, Seq> seqs;
    static inline const std::vector<int32_t> kNone{};

    void setup(const std::vector<KVGroupInfo>& groups, int64_t ctx,
               int64_t blocks_per_paged_group, int64_t n_states) {
        gs = groups;
        geom = KVGeom::of(gs);
        max_ctx = ctx;
        freelist.assign(gs.size(), {});
        refcnt.assign(gs.size(), {});
        total.assign(gs.size(), 0);
        state_free.assign(gs.size(), {});
        n_state_slots.assign(gs.size(), 0);
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged) {
                total[i] = blocks_per_paged_group;
                refcnt[i].assign((size_t)blocks_per_paged_group, 0);
                for (int64_t b = blocks_per_paged_group - 1; b >= 0; --b)
                    freelist[i].push_back((int32_t)b);
            } else {
                n_state_slots[i] = n_states;
                for (int64_t k = n_states - 1; k >= 0; --k) state_free[i].push_back((int32_t)k);
            }
        }
    }

    /* One paged group's pool, resized before any sequence holds a block of it (tp8 prefill P0's window pools). */
    void resize_group(int g, int64_t blocks) {
        total[(size_t)g] = blocks;
        refcnt[(size_t)g].assign((size_t)blocks, 0);
        freelist[(size_t)g].clear();
        for (int64_t b = blocks - 1; b >= 0; --b) freelist[(size_t)g].push_back((int32_t)b);
    }
    void retain(int32_t g, int32_t b) { ++refcnt[(size_t)g][(size_t)b]; }
    void release(int32_t g, int32_t b) {
        if (--refcnt[(size_t)g][(size_t)b] == 0) freelist[(size_t)g].push_back(b);
    }
    int32_t grab(int32_t g) {
        int32_t b = freelist[(size_t)g].back();
        freelist[(size_t)g].pop_back();
        refcnt[(size_t)g][(size_t)b] = 1;
        return b;
    }

    int64_t rows_for(int gi, int64_t n) const {
        const KVGeom::G& e = geom.g[(size_t)gi];
        if (!e.paged) return 1;
        /* WINDOW/RING recycle past their window; the real manager's blocks_for caps the same way. */
        if ((e.kind == RAD_KV_WINDOW || e.kind == RAD_KV_RING) && e.window > 0 && n > e.window)
            n = e.window;
        return (n + e.block_size - 1) / e.block_size;
    }

    /* tp8 prefill P0: `slide` makes a fixed WINDOW group slide as KVManager's does (recycle_window): its table starts
     * at block keep_from / block - (window_blocks - 1), window_blocks = ceil(window / block) + 1, never past the keep
     * floor (set_keep_floor), and grows to the ensured end, so a test sees the blocks a sequence holds between its
     * floor and its planned end and the table's first position (first_block_pos). Off: the group holds at most its
     * window, as before (every other test). */
    bool slide = false;
    std::unordered_map<uint64_t, int64_t> keep_floor_of;
    std::unordered_map<uint64_t, std::vector<int64_t>> first_of;   /* per sequence, per group: the first block */
    bool slides(size_t gi) const { return slide && geom.g[gi].kind == RAD_KV_WINDOW && geom.g[gi].window > 0; }
    int64_t win_blocks(size_t gi) const {
        const KVGeom::G& e = geom.g[gi];
        return (e.window + e.block_size - 1) / e.block_size + 1;
    }
    int64_t& first_blk(uint64_t id, size_t gi) {
        std::vector<int64_t>& f = first_of[id];
        if (f.size() < gs.size()) f.resize(gs.size(), 0);
        return f[gi];
    }
    int64_t first_blk_c(uint64_t id, size_t gi) const {
        auto it = first_of.find(id);
        return it == first_of.end() || it->second.size() <= gi ? 0 : it->second[gi];
    }
    void recycle(uint64_t id, Seq& q, int64_t keep_from) {
        auto fl = keep_floor_of.find(id);
        if (fl != keep_floor_of.end() && fl->second >= 0 && keep_from > fl->second) keep_from = fl->second;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!slides(i)) continue;
            const int64_t bs = geom.g[i].block_size;
            int64_t& first = first_blk(id, i);
            const int64_t target = std::max<int64_t>(first, keep_from / bs - (win_blocks(i) - 1));
            while (first < target && !q.blocks[i].empty()) {
                release((int32_t)i, q.blocks[i].front());
                q.blocks[i].erase(q.blocks[i].begin());
                q.dirty[i] = 0;
                ++first;
            }
            if (q.blocks[i].empty()) first = target;
        }
    }
    /* Blocks group i's table needs to reach n tokens (slide mode: from its first block). */
    int64_t need_for(uint64_t id, const Seq& q, size_t i, int64_t n) const {
        if (!slides(i)) return rows_for((int)i, n) - (int64_t)q.blocks[i].size();
        const int64_t bs = geom.g[i].block_size;
        return (n + bs - 1) / bs - first_blk_c(id, i) - (int64_t)q.blocks[i].size();
    }
    void set_keep_floor(uint64_t id, int64_t pos) override {
        if (slide) keep_floor_of[id] = pos < 0 ? -1 : pos;
    }
    void trim_window(uint64_t id, int64_t keep_from) override {
        if (!slide) return;
        auto it = seqs.find(id);
        if (it == seqs.end() || keep_from < 0) return;
        recycle(id, it->second, std::min<int64_t>(keep_from, it->second.n));
    }
    int ensure_from(uint64_t id, int64_t n, int64_t keep_from) override {
        if (!slide) return ensure(id, n);
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        if (n <= it->second.n) return RAD_OK;
        if (keep_from < 0 || keep_from > it->second.n) keep_from = it->second.n;
        recycle(id, it->second, keep_from);
        return grow(id, n);
    }

    int add_sequence(uint64_t id) override {
        if (seqs.count(id)) return RAD_OK;
        Seq q;
        q.blocks.assign(gs.size(), {});
        /* Sized for the whole context here, on arrival, so a growing table does not reach the
         * allocator inside a step the scheduler tests count allocations in. */
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged) q.blocks[i].reserve((size_t)rows_for((int)i, max_ctx));
        q.state.assign(gs.size(), -1);
        q.dirty.assign(gs.size(), 0);
        seqs.emplace(id, std::move(q));
        return RAD_OK;
    }
    bool has_sequence(uint64_t id) const override { return seqs.count(id) != 0; }

    /* A runtime window (KVManager::set_window) of `ring_w` tokens on group 1, which the scheduler
     * arms every admitted sequence with; 0 declares none. Only the floor is recorded. */
    int64_t ring_w = 0;
    std::unordered_map<uint64_t, int64_t> ring_floor;
    int64_t runtime_window_tokens(int32_t g) const override { return g == 1 ? ring_w : 0; }
    int set_window(uint64_t id, int64_t lo, int64_t w) override {
        if (ring_w <= 0 || w != ring_w || !seqs.count(id)) return RAD_E_UNSUPPORTED;
        ring_floor[id] = lo;
        return RAD_OK;
    }
    bool has_runtime_window(uint64_t id) const override { return ring_floor.count(id) != 0; }

    void free_sequence(uint64_t id) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged)
                for (int32_t b : it->second.blocks[i]) release((int32_t)i, b);
            it->second.blocks[i].clear();
            if (it->second.state[i] >= 0) {
                state_free[i].push_back(it->second.state[i]);
                it->second.state[i] = -1;
            }
        }
        seqs.erase(it);
        ring_floor.erase(id);
        first_of.erase(id);
        keep_floor_of.erase(id);
    }

    int ensure(uint64_t id, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        /* slide mode: as KVManager::ensure with no keep_from (the sequence's length) */
        if (slide && n > it->second.n) recycle(id, it->second, it->second.n);
        return grow(id, n);
    }
    int grow(uint64_t id, int64_t n) {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        Seq& q = it->second;
        /* All or nothing, as KVManager: a partial grow would leave the sequence holding blocks it
         * cannot use, and the caller unable to tell whether preempting would help. A state slot
         * is checked here too, before any block is taken. */
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) {
                if (q.state[i] < 0 && state_free[i].empty()) return RAD_E_FULL;
                continue;
            }
            int64_t need = need_for(id, q, i, n);
            if (need > 0 && need > (int64_t)freelist[i].size()) return RAD_E_FULL;
        }
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged) {
                int64_t need = need_for(id, q, i, n);
                if (need > 0) q.dirty[i] = std::min<int64_t>(q.dirty[i], (int64_t)q.blocks[i].size());
                for (int64_t k = 0; k < need; ++k) q.blocks[i].push_back(grab((int32_t)i));
            } else if (q.state[i] < 0) {
                if (state_free[i].empty()) return RAD_E_FULL;
                q.state[i] = state_free[i].back();
                state_free[i].pop_back();
                q.blocks[i].assign(1, q.state[i]);
            }
        }
        if (n > q.n) q.n = n;
        return RAD_OK;
    }

    int adopt(uint64_t id, int32_t g, const std::vector<int32_t>& blocks, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        if (n % geom.g[(size_t)g].block_size != 0) return RAD_E_INVAL;
        Seq& q = it->second;
        q.blocks[(size_t)g] = blocks;
        q.dirty[(size_t)g] = 0;
        for (int32_t b : blocks) retain(g, b);
        if (n > q.n) q.n = n;
        return RAD_OK;
    }

    /* fn4 preempt_keep, as KVManager::preempt_keep / rejoin: the whole blocks go, the partial last block and the states
     * stay; rejoin puts a hit's whole blocks back in front */
    std::unordered_map<uint64_t, int64_t> kept;
    int64_t preempt_keep(uint64_t id, int64_t n_valid) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        const int64_t nv = std::min<int64_t>(it->second.n, n_valid);
        int64_t whole = -1;
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged) {
                const int64_t bs = geom.g[i].block_size, w = nv / bs * bs;
                whole = whole < 0 ? w : std::min(whole, w);
            }
        if (whole < 0) whole = 0;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) continue;
            auto& bt = it->second.blocks[i];
            const int64_t nb = std::min<int64_t>(whole / geom.g[i].block_size, (int64_t)bt.size());
            for (int64_t b = 0; b < nb; ++b) release((int32_t)i, bt[(size_t)b]);
            bt.erase(bt.begin(), bt.begin() + nb);
            it->second.dirty[i] = 0;
        }
        kept[id] = whole;
        return whole;
    }
    int rejoin(uint64_t id, int32_t g, const std::vector<int32_t>& whole) override {
        auto it = seqs.find(id);
        auto k = kept.find(id);
        if (it == seqs.end() || k == kept.end()) return RAD_E_STATE;
        if ((int64_t)whole.size() != k->second / geom.g[(size_t)g].block_size) return RAD_E_INVAL;
        for (int32_t b : whole) retain(g, b);
        auto& bt = it->second.blocks[(size_t)g];
        bt.insert(bt.begin(), whole.begin(), whole.end());
        it->second.dirty[(size_t)g] = 0;
        bool all = true;
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged) {
                const int64_t bs = geom.g[i].block_size;
                if ((int64_t)it->second.blocks[i].size() != (it->second.n + bs - 1) / bs) all = false;
            }
        if (all) kept.erase(k);
        return RAD_OK;
    }

    int preempt(uint64_t id) override {
        kept.erase(id);
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged)
                for (int32_t b : it->second.blocks[i]) release((int32_t)i, b);
            it->second.blocks[i].clear();
            it->second.dirty[i] = 0;
            if (it->second.state[i] >= 0) {
                state_free[i].push_back(it->second.state[i]);
                it->second.state[i] = -1;
            }
        }
        it->second.n = 0;
        first_of.erase(id);
        return RAD_OK;
    }

    int rollback(uint64_t id, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        /* Nothing to give back: no entry changes, so none is marked (as KVManager). */
        if (n >= it->second.n) return RAD_OK;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) continue;   /* linear state is NOT rolled back (spec §10) */
            int64_t keep = slides(i) ? std::max<int64_t>(0, (n + geom.g[i].block_size - 1) / geom.g[i].block_size -
                                                                first_blk(id, i))
                                     : rows_for((int)i, n);
            std::vector<int32_t>& bl = it->second.blocks[i];
            while ((int64_t)bl.size() > keep) { release((int32_t)i, bl.back()); bl.pop_back(); }
            it->second.dirty[i] = std::min<int64_t>(it->second.dirty[i], (int64_t)bl.size());
        }
        it->second.n = n;
        return RAD_OK;
    }

    bool can_ever_fit(int64_t n) const override {
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged && rows_for((int)i, n) > total[i]) return false;
        return true;
    }
    bool can_grow(uint64_t id, int64_t n) const override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return false;
        const Seq& q = it->second;
        if (n <= q.n) return true;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged) {
                int64_t need = rows_for((int)i, n) - (int64_t)q.blocks[i].size();
                if (need > 0 && need > (int64_t)freelist[i].size()) return false;
            } else if (!geom.g[i].paged && q.state[i] < 0 && state_free[i].empty()) {
                return false;
            }
        }
        return true;
    }
    /* IKVManager::release_idle_states. A test leaks a slot by clearing state_free; the next idle
     * overdue step must put it back rather than wait forever. */
    int release_idle_states() override {
        int n = 0;
        for (auto& kv : seqs) {
            for (size_t i = 0; i < gs.size(); ++i) {
                if (geom.g[i].paged || kv.second.state[i] < 0) continue;
                state_free[i].push_back(kv.second.state[i]);
                kv.second.state[i] = -1;
                ++n;
            }
        }
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged || n_state_slots[i] <= 0) continue;
            std::vector<char> seen((size_t)n_state_slots[i], 0);
            for (int32_t s : state_free[i])
                if (s >= 0 && s < n_state_slots[i]) seen[(size_t)s] = 1;
            for (int64_t s = 0; s < n_state_slots[i]; ++s) {
                if (seen[(size_t)s]) continue;
                state_free[i].push_back((int32_t)s);
                ++n;
            }
        }
        return n;
    }
    int64_t available_blocks(int32_t g) const override {
        return g >= 0 && g < (int32_t)freelist.size() ? (int64_t)freelist[(size_t)g].size() : 0;
    }

    const std::vector<int32_t>& block_table(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        return it == seqs.end() ? kNone : it->second.blocks[(size_t)g];
    }
    int32_t state_slot(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        return it == seqs.end() ? -1 : it->second.state[(size_t)g];
    }
    /* Columns per sequence in a state group's index row. One is the state itself; two adds a
     * second slot beside it, 1000 + the state's, as a kernel keeping scratch beside its state
     * would ask for. A paged group always has one. */
    int64_t sidx_width = 1;
    int64_t state_index_width(int32_t g) const override {
        return geom.g[(size_t)g].paged ? 1 : sidx_width;
    }
    int state_index_row(uint64_t id, int32_t g, int32_t* out, int max) const override {
        if (!out || max <= 0) return 0;
        const int32_t s = state_slot(id, g);
        const int n = (int)std::min<int64_t>(state_index_width(g), max);
        for (int j = 0; j < n; ++j) out[j] = (s < 0 || j == 0) ? s : 1000 * j + s;
        return n;
    }
    int64_t first_block_pos(uint64_t id, int32_t g) const override {
        return slides((size_t)g) ? first_blk_c(id, (size_t)g) * geom.g[(size_t)g].block_size : 0;
    }
    int64_t take_table_changes(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return 0;
        const int64_t d = it->second.dirty[(size_t)g];
        it->second.dirty[(size_t)g] = INT64_MAX;
        return d;
    }
    /* Snapshot slots the store has given back; every other slot is held. */
    std::vector<int32_t> ck_gone;
    bool checkpoint_held(int32_t slot) const override {
        return std::find(ck_gone.begin(), ck_gone.end(), slot) == ck_gone.end();
    }
    /* Blocks an elastic pool has handed back to the card: carved and free, but not available
     * until the pool grows again. free_blocks() is bounded by that and held_blocks() is not. */
    int64_t unbacked = 0;
    int64_t free_blocks(int32_t g) const override {
        const int64_t n = (int64_t)freelist[(size_t)g].size() - unbacked;
        return n > 0 ? n : 0;
    }
    int64_t total_blocks(int32_t g) const override { return total[(size_t)g]; }
    int64_t held_blocks(int32_t g) const override {
        return total[(size_t)g] - (int64_t)freelist[(size_t)g].size();
    }
    /* A byte size per object, so the sum is checkable. The real manager reads these off the
     * carved plan; here the test sets them, and a group left at zero contributes nothing. */
    std::vector<int64_t> block_bytes, state_bytes;
    int64_t pool_bytes(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? total[(size_t)g] * bb(g)
                        : (int64_t)(state_free[(size_t)g].size() + n_states_out(g)) * sb(g);
    }
    int64_t pool_bytes_used(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? (total[(size_t)g] - free_blocks(g)) * bb(g) : n_states_out(g) * sb(g);
    }
    /* This pool is not elastic, so what it was carved to address is what it is holding. */
    int64_t pool_bytes_carved(int32_t g) const override { return pool_bytes(g); }
    int64_t bb(int32_t g) const {
        return g < (int32_t)block_bytes.size() ? block_bytes[(size_t)g] : 0;
    }
    int64_t sb(int32_t g) const {
        return g < (int32_t)state_bytes.size() ? state_bytes[(size_t)g] : 0;
    }
    /* States handed out, which the fake tracks only through the sequences holding them. */
    int64_t n_states_out(int32_t g) const {
        int64_t n = 0;
        for (const auto& kv : seqs) if (kv.second.state[(size_t)g] >= 0) ++n;
        return n;
    }
    int64_t block_tokens(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? gg.block_size : 0;
    }
    /* A streamed group's view, as KVManager::stream_view returns it. Tests install fake
     * addresses to check that the builder rebases them into each rank's pool; with `streams`
     * false the fake has none, which is every other scheduler test. */
    RadKVStream stream{};
    bool        streams = false;
    const RadKVStream* stream_view(int32_t g) const override {
        return streams && g == 0 ? &stream : nullptr;
    }
    /* Groups a test says the window snapshot carries (KVGroupPlan::win_snap): a checkpoint slot
     * then holds their window, and a hit at a snapshot seeds them (nothing to copy here). */
    std::vector<int32_t> snap;
    /* False: the snapshot is taken but window_snapshot() does not report its groups. */
    bool snap_reported = true;
    bool window_snapshot(int32_t g) const override {
        return snap_reported && std::find(snap.begin(), snap.end(), g) != snap.end();
    }
    bool has_window_snapshot() const override { return !snap.empty(); }
    int  seed_window(uint64_t, int64_t) override { return snap.empty() ? -1 : 0; }
};

struct FakePC : IPrefixCache {
    FakeKV* kv = nullptr;
    int32_t attn = 0, lin = 0, src = -1;
    int32_t branch_at = 0;       /* CacheHit::branch_at, as the real cache reports it */
    int64_t drafter_floor = -1;  /* set_drafter_floor's last value, -1 never called */
    int32_t next_slot = 0;
    int32_t reserved = 0;
    int64_t evicted = 0, ckpt_evicted = 0;
    int64_t held = 0;            /* what the cache would report holding now */
    int64_t interval = 256;
    std::vector<int32_t> groups;                     /* the paged groups this cache indexes */
    std::vector<std::vector<int32_t>> seeded;        /* blocks the cache holds a reference on */

    /* Pretend an earlier request filled the cache: take `n_tokens` worth of blocks out of the pool
     * and hold a reference on them, which is what makes a hit safe (a block the cache names is
     * never in the free list). */
    void seed(int32_t n_tokens) {
        attn = n_tokens;
        seeded.assign(groups.size(), {});
        for (size_t i = 0; i < groups.size(); ++i) {
            int32_t g = groups[i];
            int64_t nb = n_tokens / kv->geom.g[(size_t)g].block_size;
            for (int64_t b = 0; b < nb; ++b) seeded[i].push_back(kv->grab(g));
        }
    }

    std::vector<uint64_t> restores;
    std::vector<bool> restores_whole;     /* each call's whole_only */
    bool decline_whole = false;           /* a whole_only restore is declined */
    RestoreResult restore_result = RestoreResult::Whole;   /* any other restore */
    RestoreResult restore(const Request& r, bool whole_only) override {
        restores.push_back(r.id);
        restores_whole.push_back(whole_only);
        return whole_only && decline_whole ? RestoreResult::Declined : restore_result;
    }

    mutable int64_t lookups = 0;
    /* A cap (HitCap) as prefix.cpp applies one: attention ends at a block at or below it, the
     * linear state at the deepest checkpoint under that (checkpoints at multiples of `interval`,
     * none below `ckpt_lo`), and the whole hit stands if that gives back too much. */
    int64_t ckpt_lo = 0;
    CacheHit lookup(const Request&, std::vector<std::vector<int32_t>>* out,
                    const HitCap* cap) const override {
        ++lookups;
        if (out) {
            out->resize(groups.size());
            for (size_t i = 0; i < groups.size() && i < seeded.size(); ++i)
                (*out)[i] = seeded[i];
        }
        CacheHit h{ attn, lin, src };
        h.branch_at = branch_at;
        /* fn4 resume_cold (HitCap::resume_at): the deepest allowed position at or below the snapshot where one is held
         * -- this fake holds one at `lin` and at every multiple of `interval` below it (from ckpt_lo) */
        if (cap && cap->resume_at && src >= 0) {
            int64_t l = 0;
            for (int64_t p : *cap->resume_at)
                if (p > 0 && p <= lin && (p == lin || (p % interval == 0 && p >= ckpt_lo))) l = p;
            h.n_linear_tokens = (int32_t)l;
            h.checkpoint_slot = l > 0 ? src : -1;
        }
        const int64_t whole = src >= 0 ? lin : 0;
        if (cap && cap->tokens >= 0 && attn > cap->tokens / 16 * 16 && whole > cap->tokens) {
            const int64_t a = cap->tokens / 16 * 16;
            int64_t l = std::min<int64_t>(lin, a) / interval * interval;
            if (l < ckpt_lo) l = 0;
            if (whole - l <= cap->max_give_back) {
                h.n_attn_tokens = (int32_t)a;
                h.n_linear_tokens = (int32_t)l;
                h.checkpoint_slot = l > 0 ? src : -1;
                h.given_back = (int32_t)(whole - l);
                if (out)
                    for (auto& b : *out)
                        if ((int64_t)b.size() > a / 16) b.resize((size_t)(a / 16));
            }
        }
        return h;
    }
    /* qfn-on-ds lever SRB (round 58): IPrefixCache::probe -- the hit a test sets per request id (-1 = the cache cannot say) */
    std::map<uint64_t, int64_t> probe_hit;
    mutable int64_t probes = 0;
    int64_t probe(const Request& r) const override {
        ++probes;
        const auto it = probe_hit.find(r.id);
        return it == probe_hit.end() ? -1 : it->second;
    }
    const std::vector<int32_t>& cached_groups() const override { return groups; }
    void set_drafter_floor(int64_t tokens) override { drafter_floor = tokens; }

    /* What the scheduler published and asked to be evicted, recorded rather than acted on: these
     * tests check that the scheduler publishes before it frees and evicts before it preempts, not
     * the cache's own indexing, which prefix.cpp's tests cover. */
    struct Published { uint64_t seq; int64_t n_tokens; size_t n_groups; };
    std::vector<Published> published;
    int64_t evict_calls = 0, evict_grant = 0;

    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        published.push_back({ r.id, n_tokens, blocks.size() });
        return RAD_OK;
    }
    /* Grants only what a test asked for, so "the cache had nothing to give" is reachable and the
     * preemption path below it still runs. */
    int64_t evict(int64_t n) override {
        ++evict_calls;
        if (evict_grant <= 0) return 0;
        int64_t took = n < evict_grant ? n : evict_grant;
        if (evict_frees) {
            /* An entry is one block in every cached group: give that many back to the pool. */
            int64_t have = seeded.empty() ? 0 : (int64_t)seeded[0].size();
            took = std::min(took, have);
            for (size_t i = 0; i < groups.size() && i < seeded.size(); ++i)
                for (int64_t k = 0; k < took; ++k) {
                    kv->release(groups[i], seeded[i].back());
                    seeded[i].pop_back();
                }
        }
        evict_grant -= took;
        evicted += took;
        return took;
    }
    bool evict_frees = false;    /* evict() returns the seeded blocks to the pool */
    std::vector<uint64_t> holds, releases;
    void hold(const Request& r) override { holds.push_back(r.id); }
    void release(const Request& r) override { releases.push_back(r.id); }
    /* What the scheduler asked to be released, recorded; the answer is canned. */
    struct Released { std::string label; uint64_t id; int64_t keep_tokens; };
    std::vector<Released> released_calls;
    int release_rc = RAD_OK;
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionRelease* out) override {
        released_calls.push_back({ label, id, keep_tokens });
        if (out) { *out = SessionRelease{}; out->sessions = 1; out->entries = 5; out->tokens = 80; }
        return release_rc;
    }
    /* What the reserve asked for, call by call. */
    std::vector<int64_t> trims;
    int64_t trim(int64_t n) override {
        trims.push_back(n);
        return 0;
    }
    /* Cumulative, like the real cache's: the scheduler publishes it as a counter and the server
     * takes deltas, so a per-call value would read as a cache that never evicts. */
    int64_t evictions() const override { return evicted; }
    int64_t ckpt_evictions() const override { return ckpt_evicted; }
    int64_t held_tokens() const override { return held; }
    int64_t next_checkpoint_after(int64_t pos) const override {
        return ((pos / interval) + 1) * interval;
    }
    std::vector<int64_t> committed;
    void commit_checkpoint(const Request&, int64_t pos) override { committed.push_back(pos); }
    int reserve_checkpoint(const Request&, int64_t, int32_t* slot_out) override {
        ++reserved;
        *slot_out = next_slot++;
        return RAD_OK;
    }
};

}  /* namespace sched_fake */
