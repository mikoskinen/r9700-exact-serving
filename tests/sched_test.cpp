/* sched_test.cpp -- the scheduler, chunker, step batch, derivations and draft controller, against
 * fakes for core/mem and the device layer (sched_fakes.h).
 */
#include "rad_test.h"

#include "mm/processor.h"
#include "sched/mtp_couple_bind.h"
#include "sched/scheduler.h"
#include "sched/lanes.h"
#include "sched/tcopy.h"
#include "runtime/fn4_switch.h"
#include "sched/advance_impl.h"
#include "sched/derive.h"
#include "server/sink.h"

#include <algorithm>
#include <map>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <new>
#include <string>

#include <unistd.h>

using namespace rad;

/* ================================================================== allocation counter */
/* The step path must not allocate. Counting through a replaced global operator new is the only
 * reliable check: review can miss a vector that grows on the ten-thousandth request, which shows
 * up as an unreproducible latency spike. */
static long g_allocs = 0;
static bool g_counting = false;

/* malloc/free is a matched pair for a replaced global new/delete, but GCC's
 * -Wmismatched-new-delete does not model the replacement and reports every inlined delete in this
 * translation unit as a free() on a pointer that came from new. */
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(size_t n) {
    if (g_counting) ++g_allocs;
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](size_t n) { return operator new(n); }
/* The nothrow forms too (std::get_temporary_buffer uses them): left to the runtime, their blocks
 * would reach the free() below, a mismatch under ASan, and their allocations would go uncounted. */
void* operator new(size_t n, const std::nothrow_t&) noexcept {
    if (g_counting) ++g_allocs;
    return std::malloc(n ? n : 1);
}
void* operator new[](size_t n, const std::nothrow_t& t) noexcept { return operator new(n, t); }
void  operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void  operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete(void* p, size_t) noexcept { std::free(p); }
void  operator delete[](void* p, size_t) noexcept { std::free(p); }

#include "sched_fakes.h"

using namespace sched_fake;

namespace {

/* ------------------------------------------------------------------ a declared program */
/* A paged group beside a linear one: the chunk geometry reconciles the two, so a second paged
 * group would add nothing.
 *
 * The op carries a schema because that is where the chunk parameter is identified
 * (RAD_PROLE_SEQ_CHUNK). The geometry and the kernel's constraint must name the same key, since a
 * constraint on a key the geometry lacks does not hold (rad_abi.h). */
static const RadParamSpec g_gdn_ps[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "chunk", RAD_P_INT, RAD_DERIVED,  RAD_PROLE_SEQ_CHUNK },
};
static const RadOpSchema g_gdn_schema = { "gdn_chunk_scan", g_gdn_ps, 2, nullptr, 0, "test" };
static RadConstraint g_gdn_c[] = { RAD_CDIV("chunk", 64) };
static RadKernelInfo g_gdn_ki;
static KernelRow     g_gdn_row;

struct Fixture {
    Program prog;
    Config  cfg;
    FakeKV  kv;
    FakePC  pc;
    Scheduler sch;
    std::vector<void*> owned;

    ~Fixture() { sch.fini(); for (void* p : owned) std::free(p); }

    void build(int64_t blocks = 4096, int64_t states = 64, int64_t max_tok = 512,
               int64_t max_ctx = 4096, bool windowed = false, bool linear = true) {
        g_gdn_ki = RadKernelInfo{};
        g_gdn_ki.name = "gdn_fake";
        g_gdn_ki.op = "gdn_chunked";
        g_gdn_ki.constraints = g_gdn_c;
        g_gdn_ki.n_constraints = 1;
        g_gdn_row = KernelRow{ &g_gdn_ki, "fake", 0, 0 };

        prog.meta.n_ctx_train = 4096;
        /* Above the 1000 + i the requests below spell: admission refuses ids past the
         * vocabulary, and these prompts must be admissible. */
        prog.meta.n_vocab = 65536;

        /* index 0 is the sentinel every handle table in Program carries */
        prog.ops.emplace_back();
        prog.buffers.emplace_back();

        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        prog.kv_groups.push_back(attn);

        /* `linear` false: a full-attention trunk, as Gemma 4's (with `windowed`, its sliding
         * layers), and no GDN op or state buffers below. */
        if (linear) {
            KVGroupInfo lin;
            lin.name = "gdn";
            lin.decl.kind = RAD_KV_LINEAR;
            lin.decl.state_dim[0] = 128; lin.decl.state_dim[1] = 128;
            prog.kv_groups.push_back(lin);

            KVGroupInfo conv;
            conv.name = "conv";
            conv.decl.kind = RAD_KV_CONV;
            conv.decl.conv_width = 4;
            prog.kv_groups.push_back(conv);
        }

        /* A windowed paged group, as sliding-window checkpoints pair with a full group. Not
         * prefix-cached: prefix.cpp indexes only groups whose blocks below the window stay put.
         * Tests wanting a prefix cache over a WINDOW+FULL pair trim pc.groups to the cached ones. */
        if (windowed) {
            KVGroupInfo win;
            win.name = "swa";
            win.decl.kind = RAD_KV_WINDOW;
            win.decl.window = 64;
            win.block_size = 16;
            win.layers = { 2 };   /* bound to a trunk layer, the group a trunk attention reads;
                                   * a drafter's own group is bound at the trunk's layer count and
                                   * above (rad_bind_layer_kv at the drafter declare) */
            prog.kv_groups.push_back(win);
            prog.meta.n_layers = 16;   /* trunk layers 0..15; the drafter tests' appended group
                                        * stays unbound, and unbound is not trunk-owned */
        }

        if (linear) {
            OpInfo op;
            op.op = "gdn_chunk_scan";
            op.schema = &g_gdn_schema;
            Band b;
            b.hi = 4096;
            b.dom[RAD_DOMAIN_DEVICE].row = &g_gdn_row;
            /* The architecture plugin's declared geometry, in arch/common/rad_block_gdn.h's
             * spelling, the key its schema marks RAD_PROLE_SEQ_CHUNK. Both this and the kernel's
             * constraint contribute. */
            b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
            op.bands.push_back(b);
            prog.ops.push_back(op);

            add_derived("seq_start", "seq_start_index", 257);
            add_derived("chunk_bounds", "state_chunk_bounds", 4097);
            add_derived("conv_idx", "conv_state_index", 256);
        }

        cfg.max_tok = max_tok;
        cfg.max_seqs = 32;
        cfg.max_ctx = max_ctx;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        cfg.prefill_room_gate = false;
        /* The fixture runs --prefix-resume nearest (a tail snapshot at the last quantum
         * boundary, a hit at any snapshot or block); the exact-resume tests set prefix_exact
         * (--prefix-resume exact, the engine's default) themselves. */
        cfg.prefix_exact = false;
        /* Config::n_spec defaults to -1 (`auto`), which engine_bringup resolves; this fixture
         * skips that, and DraftController::configure refuses a negative depth. Speculation tests
         * set their own depth. */
        cfg.n_spec = 0;

        kv.setup(prog.kv_groups, cfg.max_ctx, blocks, states);
        pc.kv = &kv;
        pc.groups.clear();
        for (size_t i = 0; i < prog.kv_groups.size(); ++i)
            if (kv.geom.g[i].paged) pc.groups.push_back((int32_t)i);
    }

    void add_derived(const char* name, const char* derive, int64_t n, rad_kvgroup kv = 0) {
        BufferInfo b;
        b.name = name;
        b.decl.dtype = RAD_I32;
        b.decl.rank = 1;
        b.decl.shape[0] = n;
        b.decl.kind = RAD_BUF_DERIVED;
        b.decl.derive = derive;
        b.decl.kv = kv;
        b.bytes = n * 4;
        void* p = std::calloc(1, (size_t)b.bytes);
        owned.push_back(p);
        b.ptr = p;
        prog.buffers.push_back(b);
    }

    int start() {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio);
    }

    const int32_t* derived(const char* name) const {
        for (const BufferInfo& b : prog.buffers)
            if (b.name == name) return (const int32_t*)b.ptr;
        return nullptr;
    }

    uint64_t next_id = 1;
    /* A request for prompt tokens 1000, 1001, ... */
    std::unique_ptr<Request> request(int n_prompt, int max_tokens) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->prompt.resize((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) r->prompt[(size_t)i] = 1000 + i;
        return r;
    }
    /* Its id, or 0 when the scheduler refused it. */
    uint64_t add(std::unique_ptr<Request> r) {
        const uint64_t id = r->id;
        return sch.add(std::move(r)) < 0 ? 0 : id;
    }

    uint64_t submit(int n_prompt, int max_tokens = 64, int priority = 0, bool ignore_eos = false) {
        auto r = request(n_prompt, max_tokens);
        r->priority = priority;
        r->ignore_eos = ignore_eos;
        return add(std::move(r));
    }

    /* One whose prompt carries a picture the encoder has not run yet: it waits in the queue until
     * `*item` says it is encoded. */
    uint64_t submit_unencoded(int n_prompt, int max_tokens, std::shared_ptr<mm::Item>* item) {
        auto r = request(n_prompt, max_tokens);
        *item = std::make_shared<mm::Item>();
        r->media = std::make_shared<mm::PromptMedia>();
        r->media->items.push_back(mm::Placed{*item, 0});
        return add(std::move(r));
    }

    /* The same, with the server's sink attached, for what a request tells its client. */
    uint64_t submit_to(server::Sink* sink, int n_prompt, int max_tokens) {
        auto r = request(n_prompt, max_tokens);
        r->sink = sink;
        return add(std::move(r));
    }

    /* Commit the planned step: every entry sampled `token`, ended when `eos`, and accepted
     * `acc(entry)` drafts (none when `acc` is empty). */
    int commit(int32_t token, uint8_t eos = 0,
               const std::function<int32_t(const StepEntry&)>& acc = nullptr) {
        const StepPlan& p = sch.plan();
        std::vector<int32_t> a(p.e.size(), 0), t(p.e.size(), token);
        std::vector<uint8_t> e(p.e.size(), eos);
        if (acc)
            for (size_t i = 0; i < p.e.size(); ++i) a[i] = acc(p.e[i]);
        Scheduler::StepResult r;
        r.n_seq = (int)p.e.size();
        r.n_accepted = a.data();
        r.token = t.data();
        r.eos = e.data();
        return sch.commit(r);
    }

    /* Commit a step in which every sequence produced one token and none ended. */
    void commit_plain(int32_t token = 7) { CHECK_OK(commit(token)); }
};

/* A uniformly random number of `e`'s drafts accepted, 0..n_spec, from an LCG. */
int32_t random_accept(uint32_t& rng, const StepEntry& e) {
    rng = rng * 1664525u + 1013904223u;
    return e.n_spec > 0 ? (int32_t)((rng >> 16) % (uint32_t)(e.n_spec + 1)) : 0;
}

}  /* namespace */

/* ================================================================== the fakes are linked */
/* Every check below reads device memory as host memory through the weak fakes. One device call
 * core/ makes and the fakes lack pulls the real backend's dispatch.cpp.o into the link, which
 * replaces every fake; the binary then still runs, on another backend, and proves something
 * else. */
TEST(the_device_fakes_are_the_ones_linked) {
    g_fake_device = false;
    void* p = rad_dev_alloc(16, RAD_MEM_DEVICE);
    CHECK(p != nullptr);
    CHECK(g_fake_device);
    rad_dev_free(p, RAD_MEM_DEVICE);
}

/* ================================================================== geometry + chunking */
TEST(chunk_geometry_is_computed_not_configured) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    const ChunkGeometry& g = f.sch.geometry();
    CHECK_EQ(g.attn_block, 16);           /* off the paged group */
    CHECK_EQ(g.state_chunk, 64);          /* off the resolved GDN kernel's DIV constraint */
    CHECK_EQ(g.quantum, 64);              /* lcm */
    CHECK_EQ(g.checkpoint_interval, 256); /* already a multiple of the quantum */
    CHECK_EQ(g.min_max_tok, 64);
}

TEST(chunk_geometry_rounds_the_checkpoint_interval_up_to_a_legal_chunk) {
    Fixture f;
    f.build();
    f.cfg.checkpoint_interval = 200;      /* not a multiple of 64 */
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().checkpoint_interval, 256);
}

/* A model with no linear state checkpoints only for a trunk window: the snapshot of the window
 * behind a prompt's tail is what a prefix hit resumes from. A drafter's own window alone takes no
 * checkpoint and leaves the hit at any block: draft refill recomputes it. */
TEST(chunk_geometry_snapshots_a_trunk_window_and_not_a_drafters) {
    for (int drafter = 0; drafter < 2; ++drafter) {
        Program prog;
        prog.meta.n_layers = 16;
        prog.kv_groups.emplace_back();
        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        attn.layers = { 0 };
        prog.kv_groups.push_back(attn);
        KVGroupInfo win;
        win.name = "swa";
        win.decl.kind = RAD_KV_WINDOW;
        win.decl.window = 64;
        win.block_size = 16;
        win.layers = { drafter ? 16 : 1 };
        win.drafter = drafter != 0;
        prog.kv_groups.push_back(win);
        Config cfg;
        cfg.max_tok = 512;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        ChunkGeometry g;
        CHECK_OK(chunk_geometry_resolve(prog, cfg, &g));
        CHECK_EQ(g.checkpoint_interval, drafter ? 0ll : 256ll);
        CHECK(g.window_ckpt == !drafter);
        /* Bring-up asks the same before the geometry is resolved (window_snapshots), and puts a
         * block drafter's window in the snapshot only where it answers yes: a full-attention
         * trunk takes no snapshot for its drafter's window alone. */
        CHECK(window_snapshots(prog, cfg) == g.window_ckpt);
        cfg.prefix_cache = false;
        CHECK(!window_snapshots(prog, cfg));
    }
}

TEST(chunk_geometry_refuses_a_budget_it_cannot_serve) {
    Fixture f;
    f.build();
    f.cfg.max_tok = 32;                   /* below one quantum */
    log_set_level(Log::Error);
    CHECK_EQ(f.start(), RAD_E_INVAL);
}

/* Gemma-like FULL + WINDOW, with no recurrent state. A prompt-tail snapshot alone
 * cannot serve a request whose last few hundred tokens changed: it is past the
 * divergence. Keep periodic snapshots as well, so lookup can resume below it. */
TEST(window_prefill_writes_interval_snapshots_before_the_tail) {
    for (const char* policy : { "geometric-backoff", "keep-all" }) {
        Fixture f;
        f.build(4096, 64, 512, 4096, /*windowed=*/true);
        f.prog.kv_groups.erase(f.prog.kv_groups.begin() + 1, f.prog.kv_groups.begin() + 3);
        f.prog.ops.resize(1);             /* remove the recurrent scan and its chunk constraint */
        f.prog.buffers.resize(1);         /* its derived conv/state indices do not apply either */
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        f.pc.groups = { 0 };
        f.cfg.checkpoint_policy = policy;
        REQUIRE_EQ(f.start(), RAD_OK);
        CHECK(f.sch.geometry().window_ckpt);
        CHECK(f.sch.geometry().window_periodic);

        int64_t pos = 0;
        std::vector<int64_t> snapshots;
        while (pos < 1100) {
            int64_t ckpt = -1;
            const int64_t n = f.sch.plan_chunk(pos, 1100, 512, &ckpt);
            REQUIRE(n > 0);
            pos += n;
            if (ckpt > 0) {
                /* An interval snapshot ends its chunk; the tail's rides inside the last one. */
                CHECK(ckpt == pos || pos == 1100);
                CHECK(ckpt > pos - n);
                snapshots.push_back(ckpt);
            }
        }
        const std::vector<int64_t> expected = { 256, 512, 768, 1024, 1072 };
        CHECK(snapshots == expected);

        /* An aligned prompt must still snapshot before its last block: that block's
         * generation marker changes when the client replays the assistant reply. The snapshot
         * is taken inside the chunk, which is not split for it. */
        int64_t ckpt = -1;
        CHECK_EQ(f.sch.plan_chunk(768, 1024, 512, &ckpt), 256ll);
        CHECK_EQ(ckpt, 1008ll);
        /* A shared step may end before the boundary; the next chunk must reach it. */
        CHECK_EQ(f.sch.plan_chunk(0, 1100, 160, &ckpt), 160ll);
        CHECK_EQ(ckpt, -1ll);
        CHECK_EQ(f.sch.plan_chunk(160, 1100, 160, &ckpt), 96ll);
        CHECK_EQ(ckpt, 256ll);
    }
}

/* tip-only is a window snapshot opt-out, not just retention after paying for
 * copies which will be discarded. Match the original unsplit prefill schedule. */
TEST(tip_only_window_prefill_writes_only_the_tail_snapshot) {
    Fixture f;
    f.build(4096, 64, 512, 4096, /*windowed=*/true);
    f.prog.kv_groups.erase(f.prog.kv_groups.begin() + 1, f.prog.kv_groups.begin() + 3);
    f.prog.ops.resize(1);
    f.prog.buffers.resize(1);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
    f.cfg.checkpoint_policy = "tip-only";
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK(f.sch.geometry().window_ckpt);
    CHECK(!f.sch.geometry().window_periodic);

    int64_t ckpt = -1;
    CHECK_EQ(f.sch.plan_chunk(0, 1100, 512, &ckpt), 512ll);
    CHECK_EQ(ckpt, -1ll);
    CHECK_EQ(f.sch.plan_chunk(512, 1100, 512, &ckpt), 512ll);
    CHECK_EQ(ckpt, -1ll);
    CHECK_EQ(f.sch.plan_chunk(1024, 1100, 512, &ckpt), 76ll);
    CHECK_EQ(ckpt, 1072ll);
    CHECK_EQ(f.sch.plan_chunk(1072, 1100, 512, &ckpt), 28ll);
    CHECK_EQ(ckpt, -1ll);

    CHECK_EQ(f.sch.plan_chunk(768, 1024, 512, &ckpt), 256ll);
    CHECK_EQ(ckpt, 1008ll);
    CHECK_EQ(f.sch.plan_chunk(0, 256, 512, &ckpt), 256ll);
    CHECK_EQ(ckpt, -1ll);
    /* Explicit request checkpoints must still be honored. */
    CHECK_EQ(f.sch.plan_chunk(0, 1100, 512, &ckpt, 256), 256ll);
    CHECK_EQ(ckpt, 256ll);

    f.submit(1100, 1);
    for (int i = 0; i < 8 && f.sch.step(); ++i) f.commit_plain();
    CHECK_EQ(f.pc.reserved, 1);
    CHECK_EQ(f.sch.metrics().checkpoints_written, 1ll);
}

/* A hit that ends at a head other prompts share (CacheHit::branch_at) and resumes nothing there:
 * the request takes a snapshot at that point (Request::checkpoint_at), so the requests after it
 * skip the head. TranslateGemma's chunks: the same instructions, then a different text each. Not
 * when a snapshot already resumes within Scheduler::kSharedHeadMin of it, nor for an ordinary hit. */
TEST(a_hit_at_a_shared_head_takes_a_snapshot_there) {
    {   /* a window model, tip-only: the head's snapshot comes in addition to the tail's */
        Fixture f;
        f.build(4096, 64, 512, 4096, /*windowed=*/true);
        f.prog.kv_groups.erase(f.prog.kv_groups.begin() + 1, f.prog.kv_groups.begin() + 3);
        f.prog.ops.resize(1);
        f.prog.buffers.resize(1);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        f.pc.groups = { 0 };
        f.cfg.checkpoint_policy = "tip-only";
        REQUIRE_EQ(f.start(), RAD_OK);
        f.pc.seed(640);
        f.pc.branch_at = 640;
        const uint64_t a = f.submit(1100, 1);
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.find(a)->n_computed, (int64_t)0);
        CHECK_EQ(f.sch.find(a)->checkpoint_at, (int64_t)640);
        f.commit_plain();
        for (int i = 0; i < 8 && f.sch.step(); ++i) f.commit_plain();
        CHECK_EQ(f.sch.metrics().checkpoints_written, 2ll);
        const std::vector<int64_t> want = { 640, 1072 };
        CHECK(f.pc.committed == want);
    }
    for (int c = 0; c < 3; ++c) {   /* a hybrid model, interval 256, quantum 64 */
        Fixture f;
        f.build();
        CHECK_OK(f.start());
        f.pc.seed(640);
        if (c == 1) { f.pc.lin = 512; f.pc.src = 3; }   /* resumes 128 short: not worth a split */
        f.pc.branch_at = c == 2 ? 0 : 640;              /* an ordinary hit: no branch */
        const uint64_t a = f.submit(1100, 1);
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.find(a)->n_computed, (int64_t)(c == 1 ? 512 : 0));
        CHECK_EQ(f.sch.find(a)->checkpoint_at, (int64_t)(c == 0 ? 640 : 0));
    }
}

TEST(tip_only_does_not_disable_recurrent_interval_checkpoints) {
    Fixture f;
    f.build();
    f.cfg.checkpoint_policy = "tip-only";
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK(!f.sch.geometry().window_ckpt);
    int64_t ckpt = -1;
    CHECK_EQ(f.sch.plan_chunk(0, 1100, 512, &ckpt), 256ll);
    CHECK_EQ(ckpt, 256ll);
}

TEST(prefill_chunks_split_at_the_checkpoint_interval) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    int64_t ck = -1;
    /* The budget would allow 512 tokens; the split takes 256 so the chunk ends on the interval. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 256);

    CHECK_EQ(f.sch.plan_chunk(256, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 512);

    /* The final chunk ends at the prompt's last quantum boundary and writes a checkpoint there, so
     * the next turn of a conversation resumes from it instead of from 768. The remainder, under a
     * quantum, is the next chunk and writes none: the accepted cost of owning the boundaries. */
    CHECK_EQ(f.sch.plan_chunk(768, 1000, 512, &ck), 192);
    CHECK_EQ(ck, 960);
    CHECK_EQ(f.sch.plan_chunk(960, 1000, 512, &ck), 40);
    CHECK_EQ(ck, -1);
    /* A prompt that ends less than a quantum past where the chunk starts has no boundary to
     * split at, and one inside the first interval is cheap enough to prefill again. */
    CHECK_EQ(f.sch.plan_chunk(768, 830, 512, &ck), 62);
    CHECK_EQ(ck, -1);
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck), 200);
    CHECK_EQ(ck, -1);
    /* And a prompt ending on a quantum boundary takes its checkpoint at the end, unsplit. */
    CHECK_EQ(f.sch.plan_chunk(768, 960, 512, &ck), 192);
    CHECK_EQ(ck, 960);

    /* A budget short of the interval still ends on a quantum, never in the middle of a tile. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 100, &ck), 64);
    CHECK_EQ(ck, -1);

    /* A budget that cannot hold one quantum schedules nothing rather than a misaligned chunk. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 50, &ck), 0);
}

/* A request's own checkpoint (Request::checkpoint_at): the end of a head other requests share, on
 * the quantum below it. It splits a chunk as the interval does and gives way to nothing. */
TEST(prefill_chunks_split_where_the_request_asks_for_a_checkpoint) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    int64_t ck = -1;
    /* Inside the first interval, where nothing else would write one: split at 128, the quantum
     * under 150, then the rest with none. */
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck, 150), 128);
    CHECK_EQ(ck, 128);
    CHECK_EQ(f.sch.plan_chunk(128, 200, 512, &ck, 150), 72);
    CHECK_EQ(ck, -1);
    /* Past the interval checkpoint that comes first, and before the next one. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck, 600), 256);
    CHECK_EQ(ck, 256);
    CHECK_EQ(f.sch.plan_chunk(512, 1000, 512, &ck, 600), 64);
    CHECK_EQ(ck, 576);
    /* On a boundary the chunk would end on anyway, it is that boundary's checkpoint. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck, 256), 256);
    CHECK_EQ(ck, 256);
    /* A head shorter than a quantum has no boundary to keep. */
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck, 40), 200);
    CHECK_EQ(ck, -1);
    /* And a budget short of it still ends on a quantum, the checkpoint left for a later chunk. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 100, &ck, 150), 64);
    CHECK_EQ(ck, -1);
    CHECK_EQ(f.sch.plan_chunk(64, 1000, 512, &ck, 150), 64);
    CHECK_EQ(ck, 128);
}

TEST(prefill_chunks_span_intervals_on_the_same_pieces) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().span, 2);

    int64_t ck = -1;
    /* The first interval is never spanned. */
    CHECK_EQ(f.sch.plan_chunk(0, 2000, 512, &ck), 256);
    CHECK_EQ(ck, 256);
    /* From a boundary past it, two intervals and a checkpoint at the end; the one between them
     * is not written. */
    CHECK_EQ(f.sch.plan_chunk(256, 2000, 512, &ck), 512);
    CHECK_EQ(ck, 768);
    CHECK_EQ(f.sch.plan_chunk(1280, 2000, 512, &ck), 512);
    CHECK_EQ(ck, 1792);
    /* The tail checkpoint where a span of 1 puts it. */
    CHECK_EQ(f.sch.plan_chunk(1792, 2000, 512, &ck), 192);
    CHECK_EQ(ck, 1984);
    /* A last chunk that reaches the prompt's end runs through to its tail checkpoint: the pieces
     * are [512, 768) and [768, 960), as a span of 1 cuts them. */
    CHECK_EQ(f.sch.plan_chunk(512, 1000, 512, &ck), 448);
    CHECK_EQ(ck, 960);
    /* And without one, it stops on the last boundary it crosses, where a span of 1 writes the
     * prompt's deepest checkpoint; the rest is the next chunk. */
    CHECK_EQ(f.sch.plan_chunk(512, 830, 512, &ck), 256);
    CHECK_EQ(ck, 768);
    CHECK_EQ(f.sch.plan_chunk(768, 830, 512, &ck), 62);
    CHECK_EQ(ck, -1);
    /* A budget short of the span steps back to the last boundary it crosses. */
    CHECK_EQ(f.sch.plan_chunk(256, 2000, 500, &ck), 256);
    CHECK_EQ(ck, 512);
    /* Off the interval (a resume from a tail checkpoint) the chunk ends at the next boundary. */
    CHECK_EQ(f.sch.plan_chunk(960, 3000, 512, &ck), 64);
    CHECK_EQ(ck, 1024);
}

/* A span moves no checkpoint and no piece, from a prompt's start or from any checkpoint a turn
 * resumes from: every chunk end and checkpoint it makes is one a span of 1 makes, and the deepest
 * checkpoint (where the next turn resumes) is the same. */
TEST(a_span_writes_a_subset_of_the_checkpoints_and_the_same_deepest_one) {
    struct Walk { std::vector<int64_t> ends, cks; };
    auto walk = [](Fixture& f, int64_t from, int64_t target) {
        Walk w;
        for (int64_t n = from; n < target; ) {
            int64_t ck = -1;
            const int64_t c = f.sch.plan_chunk(n, target, f.cfg.max_tok, &ck);
            if (c <= 0) break;
            n += c;
            w.ends.push_back(n);
            if (ck >= 0) w.cks.push_back(ck);
        }
        return w;
    };
    Fixture one;
    one.build();                          /* interval 256, budget 512, quantum 64 */
    CHECK_OK(one.start());
    Fixture two;
    two.build();
    two.cfg.prefill_span = 2;
    CHECK_OK(two.start());
    int bad_end = 0, bad_ck = 0, bad_deep = 0, spanned = 0;
    for (int64_t target = 1; target <= 2600; target += 7) {
        const Walk ref = walk(one, 0, target);
        /* From the start, and from every checkpoint a span of 1 leaves to resume from. */
        std::vector<int64_t> from = { 0 };
        for (int64_t c : ref.cks) if (c < target) from.push_back(c);
        for (int64_t s0 : from) {
            const Walk a = walk(one, s0, target);
            const Walk b = walk(two, s0, target);
            for (int64_t e : b.ends)
                bad_end += std::find(a.ends.begin(), a.ends.end(), e) == a.ends.end();
            for (int64_t c : b.cks)
                bad_ck += std::find(a.cks.begin(), a.cks.end(), c) == a.cks.end();
            const int64_t da = a.cks.empty() ? -1 : a.cks.back();
            const int64_t db = b.cks.empty() ? -1 : b.cks.back();
            bad_deep += da != db;
            spanned += b.ends.size() < a.ends.size();
        }
    }
    CHECK_EQ(bad_end, 0);
    CHECK_EQ(bad_ck, 0);
    CHECK_EQ(bad_deep, 0);
    CHECK(spanned > 0);

    /* One prompt by hand: the first interval alone, then two a chunk, and the last chunk to the
     * tail checkpoint a span of 1 writes (1664) with the rest, under a quantum, after it. */
    int64_t ck = -1;
    CHECK_EQ(two.sch.plan_chunk(0, 1700, 512, &ck), 256);    CHECK_EQ(ck, 256);
    CHECK_EQ(two.sch.plan_chunk(256, 1700, 512, &ck), 512);  CHECK_EQ(ck, 768);
    CHECK_EQ(two.sch.plan_chunk(768, 1700, 512, &ck), 512);  CHECK_EQ(ck, 1280);
    CHECK_EQ(two.sch.plan_chunk(1280, 1700, 512, &ck), 384); CHECK_EQ(ck, 1664);
    CHECK_EQ(two.sch.plan_chunk(1664, 1700, 512, &ck), 36);  CHECK_EQ(ck, -1);
}

/* A span is for a prefill alone in its step. Alone it takes two intervals; while another request
 * is in the engine (admitted, prefilling or decoding) no step is longer than the shared budget, one
 * interval here; alone again it spans again from the next boundary. */
TEST(a_span_is_taken_only_by_a_prefill_alone_in_its_step) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().shared_tok, 256);
    const uint64_t a = f.submit(3000, 4);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 256);           /* the first interval */
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);           /* alone: a span */
    f.commit_plain();
    const uint64_t b = f.submit(100, 3);
    int with_b = 0, wide_with_b = 0, spans_after = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        const StepPlan& p = f.sch.plan();
        bool has_b = false;
        int64_t a_tok = 0;
        for (const StepEntry& e : p.e) {
            if (e.seq == b) has_b = true;
            if (e.seq == a) a_tok = e.n_tokens;
        }
        if (has_b) {
            ++with_b;
            if (p.n_tok > 256 || a_tok > 256) ++wide_with_b;
        } else if (with_b > 0 && a_tok > 256) {
            ++spans_after;                /* b has finished: a is alone again */
        }
        f.commit_plain();
    }
    CHECK(with_b >= 3);
    CHECK_EQ(wide_with_b, 0);
    CHECK(spans_after >= 1);
}

/* With a shortest spanning prefill set (set_span_min), a shorter prompt alone in the engine takes
 * one interval a step, and a longer one spans through to its last intervals. */
TEST(a_span_min_keeps_a_short_prompt_at_one_interval) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    f.sch.set_span_min(2048);
    const uint64_t a = f.submit(1500, 2);
    int wide = 0, steps = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == a && e.is_prefill) { ++steps; wide += e.n_tokens > 256; }
        f.commit_plain();
    }
    CHECK(steps >= 5);
    CHECK_EQ(wide, 0);
    const uint64_t b = f.submit(3000, 2);
    int spans = 0, last_pf = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == b && e.is_prefill) { spans += e.n_tokens > 256; last_pf = e.n_tokens; }
        f.commit_plain();
    }
    CHECK(spans >= 4);
    CHECK(last_pf > 0);
}

/* A shared step under a span is planned as a deployment without one plans it: with two requests
 * from the start, every step they share is the same, entry for entry, at budget 512 with a span of
 * two as at budget 256 with none, --busy-prefill-chunk's shares included. */
TEST(a_shared_step_under_a_span_is_planned_as_without_one) {
    std::vector<std::vector<std::pair<uint64_t, int32_t>>> plans[2];
    for (int m = 0; m < 2; ++m) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/m == 0 ? 512 : 256);
        if (m == 0) f.cfg.prefill_span = 2;
        CHECK_OK(f.start());
        const uint64_t a = f.submit(3000, 4);
        const uint64_t b = f.submit(300, 6);
        (void)a;
        for (int k = 0; k < 40; ++k) {
            if (!f.sch.step()) break;
            std::vector<std::pair<uint64_t, int32_t>> row;
            bool has_b = false;
            for (const StepEntry& e : f.sch.plan().e) {
                row.push_back({ e.seq, e.n_tokens });
                has_b = has_b || e.seq == b;
            }
            if (!has_b) break;            /* b has finished: from here a is alone */
            plans[m].push_back(row);
            f.commit_plain();
        }
    }
    CHECK(plans[0].size() >= 4);
    CHECK(plans[0] == plans[1]);
}

/* Window snapshots do not support spans, and a model without checkpoints has no interval.
 * The shared budget still follows the flag, as the engine states it to the plugin
 * (RadBuildCtx::max_mixed_tok), and steps that share it admit as before. */
TEST(a_span_needs_an_interval_grid) {
    Program prog;
    prog.meta.n_layers = 16;
    prog.kv_groups.emplace_back();
    KVGroupInfo attn;
    attn.name = "attn";
    attn.decl.kind = RAD_KV_FULL;
    attn.block_size = 16;
    attn.layers = { 0 };
    prog.kv_groups.push_back(attn);
    KVGroupInfo win;
    win.name = "swa";
    win.decl.kind = RAD_KV_WINDOW;
    win.decl.window = 64;
    win.block_size = 16;
    win.layers = { 1 };
    prog.kv_groups.push_back(win);
    Config cfg;
    cfg.max_tok = 512;
    cfg.checkpoint_interval = 256;
    cfg.prefix_cache = true;
    cfg.prefill_span = 2;
    ChunkGeometry g;
    CHECK_OK(chunk_geometry_resolve(prog, cfg, &g));
    CHECK(g.window_ckpt);
    CHECK_EQ(g.span, 1ll);
    CHECK_EQ(g.shared_tok, 256ll);

    Fixture f;
    f.build();
    f.cfg.checkpoint_interval = 0;
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().span, 1ll);
    f.submit(100, 3);
    f.submit(100, 3);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int64_t)f.sch.plan().e.size(), 2ll);
}

/* A Gemma-like model (a sliding window over a full-attention trunk, no linear state) resumes a
 * prefix hit only from a window snapshot at an earlier prompt's tail. One is taken wherever that
 * position, a block short of the prompt's end, is kWindowSnapshotMin or more, at the default
 * 2048-token interval too, so a short conversation's next turn resumes: the shortest prompt takes
 * its snapshot at the floor and one token less takes none, and a prompt past an interval takes the
 * one it always did. An interval under the floor is the floor. A 1000-token prompt runs whole, its
 * snapshot taken inside the chunk: the window's rows are still held when it ends, so unlike linear
 * state nothing makes the prompt split there. */
TEST(a_window_snapshot_is_taken_at_the_tail_of_a_short_prompt_too) {
    Fixture f;
    f.build(4096, 64, /*max_tok=*/2048, 4096, /*windowed=*/true, /*linear=*/false);
    f.cfg.checkpoint_interval = 2048;
    CHECK_OK(f.start());
    CHECK(f.sch.geometry().window_ckpt);
    const int64_t Q = f.sch.geometry().quantum, back = std::max<int64_t>(Q, 16);
    const int64_t floor = Scheduler::kWindowSnapshotMin, shortest = floor + back;
    CHECK_EQ(floor % Q, (int64_t)0);
    CHECK_EQ(f.sch.tail_checkpoint(shortest), floor);
    CHECK_EQ(f.sch.tail_checkpoint(shortest - 1), (int64_t)-1);
    CHECK_EQ(f.sch.tail_checkpoint(400), (int64_t)-1);
    CHECK_EQ(f.sch.tail_checkpoint(3000), (3000 - back) / Q * Q);
    const int64_t t = (1000 - back) / Q * Q;
    CHECK_EQ(f.sch.tail_checkpoint(1000), t);
    f.submit(1000, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, 1000);
    CHECK_EQ(b->n_checkpoints, 1);
    CHECK_EQ(b->checkpoint_tok[0], t - 1);

    Fixture g;
    g.build(4096, 64, /*max_tok=*/2048, 4096, /*windowed=*/true, /*linear=*/false);
    g.cfg.checkpoint_interval = 256;
    CHECK_OK(g.start());
    const int64_t I = g.sch.geometry().checkpoint_interval;
    CHECK_EQ(I, (int64_t)256);
    CHECK_EQ(g.sch.tail_checkpoint(I + back), I);
    CHECK_EQ(g.sch.tail_checkpoint(I + back - 1), (int64_t)-1);
}

/* A prompt longer than the step budget takes its window snapshot inside its last chunk: the chunks
 * before it are the budget's, and the last runs to the prompt's end with the snapshot at its tail
 * row. */
TEST(a_long_prompt_takes_its_window_snapshot_inside_its_last_chunk) {
    Fixture f;
    f.build(4096, 64, /*max_tok=*/512, 4096, /*windowed=*/true, /*linear=*/false);
    f.cfg.checkpoint_interval = 2048;
    CHECK_OK(f.start());
    const int64_t Q = f.sch.geometry().quantum, back = std::max<int64_t>(Q, 16);
    const int64_t t = (1000 - back) / Q * Q;
    f.submit(1000, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, 512);
    CHECK_EQ(b->n_checkpoints, 0);
    f.commit_plain();
    b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, 1000 - 512);
    CHECK_EQ(b->n_checkpoints, 1);
    CHECK_EQ(b->checkpoint_tok[0], t - 512 - 1);
}

/* Under load a prompt's chunks are held back for the speculating decode rows behind it. Every prompt
 * takes its window snapshot once, at its tail row, in whichever chunk carries that row: the final
 * one, or a held-back one cut short of the prompt's end on a quantum. */
TEST(a_window_snapshot_under_load_lands_at_the_tail_whatever_chunk_carries_it) {
    for (int64_t n = 600; n <= 1000; n += 24) {
        Fixture f;
        f.build(4096, 64, /*max_tok=*/256, 8192, /*windowed=*/true, /*linear=*/false);
        f.cfg.checkpoint_interval = 2048;
        f.cfg.n_spec = 7;
        f.cfg.busy_prefill_chunk = -1;
        CHECK_OK(f.start());
        const int64_t Q = f.sch.geometry().quantum, back = std::max<int64_t>(Q, 16);
        const int64_t t = (n - back) / Q * Q;
        for (int d = 0; d < 4; ++d) f.submit(40, 4000);
        for (int k = 0; k < 2; ++k) { CHECK(f.sch.step() != nullptr); f.commit_plain(11); }
        const uint64_t p = f.submit(n, 8);
        int snaps = 0;
        for (int k = 0; k < 64 && f.sch.find(p) && f.sch.find(p)->n_computed < n; ++k) {
            CHECK(f.sch.step() != nullptr);
            for (const StepEntry& e : f.sch.plan().e) {
                if (e.seq != p || e.ckpt_slot < 0) continue;
                ++snaps;
                if (!(e.ctx_len < t && e.ctx_len + e.n_tokens >= t && e.ctx_len + e.ckpt_tok + 1 == t))
                    fprintf(stderr, "    n %lld: chunk [%d, %d), snapshot at %d, tail %lld\n",
                            (long long)n, (int)e.ctx_len, (int)(e.ctx_len + e.n_tokens),
                            (int)(e.ctx_len + e.ckpt_tok + 1), (long long)t);
                CHECK((int64_t)e.ctx_len < t && (int64_t)e.ctx_len + e.n_tokens >= t);
                CHECK_EQ((int64_t)e.ctx_len + e.ckpt_tok + 1, t);
            }
            f.commit_plain(11);
        }
        if (snaps != 1) fprintf(stderr, "    n %lld: %d snapshots\n", (long long)n, snaps);
        CHECK_EQ(snaps, 1);
    }
}

/* ================================================================== exact resume */
/* --prefix-resume exact (the default): a hit resumes only where an idle server's cold prefill of
 * the same prompt starts a chunk, and a prompt's tail snapshot is where its last chunk starts. A
 * kernel may round a row differently at another chunk length (spec §17: the split-K count is
 * chosen from M), so a hit is bit-identical to a cold run only when it recomputes the cold run's
 * own chunks. */

/* The chunk ends and checkpoints an idle server plans for an `L`-token prompt from `from`. */
struct Cut { std::vector<int64_t> ends, ckpts; };
static Cut cut_from(const Scheduler& s, int64_t from, int64_t L, int64_t budget) {
    Cut c;
    for (int64_t pos = from; pos < L;) {
        int64_t ck = -1;
        const int64_t n = s.plan_chunk(pos, L, budget, &ck);
        if (n <= 0) break;
        pos += n;
        c.ends.push_back(pos);
        if (ck > 0) c.ckpts.push_back(ck);
    }
    return c;
}
static std::vector<int64_t> ends_after(const Cut& c, int64_t from) {
    std::vector<int64_t> v;
    for (int64_t e : c.ends) if (e > from) v.push_back(e);
    return v;
}

/* Which geometry: 0 the GDN hybrid at interval 256 / budget 512 (chunks split at the interval);
 * 1 interval 1024 past a 512 budget; 2 a span of 2; 3 a sliding-window model with periodic
 * snapshots; 4 the same, tip-only (no interval split); 5 full attention alone (no snapshots). */
static void exact_fixture(Fixture& f, int kind) {
    const bool window = kind == 3 || kind == 4, attn_only = kind == 5;
    f.build(4096, 64, 512, 8192, window, !attn_only);
    if (window) {
        /* As the window tests above: drop the GDN hybrid's linear and conv groups and its scan. */
        f.prog.kv_groups.erase(f.prog.kv_groups.begin() + 1, f.prog.kv_groups.begin() + 3);
        f.prog.ops.resize(1);
        f.prog.buffers.resize(1);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        f.pc.groups = { 0 };
        if (kind == 4) f.cfg.checkpoint_policy = "tip-only";
    }
    if (kind == 1) f.cfg.checkpoint_interval = 1024;
    if (kind == 2) f.cfg.prefill_span = 2;
    f.cfg.prefix_exact = true;
}

TEST(exact_resume_grid_is_the_chunk_an_idle_prefill_cuts) {
    struct Want { int kind; int64_t interval, grid, origin; };
    for (const Want w : { Want{ 0, 256, 256, 0 }, Want{ 1, 1024, 512, 0 }, Want{ 2, 256, 512, 256 },
                          Want{ 3, 256, 256, 0 }, Want{ 4, 256, 512, 0 }, Want{ 5, 0, 512, 0 } }) {
        Fixture f;
        exact_fixture(f, w.kind);
        REQUIRE_EQ(f.start(), RAD_OK);
        const ChunkGeometry& g = f.sch.geometry();
        CHECK(g.exact);
        CHECK_EQ(g.checkpoint_interval, w.interval);
        CHECK_EQ(g.grid, w.grid);
        CHECK_EQ(g.grid_origin, w.origin);
    }
    /* Nearest, or no prefix cache: no grid, every position its own floor. */
    for (int off = 0; off < 2; ++off) {
        Fixture f;
        exact_fixture(f, 0);
        if (off) f.cfg.prefix_cache = false; else f.cfg.prefix_exact = false;
        REQUIRE_EQ(f.start(), RAD_OK);
        CHECK(!f.sch.geometry().exact);
        CHECK_EQ(f.sch.geometry().grid_floor(300), 300);
    }
    Fixture f;
    exact_fixture(f, 2);
    REQUIRE_EQ(f.start(), RAD_OK);
    const ChunkGeometry& g = f.sch.geometry();   /* 0, 256, 768, 1280, ... */
    CHECK_EQ(g.grid_floor(255), 0);
    CHECK_EQ(g.grid_floor(256), 256);
    CHECK_EQ(g.grid_floor(767), 256);
    CHECK_EQ(g.grid_floor(1000), 768);
    CHECK(!g.on_grid(512));
}

/* An interval past the budget that is not a multiple of the budget's chunk would put the chunk
 * after every checkpoint off the cold run's grid: exact rounds it up, with a warning that names
 * both ways out; nearest keeps it. */
TEST(exact_rounds_an_interval_off_the_chunk_up_to_a_multiple_of_it) {
    for (int exact = 0; exact < 2; ++exact) {
        Fixture f;
        f.build();                            /* quantum 64 */
        f.cfg.checkpoint_interval = 768;      /* past the 512 budget, not a multiple of it */
        f.cfg.prefix_exact = exact != 0;
        log_set_level(Log::Error);
        REQUIRE_EQ(f.start(), RAD_OK);
        CHECK_EQ(f.sch.geometry().checkpoint_interval, exact ? 1024 : 768);
        if (exact) CHECK_EQ(f.sch.geometry().grid, 512);
    }
    /* A budget off the quantum chunks at the quantum below it: 500 -> 448, and the interval goes
     * to the next multiple of 448. */
    Fixture f;
    f.build(4096, 64, /*max_tok=*/500);
    f.cfg.checkpoint_interval = 1024;
    f.cfg.prefix_exact = true;
    log_set_level(Log::Error);
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK_EQ(f.sch.geometry().checkpoint_interval, 1344);
    CHECK_EQ(f.sch.geometry().grid, 448);
    int64_t ck = -1;
    CHECK_EQ(f.sch.plan_chunk(0, 5000, 500, &ck), 448);
    CHECK_EQ(f.sch.plan_chunk(896, 5000, 500, &ck), 448);
    CHECK_EQ(ck, 1344);
}

/* The tail snapshot is where the prompt's last chunk starts, a boundary an idle chunk already
 * ends on: no extra step, and an exact repeat re-prefills that last chunk whole. */
TEST(an_exact_tail_snapshot_is_where_the_last_chunk_starts) {
    Fixture f;
    exact_fixture(f, 1);                      /* interval 1024, chunks of 512 */
    REQUIRE_EQ(f.start(), RAD_OK);
    Cut c = cut_from(f.sch, 0, 1800, 512);
    CHECK((c.ends == std::vector<int64_t>{ 512, 1024, 1536, 1800 }));
    CHECK((c.ckpts == std::vector<int64_t>{ 1024, 1536 }));
    /* Its last chunk starting on the interval: that checkpoint is the tail's. */
    c = cut_from(f.sch, 0, 1100, 512);
    CHECK((c.ends == std::vector<int64_t>{ 512, 1024, 1100 }));
    CHECK((c.ckpts == std::vector<int64_t>{ 1024 }));
    /* One ending on the grid: the last chunk is [1536, 2048), and its end is an interval. */
    c = cut_from(f.sch, 0, 2048, 512);
    CHECK((c.ckpts == std::vector<int64_t>{ 1024, 1536, 2048 }));
    /* Inside the first interval nothing (as under nearest: cheap to prefill again). */
    c = cut_from(f.sch, 0, 900, 512);
    CHECK(c.ckpts.empty());
    /* A step sharing its budget may straddle the tail: split there, the snapshot written. */
    int64_t ck = -1;
    CHECK_EQ(f.sch.plan_chunk(1400, 1800, 512, &ck), 136);
    CHECK_EQ(ck, 1536);
    /* A budget short of a quantum boundary past it leaves it for the next chunk. */
    CHECK_EQ(f.sch.plan_chunk(1024, 1800, 100, &ck), 64);
    CHECK_EQ(ck, -1);

    /* Nearest cuts the same prompt's last chunk at its last quantum boundary instead. */
    Fixture n;
    n.build();
    n.cfg.checkpoint_interval = 1024;
    REQUIRE_EQ(n.start(), RAD_OK);
    c = cut_from(n.sch, 0, 1800, 512);
    CHECK((c.ends == std::vector<int64_t>{ 512, 1024, 1536, 1792, 1800 }));
    CHECK((c.ckpts == std::vector<int64_t>{ 1024, 1792 }));
}

/* The property itself, over every geometry and many prompt lengths: every checkpoint is on the
 * grid and is a chunk start of the cold run of the prompt that wrote it and of any longer prompt
 * it prefixes (an agent's next turn), and from it the hit cuts exactly the cold run's chunks. */
TEST(an_exact_hit_from_any_checkpoint_recomputes_the_cold_runs_own_chunks) {
    for (int kind = 0; kind < 6; ++kind) {
        Fixture f;
        exact_fixture(f, kind);
        REQUIRE_EQ(f.start(), RAD_OK);
        const ChunkGeometry& g = f.sch.geometry();
        int checked = 0;
        for (int64_t L = 100; L <= 3000; L += 37) {
            const Cut cold = cut_from(f.sch, 0, L, 512);
            for (int64_t c : cold.ckpts) {
                CHECK(g.on_grid(c));
                if (c >= L) continue;
                CHECK(std::find(cold.ends.begin(), cold.ends.end(), c) != cold.ends.end());
                CHECK(cut_from(f.sch, c, L, 512).ends == ends_after(cold, c));
                for (int64_t L2 : { L + 1, L + 300, L + 1500 }) {
                    const Cut cold2 = cut_from(f.sch, 0, L2, 512);
                    CHECK(std::find(cold2.ends.begin(), cold2.ends.end(), c) != cold2.ends.end());
                    CHECK(cut_from(f.sch, c, L2, 512).ends == ends_after(cold2, c));
                }
                ++checked;
            }
        }
        CHECK(kind == 5 ? checked == 0 : checked > 20);
    }
    /* What nearest gives up: its tail snapshot at a 1000-token prompt's last quantum boundary
     * (960) is no chunk start of the 1500-token turn after it, which a cold run cuts at 768 and
     * 1024, so a hit there recomputes [960, 1024) as a chunk of its own. */
    Fixture n;
    n.build();
    REQUIRE_EQ(n.start(), RAD_OK);
    const Cut c1000 = cut_from(n.sch, 0, 1000, 512);
    CHECK(std::find(c1000.ckpts.begin(), c1000.ckpts.end(), 960) != c1000.ckpts.end());
    const Cut c1500 = cut_from(n.sch, 0, 1500, 512);
    CHECK(std::find(c1500.ends.begin(), c1500.ends.end(), 960) == c1500.ends.end());
    CHECK(std::find(c1500.ends.begin(), c1500.ends.end(), 768) != c1500.ends.end());
    CHECK_EQ(cut_from(n.sch, 960, 1500, 512).ends.front(), 1024);
}

/* At admission: a hit at a snapshot resumes there (plan_chunk put it on the grid); a hit with no
 * snapshot to resume from (full attention alone) steps back to the grid, the blocks above it
 * left to the cache and recomputed in the cold run's chunk. */
TEST(an_exact_hit_steps_back_to_the_grid_and_prefills_the_cold_chunks) {
    for (int exact = 0; exact < 2; ++exact) {
        Fixture f;
        exact_fixture(f, 5);                  /* chunks of 512, no snapshots */
        f.cfg.prefix_exact = exact != 0;
        REQUIRE_EQ(f.start(), RAD_OK);
        f.pc.seed(784);                       /* 49 blocks */
        const uint64_t a = f.submit(1300, 8);
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        CHECK_EQ(f.sch.find(a)->n_computed, exact ? 512 : 784);
        CHECK_EQ(b->ctx_lens[0], exact ? 512 : 784);
        CHECK_EQ(b->n_tok, 512);              /* [512, 1024) as the cold run, or [784, 1296) */
        CHECK_EQ((int)f.kv.block_table(a, 0).size(), exact ? 64 : 81);
        CHECK_EQ(f.sch.metrics().hit_cut_tokens, exact ? 272 : 0);
    }
    /* With snapshots the hit is at one: kept as is, restored from its slot. */
    Fixture f;
    exact_fixture(f, 1);
    REQUIRE_EQ(f.start(), RAD_OK);
    f.pc.seed(1600);
    f.pc.lin = 1536;                          /* a tail snapshot, on the 512 grid */
    f.pc.src = 3;
    const uint64_t a = f.submit(1800, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 1536);
    CHECK_EQ(b->n_tok, 264);                  /* the cold run's last chunk, whole */
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 3);
}

/* A cold prompt at the scheduler: the last chunk is not split at the end, and an exact repeat's
 * hit at the tail snapshot is that chunk. */
TEST(an_exact_cold_prefill_keeps_its_last_chunk_whole) {
    Fixture f;
    exact_fixture(f, 0);                      /* interval 256, chunks of 256 */
    REQUIRE_EQ(f.start(), RAD_OK);
    f.submit(1000, 1);
    std::vector<int64_t> toks, ckpts;
    for (int i = 0; i < 8; ++i) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        toks.push_back(b->n_tok);
        for (int k = 0; k < b->n_checkpoints; ++k)
            ckpts.push_back(b->ctx_lens[0] + b->checkpoint_tok[k] + 1);
        f.commit_plain();
    }
    CHECK((toks == std::vector<int64_t>{ 256, 256, 256, 232 }));
    CHECK((ckpts == std::vector<int64_t>{ 256, 512, 768 }));
}

/* ================================================================== lever PSK design 2: anchors */
/* ds41_psk 2 (the plugin's switch, read into the scheduler by the engine) with ds41_psk_k K: a request admitted then is
 * planned -- its targets are the prompt's end, its tail snapshot and every K-th periodic checkpoint position (the exact
 * grid's points from its origin: 256, 768, 1280, ... here) -- and the step hands the plugin, per prefill entry, its next
 * target less its chunk's end. Window / linear-state snapshots are reserved at targets only; chunking is unchanged. */
struct PskRun { std::vector<int64_t> ends, ckpts, ahead; std::vector<int> planned; };
static PskRun psk_run(Fixture& f, uint64_t id, int max_steps = 64) {
    PskRun r;
    for (int i = 0; i < max_steps && f.sch.find(id) && f.sch.find(id)->n_computed < (int32_t)f.sch.find(id)->prompt.size();
         ++i) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        r.ends.push_back(b->ctx_lens[0] + b->n_tok);
        for (int k = 0; k < b->n_checkpoints; ++k) r.ckpts.push_back(b->ctx_lens[0] + b->checkpoint_tok[k] + 1);
        std::vector<int32_t> ahead, rows;
        int64_t dec = -1;
        const bool p = f.sch.psk_plan(b->step, &ahead, &rows, &dec);
        r.planned.push_back(p ? 1 : 0);
        if (p) {
            CHECK_EQ((int)ahead.size(), 1);
            CHECK_EQ((int)rows.size(), 1);
            CHECK_EQ((int64_t)rows[0], b->n_tok);
            CHECK_EQ(dec, (int64_t)0);
            r.ahead.push_back(ahead.empty() ? -999999 : ahead[0]);
        }
        f.commit_plain();
    }
    return r;
}

TEST(psk_anchors_plan_each_chunk_and_snapshot_at_targets_only) {
    /* today (no mode, and modes 0 / 1): every grid chunk end snapshots, nothing is planned */
    for (int mode : { -1, 0, 1 }) {
        Fixture f;
        exact_fixture(f, 2);                  /* interval 256, span 2: chunks [0,256), then 512 each */
        REQUIRE_EQ(f.start(), RAD_OK);
        if (mode >= 0) f.sch.set_psk_mode(mode);
        const PskRun r = psk_run(f, f.submit(3000, 1));
        CHECK((r.ends == std::vector<int64_t>{ 256, 768, 1280, 1792, 2304, 2816, 3000 }));
        CHECK((r.ckpts == std::vector<int64_t>{ 256, 768, 1280, 1792, 2304, 2816 }));
        CHECK((r.planned == std::vector<int>(7, 0)));
        CHECK_EQ(f.sch.psk_ckpt_dropped(), (int64_t)0);
    }
    /* ds41_psk 2, K 2: anchors q_2 = 768, q_4 = 1792, q_6 = 2816 (also the tail), and the end 3000 */
    Fixture f;
    exact_fixture(f, 2);
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK_EQ(f.sch.psk_k(), Scheduler::kPskK);
    CHECK_EQ(f.sch.set_psk_k(0), Scheduler::kPskK);      /* refused: kept */
    CHECK_EQ(f.sch.set_psk_k(2), 2);
    f.sch.set_psk_mode(2);
    const PskRun r = psk_run(f, f.submit(3000, 1));
    CHECK((r.ends == std::vector<int64_t>{ 256, 768, 1280, 1792, 2304, 2816, 3000 }));   /* chunks unchanged */
    CHECK((r.ckpts == std::vector<int64_t>{ 768, 1792, 2816 }));
    CHECK((r.planned == std::vector<int>(7, 1)));
    /* next target less the chunk end: [0,256) -> 768, [256,768) -> 768, [768,1280) -> 1792, ... [2816,3000) -> 3000 */
    CHECK((r.ahead == std::vector<int64_t>{ 512, 0, 512, 0, 512, 0, 0 }));
    CHECK_EQ(f.sch.psk_ckpt_dropped(), (int64_t)3);
    CHECK((f.pc.committed == std::vector<int64_t>{ 768, 1792, 2816 }));
}

/* The plan is the request's, latched at admission: a mode change while it prefills changes nothing of its chunks (their
 * targets must agree, or a cone crossing a chunk start would read a window no chunk computed), and the next request takes
 * the new mode. K = 1: every grid point is an anchor, so every snapshot is kept. */
TEST(psk_anchors_are_latched_per_request) {
    Fixture f;
    exact_fixture(f, 2);
    REQUIRE_EQ(f.start(), RAD_OK);
    f.sch.set_psk_k(4);
    f.sch.set_psk_mode(2);
    const uint64_t a = f.submit(3000, 1);
    const RadBatch* b0 = f.sch.step();                  /* admitted: [0, 256) planned */
    REQUIRE(b0 != nullptr);
    std::vector<int32_t> ahead, rows;
    int64_t dec = 0;
    CHECK(f.sch.psk_plan(b0->step, &ahead, &rows, &dec));
    CHECK(!f.sch.psk_plan(b0->step + 1, &ahead, &rows, &dec));
    CHECK_EQ(ahead[0], 1792 - 256);                     /* q_4 = 1792 */
    CHECK_EQ(b0->n_checkpoints, 0);                     /* 256 = q_1: not a target */
    f.commit_plain();
    f.sch.set_psk_mode(0);
    f.sch.set_psk_k(1);
    const PskRun r = psk_run(f, a);
    CHECK((r.planned == std::vector<int>(6, 1)));
    CHECK((r.ckpts == std::vector<int64_t>{ 1792, 2816 }));   /* q_4, and the tail q_6 */
    CHECK((r.ahead == std::vector<int64_t>{ 1024, 512, 0, 512, 0, 0 }));
    /* the next request: not planned (mode 0), every grid snapshot */
    const PskRun r2 = psk_run(f, f.submit(1000, 1));
    CHECK((r2.planned == std::vector<int>(3, 0)));
    /* mode 2 again with K = 1: every grid point is an anchor */
    f.sch.set_psk_mode(2);
    const int64_t dropped = f.sch.psk_ckpt_dropped();
    const PskRun r3 = psk_run(f, f.submit(1500, 1));
    CHECK((r3.planned == std::vector<int>(4, 1)));
    CHECK((r3.ckpts == std::vector<int64_t>{ 256, 768, 1280 }));
    CHECK_EQ(f.sch.psk_ckpt_dropped(), dropped);
}

/* The targets themselves: the next one past a position, at most the prefill's end; a request's checkpoint_at; ds41_psk
 * 3's first grid point at or past 32768 (256 + 64 x 512 = 33024 on this grid). */
TEST(psk_targets_are_the_end_the_tail_every_kth_grid_point_and_checkpoint_at) {
    Fixture f;
    exact_fixture(f, 2);
    REQUIRE_EQ(f.start(), RAD_OK);
    SchedReq s;
    s.req = f.request(50000, 1);
    s.psk_k = 8;                                        /* q_8 = 3840, q_16 = 7936, ... q_64 = 32512, q_72 = 36608 */
    CHECK_EQ(f.sch.psk_next_target(s, 0, 50000), 3840);
    CHECK_EQ(f.sch.psk_next_target(s, 3840, 50000), 7936);
    CHECK_EQ(f.sch.psk_next_target(s, 3839, 50000), 3840);
    CHECK_EQ(f.sch.psk_next_target(s, 32512, 50000), 36608);
    CHECK_EQ(f.sch.psk_next_target(s, 0, 3000), 2816);  /* the tail snapshot of a 3000-token prompt (q_6) */
    CHECK_EQ(f.sch.psk_next_target(s, 2816, 3000), 3000);
    CHECK(f.sch.psk_is_target(s, 3840, 50000));
    CHECK(!f.sch.psk_is_target(s, 3328, 50000));        /* q_7 */
    CHECK(f.sch.psk_is_target(s, 50000, 50000));
    CHECK(!f.sch.psk_is_target(s, 33024, 50000));
    s.psk_32k = true;                                   /* ds41_psk 3 */
    CHECK_EQ(f.sch.psk_next_target(s, 32512, 50000), 33024);
    CHECK(f.sch.psk_is_target(s, 33024, 50000));
    s.req->checkpoint_at = 5000;                        /* on the quantum (64): 4992 */
    CHECK_EQ(f.sch.psk_next_target(s, 3840, 50000), 4992);
    CHECK(f.sch.psk_is_target(s, 4992, 50000));
    s.psk_k = 0;                                        /* not planned: the end only */
    CHECK_EQ(f.sch.psk_next_target(s, 0, 50000), 50000);
    CHECK(!f.sch.psk_is_target(s, 3840, 50000));
}

/* A prefix hit under anchors: the cache resumes a hit at its deepest snapshot at or below it (PrefixCache::deepest_ckpt),
 * and a planned request reserved none but at its targets, so a hit at a non-anchor position (here 2304, the 1280 grid
 * snapshot not taken) falls back to the anchor below it (768), never to a window a skipped chunk left stale. The resumed
 * request's chunks are the cold run's (the exact grid) and it plans from there. */
TEST(psk_a_hit_between_anchors_resumes_at_the_anchor_below) {
    Fixture f;
    exact_fixture(f, 2);
    REQUIRE_EQ(f.start(), RAD_OK);
    f.sch.set_psk_k(2);
    f.sch.set_psk_mode(2);
    const PskRun cold = psk_run(f, f.submit(3000, 1));
    CHECK((f.pc.committed == std::vector<int64_t>{ 768, 1792, 2816 }));
    /* the cache as the real one answers a 2304-token match: attention to 2304, the deepest snapshot below it 1792 */
    f.pc.seed(2304);
    f.pc.lin = 1792;
    f.pc.src = 5;
    const uint64_t b = f.submit(3500, 1);
    const RadBatch* bb = f.sch.step();
    REQUIRE(bb != nullptr);
    CHECK_EQ(f.sch.find(b)->n_computed, 1792);
    CHECK_EQ(bb->ctx_lens[0], 1792);
    CHECK_EQ(bb->n_tok, 512);                           /* [1792, 2304): the cold run's chunk */
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 5);
    std::vector<int32_t> ahead, rows;
    int64_t dec = 0;
    CHECK(f.sch.psk_plan(bb->step, &ahead, &rows, &dec));
    CHECK_EQ(ahead[0], 2816 - 2304);                    /* q_6 = 2816 (and the 3500 prompt's tail is q_7 = 3328) */
    f.commit_plain();
    const PskRun rest = psk_run(f, b);
    CHECK((rest.ends == std::vector<int64_t>{ 2816, 3328, 3500 }));
    CHECK((rest.ahead == std::vector<int64_t>{ 0, 0, 0 }));      /* 2816 a K-th point, 3328 the tail, 3500 the end */
    (void)cold;
}

/* ================================================================== lever PSK-CR: the cone replay */
/* The production geometry (--checkpoint-interval 2048, --prefill-span 2, --max-num-batched-tokens 4160: the grid 2048 +
 * k x 4096) over a trunk with a full group (prefix-cached), linear and conv state and a window group (not cached), as
 * deepseek41's: kv_compressed / kv_index, the compressor tail, the Engram tokens, kv_window. The replay's reach is the
 * 1017's C_0 = 160 + 39 x 128 = 5152. */
static void cr_fixture(Fixture& f) {
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000, /*windowed=*/true, /*linear=*/true);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefix_exact = true;
    f.pc.groups = { 0 };   /* the full group: the window is not prefix-cached */
    f.kv.snap = { 3 };     /* the window rides in a snapshot (a hit at one resumes) */
}
struct CrStep { int64_t ctx = 0, n = 0, ahead = 0; int flag = -1, ckpts = 0, restore = -1, replay = 0; };
static std::vector<CrStep> cr_run(Fixture& f, uint64_t id, int max_steps = 64) {
    std::vector<CrStep> out;
    for (int k = 0; k < max_steps && f.sch.find(id) && f.sch.find(id)->n_computed < (int32_t)f.sch.find(id)->prompt.size();
         ++k) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        CrStep c;
        c.ctx = b->ctx_lens[0];
        c.n = b->n_tok;
        c.ckpts = b->n_checkpoints;
        c.restore = f.sch.plan().e[0].ckpt_restore;
        c.replay = f.sch.plan().e[0].cr_replay ? 1 : 0;
        std::vector<int32_t> ahead, rows, flags;
        int64_t dec = 0;
        if (f.sch.psk_plan(b->step, &ahead, &rows, &dec, &flags)) {
            CHECK_EQ((int)flags.size(), 1);
            c.ahead = ahead[0];
            c.flag = flags.empty() ? -1 : flags[0];
        }
        /* lever PSK-CR's batch: a replay chunk's full-group slots and linear-state slot are -1 (nothing of them
         * written), its window and conv slots as any prefill's; any other chunk none of that */
        const RadKVGroupBatch& full = b->kv[0];
        const RadKVGroupBatch& lin = b->kv[1];
        const RadKVGroupBatch& conv = b->kv[2];
        int full_neg = 0;   /* (the fake maps a window from position 0, so its slots say nothing here) */
        for (int64_t t = 0; t < b->n_tok; ++t) full_neg += full.slot_mapping[t] < 0;
        if (c.replay) {
            CHECK_EQ(full_neg, (int)b->n_tok);
            CHECK_EQ(lin.state_index[0], -1);
            CHECK_EQ(lin.slot_mapping[0], -1);
            CHECK(conv.state_index[0] >= 0);
        } else {
            CHECK_EQ(full_neg, 0);
            CHECK(lin.state_index[0] >= 0);
        }
        out.push_back(c);
        f.commit_plain();
    }
    return out;
}

/* A hit with no snapshot at the grid point under it (K4: the anchor below is 14336, the match 26000, its grid point
 * 22528): under ds41_psk_cr 1 the request adopts the cached full group up to 22528, starts at 14336 (the grid point under
 * 22528 - 5152) with nothing restored, runs the cold run's chunks [14336, 18432) and [18432, 22528) as replay chunks
 * (flag 1, the target 22528, the batch masks; no snapshot inside, ds41-pskcr2: one at the end 22528), then the prompt's
 * rest as any planned request's; under 0 it
 * resumes at the anchor below as before and no lever PSK-CR counter moves. The batch rows of a replay chunk are its
 * cold chunk's; its full-group slots and linear-state slot are -1 (cr_run). */
TEST(psk_cr_a_hit_without_a_snapshot_replays_the_cold_grid_chunks_to_its_grid_point) {
    for (int cr : { 1, 0 }) {
        Fixture f;
        cr_fixture(f);
        CHECK_OK(f.start());
        f.sch.set_span_min(16384);
        f.sch.set_psk_mode(2);
        f.sch.set_psk_k(4);
        CHECK_EQ(f.sch.set_psk_cr(cr, 5152), cr);
        f.pc.seed(26000);
        f.pc.lin = 14336;                         /* the K4 anchor below the match */
        f.pc.src = 5;
        const uint64_t a = f.submit(32000, 1);
        const std::vector<CrStep> st = cr_run(f, a);
        REQUIRE(st.size() >= 3);
        const SchedMetrics m = f.sch.metrics();
        if (cr) {
            CHECK_EQ(st[0].ctx, (int64_t)14336);
            CHECK_EQ(st[0].n, (int64_t)4096);
            CHECK_EQ(st[0].replay, 1);
            CHECK_EQ(st[0].flag, 1);
            CHECK_EQ(st[0].ahead, (int64_t)4096);     /* 22528 - 18432 */
            CHECK_EQ(st[0].ckpts, 0);
            CHECK_EQ(st[0].restore, -1);             /* nothing restored */
            CHECK_EQ(st[1].ctx, (int64_t)18432);
            CHECK_EQ(st[1].n, (int64_t)4096);
            CHECK_EQ(st[1].flag, 1);
            CHECK_EQ(st[1].ahead, (int64_t)0);
            CHECK_EQ(st[1].ckpts, 1);                /* ds41-pskcr2: the snapshot at the replay's end */
            CHECK_EQ(st[2].ctx, (int64_t)22528);      /* the prompt goes on from the replay's end */
            CHECK_EQ(st[2].replay, 0);
            CHECK_EQ(st[2].flag, 0);
            CHECK_EQ(st[2].ahead, (int64_t)(30720 - 26624));
            for (size_t i = 2; i < st.size(); ++i) CHECK_EQ(st[i].flag, 0);
            CHECK_EQ(m.psk_cr_hits, (int64_t)1);
            CHECK_EQ(m.psk_cr_rows, (int64_t)8192);
            CHECK_EQ(m.psk_cr_resume_tokens, (int64_t)22528);
            CHECK_EQ(m.psk_fallback_resumes, (int64_t)0);
            CHECK_EQ(m.psk_cr, 1);
            CHECK_EQ(m.psk_cr_end_ckpts, (int64_t)1);
            CHECK_EQ(m.psk_cr_cheaper, (int64_t)0);
            /* ds41-pskcr2: the replay's end, then the K4 anchor = the tail (e9e529d took the tail only) */
            CHECK((f.pc.committed == std::vector<int64_t>{ 22528, 30720 }));
        } else {
            CHECK_EQ(st[0].ctx, (int64_t)14336);
            CHECK_EQ(st[0].replay, 0);
            CHECK_EQ(st[0].flag, 0);
            CHECK_EQ(st[0].restore, 5);              /* the anchor's snapshot */
            for (const CrStep& c : st) CHECK_EQ(c.flag, 0);
            CHECK_EQ(m.psk_cr_hits + m.psk_cr_rows + m.psk_cr_resume_tokens + m.psk_cr_fb_unplanned + m.psk_cr_fb_geometry +
                         m.psk_cr_fb_adopt + m.psk_cr_not_needed + m.psk_tail_only + m.psk_tail_only_refused +
                         m.psk_cr_cheaper + m.psk_cr_end_ckpts, (int64_t)0);
            CHECK_EQ(m.psk_fallback_resumes, (int64_t)1);
            CHECK_EQ(m.psk_cr, 0);
        }
    }
}

/* The resume arithmetic itself (Scheduler::psk_cr_resume) on the production grid: the grid point under the match (and
 * under the prompt's last token), above the snapshot; the start the grid point under it less the reach. ds41-pskcr2: no
 * replay (cheaper) where the snapshot (0: the prompt's start) lies at most one grid step (4096) below that point. */
TEST(psk_cr_resume_is_the_grid_point_under_the_hit_and_its_cone) {
    Fixture f;
    cr_fixture(f);
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.set_psk_cr(1, 0), 0);                /* no reach: no replay */
    CHECK_EQ(f.sch.psk_cr_resume(26000, 32000, 0).at, (int64_t)-1);
    CHECK_EQ(f.sch.set_psk_cr(1, 5152), 1);
    struct W { int64_t attn, prompt, snap, at, from; bool cheaper; };
    for (const W w : { W{ 26000, 32000, 14336, 22528, 14336, false }, W{ 26000, 32000, 0, 22528, 14336, false },
                       W{ 22528, 32000, 0, 22528, 14336, false }, W{ 22527, 32000, 0, 18432, 10240, false },
                       W{ 26000, 22529, 0, 22528, 14336, false }, W{ 26000, 22528, 0, 18432, 10240, false },
                       W{ 6144, 9000, 0, 6144, 0, false }, W{ 2047, 9000, 0, -1, -1, false },
                       W{ 26000, 32000, 22528, -1, -1, false }, W{ 136000, 140000, 120832, 133120, 124928, false },
                       /* ds41-pskcr2: the first chunk's grid point (a cold prefill of it is cheaper than its replay over
                        * all 40 layers: gpu02 scr2 hit 4096 C +0.32 s), one grid step above the snapshot (the 24576 /
                        * 40960 / turn cases: one chunk vs two replay passes), off-grid snapshots */
                       W{ 2048, 9000, 0, -1, -1, true }, W{ 4000, 4100, 0, -1, -1, true },
                       W{ 26000, 32000, 18432, -1, -1, true }, W{ 40496, 41000, 34816, -1, -1, true },
                       W{ 44160, 44272, 38912, -1, -1, true }, W{ 26000, 32000, 18431, 22528, 14336, false },
                       W{ 40496, 41000, 30720, 38912, 30720, false }, W{ 26000, 32000, 18448, -1, -1, true } }) {
        const Scheduler::CrResume c = f.sch.psk_cr_resume(w.attn, w.prompt, w.snap);
        CHECK_EQ(c.at, w.at);
        CHECK_EQ(c.from, w.from);
        CHECK_EQ(c.cheaper, w.cheaper);
    }
}

/* ds41_psk 4 = tail-only: with ds41_psk_cr 1 a cold prompt's only targets are its tail snapshot and its end (no K-th grid
 * point), so the one snapshot is the tail's; with ds41_psk_cr 0 the request plans as 2 (K4) and is counted refused. */
TEST(psk_cr_tail_only_anchors_need_the_replay) {
    for (int cr : { 1, 0 }) {
        Fixture f;
        cr_fixture(f);
        CHECK_OK(f.start());
        f.sch.set_span_min(16384);
        f.sch.set_psk_mode(4);
        f.sch.set_psk_k(4);
        f.sch.set_psk_cr(cr, 5152);
        const uint64_t a = f.submit(30000, 1);
        const std::vector<CrStep> st = cr_run(f, a);
        for (const CrStep& c : st) CHECK_EQ(c.flag, 0);
        const SchedMetrics m = f.sch.metrics();
        if (cr) {
            CHECK((f.pc.committed == std::vector<int64_t>{ 26624 }));      /* the tail only */
            CHECK_EQ(m.psk_tail_only, (int64_t)1);
            CHECK_EQ(m.psk_tail_only_refused, (int64_t)0);
            CHECK_EQ(st[0].ahead, (int64_t)(26624 - 2048));                 /* [0, 2048)'s next target: the tail */
        } else {
            CHECK((f.pc.committed == std::vector<int64_t>{ 14336, 26624 }));   /* K4's q_4 and the tail */
            CHECK_EQ(m.psk_tail_only, (int64_t)0);
            CHECK_EQ(m.psk_tail_only_refused, (int64_t)1);
        }
    }
}

/* The other outcomes: a snapshot already at the grid point (no replay needed: resumed there), a request not planned
 * (ds41_psk 0: the fallback counted), a trunk without a window group (the geometry: counted), and a shared head past the
 * replay's end -- the request takes its snapshot there (Request::checkpoint_at) on its way after the replay. */
TEST(psk_cr_not_needed_unplanned_geometry_and_a_shared_head) {
    {   /* the snapshot at 22528 */
        Fixture f;
        cr_fixture(f);
        CHECK_OK(f.start());
        f.sch.set_psk_mode(2);
        f.sch.set_psk_k(4);
        f.sch.set_psk_cr(1, 5152);
        f.pc.seed(26000);
        f.pc.lin = 22528;
        f.pc.src = 3;
        const std::vector<CrStep> st = cr_run(f, f.submit(32000, 1));
        REQUIRE(!st.empty());
        CHECK_EQ(st[0].ctx, (int64_t)22528);
        CHECK_EQ(st[0].restore, 3);
        CHECK_EQ(f.sch.metrics().psk_cr_not_needed, (int64_t)1);
        CHECK_EQ(f.sch.metrics().psk_cr_hits, (int64_t)0);
    }
    {   /* not planned */
        Fixture f;
        cr_fixture(f);
        CHECK_OK(f.start());
        f.sch.set_psk_mode(0);
        f.sch.set_psk_cr(1, 5152);
        f.pc.seed(26000);
        f.pc.lin = 14336;
        f.pc.src = 5;
        const std::vector<CrStep> st = cr_run(f, f.submit(32000, 1));
        REQUIRE(!st.empty());
        CHECK_EQ(st[0].ctx, (int64_t)14336);
        CHECK_EQ(st[0].replay, 0);
        CHECK_EQ(f.sch.metrics().psk_cr_fb_unplanned, (int64_t)1);
    }
    {   /* no window group: the trunk has no state a replay rebuilds (the hit as before) */
        Fixture f;
        f.build(8192, 64, 4160, 60000, /*windowed=*/false, /*linear=*/true);
        f.cfg.checkpoint_interval = 2048;
        f.cfg.prefill_span = 2;
        f.cfg.prefix_exact = true;
        CHECK_OK(f.start());
        f.sch.set_psk_mode(2);
        f.sch.set_psk_k(4);
        f.sch.set_psk_cr(1, 5152);
        f.pc.seed(26000);
        f.pc.lin = 14336;
        f.pc.src = 5;
        const uint64_t a = f.submit(32000, 1);
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        CHECK_EQ(b->ctx_lens[0], 14336);
        CHECK(!f.sch.plan().e[0].cr_replay);
        CHECK_EQ(f.sch.metrics().psk_cr_fb_geometry, (int64_t)1);
        (void)a;
    }
    {   /* a head of 25600 tokens other prompts share: replayed to 22528, then a chunk split at the head, its snapshot */
        Fixture f;
        cr_fixture(f);
        CHECK_OK(f.start());
        f.sch.set_span_min(16384);
        f.sch.set_psk_mode(4);
        f.sch.set_psk_k(4);
        f.sch.set_psk_cr(1, 5152);
        f.pc.seed(25600);
        f.pc.branch_at = 25600;
        const uint64_t a = f.submit(32000, 1);
        const std::vector<CrStep> st = cr_run(f, a);
        REQUIRE(st.size() >= 4);
        CHECK_EQ(f.sch.find(a) ? f.sch.find(a)->checkpoint_at : (int64_t)25600, (int64_t)25600);
        CHECK_EQ(st[0].ctx, (int64_t)14336);
        CHECK_EQ(st[0].flag, 1);
        CHECK_EQ(st[1].flag, 1);
        CHECK_EQ(st[2].ctx, (int64_t)22528);
        CHECK_EQ(st[2].n, (int64_t)(25600 - 22528));    /* the head */
        CHECK_EQ(st[2].ckpts, 1);
        CHECK_EQ(st[2].ahead, (int64_t)0);
        /* ds41-pskcr2: the replay's end, the head and the tail (tail-only) */
        CHECK((f.pc.committed == std::vector<int64_t>{ 22528, 25600, 30720 }));
        CHECK_EQ(f.sch.metrics().psk_cr_end_ckpts, (int64_t)1);
        CHECK_EQ(f.sch.metrics().psk_cr_hits, (int64_t)1);
    }
}

/* ================================================================== ds41-pskcr2: conversations over the replay */
/* One request of a conversation on the production geometry, under arm B (ds41_psk 2 k 4, ds41_psk_cr 0), C (2 k 4, cr 1)
 * or T (4 = tail-only, cr 1): a fresh scheduler whose cache holds the conversation's global KV up to `match` and the
 * snapshots `avail` (positions on this request's path); the lookup gives the deepest of them at or below min(match,
 * n - 1), with a slot -- the real cache's (prefix.cpp: the deepest snapshot under the match). The turn's steps (cr_run:
 * the batch masks every step), the snapshots it committed and its counters. */
struct CrTurn { std::vector<CrStep> st; std::vector<int64_t> snaps; SchedMetrics m; int64_t lin = 0; };
static CrTurn cr_turn(char arm, int64_t n, int64_t match, const std::vector<int64_t>& avail, bool ti = false) {
    Fixture f;
    f.build(/*blocks=*/16384, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/70000, /*windowed=*/true, /*linear=*/true);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefix_exact = true;
    f.pc.groups = { 0 };
    f.kv.snap = { 3 };
    CHECK_OK(f.start());
    f.sch.set_span_min(16384);
    f.sch.set_psk_mode(arm == 'T' ? 4 : 2);
    f.sch.set_psk_k(4);
    CHECK_EQ(f.sch.set_psk_cr(arm == 'B' ? 0 : 1, 5152), arm == 'B' ? 0 : 1);
    if (ti) f.sch.set_tail_inner_capable(true);   /* lever TI (ds41-pf2): the engine's probe found a plugin serving it */
    CrTurn t;
    if (match > 0) {
        f.pc.seed((int32_t)match);
        for (int64_t p : avail)
            if (p <= std::min(match, n - 1)) t.lin = std::max(t.lin, p);
        f.pc.lin = (int32_t)t.lin;
        f.pc.src = t.lin > 0 ? 7 : -1;
    }
    const uint64_t id = f.submit((int)n, 1);
    t.st = cr_run(f, id, 128);
    t.snaps = f.pc.committed;
    t.m = f.sch.metrics();
    return t;
}
static std::vector<int64_t> cr_path(const std::vector<int64_t>& a, int64_t match) {   /* the snapshots on a path */
    std::vector<int64_t> o;
    for (int64_t p : a)
        if (p <= match) o.push_back(p);
    return o;
}
/* A conversation: a cold text, then turns, each `match` tokens shared with the previous prompt (turn 1: with the cold
 * text). Returns the replays it took; every turn's first step in `start` (its ctx; -1 when it replayed) and whether it
 * restored a snapshot. `drop_end` (a control): the replay's end snapshot taken out of the cache before the next turn, as
 * ds41-pskcr e9e529d never took it. */
struct CrChain { int64_t replays = 0; std::vector<int64_t> start; std::vector<int> restored; std::vector<CrTurn> t; };
static CrChain cr_chain(char arm, int64_t cold, const std::vector<std::pair<int64_t, int64_t>>& turns, bool drop_end = false) {
    CrChain c;
    std::vector<int64_t> avail = cr_turn(arm, cold, 0, {}).snaps;
    for (const auto& tn : turns) {
        avail = cr_path(avail, tn.second);
        CrTurn t = cr_turn(arm, tn.first, tn.second, avail);
        CHECK(!t.st.empty());
        if (t.st.empty()) return c;
        c.replays += t.m.psk_cr_hits;
        c.start.push_back(t.st[0].replay ? -1 : t.st[0].ctx);
        c.restored.push_back(t.st[0].restore >= 0 ? 1 : 0);
        for (int64_t p : t.snaps)
            if (!(drop_end && t.m.psk_cr_hits > 0 && p == t.m.psk_cr_resume_tokens)) avail.push_back(p);
        std::sort(avail.begin(), avail.end());
        avail.erase(std::unique(avail.begin(), avail.end()), avail.end());
        c.t.push_back(std::move(t));
    }
    return c;
}

/* The defect r00zzz-cz-tz-scr2 showed (HIT turn:40960: B 1.21 s, C / T 2.64 / 2.67 s replaying 8192 again at 34816): a
 * request served by a replay took no snapshot at the replay's end, and its tail snapshot fell there (a prompt that ends
 * within a grid step of H), so every next turn of the conversation replayed again. A Claude Code conversation: a cold text
 * of 52000 (K4: snapshots 14336 30720 47104 and the tail 51200; tail-only: 51200), then turn 1 sharing 40496 tokens of it
 * (the grid point 38912: two grid steps over K4's anchor 30720 -- C replays from 30720; T, with no anchor under it, too; B
 * falls back to 30720), then three turns, each the previous prompt + its answer + a new message (turn 3's answer of ~3000
 * tokens moves the grid point under its match one step past turn 2's tail snapshot). Every arm: turn 1 the only replay;
 * turns 2-4 resume from a snapshot (the replay's end = turn 1's tail, then turn 3's tail), restored, no replay chunk --
 * as B's do. Then the control: the same conversation under C / T with the replay's end snapshot dropped (e9e529d) replays
 * again at turn 2. */
TEST(psk_cr_a_conversation_replays_once_then_resumes_from_its_snapshots) {
    const std::vector<std::pair<int64_t, int64_t>> turns = { { 41000, 40496 }, { 41168, 41056 }, { 44272, 44160 },
                                                             { 44432, 44320 } };
    for (char arm : { 'B', 'C', 'T' }) {
        const CrChain c = cr_chain(arm, 52000, turns);
        REQUIRE(c.t.size() == 4);
        CHECK_EQ(c.replays, (int64_t)(arm == 'B' ? 0 : 1));
        CHECK_EQ(c.start[0], (int64_t)(arm == 'B' ? 30720 : -1));
        if (arm != 'B') {
            CHECK_EQ(c.t[0].st[0].ctx, (int64_t)30720);              /* the replay from the grid point under 38912 - 5152 */
            CHECK_EQ(c.t[0].m.psk_cr_resume_tokens, (int64_t)38912);
            CHECK_EQ(c.t[0].m.psk_cr_end_ckpts, (int64_t)1);
        }
        CHECK((c.t[0].snaps == std::vector<int64_t>{ 38912 }));     /* the replay's end = the tail (B: the tail) */
        CHECK((c.start == std::vector<int64_t>{ arm == 'B' ? 30720 : -1, 38912, 38912, 43008 }));
        for (size_t k = 1; k < 4; ++k) {
            CHECK_EQ(c.restored[k], 1);
            CHECK_EQ(c.t[k].m.psk_cr_hits, (int64_t)0);
            for (const CrStep& s : c.t[k].st) CHECK_EQ(s.replay, 0);
        }
        if (arm != 'B') {
            CHECK_EQ(c.t[1].m.psk_cr_not_needed, (int64_t)1);
            CHECK_EQ(c.t[2].m.psk_cr_cheaper, (int64_t)1);           /* 43008 one step over 38912: resumed there */
            CHECK_EQ(c.t[3].m.psk_cr_not_needed, (int64_t)1);
        }
        CHECK((c.t[2].snaps == std::vector<int64_t>{ 43008 }));
    }
    /* the control: without the replay's end snapshot the next turn replays again -- the check above must see it */
    for (char arm : { 'C', 'T' }) {
        const CrChain c = cr_chain(arm, 52000, turns, /*drop_end=*/true);
        CHECK_EQ(c.start[1], (int64_t)-1);                    /* turn 2 replays again ... */
        CHECK(c.replays >= 2);                                /* ... and the at-most-once check above fails */
        CHECK_EQ(c.t[1].m.psk_cr_hits, (int64_t)1);
    }
}

/* The scr2 hits that read slower than B (hit 4096: C +0.32 s; hit 24576: C +0.76 s), on the cold text above:
 * - 4096 (a prompt of 4100, match 4000): the grid point under it is the first chunk's end 2048 and nothing lies under
 *   it -- e9e529d replayed [0, 2048) over all 40 layers' cones of 2048 (82 % of its layer-rows; B's cold chunk runs 21
 *   layers + 3 small cones toward its farther target): now no replay, a cold prefill (cheaper, counted);
 * - 24576 after 20480 (scr2's round-1 order on one text): 20480 (n 18917, match 18592) resumes at 18432 -- T by a replay
 *   (no snapshot under it), C at K4's 14336 one grid step below (cheaper: one chunk, no replay); either way the request
 *   takes its snapshot at 18432 (T: the replay's end = its tail). Then 24576 (n 22290; its text shares 18592 with
 *   20480's) finds that snapshot at its grid point 18432: no replay (e9e529d: none existed under C / T, it replayed
 *   [10240, 18432) where B resumed at 18432 -- the +0.76 s); and a 23747-token one (match 22608) whose grid point 22528
 *   lies one step over it resumes there (cheaper). */
TEST(psk_cr_the_first_chunk_and_a_snapshot_one_step_below_take_the_cheaper_path) {
    for (char arm : { 'C', 'T' }) {
        const CrChain c = cr_chain(arm, 52000, { { 4100, 4000 } });
        CHECK_EQ(c.replays, (int64_t)0);
        CHECK_EQ(c.start[0], (int64_t)0);                       /* cold */
        CHECK_EQ(c.restored[0], 0);
        CHECK_EQ(c.t[0].m.psk_cr_cheaper, (int64_t)1);
        for (const CrStep& s : c.t[0].st) { CHECK_EQ(s.replay, 0); CHECK_EQ(s.flag, 0); }
    }
    for (char arm : { 'B', 'C', 'T' }) {
        const CrChain c = cr_chain(arm, 52000, { { 18917, 18592 }, { 22290, 18592 }, { 23747, 22608 } });
        CHECK_EQ(c.replays, (int64_t)(arm == 'T' ? 1 : 0));
        CHECK_EQ(c.start[0], (int64_t)(arm == 'T' ? -1 : 14336));
        if (arm == 'C') CHECK_EQ(c.t[0].m.psk_cr_cheaper, (int64_t)1);
        if (arm == 'T') {
            CHECK_EQ(c.t[0].st[0].ctx, (int64_t)10240);
            CHECK_EQ(c.t[0].m.psk_cr_end_ckpts, (int64_t)1);
        }
        CHECK((c.t[0].snaps == std::vector<int64_t>{ 18432 }));
        CHECK_EQ(c.start[1], (int64_t)18432);
        CHECK_EQ(c.restored[1], 1);
        if (arm != 'B') CHECK_EQ(c.t[1].m.psk_cr_not_needed, (int64_t)1);
        CHECK_EQ(c.start[2], (int64_t)18432);
        if (arm != 'B') CHECK_EQ(c.t[2].m.psk_cr_cheaper, (int64_t)1);
        CHECK((c.t[2].snaps == std::vector<int64_t>{ 22528 }));
    }
}

TEST(a_chunk_ending_on_a_checkpoint_carries_it_into_the_batch) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.submit(1000, 8);

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, 256);
    CHECK_EQ(b->n_checkpoints, 1);
    /* The snapshot is the state after the chunk's last token, which the split put at the interval
     * boundary. */
    CHECK_EQ(b->checkpoint_tok[0], 255);
    CHECK_EQ(b->checkpoint_slot[0], 0);
    CHECK_EQ(b->phase, RAD_PHASE_PREFILL);
}

/* ================================================================== admission */
TEST(admission_refuses_when_the_kv_pool_is_short) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);   /* 20 x 16 = 320 tokens of paged KV */
    CHECK_OK(f.start());

    f.submit(160, 8);      /* 10 blocks */
    f.submit(160, 8);      /* 10 blocks -- exactly fills the pool */
    f.submit(160, 8);      /* must not be admitted */

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_seq, 2);
    CHECK_EQ(f.kv.free_blocks(0), 0);

    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.running, 2);
    CHECK_EQ(m.queue_depth, 1);
    CHECK_EQ(m.admitted, 2);
}

/* An observer reads the copy the engine thread published, never the live state, so what it sees
 * changes only at a publication, and a publication carries exactly what the live readers say. */
TEST(observers_read_what_the_engine_thread_published) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.submit(160, 8);
    f.submit(160, 8);
    f.submit(160, 8);
    CHECK(f.sch.step() != nullptr);

    std::vector<Scheduler::ReqStat> seen(8), live(8);
    CHECK_EQ(f.sch.observed_metrics().admitted, 0);            /* nothing published yet */
    CHECK_EQ(f.sch.observed_requests(seen.data(), 8), 0);

    f.sch.publish(/*idle=*/false);
    const SchedMetrics o = f.sch.observed_metrics();
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(o.running, m.running);
    CHECK_EQ(o.queue_depth, m.queue_depth);
    CHECK_EQ(o.admitted, 2);
    const int n = f.sch.observed_requests(seen.data(), 8);
    CHECK_EQ(n, f.sch.requests(live.data(), 8));
    CHECK_EQ(n, 3);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(seen[(size_t)i].id, live[(size_t)i].id);
        CHECK_EQ(seen[(size_t)i].state, live[(size_t)i].state);
        CHECK_EQ(seen[(size_t)i].prompt_tokens, live[(size_t)i].prompt_tokens);
    }

    /* An idle publication inside 20 ms of the last one is skipped; a step's never is. */
    f.submit(160, 8);
    f.sch.publish(/*idle=*/true);
    CHECK_EQ(f.sch.observed_metrics().queue_depth, 1);
    f.sch.publish(/*idle=*/false);
    CHECK_EQ(f.sch.observed_metrics().queue_depth, 2);
    CHECK_EQ(f.sch.observed_requests(seen.data(), 8), 4);
}

/* The sequences of one API request are added under one lock hold, each accepted or refused on its
 * own: a duplicate id is refused without taking its neighbours with it. */
TEST(add_all_queues_a_set_under_one_lock_and_refuses_members_singly) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    std::unique_ptr<Request> rs[3];
    for (int i = 0; i < 3; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 8;
        rs[i]->prompt.assign(32, 1000 + i);
    }
    rs[2]->id = rs[0]->id;
    int rc[3] = { -1, -1, -1 };
    f.sch.add_all(rs, 3, rc);
    CHECK_EQ(rc[0], RAD_OK);
    CHECK_EQ(rc[1], RAD_OK);
    CHECK_EQ(rc[2], RAD_E_DUPLICATE);

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_seq, 2);
    CHECK_EQ(f.sch.metrics().admitted, 2);
}

/* A scored prompt (Request::token_scores) names real vocabulary rows, and drafts nothing. */
TEST(a_scored_request_names_real_vocabulary_rows) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int32_t n_vocab = (int32_t)f.prog.meta.n_vocab;
    REQUIRE(n_vocab > 2);
    std::unique_ptr<Request> rs[4];
    const std::vector<std::vector<int32_t>> ids = { { 1, 2 }, {}, { n_vocab }, { -1 } };
    for (int i = 0; i < 4; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 1;
        rs[i]->prompt.assign(16, 7);
        rs[i]->token_scores = std::make_shared<TokenScores>();
        rs[i]->token_scores->tokens = ids[(size_t)i];
    }
    Request* kept = rs[0].get();
    int rc[4] = { -1, -1, -1, -1 };
    f.sch.add_all(rs, 4, rc);
    CHECK_EQ(rc[0], RAD_OK);
    CHECK_EQ(rc[1], RAD_E_INVAL);
    CHECK_EQ(rc[2], RAD_E_INVAL);
    CHECK_EQ(rc[3], RAD_E_INVAL);
    CHECK(!kept->may_speculate);
}

/* The candidate COUNT is capped as well as their values: a scored step stages the union of
 * these lists as pinned host rows, so their size must not be a client's choice at the seam.
 * 4096 is orders above what a decision request names (its written answers, a few dozen), so
 * exactly the cap admits and one past it is refused by name. */
TEST(a_scored_request_names_at_most_the_candidate_cap) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    std::unique_ptr<Request> rs[2];
    for (int i = 0; i < 2; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 1;
        rs[i]->prompt.assign(16, 7);
        rs[i]->token_scores = std::make_shared<TokenScores>();
        rs[i]->token_scores->tokens.assign(4096 + i, 1);
    }
    int rc[2] = { -1, -1 };
    f.sch.add_all(rs, 2, rc);
    CHECK_EQ(rc[0], RAD_OK);         /* exactly the cap */
    CHECK_EQ(rc[1], RAD_E_INVAL);    /* one past it */
}

/* The prompt itself names real vocabulary rows too: raw id arrays arrive verbatim from the
 * request body, and an id past the embedding table must die at admission, never staged into a
 * pass -- the embedding gather would fail the step, a failed step is a device fault, and a
 * device fault stops the engine (spec §17). One bad number must not be a server kill. */
TEST(a_prompt_names_real_vocabulary_rows) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int32_t n_vocab = (int32_t)f.prog.meta.n_vocab;
    std::unique_ptr<Request> rs[5];
    const std::vector<std::vector<int32_t>> ids = { { 1, 2, n_vocab - 1 }, { n_vocab }, { -1 },
                                                    { 7, INT32_MAX }, { 0, 1, 2, n_vocab + 9 } };
    int rc[5] = { -1, -1, -1, -1, -1 };
    for (int i = 0; i < 5; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 1;
        rs[i]->prompt = ids[(size_t)i];
    }
    f.sch.add_all(rs, 5, rc);
    CHECK_EQ(rc[0], RAD_OK);
    for (int i = 1; i < 5; ++i) CHECK_EQ(rc[i], RAD_E_INVAL);
}

TEST(a_prefix_hit_starts_the_request_at_the_last_usable_checkpoint) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);       /* the attention blocks match 304 tokens deep (19 blocks) */
    f.pc.lin = 256;       /* but the last linear checkpoint is at 256 */
    f.pc.src = 3;
    int64_t free_before = f.kv.free_blocks(0);

    uint64_t a = f.submit(1000, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    /* Recompute resumes from the checkpoint rather than replaying only the linear layers over
     * (256, 300], because RadBatch carries one query length for every KV group. The chunk still
     * ends on the next interval, so the next checkpoint is still written. */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(b->ctx_lens[0], 256);
    CHECK_EQ(b->n_tok, 256);
    CHECK_EQ(b->n_checkpoints, 1);
    CHECK_EQ(b->checkpoint_tok[0], 255);

    /* The recurrent half of the hit is carried exactly once. The adopted blocks describe the
     * attention layers; the linear ones resume from this slot alone, so an entry that skipped 256
     * tokens without naming a snapshot would continue from whatever the slot last held. */
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 3);

    /* The hit is claimed, not merely counted: the sequence adopted the cache's 16 blocks, which
     * cost the pool nothing since the cache already held them, then allocated only the 16 new
     * blocks its 256-token chunk needs. An unadopted hit would attend to blocks it does not own. */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 32);
    CHECK_EQ(f.kv.block_table(a, 0)[0], f.pc.seeded[0][0]);
    CHECK_EQ(f.kv.free_blocks(0), free_before - 16);

    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.prompt_tokens, 1000);
    CHECK_EQ(m.attn_cached_tokens, 304);     /* what the block cache matched */
    CHECK_EQ(m.linear_cached_tokens, 256);   /* what was actually skipped */
    CHECK_EQ(m.checkpoints_written, 1);
    /* The checkpoint shortfall: the 48 cached tokens above the snapshot are computed again. */
    CHECK_EQ(m.hit_cut_tokens, 48);
    CHECK_EQ(m.hit_cut_no_snapshot, 0);
    CHECK_EQ(m.hit_cut_resumed_below, 0);    /* within an interval: a turn's ordinary tail */

    /* The snapshot this step takes is not offered to anyone until the step has run: the
     * reservation happens when the chunk is planned, the state exists only afterwards. */
    CHECK(f.pc.committed.empty());
    f.commit_plain();
    CHECK_EQ((int)f.pc.committed.size(), 1);
    CHECK_EQ(f.pc.committed[0], 512);        /* ctx_len 256 + ckpt_tok 255 + 1 */
}

/* A hit that resumes more than an interval below the attention blocks is counted with its own
 * reason: the snapshots in between are gone, or a restore brought back only part of the chain,
 * and hit_cut_tokens alone would not say how many admissions it was. */
TEST(a_hit_resuming_far_below_its_blocks_is_counted_as_cut) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(784);       /* 49 blocks */
    f.pc.lin = 256;       /* 528 tokens below, past the 256-token interval */
    f.pc.src = 3;

    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.hit_cut_resumed_below, 1);
    CHECK_EQ(m.hit_cut_no_snapshot, 0);
    CHECK_EQ(m.hit_cut_tokens, 528);
}

/* A restore that came back cut is counted once, by the admission it served: a request refused
 * and asked again restores again, and only the attempt that admits it counts. */
TEST(a_cut_restore_is_counted_once_per_admission) {
    Fixture f;
    f.build(/*blocks=*/12, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.restore_result = RestoreResult::Cut;

    uint64_t a = f.submit(160, 8);                  /* ten blocks of twelve */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.sch.metrics().prefix_restore_cuts, 1);

    uint64_t b = f.submit(160, 8);                  /* no room beside a: refused, asked again */
    (void)f.sch.step();
    (void)f.sch.step();
    CHECK(f.pc.restores.size() >= 3);
    CHECK_EQ(f.sch.find(b)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().prefix_restore_cuts, 1);
    (void)a;
}

/* A hit that skipped tokens with no snapshot to resume from would corrupt the linear state, so it
 * is refused outright rather than served with a zeroed state. */
TEST(a_hit_with_no_checkpoint_slot_reuses_nothing) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = -1;        /* the cache found a prefix but no snapshot covering it */

    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().linear_cached_tokens, 0);
    /* Counted once, with its reason and the whole hit as the shortfall. */
    CHECK_EQ(f.sch.metrics().hit_cut_no_snapshot, 1);
    CHECK_EQ(f.sch.metrics().hit_cut_tokens, 304);
}

/* A restore from a slot the store gave back is never built. The cache keeps a handed-out snapshot
 * until the restoring step has run, so this is a guard: the step is abandoned, the request gives
 * its hit back and requeues, and the next step prefills it from a fresh lookup. */
TEST(a_restore_from_a_slot_the_store_gave_back_is_prefilled_again) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = 3;
    f.kv.ck_gone = {3};
    const int64_t free_before = f.kv.free_blocks(0);

    const uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.kv.free_blocks(0), free_before);      /* the adopted blocks went back */

    /* The next lookup finds no snapshot, and the request runs from zero. */
    f.pc.src = -1;
    f.kv.ck_gone.clear();
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->ctx_lens[0], 0);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, -1);
}

TEST(a_hit_that_no_checkpoint_covers_replays_from_zero) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(192);
    f.pc.lin = 0;         /* nothing checkpointed yet */

    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().linear_cached_tokens, 0);
}

/* A hit is refused on a model with a paged group the prefix cache does not index. Here the
 * windowed group is paged, bound to a trunk layer, and skipped by the cache (prefix.cpp recycles
 * its blocks below the window), which pc.groups = {0} reproduces. The hit is deep and
 * checkpoint-covered, so every other clamp passes it. Adopting it would cover group 0 only, while
 * the windowed trunk layers read positions below ctx_len this sequence never wrote: the previous
 * request's KV, read as this one's prefix. No tail recompute repairs a trunk hole. */
TEST(a_hit_on_a_model_with_a_windowed_group_reuses_nothing) {
    Fixture f;
    f.build(4096, 64, 512, 4096, /*windowed=*/true);
    f.pc.groups = {0};    /* a cache over the FULL group only (windowed groups are not
                           * prefix-cacheable); must precede start(), which reads it once */
    CHECK_OK(f.start());
    f.pc.seed(304);       /* a deep, block-aligned attention hit: 19 blocks */
    f.pc.lin = 256;       /* its linear state checkpoints at 256: the resume point exists */
    f.pc.src = 3;         /* and names a snapshot, so no checkpoint clamp applies */

    uint64_t a = f.submit(300, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);

    /* The whole hit is refused, not trimmed: the request runs from zero. */
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.find(a)->n_cached, 0);
    CHECK_EQ(b->ctx_lens[0], 0);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, -1);

    /* Nothing is adopted: the 16 blocks covering the 256-token chunk are all freshly grabbed. */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 16);
    CHECK(std::find(f.pc.seeded[0].begin(), f.pc.seeded[0].end(),
                    f.kv.block_table(a, 0)[0]) == f.pc.seeded[0].end());

    /* The cache still reported the match, and the metrics say so: this is a real hit the
     * scheduler refused, not a lookup that found nothing. */
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.attn_cached_tokens, 304);
    CHECK_EQ(m.linear_cached_tokens, 0);
}

/* The same model with the trunk's window in the checkpoint slot (KVGroupPlan::win_snap beside
 * linear state, as DeepSeek-V4.1's sliding layers beside its compressor state): the hit resumes at
 * the snapshot, which restores the linear state and the window together, and the cached group's
 * blocks are adopted. */
TEST(a_hit_on_a_hybrid_whose_slot_carries_the_window_resumes_at_the_snapshot) {
    Fixture f;
    f.build(4096, 64, 512, 4096, /*windowed=*/true);
    const int32_t swa = 3;
    f.kv.snap = { swa };
    f.pc.groups = { 0 };
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = 3;

    const uint64_t a = f.submit(300, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(b->ctx_lens[0], 256);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 3);
    CHECK_EQ(f.kv.block_table(a, 0)[0], f.pc.seeded[0][0]);
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.linear_cached_tokens, 256);
    CHECK_EQ(m.hit_cut_window, 0);

    /* A hit with no snapshot under it still reuses nothing: the window has nowhere to come from. */
    Fixture g;
    g.build(4096, 64, 512, 4096, /*windowed=*/true);
    g.kv.snap = { swa };
    g.pc.groups = { 0 };
    CHECK_OK(g.start());
    g.pc.seed(304);
    g.pc.lin = 0;
    g.pc.src = -1;
    const uint64_t c = g.submit(300, 8);
    REQUIRE(g.sch.step() != nullptr);
    CHECK_EQ(g.sch.find(c)->n_computed, 0);
}

/* A trunk group the cache does not index and no slot carries is a hole no snapshot fills: a full
 * group left out refuses the hit even where the window is in the slot. */
TEST(a_trunk_group_neither_cached_nor_in_the_slot_refuses_the_hit) {
    Fixture f;
    f.build(4096, 64, 512, 4096, /*windowed=*/true);
    KVGroupInfo k;
    k.name = "fine";
    k.decl.kind = RAD_KV_FULL;
    k.block_size = 16;
    k.layers = { 3 };
    f.prog.kv_groups.push_back(k);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.kv.snap = { 3 };
    f.pc.groups = { 0 };
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = 3;
    const uint64_t a = f.submit(300, 8);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().hit_cut_window, 1);
}

/* Groups at two block sizes: a hit adopts each group's own blocks (the 8-token group twice as
 * many), and the cache's reserve is asked for in entries, from the group furthest short. */
TEST(a_hit_over_mixed_block_sizes_adopts_each_groups_own_blocks) {
    Fixture f;
    f.build(4096, 64);
    KVGroupInfo k;
    k.name = "fine";
    k.decl.kind = RAD_KV_FULL;
    k.block_size = 8;
    f.prog.kv_groups.push_back(k);
    const int32_t fine = 3;
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0, fine };
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().quantum % 16, 0ll);
    f.pc.seed(304);                                   /* 19 blocks of 16, 38 of 8 */
    f.pc.lin = 256;
    f.pc.src = 3;
    const uint64_t a = f.submit(1000, 8);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    const std::vector<int32_t> t0 = f.kv.block_table(a, 0), t1 = f.kv.block_table(a, fine);
    CHECK_EQ((int)t0.size(), 32);                     /* 16 adopted + 16 for the 256-token chunk */
    CHECK_EQ((int)t1.size(), 64);                     /* 32 adopted + 32 */
    for (int i = 0; i < 16; ++i) CHECK_EQ(t0[(size_t)i], f.pc.seeded[0][(size_t)i]);
    for (int i = 0; i < 32; ++i) CHECK_EQ(t1[(size_t)i], f.pc.seeded[1][(size_t)i]);
}

TEST(the_cache_trim_over_mixed_block_sizes_asks_for_entries) {
    Fixture f;
    f.build(/*blocks=*/64, /*states=*/64);
    KVGroupInfo k;
    k.name = "fine";
    k.decl.kind = RAD_KV_FULL;
    k.block_size = 8;
    f.prog.kv_groups.push_back(k);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 64, 64);
    f.pc.groups = { 0, 3 };
    CHECK_OK(f.start());
    f.pc.evict_grant = 1 << 20;
    f.submit(160, /*max_tokens=*/1);                 /* 10 + 20 blocks, finishes on the first commit */
    f.submit(176, /*max_tokens=*/64);                /* 11 + 22 each, still running after it */
    f.submit(176, /*max_tokens=*/64);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.held_blocks(0), 32ll);             /* 32 free against a reserve of 16 */
    CHECK_EQ(f.kv.held_blocks(3), 64ll);             /* none free against 16: 8 entries of 2 */
    f.commit_plain();
    REQUIRE(f.pc.trims.size() == 1u);
    CHECK_EQ(f.pc.trims[0], 8ll);
}

/* A request whose media is not encoded yet waits, and only it does: it is not admitted without
 * its rows, and the request behind it is asked and runs. Once its items are encoded it is
 * admitted where it stood. */
TEST(a_request_waiting_on_its_encoder_lets_the_queue_past_it) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    std::shared_ptr<mm::Item> ia, ic;
    const uint64_t a = f.submit_unencoded(160, 8, &ia);
    const uint64_t b = f.submit(160, 8);
    const uint64_t c = f.submit_unencoded(160, 8, &ic);

    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_seq, 1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.pc.lookups, 1);                    /* nothing looked up for the ones that wait */
    uint64_t q[8];
    int n = f.sch.queue_snapshot(q, 8);
    REQUIRE_EQ(n, 2);
    CHECK_EQ(q[0], a);                            /* still in the order they came */
    CHECK_EQ(q[1], c);
    f.commit_plain();

    /* Asked again every step, and admitted the step its items are in. */
    ia->encoded = true;
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 2);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
    f.commit_plain();

    ic->encoded = true;
    const RadBatch* s3 = f.sch.step();
    CHECK(s3 != nullptr);
    CHECK_EQ(s3->n_seq, 3);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.queue_snapshot(q, 8), 0);
    CHECK_EQ(f.sch.metrics().admitted, 3);
}

/* ================================================================== preemption */
TEST(preemption_takes_the_lowest_priority_victim_and_requeues_it_to_the_front) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64, /*priority=*/0);   /* arrives first, low priority */
    uint64_t b = f.submit(160, 64, /*priority=*/5);   /* arrives second, high priority */
    uint64_t c = f.submit(160, 64, /*priority=*/0);   /* stays waiting */
    CHECK(a && b && c);

    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_seq, 2);
    f.commit_plain();

    /* Both now decode at position 160, a block boundary: the next token needs a new block and
     * the pool has none. The victim must be A (lowest priority, and re-admitted last anyway),
     * not the head-of-line B. */
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 1);

    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);      /* preemption is by recompute */
    CHECK_EQ(f.sch.find(a)->state_slot, -1);

    /* Requeued to the front: it keeps its original arrival, so it sits ahead of C, which arrived
     * later at the same priority. */
    uint64_t q[8];
    int n = f.sch.queue_snapshot(q, 8);
    CHECK_EQ(n, 2);
    CHECK_EQ(q[0], a);
    CHECK_EQ(q[1], c);

    CHECK_EQ(f.sch.metrics().preemptions, 1);
    CHECK_EQ(f.sch.metrics().preempted_waiting, 1);
}

/* ================================================================== the step batch */
/* A PREEMPTED PREFILL WAITS FOR ROOM FOR THE WHOLE RECOMPUTE plus one block per running
 * sequence. Admitting it on a shortfall is what makes the prefill loop: the next decode takes
 * the block it just reserved, its next chunk is refused, and preempt_back() resets it again. */
TEST(a_preempted_prefill_waits_for_the_whole_prompt_and_running_headroom) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 60000;           /* old gate test: no fallback in its 80 steps */
    CHECK_OK(f.start());

    uint64_t a = f.submit(287, 64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);

    /* A takes 287 tokens (18 blocks), B 128 (8): the pool is full. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), 2);
    f.commit_plain();

    /* B's next token needs a new block; it is the lowest-priority victim and is reset. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    CHECK_EQ(f.sch.find(b)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().preemptions, 1);
    f.commit_plain();

    /* One block the fake pool has free and no sequence holds: the victim's own blocks plus one.
     * The gate still wants one block of headroom beside the running A. */
    f.kv.total[0] += 1;
    f.kv.refcnt[0].push_back(0);
    f.kv.freelist[0].push_back(26);

    /* B stays in the queue while A runs; it must not be admitted into a pool that has no room
     * for its whole prompt plus A's next block. */
    uint64_t q[8];
    CHECK_EQ(f.sch.queue_snapshot(q, 8), 1);
    CHECK_EQ(q[0], b);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);

    for (int i = 0; i < 80 && f.sch.find(a)->state != ReqState::Finished; ++i) {
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
        CHECK_EQ(f.sch.find(b)->n_computed, 0);
        f.commit_plain();
    }
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);

    /* A's blocks are back; B can now prefill its whole prompt. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.find(b)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().preemptions, 1);
    f.commit_plain();
}

/* A FRESH PREFILL's room gate: it starts only when its whole prompt plus one decode position
 * fits beside the runners' whole prompts and their next-token headroom. The old admission
 * reserved the first chunk alone, so a long prefill could reach 100% and then be preempted at
 * the first decode step -- the full prefill thrown away. */
TEST(a_fresh_prefill_waits_for_its_whole_prompt_and_running_headroom) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);   /* 20 x 16 = 320 tokens; prompts of 160 = 10 blocks each */
    f.cfg.max_seqs = 2;
    f.cfg.prefill_room_gate = true;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160, /*max_tokens=*/64, /*priority=*/5);
    const uint64_t b = f.submit(160, /*max_tokens=*/64, /*priority=*/0);

    /* A holds 10 blocks; B needs 10 for its prompt, plus one block for B's decode position and
     * one for A's next position: 12 > the 10 free, so B waits instead of starting a prefill it
     * would have to throw away. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), 1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
    f.commit_plain();

    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
}

/* DS41 QSA: a WINDOW group holds ~window tokens, not the whole prompt. Counting ceil(prompt /
 * block) against it keeps every long chat in Waiting while can_ever_fit (which caps at the
 * window) still says yes. Shrink the SWA pool so the unfixed gate would wait; the request must
 * still start. */
TEST(a_fresh_prefill_is_not_held_for_a_windowed_groups_full_prompt) {
    Fixture f;
    f.build(/*blocks=*/64, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/4096, /*windowed=*/true);
    f.cfg.max_seqs = 2;
    f.cfg.prefill_room_gate = true;
    int32_t win = -1;
    for (size_t i = 0; i < f.prog.kv_groups.size(); ++i)
        if (f.prog.kv_groups[i].decl.kind == RAD_KV_WINDOW) win = (int32_t)i;
    REQUIRE(win >= 0);
    f.kv.total[(size_t)win] = 8;
    f.kv.refcnt[(size_t)win].assign(8, 0);
    f.kv.freelist[(size_t)win].clear();
    for (int32_t b = 7; b >= 0; --b) f.kv.freelist[(size_t)win].push_back(b);
    CHECK_OK(f.start());

    const uint64_t a = f.submit(400, /*max_tokens=*/64);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
}

/* A long prefill gets a capped chunk when other requests are eligible: 4 eligible requests and a
 * 4096-token budget floor the cap at 1024 tokens, so the long prompt cannot take every token. */
TEST(a_long_prefill_chunk_is_capped_to_a_fair_share_when_others_exist) {
    Fixture f;
    f.build(/*blocks=*/512, /*states=*/64, /*max_tok=*/4096, /*max_ctx=*/4096);
    f.cfg.max_seqs = 4;
    f.cfg.long_prefill_token_threshold = 1024;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(3000, /*max_tokens=*/64, /*priority=*/5);
    for (int i = 0; i < 3; ++i) f.submit(300, /*max_tokens=*/64, /*priority=*/0);

    REQUIRE(f.sch.step() != nullptr);
    int64_t a_tokens = -1;
    for (const StepEntry& e : f.sch.plan().e)
        if (e.req && e.req->id == a) a_tokens = e.n_tokens;
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK(a_tokens > 0 && a_tokens <= 1024);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
}

/* The same threshold does not throttle a lone prefill: nobody is waiting behind it. */
TEST(a_lone_long_prefill_is_not_capped_by_the_long_prefill_threshold) {
    Fixture f;
    f.build(/*blocks=*/512, /*states=*/64, /*max_tok=*/4096, /*max_ctx=*/4096);
    f.cfg.max_seqs = 4;
    f.cfg.checkpoint_interval = 4096;       /* one chunk, not a 256-token interval */
    f.cfg.long_prefill_token_threshold = 1024;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(3000, /*max_tokens=*/64);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), 1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK(f.sch.plan().e[0].n_tokens > 1024);
}

/* The whole-recompute predicate counts blocks the cache can lend back, not entries it already
 * holds. With nothing running, the old gate never entered its victim loop, so a pool filled by
 * finished conversations kept an overdue preempted request in Preempted forever -- the daily's
 * "deadlocked at preempted" incident. The overdue pass reclaims cache entries before judging fit. */
TEST(a_preempted_prefill_reclaims_cache_blocks_with_no_running_request) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 0;
    f.cfg.max_seqs = 3;
    CHECK_OK(f.start());

    const uint64_t b = f.submit(287, 64, /*priority=*/0);
    const uint64_t a = f.submit(128, 64, /*priority=*/5);

    /* Both prompts fill the pool; B's next token is refused and B is preempted. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)2);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    f.commit_plain();

    /* Nothing runs after A goes: B is the only sequence, at zero, overdue. Put the pool's free
     * blocks in the cache so B's whole recompute no longer fits the free list alone. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));
    f.pc.evict_frees = true;
    f.pc.evict_grant = 1 << 20;
    f.pc.seed(160);                          /* 10 blocks cached, 16 left free */

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK(f.pc.evict_calls > 0);
    CHECK_EQ(f.sch.metrics().preempted_waiting, 0);
    f.commit_plain();
}

/* A leaked state slot (held by nobody, missing from the free list) used to make can_grow false
 * forever once nothing was running. The overdue pass puts the slot back and admits. */
TEST(an_idle_overdue_prefill_reclaims_a_leaked_state_slot) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/4);
    f.cfg.preempt_wait_ms = 0;
    f.cfg.max_seqs = 2;
    CHECK_OK(f.start());

    const uint64_t b = f.submit(287, 64, /*priority=*/0);
    const uint64_t a = f.submit(128, 64, /*priority=*/5);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    f.commit_plain();
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));

    for (size_t i = 0; i < f.kv.gs.size(); ++i)
        if (!f.kv.geom.g[i].paged) f.kv.state_free[i].clear();

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.metrics().preempted_waiting, 0);
    f.commit_plain();
}

/* can_grow on an id the manager has forgotten is false. That must not park an idle engine:
 * admit adds the sequence. */
TEST(an_idle_overdue_prefill_is_admitted_when_its_kv_sequence_is_gone) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/4);
    f.cfg.preempt_wait_ms = 0;
    f.cfg.max_seqs = 2;
    CHECK_OK(f.start());

    const uint64_t b = f.submit(287, 64, /*priority=*/0);
    const uint64_t a = f.submit(128, 64, /*priority=*/5);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    f.commit_plain();
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));
    f.kv.seqs.erase(b);

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK(f.kv.has_sequence(b));
    f.commit_plain();
}

/* Two prefills that cannot both stay up. Resetting the higher-priority one so the lower one can
 * start just swaps who is at zero, and the higher one is admitted again. The higher one must
 * finish its prompt and keep it. */
TEST(a_preempted_prefill_does_not_reset_a_higher_priority_prefill) {
    Fixture f;
    f.build(/*blocks=*/18, /*states=*/64, /*max_tok=*/256);
    f.cfg.checkpoint_interval = 64;
    f.cfg.preempt_wait_ms = 0;
    f.cfg.max_seqs = 2;
    CHECK_OK(f.start());
    f.pc.evict_frees = true;
    f.pc.evict_grant = 6;
    f.pc.seed(96);

    const uint64_t a = f.submit(192, 16, /*priority=*/5);
    const uint64_t b = f.submit(128, 16, /*priority=*/0);

    int64_t a_peak = 0;
    bool a_reset = false;
    for (int i = 0; i < 8; ++i) {
        if (!f.sch.step()) break;
        f.commit_plain();
        const Request* ra = f.sch.find(a);
        if (!ra) continue;
        if (ra->n_computed < a_peak) a_reset = true;
        if (ra->n_computed > a_peak) a_peak = ra->n_computed;
    }
    CHECK(!a_reset);
    CHECK(a_peak >= 192);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
}

/* Every program carries a sentinel KV group at index 0: decl.kind 0 (RAD_KV_FULL), block size 1, no
 * layers, and the manager's plan for it pages nothing, so it holds no blocks. The gate must not ask
 * that group for the prompt's blocks: it has none to give, and the preempted request would wait
 * for ever (engine_e2e_test's preemption case hung on it, with nothing left running). Here the
 * group is appended, not at index 0, so the fixture's bindings stay where they are. */
TEST(a_preempted_prefill_ignores_a_kv_group_that_holds_no_blocks) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    KVGroupInfo none;                        /* as the sentinel: kind 0, no block size, no layers */
    none.name = "none";
    f.prog.kv_groups.push_back(none);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 26, 64);
    const size_t ng = f.prog.kv_groups.size() - 1;
    f.kv.geom.g[ng].paged = false;           /* the manager pages nothing there ... */
    f.kv.total[ng] = 0;                      /* ... and owns no block of it */
    f.kv.freelist[ng].clear();
    f.kv.refcnt[ng].clear();
    for (int k = 63; k >= 0; --k) f.kv.state_free[ng].push_back(k);
    f.cfg.preempt_wait_ms = 60000;
    CHECK_OK(f.start());

    uint64_t a = f.submit(287, 64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), 2);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                /* B's next block: B is preempted */
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    f.commit_plain();

    for (int i = 0; i < 80 && f.sch.find(a)->state != ReqState::Finished; ++i) {
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
    }
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);

    /* A's blocks are back and nothing runs: B prefills its whole prompt again. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    f.commit_plain();
}

/* A preempted request publishes what it computed, as a finished one does: its prompt stays in the
 * cache, and the recompute adopts it instead of computing it again (a 96.6k-token prompt took about
 * 25 s of a pair of cards). The blocks stay reclaimable: the cache gives them back under pressure. */
TEST(a_preempted_request_publishes_its_prefix) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 60000;
    CHECK_OK(f.start());
    uint64_t a = f.submit(287, 64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    const size_t before = f.pc.published.size();
    CHECK(f.sch.step() != nullptr);
    REQUIRE_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    bool found = false;
    for (size_t i = before; i < f.pc.published.size(); ++i)
        if (f.pc.published[i].seq == b && f.pc.published[i].n_tokens >= 128) found = true;
    CHECK(found);
    f.commit_plain();
    (void)a;
}

/* The gate's wait is bounded: after --preempt-wait-ms an overdue preempted prefill gets room by

 * preempting the lowest-priority running request before the step is planned, so it cannot queue
 * forever behind a long decode (the 710 s stall in the daily's log). */
TEST(a_preempted_prefill_falls_back_to_a_running_victim_after_the_wait) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 0;               /* fallback on the next step */
    CHECK_OK(f.start());

    uint64_t a = f.submit(287, 64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);

    /* A and B fill the pool; B's next token needs a block and B is the victim. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), 2);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    f.commit_plain();

    /* B is the lower priority. Making room for it by resetting A just swaps who is at zero, and
     * A is re-admitted at once. B stays queued; A keeps what it has built. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.metrics().preemptions, 1);
    CHECK(f.sch.find(a)->n_computed > 0);
    f.commit_plain();
}

/* An overdue preempted prefill the pool still refuses keeps its place: nothing queued behind it is
 * admitted in that step, since the room the overdue path makes is for it. Asking the next request
 * instead let a request just forced off for it come straight back (its prompt published, so its
 * recompute needed almost nothing) and take that room again: under KV pressure the same victim was
 * forced off and re-admitted 26 times in 16 s while the overdue request waited. */
TEST(an_overdue_preempted_prefill_keeps_its_place_ahead_of_the_queue) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 0;               /* overdue from the next step */
    CHECK_OK(f.start());

    uint64_t a = f.submit(287, 64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);
    REQUIRE_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    f.commit_plain();

    /* B is overdue and does not fit beside A. C, queued behind it, would fit; it must wait. */
    uint64_t c = f.submit(16, 4, /*priority=*/0);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
        CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
        f.commit_plain();
    }
    /* Once A finishes, B goes in first. */
    for (int i = 0; i < 80 && f.sch.find(a) && f.sch.find(a)->state != ReqState::Finished; ++i) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    bool b_first = false;
    for (int i = 0; i < 4; ++i) {
        if (!f.sch.step()) continue;
        const ReqState bs = f.sch.find(b)->state, cs = f.sch.find(c)->state;
        if (bs == ReqState::Running) { b_first = true; f.commit_plain(); break; }
        CHECK(cs != ReqState::Running);
        f.commit_plain();
    }
    CHECK(b_first);
}

/* A lower-priority overdue prefill may stay queued while a higher-priority runner keeps its
 * blocks. Forcing that runner off just swaps who is at zero. The priority-5 request must not be
 * the one left in Preempted while someone else still runs. */
TEST(overdue_preempted_prefills_keep_forcing_until_one_fits) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);   /* 20 x 16 = 320 tokens */
    f.cfg.preempt_wait_ms = 0;
    f.cfg.max_seqs = 3;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 16, /*priority=*/5);
    uint64_t b = f.submit(80, 16, /*priority=*/3);
    uint64_t c = f.submit(48, 16, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    /* Fill until the pool refuses someone. Then the overdue path must not leave every preempted
     * request waiting while a runner keeps the blocks. */
    for (int i = 0; i < 8; ++i) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    const Request* ra = f.sch.find(a);
    CHECK(ra);
    CHECK(ra->state != ReqState::Preempted || f.sch.metrics().running == 0);
    CHECK(f.sch.metrics().running + f.sch.metrics().preempted_waiting > 0);
    (void)b; (void)c;
}

/* The overdue path fails a preempted prefill only when waiting cannot help. A restore that waits a
 * step for a deferred release is a wait: failing it on an idle engine lost two of ten preempted
 * requests under KV pressure once their prompts were published at the preemption (1. 11. 2026). */
TEST(an_overdue_preempted_prefill_whose_restore_waits_a_step_is_not_failed) {
    Fixture f;
    f.build(/*blocks=*/26, /*states=*/64);   /* 26 x 16 = 416 tokens */
    f.cfg.preempt_wait_ms = 0;
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    uint64_t a = f.submit(287, /*max_tokens=*/64, /*priority=*/5);
    uint64_t b = f.submit(128, 64, /*priority=*/0);
    /* B's growth meets A's blocks: B is preempted while A runs. */
    for (int i = 0; i < 8; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain();
        if (f.sch.find(b)->state == ReqState::Preempted) break;
    }
    REQUIRE_EQ((int)f.sch.find(b)->state, (int)ReqState::Preempted);
    REQUIRE_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);

    /* A is cancelled while a drafter pass names its blocks: they come back only at the next
     * commit, and nothing runs. */
    Scheduler::BlockRow rows[4];
    REQUIRE_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    REQUIRE(f.sch.block_context(1) != nullptr);
    CHECK_OK(f.sch.cancel(a));
    f.sch.block_end();
    REQUIRE(f.sch.has_deferred());
    REQUIRE_EQ(f.sch.metrics().running, 0);

    /* B's restore waits a step for that release; B must still be there after it. The cache
     * gives back what it holds when asked. */
    f.pc.decline_whole = true;
    f.pc.evict_frees = true;
    f.pc.evict_grant = 1 << 20;
    if (!f.sch.step() && f.sch.has_deferred()) f.sch.release_deferred();
    CHECK(f.sch.find(b)->state != ReqState::Failed);
    /* As the engine does when a step plans nothing: it drains, then releases what was deferred. */
    bool ran = false;
    for (int i = 0; i < 4 && !ran; ++i) {
        if (!f.sch.step()) {
            if (f.sch.has_deferred()) f.sch.release_deferred();
            continue;
        }
        ran = f.sch.find(b)->state == ReqState::Running;
        f.commit_plain();
    }
    CHECK(ran);
    CHECK(f.sch.find(b)->state != ReqState::Failed);
}

TEST(a_mixed_step_has_the_right_boundaries_and_slot_mapping) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t a = f.submit(100, 64);
    CHECK(a);
    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_tok, 100);
    f.commit_plain(4242);

    /* A is decoding at position 100; B arrives and prefills 64 tokens in the same step. */
    uint64_t b = f.submit(64, 64);
    CHECK(b);
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->phase, RAD_PHASE_MIXED);
    CHECK_EQ(s2->n_seq, 2);
    CHECK_EQ(s2->n_tok, 65);

    CHECK_EQ(s2->cu_seqlens[0], 0);
    CHECK_EQ(s2->cu_seqlens[1], 1);
    CHECK_EQ(s2->cu_seqlens[2], 65);
    CHECK_EQ(s2->q_lens[0], 1);
    CHECK_EQ(s2->q_lens[1], 64);
    CHECK_EQ(s2->ctx_lens[0], 100);
    CHECK_EQ(s2->ctx_lens[1], 0);
    CHECK_EQ(s2->max_q_len, 64);
    /* A ceiling on a coarse grid, not the exact longest context: 100 rounds up to the 1024-token
     * floor, so the step's arguments stay put while the context grows under it. */
    CHECK_EQ(s2->max_ctx_len, 1024);

    /* The decode row feeds the token that was sampled last step, not a prompt token. */
    CHECK_EQ(s2->token_ids[0], 4242);
    CHECK_EQ(s2->positions[0], 100);
    CHECK_EQ(s2->token_ids[1], 1000);
    CHECK_EQ(s2->positions[1], 0);

    /* Slot mapping for the paged group: block id times block size plus the offset in the block.
     * Position 100 is block 6, offset 4. */
    const RadKVGroupBatch& g0 = s2->kv[0];
    CHECK_EQ(g0.block_size, 16);
    const std::vector<int32_t>& bta = f.kv.block_table(a, 0);
    const std::vector<int32_t>& btb = f.kv.block_table(b, 0);
    CHECK_EQ(g0.slot_mapping[0], bta[6] * 16 + 4);
    /* B's first prefill token is the first slot of its first block. */
    CHECK_EQ(g0.slot_mapping[1], btb[0] * 16);
    /* Block tables are one row per sequence, pitched from the step's context bound: 1024 tokens
     * plus the longest query (64) and the draft lookahead (2) is 1090 tokens, 69 blocks of 16,
     * rounded up to the bound's 64-block grid: 128. Columns past a row's own blocks read -1. */
    CHECK_EQ(g0.block_table_pitch, 128);
    CHECK_EQ(g0.block_table[0], bta[0]);
    CHECK_EQ(g0.block_table[6], bta[6]);
    CHECK_EQ(g0.block_table[7], -1);
    CHECK_EQ(g0.block_table[128], btb[0]);
    CHECK_EQ(g0.seqused[0], 101);
    CHECK_EQ(g0.seqused[1], 64);
    CHECK_EQ(s2->kv_used_max_h[0], std::max(g0.seqused[0], g0.seqused[1]));

    /* The linear group is per sequence, not per token: every token of a sequence maps to its one
     * state instance, and the block table row is that slot. */
    const RadKVGroupBatch& g1 = s2->kv[1];
    CHECK_EQ(g1.slot_mapping[0], f.kv.state_slot(a, 1));
    CHECK_EQ(g1.slot_mapping[1], f.kv.state_slot(b, 1));
    CHECK_EQ(g1.state_index[0], f.kv.state_slot(a, 1));
    CHECK(f.kv.state_slot(a, 1) >= 0);
}

TEST(derived_metadata_is_computed_once_per_step_into_the_arena) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    f.submit(64, 64);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);

    const int32_t* ss = f.derived("seq_start");
    CHECK(ss != nullptr);
    CHECK_EQ(ss[0], 0);
    CHECK_EQ(ss[1], 1);
    CHECK_EQ(ss[2], 65);

    /* Chunk bounds never cross a sequence and split on absolute multiples of the chunk length.
     * The decode row spans 100..100 and the prefill row 0..63, so neither crosses 64: one chunk
     * each, boundaries at 0, 1, 65. */
    const int32_t* cb = f.derived("chunk_bounds");
    CHECK(cb != nullptr);
    CHECK_EQ(cb[0], 0);
    CHECK_EQ(cb[1], 1);
    CHECK_EQ(cb[2], 65);

    /* The conv index is the state slot's base in a rolling window of width-1 + num_spec + 1 = 4,
     * plus the cursor the accepted count moves. */
    const int32_t* ci = f.derived("conv_idx");
    CHECK(ci != nullptr);
    const StepPlan& p = f.sch.plan();
    int32_t cslot = f.kv.state_slot(p.e[0].seq, 2);   /* group 2 is the conv group */
    CHECK(cslot >= 0);
    CHECK_EQ(ci[0], cslot * 4 + (p.e[0].conv_cursor % 4));
}

/* RadBatch::lora_slot: every row of a sequence carries its request's runtime LoRA slot, -1 for a
 * base request, in the step's row order (decode rows first). A step with no adapter row carries
 * none, so the core issues no adapter op on it. */
TEST(the_step_batch_carries_each_rows_adapter_slot) {
    Fixture f;
    f.build();
    f.cfg.lora.max_loras = 2;
    CHECK_OK(f.start());

    f.submit(16, 1);
    const RadBatch* base = f.sch.step();
    CHECK(base != nullptr);
    CHECK_EQ(base->n_tok, (int64_t)16);
    CHECK(base->lora_slot == nullptr);
    f.commit_plain();

    auto r = f.request(100, 64);
    r->lora_slot = 1;
    r->lora_uid = 7;
    CHECK(f.add(std::move(r)) != 0);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK(b->lora_slot != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)100);
    for (int64_t i = 0; i < b->n_tok; ++i) CHECK_EQ(b->lora_slot[i], 1);
    f.commit_plain();

    f.submit(64, 64);
    b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)65);
    CHECK_EQ(b->lora_slot[0], 1);                          /* the adapter request's decode row */
    for (int64_t i = 1; i < 65; ++i) CHECK_EQ(b->lora_slot[i], -1);   /* the base prefill */
}

/* With no adapters served the field is null and the step slab is byte for byte what it was before
 * the field existed: the slot array is the only thing adapters add to it. */
TEST(no_adapters_no_slot_array) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.submit(32, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK(b->lora_slot == nullptr);

    Config on = f.cfg;
    on.lora.max_loras = 2;
    const int64_t off_bytes = BatchBuilder::device_bytes(f.prog, f.cfg, f.kv.geom);
    const int64_t on_bytes  = BatchBuilder::device_bytes(f.prog, on, f.kv.geom);
    CHECK(off_bytes > 0);
    CHECK_EQ(on_bytes - off_bytes, (f.cfg.max_tok * 4 + 255) / 256 * 256);
}

TEST(state_chunk_bounds_split_a_long_prefill_at_absolute_multiples) {
    DeriveInput in;
    StepPlan plan;
    int32_t cu[2] = { 0, 200 };
    std::vector<int32_t> pos(200);
    for (int i = 0; i < 200; ++i) pos[(size_t)i] = 100 + i;   /* starts mid-chunk on purpose */
    in.plan = &plan;
    in.cu_seqlens = cu;
    in.positions = pos.data();
    in.n_tok = 200;
    in.n_seq = 1;
    in.state_chunk = 64;

    int idx = derive_index("state_chunk_bounds");
    CHECK(idx >= 0);
    int32_t out[16];
    int64_t n = 0;
    CHECK_OK(derive_row(idx)->fn(in, out, 16, &n));
    /* Absolute 128, 192, 256 fall inside [100, 300): local 28, 92, 156, then the tail. */
    CHECK_EQ(n, 5);
    CHECK_EQ(out[0], 0);
    CHECK_EQ(out[1], 28);
    CHECK_EQ(out[2], 92);
    CHECK_EQ(out[3], 156);
    CHECK_EQ(out[4], 200);

    CHECK_EQ(derive_index("no_such_derivation"), -1);
}

/* ================================================================== cancellation */
TEST(cancellation_frees_blocks_immediately) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    f.commit_plain();

    /* Freed now, not at completion or at the next step: a cancelled request that kept its pool
     * share until it would have finished would shrink the pool under exactly the load that
     * produces cancellations. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);
    CHECK_EQ(f.sch.metrics().cancelled, 1);
    CHECK_OK(f.sch.reap(a));

    /* The one exception is a step still on the device, and it defers by one step, not one
     * request: releasing a block a running kernel is writing hands it to a new sequence
     * mid-step, which shows up as a quality problem rather than a crash. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    CHECK_OK(f.sch.cancel(b));
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);          /* out of the running set at once */
    CHECK_EQ(f.kv.free_blocks(0), before - 10);    /* but the blocks are still the device's */
    f.commit_plain();
    CHECK_EQ(f.kv.free_blocks(0), before);
    /* The commit must not append the token the step sampled for a request cancelled while that
     * step was running. */
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
}

/* A release of a conversation with a request of it in flight is refused, whether that request
 * is queued or running: the cache would skip its blocks as live and the request's close would
 * publish the whole chain again, so the client would have been told of a release that did not
 * happen. Once the request is gone the release reaches the cache with what the client asked. */
TEST(a_session_release_waits_for_the_conversations_own_request) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());

    auto r = f.request(160, 64);
    r->session_label = "chat-7";
    const uint64_t a = f.add(std::move(r));
    CHECK(a != 0);

    SessionRelease out;
    /* Queued, not yet admitted. */
    CHECK_EQ(f.sch.release_session("chat-7", 0, 0, &out), RAD_E_STATE);
    CHECK_EQ(f.sch.release_session("", a, 0, &out), RAD_E_STATE);
    CHECK_EQ(f.pc.released_calls.size(), 0u);

    /* Running. */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.sch.release_session("chat-7", 0, 32, &out), RAD_E_STATE);
    /* Another conversation is not in its way. */
    CHECK_OK(f.sch.release_session("chat-8", 0, 0, &out));
    REQUIRE_EQ(f.pc.released_calls.size(), 1u);
    CHECK(f.pc.released_calls[0].label == "chat-8");

    /* Finished: the release goes through with label, id and the kept prefix as given. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));
    CHECK_OK(f.sch.release_session("chat-7", 0, 48, &out));
    REQUIRE_EQ(f.pc.released_calls.size(), 2u);
    CHECK(f.pc.released_calls[1].label == "chat-7");
    CHECK_EQ(f.pc.released_calls[1].keep_tokens, 48ll);
    CHECK_EQ(out.entries, 5ll);

    /* Nothing named is a bad request, not a cache call. */
    CHECK_EQ(f.sch.release_session("", 0, 0, &out), RAD_E_INVAL);
    CHECK_EQ(f.pc.released_calls.size(), 2u);
}

/* A reap while the step is still on the device. Common: the server reaps as soon as a sink
 * finishes, and cancelling finishes it, so every mid-decode disconnect lands here. Releasing the
 * Request now would destroy it under commit(); returning its slot now would let a newly admitted
 * request take the entry and receive another client's sampled token. */
TEST(a_reap_during_the_open_step_is_deferred_to_commit) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);

    /* What Server::cancel_all and Server::retire do back to back on a disconnect. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));

    /* The Request and its slot outlive the reap by exactly one commit: neither the id nor the
     * blocks are released while a StepEntry still points at them. */
    CHECK(f.sch.find(a) != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);

    /* A request admitted in that window must not be given the slot the open step is holding. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.find(b) != nullptr);

    f.commit_plain();
    CHECK(f.sch.find(a) == nullptr);              /* released now, and exactly once */
    CHECK_EQ(f.kv.free_blocks(0), before);

    /* The request admitted in the window is untouched by the commit that reaped the other. */
    CHECK(f.sch.find(b) != nullptr);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    f.commit_plain();
    CHECK(f.sch.find(b) != nullptr);
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* ============================================ a block drafter's own cache across a prefix hit */
TEST(a_prefix_hit_does_not_stop_a_full_attention_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;                 /* copied into the scheduler at init, so it goes first */
    CHECK_OK(f.start());

    /* A real hit, both halves: the attention blocks and the linear checkpoint must reach the same
     * depth, or the scheduler recomputes from 0, ctx_len is 0 and the test passes vacuously. */
    f.pc.seed(256);
    f.pc.lin = 256;
    f.pc.src = 3;

    uint64_t a = f.submit(272, 64);
    CHECK(f.sch.step() != nullptr);                    /* recomputes 0..256 off the checkpoint */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                    /* 256..272, and this one yields a token */
    CHECK_EQ(f.sch.find(a)->n_computed, 272);
    /* Non-zero is the point: this is the floor the guard would take after a hit. */
    CHECK(f.sch.plan().e[0].ctx_len > 0);
    f.commit_plain(11);

    /* Window 0 is a drafter attending over its whole history, so it can never wait for the window
     * to slide past the hit. Every paged group here is prefix-cached, so the adopted blocks were
     * written by the previous sequence's own context pass: no hole, floor 0. A windowed drafter
     * whose group the cache skips is the draft-refill cases below. */
    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)a);
}

/* ====================================== a prefix hit and a windowed block drafter (draft refill)
 * The drafter's own windowed KV group is not prefix-cached, so a hit leaves it without K and V for
 * the cached tokens. The scheduler ends such a hit one drafter window before the prompt's end, at
 * a checkpoint, so the recompute covers the window and the drafter drafts from the first decode. */
namespace {
struct RefillRun {
    int64_t n_after_first = 0;     /* n_computed after the admitting step */
    int rows = -1;                 /* drafting rows block_begin offers at the first decode step */
    SchedMetrics m;
};
RefillRun refill_run(int n_prompt, int32_t seeded, int64_t ckpt_lo = 0) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 7;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.window = 512;
    KVGroupInfo w;                 /* the drafter's own group: paged, windowed, not prefix-cached,
                                    * unbound. The trunk clamp reads bound trunk layers only
                                    * (scheduler.h, hit_holes_trunk_), so it leaves this hole to
                                    * the refill. */
    w.name = "draft";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.window = 512;
    w.block_size = 16;
    f.prog.kv_groups.push_back(w);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
    f.pc.ckpt_lo = ckpt_lo;
    CHECK_OK(f.start());
    if (seeded > 0) { f.pc.seed(seeded); f.pc.lin = seeded; f.pc.src = 3; }
    RefillRun r;
    const uint64_t a = f.submit(n_prompt, 64);
    Scheduler::BlockRow rows[4];
    for (int st = 0; st < 40; ++st) {
        CHECK(f.sch.step() != nullptr);
        if (st == 0) r.n_after_first = f.sch.find(a)->n_computed;
        f.commit_plain(11);
        const int n = f.sch.block_begin(8, 7, 512, 99, rows, 4);
        f.sch.block_end();
        if (!f.sch.find(a)->output.empty()) { r.rows = n; break; }
    }
    r.m = f.sch.metrics();
    return r;
}
}  /* namespace */

TEST(the_refill_window_is_the_retention_policys_drafter_floor) {
    /* tip-only keeps a snapshot this deep below the tip, or a refill-capped hit has nothing to end
     * on and the request decodes without drafts (KeepTipOnly). Without a refill the floor is 0. */
    for (int spec : { 7, 0 }) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
        f.cfg.n_spec = spec;
        f.prog.drafter.kind = RAD_DRAFT_BLOCK;
        f.prog.drafter.window = 512;
        KVGroupInfo w;
        w.name = "draft";
        w.decl.kind = RAD_KV_WINDOW;
        w.decl.window = 512;
        w.block_size = 16;
        f.prog.kv_groups.push_back(w);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        f.pc.groups = { 0 };
        CHECK_OK(f.start());
        CHECK_EQ(f.pc.drafter_floor, spec ? (int64_t)512 : (int64_t)0);
    }
}

TEST(a_prefix_hit_ends_a_drafter_window_before_the_prompt_and_the_drafter_drafts_at_once) {
    const RefillRun r = refill_run(/*prompt=*/1100, /*seeded=*/1024);
    /* The cap is 1100 - 512 = 588: attention 576, and the checkpoint under it 512. */
    CHECK_EQ(r.n_after_first, 512);
    CHECK_EQ(r.rows, 1);
    CHECK_EQ(r.m.draft_refills, (int64_t)1);
    CHECK_EQ(r.m.draft_refill_tokens, (int64_t)512);
    CHECK_EQ(r.m.linear_cached_tokens, (int64_t)512);    /* what was actually skipped */
}

TEST(a_prefix_hit_already_a_window_back_is_kept_whole) {
    /* 5000 - 512 is past the 1024-token hit, so the drafter's window is recomputed anyway. */
    const RefillRun r = refill_run(5000, 1024);
    CHECK_EQ(r.n_after_first, 1024);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.rows, 1);
}

TEST(a_prefix_hit_with_no_checkpoint_near_the_cap_is_kept_whole) {
    /* No checkpoint below 2048: the cap would give back all 2048 tokens, more than two windows,
     * so the whole hit stands and the drafter waits for its window as it would without a cap. */
    const RefillRun r = refill_run(2200, 2048, /*ckpt_lo=*/2048);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.m.linear_cached_tokens, (int64_t)2048);
    CHECK_EQ(r.rows, 0);
}

/* A Gemma-like model: a full-attention trunk with a sliding window and no linear state, so a hit
 * resumes from the window snapshot taken at an earlier prompt's tail (ChunkGeometry::window_ckpt),
 * and a windowed block drafter. When that snapshot carries the trunk's window only, draft refill
 * cannot cut the hit (no snapshot a window back), the whole hit stands and the drafter waits for
 * its window; when it carries the drafter's window too (a block drafter on the card's driver,
 * KVManager::set_drafter_window_snapshot), there is no hole, no refill, and the drafter drafts from
 * the first decode. */
TEST(a_prefix_hit_whose_window_snapshot_carries_the_drafters_window_drafts_at_once) {
    for (int with_drafter = 0; with_drafter < 2; ++with_drafter) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192,
                /*windowed=*/true, /*linear=*/false);
        f.cfg.n_spec = 7;
        f.prog.drafter.kind = RAD_DRAFT_BLOCK;
        f.prog.drafter.window = 512;
        KVGroupInfo w;
        w.name = "draft";
        w.decl.kind = RAD_KV_WINDOW;
        w.decl.window = 512;
        w.block_size = 16;
        f.prog.kv_groups.push_back(w);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        const int32_t swa = 1, draft = 2;
        f.kv.snap = with_drafter ? std::vector<int32_t>{ swa, draft } : std::vector<int32_t>{ swa };
        f.pc.groups = { 0 };
        f.pc.ckpt_lo = 2048;          /* the earlier prompt's tail snapshot, none below it */
        CHECK_OK(f.start());
        CHECK(f.sch.geometry().window_ckpt);
        f.pc.seed(2048); f.pc.lin = 2048; f.pc.src = 3;
        const uint64_t a = f.submit(2200, 64);
        Scheduler::BlockRow rows[4];
        int n_rows = -1;
        for (int st = 0; st < 40; ++st) {
            CHECK(f.sch.step() != nullptr);
            if (st == 0) CHECK_EQ(f.sch.find(a)->n_computed, (int64_t)2048);
            f.commit_plain(11);
            const int n = f.sch.block_begin(8, 7, 512, 99, rows, 4);
            f.sch.block_end();
            if (!f.sch.find(a)->output.empty()) { n_rows = n; break; }
        }
        CHECK_EQ(f.sch.metrics().draft_refills, (int64_t)0);
        CHECK_EQ(n_rows, with_drafter ? 1 : 0);
    }
}

/* z-lab's DFlash on the Gemma 4 31B: a full-attention block drafter (no window), whose own group the
 * prefix cache indexes like the trunk's full layers. A hit resumes the trunk's window from the
 * window snapshot, so no paged group is left with a hole and the drafter drafts from the first
 * decode. Counting the trunk's window as a hole would put the drafter's floor at the hit, and a
 * drafter with no window to slide would not draft again that turn. */
TEST(a_full_attention_drafter_drafts_at_once_after_a_hit_the_window_snapshot_resumes) {
    for (int counted = 0; counted < 2; ++counted) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192,
                /*windowed=*/true, /*linear=*/false);
        f.cfg.n_spec = 7;
        f.prog.drafter.kind = RAD_DRAFT_BLOCK;
        f.prog.drafter.window = 0;
        KVGroupInfo d;
        d.name = "draft";
        d.decl.kind = RAD_KV_FULL;
        d.block_size = 16;
        f.prog.kv_groups.push_back(d);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        const int32_t swa = 1, draft = 2;
        f.kv.snap = { swa };
        f.kv.snap_reported = counted != 0;
        f.pc.groups = { 0, draft };
        f.pc.ckpt_lo = 2048;
        CHECK_OK(f.start());
        CHECK(f.sch.geometry().window_ckpt);
        f.pc.seed(2048); f.pc.lin = 2048; f.pc.src = 3;
        const uint64_t a = f.submit(2200, 64);
        Scheduler::BlockRow rows[4];
        int n_rows = -1;
        for (int st = 0; st < 40; ++st) {
            CHECK(f.sch.step() != nullptr);
            if (st == 0) CHECK_EQ(f.sch.find(a)->n_computed, (int64_t)2048);
            f.commit_plain(11);
            const int n = f.sch.block_begin(8, 7, 0, 99, rows, 4);
            f.sch.block_end();
            if (!f.sch.find(a)->output.empty()) { n_rows = n; break; }
        }
        CHECK_EQ(f.sch.metrics().draft_refills, (int64_t)0);
        CHECK_EQ(n_rows, counted ? 1 : 0);
    }
}

/* The short block's switch (--draft-short-from): on from the threshold, off again two below, and
 * off at one decoding sequence whatever the threshold, where the long block pays. From 2 (a tree's
 * auto) a load of 2, 1, 1 runs it, then not; from 8, 8 then 7 keeps it and 6 drops it. */
TEST(the_short_block_switches_off_at_one_sequence) {
    bool on = false;
    on = Scheduler::short_block_on(on, 2, 2); CHECK(on);
    on = Scheduler::short_block_on(on, 1, 2); CHECK(!on);
    on = Scheduler::short_block_on(on, 1, 2); CHECK(!on);
    on = Scheduler::short_block_on(on, 0, 2); CHECK(!on);
    on = Scheduler::short_block_on(false, 8, 8); CHECK(on);
    on = Scheduler::short_block_on(on, 7, 8); CHECK(on);
    on = Scheduler::short_block_on(on, 6, 8); CHECK(!on);
    on = Scheduler::short_block_on(on, 7, 8); CHECK(!on);
    CHECK(Scheduler::short_block_on(false, 1, 1));   /* from 1: every sequence runs it */
    CHECK(!Scheduler::short_block_on(true, 0, 1));
}

/* tp8 loop D19: short block tiers (RADIANCE_DRAFT_SHORT_TIERS=12:1,6:3 -- c32's lanes of 16 the 1-row block, c16's lanes
 * of 8 the 3-row one): the tier of the highest load a step reaches, each tier with the one-tier switch's hysteresis, and
 * the drafter's own block below them. The parser sorts the tiers from the highest load down and refuses malformed ones. */
TEST(short_block_tiers_pick_the_highest_load_tier_with_each_ones_hysteresis) {
    bool on[2] = {};
    const int64_t from[2] = { 12, 6 };
    struct { int64_t dec; int want; } seq[] = {
        { 16, 0 }, { 11, 0 }, { 10, 1 }, { 11, 1 }, { 12, 0 }, { 8, 1 }, { 5, 1 }, { 4, -1 }, { 5, -1 }, { 6, 1 },
        { 1, -1 }, { 2, -1 }, { 32, 0 }, { 0, -1 },
    };
    int bad = 0;
    for (const auto& s : seq) {
        const int k = Scheduler::short_tier_pick(on, from, 2, s.dec);
        if (k != s.want) { ++bad; std::fprintf(stderr, "    decoded %lld: tier %d, want %d\n", (long long)s.dec, k, s.want); }
    }
    CHECK_EQ(bad, 0);
    /* one tier = the one-tier switch */
    bool o1 = false, o2 = false;
    for (const int64_t d : { 2, 8, 7, 6, 7, 9, 1 }) {
        o1 = Scheduler::short_block_on(o1, d, 8);
        CHECK_EQ(Scheduler::short_tier_pick(&o2, from, 0, d), -1);
        const int64_t f8 = 8;
        CHECK_EQ(Scheduler::short_tier_pick(&o2, &f8, 1, d), o1 ? 0 : -1);
    }
    std::vector<std::pair<int64_t, int64_t>> t;
    std::string why;
    CHECK_OK(Scheduler::parse_short_tiers("6:3,12:1", &t, &why));
    REQUIRE_EQ(t.size(), (size_t)2);
    CHECK_EQ(t[0].first, (int64_t)12); CHECK_EQ(t[0].second, (int64_t)1);
    CHECK_EQ(t[1].first, (int64_t)6);  CHECK_EQ(t[1].second, (int64_t)3);
    CHECK_OK(Scheduler::parse_short_tiers("12:1", &t, &why));
    CHECK_EQ(t.size(), (size_t)1);
    for (const char* b : { "", "12", "12:", ":1", "12:1,", "12:1,12:3", "1:1", "12:0", "12:1;6:3", "a:1", "2:1,3:1,4:1,5:1,6:1" })
        CHECK_EQ(Scheduler::parse_short_tiers(b, &t, &why), RAD_E_INVAL);
}

/* tp8 loop D23 (fn4 draft_short_blk): the switch runs a declared short block in place of the TOP choice only -- the
 * one-tier --draft-short-from choice, or tier 0 of RADIANCE_DRAFT_SHORT_TIERS -- and never where the start runs the
 * drafter's own block or a lower tier; 0 is the start's choice. With DSpark declared at 2 and 3 rows (SHORT=2,3) and
 * --draft-short-from 12, lanes of 16 run 2 (stk) or 3 (depth 3) by the switch, lanes of 8 their own block either way. */
TEST(draft_short_blk_moves_only_the_top_short_block_choice) {
    /* the pure rule */
    CHECK_EQ(Scheduler::short_block_forced(2, true, 0), (int64_t)2);
    CHECK_EQ(Scheduler::short_block_forced(2, true, 3), (int64_t)3);
    CHECK_EQ(Scheduler::short_block_forced(0, false, 3), (int64_t)0);   /* own block: stays */
    CHECK_EQ(Scheduler::short_block_forced(3, false, 1), (int64_t)3);   /* a lower tier: stays */
    CHECK_EQ(Scheduler::short_block_forced(0, true, 3), (int64_t)0);    /* no short block chosen: stays */
    /* with the one-tier switch from 12 decoding sequences: the load sequence a c32 / c16 / c24 run walks */
    bool on = false;
    int bad = 0;
    struct { int64_t dec, forced, want; } seq[] = {
        { 16, 0, 2 }, { 16, 3, 3 }, { 8, 3, 0 }, { 11, 3, 0 }, { 12, 3, 3 }, { 11, 3, 3 }, { 10, 3, 0 }, { 16, 0, 2 },
        { 16, 2, 2 }, { 12, 3, 3 }, { 2, 3, 0 },
    };
    for (const auto& q : seq) {
        on = Scheduler::short_block_on(on, q.dec, 12);
        const int64_t got = Scheduler::short_block_forced(on ? 2 : 0, on, q.forced);
        if (got != q.want) {
            ++bad;
            std::fprintf(stderr, "    decoded %lld, forced %lld: block %lld, want %lld\n", (long long)q.dec,
                         (long long)q.forced, (long long)got, (long long)q.want);
        }
    }
    CHECK_EQ(bad, 0);
    /* with tiers 12:2,6:3: tier 0 (lanes of >= 12) takes the switch, tier 1 (lanes of 6 .. 11) keeps its 3 */
    bool ton[2] = {};
    const int64_t from[2] = { 12, 6 };
    const int64_t blk[2] = { 2, 3 };
    struct { int64_t dec, forced, want; } tseq[] = { { 16, 1, 1 }, { 8, 1, 3 }, { 4, 1, 0 }, { 16, 0, 2 }, { 6, 2, 3 } };
    for (const auto& q : tseq) {
        const int k = Scheduler::short_tier_pick(ton, from, 2, q.dec);
        const int64_t got = Scheduler::short_block_forced(k >= 0 ? blk[k] : 0, k == 0, q.forced);
        if (got != q.want) {
            ++bad;
            std::fprintf(stderr, "    tiers: decoded %lld, forced %lld: block %lld, want %lld\n", (long long)q.dec,
                         (long long)q.forced, (long long)got, (long long)q.want);
        }
    }
    CHECK_EQ(bad, 0);
}

/* tp8 loop D23: fn4 refuses draft_short_blk N > 0 before an engine registered, where no short block is served, and for a
 * length the drafter did not declare (naming the declared ones); 0 always; a request reaches the engine at the step
 * release (a scheduler-side switch: apply_pending hands it over), and GET reports it. */
TEST(draft_short_blk_takes_only_a_declared_short_block) {
    std::string err;
    const int i = fn4::switch_index("draft_short_blk");
    REQUIRE(i == (int)fn4::kDraftShortBlk);
    CHECK(fn4::sched_switch(i));
    CHECK(!fn4::follower_switch(i));   /* the leader's alone: the follower runs no drafter */
    CHECK(fn4::check("draft_short_blk", 0, &err));
    CHECK(!fn4::check("draft_short_blk", 64, &err));
    CHECK(!fn4::check("draft_short_blk", -1, &err));
    fn4::short_register(0);
    CHECK(!fn4::check("draft_short_blk", 3, &err));
    CHECK(err.find("serves no short block") != std::string::npos);
    fn4::short_register((1ull << 2) | (1ull << 3));
    CHECK(fn4::check("draft_short_blk", 2, &err));
    CHECK(fn4::check("draft_short_blk", 3, &err));
    err.clear();
    CHECK(!fn4::check("draft_short_blk", 4, &err));
    CHECK(err.find("declared: 2, 3") != std::string::npos);
    if (err.find("declared: 2, 3") == std::string::npos) std::fprintf(stderr, "    %s\n", err.c_str());
    CHECK_EQ(fn4::value(i), 0);   /* unset: the start's */
    int sw[fn4::kSwitches];
    REQUIRE(fn4::request("draft_short_blk", 3, &err));
    CHECK(fn4::apply_pending(0, sw));
    CHECK_EQ(sw[i], 3);
    fn4::note_sched(i, 3);
    CHECK_EQ(fn4::value(i), 3);
    fn4::note_short_blk(48);
    const std::string j = fn4::state_json();
    CHECK(j.find("\"draft_short_blk\": 3") != std::string::npos);
    CHECK(j.find("\"short_blk\": {\"declared_mask\": 12, \"in_force\": 3, \"passes\": 1, \"rows\": 48}") != std::string::npos);
    REQUIRE(fn4::request("draft_short_blk", 0, &err));
    CHECK(fn4::apply_pending(1, sw));
    CHECK_EQ(sw[i], 0);
    fn4::note_sched(i, 0);
    CHECK_EQ(fn4::value(i), 0);
}

/* tp8 loop D26 (DECODE round 22): ds41_attn_dec is switch 84, right after ds41_prefetch, a plugin's (libr4d) that the follower
 * takes with the leader's and the scheduler does not own; with no loaded object serving it a request is refused at the step
 * release and the value stays (the published build: nothing reads it). */
TEST(ds41_attn_dec_is_a_follower_plugin_switch_after_ds41_prefetch) {
    const int i = fn4::switch_index("ds41_attn_dec");
    REQUIRE(i == (int)fn4::kDs41AttnDec);
    CHECK_EQ(i, (int)fn4::kDs41Prefetch + 1);
    CHECK_EQ(i, 84);
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    CHECK(std::string(fn4::switch_name(i)) == "ds41_attn_dec");
}

/* tp8 loop DECODE round 23 (lever TSK merged from ds41-drafter): ds41_tsk was switch 85 here, right after ds41_attn_dec; since the
 * round-24 merge of PREFILL's ds41-pp-dev-pf (ds41_sphc .. ds41_twin = 85 .. 90) it is 91, right after ds41_twin, a plugin's
 * (libr4d) that the follower takes with the leader's and the scheduler does not own (83 on ds41-drafter: names, not numbers, cross
 * the branches). */
TEST(ds41_tsk_is_a_follower_plugin_switch_after_ds41_twin) {
    const int i = fn4::switch_index("ds41_tsk");
    REQUIRE(i == (int)fn4::kDs41Tsk);
    CHECK_EQ(i, (int)fn4::kDs41Twin + 1);
    CHECK_EQ(i, 91);
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    CHECK(std::string(fn4::switch_name(i)) == "ds41_tsk");
}

/* tp8 loop DECODE round 24 (S2: ds41-drafter merged onto ds41-pp-dev + ds41-pp-dev-pf): the DRAFTER's lever DHD switch
 * draft_head_cut (82 on ds41-drafter) is 92 here, right after ds41_tsk -- the leader's engine switch (Engine::issue_block_drafter),
 * not a plugin's, not the scheduler's, not one the follower requests; 0 / 1 taken, anything else refused with its range. */
TEST(draft_head_cut_is_the_leaders_engine_switch_after_ds41_tsk) {
    const int i = fn4::switch_index("draft_head_cut");
    REQUIRE(i == (int)fn4::kDraftHeadCut);
    CHECK_EQ(i, (int)fn4::kDs41Tsk + 1);
    CHECK_EQ(i, 92);
    CHECK_EQ(i + 1, (int)fn4::kPpFolReserve);   /* IDEAS #9's pp_fol_reserve (93) after it */
    CHECK(!fn4::sched_switch(i));
    CHECK(!fn4::follower_switch(i));
    CHECK(std::string(fn4::switch_name(i)) == "draft_head_cut");
    std::string err;
    CHECK(fn4::check("draft_head_cut", 0, &err));
    CHECK(fn4::check("draft_head_cut", 1, &err));
    CHECK(!fn4::check("draft_head_cut", 2, &err));
    CHECK(err.find("draft_head_cut takes 0 (off) or 1") != std::string::npos);
    CHECK(!fn4::check("draft_head_cut", -1, &err));
    /* the PREFILL switches kept their numbers through both merges (names cross the branches; these are on cards) */
    CHECK_EQ(fn4::switch_index("ds41_sphc"), 85);
    CHECK_EQ(fn4::switch_index("ds41_twin"), 90);
    CHECK_EQ(fn4::switch_index("ds41_attn_dec"), 84);
}

/* tp8 loop IDEAS #9 (DECODE round 24): pp_fol_reserve is 93, the last, the follower's engine switch (requested from the leader's
 * table, not a plugin's, not the scheduler's); 0 .. 63 on a node that registered nothing (the leader); a registered node (the
 * follower) takes 0 and its declared boundaries only, and says which it has. */
TEST(pp_fol_reserve_is_the_followers_switch_checked_against_its_boundaries) {
    const int i = fn4::switch_index("pp_fol_reserve");
    REQUIRE(i == (int)fn4::kPpFolReserve);
    CHECK_EQ(i, 93);
    CHECK_EQ(i + 1, (int)fn4::kPpLeadSeg);   /* tp8 loop D32: pp_lead_seg after it */
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    std::string err;
    CHECK(fn4::check("pp_fol_reserve", 0, &err));
    CHECK(fn4::check("pp_fol_reserve", 25, &err));    /* unregistered: the range only */
    CHECK(!fn4::check("pp_fol_reserve", 64, &err));
    CHECK(err.find("pp_fol_reserve takes 0 (off) or 1 .. 63") != std::string::npos);
    fn4::reserve_register((1ull << 22) | (1ull << 25));
    CHECK(fn4::check("pp_fol_reserve", 25, &err));
    CHECK(fn4::check("pp_fol_reserve", 0, &err));
    CHECK(!fn4::check("pp_fol_reserve", 24, &err));
    CHECK(err.find("is not a declared segment boundary inside W(h, t) (declared: 22, 25)") != std::string::npos);
    fn4::reserve_register(0);
    CHECK(!fn4::check("pp_fol_reserve", 25, &err));
    CHECK(err.find("holds no BG lane's segments") != std::string::npos);
    CHECK(fn4::check("pp_fol_reserve", 0, &err));
}

/* tp8 loop D32 (DECODE round 25): pp_lead_seg is 94, the last, the LEADER's engine switch (not the follower's, not a plugin's, not
 * the scheduler's); form + 16 x stride with form 0 / 1 / 2 (+ 256: no band passes, tp8 loop D36); any value on a node that registered nothing; a leader that took no
 * boundary (registered 0) takes 0 only and names RADIANCE_PP_LEAD_SEG; one that took them takes every form and stride. */
TEST(pp_lead_seg_is_the_leaders_switch_form_and_stride) {
    const int i = fn4::switch_index("pp_lead_seg");
    REQUIRE(i == (int)fn4::kPpLeadSeg);
    CHECK_EQ(i, 94);
    CHECK_EQ(i + 1, (int)fn4::kPpSegSet);   /* tp8 loop D33: pp_seg_set after it */
    /* tp8 MERGE (ds41-pp-merge merged on the dev line, round 34): pp_merge 98 / pp_merge_ms 99 after pp_tband 97, the leader's lanes'
     * (sched_switch, as pp_bg_room) */
    CHECK_EQ(fn4::switch_index("pp_merge"), 98);
    CHECK_EQ(fn4::switch_index("pp_merge_ms"), 99);
    CHECK_EQ(fn4::switch_index("pp_rband"), 100);   /* tp8 loop P41: the reverse (head) band's rule, the leader's (not a sched switch) */
    CHECK(!fn4::sched_switch(fn4::kPpRband));
    CHECK(!fn4::follower_switch(fn4::kPpRband));
    {
        std::string e2;
        CHECK(fn4::check("pp_rband", -1, &e2));
        CHECK(fn4::check("pp_rband", 0, &e2));
        CHECK(fn4::check("pp_rband", 2, &e2));
        CHECK(fn4::check("pp_rband", 18, &e2));
        CHECK(fn4::check("pp_rband", 2 + 256, &e2));
        CHECK(!fn4::check("pp_rband", 16, &e2));    /* p = 0 with a phase */
        CHECK(!fn4::check("pp_rband", 512, &e2));
        CHECK(!fn4::check("pp_rband", -2, &e2));
        CHECK(e2.find("RADIANCE_PP_RBAND_EVERY") != std::string::npos);
    }
    CHECK_EQ(fn4::switch_index("sched_srf"), 101);   /* tp8 loop P44 */
    CHECK_EQ(fn4::switch_index("aql_hdp"), 102);     /* tp8 loop U1 */
    CHECK_EQ(i + 9, (int)fn4::kSwitches);
    CHECK(!fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    std::string err;
    CHECK(fn4::check("pp_lead_seg", 0, &err));
    CHECK(fn4::check("pp_lead_seg", 2, &err));        /* unregistered: the range only */
    CHECK(fn4::check("pp_lead_seg", 34, &err));       /* decode first, every 2nd boundary */
    CHECK(!fn4::check("pp_lead_seg", 3, &err));
    CHECK(err.find("pp_lead_seg takes form + 16 x stride") != std::string::npos);
    CHECK(fn4::check("pp_lead_seg", 256, &err));      /* tp8 loop D36: + 256 = no band passes */
    CHECK(fn4::check("pp_lead_seg", 258, &err));
    CHECK(!fn4::check("pp_lead_seg", 259, &err));
    CHECK(err.find("+ 256 = no band passes") != std::string::npos);
    /* tp8 loop D40: + 8 = the tail whole, + 512 = the SP word on the head's segments; 3 / 11 / 1024 refused */
    CHECK(fn4::check("pp_lead_seg", 10, &err));       /* decode first, tail whole */
    CHECK(fn4::check("pp_lead_seg", 74, &err));       /* decode first, every 4th, tail whole */
    CHECK(fn4::check("pp_lead_seg", 512, &err));
    CHECK(fn4::check("pp_lead_seg", 546, &err));      /* 34 + 512 */
    CHECK(fn4::check("pp_lead_seg", 1023 - 5, &err)); /* 1018 = 2 + 8 + 16 x 15 + 256 + 512 */
    CHECK(!fn4::check("pp_lead_seg", 11, &err));
    CHECK(err.find("+ 8 = the tail whole, + 512 = the SP word") != std::string::npos);
    /* tp8 loop D49b: + 1024 x N (N 1 .. 63) = segments only while the decode lanes' last steps decoded <= N sequences; the
     * form bits keep their rule (1027 = form 3 + 1024 refused) */
    CHECK(fn4::check("pp_lead_seg", 1024, &err));     /* form 0 + N 1: whole anyway */
    CHECK(fn4::check("pp_lead_seg", 1026, &err));     /* decode first, N 1 */
    CHECK(fn4::check("pp_lead_seg", 2 + 1024 * 12, &err));
    CHECK(fn4::check("pp_lead_seg", 65535 - 5, &err));   /* 65530 = 1018 + 1024 x 63 */
    CHECK(!fn4::check("pp_lead_seg", 1027, &err));
    CHECK(err.find("+ 1024 x N (N 1 .. 63) = segments only while the decode lanes' last steps decoded <= N") != std::string::npos);
    /* tp8 loop F81: + 65536 = no position-order guard (the negative control); 131072 and a bad form with it refused */
    CHECK(fn4::check("pp_lead_seg", 65536, &err));
    CHECK(fn4::check("pp_lead_seg", 34 + 65536, &err));
    CHECK(fn4::check("pp_lead_seg", 65530 + 65536, &err));
    CHECK(!fn4::check("pp_lead_seg", 131072, &err));
    CHECK(!fn4::check("pp_lead_seg", 3 + 65536, &err));
    CHECK(err.find("+ 65536 = no position-order guard") != std::string::npos);
    CHECK(!fn4::check("pp_lead_seg", -1, &err));
    fn4::lead_seg_register(false);
    CHECK(fn4::check("pp_lead_seg", 0, &err));
    CHECK(!fn4::check("pp_lead_seg", 1, &err));
    CHECK(err.find("declared no leader boundary (RADIANCE_PP_LEAD_SEG=k") != std::string::npos);
    fn4::lead_seg_register(true);
    CHECK(fn4::check("pp_lead_seg", 1, &err));
    CHECK(fn4::check("pp_lead_seg", 82, &err));       /* decode first, every 5th */
    CHECK(fn4::check("pp_lead_seg", 0, &err));
    fn4::lead_seg_register(false);
}

/* tp8 loop D33 (DECODE round 26): pp_seg_set is 95 (the last until PX1's exl3_x3d_pf, 96), the FOLLOWER's engine switch (requested from the leader's table, not a
 * plugin's, not the scheduler's); n + 64 x s with n 0 .. 31, s 0 .. 7; a node that registered nothing (the leader) checks the range
 * only; a follower with no declared boundary takes 0 only and names RADIANCE_PP_SEG_LAYERS; one with boundaries takes every value. */
/* tp8 loop PX1 (PREFILL round 28): exl3_x3d_pf is 96, the last, the exl3r4d plugin's (A10 / PX2: the prefill band's one-word-pair /
 * unpacked-state decode) -- not the scheduler's; a pipeline's follower takes it from its leader as every plugin switch; with no
 * exl3r4d loaded nothing serves it, so a request is refused with a reason (a start with exl3r4d 4fd50e4 takes 0 / 1, its own range). */
TEST(exl3_x3d_pf_is_the_exl3r4d_plugins_switch_after_pp_seg_set) {
    const int i = fn4::switch_index("exl3_x3d_pf");
    REQUIRE(i == (int)fn4::kExl3X3dPf);
    CHECK_EQ(i, 96);
    CHECK_EQ(i, (int)fn4::kPpSegSet + 1);
    CHECK_EQ(i + 7, (int)fn4::kSwitches);   /* tp8 loop D39: pp_tband (97) after it; tp8 MERGE: pp_merge 98 / pp_merge_ms 99; P41: pp_rband 100; P44: sched_srf 101; U1: aql_hdp 102 */
    CHECK(std::string(fn4::switch_name(i)) == "exl3_x3d_pf");
    CHECK(!fn4::sched_switch(i));
    CHECK(fn4::follower_switch(i));
    std::string err;
    CHECK(!fn4::check("exl3_x3d_pf", 1, &err));
    CHECK(!err.empty());
}

/* tp8 loop D39 step 2c (DECODE round 31): pp_tband is 97, the last, the LEADER's engine switch (not a plugin's, not the scheduler's,
 * not the follower's: a tail band pass reaches the follower as its frame's flag); 0 / 1 / 2 (D39b), 3 (D46), 4 / 4 + 1024 x N (D50), other values refused; an unregistered node
 * (one machine, the follower) checks the range only; a leader that took no tail band (RADIANCE_PP_TAIL_BAND unset / not taken)
 * takes 0 only and names RADIANCE_PP_TAIL_BAND; one that took it takes both. */
TEST(pp_tband_is_the_leaders_switch_after_exl3_x3d_pf) {
    const int i = fn4::switch_index("pp_tband");
    REQUIRE(i == (int)fn4::kPpTband);
    CHECK_EQ(i, 97);
    CHECK_EQ(i, (int)fn4::kExl3X3dPf + 1);
    CHECK_EQ(i + 6, (int)fn4::kSwitches);   /* tp8 MERGE: pp_merge 98 / pp_merge_ms 99 after it; tp8 loop P41: pp_rband 100; P44: sched_srf 101; U1: aql_hdp 102 */
    CHECK(std::string(fn4::switch_name(i)) == "pp_tband");
    CHECK(!fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    std::string err;
    CHECK(fn4::check("pp_tband", 0, &err));
    CHECK(fn4::check("pp_tband", 1, &err));   /* unregistered: the range only */
    CHECK(fn4::check("pp_tband", 2, &err));   /* tp8 loop D39b: + the later chunks of an all-band prompt */
    CHECK(fn4::check("pp_tband", 3, &err));   /* tp8 loop D46: + decode lanes' steps beside a BG step (unregistered: the range) */
    CHECK(fn4::check("pp_tband", 4, &err));   /* tp8 loop D50: 3 in every stretch */
    CHECK(fn4::check("pp_tband", 4 + 1024, &err));        /* ... and 4 + 1024 x N, N 1 .. 63 */
    CHECK(fn4::check("pp_tband", 4 + 1024 * 63, &err));
    CHECK(fn4::check("pp_tband", 5, &err));   /* tp8 loop D51: the band follows the measured bound */
    for (const int bad : { 6, 1023, 1024, 1027, 1029, 5 + 1024, 3 + 1024 * 10, 4 + 1024 * 64, 2048 })
        CHECK(!fn4::check("pp_tband", bad, &err));
    CHECK(!fn4::check("pp_tband", 6, &err));
    CHECK(err.find("pp_tband takes -1 (back to the start's RADIANCE_PP_TAIL_BAND), 0 (no tail band passes") != std::string::npos);
    CHECK(err.find("2 (1 + the later chunks") != std::string::npos);
    CHECK(err.find("3 (1 + a decode lane's step while a BG step is in flight") != std::string::npos);
    CHECK(err.find("4 (3 in every stretch; D50), 4 + 1024 x N, N 1 .. 63") != std::string::npos);
    CHECK(err.find("or 5 (a decode step bands while the follower's card works more than the leader's") != std::string::npos);
    CHECK(fn4::check("pp_tband", -1, &err));    /* the never-set value, which the kit's arm undo POSTs back (loop lq215) */
    CHECK(!fn4::check("pp_tband", -2, &err));
    fn4::tband_register(false);
    CHECK(fn4::check("pp_tband", -1, &err));    /* also where no tail band was taken: the start's value is 'none' */
    CHECK(fn4::check("pp_tband", 0, &err));
    CHECK(!fn4::check("pp_tband", 1, &err));
    CHECK(err.find("RADIANCE_PP_TAIL_BAND=1 on the leader at start") != std::string::npos);
    CHECK(!fn4::check("pp_tband", 2, &err));
    CHECK(!fn4::check("pp_tband", 3, &err));
    CHECK(!fn4::check("pp_tband", 4, &err));
    CHECK(!fn4::check("pp_tband", 4 + 1024 * 10, &err));
    CHECK(!fn4::check("pp_tband", 5, &err));
    fn4::tband_register(true);
    CHECK(fn4::check("pp_tband", 1, &err));
    CHECK(fn4::check("pp_tband", 2, &err));
    CHECK(fn4::check("pp_tband", 0, &err));
    CHECK(!fn4::check("pp_tband", 3, &err));   /* tp8 loop D46: a tail band without the plugin's copy */
    CHECK(err.find("the plugin declared no tail band copy") != std::string::npos);
    CHECK(!fn4::check("pp_tband", 4, &err));   /* tp8 loop D50: so are 4 / 4 + 1024 x N */
    CHECK(err.find("pp_tband=4 refused: the plugin declared no tail band copy") != std::string::npos);
    CHECK(!fn4::check("pp_tband", 4 + 1024 * 10, &err));
    CHECK(!fn4::check("pp_tband", 5, &err));   /* tp8 loop D51: 5 too */
    CHECK(err.find("pp_tband=5 refused: the plugin declared no tail band copy") != std::string::npos);
    fn4::tband_register(2);                    /* ... with it */
    CHECK(fn4::check("pp_tband", 3, &err));
    CHECK(fn4::check("pp_tband", 1, &err));
    CHECK(fn4::check("pp_tband", -1, &err));
    CHECK(fn4::check("pp_tband", 4, &err));
    CHECK(fn4::check("pp_tband", 4 + 1024 * 10, &err));
    CHECK(fn4::check("pp_tband", 5, &err));    /* tp8 loop D51 */
    CHECK(!fn4::check("pp_tband", 6, &err));
    CHECK(!fn4::check("pp_tband", 3 + 1024 * 10, &err));
    fn4::tband_register(false);
}

/* tp8 loop D46: which sequences' tail band state the leader holds (core/sched/tcopy.h). Form 1 (every pass that is no tail band
 * pass carries the copy): a sequence seen fresh is held from its first row; a snapshot restore / cone replay makes it a gap for
 * good. Form R (a window of R positions; only decode-only passes carry the copy): a fresh prompt in a MIXED pass that is no band
 * pass is a gap at its end, decode-only passes write, and the sequence is held again R positions later; a fresh prompt's band
 * pass writes it; a sequence never seen is not held (a fresh prompt row of the step itself is); form 0 holds nothing. */
TEST(tcopy_holds_a_sequence_once_its_band_state_was_written_on_the_leader) {
    using namespace rad;
    auto pre = [](uint64_t s, int32_t ctx, int32_t n) { StepEntry e; e.seq = s; e.ctx_len = ctx; e.n_tokens = n; e.is_prefill = true; return e; };
    auto dec = [](uint64_t s, int32_t ctx, int32_t n) { StepEntry e; e.seq = s; e.ctx_len = ctx; e.n_tokens = n; return e; };
    tcopy::Gap g;
    /* form 1 */
    tcopy::note(g, 1, false, { pre(1, 0, 700), dec(9, 50, 3) });   /* 9 never seen fresh: first seen with context -> a gap at 50 */
    CHECK_EQ(g[1], (int64_t)-1);
    CHECK_EQ(g[9], (int64_t)50);
    CHECK(tcopy::held(g, 1, { dec(1, 700, 3) }));
    CHECK(!tcopy::held(g, 1, { dec(9, 53, 3) }));
    CHECK(!tcopy::held(g, 1, { dec(1, 700, 3), dec(9, 53, 3) }));
    CHECK(tcopy::held(g, 1, { dec(1, 703, 3), pre(2, 0, 400) }));   /* a fresh prompt row of the step itself */
    CHECK(!tcopy::held(g, 1, { dec(3, 10, 1) }));                   /* never seen */
    StepEntry r = pre(4, 512, 188);
    r.ckpt_restore = 7;
    CHECK(!tcopy::held(g, 1, { r }));
    tcopy::note(g, 1, false, { r });
    CHECK_EQ(g[4], (int64_t)512);
    CHECK(!tcopy::held(g, 1, { dec(4, 900, 1) }));
    CHECK(!tcopy::held(g, 0, { dec(1, 700, 3) }));
    /* form R = 133 */
    tcopy::Gap w;
    CHECK(!tcopy::written(133, false, { dec(5, 0, 1), pre(6, 0, 600) }));
    CHECK(tcopy::written(133, false, { dec(5, 10, 3), dec(6, 20, 2) }));
    CHECK(tcopy::written(133, true, { pre(6, 0, 600) }));
    tcopy::note(w, 133, false, { pre(6, 0, 600), dec(5, 0, 1) });   /* a MIXED pass, no band: both a gap at their ends */
    CHECK_EQ(w[6], (int64_t)600);
    CHECK_EQ(w[5], (int64_t)1);
    for (int32_t c = 600; c < 733; c += 3) {
        CHECK(!tcopy::held(w, 133, { dec(6, c, 3) }));
        tcopy::note(w, 133, false, { dec(6, c, 3) });   /* decode-only: written, the gap stays */
    }
    CHECK_EQ(w[6], (int64_t)600);
    CHECK(tcopy::held(w, 133, { dec(6, 733, 3) }));
    tcopy::note(w, 133, false, { dec(6, 733, 3), pre(7, 0, 300) });   /* a prompt beside it, no band: a new gap */
    CHECK_EQ(w[6], (int64_t)736);
    CHECK(!tcopy::held(w, 133, { dec(6, 736, 3) }));
    tcopy::note(w, 133, true, { pre(8, 0, 900) });   /* a fresh prompt's band pass */
    CHECK_EQ(w[8], (int64_t)-1);
    CHECK(tcopy::held(w, 133, { dec(8, 900, 3) }));
    tcopy::note(w, 133, true, { pre(8, 900, 900) });   /* its later chunk as a band pass: still all written */
    CHECK_EQ(w[8], (int64_t)-1);
    CHECK(!tcopy::held(w, 0, { dec(8, 1800, 1) }));
}

TEST(pp_seg_set_is_the_followers_switch_n_and_stride) {
    const int i = fn4::switch_index("pp_seg_set");
    REQUIRE(i == (int)fn4::kPpSegSet);
    CHECK_EQ(i, 95);
    CHECK_EQ(i + 8, (int)fn4::kSwitches);   /* tp8 loop PX1: exl3_x3d_pf (96) after it; tp8 loop D39: pp_tband (97); tp8 MERGE: 98 / 99; P41: 100; P44: 101; U1: 102 */
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    CHECK(std::string(fn4::switch_name(i)) == "pp_seg_set");
    std::string err;
    CHECK(fn4::check("pp_seg_set", 0, &err));
    CHECK(fn4::check("pp_seg_set", 15, &err));        /* unregistered: the range only */
    CHECK(fn4::check("pp_seg_set", 128, &err));       /* every 2nd boundary */
    CHECK(fn4::check("pp_seg_set", 7 * 64 + 31, &err));
    CHECK(!fn4::check("pp_seg_set", 32, &err));
    CHECK(err.find("pp_seg_set takes n + 64 x s") != std::string::npos);
    CHECK(!fn4::check("pp_seg_set", 512, &err));
    CHECK(!fn4::check("pp_seg_set", -1, &err));
    fn4::segset_register(0);
    CHECK(fn4::check("pp_seg_set", 0, &err));
    CHECK(!fn4::check("pp_seg_set", 15, &err));
    CHECK(err.find("declared no segment boundary (RADIANCE_PP_SEG_LAYERS") != std::string::npos);
    fn4::segset_register(21);
    CHECK(fn4::check("pp_seg_set", 15, &err));
    CHECK(fn4::check("pp_seg_set", 128, &err));
    CHECK(fn4::check("pp_seg_set", 0, &err));
    fn4::segset_register(0);
}

TEST(no_prefix_hit_gives_nothing_back) {
    const RefillRun r = refill_run(1100, 0);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.rows, 1);
}

/* ================================================================== speculation */
TEST(rejection_rolls_back_paged_kv_and_moves_the_linear_read_offset) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_spec, 4);
    CHECK_EQ(b->n_tok, 5);
    CHECK_EQ(b->q_lens[0], 5);
    CHECK_EQ(b->token_ids[0], 11);
    CHECK_EQ(b->token_ids[1], 21);
    CHECK_EQ(b->token_ids[4], 24);

    int64_t blocks_before = f.kv.free_blocks(0);
    int32_t cursor_before = f.sch.plan().e[0].conv_cursor;

    /* Two of the four drafts survive. */
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_OK(f.commit(99, 0, [](const StepEntry&) { return 2; }));

    const Request* ra = f.sch.find(a);
    /* Three tokens committed: the two accepted drafts and the bonus. */
    CHECK_EQ((int)ra->output.size(), 4);
    CHECK_EQ(ra->output[1], 21);
    CHECK_EQ(ra->output[2], 22);
    CHECK_EQ(ra->output[3], 99);
    CHECK_EQ(ra->n_computed, 163);
    CHECK_EQ(ra->n_accepted, 2);
    /* Paged KV gave back blocks for the rejected tail; the state slot did not move, since a linear
     * rollback is a read offset, not a free. */
    CHECK(f.kv.free_blocks(0) >= blocks_before);
    CHECK(f.kv.state_slot(a, 1) >= 0);

    /* The conv cursor advanced by 1 + accepted, modulo the rolling window. */
    f.submit(1, 1);   /* only to force a plan rebuild below */
    const RadBatch* b2 = f.sch.step();
    CHECK(b2 != nullptr);
    int32_t W = 4 - 1 + 4 + 1;
    CHECK_EQ(f.sch.plan().e[0].conv_cursor, (cursor_before + 3) % W);
}

/* THE COUPLED BIND AFTER A REAL COMMIT.
 *
 * Engine::finish_serial_drafter binds each round at the request's committed output length, read
 * from the plan after commit -- the value the next Sampler::build_step stages that row at. This
 * drives the same rule (sched/mtp_couple_bind.h) against the real Scheduler::commit: a live
 * commit, a max_tokens cut, and a card reused by a new request afterwards.
 *
 * The old engine formula (the draw's pre-commit base + the slot's one-based n_acc) agrees on every
 * commit of a request that is still running, and drifts by exactly the tokens a cut withheld. A
 * commit that emits fewer than the card accepted has closed the request; its plan entry is carried
 * for the bind, but the request will never sample again. Binding it would put a coupled verify on
 * a terminal row, which reads as a live round in the log without being one. */
TEST(the_coupled_bind_skips_a_request_the_commit_closed) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    CHECK_OK(f.start());
    const int64_t vocab = f.prog.meta.n_vocab;

    int32_t draft[3] = { 21, 22, 23 };
    auto readback = [&](uint64_t seq, uint64_t base, int rounds) {
        RadMtpCoupleDraw d{};
        d.seq  = seq;
        d.base = base;
        d.mode = 1;
        std::vector<RadMtpCoupleState> rb((size_t)rounds);
        for (int rd = 1; rd <= rounds; ++rd) {
            RadMtpCoupleState& s = rb[(size_t)(rd - 1)];
            s.draw   = d;
            s.round  = rd;
            s.status = 1;
            s.count  = 1;
            s.x      = draft[rd - 1];       /* what the card holds for that round */
            for (int i = 0; i < 8; ++i) s.ids[i] = -1;
            s.ids[0] = s.x;
            s.q[0]   = 1.0;
        }
        return rb;
    };
    auto slot = [&]() {
        StepSlot S{};
        S.acc = 3;
        S.n_acc = 4;
        for (int rd = 1; rd <= 3; ++rd) S.tok[rd] = draft[rd - 1];
        return S;
    };

    /* ---- a live commit: three drafts accepted, the request keeps running. */
    const uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t a_base = f.sch.find(a)->output.size();   /* the draw's pre-commit coordinate */
    CHECK_OK(f.commit(9, 0, [](const StepEntry&) { return 3; }));
    REQUIRE(f.sch.find(a) != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(a)->output.size(), (int)a_base + 4);

    MtpCoupleBindPoint pa;
    pa.found = f.sch.committed_seq(a, &pa.n_out, &pa.live);
    CHECK(pa.found);
    CHECK(pa.live);
    CHECK_EQ(pa.n_out, (uint64_t)f.sch.find(a)->output.size());

    const std::vector<RadMtpCoupleState> rba = readback(a, a_base, 3);
    const StepSlot Sa = slot();
    RadMtpCoupleState bound[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    int n_bound = -1;
    std::string why;
    CHECK_EQ(mtp_couple_bind_rounds(pa, rba[0].draw, rba.data(), 1, 3, Sa, vocab, bound,
                                    &n_bound, &why), RAD_OK);
    CHECK_EQ(n_bound, 3);
    for (int k = 0; k < 3; ++k) {
        /* The next build stages row k at n_out + k. */
        CHECK_EQ(bound[k].pos, pa.n_out + (uint64_t)k);
        CHECK(rad_mtp_couple_binds(bound[k], a, rba[0].draw.seed, pa.n_out, k, draft[k], vocab));
    }
    /* The old formula on the same data: identical, because this commit emitted its whole
     * acceptance. */
    for (int k = 0; k < 3; ++k)
        CHECK_EQ(a_base + 4 + (uint64_t)k, pa.n_out + (uint64_t)k);

    /* ---- a cut commit: the card accepted three drafts, the budget lets two through. */
    const uint64_t b = f.submit(160, /*max_tokens=*/3);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);                                     /* b's first output token */
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    CHECK_OK(f.sch.set_draft(b, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t b_base = f.sch.find(b)->output.size();
    CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? 3 : 0; }));
    REQUIRE(f.sch.find(b) != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Finished);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 3);          /* 1 + 2, not 1 + 4 */

    /* The commit snapshot still names b -- the terminal step is carried into the bind -- but it is
     * not live: nothing binds, so no coupled verify is staged for a request that is gone. */
    MtpCoupleBindPoint pb;
    pb.found = f.sch.committed_seq(b, &pb.n_out, &pb.live);
    CHECK(pb.found);
    CHECK(!pb.live);
    CHECK_EQ(pb.n_out, (uint64_t)f.sch.find(b)->output.size());

    const std::vector<RadMtpCoupleState> rbb = readback(b, b_base, 3);
    const StepSlot Sb = slot();
    CHECK_EQ(mtp_couple_bind_rounds(pb, rbb[0].draw, rbb.data(), 1, 3, Sb, vocab, bound,
                                    &n_bound, &why), RAD_OK);
    CHECK_EQ(n_bound, 0);
    /* THE OLD FORMULA would have bound b's rounds at base + n_acc + k -- two past the length it
     * actually committed, on a row no build will stage. */
    for (int k = 0; k < 3; ++k) {
        const uint64_t old_pos = b_base + 4 + (uint64_t)k;
        CHECK(old_pos != pb.n_out + (uint64_t)k);
        RadMtpCoupleState stale = rbb[(size_t)k];
        stale.pos = old_pos;
        CHECK(!rad_mtp_couple_binds(stale, b, rbb[0].draw.seed, pb.n_out, k, draft[k], vocab));
    }

    /* ---- reuse: the next request on a card binds at its own length, and the closed request's
     * rounds are not accepted by it. */
    const uint64_t c = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(c, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t c_base = f.sch.find(c)->output.size();
    CHECK_OK(f.commit(9, 0, [](const StepEntry&) { return 3; }));
    MtpCoupleBindPoint pc;
    pc.found = f.sch.committed_seq(c, &pc.n_out, &pc.live);
    CHECK(pc.found);
    CHECK(pc.live);
    CHECK_EQ(pc.n_out, (uint64_t)f.sch.find(c)->output.size());
    CHECK_EQ(pc.n_out, c_base + 4);
    CHECK(!rad_mtp_couple_binds(rbb[0], c, rbb[0].draw.seed, pc.n_out, 0, draft[0], vocab));
    RadMtpCoupleState own = readback(c, c_base, 1)[0];
    own.pos = rad_mtp_couple_verify_pos(pc.n_out, 1);
    CHECK(rad_mtp_couple_binds(own, c, own.draw.seed, pc.n_out, 0, draft[0], vocab));
}

/* THE BIND POINT OUTLIVES A REAP.
 *
 * commit() retires the step, and a request it finished that the next step does not carry is pinned
 * by nothing any more. The server calls reap() from an HTTP thread as soon as it sees the finish,
 * and reap() destroys the Request -- while Engine::finish_serial_drafter still has to bind that
 * row. The bind point is therefore copied by commit under the scheduler lock
 * (Scheduler::committed_seq), not read off the plan's StepEntry::req, which is dangling by then.
 * This reaps a non-carried terminal request between commit and bind: the plan-pointer version
 * could only answer by reading freed memory, and the snapshot answers from the copy.
 */
TEST(the_committed_bind_point_outlives_a_reap) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    CHECK_OK(f.start());
    const int64_t vocab = f.prog.meta.n_vocab;

    int32_t draft[3] = { 21, 22, 23 };
    const uint64_t a = f.submit(160, 64);                  /* live through the commit */
    const uint64_t b = f.submit(160, /*max_tokens=*/3);    /* closed on it by the cut */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    CHECK_OK(f.sch.set_draft(b, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? 3 : 0; }));
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Finished);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);

    /* The server's reap lands between commit and bind: b's Request is destroyed here. */
    CHECK_OK(f.sch.reap(b));
    CHECK(f.sch.find(b) == nullptr);

    MtpCoupleBindPoint pb;
    pb.found = f.sch.committed_seq(b, &pb.n_out, &pb.live);
    CHECK(pb.found);
    CHECK(!pb.live);                                       /* terminal: nothing binds */
    CHECK_EQ(pb.n_out, (uint64_t)3);

    MtpCoupleBindPoint pa;
    pa.found = f.sch.committed_seq(a, &pa.n_out, &pa.live);
    CHECK(pa.found);
    CHECK(pa.live);
    CHECK_EQ(pa.n_out, (uint64_t)f.sch.find(a)->output.size());

    RadMtpCoupleState rb[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    RadMtpCoupleState out[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    StepSlot S{};
    S.acc = 3;
    S.n_acc = 4;
    for (int rd = 1; rd <= 3; ++rd) S.tok[rd] = draft[rd - 1];
    int n_bound = -1;
    std::string why;
    CHECK_EQ(mtp_couple_bind_rounds(pb, RadMtpCoupleDraw{}, rb, 1, 3, S, vocab, out, &n_bound,
                                    &why), RAD_OK);
    CHECK_EQ(n_bound, 0);
}

/* A decode benchmark needs the request to last as long as asked, not as long as the model
 * chooses. EOS still stops an ordinary request, and max_tokens still stops one ignoring EOS. */
TEST(ignore_eos_runs_to_max_tokens_and_nothing_else_changes) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t plain = f.submit(8, /*max_tokens=*/4);
    uint64_t keep  = f.submit(8, /*max_tokens=*/4, /*priority=*/0, /*ignore_eos=*/true);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(7);                  /* both prefilled, one token each */

    /* Every sequence emits end-of-generation on this step. */
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK(f.sch.step() != nullptr);
    CHECK_OK(f.commit(9, 1));

    CHECK_EQ((int)f.sch.find(plain)->state, (int)ReqState::Finished);
    CHECK_EQ(f.sch.find(plain)->finish_reason, std::string("stop"));
    CHECK_EQ((int)f.sch.find(keep)->state, (int)ReqState::Running);

    /* It ends on the budget instead, and says so. */
    while (f.sch.find(keep)->state == ReqState::Running && f.sch.step() != nullptr)
        CHECK_OK(f.commit(9, 1));
    const Request* rk = f.sch.find(keep);
    CHECK_EQ((int)rk->state, (int)ReqState::Finished);
    CHECK_EQ(rk->finish_reason, std::string("length"));
    CHECK_EQ((int)rk->output.size(), 4);
}

/* Decode rows come first in a mixed step: a block that treats the two kinds differently needs each
 * as a contiguous run, since cu_seqlens is read as cu[n] and cu[n+1]. `running_` is walked in
 * priority order, so without the sort a high-priority prefill lands ahead of a decoding row. */
TEST(a_mixed_step_puts_its_decode_rows_first) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t b = f.submit(1000, 64, /*priority=*/1);   /* several chunks, sorts first */
    uint64_t a = f.submit(160,  64, /*priority=*/0);   /* one chunk, then decodes */

    const RadBatch* s0 = f.sch.step();
    CHECK(s0 != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.plan().n_seq_decode, 0);            /* both prefill: nothing decodes yet */
    f.commit_plain(11);

    /* a finished its prompt and is decoding; b is on its second chunk and still outranks it. */
    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.plan().e[0].seq, a);                /* the decode row, despite the priority */
    CHECK_EQ(f.sch.plan().e[0].is_prefill, false);
    CHECK_EQ(f.sch.plan().e[1].seq, b);
    CHECK_EQ(f.sch.plan().e[1].is_prefill, true);
    CHECK_EQ(f.sch.plan().n_seq_decode, 1);
    CHECK_EQ(s1->n_seq_decode, 1);
    CHECK_EQ(s1->phase, RAD_PHASE_MIXED);
    /* The per-half bounds: a decodes one row, b's chunk is the interval. */
    CHECK_EQ(s1->max_q_len_decode, 1);
    CHECK_EQ(s1->max_q_len_prefill, 256);
    CHECK_EQ(s1->max_q_len, 256);

    /* The entry index followed the row. `entry` is how a request finds its own result, so a sort
     * that leaves it behind hands one request another's token. */
    f.commit_plain(12);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);    /* 11 from the prefill, 12 from the decode */
    CHECK_EQ(f.sch.find(a)->output[1], 12);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);    /* still prefilling: no token yet */
}

/* A step carrying a prefill chunk still takes a draft: the blocks issue the recurrent path over
 * the decode half, because the chunked path has no num_accepted operand to roll a rejection back
 * with. These are the two shapes a refusal would fire on. */
TEST(a_queued_request_does_not_suppress_the_draft) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    f.cfg.max_seqs = 3;                      /* a slot is free, so the queue will be admitted */
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    f.submit(160, 64);
    CHECK_EQ(f.sch.metrics().queue_depth, 1);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));

    const RadBatch* s = f.sch.step();
    CHECK(s != nullptr);
    CHECK_EQ(s->n_spec, 4);                  /* the draft survives the admission */
    CHECK_EQ((int)f.sch.plan().e.size(), 3); /* and the third was admitted into the same step */
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
    CHECK_EQ(s->phase, RAD_PHASE_MIXED);
    /* The decode rows sorted first and carry the window; the prefill chunk carries none. */
    CHECK_EQ(s->n_seq_decode, 2);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
    CHECK_EQ(f.sch.plan().e[1].n_spec, 4);
    CHECK_EQ(f.sch.plan().e[2].n_spec, 0);
    /* Two decode rows of 1 + 4, so the blocks split after 10 tokens and then the chunk. A wrong
     * split lets a per-token op reach into the other half. */
    CHECK_EQ(s->n_tok_decode, 10);
    CHECK_EQ(s->max_q_len_decode, 5);
    CHECK_EQ(s->max_q_len_prefill, 160);
}

TEST(a_running_prefill_does_not_suppress_the_draft) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160,  64);
    f.submit(1000, 64);                      /* several chunks: mixed for the next few steps */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));

    const RadBatch* s = f.sch.step();
    CHECK(s != nullptr);
    CHECK_EQ(s->phase, RAD_PHASE_MIXED);
    CHECK_EQ(s->n_spec, 4);
    CHECK_EQ(s->n_seq_decode, 1);
    CHECK_EQ(s->n_tok_decode, 5);
    CHECK_EQ(f.sch.plan().e[0].seq, a);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
}

/* ================================================================== a prefill sharing a step */
/* --busy-prefill-chunk. The starving shape: a checkpoint interval as large as the step budget,
 * so a long prefill's chunk can take the whole step every step. Here 512 and 512, quantum 64; the
 * sequence is long enough for many full chunks. */
static void share_fixture(Fixture& f, int64_t mode) {
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/4096);
    f.cfg.checkpoint_interval = 512;
    f.cfg.busy_prefill_chunk = mode;
}

/* With `off`, once the long prefill owns the step a newly arriving request is not even admitted
 * until it ends. */
TEST(off_a_long_prefill_keeps_a_new_request_out_of_every_step) {
    Fixture f;
    share_fixture(f, /*off=*/0);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
    f.commit_plain();
    uint64_t b = f.submit(100, 64);
    for (int k = 0; k < 4; ++k) {
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ((int)f.sch.plan().e.size(), 1);
        CHECK_EQ(f.sch.plan().e[0].seq, a);
        CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
        f.commit_plain();
    }
    CHECK_EQ(f.sch.metrics().queue_depth, 1);
    CHECK(f.sch.find(b) != nullptr);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
}

/* auto: the same arrival is admitted in the next step, and its prompt finishes in two. */
TEST(auto_a_request_arriving_during_a_long_prefill_is_admitted_at_once) {
    Fixture f;
    share_fixture(f, /*auto=*/-1);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);       /* alone: the whole step */
    f.commit_plain();
    uint64_t b = f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
    int64_t a_tok = 0, b_tok = 0;
    for (const StepEntry& e : f.sch.plan().e) {
        if (e.seq == a) a_tok = e.n_tokens;
        if (e.seq == b) b_tok = e.n_tokens;
    }
    CHECK(a_tok >= 64 && a_tok % 64 == 0);
    CHECK(b_tok > 0);
    CHECK(a_tok + b_tok <= 512);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 1);
    f.commit_plain(11);
    /* b's 100 tokens are done by the next step and it decodes beside the prefill from then on */
    for (int k = 0; k < 4; ++k) {
        CHECK(f.sch.step() != nullptr);
        bool saw_b = false;
        for (const StepEntry& e : f.sch.plan().e) if (e.seq == b) saw_b = true;
        CHECK(saw_b);
        f.commit_plain(12);
    }
    CHECK((int)f.sch.find(b)->output.size() >= 2);
}

/* A decode row behind a long prefill in running_ (the prefill outranks it) is planned every step
 * under auto; under off it is left out of every step that begins on an interval boundary. */
static int decode_rows_seen(int64_t mode) {
    Fixture f;
    share_fixture(f, mode);
    CHECK_OK(f.start());
    uint64_t b = f.submit(100, 512, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);            /* b's whole prompt: it decodes from the next step */
    f.commit_plain(11);
    f.submit(3500, 64, /*priority=*/1);        /* outranks b: walked first */
    int seen = 0;
    for (int k = 0; k < 8; ++k) {
        CHECK(f.sch.step() != nullptr);
        for (const StepEntry& e : f.sch.plan().e) if (e.seq == b && !e.is_prefill) ++seen;
        f.commit_plain(12);
    }
    return seen;
}

TEST(auto_a_decode_row_behind_a_long_prefill_gets_every_step) {
    CHECK_EQ(decode_rows_seen(-1), 8);
    CHECK(decode_rows_seen(0) < 8);
}

/* With nothing else waiting the cap does not apply, and a lone prefill is planned chunk for chunk
 * the same under both modes, which keeps every single-request gate unchanged. */
TEST(auto_a_lone_prefill_is_planned_as_before) {
    std::vector<int32_t> plans[2];
    for (int m = 0; m < 2; ++m) {
        Fixture f;
        share_fixture(f, m == 0 ? 0 : -1);
        CHECK_OK(f.start());
        f.submit(3000, 4);
        for (int k = 0; k < 10; ++k) {
            if (!f.sch.step()) break;
            plans[m].push_back(f.sch.plan().e[0].n_tokens);
            f.commit_plain();
        }
        CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
    }
    CHECK(plans[0] == plans[1]);
}

/* An explicit N holds the chunk to N while something waits. */
TEST(an_explicit_busy_prefill_chunk_holds_the_chunk_to_it) {
    Fixture f;
    share_fixture(f, 128);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);       /* alone */
    f.commit_plain();
    f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    for (const StepEntry& e : f.sch.plan().e)
        if (e.seq == a) CHECK_EQ(e.n_tokens, 128);
}

/* ================================================================== decode-only steps between chunks */
/* --prefill-interleave. A sequence decodes while a 2000-token prompt prefills in 256-token chunks
 * (the checkpoint interval). Each step's shape is 'C' when it carries a chunk of the prompt and
 * 'D' when it carries decode rows only; `staged` records prefill_continues() after each step,
 * which arms the prefill stager for the next one. */
struct Interleaved {
    std::string shape;
    std::string staged;
    std::string more;      /* prefill_has_more() after each step (RADIANCE_STAGE_MIXED=2) */
    std::string want;      /* '+' where the step's chunk of the prompt leaves >= 64 rows of it */
    int64_t steps_held = 0;
};

/* `chunk_s` / `decode_s`: step costs fed to --prefill-interleave auto before the run (none when
 * 0), standing in for step_finished's wall times. */
static Interleaved interleave_run(int64_t n, bool decoder, int steps, double chunk_s = 0,
                                  double decode_s = 0) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = n;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(64);
    if (chunk_s > 0) f.sch.note_step_cost(true, chunk_s);
    if (decode_s > 0) f.sch.note_step_cost(false, decode_s);
    if (decoder) {
        f.submit(100, 512);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
    }
    const uint64_t p = f.submit(2000, 4);
    Interleaved out;
    for (int k = 0; k < steps; ++k) {
        if (!f.sch.step()) break;
        bool chunk = false, left = false;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == p && e.is_prefill) {
                chunk = true;
                left = 2000 - ((int64_t)e.ctx_len + e.n_tokens) >= 64;
            }
        out.shape += chunk ? 'C' : 'D';
        out.want += left ? '+' : '-';
        out.staged += f.sch.prefill_continues() ? '+' : '-';
        out.more += f.sch.prefill_has_more() ? '+' : '-';
        f.commit_plain(12);
    }
    out.steps_held = f.sch.metrics().steps_held;
    return out;
}

/* Between a commit and the next step (where an encoder pass is prepared) the committed step's
 * plan names a request the scheduler may already have released: a client cancelled it and the
 * server reaped it. prefill_continues() reads only an open step's plan. */
TEST(prefill_continues_reads_no_committed_plan) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.sch.set_stage_rows(64);
    const uint64_t p = f.submit(2000, 4);
    CHECK(f.sch.step() != nullptr);
    CHECK(f.sch.prefill_continues());
    f.commit_plain(12);
    CHECK_OK(f.sch.cancel(p));
    CHECK_OK(f.sch.reap(p));
    CHECK(!f.sch.prefill_continues());
}

TEST(interleave_off_puts_a_chunk_in_every_step) {
    const Interleaved r = interleave_run(0, /*decoder=*/true, 8);
    CHECK_EQ(r.shape, std::string("CCCCCCCC"));
    CHECK_EQ(r.staged, std::string("+++++++-"));      /* the eighth chunk is the last */
    CHECK_EQ(r.steps_held, 0);
}

/* N decode-only steps after each chunk, which admit nothing, and the stager is not armed for a
 * chunk the next step will hold: its staged experts would be dropped. */
TEST(interleave_runs_n_decode_only_steps_after_each_chunk) {
    const Interleaved r = interleave_run(2, /*decoder=*/true, 10);
    CHECK_EQ(r.shape, std::string("CDDCDDCDDC"));
    CHECK_EQ(r.staged, std::string("----------"));
    CHECK_EQ(r.steps_held, 6);
}

/* prefill_has_more() answers for the step's own chunk whatever the next step is held to: '+' on a
 * chunk that leaves at least the stager's smallest pass (64 here) of the prompt, '-' otherwise and
 * on a step that carries none. */
TEST(prefill_has_more_ignores_the_interleave_hold) {
    const Interleaved r = interleave_run(2, /*decoder=*/true, 24);
    CHECK_EQ(r.more, r.want);                            /* '+' on chunks that leave >= 64 rows */
    CHECK(r.more.find('+') != std::string::npos);
    CHECK_EQ(r.staged, std::string(r.staged.size(), '-'));   /* the hold: prefill_continues says no */
    const Interleaved off = interleave_run(0, /*decoder=*/true, 8);
    CHECK_EQ(off.more, off.staged);                      /* no hold: the same as prefill_continues */
}

/* Nothing decodes beside a lone prefill, so nothing is held. */
TEST(interleave_leaves_a_lone_prefill_alone) {
    const Interleaved r = interleave_run(2, /*decoder=*/false, 8);
    CHECK_EQ(r.shape, std::string("CCCCCCCC"));
    CHECK_EQ(r.steps_held, 0);
}

/* auto: as many decode-only steps as fit in a quarter of a chunk step's measured time at a decode
 * step's (rounded down, at most 16), one until both are measured, none where a decode step costs
 * more than that quarter. */
TEST(interleave_auto_fits_decode_steps_in_a_quarter_of_a_chunk_step) {
    auto steps = [](double chunk_s, double decode_s) {
        Fixture f;
        f.build();
        f.cfg.prefill_interleave = -1;
        CHECK_OK(f.start());
        if (chunk_s > 0) f.sch.note_step_cost(true, chunk_s);
        if (decode_s > 0) f.sch.note_step_cost(false, decode_s);
        return f.sch.interleave_steps();
    };
    CHECK_EQ(steps(0, 0), 1);
    CHECK_EQ(steps(0.4, 0), 1);
    CHECK_EQ(steps(0, 0.03), 1);
    CHECK_EQ(steps(0.4, 0.03), 3);         /* 0.1 s of 0.03 s steps */
    CHECK_EQ(steps(0.4, 0.05), 2);
    CHECK_EQ(steps(0.4, 0.2), 0);          /* a decode step dearer than the quarter */
    CHECK_EQ(steps(1.0, 0.001), 64);       /* capped at kInterleaveAutoMax */
    /* auto:SHARE: SHARE of the chunk step's time instead of a quarter. */
    Fixture k;
    k.build();
    k.cfg.prefill_interleave = -1;
    k.cfg.prefill_interleave_share = 1.0;
    CHECK_OK(k.start());
    k.sch.note_step_cost(true, 0.4);
    k.sch.note_step_cost(false, 0.04);
    CHECK_EQ(k.sch.interleave_steps(), 10);  /* 0.4 s of 0.04 s steps */
    /* A number is used as given, whatever was measured. */
    Fixture g;
    g.build();
    g.cfg.prefill_interleave = 5;
    CHECK_OK(g.start());
    g.sch.note_step_cost(true, 1.0);
    g.sch.note_step_cost(false, 0.001);
    CHECK_EQ(g.sch.interleave_steps(), 5);
    /* The costs are running means: a chunk step of 0.4 s, then one of 0.8 s, reads 0.5 s. */
    Fixture h;
    h.build();
    h.cfg.prefill_interleave = -1;
    CHECK_OK(h.start());
    h.sch.note_step_cost(true, 0.4);
    h.sch.note_step_cost(true, 0.8);
    h.sch.note_step_cost(false, 0.05);
    CHECK_EQ(h.sch.interleave_steps(), 2);  /* 0.125 / 0.05 */
}

/* auto schedules the measured count after every chunk beside a decoder, holds nothing for a lone
 * prefill, and with decode steps too dear for the quarter puts a chunk in every step. */
TEST(interleave_auto_holds_each_chunk_for_the_steps_its_costs_allow) {
    const Interleaved r = interleave_run(-1, /*decoder=*/true, 10, 0.4, 0.05);
    CHECK_EQ(r.shape, std::string("CDDCDDCDDC"));
    CHECK_EQ(r.steps_held, 6);
    const Interleaved first = interleave_run(-1, /*decoder=*/true, 6);   /* nothing measured */
    CHECK_EQ(first.shape, std::string("CDCDCD"));
    const Interleaved dear = interleave_run(-1, /*decoder=*/true, 8, 0.4, 0.2);
    CHECK_EQ(dear.shape, std::string("CCCCCCCC"));
    CHECK_EQ(dear.steps_held, 0);
    const Interleaved lone = interleave_run(-1, /*decoder=*/false, 8, 0.4, 0.01);
    CHECK_EQ(lone.shape, std::string("CCCCCCCC"));
    CHECK_EQ(lone.steps_held, 0);
}

/* After a prompt's last chunk there is nothing to interleave: the next request is admitted in the
 * next step. */
TEST(interleave_does_not_hold_after_a_prompts_last_chunk) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = 2;
    CHECK_OK(f.start());
    f.submit(100, 512);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.submit(200, 4);                      /* one chunk, beside the decoder */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().phase, RAD_PHASE_MIXED);
    f.commit_plain(12);
    const uint64_t q = f.submit(300, 4);
    CHECK(f.sch.step() != nullptr);
    bool admitted = false;
    for (const StepEntry& e : f.sch.plan().e) if (e.seq == q && e.is_prefill) admitted = true;
    CHECK(admitted);
    CHECK_EQ(f.sch.metrics().steps_held, 0);
}

/* A decode-only step of the hold may be issued ahead; it counts toward N and as a held step, and
 * once the hold is spent the next step is a full one that carries the chunk. */
TEST(interleave_counts_steps_issued_ahead_and_then_serves_the_chunk) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = 2;
    CHECK_OK(f.start());
    const uint64_t d = f.submit(100, 512);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.sch.find(d)->speculation_fixed = true;
    const uint64_t p = f.submit(2000, 4);
    CHECK(f.sch.step() != nullptr);                     /* chunk beside the decoder */
    CHECK(f.sch.step_next() == nullptr);                /* a prompt continues: no issuing ahead */
    f.commit_plain(12);

    CHECK(f.sch.step() != nullptr);                     /* held: the decoder alone */
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK(f.sch.step_next() != nullptr);                /* the hold's second step, ahead */
    f.commit_plain(12);
    f.sch.promote_next();
    CHECK_EQ(f.sch.metrics().steps_held, 2);
    CHECK(f.sch.step_next() == nullptr);                /* the hold is spent */
    f.commit_plain(12);

    CHECK(f.sch.step() != nullptr);
    bool chunk = false;
    for (const StepEntry& e : f.sch.plan().e) if (e.seq == p && e.is_prefill) chunk = true;
    CHECK(chunk);
}

/* The arithmetic at a real deployment's shape: a 2048-token step and interval, the stager arming
 * at 1025 tokens (so a held-back chunk keeps 1024 and its step-mates take the rest). */
TEST(busy_cap_keeps_the_stager_floor_and_never_leaves_a_runt) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/2048, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = 2048;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1025);
    /* nothing reserved: the budget */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 0), 2048);
    /* a little reserved at an interval boundary: 2048 - 272 would leave a 272-token runt before
     * the boundary, so the interval splits in two staged halves instead */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 272), 1024);
    /* half way to the boundary the rest of the interval already fits under the cap */
    CHECK(f.sch.busy_cap(1024, 100000, 2048, 272) >= 1024);
    CHECK_EQ(f.sch.plan_chunk(1024, 100000, f.sch.busy_cap(1024, 100000, 2048, 272), nullptr), 1024);
    /* more reserved than the floor allows for: the floor */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 1500), 1024);
    /* a prompt's last stretch, shorter than the cap: all of it */
    CHECK_EQ(f.sch.plan_chunk(0, 1500, f.sch.busy_cap(0, 1500, 2048, 272), nullptr), 1500);
    /* a budget already under the floor: never more than the budget */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 512, 64), 512);
    /* decode rows alone behind it (2 rows x (1 + 3 drafts)): no split; the chunk ends a quantum
     * short of the boundary (1984) and the next step takes the remaining 64, the pair a prefill
     * already runs with decode rows ahead of it */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 8, false, /*rows_only=*/true), 2040);
    CHECK_EQ(f.sch.plan_chunk(0, 100000, f.sch.busy_cap(0, 100000, 2048, 8, false, true), nullptr), 1984);
    /* the same reserve with a share in it splits the interval */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 8 + 256, false, false), 1024);
}

/* The share left for a request that then does not start (its media is not encoded yet) goes back
 * to the chunk that made room for it: the step is planned as if nothing had waited. */
TEST(auto_a_share_nobody_takes_goes_back_to_the_prefill) {
    Fixture f;
    share_fixture(f, -1);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    std::shared_ptr<mm::Item> ib;
    REQUIRE(f.submit_unencoded(100, 64, &ib) != 0);
    CHECK(f.sch.step() != nullptr);
    REQUIRE_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_EQ(f.sch.plan().e[0].seq, a);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
    CHECK_EQ(f.sch.metrics().queue_depth, 1);
}

/* An explicit N also holds a chunk back for rows planned ahead of it (the operator trades prefill
 * speed for every other row's latency); auto leaves those alone. */
TEST(an_explicit_busy_prefill_chunk_applies_to_rows_ahead_auto_does_not) {
    for (int64_t mode : { (int64_t)-1, (int64_t)128 }) {
        Fixture f;
        share_fixture(f, mode);
        CHECK_OK(f.start());
        uint64_t b = f.submit(100, 512, /*priority=*/1);   /* outranks the prefill: planned first */
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        uint64_t a = f.submit(3500, 64, /*priority=*/0);
        CHECK(f.sch.step() != nullptr);                    /* b decodes, a is admitted behind it */
        f.commit_plain(12);
        int64_t most = 0; bool saw_b = true;
        for (int k = 0; k < 3; ++k) {
            CHECK(f.sch.step() != nullptr);
            bool here = false;
            for (const StepEntry& e : f.sch.plan().e) {
                if (e.seq == a) most = std::max<int64_t>(most, e.n_tokens);
                if (e.seq == b) here = true;
            }
            saw_b = saw_b && here;
            f.commit_plain(12);
        }
        CHECK(saw_b);
        if (mode > 0) CHECK_EQ(most, 128);
        else          CHECK(most > 128);
    }
}

TEST(busy_cap_is_the_budget_when_off) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/2048, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.busy_prefill_chunk = 0;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1025);
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 272), 2048);
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 1500), 2048);
}

namespace {
/* A small deterministic coin: true with probability `pr`. */
struct Coin {
    uint64_t s = 12345;
    bool operator()(double pr) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return ((double)((s >> 33) & 0xFFFFFF) / (double)0x1000000) < pr;
    }
};
}  /* namespace */

TEST(the_draft_depth_is_the_configured_one_whatever_the_acceptance_trace) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 8;
    CHECK_OK(c.configure(p));
    CHECK_EQ(c.next_depth(), 8);

    /* A workload that rejects almost everything. A dynamic policy would collapse the depth here;
     * this one does not move: the depth is the flag. */
    Coin coin;
    for (int t = 0; t < 4000; ++t) {
        int d = c.next_depth();
        CHECK_EQ(d, 8);
        int a = 0;
        while (a < d && coin(0.1)) ++a;
        c.observe(d, a);
    }
    CHECK_EQ(c.next_depth(), 8);
    CHECK(c.acceptance() > 0.0f && c.acceptance() < 0.3f);

    /* The accounting is exact and charged per drafted token, not per step: it is what a scraped
     * acceptance rate divides by. */
    DraftController d2;
    DraftController::Params q;
    q.max_depth = 8;
    CHECK_OK(d2.configure(q));
    for (int i = 0; i < 100; ++i) d2.observe(8, 0);
    CHECK_EQ(d2.drafted(), 800);
    CHECK_EQ(d2.accepted(), 0);
    CHECK_NEAR(d2.acceptance(), 0.0, 1e-9);
    d2.observe(2, 5);                 /* accepted past the draft is clamped, not counted */
    CHECK_EQ(d2.accepted(), 2);
    CHECK_EQ(d2.drafted(), 802);

    /* Depth 0 is speculation off, and it configures. */
    DraftController d3;
    DraftController::Params z;
    z.max_depth = 0;
    CHECK_OK(d3.configure(z));
    CHECK_EQ(d3.next_depth(), 0);
}

/* ------------------------------------------------------------------ the adaptive window
 * --draft-window adaptive (draft.h). The controller cases turn the policy on through Params, which
 * is what Scheduler::init does with a serial drafter; the scheduler cases start a Fixture whose
 * program declares one and whose Config asks for the window. */

namespace {
DraftController adaptive_controller(int max_depth) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = max_depth;
    p.adapt = true;
    CHECK_OK(c.configure(p));
    return c;
}

/* An estimate as a test states it: acceptance `a0` at the first position and `rest` after it,
 * started, with its stop where the controller puts it. */
DraftAdapt estimate(const DraftController& c, double a0, double rest) {
    DraftAdapt s;
    s.a[0] = (float)a0;
    for (int j = 1; j < RAD_SCHED_MAX_SPEC; ++j) s.a[j] = (float)rest;
    /* The stop draft.h states: the deepest draft emitted at least kDraftMinChance of the time. */
    double p = 1.0;
    int d = 0;
    while (d < c.next_depth()) {
        p *= (double)s.a[d];
        if (p < DraftController::kDraftMinChance) break;
        ++d;
    }
    s.depth = d < DraftController::kMinDepth ? DraftController::kMinDepth : d;
    return s;
}

/* What a scheduler fixture with the adaptive window looks like: a serial head of `depth`
 * declared by the program, --num-speculative-tokens `depth`, --draft-window adaptive. */
void adaptive_fixture(Fixture& f, int depth) {
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = depth;
    f.cfg.draft_window = DraftWindow::Adaptive;
    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = depth;
}

/* What `fn` logs at Info and above: the log lines a case checks for. */
std::string info_of(const std::function<void()>& fn) {
    const Log level = log_level();
    log_set_level(Log::Info);
    std::string text = stderr_of(fn);
    log_set_level(level);
    return text;
}
}  /* namespace */

TEST(the_adaptive_window_follows_each_sequences_own_acceptance) {
    DraftController c = adaptive_controller(6);
    CHECK(c.adaptive());
    CHECK_EQ(c.next_depth(), 6);                 /* the ceiling is still the flag */
    CHECK_EQ(c.step_depth(), 0);                 /* no step has chosen D: the ceiling stands */

    /* Nothing measured yet: both start from the prior, which drafts the whole window. */
    DraftAdapt code, summary;
    CHECK_EQ(c.window_for(code), 6);
    CHECK_EQ(c.window_for(summary), 6);

    /* Two texts through one controller. The drafter almost always agrees on one; on the other it
     * agrees at the first position and rarely after it. Each stop follows its own text. */
    Coin coin;
    for (int t = 0; t < 600; ++t) {
        int d = c.window_for(code), a = 0;
        while (a < d && coin(0.95)) ++a;
        c.observe(d, a);
        c.adapt_observe(code, d, a);

        d = c.window_for(summary);
        a = 0;
        while (a < d && coin(a == 0 ? 0.6 : 0.2)) ++a;
        c.observe(d, a);
        c.adapt_observe(summary, d, a);
    }
    CHECK_EQ(c.window_for(code), 6);
    CHECK(c.window_for(summary) >= 1);
    CHECK(c.window_for(summary) <= 2);
    CHECK(code.a[0] > 0.85f);
    CHECK_NEAR(summary.a[0], 0.6, 0.2);
    CHECK(c.yield(code, 6) > c.yield(summary, 6));
    /* The global hazard is still counted, and the windows the verifies carried. */
    CHECK(c.reached(0) == 1200);
    CHECK(c.verified_at(6) >= 600);
    CHECK(c.verified_at(1) + c.verified_at(2) > 0);
    CHECK_EQ(c.verified_at(1) + c.verified_at(2) + c.verified_at(3) + c.verified_at(4) +
             c.verified_at(5) + c.verified_at(6), (int64_t)1200);

    /* A stop that went shallow comes back when the text changes: the positions it stopped
     * drafting drift toward the deepest it still drafts, so a text the drafter predicts again is
     * found without anyone drafting deep first. */
    for (int t = 0; t < 600; ++t) {
        const int d = c.window_for(summary);
        int a = 0;
        while (a < d && coin(0.95)) ++a;
        c.adapt_observe(summary, d, a);
    }
    CHECK_EQ(c.window_for(summary), 6);
}

/* A stop moves only past the hold: it is the sequence's share of the step's pass shape, and a
 * shape following every flicker of its estimate would re-record its passes for nothing
 * (runtime/ctx.h). It never goes below one draft, or the estimate could never recover. */
TEST(the_adaptive_window_moves_only_past_its_hold_and_never_below_one_draft) {
    DraftController c = adaptive_controller(6);
    DraftAdapt s;
    CHECK_EQ(c.window_for(s), 6);
    for (int t = 0; t < DraftController::kHold - 1; ++t) CHECK_EQ(c.adapt_observe(s, 6, 0), 6);
    CHECK(c.adapt_observe(s, 6, 0) < 6);                       /* the kHold-th moves */

    for (int t = 0; t < 400; ++t) c.adapt_observe(s, c.window_for(s), 0);
    CHECK_EQ(c.window_for(s), DraftController::kMinDepth);

    /* Off, the window is the flag and an observation is not taken. */
    DraftController off;
    DraftController::Params q;
    q.max_depth = 5;
    CHECK_OK(off.configure(q));
    CHECK(!off.adaptive());
    DraftAdapt w;
    CHECK_EQ(off.window_for(w), 5);
    CHECK_EQ(off.adapt_observe(w, 5, 0), 5);
    CHECK_EQ(w.depth, 0);
    off.observe(5, 2);
    CHECK_EQ(off.verified_at(5), (int64_t)0);
    off.step_time(3, 1, 12.0);                                  /* and no step is timed */
    CHECK_EQ(off.cost_samples(3, 1), (int64_t)0);

    /* Adaptive with no speculation is no policy at all. */
    DraftController zero = adaptive_controller(0);
    CHECK(!zero.adaptive());
}

/* The step cost is measured. A key (the step's deepest window, its class of running sequences)
 * with samples is its own mean; one without borrows the nearest measured key's, scaled by the
 * prior's shape; with nothing measured it is the prior. Same times, same choice. */
TEST(the_adaptive_window_prices_a_step_from_measured_step_times) {
    DraftController c = adaptive_controller(6);
    CHECK_NEAR(c.step_cost(1, 1), 1.06, 1e-12);                /* the prior */
    CHECK_NEAR(c.step_cost(6, 1), 1.36, 1e-12);

    for (int i = 0; i < 10; ++i) c.step_time(3, 1, 12.0);
    CHECK_EQ(c.cost_samples(3, 1), (int64_t)10);
    CHECK_NEAR(c.step_cost(3, 1), 12.0, 1e-9);
    CHECK_NEAR(c.step_cost(5, 1), 12.0 * 1.30 / 1.18, 1e-9);   /* borrowed, the prior's shape */
    CHECK_NEAR(c.step_cost(3, 9), 12.0, 1e-9);                 /* the nearest class */
    CHECK_EQ(c.cost_samples(3, 9), (int64_t)0);
    /* The classes: 1, 2-3, 4-7, 8-15, 16 and up. */
    c.step_time(3, 2, 20.0);
    CHECK_NEAR(c.step_cost(3, 3), 20.0, 1e-9);
    CHECK_NEAR(c.step_cost(3, 4), 20.0, 1e-9);                 /* 4-7 borrows from 2-3 */
    c.step_time(3, 16, 40.0);
    CHECK_NEAR(c.step_cost(3, 1000), 40.0, 1e-9);
    CHECK_NEAR(c.step_cost(3, 9), 40.0, 1e-9);                 /* 8-15: the nearest, 16 and up */

    /* A step that waited on something else counts for at most four times the mean. */
    const double before = c.step_cost(3, 1);
    c.step_time(3, 1, 1.0e6);
    CHECK(c.step_cost(3, 1) <= before * 4.0);
    CHECK(c.step_cost(3, 1) > before);

    /* Nonsense is not a sample. */
    c.step_time(0, 1, 5.0);
    c.step_time(7, 1, 5.0);
    c.step_time(2, 1, -1.0);
    c.step_time(2, 0, 5.0);
    CHECK_EQ(c.cost_samples(2, 1), (int64_t)0);

    /* One sequence that accepts nearly everything: on the prior the whole window pays... */
    DraftController p = adaptive_controller(6);
    DraftAdapt s = estimate(p, 0.9, 0.9);
    DraftAdapt* one[1] = { &s };
    CHECK_EQ(p.plan_step(one, 1, 1), 6);
    /* ...and where the steps past three were measured dear, three is what pays. */
    DraftController m = adaptive_controller(6);
    const double t[7] = { 0, 10.0, 11.0, 12.0, 30.0, 30.0, 30.0 };
    for (int d = 1; d <= 6; ++d) m.step_time(d, 1, t[d]);
    DraftAdapt u = estimate(m, 0.9, 0.9);
    DraftAdapt* mu[1] = { &u };
    CHECK_EQ(m.plan_step(mu, 1, 1), 3);
    CHECK_EQ(m.window_for(u), 3);
    /* Deterministic: the same times and the same estimate give the same depth. */
    DraftController m2 = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) m2.step_time(d, 1, t[d]);
    DraftAdapt u2 = estimate(m2, 0.9, 0.9);
    DraftAdapt* mu2[1] = { &u2 };
    CHECK_EQ(m2.plan_step(mu2, 1, 1), 3);
}

/* The head depth is the step's. A serial head runs as many rounds as the step's deepest window,
 * so a sequence that accepts deep is worth deep rounds only while its neighbours' loss to them
 * (every round is a pass over every drafting row) does not outweigh it. */
TEST(the_adaptive_window_chooses_the_head_depth_over_the_whole_batch) {
    DraftController c = adaptive_controller(6);
    DraftAdapt deep = estimate(c, 0.95, 0.95);
    DraftAdapt shallow = estimate(c, 0.5, 0.15);
    CHECK_EQ(deep.depth, 6);
    CHECK_EQ(shallow.depth, 1);                 /* its second draft is emitted 7.5 % of the time */

    /* Side by side on the prior: deep rounds pay for the deep sequence, and the shallow one stops
     * at its own first draft under them. */
    DraftAdapt* two[2] = { &deep, &shallow };
    CHECK_EQ(c.plan_step(two, 2, 2), 6);
    CHECK_EQ(c.window_for(deep), 6);
    CHECK_EQ(c.window_for(shallow), 1);

    /* The same deep sequence among seven shallow ones, at a class whose steps were measured to
     * grow by three units per round: eight rows pay for the rounds, and the one that gains does not
     * repay it. Every window comes down to one. */
    DraftController b = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) b.step_time(d, 8, 10.0 + 3.0 * d);
    DraftAdapt bd = estimate(b, 0.95, 0.95);
    DraftAdapt bs[7];
    DraftAdapt* eight[8] = { &bd };
    for (int i = 0; i < 7; ++i) { bs[i] = estimate(b, 0.5, 0.15); eight[i + 1] = &bs[i]; }
    CHECK_EQ(b.plan_step(eight, 8, 8), 1);
    CHECK_EQ(b.window_for(bd), 1);
    /* Alone at its own class, unmeasured, the same sequence takes the whole window. */
    DraftController a = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) a.step_time(d, 8, 10.0 + 3.0 * d);
    DraftAdapt ad = estimate(a, 0.95, 0.95);
    DraftAdapt* lone[1] = { &ad };
    CHECK(a.plan_step(lone, 1, 1) > 1);

    /* Rows that do not draft are part of the step: they emit a token whatever D is and pay for
     * every round. Here three plain rows beside one drafting row at the same measured class. */
    DraftController r = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) r.step_time(d, 4, 10.0 + 3.0 * d);
    DraftAdapt rd = estimate(r, 0.95, 0.95);
    DraftAdapt* rone[1] = { &rd };
    const int with_plain = r.plan_step(rone, 1, 4);
    DraftController r2 = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) r2.step_time(d, 4, 10.0 + 3.0 * d);
    DraftAdapt rd2 = estimate(r2, 0.95, 0.95);
    DraftAdapt* rone2[1] = { &rd2 };
    const int alone = r2.plan_step(rone2, 1, 1);    /* class 1 unmeasured: borrows class 4-7 */
    CHECK(with_plain <= alone);

    /* D moves past a hold and a hysteresis. The first step that drafts takes the best outright;
     * after that a better depth waits kHold steps. */
    DraftController h = adaptive_controller(6);
    DraftAdapt hs = estimate(h, 0.95, 0.95);
    DraftAdapt* hone[1] = { &hs };
    CHECK_EQ(h.plan_step(hone, 1, 1), 6);
    for (int d = 1; d <= 6; ++d) h.step_time(d, 1, d <= 2 ? 10.0 : 40.0);
    for (int t = 0; t < DraftController::kHold - 1; ++t) CHECK_EQ(h.plan_step(hone, 1, 1), 6);
    CHECK_EQ(h.plan_step(hone, 1, 1), 2);
    /* No drafting sequence: D stays where it is. */
    CHECK_EQ(h.plan_step(hone, 0, 3), 2);
}

/* The flag, for a drafting head (serial or block) or prompt lookup alone. --draft-window adaptive
 * with no speculation keeps the configured depth and says so at startup. The default is the fixed
 * window. */
TEST(the_adaptive_window_is_a_flag_for_a_drafting_head_and_said_when_there_is_none) {
    {
        Fixture f;
        adaptive_fixture(f, 4);
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(f.sch.draft().adaptive());
        CHECK(log.find("draft depth: adaptive") != std::string::npos);
    }
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.cfg.draft_window = DraftWindow::Fixed;
        CHECK_OK(f.start());
        CHECK(!f.sch.draft().adaptive());
        Config d;
        CHECK(d.draft_window == kDraftWindowDefault);      /* the default */
    }
    /* A block drafter takes the policy with its rounds fixed. It proposes its whole block in one
     * pass, and each sequence verifies the block's prefix up to its own window: the whole block
     * while its estimate is the prior, one draft once its drafts stop being taken. Startup says so. */
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.prog.drafter.kind = RAD_DRAFT_BLOCK;
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(f.sch.draft().adaptive());
        CHECK(log.find("draft depth: adaptive") != std::string::npos);
        CHECK(log.find("with a block drafter") != std::string::npos);
        CHECK(log.find("the window is fixed") == std::string::npos);

        const uint64_t a = f.submit(160, 2000);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 4);
        for (int t = 0; t < 64; ++t) {               /* no draft is ever taken */
            REQUIRE(f.sch.step() != nullptr);
            f.commit_plain(9);
            CHECK_OK(f.sch.set_draft(a, draft, 4));
        }
        CHECK_EQ(f.sch.find(a)->n_draft, 1);
        REQUIRE(f.sch.step() != nullptr);
        for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, 1);
        CHECK(f.sch.metrics().draft_adaptive);
        CHECK(f.sch.metrics().draft_verify_depth[4] > 0);
        CHECK(f.sch.metrics().draft_verify_depth[1] > 0);
    }
    /* Prompt lookup as the only drafter takes the policy as a block drafter does: each of its
     * drafts is a verify row (prompt_lookup_alone_follows_its_own_acceptance). */
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.prog.drafter.kind = RAD_DRAFT_NONE;
        f.prog.drafter.depth = 0;
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(f.sch.draft().adaptive());
        CHECK(log.find("--draft-window adaptive with prompt lookup only") != std::string::npos);
        CHECK(log.find("the window is fixed") == std::string::npos);

        const uint64_t a = f.submit(160, 64);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 4);
        CHECK(f.sch.metrics().draft_adaptive);
    }
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.cfg.n_spec = 0;
        const std::string log = info_of([&] { CHECK_OK(f.start()); });
        CHECK(!f.sch.draft().adaptive());
        CHECK(log.find("does not speculate") != std::string::npos);
    }
}

/* Each sequence has its own stop, and the rounds follow the deepest. The step issued ahead of a
 * commit verifies, per sequence, the first rounds up to its own window, and the drafter's longer
 * proposal is cut to that count rather than refused as a mismatch. */
TEST(the_adaptive_window_is_per_sequence_and_the_rounds_follow_the_deepest) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    const uint64_t a = f.submit(160, 2000);
    const uint64_t b = f.submit(160, 2000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    /* `a`'s drafts are never taken and `b`'s always are. */
    int32_t draft[4] = { 21, 22, 23, 24 };
    auto commit_ab = [&]() {
        CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? e.n_spec : 0; }));
    };
    for (int t = 0; t < 64; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_OK(f.sch.set_draft(b, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        commit_ab();
    }

    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    const RadBatch* s = f.sch.step();
    REQUIRE(s != nullptr);
    CHECK_EQ(f.sch.draft().step_depth(), 4);
    CHECK_EQ(s->n_spec, 4);
    CHECK_EQ(s->n_tok, 2 + 5);
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, e.seq == a ? 1 : 4);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 2.5, 1e-6);

    /* The head runs for the deepest. */
    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 2);
    CHECK_EQ(rounds, 4);
    REQUIRE(f.sch.serial_batch(0, 1) != nullptr);
    for (int rd = 2; rd <= rounds; ++rd) REQUIRE(f.sch.serial_batch(rd, 1) != nullptr);

    /* The step issued ahead: `a` verifies one of the four rounds, `b` all four. */
    f.sch.find(a)->speculation_fixed = true;
    f.sch.find(b)->speculation_fixed = true;
    f.sch.expect_draft(a, rounds);
    f.sch.expect_draft(b, rounds);
    const RadBatch* nb = f.sch.step_next();
    REQUIRE(nb != nullptr);
    CHECK_EQ(nb->n_spec, 4);
    CHECK_EQ(nb->n_tok, 2 + 5);

    /* Its commit, the watermark, and the proposals published for the step already on the card:
     * four rounds' worth to both, and each keeps what its step was planned to verify. */
    commit_ab();
    f.sch.serial_end(rounds);
    CHECK_OK(f.sch.set_draft(a, draft, rounds));
    CHECK_OK(f.sch.set_draft(b, draft, rounds));
    CHECK_EQ(f.sch.find(a)->n_draft, 1);
    CHECK_EQ(f.sch.find(b)->n_draft, 4);
    f.sch.promote_next();
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, e.seq == a ? 1 : 4);
    CHECK(f.sch.metrics().draft_verify_depth[1] > 0);
    CHECK(f.sch.metrics().draft_verify_depth[4] > 0);
}

/* A new sequence starts from the finished sequences, one vote each. Pooled verifies measure deep
 * positions almost only on texts good enough to reach them; here two texts the head predicts and
 * two it does not set the prior together. Until four have finished, the fixed prior applies. */
TEST(the_adaptive_window_starts_a_sequence_from_finished_sequences_not_pooled_verifies) {
    DraftController c = adaptive_controller(6);
    /* The pooled counts a deep-drafting good text leaves: position 5 accepted 95 % of the times
     * it was reached. They do not set the prior. */
    for (int t = 0; t < 1000; ++t) c.observe(6, 6);
    CHECK_NEAR(c.prior_at(5), DraftController::kPrior, 1e-9);
    DraftAdapt fresh;
    (void)c.window_for(fresh);
    CHECK_NEAR(fresh.a[5], DraftController::kPrior, 1e-6);

    auto finished = [&](double a0, double rest) {
        DraftAdapt s = estimate(c, a0, rest);
        s.n[0] = 100;
        return s;
    };
    /* Too few verifies to be more than its own prior: not counted. */
    DraftAdapt young = finished(0.1, 0.1);
    young.n[0] = 3;
    c.adapt_retire(young);
    c.adapt_retire(finished(0.95, 0.95));
    c.adapt_retire(finished(0.95, 0.95));
    c.adapt_retire(finished(0.3, 0.2));
    CHECK_NEAR(c.prior_at(5), DraftController::kPrior, 1e-9);   /* three: not yet */
    c.adapt_retire(finished(0.3, 0.2));
    CHECK_NEAR(c.prior_at(0), (0.95 + 0.95 + 0.3 + 0.3) / 4.0, 1e-6);
    CHECK_NEAR(c.prior_at(5), (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);
    DraftAdapt next;
    (void)c.window_for(next);
    CHECK_NEAR(next.a[5], (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);
    /* An unstarted estimate is not a sequence to count. */
    c.adapt_retire(DraftAdapt{});
    CHECK_NEAR(c.prior_at(5), (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);

    /* And through the scheduler: four requests whose head drafts are all refused finish, and the
     * fifth starts at one draft where the first four started at the whole window. */
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    std::vector<uint64_t> ids;
    for (int i = 0; i < 4; ++i) ids.push_back(f.submit(160, 48));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 0.0, 1e-6);     /* none drafted yet */
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, draft, 4));
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 4.0, 1e-6);
    for (int t = 0; t < 200 && f.sch.metrics().finished < 4; ++t) {
        for (uint64_t id : ids)
            if (f.sch.find(id)) (void)f.sch.set_draft(id, draft, 4);
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
    }
    CHECK_EQ(f.sch.metrics().finished, (int64_t)4);
    CHECK(f.sch.draft().prior_at(0) < 0.2);
    /* The step depth came down with them; a fresh sequence starting at the fixed prior would take
     * it back to four at its first cycle (the hold has long passed), and this one does not. */
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    const uint64_t e = f.submit(160, 48);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(e, draft, 4));
    REQUIRE(f.sch.step() != nullptr);                  /* its first cycle chooses D */
    f.commit_plain(11);
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    CHECK_OK(f.sch.set_draft(e, draft, 4));
    CHECK_EQ(f.sch.find(e)->n_draft, 1);
}

/* Prompt lookup is not the head. An n-gram fallback draft counts in the reported acceptance but
 * stays out of the sequence's estimate, which prices the head's next window: refused lookups leave
 * the window alone, refused head drafts bring it down. */
TEST(the_adaptive_window_is_not_fed_by_prompt_lookup) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    /* A prompt that repeats, so the lookup always has a continuation to propose. */
    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 2000;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    const uint64_t a = r->id;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);                               /* the pattern goes on */

    int lookups = 0;
    for (int t = 0; t < 4 * DraftController::kHold; ++t) {
        REQUIRE(f.sch.step() != nullptr);              /* nothing published: the lookup fills */
        const StepEntry& e = f.sch.plan().e[0];
        if (e.n_spec > 0) {
            CHECK(e.ngram);
            CHECK(e.n_spec <= f.sch.draft().step_window());
            ++lookups;
        }
        f.commit_plain(100 + (t + 1) % 8);             /* every lookup refused */
    }
    CHECK(lookups >= 2 * DraftController::kHold);
    const SchedMetrics m = f.sch.metrics();
    CHECK(m.draft_tokens > 0);                         /* reported, as ever */
    CHECK(m.draft_pos_reached[0] >= lookups);
    CHECK_EQ(m.draft_accepted, (int64_t)0);
    CHECK_NEAR(m.draft_depth_adaptive, 4.0, 1e-6);     /* the head's window did not move */

    /* The head's own refusals are what move it. */
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int t = 0; t < 4 * DraftController::kHold; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK(!f.sch.plan().e[0].ngram);
        f.commit_plain(100 + (t + 1) % 8);
    }
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 1.0, 1e-6);
}

/* With prompt lookup the only drafter, its drafts are what the adaptive window measures: refused
 * lookups bring each sequence's window, and the lookup's proposals, down to one draft. */
/* ------------------------------------------------------------------ depth 0 (draft.h)
 * The adaptive policy may also choose D = 0 -- no head pass, no verify row -- where a measured plain
 * step emits more tokens a millisecond than the window. Never on the prior: a plain step is measured
 * first, by a plain probe once the window is measured. At 0 a probe at the drafting depth every
 * probe_every cycles keeps the estimates measured, and a probe that finds the window paying again
 * leaves 0. */
namespace {
DraftController depth0_controller(int max_depth, int probe_every = 0) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = max_depth;
    p.adapt = true;
    p.depth0 = true;
    p.probe_every = probe_every;
    CHECK_OK(c.configure(p));
    return c;
}
}  /* namespace */

TEST(depth_zero_is_taken_only_where_a_measured_plain_step_beats_the_window) {
    /* Four sequences at 0.6 a position stop at 4 drafts (0.6^5 < 0.1). */
    DraftController u0 = depth0_controller(5);
    CHECK(u0.depth0());
    CHECK_EQ(u0.probe_every(), DraftController::kProbeEvery);
    DraftAdapt w[4];
    DraftAdapt* wf[4];
    for (int i = 0; i < 4; ++i) { w[i] = estimate(u0, 0.6, 0.6); wf[i] = &w[i]; }
    CHECK_EQ(w[0].depth, 4);
    /* Nothing measured: never 0, and no plain probe before the window has a price here. */
    for (int t = 0; t < 3; ++t) {
        CHECK(u0.plan_step(wf, 4, 4) > 0);
        CHECK(u0.head_runs());
    }
    CHECK_EQ(u0.probe_cycles(), (int64_t)0);

    /* Their window's steps measured at 20 + 10 d ms at four rows: d = 1 pays best, 4 x 1.6 tokens
     * in 30 ms (0.213 a ms). The plain step not measured: a plain probe, 1 + kPlainWarm + 3
     * cycles the first time, in which the head does not run and nothing verifies; its first
     * kPlainWarm plain steps are no samples. */
    DraftController c = depth0_controller(5);
    DraftAdapt s[4];
    DraftAdapt* four[4];
    for (int i = 0; i < 4; ++i) { s[i] = estimate(c, 0.6, 0.6); four[i] = &s[i]; }
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) c.step_time(d, 4, 20.0 + 10.0 * d);
    CHECK_EQ(c.plan_step(four, 4, 4), 0);
    CHECK(!c.head_runs());
    CHECK_EQ(c.step_window(), 0);
    CHECK_EQ(c.window_for(s[0]), 0);
    const int P0 = 1 + DraftController::kPlainWarm + 3;
    for (int t = 1; t < P0; ++t) {
        CHECK_EQ(c.plan_step(four, 4, 4), 0);
        c.step_time(0, 4, 28.0);                     /* plain: 4 tokens in 28 ms, 0.143 a ms */
    }
    CHECK_EQ(c.cost_samples(0, 4), (int64_t)3);
    CHECK_EQ(c.probe_cycles(), (int64_t)P0);
    /* 0.143 a ms against the window's 0.213: the window stays. */
    CHECK_EQ(c.plan_step(four, 4, 4), 1);
    CHECK(c.head_runs());
    CHECK_EQ(c.window_for(s[0]), 1);
    CHECK_EQ(c.zero_moves(), (int64_t)0);

    /* The same with a plain step of 15 ms (0.267 a ms): 0, at once (the hold starts spent). */
    DraftController z = depth0_controller(5);
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    for (int i = 0; i < 4; ++i) { u[i] = estimate(z, 0.6, 0.6); zf[i] = &u[i]; }
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) z.step_time(d, 4, 20.0 + 10.0 * d);
    for (int t = 0; t < P0; ++t) {
        CHECK_EQ(z.plan_step(zf, 4, 4), 0);
        if (t > 0) z.step_time(0, 4, 15.0);
    }
    CHECK_EQ(z.cost_samples(0, 4), (int64_t)3);
    CHECK_EQ(z.plan_step(zf, 4, 4), 0);
    CHECK_EQ(z.zero_moves(), (int64_t)1);
    CHECK(!z.head_runs());
    CHECK_EQ(z.step_depth(), 0);
    CHECK_EQ(z.window_for(u[0]), 0);
    /* At 0: kProbeEvery cycles in all, then kProbeSteps at the drafting depth (the head runs, each
     * sequence verifies its window), then 0 again while the plain step still pays. */
    const int64_t probes = z.probe_cycles();
    for (int t = 2; t < DraftController::kProbeEvery; ++t) {
        CHECK_EQ(z.plan_step(zf, 4, 4), 0);
        z.step_time(0, 4, 15.0);
    }
    for (int t = 0; t < DraftController::kProbeSteps; ++t) {
        CHECK_EQ(z.plan_step(zf, 4, 4), 1);
        CHECK(z.head_runs());
        CHECK_EQ(z.window_for(u[0]), 1);
    }
    CHECK_EQ(z.probe_cycles(), probes + DraftController::kProbeSteps);
    CHECK_EQ(z.plan_step(zf, 4, 4), 0);
    CHECK(z.zero_cycles() >= (int64_t)DraftController::kProbeEvery);
    /* A probe whose verifies find the text predictable again (every sequence at 0.98 a position,
     * the window 5 deep: 4 x 5.7 tokens in 70 ms, 0.33 a ms) leaves 0 when it ends. */
    for (int t = 0; t < DraftController::kProbeEvery && !z.head_runs(); ++t)
        (void)z.plan_step(zf, 4, 4);
    REQUIRE(z.head_runs());
    for (int i = 0; i < 4; ++i) u[i] = estimate(z, 0.98, 0.98);
    int d = 0;
    for (int t = 0; t < DraftController::kProbeSteps; ++t) d = z.plan_step(zf, 4, 4);
    CHECK(d > 1);
    CHECK(z.head_runs());
    CHECK_EQ(z.zero_moves(), (int64_t)2);
    CHECK(z.plan_step(zf, 4, 4) > 0);
    /* Off between steps: the drafting depth at once, and a plain step is no sample. */
    z.set_depth0(false);
    CHECK(z.head_runs());
    CHECK(z.plan_step(zf, 4, 4) > 0);
    const int64_t n0 = z.cost_samples(0, 4);
    z.step_time(0, 4, 15.0);
    CHECK_EQ(z.cost_samples(0, 4), n0);
    /* A probe interval of its own. */
    DraftController p = depth0_controller(5, 4);
    CHECK_EQ(p.probe_every(), 4);
    p.set_probe_every(0);
    CHECK_EQ(p.probe_every(), DraftController::kProbeEvery);
    /* Without the adaptive policy depth 0 is inert. */
    DraftController f;
    DraftController::Params fp;
    fp.max_depth = 5;
    fp.depth0 = true;
    CHECK_OK(f.configure(fp));
    CHECK(f.head_runs());
    f.step_time(0, 4, 15.0);
    CHECK_EQ(f.cost_samples(0, 4), (int64_t)0);
}

/* The situations gpu02 measured (hc-depth0, 10.11.2026; DeepSeek-V4.1 EXL3 2.25 TP4, DSpark NST 5 adaptive,
 * gg_gemv 2), replayed on the controller with synthetic step times: each cycle plans, then is fed
 * what its step would measure -- a drafting step that verifies (the head ran this cycle and the one
 * before) its time and its tokens; a plain step (the head ran in neither) its time, the first
 * kPlainWarm of each run 2.5x slower as a pass shape's first sightings run; nothing for the
 * transition steps. The estimates are held at what the policy starts from, more optimistic than
 * the measured tokens, as a new sequence's prior is.
 *   c1 essay  drafting 1.45 tokens in 26 ms, plain 16.5 ms: 0 pays 9 %
 *   c1 code   drafting 3.10 tokens in 32 ms: drafting pays 60 %, and stays
 *   c4        drafting 10.3 tokens in 79 ms, plain 27 ms: 0 pays 14 %
 *   c8        drafting 19 tokens in 117 ms, plain 40 ms: 0 pays 23 %
 * Before the fix the first plain probe's mean held two recording steps and the comparison took the
 * estimates' yield: depth 0 never fired at c1 essay or c4. */
namespace {
double zero_share(int rows, double a0, double rest, double draft_ms, double draft_tok,
                  double plain_ms, int cycles, int tail, int64_t* moves = nullptr) {
    DraftController c = depth0_controller(5);
    std::vector<DraftAdapt> s((size_t)rows);
    std::vector<DraftAdapt*> sp((size_t)rows);
    for (int i = 0; i < rows; ++i) { s[(size_t)i] = estimate(c, a0, rest); sp[(size_t)i] = &s[(size_t)i]; }
    bool prev = true;
    int plain_run = 0, at0 = 0;
    for (int t = 0; t < cycles; ++t) {
        const int d = c.plan_step(sp.data(), rows, rows);
        const bool head = c.head_runs();
        if (head && prev) c.step_time(d, rows, draft_ms, draft_tok);
        if (!head && !prev) {
            ++plain_run;
            c.step_time(0, rows, plain_run <= DraftController::kPlainWarm ? 2.5 * plain_ms : plain_ms);
        }
        if (head) plain_run = 0;
        if (!head && t >= cycles - tail) ++at0;
        prev = head;
    }
    if (moves) *moves = c.zero_moves();
    return (double)at0 / (double)tail;
}
}  /* namespace */

TEST(depth_zero_fires_where_the_measured_plain_step_pays_and_not_where_drafting_does) {
    int64_t mv = 0;
    const double essay = zero_share(1, 0.5, 0.45, 26.0, 1.45, 16.5, 600, 300, &mv);
    CHECK(essay >= 0.8);
    CHECK(mv <= 2);
    const double code = zero_share(1, 0.75, 0.7, 32.0, 3.10, 16.5, 600, 300, &mv);
    CHECK_NEAR(code, 0.0, 0.03);                     /* plain probes only */
    const double c4 = zero_share(4, 0.5, 0.45, 79.0, 10.3, 27.0, 600, 300, &mv);
    CHECK(c4 >= 0.8);
    CHECK(mv <= 2);
    const double c8 = zero_share(8, 0.5, 0.45, 117.0, 19.0, 40.0, 600, 300, &mv);
    CHECK(c8 >= 0.8);
    CHECK(mv <= 2);
    fprintf(stderr, "    share of cycles at 0: c1 essay %.2f, c1 code %.2f, c4 %.2f, c8 %.2f\n", essay, code, c4,
            c8);
}

/* Through the scheduler: at D = 0 a block drafter's rows are not asked to draft (the context pass
 * runs alone), a published proposal is cut to nothing, and the step verifies nothing. */
TEST(depth_zero_leaves_a_block_drafter_its_context_pass_and_verifies_nothing) {
    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());
    CHECK_EQ(f.sch.set_draft_depth0(1), 1);
    CHECK_EQ(f.sch.set_draft_probe(1), DraftController::kProbeEvery);   /* below 2: unchanged */
    CHECK_EQ(f.sch.set_draft_probe(64), 64);
    const uint64_t a = f.submit(160, 2000);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_EQ(f.sch.find(a)->n_draft, 4);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
    /* Measured: one-row windows cost 40 ms and more, a plain step 5 ms. Plain probes, then 0. */
    DraftController& dc = f.sch.draft();
    for (int d = 1; d <= 4; ++d)
        for (int k = 0; k < 4; ++k) dc.step_time(d, 1, 40.0 + d);
    f.commit_plain(12);
    /* What step_finished would feed: a plain step wherever the head did not run. */
    bool zero = false;
    for (int t = 0; t < 4 * DraftController::kHold && !zero; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        if (!f.sch.draft().head_runs()) dc.step_time(0, 1, 5.0);
        zero = f.sch.draft().zero_moves() > 0 && !f.sch.draft().head_runs();
        f.commit_plain(13);
    }
    REQUIRE(zero);
    /* The proposal a drafter published before the move is cut to nothing; the step verifies none. */
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
    REQUIRE(f.sch.step() != nullptr);
    if (!f.sch.draft().head_runs()) {
        CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
        Scheduler::BlockRow rows[4];
        CHECK_EQ(f.sch.block_begin(5, 4, 0, 0, rows, 4, /*on_card=*/false), 0);
        f.sch.block_end();
    }
    f.commit_plain(14);
    CHECK_EQ(f.sch.set_draft_depth0(0), 0);
    CHECK(f.sch.draft().head_runs());
}

/* ------------------------------------------------------------------ probe back-off and the lend (draft.h)
 * Probe back-off doubles the interval at 0 after each probe that leaves 0 in place, up to probe_every << N,
 * and a move starts it over. The drafter's lend (lever DL) is wanted once depth 0 has held lend_hold() cycles
 * with probes at least kLendInterval apart, and never off 0. While lent, a drafting step is no sample and a
 * plain step goes to the lent table, whose ratio to the unlent plain step prices staying at 0. */
namespace {
/* A controller settled at 0: four rows, the window's steps measured at 20 + 10 d ms, the plain step at 15 ms
 * (as depth_zero_is_taken_only_where_a_measured_plain_step_beats_the_window). */
DraftController zero_controller(int backoff, int lend, DraftAdapt* u, DraftAdapt** zf) {
    DraftController z;
    DraftController::Params p;
    p.max_depth = 5;
    p.adapt = true;
    p.depth0 = true;
    p.backoff = backoff;
    p.lend = lend;
    CHECK_OK(z.configure(p));
    for (int i = 0; i < 4; ++i) { u[i] = estimate(z, 0.6, 0.6); zf[i] = &u[i]; }
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) z.step_time(d, 4, 20.0 + 10.0 * d);
    const int P0 = 1 + DraftController::kPlainWarm + 3;
    for (int t = 0; t < P0; ++t) {
        CHECK_EQ(z.plan_step(zf, 4, 4), 0);
        if (t > 0) z.step_time(0, 4, 15.0);
    }
    CHECK_EQ(z.plan_step(zf, 4, 4), 0);
    CHECK_EQ(z.zero_moves(), (int64_t)1);
    return z;
}
/* Cycles between the starts of consecutive probes at the drafting depth over `n` cycles, each plain cycle
 * measured at 15 ms (as `lend` says it ran); `wanted`, if set, gets lend_wanted() at every cycle. */
std::vector<int> probe_gaps(DraftController& z, DraftAdapt** zf, int n, std::vector<int>* wanted = nullptr,
                            int lend = DraftController::kLendResident) {
    std::vector<int> starts;
    bool prev = false;
    for (int t = 0; t < n; ++t) {
        (void)z.plan_step(zf, 4, 4);
        const bool head = z.head_runs();
        if (head && !prev) starts.push_back(t);
        if (!head) z.step_time(0, 4, 15.0, -1.0, lend);
        if (wanted) wanted->push_back(z.lend_wanted() ? 1 : 0);
        prev = head;
    }
    std::vector<int> g;
    for (size_t i = 1; i < starts.size(); ++i) g.push_back(starts[i] - starts[i - 1]);
    return g;
}
}  /* namespace */

TEST(probe_back_off_doubles_the_interval_while_depth_zero_keeps_winning) {
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    /* Off: a probe every kProbeEvery cycles at 0, its two cycles at the drafting depth included. */
    DraftController a = zero_controller(0, 0, u, zf);
    CHECK_EQ(a.backoff(), 0);
    for (int g : probe_gaps(a, zf, 200)) CHECK_EQ(g, DraftController::kProbeEvery + 1);
    CHECK_EQ(a.probe_interval(), DraftController::kProbeEvery);

    /* Three doublings: 16 -> 32 -> 64 -> 128, then 128. */
    DraftController b = zero_controller(3, 0, u, zf);
    CHECK_EQ(b.backoff(), 3);
    const std::vector<int> gb = probe_gaps(b, zf, 700);
    REQUIRE(gb.size() >= 4);
    CHECK_EQ(gb[0], 2 * DraftController::kProbeEvery + 1);
    CHECK_EQ(gb[1], 4 * DraftController::kProbeEvery + 1);
    CHECK_EQ(gb[2], 8 * DraftController::kProbeEvery + 1);
    CHECK_EQ(gb[3], 8 * DraftController::kProbeEvery + 1);
    CHECK_EQ(b.backoff_k(), 3);
    CHECK_EQ(b.probe_interval(), 8 * DraftController::kProbeEvery);
    CHECK_EQ(b.zero_moves(), (int64_t)1);

    /* A probe that finds the window paying again (0.98 a position) leaves 0 and starts the back-off over. */
    for (int t = 0; t < 300 && !b.head_runs(); ++t) (void)b.plan_step(zf, 4, 4);
    REQUIRE(b.head_runs());
    for (int i = 0; i < 4; ++i) u[i] = estimate(b, 0.98, 0.98);
    for (int t = 0; t < DraftController::kProbeSteps; ++t) (void)b.plan_step(zf, 4, 4);
    CHECK(b.head_runs());
    CHECK_EQ(b.zero_moves(), (int64_t)2);
    CHECK_EQ(b.backoff_k(), 0);
    CHECK_EQ(b.probe_interval(), DraftController::kProbeEvery);

    /* The switch: clamped, and a change starts over. */
    b.set_backoff(99);
    CHECK_EQ(b.backoff(), DraftController::kBackoffMax);
    b.set_backoff(-1);
    CHECK_EQ(b.backoff(), 0);
    /* The interval never exceeds what draft_probe takes. */
    DraftController c = zero_controller(DraftController::kBackoffMax, 0, u, zf);
    c.set_probe_every(4096);
    (void)probe_gaps(c, zf, 50);
    CHECK(c.probe_interval() <= 4096);
}

TEST(the_drafters_lend_is_wanted_only_at_a_settled_depth_zero_with_probes_far_apart) {
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    /* On, but probes every 16 cycles: a probe while lent costs more than 16 lent cycles earn. Never. */
    DraftController a = zero_controller(0, 1, u, zf);
    CHECK_EQ(a.lend(), 1);
    CHECK_EQ(a.lend_hold(), DraftController::kLendHold);
    std::vector<int> wa;
    (void)probe_gaps(a, zf, 300, &wa);
    for (int w : wa) CHECK_EQ(w, 0);

    /* With the back-off: wanted from the cycle the interval reaches kLendInterval (after the second
     * probe), the hold long spent; and through every probe after it, which does not move 0. */
    DraftController b = zero_controller(3, 1, u, zf);
    std::vector<int> wb;
    const std::vector<int> gb = probe_gaps(b, zf, 400, &wb);
    REQUIRE(gb.size() >= 3);
    int first = -1;
    for (size_t t = 0; t < wb.size(); ++t)
        if (wb[t]) { first = (int)t; break; }
    REQUIRE(first >= 0);
    for (size_t t = (size_t)first; t < wb.size(); ++t) CHECK_EQ(wb[t], 1);
    /* The first two probes come 16 and 16 + 33 cycles after the move; the lend after the second. */
    CHECK(first >= DraftController::kProbeEvery + 2 * DraftController::kProbeEvery + 1);
    CHECK(first <= DraftController::kProbeEvery + 2 * DraftController::kProbeEvery + 1 + DraftController::kProbeSteps + 1);
    CHECK(b.probe_interval() >= DraftController::kLendInterval);

    /* A hold of its own (2..4096 cycles). */
    DraftController h = zero_controller(3, 500, u, zf);
    CHECK_EQ(h.lend_hold(), 500);
    std::vector<int> wh;
    (void)probe_gaps(h, zf, 400, &wh);
    for (int w : wh) CHECK_EQ(w, 0);

    /* Off: never, whatever the back-off. */
    DraftController o = zero_controller(3, 0, u, zf);
    std::vector<int> wo;
    (void)probe_gaps(o, zf, 400, &wo);
    for (int w : wo) CHECK_EQ(w, 0);

    /* Off 0, not wanted: a probe that finds the window paying, depth 0 switched off, or the switch off. */
    for (int t = 0; t < 300 && !b.head_runs(); ++t) (void)b.plan_step(zf, 4, 4);
    CHECK(b.lend_wanted());
    for (int i = 0; i < 4; ++i) u[i] = estimate(b, 0.98, 0.98);
    for (int t = 0; t < DraftController::kProbeSteps; ++t) (void)b.plan_step(zf, 4, 4);
    CHECK(b.head_runs());
    CHECK(!b.lend_wanted());
    DraftController c = zero_controller(3, 1, u, zf);
    for (int i = 0; i < 4; ++i) u[i] = estimate(c, 0.6, 0.6);
    (void)probe_gaps(c, zf, 200);
    CHECK(c.lend_wanted());
    c.set_depth0(false);
    CHECK(!c.lend_wanted());
    DraftController d = zero_controller(3, 1, u, zf);
    (void)probe_gaps(d, zf, 200);
    CHECK(d.lend_wanted());
    d.set_lend(0);
    CHECK(!d.lend_wanted());
    /* -1: lent at every step, drafting or not (an identity check of the lent placement). */
    d.set_depth0(false);
    CHECK(d.head_runs());
    d.set_lend(-1);
    CHECK_EQ(d.lend(), -1);
    CHECK_EQ(d.lend_hold(), 0);
    CHECK(d.lend_wanted());
    d.set_lend(-7);
    CHECK_EQ(d.lend(), 0);
    CHECK(!d.lend_wanted());
}

TEST(a_lent_step_feeds_its_own_plain_table_and_no_drafting_sample) {
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    DraftController z = zero_controller(3, 1, u, zf);
    for (int t = 0; t < 8; ++t) {                    /* past the plain warm-up at 0 */
        (void)z.plan_step(zf, 4, 4);
        if (!z.head_runs()) z.step_time(0, 4, 15.0);
    }
    const int64_t n0 = z.cost_samples(0, 4), n3 = z.cost_samples(3, 4);
    REQUIRE(n0 >= 3);
    CHECK_NEAR(z.plain_ms(4), 15.0, 1e-9);
    CHECK_EQ(z.lent_samples(4), (int64_t)0);
    /* A transition: nothing. */
    z.step_time(0, 4, 40.0, -1.0, DraftController::kLendTransit);
    z.step_time(3, 4, 99.0, 12.0, DraftController::kLendTransit);
    CHECK_EQ(z.cost_samples(0, 4), n0);
    CHECK_EQ(z.cost_samples(3, 4), n3);
    CHECK_EQ(z.lent_samples(4), (int64_t)0);
    /* Lent: a drafting step (a probe reading the drafter over the link) is none; a plain one goes to
     * the lent table, and the plain step is priced at its ratio to the unlent one (12 / 15). */
    z.step_time(3, 4, 99.0, 12.0, DraftController::kLendLent);
    CHECK_EQ(z.cost_samples(3, 4), n3);
    for (int k = 0; k < 4; ++k) z.step_time(0, 4, 12.0, -1.0, DraftController::kLendLent);
    CHECK_EQ(z.cost_samples(0, 4), n0);
    CHECK_EQ(z.lent_samples(4), (int64_t)4);
    CHECK_NEAR(z.plain_ms(4), 12.0, 1e-6);
    /* Another class has its own. */
    CHECK_EQ(z.lent_samples(1), (int64_t)0);
    /* One stalled step does not price the lend: the ratio is held to [0.5, 1.25]. */
    DraftController y = z;
    for (int k = 0; k < 200; ++k) y.step_time(0, 4, 100.0, -1.0, DraftController::kLendLent);
    CHECK(y.plain_ms(4) <= 15.0 * DraftController::kLentRatioMax + 1e-6);
    /* The switch off: the unlent plain step again. */
    z.set_lend(0);
    CHECK_NEAR(z.plain_ms(4), 15.0, 1e-9);
    z.set_lend(1);
    CHECK_NEAR(z.plain_ms(4), 12.0, 1e-6);
}

TEST(staying_at_zero_is_priced_at_the_lent_plain_step) {
    /* At 0 with the unlent plain step at 15 ms and the lent one measured at 12. The text turns more
     * predictable (0.9 a position): the window's best is 4 x 3.44 tokens in 50 ms (0.275 a ms), past the
     * unlent plain step's 0.267 by more than kHyst and short of the lent one's 0.333. Lent, 0 stays; with
     * the lend off the same history leaves 0 at the next probe. */
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    DraftController z = zero_controller(3, 1, u, zf);
    for (int t = 0; t < 8; ++t) {
        (void)z.plan_step(zf, 4, 4);
        if (!z.head_runs()) z.step_time(0, 4, 15.0);
    }
    for (int k = 0; k < 8; ++k) z.step_time(0, 4, 12.0, -1.0, DraftController::kLendLent);
    DraftController off = z;
    off.set_lend(0);
    for (int i = 0; i < 4; ++i) u[i] = estimate(z, 0.9, 0.9);
    const int64_t mz = z.zero_moves(), mo = off.zero_moves();
    for (int t = 0; t < 600; ++t) {
        (void)z.plan_step(zf, 4, 4);
        (void)off.plan_step(zf, 4, 4);
    }
    CHECK_EQ(z.zero_moves(), mz);
    CHECK(!z.head_runs() || z.probe_cycles() > 0);
    CHECK(off.zero_moves() > mo);
}

/* Lever Z0-H, the lend's hysteresis (draft_lend_hyst): at a lent 0 (unlent plain step 15 ms, lent 14), the text turns
 * predictable enough (0.98 a position) that the best window yields 4 x 5.71 tokens in 70 ms (0.326 a ms), 14 % past the
 * lent plain step's 0.286. With kHyst alone (the switch 0) 0 is left; at 10 % it is left too; at 20 % it holds, and
 * the holds are counted; with the lend off the same 20 % does not act (the unlent plain step, kHyst). A move back to 0
 * keeps kHyst (draft.cpp: the drafting side of the decision is unchanged). */
TEST(the_lend_hysteresis_holds_a_lent_zero_against_a_window_that_gains_less_than_it) {
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    DraftController z = zero_controller(3, 1, u, zf);
    for (int t = 0; t < 8; ++t) {
        (void)z.plan_step(zf, 4, 4);
        if (!z.head_runs()) z.step_time(0, 4, 15.0);
    }
    for (int k = 0; k < 8; ++k) z.step_time(0, 4, 14.0, -1.0, DraftController::kLendLent);
    CHECK_NEAR(z.plain_ms(4), 14.0, 1e-6);
    CHECK_EQ(z.lend_hyst(), 0);
    DraftController h10 = z, h20 = z, off = z;
    h10.set_lend_hyst(10);
    h20.set_lend_hyst(20);
    CHECK_EQ(h20.lend_hyst(), 20);
    h20.set_lend_hyst(101);
    CHECK_EQ(h20.lend_hyst(), 100);   /* clamped (fn4 refuses 101) */
    h20.set_lend_hyst(20);
    off.set_lend(0);
    off.set_lend_hyst(20);
    for (int i = 0; i < 4; ++i) u[i] = estimate(z, 0.98, 0.98);
    const int64_t mz = z.zero_moves(), m10 = h10.zero_moves(), m20 = h20.zero_moves(), mo = off.zero_moves();
    for (int t = 0; t < 600; ++t) {
        (void)z.plan_step(zf, 4, 4);
        (void)h10.plan_step(zf, 4, 4);
        (void)h20.plan_step(zf, 4, 4);
        (void)off.plan_step(zf, 4, 4);
    }
    CHECK(z.zero_moves() > mz);
    CHECK(h10.zero_moves() > m10);
    CHECK_EQ(h20.zero_moves(), m20);
    CHECK(h20.lend_hyst_holds() > 0);
    CHECK_EQ(z.lend_hyst_holds(), (int64_t)0);
    CHECK(off.zero_moves() > mo);
    CHECK_EQ(off.lend_hyst_holds(), (int64_t)0);
}

/* Lever Z0-O, depth 0 by sustained occupancy (draft_occ / draft_occ_k): from a controller settled at 0 over four rows,
 * the text turns predictable enough (0.98 a position) that D7 leaves 0 and drafts. With draft_occ 3 and draft_occ_k 8,
 * four rows for 8 cycles hold 0 from then on with no probe, whatever the estimates; a dip to two rows shorter than 8
 * cycles keeps it; 8 cycles at two rows hand the choice back to D7. With draft_occ 0 every cycle is D7's own. Under the
 * drafter's lend, occupancy's 0 (no probe) is a 0 the lend may take, where D7's probes every 16 cycles keep it off. */
TEST(depth_zero_by_sustained_occupancy_holds_without_probes_and_hands_back_below_its_rows) {
    DraftAdapt u[4];
    DraftAdapt* zf[4];
    DraftController z = zero_controller(0, 0, u, zf);
    CHECK_EQ(z.occ(), 0);
    CHECK_EQ(z.occ_k(), DraftController::kOccK);
    DraftController d7 = z, off = z, occ = z;
    off.set_occ_k(8);
    occ.set_occ(65);
    CHECK_EQ(occ.occ(), 64);   /* clamped (fn4 refuses 65) */
    occ.set_occ(3);
    occ.set_occ_k(8);
    CHECK_EQ(occ.occ(), 3);
    CHECK_EQ(occ.occ_k(), 8);
    for (int i = 0; i < 4; ++i) u[i] = estimate(z, 0.98, 0.98);
    int head_d7 = 0, head_off = 0, head_occ_late = 0;
    int64_t probes_at_hold = -1;
    for (int t = 0; t < 600; ++t) {
        (void)d7.plan_step(zf, 4, 4);
        (void)off.plan_step(zf, 4, 4);
        (void)occ.plan_step(zf, 4, 4);
        head_d7 += d7.head_runs() ? 1 : 0;
        head_off += off.head_runs() ? 1 : 0;
        if (t >= 8) head_occ_late += occ.head_runs() ? 1 : 0;
        if (t == 8) probes_at_hold = occ.probe_cycles();
    }
    CHECK(head_d7 > 300);                                 /* D7 drafts */
    CHECK_EQ(head_off, head_d7);                          /* draft_occ 0: D7's own choice, cycle for cycle */
    CHECK_EQ(off.zero_moves(), d7.zero_moves());
    CHECK_EQ(off.probe_cycles(), d7.probe_cycles());
    CHECK_EQ(off.occ_cycles(), (int64_t)0);
    CHECK_EQ(head_occ_late, 0);                           /* held at 0 from the 8th cycle at four rows */
    CHECK(occ.occ_held());
    CHECK_EQ(occ.occ_moves(), (int64_t)1);
    CHECK(occ.occ_cycles() >= 592);
    CHECK_EQ(occ.probe_cycles(), probes_at_hold);         /* no probe while held */
    CHECK_EQ(occ.step_depth(), 0);
    /* A dip to two rows for 7 cycles: still held. */
    for (int t = 0; t < 7; ++t) { (void)occ.plan_step(zf, 2, 2); CHECK(!occ.head_runs()); }
    for (int t = 0; t < 3; ++t) (void)occ.plan_step(zf, 4, 4);
    CHECK(occ.occ_held());
    CHECK_EQ(occ.occ_moves(), (int64_t)1);
    /* 8 cycles at two rows: D7 decides again (probes, drafting). */
    for (int t = 0; t < 8; ++t) (void)occ.plan_step(zf, 2, 2);
    CHECK(!occ.occ_held());
    CHECK_EQ(occ.occ_moves(), (int64_t)2);
    const int64_t held = occ.occ_cycles();
    int head_after = 0;
    for (int t = 0; t < 200; ++t) { (void)occ.plan_step(zf, 2, 2); head_after += occ.head_runs() ? 1 : 0; }
    CHECK(head_after > 0);
    CHECK_EQ(occ.occ_cycles(), held);
    /* Off between steps: D7 at once. */
    for (int t = 0; t < 8; ++t) (void)occ.plan_step(zf, 4, 4);
    CHECK(occ.occ_held());
    occ.set_occ(0);
    CHECK(!occ.occ_held());

    /* The lend: estimates that keep D7 at 0 (0.6), probes every 16 cycles -- under kLendInterval, so D7's 0 is never
     * lent; occupancy's is, once lend_hold() cycles have passed. */
    DraftAdapt v[4];
    DraftAdapt* vf[4];
    DraftController l7 = zero_controller(0, 1, v, vf);
    DraftController lo = l7;
    lo.set_occ(3);
    lo.set_occ_k(8);
    int want7 = 0, wanto = 0;
    for (int t = 0; t < 200; ++t) {
        (void)l7.plan_step(vf, 4, 4);
        (void)lo.plan_step(vf, 4, 4);
        if (!l7.head_runs()) l7.step_time(0, 4, 15.0);
        if (!lo.head_runs()) lo.step_time(0, 4, 15.0);
        want7 += l7.lend_wanted() ? 1 : 0;
        wanto += lo.lend_wanted() ? 1 : 0;
    }
    CHECK_EQ(want7, 0);
    CHECK(wanto > 100);
}

/* The borrow (draft_lend_borrow): a cost class with no lent ratio of its own prices its plain step with the
 * nearest class's (kBorrowMin samples at least), a lower class's as it is, a higher one's gain damped kBorrowDamp
 * a class step. Off, pricing is 9461f81's exactly. The case D7 meets at c4: lent at c8 (40 -> 35.72 ms, the
 * d0attr gain of 4.7 ms in 44, ratio 0.893), never at c4, where drafting yields 9.82 tokens in 67.85 ms (0.1447 a
 * ms, depth0b) against an unlent plain step of 29.3 ms (0.1365): unborrowed, 0 is left; borrowed, c4's plain step
 * is priced 29.3 x (1 - 0.107 x 0.75) = 26.95 ms (0.1484), within kHyst of staying, so 0 holds. */
namespace {
/* At 0 at eight rows (plain 40 ms against windows of 100 + 10 d ms), `lent8` lent plain steps of 35.72 ms there,
 * and at four rows an unlent plain step of 29.3 ms and drafting steps of 9.82 tokens in 67.85 ms measured. */
DraftController c4_controller(bool borrow, int lent8, DraftAdapt* u, DraftAdapt** uf) {
    DraftController z;
    DraftController::Params p;
    p.max_depth = 5;
    p.adapt = true;
    p.depth0 = true;
    p.lend = 1;
    p.borrow = borrow;
    CHECK_OK(z.configure(p));
    for (int i = 0; i < 8; ++i) { u[i] = estimate(z, 0.6, 0.6); uf[i] = &u[i]; }
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) z.step_time(d, 8, 100.0 + 10.0 * d);
    for (int t = 0; t < 1 + DraftController::kPlainWarm + 3 + 8; ++t) {
        (void)z.plan_step(uf, 8, 8);
        if (!z.head_runs() && t > 0) z.step_time(0, 8, 40.0);
    }
    CHECK(!z.head_runs());
    CHECK_EQ(z.zero_moves(), (int64_t)1);
    for (int k = 0; k < lent8; ++k) z.step_time(0, 8, 35.72, -1.0, DraftController::kLendLent);
    for (int k = 0; k < 4; ++k) z.step_time(0, 4, 29.3);
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) z.step_time(d, 4, 67.85, 9.82);
    for (int i = 0; i < 4; ++i) u[i] = estimate(z, 0.9, 0.9);
    return z;
}
}  /* namespace */

/* SPB, the depth cap by occupancy (draft_dcap / draft_dcap_n): eight sequences whose estimates (0.9 a position) and step
 * times (a flat 40 ms at every depth) put D7 at depth 5. With draft_dcap 3 (from the default 8 rows) the same cycles choose
 * depth 3 at once -- a controller already at 5 moves down at the first capped cycle, with no hold -- and every capped
 * cycle is counted; under 8 rows the cap does not hold, and the move back up waits D7's hold. draft_dcap 0 is D7's own
 * choice, cycle for cycle. */
TEST(the_depth_cap_holds_the_head_depth_from_its_rows_and_hands_back_below_them) {
    DraftController d7 = adaptive_controller(5);
    CHECK_EQ(d7.dcap(), 0);
    CHECK_EQ(d7.dcap_n(), DraftController::kDcapN);
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    for (int i = 0; i < 8; ++i) { u[i] = estimate(d7, 0.9, 0.9); uf[i] = &u[i]; }
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) {
            d7.step_time(d, 8, 40.0);
            d7.step_time(d, 6, 30.0);
        }
    DraftController off = d7, cap = d7;
    off.set_dcap(0);
    cap.set_dcap(65);
    CHECK_EQ(cap.dcap(), 64);   /* clamped (fn4 refuses 65) */
    cap.set_dcap(3);
    cap.set_dcap_n(0);
    CHECK_EQ(cap.dcap_n(), 1);  /* clamped (fn4 refuses 0) */
    cap.set_dcap_n(8);
    for (int t = 0; t < 20; ++t) {
        CHECK_EQ(d7.plan_step(uf, 8, 8), off.plan_step(uf, 8, 8));
        CHECK_EQ(cap.plan_step(uf, 8, 8), 3);
    }
    CHECK_EQ(d7.step_depth(), 5);
    CHECK_EQ(cap.step_window(), 3);
    CHECK_EQ(cap.dcap_cycles(), (int64_t)20);
    CHECK_EQ(cap.dcap_moves(), (int64_t)0);   /* chose 3 outright: no depth above it was in force */
    CHECK_EQ(off.dcap_cycles(), (int64_t)0);
    for (int i = 0; i < 8; ++i) CHECK_EQ(cap.window_for(u[i]), 3);
    /* Under 8 rows the cap does not hold: D7's choice comes back only past its hold (kHold cycles). */
    int at3 = 0;
    for (int t = 0; t < DraftController::kHold + 4; ++t) at3 += cap.plan_step(uf, 6, 6) == 3 ? 1 : 0;
    CHECK(at3 >= DraftController::kHold - 1);
    CHECK_EQ(cap.step_depth(), 5);
    CHECK_EQ(cap.dcap_cycles(), (int64_t)20);
    /* Back at 8 rows: down to 3 at the first cycle (a forced move), however recently the depth moved. */
    CHECK_EQ(cap.plan_step(uf, 8, 8), 3);
    CHECK_EQ(cap.dcap_moves(), (int64_t)1);
    /* Off between steps: D7's own choice again (back to 5 past its hold). */
    cap.set_dcap(0);
    int at5 = 0;
    for (int t = 0; t < 40; ++t) at5 += cap.plan_step(uf, 8, 8) == 5 ? 1 : 0;
    CHECK(at5 > 0);
    CHECK_EQ(cap.dcap_cycles(), (int64_t)21);
}

/* tp8 loop D20 (loop FINDINGS F56 item 7): draft_dcap_n 100 + N HOLDS the head depth at draft_dcap from N decode rows.
 * Eight sequences at 0.9 / 0.7 a position, step times that make depth 2 the best yield (30 / 31 / 60 / 70 / 80 ms at depths
 * 1-5): the cap (draft_dcap 3 from 8 rows) lets the yield choose 2, the hold verifies 3 every cycle from 8 rows; under them
 * the yield's choice comes back past kHold; 8 again releases the hold. */
TEST(the_depth_hold_sets_the_cap_from_its_rows_where_the_yield_would_choose_less) {
    DraftController a = adaptive_controller(5);
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    for (int i = 0; i < 8; ++i) { u[i] = estimate(a, 0.9, 0.7); uf[i] = &u[i]; }
    const double ms[6] = { 0.0, 30.0, 31.0, 60.0, 70.0, 80.0 };
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 4; ++k) {
            a.step_time(d, 8, ms[d]);
            a.step_time(d, 6, ms[d]);
        }
    DraftController cap = a, hold = a;
    cap.set_dcap(3);
    cap.set_dcap_n(8);
    hold.set_dcap(3);
    hold.set_dcap_n(108);
    CHECK(hold.dcap_hold());
    CHECK_EQ(hold.dcap_n(), 108);   /* reads back as set (fn4 note) */
    CHECK(!cap.dcap_hold());
    for (int t = 0; t < 20; ++t) {
        CHECK_EQ(cap.plan_step(uf, 8, 8), 2);
        CHECK_EQ(hold.plan_step(uf, 8, 8), 3);
    }
    CHECK_EQ(hold.dcap_held_cycles(), (int64_t)20);
    CHECK_EQ(cap.dcap_held_cycles(), (int64_t)0);
    int at2 = 0;
    for (int t = 0; t < DraftController::kHold + 4; ++t) at2 += hold.plan_step(uf, 6, 6) == 2 ? 1 : 0;
    CHECK(at2 > 0);
    CHECK_EQ(hold.plan_step(uf, 8, 8), 3);   /* back at 8 rows: held at once */
    hold.set_dcap_n(8);
    CHECK(!hold.dcap_hold());
    CHECK_EQ(hold.dcap_n(), 8);
    int at2b = 0;
    for (int t = 0; t < DraftController::kHold + 4; ++t) at2b += hold.plan_step(uf, 8, 8) == 2 ? 1 : 0;
    CHECK(at2b > 0);
    hold.set_dcap_n(164);
    CHECK(hold.dcap_hold());
    CHECK_EQ(hold.dcap_n(), 164);
}

/* DCA, a drafting depth's cost key by age (draft_cost_age; lq141 c16w): eight sequences (0.8 a position) at 8 rows under
 * draft_dcap 3. Depth 3 was measured at 90 ms (a slower drafter), then the controller sat at depth 2 (58 ms) and 400
 * decode steps ran elsewhere (16 rows): with the age off depth 2 stays for good (its key is the only one that moves);
 * with draft_cost_age 256 depth 3's old key is stale, depth 2's fresh key by the prior's ratio prices it, and the window
 * goes to 3 past its hold. A key the depth samples again is fresh again; depth 0 is never aged. */
TEST(a_stale_drafting_cost_key_yields_to_a_fresh_one_under_draft_cost_age) {
    DraftController d7 = adaptive_controller(5);
    CHECK_EQ(d7.cost_age(), 0);
    d7.set_dcap(3);
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    for (int i = 0; i < 8; ++i) { u[i] = estimate(d7, 0.8, 0.8); uf[i] = &u[i]; }
    for (int k = 0; k < 40; ++k) d7.step_time(3, 8, 90.0);
    for (int k = 0; k < 40; ++k) d7.step_time(2, 8, 58.0);
    for (int k = 0; k < 400; ++k) d7.step_time(1, 16, 50.0);
    for (int k = 0; k < 4; ++k) d7.step_time(2, 8, 58.0);
    DraftController off = d7, on = d7;
    on.set_cost_age(3);
    CHECK_EQ(on.cost_age(), 16);       /* clamped (fn4 refuses 3) */
    on.set_cost_age(256);
    CHECK_NEAR(off.step_cost(3, 8), 90.0, 1e-6);
    CHECK_NEAR(on.step_cost(3, 8), 58.0 * 1.18 / 1.12, 1e-6);
    CHECK_NEAR(on.step_cost(2, 8), 58.0, 1e-6);
    CHECK_NEAR(on.step_cost(1, 16), 50.0, 1e-6);   /* fresh keys price themselves */
    int off3 = 0, on3 = 0;
    for (int t = 0; t < 3 * DraftController::kHold; ++t) {
        off3 += off.plan_step(uf, 8, 8) == 3 ? 1 : 0;
        on3 += on.plan_step(uf, 8, 8) == 3 ? 1 : 0;
    }
    CHECK_EQ(off3, 0);
    CHECK(on3 > 0);
    CHECK_EQ(on.step_depth(), 3);
    /* Depth 3 sampled again: its key starts over (the old 90 ms is the slower drafter's) and prices it. */
    on.step_time(3, 8, 57.0);
    CHECK_NEAR(on.step_cost(3, 8), 57.0, 1e-6);
    CHECK_EQ(on.cost_samples(3, 8), (int64_t)1);
    off.step_time(3, 8, 57.0);
    CHECK(off.step_cost(3, 8) > 85.0);   /* off: the old mean moves by one sample's weight */
    /* No fresh drafting key in a class: the stale mean, as before. */
    for (int k = 0; k < 300; ++k) on.step_time(1, 16, 50.0);
    CHECK_NEAR(on.step_cost(2, 8), 58.0, 1e-6);
    on.set_cost_age(0);
    CHECK_EQ(on.cost_age(), 0);
}

TEST(the_borrow_prices_a_class_never_lent_with_its_nearest_neighbours_lent_ratio) {
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    const double r8 = 35.72 / 40.0;
    /* Off: c4 is priced at its unlent plain step whatever c8 measured (9461f81's pricing). */
    DraftController off = c4_controller(false, 32, u, uf);
    CHECK(!off.borrow());
    CHECK_NEAR(off.plain_ms(8), 40.0 * r8, 1e-6);
    bool b = true;
    CHECK_NEAR(off.plain_ms(4, &b), 29.3, 1e-9);
    CHECK(!b);
    /* On, below kBorrowMin ratio samples at c8: nothing to borrow. */
    DraftController few = c4_controller(true, (int)DraftController::kBorrowMin - 1, u, uf);
    CHECK_NEAR(few.plain_ms(4, &b), 29.3, 1e-9);
    CHECK(!b);
    /* On, kBorrowMin: c8 is one class above c4, its gain damped once. */
    DraftController on = c4_controller(true, (int)DraftController::kBorrowMin, u, uf);
    CHECK_NEAR(on.plain_ms(4, &b), 29.3 * (1.0 - (1.0 - r8) * DraftController::kBorrowDamp), 1e-6);
    CHECK(b);
    /* Two classes down (two rows: class 1), damped twice -- once it has an unlent plain step to scale. */
    CHECK_EQ(on.cost_samples(0, 2), (int64_t)0);   /* no unlent plain step there: nothing to scale */
    CHECK_NEAR(on.plain_ms(2), 0.0, 1e-12);
    on.step_time(0, 2, 20.0);
    CHECK_NEAR(on.plain_ms(2, &b), 20.0 * (1.0 - (1.0 - r8) * DraftController::kBorrowDamp * DraftController::kBorrowDamp), 1e-6);
    CHECK(b);
    /* A class with a ratio of its own keeps it: c8 itself. */
    CHECK_NEAR(on.plain_ms(8, &b), 40.0 * r8, 1e-6);
    CHECK(!b);
    /* Upward a lower class's ratio is taken as it is, and at equal distance the lower class wins: c4 lent at
     * 0.95 (16 samples), then 16 rows (class 4) -- c8 is nearer (one step, as it is); 2..3 rows (class 1) -- c0
     * has none, c4 at one step above (damped). */
    for (int k = 0; k < 16; ++k) on.step_time(0, 4, 29.3 * 0.95, -1.0, DraftController::kLendLent);
    CHECK_NEAR(on.plain_ms(4, &b), 29.3 * 0.95, 1e-3);
    CHECK(!b);
    on.step_time(0, 16, 60.0);
    CHECK_NEAR(on.plain_ms(16, &b), 60.0 * r8, 1e-6);
    CHECK(b);
    const double r4 = on.plain_ms(4) / 29.3;
    CHECK_NEAR(on.plain_ms(2, &b), 20.0 * (1.0 - (1.0 - r4) * DraftController::kBorrowDamp), 1e-6);
    CHECK(b);
    on.step_time(0, 1, 15.0);
    for (int k = 0; k < 16; ++k) on.step_time(0, 1, 15.0 * 0.97, -1.0, DraftController::kLendLent);
    CHECK_NEAR(on.plain_ms(2, &b), 20.0 * (on.plain_ms(1) / 15.0), 1e-6);   /* c0 below, c2 above: the lower */
    CHECK(b);
    /* Lend off: no borrow either. */
    on.set_lend(0);
    CHECK_NEAR(on.plain_ms(16), 60.0, 1e-9);
}

TEST(the_borrow_lets_depth_zero_hold_at_c4_where_the_lent_c8_ratio_says_it_pays) {
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    DraftController off = c4_controller(false, 32, u, uf);
    DraftController few = c4_controller(true, (int)DraftController::kBorrowMin - 1, u, uf);
    DraftController on = c4_controller(true, 32, u, uf);
    const int64_t m = off.zero_moves();
    for (int t = 0; t < 200; ++t) {
        (void)off.plan_step(uf, 4, 4);
        (void)few.plan_step(uf, 4, 4);
        (void)on.plan_step(uf, 4, 4);
    }
    CHECK(off.zero_moves() > m);                   /* 0.1447 > 0.1365 x 1.02: drafting */
    CHECK(few.zero_moves() > m);
    CHECK_EQ(on.zero_moves(), m);                  /* 0.1447 < 0.1484 x 1.02: 0 holds */
    CHECK(on.borrow_cycles() > 0);
    CHECK_EQ(off.borrow_cycles(), (int64_t)0);
    /* The switch off between steps: the same history leaves 0. */
    on.set_borrow(false);
    for (int t = 0; t < 200; ++t) (void)on.plan_step(uf, 4, 4);
    CHECK(on.zero_moves() > m);
}

/* Through the scheduler: the switches, refused until the engine set the lend up, and each step's lend state
 * as the engine's view reports it -- no sample while it moves, a lent plain step into the lent table past
 * kLendWarm cycles of a settled lend. */
TEST(the_scheduler_hands_the_lend_its_state_and_refuses_it_until_set_up) {
    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());
    CHECK(!f.sch.lend_capable());
    CHECK_EQ(f.sch.set_draft_lend(1), 0);            /* no view: refused */
    CHECK_EQ(f.sch.set_draft_lend_borrow(1), 0);
    CHECK(!f.sch.draft_lend_wanted());
    int state = DraftController::kLendResident;
    uint64_t changes = 0;
    f.sch.set_lend_view([&](uint64_t* ch) { *ch = changes; return state; }, true);
    CHECK(f.sch.lend_capable());
    CHECK_EQ(f.sch.set_draft_lend(1), 1);
    CHECK_EQ(f.sch.set_draft_probe_backoff(3), 3);
    CHECK_EQ(f.sch.set_draft_probe_backoff(9), DraftController::kBackoffMax);   /* clamped (fn4 refuses 9) */
    CHECK_EQ(f.sch.set_draft_probe_backoff(3), 3);
    CHECK_EQ(f.sch.set_draft_depth0(1), 1);

    const uint64_t a = f.submit(160, 4000);
    (void)a;
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    /* Measured: windows cost 40 ms and more at one row, a plain step 5 ms: depth 0. The steps here
     * are fed by hand (their wall time is microseconds). */
    DraftController& dc = f.sch.draft();
    for (int d = 1; d <= 4; ++d)
        for (int k = 0; k < 4; ++k) dc.step_time(d, 1, 40.0 + d);
    bool wanted = false;
    for (int t = 0; t < 600 && !wanted; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        if (!dc.head_runs()) dc.step_time(0, 1, 5.0);
        wanted = f.sch.draft_lend_wanted();
        f.commit_plain(13);
    }
    REQUIRE(wanted);
    CHECK(dc.probe_interval() >= DraftController::kLendInterval);
    /* The engine moves the slab: steps while it moves are no samples. */
    state = DraftController::kLendTransit;
    ++changes;
    const int64_t n0 = dc.cost_samples(0, 1);
    for (int t = 0; t < 6; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(14);
        f.sch.step_finished();
    }
    CHECK_EQ(dc.lent_samples(1), (int64_t)0);
    /* Lent: past kLendWarm cycles, the plain steps go to the lent table, never to the unlent one. */
    state = DraftController::kLendLent;
    ++changes;
    for (int t = 0; t < 3 * DraftController::kLendWarm; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(15);
        f.sch.step_finished();
    }
    CHECK(dc.lent_samples(1) > 0);
    CHECK(dc.lent_samples(1) <= 2 * DraftController::kLendWarm);
    CHECK_EQ(dc.cost_samples(0, 1), n0);
    CHECK(f.sch.draft_lend_wanted());
    /* Depth 0 off: drafting at once, the lend not wanted. */
    CHECK_EQ(f.sch.set_draft_depth0(0), 0);
    CHECK(!f.sch.draft_lend_wanted());
    CHECK_EQ(f.sch.set_draft_lend_borrow(1), 1);
    CHECK(f.sch.draft().borrow());
    CHECK_EQ(f.sch.set_draft_lend_borrow(2), 1);      /* not 0 or 1: unchanged */
    CHECK_EQ(f.sch.set_draft_lend_borrow(0), 0);
    CHECK_EQ(f.sch.set_draft_lend(-1), -1);           /* forced: wanted whatever the depth */
    CHECK(f.sch.draft_lend_wanted());
    CHECK_EQ(f.sch.set_draft_lend(0), 0);
    CHECK(!f.sch.draft_lend_wanted());
    /* A view that cannot lend: the switch refused and left off. */
    f.sch.set_lend_view([&](uint64_t* ch) { *ch = changes; return state; }, false);
    CHECK(!f.sch.lend_capable());
    CHECK_EQ(f.sch.set_draft_lend(1), 0);
    CHECK_EQ(f.sch.set_draft_lend(-1), 0);
    CHECK_EQ(f.sch.set_draft_lend_borrow(1), 0);
}

/* ------------------------------------------------------------------ the hybrid (scheduler.h)
 * Beside a head drafting chains, a sequence whose prompt lookup matches at least the threshold
 * back verifies the lookup's proposal, the whole window, instead of the head's; a short match or
 * none leaves the head's. Off, the head's always. A tree head cannot take it. */
TEST(the_hybrid_takes_a_long_lookup_match_over_the_heads_proposal) {
    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.set_draft_hybrid(1), 1);
    CHECK_EQ(f.sch.set_draft_hybrid_match(8), 8);

    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 2000;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;   /* a period of 8 */
    const uint64_t rep = r->id;
    CHECK_OK(f.sch.add(std::move(r)));
    const uint64_t other = f.submit(160, 2000);                         /* 1000 + i: no repeat */
    /* Each step commits what continues each text: the period for one, a new id for the other. */
    auto len = [&](uint64_t id) {
        const Request* q = f.sch.find(id);
        return (int64_t)(q->prompt.size() + q->output.size());
    };
    auto commit_next = [&]() {
        const StepPlan& p = f.sch.plan();
        std::vector<int32_t> a(p.e.size(), 0), t(p.e.size(), 0);
        std::vector<uint8_t> e(p.e.size(), 0);
        for (size_t i = 0; i < p.e.size(); ++i)
            t[i] = p.e[i].seq == rep ? (int32_t)(100 + len(rep) % 8) : (int32_t)(2000 + len(other));
        Scheduler::StepResult res;
        res.n_seq = (int)p.e.size();
        res.n_accepted = a.data();
        res.token = t.data();
        res.eos = e.data();
        CHECK_OK(f.sch.commit(res));
    };
    for (int t = 0; t < 8 && (f.sch.find(rep)->output.empty() || f.sch.find(other)->output.empty());
         ++t) {
        REQUIRE(f.sch.step() != nullptr);
        commit_next();
    }
    REQUIRE(!f.sch.find(rep)->output.empty());
    REQUIRE(!f.sch.find(other)->output.empty());
    auto entry = [&](uint64_t id) -> const StepEntry* {
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == id) return &e;
        return nullptr;
    };
    int32_t head[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(rep, head, 4));
    CHECK_OK(f.sch.set_draft(other, head, 4));
    const int64_t L = len(rep);
    REQUIRE(f.sch.step() != nullptr);
    const StepEntry* er = entry(rep);
    const StepEntry* eo = entry(other);
    REQUIRE(er != nullptr);
    REQUIRE(eo != nullptr);
    /* The periodic one continues its period: the lookup's, the whole window. */
    CHECK(er->ngram);
    CHECK_EQ(er->n_spec, 4);
    CHECK_EQ(er->draft[0], (int32_t)(100 + L % 8));
    CHECK_EQ(er->draft[3], (int32_t)(100 + (L + 3) % 8));
    /* The other has no match: the head's. */
    CHECK(!eo->ngram);
    CHECK(eo->n_spec > 0);
    CHECK_EQ(eo->draft[0], 21);
    commit_next();
    /* A threshold past the match the text gives (counted at most that far, 256): the head's. */
    CHECK_EQ(f.sch.set_draft_hybrid_match(256), 256);
    CHECK_OK(f.sch.set_draft(rep, head, 4));
    REQUIRE(f.sch.step() != nullptr);
    er = entry(rep);
    REQUIRE(er != nullptr);
    CHECK(!er->ngram);
    CHECK_EQ(er->draft[0], 21);
    commit_next();
    /* Off: the head's. */
    CHECK_EQ(f.sch.set_draft_hybrid_match(8), 8);
    CHECK_EQ(f.sch.set_draft_hybrid(0), 0);
    CHECK_OK(f.sch.set_draft(rep, head, 4));
    REQUIRE(f.sch.step() != nullptr);
    er = entry(rep);
    REQUIRE(er != nullptr);
    CHECK(!er->ngram);
    CHECK_EQ(er->draft[0], 21);
    commit_next();

    /* A tree head cannot take it. */
    Fixture t;
    adaptive_fixture(t, 4);
    t.prog.drafter.kind = RAD_DRAFT_BLOCK;
    t.prog.drafter_tree = true;
    CHECK_OK(t.start());
    CHECK_EQ(t.sch.set_draft_hybrid(1), 0);
    /* Nor a deployment with no head. */
    Fixture n;
    adaptive_fixture(n, 4);
    n.prog.drafter.kind = RAD_DRAFT_NONE;
    n.prog.drafter.depth = 0;
    CHECK_OK(n.start());
    CHECK_EQ(n.sch.set_draft_hybrid(1), 0);
}

TEST(prompt_lookup_alone_follows_its_own_acceptance) {
    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_NONE;
    f.prog.drafter.depth = 0;
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 2000;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);

    int first = 0, last = 0, lookups = 0;
    for (int t = 0; t < 6 * DraftController::kHold; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        const StepEntry& e = f.sch.plan().e[0];
        if (e.n_spec > 0) {
            CHECK(e.ngram);
            /* What the pattern says comes next, the period continued. */
            CHECK_EQ(e.draft[0], 100 + (t + 1) % 8);
            if (!first) first = e.n_spec;
            last = e.n_spec;
            ++lookups;
        }
        f.commit_plain(100 + (t + 1) % 8);             /* every lookup refused */
    }
    CHECK_EQ(lookups, 6 * DraftController::kHold);
    CHECK_EQ(first, 4);
    CHECK_EQ(last, 1);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 1.0, 1e-6);
}

/* ------------------------------------------------------------------ prompt lookup's verify gate */
/* A held proposal is scored against the text that follows, as a verify would have scored it: its
 * accepted count is the common prefix. One proposal at a time; the next is taken once it ends. */
TEST(the_lookup_gate_scores_a_held_proposal_against_the_text_that_follows) {
    LookupGate g;
    g.configure(4);
    LookupSeq q;
    /* The priors: a short match waits, a long one drafts. */
    CHECK_NEAR(g.expect(q, 0, 4), 4 * LookupGate::kPriorShort, 1e-6);
    CHECK_NEAR(g.expect(q, 1, 4), 4 * LookupGate::kPriorLong, 1e-6);
    CHECK_EQ(LookupGate::class_of(LookupGate::kLongMatch - 1), 0);
    CHECK_EQ(LookupGate::class_of(LookupGate::kLongMatch), 1);

    std::vector<int32_t> text(10, 1);                  /* the sequence so far: 10 tokens */
    TokenSeq ts;
    auto view = [&]() {
        ts.a = text.data();
        ts.na = (int64_t)text.size();
    };
    const int32_t prop[4] = { 5, 6, 7, 8 };
    g.offer(q, 10, prop, 4, 2);
    g.held(q, 4);
    const int32_t other[4] = { 40, 41, 42, 43 };
    g.offer(q, 10, other, 4, 9);                       /* pending already: counted, not taken */
    CHECK_EQ(q.proposed, (int64_t)2);
    CHECK_EQ(q.pend_cls, 0);

    text.push_back(5);                                 /* the first came true; no verdict yet */
    view();
    g.advance(q, ts);
    CHECK(q.pend_pos >= 0);
    CHECK_NEAR(q.s[0][0], 0.0, 1e-6);
    text.push_back(6);
    text.push_back(9);                                 /* two came true, the third did not */
    view();
    g.advance(q, ts);
    CHECK(q.pend_pos < 0);
    /* The prior weighs two proposals: the first evaluated moves each position a third. */
    const double w = 1.0 / (LookupGate::kPriorWeight + 1.0);
    CHECK_NEAR(q.s[0][0], w, 1e-6);
    CHECK_NEAR(q.s[0][1], w, 1e-6);
    CHECK_NEAR(q.s[0][2], 0.0, 1e-6);
    CHECK_NEAR(q.s[0][3], 0.0, 1e-6);
    CHECK_NEAR(q.s[1][0], LookupGate::kPriorLong, 1e-6);   /* the other class untouched */
    CHECK_EQ(q.held_hits, (int64_t)2);
    CHECK_EQ(g.held_hits(), (int64_t)2);
    CHECK_EQ(g.held_count(), (int64_t)1);
    CHECK_EQ(g.held_drafts(), (int64_t)4);

    /* A proposal that comes true whole ends at its last token; a verified one counts no held
     * hits but feeds the estimate the same way. */
    const int64_t at = (int64_t)text.size();
    const int32_t whole[4] = { 11, 12, 13, 14 };
    g.offer(q, at, whole, 4, 3);
    for (int32_t t : whole) text.push_back(t);
    view();
    g.advance(q, ts);
    CHECK(q.pend_pos < 0);
    CHECK(q.s[0][3] > 0.0f);
    CHECK_EQ(q.held_hits, (int64_t)2);

    /* Evidence is what moves a short match into drafting: a run of proposals that come true. */
    LookupGate g2;
    g2.configure(4);
    LookupSeq r;
    CHECK(!g2.pays(g2.expect(r, 0, 4), 4, 1, 1.0));
    std::vector<int32_t> s2(4, 1);
    for (int i = 0; i < 6; ++i) {
        const int32_t p2[4] = { 2, 3, 4, 5 };
        g2.offer(r, (int64_t)s2.size(), p2, 4, 2);
        for (int32_t t : p2) s2.push_back(t);
        TokenSeq t2;
        t2.a = s2.data();
        t2.na = (int64_t)s2.size();
        g2.advance(r, t2);
    }
    CHECK(g2.expect(r, 0, 4) > 2.9);   /* 6 / (6 + 2) a position */
    CHECK(g2.pays(g2.expect(r, 0, 4), 4, 1, 1.0));
}

/* What a verify costs: the measured step time at the step's rows with the drafts against without
 * them; between timed row counts a line, outside them the nearest scaled by the prior's shape. */
TEST(the_lookup_gate_prices_a_verify_by_measured_step_times) {
    LookupGate g;
    g.configure(4);
    /* Untimed: the prior shape, 6% a row, so a whole window pays at a quarter of a token. */
    CHECK_NEAR(g.step_cost(5) / g.step_cost(1), 1.0 + 4 * LookupGate::kRowCost, 1e-9);
    CHECK(g.pays(0.3, 4, 1, 1.0));
    CHECK(!g.pays(0.2, 4, 1, 1.0));
    CHECK(!g.pays(0.0, 4, 1, 1.0));

    /* One row timed: a verify is priced off it on the prior's shape. */
    for (int i = 0; i < 8; ++i) g.step_time(1, 10.0);
    CHECK_EQ(g.cost_samples(1), (int64_t)8);
    CHECK_NEAR(g.step_cost(5), 10.0 * (1.0 + 4 * LookupGate::kRowCost), 1e-9);

    /* Both timed: 2.5x for four drafts (a MoE reading each row's experts). 1 + E must beat it. */
    for (int i = 0; i < 8; ++i) g.step_time(5, 25.0);
    CHECK_NEAR(g.step_cost(5), 25.0, 1e-9);
    CHECK_NEAR(g.step_cost(3), 17.5, 1e-9);            /* between: the line */
    CHECK(!g.pays(1.4, 4, 1, 1.0));
    CHECK(g.pays(1.6, 4, 1, 1.0));
    /* Past the last timed count: the prior's shape from there. */
    CHECK_NEAR(g.step_cost(9), 25.0 * (1.0 + 8 * LookupGate::kRowCost) /
                                   (1.0 + 4 * LookupGate::kRowCost), 1e-9);
    /* In a step of eight rows already expecting eight tokens, the same drafts weigh against the
     * whole step: marginal cost over the step's rate. */
    for (int i = 0; i < 8; ++i) g.step_time(8, 30.0);
    for (int i = 0; i < 8; ++i) g.step_time(12, 36.0);
    CHECK(!g.pays(1.5, 4, 8, 8.0));                    /* 1.5 / 8 < 36 / 30 - 1 */
    CHECK(g.pays(1.7, 4, 8, 8.0));

    /* A step that waited on something else counts as at most four times the mean. */
    g.step_time(1, 1000.0);
    CHECK(g.step_cost(1) < 10.0 + 40.0 / 8.0);
    /* Junk is ignored. */
    g.step_time(0, 5.0);
    g.step_time(2, -1.0);
    CHECK_EQ(g.cost_samples(2), (int64_t)0);
    /* Wide steps share keys in bands of 32 rows past 128. */
    g.step_time(200, 80.0);
    g.step_time(210, 90.0);
    CHECK_EQ(g.cost_samples(205), (int64_t)2);
}

/* The index says how far the match behind a proposal runs back, up to the cap. */
TEST(prompt_lookup_reports_how_far_its_match_runs_back) {
    NgramIndex ix;
    /* 1 2 3 4 5 6 7 8 9 | 30 31 | 5 6 7 8 9: the last four match 5 6 7 8 9 at its first place,
     * which runs five back; 30 31 then follows. */
    std::vector<int32_t> s = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31, 5, 6, 7, 8, 9 };
    TokenSeq ts;
    ts.a = s.data();
    ts.na = (int64_t)s.size();
    int32_t out[4];
    int m = -1;
    CHECK_EQ(ix.propose(ts, 2, 4, 4, out, &m, 8), 4);
    CHECK_EQ(out[0], 30);
    CHECK_EQ(m, 5);
    /* A copy that runs back further is cut at the cap. */
    std::vector<int32_t> c;
    for (int i = 0; i < 20; ++i) c.push_back(100 + i);
    for (int i = 0; i < 12; ++i) c.push_back(100 + i);
    TokenSeq tc;
    tc.a = c.data();
    tc.na = (int64_t)c.size();
    NgramIndex ic;
    CHECK_EQ(ic.propose(tc, 2, 4, 4, out, &m, 8), 4);
    CHECK_EQ(out[0], 112);
    CHECK_EQ(m, 8);
    /* Without the out-parameter nothing changes. */
    NgramIndex id;
    CHECK_EQ(id.propose(tc, 2, 4, 4, out), 4);
}

namespace {
/* A lookup-only deployment (no draft head) at window `depth`, the fixed window (the default). */
void lookup_fixture(Fixture& f, int depth) {
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = depth;
    f.prog.drafter.kind = RAD_DRAFT_NONE;
    f.prog.drafter.depth = 0;
}
}  /* namespace */

/* Prompt lookup alone, the gate on by default. Text that copies a stretch it has seen drafts at
 * once and keeps drafting while its drafts come true; text that only repeats bigrams it has seen,
 * whose proposals do not come true, is never verified, yet every proposal is still scored; and a
 * reply that turns from the second kind into the first drafts as soon as it does. The text is
 * the same either way: the emitted token is the fixture's, whatever was verified. */
TEST(prompt_lookup_alone_holds_back_proposals_that_do_not_pay) {
    Fixture f;
    lookup_fixture(f, 4);
    std::string log = info_of([&]() { CHECK_OK(f.start()); });
    CHECK(log.find("prompt lookup: a proposal is verified when") != std::string::npos);
    CHECK(!f.sch.draft().adaptive());

    /* A copy: an 8-token period, 160 tokens of it; the reply goes on with it. */
    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 2000;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);
    f.sch.step_finished();                               /* a prefill step: not a sample */
    for (int64_t rows = 1; rows <= 5; ++rows)
        CHECK_EQ(f.sch.lookup_gate().cost_samples(rows), (int64_t)0);
    int64_t pos = 161;                                   /* the next token's place in the period */
    auto truth = [&](int64_t p) { return (int32_t)(100 + p % 8); };
    int verified = 0;
    for (int t = 0; t < 24; ++t) {
        REQUIRE(f.sch.step() != nullptr);
        const StepEntry& e = f.sch.plan().e[0];
        int32_t a = 0;
        if (e.n_spec > 0) {
            CHECK(e.ngram);
            ++verified;
            while (a < e.n_spec && e.draft[a] == truth(pos + a)) ++a;
        }
        CHECK_OK(f.commit(truth(pos + a), 0, [a](const StepEntry&) { return a; }));
        f.sch.step_finished();                           /* a decode step of 1 + 4 rows, timed */
        pos += a + 1;
    }
    CHECK_EQ(verified, 24);
    CHECK_EQ(f.sch.lookup_gate().cost_samples(5), (int64_t)24);
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.lookup_held, (int64_t)0);
    CHECK_EQ(m.lookup_proposals, (int64_t)24);
    CHECK_EQ(m.draft_accepted, m.draft_tokens);

    /* Recurring bigrams: the prompt holds every pair of the reply 200, 201, 202, ... followed by
     * a 50, so after each token the lookup proposes 50 on a two-token match, and it never comes
     * true. */
    Fixture g;
    lookup_fixture(g, 4);
    CHECK_OK(g.start());
    auto q = std::make_unique<Request>();
    q->id = g.next_id++;
    q->max_tokens = 2000;
    for (int i = 0; i < 60; ++i) {
        q->prompt.push_back(200 + i);
        q->prompt.push_back(201 + i);
        q->prompt.push_back(50);
    }
    q->prompt.push_back(199);
    CHECK_OK(g.sch.add(std::move(q)));
    REQUIRE(g.sch.step() != nullptr);
    g.commit_plain(200);
    int proposals = 0;
    for (int t = 0; t < 40; ++t) {
        REQUIRE(g.sch.step() != nullptr);
        CHECK_EQ(g.sch.plan().e[0].n_spec, 0);           /* held, every one */
        g.commit_plain(201 + t);
        ++proposals;
    }
    m = g.sch.metrics();
    CHECK_EQ(m.draft_tokens, (int64_t)0);
    CHECK(m.lookup_proposals >= proposals - 1);
    CHECK_EQ(m.lookup_held, m.lookup_proposals);
    CHECK_EQ(m.lookup_held_drafts, 4 * m.lookup_held);
    CHECK_EQ(m.lookup_held_hits, (int64_t)0);

    /* The reply now copies its own stretch 201.. from the start: a long match, drafted at once. */
    std::vector<int32_t> copy;
    for (int i = 0; i < 40; ++i) copy.push_back(200 + i);
    size_t k = 0;
    int drafted_steps = 0, steps = 0;
    while (k < copy.size() && steps < 40) {
        REQUIRE(g.sch.step() != nullptr);
        const StepEntry& e = g.sch.plan().e[0];
        int32_t a = 0;
        if (e.n_spec > 0) {
            ++drafted_steps;
            while (a < e.n_spec && k + (size_t)a < copy.size() &&
                   e.draft[a] == copy[k + (size_t)a])
                ++a;
        }
        CHECK_OK(g.commit(copy[std::min(k + (size_t)a, copy.size() - 1)], 0,
                          [a](const StepEntry&) { return a; }));
        k += (size_t)a + 1;
        ++steps;
    }
    CHECK(drafted_steps > 0);
    CHECK(g.sch.metrics().draft_accepted > 0);
}

namespace {
/* Prompt lookup by scanning: the longest pattern first, its latest earlier occurrence, the
 * proposal continued past the end with its own tokens. The index must answer exactly this. */
int scan_propose(const std::vector<int32_t>& s, int nmin, int nmax, int depth, int32_t* out) {
    const int64_t L = (int64_t)s.size();
    if (L < 2) return 0;
    for (int n = (int)std::min<int64_t>(nmax, L - 1); n >= nmin; --n)
        for (int64_t e = L - 2; e >= n - 1; --e) {
            bool ok = true;
            for (int k = 0; k < n && ok; ++k) ok = s[(size_t)(e - k)] == s[(size_t)(L - 1 - k)];
            if (!ok) continue;
            for (int d = 0; d < depth; ++d) {
                const int64_t i = e + 1 + d;
                out[d] = i < L ? s[(size_t)i] : out[i - L];
            }
            return depth;
        }
    return 0;
}
}  /* namespace */

/* The index against the scan, over sequences grown a few tokens at a time across the prompt /
 * output seam, on alphabets small enough to repeat often and large enough to miss, at several
 * pattern ranges; then a change of range and a shorter sequence, which rebuild it. */
TEST(prompt_lookup_index_answers_what_a_scan_answers) {
    uint32_t rng = 12345;
    auto next = [&](uint32_t mod) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) % mod; };
    const int kRanges[][2] = { { 1, 1 }, { 1, 4 }, { 2, 4 }, { 3, 7 }, { 15, 15 } };
    int64_t hits = 0, misses = 0;
    for (const uint32_t alphabet : { 2u, 5u, 40u, 3000u }) {
        for (const auto& rg : kRanges) {
            std::vector<int32_t> seq;
            const int64_t np = 50 + next(200);
            NgramIndex ix;
            ix.reset(7);
            while (seq.size() < 1500) {
                const int add = 1 + (int)next(6);
                for (int k = 0; k < add; ++k) {
                    /* Runs and copies as well as noise, so long patterns recur. */
                    if (seq.size() > 20 && next(4) == 0) {
                        const size_t from = next((uint32_t)seq.size() - 10);
                        for (int c = 0; c < 6; ++c) seq.push_back(seq[from + (size_t)c]);
                    } else {
                        seq.push_back((int32_t)next(alphabet));
                    }
                }
                TokenSeq ts;
                ts.a = seq.data();
                ts.na = std::min<int64_t>(np, (int64_t)seq.size());
                ts.b = seq.data() + ts.na;
                ts.nb = (int64_t)seq.size() - ts.na;
                int32_t want[8] = {}, got[8] = {};
                const int depth = 1 + (int)next(8);
                const int w = scan_propose(seq, rg[0], rg[1], depth, want);
                const int g = ix.propose(ts, rg[0], rg[1], depth, got);
                CHECK_EQ(g, w);
                if (g == w) CHECK(std::memcmp(got, want, sizeof(int32_t) * (size_t)w) == 0);
                (w ? hits : misses) += 1;
            }
            CHECK_EQ(ix.indexed(), (int64_t)seq.size() - 1);
            CHECK(ix.bytes() <= (size_t)(16 * (rg[1] - rg[0] + 1)) * seq.size() + 8192);
        }
    }
    CHECK(hits > 100);
    CHECK(misses > 100);

    /* Another range, then a shorter sequence (another request in the slot): both rebuild. */
    std::vector<int32_t> a = { 1, 2, 3, 9, 1, 2, 3 };
    TokenSeq ta{ a.data(), (int64_t)a.size(), nullptr, 0 };
    NgramIndex ix;
    int32_t out[4] = {};
    CHECK_EQ(ix.propose(ta, 2, 4, 2, out), 2);
    CHECK_EQ(out[0], 9);
    CHECK_EQ(out[1], 1);
    CHECK_EQ(ix.propose(ta, 4, 4, 2, out), 0);         /* no 4-gram recurs */
    std::vector<int32_t> b = { 5, 6, 5 };
    TokenSeq tb{ b.data(), (int64_t)b.size(), nullptr, 0 };
    CHECK_EQ(ix.propose(tb, 1, 4, 3, out), 3);         /* period 2, continued: 6 5 6 */
    CHECK_EQ(out[0], 6);
    CHECK_EQ(out[1], 5);
    CHECK_EQ(out[2], 6);
}

/* A lookup costs the n-grams the last commit added, not a pass over the context: over a 400k-token
 * prompt that never repeats a 2-gram, the first lookup indexes it once and the next thousand
 * lookups, a token apart, together take a small fraction of that. */
TEST(prompt_lookup_does_not_scan_the_context) {
    std::vector<int32_t> seq;
    seq.reserve(401000);
    for (int i = 0; i < 400000; ++i) seq.push_back(i);   /* every token distinct: always a miss */
    NgramIndex ix;
    int32_t out[8];
    TokenSeq ts{ seq.data(), (int64_t)seq.size(), nullptr, 0 };
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_EQ(ix.propose(ts, 2, 4, 8, out), 0);
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        seq.push_back(1000000 + i);
        TokenSeq t2{ seq.data(), (int64_t)seq.size(), nullptr, 0 };
        CHECK_EQ(ix.propose(t2, 2, 4, 8, out), 0);
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double build = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double steps = std::chrono::duration<double, std::milli>(t2 - t1).count();
    fprintf(stderr, "    index 400k tokens: %.1f ms; 1000 lookups after: %.3f ms (%.1f MiB)\n", build,
            steps, (double)ix.bytes() / 1048576.0);
    CHECK(steps < build);
    CHECK_EQ(ix.indexed(), (int64_t)seq.size() - 1);
}

/* The scheduler's half of the cost: a decode step that verified drafts is timed at its deepest
 * window and row count, and the next cycle's head depth reads it. Measured dear past one round at
 * two rows, both windows (and the rounds) come down to one. */
TEST(the_adaptive_window_times_its_steps_and_the_next_cycle_reads_them) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    const uint64_t a = f.submit(160, 2000);
    const uint64_t b = f.submit(160, 2000);
    /* Every draft taken, so both stops stay at the whole window and only the cost moves D. */
    auto commit_all = [&]() {
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.is_prefill ? 0 : e.n_spec; }));
    };
    REQUIRE(f.sch.step() != nullptr);                  /* both prompts */
    commit_all();
    f.sch.step_finished();                             /* a prefill step: not a sample */
    for (int d = 1; d <= 4; ++d) CHECK_EQ(f.sch.draft().cost_samples(d, 2), (int64_t)0);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().n_spec, 4);
    commit_all();
    f.sch.step_finished();
    CHECK_EQ(f.sch.draft().cost_samples(4, 2), (int64_t)1);

    /* The hold, then a cycle that reads steps past one round as four times dearer. */
    for (int d = 1; d <= 4; ++d)
        for (int i = 0; i < 64; ++i) f.sch.draft().step_time(d, 2, d == 1 ? 10.0 : 40.0);
    for (int t = 0; t < DraftController::kHold + 1; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_OK(f.sch.set_draft(b, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        commit_all();
    }
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, 1);
    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 2);
    CHECK_EQ(rounds, 1);
    f.sch.serial_end(0);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 1.0, 1e-6);
}

/* ================================================================== the step path */
/* The single-sequence decode floor the dashboard reads. Timing is real wall clock, so the
 * assertions hold on any machine: a step of a shape that must not sample leaves the floor at zero
 * (a min only goes down, so a wrong sample must be caught while the floor is unset), and a
 * single-sequence step sets it. */
TEST(the_decode_floor_estimator_does_not_latch_a_degenerate_bracket) {
    /* A synthetic stream: the fixture runs steps in microseconds and cannot produce a degenerate
     * 1.34 ms bracket among 15 ms steps. */
    using Est = rad::DecodeFloor;

    /* A min over raw brackets would latch a single degenerate sample; a min over window means
     * dilutes it with the steps sharing its window. */
    {
        Est est;
        const int64_t real = 15'000'000, degenerate = 1'340'000;
        est.update(degenerate);                    /* same degenerate, same position */
        for (int i = 0; i < 500; ++i) est.update(real);
        const int64_t floor = est.best();
        CHECK(floor > 0);
        CHECK(floor > degenerate);                 /* did NOT latch the degenerate */
        CHECK(floor >= real * 3 / 4 && floor <= real);
    }

    /* A high rate of degenerates (one in four; a checkpoint every 256 tokens makes them recur):
     * the floor must track the real step time, not merely beat the degenerate. */
    {
        Est est;
        const int64_t real = 15'000'000, degenerate = 1'340'000;
        for (int i = 0; i < 400; ++i) est.update((i % 4 == 0) ? degenerate : real);
        const int64_t floor = est.best();
        CHECK(floor >= real * 90 / 100);           /* tracks the real step, not the poison */
        CHECK(floor <= real);
    }

    /* A steady stream with no degenerates: the floor equals the true floor. */
    {
        Est est;
        const int64_t real = 14'500'000;
        for (int i = 0; i < 200; ++i) est.update(real);
        CHECK_EQ(est.best(), real);
    }
}

TEST(the_decode_floor_rejects_degenerate_brackets_so_the_floor_is_the_true_floor) {
    /* Window means dilute a lone degenerate but not a recurrent one. At a ~1.4% degenerate rate
     * (a decode step issued ahead behind a long host commit: prefill, or a KV checkpoint) a min
     * over hundreds of windows catches one holding two or three, and lands ~10% below the true
     * floor. So a candidate far below the running mean of accepted steps is rejected: degenerates
     * sit at ~9% of the mean and real steps at ~96%, and the 0.5 threshold is the gap's midpoint. */
    using Est = rad::DecodeFloor;
    const int64_t true_floor = 14'400'000, typical = 15'000'000, degen = 1'340'000;

    /* ~1.4% degenerates, the rest jittered between the true floor and the typical step. The
     * floor must land near the true floor, not the contaminated window mean. */
    Est est;
    uint32_t rng = 12345;
    for (int i = 0; i < 400 * Est::kWindow; ++i) {
        rng = rng * 1664525u + 1013904223u;
        const bool d = ((rng >> 16) % 1000u) < 14u;                 /* ~1.4% */
        int64_t ns;
        if (d) ns = degen;
        else {
            rng = rng * 1664525u + 1013904223u;
            ns = true_floor + (int64_t)((rng >> 8) % (typical - true_floor + 1));  /* floor..typical */
        }
        est.update(ns);
    }
    const int64_t floor = est.best();
    /* Within 5% of the true floor. The un-rejected estimator lands ~10% low here, so this fails
     * without the rejection gate. */
    CHECK(floor >= true_floor * 95 / 100);
    CHECK(floor <= true_floor * 115 / 100);

    /* The gate needs a window of accepted samples before it can judge one, so a degenerate first
     * window must be recorded by neither side, or the floor latches an unreachable value. */
    {
        Est est;
        for (int i = 0; i < Est::kWindow; ++i) est.update(degen);   /* whole first window poisoned */
        CHECK_EQ(est.best(), (int64_t)0);
        CHECK_EQ(est.recent(), (int64_t)0);
        for (int i = 0; i < 40 * Est::kWindow; ++i) est.update(typical);
        CHECK(est.best() >= typical * 95 / 100);   /* floor came from a gated window, not bootstrap */
    }

    /* The gate does not reject real steps: a clean stream records the true floor. */
    Est clean;
    for (int i = 0; i < 100 * Est::kWindow; ++i) clean.update(true_floor);
    CHECK_EQ(clean.best(), true_floor);
}

TEST(the_decode_floor_reports_a_recent_window_for_like_for_like_drift) {
    /* "How fast now" against "how fast ever" from different instruments (the engine's floor, the
     * browser's per-poll mean) disagree by order statistics alone: a min over hundreds of windows
     * sits below a min over the last sixty samples on a flat engine. So the engine reports both
     * sides from the same stream and window size. */
    using Est = rad::DecodeFloor;
    const int64_t base = 15'000'000;

    /* A steady engine with realistic run-to-run jitter: the recent window and the floor are both
     * window means of the same population, so they must agree closely. */
    {
        Est est;
        for (int i = 0; i < 400 * Est::kWindow; ++i) {
            const int64_t jitter = ((i % 7) - 3) * 200'000;      /* +/- 0.6 ms, deterministic */
            est.update(base + jitter);
        }
        const int64_t floor = est.best(), recent = est.recent();
        CHECK(floor > 0);
        CHECK(recent > 0);
        /* The gap must be small: the deadband is sized against it. */
        const int64_t gap_pct = 100 * (recent - floor) / floor;
        CHECK(gap_pct < 3);
    }

    /* A degraded engine: the recent window rises and the floor keeps the best. */
    {
        Est est;
        for (int i = 0; i < 200 * Est::kWindow; ++i) est.update(base);
        const int64_t floor = est.best();
        for (int i = 0; i < 20 * Est::kWindow; ++i) est.update(base * 6 / 5);   /* 20% slower */
        const int64_t recent = est.recent();
        CHECK(recent > floor * 115 / 100);
        /* The floor does not follow the degradation. */
        CHECK_EQ(est.best(), floor);
    }

    /* Before any window completes there is no recent reading -- the page must not divide by it. */
    {
        Est est;
        for (int i = 0; i < Est::kWindow - 1; ++i) est.update(base);
        CHECK_EQ(est.recent(), (int64_t)0);
        est.update(base);                        /* completes window 1: the un-gated warmup */
        CHECK_EQ(est.recent(), (int64_t)0);      /* warmup is recorded by neither side */
        for (int i = 0; i < Est::kWindow; ++i) est.update(base);
        CHECK(est.recent() > 0);
    }
}

TEST(the_decode_floor_records_only_single_sequence_decode_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* A prefill step samples nothing. Budget 320 tokens: this test walks 1 prefill + one window,
     * at most six windows until the floor is set and one more, 257 steps. */
    uint64_t a = f.submit(160, 320, /*priority=*/0, /*ignore_eos=*/true);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The floor is a min over window means of single-sequence decode steps, so it stays 0 until a
     * full window (DecodeFloor::kWindow) has run. */
    for (int i = 1; i < rad::DecodeFloor::kWindow; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().n_seq_decode, 1);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        f.commit_plain(12);
        f.sch.step_finished();
    }
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The step that completes the FIRST window leaves the floor unset: window 1 is the un-gated
     * warmup (the rejection gate needs a window of accepted samples before it can judge one), so
     * it is recorded by neither side. */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(13);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The first gated window sets the floor: positive, and no larger than the cumulative decode
     * time it was drawn from. The fixture's steps take microseconds and vary widely, so the gate
     * may reject some and the window take more than kWindow steps to fill. */
    for (int i = 0; i < 6 * rad::DecodeFloor::kWindow && !f.sch.metrics().step_ns_decode_best;
         ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(14);
        f.sch.step_finished();
    }
    const int64_t floor1 = f.sch.metrics().step_ns_decode_best;
    CHECK(floor1 > 0);
    CHECK(floor1 <= f.sch.metrics().step_ns_decode);

    /* A further full window can only lower the floor or leave it: it is a min over windows. */
    for (int i = 0; i < rad::DecodeFloor::kWindow; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(15);
        f.sch.step_finished();
    }
    const int64_t floor2 = f.sch.metrics().step_ns_decode_best;
    CHECK(floor2 > 0);
    CHECK(floor2 <= floor1);
    (void)a;
}

TEST(the_decode_floor_ignores_a_decode_step_issued_behind_a_prefill) {
    /* The decode step issued behind the prefill's host-side staging is a real single-sequence
     * decode step: it is eligible, and the window mean dilutes its short bracket. Decode steps
     * that begin with it produce a floor. The fixture's steps take microseconds and vary widely,
     * so the gate may reject some and a window take more than kWindow steps to fill. */
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 256, /*priority=*/0, /*ignore_eos=*/true);
    REQUIRE(f.sch.step() != nullptr);          /* prefill */
    f.commit_plain(21);
    f.sch.step_finished();

    /* Window 1 (holding the prefill-following step) is the un-gated warmup, so the floor only
     * appears once a gated window completes. */
    for (int i = 0; i < 6 * rad::DecodeFloor::kWindow && !f.sch.metrics().step_ns_decode_best;
         ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        CHECK_EQ(f.sch.plan().n_seq_decode, 1);
        CHECK(f.sch.plan().n_tok_decode >= 1); /* a token DID advance; token count is not the guard */
        f.commit_plain(22);
        f.sch.step_finished();
    }
    /* The window filled, counting the prefill-following step: the floor is set. */
    CHECK(f.sch.metrics().step_ns_decode_best > 0);
    (void)a;
}

TEST(the_decode_floor_ignores_batched_and_mixed_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* Two requests prefilled, then decoding TOGETHER: every decode step here carries two rows, so
     * the floor must still be unset at the end -- a batched step is a different measurement and
     * keying the floor to it would report the batch arriving as drift. */
    uint64_t a = f.submit(160, 64);
    uint64_t b = f.submit(160, 64);
    REQUIRE(f.sch.step() != nullptr);                  /* both prompts */
    f.commit_plain(11);
    f.sch.step_finished();
    for (int i = 0; i < 4; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().n_seq_decode, 2);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        f.commit_plain(12);
        f.sch.step_finished();
    }
    CHECK(f.sch.metrics().steps_decode > 0);            /* decode steps really ran ... */
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);   /* ... and none of them sampled */
    (void)a; (void)b;
}

TEST(the_decode_floor_ignores_a_lone_decode_row_inside_a_mixed_step) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* The subtle case: a mixed step carrying exactly ONE decode row alongside a prefill chunk.
     * n_seq_decode is 1, so a check on the row count alone would sample it -- but that row shared
     * the step with a chunk an order of magnitude dearer, and its time is not the mechanical floor.
     * The phase gate is what excludes it. */
    uint64_t b = f.submit(1000, 64, /*priority=*/1);    /* several chunks */
    uint64_t a = f.submit(160,  64, /*priority=*/0);    /* one chunk, then decodes */
    REQUIRE(f.sch.step() != nullptr);                   /* both prompts */
    f.commit_plain(11);
    f.sch.step_finished();

    REQUIRE(f.sch.step() != nullptr);                   /* a decodes, b still chunking */
    CHECK_EQ(f.sch.plan().n_seq_decode, 1);
    CHECK_EQ(f.sch.plan().phase, RAD_PHASE_MIXED);
    f.commit_plain(12);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);
    (void)a; (void)b;
}

TEST(the_step_path_does_not_allocate) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    for (int i = 0; i < 6; ++i) f.submit(64, 4096);

    /* Warm up: the first steps grow the request's block vector and its output, each sized once.
     * What must be zero is the steady-state step. */
    for (int i = 0; i < 12; ++i) {
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
    }

    g_allocs = 0;
    g_counting = true;
    for (int i = 0; i < 40; ++i) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        Scheduler::StepResult r;
        static int32_t acc[64], tok[64];
        static uint8_t eos[64];
        for (size_t k = 0; k < p.e.size(); ++k) { acc[k] = 0; tok[k] = 5; eos[k] = 0; }
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc; r.token = tok; r.eos = eos;
        f.sch.commit(r);
    }
    g_counting = false;

    /* Zero, not "few": an allocator call here is a declare-phase bug (IMPLEMENTATION.md). */
    CHECK_EQ(g_allocs, 0L);
}

/* The same for a mixed step: decode rows beside a prefill chunk, which the step reorders (decode
 * rows first) and checkpoints. A reordering through a temporary buffer reaches the allocator on
 * every such step. */
TEST(a_mixed_step_does_not_allocate) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());
    for (int i = 0; i < 6; ++i) f.submit(64, 4096);
    for (int i = 0; i < 4; ++i) {
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
    }
    /* A long prompt behind them, admitted before the count starts: the fakes' own bookkeeping
     * (the cache's checkpoint log) is sized here too. */
    f.pc.committed.reserve(64);
    f.submit(3000, 8, /*priority=*/5);             /* ahead of them, so the step reorders */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();

    g_allocs = 0;
    g_counting = true;
    int mixed = 0;
    for (int i = 0; i < 8; ++i) {
        if (!f.sch.step()) break;
        mixed += f.sch.plan().phase == RAD_PHASE_MIXED;
        const StepPlan& p = f.sch.plan();
        Scheduler::StepResult r;
        static int32_t acc[64], tok[64];
        static uint8_t eos[64];
        for (size_t k = 0; k < p.e.size(); ++k) { acc[k] = 0; tok[k] = 5; eos[k] = 0; }
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc; r.token = tok; r.eos = eos;
        f.sch.commit(r);
    }
    g_counting = false;
    CHECK_EQ(mixed, 8);
    CHECK_EQ(g_allocs, 0L);
}

/* The card's block tables are staged as changed entries, so they are checked whole after every
 * build: a paged row reads the manager's table then -1 to the pitch, a state row reads the slot.
 * The run exercises appends at block crossings, rollbacks of rejected drafts, rows reused after a
 * request ends, and a late prefill; a row staged from a stale picture would read a wrong block. */
TEST(the_cards_block_tables_match_the_manager_after_every_build) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    std::vector<uint64_t> ids;
    ids.push_back(f.submit(40, 9));
    ids.push_back(f.submit(70, 200));
    ids.push_back(f.submit(17, 90));
    uint32_t rng = 12345;
    int checked = 0;
    for (int step = 0; step < 80; ++step) {
        if (step == 6) ids.push_back(f.submit(33, 120));
        if (step == 20) ids.push_back(f.submit(90, 60));
        for (uint64_t id : ids) {
            int32_t draft[4] = { 31, 32, 33, 34 };
            (void)f.sch.set_draft(id, draft, 4);
        }
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        for (int gi = 0; gi < b->n_kv_groups; ++gi) {
            const RadKVGroupBatch& g = b->kv[gi];
            for (int64_t i = 0; i < b->n_seq; ++i) {
                const uint64_t seq = p.e[(size_t)i].seq;
                const int32_t* row = g.block_table + i * g.block_table_pitch;
                if (!f.kv.geom.g[(size_t)gi].paged) {
                    CHECK_EQ(row[0], f.kv.state_slot(seq, gi));
                    continue;
                }
                const std::vector<int32_t>& bt = f.kv.block_table(seq, gi);
                for (int64_t j = 0; j < g.block_table_pitch; ++j)
                    CHECK_EQ(row[j], j < (int64_t)bt.size() ? bt[(size_t)j] : -1);
                ++checked;
            }
        }
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
    }
    /* The run reached steady decode with rows to compare, not an empty loop. */
    CHECK(checked > 100);
}

/* A streamed group's prefill stage copies what the step's own sequences' tables name
 * (kvstream/stage.h, "Volume"), so the engine reads the step's sequences off the batch it is about
 * to run: the build's rows in order, for that batch only. */
TEST(a_built_batch_names_its_own_sequences_in_row_order) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());
    f.submit(40, 9);
    f.submit(70, 20);
    int checked = 0;
    for (int step = 0; step < 12; ++step) {
        if (step == 3) f.submit(33, 12);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        const std::vector<uint64_t>* seqs = f.sch.step_seqs(b);
        REQUIRE(seqs != nullptr);
        REQUIRE_EQ(seqs->size(), p.e.size());
        for (size_t i = 0; i < p.e.size(); ++i) CHECK_EQ((*seqs)[i], p.e[i].seq);
        ++checked;
        CHECK_OK(f.commit(0, 0));
    }
    CHECK(checked > 5);
    RadBatch other{};
    CHECK(f.sch.step_seqs(&other) == nullptr);
    CHECK(f.sch.step_seqs(nullptr) == nullptr);
}

/* A streamed group's residency map and ring bookkeeping live in each rank's KV pool,
 * but the manager the builder is given is rank 0's, so its pointers are rank 0's addresses. A
 * two-rank batch must name rank r's own copies or device r resolves through device 0's memory
 * (a page-not-present fault at --tp 2). The host mirror is the exception: RAD_MEM_HOST_MAPPED's
 * device view is its host address on every card, so it is the same pointer on both ranks.
 *
 * The bases are fake and no map is dereferenced: what is under test is the rebasing arithmetic,
 * not two cards. Both ranks use device 0 so the test runs on the host backend. */
TEST(a_two_rank_batch_rebases_stream_maps_into_each_ranks_pool_and_shares_the_mirror) {
    Fixture f;
    f.build();
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
    KVGeom kvg = KVGeom::of(f.prog.kv_groups);
    for (size_t i = 0; i < kvg.g.size(); ++i) {
        const int64_t w = f.kv.state_index_width((int32_t)i);
        if (w > 0) kvg.g[i].sidx_width = w;
    }

    /* Two pools' worth of address space, only ever differenced, never dereferenced. */
    alignas(16) char pool0[0x200], pool1[0x200];
    char* const base[2] = { pool0, pool1 };
    const int64_t res_off = 0x40, sblk_off = 0x80, sst_off = 0xc0, ctl_off = 0x100;
    alignas(16) uint8_t mirror[0x100];
    f.kv.streams = true;
    f.kv.stream.host        = mirror;
    f.kv.stream.res         = (int32_t*)(base[0] + res_off);
    f.kv.stream.slot_block  = (int32_t*)(base[0] + sblk_off);
    f.kv.stream.slot_state  = (int32_t*)(base[0] + sst_off);
    f.kv.stream.ctl         = (int32_t*)(base[0] + ctl_off);
    f.kv.stream.block_bytes = 64;
    f.kv.stream.slots       = 8;

    BatchBuilder bb;
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, nullptr, &f.prog, base[0] });
    rio.push_back({ 0, nullptr, &f.prog, base[1] });
    CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 80));
    StepPlan p;
    p.reset();
    StepEntry e;
    e.seq = 1;
    e.card = 0;
    e.n_tokens = 1;
    e.ctx_len = 79;
    p.e.push_back(e);
    p.n_tok = 1;
    REQUIRE(bb.build(p, f.kv, 0) != nullptr);

    for (int r = 0; r < 2; ++r) {
        const RadBatch* b = bb.batch_for(r);
        REQUIRE(b != nullptr);
        const RadKVStream* s = b->kv[0].stream;
        REQUIRE(s != nullptr);
        CHECK_EQ((void*)s->res,        (void*)(base[r] + res_off));
        CHECK_EQ((void*)s->slot_block, (void*)(base[r] + sblk_off));
        CHECK_EQ((void*)s->slot_state, (void*)(base[r] + sst_off));
        CHECK_EQ((void*)s->ctl,        (void*)(base[r] + ctl_off));
        /* Same mirror address, and the geometry is rank-independent. */
        CHECK_EQ((void*)s->host, (void*)mirror);
        CHECK_EQ(s->block_bytes, (int64_t)64);
        CHECK_EQ(s->slots, (int64_t)8);
        /* Groups that do not stream keep a null view. */
        for (int gi = 1; gi < b->n_kv_groups; ++gi)
            CHECK(b->kv[gi].stream == nullptr);
    }
    bb.fini();
}

/* A request's words for the plugin (Request::plugin_words) ride the prefill step that samples
 * its first token, with that row's index in word 0; a sequence without words, or one not at that
 * step, has an empty range. A request past the declared bound fails the build. */
TEST(a_requests_plugin_words_ride_the_step_that_samples_its_first_token) {
    Fixture f;
    f.build();
    f.prog.plugin_words = 8;
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
    KVGeom kvg = KVGeom::of(f.prog.kv_groups);
    for (size_t i = 0; i < kvg.g.size(); ++i) {
        const int64_t w = f.kv.state_index_width((int32_t)i);
        if (w > 0) kvg.g[i].sidx_width = w;
    }
    BatchBuilder bb;
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, nullptr, &f.prog });
    CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));
    for (uint64_t id = 1; id <= 3; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        CHECK_OK(f.kv.ensure(id, 80));
    }
    Request r1, r2, r3;
    r1.prompt.assign(8, 1);
    r2.prompt.assign(8, 1);
    r2.plugin_words = { 0, 7, 8, 9 };
    r3.prompt.assign(8, 1);
    r3.plugin_words = { 0, 5 };

    StepPlan p;
    p.reset();
    auto entry = [&](uint64_t id, const Request* r, bool last) {
        StepEntry e;
        e.seq = id;
        e.card = (int32_t)id - 1;
        e.req = r;
        e.n_tokens = 8;
        e.ctx_len = 0;
        e.is_prefill = true;
        e.produces_token = last;
        p.e.push_back(e);
        p.n_tok += 8;
    };
    entry(1, &r1, true);     /* no words */
    entry(2, &r2, true);     /* words, sampled this step */
    entry(3, &r3, false);    /* words, but a mid-prompt chunk */
    const RadBatch* b = bb.build(p, f.kv, 0);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_plugin_words, (int64_t)4);
    REQUIRE(b->plugin_cu_h != nullptr);
    CHECK_EQ(b->plugin_cu_h[0], 0);
    CHECK_EQ(b->plugin_cu_h[1], 0);
    CHECK_EQ(b->plugin_cu_h[2], 4);
    CHECK_EQ(b->plugin_cu_h[3], 4);
    /* Sequence 1 took logits row 0, so sequence 2's words open with row 1. */
    CHECK_EQ(b->plugin_words_h[0], 1);
    CHECK_EQ(b->plugin_words_h[1], 7);
    CHECK_EQ(b->plugin_words_h[3], 9);
    CHECK(b->plugin_words != nullptr && b->plugin_cu != nullptr);

    /* No sequence carries words: every field is null. */
    p.reset();
    entry(1, &r1, true);
    b = bb.build(p, f.kv, 1);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_plugin_words, (int64_t)0);
    CHECK(b->plugin_words == nullptr && b->plugin_cu_h == nullptr);

    /* Past the bound: the build fails rather than overrun the slab. */
    r3.plugin_words.assign(9, 0);
    p.reset();
    entry(3, &r3, true);
    CHECK(bb.build(p, f.kv, 2) == nullptr);
    bb.fini();
}

/* tp8 loop D47 (RADIANCE_PP_BG_ROWS=N): a builder told to carve N block-table rows (a BG lane's) reserves the tables of
 * max_seqs - N rows less, lays its span out under another hash (the follower's replay refuses a mismatch), builds a plan of N
 * sequences and refuses one of N + 1 by name; unset (0) and N >= max_seqs are today's carve and hash. The env parse takes 1 .. 256. */
TEST(a_bg_lanes_builder_carves_its_block_tables_for_its_rows_and_refuses_more) {
    Fixture f;
    f.build();
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
    KVGeom kvg = KVGeom::of(f.prog.kv_groups);
    for (size_t i = 0; i < kvg.g.size(); ++i) {
        const int64_t w = f.kv.state_index_width((int32_t)i);
        if (w > 0) kvg.g[i].sidx_width = w;
    }
    REQUIRE(f.cfg.max_seqs > 2);
    const int64_t full = BatchBuilder::device_bytes(f.prog, f.cfg, kvg), two = BatchBuilder::device_bytes(f.prog, f.cfg, kvg, 2);
    CHECK(full > 0 && two > 0 && two < full);
    CHECK_EQ(BatchBuilder::device_bytes(f.prog, f.cfg, kvg, f.cfg.max_seqs), full);
    CHECK_EQ(BatchBuilder::device_bytes(f.prog, f.cfg, kvg, f.cfg.max_seqs + 5), full);
    fprintf(stderr, "    slab %lld B at max_seqs %lld rows, %lld B at 2 rows\n", (long long)full, (long long)f.cfg.max_seqs, (long long)two);
    BatchBuilder ref, bb;
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, nullptr, &f.prog });
    CHECK_OK(ref.init(f.prog, f.cfg, geo, kvg, rio));
    bb.set_bt_rows(2);
    CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));
    CHECK_EQ(bb.bt_rows(), (int64_t)2);
    CHECK_EQ(ref.bt_rows(), f.cfg.max_seqs);
    CHECK(bb.layout_hash() != ref.layout_hash());
    {
        BatchBuilder same;
        same.set_bt_rows(f.cfg.max_seqs);   /* not below max_seqs: today's carve and hash */
        CHECK_OK(same.init(f.prog, f.cfg, geo, kvg, rio));
        CHECK_EQ(same.layout_hash(), ref.layout_hash());
        same.fini();
    }
    for (uint64_t id = 1; id <= 3; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        CHECK_OK(f.kv.ensure(id, 80));
    }
    Request r;
    r.prompt.assign(8, 1);
    StepPlan p;
    auto plan = [&](int n) {
        p.reset();
        for (int i = 1; i <= n; ++i) {
            StepEntry e;
            e.seq = (uint64_t)i;
            e.card = i - 1;
            e.req = &r;
            e.n_tokens = 8;
            e.ctx_len = 0;
            e.is_prefill = true;
            e.produces_token = true;
            p.e.push_back(e);
            p.n_tok += 8;
        }
    };
    plan(2);
    CHECK(bb.build(p, f.kv, 0) != nullptr);
    CHECK(ref.build(p, f.kv, 0) != nullptr);
    plan(3);
    CHECK(bb.build(p, f.kv, 1) == nullptr);   /* refused: more sequences than its block tables hold */
    CHECK(ref.build(p, f.kv, 1) != nullptr);
    bb.fini();
    ref.fini();
    /* tp8 loop D47b: a decode lane's rows = its cards + M below max_seqs, else max_seqs; RADIANCE_PP_LANE_ROWS takes 1 .. 256 */
    CHECK_EQ(LaneSet::lane_rows(16, 4, 32), (int64_t)20);
    CHECK_EQ(LaneSet::lane_rows(16, 0, 32), (int64_t)32);
    CHECK_EQ(LaneSet::lane_rows(16, 16, 32), (int64_t)32);
    CHECK_EQ(LaneSet::lane_rows(3, 1, 10), (int64_t)4);
    struct Env { ~Env() { unsetenv("RADIANCE_PP_BG_ROWS"); unsetenv("RADIANCE_PP_LANE_ROWS"); } } env;
    for (const auto& [v, want] : std::vector<std::pair<const char*, int64_t>>{ { "4", 4 }, { "256", 256 }, { "0", 0 }, { "999", 0 }, { "", 0 } }) {
        setenv("RADIANCE_PP_LANE_ROWS", v, 1);
        CHECK_EQ(LaneSet::lane_rows_env(), want);
    }
    for (const auto& [v, want] : std::vector<std::pair<const char*, int64_t>>{ { "4", 4 }, { "1", 1 }, { "256", 256 }, { "0", 0 },
                                                                               { "257", 0 }, { "x", 0 }, { "-3", 0 } }) {
        setenv("RADIANCE_PP_BG_ROWS", v, 1);
        CHECK_EQ(LaneSet::bg_rows_env(), want);
    }
}

/* A sequence that leaves a row and returns to it is restaged whole. Its change marks are taken
 * wherever it is written, so the old row's copy goes stale; restaging only later changes would
 * keep entries a rollback replaced earlier. The scheduler's own passes reorder rows, so this
 * drives the builder directly with explicit row orders. */
TEST(a_sequence_back_in_a_row_it_left_is_restaged_whole) {
    Fixture f;
    f.build();
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
    KVGeom kvg = KVGeom::of(f.prog.kv_groups);
    for (size_t i = 0; i < kvg.g.size(); ++i) {
        const int64_t w = f.kv.state_index_width((int32_t)i);
        if (w > 0) kvg.g[i].sidx_width = w;
    }
    BatchBuilder bb;
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, nullptr, &f.prog });
    CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));

    for (uint64_t id = 1; id <= 3; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        CHECK_OK(f.kv.ensure(id, 80));
    }
    StepPlan p;
    auto build = [&](std::initializer_list<uint64_t> rows) {
        p.reset();
        int32_t card = 0;
        for (uint64_t id : rows) {
            StepEntry e;
            e.seq = id;
            e.card = card++;
            e.n_tokens = 1;
            e.ctx_len = (int32_t)f.kv.seqs[id].n - 1;
            p.e.push_back(e);
            p.n_tok += 1;
        }
        const RadBatch* b = bb.build(p, f.kv, 0);
        CHECK(b != nullptr);
        if (!b) return;
        const RadKVGroupBatch& g = b->kv[0];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            const std::vector<int32_t>& bt = f.kv.block_table(p.e[(size_t)i].seq, 0);
            const int32_t* row = g.block_table + i * g.block_table_pitch;
            for (int64_t j = 0; j < g.block_table_pitch; ++j)
                CHECK_EQ(row[j], j < (int64_t)bt.size() ? bt[(size_t)j] : -1);
        }
    };

    build({ 1, 2, 3 });
    /* Sequence 3's last three blocks are handed to another sequence and replaced: its table keeps
     * its length and changes below it. */
    CHECK_OK(f.kv.rollback(3, 32));
    CHECK_OK(f.kv.add_sequence(4));
    CHECK_OK(f.kv.ensure(4, 48));
    CHECK_OK(f.kv.ensure(3, 80));
    build({ 1, 3 });      /* 3 staged in row 1, its marks taken there */
    build({ 1, 2, 3 });   /* and back in row 2, which still holds its old blocks */
    bb.fini();
}

/* Two conv groups, each derivation reading its own window. With one shared slot array and window,
 * only the last-declared RAD_KV_CONV group would be filled and the other's row would stay -1.
 * (Today only `conv_state_index` reads these, and no architecture calls it; conv kernels use the
 * per-group RadKVGroupBatch::state_index.) The widths differ because `conv_state_index` returns
 * `slot * window + cursor % window`, and only differing depths expose the wrong group's.
 */
TEST(each_conv_group_derives_against_its_own_window) {
    auto two_groups = [](Fixture& f) {
        KVGroupInfo conv2;
        conv2.name = "conv_deep";
        conv2.decl.kind = RAD_KV_CONV;
        conv2.decl.conv_width = 10;
        f.prog.kv_groups.push_back(conv2);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    };

    /* ---- an unnamed group is refused, not guessed at ----------------------------------------- */
    {
        Fixture f;
        f.build();                       /* declares `conv_idx` with no group -- fine at one */
        two_groups(f);
        /* The refusal logs at Error by design; the line is expected here. */
        CHECK_EQ(f.start(), RAD_E_INVAL);
    }

    /* ---- named, and each reads its own ------------------------------------------------------- */
    Fixture f;
    f.build();
    two_groups(f);
    /* Group 2 is the width-4 conv build() declared; group 3 is the deep one just added. */
    for (BufferInfo& b : f.prog.buffers) if (b.name == "conv_idx") b.decl.kv = 2;
    f.add_derived("conv_idx_deep", "conv_state_index", 256, 3);
    CHECK_OK(f.start());

    CHECK(f.submit(4) != 0);
    CHECK(f.sch.step() != nullptr);

    const int32_t* a = f.derived("conv_idx");
    const int32_t* c = f.derived("conv_idx_deep");
    CHECK(a != nullptr);
    CHECK(c != nullptr);
    /* Neither is -1. Under one shared slot array the group that is not last-declared gets -1 for
     * every sequence, so this keeps the derivation working for the first architecture to call it. */
    if (a && c) {
        CHECK(a[0] >= 0);
        CHECK(c[0] >= 0);
    }
}

/* ================================================================== the derivation registry
 *
 * A derivation named in a schema but missing here is refused by name at declare, with the known
 * list printed (spec §2.3); otherwise a kernel would read whatever the arena last held. So every
 * row is reachable by name and has a function, and an unknown name resolves to -1, not row 0.
 */
TEST(every_derivation_is_reachable_by_name_and_an_unknown_one_is_not) {
    const int n = derive_count();
    CHECK(n > 0);

    for (int i = 0; i < n; ++i) {
        const DeriveRow* r = derive_row(i);
        CHECK(r != nullptr);
        if (!r) continue;
        CHECK(r->name && *r->name);
        CHECK(r->fn != nullptr);          /* a row with no function would resolve and then write
                                           * nothing, the failure above */
        CHECK(r->doc && *r->doc);         /* it is printed in the refusal's known list */
        CHECK_EQ(derive_index(r->name), i);
    }

    /* Names are unique, or derive_index would answer with the first row and the second could
       never be selected. */
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            CHECK(std::strcmp(derive_row(i)->name, derive_row(j)->name) != 0);

    /* -1 for a name the core does not know, not 0, which is a valid index. */
    CHECK_EQ(derive_index("not_a_derivation"), -1);
    CHECK_EQ(derive_index(""), -1);
    CHECK_EQ(derive_index(nullptr), -1);
    /* A near miss on a real name is still a miss: a typo is reported, not approximated. */
    CHECK_EQ(derive_index("seq_start_idx"), -1);

    /* An index outside the table gives null, not a read past it. */
    CHECK(derive_row(-1) == nullptr);
    CHECK(derive_row(n) == nullptr);
    CHECK(derive_row(n + 1000) == nullptr);
}

/* ================================================================== request lifetime */

/* A preempted request loses its draft window. The sampler sizes rows from Request::n_draft, and
 * the request returns as a one-row prefill; a stale window of four would stage five rows and shift
 * every later request onto its neighbour's row. */
TEST(a_preempted_request_comes_back_with_no_draft_window) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64, /*priority=*/0);
    uint64_t b = f.submit(160, 64, /*priority=*/5);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_EQ(f.sch.find(a)->n_draft, 4);

    /* Both need a new block at 160 and the pool has none: A, the lower priority, is taken. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
    CHECK(f.sch.find(a)->draft == nullptr);
    f.commit_plain();

    /* When it is admitted again, the window it carries is its entry's. */
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
}

/* Each terminal id is handed over once: take_completed swaps the list out under the lock. A
 * reference into scheduler storage that nothing cleared would have every finished request walked
 * again on every step, without the lock, while an HTTP thread's cancel appends to it. */
TEST(terminal_requests_are_taken_once) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t a = f.submit(8, /*max_tokens=*/1);
    uint64_t b = f.submit(8, /*max_tokens=*/64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();                               /* A reaches its one token */
    CHECK_OK(f.sch.cancel(b));

    std::vector<uint64_t> done;
    f.sch.take_completed(&done);
    CHECK_EQ((int)done.size(), 2);
    CHECK(done[0] == a);
    CHECK(done[1] == b);
    f.sch.take_completed(&done);
    CHECK_EQ((int)done.size(), 0);
}

/* Every row names one of the card's step slots. Slots are taken on arrival, so queued requests
 * and arrivals during a pending reap are numbered past max_seqs; the advance indexes max_seqs
 * StepSlots by the row's card index, never by that slot number. */
TEST(every_row_names_one_of_the_cards_step_slots) {
    Fixture f;
    f.build();
    f.cfg.max_seqs = 2;
    CHECK_OK(f.start());

    for (int i = 0; i < 6; ++i) CHECK(f.submit(8, /*max_tokens=*/1) != 0);
    for (int round = 0; round < 3; ++round) {
        CHECK(f.sch.step() != nullptr);
        const StepPlan& p = f.sch.plan();
        CHECK_EQ((int)p.e.size(), 2);
        CHECK(p.e[0].card >= 0 && p.e[0].card < 2);
        CHECK(p.e[1].card >= 0 && p.e[1].card < 2);
        CHECK(p.e[0].card != p.e[1].card);
        /* Nothing was reaped, so the later rounds' slots did run past max_seqs. */
        if (round > 0) CHECK(p.e[0].slot >= 2 && p.e[1].slot >= 2);
        f.commit_plain();                           /* each reaches its one token */
    }
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
}

/* The context pass mirrors the trunk step row for row: the drafter reads hidden states
 * positionally, so a sequence the commit finished keeps its rows (as padding that stores nothing),
 * or every later sequence would draft from its neighbour's states. */
/* The load a block drafter's short block is chosen by counts the running sequences the step just
 * run produced a token for, not one it finished; the choice switches on at the threshold and off
 * only two below it. */
TEST(a_block_drafters_short_block_follows_the_decoded_sequences_with_hysteresis) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());
    uint64_t a = f.submit(160, /*max_tokens=*/1);
    f.submit(160, 64);
    f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 3);
    f.commit_plain();
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_EQ(f.sch.decoded_seqs(), 2);

    bool on = false;
    const int64_t from = 16;
    const int64_t load[] = { 15, 16, 15, 14, 15, 13, 14, 16 };
    const bool want[]    = { 0,  1,  1,  0,  0,  0,  0,  1 };
    for (size_t i = 0; i < sizeof load / sizeof load[0]; ++i) {
        on = Scheduler::short_block_on(on, load[i], from);
        CHECK_EQ(on, want[i]);
    }
}

TEST(a_block_drafter_context_pass_keeps_the_rows_of_a_finished_sequence) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, /*max_tokens=*/1);
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    f.commit_plain();
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    CHECK(rows[0].seq == b);

    const RadBatch* cb = f.sch.block_context(1);
    REQUIRE(cb != nullptr);
    CHECK_EQ(cb->n_seq, 2);
    CHECK_EQ(cb->n_tok, 320);
    CHECK_EQ(cb->cu_seqlens[1], 160);
    CHECK_EQ(cb->kv[0].slot_mapping[0], -1);        /* A's rows store nothing */
    CHECK(cb->kv[0].slot_mapping[160] >= 0);        /* B's rows are B's */

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK_EQ(qb->n_seq, 1);
    f.sch.block_end();
}

/* DRAFTER track round 11, lever DHD: a drafter whose every row proposes (block = depth, DSpark) cut to D < block rows
 * reads the FIRST D rows of each block into its head (block_begin's `front`), the anchor first; without `front` the
 * last D rows, as before; at D = block the two are the same rows. */
TEST(a_block_query_cut_to_the_first_rows_reads_each_block_from_its_anchor) {
    for (int mode = 0; mode < 3; ++mode) {   /* 0: front, depth 3 of 5; 1: not front, depth 3; 2: front, depth 5 */
        Fixture f;
        f.build();
        f.cfg.n_spec = 5;
        CHECK_OK(f.start());
        uint64_t a = f.submit(96, 64);
        uint64_t b = f.submit(80, 64);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
        const int depth = mode == 2 ? 5 : 3;
        Scheduler::BlockRow rows[4];
        CHECK_EQ(f.sch.block_begin(/*block=*/5, depth, /*window=*/0, /*mask=*/99, rows, 4, /*on_card=*/false,
                                   /*front=*/mode != 1), 2);
        CHECK(rows[0].seq == a);
        CHECK(rows[1].seq == b);
        REQUIRE(f.sch.block_context(1) != nullptr);
        const RadBatch* qb = f.sch.block_query(1);
        REQUIRE(qb != nullptr);
        CHECK_EQ(qb->n_seq, 2);
        CHECK_EQ(qb->n_tok, 10);
        REQUIRE_EQ(qb->n_out, 2 * depth);
        for (int s = 0; s < 2; ++s)
            for (int j = 0; j < depth; ++j)
                CHECK_EQ(qb->out_ids[s * depth + j], s * 5 + (mode == 1 ? 5 - depth : 0) + j);
        f.sch.block_end();
    }
}

/* A restore the pool cannot hold whole waits one step while a closed sequence still holds blocks
 * until the next commit: its conversation's VRAM may be the room the restore needs. It waits once,
 * and with nothing deferred it is not asked to come back whole. */
TEST(a_restore_waits_one_step_for_a_deferred_release_before_it_is_cut) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());
    f.pc.decline_whole = true;

    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.pc.restores_whole.size(), (size_t)1);
    CHECK(!f.pc.restores_whole[0]);                  /* nothing deferred: no wait asked */

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4), 1);
    CHECK(f.sch.block_context(1) != nullptr);
    CHECK_OK(f.sch.cancel(a));
    f.sch.block_end();
    REQUIRE(f.sch.has_deferred());

    uint64_t b = f.submit(200, 8);
    (void)f.sch.step();
    REQUIRE_EQ(f.pc.restores_whole.size(), (size_t)2);
    CHECK(f.pc.restores_whole[1]);
    CHECK_EQ(f.sch.find(b)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().running, 0);

    /* Once only: the next attempt takes what the pool holds, deferred release or not. */
    CHECK(f.sch.step() != nullptr);
    REQUIRE_EQ(f.pc.restores_whole.size(), (size_t)3);
    CHECK(!f.pc.restores_whole[2]);
    CHECK_EQ(f.sch.metrics().running, 1);
}

/* The wait is once per request, not once per slot. Two conversations taking turns reuse the slots
 * their earlier turns held, so a turn landing in a slot whose last tenant waited must still wait. */
TEST(a_request_in_a_reused_slot_still_waits_for_a_deferred_release) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());
    f.pc.decline_whole = true;
    Scheduler::BlockRow rows[4];
    /* Closes a running sequence while a drafter pass still names its blocks. */
    auto close_deferred = [&](uint64_t id) {
        CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4), 1);
        CHECK(f.sch.block_context(1) != nullptr);
        CHECK_OK(f.sch.cancel(id));
        f.sch.block_end();
        REQUIRE(f.sch.has_deferred());
    };

    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    close_deferred(a);

    uint64_t b = f.submit(200, 64);
    (void)f.sch.step();                               /* waits a step */
    CHECK(f.sch.step() != nullptr);                   /* then is admitted */
    f.commit_plain();
    REQUIRE_EQ(f.pc.restores_whole.size(), (size_t)3);
    CHECK(f.pc.restores_whole[1]);
    REQUIRE(!f.sch.has_deferred());

    uint64_t x = f.submit(120, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.sch.metrics().running, 2);
    CHECK_OK(f.sch.cancel(b));                        /* nothing in flight: released at once */
    CHECK_OK(f.sch.reap(b));                          /* b's slot is the next one handed out */
    close_deferred(x);

    const size_t before = f.pc.restores_whole.size();
    uint64_t c = f.submit(240, 8);
    (void)f.sch.step();
    REQUIRE_EQ(f.pc.restores_whole.size(), before + 1);
    CHECK(f.pc.restores_whole.back());
    CHECK_EQ(f.sch.find(c)->n_computed, 0);
}

/* A cancel between two drafter passes (built one at a time, lock released between) must not free
 * the blocks: the query pass would get a block-table row of -1 under a non-zero length. */
TEST(a_cancel_between_drafter_passes_keeps_the_blocks_until_the_passes_are_built) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4), 1);
    CHECK(f.sch.block_context(1) != nullptr);

    CHECK_OK(f.sch.cancel(a));
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK(f.kv.has_sequence(a));                     /* not freed under the passes */

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK(qb->kv[0].block_table[0] >= 0);            /* a real block, not -1 */
    f.sch.block_end();

    /* The drafter is done with it, so it is released at the next commit, or here for an idle
     * engine that runs none. */
    CHECK(f.sch.has_deferred());
    f.sch.release_deferred();
    CHECK(!f.kv.has_sequence(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK(!f.sch.has_deferred());
    CHECK_OK(f.sch.reap(a));
    CHECK(f.sch.find(a) == nullptr);
}

/* On the card the drafter's rows are taken before the commit and released after it, so a cancel
 * between the two passes is still held when the commit retires the step. The commit must not free
 * the blocks while block_begin's rows name them; block_end hands them to the next sweep. */
TEST(a_cancel_between_card_drafter_passes_keeps_the_blocks_past_the_commit) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4, /*on_card=*/true), 1);
    CHECK(f.sch.block_context(1) != nullptr);

    CHECK_OK(f.sch.cancel(a));
    CHECK(f.kv.has_sequence(a));

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK(qb->kv[0].block_table[0] >= 0);
    f.commit_plain();
    CHECK(f.kv.has_sequence(a));                     /* the rows still name it */
    f.sch.block_end();

    CHECK(f.sch.has_deferred());
    f.sch.release_deferred();
    CHECK(!f.kv.has_sequence(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_OK(f.sch.reap(a));
    CHECK(f.sch.find(a) == nullptr);
}

/* A failed admission gives back its checkpoint hit. The slot is named by id only and may be
 * reassigned between attempts, so trusting the old id would restore another sequence's recurrent
 * state. The next attempt looks the prompt up again, and the admission is counted once. */
TEST(a_failed_admission_gives_back_its_checkpoint_hit) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t r = f.submit(160, 64);  /* 10 blocks, and a miss */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.pc.lookups, 1);

    f.pc.seed(256);                  /* 16 blocks the cache holds */
    f.pc.lin = 256;
    f.pc.src = 3;

    /* A adopts the 16 cached blocks and needs 16 more for its first chunk; 13 are left. */
    uint64_t a = f.submit(600, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_EQ(f.pc.lookups, 2);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);          /* the hit went back */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 0);
    CHECK_EQ(f.sch.metrics().prompt_tokens, 160);   /* not counted until it is admitted */
    f.commit_plain();

    /* Meanwhile the cache's snapshot at 256 moved to another slot. */
    f.pc.src = 5;
    CHECK_OK(f.sch.cancel(r));
    CHECK_OK(f.sch.reap(r));

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.pc.lookups, 3);
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 5);
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.prompt_tokens, 160 + 600);
    CHECK_EQ(m.attn_cached_tokens, 256);
    CHECK_EQ(m.linear_cached_tokens, 256);
}

/* A refused lookahead stops drafting, not the row: the history pass writes the draft head's K and
 * V for positions the sequence already holds, and a dropped row would leave them unwritten for
 * good, since the next step's history starts at its own context. */
TEST(a_refused_lookahead_still_runs_the_history_pass) {
    Fixture f;
    f.build(/*blocks=*/10, /*states=*/64);          /* exactly the prompt; nothing to look ahead into */
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 0);   /* nobody drafts */
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK_EQ(hb->n_tok, 160);                             /* but the history is written */
    f.sch.serial_end(0);
}

/* Draft round 1 rides the history pass. A drafting verifier owes the head's history at all
 * 1 + n_spec verified positions, and the card repeats the last committed one past it, so the
 * run's last row is round 1's input: the pass names that row, and round 1 has no pass of its own. */
TEST(the_history_pass_carries_draft_round_one) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_spec, 4);

    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)a);

    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK(hb->draft_pass < 0);
    CHECK_EQ(hb->n_tok, 5);                    /* every verified position, the last included */
    CHECK_EQ(hb->n_draft_out, 1);
    CHECK_EQ(hb->draft_out_ids[0], 4);         /* the run's last row */

    CHECK(f.sch.serial_batch(1, 1) == nullptr);
    const RadBatch* r2 = f.sch.serial_batch(2, 1);
    REQUIRE(r2 != nullptr);
    CHECK_EQ(r2->draft_pass, 2);
    CHECK_EQ(r2->n_tok, 1);
    CHECK_EQ(r2->n_draft_out, 0);
    f.sch.serial_end(2);
}

/* The serial driver's two shapes over one verify step. A head with its own cache owes the history
 * of every verified position and steps one position per round; a head reading the trunk's cache
 * (rad_drafter_shared_kv) owes nothing, runs one row per drafting sequence and does not step.
 * Qwen's MTP head uses the first shape, which must not change. */
namespace {
struct SerialShape {
    int     hist_tok = 0, hist_out = 0, hist_dout = 0;
    int32_t hist_ctx = 0, hist_row = 0;
    int32_t round_ctx[4] = {0, 0, 0, 0};
};

void serial_shape(bool shared, SerialShape& r) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    f.prog.drafter_shared_kv = shared;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_spec, 4);

    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 1);
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK(hb->draft_pass < 0);
    r.hist_tok  = (int)hb->n_tok;
    r.hist_out  = (int)hb->n_out;
    r.hist_dout = (int)hb->n_draft_out;
    r.hist_ctx  = hb->ctx_lens[0];
    r.hist_row  = hb->out_ids[0];
    for (int rd = 2; rd <= 3; ++rd) {
        const RadBatch* rb = f.sch.serial_batch(rd, 1);
        REQUIRE(rb != nullptr);
        CHECK_EQ(rb->n_tok, 1);
        r.round_ctx[rd] = rb->ctx_lens[0];
    }
    f.sch.serial_end(3);
}
}  /* namespace */

TEST(a_head_with_a_cache_of_its_own_is_staged_as_it_always_was) {
    SerialShape o;
    serial_shape(false, o);
    CHECK_EQ(o.hist_tok, 5);                    /* every verified position */
    CHECK_EQ(o.hist_out, 5);                    /* one trunk row each */
    CHECK_EQ(o.hist_dout, 1);
    CHECK_EQ(o.hist_ctx, 160);                  /* the history starts at the first verified one */
    CHECK_EQ(o.hist_row, 0);                    /* the verify step's first row */
    CHECK_EQ(o.round_ctx[2], 165);              /* the furthest round 2 can sit: one past the last */
    CHECK_EQ(o.round_ctx[3], 166);              /* ...and the head steps one position a round */
}

TEST(a_head_with_no_cache_of_its_own_runs_one_row_and_stays_put) {
    SerialShape s;
    serial_shape(true, s);
    CHECK_EQ(s.hist_tok, 1);                    /* round 1 alone */
    CHECK_EQ(s.hist_out, 1);
    CHECK_EQ(s.hist_dout, 1);
    CHECK_EQ(s.hist_ctx, 164);                  /* the furthest the committed position can be */
    CHECK_EQ(s.hist_row, 0);                    /* the run's first trunk row; the card adds acc */
    CHECK_EQ(s.round_ctx[2], 164);              /* and every round stays there */
    CHECK_EQ(s.round_ctx[3], 164);
}

/* A head with no cache of its own is owed no history: a prefill chunk that drafts nothing runs no
 * pass for it, while a head with its own cache is given the whole chunk. */
TEST(a_head_with_no_cache_of_its_own_runs_no_pass_when_nothing_drafts) {
    for (int shared = 0; shared < 2; ++shared) {
        Fixture f;
        f.build(/*blocks=*/10, /*states=*/64);      /* exactly the prompt: nothing to look ahead into */
        f.cfg.n_spec = 4;
        f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 4;
        f.prog.drafter_shared_kv = shared != 0;
        CHECK_OK(f.start());
        f.submit(160, 64);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
        int rounds = 4;
        Scheduler::SerialRow rows[4];
        CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 0);
        const RadBatch* hb = f.sch.serial_batch(0, 1);
        if (shared) {
            CHECK(hb == nullptr);
        } else {
            REQUIRE(hb != nullptr);
            CHECK_EQ(hb->n_tok, 160);
        }
        f.sch.serial_end(0);
    }
}

/* A step is issued ahead of a commit only once the commit cannot change the request's
 * speculation: a request the sampler has not started (grammar unchecked) is not issued ahead; one
 * with a rewinding grammar, or none, is. */
TEST(a_step_is_issued_ahead_only_once_the_requests_speculation_is_fixed) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    const uint64_t id = f.submit(8, 64);
    CHECK(f.sch.step() != nullptr);                 /* the prompt */
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                 /* the first decode */
    CHECK(f.sch.step_next() == nullptr);            /* the sampler has not started it */

    Request* r = f.sch.find(id);
    REQUIRE(r != nullptr);
    r->sp.grammar = "root ::= \"a\"";
    r->speculation_fixed = true;                    /* a grammar whose backend rewinds */
    CHECK(f.sch.step_next() != nullptr);
}

/* No step is issued ahead behind one that ends every request it carries: its rows would
 * belong to finished requests, and a request arriving after the last token would wait out its
 * forward pass. The verify ends a request most likely when its drafts hold an end-of-generation
 * id, certainly when its first token is the request's last (max_tokens), and possibly when that
 * token and its accepted drafts reach max_tokens. An ignore_eos request's drafted end ends
 * nothing, nor do drafts without one, and a request that even every draft accepted leaves short
 * of max_tokens goes on. The refusal reserves nothing for the step it does not issue. */
TEST(no_step_is_issued_ahead_behind_one_that_ends_its_requests) {
    constexpr int32_t kEog = 2;
    int32_t ends[4] = { 21, kEog, 23, 24 }, goes_on[4] = { 21, 22, 23, 24 };
    /* One request: its prompt's token committed, then the verify of `draft`. */
    auto ahead = [&](const int32_t* draft, int max_tokens, bool ignore_eos) -> bool {
        Fixture f;
        f.build();
        f.cfg.n_spec = 4;
        CHECK_OK(f.start());
        f.sch.set_eog({ kEog, 3 });
        const uint64_t id = f.submit(160, max_tokens, 0, ignore_eos);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        CHECK_OK(f.sch.set_draft(id, draft, 4));
        const bool verify = f.sch.step() != nullptr;
        CHECK(verify);
        if (!verify) return false;
        CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
        f.sch.find(id)->speculation_fixed = true;
        f.sch.expect_draft(id, 4);
        const int64_t free_before = f.kv.free_blocks(0);
        const bool issued = f.sch.step_next() != nullptr;
        if (!issued) CHECK_EQ(f.kv.free_blocks(0), free_before);
        return issued;
    };
    CHECK(ahead(goes_on, 64, false));
    CHECK(!ahead(ends, 64, false));               /* a drafted end */
    CHECK(ahead(ends, 64, true));                 /* ignore_eos: an end-of-text like any token */
    CHECK(!ahead(goes_on, 2, false));             /* one token committed, its next the last */
    CHECK(!ahead(goes_on, 6, false));             /* 1 + its token + 4 accepted drafts reach it */
    CHECK(ahead(goes_on, 7, false));              /* all four accepted leave it a token short */

    /* Two requests: one ends, the other goes on, so the step is issued ahead for both. */
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    f.sch.set_eog({ kEog });
    const uint64_t a = f.submit(160, 64), b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, ends, 4));
    CHECK_OK(f.sch.set_draft(b, goes_on, 4));
    REQUIRE(f.sch.step() != nullptr);
    f.sch.find(a)->speculation_fixed = true;
    f.sch.find(b)->speculation_fixed = true;
    const RadBatch* nb = f.sch.step_next();
    REQUIRE(nb != nullptr);
    CHECK_EQ(nb->n_seq, 2);
}

/* The context bounds a sequence whatever max_tokens says, "no limit" included. Positions past the
 * last one the deployment can address have no rotary row or block-table column; a sequence
 * running into them would corrupt memory instead of stopping. */
TEST(a_sequence_stops_at_the_context_whatever_max_tokens_says) {
    Fixture f;
    f.build();
    f.cfg.max_ctx = 170;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, /*max_tokens=*/0);        /* unlimited */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int guard = 0; guard < 32 && f.sch.find(a)->state == ReqState::Running; ++guard) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        const StepEntry& e = f.sch.plan().e[0];
        CHECK(e.ctx_len + e.n_tokens <= 170);           /* no position at or past the end */
        /* Every draft accepted, the most a step can emit, so the bound must cut it short. */
        CHECK_OK(f.commit(9, 0, [](const StepEntry& x) { return x.n_spec; }));
    }
    const Request* ra = f.sch.find(a);
    CHECK_EQ((int)ra->state, (int)ReqState::Finished);
    CHECK_EQ(ra->finish_reason, std::string("length"));
    CHECK_EQ((int)(ra->prompt.size() + ra->output.size()), 170);

    /* A prompt that fills the context has nothing to generate into and is never run. */
    uint64_t full = f.submit(170, 8);
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ((int)f.sch.find(full)->state, (int)ReqState::Failed);

    /* Asking for more than the context holds reserves only what it can hold. */
    uint64_t big = f.submit(8, 2147483647);
    CHECK(f.sch.find(big)->output.capacity() <= (size_t)(170 + RAD_SCHED_MAX_SPEC));
}

/* The cache's reserve is measured against what is held, not what is backed. An elastic pool that
 * has handed its unused top back to the card reads almost no free blocks however little is held,
 * and trimming against that would drain the whole cache after every request. */
TEST(the_cache_trim_does_not_mistake_released_vram_for_occupancy) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.evict_grant = 1 << 20;
    f.kv.unbacked = 30;                              /* the pool gave 30 blocks back */

    f.submit(160, /*max_tokens=*/1);                 /* holds 10 of 40 */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();                                /* finishes and publishes */
    CHECK_EQ((int)f.pc.published.size(), 1);
    CHECK_EQ(f.pc.evict_calls, 0);
    CHECK(f.pc.trims.empty());
}

/* When live sequences hold more than the reserve allows, the trim asks for the shortfall once.
 * Nothing the cache can give closes it entirely, and a loop evicting until it did would also take
 * the conversation this close has just published. */
TEST(the_cache_trim_asks_for_the_shortfall_and_not_for_the_whole_cache) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.evict_grant = 1 << 20;

    f.submit(160, /*max_tokens=*/1);                 /* 10 blocks, finishes on the first commit */
    f.submit(176, /*max_tokens=*/64);                /* 11 blocks each, still running after it */
    f.submit(176, /*max_tokens=*/64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.held_blocks(0), 32ll);             /* 8 free against a reserve of 10 */
    f.commit_plain();
    CHECK_EQ((int)f.pc.published.size(), 1);
    REQUIRE(f.pc.trims.size() == 1u);
    CHECK_EQ(f.pc.trims[0], 2ll);
    CHECK_EQ(f.pc.evict_calls, 0);
}

/* The cache gives way to a new request as it does to a running one. With nothing running no other
 * path takes blocks back from the cache, so a pool full of finished conversations would keep the
 * request waiting forever. */
TEST(an_admission_short_of_blocks_takes_them_from_the_cache) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.seed(240);                                  /* 15 of 20 held by finished conversations */
    f.pc.attn = 0;                                   /* and none of them is this prompt */
    f.pc.evict_frees = true;
    f.pc.evict_grant = 1 << 20;
    const uint64_t a = f.submit(160, 8);             /* 10 blocks, 5 free */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK(f.pc.evict_calls >= 1);
    CHECK(f.pc.holds.empty());
}

/* A refused admission holds what its restore brought back and lets go once the request is gone,
 * so the next attempt finds the chain on the card instead of copying it again. */
TEST(a_refused_admission_holds_its_chain_until_the_request_goes) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.seed(240);
    f.pc.attn = 0;
    const uint64_t a = f.submit(160, 8);             /* the cache gives nothing back */
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    REQUIRE_EQ(f.pc.holds.size(), 1u);
    CHECK_EQ(f.pc.holds[0], a);
    CHECK_OK(f.sch.cancel(a));
    CHECK(std::find(f.pc.releases.begin(), f.pc.releases.end(), a) != f.pc.releases.end());
}

/* An engine that has stopped answers every request it held and refuses new ones, rather than
 * leaving their connections waiting on a step that will never come. */
TEST(shutdown_fails_what_is_queued_and_refuses_what_follows) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    uint64_t b = f.submit(160, 64);

    f.sch.shutdown("stopped");
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Failed);
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK(f.submit(8, 8) == 0);
    CHECK(f.sch.step() == nullptr);
}

/* Each of those requests says what stopped it: the sink carries the status and the sentence the
 * server puts in the 500, rather than a generic "ok". */
TEST(shutdown_fails_each_sink_with_the_status_that_stopped_the_engine) {
    server::Sink running, waiting, late;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    CHECK(f.submit_to(&running, 160, 64) != 0);
    CHECK(f.sch.step() != nullptr);
    CHECK(f.submit_to(&waiting, 160, 64) != 0);

    f.sch.shutdown("the engine stopped", RAD_E_UNSUPPORTED);
    CHECK(f.submit_to(&late, 8, 8) == 0);
    for (server::Sink* s : { &running, &waiting, &late }) {
        CHECK(s->done());
        CHECK_EQ((int)s->finish_reason(), (int)server::Finish::Error);
        CHECK_EQ(s->status(), (int)RAD_E_UNSUPPORTED);
        CHECK_EQ(std::string(s->why() ? s->why() : ""), std::string("the engine stopped"));
    }
}

/* A request's own draft counts and admission time reach its sink, so a response can report them
 * (aggregate counters cannot be split between overlapping requests). The counts are the same
 * proposed/accepted numbers /metrics sums. */
TEST(a_sink_carries_its_requests_admission_and_draft_counts) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    const uint64_t id = f.submit_to(&sink, 40, 60);
    CHECK(id != 0);
    CHECK(sink.t_admitted_ns() < 0);
    int64_t drafted = 0, accepted = 0;
    uint32_t rng = 99;
    for (int step = 0; step < 40 && !sink.done(); ++step) {
        int32_t draft[4] = { 31, 32, 33, 34 };
        (void)f.sch.set_draft(id, draft, 4);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        CHECK(sink.t_admitted_ns() >= 0);
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) {
            const int32_t acc = random_accept(rng, e);
            if (!e.is_prefill && e.n_spec > 0) {
                drafted += e.n_spec;
                accepted += acc;
            }
            return acc;
        }));
    }
    CHECK(drafted > 0);
    CHECK_EQ(sink.n_drafted(), drafted);
    CHECK_EQ(sink.n_draft_accepted(), accepted);
}

/* ================================================================== the step issued ahead */
/* A scripted model both pipelines follow. Entry i of step t accepts ahead_acc(seq, t) of its
 * drafts, then draws ahead_tok(seq, t); the drafts verified at step t are ahead_draft(seq, t, j).
 * Every token is distinct from every other and from the prompts, so prompt lookup proposes
 * nothing and a misplaced token cannot match by accident. */
namespace {
uint32_t ahead_mix(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
int32_t ahead_tok(uint64_t seq, int t) { return 20000 + (int32_t)seq * 1000 + t; }
int32_t ahead_draft(uint64_t seq, int t, int j) {
    return 100000 + (int32_t)seq * 10000 + t * 10 + j;
}
int32_t ahead_acc(uint64_t seq, int t, int n_spec) {
    return n_spec > 0 ? (int32_t)(ahead_mix(seq, (uint64_t)t) % (uint32_t)(n_spec + 1)) : 0;
}

/* A build's per-row arrays as the card holds them (device memory is host memory here). Each
 * slot is checked against the build's own block table, since two runs that allocated blocks in
 * a different order hold different block ids for the same position. */
struct BatchRows {
    std::vector<uint64_t> seq;
    std::vector<int32_t>  tok, pos, cu, q, ctx, acc, used;
    int bad_slots = 0;
    bool operator==(const BatchRows& o) const {
        return seq == o.seq && tok == o.tok && pos == o.pos && cu == o.cu && q == o.q &&
               ctx == o.ctx && acc == o.acc && used == o.used;
    }
};

BatchRows read_rows(const RadBatch* b, const StepPlan& p, const FakeKV& kv) {
    BatchRows r;
    for (int64_t i = 0; i < b->n_seq; ++i) {
        r.seq.push_back(p.e[(size_t)i].seq);
        r.q.push_back(b->q_lens[i]);
        r.ctx.push_back(b->ctx_lens[i]);
        r.acc.push_back(b->num_accepted ? b->num_accepted[i] : 1);
    }
    for (int64_t i = 0; i <= b->n_seq; ++i) r.cu.push_back(b->cu_seqlens[i]);
    for (int64_t t = 0; t < b->n_tok; ++t) {
        r.tok.push_back(b->token_ids[t]);
        r.pos.push_back(b->positions[t]);
    }
    for (int g = 0; g < b->n_kv_groups; ++g) {
        const RadKVGroupBatch& G = b->kv[g];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            r.used.push_back(G.seqused[i]);
            const int32_t* bt = G.block_table + i * G.block_table_pitch;
            for (int32_t t = b->cu_seqlens[i]; t < b->cu_seqlens[i + 1]; ++t) {
                const int64_t p_ = b->positions[t];
                int64_t want = -1;
                if (kv.geom.g[(size_t)g].paged) {
                    const int64_t bi = p_ / G.block_size;
                    if (bi < G.block_table_pitch && bt[bi] >= 0)
                        want = bt[bi] * G.block_size + p_ % G.block_size;
                } else {
                    want = kv.state_slot(r.seq[(size_t)i], g);
                }
                if (want < 0 || G.slot_mapping[t] != want) ++r.bad_slots;
            }
        }
    }
    return r;
}

struct AheadRun {
    std::vector<BatchRows> steps;
    std::vector<std::vector<int32_t>> out;
    std::vector<int64_t> held;      /* paged blocks each sequence holds at the end */
    int ahead = 0;                  /* steps that ran issued ahead */
    int card_host_disagree = 0;
};

void ahead_setup(Fixture& f, int n_spec, std::vector<uint64_t>& ids) {
    f.build();
    f.cfg.n_spec = n_spec;
    CHECK_OK(f.start());
    for (int n : { 40, 56, 72 }) {
        auto r = f.request(n, 0);
        r->speculation_fixed = true;     /* no grammar: the commit cannot change speculation */
        ids.push_back(f.add(std::move(r)));
    }
}

int ahead_commit(Fixture& f, int t) {
    const StepPlan& p = f.sch.plan();
    std::vector<int32_t> a(p.e.size()), tok(p.e.size());
    std::vector<uint8_t> eos(p.e.size(), 0);
    for (size_t i = 0; i < p.e.size(); ++i) {
        a[i] = ahead_acc(p.e[i].seq, t, p.e[i].n_spec);
        tok[i] = ahead_tok(p.e[i].seq, t);
    }
    Scheduler::StepResult r;
    r.n_seq = (int)p.e.size();
    r.n_accepted = a.data();
    r.token = tok.data();
    r.eos = eos.data();
    return f.sch.commit(r);
}

void ahead_publish(Fixture& f, const std::vector<uint64_t>& ids, int n_spec, int t) {
    for (uint64_t id : ids) {
        int32_t d[8];
        for (int j = 0; j < n_spec; ++j) d[j] = ahead_draft(id, t, j);
        if (n_spec > 0) CHECK_OK(f.sch.set_draft(id, d, n_spec));
    }
}

void ahead_finish(Fixture& f, const std::vector<uint64_t>& ids, AheadRun& run) {
    for (uint64_t id : ids) {
        run.out.push_back(f.sch.find(id)->output);
        run.held.push_back((int64_t)f.kv.block_table(id, 0).size());
    }
}

/* Every step built by step() after the commit. */
AheadRun ahead_reference(int n_spec, int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, n_spec, ids);
    AheadRun run;
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    for (int t = 0; t < steps; ++t) {
        CHECK_OK(ahead_commit(f, t));
        ahead_publish(f, ids, n_spec, t + 1);
        b = f.sch.step();
        if (!b) break;
        run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    ahead_finish(f, ids, run);
    return run;
}

/* The engine's order with a step issued ahead (Engine::step_loop): the step's sampler, the card's
 * acceptance into the StepSlots, the draft head's proposals into them, expect_draft and
 * step_next, the trunk advance on the new build, then the commit, the drafter's set_draft and
 * promote_next. */
AheadRun ahead_pipelined(int n_spec, int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, n_spec, ids);
    AheadRun run;
    std::vector<StepSlot> slots((size_t)f.cfg.max_seqs);
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    for (int t = 0; t < steps; ++t) {
        /* The sampler's rows for step t: the drafts it agrees with, then its own token. */
        const StepPlan& p = f.sch.plan();
        std::vector<int32_t> y((size_t)b->n_out + 1, -1);
        for (const StepEntry& e : p.e) {
            if (e.dev_out < 0) continue;
            const int32_t a = ahead_acc(e.seq, t, e.n_spec);
            for (int j = 0; j <= e.n_spec; ++j)
                y[(size_t)(e.dev_out + j)] = j < a ? ahead_draft(e.seq, t, j) : ahead_tok(e.seq, t);
        }
        AdvanceArgs acc = f.sch.advance_args(0);
        acc.mode = ADV_ACCEPT;
        acc.st = slots.data();
        acc.src = y.data();
        for (int i = 0; i < acc.n_seq; ++i) adv_row(acc, i);
        for (const StepEntry& e : p.e) {
            const StepSlot& S = slots[(size_t)e.card];
            if (S.acc != ahead_acc(e.seq, t, e.n_spec) || S.tok[0] != ahead_tok(e.seq, t))
                ++run.card_host_disagree;
            /* The draft head's proposals for step t + 1 (ADV_ROUND / ADV_TAKE on a card). */
            for (int j = 0; j < n_spec; ++j)
                slots[(size_t)e.card].tok[1 + j] = ahead_draft(e.seq, t + 1, j);
            if (n_spec > 0) f.sch.expect_draft(e.seq, n_spec);
        }
        const RadBatch* nb = f.sch.step_next();
        if (nb) {
            AdvanceArgs tr = f.sch.advance_args(0);
            tr.mode = ADV_TRUNK;
            tr.st = slots.data();
            for (int i = 0; i < tr.n_seq; ++i) adv_row(tr, i);
        }
        CHECK_OK(ahead_commit(f, t));
        ahead_publish(f, ids, n_spec, t + 1);
        if (nb) {
            f.sch.promote_next();
            b = nb;
            ++run.ahead;
        } else {
            b = f.sch.step();
            if (!b) break;
        }
        run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    ahead_finish(f, ids, run);
    return run;
}
}  /* namespace */

/* A step issued ahead is the step the scheduler would have built after the commit: the same
 * tokens, positions, slots (each through its own table), lengths, context and num_accepted on
 * every row, once the card's advance has filled in what the host only bounded. The host sizes
 * such a step at the furthest context it can reach and the card's acceptance places it; a
 * disagreement puts K/V at wrong positions or reads the wrong token, and the output is silently
 * wrong. Run with drafts (a serial head's pipeline) and without (no drafter). */
TEST(a_step_issued_ahead_equals_the_step_built_after_the_commit) {
    for (int n_spec : { 0, 4 }) {
        const int T = 24;
        const AheadRun a = ahead_reference(n_spec, T);
        const AheadRun b = ahead_pipelined(n_spec, T);
        CHECK_EQ(b.ahead, T);                       /* every step after the first ran ahead */
        CHECK_EQ(b.card_host_disagree, 0);
        REQUIRE_EQ(a.steps.size(), b.steps.size());
        REQUIRE_EQ(a.steps.size(), (size_t)T + 1);
        for (size_t t = 0; t < a.steps.size(); ++t) {
            const bool same = a.steps[t] == b.steps[t];
            CHECK(same);
            if (!same) std::fprintf(stderr, "    n_spec %d: step %zu differs\n", n_spec, t);
            CHECK_EQ(a.steps[t].bad_slots, 0);
            CHECK_EQ(b.steps[t].bad_slots, 0);
        }
        CHECK(a.out == b.out);
        /* The pipelined run holds blocks up to the furthest position a step issued ahead could
         * reach, and gives back rejected drafts' blocks only on a commit nothing runs ahead of:
         * never fewer than the reference, and at most one block more. */
        REQUIRE_EQ(a.held.size(), b.held.size());
        for (size_t i = 0; i < a.held.size(); ++i) {
            CHECK(b.held[i] >= a.held[i]);
            CHECK(b.held[i] <= a.held[i] + 1);
        }
    }
}

/* ================================================================== a block drafter on the card */
namespace {
/* Every pass of a block drafter's run: each step's trunk batch, then its context and query
 * passes, as the card holds them once its advances have run. */
struct BlockRun {
    std::vector<BatchRows> trunk, context, query;
    std::vector<std::vector<int32_t>> out;
    int ahead = 0;
    int card_host_disagree = 0;
};
constexpr int kBlk = 8, kBlkDepth = 7;
constexpr int32_t kBlkMask = 99;

/* The host's driver: both passes built after the commit, from the committed sequence. */
BlockRun block_reference(int steps, int blk = kBlk, int depth = kBlkDepth) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, kBlkDepth, ids);
    BlockRun run;
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    Scheduler::BlockRow rows[4];
    for (int t = 0; t < steps; ++t) {
        CHECK_OK(ahead_commit(f, t));
        CHECK_EQ(f.sch.block_begin(blk, depth, 0, kBlkMask, rows, 4), 3);
        const RadBatch* cb = f.sch.block_context(t + 1);
        if (!cb) { CHECK(cb != nullptr); return run; }
        run.context.push_back(read_rows(cb, f.sch.plan(), f.kv));
        const RadBatch* qb = f.sch.block_query(t + 1);
        if (!qb) { CHECK(qb != nullptr); return run; }
        /* The query plan is the scheduler's own; its rows are the drafting sequences in order. */
        StepPlan qp;
        for (int i = 0; i < 3; ++i) { StepEntry e; e.seq = rows[i].seq; qp.e.push_back(e); }
        run.query.push_back(read_rows(qb, qp, f.kv));
        f.sch.block_end();
        ahead_publish(f, ids, depth, t + 1);
        b = f.sch.step();
        if (!b) break;
        run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    for (uint64_t id : ids) run.out.push_back(f.sch.find(id)->output);
    return run;
}

/* The card's driver (Engine::issue_block_drafter): the step's acceptance into the StepSlots, both
 * passes built before the commit with the query rows placed by ADV_BLOCK, the proposals into the
 * slots (ADV_BTAKE), expect_draft and step_next with ADV_TRUNK, then the commit, block_end, the
 * drafts published and promote_next. */
BlockRun block_pipelined(int steps, int blk = kBlk, int depth = kBlkDepth) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, kBlkDepth, ids);
    BlockRun run;
    std::vector<StepSlot> slots((size_t)f.cfg.max_seqs);
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    Scheduler::BlockRow rows[4];
    for (int t = 0; t < steps; ++t) {
        const StepPlan& p = f.sch.plan();
        std::vector<int32_t> y((size_t)b->n_out + 1, -1);
        for (const StepEntry& e : p.e) {
            if (e.dev_out < 0) continue;
            const int32_t a = ahead_acc(e.seq, t, e.n_spec);
            for (int j = 0; j <= e.n_spec; ++j)
                y[(size_t)(e.dev_out + j)] = j < a ? ahead_draft(e.seq, t, j) : ahead_tok(e.seq, t);
        }
        AdvanceArgs acc = f.sch.advance_args(0);
        acc.mode = ADV_ACCEPT;
        acc.st = slots.data();
        acc.src = y.data();
        for (int i = 0; i < acc.n_seq; ++i) adv_row(acc, i);
        for (const StepEntry& e : p.e) {
            const StepSlot& S = slots[(size_t)e.card];
            if (S.acc != ahead_acc(e.seq, t, e.n_spec) || S.tok[0] != ahead_tok(e.seq, t))
                ++run.card_host_disagree;
        }

        CHECK_EQ(f.sch.block_begin(blk, depth, 0, kBlkMask, rows, 4, /*on_card=*/true), 3);
        const RadBatch* cb = f.sch.block_context(t + 1);
        if (!cb) { CHECK(cb != nullptr); return run; }
        run.context.push_back(read_rows(cb, f.sch.plan(), f.kv));
        const RadBatch* qb = f.sch.block_query(t + 1);
        if (!qb) { CHECK(qb != nullptr); return run; }
        AdvanceArgs q = f.sch.advance_args(0);
        q.mode = ADV_BLOCK;
        q.st = slots.data();
        for (int i = 0; i < q.n_seq; ++i) adv_row(q, i);
        StepPlan qp;
        for (int i = 0; i < 3; ++i) { StepEntry e; e.seq = rows[i].seq; qp.e.push_back(e); }
        run.query.push_back(read_rows(qb, qp, f.kv));
        /* The block's proposals for step t + 1, taken behind the anchor, at the drafter's pitch
         * (its own block's depth): a short block fills a row's front. */
        std::vector<int32_t> prop((size_t)3 * kBlkDepth);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < depth; ++j)
                prop[(size_t)(i * kBlkDepth + j)] = ahead_draft(rows[i].seq, t + 1, j);
        AdvanceArgs tk = f.sch.advance_args(0);
        tk.mode = ADV_BTAKE;
        tk.round = depth;
        tk.src = prop.data();
        tk.src_pitch = kBlkDepth;
        tk.st = slots.data();
        for (int i = 0; i < tk.n_seq; ++i) adv_row(tk, i);
        for (int i = 0; i < 3; ++i) f.sch.expect_draft(rows[i].seq, depth);

        const RadBatch* nb = f.sch.step_next();
        if (nb) {
            AdvanceArgs tr = f.sch.advance_args(0);
            tr.mode = ADV_TRUNK;
            tr.st = slots.data();
            for (int i = 0; i < tr.n_seq; ++i) adv_row(tr, i);
        }
        CHECK_OK(ahead_commit(f, t));
        f.sch.block_end();
        ahead_publish(f, ids, depth, t + 1);
        if (nb) {
            f.sch.promote_next();
            b = nb;
            ++run.ahead;
        } else {
            b = f.sch.step();
            if (!b) break;
        }
        run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    for (uint64_t id : ids) run.out.push_back(f.sch.find(id)->output);
    return run;
}
}  /* namespace */

/* A block drafter on the card runs the passes the host's driver runs after the commit: the same
 * trunk steps, and context and query passes with the same positions, lengths and slots once the
 * card's advances have filled in what the host only bounded, and the same query tokens. Its query
 * block is staged before the step's acceptance is known, at the furthest anchor; a wrong position
 * drafts from the wrong place (acceptance collapses, no error) and a wrong slot writes the
 * drafter's K/V into another block. Every step after the first runs issued ahead.
 *
 * The context pass's token ids are not compared: that pass reads hidden states and no ids
 * (Scheduler::block_context), and staged before the commit, its accepted positions are not in the
 * sequence yet. */
TEST(a_block_drafter_on_the_card_runs_the_passes_the_hosts_driver_runs) {
    const int T = 24;
    const BlockRun a = block_reference(T);
    const BlockRun b = block_pipelined(T);
    CHECK_EQ(b.ahead, T);
    CHECK_EQ(b.card_host_disagree, 0);
    REQUIRE_EQ(a.trunk.size(), (size_t)T + 1);
    REQUIRE_EQ(a.trunk.size(), b.trunk.size());
    REQUIRE_EQ(a.context.size(), b.context.size());
    REQUIRE_EQ(a.query.size(), b.query.size());
    for (size_t t = 0; t < a.trunk.size(); ++t) {
        CHECK(a.trunk[t] == b.trunk[t]);
        CHECK_EQ(b.trunk[t].bad_slots, 0);
    }
    for (size_t t = 0; t < a.context.size(); ++t) {
        BatchRows ca = a.context[t], cb = b.context[t];
        ca.tok.clear();
        cb.tok.clear();
        const bool same_c = ca == cb, same_q = a.query[t] == b.query[t];
        CHECK(same_c);
        CHECK(same_q);
        if (!same_c || !same_q)
            std::fprintf(stderr, "    step %zu: context %s, query %s\n", t,
                         same_c ? "same" : "differs", same_q ? "same" : "differs");
        CHECK_EQ(a.context[t].bad_slots, 0);
        CHECK_EQ(b.context[t].bad_slots, 0);
        CHECK_EQ(a.query[t].bad_slots, 0);
        CHECK_EQ(b.query[t].bad_slots, 0);
    }
    CHECK(a.out == b.out);
}

/* The same at DFlash2's short block (--draft-short-from: 4 rows, 3 proposals), which the card's driver
 * runs once enough sequences decode (Engine::block_shape): the query pass of the short block, its
 * proposals taken from the front of each row at the drafter's pitch, and the step issued ahead
 * verifying 3 of them. */
TEST(a_block_drafters_short_block_on_the_card_runs_the_passes_the_hosts_driver_runs) {
    const int T = 24;
    const BlockRun a = block_reference(T, 4, 3);
    const BlockRun b = block_pipelined(T, 4, 3);
    CHECK_EQ(b.ahead, T);
    CHECK_EQ(b.card_host_disagree, 0);
    REQUIRE_EQ(a.trunk.size(), (size_t)T + 1);
    REQUIRE_EQ(a.trunk.size(), b.trunk.size());
    REQUIRE_EQ(a.context.size(), b.context.size());
    REQUIRE_EQ(a.query.size(), b.query.size());
    for (size_t t = 0; t < a.trunk.size(); ++t) {
        CHECK(a.trunk[t] == b.trunk[t]);
        CHECK_EQ(b.trunk[t].bad_slots, 0);
    }
    for (size_t t = 0; t < a.context.size(); ++t) {
        BatchRows ca = a.context[t], cb = b.context[t];
        ca.tok.clear();
        cb.tok.clear();
        const bool same_c = ca == cb, same_q = a.query[t] == b.query[t];
        CHECK(same_c);
        CHECK(same_q);
        if (!same_c || !same_q)
            std::fprintf(stderr, "    step %zu: context %s, query %s\n", t,
                         same_c ? "same" : "differs", same_q ? "same" : "differs");
        CHECK_EQ(a.context[t].bad_slots, 0);
        CHECK_EQ(b.context[t].bad_slots, 0);
        CHECK_EQ(a.query[t].bad_slots, 0);
        CHECK_EQ(b.query[t].bad_slots, 0);
    }
    CHECK(a.out == b.out);
}

/* tp8 loop D18: DSpark's short block (block = depth: every row proposes, the anchor included), 2 rows and ONE row a
 * sequence -- the depth-capped lanes' form (deepseek41 RADIANCE_DS41_DSPARK_SHORT + --draft-short-from). The scheduler
 * refused blocks under 2 rows; one row is the anchor alone at the committed end, proposing the next token. The card's
 * driver runs the host's passes at each, and a 1-row query pass is one row a drafting sequence. */
TEST(a_dspark_short_block_of_one_or_two_rows_runs_the_passes_the_hosts_driver_runs) {
    const int T = 24;
    for (const int blk : { 5, 2, 1 }) {
        const BlockRun a = block_reference(T, blk, blk);
        const BlockRun b = block_pipelined(T, blk, blk);
        CHECK_EQ(b.ahead, T);
        CHECK_EQ(b.card_host_disagree, 0);
        REQUIRE_EQ(a.trunk.size(), (size_t)T + 1);
        REQUIRE_EQ(a.trunk.size(), b.trunk.size());
        REQUIRE_EQ(a.context.size(), b.context.size());
        REQUIRE_EQ(a.query.size(), b.query.size());
        int diff = 0;
        for (size_t t = 0; t < a.trunk.size(); ++t) {
            diff += !(a.trunk[t] == b.trunk[t]);
            CHECK_EQ(b.trunk[t].bad_slots, 0);
        }
        for (size_t t = 0; t < a.context.size(); ++t) {
            BatchRows ca = a.context[t], cb = b.context[t];
            ca.tok.clear();
            cb.tok.clear();
            diff += !(ca == cb) + !(a.query[t] == b.query[t]);
            CHECK_EQ(a.query[t].bad_slots, 0);
            CHECK_EQ(b.query[t].bad_slots, 0);
            /* the query pass: three drafting sequences, `blk` rows each, the anchor first at the committed end */
            REQUIRE_EQ(b.query[t].q.size(), (size_t)3);
            for (const int32_t q : b.query[t].q) CHECK_EQ(q, blk);
            CHECK_EQ(b.query[t].tok.size(), (size_t)(3 * blk));
            if (blk == 1)
                for (const int32_t tok : b.query[t].tok) CHECK(tok != kBlkMask);   /* anchors only, no mask row */
        }
        CHECK_EQ(diff, 0);
        CHECK(a.out == b.out);
        std::fprintf(stderr, "    DSpark block %d (depth %d): %zu trunk steps, %d issued ahead, %d differ\n", blk, blk,
                     b.trunk.size(), b.ahead, diff);
    }
}

/* ================================================================== the token budget per token */
/* A verify step emits up to 1 + n_spec tokens at once. The budget (max_tokens, and the context)
 * is applied token by token before each reaches the sink: trimming `output` afterwards would leave
 * the overshoot already streamed, and the same prompt at the same max_tokens would come back
 * longer with speculation on. `output` alone cannot show this; the sink is the client's view. */
TEST(a_verify_step_at_the_max_tokens_edge_streams_exactly_max_tokens) {
    server::Sink sink;
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t id = f.submit_to(&sink, 40, /*max_tokens=*/3);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(5);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(id, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
    CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));   /* all accepted */

    CHECK_EQ(sink.n_produced(), (int64_t)3);
    server::SinkToken got[8];
    REQUIRE_EQ(sink.read(got, 8), (size_t)3);
    CHECK_EQ(got[0].token, 5); CHECK_EQ(got[1].token, 21); CHECK_EQ(got[2].token, 22);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
    const Request* r = f.sch.find(id);
    CHECK_EQ((int)r->output.size(), 3);
    CHECK_EQ(r->finish_reason, std::string("length"));
}

/* The context bounds the stream the same way, with no max_tokens at all. */
TEST(a_verify_step_at_the_context_edge_streams_exactly_the_room_left) {
    server::Sink sink;
    Fixture f;
    f.build();
    f.cfg.max_ctx = 170;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t id = f.submit_to(&sink, 160, /*max_tokens=*/0);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(5);
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int guard = 0; guard < 8 && !sink.done(); ++guard) {
        CHECK_OK(f.sch.set_draft(id, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));
    }
    CHECK(sink.done());
    CHECK_EQ(sink.n_produced(), (int64_t)10);           /* 170 - 160 */
    CHECK_EQ((int)f.sch.find(id)->output.size(), 10);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
}

/* An end-of-generation token ends an ordinary request after its accepted drafts and the token are
 * streamed; with ignore_eos the request runs on, and the same tokens reach the client. */
TEST(an_end_of_generation_token_after_accepted_drafts_ends_the_request_after_them) {
    for (int ignore = 0; ignore < 2; ++ignore) {
        server::Sink sink;
        Fixture f;
        f.build();
        f.cfg.n_spec = 4;
        CHECK_OK(f.start());
        auto r = f.request(40, 64);
        r->sink = &sink;
        r->ignore_eos = ignore != 0;
        const uint64_t id = f.add(std::move(r));
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(5);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(id, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(f.commit(9, /*eos=*/1, [](const StepEntry&) { return 2; }));
        CHECK_EQ(sink.n_produced(), (int64_t)4);          /* 5, then 21 22 9 */
        if (ignore) {
            CHECK(!sink.done());
            CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Running);
        } else {
            CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Stop);
            CHECK_EQ(f.sch.find(id)->finish_reason, std::string("stop"));
        }
    }
}

/* ================================================================== a recycled slot */
/* Slots are recycled, and every piece of drafter state must start over with the request in it. A
 * serial head's history watermark left from the slot's last tenant empties the next prompt's
 * history pass, so the head drafts against another request's K/V and greedy output varies from
 * request to request with speculation on. */
TEST(a_request_in_a_recycled_slot_runs_its_serial_heads_whole_history) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    CHECK_OK(f.start());

    auto history = [&](int expect_rows) {
        int rounds = 4;
        Scheduler::SerialRow rows[4];
        (void)f.sch.serial_begin(&rounds, rows, 4);
        const RadBatch* hb = f.sch.serial_batch(0, 1);
        if (expect_rows >= 0) {
            CHECK(hb != nullptr);
            if (hb) CHECK_EQ(hb->n_tok, expect_rows);
        }
    };
    const uint64_t a = f.submit(160, 2);
    REQUIRE(f.sch.step() != nullptr);
    history(160);
    f.commit_plain(11);
    f.sch.serial_end(1);
    REQUIRE(f.sch.step() != nullptr);              /* the decode that finishes it */
    history(-1);
    f.commit_plain(12);
    f.sch.serial_end(1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_OK(f.sch.reap(a));

    f.submit(96, 8);                               /* takes the slot A left */
    REQUIRE(f.sch.step() != nullptr);
    history(96);
    f.commit_plain(13);
    f.sch.serial_end(1);
}

/* The same for a block drafter's filled range: a range left from the slot's last tenant (here a
 * prefix hit's floor at 512) would keep the next request's drafter waiting for a window that
 * never slides past it. */
TEST(a_request_in_a_recycled_slot_starts_its_block_drafters_cache_from_nothing) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 7;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.window = 512;
    KVGroupInfo w;                                 /* the drafter's own group, not prefix-cached */
    w.name = "draft";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.window = 512;
    w.block_size = 16;
    f.prog.kv_groups.push_back(w);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
    CHECK_OK(f.start());

    /* A: a hit resumed at 512 leaves its drafter's cache starting there. */
    f.pc.seed(1024); f.pc.lin = 1024; f.pc.src = 3;
    const uint64_t a = f.submit(1100, 1);
    Scheduler::BlockRow rows[4];
    for (int st = 0; st < 8 && f.sch.find(a)->state != ReqState::Finished; ++st) {
        REQUIRE(f.sch.step() != nullptr);
        if (st == 0) CHECK_EQ(f.sch.plan().e[0].ctx_len, 512);
        f.commit_plain(11);
        (void)f.sch.block_begin(8, 7, 512, 99, rows, 4);
        f.sch.block_end();
    }
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_OK(f.sch.reap(a));

    /* B, no hit, in A's slot: its drafter's cache starts at its own first row. */
    f.pc.attn = 0; f.pc.lin = 0; f.pc.src = -1;
    const uint64_t b = f.submit(96, 8);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_EQ(f.sch.block_begin(8, 7, 512, 99, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)b);
    f.sch.block_end();
}

/* ================================================================== num_accepted */
/* num_accepted is one-based on every row and staged on every step: 1 on every prefill row (first
 * chunk, warm chunk, or admission), the previous verify's accepted count + 1 on a decode row, and
 * the card's array equal to the plan whether or not the step drafted. A prefill row at 0 has a
 * warm chunk read its rolling windows at -1 (a foreign token, or the slot before); an array
 * staged only on drafting steps leaves the linear kernels a stale count. Both are silent. */
TEST(num_accepted_is_one_based_on_every_row_and_staged_on_every_step) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t big = f.submit(600, 4);          /* three chunks: 256, 256, 88 */
    const uint64_t spec = f.submit(40, 200);
    const uint64_t plain = f.submit(48, 200);
    auto r = f.sch.find(plain);
    r->may_speculate = false;
    std::unordered_map<uint64_t, int32_t> last_acc;
    uint32_t rng = 777;
    int prefill_rows = 0, verify_rows = 0, plain_rows = 0;
    for (int step = 0; step < 30; ++step) {
        int32_t draft[4] = { 31, 32, 33, 34 };
        if (f.sch.find(spec) && f.sch.find(spec)->n_computed > 0) (void)f.sch.set_draft(spec, draft, 4);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        REQUIRE(b->num_accepted != nullptr);
        const StepPlan& p = f.sch.plan();
        for (size_t i = 0; i < p.e.size(); ++i) {
            const StepEntry& e = p.e[i];
            CHECK(e.n_accepted >= 1);
            CHECK_EQ(b->num_accepted[i], e.n_accepted);
            if (e.is_prefill) {
                CHECK_EQ(e.n_accepted, 1);
                ++prefill_rows;
            } else {
                CHECK_EQ(e.n_accepted, last_acc[e.seq] + 1);
                if (e.n_spec > 0) ++verify_rows; else ++plain_rows;
            }
        }
        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) {
            const int32_t a = random_accept(rng, e);
            if (!e.is_prefill) last_acc[e.seq] = a;
            return a;
        }));
    }
    (void)big;
    CHECK(prefill_rows >= 5);
    CHECK(verify_rows > 5);
    CHECK(plain_rows > 5);
}

/* ================================================================== an embedding input */
/* An embedding input is admitted whole in one step even on a model that checkpoints its linear
 * state: the checkpoint splits do not apply to an input that never resumes. Split there, the
 * whole-or-nothing rule refused it every step, and as the head of the queue it held back every
 * request behind it for good. */
TEST(an_embedding_input_longer_than_the_checkpoint_interval_is_admitted_whole) {
    Fixture f;
    f.build();                                     /* linear state, interval 256, max_tok 512 */
    CHECK_OK(f.start());
    auto r = f.request(300, 1);
    r->embed = true;
    const uint64_t e = f.add(std::move(r));
    const uint64_t n = f.submit(40, 4);
    REQUIRE(e != 0);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ((int)f.sch.find(e)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(n)->state, (int)ReqState::Running);
    bool whole = false;
    for (const StepEntry& x : f.sch.plan().e)
        if (x.seq == e) whole = x.n_tokens == 300 && x.produces_token && x.ckpt_slot < 0;
    CHECK(whole);
}

/* The shared context check serves every embedding architecture. Exact-limit inputs need no
 * following generation slot, while generation keeps that slot and both refuse an overflow. */
TEST(the_context_boundary_distinguishes_embeddings_from_generation) {
    for (bool embed : {false, true}) {
        for (int length : {169, 170, 171}) {
            Fixture f;
            f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/170,
                    /*windowed=*/false, /*linear=*/false);
            REQUIRE_EQ(f.start(), RAD_OK);
            auto r = f.request(length, 1);
            r->embed = embed;
            const uint64_t id = f.add(std::move(r));
            REQUIRE(id != 0);
            const RadBatch* batch = f.sch.step();
            const bool fits = length < 170 || (embed && length == 170);
            if (fits) {
                REQUIRE(batch != nullptr);
                CHECK_EQ(batch->n_tok, length);
                CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Running);
            } else {
                CHECK(batch == nullptr);
                CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Failed);
            }
        }
    }
}

/* ================================================================== speculation gates */
/* A request that may not speculate (a constraining grammar that cannot snapshot) gets no draft
 * from either drafter: set_draft takes nothing and prompt lookup proposes nothing, though its
 * text repeats. A draft for it makes the sampler refuse the step. */
TEST(a_request_that_may_not_speculate_gets_no_draft_from_either_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 200;
    r->may_speculate = false;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    const uint64_t a = r->id;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);
    int32_t draft[4] = { 101, 102, 103, 104 };
    for (int t = 0; t < 20; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 0);
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
        CHECK_EQ(f.sch.plan().e[0].n_draft, 0);
        f.commit_plain(100 + (t + 1) % 8);           /* the pattern goes on */
    }
    CHECK_EQ(f.sch.metrics().draft_tokens, (int64_t)0);
}

/* ================================================================== the step issued ahead, ends */
/* A request carried by a step issued ahead outlives its end by that step: a cancel and reap keep
 * its blocks and its Request until the step ahead has committed, and take_completed withholds its
 * id until then, because that step still writes its blocks and its sampler is staged after this
 * commit. Released early, a block the step ahead writes goes to a new sequence (cross-request KV
 * corruption) and the sampler state goes away under a staged step. */
TEST(a_cancel_while_a_step_runs_ahead_keeps_the_request_until_that_step_commits) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    auto r = f.request(40, 64);
    r->speculation_fixed = true;
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    REQUIRE(f.sch.step_next() != nullptr);
    const int64_t held = f.kv.free_blocks(0);
    CHECK(held < before);

    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));
    std::vector<uint64_t> done, all;
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK(done.empty());
    CHECK(f.sch.find(a) != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), held);

    f.commit_plain();                              /* the step in flight */
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK(done.empty());
    CHECK_EQ(f.kv.free_blocks(0), held);           /* the step ahead still writes them */
    CHECK(f.sch.find(a) != nullptr);

    f.sch.promote_next();
    f.commit_plain();                              /* the step ahead */
    CHECK(f.sch.find(a) == nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before);
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK_EQ(all.size(), (size_t)1);
    CHECK(!all.empty() && all[0] == a);
}

/* A request that finishes at the commit of a step whose successor is already issued keeps its
 * blocks for that successor, which writes them, and returns them at its commit. The end is one
 * the scheduler could not see coming (an end-of-text it did not draft): a step known to end the
 * request issues nothing ahead (no_step_is_issued_ahead_behind_one_that_ends_its_requests). */
TEST(a_request_finishing_under_a_step_issued_ahead_keeps_its_blocks_for_that_step) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    auto r = f.request(40, /*max_tokens=*/64);
    r->speculation_fixed = true;
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    REQUIRE(f.sch.step_next() != nullptr);
    const int64_t held = f.kv.free_blocks(0);
    CHECK_OK(f.commit(7, /*eos=*/1));              /* the second token, an end of text: finished */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_EQ(f.kv.free_blocks(0), held);
    f.sch.promote_next();
    f.commit_plain();
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);   /* the step ahead emitted nothing */
}

/* A SCORED row (Request::token_scores) the step in flight brings to max_tokens gets no step issued
 * ahead: it would be a forward pass for a request that finishes at the commit. The step after the
 * commit is planned as usual. */
TEST(no_step_is_issued_ahead_for_a_scored_request_the_step_in_flight_finishes) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    auto r = f.request(40, /*max_tokens=*/2);
    r->speculation_fixed = true;
    r->token_scores = std::make_shared<TokenScores>();
    r->token_scores->tokens = { 1, 2 };
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.sch.step_next() == nullptr);
    f.commit_plain();                              /* the second token: finished */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);
    CHECK(f.sch.step() == nullptr);
}

/* A row without token_scores whose step in flight leaves a token to decode is an ordinary decode:
 * the step ahead is issued. (A request any step in flight finishes gets none, scored or not.) */
TEST(a_non_scored_row_keeps_the_step_issued_ahead) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    auto r = f.request(40, /*max_tokens=*/3);
    r->speculation_fixed = true;
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.sch.step_next() != nullptr);
    f.commit_plain();
    f.sch.promote_next();
    f.commit_plain();                              /* the step ahead: the third token */
    CHECK_EQ((int)f.sch.find(a)->output.size(), 3);
}

/* ================================================================== terminal paths */
/* A lone request that outgrows the whole pool fails with the reason the client sees and gives its
 * blocks back; nothing is preempted, since there is nobody to preempt and preempting itself would
 * loop. */
TEST(a_request_no_pool_can_hold_fails_kv_exhausted_and_frees_its_blocks) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/10, /*states=*/64);         /* 160 tokens */
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 150, 64);
    for (int t = 0; t < 20 && !sink.done(); ++t) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    CHECK(sink.done());
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ(f.sch.find(a)->finish_reason, std::string("kv_exhausted"));
    CHECK_EQ(sink.status(), (int)RAD_E_FULL);
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
    CHECK(f.sch.step() == nullptr);
}

/* The lowest-priority running request that cannot grow while the requests ahead of it hold the
 * pool is preempted by recompute and waits for them, rather than failed: it fits the pool on its
 * own, so failing it returns an error for a request the engine can serve. It re-enters at the
 * front of the queue and runs once they finish. Only a request running alone is failed. */
TEST(the_last_request_short_of_a_block_yields_to_the_requests_ahead_of_it) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);         /* 320 tokens */
    CHECK_OK(f.start());
    std::vector<uint64_t> ahead;
    for (int i = 0; i < 3; ++i) ahead.push_back(f.submit(70, 2));   /* 5 blocks, room to 80 */
    const uint64_t last = f.submit_to(&sink, 80, 4);                /* 5 blocks, full at 80 */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)4);
    CHECK_EQ(f.kv.free_blocks(0), (int64_t)0);
    f.commit_plain();

    /* The last needs a sixth block for position 80 and the pool has none. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)3);
    CHECK_EQ((int)f.sch.find(last)->state, (int)ReqState::Preempted);
    CHECK(f.sch.find(last)->finish_reason.empty());
    CHECK(!sink.done());
    CHECK_EQ(f.sch.metrics().preemptions, 1);
    uint64_t q[4];
    CHECK_EQ(f.sch.queue_snapshot(q, 4), 1);
    CHECK_EQ(q[0], last);
    for (uint64_t id : ahead) CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Running);
    f.commit_plain();                              /* the three finish (max_tokens 2) */

    for (int t = 0; t < 8 && !sink.done(); ++t) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    CHECK(sink.done());
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
    CHECK_EQ(sink.n_produced(), (int64_t)4);
}

/* A prompt the KV pool cannot hold even alone is refused when it is submitted, with the sink
 * failed and the reason naming the pool's size, and the queue behind it moves on. Admitted, it
 * would wait at the head of the queue forever (admission takes only what the pool holds), and
 * its client would get no reply at all. A prompt of exactly the pool's size runs. */
TEST(a_prompt_the_kv_pool_can_never_hold_is_refused_when_submitted) {
    server::Sink refused, fits;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);         /* 320 tokens, under the 4096 context */
    CHECK_OK(f.start());
    auto r = f.request(321, 8);
    r->sink = &refused;
    CHECK_EQ(f.sch.add(std::move(r)), (int)RAD_E_FULL);
    CHECK(refused.done());
    CHECK_EQ((int)refused.finish_reason(), (int)server::Finish::Error);
    CHECK_EQ(refused.status(), (int)RAD_E_FULL);
    CHECK(has(refused.why() ? refused.why() : "", "maximum context length is 320 tokens"));

    const uint64_t ok = f.submit_to(&fits, 320, 1);
    REQUIRE(ok != 0);
    for (int t = 0; t < 4 && !fits.done(); ++t) {      /* two chunks: 256, then 64 */
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().e[0].seq, ok);
        f.commit_plain();
    }
    CHECK(fits.done());
    CHECK_EQ(fits.n_produced(), (int64_t)1);
    CHECK_EQ((int)fits.finish_reason(), (int)server::Finish::Length);
}

/* A client that hung up is closed Cancelled at the next commit, and its blocks go back then. */
TEST(a_client_hang_up_closes_the_request_cancelled_at_the_next_commit) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 40, 64);
    REQUIRE(f.sch.step() != nullptr);
    sink.request_cancel();
    f.commit_plain();
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Cancelled);
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* A sink the client stopped draining fails the request rather than drop tokens from the middle of
 * the stream, and the request's blocks go back. */
TEST(a_sink_overflow_fails_the_request_and_returns_its_blocks) {
    server::Sink sink(16);                         /* the smallest ring: 16 tokens */
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 40, 64);
    for (int t = 0; t < 30 && !sink.done(); ++t) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain();
    }
    CHECK(sink.overflowed());
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ(f.sch.find(a)->finish_reason, std::string("sink_overflow"));
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Error);
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* A token whose requested logprobs could not be computed fails the request, and neither it nor
 * anything after it reaches the client: a token without the logprobs it was asked for would be a
 * silently incomplete answer. Both the prefill's token and an accepted draft's. */
TEST(a_token_whose_logprobs_failed_fails_the_request_and_is_not_sent) {
    for (int verify = 0; verify < 2; ++verify) {
        server::Sink sink;
        Fixture f;
        f.build();
        f.cfg.n_spec = 4;
        CHECK_OK(f.start());
        auto r = f.request(40, 64);
        r->sink = &sink;
        r->sp.n_probs = 0;
        const uint64_t id = f.add(std::move(r));
        REQUIRE(f.sch.step() != nullptr);
        if (verify) {
            f.commit_plain(5);
            int32_t draft[4] = { 21, 22, 23, 24 };
            CHECK_OK(f.sch.set_draft(id, draft, 4));
            REQUIRE(f.sch.step() != nullptr);
        }
        /* Rows: the first emitted token's logprobs are good; the second's failed. */
        TokenLogprobs lp[6];
        for (auto& x : lp) { x.token = 1; x.logprob = -0.5f; }
        lp[1].token = -1;
        if (!verify) lp[0].token = -1;
        int32_t acc = 3, tok = 9, first = 0;
        uint8_t eos = 0;
        Scheduler::StepResult res;
        res.n_seq = 1;
        res.n_accepted = &acc;
        res.token = &tok;
        res.eos = &eos;
        res.lp = lp;
        res.first_row = &first;
        const int64_t had = sink.n_produced();
        CHECK_OK(f.sch.commit(res));
        CHECK_EQ(sink.n_produced(), had + (verify ? 1 : 0));   /* only the good one */
        CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Failed);
        CHECK_EQ(sink.status(), (int)RAD_E_STATE);
    }
}

/* Priority before arrival: a later, higher-priority request is admitted first. */
TEST(a_later_higher_priority_request_is_admitted_first) {
    Fixture f;
    f.build();
    f.cfg.max_seqs = 1;
    CHECK_OK(f.start());
    const uint64_t lo = f.submit(40, 4, /*priority=*/0);
    const uint64_t hi = f.submit(40, 4, /*priority=*/3);
    uint64_t q[4];
    REQUIRE_EQ(f.sch.queue_snapshot(q, 4), 2);
    CHECK_EQ(q[0], hi);
    CHECK_EQ(q[1], lo);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)1);
    CHECK_EQ(f.sch.plan().e[0].seq, hi);
    CHECK_EQ((int)f.sch.find(lo)->state, (int)ReqState::Waiting);
}

/* The head of the queue blocks admission behind it: a smaller request that would fit waits too,
 * so a large request cannot be starved by a stream of small ones. */
TEST(the_head_of_the_queue_blocks_admission_behind_it) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const uint64_t x = f.submit(160, 64);          /* 10 of 20 blocks */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    const uint64_t head = f.submit(200, 4);        /* 13 blocks: does not fit beside x */
    const uint64_t small = f.submit(16, 4);        /* 1 block: would */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(x)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(head)->state, (int)ReqState::Waiting);
    CHECK_EQ((int)f.sch.find(small)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)1);
}

/* ================================================================== preemption continues */
/* Preemption by recompute is the one path that re-prefills output tokens. The re-admitted request
 * stages prompt ++ output at positions 0.., its conv cursor starts over, and its client's stream
 * goes on with no token repeated or skipped. */
TEST(a_preempted_request_continues_its_stream_exactly) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    auto r = f.request(160, 64);
    r->sink = &sink;
    const uint64_t a = f.add(std::move(r));
    const uint64_t b = f.submit(160, 64, /*priority=*/5);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);
    CHECK_EQ(sink.n_produced(), (int64_t)1);
    REQUIRE(f.sch.step() != nullptr);              /* both need a block at 160: a is preempted */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    f.commit_plain(101);
    CHECK_EQ(sink.n_produced(), (int64_t)1);
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));

    const RadBatch* rb = f.sch.step();
    REQUIRE(rb != nullptr);
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
    const StepEntry& e = f.sch.plan().e[0];
    CHECK_EQ(e.seq, a);
    CHECK_EQ(e.ctx_len, 0);
    CHECK_EQ(e.n_tokens, 161);
    CHECK_EQ(e.conv_cursor, 0);
    CHECK(e.produces_token);
    REQUIRE_EQ(rb->n_tok, (int64_t)161);
    bool staged = true;
    for (int i = 0; i < 160; ++i)
        staged = staged && rb->token_ids[i] == 1000 + i && rb->positions[i] == i;
    CHECK(staged);
    CHECK_EQ(rb->token_ids[160], 100);
    CHECK_EQ(rb->positions[160], 160);
    f.commit_plain(102);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(103);
    CHECK_EQ(sink.n_produced(), (int64_t)3);
    server::SinkToken got[4];
    REQUIRE_EQ(sink.read(got, 4), (size_t)3);
    CHECK_EQ(got[0].token, 100); CHECK_EQ(got[1].token, 102); CHECK_EQ(got[2].token, 103);
    const std::vector<int32_t> want = { 100, 102, 103 };
    CHECK(f.sch.find(a)->output == want);
}

/* With a runtime window the ring rewrites its slots in place, so a recompute cannot prefill the
 * output it replays: the prompt runs in chunks that end at the floor, then each output token a step,
 * attending as its decode step did, and the stream still goes on exactly. */
TEST(a_preempted_ring_sequence_replays_its_output_a_token_a_step) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    f.kv.ring_w = 16;
    CHECK_OK(f.start());
    auto r = f.request(160, 64);
    r->sink = &sink;
    const uint64_t a = f.add(std::move(r));
    const uint64_t b = f.submit(160, 64, /*priority=*/5);
    REQUIRE(f.sch.step() != nullptr);
    CHECK(f.kv.has_runtime_window(a));
    CHECK_EQ(f.kv.ring_floor[a], 160);
    f.commit_plain(100);
    REQUIRE(f.sch.step() != nullptr);              /* both need a block at 160: a is preempted */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    f.commit_plain(101);
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));

    /* Re-armed at the prompt's end; the prompt runs in chunks that end at it and sample
     * nothing. */
    int64_t at = 0;
    for (int k = 0; k < 8 && at < 160; ++k) {
        const RadBatch* chunk = f.sch.step();
        REQUIRE(chunk != nullptr);
        REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
        const RadKVGroupBatch& paged = chunk->kv[0];
        REQUIRE(paged.seqused_h != nullptr);
        REQUIRE(chunk->positions_h != nullptr);
        CHECK_EQ(paged.seqused_h[0], paged.seqused[0]);
        CHECK_EQ(chunk->positions_h[0], chunk->positions[0]);
        int32_t used_max = 0;
        for (int64_t i = 0; i < chunk->n_seq; ++i)
            used_max = std::max(used_max, paged.seqused[i]);
        CHECK_EQ(chunk->kv_used_max_h[0], used_max);
        CHECK_EQ(f.sch.plan().e[0].ctx_len, at);
        at += f.sch.plan().e[0].n_tokens;
        CHECK(at <= 160);
        CHECK(!f.sch.plan().e[0].produces_token);
        f.commit_plain(0);
    }
    CHECK_EQ(at, 160);
    CHECK_EQ(f.kv.ring_floor[a], 160);
    /* The replayed output token on its own, which samples the next. */
    const RadBatch* rb = f.sch.step();
    REQUIRE(rb != nullptr);
    CHECK_EQ(rb->kv_used_max_h[0], rb->kv[0].seqused[0]);
    REQUIRE(rb->kv[0].seqused_h != nullptr);
    REQUIRE(rb->positions_h != nullptr);
    CHECK_EQ(rb->kv[0].seqused_h[0], rb->kv[0].seqused[0]);
    CHECK_EQ(rb->positions_h[0], rb->positions[0]);
    CHECK_EQ(f.sch.plan().e[0].ctx_len, 160);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 1);
    CHECK(f.sch.plan().e[0].produces_token);
    CHECK_EQ(rb->token_ids[0], 100);
    CHECK_EQ(rb->positions[0], 160);
    f.commit_plain(102);
    const std::vector<int32_t> want = { 100, 102 };
    CHECK(f.sch.find(a)->output == want);
    CHECK_EQ(sink.n_produced(), (int64_t)2);
}

/* ================================================================== media rows across chunks */
/* A prompt whose tokens 50..149 are one picture's 100 encoder rows, prefilled in 64-token chunks:
 * each chunk stages exactly the rows of the picture it covers, at the batch rows of those tokens,
 * and a history pass (which reads one token on) the rows one further. A wrong offset gives the
 * model another part of the picture while the output stays fluent. Row r of the item holds the
 * value r * 4 + column, so a misplaced row reads as a wrong number. */
TEST(media_rows_are_staged_by_chunk_at_the_tokens_that_take_them) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/64);
    f.prog.encoder.modalities = RAD_MM_IMAGE;
    f.prog.encoder.n_embd = 4;
    f.prog.encoder.patch_dim = 4;
    f.prog.encoder.merge = 1;
    f.prog.encoder.max_patches = 128;
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    CHECK_OK(f.start());

    auto item = std::make_shared<mm::Item>();
    item->n_seg = 1;
    item->seg_rows = 100;
    item->encoded = true;
    item->embd.resize(100 * 4);
    for (int r = 0; r < 100; ++r)
        for (int c = 0; c < 4; ++c) item->embd[(size_t)(r * 4 + c)] = (uint16_t)(r * 4 + c);
    auto req = f.request(200, 4);
    req->media = std::make_shared<mm::PromptMedia>();
    req->media->items.push_back(mm::Placed{ item, 50 });
    mm::RopeLayout::Run run;
    run.tok = 50; run.n = 100; run.w = 10; run.base = 50; run.shift = -40; run.item = 0;
    req->media->rope.runs.push_back(run);
    f.add(std::move(req));

    /* Batch row q of a pass: the item row it must carry, or -1 for a text row. */
    auto check_pass = [&](const RadBatch* b, int64_t first_tok, int64_t shift, int64_t n) {
        int64_t want = 0;
        for (int64_t q = 0; q < n; ++q) {
            const int64_t ti = first_tok + q + shift;
            if (ti < 50 || ti >= 150) continue;
            REQUIRE(want < b->n_mm_rows);
            CHECK_EQ(b->mm_rows[want], (int32_t)q);
            const uint16_t* e = static_cast<const uint16_t*>(b->mm_embd) + want * 4;
            for (int c = 0; c < 4; ++c) CHECK_EQ((int)e[c], (int)((ti - 50) * 4 + c));
            ++want;
        }
        CHECK_EQ(b->n_mm_rows, want);
    };
    for (int chunk = 0; chunk < 4; ++chunk) {
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        REQUIRE_EQ(b->n_tok, (int64_t)(chunk < 3 ? 64 : 8));
        check_pass(b, 64 * chunk, 0, b->n_tok);
        if (chunk == 0) {
            /* M-RoPE: a picture token at its grid cell, a text token before it at its index. */
            CHECK_EQ(b->rope_mixed, 1);
            const int64_t n = b->n_tok;
            CHECK_EQ(b->rope_pos[61], 50); CHECK_EQ(b->rope_pos[n + 61], 51);
            CHECK_EQ(b->rope_pos[2 * n + 61], 51);
            CHECK_EQ(b->rope_pos[49], 49); CHECK_EQ(b->rope_pos[2 * n + 49], 49);
            int rounds = 4;
            Scheduler::SerialRow rows[4];
            (void)f.sch.serial_begin(&rounds, rows, 4);
            const RadBatch* hb = f.sch.serial_batch(0, 1);
            REQUIRE(hb != nullptr);
            REQUIRE_EQ(hb->n_tok, (int64_t)64);
            check_pass(hb, 0, /*shift=*/1, 64);
            f.sch.serial_end(0);
        }
        f.commit_plain();
    }
}

/* ================================================================== runs read both ways */
/* A picture whose placeholders the model reads both ways (RadEncoderDecl::bidirectional) is never
 * cut by a chunk: the chunk ends before it on the quantum (64 here, the linear group's chunk), or,
 * starting at it, runs past its end, and the pass names the run's rows and the keys they see
 * (RadBatch::n_bidi). Tokens 200..299 here; without the flag the first chunk would end at the
 * checkpoint interval (256), inside the picture. A run longer than any step is cut where its chunk
 * falls, and its rows then see the run up to the cut. An audio clip is read causally and cut
 * anywhere. */
namespace {

std::unique_ptr<Request> with_run(Fixture& f, int n_prompt, int tok, int n, mm::Kind kind) {
    auto item = std::make_shared<mm::Item>();
    item->kind = kind;
    item->n_seg = 1;
    item->seg_rows = n;
    item->encoded = true;
    item->embd.assign((size_t)n * 4, 0);
    auto req = f.request(n_prompt, 4);
    req->media = std::make_shared<mm::PromptMedia>();
    req->media->items.push_back(mm::Placed{ item, tok - 1 });
    mm::RopeLayout::Run run;
    run.tok = tok; run.n = n; run.w = n; run.base = tok; run.shift = 0; run.item = 0; run.flat = true;
    req->media->rope.runs.push_back(run);
    return req;
}

void media_fixture(Fixture& f, int64_t max_tok, int32_t bidi) {
    f.build(/*blocks=*/4096, /*states=*/64, max_tok);
    f.prog.encoder.modalities = RAD_MM_IMAGE;
    f.prog.encoder.n_embd = 4;
    f.prog.encoder.patch_dim = 4;
    f.prog.encoder.merge = 1;
    f.prog.encoder.max_patches = 1024;
    f.prog.encoder.bidirectional = bidi;
}

}  /* namespace */

TEST(a_run_read_both_ways_is_never_cut_by_a_chunk) {
    Fixture f;
    media_fixture(f, 256, 1);
    CHECK_OK(f.start());
    f.add(with_run(f, 400, 200, 100, mm::Kind::Image));
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)192);
    CHECK_EQ(b->n_bidi, (int64_t)0);
    f.commit_plain();
    b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)128);                  /* 192 .. 319, past the run's end */
    REQUIRE_EQ(b->n_bidi, (int64_t)1);
    CHECK_EQ(b->bidi_row[0], 8);
    CHECK_EQ(b->bidi_n[0], 100);
    CHECK_EQ(b->bidi_seq[0], 0);
    CHECK_EQ(b->bidi_used[0], 300);
    CHECK_EQ(b->n_mm_rows, (int64_t)100);
    f.commit_plain();
    b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)64);                   /* to the tail checkpoint, text only */
    CHECK_EQ(b->n_bidi, (int64_t)0);

    /* bidi_fit itself: inside the run, from before it, from its start, and with too small a
     * budget (wait) or a run no step holds (cut as it falls). */
    auto r = with_run(f, 400, 200, 100, mm::Kind::Image);
    CHECK_EQ(f.sch.bidi_fit(*r, 0, 400, 256, 256), 192);
    CHECK_EQ(f.sch.bidi_fit(*r, 0, 400, 192, 256), 192);
    CHECK_EQ(f.sch.bidi_fit(*r, 192, 400, 64, 256), 128);
    CHECK_EQ(f.sch.bidi_fit(*r, 192, 400, 64, 100), 0);
    CHECK_EQ(f.sch.bidi_fit(*r, 0, 250, 250, 256), 250);   /* the sequence ends inside it */
    auto big = with_run(f, 800, 200, 400, mm::Kind::Image);
    CHECK_EQ(f.sch.bidi_fit(*big, 192, 800, 64, 256), 64);
    /* Two video frames, <eoi><boi> apart: back past both, or on past both. */
    auto vid = with_run(f, 400, 100, 70, mm::Kind::Video);
    mm::RopeLayout::Run second = vid->media->rope.runs[0];
    second.tok = 172;
    vid->media->rope.runs.push_back(second);
    CHECK_EQ(f.sch.bidi_fit(*vid, 0, 400, 200, 256), 64);
    CHECK_EQ(f.sch.bidi_fit(*vid, 64, 400, 64, 256), 192);
    CHECK_EQ(f.sch.bidi_fit(*vid, 64, 400, 64, 128), 0);
}

TEST(a_run_longer_than_a_step_is_cut_and_reads_up_to_the_cut) {
    Fixture f;
    media_fixture(f, 256, 1);
    CHECK_OK(f.start());
    f.add(with_run(f, 800, 200, 400, mm::Kind::Image));
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)192);
    f.commit_plain();
    b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)64);
    REQUIRE_EQ(b->n_bidi, (int64_t)1);
    CHECK_EQ(b->bidi_row[0], 8);
    CHECK_EQ(b->bidi_n[0], 56);
    CHECK_EQ(b->bidi_used[0], 256);
}

TEST(runs_are_cut_anywhere_without_the_flag_and_for_audio) {
    for (int k = 0; k < 2; ++k) {
        Fixture f;
        media_fixture(f, 256, k == 0 ? 0 : 1);
        CHECK_OK(f.start());
        f.add(with_run(f, 400, 200, 100, k == 0 ? mm::Kind::Image : mm::Kind::Audio));
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        CHECK_EQ(b->n_tok, (int64_t)256);
        CHECK_EQ(b->n_bidi, (int64_t)0);
        CHECK_EQ(b->n_mm_rows, (int64_t)56);
    }
}

/* ================================================================== scored rows (KL mode) */
/* In KL mode a scored prompt names a logits row at every position it prefills, after the
 * sampler's rows and in entry order, which is the order Engine::score_rows maps back. A row
 * miscounted here scores the wrong position against the reference and biases every comparison
 * the instrument makes. */
TEST(scored_prompt_rows_follow_the_sampler_rows_in_entry_order) {
    Fixture f;
    f.build();
    f.cfg.kld_record = "/nonexistent";             /* only kld() is read here */
    CHECK_OK(f.start());
    auto x = f.request(600, 4);
    x->score = true;
    const uint64_t xid = f.add(std::move(x));
    const uint64_t y = f.submit(40, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    /* X's first chunk (0..255, no token) then Y's whole prompt (256..295, one sampled row). */
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)2);
    CHECK_EQ(f.sch.plan().e[0].seq, xid);
    REQUIRE_EQ(b->n_out, (int64_t)(1 + 256));
    CHECK_EQ(b->out_ids[0], 295);
    bool ok = true;
    for (int j = 0; j < 256; ++j) ok = ok && b->out_ids[1 + j] == j;
    CHECK(ok);
    f.commit_plain();

    /* Y decodes first (decode rows lead), then X's second chunk. */
    b = f.sch.step();
    REQUIRE(b != nullptr);
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)2);
    CHECK_EQ(f.sch.plan().e[0].seq, y);
    REQUIRE_EQ(b->n_out, (int64_t)(1 + 256));
    CHECK_EQ(b->out_ids[0], 0);
    ok = true;
    for (int j = 0; j < 256; ++j) ok = ok && b->out_ids[1 + j] == 1 + j;
    CHECK(ok);
}

/* ================================================================== the linear-state chunk walk */
/* state_chunk_bounds and state_chunk_seq against a per-token definition: tokens belong to the same
 * chunk exactly when they are in the same sequence and the same absolute multiple of C. A chunk
 * crossing a sequence or a multiple carries a partial tile's state into the next chunk, which
 * corrupts the recurrent state silently. A cap one short is refused without a write past it. */
TEST(the_state_chunk_walk_matches_a_per_token_definition) {
    const DeriveRow* bounds = derive_row(derive_index("state_chunk_bounds"));
    const DeriveRow* seqs = derive_row(derive_index("state_chunk_seq"));
    REQUIRE(bounds && seqs);
    const uint32_t seed = 0xC4A11u;
    uint32_t rng = seed;
    auto next = [&](uint32_t n) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) % n; };
    int bad = 0;
    for (int c = 0; c < 2000 && !bad; ++c) {
        const int64_t C = (int64_t)1 << (2 * next(4));          /* 1, 4, 16, 64 */
        const int n_seq = 1 + (int)next(6);
        std::vector<int32_t> cu(1, 0), pos;
        for (int s = 0; s < n_seq; ++s) {
            const int32_t p0 = (int32_t)next(300), len = (int32_t)next(5) == 0 ? 0 : (int32_t)next(80);
            for (int32_t j = 0; j < len; ++j) pos.push_back(p0 + j);
            cu.push_back(cu.back() + len);
        }
        /* The definition: a new chunk wherever (sequence, position / C) changes. */
        std::vector<int32_t> want_b(1, 0), want_s;
        for (int s = 0; s < n_seq; ++s)
            for (int32_t t = cu[(size_t)s]; t < cu[(size_t)s + 1]; ++t) {
                const bool last = t + 1 == cu[(size_t)s + 1] ||
                                  pos[(size_t)t + 1] / C != pos[(size_t)t] / C;
                if (last) { want_b.push_back(t + 1); want_s.push_back(s); }
            }
        DeriveInput in;
        in.cu_seqlens = cu.data();
        in.positions = pos.data();
        in.n_tok = (int64_t)pos.size();
        in.n_seq = n_seq;
        in.state_chunk = C;
        for (int which = 0; which < 2 && !bad; ++which) {
            const DeriveRow* row = which ? seqs : bounds;
            const std::vector<int32_t>& want = which ? want_s : want_b;
            const int64_t need = (int64_t)want.size();
            std::vector<int32_t> out((size_t)need + 8, 0x7ead);
            int64_t n = -1;
            const int st = row->fn(in, out.data(), need, &n);
            bool ok = st == RAD_OK && n == need &&
                      std::equal(want.begin(), want.end(), out.begin());
            if (need > 0) {
                std::vector<int32_t> canary((size_t)need + 8, 0x7ead);
                int64_t m = -1;
                ok = ok && row->fn(in, canary.data(), need - 1, &m) == RAD_E_FULL;
                for (size_t k = (size_t)need - 1; k < canary.size(); ++k)
                    ok = ok && canary[k] == 0x7ead;
            }
            if (!ok) {
                bad = 1;
                std::fprintf(stderr, "    seed 0x%x case %d: %s, C %lld, %d sequences, %zu tokens\n",
                             seed, c, row->name, (long long)C, n_seq, pos.size());
            }
        }
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the pass's host bounds */
/* Every row of every build, issued ahead or not, has a block-table column for every position it
 * can write: the card writes slot -1 past the pitch, which silently drops a verified token's K/V.
 * And the two host-side bounds a recorded pass is keyed by, max_ctx_len and the pitch, cross
 * their buckets on the same builds; apart, each crossing re-records every pass twice. */
TEST(every_row_has_a_table_column_and_the_pass_bounds_cross_buckets_together) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/16384);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    auto r = f.request(1000, 0);
    r->speculation_fixed = true;
    const uint64_t id = f.add(std::move(r));
    uint32_t rng = 4242;
    int32_t last_ctx = -1;
    int64_t last_pitch = -1;
    int ahead_builds = 0, crossings = 0, bad_cover = 0, apart = 0;
    auto check = [&](const RadBatch* b) {
        const RadKVGroupBatch& g = b->kv[0];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            if (g.block_table_pitch * g.block_size < (int64_t)b->ctx_lens[i] + b->q_lens[i]) ++bad_cover;
            if (b->ctx_lens[i] > b->max_ctx_len) ++bad_cover;
        }
        if (b->n_seq == 1 && b->q_lens[0] == 5) {        /* steady decode: one shape */
            if (last_ctx >= 0 && (b->max_ctx_len != last_ctx) != (g.block_table_pitch != last_pitch))
                ++apart;
            if (last_ctx >= 0 && b->max_ctx_len != last_ctx) ++crossings;
            last_ctx = b->max_ctx_len;
            last_pitch = g.block_table_pitch;
        }
    };
    int32_t draft[4] = { 31, 32, 33, 34 };
    bool open = false;                               /* a step issued ahead is current */
    for (int t = 0; t < 3000; ++t) {
        const RadBatch* b = nullptr;
        if (!open) {
            (void)f.sch.set_draft(id, draft, 4);
            b = f.sch.step();
            REQUIRE(b != nullptr);
            check(b);
        }
        open = false;
        const bool ahead = t % 3 == 0 && f.sch.plan().e[0].produces_token;
        const RadBatch* nb = nullptr;
        if (ahead) {
            f.sch.expect_draft(id, 4);
            nb = f.sch.step_next();
            if (nb) { check(nb); ++ahead_builds; }
        }
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
        if (nb) {
            CHECK_OK(f.sch.set_draft(id, draft, 4));
            f.sch.promote_next();
            open = true;
        }
    }
    CHECK_EQ(bad_cover, 0);
    CHECK_EQ(apart, 0);
    CHECK(crossings >= 4);
    CHECK(ahead_builds > 500);
}

/* ================================================================== a wider state index */
/* A state group whose kernel reads more than one column per sequence (the anchor scheme's
 * committed and scratch slots) gets the manager's whole row at the group's pitch on every build,
 * through rows reused after a request ends and sequences moving between rows. A stale or
 * misaligned column reads another sequence's recurrent state. */
TEST(a_wider_state_index_row_is_the_managers_row_on_every_build) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.kv.sidx_width = 2;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    std::vector<uint64_t> ids;
    ids.push_back(f.submit(40, 9));
    ids.push_back(f.submit(70, 200));
    ids.push_back(f.submit(17, 90));
    uint32_t rng = 2468;
    int checked = 0;
    for (int step = 0; step < 80; ++step) {
        if (step == 6) ids.push_back(f.submit(33, 120));
        if (step == 20) ids.push_back(f.submit(90, 60));
        for (uint64_t id : ids) {
            int32_t draft[4] = { 31, 32, 33, 34 };
            (void)f.sch.set_draft(id, draft, 4);
        }
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        for (int gi = 0; gi < b->n_kv_groups; ++gi) {
            const RadKVGroupBatch& g = b->kv[gi];
            if (f.kv.geom.g[(size_t)gi].paged) continue;
            REQUIRE_EQ(g.state_index_pitch, (int64_t)2);
            for (int64_t i = 0; i < b->n_seq; ++i) {
                int32_t want[2] = { -2, -2 };
                CHECK_EQ(f.kv.state_index_row(p.e[(size_t)i].seq, gi, want, 2), 2);
                CHECK_EQ(g.state_index[i * 2], want[0]);
                CHECK_EQ(g.state_index[i * 2 + 1], want[1]);
                CHECK(want[0] >= 0);
                ++checked;
            }
        }
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
    }
    CHECK(checked > 100);
}

RAD_TEST_MAIN()

/* ================================================================== the plan without a span */
namespace {

/* A 64-bit FNV-1a fingerprint of a step: every plan entry the batch builder reads and every array
 * of the RadBatch it built, each over the extent the step fills. */
struct StepFp {
    uint64_t h = 1469598103934665603ull;
    void raw(const void* p, size_t n) {
        const auto* c = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    }
    void i(int64_t v) { raw(&v, sizeof v); }
    void arr(const int32_t* p, int64_t n) {
        i(p ? n : -1);
        if (p) for (int64_t k = 0; k < n; ++k) i(p[k]);
    }
    void step(const StepPlan& p, const RadBatch& b) {
        i((int64_t)p.e.size());
        for (const StepEntry& e : p.e) {
            i((int64_t)e.seq); i(e.n_tokens); i(e.ctx_len); i(e.n_spec); i(e.is_prefill);
            i(e.ckpt_tok); i(e.ckpt_slot); i(e.ckpt_restore);
        }
        i(b.phase); i(b.n_tok); i(b.n_seq); i(b.n_out); i(b.n_ahead); i(b.max_q_len);
        i(b.max_ctx_len); i(b.n_seq_decode); i(b.n_tok_decode); i(b.n_checkpoints);
        arr(b.token_ids, b.n_tok + b.n_ahead);
        arr(b.positions, b.n_tok);
        arr(b.cu_seqlens, b.n_seq + 1);
        arr(b.q_lens, b.n_seq);
        arr(b.ctx_lens, b.n_seq);
        arr(b.out_ids, b.n_out);
        arr(b.checkpoint_tok, b.n_checkpoints);
        arr(b.checkpoint_slot, b.n_checkpoints);
        i(b.n_kv_groups);
        for (int g = 0; g < b.n_kv_groups; ++g) {
            const RadKVGroupBatch& k = b.kv[g];
            arr(k.slot_mapping, k.slot_mapping ? b.n_tok : 0);
            arr(k.seqused, k.seqused ? b.n_seq : 0);
            i(k.block_table_pitch);
            arr(k.block_table, k.block_table ? b.n_seq * k.block_table_pitch : 0);
            i(k.state_index_pitch);
            arr(k.state_index, k.state_index ? b.n_seq * std::max<int64_t>(k.state_index_pitch, 1)
                                             : 0);
        }
    }
};

/* Long and short prompts arriving while others prefill and decode, every step fingerprinted. */
uint64_t scripted_steps(int64_t max_tok, int64_t interval, int64_t busy, int* n_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, max_tok, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = interval;
    f.cfg.busy_prefill_chunk = busy;
    CHECK_OK(f.start());
    struct Arrival { int at, n_prompt, max_tokens; };
    const Arrival arrivals[] = { { 0, 3000, 6 }, { 0, 200, 9 }, { 3, 1500, 4 }, { 5, 40, 20 },
                                 { 9, 2600, 3 }, { 12, 700, 5 }, { 30, 5000, 2 } };
    StepFp fp;
    *n_steps = 0;
    for (int k = 0; k < 600; ++k) {
        for (const Arrival& a : arrivals)
            if (a.at == k) CHECK(f.submit(a.n_prompt, a.max_tokens) != 0);
        const RadBatch* b = f.sch.step();
        if (!b) {
            if (k > 30) break;
            continue;
        }
        fp.step(f.sch.plan(), *b);
        ++*n_steps;
        f.commit_plain();
    }
    return fp.h;
}

}  /* namespace */

/* Without a span the scheduler plans and stages the pinned steps: the fingerprints of a scripted
 * workload's every step (plan entries and RadBatch arrays) at budgets of one, two and four
 * intervals, with --busy-prefill-chunk auto, off and fixed, taken from the scheduler as it was
 * before --prefill-span with batch.cpp's two read-ahead chunks (k_ahead_chunks). */
TEST(without_a_span_the_scheduler_plans_and_stages_the_pinned_steps) {
    struct Case { int64_t max_tok, interval, busy; uint64_t want; };
    const Case cases[] = {
        { 256, 256, -1, 0x5a54a247b5ba614eull }, { 512, 256, -1, 0x82a560f95b47a8a1ull }, { 1024, 256, -1, 0x5336de4e63911e26ull },
        { 1024, 256, 0, 0x5336de4e63911e26ull }, { 1024, 256, 128, 0xd14e26bf5be5aa5dull }, { 2048, 512, -1, 0x740799e030270f82ull },
    };
    for (const Case& c : cases) {
        int n = 0;
        const uint64_t h = scripted_steps(c.max_tok, c.interval, c.busy, &n);
        fprintf(stderr, "    pin: max_tok %lld interval %lld busy %lld: %d steps, 0x%016llxull\n",
                (long long)c.max_tok, (long long)c.interval, (long long)c.busy, n,
                (unsigned long long)h);
        CHECK(n > 20);
        CHECK_EQ(h, c.want);
    }
}

/* A recycled slot keeps its struct: add_locked only allocates when the slot's pointer is null, so
 * every field it does not list survives from the request before. admitted_us is the one such field
 * with no backstop where it is read -- the prefill figure is computed from it whenever it is
 * non-zero, and a stamp left by an earlier request is always older, so the interval is inflated and
 * reads as a slow prefill rather than as missing data. Admission cannot precede arrival, so
 * prefill_s may never exceed ttft_s: that is the invariant that catches a stamp carried over from
 * another request. */
TEST(a_recycled_slot_does_not_carry_the_previous_request_s_admission) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    const uint64_t a = f.submit(8, 4);
    REQUIRE(a != 0);
    for (int i = 0; i < 80 && f.sch.find(a)->state != ReqState::Finished; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain();
    }
    REQUIRE_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);

    /* The slot A held, read while its row is still published. The whole case rests on B landing
     * in this same slot: if the reap ever defers, or the free list is pre-populated, B gets a
     * fresh struct, the stale stamp is gone by construction, and the invariant below would hold
     * with the bug unpatched. So the precondition is asserted rather than trusted. */
    int slot_a = -1;
    {
        std::vector<Scheduler::ReqStat> st(8);
        const int n = f.sch.requests(st.data(), 8);
        for (int j = 0; j < n; ++j) if (st[(size_t)j].id == a) slot_a = st[(size_t)j].slot;
    }
    REQUIRE(slot_a >= 0);
    CHECK_OK(f.sch.reap(a));

    /* A's slot is on the free list, so B takes the slot and the struct that came with it. */
    const uint64_t b = f.submit(8, 4);
    REQUIRE(b != 0);
    for (int i = 0; i < 80; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain();
        std::vector<Scheduler::ReqStat> st(8);
        const int n = f.sch.requests(st.data(), 8);
        for (int j = 0; j < n; ++j) {
            if (st[(size_t)j].id != b) continue;
            if (st[(size_t)j].ttft_s < 0) break;   /* no first token yet */
            /* Both figures describe this request alone, so prefill -- admission to first token --
             * is bounded by ttft, arrival to first token. */
            CHECK_EQ(st[(size_t)j].slot, slot_a);   /* the reuse this case is about */
            CHECK(st[(size_t)j].prefill_s >= 0);
            CHECK(st[(size_t)j].prefill_s <= st[(size_t)j].ttft_s);
            return;
        }
    }
    CHECK(!"the second request never reached its first token");
}

/* ================================================================== a burst of long prompts, timed
 *
 * Not a check: a measurement harness for scheduling policy, run on request only:
 *
 *   RAD_SCHED_SIM=1 [SIM_N=32 SIM_LEN=131072 SIM_OUT=512 SIM_MAXTOK=2048 SIM_INTERVAL=2048
 *                    SIM_BUSY=-1 SIM_IL=0 SIM_SHARE=0.25 SIM_SEQS=32 SIM_DECODERS=0
 *                    SIM_TRACE=1] sched_test sim_burst
 *
 * SIM_DECODERS sequences of 64-token prompts are decoding (SIM_DEC_OUT tokens each, default
 * 4096) before the burst arrives. The summary counts the steps that carried a chunk per
 * checkpoint interval prefilled.
 *
 * SIM_N prompts of SIM_LEN tokens arrive at once, each then decodes SIM_OUT tokens with three
 * drafts a step (a uniform 0..3 accepted), the stager arming at 1025 rows, under
 * --prefix-resume exact. SIM_BUSY and SIM_IL are --busy-prefill-chunk and --prefill-interleave
 * (-1 for auto, at SIM_SHARE of a chunk step's time: SIM_IL=-1 SIM_SHARE=1 is auto:1). Each step's
 * time is a cost model in seconds: SIM_DEC0 + SIM_DECR per decode row when it has one, plus
 * SIM_PF0 + SIM_PFB per row when it carries a chunk. The defaults are fitted to Qwen3.8-Flash-Next
 * at TP2 (a lone prefill at ~5.4K tokens/s on 2048-token chunks; one sequence decoding ~145
 * tokens/s, sixteen ~650 with MTP), not measured per step shape: the schedule is the scheduler's
 * own, the times are an estimate. Prints the aggregate prefill rate,
 * the time-to-first-token spread and the decode rate of sequences decoding while some prompt still
 * waits for its first token. */
static double sim_env(const char* k, double d) {
    const char* v = std::getenv(k);
    return v ? std::atof(v) : d;
}

TEST(sim_burst_of_long_prompts) {
    if (!std::getenv("RAD_SCHED_SIM")) return;
    const int     N   = (int)sim_env("SIM_N", 32);
    const int64_t L   = (int64_t)sim_env("SIM_LEN", 131072);
    const int     OUT = (int)sim_env("SIM_OUT", 512);
    const int64_t MT  = (int64_t)sim_env("SIM_MAXTOK", 2048);
    const double  dec0 = sim_env("SIM_DEC0", 0.014), decr = sim_env("SIM_DECR", 0.003);
    const double  pf0 = sim_env("SIM_PF0", 0.075), pfb = sim_env("SIM_PFB", 0.147e-3);
    Fixture f;
    const int64_t ctx = L + OUT + 4096;
    f.build(/*blocks=*/(int64_t)N * (ctx / 16 + 8) + 1024, /*states=*/2 * N + 8, MT, ctx);
    f.cfg.checkpoint_interval = (int64_t)sim_env("SIM_INTERVAL", 2048);
    f.cfg.busy_prefill_chunk = (int64_t)sim_env("SIM_BUSY", -1);
    f.cfg.prefill_interleave = (int64_t)sim_env("SIM_IL", 0);
    f.cfg.prefill_interleave_share = sim_env("SIM_SHARE", 0.25);
    f.cfg.max_seqs = (int64_t)sim_env("SIM_SEQS", 32);
    f.cfg.prefix_exact = true;
    f.cfg.n_spec = 3;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1025);
    uint32_t rng = 12345;
    int32_t draft[3] = { 7, 7, 7 };
    /* The decoders, decoding before the burst arrives. */
    std::vector<uint64_t> decs;
    const int ND = (int)sim_env("SIM_DECODERS", 0);
    const int DOUT = (int)sim_env("SIM_DEC_OUT", 4096);
    for (int i = 0; i < ND; ++i) decs.push_back(f.submit(64, DOUT, 0, /*ignore_eos=*/true));
    auto draft_all = [&](const std::vector<uint64_t>& v) {
        for (uint64_t id : v) {
            const Request* r = f.sch.find(id);
            if (r && r->state == ReqState::Running && r->n_computed >= (int64_t)r->prompt.size())
                (void)f.sch.set_draft(id, draft, 3);
        }
    };
    for (int k = 0; k < 16 && ND > 0; ++k) {
        draft_all(decs);
        if (!f.sch.step()) break;
        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
    }
    std::vector<uint64_t> ids;
    for (int i = 0; i < N; ++i) {
        auto r = f.request((int)L, OUT);
        for (size_t k = 0; k < r->prompt.size(); ++k)
            r->prompt[k] = 1000 + (int32_t)(((int64_t)k * 7 + i * 13) % 60000);
        r->ignore_eos = true;
        ids.push_back(f.add(std::move(r)));
    }
    std::vector<double> ttft(ids.size(), -1), done(ids.size(), -1);
    double t = 0, t_pf = 0, dec_s = 0;
    int64_t pf_tok = 0, steps = 0, mixed = 0, small = 0, dec_tok = 0, chunk_steps = 0;
    auto idx = [&](uint64_t id) { return (size_t)(id - ids[0]); };
    while (steps < 200000) {
        draft_all(ids);
        draft_all(decs);
        if (!f.sch.step()) break;
        const StepPlan& p = f.sch.plan();
        int64_t P = 0, D = 0, nd = 0;
        for (const StepEntry& e : p.e) {
            if (e.is_prefill) P += e.n_tokens;
            else { D += e.n_tokens; ++nd; }
        }
        const double dt = (nd ? dec0 + decr * (double)nd : 0.0) +
                          (P ? pf0 + pfb * (double)(P + D) : 0.0);
        if (std::getenv("SIM_TRACE") && (steps < 12 || steps % 97 == 0)) {
            std::printf("step %lld t=%.1f:", (long long)steps, t);
            for (const StepEntry& e : p.e)
                std::printf(" %llu:%s%d@%d", (unsigned long long)e.seq, e.is_prefill ? "P" : "d",
                            e.n_tokens, e.ctx_len);
            std::printf("\n");
        }
        ++steps;
        chunk_steps += P > 0;
        mixed += P && nd;
        small += P && P < 1025;
        pf_tok += P;
        if (P) t_pf += dt;
        bool waiting = false;
        int64_t decoders = 0;
        for (size_t k = 0; k < ids.size(); ++k) {
            if (done[k] >= 0) continue;
            if (ttft[k] >= 0) ++decoders; else waiting = true;
        }
        for (uint64_t id : decs) decoders += f.sch.find(id) != nullptr;
        std::vector<std::pair<uint64_t, size_t>> before;
        for (const StepEntry& e : p.e) before.push_back({ e.seq, e.req->output.size() });
        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
        f.sch.note_step_cost(P > 0, dt);
        t += dt;
        if (waiting) dec_s += dt * (double)decoders;
        for (const auto& b : before) {
            const Request* r = f.sch.find(b.first);
            if (!r) continue;
            const bool burst = b.first >= ids[0];
            if (burst && b.second == 0 && !r->output.empty() && ttft[idx(b.first)] < 0)
                ttft[idx(b.first)] = t;
            else if (waiting && b.second > 0) dec_tok += (int64_t)(r->output.size() - b.second);
        }
        std::vector<uint64_t> c;
        f.sch.take_completed(&c);
        for (uint64_t id : c) {
            if (id >= ids[0]) done[idx(id)] = t;
            (void)f.sch.reap(id);
        }
        /* The decoders outlive the burst; the run ends with its last prompt. */
        bool left = false;
        for (double d : done) left = left || d < 0;
        if (!left) break;
    }
    std::vector<double> tt;
    for (double x : ttft) if (x >= 0) tt.push_back(x);
    std::sort(tt.begin(), tt.end());
    if (tt.empty()) return;
    auto pct = [&](double q) { return tt[std::min(tt.size() - 1, (size_t)(q * (double)tt.size()))]; };
    const double intervals = (double)pf_tok / (double)std::max<int64_t>(f.cfg.checkpoint_interval, 1);
    char il[32];
    if (f.cfg.prefill_interleave < 0)
        std::snprintf(il, sizeof il, "auto:%g", f.cfg.prefill_interleave_share);
    else
        std::snprintf(il, sizeof il, "%lld", (long long)f.cfg.prefill_interleave);
    std::printf("sim: %d x %lld tokens beside %d decoders, max_tok %lld, busy %lld, "
                "interleave %s: %lld steps (%lld mixed, %lld with a chunk under 1025, %.2f with a "
                "chunk per interval); prefill %.0f tokens/s over the steps with a "
                "chunk, %.0f to the last first token; TTFT p50 %.0f p95 %.0f max %.0f s; decode "
                "while prompts wait %.1f tokens/s a sequence; all done at %.0f s\n",
                N, (long long)L, ND, (long long)MT, (long long)f.cfg.busy_prefill_chunk,
                il, (long long)steps, (long long)mixed,
                (long long)small, (double)chunk_steps / std::max(intervals, 1e-9),
                (double)pf_tok / std::max(t_pf, 1e-9),
                (double)N * (double)L / tt.back(), pct(0.5), pct(0.95), tt.back(),
                dec_s > 0 ? (double)dec_tok / dec_s : 0.0, t);
}

/* ================================================== the depth-0 context skip (fn4 draft_ctxskip)
 * A plain step's rows (every entry one decode token) can be kept instead of passed; a refill pass is a
 * context pass (draft_pass -1) over each live sequence's kept positions, its last `window` at most, one
 * entry a sequence at its first kept position, its rows where the ring kept them (CtxRun). The engine
 * side (the ring, the copies, when to refill) is engine_draft.cpp's; these hold the plan to its rows. */
namespace {
/* A trunk with a block drafter whose own paged window group is bound past the trunk's layers and not
 * prefix-cached; group 3 (attn, gdn, conv, draft). */
void ctx_fixture(Fixture& f, int window) {
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 7;
    f.prog.meta.n_layers = 16;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.depth = 7;
    f.prog.drafter.window = window;
    KVGroupInfo w;
    w.name = "draft";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.window = 512;   /* FakeKV maps from position 0 (first_block_pos 0): a window past every position here */
    w.block_size = 16;
    w.layers = { 16 };
    f.prog.kv_groups.push_back(w);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
}
/* One decode step: plan, commit `tok`, open the block pass; keep its rows in `slot` when keepable and
 * `slot` >= 0. Returns whether the step was keepable. */
bool ctx_step(Fixture& f, int32_t tok, int slot, int64_t* rows = nullptr) {
    if (!f.sch.step()) { CHECK(!"ctx_step: no step planned"); return false; }
    f.commit_plain(tok);
    Scheduler::BlockRow br[8];
    (void)f.sch.block_begin(8, 7, (int)f.prog.drafter.window, 99, br, 8);
    int64_t r = 0, l = 0;
    const bool k = f.sch.ctx_keepable(&r, &l);
    if (rows) *rows = r;
    if (k && slot >= 0) CHECK_EQ(f.sch.ctx_keep(slot), l);
    f.sch.block_end();
    return k;
}
const RadKVGroupBatch* kvg(const RadBatch* b, int32_t group) {
    for (int g = 0; g < b->n_kv_groups; ++g)
        if (b->kv[g].group == group) return &b->kv[g];
    return nullptr;
}
}  /* namespace */

TEST(a_kept_plain_steps_rows_refill_in_one_context_pass_at_their_positions) {
    Fixture f;
    ctx_fixture(f, 64);
    CHECK_OK(f.start());
    const uint64_t a = f.submit(100, 64), b = f.submit(100, 64);
    /* The prompts' step carries 100 rows a sequence: not a plain step. */
    CHECK(!ctx_step(f, 11, 0));
    CHECK_EQ(f.sch.ctx_pending(), 0);
    for (int k = 0; k < 5; ++k) {
        int64_t rows = 0;
        CHECK(ctx_step(f, (int32_t)(12 + k), k, &rows));
        CHECK_EQ(rows, 2);
    }
    CHECK_EQ(f.sch.ctx_pending(), 10);
    std::vector<Scheduler::CtxRun> runs;
    int64_t nr = 0, nd = 0;
    const RadBatch* rb = f.sch.ctx_refill(/*step=*/9, /*cap=*/512, /*window=*/64, &runs, &nr, &nd);
    REQUIRE(rb != nullptr);
    CHECK_EQ(rb->draft_pass, -1);
    CHECK_EQ(rb->n_seq, 2);
    CHECK_EQ(rb->n_tok, 10);
    CHECK_EQ(nr, 10);
    CHECK_EQ(nd, 0);
    CHECK_EQ(rb->cu_seqlens[1], 5);
    /* seq_ids are card indices: A's then B's, as the last step's plan has them. */
    REQUIRE_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ((uint64_t)f.sch.plan().e[0].seq, a);
    CHECK_EQ((uint64_t)f.sch.plan().e[1].seq, b);
    CHECK_EQ(rb->seq_ids[0], f.sch.plan().e[0].card);
    CHECK_EQ(rb->seq_ids[1], f.sch.plan().e[1].card);
    /* Each sequence's kept positions 100..104 (its first decode step was at 100), in order. */
    for (int j = 0; j < 5; ++j) {
        CHECK_EQ(rb->positions[j], 100 + j);
        CHECK_EQ(rb->positions[5 + j], 100 + j);
    }
    const RadKVGroupBatch* g = kvg(rb, 3);
    REQUIRE(g != nullptr);
    for (int j = 0; j < 10; ++j) CHECK(g->slot_mapping[j] >= 0);
    CHECK(g->slot_mapping[0] != g->slot_mapping[5]);   /* two sequences, two caches */
    /* A's rows were row 0 of steps 0..4, B's row 1: one run each. */
    REQUIRE_EQ((int)runs.size(), 2);
    CHECK_EQ(runs[0].dst, 0);  CHECK_EQ(runs[0].slot, 0);  CHECK_EQ(runs[0].row, 0);  CHECK_EQ(runs[0].n, 5);
    CHECK_EQ(runs[1].dst, 5);  CHECK_EQ(runs[1].slot, 0);  CHECK_EQ(runs[1].row, 1);  CHECK_EQ(runs[1].n, 5);
    /* Consumed: nothing left, and the next step keeps from its own position on. */
    CHECK(f.sch.ctx_refill(9, 512, 64, &runs, &nr, &nd) == nullptr);
    CHECK_EQ(f.sch.ctx_pending(), 0);
    CHECK(ctx_step(f, 20, 0));
    CHECK_EQ(f.sch.ctx_pending(), 2);
    rb = f.sch.ctx_refill(10, 512, 64, &runs, &nr, &nd);
    REQUIRE(rb != nullptr);
    CHECK_EQ(rb->positions[0], 105);
}

/* The refill reads only what a query reads (the last `window` positions; the rest dropped), splits at
 * the cap (one sequence's rows over two passes, positions continuing), drops a finished sequence's
 * rows, and a kept sequence that skipped a position, or a step carrying drafts, is not keepable. */
TEST(a_refill_keeps_the_window_splits_at_the_cap_and_drops_what_no_query_reads) {
    Fixture f;
    ctx_fixture(f, 3);
    CHECK_OK(f.start());
    const uint64_t a = f.submit(100, 64);
    f.submit(100, 64);
    CHECK(!ctx_step(f, 11, 0));
    for (int k = 0; k < 5; ++k) CHECK(ctx_step(f, (int32_t)(12 + k), k));
    std::vector<Scheduler::CtxRun> runs;
    int64_t nr = 0, nd = 0;
    const RadBatch* rb = f.sch.ctx_refill(9, /*cap=*/4, /*window=*/3, &runs, &nr, &nd);
    REQUIRE(rb != nullptr);
    CHECK_EQ(nd, 4);                 /* A's and B's 100, 101: older than the window */
    CHECK_EQ(nr, 4);                 /* A's last three, B's first of its last three */
    CHECK_EQ(rb->n_seq, 2);
    CHECK_EQ(rb->positions[0], 102);
    CHECK_EQ(rb->positions[2], 104);
    CHECK_EQ(rb->positions[3], 102);
    REQUIRE_EQ((int)runs.size(), 2);
    CHECK_EQ(runs[0].slot, 2);  CHECK_EQ(runs[0].n, 3);
    CHECK_EQ(runs[1].slot, 2);  CHECK_EQ(runs[1].n, 1);  CHECK_EQ(runs[1].row, 1);
    rb = f.sch.ctx_refill(9, 4, 3, &runs, &nr, &nd);
    REQUIRE(rb != nullptr);
    CHECK_EQ(nd, 0);                 /* B's two oldest were dropped in the first call */
    CHECK_EQ(nr, 2);
    CHECK_EQ(rb->n_seq, 1);
    CHECK_EQ(rb->positions[0], 103);
    CHECK_EQ(rb->positions[1], 104);
    REQUIRE_EQ((int)runs.size(), 1);
    CHECK_EQ(runs[0].slot, 3);  CHECK_EQ(runs[0].row, 1);  CHECK_EQ(runs[0].n, 2);
    CHECK(f.sch.ctx_refill(9, 4, 3, &runs, &nr, &nd) == nullptr);

    /* A step whose rows were not kept, then a keepable-looking one: A and B kept nothing, so it is
     * keepable; keep it, skip keeping the next, and the one after is not (a hole at the skipped position). */
    CHECK(ctx_step(f, 30, 0));
    CHECK(ctx_step(f, 31, -1));
    CHECK(!ctx_step(f, 32, -1));
    rb = f.sch.ctx_refill(9, 64, 3, &runs, &nr, &nd);
    REQUIRE(rb != nullptr);
    CHECK_EQ(nr, 2);

    /* A ends with its kept rows: they are dropped, B's refilled. */
    CHECK(ctx_step(f, 33, 0));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(f.commit(34, /*eos=*/1));
    CHECK(f.sch.find(a) == nullptr || f.sch.find(a)->state != ReqState::Running);
    rb = f.sch.ctx_refill(9, 64, 3, &runs, &nr, &nd);
    /* both ended at the eos commit here: nothing to pass, both sequences' rows dropped */
    CHECK(rb == nullptr);
    CHECK_EQ(nd, 2);
    CHECK_EQ(f.sch.ctx_pending(), 0);
}

TEST(a_step_carrying_drafts_is_not_a_plain_step) {
    Fixture f;
    ctx_fixture(f, 64);
    CHECK_OK(f.start());
    const uint64_t a = f.submit(100, 64);
    CHECK(!ctx_step(f, 11, 0));
    CHECK(ctx_step(f, 12, 0));
    int32_t draft[3] = { 21, 22, 23 };
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 3);
    CHECK_OK(f.commit(13, 0, [](const StepEntry&) { return 1; }));
    Scheduler::BlockRow br[4];
    (void)f.sch.block_begin(8, 7, 64, 99, br, 4);
    int64_t r = 0, l = 0;
    CHECK(!f.sch.ctx_keepable(&r, &l));
    f.sch.block_end();
    CHECK_EQ(f.sch.ctx_pending(), 1);
}

/* draft_depth0 2 (a test hook): D = 0 at every cycle but the probes, at the drafting depth every
 * probe_every cycles at 0, however the measured costs say drafting pays; 1 / 0 as before. */
TEST(forced_depth_zero_holds_zero_but_for_the_probes) {
    DraftController z;
    DraftController::Params p;
    p.max_depth = 5;
    p.adapt = true;
    p.depth0 = true;
    p.depth0_forced = true;
    p.probe_every = 4;
    REQUIRE(z.configure(p) == RAD_OK);
    CHECK_EQ(z.depth0_mode(), 2);
    DraftAdapt u;
    DraftAdapt* one[1] = { &u };
    /* Drafting measured far ahead of a plain step: the measured choice would draft. */
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 8; ++k) z.step_time(d, 1, 10.0, 6.0);
    for (int k = 0; k < 8; ++k) z.step_time(0, 1, 30.0);
    int run = 0, heads = 0, zeros = 0, worst = 0;
    for (int t = 0; t < 60; ++t) {
        z.plan_step(one, 1, 1);
        if (z.head_runs()) { ++heads; ++run; worst = std::max(worst, run); }
        else { ++zeros; run = 0; }
    }
    CHECK(zeros > heads);
    CHECK(heads > 0);
    CHECK(worst <= DraftController::kProbeSteps);
    CHECK(z.probe_cycles() > 0);
    CHECK_EQ(z.zero_moves(), 1);
    z.set_depth0(false);
    CHECK_EQ(z.depth0_mode(), 0);
    CHECK(z.head_runs());
    z.set_depth0_forced(true);
    CHECK_EQ(z.depth0_mode(), 2);
    z.set_depth0(true);   /* 1 from 2: depth 0 stays on, forced stays as it was */
    CHECK(z.depth0());

    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.set_draft_depth0(2), 2);
    CHECK(f.sch.draft().depth0_forced());
    CHECK_EQ(f.sch.set_draft_depth0(1), 1);
    CHECK(!f.sch.draft().depth0_forced());
    CHECK_EQ(f.sch.set_draft_depth0(0), 0);
}

/* draft_depth0 3 (ds41-int10), depth 0 by occupancy only: plain steps measured far cheaper than drafting (D7 under 1
 * leaves drafting for 0), draft_occ 6 / draft_occ_k 8, the lend on. Under 3: every cycle at 1 .. 5 rows drafts, with no
 * probe of either kind and no lend; at 6 rows the 8th cycle takes 0 and holds it with no probe, the lend wanted after
 * lend_hold() cycles, as under 1; 8 cycles back at 4 rows return to drafting at once. 1 / 2 / 3 / 0 switch cleanly, and
 * the scheduler takes 3 and reports it. */
TEST(depth_zero_by_occupancy_only_drafts_below_occ_rows_and_holds_zero_above) {
    DraftController::Params p;
    p.max_depth = 5;
    p.adapt = true;
    p.depth0 = true;
    p.lend = 1;
    DraftController m1;
    CHECK_OK(m1.configure(p));
    p.depth0_occ = true;
    DraftController z;
    CHECK_OK(z.configure(p));
    CHECK_EQ(z.depth0_mode(), 3);
    CHECK(z.depth0_occ_only());
    CHECK(!z.depth0_forced());
    CHECK_EQ(m1.depth0_mode(), 1);
    DraftAdapt u[8];
    DraftAdapt* uf[8];
    for (int i = 0; i < 8; ++i) { u[i] = estimate(z, 0.6, 0.6); uf[i] = &u[i]; }
    for (DraftController* c : { &z, &m1 }) {
        c->set_occ(6);
        c->set_occ_k(8);
        for (int r = 1; r <= 8; ++r) {
            for (int d = 1; d <= 5; ++d)
                for (int k = 0; k < 4; ++k) c->step_time(d, r, 20.0 + 10.0 * d);
            for (int k = 0; k < 4; ++k) c->step_time(0, r, 15.0);
        }
    }
    /* below draft_occ rows: 3 always drafts, nothing probed, no 0 to lend; 1 takes 0 there */
    int heads = 0, cycles = 0, m1_zero = 0, wanted = 0;
    for (int r = 1; r <= 5; ++r)
        for (int t = 0; t < 100; ++t) {
            (void)z.plan_step(uf, r, r);
            (void)m1.plan_step(uf, r, r);
            heads += z.head_runs() ? 1 : 0;
            wanted += z.lend_wanted() ? 1 : 0;
            m1_zero += m1.head_runs() ? 0 : 1;
            ++cycles;
            if (!m1.head_runs()) m1.step_time(0, r, 15.0);
        }
    CHECK_EQ(heads, cycles);
    CHECK_EQ(z.probe_cycles(), (int64_t)0);
    CHECK_EQ(z.zero_moves(), (int64_t)0);
    CHECK_EQ(z.occ_cycles(), (int64_t)0);
    CHECK_EQ(wanted, 0);
    CHECK(m1_zero > 0);   /* the measured choice (1) would have left drafting here */
    /* at draft_occ rows: drafting for 7 cycles, 0 from the 8th, held with no probe; the lend after lend_hold() */
    int head_late = 0, head_early = 0;
    wanted = 0;
    for (int t = 0; t < 200; ++t) {
        (void)z.plan_step(uf, 6, 6);
        if (t < 7) head_early += z.head_runs() ? 1 : 0;
        else head_late += z.head_runs() ? 1 : 0;
        if (!z.head_runs()) z.step_time(0, 6, 15.0);
        wanted += z.lend_wanted() ? 1 : 0;
    }
    CHECK_EQ(head_early, 7);
    CHECK_EQ(head_late, 0);
    CHECK(z.occ_held());
    CHECK_EQ(z.occ_moves(), (int64_t)1);
    CHECK_EQ(z.zero_moves(), (int64_t)1);
    CHECK_EQ(z.probe_cycles(), (int64_t)0);
    CHECK(wanted > 100);
    /* back below: 0 held for 7 cycles, drafting from the 8th (occupancy released), and on */
    int head_dip = 0, head_back = 0;
    for (int t = 0; t < 7; ++t) { (void)z.plan_step(uf, 4, 4); head_dip += z.head_runs() ? 1 : 0; }
    CHECK_EQ(head_dip, 0);
    for (int t = 0; t < 100; ++t) { (void)z.plan_step(uf, 4, 4); head_back += z.head_runs() ? 1 : 0; }
    CHECK_EQ(head_back, 100);
    CHECK(!z.occ_held());
    CHECK_EQ(z.occ_moves(), (int64_t)2);
    CHECK_EQ(z.zero_moves(), (int64_t)2);
    CHECK_EQ(z.probe_cycles(), (int64_t)0);
    CHECK(!z.lend_wanted());
    /* the modes switch cleanly */
    z.set_depth0_forced(true);
    CHECK_EQ(z.depth0_mode(), 2);
    CHECK(!z.depth0_occ_only());
    z.set_depth0_occ(true);
    CHECK_EQ(z.depth0_mode(), 3);
    CHECK(!z.depth0_forced());
    z.set_depth0_occ(false);
    CHECK_EQ(z.depth0_mode(), 1);
    z.set_depth0_occ(true);
    z.set_depth0(false);
    CHECK_EQ(z.depth0_mode(), 0);
    CHECK(!z.depth0_occ_only());

    Fixture f;
    adaptive_fixture(f, 4);
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.set_draft_depth0(3), 3);
    CHECK(f.sch.draft().depth0_occ_only());
    CHECK_EQ(f.sch.set_draft_depth0(2), 2);
    CHECK(!f.sch.draft().depth0_occ_only());
    CHECK_EQ(f.sch.set_draft_depth0(3), 3);
    CHECK_EQ(f.sch.set_draft_depth0(1), 1);
    CHECK(!f.sch.draft().depth0_occ_only() && !f.sch.draft().depth0_forced());
    CHECK_EQ(f.sch.set_draft_depth0(0), 0);
}

/* ================================================== lever DST: fn4 sched_interleave / sched_mixed_tok
 * The decode-only steps after a chunk and the shared budget beside decode rows, as in-process switches read at every
 * step planned. Set here as the engine sets them: a request, applied at a step release (fn4::apply_pending) between a
 * commit and the next step. The fn4 state is the process's, so every test puts the switches back. */
namespace {
struct DstSwitches {
    ~DstSwitches() {
        set("sched_interleave", -1); set("sched_mixed_tok", 0); set("sched_bi_chunk", 0);
        set("sched_short_bypass", 0); set("sched_bypass_total", 0); set("sched_noa", 0);
    }
    static bool set(const char* name, int v) {
        std::string err;
        if (!fn4::request(name, v, &err)) return false;
        int sw[fn4::kSwitches];
        (void)fn4::apply_pending(0, sw);
        return true;
    }
};

/* The DS41 line's geometry: interval 2048, MNBT 4160, span 2 (shared budget 2112), the stager's 1024-row floor. */
void dst_fixture(Fixture& f, int64_t interleave) {
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = interleave;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
}

/* One step of a long prompt `p` beside decoders: 'C' when it carries a chunk of p, 'D' otherwise; the chunk's start and
 * length, and the step's tokens. */
struct DstStep { char kind; int64_t at, n, tok; bool all_dec; };
DstStep dst_step(Fixture& f, uint64_t p, const std::vector<uint64_t>& dec) {
    DstStep d{ 'D', -1, 0, 0, true };
    CHECK(f.sch.step() != nullptr);
    const StepPlan& pl = f.sch.plan();
    d.tok = pl.n_tok;
    for (uint64_t id : dec) {
        bool in = false;
        for (const StepEntry& e : pl.e) in = in || (e.seq == id && !e.is_prefill);
        d.all_dec = d.all_dec && in;
    }
    for (const StepEntry& e : pl.e)
        if (e.seq == p && e.is_prefill) { d.kind = 'C'; d.at = e.ctx_len; d.n = e.n_tokens; }
    f.commit_plain(12);
    return d;
}
}  // namespace

TEST(dst_switches_check_and_report) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    CHECK_EQ(f.sch.geometry().shared_tok, 2112);
    std::string err;
    CHECK_EQ(fn4::value(fn4::kSchedInterleave), 8);        /* unset: the flag */
    CHECK_EQ(fn4::value(fn4::kSchedMixedTok), 2112);       /* unset: the shared budget */
    CHECK(fn4::check("sched_interleave", 0, &err));
    CHECK(fn4::check("sched_interleave", 64, &err));
    CHECK(fn4::check("sched_interleave", -1, &err));
    CHECK(!fn4::check("sched_interleave", 65, &err));
    CHECK(!fn4::check("sched_interleave", -2, &err));
    CHECK(fn4::check("sched_mixed_tok", 0, &err));
    CHECK(fn4::check("sched_mixed_tok", 1088, &err));
    CHECK(fn4::check("sched_mixed_tok", 576, &err));
    CHECK(fn4::check("sched_mixed_tok", 2112, &err));
    CHECK(fn4::check("sched_mixed_tok", 128, &err));
    CHECK(!fn4::check("sched_mixed_tok", 2176, &err));     /* above the arenas' budget */
    CHECK(!fn4::check("sched_mixed_tok", 64, &err));       /* under 128 */
    CHECK(!fn4::check("sched_mixed_tok", 1000, &err));     /* not a multiple of the quantum (64 here) */
    CHECK(DstSwitches::set("sched_interleave", 0));
    CHECK_EQ(fn4::value(fn4::kSchedInterleave), 0);
    CHECK_EQ(fn4::sched_interleave(), 0);
    CHECK(DstSwitches::set("sched_interleave", -1));
    CHECK_EQ(fn4::value(fn4::kSchedInterleave), 8);
    CHECK_EQ(fn4::sched_interleave(), -1);
    CHECK(DstSwitches::set("sched_mixed_tok", 1088));
    CHECK_EQ(fn4::value(fn4::kSchedMixedTok), 1088);
    CHECK_EQ(fn4::sched_mixed_tok(), 1088);
    CHECK(DstSwitches::set("sched_mixed_tok", 0));
    CHECK_EQ(fn4::value(fn4::kSchedMixedTok), 2112);
    CHECK_EQ(fn4::sched_mixed_tok(), 0);
    /* No shared step without a span: the budget is refused, the way back accepted. */
    Fixture g;
    g.build();
    CHECK_OK(g.start());
    CHECK(!fn4::check("sched_mixed_tok", 128, &err));
    CHECK(fn4::check("sched_mixed_tok", 0, &err));
}

/* The interleave from the next planned step: a hold under way is cut to a smaller count, 0 puts a chunk in every step,
 * -1 goes back to the flag. Chunks keep their plan: every chunk beside the decoder is 2048 rows at an interval. */
TEST(dst_interleave_switch_acts_from_the_next_planned_step) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    const uint64_t d = f.submit(100, 4000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    const uint64_t p = f.submit(40000, 4);
    std::string shape;
    int bad_chunk = 0;
    auto run = [&](int n) {
        for (int k = 0; k < n; ++k) {
            const DstStep s = dst_step(f, p, { d });
            shape += s.kind;
            if (s.kind == 'C') bad_chunk += !(s.n == 2048 && s.at % 2048 == 0);
        }
    };
    run(12);
    CHECK_EQ(shape, std::string("CDDDDDDDDCDD"));          /* the flag's 8 */
    CHECK(DstSwitches::set("sched_interleave", 2));        /* 6 of this hold left: cut to 2, then 2 after each chunk */
    shape.clear();
    run(7);
    CHECK_EQ(shape, std::string("DDCDDCD"));
    CHECK(DstSwitches::set("sched_interleave", 0));        /* the hold ends: a chunk every step */
    shape.clear();
    run(4);
    CHECK_EQ(shape, std::string("CCCC"));
    CHECK(DstSwitches::set("sched_interleave", -1));       /* the flag again, from the next chunk */
    shape.clear();
    run(11);
    CHECK_EQ(shape, std::string("CDDDDDDDDCD"));
    CHECK_EQ(bad_chunk, 0);
}

/* The shared budget: beside a decoder no step is wider than it, under today's cuts (sched_bi_chunk 0) too; a decoder
 * BEHIND the long prompt in the running order keeps its row in every step (the busy-chunk floor is at most half the
 * budget); 0 is the geometry's budget again. */
TEST(dst_mixed_tok_is_the_budget_beside_decode_rows) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 0);
    const uint64_t p = f.submit(30000, 4);
    CHECK(f.sch.step() != nullptr);                        /* alone: the first interval */
    f.commit_plain(11);
    const uint64_t d = f.submit(100, 4000);                /* behind p in the running order */
    for (int k = 0; k < 3; ++k) { CHECK(f.sch.step() != nullptr); f.commit_plain(11); }
    for (int mt : { 1088, 576, 2112 }) {
        CHECK(DstSwitches::set("sched_mixed_tok", mt == 2112 ? 0 : mt));
        int wide = 0, dropped = 0, chunks = 0;
        for (int k = 0; k < 6; ++k) {
            const DstStep s = dst_step(f, p, { d });
            if (s.kind != 'C') continue;
            ++chunks;
            wide += s.tok > mt;
            dropped += !s.all_dec;
        }
        CHECK(chunks >= 5);
        CHECK_EQ(wide, 0);
        CHECK_EQ(dropped, 0);
    }
}

/* Under option D (sched_bi_chunk 3 / 4) a smaller budget cuts on the grid it holds: 1088 -> 1024-row and 576 -> 512-row
 * mixed chunks on their grid (3 too: no wait for a 2048 cut the budget cannot hold), 2112 -> 2048; every piece >= 128
 * rows; the plan is a function of the switches, not of how many decode rows ride along (1 or 4 decoders, the same
 * boundaries). */
TEST(dst_mixed_tok_cuts_on_its_grid_under_option_d) {
    /* alone's pieces of the 20000-token prompt (an idle engine's plan_chunk walk): a piece under 128 rows is allowed
     * only where alone has it too */
    std::vector<std::pair<int64_t, int64_t>> alone;
    {
        Fixture a;
        dst_fixture(a, 0);
        for (int64_t n = 0; n < 20000; ) {
            int64_t ck = -1;
            const int64_t c = a.sch.plan_chunk(n, 20000, a.cfg.max_tok, &ck);
            if (c <= 0) break;
            alone.push_back({ n, n + c });
            n += c;
        }
        CHECK(alone.size() >= 5);
        CHECK_EQ(alone.back().second, (int64_t)20000);
    }
    auto alone_end = [&](int64_t e) {
        for (const auto& pc : alone) if (pc.second == e) return true;
        return false;
    };
    auto alone_piece = [&](int64_t b, int64_t e) {
        for (const auto& pc : alone) if (pc.first == b && pc.second == e) return true;
        return false;
    };
    for (int opt : { 3, 4 })
        for (int mt : { 2112, 1088, 576 }) {
            std::vector<int64_t> ends[2];
            for (int nd = 0; nd < 2; ++nd) {
                DstSwitches restore;
                Fixture f;
                dst_fixture(f, 0);
                CHECK(DstSwitches::set("sched_bi_chunk", opt));
                CHECK(DstSwitches::set("sched_mixed_tok", mt == 2112 ? 0 : mt));
                std::vector<uint64_t> dec;
                for (int i = 0; i < (nd ? 4 : 1); ++i) dec.push_back(f.submit(100, 4000));
                CHECK(f.sch.step() != nullptr);
                f.commit_plain(11);
                const uint64_t p = f.submit(20000, 4);
                const int64_t want = mt == 2112 ? 2048 : mt == 1088 ? 1024 : 512;
                int off = 0, small = 0, chunks = 0, starved = 0;
                int64_t done = 0;
                for (int k = 0; k < 200 && done < 20000; ++k) {
                    const DstStep s = dst_step(f, p, dec);
                    if (s.kind != 'C') { ++starved; continue; }
                    ++chunks;
                    done = s.at + s.n;
                    ends[nd].push_back(done);
                    off += !(s.n <= want && (done % want == 0 || alone_end(done)));
                    small += s.n < 128 && !alone_piece(s.at, done);
                }
                CHECK_EQ(done, (int64_t)20000);
                CHECK(chunks >= 20000 / want);
                CHECK_EQ(off, 0);
                CHECK_EQ(small, 0);
                CHECK_EQ(starved, 0);
            }
            CHECK(ends[0] == ends[1]);
        }
}

/* Set to the flag's count and to the geometry's budget, the switches plan every step as unset does. */
TEST(dst_switches_at_their_start_values_plan_as_unset) {
    std::vector<std::vector<std::pair<uint64_t, int32_t>>> plans[2];
    for (int m = 0; m < 2; ++m) {
        DstSwitches restore;
        Fixture f;
        dst_fixture(f, 8);
        if (m) { CHECK(DstSwitches::set("sched_interleave", 8)); CHECK(DstSwitches::set("sched_mixed_tok", 2112)); }
        f.submit(100, 4000);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        f.submit(9000, 4);
        f.submit(700, 4);
        for (int k = 0; k < 80; ++k) {
            if (!f.sch.step()) break;
            std::vector<std::pair<uint64_t, int32_t>> row;
            for (const StepEntry& e : f.sch.plan().e) row.push_back({ e.seq, e.n_tokens });
            plans[m].push_back(row);
            f.commit_plain(12);
        }
    }
    CHECK(plans[0].size() >= 40);
    CHECK(plans[0] == plans[1]);
}

/* ================================================== lever DST: sched_short_bypass
 * On a step the interleave holds, a waiting request whose uncached rest is one alone piece of at most N rows runs that
 * piece whole; the long prefill stays held and the held step still counts toward the hold. */
TEST(dst_short_bypass_check_and_report) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    std::string err;
    CHECK_EQ(fn4::value(fn4::kSchedShortBypass), 0);
    CHECK(fn4::check("sched_short_bypass", 0, &err));
    CHECK(fn4::check("sched_short_bypass", 1280, &err));
    CHECK(fn4::check("sched_short_bypass", 2112, &err));
    CHECK(fn4::check("sched_short_bypass", 4160, &err));     /* up to --max-num-batched-tokens */
    CHECK(!fn4::check("sched_short_bypass", 4224, &err));
    CHECK(!fn4::check("sched_short_bypass", 1000, &err));    /* not a multiple of the quantum (64 here) */
    CHECK(!fn4::check("sched_short_bypass", -1, &err));
    CHECK(DstSwitches::set("sched_short_bypass", 1280));
    CHECK_EQ(fn4::value(fn4::kSchedShortBypass), 1280);
    CHECK_EQ(fn4::sched_short_bypass(), 1280);
    CHECK(DstSwitches::set("sched_short_bypass", 0));
    CHECK_EQ(fn4::sched_short_bypass(), 0);
    CHECK_EQ(fn4::value(fn4::kSchedBypassTotal), 0);
    CHECK(fn4::check("sched_bypass_total", 4096, &err));
    CHECK(fn4::check("sched_bypass_total", 65536, &err));
    CHECK(!fn4::check("sched_bypass_total", 1000, &err));    /* not a multiple of the quantum */
    CHECK(!fn4::check("sched_bypass_total", 65600, &err));
    CHECK(DstSwitches::set("sched_bypass_total", 4096));
    CHECK_EQ(fn4::sched_bypass_total(), 4096);
    CHECK(DstSwitches::set("sched_bypass_total", 0));
    CHECK_EQ(fn4::sched_bypass_total(), 0);
}

namespace {
/* A decoder, a 40000-token prompt p, and -- on the first held step after p's first chunk -- newcomers of the given lengths.
 * Per request: its pieces (start, rows) in order and the step each started; p's step shape (C / D). */
struct BypassRun {
    std::string shape;
    std::map<uint64_t, std::vector<std::pair<int64_t, int64_t>>> pieces;
    std::map<uint64_t, int> first_step;
    std::vector<uint64_t> ids;   /* the newcomers, in submit order */
    uint64_t p = 0;
    int64_t bypassed = 0;
};
BypassRun bypass_run(int bic, int bypass, const std::vector<int>& lens, int steps, bool seed_hit = false) {
    BypassRun out;
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    CHECK(DstSwitches::set("sched_bi_chunk", bic));
    CHECK(DstSwitches::set("sched_short_bypass", bypass));
    const uint64_t d = f.submit(100, 4000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    out.p = f.submit(40000, 4);
    bool sent = false;
    for (int k = 0; k < steps; ++k) {
        if (!f.sch.step()) break;
        bool chunk = false;
        for (const StepEntry& e : f.sch.plan().e) {
            if (!e.is_prefill) continue;
            if (e.seq == out.p) chunk = true;
            if (!out.pieces.count(e.seq)) out.first_step[e.seq] = k;
            out.pieces[e.seq].push_back({ e.ctx_len, e.n_tokens });
        }
        out.shape += chunk ? 'C' : 'D';
        f.commit_plain(12);
        if (!sent && chunk) {   /* p's first chunk ran: the next steps are held */
            if (seed_hit) { f.pc.seed(4096); f.pc.lin = 4096; f.pc.src = 3; }
            for (int n : lens) out.ids.push_back(f.submit(n, 64));
            sent = true;
        }
    }
    (void)d;
    f.sch.step_finished();   /* the engine's call after a step: it reports the bypass counts to fn4 */
    std::string g = fn4::state_json();
    const size_t at = g.find("\"bypassed\": ");
    out.bypassed = at == std::string::npos ? -1 : std::atoll(g.c_str() + at + 12);
    return out;
}
/* alone's pieces of an n-token prompt from `from` (an idle engine's plan_chunk walk) */
std::vector<std::pair<int64_t, int64_t>> alone_pieces(int64_t n, int64_t from = 0) {
    std::vector<std::pair<int64_t, int64_t>> v;
    Fixture a;
    dst_fixture(a, 0);
    for (int64_t c = from; c < n; ) {
        int64_t ck = -1;
        const int64_t q = a.sch.plan_chunk(c, n, a.cfg.max_tok, &ck);
        if (q <= 0) break;
        v.push_back({ c, q });
        c += q;
    }
    return v;
}
}  // namespace

/* A 1000-token newcomer arriving during a hold runs on the next (held) step as its alone piece, under every sched_bi_chunk
 * (without the bypass it runs at the next chunk step at the earliest -- under 1 / 3 / 4, whose pieces are whole or nothing,
 * not before the long prompt leaves it room, which in this window it never does);
 * a 3000-token one (over N) waits for the next chunk step as today; p's step shape is the same with and without the bypass,
 * and p's chunk boundaries too under 1 (A) and 3 (D3; option 4's give may move them along its grid). */
TEST(dst_short_bypass_admits_a_short_newcomer_on_a_held_step) {
    const auto alone1000 = alone_pieces(1000);
    CHECK(alone1000.size() == 1);
    for (int bic : { 0, 1, 3, 4 }) {
        const BypassRun off = bypass_run(bic, 0, { 1000, 3000 }, 40);
        const BypassRun on = bypass_run(bic, 1280, { 1000, 3000 }, 40);
        CHECK_EQ(on.shape, off.shape);
        const uint64_t q1 = on.ids[0], q2 = on.ids[1];
        const int c1 = (int)on.shape.find('C');
        /* the step a request's first piece ran, -1 when none ran in the window (option D can keep a newcomer waiting until
         * the long prompt ends: the starvation the bypass exists for) */
        auto fs = [](const BypassRun& r, uint64_t id) { auto it = r.first_step.find(id); return it == r.first_step.end() ? -1 : it->second; };
        CHECK_EQ(fs(on, q1), c1 + 1);                            /* the first held step */
        CHECK(on.pieces.count(q1) && on.pieces.at(q1) == alone1000);   /* one piece, alone's */
        const int off1 = fs(off, off.ids[0]);
        CHECK(off1 < 0 || off1 > c1 + 1);                         /* today: the next chunk step at the earliest */
        if (bic == 0) CHECK(off1 > c1 + 1);                       /* A (1), D (3 / 4): may wait for the whole prompt */
        const int s2 = fs(on, q2);                                /* over N: an ordinary admission, never on a held step */
        CHECK(s2 < 0 || (s2 > c1 + 1 && on.shape[(size_t)s2] == 'C'));
        CHECK_EQ(on.bypassed, 1);
        if (bic == 1 || bic == 3) CHECK(on.pieces.at(on.p) == off.pieces.at(off.p));
        if (bic == 4) for (const auto& pc : on.pieces.at(on.p)) CHECK_EQ((pc.first + pc.second) % 1024, 0);
        /* N = 2112: a 2000-token newcomer is one alone piece and fits the held step */
        const BypassRun big = bypass_run(bic, 2112, { 2000 }, 20);
        CHECK_EQ(fs(big, big.ids[0]), (int)big.shape.find('C') + 1);
        CHECK(big.pieces.count(big.ids[0]) && big.pieces.at(big.ids[0]) == alone_pieces(2000));
    }
}

/* A prefix hit with a short uncached tail (the Claude Code turn): taken on a held step where the tail is one alone piece
 * (4600 tokens, hit 4096: the 504-row rest), left for an ordinary admission where alone would split it (5000 tokens: alone
 * cuts at its tail checkpoint 4992, so the 904-row rest is two pieces). */
TEST(dst_short_bypass_takes_a_prefix_hit_tail_only_as_one_alone_piece) {
    const BypassRun on = bypass_run(0, 1280, { 4600, 5000 }, 30, /*seed_hit=*/true);
    const uint64_t qa = on.ids[0], qb = on.ids[1];
    const int c1 = (int)on.shape.find('C');
    CHECK_EQ(on.first_step.at(qa), c1 + 1);
    CHECK(on.pieces.at(qa) == alone_pieces(4600, 4096));
    CHECK(alone_pieces(5000, 4096).size() == 2);
    CHECK(on.first_step.at(qb) > c1 + 1);
    CHECK(on.shape[(size_t)on.first_step.at(qb)] == 'C');
}

/* ================================================== lever DST: sched_bypass_total (the bypass, piece by piece)
 * With sched_bypass_total T, the bypass takes a waiting request whose uncached rest is at most T and runs its alone plan
 * piece by piece, held steps included: alone's boundaries and checkpoints; under sched_bi_chunk 3 / 4 an alone piece over
 * the held step (a long conversation's 4096-row span piece) is cut on the interval grid inside it, >= 128 rows a side. */
namespace {
struct Piece { int64_t at, n, ck; int step; bool operator==(const Piece& o) const { return at == o.at && n == o.n && ck == o.ck; } };
struct TotalRun {
    std::string shape;
    std::map<uint64_t, std::vector<Piece>> pieces;
    std::vector<uint64_t> ids;
    uint64_t p = 0;
};
/* dst_fixture (+ prefix_exact / span_min when `ds`: the DS41 line's exact resume and its 16384-token span floor); a
 * decoder; a 40000-token prompt p; on the first held step after p's first chunk a prefix hit of `hit` tokens is seeded (0:
 * none) and the newcomers of `lens` arrive. Every prefill piece with its checkpoint position (-1: none). */
TotalRun total_run(int bic, int n, int total, const std::vector<int>& lens, int steps, int64_t hit, bool ds) {
    TotalRun out;
    DstSwitches restore;
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = 8;
    f.cfg.prefix_exact = ds;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
    if (ds) f.sch.set_span_min(16384);
    CHECK(DstSwitches::set("sched_bi_chunk", bic));
    CHECK(DstSwitches::set("sched_short_bypass", n));
    CHECK(DstSwitches::set("sched_bypass_total", total));
    f.submit(100, 4000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    out.p = f.submit(40000, 4);
    bool sent = false;
    for (int k = 0; k < steps; ++k) {
        if (!f.sch.step()) break;
        bool chunk = false;
        for (const StepEntry& e : f.sch.plan().e) {
            if (!e.is_prefill) continue;
            if (e.seq == out.p) chunk = true;
            out.pieces[e.seq].push_back({ e.ctx_len, e.n_tokens, e.ckpt_slot >= 0 ? e.ctx_len + e.ckpt_tok + 1 : -1, k });
        }
        out.shape += chunk ? 'C' : 'D';
        f.commit_plain(12);
        if (!sent && chunk) {
            if (hit > 0) { f.pc.seed((int32_t)hit); f.pc.lin = (int32_t)hit; f.pc.src = 3; }
            for (int l : lens) out.ids.push_back(f.submit(l, 64));
            sent = true;
        }
    }
    (void)DstSwitches::set("sched_bypass_total", 0);
    return out;
}
/* alone's pieces of an n-token prompt from `from`, with checkpoints, on total_run's geometry (an idle engine's walk: a
 * prompt of >= 16384 tokens spans under `ds`) */
std::vector<Piece> alone_ck(int64_t n, int64_t from, bool ds) {
    std::vector<Piece> v;
    Fixture a;
    a.build(8192, 64, 4160, 60000);
    a.cfg.checkpoint_interval = 2048;
    a.cfg.prefill_span = 2;
    a.cfg.prefix_exact = ds;
    CHECK_OK(a.start());
    const bool span = !ds || n >= 16384;
    for (int64_t c = from; c < n; ) {
        int64_t ck = -1;
        /* plan_chunk answers for a span outside a step; a prompt under the span floor is planned as one that does not */
        const int64_t q = span ? a.sch.plan_chunk(c, n, a.cfg.max_tok, &ck) : a.sch.plan_chunk(c, n, 2048, &ck);
        if (q <= 0) break;
        v.push_back({ c, q, ck, 0 });
        c += q;
    }
    return v;
}
bool same_pieces(const std::vector<Piece>& a, const std::vector<Piece>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!(a[i] == b[i])) return false;
    return true;
}
}  // namespace

/* The 904-row tail (prompt 5000, hit 4096) that one-piece bypass turns away (alone splits it at its tail checkpoint 4992, in
 * the fixture's nearest-resume geometry): with T 4096 it runs on two consecutive held steps as alone's two pieces with
 * alone's checkpoint; a rest over T (prompt 9000: 4904 rows) is left to an ordinary admission; p's steps keep their shape. */
TEST(dst_bypass_total_runs_a_split_tail_piece_by_piece) {
    const auto alone = alone_ck(5000, 4096, false);
    CHECK_EQ(alone.size(), (size_t)2);
    CHECK_EQ(alone[0].ck, 4992);
    for (int bic : { 0, 3 }) {
        TotalRun off = total_run(bic, 1280, 0, { 5000, 9000 }, 30, 4096, false);
        TotalRun on = total_run(bic, 1280, 4096, { 5000, 9000 }, 30, 4096, false);
        CHECK_EQ(on.shape, off.shape);
        const int c1 = (int)on.shape.find('C');
        const auto& qa = on.pieces[on.ids[0]];
        CHECK(same_pieces(qa, alone));
        CHECK(qa.size() == 2 && qa[0].step == c1 + 1 && qa[1].step == c1 + 2);   /* two held steps in a row */
        for (const auto& pc : on.pieces[on.ids[1]]) CHECK(on.shape[(size_t)pc.step] == 'C');   /* over T: never held */
        if (bic == 3) CHECK(same_pieces(on.pieces[on.p], off.pieces[off.p]));
    }
}

/* The DS41 geometry (exact resume, span from 16384 tokens). A fresh 3000-token prompt is alone's two interval pieces: two
 * held steps under 0 and 3. A long conversation's 3000-row tail (prompt 17336, hit 14336 on the resume grid) is ONE alone
 * span piece over the held step: turned away under 0 (an ordinary admission), cut on the interval (16384) under 3, so
 * 2048 + 952 rows on two held steps. The long prompt's plan is the same with and without the bypass under 3. */
TEST(dst_bypass_total_on_the_ds_geometry) {
    const auto fresh = alone_ck(3000, 0, true);
    CHECK_EQ(fresh.size(), (size_t)2);
    const auto tail = alone_ck(17336, 14336, true);
    CHECK_EQ(tail.size(), (size_t)1);                       /* alone: one 3000-row span piece */
    for (int bic : { 0, 3 }) {
        TotalRun off = total_run(bic, 2112, 0, { 3000 }, 30, 0, true);
        TotalRun on = total_run(bic, 2112, 4096, { 3000 }, 30, 0, true);
        CHECK_EQ(on.shape, off.shape);
        const int c1 = (int)on.shape.find('C');
        const auto& q = on.pieces[on.ids[0]];
        CHECK(same_pieces(q, fresh));
        CHECK(q.size() == 2 && q[0].step == c1 + 1 && q[1].step == c1 + 2);
        if (bic == 3) CHECK(same_pieces(on.pieces[on.p], off.pieces[off.p]));
        TotalRun hit = total_run(bic, 2112, 4096, { 17336 }, 30, 14336, true);
        const auto& h = hit.pieces[hit.ids[0]];
        CHECK(!h.empty());
        if (bic == 0) {
            for (const auto& pc : h) CHECK(hit.shape[(size_t)pc.step] == 'C');   /* turned away: ordinary admission */
        } else {
            CHECK(h.size() >= 2);
            CHECK_EQ(h[0].at, (int64_t)14336);
            CHECK_EQ(h[0].n, (int64_t)2048);                 /* cut on the interval, >= 128 rows a side */
            CHECK_EQ(h[1].at, (int64_t)16384);
            CHECK_EQ(h[1].n, (int64_t)952);
            CHECK_EQ(h[1].ck, tail[0].ck);                   /* alone's checkpoint at the piece's end */
            CHECK(h[0].step == (int)hit.shape.find('C') + 1 && h[1].step == h[0].step + 1);
        }
    }
}

namespace {
int64_t dst_json_num(const std::string& g, size_t from, const char* key) {
    if (from == std::string::npos) return -1;
    const std::string k = std::string("\"") + key + "\": ";
    const size_t at = g.find(k, from);
    return at == std::string::npos ? -1 : std::atoll(g.c_str() + at + k.size());
}
}  // namespace

/* A recycled slot starts with the bypass's per-request state cleared (ds41-noa: Scheduler::add_locked): a 3000-token request the
 * bypass turned away (sched_short_bypass 1280, one-piece only) finishes and is reaped; the 1000-token request that takes
 * its slot is bypassed on the next held step (before the fix its stale bypass_no kept it off every held step, counted
 * neither bypassed nor turned away). */
TEST(dst_a_recycled_slot_starts_with_clean_bypass_state) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    CHECK(DstSwitches::set("sched_bi_chunk", 0));   /* today's cuts: the turned-away request runs as an ordinary prefill */
    CHECK(DstSwitches::set("sched_short_bypass", 1280));
    f.submit(100, 4000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    const uint64_t p = f.submit(60000 - 200, 2);
    REQUIRE(p != 0);
    uint64_t x = 0;
    int slot_x = -1;
    auto slot_of = [&](uint64_t id) {
        std::vector<Scheduler::ReqStat> st(16);
        const int n = f.sch.requests(st.data(), 16);
        for (int j = 0; j < n; ++j) if (st[(size_t)j].id == id) return st[(size_t)j].slot;
        return -1;
    };
    bool chunk = false;
    for (int s = 0; s < 400; ++s) {
        REQUIRE(f.sch.step() != nullptr);
        chunk = false;
        for (const StepEntry& e : f.sch.plan().e) chunk = chunk || (e.seq == p && e.is_prefill);
        f.commit_plain(12);
        if (!x && chunk) { x = f.submit(3000, 2); slot_x = slot_of(x); }   /* turned away on the held steps */
        if (x && f.sch.find(x) && f.sch.find(x)->state == ReqState::Finished) break;
    }
    REQUIRE(x != 0 && slot_x >= 0);
    REQUIRE(f.sch.find(x) && f.sch.find(x)->state == ReqState::Finished);
    CHECK_OK(f.sch.reap(x));
    f.sch.step_finished();
    std::string g = fn4::state_json();
    const int64_t by0 = dst_json_num(g, g.find("\"dst\": {"), "bypassed");
    CHECK(dst_json_num(g, g.find("\"dst\": {"), "bypass_refused") >= 1);
    /* the next chunk step of p, then Y arrives during the hold */
    for (int s = 0; s < 40; ++s) {
        REQUIRE(f.sch.step() != nullptr);
        chunk = false;
        for (const StepEntry& e : f.sch.plan().e) chunk = chunk || (e.seq == p && e.is_prefill);
        f.commit_plain(12);
        if (chunk) break;
    }
    REQUIRE(chunk);
    const uint64_t y = f.submit(1000, 2);
    REQUIRE(y != 0);
    CHECK_EQ(slot_of(y), slot_x);                                  /* the reuse this case is about */
    REQUIRE(f.sch.step() != nullptr);
    bool y_in = false, p_in = false;
    for (const StepEntry& e : f.sch.plan().e) {
        y_in = y_in || (e.seq == y && e.is_prefill && e.ctx_len == 0 && e.n_tokens == 1000);
        p_in = p_in || (e.seq == p && e.is_prefill);
    }
    CHECK(y_in);                                                   /* bypassed on the held step */
    CHECK(!p_in);
    f.commit_plain(12);
    f.sch.step_finished();
    g = fn4::state_json();
    CHECK_EQ(dst_json_num(g, g.find("\"dst\": {"), "bypassed"), by0 + 1);
}

/* ================================================== lever PSK design 2 on every admission path (ds41-int7-pskglue)
 * The DS41 line's geometry (total_run's `ds`: exact resume, interval 2048, span 2, span floor 16384, MNBT 4160): targets
 * q_i = 2048 + (i - 1) x 4096 (alone's span ends), the tail snapshot, the end, checkpoint_at. A run: four decoders, a long
 * prompt p beside them, on the first held step after p's first chunk the newcomers `lens`; every prefill piece with its
 * checkpoint and its plan (`ahead`, kPskNoPlan when unplanned), the snapshots committed, the scheduler's metrics. */
namespace {
struct PskPiece { int64_t at, n, ck; int32_t ahead; int step; };
struct PskModeRun {
    std::string shape;
    std::map<uint64_t, std::vector<PskPiece>> pieces;
    std::vector<int64_t> committed;
    std::vector<uint64_t> ids;
    uint64_t p = 0;
    int planned_steps = 0;
    SchedMetrics m{};
};
PskModeRun psk_mode_run(int mode, int k, int bic, int bypass, int total, int64_t L, const std::vector<int>& lens, int steps,
                        int64_t hit = 0, int64_t hit_lin = 0) {
    PskModeRun out;
    DstSwitches restore;
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = 8;
    f.cfg.prefix_exact = true;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
    f.sch.set_span_min(16384);
    CHECK(DstSwitches::set("sched_bi_chunk", bic));
    CHECK(DstSwitches::set("sched_short_bypass", bypass));
    CHECK(DstSwitches::set("sched_bypass_total", total));
    if (mode >= 0) f.sch.set_psk_mode(mode);
    f.sch.set_psk_k(k);
    for (int d = 0; d < 4; ++d) f.submit(100, 4000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    out.p = f.submit((int)L, 2);
    bool sent = false;
    for (int s = 0; s < steps; ++s) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        std::vector<int32_t> ahead, rows;
        int64_t dec = -1;
        const bool planned = f.sch.psk_plan(b->step, &ahead, &rows, &dec);
        out.planned_steps += planned ? 1 : 0;
        bool chunk = false;
        size_t pi = 0;
        for (const StepEntry& e : f.sch.plan().e) {
            if (!e.is_prefill) continue;
            if (e.seq == out.p) chunk = true;
            const int32_t a = planned && pi < ahead.size() ? ahead[pi] : Scheduler::kPskNoPlan;
            if (planned) CHECK(pi < rows.size() && rows[pi] == e.n_tokens);
            out.pieces[e.seq].push_back({ e.ctx_len, e.n_tokens, e.ckpt_slot >= 0 ? e.ctx_len + e.ckpt_tok + 1 : -1, a, s });
            ++pi;
        }
        out.shape += chunk ? 'C' : 'D';
        f.commit_plain(12);
        if (!sent && chunk) {
            if (hit > 0) { f.pc.seed((int32_t)hit); f.pc.lin = (int32_t)hit_lin; f.pc.src = 3; }
            for (int l : lens) out.ids.push_back(f.submit(l, 4));
            sent = true;
        }
    }
    out.committed = f.pc.committed;
    out.m = f.sch.metrics();
    (void)DstSwitches::set("sched_bypass_total", 0);
    return out;
}
/* the targets of an L-token prompt from `from` (K = 2 here: q_2 = 6144, q_4 = 14336, ...; the tail; L) */
std::vector<int64_t> psk_targets_of(int64_t L, int k) {
    std::vector<int64_t> t;
    for (int64_t i = k; 2048 + (i - 1) * 4096 < L; i += k) t.push_back(2048 + (i - 1) * 4096);
    const int64_t tail = L - 1 >= 2048 ? 2048 + (L - 1 - 2048) / 4096 * 4096 : -1;
    if (tail > 2048 && std::find(t.begin(), t.end(), tail) == t.end()) t.push_back(tail);
    t.push_back(L);
    std::sort(t.begin(), t.end());
    return t;
}
/* every piece planned with ahead = (the first target past its start) - its end */
void psk_check_plan(const std::vector<PskPiece>& pc, const std::vector<int64_t>& tg, const char* what) {
    for (const PskPiece& x : pc) {
        int64_t t1 = -1;
        for (int64_t t : tg) if (t > x.at) { t1 = t; break; }
        const int64_t want = t1 - (x.at + x.n);
        CHECK(x.ahead != Scheduler::kPskNoPlan);
        CHECK_EQ((int64_t)x.ahead, want);
        if (x.ahead == Scheduler::kPskNoPlan || (int64_t)x.ahead != want)
            fprintf(stderr, "    %s: piece [%lld, %lld) ahead %d, want %lld\n", what, (long long)x.at, (long long)(x.at + x.n),
                    (int)x.ahead, (long long)want);
    }
}
}  // namespace

/* sched_bi_chunk 3 (option D) beside decode rows, ds41_psk 2, K 2: p (20000 tokens, over the span floor: alone's 4096-row
 * spans, which D cuts on the 2048 grid beside decode rows; 4 / 1 / 0 as they plan) is planned at every piece; its snapshots are its targets only (6144 = q_2, 14336 = q_4, the tail 18432 = q_5),
 * each one of D's chunk ends; D's pieces are the same as under ds41_psk 0; a prefix hit at 16384 (the deepest snapshot
 * below it the anchor 14336) resumes there, planned, and is counted as a fallback. */
TEST(psk_glue_option_d_plans_every_piece_and_snapshots_at_targets) {
    const int64_t L = 20000;
    const auto tg = psk_targets_of(L, 2);
    CHECK((tg == std::vector<int64_t>{ 6144, 14336, 18432, 20000 }));
    for (int bic : { 3, 4, 1, 0 }) {
        const PskModeRun off = psk_mode_run(0, 2, bic, 0, 0, L, {}, 240);
        const PskModeRun on = psk_mode_run(2, 2, bic, 0, 0, L, {}, 240);
        const auto& po = on.pieces.at(on.p);
        const auto& pf = off.pieces.at(off.p);
        CHECK_EQ(po.size(), pf.size());
        int64_t sum = 0;
        std::vector<int64_t> ends, cks;
        for (size_t i = 0; i < po.size() && i < pf.size(); ++i) {
            CHECK(po[i].at == pf[i].at && po[i].n == pf[i].n);   /* chunking untouched */
            sum += po[i].n;
            ends.push_back(po[i].at + po[i].n);
            if (po[i].ck > 0) cks.push_back(po[i].ck);
        }
        CHECK_EQ(sum, L);
        psk_check_plan(po, tg, "option D");
        CHECK((cks == std::vector<int64_t>{ 6144, 14336, 18432 }));
        for (int64_t t : tg) CHECK(std::find(ends.begin(), ends.end(), t) != ends.end());   /* anchors on D's chunk ends */
        std::vector<int64_t> want_off;                               /* today: every grid point a piece ends on */
        for (const PskPiece& x : pf) if (x.ck > 0) want_off.push_back(x.ck);
        CHECK((want_off == std::vector<int64_t>{ 2048, 6144, 10240, 14336, 18432 }));
        CHECK_EQ(on.m.psk_latched, (int64_t)5);                   /* the four decoders' prompts and p */
        CHECK(on.m.psk_planned_steps > 0);
        CHECK_EQ(on.m.psk_ckpt_anchor, (int64_t)3);
        CHECK_EQ(on.m.psk_ckpt_dropped, (int64_t)2);
        CHECK_EQ(on.m.psk_mode, 2);
        CHECK_EQ(on.m.psk_k, 2);
        CHECK(on.shape == off.shape);
    }
    /* the hit: a 24000-token newcomer whose match the cache gives as 16384 attention / 14336 linear (the anchor below) */
    const PskModeRun hit = psk_mode_run(2, 2, 3, 0, 0, L, { 24000 }, 400, 16384, 14336);
    REQUIRE(hit.ids.size() == 1);
    const auto& ph = hit.pieces.at(hit.ids[0]);
    REQUIRE(!ph.empty());
    CHECK_EQ(ph[0].at, (int64_t)14336);
    psk_check_plan(ph, psk_targets_of(24000, 2), "hit");
    CHECK_EQ(hit.m.psk_fallback_resumes, (int64_t)1);
    CHECK_EQ(hit.m.psk_fallback_tokens, (int64_t)(16384 - 14336));
    CHECK_EQ(hit.m.psk_latched, (int64_t)6);
}

/* sched_short_bypass (and sched_bypass_total): a 1000-token newcomer runs its alone piece on a held step and is planned
 * there (its one target its end: ahead 0); a 3000-token one through the bypass total (two alone pieces on held steps), both
 * planned; under sched_bi_chunk 0 and 3. */
TEST(psk_glue_a_bypassed_newcomer_gets_its_plan) {
    for (int bic : { 0, 3 }) {
        const PskModeRun on = psk_mode_run(2, 8, bic, 1280, 0, 40000, { 1000 }, 40);
        REQUIRE(on.ids.size() == 1);
        const int c1 = (int)on.shape.find('C');
        REQUIRE(on.pieces.count(on.ids[0]));
        const auto& q = on.pieces.at(on.ids[0]);
        REQUIRE(q.size() == 1);
        CHECK_EQ(q[0].step, c1 + 1);                         /* the held step: the bypass took it */
        CHECK_EQ(q[0].n, (int64_t)1000);
        CHECK_EQ((int64_t)q[0].ahead, (int64_t)0);
        CHECK_EQ(on.m.psk_latched, (int64_t)6);                   /* four decoders, p, the newcomer */
        const PskModeRun tot = psk_mode_run(2, 8, bic, 2112, 4096, 40000, { 3000 }, 40);
        REQUIRE(tot.ids.size() == 1 && tot.pieces.count(tot.ids[0]));
        const auto& t = tot.pieces.at(tot.ids[0]);
        CHECK(t.size() == 2);
        CHECK_EQ(t[0].step, (int)tot.shape.find('C') + 1);
        psk_check_plan(t, psk_targets_of(3000, 8), "bypass total");
    }
}

/* A shared head (CacheHit::branch_at): the request that finds it resumes at the anchor below (here none: 0) and takes a
 * snapshot at the head (Request::checkpoint_at) -- under ds41_psk 2 a target, kept, its cone run -- so every later
 * request with that head resumes exactly there at any K. A 30000-token prompt, head 20480, K 8: snapshots at the head and
 * the tail (26624) only; under ds41_psk 0 the head beside every grid point. */
TEST(psk_glue_a_shared_head_becomes_a_target) {
    for (int mode : { 2, 0 }) {
        Fixture f;
        f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
        f.cfg.checkpoint_interval = 2048;
        f.cfg.prefill_span = 2;
        f.cfg.prefix_exact = true;
        CHECK_OK(f.start());
        f.sch.set_span_min(16384);
        f.sch.set_psk_mode(mode);
        f.sch.set_psk_k(8);
        f.pc.seed(20480);
        f.pc.branch_at = 20480;
        const uint64_t a = f.submit(30000, 1);
        std::vector<int32_t> aheads;
        for (int k = 0; k < 40 && f.sch.find(a); ++k) {
            const RadBatch* b = f.sch.step();
            if (!b) break;
            std::vector<int32_t> ahead, rows;
            int64_t dec = 0;
            if (f.sch.psk_plan(b->step, &ahead, &rows, &dec)) aheads.push_back(ahead[0]);
            if (k == 0) CHECK_EQ(f.sch.find(a)->checkpoint_at, (int64_t)20480);
            f.commit_plain();
        }
        if (mode == 2) {
            CHECK((f.pc.committed == std::vector<int64_t>{ 20480, 26624 }));
            CHECK(!aheads.empty());
            CHECK_EQ(f.sch.metrics().psk_ckpt_anchor, (int64_t)2);
        } else {
            CHECK((f.pc.committed == std::vector<int64_t>{ 2048, 6144, 10240, 14336, 18432, 20480, 26624 }));
            CHECK(aheads.empty());
        }
    }
}

/* Unset, ds41_psk 0 and 1: no request latched, no plan, every snapshot as today, the counters 0 -- under every
 * sched_bi_chunk and with the bypass on and off; the same pieces and snapshots as unset. */
TEST(psk_glue_unset_0_and_1_change_nothing_in_any_mode) {
    for (int bic : { 0, 1, 2, 3, 4 })
        for (int byp : { 0, 1280 }) {
            const PskModeRun u = psk_mode_run(-1, 8, bic, byp, 0, 20000, { 1000, 3000 }, 60);
            for (int mode : { 0, 1 }) {
                const PskModeRun r = psk_mode_run(mode, 8, bic, byp, 0, 20000, { 1000, 3000 }, 60);
                CHECK(r.shape == u.shape);
                CHECK(r.committed == u.committed);
                CHECK_EQ(r.pieces.size(), u.pieces.size());
                for (const auto& kv : r.pieces) {
                    const auto it = u.pieces.find(kv.first);
                    REQUIRE(it != u.pieces.end());
                    CHECK_EQ(kv.second.size(), it->second.size());
                    for (size_t i = 0; i < kv.second.size() && i < it->second.size(); ++i) {
                        CHECK(kv.second[i].at == it->second[i].at && kv.second[i].n == it->second[i].n &&
                              kv.second[i].ck == it->second[i].ck);
                        CHECK(kv.second[i].ahead == Scheduler::kPskNoPlan);
                    }
                }
                CHECK_EQ(r.planned_steps, 0);
                CHECK_EQ(r.m.psk_latched + r.m.psk_planned_steps + r.m.psk_ckpt_anchor + r.m.psk_ckpt_dropped +
                         r.m.psk_fallback_resumes + r.m.psk_fallback_tokens, (int64_t)0);
                CHECK_EQ(r.m.psk_mode, mode);
            }
            CHECK_EQ(u.planned_steps, 0);
            CHECK_EQ(u.m.psk_latched + u.m.psk_planned_steps + u.m.psk_ckpt_anchor + u.m.psk_ckpt_dropped, (int64_t)0);
            CHECK_EQ(u.m.psk_mode, -1);
        }
}

/* ================================================== lever DST: a long prompt beside c4, timed (not a check)
 * A measurement harness for the DST switches, run on request only:
 *
 *   RAD_SCHED_SIM_DST=1 [SIM_IL=8 SIM_MT=2112 SIM_BIC=0 SIM_BYP=0 SIM_LEN=65650 SIM_WARM=10 SIM_PF0=0.44 SIM_PFB=0.26e-3
 *                        SIM_DEC0=0.022 SIM_DECR=0.0015 SIM_PFSYNC=0 SIM_TRACE=1 SIM_SHAPE=1 SIM_AFTER=1] sched_test sim_dst
 *
 * SIM_SHAPE prints every step of the window (C / D, its entries); SIM_AFTER runs 60 s past the probe's first token and
 * prints every c4 request sent after the probe with its TTFT (a turnover the window does not see whole).
 *
 * r00z's loaded arm on the DS41 line's geometry (MNBT 4160, interval 2048, span 2, the stager's 1024-row floor, MAXSEQS 8,
 * --busy-prefill-chunk auto): four closed-loop decoders (prompts 935 .. 1280 tokens, 512 tokens each, then the next
 * request at once, as mb.py's c4) decode one token a step (r00z's window ran at depth 0); SIM_WARM s after they start, one
 * SIM_LEN-token prompt arrives. Step cost in seconds: SIM_DEC0 + SIM_DECR per decode row, plus SIM_PF0 + SIM_PFB per
 * prefill row when the step carries prefill rows (+ SIM_PFSYNC when they are fewer than the stager's 1024: its experts
 * read without overlap). The defaults are fitted to r00z A (a 2048-row mixed step 0.97 s, a 4096-row lone step ~1.5 s,
 * a c4 decode step ~28 ms): the schedule is the scheduler's own, the times an estimate. Over the long prompt's prefill
 * window it prints the c4 tokens a second, the gaps inside a request (p50 / p90 / p99 / max), the c4 requests' TTFT
 * (the turnovers r00z's max TPOT was), the long prompt's TTFT, and its chunks (count, sizes, off the 1024 grid). */
TEST(sim_dst_long_prompt_beside_c4) {
    if (!std::getenv("RAD_SCHED_SIM_DST")) return;
    const int64_t IL = (int64_t)sim_env("SIM_IL", 8), MT = (int64_t)sim_env("SIM_MT", 2112);
    const int BIC = (int)sim_env("SIM_BIC", 0);
    const int64_t L = (int64_t)sim_env("SIM_LEN", 65650);
    const double warm = sim_env("SIM_WARM", 10), pf0 = sim_env("SIM_PF0", 0.44), pfb = sim_env("SIM_PFB", 0.26e-3);
    const double dec0 = sim_env("SIM_DEC0", 0.022), decr = sim_env("SIM_DECR", 0.0015), pfsync = sim_env("SIM_PFSYNC", 0);
    DstSwitches restore;
    Fixture f;
    f.build(/*blocks=*/16384, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/L + 8192);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = 8;
    f.cfg.max_seqs = 8;
    f.cfg.prefix_exact = true;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
    f.sch.set_span_min(16384);
    CHECK(DstSwitches::set("sched_interleave", (int)IL));
    CHECK(DstSwitches::set("sched_mixed_tok", MT >= 2112 ? 0 : (int)MT));
    CHECK(DstSwitches::set("sched_bi_chunk", BIC));
    CHECK(DstSwitches::set("sched_short_bypass", (int)sim_env("SIM_BYP", 0)));
    static const int kPrompts[] = { 1048, 935, 1247, 1021, 1180, 1280, 1129, 999, 1089, 1271, 1139, 971 };
    int seed = 1, next_prompt = 0;
    auto req = [&](int n, int out) {
        auto r = f.request(n, out);
        for (size_t k = 0; k < r->prompt.size(); ++k)
            r->prompt[k] = 1000 + (int32_t)(((int64_t)k * 7 + seed * 131) % 60000);
        ++seed;
        r->ignore_eos = true;
        return f.add(std::move(r));
    };
    struct Dec { uint64_t id; double sent; double first; double last; };
    std::vector<Dec> dec;
    for (int i = 0; i < 4; ++i) dec.push_back({ req(kPrompts[next_prompt++ % 12], 512), 0.0, -1.0, -1.0 });
    uint64_t probe = 0;
    double t = 0, p_sent = -1, p_first = -1;
    std::vector<double> gin, nttft;
    std::vector<std::pair<double, double>> turn;   /* c4 requests sent after the probe: (sent - probe sent, their TTFT) */
    int64_t c4_win = 0, chunks = 0, off = 0, steps = 0;
    std::map<int64_t, int> sizes;
    while (steps < 400000) {
        if (!probe && t >= warm) { probe = req((int)L, 16); p_sent = t; }
        if (!f.sch.step()) break;
        const StepPlan& p = f.sch.plan();
        int64_t P = 0, nd = 0;
        for (const StepEntry& e : p.e) {
            if (e.is_prefill) P += e.n_tokens; else ++nd;
            if (probe && e.seq == probe && e.is_prefill) {
                ++chunks;
                ++sizes[e.n_tokens];
                const int64_t end = e.ctx_len + e.n_tokens;
                off += end % 1024 != 0 && end != L;
            }
        }
        const double dt = (nd ? dec0 + decr * (double)nd : 0.0) +
                          (P ? pf0 + pfb * (double)P + (P < 1024 ? pfsync : 0.0) : 0.0);
        if (std::getenv("SIM_SHAPE") && p_sent >= 0 && p_first < 0 && steps < 4000) {
            std::printf("%lld %.2f %s:", (long long)steps, t, P ? "C" : "D");
            for (const StepEntry& e : p.e)
                std::printf(" %llu:%s%d@%d", (unsigned long long)e.seq, e.is_prefill ? "P" : "d", e.n_tokens, e.ctx_len);
            std::printf("\n");
        }
        if (std::getenv("SIM_TRACE") && p_sent >= 0 && p_first < 0 && steps % 7 == 0) {
            std::printf("step %lld t=%.2f:", (long long)steps, t);
            for (const StepEntry& e : p.e)
                std::printf(" %llu:%s%d@%d", (unsigned long long)e.seq, e.is_prefill ? "P" : "d", e.n_tokens, e.ctx_len);
            std::printf("\n");
        }
        std::vector<std::pair<uint64_t, size_t>> before;
        for (const StepEntry& e : p.e) before.push_back({ e.seq, e.req->output.size() });
        CHECK_OK(f.commit(7));
        ++steps;
        t += dt;
        const bool win = p_sent >= 0 && p_first < 0;
        for (const auto& b : before) {
            const Request* r = f.sch.find(b.first);
            if (!r || r->output.size() <= b.second) continue;
            if (b.first == probe) { if (p_first < 0) p_first = t; continue; }
            for (Dec& d : dec) {
                if (d.id != b.first) continue;
                if (d.first < 0) {
                    d.first = t;
                    if (p_sent >= 0 && d.sent >= p_sent && (p_first < 0 || d.sent <= p_first)) nttft.push_back(t - d.sent);
                    if (p_sent >= 0 && d.sent >= p_sent) turn.push_back({ d.sent - p_sent, t - d.sent });
                } else if (win) {
                    gin.push_back(1000.0 * (t - d.last));
                }
                if (win) c4_win += (int64_t)(r->output.size() - b.second);
                d.last = t;
            }
        }
        std::vector<uint64_t> c;
        f.sch.take_completed(&c);
        for (uint64_t id : c) {
            (void)f.sch.reap(id);
            for (Dec& d : dec)
                if (d.id == id) d = { req(kPrompts[next_prompt++ % 12], 512), t, -1.0, -1.0 };
        }
        if (p_first >= 0 && (t > p_first + 60 || !std::getenv("SIM_AFTER"))) break;
    }
    if (std::getenv("SIM_AFTER")) {
        std::printf("sim_dst turnovers (sent after the probe, s; their TTFT, s; probe TTFT %.1f):", p_first - p_sent);
        for (const auto& x : turn) std::printf(" +%.1f:%.1f", x.first, x.second);
        std::printf("\n");
    }
    std::sort(gin.begin(), gin.end());
    std::sort(nttft.begin(), nttft.end());
    auto pc = [&](const std::vector<double>& v, double q) { return v.empty() ? 0.0 : v[std::min(v.size() - 1, (size_t)(q * (double)v.size()))]; };
    std::string sz;
    for (const auto& kv : sizes) sz += std::to_string(kv.first) + "x" + std::to_string(kv.second) + " ";
    const double w = p_first - p_sent;
    std::printf("sim_dst IL %lld MT %lld BIC %d BYP %d: TTFT %.1f s, %lld chunks [%s], %lld ending off the 1024 grid; c4 in the "
                "window %.1f tok/s; TPOT in a request p50 %.0f p90 %.0f p99 %.0f max %.0f ms; c4 requests sent in the window "
                "%zu, TTFT max %.1f s\n",
                (long long)IL, (long long)MT, BIC, (int)sim_env("SIM_BYP", 0), w, (long long)chunks, sz.c_str(), (long long)off,
                w > 0 ? (double)c4_win / w : 0.0, pc(gin, 0.5), pc(gin, 0.9), pc(gin, 0.99), gin.empty() ? 0.0 : gin.back(),
                nttft.size(), nttft.empty() ? 0.0 : nttft.back());
}

/* ------------------------------------------------------------------ lever BI class (ii): sched_bi_chunk 3 (option D)
 * The fixture's geometry: interval 256, a span of two (512 alone), max_tok 576, so the shared budget is 576 - 256 = 320:
 * 64 rows of slack beside a 256-row grid piece, as the DS line's 2112 = 2048 + 64. D cuts alone's pieces on the interval
 * grid inside them, >= 128 rows a side; alone's own pieces otherwise. The switch is the core fn4 switch the scheduler
 * reads at every step; each test sets it and puts it back to 0 (the process's other tests run today's chunking). */
namespace {
struct BiChunk {
    explicit BiChunk(int v) { set(v); }
    ~BiChunk() { set(0); }
    static void set(int v) {
        std::string err;
        CHECK(fn4::request("sched_bi_chunk", v, &err));
        int sw[fn4::kSwitches];
        (void)fn4::apply_pending(-1, sw);
        CHECK_EQ(fn4::sched_bi_chunk(), v);
    }
};
/* Run `steps` steps of the fixture; the long request's prefill pieces (start, rows) in order. */
std::vector<std::pair<int64_t, int64_t>> bi_pieces(Fixture& f, uint64_t id, int steps) {
    std::vector<std::pair<int64_t, int64_t>> out;
    int64_t at = 0;
    for (int k = 0; k < steps; ++k) {
        if (!f.sch.step()) break;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == id && e.is_prefill) {
                out.emplace_back(at, e.n_tokens);
                at += e.n_tokens;
            }
        f.commit_plain();
    }
    return out;
}
/* Alone's pieces of an L-token prompt at this geometry: the first interval, then spans of two, the rest at the end. */
std::vector<int64_t> bi_alone_ends(int64_t L) {
    std::vector<int64_t> e;
    for (int64_t x = 256; x < L; x += 512) e.push_back(x);
    e.push_back(L);
    return e;
}
void bi_check_d(const std::vector<std::pair<int64_t, int64_t>>& pc, int64_t L) {
    int64_t sum = 0;
    const std::vector<int64_t> ae = bi_alone_ends(L);
    for (const auto& p : pc) {
        sum += p.second;
        const int64_t end = p.first + p.second;
        const bool alone_end = std::find(ae.begin(), ae.end(), end) != ae.end();
        /* every piece ends on the grid or at one of alone's ends, and none is shorter than 128 rows unless it is alone's
         * whole final piece, from alone's own boundary */
        const bool alone_start = p.first == 0 || std::find(ae.begin(), ae.end(), p.first) != ae.end();
        CHECK(alone_end || end % 256 == 0);
        CHECK(p.second >= 128 || (end == L && alone_start));
        if (!(alone_end || end % 256 == 0) || !(p.second >= 128 || (end == L && alone_start)))
            std::fprintf(stderr, "    bi piece [%lld, %lld)\n", (long long)p.first, (long long)end);
    }
    CHECK_EQ(sum, L);
}
}  // namespace

/* Beside a few decode rows (within the slack) D takes the grid halves of alone's spans: 256 rows a step, alone's final
 * piece whole -- the plan the budget allows without going over it. */
TEST(bi_chunk_d_cuts_alone_spans_on_the_grid_beside_decode_rows) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/576);
    f.cfg.prefill_span = 2;
    f.cfg.prefix_exact = true;   /* the engine's default: a prompt's tail snapshot splits no idle chunk */
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().shared_tok, 320);
    BiChunk on(3);
    for (int i = 0; i < 8; ++i) CHECK(f.submit(16, 4000, 0, true) != 0);
    for (int k = 0; k < 4; ++k) { CHECK(f.sch.step() != nullptr); f.commit_plain(); }
    const int64_t L = 3000;
    const uint64_t a = f.submit((int)L, 2, 0, true);
    const auto pc = bi_pieces(f, a, 60);
    bi_check_d(pc, L);
    /* within the budget: every piece is a 256-row grid piece but alone's 184-row final one -- no step went over */
    for (size_t i = 0; i + 1 < pc.size(); ++i) CHECK_EQ(pc[i].second, 256);
    CHECK_EQ(pc.back().second, 184);
}

/* More decode rows than the slack ahead of it: no grid piece fits what is left of the shared budget. D's long prefill
 * then takes its smallest admissible piece over the shared budget (up to max_tok), so it still progresses, every piece
 * on the grid and >= 128 rows; before this it waited as long as those rows decoded. */
TEST(bi_chunk_d_progresses_with_more_decode_rows_than_the_slack) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/160, /*max_tok=*/576);
    f.cfg.prefill_span = 2;
    f.cfg.prefix_exact = true;
    f.cfg.max_seqs = 128;
    CHECK_OK(f.start());
    BiChunk on(3);
    for (int i = 0; i < 80; ++i) CHECK(f.submit(4, 4000, 0, true) != 0);
    for (int k = 0; k < 4; ++k) { CHECK(f.sch.step() != nullptr); f.commit_plain(); }
    const int64_t L = 3000;
    const uint64_t a = f.submit((int)L, 2, 0, true);
    const auto pc = bi_pieces(f, a, 120);
    bi_check_d(pc, L);
    for (const auto& p : pc) CHECK(p.second <= 576);
}


/* ================================================== lever NOA (ds41-noa): a prefill's piece over the shared budget
 * fn4 sched_noa 1 / 2 under sched_bi_chunk 3 / 4: a prefill D turns away because another prefill already holds the step's
 * shared budget takes D's piece (alone's whole, else its largest grid cut >= 128 rows a side) at the step's max_tok room;
 * 1 one such piece a step, 2 as many as fit. The DS41 line's geometry (exact resume, interval 2048, span 2 from 16384
 * tokens, MNBT 4160, shared budget 2112). A run: `decoders` decoders, a long prompt L, after L's `after_chunks`-th chunk
 * step the newcomers `lens` (a seeded prefix hit first when `hit`); every prefill piece with its checkpoint, its PSK plan
 * and its step; every step's shape and size; fn4 GET "noa" and "dst" at the end. */
namespace {
struct NoaCfg {
    int bic = 3, noa = 2, bypass = 0, total = 0, psk = -1, psk_k = 8, il = 8;
    int decoders = 4, dec_len = 100, dec_tok = 4000;
    int64_t L = 40000;            /* 0: no long prompt */
    std::vector<int> lens;
    int after_chunks = 1;         /* 0: the newcomers arrive with the first step */
    int steps = 220;
    int64_t hit = 0, hit_lin = 0;
    int64_t blocks = 8192;
    int newc_tok = 4;
};
struct NoaRun {
    std::string shape;                                   /* 'C' a step carrying a chunk of L, 'D' one that does not */
    std::map<uint64_t, std::vector<PskPiece>> pieces;
    std::vector<uint64_t> ids;
    uint64_t p = 0;
    int sent = -1;                                       /* the first step the newcomers could be planned in */
    int64_t max_step_tok = 0;
    std::vector<int> pf_entries;                         /* prefill entries a step */
    std::map<std::string, int64_t> noa;
    int64_t bypassed = -1;
    SchedMetrics m{};
};
int64_t noa_json_num(const std::string& g, size_t from, const char* key) {
    if (from == std::string::npos) return -1;
    const std::string k = std::string("\"") + key + "\": ";
    const size_t at = g.find(k, from);
    return at == std::string::npos ? -1 : std::atoll(g.c_str() + at + k.size());
}
NoaRun noa_run(const NoaCfg& c) {
    NoaRun out;
    DstSwitches restore;
    Fixture f;
    f.build(c.blocks, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = c.il;
    f.cfg.prefix_exact = true;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
    f.sch.set_span_min(16384);
    CHECK(DstSwitches::set("sched_bi_chunk", c.bic));
    CHECK(DstSwitches::set("sched_short_bypass", c.bypass));
    CHECK(DstSwitches::set("sched_bypass_total", c.total));
    CHECK(DstSwitches::set("sched_noa", c.noa));
    if (c.psk >= 0) { f.sch.set_psk_mode(c.psk); f.sch.set_psk_k(c.psk_k); }
    for (int d = 0; d < c.decoders; ++d) f.submit(c.dec_len, c.dec_tok);
    if (c.decoders > 0) { CHECK(f.sch.step() != nullptr); f.commit_plain(11); }
    if (c.L > 0) out.p = f.submit((int)c.L, 2);
    auto send = [&](int s) {
        if (c.hit > 0) { f.pc.seed((int32_t)c.hit); f.pc.lin = (int32_t)c.hit_lin; f.pc.src = 3; }
        for (int l : c.lens) out.ids.push_back(f.submit(l, c.newc_tok));
        out.sent = s;
    };
    if (c.after_chunks == 0) send(0);
    int chunks = 0;
    for (int s = 0; s < c.steps; ++s) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        std::vector<int32_t> ahead, rows;
        int64_t dec = -1;
        const bool planned = f.sch.psk_plan(b->step, &ahead, &rows, &dec);
        bool chunk = false;
        int npf = 0;
        size_t pi = 0;
        for (const StepEntry& e : f.sch.plan().e) {
            if (!e.is_prefill) continue;
            ++npf;
            if (e.seq == out.p) chunk = true;
            const int32_t a = planned && pi < ahead.size() ? ahead[pi] : Scheduler::kPskNoPlan;
            out.pieces[e.seq].push_back({ e.ctx_len, e.n_tokens, e.ckpt_slot >= 0 ? e.ctx_len + e.ckpt_tok + 1 : -1, a, s });
            ++pi;
        }
        out.shape += chunk ? 'C' : 'D';
        out.pf_entries.push_back(npf);
        out.max_step_tok = std::max<int64_t>(out.max_step_tok, f.sch.plan().n_tok);
        f.commit_plain(12);
        chunks += chunk ? 1 : 0;
        if (out.sent < 0 && c.after_chunks > 0 && chunks >= c.after_chunks) send(s + 1);
    }
    f.sch.step_finished();   /* the engine's call after a step: it reports the counts to fn4 */
    const std::string g = fn4::state_json();
    const size_t at = g.find("\"noa\": {");
    for (const char* k : { "admitted", "pieces", "rows", "displaced", "waited", "steps", "pf_steps_1", "pf_steps_2",
                           "pf_steps_3", "pf_steps_4plus" })
        out.noa[k] = noa_json_num(g, at, k);
    out.bypassed = noa_json_num(g, g.find("\"dst\": {"), "bypassed");
    out.m = f.sch.metrics();
    return out;
}
/* A request's pieces against alone's from the same start: "" when they keep option D's rule -- contiguous from alone's
 * start to its end, each inside one alone piece, ending at that piece's end or on the G grid with >= 128 rows on both
 * sides -- else why not. `strict`: alone's pieces exactly (start, rows). */
std::string noa_check(const std::vector<PskPiece>& pc, const std::vector<Piece>& alone, int64_t G, bool strict) {
    char why[200];
    if (pc.empty() || alone.empty()) return "no pieces";
    if (strict) {
        if (pc.size() != alone.size()) { std::snprintf(why, sizeof why, "%zu pieces vs alone's %zu", pc.size(), alone.size()); return why; }
        for (size_t i = 0; i < pc.size(); ++i)
            if (pc[i].at != alone[i].at || pc[i].n != alone[i].n) {
                std::snprintf(why, sizeof why, "piece %zu [%lld, +%lld) vs alone's [%lld, +%lld)", i, (long long)pc[i].at,
                              (long long)pc[i].n, (long long)alone[i].at, (long long)alone[i].n);
                return why;
            }
        return "";
    }
    int64_t at = alone[0].at;
    for (const PskPiece& x : pc) {
        if (x.at != at) { std::snprintf(why, sizeof why, "a piece starts at %lld, not %lld", (long long)x.at, (long long)at); return why; }
        const int64_t e = x.at + x.n;
        const Piece* a = nullptr;
        for (const Piece& y : alone) if (y.at <= x.at && x.at < y.at + y.n) a = &y;
        const int64_t ae = a ? a->at + a->n : -1;
        if (!a || e > ae || (e != ae && !(e % G == 0 && x.n >= 128 && ae - e >= 128))) {
            std::snprintf(why, sizeof why, "piece [%lld, %lld) is not one of D's inside alone's [%lld, %lld)", (long long)x.at,
                          (long long)e, a ? (long long)a->at : -1LL, (long long)ae);
            return why;
        }
        at = e;
    }
    if (at != alone.back().at + alone.back().n) return "the pieces do not reach the end";
    return "";
}
int noa_first(const NoaRun& r, uint64_t id) {
    const auto it = r.pieces.find(id);
    return it == r.pieces.end() || it->second.empty() ? -1 : it->second.front().step;
}
int noa_last(const NoaRun& r, uint64_t id) {
    const auto it = r.pieces.find(id);
    return it == r.pieces.end() || it->second.empty() ? -1 : it->second.back().step;
}
bool noa_same_at_n(const std::vector<PskPiece>& a, const std::vector<PskPiece>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i].at != b[i].at || a[i].n != b[i].n) return false;
    return true;
}
}  // namespace

TEST(noa_switch_check_and_report) {
    DstSwitches restore;
    Fixture f;
    dst_fixture(f, 8);
    std::string err;
    CHECK_EQ(fn4::value(fn4::kSchedNoa), 0);
    CHECK(fn4::check("sched_noa", 0, &err));
    CHECK(fn4::check("sched_noa", 1, &err));
    CHECK(fn4::check("sched_noa", 2, &err));
    CHECK(!fn4::check("sched_noa", 3, &err));
    CHECK(!fn4::check("sched_noa", -1, &err));
    CHECK(!fn4::check("no_such_switch", 1, &err));
    CHECK(err.find("sched_noa, stage_short, stage_pick, place_hb, stage_full_min, stage_full_pick, mhc_fuse, ix_lds, ix_ni, "
                   "ds41_arcoded, sched_row_cap, ds41_mix_wv, ds41_gq, ds41_e2, sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, ar_xpush, "
                   "gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, ds41_rfuse, ds41_mix_fin, route_copy_ov, ds41_wmk, ds41_moe1, sched_tail_grid, draft_dcap, draft_dcap_n, exl3_dng, ds41_f32dec, exl3_xr, sched_tail_inner, stage_lone, draft_cost_age, draft_short_blk, ds41_prefetch, ds41_attn_dec, ds41_sphc, ds41_sphc_attn, ds41_sphc_seg, pp_bg_room, pp_bg_join_ms, ds41_twin, ds41_tsk, draft_head_cut, pp_fol_reserve, pp_lead_seg, pp_seg_set, exl3_x3d_pf, pp_tband, pp_merge, pp_merge_ms, pp_rband, sched_srf, aql_hdp, epoch_bump)") !=
          std::string::npos);   /* ds41-int8: SA / HB after NOA; ds41-int9: SP after HB; int12: X3D-F exl3_x3d; int13: XPUSH ar_xpush; int14c: A4 gg_gemv_m16, then SWL / L2 last */
    CHECK(DstSwitches::set("sched_noa", 2));
    CHECK_EQ(fn4::value(fn4::kSchedNoa), 2);
    CHECK_EQ(fn4::sched_noa(), 2);
    CHECK(DstSwitches::set("sched_noa", 1));
    CHECK_EQ(fn4::sched_noa(), 1);
    CHECK(DstSwitches::set("sched_noa", 0));
    CHECK_EQ(fn4::sched_noa(), 0);
    const std::string g = fn4::state_json();
    CHECK(g.find("\"sched_noa\": 0") != std::string::npos);
    CHECK(g.find("\"noa\": {\"admitted\": ") != std::string::npos);
    CHECK(g.find("\"pf_steps_4plus\": ") != std::string::npos);
    /* No shared step without a span: 1 / 2 refused, 0 accepted. */
    Fixture h;
    h.build();
    CHECK_OK(h.start());
    CHECK(!fn4::check("sched_noa", 1, &err));
    CHECK(!fn4::check("sched_noa", 2, &err));
    CHECK(fn4::check("sched_noa", 0, &err));
}

/* Beside four decoders and a 40000-token prompt L (D's 2048-row chunks, interleave 8), newcomers of 1100 / 1330 / 4419
 * tokens arriving during a hold: without NOA (sched_noa 0) D keeps them waiting while L prefills (its chunk fills the
 * shared budget); with NOA each runs its ALONE pieces (4419: 2048 + 2048 + 323, a fresh prompt under the span floor) in
 * L's chunk steps from the first one after it arrived, never a step over max_tok; L's chunks and their steps are the same
 * as without NOA under 3 (the step's first prefill is planned as before), on D's rule under 4 (its grid share may move).
 * sched_noa 1: one NOA piece a step. */
TEST(noa_a_newcomer_beside_a_long_prefill_and_decoders_runs_its_alone_pieces) {
    const auto aL = alone_ck(40000, 0, true);
    const std::vector<int> lens{ 1100, 1330, 4419 };
    std::vector<std::vector<Piece>> al;
    for (int l : lens) al.push_back(alone_ck(l, 0, true));
    CHECK_EQ(al[2].size(), (size_t)3);
    CHECK(al[2].size() == 3 && al[2][0].n == 2048 && al[2][1].n == 2048 && al[2][2].n == 323);
    for (int bic : { 3, 4 }) {
        const int64_t G = bic == 3 ? 2048 : 1024;
        NoaCfg c;
        c.bic = bic;
        c.lens = lens;
        c.noa = 0;
        NoaRun off = noa_run(c);
        c.noa = 2;
        NoaRun on = noa_run(c);
        c.noa = 1;
        NoaRun one = noa_run(c);
        REQUIRE(on.ids.size() == 3 && off.ids.size() == 3 && one.ids.size() == 3);
        const int c1 = (int)on.shape.find('C', (size_t)on.sent);   /* the first chunk step the newcomers can ride */
        CHECK_EQ(noa_first(on, on.ids[0]), c1);
        for (size_t k = 0; k < 3; ++k) {
            /* under 3 alone's pieces exactly; under 4 D's rule (L's grid share leaves room that a 1024 cut may take) */
            CHECK_EQ(noa_check(on.pieces[on.ids[k]], al[k], G, bic == 3), std::string());
            CHECK_EQ(noa_check(one.pieces[one.ids[k]], al[k], G, bic == 3), std::string());
            /* D3 alone: they wait for the whole of L (D4 leaves a 1024-row grid step to a waiting request) */
            if (bic == 3) CHECK(noa_first(off, off.ids[k]) > noa_last(off, off.p));
            CHECK(noa_last(on, on.ids[k]) < noa_last(on, on.p));        /* NOA: done while L still prefills */
            for (const PskPiece& x : on.pieces[on.ids[k]]) CHECK(on.shape[(size_t)x.step] == 'C');   /* beside L's chunk */
        }
        CHECK_EQ(noa_check(on.pieces[on.p], aL, G, false), std::string());
        CHECK_EQ(noa_check(one.pieces[one.p], aL, G, false), std::string());
        if (bic == 3) {
            CHECK(noa_same_at_n(on.pieces[on.p], off.pieces[off.p]));
            CHECK(noa_same_at_n(one.pieces[one.p], off.pieces[off.p]));
            CHECK_EQ(on.shape.substr(0, (size_t)noa_last(on, on.p)), off.shape.substr(0, (size_t)noa_last(on, on.p)));
        }
        CHECK(on.max_step_tok <= 4160 && one.max_step_tok <= 4160 && off.max_step_tok <= 4160);
        CHECK(on.noa["admitted"] >= 1);
        CHECK_EQ(on.noa["waited"], (int64_t)0);
        CHECK_EQ(on.noa["displaced"], (int64_t)0);
        if (bic == 3) {
            CHECK_EQ(on.noa["admitted"], (int64_t)3);
            CHECK_EQ(on.noa["pieces"], (int64_t)5);                 /* 4419's second and third pieces as running prefills */
            CHECK_EQ(on.noa["rows"], (int64_t)(1100 + 1330 + 4419));
            CHECK_EQ(on.noa["steps"], (int64_t)5);
        }
        CHECK_EQ(off.noa["admitted"] + off.noa["pieces"] + off.noa["rows"] + off.noa["displaced"] + off.noa["waited"] +
                 off.noa["steps"], (int64_t)0);
        if (bic == 3) {   /* L's chunk + at most one NOA piece (under 4 a newcomer D admits in L's grid share may ride too) */
            for (int n : one.pf_entries) CHECK(n <= 2);
            CHECK_EQ(one.noa["admitted"], (int64_t)3);
        }
        CHECK(one.noa["steps"] > 0 && one.noa["pieces"] <= one.noa["steps"]);   /* sched_noa 1: one NOA piece a step */
    }
}

/* No decode rows (the r00zzz cells whose decoders had finished: no held step, the bypass never asked): L alone spans
 * (4096-row pieces); the newcomers arrive after its second piece. Without NOA they wait for the whole of L; with NOA each
 * runs its alone pieces from the very next step, L keeps D's rule (2048-row cuts while anything else runs, spans again
 * alone). */
TEST(noa_a_newcomer_beside_a_long_prefill_without_decode_rows) {
    const auto aL = alone_ck(40000, 0, true);
    const std::vector<int> lens{ 1100, 1330, 4419 };
    NoaCfg c;
    c.decoders = 0;
    c.lens = lens;
    c.after_chunks = 2;
    c.noa = 0;
    NoaRun off = noa_run(c);
    c.noa = 2;
    NoaRun on = noa_run(c);
    REQUIRE(on.ids.size() == 3);
    CHECK_EQ(noa_first(on, on.ids[0]), on.sent);
    for (size_t k = 0; k < 3; ++k) {
        CHECK_EQ(noa_check(on.pieces[on.ids[k]], alone_ck(lens[k], 0, true), 2048, true), std::string());
        CHECK(noa_first(off, off.ids[k]) > noa_last(off, off.p));
        CHECK(noa_last(on, on.ids[k]) < noa_last(on, on.p));
    }
    CHECK_EQ(noa_check(on.pieces[on.p], aL, 2048, false), std::string());
    CHECK_EQ(noa_check(off.pieces[off.p], aL, 2048, false), std::string());
    CHECK(on.max_step_tok <= 4160);
    CHECK_EQ(on.noa["admitted"], (int64_t)3);
    CHECK_EQ(on.noa["waited"], (int64_t)0);
    CHECK_EQ(on.noa["displaced"], (int64_t)0);
}

/* The over-max_tok case: a long conversation's tail (prompt 16500, a hit at 14336 on the resume grid) is ONE alone span
 * piece of 2164 rows that D cannot cut (16384 leaves 116 < 128 rows); beside L's 2048-row chunk and four decode rows it
 * is over max_tok (4216 > 4160). NOA counts it waited, and the next step it may act on DISPLACES L's chunk once: the tail
 * runs as that step's only prefill (D's own rule up to max_tok), L's next chunk one step later and unchanged; L is
 * displaced once in all. */
TEST(noa_over_max_tok_displaces_the_long_chunk_once) {
    const auto aL = alone_ck(40000, 0, true);
    const auto tail = alone_ck(16500, 14336, true);
    CHECK_EQ(tail.size(), (size_t)1);
    CHECK_EQ(tail[0].n, (int64_t)2164);
    NoaCfg c;
    c.lens = { 16500 };
    c.hit = 14336;
    c.hit_lin = 14336;
    c.noa = 0;
    NoaRun off = noa_run(c);
    c.noa = 2;
    NoaRun on = noa_run(c);
    REQUIRE(on.ids.size() == 1);
    const auto& q = on.pieces[on.ids[0]];
    CHECK_EQ(noa_check(q, tail, 2048, true), std::string());
    REQUIRE(!q.empty());
    CHECK_EQ(q[0].ck, tail[0].ck);
    CHECK(on.shape[(size_t)q[0].step] == 'D');                  /* L's chunk displaced from that step */
    CHECK_EQ(on.pf_entries[(size_t)q[0].step], 1);
    CHECK(q[0].step < noa_last(on, on.p));
    CHECK(noa_first(off, off.ids[0]) > noa_last(off, off.p));
    CHECK(noa_same_at_n(on.pieces[on.p], off.pieces[off.p]));   /* L's chunks unchanged, one step later */
    CHECK_EQ(noa_check(on.pieces[on.p], aL, 2048, false), std::string());
    CHECK_EQ(on.noa["displaced"], (int64_t)1);
    CHECK(on.noa["waited"] >= 1);
    CHECK_EQ(on.noa["admitted"], (int64_t)0);                    /* admitted as the step's only prefill: D's own */
    CHECK(on.max_step_tok <= 4160);
    /* The bound: two such tails. Each runs in a displacement step of its own; L is displaced twice in all and runs a chunk
     * between the two (a displaced prefill is not displaced again before its next chunk). */
    c.lens = { 16500, 16500 };
    NoaRun two = noa_run(c);
    REQUIRE(two.ids.size() == 2);
    const auto& t0 = two.pieces[two.ids[0]];
    const auto& t1 = two.pieces[two.ids[1]];
    REQUIRE(t0.size() == 1 && t1.size() == 1);
    CHECK_EQ(noa_check(t0, tail, 2048, true), std::string());
    CHECK_EQ(noa_check(t1, tail, 2048, true), std::string());
    CHECK(t0[0].step < t1[0].step);
    CHECK_EQ(two.noa["displaced"], (int64_t)2);
    int between = 0;
    for (const PskPiece& x : two.pieces[two.p]) between += x.step > t0[0].step && x.step < t1[0].step ? 1 : 0;
    CHECK(between >= 1);
    CHECK(noa_same_at_n(two.pieces[two.p], off.pieces[off.p]));
    CHECK(two.max_step_tok <= 4160);
}

/* Several newcomers in one step (the coordinator's c8 turnover wave, no long prompt): six decoders, three 1100-token
 * newcomers at once. D (sched_noa 0) admits one a step (the shared budget holds one), NOA 1 two, NOA 2 all three in one
 * step, each its alone piece. CONTROL: under sched_bi_chunk 0 (today's cuts; NOA does not act) the second newcomer gets a
 * PARTIAL piece beside the first -- the checker must flag it (not alone's pieces). */
TEST(noa_several_newcomers_share_one_step) {
    const auto a = alone_ck(1100, 0, true);
    NoaCfg c;
    c.L = 0;
    c.decoders = 6;
    c.lens = { 1100, 1100, 1100 };
    c.after_chunks = 0;
    c.steps = 40;
    std::map<int, NoaRun> r;
    for (int n : { 0, 1, 2 }) { c.noa = n; r[n] = noa_run(c); }
    for (int n : { 0, 1, 2 }) {
        REQUIRE(r[n].ids.size() == 3);
        for (uint64_t id : r[n].ids) CHECK_EQ(noa_check(r[n].pieces[id], a, 2048, true), std::string());
        CHECK(r[n].max_step_tok <= 4160);
    }
    const int s0 = noa_first(r[2], r[2].ids[0]);
    CHECK(noa_first(r[2], r[2].ids[1]) == s0 && noa_first(r[2], r[2].ids[2]) == s0);
    CHECK_EQ(r[2].pf_entries[(size_t)s0], 3);
    CHECK_EQ(r[2].noa["pf_steps_3"], (int64_t)1);
    CHECK_EQ(r[2].noa["admitted"], (int64_t)2);
    CHECK_EQ(r[2].noa["rows"], (int64_t)2200);
    CHECK_EQ(r[2].noa["steps"], (int64_t)1);
    CHECK(noa_first(r[1], r[1].ids[1]) == noa_first(r[1], r[1].ids[0]) && noa_first(r[1], r[1].ids[2]) > noa_first(r[1], r[1].ids[0]));
    for (int k = 1; k < 3; ++k) CHECK(noa_first(r[0], r[0].ids[(size_t)k]) > noa_first(r[0], r[0].ids[(size_t)k - 1]));
    for (int n : r[0].pf_entries) CHECK(n <= 1);                 /* D: one newcomer a step */
    /* the control */
    c.bic = 0;
    c.noa = 2;
    NoaRun t = noa_run(c);
    REQUIRE(t.ids.size() == 3);
    int bad = 0;
    for (uint64_t id : t.ids) bad += noa_check(t.pieces[id], a, 2048, true).empty() ? 0 : 1;
    CHECK(bad >= 1);
    CHECK_EQ(t.noa["pieces"], (int64_t)0);
}

/* PSK design 2 (ds41_psk 2, K 8): every request NOA admits is latched (four decoders, L and the two newcomers: 7) and
 * every NOA piece is planned with its target (1100: its end; 4419: its end through all three pieces); L's pieces too. */
TEST(noa_psk_design_2_latches_and_plans_every_noa_piece) {
    NoaCfg c;
    c.psk = 2;
    c.psk_k = 8;
    c.lens = { 1100, 4419 };
    NoaRun on = noa_run(c);
    REQUIRE(on.ids.size() == 2);
    CHECK_EQ(on.m.psk_latched, (int64_t)7);
    CHECK_EQ(on.noa["admitted"], (int64_t)2);
    psk_check_plan(on.pieces[on.ids[0]], psk_targets_of(1100, 8), "noa 1100");
    psk_check_plan(on.pieces[on.ids[1]], psk_targets_of(4419, 8), "noa 4419");
    psk_check_plan(on.pieces[on.p], psk_targets_of(40000, 8), "L beside noa");
    CHECK_EQ(noa_check(on.pieces[on.ids[1]], alone_ck(4419, 0, true), 2048, true), std::string());
    /* and the same pieces as under ds41_psk 0 */
    c.psk = 0;
    NoaRun z = noa_run(c);
    for (size_t k = 0; k < 2; ++k) CHECK(noa_same_at_n(on.pieces[on.ids[k]], z.pieces[z.ids[k]]));
    CHECK(noa_same_at_n(on.pieces[on.p], z.pieces[z.p]));
}

/* A lever-NOA piece of a lever PSK-CR cone replay is a replay chunk (17.11.2026): beside decoders and a long cold prompt that
 * holds the step's shared budget, a follow-up whose hit has no snapshot at its grid point replays to it, and the pieces the
 * running walk turns away come from NOA's pass. Each piece up to the replay's end must be flagged (StepEntry::cr_replay),
 * its full-group slots -1 and its PSK plan the replay's (flag 1, the replay's end its target), as the walk's own pieces are;
 * unflagged, the piece wrote its recomputed global KV over the cached rows and ran uncut. */
TEST(noa_a_piece_of_a_cone_replay_is_a_replay_chunk) {
    DstSwitches restore;
    Fixture f;
    cr_fixture(f);
    f.cfg.prefill_interleave = 8;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1024);
    f.sch.set_span_min(16384);
    f.sch.set_psk_mode(2);
    f.sch.set_psk_k(4);
    CHECK_EQ(f.sch.set_psk_cr(1, 5152), 1);
    CHECK(DstSwitches::set("sched_bi_chunk", 3));
    CHECK(DstSwitches::set("sched_noa", 2));
    for (int d = 0; d < 4; ++d) f.submit(100, 4000);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    const uint64_t L = f.submit(40000, 2);
    for (int k = 0; k < 2; ++k)
        if (f.sch.step()) f.commit_plain(11);
    f.pc.seed(26000);
    f.pc.lin = 14336;
    f.pc.src = 5;
    const uint64_t a = f.submit(32000, 1);
    int pieces = 0, beside = 0;
    for (int s = 0; s < 400 && f.sch.find(a) && f.sch.find(a)->n_computed < 22528; ++s) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& pl = f.sch.plan();
        int ia = -1, npf = 0, kpf = -1;
        int64_t row = 0;
        for (size_t i = 0; i < pl.e.size(); ++i) {
            if (pl.e[i].seq == a) { ia = (int)i; kpf = npf; }
            if (pl.e[i].is_prefill) ++npf;
            if (ia < 0) row += pl.e[i].n_tokens;
        }
        if (ia >= 0 && pl.e[(size_t)ia].is_prefill && pl.e[(size_t)ia].ctx_len < 22528) {
            const StepEntry& e = pl.e[(size_t)ia];
            ++pieces;
            if (npf > 1) ++beside;
            CHECK(e.cr_replay);
            CHECK(e.ctx_len + e.n_tokens <= 22528);
            int neg = 0;
            for (int64_t t = 0; t < e.n_tokens; ++t) neg += b->kv[0].slot_mapping[row + t] < 0;
            CHECK_EQ(neg, (int)e.n_tokens);
            std::vector<int32_t> ahead, rows, flags;
            int64_t dec = 0;
            REQUIRE(f.sch.psk_plan(b->step, &ahead, &rows, &dec, &flags));
            REQUIRE(kpf >= 0 && (size_t)kpf < flags.size());
            CHECK_EQ(flags[(size_t)kpf], 1);
            CHECK_EQ((int64_t)ahead[(size_t)kpf], (int64_t)(22528 - (e.ctx_len + e.n_tokens)));
        }
        f.commit_plain(11);
    }
    CHECK(pieces >= 2);
    CHECK(beside >= 1);   /* at least one replay piece shared its step with L's chunk: NOA's */
    (void)L;
}

/* With the bypass (sched_short_bypass 1280, sched_bypass_total 4096; held steps): the 1100-token newcomer is bypassed on
 * the first held step, the 1330 (over N, one alone piece) and the 4419 (over T) are turned away and NOA runs them in L's
 * next chunk steps, all on alone's pieces; L's chunks as with the bypass alone. */
TEST(noa_and_the_bypass_compose) {
    NoaCfg c;
    c.lens = { 1100, 1330, 4419 };
    c.bypass = 1280;
    c.total = 4096;
    c.noa = 0;
    NoaRun byp = noa_run(c);
    c.noa = 2;
    NoaRun both = noa_run(c);
    REQUIRE(both.ids.size() == 3);
    for (size_t k = 0; k < 3; ++k)
        CHECK_EQ(noa_check(both.pieces[both.ids[k]], alone_ck(c.lens[k], 0, true), 2048, true), std::string());
    CHECK_EQ(noa_first(both, both.ids[0]), both.sent);           /* the first held step */
    CHECK_EQ(both.bypassed, (int64_t)1);
    CHECK_EQ(both.noa["admitted"], (int64_t)2);
    CHECK(noa_same_at_n(both.pieces[both.p], byp.pieces[byp.p]));
    CHECK(noa_last(both, both.ids[2]) < noa_last(byp, byp.ids[2]));
}

/* sched_noa 0 plans as unset under every sched_bi_chunk; 1 / 2 change nothing under 0 / 1 / 2 (NOA acts under D only). */
TEST(noa_off_and_other_chunkings_plan_as_unset) {
    for (int bic : { 0, 1, 2, 3, 4 }) {
        NoaCfg c;
        c.bic = bic;
        c.lens = { 1100, 1330, 4419 };
        c.steps = 120;
        c.noa = 0;
        NoaRun z = noa_run(c);
        for (int n : { 1, 2 }) {
            if (bic == 3 || bic == 4) continue;
            c.noa = n;
            NoaRun r = noa_run(c);
            CHECK(r.shape == z.shape);
            for (const auto& kv : z.pieces) CHECK(noa_same_at_n(r.pieces[kv.first], kv.second));
            CHECK_EQ(r.noa["pieces"] + r.noa["waited"] + r.noa["displaced"], (int64_t)0);
        }
        CHECK_EQ(z.noa["pieces"] + z.noa["waited"] + z.noa["displaced"], (int64_t)0);
    }
}

/* Preemption: a pool too small for L and a NOA newcomer both. The newcomer (the latest arrival, the back of the running
 * set) is preempted by L's growth, NOA never preempts (its pieces only take room the pool grants), and the newcomer's
 * recompute runs on alone's pieces of what it then holds (prompt ++ output) from its restart. */
TEST(noa_a_preempted_newcomer_restarts_on_alone_pieces) {
    NoaCfg c;
    c.lens = { 4419 };
    c.newc_tok = 600;
    c.decoders = 1;
    c.blocks = 2800;
    c.steps = 400;
    c.noa = 2;
    NoaRun on = noa_run(c);
    REQUIRE(on.ids.size() == 1);
    const auto& q = on.pieces[on.ids[0]];
    CHECK(on.m.preemptions >= 1);
    CHECK(on.noa["admitted"] >= 1);
    /* split the newcomer's pieces at every restart (a piece at 0 after the first) */
    std::vector<std::vector<PskPiece>> runs;
    for (const PskPiece& x : q) {
        if (x.at == 0 || runs.empty()) runs.emplace_back();
        runs.back().push_back(x);
    }
    CHECK(runs.size() >= 2);
    for (const auto& rn : runs) {
        int64_t tot = 0;
        for (const PskPiece& x : rn) tot += x.n;
        if (rn.back().at + rn.back().n < 4419) continue;   /* a run the preemption cut short */
        CHECK_EQ(noa_check(rn, alone_ck(tot, 0, true), 2048, false), std::string());
    }
    CHECK_EQ(noa_check(on.pieces[on.p], alone_ck(40000, 0, true), 2048, false), std::string());
}

/* MIXED steps carrying draft rows (NST 5, eight decoders each verifying 1 + 5 rows: 48 decode rows ahead of L): NOA's room
 * is max_tok less every planned row, draft rows included, and it plans after the whole running set -- so every step
 * stays <= max_tok, every decoder keeps its row and its five drafts in every step, the newcomers run their alone pieces
 * beside L's chunk, L keeps D's rule. */
TEST(noa_mixed_steps_carrying_draft_rows) {
    for (int noa : { 0, 2 }) {
        DstSwitches restore;
        Fixture f;
        f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
        f.cfg.checkpoint_interval = 2048;
        f.cfg.prefill_span = 2;
        f.cfg.prefill_interleave = 8;
        f.cfg.prefix_exact = true;
        f.cfg.n_spec = 5;
        f.cfg.max_seqs = 16;
        CHECK_OK(f.start());
        f.sch.set_stage_rows(1024);
        f.sch.set_span_min(16384);
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        CHECK(DstSwitches::set("sched_noa", noa));
        std::vector<uint64_t> dec;
        for (int d = 0; d < 8; ++d) dec.push_back(f.submit(100, 4000));
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        const uint64_t p = f.submit(40000, 2);
        const std::vector<int> lens{ 1100, 1330, 4419 };
        std::vector<uint64_t> ids;
        std::map<uint64_t, std::vector<PskPiece>> pc;
        int64_t maxtok = 0, short_rows = 0, drafted = 0;
        int chunks = 0;
        int32_t draft[5] = { 21, 22, 23, 24, 25 };
        for (int s = 0; s < 260; ++s) {
            for (uint64_t id : dec) (void)f.sch.set_draft(id, draft, 5);
            if (!f.sch.step()) break;
            const StepPlan& pl = f.sch.plan();
            maxtok = std::max<int64_t>(maxtok, pl.n_tok);
            int nd = 0;
            for (const StepEntry& e : pl.e) {
                if (e.is_prefill) { pc[e.seq].push_back({ e.ctx_len, e.n_tokens, -1, 0, s }); continue; }
                for (uint64_t id : dec) if (e.seq == id) { ++nd; drafted += e.n_spec == 5 ? 1 : 0; }
            }
            short_rows += nd < 8 ? 1 : 0;
            bool chunk = false;
            for (const StepEntry& e : pl.e) chunk = chunk || (e.seq == p && e.is_prefill);
            f.commit_plain(12);
            chunks += chunk ? 1 : 0;
            if (chunks == 1 && ids.empty()) for (int l : lens) ids.push_back(f.submit(l, 4));
        }
        REQUIRE(ids.size() == 3);
        CHECK(maxtok <= 4160);
        CHECK_EQ(short_rows, (int64_t)0);                         /* every decoder planned in every step */
        CHECK(drafted > 0);
        CHECK_EQ(noa_check(pc[p], alone_ck(40000, 0, true), 2048, false), std::string());
        if (noa == 2)
            for (size_t k = 0; k < 3; ++k) {
                CHECK_EQ(noa_check(pc[ids[k]], alone_ck(lens[k], 0, true), 2048, true), std::string());
                REQUIRE(!pc[ids[k]].empty());
                CHECK(pc[ids[k]].back().step < pc[p].back().step);
            }
    }
}


/* ================================================== lever CAP48 (ds41-cap48): a cap on a step's decode rows
 * fn4 sched_row_cap N (RADIANCE_DECODE_ROW_CAP at start): a step's decode rows -- every decoding sequence's own row + its
 * drafts -- are held to N by trimming drafts, the deepest first (equal depths: the later sequence in plan order first),
 * never a sequence's own row; prefill rows are not counted and every prefill piece is planned as without the cap; a step
 * whose decoding sequences alone exceed N is left as planned (noted once); the step issued ahead is capped too. GET "cap":
 * the value in force and the scheduler's counts. */
namespace {
struct CapSwitch {   /* sched_row_cap back to 0 when a case ends */
    ~CapSwitch() { (void)DstSwitches::set("sched_row_cap", 0); }
};
/* The reference rule, one draft at a time: take one from the deepest proposal, the later entry among equals, until the rows
 * fit; nothing when the sequences alone do not fit. */
std::vector<int32_t> cap_ref(std::vector<int32_t> d, int64_t cap) {
    int64_t rows = (int64_t)d.size();
    for (int32_t x : d) rows += x;
    if ((int64_t)d.size() > cap) return d;
    while (rows > cap) {
        int best = -1;
        for (int i = 0; i < (int)d.size(); ++i)
            if (d[(size_t)i] > 0 && (best < 0 || d[(size_t)i] >= d[(size_t)best])) best = i;
        if (best < 0) break;
        --d[(size_t)best];
        --rows;
    }
    return d;
}
struct CapStep {
    int64_t dec_rows = 0, n_tok = 0;
    int     n_spec = 0;
    std::vector<int32_t> spec;                      /* every decode entry's n_spec, plan order */
    std::vector<uint64_t> seqs;                     /* its sequence */
    std::vector<std::pair<int64_t, int64_t>> pf;    /* every prefill entry's (ctx_len, n_tokens) */
};
struct CapCfg {
    int cap = 48, seqs = 16, depth = 5, max_seqs = 32, steps = 6;
    std::function<int(int)> depth_of;               /* the proposal's depth by sequence index (default: depth) */
    int64_t long_prompt = 0;                        /* > 0: a long prompt beside the decoders (the DS41 geometry, D3) */
    bool accept_all = true;                         /* commit: every verified draft accepted (else none) */
};
struct CapRun {
    std::vector<CapStep> st;
    std::map<std::string, int64_t> cnt, pre;   /* GET "cap" at the end, and before the drafted steps (the prompts' steps) */
    int64_t d(const char* k) const { return cnt.at(k) - pre.at(k); }
    std::vector<uint64_t> ids;
    std::string log;
};
std::map<std::string, int64_t> cap_counts() {
    const std::string g = fn4::state_json();
    std::map<std::string, int64_t> m;
    const size_t at = g.find("\"cap\": {");
    for (const char* k : { "in_force", "steps_capped", "drafts_trimmed", "ahead_capped", "steps_over", "steps_gt48", "max_rows",
                           "decode_steps" })
        m[k] = noa_json_num(g, at, k);
    return m;
}
CapRun cap_run(const CapCfg& c) {
    CapRun out;
    DstSwitches restore;
    CapSwitch restore_cap;
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/96, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.n_spec = c.depth;
    f.cfg.max_seqs = c.max_seqs;
    if (c.long_prompt > 0) {
        f.cfg.checkpoint_interval = 2048;
        f.cfg.prefill_span = 2;
        f.cfg.prefill_interleave = 8;
        f.cfg.prefix_exact = true;
    }
    CHECK(DstSwitches::set("sched_row_cap", c.cap));
    out.log = info_of([&] {
        CHECK_OK(f.start());
        if (c.long_prompt > 0) {
            f.sch.set_stage_rows(1024);
            f.sch.set_span_min(16384);
            CHECK(DstSwitches::set("sched_bi_chunk", 3));
        }
        for (int i = 0; i < c.seqs; ++i) out.ids.push_back(f.submit(100, 4000));
        /* every prompt prefilled (no drafts yet) */
        for (int k = 0; k < 8; ++k) {
            bool pf = false;
            for (uint64_t id : out.ids) { const Request* r = f.sch.find(id); pf = pf || (r && r->n_computed < 100); }
            if (!pf) break;
            REQUIRE(f.sch.step() != nullptr);
            f.commit_plain(11);
        }
        f.sch.step_finished();
        out.pre = cap_counts();
        const uint64_t p = c.long_prompt > 0 ? f.submit((int)c.long_prompt, 2) : 0;
        int32_t draft[RAD_SCHED_MAX_SPEC];
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) draft[j] = 21 + j;
        for (int s = 0; s < c.steps; ++s) {
            for (size_t i = 0; i < out.ids.size(); ++i)
                (void)f.sch.set_draft(out.ids[i], draft, c.depth_of ? c.depth_of((int)i) : c.depth);
            const RadBatch* b = f.sch.step();
            if (!b) break;
            const StepPlan& pl = f.sch.plan();
            CapStep cs;
            cs.n_tok = pl.n_tok;
            cs.n_spec = pl.n_spec;
            CHECK_EQ((int64_t)b->n_tok, pl.n_tok);
            CHECK_EQ((int)b->n_spec, pl.n_spec);
            for (const StepEntry& e : pl.e) {
                if (e.is_prefill) { cs.pf.push_back({ e.ctx_len, e.n_tokens }); continue; }
                cs.dec_rows += e.n_tokens;
                CHECK_EQ(e.n_tokens, 1 + e.n_spec);
                const Request* r = f.sch.find(e.seq);
                REQUIRE(r != nullptr);
                CHECK_EQ(r->n_draft, e.n_spec);             /* the sampler's window = the entry's */
                CHECK(e.n_spec == 0 ? r->draft == nullptr : r->draft != nullptr);
                cs.spec.push_back(e.n_spec);
                cs.seqs.push_back(e.seq);
            }
            CHECK_EQ(pl.n_tok_decode, cs.dec_rows);
            out.st.push_back(cs);
            CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return c.accept_all ? e.n_spec : 0; }));
            if (p && !f.sch.find(p)) break;
        }
        f.sch.step_finished();   /* the engine's call after a step: it reports the counts to fn4 */
    });
    out.cnt = cap_counts();
    return out;
}
}  // namespace

/* 16 sequences proposing 5 drafts each (96 rows) under sched_row_cap 48: every step carries exactly 48 decode rows, each
 * sequence verifies the first 2 of its 5 (16 x 3), the batch / the request's window / the commit agree, and the counts say
 * so; the switch's range and GET. */
TEST(cap48_holds_16_sequences_at_48_decode_rows) {
    std::string err;
    CHECK(fn4::check("sched_row_cap", 0, &err));
    CHECK(fn4::check("sched_row_cap", 48, &err));
    CHECK(fn4::check("sched_row_cap", 4096, &err));
    CHECK(!fn4::check("sched_row_cap", 4097, &err));
    CHECK(!fn4::check("sched_row_cap", -1, &err));
    CHECK_EQ(fn4::sched_row_cap(), 0);   /* RADIANCE_DECODE_ROW_CAP is unset here */
    CapCfg c;
    const CapRun r = cap_run(c);
    REQUIRE_EQ((int)r.st.size(), c.steps);
    for (const CapStep& s : r.st) {
        CHECK_EQ(s.dec_rows, (int64_t)48);
        CHECK_EQ(s.n_spec, 2);
        CHECK_EQ((int)s.spec.size(), 16);
        for (int32_t k : s.spec) CHECK_EQ(k, 2);
        CHECK(s.seqs == r.ids);   /* plan order = arrival order (one priority) */
    }
    CHECK_EQ(r.cnt.at("in_force"), (int64_t)48);
    CHECK_EQ(r.d("steps_capped"), (int64_t)c.steps);
    CHECK_EQ(r.d("drafts_trimmed"), (int64_t)(c.steps * 16 * 3));
    CHECK_EQ(r.cnt.at("ahead_capped"), (int64_t)0);
    CHECK_EQ(r.cnt.at("steps_over"), (int64_t)0);
    CHECK_EQ(r.cnt.at("steps_gt48"), (int64_t)0);
    CHECK_EQ(r.cnt.at("max_rows"), (int64_t)48);
    CHECK_EQ(r.d("decode_steps"), (int64_t)c.steps);
    /* the start line once (the switch was set before the scheduler's init, as RADIANCE_DECODE_ROW_CAP sets it); 32 >= 16
     * sequences and every step capped: nothing else to note */
    {
        int n = 0;
        for (size_t at = r.log.find("lever CAP48: decode row cap 48 ("); at != std::string::npos;
             at = r.log.find("lever CAP48: decode row cap 48 (", at + 1)) ++n;
        CHECK_EQ(n, 1);
        CHECK(r.log.find("covers up to") == std::string::npos);
        CHECK(r.log.find("even at depth 0: left as planned") == std::string::npos);
    }
    /* the verified drafts are the proposal's first two, committed in order before the bonus token */
    {
        DstSwitches restore;
        CapSwitch restore_cap;
        Fixture f;
        f.build(/*blocks=*/8192, /*states=*/96, /*max_tok=*/4160, /*max_ctx=*/60000);
        f.cfg.n_spec = 5;
        CHECK(DstSwitches::set("sched_row_cap", 48));
        CHECK_OK(f.start());
        std::vector<uint64_t> ids;
        for (int i = 0; i < 16; ++i) ids.push_back(f.submit(100, 4000));
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[5] = { 21, 22, 23, 24, 25 };
        for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, draft, 5));
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        CHECK_EQ((int)b->n_tok, 48);
        CHECK_EQ((int)b->q_lens[0], 3);
        CHECK_EQ(b->token_ids[0], 11);
        CHECK_EQ(b->token_ids[1], 21);
        CHECK_EQ(b->token_ids[2], 22);
        CHECK_EQ(b->token_ids[3], 11);   /* the next sequence's own row */
        CHECK_OK(f.commit(99, 0, [](const StepEntry& e) { return e.n_spec; }));
        const Request* ra = f.sch.find(ids[0]);
        REQUIRE(ra != nullptr);
        CHECK_EQ((int)ra->output.size(), 4);
        CHECK_EQ(ra->output[1], 21);
        CHECK_EQ(ra->output[2], 22);
        CHECK_EQ(ra->output[3], 99);
        CHECK_EQ(ra->n_computed, (int64_t)103);
    }
}

/* 32 sequences at depth 5 (192 rows) under 48: 16 drafts fit -- the first 16 sequences in plan order keep 1, the last 16
 * none; proposals of uneven depth are trimmed as the one-draft-at-a-time reference (the deepest first, the later among
 * equals); two runs of the same inputs plan the same steps (deterministic). */
TEST(cap48_trims_32_sequences_deterministically) {
    CapCfg c;
    c.seqs = 32;
    c.max_seqs = 48;
    const CapRun a = cap_run(c), b = cap_run(c);
    REQUIRE_EQ((int)a.st.size(), c.steps);
    REQUIRE_EQ((int)b.st.size(), c.steps);
    for (size_t s = 0; s < a.st.size(); ++s) {
        CHECK_EQ(a.st[s].dec_rows, (int64_t)48);
        REQUIRE_EQ((int)a.st[s].spec.size(), 32);
        for (int i = 0; i < 32; ++i) CHECK_EQ(a.st[s].spec[(size_t)i], i < 16 ? 1 : 0);
        CHECK(a.st[s].spec == b.st[s].spec);
        CHECK(a.st[s].seqs == b.st[s].seqs);
    }
    CHECK_EQ(a.d("drafts_trimmed"), (int64_t)(c.steps * (32 * 5 - 16)));
    CHECK_EQ(a.cnt.at("steps_gt48"), (int64_t)0);
    /* uneven proposals: sequence i proposes 1 + (i * 7) % 5 drafts (never 0: an empty proposal takes prompt lookup's);
     * 20 sequences, 60 drafts, cap 48 -> 28 kept */
    CapCfg u;
    u.seqs = 20;
    u.depth = 5;
    u.depth_of = [](int i) { return 1 + (i * 7) % 5; };
    u.accept_all = false;   /* every sequence at the same position each step: the same proposals every step */
    const CapRun x = cap_run(u), y = cap_run(u);
    REQUIRE_EQ((int)x.st.size(), u.steps);
    std::vector<int32_t> want0;
    for (int i = 0; i < 20; ++i) want0.push_back(1 + (i * 7) % 5);
    const std::vector<int32_t> want = cap_ref(want0, 48);
    int64_t kept = 0;
    for (int32_t k : want) kept += k;
    CHECK_EQ(kept, (int64_t)28);
    for (size_t s = 0; s < x.st.size(); ++s) {
        CHECK(x.st[s].spec == want);
        CHECK(x.st[s].spec == y.st[s].spec);
        CHECK_EQ(x.st[s].dec_rows, (int64_t)48);
    }
    /* a depth below the room is untouched: 8 sequences at 5 = 48 rows, nothing trimmed */
    CapCfg e8;
    e8.seqs = 8;
    const CapRun z = cap_run(e8);
    for (const CapStep& s : z.st) {
        CHECK_EQ(s.dec_rows, (int64_t)48);
        for (int32_t k : s.spec) CHECK_EQ(k, 5);
    }
    CHECK_EQ(z.d("steps_capped"), (int64_t)0);
    CHECK_EQ(z.cnt.at("max_rows"), (int64_t)48);
}

/* More sequences than the cap (56 at --max-num-seqs 64, depth 1): even depth 0 is 56 rows, so every step is left as
 * planned and counted as steps_over; the two notes (the cap below --max-num-seqs, a step left) are said once each. */
TEST(cap48_above_48_sequences_is_left_as_planned_and_noted_once) {
    CapCfg c;
    c.seqs = 56;
    c.max_seqs = 64;
    c.depth = 1;
    const CapRun r = cap_run(c);
    REQUIRE_EQ((int)r.st.size(), c.steps);
    for (const CapStep& s : r.st) {
        CHECK_EQ(s.dec_rows, (int64_t)112);
        for (int32_t k : s.spec) CHECK_EQ(k, 1);
    }
    CHECK_EQ(r.cnt.at("steps_capped"), (int64_t)0);
    CHECK_EQ(r.cnt.at("drafts_trimmed"), (int64_t)0);
    CHECK_EQ(r.d("steps_over"), (int64_t)c.steps);
    CHECK_EQ(r.d("steps_gt48"), (int64_t)c.steps);
    auto count = [&](const char* what) {
        int n = 0;
        for (size_t at = r.log.find(what); at != std::string::npos; at = r.log.find(what, at + 1)) ++n;
        return n;
    };
    CHECK_EQ(count("covers up to 48 decoding sequences; at --max-num-seqs 64"), 1);
    CHECK_EQ(count("more than the decode row cap 48 even at depth 0: left as planned"), 1);
    /* the same deployment at 48 sequences: capped, nothing noted about a step left */
    CapCfg d = c;
    d.seqs = 40;
    d.depth = 5;
    const CapRun q = cap_run(d);
    for (const CapStep& s : q.st) CHECK_EQ(s.dec_rows, (int64_t)48);
    CHECK_EQ(q.d("steps_over"), (int64_t)0);
    CHECK_EQ(q.d("steps_capped"), (int64_t)d.steps);
    CHECK(q.log.find("even at depth 0: left as planned") == std::string::npos);   /* no step left (the start's note is there) */
    CHECK(q.log.find("covers up to 48 decoding sequences; at --max-num-seqs 64") != std::string::npos);
}

/* CONTROL: sched_row_cap 0 plans what the switch never set plans (every proposal whole, 96 rows), and counts the steps over 48
 * rows it does not cap; the cap only ever takes drafts away. */
TEST(cap48_off_plans_as_unset) {
    CapCfg off;
    off.cap = 0;
    const CapRun r0 = cap_run(off);
    std::vector<CapStep> unset;
    {
        DstSwitches restore;
        Fixture f;
        f.build(/*blocks=*/8192, /*states=*/96, /*max_tok=*/4160, /*max_ctx=*/60000);
        f.cfg.n_spec = 5;
        f.cfg.max_seqs = 32;
        CHECK_OK(f.start());
        std::vector<uint64_t> ids;
        for (int i = 0; i < 16; ++i) ids.push_back(f.submit(100, 4000));
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[5] = { 21, 22, 23, 24, 25 };
        for (int s = 0; s < off.steps; ++s) {
            for (uint64_t id : ids) (void)f.sch.set_draft(id, draft, 5);
            REQUIRE(f.sch.step() != nullptr);
            CapStep cs;
            for (const StepEntry& e : f.sch.plan().e) { cs.dec_rows += e.n_tokens; cs.spec.push_back(e.n_spec); cs.seqs.push_back(e.seq); }
            cs.n_tok = f.sch.plan().n_tok;
            unset.push_back(cs);
            CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));
        }
    }
    REQUIRE_EQ(r0.st.size(), unset.size());
    for (size_t s = 0; s < unset.size(); ++s) {
        CHECK_EQ(r0.st[s].dec_rows, (int64_t)96);
        CHECK(r0.st[s].spec == unset[s].spec);
        CHECK_EQ(r0.st[s].n_tok, unset[s].n_tok);
        CHECK_EQ(r0.st[s].dec_rows, unset[s].dec_rows);
    }
    CHECK_EQ(r0.cnt.at("in_force"), (int64_t)0);
    CHECK_EQ(r0.cnt.at("steps_capped"), (int64_t)0);
    CHECK_EQ(r0.cnt.at("drafts_trimmed"), (int64_t)0);
    CHECK_EQ(r0.d("steps_gt48"), (int64_t)off.steps);
    CHECK_EQ(r0.cnt.at("max_rows"), (int64_t)96);
}

/* A mixed step counts its decode rows only: 16 decoders at depth 5 beside a 40000-token prompt (the DS41 line's geometry,
 * sched_bi_chunk 3): under the cap every step carries <= 48 decode rows, and the prompt's pieces -- start, rows, the step
 * they run in -- are the uncapped run's, step for step (no draft accepted, so both runs see the same positions). */
TEST(cap48_mixed_steps_keep_the_prefill_pieces) {
    CapCfg on;
    on.long_prompt = 40000;
    on.steps = 60;
    on.accept_all = false;
    CapCfg off = on;
    off.cap = 0;
    const CapRun a = cap_run(on), b = cap_run(off);
    REQUIRE_EQ(a.st.size(), b.st.size());
    int mixed = 0, over = 0;
    for (size_t s = 0; s < a.st.size(); ++s) {
        CHECK(a.st[s].dec_rows <= 48);
        CHECK(a.st[s].pf == b.st[s].pf);
        CHECK_EQ(a.st[s].n_tok - a.st[s].dec_rows, b.st[s].n_tok - b.st[s].dec_rows);   /* prefill rows */
        if (!a.st[s].pf.empty()) ++mixed;
        if (b.st[s].dec_rows > 48) ++over;
    }
    CHECK(mixed >= 4);
    CHECK(over == (int)b.st.size());
    CHECK_EQ(a.cnt.at("steps_gt48"), (int64_t)0);
    CHECK_EQ(b.d("steps_gt48"), b.d("decode_steps"));
}

/* The step issued ahead (step_next) is capped under the value of the step before it: 16 sequences each expecting 5 drafts
 * verify 2; the proposals published after its commit (5 each) are cut to the 2 it verifies; promote_next keeps them. Without
 * the cap the same flow verifies all 5. */
TEST(cap48_the_step_issued_ahead_is_capped) {
    for (int cap : { 48, 0 }) {
        DstSwitches restore;
        CapSwitch restore_cap;
        Fixture f;
        f.build(/*blocks=*/8192, /*states=*/96, /*max_tok=*/4160, /*max_ctx=*/60000);
        f.cfg.n_spec = 5;
        CHECK(DstSwitches::set("sched_row_cap", cap));
        CHECK_OK(f.start());
        std::vector<uint64_t> ids;
        for (int i = 0; i < 16; ++i) ids.push_back(f.submit(100, 4000));
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[5] = { 21, 22, 23, 24, 25 };
        for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, draft, 5));
        REQUIRE(f.sch.step() != nullptr);
        const int k = cap ? 2 : 5;
        for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, k);
        for (uint64_t id : ids) {
            f.sch.find(id)->speculation_fixed = true;
            f.sch.expect_draft(id, 5);
        }
        const RadBatch* nb = f.sch.step_next();
        REQUIRE(nb != nullptr);
        CHECK_EQ((int)nb->n_tok, 16 * (1 + k));
        CHECK_EQ((int)nb->n_spec, k);
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));
        for (uint64_t id : ids) {
            CHECK_OK(f.sch.set_draft(id, draft, 5));
            CHECK_EQ(f.sch.find(id)->n_draft, k);
        }
        f.sch.promote_next();
        for (const StepEntry& e : f.sch.plan().e) {
            CHECK_EQ(e.n_spec, k);
            CHECK_EQ(e.n_tokens, 1 + k);
        }
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));
        f.sch.step_finished();
        const auto m = cap_counts();
        CHECK_EQ(m.at("steps_capped"), cap ? (int64_t)2 : (int64_t)0);
        CHECK_EQ(m.at("ahead_capped"), cap ? (int64_t)1 : (int64_t)0);
        CHECK_EQ(m.at("steps_gt48"), cap ? (int64_t)0 : (int64_t)2);
    }
}


/* ================================================== lever CBH (ds41-int10b): the co-batch hold
 * fn4 sched_cb_hold_ms H (RADIANCE_SCHED_CB_HOLD_MS at start) / sched_cb_hold_k K: while >= 2 sequences decode, a fresh
 * newcomer's admission is held until K newcomers wait, H ms after the first arrived, or fewer than two decode; the step the
 * hold ends at asks the queue exactly as without the switch (queue order, D3's / NOA's pieces). The DS41 geometry (interval
 * 2048, MNBT 4160, span 2), sched_bi_chunk 3, sched_noa 2, six decoders, a 40 ms test clock a step. GET "cbh". */
namespace {
struct CbhSwitch {   /* sched_cb_hold_ms back to 0 and sched_cb_hold_k to 2 when a case ends */
    ~CbhSwitch() { (void)DstSwitches::set("sched_cb_hold_ms", 0); (void)DstSwitches::set("sched_cb_hold_k", 2); }
};
struct CbhCfg {
    int hold = 500, k = 2, decoders = 6, dec_tok = 4000, il = 0, steps = 24, newc_tok = 4;
    int64_t step_us = 40000;                    /* the hold's test clock advances 40 ms a step */
    std::vector<std::pair<int, int>> arrive;    /* (step, prompt tokens): submitted before that step is planned */
    int media_at = -1;                          /* >= 0: a request whose picture is never encoded arrives at that step */
    bool ahead = false;                         /* the engine's loop: step_next behind every step, promoted when issued */
    bool touch = true;                          /* false: the CBH switches are never set (their start values) */
};
struct CbhRun {
    std::vector<uint64_t> ids;                          /* the newcomers, in arrival order */
    std::map<uint64_t, std::vector<PskPiece>> pieces;   /* each newcomer's prefill entries (at, n, checkpoint, -, step) */
    std::vector<int> pf_entries, dec_entries;           /* a step's prefill / decode entries */
    std::string issued;                                 /* a step: 'S' step(), 'A' the step issued ahead, '-' none */
    std::vector<std::string> sig;                       /* a step's plan: every entry's sequence, context, rows, kind */
    int64_t max_step_tok = 0;
    std::map<std::string, int64_t> cbh;
    std::string log;
};
std::map<std::string, int64_t> cbh_counts() {
    const std::string g = fn4::state_json();
    std::map<std::string, int64_t> m;
    const size_t at = g.find("\"cbh\": {");
    for (const char* k : { "in_force_ms", "k", "holds", "by_k", "by_timeout", "by_idle", "by_other", "hold_ms", "held_steps",
                           "cob_steps", "cob_newcomers", "single_steps" })
        m[k] = noa_json_num(g, at, k);
    return m;
}
CbhRun cbh_run(const CbhCfg& c) {
    CbhRun out;
    DstSwitches restore;
    CbhSwitch restore_cbh;
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/4160, /*max_ctx=*/60000);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.prefill_span = 2;
    f.cfg.prefill_interleave = c.il;
    f.cfg.prefix_exact = true;
    int64_t clk = 0;
    std::shared_ptr<mm::Item> item;
    if (c.touch) {
        CHECK(DstSwitches::set("sched_cb_hold_ms", c.hold));
        CHECK(DstSwitches::set("sched_cb_hold_k", c.k));
    }
    out.log = info_of([&] {
        CHECK_OK(f.start());
        f.sch.set_stage_rows(1024);
        f.sch.set_span_min(16384);
        f.sch.set_cbh_clock([&clk] { return clk; });
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        CHECK(DstSwitches::set("sched_noa", 2));
        for (int d = 0; d < c.decoders; ++d) f.submit(100, c.dec_tok);
        if (c.decoders > 0) { REQUIRE(f.sch.step() != nullptr); f.commit_plain(11); }
        bool promote = false;
        for (int s = 0; s < c.steps; ++s) {
            clk = (int64_t)(s + 1) * c.step_us;
            for (const auto& a : c.arrive)
                if (a.first == s) out.ids.push_back(f.submit(a.second, c.newc_tok));
            if (s == c.media_at) (void)f.submit_unencoded(160, c.newc_tok, &item);
            if (promote) {
                f.sch.promote_next();
                out.issued += 'A';
            } else if (f.sch.step() != nullptr) {
                out.issued += 'S';
            } else {
                out.issued += '-';
                out.pf_entries.push_back(0);
                out.dec_entries.push_back(0);
                out.sig.push_back("");
                continue;
            }
            const StepPlan& pl = f.sch.plan();
            int npf = 0, ndec = 0;
            std::string sg;
            for (const StepEntry& e : pl.e) {
                sg += std::to_string(e.seq) + ":" + std::to_string(e.ctx_len) + "+" + std::to_string(e.n_tokens) +
                      (e.is_prefill ? "p " : "d ");
                if (!e.is_prefill) { ++ndec; continue; }
                ++npf;
                out.pieces[e.seq].push_back({ e.ctx_len, e.n_tokens, e.ckpt_slot >= 0 ? e.ctx_len + e.ckpt_tok + 1 : -1, 0, s });
            }
            out.pf_entries.push_back(npf);
            out.dec_entries.push_back(ndec);
            out.sig.push_back(sg);
            out.max_step_tok = std::max<int64_t>(out.max_step_tok, pl.n_tok);
            promote = false;
            if (c.ahead) {
                for (const StepEntry& e : pl.e)
                    if (Request* r = f.sch.find(e.seq)) r->speculation_fixed = true;
                promote = f.sch.step_next() != nullptr;
            }
            f.commit_plain(12);
        }
        f.sch.step_finished();   /* the engine's call after a step: it reports the counts to fn4 */
    });
    out.cbh = cbh_counts();
    return out;
}
int cbh_first(const CbhRun& r, uint64_t id) {
    const auto it = r.pieces.find(id);
    return it == r.pieces.end() || it->second.empty() ? -1 : it->second.front().step;
}
bool cbh_same_pieces(const std::vector<PskPiece>& a, const std::vector<PskPiece>& b) {   /* (at, n, checkpoint) a piece */
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i].at != b[i].at || a[i].n != b[i].n || a[i].ck != b[i].ck) return false;
    return true;
}
}  // namespace

TEST(cbh_switch_check_and_report) {
    DstSwitches restore;
    CbhSwitch restore_cbh;
    Fixture f;
    dst_fixture(f, 8);
    std::string err;
    CHECK_EQ(fn4::value(fn4::kSchedCbHoldMs), 0);
    CHECK_EQ(fn4::value(fn4::kSchedCbHoldK), 2);
    CHECK_EQ(fn4::sched_cb_hold_ms(), 0);
    CHECK_EQ(fn4::sched_cb_hold_k(), 2);
    CHECK(fn4::check("sched_cb_hold_ms", 0, &err));
    CHECK(fn4::check("sched_cb_hold_ms", 1, &err));
    CHECK(fn4::check("sched_cb_hold_ms", 60000, &err));
    CHECK(!fn4::check("sched_cb_hold_ms", 60001, &err));
    CHECK(!fn4::check("sched_cb_hold_ms", -1, &err));
    CHECK(fn4::check("sched_cb_hold_k", 2, &err));
    CHECK(fn4::check("sched_cb_hold_k", 64, &err));
    CHECK(!fn4::check("sched_cb_hold_k", 1, &err));
    CHECK(!fn4::check("sched_cb_hold_k", 0, &err));
    CHECK(!fn4::check("sched_cb_hold_k", 65, &err));
    CHECK(!fn4::check("no_such_switch", 1, &err));
    CHECK(err.find("sched_row_cap, ds41_mix_wv, ds41_gq, ds41_e2, sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, ar_xpush, gg_gemv_m16, "
                   "place_swap_lent, heat_min_gain_x100, heat_hyst_x100, ds41_rfuse, ds41_mix_fin, route_copy_ov, ds41_wmk, ds41_moe1, sched_tail_grid, draft_dcap, draft_dcap_n, exl3_dng, ds41_f32dec, exl3_xr, sched_tail_inner, stage_lone, draft_cost_age, draft_short_blk, ds41_prefetch, ds41_attn_dec, ds41_sphc, ds41_sphc_attn, ds41_sphc_seg, pp_bg_room, pp_bg_join_ms, ds41_twin, ds41_tsk, draft_head_cut, pp_fol_reserve, pp_lead_seg, pp_seg_set, exl3_x3d_pf, pp_tband, pp_merge, pp_merge_ms, pp_rband, sched_srf, aql_hdp, epoch_bump)") != std::string::npos);
    CHECK(DstSwitches::set("sched_cb_hold_ms", 250));
    CHECK(DstSwitches::set("sched_cb_hold_k", 3));
    CHECK_EQ(fn4::sched_cb_hold_ms(), 250);
    CHECK_EQ(fn4::sched_cb_hold_k(), 3);
    CHECK_EQ(fn4::value(fn4::kSchedCbHoldMs), 250);
    const std::string g = fn4::state_json();
    CHECK(g.find("\"sched_cb_hold_ms\": 250, \"sched_cb_hold_k\": 3") != std::string::npos);
    CHECK(g.find("\"cbh\": {\"in_force_ms\": 250, \"k\": 3, \"holds\": ") != std::string::npos);
    CHECK(g.find("\"single_steps\": ") != std::string::npos);
    CHECK(DstSwitches::set("sched_cb_hold_ms", 0));
    CHECK(DstSwitches::set("sched_cb_hold_k", 2));
    CHECK_EQ(fn4::sched_cb_hold_ms(), 0);
    CHECK_EQ(fn4::sched_cb_hold_k(), 2);
}

/* A newcomer A (1100 tokens: one alone piece) at step 2 and B at step 7 beside six decoders. Hold 0: each is admitted the step it
 * arrives in (two single newcomer steps). Hold 500 ms, K 2: A waits (steps 2..6 carry the six decode rows only) and both start in
 * step 7, B's arrival ending the hold by K -- each still its alone piece. The counts: one hold, by K, five held steps, 200 ms. */
TEST(cbh_holds_until_k_newcomers_wait) {
    const auto a = alone_ck(1100, 0, true);
    REQUIRE(a.size() == 1);
    CbhCfg c;
    c.arrive = { { 2, 1100 }, { 7, 1100 } };
    c.hold = 0;
    CbhRun off = cbh_run(c);
    c.hold = 500;
    CbhRun on = cbh_run(c);
    REQUIRE(off.ids.size() == 2 && on.ids.size() == 2);
    CHECK_EQ(cbh_first(off, off.ids[0]), 2);
    CHECK_EQ(cbh_first(off, off.ids[1]), 7);
    CHECK_EQ(cbh_first(on, on.ids[0]), 7);
    CHECK_EQ(cbh_first(on, on.ids[1]), 7);
    CHECK_EQ(on.pf_entries[7], 2);
    for (int s = 2; s < 7; ++s) {
        CHECK_EQ(on.pf_entries[(size_t)s], 0);
        CHECK_EQ(on.dec_entries[(size_t)s], 6);
    }
    for (int k = 0; k < 2; ++k) {
        CHECK_EQ(noa_check(off.pieces.at(off.ids[(size_t)k]), a, 2048, true), std::string());
        CHECK_EQ(noa_check(on.pieces.at(on.ids[(size_t)k]), a, 2048, true), std::string());
        CHECK(cbh_same_pieces(on.pieces.at(on.ids[(size_t)k]), off.pieces.at(off.ids[(size_t)k])));
    }
    CHECK(on.max_step_tok <= 4160);
    CHECK_EQ(on.cbh["in_force_ms"], (int64_t)500);
    CHECK_EQ(on.cbh["k"], (int64_t)2);
    CHECK_EQ(on.cbh["holds"], (int64_t)1);
    CHECK_EQ(on.cbh["by_k"], (int64_t)1);
    CHECK_EQ(on.cbh["by_timeout"] + on.cbh["by_idle"] + on.cbh["by_other"], (int64_t)0);
    CHECK_EQ(on.cbh["held_steps"], (int64_t)5);
    CHECK_EQ(on.cbh["hold_ms"], (int64_t)200);
    CHECK_EQ(on.cbh["cob_steps"], (int64_t)2);          /* the six decoders' admission and the pair */
    CHECK_EQ(on.cbh["cob_newcomers"], (int64_t)8);
    CHECK_EQ(on.cbh["single_steps"], (int64_t)0);
    CHECK_EQ(off.cbh["holds"] + off.cbh["held_steps"] + off.cbh["hold_ms"], (int64_t)0);
    CHECK_EQ(off.cbh["cob_steps"], (int64_t)1);
    CHECK_EQ(off.cbh["single_steps"], (int64_t)2);
    CHECK(on.log.find("lever CBH: co-batch hold 500 ms, K 2") != std::string::npos);
    CHECK(off.log.find("lever CBH") == std::string::npos);
}

/* A lone newcomer: the hold ends by timeout, H after its arrival on the hold's clock (200 ms = 5 steps; 1000 ms = 25). */
TEST(cbh_releases_on_timeout) {
    for (int h : { 200, 1000 }) {
        CbhCfg c;
        c.arrive = { { 2, 1100 } };
        c.hold = h;
        c.steps = 40;
        CbhRun r = cbh_run(c);
        REQUIRE(r.ids.size() == 1);
        const int n = h / 40;
        CHECK_EQ(cbh_first(r, r.ids[0]), 2 + n);
        for (int s = 2; s < 2 + n; ++s) CHECK_EQ(r.pf_entries[(size_t)s], 0);
        CHECK_EQ(noa_check(r.pieces.at(r.ids[0]), alone_ck(1100, 0, true), 2048, true), std::string());
        CHECK_EQ(r.cbh["holds"], (int64_t)1);
        CHECK_EQ(r.cbh["by_timeout"], (int64_t)1);
        CHECK_EQ(r.cbh["by_k"] + r.cbh["by_idle"] + r.cbh["by_other"], (int64_t)0);
        CHECK_EQ(r.cbh["held_steps"], (int64_t)n);
        CHECK_EQ(r.cbh["hold_ms"], (int64_t)h);
        CHECK_EQ(r.cbh["single_steps"], (int64_t)1);
    }
}

/* Never an idle server: with one decoder (or none) a newcomer is admitted the step it arrives in, nothing held; a hold standing
 * when the decoders finish ends by idle at the first step with fewer than two decoding. */
TEST(cbh_never_holds_an_idle_server) {
    for (int d : { 0, 1 }) {
        CbhCfg c;
        c.decoders = d;
        c.arrive = { { 2, 1100 } };
        c.hold = 10000;
        CbhRun r = cbh_run(c);
        REQUIRE(r.ids.size() == 1);
        CHECK_EQ(cbh_first(r, r.ids[0]), 2);
        CHECK_EQ(r.cbh["holds"] + r.cbh["held_steps"], (int64_t)0);
    }
    CbhCfg c;
    c.decoders = 2;
    c.dec_tok = 8;                  /* both end a few steps in */
    c.arrive = { { 1, 1100 } };
    c.hold = 10000;
    CbhRun r = cbh_run(c);
    REQUIRE(r.ids.size() == 1);
    const int a = cbh_first(r, r.ids[0]);
    REQUIRE(a > 1);
    for (int s = 1; s < a; ++s) {
        CHECK_EQ(r.dec_entries[(size_t)s], 2);
        CHECK_EQ(r.pf_entries[(size_t)s], 0);
    }
    CHECK(r.dec_entries[(size_t)a] < 2);
    CHECK_EQ(r.cbh["holds"], (int64_t)1);
    CHECK_EQ(r.cbh["by_idle"], (int64_t)1);
    CHECK_EQ(r.cbh["by_k"] + r.cbh["by_timeout"] + r.cbh["by_other"], (int64_t)0);
    CHECK_EQ(r.cbh["held_steps"], (int64_t)(a - 1));
}

/* K 3: three newcomers (steps 2, 4, 6) start in one step, each its alone piece (3 x 1100 + six decode rows fit MNBT 4160 under
 * NOA 2). A waiting request that may not be held (a picture never encoded) means no hold: the fresh one behind it is admitted
 * the step it arrives in, as without the switch. */
TEST(cbh_three_newcomers_wait_for_k3_and_a_request_that_may_not_be_held_holds_nothing) {
    const auto a = alone_ck(1100, 0, true);
    CbhCfg c;
    c.arrive = { { 2, 1100 }, { 4, 1100 }, { 6, 1100 } };
    c.hold = 2000;
    c.k = 3;
    CbhRun r = cbh_run(c);
    REQUIRE(r.ids.size() == 3);
    for (uint64_t id : r.ids) {
        CHECK_EQ(cbh_first(r, id), 6);
        CHECK_EQ(noa_check(r.pieces.at(id), a, 2048, true), std::string());
    }
    CHECK_EQ(r.pf_entries[6], 3);
    CHECK(r.max_step_tok <= 4160);
    CHECK_EQ(r.cbh["k"], (int64_t)3);
    CHECK_EQ(r.cbh["by_k"], (int64_t)1);
    CHECK_EQ(r.cbh["cob_newcomers"], (int64_t)(6 + 3));
    CbhCfg m;
    m.media_at = 2;
    m.arrive = { { 3, 1100 } };
    m.hold = 2000;
    CbhRun q = cbh_run(m);
    REQUIRE(q.ids.size() == 1);
    CHECK_EQ(cbh_first(q, q.ids[0]), 3);
    CHECK_EQ(q.cbh["holds"] + q.cbh["held_steps"], (int64_t)0);
}

/* A long newcomer (4419 = alone's 2048 + 2048 + 323) and a short one co-admitted after a hold: both keep alone's pieces exactly
 * (D3 + NOA 2), never a step over MNBT; the same pieces as without the hold, only later. */
TEST(cbh_co_admitted_pieces_stay_alone) {
    const auto a4 = alone_ck(4419, 0, true);
    const auto a1 = alone_ck(1100, 0, true);
    REQUIRE(a4.size() == 3);
    CbhCfg c;
    c.arrive = { { 2, 4419 }, { 5, 1100 } };
    c.hold = 0;
    CbhRun off = cbh_run(c);
    c.hold = 500;
    CbhRun on = cbh_run(c);
    REQUIRE(off.ids.size() == 2 && on.ids.size() == 2);
    CHECK_EQ(cbh_first(on, on.ids[0]), 5);
    CHECK_EQ(cbh_first(on, on.ids[1]), 5);
    CHECK_EQ(noa_check(on.pieces.at(on.ids[0]), a4, 2048, true), std::string());
    CHECK_EQ(noa_check(on.pieces.at(on.ids[1]), a1, 2048, true), std::string());
    CHECK_EQ(noa_check(off.pieces.at(off.ids[0]), a4, 2048, true), std::string());
    CHECK_EQ(noa_check(off.pieces.at(off.ids[1]), a1, 2048, true), std::string());
    CHECK(cbh_same_pieces(on.pieces.at(on.ids[0]), off.pieces.at(off.ids[0])));
    CHECK(cbh_same_pieces(on.pieces.at(on.ids[1]), off.pieces.at(off.ids[1])));
    CHECK(on.max_step_tok <= 4160 && off.max_step_tok <= 4160);
    CHECK_EQ(on.cbh["by_k"], (int64_t)1);
}

/* The engine's loop (a step issued ahead behind every step): with nothing waiting every step after the first is issued ahead.
 * Hold 0: a newcomer stops it (step_next declines while a request waits) and the next full step admits it. Hold 200 ms: the
 * steps issued ahead keep running through the hold; when the hold has run out step_next declines, and the full step after it
 * ends the hold (by timeout) and admits the newcomer. */
TEST(cbh_the_step_issued_ahead_keeps_running_through_a_hold) {
    CbhCfg c;
    c.ahead = true;
    c.arrive = { { 2, 1100 } };
    c.steps = 14;
    c.hold = 0;
    CbhRun off = cbh_run(c);
    c.hold = 200;
    CbhRun on = cbh_run(c);
    REQUIRE(off.ids.size() == 1 && on.ids.size() == 1);
    CHECK_EQ(off.issued.substr(0, 4), std::string("SAAS"));   /* step 2's step was issued before the newcomer came */
    CHECK_EQ(cbh_first(off, off.ids[0]), 3);
    CHECK_EQ(on.issued.substr(0, 9), std::string("SAAAAAAAS"));
    CHECK_EQ(cbh_first(on, on.ids[0]), 8);
    CHECK_EQ(on.cbh["holds"], (int64_t)1);
    CHECK_EQ(on.cbh["by_timeout"], (int64_t)1);
    CHECK_EQ(on.cbh["held_steps"], (int64_t)5);                 /* issued ahead at steps 2 .. 6 */
    CHECK_EQ(on.cbh["hold_ms"], (int64_t)240);                  /* from step 2's issue (120 ms) to step 8's plan (360 ms) */
    CHECK_EQ(noa_check(on.pieces.at(on.ids[0]), alone_ck(1100, 0, true), 2048, true), std::string());
}

/* CONTROL: off is today. The switches never set, set to 0, and on but unable to act (one decoder) plan every step the same --
 * newcomers, decoders, interleave 8, the step issued ahead -- and log nothing. */
TEST(cbh_off_plans_as_unset) {
    for (bool ahead : { false, true }) {
        CbhCfg c;
        c.ahead = ahead;
        c.il = 8;
        c.steps = 40;
        c.arrive = { { 1, 1100 }, { 3, 4419 }, { 4, 1330 }, { 9, 700 }, { 9, 900 }, { 20, 1100 } };
        c.touch = false;
        CbhRun unset = cbh_run(c);
        c.touch = true;
        c.hold = 0;
        CbhRun zero = cbh_run(c);
        CHECK(unset.sig == zero.sig);
        CHECK_EQ(unset.issued, zero.issued);
        CHECK(unset.log.find("lever CBH") == std::string::npos);
        CHECK_EQ(zero.cbh["holds"] + zero.cbh["held_steps"], (int64_t)0);
        CHECK_EQ(unset.cbh["in_force_ms"], (int64_t)0);
        c.decoders = 1;
        c.newc_tok = 1;             /* a newcomer ends at its prefill: never a second decoder */
        c.hold = 0;
        CbhRun one0 = cbh_run(c);
        c.hold = 500;
        CbhRun one5 = cbh_run(c);
        CHECK(one0.sig == one5.sig);
        CHECK_EQ(one0.issued, one5.issued);
        CHECK_EQ(one5.cbh["holds"], (int64_t)0);
        if (ahead) CHECK(unset.issued.find('A') != std::string::npos);
    }
}

/* ================================================================== lever G: the tail grid (ds41-tailgrid) */
/* fn4 sched_tail_grid F (0 off = today; 128 / 256 / 512 / 1024 / 2048) under --prefix-resume exact on the DS41 line's geometry
 * (interval 2048, span 2 from 16384 tokens, MNBT 4160: the resume grid 2048 + k x 4096; ds41_psk 4 + ds41_psk_cr 1 as the
 * line's pin: tail-only anchors). A plain prompt's tail snapshot moves from the grid point under its end to the deepest
 * multiple of F at or below N - 128, cutting the prompt's last cold chunk in two pieces of >= 128 rows; the next turn
 * resumes there. What the tests hold the scheduler to is option D's rule against the LEVER-OFF cold plan of the same
 * prompt: every piece lies inside one cold piece, and each of its ends is that piece's end or at least 128 rows from both
 * of the cold piece's ends -- the rule under which sched_bi_chunk 3 cuts beside other work today. */
namespace {
struct TgSwitch {
    explicit TgSwitch(int v) { CHECK(DstSwitches::set("sched_tail_grid", v)); }
    ~TgSwitch() { DstSwitches::set("sched_tail_grid", 0); }
};
using TgPieces = std::vector<std::pair<int64_t, int64_t>>;   /* [begin, end) */
TgPieces tg_pieces(const CrTurn& t) {
    TgPieces p;
    for (const CrStep& s : t.st) p.push_back({ s.ctx, s.ctx + s.n });
    return p;
}
/* "" when `on` keeps option D's rule against the cold plan `cold` (contiguous from its first piece's start to the end). */
std::string tg_rule(const TgPieces& on, const TgPieces& cold, int64_t N) {
    char why[200];
    if (on.empty() || cold.empty()) return "no pieces";
    for (size_t i = 0; i < on.size(); ++i) {
        const int64_t a = on[i].first, b = on[i].second;
        if (i > 0 && a != on[i - 1].second) return "not contiguous";
        const std::pair<int64_t, int64_t>* c = nullptr;
        for (const auto& k : cold)
            if (k.first <= a && a < k.second) c = &k;
        if (!c) { std::snprintf(why, sizeof why, "[%lld, %lld) starts in no cold piece", (long long)a, (long long)b); return why; }
        const int64_t A = c->first, B = c->second;
        const bool ok = b <= B && (a == A || a - A >= 128) && (b == B || B - b >= 128) && (b - a >= 128 || (a == A && b == B));
        if (!ok) {
            std::snprintf(why, sizeof why, "[%lld, %lld) in the cold piece [%lld, %lld)", (long long)a, (long long)b,
                          (long long)A, (long long)B);
            return why;
        }
    }
    if (on.back().second != N) return "does not reach the prompt's end";
    return "";
}
int64_t tg_json(const char* key) {
    const std::string g = fn4::state_json();
    return noa_json_num(g, g.find("\"tailgrid\": {"), key);
}
}  // namespace

TEST(lever_g_the_switch_takes_its_grids_and_starts_off) {
    std::string err;
    for (int v : { 0, 128, 256, 512, 1024, 2048 }) CHECK(fn4::check("sched_tail_grid", v, &err));
    for (int v : { -1, 1, 16, 64, 100, 384, 768, 4096 }) {
        CHECK(!fn4::check("sched_tail_grid", v, &err));
        CHECK(err.find("sched_tail_grid takes 0") != std::string::npos);
    }
    CHECK_EQ(fn4::sched_tail_grid(), 0);
    CHECK_EQ(fn4::switch_index("sched_tail_grid"), (int)fn4::kSchedTailGrid);
    CHECK(!fn4::sched_switch(fn4::kSchedTailGrid));
    {
        TgSwitch on(512);
        CHECK_EQ(fn4::sched_tail_grid(), 512);
        CHECK(fn4::state_json().find("\"sched_tail_grid\": 512") != std::string::npos);
        CHECK_EQ(tg_json("in_force"), (int64_t)512);
    }
    CHECK_EQ(fn4::sched_tail_grid(), 0);
    CHECK_EQ(tg_json("in_force"), (int64_t)0);
}

/* The rule itself: tail_checkpoint(N, F) over every F and many lengths -- today's point where F gains nothing (the end
 * within 128 rows of it, or below the first interval), else the deepest multiple of F at or below N - 128, at least F
 * past the grid point; and 0 = today's for every N. */
TEST(lever_g_the_tail_is_the_deepest_grid_multiple_that_leaves_128_rows) {
    Fixture f;
    cr_fixture(f);
    REQUIRE_EQ(f.start(), RAD_OK);
    const ChunkGeometry& g = f.sch.geometry();
    REQUIRE_EQ(g.grid, (int64_t)4096);
    REQUIRE_EQ(g.grid_origin, (int64_t)2048);
    int moved = 0;
    for (int64_t N = 100; N <= 30000; N += 61) {
        const int64_t base = f.sch.tail_checkpoint(N);
        CHECK_EQ(f.sch.tail_checkpoint(N, 0), base);
        for (int64_t F : { 128, 256, 512, 1024, 2048 }) {
            const int64_t t = f.sch.tail_checkpoint(N, F);
            if (t == base) {
                /* nothing deeper is admissible: no multiple of F in (base, N - 128] past the first interval */
                const int64_t d = (N - 128) / F * F;
                CHECK(N <= 128 || d <= std::max<int64_t>(base, 0) || d <= 2048);
                continue;
            }
            ++moved;
            CHECK(t > base);
            CHECK_EQ(t % F, (int64_t)0);
            CHECK(N - t >= 128 && N - t < 128 + F);
            CHECK(t > 2048);
            if (base > 0) CHECK(t - base >= F);
        }
    }
    CHECK(moved > 1000);
    CHECK_EQ(f.sch.tail_checkpoint(52000, 512), (int64_t)51712);
    CHECK_EQ(f.sch.tail_checkpoint(52000), (int64_t)51200);
    CHECK_EQ(f.sch.tail_checkpoint(51300, 512), (int64_t)51200);     /* 100 rows past the grid point: today's */
    CHECK_EQ(f.sch.tail_checkpoint(51840, 512), (int64_t)51712);     /* exactly 128 rows left */
    CHECK_EQ(f.sch.tail_checkpoint(51839, 512), (int64_t)51200);
}

/* A Claude Code conversation alone in the engine: a cold text, then turns that each extend the previous prompt by its
 * answer and a new message. Under F every request -- cold or resumed -- keeps option D's rule against the lever-off cold
 * plan of its own prompt; it takes exactly one snapshot, its fine tail; the next turn restores it and re-prefills
 * 128 .. 127 + F old rows where today's grid costs it up to 4096; no turn replays; every PSK target lies at or past its
 * chunk's end. With the switch at 0 the same conversation is today's (grid tails), also after fine snapshots exist. */
TEST(lever_g_a_conversation_resumes_at_its_fine_tail_inside_the_cold_runs_pieces) {
    const int64_t cold = 52000;
    const std::vector<std::pair<int64_t, int64_t>> turns = { { 52600, 51990 }, { 57000, 52590 }, { 57300, 56990 },
                                                             { 61500, 57290 }, { 61700, 61490 }, { 64000, 61690 } };
    std::vector<TgPieces> cold_off;   /* the lever-off cold plan of every prompt */
    std::vector<int64_t> tail_off;
    {
        TgSwitch off(0);
        const CrTurn c0 = cr_turn('T', cold, 0, {});
        cold_off.push_back(tg_pieces(c0));
        REQUIRE(c0.snaps.size() == 1);
        tail_off.push_back(c0.snaps[0]);
        for (const auto& tn : turns) {
            const CrTurn c = cr_turn('T', tn.first, 0, {});
            cold_off.push_back(tg_pieces(c));
            REQUIRE(c.snaps.size() == 1);
            tail_off.push_back(c.snaps[0]);
        }
        CHECK_EQ(tail_off[0], (int64_t)51200);
    }
    for (int F : { 128, 256, 512, 1024, 2048 }) {
        TgSwitch on(F);
        /* the cold text under the lever: the same pieces but the last, cut at the fine tail */
        const CrTurn c0 = cr_turn('T', cold, 0, {});
        CHECK_EQ(tg_rule(tg_pieces(c0), cold_off[0], cold), std::string(""));
        const int64_t f0 = (cold - 128) / F * F;
        CHECK((c0.snaps == std::vector<int64_t>{ f0 }));
        std::vector<int64_t> avail = c0.snaps;
        int64_t prev_n = cold, prev_tail = f0, saved = 0;
        for (size_t k = 0; k < turns.size(); ++k) {
            const int64_t N = turns[k].first, match = turns[k].second;
            avail = cr_path(avail, match);
            const CrTurn t = cr_turn('T', N, match, avail);
            REQUIRE(!t.st.empty());
            /* resumed at the previous prompt's fine tail, restored, never by a replay */
            CHECK_EQ(t.st[0].ctx, prev_tail);
            CHECK(t.st[0].restore >= 0);
            CHECK_EQ(t.m.psk_cr_hits, (int64_t)0);
            for (const CrStep& s : t.st) { CHECK_EQ(s.replay, 0); CHECK(s.ahead >= 0); }
            /* the old rows it re-prefills: 128 .. 127 + F (today: prev_n - the grid point under it) */
            CHECK(prev_n - prev_tail >= 128 && prev_n - prev_tail < 128 + F);
            saved += prev_tail - (prev_tail < 2048 ? 0 : 2048 + (prev_tail - 2048) / 4096 * 4096);
            /* option D's rule against this prompt's lever-off cold plan */
            CHECK_EQ(tg_rule(tg_pieces(t), cold_off[k + 1], N), std::string(""));
            /* one snapshot: the fine tail, a chunk's end, with its target 0 rows ahead of that chunk */
            const int64_t fk = std::max<int64_t>((N - 128) / F * F, tail_off[k + 1]);
            if (fk > t.st[0].ctx) CHECK((t.snaps == std::vector<int64_t>{ fk }));
            else                  CHECK(t.snaps.empty());   /* the prompt grew less than a grid step: the snapshot it resumed at */
            bool at_end = false;
            for (const CrStep& s : t.st)
                if (s.ctx + s.n == fk) { at_end = true; CHECK_EQ(s.ahead, (int64_t)0); CHECK_EQ(s.ckpts, 1); }
            CHECK(at_end || fk == t.st[0].ctx);
            for (int64_t p : t.snaps) avail.push_back(p);
            std::sort(avail.begin(), avail.end());
            prev_n = N;
            prev_tail = fk;
        }
        (void)saved;
    }
    /* back at 0 with the fine snapshots still cached: every prompt's cold plan is today's */
    TgSwitch off(0);
    for (size_t k = 0; k < turns.size(); ++k) {
        const CrTurn c = cr_turn('T', turns[k].first, 0, {});
        CHECK(tg_pieces(c) == cold_off[k + 1]);
        CHECK((c.snaps == std::vector<int64_t>{ tail_off[k + 1] }));
    }
}

/* Many conversations, every F: random cold lengths and turn growths (an answer of 50 .. 3000 tokens + a message of
 * 20 .. 2500), the match a few tokens short of the previous prompt (the re-rendered generation prompt). Every turn keeps
 * option D's rule against its own lever-off cold plan, resumes at the previous fine tail and never replays. */
TEST(lever_g_random_conversations_keep_the_cold_runs_pieces) {
    uint32_t rng = 20261113u;
    auto rnd = [&](int lo, int hi) { rng = rng * 1664525u + 1013904223u; return lo + (int)((rng >> 8) % (uint32_t)(hi - lo + 1)); };
    int turns_checked = 0;
    for (int conv = 0; conv < 10; ++conv) {
        const int F = (int[]){ 128, 256, 512, 1024, 2048 }[conv % 5];
        int64_t N = rnd(3000, 30000);   /* (the fixture's prompts end below 64536 tokens) */
        std::vector<int64_t> lens = { N };
        for (int k = 0; k < 5; ++k) { N += rnd(50, 3000) + rnd(20, 2500); lens.push_back(N); }
        std::vector<TgPieces> cold_off;
        {
            TgSwitch off(0);
            for (int64_t L : lens) cold_off.push_back(tg_pieces(cr_turn('T', L, 0, {})));
        }
        TgSwitch on(F);
        const CrTurn c0 = cr_turn('T', lens[0], 0, {});
        CHECK_EQ(tg_rule(tg_pieces(c0), cold_off[0], lens[0]), std::string(""));
        std::vector<int64_t> avail = c0.snaps;
        for (size_t k = 1; k < lens.size(); ++k) {
            const int64_t match = lens[k - 1] - rnd(1, 12);
            avail = cr_path(avail, match);
            const int64_t deepest = avail.empty() ? 0 : avail.back();
            const CrTurn t = cr_turn('T', lens[k], match, avail);
            REQUIRE(!t.st.empty());
            const std::string why = tg_rule(tg_pieces(t), cold_off[k], lens[k]);
            if (!why.empty()) std::fprintf(stderr, "conv %d F %d turn %zu N %lld: %s\n", conv, F, k, (long long)lens[k], why.c_str());
            CHECK_EQ(why, std::string(""));
            if (deepest > 0) { CHECK_EQ(t.st[0].ctx, deepest); CHECK(t.st[0].restore >= 0); }
            CHECK_EQ(t.m.psk_cr_hits, (int64_t)0);
            for (const CrStep& s : t.st) CHECK(s.ahead >= 0);
            for (int64_t p : t.snaps) avail.push_back(p);
            std::sort(avail.begin(), avail.end());
            ++turns_checked;
        }
    }
    CHECK_EQ(turns_checked, 50);
}

/* A prompt cut short of a cached one, ending within 128 rows past that one's fine tail: resuming there would run its last
 * rows in a piece of another form band than its cold run's, so the snapshot is declined (counted) and the request is
 * served as a hit with no snapshot -- here by lever PSK-CR's replay to the grid point -- and its pieces from there are its
 * cold run's own. From 128 rows on it resumes at the fine tail. */
TEST(lever_g_a_fine_snapshot_within_128_rows_of_a_prompts_end_is_not_resumed_at) {
    TgPieces cold_near, cold_far;
    {
        TgSwitch off(0);
        cold_near = tg_pieces(cr_turn('T', 51784, 0, {}));
        cold_far = tg_pieces(cr_turn('T', 51840, 0, {}));
    }
    TgSwitch on(512);
    const CrTurn near = cr_turn('T', 51784, 51780, { 51712 });     /* 72 rows past the fine tail of a 52000-token prompt */
    REQUIRE(!near.st.empty());
    CHECK(near.st[0].ctx != 51712);
    CHECK_EQ(near.st[0].restore, -1);
    CHECK_EQ(near.m.psk_cr_hits, (int64_t)1);                      /* the grid point 51200 by a replay */
    TgPieces rest;
    for (const CrStep& s : near.st)
        if (!s.replay) rest.push_back({ s.ctx, s.ctx + s.n });
    CHECK((rest == TgPieces{ { 51200, 51784 } }));
    CHECK(rest.back() == cold_near.back());
    const CrTurn far = cr_turn('T', 51840, 51836, { 51712 });      /* 128 rows: one piece [51712, 51840) */
    REQUIRE(!far.st.empty());
    CHECK_EQ(far.st[0].ctx, (int64_t)51712);
    CHECK(far.st[0].restore >= 0);
    CHECK((tg_pieces(far) == TgPieces{ { 51712, 51840 } }));
    CHECK_EQ(tg_rule(tg_pieces(far), cold_far, 51840), std::string(""));
}

/* Beside decode rows under the line's chunking (sched_bi_chunk 3, sched_noa 2, six decoders): a turn resumed at a fine
 * tail (a hit at 56990 of a 57000-token prompt's cache, its snapshot at 56832) keeps option D's rule against its lever-off
 * cold plan in every piece, takes its own fine tail and is counted; the long answer case crosses the cold run's next
 * chunk start and goes on on the cold grid. GET "tailgrid". */
TEST(lever_g_a_resumed_turn_beside_decode_rows_keeps_option_d) {
    struct Case { int64_t N, match, lin; };
    /* 57300: grown less than a grid step past the snapshot (one piece, no new snapshot); 58000: its own fine tail at 57856;
     * 57000 from 52224: the cold piece [51200, 55296) left over, cut by option D at 53248, then the grid chunk to the tail */
    for (const Case& c : { Case{ 57300, 56990, 56832 }, Case{ 58000, 56990, 56832 }, Case{ 57000, 52590, 52224 } }) {
        TgPieces cold;
        {
            TgSwitch off(0);
            cold = tg_pieces(cr_turn('T', c.N, 0, {}));
        }
        TgSwitch on(512);
        NoaCfg n;
        n.decoders = 6;
        n.L = 0;
        n.lens = { (int)c.N };
        n.after_chunks = 0;
        n.steps = 60;
        n.hit = c.match;
        n.hit_lin = c.lin;
        n.psk = 2;
        n.psk_k = 4;
        const NoaRun r = noa_run(n);
        REQUIRE(r.ids.size() == 1);
        const auto it = r.pieces.find(r.ids[0]);
        REQUIRE(it != r.pieces.end());
        TgPieces on_p;
        int64_t fine_ck = -1;
        const int64_t fk = (c.N - 128) / 512 * 512;
        for (const PskPiece& p : it->second) {
            on_p.push_back({ p.at, p.at + p.n });
            if (p.ck == fk) fine_ck = p.ck;
            if (p.ahead != Scheduler::kPskNoPlan) CHECK(p.ahead >= 0);
        }
        REQUIRE(!on_p.empty());
        CHECK_EQ(on_p[0].first, c.lin);
        const std::string why = tg_rule(on_p, cold, c.N);
        if (!why.empty()) std::fprintf(stderr, "N %lld: %s\n", (long long)c.N, why.c_str());
        CHECK_EQ(why, std::string(""));
        CHECK_EQ(fine_ck, fk > c.lin ? fk : (int64_t)-1);
        if (c.N == 57000)
            CHECK((on_p == TgPieces{ { 52224, 53248 }, { 53248, 55296 }, { 55296, 56832 }, { 56832, 57000 } }));
        if (c.N == 58000) CHECK((on_p == TgPieces{ { 56832, 57856 }, { 57856, 58000 } }));
        if (c.N == 57300) CHECK((on_p == TgPieces{ { 56832, 57300 } }));
        CHECK(r.max_step_tok <= 4160);
        CHECK(tg_json("latched") >= 1);
        CHECK_EQ(tg_json("resumes"), (int64_t)1);
        CHECK_EQ(tg_json("resume_rows_saved"), (int64_t)(c.lin - (2048 + (c.lin - 2048) / 4096 * 4096)));
        CHECK_EQ(tg_json("tails"), (int64_t)(fk > c.lin ? 1 : 0));
        CHECK_EQ(tg_json("tail_rows"), fk > c.lin ? c.N - fk : (int64_t)0);
        CHECK_EQ(tg_json("inert"), (int64_t)0);
    }
}


/* Lever TI (ds41-pf2, fn4 sched_tail_inner 1 beside sched_tail_grid F): lever G's fine tail is taken INSIDE the prompt's last
 * chunk, which runs whole. Against lever G alone (the same conversation, the same snapshots): every prompt's pieces are G's
 * with the two pieces around the fine tail joined into one -- exactly the lever-off cold plan's last piece --, its snapshot
 * positions are G's (the fine tail), the snapshot sits inside that chunk (one checkpoint in the step, PSK's next target
 * tail - end < 0 rows ahead: its cone covered), and every next turn resumes at the same fine tail. At 0, or where the engine
 * found no plugin serving it (not capable), lever G's plan exactly. */
namespace {
struct TiSwitch {
    explicit TiSwitch(int v) { CHECK(DstSwitches::set("sched_tail_inner", v)); }
    ~TiSwitch() { DstSwitches::set("sched_tail_inner", 0); }
};
}  // namespace

TEST(lever_ti_the_switch_takes_0_and_1_and_starts_off) {
    std::string err;
    for (int v : { 0, 1 }) CHECK(fn4::check("sched_tail_inner", v, &err));
    for (int v : { -1, 2, 512 }) {
        CHECK(!fn4::check("sched_tail_inner", v, &err));
        CHECK(err.find("sched_tail_inner takes 0") != std::string::npos);
    }
    CHECK_EQ(fn4::sched_tail_inner(), 0);
    CHECK_EQ(fn4::switch_index("sched_tail_inner"), (int)fn4::kSchedTailInner);
    CHECK(!fn4::sched_switch(fn4::kSchedTailInner));
    {
        TiSwitch on(1);
        CHECK_EQ(fn4::sched_tail_inner(), 1);
        CHECK(fn4::state_json().find("\"sched_tail_inner\": 1") != std::string::npos);
    }
    CHECK_EQ(fn4::sched_tail_inner(), 0);
}

/* tp8 loop P44 (fn4 sched_srf, upstream 67cf57b8): 0 / 1 / 2, the scheduler's (read at every step it plans, not handed over as a
 * sched switch), off at start; no -1 (its read-back is never -1: the kit's arm undo POSTs 0 / 1 / 2). */
namespace {
struct SrfSwitch {
    explicit SrfSwitch(int v) { CHECK(DstSwitches::set("sched_srf", v)); }
    ~SrfSwitch() { DstSwitches::set("sched_srf", 0); }
};
}  // namespace

TEST(srf_the_switch_takes_0_1_2_and_starts_off) {
    std::string err;
    for (int v : { 0, 1, 2 }) CHECK(fn4::check("sched_srf", v, &err));
    for (int v : { -1, 3, 256 }) {
        CHECK(!fn4::check("sched_srf", v, &err));
        CHECK(err.find("sched_srf takes 0") != std::string::npos);
    }
    CHECK_EQ(fn4::sched_srf(), 0);
    CHECK_EQ(fn4::switch_index("sched_srf"), (int)fn4::kSchedSrf);
    CHECK(!fn4::sched_switch(fn4::kSchedSrf));
    CHECK(!fn4::follower_switch(fn4::kSchedSrf));
    for (int v : { 1, 2 }) {
        SrfSwitch on(v);
        CHECK_EQ(fn4::sched_srf(), v);
        CHECK(fn4::state_json().find(fmt("\"sched_srf\": %d", v)) != std::string::npos);
    }
    CHECK_EQ(fn4::sched_srf(), 0);
}

/* tp8 loop U1 (fn4 aql_hdp, upstream 8f2e511f's first half): 0 / 1, the AQL backend's (read at every doorbell), off at start, a
 * follower switch (both nodes' doorbells); not a scheduler switch. The host backend has no doorbell: the value is only stored. */
TEST(aql_hdp_the_switch_takes_0_1_starts_off_and_acts_on_both_nodes) {
    std::string err;
    for (int v : { 0, 1 }) CHECK(fn4::check("aql_hdp", v, &err));
    for (int v : { -1, 2, 256 }) {
        CHECK(!fn4::check("aql_hdp", v, &err));
        CHECK(err.find("aql_hdp takes 0") != std::string::npos);
        CHECK(err.find("8f2e511f") != std::string::npos);
    }
    CHECK_EQ(fn4::aql_hdp(), 0);
    CHECK_EQ(fn4::switch_index("aql_hdp"), (int)fn4::kAqlHdp);
    CHECK(!fn4::sched_switch(fn4::kAqlHdp));
    CHECK(fn4::follower_switch(fn4::kAqlHdp));
    CHECK(fn4::state_json().find("\"aql_hdp\": 0") != std::string::npos);
    CHECK(DstSwitches::set("aql_hdp", 1));
    CHECK_EQ(fn4::aql_hdp(), 1);
    CHECK(fn4::state_json().find("\"aql_hdp\": 1") != std::string::npos);
    CHECK(DstSwitches::set("aql_hdp", 0));
    CHECK_EQ(fn4::aql_hdp(), 0);
}

/* sched_srf 1 = upstream 67cf57b8's rule on one scheduler (its sched_test case, ported): prompts prefilling side by side with
 * nothing decoding and no one to admit -- each step is spanned and holds one of them, the one with the fewest prompt tokens left,
 * and the others wait for it (no share of it); as soon as one decodes, the steps are shared again; no wide step holds two
 * sequences; the step after a request finished beside a prefill is shared, then spans again. At 0 the second step is today's:
 * both prompts in it. */
TEST(srf_1_concurrent_prefills_with_nothing_decoding_span_one_at_a_time_shortest_first) {
    {   /* control: sched_srf 0 shares the steps after the admission */
        Fixture f;
        f.build();
        f.cfg.prefill_span = 2;
        CHECK_OK(f.start());
        f.submit(3000, 3);
        f.submit(1300, 3);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ((int64_t)f.sch.plan().e.size(), 2ll);
        f.commit_plain();
    }
    SrfSwitch on(1);
    Fixture f;
    f.build();                            /* interval 256, budget 512, shared 256 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    const uint64_t a = f.submit(3000, 3);
    const uint64_t b = f.submit(1300, 3);
    /* Both admitted in the first step, at the shared budget. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int64_t)f.sch.plan().e.size(), 2ll);
    CHECK(f.sch.plan().n_tok <= 256);
    f.commit_plain();
    /* Then b, the shorter, alone in spanned steps up to its first token; a takes nothing. */
    std::vector<int32_t> b_chunks;
    int b_first = -1;
    for (int k = 0; k < 20 && b_first < 0; ++k) {
        CHECK(f.sch.step() != nullptr);
        const StepPlan& p = f.sch.plan();
        CHECK_EQ((int64_t)p.e.size(), 1ll);
        CHECK_EQ(p.e[0].seq, b);
        CHECK(p.e[0].is_prefill);
        b_chunks.push_back(p.e[0].n_tokens);
        if (p.e[0].produces_token) b_first = k;
        f.commit_plain();
    }
    CHECK(b_first >= 0);
    int wide = 0;
    for (int32_t n : b_chunks) wide += n > 256;
    CHECK(wide >= 1);
    /* With b decoding, every step is shared: no entry and no step past 256 tokens, a's chunk beside b's row. */
    int shared = 0, a_beside_b = 0;
    for (int k = 0; k < 6; ++k) {
        if (!f.sch.step()) break;
        const StepPlan& p = f.sch.plan();
        bool has_a = false, has_b = false;
        for (const StepEntry& e : p.e) {
            has_a = has_a || e.seq == a;
            has_b = has_b || (e.seq == b && !e.is_prefill);
        }
        if (!has_b) break;
        ++shared;
        CHECK(p.n_tok <= 256);
        a_beside_b += has_a;
        f.commit_plain();
    }
    CHECK(shared >= 2);
    CHECK_EQ(a_beside_b, shared);
    int64_t c[Scheduler::kSrfN];
    f.sch.srf_counts(c);
    CHECK(c[Scheduler::kSrfAlone] >= (int64_t)b_chunks.size());
    CHECK(c[Scheduler::kSrfOut] >= (int64_t)b_chunks.size());

    /* A step wider than the shared budget never holds more than one sequence, over a run of three prompts arriving together
     * (RadBuildCtx::max_mixed_tok). */
    Fixture g;
    g.build();
    g.cfg.prefill_span = 2;
    CHECK_OK(g.start());
    g.submit(2500, 2);
    g.submit(900, 2);
    g.submit(1700, 2);
    int bad = 0, spans = 0, done = 0;
    for (int k = 0; k < 200; ++k) {
        if (!g.sch.step()) break;
        const StepPlan& p = g.sch.plan();
        if (p.n_tok > 256) { ++spans; bad += p.e.size() != 1; }
        g.commit_plain();
        ++done;
    }
    CHECK(done < 200);
    CHECK(spans >= 3);
    CHECK_EQ(bad, 0);

    /* The step after a request finishes beside a prefill is shared: a follow-up turn its client sends at once waits for that
     * step, not for a span. Then the prefill spans again. */
    Fixture h;
    h.build();
    h.cfg.prefill_span = 2;
    CHECK_OK(h.start());
    const uint64_t d = h.submit(100, 3);
    CHECK(h.sch.step() != nullptr);
    h.commit_plain();
    const uint64_t p = h.submit(3000, 2);
    std::vector<int64_t> after;           /* p's chunks from the step after d finished */
    bool d_gone = false;
    for (int k = 0; k < 40 && after.size() < 3; ++k) {
        if (!h.sch.step()) break;
        bool has_d = false;
        int64_t p_tok = 0;
        for (const StepEntry& e : h.sch.plan().e) {
            has_d = has_d || e.seq == d;
            if (e.seq == p) p_tok = e.n_tokens;
        }
        if (!has_d) { d_gone = true; after.push_back(p_tok); }
        h.commit_plain();
    }
    CHECK(d_gone);
    CHECK_EQ((int64_t)after.size(), 3ll);
    if (after.size() == 3) {
        CHECK(after[0] > 0 && after[0] <= 256);
        CHECK(after[1] > 256 || after[2] > 256);
    }
}

TEST(lever_ti_a_conversation_takes_its_fine_tails_inside_one_last_chunk) {
    const int64_t cold = 52000;
    const std::vector<std::pair<int64_t, int64_t>> turns = { { 52600, 51990 }, { 57000, 52590 }, { 57300, 56990 },
                                                             { 61500, 57290 }, { 61700, 61490 }, { 64000, 61690 } };
    std::vector<TgPieces> cold_off;
    {
        TgSwitch off(0);
        cold_off.push_back(tg_pieces(cr_turn('T', cold, 0, {})));
        for (const auto& tn : turns) cold_off.push_back(tg_pieces(cr_turn('T', tn.first, 0, {})));
    }
    for (int F : { 128, 256, 512, 1024, 2048 }) {
        TgSwitch g(F);
        TiSwitch on(1);
        std::vector<int64_t> avail_g, avail_t;
        int joined = 0;
        for (size_t k = 0; k <= turns.size(); ++k) {
            const int64_t N = k == 0 ? cold : turns[k - 1].first, match = k == 0 ? 0 : turns[k - 1].second;
            avail_g = cr_path(avail_g, match);
            avail_t = cr_path(avail_t, match);
            const CrTurn tg = cr_turn('T', N, match, avail_g, /*ti=*/false);
            const CrTurn tt = cr_turn('T', N, match, avail_t, /*ti=*/true);
            REQUIRE(!tg.st.empty() && !tt.st.empty());
            /* the same snapshots, the same resume point */
            CHECK(tt.snaps == tg.snaps);
            CHECK_EQ(tt.st[0].ctx, tg.st[0].ctx);
            CHECK_EQ(tt.st[0].restore >= 0, tg.st[0].restore >= 0);
            const TgPieces pg = tg_pieces(tg), pt = tg_pieces(tt);
            const int64_t fsn = tg.snaps.empty() ? -1 : tg.snaps.back();
            const bool on_grid = fsn >= 2048 && (fsn - 2048) % 4096 == 0;   /* the resume grid (2048 + k x 4096): a chunk's start */
            if (fsn > 0 && !on_grid && pg.size() >= 2 && pg.back().first == fsn && pg.back().second == N) {
                /* G cut the last chunk at its fine tail: TI joins the two pieces around it */
                TgPieces want(pg.begin(), pg.end() - 2);
                want.push_back({ pg[pg.size() - 2].first, N });
                CHECK(pt == want);
                const CrStep& s = tt.st.back();
                CHECK_EQ(s.ctx + s.n, N);
                CHECK_EQ(s.ckpts, 1);
                CHECK_EQ(s.ahead, tg.snaps.back() - N);   /* the next target inside the chunk */
                CHECK(s.ahead < 0);
                ++joined;
            } else {
                CHECK(pt == pg);
            }
            /* option D's rule against the lever-off cold plan still holds (the joined piece is that plan's own) */
            CHECK_EQ(tg_rule(pt, cold_off[k], N), std::string(""));
            for (int64_t p : tg.snaps) avail_g.push_back(p);
            for (int64_t p : tt.snaps) avail_t.push_back(p);
            std::sort(avail_g.begin(), avail_g.end());
            std::sort(avail_t.begin(), avail_t.end());
        }
        CHECK(joined >= (F <= 512 ? 3 : 1));
    }
    /* not capable (no plugin serving it): lever G's plan exactly */
    {
        TgSwitch g(512);
        TiSwitch on(1);
        const CrTurn tg = cr_turn('T', cold, 0, {}, /*ti=*/false);
        CHECK(tg_pieces(tg).size() >= 2);
        CHECK_EQ(tg.st.back().ckpts, 0);
    }
}


/* Lever SAL (ds41-pf2, fn4 stage_lone): 0 / 1, the engine's (handed over as stage_short), off at start. */
TEST(lever_sal_the_switch_takes_0_and_1_and_starts_off) {
    std::string err;
    for (int v : { 0, 1 }) CHECK(fn4::check("stage_lone", v, &err));
    for (int v : { -1, 2, 1025 }) {
        CHECK(!fn4::check("stage_lone", v, &err));
        CHECK(err.find("stage_lone takes 0") != std::string::npos);
    }
    CHECK_EQ(fn4::switch_index("stage_lone"), (int)fn4::kStageLone);
    CHECK(fn4::sched_switch(fn4::kStageLone));
    CHECK(fn4::state_json().find("\"stage_lone\": 0") != std::string::npos);
}

/* ================================================================== tp8 S5: lanes (core/sched/lanes.h) */
namespace {

/* N lane schedulers over one program, one fake KV pool and no prefix cache, as a two-node pipeline's
 * leader runs them: full attention in 16-token blocks (the quantum), 64-token steps, six sequences. */
struct Lanes {
    Program prog;
    Config  cfg;
    FakeKV  kv;
    FakePC  pc;
    Scheduler s0;
    std::vector<std::unique_ptr<Scheduler>> more;
    LaneSet set;
    uint64_t next_id = 1;
    /* tp8 S5 review: a linear state group as well, so chunks end on a checkpoint-interval grid of
     * `grid` tokens as DeepSeek's do (a lent chunk then sits beside other rows of its home's step) */
    int64_t grid = 0;
    /* tp8 prefill P0: a fixed WINDOW group of `win` tokens as DeepSeek's kv_window, sliding as KVManager's (FakeKV
     * slide), its pool `win_pool` blocks (0: the fake's default); `seqs` the cards (--max-num-seqs) */
    int64_t win = 0, win_pool = 0, seqs = 6;
    int win_g = -1;
    /* tp8 loop M2: the step budget, context and stager floor (the DS line's 4160 / 1024 for lever DST's cases) */
    int64_t max_tok = 64, max_ctx = 1024, stage_rows = 0, kv_blocks = 4096;
    /* tp8 loop F22: --num-speculative-tokens (0: no drafter; with prog.drafter set before start(), a drafting deployment) */
    int n_spec = 0;
    ~Lanes() { for (auto& s : more) s->fini(); s0.fini(); }
    Scheduler& at(int l) { return l == 0 ? s0 : *more[(size_t)(l - 1)]; }
    int start(int n, int64_t split_min = 16) {
        prog.meta.n_ctx_train = std::max<int64_t>(4096, max_ctx);
        prog.meta.n_vocab = std::max<int64_t>(65536, max_ctx + 2000);   /* tp8 prefill P1: prompt ids base + i */
        prog.ops.emplace_back();
        prog.buffers.emplace_back();
        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        prog.kv_groups.push_back(attn);
        if (grid > 0) {
            KVGroupInfo lin;
            lin.name = "lin";
            lin.decl.kind = RAD_KV_LINEAR;
            lin.decl.state_dim[0] = 16; lin.decl.state_dim[1] = 16;
            prog.kv_groups.push_back(lin);
            cfg.checkpoint_interval = grid;
        }
        if (win > 0) {
            KVGroupInfo w;
            w.name = "win";
            w.decl.kind = RAD_KV_WINDOW;
            w.decl.window = win;
            w.block_size = 16;
            win_g = (int)prog.kv_groups.size();
            prog.kv_groups.push_back(w);
        }
        cfg.max_tok = max_tok;
        cfg.max_seqs = seqs;
        cfg.max_ctx = max_ctx;
        cfg.prefix_cache = false;
        cfg.prefill_room_gate = false;
        cfg.n_spec = n_spec;
        kv.setup(prog.kv_groups, cfg.max_ctx, kv_blocks, 64);
        if (win > 0) {
            kv.slide = true;
            if (win_pool > 0) kv.resize_group(win_g, win_pool);
        }
        pc.kv = &kv;
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        s0.set_lanes(n, &set);
        RAD_TRY(s0.init(prog, cfg, &kv, &pc, rio));
        std::vector<Scheduler*> all{ &s0 };
        for (int l = 1; l < n; ++l) {
            more.push_back(std::make_unique<Scheduler>(s0, l, n, &set));
            RAD_TRY(more.back()->init(prog, cfg, &kv, &pc, rio));
            all.push_back(more.back().get());
        }
        set.init(all);
        set.set_split(true, split_min);
        if (stage_rows > 0)
            for (Scheduler* l : all) l->set_stage_rows(stage_rows);
        return RAD_OK;
    }
    uint64_t submit(int lane, int n_prompt, int max_tokens, int base = 1000) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->ignore_eos = true;
        r->prompt.resize((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) r->prompt[(size_t)i] = base + i;
        const uint64_t id = r->id;
        return at(lane).add(std::move(r)) < 0 ? 0 : id;
    }
    /* Lane `l` plans with `planners` named as the lanes that plan before it plans again. */
    const RadBatch* plan(int l, std::vector<int> planners) {
        set.set_planners(std::move(planners));
        return at(l).step();
    }
    int commit(int l, int32_t token) {
        const StepPlan& p = at(l).plan();
        std::vector<int32_t> a(p.e.size(), 0), t(p.e.size(), token);
        std::vector<uint8_t> e(p.e.size(), 0);
        Scheduler::StepResult r;
        r.n_seq = (int)p.e.size();
        r.n_accepted = a.data();
        r.token = t.data();
        r.eos = e.data();
        return at(l).commit(r);
    }
    const StepEntry* entry_of(int l, uint64_t id) {
        for (const StepEntry& e : at(l).plan().e) if (e.seq == id) return &e;
        return nullptr;
    }
    const Request* req(int l, uint64_t id) { return at(l).find(id); }
    /* tp8 prefill P0: every query of lane l's last build reads, through the staged window row (the kernel's mapping,
     * libr4d r4d_ds41_sparse.hip: first = the row's last position + 1 - seqused, key j at row[(j - first) / block]),
     * the block the manager holds position j in, for each of its window's keys j in [p - win + 1, p]. Returns the
     * keys read wrong or missing. */
    int64_t window_misses(int l, const RadBatch* b) {
        if (!b || win_g < 0) return 0;
        const RadKVGroupBatch& g = b->kv[win_g];
        const int64_t bs = 16;
        int64_t bad = 0;
        const StepPlan& p = at(l).plan();
        for (size_t i = 0; i < p.e.size(); ++i) {
            const StepEntry& e = p.e[i];
            if (e.n_tokens <= 0) continue;
            const int32_t* row = g.block_table + (int64_t)i * g.block_table_pitch;
            const int64_t last = (int64_t)e.ctx_len + e.n_tokens - 1;
            const int64_t first = last + 1 - g.seqused_h[i];
            const std::vector<int32_t>& bt = kv.block_table(e.seq, win_g);
            const int64_t f0 = kv.first_block_pos(e.seq, win_g);
            for (int64_t q = e.ctx_len; q <= last; ++q)
                for (int64_t j = std::max<int64_t>(0, q - win + 1); j <= q; ++j) {
                    const int64_t r = j - first, m = j - f0;
                    const bool held = m >= 0 && m / bs < (int64_t)bt.size();
                    const bool seen = r >= 0 && r / bs < g.block_table_pitch;
                    if (!held || !seen || row[r / bs] != bt[(size_t)(m / bs)]) ++bad;
                }
        }
        return bad;
    }
};

/* tp8 prefill P0: a cold prompt through the engine's lane loop (Engine::step_loop_lanes: every lane with no step in
 * flight plans, the idle lanes after it right behind it, then the oldest tail commits). Returns the chunks of `s` in
 * the order they ran; `misses` the window keys any build mapped wrong (Lanes::window_misses). */
static std::vector<std::pair<int, int>> run_cold(Lanes& L, int lanes, uint64_t s, int64_t* misses, int* failed) {
    std::vector<std::pair<int, int>> got;
    std::vector<int> fifo;
    std::vector<char> busy((size_t)lanes, 0);
    int next = 0;
    *misses = 0;
    *failed = 0;
    for (int round = 0; round < 400; ++round) {
        const Request* r0 = L.req(0, s);
        if (!r0 || r0->state == ReqState::Failed || r0->state == ReqState::Finished || !r0->output.empty()) break;
        std::vector<int> todo;
        for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
        for (size_t at = 0; at < todo.size(); ++at) {
            const int l = todo[at];
            if (busy[(size_t)l]) continue;
            std::vector<int> order, idle_after;
            for (int j = 1; j < lanes; ++j)
                if (!busy[(size_t)((l + j) % lanes)]) {
                    order.push_back((l + j) % lanes);
                    idle_after.push_back((l + j) % lanes);
                }
            for (int f : fifo) order.push_back(f);
            const RadBatch* b = L.plan(l, order);
            if (!b) continue;
            *misses += L.window_misses(l, b);
            for (const StepEntry& e : L.at(l).plan().e)
                if (e.seq == s && e.is_prefill) got.emplace_back(e.ctx_len, e.ctx_len + e.n_tokens);
            busy[(size_t)l] = 1;
            fifo.push_back(l);
            next = (l + 1) % lanes;
            todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
        }
        if (fifo.empty()) break;
        const int l = fifo.front();
        fifo.erase(fifo.begin());
        if (L.commit(l, 5) < 0) ++*failed;
        busy[(size_t)l] = 0;
    }
    const Request* r = L.req(0, s);
    if (!r || r->state == ReqState::Failed) ++*failed;
    return got;
}

}  /* namespace */

/* Each lane holds its own run of the cards (the StepSlots), so two lanes' sequences never share a
 * slot, and a lane admits at most that many. */
TEST(lanes_take_their_own_cards) {
    Lanes L;
    REQUIRE_EQ(L.start(2), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK(L.submit(0, 8, 4) != 0);
    for (int i = 0; i < 4; ++i) CHECK(L.submit(1, 8, 4) != 0);
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    CHECK_EQ((int)L.at(0).plan().e.size(), 3);   /* 3 cards a lane of 6 */
    CHECK_EQ((int)L.at(1).plan().e.size(), 3);
    for (const StepEntry& e : L.at(0).plan().e) CHECK(e.card >= 0 && e.card < 3);
    for (const StepEntry& e : L.at(1).plan().e) CHECK(e.card >= 3 && e.card < 6);
}

/* A new request goes to the lane with the fewest live requests; a request is found in its lane. */
TEST(the_lane_set_routes_to_the_least_loaded_lane) {
    Lanes L;
    REQUIRE_EQ(L.start(3), RAD_OK);
    CHECK_EQ(L.set.pick(), 0);
    const uint64_t a = L.submit(0, 8, 4);
    CHECK_EQ(L.set.pick(), 1);
    const uint64_t b = L.submit(1, 8, 4);
    CHECK_EQ(L.set.pick(), 2);
    const uint64_t c = L.submit(2, 8, 4);
    CHECK_EQ(L.set.pick(), 0);
    CHECK(L.set.owner(a) == &L.at(0));
    CHECK(L.set.owner(b) == &L.at(1));
    CHECK(L.set.owner(c) == &L.at(2));
    CHECK(L.set.owner(999) == nullptr);
    /* RADIANCE_PP_PICK=rr: equally loaded lanes in turn */
    Lanes R;
    REQUIRE_EQ(R.start(3), RAD_OK);
    R.set.set_pick_rr(true);
    CHECK_EQ(R.set.pick(), 0);
    CHECK_EQ(R.set.pick(), 1);
    CHECK_EQ(R.set.pick(), 2);
    CHECK_EQ(R.set.pick(), 0);
    (void)R.submit(1, 8, 4);
    const int p1 = R.set.pick(), p2 = R.set.pick();
    CHECK(p1 != 1 && p2 != 1 && p1 != p2);   /* lane 1 holds a request: the other two in turn */
}

/* The newcomer's chunks over the lanes (tp8 M56: the one-process grid): a 200-token prompt that one
 * process plans as [0, 64) [64, 128) [128, 192) [192, 200) runs those chunks whole, alternating over
 * two lanes: lane 0 keeps its chunk and lends the next to lane 1, which plans it first; the home lane
 * plans the chunk after behind the one in flight and lends the last (the first token's) again; every
 * commit lands in position order on the home lane's request. */
TEST(a_newcomers_one_process_chunks_alternate_over_the_lanes_in_position_order) {
    Lanes L;
    REQUIRE_EQ(L.start(2), RAD_OK);
    const uint64_t s = L.submit(0, 200, 3);
    REQUIRE(s != 0);
    /* lane 0, step 1: its own [0, 64), lane 1's [64, 128) */
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    const StepEntry* e0 = L.entry_of(0, s);
    REQUIRE(e0 != nullptr);
    CHECK_EQ(e0->ctx_len, 0);
    CHECK_EQ(e0->n_tokens, 64);
    CHECK(!e0->produces_token);
    CHECK_EQ(L.at(0).plan().n_tok, 64);
    /* lane 1 plans next: the chunk, first, as lane 0's */
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    const StepEntry* e1 = L.entry_of(1, s);
    REQUIRE(e1 != nullptr);
    CHECK_EQ(e1->home, 0);
    CHECK_EQ(e1->slot, -1);
    CHECK_EQ(e1->ctx_len, 64);
    CHECK_EQ(e1->n_tokens, 64);
    /* lane 0 commits first (its step went first), then plans behind lane 1's chunk: its own [128, 192),
     * lane 1's [192, 200), which produces the first token */
    CHECK_OK(L.commit(0, 7));
    CHECK_EQ(L.req(0, s)->n_computed, 64);
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    e0 = L.entry_of(0, s);
    REQUIRE(e0 != nullptr);
    CHECK_EQ(e0->ctx_len, 128);
    CHECK_EQ(e0->n_tokens, 64);
    CHECK(!e0->produces_token);
    CHECK_EQ(L.req(0, s)->n_computed, 64);   /* the planning view is undone after step() */
    CHECK_OK(L.commit(1, 7));                /* lane 1's [64, 128) */
    CHECK_EQ(L.req(0, s)->n_computed, 128);
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    e1 = L.entry_of(1, s);
    REQUIRE(e1 != nullptr);
    CHECK_EQ(e1->ctx_len, 192);
    CHECK_EQ(e1->n_tokens, 8);
    CHECK(e1->produces_token);
    CHECK_OK(L.commit(0, 7));                /* lane 0's [128, 192) */
    CHECK_EQ(L.req(0, s)->n_computed, 192);
    CHECK(L.req(0, s)->output.empty());
    /* the home lane leaves the sequence out while its first token is to come from lane 1 */
    CHECK(L.plan(0, { 1 }) == nullptr);
    CHECK_OK(L.commit(1, 42));               /* lane 1's [192, 200): the first token */
    CHECK_EQ(L.req(0, s)->n_computed, 200);
    REQUIRE_EQ(L.req(0, s)->output.size(), (size_t)1);
    CHECK_EQ(L.req(0, s)->output[0], 42);
    /* and decodes it at home */
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    e0 = L.entry_of(0, s);
    REQUIRE(e0 != nullptr);
    CHECK(!e0->is_prefill);
    CHECK_EQ(e0->ctx_len, 200);
    CHECK_EQ(e0->n_tokens, 1);
    const LaneSet::Counts c = L.set.counts();
    CHECK_EQ(c.pieces, 2);
    CHECK_EQ(c.done, 2);
    CHECK_EQ(c.rows, 72);
    CHECK_EQ(c.token_pieces, 1);
    CHECK(L.set.broken().empty());
}

/* The engine's lane loop over cold prompts of several lengths, 1 / 2 / 3 lanes: every lane plans when it
 * has no step in flight (idle lanes after the one planning, then those in flight, oldest first), tails
 * commit in head order. The prompt's chunk plan -- every [start, end) any lane ran of it -- is one
 * lane's (one process's), and its chunks ran in position order. */
TEST(a_cold_prompts_chunks_over_lanes_are_one_process_chunks) {
    for (int n_prompt : { 40, 64, 100, 200, 333, 700 }) {
        std::vector<std::pair<int, int>> ref;
        for (int lanes : { 1, 2, 3 }) {
            Lanes L;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            std::vector<std::pair<int, int>> got;
            std::vector<int> fifo;
            std::vector<char> busy((size_t)lanes, 0);
            int next = 0;
            for (int round = 0; round < 200 && !(L.req(0, s) && !L.req(0, s)->output.empty()); ++round) {
                /* as Engine::step_loop_lanes: the lanes in turn from `next`, the idle ones after a lane that
                 * planned right behind it */
                std::vector<int> todo;
                for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
                for (size_t at = 0; at < todo.size(); ++at) {
                    const int l = todo[at];
                    if (busy[(size_t)l]) continue;
                    std::vector<int> order, idle_after;
                    for (int j = 1; j < lanes; ++j)
                        if (!busy[(size_t)((l + j) % lanes)]) {
                            order.push_back((l + j) % lanes);
                            idle_after.push_back((l + j) % lanes);
                        }
                    for (int f : fifo) order.push_back(f);
                    if (!L.plan(l, order)) continue;
                    for (const StepEntry& e : L.at(l).plan().e)
                        if (e.seq == s && e.is_prefill) got.emplace_back(e.ctx_len, e.ctx_len + e.n_tokens);
                    busy[(size_t)l] = 1;
                    fifo.push_back(l);
                    next = (l + 1) % lanes;
                    todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
                }
                if (fifo.empty()) break;
                const int l = fifo.front();
                fifo.erase(fifo.begin());
                CHECK_OK(L.commit(l, 5));
                busy[(size_t)l] = 0;
            }
            CHECK(L.set.broken().empty());
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            for (size_t i = 1; i < got.size(); ++i) CHECK_EQ(got[i].first, got[i - 1].second);   /* position order */
            if (lanes == 1) ref = got;
            else CHECK(got == ref);
            if (lanes == 2 && n_prompt > 64) CHECK(L.set.counts().pieces > 0);
        }
    }
}

/* A lane plans the chunks lent to it first, within its step: its own requests take what is left (a
 * whole 64-token chunk fills this lane's 64-token step, so its own prompt waits a step). */
TEST(a_borrower_plans_its_inbox_first_within_its_budget) {
    Lanes L;
    REQUIRE_EQ(L.start(2), RAD_OK);
    const uint64_t a = L.submit(0, 200, 2);
    const uint64_t b = L.submit(1, 200, 2, 3000);
    REQUIRE(L.plan(0, { 1 }) != nullptr);           /* a: [0, 64) here, [64, 128) to lane 1 */
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    int64_t own = 0, lent = 0;
    for (const StepEntry& e : L.at(1).plan().e) (e.home >= 0 ? lent : own) += e.n_tokens;
    CHECK_EQ(lent, 64);
    CHECK_EQ(own, 0);
    CHECK(L.at(1).plan().n_tok <= L.cfg.max_tok);
    CHECK_EQ(L.at(1).plan().n_tok, own + lent);
    CHECK(L.entry_of(1, a) != nullptr);
    CHECK(L.entry_of(1, b) == nullptr);
}

/* The engine names the lanes in the order they plan; a chunk taken out of that order (the third
 * before the second) would run out of position order. Until tp8 loop D53 the set named it (and the engine stopped); now the
 * piece waits in its lane's inbox until its predecessor is taken (lane_take_inbox), so nothing breaks and nothing runs out of
 * order; the set names it only after a 30 s wait (a predecessor that is nowhere). */
TEST(a_piece_taken_out_of_position_order_waits_for_its_predecessor) {
    Lanes L;
    REQUIRE_EQ(L.start(3), RAD_OK);
    const uint64_t s = L.submit(0, 200, 2);
    REQUIRE(s != 0);
    REQUIRE(L.plan(0, { 1, 2 }) != nullptr);        /* [0, 64) here, [64, 128) to 1, [128, 192) to 2 */
    CHECK_EQ(L.entry_of(0, s)->n_tokens, 64);
    CHECK(L.set.broken().empty());
    CHECK(L.plan(2, { 1, 0 }) == nullptr);          /* lane 2 before lane 1: its piece waits */
    CHECK(L.set.broken().empty());
    CHECK_EQ((int)L.set.inbox_n(2), 1);
    REQUIRE(L.plan(1, { 2, 0 }) != nullptr);        /* [64, 128) */
    CHECK_EQ((int)L.entry_of(1, s)->ctx_len, 64);
    REQUIRE(L.plan(2, { 0, 1 }) != nullptr);        /* then [128, 192) */
    CHECK_EQ((int)L.entry_of(2, s)->ctx_len, 128);
    CHECK(L.set.broken().empty());
    if (!L.set.broken().empty()) fprintf(stderr, "    %s\n", L.set.broken().c_str());
}

/* A request cancelled while a piece of its prompt runs in another lane keeps its blocks until that
 * lane commits the piece (in_step), then gives them back. */
TEST(a_cancel_while_a_piece_runs_elsewhere_waits_for_it) {
    Lanes L;
    REQUIRE_EQ(L.start(2), RAD_OK);
    const uint64_t s = L.submit(0, 100, 3);
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    CHECK_OK(L.commit(0, 7));
    CHECK_OK(L.at(0).cancel(s));
    CHECK(L.kv.has_sequence(s));                    /* lane 1's piece still writes its blocks */
    CHECK_OK(L.commit(1, 7));
    CHECK(!L.kv.has_sequence(s));                   /* returned once it committed */
}

/* tp8 S5 review: a lane step whose build fails after it lent its newcomer's next chunks takes them back.
 * Nothing of the step ran, so a chunk left in another lane's inbox would run [p, ...) over rows of
 * the home's own chunk that were never computed, and its commit would count rows that are not
 * there. The next plan plans and lends anew from the committed end, as one lane re-plans a failed
 * step. (Chunks on a 32-token grid, so the newcomer's chunk is the one-process chunk beside a short
 * prompt whose words fail the build.) */
TEST(lanes_a_failed_build_takes_back_the_chunks_it_lent) {
    Lanes L;
    L.grid = 32;
    L.prog.plugin_words = 4;                         /* a request's words past this fail its build */
    REQUIRE_EQ(L.start(2), RAD_OK);
    const uint64_t y = L.submit(0, 8, 3, 5000);      /* a short prompt sampled this step, with words */
    const uint64_t x = L.submit(0, 200, 3);          /* the newcomer whose next chunk is lent */
    REQUIRE(y != 0 && x != 0);
    L.at(0).find(y)->plugin_words.assign(9, 1);
    CHECK(L.plan(0, { 1 }) == nullptr);              /* the build fails: nothing of the step ran */
    CHECK_EQ((int64_t)L.set.inbox_n_locked(1), (int64_t)0);
    CHECK_EQ(L.req(0, x)->n_computed, 0);
    CHECK_EQ(L.set.counts().pieces, 0);
    /* the words fixed, lane 0 plans x's chunk again from 0 and lends the next one anew */
    L.at(0).find(y)->plugin_words.assign(2, 1);
    REQUIRE(L.plan(0, { 1 }) != nullptr);
    const StepEntry* e0 = L.entry_of(0, x);
    REQUIRE(e0 != nullptr);
    CHECK_EQ(e0->ctx_len, 0);
    REQUIRE_EQ((int64_t)L.set.inbox_n_locked(1), (int64_t)1);
    REQUIRE(L.plan(1, { 0 }) != nullptr);
    const StepEntry* e1 = L.entry_of(1, x);
    REQUIRE(e1 != nullptr);
    CHECK_EQ(e1->ctx_len, e0->ctx_len + e0->n_tokens);  /* the lent chunk starts where the home's ends */
    CHECK(L.set.broken().empty());
    CHECK_OK(L.commit(0, 7));
    CHECK_OK(L.commit(1, 7));
    CHECK_EQ(L.req(0, x)->n_computed, (int64_t)(e1->ctx_len + e1->n_tokens));
}

/* tp8 S5 review: the lanes share one KV pool but each preempts only its own requests. A lane whose
 * only running request cannot grow because another lane holds the pool must not fail it as "the KV
 * pool cannot hold it": one lane would preempt the request behind it. It is preempted by recompute
 * and waits for the other lane's requests, as the back of one lane's running set does, and finishes. */
TEST(lanes_a_request_another_lane_crowds_out_is_preempted_not_failed) {
    Lanes L;
    REQUIRE_EQ(L.start(2), RAD_OK);
    L.set.set_split(false, 16);
    L.kv.setup(L.prog.kv_groups, L.cfg.max_ctx, 12, 64);   /* 12 blocks of 16: 192 tokens for both lanes */
    const uint64_t a = L.submit(0, 90, 60);
    const uint64_t b = L.submit(1, 90, 60, 3000);
    REQUIRE(a != 0 && b != 0);
    int failed = 0;
    for (int round = 0; round < 400; ++round) {
        bool any = false;
        for (int l = 0; l < 2; ++l) {
            if (!L.plan(l, { 1 - l })) continue;
            any = true;
            CHECK_OK(L.commit(l, 7));
        }
        for (uint64_t id : { a, b }) {
            const Request* r = L.req(id == a ? 0 : 1, id);
            if (r && r->state == ReqState::Failed) ++failed;
        }
        if (!any || failed) break;
    }
    CHECK_EQ(failed, 0);
    for (uint64_t id : { a, b }) {
        const Request* r = L.req(id == a ? 0 : 1, id);
        REQUIRE(r != nullptr);
        CHECK(r->state == ReqState::Finished);
        CHECK_EQ((int64_t)r->output.size(), (int64_t)60);
        if (r->state != ReqState::Finished)
            fprintf(stderr, "    request %llu: %s, %zu tokens\n", (unsigned long long)id, r->finish_reason.c_str(),
                    r->output.size());
    }
}

/* tp8 loop D53: the same crowd-out with the preempted request overdue at once (preempt_wait_ms 0) and a state group as
 * DeepSeek's (its compressor tails and Engram token window are per-sequence state slots): the preempted request's lane has
 * nothing running while the other lane's request holds the shared pool. Before the fix the overdue path read "nothing
 * running" lane by lane -- it called release_idle_states(), which drops EVERY sequence's state slot (the other lane's running
 * request's too), and failed the request ("could not be admitted with nothing running"). Now it waits for the other lane: both
 * requests finish with all their tokens, and a running request's state slot never goes away or changes. */
TEST(lanes_an_overdue_preempted_request_waits_for_the_other_lane_and_keeps_its_state) {
    Lanes L;
    L.grid = 32;                    /* the linear state group (kv group 1) */
    L.cfg.preempt_wait_ms = 0;      /* overdue from the step after its preemption */
    REQUIRE_EQ(L.start(2), RAD_OK);
    L.set.set_split(false, 16);
    L.kv.setup(L.prog.kv_groups, L.cfg.max_ctx, 12, 64);   /* 12 blocks of 16: 192 tokens for both lanes */
    const uint64_t a = L.submit(0, 90, 60);
    const uint64_t b = L.submit(1, 90, 60, 3000);
    REQUIRE(a != 0 && b != 0);
    int failed = 0, lost = 0;
    long long preempted = 0;
    std::map<uint64_t, int32_t> slot_of;
    auto watch = [&] {
        for (uint64_t id : { a, b }) {
            const Request* r = L.req(id == a ? 0 : 1, id);
            if (!r) continue;
            if (r->state == ReqState::Failed) ++failed;
            if (r->state != ReqState::Running) { slot_of.erase(id); continue; }
            const auto it = L.kv.seqs.find(id);
            const int32_t st = it == L.kv.seqs.end() || it->second.state.size() < 2 ? -1 : it->second.state[1];
            const auto was = slot_of.find(id);
            if (was != slot_of.end() && was->second != st) ++lost;   /* dropped (-1) or moved under a running request */
            if (st >= 0) slot_of[id] = st;
        }
    };
    for (int round = 0; round < 600; ++round) {
        bool any = false;
        for (int l = 0; l < 2; ++l) {
            const RadBatch* bt = L.plan(l, { 1 - l });
            watch();
            if (!bt) continue;
            any = true;
            CHECK_OK(L.commit(l, 7));
            watch();
        }
        if (failed) break;
        const Request* ra = L.req(0, a);
        const Request* rb = L.req(1, b);
        if (!any && ra && rb && ra->state == ReqState::Finished && rb->state == ReqState::Finished) break;
    }
    preempted = (long long)(L.at(0).metrics().preemptions + L.at(1).metrics().preemptions);
    fprintf(stderr, "    %lld preemption(s), %d failed, %d running state slot(s) dropped or moved\n", preempted, failed, lost);
    CHECK(preempted > 0);
    CHECK_EQ(failed, 0);
    CHECK_EQ(lost, 0);
    for (uint64_t id : { a, b }) {
        const Request* r = L.req(id == a ? 0 : 1, id);
        REQUIRE(r != nullptr);
        CHECK(r->state == ReqState::Finished);
        CHECK_EQ((int64_t)r->output.size(), (int64_t)60);
        if (r->state != ReqState::Finished)
            fprintf(stderr, "    request %llu: %s, %zu tokens\n", (unsigned long long)id, r->finish_reason.c_str(), r->output.size());
    }
}

/* tp8 loop D53: lanes that plan in another order than the lender lent in (several lanes committed before a round planned: a
 * KV-copy check's drain, a merge drain). Lane 2 plans a 200-token cold prompt's first chunk and lends [64, 128) to lane 1 and
 * [128, 192) to lane 0 (its planners 1, 0); lane 0 then plans FIRST. Before the fix its take stopped the engine (the order guard:
 * "took the piece [128, 192) ... while the piece at 64 was not yet taken"); now the piece waits in lane 0's inbox, lane 1 runs
 * [64, 128), and lane 0's next step runs [128, 192): every piece in position order, nothing broken, all four tokens. */
TEST(lanes_a_piece_whose_predecessor_is_not_taken_waits_in_its_inbox) {
    Lanes L;
    REQUIRE_EQ(L.start(3), RAD_OK);
    L.set.set_split(true, 16);
    const uint64_t x = L.submit(2, 200, 4, 7000);
    REQUIRE(x != 0);
    REQUIRE(L.plan(2, { 1, 0 }) != nullptr);
    const StepEntry* h = L.entry_of(2, x);
    REQUIRE(h != nullptr);
    CHECK_EQ((int)h->ctx_len, 0);
    CHECK_EQ((int)L.set.inbox_n(1), 1);
    CHECK_EQ((int)L.set.inbox_n(0), 1);
    /* lane 0 plans before lane 1: its piece [128, 192) waits */
    CHECK(L.plan(0, { 1, 2 }) == nullptr);
    CHECK(L.set.broken().empty());
    CHECK_EQ((int)L.set.inbox_n(0), 1);
    CHECK_OK(L.commit(2, 7));
    REQUIRE(L.plan(1, { 0, 2 }) != nullptr);
    const StepEntry* p1 = L.entry_of(1, x);
    REQUIRE(p1 != nullptr);
    CHECK_EQ((int)p1->ctx_len, 64);
    CHECK_OK(L.commit(1, 7));
    REQUIRE(L.plan(0, { 1, 2 }) != nullptr);
    const StepEntry* p0 = L.entry_of(0, x);
    REQUIRE(p0 != nullptr);
    CHECK_EQ((int)p0->ctx_len, 128);
    CHECK_OK(L.commit(0, 7));
    CHECK(L.set.broken().empty());
    for (int round = 0; round < 40; ++round) {
        const Request* r = L.req(2, x);
        if (!r || r->state != ReqState::Running) break;
        bool any = false;
        for (int l : { 2, 1, 0 }) {
            if (!L.plan(l, {})) continue;
            any = true;
            CHECK_OK(L.commit(l, 7));
        }
        if (!any) break;
    }
    const Request* r = L.req(2, x);
    REQUIRE(r != nullptr);
    CHECK(r->state == ReqState::Finished);
    CHECK_EQ((int64_t)r->output.size(), (int64_t)4);
    CHECK(L.set.broken().empty());
    if (!L.set.broken().empty()) fprintf(stderr, "    broken: %s\n", L.set.broken().c_str());
}

/* tp8 S5 review: under a full pool the back of a lane's running set may be a newcomer whose piece
 * runs in another lane, which preempt_back refuses; the request that wanted its blocks then waits a
 * step and the plan moves past it. Before the fix the loop re-tried the same refused ensure forever
 * with the step lock held (the engine hung). Pool sizes are swept so some step meets that state; every
 * request finishes, none fails, the pieces commit in position order. */
TEST(lanes_a_refused_preemption_moves_the_plan_on) {
    int met = 0;
    for (int64_t pool = 14; pool <= 36; ++pool) {   /* x alone needs 13 blocks, all three 33 */
        Lanes L;
        L.grid = 32;                                 /* x's chunk then rides beside a's decode row */
        REQUIRE_EQ(L.start(2), RAD_OK);
        L.set.set_split(false, 16);
        L.kv.setup(L.prog.kv_groups, L.cfg.max_ctx, pool, 64);
        const uint64_t a = L.submit(0, 32, 120);
        const uint64_t b = L.submit(1, 32, 120, 3000);
        uint64_t x = 0;
        bool bad = false;
        /* the engine's order (engine_lanes.cpp): every idle lane plans (its planners: the idle lanes
         * after it, then those in flight, oldest first), then the oldest lane in flight commits */
        std::deque<int> fifo;
        bool busy[2] = { false, false };
        for (int round = 0; round < 4000 && !bad; ++round) {
            if (round == 6) {
                x = L.submit(0, 200, 4, 7000);
                L.set.set_split(true, 16);
            }
            for (int l = 0; l < 2; ++l) {
                if (busy[l]) continue;
                std::vector<int> order;
                if (!busy[1 - l]) order.push_back(1 - l);
                for (int f : fifo) order.push_back(f);
                if (!L.plan(l, order)) continue;
                busy[l] = true;
                fifo.push_back(l);
                /* a request the plan stepped past: running with blocks, not in this step */
                if (l == 0 && x) {
                    const Request* ra = L.req(0, a);
                    if (ra && ra->state == ReqState::Running && !L.entry_of(0, a)) ++met;
                }
            }
            if (fifo.empty()) {
                if (round > 6) break;
                continue;
            }
            const int l = fifo.front();
            fifo.pop_front();
            CHECK_OK(L.commit(l, 7));
            busy[l] = false;
            for (uint64_t id : { a, b, x }) {
                if (!id) continue;
                const Request* r = L.req(id == b ? 1 : 0, id);
                if (r && r->state == ReqState::Failed) bad = true;
            }
        }
        CHECK(!bad);
        CHECK(L.set.broken().empty());
        for (uint64_t id : { a, b, x }) {
            const Request* r = L.req(id == b ? 1 : 0, id);
            REQUIRE(r != nullptr);
            CHECK(r->state == ReqState::Finished);
        }
    }
    fprintf(stderr, "    %d step(s) planned past a request whose preemption was refused\n", met);
}

/* tp8 S5 review: a step carrying another lane's chunk is a shared step, and a step wider than the
 * shared budget must be one sequence (RadBuildCtx::max_mixed_tok). With three or more lanes two
 * lenders may lend into one borrower before it plans; the room check bounds the borrower's inbox of
 * chunks that are not wide by the shared budget, not by max_tok, so its step never holds more than
 * shared_tok rows of several sequences. (Two lanes: one lender a borrower: never reached.) */
TEST(lanes_an_inbox_stays_within_the_shared_budget) {
    Lanes L;
    L.cfg.prefill_span = 2;
    L.cfg.checkpoint_interval = 32;                  /* shared budget max(min(64, 32), 64 - 32) = 32 */
    REQUIRE_EQ(L.start(3), RAD_OK);
    const uint64_t p = L.submit(0, 84, 2);
    const uint64_t q = L.submit(1, 84, 2, 3000);
    REQUIRE(p != 0 && q != 0);
    REQUIRE(L.plan(0, { 2 }) != nullptr);            /* [0, 64) alone at home, [64, 84) with the token to lane 2 */
    REQUIRE(L.plan(1, { 2 }) != nullptr);            /* and another lane's [64, 84) for the same borrower */
    CHECK(L.set.inbox_rows_locked(2) <= 32);
    const RadBatch* rb = L.plan(2, { 0, 1 });
    REQUIRE(rb != nullptr);
    int nseq = 0;
    for (const StepEntry& e : L.at(2).plan().e) nseq += e.n_tokens > 0;
    CHECK(L.at(2).plan().n_tok <= 32 || nseq == 1);
    if (!(L.at(2).plan().n_tok <= 32 || nseq == 1))
        fprintf(stderr, "    lane 2: %lld rows of %d sequences in a shared step\n", (long long)L.at(2).plan().n_tok, nseq);
    CHECK(L.set.broken().empty());
}

/* tp8 prefill P0 (tp8/loop FINDINGS F1): a lone cold prompt over two lanes with DeepSeek's fixed window group. The home
 * lane lends its next chunk (keep floor at the committed end), then plans the chunk after it behind the lent one. With
 * the floor left at the lend's committed end the sequence held the window group's blocks over three chunks and the
 * request failed ("the KV pool cannot hold 10240 tokens and nothing else is left to preempt": 640 of 612 blocks); the
 * floor now follows the committed end, and a home chunk that still does not fit waits for its lent chunks. Every
 * query's window keys are read through its staged row from the block the manager holds them in (a row staged from the
 * table's start was cut at its max_blocks: the last queries of a lent chunk read no keys). Pool = KVManager's
 * window_pool_cap: max_seqs x (window_blocks + 1) + one step's blocks. The chunks = one lane's. */
TEST(lanes_a_lone_cold_prompt_fits_the_window_pool_and_reads_its_window) {
    for (int64_t seqs : { 2, 3, 6 })
        for (int n_prompt : { 100, 200, 333, 700 }) {
            std::vector<std::pair<int, int>> ref;
            for (int lanes : { 1, 2, 3 }) {
                if (seqs < lanes) continue;
                Lanes L;
                L.win = 16;
                L.seqs = seqs;
                const int64_t wb = (16 + 15) / 16 + 1;
                L.win_pool = seqs * (wb + 1) + 64 / 16;
                REQUIRE_EQ(L.start(lanes), RAD_OK);
                const uint64_t s = L.submit(0, n_prompt, 2);
                REQUIRE(s != 0);
                int64_t misses = 0;
                int failed = 0;
                const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
                CHECK_EQ(failed, 0);
                CHECK_EQ(misses, (int64_t)0);
                CHECK(L.set.broken().empty());
                REQUIRE(L.req(0, s) != nullptr);
                CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
                CHECK_EQ(L.req(0, s)->output.size(), (size_t)1);
                if (lanes == 1) ref = got;
                else CHECK(got == ref);
                const LaneSet::Counts c = L.set.counts();
                if (lanes >= 2 && n_prompt > 64) CHECK(c.pieces > 0);
                if (failed || misses || (lanes > 1 && got != ref))
                    fprintf(stderr, "    %lld cards, %d tokens, %d lanes: %d failed, %lld window keys missed, %zu chunks "
                                    "(one lane %zu), %lld lent, %lld window waits\n", (long long)seqs, n_prompt, lanes,
                            failed, (long long)misses, got.size(), ref.size(), (long long)c.pieces,
                            (long long)c.why[LaneSet::kWinWait]);
            }
        }
}

/* tp8 prefill P0: a window pool too small for the home's next chunk beside its lent one (8 blocks: the lend of
 * [64, 128) fits, the next home chunk [128, 192) from the keep floor at 64 needs 9): the home chunk waits a plan for its
 * lent chunk's commit instead of failing the request, then runs; the prompt's chunks are one lane's, every window key
 * read right. */
TEST(lanes_a_home_chunk_waits_for_its_lent_chunk_when_the_window_pool_is_short) {
    std::vector<std::pair<int, int>> ref;
    for (int lanes : { 1, 2 }) {
        Lanes L;
        L.win = 16;
        L.seqs = 2;
        L.win_pool = 8;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        const uint64_t s = L.submit(0, 200, 2);
        REQUIRE(s != 0);
        int64_t misses = 0;
        int failed = 0;
        const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
        CHECK_EQ(failed, 0);
        CHECK_EQ(misses, (int64_t)0);
        REQUIRE(L.req(0, s) != nullptr);
        CHECK_EQ(L.req(0, s)->n_computed, (int64_t)200);
        if (lanes == 1) ref = got;
        else {
            CHECK(got == ref);
            const LaneSet::Counts c = L.set.counts();
            CHECK(c.pieces > 0);
            CHECK(c.why[LaneSet::kWinWait] > 0);
            fprintf(stderr, "    2 lanes, window pool 8 blocks: %lld chunk(s) lent, %lld plan(s) waited for a lent chunk's "
                            "window, %lld lend(s) refused for KV\n", (long long)c.pieces,
                    (long long)c.why[LaneSet::kWinWait], (long long)c.why[LaneSet::kKv]);
        }
    }
}

/* tp8 prefill P1 (RADIANCE_PP_SPAN=0, LaneSet::set_span; FINDINGS F3 21:38): under lanes a lone cold prompt plans the span
 * plan's pieces and cuts each at its interior interval boundaries that leave >= 128 rows a side -- every boundary of the
 * span plan stays (the 64k prompt's tail: 1536 + 626, never 512 + 114, whose 114-row last piece moved the near tie on the
 * cards), the pieces between are whole intervals -- and the chunks alternate over the lanes; with the span on the lanes
 * run one lane's span plan, as before. The DS line's geometry (interval 2048, 4160, span 2, stager 1024), both the
 * nearest and --prefix-resume exact's chunk geometry (Config::pp_plan_prefix: the cards' path), a window group. */
TEST(lanes_span_off_cuts_the_span_plan_only_at_interior_interval_boundaries) {
    for (int ex : { 0, 1 })
    for (int n_prompt : { 8500, 15777, 20000, 65650 }) {
        std::vector<std::pair<int, int>> ref_s;
        for (int arm = 0; arm < 3; ++arm) {
            /* 0: one lane (the span plan), 1: 2 lanes span off, 2: 2 lanes span on */
            const int lanes = arm == 0 ? 1 : 2;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = ex == 1;
            L.max_tok = 4160;
            L.max_ctx = 70000;
            L.kv_blocks = 8192;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            if (arm == 1) L.set.set_span(false);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            if (arm == 0) { ref_s = got; continue; }
            if (arm == 2) { CHECK(got == ref_s); continue; }
            /* the span plan with each piece cut at interior interval boundaries, >= 128 rows a side */
            std::vector<std::pair<int, int>> want;
            for (const auto& [a, b] : ref_s) {
                int at = a;
                for (int e = (at / 2048 + 1) * 2048; e < b; e += 2048)
                    if (e - at >= 128 && b - e >= 128) { want.emplace_back(at, e); at = e; }
                want.emplace_back(at, b);
            }
            CHECK(got == want);
            CHECK(L.set.counts().pieces > 0);
            std::string t;
            for (size_t k = got.size() >= 3 ? got.size() - 3 : 0; k < got.size(); ++k)
                t += fmt(" [%d, %d)", got[k].first, got[k].second);
            fprintf(stderr, "    %s%d tokens: span plan %zu chunks, span off at 2 lanes %zu (%s), tail%s, %lld lent\n",
                    ex ? "exact, " : "", n_prompt, ref_s.size(), got.size(), got == want ? "= the cut span plan" : "DIFFERENT",
                    t.c_str(), (long long)L.set.counts().pieces);
        }
    }
}

/* tp8 loop P7 (RADIANCE_PP_SPAN_TAIL=X, LaneSet::set_span_tail): the span plan for the body and RADIANCE_PP_SPAN=0's cut only
 * for a span piece that starts within the prompt's last X rows -- every boundary one of the span plan's or an interval
 * boundary (the two plans the cards ran identical, lq18 / lq19), at 2 and 3 lanes, the chunks over the lanes, the window
 * keys read right; X past the prompt = RADIANCE_PP_SPAN=0's plan, X = 0 = the span plan. */
TEST(lanes_span_tail_cuts_only_the_span_pieces_in_a_prompts_last_rows) {
    for (int ex : { 0, 1 })
    for (int n_prompt : { 8500, 15777, 65650 }) {
        std::vector<std::pair<int, int>> ref_s;
        for (int arm = 0; arm < 5; ++arm) {
            /* 0: one lane (the span plan); 1 / 2: 2 / 3 lanes, X = 8192; 3: 2 lanes, X = 1 << 20 (all pieces); 4: 2 lanes, X = 0 */
            const int lanes = arm == 0 ? 1 : arm == 2 ? 3 : 2;
            const int64_t X = arm == 1 || arm == 2 ? 8192 : arm == 3 ? (1 << 20) : 0;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = ex == 1;
            L.max_tok = 4160;
            L.max_ctx = 70000;
            L.kv_blocks = 8192;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_span_tail(X);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            if (arm == 0) { ref_s = got; continue; }
            if (arm == 4) { CHECK(got == ref_s); continue; }
            /* the span plan, each piece that starts within the last X rows cut at its interior interval boundaries */
            std::vector<std::pair<int, int>> want;
            for (const auto& [a, b] : ref_s) {
                int at = a;
                if (n_prompt - a <= X)
                    for (int e = (at / 2048 + 1) * 2048; e < b; e += 2048)
                        if (e - at >= 128 && b - e >= 128) { want.emplace_back(at, e); at = e; }
                want.emplace_back(at, b);
            }
            CHECK(got == want);
            for (const auto& [a, b] : got) {        /* every boundary the span plan's or an interval boundary */
                bool span_b = false;
                for (const auto& [c, d] : ref_s) span_b = span_b || d == b;
                CHECK(span_b || b % 2048 == 0);
            }
            CHECK(L.set.counts().pieces > 0);
            std::string t;
            for (size_t k = got.size() >= 4 ? got.size() - 4 : 0; k < got.size(); ++k)
                t += fmt(" [%d, %d)", got[k].first, got[k].second);
            fprintf(stderr, "    %s%d tokens, %d lanes, X %lld: span plan %zu chunks, span tail %zu (%s), tail%s, %lld lent\n",
                    ex ? "exact, " : "", n_prompt, lanes, (long long)X, ref_s.size(), got.size(),
                    got == want ? "= the span plan cut in its last X rows" : "DIFFERENT", t.c_str(),
                    (long long)L.set.counts().pieces);
        }
    }
}

/* tp8 loop P7 (RADIANCE_PP_SPAN_SHORT=N, LaneSet::set_span_short): a prompt of at most N rows plans RADIANCE_PP_SPAN=0's cut
 * span plan (interval chunks), a longer one the span plan (or SPAN_TAIL's), at 2 and 3 lanes, the window keys read right. */
TEST(lanes_span_short_cuts_a_short_prompts_span_pieces_and_leaves_a_long_ones) {
    for (int ex : { 0, 1 })
    for (int n_prompt : { 15777, 65650 })
    for (int lanes : { 2, 3 }) {
        std::vector<std::pair<int, int>> ref_s, ref_cut;
        for (int arm = 0; arm < 3; ++arm) {
            /* 0: one lane (the span plan); 1: `lanes` lanes, SPAN=0 (the cut plan); 2: `lanes` lanes, SPAN_SHORT 24576 */
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = ex == 1;
            L.max_tok = 4160;
            L.max_ctx = 70000;
            L.kv_blocks = 8192;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            const int nl = arm == 0 ? 1 : lanes;
            REQUIRE_EQ(L.start(nl), RAD_OK);
            if (arm == 1) L.set.set_span(false);
            if (arm == 2) L.set.set_span_short(24576);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            const std::vector<std::pair<int, int>> got = run_cold(L, nl, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            if (arm == 0) { ref_s = got; continue; }
            if (arm == 1) { ref_cut = got; continue; }
            CHECK(got == (n_prompt <= 24576 ? ref_cut : ref_s));
            fprintf(stderr, "    %s%d tokens, %d lanes, SPAN_SHORT 24576: %zu chunks (%s)\n", ex ? "exact, " : "", n_prompt, lanes,
                    got.size(), got == (n_prompt <= 24576 ? ref_cut : ref_s) ? (n_prompt <= 24576 ? "= the cut plan" : "= the span plan")
                                                                         : "DIFFERENT");
        }
    }
}

/* tp8 loop P34 (RADIANCE_PP_SPAN_FIRST=F, LaneSet::set_span_first): with SPAN_SHORT 24576 (BEST's start) a cold prompt of
 * more than two intervals and at most 24,576 rows plans the cut plan with its first interval as [0, F) + [F, 2048) -- every
 * other piece the cut plan's to the row, the window keys read right (misses 0), every row computed -- at 2 and 3 lanes,
 * exact plan and nearest; a prompt of two intervals or less (a decode cell's newcomer) and one past SPAN_SHORT keep their
 * plan; F off the quantum (1000) or under kBiMinPiece (64) changes nothing. */
TEST(lanes_span_first_cuts_a_cold_prompts_first_interval_and_nothing_else) {
    for (int cards : { 0, 1 })   /* 1: BEST's start besides: SPAN_TAIL 8192, lever G 256 + TI 1 */
    for (int ex : { 0, 1 })
    for (int n_prompt : { 15777, 3000, 4500, 65650 })
    for (int lanes : { 2, 3 }) {
        std::vector<std::pair<int, int>> ref;
        for (int arm = 0; arm < 4; ++arm) {
            /* 0: SPAN_SHORT 24576 (the reference); 1: + SPAN_FIRST 1024; 2: + SPAN_FIRST 1000 (off the quantum);
             * 3: + SPAN_FIRST 64 (< kBiMinPiece) */
            std::unique_ptr<TgSwitch> tg;
            std::unique_ptr<TiSwitch> ti;
            if (cards) {
                tg = std::make_unique<TgSwitch>(256);
                ti = std::make_unique<TiSwitch>(1);
            }
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = ex == 1;
            L.max_tok = 4160;
            L.max_ctx = 70000;
            L.kv_blocks = 8192;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_span_short(24576);
            if (cards) {
                L.set.set_span_tail(8192);
                for (int l = 0; l < lanes; ++l) L.at(l).set_tail_inner_capable(true);
            }
            if (arm == 1) L.set.set_span_first(1024);
            if (arm == 2) L.set.set_span_first(1000);
            if (arm == 3) L.set.set_span_first(64);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            if (arm == 0) { ref = got; continue; }
            std::vector<std::pair<int, int>> want = ref;
            const bool cut = arm == 1 && n_prompt > 2 * 2048 && n_prompt <= 24576;
            if (cut) {
                REQUIRE(!want.empty());
                CHECK(want.front() == std::make_pair(0, 2048));
                want.front() = { 0, 1024 };
                want.insert(want.begin() + 1, { 1024, 2048 });
            }
            CHECK(got == want);
            if (arm == 1)
                fprintf(stderr, "    %s%s%d tokens, %d lanes, SPAN_SHORT 24576 + SPAN_FIRST 1024: %zu chunks, first %d-%d%s (%s)\n",
                        cards ? "SPAN_TAIL 8192 + G 256 + TI, " : "", ex ? "exact, " : "", n_prompt, lanes, got.size(), got.empty() ? -1 : got.front().first,
                        got.empty() ? -1 : got.front().second, got.size() > 1 ? (" then " + std::to_string(got[1].first) + "-" +
                        std::to_string(got[1].second)).c_str() : "", got == want ? (cut ? "= the cut plan, its first interval in two"
                                                                                        : "= the cut plan") : "DIFFERENT");
        }
    }
}

/* tp8 loop P10 / FINDINGS F14 (the ~22 ms every pass costs on both nodes): --prefill-span 4 with --max-num-batched-tokens
 * 8256 under lanes -- the cold prompt's body in 8,192-row span pieces (one interval first, then 4 intervals a piece), at 2 /
 * 3 lanes the one-lane plan to the row, every boundary on the interval grid or the prompt's end, the window keys read
 * right, the pieces lent; with SPAN_TAIL 8192 the last 8,192 rows in intervals; SPAN_SHORT 24576 gives 15,777 the cut
 * plan. Exact geometry (the cards'): 16k [0, 2048) [2048, 10240) [10240, 15777), 64k 9 chunks (7 of 8,192) ending [59392, 65650),
 * 128k 17 (15 of 8,192); so span 4 needs SPAN_TAIL / SHORT, or its last chunk runs 5.5-6.3k rows into the drain. (A
 * setting for the cards: whether an 8,192-row pass is answer-neutral is the cards' text check.) */
TEST(lanes_span_four_plans_eight_k_bodies_as_one_lane_does) {
    for (int ex : { 0, 1 })
    for (int n_prompt : { 15777, 65650, 131137 }) {
        std::vector<std::pair<int, int>> ref_s;
        for (int arm = 0; arm < 7; ++arm) {
            /* 0: one lane; 1 / 2: 2 / 3 lanes; 3: 2 lanes + SPAN_TAIL 8192 + SPAN_SHORT 24576; 4: 3 + SPAN_INNER (tp8 loop P14);
             * 5: 3 + the cards' lever G 512 + TI 1 (the last chunk carries its tail snapshot inside: lq41's end); 6: 5 + INNER */
            std::unique_ptr<TgSwitch> tg;
            std::unique_ptr<TiSwitch> ti;
            if (arm >= 5) {
                tg = std::make_unique<TgSwitch>(512);
                ti = std::make_unique<TiSwitch>(1);
            }
            const int lanes = arm == 0 ? 1 : arm == 2 ? 3 : 2;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 4;
            L.cfg.pp_plan_prefix = ex == 1;
            L.max_tok = 8256;
            L.max_ctx = 140000;
            L.kv_blocks = 16384;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            if (arm >= 3) { L.set.set_span_tail(8192); L.set.set_span_short(24576); }
            if (arm == 4 || arm == 6) L.set.set_span_inner(true);
            if (arm >= 5)
                for (int l = 0; l < lanes; ++l) L.at(l).set_tail_inner_capable(true);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            const std::vector<std::pair<int, int>> got = run_cold(L, lanes, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            int big = 0;
            for (const auto& [a, b] : got) {
                /* the cards' geometry (exact prefix resume): interval boundaries or the end, >= 128 rows; nearest's tail
                 * snapshot a block short of the end cuts a short last piece there at any span (one lane's plan too) */
                if (ex) CHECK(b % 2048 == 0 || b == n_prompt);
                if (ex) CHECK(b - a >= 128);
                CHECK(b - a <= 8192);
                big += b - a == 8192;
            }
            if (arm == 0) {
                ref_s = got;
                if (n_prompt > 20000) CHECK(big > 0);
            } else if (arm < 3) {
                CHECK(got == ref_s);
                CHECK(L.set.counts().pieces > 0);
            } else {
                /* SPAN_TAIL 8192: no piece starting in the last 8,192 rows spans an interval boundary; SHORT: 15,777 all intervals */
                for (const auto& [a, b] : got)
                    if (ex && (n_prompt - a <= 8192 || n_prompt <= 24576)) CHECK(b - a <= 2048 + 128 || b == n_prompt);
                /* + SPAN_INNER (tp8 loop P14): the last chunk too, cut before its tail snapshot -- it is the interval plan's
                 * last chunk, from the last interval boundary at least 128 rows short of the end (1,441 / 2,162 / 2,113),
                 * and the body stays in 8,192-row spans */
                if ((arm == 4 || arm == 6) && ex) {
                    REQUIRE(!got.empty());
                    const int last_iv = (n_prompt - 128) / 2048 * 2048;
                    CHECK_EQ(got.back().first, last_iv);
                    CHECK_EQ(got.back().second, n_prompt);
                    for (const auto& [a, b] : got)
                        if (n_prompt - a <= 8192 || n_prompt <= 24576) CHECK(b - a <= 2048 + 128);
                    if (n_prompt > 24576) CHECK(big > 0);
                }
            }
            std::string t;
            for (size_t k = 0; k < got.size() && k < 3; ++k) t += fmt(" [%d, %d)", got[k].first, got[k].second);
            t += " ..";
            for (size_t k = got.size() >= 3 ? got.size() - 3 : 0; k < got.size(); ++k) t += fmt(" [%d, %d)", got[k].first, got[k].second);
            fprintf(stderr, "    %sspan 4, %d tokens, %d lanes%s: %zu chunks (%d of 8192 rows)%s, %lld lent:%s\n", ex ? "exact, " : "",
                    n_prompt, lanes, arm == 6 ? " + SPAN_TAIL 8192 + SHORT 24576 + G 512 + TI + INNER" : arm == 5 ? " + SPAN_TAIL 8192 + SHORT 24576 + G 512 + TI" : arm == 4 ? " + SPAN_TAIL 8192 + SHORT 24576 + INNER" : arm == 3 ? " + SPAN_TAIL 8192 + SHORT 24576" : "", got.size(), big,
                    arm == 1 || arm == 2 ? (got == ref_s ? " = one lane's" : " DIFFERENT") : "", (long long)L.set.counts().pieces,
                    t.c_str());
        }
    }
}

/* tp8 loop P14 (RADIANCE_PP_SPAN_INNER=1): at the published --prefill-span 2 with the BEST start's settings (SPAN_TAIL 8192,
 * SPAN_SHORT 24576, the cards' lever G 512 + TI 1) a cold prompt's last chunk spans one interval + its tail, which holds no
 * interior interval boundary >= 128 rows from the end: the plan with INNER is the plan without it, at 2 and 3 lanes, for
 * the cards' 16k / 64k / 128k and a prompt whose last chunk does span two intervals (66,536: cut before its snapshot). */
TEST(lanes_span_inner_leaves_the_span_two_cold_plans_and_cuts_only_a_two_interval_last_chunk) {
    TgSwitch tg(512);
    TiSwitch ti(1);
    for (int lanes : { 2, 3 })
    for (int n_prompt : { 15777, 65650, 131137, 66536 }) {
        std::vector<std::pair<int, int>> plan[2];
        for (int inner = 0; inner < 2; ++inner) {
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = true;
            L.max_tok = 4160;
            L.max_ctx = 140000;
            L.kv_blocks = 16384;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_span_tail(8192);
            L.set.set_span_short(24576);
            L.set.set_span_inner(inner == 1);
            for (int l = 0; l < lanes; ++l) L.at(l).set_tail_inner_capable(true);
            const uint64_t s = L.submit(0, n_prompt, 2);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            plan[inner] = run_cold(L, lanes, s, &misses, &failed);
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, (int64_t)n_prompt);
            for (const auto& [a, b] : plan[inner]) {
                CHECK(b % 2048 == 0 || b == n_prompt);
                CHECK(b - a >= 128);
            }
        }
        REQUIRE(!plan[0].empty() && !plan[1].empty());
        if (n_prompt != 66536) {
            CHECK(plan[1] == plan[0]);
        } else {
            /* the last chunk [63488, 66536) carries the tail snapshot: INNER cuts it at 65536 */
            CHECK_EQ(plan[0].back().first, 63488);
            CHECK_EQ(plan[1].back().first, 65536);
            using Plan = std::vector<std::pair<int, int>>;
            const Plan head1(plan[1].begin(), plan[1].end() - 2), head0(plan[0].begin(), plan[0].end() - 1);
            CHECK(head1 == head0);
        }
        fprintf(stderr, "    span 2, %d tokens, %d lanes, G 512 + TI + SPAN_TAIL / SHORT: last chunk [%d, %d) -> with INNER "
                        "[%d, %d)%s\n", n_prompt, lanes, plan[0].back().first, plan[0].back().second, plan[1].back().first,
                plan[1].back().second, plan[1] == plan[0] ? " (the same plan)" : "");
    }
}

/* tp8 loop P21 (fn4 sched_tail_grid 256 / 128 beside sched_tail_inner 1 and lever PSK-CR's tail-only anchors, ds41_psk 4): the
 * BEST start's cold plans at 2 and 3 lanes (span 2, SPAN_TAIL 8192, SPAN_SHORT 24576) for the three cold cells. The chunk
 * plan is F 512's (the published grid's) to the row; lever G's fine tail -- the deepest multiple of F at or below N - 128 --
 * sits inside the last chunk (one snapshot there, TI) and is every chunk's next PSK target, so each chunk's `ahead` is that
 * tail less its end. deepseek41 then runs layer l over [tail - C_l, end) of each chunk (C_l = 160 + (39 - l) x 128,
 * psk_rows), so F 256's cone of the drain is smaller on both nodes: the stake printed below in row-layers (follower 20-31,
 * leader 32-39 at W(10,32)). */
TEST(lanes_a_finer_tail_grid_keeps_the_cold_plans_and_shrinks_the_drains_cone) {
    const auto C = [](int64_t l) { return (int64_t)160 + (39 - l) * 128; };
    const auto psk_rows = [](int64_t c, int64_t ahead, int64_t n) -> int64_t {
        if (ahead >= c) return 0;
        return std::min(n, std::max(c - ahead, (int64_t)160));
    };
    struct Chunk { int64_t a, b, ahead; int ck; };
    for (int lanes : { 2, 3 })
    for (int64_t n_prompt : { (int64_t)15777, (int64_t)65650, (int64_t)131137 }) {
        std::vector<Chunk> got[3];
        int64_t fol[3] = { 0, 0, 0 }, lead[3] = { 0, 0, 0 };
        for (int gi = 0; gi < 3; ++gi) {
            const int F = gi == 0 ? 512 : gi == 1 ? 256 : 128;
            TgSwitch tg(F);
            TiSwitch ti(1);
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.cfg.pp_plan_prefix = true;
            L.max_tok = 4160;
            L.max_ctx = 140000;
            L.kv_blocks = 16384;
            L.stage_rows = 1024;
            L.win = 16;
            L.seqs = 6;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_span_tail(8192);
            L.set.set_span_short(24576);
            for (int l = 0; l < lanes; ++l) {
                L.at(l).set_tail_inner_capable(true);
                L.at(l).set_psk_mode(4);
                L.at(l).set_psk_cr(1, 5152);
                L.at(l).set_psk_k(8);
            }
            const uint64_t s = L.submit(0, (int)n_prompt, 2);
            REQUIRE(s != 0);
            std::vector<int> fifo;
            std::vector<char> busy((size_t)lanes, 0);
            int next = 0, failed = 0;
            for (int round = 0; round < 400; ++round) {
                const Request* r0 = L.req(0, s);
                if (!r0 || r0->state == ReqState::Failed || r0->state == ReqState::Finished || !r0->output.empty()) break;
                std::vector<int> todo;
                for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
                for (size_t at = 0; at < todo.size(); ++at) {
                    const int l = todo[at];
                    if (busy[(size_t)l]) continue;
                    std::vector<int> order, idle_after;
                    for (int j = 1; j < lanes; ++j)
                        if (!busy[(size_t)((l + j) % lanes)]) {
                            order.push_back((l + j) % lanes);
                            idle_after.push_back((l + j) % lanes);
                        }
                    for (int f : fifo) order.push_back(f);
                    const RadBatch* b = L.plan(l, order);
                    if (!b) continue;
                    std::vector<int32_t> ahead, rows;
                    int64_t dec = 0;
                    const bool planned = L.at(l).psk_plan(b->step, &ahead, &rows, &dec);
                    size_t k = 0;
                    for (const StepEntry& e : L.at(l).plan().e) {
                        if (!e.is_prefill) continue;
                        if (e.seq == s)
                            got[gi].push_back({ e.ctx_len, e.ctx_len + e.n_tokens,
                                                planned && k < ahead.size() ? (int64_t)ahead[k] : (int64_t)Scheduler::kPskNoPlan,
                                                e.ckpt_tok });
                        ++k;
                    }
                    busy[(size_t)l] = 1;
                    fifo.push_back(l);
                    next = (l + 1) % lanes;
                    todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
                }
                if (fifo.empty()) break;
                const int l = fifo.front();
                fifo.erase(fifo.begin());
                if (L.commit(l, 5) < 0) ++failed;
                busy[(size_t)l] = 0;
            }
            CHECK_EQ(failed, 0);
            REQUIRE(L.req(0, s) != nullptr);
            CHECK_EQ(L.req(0, s)->n_computed, n_prompt);
            REQUIRE(!got[gi].empty());
            std::sort(got[gi].begin(), got[gi].end(), [](const Chunk& x, const Chunk& y) { return x.a < y.a; });
            const int64_t tail = (n_prompt - 128) / F * F;
            const Chunk& last = got[gi].back();
            CHECK_EQ(last.b, n_prompt);
            CHECK(last.a < tail);
            for (const Chunk& c : got[gi]) {
                CHECK(c.ahead != (int64_t)Scheduler::kPskNoPlan);
                CHECK_EQ(c.ahead, tail - c.b);   /* the fine tail is every chunk's next target, inside the last */
                for (int64_t l = 20; l < 40; ++l) {
                    const int64_t r = psk_rows(C(l), c.ahead, c.b - c.a);
                    (l < 32 ? fol : lead)[gi] += r;
                }
            }
            /* lever TI: the snapshot inside the last chunk, the state after its row tail - 1 */
            CHECK_EQ(last.a + (int64_t)last.ck + 1, tail);
        }
        bool same = true;
        for (int gi = 1; gi < 3; ++gi) {
            REQUIRE_EQ(got[0].size(), got[gi].size());
            for (size_t i = 0; i < got[0].size(); ++i) same = same && got[0][i].a == got[gi][i].a && got[0][i].b == got[gi][i].b;
        }
        CHECK(same);
        CHECK(fol[1] < fol[0]);
        CHECK(lead[1] < lead[0]);
        CHECK(fol[2] <= fol[1]);   /* 16k: 128's tail is 256's (15616) */
        CHECK(lead[2] <= lead[1]);
        fprintf(stderr, "    %lld tokens, %d lanes: %zu chunks, the plan %s; fine tail F 512 / 256 / 128 %lld / %lld / %lld; cone "
                        "row-layers follower 20-31 %lld / %lld / %lld, leader 32-39 %lld / %lld / %lld\n", (long long)n_prompt, lanes,
                got[0].size(), same ? "identical" : "DIFFERENT", (long long)((n_prompt - 128) / 512 * 512),
                (long long)((n_prompt - 128) / 256 * 256), (long long)((n_prompt - 128) / 128 * 128), (long long)fol[0],
                (long long)fol[1], (long long)fol[2], (long long)lead[0], (long long)lead[1], (long long)lead[2]);
    }
}

/* tp8 loop M2 (the 20:58 PROFILE: no lane step for 320-430 ms after every MIXED step): lever DST under lanes. A newcomer
 * that joins a lane beside decode rows is cut by option D on sched_mixed_tok's grid (576 -> 512-row pieces), one piece a
 * home-lane step, the same boundaries one lane plans (the plan is a function of the switches, not of the decode rows
 * beside it); the other lane plans a decode step with every one of its decoders every round meanwhile, so on the cards its
 * passes run between the newcomer's pieces instead of behind one whole-prompt MIXED pass. D's pieces are not the alone
 * chunk: never lent (counted off the one-process grid). */
TEST(lanes_dst_cuts_a_newcomer_beside_decode_rows_on_its_grid_while_the_other_lane_decodes) {
    std::vector<int64_t> ref;
    for (int lanes : { 1, 2 }) {
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.stage_rows = 1024;
        L.seqs = 12;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        L.set.set_lend_busy(false);                  /* RADIANCE_PP_LEND_BUSY=0 */
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        CHECK(DstSwitches::set("sched_mixed_tok", 576));
        std::vector<uint64_t> dec[2];
        for (int i = 0; i < 8; ++i) {
            const int l = lanes == 1 ? 0 : i % 2;
            dec[l].push_back(L.submit(l, 100, 4000, 2000 + 100 * i));
        }
        uint64_t p = 0;
        std::vector<int64_t> ends;
        int other_steps = 0, other_short = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        for (int round = 0; round < 3000; ++round) {
            if (round == 4) p = L.submit(0, 20000, 4);
            const Request* rp = p ? L.req(0, p) : nullptr;
            if (rp && rp->n_computed >= 20000) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                std::vector<int> order;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) order.push_back((l + j) % lanes);
                for (int f : fifo) order.push_back(f);
                if (!L.plan(l, order)) continue;
                for (const StepEntry& e : L.at(l).plan().e)
                    if (p && e.seq == p && e.is_prefill) ends.push_back((int64_t)e.ctx_len + e.n_tokens);
                if (l == 1 && p && rp && rp->n_computed < 20000) {
                    ++other_steps;
                    int rows = 0;
                    for (const StepEntry& e : L.at(1).plan().e)
                        for (uint64_t d : dec[1]) rows += e.seq == d && !e.is_prefill;
                    other_short += rows != (int)dec[1].size();
                }
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
            }
            if (fifo.empty()) break;
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
        }
        REQUIRE(p != 0);
        REQUIRE(L.req(0, p) != nullptr);
        CHECK_EQ(L.req(0, p)->n_computed, (int64_t)20000);
        int off = 0;
        for (size_t i = 0; i < ends.size(); ++i) {
            const int64_t n = ends[i] - (i ? ends[i - 1] : 0);
            /* every end on the 512 grid; the last piece whole where a cut would leave < 128 rows (D's rule) */
            off += ends[i] == 20000 ? !(n < 512 + 128) : !(n <= 512 && ends[i] % 512 == 0);
        }
        if (off) {
            std::string t;
            for (size_t i = 0; i < ends.size(); ++i) t += fmt(" %lld", (long long)ends[i]);
            fprintf(stderr, "    %d lane(s) ends:%s\n", lanes, t.c_str());
        }
        CHECK_EQ(off, 0);
        CHECK(ends.size() >= 20000 / 512);
        if (lanes == 1) ref = ends;
        else {
            CHECK(ends == ref);
            CHECK(other_steps >= (int)ends.size() - 1);   /* lane 1 decodes every round the newcomer's pieces run */
            CHECK_EQ(other_short, 0);
            CHECK_EQ(L.set.counts().pieces, 0);
            fprintf(stderr, "    2 lanes, sched_mixed_tok 576: the 20000-token newcomer in %zu pieces (one lane's %zu), lane 1 "
                            "planned %d decode steps meanwhile with all %zu of its decoders each, %lld off-grid (not lent)\n",
                    ends.size(), ref.size(), other_steps, dec[1].size(),
                    (long long)L.set.counts().why[LaneSet::kOffGrid]);
        }
    }
}

/* tp8 loop F71 (RADIANCE_PP_PLAN_RESERVED=1): a lane holding another lane's lent 2048-row piece (lane_reserve set it aside) plans its own
 * newcomers -- 1,200 rows whole by option D's over-budget first-prefill rule, then 1,000 whole by lever NOA's room -- and both rules read
 * the step's room as max_tok less the lane's OWN plan, so own (2,204) + the piece (2,048) = 4,252 went past max_tok 4,160 (lq187 / lq191 /
 * lq192's 'E lane N: 2048 rows of other lanes' pieces (1) beside M of its own ... step budget 4160'). Unset: named, the build refuses it,
 * the lane re-plans the same plan -- with no BG lane to take the newcomers it makes no step at all. With the env the inbox rows count:
 * NOA's newcomer waits a step, every step stays within max_tok, nothing is named, every prompt completes. Planning only (which step a
 * newcomer rides), never a pass's arithmetic. */
TEST(lanes_an_inbox_piece_counts_against_an_over_budget_first_prefill) {
    int named_off = -1, named_on = -1;
    for (int on : { 0, 1 }) {
        if (on) setenv("RADIANCE_PP_PLAN_RESERVED", "1", 1);
        else unsetenv("RADIANCE_PP_PLAN_RESERVED");
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.seqs = 12;
        REQUIRE_EQ(L.start(2), RAD_OK);
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        CHECK(DstSwitches::set("sched_noa", 2));
        std::vector<uint64_t> dec;
        for (int i = 0; i < 4; ++i) dec.push_back(L.submit(1, 100, 4000, 2000 + 100 * i));
        for (int k = 0; k < 3; ++k) {   /* lane 1's decoders prefill and decode */
            REQUIRE(L.plan(1, { 0 }) != nullptr);
            CHECK_OK(L.commit(1, 5));
        }
        int over = 0, refused = 0, steps = 0, lent_seen = 0;
        int64_t most = 0;
        uint64_t big = 0, q = 0, q2 = 0;
        const std::string err = stderr_of([&]() {
            big = L.submit(0, 4096, 4, 7000);
            REQUIRE(L.plan(0, { 1 }) != nullptr);     /* [0, 2048) at home, [2048, 4096) lent to lane 1 */
            CHECK_OK(L.commit(0, 5));
            q = L.submit(1, 1200, 4, 15000);          /* under the 2048 grid: D plans it whole over the budget (the step's first prefill) */
            q2 = L.submit(1, 1000, 4, 25000);         /* and lever NOA this one whole in the max_tok room left */
            for (int round = 0; round < 400; ++round) {
                const Request* rb = L.req(0, big);
                const Request* rq = L.req(1, q);
                const Request* rq2 = L.req(1, q2);
                if ((!rb || rb->n_computed >= 4096) && (!rq || rq->n_computed >= 1200) && (!rq2 || rq2->n_computed >= 1000)) break;
                for (int l : { 1, 0 }) {
                    const int64_t inbox = L.set.inbox_rows_locked(l);
                    const RadBatch* b = L.plan(l, { 1 - l });
                    if (!b) { ++refused; continue; }
                    ++steps;
                    const int64_t n = L.at(l).plan().n_tok;
                    most = std::max(most, n);
                    if (n > L.max_tok) ++over;
                    if (inbox > 0) ++lent_seen;
                    CHECK_OK(L.commit(l, 5));
                }
            }
        });
        int named = 0;
        for (size_t at = err.find("rows of other lanes' pieces"); at != std::string::npos; at = err.find("rows of other lanes' pieces", at + 1)) ++named;
        (on ? named_on : named_off) = named;
        const Request* rb = L.req(0, big);
        const Request* rq = L.req(1, q);
        fprintf(stderr, "    RADIANCE_PP_PLAN_RESERVED=%s: %d steps, %d plans refused, %d over max_tok (most %lld rows), %d steps with a lent piece, %d "
                        "'rows of other lanes' pieces' lines; big %lld / 4096, newcomers %lld / 1200, %lld / 1000 computed\n", on ? "1" : "unset", steps, refused,
                over, (long long)most, lent_seen, named, rb ? (long long)rb->n_computed : -1LL, rq ? (long long)rq->n_computed : -1LL,
                L.req(1, q2) ? (long long)L.req(1, q2)->n_computed : -1LL);
        CHECK_EQ(over, 0);   /* a plan over max_tok is never built (BatchBuilder refuses it) */
        if (!on) {
            /* as before: the lane's plan (4 decoders + 1,200 by D's over-budget rule + 1,000 by NOA = 2,204 of its own) beside the
             * 2,048 set aside = 4,252 > 4,160 -- named, refused, re-planned the same, refused again: with no BG lane to take the
             * newcomers (lq187 / lq192 had one) the lane makes no step */
            CHECK(named > 0);
            CHECK(refused > 0);
            continue;
        }
        CHECK(lent_seen > 0);
        CHECK(rb && rb->n_computed >= 4096);
        CHECK(rq && rq->n_computed >= 1200);
        CHECK(L.req(1, q2) && L.req(1, q2)->n_computed >= 1000);
        CHECK_EQ(named, 0);
        CHECK_EQ(refused, 0);
        CHECK(most <= L.max_tok);
    }
    CHECK(named_off > 0);   /* the case the env fixes is there without it */
    CHECK_EQ(named_on, 0);
    unsetenv("RADIANCE_PP_PLAN_RESERVED");
}

/* tp8 dev M5 (RADIANCE_PP_LEND_MIXED=1): a newcomer that option D cuts beside decode rows (sched_bi_chunk 3, lever DST's
 * sched_mixed_tok 576 -> the 512 grid; and the 2048 grid at the published 2112) lends the pieces one process cuts after its
 * home step's piece to the other lane, whose decoders keep running beside it: the newcomer's piece ends are one lane's to
 * the row (the one-process plan), pieces are lent at two lanes, and both lanes' decoders are in every step. */
TEST(lanes_lend_mixed_lends_a_newcomers_option_d_pieces_and_keeps_its_one_lane_ends) {
    struct Arm { int mixed; int64_t len; bool head; };
    /* `head`: a cut W(h > 0, t), whose leader head part writes layers [0, h)'s state (LaneSet::head_state) */
    for (const Arm a : { Arm{ 576, 1100, false }, Arm{ 576, 2266, false }, Arm{ 576, 5000, false }, Arm{ 2112, 5000, false },
                         Arm{ 576, 1100, true }, Arm{ 576, 5000, true } }) {
        std::vector<int64_t> ref;
        for (int lanes : { 1, 2, 3 }) {
            DstSwitches restore;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.max_tok = 4160;
            L.max_ctx = 60000;
            L.stage_rows = 1024;
            L.seqs = 12;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_lend_mixed(true);
            L.set.set_head_state(a.head);
            CHECK(DstSwitches::set("sched_bi_chunk", 3));
            CHECK(DstSwitches::set("sched_mixed_tok", a.mixed));
            std::vector<uint64_t> dec[3];
            for (int i = 0; i < 8; ++i) {
                const int l = lanes == 1 ? 0 : i % lanes;
                dec[l].push_back(L.submit(l, 100, 4000, 2000 + 100 * i));
            }
            uint64_t p = 0;
            std::vector<int64_t> ends;
            int short_steps = 0, lent_in_1 = 0;
            std::vector<int> fifo;
            std::vector<char> busy((size_t)lanes, 0);
            int next = 0;
            for (int round = 0; round < 3000; ++round) {
                if (round == 4) p = L.submit(0, a.len, 4);
                const Request* rp = p ? L.req(0, p) : nullptr;
                if (rp && rp->n_computed >= a.len) break;
                std::vector<int> todo;
                for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
                for (size_t at = 0; at < todo.size(); ++at) {
                    const int l = todo[at];
                    if (busy[(size_t)l]) continue;
                    std::vector<int> order;
                    for (int j = 1; j < lanes; ++j)
                        if (!busy[(size_t)((l + j) % lanes)]) order.push_back((l + j) % lanes);
                    for (int f : fifo) order.push_back(f);
                    if (!L.plan(l, order)) continue;
                    for (const StepEntry& e : L.at(l).plan().e)
                        if (p && e.seq == p && e.is_prefill) {
                            ends.push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (l != 0 && e.home == 0) ++lent_in_1;
                        }
                    if (lanes >= 2 && p && rp && rp->n_computed < a.len) {
                        int rows = 0;
                        for (const StepEntry& e : L.at(l).plan().e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                    busy[(size_t)l] = 1;
                    fifo.push_back(l);
                    next = (l + 1) % lanes;
                }
                if (fifo.empty()) break;
                const int l = fifo.front();
                fifo.erase(fifo.begin());
                CHECK_OK(L.commit(l, 5));
                busy[(size_t)l] = 0;
            }
            REQUIRE(p != 0);
            REQUIRE(L.req(0, p) != nullptr);
            CHECK_EQ(L.req(0, p)->n_computed, a.len);
            std::string t;
            for (size_t i = 0; i < ends.size(); ++i) t += fmt(" %lld", (long long)ends[i]);
            if (lanes == 1) {
                ref = ends;
                fprintf(stderr, "    mixed %d, a %lld-token newcomer%s, 1 lane: ends%s\n", a.mixed, (long long)a.len,
                        a.head ? " (head state)" : "", t.c_str());
            } else {
                /* 2 lanes: one lane's ends to the row. 3 lanes: every end one of one lane's (a long newcomer's alone-piece
                 * boundary can be reached by the cold-prompt lend of an idle engine's chunk, which then skips a D cut inside
                 * it: that path's, with RADIANCE_PP_LEND_MIXED off too) */
                bool sub = !ends.empty() && ends.back() == ref.back();
                for (int64_t x : ends) sub = sub && std::find(ref.begin(), ref.end(), x) != ref.end();
                if (lanes == 2 ? ends != ref : !sub) fprintf(stderr, "    %d lanes ends:%s\n", lanes, t.c_str());
                CHECK(lanes == 2 ? ends == ref : sub);
                CHECK(L.set.counts().pieces > 0);
                CHECK(L.set.counts().dpieces > 0);
                CHECK(lent_in_1 > 0);
                CHECK_EQ(short_steps, 0);
                fprintf(stderr, "    mixed %d, a %lld-token newcomer, %d lanes: %zu ends of one lane's; %lld pieces lent (%d carried by other "
                                "lanes, %lld from option-D steps), every step with all its lane's decoders: %s\n", a.mixed,
                        (long long)a.len, lanes, ends.size(), (long long)L.set.counts().pieces, lent_in_1,
                        (long long)L.set.counts().dpieces, short_steps == 0 ? "yes" : "NO");
            }
        }
    }
}

/* tp8 dev M5b (RADIANCE_PP_LEND_MIXED=2): the published admission (sched_mixed_tok 2112: a ~1.1k newcomer planned whole beside
 * decode rows) and, at two / three lanes, the home's entry cut at the 512-grid point nearest its middle (>= 128 rows a side) with
 * the rest lent to one free lane: every piece end on the 512 grid or the prompt's end, the newcomer admitted in the same round
 * as one lane admits it, one piece a borrower, both lanes' decoders in every step; with a head-state cut too. */
TEST(lanes_lend_split_cuts_a_whole_mixed_newcomer_on_the_512_grid_and_lends_its_rest) {
    struct Arm { int64_t len; bool head; };
    for (const Arm a : { Arm{ 1100, false }, Arm{ 2266, false }, Arm{ 700, false }, Arm{ 1100, true } }) {
        int first_round_1 = -1;
        for (int lanes : { 1, 2, 3 }) {
            DstSwitches restore;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.max_tok = 4160;
            L.max_ctx = 60000;
            L.stage_rows = 1024;
            L.seqs = 12;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_lend_split(lanes > 1);
            L.set.set_head_state(a.head);
            CHECK(DstSwitches::set("sched_bi_chunk", 3));
            std::vector<uint64_t> dec[3];
            for (int i = 0; i < 8; ++i) {
                const int l = lanes == 1 ? 0 : i % lanes;
                dec[l].push_back(L.submit(l, 100, 4000, 2000 + 100 * i));
            }
            uint64_t p = 0;
            std::vector<int64_t> ends;
            int short_steps = 0, first_round = -1, max_inbox_pieces = 0;
            std::vector<int> fifo;
            std::vector<char> busy((size_t)lanes, 0);
            int next = 0;
            for (int round = 0; round < 3000; ++round) {
                if (round == 4) p = L.submit(0, a.len, 4);
                const Request* rp = p ? L.req(0, p) : nullptr;
                if (rp && rp->n_computed >= a.len) break;
                std::vector<int> todo;
                for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
                for (size_t at = 0; at < todo.size(); ++at) {
                    const int l = todo[at];
                    if (busy[(size_t)l]) continue;
                    std::vector<int> order;
                    for (int j = 1; j < lanes; ++j)
                        if (!busy[(size_t)((l + j) % lanes)]) order.push_back((l + j) % lanes);
                    for (int f : fifo) order.push_back(f);
                    if (!L.plan(l, order)) continue;
                    int pieces = 0;
                    for (const StepEntry& e : L.at(l).plan().e)
                        if (p && e.seq == p && e.is_prefill) {
                            ends.push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (first_round < 0) first_round = round;
                            if (e.home >= 0) ++pieces;
                        }
                    max_inbox_pieces = std::max(max_inbox_pieces, pieces);
                    if (p && rp && rp->n_computed < a.len) {
                        int rows = 0;
                        for (const StepEntry& e : L.at(l).plan().e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                    busy[(size_t)l] = 1;
                    fifo.push_back(l);
                    next = (l + 1) % lanes;
                }
                if (fifo.empty()) break;
                const int l = fifo.front();
                fifo.erase(fifo.begin());
                CHECK_OK(L.commit(l, 5));
                busy[(size_t)l] = 0;
            }
            REQUIRE(p != 0);
            REQUIRE(L.req(0, p) != nullptr);
            CHECK_EQ(L.req(0, p)->n_computed, a.len);
            std::string t;
            for (size_t i = 0; i < ends.size(); ++i) t += fmt(" %lld", (long long)ends[i]);
            int off = 0;
            for (int64_t x : ends) off += x != a.len && x % 512 != 0;
            CHECK_EQ(off, 0);
            if (lanes == 1) {
                first_round_1 = first_round;
                CHECK_EQ(L.set.counts().dpieces, 0ll);
            } else {
                /* admitted whole at the published budget, as one lane admits it (the round its lane is free); a newcomer of
                 * more than one D piece is the existing lend's (2266: D cuts at 2048, the cold-prompt path lends the rest) */
                CHECK(first_round >= first_round_1 && first_round <= first_round_1 + 1);
                CHECK(max_inbox_pieces <= 1);
                if (a.len <= 2048) CHECK(L.set.counts().dpieces > 0);
                if (a.len <= 2048) CHECK_EQ((int)ends.size(), 2);
            }
            CHECK_EQ(short_steps, 0);
            fprintf(stderr, "    lend split, a %lld-token newcomer%s, %d lane(s): ends%s; admitted round %d; %lld split, %lld pieces lent; "
                            "decoders in every step: %s\n", (long long)a.len, a.head ? " (head state)" : "", lanes, t.c_str(),
                    first_round, (long long)L.set.counts().dpieces, (long long)L.set.counts().pieces, short_steps ? "NO" : "yes");
        }
    }
}

/* tp8 dev M8 (RADIANCE_PP_BG_LANE=1): the last lane holds no sequence (pick() never names it) and a newcomer planned whole beside
 * decode rows goes to it WHOLE: its chunk is the home's own plan (one entry, the prompt's end, the same rows as one lane plans),
 * the home's step goes out as a DECODE step with all its decoders, the BG lane's step is a lone PREFILL step of other lanes'
 * pieces, and the newcomer decodes in its home lane afterwards. A newcomer of more than one D piece keeps D's first piece home
 * (2266: 2048 there, the 218-row rest to the BG lane or by the cold path to a decode lane, never the BG lane's by that path).
 * DS geometry (interval 2048, MNBT 4160, span 2), 1 / 2 / 3 lanes, with a head-state cut too; the engine's planning order
 * (each lane's idle followers planned right behind it). */
TEST(lanes_bg_lane_takes_whole_mixed_newcomers_and_holds_no_sequence) {
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); unsetenv("RADIANCE_PP_BG_SPLIT"); }
    } env;
    struct Arm { int64_t len; bool head; };
    /* 3 = two BG lanes with RADIANCE_PP_BG_SPLIT=1 (tp8 dev D12's one-BG-lane form, RADIANCE_PP_BG_PIECES, was removed 18.11.) */
    for (const int k : { 1, 2, 3 })
    for (const Arm a : { Arm{ 1100, false }, Arm{ 700, false }, Arm{ 2266, false }, Arm{ 1100, true } }) {
        const bool split = k == 3;
        setenv("RADIANCE_PP_BG_LANE", k == 1 ? "1" : "2", 1);
        if (split) setenv("RADIANCE_PP_BG_SPLIT", "1", 1); else unsetenv("RADIANCE_PP_BG_SPLIT");
        std::vector<int64_t> ends_1;
        const int nbg = split ? 2 : k;
        for (int lanes : { 1, nbg + 1, nbg + 2 }) {
            DstSwitches restore;
            Lanes L;
            L.grid = 2048;
            L.cfg.prefill_span = 2;
            L.max_tok = 4160;
            L.max_ctx = 60000;
            L.stage_rows = 1024;
            L.seqs = 12;
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            L.set.set_head_state(a.head);
            CHECK(DstSwitches::set("sched_bi_chunk", 3));
            const int bg = L.set.bg_lane(), nd = lanes - L.set.bg_n();
            CHECK_EQ(bg, lanes > nbg ? lanes - nbg : -1);
            std::vector<int> p_lanes;   /* the BG lanes that ran a piece of the newcomer, in run order */
            for (int q = 0; q < 8; ++q) CHECK(!L.set.is_bg(L.set.pick()));
            std::vector<uint64_t> dec[3];
            for (int i = 0; i < 8; ++i) {
                const int l = i % nd;
                dec[l].push_back(L.submit(l, 100, 4000, 2000 + 100 * i));
            }
            uint64_t p = 0;
            std::vector<int64_t> ends;
            int short_steps = 0, bg_steps = 0, bg_not_prefill = 0, home_mixed_with_p = 0, p_decoded = 0, bg_own = 0;
            std::vector<int> fifo;
            std::vector<char> busy((size_t)lanes, 0);
            int next = 0;
            for (int round = 0; round < 3000; ++round) {
                if (round == 4) p = L.submit(0, a.len, 4);
                const Request* rp = p ? L.req(0, p) : nullptr;
                if (rp && rp->n_computed >= a.len && !rp->output.empty() && rp->output.size() >= 3) break;
                std::vector<int> todo;
                for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
                for (size_t at = 0; at < todo.size(); ++at) {
                    const int l = todo[at];
                    if (busy[(size_t)l]) continue;
                    std::vector<int> order, idle_after;
                    for (int j = 1; j < lanes; ++j)
                        if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                    for (int f : fifo) order.push_back(f);
                    if (!L.plan(l, order)) continue;
                    const StepPlan& pl = L.at(l).plan();
                    for (const StepEntry& e : pl.e) {
                        if (p && e.seq == p && e.is_prefill) {
                            ends.push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (L.set.is_bg(l)) p_lanes.push_back(l);
                            if (!L.set.is_bg(l) && pl.phase == RAD_PHASE_MIXED && e.produces_token) ++home_mixed_with_p;
                        }
                        if (p && e.seq == p && !e.is_prefill && l == 0) ++p_decoded;
                        if (L.set.is_bg(l) && e.home < 0) ++bg_own;
                    }
                    if (L.set.is_bg(l)) {
                        ++bg_steps;
                        bg_not_prefill += pl.phase != RAD_PHASE_PREFILL;
                    } else if (p && rp && rp->n_computed < a.len) {
                        int rows = 0;
                        for (const StepEntry& e : pl.e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                    busy[(size_t)l] = 1;
                    fifo.push_back(l);
                    next = (l + 1) % lanes;
                    todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
                }
                if (fifo.empty()) break;
                const int l = fifo.front();
                fifo.erase(fifo.begin());
                CHECK_OK(L.commit(l, 5));
                busy[(size_t)l] = 0;
            }
            REQUIRE(p != 0);
            REQUIRE(L.req(0, p) != nullptr);
            CHECK(L.req(0, p)->n_computed >= a.len);   /* its prompt, then the decoded tokens */
            CHECK(p_decoded > 0);
            std::string t;
            for (size_t i = 0; i < ends.size(); ++i) t += fmt(" %lld", (long long)ends[i]);
            const LaneSet::Counts c = L.set.counts();
            if (lanes == 1) {
                ends_1 = ends;
                CHECK_EQ(c.bg, 0ll);
            } else {
                /* the BG lane's steps are lone prefill steps of lent pieces only; its own requests: none */
                CHECK_EQ(bg_not_prefill, 0);
                CHECK_EQ(bg_own, 0);
                /* (2266 beside a second decode lane: the cold path lends its 218-row rest to it in D's 2048 step) */
                if (a.len <= 2048 || nd == 1) {
                    CHECK(bg_steps > 0);
                    CHECK(c.bg > 0);
                }
                /* a one-piece newcomer: the one-lane plan's chunk, run by the BG lane; never a MIXED home step with its token */
                if (a.len <= 2048 && !split) {
                    CHECK(ends == ends_1);
                    CHECK_EQ(home_mixed_with_p, 0);
                }
                /* RADIANCE_PP_BG_SPLIT: 512 + the rest, in two BG lanes, the first piece's lane first */
                if (a.len <= 2048 && split) {
                    CHECK(ends == std::vector<int64_t>({ 512, a.len }));
                    CHECK_EQ((int)p_lanes.size(), 2);
                    if (p_lanes.size() == 2) CHECK(p_lanes[0] != p_lanes[1]);
                    CHECK(c.bg_split > 0);
                    CHECK_EQ(home_mixed_with_p, 0);
                }
                CHECK(L.set.broken().empty());
            }
            CHECK_EQ(short_steps, 0);
            fprintf(stderr, "    BG lanes %d%s, a %lld-token newcomer%s, %d lane(s): ends%s (one lane:%s); %lld lent to BG lanes "
                            "(%lld rows), BG steps %d; home MIXED steps with its token %d; decoders in every step: %s; decoded %d\n",
                    nbg, split ? " + split" : "", (long long)a.len, a.head ? " (head state)" : "", lanes, t.c_str(),
                    [&] { std::string s; for (int64_t x : ends_1) s += fmt(" %lld", (long long)x); return s; }().c_str(),
                    (long long)c.bg, (long long)c.bg_rows, bg_steps, home_mixed_with_p, short_steps ? "NO" : "yes", p_decoded);
        }
    }
}

/* tp8 dev D11 (RADIANCE_PP_BG_HOLD_MS=N with RADIANCE_PP_BG_LANE=1): two newcomers planned beside decode rows in one round (lane
 * 0's and lane 1's), the BG lane's room (the shared budget 2112 of the DS geometry) for one of them. Without the hold the second
 * runs whole as a MIXED step of its home (today, M8's 'kept home: the BG lanes' room'); with the hold it is left out of its
 * home's steps -- which go out with all their decoders -- until the BG lane takes it whole; with a hold that runs out it runs at
 * home as before. In every arm each newcomer runs as ONE chunk, its whole prompt (the home's own plan: no new cut), and decodes
 * afterwards. 3 lanes (2 decode + BG), 4 decoders a decode lane, the engine's planning order. */
TEST(lanes_bg_hold_waits_for_the_bg_lane_instead_of_a_home_mixed_step) {
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); unsetenv("RADIANCE_PP_BG_HOLD_MS"); }
    } env;
    struct Arm { const char* hold; bool expire; };
    for (const Arm a : { Arm{ nullptr, false }, Arm{ "60000", false }, Arm{ "1", true } }) {
        setenv("RADIANCE_PP_BG_LANE", "1", 1);
        if (a.hold) setenv("RADIANCE_PP_BG_HOLD_MS", a.hold, 1); else unsetenv("RADIANCE_PP_BG_HOLD_MS");
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.stage_rows = 1024;
        L.seqs = 12;
        const int lanes = 3;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        REQUIRE_EQ(L.set.bg_lane(), 2);
        CHECK_EQ(L.set.bg_hold_ms(), a.hold ? (int64_t)std::atoll(a.hold) : 0ll);
        std::vector<uint64_t> dec[2];
        for (int i = 0; i < 8; ++i) dec[i % 2].push_back(L.submit(i % 2, 100, 4000, 2000 + 100 * i));
        /* newcomer 0 (lane 0) keeps the BG lane busy; 1 (lane 0) waits in its inbox; 2 (lane 1) finds no room beside 1 */
        constexpr int NC = 3;
        const int64_t len[NC] = { 1000, 1100, 1050 };
        const int home[NC] = { 0, 0, 1 }, at_round[NC] = { 4, 6, 6 };
        uint64_t nc[NC] = {};
        std::vector<int64_t> ends[NC];
        int home_mixed[NC] = {}, on_bg[NC] = {}, decoded[NC] = {}, short_steps = 0, held_rounds = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        for (int round = 0; round < 3000; ++round) {
            for (int j = 0; j < NC; ++j)
                if (round == at_round[j]) nc[j] = L.submit(home[j], (int)len[j], 4, 50000 + 2000 * j);
            bool done = true;
            for (int j = 0; j < NC && done; ++j) {
                const Request* r = nc[j] ? L.req(home[j], nc[j]) : nullptr;
                done = r && r->n_computed >= len[j] && r->output.size() >= 3;
            }
            if (done) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                /* the expiring arm: the BG lane does not take its inbox while a newcomer is held, so the hold runs out */
                if (a.expire && L.set.is_bg(l) && held_rounds > 0 && L.set.counts().hold_home == 0) continue;
                std::vector<int> order, idle_after;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                for (int f : fifo) order.push_back(f);
                const int64_t held0 = L.set.counts().hold_steps;
                if (!L.plan(l, order)) continue;
                if (L.set.counts().hold_steps > held0) ++held_rounds;
                const StepPlan& pl = L.at(l).plan();
                for (const StepEntry& e : pl.e)
                    for (int j = 0; j < NC; ++j) {
                        if (!nc[j] || e.seq != nc[j]) continue;
                        if (e.is_prefill) {
                            ends[j].push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (L.set.is_bg(l)) ++on_bg[j];
                            else if (pl.phase == RAD_PHASE_MIXED && e.produces_token) ++home_mixed[j];
                        } else if (!L.set.is_bg(l)) {
                            ++decoded[j];
                        }
                    }
                /* a decode lane's step while a newcomer is still prefilling carries all of that lane's decoders */
                if (!L.set.is_bg(l) && l < 2) {
                    bool pf = false;
                    for (int j = 0; j < NC; ++j) {
                        const Request* r = nc[j] && home[j] == l ? L.req(home[j], nc[j]) : nullptr;
                        pf = pf || (r && r->n_computed < len[j]);
                    }
                    if (pf) {
                        int rows = 0;
                        for (const StepEntry& e : pl.e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                }
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
                todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
            }
            if (fifo.empty()) break;
            /* the expiring arm: the hold's 1 ms runs out between two rounds */
            if (a.expire && held_rounds > 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
        }
        const LaneSet::Counts c = L.set.counts();
        for (int j = 0; j < NC; ++j) {
            REQUIRE(nc[j] != 0);
            REQUIRE(L.req(home[j], nc[j]) != nullptr);
            CHECK(L.req(home[j], nc[j])->n_computed >= len[j]);
            CHECK(decoded[j] > 0);
            /* one chunk, its whole prompt: the home's own plan, wherever it ran */
            CHECK(ends[j] == std::vector<int64_t>({ len[j] }));
            CHECK_EQ(on_bg[j] + home_mixed[j], 1);
        }
        CHECK(L.set.broken().empty());
        const int kept = home_mixed[0] + home_mixed[1] + home_mixed[2];
        if (!a.hold) {
            /* the room path is taken: one newcomer kept home as a MIXED step, nothing held */
            CHECK_EQ(kept, 1);
            CHECK(c.bg_skip[LaneSet::kBgRoom] > 0);
            CHECK_EQ(c.hold_reqs, 0ll);
        } else if (!a.expire) {
            /* held, then lent whole: both newcomers ran on the BG lane, every decode step with all its decoders */
            CHECK_EQ(kept, 0);
            CHECK_EQ(on_bg[0] + on_bg[1] + on_bg[2], NC);
            CHECK(c.hold_reqs >= 1);
            CHECK(c.hold_steps >= 1);
            CHECK(c.hold_lent >= 1);
            CHECK_EQ(c.hold_home, 0ll);
            CHECK_EQ(short_steps, 0);
        } else {
            /* the hold ran out: today's MIXED step at home */
            CHECK_EQ(kept, 1);
            CHECK(c.hold_reqs >= 1);
            CHECK(c.hold_steps >= 1);
            CHECK(c.hold_home >= 1);
            CHECK_EQ(c.hold_lent + c.hold_home, c.hold_reqs);
        }
        CHECK_EQ(short_steps, 0);
        fprintf(stderr, "    BG hold %s%s: ends %lld | %lld | %lld, on the BG lane %d / %d / %d, home MIXED %d / %d / %d, short decode "
                        "steps %d; held %lld newcomers over %lld steps, %lld lent, %lld home after the hold, kept home for room %lld\n",
                a.hold ? a.hold : "off", a.expire ? " (runs out)" : "", ends[0].empty() ? -1ll : (long long)ends[0].back(),
                ends[1].empty() ? -1ll : (long long)ends[1].back(), ends[2].empty() ? -1ll : (long long)ends[2].back(),
                on_bg[0], on_bg[1], on_bg[2], home_mixed[0], home_mixed[1], home_mixed[2], short_steps, (long long)c.hold_reqs, (long long)c.hold_steps, (long long)c.hold_lent, (long long)c.hold_home,
                (long long)c.bg_skip[LaneSet::kBgRoom]);
    }
}

/* tp8 loop P23b (RADIANCE_PP_BG_JOIN_MS=N with BG_ROOM): the BG lane's step in flight is predicted to commit at its plan time + its
 * rows x the BG steps' running us a row (plan to commit); no step in flight -> 0, no measurement yet -> -1 (joins). Then the gate in
 * the engine's planning order: a 500-row warm-up newcomer whose BG step takes ~30 ms to commit (the test sleeps there) gives ~60 us
 * a row; newcomer 1 (1000 rows) keeps the BG lane busy (predicted ~60 ms), of 2 / 3 (planned in one round) the first lent waits in its
 * inbox (within the shared room), the other is past it: JOIN_MS 1 sends that one home (today's MIXED step, kept 1, one 'ran at home' count), JOIN_MS 60000 lets it join (1 + 2 in
 * one BG step of 2150 rows, one 'joined' count); every newcomer one chunk = its whole prompt, all decoders every step. */
TEST(lanes_bg_join_ms_sends_a_newcomer_home_when_the_bg_step_in_flight_is_long) {
    {   /* the prediction alone (a started 3-lane set; nothing submitted) */
        Lanes L0;
        L0.grid = 2048;
        L0.cfg.prefill_span = 2;
        L0.max_tok = 4160;
        L0.max_ctx = 60000;
        L0.seqs = 12;
        REQUIRE_EQ(L0.start(3), RAD_OK);
        LaneSet& s = L0.set;
        CHECK_EQ(s.bg_remaining_us_locked(2, 1000), 0ll);                 /* nothing in flight */
        s.note_bg_take_locked(2, 1000, 1000000);
        CHECK_EQ(s.bg_remaining_us_locked(2, 1100000), -1ll);             /* no measurement yet */
        s.note_bg_commit_locked(2, 1500000);                               /* 500 ms for 1000 rows: 500 us a row */
        CHECK_EQ(s.bg_remaining_us_locked(2, 1600000), 0ll);              /* committed: none in flight */
        s.note_bg_take_locked(2, 2000, 2000000);
        CHECK_EQ(s.bg_remaining_us_locked(2, 2100000), 900000ll);         /* 2000 x 500 us - 100 ms */
        CHECK_EQ(s.bg_remaining_us_locked(2, 4000000), 0ll);              /* overdue: 0, not negative */
        s.note_bg_commit_locked(2, 2400000);                               /* 200 us a row: mean 0.8 x 500 + 0.2 x 200 = 440 */
        s.note_bg_take_locked(2, 1000, 3000000);
        CHECK_EQ(s.bg_remaining_us_locked(2, 3000000), 440000ll);
        CHECK_EQ(s.counts().bg_us_row_x100, 44000ll);
    }
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); unsetenv("RADIANCE_PP_BG_ROOM"); unsetenv("RADIANCE_PP_BG_JOIN_MS"); }
    } env;
    for (const char* join : { "1", "60000" }) {
        setenv("RADIANCE_PP_BG_LANE", "1", 1);
        setenv("RADIANCE_PP_BG_ROOM", "4160", 1);
        setenv("RADIANCE_PP_BG_JOIN_MS", join, 1);
        const bool joins = std::atoll(join) >= 1000;
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.stage_rows = 1024;
        L.seqs = 12;
        const int lanes = 3;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        REQUIRE_EQ(L.set.bg_lane(), 2);
        CHECK_EQ(L.set.bg_join_ms(), (int64_t)std::atoll(join));
        std::vector<uint64_t> dec[2];
        for (int i = 0; i < 8; ++i) dec[i % 2].push_back(L.submit(i % 2, 100, 4000, 2000 + 100 * i));
        constexpr int NC = 4;   /* 0 = the warm-up */
        const int64_t len[NC] = { 500, 1000, 1100, 1050 };
        /* the warm-up at round 1; 1 / 2 / 3 at rounds r0 / r0 + 2 / r0 + 2, r0 = the round after the warm-up's BG step committed
         * (D11's 4 / 6 / 6 timeline from there) */
        const int home[NC] = { 1, 0, 0, 1 };
        int at_round[NC] = { 1, 1 << 30, 1 << 30, 1 << 30 };
        uint64_t nc[NC] = {};
        std::vector<int64_t> ends[NC];
        int home_mixed[NC] = {}, on_bg[NC] = {}, decoded[NC] = {}, short_steps = 0, bg_two = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        for (int round = 0; round < 3000; ++round) {
            if (at_round[1] == (1 << 30) && L.set.counts().bg_us_row_x100 > 0) {
                at_round[1] = round;
                at_round[2] = at_round[3] = round + 2;
            }
            for (int j = 0; j < NC; ++j)
                if (round == at_round[j]) nc[j] = L.submit(home[j], (int)len[j], 4, 50000 + 2000 * j);
            bool done = true;
            for (int j = 0; j < NC && done; ++j) {
                const Request* r = nc[j] ? L.req(home[j], nc[j]) : nullptr;
                done = r && r->n_computed >= len[j] && r->output.size() >= 3;
            }
            if (done) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                std::vector<int> order, idle_after;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                for (int f : fifo) order.push_back(f);
                if (!L.plan(l, order)) continue;
                const StepPlan& pl = L.at(l).plan();
                int newc_here = 0;
                for (const StepEntry& e : pl.e)
                    for (int j = 0; j < NC; ++j) {
                        if (!nc[j] || e.seq != nc[j]) continue;
                        if (e.is_prefill) {
                            ends[j].push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (L.set.is_bg(l)) { ++on_bg[j]; ++newc_here; }
                            else if (pl.phase == RAD_PHASE_MIXED && e.produces_token) ++home_mixed[j];
                        } else if (!L.set.is_bg(l)) {
                            ++decoded[j];
                        }
                    }
                if (L.set.is_bg(l)) bg_two += newc_here >= 2;
                if (!L.set.is_bg(l)) {
                    bool pf = false;
                    for (int j = 0; j < NC; ++j) {
                        const Request* r = nc[j] && home[j] == l ? L.req(home[j], nc[j]) : nullptr;
                        pf = pf || (r && r->n_computed < len[j]);
                    }
                    if (pf) {
                        int rows = 0;
                        for (const StepEntry& e : pl.e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                }
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
                todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
            }
            if (fifo.empty()) break;
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            /* the warm-up's BG step takes ~30 ms from its plan to its commit: the BG steps' measured us a row */
            if (L.set.is_bg(l) && L.set.counts().bgst_n == 1 && L.set.counts().bg_us_row_x100 == 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
        }
        const LaneSet::Counts c = L.set.counts();
        for (int j = 0; j < NC; ++j) {
            REQUIRE(nc[j] != 0);
            REQUIRE(L.req(home[j], nc[j]) != nullptr);
            CHECK(L.req(home[j], nc[j])->n_computed >= len[j]);
            CHECK(decoded[j] > 0);
            CHECK(ends[j] == std::vector<int64_t>({ len[j] }));
            CHECK_EQ(on_bg[j] + home_mixed[j], 1);
        }
        CHECK(L.set.broken().empty());
        CHECK(c.bg_us_row_x100 > 0);
        CHECK_EQ(on_bg[0], 1);   /* the warm-up ran on the BG lane */
        const int kept = home_mixed[1] + home_mixed[2] + home_mixed[3];
        if (!joins) {
            CHECK_EQ(kept, 1);
            CHECK_EQ(home_mixed[2] + home_mixed[3], 1);   /* the one of the two planned together that was past the shared room */
            CHECK_EQ(bg_two, 0);
            CHECK(c.bg_join_no >= 1);
            CHECK_EQ(c.bg_join_yes, 0ll);
        } else {
            CHECK_EQ(kept, 0);
            CHECK_EQ(bg_two, 1);
            CHECK(c.bg_join_yes >= 1);
            CHECK_EQ(c.bg_join_no, 0ll);
        }
        CHECK_EQ(short_steps, 0);
        fprintf(stderr, "    BG_ROOM 4160 + JOIN_MS %s: on the BG lane %d / %d / %d / %d, home MIXED %d / %d / %d; BG steps of two %d; joined %lld, "
                        "sent home %lld; BG steps %.2f us a row\n", join, on_bg[0], on_bg[1], on_bg[2], on_bg[3], home_mixed[1],
                home_mixed[2], home_mixed[3], bg_two, (long long)c.bg_join_yes, (long long)c.bg_join_no,
                (double)c.bg_us_row_x100 / 100.0);
    }
}

/* tp8 loop P23 (RADIANCE_PP_BG_ROOM=N with RADIANCE_PP_BG_LANE=1): D11's three newcomers (1000 / 1100 / 1050 rows; 0 keeps the BG
 * lane busy, 1 and 2 are planned beside decode rows in one round while it runs). Without the switch the BG lane's room (the
 * shared budget 2112 of the DS geometry) takes 1 and keeps 2 home as a MIXED step (today); with BG_ROOM=4160 (and 3000) both go to
 * the BG lane's inbox and its next step carries them TOGETHER (2150 rows > the shared 2112, two sequences: the step lever NOA
 * plans on one machine); 1100 + 1050 > 2112 is what BG_ROOM=2112 still refuses. Every newcomer runs as ONE chunk, its whole
 * prompt (the home's own plan, no new cut), and decodes in its home afterwards; every decode step carries all its decoders.
 * 3 lanes (2 decode + BG), 4 decoders a decode lane, the engine's planning order. */
TEST(lanes_bg_room_lets_the_bg_lane_take_two_newcomers_in_one_step) {
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); unsetenv("RADIANCE_PP_BG_ROOM"); }
    } env;
    for (const char* room : { (const char*)nullptr, "4160", "3000", "2112", "100" }) {
        setenv("RADIANCE_PP_BG_LANE", "1", 1);
        if (room) setenv("RADIANCE_PP_BG_ROOM", room, 1); else unsetenv("RADIANCE_PP_BG_ROOM");
        /* 100 is out of range (256 .. 65536): not taken, today's room */
        const bool wide = room && std::atoll(room) >= 2150;
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.stage_rows = 1024;
        L.seqs = 12;
        const int lanes = 3;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        REQUIRE_EQ(L.set.bg_lane(), 2);
        CHECK_EQ(L.set.bg_room(), room && std::atoll(room) >= 256 ? (int64_t)std::atoll(room) : 0ll);
        std::vector<uint64_t> dec[2];
        for (int i = 0; i < 8; ++i) dec[i % 2].push_back(L.submit(i % 2, 100, 4000, 2000 + 100 * i));
        constexpr int NC = 3;
        const int64_t len[NC] = { 1000, 1100, 1050 };
        const int home[NC] = { 0, 0, 1 }, at_round[NC] = { 4, 6, 6 };
        uint64_t nc[NC] = {};
        std::vector<int64_t> ends[NC];
        int home_mixed[NC] = {}, on_bg[NC] = {}, decoded[NC] = {}, short_steps = 0, bg_two = 0;
        int64_t bg_rows_max = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        for (int round = 0; round < 3000; ++round) {
            for (int j = 0; j < NC; ++j)
                if (round == at_round[j]) nc[j] = L.submit(home[j], (int)len[j], 4, 50000 + 2000 * j);
            bool done = true;
            for (int j = 0; j < NC && done; ++j) {
                const Request* r = nc[j] ? L.req(home[j], nc[j]) : nullptr;
                done = r && r->n_computed >= len[j] && r->output.size() >= 3;
            }
            if (done) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                std::vector<int> order, idle_after;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                for (int f : fifo) order.push_back(f);
                if (!L.plan(l, order)) continue;
                const StepPlan& pl = L.at(l).plan();
                int newc_here = 0;
                for (const StepEntry& e : pl.e)
                    for (int j = 0; j < NC; ++j) {
                        if (!nc[j] || e.seq != nc[j]) continue;
                        if (e.is_prefill) {
                            ends[j].push_back((int64_t)e.ctx_len + e.n_tokens);
                            if (L.set.is_bg(l)) { ++on_bg[j]; ++newc_here; }
                            else if (pl.phase == RAD_PHASE_MIXED && e.produces_token) ++home_mixed[j];
                        } else if (!L.set.is_bg(l)) {
                            ++decoded[j];
                        }
                    }
                if (L.set.is_bg(l)) {
                    CHECK(pl.phase == RAD_PHASE_PREFILL);
                    CHECK(pl.n_tok <= L.max_tok);
                    bg_two += newc_here >= 2;
                    bg_rows_max = std::max<int64_t>(bg_rows_max, pl.n_tok);
                }
                if (!L.set.is_bg(l) && l < 2) {
                    bool pf = false;
                    for (int j = 0; j < NC; ++j) {
                        const Request* r = nc[j] && home[j] == l ? L.req(home[j], nc[j]) : nullptr;
                        pf = pf || (r && r->n_computed < len[j]);
                    }
                    if (pf) {
                        int rows = 0;
                        for (const StepEntry& e : pl.e)
                            for (uint64_t d : dec[l]) rows += e.seq == d && !e.is_prefill;
                        short_steps += rows != (int)dec[l].size();
                    }
                }
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
                todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
            }
            if (fifo.empty()) break;
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
        }
        const LaneSet::Counts c = L.set.counts();
        for (int j = 0; j < NC; ++j) {
            REQUIRE(nc[j] != 0);
            REQUIRE(L.req(home[j], nc[j]) != nullptr);
            CHECK(L.req(home[j], nc[j])->n_computed >= len[j]);
            CHECK(decoded[j] > 0);
            CHECK(ends[j] == std::vector<int64_t>({ len[j] }));   /* one chunk, its whole prompt, wherever it ran */
            CHECK_EQ(on_bg[j] + home_mixed[j], 1);
        }
        CHECK(L.set.broken().empty());
        const int kept = home_mixed[0] + home_mixed[1] + home_mixed[2];
        if (!wide) {
            /* today's room: one newcomer kept home as a MIXED step; no BG step of two */
            CHECK_EQ(kept, 1);
            CHECK(c.bg_skip[LaneSet::kBgRoom] > 0);
            CHECK_EQ(bg_two, 0);
            CHECK_EQ(c.bgst_b2, 0ll);
            CHECK(bg_rows_max <= 2112);
        } else {
            /* the room raised: all three on the BG lane, 1 and 2 in ONE BG step (2150 rows, over the shared budget) */
            CHECK_EQ(kept, 0);
            CHECK_EQ(on_bg[0] + on_bg[1] + on_bg[2], NC);
            CHECK_EQ(bg_two, 1);
            CHECK_EQ(c.bgst_b2, 1ll);
            CHECK_EQ(c.bgst_b2_newc, 2ll);
            CHECK_EQ(c.bgst_b2_rows, 2150ll);
            CHECK_EQ(bg_rows_max, 2150);
            CHECK_EQ(c.bgst_max_rows, 2150ll);
            CHECK_EQ(c.bg_skip[LaneSet::kBgRoom], 0ll);
        }
        CHECK_EQ(c.bgst_newc, (int64_t)(on_bg[0] + on_bg[1] + on_bg[2]));
        CHECK_EQ(short_steps, 0);
        fprintf(stderr, "    BG room %s: ends %lld | %lld | %lld, on the BG lane %d / %d / %d, home MIXED %d / %d / %d, short decode "
                        "steps %d; BG steps %lld with %lld newcomers, %lld with >= 2 (%lld rows), the widest %lld rows; kept home for "
                        "room %lld\n",
                room ? room : "off", ends[0].empty() ? -1ll : (long long)ends[0].back(),
                ends[1].empty() ? -1ll : (long long)ends[1].back(), ends[2].empty() ? -1ll : (long long)ends[2].back(),
                on_bg[0], on_bg[1], on_bg[2], home_mixed[0], home_mixed[1], home_mixed[2], short_steps, (long long)c.bgst_n,
                (long long)c.bgst_newc, (long long)c.bgst_b2, (long long)c.bgst_b2_rows, (long long)c.bgst_max_rows,
                (long long)c.bg_skip[LaneSet::kBgRoom]);
    }
}

/* tp8 dev D10 (RADIANCE_PP_ISSUE_AHEAD=1): a lane's step issued ahead (step_next, before its step's commit) takes its number from
 * the lanes' one counter when it is built -- other lanes plan before it is promoted, and a step number names one build on both
 * nodes -- and promote_next takes no other; the step issued ahead is the one step() plans there (the same rows at the same
 * positions). Two lanes, two decoders a lane, the engine's order: lane 0 plans, lane 1 plans, lane 0's step ahead, lane 1's
 * next, lane 0 commits and promotes. */
TEST(lanes_a_step_issued_ahead_takes_its_number_when_built_and_is_the_step_planned_after_the_commit) {
    for (const bool ahead : { true, false }) {
        Lanes L;
        L.seqs = 8;
        REQUIRE_EQ(L.start(2), RAD_OK);
        std::vector<uint64_t> ids;
        for (int i = 0; i < 4; ++i) ids.push_back(L.submit(i % 2, 20, 200, 1000 + 50 * i));
        /* both lanes through their prompts to decode steps */
        for (int r = 0; r < 6; ++r)
            for (int l = 0; l < 2; ++l) {
                if (!L.plan(l, { 1 - l })) continue;
                CHECK_OK(L.commit(l, 7));
            }
        /* the sampler's answer the engine stages (Request::speculation_fixed: no grammar that decides at commit) */
        for (uint64_t id : ids)
            for (int l = 0; l < 2; ++l)
                if (const Request* rq = L.req(l, id)) const_cast<Request*>(rq)->speculation_fixed = true;
        std::vector<int> nums;
        for (int r = 0; r < 20; ++r) {
            const RadBatch* b0 = L.plan(0, { 1 });
            REQUIRE(b0 != nullptr);
            CHECK(b0->phase == RAD_PHASE_DECODE);
            nums.push_back(b0->step);
            const RadBatch* n0 = nullptr;
            std::vector<std::pair<int32_t, int32_t>> want;   /* (slot, ctx) the step after lane 0's commit plans */
            if (ahead) {
                n0 = L.at(0).step_next();
                REQUIRE(n0 != nullptr);
                nums.push_back(n0->step);
            }
            const RadBatch* b1 = L.plan(1, { 0 });
            REQUIRE(b1 != nullptr);
            nums.push_back(b1->step);
            if (ahead) {
                std::vector<std::pair<int32_t, int32_t>> got;
                for (const StepEntry& e : L.at(0).plan().e) got.emplace_back(e.slot, e.ctx_len + 1);
                CHECK_OK(L.commit(0, 7));
                L.at(0).promote_next();
                std::vector<std::pair<int32_t, int32_t>> now;
                for (const StepEntry& e : L.at(0).plan().e) now.emplace_back(e.slot, e.ctx_len);
                CHECK(now == got);                       /* every row one position on, the same rows */
                CHECK_EQ(L.at(0).plan().e.size(), (size_t)2);
                CHECK_OK(L.commit(0, 7));                /* the step issued ahead, committed in its turn */
            } else {
                CHECK_OK(L.commit(0, 7));
            }
            CHECK_OK(L.commit(1, 7));
        }
        std::vector<int> sorted = nums;
        std::sort(sorted.begin(), sorted.end());
        CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());   /* no number twice */
        CHECK(std::is_sorted(nums.begin(), nums.end()));                          /* in the order they were built */
        fprintf(stderr, "    %s: %zu steps built over two lanes, numbers %d .. %d, none twice, in build order\n",
                ahead ? "with lane 0's steps issued ahead" : "without", nums.size(), nums.front(), nums.back());
    }
}

/* tp8 loop F22 (cards lq46-m8B: M8's decode lanes drafted less; /stats rows_unspec 1,582 in the c16 window vs 0-16 elsewhere): a
 * newcomer the BG lane runs WHOLE has its prompt's drafter context written there (the BG step's context pass mirrors every
 * entry), but its home lane never ran a chunk of it, so before the fix the home's first block_begin found no drafter range and
 * took draft_floor() = the prompt's end (hit_holes_draft_: no group of the drafter's is prefix-cached) -- the newcomer could
 * not propose until it had decoded the drafter's window of tokens. Now the piece's commit (lane_piece_done) records the range
 * as the home's own block_end would. The engine's order: plan, commit, then the drafter's block_begin over the plan just
 * committed (on_card false: the host form), block_end. DS geometry (interval 2048, MNBT 4160, span 2), a block drafter of
 * window 32, 8 decoders, an 1,100-token newcomer: one lane (the reference: it proposes from its prompt's step) and BG at 3 lanes
 * with the fix (from its first home decode step; lq46's build, before it: not before ~window - 1 steps). The fix's opt-out
 * RADIANCE_PP_BG_DRAFT_FLOOR=0 and this test's arm of it were removed 18.11. (tp8 loop REPORT rule 7). */
TEST(lanes_a_newcomer_the_bg_lane_ran_whole_drafts_from_its_first_home_decode_step) {
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); }
    } env;
    const int W = 32;
    struct Out { int prefill_offered = -1, first_decode_offer = -1, home_decodes = 0, bg_steps = 0; int64_t floors = 0; };
    auto run = [&](int lanes) -> Out {
        Out o;
        setenv("RADIANCE_PP_BG_LANE", "1", 1);
        DstSwitches restore;
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.max_tok = 4160;
        L.max_ctx = 60000;
        L.stage_rows = 1024;
        L.seqs = 12;
        L.n_spec = 7;
        L.prog.drafter.kind = RAD_DRAFT_BLOCK;
        L.prog.drafter.window = W;
        const int sr = L.start(lanes);
        CHECK_EQ(sr, RAD_OK);
        if (sr != RAD_OK) return o;
        CHECK(DstSwitches::set("sched_bi_chunk", 3));
        const int nd = lanes - L.set.bg_n();
        for (int i = 0; i < 8; ++i) L.submit(i % nd, 100, 4000, 2000 + 100 * i);
        uint64_t p = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        Scheduler::BlockRow rows[12];
        for (int round = 0; round < 3000; ++round) {
            if (round == 4) p = L.submit(0, 1100, 4 * W);
            const Request* rp = p ? L.req(0, p) : nullptr;
            if (rp && (int)rp->output.size() >= 2 * W) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                std::vector<int> order, idle_after;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                for (int f : fifo) order.push_back(f);
                if (!L.plan(l, order)) continue;
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
                todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
            }
            if (fifo.empty()) break;
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            bool p_prefill = false, p_decode = false;
            for (const StepEntry& e : L.at(l).plan().e)
                if (p && e.seq == p) (e.is_prefill ? p_prefill : p_decode) = true;
            if (L.set.is_bg(l)) ++o.bg_steps;
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
            /* the drafter over the step just committed, in the lane that ran it */
            const int n = L.at(l).block_begin(8, 7, W, 99, rows, 12, /*on_card=*/false);
            bool offered = false;
            for (int k = 0; k < n; ++k) offered |= p && rows[k].seq == p;
            L.at(l).block_end();
            if (p_prefill && l == 0) o.prefill_offered = offered ? 1 : 0;
            if (p_decode && l == 0) {
                if (offered && o.first_decode_offer < 0) o.first_decode_offer = o.home_decodes;
                ++o.home_decodes;
            }
        }
        o.floors = L.set.counts().bg_floor;
        CHECK(p != 0 && L.req(0, p) && (int)L.req(0, p)->output.size() >= 2 * W);
        return o;
    };
    const Out one = run(1), bg = run(3);
    /* one lane: the newcomer's MIXED step is its home's, and its drafter proposes behind it */
    CHECK_EQ(one.prefill_offered, 1);
    CHECK_EQ(one.first_decode_offer, 0);
    CHECK_EQ(one.floors, 0ll);
    /* BG with the fix: the BG lane ran the prompt; the home offers a drafting row from its first decode step */
    CHECK(bg.bg_steps > 0);
    CHECK_EQ(bg.prefill_offered, -1);
    CHECK_EQ(bg.first_decode_offer, 0);
    CHECK_EQ(bg.floors, 1ll);
    fprintf(stderr, "    a newcomer of 1,100 tokens, drafter window %d: one lane offers it a drafting row at its prompt's step (%d) and "
                    "home decode step %d; BG lane at home decode step %d of %d (%lld range opened, %d BG steps)\n", W,
            one.prefill_offered, one.first_decode_offer, bg.first_decode_offer, bg.home_decodes, (long long)bg.floors, bg.bg_steps);
}

/* tp8 loop P6 (cards lq12: lent 2048-row chunks every ~282 ms, the two stages overlapping ~15 %): a chunk another lane lent
 * carries its home request's lever PSK plan (design 2's next target less the chunk end), the value one lane plans for that
 * chunk -- not "no plan", under which the plugin runs design 1's cone over every lent chunk's end. ds41_psk 2 / K 2 on the
 * DS line's exact geometry (interval 2048, 4160, span 2), 1 / 2 / 3 lanes, a 20,000-token prompt. */
TEST(lanes_a_lent_chunk_carries_its_home_requests_psk_plan) {
    std::map<int, int64_t> ref;
    for (int lanes : { 1, 2, 3 }) {
        Lanes L;
        L.grid = 2048;
        L.cfg.prefill_span = 2;
        L.cfg.pp_plan_prefix = true;
        L.max_tok = 4160;
        L.max_ctx = 30000;
        L.stage_rows = 1024;
        L.seqs = 6;
        REQUIRE_EQ(L.start(lanes), RAD_OK);
        for (int l = 0; l < lanes; ++l) {
            L.at(l).set_psk_k(2);
            L.at(l).set_psk_mode(2);
        }
        const uint64_t s = L.submit(0, 20000, 2);
        REQUIRE(s != 0);
        std::map<int, int64_t> got;
        int lent = 0, noplan = 0;
        std::vector<int> fifo;
        std::vector<char> busy((size_t)lanes, 0);
        int next = 0;
        for (int round = 0; round < 400; ++round) {
            const Request* r0 = L.req(0, s);
            if (!r0 || !r0->output.empty() || r0->state == ReqState::Failed || r0->state == ReqState::Finished) break;
            std::vector<int> todo;
            for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
            for (size_t at = 0; at < todo.size(); ++at) {
                const int l = todo[at];
                if (busy[(size_t)l]) continue;
                std::vector<int> order, idle_after;
                for (int j = 1; j < lanes; ++j)
                    if (!busy[(size_t)((l + j) % lanes)]) { order.push_back((l + j) % lanes); idle_after.push_back((l + j) % lanes); }
                for (int f : fifo) order.push_back(f);
                const RadBatch* b = L.plan(l, order);
                if (!b) continue;
                std::vector<int32_t> ahead, rows;
                int64_t dec = 0;
                const bool planned = L.at(l).psk_plan(b->step, &ahead, &rows, &dec);
                size_t k = 0;
                for (const StepEntry& e : L.at(l).plan().e) {
                    if (!e.is_prefill) continue;
                    if (e.seq == s) {
                        const int64_t a = planned && k < ahead.size() ? ahead[k] : (int64_t)Scheduler::kPskNoPlan;
                        got[e.ctx_len] = a;
                        lent += e.home >= 0;
                        noplan += a == (int64_t)Scheduler::kPskNoPlan;
                    }
                    ++k;
                }
                busy[(size_t)l] = 1;
                fifo.push_back(l);
                next = (l + 1) % lanes;
                todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
            }
            if (fifo.empty()) break;
            const int l = fifo.front();
            fifo.erase(fifo.begin());
            CHECK_OK(L.commit(l, 5));
            busy[(size_t)l] = 0;
        }
        REQUIRE(L.req(0, s) != nullptr);
        CHECK_EQ(L.req(0, s)->n_computed, (int64_t)20000);
        CHECK_EQ(noplan, 0);
        if (lanes == 1) ref = got;
        else {
            CHECK(lent > 0);
            CHECK(got == ref);
            fprintf(stderr, "    %d lanes: %zu chunks, %d lent, every chunk's PSK plan = one lane's: %s\n", lanes, got.size(), lent,
                    got == ref ? "yes" : "NO");
        }
    }
}

/* tp8 MERGE (fn4 pp_merge; LaneSet::merge_into / split_from): lane membership in process. 4 lanes with 2 BG lanes (BEST's form), four
 * decoders a decode lane. Merged, lane 0's every step carries all eight -- each on its own card, no card twice -- lane 1 holds none
 * and is parked, and pick() names lane 0 for a request submitted meanwhile, which lane 0 admits beside them. Split, the live
 * requests go back 5 / 4 over the decode lanes, every request in exactly one lane's step a round, cards distinct over both lanes.
 * Every request's output is one token a step that produced one, through both moves (a move changes only the lane). */
TEST(lanes_merge_moves_every_decoder_into_one_lane_and_splits_them_back) {
    struct Env {
        ~Env() { unsetenv("RADIANCE_PP_BG_LANE"); }
    } env;
    setenv("RADIANCE_PP_BG_LANE", "2", 1);
    Lanes L;
    L.seqs = 12;
    L.max_tok = 256;
    REQUIRE_EQ(L.start(4), RAD_OK);
    CHECK_EQ(L.set.decode_n(), 2);
    CHECK_EQ(L.set.merged_to(), -1);
    std::vector<uint64_t> ids;
    for (int i = 0; i < 8; ++i) ids.push_back(L.submit(i % 2, 10, 400, 2000 + 100 * i));
    std::map<uint64_t, int> produced;
    int dup_cards = 0, twice = 0;
    /* one round: the named decode lanes plan (each other's planners, then the BG lanes), then the BG lanes (a newcomer lent whole
     * runs there), then every planned lane commits in that order */
    auto round = [&](const std::vector<int>& run, std::vector<size_t>* dec) {
        std::vector<int32_t> cards;
        std::map<uint64_t, int> seen;
        std::vector<int> planned, all = run;
        all.push_back(2);
        all.push_back(3);
        for (int l : all) {
            std::vector<int> order;
            for (int m : all) if (m != l) order.push_back(m);
            if (!L.plan(l, order)) continue;
            planned.push_back(l);
            size_t n = 0;
            for (const StepEntry& e : L.at(l).plan().e) {
                if (e.produces_token) ++produced[e.seq];
                if (e.home >= 0) continue;
                cards.push_back(e.card);
                if (++seen[e.seq] > 1) ++twice;
                if (!e.is_prefill) ++n;
            }
            if (dec && !L.set.is_bg(l)) dec->push_back(n);
        }
        std::sort(cards.begin(), cards.end());
        for (size_t i = 1; i < cards.size(); ++i) dup_cards += cards[i] == cards[i - 1] && cards[i] >= 0;
        for (int l : planned) CHECK_OK(L.commit(l, 7));
    };
    for (int r = 0; r < 4; ++r) round({ 0, 1 }, nullptr);
    /* merge: every movable decoder of lane 1 and its free cards into lane 0 */
    const LaneSet::MoveCounts m = L.set.merge_into(0);
    CHECK_EQ(m.moved, (int64_t)4);
    CHECK_EQ(m.lent + m.ctx + m.other + m.busy, (int64_t)0);
    CHECK(m.cards > 0);
    CHECK_EQ(L.set.merged_to(), 0);
    CHECK(L.set.parked(1));
    CHECK(!L.set.parked(0));
    CHECK(!L.set.parked(2));
    CHECK_EQ(L.set.parked_live(), (int64_t)0);
    for (uint64_t id : ids) {
        CHECK(L.req(0, id) != nullptr);
        CHECK(L.at(1).find(id) == nullptr);
        CHECK(L.set.owner(id) == &L.at(0));
    }
    for (int q = 0; q < 6; ++q) CHECK_EQ(L.set.pick(), 0);
    std::vector<size_t> dec;
    for (int r = 0; r < 5; ++r) round({ 0 }, &dec);
    for (size_t n : dec) CHECK_EQ(n, (size_t)8);   /* one lane of eight decoders */
    /* a newcomer while merged: lane 0 by pick(), admitted beside the eight */
    const uint64_t nc = L.submit(L.set.pick(), 10, 30, 9000);
    REQUIRE(nc != 0);
    ids.push_back(nc);
    dec.clear();
    for (int r = 0; r < 4; ++r) round({ 0 }, &dec);
    CHECK(L.req(0, nc) != nullptr && !L.req(0, nc)->output.empty());
    CHECK_EQ(dec.back(), (size_t)9);
    /* split: 9 live -> 5 stay, 4 to lane 1 (every other one in lane 0's running order), cards to an even share */
    const LaneSet::MoveCounts s = L.set.split_from(0);
    CHECK_EQ(s.moved, (int64_t)4);
    CHECK_EQ(L.set.merged_to(), -1);
    CHECK(!L.set.parked(1));
    int in0 = 0, in1 = 0;
    for (uint64_t id : ids) {
        const bool a = L.at(0).find(id) != nullptr, b = L.at(1).find(id) != nullptr;
        CHECK(a != b);
        in0 += a;
        in1 += b;
    }
    CHECK_EQ(in0, 5);
    CHECK_EQ(in1, 4);
    dec.clear();
    for (int r = 0; r < 5; ++r) round({ 0, 1 }, &dec);
    for (size_t i = 0; i < dec.size(); ++i) CHECK_EQ(dec[i], (size_t)(i % 2 == 0 ? 5 : 4));
    CHECK_EQ(dup_cards, 0);
    CHECK_EQ(twice, 0);
    /* every request's output: one token a producing step it ran in, whichever lane ran it */
    for (uint64_t id : ids) {
        const Request* r = L.set.owner(id) ? L.set.owner(id)->find(id) : nullptr;
        REQUIRE(r != nullptr);
        CHECK_EQ((int)r->output.size(), produced[id]);
        CHECK(r->n_computed == (int64_t)r->prompt.size() + (int64_t)r->output.size() - 1);
    }
    /* merged again with lane 0 and lane 1 both holding requests, then split: the same 5 / 4 */
    CHECK_EQ(L.set.merge_into(0).moved, (int64_t)4);
    round({ 0 }, nullptr);
    CHECK_EQ(L.set.split_from(0).moved, (int64_t)4);
    round({ 0, 1 }, nullptr);
    CHECK_EQ(dup_cards, 0);
    CHECK_EQ(twice, 0);
    fprintf(stderr, "    merged 8 decoders into lane 0 (%lld free cards moved), 9 split 5 / 4 (%lld free cards back)\n",
            (long long)m.cards, (long long)s.cards);
}

/* tp8 MERGE: the merged lane's occupancy rule acts at once. A controller drafting at 16 rows under draft_occ 17 (occupancy only,
 * draft_depth0 3) runs depth 0 at its FIRST merged cycle of 32 rows (not after draft_occ_k cycles), holds it while merged with no
 * occupancy run counted and no slab lend wanted, drafts at 16 merged rows (c16's merged lane), and drafts again at its first
 * cycle after the split. */
TEST(merged_lane_runs_depth_zero_at_once_from_draft_occ_rows_and_lends_nothing) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 5;
    p.adapt = true;
    p.depth0 = true;
    p.depth0_occ = true;
    p.lend = 1;
    CHECK_OK(c.configure(p));
    DraftAdapt u[32];
    DraftAdapt* f[32];
    for (int i = 0; i < 32; ++i) { u[i] = estimate(c, 0.9, 0.9); f[i] = &u[i]; }
    c.set_occ(17);
    CHECK(c.depth0_occ_only());
    for (int t = 0; t < 40; ++t) { (void)c.plan_step(f, 16, 16); CHECK(c.head_runs()); }
    DraftController never = c;   /* the same controller, never merged */
    c.set_merged(true);
    CHECK(c.merged());
    (void)c.plan_step(f, 32, 32);
    CHECK(!c.head_runs());
    CHECK(c.merged_zero());
    CHECK_EQ(c.step_depth(), 0);
    (void)never.plan_step(f, 32, 32);
    CHECK(never.head_runs());   /* unmerged: draft_occ_k cycles first */
    int want = 0;
    for (int t = 0; t < 100; ++t) {
        (void)c.plan_step(f, 32, 32);
        CHECK(!c.head_runs());
        if (!c.head_runs()) c.step_time(0, 32, 20.0);
        want += c.lend_wanted() ? 1 : 0;
    }
    CHECK_EQ(want, 0);
    CHECK_EQ(c.occ_cycles(), (int64_t)0);
    CHECK_EQ(c.merged_zero_cycles(), (int64_t)101);
    (void)c.plan_step(f, 16, 16);   /* c16's merged lane: 16 rows draft */
    CHECK(c.head_runs());
    CHECK(!c.merged_zero());
    (void)c.plan_step(f, 32, 32);
    CHECK(!c.head_runs());
    c.set_merged(false);
    (void)c.plan_step(f, 16, 16);   /* split: the lanes of 16 draft at once */
    CHECK(c.head_runs());
    CHECK(!c.merged_zero());
}



/* ======================================================= tp8 loop P44: fn4 sched_srf over the lanes (upstream 67cf57b8) */
namespace {
/* Several prompts submitted together (lane l_i, n_i tokens), planned as Engine::step_loop_lanes plans the lanes (run_cold's loop)
 * until every one has its first token: each prompt's chunks [at, end), the round its first token came, the plans that carried
 * prefill rows of two prompts or more, window misses and failed commits. */
struct SrfRun {
    std::vector<std::vector<std::pair<int, int>>> chunks;
    std::vector<int> first;
    int two = 0, failed = 0, rounds = 0;
    int64_t misses = 0;
};
SrfRun run_multi(Lanes& L, int lanes, const std::vector<uint64_t>& ids, const std::vector<int>& home) {
    SrfRun out;
    out.chunks.resize(ids.size());
    out.first.assign(ids.size(), -1);
    std::vector<int> fifo;
    std::vector<char> busy((size_t)lanes, 0);
    int next = 0;
    for (int round = 0; round < 1200; ++round) {
        bool all = true;
        for (size_t i = 0; i < ids.size(); ++i) {
            const Request* r = L.req(home[i], ids[i]);
            if (out.first[i] < 0 && (!r || !r->output.empty() || r->state == ReqState::Finished)) out.first[i] = round;
            if (r && r->state == ReqState::Failed) ++out.failed;
            all = all && out.first[i] >= 0;
        }
        if (all) break;
        out.rounds = round + 1;
        std::vector<int> todo;
        for (int k = 0; k < lanes; ++k) todo.push_back((next + k) % lanes);
        for (size_t at = 0; at < todo.size(); ++at) {
            const int l = todo[at];
            if (busy[(size_t)l]) continue;
            std::vector<int> order, idle_after;
            for (int j = 1; j < lanes; ++j)
                if (!busy[(size_t)((l + j) % lanes)]) {
                    order.push_back((l + j) % lanes);
                    idle_after.push_back((l + j) % lanes);
                }
            for (int f : fifo) order.push_back(f);
            const RadBatch* b = L.plan(l, order);
            if (!b) continue;
            out.misses += L.window_misses(l, b);
            std::vector<uint64_t> seqs;
            for (const StepEntry& e : L.at(l).plan().e) {
                if (!e.is_prefill) continue;
                if (std::find(seqs.begin(), seqs.end(), e.seq) == seqs.end()) seqs.push_back(e.seq);
                for (size_t i = 0; i < ids.size(); ++i)
                    if (e.seq == ids[i]) out.chunks[i].emplace_back(e.ctx_len, e.ctx_len + e.n_tokens);
            }
            out.two += seqs.size() >= 2;
            busy[(size_t)l] = 1;
            fifo.push_back(l);
            next = (l + 1) % lanes;
            todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
        }
        if (fifo.empty()) continue;
        const int l = fifo.front();
        fifo.erase(fifo.begin());
        if (L.commit(l, 5) < 0) ++out.failed;
        busy[(size_t)l] = 0;
    }
    return out;
}

/* the DS line's geometry (lanes_span_off_cuts_...'s): interval 2048, MNBT 4160, span 2, window 16; `best` = BEST's span cut
 * (RADIANCE_PP_SPAN_SHORT=24576 + RADIANCE_PP_SPAN_TAIL=8192) */
void srf_lanes(Lanes& L, int lanes, bool best) {
    L.grid = 2048;
    L.cfg.prefill_span = 2;
    L.max_tok = 4160;
    L.max_ctx = 70000;
    L.kv_blocks = 16384;
    L.stage_rows = 1024;
    L.win = 16;
    L.seqs = 8;
    REQUIRE_EQ(L.start(lanes), RAD_OK);
    if (best) {
        L.set.set_span_short(24576);
        L.set.set_span_tail(8192);
    }
}
}  /* namespace */

/* sched_srf 2 over 2 and 3 lanes: prompts submitted together to the decode lanes (pick()'s spread), nothing decoding -- every
 * prompt's chunks are exactly its lone plan (the one-process grid; run_cold of it alone), no plan carries rows of two prompts,
 * the first tokens come shortest first (equal lengths: the first arrival first), every window key is read right and nothing
 * fails; sched_srf 0 (the control) interleaves them (plans with two prompts, or a chunk off a lone plan). */
TEST(srf_2_prompts_over_the_lanes_one_at_a_time_shortest_first_each_its_lone_plan) {
    struct Case { std::vector<int> n; };
    const std::vector<Case> cases = { { { 15777, 8500 } }, { { 16384, 16384 } }, { { 16000, 9000, 12000, 20000 } },
                                      { { 40000, 6000 } } };
    for (int best : { 0, 1 })
    for (int lanes : { 2, 3 })
    for (const Case& c : cases) {
        /* each prompt alone: its lone plan at this lane count */
        std::vector<std::vector<std::pair<int, int>>> lone;
        for (size_t i = 0; i < c.n.size(); ++i) {
            Lanes L;
            srf_lanes(L, lanes, best != 0);
            const uint64_t s = L.submit(0, c.n[i], 1, 1000);
            REQUIRE(s != 0);
            int64_t misses = 0;
            int failed = 0;
            lone.push_back(run_cold(L, lanes, s, &misses, &failed));
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
        }
        for (int srf : { 0, 2 }) {
            SrfSwitch on(srf);
            Lanes L;
            srf_lanes(L, lanes, best != 0);
            std::vector<uint64_t> ids;
            std::vector<int> home;
            for (size_t i = 0; i < c.n.size(); ++i) {
                const int h = (int)(i % (size_t)lanes);   /* pick(): the fewest live requests, the lowest index */
                ids.push_back(L.submit(h, c.n[i], 1, 1000));
                home.push_back(h);
                REQUIRE(ids.back() != 0);
            }
            const SrfRun r = run_multi(L, lanes, ids, home);
            CHECK(L.set.broken().empty());
            CHECK_EQ(r.failed, 0);
            CHECK_EQ(r.misses, (int64_t)0);
            int off_lone = 0;
            for (size_t i = 0; i < c.n.size(); ++i) {
                CHECK(r.first[i] >= 0);
                off_lone += r.chunks[i] != lone[i];
                int64_t rows = 0;
                for (size_t k = 0; k < r.chunks[i].size(); ++k) {
                    rows += r.chunks[i][k].second - r.chunks[i][k].first;
                    if (k) CHECK_EQ(r.chunks[i][k].first, r.chunks[i][k - 1].second);   /* position order */
                }
                CHECK_EQ(rows, (int64_t)c.n[i]);
            }
            std::string firsts, off;
            for (size_t i = 0; i < c.n.size(); ++i) firsts += fmt("%s%d@%d", i ? " " : "", c.n[i], r.first[i]);
            /* the first piece of a prompt that is not one of its lone plan's (sched_srf 0: today's composition) */
            for (size_t i = 0; i < c.n.size(); ++i)
                for (const auto& pc : r.chunks[i])
                    if (std::find(lone[i].begin(), lone[i].end(), pc) == lone[i].end()) {
                        off += fmt("%s%d's [%d, %d)", off.empty() ? "; off-plan pieces e.g. " : ", ", c.n[i], pc.first, pc.second);
                        break;
                    }
            fprintf(stderr, "    %s%d lanes, sched_srf %d: first tokens (prompt@round) %s; %d plans with two prompts, %d of %zu "
                    "prompts off their lone plan, %d rounds%s\n", best ? "BEST's span cut, " : "", lanes, srf, firsts.c_str(), r.two,
                    off_lone, c.n.size(), r.rounds, off.c_str());
            if (srf == 0) continue;
            CHECK_EQ(r.two, 0);
            CHECK_EQ(off_lone, 0);
            /* shortest first: a shorter prompt's first token never after a longer one's (equal: the earlier id first) */
            for (size_t i = 0; i < c.n.size(); ++i)
                for (size_t j = 0; j < c.n.size(); ++j)
                    if (c.n[i] < c.n[j] || (c.n[i] == c.n[j] && i < j)) CHECK(r.first[i] <= r.first[j]);
            int64_t k[Scheduler::kSrfN] = {}, one[Scheduler::kSrfN];
            for (int l = 0; l < lanes; ++l) {
                L.at(l).srf_counts(one);
                for (int q = 0; q < Scheduler::kSrfN; ++q) k[q] += one[q];
            }
            CHECK(k[Scheduler::kSrfSteps] > 0);
            CHECK(k[Scheduler::kSrfHeld] > 0);
        }
    }
}

/* sched_srf 2 with a DECODER in another lane: the rule across the lanes is off while any decode lane decodes, and each lane runs 1's rule
 * (upstream's): two prompts sent to lane 0 beside lane 1's decoding request -- after the step that admits them (an admission makes a shared
 * step) lane 0's plans carry one of them at a time, the shorter first; every request completes, window keys read right, nothing fails. */
TEST(srf_2_with_a_decoder_in_another_lane_acts_per_lane) {
    for (int lanes : { 2, 3 }) {
        SrfSwitch on(2);
        Lanes L;
        srf_lanes(L, lanes, true);
        const uint64_t d = L.submit(1, 32, 400, 1000);
        REQUIRE(d != 0);
        /* d alone until it decodes */
        for (int k = 0; k < 4 && L.req(1, d) && L.req(1, d)->output.empty(); ++k) {
            REQUIRE(L.plan(1, { 0 }) != nullptr);
            CHECK_OK(L.commit(1, 5));
        }
        REQUIRE(L.req(1, d) != nullptr);
        REQUIRE(!L.req(1, d)->output.empty());
        const uint64_t a = L.submit(0, 9000, 1, 1000), b = L.submit(0, 5000, 1, 1000);
        REQUIRE(a != 0);
        REQUIRE(b != 0);
        const SrfRun r = run_multi(L, lanes, { a, b }, { 0, 0 });
        CHECK(L.set.broken().empty());
        CHECK_EQ(r.failed, 0);
        CHECK_EQ(r.misses, (int64_t)0);
        CHECK(r.first[0] >= 0 && r.first[1] >= 0);
        CHECK(r.first[1] <= r.first[0]);   /* b, the shorter, first */
        CHECK(L.req(1, d) && L.req(1, d)->state == ReqState::Running);   /* still decoding all along */
        int64_t k[Scheduler::kSrfN] = {}, one[Scheduler::kSrfN];
        for (int l = 0; l < lanes; ++l) {
            L.at(l).srf_counts(one);
            for (int q = 0; q < Scheduler::kSrfN; ++q) k[q] += one[q];
        }
        fprintf(stderr, "    %d lanes, a decoder in lane 1: 9000@%d 5000@%d, %d plans with two prompts, rule plans %lld (alone %lld), sat out %lld, "
                        "held %lld\n", lanes, r.first[0], r.first[1], r.two, (long long)k[Scheduler::kSrfSteps],
                (long long)k[Scheduler::kSrfAlone], (long long)k[Scheduler::kSrfOut], (long long)k[Scheduler::kSrfHeld]);
        CHECK(r.two <= 1);                       /* the admission step only */
        CHECK(k[Scheduler::kSrfOut] > 0);        /* a sat out while b ran */
        CHECK_EQ(k[Scheduler::kSrfHeld], 0ll);   /* nothing held: the rule across the lanes was off */
    }
}

/* sched_srf 2 with the window pool short (tp8 prefill P0's case: 8 / 10 / 12 blocks of 16 for a 16-key window, the window group's pool
 * against the keep floor), two cold prompts sent together to lanes 0 / 1 at 2 and 3 lanes under --prefill-span 2: both complete, every
 * window key read right, nothing fails, no plan carries both -- where the pool refuses the chosen one's chunk the KV valve lets the other
 * run (order and lone plans then follow the pool). sched_srf 0's outcome printed beside it (at 8 blocks today's composition stalls both:
 * neither has a first token after 1,200 rounds -- each holds window blocks the other's next chunk needs). */
TEST(srf_2_two_cold_prompts_beside_a_short_window_pool) {
    for (int64_t pool : { 8, 10, 12 })
    for (int lanes : { 2, 3 }) {
        std::vector<std::vector<std::pair<int, int>>> lone;
        const std::vector<int> n = { 333, 200 };
        for (size_t i = 0; i < n.size(); ++i) {
            Lanes L;
            L.win = 16;
            L.win_pool = pool;
            L.cfg.prefill_span = 2;   /* the rule acts only under a span (BEST: --prefill-span 2) */
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            const uint64_t s = L.submit(0, n[i], 1);
            int64_t misses = 0;
            int failed = 0;
            lone.push_back(run_cold(L, lanes, s, &misses, &failed));
            CHECK_EQ(failed, 0);
            CHECK_EQ(misses, (int64_t)0);
        }
        for (int srf : { 0, 2 }) {
            SrfSwitch on(srf);
            Lanes L;
            L.win = 16;
            L.win_pool = pool;
            L.cfg.prefill_span = 2;   /* the rule acts only under a span (BEST: --prefill-span 2) */
            REQUIRE_EQ(L.start(lanes), RAD_OK);
            const uint64_t a = L.submit(0, n[0], 1), b = L.submit(1, n[1], 1);
            REQUIRE(a != 0);
            REQUIRE(b != 0);
            const SrfRun r = run_multi(L, lanes, { a, b }, { 0, 1 });
            const int off = (r.chunks[0] != lone[0]) + (r.chunks[1] != lone[1]);
            fprintf(stderr, "    window pool %lld blocks, %d lanes, sched_srf %d: first tokens 333@%d 200@%d, %d failed, %lld window keys missed, "
                            "%d plans with two prompts, %d of 2 off their lone plan, %lld window waits\n", (long long)pool, lanes, srf, r.first[0],
                    r.first[1], r.failed, (long long)r.misses, r.two, off, (long long)L.set.counts().why[LaneSet::kWinWait]);
            CHECK(L.set.broken().empty());
            if (srf == 0) continue;
            CHECK_EQ(r.failed, 0);
            CHECK_EQ(r.misses, (int64_t)0);
            CHECK(r.first[0] >= 0 && r.first[1] >= 0);
            CHECK_EQ(r.two, 0);
            if (pool >= 12) {   /* room for both: the shorter first, each its lone plan */
                CHECK(r.first[1] <= r.first[0]);
                CHECK_EQ(off, 0);
            }
        }
    }
}

/* tp8 loop D55 / D55b: RADIANCE_SCHED_PUB_PROMPT=1 publishes a request's prompt when its last chunk commits (it is still
 * running), once; =2 also publishes the committed part at every prefill chunk; unset: only at its close. A recycled request slot
 * publishes again. RADIANCE_SCHED_PUB_HOLD_MS holds a fresh request whose prompt shares the head a running prompt is computing:
 * it is not admitted until that head is available (=1: the sibling's prompt done; =2: its computed end past the head), and
 * never past the bound. */
TEST(d55_pub_prompt_publishes_at_the_last_chunk_and_d55b_holds_a_sibling_for_its_head) {
    auto published_while_running = [](const char* mode, int* n_pub, int* n_mid) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/128);
        if (mode) setenv("RADIANCE_SCHED_PUB_PROMPT", mode, 1); else unsetenv("RADIANCE_SCHED_PUB_PROMPT");
        CHECK_OK(f.start());
        unsetenv("RADIANCE_SCHED_PUB_PROMPT");
        *n_pub = *n_mid = 0;
        for (int round = 0; round < 2; ++round) {   /* the second request takes the first one's recycled slot */
            const uint64_t a = f.submit(400, 8, 0, /*ignore_eos=*/true);
            for (int k = 0; k < 40; ++k) {
                const size_t before = f.pc.published.size();
                if (!f.sch.step()) break;
                f.commit_plain();
                const Request* r = f.sch.find(a);
                for (size_t i = before; i < f.pc.published.size(); ++i)
                    if (f.pc.published[i].seq == a && r && r->state == ReqState::Running) {
                        if (f.pc.published[i].n_tokens == 400) ++*n_pub;
                        else ++*n_mid;
                    }
            }
            while (f.sch.step() != nullptr) f.commit_plain();
        }
    };
    int pub = -1, mid = -1;
    published_while_running(nullptr, &pub, &mid);
    fprintf(stderr, "    unset: %d publish(es) of the whole prompt while running, %d of a part\n", pub, mid);
    CHECK_EQ(pub, 0);
    CHECK_EQ(mid, 0);
    published_while_running("1", &pub, &mid);
    fprintf(stderr, "    =1: %d publish(es) of the whole prompt while running (2 requests, the second in a recycled slot), %d of a part\n", pub, mid);
    CHECK_EQ(pub, 2);
    CHECK_EQ(mid, 0);
    published_while_running("2", &pub, &mid);
    fprintf(stderr, "    =2: %d publish(es) of the whole prompt while running, %d of a part (128-token chunks of 400)\n", pub, mid);
    CHECK_EQ(pub, 2);
    CHECK_EQ(mid, 2 * 3);

    /* the hold: A runs (400 tokens in 128-token chunks), B shares A's first 300 tokens (Fixture prompts are 0..n-1: a shared head) */
    auto held_steps = [](const char* mode, const char* hold_ms, int* held, int64_t* a_at_admit) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/128);
        if (mode) setenv("RADIANCE_SCHED_PUB_PROMPT", mode, 1);
        if (hold_ms) setenv("RADIANCE_SCHED_PUB_HOLD_MS", hold_ms, 1);
        CHECK_OK(f.start());
        unsetenv("RADIANCE_SCHED_PUB_PROMPT");
        unsetenv("RADIANCE_SCHED_PUB_HOLD_MS");
        const uint64_t a = f.submit(400, 8, 0, true);
        REQUIRE(f.sch.step() != nullptr);   /* A's first chunk: A is running */
        f.commit_plain();
        const uint64_t b = f.submit(300, 8, 0, true);   /* B's prompt = A's first 300 tokens */
        *held = 0;
        *a_at_admit = -1;
        for (int k = 0; k < 60; ++k) {
            const Request* ra = f.sch.find(a);
            const int64_t a_done = ra ? ra->n_computed : 400;
            if (!f.sch.step()) break;
            bool b_in = false;
            for (const StepEntry& e : f.sch.plan().e) b_in = b_in || e.seq == b;
            if (b_in && *a_at_admit < 0) *a_at_admit = a_done;
            if (!b_in && *a_at_admit < 0) ++*held;
            f.commit_plain();
        }
        while (f.sch.step() != nullptr) f.commit_plain();
    };
    int held = -1;
    int64_t at = -1;
    held_steps(nullptr, nullptr, &held, &at);
    fprintf(stderr, "    no hold: B planned with A's computed end at %lld, after %d step(s)\n", (long long)at, held);
    CHECK(at < 299);
    held_steps("1", "60000", &held, &at);
    fprintf(stderr, "    =1 + hold: B planned with A's computed end at %lld (its prompt 400), after %d step(s)\n", (long long)at, held);
    CHECK_EQ(at, 400);
    held_steps("2", "60000", &held, &at);
    fprintf(stderr, "    =2 + hold: B planned with A's computed end at %lld (B's head 299), after %d step(s)\n", (long long)at, held);
    CHECK(at >= 299 && at < 400);
    held_steps(nullptr, "60000", &held, &at);   /* the hold needs a publish rule (else it would only delay): off */
    fprintf(stderr, "    hold, PUB_PROMPT unset (the hold is off): B planned with A's computed end at %lld, after %d step(s)\n", (long long)at, held);
    CHECK(at < 299);
}
