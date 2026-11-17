/* pp_e2e_test.cpp -- the two-node pipeline (tp8 S4) end to end on the CPU: the radiance binary as
 * two processes, a leader and a follower, over localhost TCP, serving a tiny synthetic LLaMA
 * (tests/tiny_model.h, four layers, block fp8 via rad-convert) on the host backend with libref.
 *
 * The claim under test is the pipeline's: a step split over two processes computes the bytes one
 * process computes. libref's arithmetic is row-independent, so any difference is the pipeline's
 * (a crossing that moved the wrong rows, a step the follower replayed differently, a block table
 * it did not see). Compared byte for byte:
 *
 *   - KL recordings (--kld-record: every scored position's log-probabilities and top tokens) of
 *     one process and of the pipeline at the cuts W(0,2), W(1,3), W(2,4), W(0,3) and W(1,4) (the
 *     crossings carry one part where the receiving layer starts a fold-free chain, two -- the
 *     stream and the pending delta -- elsewhere);
 *   - greedy completions over HTTP, one at a time and four at once, one process vs W(0,2);
 *
 * and the failure paths: a follower killed mid-generation (the leader answers the request with an
 * error and exits non-zero within seconds; a new pair then serves the same tokens), a clean stop
 * of the leader (the follower exits 0), nodes whose options differ (both refuse to start, naming
 * the option), a follower with a smaller KV pool (the leader caps its own at it).
 *
 * Every child gets ROCR_VISIBLE_DEVICES=99, so a machine with a card still runs the host backend.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

namespace {

/* ------------------------------------------------------------------ children */
std::vector<std::string> engine_env(const std::vector<std::string>& more = {}) {
    std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=512",
                                   "OMP_WAIT_POLICY=passive", "OMP_NUM_THREADS=2",
                                   std::string("RADIANCE_HOME=") + RAD_E2E_HOME,
                                   "RADIANCE_PP_CONNECT_MS=60000", "RADIANCE_PP_DEAD_MS=2000" };
    e.insert(e.end(), more.begin(), more.end());
    return e;
}

std::vector<std::string> engine_argv(const std::string& model, const std::vector<std::string>& args) {
    std::vector<std::string> a = { RAD_E2E_SERVER, "--model", model, "--kernels", "libref",
                                   "--debug-accept-reference-kernels" };
    a.insert(a.end(), args.begin(), args.end());
    return a;
}

std::string tail(const std::string& log, int n = 14) {
    size_t at = log.size();
    for (int i = 0; i <= n && at != std::string::npos && at > 0; ++i) at = log.rfind('\n', at - 1);
    return at == std::string::npos || at >= log.size() ? log : log.substr(at + 1);
}

/* A free loopback port (bound and released: another process could take it, which a test then
 * reports as a failed start). */
int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bind(fd, (sockaddr*)&a, sizeof a) != 0) { close(fd); return 0; }
    socklen_t l = sizeof a;
    getsockname(fd, (sockaddr*)&a, &l);
    close(fd);
    return ntohs(a.sin_port);
}

/* A child whose output is drained on a thread into `log`; `wait_for` blocks until a line appears
 * or the child exits. */
struct Proc {
    tiny::Child child;
    std::thread drain;
    std::mutex mu;
    std::condition_variable cv;
    std::string log;
    bool eof = false;
    int status = -1;

    bool start(const std::vector<std::string>& argv, const std::vector<std::string>& env) {
        if (!child.start(argv, env)) return false;
        drain = std::thread([this] {
            char buf[4096];
            for (;;) {
                const ssize_t n = ::read(child.fd, buf, sizeof buf);
                if (n < 0 && errno == EINTR) continue;
                std::lock_guard<std::mutex> lk(mu);
                if (n <= 0) { eof = true; cv.notify_all(); return; }
                log.append(buf, (size_t)n);
                cv.notify_all();
            }
        });
        return true;
    }
    bool wait_for(const std::string& tag, int timeout_s) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::seconds(timeout_s),
                           [&] { return eof || log.find(tag) != std::string::npos; }) &&
               log.find(tag) != std::string::npos;
    }
    /* The exit status, waiting at most `timeout_s` (then SIGKILL and 124). */
    int finish(int timeout_s) {
        if (child.pid <= 0) return status;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (!cv.wait_for(lk, std::chrono::seconds(timeout_s), [&] { return eof; })) {
                lk.unlock();
                kill(child.pid, SIGKILL);
            }
        }
        if (drain.joinable()) drain.join();
        close(child.fd);
        child.fd = -1;
        std::string rest;
        status = child.wait(&rest);
        return status;
    }
    std::string output() {
        std::lock_guard<std::mutex> lk(mu);
        return log;
    }
    ~Proc() {
        if (child.pid > 0) kill(child.pid, SIGKILL);
        if (drain.joinable()) drain.join();
    }
};

/* ------------------------------------------------------------------ the checkpoint */
std::string words(tiny::Rng& r, size_t n) {
    static const char* w[] = { "the", "a", "cat", "dog", "sat", "ran", "on", "under", "mat",
                               "house", "tree", "river", "stone", "light", "dark", "quick",
                               "slow", "red", "blue", "and" };
    std::string s;
    while (s.size() < n) { s += w[r.next() % 20]; s += ' '; }
    s.resize(n);
    return s;
}

struct World {
    tiny::TempDir dir{ "pp_e2e" };
    std::string llama, llama_rad, corpus, log;
    bool ok = false;
    World() {
        if (!dir.ok()) return;
        llama = dir / "llama";
        llama_rad = dir / "llama.rad";
        if (mkdir(llama.c_str(), 0755) != 0) return;
        tiny::Llama s;
        s.layers = 4;   /* two a node at the plan's bring-up cut, any cut between */
        s.ctx = 65536;  /* room for a pool the device bounds (the cap case) */
        if (!tiny::write_llama(llama, s) || !tiny::write_text(dir / "llama.recipe", tiny::kLlamaFp8Recipe))
            return;
        tiny::Rng r(11);
        std::string body;
        for (size_t n : { 170, 230 })
            body += json{ { "prompt", words(r, n) }, { "source", "d" + std::to_string(n) } }.dump() + "\n";
        corpus = dir / "corpus.jsonl";
        if (!tiny::write_text(corpus, body)) return;
        const int st = tiny::run({ RAD_E2E_CONVERT, llama, "-o", llama_rad, "--recipe", dir / "llama.recipe" },
                                 engine_env(), &log);
        if (st != 0) { fprintf(stderr, "    rad-convert exited %d:\n%s\n", st, tail(log).c_str()); return; }
        ok = true;
    }
};
World& world() { static World w; return w; }

/* The common shape of every run here: two sequences, 64-token steps (so a document prefills in
 * several chunks, with decode rows beside them), the cache small enough to matter. */
std::vector<std::string> shape() {
    return { "--max-num-seqs", "2", "--max-num-batched-tokens", "64", "--max-model-len", "1024",
             "--prefill-interleave", "0" };
}

std::vector<std::string> pp_args(int rank, int port, const std::string& cut) {
    return { "--pp-nodes", "2", "--pp-rank", std::to_string(rank), "--pp-peer",
             "127.0.0.1:" + std::to_string(port), "--pp-cut", cut };
}

std::vector<std::string> cat(std::vector<std::string> a, const std::vector<std::string>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

bool same_recording(const std::string& a, const std::string& b, std::string* why) {
    for (const char* f : { "tokens.i32", "logp.f16", "top.bin", "rows.f32" }) {
        const std::string x = tiny::read_text(a + "/" + f), y = tiny::read_text(b + "/" + f);
        if (x.empty() || x != y) {
            *why = std::string(f) + (x.empty() ? " is missing or empty" : " differs");
            return false;
        }
    }
    return true;
}

/* A KL recording of the corpus by one process. */
std::string record_single(const std::string& out, const std::vector<std::string>& extra = {}) {
    World& w = world();
    std::string log;
    const int st = tiny::run(engine_argv(w.llama_rad, cat(cat({ "--kld-record", out, "--kld-corpus", w.corpus }, shape()),
                                                         extra)),
                             engine_env(), &log, 240);
    if (st != 0) { fprintf(stderr, "    single run exited %d:\n%s\n", st, tail(log).c_str()); return ""; }
    return out;
}

/* The same by a pipeline cut at `cut`: the follower first (it listens), then the leader. */
struct PipeRun { int leader = -1, follower = -1; std::string llog, flog; };
PipeRun record_pipeline(const std::string& out, const std::string& cut,
                        const std::vector<std::string>& fenv = {}, const std::vector<std::string>& fextra = {},
                        const std::vector<std::string>& lextra = {}, const std::vector<std::string>& lenv = {}) {
    World& w = world();
    const int port = free_port();
    PipeRun pr;
    Proc fol;
    const std::vector<std::string> kl = { "--kld-record", out + ".follower", "--kld-corpus", w.corpus };
    if (!fol.start(engine_argv(w.llama_rad, cat(cat(cat(kl, shape()), pp_args(1, port, cut)), fextra)),
                   engine_env(fenv)))
        return pr;
    fol.wait_for("waiting on", 60);
    pr.leader = tiny::run(engine_argv(w.llama_rad, cat(cat(cat({ "--kld-record", out, "--kld-corpus", w.corpus }, shape()),
                                                           pp_args(0, port, cut)), lextra)),
                          engine_env(lenv), &pr.llog, 240);
    pr.follower = fol.finish(60);
    pr.flog = fol.output();
    return pr;
}

/* ------------------------------------------------------------------ HTTP */
struct Call {
    int fd = -1;
    bool send(int port, const std::string& path, const std::string& body) {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) return false;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (sockaddr*)&a, sizeof a) != 0) return false;
        const std::string req = "POST " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
                                "Content-Type: application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
        for (size_t off = 0; off < req.size();) {
            const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return false;
            off += (size_t)n;
        }
        return true;
    }
    int read(std::string* body) {
        std::string raw;
        char buf[16384];
        for (;;) {
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            raw.append(buf, (size_t)n);
        }
        close(fd);
        fd = -1;
        const size_t h = raw.find("\r\n\r\n");
        if (raw.compare(0, 9, "HTTP/1.1 ") != 0 || h == std::string::npos) return -1;
        *body = raw.substr(h + 4);
        return std::atoi(raw.c_str() + 9);
    }
    ~Call() { if (fd >= 0) close(fd); }
};

std::vector<int32_t> complete(int port, const std::vector<int32_t>& prompt, int n, int* status = nullptr) {
    const json b = { { "prompt", prompt }, { "n_predict", n }, { "temperature", 0.0 }, { "ignore_eos", true },
                     { "return_tokens", true } };
    Call c;
    std::string out;
    const int st = c.send(port, "/completion", b.dump()) ? c.read(&out) : -1;
    if (status) *status = st;
    const json j = json::parse(out, nullptr, false);
    if (!j.is_object() || !j.contains("tokens")) return {};
    return j["tokens"].get<std::vector<int32_t>>();
}

std::vector<int32_t> prompt_ids(uint64_t seed, size_t n) {
    tiny::Rng r(seed);
    return tiny::byte_ids(words(r, n));
}

/* A server: one process, or a pipeline's leader with its follower. */
struct Serving {
    Proc fol, lead;
    int port = 0, pp_port = 0;
    bool start(const std::string& cut, const std::vector<std::string>& extra = {},
               const std::vector<std::string>& env = {}) {
        World& w = world();
        std::vector<std::string> a = cat({ "--host", "127.0.0.1", "--port", "0", "--http-threads", "8" }, shape());
        a = cat(a, extra);
        if (!cut.empty()) {
            pp_port = free_port();
            if (!fol.start(engine_argv(w.llama_rad, cat(a, pp_args(1, pp_port, cut))), engine_env(env))) return false;
            if (!fol.wait_for("waiting on", 60)) return false;
            a = cat(a, pp_args(0, pp_port, cut));
        }
        if (!lead.start(engine_argv(w.llama_rad, a), engine_env(env))) return false;
        static const std::string kTag = "listening on http://127.0.0.1:";
        if (!lead.wait_for(kTag, 120)) {
            fprintf(stderr, "    the leader did not start:\n%s\n    follower:\n%s\n", tail(lead.output()).c_str(),
                    tail(fol.output()).c_str());
            return false;
        }
        const std::string out = lead.output();
        port = std::atoi(out.c_str() + out.find(kTag) + kTag.size());
        return port > 0;
    }
};

/* Five prompts: three alone, then four at once. */
std::vector<std::vector<int32_t>> greedy_set(int port) {
    std::vector<std::vector<int32_t>> out;
    for (uint64_t s : { 3, 4, 5 }) out.push_back(complete(port, prompt_ids(s, 60 + 17 * s), 24));
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 6, 7, 8, 9 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 40 + 9 * s), 20); }));
    for (auto& x : f) out.push_back(x.get());
    return out;
}

/* ================================================================== the cases */
TEST(a_pipeline_records_what_one_process_records_at_every_cut) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_rec");
    const std::string ref = record_single(d / "single");
    REQUIRE(!ref.empty());
    for (const char* cut : { "0,2", "1,3", "2,4", "0,3", "1,4" }) {
        const std::string out = d / (std::string("pp-") + cut);
        const PipeRun pr = record_pipeline(out, cut);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    W(%s): leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", cut, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(ref, out, &why);
        if (!same) fprintf(stderr, "    W(%s): %s\n", cut, why.c_str());
        CHECK(same);
        /* The follower ran passes, and stopped because the leader did. */
        CHECK(pr.flog.find("follower: 1 steps run") != std::string::npos);
        CHECK(pr.flog.find("the leader stopped") != std::string::npos);
        CHECK(pr.llog.find("rendezvous 1 passed") != std::string::npos);
    }
}

TEST(a_pipeline_serves_the_greedy_tokens_one_process_serves) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> ref, got;
    {
        Serving one;
        REQUIRE(one.start(""));
        ref = greedy_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    {
        Serving pp;
        REQUIRE(pp.start("0,2"));
        got = greedy_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        /* A clean stop of the leader stops the follower cleanly. */
        CHECK_EQ(pp.fol.finish(30), 0);
        CHECK(pp.fol.output().find("the leader stopped") != std::string::npos);
    }
    REQUIRE_EQ(ref.size(), got.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        CHECK(!ref[i].empty());
        if (ref[i] != got[i]) fprintf(stderr, "    prompt %zu: the pipeline's tokens differ\n", i);
        CHECK(ref[i] == got[i]);
    }
}

TEST(a_killed_follower_fails_the_leaders_requests_and_a_new_pair_serves_again) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(21, 50);
    std::vector<int32_t> before;
    {
        Serving pp;
        REQUIRE(pp.start("0,2"));
        before = complete(pp.port, p, 16);
        REQUIRE(!before.empty());
        /* A long request in flight, then the follower dies under it. */
        int st = 0;
        auto long_req = std::async(std::launch::async, [&] { return complete(pp.port, p, 4000, &st); });
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        kill(pp.fol.child.pid, SIGKILL);
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<int32_t> cut = long_req.get();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "    the request ended %.2f s after the kill with status %d and %zu tokens\n", s, st,
                cut.size());
        CHECK(st != 200 || cut.size() < 4000);
        CHECK(s < 10.0);
        const int ls = pp.lead.finish(30);
        fprintf(stderr, "    the leader exited %d\n", ls);
        CHECK(ls != 0 && ls != 124);
        CHECK(pp.lead.output().find("link to the follower is down") != std::string::npos ||
              pp.lead.output().find("node link: down") != std::string::npos);
    }
    {
        Serving again;
        REQUIRE(again.start("0,2"));
        CHECK(complete(again.port, p, 16) == before);
        kill(again.lead.child.pid, SIGTERM);
        CHECK_EQ(again.lead.finish(60), 0);
        CHECK_EQ(again.fol.finish(30), 0);
    }
}

TEST(nodes_whose_options_differ_both_refuse_to_start) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_diff");
    const PipeRun pr = record_pipeline(d / "rec", "0,2", {}, { "--max-num-seqs", "3" });
    CHECK(pr.leader != 0);
    CHECK(pr.follower != 0);
    const bool named = pr.llog.find("option --max-num-seqs") != std::string::npos &&
                       pr.flog.find("option --max-num-seqs") != std::string::npos;
    if (!named) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(pr.llog).c_str(), tail(pr.flog).c_str());
    CHECK(named);
    /* A follower given another cut. */
    const PipeRun pc = record_pipeline(d / "rec2", "0,2", {}, { "--pp-cut", "0,3" });
    CHECK(pc.leader != 0 && pc.follower != 0);
    CHECK(pc.llog.find("cut: this node '0,2', the peer '0,3'") != std::string::npos);
}

/* --node-link verbs on a host without an RDMA device (or without libibverbs): both nodes refuse at the second rendezvous's
 * end, naming why, instead of serving over another link. */
TEST(the_verbs_link_without_a_device_is_refused_by_name) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_verbs");
    const PipeRun pr = record_pipeline(d / "rec", "0,2", {}, { "--node-link", "verbs" }, { "--node-link", "verbs" });
    CHECK(pr.leader != 0);
    CHECK(pr.follower != 0);
    const bool named = pr.llog.find("--node-link verbs: ") != std::string::npos &&
                       (pr.llog.find("libibverbs") != std::string::npos || pr.llog.find("no RDMA device") != std::string::npos);
    if (!named) fprintf(stderr, "    leader:\n%s\n", tail(pr.llog).c_str());
    CHECK(named);
}

TEST(a_follower_with_a_smaller_cache_caps_the_leaders_and_the_recording_holds) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_cap");
    /* 2 x 64k tokens of pool want 128 MiB at 1 KiB a token: the leader's 512 MiB device holds it,
     * the follower's 120 MiB does not. */
    const std::vector<std::string> big = { "--max-model-len", "65536" };
    const std::string ref = record_single(d / "single", big);
    REQUIRE(!ref.empty());
    const PipeRun pr = record_pipeline(d / "pp", "0,2", { "RADIANCE_HOST_VRAM_MIB=120" }, big, big);
    CHECK_EQ(pr.leader, 0);
    CHECK_EQ(pr.follower, 0);
    /* Review F1: both cap before the carve, at the follower's count, and rendezvous 2 found every
     * group's block count equal (it refuses otherwise). */
    const bool capped = pr.llog.find("KV capped on both nodes before the carve") != std::string::npos &&
                        pr.flog.find("KV capped on both nodes before the carve") != std::string::npos;
    if (!capped) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(pr.llog, 30).c_str(), tail(pr.flog, 30).c_str());
    CHECK(capped);
    const std::string tag = "KV capped on both nodes before the carve at ";
    const size_t at_l = pr.llog.find(tag), at_f = pr.flog.find(tag);
    if (at_l != std::string::npos && at_f != std::string::npos) {
        const long long nl = std::atoll(pr.llog.c_str() + at_l + tag.size());
        const long long nf = std::atoll(pr.flog.c_str() + at_f + tag.size());
        fprintf(stderr, "    both nodes capped at %lld / %lld tokens\n", nl, nf);
        CHECK_EQ(nl, nf);
        CHECK(pr.llog.find("pages " + std::to_string(nl) + ",") == std::string::npos);   /* the leader paged more */
    }
    std::string why;
    CHECK(same_recording(ref, d / "pp", &why));
}

/* Review F2 / F3: a switch POSTed to the leader reaches the follower when it acts on the follower's
 * passes (aql_acq), stays the leader's when it is the scheduler's (sched_row_cap), and a value neither
 * node takes is refused naming both -- the leader asked the follower -- while the pair serves on with
 * the tokens one process serves. GET reports the follower's values ("peer"). */
TEST(a_switch_request_reaches_the_follower_and_the_pair_serves_on) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(31, 50);
    std::vector<int32_t> ref;
    {
        Serving one;
        REQUIRE(one.start(""));
        ref = complete(one.port, p, 16);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    Serving pp;
    REQUIRE(pp.start("0,2"));
    CHECK(complete(pp.port, p, 16) == ref);
    std::string out;
    Call c;
    int st = c.send(pp.port, "/admin/fn4", json{ { "aql_acq", 1 }, { "sched_row_cap", 48 } }.dump()) ? c.read(&out) : -1;
    CHECK_EQ(st, 200);
    CHECK(complete(pp.port, p, 16) == ref);   /* the step that applies them, and its follower step */
    Call c2;
    st = c2.send(pp.port, "/admin/fn4", json{ { "gg_gemv", 7 } }.dump()) ? c2.read(&out) : -1;
    CHECK_EQ(st, 400);
    const bool both = out.find("the pipeline's other node refuses it too") != std::string::npos;
    if (!both) fprintf(stderr, "    POST gg_gemv 7: %d %s\n", st, out.c_str());
    CHECK(both);
    CHECK(complete(pp.port, p, 16) == ref);
    /* GET: the leader's values and the follower's. */
    Call g;
    bool got = false;
    if (g.send(pp.port, "/admin/fn4", "")) {
        st = g.read(&out);
        const json j = json::parse(out, nullptr, false);
        got = st == 200 && j.is_object() && j.contains("peer") && j["peer"].contains("switches") &&
              j["peer"]["switches"].value("aql_acq", -9) == 1 && !j["peer"]["switches"].contains("sched_row_cap") &&
              j["switches"].value("sched_row_cap", -9) == 48;
        if (!got) fprintf(stderr, "    GET /admin/fn4: %d %s\n", st, out.substr(0, 600).c_str());
    }
    CHECK(got);
    kill(pp.lead.child.pid, SIGTERM);
    CHECK_EQ(pp.lead.finish(60), 0);
    CHECK_EQ(pp.fol.finish(30), 0);
    const std::string flog = pp.fol.output();
    CHECK(flog.find("aql_acq=1") != std::string::npos);
    CHECK(flog.find("sched_row_cap") == std::string::npos);
    CHECK(flog.find("cannot take the leader's switch") == std::string::npos);
}

/* Robustness 5: a stopped follower (docker stop = SIGTERM) tells the leader why before it goes:
 * the leader fails its requests naming the stop, the follower exits 0. */
TEST(a_stopped_follower_tells_the_leader_and_exits_cleanly) {
    REQUIRE(world().ok);
    Serving pp;
    REQUIRE(pp.start("0,2"));
    CHECK(!complete(pp.port, prompt_ids(41, 40), 8).empty());
    kill(pp.fol.child.pid, SIGTERM);
    const int fs = pp.fol.finish(30);
    const int ls = pp.lead.finish(30);
    fprintf(stderr, "    follower exit %d, leader exit %d\n", fs, ls);
    CHECK_EQ(fs, 0);
    CHECK(ls != 0 && ls != 124);
    const bool named = pp.lead.output().find("the follower was stopped") != std::string::npos;
    if (!named) fprintf(stderr, "    leader:\n%s\n", tail(pp.lead.output()).c_str());
    CHECK(named);
}

/* ================================================================== tp8 S5: lanes */
/* The lanes' shape: six sequences (two cards a lane at three lanes), 64-token steps, so a 100-token
 * prompt prefills in two chunks and each chunk is cut into one piece a lane (RADIANCE_PP_SPLIT_MIN=16:
 * pieces of at least 16 rows on the quantum grid). */
std::vector<std::string> lane_shape(int lanes) {
    return { "--max-num-seqs", "6", "--pp-lanes", std::to_string(lanes) };
}
const std::vector<std::string> kSplitEnv = { "RADIANCE_PP_SPLIT_MIN=16" };

/* Seven alone, then eight at once of different lengths: the lanes' steps then carry different
 * sequences beside each other, and the newcomers' chunks are cut over the lanes. */
std::vector<std::vector<int32_t>> lane_set(int port) {
    std::vector<std::vector<int32_t>> out;
    for (uint64_t s : { 3, 4, 5, 11, 12, 13, 14 }) out.push_back(complete(port, prompt_ids(s, 60 + 17 * s), 24));
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 6, 7, 8, 9, 15, 16, 17, 18 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 40 + 9 * s), 20 + (int)s); }));
    for (auto& x : f) out.push_back(x.get());
    return out;
}

long long log_count(const std::string& log, const std::string& tag) {
    const size_t at = log.rfind(tag);
    if (at == std::string::npos) return -1;
    return std::atoll(log.c_str() + at + tag.size());
}

/* A sequence's tokens are the same at one lane, two and three, alone and beside other requests, as
 * one process gives them: lanes and the pieces of a newcomer's chunk change only which rows share a
 * step (step-composition invariance, on the host backend here; on cards the a1 / under-load identity
 * runs, S5-LANES.md). */
TEST(lanes_serve_the_greedy_tokens_one_process_serves) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    for (int lanes : { 1, 2, 3 }) {
        for (const char* cut : { "0,2", "1,3" }) {
            Serving pp;
            REQUIRE(pp.start(cut, lane_shape(lanes), kSplitEnv));
            const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output();
            REQUIRE_EQ(ref.size(), got.size());
            int differ = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                CHECK(!ref[i].empty());
                if (ref[i] != got[i]) ++differ;
            }
            const long long lent = log_count(llog, "newcomer split: ");
            const long long both = log_count(llog, "lane steps, ");
            const long long steps = [&] {
                const size_t at = llog.rfind(" lane steps, ");
                if (at == std::string::npos) return -1ll;
                size_t b = llog.rfind(' ', at - 1);
                return std::atoll(llog.c_str() + b + 1);
            }();
            fprintf(stderr, "    %d lane(s) W(%s): %d of %zu prompts differ; lane steps %lld, %lld with another lane in "
                            "flight at their tail; pieces lent %lld\n", lanes, cut, differ, ref.size(), steps, both, lent);
            CHECK_EQ(differ, 0);
            if (lanes > 1) {
                CHECK(llog.find(std::to_string(lanes) + " lanes, each a scheduler") != std::string::npos);
                CHECK(lent > 0);    /* newcomers' chunks were cut over the lanes */
                CHECK(both > 0);    /* and lane steps overlapped */
            } else {
                CHECK(llog.find("lanes, each a scheduler") == std::string::npos);
            }
            if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        }
    }
}

/* The KL recordings (every scored position's log-probabilities) at two and three lanes, the
 * documents' chunks cut over the lanes (and not cut: RADIANCE_PP_SPLIT=0), are one process's. */
TEST(lanes_record_what_one_process_records) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_lane_rec");
    const std::string ref = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!ref.empty());
    struct Arm { int lanes; const char* cut; bool split; };
    for (const Arm a : { Arm{ 2, "0,2", true }, Arm{ 3, "0,2", true }, Arm{ 2, "1,3", true }, Arm{ 3, "2,4", true },
                         Arm{ 2, "0,2", false } }) {
        const std::string out = d / ("pp-" + std::to_string(a.lanes) + "-" + a.cut + (a.split ? "" : "-nosplit"));
        const std::vector<std::string> env = a.split ? kSplitEnv : std::vector<std::string>{ "RADIANCE_PP_SPLIT=0" };
        const PipeRun pr = record_pipeline(out, a.cut, env, lane_shape(a.lanes), lane_shape(a.lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    %d lanes W(%s): leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", a.lanes,
                    a.cut, pr.leader, pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(ref, out, &why);
        const long long pieces = log_count(pr.llog, "newcomer split: ");
        fprintf(stderr, "    %d lanes W(%s) split %s: recording %s; pieces lent %lld\n", a.lanes, a.cut,
                a.split ? "on" : "off", same ? "identical" : why.c_str(), pieces);
        CHECK(same);
        CHECK(pr.llog.find(std::to_string(a.lanes) + " lanes, each a scheduler") != std::string::npos);
        if (a.split) CHECK(pieces > 0);
        else CHECK_EQ(pieces, 0);
    }
}

/* Stage busy stamps (RADIANCE_PP_STAMPS): each node's pass intervals, busy only while it computes (the
 * leader's head up to X1's publish, its tail from X2's arrival; the follower's pass from X1's arrival to
 * X2's publish -- the host backend's stage ops note those moments). */
struct Busy { double leader = 0, follower = 0, both = 0, window_ms = 0; size_t n_l = 0, n_f = 0; };
Busy busy_of(const std::string& prefix) {
    using Iv = std::vector<std::pair<uint64_t, uint64_t>>;
    auto load = [](const std::string& path, std::initializer_list<const char*> kinds) {
        Iv iv;
        std::istringstream in(tiny::read_text(path));
        std::string kind;
        int lane = 0, step = 0;
        unsigned long long b = 0, e = 0;
        while (in >> kind >> lane >> step >> b >> e) {
            bool want = false;
            for (const char* k : kinds) want = want || kind == k;
            if (want && e > b) iv.push_back({ b, e });
        }
        std::sort(iv.begin(), iv.end());
        Iv u;
        for (const auto& x : iv) {
            if (!u.empty() && x.first <= u.back().second) u.back().second = std::max(u.back().second, x.second);
            else u.push_back(x);
        }
        return u;
    };
    const Iv L = load(prefix + ".leader", { "head", "tail" }), F = load(prefix + ".follower", { "mid" });
    Busy r;
    r.n_l = L.size();
    r.n_f = F.size();
    if (L.empty() || F.empty()) return r;
    const uint64_t w0 = std::min(L.front().first, F.front().first), w1 = std::max(L.back().second, F.back().second);
    const double W = (double)(w1 - w0);
    auto len = [](const Iv& v) { uint64_t t = 0; for (const auto& x : v) t += x.second - x.first; return (double)t; };
    uint64_t both = 0;
    for (size_t i = 0, j = 0; i < L.size() && j < F.size();) {
        const uint64_t lo = std::max(L[i].first, F[j].first), hi = std::min(L[i].second, F[j].second);
        if (hi > lo) both += hi - lo;
        if (L[i].second < F[j].second) ++i; else ++j;
    }
    r.leader = len(L) / W;
    r.follower = len(F) / W;
    r.both = (double)both / W;
    r.window_ms = W / 1e6;
    return r;
}

/* The point of lanes: with one lane the two nodes work in turn (never both at once); with two, both
 * compute at once for much of the run, and the same requests finish sooner. Host backend, the tiny
 * model: the fractions are this machine's, the claim is their order. */
TEST(lanes_keep_both_stages_busy_at_once) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_lane_busy");
    double tps[3] = {};
    Busy got[3];
    for (int lanes : { 1, 2 }) {
        const std::string prefix = d / ("st" + std::to_string(lanes));
        Serving pp;
        REQUIRE(pp.start("0,2", lane_shape(lanes), { "RADIANCE_PP_SPLIT_MIN=16", "RADIANCE_PP_STAMPS=" + prefix }));
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<std::vector<int32_t>>> f;
        for (uint64_t s = 0; s < 12; ++s)
            f.push_back(std::async(std::launch::async, [&pp, s] { return complete(pp.port, prompt_ids(40 + s, 40 + 7 * s), 48); }));
        size_t tok = 0;
        for (auto& x : f) tok += x.get().size();
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        tps[lanes] = (double)tok / sec;
        got[lanes] = busy_of(prefix);
        fprintf(stderr, "    %d lane(s): %zu tokens in %.2f s = %.1f tok/s; window %.0f ms: leader busy %.3f, follower busy "
                        "%.3f, both at once %.3f (%zu / %zu intervals)\n", lanes, tok, sec, tps[lanes], got[lanes].window_ms,
                got[lanes].leader, got[lanes].follower, got[lanes].both, got[lanes].n_l, got[lanes].n_f);
        CHECK(got[lanes].n_l > 0 && got[lanes].n_f > 0);
    }
    CHECK(got[1].both < 0.02);          /* one lane: the nodes take turns */
    CHECK(got[2].both > 0.25);          /* two: both busy at once */
    CHECK(got[2].both > got[1].both + 0.2);
    CHECK(tps[2] > tps[1]);
}

}  /* namespace */

RAD_TEST_MAIN()
