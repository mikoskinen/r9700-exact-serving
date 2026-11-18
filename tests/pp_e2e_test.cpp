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

/* GET `path` (tp8 S6: /metrics) */
std::string get(int port, const std::string& path) {
    const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::string raw;
    if (connect(fd, (sockaddr*)&a, sizeof a) == 0) {
        const std::string req = "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
        if (::send(fd, req.data(), req.size(), MSG_NOSIGNAL) == (ssize_t)req.size()) {
            char buf[16384];
            for (;;) {
                const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) break;
                raw.append(buf, (size_t)n);
            }
        }
    }
    close(fd);
    const size_t h = raw.find("\r\n\r\n");
    return h == std::string::npos ? "" : raw.substr(h + 4);
}
/* A Prometheus counter's value in a /metrics body (the first sample of the name), -1 when absent. */
double metric(const std::string& body, const std::string& name) {
    size_t at = 0;
    while ((at = body.find(name, at)) != std::string::npos) {
        if (at == 0 || body[at - 1] == '\n') {
            const size_t sp = body.find(' ', at);
            if (sp != std::string::npos) return std::atof(body.c_str() + sp + 1);
        }
        at += name.size();
    }
    return -1;
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
    /* tp8 loop D20 (loop FINDINGS F57): the leader's own drafter knobs -- D18's short block (plugin env + --draft-short-from)
     * and D19's tiers -- are each node's own: a pair with them on the leader only starts and serves (here no drafter: the
     * tiny model's engine warns and keeps its block) */
    const PipeRun po = record_pipeline(d / "rec3", "0,2", {}, {}, { "--draft-short-from", "12" },
                                       { "RADIANCE_DS41_DSPARK_SHORT=1", "RADIANCE_DRAFT_SHORT_TIERS=12:1,6:3" });
    const bool started = po.leader == 0 && po.follower == 0 && po.llog.find("the nodes differ") == std::string::npos;
    if (!started) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(po.llog).c_str(), tail(po.flog).c_str());
    CHECK(started);
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
/* RADIANCE_PP_PICK=rr: the prompts sent one at a time go to lane 0, 1, 2, ... in turn, so every lane serves alone too */
const std::vector<std::string> kSplitEnv = { "RADIANCE_PP_SPLIT_MIN=16", "RADIANCE_PP_PICK=rr" };

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

/* A sequence's tokens are the same at one lane, two and three, alone (each lane in turn) and beside other requests, as
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

/* ------------------------------------------------------------------ tp8 S6 */
/* A wrap-around cut whose X1 carries a KV layer's entries: the follower stores them into its copy of the leader's
 * layer h - 1 cache with the same kv_store at the same slots (RADIANCE_LLAMA_PP_KVCOPY, the host stand-in for
 * deepseek41's copied sources), and the core's KV-copy check (RADIANCE_PP_KV_CHECK=1: every trunk step) finds the copy
 * byte-equal to the owner's every time; the answers stay one process's. */
TEST(a_wrap_around_cut_keeps_its_kv_copy_byte_equal_to_the_owners) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_kvc");
    const std::string ref = record_single(d / "single");
    REQUIRE(!ref.empty());
    for (const char* cut : { "1,3", "2,4", "1,4" }) {
        std::vector<std::string> env = { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" };
        if (const char* dd = std::getenv("PP_E2E_KVDUMP")) env.push_back(std::string("RADIANCE_PP_KV_CHECK_DUMP=") + dd);
        const std::string out = d / (std::string("kvc-") + cut);
        const PipeRun pr = record_pipeline(out, cut, env, {}, {}, env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    W(%s): leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", cut, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(ref, out, &why);
        if (!same) fprintf(stderr, "    W(%s): %s\n", cut, why.c_str());
        CHECK(same);
        size_t eq = 0, at = 0;
        while ((at = pr.flog.find("on both nodes (", at)) != std::string::npos) { ++eq; ++at; }
        fprintf(stderr, "    W(%s): %zu KV-copy checks equal on both nodes\n", cut, eq);
        CHECK(eq >= 4);
        const size_t bad = pr.flog.find("NOT the owner's bytes");
        if (bad != std::string::npos) {
            const size_t a0 = pr.flog.rfind('\n', bad);
            fprintf(stderr, "    W(%s): %s\n", cut, pr.flog.substr(a0 + 1, pr.flog.find('\n', bad) - a0 - 1).c_str());
        }
        CHECK(bad == std::string::npos);
        CHECK(pr.llog.find("RADIANCE_PP_KV_CHECK=1: the copied KV layers (kv_attn layer") != std::string::npos);
    }
}

/* The check's own test: the follower leaves its copy store out at one pass; the next check says so. */
TEST(a_left_out_copy_store_is_caught_by_the_kv_check) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_kvbad");
    const std::vector<std::string> env = { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" };
    const PipeRun pr = record_pipeline(d / "bad", "1,3", cat(env, { "RADIANCE_PP_TEST_KVCOPY_SKIP=2" }), {}, {}, env);
    CHECK_EQ(pr.leader, 0);
    const bool caught = pr.flog.find("NOT the owner's bytes") != std::string::npos;
    if (!caught) fprintf(stderr, "    follower:\n%s\n", tail(pr.flog, 30).c_str());
    CHECK(caught);
}

/* A prefix hit across the cut: the second request reuses the first one's cached blocks (the attention-cached
 * counter moves on the leader), and a wrap-around pipeline answers both with one process's greedy tokens -- the
 * follower's layers read the same block ids, which its own earlier steps wrote. */
TEST(a_prefix_hit_across_the_cut_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<int32_t> base = prompt_ids(31, 300);
    std::vector<int32_t> longer = base;
    const std::vector<int32_t> more = prompt_ids(32, 40);
    longer.insert(longer.end(), more.begin(), more.end());
    std::vector<std::vector<int32_t>> ref, got;
    double hit_ref = -1, hit_pp = -1;
    for (const char* cut : { "", "1,3" }) {
        Serving sv;
        REQUIRE(sv.start(cut));
        std::vector<std::vector<int32_t>>& out = *cut ? got : ref;
        out.push_back(complete(sv.port, base, 16));
        out.push_back(complete(sv.port, longer, 16));
        out.push_back(complete(sv.port, base, 8));
        (*cut ? hit_pp : hit_ref) = metric(get(sv.port, "/metrics"), "radiance:attn_cached_tokens_total");
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        if (*cut) CHECK_EQ(sv.fol.finish(30), 0);
    }
    fprintf(stderr, "    attention-cached prompt tokens: one process %.0f, W(1,3) %.0f\n", hit_ref, hit_pp);
    CHECK(hit_ref > 0);
    CHECK_EQ(hit_pp, hit_ref);
    REQUIRE_EQ(ref.size(), got.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        CHECK(!ref[i].empty());
        if (ref[i] != got[i]) fprintf(stderr, "    request %zu: the pipeline's tokens differ\n", i);
        CHECK(ref[i] == got[i]);
    }
}

/* ------------------------------------------------------------------ tp8 M56: lanes x a general cut */
namespace {
const std::vector<std::string> kLanes2 = { "--max-num-seqs", "6", "--pp-lanes", "2" };

/* Every prefill entry ([ctx, ctx + q) with q > 1) of a RADIANCE_LOG_STEPS log, in position order: one process's
 * "step N prefill ...  [0] slot=s q=Q ctx=C", a lane leader's "lane L step N ...". One request runs, so every one is
 * its prompt's. */
std::vector<std::pair<int, int>> chunks_of(const std::string& log) {
    std::vector<std::pair<int, int>> v;
    size_t at = 0;
    while ((at = log.find(" q=", at)) != std::string::npos) {
        const int q = std::atoi(log.c_str() + at + 3);
        const size_t c = log.find(" ctx=", at);
        if (c == std::string::npos) break;
        const int ctx = std::atoi(log.c_str() + c + 5);
        if (q > 1) v.emplace_back(ctx, ctx + q);
        at = c;
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}
}  /* namespace */

/* Two lanes at the wrap-around cuts W(1,3) and W(2,4), the KV copy on (the follower's copy of the leader's layer
 * h - 1, written from X1) and checked: the recordings are one process's, every check finds the copy byte-equal, and the
 * documents' chunks went over both lanes. */
TEST(two_lanes_at_a_wrap_around_cut_record_one_process_and_keep_the_kv_copy) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_m56_kvc");
    const std::string ref = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!ref.empty());
    for (const char* cut : { "1,3", "2,4" }) {
        const std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        const std::string out = d / (std::string("l2-") + cut);
        const PipeRun pr = record_pipeline(out, cut, env, kLanes2, kLanes2, env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    W(%s): leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", cut, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(ref, out, &why);
        size_t eq = 0, at = 0;
        while ((at = pr.flog.find("on both nodes (", at)) != std::string::npos) { ++eq; ++at; }
        const long long pieces = log_count(pr.llog, "newcomer split: ");
        fprintf(stderr, "    2 lanes W(%s): recording %s; %zu KV-copy checks equal; chunks lent %lld\n", cut,
                same ? "identical" : why.c_str(), eq, pieces);
        CHECK(same);
        CHECK(pr.llog.find("2 lanes, each a scheduler") != std::string::npos);
        CHECK(pieces > 0);
        CHECK(eq >= 2);
        CHECK(pr.flog.find("NOT the owner's bytes") == std::string::npos);
    }
}

/* A prefix hit with two lanes: requests alone go to lane 0, 1, 0 in turn (RADIANCE_PP_PICK=rr), so the hits are on
 * blocks another lane's request wrote; the attention-cached counter after them is one process's. Then three at once
 * (which of them hit then depends on when each lane admits: tokens only). Every greedy token is one process's, at W(0,2)
 * and the wrap-around W(1,3) with its KV copy and check. */
TEST(a_prefix_hit_with_two_lanes_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<int32_t> base = prompt_ids(31, 300);
    std::vector<int32_t> longer = base;
    const std::vector<int32_t> more = prompt_ids(32, 40);
    longer.insert(longer.end(), more.begin(), more.end());
    const std::vector<int32_t> other = prompt_ids(33, 200);
    auto run = [&](int port, double* hit_alone) {
        std::vector<std::vector<int32_t>> out;
        out.push_back(complete(port, base, 16));
        out.push_back(complete(port, longer, 16));
        out.push_back(complete(port, base, 8));
        *hit_alone = metric(get(port, "/metrics"), "radiance:attn_cached_tokens_total");
        std::vector<std::future<std::vector<int32_t>>> f;
        for (const std::vector<int32_t>* p : std::vector<const std::vector<int32_t>*>{ &longer, &other, &base })
            f.push_back(std::async(std::launch::async, [port, p] { return complete(port, *p, 12); }));
        for (auto& x : f) out.push_back(x.get());
        return out;
    };
    std::vector<std::vector<int32_t>> ref;
    double hit_ref = -1;
    {
        Serving sv;
        REQUIRE(sv.start("", { "--max-num-seqs", "6" }));
        ref = run(sv.port, &hit_ref);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
    }
    CHECK(hit_ref > 0);
    for (const char* cut : { "0,2", "1,3" }) {
        Serving sv;
        const std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        REQUIRE(sv.start(cut, kLanes2, env));
        double hit = -1;
        const std::vector<std::vector<int32_t>> got = run(sv.port, &hit);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        const std::string flog = sv.fol.output();
        int differ = 0;
        REQUIRE_EQ(ref.size(), got.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        fprintf(stderr, "    2 lanes W(%s): attention-cached prompt tokens alone %.0f (one process %.0f); %d of %zu requests "
                        "differ\n", cut, hit, hit_ref, differ, ref.size());
        CHECK_EQ(hit, hit_ref);
        CHECK_EQ(differ, 0);
        CHECK(sv.lead.output().find("2 lanes, each a scheduler") != std::string::npos);
        CHECK(flog.find("NOT the owner's bytes") == std::string::npos);
    }
}

/* A cold multi-chunk prompt at two lanes: one process's greedy tokens AND one process's chunk plan -- every chunk the
 * leader's lanes ran of it is one an idle one-process engine runs (RADIANCE_LOG_STEPS), and the chunks went over both
 * lanes (tp8 S4 hardware: pieces cut inside one process's chunks moved the cold texts). */
TEST(a_cold_prompt_at_two_lanes_is_one_processes_tokens_and_chunk_plan) {
    REQUIRE(world().ok);
    for (size_t n : { 700, 1000 }) {
        const std::vector<int32_t> p = prompt_ids(41 + n, n);
        std::vector<int32_t> ref, got;
        std::vector<std::pair<int, int>> ref_plan, got_plan;
        {
            Serving sv;
            REQUIRE(sv.start("", { "--max-num-seqs", "6" }, { "RADIANCE_LOG_STEPS=1" }));
            ref = complete(sv.port, p, 24);
            kill(sv.lead.child.pid, SIGTERM);
            CHECK_EQ(sv.lead.finish(60), 0);
            ref_plan = chunks_of(sv.lead.output());
        }
        for (const char* cut : { "0,2", "1,3" }) {
            Serving sv;
            REQUIRE(sv.start(cut, kLanes2, cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1" })));
            got = complete(sv.port, p, 24);
            kill(sv.lead.child.pid, SIGTERM);
            CHECK_EQ(sv.lead.finish(60), 0);
            CHECK_EQ(sv.fol.finish(30), 0);
            const std::string llog = sv.lead.output();
            got_plan = chunks_of(llog);
            const long long lent = log_count(llog, "newcomer split: ");
            fprintf(stderr, "    %zu-token prompt, 2 lanes W(%s): %zu chunks (one process %zu), plan %s, tokens %s, chunks "
                            "lent %lld\n", p.size(), cut, got_plan.size(), ref_plan.size(),
                    got_plan == ref_plan ? "identical" : "DIFFERENT", got == ref ? "identical" : "DIFFERENT", lent);
            CHECK(ref_plan.size() >= 4);
            CHECK(got_plan == ref_plan);
            CHECK(!ref.empty());
            CHECK(got == ref);
            CHECK(lent > 0);
            CHECK(llog.find("(lane 0's)") != std::string::npos);   /* a chunk ran in the other lane */
            /* tp8 prefill P0 / M2: the prompt's lanes line and the run's stall line, for the test side's one-run reading */
            const size_t pl = llog.find("lanes: request ");
            const size_t gl = llog.find("lane-step stalls over the run");
            CHECK(pl != std::string::npos);
            CHECK(gl != std::string::npos);
            if (pl != std::string::npos) fprintf(stderr, "    %s\n", llog.substr(pl, llog.find('\n', pl) - pl).c_str());
            if (gl != std::string::npos) fprintf(stderr, "    %s\n", llog.substr(gl, llog.find('\n', gl) - gl).c_str());
        }
    }
}

/* tp8 loop P7 (RADIANCE_PP_PF_TIMELINE=1 on the leader): a cold multi-chunk prompt at 2 and 3 lanes with the prefill
 * timeline on serves one process's tokens and plan, and the leader prints the prompt's timeline line once its first
 * token is in -- as many chunks as the plan has, every head after the first read from its events (the host backend's
 * events carry host stamps), nothing dropped -- and its per-chunk lines. tp8 loop P22: the follower, with the same switch,
 * prints its side -- the prompt's passes (as many as the chunks, every one read), their rows and X1 waits. */
TEST(the_prefill_timeline_names_a_cold_prompts_chunks_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(977, 1000);
    std::vector<int32_t> ref;
    std::vector<std::pair<int, int>> ref_plan;
    {
        Serving sv;
        REQUIRE(sv.start("", { "--max-num-seqs", "6" }, { "RADIANCE_LOG_STEPS=1" }));
        ref = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        ref_plan = chunks_of(sv.lead.output());
    }
    CHECK(ref_plan.size() >= 4);
    for (const char* lanes : { "2", "3" })
    for (const char* cut : { "0,2", "1,3" }) {
        Serving sv;
        REQUIRE(sv.start(cut, { "--max-num-seqs", "6", "--pp-lanes", lanes },
                         cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_PP_PF_TIMELINE=1" })));
        const std::vector<int32_t> got = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        const std::string llog = sv.lead.output();
        CHECK(chunks_of(llog) == ref_plan);
        CHECK(!ref.empty());
        CHECK(got == ref);
        CHECK(llog.find("RADIANCE_PP_PF_TIMELINE=1 -- a lone prefill's chunks are timed") != std::string::npos);
        const size_t tl = llog.find("prefill timeline (RADIANCE_PP_PF_TIMELINE=1");
        CHECK(tl != std::string::npos);
        if (tl == std::string::npos) continue;
        const std::string line = llog.substr(tl, llog.find('\n', tl) - tl);
        char want_n[96], want_h[48];
        std::snprintf(want_n, sizeof want_n, ", %zu chunks, %zu rows;", ref_plan.size(), p.size());
        std::snprintf(want_h, sizeof want_h, "(%zu; ", ref_plan.size());
        CHECK(line.find(want_n) != std::string::npos);
        CHECK(line.find(want_h) != std::string::npos);   /* every head read */
        CHECK(line.find("events dropped 0, unread 0") != std::string::npos);
        CHECK(llog.find("prefill timeline request ") != std::string::npos);   /* the per-chunk lines */
        fprintf(stderr, "    %s lanes W(%s): tokens %s, plan %s\n    %s\n", lanes, cut, got == ref ? "identical" : "DIFFERENT",
                chunks_of(llog) == ref_plan ? "identical" : "DIFFERENT", line.c_str());
        /* tp8 loop P22: the follower's side -- the same switch there times the prompt's lone prefill passes (as many as the
         * plan's chunks, every one read, its rows), printed once the prompt's first decode step arrived */
        const std::string flog = sv.fol.output();
        CHECK(flog.find("follower: RADIANCE_PP_PF_TIMELINE=1 -- a lone prefill's passes are timed") != std::string::npos);
        const size_t ft = flog.find("follower prefill timeline (RADIANCE_PP_PF_TIMELINE=1");
        CHECK(ft != std::string::npos);
        if (ft == std::string::npos) continue;
        const std::string fline = flog.substr(ft, flog.find('\n', ft) - ft);
        char fwant[96], fread[48];
        std::snprintf(fwant, sizeof fwant, "prompt 1, %zu passes, %zu rows (context 0 .. %zu)", ref_plan.size(), p.size(), p.size());
        std::snprintf(fread, sizeof fread, "(%zu read;", ref_plan.size());
        CHECK(fline.find(fwant) != std::string::npos);
        CHECK(fline.find(fread) != std::string::npos);
        CHECK(fline.find("events dropped 0, unread 0") != std::string::npos);
        CHECK(flog.find("follower prefill timeline prompt 1 passes 0-") != std::string::npos);
        fprintf(stderr, "    %s\n", fline.c_str());
    }
}

/* tp8 dev D8 (RADIANCE_PP_NODE_IDLE=1 on both nodes): a cold multi-chunk prompt and its decode at 2 lanes, W(0,2) and
 * W(1,3), serve one process's tokens, and each node prints its node-idle line for the run at stop -- parts read from
 * their events (none dropped or unread), rank 0's receives of the other node's crossing counted (the leader's X2, the
 * follower's X1: at W(0,2) the embedding's output), parts + gaps within the wall. */
TEST(the_node_idle_line_times_both_nodes_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(978, 700);
    std::vector<int32_t> ref;
    {
        Serving sv;
        REQUIRE(sv.start("", { "--max-num-seqs", "6" }, {}));
        ref = complete(sv.port, p, 32);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
    }
    CHECK(!ref.empty());
    for (const char* cut : { "0,2", "1,3" }) {
        Serving sv;
        REQUIRE(sv.start(cut, { "--max-num-seqs", "6", "--pp-lanes", "2" }, cat(kSplitEnv, { "RADIANCE_PP_NODE_IDLE=1" })));
        const std::vector<int32_t> got = complete(sv.port, p, 32);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        CHECK(got == ref);
        for (const auto& [log, who] : { std::pair<std::string, const char*>{ sv.lead.output(), "leader" },
                                        std::pair<std::string, const char*>{ sv.fol.output(), "follower" } }) {
            CHECK(log.find("RADIANCE_PP_NODE_IDLE=1 -- rank 0's card time") != std::string::npos);
            const std::string key = std::string("two-node pipeline, ") + who + " node idle (RADIANCE_PP_NODE_IDLE=1";
            const size_t at = log.find(key);
            CHECK(at != std::string::npos);
            if (at == std::string::npos) continue;
            const size_t run = log.find("over the run (", at);
            CHECK(run != std::string::npos);
            if (run == std::string::npos) continue;
            const std::string line = log.substr(run, log.find('\n', run) - run);
            double wall = -1, cov = -1, left = -1, part = -1, ppct = -1, wait = -1, wpct = -1, gap = -1;
            long long idles = -1, parts = -1, recvs = -1;
            const int got_n = std::sscanf(line.c_str(), "over the run (%lf s wall, %lf s covered; %lld gaps > %*s s left out, %lf s): "
                                          "in parts %lf s (%lf %%, %lld parts), of which waiting on the card for the other node's "
                                          "%*s %lf s (%lf %%, %lld receives); between parts %lf s",
                                          &wall, &cov, &idles, &left, &part, &ppct, &parts, &wait, &wpct, &recvs, &gap);
            CHECK_EQ(got_n, 11);
            CHECK(parts > 0);
            CHECK(recvs > 0);   /* the leader's X2 every tail; the follower's X1 every pass (h = 0: the embedding) */
            CHECK(wait >= 0 && wait <= part + 0.05);
            CHECK(part >= 0 && gap >= 0 && part + gap <= cov + 0.15 && cov <= wall + 0.05);
            CHECK_EQ(idles, 0);   /* one request, no pause */
            CHECK(line.find("events dropped 0, unread 0") != std::string::npos);
            fprintf(stderr, "    W(%s) %s: tokens %s\n    %s\n", cut, who, got == ref ? "identical" : "DIFFERENT", line.c_str());
        }
    }
}

/* (review of S6) The prefix hit across the cut with the KV copy on (RADIANCE_LLAMA_PP_KVCOPY + RADIANCE_PP_KV_CHECK=1):
 * the second request's hit reuses blocks whose copied layer the follower wrote in the first request's steps, and every
 * check over the handed-out blocks finds the copy = the owner's; and RADIANCE_PP_PREFIX_CACHE=0 (S4's rule, both nodes):
 * no hit at all, the same tokens (the chunk plan is the cache's either way). */
TEST(a_prefix_hit_with_the_kv_copy_and_the_cache_off_switch) {
    REQUIRE(world().ok);
    const std::vector<int32_t> base = prompt_ids(41, 300);
    std::vector<int32_t> longer = base;
    const std::vector<int32_t> more = prompt_ids(42, 40);
    longer.insert(longer.end(), more.begin(), more.end());
    struct Arm { const char* cut; std::vector<std::string> env; };
    const Arm arms[] = { { "", {} },
                         { "1,3", { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" } },
                         { "1,3", { "RADIANCE_PP_PREFIX_CACHE=0" } } };
    std::vector<std::vector<int32_t>> out[3];
    double hits[3] = { -1, -1, -1 };
    std::string flog[3], llog[3];
    for (int k = 0; k < 3; ++k) {
        Serving sv;
        REQUIRE(sv.start(arms[k].cut, {}, arms[k].env));
        out[k].push_back(complete(sv.port, base, 16));
        out[k].push_back(complete(sv.port, longer, 16));
        out[k].push_back(complete(sv.port, base, 8));
        hits[k] = metric(get(sv.port, "/metrics"), "radiance:attn_cached_tokens_total");
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        llog[k] = sv.lead.output();
        if (*arms[k].cut) {
            CHECK_EQ(sv.fol.finish(30), 0);
            flog[k] = sv.fol.output();
        }
    }
    fprintf(stderr, "    attention-cached prompt tokens: one process %.0f, W(1,3) + KV copy %.0f, W(1,3) cache off %.0f\n",
            hits[0], hits[1], hits[2]);
    CHECK(hits[0] > 0);
    CHECK_EQ(hits[1], hits[0]);
    CHECK_EQ(hits[2], 0.0);
    size_t eq = 0, at = 0;
    while ((at = flog[1].find("on both nodes (", at)) != std::string::npos) { ++eq; ++at; }
    fprintf(stderr, "    W(1,3) + KV copy: %zu KV-copy checks equal\n", eq);
    CHECK(eq >= 4);
    CHECK(flog[1].find("NOT the owner's bytes") == std::string::npos);
    CHECK(llog[2].find("RADIANCE_PP_PREFIX_CACHE=0") != std::string::npos);
    CHECK(flog[2].find("RADIANCE_PP_PREFIX_CACHE=0") != std::string::npos);
    for (int k = 1; k < 3; ++k) {
        REQUIRE_EQ(out[k].size(), out[0].size());
        for (size_t i = 0; i < out[0].size(); ++i) {
            CHECK(!out[0][i].empty());
            if (out[k][i] != out[0][i]) fprintf(stderr, "    arm %d request %zu: the pipeline's tokens differ\n", k, i);
            CHECK(out[k][i] == out[0][i]);
        }
    }
}

/* ------------------------------------------------------------------ tp8 dev 1a */
/* RADIANCE_PP_FOLLOWER_EMBED=1: at W(0, t) the follower looks the step's embedding up itself (the same op over the same
 * table) and the leader's head part issues neither the lookup nor X1. Tokens at one, two and three lanes, alone and
 * beside other requests, and the KL recordings at two and three lanes, are one process's; the follower never takes an
 * X1 (no "mid" stamp: those start at X1's arrival); at W(1,3) X1 is layer 0's residual and the setting is not taken. */
TEST(the_followers_own_embedding_serves_and_records_what_one_process_does) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_fe");
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; };
    for (const Arm a : { Arm{ 1, "0,2" }, Arm{ 2, "0,2" }, Arm{ 3, "0,2" }, Arm{ 2, "0,3" }, Arm{ 2, "1,3" } }) {
        const std::string prefix = d / ("st-" + std::to_string(a.lanes) + "-" + a.cut);
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        env.push_back("RADIANCE_PP_STAMPS=" + prefix);
        Serving pp;
        REQUIRE(pp.start(a.cut, lane_shape(a.lanes), env));
        const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        const bool h0 = a.cut[0] == '0';
        size_t mids = 0, heads = 0;
        {
            std::istringstream in(tiny::read_text(prefix + ".follower"));
            std::string line;
            while (std::getline(in, line)) mids += line.rfind("mid ", 0) == 0;
            std::istringstream il(tiny::read_text(prefix + ".leader"));
            while (std::getline(il, line)) heads += line.rfind("head ", 0) == 0;
        }
        fprintf(stderr, "    follower embeds, %d lane(s) W(%s): %d of %zu prompts differ; leader head parts %zu, follower "
                        "passes from an X1 %zu; pieces lent %lld\n", a.lanes, a.cut, differ, ref.size(), heads, mids,
                log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        if (h0) {
            CHECK(llog.find("RADIANCE_PP_FOLLOWER_EMBED=1 -- the follower looks") != std::string::npos);
            CHECK(flog.find("RADIANCE_PP_FOLLOWER_EMBED=1 -- the follower looks") != std::string::npos);
            CHECK_EQ(mids, (size_t)0);     /* the follower never started from an X1 */
            if (a.lanes > 1) CHECK(heads > 0);
        } else {
            CHECK(llog.find("RADIANCE_PP_FOLLOWER_EMBED=1 is not taken at W(1,3)") != std::string::npos);
            CHECK(mids > 0);
        }
    }
    /* the KL recordings (every scored position's log-probabilities) */
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    for (const int lanes : { 1, 2, 3 }) {
        const std::string out = d / ("pp-" + std::to_string(lanes));
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        const PipeRun pr = record_pipeline(out, "0,2", env, lane_shape(lanes), lane_shape(lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    %d lanes: leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", lanes, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    follower embeds, %d lane(s) W(0,2): recording %s; pieces lent %lld\n", lanes,
                same ? "identical" : why.c_str(), log_count(pr.llog, "newcomer split: "));
        CHECK(same);
    }
}

/* ------------------------------------------------------------------ tp8 dev 1b */
/* RADIANCE_PP_TAIL_AHEAD=1 with the follower's own embedding: a lane step's model part goes out before the lane ahead of
 * it is collected, its sampler and drafter after that commit. Tokens at two and three lanes at W(0,2) / W(0,3), alone and
 * beside other requests, and the KL recordings at two and three lanes, are one process's; the leader counts the parts
 * issued ahead. Without RADIANCE_PP_FOLLOWER_EMBED=1, or at W(1,3), the setting is not taken. */
TEST(tails_issued_ahead_serve_and_record_what_one_process_does) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_ta");
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; bool fe; };
    for (const Arm a : { Arm{ 2, "0,2", true }, Arm{ 3, "0,2", true }, Arm{ 2, "0,3", true }, Arm{ 2, "1,3", true },
                         Arm{ 2, "0,2", false } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_TAIL_AHEAD=1");
        if (a.fe) env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        Serving pp;
        REQUIRE(pp.start(a.cut, lane_shape(a.lanes), env));
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
        const bool taken = a.fe && a.cut[0] == '0';
        const long long ahead = log_count(llog, "two-node pipeline: ");   /* the last such line: the count below */
        const size_t at = llog.rfind(" lane steps' model parts were issued ahead");
        long long n_ahead = -1;
        if (at != std::string::npos) {
            const size_t b = llog.rfind("two-node pipeline: ", at);
            n_ahead = b == std::string::npos ? -1 : std::atoll(llog.c_str() + b + 19);
        }
        (void)ahead;
        fprintf(stderr, "    tail ahead, %d lane(s) W(%s)%s: %d of %zu prompts differ; model parts issued ahead %lld; "
                        "pieces lent %lld\n", a.lanes, a.cut, a.fe ? " + follower embeds" : "", differ, ref.size(), n_ahead,
                log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        if (taken) {
            CHECK(llog.find("RADIANCE_PP_TAIL_AHEAD=1 -- a lane step's model part") != std::string::npos);
            CHECK(n_ahead > 0);
        } else {
            CHECK(llog.find("RADIANCE_PP_TAIL_AHEAD=1 is not taken") != std::string::npos);
            CHECK_EQ(n_ahead, -1ll);
        }
    }
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    for (const int lanes : { 2, 3 }) {
        const std::string out = d / ("pp-" + std::to_string(lanes));
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        env.push_back("RADIANCE_PP_TAIL_AHEAD=1");
        const PipeRun pr = record_pipeline(out, "0,2", env, lane_shape(lanes), lane_shape(lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    %d lanes: leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", lanes, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    tail ahead + follower embeds, %d lane(s) W(0,2): recording %s; pieces lent %lld\n", lanes,
                same ? "identical" : why.c_str(), log_count(pr.llog, "newcomer split: "));
        CHECK(same);
        CHECK(pr.llog.find("lane steps' model parts were issued ahead") != std::string::npos);
    }
}

/* ------------------------------------------------------------------ tp8 dev D7 */
/* RADIANCE_PP_DECODE_FIRST=1: each planning round puts the lanes with decode rows alone before those with prompt rows. Tokens
 * at two and three lanes (alone, every lane in turn, and beside other requests, newcomers' chunks lent), with and without the
 * follower's own embedding + tails ahead, and the KL recordings at two and three lanes are one process's; the leader counts
 * the rounds it reordered. */
TEST(decode_lanes_first_serve_and_record_what_one_process_does) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_df");
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; bool ta; };
    for (const Arm a : { Arm{ 2, "0,2", false }, Arm{ 3, "0,2", false }, Arm{ 2, "1,3", false }, Arm{ 2, "0,2", true },
                         Arm{ 3, "0,3", true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_DECODE_FIRST=1");
        if (a.ta) {
            env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
            env.push_back("RADIANCE_PP_TAIL_AHEAD=1");
        }
        Serving pp;
        REQUIRE(pp.start(a.cut, lane_shape(a.lanes), env));
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
        long long rounds = -1;
        if (const size_t at = llog.rfind(" planning rounds put a decode lane"); at != std::string::npos) {
            const size_t b = llog.rfind("two-node pipeline: ", at);
            rounds = b == std::string::npos ? -1 : std::atoll(llog.c_str() + b + 19);
        }
        const long long held = log_count(llog, "lane with prompt rows, ");
        fprintf(stderr, "    decode first, %d lane(s) W(%s)%s: %d of %zu prompts differ; rounds reordered %lld, prompt lanes held %lld; "
                        "pieces lent %lld\n", a.lanes, a.cut, a.ta ? " + follower embeds + tails ahead" : "", differ, ref.size(),
                rounds, held, log_count(llog, "newcomer split: "));
        CHECK(held > 0);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        CHECK(llog.find("RADIANCE_PP_DECODE_FIRST=1 -- each round plans") != std::string::npos);
        CHECK(rounds > 0);
    }
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    for (const int lanes : { 2, 3 }) {
        const std::string out = d / ("pp-" + std::to_string(lanes));
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_DECODE_FIRST=1");
        const PipeRun pr = record_pipeline(out, "0,2", env, lane_shape(lanes), lane_shape(lanes), env);
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    decode first, %d lane(s) W(0,2): recording %s; pieces lent %lld\n", lanes,
                same ? "identical" : why.c_str(), log_count(pr.llog, "newcomer split: "));
        CHECK(same);
    }
}

/* ------------------------------------------------------------------ tp8 dev M4 */
/* RADIANCE_PP_SEG_LAYERS=1 (the follower's passes of >= RADIANCE_PP_SEG_ROWS rows in one-layer segments, each boundary's stream
 * saved and restored on its rank through the stage ops' loop mode, other lanes' passes between the segments) with
 * RADIANCE_PP_READY_FIRST=1 (the leader takes a later lane's landed tail before a long pass's): tokens at two and three lanes
 * and the KL recordings are one process's; the follower counts its split passes and the passes it ran between segments
 * (RADIANCE_PP_SEG_TEST_WAIT_MS holds each gap open long enough for another lane's frame to arrive on the host backend). */
TEST(follower_segments_serve_and_record_what_one_process_does) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_seg");
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    const std::vector<std::string> seg_env = { "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" };
    /* the follower's last "N passes ran in M segments, K other passes between their segments, H held" line */
    auto seg_count = [](const std::string& log, const char* what) {
        const size_t at = log.rfind(" passes ran in ");
        if (at == std::string::npos) return -1ll;
        const size_t b = log.rfind("follower: ", at);
        if (b == std::string::npos) return -1ll;
        long long p = -1, n = -1, k = -1, h = -1;
        if (std::sscanf(log.c_str() + b, "follower: %lld passes ran in %lld segments, %lld other passes between their "
                                         "segments, %lld held", &p, &n, &k, &h) != 4) return -1ll;
        const std::string w(what);
        return w == "passes" ? p : w == " segments" ? n : w == " other passes between" ? k : h;
    };
    auto rf_taken = [](const std::string& log) {
        const size_t at = log.rfind(" lane steps' tails went before an older long pass");
        if (at == std::string::npos) return -1ll;
        const size_t b = log.rfind("two-node pipeline: ", at);
        return b == std::string::npos ? -1ll : std::atoll(log.c_str() + b + 19);
    };
    struct Arm { int lanes; const char* cut; bool fe; };
    long long between_all = 0;
    for (const Arm a : { Arm{ 2, "0,2", false }, Arm{ 3, "0,3", false }, Arm{ 2, "1,3", false }, Arm{ 2, "0,2", true },
                         Arm{ 3, "1,3", true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.insert(env.end(), seg_env.begin(), seg_env.end());
        if (a.fe) env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        Serving pp;
        REQUIRE(pp.start(a.cut, lane_shape(a.lanes), env));
        const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        const long long passes = seg_count(flog, "passes"), parts = seg_count(flog, " segments"),
                        between = seg_count(flog, " other passes between");
        between_all += std::max(0ll, between);
        fprintf(stderr, "    segments, %d lane(s) W(%s)%s: %d of %zu prompts differ; %lld passes in %lld segments, %lld passes "
                        "between; ready-first %lld; pieces lent %lld\n", a.lanes, a.cut, a.fe ? " + follower embeds" : "",
                differ, ref.size(), passes, parts, between, rf_taken(llog),
                log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(flog.find("run in layer segments at the boundaries") != std::string::npos);
        CHECK(flog.find("rows with decode rows (MIXED; RADIANCE_PP_SEG_PREFILL=1") != std::string::npos);
        CHECK(passes > 0);
        CHECK(parts >= 2 * passes);
        CHECK(llog.find("RADIANCE_PP_READY_FIRST=1 -- a later lane step's tail") != std::string::npos);
        CHECK(llog.find("waits for a first X2 (RADIANCE_PP_READY_FIRST=1)") != std::string::npos);
    }
    CHECK(between_all > 0);
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    for (const Arm a : { Arm{ 2, "0,2", false }, Arm{ 3, "0,3", false }, Arm{ 2, "1,3", false } }) {
        const std::string out = d / ("pp-" + std::to_string(a.lanes) + "-" + std::string(a.cut).replace(1, 1, "-"));
        std::vector<std::string> env = kSplitEnv;
        env.insert(env.end(), seg_env.begin(), seg_env.end());
        env.push_back("RADIANCE_PP_SEG_PREFILL=1");   /* the corpus runs alone: its lone prefill chunks in segments too */
        const PipeRun pr = record_pipeline(out, a.cut, env, lane_shape(a.lanes), lane_shape(a.lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", pr.leader, pr.follower,
                    tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    segments, %d lane(s) W(%s): recording %s; follower passes in segments %lld, between %lld\n",
                a.lanes, a.cut, same ? "identical" : why.c_str(), seg_count(pr.flog, "passes"),
                seg_count(pr.flog, " other passes between"));
        CHECK(same);
        CHECK(seg_count(pr.flog, "passes") > 0);
    }
}

/* ------------------------------------------------------------------ tp8 dev M5b */
/* RADIANCE_PP_LEND_MIXED=2: a newcomer planned whole beside running decoders is cut at the 512-grid point nearest its middle and
 * its rest lent to the other lane; every request's tokens are one process's (the host backend's kernels are chunking-invariant
 * on this model, so this checks the split's plumbing: the shrunk home entry, the lent token piece, the commit) at 2 / 3 lanes. */
std::vector<std::vector<int32_t>> split_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 41, 42, 43, 44 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 200); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    for (uint64_t s : { 51, 52, 53 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 680 + 40 * (s - 51)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

TEST(lend_split_newcomers_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "1040", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    for (const int lanes : { 2, 3 }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_LEND_MIXED=2");
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(lanes));
        Serving pp;
        REQUIRE(pp.start("0,2", ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
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
        long long split = -1;
        if (const size_t at = llog.rfind(" newcomers split on the 512 grid beside decode rows"); at != std::string::npos) {
            const size_t b = llog.rfind("; ", at);
            split = std::atoll(llog.c_str() + b + 2);
        }
        fprintf(stderr, "    lend split, %d lanes W(0,2): %d of %zu requests differ; newcomers split %lld, pieces lent %lld\n", lanes,
                differ, ref.size(), split, log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        CHECK(llog.find("(RADIANCE_PP_LEND_MIXED=2, tp8 dev M5b)") != std::string::npos);
        CHECK(split > 0);
    }
}

RAD_TEST_MAIN()

/* tp8 loop P10 (RADIANCE_PP_SEND_ASYNC=N, both nodes): a payload of >= N rows is packed into the node's staging buffer
 * on the pass's stream and published from the side stream (the host backend: a helper thread, so the head's pass ends
 * before its X1 is in the ring and the next pass starts at once); passes of >= N rows are issued live. A cold
 * multi-chunk prompt at 2 / 3 lanes serves one process's tokens and chunk plan, the mixed lane set (newcomers beside
 * decoders) one process's tokens, at W(0,2) / W(1,3); the stop lines count the side-stream payloads on both nodes; with
 * the switch off nothing is said. (The tiny model's chunks are ~64 rows: N = 32 here; the cards take N = 2048.) */
TEST(async_crossings_serve_one_process_tokens_and_plan) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(1313, 1000);
    std::vector<int32_t> ref;
    std::vector<std::pair<int, int>> ref_plan;
    std::vector<std::vector<int32_t>> ref_set;
    {
        Serving sv;
        REQUIRE(sv.start("", { "--max-num-seqs", "6" }, { "RADIANCE_LOG_STEPS=1" }));
        ref = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        ref_plan = chunks_of(sv.lead.output());
    }
    {
        Serving sv;
        REQUIRE(sv.start("", { "--max-num-seqs", "6" }));
        ref_set = lane_set(sv.port);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
    }
    CHECK(ref_plan.size() >= 4);
    /* "RADIANCE_PP_SEND_ASYNC=32 -- A of this node's B X1 payloads crossed from the side stream" */
    auto async_of = [](const std::string& log, long long* all) {
        const size_t at = log.rfind("payloads crossed from the side stream");
        if (at == std::string::npos) return -1ll;
        const size_t b = log.rfind(" -- ", at);
        long long a = -1, n = -1;
        if (b == std::string::npos || std::sscanf(log.c_str() + b, " -- %lld of this node's %lld", &a, &n) != 2) return -1ll;
        if (all) *all = n;
        return a;
    };
    /* tp8 loop P10b: "RADIANCE_PP_RECV_ASYNC=32 -- A of this node's B X1 receives (all ranks) came through" */
    auto recv_of = [](const std::string& log) {
        const size_t at = log.rfind("receives (all ranks) came through the staging buffers");
        if (at == std::string::npos) return -1ll;
        const size_t b = log.rfind(" -- ", at);
        long long a = -1, n = -1;
        if (b == std::string::npos || std::sscanf(log.c_str() + b, " -- %lld of this node's %lld", &a, &n) != 2) return -1ll;
        return a;
    };
    struct Arm { const char* lanes; const char* cut; bool send, recv; };
    for (const Arm a : { Arm{ "2", "0,2", true, false }, Arm{ "2", "1,3", true, true }, Arm{ "3", "1,3", false, true },
                         Arm{ "3", "0,3", true, true } }) {
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1" });
        if (a.send) env.push_back("RADIANCE_PP_SEND_ASYNC=32");
        if (a.recv) env.push_back("RADIANCE_PP_RECV_ASYNC=32");
        const std::vector<std::string> shape = { "--max-num-seqs", "6", "--pp-lanes", a.lanes };
        std::vector<int32_t> got;
        long long lall = 0, fall = 0, la = -1, fa = -1, lr = -1, fr = -1;
        bool plan_ok = false;
        {
            Serving sv;
            REQUIRE(sv.start(a.cut, shape, env));
            got = complete(sv.port, p, 24);
            kill(sv.lead.child.pid, SIGTERM);
            CHECK_EQ(sv.lead.finish(60), 0);
            CHECK_EQ(sv.fol.finish(30), 0);
            const std::string llog = sv.lead.output(), flog = sv.fol.output();
            plan_ok = chunks_of(llog) == ref_plan;
            la = async_of(llog, &lall);
            fa = async_of(flog, &fall);
            lr = recv_of(llog);
            fr = recv_of(flog);
            CHECK((llog.find("RADIANCE_PP_SEND_ASYNC=32 -- a X1 payload of >= 32 rows") != std::string::npos) == a.send);
            CHECK((flog.find("RADIANCE_PP_SEND_ASYNC=32 -- a X2 payload of >= 32 rows") != std::string::npos) == a.send);
            CHECK((llog.find("RADIANCE_PP_RECV_ASYNC=32 -- a X2 payload of >= 32 rows") != std::string::npos) == a.recv);
            CHECK((flog.find("RADIANCE_PP_RECV_ASYNC=32 -- a X1 payload of >= 32 rows") != std::string::npos) == a.recv);
            if (got != ref || !plan_ok) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(),
                                                tail(flog, 30).c_str());
        }
        int differ = -1;
        long long sla = -1, slall = 0;
        {
            Serving sv;
            REQUIRE(sv.start(a.cut, shape, env));
            const std::vector<std::vector<int32_t>> got_set = lane_set(sv.port);
            kill(sv.lead.child.pid, SIGTERM);
            CHECK_EQ(sv.lead.finish(60), 0);
            CHECK_EQ(sv.fol.finish(30), 0);
            REQUIRE_EQ(ref_set.size(), got_set.size());
            differ = 0;
            for (size_t i = 0; i < ref_set.size(); ++i) {
                CHECK(!ref_set[i].empty());
                if (ref_set[i] != got_set[i]) ++differ;
            }
            sla = async_of(sv.lead.output(), &slall);
        }
        fprintf(stderr, "    async crossings (send %d, receive %d), %s lanes W(%s): cold tokens %s, plan %s (%zu chunks); "
                        "leader X1 %lld of %lld from the side stream, follower X2 %lld of %lld; receives through staging: "
                        "follower %lld, leader %lld; lane set %d of %zu differ, X1 %lld of %lld async\n", a.send, a.recv,
                a.lanes, a.cut, got == ref ? "identical" : "DIFFERENT", plan_ok ? "identical" : "DIFFERENT", ref_plan.size(),
                la, lall, fa, fall, fr, lr, differ, ref_set.size(), sla, slall);
        CHECK(!ref.empty());
        CHECK(got == ref);
        CHECK(plan_ok);
        CHECK_EQ(differ, 0);
        if (a.send) {
            CHECK(la > 0);          /* the cold prompt's chunks of >= 32 rows */
            CHECK(la < lall);       /* the decode steps' small payloads on the pass's stream */
            CHECK(fa >= 0);
            CHECK(sla > 0);
        }
        if (a.recv) {
            CHECK(fr > 0);          /* X1 of the cold prompt's chunks, every follower rank */
            CHECK(lr >= 0);
        }
    }
    {   /* off: no line, nothing from a side stream */
        Serving sv;
        REQUIRE(sv.start("1,3", kLanes2, kSplitEnv));
        const std::vector<int32_t> got = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        CHECK(got == ref);
        CHECK(sv.lead.output().find("RADIANCE_PP_SEND_ASYNC") == std::string::npos);
    }
}

/* ------------------------------------------------------------------ tp8 dev M8 */
/* RADIANCE_PP_BG_LANE=1 (the last lane holds no sequence; a newcomer planned whole beside decoders is lent whole to it) with
 * the follower's segments for that lane's passes (RADIANCE_PP_SEG_LAYERS=1, no RADIANCE_PP_SEG_PREFILL) and ready-first tails:
 * every request's tokens are one process's at 2 / 3 lanes, W(0,2) and W(1,3) (a head-state cut), and the leader counts the
 * newcomers it lent whole, the follower the BG passes it split. */
TEST(bg_lane_newcomers_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "1040", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; bool split; };
    /* tp8 dev round 6: + three decode lanes beside one BG lane (--pp-lanes 4, BG_LANE=1: READY 05:57's settings arm) */
    for (const Arm a : { Arm{ 3, "0,2", 1, false }, Arm{ 2, "0,2", 1, false }, Arm{ 3, "1,3", 1, false }, Arm{ 4, "0,2", 2, false },
                         Arm{ 4, "1,3", 2, false }, Arm{ 4, "0,2", 2, true }, Arm{ 4, "1,3", 2, true }, Arm{ 4, "0,2", 1, false },
                         Arm{ 4, "1,3", 1, false } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        if (a.split) env.push_back("RADIANCE_PP_BG_SPLIT=1");
        if (a.k == 2) env.push_back("RADIANCE_PP_SEG_AHEAD=0");   /* READY 01:56's setting: a segment out only after the last one ran */
        for (const char* v : { "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        long long bg = -1, passes = -1, parts = -1, between = -1, held = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        if (const size_t at = flog.rfind(" passes ran in "); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "follower: %lld passes ran in %lld segments, %lld other passes between their "
                                                    "segments, %lld held", &passes, &parts, &between, &held);
        long long split = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            if (const size_t r = llog.find(" rows; ", at); r != std::string::npos) split = std::atoll(llog.c_str() + r + 7);
        fprintf(stderr, "    BG lanes %d%s, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to BG lanes (%lld split); follower %lld "
                        "passes in %lld segments, %lld passes between; pieces lent %lld\n", a.k, a.split ? " + split" : "", a.lanes, a.cut,
                differ, ref.size(), bg, split, passes,
                parts, between, log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("are background prefill lanes (RADIANCE_PP_BG_LANE=") != std::string::npos);
        CHECK(flog.find("and every pass of the BG lane(s)") != std::string::npos);
        CHECK(bg > 0);
        CHECK(passes > 0);
        if (a.split) CHECK(split > 0);
    }
}

/* tp8 loop P22b (RADIANCE_PP_PF_TIMELINE=1 on the follower): a BG newcomer's body in one-layer segments with decode passes
 * between them is read on its own record -- one line a newcomer, every segment read ("(S read;" = its segments), its own
 * lane's X1 wait read, the passes between its segments counted -- and the follower prints the run's BG-body summary at stop
 * with every newcomer of those lines; no split pass reaches P22's prompt lines (none of them unread). Tokens are one
 * process's at 3 lanes BG 1 W(0,2) / W(1,3) and 4 lanes BG 2 at SEG_AHEAD 0 and 1. */
TEST(the_follower_times_a_bg_newcomers_segments_on_its_own_line_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "1040", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; const char* ahead; };
    for (const Arm a : { Arm{ 3, "0,2", 1, "1" }, Arm{ 3, "1,3", 1, "0" }, Arm{ 4, "0,2", 2, "0" }, Arm{ 4, "1,3", 2, "1" } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back(std::string("RADIANCE_PP_SEG_AHEAD=") + a.ahead);
        for (const char* v : { "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_PF_TIMELINE=1", "RADIANCE_PP_NODE_IDLE=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        CHECK_EQ(differ, 0);
        /* the newcomers' own lines */
        int lines = 0, whole = 0, x1_read = 0, between = 0, segs_min = 1 << 30;
        double own_sum = 0;
        for (size_t at = flog.find("tp8 loop P22b): prompt "); at != std::string::npos;
             at = flog.find("tp8 loop P22b): prompt ", at + 1)) {
            const std::string line = flog.substr(at, flog.find('\n', at) - at);
            ++lines;
            unsigned long segs = 0;
            int nread = -1;
            double own = -1, x1 = -2;
            long long btw = -1;
            if (const size_t s = line.find(" in "); s != std::string::npos) segs = std::strtoul(line.c_str() + s + 4, nullptr, 10);
            if (const size_t c = line.find("rank 0's card: passes "); c != std::string::npos)
                (void)std::sscanf(line.c_str() + c, "rank 0's card: passes %lf ms (%d read;", &own, &nread);
            if (const size_t s = line.find("waiting for X1 "); s != std::string::npos) x1 = std::atof(line.c_str() + s + 15);
            if (const size_t s = line.find("; ", line.find("segment 0 (")); s != std::string::npos) btw = std::atoll(line.c_str() + s + 2);
            if (segs >= 2 && nread == (int)segs && own >= 0) ++whole;
            if (x1 >= 0) ++x1_read;
            if (btw > 0) ++between;
            if ((int)segs < segs_min) segs_min = (int)segs;
            own_sum += own;
            if (lines == 1) fprintf(stderr, "    %s\n", line.c_str());
        }
        long long run_n = -1, x1_unread = -1;
        if (const size_t at = flog.rfind("tp8 loop P22b) over the run: "); at != std::string::npos) {
            run_n = std::atoll(flog.c_str() + at + std::strlen("tp8 loop P22b) over the run: "));
            if (const size_t u = flog.find("X1 unread ", at); u != std::string::npos) x1_unread = std::atoll(flog.c_str() + u + 10);
        }
        /* P22's prompt lines (the decode lanes' lone passes): none unread now that no split pass is one of them */
        int p22 = 0, p22_unread = 0;
        for (size_t at = flog.find("tp8 loop P22): prompt "); at != std::string::npos; at = flog.find("tp8 loop P22): prompt ", at + 1)) {
            const std::string line = flog.substr(at, flog.find('\n', at) - at);
            ++p22;
            if (line.find(", unread 0") == std::string::npos) ++p22_unread;
        }
        fprintf(stderr, "    %d lanes BG %d W(%s) SEG_AHEAD %s: %d of %zu requests differ; %d BG-body lines (%d with every segment "
                        "read, >= %d segments, X1 read %d, passes between %d, own %.1f ms in all), run summary %lld newcomers, X1 "
                        "unread %lld; P22 prompt lines %d (%d not all read)\n", a.lanes, a.k, a.cut, a.ahead, differ, ref.size(),
                lines, whole, segs_min, x1_read, between, own_sum, run_n, x1_unread, p22, p22_unread);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(lines > 0);
        CHECK_EQ(whole, lines);
        CHECK_EQ(x1_read, lines);
        CHECK(between > 0);          /* decode passes ran between some newcomer's segments */
        CHECK_EQ(run_n, (long long)lines);
        CHECK_EQ(x1_unread, 0LL);
        CHECK_EQ(p22_unread, 0);
    }
}

/* tp8 dev M9: RADIANCE_PP_LEAD_SEG=1 -- a BG lane's step runs its leader head [0, h) and tail [t, n) in one-layer segments with
 * the other lanes' parts between them, its frame shipped with the head's last segment. Every request's tokens are one process's
 * at 3 lanes (BG 1) and 4 lanes (BG 1 / BG 2), at cuts with a head boundary (W(2,3): 1), a tail boundary (W(1,2): 3; W(0,2): 3,
 * where the head is the lookup alone) and beside the follower's own segments (W(0,2) / W(1,2) / W(2,3) at SEG_LAYERS=1: the
 * follower of W(2,3) holds one layer, its BG passes run whole); the leader counts its BG steps' segments and the other lanes'
 * tails between them. */
/* tp8 dev M9b: newcomers arriving together beside decoders (the BG lane takes one, the others are kept home as MIXED steps) */
std::vector<std::vector<int32_t>> burst_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 41, 42, 43, 44 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 200); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    for (uint64_t s : { 61, 62, 63, 64 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 400 + 40 * (s - 61)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

TEST(lead_segments_serve_one_process_tokens) {
    REQUIRE(world().ok);
    /* 700 rows a step: a BG lane's inbox holds one of burst_set's 400-520-row newcomers, the next is kept home (a MIXED step) */
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref, ref_b;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    /* tp8 dev M9b: + RADIANCE_PP_LEAD_SEG_MIXED=1 -- a decode lane's MIXED newcomer step in segments too (one pass on the ring) */
    struct Arm { int lanes; const char* cut; int k; const char* lead_seg; bool mixed; bool ahead = false; };
    long long between_all = 0, msteps_all = 0;
    for (const Arm a : { Arm{ 3, "2,3", 1, "1", false }, Arm{ 3, "1,2", 1, "1", false }, Arm{ 3, "0,2", 1, "1", false },
                         Arm{ 4, "2,3", 1, "1", false }, Arm{ 4, "1,2", 2, "1", false }, Arm{ 3, "2,3", 1, "2", false },
                         Arm{ 3, "2,3", 1, "1", true }, Arm{ 3, "1,2", 1, "1", true }, Arm{ 4, "2,3", 1, "1", true },
                         Arm{ 2, "2,3", 1, "1", true }, Arm{ 3, "2,3", 1, "1", true, true }, Arm{ 3, "1,2", 1, "1", false, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back(std::string("RADIANCE_PP_LEAD_SEG=") + a.lead_seg);
        if (a.mixed) env.push_back("RADIANCE_PP_LEAD_SEG_MIXED=1");
        if (a.ahead) env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1");   /* one segment queued behind the running one */
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port), got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        for (size_t i = 0; i < ref_b.size(); ++i) {
            CHECK(!ref_b[i].empty());
            if (ref_b[i] != got_b[i]) ++differ;
        }
        long long bg = -1, steps = -1, parts = -1, between = -1, inside = -1, polls = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments, %lld "
                                                    "other lanes' tails while one was in flight (%lld of them between two of its "
                                                    "segments), %lld polls", &steps, &parts, &between, &inside, &polls);
        between_all += std::max(0LL, inside);
        long long msteps = 0;
        if (const size_t at = llog.rfind(" MIXED newcomer steps of decode lanes among them"); at != std::string::npos)
            if (const size_t b = llog.rfind("; ", at); b != std::string::npos) msteps = std::atoll(llog.c_str() + b + 2);
        msteps_all += msteps;
        long long kept = -1;
        if (const size_t at = llog.rfind("the BG lanes' room "); at != std::string::npos) kept = std::atoll(llog.c_str() + at + 19);
        if (a.mixed) fprintf(stderr, "    + LEAD_SEG_MIXED=1: %lld MIXED newcomer steps of decode lanes in segments (%lld newcomers kept home for "
                                     "the BG lane's room)\n", msteps, kept);
        fprintf(stderr, "    LEAD_SEG=%s, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to BG lanes; leader: "
                        "%lld BG steps in %lld segments, %lld other tails in flight (%lld between two segments), %lld polls\n",
                a.lead_seg, a.k, a.lanes, a.cut, differ, ref.size() + ref_b.size(), bg, steps, parts, between, inside, polls);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("RADIANCE_PP_LEAD_SEG=") != std::string::npos);
        CHECK(llog.find("is not taken") == std::string::npos || std::strcmp(a.lead_seg, "2") == 0);
        CHECK(bg > 0);
        if (std::strcmp(a.lead_seg, "1") == 0) {
            CHECK(steps > 0);
            CHECK(parts >= steps);
        }
    }
    fprintf(stderr, "    other lanes' tails between two BG segments, all arms: %lld; MIXED newcomer steps in segments: %lld\n",
            between_all, msteps_all);
    CHECK(between_all > 0);
    CHECK(msteps_all > 0);
}

/* tp8 loop P27: RADIANCE_PP_LEAD_SEG_SHIP=1 (with LEAD_SEG) -- a segmented head's frame goes to the follower once its last segment
 * has run on rank 0's card, not with that segment's issue; with RADIANCE_PP_LEAD_SEG_SHIP_TEST_WAIT_MS it is held that long past
 * the last issue too, so the frames the leader issues meanwhile (other lanes' heads) reach the follower AHEAD of it. Every
 * request's tokens are one process's at 3 lanes W(2,3) BG 1 (+ LEAD_SEG_AHEAD, + LEAD_SEG_MIXED: a decode lane's MIXED newcomer
 * step's frame held the same way), 4 lanes W(2,3) BG 2 (two BG lanes: two held frames at once), beside the follower's own
 * segments and D15's frame order; at W(1,2) (no head boundary) the switch is refused by name and the run is LEAD_SEG's. */
TEST(lead_seg_ship_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref, ref_b;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; bool ahead; bool mixed; const char* order; const char* wait; };
    long long shipped_all = 0, late_all = 0;
    for (const Arm a : { Arm{ 3, "2,3", 1, false, false, nullptr, "40" }, Arm{ 3, "2,3", 1, true, false, nullptr, "40" },
                         Arm{ 3, "2,3", 1, true, true, nullptr, "40" }, Arm{ 4, "2,3", 2, true, false, "3", "40" },
                         Arm{ 3, "2,3", 1, true, false, "7", nullptr }, Arm{ 3, "1,2", 1, true, false, nullptr, "40" } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_PP_LEAD_SEG=1");
        env.push_back("RADIANCE_PP_LEAD_SEG_SHIP=1");
        if (a.ahead) env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1");
        if (a.mixed) env.push_back("RADIANCE_PP_LEAD_SEG_MIXED=1");
        if (a.order) env.push_back(std::string("RADIANCE_PP_FOL_ORDER=") + a.order);
        if (a.wait) env.push_back(std::string("RADIANCE_PP_LEAD_SEG_SHIP_TEST_WAIT_MS=") + a.wait);
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port), got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        for (size_t i = 0; i < ref_b.size(); ++i) {
            CHECK(!ref_b[i].empty());
            if (ref_b[i] != got_b[i]) ++differ;
        }
        long long steps = -1, parts = -1;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments", &steps,
                                  &parts);
        long long shipped = -1, ahead = -1, late = -1;
        double avg = -1, mx = -1;
        const char* key = " segmented heads' frames shipped after their last segment ran (RADIANCE_PP_LEAD_SEG_SHIP=1, tp8 loop P27): ";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos) {
                shipped = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
                (void)std::sscanf(llog.c_str() + at + std::strlen(key),
                                  "held past the last segment's issue %lf ms on average, the longest %lf; %lld other frames went "
                                  "ahead of them since their first segment, %lld of them", &avg, &mx, &ahead, &late);
            }
        const bool refused = std::strcmp(a.cut, "1,2") == 0;
        fprintf(stderr, "    SHIP%s%s%s%s, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld BG steps in %lld segments; %lld frames "
                        "shipped after the run (held %.2f ms avg, %.2f max past the last issue), %lld frames ahead since the first "
                        "segment, %lld after the last issue%s\n",
                a.ahead ? " + AHEAD" : "", a.mixed ? " + MIXED" : "", a.order ? (std::string(" + FOL_ORDER=") + a.order).c_str() : "",
                a.wait ? (std::string(" + TEST_WAIT ") + a.wait).c_str() : "", a.k, a.lanes, a.cut, differ, ref.size() + ref_b.size(),
                steps, parts, shipped, avg, mx, ahead, late, refused ? " (refused: no head boundary)" : "");
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        if (refused) {
            CHECK(llog.find("RADIANCE_PP_LEAD_SEG_SHIP=1 is not taken: no head boundary") != std::string::npos);
            CHECK_EQ(shipped, -1LL);
            continue;
        }
        CHECK(llog.find("RADIANCE_PP_LEAD_SEG_SHIP=1 -- a segmented head's frame goes") != std::string::npos);
        CHECK(steps > 0);
        CHECK(shipped > 0);
        CHECK(shipped <= steps);
        CHECK(late >= 0);
        CHECK(ahead >= late);
        if (a.wait) CHECK(mx >= 40.0);
        shipped_all += std::max(0LL, shipped);
        late_all += std::max(0LL, late);
    }
    fprintf(stderr, "    frames shipped after their head ran, all arms: %lld; other frames ahead of them after the last issue: %lld\n",
            shipped_all, late_all);
    CHECK(shipped_all > 0);
    CHECK(late_all > 0);   /* the reorder the switch makes did happen, and the tokens stayed one process's */
}

/* tp8 loop P28: RADIANCE_PP_BAND=1 (the leader) with the plugin's band (the llama stand-in RADIANCE_LLAMA_PP_BAND=1, a band of
 * one layer) -- a BG lane's step of fresh prompts runs as a band pass: the leader's head over [0, h + 1), its frame marked, the
 * follower from h + 1 after storing layer h's k / v from the band X1 into its own cache; every other pass is W(h, t)'s, and the
 * follower decodes layer h from that cache afterwards. Every request's tokens are one process's at 3 lanes W(1,3) BG 1 (whole and
 * with the follower's one-layer segments), 4 lanes W(1,3) BG 2, and + the leader's segments (LEAD_SEG: a band head runs whole). */
TEST(band_passes_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref, ref_b;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; int k; bool fseg; bool lseg; bool skip = false; const char* min_seqs = nullptr; };
    long long band_all = 0, low_all = 0;
    int skip_differ = 0;
    for (const Arm a : { Arm{ 3, 1, false, false }, Arm{ 3, 1, true, false }, Arm{ 4, 2, true, false }, Arm{ 3, 1, true, true },
                         Arm{ 3, 1, false, false, true }, Arm{ 3, 1, true, false, false, "1000" } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_LLAMA_PP_BAND=1");
        env.push_back("RADIANCE_PP_BAND=1");
        env.push_back("RADIANCE_PP_READY_FIRST=1");
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        if (a.lseg) { env.push_back("RADIANCE_PP_LEAD_SEG=1"); env.push_back("RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15"); }
        if (a.skip) env.push_back("RADIANCE_PP_TEST_BAND_SKIP=1");   /* negative control: the follower leaves the band's KV out */
        if (a.min_seqs) env.push_back(std::string("RADIANCE_PP_BAND_MIN_SEQS=") + a.min_seqs);   /* never reached here: no band pass */
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("1,3", ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port), got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        for (size_t i = 0; i < ref_b.size(); ++i) {
            CHECK(!ref_b[i].empty());
            if (ref_b[i] != got_b[i]) ++differ;
        }
        long long band = -1;
        const char* key = " lane steps ran as band passes (RADIANCE_PP_BAND=1, tp8 loop P28)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                band = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        fprintf(stderr, "    BAND (llama stand-in, one layer), %d lanes BG %d W(1,3)%s%s%s: %d of %zu requests differ; %lld lane steps ran as "
                        "band passes\n", a.lanes, a.k, a.fseg ? " + follower segments" : "", a.lseg ? " + LEAD_SEG" : "",
                a.skip ? " + the store LEFT OUT (negative control)" : "", differ, ref.size() + ref_b.size(), band);
        if (a.skip) {
            skip_differ = differ;
            CHECK(band > 0);
            continue;
        }
        CHECK_EQ(differ, 0);
        if (a.min_seqs) {   /* RADIANCE_PP_BAND_MIN_SEQS above the load: every fresh BG step stays W(h, t)'s, and is counted */
            long long low = -1;
            const char* lk = " fresh BG steps ran W(h, t)'s under RADIANCE_PP_BAND_MIN_SEQS=";
            if (const size_t at = llog.rfind(lk); at != std::string::npos)
                if (const size_t b = llog.rfind("; ", at); b != std::string::npos) low = std::atoll(llog.c_str() + b + 2);
            fprintf(stderr, "    + RADIANCE_PP_BAND_MIN_SEQS=%s: %lld band passes, %lld fresh BG steps kept W(1,3)'s\n", a.min_seqs, band, low);
            CHECK_EQ(band, 0LL);
            CHECK(low > 0);
            low_all += std::max(0LL, low);
            continue;
        }
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("RADIANCE_PP_BAND=1 -- a BG lane's step of fresh prompts runs as a band pass") != std::string::npos);
        CHECK(band > 0);
        band_all += std::max(0LL, band);
    }
    fprintf(stderr, "    band passes, all arms: %lld; the negative control: %d requests differ\n", band_all, skip_differ);
    CHECK(band_all > 0);
    CHECK(skip_differ > 0);   /* the test sees a band pass whose KV did not reach the decode owner */
}

/* tp8 dev D11: RADIANCE_PP_BG_HOLD_MS=N -- a newcomer the BG lanes have no room for is held out of its home's step (that step
 * runs its decoders alone) until a BG lane takes it whole, or runs at home as a MIXED step once N ms ran out. burst_set's four
 * newcomers arrive together beside four decoders at 700 rows a step (a BG lane's inbox holds one of them): every request's tokens
 * are one process's with the hold (60 s: never runs out here) at 3 lanes W(2,3) / W(0,2), 4 lanes W(1,2) BG 1, beside the
 * leader's segments (LEAD_SEG=1) and with a 1 ms hold that runs out; the leader's stop line counts the held newcomers. */
TEST(bg_hold_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref, ref_b;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; const char* hold; bool lead_seg; };
    long long held_all = 0, lent_all = 0, home_all = 0;
    /* + READY 09:06's S-L1 form: ONE decode lane beside the BG lane (--pp-lanes 2), the leader's segments one queued ahead */
    for (const Arm a : { Arm{ 3, "2,3", "60000", false }, Arm{ 3, "0,2", "60000", false }, Arm{ 4, "1,2", "60000", false },
                         Arm{ 3, "2,3", "60000", true }, Arm{ 3, "1,2", "1", false }, Arm{ 2, "2,3", "1000", true },
                         Arm{ 2, "1,2", "1000", true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_BG_LANE=1");
        env.push_back(std::string("RADIANCE_PP_BG_HOLD_MS=") + a.hold);
        if (a.lead_seg) env.push_back("RADIANCE_PP_LEAD_SEG=1");
        if (a.lead_seg && a.lanes == 2) env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1");
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port), got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        for (size_t i = 0; i < ref_b.size(); ++i) {
            CHECK(!ref_b[i].empty());
            if (ref_b[i] != got_b[i]) ++differ;
        }
        long long bg = -1, held = -1, steps = -1, lent = -1, home = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        const std::string key = std::string("held for the BG lanes' room (RADIANCE_PP_BG_HOLD_MS=") + a.hold + ", tp8 dev D11): ";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            (void)std::sscanf(llog.c_str() + at + key.size(), "%lld newcomers over %lld steps, %lld then lent to a BG lane, %lld ran at "
                                                                "home", &held, &steps, &lent, &home);
        held_all += std::max(0LL, held);
        lent_all += std::max(0LL, lent);
        home_all += std::max(0LL, home);
        fprintf(stderr, "    BG_HOLD_MS=%s%s%s, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to the BG lane; held %lld over %lld "
                        "steps, %lld then lent, %lld home after the hold\n", a.hold, a.lead_seg ? " + LEAD_SEG=1" : "",
                a.lead_seg && a.lanes == 2 ? " + LEAD_SEG_AHEAD=1 (one decode lane)" : "", a.lanes, a.cut,
                differ, ref.size() + ref_b.size(), bg, held, steps, lent, home);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("RADIANCE_PP_BG_HOLD_MS=") != std::string::npos);
        CHECK(llog.find("RADIANCE_PP_BG_HOLD_MS=" + std::string(a.hold) + " is not taken") == std::string::npos);
        CHECK(bg > 0);
        CHECK(held >= 0);
        if (held > 0) CHECK_EQ(lent + home, held);
    }
    fprintf(stderr, "    all arms: %lld newcomers held, %lld then lent to a BG lane, %lld ran at home after the hold\n", held_all,
            lent_all, home_all);
    CHECK(held_all > 0);
    CHECK(lent_all > 0);
}

/* tp8 loop P23: RADIANCE_PP_BG_ROOM=N -- a BG lane's inbox holds up to N rows of lent newcomers, so newcomers arriving while a BG
 * step runs go to its next step together. Four decoders, then five newcomers of 600-800 rows 40 ms apart, at 2112 rows a step
 * under --prefill-span 2 --checkpoint-interval 1024 (shared budget 1088: one newcomer a BG step without the switch); the BG pass
 * slowed by its one-layer segments' test wait. Tokens = one process (the same flags) at 3 lanes W(2,3) / W(0,2) and 4 lanes W(1,3)
 * (two BG lanes) with BG_ROOM=2112, at W(2,3) without it, and at W(2,3) with the leader's BG segments (LEAD_SEG + AHEAD) / the BG
 * hold; the leader counts the BG steps that carried >= 2 newcomers. */
std::vector<std::vector<int32_t>> bg_room_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 91, 92, 93, 94 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 160); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    for (uint64_t s : { 101, 102, 103, 104, 105 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 600 + 50 * (s - 101)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

TEST(bg_room_batches_newcomers_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "10", "--max-num-batched-tokens", "2112", "--max-model-len", "2048",
                                           "--prefill-span", "2", "--checkpoint-interval", "1024" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = bg_room_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    /* + two forms the cards may stack on it: the leader's BG segments (M9: LEAD_SEG + AHEAD) and the BG hold (D11) */
    struct Arm { int lanes; const char* cut; const char* bg; const char* room; bool lead_seg = false; bool hold = false;
                 const char* join = nullptr; };
    long long b2_all = 0;
    for (const Arm a : { Arm{ 3, "2,3", "1", "2112" }, Arm{ 3, "0,2", "1", "2112" }, Arm{ 4, "1,3", "2", "2112" },
                         Arm{ 3, "2,3", "1", nullptr }, Arm{ 3, "2,3", "1", "2112", true, false },
                         Arm{ 3, "2,3", "1", "2112", false, true }, Arm{ 3, "2,3", "1", "2112", false, false, "1" },
                         Arm{ 3, "0,2", "1", "2112", false, false, "60000" } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(std::string("RADIANCE_PP_BG_LANE=") + a.bg);
        if (a.room) env.push_back(std::string("RADIANCE_PP_BG_ROOM=") + a.room);
        if (a.lead_seg) { env.push_back("RADIANCE_PP_LEAD_SEG=1"); env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1"); }
        if (a.hold) env.push_back("RADIANCE_PP_BG_HOLD_MS=60000");
        if (a.join) env.push_back(std::string("RADIANCE_PP_BG_JOIN_MS=") + a.join);   /* tp8 loop P23b */
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = bg_room_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        long long bg = -1, steps = -1, b2 = -1, widest = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        const char* key = "tp8 loop P23): ";
        if (const size_t at = llog.rfind("BG steps over the run (RADIANCE_PP_BG_ROOM="); at != std::string::npos)
            if (const size_t k = llog.find(key, at); k != std::string::npos) {
                const char* q = llog.c_str() + k + std::strlen(key);
                steps = std::atoll(q);
                if (const char* w = std::strstr(q, "newcomers; ")) b2 = std::atoll(w + std::strlen("newcomers; "));
                if (const char* w = std::strstr(q, "the widest ")) widest = std::atoll(w + std::strlen("the widest "));
            }
        if (a.room) b2_all += std::max(0LL, b2);
        fprintf(stderr, "    BG_LANE=%s BG_ROOM=%s%s%s, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to the BG lanes; BG steps "
                        "%lld, %lld with >= 2 newcomers, the widest %lld rows\n", a.bg, a.room ? a.room : "off",
                a.lead_seg ? " + LEAD_SEG=1 + AHEAD" : a.hold ? " + HOLD" : "", a.join ? (std::string(" + JOIN_MS=") + a.join).c_str() : "",
                a.lanes, a.cut, differ, ref.size(), bg, steps, b2, widest);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(bg > 0);
        if (a.join) CHECK(llog.find(std::string("RADIANCE_PP_BG_JOIN_MS=") + a.join + " -- ") != std::string::npos);
        if (a.room) {
            CHECK(llog.find("RADIANCE_PP_BG_ROOM=2112 -- ") != std::string::npos);
            CHECK(steps > 0);
            CHECK(widest <= 2112);
        } else {
            CHECK(llog.find("RADIANCE_PP_BG_ROOM=") == std::string::npos);
        }
    }
    fprintf(stderr, "    all BG_ROOM arms: %lld BG steps with >= 2 newcomers\n", b2_all);
    CHECK(b2_all > 0);
}

/* tp8 loop D15: RADIANCE_PP_FOL_ORDER (the follower) -- between a split pass's segments, bit 1: another lane's decode frame is
 * issued once its X1 has landed, the waiting ones in the order their X1s land; bit 2: a held frame (prompt rows, a lent piece)
 * holds only its own lane's frames behind it. Two loads: bg_room_set's five newcomers at 2112 rows a step with BG_ROOM (two BG
 * lanes at 4 lanes: a second BG step's frame arrives during the first one's segments and is held), and split_set + burst_set at
 * 700 rows a step (one BG lane, the other newcomers kept home as MIXED steps: held frames during a BG pass's segments). Tokens =
 * one process (the same flags) at 4 lanes W(1,3) / W(0,2) BG 2 and 3 lanes W(0,2) / W(1,3) BG 1, bits 1 / 2 / 3, and with the
 * follower's own embedding (W(0,2) + RADIANCE_PP_FOLLOWER_EMBED=1: bit 1 is not taken there), and bit 4 (a segment held back
 * up to the 10 ms grace for a frame's X1) at 7 / 5; the follower counts the frames that waited for their X1 / went before an
 * older one / passed a held one; + beside the leader's BG segments (M9 LEAD_SEG=1 + AHEAD) at 4 lanes W(1,3) BG 2, =3 / =7. */
TEST(follower_frame_order_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext_r = { "--max-num-seqs", "10", "--max-num-batched-tokens", "2112", "--max-model-len", "2048",
                                             "--prefill-span", "2", "--checkpoint-interval", "1024" };
    const std::vector<std::string> ext_b = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref_r, ref_s, ref_b;
    {
        Serving one;
        REQUIRE(one.start("", ext_r));
        ref_r = bg_room_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    {
        Serving one;
        REQUIRE(one.start("", ext_b));
        ref_s = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { bool room; int lanes; const char* cut; const char* bg; const char* order; bool fe = false; bool lseg = false; };
    long long waited_all = 0, jumped_all = 0, passed_all = 0;
    for (const Arm a : { Arm{ true, 4, "1,3", "2", "3" }, Arm{ true, 4, "0,2", "2", "3" }, Arm{ true, 4, "1,3", "2", "1" },
                         Arm{ true, 4, "1,3", "2", "2" }, Arm{ false, 3, "0,2", "1", "3" }, Arm{ false, 3, "1,3", "1", "3" },
                         Arm{ false, 4, "1,3", "2", "3" }, Arm{ false, 3, "0,2", "1", "3", true },
                         Arm{ true, 4, "1,3", "2", "7" }, Arm{ false, 3, "1,3", "1", "5" },
                         Arm{ true, 4, "1,3", "2", "3", false, true }, Arm{ false, 4, "1,3", "2", "7", false, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(std::string("RADIANCE_PP_BG_LANE=") + a.bg);
        env.push_back(std::string("RADIANCE_PP_FOL_ORDER=") + a.order);
        if (a.room) env.push_back("RADIANCE_PP_BG_ROOM=2112");
        if (a.fe) env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        if (a.lseg) {   /* + the leader's BG segments (M9): both nodes fill their waits with BG segments */
            env.push_back("RADIANCE_PP_LEAD_SEG=1");
            env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1");
            env.push_back("RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15");
        }
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = a.room ? ext_r : ext_b;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        std::vector<std::vector<int32_t>> got, ref;
        if (a.room) {
            got = bg_room_set(pp.port);
            ref = ref_r;
        } else {
            got = split_set(pp.port);
            const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
            got.insert(got.end(), got_b.begin(), got_b.end());
            ref = ref_s;
            ref.insert(ref.end(), ref_b.begin(), ref_b.end());
        }
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        long long bg = -1, passes = -1, parts = -1, between = -1, held = -1, waited = -1, jumped = -1, passed = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        if (const size_t at = flog.rfind(" passes ran in "); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "follower: %lld passes ran in %lld segments, %lld other passes between their "
                                                    "segments, %lld held", &passes, &parts, &between, &held);
        if (const size_t at = flog.rfind("frame order over the run (RADIANCE_PP_FOL_ORDER="); at != std::string::npos)
            if (const size_t k = flog.find("tp8 loop D15): ", at); k != std::string::npos)
                (void)std::sscanf(flog.c_str() + k, "tp8 loop D15): %lld decode frames waited between segments for their X1, %lld "
                                                    "went before an older waiting frame, %lld passed", &waited, &jumped, &passed);
        waited_all += std::max(0LL, waited);
        jumped_all += std::max(0LL, jumped);
        passed_all += std::max(0LL, passed);
        fprintf(stderr, "    FOL_ORDER=%s%s, BG lanes %s%s, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to BG lanes; "
                        "follower %lld passes in %lld segments, %lld passes between, %lld held; %lld waited for X1, %lld went "
                        "before an older one, %lld passed a held frame\n", a.order, a.fe ? " + FE" : a.lseg ? " + LEAD_SEG=1 + AHEAD" : "", a.bg,
                a.room ? " + BG_ROOM=2112" : "", a.lanes, a.cut, differ, ref.size(), bg, passes, parts, between, held, waited,
                jumped, passed);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(bg > 0);
        CHECK(passes > 0);
        CHECK(flog.find("RADIANCE_PP_FOL_ORDER=") != std::string::npos);
        CHECK(waited >= 0);   /* the stop line was printed */
        if (a.fe) {
            CHECK(flog.find("RADIANCE_PP_FOL_ORDER bit 1 is not taken here") != std::string::npos);
            CHECK_EQ(waited, 0);
        }
        if (a.order[0] == '7' || a.order[0] == '5')
            CHECK(flog.find("RADIANCE_PP_FOL_ORDER bit 4 -- a segment waits up to 10 ms") != std::string::npos);
        if (std::strcmp(a.order, "2") == 0) CHECK_EQ(waited, 0);
        if (std::strcmp(a.order, "1") == 0) CHECK_EQ(passed, 0);
    }
    fprintf(stderr, "    all arms: %lld frames waited for their X1, %lld went before an older one, %lld passed a held frame\n",
            waited_all, jumped_all, passed_all);
    CHECK(waited_all > 0);
    CHECK(passed_all > 0);
}

/* tp8 dev D10 round 2: steps issued ahead beside the BG lanes' split passes, the best c32 start's form (W(0, t), the follower's
 * own lookup from X1's token ids, BG lanes in one-layer segments, ready-first tails): a frame the leader issued ahead arrives
 * about a leader tail before its X1 and waits between the BG pass's segments until X1 has landed. Tokens = one process at
 * 3 lanes (BG 1) and 4 lanes (BG 2), W(0,2) and W(0,3); steps were issued ahead, newcomers went to the BG lanes. */
TEST(issued_ahead_beside_bg_lanes_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "1040", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; };
    long long deferred_all = 0;
    for (const Arm a : { Arm{ 3, "0,2", 1 }, Arm{ 4, "0,2", 2 }, Arm{ 4, "0,3", 2 } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOLLOWER_EMBED=1",
                               "RADIANCE_PP_ISSUE_AHEAD=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        long long bg = -1, n_ahead = -1, deferred = 0, passes = -1, parts = -1, between = -1, held = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        if (const size_t at = llog.rfind(" lane steps issued ahead of the collect"); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos) n_ahead = std::atoll(llog.c_str() + b + 19);
        if (const size_t at = flog.rfind(" frames issued ahead waited between"); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos) deferred = std::atoll(flog.c_str() + b + 10);
        if (const size_t at = flog.rfind(" passes ran in "); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "follower: %lld passes ran in %lld segments, %lld other passes between their "
                                                    "segments, %lld held", &passes, &parts, &between, &held);
        deferred_all += deferred;
        fprintf(stderr, "    issue ahead + FE + BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld lane steps issued ahead, "
                        "%lld waited for their X1 between segments; %lld newcomers to BG lanes; follower %lld passes in %lld segments, "
                        "%lld passes between, %lld held\n", a.k, a.lanes, a.cut, differ, ref.size(), n_ahead, deferred, bg, passes,
                parts, between, held);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(n_ahead > 0);
        CHECK(bg > 0);
        CHECK(passes > 0);
        CHECK(flog.find("this node's pass takes the step's token ids from X1") != std::string::npos);
    }
    fprintf(stderr, "    frames issued ahead that waited between segments, all arms: %lld\n", deferred_all);
}

/* ------------------------------------------------------------------ tp8 dev D10 */
/* RADIANCE_PP_ISSUE_AHEAD=1 (both nodes): an undrafted decode lane's next step is planned, its head part released and its frame
 * shipped before the step's collect (Scheduler::step_next + ADV_TRUNK on the leader; the follower reads no token id). Tokens at
 * two and three lanes, W(0,2) and W(1,3) (alone, every lane in turn, and beside newcomers whose chunks are lent), are one
 * process's, and steps are issued ahead; round 2: with the follower's own embedding too (X1 carries the step's token ids, the
 * follower takes them before its lookup), W(0,2) / W(0,3); the KL recordings at two and three lanes are one process's, W(1,3)
 * and W(0,2) with the follower's lookup. */
TEST(lane_steps_issued_ahead_of_the_collect_serve_and_record_what_one_process_does) {
    REQUIRE(world().ok);
    tiny::TempDir d("pp_ia");
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; bool fe; };
    /* tp8 dev D10 round 2: beside the follower's own lookup too (X1 carries the step's token ids, the leader's card's, which
     * the follower takes before its lookup): W(0,2) at two and three lanes, W(0,3) at two */
    for (const Arm a : { Arm{ 2, "0,2", false }, Arm{ 3, "0,2", false }, Arm{ 2, "1,3", false }, Arm{ 3, "1,3", false },
                         Arm{ 2, "0,2", true }, Arm{ 3, "0,2", true }, Arm{ 2, "0,3", true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_ISSUE_AHEAD=1");
        if (a.fe) env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        Serving pp;
        REQUIRE(pp.start(a.cut, lane_shape(a.lanes), env));
        const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) {
                ++differ;
                if (std::getenv("IA_DEBUG")) {
                    size_t k = 0;
                    while (k < ref[i].size() && k < got[i].size() && ref[i][k] == got[i][k]) ++k;
                    std::string r1, g1;
                    for (size_t j = k > 2 ? k - 2 : 0; j < k + 4 && j < ref[i].size(); ++j) r1 += std::to_string(ref[i][j]) + " ";
                    for (size_t j = k > 2 ? k - 2 : 0; j < k + 4 && j < got[i].size(); ++j) g1 += std::to_string(got[i][j]) + " ";
                    fprintf(stderr, "      prompt %zu: first difference at %zu of %zu / %zu: ref %s| got %s\n", i, k, ref[i].size(),
                            got[i].size(), r1.c_str(), g1.c_str());
                }
            }
        }
        long long n_ahead = -1;   /* the last count line's */
        if (const size_t at = llog.rfind(" lane steps issued ahead of the collect"); at != std::string::npos) {
            const size_t b = llog.rfind("two-node pipeline: ", at);
            if (b != std::string::npos) n_ahead = std::atoll(llog.c_str() + b + 19);
        }
        fprintf(stderr, "    issue ahead, %d lane(s) W(%s)%s: %d of %zu prompts differ; lane steps issued ahead %lld; pieces lent "
                        "%lld\n", a.lanes, a.cut, a.fe ? " + follower embeds" : "", differ, ref.size(), n_ahead,
                log_count(llog, "newcomer split: "));
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 20).c_str());
        CHECK(llog.find("RADIANCE_PP_ISSUE_AHEAD=1 -- an undrafted decode lane's next step") != std::string::npos);
        CHECK(n_ahead > 0);
        if (!a.fe) {
            CHECK(flog.find("follower: RADIANCE_PP_ISSUE_AHEAD=1 -- this node's pass reads no token id") != std::string::npos);
        } else {
            CHECK(llog.find("X1 carries the step's token ids, which the follower takes before it reads any (round 2)") !=
                  std::string::npos);
            CHECK(flog.find("follower: RADIANCE_PP_ISSUE_AHEAD=1 -- this node's pass takes the step's token ids from X1") !=
                  std::string::npos);
        }
    }
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    for (const int lanes : { 2, 3 }) {
        const std::string out = d / ("pp-" + std::to_string(lanes));
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_ISSUE_AHEAD=1");
        const PipeRun pr = record_pipeline(out, "1,3", env, lane_shape(lanes), lane_shape(lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    %d lanes: leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", lanes, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        long long n_rec = -1;
        if (const size_t at = pr.llog.rfind(" lane steps issued ahead of the collect"); at != std::string::npos) {
            const size_t b = pr.llog.rfind("two-node pipeline: ", at);
            if (b != std::string::npos) n_rec = std::atoll(pr.llog.c_str() + b + 19);
        }
        fprintf(stderr, "    issue ahead, %d lane(s) W(1,3): recording %s; lane steps issued ahead %lld\n", lanes,
                same ? "identical" : why.c_str(), n_rec);
        CHECK(same);
    }
    /* tp8 dev D10 round 2: the recordings with the follower's own lookup from X1's token ids (every pass's X1 carries them,
     * issued ahead or not) at W(0,2), two and three lanes */
    for (const int lanes : { 2, 3 }) {
        const std::string out = d / ("pp-fe-" + std::to_string(lanes));
        std::vector<std::string> env = kSplitEnv;
        env.push_back("RADIANCE_PP_ISSUE_AHEAD=1");
        env.push_back("RADIANCE_PP_FOLLOWER_EMBED=1");
        const PipeRun pr = record_pipeline(out, "0,2", env, lane_shape(lanes), lane_shape(lanes), env);
        if (pr.leader != 0 || pr.follower != 0)
            fprintf(stderr, "    %d lanes FE: leader %d, follower %d\n    leader:\n%s\n    follower:\n%s\n", lanes, pr.leader,
                    pr.follower, tail(pr.llog).c_str(), tail(pr.flog).c_str());
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    issue ahead + follower embeds, %d lane(s) W(0,2): recording %s\n", lanes,
                same ? "identical" : why.c_str());
        CHECK(same);
        CHECK(pr.flog.find("this node's pass takes the step's token ids from X1") != std::string::npos);
    }
}
