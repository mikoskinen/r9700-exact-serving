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
    bool start(const std::string& cut, const std::vector<std::string>& extra = {}) {
        World& w = world();
        std::vector<std::string> a = cat({ "--host", "127.0.0.1", "--port", "0", "--http-threads", "4" }, shape());
        a = cat(a, extra);
        if (!cut.empty()) {
            pp_port = free_port();
            if (!fol.start(engine_argv(w.llama_rad, cat(a, pp_args(1, pp_port, cut))), engine_env())) return false;
            if (!fol.wait_for("waiting on", 60)) return false;
            a = cat(a, pp_args(0, pp_port, cut));
        }
        if (!lead.start(engine_argv(w.llama_rad, a), engine_env())) return false;
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

RAD_TEST_MAIN()
