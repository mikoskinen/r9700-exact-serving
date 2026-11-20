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
#include <cmath>
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

/* tp8 loop D36: the same tiny llama with six layers, for a cut whose head holds a boundary AND a band layer -- W(2,5): the
 * leader's head [0, 2) with LEAD_SEG=1's boundary 1, the band layer 2 (RADIANCE_LLAMA_PP_BAND=1), the follower [2, 5) */
struct World6 {
    tiny::TempDir dir{ "pp_e2e6" };
    std::string llama, llama_rad, log;
    bool ok = false;
    World6() {
        if (!dir.ok()) return;
        llama = dir / "llama";
        llama_rad = dir / "llama.rad";
        if (mkdir(llama.c_str(), 0755) != 0) return;
        tiny::Llama s;
        s.layers = 6;
        s.ctx = 65536;
        if (!tiny::write_llama(llama, s) || !tiny::write_text(dir / "llama.recipe", tiny::kLlamaFp8Recipe)) return;
        const int st = tiny::run({ RAD_E2E_CONVERT, llama, "-o", llama_rad, "--recipe", dir / "llama.recipe" },
                                 engine_env(), &log);
        if (st != 0) { fprintf(stderr, "    rad-convert (6 layers) exited %d:\n%s\n", st, tail(log).c_str()); return; }
        ok = true;
    }
};
World6& world6() { static World6 w; return w; }

/* tp8 loop D38a: the four-layer llama with a DSpark block drafter merged (rad-convert --draft-model), its one tap at layer
 * 3 -- inside the leader's tail [t, 4) at every cut t <= 3 -- and a block of 3 (the depth it serves): the host fixture for
 * a DRAFTED decode lane (D38's issue-ahead of drafted lanes needs one to prove its tokens). */
struct WorldDs {
    tiny::TempDir dir{ "pp_e2eds" };
    std::string llama, draft, llama_rad, log;
    bool ok = false;
    WorldDs() {
        if (!dir.ok()) return;
        llama = dir / "llama";
        draft = dir / "dspark";
        llama_rad = dir / "llama.rad";
        if (mkdir(llama.c_str(), 0755) != 0 || mkdir(draft.c_str(), 0755) != 0) return;
        tiny::Llama s;
        s.layers = 4;
        s.ctx = 65536;
        s.branch = 0.1f;   /* near-bigram: the drafter's anchor row can match it (tiny::DsparkDraft) */
        tiny::DsparkDraft d;
        if (!tiny::write_llama(llama, s) || !tiny::write_dspark(draft, d) ||
            !tiny::write_text(dir / "llama.recipe", tiny::kLlamaDsparkRecipe))
            return;
        const int st = tiny::run({ RAD_E2E_CONVERT, llama, "-o", llama_rad, "--recipe", dir / "llama.recipe", "--draft-model",
                                   draft, "--max-spec", "3" },
                                 engine_env(), &log);
        if (st != 0) { fprintf(stderr, "    rad-convert (DSpark) exited %d:\n%s\n", st, tail(log).c_str()); return; }
        ok = true;
    }
};
WorldDs& worldds() { static WorldDs w; return w; }

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
    std::string model;   /* tp8 loop D36: another model file (empty: world()'s four-layer llama) */
    bool start(const std::string& cut, const std::vector<std::string>& extra = {},
               const std::vector<std::string>& env = {}) {
        World& w = world();
        const std::string& mp = model.empty() ? w.llama_rad : model;
        std::vector<std::string> a = cat({ "--host", "127.0.0.1", "--port", "0", "--http-threads", "8" }, shape());
        a = cat(a, extra);
        if (!cut.empty()) {
            pp_port = free_port();
            if (!fol.start(engine_argv(mp, cat(a, pp_args(1, pp_port, cut))), engine_env(env))) return false;
            if (!fol.wait_for("waiting on", 60)) return false;
            a = cat(a, pp_args(0, pp_port, cut));
        }
        if (!lead.start(engine_argv(mp, a), engine_env(env))) return false;
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

/* tp8 loop D38a: /completion's greedy tokens with its draft counts (timings.draft_n / draft_n_accepted; 0 when none) */
struct Drafted { std::vector<int32_t> tok; long long drafted = 0, accepted = 0; };
Drafted complete_drafted(int port, const std::vector<int32_t>& prompt, int n) {
    const json b = { { "prompt", prompt }, { "n_predict", n }, { "temperature", 0.0 }, { "ignore_eos", true },
                     { "return_tokens", true } };
    Call c;
    std::string out;
    Drafted d;
    if (!(c.send(port, "/completion", b.dump()) && c.read(&out) == 200)) return d;
    const json j = json::parse(out, nullptr, false);
    if (!j.is_object() || !j.contains("tokens")) return d;
    d.tok = j["tokens"].get<std::vector<int32_t>>();
    d.drafted = j.value(json::json_pointer("/timings/draft_n"), (long long)0);
    d.accepted = j.value(json::json_pointer("/timings/draft_n_accepted"), (long long)0);
    return d;
}
/* Six prompts: two alone (one repetitive, which a drafter can follow), then four at once. */
std::vector<Drafted> drafted_set(int port) {
    std::vector<Drafted> out;
    std::vector<int32_t> rep;
    for (int k = 0; k < 6; ++k) for (int32_t t : prompt_ids(21, 24)) rep.push_back(t);
    out.push_back(complete_drafted(port, rep, 32));
    out.push_back(complete_drafted(port, prompt_ids(22, 90), 28));
    std::vector<std::future<Drafted>> f;
    for (uint64_t s : { 23, 24, 25, 26 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete_drafted(port, prompt_ids(s, 40 + 9 * s), 24); }));
    for (auto& x : f) out.push_back(x.get());
    return out;
}

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

/* tp8 loop D38a: the DSpark fixture on one process -- a drafted run (--num-speculative-tokens 3, the block it serves)
 * returns the plain greedy tokens and drafts (some accepted, some not: the check means something). */
TEST(d38a_a_dspark_drafted_process_serves_the_plain_greedy_tokens) {
    REQUIRE(worldds().ok);
    std::vector<Drafted> plain, spec;
    for (int k = 0; k < 2; ++k) {
        Serving one;
        one.model = worldds().llama_rad;
        REQUIRE(one.start("", { "--num-speculative-tokens", k ? "3" : "0" }));
        (k ? spec : plain) = drafted_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    REQUIRE_EQ(plain.size(), spec.size());
    long long drafted = 0, accepted = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        CHECK(!plain[i].tok.empty());
        if (plain[i].tok != spec[i].tok) fprintf(stderr, "    prompt %zu: the drafted run's tokens differ\n", i);
        CHECK(plain[i].tok == spec[i].tok);
        CHECK_EQ(plain[i].drafted, 0);
        drafted += spec[i].drafted;
        accepted += spec[i].accepted;
    }
    fprintf(stderr, "    one process, DSpark block 3: %lld drafted, %lld accepted over %zu prompts\n", drafted, accepted,
            spec.size());
    CHECK(drafted > accepted);
    CHECK(accepted > 0);
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
    /* tp8 loop D23: draft_short_blk is the leader's alone; with no short block served (the tiny model drafts nothing) it is
     * refused by name, never asked of the follower, and the pair serves on */
    Call c3;
    st = c3.send(pp.port, "/admin/fn4", json{ { "draft_short_blk", 2 } }.dump()) ? c3.read(&out) : -1;
    CHECK_EQ(st, 400);
    const bool none = out.find("serves no short block") != std::string::npos;
    if (!none) fprintf(stderr, "    POST draft_short_blk 2: %d %s\n", st, out.c_str());
    CHECK(none);
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
        /* tp8 loop D41: RADIANCE_PP_HEAD_PHASES=16 -- rank 0 times every 16 whole decode heads by phase (timing only) */
        REQUIRE(pp.start("0,2", lane_shape(lanes), { "RADIANCE_PP_SPLIT_MIN=16", "RADIANCE_PP_STAMPS=" + prefix,
                                                     "RADIANCE_PP_HEAD_PHASES=16" }));
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
        /* tp8 loop D37: with lanes every whole head has its 'hiss' stamp -- rank 0's run_step entry after the head's release,
         * its return at or after the entry */
        {
            std::istringstream in(tiny::read_text(prefix + ".leader"));
            std::string kind;
            int lane = 0, step = 0;
            unsigned long long b0 = 0, e0 = 0;
            std::map<int, unsigned long long> rel;
            long long heads = 0, hiss = 0, bad = 0;
            std::vector<std::tuple<int, unsigned long long, unsigned long long>> hs;
            while (in >> kind >> lane >> step >> b0 >> e0) {
                if (kind == "head") { ++heads; rel[step] = b0; }
                if (kind == "hiss") { ++hiss; hs.emplace_back(step, b0, e0); }
            }
            for (const auto& [st, h0, h1] : hs) {
                const auto it = rel.find(st);
                if (h0 == 0 || h1 < h0 || it == rel.end() || h0 < it->second) ++bad;
            }
            fprintf(stderr, "    %d lane(s): %lld head stamps, %lld 'hiss' stamps (tp8 loop D37), %lld out of order\n", lanes, heads, hiss, bad);
            CHECK(heads > 0);
            CHECK_EQ(hiss, lanes > 1 ? heads : 0LL);   /* one lane runs engine.cpp's step loop: no 'hiss' there */
            CHECK_EQ(bad, 0LL);
            /* tp8 loop D41: with lanes, rank 0's head-phase lines (every 16 decode heads); one lane issues no head part */
            const std::string llog = pp.lead.output();
            const char* key = "radiance: rank 0 decode head issue by phase (RADIANCE_PP_HEAD_PHASES, tp8 loop D41): ";
            long long lines = 0, n = -1, played = -1, live = -1;
            for (size_t at = llog.find(key); at != std::string::npos; at = llog.find(key, at + 1)) {
                ++lines;
                (void)std::sscanf(llog.c_str() + at + std::strlen(key), "%lld heads (%lld played, %lld issued live)", &n, &played, &live);
            }
            fprintf(stderr, "    %d lane(s): %lld head-phase lines (tp8 loop D41), the last: %lld heads, %lld played, %lld live\n", lanes,
                    lines, n, played, live);
            /* tp8 loop D41b: each head-phase line has its parts line (phase 0 split, pre-stretch host steps by kind) */
            const char* keyb = "radiance: rank 0 decode head issue, the parts (RADIANCE_PP_HEAD_PHASES, tp8 loop D41b): stager begin_pass ";
            long long lines_b = 0;
            for (size_t at = llog.find(keyb); at != std::string::npos; at = llog.find(keyb, at + 1)) ++lines_b;
            CHECK_EQ(lines_b, lines);
            if (lines_b > 0) {
                const size_t at = llog.rfind(keyb);
                fprintf(stderr, "      %s\n", llog.substr(at, llog.find('\n', at) - at).c_str());
            }
            if (lanes > 1) {
                CHECK(lines > 0);
                CHECK_EQ(n, 16LL);
                CHECK(played >= 0 && live >= 0 && played + live == 16);   /* the host backend records no tapes: every head live */
                if (lines > 0) {
                    const size_t at = llog.rfind(key);
                    fprintf(stderr, "      %s\n", llog.substr(at, llog.find('\n', at) - at).c_str());
                }
            } else {
                CHECK_EQ(lines, 0LL);
            }
        }
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
            /* tp8 loop F68: the request's wait at the scheduler before its first plan, stamped (>= 0) */
            if (pl != std::string::npos) {
                const std::string ln = llog.substr(pl, llog.find('\n', pl) - pl);
                const size_t at = ln.find("; at the scheduler ");
                CHECK(at != std::string::npos);
                if (at != std::string::npos) CHECK(ln.compare(at + 19, 1, "-") != 0);
            }
            if (pl != std::string::npos) fprintf(stderr, "    %s\n", llog.substr(pl, llog.find('\n', pl) - pl).c_str());
            if (gl != std::string::npos) fprintf(stderr, "    %s\n", llog.substr(gl, llog.find('\n', gl) - gl).c_str());
        }
    }
}

/* tp8 loop P34 (RADIANCE_PP_SPAN_FIRST=F with RADIANCE_PP_SPAN_SHORT): the switch acts only where the chunk geometry has a
 * checkpoint interval (a linear state, or a window under a prefix cache: sched_test lanes_span_first_* cuts DeepSeek's plan,
 * window keys and every row checked); the four-layer llama here has neither, so at 2 lanes W(0,2) / W(1,3) and 3 lanes W(0,3)
 * with F = 512 / 256 (--prefill-span 2 --checkpoint-interval 1024) the pipeline plans and serves exactly as without it --
 * one process's tokens, the same plan -- and the leader's start line names the switch. */
TEST(span_first_changes_nothing_without_a_checkpoint_interval_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "2112", "--max-model-len", "4096",
                                           "--prefill-span", "2", "--checkpoint-interval", "1024" };
    for (size_t n : { 3300, 1900 }) {
        const std::vector<int32_t> p = prompt_ids(77 + n, n);
        std::vector<int32_t> ref;
        {
            Serving one;
            REQUIRE(one.start("", ext, { "RADIANCE_LOG_STEPS=1" }));
            ref = complete(one.port, p, 24);
            kill(one.lead.child.pid, SIGTERM);
            CHECK_EQ(one.lead.finish(60), 0);
        }
        /* arm 0: the pipeline without the switch (its plan is the reference for the arms' plans); then the switch's arms */
        struct Arm { int lanes; const char* cut; const char* first; };
        std::vector<std::pair<int, int>> base;
        for (const Arm a : { Arm{ 2, "0,2", nullptr }, Arm{ 2, "0,2", "512" }, Arm{ 2, "1,3", "256" }, Arm{ 3, "0,3", "512" } }) {
            std::vector<std::string> ex = ext;
            ex.push_back("--pp-lanes");
            ex.push_back(std::to_string(a.lanes));
            std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_PP_SPAN_SHORT=24576" });
            if (a.first) env.push_back(std::string("RADIANCE_PP_SPAN_FIRST=") + a.first);
            Serving pp;
            REQUIRE(pp.start(a.cut, ex, env));
            const std::vector<int32_t> got = complete(pp.port, p, 24);
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output();
            const std::vector<std::pair<int, int>> plan = chunks_of(llog);
            CHECK(!ref.empty());
            CHECK(got == ref);
            if (!a.first) {
                base = plan;
                fprintf(stderr, "    %zu-token prompt, %d lanes W(%s), SPAN_SHORT only: %zu chunks, first [0, %d), tokens %s\n", p.size(),
                        a.lanes, a.cut, plan.size(), plan.empty() ? -1 : plan.front().second, got == ref ? "identical" : "DIFFERENT");
                continue;
            }
            fprintf(stderr, "    %zu-token prompt, %d lanes W(%s), SPAN_FIRST %s: %zu chunks (without it %zu), first [0, %d), plan %s, "
                            "tokens %s\n", p.size(), a.lanes, a.cut, a.first, plan.size(), base.size(),
                    plan.empty() ? -1 : plan.front().second, plan == base ? "= without it" : "DIFFERENT",
                    got == ref ? "identical" : "DIFFERENT");
            CHECK(!base.empty());
            CHECK(plan == base);
            CHECK(llog.find("RADIANCE_PP_SPAN_FIRST, tp8 loop P34") != std::string::npos);
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

/* tp8 loop P37 (RADIANCE_PP_KV_CHECK_DEFER=1 with RADIANCE_PP_KV_CHECK=N): a cold multi-chunk prompt at the wrap-around cut
 * W(1,3) with the KV copy on, at 2 and 3 lanes, N = its chunks - 1 (so a check comes due inside the prompt and 2N lies past
 * its last chunk). Without the switch the due check drains the pipeline inside the prompt and the leader says so; with it
 * the check waits for the prompt's rows to be out (a line says it was deferred, none says a drain with prompt rows) -- one
 * process's tokens and chunk plan in every arm, and every check finds the copy byte-equal (the follower's lines). */
TEST(a_kv_check_deferred_past_a_cold_prompt_keeps_one_process_tokens_and_the_kv_copy) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(1039, 1000);
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
    REQUIRE(ref_plan.size() >= 4);
    const std::string every = std::to_string(ref_plan.size() - 1);
    struct Arm { const char* lanes; bool defer; };
    for (const Arm a : { Arm{ "2", false }, Arm{ "2", true }, Arm{ "3", true } }) {
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_LLAMA_PP_KVCOPY=1",
                                                        std::string("RADIANCE_PP_KV_CHECK=") + every });
        if (a.defer) env.push_back("RADIANCE_PP_KV_CHECK_DEFER=1");
        Serving sv;
        REQUIRE(sv.start("1,3", { "--max-num-seqs", "6", "--pp-lanes", a.lanes }, env));
        const std::vector<int32_t> got = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        const std::string llog = sv.lead.output(), flog = sv.fol.output();
        auto occ = [](const std::string& log, const char* tag) {
            long long n = 0;
            for (size_t at = log.find(tag); at != std::string::npos; at = log.find(tag, at + 1)) ++n;
            return n;
        };
        const long long drained = occ(llog, "drained the pipeline with prompt rows in flight");
        const long long deferred = occ(llog, "deferred while prompt rows were in flight");
        const long long eq = occ(flog, "on both nodes (");
        fprintf(stderr, "    %s lanes W(1,3), KV check every %s, defer %d: tokens %s, plan %s; checks equal %lld, due with prompt "
                        "rows in flight: drained %lld, deferred %lld\n", a.lanes, every.c_str(), a.defer ? 1 : 0,
                got == ref ? "identical" : "DIFFERENT", chunks_of(llog) == ref_plan ? "identical" : "DIFFERENT", eq, drained,
                deferred);
        CHECK(!ref.empty());
        CHECK(got == ref);
        CHECK(chunks_of(llog) == ref_plan);
        CHECK(eq >= 1);
        CHECK(flog.find("NOT the owner's bytes") == std::string::npos);
        if (a.defer) {
            CHECK(llog.find("RADIANCE_PP_KV_CHECK_DEFER=1 (tp8 loop P37)") != std::string::npos);
            CHECK(deferred >= 1);
            CHECK_EQ(drained, 0);
        } else {
            CHECK(llog.find("RADIANCE_PP_KV_CHECK_DEFER=1 (tp8 loop P37)") == std::string::npos);
            CHECK(drained >= 1);
            CHECK_EQ(deferred, 0);
        }
    }
    /* Under load (the recording: requests alone, then eight at once of different lengths = prompt rows beside decode rows
     * most rounds) with the check due every 4 trunk steps (deferred at most 4 more), at 2 lanes and at 3 lanes with a BG
     * lane: one process's recording, every check equal, and checks due with prompt rows in flight seen (deferred, or
     * drained at the bound). (At N = 1 the bound is one step: every such check drains there, as without the env.) */
    tiny::TempDir d("pp_p37");
    const std::string rec = record_single(d / "single", { "--max-num-seqs", "6" });
    REQUIRE(!rec.empty());
    struct Load { int lanes; const char* bg; };
    for (const Load a : { Load{ 2, nullptr }, Load{ 3, "RADIANCE_PP_BG_LANE=1" } }) {
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=4", "RADIANCE_PP_KV_CHECK_DEFER=1" });
        if (a.bg) env.push_back(a.bg);
        const std::string out = d / ("load-" + std::to_string(a.lanes));
        const PipeRun pr = record_pipeline(out, "1,3", env, lane_shape(a.lanes), lane_shape(a.lanes), env);
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        long long eq = 0, dfr = 0, drn = 0;
        for (size_t at = pr.flog.find("on both nodes ("); at != std::string::npos; at = pr.flog.find("on both nodes (", at + 1)) ++eq;
        for (size_t at = pr.llog.find("deferred while prompt rows"); at != std::string::npos; at = pr.llog.find("deferred while prompt rows", at + 1)) ++dfr;
        for (size_t at = pr.llog.find("at the deferral's bound"); at != std::string::npos; at = pr.llog.find("at the deferral's bound", at + 1)) ++drn;
        fprintf(stderr, "    load, %d lanes%s W(1,3), KV check every 4 + defer: recording %s; checks equal %lld, deferred %lld, drained at "
                        "the bound %lld\n", a.lanes, a.bg ? " + BG lane" : "", same ? "identical" : why.c_str(), eq, dfr, drn);
        CHECK(same);
        CHECK(eq >= 2);
        CHECK(dfr + drn >= 1);
        CHECK(pr.flog.find("NOT the owner's bytes") == std::string::npos);
        CHECK(pr.llog.find("RADIANCE_PP_KV_CHECK_DEFER=1 (tp8 loop P37)") != std::string::npos);
    }
}

/* tp8 loop P31 (RADIANCE_SLOW_ISSUE_MS on both nodes): a cold multi-chunk prompt at 2 and 3 lanes, W(0,2) and W(1,3), with
 * every issue over the threshold (0.0001 ms, passes of >= 1 row) serves one process's tokens and plan; each node says the switch at start and
 * prints slow-issue lines (op, kernel, band, own + launch, the first-launch resolves) up to its budget of 128, the 128th
 * saying so; the leader's prefill timeline names the first chunk's and the longest chunk's head issue. RADIANCE_AQL_PRELOAD=2
 * on the host backend says it has nothing to load and changes nothing; RADIANCE_AQL_MEMSET_OWNQ=1 says itself (the AQL
 * backend's hipMemset is a card's; the host backend never takes it) and changes nothing. */
TEST(a_slow_issue_line_names_the_issues_and_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(1013, 1000);
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
                         cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_PP_PF_TIMELINE=1", "RADIANCE_SLOW_ISSUE_MS=0.0001",
                                          "RADIANCE_SLOW_ISSUE_ROWS=1", "RADIANCE_AQL_PRELOAD=2",
                                          "RADIANCE_AQL_MEMSET_OWNQ=1" })));
        const std::vector<int32_t> got = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        const std::string llog = sv.lead.output(), flog = sv.fol.output();
        CHECK(chunks_of(llog) == ref_plan);
        CHECK(!ref.empty());
        CHECK(got == ref);
        for (const std::string* lg : { &llog, &flog }) {
            CHECK(lg->find("RADIANCE_SLOW_ISSUE_MS=0.0001 (tp8 loop P31)") != std::string::npos);
            CHECK(lg->find("RADIANCE_AQL_PRELOAD=2: not the AQL backend here; nothing to load") != std::string::npos);
            CHECK(lg->find("RADIANCE_AQL_MEMSET_OWNQ=1 (tp8 loop P31)") != std::string::npos);
            const size_t s0 = lg->find("slow issue (RADIANCE_SLOW_ISSUE_MS, tp8 loop P31): card ");
            CHECK(s0 != std::string::npos);
            if (s0 == std::string::npos) continue;
            const std::string sl = lg->substr(s0, lg->find('\n', s0) - s0);
            CHECK(sl.find(": op '") != std::string::npos);
            CHECK(sl.find(" kernel ") != std::string::npos);
            CHECK(sl.find(" ms + launch ") != std::string::npos);
            CHECK(sl.find("first-launch resolve(s)") != std::string::npos);
            if (lg == &llog) fprintf(stderr, "    %s\n", sl.c_str());
            const size_t p0 = lg->find("slow pass issue (RADIANCE_SLOW_ISSUE_MS, tp8 loop P31): card ");
            if (p0 != std::string::npos && lg == &llog)
                fprintf(stderr, "    %s\n", lg->substr(p0, lg->find('\n', p0) - p0).c_str());
            /* the budget: 128 lines, the 128th says so, none after it */
            size_t k = 0, at = 0;
            while ((at = lg->find("slow issue (RADIANCE_SLOW_ISSUE_MS", at)) != std::string::npos) { ++k; ++at; }
            CHECK_EQ(k, (size_t)128);
            CHECK(lg->find("(the 128th: no more issue lines)") != std::string::npos);
        }
        const size_t tl = llog.find("prefill timeline (RADIANCE_PP_PF_TIMELINE=1");
        CHECK(tl != std::string::npos);
        if (tl == std::string::npos) continue;
        const std::string line = llog.substr(tl, llog.find('\n', tl) - tl);
        CHECK(line.find("(the ranks' issue; the first chunk's ") != std::string::npos);
        CHECK(line.find(", the longest ") != std::string::npos);
        CHECK(line.find("events dropped 0, unread 0") != std::string::npos);
        fprintf(stderr, "    %s lanes W(%s): tokens %s, plan %s, pass lines leader %s\n    %s\n", lanes, cut,
                got == ref ? "identical" : "DIFFERENT", chunks_of(llog) == ref_plan ? "identical" : "DIFFERENT",
                llog.find("slow pass issue (RADIANCE_SLOW_ISSUE_MS") != std::string::npos ? "yes" : "none >= 10 ms",
                line.c_str());
    }
    /* the default row floor (512): this model's passes are <= 64 rows, so nothing prints, and the tokens are the same */
    {
        Serving sv;
        REQUIRE(sv.start("0,2", { "--max-num-seqs", "6", "--pp-lanes", "2" },
                         cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_SLOW_ISSUE_MS=0.0001" })));
        const std::vector<int32_t> got = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        CHECK(got == ref);
        for (const std::string& lg : { sv.lead.output(), sv.fol.output() }) {
            CHECK(lg.find("RADIANCE_SLOW_ISSUE_MS=0.0001 (tp8 loop P31)") != std::string::npos);
            CHECK(lg.find("slow issue (RADIANCE_SLOW_ISSUE_MS") == std::string::npos);
            CHECK(lg.find("slow pass issue (RADIANCE_SLOW_ISSUE_MS") == std::string::npos);
        }
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
            /* tp8 loop DECODE round 20: the node's VRAM line for the run, every rank's card named (four host devices) */
            const std::string vkey = std::string("two-node pipeline, ") + who + " node VRAM (RADIANCE_PP_NODE_IDLE=1): the least "
                                     "free a card had over the run, by the driver's counter: ";
            const size_t vat = log.find(vkey);
            CHECK(vat != std::string::npos);
            if (vat != std::string::npos) {
                const std::string vline = log.substr(vat, log.find('\n', vat) - vat);
                int cards = 0;
                for (size_t q = vline.find("card "); q != std::string::npos; q = vline.find("card ", q + 5)) ++cards;
                CHECK(cards >= 1);
                fprintf(stderr, "    %s\n", vline.c_str());
            }
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

/* tp8 loop P46 (RADIANCE_PP_SEND_ASH=N, both nodes; fn4 pp_send_ash): every rank's staging buffer for its share of a row-sharded
 * payload (the budget keeps it), the send rings name them, passes of >= N rows are issued live, and the switch turns the path
 * on / off between steps on both nodes. The tiny model's crossings are not row-sharded (deepseek41's SP passes are) and one rank
 * a node sends no share, so here the plumbing is what is proven: a cold multi-chunk prompt at 2 lanes W(1,3) serves one
 * process's tokens and chunk plan with the env, after POSTing pp_send_ash 0, and after 1 again; both nodes say the start line
 * and the stop line (0 shares: none row-sharded); without the env 1 is refused naming it and nothing is said.
 * (The sharded share's bytes through the staging buffer: ref_test op_stage_send_row_shards_through_rank_staging_world4.) */
TEST(p46_send_ash_staging_and_switch_serve_one_process_tokens_and_plan) {
    REQUIRE(world().ok);
    const std::vector<int32_t> p = prompt_ids(4646, 1000);
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
    auto post = [](int port, int v, int want, const char* named) {
        std::string out;
        Call c;
        const int st = c.send(port, "/admin/fn4", json{ { "pp_send_ash", v } }.dump()) ? c.read(&out) : -1;
        CHECK_EQ(st, want);
        if (named) CHECK(out.find(named) != std::string::npos);
        if (st != want) fprintf(stderr, "    POST pp_send_ash %d: %d %s\n", v, st, out.c_str());
    };
    {
        Serving sv;
        REQUIRE(sv.start("1,3", kLanes2, { "RADIANCE_LOG_STEPS=1", "RADIANCE_PP_SEND_ASH=32" }));
        const std::vector<int32_t> a = complete(sv.port, p, 24);
        post(sv.port, 0, 200, nullptr);
        const std::vector<int32_t> b = complete(sv.port, p, 24);
        post(sv.port, 1, 200, nullptr);
        const std::vector<int32_t> c = complete(sv.port, p, 24);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        const std::string llog = sv.lead.output(), flog = sv.fol.output();
        const std::vector<std::pair<int, int>> plan = chunks_of(llog);
        const bool plan_ok = plan.size() >= ref_plan.size() && std::equal(ref_plan.begin(), ref_plan.end(), plan.begin());
        const bool ls = llog.find("RADIANCE_PP_SEND_ASH=32 -- a row-sharded X1 payload of >= 32 rows") != std::string::npos;
        const bool fs = flog.find("RADIANCE_PP_SEND_ASH=32 -- a row-sharded X2 payload of >= 32 rows") != std::string::npos;
        const bool le = llog.find("shares from the side streams by rank 0, 0 share(s) over the staging buffer") != std::string::npos;
        const bool fe = flog.find("shares from the side streams by rank 0, 0 share(s) over the staging buffer") != std::string::npos;
        const bool l01 = llog.find("fn4 pp_send_ash 1 -> 0 on this node's send rings") != std::string::npos &&
                         llog.find("fn4 pp_send_ash 0 -> 1 on this node's send rings") != std::string::npos;
        const bool f01 = flog.find("fn4 pp_send_ash 1 -> 0 on this node's send rings") != std::string::npos &&
                         flog.find("fn4 pp_send_ash 0 -> 1 on this node's send rings") != std::string::npos;
        fprintf(stderr, "    RADIANCE_PP_SEND_ASH=32, 2 lanes W(1,3): tokens %s / %s / %s (env, switch 0, switch 1), plan %s (%zu chunks); "
                        "start lines leader %d follower %d, stop lines %d / %d, switch lines %d / %d\n",
                a == ref ? "identical" : "DIFFERENT", b == ref ? "identical" : "DIFFERENT", c == ref ? "identical" : "DIFFERENT",
                plan_ok ? "identical" : "DIFFERENT", ref_plan.size(), ls, fs, le, fe, l01, f01);
        if (a != ref || !plan_ok || !ls || !fs || !le || !fe)
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(!ref.empty());
        CHECK(a == ref);
        CHECK(b == ref);
        CHECK(c == ref);
        CHECK(plan_ok);
        CHECK(ls);
        CHECK(fs);
        CHECK(le);
        CHECK(fe);
        CHECK(l01);
        CHECK(f01);
    }
    {   /* without the env: 1 refused by name, 0 taken; no line */
        Serving sv;
        REQUIRE(sv.start("1,3", kLanes2));
        post(sv.port, 1, 400, "RADIANCE_PP_SEND_ASH=N");
        post(sv.port, 0, 200, nullptr);
        CHECK(complete(sv.port, p, 24) == ref);
        kill(sv.lead.child.pid, SIGTERM);
        CHECK_EQ(sv.lead.finish(60), 0);
        CHECK_EQ(sv.fol.finish(30), 0);
        /* (the refusal names the env in the leader's log; the start / stop lines are absent) */
        CHECK(sv.lead.output().find("a row-sharded X1 payload") == std::string::npos);
        CHECK(sv.lead.output().find("shares from the side streams") == std::string::npos);
        CHECK(sv.fol.output().find("RADIANCE_PP_SEND_ASH") == std::string::npos);
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

/* tp8 loop D45 (RADIANCE_PP_NODE_IDLE=2 + RADIANCE_PP_NODE_IDLE_S=1 on both nodes): the node-idle time BY STRETCH -- with a BG
 * step in flight or without -- beside D8's line, with newcomers lent whole to a BG lane (3 lanes, BG_LANE=1, W(1,3), the
 * follower's one-layer segments): every request's tokens are one process's; each node's run line is D8's (the same text, so
 * kit/d8cell.py reads it) and its by-stretch line splits it -- the two stretches' covered time and parts add up to the run
 * line's, both nodes saw time with a BG step in flight, every share is a share -- and the lines come every second. */
TEST(node_idle_by_stretch_splits_the_card_time_and_serves_one_process_tokens) {
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
    std::vector<std::string> env = kSplitEnv;
    for (const char* v : { "RADIANCE_PP_BG_LANE=1", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                           "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_NODE_IDLE=2", "RADIANCE_PP_NODE_IDLE_S=1" })
        env.push_back(v);
    std::vector<std::string> ex = ext;
    ex.push_back("--pp-lanes");
    ex.push_back("3");
    Serving pp;
    REQUIRE(pp.start("1,3", ex, env));
    const std::vector<std::vector<int32_t>> got = split_set(pp.port);
    kill(pp.lead.child.pid, SIGTERM);
    CHECK_EQ(pp.lead.finish(60), 0);
    CHECK_EQ(pp.fol.finish(30), 0);
    REQUIRE_EQ(ref.size(), got.size());
    int differ = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        CHECK(!ref[i].empty());
        if (ref[i] != got[i]) ++differ;
    }
    CHECK_EQ(differ, 0);
    for (const auto& [log, who] : { std::pair<std::string, const char*>{ pp.lead.output(), "leader" },
                                    std::pair<std::string, const char*>{ pp.fol.output(), "follower" } }) {
        CHECK(log.find("RADIANCE_PP_NODE_IDLE=2 -- + the same time by stretch") != std::string::npos);
        const std::string key = std::string("two-node pipeline, ") + who + " node idle (RADIANCE_PP_NODE_IDLE=2, tp8 dev D8): rank 0's card over ";
        const std::string skey = std::string("two-node pipeline, ") + who + " node idle by stretch (RADIANCE_PP_NODE_IDLE=2, tp8 loop D45): "
                                 "rank 0's card over ";
        const auto occ = [&](const std::string& t) {
            long long n = 0;
            for (size_t q = log.find(t); q != std::string::npos; q = log.find(t, q + t.size())) ++n;
            return n;
        };
        const long long n_lines = occ(key), n_split = occ(skey), n_sec = occ(key + "the last 1 s (");
        CHECK(n_lines >= 2);           /* the run's + at least one 1-s window */
        if (n_lines < 2 || n_sec < 1)
            for (size_t q = log.find("NODE_IDLE"); q != std::string::npos; q = log.find("NODE_IDLE", q + 9)) {
                const size_t a0 = log.rfind('\n', q), a1 = log.find('\n', q);
                fprintf(stderr, "    %s: %s\n", who, log.substr(a0 == std::string::npos ? 0 : a0 + 1, std::min<size_t>(260, a1 - a0)).c_str());
                q = a1 == std::string::npos ? q : a1;
            }
        CHECK_EQ(n_split, n_lines);    /* a by-stretch line beside each */
        CHECK(n_sec >= 1);
        const size_t at = log.find(key + "the run (");
        const size_t sat = log.find(skey + "the run: ");
        CHECK(at != std::string::npos);
        CHECK(sat != std::string::npos);
        if (at == std::string::npos || sat == std::string::npos) continue;
        const std::string line = log.substr(at + key.size(), log.find('\n', at) - at - key.size());
        const std::string sline = log.substr(sat + skey.size(), log.find('\n', sat) - sat - skey.size());
        double wall = -1, cov = -1, left = -1, part = -1, ppct = -1, wait = -1, wpct = -1, gap = -1;
        long long idles = -1, parts = -1, recvs = -1;
        CHECK_EQ(std::sscanf(line.c_str(), "the run (%lf s wall, %lf s covered; %lld gaps > %*s s left out, %lf s): in parts %lf s (%lf %%, "
                             "%lld parts), of which waiting on the card for the other node's %*s %lf s (%lf %%, %lld receives); between "
                             "parts %lf s", &wall, &cov, &idles, &left, &part, &ppct, &parts, &wait, &wpct, &recvs, &gap), 11);
        double c[2] = { -1, -1 }, of[2] = { -1, -1 }, ip[2] = { -1, -1 }, wt[2] = { -1, -1 }, bt[2] = { -1, -1 }, wk[2] = { -1, -1 };
        long long np[2] = { -1, -1 };
        const int n = std::sscanf(sline.c_str(), "the run: with a BG step in flight %lf s covered (%lf %% of it): in parts %lf %% (%lld parts), "
                                  "waiting on the card for the other node's %*s %lf %%, between parts %lf %% -> working %lf %%; without %lf s "
                                  "covered (%lf %% of it): in parts %lf %% (%lld parts), waiting on the card for the other node's %*s %lf %%, "
                                  "between parts %lf %% -> working %lf %%", &c[1], &of[1], &ip[1], &np[1], &wt[1], &bt[1], &wk[1], &c[0],
                                  &of[0], &ip[0], &np[0], &wt[0], &bt[0], &wk[0]);
        CHECK_EQ(n, 14);
        fprintf(stderr, "    %s: %s\n    %s: %s\n", who, line.c_str(), who, sline.c_str());
        if (n != 14) continue;
        CHECK(c[1] > 0);                                   /* newcomers ran on the BG lane: both nodes saw BG time */
        CHECK(std::fabs(c[0] + c[1] - cov) <= 0.15);       /* the stretches split the run's covered time (0.1-s rounding) */
        CHECK(std::fabs(of[0] + of[1] - 100.0) <= 0.15);
        CHECK_EQ(np[0] + np[1], parts);
        for (int k = 0; k < 2; ++k) {
            CHECK(ip[k] >= 0 && ip[k] <= 100.05 && bt[k] >= 0 && bt[k] <= 100.05 && wt[k] >= 0);
            CHECK(std::fabs(ip[k] + bt[k] - (c[k] > 0 ? 100.0 : 0.0)) <= 0.15);
            CHECK(wk[k] <= ip[k] + 0.05);
        }
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

/* tp8 loop D32 (LSP, DECODE round 25): fn4 pp_lead_seg on the leader -- form 2 (decode first: a decode tail whose X2 landed and its
 * lane's next head go before the BG step's next segment, one segment in the stream, parts whole while no decode lane is busy), a
 * stride (every s-th declared boundary), form 0 (whole beside the declared boundaries) and the switch POSTed between the two
 * request sets (2 -> 1 -> 0 in one process). Every request's tokens are one process's at 4 lanes BG 2 (FOL_ORDER 2, the BEST
 * form) and 3 lanes BG 1, at a head boundary (W(2,3)) and a tail one only (W(1,2)); a leader with no boundary declared refuses
 * any form but 0 by name. */
TEST(lead_seg_decode_first_serves_one_process_tokens) {
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
    struct Arm { int lanes; const char* cut; int k; const char* lead_seg; const char* form; const char* order; const char* post1;
                 const char* post2; bool rings2 = false; };
    long long first_all = 0, parts_all = 0, both_all = 0;
    for (const Arm a : { Arm{ 4, "2,3", 2, "1", "2", "2", nullptr, nullptr }, Arm{ 4, "1,2", 2, "1", "2", "2", nullptr, nullptr },
                         Arm{ 3, "2,3", 1, "1", "2", nullptr, nullptr, nullptr }, Arm{ 4, "2,3", 2, "1", "34", "2", nullptr, nullptr },
                         Arm{ 4, "2,3", 2, "1", "0", "2", nullptr, nullptr }, Arm{ 4, "2,3", 2, "1", "2", "2", "1", "0" },
                         Arm{ 4, "2,3", 2, nullptr, nullptr, "2", "2", nullptr },
                         /* tp8 loop D32: two boundary rings -- both BG lanes in segments at once (form 2, M9, a tail boundary only) */
                         Arm{ 4, "2,3", 2, "1", "2", "2", nullptr, nullptr, true }, Arm{ 4, "2,3", 2, "1", "1", "2", nullptr, nullptr, true },
                         Arm{ 4, "1,2", 2, "1", "2", "2", nullptr, nullptr, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        if (a.lead_seg) env.push_back(std::string("RADIANCE_PP_LEAD_SEG=") + a.lead_seg);
        if (a.rings2) env.push_back("RADIANCE_PP_LEAD_SEG_RINGS=2");
        if (a.form) env.push_back(std::string("RADIANCE_PP_LEAD_SEG_FORM=") + a.form);
        if (a.order) env.push_back(std::string("RADIANCE_PP_FOL_ORDER=") + a.order);
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        auto post = [&](const char* v, int want) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_lead_seg", std::atoi(v) } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, want);
            if (st != want) fprintf(stderr, "    POST pp_lead_seg %s: %d %s\n", v, st, out.c_str());
            if (want == 400) CHECK(out.find("declared no leader boundary") != std::string::npos);
        };
        if (a.post1 && !a.lead_seg) post(a.post1, 400);   /* no boundary declared: refused by name */
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        if (a.post1 && a.lead_seg) post(a.post1, 200);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        if (a.post2) {   /* applied at the next step's release: one short request after it */
            post(a.post2, 200);
            CHECK(!complete(pp.port, prompt_ids(7, 12), 4).empty());
        }
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
        long long steps = -1, parts = -1, between = -1, inside = -1, polls = -1, first = -1, whole = -1, wtail = -1, both = -1;
        int now = -1, rings = -1;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments, %lld "
                                                    "other lanes' tails while one was in flight (%lld of them between two of its "
                                                    "segments), %lld polls", &steps, &parts, &between, &inside, &polls);
        if (const size_t nw = llog.rfind(" now (tp8 loop D32): "); nw != std::string::npos)
            if (const size_t at = llog.rfind("pp_lead_seg ", nw); at != std::string::npos)
            (void)std::sscanf(llog.c_str() + at, "pp_lead_seg %d now (tp8 loop D32): %lld tails taken first while a BG step was in "
                                                 "segments, %lld BG steps ran their head whole and %lld their tail whole with no decode "
                                                 "lane busy; %d boundary ring(s), %lld BG parts began", &now, &first, &whole, &wtail, &rings,
                              &both);
        fprintf(stderr, "    LEAD_SEG=%s FORM=%s%s%s, FOL_ORDER=%s, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; leader: %lld BG "
                        "steps in %lld segments, %lld tails between two segments, %lld taken first, %lld heads / %lld tails whole with no "
                        "decode lane busy, pp_lead_seg %d at the end; %d ring(s), %lld parts began beside another BG step's\n",
                a.lead_seg ? a.lead_seg : "unset", a.form ? a.form : "unset",
                a.post1 ? (std::string(" + POST ") + a.post1).c_str() : "", a.post2 ? (std::string(" + POST ") + a.post2).c_str() : "",
                a.order ? a.order : "unset", a.k, a.lanes, a.cut, differ, ref.size() + ref_b.size(), steps, parts, inside, first, whole,
                wtail, now, rings, both);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        /* tp8 loop D32: the decode rate beside BG steps (every arm has a BG lane): the run's line counts every decode token */
        {
            long long tb = -1, tn = -1;
            double sb = -1, sn = -1;
            const char* key = "decode beside BG steps over the run (tp8 loop D32): decode tokens with a BG step in flight ";
            if (const size_t at = llog.rfind(key); at != std::string::npos)
                (void)std::sscanf(llog.c_str() + at + std::strlen(key), "%lld in %lf s = %*f tok/s, without %lld in %lf s", &tb, &sb, &tn, &sn);
            fprintf(stderr, "      decode tokens with a BG step in flight %lld in %.2f s, without %lld in %.2f s\n", tb, sb, tn, sn);
            CHECK(tb >= 0 && tn >= 0 && tb + tn > 0);
            CHECK(sb >= 0 && sn >= 0 && sb + sn > 0);
        }
        if (!a.lead_seg) {
            CHECK(steps < 0);   /* no M9 line without the boundaries */
            continue;
        }
        CHECK(llog.find("fn4 pp_lead_seg starts at ") != std::string::npos);
        CHECK_EQ(rings, a.rings2 ? 2 : 1);
        if (a.rings2) {
            CHECK(llog.find("RADIANCE_PP_LEAD_SEG_RINGS=2 -- two boundary rings") != std::string::npos);
            both_all += std::max(0LL, both);
        } else {
            CHECK_EQ(both, 0LL);
        }
        CHECK(now == (a.post2 ? std::atoi(a.post2) : std::atoi(a.form)));
        if (std::strcmp(a.form, "0") == 0) {
            CHECK_EQ(parts, 0LL);   /* form 0: every BG part whole */
        } else {
            CHECK(steps > 0);
            if (std::strcmp(a.cut, "2,3") == 0 && std::strcmp(a.form, "34") != 0) CHECK(parts > steps);
        }
        if ((std::atoi(a.form) & 15) == 2) {
            first_all += std::max(0LL, first);
            parts_all += std::max(0LL, parts);
        }
    }
    fprintf(stderr, "    form 2, all arms: %lld tails taken first while a BG step was in segments, %lld segments; two rings: %lld BG parts "
                    "began their segments beside another BG step's\n", first_all, parts_all, both_all);
    CHECK(first_all > 0);
    CHECK(both_all > 0);
    CHECK(parts_all > 0);
}

/* tp8 loop D49b (DECODE round 35): fn4 pp_lead_seg + 1024 x N -- LSP's segments only while the decode lanes' last steps decoded
 * <= N sequences (Scheduler::decoded_seqs summed over the decode lanes), the parts whole above. Every request's tokens are one
 * process's at 4 lanes BG 2 (FOL_ORDER 2) W(2,3): form 2 + 1024 x 63 (above every load: segments as form 2, the rule never acts),
 * form 2 + 1024 x 1 (the rule acts whenever two or more sequences decode) and the same POSTed back to 2 + 1024 x 63 between the
 * request sets (in process); and at 3 lanes BG 1 with N 1. */
TEST(lead_seg_load_rule_serves_one_process_tokens) {
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
    struct Arm { const char* form; const char* post; int lanes = 4; };
    long long load_all = 0, steps_all = 0;
    for (const Arm a : { Arm{ "64514", nullptr }, Arm{ "1026", nullptr }, Arm{ "1026", "64514" }, Arm{ "1026", nullptr, 3 } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.lanes == 3 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");   /* 3 lanes: one decode lane pair + BG 1 */
        for (const char* v : { "RADIANCE_PP_LEAD_SEG=1", "RADIANCE_PP_FOL_ORDER=2", "RADIANCE_PP_SEG_AHEAD=0",
                               "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        env.push_back(std::string("RADIANCE_PP_LEAD_SEG_FORM=") + a.form);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("2,3", ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        if (a.post) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_lead_seg", std::atoi(a.post) } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
            if (st != 200) fprintf(stderr, "    POST pp_lead_seg %s: %d %s\n", a.post, st, out.c_str());
        }
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        if (a.post)   /* applied at the next step's release: one short request after it, so the run line reads the value */
            CHECK(!complete(pp.port, prompt_ids(7, 12), 4).empty());
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) differ += ref[i] != got[i];
        for (size_t i = 0; i < ref_b.size(); ++i) differ += ref_b[i] != got_b[i];
        long long steps = -1, parts = -1, load = -1;
        int now = -1;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments", &steps,
                                  &parts);
        const char* key = " BG steps ran their parts whole by the decode load rule (+ 1024 x N, tp8 loop D49b)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind(", ", at); b != std::string::npos) (void)std::sscanf(llog.c_str() + b, ", %lld BG", &load);
        if (const size_t nw = llog.rfind(" now (tp8 loop D32): "); nw != std::string::npos)
            if (const size_t at = llog.rfind("pp_lead_seg ", nw); at != std::string::npos)
                (void)std::sscanf(llog.c_str() + at, "pp_lead_seg %d now", &now);
        fprintf(stderr, "    %d lanes W(2,3) pp_lead_seg %s%s: %d of %zu requests differ; leader: %lld BG steps in %lld segments, %lld BG "
                        "steps whole by the load rule, pp_lead_seg %d at the end\n",
                a.lanes, a.form, a.post ? (std::string(" + POST ") + a.post).c_str() : "", differ, ref.size() + ref_b.size(), steps,
                parts, load, now);
        CHECK_EQ(differ, 0);
        if (differ || load < 0) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("; starts with N = ") != std::string::npos);
        CHECK(load >= 0);
        CHECK_EQ(now, std::atoi(a.post ? a.post : a.form));
        if (std::strcmp(a.form, "64514") == 0) {   /* N 63 is above every load (8 sequences): the rule never acts, form 2's segments */
            CHECK_EQ(load, 0LL);
            CHECK(steps > 0);
            CHECK(parts > steps);
        }
        load_all += std::max(0LL, load);
        steps_all += std::max(0LL, steps);
    }
    fprintf(stderr, "    all arms: %lld BG steps whole by the load rule, %lld BG steps in segments\n", load_all, steps_all);
    CHECK(load_all > 0);   /* N 1: the rule acted while two or more sequences decoded */
}

/* tp8 loop D40 (DECODE round 28): fn4 pp_lead_seg + 8 (a BG step's tail whole: no tail boundary kept) and + 512 (the engine's
 * SP word, RadPpInfo.reserved[1] = RAD_PP_LSEG_SP, on the step's HEAD segments only -- the llama stand-in fails any other pass
 * that carries it). Every request's tokens are one process's at 4 lanes BG 2 (FOL_ORDER 2) W(2,3) (a head boundary, no tail one)
 * with form 2 + 8, 2 + 512, 2 + 8 + 512 POSTed to 2 between the request sets, and at W(1,2) (tail boundaries, no head one) with
 * 2 + 512 (no head segment: no word) and 2 + 8 (every tail whole: no segment at all). */
TEST(lead_seg_tail_whole_and_sp_word_serve_one_process_tokens) {
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
    struct Arm { const char* cut; const char* form; const char* post; };
    long long spseg_all = 0, tw8_all = 0;
    for (const Arm a : { Arm{ "2,3", "10", nullptr }, Arm{ "2,3", "514", nullptr }, Arm{ "2,3", "522", "2" },
                         Arm{ "1,2", "514", nullptr }, Arm{ "1,2", "10", nullptr } }) {
        std::vector<std::string> env = kSplitEnv;
        for (const char* v : { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_LEAD_SEG=1", "RADIANCE_PP_FOL_ORDER=2", "RADIANCE_PP_SEG_AHEAD=0",
                               "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        env.push_back(std::string("RADIANCE_PP_LEAD_SEG_FORM=") + a.form);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back("4");
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        if (a.post) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_lead_seg", std::atoi(a.post) } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
        }
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) differ += ref[i] != got[i];
        for (size_t i = 0; i < ref_b.size(); ++i) differ += ref_b[i] != got_b[i];
        long long steps = -1, parts = -1, spseg = -1, tw8 = -1;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments", &steps,
                                  &parts);
        const char* key = " head segments carried the SP word (+ 512), ";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("; ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "; %lld head segments carried the SP word (+ 512), %lld BG steps kept", &spseg, &tw8);
        const bool tail_b = llog.find("and its tail at [none: whole]") == std::string::npos;
        fprintf(stderr, "    W(%s) pp_lead_seg %s%s: %d of %zu requests differ; leader: %lld BG steps in %lld segments, %lld head segments "
                        "with the SP word, %lld tails kept whole by + 8 (tail boundary declared: %d)\n",
                a.cut, a.form, a.post ? (std::string(" + POST ") + a.post).c_str() : "", differ, ref.size() + ref_b.size(), steps,
                parts, spseg, tw8, (int)tail_b);
        CHECK_EQ(differ, 0);
        if (differ || spseg < 0) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("llama: the engine's SP word") == std::string::npos);
        CHECK(flog.find("llama: the engine's SP word") == std::string::npos);
        CHECK(spseg >= 0 && tw8 >= 0);
        CHECK(steps > 0);
        const int f = std::atoi(a.form);
        if (std::strcmp(a.cut, "1,2") == 0) CHECK_EQ(spseg, 0LL);   /* no head boundary: no head segment carries it */
        else if (f & 512) CHECK(spseg > 0);
        else CHECK_EQ(spseg, 0LL);
        if (!(f & 8)) CHECK_EQ(tw8, 0LL);
        else if (tail_b) CHECK(tw8 > 0);
        if (std::strcmp(a.cut, "1,2") == 0 && (f & 8)) CHECK_EQ(parts, 0LL);   /* no head boundary, the tail whole: no segment */
        spseg_all += std::max(0LL, spseg);
        tw8_all += std::max(0LL, tw8);
    }
    fprintf(stderr, "    all arms: %lld head segments with the SP word, %lld tails kept whole by + 8\n", spseg_all, tw8_all);
    CHECK(spseg_all > 0);
    CHECK(tw8_all > 0);
}

/* tp8 loop F75 (DECODE round 28; loop FINDINGS F75, COORDINATOR 20:06): the order levers of lq197's arm `all` that this branch
 * carries -- LSP (pp_lead_seg form 2, every boundary), P23s (RADIANCE_PP_BG_ROOM = MNBT + RADIANCE_PP_BG_JOIN_MS=250: batched BG
 * steps) and R9 (the follower's body reserve from layer 4) -- alone and in every pair / all three, with the KV copy on
 * (RADIANCE_LLAMA_PP_KVCOPY: X1 carries layer h - 1's KV, deepseek41's source 8 at W(10,32)) and the KV-copy check at every
 * trunk step, at W(2,5) of the six-layer llama (head boundary 1, follower segments 3 / 4), 4 lanes BG 2 FOL_ORDER 2, with
 * decoders, whole newcomers in bursts and LONG prompts (3-4 chunks of 700, lent between lanes): every request's tokens are one
 * process's and no check finds the copy different. (MERGE's pp_merge is not on this branch.) */
std::vector<std::vector<int32_t>> f75_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 71, 72, 73, 74 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 160); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (uint64_t s : { 81, 82, 83 }) {   /* a burst of whole newcomers (P23s batches them) */
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 220 + 60 * (s - 81)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (uint64_t s : { 91, 92 }) {       /* long prompts: 3 / 4 chunks */
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 1500 + 700 * (s - 91)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    for (uint64_t s : { 84, 85, 86 }) {   /* newcomers beside the long ones */
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 300 + 50 * (s - 84)), 12); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

/* tp8 loop F81: + the position-order guard's count (steps carrying a prompt piece that is not a whole fresh prompt run whole: > 0 over
 * the guarded LSP arms, 0 in 'all-ng' = pp_lead_seg + 65536, the F75 / F81 form without the guard, which segments one step more) */
TEST(f75_order_levers_alone_and_together_keep_the_kv_copy) {
    REQUIRE(world().ok);
    REQUIRE(world6().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "4096" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        one.model = world6().llama_rad;
        REQUIRE(one.start("", ext));
        ref = f75_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    /* tp8 loop F81: noguard = pp_lead_seg + 65536 (LSP without its position-order guard, the F75 / F81 form) */
    struct Arm { const char* name; bool lsp, p23, r9; bool noguard = false; };
    int bad = 0;
    long long guard_all = 0;
    for (const Arm a : { Arm{ "none", false, false, false }, Arm{ "LSP", true, false, false }, Arm{ "P23s", false, true, false },
                         Arm{ "R9", false, false, true }, Arm{ "LSP+P23s", true, true, false }, Arm{ "LSP+R9", true, false, true },
                         Arm{ "P23s+R9", false, true, true }, Arm{ "all", true, true, true }, Arm{ "all-ng", true, true, true, true } }) {
        std::vector<std::string> env = kSplitEnv;
        for (const char* v : { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2", "RADIANCE_PP_SEG_AHEAD=0",
                               "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1", "RADIANCE_PP_LEAD_SEG=1",
                               "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_LEAD_SEG_RINGS=2" })
            env.push_back(v);
        env.push_back(a.lsp ? (a.noguard ? "RADIANCE_PP_LEAD_SEG_FORM=65538" : "RADIANCE_PP_LEAD_SEG_FORM=2") : "RADIANCE_PP_LEAD_SEG_FORM=0");
        if (a.p23) { env.push_back("RADIANCE_PP_BG_ROOM=700"); env.push_back("RADIANCE_PP_BG_JOIN_MS=250"); }
        if (a.r9) env.push_back("RADIANCE_PP_FOL_RESERVE=4");
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back("4");
        Serving pp;
        pp.model = world6().llama_rad;
        REQUIRE(pp.start("2,5", ex, env));
        const std::vector<std::vector<int32_t>> got = f75_set(pp.port);
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
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const bool notowner = flog.find("NOT the owner's bytes") != std::string::npos;
        long long steps = -1, parts = -1, cont = 0;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments", &steps,
                                  &parts);
        long long lent = 0;
        for (size_t at = llog.find(" chunk(s) planned by its lane "); at != std::string::npos; at = llog.find(" chunk(s) planned by its lane ", at + 1)) {
            long long ln = 0, n = 0;
            if (std::sscanf(llog.c_str() + at, " chunk(s) planned by its lane %lld, %lld lent", &ln, &n) == 2) lent += n;
            ++cont;
        }
        long long bg_n = -1, bg_newc = -1, bg_b2 = -1, held = -1;   /* P23's run line, R9's follower line */
        if (const size_t at = llog.rfind("two-node pipeline: BG steps over the run (RADIANCE_PP_BG_ROOM="); at != std::string::npos)
            if (const size_t b = llog.find("): ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "): %lld steps with %lld newcomers; %lld steps", &bg_n, &bg_newc, &bg_b2);
        if (const size_t at = flog.rfind("follower: body reserve "); at != std::string::npos)
            if (const size_t b = flog.find("): ", at); b != std::string::npos) (void)std::sscanf(flog.c_str() + b, "): %lld BG bodies held", &held);
        long long guard = -1;   /* tp8 loop F81: steps the position-order guard kept whole */
        const char* gkey = " steps ran their parts whole by the position-order guard";
        if (const size_t at = llog.rfind(gkey); at != std::string::npos)
            if (const size_t b = llog.rfind(", ", at); b != std::string::npos) guard = std::atoll(llog.c_str() + b + 2);
        fprintf(stderr, "    %-9s: %d of %zu requests differ; KV-copy checks %lld, %lld differ%s; leader: %lld BG steps in %lld segments; "
                        "%lld prompts planned by lanes, %lld chunks lent; P23: %lld BG steps, %lld newcomers, %lld with >= 2; R9: %lld bodies held; "
                        "F81 guard kept %lld steps whole\n",
                a.name, differ, ref.size(), checks, kdiff, notowner ? " (NOT the owner's bytes)" : "", steps, parts, cont, lent, bg_n, bg_newc,
                bg_b2, held, guard);
        if (a.lsp) {
            CHECK(guard >= 0);
            if (!a.noguard) guard_all += std::max(0LL, guard);
            if (a.noguard) CHECK_EQ(guard, 0LL);
        }
        if (a.p23) CHECK(bg_n > 0);
        if (a.r9) CHECK(held >= 0);
        CHECK_EQ(differ, 0);
        CHECK(!notowner);
        CHECK(checks > 0);
        if (differ || notowner) {
            ++bad;
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        }
    }
    fprintf(stderr, "    F81 position-order guard: %lld steps kept whole over the guarded LSP arms\n", guard_all);
    CHECK(guard_all > 0);   /* steps carrying a piece of a longer prompt reach LSP here (P23s or not) */
    CHECK_EQ(bad, 0);
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

/* tp8 loop D39 (RADIANCE_PP_TAIL_BAND=1 on the leader, RADIANCE_LLAMA_PP_TBAND=1 on both): a BG step of fresh prompts runs as a
 * TAIL band pass -- the follower over [1, 2) of W(1,3), X2 at 2, then only layer 2's k / v from its own rows into its own cache
 * (the layer's first ops at the same rows); the leader's tail from 2 (layer 2 declared there too) -- and every request's tokens
 * are one process's at 2 lanes BG 1, 3 lanes BG 1 (whole, + follower segments), 4 lanes BG 2 (FOL_ORDER 2); negative control: the
 * follower leaves the k / v stores out (RADIANCE_PP_TEST_TBAND_SKIP=1) -> requests differ (the decode owner reads keys it never
 * got). Step 2c (DECODE round 31): fn4 pp_tband, the leader's in-process control -- 0 before the first set (its fresh BG steps run
 * W(1,3)'s, the band layer declared: counted on the end line), 1 before the second (tail band passes again), the same tokens; and
 * without RADIANCE_PP_TAIL_BAND the leader refuses pp_tband 1 by name (400) and runs no tail band pass. */
TEST(tail_band_passes_serve_one_process_tokens) {
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
    /* sw: 0 = fn4 pp_tband 0 then 1 (the in-process control), 1 = pp_tband 1 refused (RADIANCE_PP_TAIL_BAND unset), -1 none */
    struct Arm { int lanes; int k; bool fseg; bool fo2 = false; bool skip = false; int sw = -1; };
    long long tb_all = 0, off_all = 0;
    int skip_differ = 0;
    for (const Arm a : { Arm{ 3, 1, false }, Arm{ 3, 1, true }, Arm{ 4, 2, true, true }, Arm{ 2, 1, false },
                         Arm{ 3, 1, true, false, false, 0 }, Arm{ 3, 1, false, false, false, 1 }, Arm{ 3, 1, false, false, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_LLAMA_PP_TBAND=1");
        if (a.sw != 1) env.push_back("RADIANCE_PP_TAIL_BAND=1");
        env.push_back("RADIANCE_PP_READY_FIRST=1");
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        if (a.fo2) env.push_back("RADIANCE_PP_FOL_ORDER=2");
        if (a.skip) env.push_back("RADIANCE_PP_TEST_TBAND_SKIP=1");
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("1,3", ex, env));
        auto post = [&](int v, int want) {   /* step 2c: fn4 pp_tband on the leader */
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_tband", v } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, want);
            if (st != want) fprintf(stderr, "    POST pp_tband %d: %d %s\n", v, st, out.c_str());
            if (want == 400) CHECK(out.find("RADIANCE_PP_TAIL_BAND=1 on the leader at start") != std::string::npos);
        };
        if (a.sw == 0) post(0, 200);
        if (a.sw == 1) post(1, 400);
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        if (a.sw == 0) post(1, 200);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        /* loop lq215 (DECODE round 32): the kit's arm undo POSTs the start's read-back, -1 = the start's RADIANCE_PP_TAIL_BAND --
         * taken with and without a tail band */
        if (a.sw >= 0) post(-1, 200);
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
        long long tb = -1;
        const char* key = " lane steps ran as tail band passes (RADIANCE_PP_TAIL_BAND=1, tp8 loop D39)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                tb = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        long long off = -1;   /* step 2c: fresh BG steps fn4 pp_tband 0 kept W(1,3)'s */
        if (const size_t at = llog.rfind(" fresh BG steps ran W(h, t)'s under fn4 pp_tband 0"); at != std::string::npos)
            if (const size_t b = llog.rfind("; ", at); b != std::string::npos) off = std::atoll(llog.c_str() + b + 2);
        fprintf(stderr, "    TAIL BAND (llama stand-in, one layer), %d lanes BG %d W(1,3)%s%s%s%s: %d of %zu requests differ; %lld lane steps "
                        "ran as tail band passes, %lld fresh BG steps W(1,3)'s under pp_tband 0\n", a.lanes, a.k,
                a.fseg ? " + follower segments" : "", a.fo2 ? " + FOL_ORDER 2" : "",
                a.sw == 0 ? " + fn4 pp_tband 0 (first set) / 1 (second)" : a.sw == 1 ? " WITHOUT RADIANCE_PP_TAIL_BAND (pp_tband 1 refused)" : "",
                a.skip ? " + the k / v stores LEFT OUT (negative control)" : "", differ, ref.size() + ref_b.size(), tb, off);
        if (a.skip) {
            skip_differ = differ;
            CHECK(tb > 0);
            continue;
        }
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(differ, 0);
        if (a.sw == 1) {   /* not taken: no tail band pass, no end line */
            CHECK(llog.find("RADIANCE_PP_TAIL_BAND=1 -- a BG lane's step of fresh prompts runs as a tail band pass") == std::string::npos);
            CHECK(tb < 0);
            continue;
        }
        CHECK(llog.find("RADIANCE_PP_TAIL_BAND=1 -- a BG lane's step of fresh prompts runs as a tail band pass") != std::string::npos);
        CHECK(tb > 0);
        if (a.sw == 0) {
            CHECK(off > 0);
            off_all += std::max(0LL, off);
        } else {
            CHECK_EQ(off, 0LL);
        }
        tb_all += std::max(0LL, tb);
    }
    fprintf(stderr, "    tail band passes, all arms: %lld; kept W(1,3)'s under pp_tband 0: %lld; the negative control: %d requests differ\n",
            tb_all, off_all, skip_differ);
    CHECK(tb_all > 0);
    CHECK(off_all > 0);
    CHECK(skip_differ > 0);
}

/* tp8 loop D39j2 step 1 (DECODE round 32, QUEUE D39j2): the engine's tail band at DEPTH 2 -- RADIANCE_LLAMA_PP_TBAND=2 declares
 * layers t - 2 and t - 1 on the leader too; a BG step of fresh prompts runs W(0,3) as the follower over [0, 1), X2 at 1, then layer 1
 * whole and layer 2's k / v from its own rows (the decode owner's keys of both band layers; llama's attention is not windowed, so
 * the stand-in saves nothing -- this proves the engine's j = 2 plumbing: frames, the follower's pass end, the leader's tail from
 * t - 2, the KV of both layers on the owner), the leader's tail from 1: every request's tokens = one process's at 3 lanes BG 1
 * (whole, + follower segments) and 4 lanes BG 2 (FOL_ORDER 2); negative control: the follower's stores left out -> requests differ. */
TEST(tail_band_at_depth_two_serves_one_process_tokens) {
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
    struct Arm { int lanes; int k; bool fseg; bool fo2 = false; bool skip = false; };
    long long tb_all = 0;
    int skip_differ = 0;
    for (const Arm a : { Arm{ 3, 1, false }, Arm{ 3, 1, true }, Arm{ 4, 2, true, true }, Arm{ 3, 1, false, false, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_LLAMA_PP_TBAND=2");
        env.push_back("RADIANCE_PP_TAIL_BAND=1");
        env.push_back("RADIANCE_PP_READY_FIRST=1");
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        if (a.fo2) env.push_back("RADIANCE_PP_FOL_ORDER=2");
        if (a.skip) env.push_back("RADIANCE_PP_TEST_TBAND_SKIP=1");
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("0,3", ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (ref[i] != got[i]) ++differ;
        for (size_t i = 0; i < ref_b.size(); ++i)
            if (ref_b[i] != got_b[i]) ++differ;
        long long tb = -1;
        const char* key = " lane steps ran as tail band passes (RADIANCE_PP_TAIL_BAND=1, tp8 loop D39)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                tb = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        fprintf(stderr, "    TAIL BAND DEPTH 2 (llama stand-in), %d lanes BG %d W(0,3)%s%s%s: %d of %zu requests differ; %lld lane steps ran as "
                        "tail band passes\n", a.lanes, a.k, a.fseg ? " + follower segments" : "", a.fo2 ? " + FOL_ORDER 2" : "",
                a.skip ? " + the k / v stores LEFT OUT (negative control)" : "", differ, ref.size() + ref_b.size(), tb);
        CHECK(tb > 0);
        if (a.skip) {
            skip_differ = differ;
            continue;
        }
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(differ, 0);
        /* the engine's start line names the depth: the follower's pass ends at t - 2 = 1 */
        CHECK(llog.find("RADIANCE_PP_TAIL_BAND=1 -- a BG lane's step of fresh prompts runs as a tail band pass") != std::string::npos);
        tb_all += std::max(0LL, tb);
    }
    fprintf(stderr, "    depth-2 tail band passes, all arms: %lld; the negative control: %d requests differ\n", tb_all, skip_differ);
    CHECK(tb_all > 0);
    CHECK(skip_differ > 0);
}

/* tp8 loop D46 step 1 (DECODE round 33): the tail band's COPY -- with RADIANCE_LLAMA_PP_TBAND_COPY=1 beside TBAND=1 every pass that is
 * no tail band pass carries layer t - 1's q|k|v rows in X2 and the leader stores its k / v into its own layer t - 1 cache, so the
 * leader holds that layer's cache of every sequence, and fn4 pp_tband 3 / RADIANCE_PP_TAIL_BAND=3 runs a DECODE lane's step as a
 * tail band pass too while a BG step is in flight (the leader's layer t - 1 reads its own cache). At W(1,3) of the four-layer llama
 * (band layer 2), with S6's X1 copy and the KV-copy check on every trunk step (now two copies: layer 0 on the follower, layer 2 on
 * the leader): every request's tokens are one process's at 3 lanes BG 1 (+ follower segments), 4 lanes BG 2 FOL_ORDER 2, pp_tband 1
 * then 3 by POST in one process (the in-process control), and no check differs; decode lanes' steps ran as band passes (> 0).
 * Negative control: the leader's store left out (RADIANCE_PP_TEST_TCOPY_SKIP=1) -> tokens differ or the check names the copy. No
 * copy declared: RADIANCE_PP_TAIL_BAND=3 runs as 1 (named) and a POST of 3 is refused. tp8 loop D50: fn4 pp_tband 4 (decode steps
 * band in clean stretches too: band passes with no BG step in flight > 0), RADIANCE_PP_TAIL_BAND=1028 (4 + 1024 x 1: the load rule
 * judges clean steps) and 64516 (N = 63: every clean step kept W(1,3)'s, BG stretches still band) -- the same tokens. */
TEST(tail_band_copy_runs_decode_lane_steps_as_band_passes_with_one_process_tokens) {
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
    /* sw: 13 = POST pp_tband 1 before the first set and 3 before the second; nocopy = TBAND_COPY unset (3 runs as 1, POST 3 refused) */
    /* tp8 loop D50: tb = the start's RADIANCE_PP_TAIL_BAND (3; 1028 = 4 + 1024 x 1), tv = fn4 pp_tband POSTed before the first set
     * (4 = 3 in every stretch: decode band passes with no BG step in flight) */
    /* tp8 loop D51: bound = RADIANCE_PP_TAIL_BAND=5 with the follower's load report (RADIANCE_PP_NODE_IDLE=1 + RADIANCE_PP_LOAD_REPORT_MS
     * on both: the follower reports, the leader reads its own share) -- the band follows the measured bound */
    struct Arm { int lanes; int k; bool fseg; bool fo2 = false; bool skip = false; int sw = -1; bool nocopy = false; const char* tb = "3";
                 int tv = -1; bool bound = false; };
    long long dec_all = 0, clean_all = 0, low_all = 0;
    int skip_bad = 0;
    for (const Arm a : { Arm{ 3, 1, false }, Arm{ 3, 1, true }, Arm{ 4, 2, true, true }, Arm{ 3, 1, true, false, false, 13 },
                         Arm{ 3, 1, false, false, false, -1, true }, Arm{ 3, 1, false, false, true },
                         Arm{ 3, 1, true, false, false, -1, false, "3", 4 }, Arm{ 4, 2, true, true, false, -1, false, "1028" },
                         Arm{ 3, 1, false, false, false, -1, false, "64516" },
                         Arm{ 3, 1, true, false, false, -1, false, "5", -1, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back(std::string("RADIANCE_PP_TAIL_BAND=") + a.tb);
        for (const char* v : { "RADIANCE_LLAMA_PP_TBAND=1", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" })
            env.push_back(v);
        if (!a.nocopy) env.push_back("RADIANCE_LLAMA_PP_TBAND_COPY=1");
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        if (a.fo2) env.push_back("RADIANCE_PP_FOL_ORDER=2");
        if (a.skip) env.push_back("RADIANCE_PP_TEST_TCOPY_SKIP=1");
        if (a.bound)
            for (const char* v : { "RADIANCE_PP_NODE_IDLE=1", "RADIANCE_PP_LOAD_REPORT_MS=5", "RADIANCE_PP_TBAND_BOUND_MS=100" })
                env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("1,3", ex, env));
        auto post = [&](int v, int want, const char* why) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_tband", v } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, want);
            if (st != want) fprintf(stderr, "    POST pp_tband %d: %d %s\n", v, st, out.c_str());
            if (why) CHECK(out.find(why) != std::string::npos);
        };
        if (a.sw == 13) post(1, 200, nullptr);
        if (a.nocopy) post(3, 400, "the plugin declared no tail band copy");
        if (a.nocopy) post(4, 400, "the plugin declared no tail band copy");   /* tp8 loop D50 */
        if (a.tv > 0) post(a.tv, 200, nullptr);
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        long long dec_mid = -1;
        if (a.sw == 13) post(3, 200, nullptr);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        if (a.sw >= 0 || a.tv > 0) post(-1, 200, nullptr);
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
        long long tb = -1, dec = -1, dskip = -1, held = -1;
        const char* key = " lane steps ran as tail band passes (RADIANCE_PP_TAIL_BAND=1, tp8 loop D39)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                tb = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        const char* dkey = " decode lanes' steps ran as tail band passes while a BG step was in flight (pp_tband 3, tp8 loop D46); ";
        if (const size_t at = llog.rfind(dkey); at != std::string::npos) {
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                dec = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
            (void)std::sscanf(llog.c_str() + at + std::strlen(dkey), "%lld such steps kept W(h, t)'s for a sequence whose band state the "
                                                                     "leader does not hold; %lld sequences held", &dskip, &held);
        }
        long long dclean = -1, dlow = -1;   /* tp8 loop D50 */
        const char* ckey = " of the band passes had no BG step in flight (pp_tband 4, tp8 loop D50), ";
        if (const size_t at = llog.rfind(ckey); at != std::string::npos) {
            if (const size_t b = llog.rfind("; ", at); b != std::string::npos) dclean = std::atoll(llog.c_str() + b + 2);
            dlow = std::atoll(llog.c_str() + at + std::strlen(ckey));
        }
        (void)dec_mid;
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const bool notowner = flog.find("NOT the owner's bytes") != std::string::npos;
        const bool copy2 = flog.find("layer 2,") != std::string::npos;   /* the check named the band layer's copy */
        fprintf(stderr, "    TAIL COPY (llama stand-in), %d lanes BG %d W(1,3), TAIL_BAND=%s%s%s%s%s%s%s: %d of %zu requests differ; %lld lane steps "
                        "ran as tail band passes, %lld of them decode lanes' (pp_tband 3 / 4), %lld kept W(1,3)'s for a sequence not held, %lld "
                        "held; %lld band passes with no BG step in flight, %lld clean steps kept W(1,3)'s at a load <= N (D50); KV-copy checks "
                        "%lld, %lld differ%s%s\n", a.lanes, a.k, a.tb, a.fseg ? " + follower segments" : "", a.fo2 ? " + FOL_ORDER 2" : "",
                a.sw == 13 ? " + fn4 pp_tband 1 (first set) / 3 (second)" : "", a.tv > 0 ? " + fn4 pp_tband 4 (both sets)" : "",
                a.nocopy ? " WITHOUT the copy (3 runs as 1)" : "", a.skip ? " + the leader's store LEFT OUT (negative control)" : "", differ,
                ref.size() + ref_b.size(), tb, dec, dskip, held, dclean, dlow, checks, kdiff, notowner ? " (NOT the owner's bytes)" : "",
                copy2 ? "; layer 2 checked" : "");
        if (a.skip) {
            skip_bad = differ + (notowner ? 1 : 0);
            CHECK(dec > 0);
            continue;
        }
        if (differ || notowner) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(differ, 0);
        CHECK(!notowner);
        CHECK(checks > 0);
        CHECK_EQ(kdiff, 0LL);
        CHECK(tb > 0);
        if (a.nocopy) {
            CHECK(llog.find("RADIANCE_PP_TAIL_BAND=3 runs as 1: the plugin declared no tail band copy") != std::string::npos);
            CHECK(dec < 0);   /* no copy: no D46 line */
            continue;
        }
        CHECK(copy2);
        CHECK(llog.find("the plugin keeps the tail band layers' state on the leader for every sequence") != std::string::npos);
        CHECK(a.bound ? dec >= 0 : dec > 0);   /* tp8 loop D51: under 5 the measured bound may keep every step on the follower */
        dec_all += std::max(0LL, dec);
        CHECK(dclean >= 0);   /* tp8 loop D50: the run line's counters */
        CHECK(dlow >= 0);
        {   /* ... and the tail band line (periodic at RADIANCE_PP_BGRATE_LINE_S, here the run's end): the same counts + the judged steps */
            long long judged = -1, tbn = -1, tbc = -1;
            const char* tkey = "two-node pipeline: tail band over the run (tp8 loop D46 / D50): ";
            if (const size_t at = llog.rfind(tkey); at != std::string::npos)
                (void)std::sscanf(llog.c_str() + at + std::strlen(tkey), "%lld decode lanes' steps judged, %lld ran as tail band passes (%lld",
                                  &judged, &tbn, &tbc);
            CHECK(judged >= dec);
            CHECK(judged > 0);
            CHECK_EQ(tbn, dec);
            CHECK_EQ(tbc, dclean);
        }
        if (a.tv == 4) CHECK(dclean > 0);   /* 4: decode band passes with no BG step in flight */
        if (a.tv < 0 && std::strcmp(a.tb, "3") == 0) {
            CHECK_EQ(dclean, 0LL);   /* 3: only while a BG step is in flight */
            CHECK_EQ(dlow, 0LL);
        }
        if (std::strcmp(a.tb, "1028") == 0) {
            CHECK(llog.find("RADIANCE_PP_TAIL_BAND=1028 from the start (N = 1)") != std::string::npos);
            CHECK(dclean + dlow > 0);   /* the load rule judged clean steps */
        }
        if (std::strcmp(a.tb, "64516") == 0) {   /* N = 63: the load never above it -- every clean step kept W(1,3)'s, BG ones band */
            CHECK(llog.find("RADIANCE_PP_TAIL_BAND=64516 from the start (N = 63)") != std::string::npos);
            CHECK(dlow > 0);
            CHECK_EQ(dclean, 0LL);
        }
        if (a.bound) {   /* tp8 loop D51: the follower's load reports reached the leader and decided decode steps */
            long long jn = -1, jb = -1, jk = -1, jf = -1;
            unsigned long long rep = 0;
            const char* bkey = "two-node pipeline: the tail band by the measured bound over the run (fn4 pp_tband 5, tp8 loop D51): ";
            if (const size_t at = llog.rfind(bkey); at != std::string::npos) {
                (void)std::sscanf(llog.c_str() + at + std::strlen(bkey), "%lld decode lanes' steps judged by it, %lld banded (the follower "
                                  "busier), %lld kept the band layer on the follower (the leader busier), %lld fell back", &jn, &jb, &jk, &jf);
                if (const size_t r = llog.find("load reports came", at); r != std::string::npos)
                    if (const size_t b = llog.rfind("; ", r); b != std::string::npos) rep = std::strtoull(llog.c_str() + b + 2, nullptr, 10);
            }
            fprintf(stderr, "    BOUND (pp_tband 5): %lld decode steps judged by the measured bound, %lld banded, %lld kept on the follower, "
                            "%lld fell back to 3's rule; %llu load reports came\n", jn, jb, jk, jf, rep);
            CHECK(flog.find("RADIANCE_PP_LOAD_REPORT_MS=5 -- this node's card working share") != std::string::npos);
            CHECK(llog.find("fn4 pp_tband 5 runs a held decode lane's step as a tail band pass while the FOLLOWER's card") != std::string::npos);
            CHECK(llog.find("RADIANCE_PP_TAIL_BAND=5 from the start") != std::string::npos);
            CHECK(rep > 0);
            CHECK(jn > 0);
            CHECK_EQ(jn, jb + jk);
            CHECK(jf >= 0);
        }
        clean_all += std::max(0LL, dclean);
        low_all += std::max(0LL, dlow);
    }
    fprintf(stderr, "    all arms: %lld decode lanes' steps as tail band passes (%lld with no BG step in flight, %lld clean steps kept W(1,3)'s "
                    "by the load rule); negative control: %d requests differ / copy named\n", dec_all, clean_all, low_all, skip_bad);
    CHECK(dec_all > 0);
    CHECK(clean_all > 0);
    CHECK(skip_bad > 0);
}

/* tp8 loop D39 step 2c (DECODE round 31): the tail band in BEST's pipeline shape -- the six-layer llama at W(2,5) (the leader declares
 * layer 4 too), 4 lanes BG 2, FOL_ORDER 2, the follower's segments at every layer, the KV copy (X1 carries layer 1's KV = deepseek41's
 * source 8 at W(10,32)) with its check at EVERY trunk step, over F75's set (decoders, a burst of whole newcomers, long prompts of 3 / 4
 * chunks: their first chunks are band steps, the rest W(2,5)'s, newcomers beside them): every request's tokens = one process's and
 * the copy = the owner's bytes, with the band alone, with P23s's batched BG steps (several fresh prompts in ONE band pass), and with
 * fn4 pp_tband 0 then 1 in process (the control: the same tokens, fewer band passes). Every arm then serves ONE LONE long prompt
 * (1,900 rows: three chunks on its home decode lane, no decoder beside it -- the cold cells' shape). D39b: RADIANCE_PP_TAIL_BAND=2
 * bands those prompt-only steps on any lane, the later chunks while every chunk of the prompt was banded (the leader then holds
 * layer 4's KV of all its rows): the same tokens and copy, band passes with a later chunk > 0; negative control
 * RADIANCE_PP_TEST_TBAND_CONT_ONLY=1 (later chunks banded, first chunks not: the leader's layer 4 reads KV it never wrote) -> differ. */
TEST(d39_tail_band_keeps_one_process_tokens_and_the_kv_copy_in_bests_shape) {
    REQUIRE(world().ok);
    REQUIRE(world6().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "4096" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        one.model = world6().llama_rad;
        REQUIRE(one.start("", ext));
        ref = f75_set(one.port);
        ref.push_back(complete(one.port, prompt_ids(97, 1900), 12));   /* D39b: the lone long prompt */
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { const char* name; bool p23; bool sw; const char* tbv = "1"; bool cont_only = false; };
    long long tb_all = 0, cont_all = 0;
    int ctl_differ = 0;
    for (const Arm a : { Arm{ "band", false, false }, Arm{ "band+P23s", true, false }, Arm{ "band, pp_tband 0 / 1", false, true },
                         Arm{ "band 2 (later chunks)", false, false, "2" }, Arm{ "band 2, CONT_ONLY (ctl)", false, false, "2", true } }) {
        std::vector<std::string> env = kSplitEnv;
        for (const char* v : { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2", "RADIANCE_PP_SEG_AHEAD=0",
                               "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1", "RADIANCE_LLAMA_PP_TBAND=1" })
            env.push_back(v);
        env.push_back(std::string("RADIANCE_PP_TAIL_BAND=") + a.tbv);
        if (a.cont_only) env.push_back("RADIANCE_PP_TEST_TBAND_CONT_ONLY=1");
        if (a.p23) { env.push_back("RADIANCE_PP_BG_ROOM=700"); env.push_back("RADIANCE_PP_BG_JOIN_MS=250"); }
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back("4");
        Serving pp;
        pp.model = world6().llama_rad;
        REQUIRE(pp.start("2,5", ex, env));
        if (a.sw) {
            std::string out;
            Call c;
            CHECK_EQ(c.send(pp.port, "/admin/fn4", json{ { "pp_tband", 0 } }.dump()) ? c.read(&out) : -1, 200);
        }
        std::vector<std::vector<int32_t>> got = f75_set(pp.port);
        if (a.sw) {   /* the set again with the band back on: the same tokens */
            std::string out;
            Call c;
            CHECK_EQ(c.send(pp.port, "/admin/fn4", json{ { "pp_tband", 1 } }.dump()) ? c.read(&out) : -1, 200);
            const std::vector<std::vector<int32_t>> got2 = f75_set(pp.port);
            REQUIRE_EQ(got2.size() + 1, ref.size());
            for (size_t i = 0; i < got2.size(); ++i) CHECK(got2[i] == ref[i]);
        }
        got.push_back(complete(pp.port, prompt_ids(97, 1900), 12));   /* D39b: the lone long prompt, the cold cells' shape */
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
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const bool notowner = flog.find("NOT the owner's bytes") != std::string::npos;
        long long tb = -1, off = -1, cont = -1;
        const char* key = " lane steps ran as tail band passes (RADIANCE_PP_TAIL_BAND=1, tp8 loop D39)";
        if (const size_t at = llog.rfind(key); at != std::string::npos) {
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                tb = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
            if (const size_t b = llog.find("; ", at); b != std::string::npos) off = std::atoll(llog.c_str() + b + 2);
            if (const size_t b = llog.find(" pp_tband 0; ", at); b != std::string::npos) cont = std::atoll(llog.c_str() + b + 13);
        }
        long long bg_n = -1, bg_newc = -1, bg_b2 = -1;
        if (const size_t at = llog.rfind("two-node pipeline: BG steps over the run (RADIANCE_PP_BG_ROOM="); at != std::string::npos)
            if (const size_t b = llog.find("): ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "): %lld steps with %lld newcomers; %lld steps", &bg_n, &bg_newc, &bg_b2);
        fprintf(stderr, "    TAIL BAND W(2,5) 4 lanes BG 2 + KV copy + check, %-24s: %d of %zu requests differ; KV-copy checks %lld, %lld differ%s; "
                        "%lld tail band passes (%lld with a later chunk), %lld fresh BG steps W(2,5)'s under pp_tband 0; P23: %lld BG steps, "
                        "%lld newcomers, %lld with >= 2\n",
                a.name, differ, ref.size(), checks, kdiff, notowner ? " (NOT the owner's bytes)" : "", tb, cont, off, bg_n, bg_newc, bg_b2);
        if (a.cont_only) {   /* the negative control: later chunks banded over a leader that never wrote layer 4 of their first */
            ctl_differ = differ;
            CHECK(cont > 0);
            continue;
        }
        CHECK_EQ(differ, 0);
        if (!std::strcmp(a.tbv, "2")) {
            CHECK(cont > 0);
            cont_all += std::max(0LL, cont);
        } else {
            CHECK_EQ(cont, 0LL);
        }
        CHECK(!notowner);
        CHECK(checks > 0);
        CHECK(tb > 0);
        if (a.sw) CHECK(off > 0);
        if (a.p23) CHECK(bg_n > 0);
        if (differ || notowner) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        tb_all += std::max(0LL, tb);
    }
    fprintf(stderr, "    tail band passes, all arms: %lld (%lld with a later chunk under 2); the negative control: %d requests differ\n", tb_all,
            cont_all, ctl_differ);
    CHECK(tb_all > 0);
    CHECK(cont_all > 0);
    CHECK(ctl_differ > 0);
}

/* tp8 loop D36 (RADIANCE_PP_LEAD_SEG_BAND=1, the leader): P28's band steps through LSP's head segments -- a band step's head
 * runs [0, 1) then [1, 3) (the last segment through the band layer 2, X1 at 3), the band word on both, the held frame a band
 * frame; every request's tokens are one process's (the six-layer llama) at W(2,5) with 3 lanes BG 1 and the follower's
 * segments, 4 lanes BG 2 with FOL_ORDER 2 and pp_lead_seg form 2 (the BEST form), 3 lanes BG 1 without follower segments,
 * and the switch unset (band heads whole, as P28 built it: 0 heads in segments). */
TEST(band_heads_in_lead_segments_serve_one_process_tokens) {
    REQUIRE(world().ok);
    REQUIRE(world6().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref, ref_b;
    {
        Serving one;
        one.model = world6().llama_rad;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        ref_b = burst_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; int k; bool fseg; bool on; const char* form; bool fol_order = false; };
    long long seg_all = 0;
    /* form "258" = 2 + 256: pp_lead_seg's no-band control (every fresh BG step W(2,5)'s, the band layer declared) */
    for (const Arm a : { Arm{ 3, 1, true, true, "1" }, Arm{ 4, 2, true, true, "2", true }, Arm{ 3, 1, false, true, "1" },
                         Arm{ 3, 1, true, false, "1" }, Arm{ 4, 2, true, true, "258", true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        for (const char* v : { "RADIANCE_LLAMA_PP_BAND=1", "RADIANCE_PP_BAND=1", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_LEAD_SEG=1",
                               "RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_SEG_ROWS=16" })
            env.push_back(v);
        env.push_back(std::string("RADIANCE_PP_LEAD_SEG_FORM=") + a.form);
        if (a.on) env.push_back("RADIANCE_PP_LEAD_SEG_BAND=1");
        if (a.fol_order) env.push_back("RADIANCE_PP_FOL_ORDER=2");
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        pp.model = world6().llama_rad;
        REQUIRE(pp.start("2,5", ex, env));
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
        long long band = -1, segd = -1, off = -1;
        const char* okey = " fresh BG steps ran W(h, t)'s under pp_lead_seg + 256 (no band passes, tp8 loop D36)";
        if (const size_t at = llog.rfind(okey); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                off = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        const char* key = " lane steps ran as band passes (RADIANCE_PP_BAND=1, tp8 loop P28)";
        if (const size_t at = llog.rfind(key); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                band = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        const char* skey = " band steps ran their head in segments (RADIANCE_PP_LEAD_SEG_BAND=1, tp8 loop D36)";
        if (const size_t at = llog.rfind(skey); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                segd = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
        fprintf(stderr, "    LEAD_SEG_BAND %s, form %s, %d lanes BG %d W(2,5)%s%s (6-layer llama): %d of %zu requests differ; %lld band "
                        "passes, %lld of them with the head in segments; %lld fresh BG steps kept W(2,5)'s by + 256\n",
                a.on ? "on" : "unset", a.form, a.lanes, a.k, a.fseg ? " + follower segments" : "", a.fol_order ? " + FOL_ORDER 2" : "",
                differ, ref.size() + ref_b.size(), band, segd, off);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("is not taken") == std::string::npos);
        if (std::atoi(a.form) & 256) {   /* the no-band control: band passes 0, the steps counted */
            CHECK_EQ(band, 0LL);
            CHECK(off > 0);
            continue;
        }
        CHECK(band > 0);
        CHECK_EQ(off, -1LL);
        if (a.on) {
            CHECK(llog.find("RADIANCE_PP_LEAD_SEG_BAND=1 -- a band step's head runs in segments too, the last one [1, 3)") !=
                  std::string::npos);
            CHECK(segd > 0);
            seg_all += std::max(0LL, segd);
        } else {
            CHECK_EQ(segd, -1LL);   /* the switch unset: no line, every band head whole */
        }
    }
    fprintf(stderr, "    band heads in segments, all arms: %lld\n", seg_all);
    CHECK(seg_all > 0);
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
                 const char* join = nullptr; int sw = 0; };   /* sw: tp8 loop P23s -- the room by POST /admin/fn4 pp_bg_room */
    long long b2_all = 0;
    for (const Arm a : { Arm{ 3, "2,3", "1", "2112" }, Arm{ 3, "0,2", "1", "2112" }, Arm{ 4, "1,3", "2", "2112" },
                         Arm{ 3, "2,3", "1", nullptr }, Arm{ 3, "2,3", "1", "2112", true, false },
                         Arm{ 3, "2,3", "1", "2112", false, true }, Arm{ 3, "2,3", "1", "2112", false, false, "1" },
                         Arm{ 3, "0,2", "1", "2112", false, false, "60000" }, Arm{ 3, "2,3", "1", nullptr, false, false, nullptr, 2112 },
                         Arm{ 4, "1,3", "2", nullptr, false, false, nullptr, 2112 } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(std::string("RADIANCE_PP_BG_LANE=") + a.bg);
        if (a.room) env.push_back(std::string("RADIANCE_PP_BG_ROOM=") + a.room);
        if (a.lead_seg) { env.push_back("RADIANCE_PP_LEAD_SEG=1"); env.push_back("RADIANCE_PP_LEAD_SEG_AHEAD=1"); }
        if (a.hold) env.push_back("RADIANCE_PP_BG_HOLD_MS=60000");
        if (a.join) env.push_back(std::string("RADIANCE_PP_BG_JOIN_MS=") + a.join);   /* tp8 loop P23b */
        /* tp8 loop P22b: the follower's BG-pass lines, a batched pass's too */
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_PF_TIMELINE=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        if (a.sw) {   /* tp8 loop P23s: the env's room off, the switch's on before the set */
            Call c;
            std::string out;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_bg_room", a.sw } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
            if (st != 200) fprintf(stderr, "    POST pp_bg_room: %d %s\n", st, out.c_str());
        }
        const std::vector<std::vector<int32_t>> got = bg_room_set(pp.port);
        if (a.sw) {
            Call g;
            std::string out;
            if (g.send(pp.port, "/admin/fn4", "") && g.read(&out) == 200) {
                const json j = json::parse(out, nullptr, false);
                CHECK(j.is_object() && j.contains("switches") && j["switches"].value("pp_bg_room", -9) == a.sw);
            }
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
        if (a.room || a.sw) b2_all += std::max(0LL, b2);
        /* tp8 loop P22b: a batched BG pass is one follower line of its prompts, and the run's summary counts them */
        long long fb = 0, fnewc = -1, fpass = -1, fbatched = -1;
        for (size_t at = flog.find("tp8 loop P22b): prompt "); at != std::string::npos; at = flog.find("tp8 loop P22b): prompt ", at + 1)) {
            const std::string line = flog.substr(at, flog.find('\n', at) - at);
            if (const size_t o = line.find(") of "); o != std::string::npos && std::atoi(line.c_str() + o + 5) >= 2) ++fb;
        }
        if (const size_t at = flog.rfind("tp8 loop P22b) over the run: "); at != std::string::npos)
            (void)std::sscanf(flog.c_str() + at, "tp8 loop P22b) over the run: %lld newcomers in %lld passes (%lld passes of >= 2", &fnewc,
                              &fpass, &fbatched);
        fprintf(stderr, "    follower P22b: %lld batched pass lines; run summary %lld newcomers in %lld passes, %lld of >= 2\n", fb, fnewc,
                fpass, fbatched);
        if (b2 > 0 && std::strcmp(a.cut, "2,3") != 0) {   /* W(2,3)'s follower holds one layer: no segments, no split pass */
            CHECK(fb > 0);
            CHECK(fbatched > 0);
            CHECK(fnewc > fpass);
        }
        fprintf(stderr, "    BG_LANE=%s BG_ROOM=%s%s%s, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to the BG lanes; BG steps "
                        "%lld, %lld with >= 2 newcomers, the widest %lld rows\n", a.bg, a.room ? a.room : a.sw ? "2112 by pp_bg_room" : "off",
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
        } else if (a.sw) {   /* tp8 loop P23s: no env line, the run's BG-steps line at the switch's room, batched steps */
            CHECK(llog.find("RADIANCE_PP_BG_ROOM=2112 -- ") == std::string::npos);
            CHECK(llog.find("BG steps over the run (RADIANCE_PP_BG_ROOM=2112") != std::string::npos);
            CHECK(steps > 0);
            if (a.lanes == 3) CHECK(b2 > 0);   /* one BG lane batches (the env arm too); two BG lanes rarely need to (the env arm: 0) */
            CHECK(widest <= 2112);
        } else {
            CHECK(llog.find("RADIANCE_PP_BG_ROOM=") == std::string::npos);
        }
    }
    fprintf(stderr, "    all BG_ROOM arms: %lld BG steps with >= 2 newcomers\n", b2_all);
    CHECK(b2_all > 0);
}

/* tp8 loop D47: RADIANCE_PP_BG_ROWS=N -- a BG lane's step slab carves its block tables for N rows (not --max-num-seqs) and
 * lane_split_bg lends into it only below N pieces. bg_room_set's five newcomers under RADIANCE_PP_BG_ROOM=2112 (P23: a BG lane's
 * step may batch several) at N = 1 (one BG lane, W(2,3)) / N = 2 (two BG lanes, 4 lanes W(1,3)), without the room at N = 1, and
 * N set without a BG lane (not taken): tokens = one process (the same flags); the leader's slab / budget / start lines name the
 * rows, its P23 line counts no BG step above N newcomers, and its BG line the lends the cap turned away. */
TEST(bg_rows_cap_the_bg_lanes_slabs_and_serve_one_process_tokens) {
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
    struct Arm { int lanes; const char* cut; const char* bg; const char* room; int rows; };
    long long kept_all = 0;
    for (const Arm a : { Arm{ 3, "2,3", "1", "2112", 1 }, Arm{ 4, "1,3", "2", "2112", 2 }, Arm{ 3, "2,3", "1", nullptr, 1 },
                         Arm{ 3, "2,3", nullptr, nullptr, 1 } }) {
        std::vector<std::string> env = kSplitEnv;
        if (a.bg) env.push_back(std::string("RADIANCE_PP_BG_LANE=") + a.bg);
        if (a.room) env.push_back(std::string("RADIANCE_PP_BG_ROOM=") + a.room);
        env.push_back("RADIANCE_PP_BG_ROWS=" + std::to_string(a.rows));
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
        long long bg = -1, steps = -1, b2 = -1, b3 = -1, kept = -1;
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        if (const size_t at = llog.rfind("BG steps over the run (RADIANCE_PP_BG_ROOM="); at != std::string::npos)
            if (const size_t k = llog.find("tp8 loop P23): ", at); k != std::string::npos) {
                const char* q = llog.c_str() + k + std::strlen("tp8 loop P23): ");
                steps = std::atoll(q);
                if (const char* w = std::strstr(q, "newcomers; ")) b2 = std::atoll(w + std::strlen("newcomers; "));
                if (const char* w = std::strstr(q, " rows), ")) b3 = std::atoll(w + std::strlen(" rows), "));
            }
        const std::string kk = "-row slab turned away ";
        if (const size_t at = llog.rfind(kk); at != std::string::npos) kept = std::atoll(llog.c_str() + at + kk.size());
        if (kept > 0) kept_all += kept;
        const std::string rows_s = std::to_string(a.rows);
        const bool slab_l = llog.find("block tables for " + rows_s + " rows, not 10 (RADIANCE_PP_BG_ROWS") != std::string::npos;
        const bool slab_f = flog.find("block tables for " + rows_s + " rows, not 10 (RADIANCE_PP_BG_ROWS") != std::string::npos;
        const bool budget = llog.find("the lanes' step slabs at their block-table rows (RADIANCE_PP_BG_ROWS " + rows_s + " / ") != std::string::npos;
        fprintf(stderr, "    BG_LANE=%s BG_ROOM=%s BG_ROWS=%d, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to the BG lanes; "
                        "BG steps %lld, %lld with >= 2 newcomers, %lld with >= 3; %lld lends turned away; slab line leader %d follower %d, "
                        "budget line %d\n", a.bg ? a.bg : "-", a.room ? a.room : "off", a.rows, a.lanes, a.cut, differ, ref.size(), bg,
                steps, b2, b3, kept, (int)slab_l, (int)slab_f, (int)budget);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK(llog.find("lent pieces in its inbox beside") == std::string::npos);   /* the take-side guard never fired */
        if (!a.bg) {   /* no BG lane: not taken, every lane's slab as before */
            CHECK(llog.find("RADIANCE_PP_BG_ROWS=1 is not taken (tp8 loop D47): it needs RADIANCE_PP_BG_LANE") != std::string::npos);
            CHECK(!slab_l);
            CHECK(!slab_f);
            CHECK(!budget);
            continue;
        }
        CHECK(llog.find("RADIANCE_PP_BG_ROWS=" + rows_s + " (tp8 loop D47) -- the BG lanes' step slabs carve") != std::string::npos);
        CHECK(slab_l);
        CHECK(slab_f);
        CHECK(budget);
        CHECK(bg > 0);
        if (a.room) {
            CHECK(steps > 0);
            if (a.rows == 1) CHECK_EQ(b2, 0);
            if (a.rows == 2) CHECK_EQ(b3, 0);
        }
    }
    fprintf(stderr, "    all BG_ROWS arms: %lld lends turned away by a BG lane's rows\n", kept_all);
    CHECK(kept_all > 0);
}

/* tp8 loop D47b: RADIANCE_PP_LANE_ROWS=M -- a decode lane's step slab carves its block tables for its own cards + M rows and
 * lane_split_own lends a piece into it only below M pieces. lane_set (seven alone, then eight at once: newcomers' chunks cut
 * over the lanes in 16-row pieces) at 3 lanes W(0,2) / W(1,3), M 1 and 2: tokens = one process; the slab lines on both nodes
 * name the rows (2 cards + M at --max-num-seqs 6), the start line names M, the run-end line counts the lends M turned away;
 * + M 1 beside RADIANCE_PP_BG_ROWS=1 at 4 lanes W(1,3) with a BG lane (bg_room_set). */
TEST(lane_rows_cap_the_decode_lanes_slabs_and_serve_one_process_tokens) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "6" }));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    long long kept_all = 0;
    struct Arm { int lanes; int m; const char* cut; bool rr; };
    for (const Arm a : { Arm{ 3, 1, "0,2", true }, Arm{ 3, 1, "1,3", true }, Arm{ 3, 2, "0,2", true }, Arm{ 2, 1, "0,2", false },
                         Arm{ 2, 1, "1,3", false }, Arm{ 2, 2, "1,3", false } }) {
        const int m = a.m;
        const char* cut = a.cut;
        {
            std::vector<std::string> env = a.rr ? kSplitEnv : std::vector<std::string>{ "RADIANCE_PP_SPLIT_MIN=16" };
            env.push_back("RADIANCE_PP_LANE_ROWS=" + std::to_string(m));
            Serving pp;
            REQUIRE(pp.start(cut, lane_shape(a.lanes), env));
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
            const std::string rows_s = std::to_string(6 / a.lanes + m);   /* the lane's cards (6 over the lanes) + M */
            const bool slab_l = llog.find("block tables for " + rows_s + " rows, not 6 (RADIANCE_PP_BG_ROWS / RADIANCE_PP_LANE_ROWS") != std::string::npos;
            const bool slab_f = flog.find("block tables for " + rows_s + " rows, not 6 (RADIANCE_PP_BG_ROWS / RADIANCE_PP_LANE_ROWS") != std::string::npos;
            const long long lent = log_count(llog, "newcomer split: ");
            const long long kept = log_count(llog, "tp8 loop D47b) turned away ");
            if (kept > 0) kept_all += kept;
            fprintf(stderr, "    LANE_ROWS=%d, %d lanes W(%s)%s: %d of %zu prompts differ; pieces lent %lld, %lld lends turned away; slab "
                            "line leader %d follower %d\n", m, a.lanes, cut, a.rr ? " rr" : "", differ, ref.size(), lent, kept, (int)slab_l,
                    (int)slab_f);
            CHECK_EQ(differ, 0);
            if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
            CHECK(llog.find("RADIANCE_PP_LANE_ROWS=" + std::to_string(m) + " (tp8 loop D47b) -- a decode lane's step slab (lanes 1 ..; lane 0, the MERGE target, keeps") != std::string::npos);
            CHECK(llog.find("the lanes' step slabs at their block-table rows (RADIANCE_PP_BG_ROWS 0 / RADIANCE_PP_LANE_ROWS " +
                            std::to_string(m)) != std::string::npos);
            CHECK(slab_l);
            CHECK(slab_f);
            CHECK(lent > 0);
            CHECK(kept >= 0);
            CHECK(llog.find("lent pieces in its inbox beside") == std::string::npos);
        }
    }
    /* lane_split_own put at most one piece in a decode lane's inbox a step in every arm (0 turned away at M = 1): the cap's
     * refusal is D47's lend-time check form (bg_rows_cap... turns lends away), its take-time guard never fired */
    fprintf(stderr, "    all LANE_ROWS arms: %lld lends turned away by a decode lane's rows\n", kept_all);
    CHECK(kept_all >= 0);
    /* + beside RADIANCE_PP_BG_ROWS=1 with a BG lane (bg_room_set's newcomers under BG_ROOM) */
    const std::vector<std::string> ext = { "--max-num-seqs", "10", "--max-num-batched-tokens", "2112", "--max-model-len", "2048",
                                           "--prefill-span", "2", "--checkpoint-interval", "1024" };
    std::vector<std::vector<int32_t>> ref2;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref2 = bg_room_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    std::vector<std::string> env = kSplitEnv;
    for (const char* v : { "RADIANCE_PP_BG_LANE=1", "RADIANCE_PP_BG_ROOM=2112", "RADIANCE_PP_BG_ROWS=1", "RADIANCE_PP_LANE_ROWS=1",
                           "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                           "RADIANCE_PP_READY_FIRST=1" })
        env.push_back(v);
    std::vector<std::string> ex = ext;
    ex.push_back("--pp-lanes");
    ex.push_back("4");
    Serving pp;
    REQUIRE(pp.start("1,3", ex, env));
    const std::vector<std::vector<int32_t>> got = bg_room_set(pp.port);
    kill(pp.lead.child.pid, SIGTERM);
    CHECK_EQ(pp.lead.finish(60), 0);
    CHECK_EQ(pp.fol.finish(30), 0);
    const std::string llog = pp.lead.output();
    REQUIRE_EQ(ref2.size(), got.size());
    int differ = 0;
    for (size_t i = 0; i < ref2.size(); ++i) if (ref2[i] != got[i]) ++differ;
    /* 3 decode lanes over 10 cards: 3 / 3 / 4 cards + 1; the BG lane 1 row */
    const bool s4 = llog.find("block tables for 4 rows, not 10") != std::string::npos, s5 = llog.find("block tables for 5 rows, not 10") != std::string::npos,
               s1 = llog.find("block tables for 1 rows, not 10") != std::string::npos;
    fprintf(stderr, "    BG_ROWS=1 + LANE_ROWS=1, 4 lanes W(1,3) BG 1: %d of %zu requests differ; slab lines 4 rows %d, 5 rows %d, 1 row %d\n", differ,
            ref2.size(), (int)s4, (int)s5, (int)s1);
    CHECK_EQ(differ, 0);
    CHECK(s4);   /* lane 1: 3 cards + 1 (lane 0, the MERGE target, keeps 10) */
    CHECK(s5);   /* lane 2: 4 cards + 1 */
    CHECK(s1);
    /* + the MERGE target: RADIANCE_PP_MERGE=2 (the decode lanes merged into lane 0 always: it holds every card) beside both rows caps;
     * lane 0 keeps --max-num-seqs rows, so the merged lane's steps of all the sequences build */
    std::vector<std::string> envm = env;
    envm.push_back("RADIANCE_PP_MERGE=2");
    Serving pm;
    REQUIRE(pm.start("1,3", ex, envm));
    const std::vector<std::vector<int32_t>> gotm = bg_room_set(pm.port);
    kill(pm.lead.child.pid, SIGTERM);
    CHECK_EQ(pm.lead.finish(60), 0);
    CHECK_EQ(pm.fol.finish(30), 0);
    const std::string mlog = pm.lead.output();
    REQUIRE_EQ(ref2.size(), gotm.size());
    int differm = 0;
    for (size_t i = 0; i < ref2.size(); ++i) if (ref2[i] != gotm[i]) ++differm;
    const bool merged = mlog.find("tp8 MERGE") != std::string::npos;
    fprintf(stderr, "    + RADIANCE_PP_MERGE=2: %d of %zu requests differ; merge lines %d; refused builds %d\n", differm, ref2.size(), (int)merged,
            (int)(mlog.find("exceeds this lane's block tables") != std::string::npos));
    CHECK_EQ(differm, 0);
    CHECK(mlog.find("exceeds this lane's block tables") == std::string::npos);
    CHECK(mlog.find("lent pieces in its inbox beside") == std::string::npos);
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

/* tp8 loop IDEAS #9 (DECODE round 24): RADIANCE_PP_FOL_RESERVE=L (the follower) -- a BG newcomer's body holds its parts from layer L
 * on until another BG lane's frame arrives or RADIANCE_PP_FOL_RESERVE_MS pass; every other frame runs between meanwhile. Tokens =
 * one process (the same flags) at 4 lanes BG 2 at the wrap-around cut W(1,3) (L 2) and at W(0,2) (L 1), with FOL_ORDER 2 / 3 and
 * the BG_ROOM load, and at 3 lanes BG 1 W(1,3) (no other BG lane: every hold ends by the limit, 40 ms); at 2 lanes (no BG lane)
 * and at a boundary that is not declared (W(1,3), L 3 = t) the reserve is refused by name and the pair serves as without it. */
TEST(a_bg_bodys_reserve_serves_one_process_tokens) {
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
    struct Arm { bool room; int lanes; const char* cut; const char* bg; const char* order; const char* res; const char* ms; int taken;
                 const char* post = nullptr; };
    long long n_all = 0, frame_all = 0, time_all = 0;
    for (const Arm a : { Arm{ false, 4, "1,3", "2", "2", "2", "300", 1 }, Arm{ false, 4, "0,2", "2", "3", "1", "300", 1 },
                         Arm{ true, 4, "1,3", "2", "2", "2", "300", 1 }, Arm{ false, 3, "1,3", "1", "2", "2", "40", 1 },
                         Arm{ false, 2, "1,3", nullptr, "2", "2", "300", 0 }, Arm{ false, 4, "1,3", "2", "2", "3", "300", 0 },
                         Arm{ false, 4, "1,3", "2", "2", nullptr, "300", 2, "2" } }) {
        std::vector<std::string> env = kSplitEnv;
        if (a.bg) env.push_back(std::string("RADIANCE_PP_BG_LANE=") + a.bg);
        env.push_back(std::string("RADIANCE_PP_FOL_ORDER=") + a.order);
        if (a.res) env.push_back(std::string("RADIANCE_PP_FOL_RESERVE=") + a.res);
        env.push_back(std::string("RADIANCE_PP_FOL_RESERVE_MS=") + a.ms);
        if (a.room) env.push_back("RADIANCE_PP_BG_ROOM=2112");
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = a.room ? ext_r : ext_b;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        if (a.post) {   /* fn4 pp_fol_reserve: an undeclared boundary is the follower's refusal, then the arm's own value */
            std::string out;
            Call c0;
            int st = c0.send(pp.port, "/admin/fn4", json{ { "pp_fol_reserve", 3 } }.dump()) ? c0.read(&out) : -1;
            CHECK_EQ(st, 400);
            const bool named = out.find("is not a declared segment boundary") != std::string::npos;
            if (!named) fprintf(stderr, "    POST pp_fol_reserve 3: %d %s\n", st, out.c_str());
            CHECK(named);
            Call c1;
            st = c1.send(pp.port, "/admin/fn4", json{ { "pp_fol_reserve", std::atoi(a.post) } }.dump()) ? c1.read(&out) : -1;
            CHECK_EQ(st, 200);
            if (st != 200) fprintf(stderr, "    POST pp_fol_reserve %s: %d %s\n", a.post, st, out.c_str());
        }
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
        long long n = -1, by_frame = -1, by_time = -1;
        if (const size_t at = flog.rfind("tp8 loop IDEAS #9): "); at != std::string::npos)
            (void)std::sscanf(flog.c_str() + at, "tp8 loop IDEAS #9): %lld BG bodies held their layers L..%*d, %lld released by "
                                                 "another BG lane's frame, %lld by the", &n, &by_frame, &by_time);
        fprintf(stderr, "    RESERVE=%s%s (%s ms), FOL_ORDER=%s, BG lanes %s%s, %d lanes W(%s): %d of %zu requests differ; %lld bodies "
                        "held, %lld released by another BG frame, %lld by the limit\n", a.res ? a.res : "unset",
                a.post ? (std::string(" + POST pp_fol_reserve ") + a.post).c_str() : "", a.ms, a.order, a.bg ? a.bg : "none",
                a.room ? " + BG_ROOM=2112" : "", a.lanes, a.cut, differ, ref.size(), n, by_frame, by_time);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        if (a.taken) {
            if (a.taken == 1)
                CHECK(flog.find(std::string("follower: RADIANCE_PP_FOL_RESERVE=") + a.res + " -- a BG newcomer's body holds") !=
                      std::string::npos);
            CHECK(n > 0);
            CHECK_EQ(n, by_frame + by_time);
            if (a.bg && a.bg[0] == '1') CHECK_EQ(by_frame, 0);   /* one BG lane: no other BG lane's frame */
            n_all += std::max(0LL, n);
            frame_all += std::max(0LL, by_frame);
            time_all += std::max(0LL, by_time);
        } else {
            CHECK(flog.find(std::string("follower: RADIANCE_PP_FOL_RESERVE=") + a.res + " is not taken here: ") != std::string::npos);
            CHECK_EQ(n, -1);   /* no reserve line */
        }
    }
    fprintf(stderr, "    all arms: %lld bodies held, %lld released by another BG lane's frame, %lld by the limit\n", n_all,
            frame_all, time_all);
    CHECK(n_all > 0);
    CHECK(time_all > 0);
}

/* tp8 loop F71 (RADIANCE_PP_PLAN_RESERVED=1 on the leader, DECODE round 26): a lane's inbox rows count where a plan takes the step's
 * max_tok room (option D's over-budget first prefill, lever NOA's room) -- planning only, so every request's tokens are one process's
 * at 4 lanes BG 2 (FOL_ORDER 2, the BEST form) and 3 lanes BG 1 with lent pieces in the steps, and nothing names a step over its
 * budget (sched_test lanes_an_inbox_piece_counts_against_an_over_budget_first_prefill reproduces the overrun without the env). */
TEST(plan_reserved_serves_one_process_tokens) {
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
    struct Arm { int lanes; int bg; const char* cut; };
    long long lent_all = 0;
    for (const Arm a : { Arm{ 4, 2, "1,3" }, Arm{ 3, 1, "0,2" } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(std::string("RADIANCE_PP_BG_LANE=") + std::to_string(a.bg));
        for (const char* v : { "RADIANCE_PP_PLAN_RESERVED=1", "RADIANCE_PP_FOL_ORDER=2", "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1",
                               "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        REQUIRE_EQ(ref_b.size(), got_b.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) differ += ref[i] != got[i];
        for (size_t i = 0; i < ref_b.size(); ++i) differ += ref_b[i] != got_b[i];
        auto occ = [](const std::string& log, const std::string& tag) {
            long long n = 0;
            for (size_t at = log.find(tag); at != std::string::npos; at = log.find(tag, at + 1)) ++n;
            return n;
        };
        const long long lent = occ(llog, " lent to other lanes") - occ(llog, " 0 lent to other lanes");
        const long long named = occ(llog, "rows of other lanes' pieces");
        fprintf(stderr, "    RADIANCE_PP_PLAN_RESERVED=1, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld prompts with lent chunks; "
                        "%lld 'rows of other lanes' pieces' lines\n", a.bg, a.lanes, a.cut, differ, ref.size() + ref_b.size(), lent, named);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(named, 0LL);
        lent_all += lent;
        /* tp8 loop D35: the leader's decode-chain host line at the run's end */
        long long dch = -1;
        double dch_p50 = -1, dch_mean = -1;
        const char* dkey = "decode chain on the leader host over the run (tp8 loop D35): ";
        if (const size_t at = llog.rfind(dkey); at != std::string::npos)
            (void)std::sscanf(llog.c_str() + at + std::strlen(dkey), "%lld decode steps of the lanes: their collect -> the lane's next decode "
                                                                    "step shipped p50 %lf / mean %lf", &dch, &dch_p50, &dch_mean);
        fprintf(stderr, "      D35: %lld decode lane steps, collect -> next decode step shipped p50 %.2f / mean %.2f ms\n", dch, dch_p50, dch_mean);
        CHECK(dch > 0);
        CHECK(dch_p50 >= 0 && dch_mean >= 0);
    }
    fprintf(stderr, "    all arms: %lld prompts with lent chunks\n", lent_all);
    CHECK(lent_all > 0);
}

/* tp8 loop D33 (DECODE round 26): the follower's segment set -- RADIANCE_PP_SEG_MAX=N declares at most N - 1 boundaries (the
 * bound before D33, 16, left the follower's last segment [25,32) whole at W(10,32): the end lump), fn4 pp_seg_set picks which
 * declared boundaries split a pass (n + 64 x s: every s-th, then the first n; 0 = all). At W(0,3) SEG_LAYERS=1 the follower
 * declares 1, 2: SEG_MAX=2 declares 1 only ([1,3) whole: the lump's stand-in); in process 128 = {2}, 1 = {1}, 0 = {1, 2}, POSTed
 * between request sets, and RADIANCE_PP_SEG_SET its start value. Every request's tokens are one process's at 4 lanes BG 2
 * (FOL_ORDER 2, the BEST form) and 3 lanes BG 1, and with a body reserve at 2 beside the set {1} (the reserve's boundary splits
 * whatever the set: bodies are held); a value out of range and a follower with no boundary are refused by name. */
TEST(follower_segment_set_serves_one_process_tokens) {
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
    struct Arm { int lanes; int bg; bool seg; const char* seg_max; const char* seg_set; const char* res; const char* post1; const char* post2;
                 const char* bounds; };
    long long split_all = 0, held_all = 0;
    for (const Arm a : { Arm{ 4, 2, true, nullptr, nullptr, nullptr, "128", "1", "1, 2" },
                         Arm{ 4, 2, true, "2", nullptr, nullptr, nullptr, nullptr, "1" },
                         Arm{ 3, 1, true, "32", "128", nullptr, nullptr, "0", "1, 2" },
                         Arm{ 4, 2, true, nullptr, "1", "2", nullptr, nullptr, "1, 2" },
                         Arm{ 4, 2, false, nullptr, nullptr, nullptr, "1", nullptr, nullptr } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(std::string("RADIANCE_PP_BG_LANE=") + std::to_string(a.bg));
        env.push_back("RADIANCE_PP_FOL_ORDER=2");
        if (a.seg_max) env.push_back(std::string("RADIANCE_PP_SEG_MAX=") + a.seg_max);
        if (a.seg_set) env.push_back(std::string("RADIANCE_PP_SEG_SET=") + a.seg_set);
        if (a.res) env.push_back(std::string("RADIANCE_PP_FOL_RESERVE=") + a.res);
        if (a.seg) env.push_back("RADIANCE_PP_SEG_LAYERS=1");
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15",
                               "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start("0,3", ex, env));
        auto post = [&](int v, int want, const char* named) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_seg_set", v } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, want);
            if (st != want) fprintf(stderr, "    POST pp_seg_set %d: %d %s\n", v, st, out.c_str());
            if (named) {
                const bool has = out.find(named) != std::string::npos;
                if (!has) fprintf(stderr, "    POST pp_seg_set %d: %s\n", v, out.c_str());
                CHECK(has);
            }
        };
        if (a.seg) post(32, 400, "pp_seg_set takes n + 64 x s");
        if (a.post1 && !a.seg) post(std::atoi(a.post1), 400, "declared no segment boundary");
        if (a.post1 && a.seg) post(std::atoi(a.post1), 200, nullptr);
        const std::vector<std::vector<int32_t>> got = split_set(pp.port);
        if (a.post2) post(std::atoi(a.post2), 200, nullptr);
        const std::vector<std::vector<int32_t>> got_b = burst_set(pp.port);
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
        long long passes = -1, segs = -1, held = -1;
        if (const size_t at = flog.rfind(" passes ran in "); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "follower: %lld passes ran in %lld segments", &passes, &segs);
        if (const size_t at = flog.rfind("tp8 loop IDEAS #9): "); at != std::string::npos)
            (void)std::sscanf(flog.c_str() + at, "tp8 loop IDEAS #9): %lld BG bodies held", &held);
        std::string sets;
        for (size_t at = flog.find("follower: pp_seg_set "); at != std::string::npos; at = flog.find("follower: pp_seg_set ", at + 1)) {
            const size_t e = flog.find('\n', at);
            const std::string ln = flog.substr(at, e == std::string::npos ? std::string::npos : e - at);
            if (const size_t p = ln.find("now ends its segments at "); p != std::string::npos)
                sets += (sets.empty() ? "" : " | ") + ln.substr(10, ln.find(' ', 21) - 10) + ": " + ln.substr(p + 25);
        }
        fprintf(stderr, "    SEG_LAYERS=%s SEG_MAX=%s SEG_SET=%s RESERVE=%s%s%s, BG lanes %d, %d lanes W(0,3): %d of %zu requests differ; "
                        "follower: %lld passes in %lld segments, %lld bodies held; sets [%s]\n", a.seg ? "1" : "unset",
                a.seg_max ? a.seg_max : "unset", a.seg_set ? a.seg_set : "unset", a.res ? a.res : "unset",
                a.post1 ? (std::string(" + POST ") + a.post1).c_str() : "", a.post2 ? (std::string(" + POST ") + a.post2).c_str() : "",
                a.bg, a.lanes, differ, ref.size() + ref_b.size(), passes, segs, held, sets.c_str());
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        if (!a.seg) {
            CHECK(flog.find("tp8 loop D33") == std::string::npos);   /* nothing of D33 without boundaries */
            continue;
        }
        CHECK(flog.find(std::string("in layer segments at the boundaries ") + a.bounds + " of W(0,3)") != std::string::npos);
        CHECK(passes > 0);
        split_all += std::max(0LL, passes);
        if (a.seg_max || a.seg_set)
            CHECK(flog.find("follower: segment set (tp8 loop D33): ") != std::string::npos);
        else
            CHECK(flog.find("follower: segment set (tp8 loop D33): ") == std::string::npos);
        if (a.post1) CHECK(flog.find(std::string("follower: pp_seg_set ") + a.post1 + " (tp8 loop D33)") != std::string::npos);
        if (a.post2) CHECK(flog.find(std::string("follower: pp_seg_set ") + a.post2 + " (tp8 loop D33)") != std::string::npos);
        if (a.post1 && std::strcmp(a.post1, "128") == 0)
            CHECK(flog.find("follower: pp_seg_set 128 (tp8 loop D33): a split pass of W(0,3) now ends its segments at 2 (2 segments)") !=
                  std::string::npos);
        if (a.post2 && std::strcmp(a.post2, "1") == 0)
            CHECK(flog.find("follower: pp_seg_set 1 (tp8 loop D33): a split pass of W(0,3) now ends its segments at 1 (2 segments)") !=
                  std::string::npos);
        if (a.seg_set && std::strcmp(a.seg_set, "128") == 0)
            CHECK(flog.find("follower: pp_seg_set 128 (tp8 loop D33): a split pass of W(0,3) now ends its segments at 2 (2 segments)") !=
                  std::string::npos);
        if (a.res) {
            CHECK(held > 0);   /* the reserve's boundary 2 split the bodies under the set {1} */
            held_all += std::max(0LL, held);
        }
    }
    fprintf(stderr, "    all arms: %lld split passes, %lld bodies held under pp_seg_set 1 + reserve 2\n", split_all, held_all);
    CHECK(split_all > 0);
    CHECK(held_all > 0);
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

/* tp8 loop F73 (lq189: a card could not back the KV pool made again right after freeing it -> startup stopped): the remake is
 * retried for ~2 s, then the pool is kept at its old size (a WARN; that card's headroom stays short by the unseen part) instead of
 * stopping the start. RADIANCE_KV_PLACED_TEST_FAIL=N refuses the first N tries: 3 -> made after the retries (given back as
 * before), 400 -> kept on every rank of both nodes; the pair serves one process's greedy tokens at 2 lanes W(1,3) either way. */
TEST(kv_placed_remake_retries_then_keeps_the_pool) {
    REQUIRE(world().ok);
    const std::vector<std::string> big = { "--max-model-len", "262144" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", cat({ "--max-num-seqs", "6" }, big)));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    for (const char* fail : { "3", "400" }) {
        const std::vector<std::string> kvp = { "RADIANCE_KV_PLACED=1", "RADIANCE_KV_PLACED_TEST_MIB=48",
                                               std::string("RADIANCE_KV_PLACED_TEST_FAIL=") + fail };
        Serving pp;
        REQUIRE(pp.start("1,3", cat(lane_shape(2), big), cat(kSplitEnv, kvp)));
        const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != got[i]) ++differ;
        int kept = 0, gave = 0;
        for (const std::string* log : { &llog, &flog }) {
            if (log->find("-- kept at ") != std::string::npos && log->find("(tp8 loop F73)") != std::string::npos) ++kept;
            if (log->find("RADIANCE_KV_PLACED=1: the KV pool gave back") != std::string::npos) ++gave;
        }
        fprintf(stderr, "    TEST_FAIL=%s, 2 lanes W(1,3), RADIANCE_KV_PLACED=1: %d of %zu prompts differ; kept on %d node(s), gave back on %d\n",
                fail, differ, ref.size(), kept, gave);
        CHECK_EQ(differ, 0);
        if (std::strcmp(fail, "3") == 0) {
            CHECK_EQ(gave, 2);
        } else {
            CHECK_EQ(kept, 2);
        }
        if (differ || (std::strcmp(fail, "3") == 0 ? gave != 2 : kept != 2))
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 40).c_str(), tail(flog, 40).c_str());
    }
}

/* tp8 loop F66 (lever KVP): RADIANCE_KV_PLACED=1 on both nodes with RADIANCE_KV_PLACED_TEST_MIB=48 (48 MiB a card
 * "lost beside the budget", the host backend's stand-in for the page tables a card's mapped host memory costs):
 * each KV pool gives it back before the count, the node that pages more returns its pages past the agreed count,
 * and the pair still serves one process's greedy tokens at 1 / 2 / 3 lanes and records one process's KL at a
 * wrap-around cut. */
TEST(kv_placed_pools_serve_and_record_what_one_process_does) {
    REQUIRE(world().ok);
    const std::vector<std::string> kvp = { "RADIANCE_KV_PLACED=1", "RADIANCE_KV_PLACED_TEST_MIB=48" };
    /* a context long enough that the cache takes the whole card (6 x 262144 tokens at 1 KiB a token > 512 MiB):
     * nothing beside the pools absorbs the 48 MiB, as on the cards */
    const std::vector<std::string> big = { "--max-model-len", "262144" };
    const auto check_logs = [](const std::string& llog, const std::string& flog, const char* what, bool lead_back) {
        int gave = 0, back = 0;
        for (const std::string* log : { &llog, &flog }) {
            CHECK(log->find("VRAM placed since the budget read the cards that the budget did not count") != std::string::npos);
            if (log->find("RADIANCE_KV_PLACED=1: the KV pool gave back") != std::string::npos) ++gave;
            if (log->find("paged more than the agreed") != std::string::npos) ++back;
        }
        CHECK_EQ(gave, 2);          /* both nodes gave the 48 MiB a card back before the count */
        /* the leader on the larger card returned its pages past the follower's count (equal cards: nothing past it) */
        if (lead_back) CHECK(llog.find("paged more than the agreed") != std::string::npos);
        else CHECK_EQ(back, 0);
        const std::string tag = "KV capped on both nodes before the carve at ";
        const size_t at_l = llog.find(tag), at_f = flog.find(tag);
        CHECK(at_l != std::string::npos && at_f != std::string::npos);
        if (at_l != std::string::npos && at_f != std::string::npos)
            CHECK_EQ(std::atoll(llog.c_str() + at_l + tag.size()), std::atoll(flog.c_str() + at_f + tag.size()));
        fprintf(stderr, "    %s: gave back on %d node(s), pages past the count returned on %d\n", what, gave, back);
        if (gave != 2 || (lead_back && back < 1))
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 40).c_str(), tail(flog, 40).c_str());
    };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", cat({ "--max-num-seqs", "6" }, big)));
        ref = lane_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    for (int lanes : { 1, 2, 3 }) {
        Serving pp;
        REQUIRE(pp.start("1,3", cat(lane_shape(lanes), big), cat(kSplitEnv, kvp)));
        const std::vector<std::vector<int32_t>> got = lane_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != got[i]) ++differ;
        fprintf(stderr, "    %d lane(s) W(1,3), RADIANCE_KV_PLACED=1: %d of %zu prompts differ\n", lanes, differ, ref.size());
        CHECK_EQ(differ, 0);
        check_logs(pp.lead.output(), pp.fol.output(), lanes == 1 ? "1 lane" : lanes == 2 ? "2 lanes" : "3 lanes", false);
    }
    tiny::TempDir d("pp_kvp_rec");
    const std::string rec = record_single(d / "single", cat({ "--max-num-seqs", "6" }, big));
    REQUIRE(!rec.empty());
    for (const char* cut : { "1,3", "0,2" }) {
        const std::string out = d / (std::string("pp-") + cut);
        const std::vector<std::string> env = cat(kSplitEnv, kvp);
        /* the follower on a smaller card binds; RADIANCE_KV_PLACED is each node's own (rendezvous) */
        const PipeRun pr = record_pipeline(out, cut, cat(env, { "RADIANCE_HOST_VRAM_MIB=384" }), cat(lane_shape(2), big),
                                           cat(lane_shape(2), big), env);
        CHECK_EQ(pr.leader, 0);
        CHECK_EQ(pr.follower, 0);
        std::string why;
        const bool same = pr.leader == 0 && same_recording(rec, out, &why);
        fprintf(stderr, "    2 lanes W(%s), RADIANCE_KV_PLACED=1: recording %s\n", cut, same ? "identical" : why.c_str());
        CHECK(same);
        if (!same) fprintf(stderr, "    leader:\n%s\n", tail(pr.llog, 30).c_str());
        check_logs(pr.llog, pr.flog, cut, true);
    }
}

/* tp8 loop D38a: the DSpark drafter on a pipeline's leader (its tap, layer 3, in the leader's tail at W(0,2) / W(1,3) /
 * W(0,3)), at one lane, two and three: the drafted pair serves the drafted one process's tokens (= the plain greedy
 * tokens) and drafts, some accepted -- the host arm D38 (issue-ahead of drafted lanes) is proved against. A tap given to
 * the follower is refused by name. */
TEST(d38a_a_drafted_pipeline_serves_one_process_tokens_at_one_two_and_three_lanes) {
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    std::vector<Drafted> ref;
    {
        Serving one;
        one.model = worldds().llama_rad;
        REQUIRE(one.start("", cat(spec, { "--max-num-seqs", "6" })));
        ref = drafted_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    for (const auto& [lanes, cut] : { std::pair<int, const char*>{ 1, "0,2" }, { 2, "1,3" }, { 3, "0,3" }, { 2, "0,2" } }) {
        Serving pp;
        pp.model = worldds().llama_rad;
        REQUIRE(pp.start(cut, cat(spec, lane_shape(lanes)), kSplitEnv));
        const std::vector<Drafted> got = drafted_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        REQUIRE_EQ(ref.size(), got.size());
        long long drafted = 0, accepted = 0;
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!got[i].tok.empty());
            if (ref[i].tok != got[i].tok) ++differ;
            drafted += got[i].drafted;
            accepted += got[i].accepted;
        }
        fprintf(stderr, "    %d lane(s) W(%s): %d of %zu prompts differ; %lld drafted, %lld accepted\n", lanes, cut, differ,
                ref.size(), drafted, accepted);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(pp.lead.output()).c_str(),
                            tail(pp.fol.output()).c_str());
        CHECK_EQ(differ, 0);
        CHECK(drafted > accepted);
        CHECK(accepted > 0);
    }
    /* W(2,4): the tap (layer 3) is the follower's -- refused by name, the pair does not start */
    {
        Serving pp;
        pp.model = worldds().llama_rad;
        const bool up = pp.start("2,4", cat(spec, lane_shape(2)), kSplitEnv);
        CHECK(!up);
        const std::string f = pp.fol.output() + pp.lead.output();
        const bool named = f.find("every tap must be a leader layer") != std::string::npos;
        if (!named) fprintf(stderr, "    W(2,4):\n%s\n", tail(f).c_str());
        CHECK(named);
    }
}
/* ------------------------------------------------------------------ tp8 MERGE */
/* Four long decoders, then newcomers in two waves with a pause between: each wave's BG steps merge the decode lanes into lane 0
 * while they are in flight, the pause splits them back, and the newcomers decode on in whichever lane form comes next. */
std::vector<std::vector<int32_t>> merge_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 41, 42, 43, 44 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 260); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    for (uint64_t s : { 71, 72 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 600 + 40 * (s - 71)), 40); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    for (uint64_t s : { 81, 82 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 640 + 40 * (s - 81)), 40); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

/* fn4 pp_merge (tp8 MERGE; core/sched/lanes.h merge_into / split_from): bit 1 runs the decode lanes as ONE lane (lane 0) while a BG
 * lane has a step in flight and splits them back when none; bit 2 keeps them merged. Every request's tokens are one process's --
 * decoders moved into lane 0 and back in mid-generation, newcomers admitted while merged and split later -- at 4 lanes BG 2 W(1,3) /
 * W(0,2), 3 lanes BG 1, 4 lanes BG 1 (three decode lanes into one), always merged, and with the switch set by POST /admin/fn4 on a
 * pair started without it; the leader's merge line counts the merges, splits, merged steps and moved requests. */
TEST(merged_decode_lanes_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "1040", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = merge_set(one.port);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    /* lead_seg: RADIANCE_PP_LEAD_SEG=1 declared (W(2,3): a head boundary at 1, tail whole) -- pp_merge 5 = the merge with the BG steps'
     * leader parts in segments, 9 = the merge with them whole (bit 8), 4 = no merge and whole (BEST's form beside the declared ring) */
    /* form >= 0 (ds41-pp-merge-dev, D32 merged in): RADIANCE_PP_LEAD_SEG_FORM = pp_lead_seg's start -- 2 = decode first (the merged lane's
     * parts go between the BG step's segments); rings 2 = RADIANCE_PP_LEAD_SEG_RINGS=2 (a ring a BG lane); pp_merge 0 = D32 alone */
    struct Arm { int lanes; const char* cut; int k; int merge; bool post = false; bool lead_seg = false; int form = -1; int rings = 1; };
    for (const Arm a : { Arm{ 4, "1,3", 2, 1 }, Arm{ 4, "0,2", 2, 1 }, Arm{ 3, "1,3", 1, 1 }, Arm{ 4, "1,3", 1, 1 },
                         Arm{ 4, "1,3", 2, 2 }, Arm{ 4, "1,3", 2, 1, true }, Arm{ 4, "2,3", 2, 5, false, true },
                         Arm{ 4, "2,3", 2, 9, false, true }, Arm{ 4, "2,3", 2, 4, false, true }, Arm{ 4, "1,3", 2, 65 },
                         Arm{ 4, "1,3", 2, 64 }, Arm{ 3, "0,2", 1, 65 }, Arm{ 4, "2,3", 2, 65, false, true, 2, 2 },
                         Arm{ 4, "2,3", 2, 69, false, true, 2, 1 }, Arm{ 4, "2,3", 2, 0, false, true, 2, 2 } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        if (!a.post) env.push_back("RADIANCE_PP_MERGE=" + std::to_string(a.merge));
        if (a.lead_seg) {
            env.push_back("RADIANCE_PP_LEAD_SEG=1");
            env.push_back("RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15");
            if (a.form >= 0) env.push_back("RADIANCE_PP_LEAD_SEG_FORM=" + std::to_string(a.form));
            if (a.rings == 2) env.push_back("RADIANCE_PP_LEAD_SEG_RINGS=2");
        }
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        if (a.post) {
            Call c;
            std::string out;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_merge", a.merge } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
            if (st != 200) fprintf(stderr, "    POST pp_merge: %d %s\n", st, out.c_str());
        }
        const std::vector<std::vector<int32_t>> got = merge_set(pp.port);
        if (a.post) {
            Call g;
            std::string out;
            if (g.send(pp.port, "/admin/fn4", "") && g.read(&out) == 200) {
                const json j = json::parse(out, nullptr, false);
                CHECK(j.is_object() && j.contains("switches") && j["switches"].value("pp_merge", -9) == a.merge);
            }
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
        long long merges = -1, drains = -1, splits = -1, steps_m = -1, tok_m = -1, rows_m = -1, steps_s = -1, moved_in = -1,
                  moved_out = -1, bg = -1;
        double drain_ms = 0, merged_s = 0, wall_s = 0;
        int pm = -1;
        if (const size_t at = llog.rfind("two-node pipeline: merge over the run (fn4 pp_merge="); at != std::string::npos) {
            (void)std::sscanf(llog.c_str() + at, "two-node pipeline: merge over the run (fn4 pp_merge=%d, tp8 MERGE): %lld merges (%lld "
                                                 "drains, %lf ms), %lld splits, merged %lf s of %lf s; decode lane steps merged %lld (%lld "
                                                 "tokens, %lld rows) / split %lld", &pm, &merges, &drains, &drain_ms, &splits, &merged_s,
                              &wall_s, &steps_m, &tok_m, &rows_m, &steps_s);
            if (const size_t mi = llog.find("requests moved in ", at); mi != std::string::npos)
                (void)std::sscanf(llog.c_str() + mi, "requests moved in %lld / out %lld", &moved_in, &moved_out);
        }
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            bg = std::atoll(llog.c_str() + at + std::strlen("(RADIANCE_PP_BG_LANE, tp8 dev M8): "));
        long long lsteps = -1, lparts = -1;
        if (const size_t at = llog.rfind(" BG steps ran their leader parts in "); at != std::string::npos)
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                (void)std::sscanf(llog.c_str() + b, "two-node pipeline: %lld BG steps ran their leader parts in %lld segments", &lsteps, &lparts);
        long long tfirst = -1;
        if (const size_t at = llog.rfind(" tails taken first while a BG "); at != std::string::npos)
            if (const size_t b = llog.rfind("pp_lead_seg ", at); b != std::string::npos) {
                int v = -1;
                (void)std::sscanf(llog.c_str() + b, "pp_lead_seg %d now (tp8 loop D32): %lld", &v, &tfirst);
            }
        if (a.lead_seg) fprintf(stderr, "    + LEAD_SEG=1%s%s: %lld BG steps' leader parts in %lld segments, %lld tails taken first\n",
                                a.form == 2 ? " form 2" : "", a.rings == 2 ? " RINGS=2" : "", lsteps, lparts, tfirst);
        fprintf(stderr, "    pp_merge %d%s, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld newcomers to BG lanes; %lld merges "
                        "(%lld drains, %.1f ms), %lld splits, merged %.2f of %.2f s; lane steps merged %lld (%lld tokens, %lld rows) / split "
                        "%lld; moved in %lld / out %lld\n", a.merge, a.post ? " by POST" : "", a.k, a.lanes, a.cut, differ, ref.size(), bg,
                merges, drains, drain_ms, splits, merged_s, wall_s, steps_m, tok_m, rows_m, steps_s, moved_in, moved_out);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 40).c_str(), tail(flog, 30).c_str());
        if (a.merge != 0) CHECK_EQ(pm, a.merge);
        CHECK(bg > 0);
        if (a.lead_seg) {   /* 5: every BG step in segments (the merge is on); 9 / 4: none */
            CHECK(llog.find("RADIANCE_PP_LEAD_SEG=1 -- ") != std::string::npos);
            if (a.form >= 0) { CHECK(lsteps > 0); CHECK(lparts > lsteps); }   /* form 2: segments while a decode lane is busy */
            else if (a.merge == 5) { CHECK(lsteps > 0); CHECK(lparts > lsteps); }
            else CHECK(lsteps <= 0);
        }
        if (!(a.merge & 3)) {   /* 0: D32's form 2 alone (the control); no merge bit (4: segments with the merge on only; 64: decode tails first alone): the lanes never merge */
            CHECK(merges <= 0);
            continue;
        }
        CHECK(merges > 0);
        CHECK(steps_m > 0);
        if (a.merge & 1) {
            CHECK(moved_in > 0);   /* merged always (2): merged before the first request, every request is lane 0's by pick() */
            CHECK(splits > 0);
            CHECK(moved_out > 0);
            CHECK(steps_s > 0);
        } else {
            CHECK_EQ(merges, 1LL);
            CHECK_EQ(splits, 0LL);
        }
    }
}

/* fn4 pp_merge bit 16 + pp_merge_ms (tp8 MERGE form 2): while merged, a newcomer the BG lanes have no room for is HELD out of the
 * merged lane's steps (D11's hold, RADIANCE_PP_MERGE_HOLD_MS) instead of turning the one decode lane into a MIXED step, and the lanes
 * stay merged pp_merge_ms after the last BG work. burst_set's four newcomers arrive together beside four decoders (700 rows a step: a
 * BG inbox holds one): at 3 lanes BG 1 the BG lane takes one at a time, the rest are held; every request's tokens are one process's. */
/* four decoders, then six ~600-row newcomers at once: a BG inbox holds one (700 rows a step), so beside a busy BG lane the rest have
 * no room -- held under bit 16, MIXED steps of the merged lane without it */
std::vector<std::vector<int32_t>> hold_burst_set(int port) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t s : { 41, 42, 43, 44 })
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 12), 160); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    for (uint64_t s : { 91, 92, 93, 94, 95, 96 }) {
        f.push_back(std::async(std::launch::async, [port, s] { return complete(port, prompt_ids(s, 560 + 20 * (s - 91)), 10); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}

TEST(merge_holds_and_lingers_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        for (auto& v : hold_burst_set(one.port)) ref.push_back(v);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; int merge; int ms; };
    long long held_all = 0;
    for (const Arm a : { Arm{ 3, "1,3", 1, 17, 0 }, Arm{ 3, "1,3", 1, 1, 300 }, Arm{ 4, "1,3", 2, 17, 300 }, Arm{ 3, "0,2", 1, 17, 0 } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_PP_MERGE=" + std::to_string(a.merge));
        if (a.ms) env.push_back("RADIANCE_PP_MERGE_MS=" + std::to_string(a.ms));
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=40", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        std::vector<std::vector<int32_t>> got = split_set(pp.port);
        for (auto& v : hold_burst_set(pp.port)) got.push_back(v);
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
        long long merges = -1, splits = -1, held = -1, hsteps = -1, hlent = -1, hhome = -1, hold_rounds = -1, hyst_rounds = -1;
        int pm = -1, ms = -1;
        if (const size_t at = llog.rfind("two-node pipeline: merge over the run (fn4 pp_merge="); at != std::string::npos) {
            double d0 = 0, d1 = 0, d2 = 0;
            long long drains = 0;
            (void)std::sscanf(llog.c_str() + at, "two-node pipeline: merge over the run (fn4 pp_merge=%d, tp8 MERGE): %lld merges (%lld "
                                                 "drains, %lf ms), %lld splits, merged %lf s of %lf s", &pm, &merges, &drains, &d0, &splits,
                              &d1, &d2);
            if (const size_t h = llog.find("held for the BG lanes ", at); h != std::string::npos)
                (void)std::sscanf(llog.c_str() + h, "held for the BG lanes %lld newcomers over %lld steps (%lld lent later, %lld home after "
                                                    "the hold), rounds merged by a hold %lld / by pp_merge_ms=%d %lld", &held, &hsteps,
                                  &hlent, &hhome, &hold_rounds, &ms, &hyst_rounds);
        }
        if (const size_t at = llog.rfind("(RADIANCE_PP_BG_LANE, tp8 dev M8): "); at != std::string::npos)
            fprintf(stderr, "    %s\n", llog.substr(at, std::min<size_t>(300, llog.find('\n', at) - at)).c_str());
        fprintf(stderr, "    pp_merge %d, pp_merge_ms %d, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld merges, %lld splits; "
                        "held %lld newcomers over %lld steps (%lld lent, %lld home), rounds merged by a hold %lld / by pp_merge_ms %lld\n",
                a.merge, a.ms, a.k, a.lanes, a.cut, differ, ref.size(), merges, splits, held, hsteps, hlent, hhome, hold_rounds,
                hyst_rounds);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 40).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(pm, a.merge);
        CHECK_EQ(ms, a.ms);
        CHECK(merges > 0);
        if (a.merge & 16) {
            held_all += std::max(0LL, held);
            CHECK_EQ(std::max(0LL, held), std::max(0LL, hlent) + std::max(0LL, hhome));
        } else {
            CHECK(held <= 0);
        }
        if (a.ms) CHECK(hyst_rounds > 0);
    }
    fprintf(stderr, "    newcomers held for the BG lanes while merged, all bit-16 arms: %lld\n", held_all);
    CHECK(held_all > 0);
}

/* fn4 pp_merge bit 32 (tp8 MERGE form 3, the relay lane): while merged, a parked decode lane with nothing of its own takes whole
 * newcomers as a BG lane does (after the idle BG lanes in the planning order), its frames flagged so the follower runs them in
 * segments. split_set + burst_set (newcomers in a burst beside decoders): every request's tokens are one process's at 3 lanes BG 1
 * and 4 lanes BG 2, with and without the hold (bit 16); the relay lane ran BG steps in at least one arm. */
TEST(relay_lane_serves_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "700", "--max-model-len", "2048" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = split_set(one.port);
        for (auto& v : burst_set(one.port)) ref.push_back(v);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { int lanes; const char* cut; int k; int merge; bool lead_seg = false; };
    long long relay_all = 0;
    for (const Arm a : { Arm{ 3, "1,3", 1, 33 }, Arm{ 4, "1,3", 2, 33 }, Arm{ 3, "0,2", 1, 49 }, Arm{ 4, "1,3", 1, 33 },
                         Arm{ 4, "2,3", 2, 37, true }, Arm{ 3, "2,3", 1, 101, true } }) {
        std::vector<std::string> env = kSplitEnv;
        env.push_back(a.k == 1 ? "RADIANCE_PP_BG_LANE=1" : "RADIANCE_PP_BG_LANE=2");
        env.push_back("RADIANCE_PP_MERGE=" + std::to_string(a.merge));
        if (a.lead_seg) {   /* + the BG (and relay) steps' leader parts in segments with the merge on (bit 4) */
            env.push_back("RADIANCE_PP_LEAD_SEG=1");
            env.push_back("RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS=15");
        }
        for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                               "RADIANCE_PP_SEG_TEST_WAIT_MS=15", "RADIANCE_PP_READY_FIRST=1" })
            env.push_back(v);
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        std::vector<std::vector<int32_t>> got = split_set(pp.port);
        for (auto& v : burst_set(pp.port)) got.push_back(v);
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
        long long merges = -1, relay = -1, passes = -1, parts = -1;
        if (const size_t at = llog.rfind("two-node pipeline: merge over the run (fn4 pp_merge="); at != std::string::npos) {
            if (const size_t m = llog.find("): ", at); m != std::string::npos) merges = std::atoll(llog.c_str() + m + 3);
            if (const size_t r = llog.find("relay lane BG steps ", at); r != std::string::npos)
                relay = std::atoll(llog.c_str() + r + std::strlen("relay lane BG steps "));
        }
        if (const size_t at = flog.rfind(" passes ran in "); at != std::string::npos)
            if (const size_t b = flog.rfind("follower: ", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "follower: %lld passes ran in %lld segments", &passes, &parts);
        relay_all += std::max(0LL, relay);
        fprintf(stderr, "    pp_merge %d, BG lanes %d, %d lanes W(%s): %d of %zu requests differ; %lld merges; relay lane BG steps %lld; follower "
                        "%lld passes in %lld segments\n", a.merge, a.k, a.lanes, a.cut, differ, ref.size(), merges, relay, passes, parts);
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 40).c_str(), tail(flog, 30).c_str());
        CHECK(merges > 0);
        CHECK(relay >= 0);
    }
    fprintf(stderr, "    relay lane BG steps, all arms: %lld\n", relay_all);
    CHECK(relay_all > 0);
}

/* tp8 loop P41 step 4 (fn4 pp_rband / RADIANCE_PP_RBAND_EVERY on the leader, RADIANCE_LLAMA_PP_RBAND=1 on both): a lone cold prompt's
 * chunks run as REVERSE passes -- the leader's head over [0, h - 1) + layer h - 1's k / v only, the follower from h - 1 (X1 of
 * W(h - 1, t)) -- every p-th (from phase), and every other pass with prompt rows carries layer h - 1's q|k|v rows to the follower's
 * own cache of that layer: cold multi-chunk prompts at 2 and 3 lanes (whole, + follower segments) on W(2,3) / W(3,4) of the four-layer
 * llama are one process's tokens AND chunk plan; a prefix hit's sequence (its first chunk at ctx > 0) is never reversed (counted);
 * the split / burst sets (BG lane, decode beside prompts) are one process's; fn4 pp_rband 0 in process = no reverse pass; negative
 * control: the follower leaves the keys' store out (RADIANCE_PP_TEST_RBAND_SKIP=1) at p = 2 -> a reversed chunk reads keys it never
 * got -> tokens differ. */
TEST(reverse_band_passes_serve_one_process_tokens_and_plan) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "8", "--max-num-batched-tokens", "256", "--max-model-len", "4096" };
    const std::vector<int32_t> p1 = prompt_ids(4101, 1300), p2 = prompt_ids(4102, 900);
    std::vector<int32_t> p3 = p1;   /* a prefix hit: p1's rows, then its own */
    for (int32_t t : prompt_ids(4103, 300)) p3.push_back(t);
    std::vector<std::vector<int32_t>> ref, ref_s;
    std::vector<std::pair<int, int>> ref_plan;
    {
        Serving one;
        REQUIRE(one.start("", ext, { "RADIANCE_LOG_STEPS=1" }));
        ref.push_back(complete(one.port, p1, 24));
        ref_plan = chunks_of(one.lead.output());
        ref.push_back(complete(one.port, p2, 24));
        ref.push_back(complete(one.port, p3, 24));
        ref_s = split_set(one.port);
        for (auto& v : burst_set(one.port)) ref_s.push_back(v);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    /* every: fn4 pp_rband's start value; post0: POST pp_rband 0 before the cold prompts (the in-process control); sets: + split / burst */
    struct Arm { int lanes; const char* cut; int every; bool fseg = false; bool skip = false; bool post0 = false; bool sets = false; int bg = 0; };
    long long rev_all = 0, unclean_all = 0;
    int skip_differ = 0;
    for (const Arm a : { Arm{ 2, "2,3", 1 }, Arm{ 3, "2,3", 2, true }, Arm{ 2, "3,4", 2 + 16 }, Arm{ 3, "2,3", 2 + 256 },
                         Arm{ 3, "2,3", 1, true, false, false, true, 1 }, Arm{ 2, "2,3", 1, false, false, true },
                         Arm{ 2, "2,3", 2, false, true } }) {
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LOG_STEPS=1", "RADIANCE_LLAMA_PP_RBAND=1" });
        env.push_back("RADIANCE_PP_RBAND_EVERY=" + std::to_string(a.every));
        if (a.bg) env.push_back("RADIANCE_PP_BG_LANE=" + std::to_string(a.bg));
        if (a.fseg)
            for (const char* v : { "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16", "RADIANCE_PP_SEG_TEST_WAIT_MS=15" })
                env.push_back(v);
        if (a.skip) env.push_back("RADIANCE_PP_TEST_RBAND_SKIP=1");
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        if (a.post0) {
            std::string out;
            Call c;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "pp_rband", 0 } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
        }
        std::vector<std::vector<int32_t>> got;
        got.push_back(complete(pp.port, p1, 24));
        const std::vector<std::pair<int, int>> plan = chunks_of(pp.lead.output());
        got.push_back(complete(pp.port, p2, 24));
        got.push_back(complete(pp.port, p3, 24));
        std::vector<std::vector<int32_t>> got_s;
        if (a.sets) {
            got_s = split_set(pp.port);
            for (auto& v : burst_set(pp.port)) got_s.push_back(v);
        }
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        int differ = 0;
        REQUIRE_EQ(ref.size(), got.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            CHECK(!ref[i].empty());
            if (ref[i] != got[i]) ++differ;
        }
        if (a.sets) {
            REQUIRE_EQ(ref_s.size(), got_s.size());
            for (size_t i = 0; i < ref_s.size(); ++i)
                if (ref_s[i] != got_s[i]) ++differ;
        }
        long long rev = -1, rows = -1, unclean = -1, keyed = -1;
        if (const size_t kk = llog.rfind(" lane steps crossed X1 in the keys form"); kk != std::string::npos)   /* step 5 */
            if (const size_t b = llog.rfind("; ", kk); b != std::string::npos) keyed = std::atoll(llog.c_str() + b + 2);
        const char* key = " lane steps ran as reverse (head band) passes, ";
        if (const size_t at = llog.rfind(key); at != std::string::npos) {
            if (const size_t b = llog.rfind("two-node pipeline: ", at); b != std::string::npos)
                rev = std::atoll(llog.c_str() + b + std::strlen("two-node pipeline: "));
            rows = std::atoll(llog.c_str() + at + std::strlen(key));
            if (const size_t u = llog.find("(tp8 loop P41); ", at); u != std::string::npos)
                unclean = std::atoll(llog.c_str() + u + std::strlen("(tp8 loop P41); "));
        }
        fprintf(stderr, "    REVERSE BAND (llama stand-in), %d lanes W(%s), pp_rband %d%s%s%s%s: %d of %zu requests differ; plan %s (%zu chunks); "
                        "%lld reverse passes, %lld rows; %lld lone chunks kept W(h, t)'s (unclean sequence); %lld steps crossed the keys\n", a.lanes, a.cut, a.every,
                a.fseg ? " + follower segments" : "", a.post0 ? " + fn4 pp_rband 0 in process" : "", a.sets ? " + split / burst sets" : "",
                a.skip ? " + the keys' store LEFT OUT (negative control)" : "", differ, ref.size() + (a.sets ? ref_s.size() : 0),
                plan == ref_plan ? "= one process's" : "DIFFERENT", plan.size(), rev, rows, unclean, keyed);
        CHECK(llog.find("the plugin declared the reverse (head) band layer") != std::string::npos);
        CHECK(plan == ref_plan);
        if (a.skip) {
            skip_differ = differ;
            CHECK(rev > 0);
            continue;
        }
        if (differ) fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        CHECK_EQ(differ, 0);
        if (a.post0) {
            CHECK_EQ(rev, 0LL);
            CHECK_EQ(keyed, 0LL);   /* step 5: pp_rband 0 crosses no keys */
            continue;
        }
        if ((a.every & 15) >= 2) CHECK(keyed > 0);   /* the chunks between reversed ones carried the keys */
        CHECK(rev > 0);
        CHECK(unclean > 0);   /* p3's chunks after its prefix hit */
        CHECK(flog.find("reverse passes ran this follower from layer") == std::string::npos);   /* (llama: no plugin line; the engine's) */
        rev_all += std::max(0LL, rev);
        unclean_all += std::max(0LL, unclean);
    }
    fprintf(stderr, "    reverse passes, all arms: %lld; lone chunks kept W(h, t)'s for an unclean sequence: %lld; the negative control: %d "
                    "requests differ\n", rev_all, unclean_all, skip_differ);
    CHECK(rev_all > 0);
    CHECK(skip_differ > 0);
}

/* tp8 loop P44 (fn4 sched_srf, upstream 67cf57b8): cold prompts sent together (nothing decoding) at 2 lanes W(0,2) / W(1,3) and 3
 * lanes W(0,3), sched_srf 2 (prompts one at a time over the lanes, the fewest rows left first; a longer waiting one held), 1 (per
 * lane) and 2 POSTed in process (the test side's A/B form), serve one process's greedy tokens for every prompt -- one process run
 * with the same requests at sched_srf 0 -- and the leader's end line counts the rule's plans (and, at 2, held admissions); at 0 the
 * line names nothing done. The chunk plans (each prompt its lone plan, no plan with two prompts) are sched_test's
 * srf_2_prompts_over_the_lanes_* (DeepSeek's grid, window keys checked). */
namespace {
std::vector<std::vector<int32_t>> srf_set(int port, int n_predict) {
    std::vector<std::future<std::vector<int32_t>>> f;
    for (const auto& [seed, n] : std::vector<std::pair<uint64_t, size_t>>{ { 501, 3300 }, { 502, 1900 }, { 503, 2600 }, { 504, 1900 } }) {
        f.push_back(std::async(std::launch::async, [port, seed, n, n_predict] { return complete(port, prompt_ids(seed, n), n_predict); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& x : f) out.push_back(x.get());
    return out;
}
long long srf_field(const std::string& log, const char* tag) {
    const size_t l = log.rfind("two-node pipeline: fn4 sched_srf ");
    if (l == std::string::npos) return -1;
    const std::string ln = log.substr(l, log.find('\n', l) - l);
    const size_t at = ln.find(tag);
    if (at == std::string::npos) return -1;
    size_t b = at;
    while (b > 0 && ln[b - 1] == ' ') --b;
    while (b > 0 && std::isdigit((unsigned char)ln[b - 1])) --b;
    return std::atoll(ln.c_str() + b);
}
}  /* namespace */

TEST(srf_concurrent_cold_prompts_one_at_a_time_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> ext = { "--max-num-seqs", "6", "--max-num-batched-tokens", "2112", "--max-model-len", "4096",
                                           "--prefill-span", "2", "--checkpoint-interval", "1024" };
    std::vector<std::vector<int32_t>> ref, ref_mix;
    {
        Serving one;
        REQUIRE(one.start("", ext));
        ref = srf_set(one.port, 12);
        ref_mix = split_set(one.port);   /* decoders beside newcomers: the rule turns off and on as decode rows come and go */
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    REQUIRE(ref.size() == 4);
    for (const auto& t : ref) CHECK_EQ(t.size(), (size_t)12);
    /* bg: BEST's shape -- 4 lanes, the last two BG lanes (RADIANCE_PP_BG_LANE=2) with the follower's one-layer segments and the leader's
     * decode tails first; mix: the split set (decoders + newcomers) after the cold set, in the same start */
    struct Arm { int lanes; const char* cut; int srf; bool post; bool bg = false; bool mix = false; };
    for (const Arm a : { Arm{ 2, "0,2", 0, false }, Arm{ 2, "0,2", 2, false }, Arm{ 2, "1,3", 2, false }, Arm{ 3, "0,3", 2, false },
                         Arm{ 2, "0,2", 1, false }, Arm{ 2, "0,2", 2, true }, Arm{ 4, "1,3", 2, false, true, true },
                         Arm{ 4, "0,2", 2, false, true, true }, Arm{ 3, "1,3", 1, false, false, true } }) {
        std::vector<std::string> ex = ext;
        ex.push_back("--pp-lanes");
        ex.push_back(std::to_string(a.lanes));
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_PP_SPAN_SHORT=24576" });
        if (a.bg)
            for (const char* v : { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16",
                                   "RADIANCE_PP_READY_FIRST=1" })
                env.push_back(v);
        if (a.srf && !a.post) env.push_back("RADIANCE_SCHED_SRF=" + std::to_string(a.srf));
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        if (a.post) {
            Call c;
            std::string out;
            const int st = c.send(pp.port, "/admin/fn4", json{ { "sched_srf", a.srf } }.dump()) ? c.read(&out) : -1;
            CHECK_EQ(st, 200);
            (void)complete(pp.port, prompt_ids(9, 8), 2);   /* a step release applies it */
        }
        std::vector<std::vector<int32_t>> got = srf_set(pp.port, 12), got_mix;
        if (a.post) got = srf_set(pp.port, 12);   /* the in-process arm: a second set, the switch long applied */
        if (a.mix) got_mix = split_set(pp.port);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output();
        int differ = 0;
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i) differ += got[i] != ref[i];
        if (a.mix) {
            CHECK_EQ(got_mix.size(), ref_mix.size());
            for (size_t i = 0; i < ref_mix.size() && i < got_mix.size(); ++i) differ += got_mix[i] != ref_mix[i];
        }
        const long long plans = srf_field(llog, "lane plans under the rule"), held = srf_field(llog, "admissions held"),
                        out = srf_field(llog, "times a running prompt sat a plan out");
        fprintf(stderr, "    %d lanes%s W(%s), sched_srf %d%s%s: %d of %zu requests differ from one process; rule plans %lld, sat out %lld, "
                        "held %lld\n", a.lanes, a.bg ? " (BG 2)" : "", a.cut, a.srf, a.post ? " (POSTed)" : "", a.mix ? " + the split set" : "",
                differ, ref.size() + (a.mix ? ref_mix.size() : 0), plans, out, held);
        CHECK_EQ(got.size(), ref.size());
        CHECK_EQ(differ, 0);
        if (differ) fprintf(stderr, "    leader:\n%s\n", tail(llog, 30).c_str());
        if (a.srf == 0) {
            CHECK(plans <= 0);
            continue;
        }
        CHECK(plans > 0);
        if (a.srf == 2) CHECK(held > 0);
    }
}

/* tp8 loop D53: PREEMPTION BY RECOMPUTE THROUGH THE PIPELINE. Eight requests at once whose prompts + outputs (3,600 tokens)
 * overrun a KV pool of 600-2,600 tokens, then two alone: the leader's lane schedulers preempt by recompute (the blocks go
 * back, the recompute re-prefills prompt ++ output, adopting the published prefix), and the follower's layers read and
 * write the same block ids. Every request's greedy tokens are one process's (the default 512 MiB device: a pool that
 * never preempts), at 1 / 2 / 3 lanes, a wrap-around cut with the llama KV copy checked (RADIANCE_PP_KV_CHECK=1), and
 * BEST's shape (4 lanes BG 2, FOL_ORDER 2, the follower's segments). The pool is found, not fixed: the pair's own
 * startup report at a device size (the leader caps at the follower's): the smallest size that starts with a pool >= kPreLo
 * (on the host backend a MiB is ~2,000 tokens of this llama's pool at two layers a node, so the window is wide). */
std::vector<std::vector<int32_t>> preempt_set(int port) {
    std::vector<std::vector<int32_t>> out;
    std::vector<std::future<std::vector<int32_t>>> f;
    for (uint64_t k = 0; k < 8; ++k)
        f.push_back(std::async(std::launch::async, [port, k] {
            return complete(port, prompt_ids(31 + k, 80 + 15 * k), 300 + 5 * (int)k);
        }));
    for (auto& x : f) out.push_back(x.get());
    for (uint64_t s : { 41, 42 }) out.push_back(complete(port, prompt_ids(s, 90), 40));
    return out;
}
/* The shape: eight sequences (prompts 80-185 tokens, outputs 300-335: 3,600 tokens at once, the longest 520), 64-token steps */
std::vector<std::string> preempt_shape(int lanes) {
    std::vector<std::string> a = { "--max-num-seqs", "8" };
    if (lanes > 0) a = cat(a, { "--pp-lanes", std::to_string(lanes) });
    return a;
}
/* The pool a startup report states (" a card: N tokens" -- the leader's, after the pair's cap), -1 when absent. */
long long pool_in(const std::string& log) {
    static const std::string kTag = " a card: ";
    const size_t at = log.rfind(kTag);
    return at == std::string::npos ? -1 : std::atoll(log.c_str() + at + kTag.size());
}
constexpr long long kPreLo = 600, kPreHi = 2600;

TEST(preemption_by_recompute_on_the_pipeline_serves_one_process_tokens) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", preempt_shape(0)));
        ref = preempt_set(one.port);
        const double pre = metric(get(one.port, "/metrics"), "vllm:num_preemptions_total");
        fprintf(stderr, "    one process, pool %lld tokens: %.0f preemption(s)\n", pool_in(one.lead.output()), pre);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best; };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    long long pre_all = 0;
    const char* only = std::getenv("PP_E2E_PREEMPT_ARM");   /* a debugging aid: the arms whose name holds this text */
    for (const Arm a : { Arm{ "1 lane", "0,2", 1, false }, Arm{ "2 lanes", "0,2", 2, false }, Arm{ "3 lanes, wrap-around", "1,3", 3, false },
                         Arm{ "2 lanes, wrap-around", "1,3", 2, false }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true } }) {
        if (only && !std::strstr(a.name, only)) continue;
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        const std::vector<std::string> ex = preempt_shape(a.lanes);
        /* the device size: bisect [64, 512) MiB on the pair's own pool (each probe a start and a stop) */
        auto pool_at = [&](int mib) -> long long {
            Serving pr;
            if (!pr.start(a.cut, ex, cat(env, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(mib) }))) return -1;
            const long long p = pool_in(pr.lead.output());
            kill(pr.lead.child.pid, SIGTERM);
            pr.lead.finish(60);
            pr.fol.finish(30);
            return p;
        };
        int lo = 64, hi = 512;
        long long plo = pool_at(lo), phi = pool_at(hi);
        if (!(plo < kPreLo && phi >= kPreLo))
            fprintf(stderr, "    %s W(%s): pools %lld / %lld tokens at %d / %d MiB do not bracket %lld\n", a.name, a.cut, plo, phi, lo,
                    hi, kPreLo);
        REQUIRE(plo < kPreLo && phi >= kPreLo);
        while (hi - lo > 1) {
            const int m = (lo + hi) / 2;
            const long long p = pool_at(m);
            if (p >= kPreLo) { hi = m; phi = p; } else { lo = m; plo = p; }
        }
        if (phi >= kPreHi) fprintf(stderr, "    %s W(%s): the smallest pool >= %lld is %lld tokens (%d MiB)\n", a.name, a.cut, kPreLo, phi, hi);
        REQUIRE(phi < kPreHi);
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, cat(env, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(hi) })));
        const std::vector<std::vector<int32_t>> got = preempt_set(pp.port);
        const double pre = metric(get(pp.port, "/metrics"), "vllm:num_preemptions_total");
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0, empty = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (got[i].empty()) { ++empty; ++differ; continue; }
            if (ref[i] == got[i]) continue;
            ++differ;
            size_t j = 0;
            while (j < ref[i].size() && j < got[i].size() && ref[i][j] == got[i][j]) ++j;
            fprintf(stderr, "    request %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", i, got[i].size(),
                    ref[i].size(), j);
        }
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const bool notowner = flog.find("NOT the owner's bytes") != std::string::npos;
        const long long failed = (long long)[&] { size_t n = 0, at = 0; while ((at = llog.find("could not be admitted", at)) != std::string::npos) { ++n; ++at; } return n; }();
        if (const char* dd = std::getenv("PP_E2E_PREEMPT_LOGS")) {
            tiny::write_text(std::string(dd) + "/" + std::to_string(a.lanes) + "-" + a.cut + "-leader.log", llog);
            tiny::write_text(std::string(dd) + "/" + std::to_string(a.lanes) + "-" + a.cut + "-follower.log", flog);
        }
        fprintf(stderr, "    PREEMPTION %-26s W(%s): pool %lld tokens (%d MiB), %.0f preemption(s); %d of %zu requests differ from one "
                        "process (%d empty = failed: %lld 'could not be admitted' lines); KV-copy checks %lld, %lld differ%s\n",
                a.name, a.cut, pool_in(llog), hi, pre, differ, ref.size(), empty, failed, checks, kdiff, notowner ? " (NOT the owner's bytes)" : "");
        CHECK(pre > 0);
        CHECK_EQ(differ, 0);
        CHECK(!notowner);
        if (std::strcmp(a.cut, "0,2") != 0) CHECK(checks > 0);
        if (differ || notowner || !(pre > 0))
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        pre_all += pre > 0 ? (long long)pre : 0;
    }
    fprintf(stderr, "    preemptions, all arms: %lld\n", pre_all);
}

/* tp8 loop D53b: CLIENTS THAT GO AWAY MID-REQUEST, through the pipeline. Claude Code drops requests (an interrupt closes the
 * connection): the server cancels at the scheduler while the request's step may be in flight on both nodes, a piece of its
 * prompt lent to another lane, or its blocks about to be reused by a newcomer. Six requests that stay and six that are dropped
 * 60-660 ms after they were sent (non-streaming /completion: the transport sees the closed socket), all at once, then two alone:
 * every request that stayed answers one process's tokens, at 2 lanes W(0,2), 3 lanes W(1,3) (KV copy checked) and BEST's shape
 * (4 lanes BG 2, READY_FIRST, FOL_ORDER 2, the follower's segments), at the default pool and at the preemption case's small one
 * (cancels beside preemptions); both nodes exit 0, the KV-copy check 0 differ, no engine E line. */
std::vector<std::vector<int32_t>> cancel_set(int port, int* dropped) {
    std::vector<std::vector<int32_t>> out;
    std::vector<std::future<std::vector<int32_t>>> f;
    std::vector<std::thread> gone;
    std::atomic<int> drops{0};
    for (uint64_t k = 0; k < 6; ++k) {
        gone.emplace_back([port, k, &drops] {
            const json b = { { "prompt", prompt_ids(61 + k, 70 + 23 * k) }, { "n_predict", 400 }, { "temperature", 0.0 },
                             { "ignore_eos", true }, { "return_tokens", true } };
            Call c;
            if (!c.send(port, "/completion", b.dump())) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(60 + 120 * k));
            shutdown(c.fd, SHUT_RDWR);
            ++drops;
        });
        f.push_back(std::async(std::launch::async, [port, k] { return complete(port, prompt_ids(51 + k, 60 + 21 * k), 90 + 7 * (int)k); }));
    }
    for (auto& x : f) out.push_back(x.get());
    for (auto& t : gone) t.join();
    for (uint64_t s : { 71, 72 }) out.push_back(complete(port, prompt_ids(s, 110), 40));
    if (dropped) *dropped = drops.load();
    return out;
}

TEST(clients_that_go_away_mid_request_leave_the_others_one_process_tokens) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", { "--max-num-seqs", "12" }));
        ref = cancel_set(one.port, nullptr);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, small; };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    for (const Arm a : { Arm{ "2 lanes", "0,2", 2, false, false }, Arm{ "3 lanes, wrap-around", "1,3", 3, false, false },
                         Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, false }, Arm{ "3 lanes, wrap-around, small pool", "1,3", 3, false, true },
                         Arm{ "BEST's shape, small pool", "1,3", 4, true, true } }) {
        std::vector<std::string> env = cat(kSplitEnv, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        const std::vector<std::string> ex = { "--max-num-seqs", "12", "--pp-lanes", std::to_string(a.lanes) };
        int mib = 0;
        if (a.small) {   /* the smallest device the pair starts on (its pool caps both nodes), as the preemption case finds it */
            auto starts = [&](int m) -> long long {
                Serving pr;
                if (!pr.start(a.cut, ex, cat(env, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(m) }))) return -1;
                const long long p = pool_in(pr.lead.output());
                kill(pr.lead.child.pid, SIGTERM);
                pr.lead.finish(60);
                pr.fol.finish(30);
                return p;
            };
            int lo = 64, hi = 512;
            REQUIRE(starts(lo) < 600 && starts(hi) >= 600);
            while (hi - lo > 1) { const int m = (lo + hi) / 2; if (starts(m) >= 600) hi = m; else lo = m; }
            mib = hi;
            env.push_back("RADIANCE_HOST_VRAM_MIB=" + std::to_string(mib));
        }
        Serving pp;
        REQUIRE(pp.start(a.cut, ex, env));
        int dropped = 0;
        const std::vector<std::vector<int32_t>> got = cancel_set(pp.port, &dropped);
        const double pre = metric(get(pp.port, "/metrics"), "vllm:num_preemptions_total");
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(ref.size(), got.size());
        int differ = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] == got[i]) continue;
            ++differ;
            size_t j = 0;
            while (j < ref[i].size() && j < got[i].size() && ref[i][j] == got[i][j]) ++j;
            fprintf(stderr, "    request %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", i, got[i].size(),
                    ref[i].size(), j);
        }
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const bool notowner = flog.find("NOT the owner's bytes") != std::string::npos;
        size_t cancelled = 0, elines = 0;
        for (size_t at = 0; (at = llog.find(" done from ", at)) != std::string::npos; ++at) {
            const size_t f = llog.find("finish=", at), nl = llog.find('\n', at);
            if (f != std::string::npos && f < nl && llog.compare(f + 7, 6, "length") != 0) ++cancelled;
        }
        for (const std::string* lg : { &llog, &flog })
            for (size_t at = 0; (at = lg->find("\nE ", at)) != std::string::npos; ++at) ++elines;
        if (const char* dd = std::getenv("PP_E2E_PREEMPT_LOGS")) {
            tiny::write_text(std::string(dd) + "/cancel-" + std::to_string(a.lanes) + "-" + a.cut + (a.small ? "-s" : "") + "-leader.log", llog);
            tiny::write_text(std::string(dd) + "/cancel-" + std::to_string(a.lanes) + "-" + a.cut + (a.small ? "-s" : "") + "-follower.log", flog);
        }
        fprintf(stderr, "    CANCEL %-34s W(%s)%s: %d dropped, %zu ended other than by length, %.0f preemption(s); %d of %zu kept requests differ "
                        "from one process; KV-copy checks %lld, %lld differ%s; %zu E lines\n",
                a.name, a.cut, a.small ? (" (" + std::to_string(mib) + " MiB)").c_str() : "", dropped, cancelled, pre, differ, ref.size(), checks, kdiff,
                notowner ? " (NOT the owner's bytes)" : "", elines);
        CHECK_EQ(dropped, 6);
        CHECK(cancelled > 0);
        CHECK_EQ(differ, 0);
        CHECK(!notowner);
        if (std::strcmp(a.cut, "0,2") != 0) CHECK(checks > 0);
        CHECK_EQ(elines, (size_t)0);
        if (differ || notowner || elines)
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
    }
}

/* tp8 loop D54: PRODUCTION REQUEST FORMS through the pipeline. Every case above is greedy (temperature 0); the production client
 * (Claude Code through LiteLLM) samples, streams and stops on strings. One request set of the forms a client sends -- seeded
 * sampling with and without the cut stages (top-k / top-p / min-p), penalties sampled and greedy, DRY, a GBNF grammar greedy and
 * sampled, a stop string, three of them streamed, n = 2 (two requests over one prompt: the second a prefix hit, perhaps in another
 * lane) -- three alone, then all at once. One process answers each form ALONE first (the reference), then the same set, which must
 * give the same tokens (the draw is (seed, position), not the batch's); the pipeline must give them too at 1 / 2 lanes W(0,2), 3
 * lanes W(1,3) and BEST's shape (4 lanes BG 2, READY_FIRST, FOL_ORDER 2, the follower's one-layer segments), the last two with
 * the llama KV copy checked, and the DSpark-drafted fixture (D38a) at 2 lanes W(0,3) / 3 lanes W(1,3) against its own one
 * process. A streamed reply's chunks must assemble its non-streamed sibling's tokens; both nodes exit 0, no E line. */
namespace {

/* POST `body` to `path`; the status, the headers and the body (de-chunked when the reply came chunked, as a stream does). */
int post_raw(int port, const std::string& path, const std::string& body, std::string* headers, std::string* out) {
    Call c;
    if (!c.send(port, path, body)) return -1;
    std::string raw;
    char buf[16384];
    for (;;) {
        const ssize_t n = ::recv(c.fd, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
    }
    const size_t h = raw.find("\r\n\r\n");
    if (raw.compare(0, 9, "HTTP/1.1 ") != 0 || h == std::string::npos) return -1;
    *headers = raw.substr(0, h);
    std::string b = raw.substr(h + 4);
    std::string lower = *headers;
    for (char& ch : lower) ch = (char)std::tolower((unsigned char)ch);
    if (lower.find("transfer-encoding: chunked") != std::string::npos) {
        std::string d;
        size_t at = 0;
        for (;;) {
            const size_t eol = b.find("\r\n", at);
            if (eol == std::string::npos) break;
            const size_t len = std::strtoul(b.c_str() + at, nullptr, 16);
            if (len == 0) break;
            if (eol + 2 + len > b.size()) { d += b.substr(eol + 2); break; }
            d += b.substr(eol + 2, len);
            at = eol + 2 + len + 2;
        }
        b = d;
    }
    *out = b;
    return std::atoi(raw.c_str() + 9);
}

/* One reply: the tokens of each choice, the text of the first, the draft counts, and what was wrong with it (empty: nothing). */
struct Reply {
    int status = -1;
    std::vector<std::vector<int32_t>> tok;
    std::string text, why, stop_type;
    long long drafted = 0, accepted = 0;
};

Reply ask(int port, const json& body) {
    Reply r;
    std::string hd, out;
    r.status = post_raw(port, "/completion", body.dump(), &hd, &out);
    if (r.status != 200) { r.why = "HTTP " + std::to_string(r.status) + ": " + out.substr(0, 240); return r; }
    auto drafts = [&r](const json& j) {
        r.drafted += j.value(json::json_pointer("/timings/draft_n"), (long long)0);
        r.accepted += j.value(json::json_pointer("/timings/draft_n_accepted"), (long long)0);
    };
    if (body.value("stream", false)) {
        /* llama-server's native events: each `stop: false` chunk carries the ids that made its text; the last (`stop: true`) is the
         * whole result, its ids again when return_tokens asked for them */
        std::vector<int32_t> got, last;
        bool ended = false;
        for (size_t at = 0; (at = out.find("data: ", at)) != std::string::npos;) {
            const size_t e = out.find("\n", at);
            const json j = json::parse(out.substr(at + 6, e == std::string::npos ? std::string::npos : e - at - 6), nullptr, false);
            at = e == std::string::npos ? out.size() : e;
            if (!j.is_object()) continue;
            if (j.contains("error")) { r.why = "an error event: " + j["error"].dump().substr(0, 200); continue; }
            if (!j.value("stop", false)) {
                for (int32_t t : j.value("tokens", std::vector<int32_t>{})) got.push_back(t);
                r.text += j.value("content", std::string());
            } else {
                ended = true;
                last = j.value("tokens", std::vector<int32_t>{});
                r.stop_type = j.value("stop_type", std::string());
                drafts(j);
            }
        }
        if (!ended && r.why.empty()) r.why = "the stream ended without its last event";
        if (!last.empty() && last != got && r.why.empty())
            r.why = "the chunks assemble " + std::to_string(got.size()) + " tokens, the last event names " + std::to_string(last.size());
        r.tok.push_back(got);
        return r;
    }
    const json j = json::parse(out, nullptr, false);
    const json arr = j.is_array() ? j : json::array({ j });
    for (const json& c : arr) {
        if (!c.is_object() || !c.contains("tokens")) { r.why = "no tokens in " + c.dump().substr(0, 200); return r; }
        r.tok.push_back(c["tokens"].get<std::vector<int32_t>>());
        drafts(c);
    }
    if (!arr.empty()) {
        r.text = arr[0].value("content", std::string());
        r.stop_type = arr[0].value("stop_type", std::string());
    }
    return r;
}

/* The same form with its sampler stages taken out (temperature 0, no penalties, DRY or grammar): a control that the stages acted. */
json plain_of(json b) {
    for (const char* k : { "top_k", "top_p", "min_p", "seed", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
                           "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "grammar", "stop", "stream", "n" })
        b.erase(k);
    b["temperature"] = 0.0;
    return b;
}

/* Text made of the grammar's words only. */
bool in_grammar(const std::string& t) {
    static const char* w[] = { "cat ", "dog ", "sat on ", "the ", "mat " };
    size_t at = 0;
    while (at < t.size()) {
        bool hit = false;
        for (const char* x : w) {
            const size_t n = std::strlen(x);
            if (t.compare(at, n, x) == 0) { at += n; hit = true; break; }
            /* the reply may end inside a word (n_predict) */
            if (t.size() - at < n && std::string(x).compare(0, t.size() - at, t, at, std::string::npos) == 0) return true;
        }
        if (!hit) return false;
    }
    return true;
}

const char kProdGrammar[] = "root ::= (\"cat \" | \"dog \" | \"sat on \" | \"the \" | \"mat \")+";

/* The forms, in a fixed order. `stop` is the stop string of form 8 (found by the reference: three letters of its own greedy text). */
struct ProdForm { const char* name; json body; int sibling; };
std::vector<ProdForm> prod_forms(const std::string& stop) {
    auto base = [](uint64_t s, size_t len, int n) {
        return json{ { "prompt", prompt_ids(s, len) }, { "n_predict", n }, { "ignore_eos", true }, { "return_tokens", true } };
    };
    std::vector<ProdForm> f;
    json b;
    b = base(81, 90, 40); b["temperature"] = 0.0; f.push_back({ "greedy", b, -1 });
    b = base(82, 110, 48); b["temperature"] = 0.9; b["top_k"] = 40; b["top_p"] = 0.92; b["min_p"] = 0.03; b["seed"] = 1001;
    f.push_back({ "seeded T0.9 top-k/p min-p", b, -1 });
    b = base(83, 70, 48); b["temperature"] = 1.0; b["seed"] = 1002; f.push_back({ "seeded T1.0", b, -1 });
    b = base(84, 130, 56); b["temperature"] = 0.7; b["seed"] = 1003; b["repeat_penalty"] = 1.2; b["repeat_last_n"] = 48;
    b["presence_penalty"] = 0.5; b["frequency_penalty"] = 0.4; f.push_back({ "seeded T0.7 penalties", b, -1 });
    b = base(85, 60, 48); b["temperature"] = 0.0; b["repeat_penalty"] = 1.3; b["repeat_last_n"] = 32; b["frequency_penalty"] = 0.5;
    f.push_back({ "greedy penalties", b, -1 });
    b = base(86, 100, 48); b["temperature"] = 0.8; b["seed"] = 1005; b["dry_multiplier"] = 0.8; b["dry_base"] = 1.75;
    b["dry_allowed_length"] = 2; b["dry_penalty_last_n"] = 64; f.push_back({ "seeded T0.8 DRY", b, -1 });
    b = base(87, 80, 40); b["temperature"] = 0.0; b["grammar"] = kProdGrammar; f.push_back({ "greedy grammar", b, -1 });
    b = base(88, 75, 40); b["temperature"] = 0.8; b["seed"] = 1007; b["grammar"] = kProdGrammar; f.push_back({ "seeded grammar", b, -1 });
    b = base(89, 95, 60); b["temperature"] = 0.0; if (!stop.empty()) b["stop"] = json::array({ stop }); f.push_back({ "greedy stop string", b, -1 });
    b = f[1].body; b["stream"] = true; f.push_back({ "streamed seeded T0.9", b, 1 });
    b = f[3].body; b["stream"] = true; f.push_back({ "streamed penalties", b, 3 });
    b = f[6].body; b["stream"] = true; f.push_back({ "streamed grammar", b, 6 });
    b = base(90, 85, 44); b["temperature"] = 0.85; b["top_p"] = 0.9; b["seed"] = 1012; b["n"] = 2; f.push_back({ "n = 2 seeded", b, -1 });
    return f;
}

/* Forms 1, 6 and 12 alone, then all at once; the replies in form order (the alone ones after). */
std::vector<Reply> prod_set(int port, const std::vector<ProdForm>& forms) {
    std::vector<Reply> alone;
    for (size_t i : { (size_t)1, (size_t)6, (size_t)12 }) alone.push_back(ask(port, forms[i].body));
    std::vector<std::future<Reply>> f;
    for (const ProdForm& x : forms) f.push_back(std::async(std::launch::async, [port, &x] { return ask(port, x.body); }));
    std::vector<Reply> out;
    for (auto& x : f) out.push_back(x.get());
    for (Reply& r : alone) out.push_back(std::move(r));
    return out;
}
const size_t kProdAloneOf[] = { 1, 6, 12 };

size_t count_of(const std::string& s, const std::string& what) {
    size_t n = 0;
    for (size_t at = 0; (at = s.find(what, at)) != std::string::npos; ++at) ++n;
    return n;
}

}  /* namespace */

TEST(production_request_forms_serve_one_process_tokens) {
    REQUIRE(world().ok);
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> seqs = { "--max-num-seqs", "16" };
    /* Form 8's stop string: two printable bytes of its own greedy text from its 8th byte on (their first place there; else one byte),
     * found by each fixture's one process. A reply
     * cut by a stop string is compared by its TEXT and finish: the server cuts the text at the string, while the token ids it returns
     * are whatever the step had produced when the stop was read (llama-server's form) -- timing, not the answer. */
    struct Ref { std::vector<Reply> alone, set; std::vector<ProdForm> forms; std::string stop; };
    std::map<int, Ref> refs;   /* 0: the plain fixture, 1: the drafted one */
    auto same = [](const Reply& x, const Reply& y, size_t form) {
        return form == 8 ? x.text == y.text && x.stop_type == y.stop_type : x.tok == y.tok;
    };
    for (int ds = 0; ds < 2; ++ds) {
        Serving one;
        if (ds) one.model = worldds().llama_rad;
        REQUIRE(one.start("", ds ? cat(spec, seqs) : seqs));
        Ref& rf = refs[ds];
        {
            const Reply p = ask(one.port, prod_forms("")[8].body);
            REQUIRE(p.why.empty());
            auto ascii = [](char ch) { return ch > ' ' && ch < 0x7f; };
            for (size_t len : { (size_t)2, (size_t)1 })
                for (size_t i = 8; i + len <= p.text.size() && rf.stop.empty(); ++i)
                    if (ascii(p.text[i]) && ascii(p.text[i + len - 1]) && p.text.find(p.text.substr(i, len)) == i) rf.stop = p.text.substr(i, len);
            fprintf(stderr, "    one process (%s): form 8's stop string '%s' (its greedy text without it: %zu bytes)\n", ds ? "drafted" : "plain",
                    rf.stop.c_str(), p.text.size());
        }
        REQUIRE(!rf.stop.empty());
        rf.forms = prod_forms(rf.stop);
        const std::vector<ProdForm>& forms = rf.forms;
        for (const ProdForm& x : forms) rf.alone.push_back(ask(one.port, x.body));
        rf.set = prod_set(one.port, forms);
        /* the one process's own claims: every form served, a streamed form = its sibling, the set = each form alone */
        int bad = 0;
        for (size_t i = 0; i < forms.size(); ++i) {
            const Reply& a = rf.alone[i];
            if (!a.why.empty() || a.tok.empty() || a.tok[0].empty()) {
                ++bad;
                fprintf(stderr, "    one process (%s) form %zu '%s': %s\n", ds ? "drafted" : "plain", i, forms[i].name,
                        a.why.empty() ? "no tokens" : a.why.c_str());
                continue;
            }
            if (forms[i].sibling >= 0 && a.tok != rf.alone[(size_t)forms[i].sibling].tok) {
                ++bad;
                fprintf(stderr, "    one process (%s) form %zu '%s' streamed differs from its sibling %d\n", ds ? "drafted" : "plain", i,
                        forms[i].name, forms[i].sibling);
            }
            if (!same(rf.set[i], a, i)) {
                ++bad;
                fprintf(stderr, "    one process (%s) form %zu '%s': in the set != alone%s\n", ds ? "drafted" : "plain", i, forms[i].name,
                        rf.set[i].why.empty() ? "" : (" (" + rf.set[i].why + ")").c_str());
            }
        }
        for (size_t k = 0; k < 3; ++k)
            if (!same(rf.set[forms.size() + k], rf.alone[kProdAloneOf[k]], kProdAloneOf[k])) {
                ++bad;
                fprintf(stderr, "    one process (%s) form %zu alone twice differs\n", ds ? "drafted" : "plain", kProdAloneOf[k]);
            }
        /* the controls: every staged form answers other tokens than its plain greedy self (else the stage did nothing here and the
         * form proves nothing), the grammar's replies are its words, the stop form ended on its string */
        int acted = 0, staged = 0;
        for (size_t i : { (size_t)1, (size_t)2, (size_t)3, (size_t)4, (size_t)5, (size_t)6, (size_t)7, (size_t)12 }) {
            ++staged;
            const Reply pl = ask(one.port, plain_of(forms[i].body));
            if (!rf.alone[i].tok.empty() && pl.tok.size() == 1 && rf.alone[i].tok[0] != pl.tok[0]) ++acted;
            else fprintf(stderr, "    one process (%s) form %zu '%s' answers its plain greedy tokens: the stage did not act (%zu / %zu choices%s%s)\n",
                         ds ? "drafted" : "plain", i, forms[i].name, rf.alone[i].tok.size(), pl.tok.size(), pl.why.empty() ? "" : ": ",
                         pl.why.c_str());
        }
        const bool gram = in_grammar(rf.alone[6].text) && in_grammar(rf.alone[7].text);
        const bool stopped = rf.alone[8].stop_type == "word" && rf.alone[8].text.find(rf.stop) == std::string::npos;
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
        fprintf(stderr, "    one process (%s): %d of %d staged forms answer other tokens than plain greedy; grammar replies %s ('%s' / '%s'); "
                        "the stop form %s (stop_type '%s', %zu text bytes, %zu token ids)\n",
                ds ? "drafted" : "plain", acted, staged, gram ? "in the grammar" : "NOT in the grammar", rf.alone[6].text.substr(0, 40).c_str(),
                rf.alone[7].text.substr(0, 40).c_str(), stopped ? "ended on its string" : "did NOT end on its string",
                rf.alone[8].stop_type.c_str(), rf.alone[8].text.size(), rf.alone[8].tok.empty() ? (size_t)0 : rf.alone[8].tok[0].size());
        CHECK_EQ(acted, staged);
        CHECK(gram);
        CHECK(stopped);
        long long dr = 0, ac = 0;
        for (const Reply& r : rf.set) { dr += r.drafted; ac += r.accepted; }
        fprintf(stderr, "    one process (%s): %zu forms, %d not as claimed; the set %lld drafted / %lld accepted; n = 2's choices %s\n",
                ds ? "drafted" : "plain", forms.size(), bad, dr, ac,
                rf.alone[12].tok.size() == 2 && rf.alone[12].tok[0] != rf.alone[12].tok[1] ? "differ (seeds k, k + 1)" : "DO NOT differ");
        CHECK_EQ(bad, 0);
        CHECK_EQ(rf.alone[12].tok.size(), (size_t)2);
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, ds, kv; };
    for (const Arm a : { Arm{ "1 lane", "0,2", 1, false, false, false }, Arm{ "2 lanes", "0,2", 2, false, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, false, true },
                         Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, false, true },
                         Arm{ "drafted, 2 lanes", "0,3", 2, false, true, false },
                         Arm{ "drafted, 3 lanes, wrap-around", "1,3", 3, false, true, true },
                         Arm{ "drafted, BEST's shape 4 lanes BG 2", "1,3", 4, true, true, true } }) {
        std::vector<std::string> env = kSplitEnv;
        if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        Serving pp;
        if (a.ds) pp.model = worldds().llama_rad;
        const std::vector<std::string> ex = cat(a.ds ? cat(spec, seqs) : seqs, { "--pp-lanes", std::to_string(a.lanes) });
        REQUIRE(pp.start(a.cut, ex, env));
        const Ref& rf = refs[a.ds ? 1 : 0];
        const std::vector<ProdForm>& forms = rf.forms;
        const std::vector<Reply> got = prod_set(pp.port, forms);
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        REQUIRE_EQ(got.size(), rf.set.size());
        int differ = 0, failed = 0;
        long long dr = 0, ac = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const size_t f = i < forms.size() ? i : kProdAloneOf[i - forms.size()];
            const Reply& want = rf.alone[f];
            dr += got[i].drafted;
            ac += got[i].accepted;
            if (!got[i].why.empty()) {
                ++failed;
                fprintf(stderr, "    %s form %zu '%s'%s: %s\n", a.name, f, forms[f].name, i < forms.size() ? "" : " (alone)", got[i].why.c_str());
            }
            if (same(got[i], want, f)) continue;
            ++differ;
            if (f == 8)
                fprintf(stderr, "    %s form 8 (stop string): text %zu bytes '%s' vs one process's %zu, finish '%s' vs '%s'\n", a.name, got[i].text.size(),
                        got[i].text.substr(0, 60).c_str(), want.text.size(), got[i].stop_type.c_str(), want.stop_type.c_str());
            const size_t nc = std::min(got[i].tok.size(), want.tok.size());
            for (size_t c = 0; c < nc; ++c) {
                if (got[i].tok[c] == want.tok[c]) continue;
                size_t j = 0;
                while (j < got[i].tok[c].size() && j < want.tok[c].size() && got[i].tok[c][j] == want.tok[c][j]) ++j;
                fprintf(stderr, "    %s form %zu '%s'%s choice %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", a.name, f,
                        forms[f].name, i < forms.size() ? "" : " (alone)", c, got[i].tok[c].size(), want.tok[c].size(), j);
            }
            if (got[i].tok.size() != want.tok.size())
                fprintf(stderr, "    %s form %zu: %zu choices vs %zu\n", a.name, f, got[i].tok.size(), want.tok.size());
        }
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
        if (const char* dd = std::getenv("PP_E2E_PROD_LOGS")) {
            const std::string tag = std::string(dd) + "/prod-" + std::to_string(a.lanes) + "-" + a.cut + (a.ds ? "-ds" : "") + (a.best ? "-best" : "");
            tiny::write_text(tag + "-leader.log", llog);
            tiny::write_text(tag + "-follower.log", flog);
        }
        fprintf(stderr, "    PROD %-34s W(%s): %d of %zu replies differ from one process, %d not served as asked; %lld drafted / %lld accepted; "
                        "KV-copy checks %lld, %lld differ; %zu E lines; the stop form's token ids %zu (one process %zu)\n",
                a.name, a.cut, differ, got.size(), failed, dr, ac, checks, kdiff, elines, got[8].tok.empty() ? (size_t)0 : got[8].tok[0].size(),
                rf.alone[8].tok.empty() ? (size_t)0 : rf.alone[8].tok[0].size());
        CHECK_EQ(differ, 0);
        CHECK_EQ(failed, 0);
        CHECK_EQ(elines, (size_t)0);
        if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
        if (a.ds) CHECK(dr > 0);
        if (differ || failed || elines)
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
    }
}

/* tp8 loop D54b: CONVERSATIONS, as Claude Code holds them: each turn's prompt is the previous turn's prompt + its answer + the
 * user's next text, so every turn after the first is a prefix hit over KV that a DECODE wrote (in whichever lane served it) and
 * the next turn may land in another lane. Four sessions at once, three turns each, sampled (seeded, top-p), two of the sessions
 * streamed; one process's tokens per (session, turn) -- the sessions run at once and one after another there, the same -- and the
 * pipeline's at 2 lanes W(0,2), 3 lanes W(1,3) and BEST's shape (KV copy checked), and the DSpark-drafted fixture at BEST's shape;
 * the prefix cache must have served (attention-cached prompt tokens > 0). */
namespace {
std::vector<std::vector<int32_t>> conversations(int port, bool together) {
    auto session = [port](int s) {
        std::vector<std::vector<int32_t>> answers;
        std::vector<int32_t> prompt = prompt_ids(100 + (uint64_t)s, 150 + 20 * (size_t)s);
        for (int turn = 0; turn < 3; ++turn) {
            json b = { { "prompt", prompt }, { "n_predict", 32 }, { "ignore_eos", true }, { "return_tokens", true },
                       { "temperature", 0.8 }, { "top_p", 0.95 }, { "seed", 2000 + 10 * s + turn } };
            if (s % 2) b["stream"] = true;
            const Reply r = ask(port, b);
            answers.push_back(r.why.empty() && r.tok.size() == 1 ? r.tok[0] : std::vector<int32_t>{});
            if (answers.back().empty()) {
                fprintf(stderr, "    session %d turn %d: %s\n", s, turn, r.why.empty() ? "no tokens" : r.why.c_str());
                break;
            }
            prompt.insert(prompt.end(), answers.back().begin(), answers.back().end());
            const std::vector<int32_t> more = prompt_ids(200 + 10 * (uint64_t)s + (uint64_t)turn, 30);
            prompt.insert(prompt.end(), more.begin(), more.end());
        }
        answers.resize(3);
        return answers;
    };
    std::vector<std::vector<std::vector<int32_t>>> per(4);
    if (together) {
        std::vector<std::future<std::vector<std::vector<int32_t>>>> f;
        for (int s = 0; s < 4; ++s) f.push_back(std::async(std::launch::async, session, s));
        for (int s = 0; s < 4; ++s) per[(size_t)s] = f[(size_t)s].get();
    } else {
        for (int s = 0; s < 4; ++s) per[(size_t)s] = session(s);
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& p : per) for (auto& a : p) out.push_back(a);
    return out;
}
}  /* namespace */

TEST(conversations_whose_turns_grow_over_their_answers_serve_one_process_tokens) {
    REQUIRE(world().ok);
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> seqs = { "--max-num-seqs", "8" };
    std::map<int, std::vector<std::vector<int32_t>>> refs;
    std::map<int, double> hits;   /* a fresh one process's attention-cached prompt tokens over the sessions at once */
    for (int ds = 0; ds < 2; ++ds) {
        {
            Serving fresh;
            if (ds) fresh.model = worldds().llama_rad;
            REQUIRE(fresh.start("", ds ? cat(spec, seqs) : seqs));
            refs[ds] = conversations(fresh.port, true);
            hits[ds] = metric(get(fresh.port, "/metrics"), "radiance:attn_cached_tokens_total");
            kill(fresh.lead.child.pid, SIGTERM);
            CHECK_EQ(fresh.lead.finish(60), 0);
        }
        Serving one;
        if (ds) one.model = worldds().llama_rad;
        REQUIRE(one.start("", ds ? cat(spec, seqs) : seqs));
        const std::vector<std::vector<int32_t>> apart = conversations(one.port, false);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
        int empty = 0;
        for (const auto& a : refs[ds]) empty += a.empty();
        fprintf(stderr, "    one process (%s): 4 sessions x 3 turns, at once %s one after another (another start); %d empty; attention-cached "
                        "prompt tokens %.0f\n",
                ds ? "drafted" : "plain", apart == refs[ds] ? "=" : "!=", empty, hits[ds]);
        CHECK(apart == refs[ds]);
        CHECK_EQ(empty, 0);
        CHECK(hits[ds] > 0);
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, ds, kv; };
    for (const Arm a : { Arm{ "2 lanes", "0,2", 2, false, false, false }, Arm{ "3 lanes, wrap-around", "1,3", 3, false, false, true },
                         Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, false, true },
                         Arm{ "drafted, BEST's shape 4 lanes BG 2", "1,3", 4, true, true, true } }) {
        std::vector<std::string> env = kSplitEnv;
        if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        Serving pp;
        if (a.ds) pp.model = worldds().llama_rad;
        REQUIRE(pp.start(a.cut, cat(a.ds ? cat(spec, seqs) : seqs, { "--pp-lanes", std::to_string(a.lanes) }), env));
        const std::vector<std::vector<int32_t>> got = conversations(pp.port, true);
        const double hit = metric(get(pp.port, "/metrics"), "radiance:attn_cached_tokens_total");
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        const auto& ref = refs[a.ds ? 1 : 0];
        REQUIRE_EQ(got.size(), ref.size());
        int differ = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == ref[i]) continue;
            ++differ;
            size_t j = 0;
            while (j < got[i].size() && j < ref[i].size() && got[i][j] == ref[i][j]) ++j;
            fprintf(stderr, "    %s session %zu turn %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", a.name, i / 3, i % 3,
                    got[i].size(), ref[i].size(), j);
        }
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
        if (const char* dd = std::getenv("PP_E2E_PROD_LOGS")) {
            const std::string tag = std::string(dd) + "/conv-" + std::to_string(a.lanes) + "-" + a.cut + (a.ds ? "-ds" : "") + (a.best ? "-best" : "");
            tiny::write_text(tag + "-leader.log", llog);
            tiny::write_text(tag + "-follower.log", flog);
        }
        fprintf(stderr, "    CONV %-34s W(%s): %d of %zu turns differ from one process; attention-cached prompt tokens %.0f (one process %.0f); "
                        "KV-copy checks %lld, %lld differ; %zu E lines\n",
                a.name, a.cut, differ, got.size(), hit, hits[a.ds ? 1 : 0], checks, kdiff, elines);
        CHECK_EQ(differ, 0);
        CHECK_EQ(hit, hits[a.ds ? 1 : 0]);
        CHECK_EQ(elines, (size_t)0);
        if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
        if (differ || elines)
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
    }
}

/* tp8 loop D54c: SUBAGENTS -- requests that share a long COLD prefix (Claude Code's system prompt + tools) arriving at once, as a
 * session's parallel subagents do: six at once over a 700-byte common prefix nobody computed yet (the lanes may admit them side by
 * side, each prefilling the prefix or waiting on another's blocks), then four more 150 ms later while the first wave decodes (a
 * prefix hit over blocks that may still be in flight in another lane); greedy and seeded, every third streamed. Each request's
 * tokens = one process's for that request ALONE (and one process's at once = alone), at 2 lanes W(0,2), 3 lanes W(1,3), BEST's
 * shape (KV copy checked) and the drafted fixture at BEST's shape. */
namespace {
json subagent_body(int k) {
    std::vector<int32_t> p = prompt_ids(300, 700);
    const std::vector<int32_t> own = prompt_ids(310 + (uint64_t)k, 30 + 7 * (size_t)k);
    p.insert(p.end(), own.begin(), own.end());
    json b = { { "prompt", p }, { "n_predict", 24 }, { "ignore_eos", true }, { "return_tokens", true }, { "temperature", 0.0 } };
    if (k % 2) { b["temperature"] = 0.8; b["seed"] = 3000 + k; }
    if (k % 3 == 0) b["stream"] = true;
    return b;
}
std::vector<std::vector<int32_t>> subagents(int port, bool together) {
    std::vector<Reply> r(10);
    if (together) {
        std::vector<std::future<Reply>> f;
        for (int k = 0; k < 6; ++k) f.push_back(std::async(std::launch::async, [port, k] { return ask(port, subagent_body(k)); }));
        const char* dl = std::getenv("PP_E2E_SUB_DELAY_MS");   /* a debugging aid: the second wave's delay (default 150 ms) */
        std::this_thread::sleep_for(std::chrono::milliseconds(dl && *dl ? std::atoi(dl) : 150));
        for (int k = 6; k < 10; ++k) f.push_back(std::async(std::launch::async, [port, k] { return ask(port, subagent_body(k)); }));
        for (int k = 0; k < 10; ++k) r[(size_t)k] = f[(size_t)k].get();
    } else {
        for (int k = 0; k < 10; ++k) r[(size_t)k] = ask(port, subagent_body(k));
    }
    std::vector<std::vector<int32_t>> out;
    for (int k = 0; k < 10; ++k) {
        if (!r[(size_t)k].why.empty()) fprintf(stderr, "    subagent %d: %s\n", k, r[(size_t)k].why.c_str());
        out.push_back(r[(size_t)k].why.empty() && r[(size_t)k].tok.size() == 1 ? r[(size_t)k].tok[0] : std::vector<int32_t>{});
    }
    return out;
}
}  /* namespace */

TEST(subagents_sharing_a_cold_prefix_at_once_serve_one_process_tokens) {
    REQUIRE(world().ok);
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> seqs = { "--max-num-seqs", "12" };
    std::map<int, std::vector<std::vector<int32_t>>> refs;
    for (int ds = 0; ds < 2; ++ds) {
        std::vector<std::vector<int32_t>> at_once;
        double hit = -1;
        for (int pass = 0; pass < 2; ++pass) {   /* a fresh start each: alone, then at once */
            Serving one;
            if (ds) one.model = worldds().llama_rad;
            REQUIRE(one.start("", ds ? cat(spec, seqs) : seqs));
            if (pass == 0) refs[ds] = subagents(one.port, false);
            else {
                at_once = subagents(one.port, true);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* the idle publish of the metrics */
                hit = metric(get(one.port, "/metrics"), "radiance:attn_cached_tokens_total");
            }
            kill(one.lead.child.pid, SIGTERM);
            CHECK_EQ(one.lead.finish(60), 0);
        }
        int empty = 0;
        for (const auto& a : refs[ds]) empty += a.empty();
        fprintf(stderr, "    one process (%s): 10 subagents alone %s at once; %d empty; at once: attention-cached prompt tokens %.0f\n",
                ds ? "drafted" : "plain", at_once == refs[ds] ? "=" : "!=", empty, hit);
        CHECK(at_once == refs[ds]);
        CHECK_EQ(empty, 0);
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, ds, kv; };
    for (const Arm a : { Arm{ "1 lane", "0,2", 1, false, false, false }, Arm{ "2 lanes", "0,2", 2, false, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, false, true }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, false, true },
                         Arm{ "drafted, BEST's shape 4 lanes BG 2", "1,3", 4, true, true, true } }) {
        if (const char* only = std::getenv("PP_E2E_SUB_ARM"); only && !std::strstr(a.name, only)) continue;   /* a debugging aid */
        std::vector<std::string> env = kSplitEnv;
        if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        if (const char* xe = std::getenv("PP_E2E_SUB_ENV"); xe && *xe) {   /* a debugging aid: more env, space-separated */
            std::istringstream is(xe);
            for (std::string w; is >> w;) env.push_back(w);
        }
        Serving pp;
        if (a.ds) pp.model = worldds().llama_rad;
        REQUIRE(pp.start(a.cut, cat(a.ds ? cat(spec, seqs) : seqs, { "--pp-lanes", std::to_string(a.lanes) }), env));
        const std::vector<std::vector<int32_t>> got = subagents(pp.port, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* every lane's idle publish of its metrics */
        const double hit = metric(get(pp.port, "/metrics"), "radiance:attn_cached_tokens_total");
        kill(pp.lead.child.pid, SIGTERM);
        CHECK_EQ(pp.lead.finish(60), 0);
        CHECK_EQ(pp.fol.finish(30), 0);
        const std::string llog = pp.lead.output(), flog = pp.fol.output();
        const auto& ref = refs[a.ds ? 1 : 0];
        REQUIRE_EQ(got.size(), ref.size());
        int differ = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == ref[i]) continue;
            ++differ;
            size_t j = 0;
            while (j < got[i].size() && j < ref[i].size() && got[i][j] == ref[i][j]) ++j;
            fprintf(stderr, "    %s subagent %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", a.name, i, got[i].size(),
                    ref[i].size(), j);
        }
        long long checks = -1, kdiff = -1;
        if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
            if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
        const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
        if (const char* dd = std::getenv("PP_E2E_PROD_LOGS")) {
            const std::string tag = std::string(dd) + "/sub-" + std::to_string(a.lanes) + "-" + a.cut + (a.ds ? "-ds" : "") + (a.best ? "-best" : "");
            tiny::write_text(tag + "-leader.log", llog);
            tiny::write_text(tag + "-follower.log", flog);
        }
        fprintf(stderr, "    SUB %-34s W(%s): %d of %zu subagents differ from one process alone; attention-cached prompt tokens %.0f; KV-copy "
                        "checks %lld, %lld differ; %zu E lines\n",
                a.name, a.cut, differ, got.size(), hit, checks, kdiff, elines);
        CHECK_EQ(differ, 0);
        CHECK_EQ(elines, (size_t)0);
        if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
        if (differ || elines)
            fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
    }
}

/* tp8 loop D55 (RADIANCE_SCHED_PUB_PROMPT=1): a prompt published into the prefix cache when its last chunk commits, not only
 * at its close. A long streamed request A over a 500-byte head; once its first token is out (its prompt computed, 400 tokens
 * still to decode) three siblings over the same head arrive at once. Off (today): A is running, so its blocks are not in the
 * cache and every sibling computes the head again (0 attention-cached tokens); on: every sibling hits A's head. Each request's
 * tokens = one process's for it alone, on and off, in one process, at 2 lanes W(0,2), 3 lanes W(1,3) and BEST's shape (the
 * llama KV copy checked: a sibling's pass reads blocks A's passes wrote on the follower) and the drafted fixture at BEST's
 * shape; both nodes exit 0, no E line. */
namespace {
json sibling_body(int k) {
    std::vector<int32_t> p = prompt_ids(400, 500);
    const std::vector<int32_t> own = prompt_ids(410 + (uint64_t)k, 30 + 11 * (size_t)k);
    p.insert(p.end(), own.begin(), own.end());
    json b = { { "prompt", p }, { "n_predict", k == 0 ? 400 : 24 }, { "ignore_eos", true }, { "return_tokens", true },
               { "temperature", 0.0 } };
    if (k == 2) { b["temperature"] = 0.8; b["seed"] = 4002; }
    if (k == 0) b["stream"] = true;
    return b;
}
/* A streamed and read up to its first event, B / C / D at once, then the rest of A; the tokens in request order. */
std::vector<std::vector<int32_t>> siblings(int port, bool alone) {
    std::vector<std::vector<int32_t>> out(4);
    if (alone) {
        for (int k = 0; k < 4; ++k) {
            const Reply r = ask(port, sibling_body(k));
            if (!r.why.empty()) fprintf(stderr, "    sibling %d alone: %s\n", k, r.why.c_str());
            if (r.tok.size() == 1) out[(size_t)k] = r.tok[0];
        }
        return out;
    }
    Call a;
    if (!a.send(port, "/completion", sibling_body(0).dump())) return out;
    std::string raw;
    char buf[16384];
    for (;;) {   /* the first event: the prompt is computed (its first token is out) */
        const ssize_t n = ::recv(a.fd, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
        const size_t h = raw.find("\r\n\r\n");
        if (h != std::string::npos && raw.find("data: ", h) != std::string::npos) break;
    }
    std::vector<std::future<Reply>> f;
    for (int k = 1; k < 4; ++k) f.push_back(std::async(std::launch::async, [port, k] { return ask(port, sibling_body(k)); }));
    for (int k = 1; k < 4; ++k) {
        const Reply r = f[(size_t)k - 1].get();
        if (!r.why.empty()) fprintf(stderr, "    sibling %d: %s\n", k, r.why.c_str());
        if (r.tok.size() == 1) out[(size_t)k] = r.tok[0];
    }
    for (;;) {
        const ssize_t n = ::recv(a.fd, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
    }
    /* de-chunk and assemble A's events as ask() does */
    const size_t h = raw.find("\r\n\r\n");
    if (h == std::string::npos) return out;
    std::string b = raw.substr(h + 4), d;
    for (size_t at = 0;;) {
        const size_t eol = b.find("\r\n", at);
        if (eol == std::string::npos) break;
        const size_t len = std::strtoul(b.c_str() + at, nullptr, 16);
        if (len == 0 || eol + 2 + len > b.size()) break;
        d += b.substr(eol + 2, len);
        at = eol + 2 + len + 2;
    }
    for (size_t at = 0; (at = d.find("data: ", at)) != std::string::npos;) {
        const size_t e = d.find("\n", at);
        const json j = json::parse(d.substr(at + 6, e == std::string::npos ? std::string::npos : e - at - 6), nullptr, false);
        at = e == std::string::npos ? d.size() : e;
        if (j.is_object() && !j.value("stop", false))
            for (int32_t t : j.value("tokens", std::vector<int32_t>{})) out[0].push_back(t);
    }
    return out;
}
}  /* namespace */

TEST(d55_a_prompt_published_at_its_last_chunk_serves_its_siblings_one_process_tokens) {
    REQUIRE(world().ok);
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> seqs = { "--max-num-seqs", "8" };
    std::map<int, std::vector<std::vector<int32_t>>> refs;
    for (int ds = 0; ds < 2; ++ds) {
        Serving one;
        if (ds) one.model = worldds().llama_rad;
        REQUIRE(one.start("", ds ? cat(spec, seqs) : seqs));
        refs[ds] = siblings(one.port, true);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
        for (const auto& t : refs[ds]) CHECK(!t.empty());
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, ds, kv; };
    int armed = 0;
    for (const Arm a : { Arm{ "one process", "", 0, false, false, false }, Arm{ "2 lanes", "0,2", 2, false, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, false, true }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, false, true },
                         Arm{ "drafted, one process", "", 0, false, true, false },
                         Arm{ "drafted, BEST's shape 4 lanes BG 2", "1,3", 4, true, true, true } }) {
        for (int on = 0; on < 2; ++on) {
            std::vector<std::string> env = a.lanes ? kSplitEnv : std::vector<std::string>{};
            if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
            if (a.best) env = cat(env, kBest);
            if (on) env.push_back("RADIANCE_SCHED_PUB_PROMPT=1");
            Serving pp;
            if (a.ds) pp.model = worldds().llama_rad;
            std::vector<std::string> ex = a.ds ? cat(spec, seqs) : seqs;
            if (a.lanes) ex = cat(ex, { "--pp-lanes", std::to_string(a.lanes) });
            REQUIRE(pp.start(a.cut, ex, env));
            /* twice, the prefix cache reset in between: the second set runs in recycled request slots (SchedReq), which must
             * publish again (a slot's flag from its last tenant once kept them from it) */
            const std::vector<std::vector<int32_t>> got = siblings(pp.port, false);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* every lane's idle publish of its metrics */
            const double hit1 = metric(get(pp.port, "/metrics"), "radiance:attn_cached_tokens_total");
            {
                Call rs;
                std::string rb;
                CHECK(rs.send(pp.port, "/reset_prefix_cache", "") && rs.read(&rb) == 200);
            }
            const std::vector<std::vector<int32_t>> again = siblings(pp.port, false);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const double hit = metric(get(pp.port, "/metrics"), "radiance:attn_cached_tokens_total") - hit1;
            CHECK(again == got);
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            if (a.lanes) CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output(), flog = pp.fol.output();
            const auto& ref = refs[a.ds ? 1 : 0];
            int differ = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (got[i] == ref[i]) continue;
                ++differ;
                size_t j = 0;
                while (j < got[i].size() && j < ref[i].size() && got[i][j] == ref[i][j]) ++j;
                fprintf(stderr, "    %s D55 %s sibling %zu: %zu tokens vs one process alone %zu, the first difference at token %zu\n", a.name,
                        on ? "on" : "off", i, got[i].size(), ref[i].size(), j);
            }
            long long checks = -1, kdiff = -1;
            if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
                if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                    (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
            const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
            const bool said = llog.find("RADIANCE_SCHED_PUB_PROMPT=1 (tp8 loop D55)") != std::string::npos;
            fprintf(stderr, "    D55 %-34s %s: %d of 4 differ from one process alone (the second set %s); attention-cached prompt tokens %.0f / "
                            "%.0f (first / second set); KV-copy checks %lld, %lld differ; %zu E lines%s\n",
                    a.name, on ? "on " : "off", differ, again == got ? "the same" : "DIFFERENT", hit1, hit, checks, kdiff, elines,
                    on && !said ? "; the start line is MISSING" : "");
            CHECK_EQ(differ, 0);
            CHECK_EQ(elines, (size_t)0);
            CHECK_EQ(said, on == 1);
            if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
            if (on) { CHECK(hit1 >= 3 * 448); CHECK(hit >= 3 * 448); }   /* each sibling at least the head's whole 64-token chunks below 500 */
            else { CHECK_EQ(hit1, 0.0); CHECK_EQ(hit, 0.0); }
            if (on && hit1 > 0 && hit > 0) ++armed;
            if (differ || elines)
                fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        }
    }
    CHECK_EQ(armed, 6);
}

/* tp8 loop D55b: siblings that arrive at the SAME moment. Four prompts over one 500-byte head sent at once (no wait for a first
 * token): every lane admits its own before any prompt completes, so D55 alone cannot help (the head is computed four times);
 * RADIANCE_SCHED_PUB_HOLD_MS holds a fresh prompt whose head a running prompt is computing, and RADIANCE_SCHED_PUB_PROMPT=2
 * publishes that prompt chunk by chunk. Each request's tokens = one process's for it alone in every arm; cached tokens: off and
 * D55 alone ~0 at 2 / 3 / 4 lanes, with the hold 3 x the head's blocks (1,488); a hold never past its bound (a 1 ms bound is
 * printed: no request waits for it). */
namespace {
std::vector<std::vector<int32_t>> four_at_once(int port) {
    std::vector<std::future<Reply>> f;
    for (int k = 0; k < 4; ++k)
        f.push_back(std::async(std::launch::async, [port, k] {
            json b = sibling_body(k);
            b["n_predict"] = 32;
            b.erase("stream");
            return ask(port, b);
        }));
    std::vector<std::vector<int32_t>> out;
    for (int k = 0; k < 4; ++k) {
        const Reply r = f[(size_t)k].get();
        if (!r.why.empty()) fprintf(stderr, "    at once %d: %s\n", k, r.why.c_str());
        out.push_back(r.tok.size() == 1 ? r.tok[0] : std::vector<int32_t>{});
    }
    return out;
}
}  /* namespace */

TEST(d55b_siblings_at_the_same_moment_wait_for_the_head_and_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> seqs = { "--max-num-seqs", "8" };
    std::vector<std::vector<int32_t>> ref;
    {
        Serving one;
        REQUIRE(one.start("", seqs));
        for (int k = 0; k < 4; ++k) {
            json b = sibling_body(k);
            b["n_predict"] = 32;
            b.erase("stream");
            const Reply r = ask(one.port, b);
            ref.push_back(r.tok.size() == 1 ? r.tok[0] : std::vector<int32_t>{});
            CHECK(!ref.back().empty());
        }
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
    }
    struct Form { const char* name; std::vector<std::string> env; bool hits; };
    const std::vector<Form> forms = { { "off", {}, false }, { "D55 (=1)", { "RADIANCE_SCHED_PUB_PROMPT=1" }, false },
                                      { "=1 + hold 5000 ms", { "RADIANCE_SCHED_PUB_PROMPT=1", "RADIANCE_SCHED_PUB_HOLD_MS=5000" }, true },
                                      { "=2 + hold 5000 ms", { "RADIANCE_SCHED_PUB_PROMPT=2", "RADIANCE_SCHED_PUB_HOLD_MS=5000" }, true },
                                      { "=2 + hold 1 ms (bound)", { "RADIANCE_SCHED_PUB_PROMPT=2", "RADIANCE_SCHED_PUB_HOLD_MS=1" }, false } };
    struct Arm { const char* name; const char* cut; int lanes; bool best, kv; };
    for (const Arm a : { Arm{ "one process", "", 0, false, false }, Arm{ "2 lanes", "0,2", 2, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, true }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, true } }) {
        for (const Form& fm : forms) {
            std::vector<std::string> env = a.lanes ? kSplitEnv : std::vector<std::string>{};
            if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
            if (a.best) env = cat(env, kBest);
            env = cat(env, fm.env);
            Serving pp;
            std::vector<std::string> ex = seqs;
            if (a.lanes) ex = cat(ex, { "--pp-lanes", std::to_string(a.lanes) });
            REQUIRE(pp.start(a.cut, ex, env));
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<std::vector<int32_t>> got = four_at_once(pp.port);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* every lane's idle publish of its metrics */
            const double hit = metric(get(pp.port, "/metrics"), "radiance:attn_cached_tokens_total");
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            if (a.lanes) CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output(), flog = pp.fol.output();
            int differ = 0;
            for (size_t i = 0; i < got.size(); ++i) differ += got[i] != ref[i];
            long long checks = -1, kdiff = -1;
            if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
                if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                    (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
            const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
            fprintf(stderr, "    D55b %-28s %-24s: %d of 4 differ from one process alone; attention-cached prompt tokens %.0f; wall %.2f s; KV-copy "
                            "checks %lld, %lld differ; %zu E lines\n",
                    a.name, fm.name, differ, hit, wall, checks, kdiff, elines);
            CHECK_EQ(differ, 0);
            CHECK_EQ(elines, (size_t)0);
            if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
            if (fm.hits) CHECK(hit >= 3 * 448);
            if (differ || elines)
                fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
        }
    }
}

/* tp8 loop D56: LONG PROMPTS OVER THE POOL'S FREE SHARE BESIDE DECODERS, through the pipeline. Production (Claude Code sessions
 * of 100k+ tokens, 32 of them over the pair's 4.3 M-token pool) brings a long prompt while running sequences hold most of the
 * pool: admission must wait or preempt, and every lane schedules over ONE shared pool while each lane's admission rule
 * (fresh_prefill_can_fit / preempted_can_prefill) counts its own running sequences. Four decoders (prompts 200-290 tokens,
 * 360-390 to decode) at once, then -- while they decode -- three long prompts at once (two siblings over one 900-token head,
 * one alone, 1,160-1,500 tokens), then two short ones alone; a pool found as the preemption case's (the smallest device size
 * whose pool holds >= kLongLo tokens: each request alone fits, all at once ~6,900 do not). Every request's greedy tokens = one
 * process's on the default pool, in one process at the small pool, at 2 lanes W(0,2), 3 lanes W(1,3) and BEST's shape (the
 * llama KV copy checked), each with and without START F's publishing envs (RADIANCE_SCHED_PUB_PROMPT=2 +
 * RADIANCE_SCHED_PUB_HOLD_MS=5000: a held sibling and chunk-by-chunk entries under KV pressure); every request answered (200),
 * within kLongWallS; both nodes exit 0, no E line, the KV-copy check 0 differ. Printed per arm: pool, preemptions, wall,
 * attention-cached tokens -- the lanes' preemption count against one process's at its small pool is the thrash reading. */
namespace {
constexpr long long kLongLo = 2000, kLongHi = 4200;
constexpr double kLongWallS = 240;
std::vector<std::string> long_shape(int lanes) {
    std::vector<std::string> a = { "--max-num-seqs", "8", "--max-model-len", "2048" };
    if (lanes > 0) a = cat(a, { "--pp-lanes", std::to_string(lanes) });
    return a;
}
std::vector<int32_t> long_prompt(int k) {
    if (k == 2) return prompt_ids(560, 1500);
    std::vector<int32_t> p = prompt_ids(550, 900);
    const std::vector<int32_t> own = prompt_ids(551 + (uint64_t)k, 260 + 140 * (size_t)k);
    p.insert(p.end(), own.begin(), own.end());
    return p;
}
struct LongRun { std::vector<std::vector<int32_t>> tok; std::vector<int> status; double wall = 0; };
LongRun long_set(int port) {
    LongRun out;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::future<std::pair<int, std::vector<int32_t>>>> f;
    for (uint64_t k = 0; k < 4; ++k)
        f.push_back(std::async(std::launch::async, [port, k] {
            int st = -1;
            std::vector<int32_t> t = complete(port, prompt_ids(501 + k, 200 + 30 * k), 360 + 10 * (int)k, &st);
            return std::make_pair(st, t);
        }));
    std::this_thread::sleep_for(std::chrono::milliseconds(400));   /* the decoders' prompts are in; they decode */
    for (int k = 0; k < 3; ++k)
        f.push_back(std::async(std::launch::async, [port, k] {
            int st = -1;
            std::vector<int32_t> t = complete(port, long_prompt(k), 24 + 4 * k, &st);
            return std::make_pair(st, t);
        }));
    for (auto& x : f) {
        auto r = x.get();
        out.status.push_back(r.first);
        out.tok.push_back(std::move(r.second));
    }
    for (uint64_t s : { 571, 572 }) {
        int st = -1;
        out.tok.push_back(complete(port, prompt_ids(s, 120), 30, &st));
        out.status.push_back(st);
    }
    out.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}
}  /* namespace */

TEST(d56_long_prompts_over_the_pools_free_share_beside_decoders_serve_one_process_tokens) {
    REQUIRE(world().ok);
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    const std::vector<std::string> kPub = { "RADIANCE_SCHED_PUB_PROMPT=2", "RADIANCE_SCHED_PUB_HOLD_MS=5000" };
    LongRun ref;
    {
        Serving one;
        REQUIRE(one.start("", long_shape(0)));
        ref = long_set(one.port);
        const double pre = metric(get(one.port, "/metrics"), "vllm:num_preemptions_total");
        fprintf(stderr, "    D56 reference: one process, pool %lld tokens, %.0f preemption(s), wall %.2f s\n", pool_in(one.lead.output()), pre,
                ref.wall);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
        for (size_t i = 0; i < ref.tok.size(); ++i) { CHECK_EQ(ref.status[i], 200); CHECK(!ref.tok[i].empty()); }
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, kv; };
    const char* only = std::getenv("PP_E2E_D56_ARM");   /* a debugging aid: the arms whose name holds this text */
    long long pre_all = 0, pre_form[5] = { 0, 0, 0, 0, 0 };
    for (const Arm a : { Arm{ "one process", "", 0, false, false }, Arm{ "2 lanes", "0,2", 2, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, true }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, true } }) {
        if (only && !std::strstr(a.name, only)) continue;
        std::vector<std::string> env = a.lanes ? kSplitEnv : std::vector<std::string>{};
        if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        const std::vector<std::string> ex = long_shape(a.lanes);
        auto pool_at = [&](int mib) -> long long {
            Serving pr;
            if (!pr.start(a.cut, ex, cat(env, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(mib) }))) return -1;
            const long long p = pool_in(pr.lead.output());
            kill(pr.lead.child.pid, SIGTERM);
            pr.lead.finish(60);
            if (a.lanes) pr.fol.finish(30);
            return p;
        };
        int lo = 64, hi = 512;
        long long plo = pool_at(lo), phi = pool_at(hi);
        if (!(plo < kLongLo && phi >= kLongLo))
            fprintf(stderr, "    D56 %s: pools %lld / %lld tokens at %d / %d MiB do not bracket %lld\n", a.name, plo, phi, lo, hi, kLongLo);
        REQUIRE(plo < kLongLo && phi >= kLongLo);
        while (hi - lo > 1) {
            const int m = (lo + hi) / 2;
            const long long p = pool_at(m);
            if (p >= kLongLo) { hi = m; phi = p; } else { lo = m; plo = p; }
        }
        if (phi >= kLongHi) fprintf(stderr, "    D56 %s: the smallest pool >= %lld is %lld tokens (%d MiB)\n", a.name, kLongLo, phi, hi);
        REQUIRE(phi < kLongHi);
        /* form 4 (lane arms): fn4 sched_lane_room 1 POSTed in process on a start without the env (the card A/B's form) */
        for (int form = 0; form < (a.lanes ? 5 : 4); ++form) {
            const bool posted = form == 4, pub = (form & 1) && !posted, room = (form & 2) && !posted;
            std::vector<std::string> fenv = env;
            if (pub) fenv = cat(fenv, kPub);
            if (room) fenv.push_back("RADIANCE_SCHED_LANE_ROOM=1");
            const char* fname = form == 0 ? "default  " : form == 1 ? "pub2+hold" : form == 2 ? "lane-room" : form == 3 ? "room+pub " : "posted 1 ";
            Serving pp;
            REQUIRE(pp.start(a.cut, ex, cat(fenv, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(hi) })));
            if (posted) {
                std::string out;
                Call c;
                const int st = c.send(pp.port, "/admin/fn4", json{ { "sched_lane_room", 1 } }.dump()) ? c.read(&out) : -1;
                CHECK_EQ(st, 200);
                if (st != 200) fprintf(stderr, "    POST sched_lane_room 1: %d %s\n", st, out.c_str());
                CHECK(!complete(pp.port, prompt_ids(7, 12), 4).empty());   /* applied at the next step's release */
                CHECK(get(pp.port, "/admin/fn4").find("\"sched_lane_room\": 1") != std::string::npos);
            }
            const LongRun got = long_set(pp.port);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));   /* every lane's idle publish of its metrics */
            const std::string mt = get(pp.port, "/metrics");
            const double pre = metric(mt, "vllm:num_preemptions_total"), hit = metric(mt, "radiance:attn_cached_tokens_total");
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            if (a.lanes) CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output(), flog = pp.fol.output();
            REQUIRE_EQ(got.tok.size(), ref.tok.size());
            int differ = 0, failed = 0;
            for (size_t i = 0; i < ref.tok.size(); ++i) {
                if (got.status[i] != 200 || got.tok[i].empty()) {
                    ++failed;
                    fprintf(stderr, "    D56 %s %s request %zu: HTTP %d, %zu tokens\n", a.name, fname, i, got.status[i],
                            got.tok[i].size());
                    continue;
                }
                if (got.tok[i] == ref.tok[i]) continue;
                ++differ;
                size_t j = 0;
                while (j < ref.tok[i].size() && j < got.tok[i].size() && ref.tok[i][j] == got.tok[i][j]) ++j;
                fprintf(stderr, "    D56 %s %s request %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n", a.name,
                        fname, i, got.tok[i].size(), ref.tok[i].size(), j);
            }
            long long checks = -1, kdiff = -1;
            if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
                if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                    (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
            const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
            if (const char* dd = std::getenv("PP_E2E_D56_LOGS")) {
                const std::string base = std::string(dd) + "/" + std::to_string(a.lanes) + "-f" + std::to_string(form);
                tiny::write_text(base + "-leader.log", llog);
                if (a.lanes) tiny::write_text(base + "-follower.log", flog);
            }
            fprintf(stderr, "    D56 %-26s %s: pool %lld tokens (%d MiB), %.0f preemption(s), wall %.2f s, attention-cached %.0f; %d of %zu "
                            "differ, %d failed; KV-copy checks %lld, %lld differ; %zu E lines\n",
                    a.name, fname, pool_in(llog), hi, pre, got.wall, hit, differ, ref.tok.size(), failed, checks, kdiff, elines);
            const bool said = llog.find("RADIANCE_SCHED_LANE_ROOM=1 (tp8 loop D56)") != std::string::npos;
            CHECK_EQ(said, room);
            /* the POSTed switch names itself once, with the preemptions before it (an in-process A/B's per-arm reading) */
            CHECK_EQ(llog.find("fn4 sched_lane_room 1 from here (was 0; tp8 loop D56)") != std::string::npos, posted);
            pre_form[form] += pre > 0 && a.lanes ? (long long)pre : 0;
            CHECK_EQ(differ, 0);
            CHECK_EQ(failed, 0);
            CHECK_EQ(elines, (size_t)0);
            if (form == 0) CHECK(pre > 0);   /* the set reaches the pool (the other forms may avoid it) */
            CHECK(got.wall < kLongWallS);
            if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
            if (differ || failed || elines)
                fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
            pre_all += pre > 0 ? (long long)pre : 0;
        }
    }
    fprintf(stderr, "    D56 preemptions, all arms: %lld; the lane arms by form: default %lld, pub2+hold %lld, lane-room %lld, room+pub %lld, "
                    "posted 1 %lld\n", pre_all, pre_form[0], pre_form[1], pre_form[2], pre_form[3], pre_form[4]);
    /* the lever's own quantity, over the three lane arms: four runs read default + pub 13 / 19 / 27 vs lane-room + room+pub 6 / 8 / 7 */
    if (!only) {
        CHECK(pre_form[2] + pre_form[3] < pre_form[0] + pre_form[1]);
        CHECK(pre_form[4] < pre_form[0]);   /* the switch POSTed in process acts as the start env does */
    }
}

/* tp8 loop D56b: the same long-prompt set under KV pressure on the DSpark-DRAFTED fixture (D38a's WorldDs, 3 speculative tokens):
 * BEST drafts, and a preemption by recompute through the lanes then also drops and rebuilds a drafter's state (its KV group is
 * the same pool's) while other lanes' drafted sequences run. Every request's greedy tokens = the drafted one process's on the
 * default pool (= the plain tokens: drafts are verified), in one process at the small pool, 2 lanes W(0,2), 3 lanes W(1,3) and
 * BEST's shape (the KV copy checked), each with RADIANCE_SCHED_LANE_ROOM unset and =1; drafts proposed and accepted in every arm;
 * all answered, KV-copy 0 differ, 0 E lines. */
TEST(d56b_long_prompts_over_the_pool_with_a_drafter_serve_one_process_tokens) {
    REQUIRE(worldds().ok);
    const std::vector<std::string> spec = { "--num-speculative-tokens", "3" };
    const std::vector<std::string> kBest = { "RADIANCE_PP_BG_LANE=2", "RADIANCE_PP_READY_FIRST=1", "RADIANCE_PP_FOL_ORDER=2",
                                             "RADIANCE_PP_SEG_AHEAD=0", "RADIANCE_PP_SEG_LAYERS=1", "RADIANCE_PP_SEG_ROWS=16" };
    LongRun ref;
    {
        Serving one;
        one.model = worldds().llama_rad;
        REQUIRE(one.start("", cat(spec, long_shape(0))));
        ref = long_set(one.port);
        fprintf(stderr, "    D56b reference: drafted one process, pool %lld tokens, wall %.2f s\n", pool_in(one.lead.output()), ref.wall);
        kill(one.lead.child.pid, SIGTERM);
        CHECK_EQ(one.lead.finish(60), 0);
        for (size_t i = 0; i < ref.tok.size(); ++i) { CHECK_EQ(ref.status[i], 200); CHECK(!ref.tok[i].empty()); }
    }
    struct Arm { const char* name; const char* cut; int lanes; bool best, kv; };
    const char* only = std::getenv("PP_E2E_D56_ARM");
    long long pre_form[2] = { 0, 0 };
    for (const Arm a : { Arm{ "one process", "", 0, false, false }, Arm{ "2 lanes", "0,2", 2, false, false },
                         Arm{ "3 lanes, wrap-around", "1,3", 3, false, true }, Arm{ "BEST's shape 4 lanes BG 2", "1,3", 4, true, true } }) {
        if (only && !std::strstr(a.name, only)) continue;
        std::vector<std::string> env = a.lanes ? kSplitEnv : std::vector<std::string>{};
        if (a.kv) env = cat(env, { "RADIANCE_LLAMA_PP_KVCOPY=1", "RADIANCE_PP_KV_CHECK=1" });
        if (a.best) env = cat(env, kBest);
        const std::vector<std::string> ex = cat(spec, long_shape(a.lanes));
        auto pool_at = [&](int mib) -> long long {
            Serving pr;
            pr.model = worldds().llama_rad;
            if (!pr.start(a.cut, ex, cat(env, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(mib) }))) return -1;
            const long long p = pool_in(pr.lead.output());
            kill(pr.lead.child.pid, SIGTERM);
            pr.lead.finish(60);
            if (a.lanes) pr.fol.finish(30);
            return p;
        };
        int lo = 64, hi = 512;
        long long plo = pool_at(lo), phi = pool_at(hi);
        if (!(plo < kLongLo && phi >= kLongLo))
            fprintf(stderr, "    D56b %s: pools %lld / %lld tokens at %d / %d MiB do not bracket %lld\n", a.name, plo, phi, lo, hi, kLongLo);
        REQUIRE(plo < kLongLo && phi >= kLongLo);
        while (hi - lo > 1) {
            const int m = (lo + hi) / 2;
            const long long p = pool_at(m);
            if (p >= kLongLo) { hi = m; phi = p; } else { lo = m; plo = p; }
        }
        if (phi >= kLongHi) fprintf(stderr, "    D56b %s: the smallest pool >= %lld is %lld tokens (%d MiB)\n", a.name, kLongLo, phi, hi);
        REQUIRE(phi < kLongHi);
        for (int room = 0; room < 2; ++room) {
            Serving pp;
            pp.model = worldds().llama_rad;
            std::vector<std::string> fenv = env;
            if (room) fenv.push_back("RADIANCE_SCHED_LANE_ROOM=1");
            REQUIRE(pp.start(a.cut, ex, cat(fenv, { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(hi) })));
            const LongRun got = long_set(pp.port);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const std::string mt = get(pp.port, "/metrics");
            const double pre = metric(mt, "vllm:num_preemptions_total"), dr = metric(mt, "radiance:draft_tokens_total"),
                         acc = metric(mt, "radiance:draft_accepted_total");
            kill(pp.lead.child.pid, SIGTERM);
            CHECK_EQ(pp.lead.finish(60), 0);
            if (a.lanes) CHECK_EQ(pp.fol.finish(30), 0);
            const std::string llog = pp.lead.output(), flog = pp.fol.output();
            REQUIRE_EQ(got.tok.size(), ref.tok.size());
            int differ = 0, failed = 0;
            for (size_t i = 0; i < ref.tok.size(); ++i) {
                if (got.status[i] != 200 || got.tok[i].empty()) { ++failed; continue; }
                if (got.tok[i] == ref.tok[i]) continue;
                ++differ;
                size_t j = 0;
                while (j < ref.tok[i].size() && j < got.tok[i].size() && ref.tok[i][j] == got.tok[i][j]) ++j;
                fprintf(stderr, "    D56b %s room %d request %zu: %zu tokens vs one process's %zu, the first difference at token %zu\n",
                        a.name, room, i, got.tok[i].size(), ref.tok[i].size(), j);
            }
            long long checks = -1, kdiff = -1;
            if (const size_t at = flog.rfind(" checks, "); at != std::string::npos)
                if (const size_t b = flog.rfind("(", at); b != std::string::npos)
                    (void)std::sscanf(flog.c_str() + b, "(%lld checks, %lld differ)", &checks, &kdiff);
            const size_t elines = count_of(llog, "\nE ") + count_of(flog, "\nE ");
            fprintf(stderr, "    D56b %-26s %s: pool %lld tokens (%d MiB), %.0f preemption(s), wall %.2f s, drafted %.0f / accepted %.0f; %d of "
                            "%zu differ, %d failed; KV-copy checks %lld, %lld differ; %zu E lines\n",
                    a.name, room ? "lane-room" : "default  ", pool_in(llog), hi, pre, got.wall, dr, acc, differ, ref.tok.size(), failed,
                    checks, kdiff, elines);
            CHECK_EQ(differ, 0);
            CHECK_EQ(failed, 0);
            CHECK_EQ(elines, (size_t)0);
            CHECK(got.wall < kLongWallS);
            CHECK(dr > 0);
            CHECK(acc > 0);
            if (!room) CHECK(pre > 0);
            if (a.kv) { CHECK(checks > 0); CHECK_EQ(kdiff, 0); }
            if (differ || failed || elines)
                fprintf(stderr, "    leader:\n%s\n    follower:\n%s\n", tail(llog, 30).c_str(), tail(flog, 30).c_str());
            if (a.lanes) pre_form[room] += pre > 0 ? (long long)pre : 0;
        }
    }
    fprintf(stderr, "    D56b lane-arm preemptions: default %lld, lane-room %lld\n", pre_form[0], pre_form[1]);
    if (!only) CHECK(pre_form[1] < pre_form[0]);   /* the lever's own quantity: two runs read 10 / 12 vs 4 / 4 */
}
