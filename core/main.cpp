/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "hf_model.h"
#include "rad_internal.h"
#include "util/stallwatch.h"

#include <cstdio>
#include <cstdlib>
#include <string>

/* qfn hero image (Qwen loop round 26, COORD 10:35: an image whose DEFAULTS are the best start): the image's start
 * environment -- the file RADIANCE_IMAGE_DEFAULTS names, else $RADIANCE_HOME / RAD_DEFAULT_HOME + "/defaults.env"; no file =
 * nothing (today). Each `K=V` line (# comments and blank lines skipped) sets K only where the environment has it UNSET or
 * EMPTY, before anything reads it: a start env file's blank `K=` takes the image's value, while `K=0` or any other value the
 * caller gives stays. One line names what was set and what was kept. */
static void image_env_defaults() {
    std::string path;
    if (const char* e = std::getenv("RADIANCE_IMAGE_DEFAULTS")) {
        if (!*e) return;
        path = e;
    } else {
        const char* h = std::getenv("RADIANCE_HOME");
        path = std::string(h && *h ? h : RAD_DEFAULT_HOME) + "/defaults.env";
    }
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    std::string set, kept;
    int n_set = 0, n_kept = 0;
    char line[4096];
    while (std::fgets(line, sizeof line, f)) {
        std::string l(line);
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r' || l.back() == ' ' || l.back() == '\t')) l.pop_back();
        size_t b = l.find_first_not_of(" \t");
        if (b == std::string::npos || l[b] == '#') continue;
        l = l.substr(b);
        const size_t eq = l.find('=');
        if (eq == std::string::npos || eq == 0) continue;
        const std::string k = l.substr(0, eq), v = l.substr(eq + 1);
        const char* cur = std::getenv(k.c_str());
        if (cur && *cur) { kept += (n_kept++ ? " " : "") + k + "=" + cur; continue; }
        setenv(k.c_str(), v.c_str(), 1);
        set += (n_set++ ? " " : "") + k + "=" + v;
    }
    std::fclose(f);
    RAD_INFO("image defaults (%s): %d set where unset or empty [%s]; %d kept from the environment [%s]", path.c_str(), n_set,
             set.c_str(), n_kept, kept.c_str());
}

int main(int argc, char** argv) {
    image_env_defaults();
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }
    if (cfg.proceed_incompatible) rad::Engine::print_unsupported_banner();

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    rad::Engine engine(std::move(cfg));

    /* A startup that stops logging reports where it waits (util/stallwatch.h). Two minutes
     * comfortably covers the slowest normal step, such as a container read from a slow disk. The
     * watch goes on into serve() and ends when the server listens. */
    auto watch = std::make_unique<rad::StallWatch>("startup", 120000);
    s = engine.start();
    if (s < 0 || engine.config().debug_graph) watch.reset();
    if (s < 0) {
        /* start() logged the error; the report adds the machine and the decisions made so far. */
        engine.print_startup_report(s);
        return 1;
    }

    if (engine.config().debug_graph) return engine.dump();
    s = engine.serve(std::move(watch));
    if (s < 0) {
        /* serve() can also refuse before listening (e.g. a template default the template
         * rejects, an unholdable thinking budget). */
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        return 1;
    }
    return 0;
}
