/* force_al.h -- RADIANCE_DRAFT_FORCE_AL: a MEASUREMENT-ONLY acceptance for speculative decoding (Qwen loop round 46,
 * QUEUE Q88; the AgentX rules' synthetic acceptance). Never a served form and never a default: with it set, the text a
 * request receives is no longer the target model's.
 *
 * Why: AgentX (InferenceX's agentic benchmark) compares engines at ONE committed acceptance length per model, draft
 * length and thinking mode (the "golden AL" curves, SPEED-Bench coding), so the throughput measured is the system's and
 * not the draft head's. vLLM (synthetic_acceptance_length), SGLang (SGLANG_SIMULATE_ACC_LEN, real-draft-token) and
 * TensorRT-LLM (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS) accept a forced count; this is the same for Radiance.
 *
 * The rule (vLLM's minimum-variance schedule): a chain verify of `aux` drafts accepts m = AL - 1 drafts on average as
 * floor(m) every step and one more with probability frac(m), never more than `aux`. The accepted drafts are emitted as
 * drafted (SGLang's real-draft-token), then the target's own token after them (the bonus), so every emitted token's K/V,
 * recurrent state and conv window are the ones the verify already computed: the step does the same work as with
 * natural acceptance (the same rows, kernels and passes), only its count is set. The end-of-generation cut applies to
 * the EMITTED tokens (the drafts below the count, then the bonus), on the card and the host alike.
 *
 * The draw: a 32-bit hash of (the verify step's number, the sequence's StepSlot, a seed) -- integers the card's
 * acceptance (sched/advance_impl.h, ADV_ACCEPT) and the host's (Engine::collect_step) both hold, so the two agree on
 * every row (Engine::finish_*_drafter refuses a step where they do not). Deterministic for a seed; a fresh draw every
 * step and sequence, so the mean converges as a Bernoulli mean (no feedback through positions).
 *
 * AL by the row's own draft count: entry d of the table is the AL for a verify of d drafts (the golden curve's
 * num_speculative_tokens = d), so a step whose depth a load cap cut (draft_dcap) reads its own depth's AL. A row with
 * more drafts than the table holds reads its last entry. */
#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define RAD_FAL_FN __host__ __device__ inline
#else
#define RAD_FAL_FN inline
#endif

namespace rad {

/* Depths 0..16 (entry 0 unused: a row with no drafts accepts none). */
enum { kFalDepths = 17 };

/* One forced-acceptance setting, passed by value inside AdvanceArgs (80 bytes). `on` 0: the agreement rule (today). */
struct ForceAl {
    int32_t  on = 0;
    uint32_t seed = 0;
    uint32_t key = 0;                  /* the verify step's number (RadBatch::step), set per ADV_ACCEPT */
    uint32_t m[kFalDepths] = {};       /* AL - 1 for a verify of d drafts, fixed point 8.24 */
    uint32_t pad = 0;
};

RAD_FAL_FN uint32_t fal_mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* The forced number of accepted drafts for a chain verify of `aux` drafts of the sequence in StepSlot `slot` at step
 * `key`: floor(m) + (draw < frac(m)), at most `aux`. */
RAD_FAL_FN int32_t fal_count(const ForceAl& f, uint32_t key, int32_t slot, int32_t aux) {
    if (aux <= 0) return 0;
    const uint32_t m = f.m[aux < (int32_t)kFalDepths ? aux : (int32_t)kFalDepths - 1];
    const uint32_t u = fal_mix(key * 0x9e3779b9u ^ fal_mix((uint32_t)slot * 0x85ebca6bu + f.seed));
    const int32_t n = (int32_t)(m >> 24) + ((u >> 8) < (m & 0xffffffu) ? 1 : 0);
    return n < aux ? n : aux;
}

/* m = AL - 1 in fixed point 8.24, rounded; AL in [1, 65]. */
RAD_FAL_FN uint32_t fal_fixed(double al) {
    double m = al - 1.0;
    if (m < 0) m = 0;
    if (m > 64) m = 64;
    return (uint32_t)(m * 16777216.0 + 0.5);
}

}  /* namespace rad */

#if !defined(__HIPCC__)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace rad {

/* RADIANCE_DRAFT_FORCE_AL's value: one AL for every depth ("2.47"), or AL by the verify's draft count
 * ("1:1.65,2:2.06,3:2.32,4:2.47"; depth 1..16, AL in [1, depth + 1]). A depth not listed reads the largest listed
 * depth below it, or the smallest listed one where none is below. Returns false with `why` on a value it refuses;
 * `desc` gets the table as the start line prints it. */
inline bool fal_parse(const char* s, ForceAl* f, std::string* why, std::string* desc) {
    ForceAl out;
    double al[kFalDepths] = {};
    bool listed[kFalDepths] = {};
    if (!s || !*s) { *why = "empty"; return false; }
    const std::string v = s;
    if (v.find(':') == std::string::npos) {
        char* end = nullptr;
        const double a = std::strtod(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(a) || a < 1.0 || a > 65.0) {
            *why = "'" + v + "' is not an AL in [1, 65]";
            return false;
        }
        for (int d = 1; d < kFalDepths; ++d) { al[d] = a; listed[d] = true; }
    } else {
        size_t p = 0;
        while (p <= v.size()) {
            const size_t c = v.find(',', p);
            const std::string item = v.substr(p, c == std::string::npos ? std::string::npos : c - p);
            const size_t k = item.find(':');
            char* e1 = nullptr;
            char* e2 = nullptr;
            const long d = k == std::string::npos ? -1 : std::strtol(item.c_str(), &e1, 10);
            const double a = k == std::string::npos ? -1 : std::strtod(item.c_str() + k + 1, &e2);
            if (k == std::string::npos || e1 != item.c_str() + k || !e2 || *e2 || d < 1 || d >= kFalDepths ||
                !std::isfinite(a) || a < 1.0 || a > (double)d + 1.0) {
                *why = "'" + item + "' is not DEPTH:AL with DEPTH 1..16 and AL in [1, DEPTH + 1]";
                return false;
            }
            if (listed[d]) { *why = "depth " + std::to_string(d) + " is listed twice"; return false; }
            al[d] = a;
            listed[d] = true;
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }
    int lo = 0;
    for (int d = 1; d < kFalDepths && !lo; ++d) if (listed[d]) lo = d;
    if (!lo) { *why = "no depth listed"; return false; }
    double last = al[lo];
    std::string t;
    for (int d = 1; d < kFalDepths; ++d) {
        if (listed[d]) last = al[d];
        const double a = d < lo ? al[lo] : last;
        out.m[d] = fal_fixed(a);
        if (listed[d]) {
            char b[48];
            std::snprintf(b, sizeof b, "%s%d: %.4g", t.empty() ? "" : ", ", d, a);
            t += b;
        }
    }
    if (v.find(':') == std::string::npos) {
        char b[48];
        std::snprintf(b, sizeof b, "every depth: %.4g", al[1]);
        t = b;
    }
    out.on = 1;
    *f = out;
    if (desc) *desc = t;
    return true;
}

}  /* namespace rad */
#endif
