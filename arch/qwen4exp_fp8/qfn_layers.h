/* qfn_layers.h -- a node's trunk-layer ranges in qwen4exp_fp8 (Qwen loop round 29, COORD 15:05: Qwen tp8 = the two-node layer
 * pipeline; step S1 of FINDINGS DEV note 15:28), pure host arithmetic so arch_test checks it without a card.
 *
 * The DeepSeek line's ds41_layers.h form (548a2642) for Flash-Next's 48 trunk layers. A node's layers are ranges of trunk
 * layers, hi exclusive, ascending and disjoint:
 *
 *   RADIANCE_FN4_LAYERS=0:24            layers 0..23
 *   RADIANCE_FN4_LAYERS=24:48           layers 24..47
 *   RADIANCE_FN4_LAYERS=0:2,26:48       W(2, 26)'s leader (the wrap-around split: layer 0, the PLE layer, the tail)
 *   RADIANCE_FN4_LAYERS=2:26            W(2, 26)'s follower
 *
 * Unset, empty, or ranges that cover every layer: the published single-machine engine (Spec::on false).
 *
 * RADIANCE_FN4_LAYERS_MODE picks what the node declares besides its layers:
 *   proxy  (default)  the stage timing proxy (S1): the node runs alone, without the other machine; it keeps the embedding, the
 *                     PLE table, the final mixer, the MTP head and lm_head whatever its layers (the engine samples from its
 *                     logits). Its first layer, when that is not layer 0, reads the embedding's stream as layer 0 does (the
 *                     synthetic boundary input); a gap between two of its ranges passes the wide stream on. Answers are not
 *                     the model's; each held layer's work a step is. Any ranges run.
 *   strict            the node's own part of W(h, t) (S2): the leader holds [0, h) and [t, n_layer) -- layer 0, the PLE layer
 *                     (its token ids stay on the leader, so the crossing into the follower carries the stream only) and the
 *                     last layer (the final mixer, the MTP head and lm_head read it) -- the follower [h, t). Cross-layer reads
 *                     of Flash-Next's trunk besides the wide hc stream: none (every GDN / attention / QSA state is its own
 *                     layer's; the PLE layer reads token ids), so the two crossings are the stream at h and at t. */
#ifndef RAD_ARCH_QFN_LAYERS_H
#define RAD_ARCH_QFN_LAYERS_H

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace arch {
namespace qfnlayers {

struct Spec {
    bool on = false;                          /* ranges set that are not every layer */
    bool strict = false;                      /* RADIANCE_FN4_LAYERS_MODE=strict */
    std::vector<std::pair<int, int>> ranges;  /* [lo, hi), ascending, disjoint, adjacent ones merged */
    std::vector<char> own;                    /* a layer this node declares and runs */
    int first = -1, last = -1, n_own = 0;
    bool leader = false;                      /* holds the last trunk layer */
    std::string text;                         /* the ranges as parsed, e.g. "0:2,26:48" */
    bool held(int l) const { return !on || (l >= 0 && l < (int)own.size() && own[(size_t)l]); }
};

/* RADIANCE_FN4_LAYERS's value (`s`, null or empty: unset) and RADIANCE_FN4_LAYERS_MODE's (`mode`, null or empty: proxy) for
 * a trunk of n_layer layers. 0, or -1 with the reason in *err. */
inline int parse(const char* s, const char* mode, int n_layer, Spec* out, std::string* err) {
    *out = Spec{};
    if (mode && *mode && std::string(mode) != "proxy" && std::string(mode) != "strict") {
        *err = "RADIANCE_FN4_LAYERS_MODE=" + std::string(mode) + " is not proxy or strict";
        return -1;
    }
    out->strict = mode && std::string(mode) == "strict";
    if (!s || !*s) return 0;
    if (n_layer <= 0) {
        *err = "the model has no trunk layers";
        return -1;
    }
    std::vector<std::pair<int, int>> rs;
    const char* p = s;
    while (*p) {
        char* e = nullptr;
        if (*p < '0' || *p > '9') break;
        const long lo = std::strtol(p, &e, 10);
        if (e == p || *e != ':') break;
        p = e + 1;
        if (*p < '0' || *p > '9') break;
        const long hi = std::strtol(p, &e, 10);
        if (e == p) break;
        p = e;
        if (lo < 0 || hi > n_layer || lo >= hi) {
            *err = "RADIANCE_FN4_LAYERS=" + std::string(s) + ": the range " + std::to_string(lo) + ":" + std::to_string(hi) +
                   " is not lo:hi with 0 <= lo < hi <= " + std::to_string(n_layer);
            return -1;
        }
        if (!rs.empty() && lo < rs.back().second) {
            *err = "RADIANCE_FN4_LAYERS=" + std::string(s) + ": the ranges are not ascending and disjoint";
            return -1;
        }
        if (!rs.empty() && lo == rs.back().second) rs.back().second = (int)hi;
        else rs.emplace_back((int)lo, (int)hi);
        if (*p == ',') {
            ++p;
            if (!*p) break;   /* a trailing comma: malformed */
            continue;
        }
        if (*p) break;
        p = nullptr;
        break;
    }
    if (p != nullptr || rs.empty()) {
        *err = "RADIANCE_FN4_LAYERS=" + std::string(s) + " is not lo:hi[,lo:hi...] (trunk layers, hi exclusive)";
        return -1;
    }
    Spec& o = *out;
    o.own.assign((size_t)n_layer, 0);
    for (const auto& r : rs)
        for (int l = r.first; l < r.second; ++l) o.own[(size_t)l] = 1;
    for (int l = 0; l < n_layer; ++l) o.n_own += o.own[(size_t)l];
    o.ranges = rs;
    for (size_t i = 0; i < rs.size(); ++i)
        o.text += (i ? "," : "") + std::to_string(rs[i].first) + ":" + std::to_string(rs[i].second);
    if (o.n_own == n_layer) {   /* every layer: the published engine */
        const bool strict = o.strict;
        const std::string t = o.text;
        o = Spec{};
        o.strict = strict;
        o.text = t;
        return 0;
    }
    o.on = true;
    o.first = rs.front().first;
    o.last = rs.back().second - 1;
    o.leader = o.own[(size_t)n_layer - 1] != 0;
    return 0;
}

/* The pair a strict node belongs to: W(h, t) with the leader holding [0, h) and [t, n_layer) and the follower [h, t). The PLE
 * layer (0-based, -1 none) must sit below h, so the leader keeps the token ids and the crossing into the follower carries only
 * the stream. 0 with *h / *t, or -1 with the reason. A proxy spec is not a pair: -1. */
inline int pair_of(const Spec& s, int n_layer, int ple_layer, int* h, int* t, std::string* err) {
    if (!s.on || !s.strict) {
        *err = "not a strict node (RADIANCE_FN4_LAYERS_MODE=strict with a part of the trunk)";
        return -1;
    }
    int hh = -1, tt = -1;
    if (s.leader) {
        /* [0, h) and [t, n_layer): two ranges, or one [t, n) when h would be 0 (refused below) */
        if (s.ranges.size() != 2 || s.ranges[0].first != 0 || s.ranges[1].second != n_layer) {
            *err = "the leader of W(h, t) holds 0:h,t:" + std::to_string(n_layer) + " (got " + s.text + ")";
            return -1;
        }
        hh = s.ranges[0].second;
        tt = s.ranges[1].first;
    } else {
        if (s.ranges.size() != 1 || s.ranges[0].first == 0) {
            *err = "the follower of W(h, t) holds one range h:t with h > 0 (got " + s.text + ")";
            return -1;
        }
        hh = s.ranges[0].first;
        tt = s.ranges[0].second;
    }
    if (ple_layer >= hh) {
        *err = "the PLE layer " + std::to_string(ple_layer) + " is past the cut h = " + std::to_string(hh) +
               ": the crossing would need the token ids (h > the PLE layer)";
        return -1;
    }
    *h = hh;
    *t = tt;
    return 0;
}

/* Bytes one row of a crossing carries: the wide hc stream (hc streams x n_embd bf16) and, where the hc write hands its norm
 * scales to the next read (RADIANCE_FN4_HC_SS, W1b: `ss`), hc f32 scales. Flash-Next: 4 x 2,560 x 2 = 20,480 B (+ 16). */
inline int64_t crossing_row_bytes(int64_t hc, int64_t n_embd, bool ss) {
    return hc * n_embd * 2 + (ss ? hc * 4 : 0);
}

}  /* namespace qfnlayers */
}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_ARCH_QFN_LAYERS_H */
