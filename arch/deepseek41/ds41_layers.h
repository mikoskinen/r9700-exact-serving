/* ds41_layers.h -- a node's layer ranges in deepseek41 (tp8 plan, stage S3: RADIANCE_DS41_LAYERS), pure host arithmetic so
 * tests/ds41_layers_test.cpp checks it without a card.
 *
 * Two machines run one model as PP2 x TP4 (~/gpu/ds41/tp8/PLAN.md): each engine process declares, loads and runs only its own
 * layers. The node's layers are named by ranges of trunk layers, hi exclusive, ascending and disjoint:
 *
 *   RADIANCE_DS41_LAYERS=0:20          layers 0..19             (W(0,20)'s follower)
 *   RADIANCE_DS41_LAYERS=20:40         layers 20..39            (W(0,20)'s leader)
 *   RADIANCE_DS41_LAYERS=0:10,32:40    layers 0..9 and 32..39   (W(10,32)'s leader, the wrap-around split)
 *   RADIANCE_DS41_LAYERS=10:32         layers 10..31            (W(10,32)'s follower)
 *
 * Unset, empty, or ranges that cover every layer: the published single-machine engine (Spec::on false). The node that holds
 * the last trunk layer is the leader (the plan's roles): it holds the embedding, the final norm, the head and the DSpark
 * drafter (whose taps are the last layers).
 *
 * RADIANCE_DS41_LAYERS_MODE picks what the node declares besides its layers:
 *   proxy  (default)  the stage timing proxy: a node that runs alone, without the other machine. It declares the embedding,
 *                     the final norm and the head whatever its role (the engine samples from its logits), and its first
 *                     layer, when that is not layer 0, starts from the embedding as layer 0 does (the synthetic boundary
 *                     input: the token rows expanded over the hc streams, one-hot collapse gates); a gap between two of its
 *                     ranges passes the residual stream through (the next held layer's collapse reads the previous held
 *                     layer's output). Answers are not the model's; the step's work per layer is. Only a node whose layers
 *                     read nothing another node computes (needs() empty) runs: W(0,20)'s two nodes.
 *   strict            the node's own part only, as the two-node engine (S4) declares it: the embedding, the final norm, the
 *                     head and the drafter only on the leader, the Engram tables only where an Engram layer is held, and a
 *                     copy of every kv source a held layer reads that another node computes (the KV group bound at that
 *                     source layer: what S6's KV copies fill). Declare-only in this build (no node link): its step refuses.
 *
 * needs() lists what a node's layers read that another node computes: a kv source's compressed latent (attention), its
 * index keys (an index layer's scores), the selections of the last index layer (a compressing layer past it that is no index
 * layer itself), the candidate layer's keep words (an index layer past it), and the DSpark taps (the leader's drafter). Each
 * compressing layer reads the latest kv / index source at or below it (deepseek41.cpp read_cfg), so these are the only
 * cross-layer reads besides the residual stream. */
#ifndef RAD_ARCH_DS41_LAYERS_H
#define RAD_ARCH_DS41_LAYERS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <tuple>
#include <vector>

namespace rad {
namespace arch {
namespace ds41layers {

/* The model's per-layer facts the ranges are checked against (deepseek41's Cfg, derived in read_cfg). */
struct Geo {
    int n_layer = 0;
    std::vector<int64_t> ratio;      /* compress ratio a layer, 0: window only */
    std::vector<int> src_of, idx_of; /* the kv / index source a compressing layer reads, -1 none */
    std::vector<char> is_src, is_idx;
    int cand = -1;                   /* the indexer's candidate layer, -1 none */
    std::vector<int> ple;            /* the Engram layers */
    std::vector<int> taps;           /* the DSpark target layers (the drafter reads their input streams) */
};

struct Spec {
    bool on = false;                          /* ranges set that are not every layer */
    bool strict = false;                      /* RADIANCE_DS41_LAYERS_MODE=strict */
    std::vector<std::pair<int, int>> ranges;  /* [lo, hi), ascending, disjoint, adjacent ones merged */
    std::vector<char> own;                    /* a layer this node declares and runs */
    std::vector<int> next;                    /* the next held layer after l, -1 none (every layer's entry) */
    int first = -1, last = -1, n_own = 0;
    bool leader = false;                      /* holds the last trunk layer */
    std::string text;                         /* the ranges as parsed, e.g. "0:10,32:40" */
    bool held(int l) const { return !on || (l >= 0 && l < (int)own.size() && own[(size_t)l]); }
};

/* RADIANCE_DS41_LAYERS's value (`s`, null or empty: unset) and RADIANCE_DS41_LAYERS_MODE's (`mode`, null or empty: proxy)
 * against a model of n_layer trunk layers. 0, or -1 with the reason in *err. */
inline int parse(const char* s, const char* mode, int n_layer, Spec* out, std::string* err) {
    *out = Spec{};
    if (mode && *mode && std::string(mode) != "proxy" && std::string(mode) != "strict") {
        *err = "RADIANCE_DS41_LAYERS_MODE=" + std::string(mode) + " is not proxy or strict";
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
            *err = "RADIANCE_DS41_LAYERS=" + std::string(s) + ": the range " + std::to_string(lo) + ":" + std::to_string(hi) +
                   " is not lo:hi with 0 <= lo < hi <= " + std::to_string(n_layer);
            return -1;
        }
        if (!rs.empty() && lo < rs.back().second) {
            *err = "RADIANCE_DS41_LAYERS=" + std::string(s) + ": the ranges are not ascending and disjoint";
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
        *err = "RADIANCE_DS41_LAYERS=" + std::string(s) + " is not lo:hi[,lo:hi...] (trunk layers, hi exclusive)";
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
    o.next.assign((size_t)n_layer, -1);
    int nx = -1;
    for (int l = n_layer - 1; l >= 0; --l) {
        o.next[(size_t)l] = nx;
        if (o.own[(size_t)l]) nx = l;
    }
    return 0;
}

enum class Need { Latent, IndexKeys, Sel, Keep, Tap };
inline const char* need_name(Need k) {
    switch (k) {
    case Need::Latent: return "compressed latent";
    case Need::IndexKeys: return "index keys";
    case Need::Sel: return "selections";
    case Need::Keep: return "keep words";
    case Need::Tap: return "DSpark tap";
    }
    return "?";
}
struct Dep {
    int layer = -1;   /* the held layer that reads it (for a tap: the leader's drafter, -1) */
    int from = -1;    /* the layer that computes it, on another node */
    Need kind = Need::Latent;
};

/* What the node's layers read that another node computes, in layer order. `drafter`: the node declares the drafter (the
 * leader, or a proxy that holds the taps). */
inline std::vector<Dep> needs(const Geo& g, const Spec& s, bool drafter) {
    std::vector<Dep> v;
    if (!s.on) return v;
    for (int l = 0; l < g.n_layer; ++l) {
        if (!s.own[(size_t)l] || g.ratio[(size_t)l] <= 0) continue;
        const int src = g.src_of[(size_t)l], idx = g.idx_of[(size_t)l];
        if (src >= 0 && !s.own[(size_t)src]) v.push_back({ l, src, Need::Latent });
        if (g.is_idx[(size_t)l]) {
            if (src >= 0 && !s.own[(size_t)src]) v.push_back({ l, src, Need::IndexKeys });
            if (g.cand >= 0 && g.cand < l && g.ratio[(size_t)g.cand] == g.ratio[(size_t)l] && !s.own[(size_t)g.cand])
                v.push_back({ l, g.cand, Need::Keep });
        } else if (idx >= 0 && !s.own[(size_t)idx]) {
            v.push_back({ l, idx, Need::Sel });
        }
    }
    if (drafter)
        for (int t : g.taps)
            if (t >= 0 && t < g.n_layer && !s.own[(size_t)t]) v.push_back({ -1, t, Need::Tap });
    return v;
}

/* The foreign kv sources whose caches a strict node holds a copy of (layer, latent, index keys): every source a Latent /
 * IndexKeys need names, ascending. */
struct Copy {
    int layer = -1;
    bool latent = false, index = false;
};
inline std::vector<Copy> copies(const std::vector<Dep>& deps) {
    std::vector<Copy> c;
    for (const Dep& d : deps) {
        if (d.kind != Need::Latent && d.kind != Need::IndexKeys) continue;
        size_t i = 0;
        while (i < c.size() && c[i].layer != d.from) ++i;
        if (i == c.size()) {
            size_t at = 0;
            while (at < c.size() && c[at].layer < d.from) ++at;
            c.insert(c.begin() + (std::ptrdiff_t)at, Copy{ d.from, false, false });
            i = at;
        }
        (d.kind == Need::Latent ? c[i].latent : c[i].index) = true;
    }
    return c;
}

/* ---------------------------------------------------------------- tp8 S6: the crossings of a general cut W(h, t)
 *
 * The leader holds [0, h) and [t, n), the follower [h, t). X1 (leader -> follower) enters the follower's layer h, X2
 * (follower -> leader) the leader's layer t. A crossing carries what the receiving node's layers read that the sending
 * node computed: the residual stream at the boundary (the inputs of the boundary op, as S4's X2; at h = 0 X1 carries the
 * embedding's output instead), and the receiver's needs() -- a kv source's new entries (the receiver keeps a copy of that
 * source's cache, written at the same pages with the same store op), the selections of the last index layer before the
 * boundary, the candidate layer's keep words. Each compressing layer reads the latest kv / index source at or below it, so
 * a crossing carries at most one source's entries, one layer's selections and one keep: the receiving layers past the
 * boundary read the latest ones below it until the receiver's own next source / index layer. The DSpark taps must be the
 * leader's own (t at or below the first tap): the drafter is the leader's. */
struct Crossing {
    int ring = 0;            /* 0 X1, 1 X2 */
    int at = -1;             /* the receiver's first layer after the crossing: h (X1) or t (X2) */
    bool residual = true;    /* the boundary op's inputs; false only for X1 at h == 0 (the embedding's output) */
    int sel = -1;            /* the index layer whose selections cross, -1 none */
    int keep = -1;           /* the candidate layer whose keep words cross, -1 none */
    int src = -1;            /* the kv source whose new entries cross (the receiver's copy), -1 none */
    bool latent = false;     /* ... its compressed latent */
    bool index = false;      /* ... its index keys */
};

/* The two crossings of W(h, t) on a model of g.n_layer layers; 0, or -1 with the reason in *err (a cut the geometry does
 * not allow: outside the model, the leader without the last layer, a tap on the follower, a crossing that would carry two
 * sources' entries / two layers' selections). */
inline int crossings(const Geo& g, int h, int t, Crossing out[2], std::string* err) {
    const int n = g.n_layer;
    if (h < 0 || t <= h || t >= n) {
        *err = "W(" + std::to_string(h) + "," + std::to_string(t) + ") is not 0 <= h < t < " + std::to_string(n) +
               " (the follower holds [h, t), the leader every other layer and the last one)";
        return -1;
    }
    Spec fol, lead;
    std::string e;
    const std::string fs = std::to_string(h) + ":" + std::to_string(t);
    const std::string ls = (h > 0 ? "0:" + std::to_string(h) + "," : std::string()) + std::to_string(t) + ":" + std::to_string(n);
    if (parse(fs.c_str(), "strict", n, &fol, &e) < 0 || parse(ls.c_str(), "strict", n, &lead, &e) < 0) {
        *err = e;
        return -1;
    }
    const std::vector<Dep> df = needs(g, fol, false), dl = needs(g, lead, true);
    for (int k = 0; k < 2; ++k) {
        Crossing& c = out[k];
        c = Crossing{};
        c.ring = k;
        c.at = k == 0 ? h : t;
        c.residual = !(k == 0 && h == 0);
        for (const Dep& d : k == 0 ? df : dl) {
            if (d.kind == Need::Tap) {
                *err = "the DSpark drafter reads layer " + std::to_string(d.from) + "'s input, which W(" + std::to_string(h) +
                       "," + std::to_string(t) + ") puts on the follower (the drafter is the leader's: t <= the first tap)";
                return -1;
            }
            int* slot = d.kind == Need::Sel ? &c.sel : d.kind == Need::Keep ? &c.keep : &c.src;
            if (*slot >= 0 && *slot != d.from) {
                *err = "X" + std::to_string(k + 1) + " of W(" + std::to_string(h) + "," + std::to_string(t) + ") would carry " +
                       need_name(d.kind) + " of layers " + std::to_string(*slot) + " and " + std::to_string(d.from);
                return -1;
            }
            *slot = d.from;
            if (d.kind == Need::Latent) c.latent = true;
            if (d.kind == Need::IndexKeys) c.index = true;
        }
        /* (review of S6) The crossing ships the sender's buffers as they are after its last layer before the boundary:
         * the latent / index-key / page planes hold the LAST kv source the sender computed, the selection plane its LAST
         * index layer's. So the source / index layer that crosses must be the sender's last one below the boundary, or
         * the receiver would store another source's entries into its copy (or read another layer's selections). True of
         * every cut on DeepSeek-V4.1-Flash (sources and index layers of one ratio are consecutive); refused by name on a
         * geometry where it is not. */
        const Spec& snd = k == 0 ? lead : fol;
        const auto last_below = [&](const std::vector<char>& is, int from) {
            for (int l = c.at - 1; l > from; --l)
                if (snd.held(l) && is[(size_t)l]) return l;
            return -1;
        };
        const int ls = c.src >= 0 ? last_below(g.is_src, c.src) : -1, li = c.sel >= 0 ? last_below(g.is_idx, c.sel) : -1;
        if ((c.src >= 0 && !snd.held(c.src)) || (c.sel >= 0 && !snd.held(c.sel)) || (c.keep >= 0 && !snd.held(c.keep)) ||
            ls >= 0 || li >= 0) {
            *err = "X" + std::to_string(k + 1) + " of W(" + std::to_string(h) + "," + std::to_string(t) + ") would carry " +
                   (ls >= 0 ? "kv source " + std::to_string(c.src) + "'s entries, but the sending node computes kv source " +
                                  std::to_string(ls) + " after it (its latent planes then hold " + std::to_string(ls) + "'s)"
                    : li >= 0 ? "index layer " + std::to_string(c.sel) + "'s selections, but the sending node runs index layer " +
                                    std::to_string(li) + " after it"
                              : std::string("a layer's entries the sending node does not hold"));
            return -1;
        }
    }
    return 0;
}

/* tp8 dev M4 (loop FINDINGS F12): a follower segment boundary s of W(h, t), h < s < t. The follower's pass runs [h, s), then
 * other lanes' passes, then [s, t) -- and those passes rewrite the step's activations. The segment before s saves (and the
 * one from s restores) on this node what is read at or after s of what [h, s) computed:
 *   - what a cut W(s, t)'s X1 would carry into layer s -- the residual (the boundary op's inputs), the selections, the keep
 *     words layers [s, t) read -- without its kv source: this node holds its caches and its copies;
 *   - what this pass's X2 (at t) ships that a layer below s computed: its kv source's new entries (the latent / index-key /
 *     page planes keep the last source the follower computed until X2's send: no source lies in [s, t) then), its
 *     selections and keep words when their layer is below s.
 * Missing the second half is F12: at W(10,32) X2 ships layer 20's entries, a decode pass between segments after 20 rewrote
 * the planes, and the leader's copy of source 20 took that pass's rows. 0 with *out, or -1 with the reason. */
inline int segment_crossing(const Geo& g, int h, int s, int t, Crossing* out, std::string* err) {
    if (s <= h || s >= t) {
        *err = "a segment boundary " + std::to_string(s) + " outside (" + std::to_string(h) + ", " + std::to_string(t) + ")";
        return -1;
    }
    Crossing whole[2], at_s[2];
    if (crossings(g, h, t, whole, err) < 0 || crossings(g, s, t, at_s, err) < 0) return -1;
    Crossing c = at_s[0];
    c.src = -1;
    c.latent = c.index = false;
    const Crossing& x2 = whole[1];
    if (x2.src >= 0 && x2.src < s) {
        c.src = x2.src;
        c.latent = x2.latent;
        c.index = x2.index;
    }
    for (const auto& [mine, theirs, what] : { std::tuple<int*, int, const char*>{ &c.sel, x2.sel, "selections" },
                                              std::tuple<int*, int, const char*>{ &c.keep, x2.keep, "keep words" } }) {
        if (theirs < 0 || theirs >= s) continue;
        if (*mine >= 0 && *mine != theirs) {   /* one plane holds the last one below s: both are it on every cut that parses */
            *err = "segment boundary " + std::to_string(s) + " of W(" + std::to_string(h) + "," + std::to_string(t) +
                   ") would save the " + what + " of layers " + std::to_string(*mine) + " and " + std::to_string(theirs);
            return -1;
        }
        *mine = theirs;
    }
    *out = c;
    return 0;
}

/* tp8 S5 + S6: with lanes the leader issues a lane step's pass in two parts -- its head (layers [0, h), X1) and its tail
 * (X2, layers [t, n)) -- with other lanes' parts between them, which rewrite the step's activations. What its tail reads
 * of its head must then be kv / state (paged per sequence) or cross in X2: a layer >= t that reads the selections of an
 * index layer < h, or an index layer >= t that reads the keep words of a candidate layer < h, would read another lane's.
 * Those reads, in layer order (empty: lanes may run W(h, t)); at h = 0 there are none. */
inline std::vector<Dep> head_tail_reads(const Geo& g, int h, int t) {
    std::vector<Dep> v;
    if (h <= 0) return v;
    for (int l = std::max(t, 0); l < g.n_layer; ++l) {
        if (g.ratio[(size_t)l] <= 0) continue;
        if (g.is_idx[(size_t)l]) {
            if (g.cand >= 0 && g.cand < h && g.cand < l && g.ratio[(size_t)g.cand] == g.ratio[(size_t)l])
                v.push_back({ l, g.cand, Need::Keep });
        } else if (g.idx_of[(size_t)l] >= 0 && g.idx_of[(size_t)l] < h) {
            v.push_back({ l, g.idx_of[(size_t)l], Need::Sel });
        }
    }
    return v;
}

/* Which source layers a node binds to its compressed / index / tail groups (S6): a source it holds (its own stores), or
 * the source a crossing into it copies (the copy store, latent and / or index keys). A group none of the node's layers
 * reads is bound at its first source and zeroed (no pool carved: S4's rule). `bytes_*`: the bytes a token of one bound
 * layer (deepseek41: latent / index entry bytes over the ratio). */
struct Binds {
    std::vector<char> comp, index, tail;   /* per layer: bound */
    std::vector<char> comp_zero_r, index_zero_r;   /* per ratio (index = ratio): every bound layer of it is unread here */
};
inline Binds binds(const Geo& g, const Spec& s, const Crossing* into /* the crossing into this node, or null */) {
    const int n = g.n_layer;
    Binds b;
    b.comp.assign((size_t)n, 0);
    b.index.assign((size_t)n, 0);
    b.tail.assign((size_t)n, 0);
    int64_t rmax = 0;
    for (int l = 0; l < n; ++l) rmax = std::max<int64_t>(rmax, g.ratio[(size_t)l]);
    b.comp_zero_r.assign((size_t)rmax + 1, 1);
    b.index_zero_r.assign((size_t)rmax + 1, 1);
    for (int l = 0; l < n; ++l) {
        if (!g.is_src[(size_t)l] || g.ratio[(size_t)l] <= 0) continue;
        const bool own = s.held(l);
        const bool cpl = into && into->src == l && into->latent, cpi = into && into->src == l && into->index;
        b.comp[(size_t)l] = own || cpl;
        b.index[(size_t)l] = own || cpi;
        b.tail[(size_t)l] = own && g.ratio[(size_t)l] > 1;
    }
    /* a ratio's group is read here iff a held compressing layer of that ratio exists (every compressing layer reads its
     * source's latent; an index layer its source's index keys) */
    for (int l = 0; l < n; ++l) {
        if (!s.held(l) || g.ratio[(size_t)l] <= 0) continue;
        b.comp_zero_r[(size_t)g.ratio[(size_t)l]] = 0;
        /* read by a held index layer, or written by a held source's index-key store */
        if (g.is_idx[(size_t)l] || g.is_src[(size_t)l]) b.index_zero_r[(size_t)g.ratio[(size_t)l]] = 0;
    }
    return b;
}

/* "layers 10..13 read layer 8's compressed latent; ..." -- one clause a (kind, from) pair, in first-seen order, with the
 * span of the held layers that read it. */
inline std::string describe(const std::vector<Dep>& deps) {
    struct Group {
        Need kind;
        int from;
        std::vector<int> ls;   /* the reading layers, ascending (deps come in layer order) */
    };
    std::vector<Group> gs;
    for (const Dep& d : deps) {
        size_t i = 0;
        while (i < gs.size() && !(gs[i].kind == d.kind && gs[i].from == d.from)) ++i;
        if (i == gs.size()) gs.push_back({ d.kind, d.from, {} });
        gs[i].ls.push_back(d.layer);
    }
    std::string out;
    for (const Group& g : gs) {
        std::string who;
        if (g.kind == Need::Tap) {
            who = "the drafter reads";
        } else if (g.ls.size() == 1) {
            who = "layer " + std::to_string(g.ls[0]) + " reads";
        } else if (g.ls.back() - g.ls.front() + 1 == (int)g.ls.size()) {
            who = "layers " + std::to_string(g.ls.front()) + ".." + std::to_string(g.ls.back()) + " read";
        } else {
            who = "layers ";
            for (size_t k = 0; k < g.ls.size(); ++k) who += (k ? ", " : "") + std::to_string(g.ls[k]);
            who += " read";
        }
        out += (out.empty() ? "" : "; ") + who + " layer " + std::to_string(g.from) + "'s " + need_name(g.kind);
    }
    return out;
}

}  // namespace ds41layers
}  // namespace arch
}  // namespace rad

#endif  // RAD_ARCH_DS41_LAYERS_H
