/* ds41_pf_layers.h -- tp8 loop P42 (RADIANCE_DS41_PF_LAYERS, deepseek41): a lone cold prompt's card time BY LAYER on this node's
 * rank 0, from the card events step_bi records around every layer it issues (no --profile-ops, no sync: a timing start reads it).
 *
 * A record is one step_bi call of a lone planned prefill pass (one prefill segment, no decode row): its first position, rows,
 * the time before its first layer ("pre": the leader head's token ops) and each issued layer's ms (from its mark to the next
 * layer's, the last one to the call's end: the crossing out, the final norm / head); a layer that receives a crossing (X1 on the
 * follower, X2 on the leader's tail, a segment's restore) is split at the receive's end: "rx<l>" = the wait for the payload + its
 * copy in, "<l>" = the rest of the layer. The leader's head and tail of a chunk, a follower pass's layer segments: one record each. Records are
 * grouped into prompts in arrival order: a record at position 0 whose first layer a position-0 record of the open prompt
 * already covered starts the next prompt (the next newcomer's head, not the same chunk's tail or next segment). One prompt at a
 * time: two prompts whose chunks interleave are read as one (as the P22 line).
 *
 * At a prompt's end (the next prompt starts, or a pass that is no lone prefill pass finds every record read) its lines: the body
 * (every chunk but the last two, with >= 3 chunks) as ms a 1k rows by layer, and the last two chunks' ms by layer (the cone
 * layers past the split layer run their cone rows only there). Prompts under `min_rows` rows go into a running summary
 * (ms a 1k rows by layer over every such prompt), printed every `every` prompts. Host arithmetic only: tests/ds41_cone_test.
 *
 * Which interval a mark names (the code): a mark's id takes the time from it to the next mark. So at a layer with a receive the
 * layer's own id `l` holds the receive (the wait for the payload + its copy in) and `rx<l>` the layer's ops after it; at a layer
 * without one `l` holds the whole layer. (Read 20.11. PREFILL round 37: the follower's "10 14.25" is X1's wait, "rx10 4.86" layer
 * 10's ops; the leader's last tail "32 211.80" X2's wait, "rx32 10.18" layer 32's ops.)
 *
 * tp8 loop P42 step 2 (RADIANCE_DS41_PF_OPS=1 beside PF_LAYERS): marks INSIDE each issued layer at its op groups (sub_id(l, g),
 * the groups below). The lines above stay what they were -- a group's time folds into the layer's ops bucket (rx<l> where the
 * layer received, else l) -- and two more lines a prompt name the groups: the body in ms a 1k rows by layer and group, the last
 * two chunks in ms (a / b), each with a by-group total over the layers. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ds41pfl {

constexpr int kPre = -1;   /* the "layer" of the time before a record's first layer */
constexpr int kRx = 1000;  /* kRx + l: layer l's ops after its crossing receive (the receive itself is l's: see above) */
/* tp8 loop P42 step 2: a mark inside layer l at the start of op group g (RADIANCE_DS41_PF_OPS) */
constexpr int kSub = 100000;
constexpr int kGroups = 9;
/* e: the Engram block (lookup, copy, kv GEMM, gate); n: the attention input's collapse + norm + mix (+ a DSpark tap); q: the
 * attention's input linears (q_a / kv / the index projection), q_b, norms, rope, the window-key store; c: a kv source's
 * compressor chain + latent / index-key stores (+ the index query's rope after it); i: the indexer (score, candidates, top-k,
 * the selection's gather); a: the attention + its output projections; r: the attention all-reduce + post / collapse / norm into
 * the FFN input; f: the MoE (router, experts, shared arm, gather, all-reduce); o: the layer's end (a crossing's send, the next
 * layer's post / collapse / norm) */
/* tp8 loop P53 (RADIANCE_DS41_PF_OPS=2): finer groups that split three of the above where they start -- p: the attention's output
 * projections (o_a groups, the mix, o_b; a is then the attention core), m: the post / collapse / norm into the FFN input (r is
 * then the attention all-reduce or an SP layer's reduce-scatter), h: an SP layer's FFN-input all-gather, g: the MoE's
 * reduce-scatter (an SP layer) or its one all-reduce over the step (lever A9; f is then the MoE pass). Printed after o. */
constexpr int kGP = 9, kGM = 10, kGH = 11, kGG = 12;
constexpr int kGroups2 = 13;
inline const char* gname(int g) {
    static const char* const n[kGroups2] = { "e", "n", "q", "c", "i", "a", "r", "f", "o", "p", "m", "h", "g" };
    return g >= 0 && g < kGroups2 ? n[g] : "?";
}
inline int sub_id(int l, int g) { return kSub + l * 16 + g; }
inline bool is_sub(int id) { return id >= kSub; }
inline int sub_l(int id) { return (id - kSub) / 16; }
inline int sub_g(int id) { return (id - kSub) % 16; }

struct Rec {
    int64_t pos0 = 0, rows = 0;
    float pre_ms = 0.0f;
    std::vector<std::pair<int, float>> lms;   /* (layer, ms) in issue order */
    std::vector<std::pair<int, int64_t>> lrows;   /* P42 step 2 (PF_OPS): (layer, the rows it ran after lever PSK's cut) */
};

/* ms by layer summed over records, and the rows each layer ran in them (a layer's rows counted once a record) */
struct Sums {
    std::map<int, double> ms;
    std::map<int, int64_t> rows;
    std::map<int, int64_t> lrows;   /* P42 step 2: by layer, the rows it ran (summed over the records that told it) */
    void add(const Rec& r) {
        ms[kPre] += r.pre_ms;
        rows[kPre] += r.rows;
        for (const auto& p : r.lms) {
            ms[p.first] += p.second;
            rows[p.first] += r.rows;
        }
        for (const auto& p : r.lrows) lrows[p.first] += p.second;
    }
    void add(const Sums& o) {
        for (const auto& p : o.ms) ms[p.first] += p.second;
        for (const auto& p : o.rows) rows[p.first] += p.second;
        for (const auto& p : o.lrows) lrows[p.first] += p.second;
    }
    double total() const {
        double t = 0;
        for (const auto& p : ms) t += p.second;
        return t;
    }
};

inline std::string lname(int l) {
    return l == kPre ? std::string("pre") : l >= kRx ? "rx" + std::to_string(l - kRx) : std::to_string(l);
}

/* "pre 1.23, 0 4.56, ..." -- ms a 1k rows by layer (per1k) or ms (!per1k) */
inline std::string by_layer(const Sums& s, bool per1k) {
    std::string o;
    char b[64];
    for (const auto& p : s.ms) {
        const auto it = s.rows.find(p.first);
        const int64_t n = it == s.rows.end() ? 0 : it->second;
        const double v = per1k ? (n > 0 ? p.second * 1000.0 / (double)n : 0.0) : p.second;
        std::snprintf(b, sizeof b, "%s%s %.2f", o.empty() ? "" : ", ", lname(p.first).c_str(), v);
        o += b;
    }
    return o;
}

class Acc {
public:
    explicit Acc(int64_t min_rows = 8192, int64_t every = 128) : min_rows_(min_rows), every_(every) {}

    /* a read record; the lines of the prompt it closed (empty if none closed or that prompt was short). A record with op-group
     * marks (P42 step 2) is kept as read for the ops lines and folded into its layers for the lines above. */
    std::vector<std::string> add(const Rec& r) {
        std::vector<std::string> out;
        if (!recs_.empty() && r.pos0 == 0 && !r.lms.empty() && covered0(r.lms.front().first)) out = flush();
        bool sub = false;
        for (const auto& p : r.lms) sub = sub || is_sub(p.first);
        if (sub) {
            ops_.push_back(r);
            recs_.push_back(fold(r));
        } else {
            recs_.push_back(r);
        }
        return out;
    }
    /* P42 step 2: a record's group marks folded into their layer's ops bucket (rx<l> where the layer received, else l) */
    static Rec fold(const Rec& r) {
        Rec f;
        f.pos0 = r.pos0;
        f.rows = r.rows;
        f.pre_ms = r.pre_ms;
        f.lrows = r.lrows;
        for (const auto& p : r.lms) {
            int id = p.first;
            if (is_sub(id)) {
                const int l = sub_l(id);
                bool rx = false;
                for (const auto& q : r.lms) rx = rx || q.first == kRx + l;
                id = rx ? kRx + l : l;
            }
            bool merged = false;
            for (auto& q : f.lms)
                if (q.first == id) {
                    q.second += p.second;
                    merged = true;
                    break;
                }
            if (!merged) f.lms.push_back({ id, p.second });
        }
        return f;
    }
    bool open() const { return !recs_.empty(); }

    /* close the open prompt: its lines if it has >= min_rows rows, else into the summary (its line every `every` prompts) */
    std::vector<std::string> flush() {
        std::vector<std::string> out;
        if (recs_.empty()) return out;
        std::map<int64_t, int64_t> chunks;   /* pos0 -> rows */
        for (const Rec& r : recs_) chunks[r.pos0] = std::max(chunks[r.pos0], r.rows);
        int64_t rows = 0;
        for (const auto& c : chunks) rows = std::max(rows, c.first + c.second);
        if (rows < min_rows_) {
            for (const Rec& r : recs_) small_.add(r);
            for (const Rec& r : ops_) small_ops_.add(r);   /* P42 step 2: the short prompts' groups (BG newcomers) */
            small_rows_ += rows;
            ++small_n_;
            if (every_ > 0 && small_n_ % every_ == 0) {
                char b[256];
                std::snprintf(b, sizeof b, "%lld prompts under %lld rows (%lld rows): ms a 1k rows by layer: ", (long long)small_n_,
                              (long long)min_rows_, (long long)small_rows_);
                out.push_back(std::string(b) + by_layer(small_, true) + "; layers sum " + sum_per1k(small_));
                if (!small_ops_.ms.empty()) {
                    std::snprintf(b, sizeof b, "%lld prompts under %lld rows, ops (RADIANCE_DS41_PF_OPS): ms a 1k rows by layer: ",
                                  (long long)small_n_, (long long)min_rows_);
                    out.push_back(std::string(b) + ops_line(&small_ops_, nullptr, true));
                }
            }
            recs_.clear();
            ops_.clear();
            return out;
        }
        /* the last two chunks (with >= 3 chunks), the body the rest */
        int64_t p1 = -1, p2 = -1;
        if (chunks.size() >= 3) {
            auto it = chunks.rbegin();
            p2 = it->first;
            ++it;
            p1 = it->first;
        }
        Sums body, last1, last2;
        for (const Rec& r : recs_) (r.pos0 == p1 ? last1 : r.pos0 == p2 ? last2 : body).add(r);
        int64_t brows = 0;
        for (const auto& c : chunks)
            if (c.first != p1 && c.first != p2) brows += c.second;
        char b[512];
        std::snprintf(b, sizeof b, "prompt of %lld rows, %zu chunks, %zu records; body %zu chunks / %lld rows: ms a 1k rows by layer: ",
                      (long long)rows, chunks.size(), recs_.size(), chunks.size() - (p1 >= 0 ? 2 : 0), (long long)brows);
        out.push_back(std::string(b) + by_layer(body, true) + "; layers sum " + sum_per1k(body) + ", " +
                      fmt1(body.total()) + " ms in all");
        if (p1 >= 0) {
            std::snprintf(b, sizeof b, "prompt of %lld rows, the last two chunks %lld@%lld / %lld@%lld: ms by layer (a / b): ",
                          (long long)rows, (long long)chunks[p1], (long long)p1, (long long)chunks[p2], (long long)p2);
            std::string s(b);
            std::map<int, std::pair<double, double>> m;
            for (const auto& p : last1.ms) m[p.first].first = p.second;
            for (const auto& p : last2.ms) m[p.first].second = p.second;
            bool first = true;
            for (const auto& p : m) {
                char c[64];
                std::snprintf(c, sizeof c, "%s%s %.2f / %.2f", first ? "" : ", ", lname(p.first).c_str(), p.second.first,
                              p.second.second);
                s += c;
                first = false;
            }
            out.push_back(s + "; sums " + fmt1(last1.total()) + " / " + fmt1(last2.total()) + " ms");
        }
        if (!ops_.empty()) {   /* P42 step 2: the op groups (the records that carried group marks) */
            Sums ob, o1, o2;
            for (const Rec& r : ops_) (r.pos0 == p1 ? o1 : r.pos0 == p2 ? o2 : ob).add(r);
            std::snprintf(b, sizeof b, "prompt of %lld rows, ops (RADIANCE_DS41_PF_OPS): body %zu chunks / %lld rows, ms a 1k rows by "
                          "layer [rx = a receive, in = the layer before its first group, then the groups]: ",
                          (long long)rows, chunks.size() - (p1 >= 0 ? 2 : 0), (long long)brows);
            out.push_back(std::string(b) + ops_line(&ob, nullptr, true));
            if (p1 >= 0) {
                std::snprintf(b, sizeof b, "prompt of %lld rows, ops (RADIANCE_DS41_PF_OPS): the last two chunks %lld@%lld / %lld@%lld, "
                              "ms by layer (a / b): ", (long long)rows, (long long)chunks[p1], (long long)p1, (long long)chunks[p2],
                              (long long)p2);
                out.push_back(std::string(b) + ops_line(&o1, &o2, false));
            }
        }
        recs_.clear();
        ops_.clear();
        return out;
    }

    /* P42 step 2: one record set's (or two's: a / b) ops by layer -- "L [rx v, in v, e v, n v, ...]" in layer order, then
     * "; by group: rx v, in v, e v, ..." summed over the layers; `per1k`: ms a 1k rows (one set), else ms */
    static std::string ops_line(const Sums* a, const Sums* bb, bool per1k) {
        /* label of an id: (layer, slot) -- slot 0 = rx, 1 = in, 2 + g = group g */
        const auto key = [](const Sums& s, int id, int* l, int* slot) -> bool {
            if (id == kPre) return false;
            if (is_sub(id)) {
                *l = sub_l(id);
                *slot = 2 + sub_g(id);
                return true;
            }
            if (id >= kRx) {
                *l = id - kRx;
                *slot = 1;
                return true;
            }
            *l = id;
            *slot = s.ms.count(kRx + id) ? 0 : 1;   /* a layer that received: its own id is the receive */
            return true;
        };
        const auto val = [&](const Sums& s, int id) {
            const auto it = s.ms.find(id);
            if (it == s.ms.end()) return 0.0;
            if (!per1k) return it->second;
            const auto r = s.rows.find(id);
            return r != s.rows.end() && r->second > 0 ? it->second * 1000.0 / (double)r->second : 0.0;
        };
        std::map<int, std::map<int, std::pair<double, double>>> t;   /* layer -> slot -> (a, b) */
        std::map<int, std::pair<double, double>> tot;                  /* slot -> (a, b) */
        /* the last two chunks: each layer's rows after lever PSK's cut first (slot -1, "rows a/b": a cut layer's fit F + r x rows) */
        if (bb)
            for (int k = 0; k < 2; ++k)
                for (const auto& p : (k == 0 ? a : bb)->lrows)
                    (k == 0 ? t[p.first][-1].first : t[p.first][-1].second) = (double)p.second;
        for (int k = 0; k < 2; ++k) {
            const Sums* s = k == 0 ? a : bb;
            if (!s) continue;
            for (const auto& p : s->ms) {
                int l = 0, slot = 0;
                if (!key(*s, p.first, &l, &slot)) continue;
                const double v = val(*s, p.first);
                (k == 0 ? t[l][slot].first : t[l][slot].second) += v;
                (k == 0 ? tot[slot].first : tot[slot].second) += v;
            }
        }
        const auto sname = [](int slot) {
            return slot == -1 ? std::string("rows") : slot == 0 ? std::string("rx") : slot == 1 ? std::string("in") : std::string(gname(slot - 2));
        };
        const auto fv = [&](const std::pair<double, double>& v) {
            char c[48];
            if (bb) std::snprintf(c, sizeof c, "%.2f/%.2f", v.first, v.second);
            else std::snprintf(c, sizeof c, "%.2f", v.first);
            return std::string(c);
        };
        const auto fr = [](const std::pair<double, double>& v) {
            char c[48];
            std::snprintf(c, sizeof c, "%lld/%lld", (long long)v.first, (long long)v.second);
            return std::string(c);
        };
        std::string o;
        for (const auto& L : t) {
            o += (o.empty() ? "" : ", ") + std::to_string(L.first) + " [";
            bool first = true;
            for (const auto& S : L.second) {
                o += (first ? "" : " ") + sname(S.first) + " " + (S.first == -1 ? fr(S.second) : fv(S.second));
                first = false;
            }
            o += "]";
        }
        o += "; by group:";
        bool first = true;
        for (const auto& S : tot) {
            o += (first ? " " : ", ") + sname(S.first) + " " + fv(S.second);
            first = false;
        }
        return o;
    }

private:
    /* a position-0 record of the open prompt already holds layer l */
    bool covered0(int l) const {
        for (const Rec& r : recs_)
            if (r.pos0 == 0)
                for (const auto& p : r.lms)
                    if (p.first == l) return true;
        return false;
    }
    static std::string fmt1(double v) {
        char b[32];
        std::snprintf(b, sizeof b, "%.1f", v);
        return b;
    }
    static std::string sum_per1k(const Sums& s) {
        /* every layer's ms a 1k rows of the rows it ran, summed (pre included) */
        double t = 0;
        for (const auto& p : s.ms) {
            const auto it = s.rows.find(p.first);
            if (it != s.rows.end() && it->second > 0) t += p.second * 1000.0 / (double)it->second;
        }
        char b[32];
        std::snprintf(b, sizeof b, "%.2f", t);
        return b;
    }
    int64_t min_rows_, every_;
    std::vector<Rec> recs_;
    std::vector<Rec> ops_;   /* P42 step 2: the open prompt's records with group marks, as read */
    Sums small_;
    Sums small_ops_;   /* P42 step 2: the short prompts' records with group marks, as read */
    int64_t small_rows_ = 0, small_n_ = 0;
};

}  /* namespace ds41pfl */
