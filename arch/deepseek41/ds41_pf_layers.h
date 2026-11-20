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
 * (ms a 1k rows by layer over every such prompt), printed every `every` prompts. Host arithmetic only: tests/ds41_cone_test. */
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
constexpr int kRx = 1000;  /* kRx + l: layer l's crossing receive (its wait for the payload + the copy in), apart from its ops */

struct Rec {
    int64_t pos0 = 0, rows = 0;
    float pre_ms = 0.0f;
    std::vector<std::pair<int, float>> lms;   /* (layer, ms) in issue order */
};

/* ms by layer summed over records, and the rows each layer ran in them (a layer's rows counted once a record) */
struct Sums {
    std::map<int, double> ms;
    std::map<int, int64_t> rows;
    void add(const Rec& r) {
        ms[kPre] += r.pre_ms;
        rows[kPre] += r.rows;
        for (const auto& p : r.lms) {
            ms[p.first] += p.second;
            rows[p.first] += r.rows;
        }
    }
    void add(const Sums& o) {
        for (const auto& p : o.ms) ms[p.first] += p.second;
        for (const auto& p : o.rows) rows[p.first] += p.second;
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

    /* a read record; the lines of the prompt it closed (empty if none closed or that prompt was short) */
    std::vector<std::string> add(const Rec& r) {
        std::vector<std::string> out;
        if (!recs_.empty() && r.pos0 == 0 && !r.lms.empty() && covered0(r.lms.front().first)) out = flush();
        recs_.push_back(r);
        return out;
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
            small_rows_ += rows;
            ++small_n_;
            if (every_ > 0 && small_n_ % every_ == 0) {
                char b[256];
                std::snprintf(b, sizeof b, "%lld prompts under %lld rows (%lld rows): ms a 1k rows by layer: ", (long long)small_n_,
                              (long long)min_rows_, (long long)small_rows_);
                out.push_back(std::string(b) + by_layer(small_, true) + "; layers sum " + sum_per1k(small_));
            }
            recs_.clear();
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
        recs_.clear();
        return out;
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
    Sums small_;
    int64_t small_rows_ = 0, small_n_ = 0;
};

}  /* namespace ds41pfl */
