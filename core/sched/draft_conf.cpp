/* draft_conf.cpp -- see draft_conf.h. */
#include "draft_conf.h"

#include <cmath>
#include <cstdio>

namespace rad {
namespace dconf {

Policy policy_of(int v) {
    Policy p;
    if (v == 0) return p;
    if (v == 2) {
        p.mode = 2;
    } else if (v == 1 || (v >= 5 && v <= 95)) {
        p.mode = 1;
        p.thr = v == 1 ? 0.5 : v / 100.0;
    } else if (v == 101 || (v >= 105 && v <= 195)) {
        p.mode = 1;
        p.replace = false;
        p.thr = v == 101 ? 0.5 : (v - 100) / 100.0;
    } else {
        return Policy{};
    }
    p.logit = (float)std::log(p.thr / (1.0 - p.thr));
    return p;
}

int prefix_len(const float* logits, int n, const Policy& p) {
    int k = 0;
    /* sigmoid(x) >= thr is x >= logit(thr); a NaN fails the comparison and stops the prefix. */
    while (k < n && logits[k] >= p.logit) ++k;
    return k;
}

static int decile(float logit) {
    const double s = 1.0 / (1.0 + std::exp(-(double)logit));
    int d = (int)(s * 10.0);
    return d < 0 ? 0 : d > 9 ? 9 : d;
}

void Tracker::proposed(uint64_t seq, int step, const float* logits, int n_logits, const int32_t* alt, int n_alt,
                       int len, int cut) {
    std::lock_guard<std::mutex> lk(mu_);
    Rec& r = recs_[seq];
    r.step = step;
    r.pending = true;
    r.has_logits = logits != nullptr && n_logits > 0;
    r.n_logits = r.has_logits ? (n_logits < kMaxLen ? n_logits : (int)kMaxLen) : 0;
    for (int j = 0; j < r.n_logits; ++j) r.logits[j] = logits[j];
    r.n_alt = alt && n_alt > 0 && len > 0 ? (n_alt < kMaxAlt ? n_alt : (int)kMaxAlt) : 0;
    r.len = r.n_alt ? (len < kMaxLen ? len : (int)kMaxLen) : 0;
    for (int a = 0; a < r.n_alt; ++a)
        for (int l = 0; l < r.len; ++l) r.alt[a * kMaxLen + l] = alt[a * len + l];
    r.cut = cut;
    if (r.n_alt > sh_n_alt_) sh_n_alt_ = r.n_alt;
    if (r.len > sh_len_) sh_len_ = r.len;
    if (cut >= 0) {
        ++cuts_;
        ++cut_hist_[cut < kMaxLen ? cut : (int)kMaxLen];
        if (cut >= n_logits && n_logits > 0) ++cut_full_;
    }
}

void Tracker::close(const Open& o) {
    for (int na = o.a + 1; na <= sh_n_alt_ && na <= kMaxAlt; ++na)
        for (int L = 1; L <= o.len && L <= kMaxLen; ++L) sh_gain_[na - 1][L - 1] += o.matched < L ? o.matched : L;
}

void Tracker::verified(uint64_t seq, int step, int drafted, int accepted, const int32_t* emitted, int n_emitted) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = recs_.find(seq);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    r.step = step;
    /* Branches opened by earlier verifies take this verify's tokens first: they come after theirs. */
    for (size_t i = 0; i < r.open.size();) {
        Open& o = r.open[i];
        bool done = o.matched >= o.len;
        for (int k = 0; k < n_emitted && !done; ++k) {
            if (emitted[k] == o.want[o.matched]) ++o.matched;
            else done = true;
            if (o.matched >= o.len) done = true;
        }
        if (done) {
            close(o);
            r.open.erase(r.open.begin() + (long)i);
        } else {
            ++i;
        }
    }
    if (!r.pending) return;
    r.pending = false;
    if (accepted > drafted) accepted = drafted;
    if (r.cut >= 0) cut_drafts_ += drafted;
    /* Calibration: positions 0..accepted are reached (the last rejected unless every draft was taken). */
    if (r.has_logits)
        for (int j = 0; j < drafted && j < r.n_logits && j <= accepted; ++j) {
            const int d = decile(r.logits[j]);
            ++reached_[j][d];
            if (j < accepted) ++hit_[j][d];
        }
    if (r.n_alt <= 0) return;
    if (drafted <= 0) {
        ++sh_zero_;
        return;
    }
    ++sh_verifies_;
    sh_chain_acc_ += accepted;
    if (accepted != 0 || n_emitted < 1) return;
    ++sh_p1_miss_;
    const int32_t t1 = emitted[0];
    for (int a = 0; a < r.n_alt; ++a) {
        if (t1 < 0 || r.alt[a * kMaxLen] != t1) continue;
        ++sh_hit_[a];
        Open o;
        o.a = a;
        o.len = r.len;
        o.matched = 1;
        for (int l = 0; l < r.len; ++l) o.want[l] = r.alt[a * kMaxLen + l];
        if (o.matched >= o.len) close(o);
        else r.open.push_back(o);
        break;   /* the branches' first tokens differ */
    }
}

void Tracker::purge(int step, int age) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = recs_.begin(); it != recs_.end();) {
        if (it->second.step < step - age) {
            for (const Open& o : it->second.open) close(o);
            it = recs_.erase(it);
        } else {
            ++it;
        }
    }
}

void Tracker::load_gate(bool bounded, bool gated) {
    if (!bounded) return;
    std::lock_guard<std::mutex> lk(mu_);
    ++lg_bounded_;
    if (gated) ++lg_gated_;
}

std::string Tracker::json() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::string o = "{\"conf\": {\"rows_bounded\": " + std::to_string(lg_bounded_) + ", \"rows_gated\": " +
                    std::to_string(lg_gated_) + ", \"cuts\": " + std::to_string(cuts_) + ", \"cut_drafts\": " +
                    std::to_string(cut_drafts_) + ", \"cut_full\": " + std::to_string(cut_full_) + ", \"cut_hist\": [";
    int top = 0;
    for (int k = 0; k <= kMaxLen; ++k)
        if (cut_hist_[k]) top = k + 1;
    for (int k = 0; k < top; ++k) o += (k ? ", " : "") + std::to_string(cut_hist_[k]);
    o += "], \"calib\": [";
    int deep = 0;
    for (int j = 0; j < kMaxLen; ++j)
        for (int d = 0; d < kDeciles; ++d)
            if (reached_[j][d]) deep = j + 1;
    for (int j = 0; j < deep; ++j) {
        o += j ? ", [" : "[";
        for (int d = 0; d < kDeciles; ++d)
            o += (d ? ", [" : "[") + std::to_string(reached_[j][d]) + ", " + std::to_string(hit_[j][d]) + "]";
        o += "]";
    }
    o += "]}, \"shadow\": {\"n_alt\": " + std::to_string(sh_n_alt_) + ", \"len\": " + std::to_string(sh_len_) +
         ", \"verifies\": " + std::to_string(sh_verifies_) + ", \"chain_accepted\": " + std::to_string(sh_chain_acc_) +
         ", \"p1_miss\": " + std::to_string(sh_p1_miss_) + ", \"zero_draft\": " + std::to_string(sh_zero_) +
         ", \"alt_hit\": [";
    for (int a = 0; a < sh_n_alt_; ++a) o += (a ? ", " : "") + std::to_string(sh_hit_[a]);
    o += "], \"gain\": [";
    for (int a = 0; a < sh_n_alt_; ++a) {
        o += a ? ", [" : "[";
        for (int L = 0; L < sh_len_; ++L) o += (L ? ", " : "") + std::to_string(sh_gain_[a][L]);
        o += "]";
    }
    o += "]}}";
    return o;
}

Tracker& tracker() {
    static Tracker* t = new Tracker();
    return *t;
}

}  // namespace dconf
}  // namespace rad
