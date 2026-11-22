// Host side of lever P-2's union walk (r4d_attn_walk.h): the load list a workgroup of R rows streams through its ring, and each
// row entry's place in it. Used by r4d_selftest --perfatwalk and tests/r4d_emu/test_qwalk.cpp (the engine form would build the
// same lists on the card).
//
//   plain   the rows' page union, ascending: a tile whose four pages lie farther apart in it than the ring holds (or with an
//           entry out of ascending order) reads the cache
//   reload  the union in the same order, but after each load every row whose next FULL tile has all four pages loaded takes it
//           where the kernel's rule would (round jc = max(its previous round, last place / M)) if its oldest page is still in
//           the ring there (place >= (jc - 1) M), else that oldest page is loaded again at the end of the list until it is: so
//           every full tile reads the ring; repeats cost bytes (+5..+10 % at R 8, M 15 on the 12.11. dumps)
// Only the places of full tiles' entries are read by the kernel; the partial last tile reads the cache either way.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

// rows [r0, r1) of `table` ([rows][W], `used` keys a row, 4-key pages); appends the group's list to L and writes upos[r * W + b]
// (places relative to the group's first list entry)
inline void r4d_walk_group_list(const int32_t* table, int64_t W, const int32_t* used, int64_t r0, int64_t r1, int M, bool reload,
                                std::vector<int32_t>& L, std::vector<int32_t>& upos, std::vector<int32_t>& lastp) {
    constexpr int BS = 4, TILE = 16;
    std::vector<int32_t> base;
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = (used[r] + BS - 1) / BS;
        base.insert(base.end(), table + r * W, table + r * W + n);
    }
    std::sort(base.begin(), base.end());
    base.erase(std::unique(base.begin(), base.end()), base.end());
    const size_t g0 = L.size();
    if (!reload) {
        L.insert(L.end(), base.begin(), base.end());
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t n = (used[r] + BS - 1) / BS;
            for (int64_t b = 0; b < n; ++b)
                upos[(size_t)(r * W + b)] = (int32_t)(std::lower_bound(base.begin(), base.end(), table[r * W + b]) - base.begin());
        }
        return;
    }
    if (!base.empty() && (size_t)base.back() >= lastp.size()) lastp.resize((size_t)base.back() + 1, -1);
    struct St { int64_t ti, jr, nt; };
    std::vector<St> st;
    for (int64_t r = r0; r < r1; ++r) st.push_back({ 0, -1, used[r] / TILE });
    auto scan = [&]() {
        for (int64_t k = 0; k < r1 - r0; ++k) {
            St& s = st[(size_t)k];
            const int32_t* row = table + (r0 + k) * W;
            while (s.ti < s.nt) {
                const int32_t* e = row + 4 * s.ti;
                bool have = true;
                for (int i = 0; i < 4; ++i) have = have && lastp[(size_t)e[i]] >= 0;
                if (!have) break;
                int64_t jc;
                for (;;) {
                    int32_t mx = -1, mn = 1 << 30, old = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int32_t p = lastp[(size_t)e[i]];
                        mx = std::max(mx, p);
                        if (p < mn) { mn = p; old = i; }
                    }
                    jc = std::max<int64_t>(s.jr, mx / M);
                    if (mn >= (jc - 1) * M) break;
                    L.push_back(e[old]);
                    lastp[(size_t)e[old]] = (int32_t)(L.size() - 1 - g0);
                }
                for (int i = 0; i < 4; ++i) upos[(size_t)((r0 + k) * W + 4 * s.ti + i)] = lastp[(size_t)e[i]];
                s.jr = jc;
                ++s.ti;
            }
        }
    };
    for (const int32_t x : base) {
        L.push_back(x);
        lastp[(size_t)x] = (int32_t)(L.size() - 1 - g0);
        scan();
    }
    for (const int32_t x : base) lastp[(size_t)x] = -1;
}
