/* ds41_dkv_held.h -- tp8 loop D25 step 1 (RADIANCE_DS41_DKV_GPIECE=1): lever DKV's gathered latent rows held a decode band
 * piece. Under F58's band pieces (a decode segment above 48 rows as pieces of whole sequences of <= 48 rows) every piece
 * gathered into the same 48-row Model::selkv and the held gather was ONE entry, so at 64 rows each piece re-gathered at every
 * layer instead of once a kv source (lq167: latent_gather 143 -> 905 us a follower lane step for +33 % rows). Here each
 * piece's gather sits at its own rows of a selkv of up to 64 rows and is held by its key (the kv / index source, the
 * piece's sequences and rows); a gather into rows another entry holds drops that entry. Pure, host-tested by
 * tests/ds41_dkv_held_test.cpp: the gathered bytes are a copy of the same selected entries, so where a gather sits cannot
 * change a value -- what must hold is that a hit is only ever an entry whose rows nothing has overwritten since. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct Ds41Held {
    int src = -1, idx = -1;
    int64_t s0 = -1, s1 = -1, r0 = -1, n = 0;
};

/* `want` is held (true: a hit, issue nothing) or not (false: the caller gathers into rows [want.r0, want.r0 + want.n), and the
 * entries those rows overlap are dropped before `want` is recorded). */
inline bool ds41_held_take(std::vector<Ds41Held>& held, const Ds41Held& want) {
    for (const Ds41Held& h : held)
        if (h.src == want.src && h.idx == want.idx && h.s0 == want.s0 && h.s1 == want.s1 && h.r0 == want.r0 && h.n == want.n)
            return true;
    held.erase(std::remove_if(held.begin(), held.end(),
                              [&](const Ds41Held& h) { return h.r0 < want.r0 + want.n && want.r0 < h.r0 + h.n; }),
               held.end());
    held.push_back(want);
    return false;
}
