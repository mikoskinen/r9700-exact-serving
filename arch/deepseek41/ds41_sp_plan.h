/* ds41_sp_plan.h -- tp8 loop P20 round 2 (RADIANCE_DS41_SPHC): which layers of a lone prefill pass run their mHC stream ops
 * sequence-parallel (deepseek41.cpp step_bi; tests/ds41_sp_test.cpp holds the rule to a row-validity model of the issue).
 *
 * An SP layer runs every per-row stream op (the collapse + norm of its attention input where its own body issues it, the
 * attention mix, the post / collapse / norm of the FFN input and the FFN mix, an Engram gate, and the next layer's collapse
 * + norm or post at its end) over each rank's quarter of the rows; its sublayer outputs are reduce-scattered, its normed
 * inputs all-gathered. What it leaves sharded (the streams h0 / h1, the mixes, pre_ffn) is read whole by a non-SP layer
 * only after the streams' gather at that layer's top (h0), its own attention mix over every row (pre_attn, post, comb) and
 * its own FFN mix (pre_ffn) -- EXCEPT an in-body collapse: a layer whose previous layer issues no pcn_a for it (an Engram
 * layer, a proxy's first) collapses its attention input from pre_ffn at its top, before any mix of its own. So a layer is
 * SP only where its next issued layer is either collapsed by it (pcn_a) or SP itself. */
#pragma once

#include <cstdint>
#include <vector>

namespace ds41sp {

struct Layer {
    bool issued = false;      /* this part issues the layer (held, in the part) */
    bool self_ok = false;     /* its own conditions: an all-reduce'd attention, no DSpark tap, no crossing out after it,
                                 no PSK cut can reach it, its MoE reduces its delta in one all-reduce the plugin may take */
    int next = -1;            /* the next layer this part issues (-1: none) */
    bool next_pcn_a = false;  /* that layer's attention collapse is issued at this layer's end (pcn_a) */
    bool out_sharded = false; /* tp8 loop P20 round 3: the crossing out after it takes each rank's rows (X1 row-sharded) */
};

/* spv[i] = 1 where layer i is SP (index = layer) */
inline void plan(const std::vector<Layer>& L, std::vector<uint8_t>& spv) {
    spv.assign(L.size(), 0);
    for (int i = (int)L.size() - 1; i >= 0; --i) {
        const Layer& a = L[(size_t)i];
        if (!a.issued || !a.self_ok) continue;
        if (a.out_sharded) {   /* its outputs leave through the row-sharded crossing: nothing on this node reads them */
            spv[(size_t)i] = 1;
            continue;
        }
        if (a.next < 0 || a.next >= (int)L.size()) continue;
        spv[(size_t)i] = (a.next_pcn_a || spv[(size_t)a.next]) ? 1 : 0;
    }
}

}  // namespace ds41sp
