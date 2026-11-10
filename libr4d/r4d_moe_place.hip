// r4d_moe_place.hip: moe_scatter_placed (lever D3s) -- moe_scatter's small sort over an expert
// PLACEMENT instead of the contiguous expert-parallel split: any expert on any rank, and an expert
// held by several ranks served, for each step, by the copy the dispatch rule picks.
//
// The placement is the op's own table, built at init from three parameters:
//   place    one hex digit a model expert (n_expert_all of them): the ranks holding a copy (bit r)
//   n_home   the home run's length: rank r's local experts 0 .. n_home - 1 are model experts
//            r * n_home .., as the contiguous split places them
//   extras   comma list of the further model experts this rank holds (ascending), local n_home ..
//   cost     optional comma list of n_expert_all positive integers, each expert's load (default 16)
// `n_expert` is this rank's local count (n_home + the extras, padding allowed past them).
//
// The dispatch, the same on every rank (the same ids, the same table): every active expert with one
// copy loads its rank by its cost; then the active experts with several copies, in ascending model
// id, each go to the copy whose rank is least loaded so far (ties: the first such rank in rotation
// from id % world) and add their cost there. Every slot of an expert goes to its chosen copy, so an
// expert's weights are read once a step, on one rank. Each rank then sorts the slots it serves, by
// local expert and slot, exactly as moe_scatter sorts its slice (r4d_moe_sort.h), and every slot
// of the step lands on exactly one rank. Integer loads, so the choice does not depend on an order
// of additions. One workgroup: M * top_k <= kScatterSmall, the decode and verify band.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_moe_sort.h"
#include "r4d_moe_place.h"

using namespace r4d_moesort;

namespace {

constexpr uint32_t kPlaceWaves = 8;   /* 256 threads, as moe_scatter's small form */

struct PlaceLds {
    ScatterLds<kPlaceWaves> sl;
    int32_t act[kMaxExpert];          /* active this step */
    int32_t choice[kMaxExpert];       /* a replicated expert's chosen rank */
    int32_t list[kMaxExpert];         /* the active replicated experts, ascending */
    int32_t wt[(kMaxExpert + kPlaceWaves * kSortWave - 1) / (kPlaceWaves * kSortWave)][kPlaceWaves];
    int32_t load[kPlaceMaxRanks];
    int32_t nlist;
};

__global__ __launch_bounds__(kPlaceWaves * kSortWave) void place_scatter_kernel(
    const int32_t* __restrict__ ids, const int32_t* __restrict__ tab, int32_t* __restrict__ sorted,
    int32_t* __restrict__ off, int32_t* __restrict__ cnt, uint32_t T, uint32_t n_local,
    uint32_t n_all, int32_t world, int32_t rank) {
    __shared__ PlaceLds L;
    const uint32_t tid = threadIdx.x, lane = tid % kSortWave, wave = tid / kSortWave;
    const int32_t* mask = tab;
    const int32_t* loc  = tab + n_all;
    const int32_t* cost = tab + 2 * n_all;

    for (uint32_t e = tid; e < n_all; e += blockDim.x) L.act[e] = 0;
    for (uint32_t e = tid; e < n_local; e += blockDim.x) L.sl.cnt[e] = 0;
    if (tid < kPlaceMaxRanks) L.load[tid] = 0;
    lds_barrier();
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = ids[i];
        if (e >= 0 && e < (int32_t)n_all) L.act[e] = 1;   /* the same value from every writer */
    }
    lds_barrier();

    /* single-copy loads (integer atomics: order-free), and the replicated active experts compacted
     * in ascending id by a block scan of their flags */
    int32_t carry = 0;
    for (uint32_t b = 0, t = 0; b < n_all; b += kPlaceWaves * kSortWave, ++t) {
        const uint32_t e = b + tid;
        int32_t f = 0;
        if (e < n_all && L.act[e]) {
            const int32_t m = mask[e];
            if (__builtin_popcount((uint32_t)m) == 1) atomicAdd(&L.load[__builtin_ctz((uint32_t)m)], cost[e]);
            else f = 1;
        }
        const int32_t x = wave_incl_scan_i32(f, lane);
        if (lane == kSortWave - 1) L.wt[t][wave] = x;
        lds_barrier();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < kPlaceWaves; ++k) {
            const int32_t c = L.wt[t][k];
            below += (k < wave) ? c : 0;
            tot   += c;
        }
        if (f) L.list[carry + below + x - 1] = (int32_t)e;
        carry += tot;
    }
    if (tid == 0) L.nlist = carry;
    lds_barrier();

    /* the replicated experts, in order, each to its least-loaded copy */
    if (tid == 0) {
        int32_t ld[kPlaceMaxRanks];
        for (int r = 0; r < kPlaceMaxRanks; ++r) ld[r] = L.load[r];
        const int32_t nl = L.nlist;
        for (int32_t q = 0; q < nl; ++q) {
            const int32_t e = L.list[q], m = mask[e];
            int32_t best = -1;
            for (int32_t s = 0; s < world; ++s) {
                const int32_t r = (e + s) % world;
                if (!((m >> r) & 1)) continue;
                if (best < 0 || ld[r] < ld[best]) best = r;
            }
            L.choice[e] = best;
            if (best >= 0) ld[best] += cost[e];
        }
    }
    lds_barrier();

    /* this rank's slots, then moe_scatter's sort over the local experts */
    Protect none{};
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = ids[i];
        int32_t le = -1;
        if (e >= 0 && e < (int32_t)n_all) {
            const int32_t m = mask[e];
            const int32_t owner = __builtin_popcount((uint32_t)m) == 1 ? (int32_t)__builtin_ctz((uint32_t)m) : L.choice[e];
            if (owner == rank) le = loc[e];
        }
        L.sl.e[i] = le >= 0 && le < (int32_t)n_local ? sort_place(le, n_local, none) : -1;
    }
    lds_barrier();
    scatter_small_place<kPlaceWaves, false>(sorted, off, cnt, T, n_local, L.sl);
}

}  // namespace

/* ---------------------------------------------------------------- the table (host) */

int r4d_place_parse(const char* place, const char* extras, const char* cost, long long n_all,
                    long long n_home, long long n_local, int world, int rank, R4dPlaceTab* out,
                    std::string* why) {
    auto fail = [&](const std::string& m) { if (why) *why = m; return RAD_E_SHAPE; };
    if (!out) return RAD_E_INVAL;
    if (world < 1 || world > kPlaceMaxRanks || rank < 0 || rank >= world)
        return fail("world " + std::to_string(world) + " / rank " + std::to_string(rank));
    if (n_all < 1 || n_all > (long long)kMaxExpert || n_local < 1 || n_local > (long long)kMaxExpert)
        return fail("n_expert_all " + std::to_string(n_all) + " / n_expert " + std::to_string(n_local) +
                    " outside 1.." + std::to_string(kMaxExpert));
    if (n_home < 0 || n_home > n_local || n_home * world > n_all)
        return fail("n_home " + std::to_string(n_home));
    if (!place || (long long)std::strlen(place) != n_all)
        return fail("place names " + std::to_string(place ? std::strlen(place) : 0) + " experts, not " +
                    std::to_string(n_all));
    out->n_all = n_all;
    out->n_local = n_local;
    out->world = world;
    out->rank = rank;
    out->tab.assign((size_t)(3 * n_all), 0);
    int32_t* mask = out->tab.data();
    int32_t* loc = mask + n_all;
    int32_t* cst = mask + 2 * n_all;
    for (long long e = 0; e < n_all; ++e) {
        const char c = place[e];
        const int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (v <= 0 || v >= (1 << world))
            return fail("expert " + std::to_string(e) + " has no copy or names a rank past " +
                        std::to_string(world - 1) + " ('" + std::string(1, c) + "')");
        mask[e] = v;
        loc[e] = -1;
        cst[e] = 16;
    }
    for (long long i = 0; i < n_home; ++i) loc[rank * n_home + i] = (int32_t)i;
    long long nx = 0, prev = -1;
    for (const char* s = extras; s && *s;) {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v <= prev || v < 0 || v >= n_all)
            return fail("extras: '" + std::string(s).substr(0, 24) + "' is not an ascending model expert id");
        if (n_home && v / n_home == rank && v < n_home * world)
            return fail("extras: expert " + std::to_string(v) + " is this rank's home");
        if (n_home + nx >= n_local)
            return fail("extras: more than n_expert - n_home = " + std::to_string(n_local - n_home));
        loc[v] = (int32_t)(n_home + nx++);
        prev = v;
        s = end;
        if (*s == ',') ++s;
        else if (*s) return fail("extras: expected a comma list");
    }
    if (cost && *cost) {
        const char* s = cost;
        for (long long e = 0; e < n_all; ++e) {
            char* end = nullptr;
            const long v = std::strtol(s, &end, 10);
            if (end == s || v < 1 || v > (1L << 20)) return fail("cost: entry " + std::to_string(e));
            cst[e] = (int32_t)v;
            s = end;
            if (*s == ',') ++s;
            else if (*s || e + 1 < n_all) {
                if (e + 1 < n_all) return fail("cost: " + std::to_string(e + 1) + " entries, not " +
                                               std::to_string(n_all));
                return fail("cost: expected a comma list");
            }
        }
    }
    /* every copy this rank is named for must be one it holds */
    for (long long e = 0; e < n_all; ++e)
        if (((mask[e] >> rank) & 1) && loc[e] < 0)
            return fail("expert " + std::to_string(e) + " is placed on rank " + std::to_string(rank) +
                        ", which neither holds it at home nor lists it in extras");
    return RAD_OK;
}

void r4d_place_owner(const R4dPlaceTab& t, const int32_t* ids, long long T, int32_t* owner) {
    const long long E = t.n_all;
    const int32_t* mask = t.tab.data();
    const int32_t* cst = mask + 2 * E;
    std::vector<char> act((size_t)E, 0);
    for (long long i = 0; i < T; ++i)
        if (ids[i] >= 0 && ids[i] < E) act[(size_t)ids[i]] = 1;
    int32_t ld[kPlaceMaxRanks] = {};
    for (long long e = 0; e < E; ++e)
        if (act[(size_t)e] && __builtin_popcount((unsigned)mask[e]) == 1)
            ld[__builtin_ctz((unsigned)mask[e])] += cst[e];
    std::vector<int32_t> choice((size_t)E, -1);
    for (long long e = 0; e < E; ++e) {
        if (!act[(size_t)e] || __builtin_popcount((unsigned)mask[e]) == 1) continue;
        int32_t best = -1;
        for (int s = 0; s < t.world; ++s) {
            const int r = (int)((e + s) % t.world);
            if (!((mask[e] >> r) & 1)) continue;
            if (best < 0 || ld[r] < ld[best]) best = r;
        }
        choice[(size_t)e] = best;
        if (best >= 0) ld[best] += cst[e];
    }
    for (long long i = 0; i < T; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= E) { owner[i] = -1; continue; }
        owner[i] = __builtin_popcount((unsigned)mask[e]) == 1 ? __builtin_ctz((unsigned)mask[e])
                                                              : choice[(size_t)e];
    }
}

#ifndef R4D_EMU_HOST   /* the emulator runs the kernel and the table, not the op's card buffers */
/* ---------------------------------------------------------------- the op */

struct PlaceInst {
    R4dPlaceTab t;
    int32_t* dtab = nullptr;
};

static long long pgi(const RadParam* p, int n_p, const char* k, long long dflt) {
    for (int i = 0; i < n_p; ++i) {
        if (std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT) return p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return p[i].ihi;
    }
    return dflt;
}
static const char* pgs(const RadParam* p, int n_p, const char* k) {
    for (int i = 0; i < n_p; ++i)
        if (std::strcmp(p[i].key, k) == 0 && p[i].kind == RAD_P_STR) return p[i].sval;
    return nullptr;
}

extern "C" int r4d_rad_moe_scatter_placed_init(const RadParam* p, int n_p, int rank, int world,
                                               void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    const long long ws = pgi(p, n_p, "world_size", world);
    if (ws != world) return R4D_NO(RAD_E_SHAPE);
    PlaceInst* in = new (std::nothrow) PlaceInst();
    if (!in) return RAD_E_NOMEM;
    std::string why;
    const int rc = r4d_place_parse(pgs(p, n_p, "place"), pgs(p, n_p, "extras"), pgs(p, n_p, "cost"),
                                   pgi(p, n_p, "n_expert_all", 0), pgi(p, n_p, "n_home", 0),
                                   pgi(p, n_p, "n_expert", 0), world, rank, &in->t, &why);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "libr4d: moe_scatter_placed (rank %d): %s\n", rank, why.c_str());
        delete in;
        return rc;
    }
    const size_t bytes = in->t.tab.size() * sizeof(int32_t);
    if (hipMalloc((void**)&in->dtab, bytes) != hipSuccess ||
        hipMemcpy(in->dtab, in->t.tab.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        if (in->dtab) (void)hipFree(in->dtab);
        delete in;
        return RAD_E_DEVICE;
    }
    *out = in;
    return RAD_OK;
}

extern "C" void r4d_rad_moe_scatter_placed_fini(void* instance) {
    PlaceInst* in = (PlaceInst*)instance;
    if (!in) return;
    if (in->dtab) (void)hipFree(in->dtab);
    delete in;
}

extern "C" int r4d_rad_moe_scatter_placed(const RadArgs* a, RadStream s) {
    const PlaceInst* in = (const PlaceInst*)(a ? a->instance : nullptr);
    if (!in || !in->dtab) return R4D_NO(RAD_E_STATE);
    const RadTensor* ids    = r4d_opt(a, 0);
    const RadTensor* sorted = r4d_opt(a, 1);
    const RadTensor* off    = r4d_opt(a, 2);
    const RadTensor* cnt    = r4d_opt(a, 3);
    if (!ids || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if (!ids->data || !sorted->data || !off->data || !cnt->data) return R4D_NO(RAD_E_INVAL);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(ids) || !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt))
        return R4D_NO(RAD_E_STRIDE);
    const long long T = r4d_numel(ids);
    if (T <= 0 || T > (long long)kScatterSmall) return R4D_NO(RAD_E_SHAPE);
    const long long nl = in->t.n_local;
    if (r4d_numel(sorted) < T || r4d_numel(off) < nl + 1 || r4d_numel(cnt) < nl)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(place_scatter_kernel, dim3(1), dim3(kPlaceWaves * kSortWave), 0, (hipStream_t)s,
                       (const int32_t*)ids->data, (const int32_t*)in->dtab, (int32_t*)sorted->data,
                       (int32_t*)off->data, (int32_t*)cnt->data, (uint32_t)T, (uint32_t)nl,
                       (uint32_t)in->t.n_all, (int32_t)in->t.world, (int32_t)in->t.rank);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

#endif  /* R4D_EMU_HOST */

/* The emulator's and the selftest's handle on the kernel, over a caller's device table. */
int r4d_moe_scatter_placed_launch(const int32_t* ids, const int32_t* dtab, int32_t* sorted, int32_t* off,
                                  int32_t* cnt, long long T, long long n_local, long long n_all, int world,
                                  int rank, void* stream) {
    if (T <= 0 || T > (long long)kScatterSmall) return RAD_E_SHAPE;
    hipLaunchKernelGGL(place_scatter_kernel, dim3(1), dim3(kPlaceWaves * kSortWave), 0,
                       (hipStream_t)stream, ids, dtab, sorted, off, cnt, (uint32_t)T, (uint32_t)n_local,
                       (uint32_t)n_all, (int32_t)world, (int32_t)rank);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
