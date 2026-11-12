// r4d_ar_ll.hip: lever A3 / 1U-AR (ds41-ll) -- the four-rank exact all-reduce and moe_gather_all_reduce at decode sizes
// with the flags in the data words (LL). The rule and the wire: r4d_ar_ll.h; the kernel body: r4d_ar_ll_dev.h. Launched
// from r4d_ar_entry.hip (r4d_rad_ar_oneshot_4rank_exact, r4d_rad_moe_gather_ar_4rank) under libr4d fn4 ar_ll.
#include "r4d.h"
#include "r4d_ar_ll_dev.h"
#include <cstring>
#include <stdexcept>
#include <string>

static_assert(R4D_ARLL_LIST_ROWS == R4D_ARGL_ROWS && R4D_ARLL_LIST_TAB == R4D_ARGL_TAB,
              "r4d_ar_ll.h restates the gather's slot-list bounds");
static_assert(R4D_ARLL_MAX_BLOCKS <= R4D_AR_WIDE_MAX_BLOCKS, "an LL launch advances the parity words of its blocks");

template <bool GATHER, bool TWO>
__global__ __launch_bounds__(1024) void r4d_arll_kernel(R4dArllArgs A, R4dArGather G) {
    r4d_arll_body<GATHER, TWO>(A, G);
}

/* One LL launch. form: R4D_ARLL_ONESHOT / _TWOSHOT; gather: the gather's operands (ye ... sig as
 * r4d_ar_gather_twoshot_4rank takes them; y = out), else inp / out. peer_ll: the three peers' LL areas in ascending
 * global rank without this one. Throws on what the entry point checks first (a library fault if it escapes). */
void r4d_arll_launch(long form, long gather, const long* peer_ll, long my_ll, long seq_ctrs, long slot_units,
                     long inp, long out, long n16, long rank, long stream, long nblocks, long nthreads, long gate,
                     long gate_ctr, long ye, long ew, long sorted, long sh, long sg, long ye_ld, long ew_ld,
                     long sh_ld, long sg_ld, long top_k, long sig, long M, long n) {
    if (rank < 0 || rank >= 4) throw std::runtime_error("ar_ll: rank " + std::to_string(rank) + " out of 0 .. 3");
    if (!peer_ll || !peer_ll[0] || !peer_ll[1] || !peer_ll[2] || !my_ll || !seq_ctrs || !out)
        throw std::runtime_error("ar_ll: a null LL area, counter or output");
    if (!gather && !inp) throw std::runtime_error("ar_ll: a null input");
    if (n16 < 1 || r4d_arll_need((int)form, n16) > slot_units || (form == R4D_ARLL_TWOSHOT && n16 % 4))
        throw std::runtime_error("ar_ll: " + std::to_string(n16) + " words do not fit the form's slot of " +
                                 std::to_string(slot_units) + " units");
    if (nblocks < 1 || nblocks > R4D_ARLL_MAX_BLOCKS || nthreads < 32 || nthreads > 1024)
        throw std::runtime_error("ar_ll: grid " + std::to_string(nblocks) + " x " + std::to_string(nthreads));
    const long area = 2L * 4 * 8 * slot_units;   /* 2 parities x 4 sources x slot_units 8-byte units */
    r4d_ar_check_extent(my_ll, area, "ar_ll", "my LL area");
    for (int p = 0; p < 3; ++p) r4d_ar_check_extent(peer_ll[p], area, "ar_ll", "a peer's LL area");
    R4dArllArgs A;
    std::memset(&A, 0, sizeof A);   /* padding too: a recorded pass's audit compares the argument bytes */
    A.in4 = (const int4*)inp;
    A.out4 = (int4*)out;
    for (int p = 0; p < 3; ++p) A.peer[p] = (unsigned long long*)peer_ll[p];
    A.mine = (unsigned long long*)my_ll;
    A.seq = (unsigned int*)seq_ctrs;
    A.gate = (long long*)gate;
    A.gate_ctr = (unsigned int*)gate_ctr;
    A.slot_units = slot_units;
    A.n16 = (int)n16;
    A.rank = (int)rank;
    A.wpr = (int)(n / 8);
    A.test_tag = 0;
    A.phases = 7;
    R4dArGather G;
    std::memset(&G, 0, sizeof G);
    if (gather) {
        if (!ye || !ew || !sorted || ye_ld % 8 || (sh != 0) != (sg != 0) || n % 8 || M < 1 ||
            M > R4D_ARGL_ROWS || M * top_k > R4D_ARGL_TAB || top_k < 1 || n16 != M * n / 8)
            throw std::runtime_error("ar_ll (gather): operands outside the slot-list form");
        G.ye = (const unsigned short*)ye;
        G.ew = (const unsigned short*)ew;
        G.sorted = (const int*)sorted;
        G.sh = (const unsigned short*)sh;
        G.sg = (const unsigned short*)sg;
        G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
        G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)(sig & 1); G.ew_f32 = (int)((sig >> 1) & 1);
    }
    hipStream_t st = (hipStream_t)stream;
    const dim3 grid((unsigned)nblocks), block((unsigned)nthreads);
    if (gather) {
        if (form == R4D_ARLL_TWOSHOT) r4d_arll_kernel<true, true><<<grid, block, 0, st>>>(A, G);
        else r4d_arll_kernel<true, false><<<grid, block, 0, st>>>(A, G);
    } else {
        if (form == R4D_ARLL_TWOSHOT) r4d_arll_kernel<false, true><<<grid, block, 0, st>>>(A, G);
        else r4d_arll_kernel<false, false><<<grid, block, 0, st>>>(A, G);
    }
}
