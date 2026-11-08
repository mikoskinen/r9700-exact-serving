// r4d_ar_wide_chain.h -- the four-rank fused all-reduce + hc_write entries with a chained form (fn4, 9.11.2026): the
// same as r4d_ar_hc_write_4rank / r4d_ar_gather_hc_write_4rank (r4d.h) plus `chain`, which launches the r4dchain_ form
// (r4d_chain.h: the kernel waits for its queue's counter instead of behind the barrier bit, and adds its blocks when done).
// A separate header so r4d.h, which nearly every libr4d source includes, stays as it is.
#pragma once
void r4d_ar_hc_write_4rank_c(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                             long peer_flags0, long peer_flags1, long peer_flags2,
                             long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                             long y, long inj, long h, long M, long n, long hc, long irow,
                             long hrow, long rank, long stream, long nblocks, long nthreads,
                             long drain, long acq, long pub, long gate, long gate_ctr, long chain);
void r4d_ar_gather_hc_write_4rank_c(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long y, long inj, long h, long M, long n,
                                    long hc, long irow, long hrow, long ye, long ew, long sorted,
                                    long sh, long sg, long ye_ld, long ew_ld, long sh_ld,
                                    long sg_ld, long top_k, long sig, long rank, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub,
                                    long inv, long gate, long gate_ctr, long chain);
