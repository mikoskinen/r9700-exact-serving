/* r4d_qsa_part.h -- fn4 qsa_split (16.11.2026, qwen/QSA-SPLIT.md): the QSA indexer's score and selection split across the
 * ranks of a tensor-parallel group by block range, and the candidate-list layout the split selection hands to its merge.
 *
 * A rank owns the block range [qsa_part_cut(nc, W, r), qsa_part_cut(nc, W, r + 1)) of each sequence, nc being the
 * sequence's complete-block count (qsa_work's `nc`, identical on every rank). The cuts depend on that sequence alone, so a
 * row's ranges never depend on the other sequences of the step; they partition [0, nc) in ascending rank order, and every
 * inner cut is a multiple of 16 (the score's tile and float4 store). Any partition into ascending contiguous ranges keeps
 * the merge exact (r4d_qsa_select_bf16.hip, "the split path"): a block in the global top-k is in its range's top-k.
 *
 * A candidate list is LW = 4 + 2 * topk i32 words: [0] the count | kParOver when the list's range had more than topk live
 * blocks, [1] the selection's block bound nb (min of the table and score-plane widths at the part launch; the merge needs
 * it for the partial page and does not see the score plane), [2..3] zero, then topk keys (u32, qsa_select's order key) and
 * topk page ids, both in ascending block order. A row's lists are contiguous: list c of row m at m * pitch + c * LW. */
#ifndef R4D_QSA_PART_H
#define R4D_QSA_PART_H

#include <hip/hip_runtime.h>

static constexpr int kQsaListHead = 4;
__host__ __device__ static inline int qsa_list_words(int topk) { return kQsaListHead + 2 * topk; }

__host__ __device__ static inline int qsa_part_cut(int nc, int parts, int r) {
    if (r <= 0 || nc <= 0) return 0;
    if (r >= parts) return nc;
    const long long v = (long long)nc * r / parts;
    return (int)(v & ~15LL);
}

#endif /* R4D_QSA_PART_H */
