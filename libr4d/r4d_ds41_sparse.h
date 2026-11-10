/* r4d_ds41_sparse.h -- launches and shape hooks of DeepSeek-V4.1's sparse ops; included by r4d_plugin.h
 * inside its extern "C". */
#ifndef R4D_DS41_SPARSE_H
#define R4D_DS41_SPARSE_H

/* r4d_ds41_sparse.hip */
int     r4d_rad_ds41_index_score(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_candidates(const RadArgs*, RadStream);
int     r4d_rad_ds41_index_topk(const RadArgs*, RadStream);
int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs*);
int     r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
/* attn_sparse_sink + rope_rows_roundtrip over its output (lever K1 fold d) */
int     r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs*, RadStream);

/* r4d_ds41_sparse_shapes.inc */
int r4d_shape_index_score(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_candidates(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_index_topk(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_attn_sparse_sink(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_attn_sparse_sink_rope_roundtrip(const RadParam*, int, int, RadOpdDesc*);

/* r4d_fuse.cpp */
int r4d_fuse_attn_sparse_sink_rope_roundtrip(const RadParam*, int, uint64_t, int, RadFuseStep*);

#endif
