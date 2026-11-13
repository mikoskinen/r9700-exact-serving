# DeepSeek-V4.1-Flash-1017 at Unsloth's UD-IQ2_XXS mix as a Radiance recipe

Date: 30.10.2026. Radiance source: tag v1.14.0 (4a61d72), plus PR #266 (`deepseek41-wip`, head e30ab83) for the plugin.
Unsloth repo: `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`. I read only the GGUF headers, about 29 MiB in total
(UD-IQ1_S 9.5 MiB, UD-IQ2_XXS 10.5 MiB, DSpark BF16 GGUF 8 MiB, imatrix 1 MiB).

## Status in short

- **Recipe ready:** `deepseek41-ud-iq2xxs.recipe`. It has 1286 rules, with the same patterns and order as Radiance's
  IQ1_S recipe. Every type comes from the UD-IQ2_XXS GGUF.
- **Validated:** the same derivation run on the UD-IQ1_S GGUF reproduces Radiance's `deepseek41-ud-iq1s.recipe`
  **byte for byte** (0 of 1286 rules differ, `cmp` identical). The size model gives **251.07 GiB** for IQ1_S, which
  is exactly the plan size Radiance states.
- **Size of IQ2_XXS:** the plan is **299.23 GiB** (about 299.3 GiB as a container, 321 GB), or **173.34 GiB without the
  Engram table**.
- **Two blockers before a conversion can run**, covered under open issues below:
  1. Radiance v1.14.0 cannot read the release and has no `deepseek41` plugin. Both live only on PR #266.
  2. The kernels in PR #266 serve only q8_0, q5_K, q6_K, iq4_nl, iq2_xxs, iq1_s and mxfp4. The IQ2_XXS mix also uses
     **iq2_s, iq3_xxs, iq2_xs and q2_K** for routed experts, so declare will refuse the recipe, even with `--plan-only`.

## Files

| file | what |
|---|---|
| `gguf_header.py` | Stdlib-only GGUF header reader. It uses HTTP Range, growing the range until the tensor-info table parses; it also reads local files. `--hf REPO --prefix DIR/` reads every shard. |
| `ud-iq1s-tensors.tsv`, `ud-iq2xxs-tensors.tsv` | Every tensor: shard, name, ggml type, shape (ne0 first), elements, bytes from block size, bytes from offsets, offset. 1043 tensors each. |
| `ud-iq1s-kv.txt`, `ud-iq2xxs-kv.txt` | The GGUF metadata. Long arrays are truncated. |
| `dspark-bf16-tensors.tsv` | Header of `dspark/dspark-...-BF16.gguf`, the drafter. Used to check the kept DSpark rules and to size them. |
| `imatrix-unsloth-entries.tsv` | Header of `imatrix_unsloth.gguf_file`: 1184 entries, 558 chunks x 4096. |
| `make_recipe.py` | Maps GGUF names to recipe patterns, derives each rule, validates, and estimates sizes (GGUF block sizes, plus Radiance plane sizes from `lq_ggml.cpp` `encoding_of`). |
| `deepseek41-ud-iq2xxs.recipe` | **The deliverable.** |
| `deepseek41-ud-iq1s.regen.recipe`, `deepseek41-ud-iq1s.regen.diff` | The regenerated IQ1_S recipe and its diff against v1.14.0. The diff is empty, which means identical. |
| `report-iq1s-validation.txt`, `report-iq2xxs.txt` | `make_recipe.py` output, including the per-layer tables. |
| `rules-iq1s-to-iq2xxs.diff` | The rule-level diff between the IQ1_S recipe and the new one (606 rules differ). |
| `radiance-v1.14.0/` | Reference copies from the tag: the IQ1_S recipe, `deepseek41-hf-direct.md`, `engram_tables.py`, `imatrix_merge.py`. |

To regenerate:
```
python3 -I gguf_header.py --hf unsloth/DeepSeek-V4.1-Flash-1017-GGUF --prefix UD-IQ2_XXS/ --tsv ud-iq2xxs-tensors.tsv --kv ud-iq2xxs-kv.txt
python3 -I make_recipe.py --template radiance-v1.14.0/deepseek41-ud-iq1s.recipe --gguf ud-iq1s-tensors.tsv --dspark dspark-bf16-tensors.tsv --validate --keep-header --out deepseek41-ud-iq1s.regen.recipe
python3 -I make_recipe.py --template radiance-v1.14.0/deepseek41-ud-iq1s.recipe --gguf ud-iq2xxs-tensors.tsv --dspark dspark-bf16-tensors.tsv --header <header.txt> --out deepseek41-ud-iq2xxs.recipe
```

## Name mapping (GGUF to Radiance recipe pattern)

The plugin source on PR #266 (`arch/deepseek41/deepseek41.cpp`) uses the same fusions.

| recipe pattern | GGUF tensor(s) | note |
|---|---|---|
| `blk.N.ffn_gate_up_exps.*.weight` | `blk.N.ffn_gate_exps.weight` + `blk.N.ffn_up_exps.weight` | Fused per expert as [4608, 5120] x 384 experts. Gate and up have the same type in every layer of both mixes. |
| `blk.N.ffn_down_exps.*.weight` | `blk.N.ffn_down_exps.weight` | [5120, 2304] x 384 |
| `blk.N.ffn_gate_up_shexp.weight` | `ffn_gate_shexp` + `ffn_up_shexp` | fused |
| `blk.N.attn_output_a.g0..g7.weight` | `blk.N.attn_output_a.weight` (4096 x 8192) | Eight groups of [1024, 4096], all of one type. |
| `blk.N.attn_compressor_kvg.weight` (layers 2, 8, 14) | `attn_compressor_kv` + `attn_compressor_gate` | Fused. Layer 20 has kv only, under the same name. |
| `blk.N.glm_router.weight` | `blk.N.ffn_gate_inp.weight` | f32 |
| `blk.N.glm_router.correction` | `blk.N.exp_probs_b.bias` | f32 |
| `per_layer_token_embd.weight` | same name (256 x 768022850, 196.6 G values) | The Engram table, in the last shard. |
| all other trunk names (`attn_*`, `hc_*`, `indexer.*`, `engram_*`, norms, `token_embd`, `output`) | same name | |
| `blk.40-42.*`, `dspark.*` (86 rules) | **not in the UD GGUFs** (the DSpark drafter) | The template rule is kept. All 86 also match the DSpark BF16 GGUF's types (`blk.0-2.*`, `fc`, `markov_w1/2`, `enc.output_norm`, `output_norm`). |
| (none) | `blk.N.exp_probs_b_vl.bias` (40) | Not used. This is the release's `ffn.gate.bias_vl`, the vision router bias, which the plugin does not read. |

The rule is `cast dtype=<t>` when the GGUF type is f32, bf16 or f16, and `ggml type=<t>` otherwise.

## Validation (UD-IQ1_S GGUF against Radiance v1.14.0 `deepseek41-ud-iq1s.recipe`)

- 1286 rules: 1200 derived from the GGUF and 86 kept (DSpark). **0 differ.** The regenerated file is byte-identical,
  header included.
- No rule had to be guessed. The only lines that cannot be derived from the UD GGUF are the 86 DSpark rules, and
  those agree with the DSpark BF16 GGUF.
- The header's stated mix agrees with the GGUF: gate_up is iq1_s in 34 layers and iq2_xxs in 6; down is iq1_s in 20
  and iq2_xxs in 20; the Engram table is iq4_nl.
- **The size model reproduces 251.07 GiB exactly.** It counts the plane bytes of each logical weight as `lq_ggml.cpp`
  lays them out:
  - rows x ceil(bits x elements / 8) per plane,
  - plus the per-weight grid table (16 KiB for each iq1_s expert),
  - plus the DSpark weights: bf16 casts, and the 9 x 384 MXFP4 experts at 4.25 bpw.

  Radiance's own split agrees too: in #266, bakon gives ~107 GiB for llama.cpp blocks against 134 GiB stored. The
  model here gives 107.31 GiB and 133.49 GiB for the IQ1_S routed experts.
- The model counts 32694 logical weights; Radiance's plan states 32701. The 4 Engram hash arrays from `--add` and a
  few small unruled tensors probably make up the difference. This is not verifiable without running the plugin.

## The UD-IQ2_XXS mix

| layers | gate_up exps | down exps | shexp (gate_up, down) | attention (q_a, q_b, kv, output_a, output_b) |
|---|---|---|---|---|
| 0 | iq1_s | q2_K | q8_0 | q8_0 |
| 1 | iq1_s | iq2_xs | q8_0 | q8_0 |
| 2-4 | iq1_s | iq2_xxs | q8_0 | q8_0 |
| 5-9 | iq1_s | iq2_xs | q8_0 | q8_0 |
| 10-19 | iq2_xxs | iq2_s | q8_0 | q8_0 |
| 20-24 | iq2_xxs | iq3_xxs | q8_0 | q8_0 |
| 25-26 | iq2_s | iq3_xxs | q8_0 | q8_0 |
| 27-34 | iq2_s | iq3_xxs | q5_K | q6_K |
| 35-39 | iq2_s | iq2_s | q5_K | q6_K |

- **Counts:** gate_up is iq1_s in 10 layers, iq2_xxs in 15 and iq2_s in 15. Down is q2_K in 1, iq2_xs in 6, iq2_xxs in
  3, iq2_s in 15 and iq3_xxs in 15.
- **Other weights:** token_embd q8_0, output q6_K, engram_kv (layers 1 and 14) q6_K, and **the Engram table q5_K**
  (IQ1_S has iq4_nl). Indexer and compressor weights are cast to bf16; norms, router and hc weights to f32.
- **IQ1_S for comparison:** gate_up iq1_s/iq2_xxs; down iq1_s in 0-19 and iq2_xxs in 20-39; shexp q6_K, or q8_0 in
  14-26; attention q6_K in 0-26 and q5_K in 27-39; token_embd q6_K.
- **Same model metadata:** the two GGUFs' metadata is identical apart from split counts. Both were quantised with an
  imatrix of 592 entries and 558 chunks, which is very likely the published `imatrix_unsloth.gguf_file`.

### Sizes (GiB)

"Planes" is the .rad on disk and the plan. "Blocks" is GGUF density, which libr4d relayouts back to at load, so it is
what the cards hold.

| | IQ1_S planes | IQ1_S blocks | IQ2_XXS planes | IQ2_XXS blocks |
|---|---|---|---|---|
| routed experts (40 layers) | 133.49 (2.11 bpw) | 107.31 (1.70 bpw) | 157.59 (2.49 bpw) | 146.11 (2.31 bpw) |
| rest of the trunk | 6.66 | 6.66 | 7.84 | 7.84 |
| Engram table | 103.00 (iq4_nl) | 103.00 | 125.89 (q5_K) | 125.89 |
| DSpark drafter (bf16 + MXFP4 experts) | 7.92 | 7.92 | 7.92 | 7.92 |
| **total (plan)** | **251.07** (Radiance: 251.07) | 224.89 | **299.23** | 287.75 |
| without the Engram table | 148.07 | 121.89 | **173.34** | 161.86 |
| trunk only (no Engram, no DSpark) | 140.15 | 113.97 | 165.42 | **153.95** |

**VRAM:** the IQ2_XXS trunk at block density is 153.95 GiB, more than 4 x 32 GiB = 128 GiB. It would need
`--placement expert_tiered`, with part of the routed experts streamed from host RAM. bakon's TP4 setup already uses
that for IQ1_S, whose trunk is 113.97 GiB. On 2 x R9700 (xeon), neither mix fits without heavy streaming.

## libquant `ggml` quantiser at v1.14.0

- **Types it writes** (`kKinds[]` in `libquant/lq_ggml.cpp`): q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K iq4_nl
  iq4_xs iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m. **Every type in the IQ2_XXS mix is supported.** The image's
  `rad-convert --list-quantizers` shows the same list.
- **`--imatrix` is mandatory for iq2_xxs, iq2_xs and iq1_s** (`ggml_quantize_requires_imatrix`; iq1_m is commented out
  of that list). Without an imatrix entry for the weight, the quantiser refuses with "needs an imatrix ...
  (rad-convert --imatrix FILE)".
  - In IQ2_XXS this covers gate_up in layers 0-24 and down in layers 1-9.
  - The other types use the imatrix when an entry exists, and llama.cpp's `_ref` round-to-nearest path when not.
    q8_0 never uses it.
- **Imatrix coverage:** the Unsloth imatrix has entries for everything this needs: `ffn_{gate,up,down}_exps` (x384),
  `*_shexp`, `attn_*` (`attn_output_a` is 32768 wide = 8 groups x 4096) and `engram_kv`. It has none for
  `per_layer_token_embd` or `token_embd`, the same as llama.cpp.

## Radiance 1.14.0 image on gpu01 (`stilldeadcode/radiance:1.14.0`, 693 MB)

- **`rad-convert`** takes .gguf, .safetensors, an index.json or a checkpoint directory.
  - It has **`--plan-only`** ("declare, plan every weight's encoding, print the plan, and stop before writing
    anything") and `--report N`.
  - The plan still runs the architecture plugin's declare and checks that every op resolves to a kernel.
- **No `deepseek41` architecture:** `/opt/radiance/share/radiance/architectures/` has 30 `.so` files and none of them
  is deepseek41. **`rad-convert --plan-only` on the release therefore fails in v1.14.0.** Its error is "rad-convert
  runs the architecture plugin's declare ...".
- **v1.14.0's safetensors reader refuses F8_E8M0** (`core/format/safetensors.cpp:139`). The release stores its fp8
  and MXFP4 scales as e8m0. The release reader (`F8_E8M0`, 32x32 e8m0 blocks, MXFP4 experts, `RAD_MAP_MUL`) is also
  only on PR #266.
- **What the image ships:** `/opt/radiance/share/radiance/recipes/deepseek41-ud-iq1s.recipe` (md5-identical to the tag).
- **What it does not ship:** `engram_tables.py`, `imatrix_merge.py`, any python, `/opt/radiance/share/radiance/data`.
  The helpers come from the git tag (`data/`). They need python3 with numpy, and `imatrix_merge.py` also needs `gguf`
  (pip `gguf`). gpu01's host python has neither.

## Converting on gpu01

- **gpu01:** TR 3960X (24 cores/48 threads, 1 NUMA node), 184 GB RAM.
- **Free disk:** 405 GB on `/` and 223 GB on `/data` at 10:55 today. You quoted ~435 + ~140 GB while other downloads
  run. `~/models/ds41-release` and `/data/ds41-release-b` exist but are empty.

### Disk

| item | size |
|---|---|
| release (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, 48 safetensors) | 510.3 GB (475.3 GiB). About 203 GB of it is the fp8 Engram table and 289 GB the MXFP4 experts. |
| output `deepseek41-ud-iq2xxs.rad` | about 321.4 GB (299.3 GiB) |
| imatrix 0.79 GB + merged copy about 1.1 GB, plus Engram tables (<1 MB) | about 2 GB |
| **total at once** | **about 834 GB** |

- **Space is short by about 200-260 GB.**
- **The release does not fit on either filesystem alone.** Shards can be split across `/` and `/data` with symlinks
  in one release dir. In docker, mount both filesystems at the same paths so the symlinks resolve.
- **Ways to make it fit:**
  - free about 260 GB, for example older models on `/data` once the running download has finished;
  - write the .rad to another host (NFS/sshfs);
  - convert somewhere with at least 850 GB free.
- **IQ1_S as well:** converting IQ1_S from the same release adds another 269.6 GB of output.

### RAM and time

- bakon's IQ1_S conversion from the release took **41 GB RSS** and 2 h 21 min on 30 cores. 184 GB RAM is enough.
- Rough IQ2_XXS estimate on the 24-core 3960X: **about 2.5-4 h** wall clock. The searches differ from IQ1_S's; the
  q5_K Engram table without an imatrix takes the cheap `_ref` path, while iq2_s and iq3_xxs searches are comparable to
  iq2_xxs. This estimate is not measured.

### Commands

These follow `docs/deepseek41-hf-direct.md` and need a build with the plugin, which is PR #266 today.

```
# 0. a rad-convert with the deepseek41 plugin AND the release reader: PR #266 (refs/pull/266/head, e30ab83,
#    82 commits on 4374bea, which is v1.14.0 minus 4 commits). The 1.14.0 image cannot do it.

# 1. Engram hash tables. Reads only config.json + tokenizer.json, so it can run anywhere with python3 + numpy.
python3 data/engram_tables.py <release dir> engram-tables.safetensors

# 2. imatrix with the fused gate_up entries (needs numpy + gguf).
#    NOTE: the script reads argv[1] (in) and argv[3] (out). The doc's 2-argument form raises IndexError.
curl -LO https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/imatrix_unsloth.gguf_file
python3 data/imatrix_merge.py imatrix_unsloth.gguf_file unused imatrix-merged.gguf

# 3. chat template: the release ships none. The GGUF's template uses dict(), which Radiance cannot render.
#    Use bakon's plain chat template from PR #266 docs/deepseek41-wip.md (no tools); for tools, PR #340 (DSML).

# 4. plan first (prints the plan, writes nothing)
rad-convert <release dir> --add engram-tables.safetensors \
    --recipe ~/gpu/ds41/deepseek41-ud-iq2xxs.recipe --imatrix imatrix-merged.gguf \
    --tokenizer <release dir>/tokenizer.json --chat-template chat_template.jinja --plan-only

# 5. convert
rad-convert <release dir> --add engram-tables.safetensors \
    --recipe ~/gpu/ds41/deepseek41-ud-iq2xxs.recipe --imatrix imatrix-merged.gguf \
    --tokenizer <release dir>/tokenizer.json --chat-template chat_template.jinja \
    -o deepseek41-ud-iq2xxs.rad
```

The equivalent in docker, once an image with the plugin exists:
```
docker run --rm -v <release dir>:<release dir>:ro -v /data:/data -v ~/gpu/ds41:/w --entrypoint rad-convert <image> \
  <release dir> --add /w/engram-tables.safetensors --recipe /w/deepseek41-ud-iq2xxs.recipe \
  --imatrix /w/imatrix-merged.gguf --tokenizer <release dir>/tokenizer.json \
  --chat-template /w/chat_template.jinja -o /data/deepseek41-ud-iq2xxs.rad
```

## Open issues

1. **Where the `deepseek41` plugin lives**
   - It is on Codeberg PR #266 "deepseek41: DeepSeek-V4.1-Flash on Radiance (TP3/TP4 measured)": branch
     `deepseek41-wip` by bakon, open, explicitly "Not for merge", last updated 29.10.
     - It is a working/umbrella branch with `arch/deepseek41`, the release reader, gfx1201 kernels and
       `docs/deepseek41-wip.md`.
   - It is not in v1.14.0, `main` (8df3947, 29.10) or any tag.
   - The planned upstream split is #242 (main compile) -> imatrix `.weight` fix (#267, merged) -> host-gather
     read-ahead -> release reader + recipe -> `arch/deepseek41` on existing kernels -> kernels, one PR per family.
     - `docs/deepseek41-hf-direct.md`, the recipe and the helpers came in #313, which is merged and in v1.14.0.
     - PR #340 (DSML reply format) is split out of #266 and still open.
   - No ETA is stated. bakon was uploading his IQ1_S container as `bakon3/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-rad`; the
     repo is not public yet.
2. **Kernel gap for the IQ2_XXS mix (blocker).**
   - On #266 the ggml layout, pack/dequant (`lq_ggml_planes`, `lq_ggml_dequant`), the libref host row and libr4d
     (`r4d_ds41_ggml`: `GG_Q8_0, GG_Q5_K, GG_Q6_K, GG_IQ4_NL, GG_IQ2_XXS, GG_IQ1_S`) cover only q8_0, q5_K, q6_K,
     iq4_nl, iq2_xxs, iq1_s and mxfp4.
   - The IQ2_XXS mix needs **iq2_s, iq3_xxs, iq2_xs and q2_K** for routed experts. rad-convert's declare refuses a
     recipe whose ops resolve to no kernel ("N op(s) resolved to no kernel in any domain"), so the conversion and
     even `--plan-only` are expected to fail until those four types get:
     - pack/dequant in libquant,
     - a libref row and layout tag,
     - libr4d MoE GEMMs for speed.
   - This comes from reading the code; it has not been run.
   - None of the supported types lies between iq2_xxs (2.06 bpw) and iq4_nl (4.5 bpw), so a mix limited to them would
     not be the UD-IQ2_XXS mix.
3. **VRAM:** the IQ2_XXS trunk is 154 GiB at block density, more than the 128 GiB of 4 x R9700. Routed experts must
   be partly host-tiered (`--placement expert_tiered`), which slows decode. IQ1_S's 114 GiB fits.
4. **`imatrix_merge.py` argument bug.** The tag's doc says `imatrix_merge.py IN OUT`, but the code uses
   `sys.argv[1], sys.argv[3]`. Pass three arguments.
5. **The imatrix has no `attn_output_a.g<K>` entries.**
   - The tag's `imatrix_merge.py` adds only `ffn_gate_up_{exps,shexp}`. bakon's imatrix also had
     `attn_output_a.g<K>` = columns `[K*4096, (K+1)*4096)` of `attn_output_a`'s in_sum2, which is 32768 wide.
   - Without them, the q6_K `attn_output_a` in layers 27-39 is quantised without an imatrix (`_ref` path), so it is not
     byte-identical to the GGUF. q8_0 in layers 0-26 is unaffected.
   - The fix is small: slice the in_sum2 and copy the counts in the merge script.
6. **Engram table type:** the IQ2_XXS recipe makes it q5_K (126 GiB), against IQ1_S's iq4_nl (103 GiB). With
   `--ngram-placement disk` it stays on the host or disk, so this adds 23 GiB of disk and page cache, not VRAM.
   Swapping in iq4_nl would save that but departs from the UD mix.
7. **Chat template:** the release ships none. The GGUF's template uses `dict()`, which Radiance refuses at load. Use
   bakon's template (no tools) or wait for #340.
