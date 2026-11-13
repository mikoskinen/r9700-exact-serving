# R1: a prefix-cache resume can restore a snapshot on top of KV bytes it was not computed from

Status 10.11.2026: VERIFIED (item hcz-resume2, 15:47 UTC). Not posted anywhere.
- **Cause verified:** resumes differ exactly where a short request published the head first. Three readings, two of them byte-identical to the original bad answer. When no short request came first, the resume is exact.
- **Fix verified:** with the fix below, every hit equals its cold run, and cold output is unchanged.

## Symptom
A model with snapshotted state (linear / recurrent state, sliding window, compressor tail) runs at temperature 0 with `--prefix-resume` exact (the default here). A prompt served from the prefix cache can give a different answer than the same prompt served cold, even though cold prefill is deterministic.

Observed on DeepSeek-V4.1-Flash, EXL3 2.25 bpw, TP4, chunk 2048, 136k prompt:
- **Bad hit:** the resumed answer diverges at token 41. The logprob of token 0 already differs by 1.5e-4, so the restored state differs; this is not a near-tie flip.
- **Good hits:** six other hits in the same server were bit-identical to cold over all 512 logprobs.

## Mechanism
**Blocks: the first publisher's bytes win.** `core/mem/prefix.cpp:456`, `PrefixCache::insert`:
```cpp
if (it != map_.end() && (it->second.in_lru || has_linear_)) continue;
```
With snapshots (`has_linear_`), an existing entry always keeps the bytes it was first published with. A later request that recomputed the same blocks never replaces them.

**Snapshots: the requester's own bytes.** They are saved from the live sequence (`core/engine.cpp:1072` / `:1421` `save_checkpoints` → `core/mem/kv.cpp:1591-1715` `checkpoint_copy` / `checkpoint_save`). A snapshot at position P is therefore computed from the requester's copy of blocks 0..P. The cache keeps the first publisher's copy of those blocks.

**How the two copies differ.** When the first publisher computed those blocks in a different prefill shape, its bytes can differ from the requester's. For example, a 27-row one-chunk prefill against a 2048-row chunk: the GEMM picks among M-bands, so rounding depends on M. A hit then restores the snapshot (requester's history) next to the blocks (first publisher's bytes). The state matches neither run.

**Head entries collide by construction.** Every chat prompt starts with the same template head (BOS, role token). Any short request, such as a warm-up, a title call or a quota probe, publishes those first blocks before a long prompt arrives.

**The authors guarded only the other order.** Test `tests/mem_test.cpp:4756` `a_recomputed_shared_prefix_keeps_its_original_host_copy_and_snapshot` covers a long conversation first and a short recompute later, so the original bytes under the long snapshot stay. The comment at `prefix.cpp:452-455` gives the reason: "a recompute in another step shape can round the KV differently". The case "short publisher first, long snapshot writer later" is not guarded. In that case the snapshot rests on bytes the cache does not hold.

**Related (unverified).** `PrefixCache::reserve_checkpoint` (`prefix.cpp:869-879`) hands an existing snapshot slot at the same prefix to the next saver. So snapshots are last-writer-wins while blocks are first-writer-wins. A second long request that recomputes a shared prefix in another chunk alignment could produce the same mismatch with no short request involved.

**Reset does not clear it.** `PrefixCache::clear()` (`prefix.cpp:770`) drops entries but keeps snapshots. After `/reset_prefix_cache`, the next long prompt becomes the first publisher of the head, in the same 2048-row shape every later long prompt uses. So everything after the reset resumed exactly in the reproduction.

**Scope.** The code is core (`core/mem/prefix.cpp`). Model plugins only declare their state groups (DeepSeek-V4.1: `deepseek41.cpp:1130-1156`). The same line is present in the 1.15.1 tree (`fn4/radiance-1151`, `prefix.cpp:456`) and in fork base acc9ae58 (upstream 1.15.0 merge). Any hybrid model is exposed whenever a short request precedes a long prompt that shares its head (Qwen3-Next GDN state included).

## Minimal reproduction
Any model with a linear / window snapshot group, prefix cache on, checkpoints on, chunk C (here 2048), temperature 0, logprobs on.
1. Start the server.
2. Send a short chat request A (fewer tokens than C) that shares the chat-template head with P.
3. Send a long prompt P, at least 2 checkpoint intervals long, cold. Keep the text and logprobs (cold reference).
4. Send P again. It hits (cached ≈ len(P) rounded down to a checkpoint) and differs from step 3, already at token 0's logprob.
5. Control: `/reset_prefix_cache`, then P cold, then P as a hit. Text and logprobs are identical.

Reproduction run hcz-resume (gpu02, 10.11.2026), in order:

| step | request | cached tokens | output md5 |
|---|---|---|---|
| warm-up | 3 short questions | – | – |
| d1 | deep 136216 tokens, cold | 0 | fc895faf |
| d2 | deep, hit | 133120 | bfbea1e2 (DIFFERENT) |
| d2b | deep, hit | 133120 | bfbea1e2 (DIFFERENT) |
| s1 | swap prompt, 129024 shared with deep, hit | 129024 | differs from s2 (cold) |
| reset | | | |
| s2 | swap, cold | 0 | |
| d3 | deep, hit on s2's chain | 129024 | fc895faf (IDENTICAL) |
| n1, n2, n3 | nonce 136k cold, then 2 hits | | identical |
| m1–m4 | 36k and 74k, each cold then hit | | identical |
| copy | 28k cold, then hit | | identical |
| d4 | deep, cold again | 0 | fc895faf (identical to d1 and to a reference from another server start) |

## Fix sketch (implemented for testing: branch ds41-resfix, cee0ee3 on 3c33393)
**What the fix does** (`core/mem/prefix.{h,cpp}`, `core/mem/kvtier.h`):
- **Republish:** at insert, an existing entry at or below the publisher's deepest committed snapshot takes the publisher's blocks, when all three hold:
  - no committed snapshot already rests on the entry's current bytes ("pinned");
  - no live sequence has adopted the entry;
  - no tier transfer for it is in flight (new `IdleTiers::moving()`).
  The old host copy is dropped (`rebound`) and the new blocks are queued for copy-out.
- **Pinning:** an entry is pinned when a committed snapshot rests on its bytes. Pinned entries keep their bytes, which preserves the existing test's guarantee.
- **Instrumentation:** `Stats::republished` plus one INFO line per republish.
- **New tests:**
  - `a_head_a_short_prompt_published_first_takes_the_bytes_a_longer_prompts_snapshot_rests_on`
  - `a_head_a_live_sequence_adopted_keeps_its_bytes_for_that_sequence`

**Effect:** cold output is unchanged. A hit now restores state equal to the cold run's.

**Alternatives** (not built):
- Key snapshots by the identity or epoch of the block copy they were computed from, and use a snapshot only with that copy.
- On a hit whose snapshot and blocks come from different publishers, fall back to the deepest point where they agree and recompute from there.

## Verification plan (hcz-resume2, about 23 min)
**S1** (warm-up as in the reproduction):
- a1: nonce 136k cold, then a2 its hit;
- b1: deep cold, then b2 its hit.
- Runs with `RADIANCE_DEBUG_TIER_VERIFY` on.

**S2** (no warm-up):
- f1: deep cold as the very first request, then f2 its hit;
- reset;
- the three short warm-up questions;
- f3: deep cold, then f4 its hit.

**S3** (the fix image): the same as S1, and h1 is compared with d1.

**H3 is verified if:**
- a2 ≠ a1, b2 ≠ b1, f2 = f1, f4 ≠ f3;
- b2 and f4 are byte-identical to the reproduction's d2.

**The fix is verified if:** in S3 every hit equals its cold run, and the cold answer equals d1.
