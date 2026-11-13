# RCCL without hostcall on xeon (no PCIe atomics), 2026-10-11

## Result
- **Image:** `vllm-sly-radiance:1.0.0-rocm10.1-rccl-nohostcall` (local tag on xeon, built from
  `ghcr.io/slybase/vllm-sly-radiance:1.0.0-rocm10.1`). Build context: `~/gpu/rccl-nohostcall/ctx/`.
- **Launch script (not started):** `~/gpu/run/radiance-tp2-nohostcall.sh`. It is identical to `radiance-tp2.sh`
  except for the image and the container name (`radiance-tp2-nh`). Defaults: TP=2, port 8000, NCCL_P2P_DISABLE=1.
- **Hostcall-free:** yes. All 85 RCCL gfx1201 code objects parse, and `hidden_hostcall_buffer` drops from 13 to 0.
- **2-GPU all_reduce works without PCIe atomics.** Results are correct at every size, both eager and in a HIP graph.

## What the image contains (why there was no 2.27.7 rebuild)
- The image runs ROCm 10.1 (HIP 7.16.26385) and torch 2.12.0+rocm10.1. torch loads the system
  `/opt/rocm/core-10.1/lib/librccl.so.1`, which is **RCCL 2.30.7**. torch has no bundled copy.
- ROCm 10 uses the split "kpack" layout. The `.hip_fatbin` section in `librccl.so` is NOBITS, and the device code
  lives in `/opt/rocm/core-10.1/.kpack/rccl_lib_gfx1201.kpack`. That file holds 85 zstd-compressed code objects
  plus a msgpack TOC.
- A RCCL 2.27.7 rebuild is a poor fit here:
  - torch 2.12 imports `ncclCommSuspend`, `ncclCommResume` and `ncclCommMemStats`, which 2.27.7 does not have.
  - 2.27.7 links rocm_smi, but the image only ships amd_smi.
  - NDEBUG does not help on 2.30.x, because the device linker still declares the hostcall argument.
- **Approach instead: metadata patch of the shipped device code. `librccl.so` itself is unchanged, so the ABI is exact.**
  - In the 13 affected kernels (`ncclDevKernel_Generic_{1,2,4}` and 10 `ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_*`),
    the AMDGPU metadata kernarg kind `hidden_hostcall_buffer` is rewritten to `hidden_global_offset_x`.
  - Both strings are 22 bytes, so every msgpack length and kernarg offset is byte-identical.
  - The CLR runtime then writes 0 into that 8-byte slot instead of refusing dispatch with
    "Pcie atomics not enabled, hostcall not supported".
  - Only the 11 code objects that contain the string are recompressed. `zstd_size` and the TOC offset are updated.
- **Safety:** the only hostcall code in these kernels is device `assert()` → `__assert_fail` → `__ockl_fprintf`
  (stock 2.30.7 is built without NDEBUG). It runs only when an assert fails. With the patch, a failing assert
  becomes a GPU memory fault instead of a printed message followed by a trap. Either way the process dies, so the
  normal path is unchanged.
- **Tools** (also copied into the image under `/opt/rccl-nohostcall/`):
  - `~/gpu/rccl-nohostcall/kpack_nohostcall.py in.kpack out.kpack`
  - `verify_kpack.py` checks the archive with the official `librocm_kpack` reader plus `llvm-readelf`.
    Original: `hidden_hostcall_buffer=13`. Patched: `hidden_hostcall_buffer=0`, metadata_parse_fail=0.
- The original files are kept in `~/gpu/rccl-nohostcall/orig/`.

## Validation (one run, 11:52 UTC, both cards idle, `~/gpu/rccl-nohostcall/ar_nohostcall.py`)
Setup: 2 processes, torch.distributed nccl, bf16, NCCL_P2P_DISABLE=1, GPU_MAX_HW_QUEUES=1. Transport is host-staged
(SHM) across the two sockets.

| size | eager latency | HIP-graph latency | bus bw | correct |
|---|---|---|---|---|
| 16 KiB | 130.6 us | 28.6 us | 0.13 GB/s | yes |
| 80 KiB | 133.8 us | 46.3 us | 0.61 GB/s | yes |
| 1 MiB | 334.5 us | 320.9 us | 3.13 GB/s | yes |
| 16 MiB | 5.03 ms | – | 3.34 GB/s | yes |
| 64 MiB | 19.16 ms | – | 3.50 GB/s | yes |

**Caveat:** the test processes **hung at exit** after printing every result (destroy_process_group or process
teardown), and the container had to be killed. If vLLM shutdown hangs the same way, `docker kill radiance-tp2-nh`.

## TP2 viability estimate (Qwen3.8-27B: ~64 layers × 2 all-reduces per forward = 128 ARs; hidden 5120 → 10 KiB/token)
- **Decode** runs under cudagraph FULL_DECODE_ONLY, so the HIP-graph numbers apply.
  - Single stream with DFlash K=7 verifies 8 tokens per step, which is ~80 KiB per AR. That gives
    128 × 46 us ≈ **5.9 ms per step**.
  - Plain M=1 decode would be 128 × 29 us ≈ 3.7 ms.
  - At 8 seqs × 8 tokens (640 KiB/AR) the cost rises to roughly 128 × ~220 us ≈ 28 ms per step, which is bad for
    batched throughput.
  - TP2 halves the weight reads per card, about 7 GB each for MXFP4. A modest single-stream decode gain
    (~1.1–1.4×) is possible, but it is not guaranteed. The draft model's own all-reduces come on top.
- **Prefill** is bandwidth-bound by the 3.5 GB/s host path.
  - A 2048-token chunk is 20 MiB per AR, which is ~6 ms. Times 128 ARs, that is ~0.8 s per chunk, or about
    2,600 tok/s from communication alone.
  - That roughly equals the single-card prefill today (~2,600 tok/s), so expect **TP2 prefill to be slower than single card**.
- **Main benefit:** double the KV/VRAM headroom (64 GB total), for longer context or more concurrent sequences.
  It is not a speed win. PP2 (send/recv uses the same kernels) should also work now. It sends one 10 KiB/token hop
  per step instead of 128 ARs, so it may be the better split for decode.

## How to run the full TP2 benchmark later (GPUs must be free; stop the single-card `radiance` first)
```
ssh xeon 'docker stop radiance; nohup ~/gpu/run/radiance-tp2-nohostcall.sh > ~/gpu/log-tp2-nohostcall.txt 2>&1 &'
# wait for "Application startup complete" in the log (~6-10 min), then:
ssh xeon 'python3 ~/gpu/bench.py --url http://127.0.0.1:8000 --depths 0,16384,65536 --gen 256'
# PP2 variant: TP=1 PP=2 ~/gpu/run/radiance-tp2-nohostcall.sh
```
Re-run only the RCCL test: `MAXWAIT=1 ~/gpu/rccl-nohostcall/wait_and_test.sh` (needs ≥2 GiB free on both cards).
