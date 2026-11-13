# Xeon prefetch manifest (/data RAID0)

All LFS files sha256-verified against the HF LFS oid (GitHub digest for release assets). Scripts/logs: ~/gpu/xeon-bench/prefetch/ on xeon.

## C3 - Qwen3.8-27B-FP8 + DFlash2 drafter (vLLM) - READY
- HF cache root: `/data/huggingface` (layout: hub/models--*/{blobs,snapshots,refs})
  - `Qwen/Qwen3.8-27B-FP8` rev `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (29G, 80 files incl. mtp.safetensors, tokenizer, chat_template.jinja, preprocessor configs)
    snapshot dir: `/data/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
  - `z-lab/Qwen3.8-27B-DFlash2` rev `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (3.6G, model.safetensors bf16)
    snapshot dir: `/data/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- Usage: `docker run ... -v /data/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 ... --model Qwen/Qwen3.8-27B-FP8` and drafter `z-lab/Qwen3.8-27B-DFlash2`.
  Verified offline resolution with snapshot_download + AutoConfig inside the vllm-sly-radiance image (arch Qwen3_5ForConditionalGeneration).

## C4 - Qwen3.8-27B UD-Q8_K_XL GGUF + llama.cpp mainline (HIP + Vulkan) - READY
- Model: `/data/models/C4/Qwen3.8-27B-UD-Q8_K_XL.gguf` (31457991680 B, sha256 af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377)
- mmproj: `/data/models/C4/mmproj-F16.gguf` (927607488 B, sha256 cbb841a9...4e43e, copied from local HF cache; matches HF oid)
- llama.cpp source: `/data/build/llama.cpp` master commit `a9d27ac693ff` (2026-10-11), version 0.6.0-dev build 50. `-sm/--split-mode {none,layer,row,tensor}` present in both builds.
- Image `llamacpp-main-hip:gfx1201`: FROM rocm/dev-ubuntu-24.04:7.2.4-complete, binaries in /app (PATH, LD_LIBRARY_PATH set), ENTRYPOINT /app/llama-server. Built -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON, no RCCL.
  Host copy of binaries: `/data/build/llama.cpp/build-hip/bin` (also /data/build/img-hip/bin).
  Example: `docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -v /data/models/C4:/models:ro llamacpp-main-hip:gfx1201 -m /models/Qwen3.8-27B-UD-Q8_K_XL.gguf --mmproj /models/mmproj-F16.gguf -ngl 999 -sm tensor --host 0.0.0.0 --port 8000`
  (other tools: `--entrypoint /app/llama-bench`).
- Image `llamacpp-main-vulkan:latest`: FROM ubuntu:24.04 + mesa-vulkan-drivers (RADV) + vulkan-tools, -DGGML_VULKAN=ON -DGGML_NATIVE=ON, binaries in /app, ENTRYPOINT /app/llama-server.
  Host copy: `/data/build/llama.cpp/build-vk/bin`. Run with `--device /dev/dri` (no /dev/kfd needed).
- Note: CPU is Xeon E5-2670 v1 (Sandy Bridge, AVX only, no AVX2/FMA). The first builds used GGML_NATIVE=OFF, which defaults to AVX2/FMA and crashed with SIGILL (rc 132) on model load; both images were REBUILT at ~14:35 with GGML_NATIVE=ON and verified (llama-tokenize loads a GGUF, rc 0). Any image built before 14:35 is bad. Current good IDs: llamacpp-main-hip:gfx1201 = 41a47e504ee5, llamacpp-main-vulkan:latest = 3e8e52628400.

## C5 - Strata v0.1.41 - READY (built from source)
- IMPORTANT: Strata v0.1.41 GitHub release has only Windows assets (strata-windows-x64{,-hip,-cuda12}.zip); Linux is install-from-source (setup.sh). So the tag was cloned and the HIP engine compiled for gfx1201.
- Path: `/data/models/C5/Strata-v0.1.41` (git tag v0.1.41 = commit fb58e0dbc839)
- Engine: `/data/models/C5/Strata-v0.1.41/engine/strata` + `engine/BUILD.json` (backend hip, archs [gfx1201], isa_floor "avx" - setup auto-picked the AVX floor because this Xeon lacks AVX2; lib_dirs /opt/rocm/lib). Built exactly via setup.build_engine_hip() so setup.py recognises it ("engine already built") and will not recompile.
  llama.cpp pinned source (3cf03257) unpacked in third_party/llama.cpp. Vision encoder NOT built (AMD = CPU vision only).
- Python venv: `.venv` (python 3.12, requirements.txt installed) created INSIDE image `pf-rocmdev` - host python3.14 lacks ensurepip, so `./setup.sh` on the host would try sudo apt. Run Strata inside the container:
  `docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video --ipc=host --ulimit memlock=-1 -p 8080:8080 -u $(id -u):$(id -g) -e HOME=/tmp -v /data/models/C5:/data/models/C5 -v /home/mikoskinen/models/flashnext:/home/mikoskinen/models/flashnext:ro -w /data/models/C5/Strata-v0.1.41 pf-rocmdev ./setup.sh --backend hip --gpus all ...`
  (image `pf-rocmdev` = rocm/dev-ubuntu-24.04:7.2.4-complete + cmake ninja git python3-venv). `engine/strata --help` runs without a GPU (verified with no devices passed).
- Model: reuse C1 files in `/home/mikoskinen/models/flashnext/` (no download). Strata needs a prepared "pack" (setup.py builds it from the GGUFs: use `--gguf-dir /home/mikoskinen/models/flashnext --data-dir /data/models/C5/Strata-data`, and `--model` matching UD-IQ3_XXS) - first setup run will spend time/disk on that; point its pack dir at /data if large.

## C7 - Ornith-1.5-35B-A3B Q8_0 GGUF + mmproj - READY
- Model: `/data/models/C7/Ornith-1.5-35B-Q8_0.gguf` (37802149280 B, sha256 de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3), from ornith-ai/Ornith-1.5-35B-A3B-GGUF (the repo filename is `Ornith-1.5-35B-Q8_0.gguf`, no "A3B" in it)
- mmproj: `/data/models/C7/mmproj-Ornith-1.5-35B-BF16.gguf` (902822240 B, sha256 1921a36a...a135d; the official repo ships only a BF16 mmproj)
- GGUF metadata: arch `qwen35moe`, tokenizer pre `qwen35`, 753 tensors; mmproj arch `clip`. qwen35moe is supported by llama.cpp a9d27ac (images from C4).
- Usage: `-v /data/models/C7:/models:ro ... -m /models/Ornith-1.5-35B-Q8_0.gguf --mmproj /models/mmproj-Ornith-1.5-35B-BF16.gguf`

## C6 - CANCELLED (coordinator: IQ3_XXS already reaches 262k). Nothing downloaded; /data/models/C6 does not exist.

## C8 - JetBrains Mellum2.1-12B-A2.5B-Thinking (replaces C6) - READY
- GGUF dir: `/data/models/C8-mellum/`
  - `Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` (12925417760 B, sha256 69e0ac02a16586c2c9e991f1e3b8a1eb403882e7ab9e69ea01eb23dbbbc6841f)
  - `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf` (7029537056 B, sha256 cf09584a30423f6f80b0c9fe2f42e152520dabf08fecd1cad641ce21a7588166)
  - GGUF metadata: arch `mellum`, tokenizer pre `mellum2`, 339 tensors. llama.cpp a9d27ac supports both (LLM_ARCH_MELLUM in src/llama-arch.cpp, src/models; LLAMA_VOCAB_PRE_TYPE_MELLUM2), and llama-tokenize in both images loads the Q8_0 GGUF fine (CPU only, no GPU used).
- Safetensors (vLLM) in the HF cache: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` rev `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4` (23G, 5 shards + configs/tokenizer/chat_template.jinja)
  snapshot dir: `/data/huggingface/hub/models--JetBrains--Mellum2.1-12B-A2.5B-Thinking/snapshots/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`
  Checked offline: model_type `mellum`, architectures [MellumForCausalLM]; transformers in the vllm-sly-radiance image (vLLM 0.30.0) loads the config, and that vLLM has model_executor/models/mellum.py plus a registry entry.
  Usage: `-v /data/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 ... --model JetBrains/Mellum2.1-12B-A2.5B-Thinking`
