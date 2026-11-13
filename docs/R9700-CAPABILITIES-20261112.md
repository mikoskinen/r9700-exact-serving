# R9700 (gfx1201, Navi 48) capabilities our engine does not use yet -- 12.11.2026

Research only: nothing ran on the GPUs, no container or queue was touched. Live facts are read-only sysfs reads
on gpu02 (~18:40 FI). Source trees read: Qwen line `fn4/src-c16x3p2` (EXL3 Flash-Next tree), DeepSeek line
`ds41/exl3/radiance-int12-f4g`, EXL3 kernels `fn4/exl3-main37/kernels/exl3r4d`. Every gain below is an ESTIMATE
with its ms basis; nothing here is measured by me.

## 0. Corrections to the brief (checked against code and LEVERS)

1. **The prefill all-reduce numbers are DeepSeek, not Qwen** (LEVERS:218 PA, :219 MAR; profile dsm-225-prof),
   and they predate the 12.11. riser move (83:00 and 86:00 then shared one root complex).
   The "10.7 GB/s each way" is one phase's bytes over the whole call. `r4d_ar_twoshot_Nrank_exact.hip:1-4`:
   the two-shot "sends 2*(ws-1)/ws*N bytes per rank in two hops" = 2 x 3/4 x 42 MB = 63 MB a rank ->
   63 MB / 2.927 ms = **21.5 GB/s per direction = 81-84 % of the 25.5-26.4 GB/s pair rate**. The coordinator's
   reading is right. The DeepSeek AR is close to the Gen4 link already.
2. **Qwen's AR lane is mostly hidden.** At 128k, lane 1 is idle 8193 ms of 15.04 s waiting for lane 0.
   Exposed: ~332 ms of waits before the last piece's write, plus ~340 ms of per-op issue gaps on lane 0
   (~4.4 us an op, profiled). Together that is 672 ms = 4.6 % of the 14.64 s. At 16k x4, single-lane ARs
   cost 444 ms (~8 %) (LEVERS:60-68, P5 :77).
3. **The "+28 % hc kernels beside the AR lane" is mostly clock.** The two runs ran ~1.2x apart in clock
   (2254-2375 vs ~2786 MHz). After that, co-running costs ~0-6 % on the hc kernels and ~0 % elsewhere.
   LEVERS:57 bounds priority or fewer-CU tricks for the AR at ~0.1-0.2 s at 128k.
   Two AR-lane retunes both lost ~3 % (LEVERS:47 lane width; 133 blocks). So the lane's throughput is
   exposed, not its interference.
4. **Chained launches.** What they are (`libr4d/r4d_chain.h:1-12`, `core/device/aql.cpp:1107`):
   - A chained kernel packet goes out without the AQL barrier bit.
   - Thread 0 of every block spins with `s_sleep` until the queue's device completion counter reaches the
     blocks issued before it, and every block adds 1 when it ends.
   - Waves of the next kernel therefore sit resident on the CUs, replacing the packet processor's wait.

   Results:
   - Qwen 145 (10.11.), restarts ABBAAB: c16 0.934, c8 0.971, c4 1.003. LOST; identity identical (LEVERS:86).
   - DeepSeek: never measured. The core enables chaining only while drafting (LEVERS:255 1U-CHN, :312).
5. **AQL acquire rule.** What it is (`aql.cpp:1091-1105`, TapeRec.noacq): the CP's per-dispatch agent-scope
   acquire (the L0/L1/scalar-cache invalidate) is dropped from replayed dispatches whose inputs cannot be stale.

   Results:
   - The plain rule gave stale reads on gfx1201 (1.2-72 M stale elements over 300 trials): FUNDAMENTAL,
     closed (LEVERS:311).
   - rule-w on Qwen: no gain (146: c16 1.0044 +- 0.0092, c8 0.9973; LEVERS:87).
   - rule-w on DeepSeek: c1 +0.82 % (t95 0.47), c4 +1.09 %, c8 +0.52 %, identity identical. **It is IN the
     DeepSeek stack** (aql_acq 2; LEVERS:179). So "both were tried and lost" is only half right.
6. **The lossless coded AR wire is already parked.** It was byte-identical and -15..-17 % in isolation, but
   -0.33 % (135k) / -2.17 % (64k) on the whole prefill (LEVERS:235). Faster AR bytes alone barely reach
   the DeepSeek step.

## 1. Live facts, gpu02 (read-only)

| item | value |
|---|---|
| link | Every card: upstream port 01/46/81/c1:00.0 at **16 GT/s x16** (Rome root port). Card-internal switch link 32 GT/s. Max 32 GT/s, so the card is Gen5-capable. |
| BAR | BAR0 32 GB on all four (ReBAR on) |
| amdgpu params | pcie_p2p=Y, cwsr_enable=1, uni_mes=1, sched_policy=0, sdma_phase_quantum=32, ppfeaturemask=0xffffffff, iommu=pt (cmdline) |
| firmware | MEC 0xcda, MES 0x89, SDMA 0x798e96, SMC 0x684c00, VBIOS 113-EXT119250-100; linux-firmware 20260319 |
| clocks (snapshot, under load) | sclk 2288-2350; **fclk 2016 MHz (level 1 of 313/2016/2400)**; mclk 1258 (top level); socclk 1280 of 1476 |
| PCIe DPM | levels 16 GT/s x16 at LCLK 616 and 1143 MHz; both marked current (cannot tell which). No pinning on gpu02 (xeon pins level 2, TODO.md:9). |
| power profile | **BOOTUP_DEFAULT** active (COMPUTE = index 5 available) |
| OD | VDDGFX offset **-50 mV** (range -200..0); MCLK OD range up to 1500 MHz (now 1259); SCLK offset 0 (-500..+1000); cap 210 W (max 330, default 300) |
| write access as user | render group: pp_od_clk_voltage, pp_power_profile_mode, power_dpm_force_performance_level, power1_cap. Root only: pp_dpm_pcie, pp_dpm_fclk, pp_dpm_mclk. |
| host memory | THP madvise; AnonHugePages 173 GB (host pools are 2 MB-backed); shmem THP never (Shmem 2.5 GB) |
| not readable without root | PCIe DevCap2/DevCtl2 (10-bit tags, MPS, MRRS, Relaxed Ordering), debugfs (MES/HQD state) |

## 2. Ranked table -- exact candidates, by what can still reach the step

Exact = no change in arithmetic or summation order; batch invariance unchanged.

| # | capability | what it does | status here | exact? | bottleneck hit (measured) | gain estimate (est.) + ms basis | build+test (agent h) | needs the user? |
|---|---|---|---|---|---|---|---|---|
| 1 | Deeper VDDGFX offset at the same 210 W | More sclk for the same power. Xeon: -90 mV gave ~+200 MHz average sclk at the 210 W cap vs 0 mV (xeon-bench/od-tuning.md:31). | Partly used: -50 mV on all four. -70/-90 mV byte-identical on xeon 12.10. (8 min each, od-tuning.md:39). A later reset on xeon's hot card was suspected on -90 mV, and the setting went back to -50 (STATUS.md:220, :263). Never tried below -50 on gpu02. | yes (arithmetic); instability is the risk -> identity + selftest gate | both: prefill is clock-bound. Cards run 2254-2375 MHz at the cap vs ~2786 cold, ~1.2x on compute-bound kernels (LEVERS:57). | -50 -> -90 mV ~ +90 MHz (+3.9 %; linear from xeon). With ~70 % of lane-0 time clock-sensitive (MoE GEMMs 20.9 % are bandwidth-bound): **Qwen 128k -0.2..-0.55 s of 14.64 s (-1.5..-3.8 %)**; **DeepSeek 128k step -1.5..-3 %**; decode at the cap similar; DeepSeek 1-user ~0 (not power-limited) | 0 build; 1.5 (per-step soak + identity + prefill A/B) | no sudo (render group). Production stability risk: say it and log it. |
| 2 | PCIe 10-bit tags (plus MRRS) for in-kernel host reads | More outstanding read requests on the link. Zero-copy expert reads are cache-line reads, so they need many tags, while the DMA engine uses large requests. | UNKNOWN: DevCap2/DevCtl2 cannot be read as a user | yes (link protocol only) | DeepSeek: cold-expert stream. Misses cost 0.0135 ms a streamed MB (i9p, LEVERS:241). A cold 1-user essay at 185 MB/step = ~2.5 ms of the ~20.8 ms step, which is the whole 75.7 vs ~86 tok/s gap. The equivalent rate is ~18.5 GB/s a link (MCL row assumes 12.7 GB/s at c8) vs 28.7 GB/s for the engine's own DMA queue (`aql.cpp:2562-2580`). | Only if tag-limited (est.): 1-user cold -0.4..-0.7 ms a step (+2..+4 % tok/s); c8 decode stream (~55 MB a rank a step) up to -1..-2 ms of 36.5 ms. **0 if already enabled or if Rome's root ports lack a 10-bit completer.** | 0.1 to know (one `sudo lspci -vvv`); enabling: kernel/BIOS | sudo read; then BIOS / kernel param / reboot |
| 3 | GFX12 per-instruction SCOPE bits in the AR handshake instead of per-wave whole-cache fences | **Prefill (4 ranks, >= 131072 elements): drain 1 is forced** (`r4d_rows.cpp:11389-11410` Qwen, `:11608-11629` DeepSeek). That puts an all-thread `__threadfence_system` = `global_wb scope:SCOPE_SYS` + `global_inv scope:SCOPE_SYS` in every wave, twice a call. **Prefill and decode: acq 0 = `global_inv scope:SCOPE_SYS` in every wave** (`r4d_ar_wide.h:57-61`); decode keeps drain 3. The engine's own note: "an L2 writeback-invalidate per wave" (`r4d_ar_oneshot_2rank_exact.hip:104`). LLVM GFX12 model: `global_inv` invalidates every cache below its scope, including the L2. Lever: write-through SCOPE_SYS stores for the push plus SCOPE_SYS loads of the scratch, so no L2-wide write-back or invalidate is needed. | not used. Drain 1 was added for an unexplained stale read at the first reuse of a scratch slot on a fresh server. | yes (same bytes, same fp32 ascending-rank fold). Correctness risk = stale reads: must pass a cohbench2-style stress + identity. | Qwen: lane-0 co-running ~0-6 % of the hc kernels (LEVERS:57). DeepSeek: lane-1 overlap lost to contention; ops beside an AR x1.48 vs control x1.10 (L7, LEVERS:194). Both: decode AR latency (Qwen 97 x 89.7 us a c16 step, ~15 us of it link; DeepSeek 40 x 23.3 us + gather ARs). | Qwen 128k -0.1..-0.3 s (0.7-2 %; capped by LEVERS:57). Decode, if each call's fences cost 1-3 us (unmeasured): Qwen c16 -0.1..-0.3 ms a step; DeepSeek 1-user -0.08..-0.25 ms (0.4-1.2 %). DeepSeek prefill: only by reviving L7 (its upper bound ~-180 ms a 1206 ms chunk). Which L2 lines the hardware really drops is not documented, so measure first. | 1-2 microbench (hc_read alone vs beside a kernel issuing SYS wb/inv at the AR's per-wave rate); 4-6 libr4d mode; 1 stress; 1 A/B | no |
| 4 | Queue-to-CP-pipe placement (MES) | Forum report: a queue with a pending wait packet slows small kernels on another queue of the **same CP pipe**. R9700: ~2.5 -> 19.4 us per tiny kernel; waits on another pipe cost nothing (ROCm #6685 / TheRock #9297; MES 0x81-0x93; not AMD-confirmed). Our lane 1 waits on lane 0 with barrier-AND packets (`aql.cpp:1896-1914`). | not controlled, not measured. The engine already avoids HIP's extra default queues (`aql.cpp:4288-4293`); 4 queues a card. | yes (scheduling only) | Qwen: lane-0 issue gaps ~340 ms at 128k (~4.4 us an op). Both: dependent gap 3.88-4.04 us (LEVERS:174) vs 2.3 us in a HIP graph on this part (`r4d_gemm_fp8a8.hip:607`). | Qwen 128k 0..-0.34 s (bound = the measured gaps). DeepSeek decode: our ~3.9 us gap is far below the 19 us penalty case, so no gross penalty now; <= ~0.3 ms a step (est.). Stamping inflation and real-kernel prologues likely explain most of the gap. | 1.5-2 microbench (chain of tiny kernels on an engine-style queue; 1-3 extra queues with a pending barrier-AND; vary creation order); fix 2-4 (creation order / placeholder queue / in-kernel wait) | no |
| 5 | CU masking per queue (`hsa_amd_queue_cu_set_mask`) | Spatial split: AR lane on a few WGPs, compute on the rest. gfx12 KFD writes the mask into `compute_static_thread_mgmt_se0..7` (kfd_mqd_manager_v12.c update_cu_mask); WGP mode needs CUs in pairs. | not used (no call in either tree) | yes (placement). Keep every AR block co-resident: blocks spin on peers. | Qwen co-running (above); DeepSeek L7 contention if it is CU-local | Qwen <= 0.1-0.2 s at 128k (LEVERS:57 bound). The compute lane loses the masked WGPs (4 of 32 = -12.5 % on CU-bound kernels), so it can be net negative. | 2-3 | no |
| 6 | SDMA engines for the collective's P2P bytes | The engine already drives its own user-mode SDMA queue (KFD ioctl, `aql.cpp:2562-2830`): 28.7 GB/s host->card "at no cost to the kernels" vs runtime copy 22.0 GB/s (-8 % on kernels) vs copy kernels 2.3 GB/s. An exact AR would DMA the reduce-scatter and all-gather pushes and run only the fp32 fold on CUs. | not used for P2P (DMA lane = host copies; second engine = expert prefetch) | yes (copies are bit-exact; same fold) | Qwen co-running; DeepSeek serial prefill AR (link at ~82 %) | Qwen <= 0.1-0.3 s at 128k (same bound as #5). DeepSeek: the link stays the limit, so the coded-wire result (-0.33..-2.17 %) is the proxy ceiling. Worse beside DeepSeek's DMA staging (same link, same engines). | 15-25 | no |
| 7 | Queue priority (`hsa_amd_queue_set_priority`) | MES/CP arbitration between queues (MQD `cp_hqd_pipe_priority` / `cp_hqd_queue_priority`) | plumbed (`aql.cpp:1594`), every stream created with 0 | yes | Qwen co-running; lane-0 dispatch order | same bound as #5 (<= 0.1-0.2 s at 128k) | 0.5-1 | no |
| 8 | LDS-only barriers in the hc kernels | On gfx12 `__syncthreads()` emits `global_inv scope:SCOPE_SE` on both sides, dropping L0 lines. The engine's `lds_barrier()` (`r4d_common.h:256-262`) avoids that. | partly used: `r4d_hc_bf16.hip` has 84 `__syncthreads` vs 9 `lds_barrier` | yes (if a barrier orders only LDS) | Qwen: prefill hc_read 4105 ms = 29.6 % of lane 0 at 16-19 % of bandwidth; decode hc_read 7.8-8.8 ms a c16 step at 14-15 % | unknown (not measured); only barriers inside loops that re-read global data matter | 2-4 + byte check | no |
| 9 | Power profile COMPUTE (`pp_power_profile_mode` 5) | SMU14 DPM tuning: GFXCLK MinActiveFreq 600 (type 4), other PD coefficients | not used (BOOTUP_DEFAULT); no history in the notes | yes | both: clock behaviour at the cap and in bursty decode | unknown; likely ~0 at the cap | 0.5 | no (render group) |
| 10 | FCLK pinned at 2400 MHz | Data-fabric clock (L2 <-> MALL/memory path); reads 2016 under load now | not used | yes | both: memory-latency-bound kernels (EXL3 down 56 % load waits, LEVERS:127; hc_read decode 14-15 %). At the cap it takes power from GFX. | unknown | 0.5 | sudo (pp_dpm_fclk root-only) |
| 11 | PCIe DPM pinned at level 2 | LCLK 1143 vs 616 MHz | used on xeon (stall fix, effect never isolated: TODO.md:9-12), not on gpu02 | yes | both: AR handshake latency, host streams | unknown | 0.5 | sudo |
| 12 | Cooler cards (fan target below 80 C) | Less leakage, so more sclk at 210 W | fans 3500 rpm / 80 C; cards ~60 C after the riser move | yes | both: prefill clock | measured: prefill on 30-33 C cards +2.8 % vs warm 85 C (112, LEVERS:53). From ~60 C now: +0.5..+1.5 % (est.) | 0.5 | user (noise preference) |
| 13 | MCLK OD 1259 -> up to 1500 MHz | +19 % raw DRAM bandwidth at the same cap (memory takes power from GFX) | not used, never tried | arithmetic yes; GDDR6 errors at OC can change bytes -> exactness risk | Qwen: MoE GEMMs 2904 ms = 20.9 % of lane 0 at 77-80 % of 640 GB/s. DeepSeek: EXL3 experts 76-83 % of 640, 4.60 ms a 1-user step. | +10 % mclk: those kernels -5..-8 %, minus the GFX clock lost under the cap -> decode +1..+3 %, prefill ~0 (est.) | 1 + long soak | user (risk decision; render-group write) |
| 14 | Compute wave save/restore off (`cwsr_enable=0`) | Hypothesis: the 33 MiB of VRAM per queue (`vram_budget.h:62-65`) is the CWSR save area; x4 queues = ~132 MiB a card | not used | yes | DeepSeek: VRAM for expert units (2948 of 3840 resident) | ~14 more 9.2 MiB units a rank -> <= -0.05..-0.1 ms a 1-user step (est.). Loses preemption / hang recovery. | 1 | sudo + module param + reboot |
| 15 | Wave priority (`s_setprio`) on the critical decode chain | Raises issue priority of the critical chain's waves over overlapped side kernels (503 of 1390 DeepSeek dispatches run unordered) | not used for this; tried around WMMA in a GEMM: neutral (`r4d_gemm_8a8_prefill.h:63`) | yes | DeepSeek decode critical path | unknown; K1's calibration (profile savings x ~0.15, LEVERS:221) suggests small | 2-3 | no |

Already registered elsewhere (not repeated as new):
- LL flag-in-data decode AR (LEVERS:246, est. -0.2..-0.5 ms DeepSeek 1-user, never tried). It relies on 8-byte P2P
  stores arriving whole, a PCIe/hardware property.
- Chained launches on DeepSeek (LEVERS:255).
- DMA-engine next-layer expert prefetch (LEVERS:244: PARKED, 9-10 % of copies used, essay -4.9 %).

Suggested order:
- **Without user input:** #3 and #4 microbenchmarks (~3-4 h together, one card, no engine). They decide whether
  #3 / #5 / #6 / #7 are worth building.
- **Small user actions:** #1 (render group, stability soak) and the #2 read (one sudo lspci).

## 3. Not exact -- listed only, never recommended

| capability | why not exact | rate it would buy |
|---|---|---|
| SWMMAC 2:4 sparse WMMA (`v_swmmac_*`, the only 32-deep fp8 form) | needs 2:4 pruned weights | 2x issue vs dense (gfx1201 has no dense 16x16x32 fp8, `libr4d/isa/README.md`) |
| iu4 WMMA (`v_wmma_i32_16x16x32_iu4`) | needs 4-bit activations as well | issue ceiling 830 vs iu8 407 vs f16/bf16 207 TF/s (`r4d_gemm_w4a8_nt_m64.hip:5-11`) |
| fp8 WMMA for hc_read or dense bf16 GEMMs | quantises bf16 activations; dequant-once bf16 GEMM also not exact (LEVERS:210) | 412 vs 207 TF/s issue; hc_read is at 19-24 % of bf16 peak anyway |
| split-K with atomics / reordered reduction trees | changes summation order | -- |
| expert tensor parallelism at decode (LEVERS:242) | down K-split changes per-rank partial rounding | est. -1..-2 ms DeepSeek 1-user |
| ti8 / wht6 lossy AR wire | lossy | fewer wire bytes |
| MCLK OD (#13) | arithmetic exact, but memory errors under OC can change bytes | see #13 |

## 4. Already used (status: used now)

- Own HSA/AQL runtime (`core/device/aql.cpp`): own queues, kernarg ring in VRAM written through the BAR,
  barrier-bit control with an overlap bit per recorded dispatch, GPU-only signals, interrupts.
- **Recorded passes = the hipGraph equivalent.** A replayed dispatch is a 60-byte packet copy (`aql.cpp:1007-1015`).
  HIP's own graph measured 2.3 vs 2.8 us per dependent dispatch on this part (`r4d_gemm_fp8a8.hip:607`),
  so hipGraph adds nothing.
- User-mode SDMA queue (KFD create-queue ioctl, engine 0) for host->card copy lists. Second engine for the
  expert prefetch (`aql.cpp:2562-2830`).
- Link gate (`rad_link_hold`) between the DMA lane and the collectives.
- P2P + ReBAR (BAR0 32 GB): the exact ARs push into fine-grained peer VRAM (LEVERS:219). Lossless coded wire:
  built, parked (LEVERS:235).
- WMMA: fp8 / iu8 / bf16 / f16 forms.
- `global_load_tr_b128` transpose loads (`r4d_common.h:236-242`).
- VOPD dual issue, with an LLVM pairing patch for repeatable code (Dockerfile llvm stage).
- `s_prefetch_data` on the kernarg (exl3r4d_common.hip.h:800).
- INST_PREF_SIZE: the compiler sets the kernel-start instruction prefetch from code size (`fn4/isa/hcw.s:175`).
- Non-temporal (TH_NT) loads in many GEMMs (`r4d_args.h:321`, 24 kernel files). Not in the EXL3 or hc kernels:
  their operands are re-used through L2/MALL, so NT is not proposed there.
- `s_sleep` in every spin loop (AR, chain, junctions); LDS-scoped barriers in several kernels.
- MADV_HUGEPAGE pinned host pools (173 GB AnonHugePages live); iommu=pt.
- AQL acquire rule-w (DeepSeek stack).
- Queue profiling enabled on every queue (`aql.cpp:1593`; the stamps use it).
- Undervolt -50 mV, 210 W cap.
- Host tuning (C2 off, governor performance, numa_balancing 0).
- One root complex per card (riser move, 4-card H2D 61.5 -> 106.2 GB/s).

## 5. Dead ends / does not exist on RDNA4 / tried and lost

- Does not exist on gfx1201:
  - Dense 32-deep fp8 WMMA (llvm-mc rejects it; `libr4d/isa/README.md`).
  - Vector data prefetch into L2: `global_prefetch_b8` / `llvm.prefetch` is GFX12.5 only and ignored on
    earlier targets (LLVM AMDGPUUsage).
  - XGMI / Infinity Fabric Link: none on the R9700. The `use_xgmi_p2p` parameter is irrelevant; all P2P goes
    over PCIe.
- CDNA (MI300) only, not RDNA4:
  - sc0/sc1/nt cache bits and `buffer_wbl2`.
  - XCC partition / NPS compute modes.
  - MFMA.
- Needs new hardware: PCIe Gen5. The card is Gen5-capable, but Rome's root ports run Gen4 (16 GT/s live).
  A Genoa/Turin platform would roughly halve link-bound times (DeepSeek prefill AR at ~82 % of the link;
  the cold-expert stream).
- Not exposed to ROCm compute, as far as found:
  - MALL allocation control. gfx11+ amdgpu has a per-VA NOALLOC flag on the graphics GEM path, not in KFD/HSA.
  - Dynamic VGPR mode. The gfx120x kernel-descriptor bit exists (LLVM ENABLE_DYNAMIC_VGPR, set up by CP),
    but no HSA/HIP compute path was found. Unverified.
- Hardware counters: SQ / TA / TCP / GL2C counters read 0 under rocprofiler-sdk 1.4.1 on gfx1201, as root
  too. Only GRBM_GUI_ACTIVE, SQ_BUSY_CYCLES and SQ_WAVES work; ATT thread trace works (LEVERS:127).
  `rocprofv3 --pmc` hangs (`isa/README.md`). A newer rocprofiler-sdk is the only route; worth a check when
  ROCm moves.
- Not applicable to our engine: `GPU_MAX_HW_QUEUES` is a HIP setting and does not touch the engine's own AQL
  queues (TODO.md:234, tested 3.11.).
- Tried and lost or no gain:
  - Second SDMA engine in the same direction: no gain, -40 % on a VRAM-bound kernel stream (`aql.cpp:2578-2579`).
  - CU mode (`-mcumode`): -10 % on the fp8 prefill GEMM. `s_setprio` around WMMA: neutral
    (`r4d_gemm_8a8_prefill.h:63`).
  - Chained launches (Qwen: LOST, c16 -6.6 %); plain acquire rule (stale reads).
  - AR lane at 256 threads (-2.8..-3.7 %); 8/16/24 lane blocks (-2.8..-8 %).
- Expected to repeat a known loss: cooperative / persistent kernels with grid barriers. RDNA4 has no hardware
  grid barrier, so a grid sync is the same resident-spin mechanism that lost at c16 in the chained-launch
  test. Not proposed.

## Status after the user's review (12.11.2026 ~17:05 UTC)
- **Deeper undervolt: REJECTED by the user** ("tasa semmoista tehojen kanssa venkslaamista"). gpu02 stays at -50 mV. Fabric / memory clock tweaks are not speed levers either.
- **10-bit tags: DEAD END (hardware).** The coordinator read PCIe config space read-only via a docker container on gpu02 (no sudo, nothing written):
  - Rome root ports 00/40/80/c0:03.1: DevCap2 10-bit tag completer 0 and requester 0.
  - The cards' switch ports and GPUs: completer and requester supported, requester enable 0.
  - Extended (8-bit) tags are on everywhere; relaxed ordering is on; MPS 256 / MRRS 512; root links are Gen4 x16.
  - With no completer support at the root, host DMA cannot use 10-bit tags, so the 18.5 vs 28.7 GB/s cold-stream gap is not tags.
- **fp8 activations on hc_read / bf16 GEMMs:** not exact (new reference). hc_read runs at 19-24 % of bf16 peak, so it is not compute-bound. Not recommended.
- **Open, no user input needed:** fence-scope microbenchmark (item 3) and queue-pipe microbenchmark (item 4), each with a stale-read stress test. They are queued behind the current Qwen / DS items.
