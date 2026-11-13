# Host and hardware (EPYC Rome + 4x R9700)

`registry/build.py` generates this file from `registry/registry.csv`, so edit the CSV instead.
Sorted by status, then by gain (positive = better). Status meanings and the exactness definition: [README.md](README.md). `needs-check` = a value was missing or ambiguous in the source and was left blank.

Counts: ADOPTED 8, OPEN 9, NEEDS-USER 0, PARKED 1, CLOSED-small 0, REJECTED 14, total 32.

## ADOPTED (8)

| id | change | exact | measured by | result | gain % | date | source | check |
|---|---|---|---|---|---|---|---|---|
| HW-dimm-8x64 | **8 x 64 GB DIMMs, one 8-channel set (gpu02)** Replaced the mixed 4 x 32 GB + 4 x 64 GB population (384 GB, split into 2-channel pairs) with 8 x 64 GB (512 GB) that interleave as one 8-channel set. | n/a | microbench + other: engine-bench | STREAM triad 8 threads GB/s: 27.8 → 86.1 (+209.7 %) | +209.7 | 2026-11-02 | gpu02/bench/runs/rocm-bw/gpu02-7302p-1102-1530.txt:9-11, :120-122, gpu02/bench/runs/rocm-bw/gpu02-8x64-1102-1632.txt:9-11, :120-122, gpu02/bench/runs/logs/queue-fresh-7302p-1102-1530.log:16, :35, gpu02/bench/runs/logs/fresh-all-8x64-1102-1632.log:16, :35 |  |
| HW-L3 | **Riser move: one PCIe root complex per card (gpu02)** Re-slotted the four gpu02 cards (two on risers) so each sits on its own EPYC root complex, before, cards 83:00 and 86:00 shared root complex 80. | n/a | microbench | 4-card concurrent H2D GB/s (bw4): 61.3 → 105.3 (+71.8 %) | +71.8 | 2026-11-12 | LEVERS 3, 236, 240, 348, gpu02/bench/runs/rocm-bw/gpu02-4cards-1102-1707.txt:25-26, gpu02/bench/runs/rocm-bw/gpu02-4cards-risers-1113.txt:11-12 |  |
| HW-L22 | **Host C-state C2 off + performance governor (gpu02-host-tuning.service)** Persistent host service on gpu02 that turns off the EPYC C2 idle state and sets the CPU governor to performance (numa_balancing 0 is part of the same host tuning). | n/a | mb A/B | Flash-Next TP4 mb c8 aggregate tok/s: ~630 → ~750 (+19.0 %) | +19.0 | 2026-11-09 | LEVERS 22, 345 |  |
| HW-dp2-card-pairing | **DP2 card pairing X/Y (83+86 / c3+03) vs production pairs** Paired the cards so the two that share IO quarter 80 (83:00, 86:00) form one TP2 server (X) and c3+03 the other (Y), instead of c3+83 / 86+03. | n/a | cc_sim | 24-agent replay throughput, mean over 3 seeds: (+13.0 %) | +13.0 | 2026-11-04 | hardware-tuning-gpu02.md (The shared-root pair penalty) |  |
| HW-L240 | **Reseat loose riser after a card dropped off the bus** The user reseated the PCIE7 riser of chip 5943 after that card was lost from the PCIe bus about an hour after the riser move. | n/a | other: kernel-log |  |  | 2026-11-12 | LEVERS 240 |  |
| HW-fan-flat-curve | **Flat fan curve instead of acoustic control (gpu02)** Replaced firmware acoustic control (2400 rpm / 85 C / min PWM 20) with a fan curve of 23 % up to 85 C hotspot, rising linearly to 50 % at 100 C. | n/a | soak |  |  | 2026-10-31 | gpu02/host/fanflat.sh:2-5 |  |
| HW-fanguard | **Memory-temperature fan guard (fanguard.sh, gpu02)** Background guard: a card whose memory >= 92 C or hotspot >= 95 C gets +2 % fan (max 40 %), from 4.11. 21.15 it also steps down -2 % per 30 min when memory <= 85 C, never below the base. | n/a | soak |  |  | 2026-11-05 | gpu02/host/fanguard.sh:3-6 |  |
| HW-power-210w-50mv | **Power setting: 210 W cap + -50 mV offset** Every R9700 runs a 210 W power cap (card default 300 W, max 330 W) with a -50 mV VDDGFX offset. | n/a | other: settings |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:58, :69 |  |

## OPEN (9)

| id | change | exact | measured by | result | gain % | date | source | check |
|---|---|---|---|---|---|---|---|---|
| HW-r9700-cooler-cards | **Cooler cards (fan target below 80 C)** Candidate: run the fans to a lower target temperature so the cards leak less and clock higher at 210 W. | n/a | other: incidental | prefill 16k tok/s (cold vs warm cards): ~7570 → 7782.4 (+2.8 %) | +2.8 | 2026-11-12 | R9700-CAPABILITIES-20261112.md:80, LEVERS.md:53 |  |
| HW-r9700-cu-mask | **CU masking per queue** Candidate: give the all-reduce lane a few WGPs and the compute lane the rest (hsa_amd_queue_cu_set_mask). | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:73 |  |
| HW-r9700-cwsr-off | **Compute wave save/restore off (cwsr_enable=0)** Candidate: disable CWSR to free the ~33 MiB per queue (~132 MiB a card) assumed to be its save area. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:82 |  |
| HW-r9700-lds-barriers | **LDS-only barriers in the hc kernels** Candidate: replace __syncthreads (which also invalidates L0 on gfx12) with lds_barrier where a barrier orders only LDS. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:76 |  |
| HW-r9700-pcie-dpm | **Pin PCIe DPM at level 2 on gpu02** Candidate: lock the card PCIe DPM to level 2 (LCLK 1143 vs 616 MHz) on gpu02. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:56, R9700-CAPABILITIES-20261112.md:79 |  |
| HW-r9700-power-profile | **Power profile COMPUTE (pp_power_profile_mode 5)** Candidate: switch the SMU power profile from BOOTUP_DEFAULT to COMPUTE. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:57, R9700-CAPABILITIES-20261112.md:77 |  |
| HW-r9700-queue-pipe | **Queue-to-CP-pipe placement (MES)** Candidate: control which CP pipe each engine queue lands on, so a pending barrier-AND wait does not slow small kernels on the same pipe. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:72, R9700-CAPABILITIES-20261112.md:178 |  |
| HW-r9700-queue-priority | **Queue priority (hsa_amd_queue_set_priority)** Candidate: set MES/CP queue priorities instead of 0 on every stream. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:75 |  |
| HW-r9700-wave-priority | **Wave priority (s_setprio) on the critical decode chain** Candidate: raise issue priority of the critical decode chain over overlapped side kernels. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:83 |  |

## PARKED (1)

| id | change | exact | measured by | result | gain % | date | source | check |
|---|---|---|---|---|---|---|---|---|
| HW-r9700-gen5-platform | **PCIe Gen5 host platform** Noted that the R9700 is Gen5-capable but EPYC Rome root ports run Gen4 (16 GT/s live). | n/a | other: config-read |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:51, R9700-CAPABILITIES-20261112.md:147-149 |  |

## REJECTED (14)

| id | change | exact | measured by | result | gain % | date | source | check |
|---|---|---|---|---|---|---|---|---|
| HW-dimm-4x32-test | **Diagnostic: 4 identical 32 GB DIMMs, one 4-channel set (gpu02)** Ran gpu02 on 4 identical 32 GB DIMMs (128 GB) instead of the 8 mixed DIMMs to test the channel-pair hypothesis. | n/a | microbench | CPU read one region 16 threads GB/s: 35.9 → 69.1 (+92 %) | +92 | 2026-11-02 | hardware-tuning-gpu02.md (Memory: one interleave set) |  |
| HW-dimm-4x64 | **4 x 64 GB DIMMs (mixed vendors), one 4-channel set (gpu02)** Ran gpu02 on 2 Samsung + 2 SK hynix 64 GB DIMMs (256 GB), which interleave as one 4-channel set. | n/a | microbench + other: engine-bench | 2-card concurrent H2D GB/s (TR bw): 45.1 → 55.4 (+22.8 %) | +22.8 | 2026-11-02 | gpu02/bench/runs/rocm-bw/gpu02-7302p-1102-1530.txt:9-11, :120-122, gpu02/bench/runs/rocm-bw/gpu02-4x64-1102-1612.txt:9-11, :120-122 |  |
| HW-cpu-7302p | **CPU swap EPYC 7282 -> 7302P (gpu02)** Replaced the 2-CCD EPYC 7282 with a 4-CCD 7302P (L3 128 MiB), same DIMMs, slots and BIOS settings. | n/a | microbench + other: engine-bench | 2-card concurrent H2D GB/s (TR bw): 44.2 → 45.1 (+2.0 %) | +2.0 | 2026-11-02 | gpu02/bench/runs/rocm-bw/gpu02-fresh-1031-1616.txt:9-11, :120-122, gpu02/bench/runs/rocm-bw/gpu02-7302p-1102-1530.txt:9-11, :120-122 |  |
| HW-r9700-cu-mode | **CU mode (-mcumode) instead of WGP mode** Compiled the fp8 prefill GEMM in CU mode. | n/a | microbench | fp8 prefill GEMM speed: (-10 %) | -10 |  | R9700-CAPABILITIES-20261112.md:162 |  |
| HW-r9700-second-sdma | **Second SDMA engine in the same copy direction** Used a second SDMA engine for host-to-card copies in the same direction. | n/a | microbench | VRAM-bound kernel stream speed: (-40 %) | -40 |  | R9700-CAPABILITIES-20261112.md:161 |  |
| HW-L23 | **keepawake poller (5 ms SMU sysfs read per card)** A host process that read an SMU sysfs node on every card every 5 ms to keep starts out of the slow state. | n/a | prefill ABBA + mb A/B | prefill 16k off/on ratio (soaked rounds 2 / 3): ? → 1.0008 / 0.9995 |  | 2026-11-09 | LEVERS 23 |  |
| HW-L34 | **Power cap 330 W + 45 % fans** Raised the card power cap from 210 W to 330 W with fans at 45 %. | n/a |  | tok/s (load point not stated): (+5..+7 %) | +5..+7 | 2026-11-09 | LEVERS 34 | needs-check |
| HW-fan-quiet-test | **Quiet fan limit 1800 / 2200 rpm heat test (gpu02)** Set acoustic limit = target rpm 1800, then 2200, target 95 C, min PWM 20 %, under continuous c16 load. | n/a | soak | card 83:00 memory max C at the 1800 / 2200 rpm limit: 96 (test tripped) → 90 (held) |  | 2026-10-31 | hardware-tuning-gpu02.md (Fans) |  |
| HW-r9700-10bit-tags | **PCIe 10-bit tags for in-kernel host reads** Checked whether 10-bit PCIe tags could add outstanding reads for zero-copy expert reads, read PCIe config space read-only. | n/a | other: config-read |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:70, R9700-CAPABILITIES-20261112.md:172-176 |  |
| HW-r9700-deeper-undervolt | **Deeper undervolt on gpu02 (-50 -> -70 / -90 mV at 210 W)** Proposed a deeper VDDGFX offset on gpu02 to gain sclk at the same 210 W cap. | n/a | other: estimate |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:69, R9700-CAPABILITIES-20261112.md:171 |  |
| HW-r9700-fclk-pin | **Pin FCLK (data fabric) at 2400 MHz** Candidate: pin the card data-fabric clock at 2400 MHz (reads 2016 MHz under load). | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:78, R9700-CAPABILITIES-20261112.md:171 |  |
| HW-r9700-mclk-od | **Memory clock overdrive 1259 -> up to 1500 MHz** Candidate: raise MCLK OD for +19 % raw DRAM bandwidth at the same power cap. | n/a | other: none |  |  | 2026-11-12 | R9700-CAPABILITIES-20261112.md:81, R9700-CAPABILITIES-20261112.md:106, R9700-CAPABILITIES-20261112.md:171 |  |
| HW-r9700-setprio-wmma | **Wave priority (s_setprio) around WMMA in a GEMM** Raised wave priority around the WMMA section of a prefill GEMM. | n/a | microbench |  |  |  | R9700-CAPABILITIES-20261112.md:83, R9700-CAPABILITIES-20261112.md:162-163 |  |
| HW-xeon-undervolt-90mv | **Xeon: undervolt -90 mV at the 210 W cap** Set a -90 mV VDDGFX offset on both xeon cards (12.10.). | n/a | soak | sclk MHz under TP2 load: ? → 3062 / 3098 |  | 2026-10-16 | xeon-bench/od-tuning.md:26, R9700-CAPABILITIES-20261112.md:69 |  |
