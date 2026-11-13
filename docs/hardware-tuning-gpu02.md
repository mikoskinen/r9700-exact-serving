# Hardware tuning on gpu02

This note covers the host and PCIe changes on `gpu02`, which has an AMD EPYC 7302P (Rome, 16 cores) on
an ASRock Rack ROMED8-2T board, 512 GB of DDR4-2933 and four AMD Radeon AI PRO R9700 cards on PCIe 4.0
[R1, F220]. It's written in English from the raw measurement files in `gpu02/bench/` and from the original
Finnish results log. A number taken only from that log is cited as "gpu02-results-1031 (Finnish, not
published)" with its line. A number from a file in this repository is cited with its path and line.

The measurements use two-card and four-card DMA tests with pinned host buffers (`bw`, `bw4`,
`p2p-bw`, `dma-check`), a STREAM-style host memory test (`membw.c`, 3 x 2 GiB arrays) and short
Radiance TP2 runs with Qwen Flash-Next.

## Memory: one interleave set

The first memory population mixed sizes: 4 x 32 GB plus 4 x 64 GB, 384 GB in total [F7-8]. With the
unequal DIMMs, Rome split the address space into four regions of two channels each, and every buffer
landed on just one channel pair [F171-181, F190]. Two channels of DDR4-2933 give 46.9
GB/s in theory, and the host STREAM triad stayed at 27-33 GB/s whatever the thread count [F180].

Replacing the set with 8 x 64 GB DIMMs of the same size and speed put all eight channels in one
interleave set [F221]. The results, with the same 7302P before and after:

| measurement | mixed 384 GB | 8 x 64 GB | change | source |
|---|---|---|---|---|
| 2-card concurrent H2D / D2H / both, GB/s | 45.1 / 42.9 / 39.7 | 55.4 / 55.3 / 71.5 | +23 / +29 / +80 % | [B1] :9-11, [B2] :9-11 |
| STREAM triad 8 / 16 / 32 threads, GB/s | 27.8 / 27.1 / 26.8 | 86.1 / 84.1 / 82.8 | +210 % | [B1] :120-122, [B2] :120-122 |
| CPU read from one region, 16 / 32 threads, GB/s | 35.9 / 35.9 | 103.2 / 104.0 | +187 / +190 % | [F226] |
| Radiance 1.14.0 TP2 c16, tok/s (mean of 2 runs) | 402.65 | 438.2 | +8.8 % | [L1] :16, [L2] :16 |
| Radiance 1.15.1 TP2 c16, tok/s (mean of 2 runs) | 411.35 | 444.25 | +8.0 % | [L1] :35, [L2] :35 |

With 8 x 64 GB, two cards reading from the host at once reach the PCIe limit (2 x 27.7 GB/s) [F227].
A middle step with 4 identical 32 GB DIMMs already gave one 4-channel set and 55.4 / 55.3 GB/s
two-card DMA, and it raised the one-region CPU read from 35.9 to 69.1 GB/s [F184-191]. Two 64 GB vendors
of the same size and speed (Samsung and SK hynix) interleaved into one set without trouble [F196].
The rule is the same size and speed in every channel, not the same vendor.

The bandwidth on the mixed set also depended on where a run's buffers happened to land. The same boot
measured two-card DMA at 45 / 42 GB/s in one run and 50 / 48 GB/s in another [F149-156], which made
results drift between runs until the DIMMs were changed.

## Check the address map

After any DIMM change, read the data-fabric DRAM address map instead of trusting the DIMM count.
[`gpu02/bench/df-map.sh`](../gpu02/bench/df-map.sh) prints the DF address-map registers of each memory
controller (UMC 0-7) and prints which channels serve which address range (run it with sudo, its
comments are in Finnish). On the mixed set it showed four ranges of 64 to 128 GiB, each served by one
channel pair [F171-177]. On 8 x 64 GB all eight UMCs share 0 to 513.25 GiB [F221]. The interleave
code the script prints is 3 for both a 4-channel and an 8-channel set, so it isn't a channel count.
Count how many UMCs share a range instead [F221-222].

## The CPU swap that didn't help

Before the memory map was understood, the EPYC 7282 (2 CCDs) was swapped for a 7302P (4 CCDs) with the
same mixed DIMMs. It didn't help [F168]:

| measurement | 7282 | 7302P | source |
|---|---|---|---|
| 2-card concurrent H2D, GB/s | 44.2 | 45.1 | [B3] :9, [B1] :9 |
| STREAM triad 8 threads, GB/s | 27.6 | 27.8 | [B3] :120, [B1] :120 |
| Radiance 1.15.1 TP2 c16, tok/s | 429.2 / 443.4 | 403.2 / 419.5 (-5.7 %) | [F164] |

The limit was the channel pairs [F181]. With 8 x 64 GB the 7302P's CPU read reaches 104 GB/s, which
its four CCD links limit [F227-228].

## PCIe: one card per quadrant

The ROMED8-2T's x16 slots hang off the four IO-die quadrants as pairs: quadrant 00 has PCIE5 and PCIE2,
40 has PCIE4 and the M.2 / X550 devices, 80 has PCIE7 and PCIE1, and c0 has PCIE6 and PCIE3 [F251]. Two
cards in the same quadrant share one root port of about 36 GB/s toward host memory [F250, F322].

With all four cards in their default slots, cards 83:00 and 86:00 shared quadrant 80:

| test, pinned DMA | H2D sum | D2H sum | source |
|---|---|---|---|
| pair 83:00 + 86:00 (same quadrant 80) | 35.3 GB/s | 37.9 GB/s | [P1] :6, :11 |
| pair c3:00 + 03:00 (different quadrants) | 55.8 GB/s | 55.3 GB/s | [P2] :6, :11 |
| all 4 cards, wall-clock aggregate | 61.6 GB/s | 75.3 GB/s | [P3] :10, :19 |

In the four-card run the two cards in quadrant 80 fell to 15.4 GB/s H2D each while the other two
kept 27.9 and 27.6 GB/s [P3] :6-9, [F322]. Moving three cards onto risers so that every card has its
own quadrant raised the four-card concurrent DMA as follows:

| 4 cards at once | before (shared quadrant) | after (one per quadrant) | source |
|---|---|---|---|
| H2D | 61.3 GB/s | 105.3 GB/s | [B4] :25, [B5] :11 |
| D2H | 75.6 GB/s | 110.6 GB/s | [B4] :26, [B5] :12 |

Each card alone moves 27.7 to 27.8 GB/s after the move [B5] :7-10, so four cards now run close to four
times one card. [R9700-CAPABILITIES-20261112.md](R9700-CAPABILITIES-20261112.md) line 133 summarises
the same move as 61.5 -> 106.2 GB/s H2D.

## The shared-root pair penalty

Before the riser move, a TP2 server on cards 83:00 + 86:00 was slower than one on c3:00 + 03:00. In a
2 x TP2 replay with 8 agents, the shared-quadrant server produced 193 tok/s against 253 tok/s, with
decode -21 % and prefill -42 % [F323]. Choosing the server pairs so that only
one server had the shared quadrant gave +13.0 % on average over three seeds of a 24-agent replay
[F359-361]. The engine log measured 17.3 + 17.3 GB/s concurrent host reads on that pair against ~28
GB/s for a card alone (LEVERS.md:236, :348). The riser move removed the shared quadrant altogether.

## Host C-states and governor

Without the fix, a fresh engine start was sometimes slow, and in the A/B 2 of 2 starts were slow
without it and 2 of 2 fast with it (LEVERS.md:22). The cause was the host: the cpuidle C2 state (exit
latency 400 us) together with the schedutil governor, which was the default [H1, F6]. Disabling every C-state with an exit latency of 100 us or more and
setting the `performance` governor at boot fixed it, and the Flash-Next TP4 c8 throughput went from
~630 to ~750 tok/s (LEVERS.md:22). The fix is a oneshot systemd unit,
[`gpu02-host-tuning.service`](../gpu02/host/gpu02-host-tuning.service), which runs
[`gpu02-host-tuning`](../gpu02/host/gpu02-host-tuning) (line 9 selects the states by latency). A sysctl
file keeps `kernel.numa_balancing=0` [H2].

Disabling "Global C-state Control" in the BIOS wasn't enough on its own. After the CPU swap, Linux still
listed the C1 and C2 states with that BIOS setting off [F72, F144-145]. Check
`/sys/devices/system/cpu/cpu0/cpuidle/state*/disable` after boot, as in [R1].

## Power cap and undervolt

Every card runs at a 210 W power cap with a -50 mV VDDGFX offset. The card default is 300 W and the
maximum is 330 W (R9700-CAPABILITIES-20261112.md:58). With a cap active, an undervolt raises clocks
instead of lowering power. On the xeon, -90 mV gave about +200 MHz average sclk at 210 W
(R9700-CAPABILITIES-20261112.md:69, xeon-bench/od-tuning.md:26), and -70 / -90 mV gave byte-identical
output there. gpu02 stays at -50 mV. Raising the cap to 330 W with 45 % fans gave +5-7 % at a junction
temperature of 98 C (LEVERS.md:34), and that wasn't pursued. [`od-apply.sh`](../gpu02/host/od-apply.sh)
applies the overdrive settings at boot [F132].

## Fans

The cards sit side by side on an open bench, and the GDDR6 memory is the hottest part, not the
junction. In a quiet-fan heat test at c16 with a 2200 rpm limit, the card 83:00 stayed at 90 C memory or
below, and with an 1800 rpm limit its memory reached 96 C and tripped the test [F124-130]. The acoustic
fan control also kept speeding up and slowing down, with 83 and 61 jumps of more than 150 rpm in an hour
on the two cards [F135]. A flat curve removed the jumps (0 in the next 30 min) [F135-139].

The flat curve is set per card with [`fanflat.sh`](../gpu02/host/fanflat.sh): a fixed duty up to 85 C
hotspot, then linear to 50 % at 100 C (lines 2-5). [`fanguard.sh`](../gpu02/host/fanguard.sh) runs
beside it and adds 2 % to a card whose memory is at 92 C or more, or whose hotspot is at 95 C or more,
up to 40 %. It steps back down after 30 min at 85 C memory or less (lines 3-6).

## PCIe settings and dead ends

| setting | state | source |
|---|---|---|
| 8-bit extended tags, relaxed ordering, MPS 256 / MRRS 512, Gen4 x16 on every card | on | R9700-CAPABILITIES-20261112.md:175 |
| 10-bit tags | dead end: the Rome root ports support neither 10-bit completer nor requester, so enabling the BIOS "Ten Bit Tag" option [F70] has no effect | R9700-CAPABILITIES-20261112.md:172-173 |
| PCIe DPM pinned at a fixed level | not used on gpu02. On the xeon it was part of a bundle of fixes and its own effect was never isolated. | R9700-CAPABILITIES-20261112.md:56, :79 |
| SDMA vs blit copies (`HSA_ENABLE_SDMA`) | doesn't change the H2D limit. Host copies are DMA and cost almost no CPU. | [F32-43] |

## Lessons from the moves

Seat risers firmly. After the riser move, one card fell off the bus ("device lost from bus") on wake
from full idle with no AER message before it. The probable cause was a slightly loose riser, which was
reseated (LEVERS.md:240).

Check every card's link width after a BIOS reset. On the Threadripper test machine a cleared CMOS
switched a slot to x4 bifurcation, and an x8 card pair delivered 54-57 % of the prefill speed of an x16
pair [G28-30, G87-89].

Run the bandwidth tests after every hardware change and keep the raw output. Every number in this note
comes from such a file or from the results log that recorded them.

## Sources

- [R1] [gpu02/host/hostfacts-20261113.txt](../gpu02/host/hostfacts-20261113.txt): CPU, board, RAM, cards, cpuidle states, governor, numa_balancing (read-only queries on 13.11.2026).
- [B1] [gpu02/bench/runs/rocm-bw/gpu02-7302p-1102-1530.txt](../gpu02/bench/runs/rocm-bw/gpu02-7302p-1102-1530.txt) (7302P, mixed DIMMs).
- [B2] [gpu02/bench/runs/rocm-bw/gpu02-8x64-1102-1632.txt](../gpu02/bench/runs/rocm-bw/gpu02-8x64-1102-1632.txt) (7302P, 8 x 64 GB).
- [B3] [gpu02/bench/runs/rocm-bw/gpu02-fresh-1031-1616.txt](../gpu02/bench/runs/rocm-bw/gpu02-fresh-1031-1616.txt) (7282, mixed DIMMs).
- [B4] [gpu02/bench/runs/rocm-bw/gpu02-4cards-1102-1707.txt](../gpu02/bench/runs/rocm-bw/gpu02-4cards-1102-1707.txt) (4 cards before the riser move).
- [B5] [gpu02/bench/runs/rocm-bw/gpu02-4cards-risers-1113.txt](../gpu02/bench/runs/rocm-bw/gpu02-4cards-risers-1113.txt) (4 cards after the riser move, header line in Finnish).
- [P1] [P2] [P3] [gpu02/bench/runs/pair-dma-q1103/](../gpu02/bench/runs/pair-dma-q1103/): X_83+86-dma-1.txt, Y_c3+03-dma-1.txt, ALL4-dma-1.txt.
- [L1] [gpu02/bench/runs/logs/queue-fresh-7302p-1102-1530.log](../gpu02/bench/runs/logs/queue-fresh-7302p-1102-1530.log), [L2] [gpu02/bench/runs/logs/fresh-all-8x64-1102-1632.log](../gpu02/bench/runs/logs/fresh-all-8x64-1102-1632.log). Line 16 is the 1.14.0 c16 arm and line 35 the 1.15.1 c16 arm, two runs each.
- [H1] [gpu02/host/gpu02-host-tuning](../gpu02/host/gpu02-host-tuning) (line 9 selects the states, line 14 restores schedutil on revert) and the C2 latency in [R1]. [H2] [gpu02/host/90-gpu02-numa.conf](../gpu02/host/90-gpu02-numa.conf).
- [F*] gpu02-results-1031 (Finnish, not published), the line numbers given.
- [G*] gpu01-results-1030 (Finnish, not published), the Threadripper comparison machine, the line numbers given.
- LEVERS.md, R9700-CAPABILITIES-20261112.md and xeon-bench/od-tuning.md are in this `docs/` folder.
