# R9700 OD tuning on xeon (2026-10-12)

Load: Radiance 1.2.1 TP2 vision. `xeon-bench/sustain.py` ran 4 parallel coding prompts with 0-60k tokens of pasted stdlib source and max_tokens 1500, for 8 min per step. Temps come from `temps-dual.csv`. "steady" is the mean over the last 3 min and "peak" is the maximum. Raw data is in `xeon-bench/od/` (`*.sum`, `*.clk`, `*.log`, `uvloop2.log`).
Power cap stayed at 210 W. "spikes" means power1_average samples above 250 W.

## 1.2.1 vs 1.0.0 (bench_fresh, stock)
| | decode 0/16k/64k | prefill 16k/64k |
|---|---|---|
| 1.0.0 | 165/155/128 | 1845/1699 |
| 1.2.1 stock (run 2) | 163/161/122 | 1840/1685 |
| 1.2.1 final (-90 mV) | 164/178/129 | 1847/1701 |

There is no meaningful speed difference; the run-to-run noise is about ±10 tok/s on decode. Vision works (the test PNG was described correctly). Temperature-0 outputs were byte-identical run to run.

## Results (junction / mem: steady/peak, °C)
| Step | 06:00 junc | 06:00 mem | 06:00 fan | 83:00 junc | 83:00 mem | 83:00 fan | sclk avg 83/06 | out tok/s |
|---|---|---|---|---|---|---|---|---|
| stock (fan tgt 2100 rpm / 85 C) | 87/94 | 81/86 | 2390 | 92/97 | 93/98 | 2950 | - | 90.9 |
| fan rpm 3000 | 82/89 | 76/80 | 2750 | 89/95 | 89/94 | 3284 | - | 100.2 |
| fan rpm 3500 | 83/89 | 76/80 | 2704 | 86/93 | 85/90 | 3566 | - | 86.9 |
| rpm 3500 + target 80 C | 77/87 | 71/76 | 3172 | 84/89 | 83/88 | 3685 | - | 72.4 |
| chosen fans (06:00 3000/80, 83:00 3500/80), vo 0 | 80/86 | 73/78 | 3007 | 85/91 | 84/88 | 3739 | 2863/2903 | 96.7 |
| vo -30 | 79/86 | 73/76 | 2985 | 84/90 | 84/88 | 3725 | 2928/2971 | 101.3 |
| vo -50 | 80/86 | 73/78 | 2972 | 84/91 | 84/88 | 3717 | 2965/3011 | 101.9 |
| vo -70 | 79/86 | 73/78 | 2984 | 84/90 | 84/88 | 3712 | 3008/3058 | 88.7 |
| vo -90 | 79/86 | 73/78 | 2980 | 84/90 | 83/88 | 3703 | 3062/3098 | 69.9 |

How to read the table:
- Aggregate tok/s moves mostly with the random prompt mix. When a step had more prefill, out tok/s was lower (for example, -90 had 1415 in tok/s and the most long-context requests). Use bench_fresh for speed comparisons.
- Average board power was about 205-215 W at every step, because the cap holds it there. Spikes to 330-397 W (15-25 % of samples) appeared at every step, including -90. Undervolting did not reduce them.
- Undervolting did not lower power or temperatures, again because of the cap. Instead the cards ran at higher clocks for the same power: about +200 MHz average sclk at -90 mV. Clock-bound kernels get slightly faster.
- In this session card 83:00 ran hotter, which is the opposite of the earlier observation. It needed the higher fan setting.

## Firmware quirk (important)
On RDNA4/SMU14, committing a VDDGFX offset change through `pp_od_clk_voltage` on its own resets the PMFW fan control to defaults (2100 rpm acoustic target). Sysfs keeps showing the cached fan values, so nothing looks wrong. The quirk was confirmed by toggling vo -50 to -40 under load: the fan dropped from 3800 to 2500 rpm. Re-committing an unchanged fan value does not restore it.
Fix: stage `vo`, `acoustic_target_rpm_threshold` and `fan_target_temperature` first, then commit once with `echo c > pp_od_clk_voltage`. This was verified to keep the fans at the target. `od-apply.sh` and `odset.sh` both work this way. My first -30 mV attempt was affected and was discarded and rerun.

## Undervolt stability
I tested 0, -30, -50, -70 and -90 mV on both cards. Each step ran 8 min of sustained load. At every step the 5-prompt temperature-0 check was byte-identical to stock, journalctl showed no amdgpu timeout/reset/fault, and there were 0 request errors.
At the user's request I stopped at -90 and used it as the final setting, with no extra margin. The loop had already started -110 when I stopped it. That -110 setting was active for about 15 s at the start of a load run before I re-applied -90.

## Final settings
| card | PCI | VDDGFX offset | acoustic_target_rpm | fan_target_temperature | sclk offset | power cap |
|---|---|---|---|---|---|---|
| card0 | 83:00 | -90 mV | 3500 | 80 C | 0 | 210 W |
| card1 | 06:00 | -90 mV | 3000 | 80 C | 0 | 210 W |
The fan acoustic_limit stays at 5100 and fan_curve was not used. Under sustained load the fans settle at about 3000 rpm (06:00) and about 3700 rpm (83:00).

## Persistence
`~/gpu/od-apply.sh` is idempotent and selects cards by PCI address. It waits for sysfs, wakes each card via its renderD node, stages all values and commits once (retries up to 5 times). It logs to `~/gpu/od-apply.log`.
`~/gpu/r9700-od.service` is a oneshot unit with RemainAfterExit, ordered after systemd-udev-settle. It requires `amdgpu.ppfeaturemask=0xffffffff` to stay on the kernel cmdline.

Install (run by the user):
```
sudo cp /home/mikoskinen/gpu/r9700-od.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now r9700-od.service
systemctl status r9700-od.service; cat /home/mikoskinen/gpu/od-apply.log
```
Root writes the log file once the service runs. For manual re-apply without sudo: `bash ~/gpu/od-apply.sh` (needs the existing udev write permissions).

## Helpers
- `~/gpu/odset.sh show|<card> VO RPM TTEMP [SCLK]`
- `xeon-bench/sustain.py`, `step.sh`, `uvloop.sh`, `detcheck.py` (stock reference: `det-stock.json`), `tsum.py`
