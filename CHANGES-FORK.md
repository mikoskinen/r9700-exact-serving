# Changes in this fork

This repository is a modified fork of radiance-exl3 by Dylan Johnston
(https://github.com/Dyluhn/radiance-exl3), which is licensed under the Apache License 2.0 (see LICENSE
and NOTICE). NOTICE also carries the MIT notice of exllamav3, which applies to the EXL3 decode and
kernels. Mikael Koskinen modified the fork in 2026 for DeepSeek V4.1 Flash (EXL3 2.25 bpw) and Qwen
Flash-Next (EXL3 4.05 bpw) on AMD Radeon AI PRO R9700 cards (gfx1201).

The commits authored by Mikael Koskinen list every file changed in this fork:

    git log --all --author="Mikael Koskinen" --name-only

The repository keeps every experiment branch, including branches whose change was rejected in testing,
and a branch name alone doesn't say whether its change was adopted. The test history and the change
registry are published separately in the r9700-exact-serving repository.
