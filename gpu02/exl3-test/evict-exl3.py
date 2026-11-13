#!/usr/bin/env python3
"""evict-exl3.py -- drop the CLEAN page cache of the DeepSeek model files with posix_fadvise(DONTNEED): ~/models/ds41-exl3/*.rad,
~/models/ds41/*.rad, ~/models/glm53-maya/** (read-only open; never the production model flashnext-iq4r; no sudo). = runs/q/evict.py
(3.11.) plus the EXL3 directory. Prints one line: files, GiB, page cache before -> after, MemFree / MemAvailable GiB."""
import glob, os
H = os.path.expanduser("~/models")
def mi(k):
    for l in open("/proc/meminfo"):
        if l.startswith(k + ":"): return int(l.split()[1]) / 2**20
fs = glob.glob(H + "/ds41-exl3/*.rad") + glob.glob(H + "/ds41/*.rad") + [f for f in glob.glob(H + "/glm53-maya/**", recursive=True) if os.path.isfile(f)]
fs = [f for f in fs if "flashnext" not in f]
b, tot = mi("Cached"), 0
for f in fs:
    try:
        fd = os.open(f, os.O_RDONLY); tot += os.fstat(fd).st_size
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError as e:
        print("skip", f, e)
print("evict: %d files %.1f GiB; page cache %.1f -> %.1f GiB; MemFree %.1f GiB; MemAvailable %.1f GiB" % (
      len(fs), tot / 2**30, b, mi("Cached"), mi("MemFree"), mi("MemAvailable")))
