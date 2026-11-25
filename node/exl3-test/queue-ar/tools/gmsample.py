#!/usr/bin/env python3
"""gmsample.py OUT SECONDS UUID[,UUID...] [INTERVAL_MS] -- samples the cards' gpu_metrics (read-only sysfs) every ~5 ms (INTERVAL_MS) and pp_dpm_fclk /
pp_dpm_mclk every ~100 ms for SECONDS, into OUT (8.11.2026, queue-ar, item 91). The cards are found by their ROCR UUID in
/sys/class/drm/card*/device/unique_id; `rank` is the UUID's place in the list (the engine's rank order). Lines:
  M rank mono_ns gfx_act umc_act power_w edge_c hotspot_c mem_c cur_sclk cur_uclk avg_sclk indep_throttle_hex
  F rank mono_ns fclk_mhz mclk_mhz           (the levels marked * in pp_dpm_fclk / pp_dpm_mclk)
gpu_metrics v1.3 layout (amdgpu kgd_pp_interface.h gpu_metrics_v1_3; the R9700's SMU 14.0.2 table is that version; a card
whose header is another version is reported once and skipped). Standard library only; reads nothing else and writes OUT."""
import glob, os, re, struct, sys, time, signal


def find(uuid):
    u = uuid.lower().replace("gpu-", "")
    for d in glob.glob("/sys/class/drm/card*/device"):
        try:
            if open(os.path.join(d, "unique_id")).read().strip().lower() == u:
                return d
        except OSError:
            pass
    return None


def star(path):
    try:
        for line in open(path):
            if "*" in line:
                m = re.search(r"(\d+)\s*[Mm]hz", line)
                return int(m.group(1)) if m else -1
    except OSError:
        pass
    return -1


def main():
    out, secs, uuids = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(",")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # 9.11.: kill -> close OUT (Python 3.14 buffers 128 KiB; B500 lost all)
    ivl = float(sys.argv[4]) / 1000.0 if len(sys.argv) > 4 else 0.005   # 9.11.: the polling interval (keep-awake tests)
    devs = []
    with open(out, "w") as f:
        for r, u in enumerate(uuids):
            d = find(u)
            if not d:
                f.write(f"# rank {r} {u}: not found\n"); continue
            try:
                h = open(os.path.join(d, "gpu_metrics"), "rb").read(4)
            except OSError as e:
                f.write(f"# rank {r} {u}: gpu_metrics unreadable ({e})\n"); continue
            size, fr, cr = struct.unpack_from("<HBB", h, 0)
            if (fr, cr) != (1, 3):
                f.write(f"# rank {r} {u}: gpu_metrics v{fr}.{cr}, not 1.3 -- skipped\n"); continue
            devs.append((r, d, open(os.path.join(d, "gpu_metrics"), "rb", buffering=0)))
            f.write(f"# rank {r} {u} {d} gpu_metrics v1.3 size {size}\n")
        t_end = time.monotonic() + secs
        nxt_f = 0.0
        while time.monotonic() < t_end:
            for r, d, fh in devs:
                fh.seek(0)
                b = fh.read(120)
                if len(b) < 120:
                    continue
                mono = time.monotonic_ns()
                ga, ua, _, pw = struct.unpack_from("<HHHH", b, 16)
                te, th, tm = struct.unpack_from("<HHH", b, 4)
                ag, = struct.unpack_from("<H", b, 40)
                cg, cs, cu = struct.unpack_from("<HHH", b, 54)
                ind, = struct.unpack_from("<Q", b, 112)
                f.write(f"M {r} {mono} {ga} {ua} {pw} {te} {th} {tm} {cg} {cu} {ag} {ind:x}\n")
            now = time.monotonic()
            if now >= nxt_f:
                for r, d, _ in devs:
                    f.write(f"F {r} {time.monotonic_ns()} {star(os.path.join(d, 'pp_dpm_fclk'))} "
                            f"{star(os.path.join(d, 'pp_dpm_mclk'))}\n")
                nxt_f = now + 0.1
            time.sleep(ivl)


if __name__ == "__main__":
    main()
