#!/usr/bin/env python3
"""nvlatmet.py (10.11.2026, queue-ar dsmain, ds-main-port nvlat agent; a helper of item 57-p4zyzzzzz-ds-hszzz-nvlat, body
dsm-nvlat.sh) -- one command between two /metrics snapshots of the server:

  nvlatmet.py URL -- CMD [ARGS...]

runs CMD (its stdout / stderr inherited), then prints one line "NVM {json}": the window's decode steps, ms a decode step
(engine_step_seconds_decode_total / engine_steps_decode_total, the stepprobe.py reading), decode tokens, every step and
its seconds, CMD's exit code and wall seconds. Read-only against the server (GET /metrics). Standard library only."""
import json, re, subprocess, sys, time, urllib.request

KEYS = ["radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total", "radiance:decode_tokens_total",
        "radiance:engine_steps_total", "radiance:engine_step_seconds_total"]


def snap(url):
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    except Exception as e:   # noqa: BLE001 -- the reading is then missing, never the command
        return {"error": str(e)}
    out = {}
    for k in KEYS:
        m = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)
        out[k] = float(m.group(2)) if m else None
    return out


def main():
    a = sys.argv[1:]
    if len(a) < 3 or "--" not in a:
        print(__doc__); return 2
    url, cmd = a[0], a[a.index("--") + 1:]
    b = snap(url)
    t0 = time.time()
    rc = subprocess.call(cmd)
    t1 = time.time()
    e = snap(url)
    d = {}
    for k in KEYS:
        x, y = b.get(k), e.get(k)
        d[k.split(":")[1]] = (y - x) if x is not None and y is not None else None
    st = d.get("engine_steps_decode_total") or 0
    sec = d.get("engine_step_seconds_decode_total")
    out = {"rc": rc, "wall_s": round(t1 - t0, 2), "decode_steps": st,
           "ms_per_decode_step": round(1000.0 * sec / st, 3) if st and sec is not None else None,
           "decode_tokens": d.get("decode_tokens_total"), "all_steps": d.get("engine_steps_total"),
           "all_step_s": round(d["engine_step_seconds_total"], 3) if d.get("engine_step_seconds_total") is not None else None}
    if "error" in b or "error" in e:
        out["metrics_error"] = b.get("error") or e.get("error")
    print("NVM " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
