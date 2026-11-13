#!/usr/bin/env python3
"""prodidle.py [SECONDS] [TIMEOUT] -- wait until both gpu02 production servers (:8000, :8001) have had nothing running or queued
for SECONDS in a row (default 30; give up after TIMEOUT s, default 1500). The key is read from ~/radiance-1.15.1/compose/.env
inside this process and never printed. Prints one line "IDLE ..." (exit 0) or "BUSY ..." (exit 1). Read-only."""
import os, re, sys, time, urllib.request
need = float(sys.argv[1]) if len(sys.argv) > 1 else 30; limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1500
key = ""
for l in open(os.path.expanduser("~/radiance-1.15.1/compose/.env")):
    if l.startswith("RADIANCE_API_KEY="): key = l.split("=", 1)[1].strip().strip('"').strip("'")
def load(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": "Bearer " + key})
    try:
        txt = urllib.request.urlopen(req, timeout=5).read().decode()
    except Exception as e:
        return None
    s = 0.0
    for k in ("vllm:num_requests_running", "vllm:num_requests_waiting", "radiance:http_queue_depth"):
        for m in re.finditer("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M): s += float(m.group(2))
    return s
t0 = time.time(); quiet = None
while time.time() - t0 < limit:
    v = [load(p) for p in (8000, 8001)]
    busy = sum(x for x in v if x is not None) if any(x is not None for x in v) else 0.0
    if busy == 0:
        quiet = quiet or time.time()
        if time.time() - quiet >= need:
            print(f"IDLE: production :8000/:8001 idle for {need:.0f} s (waited {time.time()-t0:.0f} s; reachable {[x is not None for x in v]})"); sys.exit(0)
    else:
        quiet = None
    time.sleep(2)
print(f"BUSY: production still has work after {limit:.0f} s (last running+waiting+queue {busy})"); sys.exit(1)
