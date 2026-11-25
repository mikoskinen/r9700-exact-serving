#!/usr/bin/env python3
"""ovlcheck.py (11.11.2026, lever DA fidelity item) -- READBACK of an overlay from the card: the engine's own copies of overlaid
weights (RADIANCE_DUMP_BUF + RADIANCE_DUMP_DIR at -vv: rank 0's stored first row of each named weight, core/engine_probe.cpp
dump_buffers) against the overlay's intended bytes and the container's original bytes, prefix by prefix:
  card == intended          the overlay reached the card unchanged
  card == bf16(intended)    narrowed at load (the router under RADIANCE_DS41_ROUTER_BF16=1: deepseek41.cpp ds41_router_bf16)
  card == original          the overlay did NOT reach this copy
  none                      a relayout (the stored form is the kernel's) -- then only the relative deltas are reported
and the measured delta |card - original| / |original| beside the intended |intended - original| / |original| over the row.
  ovlcheck.py DUMPDIR OVERLAY RAD   -> "OVL {json}" a weight"""
import glob
import os
import sys

import numpy as np

import radread
from datrain import read_overlay


def bf16_round(f32):
    u = f32.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def as_f32(raw, dt):
    if dt == "bf16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, np.float32)


def main():
    ddir, ovp, radp = sys.argv[1:4]
    ov = read_overlay(ovp)
    rad = radread.RadFile(radp)
    files = {}
    for f in sorted(glob.glob(os.path.join(ddir, "*.bin"))):
        base = os.path.basename(f)[:-4]
        name = base.split("-", 2)[-1] if base.count("-") >= 2 else base
        for en in ov:
            if name.endswith(en):
                files[en] = f            # the last step's copy (weights do not change)
    import json
    for en, f in sorted(files.items()):
        card = open(f, "rb").read()
        e = rad.entries[en]
        cdt = rad.encoding(e)["planes"][0]["dtype"]
        orig, _, _ = rad.plane_bytes(e, 0)
        orig = bytes(orig)
        want = ov[en]
        n = len(card)
        r = {"weight": en, "bytes": n, "container_dtype": cdt}
        if card == want[:n]:
            r["card"] = "== intended"; cd = cdt
        elif cdt == "f32" and n * 2 <= len(want) and card == bf16_round(np.frombuffer(want[:2 * n], np.float32)).tobytes():
            r["card"] = "== bf16(intended)"; cd = "bf16"
        elif card == orig[:n]:
            r["card"] = "== original (overlay NOT applied)"; cd = cdt
        elif cdt == "f32" and card == bf16_round(np.frombuffer(orig[:2 * n], np.float32)).tobytes():
            r["card"] = "== bf16(original) (overlay NOT applied)"; cd = "bf16"
        else:
            r["card"] = "none (a relayout or another copy)"; cd = None
        if cd is not None:
            m = n // (2 if cd == "bf16" else 4)
            c = as_f32(card, cd)
            w = as_f32(want, cdt)[:m]
            o = as_f32(orig, cdt)[:m]
            on = float(np.linalg.norm(o)) or 1e-30
            r["intended_delta"] = round(float(np.linalg.norm(w - o)) / on, 6)
            r["card_delta"] = round(float(np.linalg.norm(c - o)) / on, 6)
            r["card_vs_intended"] = round(float(np.linalg.norm(c - w)) / on, 6)
        print("OVL " + json.dumps(r), flush=True)
    missing = sorted(set(os.path.basename(f) for f in glob.glob(os.path.join(ddir, "*.bin"))) - set(os.path.basename(f) for f in files.values()))
    print("OVL " + json.dumps({"dumped_files": len(glob.glob(os.path.join(ddir, "*.bin"))), "matched": len(files), "unmatched_files": missing[:10]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
