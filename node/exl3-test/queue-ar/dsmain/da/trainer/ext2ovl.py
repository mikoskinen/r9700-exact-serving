#!/usr/bin/env python3
"""ext2ovl.py (11.11.2026, lever DA, arm X) -- an EXTERNAL DSpark drafter fine-tune (release tensor names mtp.N.*, safetensors) to a
RADOVL1 overlay of our .rad's drafter entries (RADIANCE_WEIGHT_OVERLAY) + a confidence-head sidecar (RADIANCE_DS41_DSPARK_CONF).
The input is untrusted data: the safetensors header is parsed with bounds checks (header size, dtypes BF16 / F32 only, offsets
inside the file, no overlaps); nothing from the file is executed. Every mapped tensor must match our entry's name, shape and
dtype exactly, or widen exactly (BF16 -> our F32: norms, router, router bias); anything else REFUSES the conversion (exit 1).
Tensors with no counterpart in our container are listed, never guessed. Prints "EXT {json}" lines.

  ext2ovl.py EXT.safetensors RAD OUT.ovl OUT-conf.f32 [--ref]   (--ref: each mapped tensor's relative distance to ours)"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

import radread

NS = 3
BASE = 40


def read_st(path):
    size = os.path.getsize(path)
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    if n <= 0 or n > 50_000_000 or 8 + n > size:
        raise ValueError("bad header size %d" % n)
    h = json.loads(f.read(n).decode("utf-8"))
    h.pop("__metadata__", None)
    spans = []
    for k, v in h.items():
        if not isinstance(v, dict) or v.get("dtype") not in ("BF16", "F32"):
            raise ValueError("%s: dtype %r not accepted" % (k, v.get("dtype") if isinstance(v, dict) else v))
        shp, (a, b) = v["shape"], v["data_offsets"]
        if not all(isinstance(x, int) and x > 0 for x in shp) or not (0 <= a <= b <= size - 8 - n):
            raise ValueError("%s: bad shape / offsets" % k)
        if b - a != int(np.prod(shp)) * (2 if v["dtype"] == "BF16" else 4):
            raise ValueError("%s: byte count does not match its shape" % k)
        spans.append((a, b, k))
    spans.sort()
    for (a0, b0, k0), (a1, b1, k1) in zip(spans, spans[1:]):
        if a1 < b0:
            raise ValueError("%s and %s overlap" % (k0, k1))
    base = 8 + n

    def get(k):
        v = h[k]
        a, b = v["data_offsets"]
        f.seek(base + a)
        raw = f.read(b - a)
        if v["dtype"] == "BF16":
            return np.frombuffer(raw, np.uint16).reshape(v["shape"]), "BF16"
        return np.frombuffer(raw, np.float32).reshape(v["shape"]), "F32"
    return h, get


def plan():
    """(our entry, [external tensors], how): how = copy | rows:g | cat"""
    m = []
    for s in range(NS):
        l, p = BASE + s, "mtp.%d." % s
        m += [("blk.%d.attn_sinks.weight" % l, [p + "attn.attn_sink"], "copy"),
              ("blk.%d.attn_kv_a_norm.weight" % l, [p + "attn.kv_norm.weight"], "copy"),
              ("blk.%d.attn_q_a_norm.weight" % l, [p + "attn.q_norm.weight"], "copy"),
              ("blk.%d.attn_kv.weight" % l, [p + "attn.wkv.weight"], "copy"),
              ("blk.%d.attn_output_b.weight" % l, [p + "attn.wo_b.weight"], "copy"),
              ("blk.%d.attn_q_a.weight" % l, [p + "attn.wq_a.weight"], "copy"),
              ("blk.%d.attn_q_b.weight" % l, [p + "attn.wq_b.weight"], "copy"),
              ("blk.%d.attn_norm.weight" % l, [p + "attn_norm.weight"], "copy"),
              ("blk.%d.ffn_norm.weight" % l, [p + "ffn_norm.weight"], "copy"),
              ("blk.%d.glm_router.weight" % l, [p + "ffn.gate.weight"], "copy"),
              ("blk.%d.glm_router.correction" % l, [p + "ffn.gate.bias"], "copy"),
              ("blk.%d.ffn_gate_up_shexp.weight" % l, [p + "ffn.shared_experts.w1.weight", p + "ffn.shared_experts.w3.weight"], "cat"),
              ("blk.%d.ffn_down_shexp.weight" % l, [p + "ffn.shared_experts.w2.weight"], "copy")]
        for k in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"):
            m.append(("blk.%d.%s.weight" % (l, k), [p + k], "copy"))
        for g in range(8):
            m.append(("blk.%d.attn_output_a.g%d.weight" % (l, g), [p + "attn.wo_a.weight"], "rows:%d" % g))
    m += [("dspark.fc.weight", ["mtp.0.main_proj.weight"], "copy"),
          ("dspark.enc.output_norm.weight", ["mtp.0.main_norm.weight"], "copy"),
          ("dspark.output_norm.weight", ["mtp.%d.norm.weight" % (NS - 1)], "copy"),
          ("dspark.markov_w1.weight", ["mtp.%d.markov_head.embed.weight" % (NS - 1)], "copy"),
          ("dspark.markov_w2.weight", ["mtp.%d.markov_head.head.weight" % (NS - 1)], "copy")]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ext"); ap.add_argument("rad"); ap.add_argument("out"); ap.add_argument("conf")
    ap.add_argument("--ref", action="store_true")
    a = ap.parse_args()
    h, get = read_st(a.ext)
    rad = radread.RadFile(a.rad)
    used, ents, bad, dist = set(), [], [], {}
    for name, srcs, how in plan():
        e = rad.entries.get(name)
        if e is None:
            bad.append("%s: not an entry of our container" % name); continue
        enc = rad.encoding(e)
        if enc["scheme"] != "plain" or len(enc["planes"]) != 1:
            bad.append("%s: our entry is not single-plane plain (%s)" % (name, enc["scheme"])); continue
        odt = enc["planes"][0]["dtype"]
        miss = [s for s in srcs if s not in h]
        if miss:
            bad.append("%s: external tensor(s) missing %s" % (name, miss)); continue
        parts = [get(s) for s in srcs]
        dts = {d for _, d in parts}
        if len(dts) != 1:
            bad.append("%s: mixed dtypes %s" % (name, dts)); continue
        sdt = dts.pop()
        if how == "copy":
            x = parts[0][0]
        elif how == "cat":
            x = np.concatenate([p for p, _ in parts], 0)
        else:
            g = int(how.split(":")[1]); w = parts[0][0]
            if w.shape[0] % 8:
                bad.append("%s: wo_a rows %d not 8 groups" % (name, w.shape[0])); continue
            r = w.shape[0] // 8
            x = w[g * r:(g + 1) * r]
        if tuple(x.shape) != tuple(e.shape):
            bad.append("%s: shape %s vs ours %s" % (name, list(x.shape), list(e.shape))); continue
        if sdt == "BF16" and odt == "bf16":
            data = np.ascontiguousarray(x).tobytes(); mode = "bf16 copy"
        elif sdt == "F32" and odt == "f32":
            data = np.ascontiguousarray(x).tobytes(); mode = "f32 copy"
        elif sdt == "BF16" and odt == "f32":
            data = (np.ascontiguousarray(x).astype(np.uint32) << 16).view(np.float32).tobytes(); mode = "bf16 -> f32 (exact)"
        else:
            bad.append("%s: external %s vs our %s (not exact)" % (name, sdt, odt)); continue
        _, _, nb = rad.plane_bytes(e, 0)
        if len(data) != nb:
            bad.append("%s: %d bytes vs our plane %d" % (name, len(data), nb)); continue
        if a.ref:
            ours = rad.tensor_chunked(name) if len(e.shape) > 1 else rad.tensor(name)
            theirs = np.frombuffer(data, np.float32 if odt == "f32" else np.uint16)
            theirs = theirs if odt == "f32" else (theirs.astype(np.uint32) << 16).view(np.float32)
            theirs = theirs.reshape(ours.shape)
            d = float(np.linalg.norm(theirs - ours) / max(1e-30, np.linalg.norm(ours)))
            dist[name] = round(d, 5)
        ents.append((name, data, mode))
        used.update(srcs)
    unmapped = sorted(set(h) - used)
    conf = "mtp.%d.confidence_head.proj.weight" % (NS - 1)
    conf_ok = False
    if conf in h:
        w, d = get(conf)
        vals = w.astype(np.float32).reshape(-1) if d == "F32" else (w.astype(np.uint32) << 16).view(np.float32).reshape(-1)
        if vals.size == 5120 + 256:
            vals.astype(np.float32).tofile(a.conf); conf_ok = True; unmapped.remove(conf)
    if bad:
        print("EXT " + json.dumps({"refused": bad}))
        return 1
    with open(a.out + ".tmp", "wb") as f:
        f.write(b"RADOVL1\n"); f.write(struct.pack("<I", len(ents)))
        for name, data, _ in ents:
            nb = name.encode(); f.write(struct.pack("<I", len(nb))); f.write(nb); f.write(struct.pack("<Q", len(data))); f.write(data)
    os.replace(a.out + ".tmp", a.out)
    modes = {}
    for _, _, md in ents:
        modes[md] = modes.get(md, 0) + 1
    print("EXT " + json.dumps({"overlay": a.out, "entries": len(ents), "bytes": sum(len(d) for _, d, _ in ents),
                               "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(), "modes": modes,
                               "external_tensors": len(h), "mapped_external": len(used), "unmapped": unmapped,
                               "confidence_sidecar": a.conf if conf_ok else None}))
    if a.ref:
        print("EXT " + json.dumps({"rel_distance_to_ours": dist}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
