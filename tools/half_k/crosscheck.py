#!/usr/bin/env python3
"""Half-integer EXL3 K on REAL tensors: an independent numpy transcription of exllamav3 1.6's fractional decode
(exllamav3_ext/quant/frac.cu: frac_s, frac_window, frac_perm; codebook.cuh mul1) against the C oracle.

  crosscheck.py extract SRC_DIR OUT_DIR NAME [NAME ...]   # NAME = e.g. layers.0.ffn.experts.5.w1
      writes OUT_DIR/<NAME>.trellis.bin (raw I16 [kt, nt, 16K]), .suh.bin, .svh.bin, .meta (kt nt words)
  crosscheck.py compare OUT_DIR NAME [NAME ...]
      decodes each .trellis.bin here and compares with OUT_DIR/<NAME>.what_c.bin (fp16 W_hat [16kt][16nt], written
      by half_k_real.c), bit for bit.
Reads a shard's header and the named tensors' bytes only. numpy only."""
import json, os, struct, sys
import numpy as np

def st_index(src):
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    return idx

def st_read(src, shard, name, cache={}):
    path = os.path.join(src, shard)
    if path not in cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            cache[path] = (8 + n, json.loads(f.read(n)))
    base, hdr = cache[path]
    t = hdr[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32}[t["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(t["shape"])

def mul1(states):
    """exllamav3 mul1: fp16(1024 + bytesum(x * 0x83DCD12D)) * K_INV + K_BIAS, one fp16 FMA (exact in f64, one rounding)."""
    x = (states.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
    kinv = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])
    kbias = float(np.array([0xC931], dtype=np.uint16).view(np.float16)[0])
    return ((1024.0 + s.astype(np.float64)) * kinv + kbias).astype(np.float16)

def frac_perm(i):
    t, j = i >> 3, i & 7
    r = (t & 3) * 2 + (j & 1) + (8 if j & 2 else 0)
    c = (t >> 2) + (8 if j & 4 else 0)
    return r, c

def decode_frac(trellis, ka, mask=0xAAAA):
    """trellis int16 [kt, nt, bpb] -> W_hat fp16 [16 kt, 16 nt] (exllamav3's [k, n])."""
    kt, nt, bpb = trellis.shape
    nw = bpb // 2
    assert bpb == 16 * ka + bin(mask).count("1")
    # frac_s: ring position where window i ends; D(i) = ka + bit (i mod 16) of mask
    d = np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)])
    S = np.array([(i >> 4) * bpb + int(d[(i >> 4) * 16:(i >> 4) * 16 + (i & 15) + 1].sum()) for i in range(256)])
    words = trellis.reshape(kt * nt, nw, 2).view(np.uint16).astype(np.uint64)
    words = (words[:, :, 0] | (words[:, :, 1] << 16))                      # u32 words, int16 pairs little-endian
    R = nw * 32
    # frac_window(words, nw, start): 16 bits from ring bit `start`, MSB-first ((w_i << 32 | w_{i+1}) >> (48 - o))
    start = (S - 16) % R
    wi, o = start >> 5, start & 31
    v = (words[:, wi] << 32) | words[:, (wi + 1) % nw]
    states = ((v >> (48 - o).astype(np.uint64)) & 0xFFFF)                    # [tiles, 256]
    vals = mul1(states)
    rc = [frac_perm(i) for i in range(256)]
    out = np.zeros((kt * 16, nt * 16), dtype=np.float16)
    vals = vals.reshape(kt, nt, 256)
    for i, (r, c) in enumerate(rc):
        out[r::16, c::16] = vals[:, :, i]
    return out

def main():
    mode = sys.argv[1]
    if mode == "extract":
        src, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        idx = st_index(src)
        for name in sys.argv[4:]:
            for part in ("trellis", "suh", "svh"):
                a = st_read(src, idx[name + "." + part], name + "." + part)
                a.tofile(os.path.join(out, f"{name}.{part}.bin"))
                if part == "trellis":
                    kt, nt, w = a.shape
                    open(os.path.join(out, f"{name}.meta"), "w").write(f"{kt} {nt} {w}\n")
            m = st_read(src, idx[name + ".mul1"], name + ".mul1")
            assert int(np.uint32(m.astype(np.int64) & 0xFFFFFFFF)) == 0x83DCD12D, (name, m)
            print("extracted", name, a.shape if False else open(os.path.join(out, f"{name}.meta")).read().strip())
    elif mode == "compare":
        out = sys.argv[2]
        bad_total = 0
        for name in sys.argv[3:]:
            kt, nt, w = map(int, open(os.path.join(out, f"{name}.meta")).read().split())
            tr = np.fromfile(os.path.join(out, f"{name}.trellis.bin"), dtype=np.int16).reshape(kt, nt, w)
            assert w % 16 == 8, "a half-integer K tensor"
            ka = (w - 8) // 16
            mine = decode_frac(tr, ka)
            theirs = np.fromfile(os.path.join(out, f"{name}.what_c.bin"), dtype=np.uint16).reshape(kt * 16, nt * 16)
            bad = int((mine.view(np.uint16) != theirs).sum())
            # a negative control: the other parity carrying the extra bit must disagree
            ctrl = decode_frac(tr, ka, mask=0x5555)
            cbad = int((ctrl.view(np.uint16) != theirs).sum())
            f = mine.astype(np.float32)
            print(f"{name}: K={ka}.5 [{kt*16}x{nt*16}] mismatches {bad} of {mine.size}; control (mask 0x5555) "
                  f"mismatches {cbad}; W_hat mean {f.mean():+.4f} std {f.std():.4f}")
            bad_total += bad
        print("TOTAL mismatches", bad_total)
        sys.exit(1 if bad_total else 0)

main()
