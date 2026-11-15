#!/usr/bin/env python3
"""imatrix_body.py IN.gguf OUT.gguf -- the imatrix for a body-only conversion, stdlib only (no numpy / gguf).
From llama.cpp's imatrix (Unsloth's imatrix_unsloth.gguf_file) keep every entry that is not a routed expert's
(`*_exps.*` dropped: the experts are copied from the EXL3 container, and their per-expert counts would add an expert
profile that the current container does not have), and add the entries the deepseek41 plugin's fused weights look up,
as bakon's conversion did (radiance docs/deepseek41-wip.md):
  blk.N.ffn_gate_up_shexp.weight.{in_sum2,counts} = gate_shexp + up_shexp   (data/imatrix_merge.py's rule, f32 sums)
  blk.N.attn_output_a.g<K>.weight.in_sum2 = attn_output_a in_sum2 columns [K*4096, (K+1)*4096); counts copied.
The KV block is copied byte for byte."""
import struct, sys
src, dst = sys.argv[1:3]
b = open(src, "rb").read()
assert b[:4] == b"GGUF"
ver, nt, nkv = struct.unpack_from("<IQQ", b, 4); assert ver == 3, ver
p = 4 + 4 + 8 + 8
def rstr(p):
    n = struct.unpack_from("<Q", b, p)[0]; return b[p+8:p+8+n].decode(), p + 8 + n
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip(t, p):
    if t in SZ: return p + SZ[t]
    if t == 8: n = struct.unpack_from("<Q", b, p)[0]; return p + 8 + n
    if t == 9:
        et, cnt = struct.unpack_from("<IQ", b, p); p += 12
        for _ in range(cnt): p = skip(et, p)
        return p
    raise SystemExit(f"KV type {t}")
kv0 = p; align = 32
for _ in range(nkv):
    k, p = rstr(p); t = struct.unpack_from("<I", b, p)[0]; p += 4
    if k == "general.alignment": align = struct.unpack_from("<I", b, p)[0]
    p = skip(t, p)
kvb = b[kv0:p]
infos = []
for _ in range(nt):
    name, p = rstr(p); nd = struct.unpack_from("<I", b, p)[0]; p += 4
    dims = list(struct.unpack_from(f"<{nd}Q", b, p)); p += 8 * nd
    typ, off = struct.unpack_from("<IQ", b, p); p += 12
    infos.append((name, dims, typ, off))
ds = (p + align - 1) // align * align
T = {}
for name, dims, typ, off in infos:
    assert typ == 0, (name, typ)
    n = 1
    for d in dims: n *= d
    T[name] = (dims, b[ds + off: ds + off + 4 * n])
out = []  # (name, dims, bytes)
for name, (dims, data) in T.items():
    if "_exps." in name: continue
    out.append((name, dims, data))
def f32add(x, y):
    a = struct.unpack(f"<{len(x)//4}f", x); c = struct.unpack(f"<{len(y)//4}f", y)
    return struct.pack(f"<{len(a)}f", *[u + v for u, v in zip(a, c)])
layers = sorted({int(n.split(".")[1]) for n in T if n.startswith("blk.")})
nf = ng = 0
for L in layers:
    g, u = f"blk.{L}.ffn_gate_shexp.weight", f"blk.{L}.ffn_up_shexp.weight"
    if g + ".in_sum2" in T and u + ".in_sum2" in T:
        for suf in (".in_sum2", ".counts"):
            dg, xg = T[g + suf]; du, xu = T[u + suf]; assert dg == du
            out.append((f"blk.{L}.ffn_gate_up_shexp.weight{suf}", dg, f32add(xg, xu)))
        nf += 1
    a = f"blk.{L}.attn_output_a.weight"
    if a + ".in_sum2" in T:
        dims, x = T[a + ".in_sum2"]; cols = dims[0]; assert cols == 32768 and len(dims) == 1 or dims[1:] == [1], dims
        for K in range(8):
            out.append((f"blk.{L}.attn_output_a.g{K}.weight.in_sum2", [4096] + dims[1:], x[4 * K * 4096: 4 * (K + 1) * 4096]))
            cd, cx = T[a + ".counts"]
            out.append((f"blk.{L}.attn_output_a.g{K}.weight.counts", cd, cx))
        ng += 1
hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(out), nkv)) + kvb
data = bytearray(); offs = []
for name, dims, x in out:
    offs.append(len(data)); data += x; data += b"\0" * ((-len(data)) % align)
for (name, dims, x), off in zip(out, offs):
    nb = name.encode(); hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 0, off)
hdr += b"\0" * ((-len(hdr)) % align)
open(dst, "wb").write(bytes(hdr) + bytes(data))
print(f"{dst}: {len(out)} tensors ({len(T)} in, {sum(1 for n in T if '_exps.' in n)} expert entries dropped), "
      f"{nf} fused shexp, {ng} layers x 8 attn_output_a groups; alignment {align}")
