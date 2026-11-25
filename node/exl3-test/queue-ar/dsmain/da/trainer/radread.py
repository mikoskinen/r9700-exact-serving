"""radread.py (11.11.2026, lever DA) -- a minimal reader of the .rad container (abi/rad_format.h, abi/rad_encoding.h) for the
DSpark drafter's weights: plain bf16 / f32 entries, the affine K-quant head / embedding (i6 codes, i8 scale per [1x16], f16
scale.1 per [1x256], as q6_K converts), and the MXFP4 experts (fp4_e2m1 codes, e8m0 scale per [1x32]). numpy only; the file is
memory-mapped, nothing is written. Decoding follows the affine scheme of rad_encoding.h: v = code * scale * scale.1 ...; sub-byte
codes are a bit stream along the row, lowest bits first, every row starting on a byte."""
import mmap
import struct

import numpy as np

RAD_MAGIC = 0x31444152
DT = {1: "f32", 2: "f16", 3: "bf16", 8: "i8", 18: "fp4_e2m1", 21: "e8m0", 27: "i6", 28: "u6"}
DT_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i8": 8, "fp4_e2m1": 4, "e8m0": 8, "i6": 6, "u6": 6}
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)

HDR = struct.Struct("<IIQ" + "Q" * 5 + "Q" * 16 + "Q" * 8)          # 248 bytes
ENTRY = struct.Struct("<QQQII6qiiiIQQQ2Q")                           # 136 bytes
PLANE = struct.Struct("<QQ")
ENCP = struct.Struct("<24sII2q2q")                                   # 64 bytes
ENC_HEAD = struct.Struct("<24s24sii")                                # 56 bytes, then 8 planes


class Entry:
    __slots__ = ("name", "enc", "rank", "shape", "layer", "expert", "slot", "n_planes", "plane_first", "offset", "bytes")


class RadFile:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        h = HDR.unpack_from(self.m, 0)
        if h[0] != RAD_MAGIC:
            raise ValueError("%s: not a .rad (magic %x)" % (path, h[0]))
        (self.str_off, self.str_bytes, self.meta_off, self.meta_count, self.dir_off, self.dir_count, self.plane_off,
         self.plane_count, self.enc_off, self.enc_count) = h[8:18]
        self.entries = {}
        for i in range(self.dir_count):
            v = ENTRY.unpack_from(self.m, self.dir_off + i * ENTRY.size)
            e = Entry()
            e.name = self.s(v[0])
            e.enc, e.rank = v[3], v[4]
            e.shape = tuple(v[5:5 + e.rank])
            e.layer, e.expert, e.slot, e.n_planes, e.plane_first, e.offset, e.bytes = v[11:18]
            self.entries[e.name] = e

    def s(self, off):
        if off == 0:
            return ""
        a = self.str_off + off
        return self.m[a:self.m.find(b"\0", a)].decode()

    def encoding(self, e):
        a = self.enc_off + e.enc * (ENC_HEAD.size + 8 * ENCP.size)
        scheme, transform, n, _ = ENC_HEAD.unpack_from(self.m, a)
        planes = []
        for k in range(n):
            role, dt, kind, b0, b1, x0, x1 = ENCP.unpack_from(self.m, a + ENC_HEAD.size + k * ENCP.size)
            planes.append(dict(role=role.split(b"\0")[0].decode(), dtype=DT.get(dt, dt), kind=kind, block=(b0, b1),
                               extent=(x0, x1)))
        return dict(scheme=scheme.split(b"\0")[0].decode(), transform=transform.split(b"\0")[0].decode(), planes=planes)

    def plane_bytes(self, e, k):
        off, nb = PLANE.unpack_from(self.m, self.plane_off + (e.plane_first + k) * PLANE.size)
        return memoryview(self.m)[off:off + nb], off, nb

    def describe(self, name):
        e = self.entries[name]
        enc = self.encoding(e)
        return "%s %s %s/%s [%s]" % (name, e.shape, enc["scheme"], enc["transform"],
                                     ", ".join("%s %s %s" % (p["role"], p["dtype"], p["block"]) for p in enc["planes"]))

    # ---------------------------------------------------------------- decoding
    @staticmethod
    def _rows_cols(shape):
        if len(shape) == 1:
            return 1, shape[0]
        r = 1
        for d in shape[:-1]:
            r *= d
        return r, shape[-1]

    @staticmethod
    def _bits(buf, rows, cols, bits, signed, r0=0, r1=None):
        """rows [r0, r1) of a TILED plane of `bits`-bit elements, [rows, cols], each row a bit stream starting on a byte"""
        row_bytes = (cols * bits + 7) // 8
        r1 = rows if r1 is None else r1
        a = np.frombuffer(buf, np.uint8, rows * row_bytes).reshape(rows, row_bytes)[r0:r1]
        rows = r1 - r0
        if bits == 8:
            v = a.astype(np.int16)
            return (v - 256 * (v >= 128)) if signed else v
        if bits == 4:
            v = np.empty((rows, row_bytes * 2), np.uint8)
            v[:, 0::2] = a & 15
            v[:, 1::2] = a >> 4
            return v[:, :cols]
        if bits == 6 and cols % 4 == 0:
            b = a.reshape(rows, cols // 4, 3).astype(np.int16)
            v = np.empty((rows, cols // 4, 4), np.int16)
            v[..., 0] = b[..., 0] & 63
            v[..., 1] = (b[..., 0] >> 6) | ((b[..., 1] & 15) << 2)
            v[..., 2] = (b[..., 1] >> 4) | ((b[..., 2] & 3) << 4)
            v[..., 3] = b[..., 2] >> 2
            v = v.reshape(rows, cols)
            return (v - 64 * (v >= 32)) if signed else v
        # generic: unpack little-endian bit stream
        bitsarr = np.unpackbits(a, axis=1, bitorder="little")[:, :cols * bits].reshape(rows, cols, bits)
        w = (1 << np.arange(bits, dtype=np.int32))
        v = (bitsarr.astype(np.int32) * w).sum(-1)
        if signed:
            v = v - (1 << bits) * (v >= (1 << (bits - 1)))
        return v

    def tensor_chunked(self, name, chunk=8192, dtype=np.float32):
        """a 2-D entry decoded in row chunks (bounded scratch) into one array"""
        e = self.entries[name]
        R, C = self._rows_cols(e.shape)
        out = np.empty((R, C), dtype)
        for r0 in range(0, R, chunk):
            out[r0:min(R, r0 + chunk)] = self.tensor(name, rows=(r0, min(R, r0 + chunk)))
        return out

    def tensor(self, name, rows=None):
        """the entry decoded to float32 (np), its logical shape; `rows` = (r0, r1) decodes only those rows of [rows, cols]"""
        e = self.entries[name]
        enc = self.encoding(e)
        R, C = self._rows_cols(e.shape)
        r0, r1 = rows if rows else (0, R)
        if enc["scheme"] == "plain":
            dt = enc["planes"][0]["dtype"]
            buf, _, nb = self.plane_bytes(e, 0)
            if dt == "f32":
                a = np.frombuffer(buf, np.float32, R * C).reshape(R, C)[r0:r1]
                return a.copy() if rows else a.reshape(e.shape).copy()
            if dt == "bf16":
                a = np.frombuffer(buf, np.uint16, R * C).reshape(R, C)[r0:r1]
                f = (a.astype(np.uint32) << 16).view(np.float32)
                return f if rows else f.reshape(e.shape)
            raise ValueError("%s: plain %s not decoded" % (name, dt))
        if enc["scheme"] != "affine" or enc["transform"]:
            raise ValueError("%s: scheme %s transform %r not decoded" % (name, enc["scheme"], enc["transform"]))
        out = None
        for k, p in enumerate(enc["planes"]):
            buf, _, _ = self.plane_bytes(e, k)
            br, bc = p["block"]
            br = br or R
            bc = bc or C
            pr, pc = -(-R // br), -(-C // bc)
            dt = p["dtype"]
            bits = DT_BITS[dt]
            pa, pb = r0 // br, -(-r1 // br)          # the plane rows covering logical rows r0..r1
            if dt in ("f32", "f16", "bf16"):
                npdt = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16}[dt]
                a = np.frombuffer(buf, npdt, pr * pc).reshape(pr, pc)[pa:pb]
                a = ((a.astype(np.uint32) << 16).view(np.float32) if dt == "bf16" else a.astype(np.float32))
            elif dt == "e8m0":
                a = np.frombuffer(buf, np.uint8, pr * pc).reshape(pr, pc)[pa:pb].astype(np.int32)
                a = np.ldexp(np.float32(1.0), a - 127).astype(np.float32)
            elif dt == "fp4_e2m1":
                a = FP4[self._bits(buf, pr, pc, 4, False, pa, pb)]
            else:
                a = self._bits(buf, pr, pc, bits, dt.startswith("i"), pa, pb).astype(np.float32)
            if br > 1 or bc > 1:
                a = np.repeat(np.repeat(a, br, axis=0), bc, axis=1)
                a = a[(r0 - pa * br):(r0 - pa * br) + (r1 - r0), :C]
            role = p["role"]
            if role == "codes":
                out = a.astype(np.float32)
            elif role.startswith("scale"):
                out = out * a
            elif role == "zero":
                out = out - a
            else:
                raise ValueError("%s: plane role %s not decoded" % (name, role))
        return out if rows else out.reshape(e.shape)

    def raw_plane(self, name):
        e = self.entries[name]
        buf, off, nb = self.plane_bytes(e, 0)
        return bytes(buf), off, nb


if __name__ == "__main__":
    import sys
    f = RadFile(sys.argv[1])
    for n in sys.argv[2:]:
        print(f.describe(n))
        t = f.tensor(n, rows=(0, 2)) if len(f.entries[n].shape) > 1 else f.tensor(n)
        print("  ", t.shape, t.reshape(-1)[:8], float(np.abs(t).mean()))
