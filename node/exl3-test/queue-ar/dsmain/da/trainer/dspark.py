"""dspark.py (11.11.2026, lever DA) -- a bf16 PyTorch port of DeepSeek-V4.1's DSpark block drafter as Radiance's deepseek41 runs
it (arch/deepseek41/deepseek41.cpp dsp_*; official inference/model.py DSparkBlock / forward_spec / forward_head), for
self-distillation to the EXL3 2.25 target. Weights come from the .rad container (radread.py). The Markov walk is not trained:
it is the greedy walk over logits + W2.W1[prev] used for proposals (validation), and the teacher-forced bias in the loss.

Shapes: D 5120, hc 4, heads 64 x 512 (rope on the last 64, gptj pairs, theta 10000, no YaRN), q_lora 1280, o groups 8 x 1024,
window 128 + block 5, 128 experts top 3 (MXFP4, sqrt-softplus router + correction bias, route scale 1.5, SwiGLU clamp 10),
one shared expert, vocab 129280 (head / embedding: the trunk's own, the container's K-quant decoded in f32).

Engine numerics followed: bf16 dense GEMMs with no activation quantisation (gemm_nt at bf16), the explicit fp8 e4m3 round trips
with power-of-two (ue8m0) scales a 32-block on the window keys and on the attention output, attention in f32 with the sink in
the denominator. Options (validated against the engine's own proposals): fp8 activations into the routed experts (the
reference's act_quant before an fp4 GEMM, default on) and into the shared expert (default off)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

D, HC, NH, HD, ROPE, QL, OG, OR, WIN, BLK, NE, TOPK, NFF = 5120, 4, 64, 512, 64, 1280, 8, 1024, 128, 5, 128, 3, 2304
EPS, HC_EPS, ITERS, ROUTE_SCALE, SWIGLU = 1e-20, 1e-6, 20, 1.5, 10.0
NOISE = 128799
STAGES = (40, 41, 42)


def fp8_rt(x, block=32):
    """act_quant(x, 32, ue8m0, inplace): per 32-block power-of-two scale, e4m3 round trip; straight-through gradient"""
    sh = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    q = q.reshape(sh).to(x.dtype)
    return x + (q - x).detach() if x.requires_grad else q


def rmsnorm(x, w, eps=EPS):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def rope_tab(pos, device):
    """cos, sin [*, 32] for integer positions (gptj pairs over the last 64 dims, theta 10000)"""
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE))
    ang = pos.to(torch.float32).unsqueeze(-1) * inv
    return torch.cos(ang), torch.sin(ang)


def rope(x, cs, inverse=False):
    """rotate x[..., -64:] by (cos, sin) broadcast over x's leading dims; pairs (2i, 2i+1)"""
    c, s = cs
    if inverse:
        s = -s
    head, tail = x[..., :-ROPE], x[..., -ROPE:].float()
    a, b = tail[..., 0::2], tail[..., 1::2]
    ra, rb = a * c - b * s, a * s + b * c
    t = torch.stack([ra, rb], -1).flatten(-2).to(x.dtype)
    return torch.cat([head, t], -1)


def hc_mixes(x, fn, scale, base):
    """x [..., 4, D] -> pre [..., 4], post [..., 4], comb [..., 4, 4] (f32), as hc_split_sinkhorn"""
    xf = x.flatten(-2).float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    m = F.linear(xf, fn) * r
    pre = torch.sigmoid(m[..., :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(m[..., HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (m[..., 2 * HC:] * scale[2] + base[2 * HC:]).unflatten(-1, (HC, HC))
    comb = torch.softmax(comb, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, comb


def hc_pre(x, pre):
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def hc_post(x, res, post, comb):
    y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + (comb.unsqueeze(-1) * res.float().unsqueeze(-2)).sum(-3)
    return y.to(x.dtype)


# ---------------------------------------------------------------- MXFP4 experts (frozen), dequantised on the fly
_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fp4_dequant(codes, exps, dtype=torch.bfloat16):
    """codes [rows, cols/2] u8 (element 0 in the low nibble), exps [rows, cols/32] u8 (e8m0) -> [rows, cols] (exact in bf16)"""
    lut = _FP4.to(codes.device)
    lo, hi = lut[(codes & 15).long()], lut[(codes >> 4).long()]
    v = torch.stack([lo, hi], -1).flatten(-2)
    sc = torch.exp2(exps.float() - 127.0).repeat_interleave(32, -1)
    return (v * sc).to(dtype)


class _Expert(torch.autograd.Function):
    """one routed expert, weights frozen MXFP4: y = W2 . (w * swiglu(W13 . x)); the weights are dequantised again in
    backward, so no expert's bf16 weights outlive its own call"""

    @staticmethod
    def forward(ctx, x, w, gu_c, gu_e, dn_c, dn_e, aq, post=False):
        """post: the engine's order (moe_gemm_ggml + moe_gather): swiglu unweighted, the down GEMM's output rounded to
        bf16, the routing weight applied by the caller (w is then ignored)"""
        ctx.save_for_backward(x, w, gu_c, gu_e, dn_c, dn_e)
        ctx.aq, ctx.post = aq, post
        with torch.no_grad():
            xi = fp8_rt(x) if aq else x
            gu = (xi @ fp4_dequant(gu_c, gu_e).t()).float()
            g, u = gu[:, :NFF], gu[:, NFF:]
            u = u.clamp(-SWIGLU, SWIGLU)
            g = g.clamp(max=SWIGLU)
            h = (F.silu(g) * u * (1.0 if post else w)).to(x.dtype)
            hi = fp8_rt(h) if aq else h
            return (hi @ fp4_dequant(dn_c, dn_e).t()).float()

    @staticmethod
    def backward(ctx, gy):
        x, w, gu_c, gu_e, dn_c, dn_e = ctx.saved_tensors
        aq, post = ctx.aq, ctx.post
        if post:
            w = torch.ones_like(w)
        Wgu, Wd = fp4_dequant(gu_c, gu_e), fp4_dequant(dn_c, dn_e)
        xi = fp8_rt(x) if aq else x
        gu = (xi @ Wgu.t()).float()
        g0, u0 = gu[:, :NFF], gu[:, NFF:]
        u = u0.clamp(-SWIGLU, SWIGLU)
        g = g0.clamp(max=SWIGLU)
        sg = torch.sigmoid(g)
        silu = g * sg
        gh = (gy.to(x.dtype) @ Wd).float()                 # d h (straight through the fp8 round trip)
        gw = (gh * silu * u).sum(-1, keepdim=True)
        ghw = gh * w
        gu_u = ghw * silu * ((u0 >= -SWIGLU) & (u0 <= SWIGLU)).float()
        gu_g = ghw * u * (sg * (1 + g * (1 - sg))) * (g0 <= SWIGLU).float()
        gx = torch.cat([gu_g, gu_u], -1).to(x.dtype) @ Wgu
        return gx, (None if post else gw), None, None, None, None, None, None


class DSpark(torch.nn.Module):
    def __init__(self, rad, device, train=("attn", "shexp", "router", "hc", "norm"), head_dtype=torch.float32,
                 expert_aq=False, shared_aq=False, post_w=True, log=print):
        super().__init__()
        self.dev = torch.device(device)
        # engine numerics (11.11. after the pilot): libr4d's MXFP4 moe_gemm_ggml reads bf16 activations (no fp8 / int8
        # quantisation: r4d_ds41_mxfp4.hip header), and moe_gather weights the bf16 expert outputs after the down GEMM
        # (docs/OPS.md moe_gather): expert_aq False, post_w True are the engine's; the pilot trained with expert_aq True
        self.expert_aq, self.shared_aq, self.post_w = expert_aq, shared_aq, post_w
        self.names = {}            # parameter name -> container entry (the overlay's names)
        t = lambda n, dt=torch.bfloat16: torch.from_numpy(np.ascontiguousarray(rad.tensor(n))).to(dt)
        self.st = torch.nn.ModuleList()
        self.experts = []
        for l in STAGES:
            m = torch.nn.Module()
            spec = {
                "hc_attn_fn": ("hc", torch.float32), "hc_attn_scale": ("hc", torch.float32), "hc_attn_base": ("hc", torch.float32),
                "hc_ffn_fn": ("hc", torch.float32), "hc_ffn_scale": ("hc", torch.float32), "hc_ffn_base": ("hc", torch.float32),
                "attn_norm": ("norm", torch.float32), "ffn_norm": ("norm", torch.float32),
                "attn_q_a": ("attn", torch.bfloat16), "attn_q_a_norm": ("norm", torch.float32), "attn_q_b": ("attn", torch.bfloat16),
                "attn_kv": ("attn", torch.bfloat16), "attn_kv_a_norm": ("norm", torch.float32),
                "attn_sinks": ("attn", torch.float32), "attn_output_b": ("attn", torch.bfloat16),
                "glm_router": ("router", torch.float32), "glm_router.correction": (None, torch.float32),
                "ffn_gate_up_shexp": ("shexp", torch.bfloat16), "ffn_down_shexp": ("shexp", torch.bfloat16),
            }
            for k, (grp, dt) in spec.items():
                en = "blk.%d.%s" % (l, k if k.endswith("correction") else k + ".weight")
                p = torch.nn.Parameter(t(en, dt).to(self.dev), requires_grad=grp in train)
                pn = k.replace(".", "_")
                m.register_parameter(pn, p)
                self.names["st.%d.%s" % (l - STAGES[0], pn)] = en
            for g in range(OG):
                en = "blk.%d.attn_output_a.g%d.weight" % (l, g)
                p = torch.nn.Parameter(t(en).to(self.dev), requires_grad="attn" in train)
                m.register_parameter("attn_output_a_g%d" % g, p)
                self.names["st.%d.attn_output_a_g%d" % (l - STAGES[0], g)] = en
            self.st.append(m)
            ex = []
            for e in range(NE):
                cs = []
                for k in ("ffn_gate_up_exps", "ffn_down_exps"):
                    en = "blk.%d.%s.%d.weight" % (l, k, e)
                    ent = rad.entries[en]
                    R, C = ent.shape
                    b0, _, _ = rad.plane_bytes(ent, 0)
                    b1, _, _ = rad.plane_bytes(ent, 1)
                    codes = torch.from_numpy(np.frombuffer(b0, np.uint8, R * C // 2).reshape(R, C // 2).copy())
                    exps = torch.from_numpy(np.frombuffer(b1, np.uint8, R * C // 32).reshape(R, C // 32).copy())
                    cs += [codes.to(self.dev), exps.to(self.dev)]
                ex.append(cs)
            self.experts.append(ex)
            log("dspark: stage %d loaded" % l)
        self.out_norm = torch.nn.Parameter(t("dspark.output_norm.weight", torch.float32).to(self.dev), requires_grad="norm" in train)
        self.names["out_norm"] = "dspark.output_norm.weight"
        self.mk1 = t("dspark.markov_w1.weight").to(self.dev)
        self.mk2 = t("dspark.markov_w2.weight").to(self.dev)
        self.head = torch.from_numpy(rad.tensor_chunked("output.weight")).to(head_dtype).to(self.dev)
        emb = rad.tensor_chunked("token_embd.weight")
        self.embed = torch.from_numpy(emb).to(torch.bfloat16).to(self.dev)      # embed_lookup_ggml writes bf16
        del emb
        log("dspark: head %s, embedding, Markov tables loaded" % (head_dtype,))

    # ---------------------------------------------------------------- pieces
    def ctx_keys(self, s, main_x, pos):
        """stage s's window keys of context rows main_x [..., D] at positions pos [...]: fp8(rope(kv_norm(wkv(main_x))))"""
        m = self.st[s]
        kv = rmsnorm(main_x @ m.attn_kv.t(), m.attn_kv_a_norm)
        return fp8_rt(rope(kv, rope_tab(pos, self.dev)))

    def attn(self, s, x, pos, kctx, kmask):
        """x [B, 5, D] (normed), pos [B, 5], kctx [B, W, 512] window keys, kmask [B, W] valid -> [B, 5, D]"""
        m = self.st[s]
        B = x.shape[0]
        cs = rope_tab(pos, self.dev)
        qr = rmsnorm(x @ m.attn_q_a.t(), m.attn_q_a_norm)
        q = (qr @ m.attn_q_b.t()).unflatten(-1, (NH, HD))
        q = rope(q, (cs[0].unsqueeze(-2), cs[1].unsqueeze(-2)))
        kv = fp8_rt(rope(rmsnorm(x @ m.attn_kv.t(), m.attn_kv_a_norm), cs))
        keys = torch.cat([kctx, kv], 1)                                            # [B, W + 5, 512]
        mask = torch.cat([kmask, torch.ones(B, BLK, dtype=torch.bool, device=self.dev)], 1)
        sc = torch.einsum("bihd,bjd->bhij", q.float(), keys.float()) / math.sqrt(HD)
        sc = sc.masked_fill(~mask[:, None, None, :], float("-inf"))
        sink = m.attn_sinks.view(1, NH, 1, 1).expand(B, NH, BLK, 1)
        p = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]
        o = torch.einsum("bhij,bjd->bihd", p, keys.float()).to(x.dtype)          # [B, 5, 64, 512]
        o = rope(o, (cs[0].unsqueeze(-2), cs[1].unsqueeze(-2)), inverse=True)
        o = fp8_rt(o.flatten(-2)).unflatten(-1, (OG, NH // OG * HD))             # [B, 5, 8, 4096]
        wa = torch.stack([getattr(m, "attn_output_a_g%d" % g) for g in range(OG)])  # [8, 1024, 4096]
        oa = torch.einsum("bsgd,grd->bsgr", o, wa).flatten(-2)
        return oa @ m.attn_output_b.t()

    def moe(self, s, x):
        """x [N, D] (ffn-normed) -> [N, D]"""
        m = self.st[s]
        sc = F.linear(x.float(), m.glm_router)
        sc = torch.sqrt(F.softplus(sc))
        idx = (sc + m.glm_router_correction).topk(TOPK, -1)[1]
        w = sc.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        flat = idx.flatten()
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=NE).tolist()
        rows = torch.arange(x.shape[0], device=self.dev).repeat_interleave(TOPK)[order]
        ws = w.flatten()[order].unsqueeze(-1)
        xs_all = x[rows]
        outs, o = [], 0
        for e in range(NE):
            n = counts[e]
            if n == 0:
                continue
            gu_c, gu_e, dn_c, dn_e = self.experts[s][e]
            outs.append(_Expert.apply(xs_all[o:o + n], ws[o:o + n], gu_c, gu_e, dn_c, dn_e, self.expert_aq, self.post_w))
            o += n
        ycat = torch.cat(outs, 0)
        if self.post_w:
            ycat = ycat * ws
        y = torch.zeros(x.shape[0], D, dtype=torch.float32, device=self.dev).index_add(0, rows, ycat)
        if self.post_w:
            y = y.to(x.dtype).float()               # moe_gather narrows the gathered sum to bf16 before the shared add
        xs = fp8_rt(x) if self.shared_aq else x
        gu = (xs @ m.ffn_gate_up_shexp.t()).float()
        g, u = gu[:, :NFF].clamp(max=SWIGLU), gu[:, NFF:].clamp(-SWIGLU, SWIGLU)
        h = (F.silu(g) * u).to(x.dtype)
        h = fp8_rt(h) if self.shared_aq else h
        y = y + (h @ m.ffn_down_shexp.t()).float()
        return y.to(x.dtype)  # the shared expert's bf16 output + the (bf16) gathered sum, rounded once

    def block(self, s, h, pre_mix, pos, kctx, kmask):
        m = self.st[s]
        res = h
        a_pre, a_post, a_comb = hc_mixes(h, m.hc_attn_fn, m.hc_attn_scale, m.hc_attn_base)
        x = rmsnorm(hc_pre(h, pre_mix), m.attn_norm)
        x = self.attn(s, x, pos, kctx, kmask)
        h = hc_post(x, res, a_post, a_comb)
        res = h
        f_pre, f_post, f_comb = hc_mixes(h, m.hc_ffn_fn, m.hc_ffn_scale, m.hc_ffn_base)
        x = rmsnorm(hc_pre(h, a_pre), m.ffn_norm)
        B = x.shape[0]
        x = self.moe(s, x.reshape(B * BLK, D)).reshape(B, BLK, D)
        h = hc_post(x, res, f_post, f_comb)
        return h, f_pre

    def hidden(self, anchors_tok, pos, kctx, kmask):
        """anchors_tok [B], pos [B] (the anchor's position), kctx: per stage [B, W, 512], kmask [B, W] -> [B, 5, D] normed"""
        B = anchors_tok.shape[0]
        toks = torch.full((B, BLK), NOISE, dtype=torch.long, device=self.dev)
        toks[:, 0] = anchors_tok
        rpos = pos.unsqueeze(-1) + torch.arange(BLK, device=self.dev)
        x = self.embed[toks]                                                     # [B, 5, D]
        h = x.unsqueeze(-2).expand(B, BLK, HC, D).contiguous()
        pre = torch.zeros(B, BLK, HC, device=self.dev)
        pre[..., 0] = 1.0
        for s in range(len(STAGES)):
            h, pre = self.block(s, h, pre, rpos, kctx[s], kmask)
        return rmsnorm(hc_pre(h, pre), self.out_norm)

    def logits(self, hn, prev_tok=None):
        """hn [..., D] -> head logits f32 (+ the Markov bias of prev_tok [...] when given)"""
        z = F.linear(hn.to(self.head.dtype), self.head).float()
        if prev_tok is not None:
            z = z + F.linear(self.mk1[prev_tok].float(), self.mk2.float())
        return z

    @torch.no_grad()
    def walk(self, hn, anchor):
        """the greedy Markov walk (forward_head at temperature 0) over hn [B, 5, D] from the anchors [B] -> [B, 5] proposals"""
        z = F.linear(hn.to(self.head.dtype), self.head).float()
        out, prev = [], anchor
        mk2 = self.mk2.float()
        for i in range(BLK):
            t = (z[:, i] + F.linear(self.mk1[prev].float(), mk2)).argmax(-1)
            out.append(t)
            prev = t
        return torch.stack(out, 1)

    def trainable(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]
