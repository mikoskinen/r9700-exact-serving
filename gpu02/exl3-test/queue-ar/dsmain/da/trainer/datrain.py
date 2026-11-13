#!/usr/bin/env python3
"""datrain.py (11.11.2026, lever DA) -- self-distillation of the DSpark drafter to the EXL3 2.25 target, from the KL-record dump
(Radiance --kld-record DIR --kld-corpus FILE --kld-top 32 with RADIANCE_DS41_DA_TAPS=1 RADIANCE_KLD_DUMP=dspark_main_normed):

  DIR/kld.json        the documents (source, tokens, score_from 0, row0) and rows
  DIR/tokens.i32      every document's tokens
  DIR/top.bin         [rows, 32] {i32 id, f32 log p}: the target's next-token distribution, most likely first
  DIR/rows.f32        [rows, 3]: the target's NLL, entropy, the mass outside the top 32
  DIR/dump-dspark_main_normed.bin   [rows, 5120] bf16: main_x = main_norm(fc(taps 37-39)) of every position

Row r of a document is its position r - row0 (the logits at p predict token p + 1). An anchor a (1 <= a <= n - 6) proposes the
block a+1 .. a+5 from the context keys of positions a-128 .. a-1 and the block [x_a, noise x 4]; row i's target is the
target's distribution at position a + i (top 32 + the rest), its Markov predecessor x_{a+i} (teacher forced).

Subcommands (all print "DAT {json}" lines):
  validate   the port vs the engine's own -v proposals on the validation documents (VAL json from dagen.py), per numerics
             variant (routed / shared expert fp8 activations): proposal agreement by position and whole block
  eval       simulated acceptance (fixed window 5, greedy walk vs the target's greedy tokens) on held-out documents,
             the container's drafter or an overlay
  train      dense-only fine-tune (routed experts frozen, MXFP4), AdamW, the KL loss to the target's top 32 + rest;
             writes the adapted weights as a RADOVL1 overlay (RADIANCE_WEIGHT_OVERLAY) and evaluates before / after
  selftest   CPU: the overlay writer / reader, the fp8 round trip, the expert autograd vs autograd through dequantised
             weights, the hc mix -- no model needed"""
import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import dspark as DS

MAIN = "dump-dspark_main_normed"


def out(**k):
    print("DAT " + json.dumps(k), flush=True)


# ---------------------------------------------------------------- data
class KL:
    def __init__(self, d):
        self.dir = d
        m = json.load(open(os.path.join(d, "kld.json")))
        self.rows, self.K = m["rows"], m["top"]
        self.docs = []
        off = 0
        for i, x in enumerate(m["docs"]):
            if x.get("score_from", 0) != 0:
                raise ValueError("%s doc %d: score_from must be 0" % (d, i))
            self.docs.append(dict(i=i, source=x["source"], n=x["tokens"], row0=x["row0"], off=off))
            off += x["tokens"]
        self.tok = np.fromfile(os.path.join(d, "tokens.i32"), np.int32)
        assert len(self.tok) == off, (len(self.tok), off)
        top = np.memmap(os.path.join(d, "top.bin"), np.int32, "r", shape=(self.rows, self.K, 2))
        self.top_id = top[:, :, 0]
        self.top_lp = top[:, :, 1].view(np.float32)
        self.rf = np.fromfile(os.path.join(d, "rows.f32"), np.float32).reshape(self.rows, 3)
        dm = json.load(open(os.path.join(d, MAIN + ".json")))
        assert dm["rows"] == self.rows and dm["cols"] == DS.D and dm["row_bytes"] == 2 * DS.D, dm
        self.mx = np.memmap(os.path.join(d, MAIN + ".bin"), np.uint16, "r", shape=(self.rows, DS.D))

    def kind(self, doc):
        return doc["source"].split(":")[0]

    def anchors(self, doc):
        return range(1, doc["n"] - 5)


def bf16_rows(mx, r0, r1, device):
    a = torch.from_numpy(np.ascontiguousarray(mx[r0:r1]).view(np.int16))
    return a.view(torch.bfloat16).to(device)


def span_batch(kl, doc, a0, L, model, need_targets=True):
    """anchors a0 .. a0+L-1 of one document -> the model's inputs (+ targets)"""
    dev = model.dev
    p0 = a0 - DS.WIN                                   # first context position
    lo = max(p0, 0)
    P = a0 + L - 1 - p0                                # context positions p0 .. a0+L-2
    mx = torch.zeros(P, DS.D, dtype=torch.bfloat16, device=dev)
    mx[lo - p0:] = bf16_rows(kl.mx, doc["row0"] + lo, doc["row0"] + a0 + L - 1, dev)
    pos = torch.arange(p0, a0 + L - 1, device=dev)
    valid = pos >= 0
    anchors = torch.arange(a0, a0 + L, device=dev)
    tok = torch.from_numpy(kl.tok[doc["off"]:doc["off"] + doc["n"]].astype(np.int64)).to(dev)
    b = dict(doc=doc, a0=a0, L=L, mx=mx, pos=pos.clamp_min(0), valid=valid, anchors=anchors, anchor_tok=tok[anchors])
    if need_targets:
        rr = doc["row0"] + a0 + np.arange(L)[:, None] + np.arange(DS.BLK)[None, :]       # [L, 5] rows a+i
        b["t_id"] = torch.from_numpy(kl.top_id[rr.reshape(-1)].astype(np.int64)).to(dev).view(L, DS.BLK, -1)
        b["t_lp"] = torch.from_numpy(np.asarray(kl.top_lp[rr.reshape(-1)])).to(dev).view(L, DS.BLK, -1)
        b["t_rest"] = torch.from_numpy(kl.rf[rr.reshape(-1), 2].copy()).to(dev).view(L, DS.BLK)
        b["prev"] = tok[anchors.unsqueeze(-1) + torch.arange(DS.BLK, device=dev)]        # x_{a+i}
        b["next"] = tok[anchors.unsqueeze(-1) + torch.arange(DS.BLK, device=dev) + 1]    # x_{a+i+1}
    return b


def forward_span(model, b):
    """-> hn [L, 5, D] (normed drafter rows)"""
    kctx = []
    W = DS.WIN
    for s in range(len(DS.STAGES)):
        k = model.ctx_keys(s, b["mx"], b["pos"])                                      # [P, 512]
        k = k * b["valid"].unsqueeze(-1).to(k.dtype)
        kctx.append(k.unfold(0, W, 1).permute(0, 2, 1)[:b["L"]])                       # [L, 128, 512]
    km = b["valid"].unfold(0, W, 1)[:b["L"]]
    return model.hidden(b["anchor_tok"], b["anchors"], kctx, km)


def kl_loss(model, hn, b, wpos):
    """the drafter's distribution (head + teacher-forced Markov bias) vs the target's top 32 + rest, per row"""
    L = b["L"]
    z = model.logits(hn.reshape(L * DS.BLK, DS.D), b["prev"].reshape(-1))            # [L*5, V] f32
    lse = torch.logsumexp(z, -1, keepdim=True)
    lq = (z.gather(1, b["t_id"].reshape(L * DS.BLK, -1)) - lse)                        # log q at the target's top ids
    p = b["t_lp"].reshape(L * DS.BLK, -1).exp()
    rest = b["t_rest"].reshape(-1)
    q_top = lq.exp().sum(-1).clamp(max=1 - 1e-6)
    ce = -(p * lq).sum(-1) - rest * torch.log1p(-q_top)
    ent = -(p * b["t_lp"].reshape(L * DS.BLK, -1)).sum(-1)                             # the coarsened entropy (top part)
    kl = (ce - ent).view(L, DS.BLK)
    loss = (kl * wpos).sum(-1).mean()
    top1 = (z.argmax(-1).view(L, DS.BLK) == b["t_id"][:, :, 0]).float()
    return loss, kl.detach().mean(0), top1.mean(0)


@torch.no_grad()
def eval_docs(model, kl, docs, L=128, max_anchors=None, tag=""):
    """simulated fixed-window-5 acceptance: walk proposals vs the target's greedy token at each position, chained only while
    the document's own token follows the proposal (exact for greedy-generated text)"""
    model.eval()
    res = {}
    for doc in docs:
        k = kl.kind(doc)
        S = res.setdefault(k, dict(n=0, reach=[0] * 5, acc=[0] * 5, tps=0.0, top1=[0] * 5, rows=0))
        A = list(kl.anchors(doc))
        if max_anchors and len(A) > max_anchors:
            A = A[:max_anchors]
        for j in range(0, len(A), L):
            a0, n = A[j], min(L, len(A) - j)
            b = span_batch(kl, doc, a0, n, model)
            hn = forward_span(model, b)
            prop = model.walk(hn, b["anchor_tok"]).cpu()                               # [n, 5]
            g = b["t_id"][:, :, 0].cpu()                                               # target greedy at a+i
            nx = b["next"].cpu()                                                       # x_{a+i+1}
            for r in range(n):
                run = 0
                for i in range(DS.BLK):
                    S["reach"][i] += 1
                    if prop[r, i] == g[r, i]:
                        S["acc"][i] += 1
                        run += 1
                        if i + 1 < DS.BLK and nx[r, i] != prop[r, i]:
                            break                                                       # the text left the drafted path
                    else:
                        break
                S["tps"] += 1 + run
                S["n"] += 1
            z = model.logits(hn.reshape(-1, DS.D), b["prev"].reshape(-1)).argmax(-1).view(n, DS.BLK).cpu()
            for i in range(DS.BLK):
                S["top1"][i] += int((z[:, i] == g[:, i]).sum())
            S["rows"] += n
    for k, S in res.items():
        S["cond"] = [round(S["acc"][i] / S["reach"][i], 4) if S["reach"][i] else None for i in range(5)]
        S["tokens_a_step"] = round(S["tps"] / max(1, S["n"]), 4)
        S["top1_tf"] = [round(x / max(1, S["rows"]), 4) for x in S["top1"]]
        del S["tps"], S["top1"]
    model.train()
    return res


# ---------------------------------------------------------------- overlay
def write_overlay(model, path, base=None):
    """the trainable tensors that moved, in the container's dtype and layout, as RADOVL1 (RADIANCE_WEIGHT_OVERLAY)"""
    ents = []
    for n, p in model.trainable():
        en = model.names[n]
        t = p.detach()
        raw = t.cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            data = raw.view(torch.int16).numpy().tobytes()
        else:
            data = raw.to(torch.float32).numpy().tobytes()
        if base is not None and base.get(en) == hashlib.sha1(data).hexdigest():
            continue
        ents.append((en, data))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"RADOVL1\n")
        f.write(struct.pack("<I", len(ents)))
        for en, data in ents:
            nb = en.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<Q", len(data)))
            f.write(data)
    os.replace(tmp, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return len(ents), sum(len(d) for _, d in ents), h


def read_overlay(path):
    d = {}
    with open(path, "rb") as f:
        assert f.read(8) == b"RADOVL1\n"
        n, = struct.unpack("<I", f.read(4))
        for _ in range(n):
            nl, = struct.unpack("<I", f.read(4))
            en = f.read(nl).decode()
            nb, = struct.unpack("<Q", f.read(8))
            d[en] = f.read(nb)
        assert f.read(1) == b""
    return d


def apply_overlay(model, path, rad, skipped=None):
    """the overlay's entries into the port's tensors; the Markov tables too. Entries the port does not hold (dspark.fc /
    dspark.enc.output_norm: the port reads the engine's main_x, computed with the container's fc and main_norm) are listed
    in `skipped`, never applied."""
    ov = read_overlay(path)
    inv = {v: k for k, v in model.names.items()}
    pm = dict(model.named_parameters())
    extra = {"dspark.markov_w1.weight": "mk1", "dspark.markov_w2.weight": "mk2"}
    n = 0
    for en, data in ov.items():
        if en in extra:
            t = getattr(model, extra[en])
            v = torch.from_numpy(np.frombuffer(data, np.int16).copy()).view(torch.bfloat16)
            assert v.numel() == t.numel(), en
            setattr(model, extra[en], v.view(t.shape).to(t.device))
            n += 1
            continue
        if en not in inv:
            if skipped is not None:
                skipped.append(en)
            continue
        p = pm[inv[en]]
        e = rad.entries[en]
        if p.dtype == torch.bfloat16:
            t = torch.from_numpy(np.frombuffer(data, np.int16).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.frombuffer(data, np.float32).copy())
        assert t.numel() == p.numel(), en
        with torch.no_grad():
            p.copy_(t.view(p.shape).to(p.device))
        n += 1
    return n


# ---------------------------------------------------------------- commands
def load_model(a, train=()):
    import radread
    rad = radread.RadFile(a.rad)
    t0 = time.time()
    model = DS.DSpark(rad, a.device, train=train, head_dtype=torch.float32 if a.head32 else torch.bfloat16,
                      expert_aq=a.expert_aq, shared_aq=a.shared_aq, post_w=not a.pre_w, log=lambda s: None)
    if getattr(a, "embed_file", ""):
        # lever DA embedding screen: the drafter's token table replaced by a raw bf16 [n_vocab, D] file (the release's embed.weight)
        e = np.fromfile(a.embed_file, np.int16)
        if e.size != model.embed.numel():
            raise ValueError("%s: %d values, the table has %d" % (a.embed_file, e.size, model.embed.numel()))
        new = torch.from_numpy(e).view(torch.bfloat16).view(model.embed.shape).to(model.dev)
        rel = float((new.float() - model.embed.float()).norm() / model.embed.float().norm())
        model.embed = new
        out(event="embed_replaced", path=a.embed_file, rel_l2_vs_container=round(rel, 5))
    out(event="loaded", s=round(time.time() - t0, 1), device=a.device, trainable=sum(p.numel() for _, p in model.trainable()))
    return rad, model


def split_docs(kl, held_frac, seed):
    rnd = random.Random(seed)
    tr, he = [], []
    for d in kl.docs:
        k = kl.kind(d)
        if k.startswith("val"):
            he.append(d)
        elif rnd.random() < held_frac:
            he.append(d)
        else:
            tr.append(d)
    return tr, he


def validate_one(a, kl, val, by_src):
    """the port's walk proposals at the engine's own anchors, whole documents in spans -> agreement by position"""
    rad, model = load_model(a)
    model.eval()
    agree, n, whole = [0] * 5, 0, 0
    with torch.no_grad():
        for v in val:
            doc = by_src.get(v["source"])
            if doc is None:
                continue
            want = {anc: blk[:5] for anc, blk in v["blocks"] if 1 <= anc and anc + 5 < doc["n"]}
            if not want:
                continue
            lo, hi = min(want), max(want)
            for a0 in range(lo, hi + 1, 128):
                L = min(128, hi + 1 - a0)
                b = span_batch(kl, doc, a0, L, model, need_targets=False)
                prop = model.walk(forward_span(model, b), b["anchor_tok"]).tolist()
                for j in range(L):
                    blk = want.get(a0 + j)
                    if blk is None:
                        continue
                    n += 1
                    for i in range(5):
                        agree[i] += int(prop[j][i] == blk[i])
                    whole += int(prop[j] == blk)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(expert_aq=a.expert_aq, shared_aq=a.shared_aq, post_w=not a.pre_w, head32=a.head32, blocks=n,
                agree_by_pos=[round(x / max(1, n), 4) for x in agree], whole_block=round(whole / max(1, n), 4))


def cmd_validate(a):
    kl = KL(a.data)
    val = json.load(open(a.val))
    by_src = {d["source"]: d for d in kl.docs}
    # --variants 1 (default): exactly the CLI's numerics; N > 1: the first N of the grid below (the pilot's bug: the CLI flags
    # were overwritten by the grid's first entry, so its 4 "variants" were one)
    grid = [(a.expert_aq, a.shared_aq, a.pre_w)] if a.variants <= 1 else \
           [(False, False, False), (True, False, False), (False, False, True), (True, False, True), (False, True, False)][:a.variants]
    best = None
    for ea, sa, pw in grid:
        a.expert_aq, a.shared_aq, a.pre_w = ea, sa, pw
        r = validate_one(a, kl, val, by_src)
        out(event="validate", **r)
        if best is None or (r["whole_block"], r["agree_by_pos"][0]) > (best["whole_block"], best["agree_by_pos"][0]):
            best = r
    out(event="validate_best", **best)
    if a.out:
        json.dump(best, open(a.out, "w"))
    return 0


def cmd_eval(a):
    """--overlays none,P1,P2 (or --overlay P): each evaluated from the container's weights (restored between them) on the
    held-out documents, --kinds filtering them by kind prefix (e.g. gen,val)"""
    kl = KL(a.data)
    _, held = split_docs(kl, a.held, a.seed)
    if a.kinds:
        pre = tuple(a.kinds.split(","))
        held = [d for d in held if kl.kind(d).startswith(pre)]
    rad, model = load_model(a)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    base_mk = (model.mk1.clone(), model.mk2.clone())
    paths = a.overlays.split(",") if a.overlays else [a.overlay or "none"]
    for path in paths:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(base[n])
        model.mk1, model.mk2 = base_mk
        skipped, n_ap = [], 0
        if path and path != "none":
            n_ap = apply_overlay(model, path, rad, skipped)
        r = eval_docs(model, kl, held, max_anchors=a.max_anchors)
        out(event="eval", overlay=path, applied=n_ap, skipped=skipped, docs=len(held), **{k: v for k, v in r.items()})
    return 0


def cmd_train(a):
    torch.manual_seed(a.seed)
    kl = KL(a.data)
    train_docs, held = split_docs(kl, a.held, a.seed)
    groups = tuple(a.train.split(","))
    rad, model = load_model(a, train=groups)
    base = {}
    for n, p in model.trainable():
        raw = p.detach().cpu().contiguous()
        data = raw.view(torch.int16).numpy().tobytes() if raw.dtype == torch.bfloat16 else raw.float().numpy().tobytes()
        base[model.names[n]] = hashlib.sha1(data).hexdigest()
    # fp32 master copies for the bf16 parameters, AdamW on them
    params = [p for _, p in model.trainable()]
    master = [p.detach().float().clone().requires_grad_(True) for p in params]
    opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    wpos = torch.tensor([a.pos_decay ** i for i in range(DS.BLK)], device=model.dev)
    wpos = wpos / wpos.sum() * DS.BLK
    # spans of every training document
    spans = []
    for d in train_docs:
        A = list(kl.anchors(d))
        for j in range(0, len(A), a.span):
            spans.append((d, A[j], min(a.span, len(A) - j)))
    n_anchor = sum(s[2] for s in spans)
    steps = max(1, int(a.epochs * n_anchor / (a.span * a.spans_per_step)))
    out(event="train_start", docs=len(train_docs), held=len(held), anchors=n_anchor, steps=steps, lr=a.lr, groups=groups,
        span=a.span, spans_per_step=a.spans_per_step, epochs=a.epochs, pos_decay=a.pos_decay)
    if not a.skip_eval:
        out(event="eval_before", **eval_docs(model, kl, held, max_anchors=a.max_anchors))
    t0 = time.time()
    rnd = random.Random(a.seed)
    order = []
    step = 0
    warm = max(1, int(0.05 * steps))
    best = None
    while step < steps:
        if not order:
            order = list(range(len(spans)))
            rnd.shuffle(order)
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        tot, kls, t1s = 0.0, torch.zeros(DS.BLK, device=model.dev), torch.zeros(DS.BLK, device=model.dev)
        for k in range(a.spans_per_step):
            if not order:
                break
            d, a0, L = spans[order.pop()]
            b = span_batch(kl, d, a0, L, model)
            hn = forward_span(model, b)
            loss, klp, t1 = kl_loss(model, hn, b, wpos)
            (loss / a.spans_per_step).backward()
            tot += float(loss) / a.spans_per_step
            kls += klp / a.spans_per_step
            t1s += t1 / a.spans_per_step
        gn = 0.0
        for p, mp in zip(params, master):
            mp.grad = p.grad.float() if p.grad is not None else torch.zeros_like(mp)
            gn += float(mp.grad.square().sum())
            p.grad = None
        gn = math.sqrt(gn)
        if a.clip > 0 and gn > a.clip:
            for mp in master:
                mp.grad.mul_(a.clip / gn)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, mp in zip(params, master):
                p.copy_(mp.to(p.dtype))
        step += 1
        if step == 5 and a.max_minutes:
            # fit the schedule to the time budget: the cosine runs over the steps that fit (95 % of it, the rest for the eval)
            per = (time.time() - t0) / 5.0
            fit = max(10, int(0.95 * 60 * a.max_minutes / per))
            if fit < steps:
                out(event="steps_fitted", planned=steps, fitted=fit, s_per_step=round(per, 3))
                steps = fit
        if step % a.log_every == 0 or step == steps:
            out(event="step", step=step, of=steps, loss=round(tot, 5), kl_by_pos=[round(float(x), 4) for x in kls],
                top1_tf=[round(float(x), 4) for x in t1s], grad_norm=round(gn, 4), lr=lr, s=round(time.time() - t0, 1))
        if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
            out(event="time_limit", step=step, of=steps)
            break
    n, nb, h = write_overlay(model, a.out, base)
    out(event="overlay_written", path=a.out, entries=n, bytes=nb, sha256=h, s=round(time.time() - t0, 1))
    if not a.skip_eval:
        out(event="eval_after", **eval_docs(model, kl, held, max_anchors=a.max_anchors))
    return 0


def cmd_selftest(a):
    torch.manual_seed(0)
    # fp8 round trip: idempotent, power-of-two scales
    x = torch.randn(4, 64, dtype=torch.bfloat16) * 3
    y = DS.fp8_rt(x)
    assert torch.equal(DS.fp8_rt(y), y), "fp8 round trip not idempotent"
    # expert autograd vs plain autograd through dequantised weights
    gu_c = torch.randint(0, 256, (2 * DS.NFF, 64 // 2), dtype=torch.uint8)
    gu_e = torch.randint(120, 128, (2 * DS.NFF, 64 // 32), dtype=torch.uint8)
    dn_c = torch.randint(0, 256, (64, DS.NFF // 2), dtype=torch.uint8)
    dn_e = torch.randint(118, 126, (64, DS.NFF // 32), dtype=torch.uint8)
    xs = (torch.randn(5, 64) * 0.5).to(torch.bfloat16).requires_grad_(True)
    ws = torch.rand(5, 1, requires_grad=True)
    yo = DS._Expert.apply(xs, ws, gu_c, gu_e, dn_c, dn_e, False, False)
    gy = torch.randn_like(yo)
    gx, gw = torch.autograd.grad(yo, (xs, ws), gy)
    xr = xs.detach().float().requires_grad_(True)
    wr = ws.detach().clone().requires_grad_(True)
    Wgu, Wd = DS.fp4_dequant(gu_c, gu_e, torch.float32), DS.fp4_dequant(dn_c, dn_e, torch.float32)
    gu = xr @ Wgu.t()
    g, u = gu[:, :DS.NFF].clamp(max=DS.SWIGLU), gu[:, DS.NFF:].clamp(-DS.SWIGLU, DS.SWIGLU)
    yr = (F.silu(g) * u * wr) @ Wd.t()
    gxr, gwr = torch.autograd.grad(yr, (xr, wr), gy)
    ex = float((gx.float() - gxr).abs().max() / (gxr.abs().max() + 1e-9))
    ew = float((gw - gwr).abs().max() / (gwr.abs().max() + 1e-9))
    assert ex < 0.05 and ew < 0.05, (ex, ew)
    # hc mix: comb doubly stochastic
    h = torch.randn(2, 3, DS.HC, 16)
    fn = torch.randn((2 + DS.HC) * DS.HC, DS.HC * 16) * 0.1
    pre, post, comb = DS.hc_mixes(h, fn, torch.ones(3), torch.zeros((2 + DS.HC) * DS.HC))
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, DS.HC), atol=1e-3), comb.sum(-2)
    # rope: inverse undoes forward
    cs = DS.rope_tab(torch.arange(5), "cpu")
    q = torch.randn(5, 512)
    assert torch.allclose(DS.rope(DS.rope(q, cs), cs, inverse=True), q, atol=1e-4)
    # overlay round trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        class M:
            pass
        m = M()
        p1 = torch.nn.Parameter(torch.randn(3, 4).to(torch.bfloat16))
        p2 = torch.nn.Parameter(torch.randn(7))
        m.names = {"a": "blk.40.attn_q_a.weight", "b": "blk.40.attn_sinks.weight"}
        m.trainable = lambda: [("a", p1), ("b", p2)]
        n, nb, hh = write_overlay(m, os.path.join(td, "o.ovl"))
        d = read_overlay(os.path.join(td, "o.ovl"))
        assert n == 2 and len(d["blk.40.attn_q_a.weight"]) == 24 and len(d["blk.40.attn_sinks.weight"]) == 28
    out(event="selftest", ok=True, expert_grad_rel_err=[round(ex, 5), round(ew, 5)], torch=torch.__version__,
        hip=torch.version.hip, cuda_available=torch.cuda.is_available(),
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "eval", "train", "selftest"])
    ap.add_argument("--rad", default="/m/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad")
    ap.add_argument("--data", default="")
    ap.add_argument("--val", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--overlay", default="")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--head32", action="store_true", help="the head in f32 (validation); default bf16")
    ap.add_argument("--expert-aq", action="store_true", help="fp8 activations into the routed experts (the pilot's; not the engine's)")
    ap.add_argument("--no-expert-aq", action="store_true", help="accepted for the pilot's command lines; the default now")
    ap.add_argument("--pre-w", action="store_true", help="routing weight before the down GEMM (the reference's; the engine weights after)")
    ap.add_argument("--shared-aq", action="store_true")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--train", default="attn,shexp,router,hc,norm")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--span", type=int, default=128)
    ap.add_argument("--spans-per-step", type=int, default=4)
    ap.add_argument("--pos-decay", type=float, default=0.8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--held", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-anchors", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--embed-file", default="", help="eval: a raw bf16 [n_vocab, D] token table in place of the container's")
    ap.add_argument("--overlays", default="", help="eval: none,PATH,... each from the container's weights")
    ap.add_argument("--kinds", default="", help="eval: held-out documents whose kind starts with one of these (comma list)")
    a = ap.parse_args()
    return {"validate": cmd_validate, "eval": cmd_eval, "train": cmd_train, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
