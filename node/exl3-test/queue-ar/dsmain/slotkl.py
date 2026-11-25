#!/usr/bin/env python3
"""slotkl.py (10.11.2026, queue-ar dsmain, ds41-slot agent, lever D3s) -- the short quality check of an arithmetic switch,
IN ONE PROCESS: the KL divergence between the next-token distributions of arm A and arm B (two settings of runtime
switches, POST /admin/fn4), teacher-forced, from the server's own logprobs.

  slotkl.py --url URL --a '{"ds41_slotsum": 0}' --b '{"ds41_slotsum": 1}' --out PREFIX [--docs 40] [--gen 128]
            [--chars 1600] [--top 20] [--aa]

What Radiance returns (core/server/oai.cpp, sampler.h): logprobs for GENERATED tokens only -- on /v1/completions an integer
`logprobs` N (<= RAD_MAX_LOGPROBS = 20) gives, per token, its id, its raw logprob and the N most likely alternatives with ids
(`choices[0].logprobs.content[i].top_logprobs`); there are no prompt logprobs. A prompt's positions are computed by prefill,
and lever D3s leaves prefill on today's path, so prompt scoring could not see it anyway: the lever acts on decode-band steps.
So the check is teacher-forced through generation: every document (a slice of the Python standard library's source, the
text ds41_tests.py's text_chars builds its prompts from, never a user transcript) is continued greedily for --gen tokens
under A, then under B, one request at a time (c1: each decode step is one row, the same batch shapes in both arms). Up to and
including the first position where the two greedy continuations differ, both arms computed the next-token distribution from
the SAME prefix, so those positions are teacher-forced; after it the prefixes differ and the positions are not used.

Per teacher-forced position: KL(P_A || P_B) on the tokens in both top-N lists plus one bucket for every other token (each
side's remaining mass), which by the data-processing inequality is at most the full KL and loses only what the two spread
outside their shared head (the core's --kld-top coarsening); the top-1 agreement; |logprob_A - logprob_B| of A's token.
Reported: positions, mean / p50 / p99 / p99.9 / max KL, top-1 agreement, mean and max |dlogprob|, documents whose greedy
continuation diverged and where. --aa repeats A after B (A2): A vs A2 must be identical at c1 (the run-to-run floor of this
arm); B vs A is then the lever. Prints one "SLOTKL {json}" line; per-position rows go to PREFIX-pos.jsonl, each arm's raw
answers to PREFIX-<arm>.jsonl. Standard library only."""
import argparse, glob, json, math, os, sys, sysconfig, time, urllib.error, urllib.request


def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def docs(n, chars):
    """n slices of the standard library's source, `chars` each, every one from its own file (deterministic)."""
    src = sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/**/*.py", recursive=True))
    out, i = [], 0
    step = max(1, len(src) // max(1, n))
    while len(out) < n and i < len(src) * 2:
        f = src[(i * step + i // len(src)) % len(src)]
        i += 1
        try:
            s = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(s) < chars + 200:
            continue
        off = (len(s) - chars) // 3
        out.append(f"# === {os.path.basename(f)} ===\n" + s[off:off + chars])
    return out


def model_id(url):
    try:
        return (http(url + "/v1/models", timeout=60).get("data") or [{}])[0].get("id") or "default"
    except Exception:
        return "default"


def switch(url, sw):
    """POST the arm's switches with an epoch bump, then a 4-token request so they apply at a step; -> GET state."""
    body = dict(sw)
    body["epoch_bump"] = 1
    http(url + "/admin/fn4", body, timeout=15)
    try:
        http(url + "/v1/completions", {"model": model_id(url), "prompt": "Hello", "max_tokens": 4, "temperature": 0}, 120)
    except Exception:
        pass
    st = http(url + "/admin/fn4", timeout=15)
    bad = [k for k, v in sw.items() if st.get("switches", {}).get(k) != v]
    if bad or st.get("pending"):
        raise RuntimeError("switches not applied: wanted %s, got %s" % (json.dumps(sw), json.dumps(st)))
    return st


def run_arm(url, model, prompts, gen, top, path):
    rows = []
    with open(path, "w") as fo:
        for i, p in enumerate(prompts):
            t0 = time.time()
            d = http(url + "/v1/completions", {"model": model, "prompt": p, "max_tokens": gen, "temperature": 0,
                                               "top_p": 1, "logprobs": top, "stream": False})
            c = d["choices"][0]
            lp = (c.get("logprobs") or {}).get("content") or []
            r = {"i": i, "text": c.get("text"), "finish": c.get("finish_reason"), "s": round(time.time() - t0, 3),
                 "ids": [e.get("id") for e in lp], "lp": [e.get("logprob") for e in lp],
                 "top": [{str(a.get("id")): a.get("logprob") for a in (e.get("top_logprobs") or [])} for e in lp]}
            rows.append(r)
            fo.write(json.dumps(r) + "\n")
    return rows


def kl_coarse(ta, tb):
    """KL(P_A || P_B) over the shared head and one tail bucket; ta / tb: {id: logprob}."""
    shared = set(ta) & set(tb)
    if not shared:
        return None
    pa = {k: math.exp(ta[k]) for k in shared}
    pb = {k: math.exp(tb[k]) for k in shared}
    ra = max(0.0, 1.0 - sum(pa.values()))
    rb = max(0.0, 1.0 - sum(pb.values()))
    kl = sum(pa[k] * (ta[k] - tb[k]) for k in shared)
    if ra > 1e-12:
        kl += ra * (math.log(ra) - math.log(max(rb, 1e-30)))
    return max(0.0, kl)


def compare(A, B, posf=None, tag="B vs A"):
    kls, agree, dlp, n_pos, div, divpos, identical = [], 0, [], 0, 0, [], 0
    for a, b in zip(A, B):
        ia, ib = a["ids"], b["ids"]
        n = min(len(ia), len(ib))
        d = next((k for k in range(n) if ia[k] != ib[k]), None)
        if d is None and len(ia) == len(ib) and a.get("text") == b.get("text"):
            identical += 1
        if d is not None:
            div += 1
            divpos.append(d)
        last = n - 1 if d is None else d          # positions 0..last share their prefix
        for k in range(last + 1):
            ta, tb = a["top"][k], b["top"][k]
            kl = kl_coarse(ta, tb)
            if kl is None:
                continue
            n_pos += 1
            kls.append(kl)
            ma = max(ta, key=ta.get) if ta else None
            mb_ = max(tb, key=tb.get) if tb else None
            agree += ma == mb_
            if a["lp"][k] is not None and str(ia[k]) in tb:
                dlp.append(abs(a["lp"][k] - tb[str(ia[k])]))
            if posf:
                posf.write(json.dumps({"cmp": tag, "doc": a["i"], "pos": k, "kl": kl, "top1_same": ma == mb_,
                                       "lp_a": a["lp"][k], "lp_b": b["lp"][k] if k < len(b["lp"]) else None}) + "\n")
    kls.sort()
    q = lambda f: kls[min(len(kls) - 1, int(f * len(kls)))] if kls else None
    return {"positions": n_pos, "kl_mean": sum(kls) / len(kls) if kls else None, "kl_p50": q(0.5), "kl_p99": q(0.99),
            "kl_p999": q(0.999), "kl_max": kls[-1] if kls else None, "top1_agree": agree / n_pos if n_pos else None,
            "dlogprob_mean": sum(dlp) / len(dlp) if dlp else None, "dlogprob_max": max(dlp) if dlp else None,
            "docs": len(A), "docs_identical": identical, "docs_diverged": div,
            "first_divergence": sorted(divpos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--a", required=True, help="arm A's switches (JSON object)")
    ap.add_argument("--b", required=True, help="arm B's switches (JSON object)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--chars", type=int, default=1600)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--aa", action="store_true", help="repeat A after B: the A/A floor")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    sa, sb = json.loads(a.a), json.loads(a.b)
    model = model_id(url)
    prompts = docs(a.docs, a.chars)
    t0 = time.time()
    out = {"docs": len(prompts), "gen": a.gen, "top": a.top, "a": sa, "b": sb}
    try:
        switch(url, sa)
        A = run_arm(url, model, prompts, a.gen, a.top, a.out + "-A.jsonl")
        switch(url, sb)
        B = run_arm(url, model, prompts, a.gen, a.top, a.out + "-B.jsonl")
        A2 = None
        if a.aa:
            switch(url, sa)
            A2 = run_arm(url, model, prompts, a.gen, a.top, a.out + "-A2.jsonl")
    except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as e:
        out["error"] = str(e)[:400]
        print("SLOTKL " + json.dumps(out))
        sys.exit(1)
    if not any(r["top"] and r["top"][0] for r in A):
        out["error"] = "no top_logprobs in the answers (the server's logprobs are off?)"
        print("SLOTKL " + json.dumps(out))
        sys.exit(1)
    with open(a.out + "-pos.jsonl", "w") as pf:
        out["b_vs_a"] = compare(A, B, pf, "B vs A")
        if A2 is not None:
            out["a2_vs_a"] = compare(A, A2, pf, "A2 vs A")
    out["wall_s"] = round(time.time() - t0, 1)
    print("SLOTKL " + json.dumps(out))


if __name__ == "__main__":
    main()
