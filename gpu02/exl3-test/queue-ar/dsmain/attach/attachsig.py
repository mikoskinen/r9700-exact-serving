#!/usr/bin/env python3
"""attachsig.py (12.11.2026, queue-ar/dsmain/attach, ds-main-port ATTACH builder) -- the helpers of attach.sh, standard library only.

  sig --yaml Y --image-id ID --args ARGS [--envfile F] [--text OUT]
        the START SIGNATURE of a DS test server: prints "<sha256> <stats writers | ->". Hashed, one line each (OUT gets the same
        lines, secret names masked): the image ID; restart()'s serve line (ARGS = serve_args); the md5 of the yaml and of its
        extends file(s); every ${VAR} the yaml / extends files substitute (MODEL, MODELDIR, PROFEVERY, PROFSYNC, AQLSTAMP, the
        production .env names ...) except IMAGE (its ID is used), STATSDIR, XS2_ENVFILE, ARGS, COMPOSE_PROJECT_NAME and every
        *KEY* / *TOKEN* / *SECRET* / *PASS* name; every exported RADIANCE_ / R4D_ / HSA_ / HIP_ / ROCR_ / GPU_ / AMD_ / HCC_ /
        MIOPEN_ / RCCL_ / NCCL_ word (a superset of what reaches the container: the stack5 family's stackenv exports every LEVENV
        word it also writes to the env file); with --envfile (a real start only) each env-file line the shell does not carry with
        the same value ("envfile_only"). fn4 switches are runtime state and are not in it. "Stats writers": start words whose
        output lands under the container's /stats (a value starting /stats, RADIANCE_AQL_STAMP when AQLSTAMP is set).
  fn4diff BASE CUR      the switches to POST so the server is back at its start snapshot BASE (GET /admin/fn4 right after the
                        healthy start): a JSON object, {} = nothing differs (pending switches count as applied)
  fn4check BASE CUR     "OK epoch N" when every switch = BASE and nothing is pending, else "FAIL {...} pending {...}"
  fn4ok FILE            exit 0 when FILE is a GET /admin/fn4 answer with a switches object
  bump JSON             the JSON object + "epoch_bump": 1
  keys JSON             "N [k1, k2, ...]"
  sigdiff A B           the first 3 differing signature lines of two --text files, each as "name: 'A...' -> 'B...'" around the first
                        differing character (values from the text files: secret names are masked there)
  idle STATS            "IDLE" (/stats running 0 and waiting 0), "BUSY running R waiting W" or "UNKNOWN ..."
  selftest
CUR / STATS may be "-" (stdin)."""
import hashlib, json, os, re, sys

SECRET = re.compile(r"KEY|TOKEN|SECRET|PASS", re.I)
ENVPFX = re.compile(r"^(RADIANCE_|R4D_|HSA_|HIP_|ROCR_|GPU_|AMD_|HCC_|MIOPEN_|RCCL_|NCCL_)")
NOVAR = {"IMAGE", "STATSDIR", "XS2_ENVFILE", "ARGS", "COMPOSE_PROJECT_NAME"}
VARRE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")


def md5f(p):
    try:
        with open(p, "rb") as f: return hashlib.md5(f.read()).hexdigest()
    except OSError: return "missing"


def subst(s, env):
    def rep(m):
        v = env.get(m.group(1))
        return v if v else (m.group(3) or "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:?[-?])?([^}]*)\}", rep, s)


def extends_files(yf, env, seen):
    out = []
    try: txt = open(yf).read()
    except OSError: return out
    for m in re.finditer(r"extends:\s*\{[^}]*?file:\s*([^,}\s]+)|extends:\s*\n\s+file:\s*(\S+)", txt):
        p = subst((m.group(1) or m.group(2)).strip("'\""), env)
        if not os.path.isabs(p): p = os.path.join(os.path.dirname(os.path.abspath(yf)), p)
        p = os.path.normpath(p)
        if p in seen: continue
        seen.add(p); out.append(p); out += extends_files(p, env, seen)
    return out


def siglines(yaml, image_id, args, envfile, env):
    L = ["v1 attach start signature", "image_id=" + image_id, "args=" + " ".join(args.split())]
    L.append("yaml_md5=" + md5f(yaml))
    ext = extends_files(yaml, env, {os.path.normpath(os.path.abspath(yaml))})
    for p in ext: L.append("extends_md5[%s]=%s" % (os.path.basename(p), md5f(p)))
    names = set()
    for p in [yaml] + ext:
        try: names |= set(VARRE.findall(open(p).read()))
        except OSError: pass
    hidden = []
    for n in sorted(names - NOVAR):
        if SECRET.search(n): continue
        L.append("var %s=%s" % (n, env.get(n, "")))
    for k in sorted(env):
        if ENVPFX.match(k):
            if SECRET.search(k): hidden.append("env %s=%s" % (k, env[k])); continue
            L.append("env %s=%s" % (k, env[k]))
    if envfile:
        try: lines = open(envfile).read().splitlines()
        except OSError: lines = ["#unreadable " + envfile]
        for l in lines:
            l = l.strip()
            if not l or l.startswith("#") and not l.startswith("#unreadable"):
                continue
            k, _, v = l.partition("=")
            if env.get(k) != v:
                (hidden if SECRET.search(k) else L).append("envfile_only %s=%s" % (k, v))
    sw = []
    for l in L:
        m = re.match(r"(?:env|envfile_only) ([A-Z0-9_]+)=(/stats.*)$", l)
        if m and m.group(1) != "RADIANCE_AQL_STAMP_OUT": sw.append(m.group(1))
    if env.get("AQLSTAMP") or env.get("RADIANCE_AQL_STAMP"): sw.append("RADIANCE_AQL_STAMP")
    h = hashlib.sha256("\n".join(L + hidden).encode()).hexdigest()
    return h, sorted(set(sw)), L + ["%s=<masked>" % x.split("=", 1)[0] for x in hidden]


def load(p):
    s = sys.stdin.read() if p == "-" else open(p).read()
    return json.loads(s)


def eff_switches(cur):
    sw = dict(cur.get("switches") or {})
    pd = cur.get("pending")
    if isinstance(pd, dict): sw.update({k: v for k, v in pd.items() if k != "epoch_bump"})
    return sw


def fn4diff(base, cur):
    b, e = base["switches"], eff_switches(cur)
    return {k: v for k, v in sorted(b.items()) if e.get(k) != v}


def fn4check(base, cur):
    b, c = base["switches"], cur.get("switches") or {}
    bad = {k: [c.get(k), v] for k, v in sorted(b.items()) if c.get(k) != v}
    if not bad and not cur.get("pending"): return "OK epoch %s" % cur.get("epoch")
    return "FAIL %s pending %s" % (json.dumps(bad)[:600], json.dumps(cur.get("pending"))[:200])


def idle(s):
    r, w = s.get("running"), s.get("waiting")
    if not isinstance(r, int) or not isinstance(w, int): return "UNKNOWN running %r waiting %r" % (r, w)
    return "IDLE" if r == 0 and w == 0 else "BUSY running %d waiting %d" % (r, w)


def sigdiff(a, b, n=3):
    def rd(p):
        d = {}
        try:
            for l in open(p).read().splitlines():
                k, _, v = l.partition("=")
                d[k] = v
        except OSError: pass
        return d
    A, B = rd(a), rd(b)
    out = []
    for k in sorted(set(A) | set(B)):
        x, y = A.get(k), B.get(k)
        if x == y: continue
        if x is None or y is None:
            out.append("%s: %s" % (k, "only this start (%r)" % y[:60] if x is None else "only the server's (%r)" % x[:60]))
        else:
            i = next((j for j, (c, e) in enumerate(zip(x, y)) if c != e), min(len(x), len(y)))
            lo = max(0, i - 24)
            out.append("%s: %r -> %r" % (k, ("..." if lo else "") + x[lo:i + 24], ("..." if lo else "") + y[lo:i + 24]))
        if len(out) >= n: break
    return "; ".join(out) or "no differing line"


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    base = os.path.join(d, "base.yaml"); y = os.path.join(d, "t.yaml"); ef = os.path.join(d, "e.env")
    open(base, "w").write("services:\n  radiance:\n    environment:\n      RADIANCE_API_KEY: ${RADIANCE_API_KEY:?x}\n    user: ${RADIANCE_UID:-1000}\n")
    open(y, "w").write("services:\n  radiance:\n    extends: { file: base.yaml, service: radiance }\n    image: ${IMAGE:?}\n"
                       "    environment:\n      RADIANCE_PROFILE_EVERY: ${PROFEVERY:-}\n      RADIANCE_AQL_STAMP: ${AQLSTAMP:-}\n"
                       "      RADIANCE_AQL_STAMP_OUT: /stats/stamps.csv\n    volumes:\n      - ${STATSDIR:?}:/stats\n"
                       "    env_file:\n      - path: ${XS2_ENVFILE:-/dev/null}\n    command: --model /models/${MODEL:?} ${ARGS:?}\n")
    open(ef, "w").write("RADIANCE_A=1\nRADIANCE_B=2\n")
    env = {"MODEL": "m.rad", "RADIANCE_A": "1", "RADIANCE_B": "2", "STATSDIR": "/x/stats", "RADIANCE_API_KEY": "s3cret", "PATH": "/bin"}
    h1, sw1, t1 = siglines(y, "sha256:1", "--tp 4  --x", ef, env)
    h2, _, _ = siglines(y, "sha256:1", "--tp 4 --x", None, env)                        # predict = real when the file is covered
    h3, _, _ = siglines(y, "sha256:1", "--tp 4 --x", ef, dict(env, STATSDIR="/y"))      # STATSDIR is not in it
    h4, _, _ = siglines(y, "sha256:2", "--tp 4 --x", ef, env)                           # another image
    h5, _, _ = siglines(y, "sha256:1", "--tp 4 --x", ef, dict(env, RADIANCE_B="3"))     # a start word (+ envfile_only B=2)
    h6, _, _ = siglines(y, "sha256:1", "--tp 4 --x --mnbt 1", ef, env)                  # the serve line
    h7, _, _ = siglines(y, "sha256:1", "--tp 4 --x", ef, dict(env, RADIANCE_API_KEY="other"))   # a secret changes the hash only
    h8, sw8, _ = siglines(y, "sha256:1", "--tp 4 --x", ef, dict(env, RADIANCE_DS41_BITRACE="/stats/bt", AQLSTAMP="50"))
    open(ef, "a").write("RADIANCE_C=9\n")
    h9, _, t9 = siglines(y, "sha256:1", "--tp 4 --x", ef, env)                           # an env-file-only line
    ok = h1 == h2 == h3 and len({h1, h4, h5, h6, h7, h9}) == 6 and sw1 == [] and sw8 == ["RADIANCE_AQL_STAMP", "RADIANCE_DS41_BITRACE"]
    ok = ok and "s3cret" not in "\n".join(t1) and any(l == "envfile_only RADIANCE_C=9" for l in t9) and any(l.startswith("extends_md5[base.yaml]=") for l in t1)
    ok = ok and "var RADIANCE_UID=" in t1 and not any("STATSDIR" in l for l in t1) and "var PROFEVERY=" in t1
    b = {"switches": {"a": 1, "b": 0}, "pending": {}, "epoch": 0}
    ok = ok and fn4diff(b, {"switches": {"a": 1, "b": 2}, "pending": {}}) == {"b": 0}
    ok = ok and fn4diff(b, {"switches": {"a": 1, "b": 0}, "pending": {"a": 5, "epoch_bump": 1}}) == {"a": 1}
    ok = ok and fn4check(b, {"switches": {"a": 1, "b": 0}, "pending": {}, "epoch": 3}) == "OK epoch 3"
    ok = ok and fn4check(b, {"switches": {"a": 1, "b": 0}, "pending": {"a": 1}, "epoch": 3}).startswith("FAIL")
    ok = ok and idle({"running": 0, "waiting": 0}) == "IDLE" and idle({"running": 1, "waiting": 0}).startswith("BUSY") and idle({}).startswith("UNKNOWN")
    ta, tb = os.path.join(d, "a.txt"), os.path.join(d, "b.txt")
    open(ta, "w").write("args=--tp 4 --max-num-batched-tokens 4160 --x\nenv RADIANCE_A=1\n")
    open(tb, "w").write("args=--tp 4 --max-num-batched-tokens 4096 --x\nenv RADIANCE_B=2\n")
    sd = sigdiff(ta, tb)
    ok = ok and "4160" in sd and "4096" in sd and "env RADIANCE_A: only the server's" in sd and "env RADIANCE_B: only this start" in sd
    print("SELFTEST OK" if ok else "SELFTEST FAIL %s" % [h1[:8], h2[:8], h3[:8], h4[:8], h5[:8], h6[:8], h7[:8], h9[:8], sw1, sw8])
    return 0 if ok else 1


def main(a):
    if not a: print(__doc__); return 2
    c = a[0]
    if c == "sig":
        o = {"--yaml": None, "--image-id": None, "--args": None, "--envfile": None, "--text": None}
        i = 1
        while i < len(a):
            if a[i] in o and i + 1 < len(a): o[a[i]] = a[i + 1]; i += 2
            else: print("sig: bad argument %s" % a[i], file=sys.stderr); return 2
        if not o["--yaml"] or not o["--image-id"] or o["--args"] is None: print("sig: --yaml, --image-id and --args are needed", file=sys.stderr); return 2
        h, sw, L = siglines(o["--yaml"], o["--image-id"], o["--args"], o["--envfile"], dict(os.environ))
        if o["--text"]:
            with open(o["--text"], "w") as f: f.write("\n".join(L) + "\n")
        print(h, ",".join(sw) or "-")
        return 0
    if c == "fn4diff": print(json.dumps(fn4diff(load(a[1]), load(a[2])))); return 0
    if c == "fn4check": print(fn4check(load(a[1]), load(a[2]))); return 0
    if c == "fn4ok":
        try: d = load(a[1]); return 0 if isinstance(d.get("switches"), dict) and d["switches"] else 1
        except Exception: return 1
    if c == "bump": d = json.loads(a[1]); d["epoch_bump"] = 1; print(json.dumps(d)); return 0
    if c == "keys": d = json.loads(a[1]); print("%d %s" % (len(d), ",".join(sorted(d)) or "-")); return 0
    if c == "idle":
        try: print(idle(load(a[1])))
        except Exception as e: print("UNKNOWN (%s)" % type(e).__name__)
        return 0
    if c == "sigdiff": print(sigdiff(a[1], a[2])); return 0
    if c == "selftest": return selftest()
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
