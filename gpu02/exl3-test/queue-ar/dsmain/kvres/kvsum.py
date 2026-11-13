#!/usr/bin/env python3
"""kvsum.py (13.11.2026, queue-ar/dsmain/kvres, a builder for ds-main-port) -- the KV-reservation vs expert-residency trade-off on int16
(item 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres.sh, body kvres/dsm-kvres.sh). New; the step-log readers (opn / tsec /
read_log / wall / is_nc) are zzzc16/zc16sum.py's (= zzzfuse/fusesum.py's), verbatim.

THE KNOB (radiance-int14c, read in the source, not guessed): --host-pool-mib (lib-pr266 HOSTPOOL, the body's HP29; 19456 in every DS start).
  core/place/planner.cpp:103 expert_vram_floor = all experts + spare slots + 2 x the largest unit - the host room, and the host room is
  --host-pool-mib (core/engine_bringup.cpp:2250-2255); core/mem/vram_budget.cpp:189-199 with a lending cache: experts = that floor, the KV cache =
  elastic - floor (>= one --max-model-len sequence), the prefill staging ring out of the cache. So host pool -X MiB = expert floor +X = KV pool -X
  (less the staging ring's shrink), with --max-model-len 524288, --max-num-seqs 16 and --gpu-headroom-mib untouched. The KV cache LENDS every block
  no live request holds to the expert slab (core/engine.cpp:2141 loan_tick, core/mem/kv.cpp:882 spare_tokens), so the arms move units between
  OWNED slab slots and KV-LENT slots; the VRAM-resident total moves only where KV is held.

  kvsum.py place START_LOG --arm A --hp N [--stats ST.json] [--ref REF_START_LOG] [--ref-hp 19456] [--sw 1] [--json OUT]
      the start log's placement read-back -> one line "KVPLACE A ...": --host-pool-mib in force (must be N), the VRAM budget split (experts /
      kv cache / prefill staging), the owned slab slots a rank (slab - shadow), the KV-lendable / arena-lendable / drafter-lendable slots, the
      KV pool in tokens and sessions (the engine's own "the cache can grow to .. tokens of context, S sessions at 524288" and "the pool backs P %
      of the worst case" lines) + sessions x 128k / 260k / 500k, the deltas vs the reference start (zzzc16's int16 start, --host-pool-mib 19456);
      /stats right after the pin (the lent covered / held a rank). VIRHE: the knob not in force, or an arm below the reference pool whose
      owned slots did not grow / KV-lendable slots did not shrink (the split did not move) -> the body skips that start's readings.
  kvsum.py OUT --arms K35:16800,K60:14900 [--sw 1] [--ref-zf DIR] [--ref-zc DIR] [--c8gens 2] [--c16gens 1] [--u1n 3] [--essay-md5 M] [--json J]
      every "KVRES ..." line: PLACE / IDENTITY / c8 BLOCK / c8 / c16 BLOCK / c16 / 1u / <ARM> (one line an arm) / TABLE / SLOPE.
      References (no K0 re-run, TESTING.md 2.): c8 + c16 = zzzfuse's F blocks (int16 switches on the int15 pin, the same gens method:
      c8-2-G-F / c8-3-G-F agg 192.2 / 191.7 = 191.95 tok/s, full-row (8 seqs, 8 rows) p50 33.62 / 33.63 ms; c16-2-G-F agg 237.0, full-row (16, 16)
      51.42 ms; their window residency read from zzzfuse's raw-snaps.tar.gz); 1u = zzzc16's F reads (one discarded + 3 a block, resprobe essay
      512: 90.77 tok/s, 17.37 ms a step); the placement = zzzc16's start log (identical to zzzfuse's: 1849 / 1865 slab slots, 748 KV-lendable,
      KV 6.95 GiB, 2590442 tokens).
  kvsum.py selftest
Standard library only. Read only (writes only --json)."""
import bisect, gzip, io, json, os, re, statistics, sys, tarfile, tempfile, zlib
from datetime import datetime, timezone

TSRE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ")
STEP = re.compile(r" step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)(.*)$")
SEG = re.compile(r"\[\d+\] slot=(\d+) q=(\d+) ctx=(\d+)")
QMIN = 16      # a prefill segment has more rows than any decode / verify segment (draft depth <= 15)
NCROWS = 512   # a non-decode step of >= 512 rows is a newcomer step even when its prefill entry is not printed
IDLE = 3.0     # a step followed by > 3 s of nothing is a host gap, not a wall
REFHP = 19456
SESS = (("128k", 128000), ("260k", 260000), ("500k", 500000), ("524288", 524288))
SLOPE_REF = -2.0   # vramsweep 10.11.: c8 ms a decode step per +100 resident units a rank
MINDU = 20         # |delta units| below this: no slope
# the references (other starts; TESTING.md 2.: no baseline re-run)
REF_C8 = dict(src="zzzfuse F (c8-2-G-F / c8-3-G-F, gens 2)", blocks=("c8-2-G-F", "c8-3-G-F"), agg=(192.2, 191.7), bkt=(33.62, 33.63), dec=(33.62, 33.63), rows=8)
REF_C16 = dict(src="zzzfuse F (c16-2-G-F, gens 1)", blocks=("c16-2-G-F",), agg=(237.0,), bkt=(51.42,), dec=(51.42,), rows=16)
REF_1U = dict(src="zzzc16 F (6 reads, one discarded + 3 a block)", tps=90.77, ms=17.37, mb=0.0)
REF_PLACE = dict(src="zzzc16 start (constants)", hp=19456, owned=[1801, 1817, 1817, 1817], kvlend=[748] * 4, arlend=[384, 383, 383, 383], dlend=[221] * 4,
                 experts_gib=17.34, kv_gib=6.95, stage_gib=1.02, tokens=2590442, sessions=4.9, backs_pct=32, fl_sessions=5.1)


def opn(p):
    with open(p, "rb") as f: data = f.read()
    out = []
    while data:
        if data[:2] == b"\x1f\x8b":
            d = zlib.decompressobj(31); out.append(d.decompress(data)); out.append(d.flush()); data = d.unused_data
        else: out.append(data); break
    return io.StringIO(b"".join(out).decode("utf-8", errors="replace"))


def tsec(m):
    t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return t + (float("0." + m.group(2)) if m.group(2) else 0.0)


def read_log(path):
    """the step lines (t, step, kind, n_seq, n_tok, segs) of an engine log (docker logs -t, gzip or plain)"""
    out = []
    if not path or not os.path.exists(path): return out
    for l in opn(path):
        m = TSRE.match(l)
        if not m or " step " not in l: continue
        s = STEP.search(l.rstrip("\n"))
        if not s: continue
        out.append((tsec(m), int(s.group(1)), s.group(2), int(s.group(3)), int(s.group(4)), [tuple(int(x) for x in g) for g in SEG.findall(s.group(5))]))
    return out


def rdj(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None


def jl(p):
    out = []
    try:
        for l in open(p):
            l = l.strip()
            if not l: continue
            try: out.append(json.loads(l))
            except ValueError: pass
    except OSError: pass
    return out


def is_nc(s): return s[2] != "decode" and (any(g[1] > QMIN for g in s[5]) or s[4] >= NCROWS)


def wall(steps, j):
    if j is None or j + 1 >= len(steps): return None
    d = steps[j + 1][0] - steps[j][0]
    return 1000.0 * d if 0.0 <= d <= IDLE else None


def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def nums(v): return [x for x in v if num(x)]
def p50(v): return round(statistics.median(v), 2) if v else None
def mean(v): return round(statistics.mean(v), 2) if v else None
def f(v, n=2): return ("%." + str(n) + "f") % v if num(v) else "-"
def pc(a, b): return 100.0 * (a / b - 1.0) if num(a) and num(b) and b else None
def fp(v): return ("%+.2f %%" % v) if num(v) else "-"
def fd(v, n=0): return ("%+." + str(n) + "f") % v if num(v) else "-"
def spread(v): return round(100.0 * (max(v) - min(v)) / statistics.mean(v), 2) if len(v) > 1 and statistics.mean(v) else None
def gib(v, u): return float(v) * {"GiB": 1.0, "MiB": 1.0 / 1024, "KiB": 1.0 / 1048576, "B": 1.0 / 1073741824}.get(u, 1.0)


# ---------------------------------------------------------------- the placement read-back (a start log)
PL = dict(
    hp=re.compile(r"--host-pool-mib\s+(\d+)\s+\["),
    mml=re.compile(r"--max-model-len\s+(\d+)\s+\["),
    seqs=re.compile(r"--max-num-seqs\s+(\d+)\s+\["),
    experts=re.compile(r"^ {4}experts\s+([\d.]+) (GiB|MiB)\s+(\d+)% of it"),
    kv=re.compile(r"^ {4}kv cache\s+([\d.]+) (GiB|MiB)\s+(\d+)% of it"),
    stage=re.compile(r"^ {4}prefill staging\s+([\d.]+) (GiB|MiB)"),
    elastic=re.compile(r"^ {2}elastic\s+([\d.]+) (GiB|MiB)"),
    vw=re.compile(r"^ {2}vram_weights\s+([\d.]+) (GiB|MiB)"),
    vk=re.compile(r"^ {2}vram_kv\s+([\d.]+) (GiB|MiB)"),
    slab=re.compile(r"^I\[(\d)\] mover: (\d+) slab slots \((\d+) shadow, (\d+) more kept free out of a loan\), (\d+) pinned pool slots"),
    kvl=re.compile(r"^I(?:\[(\d)\])? mover: (\d+) slots can be lent out of the KV cache \(([\d.]+) (GiB|MiB)\)"),
    arl=re.compile(r"^I(?:\[(\d)\])? mover: (\d+) slots can be lent out of the activation arena \(([\d.]+) (GiB|MiB)\)"),
    dl=re.compile(r"^I(?:\[(\d)\])? mover: (\d+) slots can be lent out of the drafter's expert slab"),
    grow=re.compile(r"kv: the cache can grow to ([\d.]+) (GiB|MiB) a card -- (\d+) tokens of context, ([\d.]+) sessions at (\d+)"),
    backs=re.compile(r"kv: the pool backs (\d+)% of the worst case \((\d+) sequences all at (\d+) tokens\), ([\d.]+) full-length sessions"),
    hpool=re.compile(r"^host_pool\s+([\d.]+) (GiB|MiB) budget"),
    novmm=re.compile(r"no virtual memory management"),
)


def place_parse(path):
    d = dict(slab={}, kvlend=[], arlend=[], dlend=[], hpool=[], notes=[])
    if not path or not os.path.exists(path): return None
    for raw in opn(path):
        l = raw.rstrip("\n")
        m = TSRE.match(l)
        if m: l = l[m.end():]
        if l.startswith("radiance: "): l = l[10:]
        for k in ("hp", "mml", "seqs"):
            if k not in d:
                mm = PL[k].search(l)
                if mm: d[k] = int(mm.group(1))
        for k in ("experts", "kv", "stage", "elastic", "vw", "vk"):
            if k + "_gib" not in d:
                mm = PL[k].match(l)
                if mm:
                    d[k + "_gib"] = round(gib(mm.group(1), mm.group(2)), 3)
                    if k in ("experts", "kv"): d[k + "_pct"] = int(mm.group(3))
        mm = PL["slab"].match(l)
        if mm and int(mm.group(1)) not in d["slab"]: d["slab"][int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)), int(mm.group(5)))
        for k, key in (("kvl", "kvlend"), ("arl", "arlend"), ("dl", "dlend")):
            mm = PL[k].match(l)
            if mm and len(d[key]) < 4: d[key].append(int(mm.group(2)))
        mm = PL["grow"].search(l)
        if mm and "tokens" not in d:
            d.update(grow_gib=round(gib(mm.group(1), mm.group(2)), 3), tokens=int(mm.group(3)), sessions=float(mm.group(4)), sess_len=int(mm.group(5)))
        mm = PL["backs"].search(l)
        if mm and "backs_pct" not in d:
            d.update(backs_pct=int(mm.group(1)), wc_seqs=int(mm.group(2)), wc_len=int(mm.group(3)), fl_sessions=float(mm.group(4)))
        mm = PL["hpool"].match(l)
        if mm and len(d["hpool"]) < 4: d["hpool"].append(round(gib(mm.group(1), mm.group(2)), 2))
        if PL["novmm"].search(l) and not d["notes"]: d["notes"].append(l.strip()[:200])
    d["owned"] = [d["slab"][r][0] - d["slab"][r][1] for r in sorted(d["slab"])]
    d["pinned"] = [d["slab"][r][2] for r in sorted(d["slab"])]
    d["slab"] = {str(k): v for k, v in d["slab"].items()}
    if num(d.get("tokens")): d["cap"] = {lab: d["tokens"] // L for lab, L in SESS}
    return d


def capacity(d):
    """owned + KV-lendable + arena-lendable slots a rank (the trunk's VRAM home at small steps; the drafter's slab apart)"""
    o, k, a = d.get("owned") or [], d.get("kvlend") or [], d.get("arlend") or []
    if not o or not k or not a: return []
    k = k + [k[-1]] * (len(o) - len(k)); a = a + [a[-1]] * (len(o) - len(a))
    return [o[i] + k[i] + a[i] for i in range(len(o))]


def stats_place(st):
    """/stats right after the pin: kv tokens total, per rank vram / trunk / lent held"""
    if not st: return None
    e = st.get("experts") or {}
    out = dict(kv_tokens_total=st.get("kv_tokens_total"), kv_bytes_carved=st.get("kv_bytes_carved"), ranks=[])
    for r in e.get("ranks") or []:
        lent = r.get("lent") or {}
        out["ranks"].append(dict(rank=r.get("rank"), vram=r.get("vram"), trunk=r.get("trunk_vram"), pinned=r.get("pinned"),
                                 kv=[(lent.get("kv") or {}).get("covered"), (lent.get("kv") or {}).get("held")],
                                 arena=[(lent.get("arena") or {}).get("covered"), (lent.get("arena") or {}).get("held")],
                                 dslab=[(lent.get("dslab") or {}).get("covered"), (lent.get("dslab") or {}).get("held")]))
    return out


def ls(v): return "/".join(str(x) for x in v) if v else "-"


def dlist(a, b):
    """element deltas of two per-rank lists (b padded with its last), as '+x' or '+a/+b/..'"""
    if not a or not b: return "-"
    b = list(b) + [b[-1]] * (len(a) - len(b))
    d = [a[i] - b[i] for i in range(len(a))]
    return fd(d[0]) if len(set(d)) == 1 else "/".join(fd(x) for x in d)


def place_line(arm, hp, d, ref, st, sw):
    bad, info = [], []
    if d is None: return dict(ok=False, bad=["no start log"]), "KVPLACE %s: VIRHE no start log" % arm
    if d.get("hp") != hp: (bad if sw else info).append("--host-pool-mib in force %s, the arm asks %s (the knob did not take)" % (d.get("hp"), hp))
    if d.get("mml") != 524288 or d.get("seqs") != 16: (bad if sw else info).append("--max-model-len %s / --max-num-seqs %s (want 524288 / 16: only the host pool moves)" % (d.get("mml"), d.get("seqs")))
    for k in ("experts_gib", "kv_gib", "tokens"):
        if not num(d.get(k)): bad.append("no %s line in the start log" % k)
    if len(d.get("owned") or []) != 4 or len(d.get("kvlend") or []) != 4: bad.append("%d slab / %d KV-lend lines (want 4 each)" % (len(d.get("owned") or []), len(d.get("kvlend") or [])))
    R = ref or REF_PLACE
    if num(hp) and hp < (R.get("hp") or REFHP) and d.get("owned") and d.get("kvlend") and R.get("owned") and R.get("kvlend"):
        if not (min(d["owned"]) > max(R["owned"]) and max(d["kvlend"]) < min(R["kvlend"])):
            (bad if sw else info).append("the split did not move: owned %s vs %s, KV-lendable %s vs %s" % (ls(d["owned"]), ls(R["owned"]), ls(d["kvlend"]), ls(R["kvlend"])))
    cap, rcap = capacity(d), capacity(R)
    caps = d.get("cap") or {}; rcaps = R.get("cap") or {lab: R["tokens"] // L for lab, L in SESS} if num(R.get("tokens")) else {}
    sp = stats_place(st)
    stx = "-"
    if sp and sp["ranks"]:
        stx = "kv_tokens_total %s | %s" % (sp.get("kv_tokens_total"), "; ".join("r%s vram %s trunk %s pinned %s, lent kv %s/%s arena %s/%s dslab %s/%s (covered/held)" % (
            r["rank"], r["vram"], r["trunk"], r["pinned"], r["kv"][0], r["kv"][1], r["arena"][0], r["arena"][1], r["dslab"][0], r["dslab"][1]) for r in sp["ranks"]))
    txt = ("KVPLACE %s (--host-pool-mib %s in force, the arm's %s; MML %s, MAXSEQS %s; reference %s, --host-pool-mib %s): VRAM split a card experts %s GiB (ref %s, %s) / "
           "kv cache %s GiB (ref %s, %s) / prefill staging %s GiB (ref %s) | owned slab slots a rank (slab - shadow) %s (ref %s: %s) | KV-lendable %s (ref %s: %s) | arena-lendable %s "
           "(ref %s: %s) | drafter-lendable %s | trunk-capable VRAM slots a rank (owned + KV + arena lendable) %s (ref %s: %s) | KV pool %s tokens a card (ref %s: %s), %s sessions at 524288 "
           "(ref %s), backs %s %% of %s x %s (%s full-length), sessions %s (ref %s) | pinned pool slots %s, host_pool %s GiB | /stats after the pin: %s%s" % (
               arm, d.get("hp"), hp, d.get("mml"), d.get("seqs"), R.get("src", "zzzc16 start"), R.get("hp"), f(d.get("experts_gib")), f(R.get("experts_gib")),
               fd((d.get("experts_gib") or 0) - (R.get("experts_gib") or 0), 2) + " GiB" if num(d.get("experts_gib")) and num(R.get("experts_gib")) else "-",
               f(d.get("kv_gib")), f(R.get("kv_gib")), fp(pc(d.get("kv_gib"), R.get("kv_gib"))), f(d.get("stage_gib")), f(R.get("stage_gib")),
               ls(d.get("owned")), ls(R.get("owned")), dlist(d.get("owned"), R.get("owned")), ls(d.get("kvlend")), ls(R.get("kvlend")), dlist(d.get("kvlend"), R.get("kvlend")),
               ls(d.get("arlend")), ls(R.get("arlend")), dlist(d.get("arlend"), R.get("arlend")), ls(d.get("dlend")), ls(cap), ls(rcap), dlist(cap, rcap),
               d.get("tokens"), R.get("tokens"), fp(pc(d.get("tokens"), R.get("tokens"))), d.get("sessions"), R.get("sessions"), d.get("backs_pct"), d.get("wc_seqs"), d.get("wc_len"),
               d.get("fl_sessions"), " / ".join("%s x %s" % (caps.get(lab), lab) for lab, _ in SESS[:3]), " / ".join("%s x %s" % (rcaps.get(lab), lab) for lab, _ in SESS[:3]),
               ls(d.get("pinned")), ls(d.get("hpool")), stx,
               (" -- VIRHE " + "; ".join(bad)) if bad else ((" | INFO (mock, sw 0) " + "; ".join(info)) if info else " -- OK")))
    return dict(ok=not bad, bad=bad, info=info, cap=cap, ref_cap=rcap, d_owned=(mean(d["owned"]) - mean(R["owned"])) if d.get("owned") and R.get("owned") else None,
                d_kvlend=(mean(d["kvlend"]) - mean(R["kvlend"])) if d.get("kvlend") and R.get("kvlend") else None), txt


def place_main(argv):
    path, arm, hp, stp, ref, rhp, sw, jout = None, "?", None, None, None, REFHP, 1, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--arm", "--hp", "--stats", "--ref", "--ref-hp", "--sw", "--json"):
            v = argv[i + 1]; i += 2
            if a == "--arm": arm = v
            elif a == "--hp": hp = int(v)
            elif a == "--stats": stp = v
            elif a == "--ref": ref = v
            elif a == "--ref-hp": rhp = int(v)
            elif a == "--sw": sw = int(v)
            elif a == "--json": jout = v
        else: path = a; i += 1
    d = place_parse(path)
    R = place_parse(ref) if ref else None
    if R and R.get("owned"): R["src"] = "zzzc16 start log %s" % os.path.basename(ref)
    else: R = dict(REF_PLACE); R["cap"] = {lab: R["tokens"] // L for lab, L in SESS}
    R.setdefault("hp", rhp)
    st = rdj(stp) if stp else None
    res, txt = place_line(arm, hp, d, R, st, sw)
    print(txt)
    if jout:
        with open(jout, "w") as fo: json.dump(dict(arm=arm, hp=hp, place=d, ref=R, stats=stats_place(st), res=res, line=txt), fo, default=str)
    return 0


# ---------------------------------------------------------------- the window residency (samp.jsonl / st-w0 / st-w1)
def rmiss(r, k):
    v = r.get(k)
    if v is None: v = (r.get("slots") or {}).get(k)
    return v or {}


def residency(samp, st0, st1, w):
    """mean VRAM units a rank over the in-window samples; misses / hits by slot class; stream MB a decode step; lent held at the window's end"""
    S = [x for x in samp if num(x.get("t")) and (not w or w[0] <= x["t"] <= w[1])]
    out = dict(n_samp=len(S))
    if S:
        nr = max(len((x.get("s") or {}).get("experts", {}).get("ranks") or []) for x in S)
        out["vram"] = [mean(nums(((x["s"].get("experts") or {}).get("ranks") or [{}] * nr)[i].get("vram") for x in S if len((x["s"].get("experts") or {}).get("ranks") or []) > i)) for i in range(nr)]
        a, b = S[0]["s"], S[-1]["s"]
        ds = (b.get("steps_decode") or 0) - (a.get("steps_decode") or 0); dall = (b.get("steps") or 0) - (a.get("steps") or 0)
        out["dec_steps"] = ds
        ra, rb = (a.get("experts") or {}).get("ranks") or [], (b.get("experts") or {}).get("ranks") or []
        cls = {}
        mis, rou, sb = [], [], []
        for i in range(min(len(ra), len(rb))):
            m0, m1, h0, h1 = rmiss(ra[i], "miss"), rmiss(rb[i], "miss"), rmiss(ra[i], "hit"), rmiss(rb[i], "hit")
            for k in set(m1) | set(h1):
                c = cls.setdefault(k, [0, 0]); c[0] += (h1.get(k) or 0) - (h0.get(k) or 0); c[1] += (m1.get(k) or 0) - (m0.get(k) or 0)
            mis.append(((rb[i].get("routed") or 0) - (rb[i].get("routed_res") or 0)) - ((ra[i].get("routed") or 0) - (ra[i].get("routed_res") or 0)))
            rou.append((rb[i].get("routed") or 0) - (ra[i].get("routed") or 0))
            sb.append((rb[i].get("stream_bytes") or 0) - (ra[i].get("stream_bytes") or 0))
        out["cls"] = cls
        out["miss_step"] = round(statistics.mean(mis) / dall, 2) if mis and dall else None
        out["hit_pct"] = round(100.0 * (1.0 - sum(mis) / sum(rou)), 2) if rou and sum(rou) else None
        out["stream_mb_step"] = round(sum(sb) / 1e6 / dall, 1) if sb and dall else None
    held = []
    for s in (st0, st1):
        for r in ((s or {}).get("experts") or {}).get("ranks") or []:
            lent = r.get("lent") or {}
            hk, ha, hd = ((lent.get("kv") or {}).get("held") or 0), ((lent.get("arena") or {}).get("held") or 0), ((lent.get("dslab") or {}).get("held") or 0)
            hw = (lent.get("dw") or {}).get("held") or 0
            held.append((r.get("rank"), r.get("trunk_vram"), hk, ha, hd, hw))
    if held:
        rr = sorted({h[0] for h in held if h[0] is not None})
        g = lambda i, j: mean(nums(h[j] for h in held if h[0] == i))
        out["held"] = {str(i): dict(trunk=g(i, 1), kv=g(i, 2), arena=g(i, 3), dslab=g(i, 4), dw=g(i, 5), owned=(g(i, 1) - g(i, 2) - g(i, 3) - g(i, 4) - g(i, 5)) if num(g(i, 1)) else None) for i in rr}
    return out


def rmean(v): return mean(nums(v or []))


def heldstr(r):
    h = r.get("held") or {}
    if not h: return "lent held n/a"
    k = sorted(h)
    g = lambda x: mean(nums(h[i].get(x) for i in k))
    return "w0/w1 mean a rank: trunk %s = owned %s + KV-lent %s + arena-lent %s + drafter-slab-lent %s" % (f(g("trunk"), 1), f(g("owned"), 1), f(g("kv"), 1), f(g("arena"), 1), f(g("dslab"), 1))


def clsstr(r):
    c = r.get("cls") or {}
    if not c: return "slot classes n/a"
    tot = sum(v[0] + v[1] for v in c.values()) or 1
    return "hits/misses by slot class (all ranks) " + ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in sorted(c.items(), key=lambda kv: -(kv[1][0] + kv[1][1])) if v[0] or v[1])


# ---------------------------------------------------------------- the gens blocks (c8 / c16) of one start
def blocks(out, steps, n, names, samp_from=None):
    ts = [s[0] for s in steps]
    R = {r["name"]: r for r in jl(os.path.join(out, "mb.jsonl")) if r.get("n") == n and r.get("sched") == "G"}
    res = []
    for nm in names:
        r = R.get(nm)
        if not r: continue
        mk = r.get("marks") or []
        if len(mk) < 2: mk = sorted(x.get("t") for x in jl(os.path.join(out, "c8", nm + "-reqs.jsonl")) if x.get("ev") == "mark")[:2]
        b = dict(name=nm, agg=r.get("agg_dec_tps"), window_s=r.get("window_s"), requests=r.get("requests"), errors=r.get("errors"), decode_share=r.get("decode_share"),
                 ttft_p50=r.get("ttft_p50"), acc=r.get("acceptance"), tok=r.get("tok_per_decode_step"), bad=[])
        if len(mk) >= 2 and num(mk[0]) and num(mk[1]):
            w0, w1 = mk[0], mk[1]; b["win"] = [w0, w1]
            i0, i1 = bisect.bisect_left(ts, w0), bisect.bisect_right(ts, w1)
            dec = [j for j in range(i0, i1) if steps[j][2] == "decode"]
            wl = [(j, wall(steps, j)) for j in dec]; wl = [(j, x) for j, x in wl if x is not None]
            allw = [x for _, x in wl]
            rc = {}
            for j, x in wl:
                if steps[j][3] == n: rc[steps[j][4]] = rc.get(steps[j][4], 0) + 1
            nc = [j for j in range(i0, i1) if is_nc(steps[j])]
            ncw = nums(wall(steps, j) for j in nc)
            b.update(n_dec=len(allw), dec_p50=p50(allw), dec_mean=mean(allw), rowc=rc, n_nc=len(nc), nc_sum=round(sum(ncw), 1) if ncw else None,
                     nc_mean=round(statistics.mean(ncw), 1) if ncw else None)
            if not allw: b["bad"].append("no decode step in the window (engine log truncated or not captured?)")
        else: b["win"] = None; b["bad"].append("no window marks")
        if (b.get("errors") or 0) != 0 or not (b.get("requests") or 0) > 0: b["bad"].append("request errors %s of %s requests" % (b.get("errors"), b.get("requests")))
        sd = samp_from or (lambda x: (jl(os.path.join(out, "c8", x + "-samp.jsonl")), rdj(os.path.join(out, "c8", x + "-st-w0.json")), rdj(os.path.join(out, "c8", x + "-st-w1.json"))))
        sm, s0, s1 = sd(nm)
        b["res"] = residency(sm or [], s0, s1, b.get("win"))
        res.append(b)
    pool = {}
    for b in res:
        for k, v in (b.get("rowc") or {}).items(): pool[k] = pool.get(k, 0) + v
    rows = max(pool, key=lambda k: pool[k]) if pool else None
    for b in res:
        if b.get("win") and rows is not None:
            i0, i1 = bisect.bisect_left(ts, b["win"][0]), bisect.bisect_right(ts, b["win"][1])
            w = nums(wall(steps, j) for j in range(i0, i1) if steps[j][2] == "decode" and steps[j][3] == n and steps[j][4] == rows)
            b["bkt_rows"], b["bkt_p50"], b["bkt_n"] = rows, p50(w), len(w)
    return res


def armsum(bl):
    ag, bk, dp = nums(b.get("agg") for b in bl), nums(b.get("bkt_p50") for b in bl), nums(b.get("dec_p50") for b in bl)
    un = nums(rmean(b["res"].get("vram")) for b in bl if b.get("res"))
    ms = nums(b["res"].get("miss_step") for b in bl if b.get("res")); hp = nums(b["res"].get("hit_pct") for b in bl if b.get("res"))
    sm = nums(b["res"].get("stream_mb_step") for b in bl if b.get("res"))
    return dict(n=len(bl), aggs=ag, agg=mean(ag), agg_spread=spread(ag), bkts=bk, bkt=mean(bk), bkt_spread=spread(bk), decs=dp, dec=mean(dp), units=mean(un), unit_blocks=un,
                miss_step=mean(ms), hit_pct=mean(hp), stream_mb_step=mean(sm), bkt_rows=bl[0].get("bkt_rows") if bl else None)


def ref_blocks(zf, n, names):
    """zzzfuse's F blocks: agg / step-log values are the run's own (constants: its engine log is 1 GB-class, not re-read); the window residency
    from its raw-snaps.tar.gz (samp + st-w0 / st-w1) and the window marks from mb.jsonl"""
    out = dict(res=None, src=None)
    if not zf: return out
    tp = os.path.join(zf, "raw-snaps.tar.gz")
    mbr = {r["name"]: r for r in jl(os.path.join(zf, "mb.jsonl")) if r.get("n") == n}
    if not os.path.exists(tp) or not mbr: return out
    rs = []
    try:
        with tarfile.open(tp, "r:gz") as t:
            mem = {m.name: m for m in t.getmembers()}
            def rd(nm):
                m = mem.get(nm)
                if not m: return None
                return t.extractfile(m).read().decode("utf-8", errors="replace")
            for nm in names:
                r = mbr.get(nm)
                if not r or len(r.get("marks") or []) < 2: continue
                sm = [json.loads(l) for l in (rd("c8/%s-samp.jsonl" % nm) or "").splitlines() if l.strip()]
                s0 = json.loads(rd("c8/%s-st-w0.json" % nm) or "null"); s1 = json.loads(rd("c8/%s-st-w1.json" % nm) or "null")
                rs.append(residency(sm, s0, s1, r["marks"][:2]))
    except Exception as e: return dict(res=None, src="raw-snaps.tar.gz unreadable (%s)" % type(e).__name__)
    if not rs: return out
    return dict(res=rs, src="zzzfuse raw-snaps.tar.gz %s" % "+".join(names), units=mean(nums(rmean(r.get("vram")) for r in rs)),
                miss_step=mean(nums(r.get("miss_step") for r in rs)), hit_pct=mean(nums(r.get("hit_pct") for r in rs)),
                stream_mb_step=mean(nums(r.get("stream_mb_step") for r in rs)), vram=[mean(nums((r.get("vram") or [None] * 4)[i] for r in rs)) for i in range(4)])


def ref_1u(zc):
    R = [r for r in jl(os.path.join(zc, "blocks.jsonl")) if r.get("lever") == "U1" and r.get("arm") == "F"] if zc else []
    tps = nums((r.get("j") or {}).get("decode_tps") for r in R); ms = nums((r.get("j") or {}).get("ms_per_decode_step") for r in R)
    mb = nums((r.get("st") or {}).get("stream_mb_step") for r in R)
    if tps: return dict(src="zzzc16 blocks.jsonl F U1 (%d reads)" % len(tps), tps=mean(tps), ms=mean(ms), mb=mean(mb), n=len(tps))
    return dict(REF_1U, n=6)


# ---------------------------------------------------------------- the summary
def run(argv):
    out, arms, sw, zf, zc, c8g, c16g, u1n, emd5, jout = None, "K35:16800,K60:14900", 1, None, None, 2, 1, 3, "47a39dac581a", None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--arms", "--sw", "--ref-zf", "--ref-zc", "--c8gens", "--c16gens", "--u1n", "--essay-md5", "--json"):
            v = argv[i + 1]; i += 2
            if a == "--arms": arms = v
            elif a == "--sw": sw = int(v)
            elif a == "--ref-zf": zf = v
            elif a == "--ref-zc": zc = v
            elif a == "--c8gens": c8g = int(v)
            elif a == "--c16gens": c16g = int(v)
            elif a == "--u1n": u1n = int(v)
            elif a == "--essay-md5": emd5 = v
            elif a == "--json": jout = v
        else: out = a; i += 1
    if not out:
        print("KVRES VIRHE usage: OUT --arms A:HP,.."); return 2
    AL = []
    for e in arms.replace(" ", ",").split(","):
        if e and ":" in e: AL.append((e.split(":")[0], int(e.split(":")[1])))
    nbad = 0
    idrows = []
    try: idrows = [l.strip() for l in open(os.path.join(out, "identity.txt")) if l.strip()]
    except OSError: pass
    # references
    r8, r16 = ref_blocks(zf, 8, REF_C8["blocks"]), ref_blocks(zf, 16, REF_C16["blocks"])
    u1r = ref_1u(zc)
    k0u8 = r8.get("units"); k0u16 = r16.get("units")
    rp = place_parse(os.path.join(zc, "start-xzc16-n5.log.gz")) if zc else None
    if not (rp and rp.get("owned")): rp = dict(REF_PLACE)
    print("KVRES REFERENCES (no K0 re-run, TESTING.md 2.): c8 %s agg %s = %s tok/s, full-row p50 %s = %s ms, window units a rank %s (%s) | c16 %s agg %s tok/s, full-row p50 %s ms, "
          "window units %s | 1u %s %s tok/s, %s ms a step, %s MB a step | placement %s: owned %s, KV-lendable %s, KV %s tokens, %s sessions at 524288" % (
              REF_C8["src"], "/".join(f(v, 1) for v in REF_C8["agg"]), f(statistics.mean(REF_C8["agg"])), "/".join(f(v) for v in REF_C8["bkt"]), f(statistics.mean(REF_C8["bkt"])),
              f(k0u8, 1), r8.get("src") or "n/a", REF_C16["src"], f(REF_C16["agg"][0], 1), f(REF_C16["bkt"][0]), f(k0u16, 1), u1r["src"], f(u1r["tps"]), f(u1r["ms"]), f(u1r["mb"], 1),
              "zzzc16 start log" if rp.get("slab") else REF_PLACE["src"], ls(rp.get("owned")), ls(rp.get("kvlend")), rp.get("tokens"), rp.get("sessions")))
    R8 = dict(agg=statistics.mean(REF_C8["agg"]), bkt=statistics.mean(REF_C8["bkt"]), units=k0u8)
    R16 = dict(agg=REF_C16["agg"][0], bkt=REF_C16["bkt"][0], units=k0u16)
    res = dict(arms={}, ref=dict(c8=R8, c16=R16, u1=u1r))
    table = []
    for a, hp in AL:
        A = dict(hp=hp)
        P = rdj(os.path.join(out, "place-%s.json" % a))
        if P and P.get("place"):
            print("KVRES PLACE " + (P.get("line") or place_line(a, hp, P["place"], P.get("ref") or rp, P.get("stats_raw"), sw)[1])[8:])
            A["place"] = P["place"]; A["pres"] = P.get("res") or {}
            if not (P.get("res") or {}).get("ok", True) and sw: nbad += 1
        else:
            print("KVRES PLACE %s: VIRHE no place-%s.json (the start did not come up or kvplace did not run)" % (a, a)); nbad += 1
        # identity
        keys = ("xkvr-%s warm-up" % a, "xkvr-%s-a1-F-vs-ref12-a1" % a, "xkvr-%s-a1-F-vs-zzzc16-a1-F" % a)
        got = {k: next((r[len(k):].strip() for r in reversed(idrows) if r.startswith(k + " ") or r.startswith(k + ":")), None) for k in keys}
        idb = [k for k in keys if not (got[k] or "").lstrip(": ").startswith("IDENTICAL")]
        A["id_ok"] = not idb
        print("KVRES IDENTITY %s (strict, T 0 greedy, THE int16 PIN; placement only, so IDENTICAL is required): %s%s" % (
            a, " | ".join("%s %s" % (k.replace("xkvr-%s" % a, "").strip(" -") or "warm-up", (got[k] or "NOT RUN")[:80]) for k in keys),
            (" -- VIRHE " + ", ".join(idb) + " (that start's speed reads were STOPPED)") if idb and sw else ((" | INFO (mock) " + ", ".join(idb)) if idb else " -> 3 of 3 IDENTICAL")))
        if idb and sw: nbad += len(idb)
        steps = read_log(os.path.join(out, "xkvr-%s-engine-full.log.gz" % a))
        names8 = sorted({r["name"] for r in jl(os.path.join(out, "mb.jsonl")) if r.get("n") == 8 and str(r.get("name", "")).startswith("c8-%sx" % a)})
        names16 = sorted({r["name"] for r in jl(os.path.join(out, "mb.jsonl")) if r.get("n") == 16 and str(r.get("name", "")).startswith("c16-%sx" % a)})
        for n, names, RR, lab in ((8, names8, R8, "c8"), (16, names16, R16, "c16")):
            bl = blocks(out, steps, n, names)
            for b in bl:
                r = b.get("res") or {}
                print("KVRES %s BLOCK %s %s (gens %d, THE int16 PIN): agg %s tok/s, window %s s, %s requests, %s errors, decode_share %s, TTFT p50 %s s, acceptance %s | step log: %s decode steps, "
                      "p50 %s / mean %s ms | full-row bucket (%d seqs, %s rows): %s steps p50 %s ms | newcomer steps in the window %s, walls %s ms (mean %s) | window residency (%s samples): "
                      "VRAM units a rank %s (mean %s), misses %s a rank a step, hit %s %%, stream %s MB a step (all ranks) | %s | %s%s" % (
                          lab, a, b["name"], c8g if n == 8 else c16g, f(b.get("agg"), 1), b.get("window_s"), b.get("requests"), b.get("errors"), b.get("decode_share"), b.get("ttft_p50"),
                          f(b.get("acc"), 3), b.get("n_dec"), f(b.get("dec_p50")), f(b.get("dec_mean")), n, b.get("bkt_rows", "-"), b.get("bkt_n", "-"), f(b.get("bkt_p50")), b.get("n_nc"),
                          f(b.get("nc_sum"), 1), f(b.get("nc_mean"), 1), r.get("n_samp"), "/".join(f(v, 0) for v in (r.get("vram") or [])), f(rmean(r.get("vram")), 1),
                          f(r.get("miss_step")), f(r.get("hit_pct")), f(r.get("stream_mb_step"), 1), heldstr(r), clsstr(r),
                          (" -- VIRHE " + "; ".join(b["bad"])) if b["bad"] else ""))
                nbad += len(b["bad"])
            S = armsum(bl)
            A[lab] = S
            if not bl:
                if A.get("id_ok", True) or not sw: print("KVRES %s %s: not run (the start did not come up, the budget, or a STOP)" % (lab, a))
                continue
            du = (S["units"] - RR["units"]) if num(S.get("units")) and num(RR.get("units")) else None
            print("KVRES %s %s (%d block(s), gens): agg %s tok/s (blocks %s) vs K0 %s %s (%s) | full-row (%d seqs, %s rows) p50 %s ms (blocks %s) vs K0 %s (%s) | decode p50 %s ms | "
                  "window VRAM units a rank %s vs K0 %s (%s) | misses %s a rank a step (K0 %s), hit %s %% (K0 %s), stream %s MB a step (K0 %s)" % (
                      lab, a, S["n"], f(S["agg"]), "/".join(f(v, 1) for v in S["aggs"]), RR is R8 and REF_C8["src"] or REF_C16["src"], f(RR["agg"]), fp(pc(S["agg"], RR["agg"])),
                      n, S.get("bkt_rows"), f(S["bkt"]), "/".join(f(v) for v in S["bkts"]), f(RR["bkt"]), fp(pc(S["bkt"], RR["bkt"])), f(S["dec"]), f(S["units"], 1), f(RR["units"], 1),
                      fd(du, 1), f(S["miss_step"]), f((r8 if n == 8 else r16).get("miss_step")), f(S["hit_pct"]), f((r8 if n == 8 else r16).get("hit_pct")),
                      f(S["stream_mb_step"], 1), f((r8 if n == 8 else r16).get("stream_mb_step"), 1))
                  + ("" if S.get("bkt_rows") == (REF_C8 if n == 8 else REF_C16)["rows"] else
                     " | NOTE the full-row bucket is %s rows, K0's %d (the drafting state differs): the full-row ms is NOT like for like (agg and decode p50 still read)" % (
                         S.get("bkt_rows"), (REF_C8 if n == 8 else REF_C16)["rows"])))
            A[lab]["du"] = du
        # 1u
        U = [r for r in jl(os.path.join(out, "blocks.jsonl")) if r.get("set") == "%s-U1" % a]
        W = [r for r in U if r.get("lever") == "U1W"]; U = [r for r in U if r.get("lever") == "U1"]
        g = lambda r, x: (r.get("j") or {}).get(x)
        tps, ms, mb = nums(g(r, "decode_tps") for r in U), nums(g(r, "ms_per_decode_step") for r in U), nums((r.get("st") or {}).get("stream_mb_step") for r in U)
        md5 = [g(r, "text_md5") for r in W + U]
        if U:
            ub = [] if all(m == emd5 for m in md5) else ["answer md5 %s (want %s)" % (md5, emd5)]
            if len(U) != u1n: ub.append("%d reads (want %d)" % (len(U), u1n))
            print("KVRES 1u %s (resprobe essay 512, T 0, THE int16 PIN, one discarded + %d back to back): reads %s tok/s, mean %s vs K0 %s %s (%s) | ms a step %s (mean %s) vs K0 %s (%s) | "
                  "stream MB a step %s (mean %s; K0 %s) | discarded read %s tok/s at %s MB a step (residency switch after c16) | answers %s%s" % (
                      a, u1n, "/".join(f(v) for v in tps), f(mean(tps)), u1r["src"], f(u1r["tps"]), fp(pc(mean(tps), u1r["tps"])), "/".join(f(v) for v in ms), f(mean(ms)), f(u1r["ms"]),
                      fp(pc(mean(ms), u1r["ms"])), "/".join(f(v, 1) for v in mb), f(mean(mb), 1), f(u1r["mb"], 1), f(g(W[0], "decode_tps")) if W else "-",
                      f((W[0].get("st") or {}).get("stream_mb_step"), 1) if W else "-", "IDENTICAL to ref14's essay answer (%s) on all %d" % (emd5, len(md5)) if not ub or not md5 or all(m == emd5 for m in md5) else "md5 %s" % md5,
                      (" -- VIRHE " + "; ".join(ub)) if ub and sw else ((" | INFO (mock) " + "; ".join(ub)) if ub else "")))
            if ub and sw: nbad += len(ub)
            A["u1"] = dict(tps=mean(tps), ms=mean(ms), mb=mean(mb), n=len(tps))
        elif A.get("id_ok", True) or not sw: print("KVRES 1u %s: not run (the start did not come up, the budget, or a STOP)" % a)
        # the arm's line
        pl = A.get("place") or {}
        rcap = rp.get("cap") or ({lab: rp["tokens"] // L for lab, L in SESS} if num(rp.get("tokens")) else {})
        cap = pl.get("cap") or {}
        c8, c16, u1 = A.get("c8") or {}, A.get("c16") or {}, A.get("u1") or {}
        dow = (mean(pl["owned"]) - mean(rp["owned"])) if pl.get("owned") and rp.get("owned") else None
        print("KVRES %s (--host-pool-mib %s, KV pool %s): units a rank -- owned slab %s (K0 %s, %s), KV-lendable %s (K0 %s, %s), resident in the c8 window %s (K0 %s, %s) | c8 %s tok/s (%s vs "
              "int16 F %s), full-row step %s ms (%s vs %s) | c16 %s tok/s (%s vs %s), full-row %s ms (%s vs %s) | 1u %s tok/s (%s vs %s) | KV capacity a card %s tokens (K0 %s, %s): "
              "max context %s (one session; MML 524288 unchanged), %s sessions at 524288 (K0 %s), %s (K0 %s)" % (
                  a, hp, fp(pc(pl.get("kv_gib"), rp.get("kv_gib"))), f(mean(pl.get("owned") or []) if pl.get("owned") else None, 0), f(mean(rp.get("owned") or []) if rp.get("owned") else None, 0), fd(dow),
                  f(mean(pl.get("kvlend") or []) if pl.get("kvlend") else None, 0), f(mean(rp.get("kvlend") or []) if rp.get("kvlend") else None, 0),
                  fd((mean(pl["kvlend"]) - mean(rp["kvlend"])) if pl.get("kvlend") and rp.get("kvlend") else None), f(c8.get("units"), 1), f(R8["units"], 1), fd(c8.get("du"), 1),
                  f(c8.get("agg")), fp(pc(c8.get("agg"), R8["agg"])), f(R8["agg"]), f(c8.get("bkt")), fp(pc(c8.get("bkt"), R8["bkt"])), f(R8["bkt"]),
                  f(c16.get("agg")), fp(pc(c16.get("agg"), R16["agg"])), f(R16["agg"]), f(c16.get("bkt")), fp(pc(c16.get("bkt"), R16["bkt"])), f(R16["bkt"]),
                  f(u1.get("tps")), fp(pc(u1.get("tps"), u1r["tps"])), f(u1r["tps"]), pl.get("tokens"), rp.get("tokens"), fp(pc(pl.get("tokens"), rp.get("tokens"))),
                  "524288" if num(pl.get("tokens")) and pl["tokens"] >= 524288 else "-", pl.get("sessions"), rp.get("sessions"),
                  " / ".join("%s x %s" % (cap.get(lab), lab) for lab, _ in SESS[:3]), " / ".join("%s x %s" % (rcap.get(lab), lab) for lab, _ in SESS[:3])))
        A["d_owned"] = dow
        res["arms"][a] = A
        table.append((a, hp, pl, c8, c16, u1, dow))
    # TABLE
    print("KVRES TABLE | arm | host pool MiB | KV GiB | owned a rank (d) | KV-lendable (d) | c8 window units (d) | c8 tok/s (%) | c8 full-row ms (%) | c16 tok/s (%) | 1u tok/s (%) | "
          "KV tokens | sessions 524288 / 500k / 260k / 128k |")
    rcap = rp.get("cap") or ({lab: rp["tokens"] // L for lab, L in SESS} if num(rp.get("tokens")) else {})
    print("KVRES TABLE | K0 (ref) | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s / %s / %s |" % (
        REFHP, f(rp.get("kv_gib")), f(mean(rp.get("owned") or []) if rp.get("owned") else None, 0), f(mean(rp.get("kvlend") or []) if rp.get("kvlend") else None, 0), f(R8["units"], 0),
        f(R8["agg"]), f(R8["bkt"]), f(R16["agg"]), f(u1r["tps"]), rp.get("tokens"), rp.get("sessions"), rcap.get("500k"), rcap.get("260k"), rcap.get("128k")))
    for a, hp, pl, c8, c16, u1, dow in table:
        cap = pl.get("cap") or {}
        print("KVRES TABLE | %s | %s | %s | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s | %s / %s / %s / %s |" % (
            a, hp, f(pl.get("kv_gib")), f(mean(pl.get("owned") or []) if pl.get("owned") else None, 0), fd(dow),
            f(mean(pl.get("kvlend") or []) if pl.get("kvlend") else None, 0), fd((mean(pl["kvlend"]) - mean(rp["kvlend"])) if pl.get("kvlend") and rp.get("kvlend") else None),
            f(c8.get("units"), 0), fd(c8.get("du"), 0), f(c8.get("agg")), fp(pc(c8.get("agg"), R8["agg"])), f(c8.get("bkt")), fp(pc(c8.get("bkt"), R8["bkt"])),
            f(c16.get("agg")), fp(pc(c16.get("agg"), R16["agg"])), f(u1.get("tps")), fp(pc(u1.get("tps"), u1r["tps"])), pl.get("tokens"), pl.get("sessions"),
            cap.get("500k"), cap.get("260k"), cap.get("128k")))
    # SLOPE
    sl = []
    for a, hp, pl, c8, c16, u1, dow in table:
        dms = (c8.get("bkt") - R8["bkt"]) if num(c8.get("bkt")) else None
        du = c8.get("du")
        s1 = (100.0 * dms / du) if num(dms) and num(du) and abs(du) >= MINDU else None
        s2 = (100.0 * dms / dow) if num(dms) and num(dow) and abs(dow) >= MINDU else None
        sl.append("%s: c8 full-row %s ms (%s) at %s window units a rank -> %s ms per +100 resident units%s; per +100 OWNED units %s" % (
            a, fd(dms, 2), fp(pc(c8.get("bkt"), R8["bkt"])), fd(du, 1), f(s1) if num(s1) else "n/a",
            "" if num(s1) else " (the window units moved < %d a rank: no slope; the KV loan already held these bytes)" % MINDU, f(s2) if num(s2) else "n/a"))
    print("KVRES SLOPE (vs vramsweep 10.11.: %.1f ms a c8 decode step per +100 resident units a rank; cross-start spread ~1-1.5 %%): %s" % (SLOPE_REF, " | ".join(sl) or "no arm read"))
    print("KVRES SUMMARY: %d arm(s), %d guard finding(s)%s" % (len(table), nbad, " -- VIRHE see the lines above" if nbad else ""))
    if jout:
        with open(jout, "w") as fo: json.dump(res, fo, default=str)
    return 0


# ---------------------------------------------------------------- selftest
SAMPLE_START = """I   --max-num-seqs                     16                                          [command line]
I   --max-model-len                    524288                                      [command line]
I   --host-pool-mib                    %d                                       [command line]
I kv: the pool backs %d%% of the worst case (16 sequences all at 524288 tokens), %.1f full-length sessions; a larger --host-pool-mib gives the cache more
I VRAM budget, per card:
  elastic             25.31 GiB   the experts keep what the host pool cannot hold; the cache takes the rest
    experts           %.2f GiB   69%% of it
    kv cache           %.2f GiB   27%% of it -- 32%% backing of the 16 x 524288 worst case: 5.1 full-length
    prefill staging    %.2f GiB   out of the cache: two buffers of one layer's non-resident experts,
  vram_weights        20.00 GiB   static + experts
  vram_kv              %.2f GiB   the cache's share
host_pool       19.13 GiB budget,        0 B used,  19.13 GiB free
I[0] mover: %d slab slots (48 shadow, 0 more kept free out of a loan), %d pinned pool slots, 2.65 GiB static vram
I[1] mover: %d slab slots (48 shadow, 0 more kept free out of a loan), %d pinned pool slots, 2.51 GiB static vram
I[2] mover: %d slab slots (48 shadow, 0 more kept free out of a loan), %d pinned pool slots, 2.51 GiB static vram
I[3] mover: %d slab slots (48 shadow, 0 more kept free out of a loan), %d pinned pool slots, 2.51 GiB static vram
W[0] this device has no virtual memory management, so the KV cache holds its whole carve and the expert slab has nothing to grow into.
I mover: %d slots can be lent out of the KV cache (6.62 GiB), filled as it lends them
I mover: %d slots can be lent out of the KV cache (6.62 GiB), filled as it lends them
I mover: %d slots can be lent out of the KV cache (6.62 GiB), filled as it lends them
I mover: %d slots can be lent out of the KV cache (6.62 GiB), filled as it lends them
I kv: the cache can grow to %.2f GiB a card -- %d tokens of context, %.1f sessions at 524288 -- with every block it lends the experts taken back
I mover: %d slots can be lent out of the activation arena (3.40 GiB), filled as it lends them
I mover: %d slots can be lent out of the activation arena (3.39 GiB), filled as it lends them
I mover: %d slots can be lent out of the activation arena (3.39 GiB), filled as it lends them
I mover: %d slots can be lent out of the activation arena (3.39 GiB), filled as it lends them
I[0] mover: 221 slots can be lent out of the drafter's expert slab (1.96 GiB), filled as it lends them
"""


def mkstart(p, hp, exp, kv, stage, own, kvl, arl, tok, ses, gz=False, ts=False):
    txt = SAMPLE_START % (hp, 32, 5.1, exp, kv, stage, kv, own + 48, 2232, own + 64, 2216, own + 64, 2216, own + 64, 2216, kvl, kvl, kvl, kvl, kv, tok, ses, arl, arl - 1, arl - 1, arl - 1)
    if ts: txt = "".join("2026-11-13T05:00:00.000000000Z " + l + "\n" for l in txt.splitlines())
    if gz:
        with gzip.open(p, "wt") as fo: fo.write(txt)
    else:
        with open(p, "w") as fo: fo.write(txt)


def selftest():
    d = tempfile.mkdtemp(prefix="kvsum-")
    try: return _selftest(d)
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)


def _selftest(d):
    import contextlib
    ref = os.path.join(d, "ref.log.gz"); mkstart(ref, 19456, 17.34, 6.95, 1.02, 1801, 748, 384, 2590442, 4.9, gz=True)
    k35 = os.path.join(d, "start-xkvr-K35.log"); mkstart(k35, 16800, 19.93, 4.50, 0.88, 2094, 471, 368, 1638000, 3.1)
    bad = os.path.join(d, "start-bad.log"); mkstart(bad, 19456, 17.34, 6.95, 1.02, 1801, 748, 384, 2590442, 4.9, ts=True)
    P = place_parse(ref)
    assert P["hp"] == 19456 and P["owned"] == [1801, 1817, 1817, 1817] and P["kvlend"] == [748] * 4 and P["arlend"] == [384, 383, 383, 383], P
    assert P["tokens"] == 2590442 and P["cap"]["500k"] == 5 and P["cap"]["128k"] == 20 and P["kv_gib"] == 6.95 and P["stage_gib"] == 1.02 and P["notes"], P
    Q = place_parse(bad)
    assert Q["hp"] == 19456 and Q["owned"] == P["owned"], Q   # the docker logs -t form parses the same
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        place_main([k35, "--arm", "K35", "--hp", "16800", "--ref", ref, "--json", os.path.join(d, "place-K35.json")])
        place_main([bad, "--arm", "K60", "--hp", "14900", "--ref", ref, "--json", os.path.join(d, "place-K60.json")])
        place_main([bad, "--arm", "K60", "--hp", "14900", "--ref", ref, "--sw", "0"])
    o = buf.getvalue().splitlines()
    assert o[0].startswith("KVPLACE K35") and "-- OK" in o[0] and "+293" in o[0] and "-277" in o[0], o[0]
    assert "VIRHE" in o[1] and "did not take" in o[1] and "split did not move" in o[1], o[1]
    assert "INFO (mock" in o[2] and "VIRHE" not in o[2], o[2]
    # a summary over a synthetic start: 2 c8 blocks + 1 c16 + 1u
    out = os.path.join(d, "out"); os.makedirs(os.path.join(out, "c8"))
    J = rdj(os.path.join(d, "place-K35.json")); J["line"] = None
    with open(os.path.join(out, "place-K35.json"), "w") as fo: json.dump(J, fo)
    t0 = 1794534500.0
    lines = []
    t = t0
    for k in range(3000):
        kind, ns, nt = ("prefill", 1, 900) if k % 64 == 0 else ("decode", 8, 8)
        segs = " [0] slot=1 q=900 ctx=900" if kind == "prefill" else "".join(" [%d] slot=%d q=1 ctx=%d" % (s, s, 500 + k % 400) for s in range(8))
        lines.append("%sZ I step %d %s n_seq=%d n_tok=%d%s" % (datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"), k, kind, ns, nt, segs))
        t += 0.5 if kind == "prefill" else 0.0332
    with gzip.open(os.path.join(out, "xkvr-K35-engine-full.log.gz"), "wt") as fo: fo.write("\n".join(lines) + "\n")
    mb = []
    for nm, n, w0 in (("c8-K35x1-G-F", 8, t0 + 5), ("c8-K35x2-G-F", 8, t0 + 50), ("c16-K35x1-G-F", 16, t0 + 80)):
        mb.append(dict(name=nm, n=n, arm="F", sched="G", agg_dec_tps=192.5 if n == 8 else 238.0, marks=[w0, w0 + 30], window_s=30, requests=16, errors=0, decode_share=0.8, ttft_p50=0.5, acceptance=0.0))
        S = []
        for q in range(16):
            tt = w0 + 2 * q
            S.append(dict(t=tt, s=dict(steps=1000 + 60 * q, steps_decode=950 + 57 * q, experts=dict(ranks=[dict(rank=r, vram=2930 + r, routed=100000 + 7000 * q, routed_res=98000 + 6860 * q,
                     stream_bytes=1e9 + 4e7 * 60 * q, miss=dict(trunk=10 * q, kv=5 * q), hit=dict(trunk=500 * q, kv=200 * q)) for r in range(4)]))))
        with open(os.path.join(out, "c8", nm + "-samp.jsonl"), "w") as fo: fo.write("\n".join(json.dumps(x) for x in S) + "\n")
        for wn in ("w0", "w1"):
            with open(os.path.join(out, "c8", "%s-st-%s.json" % (nm, wn)), "w") as fo:
                json.dump(dict(experts=dict(ranks=[dict(rank=r, trunk_vram=2930, lent=dict(kv=dict(held=470), arena=dict(held=250), dslab=dict(held=200), dw=dict(held=0))) for r in range(4)])), fo)
    with open(os.path.join(out, "mb.jsonl"), "w") as fo: fo.write("\n".join(json.dumps(x) for x in mb) + "\n")
    with open(os.path.join(out, "blocks.jsonl"), "w") as fo:
        for lv, tp in (("U1W", 89.0), ("U1", 90.9), ("U1", 90.8), ("U1", 90.85)):
            fo.write(json.dumps(dict(lever=lv, arm="F", set="K35-U1", j=dict(decode_tps=tp, ms_per_decode_step=17.35, text_md5="47a39dac581a"), st=dict(stream_mb_step=1.0))) + "\n")
    with open(os.path.join(out, "identity.txt"), "w") as fo:
        fo.write("xkvr-K35 warm-up IDENTICAL: 3 answers\nxkvr-K35-a1-F-vs-ref12-a1 IDENTICAL: 20 of 20 answers\nxkvr-K35-a1-F-vs-zzzc16-a1-F IDENTICAL: 20 of 20 answers\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run([out, "--arms", "K35:16800", "--json", os.path.join(d, "s.json")])
    o = buf.getvalue()
    assert "KVRES c8 BLOCK K35 c8-K35x1-G-F" in o and "full-row bucket (8 seqs, 8 rows)" in o and "KVRES c16 K35" in o, o
    assert "KVRES 1u K35" in o and "90.85" in o and "KVRES TABLE | K35 | 16800" in o and "KVRES SLOPE" in o and "3 of 3 IDENTICAL" in o, o
    assert "Traceback" not in o and "KVRES SUMMARY: 1 arm(s), 0 guard finding(s)" in o, o
    s = rdj(os.path.join(d, "s.json"))
    assert abs(s["arms"]["K35"]["c8"]["bkt"] - 33.2) < 0.2 and s["arms"]["K35"]["c8"]["units"] == 2931.5, s["arms"]["K35"]["c8"]
    print("SELFTEST OK: place parse (plain / gz / docker logs -t), KVPLACE OK / VIRHE (knob not in force, split not moved) / mock INFO, the summary (c8 x2, c16, 1u, identity, TABLE, SLOPE)")
    return 0


def main(argv):
    if argv[:1] == ["selftest"]: return selftest()
    if argv[:1] == ["place"]: return place_main(argv[1:])
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
