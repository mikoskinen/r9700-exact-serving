# c4split.py LOG.gz BLOCKS.jsonl ARM : per block of ARM, the c4 stretch (decode steps with n_seq==4 only, between n_seq 1 and 8 phases):
# step classes by tokens per seq (q), wall ms = next step's timestamp - this one's (single-row-type stretches only)
import gzip, json, re, sys, statistics as st
from datetime import datetime
logf, blk, arm = sys.argv[1:4]
B = [json.loads(l) for l in open(blk)]
starts = sorted((b.get("applied_step") or 0, b["block"], b["arm"]) for b in B)
rx = re.compile(r"^(\S+)Z I step (\d+) (\w+) n_seq=(\d+) n_tok=(\d+)(.*)$")
S = []
for l in gzip.open(logf, "rt", errors="replace"):
    m = rx.match(l)
    if not m: continue
    ts = m.group(1); t = datetime.strptime(ts[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
    qs = [int(x) for x in re.findall(r" q=(\d+)", m.group(6))]
    S.append((int(m.group(2)), t, m.group(3), int(m.group(4)), int(m.group(5)), qs))
def blockof(step):
    cur = None
    for s, b, a in starts:
        if step >= s: cur = (b, a)
    return cur
for i in range(len(S) - 1):
    pass
res = {}
for i in range(len(S) - 1):
    s = S[i]; nx = S[i + 1]
    bo = blockof(s[0])
    if not bo or bo[1] != arm: continue
    if s[2] != "decode" or s[3] != 4: continue
    if nx[0] != s[0] + 1: continue
    dt = (nx[1] - s[1]) * 1000
    if dt > 400: continue
    mq = max(s[5]) if s[5] else 0
    cls = "d0 (q all 1)" if mq <= 1 else "draft (max q %s)" % ("2-3" if mq <= 3 else "4+")
    res.setdefault(bo[0], {}).setdefault(cls, []).append((dt, s[4]))
for b in sorted(res):
    tot = sum(len(v) for v in res[b].values())
    out = []
    for c, v in sorted(res[b].items()):
        ms = [x[0] for x in v]; tk = [x[1] for x in v]
        out.append("%s: %d steps (%.0f%%) wall p50 %.1f mean %.1f ms, tok/step %.2f -> %.2f ms/token" % (c, len(v), 100.0 * len(v) / tot, st.median(ms), st.mean(ms), st.mean(tk), st.mean(ms) / st.mean(tk)))
    print("block %s %s c4-type steps %d | %s" % (b, arm, tot, " | ".join(out)))
