import re,sys,datetime
log=sys.argv[1]; t0=sys.argv[2]; t1=sys.argv[3]; w0=sys.argv[4]; w1=sys.argv[5]
pat=re.compile(r'^(\S+)Z I step (\d+) (\S+) n_seq=(\d+) n_tok=(\d+)\s+(.*)$')
rows=[]
for l in open(log):
    m=pat.match(l)
    if not m: continue
    ts=m.group(1)
    if ts<t0 or ts>t1: continue
    segs=re.findall(r'\[(\d+)\] slot=(\d+) q=(\d+) ctx=(\d+)',m.group(6))
    rows.append((ts,int(m.group(2)),m.group(3),int(m.group(4)),int(m.group(5)),segs))
def tsec(ts):
    return datetime.datetime.fromisoformat(ts[:26]).timestamp()
out=[]
for i,r in enumerate(rows):
    ts,n,kind,ns,nt,segs=r
    dur=(tsec(rows[i+1][0])-tsec(ts))*1000 if i+1<len(rows) else None
    big=[s for s in segs if int(s[2])>16]
    if kind!='decode' or big:
        inwin = w0<=ts<=w1
        out.append((ts,n,kind,ns,nt,len(big),[(s[1],s[2],s[3]) for s in big],round(dur,1) if dur else None,inwin))
for o in out: print(*o)
