#!/usr/bin/env python3
"""arwsum.py (12.11.2026, ds-main-port arw builder) -- the readings of arw/dsm-arw.sh.

  arwsum.py OUTDIR       reads OUTDIR/arw.txt (the harness, "ARW ..." lines) and OUTDIR/selftest-lane.txt (r4d_selftest
                         --case ar_twoshot_4rank_exact_lane) and prints:
                           CHECKS ...        every expected comparison / timing line present, the selftest clean (else VIRHE)
                           SHAPE T=... ...   one line a shape: identity, plain vs coded times, raw-group shares
                           SUMMARY ...       coded vs plain % per shape, the identity verdict
                           ESTIMATE ...      the 128k prefill step's all_reduce time at the T=4096 reading
                           VIRHE ...         any failed check
  arwsum.py --selftest   parses a synthetic run and checks the readings (no files).
"""
import os
import re
import sys

PROF_AR_MS, PROF_STEP_MS, PROF_CALL_US = 280.0, 1224.0, 3784.0   # runs/xdsm/dsm-225-prof-pf128-op.txt (coordinator 12.11.)
KV = re.compile(r'(\w+)=([^\s\[\]]+)')


def kv(line):
    return dict(KV.findall(line))


def parse(arw_text, st_text):
    out, virhe = [], []
    lines = arw_text.splitlines()
    cfg = next((l for l in lines if l.startswith('ARW CFG ')), None)
    if cfg is None:
        return ['VIRHE arw printed no CFG line (did not start)'], 1
    c = kv(cfg)
    rows = [int(x) for x in c['rows'].split(',')]
    pats = c['pats'].split(',')
    tpats = c['tpats'].split(',')
    m = re.search(r'variants:((?: \S+)+?) pats=', cfg)
    variants = m.group(1).split() if m else []
    ids, times, pat_raw = {}, {}, {}
    for l in lines:
        if l.startswith('ARW ID '):
            d = kv(l)
            verdict = l.rsplit('-> ', 1)[-1].strip()
            ids[(int(d['T']), d['pat'], d['var'], d['form'])] = (verdict, int(d['vsP_diff']), int(d['vsOracle_diff']), d.get('path', '?'), l)
        elif l.startswith('ARW TIME '):
            d = kv(l)
            times[(int(d['T']), d['pat'], d['var'])] = {k: float(d[k]) for k in ('own_med_us', 'own_p10_us', 'own_p90_us', 'max_med_us', 'bb_us')}
        elif l.startswith('ARW PAT '):
            d = kv(l)
            pat_raw[(int(d['T']), d['pat'])] = (float(d['raw_in_pct']), float(d['raw_sum_pct']))
    for l in lines:
        if 'VIRHE' in l and l.startswith('ARW ') and not l.startswith(('ARW ID ', 'ARW DONE ', 'ARW INTERPOSE ')):
            virhe.append(l[4:].strip()[:400])
    interpose = [l for l in lines if l.startswith('ARW INTERPOSE ')]
    if len(interpose) != 3 or any(not l.rstrip().endswith('-> OK') for l in interpose):
        virhe.append('interposition not proven (%d INTERPOSE OK line(s) of 3): the wire that ran is not seen' %
                     sum(1 for l in interpose if l.rstrip().endswith('-> OK')))
    done = next((l for l in lines if l.startswith('ARW DONE ')), None)
    # expected lines
    miss_id = [(T, p, v, f) for T in rows for p in pats for v in variants for f in ('inplace', 'oop') if (T, p, v, f) not in ids]
    miss_t = [(T, p, v) for T in rows for p in tpats for v in variants if (T, p, v) not in times]
    bad_id = [k for k, x in ids.items() if x[0] not in ('IDENTICAL', 'REF')]
    diffs = sum(x[1] for x in ids.values())
    odiffs = sum(x[2] for x in ids.values())
    ncmp = sum(1 for x in ids.values() if x[0] == 'IDENTICAL')
    # the selftest case
    st_ok = st_fail = st_skip = st_coded = 0
    st_rc = None
    for l in st_text.splitlines():
        if l.startswith('  ok    '):
            st_ok += 1
            st_coded += 1 if '(coded' in l else 0
        elif l.startswith('  FAIL  ') or 'FAIL' in l.split()[:1]:
            st_fail += 1
        elif l.startswith('  skip  '):
            st_skip += 1
        mm = re.match(r'SELFTEST_RC (\d+)', l)
        if mm:
            st_rc = int(mm.group(1))
    st_clean = st_rc == 0 and st_fail == 0 and st_skip == 0 and st_ok > 0 and st_coded > 0
    if not st_clean:
        virhe.append('r4d_selftest --case ar_twoshot_4rank_exact_lane not clean: rc %s, %d ok (%d coded), %d FAIL, %d skip' %
                     (st_rc, st_ok, st_coded, st_fail, st_skip))
    if miss_id:
        virhe.append('%d identity comparison(s) did not run, first %s' % (len(miss_id), miss_id[0]))
    if miss_t:
        virhe.append('%d timing reading(s) missing, first %s' % (len(miss_t), miss_t[0]))
    if bad_id:
        k = sorted(bad_id)[0]
        virhe.append('plain vs coded NOT identical / wrong wire: %s' % ids[k][4][4:400])
    if done is None or 'IDENTICAL' not in done:
        virhe.append('arw did not finish IDENTICAL (%s)' % (done or 'no DONE line'))
    ok_all = not virhe
    out.append('CHECKS %s: %d of %d identity comparisons ran (%d variants x in place / out of place x %d patterns x %d shapes, 4 ranks each), '
               '%d differences plain vs coded, %d vs the fp32 fold%s; %d of %d timing readings; wire seen through the PLT (%d/3 '
               'INTERPOSE OK); r4d_selftest lane case rc %s, %d ok (%d coded), %d FAIL, %d skip' %
               ('OK' if ok_all else 'VIRHE', len(rows) * len(pats) * len(variants) * 2 - len(miss_id),
                len(rows) * len(pats) * len(variants) * 2, len(variants), len(pats), len(rows), diffs, odiffs,
                '' if odiffs == 0 else ' (NOTE: the library contract says bit-identical to that fold)',
                len(times), len(rows) * len(tpats) * len(variants), sum(1 for l in interpose if l.rstrip().endswith('-> OK')),
                st_rc, st_ok, st_coded, st_fail, st_skip))
    # per shape
    summ, best_all = [], []
    tp0 = tpats[0] if tpats else 'act'
    for T in rows:
        n = T * int(c.get('n', '5120'))
        idv = [ids[k] for k in ids if k[0] == T]
        same = all(x[0] in ('IDENTICAL', 'REF') for x in idv) and len(idv) == len(pats) * len(variants) * 2
        parts = ['T=%d (%d el, %.1f MB bf16): %s' % (T, n, n * 2 / 1e6, 'IDENTICAL' if same else 'NOT IDENTICAL / INCOMPLETE')]
        P = times.get((T, tp0, 'P'))
        Cc = times.get((T, tp0, 'C'))
        if P and Cc:
            pc = 100.0 * (Cc['own_med_us'] / P['own_med_us'] - 1.0)
            pcb = 100.0 * (Cc['bb_us'] / P['bb_us'] - 1.0) if P['bb_us'] > 0 else 0.0
            parts.append('%s: P %.1f us [p10 %.1f p90 %.1f; bb %.1f] | C (entry, 4x1024) %.1f us [%.1f %.1f; bb %.1f] = %+.1f %% (bb %+.1f %%)' %
                         (tp0, P['own_med_us'], P['own_p10_us'], P['own_p90_us'], P['bb_us'], Cc['own_med_us'], Cc['own_p10_us'],
                          Cc['own_p90_us'], Cc['bb_us'], pc, pcb))
            cands = [(times[(T, tp0, v)]['own_med_us'], v) for v in variants if v.startswith('C') and (T, tp0, v) in times]
            bm, bv = min(cands)
            bpct = 100.0 * (bm / P['own_med_us'] - 1.0)
            if bv != 'C':
                parts.append('best coded %s %.1f us = %+.1f %%' % (bv, bm, bpct))
            others = ', '.join('%s %+.1f' % (v, 100.0 * (times[(T, tp0, v)]['own_med_us'] / P['own_med_us'] - 1.0))
                               for v in variants if v.startswith('C') and v != 'C' and (T, tp0, v) in times)
            if others:
                parts.append('wider coded grids: %s %%' % others)
            summ.append('T%d C %+.1f %% / best %s %+.1f %%' % (T, pc, bv, bpct))
            best_all.append((T, P['own_med_us'], Cc['own_med_us'], bv, bm))
        for p2 in tpats[1:]:
            P2, C2 = times.get((T, p2, 'P')), times.get((T, p2, 'C'))
            if P2 and C2:
                parts.append('%s: P %.1f | C %.1f us = %+.1f %%' % (p2, P2['own_med_us'], C2['own_med_us'],
                                                                    100.0 * (C2['own_med_us'] / P2['own_med_us'] - 1.0)))
        raws = ', '.join('%s %.2f/%.2f' % (p, pat_raw[(T, p)][0], pat_raw[(T, p)][1]) for p in pats if (T, p) in pat_raw)
        if raws:
            parts.append('raw groups %% in/sum: %s' % raws)
        out.append('SHAPE ' + ' | '.join(parts))
    verdict = 'YES' if ok_all else 'NO'
    out.append('SUMMARY identity plain = coded byte for byte: %s (%d comparisons, %d differences, every rank, in place + out of place, '
               'patterns %s; coded wire seen on every coded launch) | coded vs plain, %s, own median (negative = coded faster): %s' %
               (verdict, ncmp, diffs, '/'.join(pats), tp0, '; '.join(summ) if summ else 'no timing'))
    b4 = next((b for b in best_all if b[0] == 4096), None)
    if b4 and ok_all:   # no estimate from a run whose reading stopped
        calls = PROF_AR_MS * 1000.0 / PROF_CALL_US
        sc = calls * (b4[1] - b4[2]) / 1000.0
        sb = calls * (b4[1] - b4[4]) / 1000.0
        out.append('ESTIMATE 128k prefill step (profile: all_reduce %.0f ms of %.0f ms, %.0f us a call -> %.0f calls of 4096 rows): this '
                   'run P %.1f us a call (%+.1f %% vs the profile\'s %.0f); coded C (entry 4x1024) %+.1f ms a step (%+.1f %% of the step), '
                   'best coded %s %+.1f ms (%+.1f %%) -- standalone launches, which is how DS runs them (no overlap)' %
                   (PROF_AR_MS, PROF_STEP_MS, PROF_CALL_US, calls, b4[1], 100.0 * (b4[1] / PROF_CALL_US - 1.0), PROF_CALL_US, -sc,
                    -100.0 * sc / PROF_STEP_MS, b4[3], -sb, -100.0 * sb / PROF_STEP_MS))
    for v in virhe:
        out.append('VIRHE ' + v)
    return out, 0 if ok_all else 1


def synth(diff=False):
    rows, pats, tpats, vs = [4096, 8], ['bin', 'act'], ['act'], ['P', 'C', 'C8']
    L = ['ARW CFG rows=4096,8 n=5120 maxrows=4160 (instance numel 21299200) reps=100 warm=5 bb=20 seed=1 variants: P C C8 pats=bin,act tpats=act']
    for s in ('r4d_ar_twoshot_4rank_exact', 'r4d_ar_twoshot_4rank_exact_coded', 'r4d_ar_twoshot_coded_ok'):
        L.append('ARW INTERPOSE %s: 1 GOT slot(s) in the library, 1 bound to this harness -> OK' % s)
    for T in rows:
        for p in pats:
            L.append('ARW PAT T=%d n=%d pat=%s raw_in_pct=1.000 raw_sum_pct=2.000 (x)' % (T, T * 5120, p))
            for v in vs:
                for f in ('inplace', 'oop'):
                    ref = v == 'P' and f == 'inplace'
                    d = 1 if (diff and v == 'C8' and T == 8) else 0
                    L.append('ARW ID T=%d n=%d pat=%s var=%s form=%s rc=0,0,0,0 path=OK [x] vsP_diff=%d vsOracle_diff=0 -> %s' %
                             (T, T * 5120, p, v, f, d, 'REF' if ref else ('VIRHE' if d else 'IDENTICAL')))
            if p in tpats:
                for i, v in enumerate(vs):
                    med = 1000.0 * T / 4096 * (1.0 - 0.1 * i)
                    L.append('ARW TIME T=%d n=%d pat=%s var=%s own_med_us=%.1f own_p10_us=%.1f own_p90_us=%.1f max_med_us=%.1f bb_us=%.1f reps=100 bb=20 '
                             'algbw_gbs=1 vsP_pct=0' % (T, T * 5120, p, v, med, med * 0.9, med * 1.1, med * 1.2, med))
    L.append('ARW DONE %s oracle_diffs=0' % ('VIRHE' if diff else 'IDENTICAL'))
    st = '  ok    a bf16, coded 1 numel 4194304 (coded, in place) == x\n  ok    b\nSELFTEST_RC 0\n'
    return '\n'.join(L) + '\n', st


def selftest():
    a, st = synth()
    out, rc = parse(a, st)
    good = rc == 0 and out[0].startswith('CHECKS OK') and any(l.startswith('SUMMARY identity plain = coded byte for byte: YES') for l in out) \
        and any('C (entry, 4x1024) 900.0' in l for l in out) and any('best coded C8' in l for l in out) and any(l.startswith('ESTIMATE') for l in out)
    a2, st2 = synth(diff=True)
    out2, rc2 = parse(a2, st2)
    good2 = rc2 == 1 and out2[0].startswith('CHECKS VIRHE') and any('NOT identical' in l for l in out2)
    out3, rc3 = parse(a, st.replace('SELFTEST_RC 0', 'SELFTEST_RC 1'))
    good3 = rc3 == 1
    out4, rc4 = parse(a.replace('-> OK', '-> VIRHE (x)', 1), st)
    good4 = rc4 == 1
    out5, rc5 = parse('\n'.join(l for l in a.splitlines() if not l.startswith('ARW TIME T=8 ')), st)
    good5 = rc5 == 1
    if good and good2 and good3 and good4 and good5:
        print('SELFTEST OK (clean run, a difference, a dirty selftest, an unproven interposition, a missing timing line)')
        return 0
    print('SELFTEST FAIL', good, good2, good3, good4, good5)
    for l in out + out2:
        print('  ', l[:300])
    return 1


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--selftest':
        return selftest()
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    try:
        a = open(os.path.join(d, 'arw.txt'), errors='replace').read()
    except OSError:
        a = ''
    try:
        st = open(os.path.join(d, 'selftest-lane.txt'), errors='replace').read()
    except OSError:
        st = ''
    out, rc = parse(a, st)
    for l in out:
        print(l)
    return rc


if __name__ == '__main__':
    sys.exit(main())
