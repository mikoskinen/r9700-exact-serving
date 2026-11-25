#!/bin/bash
# patch-body.sh (12.11.2026, queue-ar/dsmain/attach, ds-main-port ATTACH builder) -- make a stack5-family body ATTACH-capable. Two
# edits, both inert unless the wrapper runs with ATTACH=1 (unset = the body behaves exactly as before):
#   1. the cards-free check  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && {   (the real run's, not the DRY info line)
#      -> docker ps --format '{{.Names}}' | grep -iE '^radiance' | grep -vxF "${ATTACH_CLAIMED_CN:-.}" | grep -q . && {
#      (the server the wrapper's cguard claimed for this item is not "busy"; any other radiance container still is)
#   2. swnote's start-state check  s0["epoch"] == 0  -> s0["epoch"] == int(__import__("os").environ.get("ATTACH_EPOCH0") or 0)
#      (an attached server's epoch is the one its reset left; restart() exports ATTACH_EPOCH0; unset = 0 as before)
#   patch-body.sh [--check] BODY...   --check: report what would change, change nothing. Else: BODY.bak-<MMDD-HHMM>-attach, then the
#   edit via write-to-temp + mv (a running bash keeps reading its old file). Idempotent: an already patched body is reported, kept.
set -u
CHECK=0; [ "${1:-}" = --check ] && { CHECK=1; shift; }
[ $# -gt 0 ] || { sed -n '2,13p' "$0"; exit 2; }
rc=0
for b in "$@"; do
  [ -f "$b" ] || { echo "$b: no such file"; rc=1; continue; }
  out=$(python3 -I -B - "$b" "$CHECK" <<'PY'
import os, re, sys, time
b, check = sys.argv[1], sys.argv[2] == "1"
s = open(b).read()
old1 = "docker ps --format '{{.Names}}' | grep -qiE '^radiance' && {"
new1 = "docker ps --format '{{.Names}}' | grep -iE '^radiance' | grep -vxF \"${ATTACH_CLAIMED_CN:-.}\" | grep -q . && {"
old2 = 's0["epoch"] == 0'
new2 = 's0["epoch"] == int(__import__("os").environ.get("ATTACH_EPOCH0") or 0)'
L = s.split("\n"); n1 = n2 = 0
for i, l in enumerate(L):
    if old1 in l and "cards are not free" in l and not l.lstrip().startswith("echo"):
        L[i] = l.replace(old1, new1); n1 += 1
    if old2 in l:
        L[i] = L[i].replace(old2, new2); n2 += 1
p1, p2 = s.count(new1), s.count(new2)
t = "\n".join(L)
state = "cards-free check: %d patched now, %d already | epoch-0 check: %d patched now, %d already" % (n1, p1, n2, p2)
if n1 + p1 == 0: state += " | NOTE: no cards-free check found (not a stack5-family body?)"
if t == s: print("UNCHANGED " + state); sys.exit(0)
if check: print("WOULD PATCH " + state); sys.exit(0)
bak = "%s.bak-%s-attach" % (b, time.strftime("%m%d-%H%M"))
open(bak, "w").write(s); os.chmod(bak, os.stat(b).st_mode & 0o777)
tmp = "%s.tmp-attach.%d" % (b, os.getpid())
open(tmp, "w").write(t); os.chmod(tmp, os.stat(b).st_mode & 0o777); os.replace(tmp, b)
print("PATCHED %s (backup %s)" % (state, os.path.basename(bak)))
PY
) || rc=1
  echo "$b: $out"
  [ $CHECK = 1 ] || bash -n "$b" || { echo "$b: VIRHE bash -n fails after the patch"; rc=1; }
done
exit $rc
