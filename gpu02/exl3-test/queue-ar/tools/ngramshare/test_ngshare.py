#!/usr/bin/env python3
"""test_ngshare.py (12.11.2026, ds-main-port ngshare builder) -- NO GPU, NO engine, NO docker: a harness that imitates
Radiance's n-gram share-file protocol (core/format/table_share.cpp, radiance-int8 7e03384) syscall for syscall on small dummy
files in a temp dir under /dev/shm, and drives the REAL ngram-keeper.sh / ngram-shm-cleanup.sh beside this file against it
(NGSHARE_DIR / NGSHARE_RUN / NGSHARE_LOG pointed into the temp dir; `docker` on PATH replaced by a stub that prints a chosen
`docker ps` list). Removes its temp dir and stops its own keeper processes at the end.

    python3 -I test_ngshare.py            run every scenario; exit 0 only if every check passes

The imitated engine ("engine" role, one process = one Engine, its tables opened in order as copy_table does and closed in
order as ~Engine does, engine.cpp:327):
  table_share_open      table_share.cpp:121-148  open(O_RDWR|O_CREAT|O_CLOEXEC, 0644), flock LOCK_SH (blocking), fstat, retry
                                                 while st_nlink == 0
                        :151-160                 classify -> Ready: reuse
                        :162-172                 open <file>.lock (O_CREAT), flock LOCK_EX (blocking) = the fill lock
                        :174-185                 classify again -> Ready: close the lock fd, reuse
                        :186-192                 ftruncate to table_share_file_bytes(n) -> filler
  classify              :44-75                   size 0 Fresh, 56-byte header, zero Partial, magic/version/id/bytes, ready flag,
                                                 ready-file length
  fill / check          engine_bringup.cpp:445-452  filler: write the table at kTableShareHeader (stands for table_copy_fill;
                                                 the real engine writes through its MAP_SHARED mapping), reader: compare probes
                                                 (stands for table_copy_check)
  table_share_mark_ready :196-216                pwrite the header (magic, version 1, ready 1, bytes, pid, id), close the lock fd
  table_share_close     :218-249                 close lock fd; LOCK_UN; 10 x flock(LOCK_EX|LOCK_NB) with 100 ms sleeps between
                                                 tries; on LOCK_EX: if fstat(fd) dev/ino == stat(path) dev/ino, unlink the lock
                                                 file then the data file; stop on any errno but EWOULDBLOCK / EINTR; close
Not imitated (not part of the lock protocol): the mmap + pin of the table (rad_dev_map_file), the 2 MiB alignment's meaning
for the device, the placement decision's table_share_is_ready (a lock-free header read, = ngshare_hdr.py).
"""
import fcntl
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KEEPER = os.path.join(HERE, "ngram-keeper.sh")
CLEANUP = os.path.join(HERE, "ngram-shm-cleanup.sh")
MAGIC = b"RADNGRM1"
HEADER = struct.Struct("<8sIIQQ24s")
HUGE = 2 << 20
LEAVE_TRIES, LEAVE_WAIT = 10, 0.1          # kLeaveTries, kLeaveWaitUs (table_share.cpp:24-25)
KEEPER_CMD = "flock -s -n -o -E 75 /dev/fd/9 sleep infinity "


def file_bytes(n):
    return HUGE + ((n + HUGE - 1) & ~(HUGE - 1))


def key(path):
    """/proc/locks' name for a file: MAJ:MIN in hex, inode in decimal."""
    st = os.stat(path)
    return "%02x:%02x:%d" % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)


def pattern(ident, n):
    blk = hashlib.sha256(ident.encode()).digest() * 4096
    return (blk * (n // len(blk) + 1))[:n]


# ------------------------------------------------------------------ the imitated engine (subprocess role)
def classify(fd, ident, n):
    st = os.fstat(fd)
    if st.st_size == 0:
        return "fresh"
    raw = os.pread(fd, HEADER.size, 0)
    if len(raw) != HEADER.size:
        return "foreign"
    if raw == bytes(HEADER.size):
        return "partial"
    magic, ver, ready, nb, _pid, hid = HEADER.unpack(raw)
    if magic != MAGIC or ver != 1:
        return "foreign"
    if nb != n or hid.split(b"\0", 1)[0].decode() != ident:
        return "foreign"
    if not ready:
        return "partial"
    if st.st_size != file_bytes(n):
        return "foreign"
    return "ready"


def ts_open(d, ident, n):
    s = {"path": os.path.join(d, "ngram-%s.tab" % ident), "fd": -1, "lock_fd": -1, "ready": False, "filler": False}
    s["lock_path"] = s["path"] + ".lock"
    t = time.monotonic()
    while True:                                                         # :121-148
        s["fd"] = os.open(s["path"], os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        fcntl.flock(s["fd"], fcntl.LOCK_SH)
        if os.fstat(s["fd"]).st_nlink > 0:
            break
        os.close(s["fd"])
    s["sh_wait_ms"] = round((time.monotonic() - t) * 1000, 1)
    seen = classify(s["fd"], ident, n)                                  # :151
    if seen == "foreign":
        raise SystemExit("foreign file")
    if seen == "ready":                                                 # :157-160
        s["ready"] = True
        return s
    s["lock_fd"] = os.open(s["lock_path"], os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)   # :162
    t = time.monotonic()
    fcntl.flock(s["lock_fd"], fcntl.LOCK_EX)                            # :168
    s["fill_lock_wait_ms"] = round((time.monotonic() - t) * 1000, 1)
    seen = classify(s["fd"], ident, n)                                  # :174
    if seen == "foreign":
        raise SystemExit("foreign file")
    if seen == "ready":                                                 # :180-184
        os.close(s["lock_fd"])
        s["lock_fd"] = -1
        s["ready"] = True
        return s
    os.ftruncate(s["fd"], file_bytes(n))                                # :186
    s["filler"] = True
    return s


def ts_mark_ready(s, ident, n):                                         # :196-216
    h = HEADER.pack(MAGIC, 1, 1, n, os.getpid(), ident.encode().ljust(24, b"\0"))
    os.pwrite(s["fd"], h, 0)
    if s["lock_fd"] >= 0:
        os.close(s["lock_fd"])
        s["lock_fd"] = -1
    s["ready"] = True


def ts_close(s):                                                        # :218-249
    if s["lock_fd"] >= 0:
        os.close(s["lock_fd"])
    deleted, tries = False, 0
    if s["fd"] >= 0:
        fcntl.flock(s["fd"], fcntl.LOCK_UN)                             # :232
        for i in range(LEAVE_TRIES):                                    # :233
            tries += 1
            try:
                fcntl.flock(s["fd"], fcntl.LOCK_EX | fcntl.LOCK_NB)     # :234
            except BlockingIOError:                                     # EWOULDBLOCK (EINTR: Python retries itself)
                if i + 1 < LEAVE_TRIES:
                    time.sleep(LEAVE_WAIT)                              # :244
                continue
            except OSError:
                break                                                   # :243
            mine = os.fstat(s["fd"])                                    # :235-240
            try:
                named = os.stat(s["path"])
            except OSError:
                named = None
            if named and (mine.st_dev, mine.st_ino) == (named.st_dev, named.st_ino):
                for p in (s["lock_path"], s["path"]):                   # the lock file first (:229-231)
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                deleted = True
            break                                                       # :241
        os.close(s["fd"])
    return deleted, tries


def engine_role(d, tables, fill_delay):
    """tables: 'id:n,id:n'. Prints one JSON line per table when up, waits for a line on stdin, closes, prints JSON."""
    shares = []
    for spec in tables.split(","):
        ident, n = spec.split(":")
        n = int(n)
        t0 = time.monotonic()
        s = ts_open(d, ident, n)
        if s["filler"]:
            time.sleep(fill_delay)                                      # stands for table_copy_fill's 74 s + 9 s
            os.pwrite(s["fd"], pattern(ident, n), HUGE)
            ts_mark_ready(s, ident, n)
            role = "filled"
        else:
            data = pattern(ident, n)
            for off in (0, n // 3, n // 2, n - 4096):                   # stands for table_copy_check's probes
                if os.pread(s["fd"], 4096, HUGE + off) != data[off:off + 4096]:
                    raise SystemExit("probe mismatch")
            role = "reused (probes match)"
        shares.append(s)
        print(json.dumps({"ev": "up", "id": ident, "role": role, "sh_wait_ms": s["sh_wait_ms"],
                          "fill_lock_wait_ms": s.get("fill_lock_wait_ms"), "s": round(time.monotonic() - t0, 2)}), flush=True)
    sys.stdin.readline()
    for s in shares:                                                    # ~Engine closes in order (engine.cpp:327)
        t0 = time.monotonic()
        deleted, tries = ts_close(s)
        print(json.dumps({"ev": "left", "id": os.path.basename(s["path"])[6:22], "deleted": deleted, "tries": tries,
                          "s": round(time.monotonic() - t0, 2)}), flush=True)


# ------------------------------------------------------------------ the driver
class T:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="ngshare-selftest.", dir="/dev/shm")
        self.dir = os.path.join(self.tmp, "ds41ng")
        self.run = os.path.join(self.tmp, "run")
        self.log = os.path.join(self.tmp, "queue-changes.log")
        self.ps = os.path.join(self.tmp, "docker-ps.txt")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        with open(os.path.join(bindir, "docker"), "w") as f:
            f.write('#!/bin/sh\n[ "$1" = ps ] || exit 1\ncat "%s" 2>/dev/null\nexit 0\n' % self.ps)
        os.chmod(os.path.join(bindir, "docker"), 0o755)
        self.env = dict(os.environ, NGSHARE_DIR=self.dir, NGSHARE_RUN=self.run, NGSHARE_LOG=self.log,
                        PATH=bindir + ":" + os.environ["PATH"])
        self.results = []
        self.procs = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok)))
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [" + detail + "]") if detail else ""), flush=True)

    def sh(self, *args, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        r = subprocess.run(list(args), env=env, capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr).rstrip()
        for line in out.splitlines():
            print("      | " + line)
        return r.returncode, out

    def engine(self, tables, fill_delay=0.0):
        p = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "engine", self.dir, tables, str(fill_delay)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        return p

    @staticmethod
    def up(p, k):
        return [json.loads(p.stdout.readline()) for _ in range(k)]

    @staticmethod
    def leave(p, k):
        p.stdin.write("leave\n")
        p.stdin.flush()
        ev = [json.loads(p.stdout.readline()) for _ in range(k)]
        p.wait(10)
        return ev

    def path(self, ident):
        return os.path.join(self.dir, "ngram-%s.tab" % ident)

    def locks(self, ident):
        """[(mode, pid, is_keeper)] granted flock locks on the data file now."""
        try:
            st = os.stat(self.path(ident))
        except OSError:
            return []
        k = key(self.path(ident))
        out = []
        with open("/proc/locks") as f:
            for line in f:
                w = line.split()
                if len(w) >= 6 and w[1] == "FLOCK" and w[5] == k:
                    out.append((w[3], int(w[4]), self.is_keeper(int(w[4]))))
        return out

    @staticmethod
    def is_keeper(pid):
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                return f.read().replace(b"\0", b" ").decode() == KEEPER_CMD
        except OSError:
            return False

    def keeper_lock_on_any(self):
        """keeper-held locks on ANY inode (data or .lock) -> list of /proc/locks lines."""
        with open("/proc/locks") as f:
            return [l.strip() for l in f if len(l.split()) >= 6 and l.split()[1] == "FLOCK" and self.is_keeper(int(l.split()[4]))]

    def exists(self, ident):
        return os.path.exists(self.path(ident)), os.path.exists(self.path(ident) + ".lock")

    def finish(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        subprocess.run([KEEPER, "stop"], env=self.env, capture_output=True, timeout=30)
        shutil.rmtree(self.tmp, ignore_errors=True)


def scenarios(t):
    A, B, C, D = "aaaa000000000001", "bbbb000000000002", "cccc000000000003", "dddd000000000004"
    NA, NB = 3 * (1 << 20) + 4097, (1 << 20) + 13       # stand for the 91.56 GiB + 11.44 GiB tables
    os.makedirs(t.dir)

    print("S0 protocol sanity (no keeper): a filler and an engine that arrives during the fill, stopped together")
    e1 = t.engine("%s:%d" % (C, NA), fill_delay=1.0)
    time.sleep(0.3)
    e2 = t.engine("%s:%d" % (C, NA))
    u1, u2 = t.up(e1, 1)[0], t.up(e2, 1)[0]
    t.check("first engine fills, the one arriving mid-fill waits on the fill lock (:168) then reuses (:180-184)",
            u1["role"] == "filled" and u2["role"].startswith("reused") and (u2["fill_lock_wait_ms"] or 0) > 300,
            "e2 fill-lock wait %s ms" % u2["fill_lock_wait_ms"])
    l1 = t.leave(e1, 1)[0]
    l2 = t.leave(e2, 1)[0]
    t.check("of two holders only the last deletes (:233-240)", not l1["deleted"] and l2["deleted"] and t.exists(C) == (False, False),
            "first: deleted=%s after %d tries %.2f s; last: deleted=%s" % (l1["deleted"], l1["tries"], l1["s"], l2["deleted"]))

    print("S1 WITHOUT the keeper: one engine (two tables) fills, stops gracefully (SIGTERM -> ~Engine)")
    e = t.engine("%s:%d,%s:%d" % (A, NA, B, NB))
    u = t.up(e, 2)
    t.check("fresh start fills both tables", all(x["role"] == "filled" for x in u))
    lv = t.leave(e, 2)
    t.check("graceful stop, no keeper: the last holder deletes data + lock files",
            all(x["deleted"] for x in lv) and t.exists(A) == (False, False) and t.exists(B) == (False, False),
            "; ".join("%s deleted=%s tries=%d" % (x["id"][:4], x["deleted"], x["tries"]) for x in lv))

    print("S2 WITHOUT the keeper, SIGKILL (= docker rm -f): ~Engine never runs")
    e = t.engine("%s:%d" % (D, NA))
    t.up(e, 1)
    e.kill()
    e.wait()
    t.check("SIGKILL leaves the file (no close ran), no lock remains", t.exists(D)[0] and t.locks(D) == [])
    e = t.engine("%s:%d" % (D, NA))
    u = t.up(e, 1)[0]
    lv = t.leave(e, 1)[0]
    t.check("the next engine reuses it, and its graceful stop deletes it (no keeper)",
            u["role"].startswith("reused") and lv["deleted"] and not t.exists(D)[0])

    print("S3 WITH the keeper: keeper start while the filler is still filling (fill lock held, header zero)")
    e1 = t.engine("%s:%d,%s:%d" % (A, NA, B, NB), fill_delay=1.5)
    time.sleep(0.4)
    rc, out = t.sh(KEEPER, "start")
    t.check("keeper start mid-fill locks nothing (file not ready) and returns 1", rc == 1 and t.keeper_lock_on_any() == [],
            "rc %d, keeper locks %s" % (rc, t.keeper_lock_on_any()))
    bg = subprocess.Popen([KEEPER, "start", "--wait", "20", "--expect", "2"], env=t.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    u = t.up(e1, 2)
    t.check("the fills are not slowed by the keeper (each = its 1.5 s stand-in)",
            all(x["role"] == "filled" and x["s"] < 1.9 for x in u), ", ".join("%.2f s" % x["s"] for x in u))
    out = bg.communicate(timeout=30)[0]
    for line in out.rstrip().splitlines():
        print("      | " + line)
    la, lb = t.locks(A), t.locks(B)
    t.check("keeper start --wait --expect 2 holds both ready files with LOCK_SH, beside the engine's own LOCK_SH",
            bg.returncode == 0 and sorted(m for m, _, k in la) == ["READ", "READ"] and any(k for _, _, k in la)
            and any(k for _, _, k in lb), "A %s B %s" % (la, lb))
    lockkeys = {key(t.path(x) + ".lock") for x in (A, B)}
    t.check("the keeper holds no lock on any .lock file (the fill lock)",
            not [l for l in t.keeper_lock_on_any() if l.split()[5] in lockkeys], "lock files present: %s" % len(lockkeys))
    lv = t.leave(e1, 2)
    t.check("graceful stop WITH the keeper: the engine is not last, both files survive",
            not any(x["deleted"] for x in lv) and t.exists(A)[0] and t.exists(B)[0],
            "; ".join("%s deleted=%s tries=%d %.2f s" % (x["id"][:4], x["deleted"], x["tries"], x["s"]) for x in lv))

    print("S4 the next engine start reuses the kept files")
    e2 = t.engine("%s:%d,%s:%d" % (A, NA, B, NB))
    u = t.up(e2, 2)
    t.check("next start: 'reused (probes match)' for both, no fill, LOCK_SH not delayed by the keeper",
            all(x["role"].startswith("reused") and x["sh_wait_ms"] < 50 for x in u),
            ", ".join("%s %s sh_wait %.1f ms" % (x["id"][:4], x["role"], x["sh_wait_ms"]) for x in u))
    rc, out = t.sh(KEEPER, "start")
    t.check("keeper start again is idempotent (already held, no second holder)",
            rc == 0 and out.count("already held") == 2 and sum(k for _, _, k in t.locks(A)) == 1)
    rc, out = t.sh(KEEPER, "status")
    lv = t.leave(e2, 2)
    t.check("second graceful stop: files still kept", not any(x["deleted"] for x in lv) and t.exists(A)[0] and t.exists(B)[0])

    print("S5 cleanup")
    with open(t.ps, "w") as f:
        f.write("radiance-ds41-exl3 radiance-local:ds41-test\n")
    rc, out = t.sh(CLEANUP)
    t.check("refused while a radiance container runs (docker ps stub), files + keeper untouched",
            rc == 1 and t.exists(A)[0] and any(k for _, _, k in t.locks(A)))
    open(t.ps, "w").close()
    e3 = t.engine("%s:%d" % (A, NA))
    t.up(e3, 1)
    rc, out = t.sh(CLEANUP)
    t.check("refused while a non-keeper process holds a file (an engine docker ps did not name)", rc == 1 and t.exists(A)[0])
    t.leave(e3, 1)
    rc, out = t.sh(CLEANUP, extra_env={"DRYRUN": "1"})
    t.check("DRYRUN=1 changes nothing", rc == 0 and t.exists(A)[0] and any(k for _, _, k in t.locks(A)))
    rc, out = t.sh(CLEANUP)
    t.check("cleanup with nothing running: keeper stopped, data + lock files removed, directory kept",
            rc == 0 and t.exists(A) == (False, False) and t.exists(B) == (False, False) and t.keeper_lock_on_any() == []
            and os.path.isdir(t.dir) and not [x for x in os.listdir(t.run) if x.endswith(".pid")])
    with open(t.log) as f:
        log = f.read()
    print("      log: " + log.strip().replace("\n", "\n      log: "))
    t.check("cleanup logged its refusals and its removal", log.count("ngshare cleanup: REFUSED") == 2 and "removed 2 table file(s)" in log)

    print("S6 keeper on an empty directory makes no file (no O_CREAT anywhere in the keeper)")
    rc, out = t.sh(KEEPER, "start")
    t.check("keeper start on an empty dir: rc 1, directory still empty", rc == 1 and os.listdir(t.dir) == [])
    e = t.engine("%s:%d" % (A, NA))
    u = t.up(e, 1)[0]
    lv = t.leave(e, 1)[0]
    t.check("after cleanup the next engine fills afresh and (no keeper) deletes at stop", u["role"] == "filled" and lv["deleted"])


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "engine":
        engine_role(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        return 0
    t = T()
    print("temp dir %s (tmpfs), keeper %s, cleanup %s" % (t.tmp, KEEPER, CLEANUP))
    try:
        scenarios(t)
    except Exception as e:                       # noqa: BLE001 -- a crash is a failed check, then clean up
        t.check("harness ran to the end", False, repr(e))
    finally:
        t.finish()
    gone = not os.path.exists(t.tmp)
    t.check("temp dir removed", gone)
    bad = [n for n, ok in t.results if not ok]
    print("RESULT ngshare selftest: %d / %d checks pass%s" % (len(t.results) - len(bad), len(t.results),
                                                              "" if not bad else " -- FAIL: " + " | ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
