"""mockguard.py (mockrad, 12.11.2026) -- loaded into EVERY Python process of a mockrun.sh sandbox by a .pth file in the sandbox's
venv (so it holds under `python3 -I` too). With MOCKRAD_PORT set:
  * a TCP connect to a loopback port in MOCKRAD_REDIRECT (default 8090, the real server's) goes to 127.0.0.1:MOCKRAD_PORT;
    a loopback connect to MOCKRAD_PORT passes; ANY other connect (another port, another host, a unix socket) is refused
    (ConnectionRefusedError) -- so no helper can reach the real server, LiteLLM or docker.sock;
  * opening a file for WRITING outside MOCKRAD_SANDBOX (after symlinks are resolved; /dev/null and /tmp/mockrad* allowed) raises
    PermissionError -- a helper writing through a farm symlink into the real tree fails loudly instead of editing it;
  * reads of the cards' sysfs telemetry (/sys/class/drm, /sys/devices/.../drm) are refused (the real test's cards are not
    polled), and os.posix_fadvise is a no-op (no page-cache eviction of the real model files).
Every refusal is one line in MOCKRAD_STATE/guard.log (the report lists them)."""
import builtins, io, os, socket

_PORT = int(os.environ.get("MOCKRAD_PORT") or 0)
if _PORT:
    _ST = os.environ.get("MOCKRAD_STATE") or ""
    _SB = os.path.realpath(os.environ.get("MOCKRAD_SANDBOX") or "/nonexistent-sandbox")
    _RED = {int(x) for x in (os.environ.get("MOCKRAD_REDIRECT") or "8090").split(",") if x.strip().isdigit()}
    _LOOP = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::", "127.0.1.1"}

    def _glog(msg):
        try:
            with open(os.path.join(_ST, "guard.log"), "a") as f:
                import sys, time
                f.write("%s pid %d %s: %s\n" % (time.strftime("%F %T"), os.getpid(), " ".join(sys.argv[:3])[:200], msg))
        except Exception:
            pass

    def _map(sock, addr):
        if sock.family == socket.AF_UNIX:
            _glog("refused unix socket connect %r" % (addr,)); raise ConnectionRefusedError("mockguard: unix socket %r refused" % (addr,))
        if isinstance(addr, tuple) and len(addr) >= 2:
            h, p = addr[0], addr[1]
            if h in _LOOP or str(h).startswith("127."):
                if p == _PORT: return addr
                if p in _RED:
                    return ("::1", _PORT, 0, 0) if sock.family == socket.AF_INET6 else ("127.0.0.1", _PORT)
        _glog("refused connect %r" % (addr,))
        raise ConnectionRefusedError("mockguard: connect %r refused (only the mock on :%d)" % (addr, _PORT))

    _connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, addr):
        return _connect(self, _map(self, addr))

    def connect_ex(self, addr):
        try: a = _map(self, addr)
        except ConnectionRefusedError: return 111
        return _connect_ex(self, a)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex

    def _okw(path):
        try:
            if isinstance(path, int): return True
            p = os.fspath(path)
            if isinstance(p, bytes): p = p.decode(errors="replace")
            rp = os.path.realpath(p)
        except Exception:
            return True
        if rp == "/dev/null" or rp.startswith("/dev/std") or rp.startswith("/proc/self/") or rp.startswith("/dev/fd/"): return True
        if rp == _SB or rp.startswith(_SB + os.sep): return True
        if rp.startswith("/tmp/") and not rp.startswith("/tmp/claude"): return True
        return False

    def _gpu(path):
        try:
            p = os.fspath(path) if not isinstance(path, int) else ""
            if isinstance(p, bytes): p = p.decode(errors="replace")
        except Exception:
            return False
        return isinstance(p, str) and (p.startswith("/sys/class/drm") or ("/sys/devices/" in p and "/drm/" in p) or
                                       p.endswith("/gpu_metrics") or p.startswith("/dev/kfd") or p.startswith("/dev/dri"))

    _open, _osopen = builtins.open, os.open

    def open_(file, mode="r", *a, **k):
        if _gpu(file):
            _glog("refused GPU sysfs/device read %s" % file); raise PermissionError("mockguard: %s refused under the mock" % file)
        if any(c in mode for c in "wax+") and not _okw(file):
            _glog("refused write outside the sandbox: %s -> %s" % (file, os.path.realpath(os.fspath(file))))
            raise PermissionError("mockguard: write outside the sandbox refused: %s" % file)
        return _open(file, mode, *a, **k)

    def osopen(path, flags, *a, **k):
        if _gpu(path):
            _glog("refused GPU sysfs/device open %s" % path); raise PermissionError("mockguard: %s refused under the mock" % path)
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC) and not _okw(path):
            _glog("refused os.open for writing outside the sandbox: %s" % path)
            raise PermissionError("mockguard: write outside the sandbox refused: %s" % path)
        return _osopen(path, flags, *a, **k)

    builtins.open = open_
    io.open = open_
    os.open = osopen

    def _fadvise(fd, offset, length, advice):
        _glog("posix_fadvise suppressed (no page-cache eviction under the mock)")

    if hasattr(os, "posix_fadvise"): os.posix_fadvise = _fadvise
