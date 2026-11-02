# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The control server: drive a running Ochre from another program.

Off unless asked for (`--control`). It listens on a Unix socket only --
never TCP -- that only this user can open, and speaks one JSON object per
line each way:

    -> {"id": 1, "token": "...", "method": "run",
        "params": {"line": "image.resize width=100 height=50"}}
    <- {"id": 1, "ok": true, "result": {"command": "image.resize", "changed": true}}
    <- {"id": 2, "ok": false, "error": {"code": "refused", "message": "..."}}

The permission model is Kooka Pictura's (its control_policy.cpp and
control_server_gate.cpp), which has already been argued over:

- A per-session random token, written to <socket>.token with mode 0600 and
  compared in constant time. Being able to reach the socket is not enough;
  reading the token file is the proof of being this user.
- Tiers read < edit < save. read looks (and may open files); edit changes
  the document; save writes files. A request above the session's tier is
  refused -- or, in a window with a person at it, asked about.
- Folders ("roots") confine every file the session touches, symlinks
  resolved first so a link inside a root cannot point out of it. Opening
  inside a root is allowed; writing a NEW file inside one is allowed;
  overwriting a file the agent did not write this session is asked about,
  or refused when nobody is there to ask. The state directory and the
  endpoint's own files (socket, token, lock) are refused always.
- Requests are bounded: 1 MiB a line, 4 connections, 64 queued a client.

The server does the enforcing; the MCP bridge (mcp.py) enforces nothing, so
an agent cannot talk its way around a check by using a different client.

Qt-free. The window drives the same server from its event loop and supplies
the person (controlbar.py); the headless server simply has nobody to ask.
"""

import base64
import errno
import fcntl
import hmac
import io
import json
import os
import secrets
import selectors
import socket
import stat
import struct
import tempfile
from collections import deque

from . import scripting

TIERS = ("read", "edit", "save")

MAX_REQUEST_BYTES = 1 << 20
MAX_CONNECTIONS = 4
MAX_QUEUED = 64
MAX_PATH = 4096
MAX_RENDER = 2048
SEND_TIMEOUT = 5.0

ALLOW, ASK, REFUSE = "allow", "ask", "refuse"


class ControlError(Exception):
    """A request that failed, as (code, message) for the reply."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def response(self):
        return error(self.code, self.message)


def ok(result):
    return {"ok": True, "result": result}


def error(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


# ---- where things live -------------------------------------------------------

def default_socket_path():
    """$XDG_RUNTIME_DIR/ochre-control.sock, else a per-user name in /tmp."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isdir(runtime):
        return os.path.join(runtime, "ochre-control.sock")
    return os.path.join(tempfile.gettempdir(), "ochre-control-%d.sock" % os.getuid())


def token_path(socket_path):
    return socket_path + ".token"


def lock_path(socket_path):
    return socket_path + ".lock"


def canonical(path):
    """Where a path really lands: absolute, '..' gone, every symlink resolved
    (a dangling one too, so writing through it is judged by its target)."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _inside(target, directory):
    # At a separator boundary, so a root /a/b does not hold /a/b2/x.
    directory = directory.rstrip("/") or "/"
    if target == directory:
        return True
    return target.startswith(directory if directory == "/" else directory + "/")


# ---- the policy ---------------------------------------------------------------

class Policy:
    """What one control session may do.

    `live`: a person is at a window and can be asked. Without one, anything
    that would need asking is refused instead -- a headless server has no
    one to ask and must not hang waiting.
    """

    def __init__(self, tier="read", roots=(), denied=(), live=False):
        if tier not in TIERS:
            raise ValueError("tier must be one of %s, not %r" % (", ".join(TIERS), tier))
        self.tier = tier
        self.roots = []
        for root in roots:
            real = canonical(root)
            if not os.path.isdir(real):
                raise ValueError("control root is not a folder: %s" % root)
            self.roots.append(real)
        self.denied = [canonical(d) for d in denied if d]
        self.live = bool(live)
        # Files this session wrote, or that the person approved: the agent
        # may write them again without asking.
        self.known = set()

    def grants(self, tier):
        return TIERS.index(tier) <= TIERS.index(self.tier)

    def judge(self, target, writes):
        """ALLOW, ASK or REFUSE for a canonical path."""
        if any(_inside(target, d) for d in self.denied):
            return REFUSE
        overwrite = writes and os.path.lexists(target)
        if target in self.known or (any(_inside(target, r) for r in self.roots) and not overwrite):
            return ALLOW
        return ASK if self.live else REFUSE


# ---- the methods ------------------------------------------------------------------

class _Prepared:
    """A request checked and ready: what runs, and which files it touches."""

    def __init__(self, run, files=()):
        self.run = run
        self.files = list(files)          # [(canonical path, writes)]


class Gate:
    """Decides whether a request may run, runs it, and answers.

    `host` is the window, when there is one: busy_reason() says a person is
    mid-action, ask(question, answer) puts the question to them, and
    refresh() shows what a request changed. The headless gate has none.
    """

    # Each method and the tier it needs.
    METHODS = {
        "status": "read",
        "list_commands": "read",
        "list_effects": "read",
        "list_layers": "read",
        "get_pixel": "read",
        "render": "read",
        "history": "read",
        "open": "read",
        "new": "edit",
        "run": "edit",
        "undo": "edit",
        "redo": "edit",
        "save": "save",
        "export": "save",
    }

    def __init__(self, controller, policy, host=None):
        self.ctl = controller
        self.policy = policy
        self.host = host

    # ---- entry ---------------------------------------------------------------

    def call(self, method, params):
        """Synchronous form for callers with nobody to ask (the in-process
        MCP bridge): the response dict."""
        out = []
        self.serve(method, params, out.append)
        if not out:                       # only a live gate parks
            return error("refused", "this request needs a person's approval")
        return out[0]

    def serve(self, method, params, done):
        """Answer one request by calling done(response) exactly once -- now,
        or once a person has answered. Returns True if it answered now."""
        tier = self.METHODS.get(method)
        if tier is None:
            done(error("unknown_method", "unknown method: %s" % (method,)))
            return True
        try:
            prepared = self._prepare(method, params if params is not None else {})
        except ControlError as exc:
            done(exc.response())
            return True
        questions = []
        if not self.policy.grants(tier):
            if not (self.policy.live and self.host is not None):
                done(error("refused", "this session's tier is %s; %s needs %s"
                           % (self.policy.tier, method, tier)))
                return True
            questions.append("An agent wants to %s (%s tier; it was given %s)"
                             % (self._describe(method, params), tier, self.policy.tier))
        if tier != "read" and self.host is not None:
            reason = self.host.busy_reason()
            if reason:
                done(error("human_busy", "waiting for the person: %s" % reason))
                return True
        for target, writes in prepared.files:
            verdict = self.policy.judge(target, writes)
            if verdict == REFUSE or (verdict == ASK and self.host is None):
                done(error("refused", "this session may not %s %s"
                           % ("write" if writes else "read", target)))
                return True
            if verdict == ASK:
                questions.append("An agent wants to %s %s" % (
                    "overwrite" if writes and os.path.lexists(target)
                    else "write" if writes else "open", target))
        if questions:
            self.host.ask("\n".join(questions),
                          lambda answer: self._answered(answer, method, params, prepared, done))
            return False
        done(self._run(method, prepared))
        return True

    def _answered(self, answer, method, params, prepared, done):
        if answer != "allow":
            done(error("approval_timeout", "nobody answered") if answer == "timeout"
                 else error("refused", "the person declined"))
            return
        # The world may have moved while the question was up: judge again.
        try:
            again = self._prepare(method, params if params is not None else {})
        except ControlError as exc:
            done(exc.response())
            return
        if again.files != prepared.files or any(
                self.policy.judge(t, w) == REFUSE for t, w in again.files):
            done(error("refused", "the target changed"))
            return
        reason = self.host.busy_reason() if self.host is not None else None
        if reason and self.METHODS[method] != "read":
            done(error("human_busy", "waiting for the person: %s" % reason))
            return
        for target, _writes in again.files:
            self.policy.known.add(target)
        done(self._run(method, again))

    def _run(self, method, prepared):
        try:
            result = prepared.run()
        except ControlError as exc:
            return exc.response()
        except scripting.CommandError as exc:
            return error("failed", str(exc))
        except Exception as exc:                           # noqa: BLE001
            return error("failed", "%s: %s" % (type(exc).__name__, exc))
        for target, writes in prepared.files:
            if writes:
                self.policy.known.add(target)
        if self.METHODS[method] != "read" or method == "open":
            if self.host is not None:
                self.host.refresh(method)
        return ok(result)

    @staticmethod
    def _describe(method, params):
        if method == "run" and isinstance(params, dict):
            return "run %s" % (params.get("line") or params.get("command") or "a command")
        return method

    # ---- parameters --------------------------------------------------------------

    def _doc(self):
        if self.ctl.doc is None:
            raise ControlError("no_document", "no document is open")

    @staticmethod
    def _path(params, required=True):
        path = params.get("path")
        if path is None and not required:
            return None
        if not isinstance(path, str) or not path:
            raise ControlError("invalid_param", "path must be a non-empty string")
        if "\0" in path or len(path) > MAX_PATH:
            raise ControlError("invalid_param", "path is not usable")
        if not os.path.isabs(path):
            raise ControlError("invalid_param", "path must be absolute: %s" % path)
        return path

    @staticmethod
    def _int(params, key, default=None, lo=None, hi=None):
        v = params.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ControlError("invalid_param", "%s must be an integer" % key)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ControlError("invalid_param", "%s must be within %s..%s" % (key, lo, hi))
        return v

    def _prepare(self, method, params):
        if not isinstance(params, dict):
            raise ControlError("bad_request", "params must be an object")
        return getattr(self, "_m_" + method)(params)

    # ---- read ------------------------------------------------------------------------

    def _m_status(self, params):
        def run():
            ctl = self.ctl
            out = dict(ctl.status())
            out["size"] = list(out["size"]) if out["size"] else None
            out.update({
                "path": ctl.path,
                "modified": bool(ctl.doc is not None and ctl.is_modified()),
                "documents": ctl.document_titles() if ctl.doc is not None else [],
                "active": ctl.current_doc,
                "tier": self.policy.tier,
                "roots": list(self.policy.roots),
            })
            return out
        return _Prepared(run)

    def _m_list_commands(self, params):
        return _Prepared(lambda: {"commands": scripting.describe(self.ctl)})

    def _m_list_effects(self, params):
        return _Prepared(lambda: {"effects": scripting.effect_names(self.ctl)})

    def _m_list_layers(self, params):
        self._doc()

        def run():
            active = self.ctl.active_layer
            layers = [{"index": i, "name": l.name, "kind": getattr(l, "kind", "raster"),
                       "visible": bool(l.visible), "opacity": int(getattr(l, "opacity", 255)),
                       "active": l is active}
                      for i, l in enumerate(self.ctl.doc.layers())]
            return {"layers": layers, "count": len(layers)}
        return _Prepared(run)

    def _m_get_pixel(self, params):
        self._doc()
        x = self._int(params, "x", lo=0, hi=self.ctl.doc.width - 1)
        y = self._int(params, "y", lo=0, hi=self.ctl.doc.height - 1)

        def run():
            r, g, b, a = (int(v) for v in self.ctl.merged()[y, x])
            return {"x": x, "y": y, "r": r, "g": g, "b": b, "a": a}
        return _Prepared(run)

    def _m_render(self, params):
        self._doc()
        size = self._int(params, "max_dim", 512, 16, MAX_RENDER)

        def run():
            from PIL import Image
            img = Image.fromarray(self.ctl.merged(), "RGBA")
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return {"mime": "image/png", "width": img.width, "height": img.height,
                    "base64": base64.b64encode(buf.getvalue()).decode("ascii")}
        return _Prepared(run)

    def _m_history(self, params):
        self._doc()
        return _Prepared(lambda: {"labels": self.ctl.history_labels(),
                                  "position": self.ctl.history.position})

    def _m_open(self, params):
        path = self._path(params)
        target = canonical(path)

        def run():
            if not os.path.isfile(target):
                raise ControlError("io_error", "no such file: %s" % path)
            try:
                self.ctl.open_path(target)
            except Exception as exc:                       # noqa: BLE001
                raise ControlError("io_error", str(exc)) from exc
            return self._m_status({}).run()
        return _Prepared(run, [(target, False)])

    # ---- edit -----------------------------------------------------------------------

    def _m_new(self, params):
        w = self._int(params, "width", None, 1, scripting.MAX_SIDE)
        h = self._int(params, "height", None, 1, scripting.MAX_SIDE)
        if w * h > scripting.MAX_PIXELS:
            raise ControlError("invalid_param", "at most %d pixels" % scripting.MAX_PIXELS)
        bg = params.get("background", "white")
        fills = {"white": (255, 255, 255, 255), "transparent": None}
        if bg not in fills:
            raise ControlError("invalid_param", "background is white or transparent")

        def run():
            self.ctl.new_document(w, h, background=fills[bg])
            return self._m_status({}).run()
        return _Prepared(run)

    def _m_run(self, params):
        self._doc()
        try:
            if "line" in params:
                if not isinstance(params["line"], str):
                    raise ControlError("invalid_param", "line must be a string")
                name, args, kwargs = scripting.parse(params["line"])
            else:
                name = params.get("command")
                args = params.get("args", [])
                kwargs = params.get("kwargs", {})
                if not isinstance(args, list) or not isinstance(kwargs, dict):
                    raise ControlError("invalid_param", "args is a list, kwargs an object")
            op, args, kwargs = scripting.prepare(self.ctl, name, args, kwargs)
            files = [(canonical(p), False)
                     for p in scripting.file_targets(self.ctl, op, args, kwargs)]
        except scripting.CommandError as exc:
            raise ControlError("invalid_param", str(exc)) from exc
        return _Prepared(lambda: scripting.invoke(self.ctl, name, op, args, kwargs), files)

    def _m_undo(self, params):
        self._doc()

        def run():
            self.ctl.undo()
            return {"position": self.ctl.history.position}
        return _Prepared(run)

    def _m_redo(self, params):
        self._doc()

        def run():
            self.ctl.redo()
            return {"position": self.ctl.history.position}
        return _Prepared(run)

    # ---- save ------------------------------------------------------------------------

    def _m_save(self, params):
        self._doc()
        path = self._path(params, required=False) or self.ctl.path
        if not path:
            raise ControlError("invalid_param", "the document has no path; give one")
        target = canonical(path)

        def run():
            try:
                self.ctl.save_path(target)
            except Exception as exc:                       # noqa: BLE001
                raise ControlError("io_error", str(exc)) from exc
            return {"path": target, "warnings": list(getattr(self.ctl, "last_save_warnings", []))}
        return _Prepared(run, [(target, True)])

    def _m_export(self, params):
        self._doc()
        target = canonical(self._path(params))

        def run():
            from ..engine.fileio import export_flat
            try:
                export_flat(self.ctl.doc, target)
            except Exception as exc:                       # noqa: BLE001
                raise ControlError("io_error", str(exc)) from exc
            return {"path": target}
        return _Prepared(run, [(target, True)])


# ---- the socket ---------------------------------------------------------------------

class _Client:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = bytearray()
        self.queued = 0
        self.alive = True


class ControlServer:
    """The Unix socket in front of a Gate. Single-threaded: poll() does
    whatever is ready and returns, so the headless loop and the window's
    event loop drive it the same way."""

    def __init__(self, gate, socket_path=None):
        self.gate = gate
        self.socket_path = os.path.abspath(socket_path or default_socket_path())
        self.token_path = token_path(self.socket_path)
        self._token = None
        self._listener = None
        self._lock_fd = -1
        self._token_written = False
        self._selector = selectors.DefaultSelector()
        self._clients = {}             # fd -> _Client
        self._queue = deque()          # (client, line)
        self._parked = False
        self._running = False
        # The endpoint's own files are off limits to every session.
        gate.policy.denied.extend(canonical(p) for p in (
            self.socket_path, self.token_path, lock_path(self.socket_path)))

    # ---- lifecycle -----------------------------------------------------------------

    def listen(self):
        """Bind, lock and write the token. Raises OSError with a reason."""
        path = self.socket_path
        if len(path.encode()) >= 104:
            raise OSError(errno.ENAMETOOLONG, "socket path too long (Unix sockets allow "
                          "about 100 bytes): %s" % path)
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            st = None
        # Never clobber anything that is not a stale socket of ours.
        if st is not None and not stat.S_ISSOCK(st.st_mode):
            raise OSError(errno.EEXIST, "%s exists and is not a socket" % path)
        self._lock()
        try:
            if st is not None:
                os.unlink(path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            old = os.umask(0o177)
            try:
                sock.bind(path)
            finally:
                os.umask(old)
            os.chmod(path, 0o600)
            sock.listen(MAX_CONNECTIONS)
            sock.setblocking(False)
            self._listener = sock
            self._selector.register(sock, selectors.EVENT_READ, None)
            self._write_token()
        except BaseException:
            self.close()
            raise
        return self

    def _lock(self):
        # flock, not a PID file: a dead holder releases it with its last fd.
        # The file stays; unlinking a locked file would let a second server
        # lock a fresh one beside it.
        fd = os.open(lock_path(self.socket_path),
                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise OSError(errno.EADDRINUSE, "another Ochre is serving %s" % self.socket_path)
        self._lock_fd = fd

    def _write_token(self):
        self._token = secrets.token_hex(32)
        path = self.token_path
        # Remove whatever is at the name (a stale token, or something planted),
        # then create fresh and owner-only; O_EXCL and O_NOFOLLOW refuse a
        # symlink raced into place.
        try:
            if stat.S_ISDIR(os.lstat(path).st_mode):
                raise OSError(errno.EISDIR, "%s is a directory" % path)
            os.unlink(path)
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600)
        try:
            os.write(fd, self._token.encode("ascii"))
        finally:
            os.close(fd)
        self._token_written = True

    def close(self):
        for client in list(self._clients.values()):
            self._drop(client)
        self._queue.clear()
        if self._token_written:
            try:
                os.unlink(self.token_path)
            except OSError:
                pass
            self._token_written = False
        if self._listener is not None:
            try:
                self._selector.unregister(self._listener)
            except (KeyError, ValueError):
                pass
            self._listener.close()
            self._listener = None
            # Only unlink what is still a socket: never a file someone put there.
            try:
                if stat.S_ISSOCK(os.lstat(self.socket_path).st_mode):
                    os.unlink(self.socket_path)
            except OSError:
                pass
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1

    def fds(self):
        """Every descriptor worth watching, for an event loop."""
        out = [self._listener.fileno()] if self._listener is not None else []
        return out + list(self._clients)

    # ---- the loop ----------------------------------------------------------------------

    def poll(self, timeout=0.0):
        """Accept, read and answer whatever is ready."""
        if self._listener is None:
            return
        for key, _mask in self._selector.select(timeout):
            if key.fileobj is self._listener:
                self._accept()
            else:
                client = self._clients.get(key.fd)
                if client is not None:
                    self._read(client)
        self._pump()

    def serve_forever(self, stop=lambda: False):
        while not stop():
            self.poll(0.25)

    def _accept(self):
        while True:
            try:
                sock, _ = self._listener.accept()
            except (BlockingIOError, InterruptedError):
                return
            if len(self._clients) >= MAX_CONNECTIONS or not self._same_user(sock):
                sock.close()
                continue
            sock.settimeout(SEND_TIMEOUT)
            client = _Client(sock)
            self._clients[sock.fileno()] = client
            self._selector.register(sock, selectors.EVENT_READ, None)

    @staticmethod
    def _same_user(sock):
        # The socket is 0600 already; this is the belt to those braces.
        peercred = getattr(socket, "SO_PEERCRED", None)
        if peercred is None:
            return True
        try:
            _pid, uid, _gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, peercred,
                                                                  struct.calcsize("3i")))
        except OSError:
            return False
        return uid == os.getuid()

    def _drop(self, client):
        if not client.alive:
            return
        client.alive = False
        self._clients.pop(client.sock.fileno(), None)
        try:
            self._selector.unregister(client.sock)
        except (KeyError, ValueError):
            pass
        client.sock.close()

    def _read(self, client):
        try:
            data = client.sock.recv(65536)
        except (BlockingIOError, InterruptedError, socket.timeout):
            return
        except OSError:
            data = b""
        if not data:
            self._drop(client)
            return
        client.buffer += data
        while client.alive:
            newline = client.buffer.find(b"\n")
            if newline < 0:
                # An unterminated request may not grow without bound.
                if len(client.buffer) > MAX_REQUEST_BYTES:
                    self._write(client, error("bad_request", "request exceeds %d bytes"
                                              % MAX_REQUEST_BYTES))
                    self._drop(client)
                return
            line = bytes(client.buffer[:newline])
            del client.buffer[:newline + 1]
            if not line.strip():
                continue
            if client.queued >= MAX_QUEUED:
                self._write(client, error("refused", "too many requests are waiting"))
                continue
            client.queued += 1
            self._queue.append((client, line))

    def _pump(self):
        if self._running:
            return
        self._running = True
        try:
            while self._queue and not self._parked:
                client, line = self._queue.popleft()
                client.queued -= 1
                if client.alive:
                    self._handle(client, line)
        finally:
            self._running = False

    def _handle(self, client, line):
        if len(line) > MAX_REQUEST_BYTES:
            self._write(client, error("bad_request", "request exceeds %d bytes" % MAX_REQUEST_BYTES))
            return
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            request = None
        if not isinstance(request, dict):
            self._write(client, error("bad_request", "request is not a JSON object"))
            return
        rid = request.get("id")
        if rid is not None and (isinstance(rid, bool) or not isinstance(rid, int)):
            self._write(client, error("bad_request", "id must be an integer"))
            return
        method, params = request.get("method"), request.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            self._write(client, error("bad_request", "method is a string, params an object"), rid)
            return
        if not self._authorized(request.get("token")):
            self._write(client, error("unauthorized", "missing or wrong session token"), rid)
            return
        answered = []

        def done(response):
            answered.append(True)
            self._write(client, response, rid)
            if self._parked:
                self._parked = False
                self._pump()

        if not self.gate.serve(method, params, done) and not answered:
            # Waiting for a person: nothing else runs until they answer.
            self._parked = True

    def _authorized(self, offered):
        if not isinstance(offered, str) or self._token is None:
            return False
        # Constant time: the comparison does not stop at the first wrong byte.
        return hmac.compare_digest(offered.encode("utf-8", "replace"), self._token.encode("ascii"))

    def _write(self, client, response, rid=None):
        if not client.alive:
            return
        if rid is not None:
            response = dict(response, id=rid)
        try:
            client.sock.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
        except OSError:
            self._drop(client)


# ---- a client ---------------------------------------------------------------------------

class ControlClient:
    """Talks to a ControlServer: what the MCP bridge uses with --connect."""

    def __init__(self, socket_path=None, token=None, token_file=None, timeout=120.0):
        self.socket_path = socket_path or default_socket_path()
        self.token_file = token_file or token_path(self.socket_path)
        self.token = token
        self.timeout = timeout
        self._sock = None
        self._file = None
        self._next = 0

    def connect(self):
        if self.token is None:
            try:
                with open(self.token_file, "r", encoding="ascii") as fh:
                    self.token = fh.read().strip()
            except OSError as exc:
                raise ControlError("unavailable", "cannot read the token file %s: %s"
                                   % (self.token_file, exc.strerror)) from exc
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise ControlError("unavailable", "cannot connect to %s: %s"
                               % (self.socket_path, exc.strerror or exc)) from exc
        self._sock = sock
        self._file = sock.makefile("rb")
        return self

    def call(self, method, params=None):
        """The result, or ControlError(code, message)."""
        if self._sock is None:
            self.connect()
        self._next += 1
        line = json.dumps({"id": self._next, "token": self.token, "method": method,
                           "params": params or {}}) + "\n"
        try:
            self._sock.sendall(line.encode("utf-8"))
            reply = self._file.readline(MAX_REQUEST_BYTES * 64)
        except OSError as exc:
            self.close()
            raise ControlError("unavailable", "lost the control server: %s" % exc) from exc
        if not reply:
            self.close()
            raise ControlError("unavailable", "the control server closed the connection")
        response = json.loads(reply)
        if not response.get("ok"):
            err = response.get("error") or {}
            raise ControlError(err.get("code", "failed"), err.get("message", "failed"))
        return response.get("result")

    def close(self):
        for thing in (self._file, self._sock):
            if thing is not None:
                try:
                    thing.close()
                except OSError:
                    pass
        self._file = self._sock = None

