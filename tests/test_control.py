#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The control server, over its real socket.

The headless half spawns `ochre --headless --control` as a subprocess and
speaks the wire protocol to it: tokens, tiers, folders (with ../ and
symlink escapes), limits, and that nothing listens on TCP. The window half
runs the real MainWindow offscreen with a client on another thread, and
answers the approval bar by clicking it. A last check starts the real
window process with --control and stops it with SIGTERM.

Every server started here is stopped by its PID.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_control.py
"""

import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


# Short: a Unix socket path must fit in about 100 bytes.
TMP = tempfile.mkdtemp(prefix="ochre-headless-", dir="/tmp")
ENV = dict(os.environ, XDG_STATE_HOME=os.path.join(TMP, "st"),
           XDG_CONFIG_HOME=os.path.join(TMP, "cf"), XDG_RUNTIME_DIR=os.path.join(TMP, "rt"),
           OCHRE_NO_PROMPT="1")
os.makedirs(ENV["XDG_RUNTIME_DIR"], mode=0o700)
PROCS = []


def path(*parts):
    return os.path.join(TMP, *parts)


def start(*args, sock=None, gui=False, wait=True):
    sock = sock or path("s")
    cmd = [sys.executable, "-m", "ochre.ui"] + ([] if gui else ["--headless"]) + \
        ["--control", "--control-socket", sock] + list(args)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    PROCS.append(proc)
    if wait:
        end = time.monotonic() + 60
        while not os.path.exists(sock + ".token"):
            if proc.poll() is not None or time.monotonic() > end:
                raise RuntimeError("server did not start: %s" % proc.stderr.read())
            time.sleep(0.05)
    return proc


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc.stderr:
        proc.stderr.close()


class Wire:
    """A raw protocol client: one JSON object per line."""

    def __init__(self, sock=None, token=None):
        self.path = sock or path("s")
        self.token = token if token is not None else open(self.path + ".token").read()
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(30)
        self.s.connect(self.path)
        self.f = self.s.makefile("rb")
        self.n = 0

    def send_raw(self, data):
        self.s.sendall(data)
        line = self.f.readline()
        return json.loads(line) if line else None

    def call(self, method, params=None, token=None):
        self.n += 1
        req = {"id": self.n, "token": self.token if token is None else token, "method": method,
               "params": params or {}}
        return self.send_raw((json.dumps(req) + "\n").encode())

    def close(self):
        self.f.close()
        self.s.close()


def code(reply):
    return reply["error"]["code"] if not reply.get("ok") else "ok"


def main():
    try:
        headless_checks()
        window_checks()
    finally:
        for proc in PROCS:
            stop(proc)
        shutil.rmtree(TMP, ignore_errors=True)


def headless_checks():
    root, out = path("root"), path("out")
    os.makedirs(root)
    os.makedirs(out)
    img = np.zeros((6, 8, 4), np.uint8)
    img[...] = (40, 80, 120, 255)
    Image.fromarray(img, "RGBA").save(os.path.join(root, "in.png"))
    Image.fromarray(img, "RGBA").save(os.path.join(out, "secret.png"))
    with open(os.path.join(root, "old.png"), "wb") as fh:
        fh.write(b"not mine")

    # ---- off by default -------------------------------------------------------------
    p = subprocess.run([sys.executable, "-m", "ochre.ui", "--headless", "--new", "4x4"],
                       cwd=ROOT, env=ENV, capture_output=True, timeout=60)
    check("a plain headless run serves nothing",
          p.returncode == 0 and os.listdir(ENV["XDG_RUNTIME_DIR"]) == [])

    # ---- the endpoint -------------------------------------------------------------------
    proc = start("--control-tier", "read", "--control-root", root,
                 os.path.join(root, "in.png"))
    tok = path("s.token")
    check("the token file is owner-only", stat.S_IMODE(os.stat(tok).st_mode) == 0o600,
          oct(os.stat(tok).st_mode))
    check("...and the socket", stat.S_IMODE(os.stat(path("s")).st_mode) == 0o600)
    token = open(tok).read()
    check("...a fresh 256-bit random token", len(token) == 64 and int(token, 16) > 0)

    inodes = set()
    for fd in os.listdir("/proc/%d/fd" % proc.pid):
        try:
            link = os.readlink("/proc/%d/fd/%s" % (proc.pid, fd))
        except OSError:
            continue
        if link.startswith("socket:["):
            inodes.add(link[8:-1])
    unix = open("/proc/net/unix").read()
    check("the server holds its Unix socket", any(" %s " % i in unix for i in inodes), str(inodes))
    inet = []
    for table in ("tcp", "tcp6", "udp", "udp6"):
        try:
            rows = open("/proc/net/" + table).read().splitlines()[1:]
        except OSError:
            continue
        inet += [r.split()[9] for r in rows if len(r.split()) > 9]
    check("...and nothing on TCP or UDP", not (inodes & set(inet)))

    # ---- the token ----------------------------------------------------------------------
    w = Wire()
    check("the right token is served", code(w.call("status")) == "ok")
    check("the opened file is there", w.call("status")["result"]["size"] == [8, 6])
    for label, bad in (("a wrong token", "0" * 64), ("a token one character off",
                                                     token[:-1] + ("0" if token[-1] != "0" else "1")),
                       ("a prefix of the token", token[:32]), ("an empty token", ""),
                       ("a token that is not a string", 12345)):
        check("%s is refused" % label, code(w.call("status", token=bad)) == "unauthorized")
    r = w.send_raw(b'{"id": 9, "method": "status"}\n')
    check("...as is no token at all", code(r) == "unauthorized" and r["id"] == 9)

    # ---- the read tier --------------------------------------------------------------------
    check("read: list_layers", code(w.call("list_layers")) == "ok")
    check("read: get_pixel", w.call("get_pixel", {"x": 1, "y": 1})["result"]["b"] == 120)
    r = w.call("render", {"max_dim": 16})
    check("read: render is a PNG", code(r) == "ok" and r["result"]["mime"] == "image/png")
    check("read cannot edit", code(w.call("run", {"line": "filter.invert"})) == "refused")
    r = w.call("export", {"path": os.path.join(root, "x.png")})
    check("read cannot write files (export)",
          code(r) == "refused" and not os.path.exists(os.path.join(root, "x.png")))
    r = w.call("save", {"path": os.path.join(root, "x.ochre")})
    check("...nor save", code(r) == "refused" and not os.path.exists(os.path.join(root, "x.ochre")))
    check("read may open inside its folder",
          code(w.call("open", {"path": os.path.join(root, "in.png")})) == "ok")
    check("...not outside it",
          code(w.call("open", {"path": os.path.join(out, "secret.png")})) == "refused")
    check("...nor by ../",
          code(w.call("open", {"path": root + "/../out/secret.png"})) == "refused")
    os.symlink(os.path.join(out, "secret.png"), os.path.join(root, "link.png"))
    check("...nor by a symlink inside the folder",
          code(w.call("open", {"path": os.path.join(root, "link.png")})) == "refused")
    check("a relative path is refused",
          code(w.call("open", {"path": "root/in.png"})) == "invalid_param")
    check("an unknown method is named as such", code(w.call("shutdown")) == "unknown_method")

    # ---- malformed requests ---------------------------------------------------------------
    check("not JSON", code(w.send_raw(b"{nope\n")) == "bad_request")
    check("not an object", code(w.send_raw(b"[1, 2]\n")) == "bad_request")
    check("an id that is not an integer",
          code(w.send_raw(b'{"id": "x", "method": "status"}\n')) == "bad_request")
    check("params that are not an object",
          code(w.send_raw(('{"id": 1, "token": "%s", "method": "status", "params": []}\n'
                           % token).encode())) == "bad_request")
    check("the connection survives all of that", code(w.call("status")) == "ok")

    # ---- size and connection limits --------------------------------------------------------
    big = Wire()
    big.s.sendall(b"x" * ((1 << 20) + 4096))
    r = json.loads(big.f.readline() or b"null")
    check("an oversized request is refused", r is not None and code(r) == "bad_request")
    check("...and its connection closed", big.f.readline() == b"")
    big.close()
    extra = [Wire() for _ in range(3)]          # w plus three: the limit of four
    fifth = Wire()
    try:
        turned_away = fifth.f.readline() == b""
    except ConnectionResetError:
        turned_away = True
    check("a fifth connection is turned away", turned_away)
    fifth.close()
    for e in extra:
        e.close()
    time.sleep(0.3)
    check("...and a slot frees when one leaves", code(Wire().call("status")) == "ok")

    # ---- a second server on the same socket --------------------------------------------------
    p = subprocess.run([sys.executable, "-m", "ochre.ui", "--headless", "--control",
                        "--control-socket", path("s")], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=60)
    check("a second server on a live socket refuses to start",
          p.returncode == 1 and "another Ochre" in p.stderr, p.stderr)
    check("...and the first carries on", code(w.call("status")) == "ok")
    w.close()

    stop(proc)
    check("SIGTERM removes the socket and the token",
          not os.path.exists(path("s")) and not os.path.exists(tok))

    # ---- edit and save tiers -------------------------------------------------------------------
    proc = start("--control-tier", "edit", "--control-root", root, "--new", "10x10")
    w = Wire()
    r = w.call("run", {"line": "image.resize width=5"})
    check("edit: run a command", code(r) == "ok" and r["result"]["changed"] is True)
    check("...it took", w.call("status")["result"]["size"] == [5, 5])
    r = w.call("run", {"command": "image.flip", "kwargs": {"horizontal": False}})
    check("edit: the structured form", code(r) == "ok")
    check("edit: undo", code(w.call("undo")) == "ok")
    check("edit: history", w.call("history")["result"]["labels"] != [])
    check("edit: an unknown command is invalid_param",
          code(w.call("run", {"line": "image.explode"})) == "invalid_param")
    check("edit: a non-recordable method is not reachable",
          code(w.call("run", {"command": "save_path", "args": ["/tmp/x.png"]})) == "invalid_param")
    check("edit: the size limit holds over the socket",
          code(w.call("new", {"width": 100000, "height": 100000})) == "invalid_param")
    check("edit still cannot write",
          code(w.call("export", {"path": os.path.join(root, "e.png")})) == "refused")
    w.close()
    stop(proc)

    proc = start("--control-tier", "save", "--control-root", root, "--new", "10x10")
    w = Wire()
    target = os.path.join(root, "made.png")
    check("save: export a new file inside the folder",
          code(w.call("export", {"path": target})) == "ok" and os.path.isfile(target))
    check("...and overwrite it, since the agent made it",
          code(w.call("export", {"path": target})) == "ok")
    r = w.call("export", {"path": os.path.join(root, "old.png")})
    check("save: an existing file it did not make is not overwritten",
          code(r) == "refused" and open(os.path.join(root, "old.png"), "rb").read() == b"not mine")
    check("save: nothing outside the folder",
          code(w.call("export", {"path": os.path.join(out, "x.png")})) == "refused"
          and not os.path.exists(os.path.join(out, "x.png")))
    check("save: nor through ../",
          code(w.call("export", {"path": root + "/../out/y.png"})) == "refused"
          and not os.path.exists(os.path.join(out, "y.png")))
    os.symlink(out, os.path.join(root, "outdir"))
    check("save: nor through a symlinked folder",
          code(w.call("export", {"path": os.path.join(root, "outdir", "z.png")})) == "refused"
          and not os.path.exists(os.path.join(out, "z.png")))
    os.symlink(os.path.join(out, "dangling.png"), os.path.join(root, "dangling.png"))
    check("save: nor through a dangling symlink",
          code(w.call("export", {"path": os.path.join(root, "dangling.png")})) == "refused"
          and not os.path.exists(os.path.join(out, "dangling.png")))
    r = w.call("save", {"path": os.path.join(root, "doc.ochre")})
    check("save: a document", code(r) == "ok" and os.path.isfile(os.path.join(root, "doc.ochre")))
    check("...and Save again goes to the same place", code(w.call("save")) == "ok")

    # Files an effect reads are judged like any other.
    cube = os.path.join(out, "look.cube")
    with open(cube, "w") as fh:
        fh.write("LUT_3D_SIZE 2\n" + "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n")
    shutil.copy(cube, os.path.join(root, "look.cube"))
    check("an effect may read a file inside the folder",
          code(w.call("run", {"line": "filter.color_lookup file=%s"
                              % os.path.join(root, "look.cube")})) == "ok")
    check("...not outside it",
          code(w.call("run", {"line": "filter.color_lookup file=%s" % cube})) == "refused")
    stack = json.dumps([{"name": "color_lookup", "params": {"file": cube}}])
    check("...nor from inside a Filter Gallery stack",
          code(w.call("run", {"command": "filter.filter_stack",
                              "kwargs": {"stack": stack}})) == "refused")
    w.close()
    stop(proc)

    # ---- the endpoint's own files and the state folder are never reachable ----------------------
    state = os.path.join(ENV["XDG_STATE_HOME"], "ochre")
    os.makedirs(state, exist_ok=True)
    Image.fromarray(img, "RGBA").save(os.path.join(state, "inside.png"))
    proc = start("--control-tier", "save", "--control-root", TMP, "--new", "4x4")
    w = Wire()
    check("the token file cannot be opened, even inside a folder",
          code(w.call("open", {"path": path("s.token")})) == "refused")
    check("...nor overwritten",
          code(w.call("export", {"path": path("s.token")})) == "refused"
          and open(path("s.token")).read() == w.token)
    check("the state folder cannot be opened, even inside a folder",
          code(w.call("open", {"path": os.path.join(state, "inside.png")})) == "refused")
    w.close()
    stop(proc)

    # ---- never clobber a file at the socket path ------------------------------------------------
    with open(path("f"), "w") as fh:
        fh.write("precious")
    p = subprocess.run([sys.executable, "-m", "ochre.ui", "--headless", "--control",
                        "--control-socket", path("f")], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=60)
    check("a file at the socket path is left alone",
          p.returncode == 1 and open(path("f")).read() == "precious", p.stderr)
    p = subprocess.run([sys.executable, "-m", "ochre.ui", "--headless", "--control",
                        "--control-tier", "admin"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=60)
    check("an unknown tier is a usage error", p.returncode == 2, p.stderr)


def window_checks():
    try:
        from PySide6.QtWidgets import QApplication, QDialog
    except ImportError:
        print("ok: skipped window checks (no PySide6)")
        return
    from ochre.engine.settings import Settings
    from ochre.ui import controlbar
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.controller import Controller
    from ochre.ui.mainwindow import MainWindow

    data = os.path.join(ROOT, "data")
    app = QApplication.instance() or QApplication([])
    state = path("wst")
    ctl = Controller(Settings(data), data, EventBus(), addon_dirs=[], state_dir=state)
    window = MainWindow(ctl, UiSettings(data))
    ctl.new_document(20, 10)
    window.canvas.bind()
    window.resize(900, 600)
    window.show()
    root = path("wroot")
    os.makedirs(root)
    sock = path("w")
    wc = controlbar.start(window, ctl, {"tier": "edit", "roots": [root], "socket": sock}, state)
    check("the window starts its control server", wc is not None and os.path.exists(sock))
    bar = wc.server.gate.host.bar

    def ask(method, params):
        """Send from another thread; the GUI thread keeps its loop running."""
        box = {}

        def worker():
            w = Wire(sock)
            box["reply"] = w.call(method, params)
            w.close()
        t = threading.Thread(target=worker)
        t.start()
        return t, box

    def spin(until, seconds=20):
        end = time.monotonic() + seconds
        while not until() and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        return until()

    t, box = ask("run", {"line": "image.resize width=10"})
    spin(lambda: "reply" in box)
    t.join()
    check("a command within the tier runs without asking",
          code(box["reply"]) == "ok" and bar.question() is None)
    check("...and the canvas follows it", window.canvas.surface.image.width() == 10)

    # Above the tier: the bar asks, Allow arms late, Allow runs it.
    target = os.path.join(root, "w.png")
    t, box = ask("export", {"path": target})
    check("a request above the tier puts up the bar", spin(lambda: bar.question() is not None))
    check("...saying what and where", "export" in bar.question() and "save tier" in bar.question(),
          bar.question())
    check("...with Allow not yet armed", not bar.allow.isEnabled())
    check("...and the request waiting", "reply" not in box and not os.path.exists(target))
    check("Allow arms after half a second", spin(lambda: bar.allow.isEnabled(), 5))
    bar.allow.click()
    spin(lambda: "reply" in box)
    t.join()
    check("...and Allow runs it", code(box["reply"]) == "ok" and os.path.isfile(target))

    t, box = ask("export", {"path": os.path.join(root, "n.png")})
    spin(lambda: bar.question() is not None)
    bar.deny.click()
    spin(lambda: "reply" in box)
    t.join()
    check("Deny refuses", code(box["reply"]) == "refused"
          and not os.path.exists(os.path.join(root, "n.png")))

    outside = path("wout.png")
    Image.fromarray(np.zeros((3, 3, 4), np.uint8), "RGBA").save(outside)
    t, box = ask("open", {"path": outside})
    check("a file outside the folders is asked about", spin(lambda: bar.question() is not None)
          and outside in bar.question(), str(bar.question()))
    bar.deny.click()
    spin(lambda: "reply" in box)
    t.join()
    check("...and stays shut on Deny", code(box["reply"]) == "refused")

    controlbar.TIMEOUT_MS = 300
    t, box = ask("export", {"path": os.path.join(root, "t.png")})
    spin(lambda: "reply" in box)
    t.join()
    check("no answer is a no", code(box["reply"]) == "approval_timeout" and bar.question() is None)
    controlbar.TIMEOUT_MS = 60000

    dialog = QDialog(window)
    dialog.show()
    t, box = ask("run", {"line": "filter.invert"})
    spin(lambda: "reply" in box)
    t.join()
    check("an edit waits while the person has a dialog open", code(box["reply"]) == "human_busy")
    t, box = ask("status", {})
    spin(lambda: "reply" in box)
    t.join()
    check("...a read does not", code(box["reply"]) == "ok")
    dialog.close()

    wc.close()
    check("closing removes the socket and token",
          not os.path.exists(sock) and not os.path.exists(sock + ".token"))
    # Not close(): the document is modified, and closing would ask.
    window.shutdown()
    window.hide()

    # ---- the real window process, with --control ---------------------------------------------
    src = path("gin.png")
    Image.fromarray(np.zeros((7, 9, 4), np.uint8), "RGBA").save(src)
    proc = start("--control-tier", "read", "--control-root", root, src, sock=path("g"), gui=True)
    w = Wire(path("g"))
    r = w.call("status")
    check("ochre --control serves from the window, and a folder flag is not opened as a file",
          code(r) == "ok" and r["result"]["size"] == [9, 7], str(r))
    w.close()
    stop(proc)
    check("...and SIGTERM closes it cleanly",
          proc.returncode == 0 and not os.path.exists(path("g.token")), str(proc.returncode))


if __name__ == "__main__":
    main()
