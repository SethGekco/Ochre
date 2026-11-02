#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""`ochre --mcp`: real MCP JSON-RPC over the real process's stdin/stdout.

Two ways in: a private headless Ochre (the bridge runs it in-process), and
--connect to a running control server. In both the server side refuses
what the session may not do; the bridge adds nothing and removes nothing.
Every server and bridge started here is stopped by its PID.

Run: python3 tests/test_mcp.py
"""

import base64
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


TMP = tempfile.mkdtemp(prefix="ochre-headless-", dir="/tmp")
ENV = dict(os.environ, XDG_STATE_HOME=os.path.join(TMP, "st"),
           XDG_CONFIG_HOME=os.path.join(TMP, "cf"), XDG_RUNTIME_DIR=os.path.join(TMP, "rt"),
           OCHRE_NO_PROMPT="1")
os.makedirs(ENV["XDG_RUNTIME_DIR"], mode=0o700)
PROCS = []


def path(*parts):
    return os.path.join(TMP, *parts)


class Mcp:
    """A client speaking newline-delimited JSON-RPC to `ochre --mcp`."""

    def __init__(self, *args, env=None):
        self.proc = subprocess.Popen([sys.executable, "-m", "ochre.ui", "--mcp"] + list(args),
                                     cwd=ROOT, env=env or ENV, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        PROCS.append(self.proc)
        self.n = 0
        self.lines = []          # every stdout line, to check it is all protocol

    def send(self, obj):
        data = obj if isinstance(obj, bytes) else (json.dumps(obj) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def read(self, timeout=60):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise RuntimeError("no reply in %ds" % timeout)
        line = self.proc.stdout.readline()
        self.lines.append(line)
        return json.loads(line)

    def request(self, method, params=None):
        self.n += 1
        self.send({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}})
        return self.read()

    def tool(self, name, **arguments):
        r = self.request("tools/call", {"name": name, "arguments": arguments})
        return r["result"]

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        err = self.proc.stderr.read().decode()
        self.proc.stdout.close()
        self.proc.stderr.close()
        return self.proc.returncode, err


def text(result):
    return result["content"][-1]["text"]


def main():
    try:
        run_checks()
    finally:
        for proc in PROCS:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        shutil.rmtree(TMP, ignore_errors=True)


def run_checks():
    root, out = path("root"), path("out")
    os.makedirs(root)
    os.makedirs(out)
    img = np.zeros((10, 20, 4), np.uint8)
    img[...] = (30, 60, 90, 255)
    Image.fromarray(img, "RGBA").save(os.path.join(root, "in.png"))

    # ---- a private headless Ochre, tier save --------------------------------------------
    m = Mcp("--control-tier", "save", "--control-root", root)
    r = m.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "test", "version": "1"}})
    check("initialize answers with the asked version",
          r["result"]["protocolVersion"] == "2025-03-26" and "tools" in r["result"]["capabilities"]
          and r["result"]["serverInfo"]["name"] == "ochre", str(r))
    check("...and tells the agent that results are data",
          "untrusted" in r["result"]["instructions"])
    m.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = m.request("ping")
    check("a notification gets no reply (the next reply is the ping's)",
          r["id"] == m.n and r["result"] == {})
    tools = {t["name"]: t for t in m.request("tools/list")["result"]["tools"]}
    check("tools/list lists the operations with schemas",
          {"ochre_run", "ochre_open", "ochre_export", "ochre_render"} <= set(tools)
          and tools["ochre_open"]["inputSchema"]["required"] == ["path"])

    res = m.tool("ochre_open", path=os.path.join(root, "in.png"))
    check("open a file in the folder", not res.get("isError")
          and json.loads(text(res))["size"] == [20, 10], str(res))
    res = m.tool("ochre_run", line="image.resize width=10")
    check("run a command", not res.get("isError") and json.loads(text(res))["changed"], str(res))
    res = m.tool("ochre_run", command="filter.invert")
    check("...in the structured form too", not res.get("isError"), str(res))
    res = m.tool("ochre_get_pixel", x=2, y=2)
    check("read a pixel back", json.loads(text(res))["r"] == 225, str(res))
    res = m.tool("ochre_render", max_dim=64)
    png = base64.b64decode(res["content"][0]["data"])
    check("render is image content",
          res["content"][0]["type"] == "image" and png[:8] == b"\x89PNG\r\n\x1a\n"
          and "base64" not in text(res))
    target = os.path.join(root, "agent.png")
    res = m.tool("ochre_export", path=target)
    check("export inside the folder", not res.get("isError") and os.path.isfile(target), str(res))
    check("...the result as edited", Image.open(target).size == (10, 5))
    res = m.tool("ochre_export", path=os.path.join(out, "x.png"))
    check("export outside the folder is a refused tool call",
          res.get("isError") and text(res).startswith("refused")
          and not os.path.exists(os.path.join(out, "x.png")), str(res))
    res = m.tool("ochre_export", path=root + "/../out/y.png")
    check("...also by ../", res.get("isError") and not os.path.exists(os.path.join(out, "y.png")))
    res = m.tool("ochre_run", line="image.explode")
    check("a bad command is a tool error, not a protocol error",
          res.get("isError") and "unknown command" in text(res))

    r = m.request("tools/call", {"name": "ochre_shutdown", "arguments": {}})
    check("an unknown tool is invalid params", r["error"]["code"] == -32602)
    r = m.request("resources/list")
    check("an unknown method is method-not-found", r["error"]["code"] == -32601 and r["id"] == m.n)
    m.send(b"{nope\n")
    check("bad JSON is a parse error", m.read()["error"]["code"] == -32700)
    m.send(b"x" * ((1 << 20) + 10) + b"\n")
    check("an oversized line is refused", m.read()["error"]["code"] == -32600)
    check("...and the bridge carries on", m.request("ping")["result"] == {})
    rc, err = m.close()
    check("stdin closing ends it cleanly", rc == 0, err)
    check("stdout carried nothing but protocol",
          all(json.loads(l).get("jsonrpc") == "2.0" for l in m.lines))

    # ---- the default tier is read ---------------------------------------------------------
    m = Mcp("--control-root", root, os.path.join(root, "in.png"))
    m.request("initialize", {"protocolVersion": "1999-01-01"})
    res = m.tool("ochre_status")
    check("by default the tier is read", json.loads(text(res))["tier"] == "read")
    res = m.tool("ochre_run", line="filter.invert")
    check("...which cannot edit", res.get("isError") and "tier" in text(res), str(res))
    res = m.tool("ochre_export", path=os.path.join(root, "r.png"))
    check("...nor write files", res.get("isError")
          and not os.path.exists(os.path.join(root, "r.png")))
    m.close()

    # ---- bridging to a control server ---------------------------------------------------------
    sock = path("s")
    server = subprocess.Popen([sys.executable, "-m", "ochre.ui", "--headless", "--new", "8x8",
                               "--control", "--control-tier", "edit", "--control-root", root,
                               "--control-socket", sock], cwd=ROOT, env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    PROCS.append(server)
    end = time.monotonic() + 60
    while not os.path.exists(sock + ".token") and time.monotonic() < end:
        time.sleep(0.05)
    m = Mcp("--connect", sock)
    m.request("initialize", {"protocolVersion": "2024-11-05"})
    res = m.tool("ochre_status")
    check("--connect reaches the server", json.loads(text(res))["size"] == [8, 8], str(res))
    res = m.tool("ochre_run", line="image.resize width=4")
    check("...runs within its tier", not res.get("isError"))
    res = m.tool("ochre_export", path=os.path.join(root, "c.png"))
    check("...and the server, not the bridge, refuses beyond it",
          res.get("isError") and "tier" in text(res)
          and not os.path.exists(os.path.join(root, "c.png")), str(res))
    m.close()

    bad = path("bad.token")
    with open(bad, "w") as fh:
        fh.write("0" * 64)
    m = Mcp("--connect", sock, "--token-file", bad)
    m.request("initialize", {})
    res = m.tool("ochre_status")
    check("a bridge with the wrong token gets nothing",
          res.get("isError") and text(res).startswith("unauthorized"), str(res))
    m.close()
    server.send_signal(signal.SIGTERM)
    server.wait(20)

    m = Mcp("--connect", path("nothing"))
    rc, err = m.close()
    check("--connect with no server exits 1 and says why", rc == 1 and "cannot" in err, err)
    m = Mcp("--connect", sock, "--control-tier", "save")
    rc, err = m.close()
    check("--connect does not take a tier (the server decides)", rc == 2, err)

    # ---- no Qt here either -------------------------------------------------------------------
    poison = path("poison")
    os.makedirs(os.path.join(poison, "PySide6"))
    with open(os.path.join(poison, "PySide6", "__init__.py"), "w") as fh:
        fh.write("raise ImportError('the MCP path imported Qt')\n")
    env = dict(ENV, PYTHONPATH=poison)
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    m = Mcp("--control-tier", "edit", env=env)
    m.request("initialize", {})
    res = m.tool("ochre_new", width=4, height=4)
    check("--mcp runs with Qt unimportable and no display", not res.get("isError"), str(res))
    m.close()

    # ---- through run.sh, the protocol reaches the caller -------------------------------------
    p = subprocess.run([os.path.join(ROOT, "run.sh"), "--mcp"], cwd=ROOT, env=ENV, timeout=60,
                       input=b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n', capture_output=True)
    check("run.sh --mcp answers on stdout, not into the log",
          json.loads(p.stdout.splitlines()[0])["id"] == 1 if p.stdout else False, p.stderr)


if __name__ == "__main__":
    main()
