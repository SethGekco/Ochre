#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""`ochre --headless`: the real command line, run as a subprocess.

Every check here spawns the CLI the way a shell script would and looks at
its exit status, its messages and the files it wrote. One run happens with
a poisoned PySide6 first on the path, so importing Qt anywhere on the
headless path would fail it.

Run: python3 tests/test_headless.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


TMP = tempfile.mkdtemp(prefix="ochre-headless-")
ENV = dict(os.environ, XDG_STATE_HOME=os.path.join(TMP, "state"),
           XDG_CONFIG_HOME=os.path.join(TMP, "config"))
ENV.pop("QT_QPA_PLATFORM", None)


def ochre(*args, env=None, launcher=None):
    cmd = (launcher or [sys.executable, "-m", "ochre.ui"]) + ["--headless"] + list(args)
    p = subprocess.run(cmd, cwd=ROOT, env=env or ENV, capture_output=True, text=True,
                       timeout=120)
    return p.returncode, p.stdout, p.stderr


def tmp(name):
    return os.path.join(TMP, name)


def pixels(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def main():
    try:
        run_checks()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def run_checks():
    src = tmp("in.png")
    a = np.zeros((30, 40, 4), np.uint8)
    a[...] = (10, 200, 30, 255)
    a[:, 20:] = (200, 10, 30, 255)
    Image.fromarray(a, "RGBA").save(src)

    # ---- the example from the docs ----------------------------------------------
    rc, out, err = ochre(src, "--run", "image.resize width=20", "--export", tmp("out.png"))
    check("resize + export exits 0", rc == 0, err)
    px = pixels(tmp("out.png"))
    check("...writes the resized image (height follows the width)", px.shape == (15, 20, 4),
          str(px.shape))
    check("...with the picture in it", tuple(px[5, 2]) == (10, 200, 30, 255)
          and tuple(px[5, 17]) == (200, 10, 30, 255), str(px[5, 2]))
    check("...and says what it did", "ran image.resize" in out and "exported" in out, out)

    # ---- ordered steps, effects, the raw method names -------------------------------
    rc, out, err = ochre(src, "--run", "filter.invert", "--export", tmp("a.png"),
                         "--run", "flip_image false", "--run", "resize_image 4 3",
                         "--export", tmp("b.png"), "-q")
    check("steps run in order", rc == 0, err)
    check("...--quiet prints nothing", out == "", out)
    check("...the first export saw only the first step",
          tuple(pixels(tmp("a.png"))[0, 0]) == (245, 55, 225, 255)
          and pixels(tmp("a.png")).shape == (30, 40, 4))
    check("...the second saw them all", pixels(tmp("b.png")).shape == (3, 4, 4))

    rc, out, err = ochre("--new", "8x6", "--run", "color.primary #ff8000",
                         "--run", "select.all", "--run", "edit.fill", "--export", tmp("c.png"))
    check("--new, a #color argument and a fill",
          rc == 0 and tuple(pixels(tmp("c.png"))[3, 3]) == (255, 128, 0, 255), err)

    rc, out, err = ochre("--new", "8x6", "--run", "filter.levels gamma=2.0 input_high=200",
                         "--export", tmp("d.png"))
    check("effect parameters as name=value", rc == 0, err)

    # ---- documents round-trip --------------------------------------------------------
    rc, out, err = ochre(src, "--run", "layer.new name=Top", "--save", tmp("doc.ochre"))
    check("--save writes a document", rc == 0 and os.path.isfile(tmp("doc.ochre")), err)
    rc, out, err = ochre(tmp("doc.ochre"), "--run", "image.flatten", "--export", tmp("e.png"))
    check("...that opens again headless", rc == 0 and pixels(tmp("e.png")).shape == (30, 40, 4),
          err)

    # ---- failures: exit 1, one clear line, nothing written after ----------------------
    def fails(label, *args, says, code=1):
        if os.path.exists(tmp("never.png")):
            os.unlink(tmp("never.png"))
        rc, out, err = ochre(*(list(args) + ["--export", tmp("never.png")]))
        check(label, rc == code and says in err and not os.path.exists(tmp("never.png")),
              "rc=%d err=%r" % (rc, err))

    fails("an unknown command fails", src, "--run", "image.explode", says="unknown command")
    fails("an unknown effect fails", src, "--run", "filter.nope", says="unknown effect")
    fails("an unknown effect parameter fails", src, "--run", "filter.levels gamm=2",
          says="no parameter gamm")
    fails("a bad argument fails", src, "--run", "image.rotate quarter_turns=1 spin=2",
          says="image.rotate")
    fails("a missing input fails", tmp("nothing.png"), says="no such file")
    fails("a command with no document fails", "--run", "select.all", says="no document")
    fails("a layer object cannot be named from a script", src,
          "--run", "layer.flip horizontal=true layer=Background", says="active layer")
    fails("an oversized resize is refused before it allocates", src,
          "--run", "image.resize width=100000 height=100000", says="outside the limit")
    fails("a non-recordable controller method is not a command", src,
          "--run", "save_path /tmp/x.png", says="unknown command")
    fails("an unparseable command fails", src, "--run", "image.resize width='100",
          says="cannot parse")
    rc, out, err = ochre(src, "--export", tmp("nowhere") + "/sub/x.png")
    check("an unwritable export fails", rc == 1 and "cannot export" in err, err)

    rc, out, err = ochre("--frobnicate")
    check("a usage mistake exits 2 with the usage", rc == 2 and "usage:" in err, err)
    rc, out, err = ochre("--new", "big")
    check("...as does a bad --new", rc == 2 and "WxH" in err, err)
    rc, out, err = ochre()
    check("...and nothing to do", rc == 2, err)

    # ---- --list -------------------------------------------------------------------------
    rc, out, err = ochre("--list")
    check("--list names the commands and effects",
          rc == 0 and "image.resize" in out and "filter.gaussian_blur" in out, err)

    # ---- no Qt anywhere on the headless path ------------------------------------------
    poison = tmp("poison")
    os.makedirs(os.path.join(poison, "PySide6"))
    with open(os.path.join(poison, "PySide6", "__init__.py"), "w") as fh:
        fh.write("raise ImportError('the headless path imported Qt')\n")
    env = dict(ENV, PYTHONPATH=poison + os.pathsep + ENV.get("PYTHONPATH", ""))
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    rc, out, err = ochre(src, "--run", "filter.invert", "--export", tmp("noqt.png"), env=env)
    check("runs with Qt unimportable and no display", rc == 0, err)

    # ---- no session state written -----------------------------------------------------
    check("a headless run leaves the user's session alone (no session.ini, no recovery)",
          not os.path.exists(os.path.join(TMP, "state", "ochre", "session.ini"))
          and not os.path.exists(os.path.join(TMP, "state", "ochre", "recovery")),
          str(os.listdir(os.path.join(TMP, "state", "ochre"))
              if os.path.isdir(os.path.join(TMP, "state", "ochre")) else ""))

    # ---- through run.sh, output reaches the caller --------------------------------------
    # run.sh sends a non-terminal's output to a log; a windowless mode's
    # output is its answer, so that must not happen to it.
    rc, out, err = ochre("--list", launcher=[os.path.join(ROOT, "run.sh")])
    check("run.sh --headless prints to the caller, not the log",
          rc == 0 and "image.resize" in out, err)

    # ---- the command surface IS the recordable list ------------------------------------
    from ochre.engine.props import TextProperty
    from ochre.engine.settings import Settings
    from ochre.ui import scripting
    from ochre.ui.controller import RECORDABLE, Controller
    targets = list(scripting.ALIASES.values())
    check("every alias names a recordable command",
          all(op in RECORDABLE for op in targets), str(set(targets) - set(RECORDABLE)))
    check("every recordable command has exactly one alias",
          sorted(targets) == sorted(op for op in RECORDABLE if hasattr(Controller, op)))

    data = os.path.join(ROOT, "data")
    ctl = Controller(Settings(data), data, addon_dirs=[], state_dir=tmp("state2"))
    unknown = []
    for name in ctl.effects.names():
        for prop in ctl.effects.create(name).props:
            if isinstance(prop, TextProperty) and prop.name not in (
                    scripting.FILE_PROPS | scripting.NOT_FILES):
                unknown.append("%s.%s" % (name, prop.name))
    check("every text parameter of every effect is known to name a file or not",
          not unknown, str(unknown))


if __name__ == "__main__":
    main()
