#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Atomic saves: a save that fails halfway leaves the previous file exactly
as it was, and leaves no temporary file behind.

Every check breaks a REAL save -- Controller.save_path, the function File >
Save and Save As call, or the palette writer the Palette bar calls -- rather
than the helper underneath, because a helper nobody routes through protects
nothing. Two ways to break one:

  * an encoder that raises after it has started writing (the zip containers
    have their manifest out, Pillow has written its first bytes), and
  * fsync failing -- what a full or vanishing disk looks like -- after the
    whole new file has been written, the last moment a save can fail.

Run: python3 tests/test_atomicsave.py
"""

import errno
import os
import re
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import ImageFile

from ochre.engine import atomic
from ochre.engine import palettefiles as PF
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ADDONS = os.path.join(ROOT, "addons")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def controller(state):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    for ident in list(ctl.addons.addons):
        ctl.addons.trust(ident)
    ctl.load_addons(all_addons=True)
    return ctl


def paint(ctl, colour):
    """Change the picture, so a new save would write different bytes."""
    lay = ctl.active_layer
    cell = ctl.doc.cell(lay)
    cell.plane("rgba")[4:20, 4:28] = colour
    cell._touched(ctl.doc.bounds)
    ctl.compositor.invalidate()


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


class Patch:
    """Swap an attribute for the length of a with-block."""

    def __init__(self, owner, name, value):
        self.owner, self.name, self.value = owner, name, value

    def __enter__(self):
        self.old = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.owner, self.name, self.old)
        return False


def no_space(_fd):
    raise OSError(errno.ENOSPC, "No space left on device (simulated)")


def failing_writestr():
    """ZipFile.writestr that writes the first entry and raises on the next:
    the container is open and half-written when the save dies."""
    real = zipfile.ZipFile.writestr
    calls = [0]

    def writestr(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] > 1:
            raise RuntimeError("encoder failed mid-save (simulated)")
        return real(self, *args, **kwargs)
    return writestr


def failing_pillow_save():
    """Pillow's tile writer, emitting some bytes and then raising -- the
    shape of an encoder dying halfway through an image."""
    def _save(im, fp, tile, bufsize=0):
        fp.write(b"\x89PNG partial garbage")
        raise RuntimeError("image encoder failed mid-save (simulated)")
    return _save


def save_fails_cleanly(label, ctl, path, breaker, **save_kw):
    """Save once for real, change the picture, save again under `breaker`:
    the second save must fail, the first file must survive byte for byte,
    and the folder must hold nothing but that file."""
    paint(ctl, (200, 30, 30, 255))
    ctl.save_path(path, **save_kw)
    before = read(path)
    folder = os.path.dirname(path)
    paint(ctl, (30, 30, 200, 255))
    raised = None
    with breaker:
        try:
            ctl.save_path(path, **save_kw)
        except Exception as exc:                            # noqa: BLE001
            raised = exc
    check("%s: the failed save reports an error" % label, raised is not None)
    check("%s: ...and the previous file is byte-identical" % label, read(path) == before,
          "(%d bytes, was %d)" % (len(read(path)), len(before)))
    check("%s: ...and no temporary file is left behind" % label,
          os.listdir(folder) == [os.path.basename(path)], str(os.listdir(folder)))
    ctl.save_path(path, **save_kw)
    check("%s: ...and the next save succeeds and changes the file" % label,
          read(path) != before)


def main():
    state = tempfile.mkdtemp(prefix="ochre-fileio-atomic-")
    ctl = controller(state)
    ctl.new_document(32, 24, background=None)

    def fresh(name):
        return os.path.join(tempfile.mkdtemp(prefix="ochre-fileio-atomic-"), name)

    # ---- an encoder raising partway through --------------------------------------
    save_fails_cleanly(".ochre, encoder raising inside the zip", ctl, fresh("doc.ochre"),
                       Patch(zipfile.ZipFile, "writestr", failing_writestr()))
    save_fails_cleanly(".ora, encoder raising inside the zip", ctl, fresh("doc.ora"),
                       Patch(zipfile.ZipFile, "writestr", failing_writestr()))
    save_fails_cleanly(".png, Pillow's encoder raising after its first bytes", ctl,
                       fresh("flat.png"), Patch(ImageFile, "_save", failing_pillow_save()))
    save_fails_cleanly(".jpg, Pillow's encoder raising after its first bytes", ctl,
                       fresh("flat.jpg"), Patch(ImageFile, "_save", failing_pillow_save()))

    # ---- the disk failing after every byte is written -----------------------------
    for name in ("doc.ochre", "doc.ora", "doc.psd", "flat.png", "flat.webp", "flat.tif"):
        save_fails_cleanly("%s, fsync failing (disk full)" % os.path.splitext(name)[1], ctl,
                           fresh(name), Patch(os, "fsync", no_space))
    # Through the C&C addon's own writer -- the host routes addon saves, and
    # an addon saver is exactly where a raw open(path, "wb") would hide.
    if "cnc.shp" in [n for _l, n, _e, _i in ctl.addon_formats(writing=True)]:
        ctl.new_document(32, 24, background=None)
        convert = {"palette": PF.builtin("PICO-8 (16)"), "dither": "none"}
        save_fails_cleanly("addon sprite format, fsync failing", ctl, fresh("unit.shp"),
                           Patch(os, "fsync", no_space), format="cnc.shp", convert=convert)
    else:
        print("note: C&C addon not present; addon saver not exercised")

    # ---- palette files: the Palette bar's Save Palette ----------------------------
    pal_path = fresh("colours.gpl")
    PF.write(PF.builtin("PICO-8 (16)"), pal_path)
    before = read(pal_path)
    with Patch(os, "fsync", no_space):
        try:
            PF.write(PF.builtin("Game Boy (4)"), pal_path)
            raised = False
        except OSError:
            raised = True
    check("palette file: a failed write raises and keeps the old palette byte-identical",
          raised and read(pal_path) == before
          and os.listdir(os.path.dirname(pal_path)) == ["colours.gpl"])

    # ---- what the file looks like afterwards --------------------------------------
    ctl.new_document(32, 24, background=None)
    path = fresh("kept.ochre")
    ctl.save_path(path)
    os.chmod(path, 0o640)
    paint(ctl, (10, 200, 10, 255))
    ctl.save_path(path)
    check("saving over a file keeps its permission bits (0640 stays 0640)",
          stat.S_IMODE(os.stat(path).st_mode) == 0o640, oct(os.stat(path).st_mode))
    newfile = fresh("new.png")
    ctl.save_path(newfile)
    umask = atomic._umask()
    check("a new file gets the mode open() would give it, not mkstemp's 0600",
          stat.S_IMODE(os.stat(newfile).st_mode) == 0o666 & ~umask,
          "%s umask %s" % (oct(os.stat(newfile).st_mode), oct(umask)))
    real = fresh("real.ora")
    ctl.save_path(real)
    link = os.path.join(os.path.dirname(real), "link.ora")
    os.symlink(real, link)
    paint(ctl, (90, 90, 90, 255))
    before = read(real)
    ctl.save_path(link)
    check("saving through a symlink replaces the file it points at; the link stays",
          os.path.islink(link) and read(real) != before and read(link) == read(real))

    # ---- nothing writes around the helper -----------------------------------------
    # open(..., "w"/"wb"/"a") or ZipFile(path, "w") anywhere in the editor or
    # its addons is a save that truncates first. The helper itself is the one
    # place allowed to open a file for writing (via os.fdopen); ZipFile(fh,
    # "w") writes into a file object the helper handed out.
    pattern = re.compile(r"""\bopen\([^)]*,\s*["'][wax]b?\+?["']|ZipFile\([^)]*,\s*["'][wa]["']""")
    hits = []
    for base in (os.path.join(ROOT, "ochre"), ADDONS):
        for folder, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(folder, fn)
                with open(full, encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        code = line.split("#", 1)[0]
                        if pattern.search(code) and "ZipFile(fh," not in code:
                            hits.append("%s:%d" % (os.path.relpath(full, ROOT), n))
    check("no writer in ochre/ or addons/ opens its target for writing directly",
          not hits, str(hits[:5]))

    print("\nall atomic-save checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
