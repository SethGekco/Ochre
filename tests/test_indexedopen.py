#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""A paletted PNG or GIF opens index-locked, and Image > Convert Layer to RGB
lets it go.

Files are opened the way File > Open does (MainWindow.open_file), so the
test covers the importer the controller really picks. The centre of it is
exactness: two palette entries with the same colour must still be two
indices after opening, so a PNG-8 exported from Ochre comes back with the
very indices it was written with.

The files the palette model cannot hold -- partial PNG-8 alpha, several
transparent entries, an animated GIF -- must open in full colour, unchanged,
and every frame of the animation must arrive.

Also checks the shortcuts: Ctrl+Y is Redo, Ctrl+Shift+Z is Proof Colors.

Skips cleanly when PySide6 is absent.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_indexedopen.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

import numpy as np
from PIL import Image

from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow
from ochre.ui.prefsdialog import menu_actions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def paletted(idx, colours):
    """A mode-P image of `idx` on `colours` (a list of RGB triples)."""
    img = Image.fromarray(idx.astype(np.uint8), "P")
    flat = [c for rgb in colours for c in rgb]
    img.putpalette(flat + [0] * (768 - len(flat)))
    return img


def main():
    app = QApplication.instance() or QApplication([])
    state = tempfile.mkdtemp(prefix="ochre-idxopen-")
    files = tempfile.mkdtemp(prefix="ochre-idxopen-files-")
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    win.show()
    app.processEvents()
    actions = dict(menu_actions(win.menuBar()))

    check("Ctrl+Y is Redo", actions["Edit/Redo"].shortcut().toString() == "Ctrl+Y")
    check("Ctrl+Shift+Z is Proof Colors",
          actions["View/Proof Colors"].shortcut().toString() == "Ctrl+Shift+Z")

    rng = np.random.default_rng(11)
    colours = [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(16)]
    colours[5] = colours[6] = (90, 140, 30)               # one colour, two entries
    idx = rng.integers(0, 16, (20, 30)).astype(np.uint8)
    idx[:, :8] = 5
    idx[:, 8:16] = 6

    def layer_of(doc):
        return [l for l in doc.layers() if not l.is_group][0]

    # ---- PNG-8 with one transparent entry ---------------------------------
    png8 = os.path.join(files, "sprite.png")
    img = paletted(idx, colours)
    img.save(png8, transparency=0)
    check("File > Open accepted the PNG-8", win.open_file(png8) is not False)
    lay = layer_of(ctl.doc)
    check("a PNG-8 opens index-locked", lay.index_locked)
    got = ctl.doc.cell(lay).plane("index")
    check("...with every index exact, duplicate colours kept apart",
          np.array_equal(got, idx), "differs at %d px" % int((got != idx).sum()))
    check("...on its own palette", ctl.doc.palette is not None
          and tuple(ctl.doc.palette.entries[6, :3]) == colours[6])
    check("...and index 0 transparent", ctl.doc.palette.transparent == 0)

    # ---- Image > Convert Layer to RGB, then undo -----------------------------
    before = ctl.merged().copy()
    actions["Image/Convert Layer to RGB"].trigger()
    app.processEvents()
    lay = layer_of(ctl.doc)
    check("Convert Layer to RGB releases the layer", not lay.index_locked)
    check("...with no index plane left", "index" not in lay.planes)
    check("...and every pixel looking exactly as before",
          np.array_equal(ctl.merged(), before))
    check("...transparent pixels stay transparent",
          bool((ctl.merged()[idx == 0][:, 3] == 0).all()))
    actions["Edit/Undo"].trigger()
    app.processEvents()
    lay = layer_of(ctl.doc)
    check("Undo makes it indexed again", lay.index_locked
          and np.array_equal(ctl.doc.cell(lay).plane("index"), idx))
    actions["Edit/Redo"].trigger()
    app.processEvents()
    check("Redo releases it again", not layer_of(ctl.doc).index_locked)

    # ---- files the palette model cannot hold open in full colour --------------
    partial = os.path.join(files, "partial.png")
    alphas = bytes([0, 128] + [255] * 14)
    paletted(idx, colours).save(partial, transparency=alphas)
    win.open_file(partial)
    lay = layer_of(ctl.doc)
    check("PNG-8 with partial alpha opens in full colour", not lay.index_locked)
    rgba = ctl.doc.cell(lay).plane("rgba")
    check("...keeping the partial alpha", bool((rgba[idx == 1][:, 3] == 128).all()))

    two_clear = os.path.join(files, "twoclear.png")
    paletted(idx, colours).save(two_clear, transparency=bytes([0, 0] + [255] * 14))
    win.open_file(two_clear)
    check("PNG-8 with two transparent entries opens in full colour",
          not layer_of(ctl.doc).index_locked)

    anim = os.path.join(files, "anim.gif")
    frames = [paletted((idx + k) % 16, colours) for k in range(3)]
    frames[0].save(anim, save_all=True, append_images=frames[1:], duration=50, loop=0)
    win.open_file(anim)
    check("an animated GIF opens in full colour", not layer_of(ctl.doc).index_locked)
    check("...with all of its frames", len(ctl.doc.frames) == 3, "got %d" % len(ctl.doc.frames))

    still = os.path.join(files, "still.gif")
    paletted(idx, colours).save(still)
    win.open_file(still)
    lay = layer_of(ctl.doc)
    check("a single-frame GIF opens index-locked", lay.index_locked)

    win.shutdown()
    print("all indexed-open checks passed")


if __name__ == "__main__":
    main()
