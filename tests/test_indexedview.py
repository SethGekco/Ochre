#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Opening an indexed file that carries no palette: indices are indices,
kept exactly; the user picks the palette to view them through, and the
choice is remembered for the next one.

Run: python3 tests/test_indexedview.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

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


def ramp(name, colour):
    """A 256-entry palette: index i = colour scaled by i/255."""
    c = np.array(colour, np.float64)
    return PF.make([tuple(int(round(v)) for v in c * i / 255.0) for i in range(256)], name)


def main():
    d = tempfile.mkdtemp()
    folder = os.path.join(d, "pals")
    os.makedirs(os.path.join(folder, "Game"))
    PF.write(ramp("aaa", (255, 0, 0)), os.path.join(folder, "Game", "aaa.pal"))
    PF.write(ramp("unittem", (0, 255, 0)), os.path.join(folder, "Game", "unittem.pal"))
    PF.write(ramp("isotem", (0, 0, 255)), os.path.join(folder, "Game", "isotem.pal"))
    state = tempfile.mkdtemp()
    Settings.save_user(os.path.join(state, "preferences.ini"), {("Palette", "Folders"): folder})

    def fresh():
        ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
        for ident in ctl.addons.addons:
            ctl.addons.trust(ident)
        ctl.load_addons(all_addons=True)
        return ctl

    # Write an indexed sprite whose indices are known.
    ctl = fresh()
    ctl.new_document(32, 24, background=None)
    pal = PF.builtin("Grayscale (256)")
    lay = ctl.active_layer
    px = ctl.doc.cell(lay).plane("rgba")
    px[4:20, 4:28] = (120, 120, 120, 255)
    px[8:16, 8:24] = (200, 200, 200, 255)
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    path = os.path.join(d, "unittem_tank.shp")
    ctl.save_path(path, format="cnc.shp", convert={"palette": pal})

    ctl = fresh()
    ctl.open_path(path)
    il = [l for l in ctl.doc.layers() if l.index_locked][0]
    idx = ctl.doc.cell(il).plane("index").copy()
    check("an indexed file opens with its indices exactly (120 and 200 kept as indices)",
          int(idx[6, 6]) == 120 and int(idx[12, 12]) == 200 and int(idx[0, 0]) == 0)
    check("the format's guess matches the palette's own name, folder or not",
          ctl.doc.palette.name.split("/")[-1] == "unittem", ctl.doc.palette.name)
    check("...and the editor says what it did instead of asking anything",
          ctl.indexed_note and "indices" in ctl.indexed_note and ctl.swatch_key == ("document",))

    iso = [k for _g, n, k in ctl.palette_sources() if n.endswith("isotem")][0]
    pos = ctl.history.position
    ctl.set_swatch_palette(iso)
    ctl.set_document_palette(ctl.swatch_palette, iso)
    shown = ctl.compositor.composite(ctl.doc.bounds)
    check("choosing a palette re-colours the image through it...",
          tuple(shown[12, 12, :3]) == tuple(ramp("x", (0, 0, 255)).entries[200, :3]))
    check("...without changing a single index", np.array_equal(ctl.doc.cell(il).plane("index"), idx))
    check("...as one undo step", ctl.history.position == pos + 1)
    ctl.undo()
    check("...undoable back to the guess", ctl.doc.palette.name.split("/")[-1] == "unittem"
          and np.array_equal(ctl.doc.cell(il).plane("index"), idx))
    ctl.redo()

    other = os.path.join(d, "something.shp")
    ctl.save_path(other)                      # an indexed document saves directly
    ctl2 = fresh()
    ctl2.open_path(other)
    ctl3b = fresh()
    ctl3b.open_path(path)
    check("...but a file whose name names a palette still opens with that one",
          ctl3b.doc.palette.name.split("/")[-1] == "unittem")
    check("the next indexed file opens with the palette chosen last time",
          ctl2.doc.palette.name.split("/")[-1] == "isotem" and ctl2.swatch_key == iso,
          "%s %s" % (ctl2.doc.palette.name, ctl2.swatch_key))

    # ---- the Palette bar does it ---------------------------------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl3 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[ADDONS], state_dir=state)
    for ident in ctl3.addons.addons:
        ctl3.addons.trust(ident)
    ctl3.load_addons(all_addons=True)
    win = MainWindow(ctl3, UiSettings(DATA))
    win.open_file(path)
    bar = win.palette_bar
    before = ctl3.doc.cell([l for l in ctl3.doc.layers() if l.index_locked][0]).plane("index").copy()
    aaa = bar._keys.index([k for k in bar._keys if k and k[0] in ("file", "provider") and
                           str(k[-1]).endswith("aaa" + (".pal" if k[0] == "file" else ""))][0])
    bar._chosen(aaa)
    lay3 = [l for l in ctl3.doc.layers() if l.index_locked][0]
    check("picking a palette on the bar shows the indexed image through it, indices untouched",
          ctl3.doc.palette.name.split("/")[-1] == "aaa"
          and np.array_equal(ctl3.doc.cell(lay3).plane("index"), before))
    del app
    print("\nall indexed-view checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
