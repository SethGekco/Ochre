#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Building a palette from the image's colours (closest merged when there
are too many), converting every layer to a palette, and the colour picker
that opens on the palette with an All Colors tab.

Run: python3 tests/test_colourpicker.py
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

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def ncol(a):
    m = a[..., 3] > 0
    return len(np.unique(a[..., :3][m].reshape(-1, 3), axis=0))


def main():
    # Six colours, two of them near-twins of others.
    img = np.zeros((30, 60, 4), np.uint8)
    img[..., 3] = 255
    blocks = [(200, 30, 30), (203, 31, 29), (30, 200, 30), (30, 30, 200), (32, 29, 202), (240, 240, 240)]
    for k, c in enumerate(blocks):
        img[:, k * 10:(k + 1) * 10, :3] = c
    img[:5, 50:60, :3] = (200, 30, 30)              # red is the most used colour
    cols, counts = PF.used_colours([img])
    check("the image's colours are counted, most used first",
          len(cols) == 6 and tuple(cols[0]) in {(200, 30, 30)})
    plan = PF.plan_palette(cols, counts, 6)
    check("when they fit, the palette is exactly them, nothing merged",
          plan.exact and len(plan.colours) == 6 and plan.merges == [])
    plan = PF.plan_palette(cols, counts, 4)
    merged = [set(tuple(cols[i]) for i in m) for m, _r, _s in plan.merges]
    check("too many: the CLOSEST colours are the ones recommended for merging",
          len(plan.colours) == 4 and {(200, 30, 30), (203, 31, 29)} in merged
          and {(30, 30, 200), (32, 29, 202)} in merged, str(merged))
    check("...each merge shows its spread (small for near-twins)", all(s < 5 for _m, _r, s in plan.merges))
    red = [r for m, r, s in plan.merges if (200, 30, 30) in {tuple(cols[i]) for i in m}][0]
    check("...a merged colour is the pixel-weighted average (leans to the commoner red)",
          abs(red[0] - 200) <= 1)

    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(60, 30, background=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[...] = img
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    top = ctl.add_layer("top")
    tp = ctl.doc.cell(top).plane("rgba")
    tp[5:10, 5:50] = (205, 32, 30, 255)          # another near-red on a second layer
    ctl.doc.cell(top)._touched(ctl.doc.bounds)
    check("colours are gathered from every visible pixel layer", len(ctl.used_colours()[0]) == 7)
    pos = ctl.history.position
    pal = ctl.build_palette_from_image(4, apply=True)
    check("Build Palette from Image: a 4-colour palette, now the bar's",
          pal.size == 4 and ctl.swatch_palette is pal and ctl.swatch_key == ("generated",))
    check("...and with 'change the image', every layer uses only its colours, one undo step",
          ncol(ctl.doc.cell(lay).plane("rgba")) <= 4 and ncol(ctl.doc.cell(top).plane("rgba")) == 1
          and ctl.history.position == pos + 1)
    check("...the generated palette is listed on the bar",
          ("Generated", pal.name, ("generated",)) in ctl.palette_sources())
    ctl.undo()
    check("...undoable", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.build_palette_from_image(4, apply=False)
    check("without it, the image is left alone", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.set_swatch_palette(("builtin", "Game Boy (4)"))
    ctl.apply_palette_all()
    gb = {tuple(c) for c in PF.colours(ctl.swatch_palette).tolist()}
    used = {tuple(c) for l in (lay, top) for c in ctl.doc.cell(l).plane("rgba")[
        ctl.doc.cell(l).plane("rgba")[..., 3] > 0][:, :3].tolist()}
    check("Convert Image to Palette changes every layer to the palette's colours", used <= gb)
    ctl.undo()

    # ---- dialogs ---------------------------------------------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.colourpicker import BuildPaletteDialog, ColourPickerDialog
    app = QApplication.instance() or QApplication([])
    ctl.set_swatch_palette(("builtin", "PICO-8 (16)"))
    dlg = ColourPickerDialog(ctl, (0, 0, 0, 255))
    check("the colour picker opens on the palette tab, showing the chosen palette",
          dlg.tabs.currentIndex() == 0 and len(dlg.grid.colours) == 16
          and dlg.tabs.tabText(1) == "All Colors")
    dlg.grid.chosen.emit(8)
    check("...a swatch picks its colour", dlg.result_colour()[:3] == dlg.grid.colours[8])
    from PySide6.QtGui import QColor
    dlg.tabs.setCurrentIndex(1)
    dlg.chooser.setCurrentColor(QColor(12, 34, 56, 200))
    check("...the All Colors tab is the full chooser, alpha included",
          dlg.result_colour() == (12, 34, 56, 200))
    dlg.reject()
    ctl.swatch_palette = None
    ctl.swatch_key = None
    dlg = ColourPickerDialog(ctl, (0, 0, 0, 255))
    check("with no palette chosen it opens on All Colors", dlg.tabs.currentIndex() == 1)
    dlg.reject()
    b = BuildPaletteDialog(ctl)
    b.size.setValue(4)
    check("Build Palette dialog lists the recommended merges for a smaller size",
          b.list.count() >= 2 and "colours ->" in b.list.item(0).text())
    b.size.setValue(7)
    check("...and none when everything fits", b.list.count() == 0 and not b.apply.isEnabled())
    b.size.setValue(4)
    b.accept()
    check("...Build makes the palette", ctl.swatch_palette is not None and ctl.swatch_palette.size == 4)
    del app
    print("\nall colour-picker checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
