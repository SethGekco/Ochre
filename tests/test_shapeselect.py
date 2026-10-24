#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Rectangle Select with the Shapes palette: select a star, a heart, an
arrow -- any shape the Shapes tool draws -- in every combine mode.

Run: python3 tests/test_shapeselect.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import shapelib
from ochre.engine.selection import EXCLUDE, REPLACE, UNION
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def drag(ctl, a, b):
    ctl.begin_stroke(*a)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    ctl.end_stroke(*b)


def main():
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(120, 100)

    ctl.set_tool("select_rect")
    drag(ctl, (20, 20), (100, 80))
    m = ctl.selection.mask
    check("by default Rectangle Select still selects a rectangle (corners in)",
          m[21, 21] == 255 and m[78, 98] == 255 and m[10, 10] == 0)

    ctl.set_tool("select_rect", shape="star5")
    pos = ctl.history.position
    drag(ctl, (20, 20), (100, 80))
    m = ctl.selection.mask
    check("with the Star shape it selects a star: centre in, box corners out",
          m[50, 60] == 255 and m[22, 22] == 0 and m[78, 98] == 0)
    check("...antialiased at the edge", ((m > 0) & (m < 255)).any())
    check("...one undo step", ctl.history.position == pos + 1)
    ctl.undo()
    check("...undoable (back to the rectangle)", ctl.selection.mask[22, 22] == 255)

    ctl.set_tool("select_rect", shape="heart", mode=UNION)
    ctl.selection.select_all()
    ctl.selection.select_rect((0, 0, 30, 30))
    drag(ctl, (50, 30), (110, 90))
    m = ctl.selection.mask
    check("Add mode: the heart joins the rectangle already selected",
          m[10, 10] == 255 and m[60, 80] == 255 and m[95, 5] == 0)
    ctl.set_tool("select_rect", shape="ellipse", mode=EXCLUDE)
    drag(ctl, (60, 40), (100, 80))
    check("Subtract mode cuts a shape out", ctl.selection.mask[60, 80] == 0
          and ctl.selection.mask[10, 10] == 255)

    empty = []
    for kind in shapelib.CATALOG:
        ctl.selection.select_all()
        ctl.set_tool("select_rect", shape=kind, mode=REPLACE)
        drag(ctl, (20, 20), (100, 80))
        m = ctl.selection.mask
        inside = m is not None and m[20:80, 20:100].any()
        outside = m is not None and (m[:15].any() or m[85:].any() or m[:, :15].any() or m[:, 105:].any())
        if not inside or outside:
            empty.append(kind)
    check("every shape in the palette selects (%d shapes), inside its box only" % len(shapelib.CATALOG),
          not empty, str(empty))

    # ---- the Lasso has the same shapes, Freehand first ---------------------------
    ctl.selection.select_all()
    ctl.set_tool("lasso")
    check("Lasso Select defaults to Freehand", ctl.tool.option("shape") == "freehand")
    pts = [(20, 20), (100, 20), (100, 80), (60, 50), (20, 80)]
    ctl.begin_stroke(*pts[0])
    for p in pts[1:-1]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*pts[-1])
    m = ctl.selection.mask
    check("...and freehand still selects the outline drawn (the notch stays out)",
          m[30, 60] == 255 and m[75, 60] == 0)
    ctl.selection.select_all()
    ctl.set_tool("lasso", shape="star5")
    pos = ctl.history.position
    drag(ctl, (20, 20), (100, 80))
    m = ctl.selection.mask
    check("with a shape chosen, the lasso drags that shape (star: centre in, corners out)",
          m[50, 60] == 255 and m[22, 22] == 0 and ctl.history.position == pos + 1)
    ctl.set_tool("lasso", shape="heart", mode=UNION)
    drag(ctl, (0, 0), (30, 30))
    check("...in every combine mode", ctl.selection.mask[50, 60] == 255
          and ctl.selection.mask[12, 15] == 255)

    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    from ochre.ui.tooloptions import ShapePalette
    app = QApplication.instance() or QApplication([])
    c2 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=tempfile.mkdtemp())
    win = MainWindow(c2, UiSettings(DATA))
    c2.set_tool("select_rect")
    app.processEvents()
    palette = win.options_bar.widgets["shape"][1]
    check("Rectangle Select's options bar shows the Shapes palette",
          isinstance(palette, ShapePalette) and len(palette.buttons) == len(shapelib.CATALOG))
    palette._pick("star5")
    check("...and picking a shape there sets it", c2.tool.option("shape") == "star5")
    c2.set_tool("lasso")
    app.processEvents()
    lp = win.options_bar.widgets["shape"][1]
    check("the Lasso's options bar shows the palette too, Freehand first",
          isinstance(lp, ShapePalette) and list(lp.buttons)[0] == "freehand"
          and len(lp.buttons) == len(shapelib.CATALOG) + 1 and lp.value == "freehand")
    del app
    print("\nall shape-select checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
