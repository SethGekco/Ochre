#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Line / Curve, vertical text and the type masks, through the controller.

Run: python3 tests/test_typecurve.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=160, h=120):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    return ctl, ctl.doc.cell(ctl.active_layer)


def drag(ctl, a, b, mods=0):
    ctl.begin_stroke(a[0], a[1], 1.0, mods)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 1.0, mods)
    ctl.end_stroke(b[0], b[1], 1.0, mods)


def main():
    # ---- Line / Curve ----------------------------------------------------------
    ctl, cell = fresh()
    ctl.set_tool("curve", stroke=3)
    n0 = len(ctl.history_labels())
    drag(ctl, (20, 60), (140, 60))
    alpha = cell.plane("rgba")[..., 3]
    check("Line / Curve draws a straight line on the drag", alpha[60, 80] > 200)
    check("...and stays editable (handles up, nothing committed yet)",
          ctl.tool.active and len(ctl.tool.overlay()) == 3
          and len(ctl.history_labels()) == n0)
    c0 = ctl.tool._pts[1]
    drag(ctl, c0, (c0[0], 10))
    c1 = ctl.tool._pts[2]
    drag(ctl, c1, (c1[0], 10))
    alpha = cell.plane("rgba")[..., 3]
    check("dragging the control handles bends it into a curve",
          alpha[60, 80] == 0 and alpha[:40, 70:90].max() > 200)
    check("...the live preview leaves nothing behind where the line was",
          alpha[60, 60:100].max() == 0)
    ctl.finish_tool()
    check("Enter commits it as ONE history entry",
          not ctl.tool.active and ctl.history_labels()[n0:] == ["Line / Curve"])
    ctl.undo()
    check("...which undoes completely", cell.plane("rgba")[..., 3].max() == 0)

    ctl.set_tool("curve", stroke=2)
    drag(ctl, (20, 100), (60, 100))
    drag(ctl, (100, 20), (140, 20))
    check("clicking away from the handles commits and starts the next line",
          ctl.history_labels()[-1] == "Line / Curve" and ctl.tool.active
          and cell.plane("rgba")[100, 40, 3] > 0)
    ctl.cancel_stroke()
    check("Escape abandons the line being edited",
          cell.plane("rgba")[20, 120, 3] == 0 and cell.plane("rgba")[100, 40, 3] > 0)
    ctl.set_tool("curve", stroke=2)
    ctl.begin_stroke(50, 50)
    ctl.end_stroke(50, 50)
    check("a click with no drag draws nothing and leaves nothing open",
          not ctl.tool.active and cell.plane("rgba")[50, 50, 3] == 0)

    ctl, cell = fresh()
    ctl.set_tool("curve", stroke=2, arrows="end")
    drag(ctl, (20, 60), (120, 60))
    ctl.finish_tool()
    a = cell.plane("rgba")[..., 3]
    widest = max((a[:, x] > 128).sum() for x in range(100, 125))
    check("an arrowhead is drawn at the end, much wider than the line",
          widest >= 8 and (a[:, 60] > 128).sum() <= 3, "(widest %d)" % widest)
    check("...and not at the start", (a[:, 22] > 128).sum() <= 3)
    ctl.set_tool("curve", stroke=3)
    drag(ctl, (20, 100), (120, 100))
    ctl.set_tool("brush")
    check("switching tools commits a pending curve instead of losing it",
          cell.plane("rgba")[100, 70, 3] > 200 and ctl.history_labels()[-1] == "Line / Curve")

    # ---- vertical text -----------------------------------------------------------
    ctl, cell = fresh()
    ctl.set_tool("text_vertical", layer=False, text="ABC", size=24)
    ctl.begin_stroke(70, 10)
    ctl.end_stroke(70, 10)
    rect = ctl.tool.caret_rect(ctl._ctx)
    check("the vertical text caret lies ACROSS the column", rect.w > rect.h)
    ctl.finish_tool()
    ys, xs = np.nonzero(cell.plane("rgba")[..., 3] > 64)
    check("vertical text stacks top to bottom (taller than wide)",
          ys.size and (ys.max() - ys.min()) > 2 * (xs.max() - xs.min()))
    ctl.undo()
    ctl.set_tool("text_vertical", layer=False, text="AB\nCD", size=20)
    ctl.begin_stroke(40, 10)
    ctl.end_stroke(40, 10)
    ctl.finish_tool()
    ys, xs = np.nonzero(cell.plane("rgba")[..., 3] > 64)
    check("...and a second line becomes a second column",
          (xs.max() - xs.min()) > 30)
    ctl.undo()

    # caret typing in vertical mode goes down the column
    ctl.set_tool("text_vertical", layer=False, size=20)
    ctl.begin_stroke(60, 10)
    ctl.end_stroke(60, 10)
    ctl.tool.insert(ctl._ctx, "X")
    y1 = ctl.tool.caret_rect(ctl._ctx).y
    ctl.tool.insert(ctl._ctx, "Y")
    y2 = ctl.tool.caret_rect(ctl._ctx).y
    check("typing moves the vertical caret DOWN", y2 > y1)
    ctl.finish_tool()

    # ---- type masks ----------------------------------------------------------------
    ctl, cell = fresh()
    before = cell.plane("rgba").copy()
    ctl.set_tool("type_mask", text="HI", size=48)
    n0 = len(ctl.history_labels())
    ctl.begin_stroke(20, 20)
    ctl.end_stroke(20, 20)
    check("while typing, a type mask previews on the canvas",
          cell.plane("rgba")[..., 3].max() > 0)
    ctl.finish_tool()
    check("committing makes the letters the SELECTION",
          ctl.selection.mask is not None and ctl.selection.mask.max() == 255
          and ctl.selection.mask[5, 5] == 0)
    check("...and leaves the layer exactly as it was",
          np.array_equal(cell.plane("rgba"), before))
    check("...recorded as a selection change", ctl.history_labels()[n0:] == ["Type Mask"],
          str(ctl.history_labels()[n0:]))
    ctl.undo()
    check("undo restores the previous selection", ctl.selection.mask is None)
    ctl.set_tool("vertical_type_mask", text="AB", size=30)
    ctl.begin_stroke(70, 5)
    ctl.end_stroke(70, 5)
    ctl.finish_tool()
    ys, xs = np.nonzero(ctl.selection.mask)
    check("the vertical type mask selects stacked letters",
          ys.size and (ys.max() - ys.min()) > (xs.max() - xs.min()))

    print("\nall type and curve checks passed")


if __name__ == "__main__":
    main()
