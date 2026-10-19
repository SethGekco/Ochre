#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Selection tools, the Select menu, Quick Mask and selection undo.

Driven through the controller, the way the canvas drives them, and judged
by intersection-over-union against the shape that is actually there: a blue
disc on a grey ground. An assisted selection tool that gets "close" is not
good enough if close means it leaked across the canvas, so the bar is high.

Run: python3 tests/test_selecttools.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import morph, scissors, segment
from ochre.engine.geometry import Rect
from ochre.engine.selection import Selection, UNION
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(noise=False):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(100, 80, background=(220, 220, 220, 255))
    cell = ctl.doc.cell(ctl.active_layer)
    yy, xx = np.mgrid[0:80, 0:100]
    disc = (xx - 50) ** 2 + (yy - 40) ** 2 < 25 ** 2
    px = cell.plane("rgba")
    px[disc, :3] = (40, 70, 200)
    if noise:
        rng = np.random.default_rng(3)
        px[..., :3] = np.clip(px[..., :3].astype(int) + rng.integers(-15, 16, px[..., :3].shape),
                              0, 255).astype(np.uint8)
    cell._touched(ctl.doc.bounds)
    return ctl, cell, disc


def iou(ctl, disc):
    if ctl.selection.mask is None:
        return 0.0
    sel = ctl.selection.mask >= 128
    return float((sel & disc).sum()) / max(1, (sel | disc).sum())


def click(ctl, x, y, mods=0):
    ctl.begin_stroke(x, y, 1.0, mods)
    ctl.end_stroke(x, y, 1.0, mods)


def drag(ctl, pts, mods=0):
    ctl.begin_stroke(pts[0][0], pts[0][1], 1.0, mods)
    for x, y in pts[1:]:
        ctl.motion_stroke(x, y, 1.0, mods)
    ctl.end_stroke(pts[-1][0], pts[-1][1], 1.0, mods)


def main():
    # ---- morphology ---------------------------------------------------------
    sq = np.zeros((60, 60), np.uint8)
    sq[20:40, 20:40] = 255
    g = morph.grow(sq, 5)
    xs = np.nonzero(g >= 128)[1]
    check("grow is exact Euclidean: extent", (xs.min(), xs.max()) == (15, 44))
    check("...and round, not square, at the corners", int(g[15, 15]) < 128 and int(g[19, 19]) == 255)
    s = morph.shrink(sq, 5)
    xs = np.nonzero(s >= 128)[1]
    check("shrink is the dual", (xs.min(), xs.max()) == (25, 34))
    check("border straddles the edge", int((morph.border(sq, 3)[30] >= 128).sum()) == 12)
    holed = sq.copy()
    holed[25:35, 25:35] = 0
    filled = morph.remove_holes(holed)
    check("remove holes fills enclosed gaps only", int(filled[30, 30]) == 255 and int(filled[5, 5]) == 0)
    rc = morph.round_corners(sq, 6)
    check("rounded corners cut the corner but keep the straight edge",
          int(rc[20, 20]) < 128 and int(rc[20, 30]) == 255 and bool((rc <= sq).all()))

    # ---- the "no selection" semantics -----------------------------------------
    sel = Selection(10, 10)
    left = np.zeros((10, 10), np.uint8)
    left[:, :3] = 255
    sel.combine(left, UNION)
    check("ADDING TO 'NO SELECTION' GIVES JUST THE NEW SHAPE, not the whole canvas",
          sel.mask is not None and int(sel.mask[0, 8]) == 0 and int(sel.mask[0, 1]) == 255)

    # ---- marquees ---------------------------------------------------------------
    ctl, cell, disc = fresh()
    ctl.set_tool("select_row")
    click(ctl, 10, 33)
    rows = sorted(set(np.nonzero(ctl.selection.mask >= 128)[0].tolist()))
    check("single row selects one full-width row", rows == [33]
          and int((ctl.selection.mask[33] >= 128).sum()) == 100)
    check("SELECTIONS ARE UNDOABLE", ctl.history.position == 0)
    ctl.undo()
    check("...undo restores 'no selection'", ctl.selection.mask is None)
    ctl.redo()
    check("...and redo brings the row back", ctl.selection.mask is not None
          and int(ctl.selection.mask[33, 50]) == 255)
    ctl.set_tool("select_col")
    click(ctl, 77, 5)
    cols = sorted(set(np.nonzero(ctl.selection.mask >= 128)[1].tolist()))
    check("single column replaces it with one full-height column", cols == [77])

    # ---- click-by-click outlines --------------------------------------------------
    ctl, cell, disc = fresh()
    ctl.set_tool("polygon_lasso")
    for p in ((20, 20), (80, 20), (80, 60)):
        click(ctl, *p)
    check("the polygonal lasso keeps collecting between clicks", ctl.tool.active)
    kinds = [k[0] for k in ctl.tool.overlay()]
    check("...and shows its outline and anchors", kinds == ["poly", "handles"])
    ctl.hover_tool(50, 70)
    check("...with a rubber band that follows the pointer",
          ctl.tool.overlay()[0][1][-1] == (50.0, 70.0))
    ctl.finish_tool()
    m = ctl.selection.mask
    check("finishing closes the polygon", not ctl.tool.active and m[25, 70] >= 128 and m[55, 25] < 128)
    check("...as one undo entry", ctl.history.position == 0)

    ctl.set_tool("polygon_lasso")
    ctl.pick_radius = 4
    for p in ((10, 10), (40, 10), (40, 40), (10, 40), (11, 11)):
        click(ctl, *p)
    check("clicking the first anchor closes the outline", not ctl.tool.active
          and int((ctl.selection.mask >= 128).sum()) == 900)

    ctl.set_tool("polygon_lasso")
    click(ctl, 5, 5)
    click(ctl, 30, 5)
    ctl.cancel_stroke()
    check("Escape abandons an outline without touching the selection",
          not ctl.tool.active and int((ctl.selection.mask >= 128).sum()) == 900)

    ctl, cell, disc = fresh()
    cost = scissors.edge_cost(cell.plane("rgba"))
    path = scissors.path(cost, (25, 40), (50, 15))
    off = max(abs(math.hypot(x - 50, y - 40) - 24.5) for x, y in path)
    check("THE SCISSORS PATH FOLLOWS THE EDGE between anchors", off < 1.6,
          "-- strays %.2f px from the rim" % off)
    ctl.set_tool("scissors")
    for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
        click(ctl, 50 + 25 * math.cos(a), 40 + 25 * math.sin(a))
    ctl.finish_tool()
    check("six scissors clicks select the disc", iou(ctl, disc) > 0.97, "-- IoU %.3f" % iou(ctl, disc))

    # ---- colour-based ---------------------------------------------------------------
    ctl, cell, disc = fresh()
    ctl.set_tool("select_color")
    click(ctl, 50, 40)
    check("select by colour takes every matching pixel", iou(ctl, disc) == 1.0)

    ctl, cell, disc = fresh(noise=True)
    ctl.set_tool("quick_select", size=10)
    drag(ctl, [(45, 40), (55, 40)])
    check("QUICK SELECTION GROWS TO THE EDGES of what it is painted over",
          iou(ctl, disc) > 0.95, "-- IoU %.3f" % iou(ctl, disc))
    before = ctl.selection.mask.copy()
    ctl.set_tool("quick_select", size=10)
    drag(ctl, [(5, 5), (10, 5)], mods=4)
    check("alt-painting the background subtracts nothing from the subject",
          int(((ctl.selection.mask >= 128) & disc).sum()) == int(((before >= 128) & disc).sum()))

    for noisy in (False, True):
        label = "noisy" if noisy else "flat"
        ctl, cell, disc = fresh(noise=noisy)
        ctl.set_tool("object_select")
        drag(ctl, [(18, 8), (60, 40), (84, 74)])
        check("OBJECT SELECTION FINDS THE OBJECT in its box (%s colour)" % label,
              iou(ctl, disc) > 0.95, "-- IoU %.3f" % iou(ctl, disc))
        ctl, cell, disc = fresh(noise=noisy)
        ctl.set_tool("foreground_select")
        loop = [(50 + 33 * math.cos(a), 40 + 33 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
        drag(ctl, loop)
        check("FOREGROUND SELECT CUTS THE SUBJECT from inside a rough loop (%s)" % label,
              iou(ctl, disc) > 0.95, "-- IoU %.3f" % iou(ctl, disc))

    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[..., :3] = 128
    region = np.zeros((40, 40), bool)
    region[5:35, 5:35] = True
    check("segmenting a featureless image does not invent an object",
          int((segment.segment(rgba, region) >= 128).sum()) < 40 * 40)

    # ---- Select menu -------------------------------------------------------------------
    ctl, cell, disc = fresh()
    ctl.set_tool("select_rect")
    drag(ctl, [(20, 20), (60, 50)])
    n = ctl.history.position
    for op, args in ((ctl.grow_selection, (4,)), (ctl.shrink_selection, (4,)),
                     (ctl.border_selection, (3,)), (ctl.feather_selection, (2,)),
                     (ctl.sharpen_selection, ()), (ctl.remove_selection_holes, ()),
                     (ctl.round_selection, (6,))):
        op(*args)
    check("every Select menu operation is its own undo entry",
          ctl.history.position - n >= 6, "-- %d entries" % (ctl.history.position - n))
    while ctl.history.position > n:
        ctl.undo()
    check("...and undoing them all returns the original rectangle",
          int(ctl.selection.mask[20, 20]) == 255 and int(ctl.selection.mask[60, 70]) == 0)

    cell.plane("rgba")[:, :50, 3] = 0
    cell._touched(ctl.doc.bounds)
    ctl.alpha_to_selection()
    check("alpha to selection takes the layer's opacity",
          int(ctl.selection.mask[40, 20]) == 0 and int(ctl.selection.mask[40, 80]) == 255)

    # ---- Quick Mask ----------------------------------------------------------------------
    ctl, cell, disc = fresh()
    ctl.set_tool("select_rect")
    drag(ctl, [(10, 10), (40, 40)])
    check("QUICK MASK turns the selection into a paintable layer",
          ctl.toggle_quick_mask() and ctl.active_layer.name == "Quick Mask"
          and ctl.primary[:3] == (255, 255, 255))
    ctl.set_tool("brush", size=20, hardness=1.0)
    ctl.set_primary((0, 0, 0, 255))
    drag(ctl, [(30, 30), (31, 30)])
    check("...painting black on it, then leaving, deselects what was painted",
          not ctl.toggle_quick_mask() and int(ctl.selection.mask[30, 30]) == 0
          and int(ctl.selection.mask[12, 12]) == 255)
    check("...and the temporary layer is gone", [l.name for l in ctl.doc.layers()] == ["Background"])
    ctl.undo()
    check("...and the whole quick-mask edit undoes as one step",
          int(ctl.selection.mask[30, 30]) == 255)

    print("\nall selection tool checks passed")


if __name__ == "__main__":
    main()
