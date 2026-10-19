#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Transform tools, crops, the layer Move tool, alignment and the warp engine.

Through the controller, as the canvas drives them: lift on the first press,
drag, commit with finish_tool (Enter / double-click), and every transform
must be one undo entry that restores the layer exactly. Geometry is checked
on the pixels that come out, not on the tool's internal quad.

Run: python3 tests/test_transformtools.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import warp
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RED = (200, 30, 30, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=100, h=80):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    cell.fill(Rect(20, 20, 30, 20), RED)
    return ctl, cell


def drag(ctl, a, b, mods=0):
    ctl.begin_stroke(a[0], a[1], 1.0, mods)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 1.0, mods)
    ctl.end_stroke(b[0], b[1], 1.0, mods)


def click(ctl, x, y, mods=0):
    ctl.begin_stroke(x, y, 1.0, mods)
    ctl.end_stroke(x, y, 1.0, mods)


def bbox(cell, thresh=64):
    ys, xs = np.nonzero(cell.plane("rgba")[..., 3] > thresh)
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None


def main():
    # ---- the warp engine --------------------------------------------------
    src = [(0, 0), (10, 0), (10, 10), (0, 10)]
    dst = [(5, 5), (25, 8), (22, 30), (3, 26)]
    check("a homography maps its four points exactly",
          np.allclose(warp.apply(warp.homography(src, dst), src), dst, atol=1e-6))
    for n in (1, 2, 3, 4):
        m = warp.from_points(src[:n], dst[:n])
        check("handle maths: %d pin(s) map exactly" % n,
              np.allclose(warp.apply(m, src[:n]), dst[:n], atol=1e-6))
    img = np.zeros((40, 40, 4), np.uint8)
    img[10:30, 10:30] = (0, 0, 255, 128)
    out = warp.warp_homography(img, warp.rotate_about(math.radians(30), 20, 20), (0, 0, 40, 40))
    edge = out[(out[..., 3] > 0) & (out[..., 3] < 128)]
    check("resampling is alpha-weighted: a rotated edge keeps its colour",
          bool((edge[:, 2] == 255).all() and (edge[:, :2] == 0).all()))
    check("the identity transform is exact",
          np.array_equal(warp.warp_homography(img, np.eye(3), (0, 0, 40, 40)), img))
    idx = (np.arange(400).reshape(20, 20) % 7).astype(np.uint8)
    w = warp.warp_homography(idx, warp.scale_about(2, 2, 0, 0), (0, 0, 40, 40), index=True)
    check("index planes resample nearest: no invented palette entries",
          set(np.unique(w).tolist()) <= set(range(7)))
    cage = [(10, 10), (30, 10), (30, 30), (10, 30)]
    aff = warp.from_points(cage[:3], [(12, 10), (34, 12), (30, 32)])
    moved = [tuple(p) for p in warp.apply(aff, cage)]
    pts = [(20, 20), (15, 25)]
    check("mean value coordinates reproduce an affine cage deformation exactly",
          np.allclose(warp.mean_value_map(pts, cage, moved), warp.apply(aff, pts), atol=1e-6))

    # ---- unified transform ----------------------------------------------------
    ctl, cell = fresh()
    before = cell.plane("rgba").copy()
    ctl.set_tool("transform")
    drag(ctl, (35, 30), (55, 40))
    check("the transform stays pending between drags", ctl.tool.active)
    check("...showing its box and handles", [k[0] for k in ctl.tool.overlay()][:2] == ["poly", "handles"])
    ctl.finish_tool()
    check("dragging inside moves", bbox(cell) == (40, 69, 30, 49), "-- %s" % (bbox(cell),))
    check("...as one undo entry", ctl.history.position == 0)
    ctl.undo()
    check("...that restores the layer exactly", np.array_equal(cell.plane("rgba"), before))

    # The box hugs real paint even after earlier edits grew content_bbox.
    ctl.set_tool("transform")
    drag(ctl, (49.9, 39.9), (79.9, 59.9))
    ctl.finish_tool()
    check("A CORNER SCALES about the opposite corner (after an undo grew the stored bounds)",
          bbox(cell) == (20, 79, 20, 59), "-- %s" % (bbox(cell),))
    ctl.undo()
    ctl.set_tool("transform")
    drag(ctl, (49.9, 30), (64.9, 30))
    ctl.finish_tool()
    check("an edge scales one axis only", bbox(cell) == (20, 64, 20, 39), "-- %s" % (bbox(cell),))
    ctl.undo()
    ctl.set_tool("transform")
    drag(ctl, (49.9, 39.9), (79.9, 49.9), mods=1)
    ctl.finish_tool()
    b = bbox(cell)
    check("Shift keeps proportions", abs((b[1] - b[0] + 1) / float(b[3] - b[2] + 1) - 1.5) < 0.1,
          "-- %s" % (b,))
    ctl.undo()
    ctl.set_tool("transform")
    drag(ctl, (60, 30), (35, 55))           # outside the box: rotate
    ctl.finish_tool()
    b = bbox(cell)
    check("dragging outside rotates (90 degrees: 30x20 becomes 20x30)",
          (b[1] - b[0] + 1, b[3] - b[2] + 1) == (20, 30), "-- %s" % (b,))
    ctl.undo()

    ctl.set_tool("transform")
    drag(ctl, (35, 30), (40, 30))
    ctl.set_tool("brush")
    check("SWITCHING TOOLS APPLIES a pending transform instead of discarding it",
          ctl.history.position == 0 and bbox(cell)[0] == 25)
    ctl.undo()
    ctl.set_tool("transform")
    drag(ctl, (35, 30), (40, 30))
    ctl.cancel_stroke()
    check("Escape abandons it and restores everything",
          np.array_equal(cell.plane("rgba"), before) and ctl.history.position == -1)

    # A selection is transformed, and the selection follows the content.
    ctl.set_tool("select_rect")
    drag(ctl, (20, 20), (35, 40))
    ctl.set_tool("transform")
    drag(ctl, (27, 30), (67, 50))
    ctl.finish_tool()
    check("with a selection, only the selected part moves",
          cell.plane("rgba")[30, 45, 3] == 255 and cell.plane("rgba")[50, 70, 3] == 255
          and cell.plane("rgba")[30, 25, 3] == 0)
    check("...and the selection moves with it",
          ctl.selection.mask is not None and int(ctl.selection.mask[50, 70]) >= 128
          and int(ctl.selection.mask[30, 25]) < 128)

    # ---- the specialised tools ------------------------------------------------
    for name, a, b, expect in (("rotate_tool", (60, 30), (35, 55), (20, 30)),
                               ("scale_tool", (49.9, 39.9), (64.9, 49.9), (45, 30))):
        ctl, cell = fresh()
        ctl.set_tool(name)
        drag(ctl, a, b)
        ctl.finish_tool()
        bb = bbox(cell)
        check("%s: %s" % (name, "rotates" if "rot" in name else "scales"),
              (bb[1] - bb[0] + 1, bb[3] - bb[2] + 1) == expect, "-- %s" % (bb,))
    ctl, cell = fresh()
    ctl.set_tool("shear_tool")
    drag(ctl, (35, 30), (45, 30))
    ctl.finish_tool()
    top = np.nonzero(cell.plane("rgba")[21, :, 3] > 64)[0]
    bot = np.nonzero(cell.plane("rgba")[38, :, 3] > 64)[0]
    check("shear slides the top and bottom edges in opposite directions",
          top.mean() != bot.mean() and abs((top.max() - top.min()) - (bot.max() - bot.min())) <= 2)
    ctl, cell = fresh()
    ctl.set_tool("perspective_tool")
    drag(ctl, (49.9, 39.9), (70, 60))
    ctl.finish_tool()
    check("perspective drags one corner alone",
          cell.plane("rgba")[55, 65, 3] > 0 and cell.plane("rgba")[22, 22, 3] == 255)
    ctl, cell = fresh()
    cell.fill(Rect(20, 20, 10, 20), (0, 200, 0, 255))
    ctl.set_tool("flip_tool")
    click(ctl, 30, 30)
    check("flip mirrors the content in one click",
          cell.plane("rgba")[30, 45, :3].tolist() == [0, 200, 0]
          and cell.plane("rgba")[30, 25, :3].tolist() == list(RED[:3]) and not ctl.tool.active)
    ctl, cell = fresh()
    ctl.set_tool("transform3d")
    drag(ctl, (35, 30), (50, 30))
    ctl.finish_tool()
    bb = bbox(cell)
    check("3D transform tilts the plane (it narrows in perspective)",
          ctl.history.position == 0 and bb is not None and (bb[1] - bb[0] + 1) < 30)
    ctl, cell = fresh()
    ctl.set_tool("handle_transform")
    click(ctl, 20, 20)
    drag(ctl, (20, 20), (30, 25))
    ctl.finish_tool()
    check("handle transform with one pin moves", bbox(cell)[:3:2] == (30, 25),
          "-- %s" % (bbox(cell),))
    ctl, cell = fresh()
    ctl.set_tool("handle_transform")
    click(ctl, 20, 20)
    click(ctl, 50, 20)
    drag(ctl, (50, 20), (80, 20))
    ctl.finish_tool()
    bb = bbox(cell)
    check("...with two pins, dragging one apart scales", bb[1] - bb[0] + 1 >= 58,
          "-- %s" % (bb,))

    ctl, cell = fresh()
    ctl.set_tool("cage")
    for p in ((15, 15), (55, 15), (55, 45), (15, 45)):
        click(ctl, *p)
    check("the cage collects points and shows them", ctl.tool.active
          and [k[0] for k in ctl.tool.overlay()] == ["poly", "handles"])
    ctl.finish_tool()
    check("Enter closes the cage and keeps the tool active for dragging", ctl.tool.active)
    drag(ctl, (55, 45), (75, 60))
    ctl.finish_tool()
    check("dragging a cage point bends the content toward it",
          cell.plane("rgba")[50, 60, 3] > 0 and ctl.history.position == 0)

    ctl, cell = fresh()
    before = cell.plane("rgba").copy()
    ctl.set_tool("warp", size=30, mode="move")
    ctl.begin_stroke(35, 30)
    for i in range(1, 8):
        ctl.motion_stroke(35 + i * 3, 30)
    ctl.end_stroke(56, 30)
    check("warp pushes pixels along the stroke",
          int((before != cell.plane("rgba")).any(-1).sum()) > 20 and ctl.history.position == 0)
    # Changed in the options bar, as a user does -- the SAME tool, so the
    # distortion it has accumulated is still there to restore.
    ctl.set_tool_option("mode", "restore")
    ctl.set_tool_option("strength", 1.0)
    ctl.begin_stroke(35, 30)
    for i in range(1, 8):
        ctl.motion_stroke(35 + i * 3, 30)
    ctl.end_stroke(56, 30)
    check("...and Restore paints the original back",
          int((before != cell.plane("rgba")).any(-1).sum()) < 20)

    # ---- crops ------------------------------------------------------------------
    ctl, cell = fresh()
    ctl.set_tool("crop")
    drag(ctl, (10, 10), (60, 50))
    check("the crop box waits for confirmation", ctl.tool.active and ctl.doc.width == 100)
    ctl.finish_tool()
    check("Enter crops the canvas", (ctl.doc.width, ctl.doc.height) == (50, 40))
    ctl.undo()
    check("...undoably", (ctl.doc.width, ctl.doc.height) == (100, 80))
    ctl.set_tool("crop", aspect="1:1")
    drag(ctl, (10, 10), (60, 30))
    ctl.finish_tool()
    check("a locked aspect ratio holds", ctl.doc.width == ctl.doc.height)
    ctl, cell = fresh()
    ctl.set_tool("perspective_crop")
    drag(ctl, (10, 10), (90, 70))
    drag(ctl, (90, 10), (80, 20))
    ctl.finish_tool()
    check("perspective crop straightens a quad into a rectangle",
          ctl.doc.width >= 70 and ctl.doc.height >= 55 and ctl.history.position == 0)

    # ---- moving and aligning layers ---------------------------------------------
    ctl, cell = fresh()
    top = ctl.add_layer("top")
    ctl.doc.cell(top).fill(Rect(70, 50, 10, 10), (0, 0, 255, 255))
    ctl.set_active_layer(top)
    ctl.set_tool("move_layer")
    drag(ctl, (30, 25), (40, 25))
    check("MOVE LAYER auto-selects the layer under the pointer", ctl.active_layer.name == "Background")
    check("...and drags it", bbox(cell) == (30, 59, 20, 39))
    ctl.undo()
    ctl.align_layer("left")
    check("align left", bbox(cell)[0] == 0)
    ctl.align_layer("bottom")
    check("align bottom", bbox(cell)[3] == 79)
    ctl.align_layer("hcenter")
    b = bbox(cell)
    check("align centre", abs((b[0] + b[1] + 1) / 2.0 - 50) <= 1)
    ctl.offset_layer(50, 0, True)          # centred block (x 35..64) straddles the edge
    xs = np.nonzero(cell.plane("rgba")[70, :, 3] > 0)[0]
    check("offset wraps what leaves one edge onto the other", xs.min() == 0 and xs.max() == 99)
    ctl, cell = fresh()
    ctl.rotate_layer(90)
    b = bbox(cell)
    check("rotate layer by angle", (b[1] - b[0] + 1, b[3] - b[2] + 1) == (20, 30), "-- %s" % (b,))
    ctl.scale_layer(0.5)
    b = bbox(cell)
    # Within a pixel: after two bilinear resamples, edge coverage straddles
    # the alpha threshold this measures with.
    check("scale layer", abs(b[1] - b[0] + 1 - 10) <= 1 and abs(b[3] - b[2] + 1 - 15) <= 1,
          "-- %s" % (b,))
    ctl.rotate_image_by(30)
    check("rotate image by an angle grows the canvas to fit",
          (ctl.doc.width, ctl.doc.height) == (127, 120))
    ctl.undo()
    check("...undoably", (ctl.doc.width, ctl.doc.height) == (100, 80))

    print("\nall transform tool checks passed")


if __name__ == "__main__":
    main()
