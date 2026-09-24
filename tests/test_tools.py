#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tools, driven headlessly by ToolEvents -- no Qt, no widgets, no display.

The property that matters most: NO TOOL IMPLEMENTS SELECTION CLIPPING. It
falls out of routing every write through Surface.apply_masked with the
selection fused into the tool's own coverage. So the test drives each
painting tool with an active selection and asserts none of them can escape
it -- which is a statement about the architecture, not about the tools.

Run: python3 tests/test_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.history import HistoryStack
from ochre.engine.ini import IniDB
from ochre.engine.palette import grayscale
from ochre.engine.selection import REPLACE, Selection, UNION
from ochre.engine.tools import (MOD_SHIFT, ToolContext, ToolEvent,
                                ToolRegistry, constrain_angle,
                                constrain_square, drag_rect)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def setup(w=64, h=64, bg=(0, 0, 0, 255)):
    d = Document(w, h)
    lay = d.add_layer("paint")
    d.cell(lay).fill(d.bounds, bg)
    sel = Selection(w, h)
    hist = HistoryStack()
    ctx = ToolContext(d, lay, d.frame, sel, primary=(255, 0, 0, 255),
                      secondary=(0, 255, 0, 255), history=hist)
    return d, lay, sel, hist, ctx


def drag(tool, ctx, points, mods=0, button=1):
    tool.begin(ctx, ToolEvent(points[0][0], points[0][1], modifiers=mods, button=button))
    for x, y in points[1:]:
        tool.motion(ctx, ToolEvent(x, y, modifiers=mods, button=button))
    return tool.end(ctx, ToolEvent(points[-1][0], points[-1][1],
                                   modifiers=mods, button=button))


def main():
    reg = ToolRegistry()
    check("builtin tools registered", len(reg) >= 9, "-- %d" % len(reg))
    for name in ("pencil", "brush", "eraser", "bucket", "eyedropper",
                 "select_rect", "select_ellipse", "lasso", "magic_wand"):
        check("tool %s available" % name, name in reg)
    check("unknown tool creates nothing", reg.create("nope") is None)

    # ---- INI-driven defaults and ordering -------------------------------
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    reg = ToolRegistry(IniDB(data))
    brush = reg.create("brush")
    check("INI supplies tool defaults", brush.option("size") == 16,
          "-- got %r" % brush.option("size"))
    check("INI coerces floats", isinstance(brush.option("hardness"), float))
    check("INI coerces booleans", brush.option("pressure_size") is True)
    check("INI supplies labels", reg.label("brush") == "Paintbrush")
    check("INI orders tools", reg.names()[0] == "pencil",
          "-- got %r" % reg.names()[:3])
    check("overrides beat INI", reg.create("brush", size=99).option("size") == 99)

    db = IniDB()
    db.load_string("[Tool:ghost]\nClass = does_not_exist\nLabel = Ghost\n")
    r2 = ToolRegistry(db)
    check("a tool whose class is missing is simply not offered", "ghost" not in r2)

    # ---- pencil ----------------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    pencil = reg.create("pencil")
    drag(pencil, ctx, [(10, 10), (20, 10), (20, 20)])
    px = d.cell(lay).pixels
    check("pencil painted", tuple(px[10, 10]) == (255, 0, 0, 255))
    check("pencil followed the drag", tuple(px[20, 20]) == (255, 0, 0, 255))
    check("pencil left one history entry", len(hist) == 1)
    # Aliasing is the feature: a pixel artist must get hard pixels.
    band = px[10, 8:24]
    check("pencil produces no partial coverage",
          set(np.unique(band[..., 3])) <= {0, 255},
          "-- pencil antialiased, which breaks pixel art")

    hist.undo(d)
    check("pencil stroke undoes", tuple(d.cell(lay).pixels[10, 10]) == (0, 0, 0, 255))

    # ---- brush -----------------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    brush = reg.create("brush", size=12)
    drag(brush, ctx, [(32, 32), (40, 32)])
    px = d.cell(lay).pixels
    check("brush painted", tuple(px[32, 32])[:3] == (255, 0, 0))
    soft = [v for v in np.unique(px[..., 0]) if 0 < v < 255]
    check("brush edge is antialiased", len(soft) > 0)
    check("brush left one history entry", len(hist) == 1)

    # Right button paints the secondary colour.
    d, lay, sel, hist, ctx = setup()
    drag(reg.create("pencil"), ctx, [(5, 5)], button=2)
    check("right button uses the secondary colour",
          tuple(d.cell(lay).pixels[5, 5]) == (0, 255, 0, 255))

    # ---- THE selection-clipping property --------------------------------
    for tool_name, opts in (("pencil", {"size": 5}), ("brush", {"size": 20}),
                            ("eraser", {"size": 20}), ("bucket", {})):
        d, lay, sel, hist, ctx = setup()
        sel.select_rect((0, 0, 32, 64))            # left half only
        tool = reg.create(tool_name, **opts)
        drag(tool, ctx, [(30, 32), (40, 32), (50, 32)])
        px = d.cell(lay).pixels
        outside = px[:, 34:]
        if tool_name == "eraser":
            escaped = int((outside[..., 3] != 255).sum())
        else:
            escaped = int((outside[..., 0] != 0).sum())
        check("%s cannot paint outside the selection" % tool_name, escaped == 0,
              "-- %d pixels escaped" % escaped)
        inside = px[:, :30]
        if tool_name == "eraser":
            touched = int((inside[..., 3] != 255).sum())
        else:
            touched = int((inside[..., 0] != 0).sum())
        check("%s still paints inside the selection" % tool_name, touched > 0)

    # An antialiased selection edge must give antialiased clipping.
    d, lay, sel, hist, ctx = setup()
    sel.select_ellipse((10, 10, 50, 50))
    drag(reg.create("brush", size=30, hardness=1.0), ctx, [(30, 30)])
    px = d.cell(lay).pixels
    partial = int(((px[..., 0] > 0) & (px[..., 0] < 255)).sum())
    check("soft selection edges clip softly", partial > 0,
          "-- clipping hard-edged an antialiased selection")

    # ---- eraser ----------------------------------------------------------
    d, lay, sel, hist, ctx = setup(bg=(255, 255, 255, 255))
    drag(reg.create("eraser", size=16), ctx, [(32, 32)])
    check("eraser removes alpha", int(d.cell(lay).pixels[32, 32, 3]) < 255)
    check("eraser leaves distant pixels alone",
          int(d.cell(lay).pixels[2, 2, 3]) == 255)

    # ---- bucket ----------------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(0, 0, 32, 64), (0, 0, 255, 255))     # left half blue
    bucket = reg.create("bucket", tolerance=16)
    bucket.begin(ctx, ToolEvent(10, 10))
    px = cell.pixels
    check("bucket filled the contiguous region", tuple(px[10, 10]) == (255, 0, 0, 255))
    check("bucket respected the colour boundary",
          tuple(px[10, 40]) == (0, 0, 0, 255))
    check("bucket left one history entry", len(hist) == 1)
    hist.undo(d)
    check("bucket fill undoes", tuple(cell.pixels[10, 10]) == (0, 0, 255, 255))

    # ---- eyedropper ------------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    d.cell(lay).fill(Rect(4, 4, 4, 4), (12, 34, 56, 255))
    picked = reg.create("eyedropper").begin(ctx, ToolEvent(5, 5))
    check("eyedropper reports rgba", picked["rgba"] == (12, 34, 56, 255))
    check("eyedropper sets the primary colour", ctx.primary == (12, 34, 56, 255))
    check("eyedropper reports no index on an RGBA layer", picked["index"] is None)

    # On an index-locked layer the INDEX is the authoritative answer -- a
    # spriter needs to know they picked index 17, not a hex colour.
    d = Document(16, 16)
    d.bind_palette(grayscale())
    il = d.add_layer("il", planes=("index", "rgba"), authoritative=("index",))
    d.cell(il).fill(Rect(0, 0, 8, 8), 17, name="index")
    ctx2 = ToolContext(d, il, d.frame, Selection(16, 16))
    picked = reg.create("eyedropper").begin(ctx2, ToolEvent(2, 2))
    check("eyedropper reports the index when one exists", picked["index"] == 17)
    check("eyedropper still reports the colour too",
          picked["rgba"] == (17, 17, 17, 255))
    check("picking off-canvas is harmless",
          reg.create("eyedropper").begin(ctx2, ToolEvent(99, 99)) is None)

    # A pencil on an index-locked layer writes INDICES, not colours.
    ctx2.primary_index = 42
    drag(reg.create("pencil"), ctx2, [(3, 3)])
    check("pencil writes indices on an index-locked layer",
          int(d.cell(il).plane("index")[3, 3]) == 42)
    check("derived colour follows the index",
          tuple(d.cell(il).pixels[3, 3]) == (42, 42, 42, 255))

    # ---- selection tools -------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    drag(reg.create("select_rect"), ctx, [(10, 10), (30, 30)])
    check("rect select produced a selection", not sel.selects_all())
    check("rect select bbox is right", sel.bbox == Rect(10, 10, 20, 20),
          "-- got %r" % (sel.bbox,))

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("select_ellipse"), ctx, [(10, 10), (40, 40)])
    check("ellipse select produced a selection", not sel.selects_all())
    check("ellipse centre selected", int(sel.mask[25, 25]) == 255)
    check("ellipse corner excluded", int(sel.mask[11, 11]) == 0)

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("lasso"), ctx,
         [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)])
    check("lasso produced a selection", not sel.selects_all())
    check("lasso interior selected", int(sel.mask[25, 25]) == 255)

    # Union mode must add to an existing selection rather than replace it.
    d, lay, sel, hist, ctx = setup()
    sel.select_rect((0, 0, 10, 10))
    drag(reg.create("select_rect", mode=UNION), ctx, [(40, 40), (50, 50)])
    check("union mode keeps the original region", int(sel.mask[5, 5]) == 255)
    check("union mode adds the new region", int(sel.mask[45, 45]) == 255)
    check("union mode excludes elsewhere", int(sel.mask[25, 25]) == 0)

    # Cancelling a drag restores what was there before it.
    d, lay, sel, hist, ctx = setup()
    sel.select_rect((0, 0, 10, 10))
    before = sel.mask.copy()
    tool = reg.create("select_rect")
    tool.begin(ctx, ToolEvent(20, 20))
    tool.motion(ctx, ToolEvent(50, 50))
    tool.cancel(ctx)
    check("cancelling a selection drag restores the previous selection",
          np.array_equal(sel.mask, before))

    # ---- magic wand ------------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(0, 0, 32, 64), (0, 0, 255, 255))
    reg.create("magic_wand", tolerance=16).begin(ctx, ToolEvent(10, 10))
    check("magic wand selected the contiguous region", int(sel.mask[10, 10]) == 255)
    check("magic wand stopped at the colour boundary", int(sel.mask[10, 40]) == 0)

    # Two disjoint patches of the same colour: contiguous gets one, global
    # gets both. This is GIMP's Fuzzy Select vs Select by Color, behind one
    # tool with a modifier.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(2, 2, 8, 8), (200, 200, 200, 255))
    cell.fill(Rect(40, 40, 8, 8), (200, 200, 200, 255))
    reg.create("magic_wand", tolerance=8).begin(ctx, ToolEvent(4, 4))
    check("contiguous wand takes only the touched patch",
          int(sel.mask[4, 4]) == 255 and int(sel.mask[44, 44]) == 0)

    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(2, 2, 8, 8), (200, 200, 200, 255))
    cell.fill(Rect(40, 40, 8, 8), (200, 200, 200, 255))
    reg.create("magic_wand", tolerance=8).begin(
        ctx, ToolEvent(4, 4, modifiers=MOD_SHIFT))
    check("Shift inverts contiguity, reaching the far patch",
          int(sel.mask[4, 4]) == 255 and int(sel.mask[44, 44]) == 255)

    # ---- modifier geometry ----------------------------------------------
    x, y = constrain_angle(0, 0, 10, 1)
    check("constrain snaps a near-horizontal line flat", abs(y) < 1e-9,
          "-- got y=%r" % y)
    x, y = constrain_angle(0, 0, 10, 9)
    check("constrain snaps 45 degrees", abs(x - y) < 1e-6)
    check("constrain is a no-op at zero length",
          constrain_angle(5, 5, 5, 5) == (5, 5))

    check("square constraint preserves the quadrant",
          constrain_square(0, 0, 10, 3) == (10, 10)
          and constrain_square(0, 0, -10, 3) == (-10, 10))

    r = drag_rect(ToolEvent(10, 10), ToolEvent(20, 15))
    check("drag_rect normalises", r == (10, 10, 20, 15))
    r = drag_rect(ToolEvent(10, 10), ToolEvent(20, 15), square=True)
    check("drag_rect squares on demand", (r[2] - r[0]) == (r[3] - r[1]))
    r = drag_rect(ToolEvent(10, 10), ToolEvent(20, 20), centre=True)
    check("drag_rect can draw from the centre", r == (0, 0, 20, 20))

    # ---- cancel leaves no trace -----------------------------------------
    d, lay, sel, hist, ctx = setup()
    pristine = d.cell(lay).pixels.copy()
    tool = reg.create("brush", size=20)
    tool.begin(ctx, ToolEvent(32, 32))
    tool.motion(ctx, ToolEvent(40, 40))
    tool.cancel(ctx)
    check("cancelled stroke restores the canvas",
          np.array_equal(d.cell(lay).pixels, pristine))
    check("cancelled stroke leaves no history", len(hist) == 0)

    # ---- tools are genuinely Qt-free ------------------------------------
    import ochre.engine.tools as tools_pkg
    check("tool module imports no Qt",
          not any(m.startswith(("PySide", "PyQt", "shiboken"))
                  for m in sys.modules))

    print("\nall tool checks passed")


if __name__ == "__main__":
    main()
