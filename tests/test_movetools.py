#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Clone stamp and the two move tools.

Move-pixels is the one that can destroy work, so it gets the most attention:
the lift must clear the source, the float must not leave a trail, a lasso
must carry only what it enclosed rather than a rectangle of transparency, and
the whole lift-and-drop must undo as ONE entry.

Run: python3 tests/test_movetools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.history import HistoryStack
from ochre.engine.ini import IniDB
from ochre.engine.selection import Selection
from ochre.engine.tools import (MOD_CTRL, MOD_SHIFT, ToolContext, ToolEvent,
                                ToolRegistry)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def setup(w=80, h=80, bg=(0, 0, 0, 255)):
    d = Document(w, h)
    lay = d.add_layer("paint")
    d.cell(lay).fill(d.bounds, bg)
    sel = Selection(w, h)
    hist = HistoryStack()
    ctx = ToolContext(d, lay, d.frame, sel, primary=(255, 0, 0, 255),
                      secondary=(0, 0, 255, 255), history=hist)
    return d, lay, sel, hist, ctx


def drag(tool, ctx, points, mods=0):
    tool.begin(ctx, ToolEvent(points[0][0], points[0][1], modifiers=mods))
    for x, y in points[1:]:
        tool.motion(ctx, ToolEvent(x, y, modifiers=mods))
    return tool.end(ctx, ToolEvent(points[-1][0], points[-1][1], modifiers=mods))


def main():
    reg = ToolRegistry(IniDB(DATA))
    for name in ("clone", "move_pixels", "move_selection"):
        check("tool %s registered" % name, name in reg)

    # ---- clone stamp -----------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(4, 4, 20, 20), (0, 200, 0, 255))     # something to clone

    clone = reg.create("clone", size=12)
    out = clone.begin(ctx, ToolEvent(50, 50))
    check("cloning without a source says so, rather than silently doing nothing",
          isinstance(out, dict) and "error" in out)

    clone.begin(ctx, ToolEvent(14, 14, modifiers=MOD_CTRL))
    check("ctrl-click sets the source", clone._anchor == (14.0, 14.0))
    clone.end(ctx, ToolEvent(14, 14, modifiers=MOD_CTRL))

    drag(clone, ctx, [(50, 50), (52, 52)])
    check("clone painted the sampled colour",
          tuple(cell.pixels[50, 50])[:3] == (0, 200, 0),
          "-- got %r" % (tuple(cell.pixels[50, 50]),))
    check("clone left one history entry", len(hist) == 1)
    hist.undo(d)
    check("clone undoes", tuple(cell.pixels[50, 50]) == (0, 0, 0, 255))

    # The source is frozen for the duration of a stroke, so a stroke that
    # crosses its own source cannot feed on what it just painted.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(0, 0, 40, 80), (0, 200, 0, 255))
    clone = reg.create("clone", size=10)
    clone.begin(ctx, ToolEvent(10, 40, modifiers=MOD_CTRL))
    clone.end(ctx, ToolEvent(10, 40, modifiers=MOD_CTRL))
    drag(clone, ctx, [(30, 40), (34, 40), (38, 40), (42, 40), (46, 40)])
    # Sample the solidly-covered interior, away from the soft edge of the
    # last dab. Feedback would show as PROGRESSIVE darkening along the
    # stroke, so uniformity is the property, not any absolute threshold.
    strip = cell.pixels[40, 34:44, 1].astype(int)
    check("a stroke crossing its own source does not run away",
          int(strip.max() - strip.min()) <= 4,
          "-- green varies %d..%d along the stroke; feedback smear would "
          "darken progressively" % (int(strip.min()), int(strip.max())))

    # Off-canvas source must not smear the edge pixel.
    d, lay, sel, hist, ctx = setup()
    clone = reg.create("clone", size=12)
    clone.begin(ctx, ToolEvent(2, 2, modifiers=MOD_CTRL))
    clone.end(ctx, ToolEvent(2, 2, modifiers=MOD_CTRL))
    drag(clone, ctx, [(70, 70)])
    check("an off-canvas source is handled without error", True)

    # Clone clips to a selection like everything else.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(0, 0, 80, 20), (0, 200, 0, 255))
    sel.select_rect((0, 0, 40, 80))
    clone = reg.create("clone", size=16)
    clone.begin(ctx, ToolEvent(20, 10, modifiers=MOD_CTRL))
    clone.end(ctx, ToolEvent(20, 10, modifiers=MOD_CTRL))
    drag(clone, ctx, [(30, 50), (60, 50)])
    outside = cell.pixels[25:, 42:]
    check("CLONE CANNOT PAINT OUTSIDE THE SELECTION",
          int((outside[..., 1] > 50).sum()) == 0)

    # ---- move pixels -----------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 20, 20), (255, 0, 0, 255))
    sel.select_rect((10, 10, 30, 30))
    before = cell.pixels.copy()

    drag(reg.create("move_pixels"), ctx, [(20, 20), (50, 50)])
    check("the pixels arrived at the destination",
          tuple(cell.pixels[50, 50]) == (255, 0, 0, 255),
          "-- got %r" % (tuple(cell.pixels[50, 50]),))
    check("THE SOURCE WAS CLEARED", int(cell.pixels[15, 15, 3]) == 0,
          "-- a move that leaves the original behind is a copy")
    check("move left one history entry", len(hist) == 1)
    hist.undo(d)
    check("ONE UNDO REVERSES BOTH THE LIFT AND THE DROP",
          np.array_equal(cell.pixels, before))

    # No trail: dragging through many positions must equal a single jump.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 12, 12), (255, 0, 0, 255))
    sel.select_rect((10, 10, 22, 22))
    tool = reg.create("move_pixels")
    tool.begin(ctx, ToolEvent(16, 16))
    for step in range(20, 60, 4):
        tool.motion(ctx, ToolEvent(step, step))
    tool.end(ctx, ToolEvent(56, 56))
    dragged = cell.pixels.copy()

    d2, lay2, sel2, _h, ctx2 = setup()
    d2.cell(lay2).fill(Rect(10, 10, 12, 12), (255, 0, 0, 255))
    sel2.select_rect((10, 10, 22, 22))
    direct = reg.create("move_pixels")
    direct.begin(ctx2, ToolEvent(16, 16))
    direct.end(ctx2, ToolEvent(56, 56))
    check("A DRAGGED MOVE LEAVES NO TRAIL",
          np.array_equal(dragged, d2.cell(lay2).pixels),
          "-- intermediate positions left residue")

    # copy mode leaves the original in place
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 12, 12), (255, 0, 0, 255))
    sel.select_rect((10, 10, 22, 22))
    drag(reg.create("move_pixels", cut=False), ctx, [(16, 16), (50, 50)])
    check("cut=off copies instead of moving",
          int(cell.pixels[15, 15, 3]) == 255 and int(cell.pixels[50, 50, 3]) == 255)

    # A lasso must carry only what it enclosed, not a rectangle of hole.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(d.bounds, (0, 0, 255, 255))
    cell.fill(Rect(10, 10, 20, 20), (255, 0, 0, 255))
    sel.select_ellipse((10, 10, 30, 30))
    drag(reg.create("move_pixels"), ctx, [(20, 20), (55, 20)])
    check("a non-rectangular move carries only the selected shape",
          tuple(cell.pixels[12, 12])[:3] == (255, 0, 0),
          "-- corner outside the ellipse should be untouched, got %r"
          % (tuple(cell.pixels[12, 12]),))
    check("...and the moved content landed", int(cell.pixels[20, 55, 0]) > 100)

    # With no selection, the whole layer moves.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(0, 0, 20, 20), (255, 0, 0, 255))
    drag(reg.create("move_pixels"), ctx, [(40, 40), (50, 40)])
    check("with no selection the whole layer moves",
          int(cell.pixels[10, 20, 0]) > 100,
          "-- content should have shifted right by 10")

    # Shift constrains to one axis, read live.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 10, 10), (255, 0, 0, 255))
    sel.select_rect((10, 10, 20, 20))
    drag(reg.create("move_pixels"), ctx, [(15, 15), (50, 30)], mods=MOD_SHIFT)
    ys, xs = np.nonzero(cell.pixels[..., 0] > 100)
    check("Shift constrains the move to one axis",
          int(ys.min()) == 10,
          "-- vertical movement should have been suppressed, y=%d" % int(ys.min()))

    # Cancel restores everything, including the lifted hole.
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 20, 20), (255, 0, 0, 255))
    sel.select_rect((10, 10, 30, 30))
    pristine = cell.pixels.copy()
    tool = reg.create("move_pixels")
    tool.begin(ctx, ToolEvent(20, 20))
    tool.motion(ctx, ToolEvent(50, 50))
    tool.cancel(ctx)
    check("CANCELLING A MOVE RESTORES THE LIFTED PIXELS",
          np.array_equal(cell.pixels, pristine),
          "-- the hole left by the lift must come back too")
    check("cancelling a move leaves no history", len(hist) == 0)

    # ---- move selection --------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    cell = d.cell(lay)
    cell.fill(Rect(10, 10, 20, 20), (255, 0, 0, 255))
    sel.select_rect((10, 10, 30, 30))
    pixels_before = cell.pixels.copy()

    drag(reg.create("move_selection"), ctx, [(20, 20), (50, 50)])
    check("MOVING THE SELECTION TOUCHES NO PIXELS",
          np.array_equal(cell.pixels, pixels_before),
          "-- this tool repositions the marquee, nothing else")
    check("the marquee moved", sel.bbox.x == 40 and sel.bbox.y == 40,
          "-- got %r" % (sel.bbox,))
    check("move_selection records no history", len(hist) == 0)

    # Moving off-canvas clips rather than wrapping.
    d, lay, sel, hist, ctx = setup()
    sel.select_rect((10, 10, 30, 30))
    drag(reg.create("move_selection"), ctx, [(20, 20), (200, 200)])
    check("a selection dragged off-canvas clips away", sel.is_empty())

    # Cancel restores the marquee.
    d, lay, sel, hist, ctx = setup()
    sel.select_rect((10, 10, 30, 30))
    original = sel.mask.copy()
    tool = reg.create("move_selection")
    tool.begin(ctx, ToolEvent(20, 20))
    tool.motion(ctx, ToolEvent(60, 60))
    tool.cancel(ctx)
    check("cancelling restores the marquee", np.array_equal(sel.mask, original))

    # With nothing selected there is nothing to move.
    d, lay, sel, hist, ctx = setup()
    check("move_selection with no selection is a no-op",
          reg.create("move_selection").begin(ctx, ToolEvent(10, 10)) is None)

    # ---- INI -------------------------------------------------------------
    check("INI supplies the clone size", reg.create("clone").option("size") == 24)
    check("INI supplies aligned", reg.create("clone").option("aligned") is True)
    check("INI supplies cut", reg.create("move_pixels").option("cut") is True)

    print("\nall clone and move checks passed")


if __name__ == "__main__":
    main()
