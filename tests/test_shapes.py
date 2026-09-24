#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Shape and gradient tools.

Two properties carry the weight. Shapes must CLIP TO A SELECTION without
implementing clipping -- same as every other tool, because they route through
the same chokepoint. And the rubber-band preview must leave NO TRACE of the
intermediate shapes, because it restores from the same buffer that becomes
the undo entry.

Run: python3 tests/test_shapes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import shapes
from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.gradient import (CONICAL, DIAMOND, Gradient, LINEAR, RADIAL,
                                   REFLECTED, REPEAT_REFLECT, REPEAT_REPEAT,
                                   build_ramp)
from ochre.engine.history import HistoryStack
from ochre.engine.ini import IniDB
from ochre.engine.selection import Selection
from ochre.engine.tools import MOD_ALT, MOD_SHIFT, ToolContext, ToolEvent, ToolRegistry

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


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
                      secondary=(0, 0, 255, 255), history=hist)
    return d, lay, sel, hist, ctx


def drag(tool, ctx, points, mods=0, button=1):
    tool.begin(ctx, ToolEvent(points[0][0], points[0][1], modifiers=mods, button=button))
    for x, y in points[1:]:
        tool.motion(ctx, ToolEvent(x, y, modifiers=mods, button=button))
    return tool.end(ctx, ToolEvent(points[-1][0], points[-1][1],
                                   modifiers=mods, button=button))


def main():
    reg = ToolRegistry(IniDB(DATA))
    for name in ("line", "rectangle", "ellipse", "freeform", "gradient"):
        check("tool %s registered" % name, name in reg)

    # ---- coverage rasterisers -------------------------------------------
    fill = shapes.rect_coverage(32, 32, (8, 8, 24, 24), shapes.FILL)
    check("rect fill is solid inside", int(fill[16, 16]) == 255)
    check("rect fill is empty outside", int(fill[2, 2]) == 0)

    edge = shapes.rect_coverage(32, 32, (8, 8, 24, 24), shapes.OUTLINE, stroke=3)
    check("rect outline covers the edge", int(edge[8, 16]) > 128)
    check("RECT OUTLINE IS HOLLOW", int(edge[16, 16]) == 0,
          "-- an outline that fills its interior is a fill")
    check("rect outline is empty outside", int(edge[2, 2]) == 0)

    ring = shapes.ellipse_coverage(48, 48, (6, 6, 42, 42), shapes.OUTLINE, stroke=4)
    check("ellipse outline covers the rim", int(ring[24, 7]) > 100)
    check("ELLIPSE OUTLINE IS HOLLOW", int(ring[24, 24]) == 0)
    # The symmetry fix in the selection rasterisers is inherited here.
    check("ellipse outline is left-right symmetric",
          np.array_equal(ring, ring[:, ::-1]),
          "-- outlines inherit the rasteriser's symmetry")
    check("ellipse outline is top-bottom symmetric",
          np.array_equal(ring, ring[::-1, :]))

    soft = int(((ring > 0) & (ring < 255)).sum())
    check("outlines are antialiased", soft > 0)
    hard = shapes.ellipse_coverage(48, 48, (6, 6, 42, 42), shapes.OUTLINE,
                                   stroke=4, antialias=False)
    check("antialias=off gives hard edges", set(np.unique(hard)) <= {0, 255})

    line = shapes.line_coverage(32, 32, 4, 16, 28, 16, stroke=4)
    check("line covers its path", int(line[16, 16]) == 255)
    check("line does not cover elsewhere", int(line[4, 16]) == 0)
    thin = shapes.line_coverage(32, 32, 4, 16, 28, 16, stroke=1)
    check("a thinner line covers less",
          int(thin.astype(np.int64).sum()) < int(line.astype(np.int64).sum()))

    # A bent polyline must not have a notch at the joint, nor a bright seam.
    bent = shapes.polyline_coverage(48, 48, [(8, 8), (24, 24), (40, 8)], stroke=6)
    check("polyline joins are filled", int(bent[24, 24]) > 200,
          "-- a notch at the joint means the join disc is missing")
    check("polyline coverage never saturates past 255", int(bent.max()) <= 255,
          "-- summing overlapping pieces would make a bright seam")

    poly = shapes.polygon_coverage(48, 48, [(8, 8), (40, 8), (24, 40)],
                                   shapes.FILL)
    check("polygon fill works", int(poly[20, 24]) == 255)
    check("degenerate shapes are empty",
          int(shapes.polygon_coverage(16, 16, [(1, 1)], shapes.FILL).sum()) == 0)

    try:
        shapes.coverage(16, 16, "nonsense", (0, 0, 1, 1))
        check("unknown shape rejected", False, "-- no error")
    except ValueError:
        check("unknown shape rejected", True)

    b = shapes.bounds_for(64, 64, "rectangle", (10, 10, 20, 20), stroke=4)
    check("bounds pad for stroke and antialiasing",
          b.x < 10 and b.x1 > 20, "-- got %r" % (b,))
    check("bounds clip to the canvas",
          shapes.bounds_for(16, 16, "rectangle", (-50, -50, 100, 100)) ==
          Rect(0, 0, 16, 16))

    # ---- gradients -------------------------------------------------------
    ramp = build_ramp((0, 0, 0, 255), (255, 255, 255, 255))
    check("ramp spans both ends",
          tuple(ramp[0])[:3] == (0, 0, 0) and tuple(ramp[255])[:3] == (255, 255, 255))
    check("ramp is monotone", bool(np.all(np.diff(ramp[:, 0].astype(int)) >= 0)))

    # THE alpha-weighting property: fading to a TRANSPARENT colour must not
    # drag that colour into the visible end.
    ramp = build_ramp((255, 0, 0, 255), (0, 0, 255, 0))
    check("a ramp into transparency keeps its visible colour",
          int(ramp[0][0]) == 255 and int(ramp[0][2]) == 0)
    quarter = ramp[64]
    check("...and does not grey out on the way",
          int(quarter[0]) > int(quarter[2]),
          "-- a per-channel lerp would drag the invisible blue in early")
    check("alpha still interpolates", int(ramp[255][3]) == 0 and int(ramp[0][3]) == 255)

    rect = Rect(0, 0, 32, 32)
    for kind in (LINEAR, REFLECTED, DIAMOND, RADIAL, CONICAL):
        g = Gradient(kind=kind, start=(0, 16), end=(31, 16))
        out = g.render(rect)
        check("gradient %s renders" % kind, out.shape == (32, 32, 4))
        check("gradient %s is not flat" % kind, len(np.unique(out[..., 0])) > 4)

    g = Gradient(kind=LINEAR, start=(0, 16), end=(31, 16),
                 start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255))
    out = g.render(rect)
    check("linear runs along its axis",
          int(out[16, 0, 0]) < int(out[16, 31, 0]))
    check("linear is constant across its axis",
          int(out[0, 16, 0]) == int(out[31, 16, 0]))

    rev = Gradient(kind=LINEAR, start=(0, 16), end=(31, 16),
                   start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255),
                   reverse=True).render(rect)
    check("reverse flips the ramp", int(rev[16, 0, 0]) > int(rev[16, 31, 0]))

    g = Gradient(kind=RADIAL, start=(16, 16), end=(28, 16),
                 start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255))
    out = g.render(rect)
    check("radial is darkest at its centre",
          int(out[16, 16, 0]) < int(out[16, 28, 0]))
    # Pixel centres sit at +0.5, so the symmetric partner of column 4 about
    # centre 16 is column 27, not 28 -- both are 11.5 away.
    check("radial is rotationally symmetric",
          int(out[16, 4, 0]) == int(out[16, 27, 0]),
          "-- %d vs %d at equal radii"
          % (int(out[16, 4, 0]), int(out[16, 27, 0])))
    check("...and symmetric vertically too",
          int(out[4, 16, 0]) == int(out[27, 16, 0]))

    # Repeat policies
    g = Gradient(kind=LINEAR, start=(0, 0), end=(8, 0), repeat=REPEAT_REPEAT,
                 start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255))
    out = g.render(Rect(0, 0, 32, 4))
    check("repeat mode tiles the ramp",
          abs(int(out[0, 1, 0]) - int(out[0, 9, 0])) <= 8,
          "-- %d vs %d" % (int(out[0, 1, 0]), int(out[0, 9, 0])))
    g = Gradient(kind=LINEAR, start=(0, 0), end=(8, 0), repeat=REPEAT_REFLECT,
                 start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255))
    out = g.render(Rect(0, 0, 32, 4))
    check("reflect mode mirrors the ramp",
          abs(int(out[0, 7, 0]) - int(out[0, 9, 0])) <= 40)

    # Alpha-only: a gradient as a soft mask.
    base = np.empty((16, 16, 4), np.uint8)
    base[...] = (10, 200, 30, 255)
    g = Gradient(kind=LINEAR, start=(0, 0), end=(15, 0), alpha_only=True,
                 start_colour=(0, 0, 0, 255), end_colour=(0, 0, 0, 0))
    out = g.render(Rect(0, 0, 16, 16), base)
    check("alpha-only leaves colour untouched",
          tuple(out[8, 8][:3]) == (10, 200, 30),
          "-- got %r" % (tuple(out[8, 8]),))
    check("alpha-only gradients the alpha",
          int(out[8, 0, 3]) > int(out[8, 15, 3]))

    check("zero-length gradient does not divide by zero",
          Gradient(kind=LINEAR, start=(5, 5), end=(5, 5)).render(Rect(0, 0, 4, 4))
          is not None)
    for bad, kw in (("type", {"kind": "nope"}), ("repeat", {"repeat": "nope"})):
        try:
            Gradient(**kw)
            check("unknown gradient %s rejected" % bad, False, "-- no error")
        except ValueError:
            check("unknown gradient %s rejected" % bad, True)

    # ---- tools, end to end ----------------------------------------------
    d, lay, sel, hist, ctx = setup()
    drag(reg.create("rectangle", style="fill"), ctx, [(10, 10), (40, 40)])
    px = d.cell(lay).pixels
    # style=fill uses the PRIMARY colour -- it is the colour the user picked.
    # Secondary only comes into play in `both` mode, where two are needed.
    check("rectangle tool painted", tuple(px[25, 25]) == (255, 0, 0, 255),
          "-- fill-only uses the primary colour")
    check("rectangle tool left one history entry", len(hist) == 1)
    hist.undo(d)
    check("rectangle undoes", tuple(d.cell(lay).pixels[25, 25]) == (0, 0, 0, 255))

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("ellipse", style="both", stroke=3), ctx, [(8, 8), (56, 56)])
    px = d.cell(lay).pixels
    check("ellipse fill uses the secondary colour",
          tuple(px[32, 32]) == (0, 0, 255, 255))
    check("ellipse outline uses the primary colour",
          int(px[32, 9, 0]) > 100, "-- got %r" % (tuple(px[32, 9]),))

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("line", stroke=3), ctx, [(8, 32), (56, 32)])
    check("line tool painted", tuple(d.cell(lay).pixels[32, 32]) == (255, 0, 0, 255))

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("freeform", stroke=3), ctx,
         [(10, 10), (30, 20), (50, 10), (30, 50)])
    check("freeform painted along its path",
          int(d.cell(lay).pixels[..., 0].sum()) > 0)

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("gradient"), ctx, [(0, 32), (63, 32)])
    px = d.cell(lay).pixels
    check("gradient tool painted a ramp",
          int(px[32, 2, 0]) != int(px[32, 60, 0]))
    check("gradient left one history entry", len(hist) == 1)

    # ---- THE clipping property ------------------------------------------
    for name, opts in (("rectangle", {"style": "fill"}),
                       ("ellipse", {"style": "fill"}),
                       ("line", {"stroke": 6}),
                       ("freeform", {"stroke": 6}),
                       ("gradient", {})):
        d, lay, sel, hist, ctx = setup()
        sel.select_rect((0, 0, 32, 64))            # left half only
        tool = reg.create(name, **opts)
        drag(tool, ctx, [(4, 10), (30, 32), (60, 50)])
        outside = d.cell(lay).pixels[:, 34:]
        escaped = int((outside[..., 0] != 0).sum() + (outside[..., 2] != 0).sum())
        check("%s cannot paint outside the selection" % name, escaped == 0,
              "-- %d pixels escaped" % escaped)
        inside = d.cell(lay).pixels[:, :30]
        touched = int((inside[..., 0] != 0).sum() + (inside[..., 2] != 0).sum())
        check("%s still paints inside the selection" % name, touched > 0)

    # ---- THE preview property -------------------------------------------
    # Dragging through many intermediate shapes must leave no trace of them:
    # the preview restores from the same buffer that becomes the undo entry.
    d, lay, sel, hist, ctx = setup()
    pristine = d.cell(lay).pixels.copy()
    tool = reg.create("rectangle", style="fill")
    tool.begin(ctx, ToolEvent(8, 8))
    for size in range(12, 56, 4):
        tool.motion(ctx, ToolEvent(size, size))
    tool.end(ctx, ToolEvent(40, 40))
    painted = d.cell(lay).pixels.copy()

    direct_d, direct_lay, _s, _h, direct_ctx = setup()
    drag(reg.create("rectangle", style="fill"), direct_ctx, [(8, 8), (40, 40)])
    check("A DRAGGED SHAPE EQUALS THE SAME SHAPE DRAWN DIRECTLY",
          np.array_equal(painted, direct_d.cell(direct_lay).pixels),
          "-- intermediate previews left residue behind")

    hist.undo(d)
    check("undo after a long drag restores exactly",
          np.array_equal(d.cell(lay).pixels, pristine))

    # Cancel mid-drag must leave nothing.
    d, lay, sel, hist, ctx = setup()
    pristine = d.cell(lay).pixels.copy()
    tool = reg.create("ellipse")
    tool.begin(ctx, ToolEvent(10, 10))
    tool.motion(ctx, ToolEvent(50, 50))
    tool.cancel(ctx)
    check("cancelling a shape restores the canvas",
          np.array_equal(d.cell(lay).pixels, pristine))
    check("cancelling a shape leaves no history", len(hist) == 0)

    # ---- live modifiers --------------------------------------------------
    d, lay, sel, hist, ctx = setup()
    drag(reg.create("rectangle", style="fill"), ctx, [(10, 10), (50, 30)],
         mods=MOD_SHIFT)
    ys, xs = np.nonzero(d.cell(lay).pixels[..., 0] > 0)
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    check("Shift constrains a rectangle to a square", abs(w - h) <= 2,
          "-- %dx%d" % (w, h))

    d, lay, sel, hist, ctx = setup()
    drag(reg.create("rectangle", style="fill"), ctx, [(32, 32), (48, 48)],
         mods=MOD_ALT)
    ys, xs = np.nonzero(d.cell(lay).pixels[..., 0] > 0)
    check("Alt draws from the centre",
          int(xs.min()) < 32 and int(xs.max()) > 32,
          "-- span %d..%d should straddle the anchor" % (int(xs.min()), int(xs.max())))

    # ---- INI drives the defaults ----------------------------------------
    check("INI supplies the stroke width", reg.create("line").option("stroke") == 2)
    check("INI supplies the gradient kind",
          reg.create("gradient").option("kind") == "linear")
    check("overrides beat INI", reg.create("line", stroke=9).option("stroke") == 9)

    print("\nall shape and gradient checks passed")


if __name__ == "__main__":
    main()
