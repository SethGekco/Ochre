#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Selections: coverage masks, combine modes, and the None fast path.

Two properties matter most. Combining must PRESERVE ANTIALIASING -- boolean
operations on a 0/255 mask would hard-edge every combined selection. And
`mask is None` must behave identically to an all-255 mask, since the whole
hot-path optimisation rests on those being interchangeable.

Run: python3 tests/test_selection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.geometry import Rect
from ochre.engine.selection import (EXCLUDE, INTERSECT, REPLACE, Selection,
                                    UNION, XOR, rasterize_ellipse,
                                    rasterize_polygon, rasterize_rect,
                                    rgss_offsets)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- RGSS offsets ----------------------------------------------------
    off = rgss_offsets(4)
    check("rgss yields quality^2 samples", len(off) == 16)
    check("rgss samples stay inside the pixel",
          all(-0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 for x, y in off))
    ys = sorted(y for _, y in off)
    check("rgss spreads samples in y", len(set(ys)) == 16)
    xs = sorted(x for x, _ in off)
    check("rgss spreads samples in x (rotated, not aligned)", len(set(xs)) == 16)

    # ---- rectangle: exact fractional coverage ---------------------------
    m = rasterize_rect(10, 10, (2, 2, 5, 5))
    check("rect interior fully covered", int(m[3, 3]) == 255)
    check("rect exterior empty", int(m[0, 0]) == 0)
    check("rect covers exactly its area", int((m == 255).sum()) == 9)

    half = rasterize_rect(10, 10, (2.0, 2.0, 5.0, 2.5))
    check("half-covered row gets half coverage",
          120 <= int(half[2, 3]) <= 136, "-- got %d" % int(half[2, 3]))

    m = rasterize_rect(10, 10, (2.5, 2.5, 5.5, 5.5))
    check("fractional edges are antialiased", 0 < int(m[2, 2]) < 255)
    m = rasterize_rect(10, 10, (2.5, 2.5, 5.5, 5.5), antialias=False)
    check("antialias=off gives hard edges", set(np.unique(m)) <= {0, 255})

    check("off-canvas rect is empty",
          int(rasterize_rect(10, 10, (50, 50, 60, 60)).sum()) == 0)
    check("inverted rect is normalised",
          np.array_equal(rasterize_rect(10, 10, (5, 5, 2, 2)),
                         rasterize_rect(10, 10, (2, 2, 5, 5))))

    # ---- ellipse ---------------------------------------------------------
    e = rasterize_ellipse(32, 32, (4, 4, 28, 28))
    check("ellipse centre is inside", int(e[16, 16]) == 255)
    check("ellipse corner is outside", int(e[5, 5]) == 0)
    partial = int(((e > 0) & (e < 255)).sum())
    check("ellipse edge is antialiased", partial > 0,
          "-- no partially-covered pixels at all")
    check("antialiasing is confined to the rim", partial < int((e > 0).sum()) // 2,
          "-- %d of %d pixels partial" % (partial, int((e > 0).sum())))
    # A circle must rasterise identically in all four quadrants, or it looks
    # visibly lopsided. Note the box is chosen so the ellipse centre lands on
    # the array's own mirror axis (16.0 for a 32-wide array) -- an off-centre
    # ellipse legitimately does not survive an array mirror.
    check("ellipse is left-right symmetric",
          np.array_equal(e, e[:, ::-1]),
          "-- max diff %d" % int(np.abs(e.astype(int) - e[:, ::-1].astype(int)).max()))
    check("ellipse is top-bottom symmetric",
          np.array_equal(e, e[::-1, :]),
          "-- max diff %d" % int(np.abs(e.astype(int) - e[::-1, :].astype(int)).max()))
    check("ellipse is diagonally symmetric (circle, equal radii)",
          np.array_equal(e, e.T))
    area = int(e.astype(np.int64).sum()) / 255.0
    expect = np.pi * 12 * 12
    check("ellipse area is within 2%% of pi*r^2",
          abs(area - expect) / expect < 0.02,
          "-- got %.0f want %.0f" % (area, expect))

    # ---- polygon ---------------------------------------------------------
    tri = rasterize_polygon(32, 32, [(2, 2), (30, 2), (2, 30)])
    check("polygon interior filled", int(tri[6, 6]) == 255)
    check("polygon exterior empty", int(tri[28, 28]) == 0)
    partial = int(((tri > 0) & (tri < 255)).sum())
    check("polygon hypotenuse is antialiased", partial > 0,
          "-- no partially-covered pixels")
    # The soft pixels must form a diagonal band, not be scattered: a correct
    # rasteriser antialiases the hypotenuse and nothing else.
    ys, xs = np.nonzero((tri > 0) & (tri < 255))
    sums = (ys + xs)
    check("soft pixels lie along the hypotenuse",
          int(sums.max() - sums.min()) <= 3,
          "-- x+y spread %d, expected a thin diagonal band"
          % int(sums.max() - sums.min()))
    area = int(tri.astype(np.int64).sum()) / 255.0
    check("triangle area is about half the bounding box",
          abs(area - 0.5 * 28 * 28) / (0.5 * 28 * 28) < 0.05,
          "-- got %.0f" % area)
    check("degenerate polygon is empty",
          int(rasterize_polygon(8, 8, [(1, 1), (2, 2)]).sum()) == 0)

    # ---- Selection basics ------------------------------------------------
    s = Selection(32, 32)
    check("new selection selects all", s.selects_all())
    check("select-all is not empty", not s.is_empty())
    check("select-all bbox is the canvas", s.bbox == Rect(0, 0, 32, 32))
    check("mask_for returns None when everything is selected",
          s.mask_for(Rect(0, 0, 4, 4)) is None)

    s.clear()
    check("clear() deselects everything", s.is_empty() and not s.selects_all())

    s = Selection(32, 32).select_rect((8, 8, 16, 16))
    check("rect selection is active", not s.selects_all())
    check("bbox tracks the selection", s.bbox == Rect(8, 8, 8, 8),
          "-- got %r" % (s.bbox,))
    sub = s.mask_for(Rect(8, 8, 4, 4))
    check("mask_for returns coverage", sub is not None and int(sub[0, 0]) == 255)

    s.invert()
    check("invert flips coverage", int(s.mask[0, 0]) == 255 and int(s.mask[10, 10]) == 0)

    # A selection covering everything collapses back to the fast path.
    s = Selection(16, 16).select_rect((-5, -5, 25, 25))
    check("full-coverage selection collapses to None", s.selects_all(),
          "-- the hot path must go back to its fast route")

    # ---- combine modes preserve antialiasing ----------------------------
    a = Selection(32, 32)
    a.combine(rasterize_ellipse(32, 32, (2, 2, 20, 20)), REPLACE)
    soft = [v for v in np.unique(a.mask) if 0 < v < 255]
    check("source mask has soft edges", len(soft) > 0)

    for mode in (UNION, INTERSECT, EXCLUDE, XOR):
        t = a.copy()
        t.combine(rasterize_ellipse(32, 32, (12, 12, 30, 30)), mode)
        mid = [v for v in np.unique(t.mask) if 0 < v < 255]
        check("%s preserves antialiasing" % mode, len(mid) > 0,
              "-- %s hard-edged the result" % mode)

    # Semantics, on hard masks where they are unambiguous. Note these
    # deliberately leave the last two columns clear: a combination that
    # happened to cover the whole canvas would collapse to the select-all
    # fast path and there would be no mask left to inspect.
    left = np.zeros((8, 8), np.uint8); left[:, :4] = 255
    right = np.zeros((8, 8), np.uint8); right[:, 2:6] = 255

    # ...and that collapse is itself worth asserting.
    full_a = np.zeros((8, 8), np.uint8); full_a[:, :4] = 255
    full_b = np.zeros((8, 8), np.uint8); full_b[:, 2:] = 255
    t = Selection(8, 8); t.combine(full_a, REPLACE); t.combine(full_b, UNION)
    check("a union covering everything collapses to the fast path",
          t.selects_all())

    t = Selection(8, 8); t.combine(left, REPLACE); t.combine(right, UNION)
    check("union covers both", int(t.mask[0, 0]) == 255 and int(t.mask[0, 5]) == 255)

    t = Selection(8, 8); t.combine(left, REPLACE); t.combine(right, INTERSECT)
    check("intersect keeps the overlap only",
          int(t.mask[0, 3]) == 255 and int(t.mask[0, 0]) == 0 and int(t.mask[0, 6]) == 0)

    t = Selection(8, 8); t.combine(left, REPLACE); t.combine(right, EXCLUDE)
    check("exclude subtracts",
          int(t.mask[0, 0]) == 255 and int(t.mask[0, 3]) == 0)

    t = Selection(8, 8); t.combine(left, REPLACE); t.combine(right, XOR)
    check("xor keeps the symmetric difference",
          int(t.mask[0, 0]) == 255 and int(t.mask[0, 3]) == 0 and int(t.mask[0, 5]) == 255)

    t = Selection(8, 8)
    try:
        t.combine(left, "Nonsense")
        check("unknown combine mode rejected", False, "-- no error")
    except ValueError:
        check("unknown combine mode rejected", True)

    # Combining onto select-all must treat it as fully covered.
    t = Selection(8, 8)
    t.combine(left, INTERSECT)
    check("intersect against select-all yields the new mask",
          int(t.mask[0, 0]) == 255 and int(t.mask[0, 6]) == 0)

    # ---- combine_mask: the one multiply that does all clipping ----------
    s = Selection(8, 8)
    brush = np.full((4, 4), 200, dtype=np.uint8)
    check("no selection returns the brush unchanged",
          s.combine_mask(brush, Rect(0, 0, 4, 4)) is brush)

    s.select_rect((0, 0, 2, 8))          # left two columns only
    fused = s.combine_mask(brush, Rect(0, 0, 4, 4))
    check("selection scales the brush coverage",
          int(fused[0, 0]) == 200 and int(fused[0, 3]) == 0,
          "-- got %r" % (fused[0].tolist(),))

    check("combine_mask with no brush returns the selection",
          s.combine_mask(None, Rect(0, 0, 4, 4)) is not None)

    # None and an all-255 mask must be interchangeable -- the entire fast
    # path rests on this.
    explicit = Selection(8, 8)
    explicit.mask = np.full((8, 8), 255, dtype=np.uint8)
    implicit = Selection(8, 8)
    fused_e = explicit.combine_mask(brush.copy(), Rect(0, 0, 4, 4))
    fused_i = implicit.combine_mask(brush.copy(), Rect(0, 0, 4, 4))
    check("None behaves exactly like an all-255 mask",
          np.array_equal(fused_e, fused_i))

    # ---- feather ---------------------------------------------------------
    s = Selection(32, 32).select_rect((8, 8, 24, 24), antialias=False)
    check("hard rect before feathering", set(np.unique(s.mask)) <= {0, 255})
    s.feather(2)
    soft = [v for v in np.unique(s.mask) if 0 < v < 255]
    check("feather softens the edge", len(soft) > 0)
    check("feather keeps the interior solid", int(s.mask[16, 16]) == 255)
    check("feather leaves the far exterior clear", int(s.mask[0, 0]) == 0)

    nosel = Selection(8, 8)
    nosel.feather(3)
    check("feathering no selection is a no-op", nosel.selects_all())

    # ---- stencil ---------------------------------------------------------
    st = np.zeros((16, 16), dtype=bool)
    st[4:8, 4:8] = True
    s = Selection(16, 16).select_stencil(st)
    check("stencil becomes coverage",
          int(s.mask[5, 5]) == 255 and int(s.mask[0, 0]) == 0)
    check("stencil bbox is tight", s.bbox == Rect(4, 4, 4, 4))

    # ---- copy independence ----------------------------------------------
    a = Selection(8, 8).select_rect((0, 0, 4, 4))
    b = a.copy()
    b.combine(np.zeros((8, 8), np.uint8), REPLACE)
    check("copy is independent", int(a.mask[0, 0]) == 255)

    print("\nall selection checks passed")


if __name__ == "__main__":
    main()
