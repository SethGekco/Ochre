#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Zoom and pan arithmetic. No Qt, no display, no QApplication.

Coordinate bugs are miserable to chase through a widget, so the whole mapping
is pure functions and gets tested exhaustively here instead.

Run: python3 tests/test_viewstate.py
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ochre.engine.geometry import Rect
from ochre.ui.viewstate import (MAX_ZOOM, MIN_ZOOM, ZOOM_STEPS, ViewState,
                                clamp_zoom, format_zoom)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- zoom is exact, not floating point ------------------------------
    v = ViewState(100, 80)
    check("default zoom is exactly 1", v.zoom == Fraction(1, 1))
    check("zoom is a Fraction", isinstance(v.zoom, Fraction))
    check("100% formats cleanly", format_zoom(v.zoom) == "100%")

    # The reason for Fraction: a long zoom round-trip must return EXACTLY.
    start = v.zoom
    for _ in range(12):
        v.zoom_in()
    for _ in range(12):
        v.zoom_out()
    check("12 zoom-ins then 12 zoom-outs returns exactly 100%",
          v.zoom == start, "-- drifted to %s" % v.zoom)

    v = ViewState(100, 80)
    for _ in range(40):
        v.zoom_out()
    check("zoom clamps at the minimum", v.zoom == MIN_ZOOM)
    for _ in range(80):
        v.zoom_in()
    check("zoom clamps at the maximum", v.zoom == MAX_ZOOM)

    check("clamp rejects absurd values",
          clamp_zoom(Fraction(10000)) == MAX_ZOOM
          and clamp_zoom(Fraction(1, 10000)) == MIN_ZOOM)
    check("zoom steps ascend",
          all(ZOOM_STEPS[i] < ZOOM_STEPS[i + 1] for i in range(len(ZOOM_STEPS) - 1)))
    check("1:1 is one of the steps", Fraction(1, 1) in ZOOM_STEPS)
    # Every rung is an exact rational. That -- not any property of the
    # individual values -- is what makes stepping up and back down lossless:
    # it is list navigation over exact numbers, never repeated float
    # multiplication.
    check("every zoom step is an exact Fraction",
          all(isinstance(z, Fraction) for z in ZOOM_STEPS))
    check("no zoom step is a float in disguise",
          all(z == Fraction(z.numerator, z.denominator) for z in ZOOM_STEPS))
    # Round-tripping from any rung with headroom. Rungs within three steps
    # of either end legitimately saturate against the clamp, which is
    # correct behaviour rather than drift.
    drift = None
    for rung in ZOOM_STEPS[3:-3]:
        probe = ViewState(100, 80, rung)
        for _ in range(3):
            probe.zoom_in()
        for _ in range(3):
            probe.zoom_out()
        if probe.zoom != rung:
            drift = (rung, probe.zoom)
            break
    check("zoom round-trips exactly from every rung", drift is None,
          "-- %s became %s" % drift if drift else "")

    # ---- interpolation switch -------------------------------------------
    v = ViewState(100, 80, Fraction(1, 2))
    check("smooth below 100%", not v.use_nearest)
    v.zoom = Fraction(1, 1)
    check("nearest at exactly 100%", v.use_nearest)
    v.zoom = Fraction(8, 1)
    check("nearest above 100%", v.use_nearest)

    # ---- coordinate round-trip ------------------------------------------
    for zoom in (Fraction(1, 8), Fraction(1, 2), Fraction(1, 1),
                 Fraction(3, 1), Fraction(16, 1)):
        v = ViewState(200, 150, zoom)
        v.offset_x, v.offset_y = 37.0, -19.0
        worst = 0.0
        for dx, dy in ((0, 0), (1, 1), (199, 149), (100, 75), (7, 123)):
            wx, wy = v.doc_to_widget(dx, dy)
            bx, by = v.widget_to_doc(wx, wy)
            worst = max(worst, abs(bx - dx), abs(by - dy))
        check("doc->widget->doc round-trips at %s" % format_zoom(zoom),
              worst < 1e-9, "-- error %g" % worst)

    # ---- zoom_at pins the anchor ----------------------------------------
    v = ViewState(200, 150)
    v.offset_x, v.offset_y = 10.0, 20.0
    anchor = (123.0, 77.0)
    before = v.widget_to_doc(*anchor)
    v.zoom_at(Fraction(4, 1), anchor)
    after = v.widget_to_doc(*anchor)
    check("zoom_at keeps the document point under the cursor",
          abs(before[0] - after[0]) < 1e-9 and abs(before[1] - after[1]) < 1e-9,
          "-- moved from %r to %r" % (before, after))

    # ...and through a whole wheel sequence, which is where drift shows.
    v = ViewState(200, 150)
    anchor = (80.0, 60.0)
    origin = v.widget_to_doc(*anchor)
    for _ in range(6):
        v.zoom_in(anchor)
    for _ in range(6):
        v.zoom_out(anchor)
    final = v.widget_to_doc(*anchor)
    check("anchor survives a full zoom-in/zoom-out sequence",
          abs(origin[0] - final[0]) < 1e-6 and abs(origin[1] - final[1]) < 1e-6,
          "-- drifted from %r to %r" % (origin, final))

    # ---- fit and centre --------------------------------------------------
    v = ViewState(400, 200)
    v.fit(200, 200)
    cw, ch = v.canvas_size()
    check("fit makes the document fit", cw <= 200.1 and ch <= 200.1,
          "-- canvas %r" % ((cw, ch),))
    check("fit centres horizontally", abs(v.offset_x - (200 - cw) / 2) < 1e-9)
    check("fit centres vertically", abs(v.offset_y - (200 - ch) / 2) < 1e-9)

    v = ViewState(10, 10)
    v.fit(1000, 1000)
    check("fit does not exceed the maximum zoom", v.zoom <= MAX_ZOOM)

    v = ViewState(100000, 100000)
    v.fit(100, 100)
    check("fit does not go below the minimum zoom", v.zoom >= MIN_ZOOM)

    # ---- visible region --------------------------------------------------
    v = ViewState(500, 400, Fraction(1, 1))
    v.offset_x, v.offset_y = 0.0, 0.0
    vis = v.visible_doc_rect(100, 80)
    check("visible region covers the viewport", vis == Rect(0, 0, 100, 80),
          "-- got %r" % (vis,))

    v.offset_x, v.offset_y = -50.0, -40.0
    vis = v.visible_doc_rect(100, 80)
    check("panning moves the visible region", vis == Rect(50, 40, 100, 80),
          "-- got %r" % (vis,))

    v.offset_x, v.offset_y = 0.0, 0.0
    vis = v.visible_doc_rect(10000, 10000)
    check("visible region clips to the canvas", vis == Rect(0, 0, 500, 400))

    v.offset_x, v.offset_y = 100000.0, 100000.0
    check("a document panned entirely off-screen has no visible region",
          v.visible_doc_rect(100, 80) is None)

    # Rounding must be OUTWARD, or repaints leave stale slivers at the edges.
    v = ViewState(100, 100, Fraction(3, 1))
    v.offset_x = v.offset_y = 0.0
    r = v.widget_to_doc_rect(1, 1, 1, 1)      # a sub-pixel widget rect
    check("widget->doc rounds outward", r.w >= 1 and r.h >= 1,
          "-- got %r, a repaint would miss pixels" % (r,))

    # ---- panning ---------------------------------------------------------
    v = ViewState(100, 100)
    v.offset_x, v.offset_y = 0.0, 0.0
    v.pan_by(15, -25)
    check("pan_by moves the offset", (v.offset_x, v.offset_y) == (15.0, -25.0))

    # A small canvas in a big viewport is centred, not draggable into a corner.
    v = ViewState(50, 50, Fraction(1, 1))
    v.offset_x = 5000.0
    v.clamp_pan(400, 400)
    check("a canvas smaller than the viewport stays centred",
          abs(v.offset_x - 175.0) < 1e-9, "-- offset %r" % v.offset_x)

    # A big canvas may be panned, but not entirely out of sight.
    v = ViewState(4000, 4000, Fraction(1, 1))
    v.offset_x = v.offset_y = -99999.0
    v.clamp_pan(800, 600)
    cw, ch = v.canvas_size()
    check("a large canvas cannot be lost off-screen",
          v.offset_x > -cw and v.offset_y > -ch,
          "-- offset %r" % ((v.offset_x, v.offset_y),))

    # ---- rect mapping ----------------------------------------------------
    v = ViewState(100, 100, Fraction(2, 1))
    v.offset_x, v.offset_y = 10.0, 20.0
    x, y, w, h = v.doc_to_widget_rect(Rect(5, 5, 10, 10))
    check("doc rect maps to widget rect",
          (x, y, w, h) == (20.0, 30.0, 20.0, 20.0), "-- got %r" % ((x, y, w, h),))

    # ---- formatting ------------------------------------------------------
    check("exact zoom prints without decimals",
          format_zoom(Fraction(1, 2)) == "50%" and format_zoom(Fraction(8)) == "800%")
    check("inexact zoom prints with decimals",
          "." in format_zoom(Fraction(1, 3)))

    # ---- copy ------------------------------------------------------------
    v = ViewState(100, 100, Fraction(3, 1))
    v.offset_x = 42.0
    c = v.copy()
    c.offset_x = 0.0
    c.zoom = Fraction(1, 1)
    check("copy is independent", v.offset_x == 42.0 and v.zoom == Fraction(3, 1))

    # ---- genuinely Qt-free ----------------------------------------------
    check("viewstate imported no Qt",
          not any(m.startswith(("PySide", "PyQt", "shiboken")) for m in sys.modules),
          "-- Qt leaked into a module that must not need it")

    print("\nall viewstate checks passed")


if __name__ == "__main__":
    main()
