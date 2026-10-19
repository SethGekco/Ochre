#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Vector paths: the engine, the path tools, and fill / stroke / selection.

The path tools are driven through the controller exactly as the canvas
drives them -- press, drag, release, Enter -- and every gesture must be one
undo step. Selection-to-path-to-selection must come back pixel-identical,
because "trace it, tweak it, select it again" is the whole reason to have
both directions.

Run: python3 tests/test_paths.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import paths as P
from ochre.engine import transform
from ochre.engine.fileio import ochre_format
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHIFT, CTRL, ALT = 1, 2, 4


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=120, h=100):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    return ctl


def click(ctl, x, y, mods=0):
    ctl.begin_stroke(x, y, 1.0, mods)
    ctl.end_stroke(x, y, 1.0, mods)


def drag(ctl, a, b, mods=0, steps=4):
    ctl.begin_stroke(a[0], a[1], 1.0, mods)
    for i in range(1, steps + 1):
        t = i / float(steps)
        ctl.motion_stroke(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 1.0, mods)
    ctl.end_stroke(b[0], b[1], 1.0, mods)


def circle_path(cx=50, cy=50, r=40):
    k = 0.5523 * r
    return P.Path("c", [P.Subpath([
        P.Anchor(cx, cy - r, cx - k, cy - r, cx + k, cy - r, True),
        P.Anchor(cx + r, cy, cx + r, cy - k, cx + r, cy + k, True),
        P.Anchor(cx, cy + r, cx + k, cy + r, cx - k, cy + r, True),
        P.Anchor(cx - r, cy, cx - r, cy + k, cx - r, cy - k, True)], closed=True)])


def main():
    # ---- engine -------------------------------------------------------------
    m = P.fill_mask(circle_path(), 100, 100)
    area = m.sum() / 255.0
    check("a four-anchor circle fills to the area of a circle (%.0f vs %.0f)"
          % (area, math.pi * 1600), abs(area - math.pi * 1600) < 0.01 * math.pi * 1600)
    s = P.stroke_mask(circle_path(), 100, 100, 4.0)
    check("stroking it covers circumference x width",
          abs(s.sum() / 255.0 - 2 * math.pi * 40 * 4) < 0.05 * 2 * math.pi * 40 * 4)

    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 30:80] = 255
    mask[30:40, 40:50] = 0          # a hole
    mask[70:71, 10:90] = 255        # a one-pixel line
    mask[5, 5] = 255                # a single pixel
    traced = P.from_mask(mask)
    check("selection -> path finds the outline, the hole, the line and the pixel",
          len(traced.subpaths) == 4)
    for aa in (False, True):
        back = P.fill_mask(traced, 100, 100, antialias=aa)
        check("...and path -> selection gives back the SAME pixels (antialias=%s)" % aa,
              np.array_equal(back >= 128, mask >= 128))
    yy, xx = np.mgrid[:300, :300]
    ell = ((((xx - 150) / 120.0) ** 2 + ((yy - 150) / 70.0) ** 2) < 1).astype(np.uint8) * 255
    pe = P.from_mask(ell)
    check("an ellipse round-trips exactly, with far fewer anchors than edge steps",
          np.array_equal(P.fill_mask(pe, 300, 300, False) >= 128, ell >= 128)
          and sum(len(sp) for sp in pe.subpaths) < 400)

    sub = circle_path().subpaths[0]
    want = P.bezier_point(*sub.segments()[0], 0.3)
    P.insert_anchor(sub, 0, 0.3)
    check("inserting an anchor leaves the curve's shape unchanged",
          np.allclose(sub.anchors[1].x, want[0]) and np.allclose(sub.anchors[1].y, want[1]))
    check("hit testing finds an anchor, a handle and a segment",
          P.hit_anchor(circle_path(), 50.5, 10.4, 3)[2] == "anchor"
          and P.hit_anchor(circle_path(), 50 + 22.09, 10, 2)[2] == "out"
          and P.hit_segment(circle_path(), 89.9, 52, 2) is not None)
    smooth = P.catmull_rom([(0, 0), (10, 10), (20, 0)])
    check("Catmull-Rom anchors pass through their points with collinear handles",
          (smooth[1].x, smooth[1].y) == (10.0, 10.0) and abs(smooth[1].iy - smooth[1].oy) < 1e-9)
    wave = [(x, 50 + 20 * math.sin(x / 10.0)) for x in np.arange(0, 100, 0.5)]
    fit = P.fit_freehand(wave, 1.0)
    fitted = P.flatten(P.Path("f", [P.Subpath(fit)]))[0][0]
    err = max(min(abs(fy - (50 + 20 * math.sin(fx / 10.0))) for fx, fy in [pt])
              for pt in fitted)
    check("a freehand wave fits to %d anchors within 2px" % len(fit), len(fit) < 30 and err < 2.0)

    # ---- the Pen tool, through the controller ---------------------------------
    ctl = fresh()
    ctl.set_tool("pen")
    n0 = len(ctl.history_labels())
    click(ctl, 20, 20)
    check("the first click creates a Work Path with one anchor",
          ctl.doc.path is not None and ctl.doc.path.name == "Work Path"
          and len(ctl.doc.path.subpaths[0]) == 1)
    check("...and the pen keeps drawing (stays active for the next click)", ctl.tool.active)
    drag(ctl, (80, 20), (100, 40))
    a = ctl.doc.path.subpaths[0].anchors[1]
    check("a click-drag makes a smooth anchor with mirrored handles",
          a.smooth and (a.ox, a.oy) == (100.0, 40.0) and (a.ix, a.iy) == (60.0, 0.0))
    ctl.hover_tool(70, 80)
    check("between clicks the pen shows a rubber-band curve to the pointer",
          ctl.tool.overlay() and ctl.tool.overlay()[0][0] == "curve")
    click(ctl, 50, 80)
    click(ctl, 20.5, 20.5)
    sp = ctl.doc.path.subpaths[0]
    check("clicking the first anchor closes the path and stops drawing",
          sp.closed and len(sp) == 3 and not ctl.tool.active)
    check("...each anchor was its own undo step", len(ctl.history_labels()) - n0 == 4,
          str(ctl.history_labels()[n0:]))
    ctl.undo()
    check("undo reopens it", not ctl.doc.path.subpaths[0].closed)
    ctl.redo()
    check("redo closes it again", ctl.doc.path.subpaths[0].closed)

    # Enter ends an open path; the next click starts a new subpath.
    ctl.set_tool("pen")
    click(ctl, 10, 90)
    click(ctl, 40, 90)
    ctl.finish_tool()
    check("Enter ends an open subpath", not ctl.tool.active
          and len(ctl.doc.path.subpaths) == 2 and not ctl.doc.path.subpaths[1].closed)

    # Editing with the pen: drag an existing anchor when not drawing.
    before = (ctl.doc.path.subpaths[0].anchors[2].x, ctl.doc.path.subpaths[0].anchors[2].y)
    drag(ctl, before, (before[0] + 5, before[1] - 7))
    moved = ctl.doc.path.subpaths[0].anchors[2]
    check("dragging an anchor moves it (and its handles)",
          (moved.x, moved.y) == (before[0] + 5, before[1] - 7))
    ctl.undo()
    back = ctl.doc.path.subpaths[0].anchors[2]
    check("...undoably", (back.x, back.y) == before)
    # Ctrl-click a segment inserts; Ctrl+Shift-click an anchor deletes.
    n_anchor = len(ctl.doc.path.subpaths[1])
    click(ctl, 25, 90, CTRL)
    check("Ctrl-click on a segment adds an anchor", len(ctl.doc.path.subpaths[1]) == n_anchor + 1)
    click(ctl, 25, 90, CTRL | SHIFT)
    check("Ctrl+Shift-click on an anchor deletes it", len(ctl.doc.path.subpaths[1]) == n_anchor)
    # Alt on a smooth anchor's handle breaks the symmetry.
    a1 = ctl.doc.path.subpaths[0].anchors[1]
    ix0, iy0 = a1.ix, a1.iy
    drag(ctl, (a1.ox, a1.oy), (a1.ox + 10, a1.oy + 10), ALT)
    a1 = ctl.doc.path.subpaths[0].anchors[1]
    check("Alt-dragging a handle moves it alone (a cusp)",
          (a1.ix, a1.iy) == (ix0, iy0) and not a1.smooth)
    drag(ctl, (a1.x, a1.y), (a1.x + 1, a1.y + 1), ALT)
    a1 = ctl.doc.path.subpaths[0].anchors[1]
    check("Alt on an anchor converts it (here, pulls fresh smooth handles)", a1.smooth)
    ctl.set_tool("pen")
    click(ctl, a1.x, a1.y, ALT)
    a1 = ctl.doc.path.subpaths[0].anchors[1]
    check("Alt-click on an anchor makes a sharp corner", not a1.has_handles)
    ctl.set_tool("pen", polygonal=True)
    click(ctl, 100, 90)
    drag(ctl, (110, 60), (118, 70))
    ctl.finish_tool()
    check("polygonal mode makes corners even when dragged",
          not ctl.doc.path.subpaths[-1].anchors[-1].has_handles)

    # Escape mid-drag puts the path back; between clicks it just stops.
    snap = P.encode_subpath(ctl.doc.path.subpaths[0])
    ctl.set_tool("direct_select")
    a0 = ctl.doc.path.subpaths[0].anchors[0]
    ctl.begin_stroke(a0.x, a0.y)
    ctl.motion_stroke(a0.x + 20, a0.y + 20)
    ctl.cancel_stroke()
    check("Escape during a drag restores the path",
          P.encode_subpath(ctl.doc.path.subpaths[0]) == snap)

    # ---- Curvature Pen ----------------------------------------------------------
    ctl = fresh()
    ctl.set_tool("curvature_pen")
    for x, y in ((10, 50), (40, 20), (70, 50), (100, 20)):
        click(ctl, x, y)
    sp = ctl.doc.path.subpaths[0]
    mid = sp.anchors[1]
    check("curvature pen: the curve passes through every click",
          [(a.x, a.y) for a in sp.anchors] == [(10, 50), (40, 20), (70, 50), (100, 20)])
    check("...with smooth handles computed for you",
          mid.smooth and mid.has_handles and abs((mid.ix + mid.ox) / 2 - mid.x) < 1e-9)
    click(ctl, 40, 20, ALT)
    check("Alt-click turns a point into a corner", not sp.anchors[1].has_handles
          and not ctl.doc.path.subpaths[0].anchors[1].smooth)
    drag(ctl, (70, 50), (70, 70))
    check("dragging a point moves it and the curve reflows",
          (ctl.doc.path.subpaths[0].anchors[2].x, ctl.doc.path.subpaths[0].anchors[2].y)
          == (70.0, 70.0))
    ctl.finish_tool()

    # ---- Freeform Pen -------------------------------------------------------------
    ctl = fresh()
    ctl.set_tool("freeform_pen")
    pts = [(10 + i, 50 + 15 * math.sin(i / 8.0)) for i in range(0, 90)]
    ctl.begin_stroke(*pts[0])
    for p in pts[1:]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*pts[-1])
    sp = ctl.doc.path.subpaths[0]
    check("freeform pen turns a drag into a handful of smooth anchors",
          2 <= len(sp) < 20 and not sp.closed)
    ring = [(60 + 30 * math.cos(t), 50 + 30 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 80)]
    ctl.begin_stroke(*ring[0])
    for p in ring[1:]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*ring[-1])
    check("ending where you started closes it", ctl.doc.path.subpaths[-1].closed)

    # ---- Path Selection / Direct Selection ---------------------------------------
    ctl = fresh()
    ctl.doc.paths = [circle_path(), circle_path(30, 30, 10)]
    ctl.doc.paths[1].name = "small"
    ctl.doc.active_path = 0
    ctl.set_tool("path_select")
    drag(ctl, (50, 50), (60, 55))
    check("path selection drags a whole subpath by its inside",
          (ctl.doc.paths[0].subpaths[0].anchors[0].x,
           ctl.doc.paths[0].subpaths[0].anchors[0].y) == (60.0, 15.0))
    click(ctl, 30, 21)
    check("clicking another path makes it the active one", ctl.doc.active_path == 1)
    drag(ctl, (30, 30), (40, 30), ALT)
    check("Alt-drag duplicates the subpath", len(ctl.doc.paths[1].subpaths) == 2
          and ctl.doc.paths[1].subpaths[1].anchors[0].x == 40.0)
    ctl.set_tool("direct_select")
    a = ctl.doc.path.subpaths[0].anchors[1]
    ax, ay = a.x, a.y
    drag(ctl, (ax, ay), (ax + 3, ay))
    check("direct selection moves a single anchor",
          ctl.doc.path.subpaths[0].anchors[1].x == ax + 3
          and ctl.doc.path.subpaths[0].anchors[0].x == 30.0)

    # ---- fill, stroke, selection ----------------------------------------------
    ctl = fresh(100, 100)
    ctl.doc.paths = [circle_path()]
    ctl.doc.active_path = 0
    cell = ctl.doc.cell(ctl.active_layer)
    ctl.set_primary((200, 10, 10, 255))
    ctl.fill_path()
    painted = cell.plane("rgba")[..., 3]
    check("Fill Path paints the interior on the active layer",
          painted[50, 50] == 255 and painted[2, 2] == 0)
    check("...as one undo step", ctl.history_labels()[-1] == "Fill Path")
    ctl.undo()
    check("...that undoes", cell.plane("rgba")[50, 50, 3] == 0)
    ctl.stroke_path(3)
    check("Stroke Path paints the outline only",
          cell.plane("rgba")[10, 50, 3] > 200 and cell.plane("rgba")[50, 50, 3] == 0)
    ctl.undo()
    ctl.selection.select_rect((0, 0, 50, 100))
    ctl.fill_path()
    check("Fill Path honours the selection",
          cell.plane("rgba")[50, 40, 3] == 255 and cell.plane("rgba")[50, 60, 3] == 0)
    ctl.undo()
    ctl.deselect()
    ctl.path_to_selection()
    check("Path to Selection selects the interior",
          ctl.selection.mask is not None and ctl.selection.mask[50, 50] == 255
          and ctl.selection.mask[2, 2] == 0)
    check("...as an undoable selection change", ctl.history_labels()[-1] == "Path to Selection")
    sel_before = ctl.selection.mask.copy()
    ctl.selection_to_path()
    check("Selection to Path adds a new active path", len(ctl.doc.paths) == 2
          and ctl.doc.active_path == 1)
    ctl.deselect()
    ctl.path_to_selection()
    check("...and selecting that path gives the same selection back",
          np.array_equal(ctl.selection.mask >= 128, sel_before >= 128))

    # Path management.
    ctl.new_path("Mine")
    check("New Path adds and activates an empty path",
          ctl.doc.path.name == "Mine" and ctl.doc.path.is_empty)
    ctl.duplicate_path(0)
    check("Duplicate Path copies and names it", ctl.doc.paths[1].name == "c copy"
          and ctl.doc.active_path == 1)
    ctl.rename_path(1, "Renamed")
    check("Rename Path", ctl.doc.paths[1].name == "Renamed")
    n = len(ctl.doc.paths)
    ctl.delete_path(1)
    check("Delete Path", len(ctl.doc.paths) == n - 1)
    ctl.undo()
    check("...undoably", len(ctl.doc.paths) == n and ctl.doc.paths[1].name == "Renamed")

    # ---- paths follow the image -----------------------------------------------
    ctl = fresh(100, 80)
    ctl.doc.paths = [P.Path("p", [P.Subpath([P.Anchor(10, 20), P.Anchor(30, 40)])])]
    ctl.doc.active_path = 0
    ctl.resize_image(200, 160)
    a = ctl.doc.paths[0].subpaths[0].anchors[1]
    check("resizing the image scales its paths", (a.x, a.y) == (60.0, 80.0))
    ctl.undo()
    a = ctl.doc.paths[0].subpaths[0].anchors[1]
    check("...and undo brings them back", (a.x, a.y) == (30.0, 40.0))
    ctl.selection.select_rect((5, 5, 60, 60))
    ctl.crop_to_selection()
    a = ctl.doc.paths[0].subpaths[0].anchors[0]
    check("cropping moves paths with the content", (a.x, a.y) == (5.0, 15.0))
    ctl.undo()
    ctl.deselect()
    ctl.flip_image(True)
    a = ctl.doc.paths[0].subpaths[0].anchors[0]
    check("flipping mirrors them", (a.x, a.y) == (90.0, 20.0))
    ctl.undo()
    ctl.rotate_image(1)
    a = ctl.doc.paths[0].subpaths[0].anchors[0]
    check("rotating turns them with the pixels", (a.x, a.y) == (60.0, 10.0))

    # ---- the file format --------------------------------------------------------
    ctl = fresh(64, 48)
    ctl.doc.paths = [P.Path("Other"), circle_path(30, 24, 15)]
    ctl.doc.active_path = 1
    f = os.path.join(tempfile.mkdtemp(), "paths.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _sel = ochre_format.load(f)
    check("paths survive save and load, names, active and all",
          [p.name for p in doc2.paths] == ["Other", "c"] and doc2.active_path == 1
          and P.encode_subpath(doc2.paths[1].subpaths[0])
          == P.encode_subpath(ctl.doc.paths[1].subpaths[0]))

    print("\nall path checks passed")


if __name__ == "__main__":
    main()
