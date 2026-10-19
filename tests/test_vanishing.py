#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Vanishing Point: a perspective plane, stamping along it, pasting onto it.

Run: python3 tests/test_vanishing.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import vanishing as V
from ochre.engine.fileio import ochre_format
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# A floor in perspective: wide at the bottom, narrow at the top.
FLOOR = [(70.0, 20.0), (130.0, 20.0), (190.0, 140.0), (10.0, 140.0)]


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh():
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(200, 160, background=(255, 255, 255, 255))
    return ctl


def click(ctl, x, y, mods=0):
    ctl.begin_stroke(x, y, modifiers=mods)
    ctl.end_stroke(x, y, modifiers=mods)


def main():
    # ---- geometry ------------------------------------------------------------------
    check("the floor is a valid plane", V.valid(FLOOR))
    check("a bow-tie is not", not V.valid([FLOOR[0], FLOOR[2], FLOOR[1], FLOOR[3]]))
    check("clicked corners are put in plane order, whatever order they came",
          V.ordered([FLOOR[2], FLOOR[0], FLOOR[3], FLOOR[1]]) == FLOOR)
    uv = V.to_plane(FLOOR, FLOOR)
    w, h = V.edges(FLOOR)
    check("plane corners map to the plane's rectangle",
          np.allclose(uv, [(0, 0), (w, 0), (w, h), (0, h)], atol=1e-6))
    near, far = V.local_scale(FLOOR, 100, 130), V.local_scale(FLOOR, 100, 30)
    check("the plane is bigger near the viewer than far away (%.2f vs %.2f)" % (near, far),
          near > 1.5 * far)

    # ---- the tool: making a plane ------------------------------------------------------
    ctl = fresh()
    lay = ctl.active_layer
    ctl.set_tool("vanishing_point")
    n0 = len(ctl.history_labels())
    for x, y in (FLOOR[2], FLOOR[0], FLOOR[3], FLOOR[1]):
        click(ctl, x, y)
    check("four clicks make a plane, in plane order", ctl.doc.planes == [tuple(FLOOR)],
          str(ctl.doc.planes))
    check("...one undo step", ctl.history_labels()[n0:] == ["Create Plane"])
    prims = ctl.tool.overlay(ctl.doc)
    check("the tool draws the plane and its grid",
          sum(1 for p in prims if p[0] == "line") >= 18 and prims[0][0] == "poly")
    ctl.begin_stroke(*FLOOR[1])
    ctl.motion_stroke(140, 22)
    ctl.end_stroke(140, 24)
    check("dragging a corner adjusts the plane (one undo step)",
          ctl.doc.planes[0][1] == (140.0, 24.0) and ctl.history_labels()[-1] == "Edit Plane")
    ctl.undo()
    check("...undoable", ctl.doc.planes == [tuple(FLOOR)])

    # ---- stamping along the plane ---------------------------------------------------------
    cell = ctl.doc.cell(lay)
    # A red tile near the viewer.
    rgba = cell.plane("rgba")
    rgba[110:130, 90:110] = (255, 0, 0, 255)
    cell._touched(ctl.doc.bounds)
    click(ctl, 100, 120, mods=2)                 # Ctrl-click: the source
    check("Ctrl-click inside the plane sets the source", ctl.tool._source == (100.0, 120.0))
    n1 = len(ctl.history_labels())
    # Paint at the far end, a little at a time.
    ctl.begin_stroke(100, 40)
    for y in (41, 42, 43, 44):
        ctl.motion_stroke(100, y)
    ctl.end_stroke(100, 45)
    # (history_labels lists redo-able entries too; the undo above left one,
    # which this stroke replaced -- so compare the tail, not a count.)
    check("a stamp stroke is one undo step", ctl.history_labels()[-1] == "Vanishing Point"
          and ctl.history_labels()[-2] == "Create Plane", str(ctl.history_labels()[n1 - 2:]))
    img = cell.plane("rgba")
    red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    far_red = red[20:80]
    ys, xs = np.nonzero(far_red)
    check("the tile is copied to the far end", xs.size > 20)
    span_far = xs.max() - xs.min() + 1
    check("...and lands SMALLER there, foreshortened (%d px wide vs 20)" % span_far, span_far < 16)
    ctl.undo()
    check("undo takes the stamp back", not ((cell.plane("rgba")[..., 1] < 60)[20:80]).any())

    # ---- paste in perspective ---------------------------------------------------------------
    pic = np.zeros((40, 40, 4), np.uint8)
    pic[...] = (0, 0, 255, 255)
    placed = ctl.paste_in_perspective(pic)
    shown = ctl.compositor.composite(ctl.doc.bounds)
    blue = shown[..., 2] > 200
    blue &= shown[..., 0] < 60
    rows = [int(blue[y].sum()) for y in (30, 115)]
    check("Paste in Perspective lays the image onto the plane as a smart object",
          placed.kind == "smart" and blue.any())
    check("...wider near the viewer than far away (%d vs %d px)" % (rows[1], rows[0]),
          rows[1] > 1.5 * rows[0] > 0)
    check("...and stays inside the plane",
          all(V.inside(FLOOR, x + 0.5, y + 0.5) for y, x in zip(*np.nonzero(blue))))

    # ---- geometry and files ---------------------------------------------------------------
    ctl.resize_image(400, 320)
    check("resizing the image scales the plane", ctl.doc.planes[0][0] == (140.0, 40.0))
    ctl.undo()
    check("...and undo puts it back", ctl.doc.planes == [tuple(FLOOR)])
    f = os.path.join(tempfile.mkdtemp(), "v.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check(".ochre keeps the plane", doc2.planes == [tuple(FLOOR)], str(doc2.planes))
    check("Delete Perspective Plane is undoable", ctl.clear_planes() and ctl.doc.planes == []
          and (ctl.undo() or True) and ctl.doc.planes == [tuple(FLOOR)])
    print("\nall vanishing-point checks passed")


if __name__ == "__main__":
    main()
