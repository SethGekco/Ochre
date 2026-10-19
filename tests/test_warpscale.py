#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Content-Aware Scale (seam carving) and Puppet Warp (moving least squares).

Run: python3 tests/test_warpscale.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import seamcarve, warp
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=120, h=80):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=(200, 200, 200, 255))
    return ctl


def drag(ctl, a, b):
    ctl.begin_stroke(*a)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    ctl.end_stroke(*b)


def main():
    # ---- seam carving engine ------------------------------------------------------
    img = np.zeros((40, 80, 4), np.uint8)
    img[..., 3] = 255
    img[..., :3] = 180
    img[10:30, 30:50, :3] = (255, 0, 0)
    for new_w in (50, 110):
        cols, rows = seamcarve.maps(img, new_w, 40)
        out = seamcarve.apply_maps(img, cols, rows)
        red = out[..., 1] == 0
        ys, xs = np.nonzero(red)
        check("seam carving to width %d keeps the subject whole (20x20 square intact)" % new_w,
              out.shape == (40, new_w, 4) and red.sum() == 400
              and xs.max() - xs.min() == 19 and ys.max() - ys.min() == 19)
    cols, rows = seamcarve.maps(img, 80, 30)
    check("...and height works the same way, transposed",
          seamcarve.apply_maps(img, cols, rows).shape == (30, 80, 4))
    seam = seamcarve.find_seam(seamcarve.energy(img))
    check("a seam is connected (moves at most one column per row)",
          np.abs(np.diff(seam)).max() <= 1 and len(seam) == 40)

    # ---- Content-Aware Scale through the controller ------------------------------------
    ctl = fresh()
    top = ctl.add_layer("top")
    ctl.doc.cell(top).fill(Rect(50, 20, 20, 20), (0, 0, 255, 255))
    ctl.content_aware_scale(80, 80)
    check("Content-Aware Scale resizes the canvas", (ctl.doc.width, ctl.doc.height) == (80, 80))
    cells = [c for (lid, fid), c in ctl.doc.cells.items()]
    check("...every layer comes along, the same size",
          all(c.width == 80 and c.height == 80 for c in cells))
    blue = ctl.doc.cell(top).plane("rgba")[..., 2] == 255
    ys, xs = np.nonzero(blue)
    check("...and the subject keeps its shape (seams went round it)",
          blue.sum() == 400 and xs.max() - xs.min() == 19)
    ctl.undo()
    check("undo restores the original size and pixels", ctl.doc.width == 120
          and ctl.doc.cell(top).plane("rgba")[30, 60, 2] == 255)
    # Protecting a region with the selection.
    ctl = fresh(100, 40)
    cell = ctl.doc.cell(ctl.active_layer)
    rng = np.random.default_rng(0)
    cell.plane("rgba")[..., :3] = rng.integers(0, 256, (40, 100, 3), dtype=np.uint8)
    cell._touched(ctl.doc.bounds)
    keep = cell.plane("rgba")[:, 10:30].copy()
    ctl.selection.select_rect((10, 0, 30, 40))
    ctl.content_aware_scale(70, 40, protect=True)
    now = ctl.doc.cell(ctl.active_layer).plane("rgba")
    found = any(np.array_equal(now[:, x:x + 20], keep) for x in range(0, 51))
    check("a protected selection comes through uncut", found)

    # ---- MLS engine -----------------------------------------------------------------------
    pins = [(10.0, 10.0), (60.0, 10.0), (35.0, 50.0)]
    pts = np.array([[20.0, 20.0], [40.0, 30.0]])
    check("MLS with unmoved pins is the identity", np.allclose(warp.mls(pts, pins, pins), pts))
    moved = [(15.0, 12.0), (65.0, 12.0), (40.0, 52.0)]
    check("...moving every pin the same way just translates",
          np.allclose(warp.mls(pts, pins, moved), pts + [5, 2]))
    check("...and pins land exactly where they are dragged",
          np.allclose(warp.mls(np.array(pins), pins, [(10, 10), (60, 30), (35, 50)]),
                      [(10, 10), (60, 30), (35, 50)]))
    rot = [(10.0, 10.0), (10.0, 60.0)]                 # rotate the pair 90 degrees about pin 0
    out = warp.mls(np.array([[35.0, 10.0]]), [(10.0, 10.0), (60.0, 10.0)], rot, rigid=True)
    check("rigid MLS rotates without scaling", np.allclose(out, [[10.0, 35.0]], atol=1e-6))

    # ---- Puppet Warp tool -----------------------------------------------------------------------
    ctl = fresh(120, 80)
    lay = ctl.add_layer("limb")
    ctl.doc.cell(lay).fill(Rect(20, 35, 80, 10), (200, 30, 30, 255))
    before = ctl.doc.cell(lay).plane("rgba").copy()
    ctl.set_tool("puppet_warp")
    ctl.begin_stroke(25, 40)
    ctl.end_stroke(25, 40)
    ctl.begin_stroke(60, 40)
    ctl.end_stroke(60, 40)
    check("clicks place pins (nothing moves yet)", len(ctl.tool._pins) == 2
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), before))
    drag(ctl, (95, 40), (95, 40))
    drag(ctl, (95, 40), (95, 15))
    px = ctl.doc.cell(lay).plane("rgba")
    check("dragging a pin bends the shape: its end rises", px[16:22, 90:100, 3].max() > 0
          and px[38:42, 92:98, 3].max() == 0)
    check("...while the pinned end stays put", px[40, 22, 3] == 255)
    n0 = len(ctl.history_labels())
    ctl.finish_tool()
    check("Enter applies it as ONE step", ctl.history_labels()[n0:] == ["Puppet Warp"])
    ctl.undo()
    check("...that undoes exactly", np.array_equal(ctl.doc.cell(lay).plane("rgba"), before))
    ctl.set_tool("puppet_warp")
    ctl.begin_stroke(25, 40)
    ctl.end_stroke(25, 40)
    drag(ctl, (60, 40), (60, 20))
    ctl.cancel_stroke()
    check("Escape puts everything back", np.array_equal(ctl.doc.cell(lay).plane("rgba"), before)
          and not ctl.tool.active)
    print("\nall warp and scale checks passed")


if __name__ == "__main__":
    main()
