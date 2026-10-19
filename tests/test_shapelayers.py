#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Shape layers (Photoshop's shape layers / GIMP 3.2's vector layers) and the
shape tools' Shape and Path modes.

Run: python3 tests/test_shapelayers.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.compositor import Compositor
from ochre.engine.fileio import ochre_format
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WHITE, RED, BLUE = (255, 255, 255, 255), (255, 0, 0, 255), (0, 0, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def drag(ctl, a, b):
    ctl.begin_stroke(*a)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    ctl.end_stroke(*b)


def px(img, x, y):
    return tuple(int(v) for v in img[y, x])


def main():
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(100, 80, background=WHITE)
    bg = ctl.active_layer
    bg_before = ctl.doc.cell(bg).plane("rgba").copy()
    ctl.set_primary(RED)
    ctl.set_secondary(BLUE)
    ctl.set_tool("shape", mode="shape", shape="rectangle", style="both", stroke=4)
    pos = ctl.history.position
    drag(ctl, (20, 20), (60, 50))
    lay = ctl.active_layer
    img = shown(ctl)
    check("Shape mode makes a shape layer, one undo step",
          lay is not bg and lay.kind == "shape" and ctl.history.position == pos + 1
          and ctl.history_labels()[-1] == "New Shape Layer")
    check("...filled with the secondary colour, stroked with the primary",
          px(img, 40, 35) == BLUE and px(img, 20, 35) == RED and px(img, 5, 5) == WHITE)
    check("...and the layer beneath is untouched", np.array_equal(ctl.doc.cell(bg).plane("rgba"), bg_before))

    ctl.set_active_layer(bg)
    ctl.set_tool("shape", mode="shape", shape="ellipse", style="fill")
    drag(ctl, (60, 10), (90, 40))
    ell = ctl.active_layer
    from ochre.engine import shapelayer
    path = shapelayer.path_of(ell.params)
    check("an ellipse is true curves: four smooth anchors with handles",
          len(path.subpaths[0].anchors) == 4 and all(a.has_handles for a in path.subpaths[0].anchors))
    img = shown(ctl)
    check("...filled with the primary colour, corners of its box empty",
          px(img, 75, 25) == RED and px(img, 61, 11) == WHITE)

    # ---- resize stays sharp -------------------------------------------------------------
    ctl.resize_image(300, 240)
    img = shown(ctl)
    row = img[105, 50:70, 2].astype(int)              # across the rectangle's left edge at x=60
    soft = int(((row > 10) & (row < 245)).sum())
    check("resizing re-draws the shape sharp (%d soft pixels across the edge)" % soft, soft <= 3)
    check("...in its new place", px(img, 120, 105) == BLUE and px(img, 225, 75) == RED)
    ctl.undo()

    # ---- style --------------------------------------------------------------------------
    ctl.set_shape_style(lay, fill=None)
    img = shown(ctl)
    check("a shape's fill can be turned off afterwards", px(img, 40, 35) == WHITE and px(img, 20, 35) == RED)
    ctl.set_shape_style(lay, stroke_align="outside", stroke_width=4)
    img = shown(ctl)
    check("...the stroke moved outside the outline", px(img, 18, 35) == RED and px(img, 22, 35) == WHITE)
    ctl.undo()
    ctl.undo()

    # ---- path mode, line ------------------------------------------------------------------
    n_layers = len(ctl.doc.layers())
    ctl.set_tool("shape", mode="path", shape="star5")
    drag(ctl, (10, 50), (40, 78))
    check("Path mode adds a path, not a layer", len(ctl.doc.layers()) == n_layers
          and ctl.doc.paths and len(ctl.doc.paths[-1].subpaths[0].anchors) == 10)
    ctl.set_tool("line", mode="shape", stroke=3)
    drag(ctl, (5, 70), (95, 70))
    line = ctl.active_layer
    check("a line in Shape mode is a stroked, unfilled shape layer",
          line.kind == "shape" and line.params["fill"] is None and px(shown(ctl), 50, 70) == RED)

    # ---- edit the path ---------------------------------------------------------------------
    ctl.set_active_layer(lay)
    ctl.shape_to_path()
    p = ctl.doc.paths[ctl.doc.active_path]
    p.translate(10, 0)
    ctl.update_shape_from_path()
    img = shown(ctl)
    check("Edit Path, move it, Update: the shape moves", px(img, 30, 35) == RED and px(img, 20, 35) == WHITE)

    # ---- files and rasterize ------------------------------------------------------------------
    before = shown(ctl).copy()
    f = os.path.join(tempfile.mkdtemp(), "s.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check(".ochre keeps shape layers as shapes", doc2.layer(lay.id).kind == "shape"
          and np.array_equal(Compositor(doc2).composite(doc2.bounds), before))
    ctl.save_path(os.path.join(tempfile.mkdtemp(), "s.psd"))
    check("PSD says it keeps them as pixels", any("shape layer" in w for w in ctl.last_save_warnings or []))
    ctl.rasterize_layer(lay)
    check("Rasterize turns a shape into the same pixels", lay.kind == "raster"
          and np.abs(shown(ctl).astype(int) - before.astype(int)).max() <= 1)
    print("\nall shape-layer checks passed")


if __name__ == "__main__":
    main()
