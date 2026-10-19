#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Smart objects: lossless transforms, smart filters, placing, saving.

Run: python3 tests/test_smartobjects.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.fileio import ochre_format, psd
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=100, h=80):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    return ctl


def stripes(ctl, layer):
    c = ctl.doc.cell(layer)
    c.plane("rgba")[20:60, 30:70] = (200, 50, 50, 255)
    c.plane("rgba")[20:60:2, 30:70, :3] = 0              # one-pixel detail
    c._touched(ctl.doc.bounds)
    return c.plane("rgba").copy()


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def drag(ctl, a, b):
    ctl.begin_stroke(*a)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    ctl.end_stroke(*b)


def scale_down_up(ctl, corner, origin):
    """Scale Tool: drag a corner halfway to the opposite one, apply; drag it
    back, apply. The box is the layer's own (its paint, or a smart object's
    whole source)."""
    x, y = corner
    ox, oy = origin
    mid = ((x + ox) / 2.0, (y + oy) / 2.0)
    ctl.set_tool("scale_tool")
    drag(ctl, (x, y), mid)
    ctl.finish_tool()
    ctl.set_tool("scale_tool")
    drag(ctl, mid, (x, y))
    ctl.finish_tool()


def main():
    # ---- convert, and the lossless round trip --------------------------------
    ctl = fresh()
    lay = ctl.add_layer("obj")
    orig = stripes(ctl, lay)
    n0 = len(ctl.history_labels())
    ctl.convert_to_smart(lay)
    check("Convert to Smart Object looks exactly the same", np.array_equal(shown(ctl), orig))
    check("...is one undo step", ctl.history_labels()[n0:] == ["Convert to Smart Object"])
    check("...and keeps the original as the source", lay.kind == "smart" and lay.smart_source is not None)
    scale_down_up(ctl, (100, 80), (0, 0))         # a converted object's source is the canvas
    smart_after = shown(ctl)
    # The same round trip on an ordinary layer, for comparison.
    ctl2 = fresh()
    raster = ctl2.add_layer("raster")
    stripes(ctl2, raster)
    scale_down_up(ctl2, (70, 60), (30, 20))       # a raster layer's box is its paint
    raster_after = shown(ctl2)
    err_smart = np.abs(smart_after.astype(int) - orig.astype(int)).mean()
    err_raster = np.abs(raster_after.astype(int) - orig.astype(int)).mean()
    check("SCALING A SMART OBJECT DOWN AND BACK UP LOSES NOTHING (error %.2f vs raster %.2f)"
          % (err_smart, err_raster), err_smart < 0.5 and err_raster > 5 * max(err_smart, 0.1))
    ctl.undo()
    ctl.undo()
    check("transforms of a smart object undo", np.array_equal(shown(ctl), orig))
    ctl.undo()
    check("...and so does the conversion", lay.kind == "raster"
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig))

    # ---- smart filters ---------------------------------------------------------------
    ctl.redo()
    ctl.apply_effect("invert")
    check("a filter on a smart object becomes a SMART FILTER (listed, not baked)",
          [f["name"] for f in lay.params["filters"]] == ["invert"]
          and np.array_equal(ctl.doc.cell(lay.smart_source).plane("rgba"), orig))
    inv = shown(ctl)
    check("...and is applied", tuple(inv[31, 40, :3]) == (55, 205, 205))
    ctl.apply_effect("gaussian_blur", {"radius": 3})
    filters = [dict(f) for f in lay.params["filters"]]
    filters[0]["enabled"] = False
    ctl.set_smart_filters(filters)
    check("smart filters can be switched off individually", tuple(shown(ctl)[31, 40, :3]) != (55, 205, 205)
          and shown(ctl)[31, 40, 0] > 60)
    ctl.set_smart_filters([])
    check("...or removed, leaving the original", np.array_equal(shown(ctl), orig))

    # ---- geometry, save, load, rasterize -----------------------------------------------
    ctl.resize_image(200, 160)
    big = shown(ctl)
    col = big[40:120, 100, 0].astype(int)
    check("resizing the image re-renders the smart object from its source (stripes survive)",
          big.shape == (160, 200, 4) and col.max() - col.min() > 100)
    ctl.undo()
    f = os.path.join(tempfile.mkdtemp(), "smart.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    l2 = doc2.layer(lay.id)
    check("smart objects save and load: source, matrix and filters",
          l2.kind == "smart" and l2.smart_source is not None
          and np.array_equal(doc2.cell(l2.smart_source).plane("rgba"), orig))
    warn = psd.save(ctl.doc, os.path.join(tempfile.mkdtemp(), "s.psd"))
    check("a PSD keeps it as pixels and says so", any("smart object" in w for w in warn))
    dup = ctl.duplicate_layer(lay)
    check("duplicating copies the source", dup.smart_source is not None and dup.kind == "smart")
    ctl.rasterize_layer(lay)
    check("Rasterize makes it ordinary pixels", lay.kind == "raster" and lay.smart_source is None
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig))

    # ---- place embedded: full resolution inside ----------------------------------------
    ctl = fresh(100, 80)
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 256, (400, 500, 4), dtype=np.uint8)
    photo[..., 3] = 255
    placed = ctl.place_embedded(photo, "photo")
    check("Place Embedded fits a big image to the canvas as a smart object",
          placed.kind == "smart" and ctl.doc.cell(placed.smart_source).plane("rgba").shape == (400, 500, 4))
    ctl.set_tool("scale_tool")
    q = None
    ctl.begin_stroke(50, 40)
    q = ctl.tool._quad.copy()
    ctl.end_stroke(50, 40)
    ctl.finish_tool()
    check("...its transform box is the placed image's corners",
          abs(q[0][0] - 0) < 1e-6 and abs(q[2][0] - 100) < 1e-6)

    # ---- edit contents: the original in its own tab, saved back -------------------------
    parent_doc = ctl.doc
    before = shown(ctl).copy()
    n_hist = len(ctl.history_labels())
    contents = ctl.edit_smart_contents(placed)
    check("Edit Contents opens the original in a NEW tab at its own size",
          contents is ctl.doc and len(ctl.documents) == 2 and (contents.width, contents.height) == (500, 400)
          and np.array_equal(ctl.doc.cell(ctl.active_layer).plane("rgba"), photo))
    check("...titled after the layer, and a contents tab is never 'pristine' (not silently replaced)",
          "photo (contents)" in ctl.title and not ctl._pristine())
    ctl.doc.cell(ctl.active_layer).fill(ctl.doc.bounds, (255, 0, 0, 255))
    ctl.compositor.invalidate()
    check("the parent is untouched while the contents are being edited",
          np.array_equal(parent_doc.cell(placed.smart_source).plane("rgba"), photo))
    check("Update Smart Object writes the contents back", ctl.update_smart_link())
    check("...the contents tab stays in front and counts as saved",
          ctl.doc is contents and not ctl.is_modified())
    ctl.switch_document(0)
    img = shown(ctl)
    check("...and the parent now shows the new contents, through the same transform",
          (img[..., 0] == 255).all() and (img[..., 1] == 0).all()
          and ctl.history_labels()[n_hist:] == ["Replace Contents"])
    ctl.undo()
    check("...as one undo step in the parent", np.array_equal(shown(ctl), before))
    ctl.close_document(0)
    check("updating after the parent was closed is refused, not crashed",
          ctl.doc is contents and ctl.update_smart_link() is False)
    print("\nall smart-object checks passed")


if __name__ == "__main__":
    main()
