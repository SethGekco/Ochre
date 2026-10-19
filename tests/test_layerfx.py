#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Layer masks, clipping masks, adjustment and fill layers, layer styles.

Checked on what the COMPOSITE shows -- the pixels a user sees -- and
through the controller, with undo, save and load, because each of these is
a promise about appearance that the layer's own pixels do not record.

Run: python3 tests/test_layerfx.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.fileio import ochre_format
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RED = (220, 20, 20, 255)
WHITE = (255, 255, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=80, h=60):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=WHITE)
    return ctl


def shown(ctl):
    return ctl.compositor.composite(ctl.doc.bounds)


def px(ctl, x, y):
    return tuple(int(v) for v in shown(ctl)[y, x])


def drag(ctl, a, b):
    ctl.begin_stroke(*a)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    ctl.end_stroke(*b)


def main():
    # ---- layer masks --------------------------------------------------------
    ctl = fresh()
    top = ctl.add_layer("top")
    ctl.doc.cell(top).fill(ctl.doc.bounds, RED)
    ctl.compositor.invalidate()
    check("a red layer covers the canvas", px(ctl, 10, 10) == RED)
    ctl.add_layer_mask("hide")
    check("a Hide All mask hides the layer", px(ctl, 10, 10) == WHITE)
    check("...and the new mask is what you paint on", ctl.editing_mask
          and ctl.paint_target is top.mask)
    ctl.set_tool("brush", size=12, hardness=1.0)
    ctl.set_primary(WHITE)
    drag(ctl, (40, 30), (41, 30))
    ctl.compositor.invalidate()
    check("PAINTING WHITE ON THE MASK REVEALS THE LAYER THERE",
          px(ctl, 40, 30) == RED and px(ctl, 5, 5) == WHITE)
    check("...without touching the layer's own pixels",
          tuple(ctl.doc.cell(top).plane("rgba")[5, 5]) == RED)
    ctl.undo()
    ctl.compositor.invalidate()
    check("undo takes the mask stroke back", px(ctl, 40, 30) == WHITE)
    ctl.redo()
    ctl.set_mask_enabled(False)
    ctl.compositor.invalidate()
    check("disabling the mask shows the whole layer", px(ctl, 5, 5) == RED)
    ctl.set_mask_enabled(True)
    ctl.delete_layer_mask(apply=True)
    ctl.compositor.invalidate()
    check("Apply Layer Mask bakes it into the layer's alpha",
          top.mask is None and px(ctl, 40, 30) == RED and px(ctl, 5, 5) == WHITE
          and ctl.doc.cell(top).plane("rgba")[5, 5, 3] == 0)
    ctl.undo()
    ctl.compositor.invalidate()
    check("...one undo restores the mask and the pixels", top.mask is not None
          and ctl.doc.cell(top).plane("rgba")[5, 5, 3] == 255 and px(ctl, 5, 5) == WHITE)
    ctl.selection.select_rect((0, 0, 40, 60))
    ctl.delete_layer_mask()
    ctl.add_layer_mask("selection")
    ctl.compositor.invalidate()
    check("a mask From Selection shows the layer only inside it",
          px(ctl, 10, 10) == RED and px(ctl, 60, 10) == WHITE)
    ctl.deselect()

    # Masks travel with layers: delete and undo, duplicate.
    ctl.remove_layer(top)
    ctl.undo()
    ctl.compositor.invalidate()
    top = ctl.doc.layer(top.id)
    check("deleting a masked layer and undoing brings the mask back too",
          top.mask is not None and px(ctl, 60, 10) == WHITE and px(ctl, 10, 10) == RED)
    dup = ctl.duplicate_layer(top)
    check("duplicating copies the mask", dup.mask is not None and
          np.array_equal(ctl.doc.cell(dup.mask).plane("rgba"), ctl.doc.cell(top.mask).plane("rgba")))

    # ---- save and load -----------------------------------------------------------
    top.styles = {"stroke": {"size": 2, "color": (0, 0, 255)}}
    f = os.path.join(tempfile.mkdtemp(), "fx.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    t2 = doc2.layer(top.id)
    check("a mask survives save and load", t2.mask is not None and np.array_equal(
        doc2.cell(t2.mask).plane("rgba"), ctl.doc.cell(top.mask).plane("rgba")))
    check("...and so do layer styles", t2.styles.get("stroke", {}).get("color") == (0, 0, 255)
          and t2.styles["stroke"].get("size") == 2)

    # ---- clipping masks -------------------------------------------------------------
    ctl = fresh()
    base = ctl.add_layer("base")
    ctl.doc.cell(base).fill(Rect(10, 10, 20, 20), (0, 0, 255, 255))
    over = ctl.add_layer("over")
    ctl.doc.cell(over).fill(ctl.doc.bounds, RED)
    ctl.toggle_clipping(over)
    ctl.compositor.invalidate()
    check("a clipped layer shows only where the layer below has pixels",
          px(ctl, 20, 20) == RED and px(ctl, 50, 40) == WHITE)
    ctl.undo()
    ctl.compositor.invalidate()
    check("Release Clipping Mask (undo) shows it everywhere again", px(ctl, 50, 40) == RED)
    ctl.redo()
    ctl.set_active_layer(base)
    ctl.set_tool("pencil", size=4)
    ctl.set_primary((0, 255, 0, 255))
    drag(ctl, (50, 40), (52, 40))
    ctl.flush()
    check("painting on the base reveals more of the clipped layer, live",
          tuple(ctl.display[40, 51]) == RED)

    # ---- adjustment layers -----------------------------------------------------------
    ctl = fresh()
    pic = ctl.add_layer("pic")
    ctl.doc.cell(pic).fill(ctl.doc.bounds, (200, 100, 50, 255))
    adj = ctl.add_adjustment_layer("invert")
    ctl.compositor.invalidate()
    check("an Invert adjustment layer inverts everything beneath it",
          px(ctl, 5, 5)[:3] == (55, 155, 205))
    check("...without changing the layer below", tuple(ctl.doc.cell(pic).plane("rgba")[5, 5])
          == (200, 100, 50, 255))
    check("...and is one undo step", ctl.history_labels()[-1] == "New Adjustment Layer")
    ctl.set_layer_property(adj, "opacity", 128)
    ctl.compositor.invalidate()
    check("its opacity blends the adjustment", abs(px(ctl, 5, 5)[0] - 128) <= 2)
    ctl.set_layer_property(adj, "opacity", 255)
    ctl.set_active_layer(adj)
    ctl.set_editing_mask(True)
    ctl.set_tool("bucket")
    ctl.set_primary((0, 0, 0, 255))
    ctl.selection.select_rect((0, 0, 40, 60))
    ctl.begin_stroke(5, 5)
    ctl.end_stroke(5, 5)
    ctl.deselect()
    ctl.compositor.invalidate()
    check("its mask limits where it applies", px(ctl, 5, 5)[:3] == (200, 100, 50)
          and px(ctl, 60, 5)[:3] == (55, 155, 205))
    bc = ctl.add_adjustment_layer("brightness_contrast", {"brightness": 20})
    before = px(ctl, 60, 5)
    ctl.set_layer_params(bc, {"brightness": 60})
    ctl.compositor.invalidate()
    check("an adjustment layer stays EDITABLE", px(ctl, 60, 5)[0] > before[0])
    # Painting under an adjustment layer is adjusted live (the cache must not
    # flatten it above the active layer).
    ctl.set_active_layer(pic)
    ctl.set_tool("pencil", size=3)
    ctl.set_primary((0, 0, 0, 255))
    drag(ctl, (70, 50), (71, 50))
    ctl.flush()
    check("paint beneath an adjustment layer is adjusted as you paint",
          tuple(ctl.display[50, 70])[:3] != (0, 0, 0))
    ctl.merge_down(bc)
    ctl.compositor.invalidate()
    check("merging an adjustment layer down applies it to the layer below",
          ctl.doc.layer(bc.id) is None)

    # ---- fill layers -------------------------------------------------------------------
    ctl = fresh()
    ctl.set_primary((10, 200, 10, 255))
    fill = ctl.add_fill_layer("solid")
    ctl.compositor.invalidate()
    check("a solid fill layer paints the canvas at any size", px(ctl, 79, 59) == (10, 200, 10, 255))
    ctl.set_layer_params(fill, {"color": (0, 0, 0, 255)})
    ctl.compositor.invalidate()
    check("...and stays editable", px(ctl, 5, 5) == (0, 0, 0, 255))
    grad = ctl.add_fill_layer("gradient", start=(0, 0, 0, 255), end=(255, 255, 255, 255), angle=0)
    ctl.compositor.invalidate()
    check("a gradient fill runs across the canvas", px(ctl, 2, 30)[0] < 30 and px(ctl, 77, 30)[0] > 225)
    ctl.resize_image(160, 120)
    ctl.compositor.invalidate()
    check("fill layers regenerate at the new size, never blurred",
          px(ctl, 2, 60)[0] < 20 and px(ctl, 157, 60)[0] > 235)

    # ---- layer styles -------------------------------------------------------------------
    ctl = fresh(100, 80)
    shape = ctl.add_layer("shape")
    ctl.doc.cell(shape).fill(Rect(30, 20, 30, 30), (0, 0, 255, 255))
    ctl.set_layer_styles(shape, {"stroke": {"size": 3, "color": (255, 0, 0), "position": "outside"}})
    ctl.compositor.invalidate()
    check("a Stroke style outlines the layer's shape",
          px(ctl, 28, 35)[:3] == (255, 0, 0) and px(ctl, 40, 35)[:3] == (0, 0, 255)
          and px(ctl, 20, 35)[:3] == (255, 255, 255))
    check("...without touching its pixels", ctl.doc.cell(shape).plane("rgba")[35, 28, 3] == 0)
    ctl.set_layer_styles(shape, {"drop_shadow": {"distance": 6, "size": 0, "angle": 135,
                                                 "opacity": 100}})
    ctl.compositor.invalidate()
    check("a Drop Shadow falls down-right of the shape, behind it",
          px(ctl, 62, 52)[0] < 60 and px(ctl, 25, 15)[:3] == (255, 255, 255)
          and px(ctl, 40, 30)[:3] == (0, 0, 255))
    ctl.set_layer_styles(shape, {"color_overlay": {"color": (0, 255, 0)}})
    ctl.compositor.invalidate()
    check("a Color Overlay recolours the shape only", px(ctl, 40, 30)[:3] == (0, 255, 0)
          and px(ctl, 10, 10)[:3] == (255, 255, 255))
    ctl.set_layer_styles(shape, {"outer_glow": {"size": 6, "color": (255, 200, 0), "opacity": 100},
                                 "inner_glow": {"size": 6}, "inner_shadow": {},
                                 "bevel": {}, "gradient_overlay": {"opacity": 50}})
    ctl.compositor.invalidate()
    img = shown(ctl)
    check("glows, inner shadow, bevel and gradient overlay all render",
          img[35, 26, 0] > img[35, 26, 2] and not np.array_equal(img[35, 40], [0, 0, 255, 255]))
    ctl.set_active_layer(shape)
    ctl.set_layer_styles(shape, {"stroke": {"size": 2, "color": (255, 0, 0)}})
    ctl.set_tool("pencil", size=4)
    ctl.set_primary((0, 0, 255, 255))
    drag(ctl, (80, 70), (82, 70))
    ctl.flush()
    check("styles follow the layer AS YOU PAINT (stroke round the new paint)",
          tuple(ctl.display[70, 77])[:3] == (255, 0, 0))
    ctl.undo()
    ctl.rasterize_layer(shape)
    ctl.compositor.invalidate()
    check("Rasterize Layer bakes the style into pixels",
          not shape.styles and tuple(ctl.doc.cell(shape).plane("rgba")[35, 29])[:3] == (255, 0, 0))
    ctl.undo()
    check("...undoably", shape.styles.get("stroke") is not None)

    # ---- the new blend modes reach the layers dock's list ----------------------------
    names = [m.name for m in ctl.blends.ordered()]
    check("Soft Light, Hue, Luminosity and the rest are offered on layers",
          all(n in names for n in ("SoftLight", "HardLight", "Hue", "Luminosity",
                                   "Divide", "Exclusion", "LinearBurn")))
    print("\nall layer-effects checks passed")


if __name__ == "__main__":
    main()
