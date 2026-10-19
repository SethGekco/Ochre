#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Patterns, Edit > Fill / Stroke, Color Range, Focus Area, Trim, Duplicate,
Apply Image, Calculations, vector masks, Satin and Pattern Overlay.

Run: python3 tests/test_editcmds.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import paths as P
from ochre.engine import patterns
from ochre.engine.compositor import Compositor
from ochre.engine.fileio import ochre_format
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WHITE, RED, BLUE = (255, 255, 255, 255), (255, 0, 0, 255), (0, 0, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=80, h=60, bg=WHITE, state=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state or tempfile.mkdtemp())
    ctl.new_document(w, h, background=bg)
    return ctl


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def px(img, x, y):
    return tuple(int(v) for v in img[y, x])


def rect_path(x0, y0, x1, y1):
    return P.Path("r", [P.Subpath([P.Anchor(x0, y0), P.Anchor(x1, y0), P.Anchor(x1, y1),
                                   P.Anchor(x0, y1)], True)])


def main():
    # ---- patterns ---------------------------------------------------------------------
    ctl = fresh()
    lay = ctl.active_layer
    cell = ctl.doc.cell(lay)
    cell.plane("rgba")[:10, :10] = RED
    cell.plane("rgba")[:10, 10:20] = BLUE
    cell._touched(ctl.doc.bounds)
    ctl.selection.select_rect((0, 0, 20, 10))
    name = ctl.define_pattern("redblue")
    check("Define Pattern saves the selected area as a pattern",
          name == "redblue" and "redblue" in patterns.names(ctl.state_dir))
    t = patterns.tile("redblue", ctl.state_dir)
    check("...exactly as it was", t.shape == (10, 20, 4) and tuple(t[5, 5]) == RED and tuple(t[5, 15]) == BLUE)
    check("tiles survive base64 embedding", np.array_equal(patterns.decode_tile(patterns.encode_tile(t)), t))
    a = patterns.fill(t, Rect(0, 0, 40, 10))
    b = patterns.fill(t, Rect(20, 0, 20, 10))
    check("pattern fills are anchored to the canvas, so neighbouring fills join",
          np.array_equal(a[:, 20:], b))
    ctl.selection.select_all()

    # ---- pattern fill layer, embedded --------------------------------------------------
    fl = ctl.add_fill_layer("pattern", pattern="redblue")
    img = shown(ctl)
    check("a pattern fill layer repeats the pattern over the canvas",
          px(img, 25, 25) == RED and px(img, 35, 25) == BLUE and px(img, 45, 45) == RED)
    f = os.path.join(tempfile.mkdtemp(), "p.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check("...and looks the same where the pattern was never defined (tile embedded)",
          np.array_equal(Compositor(doc2).composite(doc2.bounds), img))
    ctl.remove_layer(fl)

    # ---- Edit > Fill ---------------------------------------------------------------------
    ctl = fresh()
    lay = ctl.active_layer
    ctl.selection.select_rect((10, 10, 30, 30))
    n = len(ctl.history_labels())
    ctl.set_primary(RED)
    ctl.fill_with("foreground")
    img = shown(ctl)
    check("Fill with the foreground fills the selection only",
          px(img, 20, 20) == RED and px(img, 5, 5) == WHITE and ctl.history_labels()[n:] == ["Fill"])
    ctl.undo()
    ctl.fill_with("color", colour=(0, 0, 0, 255), mode="Normal", opacity=50)
    check("...at 50% opacity", abs(px(shown(ctl), 20, 20)[0] - 128) <= 1)
    ctl.undo()
    ctl.fill_with("pattern", pattern="checker", scale=100)
    region = shown(ctl)[10:30, 10:30, 0]
    check("...with a pattern", len(np.unique(region)) == 2)
    ctl.undo()
    ctl.selection.select_all()
    top = ctl.add_layer("top")
    tc = ctl.doc.cell(top)
    tc.plane("rgba")[20:40, 20:40] = BLUE
    tc._touched(ctl.doc.bounds)
    ctl.fill_with("color", colour=RED, preserve=True)
    tpx = tc.plane("rgba")
    check("...preserving transparency: only existing pixels change colour",
          tuple(tpx[30, 30]) == RED and tpx[5, 5, 3] == 0)

    # ---- Edit > Stroke --------------------------------------------------------------------
    ctl = fresh()
    ctl.selection.select_rect((20, 20, 60, 50))
    ctl.set_primary((0, 0, 0, 255))
    for loc, inside_px, outside_px in (("inside", (22, 35), (18, 35)),
                                       ("outside", (18, 35), (22, 35)),
                                       ("center", (20, 35), (25, 35))):
        ctl.stroke_selection(4, location=loc)
        img = shown(ctl)
        check("Stroke %s: the band lies %s the edge" % (loc, loc),
              px(img, *inside_px)[0] == 0 and px(img, *outside_px)[0] == 255
              and px(img, 40, 35) == WHITE, str((px(img, *inside_px), px(img, *outside_px))))
        ctl.undo()

    # ---- Color Range and Focus Area ---------------------------------------------------------
    ctl = fresh(bg=(30, 120, 30, 255))
    lay = ctl.active_layer
    c = ctl.doc.cell(lay).plane("rgba")
    c[:, 40:] = (220, 40, 40, 255)
    c[45:, :] = (240, 240, 240, 255)
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    ctl.set_primary((220, 40, 40, 255))
    ctl.select_color_range("sampled", fuzziness=30)
    m = ctl.selection.mask
    check("Color Range (sampled) selects that colour and nothing else",
          m[20, 60] == 255 and m[20, 10] == 0 and m[50, 10] == 0)
    ctl.select_color_range("greens")
    check("...a hue family", ctl.selection.mask[20, 10] == 255 and ctl.selection.mask[20, 60] == 0)
    ctl.select_color_range("highlights")
    check("...a tonal range", ctl.selection.mask[50, 10] == 255 and ctl.selection.mask[20, 10] == 0)
    ctl.select_color_range("greens", invert=True)
    check("...inverted", ctl.selection.mask[20, 10] == 0 and ctl.selection.mask[20, 60] == 255)
    skin = np.zeros((4, 8, 4), np.uint8)
    skin[:, :4] = (224, 172, 140, 255)
    skin[:, 4:] = (40, 90, 200, 255)
    from ochre.engine import selectrange
    sm = selectrange.color_range(skin, "skin")
    check("...skin tones", sm[0, 0] > 200 and sm[0, 6] == 0)

    rng = np.random.default_rng(2)
    tex = rng.integers(0, 256, (60, 120, 4), dtype=np.uint8)
    tex[..., 3] = 255
    from ochre.engine.effects import EffectRegistry
    tex[:, 60:] = EffectRegistry().create("gaussian_blur", radius=6).apply(tex)[:, 60:]
    ctl = fresh(120, 60)
    ctl.doc.cell(ctl.active_layer).plane("rgba")[...] = tex
    ctl.doc.cell(ctl.active_layer)._touched(ctl.doc.bounds)
    ctl.select_focus_area(50)
    m = ctl.selection.mask
    check("Focus Area selects the sharp half, not the blurred one (%d vs %d)"
          % (m[:, 10:50].mean(), m[:, 70:110].mean()), m[:, 10:50].mean() > 200 > 40 > m[:, 70:110].mean())

    # ---- Trim -----------------------------------------------------------------------------------
    ctl = fresh(bg=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[15:35, 20:50] = RED
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    check("Trim cuts transparent borders to the content", ctl.trim("transparent")
          and (ctl.doc.width, ctl.doc.height) == (30, 20))
    check("...and again there is nothing to trim", ctl.trim("transparent") is False)
    ctl.undo()
    check("...undoable", (ctl.doc.width, ctl.doc.height) == (80, 60))
    ctl = fresh(bg=(9, 9, 9, 255))
    ctl.doc.cell(ctl.active_layer).plane("rgba")[10:20, 10:30] = RED
    ctl.trim("top_left", bottom=False, right=False)
    check("Trim by the top-left colour, only top and left", (ctl.doc.width, ctl.doc.height) == (70, 50))

    # ---- Duplicate ------------------------------------------------------------------------------
    ctl = fresh()
    ctl.add_layer("second")
    original = ctl.doc
    copy = ctl.duplicate_document()
    check("Image > Duplicate opens a copy in a new tab", copy is ctl.doc and copy is not original
          and len(ctl.documents) == 2 and len(copy.layers()) == 2)
    ctl.doc.cell(copy.layers()[0]).plane("rgba")[...] = RED
    check("...independent of the original", tuple(original.cells[(original.layers()[0].id,
                                                                  original.frame.id)].plane("rgba")[0, 0]) == WHITE)
    flat = ctl.duplicate_document(flatten=True)
    check("...or flattened to one layer", len(flat.layers()) == 1)

    # ---- Apply Image, Calculations ------------------------------------------------------------------
    ctl = fresh()
    base = ctl.active_layer
    ctl.doc.cell(base).plane("rgba")[...] = (200, 100, 50, 255)
    over = ctl.add_layer("over")
    ctl.doc.cell(over).plane("rgba")[...] = (128, 128, 128, 255)
    ctl.doc.cell(over)._touched(ctl.doc.bounds)
    ctl.set_active_layer(base)
    ctl.apply_image(over.id, "rgb", mode="Multiply")
    v = px(ctl.doc.cell(base).plane("rgba"), 5, 5)
    check("Apply Image: a layer multiplied onto the active one", abs(v[0] - 100) <= 1 and abs(v[1] - 50) <= 1)
    ctl.undo()
    ctl.apply_image(base.id, "red", mode="Normal")
    check("...one channel, as grey", px(ctl.doc.cell(base).plane("rgba"), 5, 5)[:3] == (200, 200, 200))
    ctl.undo()
    ch = ctl.calculations(base.id, "red", False, base.id, "green", False, "Multiply", 100, "channel")
    val = int(ctl.doc.cell(ch).plane("rgba")[5, 5, 0])
    check("Calculations: red x green into a new channel (%d)" % val,
          ch in ctl.doc.channels and abs(val - 200 * 100 // 255) <= 1)
    ctl.calculations(base.id, "red", False, base.id, "red", True, "Normal", 100, "selection")
    check("...or into the selection", ctl.selection.mask is not None
          and int(ctl.selection.mask[5, 5]) == 200)

    # ---- vector masks ---------------------------------------------------------------------------
    ctl = fresh(bg=WHITE)
    top = ctl.add_layer("top")
    ctl.doc.cell(top).plane("rgba")[...] = RED
    ctl.doc.cell(top)._touched(ctl.doc.bounds)
    ctl.doc.paths = [rect_path(20, 10, 60, 50)]
    ctl.doc.active_path = 0
    n = len(ctl.history_labels())
    ctl.add_vector_mask("path")
    img = shown(ctl)
    check("a vector mask from the active path shows the layer only inside it",
          px(img, 40, 30) == RED and px(img, 10, 30) == WHITE and len(ctl.history_labels()) == n + 1)
    check("...and is a copy: editing the path afterwards does not move it",
          top.vector_mask is not ctl.doc.paths[0])
    ctl.resize_image(160, 120)
    img = shown(ctl)
    check("it follows a resize exactly (resolution-independent)",
          px(img, 41, 30) == RED and px(img, 38, 30) == WHITE and px(img, 118, 98) == RED
          and px(img, 122, 102) == WHITE)
    ctl.undo()
    check("...and undo brings back the original size and mask",
          px(shown(ctl), 21, 11) == RED and px(shown(ctl), 18, 30) == WHITE)
    f = os.path.join(tempfile.mkdtemp(), "v.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check(".ochre keeps vector masks", np.array_equal(Compositor(doc2).composite(doc2.bounds), shown(ctl)))
    ctl.set_layer_property(top, "vector_mask_enabled", False)
    check("disabled, the whole layer shows", px(shown(ctl), 10, 30) == RED)
    ctl.undo()
    before = shown(ctl).copy()
    ctl.delete_vector_mask(apply=True)
    check("Rasterize Vector Mask moves it into the layer mask, same picture",
          top.vector_mask is None and top.mask is not None
          and np.abs(shown(ctl).astype(int) - before.astype(int)).max() <= 1)
    ctl.undo()
    check("...undoable", top.vector_mask is not None and top.mask is None)
    ctl.add_vector_mask("hide")
    check("Hide All hides everything", px(shown(ctl), 40, 30) == WHITE)

    # ---- Satin and Pattern Overlay ----------------------------------------------------------------
    from ochre.engine import layerstyles
    shape = np.zeros((40, 40, 4), np.uint8)
    shape[8:32, 8:32] = (200, 200, 200, 255)
    _b, front = layerstyles.render(shape, {"satin": {"opacity": 50}})
    check("Satin shades the inside of the shape and nothing outside",
          front is not None and front[..., 3][8:32, 8:32].max() > 0 and front[..., 3][:6].max() == 0)
    _b, front = layerstyles.render(shape, {"pattern_overlay": {"pattern": "stripes", "scale": 100}})
    inner = front[8:32, 8:32, 0]
    check("Pattern Overlay fills the shape with the pattern",
          len(np.unique(inner)) >= 2 and front[..., 3][:6].max() == 0)
    print("\nall edit-command checks passed")


if __name__ == "__main__":
    main()
