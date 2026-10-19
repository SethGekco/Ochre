#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Inpainting: the fill engine, and the tools and command built on it.

The engine is checked on textures whose right answer is known exactly --
stripes and a checkerboard with a hole cut out -- so "the fill is good" is a
measurement, not an opinion. The tools are driven through the controller as
the canvas drives them, and each gesture must be one undo entry that puts
the layer back exactly.

Run: python3 tests/test_inpaint.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import inpaint
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BLEMISH = (250, 250, 0, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def stripes(h, w, period=5, noise=0, seed=1):
    yy = np.mgrid[:h, :w][0]
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    band = (yy // period) % 2 == 0
    img[band, :3] = (200, 40, 40)
    img[~band, :3] = (30, 30, 160)
    if noise:
        n = np.random.default_rng(seed).integers(-noise, noise, (h, w, 1))
        img[..., :3] = np.clip(img[..., :3].astype(int) + n, 0, 255)
    return img


def close_fraction(a, b, mask, tol=30):
    d = np.abs(a[mask][:, :3].astype(int) - b[mask][:, :3].astype(int)).max(axis=1)
    return float((d < tol).mean())


def fresh(w=120, h=100, period=5):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    truth = stripes(h, w, period)
    cell.plane("rgba")[...] = truth
    cell._touched(ctl.doc.bounds)
    return ctl, cell, truth


def drag(ctl, pts, mods=0):
    ctl.begin_stroke(pts[0][0], pts[0][1], 1.0, mods)
    for x, y in pts[1:]:
        ctl.motion_stroke(x, y, 1.0, mods)
    return ctl.end_stroke(pts[-1][0], pts[-1][1], 1.0, mods)


def main():
    # ---- the engine ---------------------------------------------------------
    img = stripes(200, 240, noise=10)
    hole = np.zeros((200, 240), bool)
    hole[70:130, 90:160] = True
    t0 = time.time()
    out = inpaint.fill_hole(img, hole)
    took = time.time() - t0
    check("content-aware fill rebuilds stripes across a 60x70 hole",
          close_fraction(out, img, hole) > 0.97, "(%.3f)" % close_fraction(out, img, hole))
    check("...and leaves every pixel outside the hole alone",
          np.array_equal(out[~hole], img[~hole]))
    check("...in interactive time (%.2fs)" % took, took < 5.0)
    smooth = inpaint.fill_hole(img, hole, method="smooth")
    check("a smooth fill cannot rebuild texture (which is why both exist)",
          close_fraction(smooth, img, hole) < 0.5)

    yy, xx = np.mgrid[:120, :120]
    chk = np.zeros((120, 120, 4), np.uint8)
    chk[..., 3] = 255
    on = (yy // 6 + xx // 6) % 2 == 0
    chk[on, :3] = (240, 220, 30)
    chk[~on, :3] = (20, 120, 20)
    hole2 = np.zeros((120, 120), bool)
    hole2[40:80, 45:85] = True
    out2 = inpaint.fill_hole(chk, hole2)
    check("content-aware fill continues a checkerboard in phase",
          close_fraction(out2, chk, hole2) > 0.95, "(%.3f)" % close_fraction(out2, chk, hole2))

    # A hole straddling a transparent edge: premultiplied work means the
    # transparent side stays transparent and the opaque side gains no fringe.
    half = np.zeros((60, 60, 4), np.uint8)
    half[:, :30] = (0, 200, 0, 255)
    h3 = np.zeros((60, 60), bool)
    h3[20:40, 22:38] = True
    for method in ("smooth", "content_aware"):
        o = inpaint.fill_hole(half, h3, method=method)
        opaque = o[20:40, 22:26]
        clear = o[20:40, 34:38]
        dark = (opaque[..., 3] > 0) & (opaque[..., 1] < 150)
        check("%s fill next to transparency: no dark fringe" % method, not dark.any())
        if method == "smooth":
            # A harmonic fill ramps alpha across the hole -- a gradient is the
            # smoothest surface -- so the clear side is fainter, not empty.
            check("smooth fill: alpha falls toward the transparent side",
                  int(clear[..., 3].max()) < int(opaque[..., 3].min()))
        else:
            check("content-aware fill: the transparent side stays transparent",
                  int(clear[..., 3].max()) < 40)

    # Seamless blending: the source's DETAIL arrives, the border meets dest.
    # A flat destination makes the right answer exact: the correction field
    # is constant, so the square's step survives unchanged.
    flat = np.zeros((80, 80, 4), np.uint8)
    flat[...] = (50, 60, 70, 255)
    dest = inpaint.to_work(flat)
    src = dest.copy()
    src[..., :3] = 100.0
    src[35:45, 35:45, :3] = 160.0        # a bright square in a flat field
    region = np.zeros((80, 80), bool)
    region[20:60, 20:60] = True
    blended = inpaint.seamless_blend(dest, src, region)
    step = blended[40, 40, 0] - blended[30, 30, 0]
    check("seamless blend keeps the source's detail (the square's 60-level step)",
          abs(step - 60.0) < 3.0, "(step %.1f)" % step)
    edge_err = np.abs(blended[20, 20:60, :3] - dest[20, 20:60, :3]).max()
    check("...and meets the destination at the region border", edge_err < 3.0,
          "(%.1f)" % edge_err)

    # ---- Spot Healing Brush -------------------------------------------------
    ctl, cell, truth = fresh()
    cell.fill(Rect(55, 45, 6, 6), BLEMISH)
    blemished = cell.plane("rgba").copy()
    ctl.set_tool("spot_heal", size=14)
    n0 = len(ctl.history_labels())
    drag(ctl, [(57, 47), (58, 48), (59, 49)])
    spot = np.zeros((100, 120), bool)
    spot[45:51, 55:61] = True
    now = cell.plane("rgba")
    check("spot heal removes the blemish (its stripes come back)",
          close_fraction(now, truth, spot) > 0.9, "(%.3f)" % close_fraction(now, truth, spot))
    check("...without leaving the grey stroke preview behind",
          not (np.abs(now[..., :3].astype(int) - 96).max(axis=-1) < 4).any())
    check("...as ONE history entry", len(ctl.history_labels()) == n0 + 1,
          str(ctl.history_labels()[n0:]))
    far = np.ones((100, 120), bool)
    far[30:66, 40:76] = False
    check("...that touches nothing away from the stroke",
          np.array_equal(now[far], blemished[far]))
    ctl.undo()
    check("undo puts the blemish back exactly", np.array_equal(cell.plane("rgba"), blemished))
    ctl.redo()
    check("redo heals it again", close_fraction(cell.plane("rgba"), truth, spot) > 0.9)

    # Escape mid-stroke leaves nothing.
    ctl, cell, truth = fresh()
    ctl.set_tool("spot_heal", size=14)
    ctl.begin_stroke(30, 30)
    ctl.motion_stroke(40, 32)
    ctl.cancel_stroke()
    check("escape during a spot-heal stroke restores every pixel",
          np.array_equal(cell.plane("rgba"), truth))

    # The selection fences the heal: the brush sweeps over a blemish inside
    # the selection and a mark outside it; only the first may change.
    ctl, cell, truth = fresh()
    cell.fill(Rect(50, 42, 8, 6), BLEMISH)
    cell.fill(Rect(64, 43, 3, 3), (0, 255, 255, 255))
    marked = cell.plane("rgba").copy()
    ctl.selection.select_rect((48, 38, 60, 52))
    ctl.set_tool("spot_heal", size=30)
    drag(ctl, [(52, 45), (68, 45)])
    outside = np.ones((100, 120), bool)
    outside[38:52, 48:60] = False
    check("spot heal honours the selection: nothing outside it changes",
          np.array_equal(cell.plane("rgba")[outside], marked[outside]))
    inside = np.zeros((100, 120), bool)
    inside[42:48, 50:58] = True
    check("...while the selected blemish is healed",
          close_fraction(cell.plane("rgba"), truth, inside) > 0.8,
          "(%.3f)" % close_fraction(cell.plane("rgba"), truth, inside))

    # ---- Remove tool ----------------------------------------------------------
    ctl, cell, truth = fresh(160, 120)
    cell.fill(Rect(70, 50, 18, 18), BLEMISH)
    ctl.set_tool("remove", size=30)
    drag(ctl, [(74, 59), (79, 59), (84, 59)])
    obj = np.zeros((120, 160), bool)
    obj[50:68, 70:88] = True
    check("the remove tool rebuilds an 18px object from the texture around it",
          close_fraction(cell.plane("rgba"), truth, obj) > 0.9,
          "(%.3f)" % close_fraction(cell.plane("rgba"), truth, obj))

    # ---- Patch tool -------------------------------------------------------------
    ctl, cell, truth = fresh()
    cell.fill(Rect(20, 20, 10, 10), BLEMISH)
    ctl.set_tool("patch")
    # No selection yet: the first drag lassos one.
    drag(ctl, [(18, 18), (32, 18), (32, 32), (18, 32), (18, 19)])
    check("the patch tool lassos when there is no selection",
          ctl.selection.mask is not None and ctl.selection.mask[25, 25] == 255)
    # Then drag it onto clean texture a whole number of stripe periods away.
    n0 = len(ctl.history_labels())
    drag(ctl, [(25, 25), (40, 25), (55, 35)])
    area = np.zeros((100, 120), bool)
    area[20:30, 20:30] = True
    check("Patch (source): the area takes the texture it was dragged onto",
          close_fraction(cell.plane("rgba"), truth, area) > 0.9,
          "(%.3f)" % close_fraction(cell.plane("rgba"), truth, area))
    check("...the selection stays where it was",
          ctl.selection.mask[25, 25] == 255)
    labels = ctl.history_labels()[n0:]
    check("...pixels are one undo entry", labels.count("Patch Tool") == 1, str(labels))
    ctl.undo()
    check("undo restores the patched area exactly",
          (cell.plane("rgba")[25, 25] == np.array(BLEMISH, np.uint8)).all())

    ctl, cell, truth = fresh()
    cell.fill(Rect(20, 20, 10, 10), BLEMISH)
    ctl.selection.select_rect((20, 20, 30, 30))
    ctl.set_tool("patch", patch_mode="destination", seamless=False)
    drag(ctl, [(25, 25), (45, 50), (70, 60)])
    dst = np.zeros((100, 120), bool)
    dst[55:65, 65:75] = True
    copied = (cell.plane("rgba")[dst][:, :3] == np.array(BLEMISH[:3], np.uint8)).all()
    check("Patch (destination): the selection's content is copied where dropped", copied)
    check("...the original is left in place",
          (cell.plane("rgba")[25, 25, :3] == np.array(BLEMISH[:3], np.uint8)).all())
    check("...and the selection moves with it",
          ctl.selection.mask[60, 70] == 255 and ctl.selection.mask[25, 25] == 0)

    # ---- Content-Aware Move -------------------------------------------------------
    ctl, cell, truth = fresh(160, 120)
    cell.fill(Rect(20, 30, 12, 12), BLEMISH)
    ctl.selection.select_rect((20, 30, 32, 42))
    ctl.set_tool("content_move", seamless=False)
    before = cell.plane("rgba").copy()
    n0 = len(ctl.history_labels())
    drag(ctl, [(26, 36), (60, 50), (106, 66)])
    was = np.zeros((120, 160), bool)
    was[30:42, 20:32] = True
    now = cell.plane("rgba")
    check("content-aware move: where the object was is filled with texture",
          close_fraction(now, truth, was) > 0.85, "(%.3f)" % close_fraction(now, truth, was))
    at = now[60:72, 100:112]
    check("...and the object is where it was dropped",
          (at[..., :3] == np.array(BLEMISH[:3], np.uint8)).all())
    check("...with the selection following it",
          ctl.selection.mask[66, 106] == 255 and ctl.selection.mask[36, 26] == 0)
    check("...the move AND the selection change are ONE undo entry",
          ctl.history_labels()[n0:] == ["Content-Aware Move"], str(ctl.history_labels()[n0:]))
    ctl.undo()
    check("one undo returns the object and the texture exactly",
          np.array_equal(cell.plane("rgba"), before))
    check("...and the selection with them",
          ctl.selection.mask[36, 26] == 255 and ctl.selection.mask[66, 106] == 0)

    ctl, cell, truth = fresh(160, 120)
    cell.fill(Rect(20, 30, 12, 12), BLEMISH)
    ctl.selection.select_rect((20, 30, 32, 42))
    ctl.set_tool("content_move", move_mode="extend", seamless=False)
    drag(ctl, [(26, 36), (106, 66)])
    check("Extend leaves the original and makes a copy",
          (cell.plane("rgba")[36, 26, :3] == np.array(BLEMISH[:3], np.uint8)).all()
          and (cell.plane("rgba")[66, 106, :3] == np.array(BLEMISH[:3], np.uint8)).all())

    # ---- Edit > Content-Aware Fill ---------------------------------------------------
    ctl, cell, truth = fresh()
    cell.fill(Rect(40, 30, 20, 20), BLEMISH)
    blemished = cell.plane("rgba").copy()
    check("content-aware fill with nothing selected does nothing",
          ctl.inpaint_selection() is None and np.array_equal(cell.plane("rgba"), blemished))
    ctl.selection.select_rect((38, 28, 62, 52))
    ctl.inpaint_selection()
    obj = np.zeros((100, 120), bool)
    obj[30:50, 40:60] = True
    check("Edit > Content-Aware Fill rebuilds the selection",
          close_fraction(cell.plane("rgba"), truth, obj) > 0.9,
          "(%.3f)" % close_fraction(cell.plane("rgba"), truth, obj))
    check("...labelled in history", ctl.history_labels()[-1] == "Content-Aware Fill")
    ctl.undo()
    check("...and undoes exactly", np.array_equal(cell.plane("rgba"), blemished))
    ctl.inpaint_selection("smooth")
    check("Smooth Fill replaces the selection with no trace of the object",
          not (cell.plane("rgba")[obj][:, :3] == np.array(BLEMISH[:3], np.uint8)).all(-1).any())

    print("\nall inpaint checks passed")


if __name__ == "__main__":
    main()
