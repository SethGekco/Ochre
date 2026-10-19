#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Artboards and frames, driven through the tools and the controller.

Run: python3 tests/test_artboards.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from ochre.engine import regions
from ochre.engine.fileio import ochre_format, psd
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RED, BLUE, WHITE = (255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=200, h=120, bg=RED):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=bg)
    return ctl


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
    # ---- the Artboard tool --------------------------------------------------------------
    ctl = fresh()
    n0 = len(ctl.history_labels())
    ctl.set_tool("artboard")
    drag(ctl, (10, 10), (90, 70))
    boards = regions.groups(ctl.doc, regions.ARTBOARD)
    check("dragging with the Artboard tool makes an artboard",
          len(boards) == 1 and boards[0].region["rect"] == [10, 10, 80, 60], str(boards and boards[0].region))
    a1 = boards[0]
    check("...with an empty layer on it, active, ready to paint",
          ctl.active_layer.parent is a1 and not ctl.active_layer.is_group)
    check("...as ONE undo step", ctl.history_labels()[n0:] == ["New Artboard"], str(ctl.history_labels()[n0:]))
    img = shown(ctl)
    check("its white background shows inside, the canvas outside",
          px(img, 50, 40) == WHITE and px(img, 5, 5) == RED and px(img, 95, 40) == RED)
    inner = ctl.active_layer
    ctl.doc.cell(inner).fill(ctl.doc.bounds, BLUE)
    img = shown(ctl)
    check("what is painted on it is cut to its edges", px(img, 50, 40) == BLUE
          and px(img, 9, 40) == RED and px(img, 90, 40) == RED and px(img, 89, 69) == BLUE)

    # second artboard; a click selects; a drag moves with contents
    drag(ctl, (110, 20), (190, 100))
    a2 = regions.groups(ctl.doc, regions.ARTBOARD)[-1]
    check("a second artboard is named in turn", a2.name == "Artboard 2" and a2 is not a1)
    ctl.begin_stroke(50, 40)
    ctl.end_stroke(50, 40)
    check("clicking an artboard selects it", ctl.active_layer is a1)
    n1 = len(ctl.history_labels())
    drag(ctl, (50, 40), (60, 45))
    img = shown(ctl)
    check("dragging an artboard moves it AND what is on it",
          a1.region["rect"] == [20, 15, 80, 60] and px(img, 95, 70) == BLUE and px(img, 15, 40) == RED)
    check("...one undo step", ctl.history_labels()[n1:] == ["Move Artboard"], str(ctl.history_labels()[n1:]))
    ctl.undo()
    check("...that undoes back", a1.region["rect"] == [10, 10, 80, 60]
          and px(shown(ctl), 15, 40) == BLUE)

    # ---- export ------------------------------------------------------------------------
    out = tempfile.mkdtemp()
    paths = ctl.export_artboards(out)
    sizes = [Image.open(p).size for p in paths]
    first = np.asarray(Image.open(paths[0]).convert("RGBA"))
    second = np.asarray(Image.open(paths[1]).convert("RGBA"))
    check("Export Artboards writes one file per artboard, at its own size",
          len(paths) == 2 and sizes == [(80, 60), (80, 80)], str(sizes))
    check("...each holding only its own artboard (no canvas, no neighbour)",
          (first[..., 2] == 255).all() and (second == 255).all())

    # ---- geometry, undo, files ----------------------------------------------------------
    ctl.resize_image(400, 240)
    check("resizing the image scales the artboards", a1.region["rect"] == [20, 20, 160, 120])
    ctl.undo()
    check("...and undo puts them back", a1.region["rect"] == [10, 10, 80, 60])
    before = shown(ctl)
    path = os.path.join(tempfile.mkdtemp(), "a.ochre")
    ochre_format.save(ctl.doc, path)
    doc2, _ = ochre_format.load(path)
    from ochre.engine.compositor import Compositor
    check(".ochre keeps artboards", doc2.layer(a1.id).region == a1.region
          and np.array_equal(Compositor(doc2).composite(doc2.bounds), before))
    ctl.save_path(os.path.join(tempfile.mkdtemp(), "a.psd"))
    check("PSD says it keeps them as plain groups",
          any("artboard" in w for w in (ctl.last_save_warnings or [])))

    # ---- the Frame tool ------------------------------------------------------------------
    ctl = fresh(bg=WHITE)
    photo = ctl.add_layer("photo")
    ctl.doc.cell(photo).fill(ctl.doc.bounds, BLUE)
    ctl.set_tool("frame", shape="ellipse")
    n2 = len(ctl.history_labels())
    drag(ctl, (40, 20), (160, 100))
    frame = regions.frame_of(photo)
    img = shown(ctl)
    check("a frame drawn over an image layer takes it in", frame is not None
          and frame.region["shape"] == "ellipse" and ctl.history_labels()[n2:] == ["New Frame"])
    check("...and crops it to the ellipse", px(img, 100, 60) == BLUE and px(img, 42, 22) == WHITE
          and px(img, 5, 5) == WHITE and 0 < img[60, 40, 2] - img[60, 40, 1] < 255 + 1)
    ctl.undo()
    check("undo lets the layer out again", photo.parent is ctl.doc.root
          and regions.groups(ctl.doc) == [] and px(shown(ctl), 5, 5) == BLUE)

    # empty frame, filled by Place Embedded
    ctl = fresh(bg=WHITE)
    ctl.set_tool("frame", shape="rect")
    drag(ctl, (50, 30), (150, 90))
    frame = regions.groups(ctl.doc, regions.FRAME)[0]
    check("a frame over the bottom layer starts empty (the backdrop is never swallowed)",
          frame.leaves() == [] and ctl.active_layer is frame)
    pic = np.zeros((50, 200, 4), np.uint8)
    pic[...] = (0, 160, 0, 255)
    placed = ctl.place_embedded(pic, "pic")
    img = shown(ctl)
    check("Place Embedded into an active frame puts the image inside it",
          placed.parent is frame and placed.kind == "smart")
    check("...scaled to FILL the frame, cropped at its edges",
          (img[30:90, 50:150, 1] == 160).all() and px(img, 49, 60) == WHITE and px(img, 150, 60) == WHITE)
    drag(ctl, (100, 60), (110, 70))
    img = shown(ctl)
    check("dragging the frame moves the picture with it",
          frame.region["rect"] == [60, 40, 100, 60] and (img[40:100, 60:160, 1] == 160).all()
          and px(img, 55, 60) == WHITE)
    print("\nall artboard and frame checks passed")


if __name__ == "__main__":
    main()
