#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Render > Picture Frame, ported from Kooka Pictura: Kooka's own property
tests (every frame draws and spares the middle, every shape has ink, holes
cut, colors, margin, arrangement, fade, thickness, number of lines, invert,
the moulding's light), plus the editor's: registered under Render, in the
Effects menu, one undo step through the controller.

Run: python3 tests/test_pictureframe.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import pictureframe as PF
from ochre.engine.effects import EffectRegistry
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry()


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def white(w, h):
    out = np.full((h, w, 4), 255, np.uint8)
    return out


def frame(img, **kw):
    return REG.create("picture_frame", **kw).apply(img)


def changed(a, b):
    return int(np.any(a != b, axis=-1).sum())


def main():
    check("Picture Frame is registered under Render",
          "picture_frame" in REG and REG.category("picture_frame") == "Render")
    check("...with Photoshop's 47 frames, 22 flowers and 23 leaves (plus None)",
          len(PF.FRAMES) == 47 and len(PF.FLOWERS) == 23 and len(PF.LEAVES) == 24
          and len(set(PF.FRAMES)) == 47)
    fx = REG.create("picture_frame")
    check("...and Photoshop's defaults (Pinwheel, Circle flowers, Leaf 1 leaves)",
          (fx.props.get("frame"), fx.props.get("flower"), fx.props.get("leaf"),
           fx.props.get("margin"), fx.props.get("size"), fx.props.get("thickness"))
          == ("pinwheel", "circle", "leaf_1", 14, 20, 29))

    # ---- every frame draws something and leaves the middle alone ---------------------
    base = white(300, 200)
    bad, middle = [], []
    for key in PF.FRAMES:
        out = frame(base, frame=key)
        if changed(base, out) <= 200:
            bad.append(key)
        if not np.array_equal(out[80:120, 120:180], base[80:120, 120:180]):
            middle.append(key)
    check("every frame draws something", not bad, str(bad))
    check("...and leaves the middle of the picture alone", not middle, str(middle))

    # ---- every flower and leaf has a shape ----------------------------------------------
    def ink(shape):
        px = white(64, 64)
        PF.fill(px, PF.place(shape, 32.0, 32.0, 28.0, 0.0), np.zeros(3), 1.0)
        return int((px[..., 0] < 128).sum())
    thin = [("flower", i, ink(PF.flower_shape(i))) for i in range(1, 23)]
    thin += [("leaf", i, ink(PF.leaf_shape(i))) for i in range(1, 24)]
    thin = [x for x in thin if not 100 < x[2] < 64 * 64 * 9 // 10]
    check("every flower and leaf has a shape", not thin, str(thin))
    check("None draws nothing", PF.flower_shape(0) == [] and PF.leaf_shape(0) == [])

    # ---- a hole is cut out of the shape around it --------------------------------------
    px = white(100, 100)
    PF.fill(px, PF.place(PF.flower_shape(16), 50.0, 50.0, 45.0, 0.0), np.zeros(3), 1.0)
    check("a hole is cut out of the shape around it (Star Cloud)",
          px[50, 50, 0] == 0 and px[50 + 29, 50, 0] == 255 and px[50 + 42, 50, 0] == 0)

    # ---- the rasteriser ---------------------------------------------------------------
    px = np.zeros((10, 10, 4), np.uint8)
    PF.fill(px, [PF.rect(2.0, 2.0, 7.5, 6.25)], np.array([255.0, 0, 0]), 1.0)
    check("a rectangle fills exactly, edge pixels by their covered share",
          px[3, 3, 3] == 255 and px[3, 7, 3] == 128 and px[6, 3, 3] == 64
          and px[1, 3, 3] == 0 and px[3, 8, 3] == 0 and tuple(px[3, 3, :3]) == (255, 0, 0))
    px2 = np.zeros((10, 10, 4), np.uint8)
    PF.fill(px2, [PF.rect(2.0, 2.0, 7.5, 6.25), PF.rect(2.0, 2.0, 7.5, 6.25)],
            np.array([255.0, 0, 0]), 1.0)
    check("...overlapping solids union, not double", np.array_equal(px, px2))
    px3 = np.zeros((10, 10, 4), np.uint8)
    PF.fill(px3, [np.array([[0.0, 0.0], [5.0, np.nan], [5.0, 5.0]])], np.zeros(3), 1.0)
    check("...a point that is not a number never floods the row", not px3[..., 3].any())

    # ---- colors ---------------------------------------------------------------------
    out = frame(base, frame="pinwheel", vine_color=(200, 0, 0), flower_color=(0, 0, 200))
    pix = {tuple(p) for p in out[..., :3].reshape(-1, 3)}
    check("the frame draws in the colors it is given", (200, 0, 0) in pix and (0, 0, 200) in pix)
    small = white(240, 160)
    check("a frame without leaves ignores the leaf",
          np.array_equal(frame(small, frame="pinwheel", leaf="leaf_1"),
                         frame(small, frame="pinwheel", leaf="cross")))

    # ---- margin, arrangement, fade, thickness, lines, invert ---------------------------
    def first_ink(margin):
        out = frame(base, frame="line_box", margin=margin)
        return int(np.argmax(out[:100, 150, 0] < 255))
    check("a bigger margin moves the frame in", first_ink(40) > first_ink(5))
    check("the arrangement reshuffles the frame",
          np.array_equal(frame(small, frame="circle_sprinkle", arrangement=3),
                         frame(small, frame="circle_sprinkle", arrangement=3))
          and not np.array_equal(frame(small, frame="circle_sprinkle", arrangement=3),
                                 frame(small, frame="circle_sprinkle", arrangement=4)))

    big = white(400, 300)

    def ink_g(fade, x0, x1):
        out = frame(big, frame="line_box", fade=fade)
        return int((255 - out[:40, x0:x1, 1].astype(int)).sum())
    check("fade thins the middle of each side",
          ink_g(80, 180, 220) * 3 < ink_g(0, 180, 220))
    check("...and spares the corners", ink_g(80, 8, 48) * 10 > ink_g(0, 8, 48) * 7)
    check("thickness widens the lines",
          changed(base, frame(base, frame="simple_lace", thickness=120))
          > changed(base, frame(base, frame="simple_lace", thickness=10)))

    def crossings(lines):
        out = frame(big, frame="chainmail", lines=lines, flower="none")
        dark = out[:60, :, 1] < 200
        y = int(np.argmax(dark.sum(1)))
        row = dark[y]
        return int((row[1:] & ~row[:-1]).sum())
    check("number of lines packs the lines closer", crossings(15) > crossings(5) * 2)
    bad = [k for k in ("happy_vine", "snakes", "inverse_rounded_corners_1", "rounded_art_frame")
           if np.array_equal(frame(small, frame=k), frame(small, frame=k, invert=True))]
    check("invert turns a frame inside out", not bad, str(bad))
    sq = white(300, 300)
    art = frame(sq, frame="art_frame", vine_color=(150, 100, 50))
    check("the moulding is lit from the top left (its top and bottom differ)",
          not np.array_equal(art[0:60, 150, 0], art[240:300, 150, 0][::-1]))
    check("Angle turns the flowers",
          not np.array_equal(frame(small, frame="spacing", flower="star"),
                             frame(small, frame="spacing", flower="star", angle=45)))

    # ---- over transparency ----------------------------------------------------------
    clear = np.zeros((120, 160, 4), np.uint8)
    out = frame(clear, frame="line_box")
    check("on a transparent layer the frame is drawn opaque, the rest stays clear",
          out[..., 3].max() == 255 and not out[50:70, 70:90, 3].any())

    # ---- through the controller, as the menu does --------------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(120, 80, background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    pic = white(120, 80)
    pic[..., :3] = 90
    cell.plane("rgba")[...] = pic
    cell._touched(ctl.doc.bounds)
    pos = ctl.history.position
    ctl.apply_effect("picture_frame", {"frame": "snowflakes"})
    got = cell.plane("rgba").copy()
    check("applied as the menu does: one undo step, the direct result",
          ctl.history.position == pos + 1
          and np.array_equal(got, frame(pic, frame="snowflakes")))
    ctl.undo()
    check("...undoable", np.array_equal(cell.plane("rgba"), pic))

    # ---- the Effects menu, and so the command search -------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.commandsearch import build_index
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-pictureframe-"))
    win = MainWindow(ctl, UiSettings(DATA))
    check("Effects > Render > Picture Frame is in the menu and the command search",
          ("Effects", "Render", "Picture Frame") in {tuple(e.path) for e in build_index(win)})
    win.close()
    del app
    print("\nall picture frame checks passed")


if __name__ == "__main__":
    main()
