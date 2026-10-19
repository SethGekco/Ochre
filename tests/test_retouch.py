#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The retouching and paint-variant tools, through the controller.

Each tool is driven the way a user drives it -- set_tool, begin, motion,
end -- and judged on what it is FOR: blur must reduce local variance,
dodge must lighten, the background eraser must take the sampled colour and
leave its neighbour, and so on. Plus: every stroke is exactly one undo entry
and undoing it restores the layer bit for bit.

Run: python3 tests/test_retouch.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.engine.tools.retouch import blur_rgba, luminance
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RED, BLUE = (200, 40, 40, 255), (40, 60, 200, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(noise=True):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(80, 60, background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    cell.fill(Rect(0, 0, 40, 60), RED)
    cell.fill(Rect(40, 0, 40, 60), BLUE)
    if noise:
        rng = np.random.default_rng(0)
        cell.plane("rgba")[20:40, 10:70, :3] = rng.integers(0, 255, (20, 60, 3))
        cell._touched(ctl.doc.bounds)
    ctl.set_history_source()
    return ctl, cell


def stroke(ctl, pts, mods=0):
    ctl.begin_stroke(pts[0][0], pts[0][1], 1.0, mods)
    for x, y in pts[1:]:
        ctl.motion_stroke(x, y, 1.0, mods)
    ctl.end_stroke(pts[-1][0], pts[-1][1], 1.0, mods)


def one_undo(ctl, cell, before, name):
    check("%s is one undo entry" % name, ctl.history.position == 0,
          "-- position %d" % ctl.history.position)
    ctl.undo()
    check("...and undo restores the layer exactly", np.array_equal(cell.plane("rgba"), before))


def main():
    # ---- kernels -------------------------------------------------------------
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[4:12, 4:12] = (0, 0, 255, 128)
    soft = blur_rgba(rgba, 2)
    check("blur is alpha-weighted: no dark fringe at a transparent edge",
          int(soft[8, 3, 2]) == 255 and int(soft[8, 3, 0]) == 0, "-- %s" % soft[8, 3].tolist())

    # ---- blur / sharpen ------------------------------------------------------
    for name, more in (("blur", False), ("sharpen", True)):
        ctl, cell = fresh()
        before = cell.plane("rgba").copy()
        ctl.set_tool(name, size=20, strength=0.8)
        stroke(ctl, [(20, 30), (40, 30), (60, 30)])
        a, b = before[22:38, 25:55, :3].std(), cell.plane("rgba")[22:38, 25:55, :3].std()
        check("%s %s local contrast" % (name, "raises" if more else "lowers"),
              (b > a) if more else (b < a), "-- %.1f -> %.1f" % (a, b))
        one_undo(ctl, cell, before, name)

    # ---- smudge drags colour along the stroke --------------------------------
    ctl, cell = fresh(noise=False)
    before = cell.plane("rgba").copy()
    ctl.set_tool("smudge", size=14, strength=0.9)
    stroke(ctl, [(30, 10), (36, 10), (42, 10), (48, 10), (55, 10)])
    check("SMUDGE CARRIES RED INTO THE BLUE", int(cell.plane("rgba")[10, 45, 0]) > 100,
          "-- R at (45,10) is %d" % int(cell.plane("rgba")[10, 45, 0]))
    check("...and not against the stroke direction", int(cell.plane("rgba")[10, 20, 2]) < 100)
    one_undo(ctl, cell, before, "smudge")

    # ---- dodge / burn / sponge -----------------------------------------------
    ctl, cell = fresh(noise=False)
    before = cell.plane("rgba").copy()
    ctl.set_tool("dodge", size=16, range="midtones")
    stroke(ctl, [(10, 50), (30, 50)])
    check("dodge lightens", luminance(cell.plane("rgba")[50:51, 12:28]).mean()
          > luminance(before[50:51, 12:28]).mean())
    one_undo(ctl, cell, before, "dodge")
    ctl.set_tool("burn", size=16)
    stroke(ctl, [(10, 50), (30, 50)])
    check("burn darkens", luminance(cell.plane("rgba")[50:51, 12:28]).mean()
          < luminance(before[50:51, 12:28]).mean())
    ctl.undo()
    ctl.set_tool("sponge", size=16, mode="desaturate", strength=1.0)
    stroke(ctl, [(10, 50), (30, 50)])
    px = cell.plane("rgba")[50, 20].astype(int)
    check("sponge desaturates", px[:3].max() - px[:3].min() < 160,
          "-- spread %d (was 160)" % (px[:3].max() - px[:3].min()))
    ctl.undo()
    ctl.set_tool("sponge", size=16, mode="saturate", strength=1.0)
    stroke(ctl, [(50, 50), (70, 50)])
    px = cell.plane("rgba")[50, 60].astype(int)
    check("sponge in saturate mode saturates", px[:3].max() - px[:3].min() > 160)

    # ---- recolor --------------------------------------------------------------
    ctl, cell = fresh(noise=False)
    ctl.set_secondary(RED)
    ctl.set_primary((40, 200, 40, 255))
    ctl.set_tool("recolor", size=20, tolerance=40)
    stroke(ctl, [(10, 5), (30, 5), (30, 55)])
    check("RECOLOR REPLACES THE SECONDARY WITH THE PRIMARY",
          cell.plane("rgba")[5, 20].tolist() == [40, 200, 40, 255],
          "-- got %s" % cell.plane("rgba")[5, 20].tolist())
    ctl.set_tool("recolor", size=20, tolerance=40)
    stroke(ctl, [(50, 5), (70, 5)])
    check("...and leaves colours outside the tolerance alone",
          cell.plane("rgba")[5, 60].tolist() == list(BLUE))

    # ---- erasers ----------------------------------------------------------------
    ctl, cell = fresh(noise=False)
    ctl.set_tool("background_eraser", size=24, tolerance=30)
    stroke(ctl, [(45, 5), (75, 5)])
    check("THE BACKGROUND ERASER TAKES THE SAMPLED COLOUR", int(cell.plane("rgba")[5, 60, 3]) == 0)
    check("...and spares the neighbour of a different colour",
          int(cell.plane("rgba")[5, 35, 3]) == 255)
    ctl, cell = fresh(noise=False)
    before = cell.plane("rgba").copy()
    ctl.set_tool("magic_eraser")
    ctl.begin_stroke(5, 5)
    ctl.end_stroke(5, 5)
    check("the magic eraser clears the clicked region",
          int(cell.plane("rgba")[50, 5, 3]) == 0 and int(cell.plane("rgba")[50, 70, 3]) == 255)
    one_undo(ctl, cell, before, "magic eraser")

    # ---- history brush -----------------------------------------------------------
    ctl, cell = fresh()
    orig = cell.plane("rgba").copy()
    cell.fill(ctl.doc.bounds, (0, 0, 0, 255))
    ctl.set_tool("history_brush", size=10)
    stroke(ctl, [(10, 10), (70, 10)])
    check("THE HISTORY BRUSH PAINTS BACK THE ORIGINAL",
          np.array_equal(cell.plane("rgba")[10, 15:65], orig[10, 15:65]))
    check("...only where it was painted", cell.plane("rgba")[50, 40].tolist() == [0, 0, 0, 255])

    # ---- airbrush, ink, mixer, pattern ------------------------------------------
    ctl, cell = fresh(noise=False)
    ctl.set_tool("airbrush", size=20)
    check("the airbrush runs at its declared low flow", abs(ctl.tool.option("flow") - 0.12) < 1e-9)
    ctl.begin_stroke(20, 50)
    first = int(cell.plane("rgba")[50, 20, 0])
    for _ in range(15):
        ctl.tick_stroke()
    ctl.end_stroke(20, 50)
    check("THE AIRBRUSH KEEPS BUILDING WHILE HELD STILL",
          int(cell.plane("rgba")[50, 20, 0]) < first - 50,
          "-- R %d after the first dab, %d after holding" % (first, cell.plane("rgba")[50, 20, 0]))

    ctl, cell = fresh(noise=False)
    cell.fill(ctl.doc.bounds, (255, 255, 255, 255))
    ctl.set_tool("ink", size=10, speed=0.9)
    pts = [(x, 10) for x in range(5, 20)] + [(x, 10) for x in range(25, 80, 12)]
    stroke(ctl, pts)
    dark = cell.plane("rgba")[..., 0] < 128
    check("INK THINS WHEN THE PEN MOVES FAST", int(dark[:, 61].sum()) < int(dark[:, 12].sum()),
          "-- slow %d px, fast %d px" % (dark[:, 12].sum(), dark[:, 61].sum()))

    ctl, cell = fresh(noise=False)
    ctl.set_primary((255, 255, 255, 255))
    ctl.set_tool("mixer", size=12, wet=0.8, load=0.0)
    stroke(ctl, [(30, 50), (40, 50), (55, 50)])
    px = cell.plane("rgba")[50, 50]
    check("the mixer brush carries canvas colour along (red mixed into blue)",
          int(px[0]) > 40 and px.tolist() != list(BLUE), "-- %s" % px.tolist())

    ctl, cell = fresh(noise=False)
    ctl.set_primary((0, 0, 0, 255))
    ctl.set_secondary((255, 255, 255, 255))
    ctl.set_tool("pattern_stamp", size=16, pattern="checker", scale=4)
    stroke(ctl, [(10, 10), (70, 10)])
    colours = {tuple(v) for v in cell.plane("rgba")[8:12, 20:60].reshape(-1, 4)}
    check("the pattern stamp lays both pattern colours",
          (0, 0, 0, 255) in colours and (255, 255, 255, 255) in colours)

    # ---- heal and red eye ----------------------------------------------------------
    ctl, cell = fresh(noise=False)
    yy, xx = np.mgrid[0:16, 0:16]
    tex = np.where(((xx // 2 + yy // 2) % 2)[..., None], 30, -30)
    cell.plane("rgba")[2:18, 2:18, :3] = np.clip(np.array(RED[:3]) + tex, 0, 255)
    cell._touched(ctl.doc.bounds)
    ctl.set_tool("heal", size=12)
    ctl.begin_stroke(10, 10, 1.0, 2)          # ctrl-click: the source
    ctl.end_stroke(10, 10, 1.0, 2)
    before = cell.plane("rgba").copy()
    stroke(ctl, [(60, 40), (62, 40)])
    after = cell.plane("rgba")
    check("HEAL TRANSFERS TEXTURE", after[34:46, 54:66, :3].std() > before[34:46, 54:66, :3].std())
    check("...while keeping the destination's colour (still blue on average)",
          after[34:46, 54:66, 2].mean() > after[34:46, 54:66, 0].mean() + 60)

    ctl, cell = fresh(noise=False)
    ctl.set_tool("red_eye", size=16)
    stroke(ctl, [(10, 50), (30, 50)])
    check("red eye removal neutralises strong red", int(cell.plane("rgba")[50, 20, 0]) < 120)
    stroke(ctl, [(50, 50), (70, 50)])
    check("...and leaves non-red pixels alone", cell.plane("rgba")[50, 60].tolist() == list(BLUE))

    print("\nall retouch checks passed")


if __name__ == "__main__":
    main()
