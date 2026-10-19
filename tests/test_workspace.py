#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""History snapshots, guide layouts and locking, Liquify's extra modes and
freeze mask, sampled brush tips and Define Brush Tip.

Run: python3 tests/test_workspace.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import brushtips
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WHITE, RED, BLACK = (255, 255, 255, 255), (255, 0, 0, 255), (0, 0, 0, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=120, h=90, state=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state or tempfile.mkdtemp())
    ctl.new_document(w, h, background=WHITE)
    return ctl


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def drag(ctl, pts, mods=0):
    ctl.begin_stroke(*pts[0], modifiers=mods)
    for p in pts[1:-1]:
        ctl.motion_stroke(*p, modifiers=mods)
    ctl.end_stroke(*pts[-1], modifiers=mods)


def main():
    # ---- snapshots ---------------------------------------------------------------------
    ctl = fresh()
    bg = ctl.active_layer
    ctl.selection.select_rect((10, 10, 40, 40))
    ctl.fill_selection(RED)
    ctl.selection.select_all()
    snap_img = shown(ctl).copy()
    name = ctl.new_snapshot()
    check("New Snapshot keeps the document as it is", name == "Snapshot 1" and len(ctl.snapshots) == 1)
    ctl.selection.select_rect((60, 40, 100, 80))
    ctl.fill_selection(BLACK)
    ctl.selection.select_all()
    extra = ctl.add_layer("later")
    ctl.resize_image(240, 180)
    later_img = shown(ctl).copy()
    n = len(ctl.history_labels())
    ctl.restore_snapshot(0)
    check("restoring a snapshot brings back pixels, layers and size",
          np.array_equal(shown(ctl), snap_img) and len(ctl.doc.layers()) == 1
          and (ctl.doc.width, ctl.doc.height) == (120, 90))
    check("...as one new history step", ctl.history_labels()[n:] == ["Snapshot: Snapshot 1"])
    check("...with a live active layer", ctl.active_layer is not None
          and ctl.doc.layer(ctl.active_layer.id) is ctl.active_layer)
    ctl.undo()
    check("undo returns to just before the restore",
          np.array_equal(shown(ctl), later_img) and len(ctl.doc.layers()) == 2
          and ctl.doc.layer(ctl.active_layer.id) is ctl.active_layer)
    ctl.redo()
    check("...and redo restores it again", np.array_equal(shown(ctl), snap_img))
    ctl.selection.select_rect((0, 0, 5, 5))
    ctl.fill_selection(BLACK)
    check("painting after a restore works on the restored document",
          tuple(shown(ctl)[2, 2]) == BLACK)
    check("Delete Snapshot", ctl.delete_snapshot(0) and ctl.snapshots == [])

    # ---- guide layout and lock ----------------------------------------------------------
    ctl = fresh(200, 100)
    ctl.new_guide_layout(columns=3, rows=0, gutter=10, margin=20)
    vs = [p for o, p in ctl.doc.guides if o == "v"]
    check("New Guide Layout: 3 columns with gutters inside the margins (%s)" % vs,
          vs == [20.0, 66.667, 76.667, 123.333, 133.333, 180.0])
    ctl.undo()
    check("...one undo step", ctl.doc.guides == [])
    ctl.add_guide("h", 50)
    ctl.set_guides_locked(True)
    check("locked guides cannot be moved or removed",
          ctl.move_guide(0, 70) is False and ctl.remove_guide(0) is False and ctl.doc.guides == [("h", 50.0)])
    ctl.set_guides_locked(False)
    ctl.move_guide(0, 70)
    check("...unlocked they can", ctl.doc.guides == [("h", 70.0)])

    # ---- Liquify --------------------------------------------------------------------------
    def stripes():
        c = fresh(120, 90)
        px = c.doc.cell(c.active_layer).plane("rgba")
        px[:, ::6] = BLACK
        c.doc.cell(c.active_layer)._touched(c.doc.bounds)
        return c
    ctl = stripes()
    before = ctl.doc.cell(ctl.active_layer).plane("rgba").copy()
    ctl.set_tool("warp", mode="freeze", size=60, strength=1.0)
    drag(ctl, [(30, 45), (31, 45), (33, 45), (36, 45)])
    ctl.set_tool_option("mode", "move")
    drag(ctl, [(20, 45), (30, 45), (40, 45), (50, 45), (60, 45)])
    after = ctl.doc.cell(ctl.active_layer).plane("rgba")
    frozen_change = int((after[40:50, 25:35] != before[40:50, 25:35]).any(-1).sum())
    free_change = int((after[40:50, 55:65] != before[40:50, 55:65]).any(-1).sum())
    check("Liquify's freeze mask protects what it covers (%d changed vs %d unfrozen)"
          % (frozen_change, free_change), frozen_change < 10 < free_change)
    ctl = stripes()
    ctl.set_tool("warp", mode="push_left", size=50, strength=1.0)
    drag(ctl, [(60, 70), (60, 60), (60, 50), (60, 40), (60, 30)])     # dragging UP
    a = ctl.doc.cell(ctl.active_layer).plane("rgba")[50, :, 0].astype(int)
    o = stripes()
    b0 = np.nonzero(o.doc.cell(o.active_layer).plane("rgba")[50, 40:80, 0] == 0)[0]
    b1 = np.nonzero(a[40:80] < 100)[0]
    check("Push Left, dragging up, moves pixels to the left (%s -> %s)" % (b0[:4], b1[:4]),
          b1.size and b1.mean() < b0.mean() - 0.5)
    ctl = stripes()
    ctl.set_tool("warp", mode="move", size=40, strength=1.0)
    # A jagged distortion: short pushes in alternating directions.
    for y, d in ((35, 8), (45, -8), (55, 8)):
        drag(ctl, [(50, y), (50 + d, y), (50 + 2 * d, y)])

    def rough(t):
        f = t._dx[25:65, 30:80].astype(float)
        return float(np.abs(np.diff(f, axis=0)).mean())
    r0 = rough(ctl.tool)
    ctl.set_tool_option("mode", "smooth")
    for _ in range(3):
        drag(ctl, [(50, 35), (52, 45), (50, 55)])
    r1 = rough(ctl.tool)
    check("Smooth relaxes a jagged distortion (field roughness %.3f -> %.3f)" % (r0, r1),
          r0 > 0 and r1 < r0 * 0.8)

    # ---- brush tips ---------------------------------------------------------------------------
    check("built-in tips are the same every run",
          np.array_equal(brushtips._make("chalk"), brushtips._make("chalk")))
    ctl = fresh()
    ctl.set_primary(BLACK)
    ctl.set_tool("brush", size=21, tip="square", hardness=1.0)
    ctl.begin_stroke(40, 40)
    ctl.end_stroke(40, 40)
    px = ctl.doc.cell(ctl.active_layer).plane("rgba")
    check("a square tip paints its corners (a round one would not)",
          px[31, 31, 0] < 60 and px[49, 49, 0] < 60)
    ctl.undo()
    ctl.set_tool("brush", size=21, tip="square", angle=45, hardness=1.0)
    ctl.begin_stroke(40, 40)
    ctl.end_stroke(40, 40)
    px = ctl.doc.cell(ctl.active_layer).plane("rgba")
    check("...turned 45 degrees it is a diamond: tips out, corners empty",
          px[40, 52, 0] < 128 and px[31, 31, 0] > 200)
    ctl.undo()

    # Define Brush Tip: an L, drawn black on white, becomes the tip.
    state = tempfile.mkdtemp()
    ctl = fresh(state=state)
    px = ctl.doc.cell(ctl.active_layer).plane("rgba")
    px[10:40, 10:16] = BLACK
    px[34:40, 10:40] = BLACK
    ctl.doc.cell(ctl.active_layer)._touched(ctl.doc.bounds)
    ctl.selection.select_rect((5, 5, 45, 45))
    check("Define Brush Tip from the selection", ctl.define_brush_tip())
    ctl.selection.select_all()
    ctl.add_layer("paint")
    ctl.set_tool("brush", size=30, tip="custom")
    ctl.begin_stroke(80, 50)
    ctl.end_stroke(80, 50)
    p = ctl.doc.cell(ctl.active_layer).plane("rgba")[..., 3]
    ys, xs = np.nonzero(p > 128)
    lower_left = p[60:64, 66:70].max() > 128
    upper_right = p[36:40, 90:94].max() > 128
    check("...a dab with it is the L: lower-left painted, upper-right empty",
          ys.size > 0 and lower_left and not upper_right)
    brushtips.set_custom(None)
    ctl2 = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state)
    check("...and the defined tip comes back next session", brushtips._custom["tip"] is not None)
    print("\nall workspace checks passed")


if __name__ == "__main__":
    main()
