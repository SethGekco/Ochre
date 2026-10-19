#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Brush dynamics (jitter, scatter, count), brush presets, the Art History Brush.

Run: python3 tests/test_brushes.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(state=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state or tempfile.mkdtemp())
    ctl.new_document(200, 120, background=None)
    return ctl


def stroke(ctl, pts):
    ctl.begin_stroke(*pts[0])
    for p in pts[1:]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*pts[-1])


def alpha(ctl):
    return ctl.doc.cell(ctl.active_layer).plane("rgba")[..., 3]


def main():
    line = [(20 + i * 4, 60) for i in range(40)]
    ctl = fresh()
    ctl.set_tool("brush", size=8, hardness=1.0)
    stroke(ctl, line)
    plain = alpha(ctl).copy()
    rows = np.nonzero(plain.max(axis=1))[0]
    check("a plain brush lays a band as wide as the brush", rows.max() - rows.min() <= 10)
    ctl.undo()
    ctl.set_tool("brush", size=8, hardness=1.0, scatter=300, count=4)
    stroke(ctl, line)
    sc = alpha(ctl).copy()
    rows = np.nonzero(sc.max(axis=1))[0]
    check("scatter sprays dabs away from the stroke", rows.max() - rows.min() > 25)
    ctl.undo()
    ctl.set_tool("brush", size=16, hardness=1.0, scatter=0, count=1, size_jitter=80)
    stroke(ctl, [(30 + i * 30, 60) for i in range(5)])
    widths = []
    a = alpha(ctl)
    for x in (30, 60, 90, 120, 150):
        col = np.nonzero(a[:, x])[0]
        widths.append(len(col))
    check("size jitter varies the dab size", max(widths) - min(widths) >= 3, str(widths))
    ctl.undo()
    ctl.set_tool("brush", size=16, size_jitter=0, opacity_jitter=90, spacing=2.0)
    stroke(ctl, [(30 + i * 34, 60) for i in range(5)])
    peaks = [int(alpha(ctl)[60, 30 + i * 34]) for i in range(5)]
    check("opacity jitter varies how strongly each dab lands", len(set(peaks)) > 1, str(peaks))

    # ---- presets ----------------------------------------------------------------
    state = tempfile.mkdtemp()
    ctl = fresh(state)
    ctl.set_tool("brush")
    names = ctl.brush_presets()
    check("built-in presets exist for the brush", {"Hard round", "Soft round", "Spray"} <= set(names))
    ctl.apply_brush_preset("Spray")
    check("applying a preset sets the tool's options",
          ctl.tool.option("scatter") == 400 and ctl.tool.option("count") == 10)
    ctl.set_tool_option("size", 33)
    ctl.save_brush_preset("Mine")
    ctl2 = fresh(state)
    ctl2.set_tool("brush")
    check("a saved preset survives into a new session",
          ctl2.brush_presets().get("Mine", {}).get("size") == 33)
    ctl2.delete_brush_preset("Mine")
    check("...and can be deleted", "Mine" not in ctl2.brush_presets())

    # ---- Art History Brush ----------------------------------------------------------
    ctl = fresh()
    cell = ctl.doc.cell(ctl.active_layer)
    px = cell.plane("rgba")
    px[:, :100] = (220, 40, 40, 255)
    px[:, 100:] = (40, 40, 220, 255)
    cell._touched(ctl.doc.bounds)
    ctl.set_history_source()
    px[...] = (255, 255, 255, 255)
    cell._touched(ctl.doc.bounds)
    ctl.set_tool("art_history", size=3, area=40)
    stroke(ctl, [(50, 60), (52, 60)])
    stroke(ctl, [(150, 60), (152, 60)])
    now = cell.plane("rgba")
    left = now[40:80, 30:70].reshape(-1, 4)
    right = now[40:80, 130:170].reshape(-1, 4)
    painted_l = left[(left[:, :3] != 255).any(axis=1)]
    painted_r = right[(right[:, :3] != 255).any(axis=1)]
    check("the Art History Brush repaints from the source in strokes",
          len(painted_l) > 50 and len(painted_r) > 50)
    check("...in the source's colours where it paints (red left, blue right)",
          painted_l[:, 0].mean() > painted_l[:, 2].mean()
          and painted_r[:, 2].mean() > painted_r[:, 0].mean())
    check("...as one undo step per stroke", ctl.history_labels()[-1] == "Art History Brush")
    print("\nall brush checks passed")


if __name__ == "__main__":
    main()
