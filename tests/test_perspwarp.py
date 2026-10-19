#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Perspective Warp: quads laid over planes, sharing edges, pulled.

Run: python3 tests/test_perspwarp.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import perspwarp as PW
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def gradient(w=120, h=80):
    ys, xs = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0] = (xs * 2) % 256
    out[..., 1] = (ys * 3) % 256
    out[..., 2] = 120
    out[..., 3] = 255
    return out


def main():
    img = gradient()
    q = [(20.0, 10.0), (100.0, 10.0), (100.0, 70.0), (20.0, 70.0)]
    same = PW.render(img, [q], [list(q)])
    check("an unpulled quad changes nothing", np.abs(same.astype(int) - img.astype(int)).max() <= 1)
    moved = PW.render(img, [q], [[(x + 6, y) for x, y in q]])
    check("pulling every corner 6 px right moves the whole layer 6 px",
          np.abs(moved[40, 30:90].astype(int) - img[40, 24:84].astype(int)).max() <= 1)

    # Two planes sharing the edge x = 60.
    left = [(10.0, 10.0), (60.0, 10.0), (60.0, 70.0), (10.0, 70.0)]
    right = [(60.0, 10.0), (110.0, 10.0), (110.0, 70.0), (60.0, 70.0)]
    pulled_l = [(10.0, 10.0), (60.0, 2.0), (60.0, 70.0), (10.0, 70.0)]
    pulled_r = [(60.0, 2.0), (110.0, 10.0), (110.0, 70.0), (60.0, 70.0)]
    out = PW.render(img, [left, right], [pulled_l, pulled_r])
    seam = np.abs(out[10:68, 59].astype(int) - out[10:68, 60].astype(int)).max()
    inner = np.abs(img[10:68, 59].astype(int) - img[10:68, 60].astype(int)).max()
    check("two planes sharing an edge stay joined when it is pulled (seam %d, natural step %d)"
          % (seam, inner), seam <= inner + 6)
    check("...and the content near the pulled corner moved", not np.array_equal(out[5:15, 55:65], img[5:15, 55:65]))

    # ---- the tool --------------------------------------------------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(120, 80, background=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[...] = img
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    ctl.set_tool("perspective_warp")
    # The second starts a little off the first one's corner (a press ON a corner
    # drags it) and its near corners snap onto the first quad's.
    for a, b in (((10, 10), (60, 70)), ((66, 14), (110, 70))):
        ctl.begin_stroke(*a)
        ctl.motion_stroke(*b)
        ctl.end_stroke(*b)
    t = ctl.tool
    check("Layout: two quads, the second's corners joined to the first's",
          len(t._layout) == 2 and t._layout[1][0] == t._layout[0][1] and t._layout[1][3] == t._layout[0][2])
    check("...and nothing has changed yet", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.set_tool_option("mode", "warp")
    ctl.begin_stroke(60, 10)
    ctl.motion_stroke(60, 5)
    ctl.end_stroke(60, 2)
    check("Warp: dragging the shared corner moves it in BOTH quads",
          t._warp[0][1] == (60.0, 2.0) and t._warp[1][0] == (60.0, 2.0))
    check("...and the layer follows on release", not np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    pos = ctl.history.position
    ctl.finish_tool()
    check("Enter applies it as one undo step", ctl.history.position == pos + 1
          and ctl.history_labels()[-1] == "Perspective Warp")
    ctl.undo()
    check("...undoable", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.set_tool("perspective_warp")
    ctl.begin_stroke(10, 10)
    ctl.motion_stroke(60, 70)
    ctl.end_stroke(60, 70)
    ctl.set_tool_option("mode", "warp")
    ctl.begin_stroke(60, 70)
    ctl.end_stroke(70, 75)
    ctl.cancel_stroke()
    check("Escape puts the layer back", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img)
          and ctl.tool._layout == [])
    print("\nall perspective-warp checks passed")


if __name__ == "__main__":
    main()
