#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Stroke sessions: one drag, one undo entry, and free live preview.

Run: python3 tests/test_stroke.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.history import HistoryStack
from ochre.engine.palette import grayscale
from ochre.engine.stroke import StrokeSession


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def dab(session, x, y, radius=4, colour=(255, 0, 0, 255)):
    """Simulate one brush dab: checkpoint, then write."""
    r = Rect.from_points(x - radius, y - radius, x + radius, y + radius)
    session.touch(r)
    clipped = r.clipped_to(session.surface.width, session.surface.height)
    if clipped is not None:
        session.surface.fill(clipped, colour)


def main():
    # ---- one drag becomes one entry -------------------------------------
    d = Document(512, 512)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    original = d.cell(lay).pixels.copy()

    h = HistoryStack()
    s = StrokeSession(d, lay, block=64)
    for i in range(200):                       # 200 dabs along a diagonal
        dab(s, 20 + i, 20 + i)
    check("many dabs recorded", s.dab_count == 200)
    check("stroke tracked a region", s.region is not None)

    cmd = s.commit()
    h.push(cmd, "Brush stroke")
    check("200 dabs produce exactly ONE history entry", len(h) == 1,
          "-- got %d" % len(h))

    painted = d.cell(lay).pixels.copy()
    check("stroke actually painted", not np.array_equal(painted, original))

    h.undo(d)
    check("undo restores the whole stroke exactly",
          np.array_equal(d.cell(lay).pixels, original))
    h.redo(d)
    check("redo reapplies the whole stroke exactly",
          np.array_equal(d.cell(lay).pixels, painted))

    # ---- cost is O(area), not O(dabs) -----------------------------------
    d = Document(512, 512)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))

    s = StrokeSession(d, lay, block=64)
    for _ in range(500):
        dab(s, 100, 100, radius=2)             # 500 dabs, same tiny spot
    check("repeated dabs on one spot save one block", s.saved_blocks == 1,
          "-- saved %d blocks" % s.saved_blocks)

    s2 = StrokeSession(d, lay, block=64)
    dab(s2, 100, 100, radius=2)
    check("one dab saves the same single block", s2.saved_blocks == 1)
    check("500 dabs cost the same as 1 in checkpoints",
          s.saved_blocks == s2.saved_blocks)

    # ---- row-run coalescing ---------------------------------------------
    d = Document(512, 64)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    s = StrokeSession(d, lay, block=64)
    s.touch(Rect(0, 0, 512, 8))                # one wide horizontal sweep
    check("wide sweep covers every block in the row", s.saved_blocks == 8,
          "-- %d blocks" % s.saved_blocks)
    check("adjacent blocks merged into ONE copy", len(s._scratch) == 1,
          "-- %d separate copies" % len(s._scratch))

    # Non-adjacent touches must stay separate rather than covering the gap.
    s2 = StrokeSession(d, lay, block=64)
    s2.touch(Rect(0, 0, 8, 8))
    s2.touch(Rect(400, 0, 8, 8))
    check("disjoint touches stay separate copies", len(s2._scratch) == 2)

    # ---- restore() is live preview --------------------------------------
    d = Document(128, 128)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (10, 20, 30, 255))
    pristine = d.cell(lay).pixels.copy()

    s = StrokeSession(d, lay, block=32)
    # A shape tool: draw, restore, redraw, restore, redraw...
    for i in range(1, 6):
        s.restore()
        r = Rect(10, 10, 10 * i, 10 * i)
        s.touch(r)
        d.cell(lay).fill(r, (255, 255, 0, 255))
        check("preview iteration %d drew" % i,
              tuple(d.cell(lay).pixels[10, 10]) == (255, 255, 0, 255))

    s.restore()
    check("restore() returns the canvas to pristine",
          np.array_equal(d.cell(lay).pixels, pristine))

    # The committed result must match the last preview exactly -- they are
    # the same buffer, so they cannot disagree.
    final = Rect(10, 10, 50, 50)
    s.touch(final)
    d.cell(lay).fill(final, (0, 255, 255, 255))
    previewed = d.cell(lay).pixels.copy()
    cmd = s.commit()
    check("committed pixels are exactly what the preview showed",
          np.array_equal(d.cell(lay).pixels, previewed))
    cmd.undo(d)
    check("undo after previewing restores pristine",
          np.array_equal(d.cell(lay).pixels, pristine))

    # restore() over a sub-rect touches only that sub-rect.
    d = Document(64, 64)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (1, 1, 1, 255))
    s = StrokeSession(d, lay, block=16)
    s.touch(d.bounds)
    d.cell(lay).fill(d.bounds, (9, 9, 9, 255))
    s.restore(Rect(0, 0, 16, 16))
    check("partial restore restores its region",
          tuple(d.cell(lay).pixels[0, 0]) == (1, 1, 1, 255))
    check("partial restore leaves the rest alone",
          tuple(d.cell(lay).pixels[40, 40]) == (9, 9, 9, 255))

    # ---- cancel ----------------------------------------------------------
    d = Document(128, 128)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (7, 7, 7, 255))
    before = d.cell(lay).pixels.copy()
    h = HistoryStack()
    s = StrokeSession(d, lay, block=32)
    for i in range(20):
        dab(s, 20 + i * 3, 40)
    s.cancel()
    check("cancel rolls the canvas back",
          np.array_equal(d.cell(lay).pixels, before))
    check("cancel leaves no history", len(h) == 0)

    # ---- auto-commit under a long stroke --------------------------------
    d = Document(512, 512)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    start = d.cell(lay).pixels.copy()
    s = StrokeSession(d, lay, block=32, max_blocks=8)   # deliberately tiny
    for i in range(0, 480, 16):
        dab(s, i + 8, i + 8, radius=6)
    check("long stroke auto-committed", len(s._committed) > 0,
          "-- no auto-commit happened")
    cmd = s.commit()
    check("auto-committed stroke still yields one command", cmd is not None)
    after = d.cell(lay).pixels.copy()
    # Note the idiom: undo() RETURNS the command that redoes. Reusing the
    # original would just re-apply the same direction again.
    redo_cmd, _ = cmd.undo(d)
    check("auto-committed stroke undoes completely",
          np.array_equal(d.cell(lay).pixels, start))
    undo_again, _ = redo_cmd.undo(d)
    check("auto-committed stroke redoes completely",
          np.array_equal(d.cell(lay).pixels, after))
    undo_again.undo(d)
    check("auto-committed stroke cycles without drift",
          np.array_equal(d.cell(lay).pixels, start))

    # ---- nothing painted, nothing recorded ------------------------------
    d = Document(64, 64)
    lay = d.add_layer("a")
    s = StrokeSession(d, lay)
    check("a stroke that touched nothing commits nothing", s.commit() is None)
    s2 = StrokeSession(d, lay)
    s2.touch(Rect(500, 500, 10, 10))          # entirely off-canvas
    check("off-canvas touches record nothing", s2.commit() is None)

    # ---- index-locked strokes -------------------------------------------
    d = Document(64, 64)
    d.bind_palette(grayscale())
    il = d.add_layer("il", planes=("index", "rgba"), authoritative=("index",))
    d.cell(il).fill(d.bounds, 0, name="index")
    start = d.cell(il).plane("index").copy()

    s = StrokeSession(d, il, block=16)
    for i in range(30):
        r = Rect.from_points(10 + i, 10, 12 + i, 20)
        s.touch(r)
        d.cell(il).fill(r, 42, name="index")
    cmd = s.commit()
    check("index stroke painted indices", int(d.cell(il).plane("index")[15, 20]) == 42)
    check("derived colour followed", tuple(d.cell(il).pixels[15, 20]) == (42, 42, 42, 255))
    cmd.undo(d)
    check("index stroke undoes on the index plane",
          np.array_equal(d.cell(il).plane("index"), start))
    check("derived colour rebuilt after undo",
          tuple(d.cell(il).pixels[15, 20]) == (0, 0, 0, 255))

    # ---- cost check ------------------------------------------------------
    d = Document(2048, 2048)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    s = StrokeSession(d, lay, block=256)
    t0 = time.perf_counter()
    for i in range(1000):
        dab(s, 100 + (i % 800), 100 + (i % 600), radius=8)
    t_dabs = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    cmd = s.commit()
    t_commit = (time.perf_counter() - t0) * 1000
    per_dab = t_dabs / 1000.0
    check("per-dab checkpoint cost stays sub-millisecond", per_dab < 1.0,
          "-- %.3f ms/dab" % per_dab)
    print("    (1000 dabs: %.2f ms total, %.3f ms/dab; commit %.1f ms, %d KB)"
          % (t_dabs, per_dab, t_commit, cmd.nbytes() // 1024))

    print("\nall stroke checks passed")


if __name__ == "__main__":
    main()
