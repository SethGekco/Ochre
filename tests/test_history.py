#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""History: inverse-generating commands, eviction, grouping, frame awareness.

The central property under test is that undo/redo round-trips are EXACT and
repeatable. A history that is nearly right is worse than none, because the
drift only shows up after several cycles.

Run: python3 tests/test_history.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.commands import (CellSnapshot, CompoundCommand, PaletteDelta,
                                   PixelDelta, PropertyDelta)
from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.history import HistoryStack
from ochre.engine.palette import grayscale


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- PixelDelta round-trip ------------------------------------------
    d = Document(32, 32)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (10, 20, 30, 255))
    original = d.cell(lay).pixels.copy()

    region = Rect(4, 4, 8, 8)
    delta = PixelDelta.capture(d, lay.id, d.frame.id, region, "Paint")
    check("capture records something", delta is not None and delta.nbytes() > 0)

    d.cell(lay).fill(region, (200, 100, 50, 255))
    check("edit landed", tuple(d.cell(lay).pixels[4, 4]) == (200, 100, 50, 255))

    redo, rect = delta.undo(d)
    check("undo restores exactly", np.array_equal(d.cell(lay).pixels, original))
    check("undo reports its rect", rect == region)
    check("undo generates its own redo", redo is not None)

    redo2, _ = redo.undo(d)
    check("redo re-applies exactly",
          tuple(d.cell(lay).pixels[4, 4]) == (200, 100, 50, 255))
    check("redo generates an undo in turn", redo2 is not None)

    # Ten FULL cycles, asserting both ends each time. Drift in an
    # inverse-generating stack would compound, so checking only the final
    # state could miss an error that cancels itself out.
    edited = d.cell(lay).pixels.copy()
    cur = redo2                       # currently holds "restore the original"
    drift = None
    for cycle in range(10):
        cur, _ = cur.undo(d)
        if not np.array_equal(d.cell(lay).pixels, original):
            drift = "undo half of cycle %d" % cycle
            break
        cur, _ = cur.undo(d)
        if not np.array_equal(d.cell(lay).pixels, edited):
            drift = "redo half of cycle %d" % cycle
            break
    check("10 full undo/redo cycles are bit-exact at both ends", drift is None,
          "-- drifted at %s" % drift)
    # Leave the canvas in the original state for what follows.
    cur.undo(d)

    # A delta for a vanished cell must fail softly.
    gone = PixelDelta.capture(d, lay.id, d.frame.id, region)
    d.cells.clear()
    out, r = gone.undo(d)
    check("delta for a missing cell is a no-op", out is None and r is None)

    # ---- only authoritative planes are recorded -------------------------
    d = Document(16, 16)
    d.bind_palette(grayscale())
    il = d.add_layer("il", planes=("index", "rgba"), authoritative=("index",))
    d.cell(il).fill(Rect(0, 0, 8, 8), 100, name="index")
    delta = PixelDelta.capture(d, il.id, d.frame.id, Rect(0, 0, 8, 8))
    check("index-locked delta records index only",
          sorted(delta._planes) == ["index"], "-- got %r" % sorted(delta._planes))

    d.cell(il).fill(Rect(0, 0, 8, 8), 7, name="index")
    delta.undo(d)
    check("index restored", int(d.cell(il).plane("index")[0, 0]) == 100)
    check("derived cache rebuilt on undo, not stored",
          tuple(d.cell(il).pixels[0, 0]) == (100, 100, 100, 255))

    # Index-locked history really is cheaper than plain RGBA.
    d2 = Document(16, 16)
    plain = d2.add_layer("plain")
    d2.cell(plain).fill(Rect(0, 0, 8, 8), (100, 100, 100, 255))
    rgba_delta = PixelDelta.capture(d2, plain.id, d2.frame.id, Rect(0, 0, 8, 8))
    check("index-locked delta is smaller than RGBA",
          delta.nbytes() < rgba_delta.nbytes(),
          "-- index %d vs rgba %d" % (delta.nbytes(), rgba_delta.nbytes()))

    # ---- PropertyDelta ---------------------------------------------------
    d = Document(8, 8)
    lay = d.add_layer("a")
    lay.opacity = 255
    cmd = PropertyDelta(lay.id, "opacity", 255, "Change opacity")
    lay.opacity = 100
    cmd.undo(d)
    check("property restored", lay.opacity == 255)
    back, _ = PropertyDelta(lay.id, "opacity", 100).undo(d)
    check("property redo works", lay.opacity == 100)
    check("property delta is tiny", cmd.nbytes() < 256)

    cmd = PropertyDelta("nonexistent", "opacity", 5)
    out, r = cmd.undo(d)
    check("property delta for a missing target is a no-op",
          out is None and r is None)

    # ---- PaletteDelta ----------------------------------------------------
    d = Document(8, 8)
    pal = grayscale()
    d.bind_palette(pal)
    il = d.add_layer("il", planes=("index", "rgba"), authoritative=("index",))
    d.cell(il).fill(d.bounds, 5, name="index")
    before = pal.rgba(5)
    cmd = PaletteDelta(5, before)
    d.set_palette_entry(5, (255, 0, 0, 255))
    check("palette edit repaints", tuple(d.cell(il).pixels[0, 0]) == (255, 0, 0, 255))
    cmd.undo(d)
    check("palette undo restores the colour", pal.rgba(5) == before)
    check("palette undo repaints the pixels",
          tuple(d.cell(il).pixels[0, 0]) == before)
    check("palette delta costs almost nothing", cmd.nbytes() < 256)

    # ---- the stack -------------------------------------------------------
    d = Document(16, 16)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    h = HistoryStack()
    check("nothing to undo initially", not h.can_undo and not h.can_redo)

    states = []
    for i in range(1, 6):
        states.append(d.cell(lay).pixels.copy())
        cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds, "Step %d" % i)
        d.cell(lay).fill(d.bounds, (i * 40, 0, 0, 255))
        h.push(cmd, "Step %d" % i)

    check("five entries", len(h) == 5)
    check("labels in order", h.labels() == ["Step %d" % i for i in range(1, 6)])
    check("can undo", h.can_undo and not h.can_redo)

    for i in range(4, -1, -1):
        h.undo(d)
        check("undo to state %d" % i, np.array_equal(d.cell(lay).pixels, states[i]))
    check("exhausted undo", not h.can_undo and h.can_redo)
    check("extra undo is harmless", h.undo(d) is None)

    for i in range(5):
        h.redo(d)
    check("redo returns to the newest state",
          tuple(d.cell(lay).pixels[0, 0]) == (200, 0, 0, 255))
    check("extra redo is harmless", h.redo(d) is None)

    # Pushing after an undo discards the redo branch.
    h.undo(d)
    h.undo(d)
    check("redo available mid-stack", h.can_redo)
    cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds, "New branch")
    d.cell(lay).fill(d.bounds, (1, 2, 3, 255))
    h.push(cmd, "New branch")
    check("new change truncates the redo branch", not h.can_redo)
    check("truncated stack length", len(h) == 4, "-- got %d" % len(h))

    # goto() addresses any point, which is what a history panel needs.
    h.goto(0, d)
    check("goto rewinds", np.array_equal(d.cell(lay).pixels, states[1]))
    h.goto(len(h) - 1, d)
    check("goto fast-forwards", tuple(d.cell(lay).pixels[0, 0]) == (1, 2, 3, 255))
    h.goto(-1, d)
    check("goto(-1) reaches the initial state",
          np.array_equal(d.cell(lay).pixels, states[0]))

    # ---- eviction --------------------------------------------------------
    d = Document(8, 8)
    lay = d.add_layer("a")
    h = HistoryStack()
    h.max_entries = 5
    h.min_entries = 2
    for i in range(12):
        cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds, "S%d" % i)
        d.cell(lay).fill(d.bounds, (i, 0, 0, 255))
        h.push(cmd, "S%d" % i)
    check("entry cap enforced", len(h) == 5, "-- got %d" % len(h))
    check("oldest evicted first", h.labels()[0] == "S7", "-- got %r" % h.labels())
    check("position stays valid", 0 <= h.position < len(h))
    check("still usable after eviction", h.undo(d) is not None)

    # The byte cap must not evict below the floor.
    h = HistoryStack()
    h.max_bytes = 1
    h.min_entries = 3
    h.max_entries = 100
    for i in range(10):
        cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds, "B%d" % i)
        d.cell(lay).fill(d.bounds, (i, 1, 1, 255))
        h.push(cmd, "B%d" % i)
    check("MinEntries beats MaxBytes", len(h) == 3, "-- got %d" % len(h))

    # ---- grouping --------------------------------------------------------
    d = Document(8, 8)
    lay = d.add_layer("a")
    d.cell(lay).fill(d.bounds, (0, 0, 0, 255))
    start = d.cell(lay).pixels.copy()
    h = HistoryStack()
    h.begin_group()
    for i in range(3):
        cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds)
        d.cell(lay).fill(d.bounds, (i + 1, 0, 0, 255))
        h.push(cmd)
    h.end_group("Batch")
    check("group collapses to one entry", len(h) == 1)
    check("group is labelled", h.labels() == ["Batch"])
    h.undo(d)
    check("one undo reverses the whole group",
          np.array_equal(d.cell(lay).pixels, start))
    h.redo(d)
    check("one redo reapplies the whole group",
          tuple(d.cell(lay).pixels[0, 0]) == (3, 0, 0, 255))

    # Repeated cycles catch ordering errors that a single round-trip hides:
    # a reversed compound still round-trips when its children commute.
    drift = None
    for cycle in range(5):
        h.undo(d)
        if not np.array_equal(d.cell(lay).pixels, start):
            drift = "undo %d" % cycle
            break
        h.redo(d)
        if tuple(d.cell(lay).pixels[0, 0]) != (3, 0, 0, 255):
            drift = "redo %d" % cycle
            break
    check("group survives repeated undo/redo cycles", drift is None,
          "-- drifted at %s" % drift)

    # Nested groups collapse to the outermost.
    h = HistoryStack()
    h.begin_group()
    h.begin_group()
    cmd = PixelDelta.capture(d, lay.id, d.frame.id, d.bounds)
    d.cell(lay).fill(d.bounds, (9, 9, 9, 255))
    h.push(cmd)
    check("inner end_group does not push", h.end_group("inner") is None)
    h.end_group("outer")
    check("nested groups make one entry", len(h) == 1 and h.labels() == ["outer"])

    h.begin_group()
    h.push(PixelDelta.capture(d, lay.id, d.frame.id, d.bounds))
    h.abort_group()
    check("aborted group pushes nothing", len(h) == 1)

    h2 = HistoryStack()
    h2.begin_group()
    check("empty group pushes nothing", h2.end_group("empty") is None and len(h2) == 0)

    # ---- frame-aware undo ------------------------------------------------
    d = Document(8, 8)
    lay = d.add_layer("a")
    f1 = d.frame
    f2 = d.add_frame()
    d.cell(lay, f1).fill(d.bounds, (1, 1, 1, 255))
    d.cell(lay, f2).fill(d.bounds, (2, 2, 2, 255))
    h = HistoryStack()
    cmd = PixelDelta.capture(d, lay.id, f1.id, d.bounds, "Edit frame 1")
    d.cell(lay, f1).fill(d.bounds, (9, 9, 9, 255))
    h.push(cmd, "Edit frame 1", frame_id=f1.id)

    d.set_current_frame(1)
    check("now on frame 2", d.current == 1)
    h.undo(d)
    check("undo switched to the affected frame", d.current == 0,
          "-- an undo the user cannot see is the worst kind")
    check("the right frame was restored",
          tuple(d.cell(lay, f1).pixels[0, 0]) == (1, 1, 1, 255))

    # ---- CompoundCommand ordering ---------------------------------------
    d = Document(8, 8)
    lay = d.add_layer("a")
    order = []

    class Probe:
        label = "probe"
        def __init__(self, tag): self.tag = tag
        def undo(self, doc):
            order.append(self.tag)
            return Probe(self.tag), None
        def nbytes(self): return 0

    comp = CompoundCommand([Probe("a"), Probe("b"), Probe("c")])
    comp.undo(d)
    check("compound undoes in reverse order", order == ["c", "b", "a"],
          "-- got %r" % order)

    # ---- CellSnapshot ----------------------------------------------------
    d = Document(16, 16)
    lay = d.add_layer("a")
    d.cell(lay).fill(Rect(2, 2, 4, 4), (5, 6, 7, 255))
    snap = CellSnapshot.capture(d, lay.id, d.frame.id, "Delete layer")
    saved = d.cell(lay).pixels.copy()
    d.cells.clear()
    check("cell gone", not d.has_cell(lay))
    snap.undo(d)
    check("cell restored from snapshot", d.has_cell(lay))
    check("restored pixels are exact",
          np.array_equal(d.cell(lay).pixels[2:6, 2:6], saved[2:6, 2:6]))

    absent = CellSnapshot.capture(d, "nope", d.frame.id)
    check("snapshot of a missing cell records absence", absent._existed is False)

    print("\nall history checks passed")


if __name__ == "__main__":
    main()
