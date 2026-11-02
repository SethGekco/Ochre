#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The save cache: a .ochre save encodes only the planes that changed.

Driven through the controller's own Save (save_path) and its own tools, so
what is tested is what File > Save does. Every check that reuses bytes also
checks the file: it must load back to the live document's pixels, and be
byte-identical to a save made without any cache.

Run: python3 tests/test_savecache.py
"""

import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.fileio import ochre_format
from ochre.engine.fileio.ochre_format import EncodeCache
from ochre.engine.palette import Palette
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def same_pixels(doc, path):
    """Every authoritative plane of every cell in the file equals the live
    document's, frame by frame (cold frames included)."""
    loaded, _ = ochre_format.load(path)
    for f in doc.frames:
        doc.ensure_warm(f)
    for (lid, fid), surface in doc.cells.items():
        other = loaded.cells.get((lid, fid))
        for name in surface.authoritative:
            mine = surface.planes.get(name)
            if mine is None:
                continue
            if other is None or not np.array_equal(other.plane(name), mine):
                return False
    return True


def same_bytes(a, b):
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        if za.namelist() != zb.namelist():
            return False
        return all(za.read(n) == zb.read(n) for n in za.namelist())


def stroke(ctl, layer, x0, y0, x1, y1, colour):
    ctl.set_active_layer(layer)
    ctl.set_primary(colour)
    ctl.begin_stroke(x0, y0)
    ctl.motion_stroke((x0 + x1) / 2, (y0 + y1) / 2)
    ctl.end_stroke(x1, y1)


def main():
    tmp = tempfile.mkdtemp(prefix="ochre-recovery-test-")
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=os.path.join(tmp, "state"))
    ctl.new_document(160, 120)
    ctl.set_tool("brush", size=12)
    layers = [ctl.active_layer] + [ctl.add_layer("L%d" % i) for i in range(5)]
    for i, lay in enumerate(layers[1:]):
        stroke(ctl, lay, 10 + 20 * i, 10, 30 + 20 * i, 100, (40 * i, 200, 90, 255))
    path = os.path.join(tmp, "doc.ochre")
    plain = os.path.join(tmp, "plain.ochre")

    ctl.save_path(path)
    cache = ctl.save_cache
    check("the first save encodes every plane", cache.misses == 6 and cache.hits == 0,
          (cache.hits, cache.misses))
    check("...and loads back to the document", same_pixels(ctl.doc, path))

    stroke(ctl, layers[3], 5, 60, 150, 70, (255, 0, 0, 255))
    ctl.save_path(path)
    check("after a stroke on one layer, Save encodes that layer alone",
          cache.misses == 7 and cache.hits == 5, (cache.hits, cache.misses))
    check("...the file holds the new stroke", same_pixels(ctl.doc, path))
    ochre_format.save(ctl.doc, plain, selection=ctl.selection)
    check("...and is byte-identical to a save made without a cache", same_bytes(path, plain))

    ctl.undo()
    ctl.save_path(path)
    check("undo is a change too: the undone layer is encoded again",
          cache.misses == 8 and same_pixels(ctl.doc, path))

    # A write that skips _touched leaves the revision alone. The explicit
    # save's CRC check must catch it rather than write the old bytes.
    cell = ctl.doc.cell(layers[2])
    cell.plane("rgba")[50:60, 50:60] = (1, 2, 3, 255)
    ctl.save_path(path)
    check("a write that bypasses the revision is caught by Save's check",
          cache.stale == 1 and same_pixels(ctl.doc, path), cache.stale)

    # History snapshots are deep copies: restoring one must not bring back a
    # surface whose (serial, revision) names other pixels.
    ctl.new_snapshot("before")
    stroke(ctl, layers[4], 0, 0, 159, 119, (0, 0, 255, 255))
    ctl.save_path(path)
    ctl.restore_snapshot(0)
    stroke(ctl, layers[4], 159, 0, 0, 119, (0, 255, 0, 255))
    stale = cache.stale
    ctl.save_path(path)
    check("restoring a History snapshot, then editing, saves the new pixels",
          same_pixels(ctl.doc, path) and cache.stale == stale)

    check("the cache holds one entry per plane written, no history",
          len(cache) == 6 and cache.nbytes() < 6 * 160 * 120 * 4, (len(cache), cache.nbytes()))

    # ---- frames: cold ones are saved, and cached --------------------------
    ctl.new_document(48, 40)
    lay = ctl.active_layer
    for i in range(11):
        ctl.add_frame()
    for i, f in enumerate(ctl.doc.frames):
        ctl.set_current_frame(i)
        stroke(ctl, lay, 2, 2 + i * 3, 45, 2 + i * 3, (20 * i, 10, 200, 255))
    cold = [f for f in ctl.doc.frames if f.residency == "cold"]
    check("(setup: some frames are cold)", len(cold) >= 3, len(cold))
    path = os.path.join(tmp, "frames.ochre")
    ctl.save_path(path)
    check("a cold frame's cells are saved (they used to load back blank)",
          same_pixels(ctl.doc, path))
    for i in range(len(ctl.doc.frames)):
        ctl.set_current_frame(i)
    cache = ctl.save_cache
    misses = cache.misses
    ctl.set_current_frame(0)
    stroke(ctl, lay, 40, 0, 40, 39, (255, 255, 0, 255))
    ctl.save_path(path)
    check("after an edit to one frame, only that frame is encoded -- cold ones too are reused",
          cache.misses == misses + 1 and same_pixels(ctl.doc, path), cache.misses - misses)
    ochre_format.save(ctl.doc, plain, selection=ctl.selection)
    check("...byte-identical to an uncached save", same_bytes(path, plain))

    # ---- an index-locked layer: its derived rgba follows the palette ------
    entries = np.zeros((256, 4), np.uint8)
    entries[:, 3] = 255
    entries[1] = (200, 0, 0, 255)
    doc = Document(32, 32, palette=Palette(entries, "test"))
    ilay = doc.add_layer("ix", planes=("index", "rgba"), authoritative=("index",))
    doc.cell(ilay).fill(doc.bounds, 1, name="index")
    cache = EncodeCache()
    path = os.path.join(tmp, "indexed.ochre")
    ochre_format.save(doc, path, cache=cache)
    doc.set_palette_entry(1, (0, 0, 200, 255))
    ochre_format.save(doc, path, cache=cache)
    with zipfile.ZipFile(path) as z:
        rgba = ochre_format._decode_plane(z.read("frames/0000/layers/%s.rgba.png" % ilay.id), "rgba")
    check("a palette edit re-encodes the derived rgba it recolours (and only that)",
          tuple(rgba[0, 0]) == (0, 0, 200, 255) and cache.hits == 1 and cache.misses == 3,
          (tuple(rgba[0, 0]), cache.hits, cache.misses))

    # ---- the snapshot an autosave writes from a worker --------------------
    doc = Document(64, 64)
    lay = doc.add_layer("a")
    doc.cell(lay).fill(doc.bounds, (9, 9, 9, 255))
    cache = EncodeCache()
    snap = ochre_format.Snapshot(doc, None, 0, cache, verify=False)
    doc.cell(lay).fill(doc.bounds, (8, 8, 8, 255))        # edited while it was "in flight"
    ok = snap.write(os.path.join(tmp, "torn.ochre"))
    check("a plane that changes while being written is reported torn, and not cached",
          not ok and snap.torn and len(cache) == 0)
    snap = ochre_format.Snapshot(doc, None, 0, cache, verify=False)
    check("...the retry writes it whole", snap.write(os.path.join(tmp, "torn.ochre"))
          and same_pixels(doc, os.path.join(tmp, "torn.ochre")))


if __name__ == "__main__":
    main()
