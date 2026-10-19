#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Choosing which addons load, and saving to addon formats from Save As --
including an ordinary RGB drawing to an indexed format (SHP).

Run: python3 tests/test_addonsave.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import palettefiles as PF
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller, NeedsIndexed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ADDONS = os.path.join(ROOT, "addons")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(state):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    for ident in ctl.addons.addons:
        ctl.addons.trust(ident)
    return ctl


def main():
    state = tempfile.mkdtemp()
    ctl = fresh(state)
    ids = sorted(ctl.addons.addons)
    check("addons are discovered (%s)" % ids, "cnc" in ids)
    ctl.load_addons(only=["cnc"])
    states = {a.id: a.state for a in ctl.addons.addons.values()}
    check("Choose Addons: only the chosen ones load", states["cnc"] == "loaded"
          and all(s != "loaded" for k, s in states.items() if k != "cnc"), str(states))
    check("...and the choice is remembered", "cnc" not in ctl.addons.skipped()
          and ctl.addons.skipped() == set(ids) - {"cnc"})
    ctl2 = fresh(state)
    ctl2.load_addons()
    s2 = {a.id: a.state for a in ctl2.addons.addons.values()}
    check("...next start loads just those", s2["cnc"] == "loaded"
          and all(s == "not chosen" for k, s in s2.items() if k != "cnc"), str(s2))
    ctl2.load_addons(only=[])
    check("unticking a loaded addon unloads it (its formats go too)",
          ctl2.addons.addons["cnc"].state != "loaded" and not ctl2.addon_formats(writing=True))
    ctl2.load_addons(all_addons=True)
    check("Load All Addons loads every one and forgets the choice",
          all(a.state == "loaded" for a in ctl2.addons.addons.values() if a.manifest.enabled)
          and ctl2.addons.skipped() == set())

    # ---- saving to addon formats ----------------------------------------------------
    ctl = fresh(state)
    ctl.load_addons(all_addons=True)
    labels = [l for l, _n, _e, _i in ctl.addon_formats(writing=True)]
    check("Save As offers the addon's formats by name (%s)" % labels,
          any("SHP" in l for l in labels) and any("TMP" in l for l in labels))
    ctl.new_document(40, 30, background=None)
    lay = ctl.active_layer
    px = ctl.doc.cell(lay).plane("rgba")
    px[5:25, 5:35] = (200, 40, 40, 255)
    px[10:20, 10:30] = (40, 160, 40, 255)
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    out = os.path.join(tempfile.mkdtemp(), "drawing.shp")
    try:
        ctl.save_path(out, format="cnc.shp")
        asked = False
    except NeedsIndexed:
        asked = True
    check("an RGB drawing to SHP asks for a palette first (SHP stores indices)", asked
          and not os.path.exists(out))
    pal = PF.builtin("PICO-8 (16)")
    ctl.save_path(out, format="cnc.shp", convert={"palette": pal, "dither": "none"})
    check("...with one, it writes a SHP", os.path.exists(out) and os.path.getsize(out) > 8)
    check("...from an indexed COPY: the open document is still RGB",
          not lay.index_locked and ctl.doc.palette is None)
    ctl3 = fresh(state)
    ctl3.load_addons(all_addons=True)
    ctl3.open_path(out)
    back = ctl3.doc
    il = [l for l in back.layers() if l.index_locked]
    # A SHP stores indices and no palette, so compare INDICES: green and red
    # must land on the entries PICO-8 gives them, the empty corner on 0.
    from ochre.engine.quantize import quantize
    want = quantize(np.array([[[40, 160, 40, 255], [200, 40, 40, 255]]], np.uint8), pal)[0]
    idx = back.cell(il[0]).plane("index") if il else None
    check("...which opens again as an indexed sprite with the drawing's indices",
          idx is not None and int(idx[15, 20]) == int(want[0]) and int(idx[7, 7]) == int(want[1])
          and int(idx[2, 2]) == 0, str((idx[15, 20], idx[7, 7], idx[2, 2], want) if idx is not None else il))
    ctl.save_path(out)
    check("Save again re-converts with the same palette, without asking", os.path.exists(out))
    tem = os.path.join(tempfile.mkdtemp(), "tree.tem")
    ctl.save_path(tem, format="cnc.shp", convert={"palette": pal})
    with open(tem, "rb") as fh:
        head = fh.read(8)
    check("an extension two formats share writes the one picked (.tem as SHP)",
          head[:2] == b"\x00\x00" and ctl.doc.meta.get("format") == "cnc.shp")
    print("\nall addon-save checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
