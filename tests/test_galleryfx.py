#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Filter Gallery families ported from Kooka Pictura (Sketch, Brush
Strokes, Texture, Artistic and the rest): every filter driven the way the
editor drives it, plus the properties each must keep.

For each filter:
  * the registry knows it, under its family, and the Filter Gallery's tree
    and the Effects menu (hence the command search) list it;
  * it is deterministic, keeps the shape, and keeps alpha;
  * every slider at its minimum and at its maximum, and every choice,
    renders without a numeric warning (the suite runs with RuntimeWarning
    as an error) and still keeps alpha;
  * a seed, where it has one, re-rolls the pattern;
  * applied through the controller, as the menu does, it changes only the
    selection and is one undo step that undo reverses;
  * as a one-entry Filter Gallery stack it gives exactly the direct result;
  * recorded as an action and replayed, it gives the same result again.

Images are small (40x40) on purpose: the suite runs twice.

Run: python3 tests/test_galleryfx.py
"""

import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine.effects import EffectRegistry
from ochre.engine.ini import IniDB
from ochre.engine.props import ChoiceProperty, FloatProperty, IntProperty
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry(IniDB(DATA))

# (module, tuple name, family)
FAMILIES = (
    ("sketch", "SKETCH", "Sketch"),
    ("brushstrokes", "BRUSH_STROKES", "Brush Strokes"),
    ("texture", "TEXTURE", "Texture"),
    ("artistic", "ARTISTIC", "Artistic"),
    ("galleryextra", "EXTRA", None),
)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def picture(h=40, w=40):
    """Gradients, a dark block, a bright disc and a transparent corner:
    edges, flats and alpha all at once."""
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = xs * 255 // (w - 1)
    img[..., 1] = ys * 255 // (h - 1)
    img[..., 2] = 140
    img[8:20, 6:18, :3] = (30, 40, 25)
    disc = (xs - 28) ** 2 + (ys - 26) ** 2 < 49
    img[disc, :3] = (235, 220, 200)
    img[..., 3] = 255
    img[:6, :6, 3] = 0
    img[:6, 34:, 3] = 128
    return img


def families():
    out = []
    for mod, var, family in FAMILIES:
        try:
            classes = getattr(importlib.import_module("ochre.engine." + mod), var)
        except (ImportError, AttributeError):
            continue
        out.append((family, classes))
    return out


def extremes(fx):
    """Token overrides: all sliders low, all high, then each choice."""
    low, high, choices = {}, {}, []
    for prop in fx.props:
        if isinstance(prop, (IntProperty, FloatProperty)):
            low[prop.name], high[prop.name] = prop.minimum, prop.maximum
        elif isinstance(prop, ChoiceProperty):
            choices.extend({prop.name: c} for c in prop.choices if c != prop.value)
    return [low, high] + choices


def main():
    img = picture()
    fams = families()
    names = []

    # ---- each filter on its own -------------------------------------------
    for family, classes in fams:
        for cls in classes:
            n = cls.name
            names.append(n)
            check("%s is registered%s" % (n, " under " + family if family else ""),
                  n in REG and (family is None or REG.category(n) == family))
            a = REG.create(n).apply(img)
            b = REG.create(n).apply(img)
            check("%s: shape, determinism, alpha kept" % n,
                  a.shape == img.shape and a.dtype == np.uint8 and np.array_equal(a, b)
                  and np.array_equal(a[..., 3], img[..., 3]))
            check("%s changes the picture" % n, not np.array_equal(a[..., :3], img[..., :3]))
            bad = []
            for over in extremes(REG.create(n)):
                out = REG.create(n, **over).apply(img)
                if out.shape != img.shape or not np.array_equal(out[..., 3], img[..., 3]):
                    bad.append(over)
            check("%s: every slider end and every choice renders" % n, not bad, str(bad))
            if "seed" in REG.create(n).props:
                # Some settings draw no noise at all (Paint Daubs' Simple
                # brush); then some choice must.
                rerolls = []
                for over in [{}] + [o for o in extremes(REG.create(n)) if len(o) == 1]:
                    rerolls.append(not np.array_equal(REG.create(n, **over).apply(img),
                                                      REG.create(n, seed=7, **over).apply(img)))
                    if rerolls[-1]:
                        break
                check("%s: the seed re-rolls the pattern" % n, rerolls[-1])

    # ---- the UI lists them ---------------------------------------------------
    from ochre.ui import gallery
    check("the Filter Gallery shows every family",
          all(f in gallery.CATEGORIES for f, _ in fams if f))
    check("...and every new filter sits in a family it shows",
          all(REG.category(n) in gallery.CATEGORIES for n in names))

    # ---- through the controller, as the menu does ---------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(40, 40, background=None)
    lay = ctl.active_layer
    cell = ctl.doc.cell(lay)
    cell.plane("rgba")[...] = img
    cell._touched(ctl.doc.bounds)
    ctl.selection.select_rect((8, 8, 24, 24), antialias=False)
    inside = ctl.selection.mask > 0
    bad_sel, bad_undo, bad_stack, bad_replay = [], [], [], []
    for n in names:
        ctl.start_recording("probe")
        ctl.apply_effect(n)
        rec = ctl.stop_recording()
        got = cell.plane("rgba").copy()
        direct = REG.create(n).apply(img)
        if not (np.array_equal(got[~inside], img[~inside])
                and np.array_equal(got[inside], direct[inside])
                and ctl.history_labels()[-1] == REG.create(n).label):
            bad_sel.append(n)
        ctl.undo()
        if not np.array_equal(cell.plane("rgba"), img):
            bad_undo.append(n)
        stacked = REG.create("filter_stack", stack=json.dumps(
            [{"name": n, "params": {}, "enabled": True}])).apply(img)
        if not np.array_equal(stacked, direct):
            bad_stack.append(n)
        ctl.play_action(rec["steps"])
        if not (rec["steps"] and rec["steps"][0]["op"] == "apply_effect"
                and np.array_equal(cell.plane("rgba"), got)):
            bad_replay.append(n)
        ctl.undo()
    check("applied as the menu does, each changes only the selection, as one step",
          not bad_sel, str(bad_sel))
    check("...and undo restores the layer", not bad_undo, str(bad_undo))
    check("a one-entry Filter Gallery stack is the filter itself", not bad_stack, str(bad_stack))
    check("recorded as an action, each replays to the same pixels", not bad_replay,
          str(bad_replay))

    # ---- a smart object takes them as smart filters -----------------------------
    ctl.selection.clear()
    smart = ctl.convert_to_smart(lay)
    for n in names[:3]:
        ctl.apply_effect(n)
    check("on a smart object they become smart filters",
          [f["name"] for f in smart.params["filters"]] == names[:3])

    # ---- the Effects menu and the command search ----------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.commandsearch import build_index
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-galleryfx-"))
    win = MainWindow(ctl, UiSettings(DATA))
    paths = {tuple(e.path) for e in build_index(win)}
    missing = [n for n in names
               if ("Effects", REG.category(n), REG.label(n)) not in paths]
    check("the Effects menu, and so the command search, lists every one", not missing,
          str(missing))
    win.close()
    del app
    print("\nall %d gallery filters checked" % len(names))


if __name__ == "__main__":
    main()
