#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Palette bar: palette files and folders, Reduce / Apply / Indexed /
Lock, addon palette formats -- and the toolbox and panels that scroll
instead of forcing the window taller than the screen.

Run: python3 tests/test_palettebar.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import palettefiles as PF
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ADDONS = os.path.join(ROOT, "addons")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def photo(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0] = np.linspace(0, 255, w)[None, :]
    out[..., 1] = np.linspace(255, 0, h)[:, None]
    out[..., 2] = rng.integers(0, 256, (h, w))
    out[..., 3] = 255
    return out


def ncolours(a):
    return len(np.unique(a[..., :3].reshape(-1, 3), axis=0))


def main():
    # ---- files -------------------------------------------------------------------
    d = tempfile.mkdtemp()
    pal = PF.make([(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30)], "four")
    for ext in (".gpl", ".act", ".txt", ".pal"):
        p = os.path.join(d, "four" + ext)
        PF.write(pal, p)
        back = PF.read(p)
        check("%s palettes round-trip (4 colours)" % ext,
              back.size == 4 and np.array_equal(PF.colours(back), PF.colours(pal)))
    check("a short palette's padding can never be produced by a snap",
          set(pal.candidates().tolist()) <= {0, 1, 2, 3})
    for name in PF.BUILTIN:
        b = PF.builtin(name)
        check("built-in %s has its colours" % name, b.size == int(name.split("(")[1].rstrip(")")))
    os.makedirs(os.path.join(d, "folder", "SetA"))
    os.makedirs(os.path.join(d, "folder", "SetB"))
    PF.write(pal, os.path.join(d, "folder", "SetA", "alpha.pal"))
    PF.write(pal, os.path.join(d, "folder", "SetB", "beta.gpl"))
    names = [n for n, _p in PF.files_in([os.path.join(d, "folder")])]
    check("palette folders are searched with their subfolders, named Set/name",
          names == ["SetA/alpha", "SetB/beta"], str(names))
    # A 768-byte binary palette: not a format the core reads itself.
    raw = os.path.join(d, "folder", "SetA", "game.pal")
    with open(raw, "wb") as fh:
        fh.write(bytes([(i * 7) % 64 for i in range(768)]))
    try:
        PF.read(raw)
        refused = False
    except ValueError:
        refused = True
    check("a binary .pal is left to palette providers (the core does not guess)", refused)

    # ---- reduce / apply / indexed / lock --------------------------------------------------
    img = photo()
    p8 = PF.median_cut(img, 8)
    check("median cut makes an 8-colour palette", p8.size == 8)
    check("...and mapping to it leaves 8 colours or fewer", ncolours(PF.map_rgba(img, p8)) <= 8)

    state = tempfile.mkdtemp()
    Settings.save_user(os.path.join(state, "preferences.ini"),
                       {("Palette", "Folders"): os.path.join(d, "folder"),
                        ("Palette", "Default"): "SetB/beta"})
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state)
    ctl.new_document(64, 48, background=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[...] = img
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    groups = {g for g, _n, _k in ctl.palette_sources()}
    check("the bar lists built-ins and the palette folders", {"Built-in", "Folders"} <= groups)
    ctl.restore_swatch_palette()
    check("it starts on Preferences > Palette > Default", ctl.swatch_palette is not None
          and ctl.swatch_palette.size == 4)
    try:
        ctl.load_palette_file(raw)
        no_reader = False
    except ValueError:
        no_reader = True
    check("without the addon that knows it, a binary .pal will not load", no_reader)

    pos = ctl.history.position
    pal16 = ctl.reduce_colours(16)
    px = ctl.doc.cell(lay).plane("rgba")
    check("Reduce limits the layer to N colours, one undo step",
          ncolours(px) <= 16 and ctl.history.position == pos + 1 and pal16.size == 16)
    check("...and the palette it made is the bar's now", ctl.swatch_palette is pal16)
    ctl.undo()
    check("...undoable", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.set_swatch_palette(("builtin", "Game Boy (4)"))
    ctl.selection.select_rect((0, 0, 32, 48))
    ctl.apply_palette()
    px = ctl.doc.cell(lay).plane("rgba")
    check("Apply maps to the palette's colours, inside the selection",
          ncolours(px[:, :32]) <= 4 and np.array_equal(px[:, 40:], img[:, 40:]))
    ctl.undo()
    ctl.selection.select_all()
    report = ctl.convert_to_indexed(dither="none")
    check("Indexed makes an index-locked layer on the palette, bound to the document",
          report is not None and lay.index_locked and ctl.doc.palette is not None
          and ncolours(ctl.doc.cell(lay).plane("rgba")) <= 4)
    ctl.undo()
    check("...as one exact undo step", not lay.index_locked and ctl.doc.palette is None
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), img))
    ctl.redo()
    check("...and redo", lay.index_locked)
    ctl.undo()

    ctl.set_palette_lock(True)
    ctl.set_primary((200, 30, 30, 255))
    gb = {tuple(c) for c in PF.colours(ctl.swatch_palette).tolist()}
    check("Lock: a picked colour snaps to the palette", tuple(ctl.primary[:3]) in gb)
    paint = ctl.add_layer("paint")
    ctl.set_tool("brush", size=9, hardness=0.3)
    ctl.begin_stroke(10, 10)
    ctl.motion_stroke(30, 20)
    ctl.end_stroke(50, 30)
    pp = ctl.doc.cell(paint).plane("rgba")
    used = {tuple(c) for c in pp[pp[..., 3] > 0][:, :3].tolist()}
    check("Lock: a soft brush stroke keeps to the palette's colours (%d used)" % len(used),
          used and used <= gb)
    ctl.set_palette_lock(False)

    # ---- the addon reads its own palette format ----------------------------------------------
    ctl2 = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    for ident in ctl2.addons.addons:
        ctl2.addons.trust(ident)
    ctl2.load_addons()
    gp = ctl2.load_palette_file(raw)
    check("with its addon loaded, the binary .pal loads (6-bit values expanded)",
          gp is not None and int(gp.entries[1, 1]) == ((7 * 4 % 64) << 2 | (7 * 4 % 64) >> 4))
    ctl2.set_swatch_palette(("file", raw))
    ctl3 = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    ctl3.load_addons()
    ctl3.restore_swatch_palette()
    check("the chosen palette is remembered for next time", ctl3.swatch_key == ("file", raw),
          str(ctl3.swatch_key))

    # ---- the window: palette bar, scrolling toolbox ----------------------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    ctl.new_document(64, 48)
    win.canvas.bind()
    bar = win.palette_bar
    ctl.set_swatch_palette(("builtin", "PICO-8 (16)"))
    app.processEvents()
    check("the Palette bar shows the palette's swatches", len(bar.strip.colours) == 16)
    bar.strip.picked.emit(8, 1)
    bar.strip.picked.emit(12, 2)
    check("...left-click a swatch for the primary, right-click for the secondary",
          tuple(ctl.primary[:3]) == bar.strip.colours[8] and tuple(ctl.secondary[:3]) == bar.strip.colours[12])
    win.resize(900, 420)
    win.show()
    app.processEvents()
    check("the window can be made short (minimum %d px) -- panels and toolbox scroll"
          % win.minimumSizeHint().height(), win.minimumSizeHint().height() < 450)
    td = win.tools_dock
    check("the toolbox is a grid of 2 or 3 columns with every tool group in view",
          td.columns in (2, 3) and all(b.isVisible() for b in td._order))
    win.resize(900, 900)
    app.processEvents()
    check("...two columns when they fit the height", td.columns == 2, str(td.columns))
    win.resize(900, 420)
    app.processEvents()
    check("...three when two would not fit", td.columns == 3, str(td.columns))
    from PySide6.QtWidgets import QMenu
    tools_menu = [a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == "Tools"][0]
    check("the Tools menu is short: one submenu per tool group (%d entries)" % len(tools_menu.actions()),
          len(tools_menu.actions()) < 40 and any(a.menu() for a in tools_menu.actions()))
    check("...and every menu is capped in height, scrolling with arrows",
          all(m.maximumHeight() <= win._menu_cap for m in win.menuBar().findChildren(QMenu))
          and app.style().styleHint(app.style().StyleHint.SH_Menu_Scrollable) == 1)
    del app
    print("\nall palette-bar checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
