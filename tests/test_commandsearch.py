#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Help > Search (Shift+Space), driven the way a person uses it.

Every check opens the real popup on a real main window, types into its
field and presses Enter (or clicks a row), then looks at what changed in
the editor: a layer added, a tool chosen, a panel brought forward, the
Preferences dialog on the right page, an addon's command run. The ranking
is checked through the rows the popup shows, not by calling the ranker.

Skips cleanly when PySide6 is absent.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_commandsearch.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.commandsearch import clean_label
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
ADDONS = os.path.join(REPO, "addons")


def main():
    app = QApplication.instance() or QApplication([])

    check("labels lose mnemonics, the shortcut column and the ellipsis",
          clean_label("&Open...\tCtrl+O") == "Open")
    check("...but a doubled && is a real ampersand",
          clean_label("Edge && Generic") == "Edge & Generic")

    state = tempfile.mkdtemp(prefix="ochre-search-")
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[ADDONS], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    win.show()
    app.processEvents()
    win.canvas.setFocus()
    app.processEvents()

    # ---- opening: the key, and the menu ----------------------------------
    QTest.keyClick(win.canvas, Qt.Key_Space, Qt.ShiftModifier)
    pop = getattr(win, "command_search", None)
    check("Shift+Space on the canvas opens the search popup",
          pop is not None and pop.isVisible())
    check("...with the typing field focused", pop.field.hasFocus())
    check("...and empty: nothing run yet, so no recents listed", pop.results.count() == 0)
    QTest.keyClick(pop.field, Qt.Key_Escape)
    check("Esc closes it", not pop.isVisible())
    check("...and gives the focus back to the canvas", win.canvas.hasFocus())

    help_menu = [a for a in win.menuBar().actions() if a.text() == "&Help"]
    check("there is a Help menu", len(help_menu) == 1)
    search_act = [a for a in help_menu[0].menu().actions() if "Search" in a.text()]
    check("...with Search in it, showing Shift+Space",
          len(search_act) == 1 and search_act[0].shortcut().toString() == "Shift+Space")
    search_act[0].trigger()
    check("Help > Search opens the same popup", pop.isVisible())

    def rows():
        return [pop.results.item(i).text() for i in range(pop.results.count())]

    def type_query(text):
        pop.field.clear()
        QTest.keyClicks(pop.field, text)
        return rows()

    def open_and_type(text):
        # As after any dialog closes, the main window is the active one again.
        win.activateWindow()
        QTest.qWaitForWindowActive(win, 2000)
        win.canvas.setFocus()
        QTest.keyClick(win.canvas, Qt.Key_Space, Qt.ShiftModifier)
        if not pop.isVisible():
            check("Shift+Space opens the popup (for %r)" % text, False)
        return type_query(text)

    # ---- a menu command, run with Enter ----------------------------------
    found = type_query("canvas size")
    check("typing finds a menu command and shows where it lives",
          found and found[0] == "Image > Canvas Size", str(found[:3]))
    found = type_query("add layer")
    check("'add layer' puts Layers > Add Layer first",
          found and found[0] == "Layers > Add Layer", str(found[:3]))
    before = len(ctl.doc.layers())
    QTest.keyClick(pop.field, Qt.Key_Return)
    check("Enter runs it: the document has one more layer",
          len(ctl.doc.layers()) == before + 1)
    check("...and the popup closed", not pop.isVisible())

    # ---- ranking, as shown -----------------------------------------------
    found = open_and_type("layers")
    check("an exact label ranks first (Window > Layers for 'layers')",
          found and found[0] == "Window > Layers", str(found[:3]))
    found = type_query("gaus")
    check("a prefix finds Gaussian Blur", found and found[0].endswith("> Gaussian Blur"),
          str(found[:3]))
    found = type_query("gblur")
    check("a fuzzy query finds it too", any(r.endswith("> Gaussian Blur") for r in found),
          str(found[:3]))
    found = type_query("blur")
    tiers = [r.rsplit(" > ", 1)[1] for r in found]
    first_inner = next(i for i, t in enumerate(tiers) if not t.lower().startswith("blur"))
    check("labels starting with the word come before ones that merely contain it",
          all(not t.lower().startswith("blur") for t in tiers[first_inner:]), str(tiers[:8]))
    found = type_query("colour balance")
    check("British spelling finds the American label",
          found and found[0].endswith("> Color Balance"), str(found[:3]))
    found = type_query("zzqqxx")
    check("no match shows 'No matches' and an empty list",
          not found and pop.empty.isVisible())

    # ---- the arrows choose, Enter runs the chosen row --------------------
    found = type_query("flip")
    want = "Layers > Flip Vertical"
    check("'flip' lists both Image and Layers flips", want in found, str(found))
    for _ in range(found.index(want)):
        QTest.keyClick(pop.field, Qt.Key_Down)
    check("Down moves the highlight", rows()[pop.results.currentRow()] == want)
    from ochre.engine.geometry import Rect
    lay = ctl.active_layer
    cell = ctl.doc.cell(lay)
    cell.fill(Rect(0, 0, ctl.doc.width, 1), (255, 0, 0, 255))
    top_before = cell.plane("rgba")[0, 0].tolist()
    QTest.keyClick(pop.field, Qt.Key_Return)
    check("Enter runs the highlighted row: the layer flipped vertically",
          cell.plane("rgba")[0, 0].tolist() != top_before
          and cell.plane("rgba")[-1, 0].tolist() == top_before)

    found = open_and_type("flip")
    check("RECENT USE: the flip just run now leads the others of its tier",
          want in found and found.index(want) < found.index("Image > Flip Horizontal"),
          str(found[:4]))
    check("...but not past a better tier (the Flip tool is an exact label)",
          found[0] == "Tools > Transform > Flip", str(found[:4]))
    found = type_query("")
    check("an empty query lists recent picks, newest first",
          found[:2] == [want, "Layers > Add Layer"], str(found))
    with open(os.path.join(state, "search_recent.txt"), encoding="utf-8") as fh:
        saved = fh.read().splitlines()
    check("...remembered in the session's state dir (never the real one)",
          saved[:2] == ["menu:Layers/Flip Vertical", "menu:Layers/Add Layer"], str(saved))

    # ---- a click runs a row too ------------------------------------------
    found = type_query("duplicate layer")
    check("'duplicate layer' finds it", found and found[0] == "Layers > Duplicate Layer")
    before = len(ctl.doc.layers())
    item = pop.results.item(0)
    QTest.mouseClick(pop.results.viewport(), Qt.LeftButton, Qt.NoModifier,
                     pop.results.visualItemRect(item).center())
    check("clicking a row runs it", len(ctl.doc.layers()) == before + 1 and not pop.isVisible())

    # ---- tools -----------------------------------------------------------
    found = open_and_type("polygonal lasso")
    check("tools are found, under their toolbox group",
          found and found[0] == "Tools > Lasso > Polygonal Lasso", str(found[:3]))
    check("...showing the tool's key", pop.results.item(0).data(Qt.UserRole + 2) == "L")
    QTest.keyClick(pop.field, Qt.Key_Return)
    check("Enter on a tool selects it", ctl.tool_name == "polygon_lasso"
          or ctl.tools.label(ctl.tool_name) == "Polygonal Lasso", ctl.tool_name)

    # ---- panels ----------------------------------------------------------
    def shown(dock):
        # A dock behind another tab still says isVisible(); what is on
        # screen is its visible region.
        return dock.isVisible() and not dock.visibleRegion().isEmpty()

    win.history_dock.raise_()
    app.processEvents()
    check("the Histogram panel starts behind the History tab",
          not shown(win.histogram_dock))
    found = open_and_type("histogram")
    check("panels are found under Window", found and found[0] == "Window > Histogram",
          str(found[:3]))
    QTest.keyClick(pop.field, Qt.Key_Return)
    app.processEvents()
    check("Enter on a panel brings it to the front", shown(win.histogram_dock))
    win.navigator_dock.toggleViewAction().trigger()
    app.processEvents()
    check("a closed panel...", not shown(win.navigator_dock))
    open_and_type("navigator")
    QTest.keyClick(pop.field, Qt.Key_Return)
    app.processEvents()
    check("...is opened again, in front", shown(win.navigator_dock))

    # ---- Preferences pages -----------------------------------------------
    found = open_and_type("checker size")
    check("a setting's name finds its Preferences page",
          found and found[0] == "Edit > Preferences > Display", str(found[:3]))
    seen = {}

    def close_prefs():
        dlg = QApplication.activeModalWidget()
        seen["title"] = dlg.windowTitle() if dlg else None
        seen["page"] = dlg.tabs.tabText(dlg.tabs.currentIndex()) if dlg else None
        if dlg:
            dlg.reject()
    QTimer.singleShot(0, close_prefs)
    QTest.keyClick(pop.field, Qt.Key_Return)
    check("Enter opens Preferences on that page",
          seen == {"title": "Preferences", "page": "Display"}, str(seen))

    # ---- disabled commands -----------------------------------------------
    found = open_and_type("edit mask")
    check("a command that cannot run now is still listed",
          found and found[0] == "Layers > Layer Mask > Edit Mask (paint on it)", str(found[:3]))
    check("...greyed out", pop.results.item(0).data(Qt.UserRole + 1) is False)
    QTest.keyClick(pop.field, Qt.Key_Return)
    check("...and Enter does not run it (no mask to edit), the popup stays",
          not ctl.editing_mask and pop.isVisible(), "%r %r" % (ctl.editing_mask, pop.isVisible()))
    check("...saying why in the status bar",
          "not available" in win.statusBar().currentMessage())
    ctl.add_layer_mask("reveal")
    found = open_and_type("edit mask")
    check("with a mask, the same command is runnable",
          pop.results.item(0).data(Qt.UserRole + 1) is True)
    QTest.keyClick(pop.field, Qt.Key_Escape)

    # ---- addon commands come and go with their addon ---------------------
    found = open_and_type("describe document")
    check("an addon's command is not found while the addon is not loaded",
          not any("Describe Document" in r for r in found), str(found))
    QTest.keyClick(pop.field, Qt.Key_Escape)
    ctl.addons.trust("example")
    ctl.load_addons(only=["example"])
    check("the example addon loads", ctl.addons.addons["example"].state == "loaded")
    found = open_and_type("describe document")
    check("once loaded, its command is found under Addons > Commands",
          found and found[0] == "Addons > Commands > Describe Document", str(found[:3]))
    found = type_query("example")
    check("...and by its addon's name",
          "Addons > Commands > Describe Document" in found, str(found[:5]))
    type_query("describe document")
    said = {}

    def close_box():
        box = QApplication.activeModalWidget()
        said["text"] = box.text() if box is not None and hasattr(box, "text") else None
        if box:
            box.accept()
    QTimer.singleShot(0, close_box)
    QTest.keyClick(pop.field, Qt.Key_Return)
    doc = ctl.doc
    check("Enter runs the addon's command against the live document",
          said.get("text") and said["text"].startswith("%dx%d" % (doc.width, doc.height)),
          str(said))
    check("the Addons > Commands menu lists it as well",
          any("Describe Document" in a.text() for a in win.addon_commands_menu.actions()))
    ctl.load_addons(only=[])
    check("the addon unloads", ctl.addons.addons["example"].state != "loaded")
    found = open_and_type("describe document")
    check("once unloaded, its command is gone from the search",
          not any("Describe Document" in r for r in found), str(found))
    check("...and from the menu", not win.addon_commands_menu.actions())
    found = type_query("")
    check("...and from the recents, which list only what exists now",
          "Addons > Commands > Describe Document" not in found, str(found))
    QTest.keyClick(pop.field, Qt.Key_Escape)

    win.shutdown()
    shutil.rmtree(state, ignore_errors=True)
    print("\nall command search checks passed")


if __name__ == "__main__":
    main()
