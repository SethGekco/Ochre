#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Preferences and keyboard shortcuts: saved in the state dir, applied over
the shipped configuration, and found again next session.

Run: python3 tests/test_prefs.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ochre.engine.settings import Settings
from ochre.ui import keymap
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    state = tempfile.mkdtemp()
    path = os.path.join(state, "preferences.ini")
    Settings.save_user(path, {("Canvas", "DefaultWidth"): 1234, ("History", "MaxEntries"): 77,
                              ("Display", "CheckerLight"): (10, 20, 30, 255),
                              ("Nonsense", "Key"): 5})
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state)
    check("preferences in the state dir override the data INIs",
          ctl.settings.get("Canvas", "DefaultWidth") == 1234
          and ctl.settings.get("History", "MaxEntries") == 77
          and ctl.settings.get("Display", "CheckerLight") == (10, 20, 30, 255))
    check("...unknown keys are ignored", ctl.settings.get("Nonsense", "Key") is None)
    env = Settings(DATA, env={"OCHRE_DEFAULTWIDTH": "999"})
    env.apply_user(path)
    check("...and the environment still wins", env.get("Canvas", "DefaultWidth") == 999)

    keymap.save(state, {"Tools": {"brush": "Ctrl+B", "pencil": ""},
                        "Menu": {"File/Save As...": "Ctrl+Alt+S"}})
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state)
    check("tool key overrides apply at start (and an empty one removes a key)",
          ctl.tools.shortcut("brush") == "Ctrl+B" and ctl.tools.shortcut("pencil") is None)

    from PySide6.QtWidgets import QApplication
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.mainwindow import MainWindow
    from ochre.ui.prefsdialog import PreferencesDialog, ShortcutsDialog, menu_actions
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    acts = dict(menu_actions(win.menuBar()))
    check("menu key overrides apply to the window's actions",
          acts["File/Save As..."].shortcut().toString() == "Ctrl+Alt+S")
    check("...every menu command is reachable by path",
          "Edit/Preferences..." in acts and "Effects/Filter Gallery..." in acts)
    dlg = PreferencesDialog(ctl, win)
    check("the Preferences dialog shows every schema section", len(dlg.widgets) > 20)
    _d, w = dlg.widgets[("Canvas", "DefaultHeight")]
    w.setValue(777)
    dlg.accept()
    ctl2 = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state)
    check("...Save writes what changed, kept for next time",
          ctl2.settings.get("Canvas", "DefaultHeight") == 777
          and ctl2.settings.get("Canvas", "DefaultWidth") == 1234)
    sd = ShortcutsDialog(ctl, win.menuBar(), win)
    from PySide6.QtGui import QKeySequence
    sd.menu_edits["Image/Trim..."][0].setKeySequence(QKeySequence("Ctrl+Alt+T"))
    sd.accept()
    check("the Shortcuts dialog applies menu keys at once and saves them",
          acts["Image/Trim..."].shortcut().toString() == "Ctrl+Alt+T"
          and keymap.load(state)["Menu"]["Image/Trim..."] == "Ctrl+Alt+T"
          and keymap.load(state)["Menu"]["File/Save As..."] == "Ctrl+Alt+S")
    del app
    print("\nall preference checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
