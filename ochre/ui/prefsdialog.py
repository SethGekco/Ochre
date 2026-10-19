# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Edit > Preferences and Edit > Keyboard Shortcuts.

Preferences are generated from the settings schema (engine/settings.py
DEFAULTS), so a new setting appears here with no UI code; what you change
is written to <state dir>/preferences.ini, over the data INIs. Keyboard
Shortcuts edit tool keys and menu keys, written to shortcuts.ini (ui/keymap.py).
"""

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                               QKeySequenceEdit, QLabel, QLineEdit, QScrollArea, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ..engine.settings import DEFAULTS, Settings
from . import keymap
from .layerdialogs import ColourButton


class PreferencesDialog(QDialog):
    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle("Preferences")
        self.widgets = {}
        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        for section, keys in DEFAULTS.items():
            page = QWidget()
            form = QFormLayout(page)
            for key, default in keys.items():
                value = ctl.settings.get(section, key)
                if isinstance(default, bool):
                    w = QCheckBox()
                    w.setChecked(bool(value))
                elif isinstance(default, int):
                    w = QSpinBox()
                    w.setRange(-2 ** 31, 2 ** 31 - 1)
                    w.setValue(int(value))
                elif isinstance(default, tuple):
                    w = ColourButton(value)
                else:
                    w = QLineEdit(str(value))
                self.widgets[(section, key)] = (default, w)
                form.addRow(key, w)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            tabs.addTab(scroll, section)
        outer.addWidget(tabs)
        outer.addWidget(QLabel("Changes apply to new documents; some need a restart."))
        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    def values(self):
        out = {}
        for k, (default, w) in self.widgets.items():
            if isinstance(default, bool):
                out[k] = w.isChecked()
            elif isinstance(default, int):
                out[k] = w.value()
            elif isinstance(default, tuple):
                v = tuple(w.value)
                out[k] = v + (255,) * (len(default) - len(v))
            else:
                out[k] = w.text()
        return out

    def accept(self):
        """Write only what differs from the shipped configuration."""
        import os
        base = Settings(defaults=DEFAULTS, db=self.ctl.db, env={})
        changed = {k: v for k, v in self.values().items() if base.get(*k) != v}
        if self.ctl.state_dir:
            path = os.path.join(self.ctl.state_dir, "preferences.ini")
            Settings.save_user(path, changed)
            self.ctl.settings.apply_user(path)
        super().accept()


def menu_actions(menubar):
    """[(path, QAction)] for every menu command, e.g. 'File/Save As...'."""
    out = []

    def walk(menu, prefix):
        for act in menu.actions():
            if act.isSeparator():
                continue
            text = act.text().replace("&", "").split("\t")[0]
            sub = act.menu()
            if sub is not None:
                walk(sub, prefix + text + "/")
            elif text:
                out.append((prefix + text, act))
    for act in menubar.actions():
        if act.menu() is not None:
            walk(act.menu(), act.text().replace("&", "") + "/")
    return out


def apply_menu_keys(menubar, data):
    keys = data.get("Menu") or {}
    n = 0
    for path, act in menu_actions(menubar):
        if path in keys:
            act.setShortcut(QKeySequence(keys[path]))
            n += 1
    return n


class ShortcutsDialog(QDialog):
    def __init__(self, ctl, menubar, parent=None):
        super().__init__(parent)
        self.ctl, self.menubar = ctl, menubar
        self.setWindowTitle("Keyboard Shortcuts")
        self.data = keymap.load(ctl.state_dir)
        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        self.tool_edits, self.menu_edits = {}, {}
        page = QWidget()
        form = QFormLayout(page)
        for name in ctl.tools.names():
            e = QKeySequenceEdit(QKeySequence(ctl.tools.shortcut(name) or ""))
            self.tool_edits[name] = e
            form.addRow(ctl.tools.label(name), e)
        tabs.addTab(self._scroll(page), "Tools")
        page = QWidget()
        form = QFormLayout(page)
        for path, act in menu_actions(menubar):
            e = QKeySequenceEdit(act.shortcut())
            self.menu_edits[path] = (e, act)
            form.addRow(path, e)
        tabs.addTab(self._scroll(page), "Menus")
        outer.addWidget(tabs)
        outer.addWidget(QLabel("Menu keys apply now; tool keys after a restart."))
        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

    @staticmethod
    def _scroll(w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def accept(self):
        tools = {}
        for name, e in self.tool_edits.items():
            key = e.keySequence().toString()
            if key != (self.ctl.tools.shortcut(name) or ""):
                tools[name] = key
        menus = dict(self.data.get("Menu") or {})
        for path, (e, act) in self.menu_edits.items():
            key = e.keySequence().toString()
            if key != act.shortcut().toString():
                menus[path] = key
                act.setShortcut(QKeySequence(key))
        merged_tools = dict(self.data.get("Tools") or {})
        merged_tools.update(tools)
        keymap.save(self.ctl.state_dir, {"Tools": merged_tools, "Menu": menus})
        super().accept()



