# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""What Primary / Secondary open: the full colour chooser ("All Colors"),
with a tab in front of it for every picker panel an addon registers (the
Palette addon's "Palette" tab). With none loaded it is just the chooser.

A picker panel's widget emits `colour_chosen(rgba)` and may answer
`wants_focus()` to have the dialog open on it.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QTabWidget, QVBoxLayout)


class ColourPickerDialog(QDialog):
    def __init__(self, ctl, current, title="Choose colour", parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle(title)
        self.colour = tuple(current) + ((255,) if len(current) == 3 else ())
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.pages = []
        focus = None
        for panel, widget in ctl.create_panels("picker", self):
            if hasattr(widget, "colour_chosen"):
                widget.colour_chosen.connect(self._picked)
            self.tabs.addTab(widget, getattr(panel, "label", panel.name))
            self.pages.append(widget)
            if focus is None and getattr(widget, "wants_focus", lambda: False)():
                focus = self.tabs.count() - 1

        # ---- All Colors: the full chooser, embedded
        self.chooser = QColorDialog(QColor(*self.colour))
        self.chooser.setOption(QColorDialog.ShowAlphaChannel, True)
        self.chooser.setOption(QColorDialog.NoButtons, True)
        self.chooser.setWindowFlags(Qt.Widget)
        self.chooser.currentColorChanged.connect(self._chooser_changed)
        self.tabs.addTab(self.chooser, "All Colors")
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

        bottom = QHBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(48, 24)
        bottom.addWidget(self.preview)
        self.hex = QLabel()
        bottom.addWidget(self.hex, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        outer.addLayout(bottom)

        self.tabs.setCurrentIndex(focus if focus is not None else self.tabs.indexOf(self.chooser))
        self._show()

    def _picked(self, rgba):
        r, g, b, a = (tuple(rgba) + (255,))[:4]
        self.colour = (r, g, b, a)
        self.chooser.blockSignals(True)
        self.chooser.setCurrentColor(QColor(r, g, b, a))
        self.chooser.blockSignals(False)
        self._show()

    def _chooser_changed(self, c):
        self.colour = (c.red(), c.green(), c.blue(), c.alpha())
        self._show()

    def _show(self):
        r, g, b, a = self.colour
        self.preview.setStyleSheet("background:#%02x%02x%02x; border:1px solid #444" % (r, g, b))
        self.hex.setText("#%02X%02X%02X  alpha %d" % (r, g, b, a))

    def result_colour(self):
        return self.colour
