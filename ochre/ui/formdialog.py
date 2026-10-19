# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""A small dialog built from a list of fields, for the many Photoshop
commands that are "a few settings, then OK": Fill, Stroke, Color Range,
Focus Area, Trim, Apply Image, Calculations.

Fields are (key, label, kind, default, extra): kind is "int" (extra =
(min, max)), "choice" (extra = [(value, label), ...]), "bool", "color" or
"text". values() returns {key: value}.
"""

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLineEdit, QSpinBox)

from .layerdialogs import ColourButton


class FormDialog(QDialog):
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.widgets = {}
        form = QFormLayout(self)
        for key, label, kind, default, extra in fields:
            if kind == "int":
                w = QSpinBox()
                lo, hi = extra or (0, 100)
                w.setRange(lo, hi)
                w.setValue(int(default))
            elif kind == "choice":
                w = QComboBox()
                for value, text in extra:
                    w.addItem(text, value)
                w.setCurrentIndex(max(0, w.findData(default)))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            elif kind == "color":
                w = ColourButton(default)
            else:
                w = QLineEdit(str(default))
            self.widgets[key] = (kind, w)
            form.addRow(label, w)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
        out = {}
        for key, (kind, w) in self.widgets.items():
            if kind == "int":
                out[key] = w.value()
            elif kind == "choice":
                out[key] = w.currentData()
            elif kind == "bool":
                out[key] = w.isChecked()
            elif kind == "color":
                out[key] = tuple(w.value) + ((255,) if len(w.value) == 3 else ())
            else:
                out[key] = w.text()
        return out


def ask(parent, title, fields):
    """Show the form; the values, or None on Cancel."""
    dlg = FormDialog(title, fields, parent)
    return dlg.values() if dlg.exec() else None
