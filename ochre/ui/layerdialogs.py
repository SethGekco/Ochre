# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Dialogs for adjustment layers, fill layers and layer styles.

Each previews LIVE by changing the layer itself -- the compositor redraws
from it -- and on OK records one undo step through the controller (the live
changes are put back first, so history sees one clean before/after). Cancel
restores exactly what was there.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..engine import layerstyles
from .effectdialog import _Row


def _refresh(ctl, canvas=None):
    ctl.compositor.invalidate()
    ctl.invalidate()
    if canvas is not None:
        canvas.refresh()
        canvas.viewport().update()


class ColourButton(QPushButton):
    """A swatch that opens a colour dialog. value is (r, g, b) or (r, g, b, a)."""

    def __init__(self, value, on_change=None, alpha=False, parent=None):
        super().__init__(parent)
        self.value = tuple(value)
        self.alpha = alpha
        self.on_change = on_change
        self.clicked.connect(self._pick)
        self._paint()

    def _paint(self):
        r, g, b = self.value[:3]
        self.setText("#%02X%02X%02X" % (r, g, b))
        fg = "#000" if (r * 0.3 + g * 0.59 + b * 0.11) > 128 else "#fff"
        self.setStyleSheet("background:#%02x%02x%02x; color:%s" % (r, g, b, fg))

    def _pick(self):
        start = QColor(*self.value[:3], self.value[3] if len(self.value) > 3 else 255)
        opts = QColorDialog.ShowAlphaChannel if self.alpha else QColorDialog.ColorDialogOption(0)
        colour = QColorDialog.getColor(start, self, "Colour", opts)
        if colour.isValid():
            self.set_value((colour.red(), colour.green(), colour.blue(), colour.alpha())
                           if self.alpha or len(self.value) > 3 else
                           (colour.red(), colour.green(), colour.blue()))

    def set_value(self, value):
        self.value = tuple(value)
        self._paint()
        if self.on_change:
            self.on_change(self.value)


class AdjustmentLayerDialog(QDialog):
    """The settings of an adjustment layer, previewed on the canvas."""

    def __init__(self, ctl, layer, effects, parent=None):
        super().__init__(parent)
        self.ctl, self.layer = ctl, layer
        self.canvas = getattr(parent, "canvas", None)
        self.original = dict(layer.params)
        self.effect = effects.create(layer.params.get("effect"))
        self.setWindowTitle("%s (adjustment layer)" % (self.effect.label if self.effect else "?"))
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.rows = []
        if self.effect is not None:
            for prop in self.effect.props:
                if prop.name in layer.params:
                    self.effect.props.set(prop.name, layer.params[prop.name])
            for prop in self.effect.props:
                row = _Row(prop)
                row.changed.connect(self._changed)
                form.addRow(prop.ui.get("label", prop.name), row)
                self.rows.append(row)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self):
        return dict(self.effect.props.token()) if self.effect is not None else {}

    def _changed(self):
        for row in self.rows:
            row.refresh()
        self.layer.params = dict(self.original, **self.values())
        _refresh(self.ctl, self.canvas)

    def accept(self):
        new = self.values()
        self.layer.params = dict(self.original)
        self.ctl.set_layer_params(self.layer, new)
        _refresh(self.ctl, self.canvas)
        super().accept()

    def reject(self):
        self.layer.params = dict(self.original)
        _refresh(self.ctl, self.canvas)
        super().reject()


class FillLayerDialog(QDialog):
    """Solid colour or gradient for a fill layer."""

    def __init__(self, ctl, layer, parent=None):
        super().__init__(parent)
        self.ctl, self.layer = ctl, layer
        self.canvas = getattr(parent, "canvas", None)
        self.original = dict(layer.params)
        self.setWindowTitle("Fill Layer")
        p = layer.params
        form = QFormLayout(self)
        self.kind = QComboBox()
        self.kind.addItem("Solid colour", "solid")
        self.kind.addItem("Gradient", "gradient")
        self.kind.setCurrentIndex(0 if p.get("type", "solid") == "solid" else 1)
        form.addRow("Type", self.kind)
        self.color = ColourButton(p.get("color", ctl.primary), self._changed, alpha=True)
        self.start = ColourButton(p.get("start", ctl.primary), self._changed, alpha=True)
        self.end = ColourButton(p.get("end", ctl.secondary), self._changed, alpha=True)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-360, 360)
        self.angle.setValue(float(p.get("angle", 90)))
        self.angle.setSuffix("°")
        form.addRow("Colour", self.color)
        form.addRow("Start", self.start)
        form.addRow("End", self.end)
        form.addRow("Angle", self.angle)
        self.kind.currentIndexChanged.connect(lambda _i: self._changed())
        self.angle.valueChanged.connect(lambda _v: self._changed())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        if self.kind.currentData() == "solid":
            return {"type": "solid", "color": self.color.value}
        return {"type": "gradient", "start": self.start.value, "end": self.end.value,
                "angle": self.angle.value()}

    def _changed(self, *_):
        self.layer.params = dict(self.original, **self.values())
        _refresh(self.ctl, self.canvas)

    def accept(self):
        new = self.values()
        self.layer.params = dict(self.original)
        self.ctl.set_layer_params(self.layer, new)
        _refresh(self.ctl, self.canvas)
        super().accept()

    def reject(self):
        self.layer.params = dict(self.original)
        _refresh(self.ctl, self.canvas)
        super().reject()


class LayerStyleDialog(QDialog):
    """Photoshop's Layer Style dialog: tick an effect, then tune it."""

    def __init__(self, ctl, layer, parent=None):
        super().__init__(parent)
        self.ctl, self.layer = ctl, layer
        self.canvas = getattr(parent, "canvas", None)
        self.original = {k: dict(v) for k, v in layer.styles.items()}
        self.working = {k: dict(v) for k, v in layer.styles.items()}
        self.setWindowTitle("Layer Style - %s" % layer.name)
        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        self.list = QListWidget()
        self.pages = QStackedWidget()
        self.checks = {}
        for name, (label, _defaults) in layerstyles.STYLES.items():
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = name in self.working and self.working[name].get("enabled", True)
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            item.setData(Qt.UserRole, name)
            self.list.addItem(item)
            self.pages.addWidget(self._page(name))
        self.list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.list.itemChanged.connect(self._toggled)
        self.list.setCurrentRow(0)
        body.addWidget(self.list)
        body.addWidget(self.pages, 1)
        outer.addLayout(body)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _page(self, name):
        page = QWidget()
        form = QFormLayout(page)
        current = layerstyles.params(self.working, name)
        for key, default in layerstyles.STYLES[name][1].items():
            value = current[key]
            if isinstance(default, tuple):
                w = ColourButton(value, lambda v, n=name, k=key: self._set(n, k, v))
            elif key == "position":
                w = QComboBox()
                for opt in ("outside", "inside", "center"):
                    w.addItem(opt.capitalize(), opt)
                w.setCurrentIndex(max(0, w.findData(value)))
                w.currentIndexChanged.connect(
                    lambda _i, n=name, k=key, ww=w: self._set(n, k, ww.currentData()))
            else:
                w = QSpinBox()
                hi = 100 if "opacity" in key or key == "depth" else (360 if key == "angle" else 250)
                lo = -360 if key == "angle" else 0
                w.setRange(lo, max(hi, int(value)))
                w.setValue(int(value))
                w.valueChanged.connect(lambda v, n=name, k=key: self._set(n, k, int(v)))
            form.addRow(key.replace("_", " ").capitalize(), w)
        return page

    def _set(self, name, key, value):
        self.working.setdefault(name, {})[key] = value
        # Changing a setting switches the effect on, as in Photoshop.
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == name and item.checkState() != Qt.Checked:
                item.setCheckState(Qt.Checked)
        self.working[name]["enabled"] = True
        self._preview()

    def _toggled(self, item):
        name = item.data(Qt.UserRole)
        self.working.setdefault(name, {})["enabled"] = item.checkState() == Qt.Checked
        self._preview()

    def result_styles(self):
        return {k: v for k, v in self.working.items() if v.get("enabled", True)}

    def _preview(self):
        self.layer.styles = {k: dict(v) for k, v in self.result_styles().items()}
        _refresh(self.ctl, self.canvas)

    def accept(self):
        new = self.result_styles()
        self.layer.styles = {k: dict(v) for k, v in self.original.items()}
        self.ctl.set_layer_styles(self.layer, new)
        _refresh(self.ctl, self.canvas)
        super().accept()

    def reject(self):
        self.layer.styles = {k: dict(v) for k, v in self.original.items()}
        _refresh(self.ctl, self.canvas)
        super().reject()
