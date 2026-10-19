# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Layer > Layer Filters: a layer's filter list -- a smart object's smart
filters, or a pixel layer's non-destructive filters -- shown in order, each
switchable, re-editable, movable and removable, previewed live. OK records
one undo step; Cancel puts everything back. Merge bakes a pixel layer's
filters into its pixels.
"""

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QVBoxLayout)

from .effectdialog import _Row
from .layerdialogs import _refresh


class FilterSettingsDialog(QDialog):
    """One filter's settings, as the effect dialog lays them out."""

    def __init__(self, effect, parent=None):
        super().__init__(parent)
        self.effect = effect
        self.setWindowTitle(effect.label)
        form = QFormLayout(self)
        self.rows = []
        for prop in effect.props:
            row = _Row(prop)
            row.changed.connect(self._changed)
            form.addRow(prop.ui.get("label", prop.name), row)
            self.rows.append(row)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
        self.on_change = None

    def _changed(self):
        for r in self.rows:
            r.refresh()
        if self.on_change:
            self.on_change(dict(self.effect.props.token()))


class LayerFiltersDialog(QDialog):
    def __init__(self, ctl, layer, effects, parent=None):
        super().__init__(parent)
        self.ctl, self.layer, self.effects = ctl, layer, effects
        self.canvas = getattr(parent, "canvas", None)
        self.original = json.loads(json.dumps((layer.params or {}).get("filters", [])))
        self.working = json.loads(json.dumps(self.original))
        kind = "Smart Filters" if getattr(layer, "kind", "raster") == "smart" else "Layer Filters"
        self.setWindowTitle("%s - %s" % (kind, layer.name))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Applied top to bottom:"))
        self.list = QListWidget()
        self.list.itemChanged.connect(self._toggled)
        self.list.itemDoubleClicked.connect(lambda _i: self.edit())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Edit...", self.edit), ("Up", lambda: self.move(-1)),
                         ("Down", lambda: self.move(+1)), ("Delete", self.delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.merge_button = QPushButton("Merge into Layer")
        self.merge_button.clicked.connect(self.merge)
        self.merge_button.setEnabled(getattr(layer, "kind", "raster") == "raster")
        row.addWidget(self.merge_button)
        lay.addLayout(row)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.merged = False
        self._rebuild(0)

    def _rebuild(self, select):
        self.list.blockSignals(True)
        self.list.clear()
        for f in self.working:
            item = QListWidgetItem(self.effects.label(f.get("name")))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if f.get("enabled", True) else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.working:
            self.list.setCurrentRow(max(0, min(select, len(self.working) - 1)))
        self._preview()

    def _preview(self):
        self.layer.params = dict(self.layer.params, filters=json.loads(json.dumps(self.working)))
        _refresh(self.ctl, self.canvas)

    def _toggled(self, item):
        i = self.list.row(item)
        if 0 <= i < len(self.working):
            self.working[i]["enabled"] = item.checkState() == Qt.Checked
            self._preview()

    def edit(self):
        i = self.list.currentRow()
        if not (0 <= i < len(self.working)):
            return
        entry = self.working[i]
        fx = self.effects.create(entry["name"], **dict(entry.get("params") or {}))
        if fx is None:
            return
        before = dict(entry.get("params") or {})
        dlg = FilterSettingsDialog(fx, self)

        def live(token):
            entry["params"] = token
            self._preview()
        dlg.on_change = live
        if dlg.exec():
            entry["params"] = dict(fx.props.token())
        else:
            entry["params"] = before
        self._preview()

    def move(self, d):
        i = self.list.currentRow()
        j = i + d
        if 0 <= i < len(self.working) and 0 <= j < len(self.working):
            self.working[i], self.working[j] = self.working[j], self.working[i]
            self._rebuild(j)

    def delete(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.working):
            del self.working[i]
            self._rebuild(i - 1)

    def _put_back(self):
        self.layer.params = dict(self.layer.params, filters=json.loads(json.dumps(self.original)))
        if not self.original:
            self.layer.params.pop("filters", None)

    def merge(self):
        """Bake the (working) list into the pixels, as one undo step."""
        work = json.loads(json.dumps(self.working))
        self._put_back()
        self.ctl.history.begin_group()
        if work != self.original:
            self.ctl.set_smart_filters(work, self.layer)
        self.ctl.merge_layer_filters(self.layer)
        self.ctl.history.end_group("Merge Filters")
        self.merged = True
        _refresh(self.ctl, self.canvas)
        super().accept()

    def accept(self):
        work = json.loads(json.dumps(self.working))
        self._put_back()
        if work != self.original:
            self.ctl.set_smart_filters(work, self.layer)
        _refresh(self.ctl, self.canvas)
        super().accept()

    def reject(self):
        self._put_back()
        _refresh(self.ctl, self.canvas)
        super().reject()
