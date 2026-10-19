# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Filter Gallery: browse filters by thumbnail, stack several,
tune each, see the result at 100% -- then apply the whole stack as ONE step
(one undo, one action, and on a smart object one smart filter).

The stack is the engine's FilterStack effect; this dialog only edits its
JSON list, so nothing here is needed to replay a recorded gallery.
"""

import json

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .effectdialog import _Row

CATEGORIES = ("Artistic", "Stylize", "Distort", "Noise", "Blurs", "Edge & Generic")
PREVIEW = (420, 320)
THUMB = 64


def _pixmap(rgba, size=None):
    """RGBA uint8 -> QPixmap over a checkerboard, so transparency reads."""
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    pix = QPixmap(w, h)
    pix.fill(Qt.white)
    p = QPainter(pix)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if (x // 8 + y // 8) % 2:
                p.fillRect(x, y, 8, 8, Qt.lightGray)
    p.drawImage(0, 0, img)
    p.end()
    if size is not None:
        pix = pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class FilterGalleryDialog(QDialog):
    def __init__(self, ctl, effects, parent=None, source=None):
        super().__init__(parent)
        self.ctl = ctl
        self.effects = effects
        self.setWindowTitle("Filter Gallery")
        self.stack = []                     # [{"name", "params", "enabled"}]
        self._current = -1
        self._rows = []
        self._effect = None

        src = source if source is not None else self._target_pixels()
        self.full = src
        self.sample = self._crop(src)
        self.thumb_src = self._thumb_source(src)

        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        self.preview = QLabel()
        self.preview.setMinimumSize(*PREVIEW)
        self.preview.setAlignment(Qt.AlignCenter)
        body.addWidget(self.preview, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(THUMB, THUMB))
        self.tree.setMinimumWidth(220)
        groups = effects.grouped()
        for cat in CATEGORIES:
            names = [n for n in groups.get(cat, []) if n != "filter_stack"]
            if not names:
                continue
            top = QTreeWidgetItem([cat])
            self.tree.addTopLevelItem(top)
            for n in names:
                child = QTreeWidgetItem([effects.label(n)])
                child.setData(0, Qt.UserRole, n)
                top.addChild(child)
        self.tree.itemExpanded.connect(self._thumbs_for)
        self.tree.itemClicked.connect(self._picked)
        body.addWidget(self.tree)

        right = QVBoxLayout()
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        right.addWidget(self.form_host, 1)
        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._toggled)
        right.addWidget(QLabel("Applied in order, bottom first:"))
        right.addWidget(self.list)
        buttons = QHBoxLayout()
        for text, fn in (("New", self._new), ("Delete", self._delete),
                         ("Up", lambda: self._move(+1)), ("Down", lambda: self._move(-1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        right.addLayout(buttons)
        holder = QWidget()
        holder.setLayout(right)
        holder.setMinimumWidth(280)
        body.addWidget(holder)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        outer.addWidget(box)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh_preview)

        first = next((n for c in CATEGORIES for n in groups.get(c, []) if n != "filter_stack"), None)
        if first is not None:
            self.stack.append(self._entry(first))
            self._rebuild_list(0)
        self.refresh_preview()

    # ---- the pixels ---------------------------------------------------------------

    def _target_pixels(self):
        ctl = self.ctl
        target = getattr(ctl, "paint_target", ctl.active_layer)
        if target is None or target.is_group:
            return np.zeros((8, 8, 4), np.uint8)
        if getattr(target, "kind", "raster") == "smart":
            return np.array(ctl.compositor._smart_image(target))
        return np.array(ctl.doc.cell(target).plane("rgba"))

    def _crop(self, src):
        """A 100% view: filters with pixel radii look as they will, not as
        they would on a shrunken copy. Centred on the selection, if any."""
        h, w = src.shape[:2]
        pw, ph = min(w, PREVIEW[0]), min(h, PREVIEW[1])
        cx, cy = w // 2, h // 2
        sel = getattr(self.ctl, "selection", None)
        box = getattr(sel, "bbox", None)
        box = box() if callable(box) else box
        if box is not None:
            cx, cy = box.x + box.w // 2, box.y + box.h // 2
        x0 = max(0, min(w - pw, cx - pw // 2))
        y0 = max(0, min(h - ph, cy - ph // 2))
        return np.ascontiguousarray(src[y0:y0 + ph, x0:x0 + pw])

    @staticmethod
    def _thumb_source(src):
        h, w = src.shape[:2]
        step = max(1, int(max(h, w) / 96))
        return np.ascontiguousarray(src[::step, ::step])

    # ---- the stack -----------------------------------------------------------------

    def _entry(self, name):
        fx = self.effects.create(name)
        return {"name": name, "params": dict(fx.props.token()) if fx else {}, "enabled": True}

    def stack_json(self):
        return json.dumps(self.stack)

    def _rebuild_list(self, select):
        self.list.blockSignals(True)
        self.list.clear()
        for e in self.stack:
            item = QListWidgetItem(self.effects.label(e["name"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if e.get("enabled", True) else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.stack:
            self.list.setCurrentRow(max(0, min(select, len(self.stack) - 1)))
            self._select(self.list.currentRow())
        else:
            self._select(-1)

    def _select(self, row):
        self._current = row
        while self.form.rowCount():
            self.form.removeRow(0)
        self._rows = []
        self._effect = None
        if not (0 <= row < len(self.stack)):
            return
        entry = self.stack[row]
        fx = self.effects.create(entry["name"], **entry["params"])
        if fx is None:
            return
        self._effect = fx
        for prop in fx.props:
            r = _Row(prop)
            r.changed.connect(self._param_changed)
            self.form.addRow(prop.ui.get("label", prop.name), r)
            self._rows.append(r)

    def _param_changed(self):
        if self._effect is None or not (0 <= self._current < len(self.stack)):
            return
        for r in self._rows:
            r.refresh()
        self.stack[self._current]["params"] = dict(self._effect.props.token())
        self._timer.start()

    def _picked(self, item, _col=0):
        name = item.data(0, Qt.UserRole)
        if not name:
            return
        if 0 <= self._current < len(self.stack):
            self.stack[self._current] = self._entry(name)      # Photoshop: replace the selected
        else:
            self.stack.append(self._entry(name))
            self._current = len(self.stack) - 1
        self._rebuild_list(self._current)
        self._timer.start()

    def _toggled(self, item):
        row = self.list.row(item)
        if 0 <= row < len(self.stack):
            self.stack[row]["enabled"] = item.checkState() == Qt.Checked
            self._timer.start()

    def _new(self):
        base = self.stack[self._current] if 0 <= self._current < len(self.stack) else None
        entry = json.loads(json.dumps(base)) if base else None
        if entry is None:
            return
        self.stack.insert(self._current + 1, entry)
        self._rebuild_list(self._current + 1)
        self._timer.start()

    def _delete(self):
        if 0 <= self._current < len(self.stack):
            del self.stack[self._current]
            self._rebuild_list(self._current - 1)
            self._timer.start()

    def _move(self, d):
        i = self._current
        j = i + d
        if 0 <= i < len(self.stack) and 0 <= j < len(self.stack):
            self.stack[i], self.stack[j] = self.stack[j], self.stack[i]
            self._rebuild_list(j)
            self._timer.start()

    # ---- rendering -----------------------------------------------------------------

    def render(self, src):
        fx = self.effects.create("filter_stack", stack=self.stack_json())
        fx.registry = self.effects
        return fx.apply(src)

    def refresh_preview(self):
        out = self.render(self.sample)
        self.last_preview = out
        self.preview.setPixmap(_pixmap(out))

    def _thumbs_for(self, top):
        for i in range(top.childCount()):
            child = top.child(i)
            if child.icon(0).isNull():
                fx = self.effects.create(child.data(0, Qt.UserRole))
                try:
                    out = fx.apply(self.thumb_src)
                except Exception:                           # noqa: BLE001
                    continue
                child.setIcon(0, QIcon(_pixmap(out, QSize(THUMB, THUMB))))
