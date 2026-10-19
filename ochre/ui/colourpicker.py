# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Choosing colours with palettes in mind.

  ColourPickerDialog   what Primary / Secondary open: a "Palette" tab (the
                       Palette bar's palette, or any other, as a grid of
                       swatches -- the default whenever a palette is chosen)
                       and an "All Colors" tab (the full colour chooser).
  BuildPaletteDialog   a palette from the colours the image already uses.
                       When there are more than the palette may hold, the
                       closest colours are merged, and the dialog lists
                       exactly which merges it recommends before anything is
                       built; optionally the image is remapped to match.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)


class SwatchGrid(QWidget):
    """A palette as a grid of clickable swatches, 16 to a row."""

    chosen = Signal(int)
    COLS = 16
    CELL = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colours = []
        self.current = None
        self.setMouseTracking(True)

    def set_colours(self, colours):
        self.colours = [tuple(int(v) for v in c[:3]) for c in colours]
        self.current = None
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        rows = max(1, (len(self.colours) + self.COLS - 1) // self.COLS)
        return QSize(self.COLS * self.CELL + 1, rows * self.CELL + 1)

    def _index(self, pos):
        c, r = int(pos.x()) // self.CELL, int(pos.y()) // self.CELL
        i = r * self.COLS + c
        return i if 0 <= c < self.COLS and 0 <= i < len(self.colours) else None

    def paintEvent(self, _e):
        p = QPainter(self)
        for i, (r, g, b) in enumerate(self.colours):
            y, x = divmod(i, self.COLS)
            p.fillRect(x * self.CELL + 1, y * self.CELL + 1, self.CELL - 2, self.CELL - 2, QColor(r, g, b))
            if i == self.current:
                p.setPen(QColor(255, 255, 255))
                p.drawRect(x * self.CELL, y * self.CELL, self.CELL - 1, self.CELL - 1)
                p.setPen(QColor(0, 0, 0))
                p.drawRect(x * self.CELL + 1, y * self.CELL + 1, self.CELL - 3, self.CELL - 3)
        p.end()

    def mouseMoveEvent(self, e):
        i = self._index(e.position())
        if i is not None:
            r, g, b = self.colours[i]
            self.setToolTip("%d: #%02X%02X%02X" % (i, r, g, b))

    def mousePressEvent(self, e):
        i = self._index(e.position())
        if i is not None:
            self.current = i
            self.update()
            self.chosen.emit(i)


def _run_busy(widget, fn, *args):
    """Run fn under the main window's wait cursor (and repaint), if found."""
    w = widget
    while w is not None and not hasattr(w, "_busy"):
        w = w.parent()
    result = w._busy(fn, *args) if w is not None else fn(*args)
    if w is not None and hasattr(w, "_after_pixels"):
        w._after_pixels()
    return result


def _swatch_icon(colours, size=16):
    pix = QPixmap(size * len(colours) + 2 * (len(colours) - 1), size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    for k, c in enumerate(colours):
        p.fillRect(k * (size + 2), 0, size, size, QColor(*c[:3]))
    p.end()
    return QIcon(pix)


class ColourPickerDialog(QDialog):
    def __init__(self, ctl, current, title="Choose colour", parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle(title)
        self.colour = tuple(current) + ((255,) if len(current) == 3 else ())
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        # ---- Palette tab
        page = QWidget()
        box = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Palette"))
        self.combo = QComboBox()
        self._fill_combo()
        self.combo.currentIndexChanged.connect(self._palette_chosen)
        row.addWidget(self.combo, 1)
        box.addLayout(row)
        self.grid = SwatchGrid()
        self.grid.chosen.connect(self._swatch)
        box.addWidget(self.grid, 1)
        tools = QHBoxLayout()
        b = QPushButton("Build Palette from Image...")
        b.setToolTip("Make a palette of the colours this image uses")
        b.clicked.connect(self._build)
        tools.addWidget(b)
        b = QPushButton("Convert Image to Palette")
        b.setToolTip("Change every pixel layer's colours to this palette's nearest")
        b.clicked.connect(self._convert)
        tools.addWidget(b)
        box.addLayout(tools)
        self.tabs.addTab(page, "Palette")

        # ---- All Colors tab: the full chooser, embedded
        self.chooser = QColorDialog(QColor(*self.colour))
        self.chooser.setOption(QColorDialog.ShowAlphaChannel, True)
        self.chooser.setOption(QColorDialog.NoButtons, True)
        self.chooser.setWindowFlags(Qt.Widget)
        self.chooser.currentColorChanged.connect(self._chooser_changed)
        self.tabs.addTab(self.chooser, "All Colors")

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

        self._palette_chosen(self.combo.currentIndex())
        # The palette is where you start whenever there is one.
        self.tabs.setCurrentIndex(0 if ctl.swatch_palette is not None else 1)
        self._show()

    def _fill_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self._keys = []
        for g, name, key in self.ctl.palette_sources():
            self.combo.addItem(name)
            self.combo.setItemData(self.combo.count() - 1, g, Qt.ToolTipRole)
            self._keys.append(key)
        if self.ctl.swatch_key in self._keys:
            self.combo.setCurrentIndex(self._keys.index(self.ctl.swatch_key))
        self.combo.blockSignals(False)

    def _palette_chosen(self, index):
        from ..engine import palettefiles as PF
        key = self._keys[index] if 0 <= index < len(self._keys) else None
        pal = None
        if key is not None:
            try:
                pal = self.ctl.palette_for(key) if key != self.ctl.swatch_key else self.ctl.swatch_palette
            except Exception:                               # noqa: BLE001
                pal = None
        self.palette = pal
        self.grid.set_colours(PF.colours(pal).tolist() if pal is not None else [])

    def _swatch(self, i):
        r, g, b = self.grid.colours[i]
        self.colour = (r, g, b, 255)
        self.chooser.blockSignals(True)
        self.chooser.setCurrentColor(QColor(r, g, b, 255))
        self.chooser.blockSignals(False)
        self._show()

    def _chooser_changed(self, c):
        self.colour = (c.red(), c.green(), c.blue(), c.alpha())
        self._show()

    def _show(self):
        r, g, b, a = self.colour
        self.preview.setStyleSheet("background:#%02x%02x%02x; border:1px solid #444" % (r, g, b))
        self.hex.setText("#%02X%02X%02X  alpha %d" % (r, g, b, a))

    def _build(self):
        dlg = BuildPaletteDialog(self.ctl, self)
        if dlg.exec():
            self._fill_combo()
            self._palette_chosen(self.combo.currentIndex())

    def _convert(self):
        if self.palette is None:
            return
        if self.ctl.swatch_palette is not self.palette:
            self.ctl.set_swatch_palette(self._keys[self.combo.currentIndex()])
        _run_busy(self, self.ctl.apply_palette_all, "none", self.palette)

    def result_colour(self):
        return self.colour


class BuildPaletteDialog(QDialog):
    """A palette from the colours the image uses, merging the closest ones
    when there are too many -- with the merges shown before building."""

    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle("Build Palette from Image")
        self.cols, self.counts = ctl.used_colours()
        n = len(self.cols)
        outer = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(QLabel("Palette size"))
        self.size = QSpinBox()
        self.size.setRange(1, 256)
        self.size.setValue(max(1, min(256, n)))
        self.size.valueChanged.connect(self.replan)
        row.addWidget(self.size)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addWidget(QLabel("Recommended merges (closest colours first):"))
        self.list = QListWidget()
        self.list.setIconSize(QSize(120, 16))
        outer.addWidget(self.list, 1)
        self.apply = QCheckBox("Change the image to use the merged colours")
        self.apply.setChecked(True)
        outer.addWidget(self.apply)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.build_button = buttons.addButton("Build Palette", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.resize(460, 420)
        self.replan()

    def replan(self):
        from ..engine import palettefiles as PF
        n = len(self.cols)
        target = self.size.value()
        self.plan = PF.plan_palette(self.cols, self.counts, target)
        self.list.clear()
        if n == 0:
            self.summary.setText("The image has no opaque colours to build from.")
            self.build_button.setEnabled(False)
            return
        if self.plan.exact:
            self.summary.setText("The image uses %d colour(s): the palette holds them all exactly, "
                                 "nothing merged." % n)
            self.apply.setEnabled(False)
        else:
            note = " (a photograph-like image: rare colours were folded into frequent ones " \
                   "first, so the list is approximate)" if self.plan.approximate else ""
            self.summary.setText("The image uses %d colours, more than %d. The closest colours "
                                 "below will be merged into one each%s." % (n, target, note))
            self.apply.setEnabled(True)
        for members, result, spread in self.plan.merges[:400]:
            shown = [tuple(self.cols[i]) for i in members[:5]]
            item = QListWidgetItem(_swatch_icon(shown + [result]),
                                   "%d colours -> #%02X%02X%02X   (delta E up to %.1f)"
                                   % ((len(members),) + tuple(result) + (spread,)))
            item.setToolTip("Merged: " + ", ".join("#%02X%02X%02X" % tuple(self.cols[i])
                                                   for i in members[:24]))
            self.list.addItem(item)
        if len(self.plan.merges) > 400:
            self.list.addItem("... and %d more" % (len(self.plan.merges) - 400))
        self.build_button.setEnabled(True)

    def accept(self):
        apply = self.apply.isEnabled() and self.apply.isChecked()
        _run_busy(self, self.ctl.build_palette_from_image, self.size.value(), apply, self.plan)
        super().accept()
