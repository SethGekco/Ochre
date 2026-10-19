# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Palette bar: a row of the top toolbar for working with palettes.

  Palette    the palette in use: the document's, a built-in one, any file in
             the palette folders (Preferences > Palette > Folders, searched
             with their subfolders), one loaded with Load..., or an addon's.
  swatches   its colours; left-click sets the primary, right-click the
             secondary (Ctrl-click a swatch to see its index and value).
  Colours    Reduce: limit the active layer to that many colours, with a
             palette made for it (which becomes the bar's palette).
  Apply      map the active layer to the palette's colours (Dither: none,
             ordered, Floyd-Steinberg).
  Indexed    turn the active layer into an index-locked layer on the palette.
  Lock       paint with the palette's colours only.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel, QMessageBox, QSpinBox,
                               QToolBar, QToolButton, QWidget)


class SwatchStrip(QWidget):
    """A palette's colours as a compact grid, ROWS high."""

    picked = Signal(int, int)        # index, button (1 left, 2 right)
    ROWS = 4
    CELL = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colours = []
        self.setMouseTracking(True)
        self.setMinimumHeight(self.ROWS * self.CELL + 2)

    def set_colours(self, colours):
        self.colours = [tuple(int(v) for v in c[:3]) for c in colours]
        self.updateGeometry()
        self.update()

    def _cols(self):
        return max(1, (len(self.colours) + self.ROWS - 1) // self.ROWS)

    def sizeHint(self):
        return QSize(min(1024, self._cols() * self.CELL + 2), self.ROWS * self.CELL + 2)

    def minimumSizeHint(self):
        return QSize(min(256, self._cols() * self.CELL + 2), self.ROWS * self.CELL + 2)

    def _cell(self):
        cols = self._cols()
        return max(3, min(self.CELL * 2, (self.width() - 2) // cols))

    def _index_at(self, pos):
        c = self._cell()
        col, row = int(pos.x() - 1) // c, int(pos.y() - 1) // c
        i = col * self.ROWS + row                  # column-major: colours read down, then across
        return i if 0 <= row < self.ROWS and 0 <= i < len(self.colours) else None

    def paintEvent(self, _event):
        p = QPainter(self)
        c = self._cell()
        for i, (r, g, b) in enumerate(self.colours):
            col, row = divmod(i, self.ROWS)
            p.fillRect(1 + col * c, 1 + row * c, c - 1, c - 1, QColor(r, g, b))
        p.end()

    def mouseMoveEvent(self, event):
        i = self._index_at(event.position())
        if i is not None:
            r, g, b = self.colours[i]
            self.setToolTip("%d: #%02X%02X%02X  (%d, %d, %d)" % (i, r, g, b, r, g, b))

    def mousePressEvent(self, event):
        i = self._index_at(event.position())
        if i is not None:
            self.picked.emit(i, 2 if event.button() == Qt.RightButton else 1)


class PaletteBar(QToolBar):
    def __init__(self, ctl, window):
        super().__init__("Palette", window)
        self.setObjectName("PaletteBar")
        self.ctl, self.win = ctl, window
        self.setMovable(True)
        self.addWidget(QLabel(" Palette "))
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(16)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.activated.connect(self._chosen)
        self.addWidget(self.combo)
        self._button("Load...", self.load, "Load a palette file (.pal, .gpl, .act, .txt, .hex)")
        self._button("Save...", self.save, "Save the palette (.gpl, .act, .txt, or JASC .pal)")
        self._button("From Image...", self.from_image,
                     "Build a palette from the colours the image uses (closest merged if too many)")
        self.strip = SwatchStrip()
        self.strip.picked.connect(self._pick)
        self.addWidget(self.strip)
        self.addSeparator()
        self.addWidget(QLabel(" Colours "))
        self.count = QSpinBox()
        self.count.setRange(2, 256)
        self.count.setValue(16)
        self.count.setToolTip("How many colours Reduce leaves")
        self.addWidget(self.count)
        self._button("Reduce", self.reduce, "Limit the active layer to this many colours")
        self.dither = QComboBox()
        for value, text in (("none", "No dither"), ("ordered", "Ordered"), ("floyd", "Floyd-Steinberg")):
            self.dither.addItem(text, value)
        self.addWidget(self.dither)
        self._button("Apply", self.apply, "Map the active layer to the palette's colours")
        self._button("Indexed", self.indexed, "Make the active layer index-locked on this palette")
        self.lock = self._button("Lock", lambda: self.ctl.set_palette_lock(self.lock.isChecked()),
                                 "Paint with the palette's colours only")
        self.lock.setCheckable(True)
        self.refresh_sources()
        if self.ctl.swatch_palette is None:
            try:
                self.ctl.restore_swatch_palette()
            except Exception:                               # noqa: BLE001
                pass
        self.show_palette()

    def _button(self, text, slot, tip):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda *_: slot())
        self.addWidget(b)
        return b

    # ---- the list of palettes ------------------------------------------------------

    def refresh_sources(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self._keys = []
        group = None
        for g, name, key in self.ctl.palette_sources():
            if g != group:
                if group is not None:
                    self.combo.insertSeparator(self.combo.count())
                    self._keys.append(None)
                group = g
            self.combo.addItem(name if g in ("Built-in", "Document") else "%s" % name)
            self.combo.setItemData(self.combo.count() - 1, g, Qt.ToolTipRole)
            self._keys.append(key)
        self.combo.blockSignals(False)
        self._select_current()

    def _select_current(self):
        key = self.ctl.swatch_key
        if key is None:
            return
        for i, k in enumerate(self._keys):
            if k == key:
                self.combo.setCurrentIndex(i)
                return

    def _chosen(self, index):
        key = self._keys[index] if 0 <= index < len(self._keys) else None
        if key is None:
            return
        try:
            self.ctl.set_swatch_palette(key)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self.win, "Palette", str(exc))
            return
        self._view_indexed(key)
        self.show_palette()

    def _view_indexed(self, key):
        """On an indexed image, the palette chosen IS how it is shown: the
        indices stay, the colours follow (undoable), and the choice is
        remembered for the next indexed file."""
        if key and key[0] != "document" and self.ctl.is_indexed_document() \
                and self.ctl.swatch_palette is not None:
            self.ctl.set_document_palette(self.ctl.swatch_palette, key)
            self.win.statusBar().showMessage(
                "Showing the indices with %r (nothing in the image changed)."
                % (self.ctl.swatch_palette.name or "this palette"), 6000)
            self.win._after_pixels()
            if hasattr(self.win, "palette_dock"):
                self.win.palette_dock.sync()

    def show_palette(self):
        from ..engine import palettefiles as PF
        pal = self.ctl.swatch_palette
        self.strip.set_colours(PF.colours(pal).tolist() if pal is not None else [])
        if pal is not None:
            self.count.setValue(max(2, min(256, int(getattr(pal, "size", 256)))))
        self._select_current()

    # ---- actions --------------------------------------------------------------------

    def _pick(self, index, button):
        r, g, b = self.strip.colours[index]
        (self.ctl.set_secondary if button == 2 else self.ctl.set_primary)((r, g, b, 255))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "Load Palette", "",
                                              "Palettes (*.pal *.gpl *.act *.txt *.hex);;All files (*)")
        if not path:
            return
        try:
            self.ctl.load_palette_file(path)
            self.ctl.set_swatch_palette(("file", path))
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self.win, "Load Palette", str(exc))
            return
        self._view_indexed(("file", path))
        self.refresh_sources()
        self.show_palette()

    def save(self):
        from ..engine import palettefiles as PF
        if self.ctl.swatch_palette is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save Palette", "palette.gpl",
                                              "GIMP (*.gpl);;Adobe (*.act);;Paint.NET (*.txt);;JASC (*.pal)")
        if path:
            PF.write(self.ctl.swatch_palette, path)

    def from_image(self):
        from .colourpicker import BuildPaletteDialog
        if BuildPaletteDialog(self.ctl, self.win).exec():
            self.refresh_sources()
            self.show_palette()

    def reduce(self):
        self.win._busy(self.ctl.reduce_colours, self.count.value(), self.dither.currentData())
        self.refresh_sources()
        self.show_palette()
        self.win._after_pixels()

    def apply(self):
        if self.ctl.swatch_palette is None:
            QMessageBox.information(self.win, "Apply Palette", "Choose a palette first.")
            return
        self.win._busy(self.ctl.apply_palette, self.dither.currentData())
        self.win._after_pixels()

    def indexed(self):
        report = self.win._busy(self.ctl.convert_to_indexed, None,
                                self.dither.currentData() if self.dither.currentData() != "none"
                                else "none")
        if report is None:
            QMessageBox.information(self.win, "Indexed", "The active layer cannot be converted "
                                    "(already indexed, or not a pixel layer).")
        self.win._sync_all()
        self.win._after_pixels()
