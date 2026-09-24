# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Effect dialogs, generated from an effect's declared properties.

No effect lays out a dialog. It declares typed properties with constraints
and rules, and the widgets are built from that description -- which is why
every effect looks and behaves consistently, and why an addon's effect gets
a proper dialog without shipping any UI code.

Two behaviours are worth calling out because they are what make a preview
feel responsive rather than merely correct:

Preview renders only the VISIBLE region, inflated by the effect's halo, and
never the whole document. A blur of a 4000x4000 canvas is seconds of work;
the part you are looking at is milliseconds.

A slider moving cancels the in-flight render rather than queueing another.
Without that, dragging a slider builds a backlog and the preview lags further
behind the further you drag.
"""

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..engine.effects import CancelToken, Cancelled


class _Row(QWidget):
    """One property, as a labelled control plus its live value."""

    changed = Signal()

    def __init__(self, prop, parent=None):
        super().__init__(parent)
        self.prop = prop
        self.widget = None
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._build(box)
        self.refresh()

    def _build(self, box):
        prop = self.prop
        kind = prop.kind

        if kind in ("int", "float", "angle"):
            is_int = kind == "int"
            self.slider = QSlider(Qt.Horizontal)
            steps = 1 if is_int else 100
            self._steps = steps
            self.slider.setRange(int(prop.minimum * steps), int(prop.maximum * steps))
            self.spin = QSpinBox() if is_int else QDoubleSpinBox()
            self.spin.setRange(prop.minimum, prop.maximum)
            if not is_int:
                self.spin.setDecimals(int(prop.ui.get("decimals", 2)))
                self.spin.setSingleStep(0.05)
            self.slider.valueChanged.connect(self._from_slider)
            self.spin.valueChanged.connect(self._from_spin)
            box.addWidget(self.slider, 1)
            box.addWidget(self.spin)
            self.widget = self.slider

        elif kind == "bool":
            self.check = QCheckBox(prop.ui.get("description") or "")
            self.check.toggled.connect(self._from_check)
            box.addWidget(self.check, 1)
            self.widget = self.check

        elif kind == "choice":
            self.combo = QComboBox()
            for option in prop.choices:
                self.combo.addItem(str(option), option)
            self.combo.currentIndexChanged.connect(self._from_combo)
            box.addWidget(self.combo, 1)
            self.widget = self.combo

        elif kind == "color":
            self.button = QPushButton("Choose...")
            self.button.clicked.connect(self._pick_colour)
            box.addWidget(self.button, 1)
            self.widget = self.button

        else:
            self.label = QLabel(str(prop.value))
            box.addWidget(self.label, 1)
            self.widget = self.label

        reset = QPushButton("Reset")
        reset.setFixedWidth(56)
        reset.clicked.connect(self._reset)
        box.addWidget(reset)

    # ---- widget -> model -------------------------------------------------

    def _from_slider(self, raw):
        if self._guard():
            return
        self.prop.set(raw / self._steps if self.prop.kind != "int" else raw)
        self.refresh()
        self.changed.emit()

    def _from_spin(self, raw):
        if self._guard():
            return
        self.prop.set(raw)
        self.refresh()
        self.changed.emit()

    def _from_check(self, flag):
        if self._guard():
            return
        self.prop.set(flag)
        self.changed.emit()

    def _from_combo(self, _index):
        if self._guard():
            return
        self.prop.set(self.combo.currentData())
        self.changed.emit()

    def _pick_colour(self):
        chosen = QColorDialog.getColor(QColor(*self.prop.value), self,
                                       self.prop.ui.get("label", "Colour"),
                                       QColorDialog.ShowAlphaChannel)
        if chosen.isValid():
            self.prop.set((chosen.red(), chosen.green(), chosen.blue(),
                           chosen.alpha()))
            self.refresh()
            self.changed.emit()

    def _reset(self):
        self.prop.reset()
        self.refresh()
        self.changed.emit()

    _updating = False

    def _guard(self):
        return self._updating

    # ---- model -> widget -------------------------------------------------

    def refresh(self):
        """Pull the model's current state into the widgets.

        Rules can move properties the user did not touch -- linked sliders,
        a min/max pair pushing its partner -- so every row refreshes after
        any change, not just the one that was edited.
        """
        self._updating = True
        try:
            prop = self.prop
            if prop.kind in ("int", "float", "angle"):
                self.slider.setValue(int(round(prop.value * self._steps)))
                self.spin.setValue(prop.value)
            elif prop.kind == "bool":
                self.check.setChecked(bool(prop.value))
            elif prop.kind == "choice":
                index = self.combo.findData(prop.value)
                if index >= 0:
                    self.combo.setCurrentIndex(index)
            elif prop.kind == "color":
                colour = QColor(*prop.value)
                self.button.setStyleSheet("background-color: %s;" % colour.name())
            # Read-only is a fact about the MODEL; the widget only reflects it.
            if self.widget is not None:
                self.setEnabled(not prop.readonly)
        finally:
            self._updating = False


class EffectDialog(QDialog):
    """Generated configuration dialog with live preview."""

    def __init__(self, effect, controller, parent=None, debounce_ms=80):
        super().__init__(parent)
        self.effect = effect
        self.ctl = controller
        self.setWindowTitle(effect.label)
        self._cancel = None
        self._committed = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.rows = []
        for prop in effect.props:
            row = _Row(prop)
            row.changed.connect(self._on_changed)
            form.addRow(prop.ui.get("label", prop.name), row)
            self.rows.append(row)
        layout.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # The original pixels, kept so Cancel is an exact restore and each
        # preview starts from a clean source rather than from the last one.
        cell = controller.doc.cell(controller.active_layer) if controller.active_layer else None
        self._cell = cell
        self._original = None if cell is None else cell.pixels.copy()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_preview)
        self._debounce = debounce_ms
        self._render_preview()

    # ---- preview ---------------------------------------------------------

    def _on_changed(self):
        for row in self.rows:
            row.refresh()          # rules may have moved other properties
        # Restart rather than queue: dragging a slider must not build a
        # backlog of renders the user no longer cares about.
        if self._cancel is not None:
            self._cancel.cancel()
        self._timer.start(self._debounce)

    def _preview_rect(self):
        """Only what is on screen, inflated by the effect's halo."""
        canvas = getattr(self.parent(), "canvas", None)
        if canvas is None:
            return self.ctl.doc.bounds
        visible = canvas.view.visible_doc_rect(canvas.viewport().width(),
                                               canvas.viewport().height())
        if visible is None:
            return self.ctl.doc.bounds
        return visible.inflated(self.effect.halo).clipped_to(
            self.ctl.doc.width, self.ctl.doc.height) or self.ctl.doc.bounds

    def _render_preview(self):
        if self._cell is None or self._original is None:
            return
        rect = self._preview_rect()
        self._cancel = CancelToken()
        try:
            result = self.effect.apply(self._original, [rect],
                                       cancel=self._cancel)
        except Cancelled:
            return
        self._cell.plane("rgba")[rect.slice()] = result[rect.slice()]
        self._cell.dirty.add(rect)
        self.ctl.invalidate(rect)
        canvas = getattr(self.parent(), "canvas", None)
        if canvas is not None:
            canvas.refresh()
            canvas.viewport().update()

    # ---- commit / abandon ------------------------------------------------

    def accept(self):
        """Apply to the whole layer and record one undo entry.

        The previewed region is already correct, but the rest of the layer is
        not, so the commit renders everything. What it must NOT do is render
        from the previewed pixels -- it renders from the pristine original,
        or an effect would compound with itself.
        """
        if self._cell is None or self._original is None:
            return super().accept()
        if self._cancel is not None:
            self._cancel.cancel()

        from ..engine.commands import PixelDelta

        # Restore first, so the undo entry captures the true "before".
        self._cell.plane("rgba")[...] = self._original
        before = PixelDelta.capture(self.ctl.doc, self.ctl.active_layer.id,
                                    self.ctl.doc.frame.id, self.ctl.doc.bounds,
                                    self.effect.label)
        result = self.effect.apply(self._original)
        self._cell.plane("rgba")[...] = result
        self._cell.dirty.add(self.ctl.doc.bounds)
        self.ctl.history.push(before, self.effect.label,
                              frame_id=self.ctl.doc.frame.id)
        self.ctl.invalidate()
        self.ctl.dirty_since_save = True
        self.ctl.emit("history.changed")
        self._committed = True
        super().accept()

    def reject(self):
        if self._cancel is not None:
            self._cancel.cancel()
        if self._cell is not None and self._original is not None and not self._committed:
            self._cell.plane("rgba")[...] = self._original
            self._cell.dirty.add(self.ctl.doc.bounds)
            self.ctl.invalidate()
            canvas = getattr(self.parent(), "canvas", None)
            if canvas is not None:
                canvas.refresh()
                canvas.viewport().update()
        super().reject()
