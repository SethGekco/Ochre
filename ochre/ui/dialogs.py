# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Small modal dialogs: resize, canvas size, layer properties.

Each one is a thin form over a controller method that is itself tested
headlessly, so what is here is layout and validation and nothing that can
be wrong in an interesting way. Values are returned through `result()` so a
test can drive the form without exec().
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QRadioButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..engine import transform


class ResizeDialog(QDialog):
    """Image > Resize. Pixels or percent, aspect kept unless released."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resize Image")
        self.ctl = controller
        self.w0, self.h0 = controller.doc.width, controller.doc.height
        self._lock = False

        form = QFormLayout()
        self.width = QSpinBox()
        self.width.setRange(1, 32768)
        self.width.setValue(self.w0)
        self.height = QSpinBox()
        self.height.setRange(1, 32768)
        self.height.setValue(self.h0)
        self.percent = QSpinBox()
        self.percent.setRange(1, 3200)
        self.percent.setValue(100)
        self.percent.setSuffix(" %")
        self.aspect = QCheckBox("Maintain aspect ratio")
        self.aspect.setChecked(True)
        self.method = QComboBox()
        for name, label in ((transform.BILINEAR, "Bilinear"),
                            (transform.BICUBIC, "Bicubic"),
                            (transform.LANCZOS, "Lanczos (best for photos)"),
                            (transform.BOX, "Box (best for shrinking)"),
                            (transform.NEAREST, "Nearest neighbour (pixel art)")):
            self.method.addItem(label, name)
        # Pixel art is the home turf: an indexed document defaults to nearest,
        # because any other filter invents colours its palette does not have.
        if controller.doc.palette is not None:
            self.method.setCurrentIndex(self.method.count() - 1)

        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("By percentage", self.percent)
        form.addRow("", self.aspect)
        form.addRow("Resampling", self.method)

        self.width.valueChanged.connect(lambda v: self._changed("w"))
        self.height.valueChanged.connect(lambda v: self._changed("h"))
        self.percent.valueChanged.connect(lambda v: self._changed("p"))

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        self.info = QLabel("")
        layout.addWidget(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._changed("w")

    def _changed(self, who):
        if self._lock:
            return
        self._lock = True
        try:
            if who == "p":
                f = self.percent.value() / 100.0
                self.width.setValue(max(1, int(round(self.w0 * f))))
                self.height.setValue(max(1, int(round(self.h0 * f))))
            elif self.aspect.isChecked():
                if who == "w":
                    self.height.setValue(max(1, int(round(
                        self.width.value() * self.h0 / float(self.w0)))))
                else:
                    self.width.setValue(max(1, int(round(
                        self.height.value() * self.w0 / float(self.h0)))))
            if who != "p":
                self.percent.setValue(max(1, int(round(
                    100.0 * self.width.value() / float(self.w0)))))
        finally:
            self._lock = False
        self.info.setText("%d x %d  ->  %d x %d" % (
            self.w0, self.h0, self.width.value(), self.height.value()))

    def result(self):
        return (self.width.value(), self.height.value(), self.method.currentData())


class CanvasSizeDialog(QDialog):
    """Image > Canvas Size. New dimensions plus a 3x3 anchor."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas Size")
        self.ctl = controller
        self.w0, self.h0 = controller.doc.width, controller.doc.height

        form = QFormLayout()
        self.width = QSpinBox()
        self.width.setRange(1, 32768)
        self.width.setValue(self.w0)
        self.height = QSpinBox()
        self.height.setRange(1, 32768)
        self.height.setValue(self.h0)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)

        grid = QGridLayout()
        self.anchor = "center"
        self._buttons = {}
        names = [["top-left", "top", "top-right"],
                 ["left", "center", "right"],
                 ["bottom-left", "bottom", "bottom-right"]]
        for r, row in enumerate(names):
            for c, name in enumerate(row):
                btn = QRadioButton()
                btn.setToolTip(name)
                btn.setChecked(name == "center")
                btn.toggled.connect(lambda on, n=name: on and setattr(self, "anchor", n))
                grid.addWidget(btn, r, c)
                self._buttons[name] = btn
        anchor_box = QWidget()
        anchor_box.setLayout(grid)
        form.addRow("Anchor", anchor_box)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_anchor(self, name):
        if name in self._buttons:
            self._buttons[name].setChecked(True)

    def result(self):
        return (self.width.value(), self.height.value(), self.anchor)


class LayerPropertiesDialog(QDialog):
    """Layers > Properties: name, visibility, blend mode, opacity.

    Applies on OK as individual PropertyDeltas, grouped into one undo step.
    """

    def __init__(self, controller, layer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layer Properties")
        self.ctl = controller
        self.layer = layer

        form = QFormLayout()
        self.name = QLineEdit(layer.name or "")
        self.visible = QCheckBox("Visible")
        self.visible.setChecked(bool(layer.visible))
        self.blend = QComboBox()
        for mode in controller.blends.ordered():
            self.blend.addItem(mode.label, mode.name)
        idx = self.blend.findData(layer.blend)
        if idx >= 0:
            self.blend.setCurrentIndex(idx)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 255)
        self.opacity.setValue(int(layer.opacity))
        self.opacity_label = QLabel("%d%%" % round(layer.opacity * 100 / 255.0))
        self.opacity.valueChanged.connect(
            lambda v: self.opacity_label.setText("%d%%" % round(v * 100 / 255.0)))
        row = QHBoxLayout()
        row.addWidget(self.opacity, 1)
        row.addWidget(self.opacity_label)
        row_w = QWidget()
        row_w.setLayout(row)

        form.addRow("Name", self.name)
        form.addRow("", self.visible)
        if not layer.is_group:
            form.addRow("Blend mode", self.blend)
        form.addRow("Opacity", row_w)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result(self):
        return {"name": self.name.text(), "visible": self.visible.isChecked(),
                "blend": self.blend.currentData(), "opacity": self.opacity.value()}

    def apply(self):
        """Push the changed properties as one undo entry. Returns the count."""
        values = self.result()
        changed = [(k, v) for k, v in values.items()
                   if v is not None and getattr(self.layer, k, None) != v]
        if not changed:
            return 0
        self.ctl.history.begin_group()
        for attr, value in changed:
            self.ctl.set_layer_property(self.layer, attr, value)
        self.ctl.history.end_group("Layer properties")
        return len(changed)

    def accept(self):
        self.apply()
        super().accept()


class NewImageDialog(QDialog):
    """File > New. Prefilled with the last document's size."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Image")
        last = controller.session.last_size() if controller.session else None
        if last is None and controller.doc is not None:
            last = (controller.doc.width, controller.doc.height)
        w0, h0 = last or (800, 600)
        form = QFormLayout()
        self.width = QSpinBox()
        self.width.setRange(1, 32768)
        self.width.setValue(w0)
        self.height = QSpinBox()
        self.height.setRange(1, 32768)
        self.height.setValue(h0)
        self.background = QComboBox()
        for label, value in (("White", (255, 255, 255, 255)),
                             ("Transparent", None),
                             ("Black", (0, 0, 0, 255))):
            self.background.addItem(label, value)
        form.addRow("Width", self.width)
        form.addRow("Height", self.height)
        form.addRow("Background", self.background)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result(self):
        return (self.width.value(), self.height.value(), self.background.currentData())
