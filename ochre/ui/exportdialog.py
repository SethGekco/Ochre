# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""File > Export As: Photoshop's Export As / Save for Web, in one dialog.

Pick a format, a quality and a scale; the dialog ENCODES the image as you
change them and shows the real file size and a preview of exactly what
will be written -- JPEG blocking, AVIF smoothing and all -- rather than a
guess. The document itself is untouched: an export is a copy.
"""

import io

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QSlider, QVBoxLayout)

from ..engine.fileio.flat import encode_flat

FORMATS = (("PNG", "png", False), ("JPEG", "jpg", True), ("WEBP", "webp", True),
           ("AVIF", "avif", True), ("GIF", "gif", False), ("TIFF", "tif", False),
           ("QOI", "qoi", False), ("TGA", "tga", False), ("BMP", "bmp", False),
           ("ICO", "ico", False))


def human_size(n):
    for unit in ("bytes", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return ("%d %s" % (n, unit)) if unit == "bytes" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return str(n)


class ExportDialog(QDialog):
    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle("Export As")
        self._flat = ctl.compositor.composite(ctl.doc.bounds)
        self.data = b""
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        form = QFormLayout()
        self.format = QComboBox()
        for name, ext, _q in FORMATS:
            self.format.addItem("%s (.%s)" % (name, ext), name)
        form.addRow("Format", self.format)
        self.quality = QSlider(Qt.Horizontal)
        self.quality.setRange(1, 100)
        self.quality.setValue(85)
        self.quality_label = QLabel("85")
        qrow = QHBoxLayout()
        qrow.addWidget(self.quality, 1)
        qrow.addWidget(self.quality_label)
        form.addRow("Quality", qrow)
        self.lossless = QCheckBox("Lossless (WebP)")
        form.addRow("", self.lossless)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1, 1000)
        self.scale.setValue(100)
        self.scale.setSuffix(" %")
        form.addRow("Scale", self.scale)
        # Colour: embed sRGB when the document has no profile of its own, or
        # write CMYK through a press profile (JPEG and TIFF).
        from ..engine import colormanage
        self.embed = QCheckBox("Embed colour profile")
        self.embed.setChecked(True)
        form.addRow("", self.embed)
        self.cmyk = QComboBox()
        self.cmyk.addItem("RGB", "")
        for desc, path, cs in colormanage.system_profiles():
            if cs == "CMYK":
                self.cmyk.addItem("CMYK: " + desc, path)
        form.addRow("Colour", self.cmyk)
        self.dims = QLabel()
        form.addRow("Size", self.dims)
        self.size = QLabel()
        self.size.setStyleSheet("font-weight: bold")
        form.addRow("File size", self.size)
        left.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.export_button = buttons.addButton("Export...", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        left.addStretch(1)
        left.addWidget(buttons)
        outer.addLayout(left)
        self.preview = QLabel()
        self.preview.setMinimumSize(320, 240)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #444")
        outer.addWidget(self.preview, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)
        for sig in (self.format.currentIndexChanged, self.quality.valueChanged,
                    self.scale.valueChanged, self.lossless.toggled, self.embed.toggled,
                    self.cmyk.currentIndexChanged):
            sig.connect(lambda *_: self._timer.start(150))
        self.refresh()

    def options(self):
        out = {"quality": self.quality.value(), "scale": self.scale.value() / 100.0,
               "lossless": self.lossless.isChecked(), "embed_srgb": self.embed.isChecked()}
        if self.cmyk.currentData() and self.fmt() in ("JPEG", "TIFF"):
            out["cmyk_profile"] = self.cmyk.currentData()
        return out

    def fmt(self):
        return self.format.currentData()

    def refresh(self):
        fmt = self.fmt()
        lossy = dict((n, q) for n, _e, q in FORMATS)[fmt]
        self.quality.setEnabled(lossy and not (fmt == "WEBP" and self.lossless.isChecked()))
        self.lossless.setEnabled(fmt == "WEBP")
        self.cmyk.setEnabled(fmt in ("JPEG", "TIFF"))
        self.quality_label.setText(str(self.quality.value()))
        opts = self.options()
        w = max(1, int(round(self.ctl.doc.width * opts["scale"])))
        h = max(1, int(round(self.ctl.doc.height * opts["scale"])))
        self.dims.setText("%d x %d px" % (w, h))
        try:
            self.data = encode_flat(self.ctl.doc, fmt, opts, flat=self._flat)
        except Exception as exc:                                  # noqa: BLE001
            self.data = b""
            self.size.setText("cannot encode: %s" % exc)
            return
        self.size.setText(human_size(len(self.data)))
        # Decode what was encoded: the preview IS the file.
        try:
            img = Image.open(io.BytesIO(self.data))
            if img.mode == "CMYK":
                # Shown through its own profile, as it will print.
                from ..engine.fileio.flat import _cmyk_to_srgb
                img = _cmyk_to_srgb(img, img.info.get("icc_profile"))
            img = img.convert("RGBA")
        except Exception:                                         # noqa: BLE001
            return
        arr = np.ascontiguousarray(np.asarray(img))
        self._preview_buf = arr
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4,
                      QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _export(self):
        ext = dict((n, e) for n, e, _q in FORMATS)[self.fmt()]
        base = "untitled"
        if self.ctl.path:
            import os
            base = os.path.splitext(os.path.basename(self.ctl.path))[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export", "%s.%s" % (base, ext),
                                              "%s (*.%s)" % (self.fmt(), ext))
        if not path:
            return
        self.write(path)
        self.accept()

    def write(self, path):
        with open(path, "wb") as fh:
            fh.write(self.data)
        return path
