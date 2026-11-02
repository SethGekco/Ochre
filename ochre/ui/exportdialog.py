# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""File > Export As: Photoshop's Export As / Save for Web, in one dialog.

Pick a format, a quality and a scale; the dialog ENCODES the image as you
change them and shows the real file size and a preview of exactly what
will be written -- JPEG blocking, AVIF smoothing and all -- rather than a
guess. The document itself is untouched: an export is a copy.

GIF and PNG-8 are indexed: a palette of up to 256 colours, made from the
image by median cut or taken from the document, with optional dithering
and 1-bit transparency (engine/fileio/indexedexport.py). A document that
is already indexed exports on its own palette with its indices unchanged,
and the dialog says when that is what happened.
"""

import io

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

from ..engine.fileio.flat import encode_flat
from ..engine.fileio.indexedexport import INDEXED_FORMATS

FORMATS = (("PNG", "png", False), ("PNG8", "png", False), ("JPEG", "jpg", True),
           ("WEBP", "webp", True), ("AVIF", "avif", True), ("GIF", "gif", False),
           ("TIFF", "tif", False), ("QOI", "qoi", False), ("TGA", "tga", False),
           ("BMP", "bmp", False), ("ICO", "ico", False))
LABELS = {"PNG8": "PNG-8"}

DITHERS = (("none", "None"), ("ordered", "Pattern (ordered)"),
           ("floyd", "Diffusion (Floyd-Steinberg)"))


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
        self.report = {}
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        form = QFormLayout()
        self.format = QComboBox()
        for name, ext, _q in FORMATS:
            self.format.addItem("%s (.%s)" % (LABELS.get(name, name), ext), name)
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
        # A photo's EXIF and XMP: camera, date, copyright -- and where it was
        # taken. Defaults from Preferences > Export.
        self.meta = QComboBox()
        self.meta.addItem("Keep all", "keep")
        self.meta.addItem("Remove location (GPS)", "nogps")
        self.meta.addItem("Remove all", "none")
        settings = getattr(ctl, "settings", None)
        if settings is not None and settings.get("Export", "StripMetadata"):
            self.meta.setCurrentIndex(2)
        elif settings is not None and settings.get("Export", "StripGPS"):
            self.meta.setCurrentIndex(1)
        has_meta = any(k in ctl.doc.meta for k in ("exif_data", "xmp_data"))
        self.meta.setEnabled(has_meta)
        self.meta.setToolTip("" if has_meta else "This image carries no EXIF or XMP metadata")
        form.addRow("Metadata", self.meta)
        left.addLayout(form)
        left.addWidget(self._build_indexed())
        info = QFormLayout()
        self.dims = QLabel()
        info.addRow("Size", self.dims)
        self.size = QLabel()
        self.size.setStyleSheet("font-weight: bold")
        info.addRow("File size", self.size)
        left.addLayout(info)
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
                    self.cmyk.currentIndexChanged, self.palette.currentIndexChanged,
                    self.colours.valueChanged, self.dither.currentIndexChanged,
                    self.amount.valueChanged, self.transparency.toggled,
                    self.meta.currentIndexChanged):
            sig.connect(lambda *_: self._timer.start(150))
        self.refresh()

    def _build_indexed(self):
        """The GIF / PNG-8 options, shown only for those formats."""
        from .layerdialogs import ColourButton
        self.indexed_box = QWidget()
        form = QFormLayout(self.indexed_box)
        form.setContentsMargins(0, 0, 0, 0)
        self.palette = QComboBox()
        self.palette.addItem("Generated (median cut)", "generated")
        self.palette.addItem("Document palette", "document")
        doc = self.ctl.doc
        if doc.palette is None:
            self.palette.model().item(1).setEnabled(False)
        elif self.ctl.is_indexed_document():
            # An indexed image keeps its own palette unless told otherwise.
            self.palette.setCurrentIndex(1)
        form.addRow("Palette", self.palette)
        self.colours = QSpinBox()
        self.colours.setRange(2, 256)
        self.colours.setValue(256)
        form.addRow("Colours", self.colours)
        self.dither = QComboBox()
        for key, label in DITHERS:
            self.dither.addItem(label, key)
        form.addRow("Dither", self.dither)
        self.amount = QSlider(Qt.Horizontal)
        self.amount.setRange(0, 100)
        self.amount.setValue(100)
        self.amount_label = QLabel("100%")
        arow = QHBoxLayout()
        arow.addWidget(self.amount, 1)
        arow.addWidget(self.amount_label)
        form.addRow("Amount", arow)
        self.transparency = QCheckBox("Transparency")
        self.transparency.setChecked(True)
        form.addRow("", self.transparency)
        self.matte = ColourButton((255, 255, 255), on_change=lambda *_: self._timer.start(150))
        form.addRow("Matte", self.matte)
        self.indexed_note = QLabel()
        self.indexed_note.setWordWrap(True)
        form.addRow("", self.indexed_note)
        return self.indexed_box

    def options(self):
        out = {"quality": self.quality.value(), "scale": self.scale.value() / 100.0,
               "lossless": self.lossless.isChecked(), "embed_srgb": self.embed.isChecked(),
               "strip_gps": self.meta.currentData() == "nogps",
               "strip_metadata": self.meta.currentData() == "none"}
        if self.cmyk.currentData() and self.fmt() in ("JPEG", "TIFF"):
            out["cmyk_profile"] = self.cmyk.currentData()
        if self.fmt() in INDEXED_FORMATS:
            out.update({"palette": self.palette.currentData(), "colours": self.colours.value(),
                        "dither": self.dither.currentData(),
                        "dither_amount": self.amount.value(),
                        "transparency": self.transparency.isChecked(),
                        "matte": tuple(self.matte.value[:3])})
        return out

    def fmt(self):
        return self.format.currentData()

    def refresh(self):
        fmt = self.fmt()
        lossy = dict((n, q) for n, _e, q in FORMATS)[fmt]
        indexed = fmt in INDEXED_FORMATS
        self.quality.setEnabled(lossy and not (fmt == "WEBP" and self.lossless.isChecked()))
        self.lossless.setEnabled(fmt == "WEBP")
        self.cmyk.setEnabled(fmt in ("JPEG", "TIFF"))
        self.embed.setEnabled(fmt != "GIF")
        self.indexed_box.setVisible(indexed)
        self.quality_label.setText(str(self.quality.value()))
        self.amount_label.setText("%d%%" % self.amount.value())
        opts = self.options()
        w = max(1, int(round(self.ctl.doc.width * opts["scale"])))
        h = max(1, int(round(self.ctl.doc.height * opts["scale"])))
        self.dims.setText("%d x %d px" % (w, h))
        self.report = {}
        try:
            self.data = encode_flat(self.ctl.doc, fmt, opts, flat=self._flat, report=self.report)
        except Exception as exc:                                  # noqa: BLE001
            self.data = b""
            self.size.setText("cannot encode: %s" % exc)
            return
        self.size.setText(human_size(len(self.data)))
        if indexed:
            self._sync_indexed()
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

    def _sync_indexed(self):
        """Grey what does not apply, and say what was written."""
        exact = bool(self.report.get("exact"))
        generated = self.palette.currentData() == "generated"
        self.colours.setEnabled(generated)
        self.dither.setEnabled(not exact)
        self.amount.setEnabled(not exact and self.dither.currentData() != "none")
        n = self.report.get("colours", 0)
        if exact:
            self.indexed_note.setText("Exact: the document's indices and palette, unchanged "
                                      "(%d entries)." % n)
        elif generated:
            self.indexed_note.setText("%d colours%s." % (
                n, ", one of them transparent" if self.report.get("transparent") is not None
                else ""))
        else:
            self.indexed_note.setText("Mapped to the document's palette (%d entries): the "
                                      "picture is not just its indexed layers (a scale, an "
                                      "opacity, a blend, an RGB layer)." % n)

    def _export(self):
        ext = dict((n, e) for n, e, _q in FORMATS)[self.fmt()]
        base = "untitled"
        if self.ctl.path:
            import os
            base = os.path.splitext(os.path.basename(self.ctl.path))[0]
        label = LABELS.get(self.fmt(), self.fmt())
        path, _ = QFileDialog.getSaveFileName(self, "Export", "%s.%s" % (base, ext),
                                              "%s (*.%s)" % (label, ext))
        if not path:
            return
        self.write(path)
        self.accept()

    def write(self, path):
        from ..engine.atomic import write_bytes
        return write_bytes(path, self.data)
