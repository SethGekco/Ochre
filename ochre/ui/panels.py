# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Rulers, and the Info, Histogram and Navigator docks.

All of them only READ: positions and colours from the controller, the view
from the canvas. The one thing that writes is dragging a guide out of a
ruler, which goes through Controller.add_guide like any other change.
"""

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QDockWidget, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

RULER = 18


def _nice_step(px_per_unit, minimum=50):
    """A 1/2/5 x 10^n step whose ticks sit at least `minimum` px apart."""
    if px_per_unit <= 0:
        return 100
    raw = minimum / px_per_unit
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * p >= raw:
            return m * p
    return 10 * p


class Ruler(QWidget):
    """Document coordinates along one edge of the canvas. Drag out a guide."""

    def __init__(self, canvas, horizontal):
        super().__init__(canvas)
        self.canvas = canvas
        self.horizontal = horizontal
        self._drag = None
        self.setCursor(Qt.SizeVerCursor if horizontal else Qt.SizeHorCursor)
        self.setMouseTracking(True)
        self.pointer = None

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        p.setPen(self.palette().windowText().color())
        view = self.canvas.view
        p.drawLine(self.rect().bottomLeft() if self.horizontal else self.rect().topRight(),
                   self.rect().bottomRight())
        if view.rotated:
            p.drawText(self.rect(), Qt.AlignCenter, "")
            p.end()
            return
        scale = view.scale
        step = _nice_step(scale)
        length = self.width() if self.horizontal else self.height()
        offset = view.offset_x if self.horizontal else view.offset_y
        first = math.floor((-offset) / scale / step) * step
        font = QFont(self.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2))
        p.setFont(font)
        v = first
        while True:
            pos = offset + v * scale
            if pos > length:
                break
            for k in range(10):
                sub = offset + (v + k * step / 10.0) * scale
                if 0 <= sub <= length:
                    t = RULER // 2 if k == 5 else (RULER // 4 if k else RULER)
                    if self.horizontal:
                        p.drawLine(QPointF(sub, RULER), QPointF(sub, RULER - t))
                    else:
                        p.drawLine(QPointF(RULER, sub), QPointF(RULER - t, sub))
            if pos >= -40:
                label = ("%g" % v)
                if self.horizontal:
                    p.drawText(QPointF(pos + 2, 9), label)
                else:
                    p.save()
                    p.translate(9, pos + 2)
                    p.rotate(90)
                    p.drawText(QPointF(0, 0), label)
                    p.restore()
            v += step
        if self.pointer is not None:
            p.setPen(QPen(QColor(220, 40, 40), 1))
            if self.horizontal:
                p.drawLine(QPointF(self.pointer, 0), QPointF(self.pointer, RULER))
            else:
                p.drawLine(QPointF(0, self.pointer), QPointF(RULER, self.pointer))
        p.end()

    # Drag a guide out of the ruler, as in every editor with rulers.
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = True
            self.grabMouse()

    def _doc_pos(self, event):
        vp = self.canvas.viewport()
        local = vp.mapFromGlobal(event.globalPosition().toPoint())
        return local, self.canvas.view.widget_to_doc(local.x(), local.y())

    def mouseMoveEvent(self, event):
        if self._drag:
            local, (dx, dy) = self._doc_pos(event)
            self.canvas.guide_preview = ("h", dy) if self.horizontal else ("v", dx)
            self.canvas.viewport().update()

    def mouseReleaseEvent(self, event):
        if not self._drag:
            return
        self._drag = None
        self.releaseMouse()
        self.canvas.guide_preview = None
        local, (dx, dy) = self._doc_pos(event)
        vp = self.canvas.viewport()
        if vp.rect().contains(local):
            ctl = self.canvas.ctl
            if self.horizontal:
                ctl.add_guide("h", round(ctl.snap_point(0, dy)[1]) if ctl.snap_grid else round(dy))
            else:
                ctl.add_guide("v", round(ctl.snap_point(dx, 0)[0]) if ctl.snap_grid else round(dx))
        vp.update()


class InfoDock(QDockWidget):
    """Where the pointer is and what colour is under it; the samplers; the
    measurement; the selection. Photoshop's Info panel."""

    def __init__(self, controller, parent=None):
        super().__init__("Info", parent)
        self.setObjectName("Info")
        self.ctl = controller
        self.label = QLabel()
        self.label.setTextFormat(Qt.RichText)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setWidget(self.label)
        self._pointer = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.sync)

    def pointer(self, x, y):
        self._pointer = (x, y)
        if not self._timer.isActive():
            self._timer.start(60)

    def schedule(self, **_):
        if not self._timer.isActive():
            self._timer.start(120)

    @staticmethod
    def _colour_line(rgba):
        from ..engine.annotations import hsv
        if rgba is None:
            return "&mdash;"
        r, g, b, a = rgba
        h, s, v = hsv(rgba)
        return ("<span style='background:#%02x%02x%02x'>&nbsp;&nbsp;&nbsp;&nbsp;</span> "
                "RGB %d, %d, %d &nbsp;A %d &nbsp;#%02X%02X%02X &nbsp;HSV %d&deg; %d%% %d%%"
                % (r, g, b, r, g, b, a, r, g, b, h, s, v))

    def sync(self, **_):
        if not self.isVisible() or self.ctl.doc is None:
            return
        doc = self.ctl.doc
        rows = ["<b>Image</b> %d &times; %d" % (doc.width, doc.height)]
        if self._pointer is not None:
            x, y = self._pointer
            rows.append("<b>Pointer</b> %d, %d" % (int(math.floor(x)), int(math.floor(y))))
            rows.append(self._colour_line(self.ctl.sample_at(x, y)))
        sel = self.ctl.selection
        if sel is not None and sel.mask is not None:
            b = sel.bbox
            rows.append("<b>Selection</b> %d, %d &nbsp; %d &times; %d" % (b.x, b.y, b.w, b.h))
        size = 1
        tool = self.ctl.tool
        if tool is not None and getattr(tool, "name", "") == "color_sampler":
            size = int(tool.option("sample_size", "1") or 1)
        for i, (sx, sy) in enumerate(doc.samplers):
            rows.append("<b>#%d</b> (%d, %d) %s" % (i + 1, int(sx), int(sy),
                                                    self._colour_line(self.ctl.sample_at(sx, sy, size))))
        line = getattr(tool, "line", None) or self.ctl.last_measure
        if line is not None:
            from ..engine.annotations import measure
            m = measure(*line)
            rows.append("<b>Measure</b> %.1f px &nbsp;%.2f&deg; &nbsp;(dx %.1f, dy %.1f)"
                        % (m["distance"], m["angle"], m["dx"], m["dy"]))
        if doc.counts:
            rows.append("<b>Count</b> %d" % len(doc.counts))
        self.label.setText("<br>".join(rows))


class HistogramView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.channel = "value"
        self.setMinimumHeight(110)

    def sizeHint(self):
        return QSize(240, 120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        if not self.data or self.data.get("pixels", 0) <= 0:
            p.end()
            return
        chans = ["red", "green", "blue"] if self.channel == "rgb" else [self.channel]
        colours = {"red": QColor(220, 40, 40, 150), "green": QColor(40, 170, 40, 150),
                   "blue": QColor(40, 80, 220, 150), "value": QColor(40, 40, 40, 200)}
        peak = max(float(np.max(self.data[c][1:255])) if np.max(self.data[c][1:255]) > 0
                   else float(np.max(self.data[c])) for c in chans) or 1.0
        w, h = self.width(), self.height()
        for c in chans:
            path = QPainterPath(QPointF(0, h))
            for i, v in enumerate(self.data[c]):
                x = i * w / 256.0
                path.lineTo(QPointF(x, h - min(1.0, v / peak) * (h - 2)))
                path.lineTo(QPointF((i + 1) * w / 256.0, h - min(1.0, v / peak) * (h - 2)))
            path.lineTo(QPointF(w, h))
            p.fillPath(path, colours[c])
        p.end()


class HistogramDock(QDockWidget):
    """Tone distribution of the image or the active layer, within the
    selection when there is one. GIMP's Histogram dialog."""

    def __init__(self, controller, parent=None):
        super().__init__("Histogram", parent)
        self.setObjectName("Histogram")
        self.ctl = controller
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        self.channel = QComboBox()
        for key, text in (("value", "Value"), ("rgb", "RGB"), ("red", "Red"),
                          ("green", "Green"), ("blue", "Blue")):
            self.channel.addItem(text, key)
        self.source = QComboBox()
        self.source.addItem("Image", "image")
        self.source.addItem("Layer", "layer")
        row.addWidget(self.channel)
        row.addWidget(self.source)
        lay.addLayout(row)
        self.view = HistogramView()
        lay.addWidget(self.view)
        self.stats = QLabel()
        lay.addWidget(self.stats)
        self.setWidget(body)
        self.channel.currentIndexChanged.connect(self._channel)
        self.source.currentIndexChanged.connect(lambda _i: self.sync())
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.sync)
        self.visibilityChanged.connect(lambda vis: vis and self.schedule())

    def _channel(self, _i):
        self.view.channel = self.channel.currentData()
        self.view.update()

    def schedule(self, **_):
        if not self._timer.isActive():
            self._timer.start(250)

    def sync(self, **_):
        if not self.isVisible() or self.ctl.doc is None:
            return
        data = self.ctl.histogram(self.source.currentData())
        self.view.data = data
        self.view.channel = self.channel.currentData()
        self.view.update()
        self.stats.setText("Mean %.1f  Std %.1f  Median %d  Pixels %d"
                           % (data["mean"], data["std"], data["median"], int(round(data["pixels"]))))


class NavigatorView(QWidget):
    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.thumb = None
        self.setMinimumSize(160, 120)
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        return QSize(220, 160)

    def _frame(self):
        if self.thumb is None:
            return None
        tw, th = self.thumb.width(), self.thumb.height()
        s = min(self.width() / float(tw), self.height() / float(th))
        w, h = tw * s, th * s
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(60, 60, 64))
        frame = self._frame()
        canvas = self.dock.canvas
        if frame is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(frame, self.thumb)
            doc = self.dock.ctl.doc
            vp = canvas.viewport()
            corners = [canvas.view.widget_to_doc(x, y) for x, y in
                       ((0, 0), (vp.width(), 0), (vp.width(), vp.height()), (0, vp.height()))]
            sx, sy = frame.width() / doc.width, frame.height() / doc.height
            pts = [QPointF(frame.x() + cx * sx, frame.y() + cy * sy) for cx, cy in corners]
            p.setPen(QPen(QColor(230, 40, 40), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(pts)
        p.end()

    def _jump(self, pos):
        frame = self._frame()
        if frame is None or frame.width() <= 0:
            return
        doc = self.dock.ctl.doc
        dx = (pos.x() - frame.x()) / frame.width() * doc.width
        dy = (pos.y() - frame.y()) / frame.height() * doc.height
        self.dock.canvas.centre_on(dx, dy)

    def mousePressEvent(self, event):
        self._jump(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._jump(event.position())


class NavigatorDock(QDockWidget):
    """The whole image small, with the visible area outlined: click or drag
    to go there. Photoshop's and GIMP's Navigator."""

    def __init__(self, controller, canvas, parent=None):
        super().__init__("Navigator", parent)
        self.setObjectName("Navigator")
        self.ctl = controller
        self.canvas = canvas
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        self.view = NavigatorView(self)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        for text, fn in (("−", lambda: canvas.zoom_step(-1)), ("Fit", canvas.zoom_fit),
                         ("100%", canvas.zoom_actual), ("+", lambda: canvas.zoom_step(+1))):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: (f(), self.view.update()))
            row.addWidget(b)
        lay.addLayout(row)
        self.setWidget(body)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.sync)
        self.visibilityChanged.connect(lambda vis: vis and self.schedule())

    def schedule(self, **_):
        if not self._timer.isActive():
            self._timer.start(300)

    def sync(self, **_):
        if not self.isVisible() or self.ctl.doc is None:
            return
        img = self.ctl.merged()
        h, w = img.shape[:2]
        step = max(1, int(math.ceil(max(w / 400.0, h / 300.0))))
        small = np.ascontiguousarray(img[::step, ::step])
        self._buf = small
        self.view.thumb = QImage(small.data, small.shape[1], small.shape[0],
                                 small.shape[1] * 4, QImage.Format_RGBA8888)
        self.view.update()


class ActionsDock(QDockWidget):
    """Photoshop's Actions panel: record a sequence of commands once, play
    it on any image, or on a whole folder (File > Automate > Batch)."""

    def __init__(self, controller, parent=None):
        super().__init__("Actions", parent)
        self.setObjectName("Actions")
        self.ctl = controller
        from PySide6.QtWidgets import QListWidget, QPlainTextEdit
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _r: self._show_steps())
        lay.addWidget(self.list, 1)
        self.steps = QPlainTextEdit()
        self.steps.setReadOnly(True)
        self.steps.setMaximumHeight(110)
        lay.addWidget(self.steps)
        row = QHBoxLayout()
        self.record_button = QPushButton("● Record")
        self.record_button.clicked.connect(self._record)
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._play)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete)
        for b in (self.record_button, self.play_button, self.delete_button):
            row.addWidget(b)
        lay.addLayout(row)
        self.setWidget(body)
        self.sync()

    def current(self):
        item = self.list.currentItem()
        return item.text() if item is not None else None

    def sync(self, **_):
        names = sorted(self.ctl.actions())
        cur = self.current()
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(names)
        if cur in names:
            self.list.setCurrentRow(names.index(cur))
        self.list.blockSignals(False)
        rec = getattr(self.ctl, "recording", None)
        self.record_button.setText("■ Stop" if rec is not None else "● Record")
        self.play_button.setEnabled(rec is None and self.current() is not None)
        self._show_steps()

    def _show_steps(self):
        rec = getattr(self.ctl, "recording", None)
        steps = rec["steps"] if rec is not None else self.ctl.actions().get(self.current() or "", [])
        lines = []
        for s in steps:
            args = ", ".join([repr(a) for a in s.get("args", [])]
                             + ["%s=%r" % kv for kv in s.get("kwargs", {}).items()])
            lines.append("%s(%s)" % (s["op"], args))
        prefix = "Recording '%s'...\n" % rec["name"] if rec is not None else ""
        self.steps.setPlainText(prefix + "\n".join(lines))

    def _record(self):
        from PySide6.QtWidgets import QInputDialog
        if getattr(self.ctl, "recording", None) is not None:
            self.ctl.stop_recording()
        else:
            name, ok = QInputDialog.getText(self, "New Action", "Action name:",
                                            text="Action %d" % (len(self.ctl.actions()) + 1))
            if ok and name.strip():
                self.ctl.start_recording(name.strip())
        self.sync()

    def _play(self):
        name = self.current()
        if name:
            self.ctl.play_action(name)
            window = self.parent()
            if hasattr(window, "canvas"):
                window.canvas.bind(keep_view=True)
                window.canvas.refresh()

    def _delete(self):
        name = self.current()
        if name:
            self.ctl.delete_action(name)
            self.sync()



class ChannelsDock(QDockWidget):
    """The image's channels and the saved selections (GIMP's Channels dialog,
    Photoshop's Channels panel). Click Red, Green, Blue or Alpha to VIEW
    just that channel; save the selection as a channel and load it back."""

    VIEWS = (("RGB", None), ("Red", 0), ("Green", 1), ("Blue", 2), ("Alpha", 3))

    def __init__(self, controller, canvas, parent=None):
        super().__init__("Channels", parent)
        self.setObjectName("Channels")
        self.ctl, self.canvas = controller, canvas
        from PySide6.QtWidgets import QListWidget
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._picked)
        self.list.itemDoubleClicked.connect(self._rename)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Save selection", self._save), ("Load", self._load),
                         ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.setWidget(body)
        self._syncing = False

    def sync(self, **_):
        if self.ctl.doc is None:
            return
        self._syncing = True
        row = self.list.currentRow()
        self.list.clear()
        for label, _c in self.VIEWS:
            self.list.addItem(label)
        for ch in self.ctl.doc.channels:
            self.list.addItem("\u25a1 " + ch.name)
        view = getattr(self.canvas, "view_channel", None)
        self.list.setCurrentRow(row if 0 <= row < self.list.count() else
                                next((i for i, (_l, c) in enumerate(self.VIEWS) if c == view), 0))
        self._syncing = False

    def _channel_index(self):
        row = self.list.currentRow()
        return row - len(self.VIEWS) if row >= len(self.VIEWS) else None

    def _picked(self, row):
        if self._syncing or row < 0:
            return
        if row < len(self.VIEWS):
            self.canvas.view_channel = self.VIEWS[row][1]
            self.canvas.viewport().update()

    def _save(self):
        self.ctl.save_selection()
        self.sync()

    def _load(self):
        index = self._channel_index()
        if index is not None:
            self.ctl.load_selection(index)
            self.canvas.viewport().update()

    def _delete(self):
        index = self._channel_index()
        if index is not None:
            self.ctl.delete_channel(index)
            self.sync()

    def _rename(self, item):
        from PySide6.QtWidgets import QInputDialog
        index = self._channel_index()
        if index is None:
            return
        name, ok = QInputDialog.getText(self, "Channel", "Name:",
                                        text=self.ctl.doc.channels[index].name)
        if ok:
            self.ctl.rename_channel(index, name)
            self.sync()


class CompsDock(QDockWidget):
    """Photoshop's Layer Comps: snapshots of which layers are shown, with
    their opacity, modes and styles -- versions of a design in one file."""

    def __init__(self, controller, parent=None):
        super().__init__("Layer Comps", parent)
        self.setObjectName("LayerComps")
        self.ctl = controller
        from PySide6.QtWidgets import QListWidget
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _i: self._apply())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("New", self._new), ("Apply", self._apply), ("Update", self._update),
                         ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.setWidget(body)

    def sync(self, **_):
        if self.ctl.doc is None:
            return
        row = self.list.currentRow()
        self.list.clear()
        for c in self.ctl.doc.comps:
            self.list.addItem(c.get("name", "Layer Comp"))
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    def _new(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Layer Comp", "Name:",
                                        text="Layer Comp %d" % (len(self.ctl.doc.comps) + 1))
        if ok and name.strip():
            self.ctl.new_comp(name.strip())
            self.sync()

    def _apply(self):
        if self.list.currentRow() >= 0:
            self.ctl.apply_comp(self.list.currentRow())

    def _update(self):
        if self.list.currentRow() >= 0:
            self.ctl.update_comp(self.list.currentRow())

    def _delete(self):
        if self.list.currentRow() >= 0:
            self.ctl.delete_comp(self.list.currentRow())
            self.sync()
