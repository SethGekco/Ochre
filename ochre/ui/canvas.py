# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""The canvas widget, and the numpy-to-QImage bridge underneath it.

QAbstractScrollArea with a custom paintEvent, not QGraphicsView (a whole
scene graph to display one image) and not QOpenGLWidget (a GL context to
manage for a blit that is already fast enough).

BUFFER LIFETIME is the one genuinely dangerous thing in this file. QImage
constructed over a numpy buffer does NOT own that memory -- if the array is
freed while the QImage lives, Qt reads freed memory and the process dies
somewhere unrelated and unexplainable. The rules, which are absolute:

  * CanvasSurface holds a strong reference to the array for as long as the
    QImage exists. Both are replaced together, never separately.
  * The controller's `display` array is stable: resizing replaces its
    CONTENTS, never its identity.
  * Nothing outside this module may hold the QImage.

Verified on PySide6 6.11 + numpy 2.5: a numpy write is immediately visible
through the QImage, bytesPerLine equals width*4, and sub-rect copies behave.
"""

from fractions import Fraction

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPen, QPixmap,
                           QTransform)
from PySide6.QtWidgets import QAbstractScrollArea

from ..engine.geometry import Rect
from .viewstate import ViewState, format_zoom


class CanvasSurface:
    """A QImage aliasing an engine array, with the lifetime rules enforced."""

    def __init__(self, array=None):
        self._array = None
        self._image = None
        if array is not None:
            self.attach(array)

    def attach(self, array):
        """Wrap `array` with no copy. Holds a reference for as long as needed."""
        array = np.ascontiguousarray(array, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError("canvas surface needs (H, W, 4) uint8, got %s"
                             % (array.shape,))
        h, w = array.shape[:2]
        # Keep the reference FIRST. If construction throws, we are still
        # consistent; if it succeeds, the array cannot be collected while
        # the image is alive.
        self._array = array
        self._image = QImage(array.data, w, h, w * 4, QImage.Format_RGBA8888)
        return self._image

    def detach(self):
        self._image = None
        self._array = None

    @property
    def image(self):
        return self._image

    @property
    def array(self):
        return self._array

    @property
    def size(self):
        if self._array is None:
            return (0, 0)
        return (self._array.shape[1], self._array.shape[0])

    def is_valid(self):
        return self._image is not None and self._array is not None


class CanvasView(QAbstractScrollArea):
    """Displays a document and turns input into tool events."""

    cursor_moved = Signal(float, float)
    zoom_changed = Signal(str)
    status_message = Signal(str)

    def __init__(self, controller, settings, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.cfg = settings
        self.surface = CanvasSurface()
        self.view = ViewState(1, 1)
        self._panning = False
        self._pan_origin = QPoint()
        self._drawing = False
        self._checker = None

        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Repaints are coalesced rather than issued per dirty rect. Without
        # this the UI thread drowns during a fast stroke.
        self._pending = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_repaint)
        self._repaint_ms = int(self.cfg.get("Canvas", "RepaintMs") or 16)

    # ---- document binding ------------------------------------------------

    def bind(self):
        """(Re)attach to the controller's current document."""
        doc = self.ctl.doc
        if doc is None:
            self.surface.detach()
            return
        self.surface.attach(self.ctl.display)
        self.view = ViewState(doc.width, doc.height)
        self.view.fit(max(1, self.viewport().width()),
                      max(1, self.viewport().height()), margin=16)
        self.ctl.invalidate()
        self.refresh()
        self.zoom_changed.emit(format_zoom(self.view.zoom))

    def refresh(self):
        """Composite what is pending and schedule a repaint of those regions."""
        rects = self.ctl.flush()
        if not rects:
            return
        for rect in rects:
            self._mark(rect)
        if not self._timer.isActive():
            self._timer.start(self._repaint_ms)

    def _mark(self, doc_rect):
        x, y, w, h = self.view.doc_to_widget_rect(doc_rect)
        # Inflate by one: antialiased edges bleed outside their geometric
        # bounds, and without this you get a one-pixel trail behind
        # everything that moves.
        widget_rect = QRect(int(x) - 1, int(y) - 1, int(w) + 3, int(h) + 3)
        self._pending = widget_rect if self._pending is None \
            else self._pending.united(widget_rect)

    def _do_repaint(self):
        if self._pending is None:
            self.viewport().update()
        else:
            self.viewport().update(self._pending)
            self._pending = None

    # ---- painting --------------------------------------------------------

    def _checker_brush(self):
        if self._checker is not None:
            return self._checker
        size = int(self.cfg.get("Canvas", "CheckerSize") or 16)
        light = QColor(*self.cfg.get("Canvas", "CheckerLight"))
        dark = QColor(*self.cfg.get("Canvas", "CheckerDark"))
        pix = QPixmap(size * 2, size * 2)
        pix.fill(light)
        p = QPainter(pix)
        p.fillRect(0, 0, size, size, dark)
        p.fillRect(size, size, size, size, dark)
        p.end()
        self._checker = QBrush(pix)
        return self._checker

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.fillRect(event.rect(), QColor(*self.cfg.get("Canvas", "BackgroundColour")))
        if not self.surface.is_valid():
            painter.end()
            return

        cw, ch = self.view.canvas_size()
        target = QRect(int(self.view.offset_x), int(self.view.offset_y),
                       max(1, int(round(cw))), max(1, int(round(ch))))

        # Checkerboard under the image, so transparency reads as transparency
        # rather than as whatever the background happens to be.
        painter.save()
        painter.setBrushOrigin(target.topLeft())
        painter.fillRect(target.intersected(event.rect()), self._checker_brush())
        painter.restore()

        # Nearest-neighbour at or above 100%: a pixel artist must see exactly
        # the pixels they painted. Smooth below, where interpolation prevents
        # aliasing when minifying.
        painter.setRenderHint(QPainter.SmoothPixmapTransform,
                              not self.view.use_nearest)
        painter.drawImage(target, self.surface.image)

        if self.view.percent >= (self.cfg.get("Canvas", "PixelGridAbove") or 800):
            self._draw_pixel_grid(painter, event.rect(), target)
        self._draw_selection(painter, target)
        painter.end()

    def _draw_pixel_grid(self, painter, clip, target):
        scale = self.view.scale
        pen = QPen(QColor(*self.cfg.get("Canvas", "GridColour")))
        pen.setWidth(0)
        painter.setPen(pen)
        x0 = max(target.left(), clip.left())
        x1 = min(target.right(), clip.right())
        y0 = max(target.top(), clip.top())
        y1 = min(target.bottom(), clip.bottom())
        first_col = int((x0 - self.view.offset_x) / scale)
        for col in range(first_col, int((x1 - self.view.offset_x) / scale) + 2):
            wx = int(self.view.offset_x + col * scale)
            if x0 <= wx <= x1:
                painter.drawLine(wx, y0, wx, y1)
        first_row = int((y0 - self.view.offset_y) / scale)
        for row in range(first_row, int((y1 - self.view.offset_y) / scale) + 2):
            wy = int(self.view.offset_y + row * scale)
            if y0 <= wy <= y1:
                painter.drawLine(x0, wy, x1, wy)

    def _draw_selection(self, painter, target):
        sel = self.ctl.selection
        if sel is None or sel.selects_all():
            return
        box = sel.bbox
        if box.is_empty:
            return
        x, y, w, h = self.view.doc_to_widget_rect(box)
        pen = QPen(QColor(255, 255, 255, 200))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(int(x), int(y), int(w), int(h))

    # ---- input -----------------------------------------------------------

    def _doc_pos(self, pos):
        return self.view.widget_to_doc(pos.x(), pos.y())

    @staticmethod
    def _mods(event):
        from ..engine.tools import MOD_ALT, MOD_CTRL, MOD_NONE, MOD_SHIFT
        qt_mods = event.modifiers()
        out = MOD_NONE
        if qt_mods & Qt.ShiftModifier:
            out |= MOD_SHIFT
        if qt_mods & Qt.ControlModifier:
            out |= MOD_CTRL
        if qt_mods & Qt.AltModifier:
            out |= MOD_ALT
        return out

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        dx, dy = self._doc_pos(event.position())
        button = 2 if event.button() == Qt.RightButton else 1
        self._drawing = True
        self.ctl.begin_stroke(dx, dy, 1.0, self._mods(event), button)
        self.refresh()

    def mouseMoveEvent(self, event):
        pos = event.position()
        dx, dy = self._doc_pos(pos)
        self.cursor_moved.emit(dx, dy)
        if self._panning:
            point = pos.toPoint()
            delta = point - self._pan_origin
            self._pan_origin = point
            self.view.pan_by(delta.x(), delta.y())
            self.viewport().update()
            return
        if self._drawing:
            self.ctl.motion_stroke(dx, dy, 1.0, self._mods(event))
            self.refresh()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MiddleButton:
            self._panning = False
            self.unsetCursor()
            return
        if self._drawing:
            dx, dy = self._doc_pos(event.position())
            self._drawing = False
            self.ctl.end_stroke(dx, dy, 1.0, self._mods(event))
            self.refresh()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            pos = event.position()
            anchor = (pos.x(), pos.y())
            if event.angleDelta().y() > 0:
                self.view.zoom_in(anchor)
            else:
                self.view.zoom_out(anchor)
            self.zoom_changed.emit(format_zoom(self.view.zoom))
            self.viewport().update()
            event.accept()
            return
        delta = event.angleDelta()
        self.view.pan_by(delta.x(), delta.y())
        self.viewport().update()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._drawing:
            self._drawing = False
            self.ctl.cancel_stroke()
            self.refresh()
            self.viewport().update()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewport().update()

    # ---- view commands ---------------------------------------------------

    def zoom_to(self, value):
        self.view.zoom = Fraction(value)
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        self.viewport().update()

    def zoom_fit(self):
        self.view.fit(max(1, self.viewport().width()),
                      max(1, self.viewport().height()), margin=16)
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        self.viewport().update()

    def zoom_actual(self):
        self.zoom_to(Fraction(1, 1))
        self.view.center(self.viewport().width(), self.viewport().height())
        self.viewport().update()

    def sizeHint(self):
        return QSize(800, 600)
