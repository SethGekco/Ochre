# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
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

import math
from fractions import Fraction

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
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
        self._view_dragging = False
        self._needs_fit = True
        self._space_pan = False
        self._drawing = False
        self._checker = None
        # Where the pointer is, in document coordinates, or None when it has
        # left the widget. Drives the tool footprint overlay.
        self._hover = None
        self._show_while_drawing = True
        # Guides, grid and Photoshop's "extras" (samplers, counts, notes,
        # slices): what is drawn over the image, each switchable from View.
        self.show_grid = False
        self.show_guides = True
        self.show_extras = True
        self.show_path = True
        self.guide_preview = None      # ("h"|"v", pos) while a guide is dragged
        self._guide_drag = None
        self.rulers = None
        # Each open document keeps its own zoom and scroll.
        self._views = {}
        self._bound_doc = None

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

        # The text caret blinks on its own timer and is drawn as an overlay,
        # never into the layer -- it is chrome, not content, and must not end
        # up in the pixels or in undo.
        self._caret_on = True
        self._caret_timer = QTimer(self)
        self._caret_timer.timeout.connect(self._blink)
        self._caret_ms = int(self.cfg.get("Canvas", "CaretBlinkMs") or 530)

    # ---- document binding ------------------------------------------------

    def bind(self, keep_view=False):
        """(Re)attach to the controller's current document.

        keep_view is for a resize or crop of the SAME document: the display
        buffer is new but the user is mid-task, so the zoom stays and only
        the document size behind it changes.
        """
        doc = self.ctl.doc
        if doc is None:
            self.surface.detach()
            return
        self.surface.attach(self.ctl.display)
        previous = self._bound_doc
        if previous is not None and previous is not doc:
            self._views[id(previous)] = (previous, self.view.copy(), self._needs_fit)
        self._bound_doc = doc
        remembered = self._views.pop(id(doc), None)
        if not keep_view and remembered is not None and remembered[0] is doc \
                and (remembered[1].doc_w, remembered[1].doc_h) == (doc.width, doc.height):
            # Back to a document already open in another tab: where you were.
            self.view = remembered[1]
            self._needs_fit = remembered[2]
            self._fit_if_real()
            self.ctl.invalidate()
            self.refresh()
            self.zoom_changed.emit(format_zoom(self.view.zoom))
            return
        if keep_view:
            zoom, ox, oy = self.view.zoom, self.view.offset_x, self.view.offset_y
            self.view = ViewState(doc.width, doc.height, zoom=zoom)
            self.view.offset_x, self.view.offset_y = ox, oy
        else:
            self.view = ViewState(doc.width, doc.height)
            # Fit-to-window is a MODE, not a one-off calculation. bind() runs
            # during window construction, before the window is shown, when
            # the viewport is a pixel or two wide -- fitting then clamped the
            # zoom to its 1% floor, drew an 800x600 canvas as an 8x6 speck,
            # and mapped every click thousands of pixels off the document.
            # So fitting waits for a viewport worth fitting to, and keeps
            # re-fitting as the window settles until the user zooms or pans.
            self._needs_fit = True
            self._fit_if_real()
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

    # ---- text caret ------------------------------------------------------

    def _text_tool(self):
        """The active text tool, if one is mid-edit."""
        tool = self.ctl.tool
        if tool is None or getattr(tool, "name", "") != "text":
            return None
        return tool if tool.active else None

    def _blink(self):
        # The session can end without the canvas being told -- undo now
        # cancels a live tool directly on the controller. Self-stopping
        # here beats wiring a signal for it, and means any future path that
        # ends an edit cannot leave a timer running.
        if self._text_tool() is None:
            self._caret_timer.stop()
            self._caret_on = False
            return
        self._caret_on = not self._caret_on
        rect = self._caret_widget_rect()
        if rect is not None:
            self.viewport().update(rect.adjusted(-2, -2, 2, 2))

    def _start_caret(self):
        self._caret_on = True
        if not self._caret_timer.isActive():
            self._caret_timer.start(self._caret_ms)

    def _stop_caret(self):
        self._caret_timer.stop()
        self._caret_on = False
        self.viewport().update()

    def _caret_widget_rect(self):
        tool = self._text_tool()
        if tool is None:
            return None
        doc_rect = tool.caret_rect(self.ctl._context())
        if doc_rect is None:
            return None
        x, y, w, h = self.view.doc_to_widget_rect(doc_rect)
        return QRect(int(x), int(y), max(1, int(w)), max(2, int(h)))

    def _draw_caret(self, painter):
        if not self._caret_on:
            return
        rect = self._caret_widget_rect()
        if rect is None:
            return
        # Drawn in an inverting mode so it stays visible over any colour --
        # a black caret vanishes on black text, which is exactly where
        # someone is most likely to be typing.
        painter.save()
        painter.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        painter.fillRect(rect, QColor(255, 255, 255))
        painter.restore()

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

        if self.view.rotated:
            self._paint_rotated(painter)
        else:
            # Checkerboard under the image, so transparency reads as
            # transparency rather than as whatever the background happens to be.
            painter.save()
            painter.setBrushOrigin(target.topLeft())
            painter.fillRect(target.intersected(event.rect()), self._checker_brush())
            painter.restore()

            # Nearest-neighbour at or above 100%: a pixel artist must see
            # exactly the pixels they painted. Smooth below, where
            # interpolation prevents aliasing when minifying.
            painter.setRenderHint(QPainter.SmoothPixmapTransform,
                                  not self.view.use_nearest)
            painter.drawImage(target, self.surface.image)

            if self.view.percent >= (self.cfg.get("Canvas", "PixelGridAbove") or 800):
                self._draw_pixel_grid(painter, event.rect(), target)
        if self.show_grid:
            self._draw_grid(painter)
        self._draw_selection(painter, target)
        if self.show_guides or self.guide_preview is not None:
            self._draw_guides(painter)
        if self.show_extras:
            self._draw_marks(painter)
        self._draw_footprint(painter)
        self._draw_zoom_band(painter)
        self._draw_paths(painter)
        self._draw_tool_overlay(painter)
        self._draw_caret(painter)
        painter.end()
        if self.rulers is not None:
            for r in self.rulers:
                r.update()

    # ---- view tools ------------------------------------------------------

    def _view_tool(self):
        """The active tool if it moves the VIEW rather than pixels.

        Selected on the tool's own declaration rather than by name, so an
        addon can ship one without the canvas learning about it.
        """
        tool = getattr(self.ctl, "tool", None)
        if tool is not None and getattr(tool, "affects", "pixels") == "view":
            return tool
        return None

    def _tool_event(self, event, dx, dy):
        from ochre.engine.tools import ToolEvent
        return ToolEvent(dx, dy, 1.0, self._mods(event),
                         2 if event.button() == Qt.RightButton else 1)

    def _apply_view(self, request):
        """Carry out a ViewRequest. The only place view policy becomes Qt.

        Zoom anchors arrive in DOCUMENT coordinates and have to be converted
        here, because ViewState.zoom_at pins a WIDGET point -- and the whole
        point of anchoring is that the pixel under the cursor does not move,
        which is a statement about the screen.
        """
        if request is None:
            return
        self.user_took_view()
        from ochre.engine.tools.viewtools import (PAN, ROTATE, ROTATE_RESET, ZOOM_IN,
                                                  ZOOM_OUT, ZOOM_RECT)
        vp = self.viewport()
        if request.kind == ROTATE:
            self.view.rotate_by(request.angle, (vp.width() / 2.0, vp.height() / 2.0))
            vp.update()
            return
        if request.kind == ROTATE_RESET:
            self.view.set_rotation(0.0, (vp.width() / 2.0, vp.height() / 2.0))
            vp.update()
            return
        if request.kind == PAN:
            self.view.pan_by(request.dx * self.view.scale,
                             request.dy * self.view.scale)
        elif request.kind == ZOOM_RECT:
            self.view.fit_rect(request.rect, vp.width(), vp.height())
        elif request.kind in (ZOOM_IN, ZOOM_OUT):
            anchor = None
            if request.anchor is not None:
                anchor = self.view.doc_to_widget(*request.anchor)
            if request.kind == ZOOM_IN:
                self.view.zoom_in(anchor)
            else:
                self.view.zoom_out(anchor)
        else:
            return
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        vp.update()

    def _draw_tool_overlay(self, painter):
        """Whatever the active tool wants shown while it works.

        Tools describe their in-progress geometry in DOCUMENT coordinates
        (engine side, no Qt): ("poly", points, closed), ("rect", (x0, y0, x1,
        y1)), ("line", (x0, y0, x1, y1)), ("handles", points), ("text",
        (x, y), str). Drawn as a white line under a black dashed one, so it
        reads on any image, light or dark.
        """
        tool = getattr(self.ctl, "tool", None)
        if tool is None or not hasattr(tool, "overlay"):
            return
        try:
            prims = tool.overlay() or []
        except Exception:                                   # noqa: BLE001
            return
        if not prims:
            return
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF
        w2d = self.view.doc_to_widget
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        white = QPen(QColor(255, 255, 255, 230), 1.0)
        black = QPen(QColor(0, 0, 0, 230), 1.0)
        black.setDashPattern([4.0, 4.0])
        for prim in prims:
            kind = prim[0]
            if kind in ("poly", "rect", "line"):
                if kind == "poly":
                    pts = [QPointF(*w2d(x, y)) for x, y in prim[1]]
                    closed = bool(prim[2]) if len(prim) > 2 else False
                elif kind == "rect":
                    x0, y0, x1, y1 = prim[1]
                    pts = [QPointF(*w2d(x0, y0)), QPointF(*w2d(x1, y0)),
                           QPointF(*w2d(x1, y1)), QPointF(*w2d(x0, y1))]
                    closed = True
                else:
                    x0, y0, x1, y1 = prim[1]
                    pts = [QPointF(*w2d(x0, y0)), QPointF(*w2d(x1, y1))]
                    closed = False
                poly = QPolygonF(pts)
                painter.setBrush(Qt.NoBrush)
                for pen in (white, black):
                    painter.setPen(pen)
                    if closed:
                        painter.drawPolygon(poly)
                    else:
                        painter.drawPolyline(poly)
            elif kind == "curve":
                path = QPainterPath()
                for i, (p0, c0, c1, p1) in enumerate(prim[1]):
                    if i == 0:
                        path.moveTo(QPointF(*w2d(*p0)))
                    path.cubicTo(QPointF(*w2d(*c0)), QPointF(*w2d(*c1)), QPointF(*w2d(*p1)))
                painter.setBrush(Qt.NoBrush)
                for pen in (white, black):
                    painter.setPen(pen)
                    painter.drawPath(path)
            elif kind == "handles":
                painter.setPen(QPen(QColor(0, 0, 0), 1.0))
                painter.setBrush(QColor(255, 255, 255))
                for x, y in prim[1]:
                    wx, wy = w2d(x, y)
                    painter.drawRect(QRect(int(wx) - 3, int(wy) - 3, 6, 6))
            elif kind == "text":
                (x, y), text = prim[1], prim[2]
                wx, wy = w2d(x, y)
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(QPoint(int(wx) + 9, int(wy) + 15), text)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(QPoint(int(wx) + 8, int(wy) + 14), text)
        painter.restore()

    def _draw_grid(self, painter):
        """The snapping grid, in document space, through the view transform."""
        from PySide6.QtCore import QPointF
        doc = self.ctl.doc
        gx, gy = (float(v) for v in self.ctl.grid[:2])
        if gx <= 0 or gy <= 0 or min(gx, gy) * self.view.scale < 4:
            return                      # too dense to be anything but grey
        vis = self.view.visible_doc_rect(self.viewport().width(), self.viewport().height())
        if vis is None:
            return
        w2d = self.view.doc_to_widget
        painter.save()
        painter.setPen(QPen(QColor(*(self.cfg.get("Canvas", "GridColour") or (120, 120, 255, 110))), 1.0))
        x = math.floor(vis.x / gx) * gx
        while x <= vis.x1:
            painter.drawLine(QPointF(*w2d(x, 0)), QPointF(*w2d(x, doc.height)))
            x += gx
        y = math.floor(vis.y / gy) * gy
        while y <= vis.y1:
            painter.drawLine(QPointF(*w2d(0, y)), QPointF(*w2d(doc.width, y)))
            y += gy
        painter.restore()

    def _draw_guides(self, painter):
        from PySide6.QtCore import QPointF
        doc = self.ctl.doc
        w2d = self.view.doc_to_widget
        painter.save()
        guides = list(doc.guides) if self.show_guides else []
        drag = self._guide_drag
        for i, (o, pos) in enumerate(guides):
            if drag is not None and drag[0] == i:
                continue
            painter.setPen(QPen(QColor(0, 200, 230), 1.0))
            a, b = ((0, pos), (doc.width, pos)) if o == "h" else ((pos, 0), (pos, doc.height))
            painter.drawLine(QPointF(*w2d(*a)), QPointF(*w2d(*b)))
        if self.guide_preview is not None:
            o, pos = self.guide_preview
            pen = QPen(QColor(0, 200, 230), 1.0)
            pen.setDashPattern([4.0, 3.0])
            painter.setPen(pen)
            a, b = ((0, pos), (doc.width, pos)) if o == "h" else ((pos, 0), (pos, doc.height))
            painter.drawLine(QPointF(*w2d(*a)), QPointF(*w2d(*b)))
        painter.restore()

    def _draw_marks(self, painter):
        """Samplers, counts, notes and slices: Photoshop's Extras."""
        from PySide6.QtCore import QPointF
        doc = self.ctl.doc
        if not (doc.samplers or doc.counts or doc.notes or doc.slices):
            return
        w2d = self.view.doc_to_widget
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        for i, (name, (x, y, w, h)) in enumerate(doc.slices):
            pts = [QPointF(*w2d(px, py)) for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
            painter.setPen(QPen(QColor(40, 140, 255), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(pts)
            tag = QRectF(pts[0].x() + 1, pts[0].y() + 1, 22, 13)
            painter.fillRect(tag, QColor(40, 140, 255))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(tag, Qt.AlignCenter, "%02d" % (i + 1))
        for i, (x, y) in enumerate(doc.samplers):
            wx, wy = w2d(x + 0.5, y + 0.5)
            for pen in (QPen(QColor(255, 255, 255), 3.0), QPen(QColor(0, 0, 0), 1.0)):
                painter.setPen(pen)
                painter.drawEllipse(QPointF(wx, wy), 5, 5)
                painter.drawLine(QPointF(wx - 9, wy), QPointF(wx + 9, wy))
                painter.drawLine(QPointF(wx, wy - 9), QPointF(wx, wy + 9))
            painter.drawText(QPointF(wx + 7, wy + 14), str(i + 1))
        for i, (x, y) in enumerate(doc.counts):
            wx, wy = w2d(x, y)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 160, 255))
            painter.drawEllipse(QPointF(wx, wy), 3, 3)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(QPointF(wx + 5, wy - 4), str(i + 1))
            painter.setPen(QColor(0, 160, 255))
            painter.drawText(QPointF(wx + 4, wy - 5), str(i + 1))
        for i, (x, y, text) in enumerate(doc.notes):
            wx, wy = w2d(x, y)
            box = QRectF(wx - 7, wy - 7, 14, 14)
            painter.setPen(QPen(QColor(120, 90, 0), 1.0))
            painter.setBrush(QColor(255, 220, 60))
            painter.drawRect(box)
            painter.drawLine(QPointF(wx - 4, wy - 2), QPointF(wx + 4, wy - 2))
            painter.drawLine(QPointF(wx - 4, wy + 2), QPointF(wx + 3, wy + 2))
            tip = (text.splitlines() or [""])[0][:32]
            if tip:
                painter.setPen(QColor(60, 45, 0))
                painter.drawText(QPointF(wx + 10, wy + 4), tip)
        painter.restore()

    def centre_on(self, dx, dy):
        """Scroll so a document point is in the middle (the Navigator)."""
        self.user_took_view()
        vp = self.viewport()
        self.view._pin(dx, dy, vp.width() / 2.0, vp.height() / 2.0)
        vp.update()

    def set_rulers(self, on):
        from .panels import RULER, Ruler
        if on and self.rulers is None:
            self.rulers = (Ruler(self, True), Ruler(self, False))
        if self.rulers is None:
            return
        for r in self.rulers:
            r.setVisible(bool(on))
        self.setViewportMargins(RULER if on else 0, RULER if on else 0, 0, 0)
        self._place_rulers()

    def _place_rulers(self):
        if self.rulers is None or not self.rulers[0].isVisible():
            return
        from .panels import RULER
        vp = self.viewport().geometry()
        top, left = self.rulers
        top.setGeometry(vp.x(), vp.y() - RULER, vp.width(), RULER)
        left.setGeometry(vp.x() - RULER, vp.y(), RULER, vp.height())

    def _guide_hit(self, pos):
        """Index of the guide under a viewport point, within 4 screen px."""
        if not self.show_guides or self.ctl.doc is None or self.view.rotated:
            return None
        best, idx = 4.0, None
        for i, (o, p) in enumerate(self.ctl.doc.guides):
            wx, wy = self.view.doc_to_widget(p, p)
            d = abs((pos.y() - wy) if o == "h" else (pos.x() - wx))
            if d <= best:
                best, idx = d, i
        return idx

    def _moves_guides(self):
        return getattr(self.ctl, "tool_name", "") in ("move_layer", "move_pixels", "move_selection")

    def _draw_paths(self, painter):
        """The active vector path, while a path tool is in hand.

        Drawn from the document every frame rather than through a cached
        image: paths are a few hundred segments at most, and drawing them
        through the view transform keeps them exact at any zoom or rotation.
        Anchors are squares, handles are dots on thin lines, as in GIMP and
        Photoshop. View > Show Path draws the outline with any tool.
        """
        doc = getattr(self.ctl, "doc", None)
        tool = getattr(self.ctl, "tool", None)
        editing = bool(getattr(tool, "shows_paths", False))
        if doc is None or getattr(doc, "path", None) is None:
            return
        if not editing and not getattr(self, "show_path", True):
            return
        from PySide6.QtCore import QPointF
        vpath = doc.path
        w2d = self.view.doc_to_widget
        qp = QPainterPath()
        for sub in vpath.subpaths:
            if not sub.anchors:
                continue
            a0 = sub.anchors[0]
            qp.moveTo(QPointF(*w2d(a0.x, a0.y)))
            for p0, c0, c1, p1 in sub.segments():
                qp.cubicTo(QPointF(*w2d(*c0)), QPointF(*w2d(*c1)), QPointF(*w2d(*p1)))
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2.0))
        painter.drawPath(qp)
        painter.setPen(QPen(QColor(20, 90, 220), 1.0))
        painter.drawPath(qp)
        if editing:
            handle_pen = QPen(QColor(20, 90, 220, 180), 1.0)
            for a in vpath.anchors():
                ax, ay = w2d(a.x, a.y)
                for hx, hy in ((a.ix, a.iy), (a.ox, a.oy)):
                    if (hx, hy) == (a.x, a.y):
                        continue
                    wx, wy = w2d(hx, hy)
                    painter.setPen(handle_pen)
                    painter.drawLine(QPointF(ax, ay), QPointF(wx, wy))
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawEllipse(QPointF(wx, wy), 2.5, 2.5)
                painter.setPen(QPen(QColor(20, 90, 220), 1.0))
                painter.setBrush(QColor(255, 255, 255) if a.smooth else QColor(20, 90, 220))
                painter.drawRect(QRectF(ax - 3, ay - 3, 6, 6))
        painter.restore()

    def _draw_zoom_band(self, painter):
        """The zoom tool's rubber band, if it has one."""
        tool = self._view_tool()
        band = getattr(tool, "band", None) if tool is not None else None
        if not band:
            return
        x, y, w, h = band
        x0, y0 = self.view.doc_to_widget(x, y)
        x1, y1 = self.view.doc_to_widget(x + w, y + h)
        rect = QRect(int(round(x0)), int(round(y0)),
                     max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        painter.save()
        # Composed the same way as the caret, so the band stays visible over
        # whatever it is drawn on top of.
        painter.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.restore()

    # ---- the pointer -----------------------------------------------------

    CURSORS = {
        "cross": Qt.CrossCursor,
        "ibeam": Qt.IBeamCursor,
        "move": Qt.SizeAllCursor,
        "pointing": Qt.PointingHandCursor,
        "arrow": Qt.ArrowCursor,
        "blank": Qt.BlankCursor,
    }

    def apply_tool_cursor(self):
        """Set the pointer for the active tool, per tools.ini.

        A size-based tool hides the system cursor entirely and draws its own
        footprint instead: when the thing you are aiming is a 16-pixel disc,
        an arrow tells you nothing about what is about to change. Tools with
        no footprint keep a real cursor, because an invisible pointer with
        nothing drawn in its place is just a lost pointer.
        """
        cursor, footprint = self._appearance()
        if cursor == "blank" and footprint == "none":
            cursor = "cross"        # never leave the pointer invisible
        self.viewport().setCursor(self.CURSORS.get(cursor, Qt.CrossCursor))
        self.viewport().update()

    def _appearance(self):
        name = getattr(self.ctl, "tool_name", None)
        if not name:
            return ("cross", "none")
        return self.ctl.tools.appearance(name)

    def _footprint_size(self):
        """The dab diameter in document pixels, or None if not size-based."""
        tool = self.ctl.tool
        if tool is None:
            return None
        try:
            return max(1.0, float(tool.option("size", 1)))
        except (TypeError, ValueError):
            return None

    def _footprint_rect(self):
        """Where the footprint outline sits, in viewport coordinates.

        Built from the SAME floor-and-centre arithmetic the brush uses to
        place a dab, so the outline covers the pixels that would actually
        change rather than an approximation of them.
        """
        if self._hover is None:
            return None
        _cursor, shape = self._appearance()
        if shape == "none":
            return None
        dx, dy = self._hover

        if shape == "point":
            x0, y0, x1, y1 = math.floor(dx), math.floor(dy), 0, 0
            x1, y1 = x0 + 1, y0 + 1
        else:
            size = self._footprint_size()
            if size is None:
                return None
            diameter = max(1, int(round(size)))
            x0 = math.floor(dx - diameter / 2.0)
            y0 = math.floor(dy - diameter / 2.0)
            x1, y1 = x0 + diameter, y0 + diameter

        ax, ay = self.view.doc_to_widget(x0, y0)
        bx, by = self.view.doc_to_widget(x1, y1)
        return QRect(int(round(ax)), int(round(ay)),
                     max(1, int(round(bx - ax))), max(1, int(round(by - ay))))

    def _draw_footprint(self, painter):
        rect = self._footprint_rect()
        if rect is None or self._drawing and not self._show_while_drawing:
            return
        _cursor, shape = self._appearance()

        # Drawn twice, black over white, so the outline stays visible on any
        # colour underneath -- the same problem the caret solves with XOR,
        # but an outline needs to keep its shape, which XOR does not.
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setBrush(Qt.NoBrush)
        for colour, inset in ((QColor(255, 255, 255, 200), 0),
                              (QColor(0, 0, 0, 200), 1)):
            painter.setPen(QPen(colour, 1))
            box = rect.adjusted(inset, inset, -inset, -inset)
            if box.width() <= 0 or box.height() <= 0:
                break
            if shape == "circle" and box.width() > 3:
                painter.drawEllipse(box)
            else:
                painter.drawRect(box)
        painter.restore()

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

    def _view_transform(self):
        return QTransform(*self.view.affine())

    def _paint_rotated(self, painter):
        """The canvas turned by the view's rotation (Photoshop's Rotate View).

        The image is drawn through the doc->widget transform; the checkerboard
        is drawn in WIDGET space under the rotated outline, so its squares
        stay screen-sized and upright instead of zooming and turning with the
        image.
        """
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPolygonF
        w, h = self.view.doc_w, self.view.doc_h
        corners = QPolygonF([QPointF(*self.view.doc_to_widget(x, y))
                             for x, y in ((0, 0), (w, 0), (w, h), (0, h))])
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._checker_brush())
        painter.drawPolygon(corners)
        painter.setTransform(self._view_transform())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, not self.view.use_nearest)
        painter.drawImage(QRectF(0, 0, w, h), self.surface.image, QRectF(0, 0, w, h))
        if self.view.percent >= (self.cfg.get("Canvas", "PixelGridAbove") or 800):
            pen = QPen(QColor(*self.cfg.get("Canvas", "GridColour")))
            pen.setCosmetic(True)
            pen.setWidth(0)
            painter.setPen(pen)
            vis = self.view.visible_doc_rect(self.viewport().width(), self.viewport().height())
            if vis is not None:
                for col in range(vis.x, vis.x1 + 1):
                    painter.drawLine(QPointF(col, vis.y), QPointF(col, vis.y1))
                for row in range(vis.y, vis.y1 + 1):
                    painter.drawLine(QPointF(vis.x, row), QPointF(vis.x1, row))
        painter.restore()

    def _selection_path(self):
        """The selection outline as a QPainterPath in document space, cached
        until the selection changes."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPainterPath
        from ..engine.selection import outline_segments
        sel = self.ctl.selection
        key = (id(sel.mask), sel.mask.shape, int(sel.mask[0, 0]), int(sel.mask[-1, -1]),
               sel.bbox)
        if getattr(self, "_sel_key", None) == key:
            return self._sel_path
        path = QPainterPath()
        for x0, y0, x1, y1 in outline_segments(sel.mask):
            path.moveTo(QPointF(x0, y0))
            path.lineTo(QPointF(x1, y1))
        self._sel_key, self._sel_path = key, path
        return path

    def _draw_selection(self, painter, target):
        """Marching ants along the REAL outline of the selection.

        This used to draw the selection's bounding rectangle, so an elliptical
        or lasso selection showed as a box and there was no way to see what
        was actually selected.
        """
        sel = self.ctl.selection
        if sel is None or sel.selects_all():
            if getattr(self, "_ants_timer", None) is not None:
                self._ants_timer.stop()
            return
        if sel.bbox.is_empty:
            return
        path = self._selection_path()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setTransform(self._view_transform())
        painter.setBrush(Qt.NoBrush)
        white = QPen(QColor(255, 255, 255))
        white.setCosmetic(True)
        white.setWidth(0)
        painter.setPen(white)
        painter.drawPath(path)
        black = QPen(QColor(0, 0, 0))
        black.setCosmetic(True)
        black.setWidth(0)
        black.setDashPattern([4.0, 4.0])
        black.setDashOffset(getattr(self, "_ants_phase", 0))
        painter.setPen(black)
        painter.drawPath(path)
        painter.restore()
        if getattr(self, "_ants_timer", None) is None:
            self._ants_timer = QTimer(self)
            self._ants_timer.timeout.connect(self._march)
        if not self._ants_timer.isActive():
            self._ants_timer.start(150)

    def _march(self):
        sel = self.ctl.selection
        if sel is None or sel.selects_all() or not self.isVisible():
            self._ants_timer.stop()
            return
        self._ants_phase = (getattr(self, "_ants_phase", 0) + 1) % 8
        x, y, w, h = self.view.doc_to_widget_rect(sel.bbox)
        self.viewport().update(QRect(int(x) - 2, int(y) - 2, int(w) + 5, int(h) + 5))

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
        tool = self._text_tool()
        if tool is not None and event.button() == Qt.LeftButton:
            dx, dy = self._doc_pos(event.position())
            box = tool.caret_rect(self.ctl._context())
            inside = box is not None and box.inflated(
                max(8, int(tool.option("size", 24)))).contains(int(dx), int(dy))
            if inside:
                # Reposition within the text being edited.
                tool.caret_from_point(self.ctl._context(), dx, dy)
                self._start_caret()
                self.viewport().update()
                return
            # Clicking away finishes the edit, then falls through so the
            # click also starts the next one where the user pointed.
            self.commit_text()

        self.setFocus(Qt.MouseFocusReason)
        # Space held turns any tool into the hand, as in Paint.NET, GIMP and
        # Photoshop. The left button then pans exactly like the middle one.
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and getattr(self, "_space_pan", False)):
            self.user_took_view()
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        view_tool = self._view_tool()
        if view_tool is not None:
            dx, dy = self._doc_pos(event.position())
            self._view_dragging = True
            self._apply_view(view_tool.begin(self.ctl._context(),
                                             self._tool_event(event, dx, dy)))
            return

        dx, dy = self._doc_pos(event.position())
        button = 2 if event.button() == Qt.RightButton else 1
        # Guides move with the Move tools, as in Photoshop and GIMP.
        if event.button() == Qt.LeftButton and self._moves_guides():
            hit = self._guide_hit(event.position())
            if hit is not None:
                self._guide_drag = (hit, self.ctl.doc.guides[hit][0])
                self.guide_preview = self.ctl.doc.guides[hit]
                return
        # "On a point" means a few SCREEN pixels, whatever the zoom.
        self.ctl.pick_radius = 6.0 / max(self.view.scale, 1e-6)
        self._drawing = True
        # Tools that keep working while the pointer rests (the airbrush) get
        # ticks, so holding still builds paint up instead of doing nothing.
        if getattr(self.ctl.tool, "continuous", False):
            if not hasattr(self, "_tick_timer"):
                self._tick_timer = QTimer(self)
                self._tick_timer.timeout.connect(self._tick)
            self._tick_timer.start(40)
        self.ctl.begin_stroke(dx, dy, 1.0, self._mods(event), button)
        self.refresh()
        if self._text_tool() is not None:
            self._start_caret()

    def mouseMoveEvent(self, event):
        pos = event.position()
        dx, dy = self._doc_pos(pos)
        self.cursor_moved.emit(dx, dy)
        self._move_hover(dx, dy)
        if self.rulers is not None:
            self.rulers[0].pointer, self.rulers[1].pointer = pos.x(), pos.y()
        if self._guide_drag is not None:
            orient = self._guide_drag[1]
            self.guide_preview = (orient, round(dy) if orient == "h" else round(dx))
            self.viewport().update()
            return
        if self._moves_guides() and not self._drawing and not self._panning:
            hit = self._guide_hit(pos)
            if hit is not None:
                o = self.ctl.doc.guides[hit][0]
                self.viewport().setCursor(Qt.SplitVCursor if o == "h" else Qt.SplitHCursor)
                self._on_guide = True
            elif getattr(self, "_on_guide", False):
                self._on_guide = False
                self.apply_tool_cursor()
        if self._panning:
            point = pos.toPoint()
            delta = point - self._pan_origin
            self._pan_origin = point
            self.view.pan_by(delta.x(), delta.y())
            self.viewport().update()
            return
        if self._view_dragging:
            tool = self._view_tool()
            if tool is not None:
                self._apply_view(tool.motion(self.ctl._context(),
                                             self._tool_event(event, dx, dy)))
                self.viewport().update()
            return
        if self._drawing:
            self.ctl.motion_stroke(dx, dy, 1.0, self._mods(event))
            self.refresh()
            if hasattr(self.ctl.tool, "overlay"):
                self.viewport().update()
        elif self.ctl.tool is not None and self.ctl.tool.active:
            if self.ctl.hover_tool(dx, dy):
                self.viewport().update()

    def mouseDoubleClickEvent(self, event):
        # Finishes a click-by-click tool (polygonal lasso, scissors). The
        # press and release of the double-click have already been delivered.
        tool = self.ctl.tool
        if tool is not None and tool.active and hasattr(tool, "finish"):
            self.ctl.finish_tool()
            self.refresh()
            self.viewport().update()
            return
        super().mouseDoubleClickEvent(event)

    def _tick(self):
        if not self._drawing:
            self._tick_timer.stop()
            return
        if self.ctl.tick_stroke() is not None:
            self.refresh()

    def mouseReleaseEvent(self, event):
        if self._guide_drag is not None:
            index, orient = self._guide_drag
            self._guide_drag = None
            preview, self.guide_preview = self.guide_preview, None
            inside = self.viewport().rect().contains(event.position().toPoint())
            if not inside:
                self.ctl.remove_guide(index)          # dragged back off the canvas
            elif preview is not None:
                self.ctl.move_guide(index, preview[1])
            self.viewport().update()
            return
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            if getattr(self, "_space_pan", False):
                self.viewport().setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
                self.apply_tool_cursor()
            return
        if self._view_dragging:
            self._view_dragging = False
            tool = self._view_tool()
            if tool is not None:
                dx, dy = self._doc_pos(event.position())
                self._apply_view(tool.end(self.ctl._context(),
                                          self._tool_event(event, dx, dy)))
            self.viewport().update()
            return
        if self._drawing:
            if hasattr(self, "_tick_timer"):
                self._tick_timer.stop()
            dx, dy = self._doc_pos(event.position())
            self._drawing = False
            # Inpainting tools do seconds of work on release; say so.
            slow = bool(getattr(self.ctl.tool, "slow_commit", False))
            if slow:
                from PySide6.QtWidgets import QApplication
                QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.ctl.end_stroke(dx, dy, 1.0, self._mods(event))
            finally:
                if slow:
                    QApplication.restoreOverrideCursor()
            self.refresh()
            # A text tool stays active after mouse-up so typing can begin.
            if self._text_tool() is not None:
                self._start_caret()

    def wheelEvent(self, event):
        self.user_took_view()
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

    def event(self, event):
        # Qt offers every keystroke to the window's shortcuts BEFORE the
        # focused widget sees it. Accepting ShortcutOverride claims the key
        # back, which is how a text box gets to type a "P" instead of
        # switching to the pencil. This is the sanctioned mechanism; checking
        # inside each shortcut's handler would be too late, the key would be
        # consumed already.
        if event.type() == QEvent.ShortcutOverride:
            if self._text_tool() is not None or self._drawing:
                event.accept()
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        # Text editing owns the keyboard while a text tool is mid-edit.
        tool = self._text_tool()
        if tool is not None and self._handle_text_key(tool, event):
            return

        if event.key() == Qt.Key_Space and not event.isAutoRepeat() and not self._drawing:
            self._space_pan = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return

        if event.key() == Qt.Key_Escape and (self._drawing or (
                self.ctl.tool is not None and self.ctl.tool.active)):
            self._drawing = False
            self.ctl.cancel_stroke()
            self.refresh()
            self.viewport().update()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.ctl.tool is not None \
                and self.ctl.tool.active and hasattr(self.ctl.tool, "finish"):
            self.ctl.finish_tool()
            self.refresh()
            self.viewport().update()
            return
        # Paint.NET's two selection keys: Delete clears, Backspace fills
        # with the primary colour. Both are undoable edits on the controller.
        if event.key() == Qt.Key_Delete and not event.modifiers():
            self.ctl.erase_selection()
            self.refresh()
            return
        if event.key() == Qt.Key_Backspace and not event.modifiers():
            self.ctl.fill_selection()
            self.refresh()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            if not self._panning:
                self.apply_tool_cursor()
            return
        super().keyReleaseEvent(event)

    def _handle_text_key(self, tool, event):
        """Map a key onto the tool's edit operations. True if consumed.

        Deliberately thin: every operation already exists on the tool and is
        tested headlessly, so this maps keys and nothing more.
        """
        ctx = self.ctl._context()
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)

        if key == Qt.Key_Escape:
            self.cancel_text()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if ctrl:
                # Ctrl+Enter finishes; plain Enter is a newline, because
                # this is a multi-line text box.
                self.commit_text()
            else:
                tool.insert(ctx, "\n")
                self._after_text_edit()
            return True
        if key == Qt.Key_Backspace:
            tool.backspace(ctx)
            self._after_text_edit()
            return True
        if key == Qt.Key_Delete:
            tool.delete(ctx)
            self._after_text_edit()
            return True
        if key in (Qt.Key_Left, Qt.Key_Right):
            tool.move_caret(-1 if key == Qt.Key_Left else 1)
            self._after_caret_move()
            return True
        if key in (Qt.Key_Up, Qt.Key_Down):
            tool.caret_line(-1 if key == Qt.Key_Up else 1)
            self._after_caret_move()
            return True
        if key == Qt.Key_Home:
            tool.caret_home()
            self._after_caret_move()
            return True
        if key == Qt.Key_End:
            tool.caret_end()
            self._after_caret_move()
            return True

        text = event.text()
        if text and text.isprintable():
            tool.insert(ctx, text)
            self._after_text_edit()
            return True
        return False

    def _after_text_edit(self):
        self.refresh()
        self._start_caret()
        self.viewport().update()

    def _after_caret_move(self):
        self._start_caret()
        self.viewport().update()

    def commit_text(self):
        """Finish a text edit and push its history entry."""
        tool = self._text_tool()
        if tool is None:
            return None
        # Through the controller, so a type mask's selection is recorded
        # (and undoable) like any other selection change.
        cmd = self.ctl.finish_tool()
        self._stop_caret()
        self.ctl.emit("history.changed")
        self.refresh()
        self.viewport().update()
        return cmd

    def cancel_text(self):
        tool = self._text_tool()
        if tool is None:
            return None
        rect = tool.cancel(self.ctl._context())
        self._stop_caret()
        self.refresh()
        self.viewport().update()
        return rect

    def leaveEvent(self, event):
        """The footprint must not be left stranded when the pointer goes."""
        self._move_hover(None, None)
        super().leaveEvent(event)

    def enterEvent(self, event):
        self.apply_tool_cursor()
        super().enterEvent(event)

    def _move_hover(self, dx, dy):
        """Update the hover position, repainting only what the move touched."""
        old = self._footprint_rect()
        self._hover = None if dx is None else (dx, dy)
        new = self._footprint_rect()
        if old is None and new is None:
            return
        area = new if old is None else (old if new is None else old.united(new))
        # Two pixels of slack: the outline is drawn one inside the rect and
        # antialiasing off still leaves the pen straddling the boundary.
        self.viewport().update(area.adjusted(-2, -2, 2, 2))

    # Below this the viewport is still being laid out, not a real view.
    FIT_MIN_VIEWPORT = 64

    def _fit_if_real(self):
        vp = self.viewport()
        if not getattr(self, "_needs_fit", False):
            return False
        if vp.width() < self.FIT_MIN_VIEWPORT or vp.height() < self.FIT_MIN_VIEWPORT:
            return False
        self.view.fit(vp.width(), vp.height(), margin=16)
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        vp.update()
        return True

    def user_took_view(self):
        """The user zoomed or panned: stop auto-fitting on resize."""
        self._needs_fit = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_rulers()
        self._fit_if_real()
        self.viewport().update()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_if_real()

    def viewportEvent(self, event):
        # The VIEWPORT's own resize, not the scroll area's: Qt lays the
        # viewport out in a separate pass, so on the scroll area's resize
        # event the viewport can still be at its old size, and fitting to
        # that stale size was timing-dependent -- it passed alone and could
        # fail under load.
        if event.type() == QEvent.Resize:
            self._fit_if_real()
        return super().viewportEvent(event)

    # ---- view commands ---------------------------------------------------

    def zoom_to(self, value):
        self.user_took_view()
        self.view.zoom = Fraction(value)
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        self.viewport().update()

    def zoom_fit(self):
        # Explicitly asking for fit puts the view back into fit mode.
        self._needs_fit = True
        if not self._fit_if_real():
            self.view.fit(max(1, self.viewport().width()),
                          max(1, self.viewport().height()), margin=16)
            self.zoom_changed.emit(format_zoom(self.view.zoom))
            self.viewport().update()

    def zoom_step(self, direction):
        """Zoom in (+1) or out (-1) about the centre of the view.

        A method on the canvas, not a menu binding to view.zoom_in: bind()
        REPLACES self.view, so a menu action bound to the old ViewState's
        method kept zooming a view nobody could see.
        """
        self.user_took_view()
        vp = self.viewport()
        anchor = (vp.width() / 2.0, vp.height() / 2.0)
        if direction > 0:
            self.view.zoom_in(anchor)
        else:
            self.view.zoom_out(anchor)
        self.zoom_changed.emit(format_zoom(self.view.zoom))
        vp.update()

    def rotate_view(self, degrees=None):
        """Turn the view by `degrees`, or back to upright when None."""
        import math as _m
        vp = self.viewport()
        anchor = (vp.width() / 2.0, vp.height() / 2.0)
        if degrees is None:
            self.view.set_rotation(0.0, anchor)
        else:
            self.view.rotate_by(_m.radians(degrees), anchor)
        self.user_took_view()
        vp.update()

    def zoom_actual(self):
        self.zoom_to(Fraction(1, 1))
        self.view.center(self.viewport().width(), self.viewport().height())
        self.viewport().update()

    def sizeHint(self):
        return QSize(800, 600)
