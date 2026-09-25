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
        # Where the pointer is, in document coordinates, or None when it has
        # left the widget. Drives the tool footprint overlay.
        self._hover = None
        self._show_while_drawing = True

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

    # ---- text caret ------------------------------------------------------

    def _text_tool(self):
        """The active text tool, if one is mid-edit."""
        tool = self.ctl.tool
        if tool is None or getattr(tool, "name", "") != "text":
            return None
        return tool if tool.active else None

    def _blink(self):
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
        self._draw_footprint(painter)
        self._draw_caret(painter)
        painter.end()

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
        if self._text_tool() is not None:
            self._start_caret()

    def mouseMoveEvent(self, event):
        pos = event.position()
        dx, dy = self._doc_pos(pos)
        self.cursor_moved.emit(dx, dy)
        self._move_hover(dx, dy)
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
            # A text tool stays active after mouse-up so typing can begin.
            if self._text_tool() is not None:
                self._start_caret()

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
        # Text editing owns the keyboard while a text tool is mid-edit.
        tool = self._text_tool()
        if tool is not None and self._handle_text_key(tool, event):
            return

        if event.key() == Qt.Key_Escape and self._drawing:
            self._drawing = False
            self.ctl.cancel_stroke()
            self.refresh()
            self.viewport().update()
            return
        super().keyPressEvent(event)

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
        cmd = tool.commit_text(self.ctl._context())
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
