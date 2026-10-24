# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tool and shape icons, drawn rather than shipped.

Every icon is a few QPainter calls in a 24x24 design grid, rendered at the
display's real pixel density -- so they are crisp on any screen, follow the
theme's text colour in light and dark modes, and need no asset pipeline.

They are placeholders in the honest sense: a PNG or SVG named after the
icon in data/icons/ (or ~/.local/share/ochre/icons/) replaces the drawn one
with no code change, which is how a proper icon set gets dropped in later.

Shape icons are not drawn by hand at all: they are the catalog's own rings
(engine/shapelib.py) traced into a QPainterPath, so the chooser cannot show
a shape the canvas would draw differently.
"""

import math
import os

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QLinearGradient,
                           QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import QApplication

ACCENT = QColor(205, 120, 45)          # ochre
ACCENT_SOFT = QColor(205, 120, 45, 90)
SELECT = QColor(70, 130, 200)

_SEARCH_DIRS = []


def search_dirs():
    if not _SEARCH_DIRS:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _SEARCH_DIRS.extend([
            os.path.join(os.environ.get("XDG_DATA_HOME",
                                        os.path.expanduser("~/.local/share")), "ochre", "icons"),
            os.path.join(here, "data", "icons"),
        ])
    return _SEARCH_DIRS


def _file_icon(name):
    for d in search_dirs():
        for ext in (".svg", ".png"):
            path = os.path.join(d, name + ext)
            if os.path.isfile(path):
                return QIcon(path)
    return None


def _fg():
    app = QApplication.instance()
    if app is None:
        return QColor(40, 40, 40)
    return app.palette().color(QPalette.WindowText)


class G:
    """A painter in a 24-unit design grid, with the icon set's house style."""

    def __init__(self, painter, size, fg):
        self.p = painter
        self.fg = fg
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.scale(size / 24.0, size / 24.0)

    def _pen(self, w=1.6, color=None, dash=False):
        pen = QPen(color or self.fg, w)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        if dash:
            pen.setDashPattern([2.0, 1.6])
            pen.setCapStyle(Qt.FlatCap)
        return pen

    def _apply(self, fill, w, color, dash):
        self.p.setPen(self._pen(w, color, dash) if w else Qt.NoPen)
        self.p.setBrush(QBrush(fill) if fill is not None else Qt.NoBrush)

    def line(self, *pts, w=1.6, color=None, dash=False):
        self._apply(None, w, color, dash)
        self.p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))

    def poly(self, pts, fill=None, w=1.6, color=None, dash=False):
        self._apply(fill, w, color, dash)
        self.p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))

    def rect(self, x, y, w_, h_, fill=None, w=1.6, color=None, dash=False, r=0.0):
        self._apply(fill, w, color, dash)
        if r:
            self.p.drawRoundedRect(QRectF(x, y, w_, h_), r, r)
        else:
            self.p.drawRect(QRectF(x, y, w_, h_))

    def ellipse(self, cx, cy, rx, ry, fill=None, w=1.6, color=None, dash=False):
        self._apply(fill, w, color, dash)
        self.p.drawEllipse(QPointF(cx, cy), rx, ry)

    def path(self, path, fill=None, w=1.6, color=None, dash=False):
        self._apply(fill, w, color, dash)
        self.p.drawPath(path)

    def curve(self, start, c1, c2, end, w=1.6, color=None, dash=False):
        path = QPainterPath(QPointF(*start))
        path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*end))
        self.path(path, None, w, color, dash)

    def text(self, x, y, s, size=14, bold=True, color=None, family=None):
        font = QFont(family or "DejaVu Sans")
        font.setPixelSize(int(size))
        font.setBold(bold)
        self.p.setFont(font)
        self.p.setPen(color or self.fg)
        self.p.drawText(QPointF(x, y), s)

    def star(self, cx, cy, r, n=4, inner=0.35, fill=None, w=1.2):
        pts = []
        for i in range(n * 2):
            a = -math.pi / 2 + math.pi * i / n
            s = r if i % 2 == 0 else r * inner
            pts.append((cx + math.cos(a) * s, cy + math.sin(a) * s))
        self.poly(pts, fill=fill if fill is not None else ACCENT, w=w)


# ---- reusable parts ------------------------------------------------------------

def _brush(g, tip=ACCENT):
    g.line((19, 3), (11, 13), w=2.2)
    g.path(_blob(), fill=tip, w=1.2)


def _blob():
    p = QPainterPath(QPointF(11.5, 12.5))
    p.cubicTo(QPointF(8, 11), QPointF(5, 14), QPointF(5.5, 17))
    p.cubicTo(QPointF(5.8, 19.5), QPointF(4, 20.5), QPointF(3, 21))
    p.cubicTo(QPointF(8, 21.5), QPointF(12.5, 19), QPointF(13, 15))
    p.closeSubpath()
    return p


def _pencil(g, fill=ACCENT):
    g.poly([(17, 3), (21, 7), (9, 19), (5, 15)], fill=fill)
    g.poly([(5, 15), (9, 19), (3.5, 20.5)], fill=g.fg, w=1.0)


def _drop(g, cx, cy, s=1.0, fill=ACCENT):
    p = QPainterPath(QPointF(cx, cy - 6 * s))
    p.cubicTo(QPointF(cx + 5 * s, cy), QPointF(cx + 4 * s, cy + 5 * s), QPointF(cx, cy + 5 * s))
    p.cubicTo(QPointF(cx - 4 * s, cy + 5 * s), QPointF(cx - 5 * s, cy), QPointF(cx, cy - 6 * s))
    g.path(p, fill=fill, w=1.3)


def _marquee(g, x=3, y=4, w=18, h=16, ellipse=False):
    if ellipse:
        g.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, dash=True, w=1.5)
    else:
        g.rect(x, y, w, h, dash=True, w=1.5)


def _arrow4(g, cx=12, cy=12, r=9, w=1.6):
    g.line((cx - r, cy), (cx + r, cy), w=w)
    g.line((cx, cy - r), (cx, cy + r), w=w)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        tx, ty = cx + dx * r, cy + dy * r
        px, py = -dy, dx
        g.poly([(tx + dx * 0.5, ty + dy * 0.5), (tx - dx * 3 + px * 2.5, ty - dy * 3 + py * 2.5),
                (tx - dx * 3 - px * 2.5, ty - dy * 3 - py * 2.5)], fill=g.fg, w=0.8)


def _magnifier(g, inner=None):
    g.ellipse(10, 10, 6.5, 6.5, fill=inner, w=2.0)
    g.line((15, 15), (21, 21), w=3.0)


def _stamp(g, top=ACCENT):
    g.rect(9.5, 3, 5, 6, fill=top, r=2)
    g.line((12, 9), (12, 13), w=2.4)
    g.rect(4.5, 13, 15, 4.5, fill=g.fg, w=1.2, r=1)
    g.line((4, 20), (20, 20), w=1.4)


def _cursor(g, x=6, y=3, fill=None):
    g.poly([(x, y), (x, y + 15), (x + 4, y + 11.5), (x + 7, y + 17.5),
            (x + 9.3, y + 16.4), (x + 6.5, y + 10.5), (x + 11.5, y + 10.5)],
           fill=fill if fill is not None else g.fg, w=1.0)


def _bubble_bezier(g):
    g.curve((3, 19), (6, 4), (14, 22), (21, 5), w=2.0)
    for x, y in ((3, 19), (21, 5)):
        g.rect(x - 1.8, y - 1.8, 3.6, 3.6, fill=ACCENT, w=1.0)


# ---- the icons -------------------------------------------------------------------

def i_select_rect(g):      _marquee(g)
def i_select_ellipse(g):   _marquee(g, ellipse=True)


def i_select_row(g):
    g.rect(2, 10, 20, 4, dash=True, w=1.4)
    g.line((2, 6), (22, 6), w=0.6, color=QColor(128, 128, 128))
    g.line((2, 18), (22, 18), w=0.6, color=QColor(128, 128, 128))


def i_select_col(g):
    g.rect(10, 2, 4, 20, dash=True, w=1.4)
    g.line((6, 2), (6, 22), w=0.6, color=QColor(128, 128, 128))
    g.line((18, 2), (18, 22), w=0.6, color=QColor(128, 128, 128))


def i_lasso(g):
    g.ellipse(12, 9, 8.5, 5.5, w=1.6)
    g.curve((6, 13), (5, 17), (9, 18), (8, 22), w=1.6)


def i_polygon_lasso(g):
    g.poly([(3, 9), (12, 3), (21, 8), (17, 17), (7, 15)], dash=True, w=1.5)
    for x, y in ((3, 9), (12, 3), (21, 8), (17, 17), (7, 15)):
        g.rect(x - 1.3, y - 1.3, 2.6, 2.6, fill=ACCENT, w=0.8)


def i_magnetic_lasso(g):
    g.ellipse(11, 9, 8, 5.5, dash=True, w=1.5)
    g.path(_horseshoe(15, 15), fill=None, w=2.4, color=QColor(200, 50, 50))


def _horseshoe(cx, cy):
    p = QPainterPath(QPointF(cx - 4, cy - 1))
    p.lineTo(cx - 4, cy + 3)
    p.arcTo(QRectF(cx - 4, cy - 1, 8, 8), 180, 180)
    p.lineTo(cx + 4, cy - 1)
    return p


def i_scissors(g):
    g.ellipse(6.5, 17, 3.2, 3.2, w=1.6)
    g.ellipse(17.5, 17, 3.2, 3.2, w=1.6)
    g.line((8.5, 14.5), (17, 3), w=1.8)
    g.line((15.5, 14.5), (7, 3), w=1.8)


def i_magic_wand(g):
    g.line((4, 20), (15, 9), w=2.6)
    g.star(17.5, 6.5, 5.5, 4, 0.32)
    g.star(7, 6, 2.3, 4, 0.3, w=0.8)


def i_select_color(g):
    for i, c in enumerate((QColor(220, 60, 60), QColor(60, 170, 80), QColor(60, 110, 220))):
        g.rect(3 + i * 6, 4, 5, 5, fill=c, w=0.8)
    g.rect(3, 12, 17, 8, dash=True, w=1.4)
    g.rect(6, 14.5, 4, 3, fill=QColor(220, 60, 60), w=0)
    g.rect(13, 14.5, 4, 3, fill=QColor(220, 60, 60), w=0)


def i_quick_select(g):
    g.ellipse(11, 12, 8, 7, dash=True, w=1.4)
    g.ellipse(15, 14, 4, 4, fill=ACCENT_SOFT, w=1.2)
    g.line((17.5, 17), (21, 21), w=2.2)


def i_foreground_select(g):
    g.ellipse(12, 8, 3.5, 3.5, fill=ACCENT, w=1.2)
    g.path(_shoulders(), fill=ACCENT, w=1.2)
    g.rect(3, 3, 18, 18, dash=True, w=1.2)


def _shoulders():
    p = QPainterPath(QPointF(5.5, 20))
    p.cubicTo(QPointF(6, 14), QPointF(9, 12.5), QPointF(12, 12.5))
    p.cubicTo(QPointF(15, 12.5), QPointF(18, 14), QPointF(18.5, 20))
    p.closeSubpath()
    return p


def i_object_select(g):
    g.rect(3, 3, 18, 18, dash=True, w=1.2)
    g.poly([(7, 18), (10, 9), (14, 13), (17, 7), (19, 18)], fill=ACCENT_SOFT, w=1.4)


def i_quick_mask(g):
    g.rect(3, 4, 18, 16, fill=QColor(220, 70, 70, 110), w=1.4)
    g.ellipse(12, 12, 5, 4.5, fill=QColor(255, 255, 255), w=1.4)


def i_move_pixels(g):
    _arrow4(g)
    g.rect(9.5, 9.5, 5, 5, fill=ACCENT, w=1.0)


def i_move_selection(g):
    _arrow4(g)
    g.rect(8.5, 8.5, 7, 7, dash=True, w=1.2)


def i_align(g):
    g.line((4, 3), (4, 21), w=1.6)
    g.rect(6, 5, 13, 5, fill=ACCENT, w=1.2)
    g.rect(6, 14, 8, 5, fill=ACCENT_SOFT, w=1.2)


def i_crop(g):
    g.line((7, 2), (7, 17), (22, 17), w=2.2)
    g.line((2, 7), (17, 7), (17, 22), w=2.2)


def i_perspective_crop(g):
    g.poly([(6, 5), (18, 3), (21, 20), (3, 19)], dash=True, w=1.4)
    g.line((8, 2), (6, 5), (4, 9), w=2.0)
    g.line((16, 22), (21, 20), (22, 16), w=2.0)


def i_slice(g):
    g.line((3, 20), (16, 5), w=2.4)
    g.poly([(16, 5), (21, 3), (19, 8)], fill=ACCENT, w=1.0)
    g.rect(3, 3, 8, 8, dash=True, w=1.0)


def i_artboard(g):
    g.rect(6, 6, 15, 15, fill=QColor(255, 255, 255), w=1.2)
    g.line((3, 6), (21, 6), w=1.0)
    g.line((6, 3), (6, 21), w=1.0)
    g.rect(9, 9, 9, 4, fill=ACCENT, w=0.8)


def i_vanishing_point(g):
    g.poly([(3, 20), (21, 20), (16, 7), (8, 7)], w=1.3)
    g.line((12, 7), (12, 20), w=0.9)
    g.line((5.5, 13.5), (18.5, 13.5), w=0.9)
    g.line((8, 7), (12, 2), w=0.8, color=ACCENT)
    g.line((16, 7), (12, 2), w=0.8, color=ACCENT)


def i_perspective_warp(g):
    g.poly([(3, 18), (11, 15), (11, 5), (3, 7)], w=1.2)
    g.poly([(11, 15), (21, 19), (21, 4), (11, 5)], w=1.2)
    for x, y in ((3, 18), (11, 15), (11, 5), (3, 7), (21, 19), (21, 4)):
        g.rect(x - 1.4, y - 1.4, 2.8, 2.8, fill=ACCENT, w=0.7)


def i_blur_gallery(g):
    g.rect(3, 3, 18, 18, w=1.0, dash=True)
    for x, y, r in ((8, 9, 3.2), (16, 15, 2.2)):
        g.rect(x - r, y - r, 2 * r, 2 * r, fill=ACCENT, w=0.8, r=r)


def i_frame(g):
    g.rect(4, 4, 16, 16, w=1.4)
    g.line((4, 4), (20, 20), w=1.0)
    g.line((20, 4), (4, 20), w=1.0)


def i_transform(g):
    g.rect(5, 5, 14, 14, w=1.4)
    for x, y in ((5, 5), (19, 5), (5, 19), (19, 19), (12, 5), (12, 19), (5, 12), (19, 12)):
        g.rect(x - 1.6, y - 1.6, 3.2, 3.2, fill=ACCENT, w=0.8)


def i_rotate(g):
    path = QPainterPath()
    path.arcMoveTo(QRectF(4, 4, 16, 16), 30)
    path.arcTo(QRectF(4, 4, 16, 16), 30, 280)
    g.path(path, w=2.0)
    g.poly([(18.9, 7.5), (20.5, 2.5), (15, 4.5)], fill=g.fg, w=0.8)
    g.rect(9, 9, 6, 6, fill=ACCENT, w=1.0)


def i_scale(g):
    g.rect(3, 9, 12, 12, w=1.4)
    g.rect(3, 3, 18, 18, dash=True, w=1.2)
    g.line((10, 14), (20, 4), w=1.6)
    g.poly([(21, 3), (21, 8.5), (15.5, 3)], fill=g.fg, w=0.8)


def i_shear(g):
    g.poly([(8, 5), (21, 5), (16, 19), (3, 19)], fill=ACCENT_SOFT, w=1.5)
    g.rect(3, 5, 13, 14, dash=True, w=1.0)


def i_perspective(g):
    g.poly([(7, 4), (17, 4), (21, 20), (3, 20)], fill=ACCENT_SOFT, w=1.5)
    g.line((12, 4), (12, 20), w=0.8)
    g.line((5, 12), (19, 12), w=0.8)


def i_transform3d(g):
    g.poly([(4, 8), (13, 4), (20, 8), (11, 12)], fill=ACCENT_SOFT, w=1.3)
    g.poly([(4, 8), (11, 12), (11, 21), (4, 17)], fill=ACCENT, w=1.3)
    g.poly([(11, 12), (20, 8), (20, 17), (11, 21)], w=1.3)


def i_handle_transform(g):
    g.poly([(4, 6), (18, 4), (20, 19), (6, 18)], w=1.3)
    for x, y in ((8, 9), (16, 15)):
        g.ellipse(x, y, 2.4, 2.4, fill=ACCENT, w=1.0)
    g.line((8, 9), (16, 15), w=0.8, dash=True)


def i_flip(g):
    g.line((12, 2), (12, 22), w=1.0, dash=True)
    g.poly([(10, 5), (3, 19), (10, 19)], fill=ACCENT, w=1.3)
    g.poly([(14, 5), (21, 19), (14, 19)], w=1.3)


def i_art_history(g):
    i_history_brush(g)
    g.curve((2, 22), (6, 18), (8, 22), (12, 18), w=1.4, color=ACCENT)


def i_puppet_warp(g):
    # A bent limb with pins: the puppet.
    g.curve((4, 20), (6, 10), (14, 14), (20, 4), w=3.0, color=ACCENT_SOFT)
    for x, y in ((4, 20), (11, 12), (20, 4)):
        g.ellipse(x, y, 2.2, 2.2, fill=ACCENT, w=0.8)


def i_cage(g):
    pts = [(4, 6), (11, 3), (20, 6), (21, 15), (14, 21), (5, 18)]
    g.poly(pts, w=1.3)
    for x, y in pts:
        g.ellipse(x, y, 1.7, 1.7, fill=ACCENT, w=0.8)
    g.ellipse(12, 12, 3.5, 3, fill=ACCENT_SOFT, w=1.0)


def i_warp(g):
    for i in range(4):
        y = 5 + i * 4.5
        g.curve((3, y), (9, y - 3 + i), (15, y + 3 - i), (21, y), w=1.2)
    g.ellipse(12, 12, 3, 3, fill=ACCENT, w=1.0)


def i_eyedropper(g):
    g.line((14, 10), (5, 19), w=2.2)
    g.poly([(13, 7), (17, 11), (19.5, 8.5), (15.5, 4.5)], fill=g.fg, w=1.0)
    g.line((3.5, 20.5), (5, 19), w=1.6, color=ACCENT)


def i_sampler(g):
    i_eyedropper(g)
    g.ellipse(18, 18, 3.4, 3.4, fill=ACCENT, w=1.0)
    g.line((18, 13.5), (18, 22.5), w=0.8)
    g.line((13.5, 18), (22.5, 18), w=0.8)


def i_measure(g):
    g.poly([(2, 17), (17, 2), (22, 7), (7, 22)], fill=ACCENT_SOFT, w=1.3)
    for i in range(1, 6):
        x = 2 + i * 2.6
        y = 17 - i * 2.6
        g.line((x, y), (x + (2 if i % 2 else 1.2), y + (2 if i % 2 else 1.2)), w=1.0)


def i_count(g):
    g.text(2, 12, "1", 10)
    g.text(11, 12, "2", 10)
    g.text(6, 22, "3", 10, color=ACCENT)


def i_note(g):
    g.poly([(4, 3), (20, 3), (20, 15), (14, 21), (4, 21)], fill=QColor(250, 225, 120), w=1.3)
    g.poly([(14, 21), (14, 15), (20, 15)], w=1.0)
    for y in (8, 11.5):
        g.line((7, y), (17, y), w=1.0)


def i_heal(g):
    g.rect(3, 9, 18, 6, fill=ACCENT_SOFT, w=1.3, r=3)
    g.rect(9, 3, 6, 18, fill=ACCENT_SOFT, w=1.3, r=3)
    g.rect(9, 9, 6, 6, fill=ACCENT, w=0)


def i_spot_heal(g):
    i_heal(g)
    g.star(19, 5, 3, 4, 0.3, w=0.7)


def i_patch(g):
    g.path(_patch_path(), fill=ACCENT_SOFT, w=1.4, dash=True)
    g.line((14, 14), (20, 20), w=1.8)
    g.poly([(21, 21), (21, 16), (16, 21)], fill=g.fg, w=0.6)


def _patch_path():
    p = QPainterPath(QPointF(4, 6))
    p.cubicTo(QPointF(8, 2), QPointF(15, 3), QPointF(14, 8))
    p.cubicTo(QPointF(13, 13), QPointF(7, 14), QPointF(4, 6))
    return p


def i_content_move(g):
    _arrow4(g, r=6)
    g.rect(3, 3, 6, 6, dash=True, w=1.0)
    g.rect(15, 15, 6, 6, fill=ACCENT, w=1.0)


def i_remove(g):
    g.rect(3, 13, 12, 6, fill=g.fg, w=0, r=1)
    g.poly([(8, 13), (15, 6), (21, 12), (15, 19), (13, 19)], fill=ACCENT, w=1.2)
    g.star(6, 6, 3, 4, 0.3, w=0.7)


def i_red_eye(g):
    g.ellipse(12, 12, 9, 5.5, w=1.4)
    g.ellipse(12, 12, 3.6, 3.6, fill=QColor(210, 40, 40), w=1.0)
    g.ellipse(12, 12, 1.4, 1.4, fill=g.fg, w=0)


def i_clone(g):    _stamp(g)


def i_pattern_stamp(g):
    _stamp(g, top=None)
    for i in range(3):
        g.rect(5.5 + i * 4.8, 13.7, 2.4, 2.4, fill=ACCENT, w=0)


def i_perspective_clone(g):
    _stamp(g)
    g.poly([(15, 2), (22, 2), (22, 9), (17, 7)], w=0.9, dash=True)


def i_blur(g):
    _drop(g, 12, 12, 1.4, fill=ACCENT_SOFT)


def i_sharpen(g):
    g.poly([(12, 2), (19, 21), (5, 21)], fill=ACCENT_SOFT, w=1.4)


def i_smudge(g):
    g.path(_finger(), fill=ACCENT_SOFT, w=1.3)
    g.curve((3, 20), (7, 15), (10, 21), (14, 17), w=1.4, color=ACCENT)


def _finger():
    p = QPainterPath(QPointF(10, 17))
    p.lineTo(17, 4)
    p.cubicTo(QPointF(19, 2), QPointF(22, 4), QPointF(20.5, 6.5))
    p.lineTo(14, 19)
    p.cubicTo(QPointF(13, 21), QPointF(9, 20), QPointF(10, 17))
    return p


def i_dodge(g):
    g.line((15, 13), (21, 21), w=2.4)
    g.ellipse(10, 9, 6.5, 6.5, fill=QColor(255, 255, 255), w=1.6)


def i_burn(g):
    g.line((6, 21), (11, 13), w=2.4)
    g.path(_burn_hand(), fill=ACCENT_SOFT, w=1.3)


def _burn_hand():
    p = QPainterPath(QPointF(8, 14))
    p.cubicTo(QPointF(5, 9), QPointF(9, 3), QPointF(14, 4))
    p.cubicTo(QPointF(20, 5), QPointF(21, 11), QPointF(17, 14))
    p.closeSubpath()
    return p


def i_sponge(g):
    g.rect(4, 7, 16, 11, fill=QColor(240, 200, 80), w=1.4, r=3)
    for x, y in ((8, 11), (13, 10), (16, 14), (10, 15)):
        g.ellipse(x, y, 1.1, 1.1, fill=g.fg, w=0)


def i_history_brush(g):
    _brush(g)
    path = QPainterPath()
    path.arcMoveTo(QRectF(13, 13, 9, 9), 90)
    path.arcTo(QRectF(13, 13, 9, 9), 90, 270)
    g.path(path, w=1.2)
    g.poly([(17.5, 13), (15.5, 11.5), (15.5, 14.5)], fill=g.fg, w=0.5)


def i_brush(g):    _brush(g)
def i_pencil(g):   _pencil(g)


def i_airbrush(g):
    g.rect(13, 4, 5, 10, fill=ACCENT, w=1.2, r=1.5)
    g.line((15.5, 14), (15.5, 20), w=2.0)
    g.line((12, 7), (6, 4), w=1.2)
    for x, y in ((4, 9), (3, 13), (6, 11), (5, 16), (8, 14), (2, 18)):
        g.ellipse(x, y, 0.8, 0.8, fill=g.fg, w=0)


def i_ink(g):
    g.poly([(12, 2), (17, 10), (13.5, 21), (10.5, 21), (7, 10)], fill=ACCENT, w=1.3)
    g.line((12, 7), (12, 15), w=1.0)
    g.ellipse(12, 15.5, 1.3, 1.3, fill=g.fg, w=0)


def i_mypaint(g):
    _brush(g, tip=QColor(120, 180, 90))
    g.ellipse(18, 18, 3, 3, fill=QColor(120, 180, 90, 120), w=1.0)


def i_recolor(g):
    _brush(g, tip=QColor(60, 140, 220))
    g.rect(15, 14, 6, 6, fill=ACCENT, w=1.0)


def i_mixer(g):
    _brush(g, tip=QColor(150, 90, 200))
    _drop(g, 18, 16, 0.6, fill=QColor(60, 140, 220))


def i_eraser(g):
    g.poly([(10, 4), (21, 15), (14, 21), (3, 10)], fill=QColor(240, 150, 160), w=1.3)
    g.poly([(3, 10), (10, 4), (13.5, 7.5), (6.5, 13.5)], fill=g.fg, w=1.0)
    g.line((12, 21), (21, 21), w=1.0)


def i_background_eraser(g):
    i_eraser(g)
    g.rect(15, 2, 7, 7, dash=True, w=1.0)


def i_magic_eraser(g):
    i_eraser(g)
    g.star(19, 5, 3.5, 4, 0.3, w=0.7)


def i_bucket(g):
    g.poly([(4, 10), (12, 2), (20, 10), (12, 18)], fill=ACCENT_SOFT, w=1.5)
    g.line((4, 10), (20, 10), w=1.2)
    _drop(g, 20, 17, 0.55, fill=ACCENT)


def i_gradient(g):
    grad = QLinearGradient(3, 0, 21, 0)
    grad.setColorAt(0, g.fg)
    grad.setColorAt(1, QColor(255, 255, 255, 0))
    g.p.setPen(g._pen(1.4))
    g.p.setBrush(QBrush(grad))
    g.p.drawRect(QRectF(3, 5, 18, 14))


def i_pen(g):
    g.poly([(12, 2), (19, 12), (15, 19), (9, 19), (5, 12)], fill=None, w=1.5)
    g.line((12, 2), (12, 11), w=1.2)
    g.ellipse(12, 12, 1.4, 1.4, fill=g.fg, w=0)
    g.rect(8, 19, 8, 3, fill=g.fg, w=0)


def _pen_small(g):
    g.poly([(9, 2), (14, 9), (11, 14), (7, 14), (4, 9)], fill=None, w=1.3)
    g.line((9, 2), (9, 8), w=1.0)
    g.rect(6, 14, 6, 2.5, fill=g.fg, w=0)


def i_add_anchor(g):
    _pen_small(g)
    g.line((15, 18), (22, 18), w=2.0)
    g.line((18.5, 14.5), (18.5, 21.5), w=2.0)


def i_delete_anchor(g):
    _pen_small(g)
    g.line((15, 18), (22, 18), w=2.0)


def i_convert_point(g):
    g.line((4, 20), (12, 6), w=1.6)
    g.line((12, 6), (20, 20), w=1.6)
    g.rect(10, 4, 4, 4, fill=ACCENT, w=1.0)


def i_freeform_pen(g):
    i_pen(g)
    g.curve((2, 6), (5, 2), (6, 9), (9, 5), w=1.2, color=ACCENT)


def i_curvature_pen(g):
    i_pen(g)
    g.curve((1, 22), (3, 14), (6, 22), (8, 16), w=1.2, color=ACCENT)


def i_path_select(g):
    _cursor(g)


def i_direct_select(g):
    _cursor(g, fill=QColor(255, 255, 255))


def i_line(g):
    g.line((4, 20), (20, 4), w=2.2)
    for x, y in ((4, 20), (20, 4)):
        g.rect(x - 1.8, y - 1.8, 3.6, 3.6, fill=ACCENT, w=1.0)


def i_curve(g):     _bubble_bezier(g)


def i_shape(g):
    g.rect(3, 3, 11, 11, fill=ACCENT_SOFT, w=1.4)
    g.ellipse(15, 15, 6, 6, fill=ACCENT, w=1.4)


def i_freeform(g):
    p = QPainterPath(QPointF(4, 12))
    p.cubicTo(QPointF(3, 3), QPointF(13, 1), QPointF(16, 6))
    p.cubicTo(QPointF(21, 13), QPointF(19, 21), QPointF(11, 20))
    p.cubicTo(QPointF(6, 19.5), QPointF(5, 16), QPointF(4, 12))
    g.path(p, fill=ACCENT_SOFT, w=1.5)


def i_text(g):            g.text(5, 20, "T", 20)


def i_text_vertical(g):
    g.text(8, 11, "T", 10)
    g.text(8, 21, "T", 10)
    g.line((3, 3), (3, 21), w=1.0)


def i_text_mask(g):
    g.text(5, 20, "T", 20, color=QColor(0, 0, 0, 0))
    font = QFont("DejaVu Sans")
    font.setPixelSize(20)
    font.setBold(True)
    path = QPainterPath()
    path.addText(QPointF(5, 20), font, "T")
    g.path(path, dash=True, w=1.0)


def i_zoom(g):            _magnifier(g)


def i_pan(g):
    for i, (x, top) in enumerate(((7, 6), (10, 3.5), (13, 3), (16, 5))):
        g.rect(x - 1.4, top, 2.8, 11 - top + 3, fill=QColor(255, 255, 255), w=1.2, r=1.4)
    g.rect(5.6, 11, 12, 10, fill=QColor(255, 255, 255), w=1.2, r=4)
    g.line((5.6, 14), (3, 11), w=1.2)


def i_rotate_view(g):
    i_pan(g)
    path = QPainterPath()
    path.arcMoveTo(QRectF(14, 13, 9, 9), 180)
    path.arcTo(QRectF(14, 13, 9, 9), 180, -200)
    g.path(path, w=1.3, color=ACCENT)


def i_gegl(g):
    g.rect(3, 3, 18, 18, w=1.2, r=3)
    g.text(5, 17, "fx", 12)


ICONS = {name[2:]: fn for name, fn in globals().items()
         if name.startswith("i_") and callable(fn)}


def has_drawn(name):
    return name in ICONS


def tool_pixmap(name, size=24, fg=None, flyout=False, dpr=None):
    """A crisp pixmap for an icon name at `size` logical pixels."""
    app = QApplication.instance()
    dpr = dpr or (app.devicePixelRatio() if app is not None else 1.0)
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.transparent)
    painter = QPainter(px)
    try:
        g = G(painter, size, fg or _fg())
        fn = ICONS.get(name)
        if fn is None:
            # Unknown: the tool's initial, so it is at least identifiable.
            g.text(6, 18, (name[:1] or "?").upper(), 15)
        else:
            fn(g)
        if flyout:
            # The corner triangle every toolbox uses to say "there is more
            # under this button".
            g.poly([(22.5, 17), (22.5, 22.5), (17, 22.5)], fill=g.fg, w=0)
    finally:
        painter.end()
    return px


def tool_icon(name, size=24, flyout=False):
    """A QIcon: a user-supplied file if there is one, else the drawn icon."""
    found = _file_icon(name)
    if found is not None and not flyout:
        return found
    icon = QIcon()
    for s in (size, size * 2):
        icon.addPixmap(tool_pixmap(name, s, flyout=flyout, dpr=1.0))
    return icon


def shape_path(ident, box=(2.5, 2.5, 21.5, 21.5)):
    """The catalog shape as a QPainterPath, even-odd, in the 24-unit grid."""
    from ..engine import shapelib
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    for ring in shapelib.rings(ident, box):
        if len(ring) < 3:
            continue
        path.moveTo(QPointF(*ring[0]))
        for x, y in ring[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()
    return path


def shape_icon(ident, size=24):
    icon = QIcon()
    for s in (size, size * 2):
        px = QPixmap(s, s)
        px.fill(Qt.transparent)
        painter = QPainter(px)
        try:
            g = G(painter, s, _fg())
            if ident == "freehand":
                # Not a catalog shape: the lasso's own drawing, scaled.
                painter.scale(s / 24.0, s / 24.0)
                ICONS["lasso"](G(painter, 24, _fg()))
            else:
                g.path(shape_path(ident), fill=QColor(205, 228, 246), w=1.2,
                       color=QColor(60, 110, 170))
        finally:
            painter.end()
        icon.addPixmap(px)
    return icon
