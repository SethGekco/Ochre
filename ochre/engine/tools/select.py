# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Selection tools: rectangle, ellipse, lasso, magic wand.

All four produce an antialiased coverage mask and hand it to the selection's
combine step, so the combine mode (replace / union / exclude / intersect /
xor) is shared machinery rather than something each tool reimplements.

The in-progress drag composes with the committed selection for display
without mutating it, which is what makes Shift-drag-to-add preview correctly
and Escape free.
"""

import math

import numpy as np

from .. import accel

from ..geometry import Rect
from ..selection import EXCLUDE, REPLACE, UNION, Selection
from .base import Tool, drag_rect
from . import options as O


class _DragSelectTool(Tool):
    changes_selection = True
    """Shared: remember the committed selection, preview against a copy."""

    wants_stroke = False
    shape = "rect"

    def on_begin(self, ctx, event):
        self._start = event
        self._base = ctx.selection.copy() if ctx.selection is not None else None
        return None

    def _rect(self, event):
        # Modifiers are read live, so toggling Shift mid-drag updates the
        # preview rather than latching whatever was held at mouse-down.
        return drag_rect(self._start, event,
                         square=event.has(1), centre=event.has(4))

    def _apply(self, ctx, event, commit):
        if ctx.selection is None or self._base is None:
            return None
        mode = self.option("mode", REPLACE)
        antialias = bool(self.option("antialias", True))
        working = self._base.copy()
        rect = self._rect(event)
        if self.shape == "ellipse":
            working.select_ellipse(rect, mode, antialias)
        else:
            working.select_rect(rect, mode, antialias)
        if commit:
            ctx.selection.mask = working.mask
            ctx.selection._bbox = None
            ctx.selection.geometry = working.geometry
        return working

    def on_motion(self, ctx, event):
        return self._apply(ctx, event, commit=False)

    def on_end(self, ctx, event):
        result = self._apply(ctx, event, commit=True)
        self._base = None
        return result

    def on_cancel(self, ctx):
        if self._base is not None and ctx.selection is not None:
            ctx.selection.mask = self._base.mask
            ctx.selection._bbox = None
        self._base = None
        return None


class RectSelectTool(_DragSelectTool):
    name = "select_rect"
    label = "Rectangle Select"
    options_spec = (O.selection_mode(), O.antialias(True))
    shape = "rect"


class EllipseSelectTool(_DragSelectTool):
    name = "select_ellipse"
    label = "Ellipse Select"
    options_spec = (O.selection_mode(), O.antialias(True))
    shape = "ellipse"


class LassoTool(Tool):
    changes_selection = True
    """Freehand polygon selection."""

    name = "lasso"
    label = "Lasso Select"
    options_spec = (O.selection_mode(), O.antialias(True))
    wants_stroke = False

    def on_begin(self, ctx, event):
        self._points = [(event.x, event.y)]
        self._base = ctx.selection.copy() if ctx.selection is not None else None
        return None

    def on_motion(self, ctx, event):
        # Skip points that add no detail; a dense freehand path makes the
        # even-odd test needlessly expensive without changing the result.
        if self._points:
            lx, ly = self._points[-1]
            if abs(event.x - lx) < 1.0 and abs(event.y - ly) < 1.0:
                return None
        self._points.append((event.x, event.y))
        return None

    def overlay(self):
        pts = getattr(self, "_points", None)
        return [("poly", list(pts), False)] if pts and len(pts) > 1 else []

    def on_end(self, ctx, event):
        if ctx.selection is None or len(self._points) < 3:
            self._points = []
            return None
        working = self._base.copy() if self._base is not None else \
            Selection(ctx.doc.width, ctx.doc.height)
        working.select_polygon(self._points, self.option("mode", REPLACE),
                              bool(self.option("antialias", True)))
        ctx.selection.mask = working.mask
        ctx.selection._bbox = None
        ctx.selection.geometry = working.geometry
        self._points = []
        self._base = None
        return working

    def on_cancel(self, ctx):
        if self._base is not None and ctx.selection is not None:
            ctx.selection.mask = self._base.mask
            ctx.selection._bbox = None
        self._points = []
        self._base = None
        return None


class MagicWandTool(Tool):
    changes_selection = True
    """Contiguous colour selection -- the tool GIMP calls Fuzzy Select.

    Contiguous by default; the `global` option selects every similar pixel
    regardless of connectivity, which is GIMP's separate "Select by Color".
    Having both behind one tool with a modifier is Paint.NET's arrangement
    and is the more discoverable of the two.
    """

    name = "magic_wand"
    label = "Magic Wand"
    force_global = False
    options_spec = (O.selection_mode(), O.tolerance(32),
                    O.Opt("global", O.BOOL, False, "Global",
                          tip="Select every similar pixel in the layer (GIMP's Select by Color)."),
                    O.Opt("feather", O.INT, 0, "Feather", 0, 200, suffix=" px"))
    wants_stroke = False

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None or ctx.selection is None:
            return None
        if not cell.bounds.contains(event.ix, event.iy):
            return None

        tolerance = int(self.option("tolerance", 32))
        mode = self.option("mode", REPLACE)
        # Shift inverts contiguity, matching Paint.NET.
        is_global = (bool(self.option("global", False)) or self.force_global) ^ event.has(1)

        if is_global:
            ref = cell.pixels[event.iy, event.ix].astype(np.int32)
            a_ref = int(ref[3])
            total = np.zeros(cell.pixels.shape[:2], dtype=np.int64)
            for c in range(3):
                d = cell.pixels[..., c].astype(np.int32) - int(ref[c])
                total += (1 + d * d) * a_ref // 256
            d = cell.pixels[..., 3].astype(np.int32) - a_ref
            total += d * d
            stencil = total <= (tolerance * tolerance * 4)
        else:
            stencil = accel.flood_fill(cell.pixels, event.ix, event.iy, tolerance)

        ctx.selection.select_stencil(stencil, mode)
        feather = int(self.option("feather", 0))
        if feather:
            ctx.selection.feather(feather)
        self.active = False
        return ctx.selection


# ---- more marquees ----------------------------------------------------------------

class SingleRowTool(Tool):
    """Select one full-width row of pixels. Photoshop's Single Row Marquee."""

    name = "select_row"
    label = "Single Row Select"
    wants_stroke = False
    changes_selection = True
    options_spec = (O.selection_mode(),)
    axis = "row"

    def on_begin(self, ctx, event):
        self.active = False
        if ctx.selection is None:
            return None
        w, h = ctx.doc.width, ctx.doc.height
        if self.axis == "row":
            y = min(max(event.iy, 0), h - 1)
            rect = (0, y, w, y + 1)
        else:
            x = min(max(event.ix, 0), w - 1)
            rect = (x, 0, x + 1, h)
        ctx.selection.select_rect(rect, self.option("mode", REPLACE), False)
        return ctx.selection


class SingleColumnTool(SingleRowTool):
    name = "select_col"
    label = "Single Column Select"
    axis = "col"


class SelectByColorTool(MagicWandTool):
    """Select every pixel of a similar colour in the layer. GIMP's own tool
    for what the magic wand does in global mode."""

    name = "select_color"
    label = "Select by Color"
    force_global = True
    options_spec = (O.selection_mode(), O.tolerance(32),
                    O.Opt("feather", O.INT, 0, "Feather", 0, 200, suffix=" px"))


# ---- click-by-click outlines ------------------------------------------------------

class _ClickOutlineTool(Tool):
    """Shared by the polygonal lasso and intelligent scissors.

    The tool stays active between clicks, collecting anchors; the canvas
    shows the outline so far plus a rubber band to the pointer. Clicking the
    first anchor, double-clicking or pressing Enter closes it; Escape
    abandons it. Between anchors, `segment()` decides the path -- a straight
    line for the polygonal lasso, an edge-following one for scissors.
    """

    wants_stroke = False
    changes_selection = True
    multi_click = True
    options_spec = (O.selection_mode(), O.antialias(True))

    def __init__(self, **options):
        super().__init__(**options)
        self._anchors = []
        self._path = []
        self._hover = None

    def segment(self, ctx, a, b):
        return [b]

    def on_begin(self, ctx, event):
        pt = (event.x, event.y)
        if not self._anchors:
            self._base = ctx.selection.copy() if ctx.selection is not None else None
            self._anchors = [pt]
            self._path = [pt]
            self._hover = pt
            return None
        first = self._anchors[0]
        radius = max(2.0, float(getattr(ctx, "pick_radius", 4.0)))
        if len(self._anchors) >= 3 and math.hypot(pt[0] - first[0], pt[1] - first[1]) <= radius:
            self._close_next = True
            return None
        self._path.extend(self.segment(ctx, self._anchors[-1], pt))
        self._anchors.append(pt)
        return None

    def on_end(self, ctx, event):
        if getattr(self, "_close_next", False):
            self._close_next = False
            return self.finish(ctx)
        self.active = True                 # keep collecting anchors
        return None

    def hover(self, ctx, event):
        self._hover = (event.x, event.y)
        return True

    def overlay(self):
        if not self._path:
            return []
        out = [("poly", list(self._path) + ([self._hover] if self._hover else []), False),
               ("handles", list(self._anchors))]
        return out

    def finish(self, ctx):
        self.active = False
        anchors, path = self._anchors, self._path
        self._anchors, self._path, self._hover = [], [], None
        if ctx.selection is None or len(anchors) < 3:
            return None
        closing = self.segment(ctx, anchors[-1], anchors[0])
        points = path + closing
        working = self._base.copy() if self._base is not None else \
            Selection(ctx.doc.width, ctx.doc.height)
        working.select_polygon(points, self.option("mode", REPLACE),
                               bool(self.option("antialias", True)))
        ctx.selection.mask = working.mask
        ctx.selection._bbox = None
        ctx.selection.geometry = working.geometry
        return working

    def on_cancel(self, ctx):
        self._anchors, self._path, self._hover = [], [], None
        return None


class PolygonLassoTool(_ClickOutlineTool):
    """Straight-edged outline, one click per corner. Photoshop's Polygonal Lasso."""

    name = "polygon_lasso"
    label = "Polygonal Lasso"


class ScissorsTool(_ClickOutlineTool):
    """Edge-snapping outline. GIMP's Scissors Select / Photoshop's Magnetic Lasso.

    Click along the edge of an object; between clicks the outline follows
    the strongest edges rather than a straight line. Fewer clicks are needed
    where the edge is clear and more where it is faint -- which is the whole
    reason the tool exists.
    """

    name = "scissors"
    label = "Intelligent Scissors"

    def segment(self, ctx, a, b):
        from .. import scissors
        cell = ctx.cell
        if cell is None:
            return [b]
        if getattr(self, "_cost", None) is None or self._cost.shape != (cell.height, cell.width):
            self._cost = scissors.edge_cost(cell.plane("rgba"))
        pts = scissors.path(self._cost, a, b)
        return [(x + 0.5, y + 0.5) for x, y in pts[1:]] or [b]

    def on_begin(self, ctx, event):
        if not self._anchors:
            self._cost = None
        return super().on_begin(ctx, event)


# ---- assisted selection -------------------------------------------------------------

class QuickSelectTool(Tool):
    """Paint over an object and the selection grows to its edges.

    Photoshop's Quick Selection. Each dab floods outward from the brush,
    within a tolerance of the colour under it, but no further than a few
    brush-widths -- so it spreads across similar colour and stops at edges
    without leaking across the whole canvas. Alt subtracts.
    """

    name = "quick_select"
    label = "Quick Selection"
    wants_stroke = False
    changes_selection = True
    options_spec = (O.size(24), O.tolerance(40),
                    O.Opt("reach", O.INT, 3, "Reach", 1, 10,
                          tip="How far, in brush widths, each dab may spread."))

    def on_begin(self, ctx, event):
        if ctx.selection is None or ctx.cell is None:
            self.active = False
            return None
        self._base = ctx.selection.copy()
        self._acc = np.zeros((ctx.doc.height, ctx.doc.width), dtype=bool)
        self._subtract = event.has(4)
        self._last = None
        return self._dab(ctx, event.x, event.y)

    def on_motion(self, ctx, event):
        size = float(self.option("size", 24))
        if self._last is not None and math.hypot(event.x - self._last[0],
                                                 event.y - self._last[1]) < size / 2.0:
            return None
        return self._dab(ctx, event.x, event.y)

    def _dab(self, ctx, x, y):
        cell = ctx.cell
        self._last = (x, y)
        size = max(2.0, float(self.option("size", 24)))
        reach = int(size * float(self.option("reach", 3)))
        cx, cy = int(x), int(y)
        if not cell.bounds.contains(cx, cy):
            return None
        x0, y0 = max(0, cx - reach), max(0, cy - reach)
        x1, y1 = min(cell.width, cx + reach + 1), min(cell.height, cy + reach + 1)
        px = cell.plane("rgba")[y0:y1, x0:x1]
        r = int(size / 2)
        # Seed from every pixel under the brush, not just its centre, so a
        # dab over a little texture still grabs the region it is on.
        stencil = np.zeros(px.shape[:2], dtype=bool)
        tol = int(self.option("tolerance", 40))
        for sy in range(max(0, cy - r - y0), min(px.shape[0], cy + r - y0 + 1), max(1, r // 2)):
            for sx in range(max(0, cx - r - x0), min(px.shape[1], cx + r - x0 + 1), max(1, r // 2)):
                if stencil[sy, sx]:
                    continue
                stencil |= accel.flood_fill(px, sx, sy, tol)
        self._acc[y0:y1, x0:x1] |= stencil
        self._apply(ctx)
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def _apply(self, ctx):
        working = self._base.copy()
        working.select_stencil(self._acc, EXCLUDE if self._subtract else UNION)
        ctx.selection.mask = working.mask
        ctx.selection._bbox = None

    def on_end(self, ctx, event):
        self._base = None
        return ctx.selection

    def on_cancel(self, ctx):
        if getattr(self, "_base", None) is not None:
            ctx.selection.mask = self._base.mask
            ctx.selection._bbox = None
        self._base = None
        return None


class ObjectSelectTool(_DragSelectTool):
    """Drag a box around an object and the object is selected.

    Photoshop's Object Selection does this with a trained model; this does it
    with colour models (see engine/segment.py), which is honest about what a
    numpy engine can do: excellent when the object differs in colour from
    what surrounds it, weak when it does not. The result is an ordinary
    selection, so it can be refined with any other tool.
    """

    name = "object_select"
    label = "Object Selection"
    shape = "rect"
    options_spec = (O.selection_mode(),)

    def overlay(self):
        r = getattr(self, "_band", None)
        return [("rect", r)] if r else []

    def on_motion(self, ctx, event):
        self._band = self._rect(event)
        return None

    def on_end(self, ctx, event):
        from .. import segment
        rect = self._rect(event)
        self._band = None
        cell = ctx.cell
        if cell is None or ctx.selection is None or self._base is None:
            return None
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(ctx.doc.width, x1), min(ctx.doc.height, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        region = np.zeros((ctx.doc.height, ctx.doc.width), dtype=bool)
        region[y0:y1, x0:x1] = True
        mask = segment.segment(cell.plane("rgba"), region)
        working = self._base.copy()
        working.select_stencil(mask >= 128, self.option("mode", REPLACE))
        ctx.selection.mask = working.mask
        ctx.selection._bbox = None
        self._base = None
        return working


class ForegroundSelectTool(LassoTool):
    """Draw a rough loop around a subject and it is cut out of its background.

    GIMP's Foreground Select, reduced to its first step: the loop says where
    the subject is, colour models separate it from what surrounds it. As with
    Object Selection, this is the classical method, not a trained one.
    """

    name = "foreground_select"
    label = "Foreground Select"
    options_spec = (O.selection_mode(),)

    def on_end(self, ctx, event):
        from .. import segment
        from ..selection import rasterize_polygon
        if ctx.selection is None or len(self._points) < 3 or ctx.cell is None:
            self._points = []
            return None
        region = rasterize_polygon(ctx.doc.width, ctx.doc.height, self._points, False) >= 128
        self._points = []
        mask = segment.segment(ctx.cell.plane("rgba"), region)
        working = self._base.copy() if self._base is not None else \
            Selection(ctx.doc.width, ctx.doc.height)
        working.select_stencil(mask >= 128, self.option("mode", REPLACE))
        ctx.selection.mask = working.mask
        ctx.selection._bbox = None
        self._base = None
        return working
