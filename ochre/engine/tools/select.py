# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Selection tools: rectangle, ellipse, lasso, magic wand.

All four produce an antialiased coverage mask and hand it to the selection's
combine step, so the combine mode (replace / union / exclude / intersect /
xor) is shared machinery rather than something each tool reimplements.

The in-progress drag composes with the committed selection for display
without mutating it, which is what makes Shift-drag-to-add preview correctly
and Escape free.
"""

import numpy as np

from .. import accel
from ..selection import REPLACE, Selection
from .base import Tool, drag_rect


class _DragSelectTool(Tool):
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
    shape = "rect"


class EllipseSelectTool(_DragSelectTool):
    name = "select_ellipse"
    label = "Ellipse Select"
    shape = "ellipse"


class LassoTool(Tool):
    """Freehand polygon selection."""

    name = "lasso"
    label = "Lasso Select"
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
    """Contiguous colour selection -- the tool GIMP calls Fuzzy Select.

    Contiguous by default; the `global` option selects every similar pixel
    regardless of connectivity, which is GIMP's separate "Select by Color".
    Having both behind one tool with a modifier is Paint.NET's arrangement
    and is the more discoverable of the two.
    """

    name = "magic_wand"
    label = "Magic Wand"
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
        is_global = bool(self.option("global", False)) ^ event.has(1)

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
