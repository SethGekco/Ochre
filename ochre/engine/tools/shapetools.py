# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Shape and gradient tools.

These are the tools that earn back the stroke session's `restore()`. A shape
is not painted incrementally like a brush stroke -- it is redrawn from
scratch on every mouse-move as the user drags. So each motion event restores
the pixels the shape last covered and draws the new one, and because the
buffer it restores from is the SAME buffer that becomes the undo entry, the
rubber-band preview and the committed result cannot disagree. No separate
preview layer exists, and none is needed.

Modifiers are read live on every motion event rather than latched at
mouse-down, so pressing or releasing Shift mid-drag updates the shape
immediately. That is the behaviour people expect and the one most
implementations get wrong by sampling modifiers once.
"""

import math

import numpy as np

from .. import shapes
from ..geometry import Rect
from ..gradient import (CONICAL, DIAMOND, Gradient, LINEAR, RADIAL,
                        REFLECTED, REPEAT_CLAMP)
from ..stroke import StrokeSession
from .base import Tool, constrain_angle, drag_rect


class _DragTool(Tool):
    """Shared: open a session, redraw on every move, commit once."""

    def _begin_session(self, ctx):
        s = ctx.settings
        return StrokeSession(
            ctx.doc, ctx.layer, ctx.frame,
            block=256 if s is None else s.get("History", "BlockSize"),
            max_blocks=512 if s is None else s.get("History", "MaxStrokeBlocks"),
            level=1 if s is None else s.get("History", "CompressLevel"),
            label=self.label)

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            return None
        self._session = self._begin_session(ctx)
        self._start = event
        self._last_rect = None
        return None

    def on_motion(self, ctx, event):
        if self._session is None:
            return None
        # Put back whatever the previous preview covered, then draw again.
        # Same buffer as the undo snapshot, so they cannot drift apart.
        restored = self._session.restore(self._last_rect)
        drawn = self._render(ctx, event)
        self._last_rect = drawn
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def on_end(self, ctx, event):
        if self._session is None:
            return None
        self._session.restore(self._last_rect)
        self._render(ctx, event)
        cmd = self._session.commit()
        self._session = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        if self._session is None:
            return None
        rect = self._session.cancel()
        self._session = None
        return rect

    def _render(self, ctx, event):
        raise NotImplementedError

    def _paint(self, ctx, cov, rect, colour=None):
        """Apply a coverage plane as colour, through the one chokepoint."""
        cell = ctx.cell
        clipped = rect.clipped_to(cell.width, cell.height)
        if clipped is None:
            return None
        self._session.touch(clipped)
        window = cov[clipped.y - rect.y:clipped.y - rect.y + clipped.h,
                     clipped.x - rect.x:clipped.x - rect.x + clipped.w]
        src = np.empty((clipped.h, clipped.w, 4), dtype=np.uint8)
        src[:] = colour if colour is not None else ctx.colour_for(event_none())
        cell.apply_masked(clipped, src, ctx.coverage(window, clipped))
        return clipped


def event_none():
    """A left-button stand-in, for paths that already resolved the colour."""
    from .base import ToolEvent
    return ToolEvent(0, 0)


class _ShapeTool(_DragTool):
    """Rectangle, ellipse and line share everything but their geometry."""

    kind = "rectangle"
    default_stroke = 2

    def _geometry(self, event):
        # Live modifiers: Shift constrains, Alt draws from the centre.
        if self.kind == "line":
            x0, y0 = self._start.x, self._start.y
            x1, y1 = event.x, event.y
            if event.has(1):
                x1, y1 = constrain_angle(x0, y0, x1, y1)
            return (x0, y0, x1, y1)
        return drag_rect(self._start, event,
                         square=event.has(1), centre=event.has(4))

    def _render(self, ctx, event):
        cell = ctx.cell
        geometry = self._geometry(event)
        stroke = float(self.option("stroke", self.default_stroke))
        style = self.option("style", shapes.BOTH if self.kind != "line"
                            else shapes.OUTLINE)
        rect = shapes.bounds_for(cell.width, cell.height, self.kind,
                                 geometry, stroke)
        if rect is None:
            return None

        colour = ctx.colour_for(event)
        antialias = bool(self.option("antialias", True))

        if style == shapes.BOTH and self.kind != "line":
            # Fill with the secondary colour and stroke with the primary, so
            # a two-colour shape is one drag rather than two.
            fill = shapes.coverage(cell.width, cell.height, self.kind,
                                   geometry, shapes.FILL, stroke, antialias)
            self._paint_region(ctx, fill, rect, ctx.secondary)
            edge = shapes.coverage(cell.width, cell.height, self.kind,
                                   geometry, shapes.OUTLINE, stroke, antialias)
            self._paint_region(ctx, edge, rect, colour)
            return rect

        cov = shapes.coverage(cell.width, cell.height, self.kind, geometry,
                              style, stroke, antialias)
        self._paint_region(ctx, cov, rect, colour)
        return rect

    def _paint_region(self, ctx, full_cov, rect, colour):
        cell = ctx.cell
        self._session.touch(rect)
        window = full_cov[rect.slice()]
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = colour
        cell.apply_masked(rect, src, ctx.coverage(window, rect))


class LineTool(_ShapeTool):
    name = "line"
    label = "Line"
    kind = "line"
    default_stroke = 2


class RectangleTool(_ShapeTool):
    name = "rectangle"
    label = "Rectangle"
    kind = "rectangle"


class EllipseTool(_ShapeTool):
    name = "ellipse"
    label = "Ellipse"
    kind = "ellipse"


class FreeformTool(_DragTool):
    """Freehand outline, optionally closed and filled."""

    name = "freeform"
    label = "Freeform Shape"

    def on_begin(self, ctx, event):
        result = super().on_begin(ctx, event)
        self._points = [(event.x, event.y)]
        return result

    def on_motion(self, ctx, event):
        if self._session is None:
            return None
        # Drop points that add no detail; a dense freehand path makes the
        # polygon rasteriser needlessly slow without changing the result.
        lx, ly = self._points[-1]
        if math.hypot(event.x - lx, event.y - ly) < 1.5:
            return None
        self._points.append((event.x, event.y))
        return super().on_motion(ctx, event)

    def _render(self, ctx, event):
        cell = ctx.cell
        if len(self._points) < 2:
            return None
        stroke = float(self.option("stroke", 2))
        style = self.option("style", shapes.OUTLINE)
        rect = shapes.bounds_for(cell.width, cell.height, "freeform",
                                 self._points, stroke)
        if rect is None:
            return None
        cov = shapes.coverage(cell.width, cell.height, "freeform",
                              self._points, style, stroke,
                              bool(self.option("antialias", True)))
        self._session.touch(rect)
        window = cov[rect.slice()]
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = ctx.colour_for(event)
        cell.apply_masked(rect, src, ctx.coverage(window, rect))
        return rect

    def on_cancel(self, ctx):
        self._points = []
        return super().on_cancel(ctx)


class GradientTool(_DragTool):
    """Drag to define the axis; the gradient fills the selection or the layer.

    Unlike the shapes, a gradient covers a region rather than tracing one, so
    it renders over the selection's bounds -- or the whole layer when nothing
    is selected.
    """

    name = "gradient"
    label = "Gradient"

    KINDS = {"linear": LINEAR, "reflected": REFLECTED, "diamond": DIAMOND,
             "radial": RADIAL, "conical": CONICAL}

    def _render(self, ctx, event):
        cell = ctx.cell
        x0, y0 = self._start.x, self._start.y
        x1, y1 = event.x, event.y
        if event.has(1):                       # Shift constrains the axis
            x1, y1 = constrain_angle(x0, y0, x1, y1)

        # Confine to the selection when there is one: a gradient over the
        # whole layer, then masked, wastes most of its work.
        if ctx.selection is not None and not ctx.selection.selects_all():
            rect = ctx.selection.bbox.clipped_to(cell.width, cell.height)
        else:
            rect = cell.bounds
        if rect is None or rect.is_empty:
            return None

        kind = self.KINDS.get(str(self.option("kind", "linear")), LINEAR)
        gradient = Gradient(
            kind=kind, start=(x0, y0), end=(x1, y1),
            start_colour=ctx.colour_for(event),
            end_colour=(ctx.secondary if event.button != 2 else ctx.primary),
            repeat=str(self.option("repeat", REPEAT_CLAMP)),
            alpha_only=bool(self.option("alpha_only", False)),
            reverse=bool(self.option("reverse", False)))

        self._session.touch(rect)
        base = cell.plane("rgba")[rect.slice()] if gradient.alpha_only else None
        src = gradient.render(rect, base)
        cell.apply_masked(rect, src, ctx.coverage(None, rect))
        return rect


class TextTool(_DragTool):
    """Place and re-render text, live.

    The interaction a text tool needs is different from a shape's: you click
    once, then TYPE, and the result must update as you go. So the session
    deliberately stays open after mouse-up -- `on_end` finishes POSITIONING,
    not the edit. A caller keeps editing by calling `set_text()` and finishes
    with `commit_text()`.

    That is the same restore-and-redraw loop the shapes use, so live text
    preview costs nothing extra and, because it restores from the buffer that
    becomes the undo entry, what you see while typing is exactly what gets
    committed.

    The in-canvas caret and keyboard handling belong to the UI; everything
    below is headless and scriptable.
    """

    name = "text"
    label = "Text"

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            return None
        self._session = self._begin_session(ctx)
        self._anchor = (event.x, event.y)
        self._last_rect = None
        return self._render(ctx, event)

    def on_motion(self, ctx, event):
        """Dragging repositions the text rather than resizing it."""
        if self._session is None:
            return None
        self._anchor = (event.x, event.y)
        return super().on_motion(ctx, event)

    def on_end(self, ctx, event):
        """Finish POSITIONING. The session stays open so typing can continue."""
        if self._session is None:
            return None
        self._anchor = (event.x, event.y)
        restored = self._session.restore(self._last_rect)
        drawn = self._render(ctx, event)
        self._last_rect = drawn
        self.active = True          # still editing
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def set_text(self, ctx, text):
        """Replace the string and redraw. What a UI calls on every keystroke."""
        self.set_option("text", text)
        if self._session is None:
            return None
        restored = self._session.restore(self._last_rect)
        drawn = self._render(ctx, None)
        self._last_rect = drawn
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def commit_text(self, ctx):
        """Finish the edit and push one history entry."""
        if self._session is None:
            return None
        cmd = self._session.commit()
        self._session = None
        self.active = False
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def _render(self, ctx, event):
        from .. import text as textmod

        cell = ctx.cell
        body = str(self.option("text", ""))
        if not body:
            return None

        cov, w, h = textmod.render_coverage(
            body,
            family=str(self.option("font", "DejaVu Sans")),
            size=int(self.option("size", 24)),
            align=str(self.option("align", textmod.LEFT)),
            line_spacing=float(self.option("line_spacing", 1.0)),
            antialias=bool(self.option("antialias", True)))
        if w == 0 or h == 0:
            return None

        ax, ay = self._anchor
        rect = textmod.place(cov, ax, ay, str(self.option("align", textmod.LEFT)))
        clipped = rect.clipped_to(cell.width, cell.height)
        if clipped is None:
            return None

        window = cov[clipped.y - rect.y:clipped.y - rect.y + clipped.h,
                     clipped.x - rect.x:clipped.x - rect.x + clipped.w]
        self._session.touch(clipped)

        colour = ctx.colour_for(event) if event is not None else ctx.primary
        src = np.empty((clipped.h, clipped.w, 4), dtype=np.uint8)
        src[:] = colour
        cell.apply_masked(clipped, src, ctx.coverage(window, clipped))
        return clipped
