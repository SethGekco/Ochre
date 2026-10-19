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
from . import options as O


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
    """Every closed shape, plus the line, share all but their geometry."""

    kind = "rectangle"
    default_stroke = 2

    @property
    def shape(self):
        """Which shape to draw. One TOOL with a choice, not nine tools.

        Paint.NET settled on this and it is the right call: rectangle,
        pentagon and star differ by one dropdown, not by anything a user
        thinks of as a different activity, and nine near-identical toolbar
        buttons push everything else off the strip. The INI default keeps
        rectangle as what you get without choosing.
        """
        from .. import shapelib
        name = str(self.option("shape", self.kind) or self.kind).lower()
        name = _SHAPE_ALIASES.get(name, name)
        if name != "line" and name not in shapelib.CATALOG \
                and name not in shapes.POLYGONS:
            return self.kind          # unknown name: fall back, never raise
        return name

    def _geometry(self, event):
        # Live modifiers: Shift constrains, Alt draws from the centre.
        if self.shape == "line":
            x0, y0 = self._start.x, self._start.y
            x1, y1 = event.x, event.y
            if event.has(1):
                x1, y1 = constrain_angle(x0, y0, x1, y1)
            return (x0, y0, x1, y1)
        return drag_rect(self._start, event,
                         square=event.has(1), centre=event.has(4))

    def _render(self, ctx, event):
        cell = ctx.cell
        kind = self.shape
        geometry = self._geometry(event)
        stroke = float(self.option("stroke", self.default_stroke))
        style = self.option("style", shapes.BOTH if kind != "line"
                            else shapes.OUTLINE)
        rect = shapes.bounds_for(cell.width, cell.height, kind,
                                 geometry, stroke)
        if rect is None:
            return None

        colour = ctx.colour_for(event)
        antialias = bool(self.option("antialias", True))

        if style == shapes.BOTH and kind != "line":
            # Fill with the secondary colour and stroke with the primary, so
            # a two-colour shape is one drag rather than two.
            params = self._shape_params(kind)
            fill = shapes.coverage(cell.width, cell.height, kind,
                                   geometry, shapes.FILL, stroke, antialias, **params)
            self._paint_region(ctx, fill, rect, ctx.secondary)
            edge = shapes.coverage(cell.width, cell.height, kind,
                                   geometry, shapes.OUTLINE, stroke, antialias, **params)
            self._paint_region(ctx, edge, rect, colour)
            return rect

        cov = shapes.coverage(cell.width, cell.height, kind, geometry,
                              style, stroke, antialias, **self._shape_params(kind))
        self._paint_region(ctx, cov, rect, colour)
        return rect

    def _shape_params(self, kind):
        radius = self.option("radius", 0)
        if kind in ("rounded_rectangle", "callout_rounded") and radius:
            return {"radius": float(radius)}
        return {}

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
    options_spec = (O.stroke(2), O.antialias(True))
    kind = "line"
    default_stroke = 2


class ShapeTool(_ShapeTool):
    """Rectangle by default; ellipse, polygons and stars by option."""

    name = "shape"
    label = "Shapes"
    kind = "rectangle"

    @staticmethod
    def choices():
        """Offered shapes, in the order a chooser should list them."""
        from .. import shapelib
        return tuple(shapelib.CATALOG)

    @classmethod
    def _spec(cls):
        from .. import shapelib
        return (O.Opt("shape", O.CHOICE, "rectangle", "Shape",
                      choices=tuple(shapelib.CATALOG),
                      labels={k: shapelib.label(k) for k in shapelib.CATALOG},
                      groups=shapelib.grouped()),
                O.fill_style("both"), O.stroke(2), O.antialias(True),
                O.Opt("radius", O.INT, 0, "Corner radius", 0, 2000, suffix=" px",
                      tip="Rounded shapes only. 0 picks a radius from the size."))


ShapeTool.options_spec = ShapeTool._spec()

_SHAPE_ALIASES = {"circle": "ellipse", "oval": "ellipse", "square": "rectangle",
                  "rect": "rectangle", "star": "star5", "rounded": "rounded_rectangle",
                  "roundrect": "rounded_rectangle", "arrow_right": "arrow",
                  "x": "cross", "tick": "check", "bolt": "lightning"}


# Kept so an existing tools.ini or a script naming them still resolves.
# Both are the same tool with its shape pre-chosen.
class RectangleTool(ShapeTool):
    name = "rectangle"
    label = "Rectangle"
    kind = "rectangle"
    hidden = True


class EllipseTool(ShapeTool):
    name = "ellipse"
    label = "Ellipse"
    kind = "ellipse"
    hidden = True


class FreeformTool(_DragTool):
    """Freehand outline, optionally closed and filled."""

    name = "freeform"
    label = "Freeform Shape"
    options_spec = (O.fill_style("outline"), O.stroke(2), O.antialias(True))

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
    options_spec = (O.Opt("kind", O.CHOICE, "linear", "Type",
                          choices=("linear", "reflected", "diamond", "radial", "conical")),
                    O.Opt("repeat", O.CHOICE, "clamp", "Repeat",
                          choices=("clamp", "repeat", "reflect"),
                          labels={"clamp": "None", "repeat": "Sawtooth", "reflect": "Triangular"}),
                    O.Opt("alpha_only", O.BOOL, False, "Transparency mode"),
                    O.Opt("reverse", O.BOOL, False, "Reverse"))

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
    # Typed text is work: switching tools commits it rather than throwing
    # it away, as Paint.NET does.
    commit_on_switch = True
    options_spec = (O.Opt("font", O.FONT, "DejaVu Sans", "Font"),
                    O.Opt("size", O.INT, 24, "Size", 4, 1000, suffix=" px"),
                    O.Opt("align", O.CHOICE, "left", "Align", choices=("left", "center", "right")),
                    O.Opt("line_spacing", O.FLOAT, 1.0, "Line spacing", 0.5, 4.0, 0.1),
                    O.antialias(True),
                    O.Opt("orientation", O.CHOICE, "horizontal", "Direction",
                          choices=("horizontal", "vertical"),
                          tip="Vertical sets characters top to bottom, columns "
                              "right to left (Photoshop's Vertical Type)."),
                    O.Opt("output", O.CHOICE, "pixels", "Makes",
                          choices=("pixels", "selection"),
                          labels={"pixels": "Text", "selection": "Selection (type mask)"},
                          tip="Selection makes the letters a selection instead of "
                              "paint: Photoshop's Type Mask tools."),
                    O.Opt("layer", O.BOOL, True, "Editable text layer",
                          tip="On: the text goes on its own text layer and stays "
                              "editable -- click it again to change it. Off: it is "
                              "painted onto the current layer, as in Paint.NET."),
                    O.Opt("box_width", O.INT, 0, "Box", 0, 20000, 10, suffix=" px",
                          tip="Paragraph box width: lines wrap to fit. 0 is point text. "
                              "Dragging when you start a text layer sets it."),
                    O.Opt("warp", O.CHOICE, "none", "Warp",
                          choices=("none", "arc", "arch", "bulge", "flag", "wave", "rise", "fish"),
                          tip="Photoshop's Warp Text styles (text layers)."),
                    O.Opt("bend", O.INT, 50, "Bend", -100, 100, 5, suffix=" %"))

    # Keys a text layer's params share with this tool's options.
    _LAYER_KEYS = ("text", "font", "size", "align", "line_spacing", "orientation",
                   "antialias", "box_width", "warp", "bend")

    @property
    def changes_selection(self):
        """A type mask's result is a selection, recorded as one."""
        return self.option("output", "pixels") == "selection"

    @property
    def vertical(self):
        return self.option("orientation", "horizontal") == "vertical"

    def _display_text(self):
        """The string as laid out: wrapped to the box, when there is one.
        Wrapping only turns spaces into newlines, so indices are unchanged."""
        body = str(self.option("text", ""))
        width = int(self.option("box_width", 0) or 0)
        if width > 0 and not self.vertical and getattr(self, "_edit", None) is not None:
            from .. import textlayer
            body = textlayer.wrap(body, str(self.option("font", "DejaVu Sans")),
                                  int(self.option("size", 24)), width)
        return body

    def _anchor_x(self):
        """Where alignment hangs from: the anchor, or the box edge or middle."""
        from .. import text as textmod
        ax = self._anchor[0]
        width = int(self.option("box_width", 0) or 0)
        if width > 0 and not self.vertical and getattr(self, "_edit", None) is not None:
            align = self._align()
            ax += width / 2.0 if align == textmod.CENTER else width if align == textmod.RIGHT else 0
        return ax

    def _layout(self):
        from .. import text as textmod
        body = self._display_text()
        if self.vertical:
            return textmod.vertical_layout(
                body, family=str(self.option("font", "DejaVu Sans")),
                size=int(self.option("size", 24)),
                line_spacing=float(self.option("line_spacing", 1.0)),
                antialias=bool(self.option("antialias", True)))
        return textmod.layout(
            body, family=str(self.option("font", "DejaVu Sans")),
            size=int(self.option("size", 24)),
            align=str(self.option("align", textmod.LEFT)),
            line_spacing=float(self.option("line_spacing", 1.0)),
            antialias=bool(self.option("antialias", True)))

    def _align(self):
        from .. import text as textmod
        # Vertical text anchors at its top-left; alignment is a horizontal idea.
        return textmod.LEFT if self.vertical else str(self.option("align", textmod.LEFT))

    # ---- text layers -----------------------------------------------------
    # When the layer under the tool is a TEXT layer (the controller makes
    # or picks one), editing changes that layer's params instead of painting:
    # the compositor renders them. self._edit holds the session.

    def _layer_params(self):
        p = {k: self.option(k) for k in self._LAYER_KEYS}
        p["text"] = str(p["text"] or "")
        p["color"] = tuple(self._colour)
        p["x"], p["y"] = float(self._anchor[0]), float(self._anchor[1])
        return p

    def _apply(self, ctx):
        """Push the tool's state into the layer; return the rect to repaint."""
        from .. import layerstyles, textlayer
        layer = self._edit["layer"]
        old = textlayer.bounds(layer.params)
        layer.params = self._layer_params()
        new = textlayer.bounds(layer.params)
        rects = [r for r in (old, new, self._edit.get("rect")) if r is not None]
        self._edit["rect"] = new
        if not rects:
            return None
        rect = rects[0]
        for r in rects[1:]:
            rect = rect.union(r)
        rect = rect.inflated(layerstyles.reach(getattr(layer, "styles", {})) + 2)
        clipped = rect.clipped_to(ctx.doc.width, ctx.doc.height)
        if clipped is not None:
            ctx.doc.cell(layer).dirty.add(clipped)
        return clipped

    def _begin_layer_edit(self, ctx, event):
        layer = ctx.layer
        new = bool(getattr(ctx, "text_layer_new", False)) or not layer.params.get("text")
        self._edit = {"layer": layer, "new": new, "rect": None,
                      "before": None if new else dict(layer.params),
                      "start": (event.x, event.y)}
        if new:
            self._anchor = (event.x, event.y)
            self._colour = tuple(ctx.colour_for(event))
            self.set_option("box_width", 0)
        else:
            from .. import textlayer
            p = textlayer.full(layer.params)
            for k in self._LAYER_KEYS:
                self.set_option(k, p[k])
            self._anchor = (float(p["x"]), float(p["y"]))
            colour = p["color"]
            self._colour = tuple(int(v) for v in (colour.split(",") if isinstance(colour, str) else colour))
            self.caret_from_point(ctx, event.x, event.y)
        return self._apply(ctx)

    def options_changed(self, ctx):
        """Font, size, warp... changed in the options bar mid-edit."""
        if getattr(self, "_edit", None) is not None:
            return self._apply(ctx)
        return None

    def set_colour(self, ctx, rgba):
        """The primary colour changed while editing a text layer: recolour it."""
        if getattr(self, "_edit", None) is None:
            return None
        self._colour = tuple(rgba)
        return self._apply(ctx)

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            return None
        self._edit = None
        if getattr(ctx.layer, "kind", "raster") == "text" and not self.changes_selection:
            return self._begin_layer_edit(ctx, event)
        self._session = self._begin_session(ctx)
        self._anchor = (event.x, event.y)
        self._last_rect = None
        return self._render(ctx, event)

    def on_motion(self, ctx, event):
        """Dragging repositions the text rather than resizing it -- except
        when starting a text layer, where a drag draws its paragraph box."""
        if getattr(self, "_edit", None) is not None:
            if not self._edit["new"]:
                return None
            sx, sy = self._edit["start"]
            width = abs(event.x - sx)
            self._anchor = (min(sx, event.x), sy)
            self.set_option("box_width", int(width) if width > 8 else 0)
            return self._apply(ctx)
        if self._session is None:
            return None
        self._anchor = (event.x, event.y)
        return super().on_motion(ctx, event)

    def on_end(self, ctx, event):
        """Finish POSITIONING. The session stays open so typing can continue."""
        if getattr(self, "_edit", None) is not None:
            if self._edit["new"]:
                self.on_motion(ctx, event)
            self.active = True          # still editing: typing goes to the layer
            return self._edit.get("rect")
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

    # ---- editing -------------------------------------------------------
    # The caret lives here, not in the UI. It is an integer index into a
    # string -- plain data, no Qt -- so the whole edit loop stays scriptable
    # and headlessly testable like every other tool. The UI's job shrinks to
    # mapping key events onto these methods.

    @property
    def caret(self):
        return max(0, min(int(getattr(self, "_caret", 0)),
                          len(str(self.option("text", "")))))

    @caret.setter
    def caret(self, index):
        self._caret = max(0, min(int(index), len(str(self.option("text", "")))))

    def insert(self, ctx, chunk):
        """Type at the caret. Handles multi-character input, e.g. a paste."""
        body = str(self.option("text", ""))
        at = self.caret
        self._caret = at + len(chunk)
        return self.set_text(ctx, body[:at] + chunk + body[at:])

    def backspace(self, ctx):
        body = str(self.option("text", ""))
        at = self.caret
        if at == 0:
            return None
        self._caret = at - 1
        return self.set_text(ctx, body[:at - 1] + body[at:])

    def delete(self, ctx):
        body = str(self.option("text", ""))
        at = self.caret
        if at >= len(body):
            return None
        return self.set_text(ctx, body[:at] + body[at + 1:])

    def move_caret(self, delta):
        self.caret = self.caret + int(delta)
        return self.caret

    def caret_home(self):
        """To the start of the current LINE, not of the whole string."""
        body = str(self.option("text", ""))
        self.caret = body.rfind("\n", 0, self.caret) + 1
        return self.caret

    def caret_end(self):
        body = str(self.option("text", ""))
        nxt = body.find("\n", self.caret)
        self.caret = len(body) if nxt < 0 else nxt
        return self.caret

    def caret_line(self, delta):
        """Up or down a line, keeping the column where possible."""
        body = str(self.option("text", ""))
        before = body.rfind("\n", 0, self.caret) + 1
        column = self.caret - before
        if delta < 0:
            prev = body.rfind("\n", 0, max(0, before - 1))
            if before == 0:
                return self.caret
            self.caret = min(prev + 1 + column, before - 1)
        else:
            nxt = body.find("\n", self.caret)
            if nxt < 0:
                return self.caret
            after = body.find("\n", nxt + 1)
            limit = len(body) if after < 0 else after
            self.caret = min(nxt + 1 + column, limit)
        return self.caret

    def caret_rect(self, ctx):
        """Where to draw the caret, in DOCUMENT coordinates.

        Returns a 1-pixel-wide Rect. Returned even when the string is empty,
        which is exactly when a caret matters most -- otherwise clicking to
        start typing shows nothing at all.
        """
        from .. import text as textmod

        if self._anchor is None:
            return None
        lay = self._layout()
        cx, cy, extent = lay.caret_at(self.caret)

        ax, ay = self._anchor_x(), self._anchor[1]
        origin = textmod.place(lay.coverage, ax, ay, self._align()) if lay.width else None
        ox = origin.x if origin is not None else int(round(ax))
        oy = origin.y if origin is not None else int(round(ay))
        if getattr(lay, "vertical", False):
            # Between two stacked characters the caret lies across the column.
            return Rect(ox + cx, oy + cy, max(1, int(extent)), 1)
        return Rect(ox + cx, oy + cy, 1, max(1, int(extent)))

    def caret_from_point(self, ctx, x, y):
        """Click-to-position: move the caret nearest a document point."""
        from .. import text as textmod

        if self._anchor is None:
            return self.caret
        lay = self._layout()
        ax, ay = self._anchor_x(), self._anchor[1]
        origin = textmod.place(lay.coverage, ax, ay, self._align()) if lay.width else None
        ox = origin.x if origin is not None else int(round(ax))
        oy = origin.y if origin is not None else int(round(ay))
        self.caret = lay.index_at(x - ox, y - oy)
        return self.caret

    def set_text(self, ctx, text):
        """Replace the string and redraw. What a UI calls on every keystroke."""
        self.set_option("text", text)
        self._caret = min(getattr(self, "_caret", len(text)), len(text))
        if getattr(self, "_edit", None) is not None:
            return self._apply(ctx)
        if self._session is None:
            return None
        restored = self._session.restore(self._last_rect)
        drawn = self._render(ctx, None)
        self._last_rect = drawn
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def finish(self, ctx):
        return self.commit_text(ctx)

    def _commit_layer(self, ctx):
        from ..commands import PropertyDelta, StructureDelta
        edit, self._edit = self._edit, None
        self.active = False
        layer = edit["layer"]
        body = str(self.option("text", ""))
        rect = edit.get("rect")
        if edit["new"]:
            if not body.strip():
                # Clicked and typed nothing: no layer, no history.
                ctx.doc.remove_layer(layer)
                self._clear()
                return rect or ctx.doc.bounds
            layer.name = body.strip().splitlines()[0][:40]
            if ctx.history is not None:
                ctx.history.push(StructureDelta.insertion(ctx.doc, layer, "Text"), "Text")
        elif layer.params != edit["before"] and ctx.history is not None:
            ctx.history.push(PropertyDelta(layer.id, "params", edit["before"], "Edit Text"),
                             "Edit Text")
        self._clear()
        return rect or ctx.doc.bounds

    def commit_text(self, ctx):
        """Finish the edit and push one history entry."""
        if getattr(self, "_edit", None) is not None:
            return self._commit_layer(ctx)
        if self._session is None:
            return None
        if self.changes_selection:
            return self._commit_mask(ctx)
        cmd = self._session.commit()
        self._session = None
        self.active = False
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        self._clear()
        return cmd

    def on_cancel(self, ctx):
        edit = getattr(self, "_edit", None)
        if edit is not None:
            self._edit = None
            rect = edit.get("rect") or ctx.doc.bounds
            if edit["new"]:
                ctx.doc.remove_layer(edit["layer"])
            else:
                edit["layer"].params = edit["before"]
                ctx.doc.cell(edit["layer"]).dirty.add(ctx.doc.bounds)
            self._clear()
            return rect
        rect = super().on_cancel(ctx)
        self._clear()
        return rect

    def _clear(self):
        """Forget the string once the edit is over.

        The tool instance is reused for every placement, so without this the
        NEXT click starts holding the last thing typed -- and because
        on_begin renders immediately, it does not merely prefill the box, it
        stamps the old text down at the new spot.

        Cleared when the edit FINISHES rather than when one starts, so that
        a script can still do create("text", text="...") and have the first
        placement use it.
        """
        self.set_option("text", "")
        self._caret = 0

    def _coverage(self):
        """(rect, window) of the current text in document space, or None."""
        from .. import text as textmod
        body = str(self.option("text", ""))
        if not body or self._anchor is None:
            return None
        if self.vertical:
            cov = self._layout().coverage
        else:
            cov, _w, _h = textmod.render_coverage(
                body,
                family=str(self.option("font", "DejaVu Sans")),
                size=int(self.option("size", 24)),
                align=str(self.option("align", textmod.LEFT)),
                line_spacing=float(self.option("line_spacing", 1.0)),
                antialias=bool(self.option("antialias", True)))
        if cov.size == 0:
            return None
        return textmod.place(cov, self._anchor[0], self._anchor[1], self._align()), cov

    def _render(self, ctx, event):
        cell = ctx.cell
        placed = self._coverage()
        if placed is None:
            return None
        rect, cov = placed
        clipped = rect.clipped_to(cell.width, cell.height)
        if clipped is None:
            return None

        window = cov[clipped.y - rect.y:clipped.y - rect.y + clipped.h,
                     clipped.x - rect.x:clipped.x - rect.x + clipped.w]
        self._session.touch(clipped)

        src = np.empty((clipped.h, clipped.w, 4), dtype=np.uint8)
        if self.changes_selection:
            # A type mask previews the way Photoshop's does: a translucent
            # red overlay while typing, the selection only on commit. The
            # preview is restored away then, so it never reaches the layer.
            src[:] = (255, 40, 40, 255)
            cell.apply_masked(clipped, src, (window.astype(np.uint16) * 140 // 255
                                             ).astype(np.uint8))
            return clipped
        colour = ctx.colour_for(event) if event is not None else ctx.primary
        src[:] = colour
        cell.apply_masked(clipped, src, ctx.coverage(window, clipped))
        return clipped

    def _commit_mask(self, ctx):
        """Type mask: the letters become the selection; the layer is untouched."""
        placed = self._coverage()
        self._session.cancel()
        self._session = None
        self.active = False
        self._clear()
        if placed is None or ctx.selection is None:
            return None
        rect, cov = placed
        doc = ctx.doc
        mask = np.zeros((doc.height, doc.width), dtype=np.uint8)
        clipped = rect.clipped_to(doc.width, doc.height)
        if clipped is None:
            return None
        mask[clipped.slice()] = cov[clipped.y - rect.y:clipped.y - rect.y + clipped.h,
                                    clipped.x - rect.x:clipped.x - rect.x + clipped.w]
        from ..selection import REPLACE
        ctx.selection.combine(mask, REPLACE)
        ctx.selection._bbox = None
        return clipped


class CurveTool(_DragTool):
    """Paint.NET's Line / Curve: draw a line, then bend it by its handles.

    Drag to draw a straight line. It stays editable: four handles appear --
    the two ends and two control points -- and dragging any of them reshapes
    the line into a cubic Bezier, redrawn live from the stroke session.
    Enter, a double-click or switching tools commits it; clicking away from
    the handles commits it and starts the next line there; Escape abandons
    it. Arrowheads at either end follow the curve's direction.
    """

    name = "curve"
    label = "Line / Curve"
    multi_click = True
    commit_on_switch = True
    options_spec = (O.stroke(2), O.antialias(True),
                    O.Opt("arrows", O.CHOICE, "none", "Arrows",
                          choices=("none", "end", "start", "both"),
                          labels={"none": "None", "end": "At end", "start": "At start",
                                  "both": "Both ends"}))

    def __init__(self, **options):
        super().__init__(**options)
        self._pts = None
        self._grab = None
        self._colour = None

    def on_begin(self, ctx, event):
        if self._pts is not None:
            r = max(float(ctx.pick_radius), 4.0)
            best, grab = r * r, None
            for i, (px, py) in enumerate(self._pts):
                d = (px - event.x) ** 2 + (py - event.y) ** 2
                if d <= best:
                    best, grab = d, i
            if grab is not None:
                self._grab = (grab, (event.x, event.y), list(self._pts))
                return None
            self._commit(ctx)
            self.active = True          # this press goes on to draw the next one
        if ctx.cell is None:
            self.active = False
            return None
        self._session = self._begin_session(ctx)
        self._last_rect = None
        self._colour = ctx.colour_for(event)
        self._pts = [(event.x, event.y)] * 4
        self._grab = ("new",)
        return None

    def on_motion(self, ctx, event):
        if self._session is None or self._grab is None:
            return None
        if self._grab[0] == "new":
            x0, y0 = self._pts[0]
            x1, y1 = event.x, event.y
            if event.has(1):
                x1, y1 = constrain_angle(x0, y0, x1, y1)
            self._pts = [(x0, y0), (x0 + (x1 - x0) / 3.0, y0 + (y1 - y0) / 3.0),
                         (x0 + (x1 - x0) * 2 / 3.0, y0 + (y1 - y0) * 2 / 3.0), (x1, y1)]
        else:
            i, (sx, sy), orig = self._grab
            dx, dy = event.x - sx, event.y - sy
            pts = list(orig)
            pts[i] = (orig[i][0] + dx, orig[i][1] + dy)
            self._pts = pts
        restored = self._session.restore(self._last_rect)
        drawn = self._render(ctx, event)
        self._last_rect = drawn
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def on_end(self, ctx, event):
        rect = self.on_motion(ctx, event)
        grab, self._grab = self._grab, None
        if grab is not None and grab[0] == "new":
            (x0, y0), (x1, y1) = self._pts[0], self._pts[3]
            if math.hypot(x1 - x0, y1 - y0) < 1.0:
                # A click with no drag draws nothing and leaves nothing open.
                self.on_cancel(ctx)
                self.active = False
                return rect
        self.active = self._pts is not None      # still editable
        return rect

    def finish(self, ctx):
        return self._commit(ctx)

    def _commit(self, ctx):
        if self._session is None:
            self._pts = None
            self.active = False
            return None
        cmd = self._session.commit()
        self._session = None
        self._pts = None
        self.active = False
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        rect = None
        if self._session is not None:
            rect = self._session.cancel()
        self._session = None
        self._pts = None
        self._grab = None
        return rect

    def _arrow(self, tip, toward, size):
        from .. import paths as P
        tx, ty = tip
        dx, dy = tx - toward[0], ty - toward[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        ux, uy = dx / L, dy / L
        bx, by = tx - ux * size, ty - uy * size
        nx, ny = -uy * size * 0.55, ux * size * 0.55
        return P.Subpath([P.Anchor(tx + ux * size * 0.15, ty + uy * size * 0.15),
                          P.Anchor(bx + nx, by + ny), P.Anchor(bx - nx, by - ny)], closed=True)

    def _render(self, ctx, event):
        from .. import paths as P
        cell = ctx.cell
        p0, c0, c1, p1 = self._pts
        stroke = float(self.option("stroke", 2))
        aa = bool(self.option("antialias", True))
        line = P.Path("", [P.Subpath([P.Anchor(p0[0], p0[1], ox=c0[0], oy=c0[1]),
                                      P.Anchor(p1[0], p1[1], ix=c1[0], iy=c1[1])])])
        cov = P.stroke_mask(line, cell.width, cell.height, stroke, aa)
        arrows = self.option("arrows", "none")
        size = max(10.0, stroke * 5.0)
        heads = []

        def toward(end, ctrl, other_ctrl, other_end):
            for q in (ctrl, other_ctrl, other_end):
                if math.hypot(q[0] - end[0], q[1] - end[1]) > 1e-6:
                    return q
            return end
        if arrows in ("end", "both"):
            heads.append(self._arrow(p1, toward(p1, c1, c0, p0), size))
        if arrows in ("start", "both"):
            heads.append(self._arrow(p0, toward(p0, c0, c1, p1), size))
        heads = [h for h in heads if h is not None]
        if heads:
            cov = np.maximum(cov, P.fill_mask(P.Path("", heads), cell.width, cell.height, aa))
        ys, xs = np.nonzero(cov)
        if ys.size == 0:
            return None
        rect = Rect.from_bounds(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self._session.touch(rect)
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = self._colour if self._colour is not None else ctx.primary
        cell.apply_masked(rect, src, ctx.coverage(cov[rect.slice()], rect))
        return rect

    def overlay(self):
        if self._pts is None:
            return []
        p0, c0, c1, p1 = self._pts
        return [("line", (p0[0], p0[1], c0[0], c0[1])), ("line", (p1[0], p1[1], c1[0], c1[1])),
                ("handles", [p0, c0, c1, p1])]
