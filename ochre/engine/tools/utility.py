# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tools that look, mark and plan rather than paint.

  Measure        drag to measure distance and angle (GIMP's Measure,
                 Photoshop's Ruler); Image > Straighten levels the line.
  Color Sampler  up to ten fixed points whose colour the Info dock keeps
                 reading out while you work.
  Count          click to drop numbered markers -- for counting things.
  Note           pin a text note to a spot in the image.
  Slice          drag rectangles to cut the image into pieces for File >
                 Export Slices.

The marks themselves live on the Document (engine/annotations.py) and the
canvas draws them whatever tool is in hand, as Photoshop's Extras are. On
every one: drag a mark to move it, Alt-click to delete it. Each change is one
undo step (MarksDelta). None of them write pixels.
"""

from .. import annotations as A
from ..commands import MarksDelta
from . import options as O
from .base import MOD_ALT, MOD_SHIFT, Tool, constrain_angle


def _nearest(points, x, y, radius):
    best, idx = radius * radius, None
    for i, p in enumerate(points):
        d = (p[0] - x) ** 2 + (p[1] - y) ** 2
        if d <= best:
            best, idx = d, i
    return idx


class _MarkTool(Tool):
    """Shared: one list of points on the document; add, drag, Alt-delete."""

    wants_stroke = False
    attr = "counts"
    limit = None

    def __init__(self, **options):
        super().__init__(**options)
        self._before = None
        self._drag = None

    def _make(self, x, y):
        return (x, y)

    def _moved(self, item, x, y):
        return (x, y) + tuple(item[2:])

    def on_begin(self, ctx, event):
        doc = ctx.doc
        self._before = MarksDelta(doc, self.label, (self.attr,))
        items = list(getattr(doc, self.attr))
        hit = _nearest(items, event.x, event.y, max(ctx.pick_radius, 3.0))
        if hit is not None:
            if event.has(MOD_ALT):
                del items[hit]
                setattr(doc, self.attr, items)
                return doc.bounds
            self._drag = (hit, event.x - items[hit][0], event.y - items[hit][1])
            return None
        if event.has(MOD_ALT):
            return None
        if self.limit is not None and len(items) >= self.limit:
            return {"error": "at most %d %s" % (self.limit, self.label.lower() + "s")}
        item = self._make(event.x, event.y)
        if item is None:
            return None
        items.append(item)
        setattr(doc, self.attr, items)
        self._drag = (len(items) - 1, 0.0, 0.0)
        return doc.bounds

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        i, ox, oy = self._drag
        items = list(getattr(ctx.doc, self.attr))
        if i >= len(items):
            return None
        items[i] = self._moved(items[i], event.x - ox, event.y - oy)
        setattr(ctx.doc, self.attr, items)
        return ctx.doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        before, self._before = self._before, None
        if before is not None and ctx.history is not None and not before.same_as(ctx.doc):
            ctx.history.push(before, self.label)
        return ctx.doc.bounds

    def on_cancel(self, ctx):
        if self._before is not None:
            self._before.undo(ctx.doc)
        self._before = None
        self._drag = None
        return ctx.doc.bounds


class CountTool(_MarkTool):
    """Click to drop a numbered marker; Alt-click one to remove it."""

    name = "count"
    label = "Count"
    attr = "counts"


class ColorSamplerTool(_MarkTool):
    """Fixed sample points whose colours the Info dock keeps reading out."""

    name = "color_sampler"
    label = "Color Sampler"
    attr = "samplers"
    limit = A.MAX_SAMPLERS
    options_spec = (
        O.Opt("sample_size", O.CHOICE, "1", "Sample",
              choices=("1", "3", "5", "11", "31"),
              labels={"1": "Point", "3": "3x3 average", "5": "5x5 average",
                      "11": "11x11 average", "31": "31x31 average"}),
    )


class NoteTool(_MarkTool):
    """Pin a text note to the image. Click a note to edit it.

    Typing needs a text box, which is the UI's business: the tool says what
    it wants through `request`, which the controller hands on to the window.
    """

    name = "note"
    label = "Note"
    attr = "notes"

    def __init__(self, **options):
        super().__init__(**options)
        self.request = None
        self._pressed = None

    def _make(self, x, y):
        # A new note is not added until it has text: the window asks for it
        # and calls Controller.add_note, so cancelling leaves nothing behind.
        self.request = ("new_note", (x, y))
        return None

    def on_begin(self, ctx, event):
        self._pressed = (event.x, event.y)
        return super().on_begin(ctx, event)

    def on_end(self, ctx, event):
        if self._drag is not None and self._pressed is not None:
            moved = abs(event.x - self._pressed[0]) + abs(event.y - self._pressed[1]) >= 2
            if not moved:
                self.request = ("edit_note", self._drag[0])
        return super().on_end(ctx, event)


class SliceTool(Tool):
    """Drag a rectangle to make a slice; drag inside one to move it."""

    name = "slice"
    label = "Slice"
    wants_stroke = False

    def __init__(self, **options):
        super().__init__(**options)
        self._before = None
        self._mode = None
        self._band = None

    def on_begin(self, ctx, event):
        doc = ctx.doc
        self._before = MarksDelta(doc, self.label, ("slices",))
        for i in range(len(doc.slices) - 1, -1, -1):
            name, (x, y, w, h) = doc.slices[i]
            if x <= event.x < x + w and y <= event.y < y + h:
                if event.has(MOD_ALT):
                    doc.slices = doc.slices[:i] + doc.slices[i + 1:]
                    self._mode = None
                    return doc.bounds
                self._mode = ("move", i, event.x - x, event.y - y)
                return None
        self._mode = ("new", event.x, event.y)
        self._band = (event.x, event.y, event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        mode = self._mode
        if mode is None:
            return None
        doc = ctx.doc
        if mode[0] == "move":
            _m, i, ox, oy = mode
            name, (_x, _y, w, h) = doc.slices[i]
            slices = list(doc.slices)
            slices[i] = (name, (int(round(event.x - ox)), int(round(event.y - oy)), w, h))
            doc.slices = slices
        else:
            x1, y1 = event.x, event.y
            if event.has(MOD_SHIFT):
                side = max(abs(x1 - mode[1]), abs(y1 - mode[2]))
                x1 = mode[1] + (side if x1 >= mode[1] else -side)
                y1 = mode[2] + (side if y1 >= mode[2] else -side)
            self._band = (mode[1], mode[2], x1, y1)
        return doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        doc = ctx.doc
        mode, self._mode = self._mode, None
        if mode is not None and mode[0] == "new" and self._band is not None:
            x0, y0, x1, y1 = self._band
            xa, xb = sorted((int(round(x0)), int(round(x1))))
            ya, yb = sorted((int(round(y0)), int(round(y1))))
            xa, ya = max(0, xa), max(0, ya)
            xb, yb = min(doc.width, xb), min(doc.height, yb)
            if xb - xa >= 1 and yb - ya >= 1:
                n = len(doc.slices) + 1
                names = {s[0] for s in doc.slices}
                while "slice_%02d" % n in names:
                    n += 1
                doc.slices = list(doc.slices) + [("slice_%02d" % n, (xa, ya, xb - xa, yb - ya))]
        self._band = None
        before, self._before = self._before, None
        if before is not None and ctx.history is not None and not before.same_as(doc):
            ctx.history.push(before, self.label)
        return doc.bounds

    def on_cancel(self, ctx):
        if self._before is not None:
            self._before.undo(ctx.doc)
        self._before = None
        self._mode = None
        self._band = None
        return ctx.doc.bounds

    def overlay(self):
        if self._band is None:
            return []
        x0, y0, x1, y1 = self._band
        return [("rect", (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))]


class MeasureTool(Tool):
    """Drag to measure. The line stays until the next drag; drag an end to adjust.

    Shift snaps the angle to 15 degrees. Image > Straighten rotates the image
    so the measured line comes out level (or plumb, if it is nearer
    vertical) -- GIMP's Straighten, Photoshop's Straighten Layer.
    """

    name = "measure"
    label = "Measure"
    wants_stroke = False

    def __init__(self, **options):
        super().__init__(**options)
        self.line = None
        self._grab = None

    @property
    def measurement(self):
        if self.line is None:
            return None
        return A.measure(*self.line)

    def on_begin(self, ctx, event):
        if self.line is not None:
            x0, y0, x1, y1 = self.line
            r = max(ctx.pick_radius, 3.0)
            if (event.x - x1) ** 2 + (event.y - y1) ** 2 <= r * r:
                self._grab = 1
                return None
            if (event.x - x0) ** 2 + (event.y - y0) ** 2 <= r * r:
                self._grab = 0
                return None
        self.line = (event.x, event.y, event.x, event.y)
        self._grab = 1
        return None

    def on_motion(self, ctx, event):
        if self._grab is None or self.line is None:
            return None
        x0, y0, x1, y1 = self.line
        if self._grab == 1:
            nx, ny = event.x, event.y
            if event.has(MOD_SHIFT):
                nx, ny = constrain_angle(x0, y0, nx, ny)
            self.line = (x0, y0, nx, ny)
        else:
            nx, ny = event.x, event.y
            if event.has(MOD_SHIFT):
                nx, ny = constrain_angle(x1, y1, nx, ny)
            self.line = (nx, ny, x1, y1)
        return ctx.doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._grab = None
        if self.line is not None:
            x0, y0, x1, y1 = self.line
            if abs(x1 - x0) < 0.5 and abs(y1 - y0) < 0.5:
                self.line = None
        return ctx.doc.bounds

    def status(self):
        m = self.measurement
        if m is None:
            return ""
        return "%.1f px, %.2f\u00b0  (dx %.1f, dy %.1f)" % (
            m["distance"], m["angle"], m["dx"], m["dy"])

    def overlay(self):
        if self.line is None:
            return []
        x0, y0, x1, y1 = self.line
        m = self.measurement
        return [("line", (x0, y0, x1, y1)), ("handles", [(x0, y0), (x1, y1)]),
                ("text", (x1, y1), "%.1f px  %.1f\u00b0" % (m["distance"], m["angle"]))]


class SliceSelectTool(SliceTool):
    """Move and delete existing slices without making new ones by accident."""

    name = "slice_select"
    label = "Slice Select"

    def on_begin(self, ctx, event):
        result = super().on_begin(ctx, event)
        if self._mode is not None and self._mode[0] == "new":
            self._mode = None
            self._band = None
        return result
