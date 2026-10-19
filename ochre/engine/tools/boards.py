# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Artboard and Frame tools (engine/regions.py has the model).

  Artboard  drag on empty canvas to make an artboard; drag an artboard to
            move it with everything on it; click one to select it.
  Frame     drag to make a rectangular or elliptical frame. Drawn while an
            image layer is active, that layer goes into the frame and is
            cropped to it; otherwise the frame starts empty and File > Place
            Embedded fills it. Drag a frame to move it; click to select.

Neither tool writes pixels or touches the tree itself: each says what it
wants through `request`, which the controller carries out as one undo step.
"""

import math

from .. import regions
from . import options as O
from .base import MOD_SHIFT, Tool


class _RegionTool(Tool):
    kind = None
    wants_stroke = False
    commit_on_switch = False

    def __init__(self, **options):
        super().__init__(**options)
        self._mode = None
        self._band = None
        self.request = None

    def _shape(self):
        return "rect"

    def on_begin(self, ctx, event):
        hit = regions.at(ctx.doc, event.x, event.y, self.kind)
        if hit is not None:
            x, y, w, h = hit.region["rect"]
            self._mode = ("move", hit.id, event.x, event.y)
            self._band = (x, y, x + w, y + h)
        else:
            self._mode = ("new", event.x, event.y)
            self._band = (event.x, event.y, event.x, event.y)
        return None

    def _delta(self, event, mode=None):
        mode = mode or self._mode
        return event.x - mode[2], event.y - mode[3]

    def on_motion(self, ctx, event):
        if self._mode is None:
            return None
        if self._mode[0] == "move":
            group = ctx.doc.layer(self._mode[1])
            x, y, w, h = group.region["rect"]
            dx, dy = (int(round(v)) for v in self._delta(event))
            self._band = (x + dx, y + dy, x + w + dx, y + h + dy)
        else:
            _m, x0, y0 = self._mode
            x1, y1 = event.x, event.y
            if event.has(MOD_SHIFT):
                side = max(abs(x1 - x0), abs(y1 - y0))
                x1 = x0 + (side if x1 >= x0 else -side)
                y1 = y0 + (side if y1 >= y0 else -side)
            self._band = (x0, y0, x1, y1)
        return ctx.doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        mode, self._mode = self._mode, None
        band, self._band = self._band, None
        if mode is None or band is None:
            return ctx.doc.bounds
        if mode[0] == "move":
            dx, dy = (int(round(v)) for v in self._delta(event, mode))
            if dx == 0 and dy == 0:
                self.request = ("select_layer", mode[1])
            else:
                self.request = ("move_region", (mode[1], dx, dy))
            return ctx.doc.bounds
        x0, y0, x1, y1 = band
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(ctx.doc.width, xb), min(ctx.doc.height, yb)
        if xb - xa >= 2 and yb - ya >= 2:
            self.request = ("add_region", self._region_args(ctx, (xa, ya, xb - xa, yb - ya)))
        return ctx.doc.bounds

    def _region_args(self, ctx, rect):
        return {"kind": self.kind, "rect": rect, "shape": self._shape()}

    def on_cancel(self, ctx):
        self._mode = None
        self._band = None
        return ctx.doc.bounds

    def overlay(self):
        if self._band is None:
            return []
        x0, y0, x1, y1 = self._band
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        if self._shape() == "ellipse" and (self._mode is None or self._mode[0] == "new"):
            cx, cy, rx, ry = (xa + xb) / 2.0, (ya + yb) / 2.0, (xb - xa) / 2.0, (yb - ya) / 2.0
            pts = [(cx + rx * math.cos(t * math.pi / 24), cy + ry * math.sin(t * math.pi / 24))
                   for t in range(48)]
            return [("poly", pts, True)]
        return [("rect", (xa, ya, xb, yb))]


class ArtboardTool(_RegionTool):
    """Photoshop's Artboard tool."""

    name = "artboard"
    label = "Artboard"
    kind = regions.ARTBOARD
    options_spec = (
        O.Opt("background", O.CHOICE, "white", "Background",
              choices=("white", "black", "transparent")),
    )
    _COLOURS = {"white": (255, 255, 255, 255), "black": (0, 0, 0, 255), "transparent": None}

    def _region_args(self, ctx, rect):
        args = super()._region_args(ctx, rect)
        args["background"] = self._COLOURS.get(self.option("background", "white"))
        return args


class FrameTool(_RegionTool):
    """Photoshop's Frame tool."""

    name = "frame"
    label = "Frame"
    kind = regions.FRAME
    options_spec = (
        O.Opt("shape", O.CHOICE, "rect", "Shape", choices=("rect", "ellipse"),
              labels={"rect": "Rectangle", "ellipse": "Ellipse"}),
        O.Opt("adopt", O.BOOL, True, "Put the active layer in it",
              tip="Drawn over an image layer, the frame takes that layer in and crops it."),
    )

    def _shape(self):
        return self.option("shape", "rect")

    def _region_args(self, ctx, rect):
        args = super()._region_args(ctx, rect)
        layer = ctx.layer
        if self.option("adopt", True) and layer is not None and not layer.is_group \
                and getattr(layer, "kind", "raster") in ("raster", "smart") \
                and regions.frame_of(layer) is None and not _is_bottom(ctx.doc, layer):
            cell = ctx.doc.cells.get((layer.id, ctx.frame.id))
            if getattr(layer, "kind", "raster") == "smart" or (cell is not None and not cell.is_empty()):
                args["adopt"] = layer.id
        return args


def _is_bottom(doc, layer):
    """The bottom layer is the backdrop (Photoshop's Background), never
    swallowed by a frame drawn over it."""
    leaves = doc.layers()
    return bool(leaves) and leaves[0] is layer
