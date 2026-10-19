# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Blur Gallery, on the canvas.

  Field       click to drop pins, each with its own blur (the Blur option
              sets the selected pin); drag a pin to move it, Alt-click to
              remove it. Blur flows smoothly between pins.
  Iris        drag out an ellipse: sharp inside, blurring outside.
  Tilt-Shift  drag along the line that should stay sharp; blur grows away
              from it on both sides.

The layer previews as you go; Enter applies (one undo step), Escape puts
it back.
"""

import json
import math

import numpy as np

from .. import morefilters as MF
from ..stroke import StrokeSession
from . import options as O
from .base import MOD_ALT, Tool


class BlurGalleryTool(Tool):
    name = "blur_gallery"
    label = "Blur Gallery"
    snaps = False
    multi_click = True
    commit_on_switch = True
    options_spec = (
        O.Opt("mode", O.CHOICE, "field", "Blur", choices=("field", "iris", "tilt_shift"),
              labels={"field": "Field", "iris": "Iris", "tilt_shift": "Tilt-Shift"}),
        O.Opt("blur", O.INT, 15, "Amount", 0, 200, suffix=" px"),
        O.Opt("feather", O.INT, 40, "Feather", 0, 100, suffix=" %"),
        O.Opt("band", O.INT, 40, "Sharp band", 0, 2000, suffix=" px",
              tip="Tilt-Shift: how wide the sharp band is."),
    )

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self.pins = []            # field: [[x, y, blur], ...]
        self.selected = None
        self.shape = None         # iris (x0, y0, x1, y1) / tilt-shift (x0, y0, x1, y1)
        self._drag = None
        self._session = None
        self._orig = None

    def _radius(self, ctx):
        return max(5.0, 2.0 * float(getattr(ctx, "pick_radius", 4.0)))

    # ---- events -------------------------------------------------------------------

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            self.active = False
            return None
        if ctx.cell.index_locked:
            # A blur makes colours no palette has -- the effect policy refuses.
            self.active = False
            return {"error": "Blur Gallery needs an RGB layer"}
        mode = self.option("mode", "field")
        if mode == "field":
            r = self._radius(ctx)
            for i, (x, y, _b) in enumerate(self.pins):
                if math.hypot(x - event.x, y - event.y) <= r:
                    if event.has(MOD_ALT):
                        del self.pins[i]
                        self.selected = None
                        self._drag = ("render",)
                    else:
                        self.selected = i
                        self._drag = ("pin", i)
                    return None
            self.pins.append([float(event.x), float(event.y), float(self.option("blur", 15))])
            self.selected = len(self.pins) - 1
            self._drag = ("pin", self.selected)
            return None
        self._drag = ("shape", (event.x, event.y))
        self.shape = (event.x, event.y, event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        if self._drag[0] == "pin":
            p = self.pins[self._drag[1]]
            p[0], p[1] = float(event.x), float(event.y)
        elif self._drag[0] == "shape":
            x0, y0 = self._drag[1]
            self.shape = (x0, y0, event.x, event.y)
        return None

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        self.active = True
        return self._render(ctx)

    def options_changed(self, ctx):
        """A new Amount sets the selected pin; Feather and Band re-render."""
        if self.option("mode", "field") == "field" and self.selected is not None \
                and self.selected < len(self.pins):
            self.pins[self.selected][2] = float(self.option("blur", 15))
        return self._render(ctx)

    # ---- rendering ------------------------------------------------------------------

    def blurred(self, src):
        """src through the current settings (also what Enter applies)."""
        mode = self.option("mode", "field")
        h, w = src.shape[:2]
        amount = float(self.option("blur", 15))
        if mode == "field":
            return MF.field_blur(src, self.pins) if self.pins else src.copy()
        if self.shape is None:
            return src.copy()
        x0, y0, x1, y1 = self.shape
        if mode == "iris":
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
            if rx < 2 or ry < 2:
                return src.copy()
            fx = MF.IrisBlur()
            return fx.compute({"center_x": cx / w * 100.0, "center_y": cy / h * 100.0,
                               "radius_x": rx / w * 100.0, "radius_y": ry / h * 100.0,
                               "angle": 0.0, "feather": float(self.option("feather", 40)),
                               "blur": amount}, src)
        # Tilt-shift: distance from the dragged line, sharp within band/2.
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy)
        if n < 2:
            return src.copy()
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        d = np.abs((xs + 0.5 - x0) * dy - (ys + 0.5 - y0) * dx) / n
        half = float(self.option("band", 40)) / 2.0
        ramp = max(1.0, half * (1 + float(self.option("feather", 40)) / 25.0))
        m = np.clip((d - half) / ramp, 0, 1)
        return MF._blend_by(src, MF._blur_rgba(src, amount), m)

    def _render(self, ctx):
        cell = ctx.cell
        if cell is None:
            return None
        if self._session is None:
            self._session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)
            self._session.touch(ctx.doc.bounds)
            self._orig = cell.plane("rgba").copy()
        cell.plane("rgba")[...] = self.blurred(self._orig)
        cell._touched(ctx.doc.bounds)
        return ctx.doc.bounds

    def finish(self, ctx):
        self.active = False
        session = self._session
        self._reset()
        if session is None:
            return None
        cmd = session.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return ctx.doc.bounds

    def on_cancel(self, ctx):
        rect = self._session.cancel() if self._session is not None else None
        self._reset()
        return rect or ctx.doc.bounds

    def pins_json(self):
        return json.dumps(self.pins)

    # ---- drawing ------------------------------------------------------------------------

    def overlay(self):
        out = []
        mode = self.option("mode", "field")
        if mode == "field":
            for i, (x, y, b) in enumerate(self.pins):
                r = 8 if i == self.selected else 6
                out.append(("poly", [(x + r * math.cos(t), y + r * math.sin(t))
                                     for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)], True))
                out.append(("text", (x + 10, y - 10), "%d px" % int(b)))
            return out
        if self.shape is None:
            return out
        x0, y0, x1, y1 = self.shape
        if mode == "iris":
            cx, cy, rx, ry = (x0 + x1) / 2.0, (y0 + y1) / 2.0, abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
            out.append(("poly", [(cx + rx * math.cos(t), cy + ry * math.sin(t))
                                 for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)], True))
            return out
        out.append(("line", (x0, y0, x1, y1)))
        return out
