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
  Spin        click to drop a spin (drag out to size it), click its centre
              to select or drag it, Alt-click to remove it. Blur Angle and
              Feather set the selected spin.
  Path        click to pin stop points; the path is the curve through
              them. Shift-click starts another path; drag a point to move
              it, Alt-click to remove it. Amount sets the selected path's
              end point speeds.

The layer previews as you go; Enter applies (one undo step), Escape puts
it back.
"""

import json
import math

import numpy as np

from .. import galleryextra as GE
from .. import morefilters as MF
from ..stroke import StrokeSession
from . import options as O
from .base import MOD_ALT, MOD_SHIFT, Tool


class BlurGalleryTool(Tool):
    name = "blur_gallery"
    label = "Blur Gallery"
    snaps = False
    multi_click = True
    commit_on_switch = True
    options_spec = (
        O.Opt("mode", O.CHOICE, "field", "Blur",
              choices=("field", "iris", "tilt_shift", "spin", "path"),
              labels={"field": "Field", "iris": "Iris", "tilt_shift": "Tilt-Shift",
                      "spin": "Spin", "path": "Path"}),
        O.Opt("blur", O.INT, 15, "Amount", 0, 200, suffix=" px"),
        O.Opt("feather", O.INT, 40, "Feather", 0, 100, suffix=" %"),
        O.Opt("band", O.INT, 40, "Sharp band", 0, 2000, suffix=" px",
              tip="Tilt-Shift: how wide the sharp band is."),
        O.Opt("blur_angle", O.INT, 15, "Blur Angle", 0, 360, suffix=" deg",
              tip="Spin: how far the selected spin turns."),
    )

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self.pins = []            # field: [[x, y, blur], ...]
        self.spins = []           # spin: [[x, y, rx, ry, angle, feather, blur_angle, 0, 0], ...]
        self.paths = []           # path: [{"points": [[x, y], ...], "speed": px}, ...]
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
        if mode == "spin":
            return self._begin_spin(ctx, event)
        if mode == "path":
            return self._begin_path(ctx, event)
        self._drag = ("shape", (event.x, event.y))
        self.shape = (event.x, event.y, event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        if self._drag[0] == "pin":
            p = self.pins[self._drag[1]]
            p[0], p[1] = float(event.x), float(event.y)
        elif self._drag[0] == "spin":
            s = self.spins[self._drag[1]]
            s[0], s[1] = float(event.x), float(event.y)
        elif self._drag[0] == "spin_size":
            _, i, x0, y0 = self._drag
            r = math.hypot(event.x - x0, event.y - y0)
            if r >= 3:
                self.spins[i][2] = self.spins[i][3] = r
        elif self._drag[0] == "point":
            _, i, k = self._drag
            self.paths[i]["points"][k] = [float(event.x), float(event.y)]
        elif self._drag[0] == "shape":
            x0, y0 = self._drag[1]
            self.shape = (x0, y0, event.x, event.y)
        return None

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        self.active = True
        return self._render(ctx)

    def _begin_spin(self, ctx, event):
        r = self._radius(ctx)
        for i, s in enumerate(self.spins):
            if math.hypot(s[0] - event.x, s[1] - event.y) <= r:
                if event.has(MOD_ALT):
                    del self.spins[i]
                    self.selected = None
                    self._drag = ("render",)
                else:
                    self.selected = i
                    self._drag = ("spin", i)
                return None
        if len(self.spins) >= GE.MAX_BLUR_PINS:
            return None
        # Kooka's opening spin: a circle 0.13 of the width across.
        radius = 0.13 * ctx.doc.bounds.w
        self.spins.append([float(event.x), float(event.y), radius, radius, 0.0,
                           float(self.option("feather", 40)) / 100.0,
                           float(self.option("blur_angle", 15)), 0.0, 0.0])
        self.selected = len(self.spins) - 1
        self._drag = ("spin_size", self.selected, float(event.x), float(event.y))
        return None

    def _begin_path(self, ctx, event):
        r = self._radius(ctx)
        for i, path in enumerate(self.paths):
            for k, (x, y) in enumerate(path["points"]):
                if math.hypot(x - event.x, y - event.y) <= r:
                    if event.has(MOD_ALT):
                        del path["points"][k]
                        if not path["points"]:
                            del self.paths[i]
                        self.selected = None if not path["points"] else i
                        self._drag = ("render",)
                    else:
                        self.selected = i
                        self._drag = ("point", i, k)
                    return None
        point = [float(event.x), float(event.y)]
        if (event.has(MOD_SHIFT) or self.selected is None or self.selected >= len(self.paths)
                or len(self.paths[self.selected]["points"]) >= GE.MAX_PATH_POINTS):
            if len(self.paths) >= GE.MAX_BLUR_PINS:
                return None
            self.paths.append({"points": [], "speed": float(self.option("blur", 15))})
            self.selected = len(self.paths) - 1
        pts = self.paths[self.selected]["points"]
        pts.append(point)
        self._drag = ("point", self.selected, len(pts) - 1)
        return None

    def options_changed(self, ctx):
        """A new Amount sets the selected pin (Field) or path's end speeds
        (Path), Blur Angle and Feather the selected spin; the rest re-render."""
        mode = self.option("mode", "field")
        sel = self.selected
        if mode == "field" and sel is not None and sel < len(self.pins):
            self.pins[sel][2] = float(self.option("blur", 15))
        elif mode == "spin" and sel is not None and sel < len(self.spins):
            self.spins[sel][5] = float(self.option("feather", 40)) / 100.0
            self.spins[sel][6] = float(self.option("blur_angle", 15))
        elif mode == "path" and sel is not None and sel < len(self.paths):
            self.paths[sel]["speed"] = float(self.option("blur", 15))
        return self._render(ctx)

    # ---- rendering ------------------------------------------------------------------

    def blurred(self, src):
        """src through the current settings (also what Enter applies)."""
        mode = self.option("mode", "field")
        h, w = src.shape[:2]
        amount = float(self.option("blur", 15))
        if mode == "field":
            return MF.field_blur(src, self.pins) if self.pins else src.copy()
        if mode == "spin":
            return GE.spin_blur(src, self.spins_fractions(w, h))
        if mode == "path":
            return GE.path_blur(src, self.paths_fractions(w, h), speed=100)
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

    def spins_fractions(self, w, h):
        """The spins as Spin Blur's `spins` JSON takes them (fractions)."""
        return [[x / w, y / h, rx / w, ry / w, a, f, b, px / w, py / h]
                for x, y, rx, ry, a, f, b, px, py in self.spins]

    def paths_fractions(self, w, h):
        """The paths with two points or more, as Path Blur's `paths` JSON
        takes them (fractions; the speed at both ends)."""
        return [{"points": [[x / w, y / h] for x, y in p["points"]],
                 "start_speed": p["speed"], "end_speed": p["speed"]}
                for p in self.paths if len(p["points"]) >= 2]

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
        ring = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        if mode == "spin":
            for i, (x, y, rx, ry, _a, f, b, _px, _py) in enumerate(self.spins):
                out.append(("poly", [(x + rx * math.cos(t), y + ry * math.sin(t)) for t in ring], True))
                if f > 0:
                    out.append(("poly", [(x + rx * f * math.cos(t), y + ry * f * math.sin(t))
                                         for t in ring[::2]], True))
                r = 8 if i == self.selected else 6
                out.append(("poly", [(x + r * math.cos(t), y + r * math.sin(t))
                                     for t in ring[::3]], True))
                out.append(("text", (x + 10, y - 10), "%d deg" % int(b)))
            return out
        if mode == "path":
            for i, p in enumerate(self.paths):
                pts = p["points"]
                if len(pts) >= 2:
                    out.append(("poly", [tuple(q) for q in GE.catmull_rom(pts)], False))
                r = 6 if i == self.selected else 4
                for x, y in pts:
                    out.append(("poly", [(x + r * math.cos(t), y + r * math.sin(t))
                                         for t in ring[::3]], True))
                if pts:
                    out.append(("text", (pts[-1][0] + 10, pts[-1][1] - 10), "%d px" % int(p["speed"])))
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
