# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Perspective Warp tool (engine/perspwarp.py has the geometry).

  Layout  drag rectangles over the planes of the picture; a corner dropped
          near another quad's corner joins it, so planes share edges. Drag
          a corner to fit it to the picture.
  Warp    drag corners to the new perspective -- joined corners move
          together. The layer follows on release.
  Enter   applies (one undo step); Escape puts the layer back.
"""

import math

from .. import perspwarp as PW

from ..stroke import StrokeSession
from . import options as O
from .base import Tool


class PerspectiveWarpTool(Tool):
    name = "perspective_warp"
    label = "Perspective Warp"
    snaps = False
    multi_click = True
    commit_on_switch = True
    options_spec = (O.Opt("mode", O.CHOICE, "layout", "Mode", choices=("layout", "warp"),
                          labels={"layout": "Layout", "warp": "Warp"}),)

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self._layout = []
        self._warp = []
        self._drag = None
        self._band = None
        self._session = None
        self._orig = None

    def _radius(self, ctx):
        return max(4.0, 1.6 * float(getattr(ctx, "pick_radius", 4.0)))

    def _hit(self, ctx, x, y, quads):
        r = self._radius(ctx)
        for qi, q in enumerate(quads):
            for vi, (px, py) in enumerate(q):
                if math.hypot(px - x, py - y) <= r:
                    return qi, vi
        return None

    def _shared(self, qi, vi):
        """Every (quad, vertex) at the same layout point -- joined corners."""
        px, py = self._layout[qi][vi]
        return [(a, b) for a, q in enumerate(self._layout) for b, (x, y) in enumerate(q)
                if abs(x - px) < 0.5 and abs(y - py) < 0.5]

    # ---- events -----------------------------------------------------------------

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            self.active = False
            return None
        mode = self.option("mode", "layout")
        quads = self._layout if mode == "layout" else self._warp
        hit = self._hit(ctx, event.x, event.y, quads)
        if hit is not None:
            self._drag = ("vertex", self._shared(*hit), mode)
        elif mode == "layout":
            self._drag = ("new", (event.x, event.y))
            self._band = (event.x, event.y, event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        if self._drag[0] == "new":
            x0, y0 = self._drag[1]
            self._band = (x0, y0, event.x, event.y)
            return None
        _k, group, mode = self._drag
        for qi, vi in group:
            self._warp[qi][vi] = (float(event.x), float(event.y))
            if mode == "layout":
                self._layout[qi][vi] = (float(event.x), float(event.y))
        return None

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        drag, self._drag = self._drag, None
        self.active = True                 # stays open until Enter / Escape
        if drag is None:
            return None
        if drag[0] == "new":
            x0, y0, x1, y1 = self._band
            self._band = None
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            if xb - xa < 4 or yb - ya < 4:
                return None
            corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
            # A corner dropped near an existing one joins it.
            r = self._radius(ctx) * 2
            snapped = []
            for cx, cy in corners:
                best = None
                for q in self._layout:
                    for px, py in q:
                        d = math.hypot(px - cx, py - cy)
                        if d <= r and (best is None or d < best[0]):
                            best = (d, (px, py))
                snapped.append(best[1] if best else (float(cx), float(cy)))
            self._layout.append([tuple(p) for p in snapped])
            self._warp.append([tuple(p) for p in snapped])
            return None
        if drag[2] == "warp":
            return self._render(ctx)
        return None

    def _render(self, ctx):
        cell = ctx.cell
        # The AUTHORITATIVE plane: indices on an index-locked layer.
        plane = "index" if cell.index_locked else "rgba"
        if self._session is None:
            self._session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)
            self._session.touch(ctx.doc.bounds)
            self._orig = cell.plane(plane).copy()
        if plane == "index":
            from ..transform import _transparent_for
            out = PW.render_index(self._orig, self._layout, self._warp, _transparent_for(cell))
        else:
            out = PW.render(self._orig, self._layout, self._warp)
        cell.plane(plane)[...] = out
        cell._touched(ctx.doc.bounds)
        cell.refresh_derived()
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
        rect = None
        if self._session is not None:
            rect = self._session.cancel()
        self._reset()
        return rect or ctx.doc.bounds

    # ---- drawing -------------------------------------------------------------------

    def overlay(self):
        mode = self.option("mode", "layout")
        quads = self._layout if mode == "layout" else self._warp
        out = []
        for q in quads:
            out.append(("poly", [tuple(p) for p in q], True))
            for seg in PW.quad_grid(q):
                out.append(("line", seg))
            out.append(("handles", [tuple(p) for p in q]))
        if self._band is not None:
            x0, y0, x1, y1 = self._band
            out.append(("rect", (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))
        return out



