# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Painting tools: pencil, brush, eraser, bucket fill, eyedropper.

Every one of these routes its pixels through Surface.apply_masked with the
selection fused into its own coverage, so selection clipping is not something
any of them implements -- it is something none of them can avoid.

Dab spacing is in units of brush diameter, so a fast drag lays down evenly
spaced dabs rather than a dotted line, and a slow one does not waste work
stacking hundreds of dabs on the same pixel.
"""

import math

import numpy as np

from .. import accel
from ..geometry import Rect
from ..stroke import StrokeSession
from .base import Tool, constrain_angle


def _circle_coverage(diameter, hardness=1.0, antialias=True):
    """A round dab's coverage, 0..255.

    hardness 1.0 is a hard edge with a single pixel of antialiasing; lower
    values ramp the falloff inward from the rim.
    """
    d = max(1, int(round(diameter)))
    if d == 1:
        return np.full((1, 1), 255, dtype=np.uint8)
    r = d / 2.0
    ys = np.arange(d, dtype=np.float64)[:, None] + 0.5 - r
    xs = np.arange(d, dtype=np.float64)[None, :] + 0.5 - r
    dist = np.hypot(xs, ys)
    if not antialias:
        return (dist <= r).astype(np.uint8) * 255
    hardness = min(max(float(hardness), 0.0), 1.0)
    inner = r * hardness
    if inner >= r:
        cov = np.clip(r - dist + 0.5, 0.0, 1.0)         # 1px analytic edge
    else:
        cov = np.clip((r - dist) / max(r - inner, 1e-6), 0.0, 1.0)
    return np.rint(cov * 255.0).astype(np.uint8)


class _StrokeTool(Tool):
    """Shared machinery: open a session, lay dabs, commit one entry."""

    default_size = 8

    def _begin_session(self, ctx):
        s = ctx.settings
        block = 256 if s is None else s.get("History", "BlockSize")
        max_blocks = 512 if s is None else s.get("History", "MaxStrokeBlocks")
        level = 1 if s is None else s.get("History", "CompressLevel")
        return StrokeSession(ctx.doc, ctx.layer, ctx.frame,
                             block=block, max_blocks=max_blocks,
                             level=level, label=self.label)

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            return None
        self._session = self._begin_session(ctx)
        self._last = (event.x, event.y)
        return self._dab(ctx, event, event.x, event.y)

    def on_motion(self, ctx, event):
        if self._session is None:
            return None
        x0, y0 = self._last
        x1, y1 = event.x, event.y
        if event.has(1):                       # Shift constrains, live
            x1, y1 = constrain_angle(x0, y0, x1, y1)

        size = float(self.option("size", self.default_size))
        spacing = max(0.05, float(self.option("spacing", 0.25))) * max(size, 1.0)
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < spacing:
            return None

        covered = None
        steps = int(dist / spacing)
        for i in range(1, steps + 1):
            t = (i * spacing) / dist
            r = self._dab(ctx, event, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            covered = r if covered is None else covered.union(r)
        self._last = (x0 + (x1 - x0) * (steps * spacing / dist),
                      y0 + (y1 - y0) * (steps * spacing / dist))
        return covered

    def on_end(self, ctx, event):
        if self._session is None:
            return None
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

    def _dab(self, ctx, event, x, y):
        raise NotImplementedError


class PencilTool(_StrokeTool):
    """Hard single-pixel-precision drawing. No antialiasing, ever.

    Aliasing is the feature here: pixel artists need to know exactly which
    pixels changed, and a soft edge on a 60x60 sprite is a defect.
    """

    name = "pencil"
    label = "Pencil"
    default_size = 1

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1, int(self.option("size", self.default_size)))
        half = size // 2
        ix, iy = int(math.floor(x)), int(math.floor(y))
        rect = Rect(ix - half, iy - half, size, size).clipped_to(
            cell.width, cell.height)
        if rect is None:
            return None
        self._session.touch(rect)

        index = ctx.index_for(event)
        if index is not None and cell.index_locked:
            cov = ctx.coverage(None, rect)
            if cov is None:
                cell.fill(rect, index, name="index")
            else:
                src = np.full((rect.h, rect.w), index, dtype=np.uint8)
                cell.apply_masked(rect, src, (cov >= 128).astype(np.uint8) * 255,
                                  name="index")
            return rect

        colour = ctx.colour_for(event)
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = colour
        cov = ctx.coverage(None, rect)
        if cov is not None:
            cov = (cov >= 128).astype(np.uint8) * 255      # keep it hard
        cell.apply_masked(rect, src, cov)
        return rect


class BrushTool(_StrokeTool):
    """Antialiased round brush with hardness and flow."""

    name = "brush"
    label = "Paintbrush"
    default_size = 16

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1, float(self.option("size", self.default_size)))
        if self.option("pressure_size", True) and event.pressure < 1.0:
            size = max(1.0, size * event.pressure)
        cov = _circle_coverage(size, float(self.option("hardness", 0.85)),
                               antialias=True)
        d = cov.shape[0]
        ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
        rect = Rect(ox, oy, d, d)
        clipped = rect.clipped_to(cell.width, cell.height)
        if clipped is None:
            return None
        cov = cov[clipped.y - oy:clipped.y - oy + clipped.h,
                  clipped.x - ox:clipped.x - ox + clipped.w]

        flow = float(self.option("flow", 1.0))
        if flow < 1.0:
            cov = accel.spec._mul(cov.astype(np.uint32),
                                  np.uint32(int(flow * 255))).astype(np.uint8)
        self._session.touch(clipped)

        src = np.empty((clipped.h, clipped.w, 4), dtype=np.uint8)
        src[:] = ctx.colour_for(event)
        cell.apply_masked(clipped, src, ctx.coverage(cov, clipped))
        return clipped


class EraserTool(BrushTool):
    """Removes alpha instead of adding colour."""

    name = "eraser"
    label = "Eraser"

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1, float(self.option("size", self.default_size)))
        cov = _circle_coverage(size, float(self.option("hardness", 0.85)))
        d = cov.shape[0]
        ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
        clipped = Rect(ox, oy, d, d).clipped_to(cell.width, cell.height)
        if clipped is None:
            return None
        cov = cov[clipped.y - oy:clipped.y - oy + clipped.h,
                  clipped.x - ox:clipped.x - ox + clipped.w]
        cov = ctx.coverage(cov, clipped)
        self._session.touch(clipped)

        plane = cell.plane("rgba")[clipped.slice()]
        keep = (255 - (cov if cov is not None else 255)).astype(np.uint8)
        plane[..., 3] = accel.spec._mul(plane[..., 3].astype(np.uint32),
                                        keep.astype(np.uint32)).astype(np.uint8)
        cell._touched(clipped)
        return clipped


class BucketFillTool(Tool):
    """Flood fill with Paint.NET's alpha-aware tolerance metric."""

    name = "bucket"
    label = "Paint Bucket"

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            return None
        tolerance = int(self.option("tolerance", 32))
        # The selection is enforced by pre-seeding its complement as already
        # visited, so the fill physically cannot escape rather than being
        # tested per pixel.
        blocked = None
        if ctx.selection is not None and not ctx.selection.selects_all():
            blocked = ctx.selection.mask < 128

        stencil = accel.flood_fill(cell.pixels, event.ix, event.iy,
                                   tolerance, mask=blocked)
        if not stencil.any():
            return None
        ys, xs = np.nonzero(stencil)
        rect = Rect.from_points(int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max()))

        session = self._begin_session(ctx)
        session.touch(rect)
        cov = (stencil[rect.slice()].astype(np.uint8)) * 255
        cov = ctx.coverage(cov, rect)
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = ctx.colour_for(event)
        cell.apply_masked(rect, src, cov)

        cmd = session.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        self.active = False
        return rect

    def _begin_session(self, ctx):
        s = ctx.settings
        return StrokeSession(ctx.doc, ctx.layer, ctx.frame,
                             block=256 if s is None else s.get("History", "BlockSize"),
                             label=self.label)


class EyedropperTool(Tool):
    """Picks a colour, and an index when the layer is index-locked.

    A spriter needs to know they picked index 17, not #A85C20 -- so when an
    index plane exists it is the authoritative answer and the RGB is derived.
    """

    name = "eyedropper"
    label = "Color Picker"
    wants_stroke = False

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            return None
        if not cell.bounds.contains(event.ix, event.iy):
            return None
        picked_index = None
        if cell.index_locked:
            picked_index = int(cell.plane("index")[event.iy, event.ix])
        rgba = tuple(int(v) for v in cell.pixels[event.iy, event.ix])
        if event.button == 2:
            ctx.secondary = rgba
            ctx.secondary_index = picked_index
        else:
            ctx.primary = rgba
            ctx.primary_index = picked_index
        self.active = False
        return {"rgba": rgba, "index": picked_index}
