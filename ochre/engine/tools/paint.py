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
from . import options as O


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

    snaps = False

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
    options_spec = (O.size(1, 256),)
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
    options_spec = (O.size(16), O.hardness(0.85), O.opacity(1.0),
                    O.Opt("pressure_size", O.BOOL, True, "Pressure size",
                          tip="Tablet pressure scales the brush."),
                    # Photoshop's Shape Dynamics and Scattering, the parts
                    # that make a brush feel like more than a stamp.
                    O.Opt("size_jitter", O.INT, 0, "Size jitter", 0, 100, 5, suffix=" %"),
                    O.Opt("opacity_jitter", O.INT, 0, "Opacity jitter", 0, 100, 5, suffix=" %"),
                    O.Opt("scatter", O.INT, 0, "Scatter", 0, 500, 10, suffix=" %",
                          tip="How far dabs stray from the stroke, as a share of the size."),
                    O.Opt("count", O.INT, 1, "Count", 1, 16, 1,
                          tip="Dabs laid at each step (with scatter: a spray)."),
                    # Photoshop's Brush Tip Shape: a sampled tip, turned.
                    O.Opt("tip", O.CHOICE, "round", "Tip",
                          choices=("round", "square", "chalk", "spatter", "star", "leaf",
                                   "grass", "custom"),
                          labels={"custom": "Defined (Edit > Define Brush Tip)"}),
                    O.Opt("angle", O.INT, 0, "Angle", -180, 180, 5, suffix="\u00b0"),
                    O.Opt("angle_jitter", O.INT, 0, "Angle jitter", 0, 100, 5, suffix=" %"))
    default_size = 16

    def _dab(self, ctx, event, x, y):
        """One step of the stroke: one dab, or a jittered, scattered handful."""
        sj = int(self.option("size_jitter", 0) or 0)
        oj = int(self.option("opacity_jitter", 0) or 0)
        sc = int(self.option("scatter", 0) or 0)
        n = max(1, int(self.option("count", 1) or 1))
        aj = int(self.option("angle_jitter", 0) or 0)
        if not (sj or oj or sc or aj) and n == 1:
            return self._one_dab(ctx, event, x, y, 1.0, 1.0)
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = self._rng = np.random.default_rng(1)
        size = max(1.0, float(self.option("size", self.default_size)))
        covered = None
        for _ in range(n):
            ks = 1.0 - rng.random() * sj / 100.0
            ko = 1.0 - rng.random() * oj / 100.0
            ang = rng.random() * 2 * math.pi
            dist = rng.random() * sc / 100.0 * size
            self._turn = (rng.random() * 2 - 1) * 180.0 * aj / 100.0
            r = self._one_dab(ctx, event, x + math.cos(ang) * dist, y + math.sin(ang) * dist, ks, ko)
            if r is not None:
                covered = r if covered is None else covered.union(r)
        return covered

    def _one_dab(self, ctx, event, x, y, size_k, flow_k):
        cell = ctx.cell
        size = max(1, float(self.option("size", self.default_size)) * size_k)
        if self.option("pressure_size", True) and event.pressure < 1.0:
            size = max(1.0, size * event.pressure)
        from .. import brushtips
        cov = brushtips.coverage(self.option("tip", "round"), size,
                                 float(self.option("angle", 0) or 0) + getattr(self, "_turn", 0.0))
        if cov is None:
            cov = _circle_coverage(size, float(self.option("hardness", 0.85)), antialias=True)
        dh, dw = cov.shape
        ox, oy = int(math.floor(x - dw / 2.0)), int(math.floor(y - dh / 2.0))
        rect = Rect(ox, oy, dw, dh)
        clipped = rect.clipped_to(cell.width, cell.height)
        if clipped is None:
            return None
        cov = cov[clipped.y - oy:clipped.y - oy + clipped.h,
                  clipped.x - ox:clipped.x - ox + clipped.w]

        flow = float(self.option("flow", 1.0)) * flow_k
        if flow < 1.0:
            cov = accel.spec._mul(cov.astype(np.uint32),
                                  np.uint32(int(flow * 255))).astype(np.uint8)
        self._session.touch(clipped)

        src = np.empty((clipped.h, clipped.w, 4), dtype=np.uint8)
        src[:] = ctx.colour_for(event)
        cov = ctx.coverage(cov, clipped)
        if cell.index_locked:
            # An index cannot be partly changed, so a dab below 50% coverage
            # writes nothing -- and a low-flow brush (the airbrush at 12%)
            # would never write anything at all, however long it was held.
            # Coverage is therefore ACCUMULATED over the stroke, and a pixel
            # takes the new index once the total crosses the threshold:
            # paint building up, which is what flow is supposed to mean.
            acc = getattr(self, "_acc", None)
            if acc is None or acc.shape != (cell.height, cell.width):
                acc = self._acc = np.zeros((cell.height, cell.width), dtype=np.uint16)
            window = acc[clipped.slice()]
            window += (np.full(window.shape, 255, np.uint16) if cov is None
                       else cov.astype(np.uint16))
            np.minimum(window, 1024, out=window)
            mask = np.where(window >= 128, 255, 0).astype(np.uint8)
            if mask.any():
                cell.apply_masked(clipped, src, mask)
            return clipped
        cell.apply_masked(clipped, src, cov)
        return clipped

    def on_begin(self, ctx, event):
        self._acc = None
        # Seeded per stroke, so a jittered stroke is reproducible (and its
        # preview is what lands) rather than shuffling on every repaint.
        self._strokes = getattr(self, "_strokes", 0) + 1
        self._rng = np.random.default_rng(self._strokes)
        return super().on_begin(ctx, event)


class EraserTool(BrushTool):
    """Removes alpha instead of adding colour."""

    name = "eraser"
    label = "Eraser"
    options_spec = (O.size(16), O.hardness(0.85))

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

        # Through erase_masked, not by editing the rgba plane directly: on an
        # index-locked layer that plane is a cache regenerated from indices,
        # so erasing it erased nothing that survived.
        return cell.erase_masked(clipped, cov)


class BucketFillTool(Tool):
    """Flood fill with Paint.NET's alpha-aware tolerance metric."""

    name = "bucket"
    label = "Paint Bucket"
    options_spec = (O.tolerance(32),
                    O.Opt("global", O.BOOL, False, "Global fill",
                          tip="Fill every similar pixel, not just the connected area."))

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
    # Paint.NET's sampling options: an averaged square, from the layer or
    # from the merged image.
    options_spec = (
        O.Opt("sample_size", O.CHOICE, "1", "Sampling",
              choices=("1", "3", "5", "11", "31"),
              labels={"1": "Single pixel", "3": "3x3 region", "5": "5x5 region",
                      "11": "11x11 region", "31": "31x31 region"}),
        O.Opt("sample_from", O.CHOICE, "layer", "From",
              choices=("layer", "image"), labels={"layer": "Layer", "image": "Image"}),
    )

    def on_begin(self, ctx, event):
        from .. import annotations
        cell = ctx.cell
        if cell is None:
            return None
        if not cell.bounds.contains(event.ix, event.iy):
            return None
        size = int(self.option("sample_size", "1") or 1)
        from_image = self.option("sample_from", "layer") == "image" and ctx.composite is not None
        picked_index = None
        # An index is only meaningful for one pixel of the layer itself.
        if cell.index_locked and size == 1 and not from_image:
            picked_index = int(cell.plane("index")[event.iy, event.ix])
        source = ctx.composite() if from_image else cell.pixels
        if size == 1:
            rgba = tuple(int(v) for v in source[event.iy, event.ix])
        else:
            rgba = annotations.sample(source, event.x, event.y, size)
        if event.button == 2:
            ctx.secondary = rgba
            ctx.secondary_index = picked_index
        else:
            ctx.primary = rgba
            ctx.primary_index = picked_index
        self.active = False
        return {"rgba": rgba, "index": picked_index}
