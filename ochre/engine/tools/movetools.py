# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Clone stamp, and the two move tools.

Moving pixels is the one interaction that does not fit the "mark dirty and
redraw" pattern the other tools share, because it is a LIFT: the selected
pixels leave the layer, float, and land somewhere else. Get it wrong and the
source region either does not clear, or clears and cannot be undone, or the
floating copy composites against its own trail.

The model here is the classic float-and-stamp, built on the stroke session:

  mouse-down   copy the selected pixels out, erase them from the layer
  mouse-move   restore everything touched so far, blit the float at the
               new offset
  mouse-up     commit -- one history entry covering the lift AND the drop

Because the restore comes from the same scratch buffer that becomes the undo
entry, the float cannot leave a trail and the preview cannot disagree with
the result. That is the third distinct tool shape `restore()` has now paid
for, after the shapes and text.

Move-SELECTION is deliberately a separate tool from move-PIXELS. They look
identical and do completely different things -- one repositions the marquee,
the other repositions image content -- and conflating them is a reliable way
to destroy someone's work by accident.
"""

import math

import numpy as np

from ..geometry import Rect
from ..stroke import StrokeSession
from .base import Tool, constrain_angle
from .paint import _circle_coverage


class CloneStampTool(Tool):
    """Paint with pixels sampled from elsewhere.

    Ctrl-click sets the source anchor; dragging then paints, sampling at a
    constant offset from the brush.

    The source is sampled from a SNAPSHOT taken when the stroke begins, not
    from the live layer. Photoshop samples live, which lets a stroke feed on
    what it just painted -- occasionally useful, frequently a runaway smear
    when the source and destination overlap. Freezing the source for the
    duration of a stroke keeps the result predictable, and previous strokes
    are still visible to the next one, so nothing is actually lost.
    """

    name = "clone"
    label = "Clone Stamp"
    default_size = 24

    def __init__(self, **options):
        super().__init__(**options)
        self._anchor = None          # where the source was set
        self._offset = None          # destination -> source delta
        self._snapshot = None

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            return None

        # Ctrl-click sets the source rather than painting.
        if event.has(2):
            self._anchor = (event.x, event.y)
            self._offset = None
            return {"anchor": self._anchor}

        if self._anchor is None:
            # Nothing to clone from yet. Say so rather than silently no-op.
            return {"error": "set a clone source first (ctrl-click)"}

        if self._offset is None or not self.option("aligned", True):
            self._offset = (self._anchor[0] - event.x, self._anchor[1] - event.y)

        s = ctx.settings
        self._session = StrokeSession(
            ctx.doc, ctx.layer, ctx.frame,
            block=256 if s is None else s.get("History", "BlockSize"),
            max_blocks=512 if s is None else s.get("History", "MaxStrokeBlocks"),
            level=1 if s is None else s.get("History", "CompressLevel"),
            label=self.label)
        self._snapshot = cell.plane("rgba").copy()
        self._last = (event.x, event.y)
        return self._dab(ctx, event, event.x, event.y)

    def on_motion(self, ctx, event):
        if self._session is None:
            return None
        x0, y0 = self._last
        x1, y1 = event.x, event.y
        size = float(self.option("size", self.default_size))
        spacing = max(0.05, float(self.option("spacing", 0.15))) * max(size, 1.0)
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < spacing:
            return None

        covered = None
        steps = int(dist / spacing)
        for i in range(1, steps + 1):
            t = (i * spacing) / dist
            r = self._dab(ctx, event, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            covered = r if covered is None else covered.union(r)
        travelled = steps * spacing / dist
        self._last = (x0 + (x1 - x0) * travelled, y0 + (y1 - y0) * travelled)
        return covered

    def on_end(self, ctx, event):
        if self._session is None:
            return None
        cmd = self._session.commit()
        self._session = None
        self._snapshot = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        if self._session is None:
            return None
        rect = self._session.cancel()
        self._session = None
        self._snapshot = None
        return rect

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1, float(self.option("size", self.default_size)))
        cov = _circle_coverage(size, float(self.option("hardness", 0.8)))
        d = cov.shape[0]

        ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
        dest = Rect(ox, oy, d, d).clipped_to(cell.width, cell.height)
        if dest is None:
            return None

        dx, dy = self._offset
        src = Rect(dest.x + int(round(dx)), dest.y + int(round(dy)),
                   dest.w, dest.h)
        # Only the part of the source that actually exists can be cloned;
        # sampling off-canvas would smear the edge pixel.
        src_clipped = src.clipped_to(cell.width, cell.height)
        if src_clipped is None:
            return None
        shift_x = src_clipped.x - src.x
        shift_y = src_clipped.y - src.y
        dest = Rect(dest.x + shift_x, dest.y + shift_y,
                    src_clipped.w, src_clipped.h).clipped_to(cell.width, cell.height)
        if dest is None:
            return None
        src_clipped = Rect(src_clipped.x, src_clipped.y, dest.w, dest.h)

        window = cov[dest.y - oy:dest.y - oy + dest.h,
                     dest.x - ox:dest.x - ox + dest.w]
        self._session.touch(dest)
        patch = self._snapshot[src_clipped.slice()]
        cell.apply_masked(dest, patch, ctx.coverage(window, dest))
        return dest

    def reset_source(self):
        self._anchor = None
        self._offset = None


class MovePixelsTool(Tool):
    """Lift the selected pixels and put them somewhere else.

    With no selection this moves the whole layer, which is what a user
    dragging with nothing selected means.
    """

    name = "move_pixels"
    label = "Move Selected Pixels"

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            return None

        sel = ctx.selection
        region = (cell.bounds if sel is None or sel.selects_all()
                  else sel.bbox.clipped_to(cell.width, cell.height))
        if region is None or region.is_empty:
            return None

        s = ctx.settings
        self._session = StrokeSession(
            ctx.doc, ctx.layer, ctx.frame,
            block=256 if s is None else s.get("History", "BlockSize"),
            max_blocks=512 if s is None else s.get("History", "MaxStrokeBlocks"),
            level=1 if s is None else s.get("History", "CompressLevel"),
            label=self.label)

        self._origin = region
        self._start = (event.x, event.y)
        self._last_rect = None

        # Lift: copy the pixels out, masked by the selection so a lasso
        # carries only what it enclosed.
        patch = cell.plane("rgba")[region.slice()].copy()
        mask = None if sel is None or sel.selects_all() else sel.mask_for(region)
        if mask is not None:
            from ..arith import mul255
            patch = patch.copy()
            patch[..., 3] = mul255(patch[..., 3], mask)
        self._float = patch
        self._mask = mask

        self._cut = bool(self.option("cut", True))
        self._erase(ctx)
        return self._blit(ctx, 0, 0)

    def _erase(self, ctx):
        """Clear the source region. `cut` off means copy rather than move.

        This has to be RE-APPLIED after every restore, not done once. The
        stroke session restores PRE-STROKE pixels, which includes the content
        this erased -- so restoring to redraw the float at a new position also
        silently undoes the lift, and the move becomes a copy.
        """
        if not self._cut:
            return None
        cell = ctx.cell
        region = self._origin
        self._session.touch(region)
        target = cell.plane("rgba")[region.slice()]
        if self._mask is None:
            target[...] = 0
        else:
            from ..arith import mul255
            target[..., 3] = mul255(target[..., 3],
                                    (255 - self._mask).astype(np.uint8))
        cell._touched(region)
        return region

    def on_motion(self, ctx, event):
        if self._session is None:
            return None
        dx = event.x - self._start[0]
        dy = event.y - self._start[1]
        if event.has(1):
            # Shift constrains to the dominant axis, read live.
            if abs(dx) >= abs(dy):
                dy = 0.0
            else:
                dx = 0.0
        # Restore the previous float AND the source region, then re-apply the
        # lift before drawing. Restoring only the float would leave the hole
        # filled in wherever the two overlap.
        scope = self._last_rect
        if self._cut:
            scope = self._origin if scope is None else scope.union(self._origin)
        restored = self._session.restore(scope)
        self._erase(ctx)
        drawn = self._blit(ctx, dx, dy)
        if restored is None:
            return drawn
        return restored if drawn is None else restored.union(drawn)

    def on_end(self, ctx, event):
        if self._session is None:
            return None
        self.on_motion(ctx, event)
        cmd = self._session.commit()
        self._session = None
        self._float = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        if self._session is None:
            return None
        rect = self._session.cancel()
        self._session = None
        self._float = None
        return rect

    def _blit(self, ctx, dx, dy):
        """Place the floating pixels at the current offset."""
        cell = ctx.cell
        target = self._origin.translated(int(round(dx)), int(round(dy)))
        clipped = target.clipped_to(cell.width, cell.height)
        if clipped is None:
            self._last_rect = None
            return None

        sx = clipped.x - target.x
        sy = clipped.y - target.y
        patch = self._float[sy:sy + clipped.h, sx:sx + clipped.w]

        self._session.touch(clipped)
        # The float composites over whatever it lands on, rather than
        # replacing it -- a moved lasso selection must not bring a
        # rectangular hole of transparency with it.
        cell.apply_masked(clipped, patch, None)
        self._last_rect = clipped
        return clipped


class MoveSelectionTool(Tool):
    """Reposition the marquee. Touches no pixels at all.

    Separate from Move Selected Pixels on purpose: the two look the same and
    do entirely different things, and merging them is how people destroy work
    by accident.
    """

    name = "move_selection"
    label = "Move Selection"
    wants_stroke = False

    def on_begin(self, ctx, event):
        if ctx.selection is None or ctx.selection.selects_all():
            return None
        self._start = (event.x, event.y)
        self._base = ctx.selection.mask.copy()
        return None

    def on_motion(self, ctx, event):
        if getattr(self, "_base", None) is None:
            return None
        dx = int(round(event.x - self._start[0]))
        dy = int(round(event.y - self._start[1]))
        if event.has(1):
            if abs(dx) >= abs(dy):
                dy = 0
            else:
                dx = 0
        moved = np.zeros_like(self._base)
        h, w = self._base.shape
        src = Rect(0, 0, w, h).intersect(Rect(-dx, -dy, w, h))
        if src is not None:
            dst = src.translated(dx, dy).clipped_to(w, h)
            if dst is not None:
                moved[dst.slice()] = self._base[
                    dst.y - dy:dst.y - dy + dst.h,
                    dst.x - dx:dst.x - dx + dst.w]
        ctx.selection.mask = moved
        ctx.selection._bbox = None
        return ctx.selection.bbox

    def on_end(self, ctx, event):
        result = self.on_motion(ctx, event)
        self._base = None
        return result

    def on_cancel(self, ctx):
        if getattr(self, "_base", None) is not None and ctx.selection is not None:
            ctx.selection.mask = self._base
            ctx.selection._bbox = None
        self._base = None
        return None
