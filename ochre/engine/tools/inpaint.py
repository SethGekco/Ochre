# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tools that make things disappear: spot heal, remove, patch, content-aware move.

Each is a way of saying WHERE the hole is, handed to engine/inpaint.py for
the filling:

  Spot Healing Brush  paint over a blemish; on release, what you painted is
                      rebuilt from its surroundings. No source to pick.
  Remove              the same gesture, tuned for bigger objects (larger
                      brush, larger patches, a wider search).
  Patch               lasso an area, drag it onto good texture: the area
                      takes that texture, blended seamlessly. Or the other
                      way round ("Destination") -- drag the area elsewhere
                      and it is copied there, blended in.
  Content-Aware Move  lasso something, drag it: it lands where you drop it,
                      blended in, and where it was is filled as if it had
                      never been there.

The patch and move tools need a selection, and when there is none (or the
press lands outside it) they draw one, freehand, like the lasso -- which is
how Photoshop's Patch tool behaves and why it does not need a second tool.

Every write is limited to the pixels that were meant to change. On an
index-locked layer that matters: rewriting an unchanged pixel through the
palette could swap it to a duplicate entry with the same colour.
"""

import math

import numpy as np

from .. import inpaint
from ..geometry import Rect
from ..selection import REPLACE, UNION, EXCLUDE, Selection
from . import options as O
from .base import MOD_ALT, MOD_SHIFT, Tool
from .paint import _circle_coverage, _StrokeTool


def _hole_rect(mask, pad=0):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    return Rect.from_bounds(max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def _shift_mask(mask, dx, dy):
    """mask moved by (dx, dy); what slides off the canvas is lost."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    src = Rect(0, 0, w, h).intersect(Rect(-dx, -dy, w, h))
    if src is not None:
        dst = src.translated(dx, dy).clipped_to(w, h)
        if dst is not None:
            out[dst.slice()] = mask[dst.y - dy:dst.y - dy + dst.h,
                                    dst.x - dx:dst.x - dx + dst.w]
    return out


def _sample(image, rect, ox, oy):
    """image over rect displaced by (ox, oy), clamped at the canvas edge."""
    h, w = image.shape[:2]
    ys = np.clip(np.arange(rect.y, rect.y + rect.h) + oy, 0, h - 1)
    xs = np.clip(np.arange(rect.x, rect.x + rect.w) + ox, 0, w - 1)
    return image[ys[:, None], xs[None, :]]


_TYPE = O.Opt("type", O.CHOICE, "content_aware", "Type",
              choices=("content_aware", "smooth"),
              labels={"content_aware": "Content-aware", "smooth": "Smooth"},
              tip="Content-aware copies texture from around the spot; smooth "
                  "blends the colours across it, which suits flat surfaces.")
_DETAIL = O.Opt("patch", O.INT, 7, "Detail", 3, 21, 2, suffix=" px",
                tip="Patch size the fill copies texture in. Larger keeps "
                    "bigger structure intact; smaller follows fine detail.")


class SpotHealTool(_StrokeTool):
    """Paint over a blemish; on release it is rebuilt from what surrounds it.

    While painting, the stroke shows as a translucent grey so you can see
    what will be replaced -- the same buffer that is restored before the
    fill, so the preview leaves nothing behind.
    """

    name = "spot_heal"
    label = "Spot Healing Brush"
    default_size = 20
    options_spec = (O.size(20), _TYPE, _DETAIL)
    slow_commit = True
    _tint = np.array((96, 96, 96, 255), dtype=np.uint8)

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            return None
        self._mask = np.zeros((cell.height, cell.width), dtype=bool)
        return super().on_begin(ctx, event)

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1.0, float(self.option("size", self.default_size)))
        cov = _circle_coverage(size, 1.0, False) >= 128
        d = cov.shape[0]
        ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
        rect = Rect(ox, oy, d, d).clipped_to(cell.width, cell.height)
        if rect is None:
            return None
        foot = cov[rect.y - oy:rect.y - oy + rect.h, rect.x - ox:rect.x - ox + rect.w]
        sel = ctx.coverage(None, rect)
        if sel is not None:
            foot = foot & (sel > 0)
        new = foot & ~self._mask[rect.slice()]
        if not new.any():
            return None
        self._mask[rect.slice()] |= new
        self._session.touch(rect)
        # 160/255: over the threshold an index-locked layer uses, so the
        # preview shows there too.
        cell.lerp_masked(rect, self._tint, (new * 160).astype(np.uint8))
        return rect

    def _fill(self, ctx, rgba, hole):
        method = self.option("type", "content_aware")
        return inpaint.fill_hole(rgba, hole, method=method,
                                 patch=int(self.option("patch", 7)))

    def on_end(self, ctx, event):
        if self._session is None:
            return None
        cell = ctx.cell
        hole = self._mask
        self._mask = None
        rect = _hole_rect(hole)
        if rect is None:
            return super().on_end(ctx, event)
        self._session.restore()
        rgba = cell.plane("rgba")
        filled = self._fill(ctx, rgba, hole)
        self._session.touch(rect)
        cov = hole[rect.slice()].astype(np.uint8) * 255
        sel = ctx.coverage(None, rect)
        if sel is not None:
            cov = np.minimum(cov, sel)
        cell.lerp_masked(rect, filled[rect.slice()], cov)
        super().on_end(ctx, event)
        return rect


class RemoveTool(SpotHealTool):
    """Paint over an object to remove it -- spot healing for bigger things.

    Photoshop's Remove tool invents content with a neural network. This one
    cannot invent anything: it rebuilds the area from texture elsewhere in
    the image, with larger patches and a wider search than the spot healing
    brush. Excellent on grass, sky, walls and water; visibly repetitive when
    the object covers most of what could replace it.
    """

    name = "remove"
    label = "Remove Tool"
    default_size = 40
    options_spec = (O.size(40), O.Opt("patch", O.INT, 9, "Detail", 3, 21, 2, suffix=" px",
                                      tip=_DETAIL.tip))

    def _fill(self, ctx, rgba, hole):
        ys, xs = np.nonzero(hole)
        span = max(int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1)
        return inpaint.fill_hole(rgba, hole, method="content_aware",
                                 patch=int(self.option("patch", 9)),
                                 margin=int(max(48, span * 2.5)))


# ---- drag-a-selection tools ----------------------------------------------------

class _SelectionDragTool(Tool):
    """Press inside the selection and drag it; press outside to lasso a new one."""

    snaps = False         # moves by an offset; snapping the pointer is wrong

    changes_selection = True
    wants_stroke = True
    slow_commit = True

    def __init__(self, **options):
        super().__init__(**options)
        self._lasso = None
        self._drag = None

    # -- lasso half ------------------------------------------------------

    def _lasso_mode(self, event):
        if event.has(MOD_SHIFT):
            return UNION
        if event.has(MOD_ALT):
            return EXCLUDE
        return REPLACE

    def _inside(self, ctx, event):
        sel = ctx.selection
        if sel is None or sel.mask is None or sel.selects_all():
            return False
        x, y = event.ix, event.iy
        h, w = sel.mask.shape
        return 0 <= x < w and 0 <= y < h and sel.mask[y, x] >= 128

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            self.active = False
            return None
        if self._inside(ctx, event) and not (event.modifiers & (MOD_SHIFT | MOD_ALT)):
            from ..stroke import StrokeSession
            s = ctx.settings
            self._drag = {
                "start": (event.x, event.y),
                "base": ctx.selection.mask.copy(),
                "snap": ctx.cell.plane("rgba").copy(),
                "offset": (0, 0),
                "session": StrokeSession(
                    ctx.doc, ctx.layer, ctx.frame,
                    block=256 if s is None else s.get("History", "BlockSize"),
                    max_blocks=1 << 20,
                    level=1 if s is None else s.get("History", "CompressLevel"),
                    label=self.label),
            }
            return None
        self._lasso = [(event.x, event.y)]
        self._lasso_op = self._lasso_mode(event)
        return None

    def on_motion(self, ctx, event):
        if self._lasso is not None:
            lx, ly = self._lasso[-1]
            if abs(event.x - lx) >= 1.0 or abs(event.y - ly) >= 1.0:
                self._lasso.append((event.x, event.y))
            return None
        if self._drag is None:
            return None
        dx = int(round(event.x - self._drag["start"][0]))
        dy = int(round(event.y - self._drag["start"][1]))
        if event.has(MOD_SHIFT):
            if abs(dx) >= abs(dy):
                dy = 0
            else:
                dx = 0
        if (dx, dy) == self._drag["offset"]:
            return None
        self._drag["offset"] = (dx, dy)
        sess = self._drag["session"]
        old = sess.restore()
        rect = self.preview(ctx, dx, dy)
        if old is None:
            return rect
        return old if rect is None else old.union(rect)

    def on_end(self, ctx, event):
        if self._lasso is not None:
            pts, self._lasso = self._lasso, None
            if len(pts) < 3 or ctx.selection is None:
                return None
            mode = self._lasso_op
            working = (ctx.selection.copy() if mode != REPLACE
                       else Selection(ctx.doc.width, ctx.doc.height))
            working.select_polygon(pts, mode, True)
            ctx.selection.mask = working.mask
            ctx.selection._bbox = None
            ctx.selection.geometry = working.geometry
            return working.bbox
        if self._drag is None:
            return None
        self.on_motion(ctx, event)
        drag, self._drag = self._drag, None
        sess = drag["session"]
        dx, dy = drag["offset"]
        if (dx, dy) == (0, 0):
            sess.cancel()
            return None
        sess.restore()
        rect = self.apply(ctx, drag, dx, dy)
        cmd = sess.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return rect

    def on_cancel(self, ctx):
        self._lasso = None
        if self._drag is None:
            return None
        drag, self._drag = self._drag, None
        rect = drag["session"].cancel()
        ctx.selection.mask = drag["base"]
        ctx.selection._bbox = None
        return rect

    def overlay(self):
        if self._lasso is not None and len(self._lasso) > 1:
            return [("poly", list(self._lasso), False)]
        if self._drag is not None:
            r = _hole_rect(self._drag["base"] >= 128)
            if r is None:
                return []
            dx, dy = self._drag["offset"]
            out = [("rect", (r.x + dx, r.y + dy, r.x1 + dx, r.y1 + dy))]
            if (dx, dy) != (0, 0):
                out.append(("rect", (r.x, r.y, r.x1, r.y1)))
            return out
        return []

    # -- the pixel part, per tool ----------------------------------------

    def _write(self, ctx, drag, rect, rgba_window, cov):
        if rect is None or cov is None or not cov.any():
            return None
        drag["session"].touch(rect)
        ctx.cell.lerp_masked(rect, rgba_window, cov)
        return rect

    def _blend_into(self, base, window, region_cov, ox, oy, seamless):
        """New pixels for `window`: base, with base displaced by (ox, oy) laid
        over the region -- Poisson-blended or straight."""
        dest = inpaint.to_work(base[window.slice()])
        src = inpaint.to_work(_sample(base, window, ox, oy))
        region = region_cov >= 1
        if seamless and region.any():
            out = inpaint.seamless_blend(dest, src, region)
        else:
            out = dest.copy()
            out[region] = src[region]
        return inpaint.from_work(out)

    def preview(self, ctx, dx, dy):
        return self.apply(ctx, self._drag, dx, dy)

    def apply(self, ctx, drag, dx, dy):
        raise NotImplementedError


class PatchTool(_SelectionDragTool):
    """Lasso an area, then drag it onto good texture; the area takes it.

    "Source" (the default): the selection is the damaged area, and you drag
    it to where the replacement should come from. "Destination": the
    selection is the good part, and you drag it to where it should be
    copied. Either way the landing is blended seamlessly, so the patch
    keeps the texture and takes on the surrounding tone.
    """

    name = "patch"
    label = "Patch Tool"
    options_spec = (
        O.Opt("patch_mode", O.CHOICE, "source", "Patch",
              choices=("source", "destination"),
              labels={"source": "Source", "destination": "Destination"},
              tip="Source: drag the damaged area onto good texture. "
                  "Destination: drag good texture onto the damaged area."),
        O.Opt("seamless", O.BOOL, True, "Blend edges",
              tip="Match the patch's tone to its new surroundings."),
    )

    def apply(self, ctx, drag, dx, dy):
        base, snap = drag["base"], drag["snap"]
        seamless = bool(self.option("seamless", True))
        if self.option("patch_mode", "source") == "destination":
            region = _shift_mask(base, dx, dy)
            ox, oy = -dx, -dy
        else:
            region = base
            ox, oy = dx, dy
        rect = _hole_rect(region >= 1, pad=1)
        if rect is None:
            return None
        cov = region[rect.slice()]
        out = self._blend_into(snap, rect, cov, ox, oy, seamless)
        written = self._write(ctx, drag, rect, out, cov)
        if self.option("patch_mode", "source") == "destination" and ctx.selection is not None:
            ctx.selection.mask = region
            ctx.selection._bbox = None
        return written


class ContentAwareMoveTool(_SelectionDragTool):
    """Lasso something and drag it: it moves, and where it was is filled in.

    While dragging, the selection is shown moving as a plain copy, which is
    instant; on release the vacated area is rebuilt by content-aware fill
    and the moved part is blended into its new surroundings. "Extend"
    leaves the original where it was -- a seamless duplicate.
    """

    name = "content_move"
    label = "Content-Aware Move"
    options_spec = (
        O.Opt("move_mode", O.CHOICE, "move", "Mode", choices=("move", "extend"),
              labels={"move": "Move", "extend": "Extend"},
              tip="Move fills where the selection was; Extend leaves it, "
                  "making a blended copy."),
        O.Opt("seamless", O.BOOL, True, "Blend edges",
              tip="Match the moved part's tone to where it lands."),
        _DETAIL,
    )

    def preview(self, ctx, dx, dy):
        drag = self._drag
        region = _shift_mask(drag["base"], dx, dy)
        rect = _hole_rect(region >= 1)
        if rect is None:
            return None
        cov = region[rect.slice()]
        win = _sample(drag["snap"], rect, -dx, -dy)
        return self._write(ctx, drag, rect, win, cov)

    def apply(self, ctx, drag, dx, dy):
        base, snap = drag["base"], drag["snap"]
        moved = _shift_mask(base, dx, dy)
        canvas = snap
        hole = None
        if self.option("move_mode", "move") == "move":
            # Grow the vacated area a little so the object's antialiased rim,
            # which the selection only partly covered, goes with it.
            hole = inpaint._dilate(base >= 1, 2) & ~(moved >= 128)
            if hole.any():
                canvas = inpaint.fill_hole(snap, hole, method="content_aware",
                                           patch=int(self.option("patch", 7)))
        rect_dst = _hole_rect(moved >= 1, pad=1)
        if rect_dst is not None:
            # Lay the moved content over the FILLED canvas, sampling the
            # content from the snapshot, which still has it.
            work = canvas.copy()
            dest = inpaint.to_work(canvas[rect_dst.slice()])
            src = inpaint.to_work(_sample(snap, rect_dst, -dx, -dy))
            region = moved[rect_dst.slice()] >= 1
            if bool(self.option("seamless", True)) and region.any():
                out = inpaint.seamless_blend(dest, src, region)
            else:
                out = dest.copy()
                out[region] = src[region]
            # Soft selection edges: lerp by the moved coverage, once, here;
            # the write below then lands these values exactly.
            t = (moved[rect_dst.slice()].astype(np.float32) / 255.0)[..., None]
            out = dest + (out - dest) * t
            work[rect_dst.slice()] = inpaint.from_work(out)
            canvas = work
        changed = moved >= 1
        if hole is not None:
            changed |= hole
        cov = changed.astype(np.uint8) * 255
        rect = _hole_rect(changed)
        if rect is None:
            return None
        written = self._write(ctx, drag, rect, canvas[rect.slice()], cov[rect.slice()])
        if ctx.selection is not None:
            ctx.selection.mask = moved
            ctx.selection._bbox = None
        return written
