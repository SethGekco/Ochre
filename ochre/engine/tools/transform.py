# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Transform tools: unified, rotate, scale, shear, perspective, flip, 3D,
handle, cage, warp -- plus crop, perspective crop and the layer Move tool.

The free-transform family shares one engine, `_FloatTransform`:

  1. The first press LIFTS the selection (or, with none, the layer's
     content) into a floating patch and erases where it was.
  2. The transform is kept as the four DESTINATION corners of that patch --
     a quad -- and every drag edits the quad: move it, scale it about the
     opposite corner, rotate it about its centre, drag one corner alone for
     perspective, slide one edge for shear. The quad's homography is the
     transform. Keeping corners rather than composing matrices is what makes
     "drag this corner exactly here" exact, and it is what the handles show.
  3. Every drag re-renders from the stroke session: restore what was
     touched, erase the source again, draw the patch through the new quad.
     Preview and result are the same buffer, so the float cannot leave a
     trail -- the same pattern Move Selected Pixels uses.
  4. Enter, double-click or switching tools commits, at full quality, as
     ONE undo entry; Escape puts everything back.

Dragging previews with nearest-neighbour sampling, because bilinear at
600x600 is ~95 ms and a drag must keep up with the pointer; the commit is
rendered with the chosen interpolation. Index-locked layers transform their
index plane, nearest always -- halfway between two palette entries is a
third colour.
"""

import math

import numpy as np

from .. import warp
from ..geometry import Rect
from ..stroke import StrokeSession
from . import options as O
from .base import MOD_ALT, MOD_CTRL, MOD_SHIFT, Tool, drag_rect

CORNERS, EDGES = (0, 1, 2, 3), (4, 5, 6, 7)


def _interp():
    return O.Opt("interpolation", O.CHOICE, "bilinear", "Interpolation",
                 choices=("bilinear", "nearest"),
                 labels={"bilinear": "Smooth", "nearest": "Nearest (pixel art)"})


def _session(ctx, label):
    s = ctx.settings
    return StrokeSession(ctx.doc, ctx.layer, ctx.frame,
                         block=256 if s is None else s.get("History", "BlockSize"),
                         max_blocks=4096 if s is None else s.get("History", "MaxStrokeBlocks"),
                         level=1 if s is None else s.get("History", "CompressLevel"),
                         label=label)


class _FloatTransform(Tool):
    """Lift, transform with handles, commit. Subclasses choose what drags do."""

    snaps = False         # moves by an offset; snapping the pointer is wrong

    wants_stroke = False
    multi_click = True
    commit_on_switch = True
    changes_selection = True
    options_spec = (_interp(),)

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self._smart = None
        self._lifted = False
        self._float = None
        self._float_index = None
        self._quad = None
        self._drag = None
        self._last_rect = None
        self._session = None
        self._sel_mask = None

    # ---- lifting -----------------------------------------------------------

    # ---- smart objects: transform the MATRIX, not the pixels ------------------
    # A smart object keeps its original; the transform tools edit where it
    # lands (its source-to-document homography), and the compositor renders
    # from the original every time -- Photoshop's non-destructive transform.

    def _smart_layer(self, ctx):
        layer = ctx.layer
        return layer if getattr(layer, "kind", "raster") == "smart" else None

    def _lift_smart(self, ctx, layer):
        from .. import smartobject
        p = layer.params
        self._smart = layer
        self._params0 = dict(p)
        sw, sh = int(p.get("src_w", ctx.doc.width)), int(p.get("src_h", ctx.doc.height))
        self._float = np.broadcast_to(np.zeros(4, np.uint8), (sh, sw, 4))
        self._quad0 = np.asarray(smartobject.corners(p), np.float64)
        self._quad = self._quad0.copy()
        self._had_selection = False
        self._origin = ctx.doc.bounds
        self._lifted = True
        return True

    def _render_smart(self, ctx):
        layer = self._smart
        layer.params = dict(self._params0, matrix=[float(v) for v in self.matrix().ravel()])
        ctx.doc.cell(layer).dirty.add(ctx.doc.bounds)
        return ctx.doc.bounds

    def _lift(self, ctx):
        cell = ctx.cell
        if cell is None:
            return False
        smart = self._smart_layer(ctx)
        if smart is not None:
            return self._lift_smart(ctx, smart)
        sel = ctx.selection
        if sel is None or sel.selects_all():
            # The box hugs what is actually painted, as Photoshop's does. The
            # stored content_bbox only grows, so after any earlier move it
            # encloses empty space -- handles in mid-air, a click meant for a
            # corner landing "inside", rotation about the wrong centre.
            region = cell.tight_bbox()
            mask = None
        else:
            region = sel.bbox.clipped_to(cell.width, cell.height)
            mask = sel.mask_for(region)
        if region is None or region.is_empty:
            return False
        self._session = _session(ctx, self.label)
        self._origin = region
        patch = cell.plane("rgba")[region.slice()].copy()
        if mask is not None:
            patch[..., 3] = (patch[..., 3].astype(np.uint16) * mask // 255).astype(np.uint8)
        self._float = patch
        self._mask = np.full(region.h * region.w, 255, np.uint8).reshape(region.h, region.w) \
            if mask is None else mask.copy()
        if cell.index_locked:
            self._float_index = cell.plane("index")[region.slice()].copy()
        self._had_selection = mask is not None
        w, h = region.w, region.h
        x, y = region.x, region.y
        self._quad0 = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.float64)
        self._quad = self._quad0.copy()
        self._erase(ctx)
        self._lifted = True
        return True

    def _erase(self, ctx):
        cell = ctx.cell
        self._session.touch(self._origin)
        cell.erase_masked(self._origin, None if not self._had_selection else self._mask)

    # ---- the transform -----------------------------------------------------

    def matrix(self):
        """Patch coordinates -> document coordinates."""
        w, h = self._float.shape[1], self._float.shape[0]
        return warp.homography([(0, 0), (w, 0), (w, h), (0, h)], self._quad)

    def _render(self, ctx, final=False):
        if getattr(self, "_smart", None) is not None:
            return self._render_smart(ctx)
        cell = ctx.cell
        scope = self._last_rect.union(self._origin) if self._last_rect is not None else self._origin
        self._session.restore(scope)
        self._erase(ctx)
        H = self.matrix()
        fh, fw = self._float.shape[:2]
        try:
            x0, y0, x1, y1 = warp.bounds_of(H, fw, fh)
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return scope
        rect = Rect.from_bounds(x0 - 1, y0 - 1, x1 + 1, y1 + 1).clipped_to(cell.width, cell.height)
        if rect is None:
            self._last_rect = None
            return scope
        box = (rect.x, rect.y, rect.x1, rect.y1)
        self._session.touch(rect)
        method = warp.BILINEAR if final and self.option("interpolation", "bilinear") == "bilinear" \
            else warp.NEAREST
        if cell.index_locked and self._float_index is not None:
            idx = warp.warp_homography(self._float_index, H, box, index=True)
            cov = warp.warp_homography(self._mask, H, box, index=True)
            write = (idx >= 0) & (cov >= 128)
            mask = write.astype(np.uint8) * 255
            if write.any():
                cell.apply_masked(rect, np.clip(idx, 0, 255).astype(np.uint8), mask, name="index")
        else:
            patch = warp.warp_homography(self._float, H, box, method)
            cell.apply_masked(rect, patch, None)
        self._last_rect = rect
        return rect.union(scope)

    # ---- handles -------------------------------------------------------------

    def handles(self):
        q = self._quad
        mids = [(q[i] + q[(i + 1) % 4]) / 2.0 for i in range(4)]
        return [tuple(p) for p in q] + [tuple(p) for p in mids]

    def _hit(self, ctx, x, y):
        """Which handle (0-7) is under the point, 'inside', or 'outside'."""
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        for i, (hx, hy) in enumerate(self.handles()):
            if math.hypot(x - hx, y - hy) <= r:
                return i
        inside = warp.point_in_polygon(np.array([x]), np.array([y]), self._quad)[0]
        return "inside" if inside else "outside"

    def overlay(self):
        if not self._lifted or self._quad is None:
            return []
        q = [tuple(p) for p in self._quad]
        c = tuple(self._quad.mean(axis=0))
        return [("poly", q, True), ("handles", self.handles()), ("handles", [c])]

    # ---- events ------------------------------------------------------------

    def on_begin(self, ctx, event):
        if not self._lifted and not self._lift(ctx):
            self.active = False
            return None
        self._drag = {"what": self._hit(ctx, event.x, event.y), "start": (event.x, event.y),
                      "quad": self._quad.copy()}
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        new = self.drag_quad(self._drag, event)
        if new is None:
            return None
        self._quad = np.asarray(new, dtype=np.float64)
        return self._render(ctx)

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        self.active = True                  # stay: more drags may follow
        return None

    def hover(self, ctx, event):
        return False

    def finish(self, ctx):
        if not self._lifted:
            self._reset()
            return None
        if getattr(self, "_smart", None) is not None:
            from ..commands import PropertyDelta
            layer = self._smart
            self._render_smart(ctx)
            if layer.params != self._params0 and ctx.history is not None:
                ctx.history.push(PropertyDelta(layer.id, "params", self._params0, self.label),
                                 self.label)
            self._reset()
            self.active = False
            return ctx.doc.bounds
        rect = self._render(ctx, final=True)
        cmd = self._session.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        # The selection follows the content it outlined.
        if self._had_selection and ctx.selection is not None:
            H = self.matrix()
            m = warp.warp_homography(self._mask, H, (0, 0, ctx.doc.width, ctx.doc.height), index=True)
            ctx.selection.mask = np.clip(m, 0, 255).astype(np.uint8)
            ctx.selection._bbox = None
        self._reset()
        self.active = False
        return rect

    def on_cancel(self, ctx):
        if getattr(self, "_smart", None) is not None:
            self._smart.params = dict(self._params0)
            ctx.doc.cell(self._smart).dirty.add(ctx.doc.bounds)
            self._reset()
            return ctx.doc.bounds
        if self._session is not None:
            rect = self._session.cancel()
            self._reset()
            return rect
        self._reset()
        return None

    # ---- the drag geometry (unified) ------------------------------------------

    def drag_quad(self, drag, event):
        what = drag["what"]
        if what == "inside":
            return self._move(drag, event)
        if what == "outside":
            return self._rotate(drag, event)
        if event.has(MOD_CTRL):
            return self._perspective(drag, event) if what in CORNERS else self._shear(drag, event)
        return self._scale(drag, event)

    @staticmethod
    def _move(drag, event):
        dx, dy = event.x - drag["start"][0], event.y - drag["start"][1]
        if event.has(MOD_SHIFT):
            if abs(dx) >= abs(dy):
                dy = 0.0
            else:
                dx = 0.0
        return drag["quad"] + (dx, dy)

    @staticmethod
    def _rotate(drag, event):
        q = drag["quad"]
        c = q.mean(axis=0)
        a0 = math.atan2(drag["start"][1] - c[1], drag["start"][0] - c[0])
        a1 = math.atan2(event.y - c[1], event.x - c[0])
        ang = a1 - a0
        if event.has(MOD_SHIFT):
            step = math.radians(15)
            ang = round(ang / step) * step
        R = warp.rotate_about(ang, c[0], c[1])
        return warp.apply(R, q)

    @staticmethod
    def _scale(drag, event):
        q = drag["quad"]
        what = drag["what"]
        A = q[1] - q[0]
        B = q[3] - q[0]
        basis = np.array([A, B]).T
        try:
            inv = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            return None
        m = np.array([event.x, event.y])
        if what in CORNERS:
            o = q[(what + 2) % 4]
            a, b = inv @ (m - o)
            a0, b0 = inv @ (q[what] - o)
            fa = a / a0 if abs(a0) > 1e-9 else 1.0
            fb = b / b0 if abs(b0) > 1e-9 else 1.0
            if event.has(MOD_SHIFT):
                f = fa if abs(fa) > abs(fb) else fb
                fa = fb = f
        else:
            edge = what - 4                     # 0 top, 1 right, 2 bottom, 3 left
            if edge in (0, 2):
                o = q[3] if edge == 0 else q[0]
                _, b = inv @ (m - o)
                _, b0 = inv @ (q[edge * 1 if edge == 0 else 2] - o)
                fa, fb = 1.0, (b / b0 if abs(b0) > 1e-9 else 1.0)
            else:
                o = q[0] if edge == 1 else q[1]
                a, _ = inv @ (m - o)
                a0, _ = inv @ (q[1 if edge == 1 else 0] - o)
                fa, fb = (a / a0 if abs(a0) > 1e-9 else 1.0), 1.0
            if event.has(MOD_ALT):
                pass
        out = []
        for p in q:
            a, b = inv @ (p - o)
            out.append(o + fa * a * A + fb * b * B)
        return np.array(out)

    @staticmethod
    def _perspective(drag, event):
        q = drag["quad"].copy()
        i = drag["what"]
        q[i] = q[i] + (event.x - drag["start"][0], event.y - drag["start"][1])
        return q

    @staticmethod
    def _shear(drag, event):
        q = drag["quad"].copy()
        edge = drag["what"] - 4
        i, j = edge, (edge + 1) % 4
        d = q[j] - q[i]
        n = np.hypot(*d)
        if n < 1e-9:
            return q
        u = d / n
        delta = np.array([event.x - drag["start"][0], event.y - drag["start"][1]])
        along = u * float(delta @ u)
        q[i] = q[i] + along
        q[j] = q[j] + along
        return q


class TransformTool(_FloatTransform):
    """Unified transform -- Photoshop's Free Transform, GIMP's Unified Transform.

    Drag inside to move, a corner or edge to scale (Shift keeps proportions),
    outside the box to rotate (Shift snaps to 15 degrees), Ctrl+corner for
    perspective, Ctrl+edge to shear. Enter or double-click applies.
    """

    name = "transform"
    label = "Unified Transform"


class RotateTool(_FloatTransform):
    """Drag anywhere to rotate about the centre. Shift snaps to 15 degrees."""

    name = "rotate_tool"
    label = "Rotate"

    def drag_quad(self, drag, event):
        return self._rotate(drag, event)


class ScaleTool(_FloatTransform):
    """Corners and edges scale, inside moves. Shift keeps proportions."""

    name = "scale_tool"
    label = "Scale"

    def drag_quad(self, drag, event):
        what = drag["what"]
        if what == "inside":
            return self._move(drag, event)
        if what == "outside":
            q = drag["quad"]
            d = [math.hypot(event.x - p[0], event.y - p[1]) for p in q]
            drag = dict(drag, what=int(np.argmin(d)))
        return self._scale(drag, event)


class ShearTool(_FloatTransform):
    """Drag sideways to shear horizontally, up or down to shear vertically."""

    name = "shear_tool"
    label = "Shear"

    def drag_quad(self, drag, event):
        q = drag["quad"]
        dx, dy = event.x - drag["start"][0], event.y - drag["start"][1]
        c = q.mean(axis=0)
        h = max(1e-6, np.hypot(*(q[3] - q[0])))
        w = max(1e-6, np.hypot(*(q[1] - q[0])))
        if abs(dx) >= abs(dy):
            S = warp.shear_about(dx / h * 2.0, 0.0, c[0], c[1])
        else:
            S = warp.shear_about(0.0, dy / w * 2.0, c[0], c[1])
        return warp.apply(S, q)


class PerspectiveTool(_FloatTransform):
    """Drag each corner freely; drag inside to move."""

    name = "perspective_tool"
    label = "Perspective"

    def drag_quad(self, drag, event):
        what = drag["what"]
        if what in CORNERS:
            return self._perspective(drag, event)
        if what == "inside":
            return self._move(drag, event)
        q = drag["quad"]
        d = [math.hypot(event.x - p[0], event.y - p[1]) for p in q]
        return self._perspective(dict(drag, what=int(np.argmin(d))), event)


class Transform3DTool(_FloatTransform):
    """Tilt the selection in 3-D. Drag sideways to turn it about the vertical
    axis, up and down about the horizontal; Shift spins it in the plane."""

    name = "transform3d"
    label = "3D Transform"
    options_spec = (_interp(),
                    O.Opt("distance", O.FLOAT, 1.6, "Camera distance", 0.5, 10.0, 0.1,
                          tip="Smaller is a wider lens and stronger perspective."))

    def drag_quad(self, drag, event):
        q0 = self._quad0
        w = q0[1][0] - q0[0][0]
        h = q0[3][1] - q0[0][1]
        dx, dy = event.x - drag["start"][0], event.y - drag["start"][1]
        base = getattr(self, "_angles", (0.0, 0.0, 0.0))
        if event.has(MOD_SHIFT):
            rx, ry, rz = base[0], base[1], base[2] + dx / max(w, 1.0) * math.pi
        else:
            rx = base[0] - dy / max(h, 1.0) * math.pi * 0.75
            ry = base[1] + dx / max(w, 1.0) * math.pi * 0.75
            rz = base[2]
        self._pending = (rx, ry, rz)
        H = warp.rotation3d(w, h, rx, ry, rz, float(self.option("distance", 1.6)) * max(w, h))
        local = warp.apply(H, [(0, 0), (w, 0), (w, h), (0, h)])
        return local + q0[0]

    def on_end(self, ctx, event):
        out = super().on_end(ctx, event)
        if getattr(self, "_pending", None) is not None:
            self._angles = self._pending
        return out

    def _reset(self):
        super()._reset()
        self._angles = (0.0, 0.0, 0.0)
        self._pending = None


class FlipTool(_FloatTransform):
    """Click to flip the selection (or layer) horizontally; Shift for vertically.
    Applied at once -- there is nothing to adjust."""

    name = "flip_tool"
    label = "Flip"
    multi_click = False
    options_spec = (O.Opt("direction", O.CHOICE, "horizontal", "Direction",
                          choices=("horizontal", "vertical")),)

    def on_begin(self, ctx, event):
        if not self._lift(ctx):
            self.active = False
            return None
        vertical = (self.option("direction", "horizontal") == "vertical") ^ event.has(MOD_SHIFT)
        q = self._quad
        self._quad = q[[3, 2, 1, 0]] if vertical else q[[1, 0, 3, 2]]
        return None

    def on_end(self, ctx, event):
        self.active = True
        rect = self.finish(ctx)
        return rect


class HandleTransformTool(_FloatTransform):
    """Pin points and drag them. One handle moves, two rotate and scale,
    three add shear, four give full perspective -- GIMP's Handle Transform.

    Click on the image to drop a handle (up to four); drag a handle to move
    where that point goes. The transform is whatever best maps the handles'
    starting points to where they now are.
    """

    name = "handle_transform"
    label = "Handle Transform"

    def _reset(self):
        super()._reset()
        self._pins = []          # [[source(x, y), destination(x, y)], ...]

    def on_begin(self, ctx, event):
        if not self._lifted and not self._lift(ctx):
            self.active = False
            return None
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        for i, (_src, dst) in enumerate(self._pins):
            if math.hypot(event.x - dst[0], event.y - dst[1]) <= r:
                self._drag = {"pin": i, "start": (event.x, event.y), "from": tuple(dst)}
                return None
        if len(self._pins) < 4:
            # A new pin's SOURCE is where that point currently comes from, so
            # adding a pin never jolts what is already transformed.
            H = self._doc_matrix()
            src = tuple(warp.apply(np.linalg.inv(H), [(event.x, event.y)])[0])
            self._pins.append([src, (event.x, event.y)])
        self._drag = None
        return None

    def _doc_matrix(self):
        if not self._pins:
            return np.eye(3)
        return warp.from_points([p[0] for p in self._pins], [p[1] for p in self._pins])

    def drag_quad(self, drag, event):
        if drag is None or "pin" not in drag:
            return None
        i = drag["pin"]
        self._pins[i][1] = (drag["from"][0] + event.x - drag["start"][0],
                            drag["from"][1] + event.y - drag["start"][1])
        return warp.apply(self._doc_matrix(), self._quad0)

    def handles(self):
        return [tuple(p[1]) for p in self._pins] if self._lifted else []

    def overlay(self):
        if not self._lifted:
            return []
        return [("poly", [tuple(p) for p in self._quad], True), ("handles", self.handles())]


class CageTool(Tool):
    """Deform part of the image by moving the corners of a cage around it.

    Click to place cage points around what you want to bend, and close the
    cage by clicking the first point (or Enter). Then drag the points:
    everything inside the cage follows smoothly. Enter applies, Escape
    abandons. GIMP's Cage Transform, rendered by mean-value coordinates.
    """

    snaps = False         # moves by an offset; snapping the pointer is wrong

    name = "cage"
    label = "Cage Transform"
    wants_stroke = False
    multi_click = True
    commit_on_switch = True
    options_spec = (_interp(),)

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self._cage = []
        self._moved = None
        self._closed = False
        self._session = None
        self._drag = None
        self._hover = None
        self._last = None

    def overlay(self):
        if not self._cage:
            return []
        pts = [tuple(p) for p in (self._moved if self._moved is not None else self._cage)]
        out = [("poly", pts + ([] if self._closed or self._hover is None else [self._hover]),
                self._closed), ("handles", pts)]
        return out

    def hover(self, ctx, event):
        if self._closed:
            return False
        self._hover = (event.x, event.y)
        return True

    def on_begin(self, ctx, event):
        pt = (event.x, event.y)
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        if not self._closed:
            if len(self._cage) >= 3 and math.hypot(pt[0] - self._cage[0][0],
                                                   pt[1] - self._cage[0][1]) <= r:
                self._close(ctx)
            else:
                self._cage.append(pt)
            return None
        for i, p in enumerate(self._moved):
            if math.hypot(pt[0] - p[0], pt[1] - p[1]) <= r:
                self._drag = {"i": i, "start": pt, "from": tuple(p)}
                return None
        self._drag = None
        return None

    def _close(self, ctx):
        cell = ctx.cell
        if cell is None or len(self._cage) < 3:
            return
        self._closed = True
        self._moved = [tuple(p) for p in self._cage]
        xs = [p[0] for p in self._cage]
        ys = [p[1] for p in self._cage]
        self._region = Rect.from_bounds(int(math.floor(min(xs))), int(math.floor(min(ys))),
                                        int(math.ceil(max(xs))) + 1, int(math.ceil(max(ys))) + 1
                                        ).clipped_to(cell.width, cell.height)
        self._snapshot = cell.plane("rgba").copy()
        self._snapshot_index = cell.plane("index").copy() if cell.index_locked else None
        self._session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)

    def on_motion(self, ctx, event):
        if not self._closed or self._drag is None:
            return None
        i = self._drag["i"]
        self._moved[i] = (self._drag["from"][0] + event.x - self._drag["start"][0],
                          self._drag["from"][1] + event.y - self._drag["start"][1])
        return self._render(ctx, final=False)

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        self.active = True
        return None

    def _render(self, ctx, final):
        cell = ctx.cell
        src_poly = np.asarray(self._cage, dtype=np.float64)
        dst_poly = np.asarray(self._moved, dtype=np.float64)
        xs, ys = dst_poly[:, 0], dst_poly[:, 1]
        out_rect = Rect.from_bounds(int(math.floor(xs.min())) - 1, int(math.floor(ys.min())) - 1,
                                    int(math.ceil(xs.max())) + 2, int(math.ceil(ys.max())) + 2
                                    ).clipped_to(cell.width, cell.height)
        scope = self._region if self._last is None else self._last.union(self._region)
        if out_rect is not None:
            scope = scope.union(out_rect)
        self._session.restore(scope)
        self._session.touch(self._region)
        # The original cage interior goes, then the deformed version lands.
        from ..selection import rasterize_polygon
        hole = rasterize_polygon(cell.width, cell.height, [tuple(p) for p in src_poly], False)
        cell.erase_masked(self._region, hole[self._region.slice()])
        if out_rect is None:
            self._last = None
            return scope
        self._session.touch(out_rect)
        gy, gx = np.mgrid[out_rect.y:out_rect.y1, out_rect.x:out_rect.x1]
        px, py = gx + 0.5, gy + 0.5
        inside = warp.point_in_polygon(px, py, dst_poly)
        if inside.any():
            pts = np.stack([px[inside], py[inside]], axis=1)
            src = warp.mean_value_map(pts, dst_poly, src_poly)
            if cell.index_locked and self._snapshot_index is not None:
                idx = warp.sample_index(self._snapshot_index, src[:, 0], src[:, 1])
                plane = cell.plane("index")[out_rect.slice()]
                vals = plane[inside]
                ok = idx >= 0
                vals[ok] = idx[ok]
                plane[inside] = vals
                cell._after_write(out_rect, "index")
            else:
                method = warp.BILINEAR if final and self.option("interpolation", "bilinear") \
                    == "bilinear" else warp.NEAREST
                samp = warp.warp_field(self._snapshot, src[:, 0], src[:, 1], method)
                patch = np.zeros((out_rect.h, out_rect.w, 4), dtype=np.uint8)
                patch[inside] = samp
                cell.apply_masked(out_rect, patch, None)
        self._last = out_rect
        return scope

    def finish(self, ctx):
        if not self._closed:
            if len(self._cage) >= 3:
                self._close(ctx)
                self.active = True
                return None
            self._reset()
            self.active = False
            return None
        rect = self._render(ctx, final=True)
        cmd = self._session.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        self._reset()
        self.active = False
        return rect

    def on_cancel(self, ctx):
        rect = self._session.cancel() if self._session is not None else None
        self._reset()
        return rect


class WarpTool(Tool):
    """Push pixels around with a brush -- GIMP's Warp Transform, Photoshop's
    Liquify. Move drags pixels along; Grow and Shrink bloat and pinch;
    Swirl twists; Restore paints the original back. Each stroke is one undo
    entry, and strokes accumulate on the same distortion until you switch
    tool or layer.
    """

    snaps = False         # moves by an offset; snapping the pointer is wrong

    name = "warp"
    label = "Warp Transform"
    wants_stroke = False
    options_spec = (O.size(60), O.Opt("strength", O.FLOAT, 0.5, "Strength", 0.01, 1.0, 0.05,
                                      percent=True),
                    O.Opt("mode", O.CHOICE, "move", "Behaviour",
                          choices=("move", "grow", "shrink", "swirl_cw", "swirl_ccw", "restore"),
                          labels={"swirl_cw": "Swirl clockwise",
                                  "swirl_ccw": "Swirl counter-clockwise"}))

    def __init__(self, **options):
        super().__init__(**options)
        self._field_key = None

    def on_begin(self, ctx, event):
        cell = ctx.cell
        if cell is None:
            self.active = False
            return None
        key = (ctx.layer.id, ctx.frame.id, cell.width, cell.height,
               ctx.history.position if ctx.history is not None else None)
        if self._field_key != key:
            # A fresh field whenever the layer or the history moved under us:
            # an undo must not leave the tool warping from stale pixels.
            self._orig = cell.plane("rgba").copy()
            self._orig_index = cell.plane("index").copy() if cell.index_locked else None
            self._dx = np.zeros((cell.height, cell.width), np.float32)
            self._dy = np.zeros((cell.height, cell.width), np.float32)
        self._session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)
        self._last = (event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        x0, y0 = self._last
        dist = math.hypot(event.x - x0, event.y - y0)
        size = float(self.option("size", 60))
        if dist < max(1.0, size * 0.08):
            return None
        rect = self._dab(ctx, event.x, event.y, event.x - x0, event.y - y0)
        self._last = (event.x, event.y)
        return rect

    def _dab(self, ctx, x, y, mx, my):
        cell = ctx.cell
        r = max(2.0, float(self.option("size", 60)) / 2.0)
        s = float(self.option("strength", 0.5))
        mode = self.option("mode", "move")
        box = Rect.from_bounds(int(x - r) - 1, int(y - r) - 1, int(x + r) + 2, int(y + r) + 2
                               ).clipped_to(cell.width, cell.height)
        if box is None:
            return None
        gy, gx = np.mgrid[box.y:box.y1, box.x:box.x1].astype(np.float32)
        px, py = gx + 0.5 - x, gy + 0.5 - y
        d = np.sqrt(px * px + py * py) / r
        fall = np.clip(1.0 - d, 0.0, 1.0) ** 2 * s
        dx = self._dx[box.slice()]
        dy = self._dy[box.slice()]
        if mode == "move":
            dx -= fall * mx
            dy -= fall * my
        elif mode in ("grow", "shrink"):
            k = 0.25 * (1.0 if mode == "grow" else -1.0)
            dx -= fall * px * k
            dy -= fall * py * k
        elif mode in ("swirl_cw", "swirl_ccw"):
            ang = 0.25 * (1.0 if mode == "swirl_cw" else -1.0)
            dx -= fall * (-py) * ang
            dy -= fall * px * ang
        else:
            keep = 1.0 - np.clip(fall * 1.5, 0.0, 1.0)
            dx *= keep
            dy *= keep
        sx = gx + 0.5 + dx
        sy = gy + 0.5 + dy
        self._session.touch(box)
        if cell.index_locked and self._orig_index is not None:
            idx = warp.sample_index(self._orig_index, sx, sy)
            plane = cell.plane("index")[box.slice()]
            ok = idx >= 0
            plane[ok] = idx[ok]
            cell._after_write(box, "index")
        else:
            out = warp.warp_field(self._orig, sx, sy, warp.BILINEAR)
            cell.lerp_masked(box, out, ctx.coverage(None, box))
        return box

    def on_end(self, ctx, event):
        if self._session is None:
            return None
        cmd = self._session.commit()
        self._session = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        self._field_key = (ctx.layer.id, ctx.frame.id, ctx.doc.width, ctx.doc.height,
                           ctx.history.position if ctx.history is not None else None)
        return cmd

    def on_cancel(self, ctx):
        if getattr(self, "_session", None) is None:
            return None
        rect = self._session.cancel()
        self._session = None
        return rect


class CropTool(Tool):
    """Drag a box, adjust it with the handles, press Enter to crop.

    Everything outside darkens while you decide; nothing is cut until you
    commit, and then the whole image -- every layer and frame -- crops as one
    undo entry. Aspect ratio can be locked.
    """

    name = "crop"
    label = "Crop"
    wants_stroke = False
    multi_click = True
    options_spec = (O.Opt("aspect", O.CHOICE, "free", "Aspect",
                          choices=("free", "1:1", "4:3", "3:2", "16:9", "original")),)

    def __init__(self, **options):
        super().__init__(**options)
        self._box = None
        self._drag = None

    def _aspect(self, ctx):
        a = self.option("aspect", "free")
        if a == "free":
            return None
        if a == "original":
            return ctx.doc.width / float(ctx.doc.height)
        w, h = (float(v) for v in a.split(":"))
        return w / h

    def overlay(self):
        if self._box is None:
            return []
        x0, y0, x1, y1 = self._box
        hs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1),
              ((x0 + x1) / 2, y0), (x1, (y0 + y1) / 2), ((x0 + x1) / 2, y1), (x0, (y0 + y1) / 2)]
        label = "%d x %d" % (abs(x1 - x0), abs(y1 - y0))
        return [("rect", self._box), ("handles", hs), ("text", (x0, y1), label)]

    def on_begin(self, ctx, event):
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        if self._box is not None:
            x0, y0, x1, y1 = self._box
            hs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for i, (hx, hy) in enumerate(hs):
                if math.hypot(event.x - hx, event.y - hy) <= r:
                    opp = hs[(i + 2) % 4]
                    self._drag = ("corner", opp)
                    return None
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._drag = ("move", (event.x, event.y), self._box)
                return None
        self._drag = ("new", (event.x, event.y))
        self._box = (event.x, event.y, event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        kind = self._drag[0]
        if kind == "move":
            sx, sy = self._drag[1]
            x0, y0, x1, y1 = self._drag[2]
            dx, dy = event.x - sx, event.y - sy
            self._box = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            return None
        ax, ay = self._drag[1]
        x, y = event.x, event.y
        aspect = self._aspect(ctx)
        if aspect is not None or event.has(MOD_SHIFT):
            aspect = aspect or 1.0
            w, h = abs(x - ax), abs(y - ay)
            if w / max(h, 1e-6) > aspect:
                w = h * aspect
            else:
                h = w / aspect
            x = ax + (w if x >= ax else -w)
            y = ay + (h if y >= ay else -h)
        self._box = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
        return None

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._drag = None
        self.active = True
        return None

    def finish(self, ctx):
        box, self._box = self._box, None
        self.active = False
        if box is None:
            return None
        x0, y0, x1, y1 = box
        rect = Rect.from_bounds(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        if rect.w < 1 or rect.h < 1:
            return None
        return {"crop": rect}

    def on_cancel(self, ctx):
        self._box = None
        self._drag = None
        return None


class PerspectiveCropTool(CropTool):
    """Drag a box, then drag its corners onto something that should be
    rectangular -- a photographed page, a sign -- and Enter straightens it."""

    name = "perspective_crop"
    label = "Perspective Crop"
    options_spec = ()

    def __init__(self, **options):
        super().__init__(**options)
        self._quad = None

    def overlay(self):
        if self._quad is None:
            return super().overlay()
        q = [tuple(p) for p in self._quad]
        return [("poly", q, True), ("handles", q)]

    def on_begin(self, ctx, event):
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        if self._quad is not None:
            for i, (hx, hy) in enumerate(self._quad):
                if math.hypot(event.x - hx, event.y - hy) <= r:
                    self._drag = ("pin", i, (event.x, event.y), tuple(self._quad[i]))
                    return None
            return None
        return super().on_begin(ctx, event)

    def on_motion(self, ctx, event):
        if self._drag is not None and self._drag[0] == "pin":
            _k, i, (sx, sy), (fx, fy) = self._drag
            self._quad[i] = (fx + event.x - sx, fy + event.y - sy)
            return None
        return super().on_motion(ctx, event)

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        if self._quad is None and self._box is not None:
            x0, y0, x1, y1 = self._box
            if abs(x1 - x0) >= 2 and abs(y1 - y0) >= 2:
                self._quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self._drag = None
        self.active = True
        return None

    def finish(self, ctx):
        quad, self._quad, self._box = self._quad, None, None
        self.active = False
        if quad is None:
            return None
        return {"perspective_crop": [tuple(map(float, p)) for p in quad]}

    def on_cancel(self, ctx):
        self._quad = None
        return super().on_cancel(ctx)


class MoveLayerTool(Tool):
    """Drag the whole layer -- Photoshop's Move tool (V), GIMP's Move in
    layer mode. With Auto-select on, pressing on the image first makes the
    topmost layer with paint under the pointer the active one."""

    snaps = False         # moves by an offset; snapping the pointer is wrong

    name = "move_layer"
    label = "Move Layer"
    wants_stroke = False
    picks_layer = True
    options_spec = (O.Opt("auto_select", O.BOOL, True, "Auto-select layer",
                          tip="Clicking picks the topmost layer with paint under the pointer."),)

    def on_begin(self, ctx, event):
        cell = ctx.cell
        region = None if cell is None else cell.tight_bbox()
        if region is None:
            self.active = False
            return None
        self._session = _session(ctx, self.label)
        self._start = (event.x, event.y)
        self._origin = region
        self._float = cell.plane("rgba")[self._origin.slice()].copy()
        self._float_index = cell.plane("index")[self._origin.slice()].copy() \
            if cell.index_locked else None
        self._last = None
        self._session.touch(self._origin)
        cell.erase_masked(self._origin, None)
        return self._place(ctx, 0, 0)

    def _place(self, ctx, dx, dy):
        cell = ctx.cell
        target = self._origin.translated(int(round(dx)), int(round(dy)))
        rect = target.clipped_to(cell.width, cell.height)
        self._last = rect
        if rect is None:
            return None
        sx, sy = rect.x - target.x, rect.y - target.y
        self._session.touch(rect)
        if self._float_index is not None:
            idx = self._float_index[sy:sy + rect.h, sx:sx + rect.w]
            a = self._float[sy:sy + rect.h, sx:sx + rect.w, 3]
            cell.apply_masked(rect, idx, np.where(a >= 128, 255, 0).astype(np.uint8), name="index")
        else:
            cell.apply_masked(rect, self._float[sy:sy + rect.h, sx:sx + rect.w], None)
        return rect

    def on_motion(self, ctx, event):
        if getattr(self, "_session", None) is None:
            return None
        dx, dy = event.x - self._start[0], event.y - self._start[1]
        if event.has(MOD_SHIFT):
            if abs(dx) >= abs(dy):
                dy = 0
            else:
                dx = 0
        scope = self._origin if self._last is None else self._origin.union(self._last)
        self._session.restore(scope)
        self._session.touch(self._origin)
        ctx.cell.erase_masked(self._origin, None)
        drawn = self._place(ctx, dx, dy)
        return scope if drawn is None else scope.union(drawn)

    def on_end(self, ctx, event):
        if getattr(self, "_session", None) is None:
            return None
        self.on_motion(ctx, event)
        cmd = self._session.commit()
        self._session = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        if getattr(self, "_session", None) is None:
            return None
        rect = self._session.cancel()
        self._session = None
        return rect


class PuppetWarpTool(Tool):
    """Pin the layer and drag the pins: Photoshop's Puppet Warp.

    Click to place pins on what should bend (an arm, a tail, a branch);
    drag a pin and the image follows as rigidly as the other pins allow --
    moving least squares deformation, so nothing shears or tears. Alt-click
    a pin to remove it. Enter, a double-click or switching tools applies;
    Escape puts everything back.
    """

    name = "puppet_warp"
    label = "Puppet Warp"
    wants_stroke = False
    multi_click = True
    commit_on_switch = True
    snaps = False
    options_spec = (O.Opt("mode", O.CHOICE, "rigid", "Mode", choices=("rigid", "normal"),
                          labels={"rigid": "Rigid", "normal": "Normal (allows stretch)"},
                          tip="Rigid keeps shapes from scaling; Normal lets them stretch "
                              "between pins."),
                    _interp())

    def __init__(self, **options):
        super().__init__(**options)
        self._reset()

    def _reset(self):
        self._orig = []
        self._pins = []
        self._session = None
        self._snapshot = None
        self._snapshot_index = None
        self._drag = None

    def overlay(self):
        out = []
        for (ox, oy), (x, y) in zip(self._orig, self._pins):
            if (ox, oy) != (x, y):
                out.append(("line", (ox, oy, x, y)))
        if self._pins:
            out.append(("handles", [tuple(p) for p in self._pins]))
        return out

    def _lift(self, ctx):
        cell = ctx.cell
        self._snapshot = cell.plane("rgba").copy()
        self._snapshot_index = cell.plane("index").copy() if cell.index_locked else None
        self._session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)

    def on_begin(self, ctx, event):
        if ctx.cell is None:
            self.active = False
            return None
        if self._session is None:
            self._lift(ctx)
        pt = (event.x, event.y)
        r = max(3.0, float(getattr(ctx, "pick_radius", 4.0)) * 1.4)
        for i, p in enumerate(self._pins):
            if math.hypot(pt[0] - p[0], pt[1] - p[1]) <= r:
                if event.has(4):                       # Alt: remove the pin
                    del self._pins[i]
                    del self._orig[i]
                    return self._render(ctx, final=False)
                self._drag = {"i": i, "start": pt, "from": tuple(p)}
                return None
        self._orig.append(pt)
        self._pins.append(pt)
        self._drag = {"i": len(self._pins) - 1, "start": pt, "from": pt}
        return None

    def on_motion(self, ctx, event):
        if self._drag is None:
            return None
        i = self._drag["i"]
        self._pins[i] = (self._drag["from"][0] + event.x - self._drag["start"][0],
                         self._drag["from"][1] + event.y - self._drag["start"][1])
        return self._render(ctx, final=False)

    def on_end(self, ctx, event):
        rect = self.on_motion(ctx, event)
        self._drag = None
        self.active = True
        return rect

    def _render(self, ctx, final):
        cell = ctx.cell
        bounds = cell.bounds
        self._session.restore(bounds)
        self._session.touch(bounds)
        rigid = self.option("mode", "rigid") == "rigid"
        sx, sy = warp.mls_field(cell.width, cell.height, self._orig, self._pins, rigid)
        if cell.index_locked and self._snapshot_index is not None:
            idx = warp.sample_index(self._snapshot_index, sx, sy)
            clear = 0 if cell.palette is None or cell.palette.transparent is None \
                else int(cell.palette.transparent)
            cell.plane("index")[...] = np.where(idx >= 0, idx, clear).astype(np.uint8)
            cell._after_write(bounds, "index")
        else:
            method = warp.BILINEAR if final and self.option("interpolation", "bilinear") \
                == "bilinear" else warp.NEAREST
            cell.plane("rgba")[...] = warp.warp_field(self._snapshot, sx, sy, method)
            cell._touched(bounds)
        return bounds

    def finish(self, ctx):
        if self._session is None:
            self._reset()
            self.active = False
            return None
        moved = any(a != b for a, b in zip(self._orig, self._pins))
        if moved:
            self._render(ctx, final=True)
            cmd = self._session.commit()
            if cmd is not None and ctx.history is not None:
                ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        else:
            self._session.cancel()
        self._reset()
        self.active = False
        return ctx.doc.bounds

    def on_cancel(self, ctx):
        rect = None
        if self._session is not None:
            rect = self._session.cancel()
        self._reset()
        return rect or ctx.doc.bounds
