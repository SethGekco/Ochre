# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Selections as antialiased coverage masks.

A selection is a full-canvas (H, W) uint8 coverage plane, or None meaning
"everything". Mask rather than path, for three reasons that compound:

  - Boolean combination of a lasso with a magic wand with an ellipse is
    trivial on masks and genuinely hard on paths.
  - The magic wand has no path representation at all; it is inherently a
    per-pixel result.
  - Antialiasing IS coverage. A value of 0..255 is the antialiased edge,
    natively. A path would have to rasterise to exactly this before any
    pixel operation could use it, so a path representation is a mask plus
    extra state to keep in sync.

`mask is None` is the load-bearing optimisation. "No selection" is
overwhelmingly the common state, and an identity check keeps it free in the
brush hot path -- no 16 MB allocation, no masking work, no per-pixel test.

Combine modes are multiplicative rather than boolean so antialiased edges
survive combination. Boolean AND/OR on a 0/255 mask would hard-edge every
combined selection, which is exactly the artefact users notice.
"""

import numpy as np

from .arith import mul255
from .geometry import Rect

REPLACE = "Replace"
UNION = "Union"
EXCLUDE = "Exclude"
INTERSECT = "Intersect"
XOR = "Xor"

MODES = (REPLACE, UNION, EXCLUDE, INTERSECT, XOR)


def rgss_offsets(quality=4):
    """Rotated-grid supersampling offsets, Paint.NET's GetRgssOffsets.

    A rotated grid beats an aligned NxN grid at the same sample count on
    near-horizontal and near-vertical edges, which is where aliasing is most
    visible. Same cost, better result, twenty lines.
    """
    n = quality * quality
    out = []
    for i in range(n):
        y = (i + 1.0) / (n + 1.0)
        x = y * quality
        x -= int(x)
        out.append((x - 0.5, y - 0.5))
    return out


def symmetric_offsets(quality=4):
    """RGSS mirrored into all four quadrants, so the pattern is symmetric.

    Plain RGSS is unbiased -- its centroid sits exactly on the pixel centre --
    but it is not MIRROR-symmetric, and that shows. Rasterising a circle with
    it produces rim coverage differing by up to 16/255 between the left and
    right halves. Sub-pixel, but a visibly lopsided circle is precisely the
    artefact that looks broken in pixel art.

    Taking a quarter-size rotated grid and reflecting it into all four
    quadrants restores exact symmetry under both mirrors while keeping the
    rotation that makes RGSS worth using, at the same total sample count.
    """
    base = rgss_offsets(max(1, quality // 2))
    out = []
    for x, y in base:
        ax, ay = abs(x), abs(y)
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                out.append((sx * ax, sy * ay))
    return out


# ---- rasterisers --------------------------------------------------------

def rasterize_rect(width, height, rect, antialias=True):
    """Exact fractional coverage for an axis-aligned rectangle.

    Analytic rather than supersampled: a rectangle's edge coverage is exactly
    the overlapped fraction of each pixel, so there is no reason to approximate
    it.  `rect` may carry float bounds.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = float(rect[0]), float(rect[1])
    x1, y1 = float(rect[2]), float(rect[3])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return mask

    if not antialias:
        xi0, yi0 = max(0, int(round(x0))), max(0, int(round(y0)))
        xi1, yi1 = min(width, int(round(x1))), min(height, int(round(y1)))
        if xi1 > xi0 and yi1 > yi0:
            mask[yi0:yi1, xi0:xi1] = 255
        return mask

    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    cov_x = np.clip(np.minimum(xs + 1.0, x1) - np.maximum(xs, x0), 0.0, 1.0)
    cov_y = np.clip(np.minimum(ys + 1.0, y1) - np.maximum(ys, y0), 0.0, 1.0)
    cov = np.outer(cov_y, cov_x)
    return np.rint(cov * 255.0).astype(np.uint8)


def _supersample(width, height, inside_fn, bounds, quality=4, origin=(0.0, 0.0)):
    """Accumulate coverage by testing `quality**2` samples per pixel.

    Only pixels inside `bounds` are tested; everything else is known empty,
    which is what keeps a small ellipse on a large canvas cheap.

    `origin` is subtracted from the pixel centre BEFORE the sample offset is
    added, and that ordering is load-bearing rather than stylistic. Computing
    (centre + offset) - origin instead lets mirror-image samples round
    differently: a circle's rim evaluated at 0.9999999999999999 on one side
    and 1.0000000000000002 on the other straddles a `<= 1.0` test and
    produces a visibly lopsided circle. Centring first makes the two exactly
    negated, because IEEE negation is exact and `9.5 - 0.1` rounds the same
    way whichever sign it carries. Callers that do not care pass (0, 0).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    r = bounds.clipped_to(width, height)
    if r is None:
        return mask
    ox, oy = float(origin[0]), float(origin[1])
    ys = (np.arange(r.y, r.y1, dtype=np.float64) + 0.5 - oy)[:, None]
    xs = (np.arange(r.x, r.x1, dtype=np.float64) + 0.5 - ox)[None, :]
    # Symmetric, so a circle rasterises identically in all four quadrants.
    offsets = symmetric_offsets(quality)
    acc = np.zeros((r.h, r.w), dtype=np.float64)
    for dx, dy in offsets:
        acc += inside_fn(xs + dx, ys + dy)
    acc /= len(offsets)
    mask[r.slice()] = np.rint(acc * 255.0).astype(np.uint8)
    return mask


def rasterize_ellipse(width, height, rect, antialias=True, quality=4):
    """Coverage for an ellipse inscribed in rect."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) / 2.0, 1e-9), max((y1 - y0) / 2.0, 1e-9)
    bounds = Rect.from_bounds(int(np.floor(x0)), int(np.floor(y0)),
                              int(np.ceil(x1)) + 1, int(np.ceil(y1)) + 1)

    def inside(ux, uy):
        # ux, uy are already relative to the ellipse centre.
        return (((ux / rx) ** 2 + (uy / ry) ** 2) <= 1.0).astype(np.float64)

    return _supersample(width, height, inside, bounds,
                        quality if antialias else 1, origin=(cx, cy))


def rasterize_polygon(width, height, points, antialias=True, quality=4):
    """Even-odd coverage for a closed polygon. Backs the lasso."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return np.zeros((height, width), dtype=np.uint8)
    bounds = Rect.from_bounds(int(np.floor(pts[:, 0].min())),
                              int(np.floor(pts[:, 1].min())),
                              int(np.ceil(pts[:, 0].max())) + 1,
                              int(np.ceil(pts[:, 1].max())) + 1)
    x1s, y1s = pts[:, 0], pts[:, 1]
    x2s, y2s = np.roll(x1s, -1), np.roll(y1s, -1)

    def inside(px, py):
        acc = np.zeros(np.broadcast(px, py).shape, dtype=bool)
        for i in range(len(x1s)):
            ax, ay, bx, by = x1s[i], y1s[i], x2s[i], y2s[i]
            if ay == by:
                continue
            straddles = (ay > py) != (by > py)
            # Horizontal position of the edge at this scanline.
            cross = (bx - ax) * (py - ay) / (by - ay) + ax
            acc ^= straddles & (px < cross)
        return acc.astype(np.float64)

    return _supersample(width, height, inside, bounds,
                        quality if antialias else 1)


# ---- the selection ------------------------------------------------------

class Selection:
    """Document-level. Spans frames, so one marquee applies across an animation."""

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.mask = None            # None means "everything selected"
        self._bbox = None
        self.geometry = []          # retained shapes, for re-editing and save

    # ---- state -----------------------------------------------------------

    def selects_all(self):
        """True when there is no active selection. The hot-path fast check."""
        return self.mask is None

    def is_empty(self):
        return self.mask is not None and not self.mask.any()

    @property
    def bounds(self):
        return Rect(0, 0, self.width, self.height)

    @property
    def bbox(self):
        """Tight bounds of the selected region, or the whole canvas if none."""
        if self.mask is None:
            return self.bounds
        if self._bbox is None:
            ys, xs = np.nonzero(self.mask)
            if ys.size == 0:
                self._bbox = Rect.empty()
            else:
                self._bbox = Rect.from_points(int(xs.min()), int(ys.min()),
                                              int(xs.max()), int(ys.max()))
        return self._bbox

    def select_all(self):
        self.mask = None
        self._bbox = None
        self.geometry = []
        return self

    def clear(self):
        """Deselect everything. Distinct from select_all: nothing is writable."""
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self._bbox = None
        self.geometry = []
        return self

    def invert(self):
        if self.mask is None:
            return self.clear()
        self.mask = (255 - self.mask).astype(np.uint8)
        self._bbox = None
        return self

    # ---- masking ---------------------------------------------------------

    def mask_for(self, rect):
        """Coverage over rect, or None when everything is selected.

        Returning None rather than an all-255 array is the point: callers
        branch once on None and skip the multiply entirely.
        """
        if self.mask is None:
            return None
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        return self.mask[r.slice()]

    def combine_mask(self, other, rect):
        """Fuse the selection into a tool's own coverage over rect.

        One multiply, and selection clipping is done. Antialiased selection
        edges therefore give antialiased clipping for free, and there is no
        separate clipping path anywhere in the engine to keep in sync.
        """
        sel = self.mask_for(rect)
        if sel is None:
            return other
        if other is None:
            return sel
        return mul255(other, sel)

    # ---- combination -----------------------------------------------------

    def combine(self, new_mask, mode=REPLACE):
        """Apply a freshly-rasterised mask under a combine mode.

        Multiplicative forms throughout, so a feathered edge stays feathered
        through any sequence of combinations.
        """
        new_mask = np.asarray(new_mask, dtype=np.uint8)
        if mode == REPLACE:
            self.mask = new_mask
        else:
            old = (self.mask if self.mask is not None
                   else np.full((self.height, self.width), 255, dtype=np.uint8))
            if mode == UNION:
                self.mask = np.maximum(old, new_mask)
            elif mode == INTERSECT:
                self.mask = mul255(old, new_mask)
            elif mode == EXCLUDE:
                self.mask = mul255(old, (255 - new_mask).astype(np.uint8))
            elif mode == XOR:
                both = mul255(old, new_mask).astype(np.int32)
                self.mask = np.clip(old.astype(np.int32) + new_mask.astype(np.int32)
                                    - 2 * both, 0, 255).astype(np.uint8)
            else:
                raise ValueError("unknown combine mode %r" % (mode,))
        self._bbox = None
        # A selection covering everything is the same as no selection, and
        # collapsing to None puts the hot path back on its fast route.
        if self.mask is not None and self.mask.min() == 255:
            self.mask = None
        return self

    # ---- shape helpers ---------------------------------------------------

    def select_rect(self, rect, mode=REPLACE, antialias=True):
        mask = rasterize_rect(self.width, self.height, rect, antialias)
        self.geometry.append(("rect", tuple(rect), mode))
        return self.combine(mask, mode)

    def select_ellipse(self, rect, mode=REPLACE, antialias=True, quality=4):
        mask = rasterize_ellipse(self.width, self.height, rect, antialias, quality)
        self.geometry.append(("ellipse", tuple(rect), mode))
        return self.combine(mask, mode)

    def select_polygon(self, points, mode=REPLACE, antialias=True, quality=4):
        mask = rasterize_polygon(self.width, self.height, points, antialias, quality)
        self.geometry.append(("polygon", tuple(map(tuple, points)), mode))
        return self.combine(mask, mode)

    def select_stencil(self, stencil, mode=REPLACE):
        """Adopt a boolean stencil, e.g. from a flood fill."""
        mask = (np.asarray(stencil, dtype=bool).astype(np.uint8)) * 255
        return self.combine(mask, mode)

    def feather(self, radius):
        """Soften the edge with a separable box blur.

        A box pass is deliberate over a true Gaussian: it is what a coverage
        mask wants, it is separable and cheap, and the difference is invisible
        at the radii anyone feathers a selection by.
        """
        if self.mask is None or radius <= 0:
            return self
        k = int(radius) * 2 + 1
        acc = self.mask.astype(np.float64)
        pad = int(radius)
        for axis in (0, 1):
            padded = np.pad(acc, [(pad, pad) if a == axis else (0, 0)
                                  for a in range(2)], mode="edge")
            csum = np.cumsum(padded, axis=axis)
            lo = np.take(csum, np.arange(0, acc.shape[axis]), axis=axis)
            hi = np.take(csum, np.arange(k - 1, k - 1 + acc.shape[axis]), axis=axis)
            acc = (hi - lo + np.take(padded, np.arange(0, acc.shape[axis]),
                                     axis=axis)) / float(k)
        self.mask = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
        self._bbox = None
        return self

    def copy(self):
        out = Selection(self.width, self.height)
        out.mask = None if self.mask is None else self.mask.copy()
        out.geometry = list(self.geometry)
        return out
