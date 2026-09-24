# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Gradients.

The decomposition is Paint.NET's, and it is worth copying because it reduces
five gradient types to two scalar functions each:

    compute_t(x, y)   the geometry -- where this pixel sits along the gradient
    bound_t(t)        the policy   -- how values outside 0..1 are treated

Everything else -- the colour ramp, the alpha handling, the compositing -- is
shared. A new gradient type is about fifteen lines.

One detail is load-bearing and easy to get wrong. The 256-entry colour ramp
is built with ALPHA-WEIGHTED interpolation, not per-channel lerp. Paint.NET
ships both and its own documentation warns that the naive one "does not
properly take into account the alpha channel's effect on color blending";
the gradient renderer correctly uses the other. Interpolating from an opaque
red to a transparent blue with a plain lerp drags the invisible blue's colour
into the visible end and greys the whole ramp.

There is also an ALPHA-ONLY mode, which leaves colour alone and gradients
only the alpha channel. That is what makes a gradient usable as a soft mask,
and most editors do not have it.
"""

import math

import numpy as np

LINEAR = "linear"
REFLECTED = "reflected"
DIAMOND = "diamond"
RADIAL = "radial"
CONICAL = "conical"
TYPES = (LINEAR, REFLECTED, DIAMOND, RADIAL, CONICAL)

REPEAT_CLAMP = "clamp"
REPEAT_REPEAT = "repeat"
REPEAT_REFLECT = "reflect"
REPEATS = (REPEAT_CLAMP, REPEAT_REPEAT, REPEAT_REFLECT)


def build_ramp(start, end, steps=256):
    """A colour ramp, interpolated with alpha weighting.

    Returns (steps, 4) uint8. The colour channels are weighted by each end's
    alpha and divided by the interpolated alpha, so a ramp into transparency
    keeps its visible colour instead of fading toward whatever the invisible
    end happened to be.
    """
    t = (np.arange(steps, dtype=np.float64) / (steps - 1))[:, None]
    c0 = np.asarray(start, dtype=np.float64)
    c1 = np.asarray(end, dtype=np.float64)

    a0, a1 = c0[3], c1[3]
    out_a = a0 * (1.0 - t[:, 0]) + a1 * t[:, 0]

    w0 = a0 * (1.0 - t[:, 0])
    w1 = a1 * t[:, 0]
    total = w0 + w1

    out = np.zeros((steps, 4), dtype=np.float64)
    nonzero = total > 1e-9
    for c in range(3):
        num = c0[c] * w0 + c1[c] * w1
        # Where both ends are transparent there is no colour information at
        # all; fall back to the plain midpoint rather than dividing by zero.
        plain = c0[c] * (1.0 - t[:, 0]) + c1[c] * t[:, 0]
        out[:, c] = np.where(nonzero, num / np.where(nonzero, total, 1.0), plain)
    out[:, 3] = out_a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class Gradient:
    """One gradient: geometry, ramp policy, and rendering."""

    def __init__(self, kind=LINEAR, start=(0, 0), end=(1, 0),
                 start_colour=(0, 0, 0, 255), end_colour=(255, 255, 255, 255),
                 repeat=REPEAT_CLAMP, alpha_only=False, reverse=False):
        if kind not in TYPES:
            raise ValueError("unknown gradient type %r" % (kind,))
        if repeat not in REPEATS:
            raise ValueError("unknown repeat mode %r" % (repeat,))
        self.kind = kind
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.start_colour = tuple(start_colour)
        self.end_colour = tuple(end_colour)
        self.repeat = repeat
        self.alpha_only = bool(alpha_only)
        self.reverse = bool(reverse)

    # ---- geometry --------------------------------------------------------

    def compute_t(self, xs, ys):
        """Unbounded position along the gradient. May fall outside 0..1."""
        x0, y0 = self.start
        x1, y1 = self.end
        dx, dy = x1 - x0, y1 - y0

        if self.kind == RADIAL:
            radius = math.hypot(dx, dy)
            if radius < 1e-9:
                return np.zeros(np.broadcast(xs, ys).shape, dtype=np.float64)
            return np.hypot(xs - x0, ys - y0) / radius

        if self.kind == CONICAL:
            # Normalised so the seam lands on the end point rather than at an
            # arbitrary compass direction.
            base = math.atan2(dy, dx)
            ang = np.arctan2(ys - y0, xs - x0) - base
            return ((ang / (2.0 * math.pi)) + 1.0) % 1.0

        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return np.zeros(np.broadcast(xs, ys).shape, dtype=np.float64)

        along = ((xs - x0) * dx + (ys - y0) * dy) / length_sq
        if self.kind in (LINEAR, REFLECTED):
            return along
        # Diamond: distance along both the axis and its perpendicular.
        across = ((xs - x0) * -dy + (ys - y0) * dx) / length_sq
        return np.abs(along) + np.abs(across)

    def bound_t(self, t):
        """Map unbounded t into 0..1 according to type and repeat policy."""
        if self.kind == REFLECTED:
            t = np.abs(t)
        if self.repeat == REPEAT_REPEAT:
            t = np.mod(t, 1.0)
        elif self.repeat == REPEAT_REFLECT:
            t = np.abs(np.mod(t + 1.0, 2.0) - 1.0)
        else:
            t = np.clip(t, 0.0, 1.0)
        return t

    # ---- rendering -------------------------------------------------------

    def render(self, rect, base=None):
        """Render over `rect`, returning (h, w, 4) uint8.

        With `alpha_only`, `base` supplies the colour being masked: the
        gradient scales its alpha and leaves the RGB untouched, which is how
        a gradient becomes a soft mask rather than a paint.
        """
        ys = np.arange(rect.y, rect.y1, dtype=np.float64)[:, None] + 0.5
        xs = np.arange(rect.x, rect.x1, dtype=np.float64)[None, :] + 0.5

        t = self.bound_t(self.compute_t(xs, ys))
        if self.reverse:
            t = 1.0 - t
        index = np.clip(np.rint(t * 255.0), 0, 255).astype(np.uint8)
        index = np.broadcast_to(index, (rect.h, rect.w))

        ramp = build_ramp(self.start_colour, self.end_colour)

        if not self.alpha_only:
            return np.ascontiguousarray(ramp[index])

        out = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        if base is None:
            out[..., :3] = np.asarray(self.start_colour, dtype=np.uint8)[:3]
        else:
            out[..., :3] = np.asarray(base, dtype=np.uint8)[..., :3]
        out[..., 3] = ramp[index][..., 3]
        return out

    def coverage(self, rect):
        """Just the alpha channel, as a coverage plane.

        Useful for driving a mask directly rather than painting colour.
        """
        ys = np.arange(rect.y, rect.y1, dtype=np.float64)[:, None] + 0.5
        xs = np.arange(rect.x, rect.x1, dtype=np.float64)[None, :] + 0.5
        t = self.bound_t(self.compute_t(xs, ys))
        if self.reverse:
            t = 1.0 - t
        return np.clip(np.rint(np.broadcast_to(t, (rect.h, rect.w)) * 255.0),
                       0, 255).astype(np.uint8)
