# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Brushes that change the pixels under them instead of laying down colour.

Blur, sharpen, smudge, dodge, burn, sponge, recolor, background eraser,
history brush, mixer brush and heal are one design: a round dab whose
coverage says HOW MUCH, and a `process()` that says WHAT -- given the pixels
under the dab (plus a halo, for anything that looks at neighbours), return
what they should become. The base class does everything else: dab spacing,
the stroke session that makes one stroke one undo entry, selection clipping,
and the write.

The write is Surface.lerp_masked, not apply_masked. Painting composites
colour OVER what is there, which can only ever add; a blur has to be able to
make a pixel more transparent and a dodge has to be able to move it to any
value. Strength scales the coverage, so a weak brush and a soft edge are the
same mechanism.

Everything here works on index-locked layers too, because lerp_masked lands
its result as palette indices. That is not always what a pixel artist wants
-- blurring a 16-colour sprite snaps to the nearest entries -- but it is the
honest result, and it is never silently discarded on save.

Also here: the airbrush and ink pens (colour-laying brushes with their own
dynamics), the pattern stamp, the magic eraser and red-eye removal.
"""

import math

import numpy as np

from .. import accel
from ..geometry import Rect
from . import options as O
from .base import MOD_CTRL, Tool
from .paint import BrushTool, _circle_coverage, _StrokeTool


# ---- small image kernels, pure numpy ----------------------------------------------

def _box1d(a, r, axis):
    """Mean over a (2r+1) window along an axis, edges clamped. Float in, float out."""
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    padded = np.pad(a, pad, mode="edge")
    c = np.cumsum(padded, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur_rgba(rgba, radius, passes=2):
    """Alpha-weighted box blur (two passes approximate a gaussian).

    Premultiplied while filtering, divided back out after -- averaging
    straight colour across differing alpha is what produces dark fringes
    around every transparent edge.
    """
    radius = int(max(0, radius))
    a = rgba[..., 3].astype(np.float64) / 255.0
    work = np.empty(rgba.shape[:2] + (4,), dtype=np.float64)
    work[..., :3] = rgba[..., :3].astype(np.float64) * a[..., None]
    work[..., 3] = a
    for _ in range(passes):
        work = _box1d(work, radius, 0)
        work = _box1d(work, radius, 1)
    alpha = work[..., 3]
    safe = np.where(alpha > 1e-6, alpha, 1.0)
    out = np.empty(rgba.shape, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(work[..., :3] / safe[..., None]), 0, 255).astype(np.uint8)
    out[..., :3][alpha <= 1e-6] = 0
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return out


def luminance(rgb):
    rgb = rgb.astype(np.float64)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


def _dab_window(cell, x, y, size, hardness, antialias=True):
    cov = _circle_coverage(size, hardness, antialias)
    d = cov.shape[0]
    ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
    clipped = Rect(ox, oy, d, d).clipped_to(cell.width, cell.height)
    if clipped is None:
        return None, None, None
    window = cov[clipped.y - oy:clipped.y - oy + clipped.h,
                 clipped.x - ox:clipped.x - ox + clipped.w]
    return clipped, window, (ox, oy, d)


def _scale_cov(cov, amount):
    amount = min(max(float(amount), 0.0), 1.0)
    if amount >= 1.0:
        return cov
    return accel.spec._mul(cov.astype(np.uint32),
                           np.uint32(int(round(amount * 255)))).astype(np.uint8)


def _crop(patch, big, rect):
    """The `rect` part of a patch that was read over `big`."""
    return patch[rect.y - big.y:rect.y - big.y + rect.h,
                 rect.x - big.x:rect.x - big.x + rect.w]


def _strength(default=0.5, label="Strength"):
    return O.Opt("strength", O.FLOAT, default, label, 0.01, 1.0, 0.05, percent=True)


# ---- the filter brush -------------------------------------------------------------

class _FilterBrush(_StrokeTool):
    """A dab that transforms the pixels under it. Subclasses write process()."""

    default_size = 24
    default_hardness = 0.5
    default_strength = 0.5
    options_spec = (O.size(24), O.hardness(0.5), _strength())

    def halo(self, size):
        """Extra pixels around the dab that process() needs to see."""
        return 0

    def process(self, ctx, event, src, big, rect, x, y):
        """src: uint8 RGBA over `big` (rect plus halo). Return rect-sized RGBA."""
        raise NotImplementedError

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1.0, float(self.option("size", self.default_size)))
        rect, cov, _ = _dab_window(cell, x, y, size,
                                   float(self.option("hardness", self.default_hardness)))
        if rect is None:
            return None
        cov = _scale_cov(cov, self.option("strength", self.default_strength))
        cov = ctx.coverage(cov, rect)
        if cov is not None and not cov.any():
            return None
        h = int(self.halo(size))
        big = rect.inflated(h).clipped_to(cell.width, cell.height) if h else rect
        src = cell.plane("rgba")[big.slice()].copy()
        out = self.process(ctx, event, src, big, rect, x, y)
        if out is None:
            return None
        self._session.touch(rect)
        cell.lerp_masked(rect, out, cov)
        return rect


class BlurBrushTool(_FilterBrush):
    """Soften where you paint. GIMP's Blur, Photoshop's Blur tool."""

    name = "blur"
    label = "Blur"

    def halo(self, size):
        return max(1, int(size / 8)) * 2 + 1

    def process(self, ctx, event, src, big, rect, x, y):
        r = max(1, int(float(self.option("size", 24)) / 8))
        return _crop(blur_rgba(src, r), big, rect)


class SharpenBrushTool(_FilterBrush):
    """Crisp up where you paint: an unsharp mask applied per dab."""

    name = "sharpen"
    label = "Sharpen"
    options_spec = (O.size(24), O.hardness(0.5), _strength(0.3))

    def halo(self, size):
        return 3

    def process(self, ctx, event, src, big, rect, x, y):
        soft = blur_rgba(src, 1).astype(np.int32)
        orig = src.astype(np.int32)
        out = orig.copy()
        out[..., :3] = np.clip(orig[..., :3] + (orig[..., :3] - soft[..., :3]), 0, 255)
        return _crop(out.astype(np.uint8), big, rect)


class SmudgeTool(_FilterBrush):
    """Drag paint along the stroke, like a finger through wet paint.

    Keeps a buffer the size of the dab that carries colour from one dab to
    the next: each dab paints the buffer down and then picks up some of what
    it landed on, which is what makes the smear trail off naturally.
    """

    name = "smudge"
    label = "Smudge"
    options_spec = (O.size(24), O.hardness(0.5), _strength(0.6, "Strength"))

    def on_begin(self, ctx, event):
        self._buf = None
        return super().on_begin(ctx, event)

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1.0, float(self.option("size", 24)))
        rect, cov, (ox, oy, d) = _dab_window(cell, x, y, size,
                                             float(self.option("hardness", 0.5)))
        if rect is None:
            return None
        here = np.zeros((d, d, 4), dtype=np.float64)
        sub = (slice(rect.y - oy, rect.y - oy + rect.h), slice(rect.x - ox, rect.x - ox + rect.w))
        here[sub] = cell.plane("rgba")[rect.slice()]
        rate = float(self.option("strength", 0.6))
        if getattr(self, "_buf", None) is None or self._buf.shape != here.shape:
            self._buf = here.copy()
            return None          # the first dab only loads the finger
        out = np.clip(np.rint(self._buf[sub]), 0, 255).astype(np.uint8)
        cov = ctx.coverage(cov, rect)
        self._session.touch(rect)
        cell.lerp_masked(rect, out, cov)
        # Pick up some of what was under the finger before it moved.
        self._buf = self._buf * rate + here * (1.0 - rate)
        return rect


_RANGES = ("shadows", "midtones", "highlights")


def _range_weight(lum, which):
    if which == "shadows":
        return np.clip(1.0 - lum * 1.6, 0.0, 1.0) ** 1.5
    if which == "highlights":
        return np.clip((lum - 0.375) * 1.6, 0.0, 1.0) ** 1.5
    return np.clip(1.0 - np.abs(lum - 0.5) * 2.0, 0.0, 1.0)


class DodgeTool(_FilterBrush):
    """Lighten where you paint, mostly in the chosen tonal range."""

    name = "dodge"
    label = "Dodge"
    options_spec = (O.size(32), O.hardness(0.4), _strength(0.5, "Exposure"),
                    O.Opt("range", O.CHOICE, "midtones", "Range", choices=_RANGES))
    sign = +1

    def process(self, ctx, event, src, big, rect, x, y):
        rgb = src[..., :3].astype(np.float64)
        w = _range_weight(luminance(src), self.option("range", "midtones"))[..., None]
        if self.sign > 0:
            rgb = rgb + (255.0 - rgb) * 0.5 * w
        else:
            rgb = rgb * (1.0 - 0.5 * w)
        out = src.copy()
        out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return out


class BurnTool(DodgeTool):
    """Darken where you paint, mostly in the chosen tonal range."""

    name = "burn"
    label = "Burn"
    options_spec = (O.size(32), O.hardness(0.4), _strength(0.5, "Exposure"),
                    O.Opt("range", O.CHOICE, "midtones", "Range", choices=_RANGES))
    sign = -1


class SpongeTool(_FilterBrush):
    """Saturate or desaturate where you paint."""

    name = "sponge"
    label = "Sponge"
    options_spec = (O.size(32), O.hardness(0.4), _strength(0.5, "Flow"),
                    O.Opt("mode", O.CHOICE, "desaturate", "Mode",
                          choices=("saturate", "desaturate")))

    def process(self, ctx, event, src, big, rect, x, y):
        rgb = src[..., :3].astype(np.float64)
        grey = (luminance(src) * 255.0)[..., None]
        factor = 1.6 if self.option("mode", "desaturate") == "saturate" else 0.0
        out = src.copy()
        out[..., :3] = np.clip(np.rint(grey + (rgb - grey) * factor), 0, 255).astype(np.uint8)
        return out


def _colour_distance(rgba, ref):
    d = rgba[..., :3].astype(np.float64) - np.asarray(ref[:3], dtype=np.float64)
    return np.sqrt((d * d).sum(axis=-1)) / math.sqrt(3.0)       # 0..255


class RecolorTool(_FilterBrush):
    """Swap one colour for another, keeping the shading. Paint.NET's Recolor,
    Photoshop's Color Replacement.

    By default it replaces the SECONDARY colour with the primary (right
    button: the other way round), like Paint.NET. "Once" samples the colour
    under the brush when the stroke starts, "Continuous" at every dab, which
    is Photoshop's behaviour. The shift is applied rather than the colour
    stamped, so a shaded red becomes a similarly shaded blue.
    """

    name = "recolor"
    label = "Recolor"
    default_strength = 1.0
    options_spec = (O.size(24), O.hardness(0.6), O.tolerance(48),
                    O.Opt("sampling", O.CHOICE, "secondary", "Sampling",
                          choices=("secondary", "once", "continuous"),
                          labels={"secondary": "Secondary colour", "once": "Once",
                                  "continuous": "Continuous"}))

    def on_begin(self, ctx, event):
        self._target = None
        return super().on_begin(ctx, event)

    def _colours(self, ctx, event, src, big, x, y):
        new = ctx.colour_for(event)
        mode = self.option("sampling", "secondary")
        if mode == "secondary":
            target = ctx.primary if event.button == 2 else ctx.secondary
            return target, new
        cx = min(max(int(x) - big.x, 0), src.shape[1] - 1)
        cy = min(max(int(y) - big.y, 0), src.shape[0] - 1)
        if mode == "continuous" or self._target is None:
            self._target = tuple(int(v) for v in src[cy, cx])
        return self._target, new

    def process(self, ctx, event, src, big, rect, x, y):
        target, new = self._colours(ctx, event, src, big, x, y)
        tol = max(1.0, float(self.option("tolerance", 48)))
        w = np.clip(1.0 - _colour_distance(src, target) / tol, 0.0, 1.0)[..., None]
        shift = np.asarray(new[:3], np.float64) - np.asarray(target[:3], np.float64)
        rgb = src[..., :3].astype(np.float64)
        out = src.copy()
        out[..., :3] = np.clip(np.rint(rgb + shift * w), 0, 255).astype(np.uint8)
        return _crop(out, big, rect)


class BackgroundEraserTool(_FilterBrush):
    """Erase only pixels resembling the colour under the brush's centre.

    Photoshop's Background Eraser: run it along the edge of a subject and it
    takes the background and leaves the subject, because only colours near
    the sampled one are erased.
    """

    name = "background_eraser"
    label = "Background Eraser"
    default_strength = 1.0
    options_spec = (O.size(32), O.hardness(0.7), O.tolerance(40),
                    O.Opt("sampling", O.CHOICE, "continuous", "Sampling",
                          choices=("once", "continuous")))

    def on_begin(self, ctx, event):
        self._target = None
        return super().on_begin(ctx, event)

    def process(self, ctx, event, src, big, rect, x, y):
        cx = min(max(int(x) - big.x, 0), src.shape[1] - 1)
        cy = min(max(int(y) - big.y, 0), src.shape[0] - 1)
        if self.option("sampling", "continuous") == "continuous" or self._target is None:
            self._target = tuple(int(v) for v in src[cy, cx])
        tol = max(1.0, float(self.option("tolerance", 40)))
        w = np.clip(1.0 - _colour_distance(src, self._target) / tol, 0.0, 1.0)
        out = src.copy()
        out[..., 3] = np.clip(np.rint(src[..., 3] * (1.0 - w)), 0, 255).astype(np.uint8)
        return out


class HistoryBrushTool(_FilterBrush):
    """Paint the layer back toward an earlier state.

    The source is the document as it was opened, or whatever "Set History
    Brush Source" last captured. Photoshop's History Brush; the closest GIMP
    gets is painting from a duplicate layer by hand.
    """

    name = "history_brush"
    label = "History Brush"
    default_strength = 1.0
    options_spec = (O.size(24), O.hardness(0.6), _strength(1.0, "Opacity"))

    def on_begin(self, ctx, event):
        self._source = None
        if ctx.history_source is not None and ctx.layer is not None:
            self._source = ctx.history_source(ctx.layer.id, ctx.frame.id)
        if self._source is None:
            self.active = False
            return {"error": "no history source for this layer"}
        return super().on_begin(ctx, event)

    def process(self, ctx, event, src, big, rect, x, y):
        s = self._source
        if s is None or s.shape[:2] != (ctx.doc.height, ctx.doc.width):
            return None
        return s[rect.slice()]


class MixerBrushTool(_FilterBrush):
    """Paint that mixes with what is already on the canvas.

    The brush carries a reservoir that starts as the primary colour, takes on
    some of the canvas under every dab (Wet) and is topped back up toward the
    primary (Load). Photoshop's Mixer Brush, reduced to the two controls that
    do most of the work.
    """

    name = "mixer"
    label = "Mixer Brush"
    options_spec = (O.size(24), O.hardness(0.6), _strength(0.8, "Flow"),
                    O.Opt("wet", O.FLOAT, 0.5, "Wet", 0.0, 1.0, 0.05, percent=True),
                    O.Opt("load", O.FLOAT, 0.2, "Load", 0.0, 1.0, 0.05, percent=True))

    def on_begin(self, ctx, event):
        self._reservoir = np.asarray(ctx.colour_for(event), dtype=np.float64)
        return super().on_begin(ctx, event)

    def process(self, ctx, event, src, big, rect, x, y):
        a = src[..., 3].astype(np.float64) / 255.0
        total = a.sum()
        if total > 1e-6:
            under = np.empty(4)
            under[:3] = (src[..., :3].astype(np.float64) * a[..., None]).sum(axis=(0, 1)) / total
            under[3] = src[..., 3].mean()
            wet = float(self.option("wet", 0.5))
            self._reservoir = self._reservoir * (1 - wet) + under * wet
        load = float(self.option("load", 0.2))
        self._reservoir = self._reservoir * (1 - load) + \
            np.asarray(ctx.colour_for(event), np.float64) * load
        colour = np.clip(np.rint(self._reservoir), 0, 255).astype(np.uint8)
        colour[3] = 255
        return np.broadcast_to(colour, (rect.h, rect.w, 4)).copy()


class HealTool(_FilterBrush):
    """Clone texture from a source while keeping the destination's colour.

    Ctrl-click sets the source, then paint, exactly as with the clone stamp.
    The difference is what lands: the source's DETAIL (itself minus its own
    blur) added to the destination's smooth colour (its blur). A blemish
    healed from clear skin nearby takes the skin's texture and the
    surrounding tone, instead of a pasted patch with a visible seam.
    """

    name = "heal"
    label = "Healing Brush"
    default_strength = 1.0
    options_spec = (O.size(24), O.hardness(0.5), _strength(1.0, "Opacity"),
                    O.Opt("aligned", O.BOOL, True, "Aligned"))

    def __init__(self, **options):
        super().__init__(**options)
        self._anchor = None
        self._offset = None

    def on_begin(self, ctx, event):
        if event.has(MOD_CTRL):
            self._anchor = (event.x, event.y)
            self._offset = None
            self.active = False
            return {"anchor": self._anchor}
        if self._anchor is None:
            self.active = False
            return {"error": "set a heal source first (ctrl-click)"}
        if self._offset is None or not self.option("aligned", True):
            self._offset = (self._anchor[0] - event.x, self._anchor[1] - event.y)
        cell = ctx.cell
        self._snapshot = None if cell is None else cell.plane("rgba").copy()
        return super().on_begin(ctx, event)

    def halo(self, size):
        return max(2, int(size / 4))

    def process(self, ctx, event, src, big, rect, x, y):
        snap = self._snapshot
        dx, dy = int(round(self._offset[0])), int(round(self._offset[1]))
        sbig = big.translated(dx, dy)
        if sbig.clipped_to(snap.shape[1], snap.shape[0]) != sbig:
            return None                       # source runs off the canvas
        source = snap[sbig.slice()]
        r = max(1, int(float(self.option("size", 24)) / 8))
        dst_soft = blur_rgba(src, r).astype(np.int32)
        src_soft = blur_rgba(source, r).astype(np.int32)
        out = src.astype(np.int32).copy()
        out[..., :3] = np.clip(dst_soft[..., :3] + source[..., :3].astype(np.int32)
                               - src_soft[..., :3], 0, 255)
        return _crop(out.astype(np.uint8), big, rect)


class RedEyeTool(_FilterBrush):
    """Paint over a red pupil to neutralise it. Only strongly red pixels move."""

    name = "red_eye"
    label = "Red Eye Removal"
    default_strength = 1.0
    options_spec = (O.size(24), O.hardness(0.7), _strength(1.0, "Amount"))

    def process(self, ctx, event, src, big, rect, x, y):
        rgb = src[..., :3].astype(np.float64)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        others = np.maximum(g, b)
        redness = np.clip((r - others * 1.25) / 80.0, 0.0, 1.0)
        out = src.copy()
        out[..., 0] = np.clip(np.rint(r - (r - (g + b) / 2.0) * redness), 0, 255).astype(np.uint8)
        return out


# ---- colour-laying pens with their own dynamics ------------------------------------

class AirbrushTool(BrushTool):
    """Soft, low-flow paint that keeps building while the button is held.

    `continuous` asks the canvas to send ticks while the pointer is still, so
    holding the airbrush in one place darkens the spot -- the property that
    separates an airbrush from a soft paintbrush in GIMP and Photoshop.
    Scatter jitters each dab for the spray look.
    """

    name = "airbrush"
    label = "Airbrush"
    continuous = True
    options_spec = (O.size(40), O.hardness(0.2),
                    O.Opt("flow", O.FLOAT, 0.12, "Rate", 0.01, 1.0, 0.01, percent=True),
                    O.Opt("scatter", O.FLOAT, 0.0, "Scatter", 0.0, 1.0, 0.05, percent=True),
                    O.Opt("pressure_size", O.BOOL, True, "Pressure size"))

    def on_begin(self, ctx, event):
        self._rng = np.random.default_rng(1234)
        self._event = event
        return super().on_begin(ctx, event)

    def _dab(self, ctx, event, x, y):
        scatter = float(self.option("scatter", 0.0))
        if scatter > 0:
            size = float(self.option("size", 40))
            ang = self._rng.uniform(0, 2 * math.pi)
            dist = self._rng.uniform(0, scatter * size / 2.0)
            x, y = x + math.cos(ang) * dist, y + math.sin(ang) * dist
        self._event = event
        return super()._dab(ctx, event, x, y)

    def tick(self, ctx):
        """One more dab where the pointer rests."""
        if self._session is None:
            return None
        x, y = self._last
        return self._dab(ctx, self._event, x, y)


def _nib_coverage(size, aspect, angle_deg, antialias=True):
    """A rotated elliptical nib: the calligraphy pen's footprint."""
    d = max(1, int(math.ceil(size)) + 2)
    r = size / 2.0
    ys = np.arange(d, dtype=np.float64)[:, None] + 0.5 - d / 2.0
    xs = np.arange(d, dtype=np.float64)[None, :] + 0.5 - d / 2.0
    t = math.radians(angle_deg)
    u = xs * math.cos(t) + ys * math.sin(t)
    v = -xs * math.sin(t) + ys * math.cos(t)
    ry = max(0.5, r * max(0.05, float(aspect)))
    if not antialias:
        return (((u / r) ** 2 + (v / ry) ** 2) <= 1.0).astype(np.uint8) * 255
    acc = np.zeros((d, d))
    for oy in (-0.25, 0.25):
        for ox in (-0.25, 0.25):
            uu = (xs + ox) * math.cos(t) + (ys + oy) * math.sin(t)
            vv = -(xs + ox) * math.sin(t) + (ys + oy) * math.cos(t)
            acc += ((uu / r) ** 2 + (vv / ry) ** 2) <= 1.0
    return np.rint(acc / 4.0 * 255.0).astype(np.uint8)


class InkTool(_StrokeTool):
    """A calligraphy pen: an angled elliptical nib that thins as you speed up.

    GIMP's Ink tool. Speed is measured as distance between pointer events,
    which is what the pointer actually reports; a fast flick spaces events
    out, so the line narrows exactly where a real nib would run dry.
    """

    name = "ink"
    label = "Ink"
    default_size = 12
    options_spec = (O.size(12, 300),
                    O.Opt("angle", O.INT, 30, "Nib angle", -90, 90, suffix="\u00b0"),
                    O.Opt("aspect", O.FLOAT, 0.35, "Nib shape", 0.05, 1.0, 0.05, percent=True,
                          tip="100% is a round nib; lower is flatter."),
                    O.Opt("speed", O.FLOAT, 0.6, "Speed thinning", 0.0, 1.0, 0.05, percent=True))

    def on_begin(self, ctx, event):
        self._scale = 1.0
        return super().on_begin(ctx, event)

    def on_motion(self, ctx, event):
        if self._session is not None:
            x0, y0 = self._last
            step = math.hypot(event.x - x0, event.y - y0)
            size = float(self.option("size", 12))
            speed = min(1.0, step / max(3.0, size * 1.5))
            target = 1.0 - float(self.option("speed", 0.6)) * 0.85 * speed
            self._scale = self._scale * 0.6 + target * 0.4      # smoothed
        return super().on_motion(ctx, event)

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        size = max(1.0, float(self.option("size", 12)) * getattr(self, "_scale", 1.0)
                   * (event.pressure if event.pressure < 1.0 else 1.0))
        cov = _nib_coverage(size, self.option("aspect", 0.35), self.option("angle", 30))
        d = cov.shape[0]
        ox, oy = int(math.floor(x - d / 2.0)), int(math.floor(y - d / 2.0))
        rect = Rect(ox, oy, d, d).clipped_to(cell.width, cell.height)
        if rect is None:
            return None
        cov = cov[rect.y - oy:rect.y - oy + rect.h, rect.x - ox:rect.x - ox + rect.w]
        self._session.touch(rect)
        src = np.empty((rect.h, rect.w, 4), dtype=np.uint8)
        src[:] = ctx.colour_for(event)
        cell.apply_masked(rect, src, ctx.coverage(cov, rect))
        return rect


# ---- pattern stamp -------------------------------------------------------------------

PATTERNS = ("checker", "stripes", "diagonal", "dots", "bricks", "grid", "noise", "clipboard")


def pattern_tile(kind, size, fg, bg, clip=None):
    """One tile of a two-colour pattern (or the clipboard), as RGBA."""
    n = max(2, int(size))
    if kind == "clipboard" and clip is not None:
        return np.ascontiguousarray(clip.rgba)
    yy, xx = np.mgrid[0:n * 2, 0:n * 2]
    if kind == "stripes":
        on = (yy // n) % 2 == 0
    elif kind == "diagonal":
        on = ((xx + yy) // n) % 2 == 0
    elif kind == "dots":
        c = (xx % n - n / 2.0) ** 2 + (yy % n - n / 2.0) ** 2
        on = c <= (n * 0.3) ** 2
    elif kind == "bricks":
        row = yy // n
        on = ((xx + (row % 2) * n) % (2 * n) < max(1, n // 6)) | (yy % n < max(1, n // 6))
    elif kind == "grid":
        on = (xx % n < max(1, n // 8)) | (yy % n < max(1, n // 8))
    elif kind == "noise":
        on = np.random.default_rng(7).random((n * 2, n * 2)) < 0.5
    else:
        on = ((xx // n) + (yy // n)) % 2 == 0
    tile = np.empty(on.shape + (4,), dtype=np.uint8)
    tile[on] = fg
    tile[~on] = bg
    return tile


class PatternStampTool(_StrokeTool):
    """Paint with a repeating pattern, anchored to the canvas so strokes join."""

    name = "pattern_stamp"
    label = "Pattern Stamp"
    default_size = 32
    options_spec = (O.size(32), O.hardness(0.8),
                    O.Opt("pattern", O.CHOICE, "checker", "Pattern", choices=PATTERNS),
                    O.Opt("scale", O.INT, 8, "Pattern size", 2, 256, suffix=" px"))

    def on_begin(self, ctx, event):
        self._tile = pattern_tile(self.option("pattern", "checker"), self.option("scale", 8),
                                  ctx.primary, ctx.secondary, ctx.clip)
        return super().on_begin(ctx, event)

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        rect, cov, _ = _dab_window(cell, x, y, max(1.0, float(self.option("size", 32))),
                                   float(self.option("hardness", 0.8)))
        if rect is None:
            return None
        th, tw = self._tile.shape[:2]
        ys = (np.arange(rect.y, rect.y1) % th)[:, None]
        xs = (np.arange(rect.x, rect.x1) % tw)[None, :]
        patch = self._tile[ys, xs]
        self._session.touch(rect)
        cell.apply_masked(rect, patch, ctx.coverage(cov, rect))
        return rect


# ---- magic eraser ----------------------------------------------------------------------

class MagicEraserTool(Tool):
    """Click to erase every similar pixel -- the magic wand, deleting.

    Photoshop's Magic Eraser. Contiguous by default; Global takes every
    similar pixel in the layer. One click, one undo entry.
    """

    name = "magic_eraser"
    label = "Magic Eraser"
    options_spec = (O.tolerance(32),
                    O.Opt("global", O.BOOL, False, "Global",
                          tip="Erase every similar pixel, not only the connected area."))

    def on_begin(self, ctx, event):
        from ..stroke import StrokeSession
        cell = ctx.cell
        self.active = False
        if cell is None or not cell.bounds.contains(event.ix, event.iy):
            return None
        tolerance = int(self.option("tolerance", 32))
        px = cell.plane("rgba")
        if self.option("global", False) ^ event.has(1):
            ref = px[event.iy, event.ix]
            stencil = _colour_distance(px, ref) <= tolerance * 0.75
            stencil &= np.abs(px[..., 3].astype(int) - int(ref[3])) <= tolerance
        else:
            stencil = accel.flood_fill(px, event.ix, event.iy, tolerance)
        ys, xs = np.nonzero(stencil)
        if ys.size == 0:
            return None
        rect = Rect.from_points(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mask = (stencil[rect.slice()].astype(np.uint8) * 255)
        mask = ctx.coverage(mask, rect)
        session = StrokeSession(ctx.doc, ctx.layer, ctx.frame, label=self.label)
        session.touch(rect)
        cell.erase_masked(rect, mask)
        cmd = session.commit()
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return rect


class ArtHistoryBrushTool(_StrokeTool):
    """Photoshop's Art History Brush: repaint from the history source in
    short stylised strokes.

    Each step scatters several little strokes over an area around the
    pointer. Each takes its colour from the source at its own spot and lies
    ACROSS the source's local gradient -- along edges, not over them -- so
    the image comes back as if painted. Style picks the stroke length and
    how loose they are; Fidelity how far colours may wander.
    """

    name = "art_history"
    label = "Art History Brush"
    default_size = 4
    STYLES = {"tight_short": (4, 0.2), "tight_medium": (8, 0.25), "loose_medium": (10, 0.6),
              "loose_long": (18, 0.7), "dab": (2, 0.4)}
    options_spec = (O.size(4, label="Stroke width"),
                    O.Opt("style", O.CHOICE, "tight_medium", "Style",
                          choices=tuple(STYLES),
                          labels={"tight_short": "Tight short", "tight_medium": "Tight medium",
                                  "loose_medium": "Loose medium", "loose_long": "Loose long",
                                  "dab": "Dab"}),
                    O.Opt("area", O.INT, 30, "Area", 2, 300, 2, suffix=" px"),
                    O.Opt("fidelity", O.INT, 90, "Fidelity", 0, 100, 5, suffix=" %"))

    def on_begin(self, ctx, event):
        self._source = None
        if ctx.history_source is not None and ctx.layer is not None:
            self._source = ctx.history_source(ctx.layer.id, ctx.frame.id)
        if self._source is None:
            self.active = False
            return {"error": "no history source for this layer"}
        g = luminance(self._source[..., :3])
        self._gy, self._gx = np.gradient(g)
        self._rng = np.random.default_rng(int(event.x * 31 + event.y * 17) & 0xFFFF)
        return super().on_begin(ctx, event)

    def _dab(self, ctx, event, x, y):
        cell = ctx.cell
        src = self._source
        h, w = src.shape[:2]
        length, loose = self.STYLES.get(self.option("style", "tight_medium"), (8, 0.25))
        width = max(1.0, float(self.option("size", 4)))
        area = float(self.option("area", 30))
        fidelity = int(self.option("fidelity", 90)) / 100.0
        n = max(1, int(area * area / (length * width * 6)))
        covered = None
        rng = self._rng
        for _ in range(n):
            a = rng.random() * 2 * math.pi
            r = math.sqrt(rng.random()) * area / 2.0
            px, py = x + math.cos(a) * r, y + math.sin(a) * r
            ix, iy = int(px), int(py)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            colour = src[iy, ix].astype(np.float64)
            if fidelity < 1.0:
                colour[:3] += (rng.random(3) - 0.5) * 255 * (1 - fidelity)
            gx, gy = self._gx[iy, ix], self._gy[iy, ix]
            # Along the edge: perpendicular to the gradient, jittered.
            ang = math.atan2(gy, gx) + math.pi / 2 + (rng.random() - 0.5) * math.pi * loose
            ln = length * (0.6 + rng.random() * 0.8)
            rect = self._stroke(ctx, cell, px, py, ang, ln, width,
                                np.clip(colour, 0, 255).astype(np.uint8))
            if rect is not None:
                covered = rect if covered is None else covered.union(rect)
        return covered

    def _stroke(self, ctx, cell, x, y, ang, length, width, colour):
        dx, dy = math.cos(ang) * length / 2.0, math.sin(ang) * length / 2.0
        x0, y0, x1, y1 = x - dx, y - dy, x + dx, y + dy
        half = width / 2.0
        box = Rect.from_bounds(int(math.floor(min(x0, x1) - half - 1)), int(math.floor(min(y0, y1) - half - 1)),
                               int(math.ceil(max(x0, x1) + half + 1)), int(math.ceil(max(y0, y1) + half + 1)))
        rect = box.clipped_to(cell.width, cell.height)
        if rect is None:
            return None
        yy, xx = np.mgrid[rect.y:rect.y1, rect.x:rect.x1].astype(np.float64)
        xx += 0.5
        yy += 0.5
        vx, vy = x1 - x0, y1 - y0
        L2 = max(vx * vx + vy * vy, 1e-9)
        t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / L2, 0, 1)
        d = np.hypot(xx - (x0 + t * vx), yy - (y0 + t * vy))
        cov = np.clip(half + 0.5 - d, 0, 1)
        cov = np.rint(cov * 255).astype(np.uint8)
        if not cov.any():
            return None
        self._session.touch(rect)
        block = np.empty((rect.h, rect.w, 4), np.uint8)
        block[:] = colour
        cell.apply_masked(rect, block, ctx.coverage(cov, rect))
        return rect
