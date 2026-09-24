# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Effects and adjustments.

The shape is Paint.NET's, and three parts of it are load-bearing:

`prepare()` runs ONCE per render, before any tile. Expensive setup -- building
LUTs, resolving a kernel, reading token values into plain fields -- belongs
there, so `render_tile()` reads only immutable precomputed state. That is what
makes tiling safe to parallelise later.

Rendering takes a LIST of rectangles rather than one, so a scheduler can hand
out work without allocating sub-arrays, and so cancellation has a natural
granularity: one tile.

The source is marked read-only for the duration. In numpy that is one line,
and it turns "the effect scribbled on its own input" from silent corruption
into an immediate exception.

Adjustments are a special case of effect: per-pixel, LUT-driven, and therefore
exact and fast. Building a 256-entry table and doing `lut[arr]` is already
C-speed in numpy, which is why none of them are candidates for the accelerator.
"""

import numpy as np

from .geometry import Rect
from .props import (BoolProperty, ChoiceProperty, ColorProperty, FloatProperty,
                    IntProperty, PropertyCollection)


class Cancelled(Exception):
    """Raised out of a render when the caller asked it to stop."""


class CancelToken:
    """Cooperative cancellation, checked once per tile."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled

    def check(self):
        if self._cancelled:
            raise Cancelled()


class Effect:
    """Base class. Subclasses declare properties and render tiles."""

    name = "effect"
    label = "Effect"
    category = "Effects"
    # True when output depends only on the matching input pixel. Such effects
    # can be tiled with no halo and previewed on exactly the visible region.
    pointwise = False
    # Pixels of context a tile needs on each side. A blur reads its
    # neighbours, so tiling without a halo would show seams at tile edges.
    halo = 0

    def __init__(self):
        self.props = self.declare()
        self.configure_ui(self.props)

    # ---- declaration -----------------------------------------------------

    def declare(self):
        return PropertyCollection()

    def configure_ui(self, props):
        """Optional presentation tweaks. The model works without this."""

    # ---- rendering -------------------------------------------------------

    def prepare(self, token, source):
        """Expensive setup, once per render. Store results on self."""

    def render_tile(self, src, dst, rect):
        """Transform src[rect] into dst[rect]. src is read-only."""
        raise NotImplementedError

    # ---- driving ---------------------------------------------------------

    def apply(self, source, rects=None, token=None, cancel=None, progress=None,
              tile=256, out=None):
        """Render over the given rectangles. Returns the destination array.

        `source` is (H, W, 4) uint8. `rects` defaults to the whole surface.
        """
        source = np.ascontiguousarray(source, dtype=np.uint8)
        h, w = source.shape[:2]
        token = self.props.token() if token is None else dict(token)

        # An effect must not be able to modify its own input. One line.
        readonly = source.view()
        readonly.flags.writeable = False

        dst = np.array(source, copy=True) if out is None else out
        if out is not None:
            dst[...] = source

        self.prepare(token, readonly)

        whole = Rect(0, 0, w, h)
        regions = [whole] if not rects else [r for r in
                                             (rc.clipped_to(w, h) for rc in rects)
                                             if r is not None]
        tiles = []
        for region in regions:
            tiles.extend(_split(region, tile))

        for index, rect in enumerate(tiles):
            if cancel is not None:
                cancel.check()
            self.render_tile(readonly, dst, rect)
            if progress is not None:
                progress(index + 1, len(tiles), rect)
        return dst

    def preview(self, source, rect, token=None, cancel=None):
        """Render only a region, honouring the halo so tiles do not seam."""
        h, w = source.shape[:2]
        padded = rect.inflated(self.halo).clipped_to(w, h)
        return self.apply(source, [padded], token=token, cancel=cancel)


def _split(rect, tile):
    """Break a rectangle into tiles.

    Tile size matters more than it looks: numpy releases the GIL inside ufunc
    calls on arrays above a small threshold, so tiles genuinely parallelise --
    but per-tile Python overhead HOLDS the GIL. Tiles must be large enough
    that the C work dominates; 32x32 would be dispatch-bound and would scale
    negatively.
    """
    if tile <= 0 or (rect.w <= tile and rect.h <= tile):
        return [rect]
    out = []
    for y in range(rect.y, rect.y1, tile):
        for x in range(rect.x, rect.x1, tile):
            piece = Rect(x, y, min(tile, rect.x1 - x), min(tile, rect.y1 - y))
            if not piece.is_empty:
                out.append(piece)
    return out


# ---- adjustments ---------------------------------------------------------

class Adjustment(Effect):
    """A pointwise effect driven by 256-entry lookup tables.

    `lut[arr]` is already C-speed fancy indexing, so these are exact,
    allocation-light and not worth accelerating further.
    """

    category = "Adjustments"
    pointwise = True

    def build_luts(self, token):
        """Return (lut_r, lut_g, lut_b) or a single lut applied to all three."""
        raise NotImplementedError

    def prepare(self, token, source):
        luts = self.build_luts(token)
        if isinstance(luts, np.ndarray):
            luts = (luts, luts, luts)
        self._luts = tuple(np.asarray(l, dtype=np.uint8) for l in luts)

    def render_tile(self, src, dst, rect):
        window = src[rect.slice()]
        target = dst[rect.slice()]
        for c in range(3):
            target[..., c] = self._luts[c][window[..., c]]
        target[..., 3] = window[..., 3]          # adjustments never touch alpha


class BrightnessContrast(Adjustment):
    name = "brightness_contrast"
    label = "Brightness / Contrast"

    def declare(self):
        return PropertyCollection([
            IntProperty("brightness", 0, -100, 100),
            IntProperty("contrast", 0, -100, 100),
        ])

    def build_luts(self, token):
        brightness = token["brightness"]
        contrast = token["contrast"]
        x = np.arange(256, dtype=np.float64)
        # Contrast pivots around mid-grey; the multiplier is the same curve
        # Paint.NET uses, steepening toward vertical as contrast approaches
        # its maximum.
        if contrast >= 0:
            factor = 1.0 + contrast / 50.0
        else:
            factor = 1.0 + contrast / 100.0
        y = (x - 127.5) * factor + 127.5 + brightness * 2.55
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)


class Levels(Adjustment):
    name = "levels"
    label = "Levels"

    def declare(self):
        props = PropertyCollection([
            IntProperty("input_low", 0, 0, 255),
            IntProperty("input_high", 255, 0, 255),
            FloatProperty("gamma", 1.0, 0.1, 10.0, decimals=2),
            IntProperty("output_low", 0, 0, 255),
            IntProperty("output_high", 255, 0, 255),
        ])
        # Soft pairs: dragging one past the other carries it along.
        props.add_rule(__import__("ochre.engine.props", fromlist=["MinMaxPair"])
                       .MinMaxPair("input_low", "input_high"))
        props.add_rule(__import__("ochre.engine.props", fromlist=["MinMaxPair"])
                       .MinMaxPair("output_low", "output_high"))
        return props

    def build_luts(self, token):
        lo, hi = token["input_low"], token["input_high"]
        olo, ohi = token["output_low"], token["output_high"]
        gamma = max(0.1, token["gamma"])
        x = np.arange(256, dtype=np.float64)
        span = max(1, hi - lo)
        norm = np.clip((x - lo) / span, 0.0, 1.0)
        y = olo + (ohi - olo) * (norm ** (1.0 / gamma))
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)


class HueSaturation(Adjustment):
    name = "hue_saturation"
    label = "Hue / Saturation / Lightness"

    def declare(self):
        return PropertyCollection([
            IntProperty("hue", 0, -180, 180),
            IntProperty("saturation", 100, 0, 200),
            IntProperty("lightness", 0, -100, 100),
        ])

    def prepare(self, token, source):
        self._hue = token["hue"]
        self._sat = token["saturation"] / 100.0
        self._light = token["lightness"]

    def render_tile(self, src, dst, rect):
        window = src[rect.slice()].astype(np.float64)
        rgb = window[..., :3]

        if self._hue:
            rgb = _rotate_hue(rgb, self._hue)
        if self._sat != 1.0:
            # Saturation is a lerp toward intensity in RGB, not an HSV round
            # trip -- that is what Paint.NET does, and matching it matters if
            # someone compares output.
            intensity = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                         + 0.114 * rgb[..., 2])[..., None]
            rgb = intensity + (rgb - intensity) * self._sat
        if self._light:
            if self._light > 0:
                rgb = rgb + (255.0 - rgb) * (self._light / 100.0)
            else:
                rgb = rgb * (1.0 + self._light / 100.0)

        target = dst[rect.slice()]
        target[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        target[..., 3] = src[rect.slice()][..., 3]


def _rotate_hue(rgb, degrees):
    """Rotate hue with the YIQ matrix -- cheaper and smoother than HSV."""
    import math
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    one_third = 1.0 / 3.0
    sqrt_third = math.sqrt(one_third)
    a = cos_a + (1.0 - cos_a) * one_third
    b = one_third * (1.0 - cos_a) - sqrt_third * sin_a
    c = one_third * (1.0 - cos_a) + sqrt_third * sin_a
    r, g, bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([r * a + g * b + bl * c,
                     r * c + g * a + bl * b,
                     r * b + g * c + bl * a], axis=-1)


class Invert(Adjustment):
    name = "invert"
    label = "Invert Colors"

    def build_luts(self, token):
        return (255 - np.arange(256)).astype(np.uint8)


class Posterize(Adjustment):
    name = "posterize"
    label = "Posterize"

    def declare(self):
        props = PropertyCollection([
            IntProperty("red", 16, 2, 64),
            IntProperty("green", 16, 2, 64),
            IntProperty("blue", 16, 2, 64),
            BoolProperty("linked", True),
        ])
        from .props import LinkValues
        props.add_rule(LinkValues(["red", "green", "blue"], "linked"))
        return props

    def build_luts(self, token):
        def ramp(levels):
            x = np.arange(256, dtype=np.float64)
            step = 255.0 / (levels - 1)
            return np.clip(np.rint(np.rint(x / step) * step), 0, 255).astype(np.uint8)
        return (ramp(token["red"]), ramp(token["green"]), ramp(token["blue"]))


class Desaturate(Adjustment):
    name = "desaturate"
    label = "Black and White"

    def prepare(self, token, source):
        pass

    def render_tile(self, src, dst, rect):
        window = src[rect.slice()]
        # Rec.601 luma, the weights Paint.NET uses throughout.
        grey = (0.299 * window[..., 0] + 0.587 * window[..., 1]
                + 0.114 * window[..., 2])
        grey = np.clip(np.rint(grey), 0, 255).astype(np.uint8)
        target = dst[rect.slice()]
        for c in range(3):
            target[..., c] = grey
        target[..., 3] = window[..., 3]


class Sepia(Desaturate):
    name = "sepia"
    label = "Sepia"

    def render_tile(self, src, dst, rect):
        super().render_tile(src, dst, rect)
        target = dst[rect.slice()]
        grey = target[..., 0].astype(np.float64)
        target[..., 0] = np.clip(np.rint(grey * 1.07), 0, 255).astype(np.uint8)
        target[..., 2] = np.clip(np.rint(grey * 0.74), 0, 255).astype(np.uint8)


# ---- non-pointwise effects ----------------------------------------------

class GaussianBlur(Effect):
    """Separable tent blur, alpha-weighted.

    Two things here are the difference between a correct blur and the usual
    broken one.

    The kernel is a TENT, not a true Gaussian -- Paint.NET's weights are
    [16, 32, ... 16(r+1), ... 32, 16], applied separably, which is a tent
    squared and approximates a Gaussian closely enough that nobody has ever
    complained. It is also exactly what makes output match Paint.NET's.

    And the colour is weighted by ALPHA, with the result divided by the summed
    alpha rather than the summed weight. Blur straight RGB across differing
    alpha and every transparent edge grows a dark halo. This is the single
    most commonly botched operation in a hand-rolled editor.
    """

    name = "gaussian_blur"
    label = "Gaussian Blur"

    def declare(self):
        return PropertyCollection([IntProperty("radius", 4, 0, 200)])

    @property
    def halo(self):
        return int(self.props.get("radius", 0))

    def prepare(self, token, source):
        radius = max(0, int(token["radius"]))
        self._radius = radius
        if radius == 0:
            self._kernel = None
            return
        size = 1 + radius * 2
        weights = np.zeros(size, dtype=np.float64)
        for i in range(radius + 1):
            weights[i] = 16.0 * (i + 1)
            weights[size - i - 1] = weights[i]
        self._kernel = weights / weights.sum()

    def render_tile(self, src, dst, rect):
        if self._kernel is None:
            dst[rect.slice()] = src[rect.slice()]
            return
        r = self._radius
        h, w = src.shape[:2]
        padded = rect.inflated(r).clipped_to(w, h)
        window = src[padded.slice()].astype(np.float64)

        alpha = window[..., 3]
        # Premultiply, blur both, divide back out. This is the alpha weighting.
        weighted = window[..., :3] * alpha[..., None]

        blurred_rgb = _separable(weighted, self._kernel)
        blurred_a = _separable(alpha[..., None], self._kernel)[..., 0]

        safe = np.where(blurred_a > 1e-6, blurred_a, 1.0)
        out_rgb = np.where(blurred_a[..., None] > 1e-6,
                           blurred_rgb / safe[..., None], 0.0)

        oy, ox = rect.y - padded.y, rect.x - padded.x
        target = dst[rect.slice()]
        target[..., :3] = np.clip(
            np.rint(out_rgb[oy:oy + rect.h, ox:ox + rect.w]), 0, 255).astype(np.uint8)
        target[..., 3] = np.clip(
            np.rint(blurred_a[oy:oy + rect.h, ox:ox + rect.w]), 0, 255).astype(np.uint8)


def _separable(arr, kernel):
    """Convolve along both axes with a 1-D kernel, edge-extended."""
    radius = (len(kernel) - 1) // 2
    out = arr
    for axis in (0, 1):
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, weight in enumerate(kernel):
            if weight == 0.0:
                continue
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += padded[tuple(sl)] * weight
        out = acc
    return out


class Sharpen(Effect):
    name = "sharpen"
    label = "Sharpen"

    def declare(self):
        return PropertyCollection([IntProperty("amount", 2, 1, 20)])

    @property
    def halo(self):
        return 2

    def prepare(self, token, source):
        self._amount = token["amount"] / 10.0
        self._blur = GaussianBlur()
        self._blur.prepare({"radius": 2}, source)

    def render_tile(self, src, dst, rect):
        h, w = src.shape[:2]
        padded = rect.inflated(2).clipped_to(w, h)
        blurred = np.array(src[padded.slice()], copy=True)
        self._blur.render_tile(src, _Shifted(blurred, padded), padded)

        oy, ox = rect.y - padded.y, rect.x - padded.x
        original = src[rect.slice()].astype(np.float64)
        soft = blurred[oy:oy + rect.h, ox:ox + rect.w].astype(np.float64)
        sharp = original + (original - soft) * self._amount
        target = dst[rect.slice()]
        target[..., :3] = np.clip(np.rint(sharp[..., :3]), 0, 255).astype(np.uint8)
        target[..., 3] = src[rect.slice()][..., 3]


class _Shifted:
    """Lets a tile renderer write into an array indexed in document space."""

    def __init__(self, array, origin):
        self._array = array
        self._origin = origin

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value


class AddNoise(Effect):
    name = "add_noise"
    label = "Add Noise"

    def declare(self):
        return PropertyCollection([
            IntProperty("intensity", 32, 0, 255),
            IntProperty("seed", 0, 0, 65535),
            BoolProperty("monochrome", False),
        ])

    def prepare(self, token, source):
        self._intensity = token["intensity"]
        self._mono = token["monochrome"]
        self._seed = token["seed"]

    def render_tile(self, src, dst, rect):
        # Noise is a pure function of ABSOLUTE pixel coordinate, not of a
        # per-tile random sequence. Seeding a generator per tile is the
        # obvious approach and it is wrong: the draw order depends on the
        # tile's width, so the same pixel gets different noise depending on
        # how the region happened to be split. A coordinate hash makes the
        # result identical whether rendered in one pass or in 4x4 tiles,
        # which is what lets a preview and a final render agree.
        yy, xx = np.mgrid[rect.y:rect.y1, rect.x:rect.x1]
        window = src[rect.slice()].astype(np.int32)
        span = 2 * self._intensity + 1

        channels = 1 if self._mono else 3
        noise = np.empty((rect.h, rect.w, channels), dtype=np.int32)
        for c in range(channels):
            noise[..., c] = _hash_noise(xx, yy, self._seed, c, span) - self._intensity

        out = np.clip(window[..., :3] + noise, 0, 255).astype(np.uint8)
        target = dst[rect.slice()]
        target[..., :3] = out
        target[..., 3] = src[rect.slice()][..., 3]


def _hash_noise(xx, yy, seed, channel, span):
    """A stateless integer hash over pixel coordinates, in [0, span).

    An xorshift-multiply mix, kept in uint32 throughout so it behaves
    identically in numpy and in C should this ever move to the accelerator.
    """
    # The scalar term is folded in Python and masked to 32 bits. Doing it in
    # numpy scalars would warn on overflow -- and the wraparound is the whole
    # point of a hash, not a mistake worth warning about. Array-to-array
    # operations wrap silently, which is what the mixing steps below rely on.
    base = np.uint32((seed * 1274126177 + channel * 2246822519) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        h = (xx.astype(np.uint32) * np.uint32(374761393)
             + yy.astype(np.uint32) * np.uint32(668265263)
             + base)
        h ^= h >> np.uint32(13)
        h *= np.uint32(1274126177)
        h ^= h >> np.uint32(16)
        h *= np.uint32(2654435761)
        h ^= h >> np.uint32(15)
    return (h % np.uint32(span)).astype(np.int32)


# ---- registry ------------------------------------------------------------

BUILTIN = (BrightnessContrast, Levels, HueSaturation, Invert, Posterize,
           Desaturate, Sepia, GaussianBlur, Sharpen, AddNoise)


class EffectRegistry:
    """Known effects, grouped and ordered from INI."""

    def __init__(self, db=None, extra=()):
        self._classes = {cls.name: cls for cls in BUILTIN}
        for cls in extra:
            self._classes[cls.name] = cls
        self._labels = {}
        self._order = {}
        self._categories = {}
        self._defaults = {}
        if db is not None:
            self.load(db)

    def load(self, db):
        reserved = {"Class", "Label", "Order", "Category"}
        for name in db.sections("Effect"):
            section = "Effect:" + name
            target = db.get(section, "Class", name)
            if target not in self._classes:
                continue
            self._classes[name] = self._classes[target]
            self._labels[name] = db.get(section, "Label", name)
            self._order[name] = db.getint(section, "Order", 0)
            self._categories[name] = db.get(section, "Category", "")
            values = {}
            for key in db.keys(section):
                if key not in reserved:
                    values[key.lower()] = db.get(section, key)
            self._defaults[name] = values
        return self

    def create(self, name, **overrides):
        cls = self._classes.get(name)
        if cls is None:
            return None
        effect = cls()
        for key, raw in self._defaults.get(name, {}).items():
            if key in effect.props:
                effect.props.set(key, raw)
        for key, value in overrides.items():
            if key in effect.props:
                effect.props.set(key, value)
        return effect

    def label(self, name):
        return self._labels.get(name, getattr(self._classes.get(name), "label", name))

    def category(self, name):
        return (self._categories.get(name)
                or getattr(self._classes.get(name), "category", "Effects"))

    def names(self):
        return sorted(self._classes, key=lambda n: (self._order.get(n, 999), n))

    def grouped(self):
        out = {}
        for name in self.names():
            out.setdefault(self.category(name), []).append(name)
        return out

    def __contains__(self, name):
        return name in self._classes

    def __len__(self):
        return len(self._classes)
