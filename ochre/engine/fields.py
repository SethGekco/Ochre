# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Float-field machinery for the Filter Gallery families (Sketch, Brush
Strokes, Texture, Artistic): seeded positional noise, an exact Gaussian,
streaks along a direction, Sobel, max filters, medians chosen by
brightness, and the four relief surfaces (Brick, Burlap, Canvas,
Sandstone) with the light that shades them.

The algorithms and their constants are ported from Kooka Pictura's
photorust engine (crates/pictura-filters/src/photorust, GPL-3.0, itself
from perfecto25/photorust). Kooka runs them pixel by pixel on rayon; here
every one is a whole-array numpy expression, so a "field" is a float32
(H, W) array and a picture is float (H, W, 3) in 0..255.

Two deliberate differences from the Rust, both at the image border only:
streak and median windows clamp to the edge instead of dropping the
samples that fall outside it. Everywhere else the arithmetic is the same.

Noise is the same 32-bit integer hash as Kooka's, so a seed gives the same
pattern every time, in the preview and on OK. Seed 0 is the base pattern;
any other seed re-rolls every primitive at once.
"""

import math

import numpy as np

M32 = 0xFFFFFFFF

LIGHTS = ("bottom", "bottom_left", "left", "top_left", "top", "top_right", "right",
          "bottom_right")
LIGHT_LABELS = {"bottom": "Bottom", "bottom_left": "Bottom Left", "left": "Left",
                "top_left": "Top Left", "top": "Top", "top_right": "Top Right",
                "right": "Right", "bottom_right": "Bottom Right"}
_D = math.sqrt(0.5)
_TOWARDS = {"bottom": (0.0, 1.0), "bottom_left": (-_D, _D), "left": (-1.0, 0.0),
            "top_left": (-_D, -_D), "top": (0.0, -1.0), "top_right": (_D, -_D),
            "right": (1.0, 0.0), "bottom_right": (_D, _D)}

DIRECTIONS = ("right_diagonal", "horizontal", "left_diagonal", "vertical")
DIRECTION_LABELS = {"right_diagonal": "Right Diagonal", "horizontal": "Horizontal",
                    "left_diagonal": "Left Diagonal", "vertical": "Vertical"}
_ANGLE = {"right_diagonal": 45.0, "horizontal": 0.0, "left_diagonal": 135.0,
          "vertical": 90.0}

TEXTURES = ("brick", "burlap", "canvas", "sandstone")
TEXTURE_LABELS = {"brick": "Brick", "burlap": "Burlap", "canvas": "Canvas",
                  "sandstone": "Sandstone"}


def towards(light):
    """Unit vector a light shines towards, image axes (y down)."""
    return _TOWARDS.get(light, _TOWARDS["bottom"])


def light_name(value):
    """A light given as a name or as Kooka's 0..7 slider index."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return LIGHTS[int(round(value)) % 8]
    return value if value in _TOWARDS else "bottom"


def along(direction_or_angle):
    """(dx, dy) of a stroke direction or an angle in degrees anticlockwise."""
    angle = _ANGLE.get(direction_or_angle, direction_or_angle)
    r = math.radians(float(angle))
    return math.cos(r), -math.sin(r)


# ---- noise -------------------------------------------------------------------------

def _u32(v):
    # At least 1-D: numpy warns on (intended) wrap-around in SCALAR integer
    # arithmetic but not in array arithmetic.
    return np.atleast_1d(np.asarray(v, np.int64)).astype(np.uint32)


def _hash2_scalar(x, y):
    h = ((x * 0x9E3779B1) & M32) ^ ((y * 0x85EBCA77) & M32)
    h ^= h >> 15
    h = (h * 0x2545F491) & M32
    h ^= h >> 13
    return h


def fold_seed(seed):
    """A 32-bit mix of a user seed: 0 keeps the base pattern."""
    seed = int(seed)
    if seed == 0:
        return 0
    return max(1, _hash2_scalar(seed & M32, ((seed >> 32) ^ 0x5EED) & M32))


def noise(a, b, seed=0):
    """Deterministic 0..1 noise from two whole numbers (arrays)."""
    s = np.uint32((fold_seed(seed) * 0x85EBCA77) & M32)
    h = (_u32(a) * np.uint32(0x27D4EB2D)) ^ (_u32(b) * np.uint32(0x165667B1)) \
        ^ np.uint32(0x9E3779B9) ^ s
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x2545F491)
    h ^= h >> np.uint32(13)
    return (h % np.uint32(1024)).astype(np.float32) / 1023.0


def speck(x, y, salt, seed=0):
    """Two noises summed and centred: -1..1, a little bell-shaped."""
    s = salt * 977
    return noise(x + s, y - s, seed) + noise(y * 3 + s, x * 5 + s, seed) - 1.0


def lattice(x, y, salt, seed=0):
    s = np.uint32((fold_seed(seed) * 0x27D4EB2D) & M32)
    h = (_u32(x) * np.uint32(0x9E3779B1)) ^ (_u32(y) * np.uint32(0x85EBCA77)) \
        ^ np.uint32((salt * 0xC2B2AE3D) & M32) ^ s
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x2545F491)
    h ^= h >> np.uint32(13)
    return (h & np.uint32(0xFFFF)).astype(np.float32) / 65535.0


def value_noise(u, v, cell, salt, seed=0):
    """Smooth value noise on a grid of `cell` pixels, 0..1."""
    gu, gv = np.asarray(u, np.float32) / cell, np.asarray(v, np.float32) / cell
    iu, iv = np.floor(gu), np.floor(gv)
    fu, fv = gu - iu, gv - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    iu, iv = iu.astype(np.int64), iv.astype(np.int64)
    top = lattice(iu, iv, salt, seed) * (1 - fu) + lattice(iu + 1, iv, salt, seed) * fu
    bottom = lattice(iu, iv + 1, salt, seed) * (1 - fu) + lattice(iu + 1, iv + 1, salt, seed) * fu
    return top * (1 - fv) + bottom * fv


def grid(h, w):
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.int64), ys.astype(np.int64)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---- fields ------------------------------------------------------------------------

def luma(rgb):
    """Rec. 601 brightness of a float or uint8 (..., 3+) array, 0..255."""
    return (0.299 * rgb[..., 0].astype(np.float32) + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]).astype(np.float32)


def blur(field, sigma):
    """An exact Gaussian (taps to 3 sigma), edges clamped; any trailing axes.
    Kooka's blur_field: the stroke filters' constants are tuned to it, so a
    box approximation would shift every look."""
    sigma = float(sigma)
    if sigma <= 0:
        return np.asarray(field, np.float32)
    taps = int(math.ceil(sigma * 3.0))
    k = np.exp(-(np.arange(-taps, taps + 1, dtype=np.float64) ** 2) / (2 * sigma * sigma))
    k = (k / k.sum()).astype(np.float32)
    out = np.asarray(field, np.float32)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (taps, taps)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros(out.shape, np.float32)
        for j, wt in enumerate(k):
            acc += wt * (p[j:j + n] if axis == 0 else p[:, j:j + n])
        out = acc
    return out


def blur_rgb(src, sigma):
    """Gaussian of an RGBA uint8 picture, premultiplied (no dark fringes);
    returns float (H, W, 3)."""
    a = src[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([src[..., :3].astype(np.float32) * a, a], axis=-1)
    pm = blur(pm, sigma)
    alpha = pm[..., 3:4]
    rgb = np.where(alpha > 1e-4, pm[..., :3] / np.maximum(alpha, 1e-4),
                   src[..., :3].astype(np.float32))
    return np.clip(rgb, 0, 255)


def unit_spread(field):
    """Scaled to a root-mean-square of one."""
    field = np.asarray(field, np.float32)
    spread = float(np.sqrt(np.mean(field.astype(np.float64) ** 2))) if field.size else 0.0
    return field / spread if spread > 0 else field


def _shift_clamped(a, dx, dy):
    """a[y + dy, x + dx], indices clamped to the canvas."""
    h, w = a.shape[:2]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[ys][:, xs]


def _offsets(direction, half):
    dx, dy = direction
    return [(int(math.floor(dx * s + 0.5)) if dx * s >= 0 else -int(math.floor(-dx * s + 0.5)),
             int(math.floor(dy * s + 0.5)) if dy * s >= 0 else -int(math.floor(-dy * s + 0.5)))
            for s in range(-half, half + 1)]


def line_mean(field, direction, half):
    """Mean of a field along a direction, `half` steps either side; samples
    beyond the canvas are left out (Kooka's streak averaging)."""
    h, w = field.shape[:2]
    offs = _offsets(direction, half)
    m = max([1] + [max(abs(a), abs(b)) for a, b in offs])
    p = np.pad(field, ((m, m), (m, m)), mode="constant")
    ones = np.pad(np.ones((h, w), np.float32), m, mode="constant")
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    for ox, oy in offs:
        total += p[m + oy:m + oy + h, m + ox:m + ox + w]
        count += ones[m + oy:m + oy + h, m + ox:m + ox + w]
    return total / np.maximum(count, 1.0)


def streaked_noise(h, w, length, salt, direction, seed=0):
    """White noise averaged along a direction over `length` pixels, scaled
    to a spread of one: streaks about as long as a stroke."""
    xs, ys = grid(h, w)
    n = speck(xs, ys, salt, seed)
    steps = max(1, int(round(length)))
    half = int(steps / 2)
    return unit_spread(line_mean(n, direction, half))


def streak_field(field, direction, length):
    steps = max(int(round(length)), 1) // 2
    return line_mean(field, direction, steps)


def sobel(field):
    """Sobel magnitude, edges clamped."""
    p = np.pad(field, 1, mode="edge")
    h, w = field.shape
    at = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = at(-1, 1) + 2 * at(0, 1) + at(1, 1) - at(-1, -1) - 2 * at(0, -1) - at(1, -1)
    gy = at(1, -1) + 2 * at(1, 0) + at(1, 1) - at(-1, -1) - 2 * at(-1, 0) - at(-1, 1)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def widest_nearby(field, radius):
    """Each value replaced by the largest within `radius`, across then down."""
    radius = int(radius)
    if radius <= 0:
        return field
    out = field
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        p = np.pad(out, pad, mode="edge")
        acc = out.copy()
        for j in range(2 * radius + 1):
            acc = np.maximum(acc, p[j:j + n] if axis == 0 else p[:, j:j + n])
        out = acc
    return out


def median_by_luma(rgb, offsets):
    """Each pixel takes the colour of the sample, among `offsets`, whose
    brightness is the median: a median that never invents a colour. Edges
    clamp. Worked in bands of rows so the sample stack stays small."""
    h, w = rgb.shape[:2]
    r = rgb.astype(np.int32)
    lum = 299 * r[..., 0] + 587 * r[..., 1] + 114 * r[..., 2]
    k = len(offsets)
    mid = k // 2
    out = np.empty_like(rgb)
    m = max([0] + [max(abs(a), abs(b)) for a, b in offsets])
    pl = np.pad(lum, m, mode="edge")
    prgb = np.pad(rgb, ((m, m), (m, m), (0, 0)), mode="edge")
    rows = max(1, int(3_000_000 // max(1, w * k)))
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        n = y1 - y0
        stack = np.stack([pl[m + y0 + oy:m + y0 + oy + n, m + ox:m + ox + w]
                          for ox, oy in offsets])
        pick = np.argpartition(stack, mid, axis=0)[mid]
        oxs = np.array([o[0] for o in offsets])[pick]
        oys = np.array([o[1] for o in offsets])[pick]
        yy = np.arange(y0, y1)[:, None] + oys + m
        xx = np.arange(w)[None, :] + oxs + m
        out[y0:y1] = prgb[yy, xx]
    return out


def median_along(rgb, direction, length):
    """median_by_luma over a line of samples along a direction."""
    steps = max(int(round(length)), 1) // 2
    return median_by_luma(rgb, _offsets(direction, steps))


def median_square(rgb, reach):
    """median_by_luma over a (2 reach + 1) square."""
    reach = int(reach)
    if reach <= 0:
        return rgb
    return median_by_luma(rgb, [(dx, dy) for dy in range(-reach, reach + 1)
                                for dx in range(-reach, reach + 1)])


def displace_nearest(rgb, xs, ys):
    """Sample a picture at rounded coordinates, clamped to the canvas."""
    h, w = rgb.shape[:2]
    sx = np.clip(np.floor(xs + 0.5), 0, w - 1).astype(np.intp)
    sy = np.clip(np.floor(ys + 0.5), 0, h - 1).astype(np.intp)
    return rgb[sy, sx]


def gradient(field):
    """Central differences (one-sided at the border), as Kooka's relief takes them."""
    p = np.pad(field, 1, mode="edge")
    h, w = field.shape
    dx = p[1:-1, 2:] - p[1:-1, :-2]
    dy = p[2:, 1:-1] - p[:-2, 1:-1]
    xd = np.full(w, 2.0, np.float32)
    yd = np.full(h, 2.0, np.float32)
    if w > 1:
        xd[0] = xd[-1] = 1.0
    if h > 1:
        yd[0] = yd[-1] = 1.0
    return dx / xd[None, :], dy / yd[:, None]


# ---- relief surfaces ---------------------------------------------------------------

def _brick(u, v, seed):
    course, length, mortar, joint_depth, grit_k = 9.0, 40.0, 1.4, 0.35, 0.9
    row = np.floor(v / course)
    offset = np.where(np.mod(row, 2.0) > 0.5, length / 2, 0.0)
    fy = v - row * course
    fx = np.mod(u + offset, length)
    bed = np.clip((np.minimum(fy, course - fy) - mortar * 0.5) / mortar, 0, 1)
    head = np.clip((np.minimum(fx, length - fx) - mortar * 0.5) / mortar, 0, 1)
    joints = bed * (1 - joint_depth * (1 - head))
    grit = 0.6 * value_noise(u * 2, v, 3.0, 3, seed) + 0.4 * value_noise(u * 2, v, 1.5, 5, seed)
    return joints * (1 - grit_k + grit_k * grit)


def _burlap(u, v, seed):
    pitch, run, kink, weave_k, drift, step, groove, crossing = \
        7.2, 4.0, 1.4, 0.35, 12.0, 0.45, 1.3, 0.15
    below = np.floor(v / pitch).astype(np.int64)
    hgt = np.ones(u.shape, np.float32)
    for row in (below, below + 1):
        wander = (value_noise(u, row * pitch, drift, 26, seed) - 0.5) * 2
        t = u / run + 0.5 * np.mod(row, 2) + wander
        k = np.floor(t).astype(np.int64)
        blend = smooth((t - k - (1 - step)) / step)

        def at(kk):
            weave = np.where(np.mod(kk + row, 2) == 0, 1.0, -1.0)
            shift = (weave_k * weave + (lattice(kk, row, 23, seed) - 0.5) * 2) * kink
            depth = np.clip(lattice(kk, row, 24, seed) * 1.4 - 0.1, 0.25, 1.0)
            return shift, depth
        (s0, d0), (s1, d1) = at(k), at(k + 1)
        shift = s0 + (s1 - s0) * blend
        depth = d0 + (d1 - d0) * blend
        d = (v - (row * pitch + shift)) / groove
        hgt = hgt - depth * np.exp(-d * d)
        across = (t - (k + 1)) * run
        under = (v - row * pitch) / pitch
        inside = (under >= 0) & (under < 1)
        hgt = hgt - np.where(inside, crossing * np.exp(-(across * across) / 0.8)
                             * np.sin(math.pi * under), 0.0)
    fibre = value_noise(u * 0.7, v * 1.4, 1.2, 7, seed)
    return np.maximum(hgt * (0.75 + 0.25 * fibre), 0.0)


def _canvas(u, v, seed):
    pitch, wander = 3.0, 0.5
    u2 = u + (value_noise(u, v, 6.0, 41, seed) - 0.5) * 2 * wander
    v2 = v + (value_noise(u2, v, 6.0, 42, seed) - 0.5) * 2 * wander
    u, v = u2, v2
    down = np.abs(np.sin(u / pitch * math.pi))
    across = np.abs(np.sin(v / pitch * math.pi))
    over = np.mod(np.floor(u / pitch) + np.floor(v / pitch), 2.0)
    weave = np.where(over > 0.5, 0.6 * down + 0.3 * across, 0.4 * down + 0.5 * across)
    slub = 0.75 + 0.5 * value_noise(u, v, 3.0, 43, seed)
    return (weave * slub + 0.2 * value_noise(u, v, 2.0, 4, seed)) * 0.85


def _sandstone(u, v, seed):
    return (0.5 * value_noise(u, v, 2.0, 1, seed) + 0.3 * value_noise(u, v, 5.0, 2, seed)
            + 0.2 * value_noise(u, v, 12.0, 3, seed))


_SURFACES = {"brick": _brick, "burlap": _burlap, "canvas": _canvas, "sandstone": _sandstone}


def height_map(texture, h, w, scaling, seed=0):
    """A surface's height, 0..1, at `scaling` percent (50..200)."""
    zoom = 100.0 / min(200, max(50, scaling))
    xs, ys = grid(h, w)
    fn = _SURFACES.get(texture, _canvas)
    return np.asarray(fn(xs.astype(np.float32) * zoom, ys.astype(np.float32) * zoom, seed),
                      np.float32)


SHADE_BY_COLOUR = 0.8
SHADE_BY_LIGHT = 150.0


def patchiness(h, w, patchy, seed=0):
    if patchy <= 0:
        return 1.0
    xs, ys = grid(h, w)
    xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    n = 0.6 * value_noise(xs, ys, 36.0, 51, seed) + 0.4 * value_noise(xs, ys, 36.0 * 0.45, 52, seed)
    t = smooth((n - 0.3) / 0.25)
    return 1 - patchy * (1 - t)


def relief(rgb, texture, scaling, amount, light, invert=False, seed=0, dark_bias=0.0,
           bevel=0.6, crisp=1.0, occlusion=0.0, patchy=0.0, gain=1.0):
    """Shade a float picture (H, W, 3) by a lit surface: Texturizer, and the
    finish several Sketch and Artistic filters lay over their result.
    `amount` is Relief, 0..50."""
    amount = min(50, max(0, amount)) / 50.0
    if amount <= 0:
        return rgb
    h, w = rgb.shape[:2]
    field = height_map(texture, h, w, scaling, seed)
    if invert:
        field = 1 - field
    field = blur(field, bevel)
    lx, ly = towards(light_name(light))
    dx, dy = gradient(field)
    lightness = luma(rgb) / 255.0
    weight = 1 - dark_bias * lightness
    slope = -(dx * lx + dy * ly)
    slope = np.sign(slope) * np.abs(slope) ** crisp
    patch = patchiness(h, w, patchy, seed)
    shade = slope * amount * weight * patch * gain
    groove = np.maximum(0.5 - field, 0) * 2 * occlusion * amount * weight * patch
    out = rgb * (1 + shade[..., None] * SHADE_BY_COLOUR) + (shade * SHADE_BY_LIGHT - groove)[..., None]
    return np.clip(out, 0, 255)


def put_rgb(src, rgb):
    """The source with its colour replaced and its alpha kept."""
    out = src.copy()
    out[..., :3] = np.clip(np.floor(np.asarray(rgb, np.float32) + 0.5), 0, 255).astype(np.uint8)
    return out


def swatches(ink, paper, amount):
    """Mix two colours by a 0..1 field: 0 is `paper`, 1 is `ink`."""
    ink = np.asarray(ink[:3], np.float32)
    paper = np.asarray(paper[:3], np.float32)
    return paper + (ink - paper) * np.asarray(amount, np.float32)[..., None]


def rnd(x):
    """Rust's round(): halves away from zero (Python's round() goes to even)."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def fill_holes(field):
    """Fill whatever a mask (> 0.5 inside) encloses: anything outside it that
    cannot be reached from the edge of the frame becomes inside."""
    from .accel import flood_fill
    outside = field <= 0.5
    padded = np.pad(outside, 1, constant_values=True).astype(np.uint8) * 255
    reach = flood_fill(padded, 0, 0, 0)[1:-1, 1:-1]
    return np.where(reach, field, np.maximum(field, 1.0)).astype(np.float32)


def close_gaps(field, reach):
    """Stop up the gaps in a mask narrower than `reach`, keeping its outer edge."""
    spread = blur(field, reach * 0.5)
    return np.maximum(field, (spread > 0.5).astype(np.float32))
