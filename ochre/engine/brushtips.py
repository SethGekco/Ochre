# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Sampled brush tips: Photoshop's brush tip shapes and Edit > Define Brush
Preset.

A tip is a square-ish uint8 coverage map (255 = full paint). The built-ins
are generated, so nothing is shipped as an image; a user tip is made from
the selection -- dark pixels paint, as in Photoshop -- and kept in the state
directory. The brush scales a tip to its size and may turn it per dab.
"""


import os

import numpy as np

BUILTIN = ("round", "square", "chalk", "spatter", "star", "leaf", "grass")
CUSTOM = "custom"
_custom = {"tip": None}
_cache = {}


def _grid(n=64):
    c = (np.arange(n) + 0.5 - n / 2.0) / (n / 2.0)
    return np.meshgrid(c, c)


def _make(name, n=64):
    x, y = _grid(n)
    r = np.hypot(x, y)
    # A stable seed (str hash() changes per process): the same tip every run.
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(name)))
    if name == "square":
        m = np.clip((1 - np.maximum(np.abs(x), np.abs(y))) * n / 2.0, 0, 1)
    elif name == "chalk":
        m = np.clip((1 - r) * 6, 0, 1) * (rng.random((n, n)) < 0.55 + 0.4 * (1 - r))
    elif name == "spatter":
        m = np.zeros((n, n))
        for _ in range(28):
            cx, cy = rng.normal(0, 0.45, 2)
            rad = rng.uniform(0.04, 0.16)
            m = np.maximum(m, np.clip((rad - np.hypot(x - cx, y - cy)) * n / 2.0, 0, 1))
    elif name == "star":
        ang = np.arctan2(y, x)
        edge = 0.55 + 0.4 * np.cos(5 * ang)
        m = np.clip((edge - r) * n / 2.0, 0, 1)
    elif name == "leaf":
        m = np.clip((1 - np.hypot(x / 0.45, y)) * n / 4.0, 0, 1)
    elif name == "grass":
        m = np.zeros((n, n))
        for _ in range(9):
            cx = rng.uniform(-0.6, 0.6)
            lean = rng.uniform(-0.3, 0.3)
            blade = np.abs(x - cx - lean * (1 - y) / 2) < 0.06 * (1 - (y + 1) / 2) + 0.01
            m = np.maximum(m, blade * (y > rng.uniform(-1, 0.2)))
    else:                                                    # round
        m = np.clip((1 - r) * n / 2.0, 0, 1)
    return np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)


def set_custom(tip):
    _custom["tip"] = None if tip is None else np.ascontiguousarray(tip, np.uint8)
    for k in [k for k in _cache if k[0] == CUSTOM]:
        del _cache[k]


def tip_from_image(rgba):
    """A tip from pixels: dark and opaque paints (Photoshop's rule), cropped
    to what paints."""
    a = np.asarray(rgba)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    cov = (255.0 - lum) * (a[..., 3] / 255.0)
    cov = np.clip(np.rint(cov), 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(cov > 8)
    if not ys.size:
        return None
    return np.ascontiguousarray(cov[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def path_for(state_dir):
    return os.path.join(state_dir, "brush_tip.png") if state_dir else None


def save_custom(state_dir, tip):
    from PIL import Image
    p = path_for(state_dir)
    if p:
        Image.fromarray(tip, "L").save(p)
    set_custom(tip)


def load_custom(state_dir):
    from PIL import Image
    p = path_for(state_dir)
    if p and os.path.exists(p):
        set_custom(np.asarray(Image.open(p).convert("L")))
        return True
    return False


def coverage(name, size, angle=0.0):
    """The tip scaled to `size` (its longer side) and turned by `angle`
    degrees, as uint8 coverage -- or None for the plain round brush, which
    keeps its own analytic edge and hardness."""
    if name in (None, "", "round"):
        return None
    base = _custom["tip"] if name == CUSTOM else None
    if name == CUSTOM and base is None:
        return None
    size = max(1, int(round(size)))
    key = (name, size, int(round(angle)) % 360)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    from PIL import Image
    if base is None:
        base = _make(name)
    h, w = base.shape
    s = size / float(max(h, w))
    im = Image.fromarray(base, "L").resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                           Image.BILINEAR)
    if key[2]:
        im = im.rotate(-key[2], resample=Image.BILINEAR, expand=True)
    out = np.asarray(im).copy()
    if len(_cache) > 256:
        _cache.clear()
    _cache[key] = out
    return out



