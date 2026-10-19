# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Select > Color Range and Select > Focus Area.

  color_range  Photoshop's Color Range: a soft selection of sampled colours
               (within a fuzziness), of a hue family (reds ... magentas), of
               a tonal range (highlights / midtones / shadows) or of skin
               tones. Soft: a colour just outside the fuzziness is partly
               selected, as in Photoshop.
  focus_area   Photoshop's Focus Area: what is in focus, found by local
               sharpness (the energy of the Laplacian over a neighbourhood)
               against the image's own range, then snapped to edges with the
               guided filter. A classical stand-in for Photoshop's version.
"""

import numpy as np

MODES = ("sampled", "reds", "yellows", "greens", "cyans", "blues", "magentas",
         "highlights", "midtones", "shadows", "skin")
_HUES = {"reds": 0.0, "yellows": 60.0, "greens": 120.0, "cyans": 180.0, "blues": 240.0,
         "magentas": 300.0}


def _hsv(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = 60.0 * (((g - b)[rm] / d[rm]) % 6)
    h[gm] = 60.0 * ((b - r)[gm] / d[gm] + 2)
    h[bm] = 60.0 * ((r - g)[bm] / d[bm] + 4)
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


def color_range(rgba, mode="sampled", samples=((0, 0, 0),), fuzziness=40):
    """Soft selection mask (uint8) of the image's colours in a range."""
    rgb = rgba[..., :3].astype(np.float64)
    if mode == "sampled":
        # Distance to the nearest sample; fully selected within fuzziness/2,
        # fading out to fuzziness.
        best = None
        for s in samples:
            d = np.sqrt(((rgb - np.asarray(s[:3], np.float64)) ** 2).sum(-1))
            best = d if best is None else np.minimum(best, d)
        f = max(1.0, float(fuzziness)) * 1.0
        m = np.clip((f - best) / (f * 0.5), 0, 1)
    elif mode in _HUES:
        h, s, v = _hsv(rgb / 255.0)
        dh = np.abs((h - _HUES[mode] + 180.0) % 360.0 - 180.0)
        m = np.clip((45.0 - dh) / 15.0, 0, 1) * np.clip(s * 4.0, 0, 1) * np.clip(v * 4.0, 0, 1)
    elif mode in ("highlights", "midtones", "shadows"):
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        if mode == "highlights":
            m = np.clip((lum - 0.6) / 0.15, 0, 1)
        elif mode == "shadows":
            m = np.clip((0.4 - lum) / 0.15, 0, 1)
        else:
            m = np.clip(1 - np.abs(lum - 0.5) / 0.25, 0, 1)
    elif mode == "skin":
        # Skin tones in YCbCr: a well-known compact cluster (Chai and Ngan).
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        dcb = np.clip(1 - np.maximum(0, np.maximum(77 - cb, cb - 127)) / 8.0, 0, 1)
        dcr = np.clip(1 - np.maximum(0, np.maximum(133 - cr, cr - 173)) / 8.0, 0, 1)
        m = dcb * dcr
    else:
        raise ValueError("unknown colour range %r" % (mode,))
    m = m * (rgba[..., 3] / 255.0)
    return np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)


def sharpness(rgba, radius=None):
    """Local sharpness: Laplacian energy averaged over a neighbourhood."""
    from .refine import _box, luminance
    lum = luminance(rgba)
    lap = np.abs(4 * lum - np.roll(lum, 1, 0) - np.roll(lum, -1, 0)
                 - np.roll(lum, 1, 1) - np.roll(lum, -1, 1))
    lap[0, :] = lap[-1, :] = 0
    lap[:, 0] = lap[:, -1] = 0
    h, w = lum.shape
    r = radius or max(3, int(max(h, w) / 60))
    return _box(lap, r)


def focus_area(rgba, in_focus=50, noise=0):
    """Soft mask (uint8) of the in-focus parts. in_focus (0..100) sets how
    sharp a region must be, relative to the sharpest in the image; noise
    (0..100) ignores fine texture that is only noise."""
    from .refine import _box, guided_filter, luminance
    s = sharpness(rgba)
    if noise:
        s = np.maximum(s - np.percentile(s, 30) * noise / 50.0, 0)
    top = np.percentile(s, 99.5)
    if top <= 1e-9:
        return np.zeros(rgba.shape[:2], np.uint8)
    rel = s / top
    level = 0.5 * (1 - in_focus / 100.0) + 0.02
    m = np.clip((rel - level * 0.6) / max(level * 0.8, 1e-3), 0, 1)
    h, w = m.shape
    m = _box(m, max(2, int(max(h, w) / 80)))
    m = guided_filter(luminance(rgba), m, max(4, int(max(h, w) / 50)), 1e-3)
    return np.clip(np.rint(np.clip(m, 0, 1) * 255), 0, 255).astype(np.uint8)
