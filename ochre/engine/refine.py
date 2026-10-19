# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Refine Edge, Sky Replacement and Merge to HDR -- three classical
algorithms behind three Photoshop features that are often assumed to need
machine learning.

  refine_mask    Photoshop's Select and Mask / Refine Edge: a rough
                 selection snapped to the image's real edges with a GUIDED
                 FILTER (He, Sun and Tang, 2010) -- the mask is fitted, in
                 every small window, as a linear function of the image's
                 brightness, so it follows hair and leaves where the image
                 does. Then smooth / feather / contrast / shift edge.
  sky_mask       Photoshop's Sky Replacement's selection: a colour model
                 learned from the top of the frame, kept only where it is
                 connected to the top, refined with the guided filter.
  exposure_fusion  Merge to HDR as exposure fusion (Mertens, Kautz and Van
                 Reeth, 2007): bracketed shots blended by per-pixel weights
                 for contrast, saturation and good exposure, through
                 Laplacian pyramids so there are no seams. Displayable
                 directly -- no separate tone mapping.

What these are not: Photoshop's versions of Select Subject and Sky
Replacement use trained networks. These cannot recognise a sky that looks
like a wall; they find what the image's colours and edges say.
"""

import numpy as np


def _box(a, r):
    """Mean over a (2r+1) square, edges clamped. Float in, float out."""
    if r < 1:
        return a
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = out.shape[axis]
        out = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
               - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return out


def luminance(rgba):
    a = rgba.astype(np.float64)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0


def guided_filter(guide, src, radius, eps):
    """The guided filter: src (0..1) made to follow guide's edges."""
    I = guide.astype(np.float64)
    p = src.astype(np.float64)
    mI, mp = _box(I, radius), _box(p, radius)
    cov = _box(I * p, radius) - mI * mp
    var = _box(I * I, radius) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _box(a, radius) * I + _box(b, radius)


def _gauss(m, radius):
    out = m
    r = max(1, int(round(radius / 2.0)))
    for _ in range(3):
        out = _box(out, r)
    return out


def refine_mask(rgba, mask, radius=4, smooth=0, feather=0.0, contrast=0, shift=0):
    """A refined selection mask (uint8) from a rough one.

    radius   how far the edge may move to meet the image (guided filter)
    smooth   0..100: rounds off a jagged outline before refining
    feather  px: soft edge after refining
    contrast 0..100: sharpens the transition
    shift    -100..100: moves the edge out (+) or in (-)
    """
    m = np.asarray(mask, np.float64) / 255.0
    if smooth:
        m = _gauss(m, 1 + smooth / 10.0)
        m = (m >= 0.5).astype(np.float64) * 0.5 + m * 0.5
    if radius:
        m = guided_filter(luminance(rgba), m, int(radius), 1e-3)
    if shift:
        m = m + shift / 200.0
    if contrast:
        k = 1.0 + contrast / 10.0
        m = (m - 0.5) * k + 0.5
    m = np.clip(m, 0, 1)
    if feather:
        m = _gauss(m, feather)
    return np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)


# ---- sky ---------------------------------------------------------------------------

def sky_mask(rgba, band=0.08, tolerance=3.0, shift=0, radius=8, rounds=12):
    """Where the sky is: smooth, like the top of the frame, connected to it.

    Region growing in colour space: a model learned from the top band finds
    a first region, the model is re-learned from that region, and so on --
    so a sky that shades from deep blue overhead to pale at the horizon is
    followed down rather than cut off. Texture stops it at the ground, and
    keeping only what connects to the top keeps a blue car out.
    """
    from .morph import component_at
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float64)
    lum = luminance(rgba)
    texture = np.sqrt(np.maximum(_box(lum * lum, 2) - _box(lum, 2) ** 2, 0))
    smooth = texture < 0.035 * (1 + shift / 100.0)
    region = np.zeros((h, w), bool)
    region[:max(2, int(h * band))] = True
    region &= smooth | (np.arange(h)[:, None] < 2)
    limit = tolerance * (1 + shift / 100.0)
    keep = region
    for _ in range(rounds):
        sample = rgb[region].reshape(-1, 3)
        if len(sample) < 8:
            break
        mean = sample.mean(axis=0)
        cov = np.cov(sample.T) + np.eye(3) * 36.0
        d = rgb - mean
        md = np.sqrt(np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d))
        cand = (md < limit) & smooth
        cand_u8 = cand.astype(np.uint8) * 255
        keep = np.zeros_like(cand)
        for x in range(0, w, max(1, w // 32)):
            if cand[0, x] and not keep[0, x]:
                keep |= np.asarray(component_at(cand_u8, x, 0), bool)
        if keep.sum() <= region.sum():
            break
        region = keep
    m = guided_filter(lum, keep.astype(np.float64), radius, 1e-3)
    return np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)


def sky_preset(name, width, height, seed=0):
    """A replacement sky: 'blue', 'sunset', 'overcast' or 'clouds'."""
    t = (np.arange(height, dtype=np.float64) / max(1, height - 1))[:, None, None]
    stops = {"blue": ((40, 90, 190), (170, 205, 240)), "sunset": ((70, 40, 110), (250, 150, 70)),
             "overcast": ((150, 155, 165), (210, 212, 218)), "clouds": ((60, 120, 210), (175, 210, 245))}
    a, b = (np.array(c, np.float64) for c in stops.get(name, stops["blue"]))
    sky = a + (b - a) * t ** 0.8
    sky = np.broadcast_to(sky, (height, width, 3)).copy()
    if name == "clouds":
        from .filters import _value_noise
        n = _value_noise(height, width, max(width, height) / 3.0, seed, 6, 0.55)
        n = np.clip((n - 0.45) * 2.2, 0, 1)[..., None]
        sky = sky + (np.array((250, 250, 252)) - sky) * n
    out = np.empty((height, width, 4), np.uint8)
    out[..., :3] = np.clip(np.rint(sky), 0, 255)
    out[..., 3] = 255
    return out


# ---- exposure fusion -------------------------------------------------------------

_KERNEL = np.array([1, 4, 6, 4, 1], np.float64) / 16.0


def _blur(a):
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (2, 2)
        p = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        acc = 0
        for k in range(5):
            acc = acc + _KERNEL[k] * np.take(p, np.arange(k, k + n), axis=axis)
        out = acc
    return out


def _down(a):
    return _blur(a)[::2, ::2]


def _up(a, shape):
    out = np.zeros(shape[:2] + a.shape[2:], np.float64)
    out[::2, ::2] = a
    return _blur(out) * 4.0


def _gaussian_pyramid(a, levels):
    pyr = [a]
    for _ in range(levels - 1):
        pyr.append(_down(pyr[-1]))
    return pyr


def _laplacian_pyramid(a, levels):
    g = _gaussian_pyramid(a, levels)
    return [g[i] - _up(g[i + 1], g[i].shape) for i in range(levels - 1)] + [g[-1]]


def exposure_fusion(images, contrast=1.0, saturation=1.0, exposure=1.0):
    """Fuse bracketed exposures (uint8 RGBA, same size) into one image."""
    imgs = [np.asarray(im, np.float64)[..., :3] / 255.0 for im in images]
    h, w = imgs[0].shape[:2]
    weights = []
    for im in imgs:
        g = im.mean(axis=-1)
        lap = np.abs(4 * g - np.roll(g, 1, 0) - np.roll(g, -1, 0) - np.roll(g, 1, 1) - np.roll(g, -1, 1))
        sat = im.std(axis=-1)
        well = np.prod(np.exp(-((im - 0.5) ** 2) / (2 * 0.2 ** 2)), axis=-1)
        weights.append((lap ** contrast) * (sat ** saturation) * (well ** exposure) + 1e-12)
    total = np.sum(weights, axis=0)
    weights = [wt / total for wt in weights]
    levels = max(1, int(np.floor(np.log2(min(h, w)))) - 2)
    fused = None
    for im, wt in zip(imgs, weights):
        lp = _laplacian_pyramid(im, levels)
        gw = _gaussian_pyramid(wt[..., None], levels)
        part = [l * g for l, g in zip(lp, gw)]
        fused = part if fused is None else [f + p for f, p in zip(fused, part)]
    out = fused[-1]
    for lev in range(levels - 2, -1, -1):
        out = _up(out, fused[lev].shape) + fused[lev]
    res = np.empty((h, w, 4), np.uint8)
    res[..., :3] = np.clip(np.rint(out * 255), 0, 255)
    res[..., 3] = 255
    return res
