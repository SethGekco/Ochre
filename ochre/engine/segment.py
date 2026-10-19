# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Separating an object from its background, without a neural network.

Photoshop's Object Selection is a trained model; GIMP's Foreground Select is
a matting algorithm. Neither is available to a numpy-only engine, and
pretending otherwise would be dishonest. What is here is the classical
method underneath both of their ancestors -- colour models, as in GrabCut's
first stage:

  * the user marks a region (a box or a loop): outside it is certainly
    background, and the middle of it is probably the object;
  * each side gets a small k-means colour model;
  * every pixel inside the region is labelled by which model explains it
    better, the labels are smoothed, and the connected piece containing the
    middle is kept;
  * repeat a few times, re-learning the models from the labels.

It works well on an object that differs in colour from its surroundings and
poorly on one that does not -- which is the honest boundary of the method,
and why the result is a selection the user can still edit.
"""

import numpy as np

from . import morph


def _kmeans(points, k, iters=8, seed=0):
    """Plain k-means on float colours. Returns centres (k, 3)."""
    n = len(points)
    if n == 0:
        return np.zeros((1, 3))
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(n, k, replace=False)].astype(np.float64)
    for _ in range(iters):
        d = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for j in range(k):
            sel = points[lab == j]
            if len(sel):
                centres[j] = sel.mean(axis=0)
    return centres


def _nearest_dist(points, centres):
    d = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d.min(axis=1))


def _sample(points, limit=4000, seed=0):
    if len(points) <= limit:
        return points
    idx = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[idx]


def segment(rgba, region, seed_point=None, sure_fg=None, iterations=3, k=5):
    """Foreground mask (uint8 0/255) inside `region` (bool, same shape).

    `region` is where the object may be; everything outside is background.
    `sure_fg`, if given, is a bool mask of pixels the user insisted are the
    object (foreground-select strokes). `seed_point` (x, y) chooses which
    connected piece to keep; default is the region's centre of mass.
    """
    rgb = rgba[..., :3].astype(np.float64)
    h, w = region.shape
    region = np.asarray(region, dtype=bool)
    if not region.any():
        return np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.nonzero(region)
    if seed_point is None:
        seed_point = (int(xs.mean()), int(ys.mean()))
    # Background evidence: a ring just outside the region.
    ring = (morph.grow(region.astype(np.uint8) * 255, 4) >= 128) & ~region
    if not ring.any():
        ring = ~region
    # Initial foreground guess: the inner part of the region.
    inner = morph.shrink(region.astype(np.uint8) * 255,
                         max(2, int(0.18 * min(np.ptp(xs) + 1, np.ptp(ys) + 1)))) >= 128
    fg = inner if inner.any() else region.copy()
    if sure_fg is not None:
        fg = fg | sure_fg

    inside = rgb[region]
    for it in range(iterations):
        fg_pts = _sample(rgb[fg & region], seed=it)
        # The first pass learns the background ONLY from outside the region.
        # The band between the initial guess and the region's edge usually
        # contains some of the object; feeding that to the background model
        # makes object pixels tie between the two models, and on flat colour
        # every tie went to the background -- an empty result. A noisy test
        # image hid it, because noise breaks ties.
        if it == 0 or not (region & ~fg).any():
            bg_pts = _sample(rgb[ring], seed=it + 7)
        else:
            bg_pts = _sample(np.concatenate([rgb[ring], rgb[region & ~fg]]), seed=it + 7)
        if len(fg_pts) == 0 or len(bg_pts) == 0:
            break
        cf = _kmeans(fg_pts, k, seed=it)
        cb = _kmeans(bg_pts, k, seed=it + 3)
        df = _nearest_dist(inside, cf)
        db = _nearest_dist(inside, cb)
        score = np.zeros((h, w))
        score[region] = db - df                       # >0: looks like the object
        # Smooth the decision so isolated pixels follow their neighbours.
        prob = (score > 0).astype(np.uint8) * 255
        prob = _box_smooth(prob, 2)
        fg = (prob >= 128) & region
        if sure_fg is not None:
            fg |= sure_fg & region
        sx, sy = seed_point
        if 0 <= sx < w and 0 <= sy < h and fg[sy, sx]:
            fg = morph.component_at(fg.astype(np.uint8) * 255, sx, sy)
        elif fg.any():
            fg = morph.largest_component(fg.astype(np.uint8) * 255)
    out = fg.astype(np.uint8) * 255
    return morph.remove_holes(out)


def _box_smooth(mask, r):
    m = mask.astype(np.float64)
    for axis in (0, 1):
        pad = [(r, r) if a == axis else (0, 0) for a in range(2)]
        p = np.pad(m, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = m.shape[axis]
        m = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
             - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return np.clip(np.rint(m), 0, 255).astype(np.uint8)


def _otsu(values):
    """Otsu's threshold for a 1-D array of non-negative floats."""
    v = np.asarray(values, np.float64)
    if v.size == 0 or v.max() <= v.min():
        return float(v.max()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=128)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    var = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return float(mids[np.argmax(var)])


def subject_mask(rgba, margin=0.04, refine_radius=None, iterations=4):
    """Select > Subject / Remove Background without a trained model.

    The frame's border is the background's evidence. A first guess is pure
    saliency -- how far each pixel's colour is from every colour on the
    border, split by Otsu's threshold -- then colour models for subject and
    background are re-learned from that guess a few times, the largest
    coherent object is kept with its holes filled, and the edge is snapped
    to the image with the guided filter. Works when the subject differs
    from its surroundings and does not fill the frame; uint8 0..255."""
    from .refine import refine_mask
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float64)
    m = max(2, int(round(min(h, w) * margin)))
    border = np.zeros((h, w), bool)
    border[:m] = border[-m:] = True
    border[:, :m] = border[:, -m:] = True
    interior = ~border
    bg_c = _kmeans(_sample(rgb[border], seed=1), 8, seed=1)
    d = np.zeros((h, w))
    d[interior] = _nearest_dist(rgb[interior], bg_c)
    # A photographer puts the subject in focus: colour distinctness counts
    # for more where the image is sharp (local Laplacian energy).
    from .selectrange import sharpness
    sh = sharpness(rgba, max(3, int(min(h, w) / 40)))
    top = np.percentile(sh[interior], 99) if interior.any() else 0
    if top > 1e-9:
        d = d * (0.35 + 0.65 * np.clip(sh / top, 0, 1))
    fg = interior & (d > _otsu(d[interior]))
    for it in range(iterations):
        if not fg.any():
            return np.zeros((h, w), np.uint8)
        cf = _kmeans(_sample(rgb[fg], seed=it), 6, seed=it)
        cb = _kmeans(_sample(np.concatenate([rgb[border], rgb[interior & ~fg]]), seed=it + 5), 8,
                     seed=it + 5)
        score = np.full((h, w), -1.0)
        score[interior] = _nearest_dist(rgb[interior], cb) - _nearest_dist(rgb[interior], cf)
        prob = _box_smooth((score > 0).astype(np.uint8) * 255, 2)
        fg = (prob >= 128) & interior
        if fg.any():
            fg = np.asarray(morph.largest_component(fg.astype(np.uint8) * 255), bool)
    hard = morph.remove_holes(fg.astype(np.uint8) * 255)
    if not hard.any():
        return hard
    r = refine_radius or max(2, int(min(h, w) / 120))
    return refine_mask(rgba, hard, radius=r, contrast=20)
