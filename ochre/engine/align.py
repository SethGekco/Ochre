# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Auto-Align, Auto-Blend and Photomerge: lining images up and blending
them, all classical.

  features    Harris corners (Harris and Stephens, 1988), spread over the
              frame by a grid, each described by a normalised patch of the
              blurred image around it.
  match       nearest neighbours with Lowe's ratio test and a mutual check.
  ransac      a homography (or similarity) fitted to the matches by RANSAC,
              refined on the inliers.
  register    one image onto another: the transform taking its pixels into
              the reference frame, with the inlier count as confidence.
  focus_stack Auto-Blend's "stack images": per pixel, the sharpest source,
              chosen softly and blended through Laplacian pyramids so there
              are no seams (the same machinery as exposure fusion).
  panorama_blend  Auto-Blend's "panorama": each image weighted by distance
              from its own edge, then pyramid-blended.

Plain numpy. Rotation-invariant descriptors are not needed for what this
is for (bracketed, stacked or panning shots), and are not attempted.
"""

import math

import numpy as np

from . import warp


def _grey(rgba):
    a = rgba.astype(np.float64)
    g = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0
    if rgba.shape[-1] == 4:
        g = g * (a[..., 3] / 255.0)
    return g


def _blur(a, sigma):
    if sigma < 0.3:
        return a
    r = max(1, int(round(math.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = a
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
            n = out.shape[axis]
            out = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
                   - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return out


def _max_filter(a, r):
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="constant", constant_values=-np.inf)
        n = out.shape[axis]
        acc = np.take(p, np.arange(0, n), axis=axis)
        for k in range(1, 2 * r + 1):
            acc = np.maximum(acc, np.take(p, np.arange(k, k + n), axis=axis))
        out = acc
    return out


PATCH = 8          # descriptor grid (PATCH x PATCH samples)
SPACING = 2.0      # sample spacing in pixels (so a 16 px window)


def features(rgba, count=600, alpha=None):
    """(points (n, 2) as x, y; descriptors (n, PATCH*PATCH)) for an image."""
    g = _grey(rgba)
    h, w = g.shape
    gy, gx = np.gradient(_blur(g, 1.0))
    sxx, syy, sxy = _blur(gx * gx, 1.5), _blur(gy * gy, 1.5), _blur(gx * gy, 1.5)
    resp = sxx * syy - sxy * sxy - 0.05 * (sxx + syy) ** 2
    margin = int(PATCH * SPACING / 2) + 2
    resp[:margin] = resp[-margin:] = -np.inf
    resp[:, :margin] = resp[:, -margin:] = -np.inf
    if alpha is not None:
        # Corners on a transparent border are the border, not the scene.
        solid = _blur((alpha > 200).astype(np.float64), 2.0) > 0.999
        resp[~solid] = -np.inf
    finite = resp[np.isfinite(resp)]
    if not finite.size or finite.max() <= 0:
        return np.zeros((0, 2)), np.zeros((0, PATCH * PATCH))
    # Relative to this image's strongest corner: Harris responses scale with
    # the fourth power of contrast, so no absolute threshold fits every image.
    peaks = (resp == _max_filter(resp, 3)) & np.isfinite(resp) & (resp > finite.max() * 0.005)
    ys, xs = np.nonzero(peaks)
    if not ys.size:
        return np.zeros((0, 2)), np.zeros((0, PATCH * PATCH))
    vals = resp[ys, xs]
    # Spread them: the best few per grid cell, then the best overall.
    cells = 8
    cell_id = (ys * cells // h) * cells + (xs * cells // w)
    order = np.lexsort((-vals, cell_id))
    keep, per = [], max(1, count // (cells * cells))
    last, run = -1, 0
    for i in order:
        if cell_id[i] != last:
            last, run = cell_id[i], 0
        if run < per:
            keep.append(i)
        run += 1
    keep = np.asarray(keep)
    rest = np.setdiff1d(np.argsort(-vals)[:count * 2], keep)
    keep = np.concatenate([keep, rest])[:count]
    pts = np.stack([xs[keep] + 0.5, ys[keep] + 0.5], axis=1).astype(np.float64)
    return pts, describe(g, pts)


def describe(g, pts):
    """Normalised PATCH x PATCH samples of the blurred image around each point."""
    gb = _blur(g, SPACING * 0.6)
    h, w = gb.shape
    offs = (np.arange(PATCH) - (PATCH - 1) / 2.0) * SPACING
    ox, oy = np.meshgrid(offs, offs)
    sx = pts[:, 0:1] - 0.5 + ox.ravel()[None, :]
    sy = pts[:, 1:2] - 0.5 + oy.ravel()[None, :]
    x0 = np.clip(np.floor(sx).astype(np.int64), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(np.int64), 0, h - 2)
    tx, ty = np.clip(sx - x0, 0, 1), np.clip(sy - y0, 0, 1)
    v = (gb[y0, x0] * (1 - tx) * (1 - ty) + gb[y0, x0 + 1] * tx * (1 - ty)
         + gb[y0 + 1, x0] * (1 - tx) * ty + gb[y0 + 1, x0 + 1] * tx * ty)
    v = v - v.mean(axis=1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def match(da, db, ratio=0.8):
    """Index pairs (i in a, j in b) passing the ratio test both ways."""
    if not len(da) or len(db) < 2:
        return np.zeros((0, 2), np.int64)
    sim = da @ db.T                                   # cosine similarity
    d = np.sqrt(np.maximum(2 - 2 * sim, 0))
    j = np.argsort(d, axis=1)[:, :2]
    best = d[np.arange(len(da)), j[:, 0]]
    second = d[np.arange(len(da)), j[:, 1]]
    ok = best < ratio * second
    back = np.argmin(d, axis=0)
    mutual = back[j[:, 0]] == np.arange(len(da))
    keep = np.nonzero(ok & mutual)[0]
    return np.stack([keep, j[keep, 0]], axis=1)


def _similarity(src, dst):
    """Least-squares similarity (scale, rotation, translation) src -> dst."""
    n = len(src)
    a = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    a[0::2] = np.stack([src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)], 1)
    a[1::2] = np.stack([src[:, 1], src[:, 0], np.zeros(n), np.ones(n)], 1)
    b[0::2], b[1::2] = dst[:, 0], dst[:, 1]
    p, *_ = np.linalg.lstsq(a, b, rcond=None)
    s, r, tx, ty = p
    return np.array([[s, -r, tx], [r, s, ty], [0, 0, 1.0]])


def _homography_ls(src, dst):
    """DLT over all pairs, with Hartley normalisation."""
    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-9)
        t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return t, (p - c) * s
    ta, a = norm(src)
    tb, b = norm(dst)
    h = warp.homography(a, b) if len(a) == 4 else _dlt(a, b)
    out = np.linalg.inv(tb) @ h @ ta
    return out / out[2, 2]


def _dlt(src, dst):
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def ransac(src, dst, model="homography", threshold=3.0, iterations=1500, seed=0):
    """(3x3 transform src -> dst, inlier mask). None when it cannot fit."""
    n = len(src)
    need = 4 if model == "homography" else 2
    if n < need + 2:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    fit = _homography_ls if model == "homography" else _similarity
    for _ in range(iterations):
        idx = rng.choice(n, need, replace=False)
        try:
            m = fit(src[idx], dst[idx])
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
        if not np.all(np.isfinite(m)):
            continue
        err = np.hypot(*(warp.apply(m, src) - dst).T)
        inl = err < threshold
        c = int(inl.sum())
        if c > best_count:
            best, best_count = inl, c
            if c > 0.9 * n:
                break
    if best is None or best_count < need + 2:
        return None, np.zeros(n, bool)
    m = fit(src[best], dst[best])
    err = np.hypot(*(warp.apply(m, src) - dst).T)
    inl = err < threshold
    return fit(src[inl], dst[inl]), inl


def register(moving, reference, model="homography"):
    """(3x3 taking `moving`'s pixel coordinates into `reference`'s, inlier
    count). (None, 0) when the two do not match."""
    pa, da = features(moving, alpha=moving[..., 3] if moving.shape[-1] == 4 else None)
    pb, db = features(reference, alpha=reference[..., 3] if reference.shape[-1] == 4 else None)
    m = match(da, db)
    if len(m) < 6:
        return None, 0
    hm, inl = ransac(pa[m[:, 0]], pb[m[:, 1]], model)
    if hm is None:
        return None, 0
    return hm, int(inl.sum())


# ---- blending -----------------------------------------------------------------------

def _pyramid_blend(images, weights):
    """Blend RGBA images (float 0..1) by per-pixel weights through Laplacian
    pyramids -- seamless, as in exposure fusion (engine/refine.py)."""
    from .refine import _gaussian_pyramid, _laplacian_pyramid, _up
    h, w = images[0].shape[:2]
    levels = max(1, int(np.floor(np.log2(max(2, min(h, w))))) - 3)
    total = np.sum(weights, axis=0) + 1e-12
    fused = None
    for im, wt in zip(images, weights):
        lp = _laplacian_pyramid(im, levels)
        gw = _gaussian_pyramid((wt / total)[..., None], levels)
        part = [l * g for l, g in zip(lp, gw)]
        fused = part if fused is None else [f + p for f, p in zip(fused, part)]
    out = fused[-1]
    for lev in range(levels - 2, -1, -1):
        out = _up(out, fused[lev].shape) + fused[lev]
    return out


def _premul(rgba):
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    out = np.empty(rgba.shape, np.float64)
    out[..., :3] = rgba[..., :3] / 255.0 * a
    out[..., 3:] = a
    return out


def _unpremul(f):
    a = np.clip(f[..., 3:4], 0, 1)
    rgb = np.where(a > 1e-6, f[..., :3] / np.maximum(a, 1e-6), 0)
    out = np.empty(f.shape, np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * 255), 0, 255)
    out[..., 3:] = np.clip(np.rint(a * 255), 0, 255)
    return out


def focus_stack(images, softness=2.0):
    """Auto-Blend "stack images": each pixel from the sharpest source.

    Sharpness is the Laplacian's energy, smoothed; the weights are pushed
    hard toward the winner (a soft arg-max) so detail is not averaged away,
    then pyramid-blended so the joins do not show."""
    sharp = []
    for im in images:
        g = _grey(im)
        lap = np.abs(4 * g - np.roll(g, 1, 0) - np.roll(g, -1, 0) - np.roll(g, 1, 1) - np.roll(g, -1, 1))
        s = _blur(lap, softness)
        s[im[..., 3] < 8] = 0
        sharp.append(s)
    sharp = np.asarray(sharp)
    top = sharp.max(axis=0, keepdims=True) + 1e-9
    weights = (sharp / top) ** 12 + 1e-6
    weights *= np.asarray([im[..., 3] > 8 for im in images], np.float64)
    return _unpremul(_pyramid_blend([_premul(im) for im in images], list(weights)))


def panorama_blend(images):
    """Auto-Blend "panorama": weight by distance from each image's own edge
    (where it has pixels), pyramid-blend, so seams and exposure steps
    dissolve over the overlap."""
    from .morph import distance_sq
    weights = []
    for im in images:
        solid = im[..., 3] > 8
        d = np.sqrt(distance_sq(~solid, 400).astype(np.float64))
        weights.append(d * solid)
    blended = _pyramid_blend([_premul(im) for im in images], weights)
    out = _unpremul(blended)
    cover = np.any([im[..., 3] > 8 for im in images], axis=0)
    out[~cover] = 0
    return out


def canvas_for(sizes, transforms):
    """The bounding box (x0, y0, x1, y1) of images of `sizes` under
    `transforms` -- the canvas a panorama needs."""
    xs, ys = [], []
    for (w, h), hm in zip(sizes, transforms):
        c = warp.apply(hm, [(0, 0), (w, 0), (w, h), (0, h)])
        xs.extend(c[:, 0])
        ys.extend(c[:, 1])
    return (int(math.floor(min(xs))), int(math.floor(min(ys))),
            int(math.ceil(max(xs))), int(math.ceil(max(ys))))
