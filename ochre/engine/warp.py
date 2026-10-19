# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Geometric resampling: homographies, cages and displacement fields.

Every transform tool -- scale, rotate, shear, perspective, 3D, handle,
cage, warp, perspective crop -- reduces to one operation: for each pixel of
the DESTINATION, find where it came from in the source and sample there.
Inverse mapping, never forward: forward mapping leaves holes wherever the
transform stretches, and every editor that does it shows them.

Sampling is alpha-weighted (premultiplied while interpolating), because
averaging straight colour across differing alpha produces a dark fringe
along every transparent edge -- the commonest defect in hand-rolled
transform code. Index planes are always sampled nearest: halfway between
two palette entries is a third, unrelated colour.
"""

import math

import numpy as np

NEAREST, BILINEAR = "nearest", "bilinear"


# ---- homographies ---------------------------------------------------------------

def homography(src, dst):
    """3x3 H with H @ (x, y, 1) ~ (u, v, 1) for four point pairs (DLT)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def from_points(src, dst):
    """The transform that best maps 1-4 point pairs: what Handle Transform does.

    One pair translates, two give a similarity (move, uniform scale,
    rotate), three an affine (adds shear and independent scale), four a
    full perspective. That progression is GIMP's, and it is why a single
    handle feels like moving and four feel like pinning corners.
    """
    src = [tuple(map(float, p)) for p in src]
    dst = [tuple(map(float, p)) for p in dst]
    n = min(len(src), len(dst))
    if n == 0:
        return np.eye(3)
    if n == 1:
        (x, y), (u, v) = src[0], dst[0]
        return translate(u - x, v - y)
    if n == 2:
        (x0, y0), (x1, y1) = src[:2]
        (u0, v0), (u1, v1) = dst[:2]
        a = complex(x1 - x0, y1 - y0)
        b = complex(u1 - u0, v1 - v0)
        if abs(a) < 1e-9:
            return translate(u0 - x0, v0 - y0)
        z = b / a
        m = np.array([[z.real, -z.imag, 0], [z.imag, z.real, 0], [0, 0, 1.0]])
        t = complex(u0, v0) - z * complex(x0, y0)
        m[0, 2], m[1, 2] = t.real, t.imag
        return m
    if n == 3:
        a = np.array([[x, y, 1] for x, y in src[:3]])
        bx = np.array([u for u, _ in dst[:3]])
        by = np.array([v for _, v in dst[:3]])
        try:
            rx = np.linalg.solve(a, bx)
            ry = np.linalg.solve(a, by)
        except np.linalg.LinAlgError:
            return from_points(src[:2], dst[:2])
        return np.array([rx, ry, [0, 0, 1.0]])
    return homography(src[:4], dst[:4])


def translate(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def scale_about(sx, sy, cx, cy):
    return translate(cx, cy) @ np.diag([sx, sy, 1.0]) @ translate(-cx, -cy)


def rotate_about(radians, cx, cy):
    c, s = math.cos(radians), math.sin(radians)
    return translate(cx, cy) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ translate(-cx, -cy)


def shear_about(kx, ky, cx, cy):
    return translate(cx, cy) @ np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1.0]]) @ translate(-cx, -cy)


def rotation3d(width, height, rx, ry, rz=0.0, distance=None):
    """A plane of width x height tilted in 3-D, seen through a pinhole.

    rx tips it about the horizontal axis, ry about the vertical, rz spins it
    in the plane. The four corners are rotated about the plane's centre and
    projected, and the homography through them is the transform -- GIMP's 3D
    Transform reduced to the part that maps onto a flat image.
    """
    w, h = float(width), float(height)
    f = float(distance or max(w, h) * 1.6)
    cx, cy = w / 2.0, h / 2.0
    cxr, sxr = math.cos(rx), math.sin(rx)
    cyr, syr = math.cos(ry), math.sin(ry)
    czr, szr = math.cos(rz), math.sin(rz)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = []
    for x, y in corners:
        X, Y, Z = x - cx, y - cy, 0.0
        X, Y = X * czr - Y * szr, X * szr + Y * czr
        Y, Z = Y * cxr - Z * sxr, Y * sxr + Z * cxr
        X, Z = X * cyr + Z * syr, -X * syr + Z * cyr
        s = f / max(f + Z, 1e-6)
        dst.append((cx + X * s, cy + Y * s))
    return homography(corners, dst)


def apply(h, pts):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    q = np.hstack([pts, ones]) @ h.T
    w = q[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return q[:, :2] / w


def bounds_of(h, width, height):
    """Integer (x0, y0, x1, y1) enclosing a width x height source under h."""
    c = apply(h, [(0, 0), (width, 0), (width, height), (0, height)])
    return (int(math.floor(c[:, 0].min())), int(math.floor(c[:, 1].min())),
            int(math.ceil(c[:, 0].max())), int(math.ceil(c[:, 1].max())))


# ---- sampling -------------------------------------------------------------------

def _sample_rgba(src, sx, sy, method):
    """Sample src (h, w, 4) uint8 at float coords; outside is transparent."""
    h, w = src.shape[:2]
    if method == NEAREST:
        ix = np.floor(sx).astype(np.int64)
        iy = np.floor(sy).astype(np.int64)
        inside = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
        out = np.zeros(sx.shape + (4,), dtype=np.uint8)
        out[inside] = src[iy[inside], ix[inside]]
        return out
    # Bilinear on premultiplied colour, sampling pixel CENTRES.
    fx, fy = sx - 0.5, sy - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx, ty = fx - x0, fy - y0
    a = src[..., 3].astype(np.float64) / 255.0
    pre = np.empty(src.shape[:2] + (4,), dtype=np.float64)
    pre[..., :3] = src[..., :3] * a[..., None]
    pre[..., 3] = a
    acc = np.zeros(sx.shape + (4,), dtype=np.float64)
    for dy, wy in ((0, 1.0 - ty), (1, ty)):
        for dx, wx in ((0, 1.0 - tx), (1, tx)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
            wgt = (wx * wy)[..., None]
            vals = np.zeros(sx.shape + (4,), dtype=np.float64)
            vals[ok] = pre[yy[ok], xx[ok]]
            acc += vals * wgt
    alpha = acc[..., 3]
    out = np.zeros(sx.shape + (4,), dtype=np.uint8)
    safe = np.where(alpha > 1e-6, alpha, 1.0)
    out[..., :3] = np.clip(np.rint(acc[..., :3] / safe[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    out[..., :3][alpha <= 1e-6] = 0
    return out


def _grid(x0, y0, x1, y1):
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5


def warp_homography(src, h, out_box, method=BILINEAR, index=False):
    """Render src through h into the destination box (x0, y0, x1, y1).

    h maps SOURCE pixel coordinates to destination coordinates; sampling
    uses its inverse. Returns an array the size of the box. With index=True
    src is a 2-D index plane, sampled nearest, outside = -1.
    """
    x0, y0, x1, y1 = out_box
    if x1 <= x0 or y1 <= y0:
        shape = (max(0, y1 - y0), max(0, x1 - x0))
        return np.zeros(shape + (() if index else (4,)), dtype=np.int16 if index else np.uint8)
    inv = np.linalg.inv(h)
    gx, gy = _grid(x0, y0, x1, y1)
    den = inv[2, 0] * gx + inv[2, 1] * gy + inv[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    sx = (inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]) / den
    sy = (inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]) / den
    if index:
        return sample_index(src, sx, sy)
    return _sample_rgba(src, sx, sy, method)


def sample_index(src, sx, sy):
    h, w = src.shape[:2]
    ix = np.floor(sx).astype(np.int64)
    iy = np.floor(sy).astype(np.int64)
    inside = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
    out = np.full(sx.shape, -1, dtype=np.int16)
    out[inside] = src[iy[inside], ix[inside]]
    return out


def warp_field(src, sx, sy, method=BILINEAR):
    """Sample src at explicit per-pixel source coordinates (displacement maps)."""
    return _sample_rgba(src, sx, sy, method)


# ---- cage: mean value coordinates ----------------------------------------------

def mean_value_map(points, cage_from, cage_to):
    """Map points inside cage_from to the matching position in cage_to.

    Mean value coordinates (Floater 2003) of each point with respect to the
    first cage, applied to the second. Used for the cage transform as an
    INVERSE map: coordinates are computed against the deformed cage and
    applied to the original, which is exact for affine deformations and a
    close approximation otherwise -- the standard trade for being able to
    render by inverse mapping.
    """
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(cage_from, dtype=np.float64)
    t = np.asarray(cage_to, dtype=np.float64)
    n = len(v)
    d = v[None, :, :] - p[:, None, :]                    # (P, n, 2)
    r = np.hypot(d[..., 0], d[..., 1])
    r = np.maximum(r, 1e-9)
    nxt = np.roll(np.arange(n), -1)
    cross = d[..., 0] * d[:, nxt, 1] - d[..., 1] * d[:, nxt, 0]
    dot = d[..., 0] * d[:, nxt, 0] + d[..., 1] * d[:, nxt, 1]
    angle = np.arctan2(cross, dot)                        # signed angle at p
    tan_half = np.tan(angle / 2.0)
    prev = np.roll(np.arange(n), 1)
    w = (tan_half[:, prev] + tan_half) / r
    total = w.sum(axis=1, keepdims=True)
    total = np.where(np.abs(total) < 1e-12, 1e-12, total)
    lam = w / total
    return lam @ t


def point_in_polygon(px, py, poly):
    poly = np.asarray(poly, dtype=np.float64)
    inside = np.zeros(np.broadcast(px, py).shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if ay == by:
            continue
        cond = (ay > py) != (by > py)
        xc = (bx - ax) * (py - ay) / (by - ay) + ax
        inside ^= cond & (px < xc)
    return inside


# ---- moving least squares (Puppet Warp) ---------------------------------------------

def mls(v, p, q, rigid=True, alpha=1.0):
    """Moving least squares deformation of points v by handles p -> q.

    Schaefer, McPhail and Warren (2006), in the closed form complex numbers
    allow in 2-D: at each point, the best similarity (scale x rotation)
    taking the weighted handles p onto q is
        M = sum(w q^ conj(p^)) / sum(w |p^|^2)
    and the RIGID variant keeps only its rotation, M / |M|. v is (N, 2),
    p and q (k, 2); returns (N, 2). One handle simply translates.
    """
    v = np.asarray(v, np.float64)
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.asarray(q, np.float64).reshape(-1, 2)
    if len(p) == 0:
        return v.copy()
    if len(p) == 1:
        return v + (q[0] - p[0])
    vc = v[:, 0] + 1j * v[:, 1]
    pc = p[:, 0] + 1j * p[:, 1]
    qc = q[:, 0] + 1j * q[:, 1]
    d2 = np.abs(vc[:, None] - pc[None, :]) ** 2
    exact = d2 < 1e-12
    w = 1.0 / np.maximum(d2, 1e-12) ** alpha
    ws = w.sum(axis=1)
    ps = (w * pc[None, :]).sum(axis=1) / ws
    qs = (w * qc[None, :]).sum(axis=1) / ws
    ph = pc[None, :] - ps[:, None]
    qh = qc[None, :] - qs[:, None]
    num = (w * qh * np.conj(ph)).sum(axis=1)
    if rigid:
        mag = np.abs(num)
        M = np.where(mag > 1e-12, num / np.where(mag > 1e-12, mag, 1), 1.0)
    else:
        den = (w * np.abs(ph) ** 2).sum(axis=1)
        M = num / np.where(den > 1e-12, den, 1)
    f = M * (vc - ps) + qs
    # A point exactly on a handle goes exactly where the handle goes.
    if exact.any():
        rows, cols = np.nonzero(exact)
        f[rows] = qc[cols]
    return np.stack([f.real, f.imag], axis=1)


def mls_field(width, height, p, q, rigid=True, step=8):
    """Per-pixel source coordinates for an MLS warp of a width x height
    image, evaluated on a coarse grid and interpolated (MLS is smooth, and
    a full-resolution evaluation is pins x pixels of complex arithmetic).

    The roles are swapped -- output points are mapped from the MOVED handles
    back to the original ones -- so the result is an inverse map, ready for
    warp_field. Returns (sx, sy), each (height, width).
    """
    gx = np.arange(0, width + step, step, dtype=np.float64)
    gy = np.arange(0, height + step, step, dtype=np.float64)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    src = mls(pts, q, p, rigid).reshape(len(gy), len(gx), 2)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xs += 0.5
    ys += 0.5
    fx, fy = xs / step, ys / step
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, len(gx) - 2)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, len(gy) - 2)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    a = src[y0, x0]
    b = src[y0, x0 + 1]
    c = src[y0 + 1, x0]
    d = src[y0 + 1, x0 + 1]
    out = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
    return out[..., 0], out[..., 1]
