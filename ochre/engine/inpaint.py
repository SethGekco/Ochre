# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Filling holes in an image.

Three algorithms, from cheap to expensive, and every inpainting tool is one
of them plus a way of saying where the hole is:

  harmonic_fill    The smoothest surface through the hole's border (Laplace's
                   equation). No texture at all, but instant, and exactly
                   right for small blemishes on smooth surfaces. Also the
                   engine of seamless blending below.

  seamless_blend   Poisson image editing (Perez et al., 2003), solved the
                   membrane way: paste the source, then add the harmonic
                   interpolation of the colour mismatch along the border.
                   The source keeps its detail, the seam disappears. This is
                   what the Patch tool and Content-Aware Move land with.

  content_aware_fill
                   Exemplar-based completion (Wexler, Shechtman and Irani,
                   2007 -- the method Photoshop's Content-Aware Fill grew
                   from), with PatchMatch (Barnes et al., 2009) for the
                   nearest-neighbour search. The hole is filled with texture
                   copied from the rest of the image so that every patch
                   overlapping the hole looks like some patch outside it.

Everything is premultiplied while it works, so a fill next to a transparent
edge does not drag a dark fringe in, and everything takes plain arrays -- no
Document, no Surface -- so it can be tested and reused on its own.

What this is NOT: Photoshop's generative "Remove" invents content with a
neural network. Nothing here invents anything; it can only rearrange what is
already in the picture. On a hole bigger than the texture around it, that is
visible, and no amount of tuning changes it.
"""

import math

import numpy as np


# ---- premultiplied float working space --------------------------------------------

def to_work(rgba):
    """uint8 straight RGBA -> float32 premultiplied, 0..255 in every channel."""
    a = rgba[..., 3:4].astype(np.float32)
    out = np.empty(rgba.shape, dtype=np.float32)
    out[..., :3] = rgba[..., :3].astype(np.float32) * (a / 255.0)
    out[..., 3:] = a
    return out


def from_work(work):
    """The inverse. Fully transparent pixels come back as (0, 0, 0, 0)."""
    a = np.clip(work[..., 3], 0.0, 255.0)
    safe = np.where(a > 1e-3, a, 1.0)
    out = np.empty(work.shape, dtype=np.uint8)
    rgb = np.clip(work[..., :3] * (255.0 / safe[..., None]), 0.0, 255.0)
    rgb[a <= 1e-3] = 0.0
    out[..., :3] = np.rint(rgb).astype(np.uint8)
    out[..., 3] = np.rint(a).astype(np.uint8)
    return out


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


# ---- harmonic fill ----------------------------------------------------------------

def _relax(u, hole, iterations):
    """Jacobi sweeps of the 4-neighbour mean, written only inside the hole."""
    if not hole.any():
        return u
    y0, x0, y1, x1 = _bbox(hole)
    h, w = hole.shape
    # One pixel of known border around the hole is all a sweep reads.
    ya, xa, yb, xb = max(0, y0 - 1), max(0, x0 - 1), min(h, y1 + 1), min(w, x1 + 1)
    sub = u[ya:yb, xa:xb]
    m = hole[ya:yb, xa:xb]
    for _ in range(iterations):
        p = np.pad(sub, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25
        sub[m] = avg[m]
    return u


def harmonic_fill(values, hole, iterations=40):
    """Replace values inside `hole` with the smoothest fill through its border.

    values: float (H, W, C), modified in place and returned. hole: bool (H, W).

    Solved coarse-to-fine: the hole is shrunk by halving until it is a few
    pixels across, filled there, and each finer level starts from the coarser
    answer, so a few dozen sweeps per level converge where plain Jacobi on a
    large hole would need tens of thousands.
    """
    hole = np.asarray(hole, dtype=bool)
    if not hole.any():
        return values
    if hole.all():
        mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
        values[...] = mean
        return values
    h, w = hole.shape
    if min(h, w) > 4 and hole.sum() > 16:
        # Coarser level: a 2x2 block is known if ANY of its pixels is, and
        # takes the mean of those. Odd edges are padded by replication.
        ph, pw = h + (h & 1), w + (w & 1)
        v = np.pad(values, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        k = np.pad(~hole, ((0, ph - h), (0, pw - w)), mode="edge").astype(np.float32)
        kv = v * k[..., None]
        sv = kv[0::2, 0::2] + kv[1::2, 0::2] + kv[0::2, 1::2] + kv[1::2, 1::2]
        sk = k[0::2, 0::2] + k[1::2, 0::2] + k[0::2, 1::2] + k[1::2, 1::2]
        coarse_hole = sk == 0
        coarse = sv / np.maximum(sk, 1.0)[..., None]
        coarse = harmonic_fill(coarse, coarse_hole, iterations)
        up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:h, :w]
        values[hole] = up[hole]
    else:
        known = values[~hole]
        values[hole] = known.mean(axis=0)
    return _relax(values, hole, iterations)


def seamless_blend(dest, src, region, iterations=60):
    """Poisson-blend `src` into `dest` over `region` (all float, same shape).

    Inside the region the result keeps src's gradients; on its border it
    meets dest exactly. Equivalent to solving the Poisson equation with the
    source as guidance field, done as: src + harmonic(dest - src).
    """
    region = np.asarray(region, dtype=bool)
    diff = dest - src
    diff = harmonic_fill(diff, region, iterations)
    out = dest.copy()
    out[region] = src[region] + diff[region]
    return out


# ---- morphology helpers (square structuring element, enough for patches) ----------

def _dilate(mask, r):
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    p = np.pad(mask, r, mode="constant")
    h, w = mask.shape
    for dy in range(-r, r + 1):
        row = p[r + dy:r + dy + h]
        for dx in range(-r, r + 1):
            out |= row[:, r + dx:r + dx + w]
    return out


def _erode(mask, r):
    return ~_dilate(~mask, r)


# ---- content-aware fill ------------------------------------------------------------

def _downsample(work, hole, source):
    h, w = hole.shape
    ph, pw = h + (h & 1), w + (w & 1)
    v = np.pad(work, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    hl = np.pad(hole, ((0, ph - h), (0, pw - w)), mode="edge")
    sr = np.pad(source, ((0, ph - h), (0, pw - w)), mode="constant")
    vs = (v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[1::2, 1::2]) * 0.25
    # The hole grows when halved (any covered pixel), the source shrinks
    # (all four must be usable): a coarse level must never copy from, or
    # leave unfilled, something the fine level would not.
    hs = hl[0::2, 0::2] | hl[1::2, 0::2] | hl[0::2, 1::2] | hl[1::2, 1::2]
    ss = sr[0::2, 0::2] & sr[1::2, 0::2] & sr[0::2, 1::2] & sr[1::2, 1::2]
    return vs.astype(np.float32), hs, ss & ~hs


class _Level:
    """One pyramid level, padded by the patch radius so every centre is legal."""

    def __init__(self, work, hole, source, r):
        self.r = r
        self.H, self.W = hole.shape
        self.img = np.pad(work, ((r, r), (r, r), (0, 0)), mode="edge").astype(np.float32)
        self.hole = np.pad(hole, r, mode="constant")
        src = np.pad(source & ~hole, r, mode="constant")
        # A source patch must lie wholly in the source region.
        self.valid = _erode(src, r)
        self.pw = self.W + 2 * r
        targets = _dilate(self.hole, r)
        inner = np.zeros_like(targets)
        inner[r:r + self.H, r:r + self.W] = True
        targets &= inner
        ty, tx = np.nonzero(targets)
        self.ty, self.tx = ty.astype(np.int32), tx.astype(np.int32)
        vy, vx = np.nonzero(self.valid)
        self.vy, self.vx = vy.astype(np.int32), vx.astype(np.int32)
        offs = np.arange(-r, r + 1)
        self.offs = (offs[:, None] * self.pw + offs[None, :]).ravel()

    def refresh_padding(self):
        r = self.r
        if r:
            self.img[...] = np.pad(self.img[r:r + self.H, r:r + self.W],
                                   ((r, r), (r, r), (0, 0)), mode="edge")

    @property
    def unpadded(self):
        r = self.r
        return self.img[r:r + self.H, r:r + self.W]

    def is_valid(self, sy, sx):
        ok = (sy >= 0) & (sy < self.valid.shape[0]) & (sx >= 0) & (sx < self.valid.shape[1])
        out = np.zeros(sy.shape, dtype=bool)
        out[ok] = self.valid[sy[ok], sx[ok]]
        return out


_CHUNK = 16384


def _patches(flat, centres, offs):
    return flat[centres[:, None] + offs[None, :]]


# How much a pixel inside the hole counts when matching patches, against 1
# for a known pixel. The hole holds only the current guess; at full weight a
# smooth first guess talks the search into smooth sources and the fill comes
# out as a flat disc. Chosen by eye on real photographs (0.3 against 1.0).
HOLE_WEIGHT = 0.3


def _dist(level, tpatch, tw, sy, sx):
    """Weighted sum of squared differences, target patches vs source patches.

    tw weights each patch pixel: known pixels count fully, pixels inside the
    hole (which hold only the current GUESS) count for less, so the guess
    cannot talk the search into matching itself.
    """
    flat = level.img.reshape(-1, level.img.shape[-1])
    out = np.empty(sy.shape[0], dtype=np.float32)
    cen = sy * level.pw + sx
    for i in range(0, cen.shape[0], _CHUNK):
        sp = _patches(flat, cen[i:i + _CHUNK], level.offs)
        d = sp - tpatch[i:i + _CHUNK]
        out[i:i + _CHUNK] = np.einsum("np,npc,npc->n", tw[i:i + _CHUNK], d, d)
    return out


def _patchmatch(level, ny, nx, rng, iterations):
    """Improve the nearest-neighbour field in place. Vectorised PatchMatch.

    The original is sequential (scan order carries good matches along);
    this does the same propagation for every target at once, with jumps of
    4, 2 and 1 pixels so a good match still spreads across the hole in a
    few passes. Random search then samples ever-smaller windows around the
    current match, which is what escapes local minima.
    """
    flat = level.img.reshape(-1, level.img.shape[-1])
    ty, tx = level.ty, level.tx
    tcen = ty * level.pw + tx
    tpatch = np.concatenate([_patches(flat, tcen[i:i + _CHUNK], level.offs)
                             for i in range(0, tcen.shape[0], _CHUNK)])
    holef = level.hole.ravel()
    tw = np.concatenate([np.where(holef[tcen[i:i + _CHUNK, None] + level.offs[None, :]],
                                  np.float32(HOLE_WEIGHT), np.float32(1.0))
                         for i in range(0, tcen.shape[0], _CHUNK)]).astype(np.float32)
    sy, sx = ny[ty, tx].copy(), nx[ty, tx].copy()
    best = _dist(level, tpatch, tw, sy, sx)
    hh, ww = level.valid.shape
    max_r = max(hh, ww)

    def attempt(cy, cx):
        nonlocal best
        ok = level.is_valid(cy, cx)
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        d = _dist(level, tpatch[idx], tw[idx], cy[idx], cx[idx])
        better = d < best[idx]
        j = idx[better]
        sy[j], sx[j], best[j] = cy[j], cx[j], d[better]

    for _ in range(iterations):
        for jump in (4, 2, 1):
            for dy, dx in ((0, -jump), (0, jump), (-jump, 0), (jump, 0)):
                py, px = ty + dy, tx + dx
                inside = (py >= 0) & (py < hh) & (px >= 0) & (px < ww)
                py, px = np.where(inside, py, ty), np.where(inside, px, tx)
                ny[ty, tx], nx[ty, tx] = sy, sx
                cy, cx = ny[py, px] - dy, nx[py, px] - dx
                has = inside & (ny[py, px] >= 0)
                cy, cx = np.where(has, cy, -1), np.where(has, cx, -1)
                attempt(cy, cx)
        radius = max_r
        while radius >= 1:
            cy = sy + rng.integers(-radius, radius + 1, size=sy.shape, dtype=np.int32)
            cx = sx + rng.integers(-radius, radius + 1, size=sx.shape, dtype=np.int32)
            attempt(cy, cx)
            radius //= 2
    ny[ty, tx], nx[ty, tx] = sy, sx
    return best


def _vote(level, ny, nx, dist, pct=75.0):
    """Every hole pixel becomes the weighted mean of the patches covering it."""
    flat = level.img.reshape(-1, level.img.shape[-1])
    C = flat.shape[1]
    ty, tx = level.ty, level.tx
    sy, sx = ny[ty, tx], nx[ty, tx]
    # Wexler's similarity weight: a good match counts for more.
    sigma2 = float(np.percentile(dist, pct)) if dist.size else 1.0
    wgt = np.exp(-dist / (2.0 * max(sigma2, 1e-3))).astype(np.float64)
    tcen = (ty * level.pw + tx).astype(np.int64)
    scen = (sy * level.pw + sx).astype(np.int64)
    holef = level.hole.ravel()
    n = flat.shape[0]
    acc = np.zeros((n, C), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for off in level.offs:
        p = tcen + off
        sel = holef[p]
        if not sel.any():
            continue
        pp, ww = p[sel], wgt[sel]
        vals = flat[scen[sel] + off]
        wsum += np.bincount(pp, weights=ww, minlength=n)
        for c in range(C):
            acc[:, c] += np.bincount(pp, weights=ww * vals[:, c], minlength=n)
    got = wsum > 0
    upd = holef & got
    flat[upd] = (acc[upd] / wsum[upd, None]).astype(np.float32)
    # The padding replicates the edge; where the hole touches the edge the
    # padding must follow the new values or patches there compare stale data.
    level.refresh_padding()


def _centre_assign(level, ny, nx):
    """Each hole pixel takes the centre of its own best match: no averaging."""
    flat = level.img.reshape(-1, level.img.shape[-1])
    hy, hx = np.nonzero(level.hole)
    sy, sx = ny[hy, hx], nx[hy, hx]
    ok = sy >= 0
    flat[hy[ok] * level.pw + hx[ok]] = flat[sy[ok] * level.pw + sx[ok]]
    level.refresh_padding()


# The last pass copies each hole pixel from the centre of its own best match
# instead of averaging every patch over it: averaging is what blurs fine
# detail (stars, grain) out of a fill. Adaptation then removes any seam.
FINAL = "centre"


def _adapt(level, ny, nx, iterations=60):
    """Gradient-domain colour adaptation of the fill (Photoshop's "Color
    Adaptation"), done properly.

    The fill already equals the original outside the hole, so blending it
    against the original finds nothing to correct. The seam lives in the
    TEXTURE: the pixel just outside the hole should equal what the copied
    texture would have had there. So for each border pixel q next to a hole
    pixel p matched to source s(p), the mismatch is
        original(q) - image(s(p) + (q - p))
    and that mismatch, spread smoothly across the hole (harmonic), is added
    to the fill. Exactly Poisson cloning per pixel.
    """
    hole = level.hole
    img = level.img
    H2, W2 = hole.shape
    ring = _dilate(hole, 1) & ~hole
    acc = np.zeros(img.shape, dtype=np.float64)
    cnt = np.zeros(hole.shape, dtype=np.float64)
    qy, qx = np.nonzero(ring)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        py, px = qy + dy, qx + dx
        ok = (py >= 0) & (py < H2) & (px >= 0) & (px < W2)
        ok[ok] = hole[py[ok], px[ok]]
        if not ok.any():
            continue
        sy, sx = ny[py[ok], px[ok]], nx[py[ok], px[ok]]
        good = sy >= 0
        q_y, q_x = qy[ok][good], qx[ok][good]
        cy = np.clip(sy[good] - dy, 0, H2 - 1)
        cx = np.clip(sx[good] - dx, 0, W2 - 1)
        acc[q_y, q_x] += img[q_y, q_x] - img[cy, cx]
        cnt[q_y, q_x] += 1.0
    known = cnt > 0
    if not known.any():
        return
    diff = np.zeros(img.shape, dtype=np.float32)
    diff[known] = (acc[known] / cnt[known, None]).astype(np.float32)
    # Solve over everything that is not a measured border pixel, so pixels
    # far from the hole cannot pin the solution to zero.
    harmonic_fill(diff, ~known, iterations)
    img[hole] = img[hole] + diff[hole]
    level.refresh_padding()


def _random_nnf(level, rng):
    ny = np.full(level.valid.shape, -1, dtype=np.int32)
    nx = np.full(level.valid.shape, -1, dtype=np.int32)
    k = rng.integers(0, level.vy.shape[0], size=level.ty.shape[0])
    ny[level.ty, level.tx] = level.vy[k]
    nx[level.ty, level.tx] = level.vx[k]
    return ny, nx


def _upsample_nnf(coarse, cny, cnx, fine, rng):
    """Carry the matches up a level: a coarse match at s becomes 2s + parity."""
    r_c, r_f = coarse.r, fine.r
    ny = np.full(fine.valid.shape, -1, dtype=np.int32)
    nx = np.full(fine.valid.shape, -1, dtype=np.int32)
    ty, tx = fine.ty, fine.tx
    uy, ux = ty - r_f, tx - r_f                       # unpadded fine coords
    cy = np.clip(uy // 2, 0, coarse.H - 1) + r_c
    cx = np.clip(ux // 2, 0, coarse.W - 1) + r_c
    my, mx = cny[cy, cx], cnx[cy, cx]
    have = my >= 0
    sy = (my - r_c) * 2 + (uy & 1) + r_f
    sx = (mx - r_c) * 2 + (ux & 1) + r_f
    ok = have & fine.is_valid(sy, sx)
    k = rng.integers(0, fine.vy.shape[0], size=ty.shape[0])
    ny[ty, tx] = np.where(ok, sy, fine.vy[k])
    nx[ty, tx] = np.where(ok, sx, fine.vx[k])
    return ny, nx


def content_aware_fill(rgba, hole, source=None, patch=7, seed=0, quality=1.0,
                       margin=None, adapt=True):
    """Fill `hole` with texture from elsewhere in the image.

    rgba    uint8 (H, W, 4) straight alpha. Not modified.
    hole    bool (H, W): True where pixels are to be replaced.
    source  bool (H, W) or None: where texture may come from (default:
            everywhere outside the hole, within `margin` of it).
    patch   patch width in pixels (odd). Bigger copies larger structure.
    margin  how far around the hole to look for texture. Default scales with
            the hole, which keeps the fill local -- nearby texture is almost
            always the right texture -- and keeps the cost bounded.

    Returns (rgba_out, rect) where rect is (y0, x0, y1, x1), the region that
    changed; pixels outside the hole are returned untouched.
    """
    hole = np.asarray(hole, dtype=bool)
    H, W = hole.shape
    bb = _bbox(hole)
    out = rgba.copy()
    if bb is None:
        return out, None
    y0, x0, y1, x1 = bb
    span = max(y1 - y0, x1 - x0)
    if margin is None:
        margin = int(max(3 * patch, span * 1.5, 32))
    cy0, cx0 = max(0, y0 - margin), max(0, x0 - margin)
    cy1, cx1 = min(H, y1 + margin), min(W, x1 + margin)
    crop_hole = hole[cy0:cy1, cx0:cx1]
    crop_src = ~crop_hole if source is None else (np.asarray(source, bool)[cy0:cy1, cx0:cx1]
                                                  & ~crop_hole)
    work = to_work(rgba[cy0:cy1, cx0:cx1])

    patch = max(3, int(patch) | 1)
    r = patch // 2
    rng = np.random.default_rng(seed)

    # Pyramid, coarsest last. Stop halving once the hole is about two
    # patches across, or the image would get too small to hold one.
    levels_data = [(work, crop_hole, crop_src)]
    while True:
        w_, h_, s_ = levels_data[-1]
        hb = _bbox(h_)
        hspan = 0 if hb is None else max(hb[2] - hb[0], hb[3] - hb[1])
        if hspan <= 2 * patch or min(h_.shape) < 4 * patch:
            break
        nw, nh, ns = _downsample(w_, h_, s_)
        if _erode(np.pad(ns, r, mode="constant"), r).sum() < 8:
            break                      # the coarse level would have no source
        levels_data.append((nw, nh, ns))

    levels = [_Level(w_, h_, s_, r) for (w_, h_, s_) in levels_data]
    coarsest = levels[-1]
    if coarsest.vy.size == 0 or coarsest.ty.size == 0:
        # Nowhere to copy from: the honest fallback is a smooth fill.
        filled = harmonic_fill(work.copy(), crop_hole)
        out[cy0:cy1, cx0:cx1] = from_work(filled)
        return out, (cy0, cx0, cy1, cx1)

    coarsest.unpadded[...] = harmonic_fill(coarsest.unpadded.copy(), levels_data[-1][1])
    coarsest.img[...] = np.pad(coarsest.unpadded.copy(),
                               ((r, r), (r, r), (0, 0)), mode="edge")
    ny, nx = _random_nnf(coarsest, rng)

    n_levels = len(levels)
    for li in range(n_levels - 1, -1, -1):
        level = levels[li]
        if li != n_levels - 1:
            prev = levels[li + 1]
            if level.vy.size == 0:
                break
            ny, nx = _upsample_nnf(prev, ny, nx, level, rng)
            # Initialise the hole by voting with the carried-up matches on
            # this level's own pixels: sharper than upsampling the colours.
            d0 = np.zeros(level.ty.shape[0], dtype=np.float32)
            _vote(level, ny, nx, d0)
        # More EM iterations where they are cheap (coarse), fewer where
        # each one costs four times as much.
        depth = n_levels - 1 - li
        em = max(2, int(round((6 - depth) * quality)))
        pm = 2 if li else 1
        for it in range(em):
            dist = _patchmatch(level, ny, nx, rng, pm)
            last = li == 0 and it == em - 1
            if last and FINAL == "centre":
                _centre_assign(level, ny, nx)
            else:
                _vote(level, ny, nx, dist)

    finest = levels[0]
    if adapt:
        _adapt(finest, ny, nx)
    result = from_work(finest.unpadded)
    region = out[cy0:cy1, cx0:cx1]
    region[crop_hole] = result[crop_hole]
    return out, (cy0, cx0, cy1, cx1)


def fill_hole(rgba, hole, method="content_aware", **kw):
    """Dispatch by name: 'content_aware' or 'smooth'. Returns new uint8 RGBA."""
    hole = np.asarray(hole, dtype=bool)
    if method == "smooth":
        bb = _bbox(hole)
        if bb is None:
            return rgba.copy()
        y0, x0, y1, x1 = bb
        H, W = hole.shape
        pad = 4
        ya, xa, yb, xb = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad), min(W, x1 + pad)
        work = to_work(rgba[ya:yb, xa:xb])
        harmonic_fill(work, hole[ya:yb, xa:xb])
        out = rgba.copy()
        res = from_work(work)
        sub = out[ya:yb, xa:xb]
        sub[hole[ya:yb, xa:xb]] = res[hole[ya:yb, xa:xb]]
        return out
    out, _ = content_aware_fill(rgba, hole, **kw)
    return out


def coverage_to_hole(coverage, threshold=1):
    """A uint8 coverage mask (a selection, a brush footprint) as a bool hole."""
    return np.asarray(coverage) >= threshold


def blend_result(original, filled, coverage):
    """Lerp filled over original by a 0..255 coverage. Soft edges for a fill."""
    t = (np.asarray(coverage, dtype=np.float32) / 255.0)[..., None]
    o = to_work(original)
    f = to_work(filled)
    return from_work(o + (f - o) * t)
