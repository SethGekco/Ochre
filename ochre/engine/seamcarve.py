# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Content-Aware Scale: resize by removing or repeating the least important
seams (Avidan and Shamir's seam carving, 2007), Photoshop's Content-Aware
Scale and GIMP's Liquid Rescale.

A seam is a connected path of pixels from top to bottom, one per row,
moving at most one column per row. Taking out the seam whose pixels change
least (lowest gradient energy) narrows the image by one column while
leaving the interesting parts -- faces, edges, text -- alone; repeating
low-energy seams widens it. Height works the same on the transposed image.

The seams are found ONCE on the merged image and the resulting column map
is applied to every layer, mask and frame, so the layers stay registered
with each other. A protect mask (the selection, in the editor) makes its
pixels too expensive to cut.
"""

import numpy as np

PROTECT_COST = 1e6


def energy(rgba, protect=None):
    """Gradient magnitude of luminance, plus a step for alpha edges."""
    a = rgba.astype(np.float64)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lum = lum * (a[..., 3] / 255.0) + a[..., 3] * 0.5
    gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:])) + np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, append=lum[-1:])) + np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
    e = gx + gy
    if protect is not None:
        e = e + (np.asarray(protect, np.float64) / 255.0) * PROTECT_COST
    return e


def find_seam(e):
    """The lowest-energy vertical seam: one column index per row."""
    h, w = e.shape
    M = e.copy()
    back = np.zeros((h, w), np.int8)
    inf = np.full((1,), np.inf)
    for y in range(1, h):
        prev = M[y - 1]
        left = np.concatenate((inf, prev[:-1]))
        right = np.concatenate((prev[1:], inf))
        choice = np.argmin(np.stack((left, prev, right)), axis=0)
        M[y] += np.choose(choice, (left, prev, right))
        back[y] = choice.astype(np.int8) - 1
    seam = np.empty(h, np.int64)
    seam[-1] = int(np.argmin(M[-1]))
    for y in range(h - 1, 0, -1):
        seam[y - 1] = seam[y] + back[y, seam[y]]
    return seam


def _delete(arr, seam):
    h, w = arr.shape[:2]
    keep = np.ones((h, w), bool)
    keep[np.arange(h), seam] = False
    return arr[keep].reshape((h, w - 1) + arr.shape[2:])


def column_map(rgba, new_w, protect=None):
    """(h, new_w) array: which source column each output pixel comes from.

    Narrowing removes seams; widening finds the seams that WOULD be removed
    first and repeats each of them, in passes of at most half the width so
    the same seam is not stretched over and over.
    """
    h, w = rgba.shape[:2]
    idx = np.tile(np.arange(w, dtype=np.int64), (h, 1))
    if new_w == w:
        return idx
    if new_w < w:
        e = energy(rgba, protect)
        for _ in range(w - new_w):
            seam = find_seam(e)
            e = _delete(e, seam)
            idx = _delete(idx, seam)
        return idx
    current = idx
    remaining = new_w - w
    base = rgba
    prot = protect
    while remaining > 0:
        cw = current.shape[1]
        k = min(remaining, max(1, cw // 2))
        e = energy(base, prot)
        pos = np.tile(np.arange(cw, dtype=np.int64), (h, 1))
        chosen = np.zeros((h, cw), bool)
        for _ in range(k):
            seam = find_seam(e)
            chosen[np.arange(h), pos[np.arange(h), seam]] = True
            e = _delete(e, seam)
            pos = _delete(pos, seam)
        # Repeat every chosen pixel once, in place.
        reps = 1 + chosen.astype(np.int64)
        out = np.empty((h, cw + k), np.int64)
        for y in range(h):
            out[y] = np.repeat(current[y], reps[y])
        current = out
        remaining -= k
        if remaining > 0:
            base = np.take_along_axis(rgba, current[..., None], axis=1)
            prot = None if protect is None else np.take_along_axis(protect, current, axis=1)
    return current


def apply_columns(plane, colmap):
    """A plane (h, w[, c]) remapped by a column map from column_map()."""
    if plane.ndim == 3:
        return np.take_along_axis(plane, colmap[..., None], axis=1)
    return np.take_along_axis(plane, colmap, axis=1)


def maps(rgba, new_w, new_h, protect=None):
    """Column map for the width, then row map (computed on the narrowed
    image, transposed) for the height."""
    cols = column_map(rgba, new_w, protect)
    narrowed = apply_columns(rgba, cols)
    prot = None if protect is None else apply_columns(protect, cols)
    rows_t = column_map(np.ascontiguousarray(narrowed.transpose(1, 0, 2)), new_h,
                        None if prot is None else np.ascontiguousarray(prot.T))
    return cols, rows_t


def apply_maps(plane, cols, rows_t):
    out = apply_columns(plane, cols)
    t = out.transpose(1, 0, 2) if out.ndim == 3 else out.T
    t = apply_columns(np.ascontiguousarray(t), rows_t)
    return np.ascontiguousarray(t.transpose(1, 0, 2) if t.ndim == 3 else t.T)
