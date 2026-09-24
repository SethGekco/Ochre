# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Mapping RGBA into a FIXED palette.

Note what this module is not. It does not GENERATE palettes -- no median cut,
no octree, no k-means. Palette generation is genuinely contested, genuinely
algorithmic, and almost never wanted when the palette is dictated by a file
format. Matching into a palette you were given is the operation that actually
gets used, so that is what the core provides; generation belongs in an addon.

Protected indices are excluded structurally, not by preference. The snap LUT
is built over the unprotected entries only, so a reserved ramp or a
designated shadow index cannot be produced by any quantisation at all. That
is what lets a format addon promise those indices survive a conversion
untouched.

Two dithering modes, for two genuinely different situations:

  ordered  Bayer threshold matrix. Fully vectorised, so it runs at numpy
           speed on any canvas size, and it is deterministic and tile-
           independent -- the same pixel gets the same result regardless of
           how the image was split up.

  floyd    Floyd-Steinberg error diffusion. Better-looking on photographic
           content, but inherently SEQUENTIAL: each pixel's error feeds its
           neighbours, so it cannot be vectorised and costs a Python-level
           pass per pixel. Fine for sprites, painful above a megapixel, and
           the code says so rather than quietly taking a minute.
"""

import numpy as np

from .arith import mul255

NONE = "none"
ORDERED = "ordered"
FLOYD = "floyd"
MODES = (NONE, ORDERED, FLOYD)

# Above this pixel count, Floyd-Steinberg is refused unless forced. The limit
# is about honesty: error diffusion in Python is roughly a microsecond per
# pixel, so a 4000x4000 image would take a minute with no progress reported.
FLOYD_LIMIT = 1 << 20

# Classic 8x8 Bayer matrix, values 0..63.
_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.int16)


def quantize(rgba, palette, dither=NONE, bits=5, strength=1.0, force=False,
             origin=(0, 0)):
    """Map (H, W, 4) uint8 to a (H, W) uint8 index plane.

    `origin` is the image's position in a larger canvas, so an ordered dither
    tiles seamlessly when a region is quantised on its own.
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("quantize needs (H, W, 4) uint8, got %s" % (rgba.shape,))
    if dither not in MODES:
        raise ValueError("unknown dither mode %r" % (dither,))

    if dither == FLOYD:
        pixels = rgba.shape[0] * rgba.shape[1]
        if pixels > FLOYD_LIMIT and not force:
            raise ValueError(
                "Floyd-Steinberg on %d pixels would take roughly %.0f s in "
                "Python; use dither='ordered' or pass force=True"
                % (pixels, pixels / 1e6))
        return _floyd(rgba, palette, bits, strength)

    if dither == ORDERED:
        return _ordered(rgba, palette, bits, strength, origin)

    indices = _snap_exact_first(rgba[..., :3], palette, bits)
    return _apply_transparency(indices, rgba, palette)


def _snap_exact_first(rgb, palette, bits):
    """Snap, but resolve colours already in the palette EXACTLY.

    The snap LUT quantises its lookup to `bits` per channel, so a colour that
    is literally a palette entry can still land on a neighbour: at 5 bits,
    grey 77 falls in the cell whose midpoint is 76. Off by one is invisible
    on a photograph and fatal for a format round-trip, where re-saving an
    untouched sprite must reproduce the original bytes.

    So exact matches are resolved first, by value, and the LUT only handles
    colours that were not already in the palette.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    indices = np.asarray(palette.snap(rgb, bits), dtype=np.uint8)

    packed = (rgb[..., 0].astype(np.uint32) << 16
              | rgb[..., 1].astype(np.uint32) << 8
              | rgb[..., 2].astype(np.uint32))

    entries = palette.entries[:, :3].astype(np.uint32)
    pal_packed = (entries[:, 0] << 16) | (entries[:, 1] << 8) | entries[:, 2]

    # Prefer the LOWEST index among duplicates, so a palette carrying the
    # same colour twice resolves deterministically rather than by whichever
    # happened to sort last.
    candidates = [i for i in range(len(pal_packed))
                  if i not in palette.protected and i != palette.transparent]
    if not candidates:
        return indices
    cand = np.array(candidates, dtype=np.int64)
    values = pal_packed[cand]
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_index = cand[order]

    pos = np.searchsorted(sorted_values, packed)
    pos_clipped = np.clip(pos, 0, len(sorted_values) - 1)
    hit = sorted_values[pos_clipped] == packed
    if hit.any():
        indices = indices.copy()
        indices[hit] = sorted_index[pos_clipped][hit].astype(np.uint8)
    return indices


def _apply_transparency(indices, rgba, palette):
    """Force fully transparent pixels onto the palette's transparent index.

    Without this, a transparent pixel's meaningless RGB gets matched to some
    arbitrary opaque colour, and the moment the image is re-rendered the
    transparent regions light up.
    """
    indices = np.asarray(indices, dtype=np.uint8)
    if palette.transparent is None:
        return indices
    out = indices.copy()
    out[rgba[..., 3] < 128] = palette.transparent
    return out


def _ordered(rgba, palette, bits, strength, origin):
    """Bayer-threshold dithering. Vectorised, deterministic, tile-safe."""
    h, w = rgba.shape[:2]
    oy, ox = int(origin[1]), int(origin[0])
    # Indexed by ABSOLUTE position so a region quantised alone lines up with
    # the same region quantised as part of the whole image.
    ys = (np.arange(h) + oy) % 8
    xs = (np.arange(w) + ox) % 8
    threshold = _BAYER8[ys][:, xs].astype(np.int16)

    # Spread is the size of one quantisation step, so the dither nudges a
    # colour at most as far as the gap it is trying to hide.
    step = 256 >> bits
    amount = (threshold - 32) * step * float(strength) / 64.0

    nudged = np.clip(rgba[..., :3].astype(np.float64) + amount[..., None],
                     0, 255).astype(np.uint8)
    return _apply_transparency(palette.snap(nudged, bits), rgba, palette)


def _floyd(rgba, palette, bits, strength):
    """Floyd-Steinberg error diffusion. Sequential by nature."""
    h, w = rgba.shape[:2]
    work = rgba[..., :3].astype(np.float64)
    entries = palette.entries[:, :3].astype(np.float64)
    out = np.zeros((h, w), dtype=np.uint8)
    lut = palette.build_snap_lut(bits)
    shift = 8 - bits
    levels = 1 << bits
    amount = float(strength)
    # Same exactness rule as the undithered path: a colour already in the
    # palette resolves to its own index, not to a LUT neighbour.
    exact_map = {}
    for i in range(len(entries) - 1, -1, -1):
        if i in palette.protected or i == palette.transparent:
            continue
        exact_map[tuple(int(v) for v in entries[i])] = i

    for y in range(h):
        row = work[y]
        for x in range(w):
            old = row[x]
            clamped = np.clip(old, 0, 255).astype(np.int32)
            exact = exact_map.get((int(clamped[0]), int(clamped[1]), int(clamped[2])))
            if exact is not None:
                index = exact
            else:
                q = clamped >> shift
                index = int(lut[(q[0] * levels + q[1]) * levels + q[2]])
            out[y, x] = index
            error = (old - entries[index]) * amount
            # The classic 7/16, 3/16, 5/16, 1/16 distribution.
            if x + 1 < w:
                row[x + 1] += error * (7.0 / 16.0)
            if y + 1 < h:
                nxt = work[y + 1]
                if x > 0:
                    nxt[x - 1] += error * (3.0 / 16.0)
                nxt[x] += error * (5.0 / 16.0)
                if x + 1 < w:
                    nxt[x + 1] += error * (1.0 / 16.0)
    return _apply_transparency(out, rgba, palette)


# ---- error reporting -----------------------------------------------------

def quantization_error(rgba, indices, palette):
    """Mean and worst per-channel error, for reporting a conversion's cost."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    rendered = palette.render_entries[np.asarray(indices, dtype=np.uint8)]
    opaque = rgba[..., 3] >= 128
    if not opaque.any():
        return {"mean": 0.0, "max": 0, "pixels": 0}
    diff = np.abs(rendered[..., :3].astype(np.int32)
                  - rgba[..., :3].astype(np.int32))[opaque]
    return {"mean": float(diff.mean()), "max": int(diff.max()),
            "pixels": int(opaque.sum())}


def palette_usage(indices):
    """Which indices an image actually uses, and how often."""
    counts = np.bincount(np.asarray(indices, dtype=np.uint8).ravel(), minlength=256)
    return {i: int(c) for i, c in enumerate(counts) if c}


# ---- document conversion -------------------------------------------------

def convert_layer_to_indexed(doc, layer, palette=None, dither=NONE, bits=5,
                             strength=1.0):
    """Convert a layer to index-locked across every frame.

    Returns a per-cell error report, because a conversion is lossy and the
    user deserves to be told how lossy before they keep it.
    """
    palette = palette or doc.palette
    if palette is None:
        raise ValueError("bind a palette before converting a layer to indexed")
    if doc.palette is None:
        doc.bind_palette(palette)

    from .surface import Surface

    report = {}
    layer.lock_to_index()
    for key, surface in list(doc.cells_for_layer(layer).items()):
        replacement = Surface(doc.width, doc.height, planes=layer.planes,
                              authoritative=layer.authoritative,
                              palette=palette)
        source = surface.planes.get("rgba")
        if source is not None:
            indices = quantize(source, palette, dither, bits, strength)
            replacement.plane("index")[...] = indices
            replacement.refresh_derived()
            replacement.content_bbox = surface.content_bbox or surface.bounds
            report[key] = quantization_error(source, indices, palette)
        doc.cells[key] = replacement
    return report
