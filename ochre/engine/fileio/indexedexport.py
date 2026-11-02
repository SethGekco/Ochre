# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Indexed export: GIF and PNG-8, as Export As writes them.

Photoshop's Save for Web (and Kooka Pictura's port of it) in Ochre's terms:
a palette of N colours made from the image by median cut, or the
document's own palette; optional dithering with an amount; 1-bit
transparency, with partly transparent pixels laid over a matte.

The one promise that matters most here is EXACTNESS. A document that is
already indexed -- an opened PNG-8 or GIF, a sprite on a game palette --
has indices that mean something beyond their colours: two entries can
share a colour, a range can be reserved. Exporting it on its own palette
must write those indices and that palette unchanged, not re-quantize the
colours and hope the same indices come back. So when the visible picture
is exactly the stacked index planes (checked against the real composite,
not assumed), the indices are written as they are. Anything that is not
exactly that -- a half-transparent layer, a blend mode, a resize --
is mapped to the palette instead, and the report says so.

Options (a dict, every key optional):

    colours        2..256 for a generated palette (default 256)
    palette        "generated" or "document" (default "generated")
    dither         "none", "ordered" or "floyd" (default "none")
    dither_amount  0..100, percent of full strength (default 100)
    transparency   alpha below 128 becomes a transparent index (default on)
    matte          (r, g, b) under partly transparent pixels (default white)

`report`, a dict when given, is filled with: exact (bool), colours (how
many palette entries were written), transparent (the index, or None).
"""

import io

import numpy as np
from PIL import Image

from ..palettefiles import median_cut
from ..quantize import FLOYD, NONE, ORDERED, quantize

INDEXED_FORMATS = {"GIF": "GIF", "PNG8": "PNG"}     # export name -> Pillow format
MATTE = (255, 255, 255)
_DITHER = {"none": NONE, "ordered": ORDERED, "floyd": FLOYD}


def exact_indices(doc, flat, frame=None):
    """The document's visible picture as palette indices, exactly -- or None
    when the picture is not simply its index-locked layers stacked.

    Every visible layer must be index-locked; the top-most non-transparent
    index wins at each pixel. The result is then rendered through the
    palette and compared with `flat` (the real composite): if any pixel
    differs -- an opacity, a blend mode, a mask, a style, an effect --
    the stacking was not the whole story and None is returned.
    """
    pal = doc.palette
    if pal is None or flat is None:
        return None
    frame = doc.frame if frame is None else frame
    leaves = [l for l in doc.layers() if l.effectively_visible()]
    if not leaves or not all(l.index_locked for l in leaves):
        return None
    out = np.full((doc.height, doc.width), -1, np.int16)
    render = pal.render_entries
    for layer in leaves:
        cell = doc.cells.get((layer.id, frame.id))
        if cell is None:
            continue
        cell_pal = getattr(cell, "palette", None)
        if cell_pal is not None and cell_pal is not pal:
            return None
        idx = cell.plane("index")
        shown = render[idx][..., 3] > 0
        out[shown] = idx[shown]
    if (out < 0).any():
        if pal.transparent is None:
            return None
        out[out < 0] = int(pal.transparent)
    out = out.astype(np.uint8)
    flat = np.asarray(flat)
    if flat.shape[:2] != out.shape:
        return None
    drawn = render[out]
    clear = drawn[..., 3] == 0
    if not np.array_equal(flat[..., 3][clear], drawn[..., 3][clear]):
        return None
    if not np.array_equal(flat[~clear], drawn[~clear]):
        return None
    return out


def _flatten(rgba, matte, transparency):
    """(work, has_clear): colours laid over the matte, alpha 255 -- or 0
    where the pixel becomes the transparent index."""
    from .flat import flatten_onto
    rgba = np.asarray(rgba, np.uint8)
    work = flatten_onto(rgba, matte)
    clear = (rgba[..., 3] < 128) if transparency else np.zeros(rgba.shape[:2], bool)
    work[..., 3] = np.where(clear, 0, 255).astype(np.uint8)
    return work, bool(clear.any())


def _spread(rgb):
    """The ordered dither's swing for a palette: the typical gap from a
    colour to its nearest neighbour (largest channel difference), so the
    pattern reaches the next colour and not far past it."""
    cols = np.asarray(rgb, np.int32).reshape(-1, 3)
    if len(cols) < 2:
        return 64.0
    gap = np.abs(cols[:, None, :] - cols[None, :, :]).max(-1)
    np.fill_diagonal(gap, 1 << 20)
    return float(np.clip(np.median(gap.min(1)), 8, 128))


def index_image(doc, flat, options=None, scaled=False):
    """(indices (H, W) uint8, palette rgb (n, 3) uint8, transparent index or
    None, exact) for `flat` -- the composite, already scaled if `scaled`."""
    options = options or {}
    matte = tuple(int(v) for v in (options.get("matte") or MATTE)[:3])
    transparency = bool(options.get("transparency", True))
    mode = _DITHER.get(options.get("dither", "none"), NONE)
    strength = max(0.0, min(1.0, float(options.get("dither_amount", 100)) / 100.0))
    if strength <= 0.0:
        mode = NONE

    if options.get("palette") == "document" and doc.palette is not None:
        pal = doc.palette
        idx = None if scaled else exact_indices(doc, flat)
        if idx is not None:
            # Written as they are. The palette is every real entry, and at
            # least up to the highest index used.
            n = max(int(getattr(pal, "size", 256) or 256), int(idx.max()) + 1)
            clear = pal.transparent if transparency else None
            return idx, np.array(pal.entries[:n, :3]), clear, True
        work, has_clear = _flatten(flat, matte, transparency)
        target = pal.copy()
        if pal.transparent is None:
            # No index set aside for "clear": nothing can be transparent.
            work[..., 3] = 255
            has_clear = False
        real = [i for i in target.candidates() if i < int(getattr(pal, "size", 256) or 256)]
        idx = quantize(work, target, mode, strength=strength, force=True,
                       spread=_spread(target.entries[real, :3]))
        n = max(int(getattr(pal, "size", 256) or 256), int(idx.max()) + 1)
        clear = pal.transparent if (transparency and has_clear) else None
        return idx, np.array(pal.entries[:n, :3]), clear, False

    colours = max(2, min(256, int(options.get("colours", 256))))
    work, has_clear = _flatten(flat, matte, transparency)
    budget = max(1, colours - 1 if has_clear else colours)
    opaque = work.copy()
    opaque[..., 3] = 255
    if has_clear:
        # The palette is made from the pixels that will be drawn.
        opaque = opaque[work[..., 3] > 0][None]
    pal = median_cut(opaque, budget)
    n = int(pal.size)
    spread = _spread(pal.entries[:n, :3])
    if has_clear:
        pal.set_entry(n, matte)
        pal.set_transparent(n)
        n += 1
    idx = quantize(work, pal, mode, strength=strength, force=True, spread=spread)
    return idx, np.array(pal.entries[:n, :3]), (n - 1 if has_clear else None), False


def encode_indexed(doc, fmt, options=None, flat=None, scaled=False, report=None, extra=None):
    """The GIF or PNG-8 file's bytes. `extra` is passed to Pillow's save
    (an embedded profile, for PNG-8)."""
    name = INDEXED_FORMATS[fmt.upper()]
    idx, rgb, clear, exact = index_image(doc, flat, options, scaled)
    h, w = idx.shape
    img = Image.frombytes("P", (w, h), np.ascontiguousarray(idx).tobytes())
    img.putpalette(np.ascontiguousarray(rgb, np.uint8).tobytes(), rawmode="RGB")
    kw = dict(extra or {})
    if clear is not None:
        kw["transparency"] = int(clear)
    # No optimizing: Pillow's would drop and renumber palette entries, and
    # the indices are the point.
    kw["optimize"] = False
    buf = io.BytesIO()
    img.save(buf, format=name, **kw)
    if report is not None:
        report.update({"exact": exact, "colours": int(len(rgb)), "transparent": clear})
    return buf.getvalue()
