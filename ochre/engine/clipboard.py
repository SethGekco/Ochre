# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Cut, copy and paste, in document terms.

A Clip is a block of straight-alpha RGBA with the selection already folded
into its alpha, plus -- when it came from an index-locked layer -- the
authoritative indices too. That second plane is why the clipboard lives in
the engine instead of being a QImage: pasting sprite art onto another sprite
with the same palette must move INDICES, not colours. Going through colour
would put every reserved entry back through nearest-match, which is the one
thing the indexed model exists to prevent.

The UI mirrors every Clip to the system clipboard as an image and reads
foreign images back from it, so this is not a walled garden; it is just
lossless within the application.
"""

import numpy as np

from .arith import mul255
from .geometry import Rect
from .surface import INDEX, RGBA


class Clip:
    __slots__ = ("rgba", "index", "palette", "origin")

    def __init__(self, rgba, origin, index=None, palette=None):
        self.rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        self.index = None if index is None else np.ascontiguousarray(index, dtype=np.uint8)
        self.palette = palette          # the Palette the indices mean something in
        self.origin = origin            # where it was cut from, document coords

    @property
    def width(self):
        return self.rgba.shape[1]

    @property
    def height(self):
        return self.rgba.shape[0]

    @staticmethod
    def from_rgba(rgba, origin=None):
        rgba = np.asarray(rgba, dtype=np.uint8)
        return Clip(rgba, origin or Rect(0, 0, rgba.shape[1], rgba.shape[0]))


def copy_selection(doc, layer, frame, selection, merged=False, compositor=None):
    """Lift the selected pixels. `merged` takes the flattened image instead."""
    rect = selection.bbox.clipped_to(doc.width, doc.height)
    if rect is None or rect.is_empty:
        return None
    mask = selection.mask_for(rect)

    if merged:
        if compositor is None:
            raise ValueError("copy merged needs the compositor")
        rgba = compositor.composite(rect, frame)
        index = palette = None
    else:
        cell = doc.cells.get((layer.id, frame.id))
        if cell is None:
            rgba = np.zeros((rect.h, rect.w, 4), dtype=np.uint8)
            index = palette = None
        else:
            rgba = cell.plane(RGBA)[rect.slice()].copy()
            index = palette = None
            if cell.index_locked:
                index = cell.plane(INDEX)[rect.slice()].copy()
                palette = cell.palette

    if mask is not None:
        rgba[..., 3] = mul255(rgba[..., 3], mask)
        if index is not None:
            clear = 0 if palette is None or palette.transparent is None else int(palette.transparent)
            index = np.where(mask >= 128, index, clear).astype(np.uint8)
    return Clip(rgba, rect, index, palette)


def paste(doc, layer, frame, clip, at=None):
    """Lay a clip onto a layer. Returns the rect it landed in, or None.

    Indices are written directly when the target shares the clip's palette,
    through the same coverage-threshold path every tool uses; anything else
    composites as colour and, on an indexed target, snaps. Either way the
    write goes through apply_masked, so selection clipping is not consulted
    here on purpose -- a paste lands where you put it.
    """
    cell = doc.cell(layer, frame)
    x, y = (clip.origin.x, clip.origin.y) if at is None else (int(at[0]), int(at[1]))
    target = Rect(x, y, clip.width, clip.height)
    landed = target.clipped_to(doc.width, doc.height)
    if landed is None:
        return None
    sx, sy = landed.x - target.x, landed.y - target.y
    window = (slice(sy, sy + landed.h), slice(sx, sx + landed.w))

    same_palette = (clip.index is not None and cell.index_locked
                    and cell.palette is not None and clip.palette is not None
                    and np.array_equal(cell.palette.entries, clip.palette.entries))
    if same_palette:
        clear = 0 if clip.palette.transparent is None else int(clip.palette.transparent)
        src = clip.index[window]
        cover = np.where(src != clear, 255, 0).astype(np.uint8)
        cell.apply_masked(landed, src, cover, name=INDEX)
    else:
        cell.apply_masked(landed, clip.rgba[window], None)
    return landed
