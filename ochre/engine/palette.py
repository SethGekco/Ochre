# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Palette: 256 RGBA entries, annotated index ranges, and nearest-colour snap.

This module is deliberately format-agnostic. It knows that a palette can carry
named index ranges with a role hint; it does not know what any of them mean.
An addon that understands a particular game's palette supplies the annotations
from its own data files. Nothing here has heard of any specific game, and
nothing here parses any on-disk palette format -- that is addon territory too.

Protected indices are the mechanism that makes annotations load-bearing rather
than decorative: an index marked protected is excluded from the snap LUT's
candidate set, so no quantiser, resampler or nearest-colour match can ever
produce it. That is a structural guarantee, not a heuristic.
"""

import numpy as np

LENGTH = 256


class Palette:
    """An indexed colour table. Document-level state, shared by every frame."""

    def __init__(self, entries=None, name="", protected=(), transparent=None,
                 annotations=()):
        if entries is None:
            entries = np.zeros((LENGTH, 4), dtype=np.uint8)
            entries[:, 3] = 255
        entries = np.asarray(entries, dtype=np.uint8)
        if entries.shape == (LENGTH, 3):
            rgba = np.empty((LENGTH, 4), dtype=np.uint8)
            rgba[:, :3] = entries
            rgba[:, 3] = 255
            entries = rgba
        if entries.shape != (LENGTH, 4):
            raise ValueError("palette needs %d RGB or RGBA entries, got %s"
                             % (LENGTH, (entries.shape,)))
        # A palette always owns a writable copy. Arrays arriving from Pillow
        # are read-only, and aliasing a caller's buffer would mean a palette
        # edit silently mutating whatever it was built from. One kilobyte.
        self.entries = np.array(entries, dtype=np.uint8, copy=True)
        self.name = name
        self.transparent = transparent
        self.annotations = list(annotations)   # (lo, hi, label, role)
        self._protected = frozenset(int(i) for i in protected)
        self._snap_lut = None
        self._snap_bits = None
        self._render = None
        self.revision = 0                       # bumped on any mutation

    # ---- rendering -------------------------------------------------------

    @property
    def render_entries(self):
        """Entries as they should be drawn: the transparent index gets alpha 0.

        Kept separate from `entries` so the authored colour of the transparent
        index survives a round-trip. Formats that designate an index as
        transparent still store a real RGB value there, and silently zeroing
        it would lose data the file legitimately carries.
        """
        if self.transparent is None:
            return self.entries
        if self._render is None:
            self._render = self.entries.copy()
            self._render[int(self.transparent), 3] = 0
        return self._render

    # ---- annotations -----------------------------------------------------

    def annotate(self, lo, hi, label, role=""):
        """Mark an index range with a label and an opaque role hint.

        The core stores and displays these. It never interprets `role`; that
        string is meaningful only to whoever supplied it.
        """
        lo, hi = int(lo), int(hi)
        if lo > hi:
            lo, hi = hi, lo
        self.annotations.append((lo, hi, label, role))
        self.revision += 1

    def annotation_for(self, index):
        """The first annotation covering index, or None."""
        for lo, hi, label, role in self.annotations:
            if lo <= index <= hi:
                return (lo, hi, label, role)
        return None

    # ---- protection ------------------------------------------------------

    @property
    def protected(self):
        return self._protected

    def protect(self, *indices):
        """Exclude indices from ever being produced by a snap."""
        new = set(self._protected)
        for spec in indices:
            if isinstance(spec, tuple) and len(spec) == 2:
                new.update(range(int(spec[0]), int(spec[1]) + 1))
            else:
                new.add(int(spec))
        self._protected = frozenset(i for i in new if 0 <= i < LENGTH)
        self._invalidate_snap()

    def unprotect_all(self):
        self._protected = frozenset()
        self._invalidate_snap()

    def candidates(self):
        """Indices a snap is allowed to emit: unprotected, and not transparent."""
        return np.array([i for i in range(LENGTH)
                         if i not in self._protected and i != self.transparent],
                        dtype=np.int32)

    # ---- mutation --------------------------------------------------------

    def set_entry(self, index, rgba):
        """Recolour one entry. Callers must refresh derived surfaces."""
        index = int(index)
        if not 0 <= index < LENGTH:
            raise IndexError("palette index %d out of range" % index)
        value = tuple(rgba)
        if len(value) == 3:
            value = value + (255,)
        self.entries[index] = value
        self._invalidate_snap()
        return index

    def set_transparent(self, index):
        """Designate an index as transparent, or None for fully opaque."""
        self.transparent = None if index is None else int(index)
        self._invalidate_snap()

    def _invalidate_snap(self):
        self._snap_lut = None
        self._snap_bits = None
        self._render = None
        self.revision += 1

    # ---- nearest-colour snap --------------------------------------------

    def build_snap_lut(self, bits=5):
        """Quantised nearest-colour table over the *unprotected* entries.

        5 bits per channel is 32768 cells, roughly 32 KB, built in ~0.2 s and
        cached until the palette changes. 6 bits is more accurate but costs
        over a second to build, which is rarely worth it.

        Excluding protected indices from the candidate set is what makes
        protection a guarantee: a protected colour is not merely unlikely to
        be chosen, it is absent from the table and therefore unreachable.
        """
        if self._snap_lut is not None and self._snap_bits == bits:
            return self._snap_lut

        levels = 1 << bits
        shift = 8 - bits
        cand = self.candidates()
        if cand.size == 0:
            raise ValueError("palette has no unprotected entries to snap to")

        # Representative colour of each quantised cell: the midpoint, so the
        # table is not biased toward black.
        axis = (np.arange(levels, dtype=np.int32) << shift) + (1 << (shift - 1)) \
            if shift > 0 else np.arange(levels, dtype=np.int32)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"),
                        axis=-1).reshape(-1, 3).astype(np.int32)

        pal = self.entries[cand, :3].astype(np.int32)
        out = np.empty(grid.shape[0], dtype=np.uint8)

        # Chunked so the distance matrix stays cache-friendly rather than
        # allocating cells x candidates x 3 in one go.
        step = 4096
        for start in range(0, grid.shape[0], step):
            block = grid[start:start + step]
            d = block[:, None, :] - pal[None, :, :]
            dist = np.einsum("ijk,ijk->ij", d, d)
            out[start:start + step] = cand[np.argmin(dist, axis=1)].astype(np.uint8)

        self._snap_lut = out
        self._snap_bits = bits
        return out

    def snap(self, rgb, bits=5):
        """Map RGB (..., 3) or RGBA (..., 4) to nearest unprotected indices."""
        lut = self.build_snap_lut(bits)
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        shift = 8 - bits
        q = (arr >> shift).astype(np.int32)
        levels = 1 << bits
        flat = (q[..., 0] * levels + q[..., 1]) * levels + q[..., 2]
        return lut[flat]

    def index_of(self, rgba):
        """Exact match, or None. Used to keep a round-trip lossless."""
        value = tuple(rgba)
        if len(value) == 3:
            value = value + (255,)
        hits = np.nonzero((self.entries == np.array(value, dtype=np.uint8)).all(axis=1))[0]
        return int(hits[0]) if hits.size else None

    # ---- misc ------------------------------------------------------------

    def rgba(self, index):
        return tuple(int(v) for v in self.entries[int(index)])

    def copy(self):
        return Palette(self.entries.copy(), self.name, self._protected,
                       self.transparent, list(self.annotations))

    def __len__(self):
        return LENGTH

    def __repr__(self):
        return ("Palette(name=%r, protected=%d, transparent=%r, annotations=%d)"
                % (self.name, len(self._protected), self.transparent,
                   len(self.annotations)))


def grayscale():
    """The one palette the core ships. Anything else comes from an addon."""
    entries = np.zeros((LENGTH, 4), dtype=np.uint8)
    ramp = np.arange(LENGTH, dtype=np.uint8)
    entries[:, 0] = entries[:, 1] = entries[:, 2] = ramp
    entries[:, 3] = 255
    return Palette(entries, name="Grayscale")
