# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Stroke sessions: one drag becomes one undo entry, and preview comes free.

The obvious designs are both wrong. Snapshotting per dab is O(dabs) and far
too slow -- a stroke can be thousands of dabs. Accumulating a region and
snapshotting at mouse-up needs expensive region unions on the hot path, and
gives you nothing to render a live preview from.

The design Paint.NET uses, and this follows, is lazy copy-on-first-touch at
BLOCK granularity:

  - A bool array tracks which blocks have been saved. Looking up a bit is the
    entire hot-path bookkeeping cost -- no region arithmetic at all.
  - The first time a stroke touches a block, that block's authoritative planes
    are copied to a scratch buffer, uncompressed. Compressing per dab would
    cost milliseconds per input event.
  - Runs of adjacent unsaved blocks WITHIN A ROW are coalesced into a single
    copy, so a wide stroke performs a few large copies rather than hundreds of
    small ones.
  - At mouse-up the touched region is assembled and compressed exactly once.

Cost is therefore O(area touched), paid once, regardless of dab count.

The second payoff is the one that is easy to miss: the scratch buffer is also
a live-preview buffer. `restore()` puts the original pixels back, so a shape,
line or gradient tool re-renders on every mouse-move by restoring and
redrawing. The undo snapshot and the rubber-band preview are the same memory,
which means interactive shape preview needs no separate preview layer and
cannot drift out of sync with what will actually be committed.
"""

import numpy as np

from .commands import PixelDelta
from .geometry import Rect


class StrokeSession:
    """One interaction on one cell, from mouse-down to mouse-up."""

    def __init__(self, doc, layer, frame=None, block=256, max_blocks=512,
                 level=1, label="Paint"):
        self.doc = doc
        self.layer = layer
        self.frame = doc.frame if frame is None else frame
        self.layer_id = getattr(layer, "id", layer)
        self.frame_id = self.frame.id
        self.block = int(block)
        self.max_blocks = int(max_blocks)
        self.level = level
        self.label = label

        self.surface = doc.cell(layer, self.frame)
        if self.surface is None:
            raise ValueError("no cell for layer %r" % (self.layer_id,))

        rows = (self.surface.height + self.block - 1) // self.block
        cols = (self.surface.width + self.block - 1) // self.block
        self._saved = np.zeros((rows, cols), dtype=bool)
        self._scratch = {}          # (by, bx) -> {plane: uncompressed array}
        self._union = None
        self._committed = []        # auto-commits, if the stroke ran long
        self.dab_count = 0

    # ---- bookkeeping -----------------------------------------------------

    @property
    def saved_blocks(self):
        return int(self._saved.sum())

    @property
    def region(self):
        """Union of everything touched so far, or None."""
        return self._union

    def _block_rect(self, by, bx):
        return Rect(bx * self.block, by * self.block,
                    self.block, self.block).clipped_to(
                        self.surface.width, self.surface.height)

    def _block_range(self, rect):
        r = rect.clipped_to(self.surface.width, self.surface.height)
        if r is None:
            return None
        return (r.y // self.block, (r.y1 - 1) // self.block,
                r.x // self.block, (r.x1 - 1) // self.block)

    # ---- the hot path ----------------------------------------------------

    def touch(self, rect):
        """Checkpoint any not-yet-saved blocks under rect. Call before writing.

        Adjacent unsaved blocks in a row are merged into one copy, which is
        what keeps a wide stroke from doing hundreds of tiny memcpys.
        """
        span = self._block_range(rect)
        if span is None:
            return None
        y0, y1, x0, x1 = span
        self.dab_count += 1

        for by in range(y0, y1 + 1):
            run_start = None
            for bx in range(x0, x1 + 2):        # one past the end, to flush
                unsaved = bx <= x1 and not self._saved[by, bx]
                if unsaved and run_start is None:
                    run_start = bx
                elif not unsaved and run_start is not None:
                    self._save_run(by, run_start, bx - 1)
                    run_start = None

        r = rect.clipped_to(self.surface.width, self.surface.height)
        if r is not None:
            self._union = r if self._union is None else self._union.union(r)

        # A pathological drag must not exhaust memory: fold what we have into
        # a committed delta and start a fresh session's worth of checkpoints.
        if self.saved_blocks > self.max_blocks:
            self._auto_commit()
        return r

    def _save_run(self, by, bx0, bx1):
        """Copy one horizontal run of blocks in a single operation."""
        first, last = self._block_rect(by, bx0), self._block_rect(by, bx1)
        if first is None or last is None:
            return
        run = first.union(last)
        planes = self.surface.read(run)
        for bx in range(bx0, bx1 + 1):
            self._saved[by, bx] = True
        self._scratch[(by, bx0, bx1)] = (run, planes)

    # ---- preview ---------------------------------------------------------

    def restore(self, rect=None):
        """Put the original pixels back over rect (or everything touched).

        This is what makes live rubber-band preview free. A shape tool calls
        restore() then redraws on every mouse-move; the buffer it restores
        from is the same one that will become the undo entry, so the preview
        and the committed result cannot disagree.
        """
        target = self._union if rect is None else rect
        if target is None:
            return None
        touched = None
        for (run, planes) in self._scratch.values():
            overlap = run.intersect(target)
            if overlap is None:
                continue
            oy, ox = overlap.y - run.y, overlap.x - run.x
            for name, arr in planes.items():
                self.surface.plane(name)[overlap.slice()] = \
                    arr[oy:oy + overlap.h, ox:ox + overlap.w]
            touched = overlap if touched is None else touched.union(overlap)
        if touched is not None:
            self.surface.refresh_derived(touched)
            self.surface.dirty.add(touched)
        return touched

    # ---- commit ----------------------------------------------------------

    def _assemble_before(self, region):
        """Rebuild the pre-stroke pixels over region from the checkpoints."""
        out = {}
        for name in self.surface.authoritative:
            current = self.surface.plane(name)
            out[name] = np.array(current[region.slice()], copy=True)
        for (run, planes) in self._scratch.values():
            overlap = run.intersect(region)
            if overlap is None:
                continue
            sy, sx = overlap.y - run.y, overlap.x - run.x
            dy, dx = overlap.y - region.y, overlap.x - region.x
            for name, arr in planes.items():
                if name not in out:
                    continue
                out[name][dy:dy + overlap.h, dx:dx + overlap.w] = \
                    arr[sy:sy + overlap.h, sx:sx + overlap.w]
        return out

    def _auto_commit(self):
        delta = self._build_delta()
        if delta is not None:
            self._committed.append(delta)
        self._saved[...] = False
        self._scratch.clear()
        self._union = None

    def _build_delta(self):
        if self._union is None or not self._scratch:
            return None
        region = self._union
        before = self._assemble_before(region)
        return PixelDelta(self.layer_id, self.frame_id, region, before,
                          self.label, self.level)

    def commit(self):
        """Finish the stroke and return one command, or None if nothing moved.

        Compression happens here, once, rather than per dab -- which is the
        whole reason checkpoints are held uncompressed during the drag.
        """
        delta = self._build_delta()
        parts = self._committed + ([delta] if delta is not None else [])
        self._scratch.clear()
        self._saved[...] = False
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        from .commands import CompoundCommand
        return CompoundCommand(parts, self.label)

    def cancel(self):
        """Roll the whole stroke back. Leaves no history behind."""
        for delta in reversed(self._committed):
            delta.undo(self.doc)
        touched = self.restore()
        self._committed.clear()
        self._scratch.clear()
        self._saved[...] = False
        self._union = None
        return touched
