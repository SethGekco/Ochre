# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Vanishing Point tool (engine/vanishing.py has the geometry).

  Edit Plane  click four corners of something rectangular seen in
              perspective -- a wall, a floor, a box side -- in any order;
              then drag a corner to adjust. Click elsewhere to start over.
  Stamp       Ctrl-click inside the plane to set the source, then paint:
              the brush is a circle ON the plane, so it foreshortens as it
              recedes, and what it copies is carried along the plane --
              a tile from the near end lands smaller at the far end.

The plane is a document mark (doc.planes), drawn with its grid while the
tool is in hand; edits to it are undoable, and each stamp stroke is one
undo step on the layer.
"""

import math

from .. import vanishing as V
from ..commands import MarksDelta
from ..geometry import Rect
from ..stroke import StrokeSession
from . import options as O
from .base import Tool


class VanishingPointTool(Tool):
    name = "vanishing_point"
    label = "Vanishing Point"
    snaps = False
    overlay_needs_doc = True      # the plane is document state, drawn before any click
    options_spec = (
        O.Opt("mode", O.CHOICE, "stamp", "Mode", choices=("plane", "stamp"),
              labels={"plane": "Edit Plane", "stamp": "Stamp"}),
        O.size(40),
        O.hardness(0.5),
        O.Opt("aligned", O.BOOL, True, "Aligned",
              tip="Keep the source offset between strokes."),
        O.Opt("grid", O.INT, 8, "Grid", minimum=2, maximum=40),
    )

    def __init__(self, **options):
        super().__init__(**options)
        self._clicks = []
        self._corner = None
        self._marks_before = None
        self._source = None          # image point
        self._offset = None          # plane units, source - destination
        self._session = None
        self._snapshot = None
        self._last = None

    # ---- helpers ---------------------------------------------------------------

    @staticmethod
    def plane(doc):
        planes = getattr(doc, "planes", None) or []
        return planes[-1] if planes else None

    def _near_corner(self, ctx, plane, x, y):
        r = max(4.0, 2.0 * float(getattr(ctx, "pick_radius", 4.0)))
        for i, (cx, cy) in enumerate(plane):
            if math.hypot(cx - x, cy - y) <= r:
                return i
        return None

    # ---- events ---------------------------------------------------------------

    def on_begin(self, ctx, event):
        doc = ctx.doc
        plane = self.plane(doc)
        mode = self.option("mode", "stamp")
        if plane is not None:
            corner = self._near_corner(ctx, plane, event.x, event.y)
            if corner is not None and not event.has(2):
                self._marks_before = MarksDelta(doc, "Edit Plane", ("planes",))
                self._corner = corner
                return doc.bounds
        if plane is None or mode == "plane":
            self._clicks.append((float(event.x), float(event.y)))
            if len(self._clicks) == 4:
                quad = V.ordered(self._clicks)
                self._clicks = []
                if V.valid(quad):
                    before = MarksDelta(doc, "Create Plane", ("planes",))
                    doc.planes = [tuple(quad)]
                    if ctx.history is not None:
                        ctx.history.push(before, "Create Plane")
                    self._source = self._offset = None
                else:
                    return {"error": "those four points do not make a plane"}
            return doc.bounds

        # Stamp.
        if event.has(2):
            if not V.inside(plane, event.x, event.y):
                return {"error": "set the source inside the plane"}
            self._source = (float(event.x), float(event.y))
            self._offset = None
            return doc.bounds
        if self._source is None:
            return {"error": "ctrl-click inside the plane to set a source first"}
        cell = ctx.cell
        if cell is None:
            return None
        if not V.inside(plane, event.x, event.y):
            return None
        d = V.to_plane(plane, [(event.x, event.y)])[0]
        if self._offset is None or not self.option("aligned", True):
            s = V.to_plane(plane, [self._source])[0]
            self._offset = (float(s[0] - d[0]), float(s[1] - d[1]))
        st = ctx.settings
        self._session = StrokeSession(
            doc, ctx.layer, ctx.frame,
            block=256 if st is None else st.get("History", "BlockSize"),
            max_blocks=512 if st is None else st.get("History", "MaxStrokeBlocks"),
            level=1 if st is None else st.get("History", "CompressLevel"),
            label=self.label)
        # The AUTHORITATIVE plane: indices on an index-locked layer, sampled
        # nearest, so cloned pixels keep their exact palette entries.
        self._plane = "index" if cell.index_locked else "rgba"
        self._snapshot = cell.plane(self._plane).copy()
        self._last = (float(d[0]), float(d[1]))
        return self._dab(ctx, self._last)

    def on_motion(self, ctx, event):
        doc = ctx.doc
        if self._corner is not None:
            plane = list(self.plane(doc))
            plane[self._corner] = (float(event.x), float(event.y))
            if V.valid(plane):
                doc.planes = list(doc.planes[:-1]) + [tuple(plane)]
            return doc.bounds
        if self._session is None:
            return None
        plane = self.plane(doc)
        uv = V.to_plane(plane, [(event.x, event.y)])[0]
        x0, y0 = self._last
        size = max(1.0, float(self.option("size", 40)))
        spacing = 0.2 * size
        dist = math.hypot(uv[0] - x0, uv[1] - y0)
        if dist < spacing:
            return None
        covered = None
        steps = int(dist / spacing)
        for i in range(1, steps + 1):
            t = i * spacing / dist
            r = self._dab(ctx, (x0 + (uv[0] - x0) * t, y0 + (uv[1] - y0) * t))
            if r is not None:
                covered = r if covered is None else covered.union(r)
        t = steps * spacing / dist
        self._last = (x0 + (uv[0] - x0) * t, y0 + (uv[1] - y0) * t)
        return covered

    def on_end(self, ctx, event):
        if self._corner is not None:
            self.on_motion(ctx, event)
            self._corner = None
            before, self._marks_before = self._marks_before, None
            if before is not None and ctx.history is not None and not before.same_as(ctx.doc):
                ctx.history.push(before, "Edit Plane")
            return ctx.doc.bounds
        if self._session is None:
            return None
        self.on_motion(ctx, event)
        cmd = self._session.commit()
        self._session = None
        self._snapshot = None
        if cmd is not None and ctx.history is not None:
            ctx.history.push(cmd, self.label, frame_id=ctx.frame.id)
        return cmd

    def on_cancel(self, ctx):
        self._clicks = []
        if self._corner is not None:
            if self._marks_before is not None:
                self._marks_before.undo(ctx.doc)
            self._corner = None
            self._marks_before = None
            return ctx.doc.bounds
        if self._session is None:
            return ctx.doc.bounds
        rect = self._session.cancel()
        self._session = None
        self._snapshot = None
        return rect

    def _dab(self, ctx, uv):
        doc = ctx.doc
        cell = ctx.cell
        res = V.clone_dab(self._snapshot, self.plane(doc), uv, self._offset,
                          max(0.5, float(self.option("size", 40)) / 2.0),
                          float(self.option("hardness", 0.5)), doc.width, doc.height)
        if res is None:
            return None
        x0, y0, patch, cov = res
        rect = Rect(x0, y0, cov.shape[1], cov.shape[0])
        self._session.touch(rect)
        cell.apply_masked(rect, patch, ctx.coverage(cov, rect), name=self._plane)
        return rect

    # ---- what the canvas draws ---------------------------------------------------

    def overlay(self, doc=None):
        out = []
        plane = None if doc is None else self.plane(doc)
        if plane is not None:
            out.append(("poly", [tuple(p) for p in plane], True))
            for seg in V.grid(plane, int(self.option("grid", 8))):
                out.append(("line", seg))
            out.append(("handles", [tuple(p) for p in plane]))
        if self._clicks:
            out.append(("poly", list(self._clicks), False))
            out.append(("handles", list(self._clicks)))
        if self._source is not None:
            x, y = self._source
            out.append(("line", (x - 6, y, x + 6, y)))
            out.append(("line", (x, y - 6, x, y + 6)))
        return out



