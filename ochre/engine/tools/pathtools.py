# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tools that draw and edit vector paths rather than pixels.

  Pen              click for a corner, drag for a smooth point, click the
                   first point to close; Enter or double-click ends an open
                   path. Photoshop's Pen and GIMP's Paths tool in Design mode.
  Curvature Pen    click points and the curve passes smoothly through all of
                   them, handles computed for you; Alt-click a point to make
                   it a corner.
  Freeform Pen     drag freehand; on release the line becomes a few smooth
                   anchors.
  Path Selection   drag a whole subpath (Shift: the whole path; Alt: a copy).
                   Click a path to make it the active one.
  Direct Selection drag single anchors and handles.

Editing works in every one of them the way it does in Photoshop and GIMP:
drag an anchor or handle to move it (Alt breaks a smooth point's handles
apart, Shift snaps a handle to 45 degrees), Ctrl-click a segment to add an
anchor, Ctrl+Shift-click an anchor to delete it, and Alt-click an anchor to
turn it into a corner (Alt-drag pulls new smooth handles out of it).

None of these touch pixels. A path lands on a layer only through Fill Path
and Stroke Path, or becomes a selection through Path to Selection. Each
gesture is one undo step, recorded as a PathDelta.
"""

import math

from .. import paths as P
from ..commands import PathDelta
from . import options as O
from .base import MOD_ALT, MOD_CTRL, MOD_SHIFT, Tool, constrain_angle


def _new_name(doc):
    if not doc.paths:
        return "Work Path"
    names = {p.name for p in doc.paths}
    n = len(doc.paths) + 1
    while "Path %d" % n in names:
        n += 1
    return "Path %d" % n


class _PathTool(Tool):
    """Shared: the active path, per-gesture undo, and anchor/handle editing."""

    wants_stroke = False
    shows_paths = True
    cursor = "crosshair"

    def __init__(self, **options):
        super().__init__(**options)
        self._doc = None
        self._before = None
        self._drag = None
        self._hover = None

    # -- the path ----------------------------------------------------------

    def _path(self, ctx, create=False):
        doc = ctx.doc
        if doc.path is None and create:
            doc.paths.append(P.Path(_new_name(doc)))
            doc.active_path = len(doc.paths) - 1
        return doc.path

    def _snapshot(self, ctx):
        self._doc = ctx.doc
        self._before = PathDelta.capture(ctx.doc, self.label)

    def _record(self, ctx):
        """Push the gesture as one undo step, if it changed anything."""
        before, self._before = self._before, None
        if before is None or ctx.history is None or before.same_as(ctx.doc):
            return False
        ctx.history.push(before, self.label)
        return True

    # -- editing existing anchors (all path tools) --------------------------

    def _edit_press(self, ctx, event, allow_plain=True):
        """Begin an edit if the press lands on something editable."""
        path = ctx.doc.path
        if path is None:
            return False
        r = ctx.pick_radius
        x, y = event.x, event.y
        hit = P.hit_anchor(path, x, y, r)
        if hit is not None:
            si, ai, part = hit
            sub = path.subpaths[si]
            a = sub.anchors[ai]
            if part == "anchor" and event.has(MOD_CTRL) and event.has(MOD_SHIFT):
                del sub.anchors[ai]
                if not sub.anchors:
                    del path.subpaths[si]
                elif len(sub.anchors) < 3:
                    sub.closed = sub.closed and len(sub.anchors) > 2
                self._drag = None
                return True
            if part == "anchor" and event.has(MOD_ALT):
                # Convert point: a click makes it a corner, a drag pulls new
                # smooth handles out of it.
                a.corner()
                self._drag = ("pull", si, ai, (x, y), a.copy())
                return True
            if not allow_plain and not event.has(MOD_CTRL):
                return False
            self._drag = (part, si, ai, (x, y), a.copy())
            return True
        if event.has(MOD_CTRL):
            seg = P.hit_segment(path, x, y, r)
            if seg is not None:
                si, gi, t = seg
                ai = P.insert_anchor(path.subpaths[si], gi, t)
                self._drag = ("anchor", si, ai, (x, y), path.subpaths[si].anchors[ai].copy())
                return True
        return False

    def _edit_motion(self, ctx, event):
        path = ctx.doc.path
        if self._drag is None or path is None:
            return None
        part, si, ai, (sx, sy), orig = self._drag
        if si >= len(path.subpaths) or ai >= len(path.subpaths[si].anchors):
            self._drag = None
            return None
        a = path.subpaths[si].anchors[ai]
        x, y = event.x, event.y
        if part == "anchor":
            dx, dy = x - sx, y - sy
            if event.has(MOD_SHIFT):
                if abs(dx) >= abs(dy):
                    dy = 0.0
                else:
                    dx = 0.0
            a.x, a.y = orig.x + dx, orig.y + dy
            a.ix, a.iy = orig.ix + dx, orig.iy + dy
            a.ox, a.oy = orig.ox + dx, orig.oy + dy
        elif part in ("in", "out", "pull"):
            if event.has(MOD_SHIFT):
                x, y = constrain_angle(a.x, a.y, x, y, 45.0)
            if part == "pull":
                a.set_out(x, y, mirror=True)
                a.smooth = True
            else:
                breaking = event.has(MOD_ALT)
                if breaking:
                    a.smooth = False
                mirror = a.smooth and not breaking
                if part == "in":
                    a.set_in(x, y, mirror=mirror, keep_length=True)
                else:
                    a.set_out(x, y, mirror=mirror, keep_length=True)
        return ctx.doc.bounds

    # -- shared lifecycle ----------------------------------------------------

    def hover(self, ctx, event):
        self._hover = (event.x, event.y)
        return True

    def on_cancel(self, ctx):
        self._drag = None
        if self._before is not None and not self._before.same_as(ctx.doc):
            # Escape mid-drag: put the path back as the press found it.
            self._before.undo(ctx.doc)
        self._before = None
        return ctx.doc.bounds


class PenTool(_PathTool):
    """Click corners, drag smooth points, click the first point to close."""

    name = "pen"
    label = "Pen"
    multi_click = True
    options_spec = (
        O.Opt("polygonal", O.BOOL, False, "Polygonal",
              tip="Straight segments only: dragging does not pull handles."),
    )

    def __init__(self, **options):
        super().__init__(**options)
        self._building = None          # index of the subpath being drawn
        self._mode = None

    def _building_sub(self, ctx):
        path = ctx.doc.path
        if path is None or self._building is None or self._building >= len(path.subpaths):
            self._building = None
            return None
        return path.subpaths[self._building]

    def on_begin(self, ctx, event):
        self._snapshot(ctx)
        self._mode = None
        sub = self._building_sub(ctx)
        if sub is not None and len(sub.anchors) >= 2:
            first = sub.anchors[0]
            if math.hypot(event.x - first.x, event.y - first.y) <= ctx.pick_radius:
                sub.closed = True
                self._mode = ("close", self._building)
                return None
        # Editing: plain drags on anchors when not drawing; Ctrl while drawing.
        if self._edit_press(ctx, event, allow_plain=sub is None):
            self._mode = ("edit",)
            return None
        path = self._path(ctx, create=True)
        if sub is None:
            path.subpaths.append(P.Subpath())
            self._building = len(path.subpaths) - 1
            sub = path.subpaths[self._building]
        sub.anchors.append(P.Anchor(event.x, event.y))
        self._mode = ("new", self._building, len(sub.anchors) - 1)
        return None

    def on_motion(self, ctx, event):
        mode = self._mode
        if mode is None:
            return None
        if mode[0] == "edit":
            return self._edit_motion(ctx, event)
        if self.option("polygonal", False):
            return None
        path = ctx.doc.path
        if path is None:
            return None
        if mode[0] == "new":
            a = path.subpaths[mode[1]].anchors[mode[2]]
        else:
            a = path.subpaths[mode[1]].anchors[0]
        x, y = event.x, event.y
        if math.hypot(x - a.x, y - a.y) < 1.0:
            return None
        if event.has(MOD_SHIFT):
            x, y = constrain_angle(a.x, a.y, x, y, 45.0)
        cusp = event.has(MOD_ALT) and mode[0] == "new"
        a.set_out(x, y, mirror=not cusp)
        a.smooth = not cusp
        return ctx.doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        mode, self._mode = self._mode, None
        self._drag = None
        self._record(ctx)
        if mode is not None and mode[0] == "close":
            self._building = None
        # Stay active while a subpath is being drawn, so the next click
        # continues it and the rubber band follows the pointer.
        self.active = self._building is not None
        return ctx.doc.bounds

    def finish(self, ctx):
        """Enter / double-click: end the open subpath here."""
        sub = self._building_sub(ctx)
        # A double-click delivers a second press first; drop the duplicate
        # anchor it left on top of the last one.
        if sub is not None and len(sub.anchors) >= 2:
            a, b = sub.anchors[-1], sub.anchors[-2]
            if math.hypot(a.x - b.x, a.y - b.y) < 0.5 and not a.has_handles:
                sub.anchors.pop()
        self._building = None
        self.active = False
        return None

    def on_cancel(self, ctx):
        # Escape keeps what has been drawn and just stops drawing.
        if self._mode is not None:
            super().on_cancel(ctx)
        self._mode = None
        self._building = None
        return ctx.doc.bounds

    def overlay(self):
        doc = self._doc
        if doc is None or self._building is None or self._hover is None:
            return []
        path = doc.path
        if path is None or self._building >= len(path.subpaths):
            return []
        sub = path.subpaths[self._building]
        if not sub.anchors or self._mode is not None:
            return []
        a = sub.anchors[-1]
        hx, hy = self._hover
        return [("curve", [((a.x, a.y), (a.ox, a.oy), (hx, hy), (hx, hy))])]


class CurvaturePenTool(PenTool):
    """Click points; the curve passes smoothly through every one of them.

    Handles are computed (Catmull-Rom), so there is nothing to drag out.
    Alt-click a point to make it a corner or smooth again; drag a point to
    move it and the curve reflows; click the first point to close.
    """

    name = "curvature_pen"
    label = "Curvature Pen"
    options_spec = ()

    @staticmethod
    def _reflow(sub):
        pts = [(a.x, a.y) for a in sub.anchors]
        corners = [i for i, a in enumerate(sub.anchors) if not a.smooth]
        fresh = P.catmull_rom(pts, sub.closed, corners)
        for a, f in zip(sub.anchors, fresh):
            a.ix, a.iy, a.ox, a.oy = f.ix, f.iy, f.ox, f.oy

    def on_begin(self, ctx, event):
        self._snapshot(ctx)
        self._mode = None
        path = ctx.doc.path
        sub = self._building_sub(ctx)
        if path is not None:
            hit = P.hit_anchor(path, event.x, event.y, ctx.pick_radius, handles=False)
            if hit is not None:
                si, ai, _part = hit
                target = path.subpaths[si]
                if si == self._building and ai == 0 and len(target.anchors) >= 2 \
                        and not event.has(MOD_ALT):
                    target.closed = True
                    self._reflow(target)
                    self._mode = ("close", si)
                    return None
                if event.has(MOD_ALT):
                    a = target.anchors[ai]
                    a.smooth = not a.smooth
                    if not a.smooth:
                        a.corner()
                    self._reflow(target)
                    self._mode = ("toggle",)
                    return None
                self._mode = ("move", si, ai, (event.x, event.y),
                              (target.anchors[ai].x, target.anchors[ai].y))
                return None
        path = self._path(ctx, create=True)
        if sub is None:
            path.subpaths.append(P.Subpath())
            self._building = len(path.subpaths) - 1
            sub = path.subpaths[self._building]
        sub.anchors.append(P.Anchor(event.x, event.y, smooth=True))
        self._reflow(sub)
        self._mode = ("added",)
        return None

    def on_motion(self, ctx, event):
        mode = self._mode
        if mode is None or mode[0] != "move":
            return None
        path = ctx.doc.path
        _m, si, ai, (sx, sy), (ox, oy) = mode
        sub = path.subpaths[si]
        a = sub.anchors[ai]
        a.x, a.y = ox + event.x - sx, oy + event.y - sy
        self._reflow(sub)
        return ctx.doc.bounds

    def overlay(self):
        doc = self._doc
        if doc is None or self._building is None or self._hover is None or self._mode:
            return []
        path = doc.path
        if path is None or self._building >= len(path.subpaths):
            return []
        sub = path.subpaths[self._building]
        if not sub.anchors:
            return []
        # Preview the curve as it would be with the pointer as the next point.
        trial = P.Subpath([a.copy() for a in sub.anchors] + [P.Anchor(*self._hover, smooth=True)])
        self._reflow(trial)
        segs = trial.segments()[-2:]
        return [("curve", segs)]


class FreeformPenTool(_PathTool):
    """Drag freehand; the line becomes smooth anchors when you let go."""

    name = "freeform_pen"
    label = "Freeform Pen"
    options_spec = (
        O.Opt("fit", O.FLOAT, 2.0, "Curve fit", 0.5, 10.0, 0.5, suffix=" px",
              tip="How far the path may stray from what you drew. Higher "
                  "gives fewer, smoother anchors."),
    )

    def on_begin(self, ctx, event):
        self._snapshot(ctx)
        if self._edit_press(ctx, event, allow_plain=False):
            self._points = None
            return None
        self._points = [(event.x, event.y)]
        return None

    def on_motion(self, ctx, event):
        if self._points is None:
            return self._edit_motion(ctx, event)
        lx, ly = self._points[-1]
        if abs(event.x - lx) >= 0.75 or abs(event.y - ly) >= 0.75:
            self._points.append((event.x, event.y))
        return None

    def on_end(self, ctx, event):
        pts, self._points = self._points, None
        self._drag = None
        if pts is not None and len(pts) >= 2:
            sx, sy = pts[0]
            closed = len(pts) > 8 and math.hypot(event.x - sx, event.y - sy) <= 2 * ctx.pick_radius
            anchors = P.fit_freehand(pts, float(self.option("fit", 2.0)), closed=closed)
            if len(anchors) >= 2:
                path = self._path(ctx, create=True)
                path.subpaths.append(P.Subpath(anchors, closed))
        self._record(ctx)
        return ctx.doc.bounds

    def overlay(self):
        pts = getattr(self, "_points", None)
        return [("poly", list(pts), False)] if pts and len(pts) > 1 else []


def _subpath_at(path, x, y, radius):
    """Index of the subpath under (x, y): on its outline, or inside it."""
    for si, sub in enumerate(path.subpaths):
        single = P.Path("", [sub])
        if P.hit_anchor(single, x, y, radius, handles=False) is not None:
            return si
        if P.hit_segment(single, x, y, radius) is not None:
            return si
    for si in range(len(path.subpaths) - 1, -1, -1):
        sub = path.subpaths[si]
        if sub.closed and P.point_in_path(P.Path("", [sub]), x, y):
            return si
    return None


class PathSelectTool(_PathTool):
    """Move whole subpaths. Shift moves the whole path, Alt drags a copy."""

    name = "path_select"
    label = "Path Selection"

    def on_begin(self, ctx, event):
        self._snapshot(ctx)
        self._move = None
        doc = ctx.doc
        # The active path first, then the others: clicking another path
        # makes it the active one, as in the Paths panel.
        order = ([doc.active_path] if doc.path is not None else []) + \
            [i for i in range(len(doc.paths)) if i != doc.active_path]
        for pi in order:
            si = _subpath_at(doc.paths[pi], event.x, event.y, ctx.pick_radius)
            if si is None:
                continue
            doc.active_path = pi
            path = doc.paths[pi]
            if event.has(MOD_ALT):
                path.subpaths.append(path.subpaths[si].copy())
                si = len(path.subpaths) - 1
            self._move = (None if event.has(MOD_SHIFT) else si, (event.x, event.y))
            return doc.bounds
        return None

    def on_motion(self, ctx, event):
        if self._move is None or ctx.doc.path is None:
            return None
        si, (lx, ly) = self._move
        dx, dy = event.x - lx, event.y - ly
        targets = ctx.doc.path.subpaths if si is None else [ctx.doc.path.subpaths[si]]
        for sub in targets:
            for a in sub.anchors:
                a.move(dx, dy)
        self._move = (si, (event.x, event.y))
        return ctx.doc.bounds

    def on_end(self, ctx, event):
        self.on_motion(ctx, event)
        self._move = None
        self._record(ctx)
        return ctx.doc.bounds


class DirectSelectTool(_PathTool):
    """Move single anchors and handles of the active path."""

    name = "direct_select"
    label = "Direct Selection"

    def on_begin(self, ctx, event):
        self._snapshot(ctx)
        self._edit_press(ctx, event, allow_plain=True)
        return ctx.doc.bounds

    def on_motion(self, ctx, event):
        return self._edit_motion(ctx, event)

    def on_end(self, ctx, event):
        self._edit_motion(ctx, event)
        self._drag = None
        self._record(ctx)
        return ctx.doc.bounds
