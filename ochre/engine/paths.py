# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Vector paths: cubic Bezier outlines that live beside the pixels.

A path is what the Pen tool draws and what GIMP's Paths dialog and
Photoshop's Paths panel list. It is not paint: it is a resolution-free
outline that can later be FILLED, STROKED, or turned into a SELECTION, and a
selection can be turned back into one. Keeping it separate from the pixels
is the whole point -- a path can be reshaped indefinitely and only lands on
a layer when asked.

Model, kept deliberately plain so it serialises as text and copies cheaply:

  Path      a name and a list of subpaths
  Subpath   a list of anchors and a closed flag
  Anchor    a point with an IN handle and an OUT handle, in absolute
            document coordinates. A handle equal to its anchor means "no
            handle": the segment leaves that anchor straight.

A segment from anchor A to anchor B is the cubic Bezier
(A, A.out, B.in, B). Filling uses the even-odd rule over every subpath, the
same as the selection engine's polygon rule, so a path with a hole in it
fills with a hole.
"""

import math

import numpy as np
from PIL import Image, ImageDraw


class Anchor:
    __slots__ = ("x", "y", "ix", "iy", "ox", "oy", "smooth")

    def __init__(self, x, y, ix=None, iy=None, ox=None, oy=None, smooth=False):
        self.x, self.y = float(x), float(y)
        self.ix = self.x if ix is None else float(ix)
        self.iy = self.y if iy is None else float(iy)
        self.ox = self.x if ox is None else float(ox)
        self.oy = self.y if oy is None else float(oy)
        # A smooth anchor keeps its handles collinear when one is dragged.
        self.smooth = bool(smooth)

    def copy(self):
        return Anchor(self.x, self.y, self.ix, self.iy, self.ox, self.oy, self.smooth)

    @property
    def has_handles(self):
        return (self.ix, self.iy) != (self.x, self.y) or (self.ox, self.oy) != (self.x, self.y)

    def move(self, dx, dy):
        """Move the anchor and both its handles together."""
        self.x += dx
        self.y += dy
        self.ix += dx
        self.iy += dy
        self.ox += dx
        self.oy += dy

    def set_out(self, x, y, mirror=True, keep_length=False):
        """Place the OUT handle; mirror keeps the IN handle collinear.

        keep_length leaves the other handle's length alone and only turns it
        -- Photoshop's behaviour when reshaping a smooth point -- otherwise
        it becomes the exact reflection, which is what dragging out a new
        anchor wants.
        """
        self.ox, self.oy = float(x), float(y)
        if mirror:
            self.ix, self.iy = self._opposite(self.ox, self.oy, self.ix, self.iy, keep_length)

    def set_in(self, x, y, mirror=True, keep_length=False):
        self.ix, self.iy = float(x), float(y)
        if mirror:
            self.ox, self.oy = self._opposite(self.ix, self.iy, self.ox, self.oy, keep_length)

    def _opposite(self, hx, hy, ox, oy, keep_length):
        dx, dy = hx - self.x, hy - self.y
        if not keep_length:
            return self.x - dx, self.y - dy
        length = math.hypot(ox - self.x, oy - self.y)
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return ox, oy
        if length < 1e-9:
            return self.x - dx, self.y - dy
        return self.x - dx / d * length, self.y - dy / d * length

    def corner(self):
        """Retract both handles: a sharp corner."""
        self.ix, self.iy = self.x, self.y
        self.ox, self.oy = self.x, self.y
        self.smooth = False

    def __eq__(self, other):
        return isinstance(other, Anchor) and all(
            abs(getattr(self, k) - getattr(other, k)) < 1e-9
            for k in ("x", "y", "ix", "iy", "ox", "oy"))

    def __repr__(self):
        return "Anchor(%.1f, %.1f)" % (self.x, self.y)


class Subpath:
    __slots__ = ("anchors", "closed")

    def __init__(self, anchors=None, closed=False):
        self.anchors = list(anchors or [])
        self.closed = bool(closed)

    def copy(self):
        return Subpath([a.copy() for a in self.anchors], self.closed)

    def segments(self):
        """Every cubic as (p0, c0, c1, p1), including the closing one."""
        a = self.anchors
        n = len(a)
        out = []
        for i in range(n - 1 + (1 if self.closed and n > 1 else 0)):
            s, e = a[i], a[(i + 1) % n]
            out.append(((s.x, s.y), (s.ox, s.oy), (e.ix, e.iy), (e.x, e.y)))
        return out

    def __len__(self):
        return len(self.anchors)


class Path:
    __slots__ = ("name", "subpaths")

    def __init__(self, name="Path", subpaths=None):
        self.name = name
        self.subpaths = list(subpaths or [])

    def copy(self):
        return Path(self.name, [s.copy() for s in self.subpaths])

    @property
    def is_empty(self):
        return not any(len(s) for s in self.subpaths)

    def anchors(self):
        for s in self.subpaths:
            for a in s.anchors:
                yield a

    def translate(self, dx, dy):
        for a in self.anchors():
            a.move(dx, dy)
        return self

    def map_points(self, fn):
        """Apply fn((x, y)) -> (x, y) to every anchor and handle."""
        for a in self.anchors():
            a.x, a.y = fn((a.x, a.y))
            a.ix, a.iy = fn((a.ix, a.iy))
            a.ox, a.oy = fn((a.ox, a.oy))
        return self

    def bounds(self):
        """(x0, y0, x1, y1) of the flattened outline, or None if empty."""
        pts = [p for line, _c in flatten(self) for p in line]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def __repr__(self):
        return "Path(%r, %d subpaths, %d anchors)" % (
            self.name, len(self.subpaths), sum(len(s) for s in self.subpaths))


# ---- Bezier maths ---------------------------------------------------------------

def bezier_point(p0, c0, c1, p1, t):
    u = 1.0 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (a * p0[0] + b * c0[0] + c * c1[0] + d * p1[0],
            a * p0[1] + b * c0[1] + c * c1[1] + d * p1[1])


def split_cubic(p0, c0, c1, p1, t):
    """de Casteljau: the two halves of a cubic at t."""
    def lerp(a, b):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    a, b, c = lerp(p0, c0), lerp(c0, c1), lerp(c1, p1)
    d, e = lerp(a, b), lerp(b, c)
    m = lerp(d, e)
    return (p0, a, d, m), (m, e, c, p1)


def _flat_enough(p0, c0, c1, p1, tol):
    # Distance of the control points from the chord, squared, against tol^2.
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d2 = dx * dx + dy * dy
    if d2 < 1e-12:
        return (math.hypot(c0[0] - p0[0], c0[1] - p0[1]) <= tol
                and math.hypot(c1[0] - p0[0], c1[1] - p0[1]) <= tol)
    for cx, cy in (c0, c1):
        cross = (cx - p0[0]) * dy - (cy - p0[1]) * dx
        if cross * cross > tol * tol * d2:
            return False
    return True


def flatten_cubic(p0, c0, c1, p1, tol=0.2, out=None):
    """Points along a cubic (excluding p0), within tol pixels of the curve."""
    out = [] if out is None else out
    stack = [(p0, c0, c1, p1, 0)]
    while stack:
        q0, q1, q2, q3, depth = stack.pop()
        if depth >= 16 or _flat_enough(q0, q1, q2, q3, tol):
            out.append(q3)
            continue
        left, right = split_cubic(q0, q1, q2, q3, 0.5)
        stack.append(right + (depth + 1,))
        stack.append(left + (depth + 1,))
    return out


def flatten_subpath(sub, tol=0.2):
    if not sub.anchors:
        return []
    a0 = sub.anchors[0]
    pts = [(a0.x, a0.y)]
    for seg in sub.segments():
        flatten_cubic(*seg, tol=tol, out=pts)
    return pts


def flatten(path, tol=0.2):
    """[(points, closed), ...] for every non-empty subpath."""
    return [(flatten_subpath(s, tol), s.closed) for s in path.subpaths if s.anchors]


# ---- rasterising -------------------------------------------------------------------

def _supersample(w, h, antialias):
    if not antialias:
        return 1
    area = w * h
    return 4 if area <= 512 * 512 else 3 if area <= 1024 * 1024 else 2 if area <= 2048 * 2048 else 1


def _box(lines, width, height, pad):
    xs = [p[0] for line in lines for p in line]
    ys = [p[1] for line in lines for p in line]
    if not xs:
        return None
    x0 = max(0, int(math.floor(min(xs))) - pad)
    y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(width, int(math.ceil(max(xs))) + pad)
    y1 = min(height, int(math.ceil(max(ys))) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def scan_fill(rings, x0, y0, w, h, ss=1):
    """Even-odd coverage of closed rings over a w x h window at (x0, y0).

    Exact scan conversion: a (sub)pixel is inside when its CENTRE is, which
    is what makes a selection traced to a path and filled again come back
    pixel-identical. (Pillow's polygon fill rounds vertices and includes both
    boundary edges, so a 50-pixel rectangle comes out 51 wide.) Every edge's
    crossings with every scanline are computed at once, then a running
    parity along each row fills the spans. Returns a bool (h*ss, w*ss) array.
    """
    rows, cols = h * ss, w * ss
    counts = np.zeros(rows * (cols + 1), dtype=np.int64)
    for ring in rings:
        p = np.asarray(ring, dtype=np.float64)
        if len(p) < 3:
            continue
        ax = (p[:, 0] - x0) * ss - 0.5
        ay = (p[:, 1] - y0) * ss - 0.5
        bx, by = np.roll(ax, -1), np.roll(ay, -1)
        keep = ay != by
        ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
        # Half-open in y: an edge owns the scanlines in [min, max).
        j0 = np.clip(np.ceil(np.minimum(ay, by)), 0, rows).astype(np.int64)
        j1 = np.clip(np.ceil(np.maximum(ay, by)), 0, rows).astype(np.int64)
        n = np.maximum(j1 - j0, 0)
        total = int(n.sum())
        if total == 0:
            continue
        idx = np.repeat(np.arange(len(n)), n)
        start = np.cumsum(n) - n
        j = j0[idx] + (np.arange(total) - start[idx])
        x = ax[idx] + (j - ay[idx]) * ((bx - ax) / (by - ay))[idx]
        c = np.clip(np.ceil(x), 0, cols).astype(np.int64)
        counts += np.bincount(j * (cols + 1) + c, minlength=counts.size)
    parity = np.cumsum(counts.reshape(rows, cols + 1), axis=1)[:, :cols]
    return (parity & 1).astype(bool)


def fill_mask(path, width, height, antialias=True):
    """Full-canvas uint8 coverage of the path's interior, even-odd.

    Open subpaths fill as if closed, which is what both GIMP and Photoshop
    do when asked to fill or select an open path.
    """
    out = np.zeros((height, width), dtype=np.uint8)
    rings = [pts for pts, _c in flatten(path) if len(pts) >= 3]
    box = _box(rings, width, height, 2)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    ss = _supersample(x1 - x0, y1 - y0, antialias)
    acc = scan_fill(rings, x0, y0, x1 - x0, y1 - y0, ss)
    out[y0:y1, x0:x1] = _resolve(acc, ss, y1 - y0, x1 - x0)
    return out


def stroke_mask(path, width, height, line_width=2.0, antialias=True):
    """Full-canvas uint8 coverage of the path's outline, round joins and caps."""
    out = np.zeros((height, width), dtype=np.uint8)
    lines = [(pts, closed) for pts, closed in flatten(path) if len(pts) >= 2]
    pad = int(math.ceil(line_width / 2.0)) + 2
    box = _box([p for p, _c in lines], width, height, pad)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    ss = _supersample(x1 - x0, y1 - y0, antialias)
    W, H = (x1 - x0) * ss, (y1 - y0) * ss
    img = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(img)
    wpx = max(1, int(round(line_width * ss)))
    rad = wpx / 2.0
    for pts, closed in lines:
        # Pillow puts pixel CENTRES on integers; Ochre puts corners there.
        local = [((x - x0) * ss - 0.5, (y - y0) * ss - 0.5) for x, y in pts]
        if closed:
            local = local + local[:2]
        draw.line(local, fill=1, width=wpx, joint="curve")
        if wpx > 2:
            # Round caps (and a guarantee at sharp joins).
            ends = local if len(local) < 64 else [local[0], local[-1]]
            for x, y in ends:
                draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=1)
    acc = np.asarray(img, dtype=bool)
    out[y0:y1, x0:x1] = _resolve(acc, ss, y1 - y0, x1 - x0)
    return out


def _resolve(acc, ss, h, w):
    if ss == 1:
        return acc.astype(np.uint8) * 255
    return np.rint(acc.reshape(h, ss, w, ss).mean(axis=(1, 3)) * 255.0).astype(np.uint8)


# ---- selection to path ------------------------------------------------------------

def _trace_loops(inside):
    """Closed boundary loops of a boolean mask, on pixel corners.

    Each boundary edge is directed with the inside on its right, so every
    loop is closed and outer boundaries and holes come out with opposite
    windings -- which the even-odd fill does not need, but which keeps the
    traced outline faithful at pinch points.
    """
    p = np.pad(inside, 1, constant_values=False)
    core = p[1:-1, 1:-1]
    edges = {}
    ys, xs = np.nonzero(core & ~p[:-2, 1:-1])           # top
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x, y), []).append((x + 1, y))
    ys, xs = np.nonzero(core & ~p[1:-1, 2:])            # right
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x + 1, y), []).append((x + 1, y + 1))
    ys, xs = np.nonzero(core & ~p[2:, 1:-1])            # bottom
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x + 1, y + 1), []).append((x, y + 1))
    ys, xs = np.nonzero(core & ~p[1:-1, :-2])           # left
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x, y + 1), []).append((x, y))
    loops = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        cur = start
        while True:
            nxt = edges[cur].pop()
            if not edges[cur]:
                del edges[cur]
            if nxt == start:
                break
            loop.append(nxt)
            cur = nxt
            if cur not in edges:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def simplify(points, tolerance=1.0, closed=True):
    """Ramer-Douglas-Peucker. Keeps the shape within `tolerance` pixels."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return [tuple(p) for p in pts]
    if closed:
        # Split the ring at its two most distant points and simplify halves.
        far = int(np.argmax(((pts - pts[0]) ** 2).sum(axis=1)))
        a = simplify(np.vstack([pts[:far + 1]]), tolerance, closed=False)
        b = simplify(np.vstack([pts[far:], pts[:1]]), tolerance, closed=False)
        return a[:-1] + b[:-1]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = pts[i + 1:j]
        d = b - a
        L = math.hypot(d[0], d[1])
        if L < 1e-12:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs((seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return [tuple(p) for p in pts[keep]]


def from_mask(mask, threshold=128, tolerance=0.5, name="Selection", smooth=False):
    """Selection to path: the mask's outline as a path of corner anchors.

    `tolerance` is how far, in pixels, the path may stray from the traced
    pixel edges. At the default 0.5 every axis-aligned feature -- a single
    pixel, a one-pixel line -- survives and fills back pixel-identical, while
    a diagonal staircase still collapses to one straight edge instead of an
    anchor per step. smooth=True gives every anchor Catmull-Rom handles, for
    organic shapes.
    """
    inside = np.asarray(mask) >= threshold
    path = Path(name)
    for loop in _trace_loops(inside):
        pts = simplify(loop, tolerance, closed=True)
        if len(pts) < 3:
            continue
        if smooth:
            anchors = catmull_rom(pts, closed=True)
        else:
            anchors = [Anchor(x, y) for x, y in pts]
        path.subpaths.append(Subpath(anchors, closed=True))
    return path


# ---- curve construction ---------------------------------------------------------

def catmull_rom(points, closed=False, corners=(), tension=1.0):
    """Anchors whose curve passes smoothly THROUGH every point.

    The Curvature Pen's rule: a smooth point's handles lie along the line
    through its neighbours, a sixth of that span each way (scaled by
    tension). Indices in `corners` get no handles.
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    corners = set(corners)
    out = []
    k = tension / 6.0
    for i, (x, y) in enumerate(pts):
        if i in corners or n < 2:
            out.append(Anchor(x, y))
            continue
        if closed:
            px, py = pts[(i - 1) % n]
            nx, ny = pts[(i + 1) % n]
        elif i == 0:
            # An open end has one neighbour: reflect it to make the other.
            nx, ny = pts[1]
            px, py = 2 * x - nx, 2 * y - ny
        elif i == n - 1:
            px, py = pts[i - 1]
            nx, ny = 2 * x - px, 2 * y - py
        else:
            px, py = pts[i - 1]
            nx, ny = pts[i + 1]
        tx, ty = (nx - px) * k, (ny - py) * k
        a = Anchor(x, y, x - tx, y - ty, x + tx, y + ty, smooth=True)
        if not closed and i == 0:
            a.ix, a.iy = x, y
        if not closed and i == n - 1:
            a.ox, a.oy = x, y
        out.append(a)
    return out


def fit_freehand(points, tolerance=2.0, closed=False):
    """A freehand drag as a few smooth anchors.

    Simplify to the points that matter, then pass a Catmull-Rom curve
    through them. Not a least-squares Bezier fit (Schneider's algorithm),
    which would use fewer anchors for the same accuracy -- but every anchor
    here lies exactly on what was drawn, which is what someone editing the
    result afterwards expects.
    """
    pts = simplify(points, tolerance, closed=closed)
    if len(pts) < 2:
        return []
    # A sharp turn in the drawn line stays sharp.
    corners = []
    for i in range(1, len(pts) - 1):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[i + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        if l1 > 1e-9 and l2 > 1e-9:
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            if cos < -0.1:                       # turned more than ~95 degrees
                corners.append(i)
    return catmull_rom(pts, closed=closed, corners=corners)


# ---- hit testing -------------------------------------------------------------------

def hit_anchor(path, x, y, radius, handles=True):
    """(subpath, anchor, part) nearest within radius, part in anchor/in/out."""
    best, found = radius * radius, None
    for si, sub in enumerate(path.subpaths):
        for ai, a in enumerate(sub.anchors):
            cands = [("anchor", a.x, a.y)]
            if handles:
                if (a.ix, a.iy) != (a.x, a.y):
                    cands.append(("in", a.ix, a.iy))
                if (a.ox, a.oy) != (a.x, a.y):
                    cands.append(("out", a.ox, a.oy))
            for part, px, py in cands:
                d = (px - x) ** 2 + (py - y) ** 2
                # Anchors win ties against their own handles.
                if d < best or (d == best and part == "anchor"):
                    best, found = d, (si, ai, part)
    return found


def hit_segment(path, x, y, radius):
    """(subpath, segment index, t) of the nearest segment within radius."""
    best, found = radius * radius, None
    for si, sub in enumerate(path.subpaths):
        for gi, seg in enumerate(sub.segments()):
            # Sample, then refine around the best sample.
            ts = np.linspace(0.0, 1.0, 33)
            pts = np.array([bezier_point(*seg, t) for t in ts])
            d = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
            k = int(np.argmin(d))
            lo, hi = ts[max(0, k - 1)], ts[min(32, k + 1)]
            for _ in range(12):
                m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
                p1, p2 = bezier_point(*seg, m1), bezier_point(*seg, m2)
                if (p1[0] - x) ** 2 + (p1[1] - y) ** 2 < (p2[0] - x) ** 2 + (p2[1] - y) ** 2:
                    hi = m2
                else:
                    lo = m1
            t = (lo + hi) / 2.0
            p = bezier_point(*seg, t)
            dd = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if dd <= best:
                best, found = dd, (si, gi, t)
    return found


def insert_anchor(sub, segment, t):
    """Split a segment at t with a new anchor; the curve's shape is unchanged."""
    n = len(sub.anchors)
    a, b = sub.anchors[segment], sub.anchors[(segment + 1) % n]
    left, right = split_cubic((a.x, a.y), (a.ox, a.oy), (b.ix, b.iy), (b.x, b.y), t)
    a.ox, a.oy = left[1]
    mid = Anchor(left[3][0], left[3][1], left[2][0], left[2][1], right[1][0], right[1][1],
                 smooth=True)
    b.ix, b.iy = right[2]
    sub.anchors.insert(segment + 1, mid)
    return segment + 1


def point_in_path(path, x, y):
    """Even-odd containment against the flattened outline."""
    inside = False
    for pts, _closed in flatten(path, 0.5):
        n = len(pts)
        if n < 3:
            continue
        for i in range(n):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if xi > x:
                    inside = not inside
    return inside


# ---- text serialisation (the .ochre manifest) ---------------------------------

def encode_subpath(sub):
    """'closed;x,y,ix,iy,ox,oy,s;...' -- one INI value per subpath."""
    parts = ["1" if sub.closed else "0"]
    for a in sub.anchors:
        parts.append(",".join("%.10g" % v for v in (a.x, a.y, a.ix, a.iy, a.ox, a.oy))
                     + (",1" if a.smooth else ",0"))
    return ";".join(parts)


def decode_subpath(text):
    fields = [f for f in str(text).split(";") if f.strip()]
    if not fields:
        return Subpath()
    sub = Subpath(closed=fields[0].strip() == "1")
    for f in fields[1:]:
        v = [float(t) for t in f.split(",")]
        if len(v) < 6:
            continue
        sub.anchors.append(Anchor(*v[:6], smooth=len(v) > 6 and v[6] >= 0.5))
    return sub
