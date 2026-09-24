#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Rect and DirtyRegion.

Run: python3 tests/test_geometry.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ochre.engine.geometry import Rect, DirtyRegion


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- construction ----
    r = Rect(10, 20, 30, 40)
    check("rect bounds", (r.x1, r.y1) == (40, 60))
    check("rect area", r.area == 1200)
    check("rect not empty", not r.is_empty)
    check("zero-size rect is empty", Rect(5, 5, 0, 9).is_empty)
    check("negative-size rect is empty", Rect(5, 5, -2, 9).is_empty)
    check("empty rect has zero area", Rect(5, 5, -2, 9).area == 0)

    check("from_bounds", Rect.from_bounds(1, 2, 5, 9) == Rect(1, 2, 4, 7))
    check("from_bounds normalises inverted",
          Rect.from_bounds(5, 9, 1, 2) == Rect(1, 2, 4, 7))
    # A drag from (3,3) to (5,7) must include both endpoints.
    check("from_points is inclusive",
          Rect.from_points(3, 3, 5, 7) == Rect(3, 3, 3, 5))
    check("from_points handles reversed drag",
          Rect.from_points(5, 7, 3, 3) == Rect(3, 3, 3, 5))
    check("from_points single pixel",
          Rect.from_points(4, 4, 4, 4) == Rect(4, 4, 1, 1))

    # ---- union ----
    a, b = Rect(0, 0, 10, 10), Rect(20, 20, 5, 5)
    check("union covers both", a.union(b) == Rect(0, 0, 25, 25))
    check("union with empty is identity", a.union(Rect.empty()) == a)
    check("union with None is identity", a.union(None) == a)
    check("empty union other is other", Rect.empty().union(b) == b)
    check("union is commutative", a.union(b) == b.union(a))

    # ---- intersect ----
    check("intersect overlapping",
          Rect(0, 0, 10, 10).intersect(Rect(5, 5, 10, 10)) == Rect(5, 5, 5, 5))
    check("intersect disjoint is None",
          Rect(0, 0, 4, 4).intersect(Rect(10, 10, 4, 4)) is None)
    # Half-open rects that merely abut do not overlap.
    check("touching edges do not intersect",
          Rect(0, 0, 5, 5).intersect(Rect(5, 0, 5, 5)) is None)
    check("intersect with None is None", a.intersect(None) is None)

    # ---- clipping ----
    check("clip inside canvas", Rect(2, 2, 4, 4).clipped_to(100, 100) == Rect(2, 2, 4, 4))
    check("clip straddling edge",
          Rect(-5, -5, 20, 20).clipped_to(10, 10) == Rect(0, 0, 10, 10))
    check("clip fully outside is None",
          Rect(50, 50, 5, 5).clipped_to(10, 10) is None)

    # ---- inflate / translate / contains ----
    check("inflate grows all sides", Rect(10, 10, 5, 5).inflated(2) == Rect(8, 8, 9, 9))
    check("inflate negative shrinks", Rect(10, 10, 9, 9).inflated(-2) == Rect(12, 12, 5, 5))
    check("over-shrink collapses, not inverts", Rect(10, 10, 4, 4).inflated(-5).is_empty)
    check("translate", Rect(1, 2, 3, 4).translated(10, 20) == Rect(11, 22, 3, 4))
    check("contains inside", r.contains(10, 20))
    check("contains excludes far edge", not r.contains(40, 60))
    check("contains_rect true", Rect(0, 0, 10, 10).contains_rect(Rect(2, 2, 3, 3)))
    check("contains_rect false", not Rect(0, 0, 10, 10).contains_rect(Rect(8, 8, 5, 5)))
    check("contains_rect of empty is true", Rect(0, 0, 10, 10).contains_rect(Rect.empty()))

    # ---- numpy slice shape ----
    rs, cs = Rect(3, 7, 4, 5).slice()
    check("slice is (rows, cols)",
          (rs.start, rs.stop, cs.start, cs.stop) == (7, 12, 3, 7))

    # ---- hashable / immutable ----
    check("rect is hashable", len({Rect(1, 1, 2, 2), Rect(1, 1, 2, 2)}) == 1)
    try:
        r.x = 99
        check("rect is frozen", False, "-- assignment succeeded")
    except AttributeError:
        check("rect is frozen", True)

    # ---- DirtyRegion ----
    d = DirtyRegion(max_rects=4)
    check("new region is clean", not d and len(d) == 0)
    d.add(Rect(0, 0, 10, 10))
    check("add makes it dirty", bool(d) and len(d) == 1)
    d.add(Rect.empty())
    check("empty add is ignored", len(d) == 1)
    d.add(None)
    check("None add is ignored", len(d) == 1)
    # A repeated dab at the same spot must not accumulate rects.
    d.add(Rect(0, 0, 10, 10))
    check("duplicate rect absorbed", len(d) == 1)
    d.add(Rect(2, 2, 3, 3))
    check("contained rect absorbed", len(d) == 1)
    d.add(Rect(0, 0, 20, 20))
    check("containing rect replaces smaller", len(d) == 1 and d.bounds() == Rect(0, 0, 20, 20))

    # Cap enforcement: 6 disjoint rects into a cap of 4.
    d = DirtyRegion(max_rects=4)
    for i in range(6):
        d.add(Rect(i * 100, 0, 10, 10))
    check("region respects max_rects", len(d) == 4, "-- got %d" % len(d))
    bounds = d.bounds()
    check("merging still covers everything",
          bounds.x <= 0 and bounds.x1 >= 510, "-- bounds %r" % (bounds,))

    # Merging must pick the cheapest pair: two near neighbours, one far away.
    d = DirtyRegion(max_rects=2)
    d.add(Rect(0, 0, 10, 10))
    d.add(Rect(12, 0, 10, 10))       # close to the first
    d.add(Rect(5000, 5000, 10, 10))  # far from both
    check("cheapest pair merged, far rect kept separate",
          any(r.area == 10 * 10 for r in d), "-- rects %r" % (list(d),))

    total = d.total_area()
    check("total_area sums rects", total > 0)

    got = d.take()
    check("take returns rects", len(got) == 2)
    check("take leaves region clean", not d and d.bounds() is None)

    d.add(Rect(1, 1, 2, 2))
    d.clear()
    check("clear empties region", not d)

    print("\nall geometry checks passed")


if __name__ == "__main__":
    main()
