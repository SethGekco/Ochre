#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Auto-Align, Auto-Blend, Photomerge and Load Files into Stack.

Run: python3 tests/test_align.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ochre.engine import align
from ochre.engine import warp as W
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def scene(w=900, h=320, seed=0):
    """Something with corners, as a photograph has: shapes on a gradient,
    a little noise, a little blur."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 100 * xx / w, 80 + 80 * yy / h, 120 + 40 * np.sin(xx / 70.0)], -1)
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    for _ in range(160):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(6, 40))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.rectangle([x, y, x + s, y + int(s * rng.uniform(0.4, 1.6))], fill=col)
        else:
            d.ellipse([x, y, x + s, y + s], fill=col)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im).astype(np.float64) + rng.normal(0, 2, (h, w, 3))
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(a, 0, 255)
    out[..., 3] = 255
    return out


def fresh_ctl():
    return Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())


def main():
    big = scene()
    ref = np.ascontiguousarray(big[10:310, 100:500])

    # ---- registration ---------------------------------------------------------------
    true = W.rotate_about(np.radians(4), 200, 150) @ W.translate(15, -9)
    moved = W.warp_homography(ref, true, (0, 0, 400, 300))
    hm, n = align.register(moved, ref)
    pts = np.array([(40, 40), (360, 40), (200, 260), (60, 250)], float)
    err = np.abs(W.apply(hm, W.apply(true, pts)) - pts).max() if hm is not None else 99
    check("register recovers a rotation and shift to within a pixel (%d matches, %.2f px)" % (n, err),
          hm is not None and n >= 20 and err < 1.0)
    hs, ns = align.register(moved, ref, "similarity")
    err = np.abs(W.apply(hs, W.apply(true, pts)) - pts).max() if hs is not None else 99
    check("...and with the similarity model (%.2f px)" % err, hs is not None and err < 1.0)
    other = scene(400, 300, seed=9)
    hx, nx = align.register(other, ref)
    check("two unrelated images do not register (%d matches)" % nx, hx is None or nx < 12)

    # ---- Auto-Align -----------------------------------------------------------------------
    ctl = fresh_ctl()
    ctl.new_document(400, 300, background=None)
    base = ctl.active_layer
    ctl.doc.cell(base).plane("rgba")[...] = ref
    ctl.doc.cell(base)._touched(ctl.doc.bounds)
    lay = ctl.add_layer("shifted")
    ctl.doc.cell(lay).plane("rgba")[...] = moved
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    n0 = len(ctl.history_labels())
    report = ctl.auto_align_layers()
    a = ctl.doc.cell(lay).plane("rgba").astype(int)
    overlap = (a[..., 3] == 255)
    diff = np.abs(a[..., :3] - ref[..., :3].astype(int))[overlap].mean()
    check("Auto-Align Layers puts the moved layer back over the reference (mean diff %.1f)" % diff,
          report.get("shifted", 0) > 0 and diff < 6)
    check("...as one undo step", len(ctl.history_labels()) == n0 + 1)
    ctl.undo()
    check("...undoable", np.array_equal(ctl.doc.cell(lay).plane("rgba"), moved))

    # ---- Auto-Blend: focus stack ------------------------------------------------------------
    from ochre.engine.effects import EffectRegistry
    blurred = EffectRegistry().create("gaussian_blur", radius=5).apply(ref)
    near = ref.copy()
    near[:, 200:] = blurred[:, 200:]              # left half sharp
    far = ref.copy()
    far[:, :200] = blurred[:, :200]               # right half sharp
    ctl = fresh_ctl()
    ctl.new_document(400, 300, background=None)
    ctl.doc.cell(ctl.active_layer).plane("rgba")[...] = near
    ctl.doc.cell(ctl.active_layer)._touched(ctl.doc.bounds)
    l2 = ctl.add_layer("far")
    ctl.doc.cell(l2).plane("rgba")[...] = far
    ctl.doc.cell(l2)._touched(ctl.doc.bounds)
    out_layer = ctl.auto_blend_layers("stack")
    out = ctl.doc.cell(out_layer).plane("rgba")

    def err_to_ref(x):
        return np.abs(x[..., :3].astype(int) - ref[..., :3].astype(int))[:, 20:380].mean()
    check("Auto-Blend (stack) takes the sharp half of each: closer to the all-sharp image "
          "(%.1f vs %.1f / %.1f)" % (err_to_ref(out), err_to_ref(near), err_to_ref(far)),
          err_to_ref(out) < 0.5 * min(err_to_ref(near), err_to_ref(far)))

    # ---- Photomerge, with chaining ------------------------------------------------------------
    crops = [np.ascontiguousarray(big[0 + dy:300 + dy, x:x + 300]) for x, dy in
             ((0, 5), (200, 12), (400, 0), (600, 8))]
    ctl = fresh_ctl()
    doc, report = ctl.photomerge(crops)
    check("Photomerge places every photo, the far-left one through its neighbour (%s)" % report,
          all(v != 0 for v in report.values()) and len(report) == 4)
    check("...on a canvas the size of the scene they cover (%dx%d, expect ~900x312)"
          % (doc.width, doc.height), abs(doc.width - 900) <= 6 and abs(doc.height - 312) <= 6)
    pano = doc.cell(doc.layers()[-1]).plane("rgba")
    region = pano[20:280, 100:800]
    # Where the panorama shows the scene, compare against it at its offset.
    best = min(np.abs(region[..., :3].astype(int) - big[dy + 20:dy + 280, 100 + dx:800 + dx, :3].astype(int)).mean()
               for dx in range(-3, 4) for dy in range(-3, 9) if dy + 20 >= 0 and dy + 280 <= 320)
    check("...and the blended panorama reproduces the scene (mean diff %.1f)" % best, best < 8)
    check("...one layer per photo plus the blend", len(doc.layers()) == 5 and doc.layers()[-1].name == "Panorama")

    # ---- Load Files into Stack ----------------------------------------------------------------
    d = tempfile.mkdtemp()
    files = []
    for k, c in enumerate(crops[:2]):
        p = os.path.join(d, "shot_%d.png" % k)
        Image.fromarray(c).save(p)
        files.append(p)
    ctl = fresh_ctl()
    doc = ctl.load_files_into_stack(files)
    check("Load Files into Stack: one layer per file, named after it",
          [l.name for l in doc.layers()] == ["shot_0", "shot_1"] and (doc.width, doc.height) == (300, 300))
    print("\nall alignment checks passed")


if __name__ == "__main__":
    main()
