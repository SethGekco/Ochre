#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Spin Blur and Path Blur as Kooka has them: several spins, the strobe's
flash duration, curved and several paths, the Blur Gallery's Noise pass,
and the Blur Gallery tool's Spin and Path modes on the canvas.

Run: python3 tests/test_spinpath.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import galleryextra as GE
from ochre.engine.effects import EffectRegistry
from ochre.engine.settings import Settings
from ochre.engine.tools.base import MOD_ALT, MOD_SHIFT
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry()


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.kron(rng.integers(30, 220, (h // 4 + 1, w // 4 + 1, 3)),
                           np.ones((4, 4, 1)))[:h, :w]
    out[..., 3] = 255
    return out


def stripes(w, h, horizontal=True):
    """Black and white bands 3 px wide: a blur along them leaves them be,
    a blur across them greys them."""
    out = np.zeros((h, w, 4), np.uint8)
    idx = np.arange(h if horizontal else w) // 3 % 2 * 255
    if horizontal:
        out[..., :3] = idx[:, None, None]
    else:
        out[..., :3] = idx[None, :, None]
    out[..., 3] = 255
    return out


def change(a, b, ys, xs):
    return np.abs(a[ys, xs].astype(int) - b[ys, xs]).mean()


def main():
    tex = texture(96, 64)

    # ---- Spin Blur ---------------------------------------------------------------------
    one = [0.3, 0.5, 0.15, 0.15, 0.0, 0.5, 45.0, 0.0, 0.0]
    slider = REG.create("spin_blur", center_x=30, center_y=50, radius_x=0.15, radius_y=0.15,
                        blur_angle=45).apply(tex)
    check("one spin as JSON is the sliders' spin",
          np.array_equal(REG.create("spin_blur", spins=json.dumps([one])).apply(tex), slider))
    two = [one, [0.75, 0.5, 0.12, 0.2, 30.0, 0.3, 90.0, 0.0, 0.0]]
    both = REG.create("spin_blur", spins=json.dumps(two)).apply(tex)
    check("two spins blur both places",
          change(both, tex, slice(26, 38), slice(20, 38)) > 5
          and change(both, tex, slice(26, 38), slice(66, 78)) > 5)
    check("...and leave the rest alone", np.array_equal(both[:, 48:56], tex[:, 48:56]))
    check("...the first spin's area is the single spin's result",
          np.array_equal(both[:, :45], slider[:, :45]))
    check("an empty spin list changes nothing",
          np.array_equal(REG.create("spin_blur", spins="[]").apply(tex), tex))
    check("unparsable spins fall back to the sliders",
          np.array_equal(REG.create("spin_blur", spins="{oops", center_x=30, center_y=50,
                                    radius_x=0.15, radius_y=0.15, blur_angle=45).apply(tex),
                         slider))
    big = dict(radius_x=0.6, radius_y=0.6, blur_angle=120)
    smooth = REG.create("spin_blur", **big).apply(tex)
    strobe = REG.create("spin_blur", strobe_strength=100, strobe_flashes=3, **big).apply(tex)
    longer = REG.create("spin_blur", strobe_strength=100, strobe_flashes=3,
                        strobe_duration=40, **big).apply(tex)
    check("the strobe changes the spin (flashes instead of a smear)",
          not np.array_equal(strobe, smooth))
    check("...and a longer flash duration smears each flash",
          not np.array_equal(longer, strobe)
          and change(longer, smooth, slice(None), slice(None))
          < change(strobe, smooth, slice(None), slice(None)))
    # One tap lit, all the weight on it: the flash is the tap nearest the
    # arc's middle, within half a tap of the pixel itself -- nearly sharp.
    single = REG.create("spin_blur", strobe_strength=100, strobe_flashes=1,
                        strobe_duration=0, **big).apply(tex)
    check("one instant flash at full strength is all but sharp",
          change(single, tex, slice(None), slice(None))
          < change(smooth, tex, slice(None), slice(None)) / 4)

    # ---- Noise ---------------------------------------------------------------------------
    base = REG.create("spin_blur", spins=json.dumps(two)).apply(tex)
    noisy = REG.create("spin_blur", spins=json.dumps(two), noise_amount=40).apply(tex)
    # Outside both ellipses: the top and bottom bands and the gap between.
    sharp = np.zeros(tex.shape[:2], bool)
    sharp[:10], sharp[54:], sharp[:, 46:58] = True, True, True
    check("Noise adds grain to the blurred pixels", change(noisy, base, slice(None), slice(None)) > 1)
    check("...and never to the sharp ones", np.array_equal(noisy[sharp], tex[sharp]))
    check("...alpha untouched", np.array_equal(noisy[..., 3], base[..., 3]))
    reseed = REG.create("spin_blur", spins=json.dumps(two), noise_amount=40,
                        noise_seed=9).apply(tex)
    check("...the seed re-rolls it", not np.array_equal(reseed, noisy))
    kinds = [REG.create("spin_blur", spins=json.dumps(two), noise_amount=40,
                        noise_kind=k).apply(tex) for k in GE.NOISE_KINDS]
    check("...Gaussian, Uniform and Grain differ",
          not np.array_equal(kinds[0], kinds[1]) and not np.array_equal(kinds[1], kinds[2]))
    grained = (kinds[0] != base).any(-1)
    mono = kinds[0][grained][:, :3].astype(int) - base[grained][:, :3]
    colored = REG.create("spin_blur", spins=json.dumps(two), noise_amount=40,
                         noise_color=100).apply(tex)[grained][:, :3].astype(int) \
        - base[grained][:, :3]
    unclipped = (base[grained][:, :3] > 60).all(1) & (base[grained][:, :3] < 195).all(1)
    check("...monochrome by default, colored at Color 100",
          (mono[unclipped, 0] == mono[unclipped, 1]).mean() > 0.95
          and (colored[unclipped, 0] == colored[unclipped, 1]).mean() < 0.5)
    hi = REG.create("spin_blur", spins=json.dumps(two), noise_amount=40,
                    noise_highlights=100).apply(tex)
    check("...Highlights damps it", change(hi, base, slice(None), slice(None))
          < change(noisy, base, slice(None), slice(None)))
    flat = REG.create("path_blur", noise_amount=60).apply(tex)
    check("Path Blur takes the Noise pass too",
          not np.array_equal(flat, REG.create("path_blur").apply(tex)))

    # ---- Path Blur ------------------------------------------------------------------------
    hs = stripes(96, 64, horizontal=True)
    along = [{"points": [[0.1, 0.5], [0.9, 0.5]], "start_speed": 20, "end_speed": 20}]
    across = [{"points": [[0.5, 0.1], [0.5, 0.9]], "start_speed": 20, "end_speed": 20}]
    a = REG.create("path_blur", speed=100, paths=json.dumps(along)).apply(hs)
    b = REG.create("path_blur", speed=100, paths=json.dumps(across)).apply(hs)
    check("a path along the stripes leaves them, one across them greys them",
          change(a, hs, slice(10, 54), slice(20, 76)) < 1
          and change(b, hs, slice(10, 54), slice(20, 76)) > 40)
    check("one straight path as JSON is the sliders' path",
          np.array_equal(REG.create("path_blur", start_x=10, start_y=50, end_x=90, end_y=50,
                                    start_speed=20, end_speed=20, speed=100).apply(hs), a))
    # Two paths: along the stripes at the top, across them at the bottom.
    mixed = [{"points": [[0.1, 0.05], [0.9, 0.05]], "start_speed": 12, "end_speed": 12},
             {"points": [[0.5, 0.95], [0.5, 0.7]], "start_speed": 12, "end_speed": 12}]
    m = REG.create("path_blur", speed=100, paths=json.dumps(mixed)).apply(hs)
    check("two paths: each region follows its nearer path",
          change(m, hs, slice(0, 10), slice(20, 76)) < 2
          and change(m, hs, slice(52, 64), slice(30, 66)) > 30)
    # A curve: running along the stripes at its start, then bending down
    # across them.
    bend = [{"points": [[0.05, 0.2], [0.55, 0.2], [0.85, 0.5], [0.9, 0.95]],
             "start_speed": 12, "end_speed": 12}]
    c = REG.create("path_blur", speed=100, paths=json.dumps(bend)).apply(hs)
    lead = change(c, hs, slice(8, 18), slice(4, 30))
    turn = change(c, hs, slice(40, 60), slice(80, 92))
    check("a curved path: nearly sharp where it runs along the stripes, blurred where it "
          "turns across them (%.1f vs %.1f)" % (lead, turn), lead < 15 and turn > 60)
    samples = GE.catmull_rom([[0, 0], [10, 0], [10, 10]])
    check("the curve goes through every point",
          np.allclose(samples[0], [0, 0]) and np.allclose(samples[16], [10, 0])
          and np.allclose(samples[-1], [10, 10]) and len(samples) == 33)
    # The trail bends: a ring path, a dot on it, the dot smeared along the ring.
    dot = np.zeros((64, 64, 4), np.uint8)
    dot[..., 3] = 255
    dot[10:13, 31:34, :3] = 255
    ring = [[0.5 + 0.34 * np.cos(t), 0.5 + 0.34 * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, 25)]
    r = REG.create("path_blur", speed=100, paths=json.dumps(
        [{"points": ring, "start_speed": 30, "end_speed": 30}])).apply(dot)
    lit = np.argwhere(r[..., 0] > 2)
    radii = np.hypot(lit[:, 0] + 0.5 - 32, lit[:, 1] + 0.5 - 32)
    check("...and a trail bends with the path (a dot smeared along a ring stays on it)",
          len(lit) > 12 and radii.min() > 17 and radii.max() < 25, "%s" % (radii.min(),))
    check("an empty path list changes nothing",
          np.array_equal(REG.create("path_blur", paths="[]").apply(hs), hs))
    check("strobe and rear sync with every flash count render",
          all(REG.create("path_blur", speed=100, paths=json.dumps(across), strobe_strength=70,
                         strobe_flashes=n, rear_sync=True).apply(hs).shape == hs.shape
              for n in (1, 2, 5, 100)))

    # ---- the Blur Gallery tool's Spin and Path modes ---------------------------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(96, 64, background=None)
    lay = ctl.active_layer
    cell = ctl.doc.cell(lay)
    cell.plane("rgba")[...] = tex
    cell._touched(ctl.doc.bounds)
    ctl.set_tool("blur_gallery", mode="spin", blur_angle=40, feather=50)
    t = ctl.tool
    ctl.begin_stroke(28, 32)
    ctl.motion_stroke(40, 32)
    ctl.end_stroke(44, 32)
    ctl.begin_stroke(72, 32)
    ctl.end_stroke(72, 32)
    check("Spin: clicks drop spins, a drag sizes one",
          len(t.spins) == 2 and t.spins[0][2] == 16.0 and t.spins[1][2] == 0.13 * 96)
    ctl.set_tool_option("blur_angle", 90)
    check("...Blur Angle sets the selected spin", [s[6] for s in t.spins] == [40.0, 90.0])
    live = cell.plane("rgba").copy()
    want = GE.spin_blur(tex, t.spins_fractions(96, 64))
    check("...the layer previews both spins", np.array_equal(live, want)
          and change(live, tex, slice(26, 38), slice(20, 36)) > 5)
    ctl.begin_stroke(72, 32)
    ctl.motion_stroke(70, 40)
    ctl.end_stroke(70, 40)
    check("...a spin can be dragged by its centre", t.spins[1][:2] == [70.0, 40.0])
    ctl.begin_stroke(70, 40, modifiers=MOD_ALT)
    ctl.end_stroke(70, 40, modifiers=MOD_ALT)
    check("...Alt-click removes it", len(t.spins) == 1)
    check("...and the overlay draws it", any(k == "poly" for k, *_ in t.overlay()))
    pos = ctl.history.position
    live = cell.plane("rgba").copy()
    ctl.finish_tool()
    check("...Enter applies it as one undo step", ctl.history.position == pos + 1
          and np.array_equal(cell.plane("rgba"), live))
    ctl.undo()
    check("...undoable", np.array_equal(cell.plane("rgba"), tex))

    cell.plane("rgba")[...] = hs = stripes(96, 64)
    cell._touched(ctl.doc.bounds)
    ctl.set_tool("blur_gallery", mode="path", blur=20)
    t = ctl.tool
    for x, y in ((48, 4), (48, 30)):
        ctl.begin_stroke(x, y)
        ctl.end_stroke(x, y)
    check("Path: clicks pin stop points on one path", len(t.paths) == 1
          and t.paths[0]["points"] == [[48.0, 4.0], [48.0, 30.0]])
    ctl.begin_stroke(48, 30)
    ctl.motion_stroke(48, 60)
    ctl.end_stroke(48, 60)
    check("...a point can be dragged", t.paths[0]["points"][1] == [48.0, 60.0])
    live = cell.plane("rgba").copy()
    check("...and the layer previews the path (across the stripes: greyed)",
          change(live, hs, slice(10, 54), slice(20, 76)) > 30)
    ctl.begin_stroke(10, 10, modifiers=MOD_SHIFT)
    ctl.end_stroke(10, 10, modifiers=MOD_SHIFT)
    ctl.begin_stroke(20, 12)
    ctl.end_stroke(20, 12)
    check("...Shift-click starts another path, later clicks extend it",
          len(t.paths) == 2 and len(t.paths[1]["points"]) == 2)
    ctl.set_tool_option("blur", 5)
    check("...Amount sets the selected path's speed",
          [p["speed"] for p in t.paths] == [20.0, 5.0])
    ctl.begin_stroke(20, 12, modifiers=MOD_ALT)
    ctl.end_stroke(20, 12, modifiers=MOD_ALT)
    check("...Alt-click removes a point (a one-point path draws nothing)",
          len(t.paths[1]["points"]) == 1 and len(t.paths_fractions(96, 64)) == 1)
    check("...the overlay draws the curve", any(k == "poly" and o[1] is False
                                               for k, *o in t.overlay()))
    ctl.cancel_stroke()
    check("Escape puts the layer back", np.array_equal(cell.plane("rgba"), hs))


if __name__ == "__main__":
    main()
