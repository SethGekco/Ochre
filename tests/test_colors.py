#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Colour adjustments: each checked on the property that defines it.

Run: python3 tests/test_colors.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import colors
from ochre.engine.effects import EffectRegistry
from ochre.engine.fileio import ochre_format
from ochre.engine.ini import IniDB
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry(IniDB(DATA))


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def run(name, img, **params):
    return REG.create(name, **params).apply(img)


def solid(rgb, a=255, size=(4, 4)):
    out = np.zeros(size + (4,), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = a
    return out


def ramp():
    x = np.arange(256, dtype=np.uint8)
    img = np.zeros((1, 256, 4), np.uint8)
    img[0, :, 0] = img[0, :, 1] = img[0, :, 2] = x
    img[..., 3] = 255
    return img


def main():
    rng = np.random.default_rng(3)
    photo = rng.integers(0, 256, (24, 24, 4), dtype=np.uint8)
    photo[..., 3] = 255

    # ---- Curves -------------------------------------------------------------
    check("an identity curve changes nothing", np.array_equal(run("curves", photo), photo))
    lut = colors.monotone_curve([(0, 0), (64, 10), (128, 200), (255, 255)])
    check("a curve passes through its points", lut[64] == 10 and lut[128] == 200)
    check("...and never overshoots between them (monotone)", bool(np.all(np.diff(lut.astype(int)) >= 0)))
    out = run("curves", ramp(), rgb=((0, 0), (128, 180), (255, 255)))
    check("lifting the midpoint brightens midtones, keeps the ends",
          out[0, 128, 0] == 180 and out[0, 0, 0] == 0 and out[0, 255, 0] == 255)
    out = run("curves", solid((100, 100, 100)), red=((0, 0), (100, 160), (255, 255)))
    check("a channel curve acts on that channel only", tuple(out[0, 0, :3]) == (160, 100, 100))
    check("adjustments never touch alpha",
          np.array_equal(run("curves", solid((9, 9, 9), 77), rgb="0,50;255,255")[..., 3], np.full((4, 4), 77)))

    # ---- Color Balance, Exposure, Threshold --------------------------------------
    dark, light = solid((30, 30, 30)), solid((230, 230, 230))
    d2 = run("color_balance", dark, shadows_cyan_red=80, preserve_luminosity=False)
    l2 = run("color_balance", light, shadows_cyan_red=80, preserve_luminosity=False)
    check("Color Balance's shadow shift moves the shadows, not the highlights",
          d2[0, 0, 0] > 60 and abs(int(l2[0, 0, 0]) - 230) <= 3)
    d3 = run("color_balance", dark, shadows_cyan_red=80, preserve_luminosity=True)
    check("...and Preserve Luminosity keeps the lightness",
          abs(colors._hsl_lightness(d3[0, 0, :3].astype(float)) - 30) < 2 and d3[0, 0, 0] > d3[0, 0, 1])
    e = run("exposure", solid((118, 118, 118)), exposure=1.0)
    check("+1 stop of exposure doubles the light (in linear terms)",
          abs(e[0, 0, 0] - 161) <= 2, str(e[0, 0]))
    t = run("threshold", ramp(), level=100)
    check("Threshold splits at its level", t[0, 99, 0] == 0 and t[0, 100, 0] == 255)

    # ---- Vibrance, Channel Mixer, Photo Filter, Gradient Map -------------------------
    dull, vivid = solid((120, 110, 100)), solid((250, 20, 20))
    dv = run("vibrance", dull, vibrance=80)
    vv = run("vibrance", vivid, vibrance=80)
    gain_dull = (int(dv[0, 0, 0]) - int(dv[0, 0, 2])) / 20.0
    gain_vivid = (int(vv[0, 0, 0]) - int(vv[0, 0, 2])) / 230.0
    check("Vibrance boosts dull colours more than vivid ones", gain_dull > gain_vivid)
    sw = run("channel_mixer", solid((200, 50, 10)), red_from_red=0, red_from_blue=100,
             blue_from_blue=0, blue_from_red=100)
    check("Channel Mixer swaps channels", tuple(sw[0, 0, :3]) == (10, 50, 200))
    mono = run("channel_mixer", solid((200, 50, 10)), monochrome=True)
    check("...and makes monochrome from the red row", len(set(mono[0, 0, :3])) == 1)
    pf = run("photo_filter", solid((128, 128, 128)), color=(255, 0, 0, 255), density=50)
    check("a Photo Filter tints toward its colour, keeping lightness",
          pf[0, 0, 0] > pf[0, 0, 2] and abs(colors._lum(pf[0, 0, :3].astype(float)) - 128) < 2)
    gm = run("gradient_map", ramp(), shadows=(255, 0, 0, 255), highlights=(0, 0, 255, 255))
    check("a Gradient Map maps dark to one colour, light to the other",
          tuple(gm[0, 0, :3]) == (255, 0, 0) and tuple(gm[0, 255, :3]) == (0, 0, 255))

    # ---- Black & White, Colorize, Temperature ------------------------------------------
    red, blue = solid((255, 0, 0)), solid((0, 0, 255))
    bw_r = run("black_white", red)
    bw_b = run("black_white", blue)
    check("Black & White makes grey", len(set(bw_r[0, 0, :3])) == 1)
    check("...each hue at its own lightness (reds 40, blues 20 by default)",
          bw_r[0, 0, 0] > bw_b[0, 0, 0])
    bw_r2 = run("black_white", red, reds=150)
    check("...which you choose", bw_r2[0, 0, 0] > bw_r[0, 0, 0])
    cz = run("colorize", ramp(), hue=240, saturation=80)
    mid = cz[0, 128]
    check("Colorize tints every level with one hue", mid[2] > mid[0] and mid[2] > mid[1]
          and cz[0, 0, 0] == 0 and cz[0, 255, 2] == 255)
    warm = run("color_temperature", solid((180, 180, 180)), original=6500, intended=3500)
    check("Color Temperature toward a lower kelvin warms (red over blue)",
          warm[0, 0, 0] > warm[0, 0, 2])

    # ---- Replace Color, Color to Alpha ----------------------------------------------------
    two = np.concatenate([solid((250, 10, 10), size=(2, 2)), solid((10, 250, 10), size=(2, 2))], axis=1)
    rc = run("replace_color", two, color=(255, 0, 0, 255), fuzziness=60, hue=120)
    check("Replace Color changes only colours near the chosen one",
          rc[0, 0, 1] > rc[0, 0, 0] and tuple(rc[0, 3, :3]) == (10, 250, 10))
    ca = run("color_to_alpha", np.concatenate([solid((255, 255, 255), size=(1, 1)),
                                               solid((128, 128, 128), size=(1, 1)),
                                               solid((0, 0, 0), size=(1, 1))], axis=1))
    check("Color to Alpha: white becomes transparent", ca[0, 0, 3] == 0)
    check("...grey becomes HALF-transparent BLACK (unmixed, no halo)",
          abs(int(ca[0, 1, 3]) - 128) <= 1 and max(ca[0, 1, :3]) <= 1)
    check("...black stays opaque black", ca[0, 2, 3] == 255 and max(ca[0, 2, :3]) == 0)

    # ---- the automatic ones -----------------------------------------------------------------
    flat = (ramp().astype(np.float64) * 0.4 + 60).astype(np.uint8)
    flat[..., 3] = 255
    al = run("auto_levels", flat, clip=0.0)
    check("Auto Levels stretches to the full range", al[..., 0].min() == 0 and al[..., 0].max() == 255)
    cast = photo.copy()
    cast[..., 2] = (cast[..., 2] * 0.5).astype(np.uint8)
    ac = run("auto_color", cast)
    means = ac[..., :3].reshape(-1, 3).mean(axis=0)
    check("Auto Color neutralises a colour cast (channel means within 12)",
          means.max() - means.min() < 12, str(means))
    eq = run("equalize", flat)
    hist = np.bincount(eq[..., 0].ravel(), minlength=256)
    check("Equalize spreads the levels across the range", eq[..., 0].max() == 255
          and (hist > 0).sum() > 60)
    check("the automatic ones are kept out of adjustment layers",
          all(getattr(REG._classes[n], "global_stats", False)
              for n in ("auto_levels", "auto_contrast", "auto_color", "equalize"))
          and not getattr(REG._classes["curves"], "global_stats", False))

    # ---- through the controller: as an adjustment layer, saved and loaded ------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(32, 32, background=(100, 100, 100, 255))
    adj = ctl.add_adjustment_layer("curves", {"rgb": ((0, 0), (100, 200), (255, 255))})
    ctl.compositor.invalidate()
    check("Curves works as an adjustment layer",
          tuple(ctl.compositor.composite(ctl.doc.bounds)[5, 5, :3]) == (200, 200, 200))
    f = os.path.join(tempfile.mkdtemp(), "curves.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check("...and its curve survives save and load",
          tuple(doc2.layer(adj.id).params["rgb"]) == ((0, 0), (100, 200), (255, 255)))

    # ---- Match Color between two open documents ----------------------------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(40, 40, background=(200, 120, 60, 255))
    warm_cell = ctl.doc.cell(ctl.active_layer)
    warm_cell.plane("rgba")[:, :20, :3] = (230, 150, 80)
    warm_cell._touched(ctl.doc.bounds)
    ctl.set_tool("pencil")
    ctl.begin_stroke(1, 1)
    ctl.end_stroke(1, 1)
    ctl.new_document(40, 40, background=(60, 90, 160, 255))
    cool = ctl.doc.cell(ctl.active_layer)
    cool.plane("rgba")[:, :20, :3] = (40, 70, 140)
    cool._touched(ctl.doc.bounds)
    ctl.match_color(0)
    after = cool.plane("rgba")[..., :3].reshape(-1, 3).mean(axis=0)
    check("Match Color gives one image another's colour (warm now)", after[0] > after[2] + 40,
          str(after))
    ctl.undo()
    check("...undoably", tuple(cool.plane("rgba")[30, 30, :3]) == (60, 90, 160))

    print("\nall colour checks passed")


if __name__ == "__main__":
    main()
