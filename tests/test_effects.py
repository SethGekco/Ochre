#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Properties, rules, adjustments and effects.

Two assertions carry most of the weight. TILING MUST BE INVISIBLE: rendering
an effect in tiles has to produce output byte-identical to rendering it in
one pass, or every blur shows seams. And BLUR MUST BE ALPHA-WEIGHTED, or
every transparent edge grows a dark halo -- the single most commonly botched
operation in a hand-rolled editor.

Run: python3 tests/test_effects.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.effects import (AddNoise, BrightnessContrast, Cancelled,
                                  CancelToken, Desaturate, EffectRegistry,
                                  GaussianBlur, Invert, Levels, Posterize)
from ochre.engine.geometry import Rect
from ochre.engine.ini import IniDB
from ochre.engine.props import (BoolProperty, ChoiceProperty, ColorProperty,
                                FloatProperty, IntProperty, LinkValues,
                                MinMaxPair, PropertyCollection, ReadOnlyWhen)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- properties ------------------------------------------------------
    p = IntProperty("size", 10, 1, 100)
    check("int default", p.value == 10)
    p.value = 50
    check("int assignment", p.value == 50)
    p.value = 500
    check("out-of-range clamps by default", p.value == 100)
    p.value = "42"
    check("int coerces from a string", p.value == 42)
    p.value = "nonsense"
    check("uncoercible input leaves the value alone", p.value == 42)
    p.reset()
    check("reset returns the default", p.value == 10)

    strict = IntProperty("s", 5, 0, 10, on_failure="throw")
    try:
        strict.value = 99
        check("strict mode raises", False, "-- no error")
    except ValueError:
        check("strict mode raises", True)

    f = FloatProperty("gamma", 1.0, 0.1, 10.0)
    f.value = 2.5
    check("float assignment", abs(f.value - 2.5) < 1e-9)

    a = __import__("ochre.engine.props", fromlist=["AngleProperty"]).AngleProperty("angle")
    a.value = 370.0
    check("angle wraps rather than clamps", abs(a.value - 10.0) < 1e-9,
          "-- got %r" % a.value)
    a.value = -10.0
    check("negative angle wraps", abs(a.value - 350.0) < 1e-9)

    b = BoolProperty("on", False)
    b.value = "yes"
    check("bool coerces from a string", b.value is True)

    c = ChoiceProperty("mode", "a", ["a", "b", "c"])
    c.value = "b"
    check("choice accepts a listed option", c.value == "b")
    c.value = "zzz"
    check("choice rejects an unlisted option", c.value == "a")

    col = ColorProperty("tint", (0, 0, 0, 255))
    col.value = (10, 20, 30)
    check("colour widens RGB to RGBA", col.value == (10, 20, 30, 255))

    # read-only is model state
    p = IntProperty("locked", 5, 0, 10)
    p.readonly = True
    p.value = 9
    check("read-only rejects a normal write", p.value == 5)
    p.set(9, force=True)
    check("read-only yields to a forced write", p.value == 9)

    # change notification
    seen = []
    p = IntProperty("watched", 0, 0, 10)
    p.subscribe(lambda prop: seen.append(prop.value))
    p.value = 3
    p.value = 3                       # no change, so no notification
    p.value = 4
    check("listeners fire only on real changes", seen == [3, 4], "-- %r" % seen)

    # ---- collections -----------------------------------------------------
    props = PropertyCollection([IntProperty("a", 1, 0, 10),
                                IntProperty("b", 2, 0, 10)])
    check("collection length", len(props) == 2)
    check("membership", "a" in props and "z" not in props)
    check("ordered", [p.name for p in props] == ["a", "b"])
    token = props.token()
    check("token snapshots values", token == {"a": 1, "b": 2})
    props.set("a", 7)
    check("token is a snapshot, not a view", props.token()["a"] == 7 and token["a"] == 1)
    props.apply(token)
    check("apply restores a token", props.get("a") == 1)

    try:
        props.add(IntProperty("a", 0, 0, 1))
        check("duplicate property rejected", False, "-- no error")
    except ValueError:
        check("duplicate property rejected", True)

    described = props.describe()
    check("describe exposes everything a dialog needs",
          described[0]["kind"] == "int" and described[0]["minimum"] == 0
          and "label" in described[0]["ui"])

    # ---- rules -----------------------------------------------------------
    props = PropertyCollection([IntProperty("r", 16, 2, 64),
                                IntProperty("g", 16, 2, 64),
                                IntProperty("b", 16, 2, 64),
                                BoolProperty("linked", True)])
    props.add_rule(LinkValues(["r", "g", "b"], "linked"))
    props.set("g", 30)
    check("linked properties follow the one that moved",
          props.get("r") == 30 and props.get("b") == 30)
    props.set("linked", False)
    props.set("r", 5)
    check("unlinking stops propagation",
          props.get("g") == 30 and props.get("r") == 5)

    # Rules reject their own misuse at bind time.
    bad = PropertyCollection([IntProperty("x", 1, 0, 10), BoolProperty("s", True)])
    try:
        bad.add_rule(LinkValues(["x", "s"], "s"))
        check("a link rule refuses to link its own switch", False, "-- no error")
    except ValueError:
        check("a link rule refuses to link its own switch", True)

    mismatched = PropertyCollection([IntProperty("p", 1, 0, 10),
                                     IntProperty("q", 1, 0, 999),
                                     BoolProperty("k", True)])
    try:
        mismatched.add_rule(LinkValues(["p", "q"], "k"))
        check("a link rule refuses mismatched ranges", False, "-- no error")
    except ValueError:
        check("a link rule refuses mismatched ranges", True,
              "-- linking would silently clamp")

    doubled = PropertyCollection([IntProperty("p", 1, 0, 10),
                                  IntProperty("q", 1, 0, 10),
                                  BoolProperty("k1", True),
                                  BoolProperty("k2", True)])
    doubled.add_rule(LinkValues(["p", "q"], "k1"))
    try:
        doubled.add_rule(LinkValues(["p", "q"], "k2"))
        check("two link rules cannot share a property", False, "-- no error")
    except ValueError:
        check("two link rules cannot share a property", True)

    # ReadOnlyWhen
    props = PropertyCollection([ChoiceProperty("shape", "linear", ["linear", "radial"]),
                                IntProperty("angle", 0, 0, 359)])
    props.add_rule(ReadOnlyWhen("angle", "shape", when="radial"))
    check("gated property starts writable", not props["angle"].readonly)
    props.set("shape", "radial")
    check("gate closes on the trigger value", props["angle"].readonly)
    props.set("shape", "linear")
    check("gate reopens", not props["angle"].readonly)

    # MinMaxPair pushes rather than refuses
    props = PropertyCollection([IntProperty("lo", 0, 0, 255),
                                IntProperty("hi", 255, 0, 255)])
    props.add_rule(MinMaxPair("lo", "hi"))
    props.set("lo", 200)
    check("low stays where it was put", props.get("lo") == 200)
    check("high is not disturbed while still above", props.get("hi") == 255)
    props.set("hi", 100)
    check("pushing high below low carries low along", props.get("lo") == 100,
          "-- soft pairing must move the partner, not block the input")

    clone = props.clone()
    clone.set("lo", 5)
    check("cloned collections are independent", props.get("lo") == 100)
    check("cloned collections keep their rules", len(clone.rules) == 1)

    # ---- adjustments -----------------------------------------------------
    src = np.zeros((16, 16, 4), dtype=np.uint8)
    src[..., 0] = 100
    src[..., 1] = 150
    src[..., 2] = 200
    src[..., 3] = 255

    out = Invert().apply(src)
    check("invert flips colour", tuple(out[0, 0]) == (155, 105, 55, 255))
    check("invert leaves alpha alone", int(out[0, 0, 3]) == 255)

    effect = BrightnessContrast()
    out = effect.apply(src, token={"brightness": 0, "contrast": 0})
    check("neutral brightness/contrast is the identity",
          np.array_equal(out, src), "-- a no-op adjustment must not alter pixels")
    out = effect.apply(src, token={"brightness": 50, "contrast": 0})
    check("brightness raises", int(out[0, 0, 0]) > 100)
    out = effect.apply(src, token={"brightness": -50, "contrast": 0})
    check("negative brightness lowers", int(out[0, 0, 0]) < 100)

    out = Levels().apply(src, token={"input_low": 0, "input_high": 255,
                                     "gamma": 1.0, "output_low": 0,
                                     "output_high": 255})
    check("neutral levels is the identity", np.array_equal(out, src))

    out = Desaturate().apply(src)
    check("desaturate makes the channels equal",
          out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2])
    expect = int(round(0.299 * 100 + 0.587 * 150 + 0.114 * 200))
    check("desaturate uses Rec.601 luma", int(out[0, 0, 0]) == expect,
          "-- got %d want %d" % (int(out[0, 0, 0]), expect))

    out = Posterize().apply(src, token={"red": 2, "green": 2, "blue": 2,
                                        "linked": True})
    check("posterize quantises to the requested levels",
          set(np.unique(out[..., 0])) <= {0, 255})

    # Adjustments must never touch alpha.
    faded = src.copy()
    faded[..., 3] = 77
    for cls in (Invert, Desaturate, Posterize, BrightnessContrast, Levels):
        got = cls().apply(faded)
        check("%s preserves alpha" % cls.name, int(got[0, 0, 3]) == 77)

    # ---- THE tiling assertion -------------------------------------------
    rng = np.random.default_rng(5)
    noisy = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
    for effect, token in ((GaussianBlur(), {"radius": 5}),
                          (BrightnessContrast(), {"brightness": 20, "contrast": 30}),
                          (Desaturate(), {})):
        one_pass = effect.apply(noisy, token=token, tile=0)
        tiled = effect.apply(noisy, token=token, tile=16)
        check("%s tiles invisibly" % effect.name,
              np.array_equal(one_pass, tiled),
              "-- %d pixels differ at tile boundaries"
              % int(np.count_nonzero((one_pass != tiled).any(axis=2))))

    # ---- THE alpha-weighting assertion ----------------------------------
    # A white opaque disc on a field of TRANSPARENT BLACK. Blurring straight
    # RGB drags the invisible black in and greys the edge; weighting by alpha
    # does not.
    canvas = np.zeros((48, 48, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:48, 0:48]
    disc = ((xx - 24) ** 2 + (yy - 24) ** 2) <= 10 ** 2
    canvas[disc] = (255, 255, 255, 255)

    blurred = GaussianBlur().apply(canvas, token={"radius": 4})
    edge = blurred[(blurred[..., 3] > 20) & (blurred[..., 3] < 235)]
    check("blur produced a soft edge to examine", len(edge) > 0)
    darkest = int(edge[..., :3].min())
    check("BLUR IS ALPHA-WEIGHTED: no dark halo at transparent edges",
          darkest >= 250,
          "-- edge colour fell to %d; straight-RGB blur dragged in invisible black"
          % darkest)

    # A fully transparent input must stay fully transparent.
    empty = np.zeros((32, 32, 4), dtype=np.uint8)
    check("blurring nothing yields nothing",
          int(GaussianBlur().apply(empty, token={"radius": 3}).sum()) == 0)

    check("radius 0 blur is the identity",
          np.array_equal(GaussianBlur().apply(noisy, token={"radius": 0}), noisy))

    # ---- source immutability --------------------------------------------
    class Vandal(BrightnessContrast):
        name = "vandal"

        def render_tile(self, src, dst, rect):
            src[rect.slice()] = 0          # must not be allowed

    try:
        Vandal().apply(src)
        check("an effect cannot write to its own source", False, "-- write succeeded")
    except ValueError:
        check("an effect cannot write to its own source", True)

    # ---- cancellation ----------------------------------------------------
    token_obj = CancelToken()
    check("token starts live", not token_obj.cancelled)

    calls = []

    class Slow(Desaturate):
        name = "slow"

        def render_tile(self, s, d, rect):
            calls.append(rect)
            if len(calls) == 3:
                token_obj.cancel()
            super().render_tile(s, d, rect)

    big = np.zeros((128, 128, 4), dtype=np.uint8)
    try:
        Slow().apply(big, cancel=token_obj, tile=32)
        check("cancellation interrupts the render", False, "-- ran to completion")
    except Cancelled:
        check("cancellation interrupts the render", True)
    check("cancellation is granular, not all-or-nothing",
          3 <= len(calls) < 16, "-- %d tiles ran" % len(calls))

    # ---- progress --------------------------------------------------------
    seen = []
    Desaturate().apply(big, progress=lambda i, n, r: seen.append((i, n)), tile=32)
    check("progress is reported per tile", len(seen) == 16, "-- %d" % len(seen))
    check("progress counts up to the total", seen[-1] == (16, 16))

    # ---- determinism -----------------------------------------------------
    n1 = AddNoise().apply(src, token={"intensity": 40, "seed": 7, "monochrome": False})
    n2 = AddNoise().apply(src, token={"intensity": 40, "seed": 7, "monochrome": False})
    check("noise is deterministic for a seed", np.array_equal(n1, n2))
    n3 = AddNoise().apply(src, token={"intensity": 40, "seed": 8, "monochrome": False})
    check("a different seed gives different noise", not np.array_equal(n1, n3))
    # ...and independent of how the region was tiled, which a naive
    # per-tile RNG would get wrong.
    n4 = AddNoise().apply(src, token={"intensity": 40, "seed": 7, "monochrome": False},
                          tile=4)
    check("noise does not depend on tile size", np.array_equal(n1, n4),
          "-- a per-tile RNG must be seeded from tile POSITION, not order")

    # ---- preview honours the halo ---------------------------------------
    blur = GaussianBlur()
    blur.props.set("radius", 6)
    check("a blur declares a halo", blur.halo == 6)
    check("a pointwise adjustment needs no halo", Desaturate().halo == 0)
    check("adjustments are marked pointwise", Desaturate.pointwise)
    check("blur is not pointwise", not GaussianBlur.pointwise)

    # ---- registry --------------------------------------------------------
    reg = EffectRegistry(IniDB(DATA))
    check("effects registered", len(reg) >= 10)
    check("INI supplies labels", reg.label("gaussian_blur") == "Gaussian Blur")
    check("INI supplies categories", reg.category("gaussian_blur") == "Blurs")
    groups = reg.grouped()
    check("effects are grouped for a menu", "Adjustments" in groups and "Blurs" in groups)
    made = reg.create("gaussian_blur")
    check("INI supplies property defaults", made.props.get("radius") == 4)
    check("overrides beat INI", reg.create("gaussian_blur", radius=9).props.get("radius") == 9)
    check("unknown effect creates nothing", reg.create("nope") is None)

    db = IniDB()
    db.load_string("[Effect:ghost]\nClass = does_not_exist\n")
    check("an effect whose class is missing is not offered",
          "ghost" not in EffectRegistry(db))

    print("\nall effect checks passed")


if __name__ == "__main__":
    main()
