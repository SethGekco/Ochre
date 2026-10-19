# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""What a tool lets the user adjust, declared next to the tool.

Every tool lists its adjustable options as `options_spec`. The UI generates
the options bar from that list and nothing else, so a tool -- including one
shipped by an addon -- gets controls for its options without the window
learning its name. This is the effect-dialog arrangement applied to tools,
and for the same reason: hand-built per-tool panels are how options end up
existing in code and in the INI while being unreachable from the screen,
which is exactly what had happened to brush size and the shape chooser.

Descriptors are deliberately inert: a name, a kind, a default, limits and
labels. Values live on the tool instance (`tool.option(name)`), the INI
supplies per-tool defaults, and the controller remembers what the user
chose. `coerce()` is the one place a raw value -- from a widget, an INI file
or a remembered session -- is turned into something the tool can trust.
"""

INT, FLOAT, BOOL, CHOICE, TEXT, FONT = "int", "float", "bool", "choice", "text", "font"


class Opt:
    __slots__ = ("name", "kind", "default", "label", "minimum", "maximum",
                 "step", "choices", "labels", "groups", "tip", "suffix",
                 "decimals", "percent")

    def __init__(self, name, kind, default, label=None, minimum=None, maximum=None,
                 step=None, choices=None, labels=None, groups=None, tip="",
                 suffix="", decimals=2, percent=False):
        self.name = name
        self.kind = kind
        self.default = default
        self.label = label or name.replace("_", " ").capitalize()
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.choices = tuple(choices) if choices is not None else None
        # Display names for choices; defaults to the choice itself, tidied.
        self.labels = dict(labels or {})
        # For large choice sets (the shape chooser): ((heading, (choice, ...)),
        # ...). A UI may lay these out as a categorised palette.
        self.groups = tuple(groups) if groups else None
        self.tip = tip
        self.suffix = suffix
        self.decimals = decimals
        # Show a 0..1 float as 0..100 %. Purely presentational.
        self.percent = percent

    def label_for(self, choice):
        if choice in self.labels:
            return self.labels[choice]
        return str(choice).replace("_", " ").capitalize()

    def coerce(self, value):
        """A raw value made safe: typed, clamped, or the default."""
        if value is None:
            return self.default
        try:
            if self.kind == BOOL:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "on", "true", "yes")
                return bool(value)
            if self.kind == INT:
                v = int(round(float(value)))
            elif self.kind == FLOAT:
                v = float(value)
            elif self.kind == CHOICE:
                v = str(value)
                if self.choices is not None and v not in self.choices:
                    low = v.lower()
                    match = [c for c in self.choices if str(c).lower() == low]
                    return match[0] if match else self.default
                return v
            else:
                return str(value)
        except (TypeError, ValueError):
            return self.default
        if self.minimum is not None:
            v = max(self.minimum, v)
        if self.maximum is not None:
            v = min(self.maximum, v)
        return v

    def __repr__(self):
        return "Opt(%r, %s, %r)" % (self.name, self.kind, self.default)


# ---- the options most tools share -------------------------------------------
#
# Functions rather than shared instances so each tool can tune the default
# without mutating another tool's descriptor.

def size(default=16, maximum=2000, label="Size"):
    return Opt("size", INT, default, label, 1, maximum, suffix=" px",
               tip="Brush diameter. [ and ] change it from the keyboard.")


def hardness(default=0.85):
    return Opt("hardness", FLOAT, default, "Hardness", 0.0, 1.0, 0.05, percent=True,
               tip="100% is a crisp edge; lower values soften it inward.")


def opacity(default=1.0, name="flow", label="Flow"):
    return Opt(name, FLOAT, default, label, 0.01, 1.0, 0.05, percent=True)


def antialias(default=True):
    return Opt("antialias", BOOL, default, "Antialias",
               tip="Smooth edges. Turn off for pixel art.")


def tolerance(default=32):
    return Opt("tolerance", INT, default, "Tolerance", 0, 255,
               tip="How different a colour may be and still count as the same.")


def selection_mode():
    # The selection engine's own constants, not a parallel spelling: the
    # first version declared "replace" while the engine expects "Replace",
    # which no test noticed until declared defaults started being applied.
    from ..selection import EXCLUDE, INTERSECT, MODES, REPLACE, UNION, XOR
    return Opt("mode", CHOICE, REPLACE, "Mode", choices=MODES,
               labels={REPLACE: "Replace", UNION: "Add (union)", EXCLUDE: "Subtract",
                       INTERSECT: "Intersect", XOR: "Invert (xor)"},
               tip="How a new selection combines with the existing one.")


def stroke(default=2):
    return Opt("stroke", INT, default, "Brush width", 1, 500, suffix=" px")


def fill_style(default="both"):
    return Opt("style", CHOICE, default, "Draw", choices=("outline", "fill", "both"),
               labels={"outline": "Outline", "fill": "Fill",
                       "both": "Outline + fill"},
               tip="Outline uses the primary colour, fill the secondary.")
