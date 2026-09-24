# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Typed properties with constraints and cross-property rules.

This is Paint.NET's PropertySystem idea, and the reason to copy it is that an
effect declares WHAT it is configurable by, never HOW that is presented. The
dialog is then generated. Hundreds of community plugins looked consistent
because none of them laid out a dialog by hand.

Two consequences of that split are worth stating because they are easy to
get wrong:

READ-ONLY LIVES ON THE PROPERTY, not on the widget. "Angle is disabled while
the gradient is radial" is a fact about the model, and putting it there is
what keeps the whole effect system scriptable and headless -- a batch run
enforces the same constraints a dialog does, because they are the same
constraints.

RULES VALIDATE THEMSELVES AT BIND TIME. Two rules linking the same property,
or linking properties whose ranges differ, are bugs you would otherwise ship
and never notice. They raise here instead.

Validation failure is configurable, defaulting to clamp. A slider that
silently clamps is better for a user than an effect that raises; a developer
running with OCHRE_STRICT_PROPS=1 gets the exception instead.
"""

import os

THROW = "throw"
CLAMP = "clamp"
IGNORE = "ignore"

_STRICT = os.environ.get("OCHRE_STRICT_PROPS", "").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_FAILURE = THROW if _STRICT else CLAMP


class Property:
    """One named, typed, constrained value."""

    kind = "value"

    def __init__(self, name, default, label=None, description="",
                 readonly=False, on_failure=None, **ui):
        self.name = name
        self.default = default
        self._value = default
        self._readonly = bool(readonly)
        self.on_failure = on_failure or DEFAULT_FAILURE
        self.listeners = []
        # Presentation hints. The model does not read these; the dialog does.
        self.ui = dict(ui)
        self.ui.setdefault("label", label or _humanise(name))
        self.ui.setdefault("description", description)

    # ---- value -----------------------------------------------------------

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        self.set(new)

    def set(self, new, force=False):
        if self._readonly and not force:
            return self._value
        coerced = self.coerce(new)
        problem = self.validate(coerced)
        if problem is not None:
            if self.on_failure == THROW:
                raise ValueError("%s: %s" % (self.name, problem))
            if self.on_failure == IGNORE:
                return self._value
            coerced = self.clamp(coerced)
        if coerced == self._value:
            return self._value
        self._value = coerced
        for listener in list(self.listeners):
            listener(self)
        return self._value

    def reset(self):
        return self.set(self.default, force=True)

    # ---- hooks subclasses override --------------------------------------

    def coerce(self, value):
        return value

    def validate(self, value):
        return None

    def clamp(self, value):
        return value

    # ---- read-only is model state, not widget state ---------------------

    @property
    def readonly(self):
        return self._readonly

    @readonly.setter
    def readonly(self, flag):
        self._readonly = bool(flag)

    def subscribe(self, fn):
        self.listeners.append(fn)
        return fn

    def clone(self):
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.listeners = []
        out.ui = dict(self.ui)
        return out

    def __repr__(self):
        return "%s(%r=%r)" % (type(self).__name__, self.name, self._value)


class IntProperty(Property):
    kind = "int"

    def __init__(self, name, default, minimum=0, maximum=100, **kw):
        self.minimum, self.maximum = int(minimum), int(maximum)
        super().__init__(name, int(default), **kw)

    def coerce(self, value):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return self._value

    def validate(self, value):
        if not self.minimum <= value <= self.maximum:
            return "%d outside %d..%d" % (value, self.minimum, self.maximum)
        return None

    def clamp(self, value):
        return max(self.minimum, min(self.maximum, value))


class FloatProperty(Property):
    kind = "float"

    def __init__(self, name, default, minimum=0.0, maximum=1.0, decimals=2, **kw):
        self.minimum, self.maximum = float(minimum), float(maximum)
        kw.setdefault("decimals", decimals)
        super().__init__(name, float(default), **kw)

    def coerce(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return self._value

    def validate(self, value):
        if not self.minimum <= value <= self.maximum:
            return "%g outside %g..%g" % (value, self.minimum, self.maximum)
        return None

    def clamp(self, value):
        return max(self.minimum, min(self.maximum, value))


class AngleProperty(FloatProperty):
    """Degrees, wrapped rather than clamped -- 370 means 10, not 360."""

    kind = "angle"

    def __init__(self, name, default=0.0, **kw):
        super().__init__(name, default, 0.0, 360.0, **kw)

    def coerce(self, value):
        try:
            return float(value) % 360.0
        except (TypeError, ValueError):
            return self._value


class BoolProperty(Property):
    kind = "bool"

    def coerce(self, value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


class ChoiceProperty(Property):
    kind = "choice"

    def __init__(self, name, default, choices, **kw):
        self.choices = list(choices)
        super().__init__(name, default, **kw)

    def validate(self, value):
        if value not in self.choices:
            return "%r is not one of %r" % (value, self.choices)
        return None

    def clamp(self, value):
        return self.default if value not in self.choices else value


class ColorProperty(Property):
    kind = "color"

    def coerce(self, value):
        try:
            parts = tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return self._value
        if len(parts) == 3:
            parts = parts + (255,)
        return parts if len(parts) == 4 else self._value

    def validate(self, value):
        if any(not 0 <= v <= 255 for v in value):
            return "channel outside 0..255"
        return None

    def clamp(self, value):
        return tuple(max(0, min(255, v)) for v in value)


# ---- rules --------------------------------------------------------------

class Rule:
    """A relationship between properties, enforced by the model."""

    def bind(self, collection):
        self.collection = collection
        self.validate(collection)
        self.attach(collection)
        self.sync()
        return self

    def validate(self, collection):
        pass

    def attach(self, collection):
        pass

    def sync(self):
        pass


class LinkValues(Rule):
    """Keep several properties equal while a boolean is set.

    Whichever property the user last moved is the one that propagates -- drag
    the green slider with "link" on and red and blue follow it, not the other
    way round.
    """

    def __init__(self, targets, switch, inverse=False):
        self.targets = list(targets)
        self.switch = switch
        self.inverse = bool(inverse)
        self._last = None
        self._busy = False

    def validate(self, collection):
        if self.switch in self.targets:
            raise ValueError("LinkValues: the switch cannot be one of its targets")
        for name in self.targets + [self.switch]:
            if name not in collection:
                raise ValueError("LinkValues: unknown property %r" % name)
        # Linked properties with different ranges would silently clamp each
        # other, which looks like a bug in the effect rather than in the rule.
        spans = {(getattr(collection[n], "minimum", None),
                  getattr(collection[n], "maximum", None)) for n in self.targets}
        if len(spans) > 1:
            raise ValueError("LinkValues: targets have differing ranges %r" % spans)
        # Two link rules over one property fight each other forever.
        for other in collection.rules:
            if other is self or not isinstance(other, LinkValues):
                continue
            shared = set(self.targets) & set(other.targets)
            if shared:
                raise ValueError("LinkValues: %r is already linked by another rule"
                                 % sorted(shared))

    def attach(self, collection):
        for name in self.targets:
            collection[name].subscribe(self._changed)
        collection[self.switch].subscribe(lambda _p: self.sync())

    def _changed(self, prop):
        self._last = prop.name
        self.sync()

    def _active(self):
        on = bool(self.collection[self.switch].value)
        return (not on) if self.inverse else on

    def sync(self):
        if self._busy or not self._active():
            return
        source = self._last or self.targets[0]
        if source not in self.targets:
            return
        self._busy = True
        try:
            value = self.collection[source].value
            for name in self.targets:
                if name == source:
                    continue
                self.collection[name].set(value, force=True)
        finally:
            self._busy = False


class ReadOnlyWhen(Rule):
    """Disable a property based on another's value.

    Modelled as read-only on the property rather than enabled on a widget,
    so a script hits the same constraint a dialog does.
    """

    def __init__(self, target, source, when=True):
        self.target, self.source, self.when = target, source, when

    def validate(self, collection):
        for name in (self.target, self.source):
            if name not in collection:
                raise ValueError("ReadOnlyWhen: unknown property %r" % name)
        if self.target == self.source:
            raise ValueError("ReadOnlyWhen: a property cannot gate itself")

    def attach(self, collection):
        collection[self.source].subscribe(lambda _p: self.sync())

    def sync(self):
        self.collection[self.target].readonly = \
            self.collection[self.source].value == self.when


class MinMaxPair(Rule):
    """Keep a low/high pair ordered by pushing the partner, not by refusing.

    "Soft" in Paint.NET's sense: dragging the low slider past the high one
    carries the high one along instead of blocking at it, which is what feels
    right in a levels dialog.
    """

    def __init__(self, low, high):
        self.low, self.high = low, high
        self._busy = False

    def validate(self, collection):
        for name in (self.low, self.high):
            if name not in collection:
                raise ValueError("MinMaxPair: unknown property %r" % name)

    def attach(self, collection):
        collection[self.low].subscribe(self._low_moved)
        collection[self.high].subscribe(self._high_moved)

    def _low_moved(self, prop):
        if self._busy:
            return
        high = self.collection[self.high]
        if prop.value > high.value:
            self._busy = True
            try:
                high.set(prop.value, force=True)
            finally:
                self._busy = False

    def _high_moved(self, prop):
        if self._busy:
            return
        low = self.collection[self.low]
        if prop.value < low.value:
            self._busy = True
            try:
                low.set(prop.value, force=True)
            finally:
                self._busy = False


# ---- collection ---------------------------------------------------------

class PropertyCollection:
    """An ordered set of properties plus the rules relating them."""

    def __init__(self, properties=(), rules=()):
        self._props = {}
        self.order = []
        self.rules = []
        for prop in properties:
            self.add(prop)
        for rule in rules:
            self.add_rule(rule)

    def add(self, prop):
        if prop.name in self._props:
            raise ValueError("duplicate property %r" % prop.name)
        self._props[prop.name] = prop
        self.order.append(prop.name)
        return prop

    def add_rule(self, rule):
        rule.bind(self)
        self.rules.append(rule)
        return rule

    def __getitem__(self, name):
        return self._props[name]

    def __contains__(self, name):
        return name in self._props

    def __iter__(self):
        return (self._props[n] for n in self.order)

    def __len__(self):
        return len(self.order)

    def get(self, name, default=None):
        prop = self._props.get(name)
        return default if prop is None else prop.value

    def set(self, name, value):
        return self._props[name].set(value)

    def token(self):
        """An immutable snapshot of every value.

        Effects render from a token, never from the live collection, so a
        slider moving mid-render cannot tear the parameters.
        """
        return {name: self._props[name].value for name in self.order}

    def apply(self, token):
        for name, value in (token or {}).items():
            if name in self._props:
                self._props[name].set(value, force=True)
        return self

    def reset(self):
        for prop in self:
            prop.reset()
        return self

    def clone(self):
        """A detached copy with the same rules re-bound to it."""
        out = PropertyCollection([p.clone() for p in self])
        for rule in self.rules:
            import copy
            fresh = copy.copy(rule)
            fresh._busy = False
            out.add_rule(fresh)
        return out

    def describe(self):
        """Everything a dialog needs, with no Qt types involved."""
        return [{"name": p.name, "kind": p.kind, "value": p.value,
                 "default": p.default, "readonly": p.readonly,
                 "minimum": getattr(p, "minimum", None),
                 "maximum": getattr(p, "maximum", None),
                 "choices": getattr(p, "choices", None),
                 "ui": dict(p.ui)}
                for p in self]


def _humanise(name):
    return name.replace("_", " ").strip().capitalize()
