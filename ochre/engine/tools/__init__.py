# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Tools: Qt-free, event-driven, and INI-registered.

A tool receives ToolEvents -- plain data, not widget types -- so the whole
layer is testable headless and scriptable. The registry is populated from
data/tools.ini, which means an addon adds a tool by shipping an INI section
and a class, without the core knowing anything about it.
"""

from .base import (MOD_ALT, MOD_CTRL, MOD_NONE, MOD_SHIFT, Tool, ToolContext,
                   ToolEvent, constrain_angle, constrain_square, drag_rect)
from .paint import (BrushTool, BucketFillTool, EraserTool, EyedropperTool,
                    PencilTool)
from .movetools import (CloneStampTool, MovePixelsTool,
                        MoveSelectionTool)
from .select import (EllipseSelectTool, LassoTool, MagicWandTool,
                     RectSelectTool)
from .shapetools import (EllipseTool, FreeformTool, GradientTool, LineTool,
                         RectangleTool, TextTool)
from .viewtools import PanTool, ViewRequest, ZoomTool

BUILTIN = {
    cls.name: cls for cls in (
        PencilTool, BrushTool, EraserTool, BucketFillTool, EyedropperTool,
        RectSelectTool, EllipseSelectTool, LassoTool, MagicWandTool,
        LineTool, RectangleTool, EllipseTool, FreeformTool, GradientTool,
        TextTool, CloneStampTool, MovePixelsTool, MoveSelectionTool,
        ZoomTool, PanTool,
    )
}


class ToolRegistry:
    """Known tools, with their INI-declared defaults."""

    def __init__(self, db=None, extra=None):
        self._classes = dict(BUILTIN)
        if extra:
            self._classes.update(extra)
        self._defaults = {}
        self._order = {}
        self._labels = {}
        self._appearance = {}
        if db is not None:
            self.load(db)

    def load(self, db):
        """Read [Tool:Name] sections. Unknown keys become tool options."""
        # Cursor and Footprint are presentation, read by the canvas through
        # appearance() below. They are reserved so they do not silently
        # become tool options and end up passed to a tool's constructor --
        # the engine has no business knowing what a cursor is.
        reserved = {"Class", "Label", "Order", "TargetPlane",
                    "Cursor", "Footprint"}
        for name in db.sections("Tool"):
            section = "Tool:" + name
            cls_name = db.get(section, "Class", name)
            if cls_name not in self._classes:
                continue        # a tool whose code is absent is simply not offered
            key = name
            self._classes[key] = self._classes[cls_name]
            self._labels[key] = db.get(section, "Label", key)
            self._order[key] = db.getint(section, "Order", 0)
            opts = {}
            for opt in db.keys(section):
                if opt in reserved:
                    continue
                raw = db.get(section, opt)
                opts[opt.lower()] = _coerce(raw)
            plane = db.get(section, "TargetPlane")
            if plane:
                opts["target_plane"] = plane
            self._defaults[key] = opts
            self._appearance[key] = (
                (db.get(section, "Cursor") or "cross").lower(),
                (db.get(section, "Footprint") or "none").lower())
        return self

    def appearance(self, name):
        """(cursor, footprint) for a tool -- how the POINTER should look.

        Kept here rather than in the UI because it is per-tool configuration
        like everything else in tools.ini, which is also what lets an addon's
        tool declare its own cursor without the canvas learning its name.
        The two strings are opaque to the engine; the canvas interprets them.
        """
        return self._appearance.get(name, ("cross", "none"))

    def create(self, name, **overrides):
        cls = self._classes.get(name)
        if cls is None:
            return None
        options = dict(self._defaults.get(name, {}))
        options.update(overrides)
        tool = cls(**options)
        plane = options.get("target_plane")
        if plane:
            tool.target_plane = plane
        return tool

    def label(self, name):
        return self._labels.get(name, getattr(self._classes.get(name), "label", name))

    def names(self):
        return sorted(self._classes, key=lambda n: (self._order.get(n, 999), n))

    def __contains__(self, name):
        return name in self._classes

    def __len__(self):
        return len(self._classes)


def _coerce(raw):
    """Best-effort typing for an INI option value."""
    text = (raw or "").strip()
    low = text.lower()
    if low in ("on", "true", "yes"):
        return True
    if low in ("off", "false", "no"):
        return False
    try:
        return int(text, 0)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


__all__ = [
    "Tool", "ToolEvent", "ToolContext", "ToolRegistry", "BUILTIN",
    "PencilTool", "BrushTool", "EraserTool", "BucketFillTool",
    "EyedropperTool", "RectSelectTool", "EllipseSelectTool", "LassoTool",
    "MagicWandTool", "LineTool", "RectangleTool", "EllipseTool",
    "FreeformTool", "GradientTool", "TextTool", "CloneStampTool", "MovePixelsTool",
    "MoveSelectionTool", "constrain_angle", "constrain_square", "drag_rect",
    "MOD_NONE", "MOD_SHIFT", "MOD_CTRL", "MOD_ALT",
]
