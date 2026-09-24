# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Ochre's engine: the I/O-free, Qt-free core.

Nothing in this package may import PySide6, shiboken6 or PyQt. That rule is
enforced by tests/test_no_qt.py, not by convention.
"""

from . import accel
from .arith import blend_over, lerp255, mul255
from .blend import BlendMode, BlendRegistry
from .commands import (CellSnapshot, Command, CompoundCommand,
                       PaletteDelta, PixelDelta, PropertyDelta)
from .compositor import Compositor
from .document import Document
from .effects import (Adjustment, CancelToken, Cancelled, Effect,
                      EffectRegistry)
from . import fileio
from .frame import COLD, Frame, UNLOADED, WARM
from .geometry import DirtyRegion, Rect
from .gradient import Gradient
from .history import HistoryEntry, HistoryStack
from .ini import IniDB
from .layer import Layer, LayerGroup, LayerNode
from .palette import Palette, grayscale
from .quantize import (convert_layer_to_indexed, palette_usage,
                       quantization_error, quantize)
from .props import (BoolProperty, ChoiceProperty, ColorProperty,
                    FloatProperty, IntProperty, PropertyCollection)
from .selection import Selection
from . import shapes, text
from .settings import DEFAULTS, Settings
from .stroke import StrokeSession
from .tools import Tool, ToolContext, ToolEvent, ToolRegistry
from .surface import Surface

__all__ = [
    "accel", "BlendMode", "BlendRegistry", "Compositor",
    "Command", "PixelDelta", "PropertyDelta", "PaletteDelta",
    "CompoundCommand", "CellSnapshot", "HistoryStack", "HistoryEntry",
    "Effect", "Adjustment", "EffectRegistry", "CancelToken", "Cancelled",
    "quantize", "convert_layer_to_indexed", "quantization_error",
    "palette_usage", "PropertyCollection", "IntProperty", "FloatProperty", "BoolProperty",
    "ChoiceProperty", "ColorProperty",
    "fileio", "StrokeSession", "Gradient", "shapes", "text", "Selection", "Tool", "ToolEvent", "ToolContext",
    "ToolRegistry", "Document", "Frame", "Layer", "LayerGroup", "LayerNode", "Palette",
    "Surface", "Rect", "DirtyRegion", "IniDB", "Settings", "DEFAULTS",
    "grayscale", "mul255", "lerp255", "blend_over",
    "WARM", "COLD", "UNLOADED",
]
