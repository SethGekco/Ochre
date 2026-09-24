# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""UI settings. Same rule as the engine: the DEFAULTS dict is the schema.

Precedence is DEFAULTS < data/ui.ini < OCHRE_<KEY>, and the type of each
default drives coercion, so adding a setting is one line and gets typing,
INI override and environment override for free.
"""

from ..engine.settings import Settings, _coerce      # noqa: F401

UI_DEFAULTS = {
    "Window": {
        "Width": 1280,
        "Height": 800,
        "Title": "Ochre",
    },
    "Canvas": {
        "CheckerSize": 16,
        "CheckerLight": (200, 200, 200, 255),
        "CheckerDark": (160, 160, 160, 255),
        # Above this zoom a pixel grid is drawn. Pixel art wants it; a photo
        # at 200% does not, which is why it is a threshold and not a toggle.
        "PixelGridAbove": 800,
        "GridColour": (128, 128, 128, 90),
        # Repaints are coalesced on a timer. Without this, tiles completing
        # at high frequency drown the UI thread and a multithreaded render
        # ends up SLOWER than a single-threaded one.
        "RepaintMs": 16,
        "BackgroundColour": (60, 62, 68, 255),
    },
    "Tools": {
        "DefaultTool": "brush",
    },
}


class UiSettings(Settings):
    def __init__(self, data_dir=None, env=None, db=None):
        super().__init__(data_dir, UI_DEFAULTS, env, db)
