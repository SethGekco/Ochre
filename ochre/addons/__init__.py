# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Ochre's addon system.

An addon is a directory containing `addon.ini` and an entry module exposing
`register(host)`. Nothing runs during discovery; nothing loads until asked.

Only PanelProvider may import Qt, so an addon that ships a panel still loads
cleanly in a headless session.
"""

from .api import (API_VERSION, AddonError, CommandProvider, FormatProvider,
                  Host, PaletteProvider, PanelProvider, compatible)
from .loader import Addon, AddonManager, Registry
from .manifest import Manifest

__all__ = [
    "API_VERSION", "compatible", "Host", "AddonError",
    "FormatProvider", "PaletteProvider", "CommandProvider", "PanelProvider",
    "AddonManager", "Registry", "Addon", "Manifest",
]
