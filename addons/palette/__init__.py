# Ochre Palette addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""Palettes in the interface: the Palette bar on the top toolbar, and the
"Palette" tab of the Primary / Secondary colour picker.

The palette machinery itself (reading palette files, reducing, mapping,
indexed layers) is the core's; this addon is the UI for it, so switching it
off takes the bar and the tab away and leaves everything else working.
Other addons that are about palettes say `Requires = palette` and turn it on
with them.

Both contributions are panels, created lazily and only when a UI exists, so
the addon loads cleanly in a headless session.
"""


class PaletteBarPanel:
    name = "palette.bar"
    label = "Palette"
    area = "toolbar"

    def create(self, host, parent):
        from .palettebar import PaletteBar
        return PaletteBar(host.controller, parent)


class PaletteTabPanel:
    name = "palette.picker"
    label = "Palette"
    area = "picker"

    def create(self, host, parent):
        from .pickertab import PaletteTab
        return PaletteTab(host.controller, parent)


def register(host):
    host.register_panel(PaletteBarPanel())
    host.register_panel(PaletteTabPanel())
