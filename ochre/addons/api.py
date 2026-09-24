# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The addon contract.

This is the whole public surface an addon programs against. It is small on
purpose: everything an addon can do, it does by handing an object to the Host
during `register()`.

Protocols, not abstract base classes. An addon does not import a base class
and subclass it; it presents an object with the right shape. That keeps the
coupling to a documented set of attribute names rather than to Ochre's class
hierarchy, which means an addon keeps working across refactors that would
break inheritance, and it can be tested with a plain stub.

The test applied to every decision here: COULD A THIRD PARTY SHIP THE C&C
ADDON WITH NO COMMIT ACCESS TO OCHRE? Anywhere the answer is no, this file is
defective -- not the addon.

Only PANELS may import Qt. Everything else must work headlessly, so that
batch conversion, scripted export and the test suite do not need a display.
"""

from typing import Protocol, runtime_checkable

#: (major, minor). Major changes break addons; minor additions do not.
API_VERSION = (1, 0)


def compatible(required):
    """Is `required` (a (major, minor) tuple) satisfied by this host?"""
    if not required:
        return True, ""
    major, minor = required[0], (required[1] if len(required) > 1 else 0)
    if major != API_VERSION[0]:
        return False, ("needs API %d.x, host provides %d.%d"
                       % (major, API_VERSION[0], API_VERSION[1]))
    if minor > API_VERSION[1]:
        return False, ("needs API %d.%d, host provides %d.%d"
                       % (major, minor, API_VERSION[0], API_VERSION[1]))
    return True, ""


# ---- extension points ----------------------------------------------------

@runtime_checkable
class FormatProvider(Protocol):
    """Reads and/or writes a file format.

    `sniff` exists because extensions lie. It is handed the first few KB and
    answers whether this provider recognises the content.
    """

    name: str
    extensions: tuple           # ("shp",) -- no dots
    can_read: bool
    can_write: bool

    def sniff(self, head: bytes, path: str) -> bool: ...

    def load(self, path: str, host) -> object: ...      # -> Document

    def save(self, document, path: str, host, options: dict) -> None: ...


@runtime_checkable
class PaletteProvider(Protocol):
    """Supplies named palettes, and the annotations that give them meaning.

    This is the seam that keeps game knowledge out of the core. Ochre knows a
    palette may carry annotated index ranges; it never learns that any
    particular range means anything. A provider supplies both.
    """

    name: str

    def names(self) -> list: ...

    def load(self, name: str, host) -> object: ...      # -> Palette


@runtime_checkable
class CommandProvider(Protocol):
    """A callable action, surfaced in a menu.

    Commands are how an addon does something the core has no concept of --
    the split/stitch of a tile sheet, a batch conversion, a format-specific
    validation pass.
    """

    name: str
    label: str
    category: str

    def run(self, host, **kwargs) -> object: ...


@runtime_checkable
class PanelProvider(Protocol):
    """A dock widget. THE ONLY extension point permitted to import Qt.

    `create` is called lazily, and only when a UI actually exists, so an
    addon that ships a panel still loads cleanly in a headless session.
    """

    name: str
    label: str
    area: str                   # left | right | top | bottom

    def create(self, host, parent): ...


# ---- what an addon is handed --------------------------------------------

class Host:
    """The addon's view of Ochre. Passed to `register(host)`.

    Deliberately a facade rather than the application object: an addon
    reaches only what is on here, which is what makes the surface reviewable
    and keeps a refactor of the app from breaking every addon.
    """

    def __init__(self, registry, addon, settings=None, db=None, controller=None):
        self._registry = registry
        self._addon = addon
        self.settings = settings
        self.db = db
        self.controller = controller
        self.api_version = API_VERSION

    # ---- identity --------------------------------------------------------

    @property
    def addon_id(self):
        return self._addon.id

    @property
    def directory(self):
        return self._addon.directory

    def data_path(self, *parts):
        """A path inside the ADDON's own directory.

        Addons ship their own data -- palettes, lookup tables, annotations --
        and read it from here. The core's data/ is not theirs to write.
        """
        import os
        return os.path.join(self._addon.directory, *parts)

    def read_ini(self, *parts):
        """Load one of the addon's own INI files, house conventions and all."""
        from ..engine.ini import IniDB
        db = IniDB()
        db.load_file(self.data_path(*parts))
        return db

    def log(self, message):
        self._addon.messages.append(str(message))
        return message

    # ---- registration ----------------------------------------------------

    def register_format(self, provider):
        return self._registry.add("formats", provider.name, provider, self._addon)

    def register_palette(self, provider):
        return self._registry.add("palettes", provider.name, provider, self._addon)

    def register_command(self, provider):
        return self._registry.add("commands", provider.name, provider, self._addon)

    def register_panel(self, provider):
        return self._registry.add("panels", provider.name, provider, self._addon)

    def register_effect(self, effect_class, name=None):
        key = name or getattr(effect_class, "name", None)
        return self._registry.add("effects", key, effect_class, self._addon)

    def register_tool(self, tool_class, name=None):
        key = name or getattr(tool_class, "name", None)
        return self._registry.add("tools", key, tool_class, self._addon)

    def register_plane(self, name):
        """Declare a named non-colour plane, e.g. a per-pixel height channel.

        The core stores, undoes and serialises it; it never interprets it.
        Declaring it here is what turns a typo into an error rather than a
        silently orphaned array.
        """
        return self._registry.add("planes", name, name, self._addon)


class AddonError(Exception):
    """An addon misbehaved. Always names which one."""

    def __init__(self, addon_id, message):
        super().__init__("addon %r: %s" % (addon_id, message))
        self.addon_id = addon_id
