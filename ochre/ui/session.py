# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Small facts remembered between runs. Plain INI in the state directory.

Currently one: the dimensions of the last document that was open, so the
untitled canvas a fresh launch starts with is the size you were last
working at. Qt-free, and an INI rather than QSettings for the same reason as
everything else here -- readable with `cat`, fixable with an editor.
"""

import configparser
import os


class Session:
    def __init__(self, state_dir):
        self.path = os.path.join(state_dir, "session.ini") if state_dir else None
        self._cp = configparser.ConfigParser()
        if self.path and os.path.isfile(self.path):
            try:
                self._cp.read(self.path, encoding="utf-8")
            except (OSError, configparser.Error):
                self._cp = configparser.ConfigParser()

    def last_size(self):
        """(width, height) of the last document, or None."""
        try:
            w = self._cp.getint("Last", "Width")
            h = self._cp.getint("Last", "Height")
        except (configparser.Error, ValueError):
            return None
        if 1 <= w <= 65535 and 1 <= h <= 65535:
            return (w, h)
        return None

    # ---- per-tool options --------------------------------------------------

    def tool_options(self, tool):
        """{option: raw string} the user last chose for a tool."""
        section = "Tool:" + tool
        if not self._cp.has_section(section):
            return {}
        return dict(self._cp.items(section))

    def remember_tool_option(self, tool, key, value):
        section = "Tool:" + tool
        if not self._cp.has_section(section):
            self._cp.add_section(section)
        if isinstance(value, bool):
            value = "on" if value else "off"
        self._cp.set(section, key, str(value))
        return self._write()

    # ---- simple preferences -------------------------------------------------

    def flag(self, key, default=False):
        try:
            return self._cp.getboolean("Prefs", key)
        except (configparser.Error, ValueError):
            return default

    def set_flag(self, key, value):
        if not self._cp.has_section("Prefs"):
            self._cp.add_section("Prefs")
        self._cp.set("Prefs", key, "on" if value else "off")
        return self._write()

    def _write(self):
        if not self.path:
            return False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                self._cp.write(f)
            return True
        except OSError:
            return False

    def remember_size(self, width, height):
        if not self.path:
            return False
        if self.last_size() == (int(width), int(height)):
            return True
        if not self._cp.has_section("Last"):
            self._cp.add_section("Last")
        self._cp.set("Last", "Width", str(int(width)))
        self._cp.set("Last", "Height", str(int(height)))
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                self._cp.write(f)
            return True
        except OSError:
            return False
