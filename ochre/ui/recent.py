# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The recent-files list. Plain text, one path per line, newest first.

Qt-free so it can be tested without a display, and a text file rather than
QSettings so it can be read with `cat` and fixed with an editor, which is the
same reason every other setting in this project is an INI.
"""

import os


class RecentFiles:
    def __init__(self, state_dir, limit=10):
        self.path = os.path.join(state_dir, "recent.txt") if state_dir else None
        self.limit = int(limit)
        self.items = []
        self.load()

    def load(self):
        self.items = []
        if not self.path or not os.path.isfile(self.path):
            return self.items
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line and line not in self.items:
                        self.items.append(line)
        except OSError:
            pass
        self.items = self.items[:self.limit]
        return self.items

    def save(self):
        if not self.path:
            return False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.items) + ("\n" if self.items else ""))
            return True
        except OSError:
            return False

    def add(self, path):
        path = os.path.abspath(path)
        self.items = [path] + [p for p in self.items if p != path]
        self.items = self.items[:self.limit]
        self.save()
        return self.items

    def remove(self, path):
        self.items = [p for p in self.items if p != path]
        self.save()

    def existing(self):
        """Only paths that are still there. Stale entries are kept on disk
        until the user picks one and it fails -- a file on an unmounted
        drive is not gone, it is away."""
        return [p for p in self.items if os.path.exists(p)]
