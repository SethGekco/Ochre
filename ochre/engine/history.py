# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The history stack: linear, index-addressable, frame-aware.

Linear, not a branching tree. Paint.NET's history is linear and its
discoverability is a feature -- a list you can click any point in beats a
graph nobody can navigate.

Because commands generate their own inverse (see commands.py), undo and redo
are one code path: the stack holds a command, calls undo() on it, and stores
the opposite it returns back in the same slot. The slot always holds "the
thing that reverses the current state", whichever direction that happens to
be. There is no second implementation that could drift.

One subtlety that matters in a multi-frame document: if an entry targets a
frame other than the current one, the document switches to that frame before
applying. Anything else produces invisible undos, which is the single most
confusing thing a multi-frame editor can do to someone.
"""


class HistoryEntry:
    """One slot: a command, plus what the UI needs to describe it."""

    __slots__ = ("command", "label", "frame_id")

    def __init__(self, command, label=None, frame_id=None):
        self.command = command
        self.label = label or getattr(command, "label", "Change")
        self.frame_id = frame_id

    def nbytes(self):
        return self.command.nbytes()

    def __repr__(self):
        return "HistoryEntry(%r, %d bytes)" % (self.label, self.nbytes())


class HistoryStack:
    """Undo/redo for one document."""

    def __init__(self, settings=None):
        self.entries = []
        self.position = -1          # index of the last APPLIED entry
        self._nbytes = 0
        self._group = None
        self._group_depth = 0
        s = settings
        self.max_bytes = self._cfg(s, "MaxBytes", 1 << 30)
        self.max_entries = self._cfg(s, "MaxEntries", 200)
        self.min_entries = self._cfg(s, "MinEntries", 10)
        self.on_change = None       # optional callback, for a UI to subscribe

    @staticmethod
    def _cfg(settings, key, default):
        if settings is None:
            return default
        value = settings.get("History", key)
        return default if value is None else value

    # ---- state -----------------------------------------------------------

    @property
    def can_undo(self):
        return self.position >= 0

    @property
    def can_redo(self):
        return self.position < len(self.entries) - 1

    def nbytes(self):
        return self._nbytes

    def __len__(self):
        return len(self.entries)

    def labels(self):
        """Entry labels in order, for a history panel."""
        return [e.label for e in self.entries]

    def clear(self):
        self.entries = []
        self.position = -1
        self._nbytes = 0
        self._notify()

    def _notify(self):
        if self.on_change is not None:
            self.on_change(self)

    # ---- recording -------------------------------------------------------

    def push(self, command, label=None, frame_id=None):
        """Record an already-applied change.

        Everything after the current position is discarded -- the standard
        linear-history contract: making a new change after undoing abandons
        the redo branch.
        """
        if command is None:
            return None
        if self._group is not None:
            self._group.append(command)
            return None

        del self.entries[self.position + 1:]
        entry = HistoryEntry(command, label, frame_id)
        self.entries.append(entry)
        self.position = len(self.entries) - 1
        self._nbytes = sum(e.nbytes() for e in self.entries)
        self._evict()
        self._notify()
        return entry

    def _evict(self):
        """Trim from the OLDEST end, but never below min_entries.

        The floor matters: a document whose single delta exceeds max_bytes
        would otherwise evict itself down to nothing and leave the user with
        no undo at all, which is worse than exceeding a soft memory target.
        """
        while len(self.entries) > self.max_entries and len(self.entries) > self.min_entries:
            self._nbytes -= self.entries[0].nbytes()
            del self.entries[0]
            self.position -= 1
        while (self._nbytes > self.max_bytes
               and len(self.entries) > self.min_entries):
            self._nbytes -= self.entries[0].nbytes()
            del self.entries[0]
            self.position -= 1
        self.position = max(self.position, -1)

    # ---- grouping --------------------------------------------------------

    def begin_group(self):
        """Collect subsequent pushes into one entry. Re-entrant."""
        self._group_depth += 1
        if self._group is None:
            self._group = []
        return self

    def end_group(self, label="Multiple changes"):
        """Close the group and push it as a single undoable entry."""
        self._group_depth -= 1
        if self._group_depth > 0:
            return None
        children, self._group = self._group, None
        self._group_depth = 0
        if not children:
            return None
        if len(children) == 1:
            return self.push(children[0], label)
        from .commands import CompoundCommand
        return self.push(CompoundCommand(children, label), label)

    def abort_group(self):
        """Discard a group without pushing. For a cancelled interaction."""
        self._group = None
        self._group_depth = 0

    # ---- traversal -------------------------------------------------------

    def _apply(self, index, doc):
        """Run the command at index and store the opposite back in its slot."""
        entry = self.entries[index]
        if entry.frame_id is not None and doc is not None:
            target = doc.frame_index(entry.frame_id)
            # Switch first, or the change happens somewhere nobody is looking.
            if target is not None and target != doc.current:
                doc.set_current_frame(target)
        opposite, rect = entry.command.undo(doc)
        if opposite is None:
            return None
        before = entry.nbytes()
        entry.command = opposite
        self._nbytes += entry.nbytes() - before
        return rect

    def undo(self, doc):
        if not self.can_undo:
            return None
        rect = self._apply(self.position, doc)
        self.position -= 1
        self._notify()
        return rect

    def redo(self, doc):
        if not self.can_redo:
            return None
        rect = self._apply(self.position + 1, doc)
        self.position += 1
        self._notify()
        return rect

    def goto(self, index, doc):
        """Move to any point in the history. Backs a clickable history panel."""
        index = max(-1, min(int(index), len(self.entries) - 1))
        covered = None
        while self.position > index:
            r = self.undo(doc)
            covered = r if covered is None else covered.union(r)
        while self.position < index:
            r = self.redo(doc)
            covered = r if covered is None else covered.union(r)
        return covered
