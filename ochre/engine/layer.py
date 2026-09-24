# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""The layer tree.

Layers are document-level and ordered; they are *not* where pixels live. A
layer describes identity and compositing intent -- name, opacity, blend mode,
visibility, which planes its pixels carry -- while the pixels themselves live
in cells, one per (layer, frame) intersection. That separation is what lets a
layer called "Body" persist across every frame of an animation instead of a
multi-frame document degenerating into a pile of layers pretending to be
frames.

Groups exist from the start. Retrofitting a tree onto a flat list touches
history, the file format, selection and every tool, so the tree is built now
even though the UI for it can arrive whenever.

Group semantics are "isolated": a group composites its children into a scratch
buffer, then blends that result into its parent using the group's own opacity,
blend mode and visibility. Pass-through groups are deliberately not supported
-- they force the compositor to interleave a group's children into the parent
stack, which defeats the below/above cache that makes stroke compositing
independent of layer count.
"""

import itertools

_ids = itertools.count(1)


def _new_id(prefix):
    return "%s%04d" % (prefix, next(_ids))


class LayerNode:
    """Shared identity and compositing state."""

    def __init__(self, name="", opacity=255, visible=True, blend="Normal",
                 layer_id=None):
        self.id = layer_id or _new_id("L")
        self.name = name
        # Integer 0..255, matching the accelerator's integer contract. A float
        # here would silently reintroduce the floating-point ambiguity the
        # whole arithmetic design exists to avoid.
        self.opacity = int(opacity)
        self.visible = bool(visible)
        self.blend = blend
        self.parent = None
        self.meta = {}          # str -> str, round-tripped by the file format

    @property
    def is_group(self):
        return False

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def depth(self):
        return sum(1 for _ in self.ancestors())

    def effectively_visible(self):
        """Visible, and not inside a hidden group."""
        return self.visible and all(a.visible for a in self.ancestors())

    def __repr__(self):
        return "%s(id=%r, name=%r)" % (type(self).__name__, self.id, self.name)


class Layer(LayerNode):
    """A leaf. Declares which planes its cells carry, and which are truth."""

    def __init__(self, name="", planes=("rgba",), authoritative=None, **kw):
        super().__init__(name=name, **kw)
        self.planes = tuple(planes)
        self.authoritative = tuple(authoritative if authoritative is not None
                                   else (self.planes[0],))
        for plane in self.authoritative:
            if plane not in self.planes:
                raise ValueError("authoritative plane %r not in planes %r"
                                 % (plane, self.planes))

    @property
    def index_locked(self):
        return "index" in self.authoritative

    def lock_to_index(self):
        """Make this layer index-locked: index becomes truth, rgba a cache."""
        if "index" not in self.planes:
            self.planes = ("index",) + self.planes
        self.authoritative = tuple(["index"] + [p for p in self.authoritative
                                                if p not in ("index", "rgba")])

    def unlock_from_index(self):
        """Drop back to plain RGBA. The index plane is discarded by the caller."""
        self.planes = tuple(p for p in self.planes if p != "index")
        self.authoritative = ("rgba",) + tuple(p for p in self.authoritative
                                               if p not in ("index", "rgba"))

    def add_plane(self, name):
        """Attach a named non-colour plane. Authoritative by default -- a
        plane nobody can reconstruct must be saved and undone."""
        if name in self.planes:
            return False
        self.planes = self.planes + (name,)
        if name not in self.authoritative:
            self.authoritative = self.authoritative + (name,)
        return True


class LayerGroup(LayerNode):
    """A branch. Owns an ordered list of children, bottom-first."""

    def __init__(self, name="", children=(), **kw):
        super().__init__(name=name, **kw)
        self.children = []
        for child in children:
            self.add(child)

    @property
    def is_group(self):
        return True

    # ---- structure -------------------------------------------------------

    def add(self, node, index=None):
        if node is self or node in node.ancestors():
            raise ValueError("cannot add a node into itself")
        if any(node is a for a in self.ancestors()) or node is self:
            raise ValueError("cannot add an ancestor as a child")
        if node.parent is not None:
            node.parent.remove(node)
        node.parent = self
        if index is None:
            self.children.append(node)
        else:
            self.children.insert(max(0, min(index, len(self.children))), node)
        return node

    def remove(self, node):
        try:
            self.children.remove(node)
        except ValueError:
            return False
        node.parent = None
        return True

    def index_of(self, node):
        for i, child in enumerate(self.children):
            if child is node:
                return i
        return None

    # ---- traversal -------------------------------------------------------

    def walk(self, include_self=False):
        """Depth-first, bottom-first. Groups yielded before their children."""
        if include_self:
            yield self
        for child in self.children:
            yield child
            if child.is_group:
                yield from child.walk()

    def leaves(self):
        return [n for n in self.walk() if not n.is_group]

    def flat(self, depth=0):
        """[(depth, node)] for a panel. Top-most first, as a UI shows it."""
        out = []
        for child in reversed(self.children):
            out.append((depth, child))
            if child.is_group:
                out.extend(child.flat(depth + 1))
        return out

    def find(self, layer_id):
        for node in self.walk(include_self=True):
            if node.id == layer_id:
                return node
        return None

    def __len__(self):
        return len(self.children)

    def __bool__(self):
        # Without this, __len__ makes an EMPTY GROUP FALSY, and every
        # `parent or default` idiom silently discards it. A group object
        # always exists regardless of how many children it holds.
        return True

    def __iter__(self):
        return iter(self.children)
