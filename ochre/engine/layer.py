# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
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


MASK_SUFFIX = "~mask"
PASS_THROUGH = "PassThrough"

# What a leaf layer IS. A raster layer holds pixels; an adjustment layer
# holds an adjustment (effect name + parameters) applied to everything below
# it; a fill layer generates its content (a colour or a gradient) at any
# size. Adjustment and fill layers are Photoshop's; they keep edits
# re-editable instead of baked in.
RASTER, ADJUSTMENT, FILL, TEXT, SMART = "raster", "adjustment", "fill", "text", "smart"
GENERATED = (FILL, TEXT, SMART)   # kinds whose pixels are made from params, not stored
SRC_SUFFIX = "~src"


class Layer(LayerNode):
    """A leaf. Declares which planes its cells carry, and which are truth."""

    def __init__(self, name="", planes=("rgba",), authoritative=None, **kw):
        super().__init__(name=name, **kw)
        self.kind = RASTER
        self.params = {}          # adjustment / fill settings
        self.clip = False         # clipped to the layer below (a clipping mask)
        self.mask = None          # LayerMask, or None
        self.mask_enabled = True
        self.smart_source = None  # SmartSource, for a smart object
        self.styles = {}          # layer styles, engine/layerstyles.py
        # Blend If (Photoshop's Blending Options): this layer's and the
        # underlying image's luminance ranges, each (black lo, black hi,
        # white lo, white hi); None means off.
        self.blend_if = None
        # Photoshop's Fill opacity: scales the layer's own pixels but not its
        # layer styles (a stroke stays at full strength at Fill 0).
        self.fill_opacity = 255
        # A vector mask: a Path whose inside shows the layer, resolution-
        # independent (it follows resizes exactly). Never mutated in place --
        # geometry changes replace it -- so history can hold references.
        self.vector_mask = None
        self.vector_mask_enabled = True
        # Knockout: "none", "shallow" (punch through to the bottom of the
        # enclosing group) or "deep" (to the bottom layer of the image).
        self.knockout = "none"
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


class LayerMask(Layer):
    """A layer's mask: greyscale, white shows the layer, black hides it.

    It is a Layer in its own right -- its cells sit in the document's cell
    map under the owner's id plus MASK_SUFFIX -- so every brush, gradient,
    fill, filter, undo step and the file format work on a mask with no
    special case at all. It is never in the tree; Document.layer() finds it
    through its owner.
    """

    is_mask = True

    def __init__(self, owner):
        super().__init__(name=owner.name + " (mask)", planes=("rgba",),
                         layer_id=owner.id + MASK_SUFFIX)
        self.owner = owner

    def effectively_visible(self):
        return False


class SmartSource(Layer):
    """A smart object's original pixels, at their own size.

    The smart object layer renders them through its transform and smart
    filters, so scaling down and back up loses nothing. Stored like a mask:
    cells keyed by the owner's id + SRC_SUFFIX, found through the owner.
    """

    is_source = True

    def __init__(self, owner, size=None):
        super().__init__(name=owner.name + " (source)", planes=("rgba",),
                         layer_id=owner.id + SRC_SUFFIX)
        self.owner = owner
        self.size = tuple(size) if size else None     # (w, h); None = canvas size

    def effectively_visible(self):
        return False


class Channel(Layer):
    """A saved selection: GIMP's channels, Photoshop's alpha channels.

    Greyscale, white selected; stored like a mask (its cells live in the
    document's cell map under its own id), so every geometry change, undo
    step and the file format carry it for free. Lives in doc.channels, not
    in the layer tree.
    """

    is_channel = True

    def __init__(self, name="Alpha", layer_id=None):
        super().__init__(name=name, planes=("rgba",), layer_id=layer_id)

    def effectively_visible(self):
        return False


class LayerGroup(LayerNode):
    """A branch. Owns an ordered list of children, bottom-first."""

    def __init__(self, name="", children=(), **kw):
        # Pass Through by default, as in Photoshop: the group is only an
        # organiser and its layers blend with everything beneath it. Any
        # other mode ISOLATES the group -- its layers composite among
        # themselves first, and the result blends in with the group's mode
        # and opacity -- which is what GIMP does and what a PSD group marked
        # anything but "pass" means.
        kw.setdefault("blend", PASS_THROUGH)
        super().__init__(name=name, **kw)
        self.children = []
        # An artboard or a frame: the group's picture cut to a region
        # (engine/regions.py), or None for an ordinary group.
        self.region = None
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
