# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The command surface a script may drive: the headless CLI, the control
server and the MCP bridge all go through here.

It is the controller's RECORDABLE whitelist and nothing more. That list was
drawn up so an actions file -- data -- could be replayed safely; a command
line, a socket and an agent are data in exactly the same sense, so they get
the same list rather than a second one that could drift from it.

A command is written the way a person types it:

    image.resize width=100 height=50
    image.resize width=100              (the height keeps the aspect)
    filter.gaussian_blur radius=3
    color.primary #ff8000
    resize_image 100 50

The dotted names are aliases for the recordable method names, which are
accepted too. Values are JSON where they parse as JSON (numbers, true,
[255, 0, 0, 255]), "#rrggbb" or "#rrggbbaa" is a color, and anything else is
a plain string.

Qt-free, like the controller it drives.
"""

import inspect
import json
import os
import re
import shlex

from .controller import RECORDABLE, Controller, _tuples


# Dotted names, grouped the way the menus are. Every RECORDABLE method has
# exactly one; tests/test_headless.py holds the two lists together.
ALIASES = {
    "image.resize": "resize_image",
    "image.canvas_size": "resize_canvas",
    "image.crop_to_selection": "crop_to_selection",
    "image.flip": "flip_image",
    "image.rotate": "rotate_image",
    "image.rotate_by": "rotate_image_by",
    "image.content_aware_scale": "content_aware_scale",
    "image.straighten": "straighten",
    "image.flatten": "flatten",
    "image.slices_from_guides": "slices_from_guides",
    "effect.apply": "apply_effect",
    "edit.fill": "fill_selection",
    "edit.erase": "erase_selection",
    "edit.content_aware_fill": "inpaint_selection",
    "select.all": "select_all",
    "select.none": "deselect",
    "select.invert": "invert_selection",
    "select.grow": "grow_selection",
    "select.shrink": "shrink_selection",
    "select.border": "border_selection",
    "select.feather": "feather_selection",
    "select.round": "round_selection",
    "select.remove_holes": "remove_selection_holes",
    "select.save": "save_selection",
    "select.load": "load_selection",
    "select.refine": "refine_selection",
    "select.sky": "select_sky",
    "layer.new": "add_layer",
    "layer.duplicate": "duplicate_layer",
    "layer.merge_down": "merge_down",
    "layer.flip": "flip_layer",
    "layer.rotate": "rotate_layer",
    "layer.scale": "scale_layer",
    "layer.offset": "offset_layer",
    "layer.align": "align_layer",
    "layer.add_mask": "add_layer_mask",
    "layer.delete_mask": "delete_layer_mask",
    "color.primary": "set_primary",
    "color.secondary": "set_secondary",
    "view.add_guide": "add_guide",
    "view.clear_guides": "clear_guides",
}

# Arguments that take a live object (a layer, a group). A script has no
# handle on one, and a string where an object belongs fails somewhere deep;
# every command acts on the active layer instead, as a recorded action does.
OBJECT_PARAMS = frozenset(("layer", "parent"))

# Upper bounds on what one command may ask for. An agent that asks for a
# 100000 x 100000 canvas should get an error, not take the machine down.
MAX_SIDE = 30000
MAX_PIXELS = 50_000_000
MAX_RADIUS = 1000
MAX_SCALE = 64.0

# Effect properties that name a file the effect reads. Every TextProperty of
# every effect is either here or in NOT_FILES; the test that enumerates them
# fails on a new one, so a new file-reading parameter cannot slip past the
# control server's folder check unnoticed.
FILE_PROPS = frozenset(("file",))
NOT_FILES = frozenset(("ranges", "pins", "stack"))

_COLOR = re.compile(r"^#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


class CommandError(ValueError):
    """A command that cannot run, with a message meant for a person."""


def _value(text):
    m = _COLOR.match(text)
    if m:
        rgb = bytes.fromhex(m.group(1))
        alpha = int(m.group(2), 16) if m.group(2) else 255
        return (rgb[0], rgb[1], rgb[2], alpha)
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse(line):
    """'image.resize width=100' -> ('image.resize', [], {'width': 100})."""
    try:
        words = shlex.split(line)
    except ValueError as exc:
        raise CommandError("cannot parse %r: %s" % (line, exc)) from exc
    if not words:
        raise CommandError("empty command")
    name, args, kwargs = words[0], [], {}
    for word in words[1:]:
        key, eq, raw = word.partition("=")
        if eq and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            kwargs[key] = _value(raw)
        else:
            args.append(_value(word))
    return name, args, kwargs


def resolve(name):
    """(recordable method name, effect name or None) for a command name."""
    if not isinstance(name, str):
        raise CommandError("a command name is a string")
    if name.startswith("filter."):
        return "apply_effect", name[len("filter."):]
    op = ALIASES.get(name, name)
    if op not in RECORDABLE or not hasattr(Controller, op):
        raise CommandError("unknown command %r (see --list)" % name)
    return op, None


def prepare(ctl, name, args=(), kwargs=None):
    """Check a command completely before anything runs. Returns
    (op, args, kwargs) ready to call, or raises CommandError."""
    op, effect = resolve(name)
    args = list(_tuples(list(args or [])))
    kwargs = dict(_tuples(dict(kwargs or {})))
    if op == "apply_effect":
        args, kwargs = _effect_call(ctl, effect, args, kwargs)
    if op in ("resize_image", "content_aware_scale") and not args and ctl.doc is not None:
        _keep_aspect(ctl, name, kwargs)
    sig = inspect.signature(getattr(ctl, op))
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError as exc:
        raise CommandError("%s: %s (takes %s)" % (name, exc, _params(sig))) from exc
    for key in bound.arguments:
        if key in OBJECT_PARAMS:
            raise CommandError("%s: %r is not available to a script; commands act on the "
                               "active layer" % (name, key))
    _check_limits(ctl, name, op, bound.arguments)
    return op, list(bound.args), dict(bound.kwargs)


def _effect_call(ctl, effect, args, kwargs):
    """apply_effect's arguments: the effect name, and its parameters given
    either as params={...} or loose (filter.gaussian_blur radius=3)."""
    if effect is None:
        if args:
            effect, args = args[0], args[1:]
        else:
            effect = kwargs.pop("name", None)
    if args:
        raise CommandError("%s: give effect parameters as name=value" % effect)
    params = dict(kwargs.pop("params", None) or {})
    params.update(kwargs)
    if not isinstance(effect, str) or effect not in ctl.effects.names():
        raise CommandError("unknown effect %r (see --list)" % (effect,))
    fx = ctl.effects.create(effect)
    unknown = sorted(k for k in params if k not in fx.props)
    if unknown:
        raise CommandError("%s: no parameter %s (has %s)"
                           % (effect, ", ".join(unknown), ", ".join(fx.props.order) or "none"))
    return [effect], {"params": params} if params else {}


def _keep_aspect(ctl, name, kwargs):
    """image.resize width=100: the height follows, as in the Image Size
    dialog with the chain linked."""
    w, h = kwargs.get("width"), kwargs.get("height")
    if (w is None) == (h is None):
        return
    if w is not None:
        kwargs["height"] = max(1, round(_num(name, "width", w) * ctl.doc.height / ctl.doc.width))
    else:
        kwargs["width"] = max(1, round(_num(name, "height", h) * ctl.doc.width / ctl.doc.height))


def _num(name, key, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CommandError("%s: %s must be a number, not %r" % (name, key, v))
    return v


def _check_limits(ctl, name, op, a):
    if op in ("resize_image", "resize_canvas", "content_aware_scale"):
        w, h = _num(name, "width", a.get("width")), _num(name, "height", a.get("height"))
        if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE) or w * h > MAX_PIXELS:
            raise CommandError("%s: %dx%d is outside the limit (sides 1..%d, at most %d pixels)"
                               % (name, w, h, MAX_SIDE, MAX_PIXELS))
    for key in ("radius", "feather", "smooth"):
        if key in a and abs(_num(name, key, a[key])) > MAX_RADIUS:
            raise CommandError("%s: %s is limited to %d" % (name, key, MAX_RADIUS))
    if op == "scale_layer":
        for key in ("factor_x", "factor_y"):
            v = a.get(key)
            if v is not None and not 0 < _num(name, key, v) <= MAX_SCALE:
                raise CommandError("%s: %s must be above 0 and at most %g" % (name, key, MAX_SCALE))


def file_targets(ctl, op, args, kwargs):
    """Files a prepared command would read: today, only effect parameters
    such as Color Lookup's .cube file, including inside a Filter Gallery
    stack. The control server judges each against its folders."""
    if op != "apply_effect":
        return []
    named = inspect.signature(getattr(ctl, op)).bind(*args, **kwargs).arguments
    return _effect_files(ctl, named["name"], named.get("params") or {}, depth=0)


def _effect_files(ctl, effect, params, depth):
    from ..engine.photo import FilterStack
    fx = ctl.effects.create(effect, **dict(params))
    if fx is None:
        return []
    out = []
    for prop in fx.props:
        value = fx.props.get(prop.name)
        if prop.name in FILE_PROPS and isinstance(value, str) and value.strip():
            out.append(value.strip())
        elif isinstance(fx, FilterStack) and prop.name == "stack" and depth < 4:
            for item in FilterStack.parse(value):
                out.extend(_effect_files(ctl, item["name"], item.get("params") or {}, depth + 1))
    return out


def run(ctl, name, args=(), kwargs=None):
    """Prepare and run one command. Returns {"command", "changed"}."""
    if ctl.doc is None:
        raise CommandError("no document is open")
    op, args, kwargs = prepare(ctl, name, args, kwargs)
    return invoke(ctl, name, op, args, kwargs)


def invoke(ctl, name, op, args, kwargs):
    """Run a command prepare() accepted."""
    before = (ctl.doc.width, ctl.doc.height)
    try:
        result = getattr(ctl, op)(*args, **kwargs)
    except CommandError:
        raise
    except Exception as exc:                               # noqa: BLE001
        raise CommandError("%s failed: %s" % (name, exc)) from exc
    if before != (ctl.doc.width, ctl.doc.height):
        ctl._after_geometry_change()
    return {"command": name, "changed": result is not None and result is not False}


def _params(sig):
    out = []
    for p in sig.parameters.values():
        if p.name in OBJECT_PARAMS:
            continue
        if p.default is inspect.Parameter.empty:
            out.append(p.name)
        else:
            try:
                default = json.dumps(p.default)
            except TypeError:
                default = repr(p.default)
            out.append("%s=%s" % (p.name, default))
    return " ".join(out) or "nothing"


def describe(ctl=None):
    """Every command, as [{"name", "method", "params"}], for --list and
    for list_commands. Effects come separately, from effect_names()."""
    by_op = {op: alias for alias, op in ALIASES.items()}
    out = []
    for op in RECORDABLE:
        if not hasattr(Controller, op):
            continue
        sig = inspect.signature(getattr(Controller, op))
        params = [p for p in sig.parameters.values() if p.name != "self"]
        out.append({"name": by_op.get(op, op), "method": op,
                    "params": _params(sig.replace(parameters=params))})
    return out


def effect_names(ctl):
    return [{"name": n, "label": ctl.effects.label(n),
             "params": list(ctl.effects.create(n).props.order)} for n in ctl.effects.names()]


def default_paths():
    """Where a script finds Ochre's data, addons and state -- the same
    places the window uses."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    state = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
    return {"data": os.path.join(root, "data"),
            "addons": [os.path.join(root, "addons"), os.path.join(config, "ochre", "addons")],
            "state": os.path.join(state, "ochre")}


def headless_controller():
    """A Controller with no window and no one to ask.

    It reads the user's preferences and loads the addons they already
    trusted (nothing new is trusted: there is no one to ask), but writes no
    session state -- a scripted run must not change what the next window
    opens at."""
    from ..engine.settings import Settings
    paths = default_paths()
    ctl = Controller(Settings(paths["data"]), paths["data"], addon_dirs=paths["addons"],
                     state_dir=paths["state"], trust_callback=lambda addon, digest: False)
    ctl.session.path = None
    ctl.load_addons()
    return ctl
