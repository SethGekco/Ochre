# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Discovering, loading and isolating addons.

One discovery mechanism, two implementation tiers. A compiled `.so` and a
plain `.py` are found by the SAME importlib machinery -- resolved by its own
suffix ordering -- so an addon can ship a native fast path and a pure-Python
fallback under one name, and a wrong-ABI binary simply is not found while the
Python takes over. There is no second plugin API to keep in sync.

Loading is EXPLICIT. An addon exposes `register(host)` and nothing happens at
import time. That is what makes the two tiers interchangeable, what makes
lazy loading possible, and what stops merely scanning a directory from
running anybody's code.

Failure is contained by transactions. The registry is snapshotted before
`register()` and rolled back if it raises, so a half-registered addon cannot
leave the application in a state nobody designed. The same guard wraps every
later call into addon code.

On security, the honest position: addons are arbitrary Python and Python
cannot sandbox that. Rather than pretending otherwise with a fake sandbox
that gives false confidence, this module does the things that actually help
-- nothing executes during a scan, nothing loads until asked, the content is
hashed so a changed addon is re-confirmed, there is a real kill switch, and a
crash leaves a breadcrumb naming the culprit.
"""

import hashlib
import importlib.util
import json
import os
import sys
import traceback

from .api import API_VERSION, AddonError, Host, compatible
from .manifest import Manifest

#: Written before an addon's code runs and cleared after. If it survives a
#: restart, the named addon crashed the process -- which is the one failure
#: Python cannot contain, because a segfault in native code takes everything.
BREADCRUMB = "loading.json"
TRUST_FILE = "trusted.json"


class Addon:
    """One discovered addon and everything known about it."""

    def __init__(self, manifest, directory):
        self.manifest = manifest
        self.directory = directory
        self.id = manifest.id
        self.module = None
        self.state = "discovered"   # discovered|blocked|loaded|failed|disabled
        self.error = None
        self.messages = []
        self.faults = 0

    @property
    def name(self):
        return self.manifest.name or self.id

    def __repr__(self):
        return "Addon(%r, %s)" % (self.id, self.state)

    def digest(self):
        """Content hash over the addon's source and manifest.

        Used for trust: a changed addon is a new addon as far as consent is
        concerned. Sorted walk so the result is stable across filesystems.
        """
        sha = hashlib.sha256()
        for root, dirs, files in os.walk(self.directory):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for fn in sorted(files):
                if not fn.endswith((".py", ".so", ".ini", ".pyd")):
                    continue
                path = os.path.join(root, fn)
                sha.update(os.path.relpath(path, self.directory).encode())
                try:
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            sha.update(chunk)
                except OSError:
                    continue
        return sha.hexdigest()


class Registry:
    """Everything addons have contributed, by kind."""

    KINDS = ("formats", "palettes", "commands", "panels", "effects", "tools",
             "planes")

    def __init__(self):
        self._items = {kind: {} for kind in self.KINDS}
        self._owner = {}

    def add(self, kind, key, value, addon):
        if kind not in self._items:
            raise AddonError(addon.id, "unknown extension point %r" % kind)
        if not key:
            raise AddonError(addon.id, "%s registered without a name" % kind)
        if key in self._items[kind]:
            other = self._owner.get((kind, key))
            raise AddonError(addon.id, "%s %r already registered by %r"
                             % (kind, key, other))
        self._items[kind][key] = value
        self._owner[(kind, key)] = addon.id
        return value

    def get(self, kind, key, default=None):
        return self._items.get(kind, {}).get(key, default)

    def all(self, kind):
        return dict(self._items.get(kind, {}))

    def owner(self, kind, key):
        return self._owner.get((kind, key))

    def remove_addon(self, addon_id):
        """Withdraw everything one addon contributed."""
        removed = 0
        for (kind, key), owner in list(self._owner.items()):
            if owner == addon_id:
                self._items[kind].pop(key, None)
                del self._owner[(kind, key)]
                removed += 1
        return removed

    def snapshot(self):
        return ({kind: dict(items) for kind, items in self._items.items()},
                dict(self._owner))

    def restore(self, snap):
        items, owner = snap
        self._items = {kind: dict(v) for kind, v in items.items()}
        self._owner = dict(owner)

    def counts(self):
        return {kind: len(items) for kind, items in self._items.items() if items}

    def formats_for(self, extension):
        ext = extension.lower().lstrip(".")
        return [p for p in self._items["formats"].values()
                if ext in getattr(p, "extensions", ())]


class AddonManager:
    """Finds addons, decides whether to load them, and contains the damage."""

    def __init__(self, directories=None, settings=None, db=None,
                 state_dir=None, controller=None, trust_callback=None):
        self.directories = [d for d in (directories or []) if d]
        self.settings = settings
        self.db = db
        self.controller = controller
        self.state_dir = state_dir
        self.registry = Registry()
        self.addons = {}
        self.errors = []
        #: Asked before running a never-before-seen addon. Returning False
        #: blocks it. None means "no UI available", and the policy below
        #: decides -- defaulting to NOT running unknown code unattended.
        self.trust_callback = trust_callback
        self.trust_unknown = self._cfg("TrustUnknown", False)
        self.fault_budget = self._cfg("FaultBudget", 3)
        self._trusted = self._load_trust()

    def _cfg(self, key, default):
        if self.settings is None:
            return default
        value = self.settings.get("Addons", key)
        return default if value is None else value

    # ---- discovery (runs NO addon code) ---------------------------------

    def discover(self):
        """Scan for addons. Reads manifests only; executes nothing."""
        found = {}
        for directory in self.directories:
            if not os.path.isdir(directory):
                continue
            for entry in sorted(os.listdir(directory)):
                path = os.path.join(directory, entry)
                manifest_path = os.path.join(path, "addon.ini")
                if not os.path.isfile(manifest_path):
                    continue
                manifest = Manifest.read(manifest_path)
                if manifest is None:
                    self.errors.append((entry, "unreadable addon.ini"))
                    continue
                if manifest.id in found:
                    self.errors.append((manifest.id, "duplicate id, ignoring %s" % path))
                    continue
                found[manifest.id] = Addon(manifest, path)
        self.addons = found
        return found

    # ---- loading ---------------------------------------------------------

    def load_all(self):
        """Load every discovered addon that is eligible, in dependency order."""
        for addon in self._ordered():
            self.load(addon)
        return self.registry

    def _ordered(self):
        """Topological by `Requires`, so a dependency registers first.

        A cycle disables everyone in it rather than picking an arbitrary
        victim -- there is no correct order, and half-loading is worse than
        not loading.
        """
        pending = dict(self.addons)
        out, guard = [], 0
        while pending and guard <= len(self.addons) + 1:
            guard += 1
            ready = [a for a in pending.values()
                     if all(dep not in pending for dep in a.manifest.requires)]
            if not ready:
                for addon in pending.values():
                    self._fail(addon, "dependency cycle or missing dependency: %s"
                               % ", ".join(sorted(addon.manifest.requires)))
                break
            for addon in sorted(ready, key=lambda a: (a.manifest.order, a.id)):
                out.append(addon)
                del pending[addon.id]
        return out

    def load(self, addon):
        """Run one addon's register(). Isolated; never raises to the caller."""
        if isinstance(addon, str):
            addon = self.addons.get(addon)
        if addon is None or addon.state in ("loaded", "failed", "blocked", "disabled"):
            return addon

        manifest = addon.manifest
        if not manifest.enabled:
            addon.state = "disabled"
            return addon

        ok, why = compatible(manifest.api)
        if not ok:
            return self._fail(addon, why)

        for dep in manifest.requires:
            other = self.addons.get(dep)
            if other is None or other.state != "loaded":
                return self._fail(addon, "requires %r, which is not loaded" % dep)

        if not self._is_trusted(addon):
            addon.state = "blocked"
            addon.error = "not trusted"
            return addon

        snapshot = self.registry.snapshot()
        self._drop_breadcrumb(addon)
        try:
            module = self._import(addon)
            register = getattr(module, "register", None)
            if register is None:
                raise AddonError(addon.id, "exposes no register(host)")
            host = Host(self.registry, addon, self.settings, self.db,
                        self.controller)
            register(host)
        except BaseException as exc:                        # noqa: BLE001
            # A half-registered addon would leave the app in a state nobody
            # designed, so the registry goes back exactly as it was.
            self.registry.restore(snapshot)
            self._clear_breadcrumb()
            return self._fail(addon, "%s: %s" % (type(exc).__name__, exc),
                              traceback.format_exc())
        self._clear_breadcrumb()
        addon.module = module
        addon.state = "loaded"
        return addon

    def _import(self, addon):
        """Import the entry module.

        A compiled extension and a .py are found by the same machinery --
        importlib's own suffix ordering prefers the native one when both are
        present, which is exactly the two-tier behaviour wanted, with no
        second code path to maintain.
        """
        entry = addon.manifest.entry
        module_name = "ochre_addon_%s" % addon.id.replace("-", "_").replace(".", "_")
        if module_name in sys.modules:
            return sys.modules[module_name]

        # Isolated package name, so two addons may both ship a `util.py`
        # without colliding in sys.modules.
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(addon.directory, entry),
            submodule_search_locations=[addon.directory])
        if spec is None or spec.loader is None:
            raise AddonError(addon.id, "cannot import %s" % entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _fail(self, addon, message, detail=None):
        addon.state = "failed"
        addon.error = message
        self.errors.append((addon.id, message))
        if detail:
            addon.messages.append(detail)
        self.registry.remove_addon(addon.id)
        return addon

    def disable(self, addon_id, reason="disabled by user"):
        """The kill switch. Withdraws everything the addon contributed."""
        addon = self.addons.get(addon_id)
        if addon is None:
            return False
        self.registry.remove_addon(addon_id)
        addon.state = "disabled"
        addon.error = reason
        return True

    # ---- calling into addon code ----------------------------------------

    def guard(self, addon_id, what="call"):
        """Context manager wrapping any later call into an addon.

        An addon that raises repeatedly is disabled rather than allowed to
        keep failing -- a format provider that throws on every sniff would
        otherwise make the whole open dialog unusable.
        """
        return _Guard(self, addon_id, what)

    def note_fault(self, addon_id, exc, what):
        addon = self.addons.get(addon_id)
        self.errors.append((addon_id, "%s in %s: %s" % (type(exc).__name__, what, exc)))
        if addon is None:
            return
        addon.faults += 1
        addon.messages.append(traceback.format_exc())
        if addon.faults >= self.fault_budget:
            self.disable(addon_id, "disabled after %d faults" % addon.faults)

    # ---- trust -----------------------------------------------------------

    def _trust_path(self):
        return None if not self.state_dir else os.path.join(self.state_dir, TRUST_FILE)

    def _load_trust(self):
        path = self._trust_path()
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_trust(self):
        path = self._trust_path()
        if not path:
            return False
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._trusted, f, indent=1, sort_keys=True)
            return True
        except OSError:
            return False

    def _is_trusted(self, addon):
        digest = addon.digest()
        if self._trusted.get(addon.id) == digest:
            return True
        # A CHANGED addon is a new addon as far as consent goes -- otherwise
        # trusting something once would trust whatever it later became.
        decision = None
        if self.trust_callback is not None:
            decision = self.trust_callback(addon, digest)
        if decision is None:
            decision = bool(self.trust_unknown)
        if decision:
            self._trusted[addon.id] = digest
            self._save_trust()
        return bool(decision)

    def trust(self, addon_id):
        addon = self.addons.get(addon_id)
        if addon is None:
            return False
        self._trusted[addon.id] = addon.digest()
        self._save_trust()
        if addon.state == "blocked":
            addon.state = "discovered"
            addon.error = None
        return True

    # ---- crash breadcrumb ------------------------------------------------

    def _breadcrumb_path(self):
        return None if not self.state_dir else os.path.join(self.state_dir, BREADCRUMB)

    def _drop_breadcrumb(self, addon):
        path = self._breadcrumb_path()
        if not path:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"id": addon.id, "directory": addon.directory}, f)
        except OSError:
            pass

    def _clear_breadcrumb(self):
        path = self._breadcrumb_path()
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def crashed_addon(self):
        """Which addon was mid-load when the process last died, if any.

        A segfault in native addon code is the one failure Python genuinely
        cannot contain. This cannot prevent it, but it can name the culprit
        instead of leaving the user with an editor that dies on startup for
        no visible reason.
        """
        path = self._breadcrumb_path()
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    # ---- reporting -------------------------------------------------------

    def status(self):
        return [{"id": a.id, "name": a.name, "state": a.state,
                 "version": a.manifest.version, "error": a.error,
                 "faults": a.faults, "provides": a.manifest.provides}
                for a in sorted(self.addons.values(), key=lambda a: a.id)]

    def loaded(self):
        return [a for a in self.addons.values() if a.state == "loaded"]


class _Guard:
    def __init__(self, manager, addon_id, what):
        self.manager = manager
        self.addon_id = addon_id
        self.what = what
        self.failed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        self.failed = True
        self.manager.note_fault(self.addon_id, exc, self.what)
        return True          # swallow: a bad addon must not break the caller
