#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The addon host: discovery, isolation, trust, and the two tiers.

The assertions that matter are the negative ones. Discovery must execute
NOTHING. A crashing addon must leave the registry byte-identical. A blocked
addon must contribute nothing. Those are what make it safe to let strangers'
code into the process at all.

Run: python3 tests/test_addons.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ochre.addons import API_VERSION, AddonManager, Manifest, compatible
from ochre.engine.ini import IniDB
from ochre.engine.settings import Settings


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def write_addon(root, ident, entry_body, **manifest):
    path = os.path.join(root, ident)
    os.makedirs(path, exist_ok=True)
    fields = {"Id": ident, "Name": ident.title(), "Version": "1.0.0",
              "Api": "1.0", "Entry": "__init__.py"}
    fields.update(manifest)
    with open(os.path.join(path, "addon.ini"), "w") as f:
        f.write("[Addon]\n")
        for key, value in fields.items():
            f.write("%s = %s\n" % (key, value))
    with open(os.path.join(path, "__init__.py"), "w") as f:
        f.write(entry_body)
    return path


GOOD = '''
class Fmt:
    name = "demo"
    extensions = ("demo",)
    can_read = True
    can_write = True
    def sniff(self, head, path): return head.startswith(b"DEMO")
    def load(self, path, host): return "loaded:" + path
    def save(self, document, path, host, options): return path

class Cmd:
    name = "demo.shout"
    label = "Shout"
    category = "Demo"
    def run(self, host, **kw): return "shouted"

REGISTERED = []

def register(host):
    REGISTERED.append(host.addon_id)
    host.register_format(Fmt())
    host.register_command(Cmd())
    host.register_plane("demo_depth")
    host.log("demo addon registered")
'''

# Registers something, THEN raises. The registry must not keep the partial
# work -- that is the whole point of the transaction.
PARTIAL = '''
class Fmt:
    name = "halfway"
    extensions = ("half",)
    can_read = True
    can_write = False
    def sniff(self, head, path): return False
    def load(self, path, host): return None
    def save(self, d, p, h, o): pass

def register(host):
    host.register_format(Fmt())
    host.register_plane("halfway_plane")
    raise RuntimeError("boom halfway through registering")
'''

IMPORT_BOOM = '''
raise ImportError("this addon cannot even be imported")

def register(host):
    pass
'''

NO_REGISTER = '''
VALUE = 1
'''

# Proves nothing runs at scan time: it would blow up on import if it did.
TRIPWIRE = '''
import os
with open(os.path.join(os.path.dirname(__file__), "EXECUTED"), "w") as f:
    f.write("import-time code ran")

def register(host):
    pass
'''

FAULTY_CALL = '''
class Fmt:
    name = "faulty"
    extensions = ("faulty",)
    can_read = True
    can_write = False
    def sniff(self, head, path): raise RuntimeError("sniff always explodes")
    def load(self, path, host): return None
    def save(self, d, p, h, o): pass

def register(host):
    host.register_format(Fmt())
'''


def main():
    root = tempfile.mkdtemp(prefix="ochre-addons-")
    state = os.path.join(root, "_state")

    # ---- manifests -------------------------------------------------------
    path = write_addon(root, "demo", GOOD, Provides="formats, commands",
                       Author="Someone", License="GPL-3.0")
    m = Manifest.read(os.path.join(path, "addon.ini"))
    check("manifest parses", m is not None and m.id == "demo")
    check("manifest reads the version", m.version == "1.0.0")
    check("manifest parses the API as a tuple", m.api == (1, 0))
    check("manifest splits Provides", m.provides == ("formats", "commands"))
    check("manifest defaults Enabled to on", m.enabled is True)

    with open(os.path.join(path, "addon.ini"), "a") as f:
        f.write("FutureKey = 42\nAnotherThing = yes\n")
    m = Manifest.read(os.path.join(path, "addon.ini"))
    check("unknown manifest keys are ignored", m is not None and m.id == "demo")

    check("a directory without addon.ini is not an addon",
          Manifest.read(os.path.join(root, "nope", "addon.ini")) is None)

    # ---- API compatibility ----------------------------------------------
    check("matching API accepted", compatible((1, 0))[0])
    check("older minor accepted", compatible((1,))[0])
    check("newer minor rejected", not compatible((1, 99))[0])
    check("different major rejected", not compatible((2, 0))[0])
    check("rejection explains itself", "provides" in compatible((2, 0))[1])

    # ---- discovery executes NOTHING -------------------------------------
    trip = write_addon(root, "tripwire", TRIPWIRE)
    mgr = AddonManager([root], state_dir=state)
    found = mgr.discover()
    check("discovery found the addons", "demo" in found and "tripwire" in found)
    check("DISCOVERY EXECUTES NO ADDON CODE",
          not os.path.exists(os.path.join(trip, "EXECUTED")),
          "-- merely scanning a directory ran somebody's code")
    check("nothing is registered by discovery alone",
          mgr.registry.counts() == {})
    check("discovered addons report as such",
          all(a.state == "discovered" for a in found.values()))

    # ---- trust gates loading --------------------------------------------
    mgr.load_all()
    check("untrusted addons are BLOCKED by default",
          mgr.addons["demo"].state == "blocked",
          "-- unknown code must not run unattended")
    check("a blocked addon contributes nothing", mgr.registry.counts() == {})
    check("tripwire still never executed",
          not os.path.exists(os.path.join(trip, "EXECUTED")))

    # Approving one addon must not approve the others.
    mgr.trust("demo")
    mgr.load(mgr.addons["demo"])
    check("an approved addon loads", mgr.addons["demo"].state == "loaded")
    check("approving one addon does not approve another",
          mgr.addons["tripwire"].state == "blocked")
    check("the loaded addon registered its format",
          mgr.registry.get("formats", "demo") is not None)
    check("...its command", mgr.registry.get("commands", "demo.shout") is not None)
    check("...and its named plane",
          mgr.registry.get("planes", "demo_depth") is not None)
    check("the registry knows who owns what",
          mgr.registry.owner("formats", "demo") == "demo")

    # Trust is content-addressed: a CHANGED addon is a new addon.
    digest_before = mgr.addons["demo"].digest()
    with open(os.path.join(path, "__init__.py"), "a") as f:
        f.write("\n# changed\n")
    check("editing an addon changes its digest",
          mgr.addons["demo"].digest() != digest_before,
          "-- trust would otherwise carry over to different code")

    # ---- a trust callback can decide interactively ----------------------
    asked = []

    def approve_only_demo(addon, digest):
        asked.append(addon.id)
        return addon.id == "demo"

    mgr2 = AddonManager([root], state_dir=os.path.join(root, "_state2"),
                        trust_callback=approve_only_demo)
    mgr2.discover()
    mgr2.load_all()
    check("the host asks before running unknown code", "tripwire" in asked)
    check("a declined addon stays blocked",
          mgr2.addons["tripwire"].state == "blocked")
    check("an approved addon runs", mgr2.addons["demo"].state == "loaded")

    # ---- fault isolation: the transaction -------------------------------
    root2 = tempfile.mkdtemp(prefix="ochre-addons2-")
    state2 = os.path.join(root2, "_state")
    write_addon(root2, "good", GOOD)
    write_addon(root2, "partial", PARTIAL)
    write_addon(root2, "importboom", IMPORT_BOOM)
    write_addon(root2, "noregister", NO_REGISTER)

    mgr3 = AddonManager([root2], state_dir=state2)
    mgr3.discover()
    for ident in mgr3.addons:
        mgr3.trust(ident)
    before = mgr3.registry.counts()
    mgr3.load_all()

    check("the good addon loaded", mgr3.addons["good"].state == "loaded")
    check("an addon that raises mid-register FAILS",
          mgr3.addons["partial"].state == "failed")
    check("PARTIAL REGISTRATION IS ROLLED BACK",
          mgr3.registry.get("formats", "halfway") is None
          and mgr3.registry.get("planes", "halfway_plane") is None,
          "-- a half-registered addon leaves the app in a state nobody designed")
    check("a rollback does not disturb other addons",
          mgr3.registry.get("formats", "demo") is not None)
    check("an addon that cannot be imported fails cleanly",
          mgr3.addons["importboom"].state == "failed")
    check("an addon with no register() fails cleanly",
          mgr3.addons["noregister"].state == "failed")
    check("failures explain themselves",
          "register" in (mgr3.addons["noregister"].error or ""))
    check("every failure is reported", len(mgr3.errors) >= 3)
    check("one bad addon does not stop the rest",
          mgr3.addons["good"].state == "loaded")

    # ---- disabled in the manifest ---------------------------------------
    write_addon(root2, "off", GOOD, Enabled="off", Id="off")
    mgr4 = AddonManager([root2], state_dir=state2)
    mgr4.discover()
    mgr4.trust("off")
    mgr4.load(mgr4.addons["off"])
    check("Enabled=off is respected", mgr4.addons["off"].state == "disabled")

    # ---- API mismatch ----------------------------------------------------
    write_addon(root2, "future", GOOD, Id="future", Api="9.0")
    mgr5 = AddonManager([root2], state_dir=state2)
    mgr5.discover()
    mgr5.trust("future")
    mgr5.load(mgr5.addons["future"])
    check("an addon for a future API is refused",
          mgr5.addons["future"].state == "failed")
    check("...with a message naming both versions",
          "9" in (mgr5.addons["future"].error or ""))

    # ---- dependencies ----------------------------------------------------
    root3 = tempfile.mkdtemp(prefix="ochre-addons3-")
    write_addon(root3, "base", GOOD, Id="base")
    write_addon(root3, "dependent", GOOD.replace('"demo"', '"dependent"')
                .replace("demo.shout", "dependent.shout")
                .replace("demo_depth", "dependent_depth"),
                Id="dependent", Requires="base", Order="200")
    mgr6 = AddonManager([root3], state_dir=os.path.join(root3, "_s"))
    mgr6.discover()
    for ident in mgr6.addons:
        mgr6.trust(ident)
    mgr6.load_all()
    check("a dependency loads before its dependant",
          mgr6.addons["base"].state == "loaded"
          and mgr6.addons["dependent"].state == "loaded")

    write_addon(root3, "orphan", GOOD, Id="orphan", Requires="nonexistent")
    mgr7 = AddonManager([root3], state_dir=os.path.join(root3, "_s"))
    mgr7.discover()
    for ident in mgr7.addons:
        mgr7.trust(ident)
    mgr7.load_all()
    check("an addon with a missing dependency is refused",
          mgr7.addons["orphan"].state == "failed")
    check("...and does not take its siblings with it",
          mgr7.addons["base"].state == "loaded")

    # A dependency cycle disables everyone in it rather than picking a victim.
    root4 = tempfile.mkdtemp(prefix="ochre-addons4-")
    write_addon(root4, "acycle", GOOD, Id="acycle", Requires="bcycle")
    write_addon(root4, "bcycle", GOOD.replace('"demo"', '"b"')
                .replace("demo.shout", "b.shout").replace("demo_depth", "b_depth"),
                Id="bcycle", Requires="acycle")
    mgr8 = AddonManager([root4], state_dir=os.path.join(root4, "_s"))
    mgr8.discover()
    for ident in mgr8.addons:
        mgr8.trust(ident)
    mgr8.load_all()
    check("a dependency cycle disables the whole cycle",
          mgr8.addons["acycle"].state == "failed"
          and mgr8.addons["bcycle"].state == "failed")

    # ---- duplicate registration ------------------------------------------
    root5 = tempfile.mkdtemp(prefix="ochre-addons5-")
    write_addon(root5, "first", GOOD, Id="first")
    write_addon(root5, "second", GOOD, Id="second")     # same provider names
    mgr9 = AddonManager([root5], state_dir=os.path.join(root5, "_s"))
    mgr9.discover()
    for ident in mgr9.addons:
        mgr9.trust(ident)
    mgr9.load_all()
    loaded = [a.id for a in mgr9.loaded()]
    check("two addons claiming the same name: one wins, one fails",
          len(loaded) == 1, "-- %r" % loaded)
    check("the collision is reported",
          any("already registered" in msg for _i, msg in mgr9.errors))

    # ---- guard: faults during later calls -------------------------------
    root6 = tempfile.mkdtemp(prefix="ochre-addons6-")
    write_addon(root6, "faulty", FAULTY_CALL, Id="faulty")
    mgr10 = AddonManager([root6], state_dir=os.path.join(root6, "_s"))
    mgr10.discover()
    mgr10.trust("faulty")
    mgr10.load_all()
    check("the faulty addon loaded fine", mgr10.addons["faulty"].state == "loaded")

    provider = mgr10.registry.get("formats", "faulty")
    for _ in range(mgr10.fault_budget):
        with mgr10.guard("faulty", "sniff") as g:
            provider.sniff(b"xxxx", "x.faulty")
        check("a raising addon call does not propagate", g.failed)
    check("an addon that keeps faulting is disabled",
          mgr10.addons["faulty"].state == "disabled",
          "-- a provider that throws on every call would break the open dialog")
    check("a disabled addon's contributions are withdrawn",
          mgr10.registry.get("formats", "faulty") is None)

    # ---- kill switch -----------------------------------------------------
    check("the good addon is still registered",
          mgr3.registry.get("formats", "demo") is not None)
    mgr3.disable("good")
    check("disabling withdraws everything it contributed",
          mgr3.registry.get("formats", "demo") is None
          and mgr3.registry.get("commands", "demo.shout") is None)
    check("the addon is marked disabled", mgr3.addons["good"].state == "disabled")

    # ---- crash breadcrumb ------------------------------------------------
    check("no breadcrumb after a clean run", mgr3.crashed_addon() is None)
    import json
    os.makedirs(state2, exist_ok=True)
    with open(os.path.join(state2, "loading.json"), "w") as f:
        json.dump({"id": "suspicious", "directory": "/tmp/x"}, f)
    mgr11 = AddonManager([root2], state_dir=state2)
    crashed = mgr11.crashed_addon()
    check("a surviving breadcrumb names the culprit",
          crashed is not None and crashed["id"] == "suspicious",
          "-- a native crash is the one failure Python cannot contain")

    # ---- status reporting ------------------------------------------------
    rows = mgr3.status()
    check("status lists every addon", len(rows) == len(mgr3.addons))
    check("status carries state and errors",
          all("state" in r and "error" in r for r in rows))

    # ---- host facade ------------------------------------------------------
    mgr12 = AddonManager([root], state_dir=state)
    mgr12.discover()
    mgr12.trust("demo")
    mgr12.load(mgr12.addons["demo"])
    addon = mgr12.addons["demo"]
    check("the addon logged through the host",
          any("registered" in m for m in addon.messages))
    from ochre.addons.api import Host
    host = Host(mgr12.registry, addon)
    check("data_path points inside the addon's own directory",
          host.data_path("x.ini").startswith(addon.directory),
          "-- an addon ships and reads its own data")
    check("the host exposes the API version", host.api_version == API_VERSION)

    # ---- addons are Qt-free unless they are panels ----------------------
    check("loading addons pulled in no Qt",
          not any(m.startswith(("PySide", "PyQt", "shiboken")) for m in sys.modules),
          "-- only PanelProvider may need Qt, and only when a UI exists")

    # ---- the shipped example addon, end to end --------------------------
    # This is the claim the whole design rests on: an addon adds a file
    # format, a palette and a command with NO change to Ochre.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = os.path.join(repo, "addons")
    if os.path.isdir(os.path.join(example_dir, "example")):
        import numpy as np
        from ochre.engine.geometry import Rect
        from ochre.ui.controller import Controller

        st = tempfile.mkdtemp()
        work = tempfile.mkdtemp()
        ctl = Controller(Settings(os.path.join(repo, "data")),
                         os.path.join(repo, "data"),
                         addon_dirs=[example_dir], state_dir=st)
        check("the example addon is discovered", "example" in ctl.addons.addons)
        ctl.addons.trust("example")
        loaded = ctl.load_addons()
        check("the example addon loads", [a.id for a in loaded] == ["example"])
        check("it registered a format, a palette set and a command",
              ctl.addons.registry.counts() ==
              {"formats": 1, "palettes": 1, "commands": 1},
              "-- %r" % ctl.addons.registry.counts())
        check("its extension reaches the application",
              "x1" in ctl.addon_extensions())

        prov = ctl.addons.registry.get("palettes", "example.palettes")
        pal = prov.load(prov.names()[0], ctl.addons_host("example"))
        check("an addon supplies a palette", pal is not None)
        check("...with protected ranges the CORE never defined",
              len(pal.protected) == 16)
        check("...and annotations the core renders without understanding",
              [a[2] for a in pal.annotations]
              == ["Transparent", "Reserved ramp", "Unlit"])

        ctl.new_document(24, 16, (0, 0, 0, 255))
        ctl.doc.bind_palette(pal)
        lay = ctl.doc.add_layer("art", planes=("index", "rgba"),
                                authoritative=("index",))
        cell = ctl.doc.cell(lay)
        for i in range(24):
            cell.fill(Rect(i, 0, 1, 16), (i * 7) % 256, name="index")
        original = cell.plane("index").copy()

        target = os.path.join(work, "demo.x1")
        ctl.save_path(target)
        check("saving routes through the addon", os.path.exists(target))

        ctl2 = Controller(Settings(os.path.join(repo, "data")),
                          os.path.join(repo, "data"),
                          addon_dirs=[example_dir], state_dir=st)
        ctl2.load_addons()
        ctl2.open_path(target)
        check("opening routes through the addon", ctl2.doc.width == 24)
        check("ADDON FORMAT ROUND-TRIPS THE INDEX PLANE BIT-EXACTLY",
              np.array_equal(ctl2.doc.cell(ctl2.doc.layers()[0]).plane("index"),
                             original))
        check("the reopened layer is index-locked",
              ctl2.doc.layers()[0].index_locked)

        fmt = ctl2.addons.registry.get("formats", "example.x1")
        with open(target, "rb") as f:
            head = f.read(64)
        check("the addon sniffs its own file by content", fmt.sniff(head, target))
        check("...and rejects something else", not fmt.sniff(b"nope", target))

        cmd = ctl2.addons.registry.get("commands", "example.describe")
        check("an addon command runs against the live document",
              "24x16" in cmd.run(ctl2.addons_host("example")))

        # Disabling must genuinely withdraw the format.
        ctl2.addons.disable("example")
        check("disabling an addon withdraws its format",
              ctl2._addon_format_for(target) is None)

        shutil.rmtree(st, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)

    for d in (root, root2, root3, root4, root5, root6):
        shutil.rmtree(d, ignore_errors=True)

    print("\nall addon checks passed")


if __name__ == "__main__":
    main()
