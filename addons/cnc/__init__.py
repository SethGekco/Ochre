# Ochre C&C addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""Command & Conquer formats for Ochre: SHP sprites and TMP terrain.

One distributable, two INDEPENDENT providers. Either can be switched off in
data/cnc.ini without disturbing the other; they share only the palette layer
and this registration.

Nothing in Ochre's core knows any of this exists. Everything game-specific --
the SHP and TMP codecs, the 6-bit palette expansion, the meaning of index 0
or of the 16-31 ramp, the land and ramp enumerations -- lives in this
directory and reaches the editor as generic registrations and opaque
annotations. That was the whole point of the addon boundary, and this addon
is the proof it holds.

Format references: ModdingWiki (SHP TS, RLE-Zero), ModEnc (PAL), XCC
Utilities, and FS-21's editors. See docs/MINING.md.
"""

from . import cncpal, cncshp, cnctmp


class SplitShadowsCommand:
    """Report a unit SHP's art/shadow split.

    TS/RA2 unit sprites conventionally hold 2N frames: N of art followed by N
    of 1-bit shadow. Nothing in the file says so, which is exactly why an
    editor that assumes frame count equals animation length plays every
    animation twice, the second time as black silhouettes.
    """

    name = "cnc.shadow_split"
    label = "Report Shadow Split"
    category = "C&C"

    def run(self, host, **kwargs):
        doc = getattr(host.controller, "doc", None)
        if doc is None:
            return "No document open."
        count = len(doc.frames)
        if count < 2 or count % 2:
            return ("%d frames: not an even count, so this is probably not a "
                    "unit sprite with paired shadows." % count)
        half = count // 2
        layer = next((l for l in doc.layers() if l.index_locked), None)
        if layer is None:
            return "%d frames; no indexed layer to inspect." % count

        # A shadow frame uses only index 0 and index 1.
        shadowish = 0
        for frame in doc.frames[half:]:
            cell = doc.cells.get((layer.id, frame.id))
            if cell is None or cell.planes.get("index") is None:
                continue
            used = set(int(v) for v in set(cell.plane("index").ravel().tolist()))
            if used <= {0, 1}:
                shadowish += 1
        return ("%d frames. If this is a unit sprite, frames 0-%d are art and "
                "%d-%d are shadows; %d of those %d use only indices 0 and 1, "
                "which is what a shadow frame looks like."
                % (count, half - 1, half, count - 1, shadowish, half))


class TerrainInfoCommand:
    """Describe the current TMP tile using the addon's own enumerations."""

    name = "cnc.terrain_info"
    label = "Tile Terrain Info"
    category = "C&C"

    def __init__(self, db):
        self._db = db

    def run(self, host, **kwargs):
        doc = getattr(host.controller, "doc", None)
        if doc is None or doc.meta.get("format") != "cnc.tmp":
            return "Not a TMP document."
        frame = doc.frame
        if frame.meta.get("cnc.empty") == "1":
            return "Tile %s is an empty slot." % frame.meta.get("cnc.slot", "?")
        land = frame.meta.get("cnc.land", "0")
        ramp = frame.meta.get("cnc.ramp", "0")
        return ("Tile %s  -  height %s\n"
                "land %s (%s)\nramp %s (%s)\n\n"
                "The numbers are authoritative and are written back verbatim; "
                "several distinct values share one label."
                % (frame.meta.get("cnc.slot", "?"),
                   frame.meta.get("cnc.height", "0"),
                   land, self._db.get("LandType", land, "unknown"),
                   ramp, self._db.get("RampType", ramp, "unknown")))


def register(host):
    """Register both providers. Either may be disabled independently."""
    config = host.read_ini("data", "cnc.ini")
    palettes = cncpal.CncPalettes(host)
    host.register_palette(palettes)

    pal_db = host.read_ini("data", "palettes.ini")
    terrain_db = host.read_ini("data", "terrain.ini")

    enabled = []
    if config.getbool("Providers", "Shp", True):
        host.register_format(cncshp.ShpFormat(palettes, pal_db))
        host.register_command(SplitShadowsCommand())
        enabled.append("SHP")
    if config.getbool("Providers", "Tmp", True):
        host.register_format(cnctmp.TmpFormat(palettes, pal_db))
        host.register_command(TerrainInfoCommand(terrain_db))
        # TMP's per-pixel depth is not colour, so it rides on a named plane.
        # Declaring it turns a typo into an error rather than an orphaned
        # 16 MB array.
        host.register_plane("height")
        enabled.append("TMP")

    host.log("C&C addon registered: %s" % (", ".join(enabled) or "nothing"))
