# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Palettes the user works with: built-in ones, palette files, and palettes
made from an image.

  read / write   JASC-PAL (Paint Shop Pro's text .pal), GIMP .gpl, Adobe
                 .act, Paint.NET .txt, and plain hex lists. Binary .pal
                 files of other kinds are left to palette providers (addons),
                 which know what their bytes mean.
  BUILTIN        generic palettes: greys, web-safe, the classic 16-colour
                 PC set, CGA, Game Boy, PICO-8, Commodore 64.
  median_cut     an N-colour palette for an image (Heckbert's median cut,
                 refined by a few k-means passes) -- "limit to N colours".
  map_rgba       an image's colours replaced by a palette's nearest (with
                 optional ordered or Floyd-Steinberg dither), alpha kept.

A palette shorter than 256 colours is padded, and the padding is PROTECTED
so no snap can ever produce it; `size` says how many entries are real.
"""

import os

import numpy as np

from .palette import LENGTH, Palette

EXTENSIONS = (".pal", ".gpl", ".act", ".txt", ".hex")


def make(colours, name="Palette"):
    """A Palette from a list of (r, g, b[, a]) -- padded and protected."""
    cols = [tuple(int(v) for v in c[:3]) for c in colours][:LENGTH]
    entries = np.zeros((LENGTH, 4), np.uint8)
    entries[:, 3] = 255
    for i, c in enumerate(cols):
        entries[i, :3] = c
    pal = Palette(entries, name=name)
    pal.size = len(cols)
    if len(cols) < LENGTH:
        pal.protect((len(cols), LENGTH - 1))
    return pal


def colours(palette):
    """The real entries, as an (n, 3) uint8 array."""
    n = int(getattr(palette, "size", LENGTH) or LENGTH)
    return np.asarray(palette.entries[:n, :3])


# ---- files ------------------------------------------------------------------------

def read(path):
    """A palette file as a Palette, or ValueError when this module cannot
    tell what it is (a binary .pal is a provider's business)."""
    with open(path, "rb") as fh:
        data = fh.read()
    name = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower()
    text = data[:4096].decode("latin-1")
    if text.startswith("JASC-PAL"):
        return make(_numbers(data.decode("latin-1").splitlines()[3:]), name)
    if text.startswith("GIMP Palette"):
        rows = []
        for line in data.decode("utf-8", "replace").splitlines()[1:]:
            line = line.strip()
            if not line or line.startswith("#") or ":" in line.split()[0]:
                continue
            parts = line.split()
            try:
                rows.append([int(p) for p in parts[:3]])
            except ValueError:
                continue
        return make(rows, name)
    if ext == ".act" and len(data) in (768, 772):
        n = LENGTH
        if len(data) == 772:
            n = int.from_bytes(data[768:770], "big") or LENGTH
        rgb = np.frombuffer(data[:768], np.uint8).reshape(LENGTH, 3)[:n]
        return make(rgb.tolist(), name)
    if ext in (".txt", ".hex"):
        rows = []
        for line in data.decode("utf-8", "replace").splitlines():
            line = line.strip().lstrip("#")
            if not line or line.startswith(";"):
                continue
            try:
                v = int(line[-6:], 16)
            except ValueError:
                continue
            if len(line) >= 6:
                rows.append([(v >> 16) & 255, (v >> 8) & 255, v & 255])
        if rows:
            return make(rows, name)
    raise ValueError("not a palette format Ochre reads itself: %s" % os.path.basename(path))


def _numbers(lines):
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append([int(p) & 255 for p in parts[:3]])
            except ValueError:
                pass
    return rows


def write(palette, path):
    """Write by extension: .gpl, .act, .txt (Paint.NET) or .pal (JASC text)."""
    cols = colours(palette)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gpl":
        body = "GIMP Palette\nName: %s\nColumns: 16\n#\n" % (palette.name or "Palette")
        body += "".join("%3d %3d %3d\tIndex %d\n" % (r, g, b, i) for i, (r, g, b) in enumerate(cols))
        data = body.encode("utf-8")
    elif ext == ".act":
        rgb = np.zeros((LENGTH, 3), np.uint8)
        rgb[:len(cols)] = cols
        data = rgb.tobytes() + len(cols).to_bytes(2, "big") + b"\xff\xff"
    elif ext == ".txt":
        data = ("; Paint.NET Palette File\n" + "".join(
            "FF%02X%02X%02X\n" % tuple(c) for c in cols)).encode("ascii")
    else:
        data = ("JASC-PAL\n0100\n%d\n" % len(cols) + "".join(
            "%d %d %d\n" % tuple(c) for c in cols)).encode("ascii")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def files_in(folders, recursive=True):
    """[(display name, path)] of palette-looking files in the folders; the
    name includes the subfolder ("Set/name") when a folder has several."""
    out = []
    for folder in folders:
        folder = os.path.expanduser(folder.strip())
        if not folder or not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                if f.lower().endswith(EXTENSIONS):
                    rel = os.path.relpath(os.path.join(root, f), folder)
                    out.append((os.path.splitext(rel)[0].replace(os.sep, "/"), os.path.join(root, f)))
            if not recursive:
                break
    return out


# ---- built-in palettes ---------------------------------------------------------------

def _hexes(s):
    return [((v >> 16) & 255, (v >> 8) & 255, v & 255) for v in (int(h, 16) for h in s.split())]


def _greys(n):
    return [(round(i * 255 / (n - 1)),) * 3 for i in range(n)]


def _web():
    steps = (0, 51, 102, 153, 204, 255)
    return [(r, g, b) for r in steps for g in steps for b in steps]


BUILTIN = {
    "Grayscale (4)": lambda: _greys(4),
    "Grayscale (16)": lambda: _greys(16),
    "Grayscale (256)": lambda: _greys(256),
    "Web-safe (216)": _web,
    "Classic PC (16)": lambda: _hexes("000000 800000 008000 808000 000080 800080 008080 c0c0c0 "
                                      "808080 ff0000 00ff00 ffff00 0000ff ff00ff 00ffff ffffff"),
    "CGA (4)": lambda: _hexes("000000 55ffff ff55ff ffffff"),
    "Game Boy (4)": lambda: _hexes("0f380f 306230 8bac0f 9bbc0f"),
    "PICO-8 (16)": lambda: _hexes("000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 "
                                  "ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa"),
    "Commodore 64 (16)": lambda: _hexes("000000 ffffff 880000 aaffee cc44cc 00cc55 0000aa eeee77 "
                                        "dd8855 664400 ff7777 333333 777777 aaff66 0088ff bbbbbb"),
}


def builtin(name):
    return make(BUILTIN[name](), name)


# ---- making and applying palettes ------------------------------------------------------

def median_cut(rgba, n, refine=4):
    """An n-colour palette for the image's (opaque-ish) pixels."""
    a = np.asarray(rgba)
    pix = a[..., :3][a[..., 3] >= 128].reshape(-1, 3).astype(np.float64)
    if pix.size == 0:
        return make([(0, 0, 0)] * max(1, n), "Empty")
    if len(pix) > 200000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 200000, replace=False)]
    uniq = np.unique(pix.astype(np.uint8), axis=0)
    if len(uniq) <= n:
        return make(uniq.tolist(), "%d colours" % len(uniq))
    boxes = [pix]
    while len(boxes) < n:
        # Split the box with the widest channel range (by pixel count, ties).
        i = max(range(len(boxes)), key=lambda k: (np.ptp(boxes[k], axis=0).max() * len(boxes[k]) ** 0.5
                                                  if len(boxes[k]) > 1 else -1))
        box = boxes.pop(i)
        if len(box) < 2:
            boxes.append(box)
            break
        ch = int(np.argmax(np.ptp(box, axis=0)))
        order = np.argsort(box[:, ch], kind="stable")
        mid = len(box) // 2
        boxes += [box[order[:mid]], box[order[mid:]]]
    centres = np.array([b.mean(axis=0) for b in boxes])
    for _ in range(refine):                                   # k-means polish
        d = ((pix[:, None, :] - centres[None, :, :]) ** 2).sum(-1) if len(pix) * len(centres) < 4e7 \
            else None
        if d is None:
            break
        lab = d.argmin(1)
        for k in range(len(centres)):
            sel = pix[lab == k]
            if len(sel):
                centres[k] = sel.mean(axis=0)
    cols = np.clip(np.rint(centres), 0, 255).astype(np.uint8)
    cols = cols[np.lexsort((cols[:, 2], cols[:, 1], cols[:, 0]))]
    return make(cols.tolist(), "%d colours" % len(cols))


def map_rgba(rgba, palette, dither="none"):
    """The image in the palette's colours (nearest; optional 'ordered' or
    'floyd' dither); alpha kept, fully transparent pixels untouched."""
    from .quantize import FLOYD, NONE, ORDERED, quantize
    a = np.ascontiguousarray(rgba, np.uint8)
    mode = {"none": NONE, "ordered": ORDERED, "floyd": FLOYD}.get(dither, NONE)
    work = a.copy()
    work[..., 3] = 255                     # colour only: alpha is the layer's business
    idx = quantize(work, palette, mode, force=True)
    out = a.copy()
    out[..., :3] = palette.entries[idx, :3]
    out[a[..., 3] == 0, :3] = a[a[..., 3] == 0, :3]
    return out


def nearest(palette, rgb):
    """The palette colour nearest to rgb."""
    cols = colours(palette).astype(np.int64)
    d = ((cols - np.asarray(rgb[:3], np.int64)) ** 2).sum(1)
    return tuple(int(v) for v in cols[int(np.argmin(d))])


# ---- a palette from the colours an image uses ---------------------------------------

def used_colours(images, alpha_min=128):
    """(colours (n, 3) uint8, counts (n,)) of the colours the images use,
    counting pixels at least `alpha_min` opaque, most used first."""
    stacks = [np.asarray(im)[..., :3][np.asarray(im)[..., 3] >= alpha_min] for im in images]
    pix = np.concatenate([s.reshape(-1, 3) for s in stacks]) if stacks else np.zeros((0, 3), np.uint8)
    if not len(pix):
        return np.zeros((0, 3), np.uint8), np.zeros(0, np.int64)
    key = (pix[:, 0].astype(np.int64) << 16) | (pix[:, 1].astype(np.int64) << 8) | pix[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    uniq, counts = uniq[order], counts[order]
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], 1).astype(np.uint8)
    return cols, counts


def _lab(cols):
    """sRGB (n, 3) uint8 -> CIE Lab (n, 3) float, D65 (plain formula)."""
    c = np.asarray(cols, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116.0)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


class PalettePlan:
    """How a palette is built from used colours.

    colours   the final palette (k, 3) uint8
    groups    for each final colour, the indices (into `used`) merged into it
    used      the image's colours (n, 3) uint8; counts how many pixels each
    exact     True when nothing had to be merged
    merges    the recommended merges, closest first: (members, result, delta E)
    """

    def __init__(self, used, counts, colours, groups, merges, exact, approximate=False):
        self.used, self.counts = used, counts
        self.colours, self.groups = colours, groups
        self.merges, self.exact, self.approximate = merges, exact, approximate

    def palette(self, name="From image"):
        return make(self.colours.tolist(), name)

    def mapping(self):
        """{used colour (r, g, b): palette colour (r, g, b)}."""
        out = {}
        for k, members in enumerate(self.groups):
            for i in members:
                out[tuple(int(v) for v in self.used[i])] = tuple(int(v) for v in self.colours[k])
        return out


def plan_palette(used, counts, target, limit=1536):
    """A palette of at most `target` colours for colours used `counts` times.

    Fits already: the used colours themselves, most used first. Too many:
    the two CLOSEST colours (delta E in Lab) are merged into their
    pixel-weighted average, over and over, until it fits -- each merge
    recorded so the user can see what would be joined. Above `limit`
    distinct colours (a photograph), the rarest are first folded into their
    nearest frequent colour, and the plan says it is approximate."""
    used = np.asarray(used, np.uint8)
    counts = np.asarray(counts, np.int64)
    n = len(used)
    target = max(1, int(target))
    if n <= target:
        return PalettePlan(used, counts, used.copy(), [[i] for i in range(n)], [], True)
    lab = _lab(used)
    weight = counts.astype(np.float64)
    groups = [[i] for i in range(n)]
    approximate = False
    alive_idx = np.arange(n)
    if n > limit:
        approximate = True
        keep = np.arange(limit)                       # used is sorted most-used first
        rest = np.arange(limit, n)
        near = np.argmin(((lab[rest][:, None, :] - lab[keep][None, :, :]) ** 2).sum(-1), axis=1) \
            if len(rest) * limit < 6e7 else None
        if near is None:
            near = np.array([np.argmin(((lab[keep] - lab[r]) ** 2).sum(-1)) for r in rest])
        for r, k in zip(rest, near):
            groups[k].append(int(r))
            w = weight[k] + weight[r]
            lab[k] = (lab[k] * weight[k] + lab[r] * weight[r]) / w
            weight[k] = w
        alive_idx = keep
    m = len(alive_idx)
    L = lab[alive_idx].copy()
    W = weight[alive_idx].copy()
    G = [groups[i] for i in alive_idx]
    alive = np.ones(m, bool)
    D = np.sqrt(((L[:, None, :] - L[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    nd = D[np.arange(m), nn]
    merges = []
    count = m
    while count > target:
        i = int(np.argmin(np.where(alive, nd, np.inf)))
        j = int(nn[i])
        dist = float(D[i, j])
        w = W[i] + W[j]
        L[i] = (L[i] * W[i] + L[j] * W[j]) / w
        W[i] = w
        G[i] = G[i] + G[j]
        alive[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        row = np.sqrt(((L - L[i]) ** 2).sum(-1))
        row[~alive] = np.inf
        row[i] = np.inf
        D[i, :] = row
        D[:, i] = row
        stale = alive & ((nn == i) | (nn == j))
        stale[i] = True
        for r in np.nonzero(stale)[0]:
            nn[r] = int(np.argmin(D[r]))
            nd[r] = D[r, nn[r]]
        better = alive & (row < nd)
        nn[better] = i
        nd[better] = row[better]
        nd[~alive] = np.inf
        merges.append((i, dist))
        count -= 1
    finals = np.nonzero(alive)[0]
    # The colour of a merged group: its pixel-weighted average in RGB (what
    # the eye sees as the group's colour), rounded.
    colours, out_groups = [], []
    for k in finals:
        members = G[k]
        cw = counts[members].astype(np.float64)
        avg = (used[members].astype(np.float64) * cw[:, None]).sum(0) / cw.sum()
        colours.append(np.clip(np.rint(avg), 0, 255))
        out_groups.append(sorted(members, key=lambda t: -counts[t]))
    order = np.argsort([-counts[g].sum() for g in out_groups], kind="stable")
    colours = np.asarray(colours, np.uint8)[order]
    out_groups = [out_groups[o] for o in order]
    # Recommendations: every final colour that joins several used ones,
    # tightest groups first.
    recs = []
    for c, members in zip(colours, out_groups):
        if len(members) > 1:
            spread = float(np.sqrt(((_lab(used[members]) - _lab(c[None, :])) ** 2).sum(-1)).max())
            recs.append((members, tuple(int(v) for v in c), spread))
    recs.sort(key=lambda r: r[2])
    return PalettePlan(used, counts, colours, out_groups, recs, False, approximate)


def remap(rgba, mapping):
    """Replace colours exactly per a {(r,g,b): (r,g,b)} mapping (alpha kept);
    colours not in it go to the nearest mapped result."""
    a = np.array(rgba, copy=True)
    if not mapping:
        return a
    src = np.array(list(mapping.keys()), np.int64)
    dst = np.array(list(mapping.values()), np.uint8)
    key = (a[..., 0].astype(np.int64) << 16) | (a[..., 1].astype(np.int64) << 8) | a[..., 2]
    skey = (src[:, 0] << 16) | (src[:, 1] << 8) | src[:, 2]
    order = np.argsort(skey)
    skey, dst_sorted = skey[order], dst[order]
    pos = np.clip(np.searchsorted(skey, key), 0, len(skey) - 1)
    hit = skey[pos] == key
    out = a.copy()
    out[..., :3][hit] = dst_sorted[pos[hit]]
    if (~hit).any():
        pal = make(sorted(set(map(tuple, dst.tolist()))), "mapped")
        miss = map_rgba(a, pal)
        out[..., :3][~hit] = miss[..., :3][~hit]
    out[a[..., 3] == 0, :3] = a[a[..., 3] == 0, :3]
    return out
