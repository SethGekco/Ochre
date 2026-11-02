# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Lens profiles for Lens Correction, read from lensfun's database.

lensfun (https://lensfun.github.io) is the free lens-calibration database
darktable, RawTherapee and GIMP use. Its data is CC BY-SA 3.0, so Ochre
ships none of it: this module reads whatever copy the system or the user
installed (the lensfun-data package, lensfun-update-data, or XML files
dropped into ~/.local/share/lensfun) at run time. With none installed there
are no profiles, and Lens Correction's Profile tab says so.

THE MODELS. lensfun measures distortion and transverse chromatic aberration
(TCA) with radius 1 at half the calibration sensor's SHORTER side, and
vignetting with radius 1 at its corner (lensfun's modifier.cpp). Lens
Correction works with radius 1 at the image's half-diagonal, so each model
is converted exactly on reading: with s the half-diagonal in lensfun's
units (hypot(aspect, 1), aspect 3:2 unless the database says otherwise),
a term in r^i scales by s^i. ptlens and poly3 distortion also have their
centre magnification d divided out, as lensfun's own
rescale_polynomial_coefficients does, so the middle of the frame keeps
its size. The result, for every calibration, is:

  distortion  [c1..c6]: an undistorted point at radius r shows the source
              at r * (1 + c1 r + c2 r^2 + ... + c6 r^6)
  ca          per channel [v, c, b]: red / green / blue sample at
              r * (v + c r + b r^2) (green is [1, 0, 0])
  vignette    [k1, k2, k3]: the lens dims by 1 + k1 r^2 + k2 r^4 + k3 r^6,
              which the correction divides out

A calibration made on one camera serves another by its crop factor: the
image's corner sits at cal_crop / camera_crop of the calibration's, and each
r^i term scales by that ratio to the i.

INTERPOLATION. Distortion and TCA are measured at a few focal lengths; a
shot between two is interpolated linearly in focal length (each term of the
converted polynomial, so the MAP is interpolated, whatever model each end
used), and one outside the range takes the nearest end. Vignetting depends
on focal length, aperture and focus distance, and is interpolated the way
lensfun does it: inverse-distance weighting (power 3.5) over focal length
scaled to the lens's range, 4 / f-number and 0.1 / distance, so that what
matters optically -- wide apertures, near focus -- is where the spacing is
widest. Kooka Pictura (crates/pictura-filters/src/distort/lens_correction/,
lensfun.rs and profile.rs), where the reading and conversion come from,
picks the single nearest calibration instead.

Everything here is Qt-free: the dialog asks, Lens Correction applies.
"""

import json
import math
import os
import re
import xml.etree.ElementTree as ET

# The variable that, when set, is the ONLY place looked in. Tests point it
# at a synthetic database; a user can point it at a private copy.
ENV_DIR = "OCHRE_LENSFUN_DIR"

# lensfun's default sensor aspect when a lens does not give one.
DEFAULT_ASPECT = 1.5

# The identity model: what a profile-less correction applies.
NO_MODEL = {"distortion": [0.0] * 6, "ca": [[1.0, 0.0, 0.0]] * 3, "vignette": [0.0] * 3}


# ---- where the database lives -------------------------------------------------------

def _data_home(env):
    home = env.get("XDG_DATA_HOME") or ""
    if home:
        return home
    user = env.get("HOME") or ""
    return os.path.join(user, ".local", "share") if user else None


def search_roots(env=None):
    """lensfun's directories, lowest priority first. OCHRE_LENSFUN_DIR, when
    set, is the only one; otherwise the system's, then the user's, as
    lensfun itself searches them."""
    env = os.environ if env is None else env
    only = env.get(ENV_DIR) or ""
    if only:
        return [only]
    roots = ["/usr/share/lensfun", "/usr/local/share/lensfun"]
    dirs = env.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    # The first XDG_DATA_DIRS entry matters most, so it goes last.
    roots += [os.path.join(d, "lensfun") for d in reversed(dirs.split(":")) if d]
    roots.append("/var/lib/lensfun-updates")
    home = _data_home(env)
    if home:
        roots += [os.path.join(home, "lensfun"), os.path.join(home, "lensfun", "updates")]
    out = []
    for root in roots:
        # A repeat keeps its LATER place: it is the more important one.
        if root in out:
            out.remove(root)
        out.append(root)
    return out


def database_files(roots):
    """The XML files to read: each root's version_1/, version_2/ and the root
    itself. A file name seen again in a later place replaces the earlier
    file, as lensfun's update mechanism intends."""
    by_name = {}
    for root in roots:
        for sub in ("version_1", "version_2", ""):
            folder = os.path.join(root, sub) if sub else root
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                continue
            for name in names:
                path = os.path.join(folder, name)
                if name.lower().endswith(".xml") and os.path.isfile(path):
                    by_name[name] = path
    return [by_name[k] for k in sorted(by_name)]


# ---- reading the XML ----------------------------------------------------------------

class Camera:
    __slots__ = ("maker", "model", "mount", "crop")

    def __init__(self, maker, model, mount, crop):
        self.maker, self.model, self.mount, self.crop = maker, model, mount, crop

    @property
    def name(self):
        return _joined(self.maker, self.model)


class Calibration:
    """One <calibration>: its sensor and its converted entries."""
    __slots__ = ("crop", "distortion", "tca", "vignetting")

    def __init__(self, crop):
        self.crop = crop
        self.distortion = []        # [(focal, [c1..c6])]
        self.tca = []               # [(focal, [[v, c, b] x 3])]
        self.vignetting = []        # [(focal, aperture, distance, [k1, k2, k3])]

    def empty(self):
        return not (self.distortion or self.tca or self.vignetting)


class Lens:
    __slots__ = ("maker", "model", "mounts", "crop", "calibrations", "min_focal", "max_focal")

    def __init__(self, maker, model, mounts, crop, calibrations):
        self.maker, self.model, self.mounts, self.crop = maker, model, mounts, crop
        self.calibrations = calibrations
        focals = [e[0] for c in calibrations for e in c.distortion + c.tca + c.vignetting]
        self.min_focal = min(focals) if focals else 0.0
        self.max_focal = max(focals) if focals else 0.0

    @property
    def name(self):
        return _joined(self.maker, self.model)

    def corrects(self):
        """Which corrections any calibration carries: (dist, tca, vig)."""
        return (any(c.distortion for c in self.calibrations),
                any(c.tca for c in self.calibrations),
                any(c.vignetting for c in self.calibrations))


def _joined(maker, model):
    """"Canon EF 50mm" rather than "Canon Canon EF 50mm"."""
    if model.lower().startswith(maker.lower()):
        return model
    return "%s %s" % (maker, model)


def _ratio(text):
    """"1.5" or "3:2" as a positive float; None when it is neither."""
    try:
        if ":" in text:
            a, b = text.split(":", 1)
            value = float(a) / float(b)
        else:
            value = float(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _text(elem, tag):
    """A child's text, preferring the untranslated one (no lang=) and then
    English, as lensfun shows names."""
    best = None
    for child in elem.findall(tag):
        lang = child.get("lang")
        value = (child.text or "").strip()
        if lang is None:
            return value
        if lang == "en" and best is None:
            best = value
    return best


def _num(attrs, key, default=None):
    value = attrs.get(key)
    if value is None:
        if default is None:
            raise ValueError(key)
        return default
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(key)
    return v


def _distortion(attrs, s):
    """[c1..c6] on the half-diagonal; None for a model this does not apply."""
    model = attrs.get("model", "")
    k = lambda key: _num(attrs, key, 0.0)
    if model == "poly3":
        k1 = k("k1")
        d = 1.0 - k1
        if abs(d) < 1e-9:
            raise ValueError("poly3")
        return [0.0, k1 * s * s / d ** 3, 0.0, 0.0, 0.0, 0.0]
    if model == "poly5":
        return [0.0, k("k1") * s * s, 0.0, k("k2") * s ** 4, 0.0, 0.0]
    if model == "ptlens":
        a, b, c = k("a"), k("b"), k("c")
        d = 1.0 - a - b - c
        if abs(d) < 1e-9:
            raise ValueError("ptlens")
        return [c * s / d ** 2, b * s * s / d ** 3, a * s ** 3 / d ** 4, 0.0, 0.0, 0.0]
    return None


def _tca(attrs, s):
    """Red, green and blue [v, c, b] on the half-diagonal, or None."""
    model = attrs.get("model", "")
    k = lambda key, default: _num(attrs, key, default)
    if model == "linear":
        return [[k("kr", 1.0), 0.0, 0.0], [1.0, 0.0, 0.0], [k("kb", 1.0), 0.0, 0.0]]
    if model == "poly3":
        return [[k("vr", 1.0), k("cr", 0.0) * s, k("br", 0.0) * s * s],
                [1.0, 0.0, 0.0],
                [k("vb", 1.0), k("cb", 0.0) * s, k("bb", 0.0) * s * s]]
    return None


def _vignetting(attrs):
    if attrs.get("model", "") != "pa":
        return None
    return [_num(attrs, "k1", 0.0), _num(attrs, "k2", 0.0), _num(attrs, "k3", 0.0)]


def _read_lens(elem):
    """A <lens> as a Lens; None when it is not rectilinear, is malformed, or
    has nothing Lens Correction can apply."""
    maker, model = _text(elem, "maker"), _text(elem, "model")
    if not maker or not model:
        return None
    kind = _text(elem, "type")
    if kind and kind != "rectilinear":
        return None
    crop_text, aspect_text = _text(elem, "cropfactor"), _text(elem, "aspect-ratio")
    crop = _ratio(crop_text) if crop_text else None
    aspect = _ratio(aspect_text) if aspect_text else None
    if (crop_text and crop is None) or (aspect_text and aspect is None):
        return None
    mounts = [(m.text or "").strip() for m in elem.findall("mount") if (m.text or "").strip()]
    calibrations = []
    try:
        for cal in elem.findall("calibration"):
            c_crop = _ratio(cal.get("cropfactor")) if cal.get("cropfactor") else crop
            c_aspect = _ratio(cal.get("aspect-ratio")) if cal.get("aspect-ratio") else aspect
            s = math.hypot(c_aspect or DEFAULT_ASPECT, 1.0)
            out = Calibration(c_crop)
            for entry in cal:
                attrs = entry.attrib
                if entry.tag not in ("distortion", "tca", "vignetting"):
                    continue
                focal = _num(attrs, "focal")
                if entry.tag == "distortion":
                    terms = _distortion(attrs, s)
                    if terms is not None:
                        out.distortion.append((focal, terms))
                elif entry.tag == "tca":
                    terms = _tca(attrs, s)
                    if terms is not None:
                        out.tca.append((focal, terms))
                else:
                    terms = _vignetting(attrs)
                    if terms is not None:
                        aperture = _num(attrs, "aperture", 0.0)
                        distance = _num(attrs, "distance", 0.0)
                        out.vignetting.append((focal, aperture, distance, terms))
            for entries in (out.distortion, out.tca, out.vignetting):
                entries.sort(key=lambda e: e[:-1])
            if not out.empty():
                calibrations.append(out)
    except ValueError:
        return None                       # a bad number: skip the lens, as lensfun does
    if not calibrations:
        return None
    return Lens(maker, model, mounts, crop, calibrations)


def _read_camera(elem):
    maker, model = _text(elem, "maker"), _text(elem, "model")
    crop_text = _text(elem, "cropfactor")
    crop = _ratio(crop_text) if crop_text else None
    if not maker or not model or crop is None:
        return None
    return Camera(maker, model, (_text(elem, "mount") or ""), crop)


def parse(text):
    """(cameras, lenses) in one lensfun XML document. Malformed XML raises
    ValueError; a single bad lens or camera is skipped."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(str(exc)) from None
    cameras = [c for c in (_read_camera(e) for e in root.iter("camera")) if c is not None]
    lenses = [l for l in (_read_lens(e) for e in root.iter("lens")) if l is not None]
    return cameras, lenses


# ---- the database -------------------------------------------------------------------

class LensDatabase:
    """Every camera and lens in the installed lensfun files."""

    def __init__(self, cameras=(), lenses=(), files=(), errors=()):
        self.cameras = list(cameras)
        self.lenses = list(lenses)
        self.files = list(files)
        self.errors = list(errors)          # [(path, message)] skipped files

    def __bool__(self):
        return bool(self.lenses)

    @classmethod
    def load(cls, roots=None):
        roots = search_roots() if roots is None else roots
        cameras, lenses, files, errors = [], [], [], []
        for path in database_files(roots):
            try:
                with open(path, "rb") as fh:
                    c, l = parse(fh.read())
            except (OSError, ValueError) as exc:
                errors.append((path, str(exc)))
                continue
            files.append(path)
            cameras += c
            lenses += l
        return cls(cameras, lenses, files, errors)

    # ---- matching a photo ----------------------------------------------------------

    def find_camera(self, make, model):
        """The camera an EXIF Make / Model names, or None."""
        if not model:
            return None
        best, best_score = None, 0.0
        for cam in self.cameras:
            if make and not _same_maker(cam.maker, make):
                continue
            score = _similarity(cam.model, model)
            if score > best_score:
                best, best_score = cam, score
        return best if best_score >= 0.8 else None

    def match_lenses(self, lens_name, maker="", camera=None):
        """Lenses for an EXIF lens name, best first: [(score, Lens)].

        Names are compared word by word (lowercase, digits split from
        letters, so "EF24-70mm" meets "EF 24-70mm"). Every number in the
        photo's name must be in the lens's too -- 50mm f/1.8 is not 50mm
        f/1.4, however alike the words. A lens whose maker matches gains,
        and with a known camera a lens for another mount is dropped, as
        lensfun does. AUTO_MATCH is the score worth choosing unasked."""
        if not lens_name:
            return []
        numbers = set(_numbers(lens_name))
        out = []
        for lens in self.lenses:
            if camera is not None and camera.mount and lens.mounts \
                    and camera.mount not in lens.mounts:
                continue
            if not numbers <= set(_numbers(lens.model)):
                continue
            score = max(_similarity(lens.model, lens_name), _similarity(lens.name, lens_name))
            if maker and _same_maker(lens.maker, maker):
                score += 0.1
            if score >= 0.6:
                out.append((score, lens))
        out.sort(key=lambda p: (-p[0], p[1].name.lower()))
        return out

    def lens_by_name(self, name):
        for lens in self.lenses:
            if lens.name == name:
                return lens
        return None


_CACHE = {}


def installed():
    """The installed database, read once and again only when a file in it
    (or the list of files, or OCHRE_LENSFUN_DIR) changes."""
    roots = search_roots()
    files = database_files(roots)
    key = []
    for path in files:
        try:
            key.append((path, os.stat(path).st_mtime_ns))
        except OSError:
            key.append((path, None))
    key = tuple(key)
    if _CACHE.get("key") != key:
        _CACHE["db"] = LensDatabase.load(roots)
        _CACHE["key"] = key
    return _CACHE["db"]


_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")

# A lens match at least this good is picked for the user from the EXIF.
AUTO_MATCH = 0.85


def _words(text):
    return _WORD.findall((text or "").lower())


def _numbers(text):
    return [w for w in _words(text) if w[0].isdigit()]


def _similarity(a, b):
    """Dice coefficient over the two names' words: 1 when they are the same
    words in any order, 0 when they share none."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    rest = list(wb)
    common = 0
    for w in wa:
        if w in rest:
            rest.remove(w)
            common += 1
    return 2.0 * common / (len(wa) + len(wb))


def _same_maker(a, b):
    """"NIKON CORPORATION" is Nikon; "OLYMPUS IMAGING CORP." is Olympus."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and (wa[0] == wb[0])


# ---- a lens's model for one shot ------------------------------------------------------

def _lerp_focal(entries, focal):
    """entries [(focal, terms)] sorted by focal: the terms at `focal`,
    linear between neighbours, the nearest end outside. None when empty."""
    if not entries:
        return None
    if focal is None or focal <= entries[0][0]:
        return entries[0][1]
    if focal >= entries[-1][0]:
        return entries[-1][1]
    for (f0, t0), (f1, t1) in zip(entries, entries[1:]):
        if f0 <= focal <= f1:
            w = 0.0 if f1 == f0 else (focal - f0) / (f1 - f0)
            return _mix(t0, t1, w)
    return entries[-1][1]


def _mix(a, b, w):
    if isinstance(a, list):
        return [_mix(x, y, w) for x, y in zip(a, b)]
    return a + (b - a) * w


def _vignette_at(lens, entries, focal, aperture, distance):
    """lensfun's inverse-distance weighting; None when nothing is close."""
    if not entries:
        return None
    span = lens.max_focal - lens.min_focal

    def where(f, a, d):
        x = f - lens.min_focal
        if span > 0:
            x /= span
        return (x, 4.0 / a if a > 0 else 0.0, 0.1 / d if d > 0 else 0.0)

    shot = where(focal if focal else entries[0][0],
                 aperture if aperture else entries[0][1],
                 distance if distance else 1000.0)
    total, acc, nearest = 0.0, [0.0, 0.0, 0.0], float("inf")
    for f, a, d, terms in entries:
        at = where(f, a if a else (aperture or 0.0), d if d else (distance or 1000.0))
        gap = math.sqrt(sum((p - q) ** 2 for p, q in zip(at, shot)))
        if gap < 1e-4:
            return list(terms)
        nearest = min(nearest, gap)
        weight = 1.0 / gap ** 3.5
        acc = [x + weight * t for x, t in zip(acc, terms)]
        total += weight
    if nearest > 1.0 or total <= 0:
        return None
    return [x / total for x in acc]


def _rescaled(model, t):
    """The model for an image whose half-diagonal is `t` times the
    calibration's: each r^i term times t^i."""
    return {"distortion": [c * t ** (i + 1) for i, c in enumerate(model["distortion"])],
            "ca": [[v, c * t, b * t * t] for v, c, b in model["ca"]],
            "vignette": [k * t ** (2 * (i + 1)) for i, k in enumerate(model["vignette"])]}


def pick_calibration(lens, camera_crop=None):
    """The calibration made on the sensor nearest the camera's."""
    cals = lens.calibrations
    if camera_crop is None or len(cals) == 1:
        return cals[0]
    return min(cals, key=lambda c: abs((c.crop or camera_crop) - camera_crop))


def lens_model(lens, focal=None, aperture=None, distance=None, camera_crop=None):
    """{distortion, ca, vignette} for `lens` at one shot's settings, on the
    image's half-diagonal (see the module docstring)."""
    cal = pick_calibration(lens, camera_crop)
    model = {"distortion": _lerp_focal(cal.distortion, focal) or NO_MODEL["distortion"],
             "ca": _lerp_focal(cal.tca, focal) or NO_MODEL["ca"],
             "vignette": _vignette_at(lens, cal.vignetting, focal, aperture, distance)
             or NO_MODEL["vignette"]}
    model = {k: [list(x) if isinstance(x, list) else x for x in v] for k, v in model.items()}
    if cal.crop and camera_crop:
        model = _rescaled(model, cal.crop / camera_crop)
    return model


def model_json(model):
    return json.dumps(model, separators=(",", ":"))


def model_from_json(text):
    """A model stored as JSON text, validated; None when empty or bad."""
    if not text:
        return None
    try:
        data = json.loads(text)
        out = {"distortion": [float(v) for v in data["distortion"]][:6],
               "ca": [[float(v) for v in row][:3] for row in data["ca"]][:3],
               "vignette": [float(v) for v in data["vignette"]][:3]}
    except (ValueError, TypeError, KeyError):
        return None
    if len(out["distortion"]) != 6 or len(out["vignette"]) != 3 or len(out["ca"]) != 3 \
            or any(len(r) != 3 for r in out["ca"]):
        return None
    flat = out["distortion"] + out["vignette"] + [v for r in out["ca"] for v in r]
    if not all(math.isfinite(v) for v in flat):
        return None
    return out
