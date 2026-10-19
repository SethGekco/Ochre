# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Colour management through littleCMS (Pillow's ImageCms).

The working pixels stay 8-bit RGBA. What this adds is what the numbers
MEAN:

  * a document may carry an ICC profile -- picked up from an imported file,
    or set with Edit > Assign Profile / Convert to Profile -- kept in
    doc.meta (name in "icc", the profile itself base64 in "icc_data"), so it
    survives a save. No profile means sRGB.
  * the view converts from the document's profile to the monitor's (sRGB
    unless Preferences names another), optionally through a SOFT PROOF
    profile (View > Proof Colors: how it will print on that press or
    paper), with a GAMUT WARNING marking colours the proof profile cannot
    reproduce.
  * export embeds the profile, and can write CMYK through a chosen profile.

Profiles are found in the usual places (/usr/share/color/icc, ~/.local/
share/icc, ~/.color/icc). Nothing here knows any particular press or
paper; whatever profiles the system has are what is offered.
"""

import base64
import io
import os

import numpy as np

INTENTS = {"perceptual": 0, "relative": 1, "saturation": 2, "absolute": 3}
SEARCH = ("/usr/share/color/icc", "/usr/local/share/color/icc",
          os.path.expanduser("~/.local/share/icc"), os.path.expanduser("~/.color/icc"))
GAMUT_COLOUR = (128, 128, 128)      # Photoshop's default gamut warning grey
GAMUT_DELTA_E = 6.0                  # a round trip further than this is out of gamut


def _cms():
    from PIL import ImageCms
    return ImageCms


def srgb():
    return _cms().createProfile("sRGB")


def open_profile(source):
    """An ImageCmsProfile from a path, raw bytes, or None/"sRGB" (sRGB)."""
    cms = _cms()
    if source in (None, "", "sRGB", "srgb"):
        return cms.ImageCmsProfile(srgb())
    if isinstance(source, (bytes, bytearray)):
        return cms.ImageCmsProfile(io.BytesIO(bytes(source)))
    return cms.ImageCmsProfile(source)


def describe(profile):
    cms = _cms()
    try:
        return cms.getProfileDescription(profile).strip() or "Unnamed profile"
    except cms.PyCMSError:
        return "Unnamed profile"


def colour_space(profile):
    """'RGB', 'CMYK', 'GRAY', 'LAB' ... from the profile header."""
    try:
        return profile.profile.xcolor_space.strip().upper()
    except AttributeError:
        return "RGB"


def profile_bytes(profile):
    return profile.tobytes()


_SCAN = {}


def system_profiles(dirs=SEARCH):
    """[(description, path, colour space)] for every readable ICC profile
    (scanned once per session per set of folders)."""
    key = tuple(dirs)
    if key in _SCAN:
        return list(_SCAN[key])
    _SCAN[key] = _scan(dirs)
    return list(_SCAN[key])


def _scan(dirs):
    out, seen = [], set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in sorted(files):
                if not f.lower().endswith((".icc", ".icm")):
                    continue
                path = os.path.join(root, f)
                try:
                    p = open_profile(path)
                except Exception:                           # noqa: BLE001
                    continue
                desc = describe(p)
                key = (desc, colour_space(p))
                if key in seen:
                    continue
                seen.add(key)
                out.append((desc, path, colour_space(p)))
    return sorted(out, key=lambda t: (t[2], t[0].lower()))


# ---- the document's profile -------------------------------------------------------

def doc_profile(doc):
    """The document's profile, or None for sRGB."""
    data = (getattr(doc, "meta", {}) or {}).get("icc_data")
    if not data:
        return None
    try:
        return open_profile(base64.b64decode(data))
    except Exception:                                       # noqa: BLE001
        return None


def set_doc_profile(doc, profile):
    """Tag the document (None = sRGB, untagged). Returns the old meta values."""
    old = (doc.meta.get("icc"), doc.meta.get("icc_data"))
    if profile is None:
        doc.meta.pop("icc", None)
        doc.meta.pop("icc_data", None)
    else:
        doc.meta["icc"] = describe(profile)
        doc.meta["icc_data"] = base64.b64encode(profile_bytes(profile)).decode("ascii")
    return old


def is_srgb(profile):
    return profile is None or describe(profile).lower().startswith("srgb")


# ---- transforms on RGBA arrays ------------------------------------------------------

def _rgb_image(rgba):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]), "RGB")


def convert(rgba, src=None, dst=None, intent="perceptual", bpc=True):
    """RGBA from one RGB profile to another (None = sRGB); alpha kept."""
    cms = _cms()
    src = src or open_profile(None)
    dst = dst or open_profile(None)
    flags = cms.Flags.BLACKPOINTCOMPENSATION if bpc else cms.Flags.NONE
    t = cms.buildTransform(src, dst, "RGB", "RGB", INTENTS.get(intent, 0), flags)
    out = np.array(rgba, copy=True)
    out[..., :3] = np.asarray(cms.applyTransform(_rgb_image(rgba), t))
    return out


def to_lab(rgba, src=None):
    """Lab (L 0..100, a, b) floats for RGBA in a profile (None = sRGB)."""
    cms = _cms()
    t = cms.buildTransform(src or open_profile(None), cms.createProfile("LAB"), "RGB", "LAB")
    lab = np.asarray(cms.applyTransform(_rgb_image(rgba), t)).astype(np.float64)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] = np.where(lab[..., 1:] > 127, lab[..., 1:] - 256, lab[..., 1:])
    return lab


class ViewTransform:
    """Document pixels -> what the monitor should show: through the
    document's profile, an optional proof profile, and the monitor's."""

    def __init__(self, doc_prof=None, monitor=None, proof=None, intent="perceptual",
                 bpc=True, gamut=False):
        cms = _cms()
        self.doc_prof = doc_prof or open_profile(None)
        self.monitor = monitor or open_profile(None)
        self.proof = proof
        self.gamut = bool(gamut and proof is not None)
        self.intent = INTENTS.get(intent, 0)
        flags = cms.Flags.BLACKPOINTCOMPENSATION if bpc else cms.Flags.NONE
        if proof is not None:
            mode = colour_space(proof)
            pmode = {"CMYK": "CMYK", "GRAY": "L"}.get(mode, "RGB")
            self._to_proof = cms.buildTransform(self.doc_prof, proof, "RGB", pmode, self.intent, flags)
            self._from_proof = cms.buildTransform(proof, self.monitor, pmode, "RGB",
                                                  INTENTS["relative"], flags)
            self._direct = None
        else:
            self._direct = cms.buildTransform(self.doc_prof, self.monitor, "RGB", "RGB",
                                              self.intent, flags)

    @property
    def identity(self):
        return self.proof is None and is_srgb(self.doc_prof) and is_srgb(self.monitor)

    def apply(self, rgba):
        cms = _cms()
        img = _rgb_image(rgba)
        out = np.array(rgba, copy=True)
        if self._direct is not None:
            out[..., :3] = np.asarray(cms.applyTransform(img, self._direct))
            return out
        proofed = cms.applyTransform(cms.applyTransform(img, self._to_proof), self._from_proof)
        out[..., :3] = np.asarray(proofed)
        if self.gamut:
            mask = out_of_gamut(rgba, self.proof, self.doc_prof, self._to_proof)
            out[..., :3][mask] = GAMUT_COLOUR
        return out


def out_of_gamut(rgba, proof, src=None, to_proof=None, threshold=GAMUT_DELTA_E):
    """Boolean mask: colours the proof profile cannot reproduce -- those
    whose round trip through it (relative colorimetric, no compensation)
    lands further than `threshold` delta-E away."""
    cms = _cms()
    src = src or open_profile(None)
    mode = {"CMYK": "CMYK", "GRAY": "L"}.get(colour_space(proof), "RGB")
    there = cms.buildTransform(src, proof, "RGB", mode, INTENTS["relative"])
    back = cms.buildTransform(proof, src, mode, "RGB", INTENTS["relative"])
    trip = np.array(rgba, copy=True)
    trip[..., :3] = np.asarray(cms.applyTransform(cms.applyTransform(_rgb_image(rgba), there), back))
    d = np.sqrt(((to_lab(rgba, src) - to_lab(trip, src)) ** 2).sum(-1))
    return d > threshold


def to_cmyk_image(rgba, cmyk_profile, src=None, intent="perceptual", matte=(255, 255, 255)):
    """A PIL CMYK image (alpha matted on white) through a CMYK profile, for
    printing formats (JPEG, TIFF)."""
    cms = _cms()
    from PIL import Image
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = rgba[..., :3] * a + np.asarray(matte, np.float64) * (1 - a)
    img = Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB")
    t = cms.buildTransform(src or open_profile(None), cmyk_profile, "RGB", "CMYK",
                           INTENTS.get(intent, 0), cms.Flags.BLACKPOINTCOMPENSATION)
    return cms.applyTransform(img, t)


def to_gray(rgba, src=None):
    """RGBA whose colour is the profile's grey for each pixel (L* kept)."""
    lab = to_lab(rgba, src)
    cms = _cms()
    from PIL import Image
    l8 = np.clip(np.rint(lab[..., 0] * 255.0 / 100.0), 0, 255).astype(np.uint8)
    # Pillow's LAB stores a* and b* as signed bytes: neutral is 0, not 128.
    labimg = np.stack([l8, np.zeros_like(l8), np.zeros_like(l8)], -1)
    t = cms.buildTransform(cms.createProfile("LAB"), src or open_profile(None), "LAB", "RGB")
    rgb = np.asarray(cms.applyTransform(Image.fromarray(labimg, "LAB"), t))
    out = np.array(rgba, copy=True)
    g = rgb.mean(axis=-1).round().astype(np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = g
    return out
