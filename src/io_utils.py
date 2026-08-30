"""
Φόρτωση, μετατροπή και αποθήκευση εικόνων.

Σύμβαση όλου του project:
  - Δουλεύουμε σε GRAYSCALE, uint8, εύρος [0, 255].
  - Οι πράξεις γίνονται σε float64 και επιστρέφουμε πάντα σε uint8 με clipping.
"""
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

from . import config


def ensure_dir(path) -> Path:
    """Δημιουργεί τον φάκελο (και τους γονείς του) αν δεν υπάρχει."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_uint8(img) -> np.ndarray:
    """Ασφαλής μετατροπή σε uint8 [0,255] με clipping (όχι wrap-around)."""
    return np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)


def to_gray(img) -> np.ndarray:
    """Μετατροπή σε 8-bit grayscale (αν είναι ήδη gray, επιστρέφεται ως έχει)."""
    arr = np.asarray(img)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        # skimage επιστρέφει μερικές φορές float [0,1]
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = to_uint8(arr)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def resize(img, size=config.TARGET_SIZE) -> np.ndarray:
    """Αλλαγή μεγέθους σε (width, height). INTER_AREA για σμίκρυνση."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w > size[0] or h > size[1]) else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def load_reference(spec) -> np.ndarray:
    """
    Φορτώνει εικόνα αναφοράς από το dict του config.REFERENCE_IMAGES:
    "loader" = συνάρτηση του skimage.data, ή "path" = δικό μας αρχείο.
    """
    if spec.get("path"):
        raw = cv2.imread(str(spec["path"]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Δεν βρέθηκε η εικόνα: {spec['path']}")
        # Το OpenCV διαβάζει BGR, ενώ η to_gray περιμένει RGB.
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = to_gray(raw)
    else:
        img = to_gray(getattr(skdata, spec["loader"])())
    return resize(img)


def load_all_references() -> dict:
    """Επιστρέφει {key: (spec, image)} για όλες τις εικόνες αναφοράς."""
    return {s["key"]: (s, load_reference(s)) for s in config.REFERENCE_IMAGES}


def save_image(img, path) -> Path:
    """Αποθήκευση grayscale εικόνας σε PNG (lossless)."""
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), to_uint8(img))
    if not ok:
        raise IOError(f"Αποτυχία εγγραφής: {path}")
    return path
