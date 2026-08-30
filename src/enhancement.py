"""
ΣΤΑΔΙΟ 6 — Βελτίωση εικόνας (ενίσχυση αντίθεσης).

Τρεις τεχνικές:
  1. Γραμμική διάταση αντίθεσης — απεικόνιση ενός εύρους εντάσεων στο [0,255].
  2. Ισοστάθμιση ιστογράμματος (HE) — καθολική, με την CDF ως συνάρτηση
     μετασχηματισμού.
  3. CLAHE — τοπική ισοστάθμιση σε πλακίδια, με ψαλίδισμα (clip limit) του
     τοπικού ιστογράμματος.

Επειδή η βελτίωση αλλάζει σκόπιμα τις εντάσεις, δεν κρίνεται με PSNR ως προς
την αναφορά αλλά με αντίθεση, εντροπία και αξιοποίηση των επιπέδων γκρι.
"""
import cv2
import numpy as np

from . import config
from .io_utils import to_uint8


def contrast_stretch(img, plo=None, phi=None):
    """Γραμμική διάταση: το [p_lo, p_hi] εκατοστημόριο απεικονίζεται στο [0,255]."""
    plo = config.STRETCH_PERCENTILES[0] if plo is None else plo
    phi = config.STRETCH_PERCENTILES[1] if phi is None else phi
    f = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(f, [plo, phi])
    if hi <= lo:
        return to_uint8(f)
    return to_uint8((f - lo) / (hi - lo) * 255.0)


def histogram_equalization(img):
    """Καθολική ισοστάθμιση ιστογράμματος (η CDF ως συνάρτηση μετασχηματισμού)."""
    return cv2.equalizeHist(np.asarray(img, dtype=np.uint8))


def clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization."""
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    return op.apply(np.asarray(img, dtype=np.uint8))


def method_specs():
    """Οι μέθοδοι που δοκιμάζονται, με σειρά αυξανόμενης πολυπλοκότητας."""
    specs = [
        {"key": "stretch", "short": "Γραμμική διάταση",
         "fn": contrast_stretch},
        {"key": "he", "short": "Ισοστάθμιση (HE)",
         "fn": histogram_equalization},
    ]
    for cl, tg in config.CLAHE_PARAMS:
        specs.append({
            "key": f"clahe_c{cl:g}_t{tg[0]}",
            "short": f"CLAHE clip={cl:g}, {tg[0]}×{tg[1]}",
            "fn": lambda im, cl=cl, tg=tg: clahe(im, cl, tg),
        })
    return specs


def transfer_curve(img) -> np.ndarray:
    """Η συνάρτηση μετασχηματισμού s = T(r) της HE: κανονικοποιημένη CDF."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(h)
    return cdf / cdf[-1] * 255.0


def entropy(img) -> float:
    """Εντροπία Shannon του ιστογράμματος (bits/pixel), max = 8 για 8-bit."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def used_levels_pct(img) -> float:
    """Ποσοστό των 256 επιπέδων γκρι που εμφανίζονται πράγματι στην εικόνα."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256)
    return float((h > 0).mean() * 100.0)


def profile(img) -> dict:
    """Τα μέτρα βελτίωσης μαζί."""
    f = np.asarray(img, dtype=np.float64)
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "entropy": entropy(img),
        "used_pct": used_levels_pct(img),
    }
