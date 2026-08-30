"""
ΣΤΑΔΙΟ 7 — Ανίχνευση ακμών.

Δύο προσεγγίσεις:
  • Sobel — τελεστής 1ης παραγώγου· δίνει συνεχές μέτρο κλίσης και χρειάζεται
    κατώφλι για να γίνει δυαδικός χάρτης ακμών.
  • Canny — γκαουσιανή εξομάλυνση, κλίση Sobel, καταστολή μη μεγίστων και
    υστέρηση με δύο κατώφλια.

Η αξιολόγηση των χαρτών ακμών γίνεται με precision / recall / F1 και ανοχή
θέσης λίγων pixel.
"""
import cv2
import numpy as np

from . import config
from .io_utils import to_uint8


def sobel_components(img, ksize=None):
    """Οι μερικές παράγωγοι Gx, Gy σε float64."""
    k = config.SOBEL_KSIZE if ksize is None else ksize
    f = np.asarray(img, dtype=np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=k)
    return gx, gy


def sobel_magnitude(img, ksize=None) -> np.ndarray:
    """Μέτρο κλίσης |∇f| = sqrt(Gx² + Gy²), κανονικοποιημένο σε uint8 για προβολή."""
    gx, gy = sobel_components(img, ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = float(mag.max())
    return to_uint8(mag / m * 255.0) if m > 0 else np.zeros(mag.shape, np.uint8)


def sobel_edges(img, percentile=None, ksize=None) -> np.ndarray:
    """Δυαδικός χάρτης ακμών: |∇f| πάνω από το δοσμένο εκατοστημόριο (0/255)."""
    p = config.SOBEL_EDGE_PERCENTILE if percentile is None else percentile
    gx, gy = sobel_components(img, ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.percentile(mag, p)
    return ((mag >= thr).astype(np.uint8)) * 255


def canny(img, low=100, high=200) -> np.ndarray:
    """Ανιχνευτής Canny (το OpenCV εφαρμόζει εσωτερικά τα 4 βήματα)."""
    return cv2.Canny(np.asarray(img, dtype=np.uint8), int(low), int(high))


def _within_tolerance(mask_ref, mask_test, tol):
    """Ποσοστό των pixel του mask_test εντός `tol` pixel από ακμή του mask_ref."""
    ref = (np.asarray(mask_ref) > 0).astype(np.uint8)
    test = np.asarray(mask_test) > 0
    if test.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 3)
    return float((dist[test] <= tol).mean())


def edge_metrics(ref_edges, test_edges, tol=None) -> dict:
    """
    Σύγκριση δύο δυαδικών χαρτών ακμών, με ανοχή θέσης `tol` pixel:
      precision : ποσοστό των ανιχνευμένων ακμών που είναι πραγματικές
      recall    : ποσοστό των πραγματικών ακμών που βρέθηκε
      f1        : αρμονικός μέσος των δύο
      edge_pct  : ποσοστό pixel που χαρακτηρίστηκαν ακμή
    """
    tol = config.EDGE_TOLERANCE if tol is None else tol
    precision = _within_tolerance(ref_edges, test_edges, tol)
    recall = _within_tolerance(test_edges, ref_edges, tol)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edge_pct": float((np.asarray(test_edges) > 0).mean() * 100.0),
    }


def canny_auto(img, percentile=None, ratio=0.4) -> np.ndarray:
    """
    Canny με προσαρμοστικά κατώφλια: high = το `percentile` εκατοστημόριο του
    |∇f| της ίδιας της εικόνας, low = ratio·high. Χρειάζεται επειδή η
    εξομάλυνση μειώνει τις κλίσεις και τα σταθερά κατώφλια αδικούν τη
    φιλτραρισμένη εικόνα.
    """
    p = config.SOBEL_EDGE_PERCENTILE if percentile is None else percentile
    gx, gy = sobel_components(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hi = float(np.percentile(mag, p))
    return canny(img, max(1, int(hi * ratio)), max(2, int(hi)))
