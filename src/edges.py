"""
STAGE 7 — Edge detection.

Two approaches:
  • Sobel — 1st-derivative operator; gives a continuous gradient magnitude and
    needs a threshold to become a binary edge map.
  • Canny — Gaussian smoothing, Sobel gradient, non-maximum suppression and
    hysteresis with two thresholds.

Edge maps are evaluated with precision / recall / F1 and a position tolerance
of a few pixels.
"""
import cv2
import numpy as np

from . import config
from .io_utils import to_uint8


def sobel_components(img, ksize=None):
    """The partial derivatives Gx, Gy in float64."""
    k = config.SOBEL_KSIZE if ksize is None else ksize
    f = np.asarray(img, dtype=np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=k)
    return gx, gy


def sobel_magnitude(img, ksize=None) -> np.ndarray:
    """Gradient magnitude |∇f| = sqrt(Gx² + Gy²), normalized to uint8 for display."""
    gx, gy = sobel_components(img, ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = float(mag.max())
    return to_uint8(mag / m * 255.0) if m > 0 else np.zeros(mag.shape, np.uint8)


def sobel_edges(img, percentile=None, ksize=None) -> np.ndarray:
    """Binary edge map: |∇f| above the given percentile (0/255)."""
    p = config.SOBEL_EDGE_PERCENTILE if percentile is None else percentile
    gx, gy = sobel_components(img, ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.percentile(mag, p)
    return ((mag >= thr).astype(np.uint8)) * 255


def canny(img, low=100, high=200) -> np.ndarray:
    """Canny detector (OpenCV applies the 4 steps internally)."""
    return cv2.Canny(np.asarray(img, dtype=np.uint8), int(low), int(high))


def _within_tolerance(mask_ref, mask_test, tol):
    """Fraction of mask_test pixels within `tol` pixels of an edge in mask_ref."""
    ref = (np.asarray(mask_ref) > 0).astype(np.uint8)
    test = np.asarray(mask_test) > 0
    if test.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 3)
    return float((dist[test] <= tol).mean())


def edge_metrics(ref_edges, test_edges, tol=None) -> dict:
    """
    Comparison of two binary edge maps, with a position tolerance of `tol` pixels:
      precision : fraction of detected edges that are real
      recall    : fraction of real edges that were found
      f1        : harmonic mean of the two
      edge_pct  : percentage of pixels classified as edge
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
    Canny with adaptive thresholds: high = the `percentile` percentile of the
    image's own |∇f|, low = ratio·high. Needed because smoothing lowers the
    gradients, so fixed thresholds are unfair to the filtered image.
    """
    p = config.SOBEL_EDGE_PERCENTILE if percentile is None else percentile
    gx, gy = sobel_components(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hi = float(np.percentile(mag, p))
    return canny(img, max(1, int(hi * ratio)), max(2, int(hi)))
