"""
STAGE 6 — Image enhancement (contrast enhancement).

Three techniques:
  1. Linear contrast stretching — maps a range of intensities onto [0,255].
  2. Histogram equalization (HE) — global, with the CDF as the transformation
     function.
  3. CLAHE — local equalization on tiles, with clipping (clip limit) of the
     local histogram.

Because enhancement deliberately changes the intensities, it is not judged by
PSNR against the reference but by contrast, entropy and gray-level usage.
"""
import cv2
import numpy as np

from . import config
from .io_utils import to_uint8


def contrast_stretch(img, plo=None, phi=None):
    """Linear stretch: the [p_lo, p_hi] percentile range is mapped onto [0,255]."""
    plo = config.STRETCH_PERCENTILES[0] if plo is None else plo
    phi = config.STRETCH_PERCENTILES[1] if phi is None else phi
    f = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(f, [plo, phi])
    if hi <= lo:
        return to_uint8(f)
    return to_uint8((f - lo) / (hi - lo) * 255.0)


def histogram_equalization(img):
    """Global histogram equalization (the CDF as the transformation function)."""
    return cv2.equalizeHist(np.asarray(img, dtype=np.uint8))


def clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization."""
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    return op.apply(np.asarray(img, dtype=np.uint8))


def method_specs():
    """The methods under test, in order of increasing complexity."""
    specs = [
        {"key": "stretch", "short": "Linear stretch",
         "fn": contrast_stretch},
        {"key": "he", "short": "Equalization (HE)",
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
    """The HE transformation function s = T(r): the normalized CDF."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(h)
    return cdf / cdf[-1] * 255.0


def entropy(img) -> float:
    """Shannon entropy of the histogram (bits/pixel), max = 8 for 8-bit."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def used_levels_pct(img) -> float:
    """Percentage of the 256 gray levels that actually occur in the image."""
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256)
    return float((h > 0).mean() * 100.0)


def profile(img) -> dict:
    """All enhancement measures together."""
    f = np.asarray(img, dtype=np.float64)
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "entropy": entropy(img),
        "used_pct": used_levels_pct(img),
    }
