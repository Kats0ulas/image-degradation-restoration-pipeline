"""
STAGE 4 — Denoising in the SPATIAL domain.

Three filters: Gaussian (linear, convolution), Median (non-linear, order
statistic filter) and Bilateral (non-linear, edge-preserving).

All accept and return uint8 [0,255].
"""
import cv2
import numpy as np

from . import config


def gaussian(img, ksize=5, sigma=0):
    """Convolution with a Gaussian kernel. sigma=0 => computed from ksize."""
    return cv2.GaussianBlur(np.asarray(img, dtype=np.uint8), (ksize, ksize), sigma)


def median(img, ksize=3):
    """Median filter over a ksize×ksize window (ksize must be odd)."""
    return cv2.medianBlur(np.asarray(img, dtype=np.uint8), ksize)


def bilateral(img, d=9, sigma_color=50, sigma_space=50):
    """Bilateral filter: edge-preserving smoothing."""
    return cv2.bilateralFilter(np.asarray(img, dtype=np.uint8),
                               d, float(sigma_color), float(sigma_space))


def filter_specs():
    """
    All filter × parameter combinations. Each spec: key (file names),
    family, short (plot label), size (smoothing strength index).

    CAUTION — "size" does NOT share units across families:
      Gaussian / Median : kernel size k in pixels          (3 … 9)
      Bilateral         : sigmaColor, in gray levels        (25 … 100)
    It is only used for sorting within the same family and for the x axis of
    the Stage 4 PSNR–k plot, which contains only Gaussian and Median.
    If Bilateral is ever added to the same plot, the two scales would be
    compared as if they were the same — that needs a separate field, not this one.
    """
    specs = []
    for k, s in config.GAUSSIAN_PARAMS:
        specs.append({"key": f"gauss_k{k}", "family": "gaussian",
                      "short": f"Gaussian {k}×{k}", "size": k,
                      "fn": lambda im, k=k, s=s: gaussian(im, k, s)})
    for k in config.MEDIAN_PARAMS:
        specs.append({"key": f"median_k{k}", "family": "median",
                      "short": f"Median {k}×{k}", "size": k,
                      "fn": lambda im, k=k: median(im, k)})
    for d, sc, ss in config.BILATERAL_PARAMS:
        specs.append({"key": f"bilat_c{sc}_s{ss}", "family": "bilateral",
                      "short": f"Bilateral σc={sc}, σs={ss}", "size": sc,
                      "fn": lambda im, d=d, sc=sc, ss=ss: bilateral(im, d, sc, ss)})
    return specs


FAMILY_LABEL = {
    "gaussian": "Gaussian (linear)",
    "median": "Median (order statistic)",
    "bilateral": "Bilateral (edge-preserving)",
}


def apply_all(img) -> list:
    """[(spec, filtered_image), ...] for every filter/parameter combination."""
    return [(s, s["fn"](img)) for s in filter_specs()]


def sharpness(img) -> float:
    """Variance of the Laplacian; compared with the value of the reference image."""
    lap = cv2.Laplacian(np.asarray(img, dtype=np.uint8), cv2.CV_64F, ksize=3)
    return float(lap.var())
