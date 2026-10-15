"""
Quality metrics between a reference image and a processed one.

  PSNR : peak signal-to-noise ratio (dB), based on the MSE.
  SSIM : structural similarity index [0,1] (luminance, contrast, structure).

Both are computed on uint8 [0,255] (data_range = 255).
"""
import numpy as np
from skimage.metrics import structural_similarity

from .io_utils import to_uint8


def mse(ref, test) -> float:
    """Mean squared error in gray levels."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(ref, test, data_range: float = 255.0) -> float:
    """PSNR in dB. Returns inf when the images are identical."""
    e = mse(ref, test)
    if e == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / e))


def ssim(ref, test) -> float:
    """SSIM with the skimage defaults (uniform 7×7 window)."""
    return float(structural_similarity(to_uint8(ref), to_uint8(test), data_range=255))


def evaluate(ref, test) -> dict:
    """All metrics together, as a dict."""
    return {"mse": mse(ref, test), "psnr": psnr(ref, test), "ssim": ssim(ref, test)}
