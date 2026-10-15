"""
Loading, conversion and saving of images.

Project-wide convention:
  - We work in GRAYSCALE, uint8, range [0, 255].
  - Operations are done in float64 and we always return uint8 with clipping.
"""
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

from . import config


def ensure_dir(path) -> Path:
    """Creates the directory (and its parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_uint8(img) -> np.ndarray:
    """Safe conversion to uint8 [0,255] with clipping (no wrap-around)."""
    return np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)


def to_gray(img) -> np.ndarray:
    """Conversion to 8-bit grayscale (if it is already gray, it is returned as is)."""
    arr = np.asarray(img)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        # skimage sometimes returns float [0,1]
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = to_uint8(arr)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def resize(img, size=config.TARGET_SIZE) -> np.ndarray:
    """Resize to (width, height). INTER_AREA for downscaling."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w > size[0] or h > size[1]) else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def load_reference(spec) -> np.ndarray:
    """
    Loads a reference image from a config.REFERENCE_IMAGES dict:
    "loader" = a skimage.data function, or "path" = our own file.
    """
    if spec.get("path"):
        raw = cv2.imread(str(spec["path"]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Image not found: {spec['path']}")
        # OpenCV reads BGR, while to_gray expects RGB.
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = to_gray(raw)
    else:
        img = to_gray(getattr(skdata, spec["loader"])())
    return resize(img)


def load_all_references() -> dict:
    """Returns {key: (spec, image)} for all reference images."""
    return {s["key"]: (s, load_reference(s)) for s in config.REFERENCE_IMAGES}


def save_image(img, path) -> Path:
    """Saves a grayscale image as PNG (lossless)."""
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), to_uint8(img))
    if not ok:
        raise IOError(f"Failed to write: {path}")
    return path
