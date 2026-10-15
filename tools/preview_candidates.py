"""
Helper tool (outside the pipeline): preview of candidate reference images.

Run:      python tools/preview_candidates.py
Produces: results/stage1_reference/candidates.png

To change the reference images, edit config.REFERENCE_IMAGES.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src import config, io_utils, viz

CANDIDATES = [
    ("camera",  "camera — strong edges, texture"),
    ("moon",    "moon — smooth tones, low contrast"),
    ("astronaut", "astronaut — mixed (face + detail)"),
    ("chelsea", "chelsea — rich texture (fur)"),
    ("coins",   "coins — uniform background, circular edges"),
    ("clock",   "clock — smooth background, few sharp edges"),
    ("brick",   "brick — periodic texture"),
    ("cell",    "cell — low contrast, biomedical"),
    ("coffee",  "coffee — smooth transitions + detail"),
]


def main():
    items = []
    for loader, label in CANDIDATES:
        img = io_utils.load_reference({"loader": loader})
        f = img.astype(np.float64)
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
        grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
        items.append((img, f"{label}\nstd={f.std():.0f}  |∇|={grad:.0f}"))

    out = config.STAGE_DIRS[1] / "candidates.png"
    viz.plot_grid(items, out, ncols=3, figsize_per=(3.4, 3.9),
                  suptitle="Candidate reference images (skimage.data)")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
