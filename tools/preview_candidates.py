"""
Βοηθητικό εργαλείο (εκτός αλυσίδας): προεπισκόπηση υποψήφιων εικόνων αναφοράς.

Τρέξε:  python tools/preview_candidates.py
Παράγει: results/stage1_reference/candidates.png

Αν θέλεις να αλλάξεις τις εικόνες αναφοράς, άλλαξε το config.REFERENCE_IMAGES.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src import config, io_utils, viz

CANDIDATES = [
    ("camera",  "camera — έντονες ακμές, υφή"),
    ("moon",    "moon — ομαλοί τόνοι, χαμηλή αντίθεση"),
    ("astronaut", "astronaut — μικτή (πρόσωπο + λεπτομέρεια)"),
    ("chelsea", "chelsea — πλούσια υφή (τρίχωμα)"),
    ("coins",   "coins — ομοιόμορφο φόντο, κυκλικές ακμές"),
    ("clock",   "clock — ομαλό φόντο, λίγες κοφτές ακμές"),
    ("brick",   "brick — περιοδική υφή"),
    ("cell",    "cell — χαμηλή αντίθεση, βιοϊατρική"),
    ("coffee",  "coffee — ομαλές μεταβάσεις + λεπτομέρεια"),
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
                  suptitle="Υποψήφιες εικόνες αναφοράς (skimage.data)")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
