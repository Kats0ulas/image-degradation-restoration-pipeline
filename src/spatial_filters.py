"""
ΣΤΑΔΙΟ 4 — Αποθορυβοποίηση στο ΧΩΡΙΚΟ πεδίο.

Τρία φίλτρα: Gaussian (γραμμικό, συνέλιξη), Median (μη γραμμικό, φίλτρο τάξης)
και Bilateral (μη γραμμικό, με διατήρηση ακμών).

Όλα δέχονται και επιστρέφουν uint8 [0,255].
"""
import cv2
import numpy as np

from . import config


def gaussian(img, ksize=5, sigma=0):
    """Συνέλιξη με γκαουσιανό πυρήνα. sigma=0 => υπολογίζεται από το ksize."""
    return cv2.GaussianBlur(np.asarray(img, dtype=np.uint8), (ksize, ksize), sigma)


def median(img, ksize=3):
    """Φίλτρο διαμέσου σε παράθυρο ksize×ksize (το ksize πρέπει να είναι περιττό)."""
    return cv2.medianBlur(np.asarray(img, dtype=np.uint8), ksize)


def bilateral(img, d=9, sigma_color=50, sigma_space=50):
    """Διπλευρικό φίλτρο: εξομάλυνση με διατήρηση ακμών."""
    return cv2.bilateralFilter(np.asarray(img, dtype=np.uint8),
                               d, float(sigma_color), float(sigma_space))


def filter_specs():
    """
    Όλοι οι συνδυασμοί φίλτρου × παραμέτρων. Κάθε spec: key (ονόματα αρχείων),
    family, short (ετικέτα γραφημάτων), size (δείκτης έντασης εξομάλυνσης).
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
    "gaussian": "Gaussian (γραμμικό)",
    "median": "Median (τάξης)",
    "bilateral": "Bilateral (διατήρηση ακμών)",
}


def apply_all(img) -> list:
    """[(spec, filtered_image), ...] για κάθε συνδυασμό φίλτρου/παραμέτρων."""
    return [(s, s["fn"](img)) for s in filter_specs()]


def sharpness(img) -> float:
    """Διασπορά του Λαπλασιανού· συγκρίνεται με την τιμή της εικόνας αναφοράς."""
    lap = cv2.Laplacian(np.asarray(img, dtype=np.uint8), cv2.CV_64F, ksize=3)
    return float(lap.var())
