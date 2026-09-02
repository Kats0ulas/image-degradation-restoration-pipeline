"""
Μετρικές ποιότητας ανάμεσα σε μια εικόνα αναφοράς και μια επεξεργασμένη.

  PSNR : λόγος σήματος προς θόρυβο κορυφής (dB), βασισμένος στο MSE.
  SSIM : δείκτης δομικής ομοιότητας [0,1] (φωτεινότητα, αντίθεση, δομή).

Και οι δύο υπολογίζονται σε uint8 [0,255] (data_range = 255).
"""
import numpy as np
from skimage.metrics import structural_similarity

from .io_utils import to_uint8


def mse(ref, test) -> float:
    """Μέσο τετραγωνικό σφάλμα σε επίπεδα γκρι."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(ref, test, data_range: float = 255.0) -> float:
    """PSNR σε dB. Επιστρέφει inf όταν οι εικόνες είναι πανομοιότυπες."""
    e = mse(ref, test)
    if e == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / e))


def ssim(ref, test) -> float:
    """SSIM με τις προεπιλογές του skimage (ομοιόμορφο παράθυρο 7×7)."""
    return float(structural_similarity(to_uint8(ref), to_uint8(test), data_range=255))


def evaluate(ref, test) -> dict:
    """Όλες οι μετρικές μαζί, ως dict."""
    return {"mse": mse(ref, test), "psnr": psnr(ref, test), "ssim": ssim(ref, test)}
